"""Build immutable, versioned sequence-geometry pairing manifests."""

from __future__ import annotations

import csv
import json
import numbers
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.data.sequence_geometry import (
    MODIFIED_RESIDUE_MAPPING_VERSION,
    PAIRING_SCHEMA_VERSION,
    VOCABULARY_VERSION,
    SequenceGeometryVocabulary,
)
from protein_distance_diffusion.evaluation.sequence_readiness import (
    canonical_sequence,
    sequence_sha256,
    sha256_file,
)
from protein_distance_diffusion.evaluation.sequence_readiness_storage import SequenceReadinessArtifactReader

PRACTICAL_PAIRING_CLASSES = frozenset(
    {
        "verified_sequence_geometry_pair",
        "unique_sequence_pair_coordinate_unavailable",
        "residue_identity_provenance_incomplete",
    }
)
ELIGIBILITY_POLICIES = frozenset({"strict", "practical", "all_with_status"})
ELIGIBILITY_POLICY_VERSION = "sequence_geometry_pairing_eligibility_v1"
AUTHORITATIVE_STATUS_FIELDS = (
    "v3_pairing_classification",
    "forensic_root_cause",
    "strict_provenance_status",
    "practical_training_eligibility",
    "source_identity_status",
)
STATUS_FIELD_ALIASES = {
    "training_eligibility": "practical_training_eligibility",
    "refined_primary_classification": "forensic_root_cause",
}
AUTHORITATIVE_SOURCE_FIELDS = {
    "practical_eligibility": (
        *AUTHORITATIVE_STATUS_FIELDS,
        "refined_primary_classification",
        "matrix_path",
    ),
    "strict_eligibility": (
        *AUTHORITATIVE_STATUS_FIELDS,
        "refined_primary_classification",
        "matrix_path",
    ),
    "unresolved_eligibility": (
        *AUTHORITATIVE_STATUS_FIELDS,
        "refined_primary_classification",
        "matrix_path",
    ),
    "forensic_transition": (*AUTHORITATIVE_STATUS_FIELDS, "refined_primary_classification"),
}
CANDIDATE_BOOLEAN_FIELDS = (
    "author_chain_match",
    "label_chain_match",
    "sequence_match",
    "strong_evidential_match",
    "identity_mapping_available",
    "identity_mapping_ambiguous",
    "insertion_codes_match",
    "trim_metadata_available",
    "trim_metadata_compatible",
)
FORENSIC_PAIRING_COMPATIBILITY = {
    "coordinate_disagreement": frozenset({"unresolved_ambiguity"}),
    "demonstrated_sequence_matrix_mismatch": frozenset({"unresolved_ambiguity"}),
    "historical_missing_calpha_selection_unreproducible": frozenset(
        {"unique_sequence_pair_coordinate_unavailable", "unresolved_ambiguity"}
    ),
    "legacy_trim_metadata_inconsistency": frozenset({"verified_sequence_geometry_pair"}),
    "matrix_sequence_absent_from_author_linked_polymer": frozenset({"unresolved_ambiguity"}),
    "residue_id_namespace_mismatch": frozenset(
        {
            "unique_sequence_pair_coordinate_unavailable",
            "unresolved_ambiguity",
            "verified_sequence_geometry_pair",
        }
    ),
    "unresolved_ambiguity": frozenset({"residue_identity_provenance_incomplete", "unresolved_ambiguity"}),
    "verified_sequence_geometry_pair": frozenset({"verified_sequence_geometry_pair"}),
    "residue_identity_provenance_incomplete": frozenset({"residue_identity_provenance_incomplete"}),
    "unique_sequence_pair_coordinate_unavailable": frozenset({"unique_sequence_pair_coordinate_unavailable"}),
    "multiple_author_linked_polymer_candidates": frozenset({"unresolved_ambiguity"}),
}
EXCLUSION_REASON_ORDER = (
    "original_split_excluded",
    "not_selected_by_eligibility_policy",
    "unresolved_ambiguity",
    "noncanonical_sequence",
    "sequence_matrix_length_mismatch",
    "manifest_npz_sequence_mismatch",
    "matrix_path_missing",
    "coordinate_matrix_mismatch",
)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if not isinstance(value, (list, dict, tuple)):
        missing = pd.isna(value)
        if bool(missing):
            return None
    return value


def normalize_evidence_boolean(value: Any, *, field: str, allow_missing: bool) -> bool | None:
    """Parse audit Boolean evidence without applying Python truthiness to strings."""
    missing = value is None or (not isinstance(value, str) and bool(pd.isna(value))) or value == ""
    if missing:
        if allow_missing:
            return None
        raise ValueError(f"Candidate Boolean evidence {field} is required")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        if int(value) in {0, 1}:
            return bool(value)
        raise ValueError(f"Invalid Boolean evidence for {field}: {value!r}")
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"Invalid Boolean evidence for {field}: {value!r}")


def _optional_bool(value: Any, field: str) -> bool:
    return normalize_evidence_boolean(value, field=field, allow_missing=True) is True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pairing_classification(row: dict[str, Any]) -> str:
    eligibility = row.get("training_eligibility", row.get("practical_training_eligibility"))
    if eligibility == "excluded_or_unresolved":
        return "unresolved_ambiguity"
    namespace_reconciled = any(
        _optional_bool(row.get(key), key)
        for key in (
            "auth_residue_ids_match",
            "label_residue_ids_match",
            "zero_based_positions_match",
            "one_based_positions_match",
        )
    )
    if row.get("strict_provenance_status") == "residue_identity_provenance_incomplete" and not namespace_reconciled:
        return "residue_identity_provenance_incomplete"
    if row.get("coordinate_matrix_status") == "unavailable":
        return "unique_sequence_pair_coordinate_unavailable"
    return "verified_sequence_geometry_pair"


def _forensic_root_cause(row: dict[str, Any]) -> str | None:
    return _clean(row.get("forensic_root_cause", row.get("refined_primary_classification")))


def _validate_forensic_pairing_relationship(
    forensic_root_cause: str | None,
    pairing_classification: str,
    *,
    sample_id: str,
) -> None:
    if forensic_root_cause is None:
        return
    allowed = FORENSIC_PAIRING_COMPATIBILITY.get(str(forensic_root_cause))
    if allowed is not None and pairing_classification not in allowed:
        raise ValueError(
            f"Incompatible forensic root cause and pairing classification for {sample_id}: "
            f"{forensic_root_cause!r} versus {pairing_classification!r}"
        )


def _hash_for_path(input_hashes: dict[str, str], path: Path) -> str | None:
    resolved = path.resolve()
    for recorded_path, digest in input_hashes.items():
        if Path(recorded_path).resolve() == resolved:
            return str(digest)
    return None


def _require_unique(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame:
        raise ValueError(f"{label} has no {column} column")
    duplicates = frame.loc[frame[column].astype(str).duplicated(keep=False), column]
    if not duplicates.empty:
        raise ValueError(f"{label} contains duplicate {column}: {duplicates.iloc[0]}")


def _load_inputs(
    processed_manifest: Path,
    train_manifest: Path,
    validation_manifest: Path,
    audit_dir: Path,
    normalization_file: Path,
    *,
    allow_pilot_evidence: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, Any]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, str],
]:
    protocol_path = audit_dir / "sequence_readiness_protocol.json"
    reader = SequenceReadinessArtifactReader(audit_dir)
    required = (
        processed_manifest,
        train_manifest,
        validation_manifest,
        normalization_file,
        protocol_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pairing input(s): {', '.join(missing)}")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "completed":
        raise ValueError("Sequence-readiness audit is not completed")
    audit_mode = protocol.get("audit_mode")
    if audit_mode == "raw-pilot" and not allow_pilot_evidence:
        raise ValueError("Pilot evidence requires --allow-pilot-evidence")
    if audit_mode not in {"raw-pilot", "raw-full"}:
        raise ValueError(f"Pairing build requires a raw-pilot or raw-full audit, got {audit_mode!r}")
    if audit_mode == "raw-full" and int(protocol.get("report_semantics_version", 0)) < 4:
        raise ValueError("Definitive pairing build requires raw-full report semantics version 4 or later")
    recorded_hashes = protocol.get("input_hashes", {})
    for path in (processed_manifest, train_manifest, validation_manifest):
        expected = _hash_for_path(recorded_hashes, path)
        if expected is None:
            raise ValueError(f"Audit protocol does not identify manifest input: {path}")
        if sha256_file(path) != expected:
            raise ValueError(f"Audit manifest hash mismatch: {path}")
    scientific_paths = reader.scientific_paths()
    if not scientific_paths:
        raise FileNotFoundError(f"Sequence-readiness audit has no readable scientific tables: {audit_dir}")
    summary_tables = (
        "practical_eligibility",
        "strict_eligibility",
        "unresolved_eligibility",
        "forensic_transition",
    )
    summary_paths = [path for table in summary_tables for path in reader.table_paths(table)]
    hashed_inputs = {str(path): sha256_file(path) for path in [*required, *scientific_paths, *summary_paths]}
    processed = pd.read_parquet(processed_manifest)
    train = pd.read_parquet(train_manifest)
    validation = pd.read_parquet(validation_manifest)
    alignments = list(reader.iter_records("matrix_pair_alignments"))
    candidates = reader.frame("residue_id_convention_evidence")
    practical = reader.frame("practical_eligibility")
    strict = reader.frame("strict_eligibility")
    unresolved = reader.frame("unresolved_eligibility")
    transition = reader.frame("forensic_transition")
    return (
        processed,
        train,
        validation,
        alignments,
        candidates,
        practical,
        strict,
        unresolved,
        transition,
        protocol,
        hashed_inputs,
    )


def _candidate_evidence(candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    if candidates.empty:
        return evidence
    physical_key = ["source_file", "model_number", "entity_id", "label_asym_id", "auth_asym_id"]
    missing = [column for column in ["sample_id", *physical_key, *CANDIDATE_BOOLEAN_FIELDS] if column not in candidates]
    if missing:
        raise ValueError(f"Candidate evidence is missing column(s): {', '.join(missing)}")
    for sample_id, group in candidates.groupby("sample_id", sort=False):
        normalized = group.copy()
        for field in CANDIDATE_BOOLEAN_FIELDS:
            normalized[field] = normalized[field].map(
                lambda value, name=field: normalize_evidence_boolean(value, field=name, allow_missing=False)
            )
        author_statuses = normalized.groupby(physical_key, dropna=False)["author_chain_match"].nunique()
        if (author_statuses > 1).any():
            raise ValueError(f"Inconsistent author-chain evidence within physical candidate for {sample_id}")
        physical = normalized.drop_duplicates(physical_key)
        strong_rows = normalized[normalized["strong_evidential_match"]]
        author_physical = normalized[normalized["author_chain_match"]].drop_duplicates(physical_key)
        non_author_physical = normalized[~normalized["author_chain_match"]].drop_duplicates(physical_key)
        strong_physical_all = strong_rows.drop_duplicates(physical_key)
        strong_author_rows = strong_rows[strong_rows["author_chain_match"]]
        strong_author_physical = strong_author_rows.drop_duplicates(physical_key)
        strong_non_author_physical = strong_rows[~strong_rows["author_chain_match"]].drop_duplicates(physical_key)
        selected = strong_author_physical if len(strong_author_physical) == 1 else author_physical
        convention_field = "residue_id_convention" if "residue_id_convention" in strong_rows else "convention"
        supported_conventions = sorted(set(strong_author_rows[convention_field].dropna().astype(str)))
        rejected_decoys = [
            {
                "source_file": _clean(row.get("source_file")),
                "model_number": _clean(row.get("model_number")),
                "entity_id": _clean(row.get("entity_id")),
                "label_asym_id": _clean(row.get("label_asym_id")),
                "auth_asym_id": _clean(row.get("auth_asym_id")),
                "rejection_reason": "author_chain_mismatch",
                "has_strong_evidence": bool(
                    ((strong_non_author_physical[physical_key] == row[physical_key].values).all(axis=1)).any()
                ),
            }
            for _, row in non_author_physical.iterrows()
        ]
        identity: dict[str, Any] = {}
        if len(selected) == 1:
            row = selected.iloc[0]
            identity = {
                "label_asym_id": _clean(row.get("label_asym_id")),
                "auth_asym_id": _clean(row.get("auth_asym_id")),
                "entity_id": _clean(row.get("entity_id")),
                "candidate_model_number": _clean(row.get("model_number")),
                "candidate_source_file": _clean(row.get("source_file")),
            }
        evidence[str(sample_id)] = {
            **identity,
            "raw_physical_candidate_count": len(physical),
            "candidate_evidence_row_count": len(group),
            "author_linked_physical_candidate_count": len(author_physical),
            "non_author_linked_candidate_count": len(non_author_physical),
            "strong_physical_candidate_count_all": len(strong_physical_all),
            "strong_author_linked_physical_candidate_count": len(strong_author_physical),
            "strong_non_author_linked_candidate_count": len(strong_non_author_physical),
            "strong_evidence_row_count": len(strong_rows),
            "supported_residue_id_conventions": supported_conventions,
            "rejected_candidate_evidence": rejected_decoys,
            "selected_candidate_unique": len(selected) == 1,
        }
    return evidence


def _authoritative_eligibility(
    alignments: list[dict[str, Any]],
    practical: pd.DataFrame,
    strict: pd.DataFrame,
    unresolved: pd.DataFrame,
    transition: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    for label, frame in (
        ("practical eligibility", practical),
        ("strict eligibility", strict),
        ("unresolved summary", unresolved),
        ("v2/v3 transition table", transition),
    ):
        _require_unique(frame, "sample_id", label)
    practical_ids = set(practical["sample_id"].astype(str))
    strict_ids = set(strict["sample_id"].astype(str))
    unresolved_ids = set(unresolved["sample_id"].astype(str))
    overlap = practical_ids & unresolved_ids
    if overlap:
        raise ValueError(f"Sample appears in both eligible and unresolved tables: {sorted(overlap)[0]}")
    if not strict_ids <= practical_ids:
        missing_strict = sorted(strict_ids - practical_ids)[0]
        raise ValueError(f"Strict-eligible sample is absent from practical eligibility: {missing_strict}")
    alignment_ids = [str(row["sample_id"]) for row in alignments]
    if len(alignment_ids) != len(set(alignment_ids)):
        raise ValueError("Audit alignment table contains duplicate sample IDs")
    membership_ids = practical_ids | unresolved_ids
    if membership_ids != set(alignment_ids) or len(practical) + len(unresolved) != len(alignment_ids):
        raise ValueError("Authoritative eligibility tables must contain exactly one row per audited sample")

    result = {
        sample_id: {
            "sample_id": sample_id,
            "practical_eligible_member": sample_id in practical_ids,
            "strict_eligible_member": sample_id in strict_ids,
            "unresolved_member": sample_id in unresolved_ids,
            "_field_sources": {},
        }
        for sample_id in alignment_ids
    }

    def merge_row(sample_id: str, row: dict[str, Any], source: str, fields: tuple[str, ...]) -> None:
        if sample_id not in result:
            raise ValueError(f"{source} sample is absent from audited cohort: {sample_id}")
        merged = result[sample_id]
        sources = merged["_field_sources"]
        for input_field in fields:
            output_field = STATUS_FIELD_ALIASES.get(input_field, input_field)
            if input_field not in row:
                continue
            value = _clean(row[input_field])
            if value is None:
                continue
            normalized = str(value)
            existing = _clean(merged.get(output_field))
            if existing is not None and str(existing) != normalized:
                raise ValueError(
                    f"Authoritative {output_field} contradiction for {sample_id}: "
                    f"{sources[output_field]}={existing!r}, {source}={normalized!r}"
                )
            merged[output_field] = normalized
            sources[output_field] = source

    alignment_index = {str(row["sample_id"]): row for row in alignments}
    for sample_id, row in alignment_index.items():
        computed_classification = _pairing_classification(row)
        recorded_classification = _clean(row.get("v3_pairing_classification"))
        if recorded_classification is not None and str(recorded_classification) != computed_classification:
            raise ValueError(
                f"Authoritative v3_pairing_classification contradiction for {sample_id}: "
                f"alignment evidence={computed_classification!r}, recorded={recorded_classification!r}"
            )
        forensic_root_cause = _forensic_root_cause(row)
        _validate_forensic_pairing_relationship(
            forensic_root_cause,
            computed_classification,
            sample_id=sample_id,
        )
        merge_row(
            sample_id,
            {
                "v3_pairing_classification": computed_classification,
                "forensic_root_cause": forensic_root_cause,
                "strict_provenance_status": row.get("strict_provenance_status"),
                "practical_training_eligibility": row.get(
                    "practical_training_eligibility", row.get("training_eligibility")
                ),
                "source_identity_status": row.get("source_identity_status"),
            },
            "seqres_atom_matrix_alignments.jsonl",
            (
                "v3_pairing_classification",
                "forensic_root_cause",
                "strict_provenance_status",
                "practical_training_eligibility",
                "source_identity_status",
            ),
        )
    for source, frame in (
        ("practical_eligibility", practical),
        ("strict_eligibility", strict),
        ("unresolved_eligibility", unresolved),
        ("forensic_transition", transition),
    ):
        for row in frame.to_dict("records"):
            sample_id = str(row["sample_id"])
            recorded_pairing = _clean(row.get("v3_pairing_classification"))
            forensic_root_cause = _forensic_root_cause(row)
            if recorded_pairing is not None:
                _validate_forensic_pairing_relationship(
                    forensic_root_cause,
                    str(recorded_pairing),
                    sample_id=sample_id,
                )
            merge_row(sample_id, row, source, AUTHORITATIVE_SOURCE_FIELDS[source])

    for sample_id, merged in result.items():
        eligibility = merged.get("practical_training_eligibility")
        if merged["unresolved_member"] and eligibility != "excluded_or_unresolved":
            raise ValueError(f"Unresolved sample lacks excluded_or_unresolved eligibility: {sample_id}")
        if merged["practical_eligible_member"] and eligibility == "excluded_or_unresolved":
            raise ValueError(f"Unresolved sample appears in practical eligibility table: {sample_id}")
    return result


def _split_index(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, dict[str, Any]]:
    _require_unique(train, "sample_id", "train manifest")
    _require_unique(validation, "sample_id", "validation manifest")
    overlap = set(train["sample_id"].astype(str)) & set(validation["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"Train/validation sample leakage: {sorted(overlap)[0]}")
    result: dict[str, dict[str, Any]] = {}
    for split, frame in (("train", train), ("validation", validation)):
        for row in frame.to_dict("records"):
            row["original_split"] = split
            result[str(row["sample_id"])] = row
    return result


def _validate_split_rows(processed_index: dict[str, dict[str, Any]], split_index: dict[str, dict[str, Any]]) -> None:
    identity_fields = ("pdb_id", "chain_id", "model_number", "sequence", "length", "path")
    for sample_id, split_row in split_index.items():
        if sample_id not in processed_index:
            raise ValueError(f"Split sample is absent from processed manifest: {sample_id}")
        processed_row = processed_index[sample_id]
        for field in identity_fields:
            if str(split_row.get(field)) != str(processed_row.get(field)):
                raise ValueError(f"Split/processed {field} mismatch for {sample_id}")
        stored_hash = _clean(split_row.get("sequence_hash"))
        if stored_hash is not None and str(stored_hash) != sequence_sha256(str(processed_row["sequence"])):
            raise ValueError(f"Stored split sequence hash mismatch for {sample_id}")


def _assert_no_leakage(rows: pd.DataFrame) -> None:
    selected = rows[rows["eligible_for_training"]]
    train = selected[selected["original_split"] == "train"]
    validation = selected[selected["original_split"] == "validation"]
    for field in ("sequence_sha256", "cluster_id", "split_group_id", "pdb_id", "sample_id"):
        train_values = set(train[field].dropna().astype(str))
        validation_values = set(validation[field].dropna().astype(str))
        overlap = train_values & validation_values
        if overlap:
            raise ValueError(f"Train/validation {field} leakage: {sorted(overlap)[0]}")


def _policy_selected(policy: str, classification: str, row: dict[str, Any]) -> bool:
    if classification == "unresolved_ambiguity":
        return policy == "all_with_status"
    if policy == "all_with_status":
        return True
    if policy == "practical":
        return bool(row.get("practical_eligible_member")) and classification in PRACTICAL_PAIRING_CLASSES
    return bool(row.get("strict_eligible_member"))


def _json_reference(value: Any, field: str) -> str:
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return f"npz:{field}"
    return f"audit:{field}"


def _ordered_exclusion_reasons(reasons: list[str]) -> list[str]:
    positions = {reason: index for index, reason in enumerate(EXCLUSION_REASON_ORDER)}
    return sorted(set(reasons), key=lambda reason: (positions.get(reason, len(positions)), reason))


def _membership_counts(rows: pd.DataFrame) -> dict[str, int]:
    pairing_eligible = rows["pairing_eligible"].astype(bool)
    train = rows["original_split"] == "train"
    validation = rows["original_split"] == "validation"
    split_excluded = ~(train | validation)
    model_eligible = rows["eligible_for_training"].astype(bool)
    return {
        "all_pair_count": len(rows),
        "pairing_eligible_count": int(pairing_eligible.sum()),
        "pairing_ineligible_count": int((~pairing_eligible).sum()),
        "original_train_count": int(train.sum()),
        "original_validation_count": int(validation.sum()),
        "original_split_excluded_count": int(split_excluded.sum()),
        "eligible_train_count": int((model_eligible & train).sum()),
        "eligible_validation_count": int((model_eligible & validation).sum()),
        "pairing_eligible_but_split_excluded_count": int((pairing_eligible & split_excluded).sum()),
        "pairing_ineligible_train_count": int((~pairing_eligible & train).sum()),
        "pairing_ineligible_validation_count": int((~pairing_eligible & validation).sum()),
        "pairing_ineligible_and_split_excluded_count": int((~pairing_eligible & split_excluded).sum()),
        "derived_dataset_excluded_count": int((~model_eligible).sum()),
    }


def _validate_membership_invariants(rows: pd.DataFrame) -> None:
    counts = _membership_counts(rows)
    total = counts["all_pair_count"]
    if counts["pairing_eligible_count"] + counts["pairing_ineligible_count"] != total:
        raise ValueError("Pairing eligibility counts do not cover all audited pairs")
    if (
        counts["original_train_count"] + counts["original_validation_count"] + counts["original_split_excluded_count"]
        != total
    ):
        raise ValueError("Original split counts do not cover all audited pairs")
    if (
        counts["eligible_train_count"] + counts["eligible_validation_count"] + counts["derived_dataset_excluded_count"]
        != total
    ):
        raise ValueError("Derived dataset membership counts do not cover all audited pairs")
    encoded_reasons = rows["exclusion_reasons"].map(json.loads)
    if encoded_reasons[rows["eligible_for_training"]].map(bool).any():
        raise ValueError("Eligible train/validation row has an exclusion reason")
    if (~encoded_reasons[~rows["eligible_for_training"]].map(bool)).any():
        raise ValueError("Derived-excluded row has no exclusion reason")
    if rows.loc[~rows["pairing_eligible"], "eligible_for_training"].any():
        raise ValueError("Pairing-ineligible row is model eligible")
    if rows.loc[~rows["original_split"].isin(["train", "validation"]), "eligible_for_training"].any():
        raise ValueError("Original-split-excluded row is model eligible")


def _build_rows(
    processed: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    alignments: list[dict[str, Any]],
    candidates: pd.DataFrame,
    practical: pd.DataFrame,
    strict: pd.DataFrame,
    unresolved: pd.DataFrame,
    transition: pd.DataFrame,
    normalization_file: Path,
    normalization_sha256: str,
    policy: str,
    evidence_source: str,
    audit_version: Any,
    *,
    candidate_index_override: dict[str, dict[str, Any]] | None = None,
    eligibility_index_override: dict[str, dict[str, Any]] | None = None,
    precomputed_matrix_hashes: dict[str, str] | None = None,
) -> pd.DataFrame:
    _require_unique(processed, "sample_id", "processed manifest")
    _require_unique(processed, "path", "processed manifest")
    processed_index = {str(row["sample_id"]): row for row in processed.to_dict("records")}
    split_index = _split_index(train, validation)
    _validate_split_rows(processed_index, split_index)
    candidate_index = (
        candidate_index_override if candidate_index_override is not None else _candidate_evidence(candidates)
    )
    eligibility_index = (
        eligibility_index_override
        if eligibility_index_override is not None
        else _authoritative_eligibility(alignments, practical, strict, unresolved, transition)
    )
    missing_candidate_evidence = sorted(set(eligibility_index) - set(candidate_index))
    if missing_candidate_evidence:
        raise ValueError(f"Missing candidate evidence for audited sample: {missing_candidate_evidence[0]}")
    practical_ids = set(practical["sample_id"].astype(str))
    seen: set[str] = set()
    output = []
    for alignment in alignments:
        sample_id = str(alignment["sample_id"])
        if sample_id in seen:
            raise ValueError(f"Audit contains duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        if sample_id not in processed_index:
            raise ValueError(f"Audit sample is absent from processed manifest: {sample_id}")
        manifest = processed_index[sample_id]
        split = split_index.get(sample_id, {})
        authoritative = eligibility_index[sample_id]
        if alignment.get("split") in {"train", "validation"} and alignment.get("split") != split.get("original_split"):
            raise ValueError(f"Audit/split assignment mismatch for {sample_id}")
        sequence = str(manifest["sequence"])
        matrix_path = Path(str(manifest["path"]))
        audit_matrix_path = Path(str(alignment.get("matrix_path", matrix_path)))
        eligibility_matrix_path = Path(str(authoritative.get("matrix_path", matrix_path)))
        if audit_matrix_path != matrix_path or eligibility_matrix_path != matrix_path:
            raise ValueError(f"Manifest/audit matrix path contradiction for {sample_id}")
        matrix_length = int(alignment.get("matrix_actual_length") or manifest["length"])
        classification = _pairing_classification(alignment)
        if str(authoritative.get("v3_pairing_classification")) != classification:
            raise ValueError(f"Pairing classification contradiction for {sample_id}")
        authoritative_eligibility = str(authoritative.get("practical_training_eligibility"))
        if authoritative_eligibility != str(alignment.get("training_eligibility")):
            raise ValueError(f"Practical eligibility contradiction for {sample_id}")
        if sample_id in practical_ids and classification == "unresolved_ambiguity":
            raise ValueError(f"Unresolved sample appears in practical eligibility table: {sample_id}")
        reasons: list[str] = []
        if not canonical_sequence(sequence):
            reasons.append("noncanonical_sequence")
        if int(manifest["length"]) != len(sequence) or matrix_length != len(sequence):
            reasons.append("sequence_matrix_length_mismatch")
        if not _optional_bool(
            alignment.get("manifest_npz_sequence_match", alignment.get("matrix_manifest_sequence_match")),
            "manifest_npz_sequence_match",
        ):
            reasons.append("manifest_npz_sequence_mismatch")
        if not matrix_path.is_file():
            reasons.append("matrix_path_missing")
        if classification == "unresolved_ambiguity":
            reasons.append("unresolved_ambiguity")
        if alignment.get("coordinate_matrix_status") == "failed":
            reasons.append("coordinate_matrix_mismatch")
        pairing_eligible = (
            authoritative_eligibility != "excluded_or_unresolved"
            and classification in PRACTICAL_PAIRING_CLASSES
            and not reasons
        )
        selected = _policy_selected(policy, classification, authoritative)
        original_split = split.get("original_split", "excluded")
        eligible = pairing_eligible and selected and original_split in {"train", "validation"}
        if selected and not pairing_eligible and reasons and classification != "unresolved_ambiguity":
            raise ValueError(f"Pairing integrity failure for {sample_id}: {', '.join(reasons)}")
        identity = candidate_index.get(sample_id, {})
        if identity.get("strong_author_linked_physical_candidate_count", 0) > 1:
            raise ValueError(f"Multiple strong author-linked candidates for {sample_id}")
        equivalent_unique_decision = (
            _optional_bool(alignment.get("unique_author_linked_candidate"), "unique_author_linked_candidate")
            and int(alignment.get("author_linked_candidate_count") or 0) == 1
        )
        if classification != "unresolved_ambiguity":
            if identity.get("author_linked_physical_candidate_count", 0) == 0:
                raise ValueError(f"No author-linked candidate for eligible pairing {sample_id}")
            if not identity.get("selected_candidate_unique") or not (
                identity.get("strong_author_linked_physical_candidate_count") == 1 or equivalent_unique_decision
            ):
                raise ValueError(f"Candidate evidence contradicts practical eligibility for {sample_id}")
        source_file = str(manifest.get("source_file") or alignment.get("source_file") or "")
        selected_candidate_source = identity.get("candidate_source_file")
        if selected_candidate_source is not None and Path(str(selected_candidate_source)) != Path(source_file):
            raise ValueError(f"Selected candidate source contradiction for {sample_id}")
        if original_split not in {"train", "validation"}:
            reasons.append("original_split_excluded")
        if pairing_eligible and not selected:
            reasons.append("not_selected_by_eligibility_policy")
        reasons = _ordered_exclusion_reasons(reasons)
        residue_status = (
            str(alignment.get("selected_residue_id_convention"))
            if any(
                _optional_bool(alignment.get(key), key)
                for key in (
                    "auth_residue_ids_match",
                    "label_residue_ids_match",
                    "zero_based_positions_match",
                    "one_based_positions_match",
                )
            )
            else "incomplete_or_unresolved"
        )
        trim_status = (
            "compatible"
            if _optional_bool(alignment.get("retained_interval_match"), "retained_interval_match")
            else "unavailable"
            if not _optional_bool(alignment.get("trim_metadata_available"), "trim_metadata_available")
            else "incompatible"
        )
        output.append(
            {
                "schema_version": PAIRING_SCHEMA_VERSION,
                "sample_id": sample_id,
                "pdb_id": str(manifest["pdb_id"]),
                "source_file": source_file,
                "chain_id": str(manifest["chain_id"]),
                "label_asym_id": identity.get("label_asym_id") or _clean(alignment.get("raw_label_asym_id")),
                "auth_asym_id": identity.get("auth_asym_id") or _clean(alignment.get("raw_auth_asym_id")),
                "entity_id": identity.get("entity_id"),
                "model_number": int(manifest.get("model_number", alignment.get("model_id", 1))),
                "sequence": sequence,
                "sequence_length": len(sequence),
                "sequence_sha256": sequence_sha256(sequence),
                "sequence_token_ids_reference": f"vocabulary:{VOCABULARY_VERSION}",
                "vocabulary_version": VOCABULARY_VERSION,
                "sequence_source": _clean(alignment.get("sequence_source")) or "unknown",
                "modified_residue_mapping_version": MODIFIED_RESIDUE_MAPPING_VERSION,
                "matrix_path": str(matrix_path),
                "matrix_length": matrix_length,
                "matrix_sha256": (
                    precomputed_matrix_hashes.get(sample_id)
                    if precomputed_matrix_hashes is not None
                    else sha256_file(matrix_path)
                    if matrix_path.is_file()
                    else None
                ),
                "distance_semantics": "C-alpha/C-alpha Euclidean distance in angstroms",
                "normalization_file": str(normalization_file),
                "normalization_sha256": normalization_sha256,
                "residue_ids": _json_reference(alignment.get("residue_ids"), "residue_ids"),
                "insertion_codes": _json_reference(alignment.get("insertion_codes"), "insertion_codes"),
                "selected_altlocs": _json_reference(alignment.get("selected_altlocs"), "selected_altlocs"),
                "residue_mask": "implicit:all_retained_residues_valid",
                "coordinate_availability": str(alignment.get("coordinate_matrix_status") or "unavailable"),
                "pairing_classification": classification,
                "forensic_root_cause": str(authoritative.get("forensic_root_cause") or "unknown"),
                "pairing_eligible": pairing_eligible,
                "practical_training_eligibility": authoritative_eligibility,
                "strict_provenance_status": str(authoritative.get("strict_provenance_status") or "unknown"),
                "source_identity_status": str(authoritative.get("source_identity_status") or "unknown"),
                "author_chain_unique": _optional_bool(
                    alignment.get("unique_author_linked_candidate"), "unique_author_linked_candidate"
                ),
                "candidate_count": int(alignment.get("competing_candidate_count") or 0),
                "manifest_matrix_association_count": 1,
                "raw_polymer_candidate_count": int(alignment.get("raw_match_count") or 0),
                "raw_physical_candidate_count": int(identity.get("raw_physical_candidate_count", 0)),
                "author_linked_physical_candidate_count": int(
                    identity.get("author_linked_physical_candidate_count", 0)
                ),
                "non_author_linked_candidate_count": int(identity.get("non_author_linked_candidate_count", 0)),
                "candidate_evidence_row_count": int(identity.get("candidate_evidence_row_count", 0)),
                "strong_physical_candidate_count_all": int(identity.get("strong_physical_candidate_count_all", 0)),
                "strong_author_linked_physical_candidate_count": int(
                    identity.get("strong_author_linked_physical_candidate_count", 0)
                ),
                "strong_non_author_linked_candidate_count": int(
                    identity.get("strong_non_author_linked_candidate_count", 0)
                ),
                "strong_evidence_row_count": int(identity.get("strong_evidence_row_count", 0)),
                "supported_residue_id_conventions": json.dumps(
                    identity.get("supported_residue_id_conventions", []), sort_keys=True
                ),
                "selected_candidate_unique": bool(identity.get("selected_candidate_unique", False)),
                "rejected_candidate_evidence": json.dumps(
                    identity.get("rejected_candidate_evidence", []), sort_keys=True
                ),
                "sequence_occ_matrix_match": int(alignment.get("matrix_sequence_occurrence_count_in_candidate") or 0)
                == 1,
                "residue_identity_match_status": residue_status,
                "coordinate_matrix_match_status": str(alignment.get("coordinate_matrix_status") or "unavailable"),
                "coordinate_max_abs_error_angstrom": _clean(
                    alignment.get("coordinate_distance_max_abs_error_angstrom")
                ),
                "terminal_trim_metadata_status": trim_status,
                "missing_calpha_policy": str(manifest.get("missing_calpha_policy") or "reject"),
                "exclusion_reasons": json.dumps(reasons, sort_keys=True),
                "evidence_source": evidence_source,
                "audit_version": audit_version,
                "eligibility_policy": policy,
                "policy_selected": selected,
                "selected_by_eligibility_policy": selected,
                "eligible_for_training": eligible,
                "dataset_membership_status": (f"eligible_{original_split}" if eligible else "excluded"),
                "original_split": original_split,
                "cluster_id": _clean(split.get("cluster_id")),
                "split_group_id": _clean(split.get("split_group_id")),
                "exact_sequence_count": _clean(split.get("exact_sequence_count")),
                "sample_weight": _clean(split.get("sample_weight")),
                "experimental_method": _clean(manifest.get("experimental_method")),
                "requested_length": _clean(alignment.get("requested_length")),
                "actual_length": int(manifest["length"]),
                "original_chain_length": int(manifest.get("original_chain_length", manifest["length"])),
                "trimmed_n_terminal_residues": int(manifest.get("trimmed_n_terminal_residues", 0)),
                "trimmed_c_terminal_residues": int(manifest.get("trimmed_c_terminal_residues", 0)),
                "terminal_trimming_applied": bool(manifest.get("terminal_trimming_applied", False)),
                "trimmed_fraction": float(manifest.get("trimmed_fraction", 0.0)),
                "max_terminal_trim_fraction": _clean(manifest.get("max_terminal_trim_fraction")),
            }
        )
    frame = pd.DataFrame(output)
    if frame.empty:
        raise ValueError("Sequence-readiness audit contains no pairing rows")
    if frame["matrix_path"].duplicated().any():
        duplicate = frame.loc[frame["matrix_path"].duplicated(keep=False), "matrix_path"].iloc[0]
        raise ValueError(f"Audit contains duplicate matrix path: {duplicate}")
    _assert_no_leakage(frame)
    _validate_membership_invariants(frame)
    return frame


def _write_summaries(directory: Path, rows: pd.DataFrame) -> None:
    pairing = (
        rows.groupby(
            [
                "pairing_classification",
                "practical_training_eligibility",
                "pairing_eligible",
                "original_split",
                "selected_by_eligibility_policy",
                "dataset_membership_status",
            ],
            dropna=False,
        )
        .size()
        .rename("pair_count")
        .reset_index()
    )
    pairing.to_csv(directory / "pairing_summary.csv", index=False)
    reasons = Counter(
        reason
        for encoded in rows.loc[~rows["eligible_for_training"], "exclusion_reasons"]
        for reason in json.loads(encoded)
    )
    with (directory / "exclusion_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["exclusion_reason", "pair_count"])
        writer.writeheader()
        ordered = _ordered_exclusion_reasons(list(reasons))
        writer.writerows({"exclusion_reason": reason, "pair_count": reasons[reason]} for reason in ordered)


def build_sequence_geometry_pairing(
    *,
    processed_manifest: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    audit_dir: str | Path,
    normalization_file: str | Path,
    output_dir: str | Path,
    eligibility_policy: str,
    allow_pilot_evidence: bool = False,
    state_dir: str | Path | None = None,
    resume: bool = False,
    batch_size: int = 4_096,
    max_memory_mib: int = 4_096,
    checkpoint_frequency: int = 1_000,
    maximum_failure_examples: int = 100,
) -> dict[str, Any]:
    """Validate evidence and atomically publish a derived pairing directory."""
    if eligibility_policy not in ELIGIBILITY_POLICIES:
        raise ValueError(f"eligibility_policy must be one of {sorted(ELIGIBILITY_POLICIES)}")
    processed_path = Path(processed_manifest)
    train_path = Path(train_manifest)
    validation_path = Path(validation_manifest)
    audit_path = Path(audit_dir)
    normalization_path = Path(normalization_file)
    destination = Path(output_dir)
    audit_protocol = json.loads((audit_path / "sequence_readiness_protocol.json").read_text())
    if (
        audit_protocol.get("audit_mode") == "raw-full"
        and audit_protocol.get("raw_inputs_unchanged") is True
        and SequenceReadinessArtifactReader(audit_path).storage_profile == "compact-v1"
    ):
        from protein_distance_diffusion.data.pairing_streaming import streaming_build_sequence_geometry_pairing

        streaming_state = (
            Path(state_dir) if state_dir is not None else destination.parent / f".{destination.name}.pairing-state"
        )
        return streaming_build_sequence_geometry_pairing(
            processed_manifest=processed_path,
            train_manifest=train_path,
            validation_manifest=validation_path,
            audit_dir=audit_path,
            normalization_file=normalization_path,
            output_dir=destination,
            state_dir=streaming_state,
            eligibility_policy=eligibility_policy,
            resume=resume,
            batch_size=batch_size,
            max_memory_mib=max_memory_mib,
            checkpoint_frequency=checkpoint_frequency,
            maximum_failure_examples=maximum_failure_examples,
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite derived output: {destination}")
    (
        processed,
        train,
        validation,
        alignments,
        candidates,
        practical,
        strict,
        unresolved,
        transition,
        audit_protocol,
        input_hashes,
    ) = _load_inputs(
        processed_path,
        train_path,
        validation_path,
        audit_path,
        normalization_path,
        allow_pilot_evidence=allow_pilot_evidence,
    )
    rows = _build_rows(
        processed,
        train,
        validation,
        alignments,
        candidates,
        practical,
        strict,
        unresolved,
        transition,
        normalization_path,
        input_hashes[str(normalization_path)],
        eligibility_policy,
        str(audit_path),
        audit_protocol.get("report_semantics_version"),
    )
    if (
        audit_protocol.get("audit_mode") == "raw-full"
        and audit_protocol.get("raw_inputs_unchanged") is True
        and SequenceReadinessArtifactReader(audit_path).storage_profile == "compact-v1"
    ):
        unresolved_selected = rows[rows["selected_by_eligibility_policy"] & ~rows["pairing_eligible"]]
        if not unresolved_selected.empty:
            raise ValueError("Definitive raw-full output cannot include unresolved selected pairs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        rows.to_parquet(temporary / "all_pairs.parquet", index=False)
        rows[rows["eligible_for_training"] & (rows["original_split"] == "train")].to_parquet(
            temporary / "eligible_train.parquet", index=False
        )
        rows[rows["eligible_for_training"] & (rows["original_split"] == "validation")].to_parquet(
            temporary / "eligible_validation.parquet", index=False
        )
        rows[~rows["eligible_for_training"]].to_parquet(temporary / "excluded_pairs.parquet", index=False)
        rows[~rows["pairing_eligible"]].to_parquet(temporary / "pairing_ineligible.parquet", index=False)
        rows[rows["pairing_eligible"] & ~rows["original_split"].isin(["train", "validation"])].to_parquet(
            temporary / "pairing_eligible_original_split_excluded.parquet", index=False
        )
        _write_summaries(temporary, rows)
        _atomic_json(temporary / "vocabulary.json", SequenceGeometryVocabulary().as_dict())
        schema = {
            "schema_version": PAIRING_SCHEMA_VERSION,
            "columns": {column: str(dtype) for column, dtype in rows.dtypes.items()},
            "large_array_references": {
                "residue_ids": "audit JSON or NPZ residue_ids",
                "insertion_codes": "audit JSON or NPZ insertion_codes",
                "selected_altlocs": "audit JSON or NPZ selected_altlocs",
                "residue_mask": "all retained residues are valid",
            },
        }
        _atomic_json(temporary / "schema.json", schema)
        current_hashes = {path: sha256_file(path) for path in input_hashes}
        if current_hashes != input_hashes:
            raise RuntimeError("Input files changed during pairing build")
        with (temporary / "input_hashes.sha256").open("w") as handle:
            for path, digest in sorted(input_hashes.items()):
                handle.write(f"{digest}  {path}\n")
        protocol = {
            "status": "completed",
            "schema_version": PAIRING_SCHEMA_VERSION,
            "eligibility_policy": eligibility_policy,
            "audit_mode": audit_protocol.get("audit_mode"),
            "allow_pilot_evidence": bool(allow_pilot_evidence),
            "input_hashes": input_hashes,
            "input_hashes_preserved": True,
            **_membership_counts(rows),
            "dataset_exclusion_semantics": (
                "derived_dataset_excluded_count is the union of pairing-ineligible, original-split-excluded, "
                "and policy-unselected rows"
            ),
            "pairing_classification_counts": {
                str(key): int(value) for key, value in rows["pairing_classification"].value_counts().items()
            },
            "manifest_matrix_association_count": {
                str(key): int(value)
                for key, value in rows["manifest_matrix_association_count"].value_counts().sort_index().items()
            },
            "raw_physical_candidate_count": {
                str(key): int(value)
                for key, value in rows["raw_physical_candidate_count"].value_counts().sort_index().items()
            },
            "author_linked_physical_candidate_count": {
                str(key): int(value)
                for key, value in rows["author_linked_physical_candidate_count"].value_counts().sort_index().items()
            },
            "non_author_linked_candidate_count": {
                str(key): int(value)
                for key, value in rows["non_author_linked_candidate_count"].value_counts().sort_index().items()
            },
            "candidate_evidence_row_count": int(rows["candidate_evidence_row_count"].sum()),
            "strong_physical_candidate_count_all": {
                str(key): int(value)
                for key, value in rows["strong_physical_candidate_count_all"].value_counts().sort_index().items()
            },
            "strong_author_linked_physical_candidate_count": {
                str(key): int(value)
                for key, value in rows["strong_author_linked_physical_candidate_count"]
                .value_counts()
                .sort_index()
                .items()
            },
            "strong_non_author_linked_candidate_count": {
                str(key): int(value)
                for key, value in rows["strong_non_author_linked_candidate_count"].value_counts().sort_index().items()
            },
            "strong_evidence_row_count": int(rows["strong_evidence_row_count"].sum()),
            "selected_candidate_unique": {
                str(key).lower(): int(value) for key, value in rows["selected_candidate_unique"].value_counts().items()
            },
            "physical_candidate_key_fields": [
                "source_file",
                "model_number",
                "entity_id",
                "label_asym_id",
                "auth_asym_id",
            ],
            "source_audit_protocol_sha256": input_hashes[str(audit_path / "sequence_readiness_protocol.json")],
            "geometry_validation_basis": (
                "completed audit NPZ metadata validation plus per-row matrix dimensions; "
                "full finite-value validation is enforced when a sample is loaded"
            ),
        }
        _atomic_json(temporary / "protocol.json", protocol)
        temporary.replace(destination)
        return protocol
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_sequence_geometry_pairing(
    *,
    processed_manifest: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    audit_dir: str | Path,
    normalization_file: str | Path,
    validation_report: str | Path,
    eligibility_policy: str,
    allow_pilot_evidence: bool = False,
    expected_total: int | None = None,
    expected_eligible: int | None = None,
    expected_excluded: int | None = None,
    state_dir: str | Path | None = None,
    resume: bool = False,
    batch_size: int = 4_096,
    max_memory_mib: int = 4_096,
    checkpoint_frequency: int = 1_000,
    maximum_failure_examples: int = 100,
) -> dict[str, Any]:
    """Validate all audited rows and publish only a concise diagnostic report."""
    if eligibility_policy not in ELIGIBILITY_POLICIES:
        raise ValueError(f"eligibility_policy must be one of {sorted(ELIGIBILITY_POLICIES)}")
    processed_path = Path(processed_manifest)
    train_path = Path(train_manifest)
    validation_path = Path(validation_manifest)
    audit_path = Path(audit_dir)
    normalization_path = Path(normalization_file)
    audit_protocol = json.loads((audit_path / "sequence_readiness_protocol.json").read_text())
    if (
        audit_protocol.get("audit_mode") == "raw-full"
        and audit_protocol.get("raw_inputs_unchanged") is True
        and SequenceReadinessArtifactReader(audit_path).storage_profile == "compact-v1"
    ):
        from protein_distance_diffusion.data.pairing_streaming import streaming_validate_sequence_geometry_pairing

        report_path = Path(validation_report)
        streaming_state = (
            Path(state_dir) if state_dir is not None else report_path.parent / f".{report_path.stem}.pairing-state"
        )
        return streaming_validate_sequence_geometry_pairing(
            processed_manifest=processed_path,
            train_manifest=train_path,
            validation_manifest=validation_path,
            audit_dir=audit_path,
            normalization_file=normalization_path,
            validation_report=report_path,
            state_dir=streaming_state,
            eligibility_policy=eligibility_policy,
            resume=resume,
            batch_size=batch_size,
            max_memory_mib=max_memory_mib,
            checkpoint_frequency=checkpoint_frequency,
            maximum_failure_examples=maximum_failure_examples,
            expected_total=expected_total,
            expected_eligible=expected_eligible,
            expected_excluded=expected_excluded,
        )
    (
        processed,
        train,
        validation,
        alignments,
        candidates,
        practical,
        strict,
        unresolved,
        transition,
        audit_protocol,
        input_hashes,
    ) = _load_inputs(
        processed_path,
        train_path,
        validation_path,
        audit_path,
        normalization_path,
        allow_pilot_evidence=allow_pilot_evidence,
    )

    def grouped(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {str(sample_id): group.copy() for sample_id, group in frame.groupby("sample_id", sort=False)}

    grouped_inputs = {
        "processed": grouped(processed),
        "train": grouped(train),
        "validation": grouped(validation),
        "candidates": grouped(candidates),
        "practical": grouped(practical),
        "strict": grouped(strict),
        "unresolved": grouped(unresolved),
        "transition": grouped(transition),
    }
    empty = {
        name: frame.iloc[0:0].copy()
        for name, frame in (
            ("processed", processed),
            ("train", train),
            ("validation", validation),
            ("candidates", candidates),
            ("practical", practical),
            ("strict", strict),
            ("unresolved", unresolved),
            ("transition", transition),
        )
    }
    failures: dict[str, list[str]] = {}
    successful_rows = []

    def record(reason: str, sample_ids: list[str]) -> None:
        failures.setdefault(reason, []).extend(sample_ids)

    alignment_counts = Counter(str(row["sample_id"]) for row in alignments)
    for sample_id, count in alignment_counts.items():
        if count != 1:
            record("duplicate_audit_alignment", [sample_id])
    for path, group in processed.groupby("path", sort=False):
        sample_ids = sorted(set(group["sample_id"].astype(str)))
        if len(sample_ids) > 1:
            record(f"matrix_path_shared_by_multiple_samples:{path}", sample_ids)

    for alignment in alignments:
        sample_id = str(alignment["sample_id"])
        try:
            row_frame = _build_rows(
                grouped_inputs["processed"].get(sample_id, empty["processed"]),
                grouped_inputs["train"].get(sample_id, empty["train"]),
                grouped_inputs["validation"].get(sample_id, empty["validation"]),
                [alignment],
                grouped_inputs["candidates"].get(sample_id, empty["candidates"]),
                grouped_inputs["practical"].get(sample_id, empty["practical"]),
                grouped_inputs["strict"].get(sample_id, empty["strict"]),
                grouped_inputs["unresolved"].get(sample_id, empty["unresolved"]),
                grouped_inputs["transition"].get(sample_id, empty["transition"]),
                normalization_path,
                input_hashes[str(normalization_path)],
                eligibility_policy,
                str(audit_path),
                audit_protocol.get("report_semantics_version"),
            )
            successful_rows.append(row_frame)
        except Exception as exc:
            record(f"{type(exc).__name__}:{exc}", [sample_id])

    combined = pd.concat(successful_rows, ignore_index=True) if successful_rows else pd.DataFrame()
    if not combined.empty:
        try:
            _assert_no_leakage(combined)
            _validate_membership_invariants(combined)
        except Exception as exc:
            record(f"{type(exc).__name__}:{exc}", sorted(set(combined["sample_id"].astype(str))))

    total = len(alignments)
    validated_eligible = (
        int((combined["practical_training_eligibility"] != "excluded_or_unresolved").sum()) if not combined.empty else 0
    )
    validated_excluded = (
        int((combined["practical_training_eligibility"] == "excluded_or_unresolved").sum()) if not combined.empty else 0
    )
    for name, observed, expected in (
        ("total", total, expected_total),
        ("eligible", validated_eligible, expected_eligible),
        ("excluded", validated_excluded, expected_excluded),
    ):
        if expected is not None and observed != expected:
            record(f"expected_{name}_count:{expected}:observed:{observed}", ["__contract__"])

    current_hashes = {path: sha256_file(path) for path in input_hashes}
    raw_inputs_unchanged = current_hashes == input_hashes
    if not raw_inputs_unchanged:
        record("input_hash_changed_during_validation", ["__global__"])
    normalized_failures = {reason: sorted(set(sample_ids)) for reason, sample_ids in sorted(failures.items())}
    report = {
        "status": "passed" if not normalized_failures else "failed",
        "schema_version": PAIRING_SCHEMA_VERSION,
        "eligibility_policy": eligibility_policy,
        "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
        "audit_mode": audit_protocol.get("audit_mode"),
        "total_audited_pairs": total,
        "expected_total_count": expected_total,
        "expected_eligible_count": expected_eligible,
        "expected_excluded_count": expected_excluded,
        "validated_eligible_count": validated_eligible,
        "validated_excluded_count": validated_excluded,
        "validation_count_semantics": (
            "expected/validated eligible and excluded counts refer to pairing eligibility; "
            "validated_membership_counts describes derived train/validation membership"
        ),
        "validated_pairing_eligible_count": validated_eligible,
        "validated_pairing_ineligible_count": validated_excluded,
        "validated_membership_counts": _membership_counts(combined) if not combined.empty else {},
        "failure_sample_ids_by_reason": normalized_failures,
        "failure_count": sum(len(sample_ids) for sample_ids in normalized_failures.values()),
        "representative_candidate_counts": {
            "raw_physical_candidate_count": (
                {
                    str(key): int(value)
                    for key, value in combined["raw_physical_candidate_count"].value_counts().sort_index().items()
                }
                if not combined.empty
                else {}
            ),
            "strong_author_linked_physical_candidate_count": (
                {
                    str(key): int(value)
                    for key, value in combined["strong_author_linked_physical_candidate_count"]
                    .value_counts()
                    .sort_index()
                    .items()
                }
                if not combined.empty
                else {}
            ),
        },
        "input_hashes": input_hashes,
        "raw_inputs_unchanged": raw_inputs_unchanged,
        "matrix_reads_performed": 0,
        "matrix_files_hashed": len(combined),
    }
    report_path = Path(validation_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    return report
