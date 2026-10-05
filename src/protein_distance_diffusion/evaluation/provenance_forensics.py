"""Bounded sequence/matrix provenance-forensics helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from protein_distance_diffusion.data.preprocess import compute_distance_matrix

CLASSIFICATIONS = {
    "audit_chain_entity_mapping_bug",
    "audit_residue_id_convention_bug",
    "audit_coordinate_reconstruction_bug",
    "historical_preprocessing_policy_difference",
    "legacy_trim_metadata_inconsistency",
    "raw_source_version_unverifiable",
    "genuine_sequence_matrix_mismatch",
    "coordinate_provenance_unavailable",
    "unresolved_ambiguity",
}
REFINED_CLASSIFICATIONS = {
    "verified_sequence_geometry_pair",
    "unique_sequence_pair_coordinate_unavailable",
    "residue_id_namespace_mismatch",
    "historical_missing_calpha_selection_unreproducible",
    "multiple_author_linked_polymer_candidates",
    "matrix_sequence_absent_from_author_linked_polymer",
    "coordinate_disagreement",
    "demonstrated_sequence_matrix_mismatch",
    "unresolved_ambiguity",
}
CONVENTIONS = ("auth_seq_id_insertion", "label_seq_id", "position_zero_based", "position_one_based")


def _values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(json.loads(value))


def _ordered_indices(raw_ids: list[str], matrix_ids: list[str]) -> tuple[list[int], bool]:
    """Find at most two ordered identity paths without materializing combinations."""
    states: list[list[int]] = [[]]
    for matrix_id in matrix_ids:
        positions = [index for index, raw_id in enumerate(raw_ids) if raw_id == matrix_id]
        next_states = [
            state + [position] for state in states for position in positions if not state or position > state[-1]
        ]
        deduplicated: dict[tuple[int, ...], list[int]] = {tuple(state): state for state in next_states}
        states = sorted(deduplicated.values(), key=lambda state: (state[-1], state))[:2]
        if not states:
            return [], False
    return states[0], len(states) > 1


def convention_indices(raw: dict[str, Any], matrix_ids: list[str], convention: str) -> tuple[list[int], bool]:
    """Map stored residue IDs to one documented raw-residue convention."""
    if convention == "auth_seq_id_insertion":
        auth = [str(value) for value in _values(raw.get("auth_sequence_ids"))]
        insertions = [str(value) for value in _values(raw.get("insertion_codes"))]
        return _ordered_indices(
            [f"{number}{insertion}" for number, insertion in zip(auth, insertions, strict=True)], matrix_ids
        )
    if convention == "label_seq_id":
        return _ordered_indices([str(value) for value in _values(raw.get("label_sequence_ids"))], matrix_ids)
    offset = 0 if convention == "position_zero_based" else 1
    try:
        indices = [int(value) - offset for value in matrix_ids]
    except ValueError:
        return [], False
    if not indices or min(indices) < 0 or max(indices) >= len(_values(raw.get("residue_tokens"))):
        return [], False
    return indices, len(set(indices)) != len(indices) or indices != sorted(indices)


def matrix_errors(left: np.ndarray, right: np.ndarray) -> tuple[float | None, float | None]:
    """Return RMSE and maximum absolute error for equally shaped matrices."""
    if left.shape != right.shape or left.ndim != 2:
        return None, None
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(difference * difference))), float(np.max(np.abs(difference), initial=0.0))


def score_candidate(
    raw: dict[str, Any],
    *,
    matrix_sequence: str,
    matrix_ids: list[str],
    matrix: np.ndarray,
    metadata: dict[str, Any],
    requested_chain: str,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Score one protein chain/model under each explicit residue-ID convention."""
    tokens = [str(value) for value in _values(raw.get("residue_tokens"))]
    coordinates = _values(raw.get("selected_calpha_coordinates"))
    label_ids = [str(value) for value in _values(raw.get("label_sequence_ids"))]
    raw_insertions = [str(value) for value in _values(raw.get("insertion_codes"))]
    stored_insertions = [str(value) for value in metadata.get("retained_insertion_codes") or []]
    seqres = str(raw.get("seqres_sequence") or "")
    atom = str(raw.get("atom_sequence") or "")

    def occurrences(evidence: str) -> int:
        return sum(
            evidence.startswith(matrix_sequence, offset)
            for offset in range(max(len(evidence) - len(matrix_sequence) + 1, 0))
        )

    seqres_occurrences = occurrences(seqres)
    atom_occurrences = occurrences(atom)
    scores = []
    for convention in CONVENTIONS:
        indices, ambiguous = convention_indices(raw, matrix_ids, convention)
        selected_sequence = "".join(tokens[index] for index in indices) if indices else ""
        coordinate_status = "unavailable"
        rmse = maximum = None
        if indices and len(coordinates) == len(tokens):
            selected_coordinates = [coordinates[index] for index in indices]
            if all(value is not None for value in selected_coordinates):
                reconstructed = compute_distance_matrix(np.asarray(selected_coordinates, dtype=np.float32))
                rmse, maximum = matrix_errors(matrix, reconstructed)
                coordinate_status = "passed" if maximum is not None and maximum <= tolerance else "failed"
        contiguous = bool(indices) and indices == list(range(indices[0], indices[-1] + 1))
        expected_n = indices[0] if indices else None
        expected_c = len(tokens) - indices[-1] - 1 if indices else None
        start = metadata.get("retained_start_label_seq_id")
        end = metadata.get("retained_end_label_seq_id")
        trimmed_n = metadata.get("trimmed_n_terminal_residues")
        trimmed_c = metadata.get("trimmed_c_terminal_residues")
        trim_fields_available = all(value is not None for value in (start, end, trimmed_n, trimmed_c))
        trim_compatible = bool(
            indices
            and contiguous
            and trim_fields_available
            and str(start) == label_ids[indices[0]]
            and str(end) == label_ids[indices[-1]]
            and int(trimmed_n) == expected_n
            and int(trimmed_c) == expected_c
        )
        sequence_match = bool(indices) and selected_sequence == matrix_sequence
        scores.append(
            {
                "label_asym_id": raw.get("label_asym_id"),
                "auth_asym_id": raw.get("auth_asym_id"),
                "entity_id": raw.get("entity_id"),
                "polymer_type": raw.get("polymer_type"),
                "model_number": str(raw.get("model_id")),
                "requested_chain": requested_chain,
                "author_chain_match": str(raw.get("auth_asym_id")) == requested_chain,
                "label_chain_match": str(raw.get("label_asym_id")) == requested_chain,
                "convention": convention,
                "raw_atom_sequence": raw.get("atom_sequence"),
                "raw_seqres_sequence": raw.get("seqres_sequence"),
                "raw_entity_poly_sequence": raw.get("entity_poly_sequence"),
                "raw_sequence_length": len(tokens),
                "seqres_sequence_occurrence_count": seqres_occurrences,
                "atom_sequence_occurrence_count": atom_occurrences,
                "matrix_sequence_occurrence_count_in_candidate": max(seqres_occurrences, atom_occurrences),
                "mapped_indices": json.dumps(indices),
                "identity_mapping_available": bool(indices),
                "identity_mapping_ambiguous": ambiguous,
                "selected_sequence": selected_sequence or None,
                "sequence_match": sequence_match,
                "insertion_codes_match": bool(
                    indices and stored_insertions and [raw_insertions[index] for index in indices] == stored_insertions
                ),
                "contiguous_interval": contiguous,
                "trim_metadata_available": trim_fields_available,
                "trim_metadata_compatible": trim_compatible,
                "recorded_retained_start_label_seq_id": start,
                "recorded_retained_end_label_seq_id": end,
                "recorded_trimmed_n_terminal_residues": trimmed_n,
                "recorded_trimmed_c_terminal_residues": trimmed_c,
                "expected_trimmed_n": expected_n,
                "expected_trimmed_c": expected_c,
                "coordinate_status": coordinate_status,
                "coordinate_rmse_angstrom": rmse,
                "coordinate_max_abs_error_angstrom": maximum,
                "strong_evidential_match": bool(
                    not ambiguous
                    and sequence_match
                    and (coordinate_status == "passed" or convention in {"auth_seq_id_insertion", "label_seq_id"})
                ),
            }
        )
    return scores


def independent_pair_evidence(
    case: dict[str, Any], scores: list[dict[str, Any]], *, source_identity_status: str
) -> dict[str, Any]:
    """Summarize independent namespaces without collapsing missing into failed."""
    author_rows = [row for row in scores if bool(row.get("author_chain_match"))]
    author_keys = {
        (str(row.get("label_asym_id")), str(row.get("entity_id")), str(row.get("model_number"))) for row in author_rows
    }
    selected_rows = author_rows if len(author_keys) == 1 else []

    def namespace_match(name: str) -> bool:
        rows = [row for row in selected_rows if row.get("convention") == name]
        return bool(
            rows
            and rows[0].get("identity_mapping_available")
            and not rows[0].get("identity_mapping_ambiguous")
            and rows[0].get("sequence_match")
        )

    matrix_sequence = str(case.get("matrix_sequence") or case.get("sequence") or "")

    def candidate_occurrences(row: dict[str, Any]) -> int:
        if "matrix_sequence_occurrence_count_in_candidate" in row:
            return int(row["matrix_sequence_occurrence_count_in_candidate"])
        evidence = str(row.get("raw_seqres_sequence") or row.get("raw_atom_sequence") or "")
        return sum(
            evidence.startswith(matrix_sequence, offset)
            for offset in range(max(len(evidence) - len(matrix_sequence) + 1, 0))
        )

    occurrences = max((candidate_occurrences(row) for row in selected_rows), default=0)
    coordinate_statuses = {str(row.get("coordinate_status", "unavailable")) for row in selected_rows}
    matching_rows = [
        row for row in selected_rows if row.get("sequence_match") and not row.get("identity_mapping_ambiguous")
    ]
    return {
        "manifest_npz_sequence_match": bool(case.get("matrix_manifest_sequence_match", True)),
        "unique_author_linked_candidate": len(author_keys) == 1,
        "author_linked_candidate_count": len(author_keys),
        "matrix_sequence_occurrence_count_in_candidate": occurrences,
        "retained_interval_match": any(bool(row.get("trim_metadata_compatible")) for row in selected_rows),
        "trim_metadata_available": any(bool(row.get("trim_metadata_available")) for row in matching_rows),
        "auth_residue_ids_match": namespace_match("auth_seq_id_insertion"),
        "label_residue_ids_match": namespace_match("label_seq_id"),
        "zero_based_positions_match": namespace_match("position_zero_based"),
        "one_based_positions_match": namespace_match("position_one_based"),
        "insertion_codes_match": any(bool(row.get("insertion_codes_match")) for row in selected_rows),
        "raw_calpha_available_for_selected_interval": bool(coordinate_statuses - {"unavailable"}),
        "coordinate_matrix_match": "passed" in coordinate_statuses,
        "coordinate_matrix_status": (
            "passed"
            if "passed" in coordinate_statuses
            else "failed"
            if "failed" in coordinate_statuses
            else "unavailable"
        ),
        "source_identity_status": source_identity_status,
        "competing_candidate_count": len(
            {(str(row.get("label_asym_id")), str(row.get("entity_id")), str(row.get("model_number"))) for row in scores}
        ),
    }


def refine_pair_status(case: dict[str, Any], scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Refine primary evidence, strict provenance, and practical eligibility."""
    source_status = str(case.get("source_identity_status") or "no_state_evidence")
    evidence = independent_pair_evidence(case, scores, source_identity_status=source_status)
    original_primary = str(case.get("primary_classification", ""))
    exact_namespace = evidence["auth_residue_ids_match"] or evidence["zero_based_positions_match"]
    alternate_namespace = evidence["label_residue_ids_match"] or evidence["one_based_positions_match"]
    complete_pair = bool(
        evidence["unique_author_linked_candidate"]
        and evidence["manifest_npz_sequence_match"]
        and evidence["matrix_sequence_occurrence_count_in_candidate"] == 1
        and exact_namespace
        and evidence["coordinate_matrix_match"]
    )
    historical_identity_incomplete = (
        case.get("alignment_reason") == "unique_sequence_interval_lacks_residue_identity_provenance"
    )
    identity_incomplete_exact_geometry = bool(
        evidence["unique_author_linked_candidate"]
        and evidence["manifest_npz_sequence_match"]
        and evidence["matrix_sequence_occurrence_count_in_candidate"] == 1
        and evidence["coordinate_matrix_match"]
        and (historical_identity_incomplete or not (exact_namespace or alternate_namespace))
        and evidence["competing_candidate_count"] == 1
    )

    stale_trim_metadata = bool(
        complete_pair and evidence["trim_metadata_available"] and not evidence["retained_interval_match"]
    )
    if original_primary in {
        "audit_chain_entity_mapping_bug",
        "audit_residue_id_convention_bug",
        "legacy_trim_metadata_inconsistency",
    }:
        refined = original_primary
    elif stale_trim_metadata:
        refined = "legacy_trim_metadata_inconsistency"
    elif identity_incomplete_exact_geometry:
        refined = "unresolved_ambiguity"
    elif evidence["author_linked_candidate_count"] > 1:
        refined = "multiple_author_linked_polymer_candidates"
    elif not evidence["unique_author_linked_candidate"]:
        refined = "unresolved_ambiguity"
    elif evidence["matrix_sequence_occurrence_count_in_candidate"] == 0:
        refined = "matrix_sequence_absent_from_author_linked_polymer"
    elif evidence["competing_candidate_count"] > 1 and evidence["coordinate_matrix_status"] != "passed":
        refined = "unresolved_ambiguity"
    elif evidence["coordinate_matrix_status"] == "failed":
        refined = "coordinate_disagreement"
    elif alternate_namespace and not exact_namespace:
        refined = "residue_id_namespace_mismatch"
    elif evidence["coordinate_matrix_status"] == "unavailable" and not exact_namespace:
        refined = "historical_missing_calpha_selection_unreproducible"
    elif evidence["coordinate_matrix_status"] == "unavailable":
        refined = "unique_sequence_pair_coordinate_unavailable"
    elif complete_pair:
        refined = "verified_sequence_geometry_pair"
    else:
        refined = "demonstrated_sequence_matrix_mismatch"

    historical_sha = source_status == "historical_sha_verified"
    if complete_pair and historical_sha:
        strict_provenance = "provenance_complete"
        eligibility = "strict_verified_pair"
    elif identity_incomplete_exact_geometry:
        strict_provenance = "residue_identity_provenance_incomplete"
        eligibility = "conditionally_verified_pair"
    elif complete_pair:
        strict_provenance = "historical_source_sha_unavailable"
        eligibility = "conditionally_verified_pair"
    elif (
        evidence["unique_author_linked_candidate"]
        and evidence["manifest_npz_sequence_match"]
        and evidence["matrix_sequence_occurrence_count_in_candidate"] == 1
        and alternate_namespace
        and evidence["coordinate_matrix_status"] != "failed"
        and evidence["competing_candidate_count"] == 1
    ):
        strict_provenance = "residue_id_namespace_requires_migration"
        eligibility = "conditionally_verified_pair"
    elif (
        evidence["unique_author_linked_candidate"]
        and evidence["manifest_npz_sequence_match"]
        and evidence["matrix_sequence_occurrence_count_in_candidate"] == 1
        and evidence["coordinate_matrix_status"] == "unavailable"
        and evidence["competing_candidate_count"] == 1
    ):
        strict_provenance = "coordinate_or_residue_provenance_incomplete"
        eligibility = "conditionally_verified_pair"
    else:
        strict_provenance = "contradictory_or_unresolved"
        eligibility = "excluded_or_unresolved"
    return {
        **evidence,
        "refined_primary_classification": refined,
        "strict_provenance_status": strict_provenance,
        "training_eligibility": eligibility,
    }


def select_unique_candidate(scores: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Select only a unique author-chain-supported evidential candidate."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        key = (str(score["label_asym_id"]), str(score["entity_id"]), str(score["model_number"]))
        grouped[key].append(score)
    matches = []
    for rows in grouped.values():
        supported = [row for row in rows if row["author_chain_match"] and row["strong_evidential_match"]]
        if supported:
            preference = {name: index for index, name in enumerate(CONVENTIONS)}
            matches.append(sorted(supported, key=lambda row: preference[row["convention"]])[0])
    if len(matches) == 1:
        return matches[0], "unique_author_chain_evidential_match"
    if len(matches) > 1:
        return None, "multiple_author_chain_candidates_remain_evidentially_plausible"
    return None, "no_author_chain_candidate_has_independent_sequence_or_coordinate_support"


def classify_case(
    *,
    original: dict[str, Any],
    scores: list[dict[str, Any]],
    source_identity_status: str,
    npz_internal_coordinate_status: str,
) -> tuple[str, list[str], dict[str, Any] | None, str]:
    """Assign exactly one primary classification and retain secondary evidence."""
    selected, selection_reason = select_unique_candidate(scores)
    secondary = [f"source_identity:{source_identity_status}", f"candidate_selection:{selection_reason}"]
    if source_identity_status == "state_identity_changed":
        return "raw_source_version_unverifiable", secondary, selected, selection_reason
    if npz_internal_coordinate_status == "failed":
        return "audit_coordinate_reconstruction_bug", secondary, selected, selection_reason
    if selected is None:
        any_sequence = any(row["sequence_match"] for row in scores)
        any_coordinates = any(row["coordinate_status"] == "passed" for row in scores)
        if not any_sequence and not any_coordinates and source_identity_status != "historical_sha_verified":
            return "raw_source_version_unverifiable", secondary, None, selection_reason
        return "unresolved_ambiguity", secondary, None, selection_reason
    if str(selected["label_asym_id"]) != str(original.get("raw_label_asym_id")):
        return "audit_chain_entity_mapping_bug", secondary, selected, selection_reason
    if selected["convention"] != "auth_seq_id_insertion":
        return "audit_residue_id_convention_bug", secondary, selected, selection_reason
    if selected["coordinate_status"] == "unavailable":
        return "coordinate_provenance_unavailable", secondary, selected, selection_reason
    if selected["sequence_match"] and selected["coordinate_status"] == "passed":
        if original.get("alignment_reason") == "residue_ids_match_but_recorded_terminal_trim_metadata_disagrees":
            return "legacy_trim_metadata_inconsistency", secondary, selected, selection_reason
        if not selected["trim_metadata_compatible"] and bool(original.get("terminal_trimming_applied")):
            return "historical_preprocessing_policy_difference", secondary, selected, selection_reason
    if any(row["sequence_match"] for row in scores):
        return "historical_preprocessing_policy_difference", secondary, selected, selection_reason
    if source_identity_status == "historical_sha_verified":
        return "genuine_sequence_matrix_mismatch", secondary, selected, selection_reason
    return "raw_source_version_unverifiable", secondary, selected, selection_reason


def inspect_npz_forensics(path: str | Path) -> dict[str, Any]:
    """Load one bounded NPZ and distinguish physical from normalized geometry."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        coordinates = np.asarray(data["ca_coordinates"], dtype=np.float32)
        matrix = np.asarray(data["distance_matrix"], dtype=np.float32)
        coordinate_matrix = compute_distance_matrix(coordinates)
        rmse, maximum = matrix_errors(matrix, coordinate_matrix)
        return {
            "sequence": str(data["sequence"]),
            "residue_ids": [str(value) for value in data["residue_ids"].tolist()],
            "chain_id": str(data["chain_id"]),
            "pdb_id": str(data["pdb_id"]),
            "metadata": metadata,
            "distance_matrix": matrix,
            "coordinate_matrix_rmse_angstrom": rmse,
            "coordinate_matrix_max_abs_error_angstrom": maximum,
            "coordinate_matrix_status": "passed" if maximum is not None and maximum <= 1e-4 else "failed",
            "distance_matrix_semantics": "physical_calpha_angstrom",
        }
