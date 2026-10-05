"""Bounded, read-only source audit for E006 rich backbone geometry."""

from __future__ import annotations

import csv
import gzip
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.constants import STANDARD_AA3_TO_1

AUDIT_SCHEMA_VERSION = "e006_rich_geometry_source_audit_v6"
SAMPLER_VERSION = "e006_stratified_sha256_v1"
CHIRALITY_CONVENTION_VERSION = "e006_native_cb_ca_n_c_scalar_triple_v2"
PSEUDO_CB_CONVENTION_VERSION = "proteinmpnn_pseudo_cb_058273431_v1"
TARGET_MAPPING_POLICY_VERSION = "e006_verified_pairing_target_mapping_v2"
CANONICALIZATION_POLICY_VERSION = "e006_authoritative_ccd_parent_v1"
COORDINATE_ANCHOR_POLICY_VERSION = "e006_npz_calpha_provenance_anchor_v1"
BACKBONE_ATOMS = ("N", "CA", "C", "O", "CB")
EXPLICIT_D_RESIDUES = {"DAL": "A"}
TARGET_RESIDUE_CLASSIFICATIONS = frozenset(
    {
        "canonical_target_residue",
        "mapped_modified_target_residue",
        "masked_target_missing_atoms",
        "ignored_unmapped_source_component",
        "excluded_unresolved_target_modification",
        "excluded_identity_mapping_contradiction",
        "excluded_stereochemical_contradiction",
        "excluded_identity_mapping_unavailable",
        "excluded_invalid_residue_id_serialization",
        "excluded_residue_id_namespace_contradiction",
        "excluded_identifier_locator_unavailable",
        "excluded_npz_residue_order_contradiction",
        "excluded_npz_internal_geometry_contradiction",
        "excluded_npz_calpha_anchor_unavailable",
        "excluded_npz_calpha_anchor_ambiguous",
        "excluded_source_npz_coordinate_contradiction",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if platform.system() != "Darwin" else value / 2**20


def enforce_rss_limit(maximum_rss_mib: float) -> None:
    current = _rss_mib()
    peak = _peak_rss_mib()
    if current > maximum_rss_mib or peak > maximum_rss_mib:
        raise MemoryError(
            f"E006 source audit RSS limit exceeded: limit={maximum_rss_mib:.1f} MiB, "
            f"current={current:.1f} MiB, peak={peak:.1f} MiB"
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    columns = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def resolve_within_roots(path: str | Path, roots: list[str | Path]) -> Path:
    """Resolve an input locator and reject traversal outside configured roots."""
    candidate = Path(path).expanduser().resolve(strict=True)
    allowed = [Path(root).expanduser().resolve(strict=True) for root in roots]
    if not any(candidate.is_relative_to(root) for root in allowed):
        raise ValueError(f"input_path_outside_protected_roots:{candidate}")
    if not candidate.is_file():
        raise ValueError(f"input_locator_is_not_a_file:{candidate}")
    return candidate


def parquet_schema_inventory(name: str, path: str | Path) -> dict[str, Any]:
    """Inspect Arrow and row-group metadata without materializing table contents."""
    source = Path(path)
    dataset = ds.dataset(str(source), format="parquet")
    files = sorted(Path(fragment.path).resolve() for fragment in dataset.get_fragments())
    fragments = []
    total_rows = 0
    for file_path in files:
        metadata = pq.ParquetFile(file_path).metadata
        rows = int(metadata.num_rows)
        total_rows += rows
        fragments.append(
            {
                "path": str(file_path),
                "row_count": rows,
                "row_group_count": int(metadata.num_row_groups),
                "size_bytes": file_path.stat().st_size,
            }
        )
    return {
        "name": name,
        "path": str(source),
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in dataset.schema
        ],
        "row_count": total_rows,
        "fragment_count": len(fragments),
        "fragments": fragments,
    }


def _parquet_files(inventory: dict[str, Any]) -> list[Path]:
    return [Path(fragment["path"]) for fragment in inventory.get("fragments", [])]


def _length_bin(length: int) -> str:
    for limit in (64, 128, 256, 384, 500):
        if length <= limit:
            return f"le_{limit}"
    return "above_500"


def scan_stratified_partition(
    path: str | Path,
    *,
    split: str,
    target: int,
    seed: int,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan projected metadata while retaining at most target rows per stratum."""
    if not 1 <= batch_size <= 4096 or target < 1:
        raise ValueError("invalid bounded panel scanner settings")
    dataset = ds.dataset(str(path), format="parquet")
    names = set(dataset.schema.names)
    method = next((name for name in ("experimental_method", "method") if name in names), None)
    pairing = next((name for name in ("pairing_classification", "v3_pairing_classification") if name in names), None)
    required = {"sample_id", "sequence", "sequence_length", "matrix_path"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"pairing partition is missing columns: {', '.join(missing)}")
    optional = {
        "pdb_id",
        "chain_id",
        "auth_asym_id",
        "label_asym_id",
        "entity_id",
        "model_id",
        "model_number",
        "source_file",
        "source_sha256",
        "matrix_sha256",
        "schema_version",
        "sequence_sha256",
        "modified_residue_mapping_version",
        "residue_ids",
        "insertion_codes",
        "selected_residue_id_convention",
        "auth_residue_ids_match",
        "label_residue_ids_match",
        "zero_based_positions_match",
        "one_based_positions_match",
        "auth_residue_ids",
        "label_residue_ids",
        "zero_based_residue_ids",
        "one_based_residue_ids",
        "practical_training_eligibility",
    }
    columns = sorted(required | (optional & names) | {name for name in (method, pairing) if name})
    heaps: dict[tuple[str, str, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    available = Counter()
    length_statistics: dict[str, Counter[str]] = defaultdict(Counter)
    seen: dict[str, str] = {}
    rejected = Counter()
    for batch in dataset.scanner(columns=columns, batch_size=batch_size, use_threads=False).to_batches():
        for source_row in batch.to_pylist():
            sample_id = str(source_row.get("sample_id") or "").strip()
            if not sample_id:
                rejected["missing_sample_id"] += 1
                continue
            try:
                length = int(source_row["sequence_length"])
            except (TypeError, ValueError):
                rejected["invalid_length"] += 1
                continue
            if not 1 <= length <= 500:
                rejected["length_outside_1_500"] += 1
                continue
            if str(source_row.get("practical_training_eligibility") or "") == "excluded_or_unresolved":
                rejected["ineligible"] += 1
                continue
            row = dict(source_row)
            row.update(
                sample_id=sample_id,
                split=split,
                pairing_source_artifact=str(Path(path)),
                experimental_method=str(row.get(method) or "unknown"),
                pairing_classification=str(row.get(pairing) or "unknown"),
                length_bin=_length_bin(length),
            )
            row_hash = _canonical_hash(row)
            if sample_id in seen:
                rejected["duplicate_sample_id"] += 1
                if seen[sample_id] != row_hash:
                    raise ValueError(f"conflicting_duplicate_sample_id:{sample_id}")
                continue
            seen[sample_id] = row_hash
            stratum = (row["length_bin"], row["experimental_method"], row["pairing_classification"])
            available[stratum] += 1
            length_statistics[row["length_bin"]]["sample_count"] += 1
            length_statistics[row["length_bin"]]["residue_count"] += length
            length_statistics[row["length_bin"]]["squared_length_sum"] += length * length
            rank = int(hashlib.sha256(f"{SAMPLER_VERSION}:{seed}:{sample_id}".encode()).hexdigest(), 16)
            heap = heaps[stratum]
            entry = (-rank, sample_id, row)
            if len(heap) < target:
                heapq.heappush(heap, entry)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    retained = [entry[2] for heap in heaps.values() for entry in heap]
    selected, diagnostics = stratified_select(
        retained,
        target=target,
        seed=seed,
        stratum_fields=("length_bin", "experimental_method", "pairing_classification"),
    )
    selected_counts = Counter(
        (row["length_bin"], row["experimental_method"], row["pairing_classification"]) for row in selected
    )
    diagnostics.update(
        filtered_candidate_count=sum(available.values()),
        rejected_counts=dict(rejected),
        per_stratum=[
            {
                "length_bin": key[0],
                "experimental_method": key[1],
                "pairing_classification": key[2],
                "available_count": available[key],
                "selected_count": selected_counts[key],
            }
            for key in sorted(available)
        ],
        length_statistics={key: dict(values) for key, values in sorted(length_statistics.items())},
    )
    return selected, diagnostics


def _scan_matching_rows(
    path: str | Path, sample_ids: set[str], desired_columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    dataset = ds.dataset(str(path), format="parquet")
    if "sample_id" not in dataset.schema.names:
        return []
    columns = [name for name in desired_columns if name in dataset.schema.names]
    if "sample_id" not in columns:
        columns.insert(0, "sample_id")
    rows = []
    scanner = dataset.scanner(
        columns=columns,
        filter=ds.field("sample_id").isin(sorted(sample_ids)),
        batch_size=4096,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        rows.extend(batch.to_pylist())
    return rows


def enrich_source_locators(
    rows: list[dict[str, Any]], *, processed_manifest: str | Path, audit_provenance: str | Path
) -> list[dict[str, Any]]:
    """Resolve selected sample locators from manifests and compact raw evidence."""
    sample_ids = {str(row["sample_id"]) for row in rows}
    fields = (
        "sample_id",
        "pdb_id",
        "chain_id",
        "auth_asym_id",
        "label_asym_id",
        "entity_id",
        "model_id",
        "model_number",
        "source_file",
        "source_sha256",
        "matrix_path",
        "matrix_sha256",
        "parser_backend",
        "missing_calpha_policy",
        "terminal_trimming_applied",
        "trimmed_n_terminal_residues",
        "trimmed_c_terminal_residues",
        "preprocessing_config_sha256",
        "sequence_sha256",
        "modified_residue_mapping_version",
        "residue_ids",
        "insertion_codes",
        "selected_residue_id_convention",
        "auth_residue_ids_match",
        "label_residue_ids_match",
        "zero_based_positions_match",
        "one_based_positions_match",
        "auth_residue_ids",
        "label_residue_ids",
        "zero_based_residue_ids",
        "one_based_residue_ids",
    )
    evidence: dict[str, dict[str, Any]] = defaultdict(dict)
    audit_identifier_records: dict[str, dict[str, Any]] = {}
    for record in _scan_matching_rows(processed_manifest, sample_ids, fields):
        evidence[str(record["sample_id"])].update({key: value for key, value in record.items() if value is not None})
    audit_root = Path(audit_provenance)
    alignment_path = audit_root / "tables" / "matrix_pair_alignments"
    source_ids = set()
    if alignment_path.is_dir():
        audit_fields = (
            *fields,
            "source_id",
            "selected_model_number",
            "selected_auth_asym_id",
            "selected_label_asym_id",
            "modified_residue_mapping_version",
        )
        for record in _scan_matching_rows(alignment_path, sample_ids, audit_fields):
            sample_id = str(record["sample_id"])
            if sample_id in audit_identifier_records:
                raise ValueError(f"duplicate_compact_audit_source_record:{sample_id}")
            audit_identifier_records[sample_id] = dict(record)
            evidence[sample_id].update({key: value for key, value in record.items() if value is not None})
            if record.get("source_id"):
                source_ids.add(str(record["source_id"]))
    source_identity_path = audit_root / "tables" / "source_identity"
    identities: dict[str, dict[str, Any]] = {}
    if source_ids and source_identity_path.is_dir():
        dataset = ds.dataset(str(source_identity_path), format="parquet")
        columns = [name for name in ("source_id", "source_file", "source_sha256") if name in dataset.schema.names]
        scanner = dataset.scanner(
            columns=columns,
            filter=ds.field("source_id").isin(sorted(source_ids)),
            batch_size=4096,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            for record in batch.to_pylist():
                identities[str(record["source_id"])] = record
    output = []
    for row in rows:
        sample_id = str(row["sample_id"])
        merged = {
            **evidence.get(sample_id, {}),
            **{key: value for key, value in row.items() if value is not None},
        }
        merged["_identifier_audit_source_record"] = audit_identifier_records.get(sample_id)
        merged["_identifier_audit_source_artifact"] = str(alignment_path)
        source_id = merged.get("source_id")
        if source_id and str(source_id) in identities:
            for key, value in identities[str(source_id)].items():
                merged.setdefault(key, value)
        if not merged.get("model_number"):
            merged["model_number"] = merged.get("selected_model_number") or merged.get("model_id") or 1
        if not merged.get("chain_id"):
            merged["chain_id"] = (
                merged.get("selected_auth_asym_id")
                or merged.get("auth_asym_id")
                or merged.get("selected_label_asym_id")
                or merged.get("label_asym_id")
            )
        provenance = {
            key: merged.get(key)
            for key in (
                "parser_backend",
                "missing_calpha_policy",
                "terminal_trimming_applied",
                "trimmed_n_terminal_residues",
                "trimmed_c_terminal_residues",
                "preprocessing_config_sha256",
                "source_sha256",
                "matrix_sha256",
            )
            if key in merged
        }
        merged["matrix_generation_provenance_json"] = json.dumps(provenance, sort_keys=True)
        output.append(merged)
    return output


def raw_audit_schema_inventory(path: str | Path) -> dict[str, Any]:
    """Inventory compact raw-audit tables and their source-locator fields."""
    root = Path(path)
    tables = {}
    table_root = root / "tables"
    if table_root.is_dir():
        for directory in sorted(item for item in table_root.iterdir() if item.is_dir()):
            parquet_files = list(directory.glob("*.parquet"))
            if parquet_files:
                tables[directory.name] = parquet_schema_inventory(directory.name, directory)
    protocols = []
    for name in ("sequence_readiness_protocol.json", "run_config.json", "schema.json"):
        candidate = root / name
        if candidate.is_file():
            protocols.append(str(candidate.resolve()))
    locator_names = {
        "source_file",
        "source_id",
        "model_number",
        "model_id",
        "chain_id",
        "auth_asym_id",
        "label_asym_id",
        "entity_id",
        "residue_number",
        "insertion_code",
        "alternate_location",
        "sequence_mapping",
    }
    available = sorted(
        {field["name"] for table in tables.values() for field in table["schema"] if field["name"] in locator_names}
    )
    return {"path": str(root), "tables": tables, "protocol_files": protocols, "available_locator_fields": available}


@dataclass(frozen=True)
class BackboneResidue:
    residue_name: str
    one_letter: str | None
    residue_number: str
    insertion_code: str
    atoms: dict[str, np.ndarray]
    alternate_location_count: int = 0
    selected_altlocs: dict[str, str | None] | None = None
    chirality_class: str | None = None
    label_sequence_id: str | None = None
    auth_sequence_id: str | None = None
    entity_id: str | None = None
    polymer_member: bool = True
    authoritative_parent_comp_id: str | None = None
    authoritative_parent_source: str | None = None
    atom_candidates: dict[str, tuple[tuple[str | None, float | None, np.ndarray], ...]] | None = None


@dataclass(frozen=True)
class NpzGeometryEvidence:
    residue_ids: tuple[str, ...]
    ca_coordinates: np.ndarray
    residue_mask: np.ndarray
    distance_matrix: np.ndarray
    metadata: dict[str, Any]
    metadata_sha256: str
    inventory: dict[str, list[int]]
    sample_id: str | None
    pdb_id: str | None
    chain_id: str | None
    sequence: str | None
    internal_consistent: bool
    internal_rmse_angstrom: float
    internal_maximum_error_angstrom: float
    validation_errors: tuple[str, ...]


def select_atom_alternate_locations(
    candidates: dict[str, list[tuple[str | None, float | None, np.ndarray]]],
) -> tuple[dict[str, np.ndarray], dict[str, str | None], int]:
    """Select one deterministic atom conformer by occupancy, blank, then A."""
    selected: dict[str, np.ndarray] = {}
    selected_altlocs: dict[str, str | None] = {}
    alternate_count = 0
    for atom_name, values in candidates.items():
        if not values:
            continue
        alternate_count += max(0, len(values) - 1)

        def key(item: tuple[str | None, float | None, np.ndarray]) -> tuple[float, int, str]:
            altloc, occupancy, _ = item
            finite_occupancy = float(occupancy) if occupancy is not None and math.isfinite(occupancy) else -1.0
            preference = 0 if not altloc else 1 if altloc == "A" else 2
            return (-finite_occupancy, preference, altloc or "")

        altloc, _, coordinate = sorted(values, key=key)[0]
        selected[atom_name] = np.asarray(coordinate, dtype=np.float64)
        selected_altlocs[atom_name] = altloc
    return selected, selected_altlocs, alternate_count


def _npz_text(value: Any) -> str:
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError("expected_scalar_npz_value")
    item = scalar.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_npz_geometry_evidence(
    path: str | Path,
    *,
    internal_tolerance_angstrom: float = 1e-4,
) -> NpzGeometryEvidence:
    """Read and internally validate one immutable processed geometry sample."""
    required = {"residue_ids", "ca_coordinates", "residue_mask", "distance_matrix", "metadata"}
    errors: list[str] = []
    with np.load(Path(path), allow_pickle=False) as sample:
        inventory = {key: list(np.asarray(sample[key]).shape) for key in sample.files}
        missing = sorted(required - set(sample.files))
        errors.extend(f"missing_npz_field:{field}" for field in missing)
        residue_values = np.asarray(sample["residue_ids"]) if "residue_ids" in sample else np.asarray([])
        coordinates = (
            np.asarray(sample["ca_coordinates"], dtype=np.float64)
            if "ca_coordinates" in sample
            else np.empty((0, 3), dtype=np.float64)
        )
        mask_raw = np.asarray(sample["residue_mask"]) if "residue_mask" in sample else np.asarray([])
        matrix = (
            np.asarray(sample["distance_matrix"], dtype=np.float64)
            if "distance_matrix" in sample
            else np.empty((0, 0), dtype=np.float64)
        )
        metadata_raw = np.asarray(sample["metadata"]) if "metadata" in sample else np.asarray("")
        identities = {}
        for field in ("sample_id", "pdb_id", "chain_id", "sequence"):
            if field in sample:
                try:
                    identities[field] = _npz_text(sample[field])
                except (TypeError, ValueError, UnicodeDecodeError):
                    errors.append(f"invalid_npz_scalar:{field}")

    residue_ids: list[str] = []
    if residue_values.ndim != 1:
        errors.append("invalid_npz_residue_ids_shape")
    else:
        try:
            residue_ids = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in residue_values]
        except UnicodeDecodeError:
            errors.append("invalid_npz_residue_id_encoding")
        if any(not item for item in residue_ids):
            errors.append("empty_npz_residue_id")
        if len(residue_ids) != len(set(residue_ids)):
            errors.append("duplicate_npz_residue_id")
    length = len(residue_ids)
    if coordinates.shape != (length, 3):
        errors.append("invalid_npz_calpha_shape")
    if mask_raw.shape != (length,):
        errors.append("invalid_npz_residue_mask_shape")
        residue_mask = np.zeros(length, dtype=np.bool_)
    else:
        if mask_raw.dtype != np.bool_ and not np.isin(mask_raw, [0, 1]).all():
            errors.append("invalid_npz_residue_mask_values")
        residue_mask = mask_raw.astype(np.bool_, copy=False)
    if matrix.shape != (length, length):
        errors.append("invalid_npz_distance_matrix_shape")
    if not np.isfinite(coordinates).all():
        errors.append("nonfinite_npz_calpha_coordinates")
    if not np.isfinite(matrix).all():
        errors.append("nonfinite_npz_distance_matrix")

    metadata_text = ""
    metadata: dict[str, Any] = {}
    try:
        metadata_text = _npz_text(metadata_raw)
        parsed_metadata = json.loads(metadata_text)
        if not isinstance(parsed_metadata, dict):
            errors.append("invalid_npz_metadata_type")
        else:
            metadata = parsed_metadata
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        errors.append("invalid_npz_metadata")
    metadata_sha256 = hashlib.sha256(metadata_text.encode("utf-8")).hexdigest()
    metadata_residue_ids = metadata.get("residue_ids")
    if metadata_residue_ids is not None and (
        not isinstance(metadata_residue_ids, list)
        or tuple(str(item) for item in metadata_residue_ids) != tuple(residue_ids)
    ):
        errors.append("npz_metadata_residue_ids_mismatch")

    rmse = math.nan
    maximum = math.nan
    if (
        coordinates.shape == (length, 3)
        and matrix.shape == (length, length)
        and np.isfinite(coordinates).all()
        and np.isfinite(matrix).all()
    ):
        recomputed = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
        error = recomputed - matrix
        rmse = float(np.sqrt(np.mean(error**2))) if error.size else 0.0
        maximum = float(np.max(np.abs(error))) if error.size else 0.0
        if rmse > internal_tolerance_angstrom or maximum > internal_tolerance_angstrom:
            errors.append("npz_internal_distance_matrix_mismatch")
    return NpzGeometryEvidence(
        residue_ids=tuple(residue_ids),
        ca_coordinates=coordinates,
        residue_mask=residue_mask,
        distance_matrix=matrix,
        metadata=metadata,
        metadata_sha256=metadata_sha256,
        inventory=inventory,
        sample_id=identities.get("sample_id"),
        pdb_id=identities.get("pdb_id"),
        chain_id=identities.get("chain_id"),
        sequence=identities.get("sequence"),
        internal_consistent=not errors,
        internal_rmse_angstrom=rmse,
        internal_maximum_error_angstrom=maximum,
        validation_errors=tuple(sorted(set(errors))),
    )


def _residue_atom_candidates(
    residue: BackboneResidue,
) -> dict[str, tuple[tuple[str | None, float | None, np.ndarray], ...]]:
    if residue.atom_candidates is not None:
        return residue.atom_candidates
    return {
        atom: (((residue.selected_altlocs or {}).get(atom), None, np.asarray(coordinate, dtype=np.float64)),)
        for atom, coordinate in residue.atoms.items()
    }


def reconcile_npz_calpha_anchors(
    residues: list[BackboneResidue],
    *,
    resolved_target_residue_ids: list[str],
    npz: NpzGeometryEvidence,
    anchor_tolerance_angstrom: float = 1e-4,
    scientific_tolerance_angstrom: float = 0.05,
    maximum_examples: int = 100,
    expected_sample_id: str | None = None,
    expected_pdb_id: str | None = None,
    expected_chain_id: str | None = None,
    expected_sequence: str | None = None,
    expected_model_number: int | None = None,
    expected_source_path: str | Path | None = None,
    recorded_source_sha256: str | None = None,
    actual_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Anchor source conformers to immutable NPZ C-alpha coordinates."""
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def fail(category: str, evidence: dict[str, Any]) -> None:
        if len(failures[category]) < maximum_examples:
            failures[category].append(evidence)

    metadata_contradictions = []
    identity_expectations = {
        "sample_id": (npz.sample_id, expected_sample_id),
        "pdb_id": (npz.pdb_id, expected_pdb_id),
        "chain_id": (npz.chain_id, expected_chain_id),
        "sequence": (npz.sequence, expected_sequence),
    }
    for field, (observed, expected) in identity_expectations.items():
        if observed is not None and expected is not None and str(observed) != str(expected):
            metadata_contradictions.append(f"npz_{field}_mismatch")
    metadata_model = npz.metadata.get("model_number")
    if metadata_model is not None and expected_model_number is not None:
        try:
            if int(metadata_model) != int(expected_model_number):
                metadata_contradictions.append("npz_model_number_mismatch")
        except (TypeError, ValueError):
            metadata_contradictions.append("invalid_npz_model_number")
    metadata_source = npz.metadata.get("source_file")
    if metadata_source and expected_source_path:
        if Path(str(metadata_source)).resolve() != Path(expected_source_path).resolve():
            metadata_contradictions.append("npz_source_path_mismatch")
    metadata_source_hash = npz.metadata.get("source_sha256")
    expected_hashes = [value for value in (recorded_source_sha256, metadata_source_hash) if value]
    if actual_source_sha256 and any(str(value) != actual_source_sha256 for value in expected_hashes):
        metadata_contradictions.append("source_sha256_mismatch")
    for reason in npz.validation_errors:
        fail("npz_internal_geometry_contradiction", {"reason": reason})
    for reason in metadata_contradictions:
        fail("source_npz_coordinate_contradiction", {"reason": reason})

    residue_order_agreement = tuple(resolved_target_residue_ids) == npz.residue_ids
    if not residue_order_agreement:
        fail(
            "npz_residue_order_contradiction",
            {
                "resolved_target_residue_ids": resolved_target_residue_ids[:maximum_examples],
                "npz_residue_ids": list(npz.residue_ids[:maximum_examples]),
            },
        )

    anchored: list[BackboneResidue] = []
    candidate_counts = [len(_residue_atom_candidates(residue).get("CA", ())) for residue in residues]
    selected_conformers: list[str | None] = []
    coordinate_errors: list[float] = []
    unavailable_count = 0
    ambiguous_count = 0
    coordinate_contradiction_count = 0
    blank_fallback_count = 0
    conflicting_nonblank_count = 0
    if residue_order_agreement and npz.internal_consistent and len(residues) == len(npz.residue_ids):
        for index, (residue, npz_coordinate, residue_valid) in enumerate(
            zip(residues, npz.ca_coordinates, npz.residue_mask, strict=True)
        ):
            residue_id = resolved_target_residue_ids[index]
            candidates_by_atom = _residue_atom_candidates(residue)
            ca_candidates = list(candidates_by_atom.get("CA", ()))
            if not residue_valid or not ca_candidates:
                unavailable_count += 1
                fail("npz_calpha_anchor_unavailable", {"residue_id": residue_id, "candidate_count": len(ca_candidates)})
                continue
            distances = [float(np.linalg.norm(np.asarray(item[2]) - npz_coordinate)) for item in ca_candidates]
            matches = [
                candidate_index
                for candidate_index, distance in enumerate(distances)
                if distance <= anchor_tolerance_angstrom
            ]
            if not matches:
                coordinate_contradiction_count += 1
                coordinate_errors.append(min(distances))
                fail(
                    "source_npz_coordinate_contradiction",
                    {"residue_id": residue_id, "nearest_error_angstrom": min(distances)},
                )
                continue
            if len(matches) != 1:
                ambiguous_count += 1
                coordinate_errors.append(min(distances))
                fail(
                    "npz_calpha_anchor_ambiguous",
                    {"residue_id": residue_id, "matching_candidate_count": len(matches)},
                )
                continue
            selected_ca = ca_candidates[matches[0]]
            selected_altloc = selected_ca[0]
            conflicting_nonblank_count += sum(bool(item[0]) and item[0] != selected_altloc for item in ca_candidates)
            atoms = {"CA": np.asarray(selected_ca[2], dtype=np.float64)}
            selected_altlocs: dict[str, str | None] = {"CA": selected_altloc}
            for atom_name in BACKBONE_ATOMS:
                if atom_name == "CA":
                    continue
                atom_candidates = list(candidates_by_atom.get(atom_name, ()))
                conflicting_nonblank_count += sum(
                    bool(item[0]) and item[0] != selected_altloc for item in atom_candidates
                )
                exact = [item for item in atom_candidates if item[0] == selected_altloc]
                blank = [item for item in atom_candidates if item[0] is None]
                compatible = exact if exact else blank if selected_altloc is not None else []
                if len(compatible) != 1:
                    continue
                selected_atom = compatible[0]
                atoms[atom_name] = np.asarray(selected_atom[2], dtype=np.float64)
                selected_altlocs[atom_name] = selected_atom[0]
                if selected_altloc is not None and selected_atom[0] is None:
                    blank_fallback_count += 1
            anchored.append(replace(residue, atoms=atoms, selected_altlocs=selected_altlocs))
            selected_conformers.append(selected_altloc)
            coordinate_errors.append(distances[matches[0]])
    elif residue_order_agreement and npz.internal_consistent and len(residues) != len(npz.residue_ids):
        fail(
            "npz_residue_order_contradiction",
            {"resolved_residue_count": len(residues), "npz_residue_count": len(npz.residue_ids)},
        )
        residue_order_agreement = False

    exclusion_reason = None
    if not npz.internal_consistent:
        exclusion_reason = "excluded_npz_internal_geometry_contradiction"
    elif not residue_order_agreement:
        exclusion_reason = "excluded_npz_residue_order_contradiction"
    elif unavailable_count:
        exclusion_reason = "excluded_npz_calpha_anchor_unavailable"
    elif ambiguous_count:
        exclusion_reason = "excluded_npz_calpha_anchor_ambiguous"
    elif coordinate_contradiction_count or metadata_contradictions:
        exclusion_reason = "excluded_source_npz_coordinate_contradiction"
    fully_anchored = exclusion_reason is None and len(anchored) == len(residues) == len(npz.residue_ids)
    source_rmse = float(np.sqrt(np.mean(np.square(coordinate_errors)))) if coordinate_errors else math.nan
    source_maximum = max(coordinate_errors, default=math.nan)
    source_consistent = bool(
        fully_anchored
        and math.isfinite(source_rmse)
        and math.isfinite(source_maximum)
        and source_rmse <= scientific_tolerance_angstrom
        and source_maximum <= scientific_tolerance_angstrom
    )
    source_hash_agreement = not expected_hashes or bool(
        actual_source_sha256 and all(str(value) == actual_source_sha256 for value in expected_hashes)
    )
    return {
        "_anchored_residues": anchored if fully_anchored else [],
        "coordinate_anchor_exclusion_reason": exclusion_reason,
        "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
        "npz_residue_id_order_agreement": residue_order_agreement,
        "npz_residue_ids_json": json.dumps(npz.residue_ids),
        "npz_internal_matrix_consistency": npz.internal_consistent,
        "npz_internal_validation_errors_json": json.dumps(npz.validation_errors),
        "npz_internal_matrix_rmse_angstrom": npz.internal_rmse_angstrom,
        "npz_internal_matrix_maximum_error_angstrom": npz.internal_maximum_error_angstrom,
        "ca_candidate_counts_json": json.dumps(candidate_counts),
        "uniquely_anchored_residue_count": len(anchored),
        "unavailable_calpha_anchor_count": unavailable_count,
        "ambiguous_calpha_anchor_count": ambiguous_count,
        "source_npz_coordinate_contradiction_count": coordinate_contradiction_count,
        "source_to_npz_calpha_coordinate_rmse_angstrom": source_rmse,
        "source_to_npz_calpha_coordinate_maximum_error_angstrom": source_maximum,
        "source_to_npz_calpha_anchor_consistency": source_consistent,
        "selected_calpha_conformers_json": json.dumps(selected_conformers),
        "blank_altloc_fallback_count": blank_fallback_count,
        "conflicting_nonblank_conformer_count": conflicting_nonblank_count,
        "coordinate_anchor_failure_examples_json": json.dumps(failures, sort_keys=True),
        "npz_metadata_sha256": npz.metadata_sha256,
        "npz_metadata_validation_errors_json": json.dumps(metadata_contradictions),
        "npz_metadata_validation_error_count": len(metadata_contradictions),
        "recorded_source_sha256": recorded_source_sha256,
        "source_sha256_agreement": source_hash_agreement,
    }


def pseudo_cb_coordinate(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Construct ProteinMPNN pseudo-CB: b=CA-N, c=C-CA, a=b cross c.

    The fixed coefficients are independent of the native CB used to assess
    agreement. They are the convention used by ProteinMPNN-style featurizers.
    """
    b = np.asarray(ca, dtype=np.float64) - np.asarray(n, dtype=np.float64)
    c_vector = np.asarray(c, dtype=np.float64) - np.asarray(ca, dtype=np.float64)
    a = np.cross(b, c_vector)
    if min(np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c_vector)) < 1e-8:
        raise ValueError("degenerate_backbone_for_pseudo_cb")
    return np.asarray(ca, dtype=np.float64) - 0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vector


def local_frame(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a residue-local right-handed frame and its determinant."""
    x = np.asarray(c, dtype=np.float64) - np.asarray(ca, dtype=np.float64)
    helper = np.asarray(n, dtype=np.float64) - np.asarray(ca, dtype=np.float64)
    x_norm = np.linalg.norm(x)
    z = np.cross(x, helper)
    z_norm = np.linalg.norm(z)
    if min(x_norm, z_norm) < 1e-8:
        raise ValueError("degenerate_backbone_frame")
    x /= x_norm
    z /= z_norm
    y = np.cross(z, x)
    frame = np.stack((x, y, z), axis=1)
    return frame, float(np.linalg.det(frame))


def frame_quality(frame: np.ndarray, *, tolerance: float = 1e-5) -> dict[str, float | bool]:
    """Validate a constructed frame numerically; this is not chirality evidence."""
    value = np.asarray(frame, dtype=np.float64)
    finite = bool(value.shape == (3, 3) and np.isfinite(value).all())
    if not finite:
        return {
            "frame_valid": False,
            "frame_determinant": math.nan,
            "frame_orthogonality_error": math.inf,
            "frame_unit_norm_error": math.inf,
        }
    gram = value.T @ value
    determinant = float(np.linalg.det(value))
    orthogonality_error = float(np.max(np.abs(gram - np.diag(np.diag(gram)))))
    unit_norm_error = float(np.max(np.abs(np.diag(gram) - 1.0)))
    valid = bool(
        orthogonality_error <= tolerance and unit_norm_error <= tolerance and abs(determinant - 1.0) <= tolerance
    )
    return {
        "frame_valid": valid,
        "frame_determinant": determinant,
        "frame_orthogonality_error": orthogonality_error,
        "frame_unit_norm_error": unit_norm_error,
    }


def frame_is_right_handed(frame: np.ndarray, *, tolerance: float = 1e-5) -> bool:
    return bool(frame_quality(frame, tolerance=tolerance)["frame_valid"])


def native_chirality_signed_volume(n: np.ndarray, ca: np.ndarray, c: np.ndarray, cb: np.ndarray) -> float:
    """Return ``(CB-CA) dot ((CA-N) cross (CA-C))``.

    With this exact vector order, canonical L amino acids have positive
    volume and D amino acids have negative volume. Proper rigid transforms
    preserve the value; a reflection reverses its sign.
    """
    n_value = np.asarray(n, dtype=np.float64)
    ca_value = np.asarray(ca, dtype=np.float64)
    c_value = np.asarray(c, dtype=np.float64)
    cb_value = np.asarray(cb, dtype=np.float64)
    values = np.stack((n_value, ca_value, c_value, cb_value))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite_native_chirality_coordinates")
    return float(np.dot(cb_value - ca_value, np.cross(ca_value - n_value, ca_value - c_value)))


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if min(left_norm, right_norm) < 1e-8:
        raise ValueError("degenerate_pseudo_cb_angle")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _chirality_class(residue: BackboneResidue) -> str:
    if residue.chirality_class is not None:
        return residue.chirality_class
    if residue.one_letter == "G":
        return "glycine_achiral"
    if residue.one_letter in set(STANDARD_AA3_TO_1.values()):
        return "canonical_l"
    return "noncanonical_unknown"


def residue_geometry_diagnostics(
    residue: BackboneResidue,
    *,
    pseudo_cb_maximum_distance_angstrom: float = 0.5,
    pseudo_cb_maximum_angle_degrees: float = 20.0,
    signed_volume_epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Audit frame, native chirality, and pseudo-CB as independent concepts."""
    residue_id = f"{residue.residue_number}{residue.insertion_code}"
    result: dict[str, Any] = {
        "residue_id": residue_id,
        "residue_name": residue.residue_name,
        "chirality_class": _chirality_class(residue),
        "frame_valid": False,
        "frame_determinant": math.nan,
        "frame_orthogonality_error": math.nan,
        "frame_unit_norm_error": math.nan,
        "native_chirality_available": False,
        "native_chirality_signed_volume": math.nan,
        "native_chirality_valid": None,
        "pseudo_cb_available": False,
        "pseudo_cb_native_distance_angstrom": math.nan,
        "pseudo_cb_native_angle_degrees": math.nan,
        "pseudo_cb_agreement": None,
        "chirality_reason": "backbone_atoms_unavailable",
        "frame_convention": "x=C-CA; z=normalize(x_cross_(N-CA)); y=z_cross_x",
        "chirality_convention_version": CHIRALITY_CONVENTION_VERSION,
        "pseudo_cb_convention_version": PSEUDO_CB_CONVENTION_VERSION,
    }
    if not all(atom in residue.atoms for atom in ("N", "CA", "C")):
        return result
    try:
        frame, _ = local_frame(residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"])
        result.update(frame_quality(frame))
        pseudo = pseudo_cb_coordinate(residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"])
        result["pseudo_cb_available"] = True
    except ValueError:
        result["chirality_reason"] = "degenerate_backbone_geometry"
        return result

    chirality_class = str(result["chirality_class"])
    if chirality_class == "glycine_achiral":
        result["chirality_reason"] = "glycine_achiral"
        return result
    if chirality_class == "noncanonical_unknown":
        result["chirality_reason"] = "noncanonical_configuration_unknown"
        return result
    if "CB" not in residue.atoms:
        result["chirality_reason"] = "native_cb_unavailable"
        return result

    native_vector = residue.atoms["CB"] - residue.atoms["CA"]
    pseudo_vector = pseudo - residue.atoms["CA"]
    result["pseudo_cb_native_distance_angstrom"] = float(np.linalg.norm(pseudo - residue.atoms["CB"]))
    result["pseudo_cb_native_angle_degrees"] = _angle_degrees(pseudo_vector, native_vector)
    result["pseudo_cb_agreement"] = bool(
        result["pseudo_cb_native_distance_angstrom"] <= pseudo_cb_maximum_distance_angstrom
        and result["pseudo_cb_native_angle_degrees"] <= pseudo_cb_maximum_angle_degrees
    )
    signed_volume = native_chirality_signed_volume(
        residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"], residue.atoms["CB"]
    )
    result["native_chirality_available"] = abs(signed_volume) > signed_volume_epsilon
    result["native_chirality_signed_volume"] = signed_volume
    if not result["native_chirality_available"]:
        result["chirality_reason"] = "degenerate_native_chirality_volume"
        return result
    expected_positive = chirality_class in {"canonical_l", "mapped_modified_l"}
    result["native_chirality_valid"] = bool(signed_volume > 0 if expected_positive else signed_volume < 0)
    configuration = "l" if expected_positive else "d"
    result["chirality_reason"] = (
        f"native_{configuration}_configuration_valid"
        if result["native_chirality_valid"]
        else f"native_{configuration}_sign_contradiction"
    )
    return result


def dihedral_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Return a signed torsion in radians, or fail for degenerate geometry."""
    b0 = np.asarray(b) - np.asarray(a)
    b1 = np.asarray(c) - np.asarray(b)
    b2 = np.asarray(d) - np.asarray(c)
    norm = np.linalg.norm(b1)
    if norm < 1e-8:
        raise ValueError("degenerate_torsion")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-8:
        raise ValueError("degenerate_torsion")
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def _connected(left: BackboneResidue, right: BackboneResidue, threshold: float) -> bool:
    if "C" not in left.atoms or "N" not in right.atoms:
        return False
    return bool(np.linalg.norm(left.atoms["C"] - right.atoms["N"]) <= threshold)


def compare_ca_distance_matrix(
    residues: list[BackboneResidue], stored_matrix: np.ndarray
) -> dict[str, float | int | bool]:
    """Compare directly available CA pairs with a stored distance matrix."""
    matrix = np.asarray(stored_matrix, dtype=np.float64)
    if matrix.shape != (len(residues), len(residues)):
        return {
            "comparable": False,
            "valid_pair_count": 0,
            "maximum_error_angstrom": math.nan,
            "rmse_angstrom": math.nan,
        }
    indices = [index for index, residue in enumerate(residues) if "CA" in residue.atoms]
    if len(indices) < 2:
        return {
            "comparable": False,
            "valid_pair_count": 0,
            "maximum_error_angstrom": math.nan,
            "rmse_angstrom": math.nan,
        }
    coordinates = np.stack([residues[index].atoms["CA"] for index in indices])
    derived = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    stored = matrix[np.ix_(indices, indices)]
    upper = np.triu_indices(len(indices), k=1)
    error = derived[upper] - stored[upper]
    return {
        "comparable": True,
        "valid_pair_count": int(error.size),
        "maximum_error_angstrom": float(np.max(np.abs(error))),
        "rmse_angstrom": float(np.sqrt(np.mean(error**2))),
    }


def analyze_backbone_residues(
    residues: list[BackboneResidue],
    *,
    expected_sequence: str,
    stored_matrix: np.ndarray,
    peptide_bond_threshold_angstrom: float = 2.0,
    pseudo_cb_maximum_distance_angstrom: float = 0.5,
    pseudo_cb_maximum_angle_degrees: float = 20.0,
) -> dict[str, Any]:
    """Measure reproducible rich-geometry coverage without deriving sidecars."""
    length = len(residues)
    coordinate_sequence = "".join(residue.one_letter or "X" for residue in residues)
    identity_matches = sum(
        observed == expected for observed, expected in zip(coordinate_sequence, expected_sequence, strict=False)
    )
    denominator = max(len(expected_sequence), 1)
    atom_counts = {atom: sum(atom in residue.atoms for residue in residues) for atom in BACKBONE_ATOMS}
    residue_diagnostics = [
        residue_geometry_diagnostics(
            residue,
            pseudo_cb_maximum_distance_angstrom=pseudo_cb_maximum_distance_angstrom,
            pseudo_cb_maximum_angle_degrees=pseudo_cb_maximum_angle_degrees,
        )
        for residue in residues
    ]
    frame_mask = [bool(row["frame_valid"]) for row in residue_diagnostics]
    pseudo_cb_mask = [bool(row["pseudo_cb_available"]) for row in residue_diagnostics]
    frame_count = sum(frame_mask)
    invalid_frame_count = sum(
        all(atom in residue.atoms for atom in ("N", "CA", "C")) and not frame_mask[index]
        for index, residue in enumerate(residues)
    )
    determinants = [float(row["frame_determinant"]) for row in residue_diagnostics if row["frame_valid"]]
    pseudo_cb_count = sum(pseudo_cb_mask)
    eligible_chirality = [row for row in residue_diagnostics if row["native_chirality_available"]]
    valid_chirality_count = sum(row["native_chirality_valid"] is True for row in eligible_chirality)
    invalid_chirality = [row for row in eligible_chirality if row["native_chirality_valid"] is False]
    invalid_chirality_count = len(invalid_chirality)
    pseudo_cb_compared = [row for row in residue_diagnostics if row["pseudo_cb_agreement"] is not None]
    pseudo_cb_disagreements = [row for row in pseudo_cb_compared if not row["pseudo_cb_agreement"]]
    connected = [
        _connected(residues[index], residues[index + 1], peptide_bond_threshold_angstrom)
        for index in range(max(0, length - 1))
    ]
    phi_mask = [False] * length
    psi_mask = [False] * length
    omega_mask = [False] * length
    for index in range(length):
        try:
            if index > 0 and connected[index - 1]:
                dihedral_angle(
                    residues[index - 1].atoms["C"],
                    residues[index].atoms["N"],
                    residues[index].atoms["CA"],
                    residues[index].atoms["C"],
                )
                phi_mask[index] = True
        except (KeyError, ValueError):
            pass
        try:
            if index < length - 1 and connected[index]:
                dihedral_angle(
                    residues[index].atoms["N"],
                    residues[index].atoms["CA"],
                    residues[index].atoms["C"],
                    residues[index + 1].atoms["N"],
                )
                psi_mask[index] = True
        except (KeyError, ValueError):
            pass
        try:
            if index > 0 and connected[index - 1]:
                dihedral_angle(
                    residues[index - 1].atoms["CA"],
                    residues[index - 1].atoms["C"],
                    residues[index].atoms["N"],
                    residues[index].atoms["CA"],
                )
                omega_mask[index] = True
        except (KeyError, ValueError):
            pass
    reasons = []
    if length != len(expected_sequence):
        reasons.append("sequence_coordinate_length_mismatch")
    if identity_matches != len(expected_sequence):
        reasons.append("residue_identity_mismatch")
    if atom_counts["CA"] != length:
        reasons.append("missing_calpha")
    if frame_count != length:
        reasons.append("incomplete_or_invalid_local_frames")
    if invalid_chirality_count:
        reasons.append("chirality_contradiction")
    if pseudo_cb_disagreements:
        reasons.append("pseudo_cb_disagreement")
    matrix_comparison = compare_ca_distance_matrix(residues, stored_matrix)
    return {
        "sequence_length": len(expected_sequence),
        "mapped_coordinate_residue_count": length,
        "coordinate_sequence": coordinate_sequence,
        "residue_numbers_json": json.dumps([residue.residue_number for residue in residues]),
        "insertion_codes_json": json.dumps([residue.insertion_code for residue in residues]),
        "selected_altlocs_json": json.dumps([residue.selected_altlocs or {} for residue in residues], sort_keys=True),
        "sequence_to_coordinate_identity_mask_json": json.dumps(
            [observed == expected for observed, expected in zip(coordinate_sequence, expected_sequence, strict=False)]
        ),
        "atom_availability_masks_json": json.dumps(
            {atom: [atom in residue.atoms for residue in residues] for atom in BACKBONE_ATOMS}, sort_keys=True
        ),
        "local_frame_mask_json": json.dumps(frame_mask),
        "pseudo_cb_mask_json": json.dumps(pseudo_cb_mask),
        "residue_geometry_records_json": json.dumps(residue_diagnostics, sort_keys=True),
        "chain_continuity_mask_json": json.dumps(connected),
        "phi_mask_json": json.dumps(phi_mask),
        "psi_mask_json": json.dumps(psi_mask),
        "omega_mask_json": json.dumps(omega_mask),
        "exact_residue_identity_agreement": coordinate_sequence == expected_sequence,
        "residue_identity_match_fraction": identity_matches / denominator,
        **{f"{atom.lower()}_availability_fraction": atom_counts[atom] / max(length, 1) for atom in BACKBONE_ATOMS},
        "complete_local_frame_fraction": frame_count / max(length, 1),
        "right_handed_frame_count": frame_count,
        "invalid_frame_count": invalid_frame_count,
        "minimum_frame_determinant": min(determinants, default=math.nan),
        "maximum_frame_determinant": max(determinants, default=math.nan),
        "frame_determinants_json": json.dumps(determinants),
        "frame_right_handed_signs_json": json.dumps([value > 0 for value in determinants]),
        "chirality_eligible_residue_count": len(eligible_chirality),
        "chirality_valid_residue_count": valid_chirality_count,
        "chirality_valid_fraction": valid_chirality_count / max(len(eligible_chirality), 1),
        "invalid_chirality_count": invalid_chirality_count,
        "chirality_invalid_residue_examples_json": json.dumps([row["residue_id"] for row in invalid_chirality[:100]]),
        "chirality_convention_version": CHIRALITY_CONVENTION_VERSION,
        "pseudo_cb_compared_residue_count": len(pseudo_cb_compared),
        "pseudo_cb_agreement_count": len(pseudo_cb_compared) - len(pseudo_cb_disagreements),
        "pseudo_cb_disagreement_count": len(pseudo_cb_disagreements),
        "pseudo_cb_convention_version": PSEUDO_CB_CONVENTION_VERSION,
        "phi_computable_fraction": sum(phi_mask) / max(length, 1),
        "psi_computable_fraction": sum(psi_mask) / max(length, 1),
        "omega_computable_fraction": sum(omega_mask) / max(length, 1),
        "pseudo_cb_computable_fraction": pseudo_cb_count / max(length, 1),
        "chain_break_count": connected.count(False),
        "insertion_code_count": sum(bool(residue.insertion_code) for residue in residues),
        "alternate_location_count": sum(residue.alternate_location_count for residue in residues),
        "alternate_location_policy": (
            COORDINATE_ANCHOR_POLICY_VERSION
            if any(residue.atom_candidates is not None for residue in residues)
            else "highest_occupancy_then_blank_then_A_then_lexical"
        ),
        "matrix_comparison": matrix_comparison,
        "rich_features_reproducible": not reasons and invalid_frame_count == 0 and invalid_chirality_count == 0,
        "unavailable_reasons": reasons,
        "orientation_from_distances_inferred": False,
    }


def _clean_altloc(value: Any) -> str | None:
    text = str(value or "").strip().strip(chr(0))
    return None if text in {"", ".", "?"} else text


def _mmcif_columns(block: Any, prefix: str) -> dict[str, list[str]]:
    table = block.find_mmcif_category(prefix)
    if not table:
        return {}
    tags = [str(tag).removeprefix(prefix) for tag in table.tags]
    rows = [[str(value) for value in row] for row in table]
    if not tags or not rows:
        return {}
    if any(len(row) != len(tags) for row in rows):
        raise ValueError(f"{prefix} contains rows with inconsistent widths")
    return {tag: [row[index] for row in rows] for index, tag in enumerate(tags)}


def _mmcif_column(columns: dict[str, list[str]], *names: str, default: str = "?") -> list[str]:
    for name in names:
        if name in columns:
            return columns[name]
    if not columns:
        return []
    return [default] * len(next(iter(columns.values())))


def _authoritative_parent(value: Any) -> str | None:
    parent = _clean_altloc(value)
    if parent is None:
        return None
    values = [item.strip().upper() for item in parent.replace(";", ",").split(",") if item.strip()]
    if len(values) != 1 or values[0] not in STANDARD_AA3_TO_1:
        return None
    return values[0]


def _component_and_scheme_metadata(block: Any) -> dict[str, Any]:
    chemical = _mmcif_columns(block, "_chem_comp.")
    parents = {
        str(component).upper(): _authoritative_parent(parent)
        for component, parent in zip(
            _mmcif_column(chemical, "id"),
            _mmcif_column(chemical, "mon_nstd_parent_comp_id"),
            strict=False,
        )
    }
    parent_sources = {
        component: "_chem_comp.mon_nstd_parent_comp_id" for component, parent in parents.items() if parent is not None
    }
    scheme = _mmcif_columns(block, "_pdbx_poly_seq_scheme.")
    scheme_rows = []
    for label_chain, auth_chain, entity, label_id, auth_id, insertion, component in zip(
        _mmcif_column(scheme, "asym_id"),
        _mmcif_column(scheme, "pdb_strand_id", "auth_asym_id"),
        _mmcif_column(scheme, "entity_id"),
        _mmcif_column(scheme, "seq_id"),
        _mmcif_column(scheme, "auth_seq_num", "pdb_seq_num"),
        _mmcif_column(scheme, "pdb_ins_code"),
        _mmcif_column(scheme, "mon_id"),
        strict=False,
    ):
        scheme_rows.append(
            {
                "label_chain": _clean_altloc(label_chain),
                "auth_chain": _clean_altloc(auth_chain),
                "entity_id": _clean_altloc(entity),
                "label_id": _clean_altloc(label_id),
                "auth_id": _clean_altloc(auth_id),
                "insertion": _clean_altloc(insertion) or "",
                "component": str(component).upper(),
            }
        )
    modified = _mmcif_columns(block, "_pdbx_struct_mod_residue.")
    position_parents = {}
    for label_chain, auth_chain, label_id, auth_id, insertion, parent in zip(
        _mmcif_column(modified, "label_asym_id"),
        _mmcif_column(modified, "auth_asym_id"),
        _mmcif_column(modified, "label_seq_id"),
        _mmcif_column(modified, "auth_seq_id"),
        _mmcif_column(modified, "PDB_ins_code", "pdb_ins_code"),
        _mmcif_column(modified, "parent_comp_id"),
        strict=False,
    ):
        parent_name = _authoritative_parent(parent)
        if parent_name is not None:
            position_parents[
                (
                    _clean_altloc(label_chain),
                    _clean_altloc(auth_chain),
                    _clean_altloc(label_id),
                    _clean_altloc(auth_id),
                    _clean_altloc(insertion) or "",
                )
            ] = parent_name
    return {
        "component_parents": parents,
        "component_parent_sources": parent_sources,
        "scheme_rows": scheme_rows,
        "position_parents": position_parents,
    }


def parse_backbone_mmcif(
    path: str | Path,
    *,
    chain_id: str,
    model_number: int,
    residue_mappings: dict[str, str] | None = None,
    stage_observer: Callable[[str, float, int, int], None] | None = None,
    parse_mode: str = "single_parse",
) -> tuple[list[BackboneResidue], dict[str, Any]]:
    """Parse polymer residues from one chain/model and inventory chain extras."""
    try:
        import gemmi
    except ModuleNotFoundError as error:
        raise RuntimeError("Gemmi is required for the E006 source audit") from error
    # Kept for call compatibility. V3 never treats this name map as
    # authoritative modified-residue evidence.
    configured_name_mappings = dict(residue_mappings or {})
    source_path = Path(path)
    started = time.perf_counter()
    if source_path.suffix.lower() == ".gz":
        with gzip.open(source_path, "rt", encoding="utf-8") as handle:
            mmcif_text = handle.read()
    else:
        mmcif_text = source_path.read_text(encoding="utf-8")
    if stage_observer is not None:
        stage_observer("mmcif_gzip_reading", time.perf_counter() - started, source_path.stat().st_size, 0)
    started = time.perf_counter()
    document = gemmi.cif.read_string(mmcif_text)
    block = document.sole_block()
    source_metadata = _component_and_scheme_metadata(block)
    if parse_mode == "single_parse":
        structure = gemmi.make_structure_from_block(block)
    elif parse_mode == "legacy_double_parse":
        structure_document = gemmi.cif.read_string(mmcif_text)
        structure = gemmi.make_structure_from_block(structure_document.sole_block())
        del structure_document
    else:
        raise ValueError(f"unknown_mmcif_parse_mode:{parse_mode}")
    if stage_observer is not None:
        stage_observer("gemmi_parsing", time.perf_counter() - started, len(mmcif_text.encode("utf-8")), 0)
    del block, document, mmcif_text
    scheme_rows = source_metadata["scheme_rows"]
    chain_scheme = [row for row in scheme_rows if str(chain_id) in {str(row["label_chain"]), str(row["auth_chain"])}]
    candidate_chain_names = {
        str(chain_id),
        *(str(row["auth_chain"]) for row in chain_scheme if row["auth_chain"] is not None),
    }
    model_sequences: dict[str, str] = {}
    selected_residues: list[BackboneResidue] | None = None
    selected_model_name = None
    selected_ignored_components: list[dict[str, Any]] = []
    model_residues: dict[str, list[BackboneResidue]] = {}
    for model_index, model in enumerate(structure, start=1):
        numeric_model = int(model.num) if int(model.num) > 0 else model_index
        model_name = str(numeric_model)
        chain = next((item for item in model if str(item.name) in candidate_chain_names), None)
        if chain is None:
            continue
        parsed = []
        ignored_components = []
        for residue in chain:
            raw_name = str(residue.name).upper()
            auth_id = str(residue.seqid.num)
            insertion = _clean_altloc(residue.seqid.icode) or ""
            scheme_match = next(
                (
                    row
                    for row in chain_scheme
                    if row["auth_id"] == auth_id and row["insertion"] == insertion and row["component"] == raw_name
                ),
                None,
            )
            polymer_member = bool(scheme_match) if scheme_rows else residue.het_flag == "A"
            parent = source_metadata["component_parents"].get(raw_name)
            parent_source = source_metadata["component_parent_sources"].get(raw_name)
            if scheme_match is not None:
                position_key = (
                    scheme_match["label_chain"],
                    scheme_match["auth_chain"],
                    scheme_match["label_id"],
                    scheme_match["auth_id"],
                    insertion,
                )
                position_parent = source_metadata["position_parents"].get(position_key)
                if position_parent is not None:
                    if parent in {None, position_parent}:
                        parent = position_parent
                        parent_source = "_pdbx_struct_mod_residue.parent_comp_id"
                    else:
                        parent = None
                        parent_source = "conflicting_authoritative_parent_records"
            if raw_name in EXPLICIT_D_RESIDUES:
                one_letter = EXPLICIT_D_RESIDUES[raw_name]
                chirality_class = "explicit_d"
            else:
                one_letter = STANDARD_AA3_TO_1.get(raw_name)
                if one_letter is None and parent is not None:
                    one_letter = STANDARD_AA3_TO_1[parent]
                if one_letter == "G":
                    chirality_class = "glycine_achiral"
                elif one_letter is not None and parent is not None:
                    chirality_class = "mapped_modified_l"
                elif one_letter is not None:
                    chirality_class = "canonical_l"
                else:
                    chirality_class = "noncanonical_unknown"
            candidates: dict[str, list[tuple[str | None, float | None, np.ndarray]]] = defaultdict(list)
            for atom in residue:
                atom_name = str(atom.name).strip().upper()
                if atom_name not in BACKBONE_ATOMS:
                    continue
                coordinate = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                if np.isfinite(coordinate).all():
                    candidates[atom_name].append((_clean_altloc(atom.altloc), float(atom.occ), coordinate))
            atoms, altlocs, alternate_count = select_atom_alternate_locations(candidates)
            if not polymer_member:
                ignored_components.append(
                    {
                        "residue_name": raw_name,
                        "auth_sequence_id": auth_id,
                        "insertion_code": insertion,
                        "model_number": numeric_model,
                        "chain_id": str(chain.name),
                        "classification": "ignored_unmapped_source_component",
                    }
                )
                continue
            parsed.append(
                BackboneResidue(
                    residue_name=raw_name,
                    one_letter=one_letter,
                    residue_number=auth_id,
                    insertion_code=insertion,
                    atoms=atoms,
                    alternate_location_count=alternate_count,
                    selected_altlocs=altlocs,
                    chirality_class=chirality_class,
                    label_sequence_id=scheme_match["label_id"] if scheme_match else None,
                    auth_sequence_id=auth_id,
                    entity_id=scheme_match["entity_id"] if scheme_match else None,
                    polymer_member=True,
                    authoritative_parent_comp_id=parent,
                    authoritative_parent_source=parent_source,
                    atom_candidates={name: tuple(values) for name, values in candidates.items()},
                )
            )
        model_sequences[str(numeric_model)] = "".join(item.one_letter or "X" for item in parsed)
        model_residues[str(numeric_model)] = parsed
        if numeric_model == int(model_number):
            selected_residues = parsed
            selected_model_name = model_name
            selected_ignored_components = ignored_components
    if selected_residues is None:
        raise ValueError(f"selected_chain_or_model_unavailable:{chain_id}:{model_number}")
    return selected_residues, {
        "nmr_model_count": len(model_sequences),
        "selected_model_number": int(model_number),
        "selected_model_name": selected_model_name,
        "model_sequences_json": json.dumps(model_sequences, sort_keys=True),
        "source_polymer_cross_model_sequence_consistent": len(set(model_sequences.values())) <= 1,
        "cross_model_sequence_consistent": len(set(model_sequences.values())) <= 1,
        "ignored_source_components_json": json.dumps(selected_ignored_components, sort_keys=True),
        "ignored_source_component_count": len(selected_ignored_components),
        "configured_residue_name_mappings_are_authoritative": False,
        "configured_residue_name_mapping_count": len(configured_name_mappings),
        "_model_residues": model_residues,
    }


@dataclass(frozen=True)
class IdentifierDecodeResult:
    values: tuple[str, ...]
    status: str
    representation: str
    duplicate_identifiers: bool
    diagnostic: dict[str, Any]


@dataclass(frozen=True)
class IdentifierLocatorResolution:
    payload: Any
    status: str
    original_locator: str | None
    source_artifact: str
    source_record_key: str
    resolved_field_name: str | None
    resolved_raw_type: str | None
    resolved_payload_sha256: str | None
    diagnostic: dict[str, Any]


def _raw_identifier_evidence(value: Any) -> tuple[bytes, str]:
    if isinstance(value, bytes):
        payload = value
        preview = repr(value[:80])
    elif isinstance(value, str):
        payload = value.encode("utf-8", errors="surrogatepass")
        preview = repr(value[:160])
    else:
        rendered = repr(value)
        payload = rendered.encode("utf-8", errors="backslashreplace")
        preview = rendered[:160]
    return payload, preview[:160]


def resolve_identifier_locator(
    *,
    sample_id: str,
    field_name: str,
    raw_value: Any,
    audit_source_record: dict[str, Any] | None,
    audit_source_artifact: str,
    direct_source_artifact: str,
) -> IdentifierLocatorResolution:
    """Resolve one symbolic audit locator without interpreting its payload."""
    locator = raw_value if isinstance(raw_value, str) and raw_value.startswith("audit:") else None
    if locator is None:
        payload_bytes, _ = _raw_identifier_evidence(raw_value)
        raw_type = f"{type(raw_value).__module__}.{type(raw_value).__qualname__}"
        return IdentifierLocatorResolution(
            payload=raw_value,
            status="direct_payload",
            original_locator=None,
            source_artifact=direct_source_artifact,
            source_record_key=str(sample_id),
            resolved_field_name=field_name,
            resolved_raw_type=raw_type,
            resolved_payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            diagnostic={
                "sample_id": str(sample_id),
                "field_name": field_name,
                "original_locator": None,
                "locator_resolution_status": "direct_payload",
                "source_artifact": direct_source_artifact,
                "source_record_key": str(sample_id),
                "resolved_field_name": field_name,
                "resolved_raw_type": raw_type,
                "resolved_payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            },
        )
    referenced_field = locator.removeprefix("audit:")
    valid_syntax = bool(referenced_field) and all(
        character.isalnum() or character == "_" for character in referenced_field
    )
    status = "resolved"
    payload = None
    if not valid_syntax:
        status = "invalid_locator_syntax"
    elif audit_source_record is None:
        status = "referenced_record_unavailable"
    elif audit_source_record.get("sample_id") is not None and str(audit_source_record["sample_id"]) != str(sample_id):
        status = "referenced_record_key_mismatch"
    elif referenced_field not in audit_source_record:
        status = "referenced_field_unavailable"
    else:
        payload = audit_source_record[referenced_field]
        if isinstance(payload, str) and payload.startswith("audit:"):
            status = "nested_locator_rejected"
            payload = None
    raw_type = f"{type(payload).__module__}.{type(payload).__qualname__}" if status == "resolved" else None
    payload_hash = None
    if status == "resolved":
        payload_bytes, _ = _raw_identifier_evidence(payload)
        payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    diagnostic = {
        "sample_id": str(sample_id),
        "field_name": field_name,
        "original_locator": locator,
        "locator_resolution_status": status,
        "source_artifact": audit_source_artifact,
        "source_record_key": str(sample_id),
        "resolved_field_name": referenced_field if valid_syntax else None,
        "resolved_raw_type": raw_type,
        "resolved_payload_sha256": payload_hash,
    }
    return IdentifierLocatorResolution(
        payload=payload,
        status=status,
        original_locator=locator,
        source_artifact=audit_source_artifact,
        source_record_key=str(sample_id),
        resolved_field_name=referenced_field if valid_syntax else None,
        resolved_raw_type=raw_type,
        resolved_payload_sha256=payload_hash,
        diagnostic=diagnostic,
    )


def decode_identifier_array(
    *,
    sample_id: str,
    field_name: str,
    residue_id_convention: str,
    raw_value: Any,
) -> IdentifierDecodeResult:
    """Strictly decode one documented residue-identifier representation."""
    if isinstance(raw_value, str) and raw_value.startswith("audit:"):
        raise ValueError("identifier_locator_reached_payload_decoder")
    raw_bytes, preview = _raw_identifier_evidence(raw_value)
    diagnostic = {
        "sample_id": str(sample_id),
        "field_name": str(field_name),
        "selected_residue_id_convention": str(residue_id_convention),
        "raw_type": f"{type(raw_value).__module__}.{type(raw_value).__qualname__}",
        "raw_byte_length": len(raw_bytes),
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "preview": preview,
    }

    representation = "malformed"
    parsed: Any = None
    status = "ok"
    if raw_value is None or (isinstance(raw_value, (float, np.floating)) and math.isnan(float(raw_value))):
        representation = "missing"
        status = "missing"
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if not text or text.casefold() in {"null", "none", "nan"}:
            representation = "missing"
            status = "missing"
        else:
            representation = "json_array_string"
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                representation = "malformed"
                status = "malformed"
    elif isinstance(raw_value, bytes):
        representation = "utf8_json_bytes"
        try:
            text = raw_value.decode("utf-8")
            parsed = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            representation = "malformed"
            status = "malformed"
    elif isinstance(raw_value, (list, tuple)):
        representation = "native_array"
        parsed = list(raw_value)
    elif isinstance(raw_value, np.ndarray):
        if raw_value.ndim == 0:
            scalar = raw_value.item()
            if isinstance(scalar, (float, np.floating)) and math.isnan(float(scalar)):
                representation = "missing"
                status = "missing"
            else:
                representation = "wrong_json_type"
                status = "wrong_json_type"
        else:
            representation = "native_array"
            parsed = raw_value.tolist()
    elif isinstance(raw_value, (pa.Array, pa.ChunkedArray)):
        representation = "native_array"
        parsed = raw_value.to_pylist()
    elif isinstance(raw_value, pa.Scalar):
        arrow_value = raw_value.as_py()
        if isinstance(arrow_value, list):
            representation = "native_array"
            parsed = arrow_value
        else:
            representation = "wrong_json_type"
            status = "wrong_json_type"
    else:
        representation = "wrong_json_type"
        status = "wrong_json_type"

    if status == "ok" and not isinstance(parsed, list):
        representation = "wrong_json_type"
        status = "wrong_json_type"
    values = []
    if status == "ok":
        for item in parsed:
            if isinstance(item, (list, tuple, dict, np.ndarray)) or isinstance(item, bool) or item is None:
                status = "malformed"
                representation = "malformed"
                break
            if isinstance(item, (float, np.floating)) and not math.isfinite(float(item)):
                status = "malformed"
                representation = "malformed"
                break
            if not isinstance(item, (str, int, float, np.integer, np.floating)):
                status = "malformed"
                representation = "malformed"
                break
            identifier = str(item)
            if not identifier.strip() and field_name != "insertion_codes":
                status = "malformed"
                representation = "malformed"
                break
            values.append(identifier)
    if status != "ok":
        values = []
    duplicate = len(values) != len(set(values))
    diagnostic.update(
        status=status,
        representation=representation,
        decoded_count=len(values),
        duplicate_identifiers=duplicate,
    )
    return IdentifierDecodeResult(tuple(values), status, representation, duplicate, diagnostic)


def resolve_verified_target_mapping(
    source_residues: list[BackboneResidue],
    *,
    sample_id: str,
    expected_sequence: str,
    residue_ids: Any,
    insertion_codes: Any,
    residue_id_convention: str,
    pairing_canonicalization_version: str | None,
    alternative_residue_ids: dict[str, Any] | None = None,
    authorized_residue_id_conventions: set[str] | None = None,
    audit_source_record: dict[str, Any] | None = None,
    audit_source_artifact: str = "compact_audit:matrix_pair_alignments",
    direct_source_artifact: str = "pairing_dataset",
) -> dict[str, Any]:
    """Resolve only the matrix-building residue interval from pairing evidence."""
    convention = str(residue_id_convention or "")
    diagnostics: list[dict[str, Any]] = []

    def resolve_and_decode(
        raw_value: Any, field_name: str, selected_convention: str
    ) -> tuple[IdentifierLocatorResolution, IdentifierDecodeResult]:
        locator = resolve_identifier_locator(
            sample_id=sample_id,
            field_name=field_name,
            raw_value=raw_value,
            audit_source_record=audit_source_record,
            audit_source_artifact=audit_source_artifact,
            direct_source_artifact=direct_source_artifact,
        )
        if locator.status not in {"resolved", "direct_payload"}:
            decoded = IdentifierDecodeResult(
                (),
                "not_run_locator_unavailable",
                "unavailable",
                False,
                {
                    "sample_id": sample_id,
                    "field_name": field_name,
                    "selected_residue_id_convention": selected_convention,
                    "status": "not_run_locator_unavailable",
                    "representation": "unavailable",
                    "decoded_count": 0,
                    "duplicate_identifiers": False,
                },
            )
        else:
            decoded = decode_identifier_array(
                sample_id=sample_id,
                field_name=field_name,
                residue_id_convention=selected_convention,
                raw_value=locator.payload,
            )
        diagnostics.append(
            {
                **decoded.diagnostic,
                **locator.diagnostic,
                "decoder_status": decoded.status,
                "representation": decoded.representation,
                "decoded_count": len(decoded.values),
                "duplicate_identifiers": decoded.duplicate_identifiers,
            }
        )
        return locator, decoded

    identifier_locator, identifier_decode = resolve_and_decode(residue_ids, "residue_ids", convention)
    original_identifier_locator = identifier_locator.original_locator
    fallback_used = False
    fallback_field = None
    fallback_locator: IdentifierLocatorResolution | None = None
    alternatives = alternative_residue_ids or {}
    authorized = authorized_residue_id_conventions or set()
    if identifier_decode.status == "missing":
        candidates = []
        for alternative_convention, value in alternatives.items():
            if alternative_convention not in authorized:
                continue
            alternative_locator, decoded = resolve_and_decode(
                value,
                f"residue_ids:{alternative_convention}",
                alternative_convention,
            )
            if decoded.status != "missing":
                candidates.append((alternative_convention, alternative_locator, decoded))
        if len(candidates) == 1:
            convention, fallback_locator, identifier_decode = candidates[0]
            identifier_locator = fallback_locator
            fallback_used = True
            fallback_field = identifier_decode.diagnostic["field_name"]
        elif len(candidates) > 1:
            identifier_decode = IdentifierDecodeResult(
                (),
                "namespace_contradiction",
                "missing",
                False,
                identifier_decode.diagnostic,
            )
    insertion_locator, insertion_decode = resolve_and_decode(insertion_codes, "insertion_codes", convention)
    identifiers = list(identifier_decode.values)
    insertions = list(insertion_decode.values)
    mapping_errors = []
    mapping_exclusion_reason = None
    locator_failures = {
        "invalid_locator_syntax",
        "referenced_record_unavailable",
        "referenced_record_key_mismatch",
        "referenced_field_unavailable",
        "nested_locator_rejected",
    }
    if identifier_locator.status in locator_failures or insertion_locator.status in locator_failures:
        mapping_exclusion_reason = "excluded_identifier_locator_unavailable"
        mapping_errors.extend(
            f"identifier_locator_{item.status}"
            for item in (identifier_locator, insertion_locator)
            if item.status in locator_failures
        )
    elif identifier_decode.status == "missing":
        mapping_exclusion_reason = "excluded_identity_mapping_unavailable"
        mapping_errors.append("target_residue_ids_missing")
    elif identifier_decode.status in {"malformed", "wrong_json_type"}:
        mapping_exclusion_reason = "excluded_invalid_residue_id_serialization"
        mapping_errors.append("invalid_residue_id_serialization")
    elif identifier_decode.status == "namespace_contradiction":
        mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
        mapping_errors.append("multiple_provenance_authorized_namespace_fallbacks")
    if insertion_decode.status in {"malformed", "wrong_json_type"}:
        mapping_exclusion_reason = "excluded_invalid_residue_id_serialization"
        mapping_errors.append("invalid_insertion_code_serialization")
    if (
        identifier_locator.status in {"resolved", "direct_payload"}
        and insertion_locator.status in {"resolved", "direct_payload"}
        and (
            identifier_locator.source_artifact != insertion_locator.source_artifact
            or identifier_locator.source_record_key != insertion_locator.source_record_key
        )
    ):
        mapping_errors.append("identifier_fields_from_different_source_records")
        mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
    if identifier_decode.status == "ok" and not identifiers and expected_sequence:
        mapping_errors.append("target_residue_ids_missing")
        mapping_exclusion_reason = "excluded_identity_mapping_unavailable"
    elif identifier_decode.status == "ok" and len(identifiers) != len(expected_sequence):
        mapping_errors.append("target_residue_id_count_mismatch")
        mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
    if (
        identifier_decode.status == "ok"
        and insertion_decode.status == "ok"
        and insertions
        and len(insertions) != len(identifiers)
    ):
        mapping_errors.append("target_insertion_code_count_mismatch")
        mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
    if not insertions:
        insertions = [""] * len(identifiers)

    selected_indices: list[int] = []
    if not mapping_errors:
        if convention == "auth_seq_id_insertion":
            source_ids = [
                f"{residue.auth_sequence_id or residue.residue_number}{residue.insertion_code}"
                for residue in source_residues
            ]
            target_ids = [
                value if insertion and value.endswith(insertion) else f"{value}{insertion}"
                for value, insertion in zip(identifiers, insertions, strict=True)
            ]
            selected_indices, ambiguous = _ordered_target_indices(source_ids, target_ids)
        elif convention == "label_seq_id":
            source_ids = [str(residue.label_sequence_id or "") for residue in source_residues]
            selected_indices, ambiguous = _ordered_target_indices(source_ids, identifiers)
        elif convention in {"position_zero_based", "position_one_based"}:
            offset = 0 if convention == "position_zero_based" else 1
            try:
                selected_indices = [int(value) - offset for value in identifiers]
            except ValueError:
                selected_indices = []
            ambiguous = len(set(selected_indices)) != len(selected_indices)
            if any(index < 0 or index >= len(source_residues) for index in selected_indices):
                selected_indices = []
        else:
            selected_indices = []
            ambiguous = False
            mapping_errors.append("unsupported_or_missing_residue_id_convention")
            mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
        if not selected_indices or len(selected_indices) != len(identifiers):
            mapping_errors.append("target_residue_locator_unresolved")
            mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"
        if ambiguous:
            mapping_errors.append("target_residue_locator_ambiguous")
            mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"

    if convention == "auth_seq_id_insertion" and identifiers and len(insertions) == len(identifiers):
        effective_ids = [
            value if insertion and value.endswith(insertion) else f"{value}{insertion}"
            for value, insertion in zip(identifiers, insertions, strict=True)
        ]
    else:
        effective_ids = identifiers
    if len(effective_ids) != len(set(effective_ids)):
        mapping_errors.append("duplicate_effective_residue_identifiers")
        mapping_exclusion_reason = "excluded_residue_id_namespace_contradiction"

    selected = [source_residues[index] for index in selected_indices] if not mapping_errors else []
    selected_index_set = set(selected_indices)
    ignored = [residue for index, residue in enumerate(source_residues) if index not in selected_index_set]
    classifications = []
    observed_tokens = []
    unresolved_modifications = []
    explicit_d_residues = []
    modified_mapping_evidence = []
    missing_atom_residues = []
    for residue in selected:
        raw_name = residue.residue_name.upper()
        if residue.chirality_class == "explicit_d":
            token = str(residue.one_letter or "X")
            base_classification = "excluded_stereochemical_contradiction"
            explicit_d_residues.append(f"{residue.residue_number}{residue.insertion_code}:{raw_name}")
        elif raw_name in STANDARD_AA3_TO_1:
            token = STANDARD_AA3_TO_1[raw_name]
            base_classification = "canonical_target_residue"
        elif (
            residue.authoritative_parent_comp_id in STANDARD_AA3_TO_1
            and residue.authoritative_parent_source
            in {
                "_chem_comp.mon_nstd_parent_comp_id",
                "_pdbx_struct_mod_residue.parent_comp_id",
            }
            and pairing_canonicalization_version
        ):
            token = STANDARD_AA3_TO_1[str(residue.authoritative_parent_comp_id)]
            base_classification = "mapped_modified_target_residue"
            modified_mapping_evidence.append(
                {
                    "residue_id": f"{residue.residue_number}{residue.insertion_code}",
                    "source_component_id": raw_name,
                    "parent_component_id": residue.authoritative_parent_comp_id,
                    "parent_provenance": residue.authoritative_parent_source,
                    "canonical_token": token,
                }
            )
        else:
            token = "X"
            base_classification = "excluded_unresolved_target_modification"
            unresolved_modifications.append(f"{residue.residue_number}{residue.insertion_code}:{raw_name}")
        observed_tokens.append(token)
        expected_atoms = {"N", "CA", "C", "O"} | ({"CB"} if token != "G" else set())
        missing_atoms = sorted(expected_atoms - residue.atoms.keys())
        if missing_atoms and not base_classification.startswith("excluded_"):
            classification = "masked_target_missing_atoms"
            missing_atom_residues.append(
                {
                    "residue_id": f"{residue.residue_number}{residue.insertion_code}",
                    "residue_name": raw_name,
                    "missing_atoms": missing_atoms,
                    "underlying_classification": base_classification,
                }
            )
        else:
            classification = base_classification
        classifications.append(classification)

    observed_sequence = "".join(observed_tokens)
    if selected and observed_sequence != expected_sequence:
        mapping_errors.append("target_sequence_identity_mismatch")
    if unresolved_modifications:
        mapping_errors.append("unresolved_target_modification")
    return {
        "target_residues": selected,
        "mapping_verified": not mapping_errors,
        "mapping_errors": sorted(set(mapping_errors)),
        "mapping_exclusion_reason": mapping_exclusion_reason,
        "identifier_decode_diagnostics_json": json.dumps(diagnostics, sort_keys=True),
        "residue_ids_original_locator": original_identifier_locator,
        "residue_ids_source_artifact": identifier_locator.source_artifact,
        "residue_ids_source_record_key": identifier_locator.source_record_key,
        "residue_ids_resolved_field_name": identifier_locator.resolved_field_name,
        "residue_ids_resolved_raw_type": identifier_locator.resolved_raw_type,
        "residue_ids_resolved_payload_sha256": identifier_locator.resolved_payload_sha256,
        "residue_ids_decoded_identifier_count": len(identifier_decode.values),
        "residue_ids_locator_resolution_status": identifier_locator.status,
        "residue_ids_decoder_status": identifier_decode.status,
        "insertion_codes_original_locator": insertion_locator.original_locator,
        "insertion_codes_source_artifact": insertion_locator.source_artifact,
        "insertion_codes_source_record_key": insertion_locator.source_record_key,
        "insertion_codes_resolved_field_name": insertion_locator.resolved_field_name,
        "insertion_codes_resolved_raw_type": insertion_locator.resolved_raw_type,
        "insertion_codes_resolved_payload_sha256": insertion_locator.resolved_payload_sha256,
        "insertion_codes_decoded_identifier_count": len(insertion_decode.values),
        "insertion_codes_locator_resolution_status": insertion_locator.status,
        "insertion_codes_decoder_status": insertion_decode.status,
        "residue_id_representation": identifier_decode.representation,
        "residue_id_decode_status": identifier_decode.status,
        "residue_id_duplicate_identifiers": identifier_decode.duplicate_identifiers,
        "insertion_code_representation": insertion_decode.representation,
        "insertion_code_decode_status": insertion_decode.status,
        "insertion_code_duplicate_identifiers": insertion_decode.duplicate_identifiers,
        "residue_id_fallback_used": fallback_used,
        "residue_id_fallback_field": fallback_field,
        "observed_target_sequence": observed_sequence,
        "resolved_target_residue_ids": [
            f"{residue.auth_sequence_id or residue.residue_number}{residue.insertion_code}" for residue in selected
        ],
        "resolved_target_residue_ids_json": json.dumps(
            [f"{residue.auth_sequence_id or residue.residue_number}{residue.insertion_code}" for residue in selected]
        ),
        "target_residue_classifications": classifications,
        "target_residue_classifications_json": json.dumps(classifications),
        "unresolved_target_modification_examples_json": json.dumps(unresolved_modifications[:100]),
        "modified_target_mapping_evidence_json": json.dumps(modified_mapping_evidence[:100], sort_keys=True),
        "explicit_d_target_residue_examples_json": json.dumps(explicit_d_residues[:100]),
        "explicit_d_target_residue_count": len(explicit_d_residues),
        "masked_missing_atom_residue_count": len(missing_atom_residues),
        "masked_missing_atom_residues_json": json.dumps(missing_atom_residues[:100], sort_keys=True),
        "ignored_unmapped_source_component_count": len(ignored),
        "ignored_unmapped_source_components_json": json.dumps(
            [
                {
                    "residue_id": f"{residue.residue_number}{residue.insertion_code}",
                    "residue_name": residue.residue_name,
                    "classification": "ignored_unmapped_source_component",
                }
                for residue in ignored[:100]
            ],
            sort_keys=True,
        ),
        "mapping_version": TARGET_MAPPING_POLICY_VERSION,
        "canonicalization_version": CANONICALIZATION_POLICY_VERSION,
        "pairing_canonicalization_version": pairing_canonicalization_version,
        "residue_id_convention": convention,
    }


def _ordered_target_indices(source_ids: list[str], target_ids: list[str]) -> tuple[list[int], bool]:
    states: list[list[int]] = [[]]
    for target in target_ids:
        positions = [index for index, source in enumerate(source_ids) if source == target]
        candidates = [state + [index] for state in states for index in positions if not state or index > state[-1]]
        states = [list(value) for value in sorted(set(map(tuple, candidates)))[:2]]
        if not states:
            return [], False
    return states[0], len(states) > 1


def resolve_rich_geometry_eligibility(
    mapping: dict[str, Any],
    analysis: dict[str, Any],
    *,
    cross_model_sequence_consistent: bool = True,
) -> dict[str, Any]:
    """Turn measured evidence into one documented v5 eligibility decision."""
    exclusion_reason = mapping.get("coordinate_anchor_exclusion_reason") or mapping.get("mapping_exclusion_reason")
    mapping_errors = set(mapping["mapping_errors"])
    if exclusion_reason is not None:
        pass
    elif "unresolved_target_modification" in mapping_errors:
        exclusion_reason = "excluded_unresolved_target_modification"
    elif mapping_errors or not cross_model_sequence_consistent:
        exclusion_reason = "excluded_identity_mapping_contradiction"
    elif (
        int(mapping.get("explicit_d_target_residue_count", 0)) > 0
        or int(analysis.get("invalid_chirality_count", 0)) > 0
    ):
        exclusion_reason = "excluded_stereochemical_contradiction"
    if exclusion_reason is None and not mapping.get("source_to_npz_calpha_anchor_consistency", True):
        exclusion_reason = "excluded_source_npz_coordinate_contradiction"
    eligible = exclusion_reason is None
    residue_classes = set(mapping.get("target_residue_classifications", []))
    eligible_classification = (
        "masked_target_missing_atoms"
        if "masked_target_missing_atoms" in residue_classes
        else "mapped_modified_target_residue"
        if "mapped_modified_target_residue" in residue_classes
        else "canonical_target_residue"
    )
    return {
        "rich_geometry_eligible": eligible,
        "eligibility_classification": exclusion_reason or eligible_classification,
        "exclusion_reason": exclusion_reason,
        "resolved_exclusion_count": int(not eligible),
        "unexplained_contradiction_count": 0,
        "ignored_source_component_count": int(mapping["ignored_unmapped_source_component_count"]),
        "masked_missing_atom_residue_count": int(mapping["masked_missing_atom_residue_count"]),
        "mapping_version": TARGET_MAPPING_POLICY_VERSION,
        "canonicalization_version": CANONICALIZATION_POLICY_VERSION,
        "chirality_version": CHIRALITY_CONVENTION_VERSION,
    }


def stratified_select(
    rows: list[dict[str, Any]], *, target: int, seed: int, stratum_fields: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically select all sparse strata, then redistribute globally."""
    if target < 1:
        raise ValueError("target must be positive")
    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("selected candidate has no sample_id")
        if sample_id in unique:
            duplicate_count += 1
            if _canonical_hash(unique[sample_id]) != _canonical_hash(row):
                raise ValueError(f"conflicting_duplicate_sample_id:{sample_id}")
            continue
        unique[sample_id] = row
    if len(unique) < target:
        raise ValueError(f"insufficient_unique_candidates:requested={target},available={len(unique)}")
    strata: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in unique.values():
        key = tuple(str(row.get(field) or "unknown") for field in stratum_fields)
        rank = hashlib.sha256(f"{SAMPLER_VERSION}:{seed}:{row['sample_id']}".encode()).hexdigest()
        strata[key].append((rank, row))
    for values in strata.values():
        values.sort(key=lambda item: (item[0], item[1]["sample_id"]))
    if len(strata) > target:
        raise ValueError(f"panel_too_small_for_all_strata:target={target},strata={len(strata)}")
    quota = max(1, target // len(strata))
    selected = [item for key in sorted(strata) for item in strata[key][:quota]]
    if len(selected) > target:
        selected = selected[:target]
    initial_count = len(selected)
    selected_ids = {row["sample_id"] for _, row in selected}
    if len(selected) < target:
        remaining = [
            (hashlib.sha256(f"{SAMPLER_VERSION}:backfill:{seed}:{row['sample_id']}".encode()).hexdigest(), row)
            for values in strata.values()
            for _, row in values
            if row["sample_id"] not in selected_ids
        ]
        remaining.sort(key=lambda item: (item[0], item[1]["sample_id"]))
        selected.extend(remaining[: target - len(selected)])
    output = [row for _, row in selected]
    counts = Counter(tuple(str(row.get(field) or "unknown") for field in stratum_fields) for row in output)
    diagnostics = {
        "requested_count": target,
        "available_unique_count": len(unique),
        "initially_stratified_count": initial_count,
        "backfilled_count": target - initial_count,
        "final_count": len(output),
        "duplicate_sample_id_count": duplicate_count,
        "sampler_version": SAMPLER_VERSION,
        "seed": seed,
        "sample_id_sha256": _canonical_hash([row["sample_id"] for row in output]),
        "per_stratum": [
            {
                **dict(zip(stratum_fields, key, strict=True)),
                "available_count": len(strata[key]),
                "selected_count": counts[key],
            }
            for key in sorted(strata)
        ],
    }
    return output, diagnostics


def sidecar_storage_estimates(
    lengths: list[int], *, pair_channels: int = 16, residue_channels: int = 32, compression_ratio: float = 0.45
) -> list[dict[str, Any]]:
    """Estimate per-sample disk and tensor memory without building features."""
    if pair_channels < 1 or residue_channels < 1 or not 0 < compression_ratio <= 1:
        raise ValueError("invalid sidecar storage-estimate configuration")
    rows = []
    for length in lengths:
        dense32 = length * length * pair_channels * 4
        dense16 = length * length * pair_channels * 2
        residue = length * residue_channels * 4
        reconstructible_runtime = residue + dense32
        rows.append(
            {
                "length": int(length),
                "dense_float32_disk_bytes": dense32,
                "dense_float16_disk_bytes": dense16,
                "residue_reconstructible_disk_bytes": residue,
                "compressed_npz_or_zstd_estimated_bytes": int(dense16 * compression_ratio),
                "dense_float32_runtime_bytes": dense32,
                "dense_float16_runtime_bytes": dense16,
                "residue_plus_reconstructed_pair_runtime_bytes": reconstructible_runtime,
                "assumptions": {
                    "pair_channels": pair_channels,
                    "residue_channels": residue_channels,
                    "compression_ratio": compression_ratio,
                },
            }
        )
    return rows


def full_corpus_storage_estimates(
    length_statistics: dict[str, dict[str, int]],
    *,
    pair_channels: int,
    residue_channels: int,
    compression_ratio: float,
) -> list[dict[str, Any]]:
    """Project full-corpus storage from streamed counts and exact lengths."""
    output = []
    for length_bin, values in sorted(length_statistics.items()):
        sample_count = int(values["sample_count"])
        residue_count = int(values["residue_count"])
        squared_length_sum = int(values["squared_length_sum"])
        dense32 = squared_length_sum * pair_channels * 4
        dense16 = squared_length_sum * pair_channels * 2
        residue = residue_count * residue_channels * 4
        output.append(
            {
                "length_bin": length_bin,
                "sample_count": sample_count,
                "dense_float32_total_disk_bytes": dense32,
                "dense_float16_total_disk_bytes": dense16,
                "residue_reconstructible_total_disk_bytes": residue,
                "compressed_npz_or_zstd_estimated_total_bytes": int(dense16 * compression_ratio),
                "mean_dense_float32_runtime_bytes_per_sample": dense32 / max(sample_count, 1),
                "mean_dense_float16_runtime_bytes_per_sample": dense16 / max(sample_count, 1),
                "mean_residue_plus_reconstructed_pair_runtime_bytes_per_sample": (dense32 + residue)
                / max(sample_count, 1),
            }
        )
    return output


def phase1_authorization(records: list[dict[str, Any]], criteria: dict[str, float]) -> dict[str, Any]:
    """Apply unchanged gates to the explicitly attested eligible subset."""
    eligible = [row for row in records if row.get("rich_geometry_eligible", True)]
    internally_consistent = [row for row in eligible if row.get("npz_internal_matrix_consistency", True)]
    anchored = [row for row in internally_consistent if row.get("source_to_npz_calpha_anchor_consistency", True)]
    verified = [row for row in anchored if row.get("source_coordinates_verified")]
    residue_total = sum(int(row.get("sequence_length", 0)) for row in verified)
    weighted_identity = sum(
        float(row.get("residue_identity_match_fraction", 0)) * int(row["sequence_length"]) for row in verified
    )
    weighted_frames = sum(
        float(row.get("complete_local_frame_fraction", 0)) * int(row["sequence_length"]) for row in verified
    )
    weighted_phi_psi = sum(
        min(float(row.get("phi_computable_fraction", 0)), float(row.get("psi_computable_fraction", 0)))
        * int(row["sequence_length"])
        for row in verified
    )
    anchored_residue_count = sum(
        int(row.get("uniquely_anchored_residue_count", row.get("sequence_length", 0))) for row in verified
    )
    squared_error = sum(
        float(row.get("source_to_npz_calpha_coordinate_rmse_angstrom", row.get("matrix_rmse_angstrom", 0))) ** 2
        * int(row.get("uniquely_anchored_residue_count", row.get("sequence_length", 0)))
        for row in verified
        if math.isfinite(
            float(row.get("source_to_npz_calpha_coordinate_rmse_angstrom", row.get("matrix_rmse_angstrom", math.nan)))
        )
    )
    values = {
        "verified_source_fraction": len(verified) / max(len(eligible), 1),
        "mapped_identity_fraction": weighted_identity / max(residue_total, 1),
        "complete_frame_fraction": weighted_frames / max(residue_total, 1),
        "phi_psi_fraction": weighted_phi_psi / max(residue_total, 1),
        "matrix_rmse_angstrom": math.sqrt(squared_error / anchored_residue_count)
        if anchored_residue_count
        else math.inf,
        "npz_internal_matrix_consistency": len(internally_consistent) == len(eligible),
        "source_to_npz_calpha_anchor_consistency": len(anchored) == len(eligible),
        "unexplained_contradiction_count": sum(int(row.get("unexplained_contradiction_count", 0)) for row in records),
    }
    checks = {
        "eligible_subset_nonempty": bool(eligible),
        "npz_internal_matrix_consistency": values["npz_internal_matrix_consistency"],
        "source_to_npz_calpha_anchor_consistency": values["source_to_npz_calpha_anchor_consistency"],
        "verified_source_fraction": values["verified_source_fraction"] >= criteria["minimum_verified_source_fraction"],
        "mapped_identity_fraction": values["mapped_identity_fraction"] >= criteria["minimum_mapped_identity_fraction"],
        "complete_frame_fraction": values["complete_frame_fraction"] >= criteria["minimum_complete_frame_fraction"],
        "phi_psi_fraction": values["phi_psi_fraction"] >= criteria["minimum_phi_psi_fraction"],
        "matrix_rmse_angstrom": values["matrix_rmse_angstrom"] <= criteria["maximum_matrix_rmse_angstrom"],
        "unexplained_contradiction_count": values["unexplained_contradiction_count"]
        <= int(criteria["maximum_unexplained_contradictions"]),
    }
    population = {
        "audit_panel_sample_count": len(records),
        "eligible_subset_count": len(eligible),
        "resolved_excluded_count": sum(int(row.get("resolved_exclusion_count", 0)) for row in records),
        "unexplained_record_count": sum(int(row.get("unexplained_contradiction_count", 0)) > 0 for row in records),
        "denominator_policy": "coverage gates use only explicitly eligible records; all exclusions remain counted",
    }
    return {
        "authorized": all(checks.values()),
        "values": values,
        "checks": checks,
        "criteria": criteria,
        "authorization_population": population,
    }


def eligibility_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "audit_panel_sample_count": len(records),
        "unexplained_contradiction_count": sum(int(row.get("unexplained_contradiction_count", 0)) for row in records),
        "resolved_exclusion_count": sum(int(row.get("resolved_exclusion_count", 0)) for row in records),
        "ignored_source_component_count": sum(int(row.get("ignored_source_component_count", 0)) for row in records),
        "masked_missing_atom_residue_count": sum(
            int(row.get("masked_missing_atom_residue_count", 0)) for row in records
        ),
        "rich_geometry_eligible_count": sum(bool(row.get("rich_geometry_eligible", True)) for row in records),
        "rich_geometry_excluded_count": sum(not bool(row.get("rich_geometry_eligible", True)) for row in records),
    }
    counts["exclusion_reason_counts"] = dict(
        sorted(Counter(str(row["exclusion_reason"]) for row in records if row.get("exclusion_reason")).items())
    )
    return counts


def identifier_representation_inventory(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count raw representation and validation outcomes without retaining values."""
    output_counters: dict[str, Counter[str]] = {
        "residue_ids": Counter(),
        "insertion_codes": Counter(),
    }
    combined = Counter()
    locator_statuses = Counter()
    decoder_statuses = Counter()
    for row in records:
        try:
            diagnostics = json.loads(str(row.get("identifier_decode_diagnostics_json") or "[]"))
        except json.JSONDecodeError:
            diagnostics = []
        if not diagnostics:
            diagnostics = [
                {
                    "field_name": "residue_ids",
                    "representation": row.get("residue_id_representation"),
                    "status": row.get("residue_id_decode_status"),
                    "duplicate_identifiers": row.get("residue_id_duplicate_identifiers"),
                },
                {
                    "field_name": "insertion_codes",
                    "representation": row.get("insertion_code_representation"),
                    "status": row.get("insertion_code_decode_status"),
                    "duplicate_identifiers": row.get("insertion_code_duplicate_identifiers"),
                },
            ]
        for diagnostic in diagnostics:
            field = "insertion_codes" if diagnostic.get("field_name") == "insertion_codes" else "residue_ids"
            counter = output_counters[field]
            representation = diagnostic.get("representation")
            status = diagnostic.get("status")
            locator_status = diagnostic.get("locator_resolution_status")
            decoder_status = diagnostic.get("decoder_status", status)
            if locator_status:
                locator_statuses[str(locator_status)] += 1
            if decoder_status:
                decoder_statuses[str(decoder_status)] += 1
            if representation:
                counter[str(representation)] += 1
                combined[str(representation)] += 1
            if status in {"malformed", "wrong_json_type"} and status != representation:
                counter[str(status)] += 1
                combined[str(status)] += 1
            if diagnostic.get("duplicate_identifiers"):
                counter["duplicate_identifiers"] += 1
                combined["duplicate_identifiers"] += 1
    output = {field: dict(sorted(counter.items())) for field, counter in output_counters.items()}
    output["all_identifier_fields"] = {
        key: combined.get(key, 0)
        for key in (
            "native_array",
            "json_array_string",
            "utf8_json_bytes",
            "missing",
            "malformed",
            "wrong_json_type",
            "duplicate_identifiers",
        )
    }
    output["locator_resolution_statuses"] = dict(sorted(locator_statuses.items()))
    output["decoder_statuses"] = dict(sorted(decoder_statuses.items()))
    return output


def _provenance_true(value: Any) -> bool:
    return value is True or (isinstance(value, (str, int)) and str(value).strip().casefold() in {"1", "true"})


def _alternative_identifier_inputs(row: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    fields = {
        "auth_seq_id_insertion": ("auth_residue_ids", "auth_residue_ids_match"),
        "label_seq_id": ("label_residue_ids", "label_residue_ids_match"),
        "position_zero_based": ("zero_based_residue_ids", "zero_based_positions_match"),
        "position_one_based": ("one_based_residue_ids", "one_based_positions_match"),
    }
    alternatives = {convention: row.get(field) for convention, (field, _) in fields.items()}
    authorized = {convention for convention, (_, provenance) in fields.items() if _provenance_true(row.get(provenance))}
    return alternatives, authorized


def project_full_corpus_eligibility(
    records: list[dict[str, Any]], split_diagnostics: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Project panel rates to the scanned corpus without claiming full adjudication."""
    total = sum(int(value.get("filtered_candidate_count", 0)) for value in split_diagnostics.values())
    eligible = sum(bool(row.get("rich_geometry_eligible")) for row in records)
    rate = eligible / max(len(records), 1)
    observed: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)
    for row in records:
        key = (
            str(row.get("split") or "unknown"),
            str(row.get("length_bin") or "unknown"),
            str(row.get("experimental_method") or "unknown"),
            str(row.get("pairing_classification") or "unknown"),
        )
        observed[key].append(bool(row.get("rich_geometry_eligible")))
    projected = 0.0
    unobserved_candidates = 0
    strata = []
    for split, diagnostics in sorted(split_diagnostics.items()):
        for item in diagnostics.get("per_stratum", []):
            key = (
                split,
                str(item["length_bin"]),
                str(item["experimental_method"]),
                str(item["pairing_classification"]),
            )
            available = int(item["available_count"])
            values = observed.get(key, [])
            stratum_rate = sum(values) / len(values) if values else rate
            unobserved_candidates += available if not values else 0
            projected += available * stratum_rate
            strata.append(
                {
                    "split": split,
                    "length_bin": key[1],
                    "experimental_method": key[2],
                    "pairing_classification": key[3],
                    "full_candidate_count": available,
                    "panel_observation_count": len(values),
                    "panel_eligible_fraction": stratum_rate,
                    "used_global_fallback": not values,
                }
            )
    projected_eligible = int(round(projected if strata else total * rate))
    return {
        "projection_only": True,
        "definitive_full_corpus_counts": False,
        "scanned_filtered_candidate_count": total,
        "audit_panel_sample_count": len(records),
        "panel_eligible_fraction": rate,
        "projected_rich_geometry_eligible_count": projected_eligible,
        "projected_rich_geometry_excluded_count": total - projected_eligible,
        "unobserved_stratum_candidate_count": unobserved_candidates,
        "stratum_projections": strata,
        "method": "panel stratum rates applied to scanned pairing strata; global panel rate only for unobserved strata",
    }


def eligibility_manifest_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the immutable Phase-0 decision surface without copying coordinates."""
    output = []
    for row in records:
        output.append(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "sample_id": row.get("sample_id"),
                "split": row.get("split"),
                "eligible": bool(row.get("rich_geometry_eligible", True)),
                "exclusion_reason": row.get("exclusion_reason"),
                "source_path": row.get("source_file"),
                "source_sha256": row.get("source_sha256"),
                "model_number": row.get("selected_model_number", row.get("model_number")),
                "chain_id": row.get("chain_id"),
                "sequence_sha256": row.get("sequence_sha256")
                or hashlib.sha256(str(row.get("sequence") or "").encode()).hexdigest(),
                "matrix_sha256": row.get("matrix_sha256"),
                "npz_path": row.get("npz_path", row.get("matrix_path")),
                "npz_sha256": row.get("npz_sha256", row.get("matrix_sha256")),
                "npz_metadata_sha256": row.get("npz_metadata_sha256"),
                "npz_residue_id_order_agreement": row.get("npz_residue_id_order_agreement"),
                "npz_residue_ids_json": row.get("npz_residue_ids_json"),
                "resolved_target_residue_ids_json": row.get("resolved_target_residue_ids_json"),
                "npz_internal_matrix_consistency": row.get("npz_internal_matrix_consistency"),
                "npz_internal_matrix_rmse_angstrom": row.get("npz_internal_matrix_rmse_angstrom"),
                "npz_internal_matrix_maximum_error_angstrom": row.get("npz_internal_matrix_maximum_error_angstrom"),
                "source_to_npz_calpha_anchor_consistency": row.get("source_to_npz_calpha_anchor_consistency"),
                "source_to_npz_calpha_coordinate_rmse_angstrom": row.get(
                    "source_to_npz_calpha_coordinate_rmse_angstrom"
                ),
                "source_to_npz_calpha_coordinate_maximum_error_angstrom": row.get(
                    "source_to_npz_calpha_coordinate_maximum_error_angstrom"
                ),
                "coordinate_anchor_policy_version": row.get(
                    "coordinate_anchor_policy_version", COORDINATE_ANCHOR_POLICY_VERSION
                ),
                "source_sha256_agreement": row.get("source_sha256_agreement"),
                "npz_residue_mask_json": row.get("npz_residue_mask_json"),
                "ca_candidate_counts_json": row.get("ca_candidate_counts_json"),
                "selected_calpha_conformers_json": row.get("selected_calpha_conformers_json"),
                "uniquely_anchored_residue_count": row.get("uniquely_anchored_residue_count", 0),
                "unavailable_calpha_anchor_count": row.get("unavailable_calpha_anchor_count", 0),
                "ambiguous_calpha_anchor_count": row.get("ambiguous_calpha_anchor_count", 0),
                "mapping_version": row.get("mapping_version", TARGET_MAPPING_POLICY_VERSION),
                "canonicalization_version": row.get("canonicalization_version", CANONICALIZATION_POLICY_VERSION),
                "pairing_canonicalization_version": row.get("pairing_canonicalization_version"),
                "chirality_version": row.get("chirality_version", CHIRALITY_CONVENTION_VERSION),
                "complete_local_frame_fraction": row.get("complete_local_frame_fraction"),
                "phi_computable_fraction": row.get("phi_computable_fraction"),
                "psi_computable_fraction": row.get("psi_computable_fraction"),
                "pseudo_cb_computable_fraction": row.get("pseudo_cb_computable_fraction"),
                "masked_missing_atom_residue_count": row.get("masked_missing_atom_residue_count", 0),
                "matrix_valid_pair_count": row.get("matrix_valid_pair_count", 0),
                "atom_availability_masks_json": row.get("atom_availability_masks_json"),
                "local_frame_mask_json": row.get("local_frame_mask_json"),
                "chain_continuity_mask_json": row.get("chain_continuity_mask_json"),
                "residue_ids_original_locator": row.get("residue_ids_original_locator"),
                "residue_ids_source_artifact": row.get("residue_ids_source_artifact"),
                "residue_ids_source_record_key": row.get("residue_ids_source_record_key"),
                "residue_ids_resolved_field_name": row.get("residue_ids_resolved_field_name"),
                "residue_ids_resolved_raw_type": row.get("residue_ids_resolved_raw_type"),
                "residue_ids_resolved_payload_sha256": row.get("residue_ids_resolved_payload_sha256"),
                "residue_ids_decoded_identifier_count": row.get("residue_ids_decoded_identifier_count"),
                "residue_ids_locator_resolution_status": row.get("residue_ids_locator_resolution_status"),
                "residue_ids_decoder_status": row.get("residue_ids_decoder_status"),
                "insertion_codes_original_locator": row.get("insertion_codes_original_locator"),
                "insertion_codes_source_artifact": row.get("insertion_codes_source_artifact"),
                "insertion_codes_source_record_key": row.get("insertion_codes_source_record_key"),
                "insertion_codes_resolved_field_name": row.get("insertion_codes_resolved_field_name"),
                "insertion_codes_resolved_raw_type": row.get("insertion_codes_resolved_raw_type"),
                "insertion_codes_resolved_payload_sha256": row.get("insertion_codes_resolved_payload_sha256"),
                "insertion_codes_decoded_identifier_count": row.get("insertion_codes_decoded_identifier_count"),
                "insertion_codes_locator_resolution_status": row.get("insertion_codes_locator_resolution_status"),
                "insertion_codes_decoder_status": row.get("insertion_codes_decoder_status"),
                "residue_id_fallback_used": bool(row.get("residue_id_fallback_used", False)),
            }
        )
    return output


class AuditHeartbeat:
    """Atomically expose bounded audit progress."""

    def __init__(self, path: Path, total: int) -> None:
        self.path = path
        self.total = int(total)

    def update(
        self,
        stage: str,
        processed: int,
        *,
        status: str = "running",
        total: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        stage_total = self.total if total is None else int(total)
        payload = {
            "status": status,
            "stage": stage,
            "processed": int(processed),
            "total": stage_total,
            "percentage": 100.0 * int(processed) / max(stage_total, 1),
            "heartbeat_utc": _utc_now(),
            **extra,
        }
        _atomic_json(self.path, payload)
        return payload


def aggregate_coverage(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row.get(field) or "unknown")].append(row)
    output = []
    for value, rows in sorted(groups.items()):
        output.append(
            {
                field: value,
                "sample_count": len(rows),
                "verified_source_fraction": float(
                    np.mean([bool(row.get("source_coordinates_verified")) for row in rows])
                ),
                "identity_agreement_fraction": float(
                    np.mean([float(row.get("residue_identity_match_fraction", 0)) for row in rows])
                ),
                "complete_frame_fraction": float(
                    np.mean([float(row.get("complete_local_frame_fraction", 0)) for row in rows])
                ),
                "phi_computable_fraction": float(
                    np.mean([float(row.get("phi_computable_fraction", 0)) for row in rows])
                ),
                "psi_computable_fraction": float(
                    np.mean([float(row.get("psi_computable_fraction", 0)) for row in rows])
                ),
                "omega_computable_fraction": float(
                    np.mean([float(row.get("omega_computable_fraction", 0)) for row in rows])
                ),
                "pseudo_cb_computable_fraction": float(
                    np.mean([float(row.get("pseudo_cb_computable_fraction", 0)) for row in rows])
                ),
                **{
                    f"{atom.lower()}_availability_fraction": float(
                        np.mean([float(row.get(f"{atom.lower()}_availability_fraction", 0)) for row in rows])
                    )
                    for atom in BACKBONE_ATOMS
                },
                "matrix_comparable_sample_count": sum(bool(row.get("matrix_comparable")) for row in rows),
                "matrix_rmse_angstrom_mean": float(
                    np.mean(
                        [
                            float(row["matrix_rmse_angstrom"])
                            for row in rows
                            if math.isfinite(float(row.get("matrix_rmse_angstrom", math.nan)))
                        ]
                        or [math.nan]
                    )
                ),
            }
        )
    return output


def publish_audit_outputs(
    output_dir: str | Path,
    *,
    records: list[dict[str, Any]],
    schema_inventory: dict[str, Any],
    protocol: dict[str, Any],
    failure_example_limit: int = 100,
) -> dict[str, Any]:
    """Atomically publish derived E006 tables and completed protocol."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    failures: dict[str, list[str]] = defaultdict(list)
    failure_totals = Counter()
    for row in records:
        reasons = list(row.get("unavailable_reasons", []))
        if row.get("exclusion_reason"):
            reasons.append(str(row["exclusion_reason"]))
        for reason in dict.fromkeys(reasons):
            failure_totals[str(reason)] += 1
            if len(failures[str(reason)]) < failure_example_limit:
                failures[str(reason)].append(str(row["sample_id"]))
    coverage = {
        "sample_count": len(records),
        "panel_observed_eligibility_counts": eligibility_counts(records),
        "full_corpus_eligibility_projection": protocol.get("full_corpus_eligibility_projection"),
        "failure_counts": dict(failure_totals),
        "storage_estimates": protocol["storage_estimates"],
        "phase1_authorization": protocol["phase1_authorization"],
    }
    _atomic_parquet(output / "sample_records.parquet", records)
    eligibility_path = output / "eligibility_manifest.parquet"
    _atomic_parquet(eligibility_path, eligibility_manifest_rows(records))
    _atomic_json(output / "coverage_summary.json", coverage)
    _atomic_json(output / "source_schema_inventory.json", schema_inventory)
    _atomic_json(
        output / "failure_examples.json",
        {
            "maximum_examples_per_category": failure_example_limit,
            "total_counts": dict(failure_totals),
            "examples": dict(failures),
        },
    )
    for field, filename in (
        ("experimental_method", "coverage_by_method.csv"),
        ("length_bin", "coverage_by_length.csv"),
        ("pairing_classification", "coverage_by_pairing_classification.csv"),
    ):
        _atomic_csv(output / filename, aggregate_coverage(records, field))
    completed = {
        **protocol,
        "eligibility_manifest": {
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
            "row_count": len(records),
            "schema_version": AUDIT_SCHEMA_VERSION,
        },
        "status": "completed",
        "failure": None,
        "authorizes_phase1": protocol["phase1_authorization"]["authorized"],
        "completed_utc": _utc_now(),
    }
    _atomic_json(output / "protocol.json", completed)
    return completed


def validate_audit_config(config: dict[str, Any]) -> None:
    if config.get("schema_version", AUDIT_SCHEMA_VERSION) != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {AUDIT_SCHEMA_VERSION}")
    if not 1 <= int(config.get("panel_size", 0)) <= 512:
        raise ValueError("panel_size must be in [1, 512]")
    if not 1 <= int(config.get("maximum_selected_structures", 0)) <= 512:
        raise ValueError("maximum_selected_structures must be in [1, 512]")
    if int(config["panel_size"]) > int(config["maximum_selected_structures"]):
        raise ValueError("panel_size exceeds maximum_selected_structures")
    if not 1 <= int(config.get("maximum_failure_examples", 0)) <= 100:
        raise ValueError("maximum_failure_examples must be in [1, 100]")
    if not 0 < float(config.get("maximum_rss_mib", 0)) <= 2048:
        raise ValueError("maximum_rss_mib must be in (0, 2048]")
    if float(config.get("pseudo_cb_maximum_distance_angstrom", 0.5)) < 0:
        raise ValueError("pseudo_cb_maximum_distance_angstrom must be non-negative")
    pseudo_angle = float(config.get("pseudo_cb_maximum_angle_degrees", 20.0))
    if not 0 <= pseudo_angle <= 180:
        raise ValueError("pseudo_cb_maximum_angle_degrees must be in [0, 180]")
    for field in ("npz_internal_matrix_tolerance_angstrom", "npz_calpha_anchor_tolerance_angstrom"):
        tolerance = float(config.get(field, 1e-4))
        if not 0 < tolerance <= 0.001:
            raise ValueError(f"{field} must be in (0, 0.001]")
    criteria = config.get("phase1_authorization", {})
    for key in (
        "minimum_verified_source_fraction",
        "minimum_mapped_identity_fraction",
        "minimum_complete_frame_fraction",
        "minimum_phi_psi_fraction",
    ):
        if not 0 <= float(criteria.get(key, -1)) <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if float(criteria.get("maximum_matrix_rmse_angstrom", -1)) < 0:
        raise ValueError("maximum_matrix_rmse_angstrom must be non-negative")
    if int(criteria.get("maximum_unexplained_contradictions", -1)) < 0:
        raise ValueError("maximum_unexplained_contradictions must be non-negative")


def write_failed_protocol(
    output_dir: str | Path,
    base: dict[str, Any],
    error: BaseException,
    *,
    failure_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = base.get("input_hashes_before", {})
    after = {path: sha256_file(path) if Path(path).is_file() else None for path in before}
    context = failure_context or {}
    payload = {
        **base,
        "status": "failed",
        "failure": {
            "stage": context.get("stage"),
            "sample_id": context.get("sample_id"),
            "field_name": context.get("field_name"),
            "type": type(error).__name__,
            "message": str(error)[:500],
            "bounded_context": {
                str(key): str(value)[:160]
                for key, value in context.items()
                if key not in {"stage", "sample_id", "field_name"}
            },
        },
        "authorizes_phase1": False,
        "current_rss_mib": _rss_mib(),
        "peak_rss_mib": _peak_rss_mib(),
        "input_hashes_after": after,
        "dataset_inputs_unchanged": bool(before) and before == after,
        "failed_utc": _utc_now(),
    }
    _atomic_json(Path(output_dir) / "protocol.json", payload)
    return payload


def run_e006_geometry_source_audit(config_path: str | Path) -> dict[str, Any]:
    """Run the real read-only audit; callers must explicitly invoke this command."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    validate_audit_config(config)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E006 source audit: {output}")
    scientific_roots = [Path(value).resolve() for value in config["scientific_input_roots"]]
    resolved_output = output.resolve()
    if any(resolved_output.is_relative_to(root) for root in scientific_roots):
        raise ValueError("output directory is inside a protected scientific-input root")
    output.mkdir(parents=True)
    started = time.monotonic()
    heartbeat = AuditHeartbeat(output / "heartbeat.json", int(config["panel_size"]))
    base = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "code_version": AUDIT_SCHEMA_VERSION,
        "chirality_convention_version": CHIRALITY_CONVENTION_VERSION,
        "pseudo_cb_convention_version": PSEUDO_CB_CONVENTION_VERSION,
        "target_mapping_policy_version": TARGET_MAPPING_POLICY_VERSION,
        "canonicalization_policy_version": CANONICALIZATION_POLICY_VERSION,
        "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
        "status": "running",
        "configuration_path": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "started_utc": _utc_now(),
        "failure": None,
        "authorizes_phase1": False,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "gemmi": importlib.metadata.version("gemmi"),
        },
    }
    processed_count = 0
    failure_context: dict[str, Any] = {"stage": "initialization", "sample_id": None, "field_name": None}
    try:
        failure_context["stage"] = "schema_inventory"
        heartbeat.update("schema_inventory", 0)
        sources = {
            "pairing_train": Path(config["pairing_dataset"]) / "eligible_train.parquet",
            "pairing_validation": Path(config["pairing_dataset"]) / "eligible_validation.parquet",
            "processed_manifest": Path(config["processed_manifest"]),
        }
        inventory = {name: parquet_schema_inventory(name, path) for name, path in sources.items()}
        inventory["audit_provenance"] = raw_audit_schema_inventory(config["audit_provenance"])
        protected_paths = {config_path.resolve()}
        for item in inventory.values():
            if "fragments" in item:
                protected_paths.update(_parquet_files(item))
        for table in inventory["audit_provenance"]["tables"].values():
            protected_paths.update(_parquet_files(table))
        protected_paths.update(Path(path) for path in inventory["audit_provenance"]["protocol_files"])
        for name in ("protocol.json", "schema.json", "vocabulary.json"):
            path = Path(config["pairing_dataset"]) / name
            if path.is_file():
                protected_paths.add(path.resolve())
        ordered_protected_paths = sorted(protected_paths, key=str)
        input_hashes_before = {}
        failure_context["stage"] = "input_hashing"
        for path_index, path in enumerate(ordered_protected_paths, start=1):
            input_hashes_before[str(path)] = sha256_file(path)
            heartbeat.update("input_hashing", path_index, total=len(ordered_protected_paths))
            enforce_rss_limit(float(config["maximum_rss_mib"]))
        base["input_hashes_before"] = input_hashes_before
        enforce_rss_limit(float(config["maximum_rss_mib"]))
        # Selection and locator reconciliation deliberately use only schema-selected columns.
        rows = []
        failure_context["stage"] = "panel_selection"
        split_diagnostics = {}
        train_target = int(config["panel_size"]) // 2
        for split, path in (("train", sources["pairing_train"]), ("validation", sources["pairing_validation"])):
            target = train_target if split == "train" else int(config["panel_size"]) - train_target
            selected, selection_diagnostics = scan_stratified_partition(
                path,
                split=split,
                target=target,
                seed=int(config["seed"]) + (0 if split == "train" else 1),
            )
            rows.extend(selected)
            split_diagnostics[split] = selection_diagnostics
        panel, panel_diagnostics = stratified_select(
            rows,
            target=int(config["panel_size"]),
            seed=int(config["seed"]),
            stratum_fields=("split", "length_bin", "experimental_method", "pairing_classification"),
        )
        panel_diagnostics["split_scans"] = split_diagnostics
        panel = enrich_source_locators(
            panel,
            processed_manifest=sources["processed_manifest"],
            audit_provenance=config["audit_provenance"],
        )
        heartbeat.update("source_validation", 0)
        records = []
        for index, row in enumerate(panel, start=1):
            audit_identifier_record = row.pop("_identifier_audit_source_record", None)
            audit_identifier_artifact = str(
                row.pop("_identifier_audit_source_artifact", "compact_audit:matrix_pair_alignments")
            )
            pairing_source_artifact = str(row.get("pairing_source_artifact") or "pairing_dataset")
            failure_context.update(stage="source_validation", sample_id=str(row.get("sample_id")), field_name=None)
            matrix_path = resolve_within_roots(row["matrix_path"], config["allowed_matrix_roots"])
            protected_paths.add(matrix_path)
            input_hashes_before[str(matrix_path)] = sha256_file(matrix_path)
            npz_evidence = load_npz_geometry_evidence(
                matrix_path,
                internal_tolerance_angstrom=float(config["npz_internal_matrix_tolerance_angstrom"]),
            )
            npz_inventory = npz_evidence.inventory
            matrix = npz_evidence.distance_matrix
            source_value = row.get("source_file")
            if not source_value:
                records.append(
                    {
                        **row,
                        "source_coordinates_verified": False,
                        "npz_path": str(matrix_path),
                        "npz_sha256": input_hashes_before[str(matrix_path)],
                        "matrix_sha256": input_hashes_before[str(matrix_path)],
                        "npz_metadata_sha256": npz_evidence.metadata_sha256,
                        "npz_internal_matrix_consistency": npz_evidence.internal_consistent,
                        "npz_internal_matrix_rmse_angstrom": npz_evidence.internal_rmse_angstrom,
                        "npz_internal_matrix_maximum_error_angstrom": (npz_evidence.internal_maximum_error_angstrom),
                        "npz_residue_mask_json": json.dumps(npz_evidence.residue_mask.tolist()),
                        "npz_keys_and_shapes_json": json.dumps(npz_inventory, sort_keys=True),
                        "unavailable_reasons": ["source_locator_unavailable"],
                        "unexplained_contradiction_count": 1,
                        "resolved_exclusion_count": 0,
                        "ignored_source_component_count": 0,
                        "masked_missing_atom_residue_count": 0,
                        "rich_geometry_eligible": False,
                        "eligibility_classification": "unexplained_source_resolution_failure",
                        "exclusion_reason": "unexplained_source_resolution_failure",
                    }
                )
                processed_count = index
                heartbeat.update("source_validation", index)
                enforce_rss_limit(float(config["maximum_rss_mib"]))
                continue
            structure_path = resolve_within_roots(source_value, config["allowed_structure_roots"])
            protected_paths.add(structure_path)
            input_hashes_before[str(structure_path)] = sha256_file(structure_path)
            actual_source_sha256 = input_hashes_before[str(structure_path)]
            try:
                residues, model_metadata = parse_backbone_mmcif(
                    structure_path,
                    chain_id=str(row.get("chain_id") or row.get("auth_asym_id") or row.get("label_asym_id")),
                    model_number=int(row.get("model_number") or 1),
                    residue_mappings=config.get("residue_mappings"),
                )
            except (RuntimeError, ValueError) as error:
                records.append(
                    {
                        **row,
                        "source_coordinates_verified": False,
                        "source_file": str(structure_path),
                        "source_sha256": input_hashes_before[str(structure_path)],
                        "npz_path": str(matrix_path),
                        "npz_sha256": input_hashes_before[str(matrix_path)],
                        "matrix_sha256": input_hashes_before[str(matrix_path)],
                        "npz_metadata_sha256": npz_evidence.metadata_sha256,
                        "npz_internal_matrix_consistency": npz_evidence.internal_consistent,
                        "npz_internal_matrix_rmse_angstrom": npz_evidence.internal_rmse_angstrom,
                        "npz_internal_matrix_maximum_error_angstrom": (npz_evidence.internal_maximum_error_angstrom),
                        "npz_residue_mask_json": json.dumps(npz_evidence.residue_mask.tolist()),
                        "npz_keys_and_shapes_json": json.dumps(npz_inventory, sort_keys=True),
                        "unavailable_reasons": [f"structure_or_selection_unavailable:{type(error).__name__}"],
                        "source_error_message": str(error),
                        "orientation_from_distances_inferred": False,
                        "unexplained_contradiction_count": 1,
                        "resolved_exclusion_count": 0,
                        "ignored_source_component_count": 0,
                        "masked_missing_atom_residue_count": 0,
                        "rich_geometry_eligible": False,
                        "eligibility_classification": "unexplained_source_resolution_failure",
                        "exclusion_reason": "unexplained_source_resolution_failure",
                    }
                )
                processed_count = index
                heartbeat.update("source_validation", index)
                enforce_rss_limit(float(config["maximum_rss_mib"]))
                continue
            convention = row.get("selected_residue_id_convention")
            if not convention:
                convention = next(
                    (
                        name
                        for field, name in (
                            ("auth_residue_ids_match", "auth_seq_id_insertion"),
                            ("label_residue_ids_match", "label_seq_id"),
                            ("zero_based_positions_match", "position_zero_based"),
                            ("one_based_positions_match", "position_one_based"),
                        )
                        if row.get(field)
                    ),
                    "",
                )
            alternative_identifiers, authorized_conventions = _alternative_identifier_inputs(row)
            failure_context.update(stage="target_identifier_mapping", field_name="residue_ids")
            mapping = resolve_verified_target_mapping(
                residues,
                sample_id=str(row["sample_id"]),
                expected_sequence=str(row["sequence"]),
                residue_ids=row.get("residue_ids"),
                insertion_codes=row.get("insertion_codes"),
                residue_id_convention=str(convention),
                pairing_canonicalization_version=(
                    str(row["modified_residue_mapping_version"])
                    if row.get("modified_residue_mapping_version")
                    else None
                ),
                alternative_residue_ids=alternative_identifiers,
                authorized_residue_id_conventions=authorized_conventions,
                audit_source_record=audit_identifier_record,
                audit_source_artifact=audit_identifier_artifact,
                direct_source_artifact=pairing_source_artifact,
            )
            all_model_residues = model_metadata.pop("_model_residues")
            model_target_mappings = [
                resolve_verified_target_mapping(
                    model_rows,
                    sample_id=str(row["sample_id"]),
                    expected_sequence=str(row["sequence"]),
                    residue_ids=row.get("residue_ids"),
                    insertion_codes=row.get("insertion_codes"),
                    residue_id_convention=str(convention),
                    pairing_canonicalization_version=(
                        str(row["modified_residue_mapping_version"])
                        if row.get("modified_residue_mapping_version")
                        else None
                    ),
                    alternative_residue_ids=alternative_identifiers,
                    authorized_residue_id_conventions=authorized_conventions,
                    audit_source_record=audit_identifier_record,
                    audit_source_artifact=audit_identifier_artifact,
                    direct_source_artifact=pairing_source_artifact,
                )
                for model_rows in all_model_residues.values()
            ]
            model_metadata["cross_model_sequence_consistent"] = bool(
                model_target_mappings
                and all(
                    item["mapping_verified"] and item["observed_target_sequence"] == str(row["sequence"])
                    for item in model_target_mappings
                )
            )
            model_metadata["target_model_sequences_json"] = json.dumps(
                {
                    model: item["observed_target_sequence"]
                    for model, item in zip(all_model_residues, model_target_mappings, strict=True)
                },
                sort_keys=True,
            )
            failure_context.update(stage="source_geometry_analysis", field_name=None)
            mapping["ignored_unmapped_source_component_count"] += int(
                model_metadata.get("ignored_source_component_count", 0)
            )
            target_residues = mapping.pop("target_residues")
            if mapping["mapping_verified"]:
                anchor = reconcile_npz_calpha_anchors(
                    target_residues,
                    resolved_target_residue_ids=list(mapping["resolved_target_residue_ids"]),
                    npz=npz_evidence,
                    anchor_tolerance_angstrom=float(config["npz_calpha_anchor_tolerance_angstrom"]),
                    scientific_tolerance_angstrom=float(config["phase1_authorization"]["maximum_matrix_rmse_angstrom"]),
                    maximum_examples=int(config["maximum_failure_examples"]),
                    expected_sample_id=str(row["sample_id"]),
                    expected_pdb_id=str(row.get("pdb_id") or ""),
                    expected_chain_id=str(row.get("chain_id") or ""),
                    expected_sequence=str(row["sequence"]),
                    expected_model_number=int(row.get("model_number") or 1),
                    expected_source_path=structure_path,
                    recorded_source_sha256=(str(row["source_sha256"]) if row.get("source_sha256") else None),
                    actual_source_sha256=actual_source_sha256,
                )
                anchored_residues = anchor.pop("_anchored_residues")
            else:
                anchored_residues = []
                anchor = {
                    "coordinate_anchor_exclusion_reason": None,
                    "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
                    "npz_residue_id_order_agreement": False,
                    "npz_internal_matrix_consistency": npz_evidence.internal_consistent,
                    "npz_internal_validation_errors_json": json.dumps(npz_evidence.validation_errors),
                    "npz_internal_matrix_rmse_angstrom": npz_evidence.internal_rmse_angstrom,
                    "npz_internal_matrix_maximum_error_angstrom": (npz_evidence.internal_maximum_error_angstrom),
                    "uniquely_anchored_residue_count": 0,
                    "source_to_npz_calpha_anchor_consistency": False,
                    "npz_metadata_sha256": npz_evidence.metadata_sha256,
                }
            mapping.update(anchor)
            analysis = analyze_backbone_residues(
                anchored_residues,
                expected_sequence=str(row["sequence"]),
                stored_matrix=matrix,
                peptide_bond_threshold_angstrom=float(config["peptide_bond_threshold_angstrom"]),
                pseudo_cb_maximum_distance_angstrom=float(config["pseudo_cb_maximum_distance_angstrom"]),
                pseudo_cb_maximum_angle_degrees=float(config["pseudo_cb_maximum_angle_degrees"]),
            )
            eligibility = resolve_rich_geometry_eligibility(
                mapping,
                analysis,
                cross_model_sequence_consistent=bool(model_metadata["cross_model_sequence_consistent"]),
            )
            comparison = analysis.pop("matrix_comparison")
            records.append(
                {
                    **row,
                    **analysis,
                    **mapping,
                    **eligibility,
                    **model_metadata,
                    "source_coordinates_verified": True,
                    "source_file": str(structure_path),
                    "source_sha256": actual_source_sha256,
                    "npz_path": str(matrix_path),
                    "npz_sha256": input_hashes_before[str(matrix_path)],
                    "matrix_sha256": input_hashes_before[str(matrix_path)],
                    "npz_keys_and_shapes_json": json.dumps(npz_inventory, sort_keys=True),
                    "npz_residue_mask_json": json.dumps(npz_evidence.residue_mask.tolist()),
                    "matrix_comparable": comparison["comparable"],
                    "matrix_valid_pair_count": comparison["valid_pair_count"],
                    "matrix_maximum_error_angstrom": comparison["maximum_error_angstrom"],
                    "matrix_rmse_angstrom": comparison["rmse_angstrom"],
                }
            )
            processed_count = index
            heartbeat.update("source_validation", index)
            enforce_rss_limit(float(config["maximum_rss_mib"]))
        failure_context.update(stage="aggregation", sample_id=None, field_name=None)
        authorization = phase1_authorization(records, config["phase1_authorization"])
        observed_counts = eligibility_counts(records)
        full_projection = project_full_corpus_eligibility(records, split_diagnostics)
        identifier_inventory = identifier_representation_inventory(records)
        inventory["target_identifier_representations"] = identifier_inventory
        npz_schemas = Counter(str(row.get("npz_keys_and_shapes_json") or "unavailable") for row in records)
        inventory["selected_npz_schemas"] = [
            {"keys_and_shapes_json": key, "sample_count": count} for key, count in sorted(npz_schemas.items())
        ]
        storage_config = config["storage_estimation"]
        storage_estimates = {
            "representative_samples": sidecar_storage_estimates(
                [64, 128, 256, 384, 500],
                pair_channels=int(storage_config["pair_channels"]),
                residue_channels=int(storage_config["residue_channels"]),
                compression_ratio=float(storage_config["compression_ratio"]),
            ),
            "full_eligible_training_corpus_by_length": full_corpus_storage_estimates(
                split_diagnostics["train"]["length_statistics"],
                pair_channels=int(storage_config["pair_channels"]),
                residue_channels=int(storage_config["residue_channels"]),
                compression_ratio=float(storage_config["compression_ratio"]),
            ),
            "compression_is_planning_assumption": True,
        }
        protocol = {
            **base,
            "panel_selection": panel_diagnostics,
            "input_hashes_before": input_hashes_before,
            "storage_estimates": storage_estimates,
            "phase1_authorization": authorization,
            "panel_observed_eligibility_counts": observed_counts,
            "full_corpus_eligibility_projection": full_projection,
            "target_identifier_representation_inventory": identifier_inventory,
            "elapsed_seconds": time.monotonic() - started,
            "current_rss_mib": _rss_mib(),
            "peak_rss_mib": _peak_rss_mib(),
        }
        input_hashes_after = {path: sha256_file(path) for path in input_hashes_before}
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("scientific_input_mutation_detected")
        protocol.update(input_hashes_after=input_hashes_after, dataset_inputs_unchanged=True)
        failure_context["stage"] = "report_publication"
        completed = publish_audit_outputs(
            output,
            records=records,
            schema_inventory=inventory,
            protocol=protocol,
            failure_example_limit=int(config["maximum_failure_examples"]),
        )
        protocol_path = output / "protocol.json"
        heartbeat.update(
            "finalization",
            len(records),
            status="completed",
            completed_utc=_utc_now(),
            protocol_path=str(protocol_path),
            protocol_sha256=sha256_file(protocol_path),
        )
        return completed
    except BaseException as error:
        heartbeat.update(
            "failed",
            processed_count,
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error)[:500],
            sample_id=failure_context.get("sample_id"),
            field_name=failure_context.get("field_name"),
        )
        base["elapsed_seconds"] = time.monotonic() - started
        write_failed_protocol(output, base, error, failure_context=failure_context)
        raise
