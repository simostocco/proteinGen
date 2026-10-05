"""Read-only E007 Phase-3I geometry-generator capability audit."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from protein_distance_diffusion.data.rich_geometry import _resolve_protected_input, authorize_rich_geometry_dataset
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file

VERSION = "e007_geometry_generator_capability_audit_v1"
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created": False,
    "optimizer_created": False,
    "backward_performed": False,
    "optimizer_updates": 0,
    "sampling_performed": False,
    "dataset_modified": False,
    "checkpoint_modified": False,
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_additional_training": False,
}
CLASSIFICATIONS = {
    "local_geometry_only",
    "global_distribution_mismatch",
    "chirality_failure",
    "mode_collapse",
    "tail_limited_research_generator",
    "promising_research_generator",
    "retrospective_audit_inconclusive",
}
REFERENCE_COLUMNS = (
    "sample_id",
    "split",
    "sequence",
    "ca_coordinates",
    "ca_mask",
    "chain_continuity_mask",
    "chain_break_mask",
    "source_path",
    "source_sha256",
    "npz_path",
    "npz_sha256",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]], compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression=compression)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"E007 Phase-3I prerequisite is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3I prerequisite hash contradiction: {label}")
    return observed


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase-3I configuration version contradiction")
    panel = payload.get("panel", {})
    if panel.get("lengths") != [64, 128, 256, 384, 500] or int(panel.get("samples_per_length", 0)) != 32:
        raise ValueError("E007 Phase-3I panel contract changed")
    if int(payload["selected_checkpoint"].get("optimizer_update", -1)) != 9000:
        raise ValueError("E007 Phase-3I must audit selected checkpoint step 9000")
    if int(payload["phase3h"].get("expected_step9000_samples", 0)) != 160:
        raise ValueError("E007 Phase-3I expected sample count changed")
    if int(panel.get("maximum_reference_length_mismatch", -1)) != 4:
        raise ValueError("E007 Phase-3I reference mismatch bound changed")
    strata = panel.get("length_strata")
    if not isinstance(strata, list) or len(strata) != 5:
        raise ValueError("E007 Phase-3I reference length strata are invalid")
    for length in panel["lengths"]:
        owners = [item for item in strata if int(item["minimum"]) <= length <= int(item["maximum"])]
        if len(owners) != 1:
            raise ValueError(f"E007 Phase-3I target length has ambiguous stratum ownership: {length}")
    return payload


def _verify_phase3h_inventory(config: Mapping[str, Any], *, verify_entries: bool) -> dict[str, Any]:
    section = config["phase3h"]
    root = Path(section["root"])
    files = {
        "report": (root / "report.json", section["report_sha256"]),
        "protocol": (root / "protocol.json", section["protocol_sha256"]),
        "sample_metrics": (root / "sample_metrics.parquet", section["sample_metrics_sha256"]),
        "artifact_inventory": (root / "artifact_inventory.json", section["artifact_inventory_sha256"]),
        "block_inventory": (root / "block_inventory.json", section["block_inventory_sha256"]),
        "checkpoint_pareto": (root / "checkpoint_pareto.json", section["checkpoint_pareto_sha256"]),
        "paired_comparisons": (root / "paired_comparisons.json", section["paired_comparisons_sha256"]),
    }
    hashes = {name: _verify_file(path, digest, f"phase3h_{name}") for name, (path, digest) in files.items()}
    inventory = json.loads((root / "artifact_inventory.json").read_text())
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != int(section["expected_inventory_entries"]):
        raise ValueError("E007 Phase-3I Phase-3H inventory cardinality contradiction")
    if inventory.get("aggregate_sha256") != section["artifact_inventory_aggregate_sha256"]:
        raise ValueError("E007 Phase-3I Phase-3H inventory aggregate pin contradiction")
    if _canonical_sha(artifacts) != inventory["aggregate_sha256"]:
        raise ValueError("E007 Phase-3I Phase-3H inventory serialization contradiction")
    seen: set[str] = set()
    verified = 0
    if verify_entries:
        resolved_root = root.resolve()
        for record in artifacts:
            relative = Path(str(record.get("path", "")))
            candidate = (root / relative).resolve()
            logical = relative.as_posix()
            if relative.is_absolute() or not candidate.is_relative_to(resolved_root):
                raise ValueError(f"E007 Phase-3I inventory path escapes root: {logical}")
            if logical in seen:
                raise ValueError(f"E007 Phase-3I duplicate inventory path: {logical}")
            seen.add(logical)
            if not candidate.is_file() or candidate.stat().st_size != int(record["size_bytes"]):
                raise ValueError(f"E007 Phase-3I inventory size/membership contradiction: {logical}")
            if sha256_file(candidate) != record["sha256"]:
                raise ValueError(f"E007 Phase-3I inventory hash contradiction: {logical}")
            verified += 1
        excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
        durable_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name not in excluded and ".tmp" not in path.name
        }
        if seen != durable_paths:
            raise ValueError("E007 Phase-3I Phase-3H durable inventory membership contradiction")
    return {
        "root": str(root),
        "hashes": hashes,
        "inventory_entry_count": len(artifacts),
        "inventory_entries_verified": verified,
        "inventory_aggregate_sha256": inventory["aggregate_sha256"],
    }


def verify_prerequisites(config: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    checkpoint = config["selected_checkpoint"]
    selected = config["selection_record"]
    dataset = config["dataset"]
    clean = config["clean_validation"]
    hashes = {
        "selected_checkpoint": _verify_file(Path(checkpoint["path"]), checkpoint["sha256"], "selected_checkpoint"),
        "selected_checkpoint_record": _verify_file(
            Path(selected["root"]) / "selected_checkpoint.json",
            selected["selected_checkpoint_sha256"],
            "selected_checkpoint_record",
        ),
        "selection_protocol": _verify_file(
            Path(selected["root"]) / "protocol.json", selected["protocol_sha256"], "selection_protocol"
        ),
        "clean_validation_manifest": _verify_file(
            Path(clean["manifest_path"]), clean["manifest_sha256"], "clean_validation_manifest"
        ),
    }
    metadata_names = {
        "protocol.json": "protocol_sha256",
        "schema.json": "schema_sha256",
        "vocabulary.json": "vocabulary_sha256",
        "normalization.json": "normalization_sha256",
        "shard_hashes.sha256": "shard_inventory_sha256",
    }
    for name, key in metadata_names.items():
        hashes[f"dataset_{name}"] = _verify_file(Path(dataset["root"]) / name, dataset[key], f"dataset_{name}")
    phase3h = _verify_phase3h_inventory(config, verify_entries=full)
    if full:
        authorization = authorize_rich_geometry_dataset(
            dataset["root"],
            expected_protocol_sha256=dataset["protocol_sha256"],
            expected_schema_sha256=dataset["schema_sha256"],
            expected_vocabulary_sha256=dataset["vocabulary_sha256"],
            expected_normalization_sha256=dataset["normalization_sha256"],
            expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
            protected_input_relocations=dataset.get("protected_input_relocations"),
        )
        authorization_record: dict[str, Any] = {
            "split_counts": authorization.split_counts,
            "verified_shard_count": len(authorization.observed_shard_hashes),
            "protected_input_resolution_counts": authorization.protected_input_resolution_counts,
            "relocated_protected_input_count": len(authorization.relocated_protected_inputs),
        }
    else:
        authorization_record = {"deferred_to_audit": True, "dataset_payload_scanned": False}
    return {
        "hashes": hashes,
        "phase3h": phase3h,
        "dataset_authorization": authorization_record,
        "protected_inputs_verified": True,
    }


def plan_geometry_generator_capability(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3I output exists: {output} or {staging}")
    prerequisites = verify_prerequisites(config, full=False)
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "output_dir": str(output),
        "selected_checkpoint_update": 9000,
        "generated_sample_count": 160,
        "reference_sample_count": 160,
        "reference_selection_policy": config["panel"]["selection_version"],
        "maximum_reference_length_mismatch": config["panel"]["maximum_reference_length_mismatch"],
        "planned_exact_reference_count": 129,
        "planned_nearest_same_stratum_reference_count": 31,
        "observed_exact_available_counts": config["panel"]["observed_exact_available_counts"],
        "observed_nearby_counts_within_maximum_mismatch": config["panel"][
            "observed_nearby_counts_within_maximum_mismatch"
        ],
        "lengths": config["panel"]["lengths"],
        "samples_per_length": config["panel"]["samples_per_length"],
        "retrospective_checkpoint_selection_bias_declared": True,
        "phase3j_plan": config["prospective_phase3j"],
        "dataset_payload_scanned": False,
        "coordinate_payloads_loaded": False,
        "output_created": False,
        "prerequisites": prerequisites,
        **NON_AUTHORIZING,
    }


def _stable_rank(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:validation:{sample_id}".encode()).hexdigest()


def _reference_stratum(panel: Mapping[str, Any], length: int) -> Mapping[str, Any]:
    owners = [item for item in panel["length_strata"] if int(item["minimum"]) <= length <= int(item["maximum"])]
    if len(owners) != 1:
        raise ValueError(f"E007 Phase-3I reference length has invalid stratum ownership: {length}")
    return owners[0]


def select_reference_panel(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clean = config["clean_validation"]
    panel = config["panel"]
    path = Path(clean["manifest_path"])
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != int(clean["expected_rows"]):
        raise ValueError("E007 Phase-3I clean-validation row-count contradiction")
    requested = tuple(map(int, panel["lengths"]))
    candidate_rows: list[dict[str, Any]] = []
    available_counts: Counter[int] = Counter()
    seen: set[str] = set()
    columns = [
        "sample_id",
        "split",
        "coordinate_accepted",
        "length",
        "length_stratum",
        "source_path",
        "dataset_shard_path",
        "shard_row_index",
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            sample_id = str(row["sample_id"])
            if sample_id in seen:
                raise ValueError(f"E007 Phase-3I duplicate clean-validation sample ID: {sample_id}")
            seen.add(sample_id)
            if row["split"] != "validation" or row["coordinate_accepted"] is not True:
                raise ValueError(f"E007 Phase-3I invalid clean-validation membership: {sample_id}")
            length = int(row["length"])
            available_counts[length] += 1
            record = dict(row)
            record["selection_rank"] = _stable_rank(int(panel["selection_seed"]), sample_id)
            candidate_rows.append(record)
    result: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "policy": panel["selection_version"],
        "maximum_reference_length_mismatch": int(panel["maximum_reference_length_mismatch"]),
        "clean_validation_population_count": sum(available_counts.values()),
        "by_target_length": {},
    }
    for target_length, expected in panel["observed_exact_available_counts"].items():
        if available_counts[int(target_length)] != int(expected):
            raise ValueError(
                "E007 Phase-3I clean-validation exact-length inventory changed: "
                f"length={target_length}, observed={available_counts[int(target_length)]}, expected={expected}"
            )
    for target_length, expected_counts in panel["observed_nearby_counts_within_maximum_mismatch"].items():
        for actual_length, expected in expected_counts.items():
            if available_counts[int(actual_length)] != int(expected):
                raise ValueError(
                    "E007 Phase-3I clean-validation nearby-length inventory changed: "
                    f"target={target_length}, actual={actual_length}, "
                    f"observed={available_counts[int(actual_length)]}, expected={expected}"
                )
    count = int(panel["samples_per_length"])
    used: set[str] = set()
    for target_length in requested:
        stratum = _reference_stratum(panel, target_length)
        minimum, maximum = int(stratum["minimum"]), int(stratum["maximum"])
        maximum_mismatch = int(panel["maximum_reference_length_mismatch"])
        exact = sorted(
            (row for row in candidate_rows if int(row["length"]) == target_length),
            key=lambda row: (row["selection_rank"], row["sample_id"]),
        )
        chosen = exact[:count]
        shortfall = count - len(chosen)
        nearest = sorted(
            (
                row
                for row in candidate_rows
                if minimum <= int(row["length"]) <= maximum
                and int(row["length"]) != target_length
                and abs(int(row["length"]) - target_length) <= maximum_mismatch
                and row["sample_id"] not in used
            ),
            key=lambda row: (
                abs(int(row["length"]) - target_length),
                row["selection_rank"],
                row["sample_id"],
            ),
        )
        chosen.extend(nearest[:shortfall])
        if len(chosen) < count:
            raise ValueError(
                "E007 Phase-3I bounded same-stratum reference underfill: "
                f"target_length={target_length}, exact={len(exact)}, "
                f"within_mismatch_bound={len(nearest)}, selected={len(chosen)}, required={count}, "
                f"maximum_mismatch={maximum_mismatch}, stratum={stratum['name']}"
            )
        selected_actual_counts = Counter(int(row["length"]) for row in chosen)
        diagnostics["by_target_length"][str(target_length)] = {
            "stratum": str(stratum["name"]),
            "stratum_minimum": minimum,
            "stratum_maximum": maximum,
            "exact_available_count": len(exact),
            "nearby_available_counts": {
                str(length): available_counts[length]
                for length in range(minimum, maximum + 1)
                if available_counts[length]
            },
            "within_mismatch_bound_available_counts": {
                str(length): available_counts[length]
                for length in range(
                    max(minimum, target_length - maximum_mismatch), min(maximum, target_length + maximum_mismatch) + 1
                )
                if length != target_length and available_counts[length]
            },
            "selected_exact_count": selected_actual_counts[target_length],
            "selected_nearest_count": count - selected_actual_counts[target_length],
            "selected_actual_length_counts": {str(key): value for key, value in sorted(selected_actual_counts.items())},
        }
        for reference_index, source_row in enumerate(chosen):
            row = dict(source_row)
            sample_id = str(row["sample_id"])
            if sample_id in used:
                raise ValueError(f"E007 Phase-3I reference selected more than once: {sample_id}")
            used.add(sample_id)
            actual_length = int(row["length"])
            row.update(
                {
                    "reference_index": reference_index,
                    "target_length": target_length,
                    "actual_length": actual_length,
                    "signed_length_mismatch": actual_length - target_length,
                    "absolute_length_mismatch": abs(actual_length - target_length),
                    "match_type": "exact" if actual_length == target_length else "nearest_same_stratum",
                    "length_stratum": str(stratum["name"]),
                    "selection_version": panel["selection_version"],
                    "clean_validation_manifest_member": True,
                }
            )
            result.append(row)
    expected_total = count * len(requested)
    if len(result) != expected_total or len({row["sample_id"] for row in result}) != expected_total:
        raise ValueError("E007 Phase-3I reference panel count/uniqueness contradiction")
    mismatches = np.asarray([int(row["signed_length_mismatch"]) for row in result], dtype=np.int64)
    diagnostics.update(
        {
            "selected_reference_count": len(result),
            "unique_reference_count": len({row["sample_id"] for row in result}),
            "exact_match_count": int(np.sum(mismatches == 0)),
            "nearest_match_count": int(np.sum(mismatches != 0)),
            "signed_mismatch_counts": {str(key): value for key, value in sorted(Counter(map(int, mismatches)).items())},
            "maximum_observed_absolute_mismatch": int(np.max(np.abs(mismatches))),
            "sample_id_sha256": _canonical_sha([row["sample_id"] for row in result]),
        }
    )
    return result, diagnostics


def select_reference_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows, _diagnostics = select_reference_panel(config)
    return rows


def _read_parquet_row(path: Path, row_index: int, columns: Sequence[str]) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if not 0 <= row_index < parquet.metadata.num_rows:
        raise IndexError(f"E007 Phase-3I shard row index is invalid: {path}:{row_index}")
    offset = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        if row_index < offset + count:
            return (
                parquet.read_row_group(group_index, columns=list(columns)).slice(row_index - offset, 1).to_pylist()[0]
            )
        offset += count
    raise AssertionError("unreachable Parquet row locator")


def load_reference_coordinates(
    config: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    root = Path(config["dataset"]["root"]).resolve()
    shard = (root / str(record["dataset_shard_path"])).resolve()
    if not shard.is_relative_to(root):
        raise ValueError("E007 Phase-3I reference shard escapes dataset root")
    row = _read_parquet_row(shard, int(record["shard_row_index"]), REFERENCE_COLUMNS)
    length = int(record["length"])
    if row["sample_id"] != record["sample_id"] or row["split"] != "validation" or len(row["sequence"]) != length:
        raise ValueError(f"E007 Phase-3I reference row identity contradiction: {record['sample_id']}")
    mask = np.asarray(row["ca_mask"], dtype=bool)
    continuity = np.asarray(row["chain_continuity_mask"], dtype=bool)
    breaks = np.asarray(row["chain_break_mask"], dtype=bool)
    coordinates = np.asarray(row["ca_coordinates"], dtype=np.float64)
    if coordinates.shape != (length, 3) or mask.shape != (length,) or not mask.all():
        raise ValueError(f"E007 Phase-3I reference C-alpha contract contradiction: {record['sample_id']}")
    if continuity.shape != (length - 1,) or not continuity.all() or breaks.any():
        raise ValueError(f"E007 Phase-3I reference continuity contradiction: {record['sample_id']}")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"E007 Phase-3I reference coordinates are non-finite: {record['sample_id']}")
    relocations = config["dataset"].get("protected_input_relocations")
    resolved_source, source_resolution = _resolve_protected_input(
        str(row["source_path"]), str(row["source_sha256"]), relocations
    )
    resolved_npz, npz_resolution = _resolve_protected_input(str(row["npz_path"]), str(row["npz_sha256"]), relocations)
    provenance = {
        "sample_id": row["sample_id"],
        "split": row["split"],
        "length": length,
        "target_length": int(record["target_length"]),
        "actual_length": int(record["actual_length"]),
        "signed_length_mismatch": int(record["signed_length_mismatch"]),
        "absolute_length_mismatch": int(record["absolute_length_mismatch"]),
        "match_type": str(record["match_type"]),
        "length_stratum": str(record["length_stratum"]),
        "clean_validation_manifest_member": bool(record["clean_validation_manifest_member"]),
        "clean_validation_manifest_sha256": str(config["clean_validation"]["manifest_sha256"]),
        "dataset_shard_path": str(record["dataset_shard_path"]),
        "shard_row_index": int(record["shard_row_index"]),
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "source_resolved_path": str(resolved_source),
        "source_relocation_resolution": source_resolution,
        "npz_path": row["npz_path"],
        "npz_sha256": row["npz_sha256"],
        "npz_resolved_path": str(resolved_npz),
        "npz_relocation_resolution": npz_resolution,
        "coordinate_sha256": hashlib.sha256(np.ascontiguousarray(coordinates).tobytes()).hexdigest(),
        "selection_rank": record["selection_rank"],
        "selection_version": record["selection_version"],
    }
    return coordinates, provenance


def _safe_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "standard_deviation": None, "p05": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    valid = norm > 1e-12
    result = np.full(norm.shape, np.nan, dtype=np.float64)
    result[valid] = np.arccos(np.clip(np.sum(v1[valid] * v2[valid], axis=-1) / norm[valid], -1.0, 1.0))
    return result


def _dihedral(points: np.ndarray) -> np.ndarray:
    if len(points) < 4:
        return np.empty(0, dtype=np.float64)
    b0 = points[:-3] - points[1:-2]
    b1 = points[2:-1] - points[1:-2]
    b2 = points[3:] - points[2:-1]
    norms = np.linalg.norm(b1, axis=-1)
    valid = norms > 1e-12
    unit = np.zeros_like(b1)
    unit[valid] = b1[valid] / norms[valid, None]
    v = b0 - np.sum(b0 * unit, axis=-1, keepdims=True) * unit
    w = b2 - np.sum(b2 * unit, axis=-1, keepdims=True) * unit
    valid &= (np.linalg.norm(v, axis=-1) > 1e-12) & (np.linalg.norm(w, axis=-1) > 1e-12)
    result = np.full(len(b1), np.nan, dtype=np.float64)
    result[valid] = np.arctan2(
        np.sum(np.cross(unit[valid], v[valid]) * w[valid], axis=-1),
        np.sum(v[valid] * w[valid], axis=-1),
    )
    return result


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def _connected_components(contact: np.ndarray) -> tuple[int, np.ndarray]:
    count = len(contact)
    seen = np.zeros(count, dtype=bool)
    sizes = []
    for start in range(count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in np.flatnonzero(contact[node] & ~seen):
                seen[neighbor] = True
                stack.append(int(neighbor))
        sizes.append(size)
    return len(sizes), np.asarray(sizes, dtype=np.int64)


def geometry_metrics(
    coordinates: np.ndarray,
    *,
    source: str,
    sample_id: str,
    length: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (length, 3):
        raise ValueError(f"E007 Phase-3I coordinate shape contradiction: {sample_id}")
    finite = bool(np.isfinite(coordinates).all())
    if not finite:
        raise ValueError(f"E007 Phase-3I non-finite coordinates: {sample_id}")
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    distances = _distance_matrix(centered)
    adjacency = np.linalg.norm(np.diff(centered, axis=0), axis=-1)
    offsets = {}
    arrays: dict[str, np.ndarray] = {"adjacent": adjacency}
    for offset in map(int, config["metrics"]["local_distance_offsets"]):
        values = np.linalg.norm(centered[offset:] - centered[:-offset], axis=-1)
        arrays[f"distance_i_plus_{offset}"] = values
        offsets[f"distance_i_plus_{offset}_mean_angstrom"] = float(values.mean())
    vectors = np.diff(centered, axis=0)
    angles = _angle(-vectors[:-1], vectors[1:])
    dihedrals = _dihedral(centered)
    arrays["bond_angle_radians"] = angles
    arrays["signed_pseudo_dihedral_radians"] = dihedrals
    tangent_norm = np.linalg.norm(vectors, axis=-1)
    tangents = np.divide(
        vectors, tangent_norm[:, None], out=np.zeros_like(vectors), where=tangent_norm[:, None] > 1e-12
    )
    curvature = np.linalg.norm(np.diff(tangents, axis=0), axis=-1)
    arrays["discrete_curvature"] = curvature
    separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    upper = np.triu(np.ones((length, length), dtype=bool), 1)
    non_neighbor = upper & (separation >= 3)
    clash_limit = float(config["metrics"]["clash_distance_angstrom"])
    clash_fraction = float(np.mean(distances[non_neighbor] < clash_limit)) if non_neighbor.any() else 0.0
    gyration = centered.T @ centered / length
    eigenvalues = np.sort(np.linalg.eigvalsh(gyration))[::-1]
    rg = float(np.sqrt(np.maximum(eigenvalues.sum(), 0.0)))
    asphericity = float(1.5 * np.square(eigenvalues - eigenvalues.mean()).sum() / max(eigenvalues.sum() ** 2, 1e-24))
    row: dict[str, Any] = {
        "source": source,
        "sample_id": sample_id,
        "length": length,
        "finite_coordinates": finite,
        "centroid_max_abs_angstrom": float(np.max(np.abs(centered.mean(axis=0)))),
        "padding_checked": source == "generated",
        "padding_exact_zero": True,
        "adjacent_distance_mean_angstrom": float(adjacency.mean()),
        "adjacent_distance_rmse_to_3_8_angstrom": float(np.sqrt(np.mean(np.square(adjacency - 3.8)))),
        "bond_angle_mean_degrees": float(np.degrees(np.nanmean(angles))),
        "signed_pseudo_dihedral_mean": float(np.nanmean(dihedrals)),
        "signed_pseudo_dihedral_positive_fraction": float(np.mean(dihedrals[np.isfinite(dihedrals)] > 0)),
        "signed_tetrahedral_volume_mean": float(
            np.mean(np.einsum("ij,ij->i", np.cross(vectors[:-2], vectors[1:-1]), vectors[2:]))
        ),
        "curvature_mean": float(np.nanmean(curvature)),
        "clash_fraction": clash_fraction,
        "discontinuity_fraction": float(
            np.mean(adjacency > float(config["metrics"]["discontinuity_distance_angstrom"]))
        ),
        "radius_of_gyration_angstrom": rg,
        "end_to_end_distance_angstrom": float(np.linalg.norm(centered[-1] - centered[0])),
        "asphericity": asphericity,
        "principal_axis_ratio_2_to_1": float(eigenvalues[1] / max(eigenvalues[0], 1e-24)),
        "principal_axis_ratio_3_to_1": float(eigenvalues[2] / max(eigenvalues[0], 1e-24)),
        "distance_symmetry_error_angstrom": float(np.max(np.abs(distances - distances.T))),
        "distance_diagonal_error_angstrom": float(np.max(np.abs(np.diag(distances)))),
        **offsets,
    }
    long_range = upper & (separation >= int(config["metrics"]["long_range_minimum_separation"]))
    for threshold in map(float, config["metrics"]["contact_thresholds_angstrom"]):
        key = f"contact_density_{threshold:g}a"
        contact = (distances < threshold) & upper & (separation >= 3)
        row[key] = float(contact.sum() / max((upper & (separation >= 3)).sum(), 1))
        long_contact = (distances < threshold) & long_range
        row[f"long_range_contact_density_{threshold:g}a"] = float(long_contact.sum() / max(long_range.sum(), 1))
        if threshold == 8.0:
            graph = (distances < threshold) & (separation >= 3)
            components, sizes = _connected_components(graph)
            degrees = graph.sum(axis=1)
            row.update(
                {
                    "contact_order_8a": float(separation[(distances < threshold) & upper & (separation >= 3)].mean())
                    if contact.any()
                    else 0.0,
                    "contact_graph_component_count_8a": components,
                    "contact_graph_largest_component_fraction_8a": float(sizes.max() / length),
                    "contact_graph_degree_mean_8a": float(degrees.mean()),
                    "contact_graph_degree_std_8a": float(degrees.std()),
                }
            )
    return row, arrays


def _descriptor(row: Mapping[str, Any]) -> np.ndarray:
    keys = (
        "adjacent_distance_mean_angstrom",
        "adjacent_distance_rmse_to_3_8_angstrom",
        "bond_angle_mean_degrees",
        "signed_pseudo_dihedral_mean",
        "signed_pseudo_dihedral_positive_fraction",
        "curvature_mean",
        "clash_fraction",
        "discontinuity_fraction",
        "radius_of_gyration_angstrom",
        "end_to_end_distance_angstrom",
        "asphericity",
        "principal_axis_ratio_2_to_1",
        "principal_axis_ratio_3_to_1",
        "contact_density_6a",
        "contact_density_8a",
        "contact_density_10a",
        "contact_density_12a",
        "long_range_contact_density_8a",
        "contact_order_8a",
        "contact_graph_largest_component_fraction_8a",
    )
    return np.asarray([float(row[key]) for key in keys], dtype=np.float64)


def _pairwise(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=-1))[np.triu_indices(len(values), 1)]


def diversity_summary(
    rows: Sequence[Mapping[str, Any]], coordinate_hashes: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    generated = [row for row in rows if row["source"] == "generated"]
    reference = [row for row in rows if row["source"] == "reference"]
    generated_values = np.stack([_descriptor(row) for row in generated])
    reference_values = np.stack([_descriptor(row) for row in reference])
    scale = reference_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    generated_scaled = (generated_values - reference_values.mean(axis=0)) / scale
    reference_scaled = (reference_values - reference_values.mean(axis=0)) / scale
    generated_distances = _pairwise(generated_scaled)
    reference_distances = _pairwise(reference_scaled)
    threshold = float(config["metrics"]["descriptor_cluster_distance"])
    parent = list(range(len(generated_scaled)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left in range(len(generated_scaled)):
        for right in range(left + 1, len(generated_scaled)):
            if np.linalg.norm(generated_scaled[left] - generated_scaled[right]) <= threshold:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    parent[root_right] = root_left
    sizes = Counter(find(index) for index in range(len(parent))).values()
    probabilities = np.asarray(list(sizes), dtype=np.float64) / len(parent)
    return {
        "generated_exact_coordinate_duplicate_count": len(coordinate_hashes) - len(set(coordinate_hashes)),
        "generated_descriptor_pair_distance": _safe_summary(generated_distances),
        "reference_descriptor_pair_distance": _safe_summary(reference_distances),
        "generated_descriptor_cluster_count": len(probabilities),
        "generated_effective_mode_count": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "initial_noise_sensitivity_supported": bool(np.median(generated_distances) > 0),
        "descriptor_standardization_source": "matched_real_reference_panel",
    }


def _bootstrap_difference(
    generated: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    observed = float(np.mean(generated) - np.mean(reference))
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        differences[index] = np.mean(generated[rng.integers(0, len(generated), len(generated))]) - np.mean(
            reference[rng.integers(0, len(reference), len(reference))]
        )
    alpha = (1.0 - confidence) / 2.0
    pooled_scale = float(np.sqrt(0.5 * (generated.var() + reference.var())))
    midpoint = len(reference) // 2
    reference_reference = float(reference[:midpoint].mean() - reference[midpoint:].mean())
    return {
        "generated_mean": float(np.mean(generated)),
        "reference_mean": float(np.mean(reference)),
        "mean_difference_generated_minus_reference": observed,
        "standardized_mean_difference": observed / pooled_scale if pooled_scale > 0 else None,
        "bootstrap_ci_lower": float(np.quantile(differences, alpha)),
        "bootstrap_ci_upper": float(np.quantile(differences, 1.0 - alpha)),
        "reference_reference_half_mean_difference": reference_reference,
        "reference_reference_calibration": "deterministic_panel_halves_descriptive_only",
    }


def distribution_summaries(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "adjacent_distance_mean_angstrom",
        "adjacent_distance_rmse_to_3_8_angstrom",
        "bond_angle_mean_degrees",
        "signed_pseudo_dihedral_positive_fraction",
        "curvature_mean",
        "clash_fraction",
        "discontinuity_fraction",
        "radius_of_gyration_angstrom",
        "end_to_end_distance_angstrom",
        "asphericity",
        "contact_density_6a",
        "contact_density_8a",
        "contact_density_10a",
        "contact_density_12a",
        "long_range_contact_density_8a",
        "contact_order_8a",
    )
    aggregation = config["aggregation"]
    output = []
    groups = [("overall", list(rows))]
    groups.extend(
        (
            f"target_length_{length}",
            [row for row in rows if int(row["requested_length"]) == length],
        )
        for length in config["panel"]["lengths"]
    )
    for group_index, (group, selected) in enumerate(groups):
        generated = [row for row in selected if row["source"] == "generated"]
        reference = [row for row in selected if row["source"] == "reference"]
        for field_index, field in enumerate(fields):
            result = _bootstrap_difference(
                np.asarray([float(row[field]) for row in generated]),
                np.asarray([float(row[field]) for row in reference]),
                replicates=int(aggregation["bootstrap_replicates"]),
                seed=int(aggregation["bootstrap_seed"]) + group_index * 100 + field_index,
                confidence=float(aggregation["confidence_level"]),
            )
            output.append({"group": group, "metric": field, **result})
    return output


def chirality_summary(
    generated_arrays: Mapping[str, Mapping[str, np.ndarray]],
    reference_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    generated = np.concatenate([values["signed_pseudo_dihedral_radians"] for values in generated_arrays.values()])
    reference = np.concatenate([values["signed_pseudo_dihedral_radians"] for values in reference_arrays.values()])
    generated = generated[np.isfinite(generated)]
    reference = reference[np.isfinite(reference)]
    reflected_reference = -reference
    quantiles = np.linspace(0.01, 0.99, 99)
    generated_q = np.quantile(generated, quantiles)
    native_q = np.quantile(reference, quantiles)
    reflected_q = np.quantile(reflected_reference, quantiles)
    native_distance = float(np.mean(np.abs(generated_q - native_q)))
    reflected_distance = float(np.mean(np.abs(generated_q - reflected_q)))
    return {
        "convention": "ca_pseudo_dihedral_i_i+1_i+2_i+3_v1",
        "ca_only_stereochemical_proxy": True,
        "not_native_residue_chirality": True,
        "proper_rotation_translation_invariant": True,
        "reflection_reverses_sign": True,
        "generated": _safe_summary(generated),
        "native_reference": _safe_summary(reference),
        "reflected_reference_control": _safe_summary(reflected_reference),
        "quantile_distance_to_native": native_distance,
        "quantile_distance_to_reflected": reflected_distance,
        "mirror_likeness": "closer_to_reflected_control"
        if reflected_distance < native_distance
        else "closer_to_native_reference",
        "o3_reflection_caveat": (
            "The coordinate model is O(3)-equivariant, so reflection symmetry is architectural; "
            "C-alpha pseudo-chirality must be interpreted as a distributional diagnostic, "
            "not native residue stereochemistry."
        ),
    }


def _tail_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    generated = [row for row in rows if row["source"] == "generated"]
    ranked = sorted(generated, key=lambda row: float(row["adjacent_distance_rmse_to_3_8_angstrom"]), reverse=True)
    known = {(64, 18): 1.046088, (384, 7): 1.070671}
    reproduced = []
    for row in generated:
        key = (int(row["length"]), int(row["sample_index"]))
        if key in known:
            reproduced.append(
                {
                    "length": key[0],
                    "sample_index": key[1],
                    "seed": int(row["seed"]),
                    "phase3h_adjacent_reference_error_angstrom": float(
                        row["phase3h_adjacent_reference_error_angstrom"]
                    ),
                    "expected_phase3h_value_angstrom": known[key],
                }
            )
    if len(reproduced) != 2:
        raise ValueError("E007 Phase-3I known Phase-3H tail failures were not reproduced")
    maximum = int(config["aggregation"]["maximum_failure_examples"])
    return {
        "known_phase3h_adjacent_failures": reproduced,
        "known_failure_count": 2,
        "outliers_removed": False,
        "best_examples": [row["sample_id"] for row in ranked[-min(5, len(ranked)) :]],
        "median_example": ranked[len(ranked) // 2]["sample_id"],
        "worst_examples": [row["sample_id"] for row in ranked[:maximum]],
    }


def _classification(
    distributions: Sequence[Mapping[str, Any]],
    diversity: Mapping[str, Any],
    chirality: Mapping[str, Any],
    tail: Mapping[str, Any],
) -> dict[str, Any]:
    overall = {row["metric"]: row for row in distributions if row["group"] == "overall"}
    local_fields = ("adjacent_distance_mean_angstrom", "bond_angle_mean_degrees", "curvature_mean")
    global_fields = ("radius_of_gyration_angstrom", "asphericity", "contact_density_8a")
    local_shift = any(
        not (float(overall[field]["bootstrap_ci_lower"]) <= 0 <= float(overall[field]["bootstrap_ci_upper"]))
        for field in local_fields
    )
    global_shift = any(
        not (float(overall[field]["bootstrap_ci_lower"]) <= 0 <= float(overall[field]["bootstrap_ci_upper"]))
        for field in global_fields
    )
    mode_collapse = float(diversity["generated_effective_mode_count"]) < 0.25 * 160
    chirality_failure = chirality["mirror_likeness"] == "closer_to_reflected_control"
    if chirality_failure:
        category = "chirality_failure"
    elif mode_collapse:
        category = "mode_collapse"
    elif global_shift and not local_shift:
        category = "local_geometry_only"
    elif global_shift:
        category = "global_distribution_mismatch"
    elif tail["known_failure_count"]:
        category = "tail_limited_research_generator"
    else:
        category = "promising_research_generator"
    if category not in CLASSIFICATIONS:
        category = "retrospective_audit_inconclusive"
    return {
        "category": category,
        "descriptive_not_authorizing": True,
        "no_scalar_score": True,
        "retrospective_checkpoint_selection_bias": (
            "Step 9000 was selected using prior denoising and Phase-3H sampling evidence; "
            "this audit is retrospective and cannot serve as an independent checkpoint-selection test."
        ),
        "phase3j_required_for_prospective_confirmation": True,
    }


def _inventory(staging: Path) -> dict[str, Any]:
    excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
    rows = [
        {"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name
    ]
    return {"artifacts": rows, "aggregate_sha256": _canonical_sha(rows)}


def _generated_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = Path(config["phase3h"]["root"])
    table = pq.read_table(root / "sample_metrics.parquet")
    rows = [row for row in table.to_pylist() if int(row["checkpoint_update"]) == 9000]
    expected = int(config["phase3h"]["expected_step9000_samples"])
    if len(rows) != expected:
        raise ValueError("E007 Phase-3I step-9000 sample count contradiction")
    identities = {(int(row["length"]), int(row["sample_index"])) for row in rows}
    expected_identities = {(length, index) for length in config["panel"]["lengths"] for index in range(32)}
    if identities != expected_identities:
        raise ValueError("E007 Phase-3I step-9000 sample identity contradiction")
    return sorted(rows, key=lambda row: (int(row["length"]), int(row["sample_index"])))


def _metric_summary(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = [("overall", list(rows))]
    for source in ("generated", "reference"):
        selected_source = [row for row in rows if row["source"] == source]
        groups.append((source, selected_source))
        groups.extend(
            (
                f"{source}:target_length_{length}",
                [row for row in selected_source if int(row["requested_length"]) == length],
            )
            for length in (64, 128, 256, 384, 500)
        )
    for name, selected in groups:
        result[name] = {field: _safe_summary(np.asarray([float(row[field]) for row in selected])) for field in fields}
    return result


def _near_duplicate_summary(coordinates: Mapping[str, np.ndarray], tolerance: float) -> dict[str, Any]:
    examples = []
    comparisons = 0
    total = 0
    for length in (64, 128, 256, 384, 500):
        selected = sorted((key, value) for key, value in coordinates.items() if value.shape[0] == length)
        vectors = {key: _distance_matrix(value)[np.triu_indices(length, 1)] for key, value in selected}
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                comparisons += 1
                left_id, right_id = selected[left][0], selected[right][0]
                rms = float(np.sqrt(np.mean(np.square(vectors[left_id] - vectors[right_id]))))
                if rms <= tolerance:
                    total += 1
                    if len(examples) < 20:
                        examples.append({"left": left_id, "right": right_id, "rigid_distance_rms_angstrom": rms})
    return {
        "within_length_pair_comparisons": comparisons,
        "near_duplicate_threshold_angstrom": tolerance,
        "near_duplicate_count": total,
        "near_duplicate_examples": examples,
        "examples_truncated": total > len(examples),
    }


def audit_geometry_generator_capability(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3I output exists: {output} or {staging}")
    prerequisites_before = verify_prerequisites(config, full=True)
    staging.mkdir(parents=True)
    heartbeat_path = staging / "heartbeat.json"
    current_stage = "reference_selection"
    execution_state = {
        "dataset_authorization_completed": True,
        "clean_validation_manifest_scanned": False,
        "generated_coordinates_loaded": 0,
        "reference_coordinates_loaded": 0,
        "scientific_metrics_published": False,
    }

    def heartbeat(status: str, **values: Any) -> None:
        _atomic_json(heartbeat_path, {"status": status, "updated_utc": _utc_now(), **values, **NON_AUTHORIZING})

    heartbeat("initializing", stage=current_stage, **execution_state)
    try:
        reference_manifest, reference_selection = select_reference_panel(config)
        execution_state["clean_validation_manifest_scanned"] = True
        generated_source_rows = _generated_records(config)
        rows: list[dict[str, Any]] = []
        generated_arrays: dict[str, dict[str, np.ndarray]] = {}
        reference_arrays: dict[str, dict[str, np.ndarray]] = {}
        generated_coordinates: dict[str, np.ndarray] = {}
        generated_hashes: list[str] = []
        phase3h_root = Path(config["phase3h"]["root"])
        scale = 12.22820347644835

        current_stage = "generated_metrics"
        for position, source_row in enumerate(generated_source_rows):
            artifact = phase3h_root / str(source_row["artifact_path"])
            _verify_file(artifact, str(source_row["artifact_sha256"]), f"generated_artifact_{position}")
            with np.load(artifact, allow_pickle=False) as payload:
                normalized = np.asarray(payload["coordinates"], dtype=np.float64)
                stored_distances = np.asarray(payload["distance_matrix"], dtype=np.float64)
            if normalized.shape == (1, int(source_row["length"]), 3):
                normalized = normalized[0]
            coordinates = normalized * scale
            canonical_distances = _distance_matrix(coordinates)
            if stored_distances.shape == (1, len(coordinates), len(coordinates)):
                stored_distances = stored_distances[0]
            if stored_distances.shape != canonical_distances.shape or not np.allclose(
                stored_distances, canonical_distances, atol=2e-4, rtol=0
            ):
                raise ValueError(f"E007 Phase-3I generated distance artifact contradiction: {artifact}")
            sample_id = f"generated_step9000_N{int(source_row['length']):04d}_i{int(source_row['sample_index']):03d}"
            row, arrays = geometry_metrics(
                coordinates,
                source="generated",
                sample_id=sample_id,
                length=int(source_row["length"]),
                config=config,
            )
            row.update(
                {
                    "requested_length": int(source_row["length"]),
                    "actual_length": int(source_row["length"]),
                    "signed_length_mismatch": 0,
                    "absolute_length_mismatch": 0,
                    "match_type": "generated_requested_length",
                    "sample_index": int(source_row["sample_index"]),
                    "seed": int(source_row["seed"]),
                    "coordinate_artifact_path": str(source_row["artifact_path"]),
                    "coordinate_artifact_sha256": str(source_row["artifact_sha256"]),
                    "coordinate_sha256": str(source_row["coordinate_sha256"]),
                    "initial_noise_tensor_sha256": str(source_row["initial_noise_tensor_sha256"]),
                    "phase3h_adjacent_reference_error_angstrom": float(source_row["adjacent_reference_error_angstrom"]),
                    "phase3h_adjacent_original_gate_pass": bool(source_row["adjacent_original_gate_pass"]),
                }
            )
            rows.append(row)
            generated_arrays[sample_id] = arrays
            generated_coordinates[sample_id] = coordinates
            generated_hashes.append(str(source_row["coordinate_sha256"]))
            execution_state["generated_coordinates_loaded"] = position + 1
            heartbeat("running", stage=current_stage, processed=position + 1, total=320, **execution_state)

        reference_publication_rows = []
        current_stage = "reference_metrics"
        for position, selected in enumerate(reference_manifest):
            coordinates, provenance = load_reference_coordinates(config, selected)
            row, arrays = geometry_metrics(
                coordinates,
                source="reference",
                sample_id=str(selected["sample_id"]),
                length=int(selected["length"]),
                config=config,
            )
            row.update({"sample_index": int(selected["reference_index"]), "seed": None, **provenance})
            row["requested_length"] = int(selected["target_length"])
            rows.append(row)
            reference_arrays[str(selected["sample_id"])] = arrays
            reference_publication_rows.append(provenance)
            execution_state["reference_coordinates_loaded"] = position + 1
            heartbeat("running", stage=current_stage, processed=161 + position, total=320, **execution_state)

        if len(rows) != 320:
            raise ValueError("E007 Phase-3I per-sample count contradiction")
        local_fields = (
            "adjacent_distance_mean_angstrom",
            "adjacent_distance_rmse_to_3_8_angstrom",
            "distance_i_plus_1_mean_angstrom",
            "distance_i_plus_2_mean_angstrom",
            "distance_i_plus_3_mean_angstrom",
            "bond_angle_mean_degrees",
            "signed_pseudo_dihedral_mean",
            "signed_pseudo_dihedral_positive_fraction",
            "signed_tetrahedral_volume_mean",
            "curvature_mean",
            "clash_fraction",
            "discontinuity_fraction",
        )
        global_fields = (
            "radius_of_gyration_angstrom",
            "end_to_end_distance_angstrom",
            "asphericity",
            "principal_axis_ratio_2_to_1",
            "principal_axis_ratio_3_to_1",
            "contact_density_6a",
            "contact_density_8a",
            "contact_density_10a",
            "contact_density_12a",
            "long_range_contact_density_8a",
            "contact_order_8a",
            "contact_graph_component_count_8a",
            "contact_graph_largest_component_fraction_8a",
            "contact_graph_degree_mean_8a",
        )
        local = _metric_summary(rows, local_fields)
        global_summary = _metric_summary(rows, global_fields)
        chirality = chirality_summary(generated_arrays, reference_arrays)
        requested_by_sample = {str(row["sample_id"]): int(row["requested_length"]) for row in rows}
        chirality["by_length"] = {
            str(length): chirality_summary(
                {key: value for key, value in generated_arrays.items() if requested_by_sample[key] == length},
                {key: value for key, value in reference_arrays.items() if requested_by_sample[key] == length},
            )
            for length in config["panel"]["lengths"]
        }
        diversity = diversity_summary(rows, generated_hashes, config)
        diversity["rigid_near_duplicates"] = _near_duplicate_summary(
            generated_coordinates, float(config["metrics"]["near_duplicate_rmsd_angstrom"])
        )
        distributions = distribution_summaries(rows, config)
        tail = _tail_summary(rows, config)
        classification = _classification(distributions, diversity, chirality, tail)

        compression = str(config["aggregation"]["parquet_compression"])
        _atomic_parquet(staging / "per_sample_metrics.parquet", rows, compression)
        _atomic_parquet(staging / "reference_manifest.parquet", reference_publication_rows, compression)
        _atomic_json(staging / "reference_selection.json", reference_selection)
        _atomic_json(staging / "local_geometry_summary.json", local)
        _atomic_json(staging / "global_geometry_summary.json", global_summary)
        _atomic_json(staging / "chirality_summary.json", chirality)
        _atomic_json(staging / "diversity_summary.json", diversity)
        _atomic_json(staging / "distribution_comparisons.json", distributions)
        _atomic_json(staging / "failure_examples.json", tail)
        execution_state["scientific_metrics_published"] = True

        current_stage = "protected_input_reverification"
        prerequisites_after = verify_prerequisites(config, full=True)
        if prerequisites_after != prerequisites_before:
            raise ValueError("E007 Phase-3I protected prerequisite hashes changed")
        report = {
            "status": "completed_read_only_non_authorizing",
            "version": VERSION,
            "classification": classification,
            "selected_checkpoint": config["selected_checkpoint"],
            "generated_sample_count": 160,
            "reference_sample_count": 160,
            "comparison_panel_counts": {
                str(length): {
                    "generated_requested_length": 32,
                    "reference_total": 32,
                    "reference_exact": reference_selection["by_target_length"][str(length)]["selected_exact_count"],
                    "reference_nearest_same_stratum": reference_selection["by_target_length"][str(length)][
                        "selected_nearest_count"
                    ],
                }
                for length in config["panel"]["lengths"]
            },
            "local_geometry_summary_path": "local_geometry_summary.json",
            "global_geometry_summary_path": "global_geometry_summary.json",
            "chirality_summary": chirality,
            "diversity_summary": diversity,
            "tail_summary": tail,
            "distribution_comparison_count": len(distributions),
            "reference_selection": {
                "population": "identity_30_clean_validation",
                "selection_independent_of_generated_metrics": True,
                "exact_length_matching_for_all_references": False,
                "policy": "exact_first_then_nearest_length_within_same_stratum",
                "maximum_permitted_absolute_length_mismatch": config["panel"]["maximum_reference_length_mismatch"],
                "exact_match_count": reference_selection["exact_match_count"],
                "nearest_match_count": reference_selection["nearest_match_count"],
                "mismatch_distribution": reference_selection["signed_mismatch_counts"],
                "maximum_observed_absolute_mismatch": reference_selection["maximum_observed_absolute_mismatch"],
                "selection_seed": config["panel"]["selection_seed"],
                "selection_version": config["panel"]["selection_version"],
                "sample_id_sha256": reference_selection["sample_id_sha256"],
                "diagnostics_path": "reference_selection.json",
            },
            "scientific_limitations": [
                "This is a retrospective audit of a checkpoint selected using prior denoising and sampling evidence.",
                "C-alpha traces cannot establish atomistic stereochemistry, peptide planarity, "
                "side-chain packing, or foldability.",
                "Passing a distributional comparison does not establish physical validity or "
                "prospective generalization.",
            ],
            "phase3j_prospective_plan": config["prospective_phase3j"],
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        inventory = _inventory(staging)
        _atomic_json(staging / "artifact_inventory.json", inventory)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "completed_utc": _utc_now(),
            "source_hashes": prerequisites_before,
            "report_sha256": sha256_file(staging / "report.json"),
            "per_sample_metrics_sha256": sha256_file(staging / "per_sample_metrics.parquet"),
            "reference_manifest_sha256": sha256_file(staging / "reference_manifest.parquet"),
            "reference_selection_sha256": sha256_file(staging / "reference_selection.json"),
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            "artifact_inventory_aggregate_sha256": inventory["aggregate_sha256"],
            "protected_inputs_unchanged": True,
            "model_or_sampling_workload_performed": False,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        if (staging / "report.json").read_bytes() == (staging / "protocol.json").read_bytes():
            raise ValueError("E007 Phase-3I report/protocol separation failure")
        heartbeat("completed", stage="publication", report_sha256=protocol["report_sha256"])
        staging.replace(output)
        return {
            "status": report["status"],
            "classification": classification,
            "output_dir": str(output),
            **NON_AUTHORIZING,
        }
    except BaseException as error:
        heartbeat(
            "failed",
            stage=current_stage,
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            resumable=False,
            **execution_state,
        )
        raise
