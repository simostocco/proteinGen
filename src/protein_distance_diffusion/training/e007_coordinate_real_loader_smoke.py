"""Bounded real-loader and no-update production-model smoke for E007 Phase 3E-B."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import resource
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import (
    E007CoordinateDataset,
    collate_e007_coordinates,
    stable_center_valid_coordinates,
)
from protein_distance_diffusion.data.rich_geometry import (
    RichDatasetAuthorization,
    authorize_rich_geometry_dataset,
)
from protein_distance_diffusion.evaluation.distance_matrix_quality import (
    MatrixQualityConfig,
    assess_distance_matrix,
)
from protein_distance_diffusion.models.coordinate_equivariance import (
    coordinate_backend_policy,
    coordinate_model_execution_context,
    equivariance_criterion,
    equivariance_metrics,
    require_coordinate_backend_policy,
)
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import (
    EquivariantPairCoordinateUNet,
)
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_objective_pilot import _gradient_norms
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file, verify_metadata

SMOKE_VERSION = "e007_coordinate_real_loader_smoke_v1"
NORMALIZATION_SCALE_ANGSTROM = 12.22820347644835
CENTERING_ATOL_NORMALIZED = 2e-6
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "parameters_unchanged": True,
    "reverse_sampling_executed": False,
    "checkpoints_created": False,
    "dataset_modified": False,
}


def _memory_telemetry(device: torch.device | None = None) -> dict[str, float | None]:
    current_rss_mib = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current_rss_mib = int(line.split()[1]) / 1024.0
                break
    except OSError:
        pass
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if current_rss_mib is None:
        current_rss_mib = peak_rss_mib
    cuda = device is not None and device.type == "cuda"
    return {
        "current_rss_mib": current_rss_mib,
        "peak_rss_mib": peak_rss_mib,
        "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20 if cuda else None,
        "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20 if cuda else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != SMOKE_VERSION:
        raise ValueError("E007 Phase-3E-B configuration version contradiction")
    if float(payload.get("coordinate_scale_angstrom", 0)) != NORMALIZATION_SCALE_ANGSTROM:
        raise ValueError("E007 Phase-3E-B coordinate scale contradiction")
    if int(payload.get("samples_per_split_stratum", 0)) < 2:
        raise ValueError("E007 Phase-3E-B requires at least two samples per split and stratum")
    if int(payload.get("expected_parameter_count", 0)) != 7_586_505:
        raise ValueError("E007 Phase-3E-B production parameter-count contract changed")
    if payload.get("mixed_precision") is not False:
        raise ValueError("E007 Phase-3E-B FP32 smoke contract changed")
    coordinate_backend_policy(payload.get("numerics"))
    return payload


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3E-B prerequisite hash contradiction: {label}")
    return observed


def verify_smoke_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    """Verify only pinned reports and metadata; never open a scientific shard."""
    phase3e = config["phase3e_a"]
    directory = Path(phase3e["directory"])
    files = {
        "normalization": (directory / "normalization.json", phase3e["normalization_sha256"]),
        "normalization_report": (directory / "report.json", phase3e["report_sha256"]),
        "normalization_protocol": (directory / "protocol.json", phase3e["protocol_sha256"]),
        "normalization_length_statistics": (
            directory / "length_stratified_statistics.json",
            phase3e["length_statistics_sha256"],
        ),
        "normalization_per_sample_audit": (
            directory / "per_sample_audit.jsonl.gz",
            phase3e["per_sample_audit_sha256"],
        ),
        "phase3d_config": (Path(config["phase3d"]["config_path"]), config["phase3d"]["config_sha256"]),
        "phase3d_report": (Path(config["phase3d"]["report_path"]), config["phase3d"]["report_sha256"]),
        "phase3d_protocol": (
            Path(config["phase3d"]["protocol_path"]),
            config["phase3d"]["protocol_sha256"],
        ),
        "coordinate_model_contract": (
            Path(config["coordinate_model_contract"]["path"]),
            config["coordinate_model_contract"]["sha256"],
        ),
    }
    hashes = {name: _verify_hash(path, expected, name) for name, (path, expected) in files.items()}
    normalization = json.loads((directory / "normalization.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if normalization.get("version") != phase3e["required_version"]:
        raise ValueError("E007 Phase-3E-A normalization version contradiction")
    if float(normalization.get("coordinate_scale_angstrom", 0)) != NORMALIZATION_SCALE_ANGSTROM:
        raise ValueError("E007 Phase-3E-A normalization scale contradiction")
    if protocol.get("status") != "completed_non_authorizing" or report.get("status") != "completed_non_authorizing":
        raise ValueError("E007 Phase-3E-A normalization is incomplete")
    if any(bool(normalization.get(key)) for key in NON_AUTHORIZING if key.startswith("authorizes_")):
        raise ValueError("E007 Phase-3E-A unexpectedly authorizes training")
    phase3d = json.loads(Path(config["phase3d"]["report_path"]).read_text())
    if phase3d.get("status") != "completed" or phase3d.get("classification") != "small_capacity_sufficient":
        raise ValueError("E007 Phase-3D evidence contradiction")
    if phase3d.get("parameter_counts", {}).get("production") != 7_586_505:
        raise ValueError("E007 Phase-3D production parameter count contradiction")
    model_contract = json.loads(Path(config["coordinate_model_contract"]["path"]).read_text())
    generator_path = Path(model_contract["configuration"]["generator_path"])
    generator_hash = _verify_hash(
        generator_path,
        model_contract["configuration"]["generator_sha256"],
        "production_generator_config",
    )
    generator = yaml.safe_load(generator_path.read_text())
    if generator["model"] != config["model"]:
        raise ValueError("E007 Phase-3E-B does not use the exact production architecture")
    metadata = verify_metadata(config["dataset"])
    schema = json.loads((Path(config["dataset"]["root"]) / "schema.json").read_text())
    available_group_fields = sorted(set(config["group_identifier_columns"]) & set(schema["columns"]))
    return {
        "hashes": {
            **hashes,
            "production_generator_config": generator_hash,
            **{f"dataset_{key}": value for key, value in metadata["hashes"].items()},
        },
        "normalization": normalization,
        "normalization_report": report,
        "dataset_metadata": metadata,
        "available_group_identifier_columns": available_group_fields,
    }


def _output_for_mode(config: dict[str, Any], mode: str) -> Path:
    if mode == "loader-smoke":
        return Path(config["loader_output_dir"])
    if mode == "forward-backward-smoke":
        return Path(config["forward_backward_output_dir"])
    raise ValueError(f"unknown E007 Phase-3E-B mode: {mode}")


def _require_output_absent(output: Path) -> None:
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3E-B output already exists: {output} or {staging}")


def _verify_completed_loader_output(config: dict[str, Any]) -> dict[str, str] | None:
    output = Path(config["loader_output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if staging.exists():
        raise FileExistsError(f"E007 Phase-3E-B loader staging output exists: {staging}")
    if not output.exists():
        return None
    expected = config.get("completed_loader_output")
    if not isinstance(expected, dict):
        raise FileExistsError(f"E007 Phase-3E-B unpinned loader output exists: {output}")
    files = {
        "report": "report.json",
        "protocol": "protocol.json",
        "heartbeat": "heartbeat.json",
        "selected_panel_manifest": "selected_panel_manifest.json",
        "detailed_metrics": "detailed_metrics.jsonl.gz",
    }
    missing = [filename for filename in files.values() if not (output / filename).is_file()]
    if missing:
        raise FileExistsError(f"E007 Phase-3E-B loader output is incomplete: {missing}")
    hashes = {
        name: _verify_hash(output / filename, expected[f"{name}_sha256"], f"completed loader {name}")
        for name, filename in files.items()
    }
    report = json.loads((output / "report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    if report.get("status") != "completed_non_authorizing" or protocol.get("status") != "completed_non_authorizing":
        raise ValueError("E007 Phase-3E-B pinned loader output is incomplete")
    if any(report.get(key) != value for key, value in NON_AUTHORIZING.items()):
        raise ValueError("E007 Phase-3E-B pinned loader output authorization contradiction")
    return hashes


def plan_real_loader_smoke(config_path: str | Path) -> dict[str, Any]:
    """Plan both future smoke modes without scanning coordinates or creating a model."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    completed_loader_hashes = _verify_completed_loader_output(config)
    _require_output_absent(_output_for_mode(config, "forward-backward-smoke"))
    prerequisites = verify_smoke_prerequisites(config)
    return {
        "status": "planned_phase3e_b_non_authorizing",
        "version": SMOKE_VERSION,
        "loader_output_dir": config["loader_output_dir"],
        "forward_backward_output_dir": config["forward_backward_output_dir"],
        "coordinate_scale_angstrom": NORMALIZATION_SCALE_ANGSTROM,
        "normalization_interpretation": (
            "coordinate_payloads_scanned=false refers to external NPZ/source payloads; "
            "authorized projected sidecar coordinate columns were scanned"
        ),
        "capacity_interpretation": (
            "production capacity is an owner engineering choice; Phase 3D did not prove it necessary"
        ),
        "model_implementation": "EquivariantPairCoordinateUNet",
        "expected_parameter_count": 7_586_505,
        "prediction_parameterization": "coordinate_v",
        "numerical_backend_policy": config["numerics"],
        "random_initialization": True,
        "pretrained_weights_loaded": False,
        "sequence_inputs": False,
        "clean_coordinate_feature_inputs": False,
        "panel_contract": {
            "splits": ["train", "validation"],
            "length_strata": config["length_strata"],
            "samples_per_split_stratum": int(config["samples_per_split_stratum"]),
            "minimum_total_selected_samples": 2
            * len(config["length_strata"])
            * int(config["samples_per_split_stratum"]),
            "selection_version": config["selection_version"],
            "requires_nonmultiple_of_eight": True,
            "requires_maximum_or_near_maximum_length": True,
        },
        "leakage_audit_contract": {
            "duplicate_sample_ids": True,
            "exact_coordinate_hashes": True,
            "exact_rigid_shape_distance_hashes": True,
            "available_group_identifier_columns": prerequisites["available_group_identifier_columns"],
            "cross_split_leakage_requires_dataset_review": True,
        },
        "batch_regimes": config["batch_regimes"],
        "dynamic_square_padding_factor": int(config["expected_downsample_factor"]),
        "prerequisite_hashes": prerequisites["hashes"],
        "configuration_sha256": sha256_file(config_path),
        "coordinate_payloads_scanned": False,
        "completed_loader_output": {
            "present": completed_loader_hashes is not None,
            "verified_hashes": completed_loader_hashes,
        },
        "model_created": False,
        "forward_executed": False,
        "backward_executed": False,
        **NON_AUTHORIZING,
    }


def _authorize_dataset(config: dict[str, Any]) -> RichDatasetAuthorization:
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["root"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
    )


def _protected_identity(authorization: RichDatasetAuthorization, prerequisite_hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "prerequisite_hashes": prerequisite_hashes,
        "dataset_split_counts": authorization.split_counts,
        "scientific_shard_count": len(authorization.observed_shard_hashes),
        "scientific_shard_hashes_sha256": _canonical_sha256(authorization.observed_shard_hashes),
    }


def _length_stratum(length: int, strata: list[dict[str, Any]]) -> str:
    matches = [str(record["name"]) for record in strata if int(record["minimum"]) <= length <= int(record["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 Phase-3E-B length has no unique stratum: {length}")
    return matches[0]


def coordinate_hash(row: dict[str, Any]) -> str:
    coordinates = row["coordinates"].detach().cpu().numpy()
    mask = row["residue_mask"].detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coordinates, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def rigid_shape_hash(row: dict[str, Any]) -> str:
    coordinates = row["coordinates"].detach().cpu().double()
    mask = row["residue_mask"].detach().cpu().bool()
    valid = coordinates[mask]
    distances = coordinates_to_distance_matrix(valid, diagnostic_float64=True)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    values = np.ascontiguousarray(distances[upper].numpy(), dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _selection_rank(seed: int, split: str, stratum: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{split}\0{stratum}\0{sample_id}".encode()).hexdigest()


def _offer(records: list[dict[str, Any]], record: dict[str, Any], maximum: int, key: str = "rank") -> None:
    records.append(record)
    records.sort(key=lambda item: (item[key], item["sample_id"]))
    del records[maximum:]


def _finalize_stratum_selection(pool: dict[str, Any], *, count: int, require_longest: bool) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []

    def add(record: dict[str, Any] | None) -> None:
        if record is not None and all(item["sample_id"] != record["sample_id"] for item in selected):
            selected.append(record)

    add(pool["nonmultiple"][0] if pool["nonmultiple"] else None)
    if require_longest:
        add(pool["longest"][0] if pool["longest"] else None)
    for record in pool["overall"]:
        add(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"E007 Phase-3E-B stratum has {len(selected)} selectable accepted rows, expected {count}")
    return sorted(selected, key=lambda item: (item["rank"], item["sample_id"]))


def _scan_panels_and_leakage(
    datasets: dict[str, E007CoordinateDataset], config: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    count = int(config["samples_per_split_stratum"])
    names = [str(record["name"]) for record in config["length_strata"]]
    final_name = names[-1]
    pools = {
        split: {
            name: {"overall": [], "nonmultiple": [], "longest": [], "maximum_length": -1, "available": 0}
            for name in names
        }
        for split in ("train", "validation")
    }
    identities = {
        split: {
            "sample_ids": Counter(),
            "coordinate_hashes": Counter(),
            "rigid_shape_hashes": Counter(),
            "candidate_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
        }
        for split in ("train", "validation")
    }
    for split, dataset in datasets.items():
        for index in range(len(dataset)):
            row = dataset[index]
            identities[split]["candidate_count"] += 1
            if row["split"] != split:
                raise ValueError("E007 Phase-3E-B physical/row split contradiction")
            if not row["accepted_contiguous_single_chain"]:
                identities[split]["rejected_count"] += 1
                continue
            identities[split]["accepted_count"] += 1
            sample_id = str(row["sample_id"])
            coordinate = coordinate_hash(row)
            shape = rigid_shape_hash(row)
            identities[split]["sample_ids"][sample_id] += 1
            identities[split]["coordinate_hashes"][coordinate] += 1
            identities[split]["rigid_shape_hashes"][shape] += 1
            length = int(row["sequence_length"])
            stratum = _length_stratum(length, config["length_strata"])
            rank = _selection_rank(int(config["seed"]), split, stratum, sample_id)
            record = {
                "sample_id": sample_id,
                "split": split,
                "dataset_index": index,
                "sequence_length": length,
                "length_stratum": stratum,
                "rank": rank,
                "coordinate_sha256": coordinate,
                "rigid_shape_distance_sha256": shape,
                "source_sha256": row["source_sha256"],
                "npz_sha256": row["npz_sha256"],
                "row": row,
            }
            pool = pools[split][stratum]
            pool["available"] += 1
            _offer(pool["overall"], record, count * 3)
            if length % 8:
                _offer(pool["nonmultiple"], record, count)
            if length > pool["maximum_length"]:
                pool["maximum_length"] = length
                pool["longest"] = [record]
            elif length == pool["maximum_length"]:
                _offer(pool["longest"], record, count)
    selected = {}
    manifest = []
    for split in ("train", "validation"):
        split_rows = []
        for name in names:
            records = _finalize_stratum_selection(pools[split][name], count=count, require_longest=name == final_name)
            split_rows.extend(records)
            manifest.extend({key: value for key, value in record.items() if key != "row"} for record in records)
        selected[split] = split_rows
    leakage: dict[str, Any] = {
        "within_split": {},
        "cross_split": {},
        "group_identifier_columns": {
            "requested": list(config["group_identifier_columns"]),
            "available": [],
            "status": "not_present_in_authoritative_sidecar_schema",
        },
    }
    for split in ("train", "validation"):
        leakage["within_split"][split] = {
            "candidate_count": identities[split]["candidate_count"],
            "accepted_count": identities[split]["accepted_count"],
            "rejected_count": identities[split]["rejected_count"],
            "duplicate_sample_id_rows": sum(value - 1 for value in identities[split]["sample_ids"].values()),
            "duplicate_coordinate_hash_rows": sum(
                value - 1 for value in identities[split]["coordinate_hashes"].values()
            ),
            "duplicate_rigid_shape_hash_rows": sum(
                value - 1 for value in identities[split]["rigid_shape_hashes"].values()
            ),
        }
    for name in ("sample_ids", "coordinate_hashes", "rigid_shape_hashes"):
        overlap = sorted(set(identities["train"][name]) & set(identities["validation"][name]))
        leakage["cross_split"][name] = {
            "overlapping_unique_key_count": len(overlap),
            "bounded_examples": overlap[:100],
            "examples_truncated": len(overlap) > 100,
        }
    leakage_detected = any(record["overlapping_unique_key_count"] > 0 for record in leakage["cross_split"].values())
    leakage["leakage_detected"] = leakage_detected
    leakage["classification"] = "dataset_review_required" if leakage_detected else "no_cross_split_leakage_detected"
    leakage["scientific_authorization_refused"] = leakage_detected
    return selected, {
        "selected_manifest": manifest,
        "selected_manifest_sha256": _canonical_sha256(manifest),
        "leakage": leakage,
        "available_counts_by_split_stratum": {
            split: {name: pools[split][name]["available"] for name in names} for split in ("train", "validation")
        },
    }


def prepare_coordinate_batch(rows: list[dict[str, Any]], scale: float, downsample_factor: int) -> dict[str, Any]:
    if scale != NORMALIZATION_SCALE_ANGSTROM:
        raise ValueError("E007 Phase-3E-B batch scale contradiction")
    if any("sequence" in row or "token_ids" in row for row in rows):
        raise ValueError("E007 coordinate model batch contains sequence identities")
    physical = collate_e007_coordinates(rows, augment_rotation=False)
    normalized = physical["coordinates"] / scale
    maximum = int(physical["lengths"].max())
    square_padded = math.ceil(maximum / downsample_factor) * downsample_factor
    mask = physical["residue_mask"]
    if torch.count_nonzero(normalized[~mask]) != 0:
        raise ValueError("E007 normalized coordinate padding is nonzero")
    centered_mean = (normalized * mask[..., None]).sum(dim=1) / mask.sum(dim=1).clamp_min(1)[:, None]
    centering_max_abs = centered_mean.abs().amax(dim=1)
    if bool((centering_max_abs > CENTERING_ATOL_NORMALIZED).any()):
        offenders = [
            {
                "sample_id": physical["sample_ids"][index],
                "maximum_absolute_centroid_component": float(centering_max_abs[index]),
                "rms_centroid_residual": float(centered_mean[index].square().mean().sqrt()),
                "valid_residue_count": int(mask[index].sum()),
            }
            for index in torch.nonzero(centering_max_abs > CENTERING_ATOL_NORMALIZED).flatten().tolist()
        ]
        raise ValueError(
            "E007 normalized coordinates are not centered: "
            f"criterion=max_abs<={CENTERING_ATOL_NORMALIZED}, rtol=0, offenders={offenders[:20]}"
        )
    return {
        **physical,
        "physical_coordinates_angstrom": physical["coordinates"],
        "coordinates": normalized,
        "coordinate_scale_angstrom": scale,
        "unpadded_maximum_length": maximum,
        "square_padded_length": square_padded,
        "downsample_factor": downsample_factor,
        "sequence_inputs": False,
        "clean_coordinate_features": False,
        "centering_maximum_absolute_component": centering_max_abs,
        "centering_rms_residual": centered_mean.square().mean(dim=1).sqrt(),
    }


def coordinate_preparation_diagnostics(
    row: dict[str, Any],
    scale: float,
    downsample_factor: int,
    *,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Report each numerical centering stage without changing scientific data."""
    coordinates = row["coordinates"]
    mask = row["residue_mask"].bool()
    source64 = coordinates[mask].double().mean(dim=0)
    source32 = coordinates[mask].float().mean(dim=0)
    centered = stable_center_valid_coordinates(coordinates.float(), mask)
    centered_centroid = centered[mask].mean(dim=0)
    normalized = centered / scale
    normalized_centroid = normalized[mask].mean(dim=0)
    padded_length = math.ceil(int(row["sequence_length"]) / downsample_factor) * downsample_factor
    padded = torch.nn.functional.pad(normalized, (0, 0, 0, padded_length - len(mask)))
    padded_mask = torch.nn.functional.pad(mask, (0, padded_length - len(mask)), value=False)
    padded_centroid = padded[padded_mask].mean(dim=0)
    target_device = device or torch.device("cpu")
    transferred = padded.to(target_device)
    transferred_mask = padded_mask.to(target_device)
    transferred_centroid = transferred[transferred_mask].mean(dim=0).cpu()

    def summary(value: torch.Tensor) -> dict[str, Any]:
        value = value.detach().double().cpu()
        return {
            "components": value.tolist(),
            "maximum_absolute_component": float(value.abs().max()),
            "rms_residual": float(value.square().mean().sqrt()),
        }

    padded_values = padded[~padded_mask]
    return {
        "sample_id": str(row["sample_id"]),
        "sequence_length": int(row["sequence_length"]),
        "valid_residue_count": int(mask.sum()),
        "source_valid_centroid_float64_angstrom": summary(source64),
        "source_valid_centroid_float32_angstrom": summary(source32),
        "after_centering_float32_angstrom": summary(centered_centroid),
        "after_normalization_float32": summary(normalized_centroid),
        "after_padding_float32": summary(padded_centroid),
        "after_device_transfer_float32": summary(transferred_centroid),
        "padded_length": padded_length,
        "maximum_padded_coordinate_magnitude": (float(padded_values.abs().max()) if padded_values.numel() else 0.0),
        "finite_valid_coordinates": bool(torch.isfinite(coordinates[mask]).all()),
        "ca_mask_all_valid": bool(mask.all()),
        "continuity_mask_all_valid": bool(row["chain_continuity_mask"].bool().all()),
        "criterion": {
            "formula": "max(abs(sum(normalized * valid_mask) / valid_residue_count)) <= atol",
            "dtype": "float32",
            "absolute_tolerance": CENTERING_ATOL_NORMALIZED,
            "relative_tolerance": 0.0,
            "includes_padding": False,
            "denominator": "valid_residue_count",
        },
    }


def make_deterministic_corruption(
    batch: dict[str, Any], diffusion: CoordinateVPDiffusion, *, seed: int, device: torch.device
) -> dict[str, Any]:
    clean = batch["coordinates"].to(device)
    mask = batch["residue_mask"].to(device)
    timesteps = torch.arange(clean.shape[0], device=device, dtype=torch.long)
    timesteps = (timesteps * max(diffusion.timesteps - 1, 1) // max(clean.shape[0] - 1, 1)).clamp_max(
        diffusion.timesteps - 1
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    diffused = diffusion.make_training_batch(clean, mask, timesteps=timesteps, generator=generator)
    # make_training_batch centers clean once before training_target centers it
    # again; training_target also re-centers its supplied noise in float32.
    # Match those operations rather than testing float32 centering idempotence.
    canonical_clean = center_coordinates(center_coordinates(clean, mask).float(), mask)
    canonical_noise = center_coordinates(diffused.coordinate_noise.float(), mask)
    alpha, sigma = diffusion.alpha_sigma(timesteps, clean)
    expected = center_coordinates(alpha * canonical_noise - sigma * canonical_clean, mask)
    if not torch.equal(diffused.coordinate_v_target, expected):
        if not torch.allclose(diffused.coordinate_v_target, expected, atol=1e-7, rtol=0):
            raise ValueError("E007 coordinate-v target identity contradiction")
    reconstructed = diffusion.reconstruct_x0(diffused.noisy_coordinates, timesteps, diffused.coordinate_v_target, mask)
    if not torch.allclose(reconstructed, canonical_clean, atol=2e-6, rtol=0):
        raise ValueError("E007 exact coordinate-v reconstruction identity failed")
    return {"batch": diffused, "timesteps": timesteps, "clean": clean, "mask": mask}


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _distance_checks(coordinates: torch.Tensor, mask: torch.Tensor, tolerance: float) -> dict[str, Any]:
    length = int(mask.sum())
    physical = coordinates[mask] * NORMALIZATION_SCALE_ANGSTROM
    distances = coordinates_to_distance_matrix(physical, diagnostic_float64=True)
    quality = assess_distance_matrix(
        distances,
        sequence_length=length,
        config=MatrixQualityConfig(
            triangle_tolerance_angstrom=tolerance,
            eigenvalue_tolerance=tolerance,
        ),
    )
    passed = (
        quality["finite"]
        and quality["diagonal_error_max_angstrom"] == 0.0
        and quality["symmetry_error_max_angstrom"] <= tolerance
        and quality["negative_distance_count"] == 0
        and quality["triangle_violation_max_angstrom"] <= tolerance
        and quality["negative_eigenmass_fraction"] <= tolerance
        and quality["rank3_residual_energy_fraction"] <= tolerance
    )
    return {"passed": passed, "quality": quality}


def _loader_checks(selected: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for split in ("train", "validation"):
        rows = [record["row"] for record in selected[split]]
        batch = prepare_coordinate_batch(
            rows,
            float(config["coordinate_scale_angstrom"]),
            int(config["expected_downsample_factor"]),
        )
        physical_distances = coordinates_to_distance_matrix(
            batch["physical_coordinates_angstrom"], batch["residue_mask"], diagnostic_float64=True
        )
        roundtrip_distances = coordinates_to_distance_matrix(
            batch["coordinates"] * float(config["coordinate_scale_angstrom"]),
            batch["residue_mask"],
            diagnostic_float64=True,
        )
        maximum_roundtrip_error = float((physical_distances - roundtrip_distances).abs().max())
        first = make_deterministic_corruption(
            batch, CoordinateVPDiffusion(config["diffusion_steps"]), seed=91, device=torch.device("cpu")
        )
        second = make_deterministic_corruption(
            batch, CoordinateVPDiffusion(config["diffusion_steps"]), seed=91, device=torch.device("cpu")
        )
        deterministic = torch.equal(first["batch"].noisy_coordinates, second["batch"].noisy_coordinates)
        record = {
            "split": split,
            "sample_count": len(rows),
            "sample_ids": batch["sample_ids"],
            "lengths": batch["lengths"].tolist(),
            "maximum_length": batch["unpadded_maximum_length"],
            "square_padded_length": batch["square_padded_length"],
            "pair_mask_exact": bool(
                torch.equal(
                    batch["pair_mask"],
                    batch["residue_mask"][:, :, None] & batch["residue_mask"][:, None, :],
                )
            ),
            "padded_coordinates_exact_zero": bool(
                torch.count_nonzero(batch["coordinates"][~batch["residue_mask"]]) == 0
            ),
            "continuity_masks_exact": all(
                bool(row["chain_continuity_mask"].all())
                and int(row["chain_continuity_mask"].numel()) == int(row["sequence_length"]) - 1
                for row in rows
            ),
            "normalization_divisor_angstrom": float(config["coordinate_scale_angstrom"]),
            "normalized_valid_coordinate_rms": float(
                batch["coordinates"][batch["residue_mask"]].square().mean().sqrt()
            ),
            "maximum_roundtrip_distance_error_angstrom": maximum_roundtrip_error,
            "deterministic_corruption": deterministic,
            "sequence_inputs": False,
            "clean_coordinate_features": False,
        }
        checks.append(record)
    for record in checks:
        if not record["deterministic_corruption"]:
            raise ValueError("E007 Phase-3E-B corruption replay is not deterministic")
        if (
            not record["pair_mask_exact"]
            or not record["padded_coordinates_exact_zero"]
            or not record["continuity_masks_exact"]
        ):
            raise ValueError("E007 Phase-3E-B mask or padding contract failed")
        if record["maximum_roundtrip_distance_error_angstrom"] > float(config["distance_geometry_tolerance"]):
            raise ValueError("E007 Phase-3E-B physical distance round trip failed")
    return {"split_checks": checks, "passed": True}


def _batch_records(selected: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> list[dict[str, Any]]:
    regimes = {record["maximum_length"]: record for record in config["batch_regimes"]}
    output = []
    for split in ("train", "validation"):
        by_stratum: dict[str, list[dict[str, Any]]] = {}
        for record in selected[split]:
            by_stratum.setdefault(record["length_stratum"], []).append(record)
        for stratum_record in config["length_strata"]:
            name = str(stratum_record["name"])
            maximum = int(stratum_record["maximum"])
            regime = regimes[maximum]
            size = min(int(regime["physical_batch_size"]), len(by_stratum[name]))
            output.append(
                {
                    "split": split,
                    "stratum": name,
                    "regime": regime["name"],
                    "rows": [record["row"] for record in by_stratum[name][:size]],
                }
            )
    return output


def _forward_backward_checks_active(
    selected: dict[str, list[dict[str, Any]]], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    policy = coordinate_backend_policy(config["numerics"])
    require_coordinate_backend_policy(policy, device)
    torch.manual_seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
    model = EquivariantPairCoordinateUNet(**config["model"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(config["expected_parameter_count"]):
        raise ValueError(f"E007 production parameter-count contradiction: {parameter_count}")
    if model.downsample_factor != int(config["expected_downsample_factor"]):
        raise ValueError("E007 production downsample-factor contradiction")
    before = _parameter_sha256(model)
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    gradient_coverage = {"full_model": False, "pair_grid_unet_trunk": False, "coefficient_head": False}
    records = []
    for batch_index, specification in enumerate(_batch_records(selected, config)):
        prepared = prepare_coordinate_batch(
            specification["rows"],
            float(config["coordinate_scale_angstrom"]),
            int(config["expected_downsample_factor"]),
        )
        corruption = make_deterministic_corruption(
            prepared,
            diffusion,
            seed=int(config["seed"]) + batch_index,
            device=device,
        )
        diffused = corruption["batch"]
        lengths = prepared["lengths"].to(device)
        continuity = prepared["chain_continuity_mask"].to(device)
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(
            diffused.noisy_coordinates,
            diffused.timesteps,
            lengths,
            corruption["mask"],
            continuity,
        )
        prediction = output["v_prediction"]
        valid = corruption["mask"][..., None].expand_as(prediction)
        loss = (prediction[valid] - diffused.coordinate_v_target[valid]).square().mean()
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("E007 Phase-3E-B produced a non-finite forward")
        loss.backward()
        named_gradients = {
            name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None
        }
        if not named_gradients or any(not bool(torch.isfinite(value).all()) for value in named_gradients.values()):
            raise FloatingPointError("E007 Phase-3E-B gradients are missing or non-finite")
        gradient_norms = _gradient_norms(named_gradients)
        for name, value in gradient_norms.items():
            gradient_coverage[name] |= value > 0
        if torch.count_nonzero(prediction[~corruption["mask"]]) != 0:
            raise ValueError("E007 production output has nonzero padding")
        centered_mean = (prediction * corruption["mask"][..., None]).sum(dim=1) / corruption["mask"].sum(
            dim=1
        ).clamp_min(1)[:, None]
        centered_error = float(centered_mean.abs().max().detach().cpu())
        if centered_error > float(config["equivariance_atol"]):
            raise ValueError("E007 Phase-3E-B output centering failed")
        model.eval()
        with torch.no_grad():
            repeated = model(
                diffused.noisy_coordinates,
                diffused.timesteps,
                lengths,
                corruption["mask"],
                continuity,
            )["v_prediction"]
            repeated_again = model(
                diffused.noisy_coordinates,
                diffused.timesteps,
                lengths,
                corruption["mask"],
                continuity,
            )["v_prediction"]
        deterministic = torch.equal(repeated, repeated_again)
        extra_padding = int(config["expected_downsample_factor"])
        padded_coordinates = torch.nn.functional.pad(diffused.noisy_coordinates, (0, 0, 0, extra_padding))
        padded_mask = torch.nn.functional.pad(corruption["mask"], (0, extra_padding), value=False)
        padded_continuity = torch.nn.functional.pad(continuity, (0, extra_padding), value=False)
        with torch.no_grad():
            padded_prediction = model(
                padded_coordinates,
                diffused.timesteps,
                lengths,
                padded_mask,
                padded_continuity,
            )["v_prediction"]
        padding_error = float((padded_prediction[:, : prediction.shape[1]] - repeated).abs().max().cpu())
        padded_tail_zero = bool(torch.count_nonzero(padded_prediction[:, prediction.shape[1] :]) == 0)
        if padding_error > float(config["equivariance_atol"]) or not padded_tail_zero:
            raise ValueError("E007 Phase-3E-B padding invariance failed")
        reconstructed = diffusion.reconstruct_x0(
            diffused.noisy_coordinates, diffused.timesteps, repeated, corruption["mask"]
        )
        distance_checks = [
            _distance_checks(
                reconstructed[index],
                corruption["mask"][index],
                float(config["distance_geometry_tolerance"]),
            )
            for index in range(reconstructed.shape[0])
        ]
        if not all(record["passed"] for record in distance_checks):
            raise ValueError("E007 Phase-3E-B derived distance geometry failed")
        records.append(
            {
                "split": specification["split"],
                "stratum": specification["stratum"],
                "regime": specification["regime"],
                "sample_ids": prepared["sample_ids"],
                "lengths": prepared["lengths"].tolist(),
                "physical_batch_size": len(prepared["sample_ids"]),
                "square_padded_length": prepared["square_padded_length"],
                "loss": float(loss.detach().cpu()),
                "output_shape": list(prediction.shape),
                "gradient_norms": gradient_norms,
                "output_centered_max_abs": centered_error,
                "deterministic_replay": deterministic,
                "additional_padding_maximum_error": padding_error,
                "additional_padding_exact_zero": padded_tail_zero,
                "distance_checks": distance_checks,
            }
        )
        model.zero_grad(set_to_none=True)
    if not all(gradient_coverage.values()):
        raise ValueError(f"E007 Phase-3E-B gradient coverage failed: {gradient_coverage}")
    equivariance = _equivariance_check(model, selected["train"][0]["row"], config, device)
    after = _parameter_sha256(model)
    if before != after:
        raise ValueError("E007 Phase-3E-B parameters changed without an optimizer")
    return {
        "parameter_count": parameter_count,
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameters_unchanged": True,
        "gradient_coverage": gradient_coverage,
        "batch_results": records,
        "equivariance": equivariance,
        "memory": _memory_telemetry(device),
    }


def _forward_backward_checks(
    selected: dict[str, list[dict[str, Any]]], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    with coordinate_model_execution_context(config["numerics"], device) as backend:
        result = _forward_backward_checks_active(selected, config, device)
    result["numerical_backend"] = backend
    return result


def _equivariance_check(
    model: EquivariantPairCoordinateUNet,
    row: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prepared = prepare_coordinate_batch(
        [row], float(config["coordinate_scale_angstrom"]), int(config["expected_downsample_factor"])
    )
    coordinates = prepared["coordinates"].to(device)
    mask = prepared["residue_mask"].to(device)
    continuity = prepared["chain_continuity_mask"].to(device)
    lengths = prepared["lengths"].to(device)
    timestep = torch.tensor([173], device=device)
    transforms = {
        "identity": torch.eye(3, device=device),
        "axis_rotation": torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device),
        "reflection": torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], device=device),
    }
    model.eval()
    records = []
    policy = coordinate_backend_policy(config["numerics"])
    active_backend = require_coordinate_backend_policy(policy, device)
    with torch.no_grad():
        base = model(coordinates, timestep, lengths, mask, continuity)
        replay = model(coordinates, timestep, lengths, mask, continuity)
        replay_error = float((replay["v_prediction"] - base["v_prediction"]).abs().max().cpu())
        for name, transform in transforms.items():
            transformed_coordinates = coordinates @ transform
            transformed = model(transformed_coordinates, timestep, lengths, mask, continuity)
            metrics = equivariance_metrics(
                reference=base,
                transformed=transformed,
                transformation=transform,
                reference_coordinates=coordinates,
                transformed_coordinates=transformed_coordinates,
                residue_mask=mask,
            )
            criterion = equivariance_criterion(
                metrics,
                absolute_tolerance=float(config["equivariance_atol"]),
                relative_l2_tolerance=float(config["equivariance_relative_l2_tolerance"]),
                coefficient_tolerance=float(config["equivariance_coefficient_tolerance"]),
            )
            records.append({"transformation": name, "metrics": metrics, "criterion": criterion})
    failed = [record for record in records if not record["criterion"]["passed"]]
    if replay_error != 0 or failed:
        raise ValueError(
            f"E007 Phase-3E-B strict O(3) equivariance failed: replay_error={replay_error}, failed={failed[:3]}"
        )
    return {
        "passed": True,
        "model_mode": "eval",
        "dropout_inactive": True,
        "autocast_enabled": False,
        "tf32_disabled": True,
        "active_backend": active_backend,
        "replay_error": replay_error,
        "transformations": records,
    }


def _write_selected_manifest(path: Path, selected: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    records = []
    for split in ("train", "validation"):
        records.extend({key: value for key, value in record.items() if key != "row"} for record in selected[split])
    _atomic_json(path, {"version": SMOKE_VERSION, "records": records, **NON_AUTHORIZING})
    return {"path": path.name, "sha256": sha256_file(path), "row_count": len(records)}


def run_real_loader_smoke(config_path: str | Path, *, mode: str) -> dict[str, Any]:
    """Run one bounded real-data smoke mode with no optimizer or parameter update."""
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    output = _output_for_mode(config, mode)
    _require_output_absent(output)
    prerequisites = verify_smoke_prerequisites(config)
    staging = output.with_name(f".{output.name}.inprogress")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started_utc = datetime.now(UTC).isoformat()
    started = time.monotonic()
    _atomic_json(
        staging / "heartbeat.json",
        {"status": "running", "mode": mode, "stage": "dataset_authorization", **NON_AUTHORIZING},
    )
    try:
        authorization_before = _authorize_dataset(config)
        protected_before = _protected_identity(authorization_before, prerequisites["hashes"])
        datasets = {
            split: E007CoordinateDataset(authorization_before, split=split) for split in ("train", "validation")
        }
        _atomic_json(
            staging / "heartbeat.json",
            {"status": "running", "mode": mode, "stage": "panel_and_leakage_scan", **NON_AUTHORIZING},
        )
        selected, panel_evidence = _scan_panels_and_leakage(datasets, config)
        panel_evidence["leakage"]["group_identifier_columns"]["available"] = prerequisites[
            "available_group_identifier_columns"
        ]
        if prerequisites["available_group_identifier_columns"]:
            raise ValueError("E007 Phase-3E-B group identifiers require an explicit projected audit implementation")
        loader_checks = _loader_checks(selected, config)
        leakage_detected = bool(panel_evidence["leakage"]["leakage_detected"])
        model_evidence = None
        model_executed = mode == "forward-backward-smoke" and not leakage_detected
        if model_executed:
            device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
            model_evidence = _forward_backward_checks(selected, config, device)
        manifest = _write_selected_manifest(staging / "selected_panel_manifest.json", selected)
        with _deterministic_gzip_text(staging / "detailed_metrics.jsonl.gz") as details:
            for record in loader_checks["split_checks"]:
                details.write(json.dumps({"kind": "loader", **record}, sort_keys=True) + "\n")
            if model_evidence is not None:
                for record in model_evidence["batch_results"]:
                    details.write(json.dumps({"kind": "forward_backward", **record}, sort_keys=True) + "\n")
        authorization_after = _authorize_dataset(config)
        protected_after = _protected_identity(authorization_after, prerequisites["hashes"])
        if protected_before != protected_after:
            raise ValueError("E007 Phase-3E-B protected inputs changed")
        report = {
            "version": SMOKE_VERSION,
            "status": "completed_non_authorizing",
            "mode": mode,
            "classification": (
                "dataset_review_required" if leakage_detected else "bounded_smoke_passed_non_authorizing"
            ),
            "output_dir": str(output),
            "started_utc": started_utc,
            "completed_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "coordinate_scale_angstrom": NORMALIZATION_SCALE_ANGSTROM,
            "coordinate_units_before_normalization": "angstrom",
            "normalization_interpretation": (
                "Phase-3E-A coordinate_payloads_scanned=false refers to external NPZ/source payloads; "
                "authorized projected sidecar coordinate columns were scanned"
            ),
            "capacity_interpretation": (
                "production capacity is an owner engineering choice; Phase 3D did not prove it necessary"
            ),
            "panel_evidence": panel_evidence,
            "loader_checks": loader_checks,
            "model_evidence": model_evidence,
            "selected_panel_manifest": manifest,
            "detailed_metrics": {
                "path": "detailed_metrics.jsonl.gz",
                "sha256": sha256_file(staging / "detailed_metrics.jsonl.gz"),
            },
            "protected_identity_before": protected_before,
            "protected_identity_after": protected_after,
            "protected_inputs_unchanged": True,
            "memory": _memory_telemetry(
                torch.device(config["device"] if torch.cuda.is_available() else "cpu") if model_executed else None
            ),
            "model_execution_refused_for_leakage": mode == "forward-backward-smoke" and leakage_detected,
            "model_created": model_executed,
            "forward_executed": model_executed,
            "backward_executed": model_executed,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": SMOKE_VERSION,
            "status": "completed_non_authorizing",
            "mode": mode,
            "classification": report["classification"],
            "report_sha256": sha256_file(staging / "report.json"),
            "selected_panel_manifest_sha256": manifest["sha256"],
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "mode": mode,
                "classification": report["classification"],
                "completed_utc": report["completed_utc"],
                "report_sha256": protocol["report_sha256"],
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return report
    except BaseException as error:
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "mode": mode,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "failed_utc": datetime.now(UTC).isoformat(),
                **NON_AUTHORIZING,
            },
        )
        raise
