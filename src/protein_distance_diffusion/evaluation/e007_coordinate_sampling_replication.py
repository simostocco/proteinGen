"""Independent read-only checkpoint sampling replication for E007 Phase 3G."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
    centered_coordinate_noise,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file
from protein_distance_diffusion.training.e007_coordinate_real_pilot import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SCALE,
    _authorize,
    _bounded_ranked_rows,
    _clean_validation_ids,
)

VERSION = "e007_coordinate_sampling_replication_v1"
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created_for_training": False,
    "optimizer_created": False,
    "backward_performed": False,
    "optimizer_updates": 0,
    "dataset_modified": False,
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
}
NUMERIC_FIELDS = (
    "centroid_max_abs_angstrom",
    "centroid_rms_angstrom",
    "distance_symmetry_error_angstrom",
    "distance_diagonal_error_angstrom",
    "sampled_maximum_triangle_violation_angstrom",
    "centered_gram_negative_eigenmass_fraction",
    "adjacent_reference_error_angstrom",
    "radius_of_gyration_absolute_reference_error_angstrom",
    "radius_of_gyration_relative_reference_error",
    "non_neighbor_clash_fraction",
    "non_neighbor_clash_reference_excess",
    "contact_density_6a_reference_error",
    "contact_density_8a_reference_error",
    "contact_density_10a_reference_error",
    "long_range_contact_density_8a",
    "neighborhood_count_mean_8a",
    "neighborhood_count_std_8a",
)
PARETO_FIELDS = (
    "adjacent_reference_error_mean",
    "radius_relative_reference_error_mean",
    "clash_reference_excess_mean",
    "contact_density_reference_error_mean",
    "coordinate_duplicate_fraction",
    "rigid_distance_duplicate_fraction",
    "adjacent_failure_fraction",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], compression: str) -> None:
    if not rows:
        raise ValueError("E007 Phase-3G cannot publish an empty Parquet table")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression=compression)
    temporary.replace(path)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def directory_fingerprint(directory: Path) -> str:
    """Match the immutable shell inventory fingerprint used for Phase 3F."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(f"{sha256_file(path)}  {path}\n".encode())
    return digest.hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase-3G configuration version contradiction")
    if payload.get("lengths") != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase-3G sampling lengths changed")
    if int(payload.get("samples_per_length", 0)) != 32:
        raise ValueError("E007 Phase-3G requires exactly 32 samples per checkpoint/length block")
    if [int(item["optimizer_update"]) for item in payload.get("checkpoints", [])] != [250, 750, 1000]:
        raise ValueError("E007 Phase-3G checkpoint set changed")
    if float(payload.get("coordinate_scale_angstrom", 0)) != EXPECTED_SCALE:
        raise ValueError("E007 Phase-3G coordinate scale changed")
    if int(payload.get("expected_parameter_count", 0)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3G parameter-count contract changed")
    if float(payload["metrics"].get("adjacent_error_limit_angstrom", -1)) != 1.0:
        raise ValueError("E007 Phase-3G original adjacent-error gate changed")
    seeds = paired_seed_records(payload)
    known = set(map(int, payload.get("known_prior_sampling_seeds", [])))
    if any(int(row["seed"]) in known for row in seeds):
        raise ValueError("E007 Phase-3G seed overlaps a known Phase-3B/3D/3F seed")
    return payload


def verify_prerequisites(config: Mapping[str, Any]) -> dict[str, Any]:
    phase3f = config["phase3f"]
    source = Path(phase3f["source_dir"])
    files = {
        "phase3f_report": (source / "report.json", phase3f["report_sha256"]),
        "phase3f_protocol": (source / "protocol.json", phase3f["protocol_sha256"]),
        "phase3f_config": (Path(phase3f["config_path"]), phase3f["config_sha256"]),
        "decision_correction_report": (
            Path(phase3f["decision_correction_report_path"]),
            phase3f["decision_correction_report_sha256"],
        ),
        "decision_correction_protocol": (
            Path(phase3f["decision_correction_protocol_path"]),
            phase3f["decision_correction_protocol_sha256"],
        ),
    }
    for item in config["checkpoints"]:
        files[f"checkpoint_{int(item['optimizer_update'])}"] = (Path(item["path"]), item["sha256"])
    observed = {}
    for name, (path, expected) in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"E007 Phase-3G prerequisite is absent: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != expected:
            raise ValueError(f"E007 Phase-3G prerequisite hash contradiction: {name}")
    fingerprint = directory_fingerprint(source)
    if fingerprint != phase3f["aggregate_fingerprint"]:
        raise ValueError("E007 Phase-3F aggregate fingerprint contradiction")
    report = json.loads((source / "report.json").read_text())
    protocol = json.loads((source / "protocol.json").read_text())
    correction = json.loads(Path(phase3f["decision_correction_report_path"]).read_text())
    if report.get("optimizer_updates") != 1000 or protocol.get("optimizer_updates") != 1000:
        raise ValueError("E007 Phase-3F completion evidence contradicts 1,000 updates")
    corrected = correction.get("decision_audit", {}).get("corrected_classification")
    if corrected != "denoising_learned_but_sampling_not_learned":
        raise ValueError("E007 Phase-3F corrected decision contradiction")
    return {"hashes": observed, "phase3f_aggregate_fingerprint": fingerprint}


def paired_seed_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build checkpoint-independent initial-noise identities."""
    records = []
    base = int(config["sampling_seed_base"])
    namespace = str(config["sampling_seed_namespace"])
    for length_index, length in enumerate(map(int, config["lengths"])):
        for sample_index in range(int(config["samples_per_length"])):
            seed = base + length_index * 100_000 + sample_index
            records.append(
                {
                    "length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "noise_identity_sha256": _canonical_sha(
                        {
                            "namespace": namespace,
                            "seed": seed,
                            "length": length,
                            "shape": [1, length, 3],
                            "noise_algorithm": "centered_coordinate_noise_torch_generator_v1",
                        }
                    ),
                    "reverse_draw_provenance_sha256": _canonical_sha(
                        {
                            "sampler": "deterministic_ddim",
                            "diffusion_steps": int(config["diffusion_steps"]),
                            "stochastic_reverse_draws": 0,
                        }
                    ),
                }
            )
    return records


def plan_sampling_replication(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    prerequisites = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3G output exists: {output} or {staging}")
    seeds = paired_seed_records(config)
    block_count = len(config["checkpoints"]) * len(config["lengths"])
    total = block_count * int(config["samples_per_length"])
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "output_dir": str(output),
        "checkpoints": [int(item["optimizer_update"]) for item in config["checkpoints"]],
        "lengths": config["lengths"],
        "samples_per_checkpoint_length": config["samples_per_length"],
        "block_count": block_count,
        "total_samples": total,
        "paired_seed_count": len(seeds),
        "paired_seed_manifest_sha256": _canonical_sha(seeds),
        "independent_seed_namespace": config["sampling_seed_namespace"],
        "original_adjacent_error_limit_angstrom": 1.0,
        "checkpoint_length_block_execution": True,
        "resume_boundary": "completed checkpoint-length block",
        "model_created": False,
        "coordinate_samples_generated": False,
        "output_created": False,
        "prerequisite_hashes": prerequisites,
        **NON_AUTHORIZING,
    }


def initial_noise_provenance(length: int, seed: int, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    noise = centered_coordinate_noise(torch.empty((1, length, 3), device=device), mask, generator=generator)
    value = noise.detach().cpu().contiguous()
    return {
        "tensor_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "centroid_max_abs": float(value.mean(dim=1).abs().max()),
    }


def _quantiles(values: torch.Tensor, probabilities: Sequence[float]) -> list[float]:
    if values.numel() == 0:
        return []
    q = values.new_tensor(list(probabilities))
    return [float(value) for value in torch.quantile(values, q).cpu()]


def _sampled_triangle_error(distances: torch.Tensor, *, count: int, seed: int) -> float:
    length = distances.shape[0]
    if length < 3:
        return 0.0
    generator = torch.Generator(device=distances.device).manual_seed(seed)
    indices = torch.randint(length, (count, 3), generator=generator, device=distances.device)
    left = distances[indices[:, 0], indices[:, 2]]
    right = distances[indices[:, 0], indices[:, 1]] + distances[indices[:, 1], indices[:, 2]]
    return float((left - right).clamp_min(0).max().cpu())


def coordinate_metrics(
    coordinates: torch.Tensor,
    *,
    checkpoint: int,
    length: int,
    sample_index: int,
    seed: int,
    reference: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Compute only O(3)-compatible physical diagnostics from one independent sample."""
    values = coordinates[0].detach().double() * float(config["coordinate_scale_angstrom"])
    finite = bool(torch.isfinite(values).all())
    distances = coordinates_to_distance_matrix(values, diagnostic_float64=True)
    length_observed = values.shape[0]
    if length_observed != length:
        raise ValueError("E007 Phase-3G sampled length contradiction")
    indices = torch.arange(length, device=values.device)
    separation = (indices[:, None] - indices[None, :]).abs()
    upper_non_neighbor = torch.triu(separation > 1, diagonal=1)
    upper_long_range = torch.triu(
        separation >= int(config["metrics"]["long_range_minimum_sequence_separation"]), diagonal=1
    )
    adjacent = distances.diagonal(offset=1)
    rg = values.square().sum(dim=-1).mean().sqrt()
    squared = distances.square()
    centered_gram = -0.5 * (
        squared - squared.mean(dim=0, keepdim=True) - squared.mean(dim=1, keepdim=True) + squared.mean()
    )
    eigenvalues = torch.linalg.eigvalsh(centered_gram)
    negative_mass = eigenvalues.clamp_max(0).abs().sum() / eigenvalues.abs().sum().clamp_min(1e-12)
    threshold = float(config["metrics"]["neighborhood_threshold_angstrom"])
    neighborhoods = ((distances < threshold) & (separation > 1)).sum(dim=1).double()
    contacts = {}
    for contact in map(float, config["metrics"]["contact_thresholds_angstrom"]):
        key = f"contact_density_{int(contact)}a"
        density = float((distances[upper_non_neighbor] < contact).double().mean())
        contacts[key] = density
        contacts[f"{key}_reference_error"] = abs(density - float(reference[key]))
    clash = float((distances[upper_non_neighbor] < 3.0).double().mean())
    adjacent_mean = float(adjacent.mean())
    adjacent_error = abs(adjacent_mean - float(reference["adjacent_distance_mean_angstrom"]))
    rg_value = float(rg)
    rg_absolute = abs(rg_value - float(reference["radius_of_gyration_angstrom"]))
    rg_relative = rg_absolute / max(float(reference["radius_of_gyration_angstrom"]), 1e-12)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    coordinate_sha = hashlib.sha256(values.cpu().contiguous().numpy().tobytes()).hexdigest()
    rigid_values = distances[upper].cpu().contiguous()
    rigid_sha = hashlib.sha256(rigid_values.numpy().tobytes()).hexdigest()
    centroid = values.mean(dim=0)
    triangle_error = _sampled_triangle_error(
        distances,
        count=int(config["metrics"]["triangle_diagnostic_triplets"]),
        seed=seed + 97,
    )
    row = {
        "checkpoint_update": checkpoint,
        "length": length,
        "sample_index": sample_index,
        "seed": seed,
        "finite_coordinates": finite,
        "biological_mask_exact": True,
        "padded_position_count": 0,
        "padded_zeros_exact": True,
        "centroid_max_abs_angstrom": float(centroid.abs().max()),
        "centroid_rms_angstrom": float(centroid.square().mean().sqrt()),
        "o3_compatible_derived_geometry": True,
        "distance_diagonal_error_angstrom": float(distances.diagonal().abs().max()),
        "distance_symmetry_error_angstrom": float((distances - distances.T).abs().max()),
        "triangle_valid_by_euclidean_construction": finite,
        "sampled_maximum_triangle_violation_angstrom": triangle_error,
        "centered_gram_negative_eigenmass_fraction": float(negative_mass),
        "adjacent_distance_mean_angstrom": adjacent_mean,
        "adjacent_distance_quantiles_angstrom": _quantiles(adjacent, [0.1, 0.5, 0.9]),
        "adjacent_reference_error_angstrom": adjacent_error,
        "adjacent_original_gate_pass": adjacent_error <= float(config["metrics"]["adjacent_error_limit_angstrom"]),
        "radius_of_gyration_angstrom": rg_value,
        "radius_of_gyration_absolute_reference_error_angstrom": rg_absolute,
        "radius_of_gyration_relative_reference_error": rg_relative,
        "non_neighbor_clash_fraction": clash,
        "non_neighbor_clash_reference_excess": max(0.0, clash - float(reference["non_neighbor_clash_fraction"])),
        **contacts,
        "long_range_contact_density_8a": float((distances[upper_long_range] < 8.0).double().mean()),
        "neighborhood_count_mean_8a": float(neighborhoods.mean()),
        "neighborhood_count_std_8a": float(neighborhoods.std(unbiased=False)),
        "neighborhood_count_quantiles_8a": _quantiles(neighborhoods, [0.1, 0.5, 0.9]),
        "coordinate_sha256": coordinate_sha,
        "rigid_distance_sha256": rigid_sha,
    }
    return row, rigid_values


def _reference_record(coordinates: torch.Tensor, config: Mapping[str, Any]) -> dict[str, float]:
    values = center_coordinates(coordinates[None].double(), torch.ones((1, len(coordinates)), dtype=torch.bool))[0]
    distances = coordinates_to_distance_matrix(values, diagnostic_float64=True)
    indices = torch.arange(len(values))
    separation = (indices[:, None] - indices[None, :]).abs()
    non_neighbor = torch.triu(separation > 1, diagonal=1)
    record = {
        "adjacent_distance_mean_angstrom": float(distances.diagonal(offset=1).mean()),
        "radius_of_gyration_angstrom": float(values.square().sum(dim=-1).mean().sqrt()),
        "non_neighbor_clash_fraction": float((distances[non_neighbor] < 3.0).double().mean()),
    }
    for contact in map(float, config["metrics"]["contact_thresholds_angstrom"]):
        record[f"contact_density_{int(contact)}a"] = float((distances[non_neighbor] < contact).double().mean())
    return record


def _reference_distributions(config: Mapping[str, Any]) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    phase3f_config = yaml.safe_load(Path(config["phase3f"]["config_path"]).read_text())
    authorization = _authorize(phase3f_config)
    clean_ids = _clean_validation_ids(phase3f_config)
    strata = phase3f_config["length_strata"]
    selected = _bounded_ranked_rows(
        E007CoordinateDataset(authorization, split="validation"),
        strata=strata,
        capacities={item["name"]: int(phase3f_config["validation_samples_per_stratum"]) for item in strata},
        seed=int(phase3f_config["seed"]),
        purpose="identity_30_clean_validation_panel",
        permitted_ids=clean_ids,
        expected_eligible_count=int(phase3f_config["clean_validation"]["accepted_validation_count"]),
    )
    panel = json.loads((Path(config["phase3f"]["source_dir"]) / "panel_manifest.json").read_text())["validation"]
    observed_ids = {name: [str(row["sample_id"]) for row in rows] for name, rows in selected.items()}
    if observed_ids != panel:
        raise ValueError("E007 Phase-3G reconstructed validation reference panel contradiction")
    references = {}
    for stratum in strata:
        length = int(stratum["maximum"])
        rows = [_reference_record(row["coordinates"], config) for row in selected[stratum["name"]]]
        references[length] = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    return references, {"sample_ids": observed_ids, "sample_id_sha256": _canonical_sha(observed_ids)}


def _bootstrap_ci(values: Sequence[float], *, seed: int, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("E007 Phase-3G bootstrap requires observations")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = array[rng.integers(0, len(array), len(array))].mean()
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def summarize_block(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("E007 Phase-3G cannot summarize an empty block")
    summary: dict[str, Any] = {"count": len(rows), "metrics": {}}
    bootstrap_seed = int(config["aggregation"]["bootstrap_seed"])
    replicates = int(config["aggregation"]["bootstrap_replicates"])
    for offset, field in enumerate(NUMERIC_FIELDS):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        summary["metrics"][field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "standard_deviation": float(values.std()),
            "quantile_05": float(np.quantile(values, 0.05)),
            "quantile_25": float(np.quantile(values, 0.25)),
            "quantile_75": float(np.quantile(values, 0.75)),
            "quantile_95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
            "mean_bootstrap_ci_95": _bootstrap_ci(values, seed=bootstrap_seed + offset, replicates=replicates),
        }
    passes = [float(bool(row["adjacent_original_gate_pass"])) for row in rows]
    failures = len(rows) - int(sum(passes))
    worst = max(rows, key=lambda row: (float(row["adjacent_reference_error_angstrom"]), int(row["seed"])))
    ordered = sorted(rows, key=lambda row: (float(row["adjacent_reference_error_angstrom"]), int(row["seed"])))
    representatives = {}
    for label, quantile in (("median", 0.5), ("p95", 0.95), ("worst", 1.0)):
        target = float(np.quantile([row["adjacent_reference_error_angstrom"] for row in ordered], quantile))
        selected = min(ordered, key=lambda row: (abs(row["adjacent_reference_error_angstrom"] - target), row["seed"]))
        representatives[label] = {
            "sample_index": selected["sample_index"],
            "seed": selected["seed"],
            "adjacent_reference_error_angstrom": selected["adjacent_reference_error_angstrom"],
            "artifact_path": selected["artifact_path"],
        }
    joint = [
        bool(row["adjacent_original_gate_pass"])
        and float(row["radius_of_gyration_relative_reference_error"])
        <= float(config["metrics"]["maximum_radius_of_gyration_relative_error"])
        and float(row["non_neighbor_clash_fraction"]) <= float(config["metrics"]["maximum_non_neighbor_clash_fraction"])
        for row in rows
    ]
    summary.update(
        {
            "original_all_record_gate_pass": failures == 0,
            "adjacent_pass_count": int(sum(passes)),
            "adjacent_failure_count": failures,
            "adjacent_pass_fraction": float(np.mean(passes)),
            "adjacent_pass_fraction_bootstrap_ci_95": _bootstrap_ci(
                passes,
                seed=bootstrap_seed + 1000,
                replicates=replicates,
            ),
            "joint_polymer_quality_pass_fraction": float(np.mean(joint)),
            "worst_record": {
                "sample_index": worst["sample_index"],
                "seed": worst["seed"],
                "adjacent_reference_error_angstrom": worst["adjacent_reference_error_angstrom"],
            },
            "one_record_removal_allows_hard_gate_pass": failures == 1,
            "one_record_removal_is_descriptive_only": True,
            "representative_samples": representatives,
        }
    )
    return summary


def duplicate_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    coordinate = Counter(str(row["coordinate_sha256"]) for row in rows)
    rigid = Counter(str(row["rigid_distance_sha256"]) for row in rows)
    coordinate_duplicates = sum(value * (value - 1) // 2 for value in coordinate.values())
    rigid_duplicates = sum(value * (value - 1) // 2 for value in rigid.values())
    pairs = sum(count * (count - 1) // 2 for count in Counter(int(row["length"]) for row in rows).values())
    return {
        "comparable_pair_count": pairs,
        "coordinate_duplicate_pair_count": coordinate_duplicates,
        "rigid_distance_duplicate_pair_count": rigid_duplicates,
        "coordinate_duplicate_fraction": coordinate_duplicates / max(pairs, 1),
        "rigid_distance_duplicate_fraction": rigid_duplicates / max(pairs, 1),
        "coordinate_tolerance_angstrom": float(config["metrics"]["duplicate_coordinate_tolerance_angstrom"]),
        "rigid_distance_tolerance_angstrom": float(config["metrics"]["duplicate_rigid_distance_tolerance_angstrom"]),
        "note": (
            "exact SHA duplicates are reported; tolerance thresholds are retained for future bounded near-duplicate "
            "extension"
        ),
    }


def checkpoint_pareto(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for checkpoint in sorted({int(row["checkpoint_update"]) for row in rows}):
        selected = [row for row in rows if int(row["checkpoint_update"]) == checkpoint]
        duplicate = duplicate_summary(selected, config)
        candidates.append(
            {
                "checkpoint_update": checkpoint,
                "adjacent_reference_error_mean": float(
                    np.mean([row["adjacent_reference_error_angstrom"] for row in selected])
                ),
                "radius_relative_reference_error_mean": float(
                    np.mean([row["radius_of_gyration_relative_reference_error"] for row in selected])
                ),
                "clash_reference_excess_mean": float(
                    np.mean([row["non_neighbor_clash_reference_excess"] for row in selected])
                ),
                "contact_density_reference_error_mean": float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    row["contact_density_6a_reference_error"],
                                    row["contact_density_8a_reference_error"],
                                    row["contact_density_10a_reference_error"],
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "coordinate_duplicate_fraction": duplicate["coordinate_duplicate_fraction"],
                "rigid_distance_duplicate_fraction": duplicate["rigid_distance_duplicate_fraction"],
                "adjacent_failure_fraction": float(
                    np.mean([not row["adjacent_original_gate_pass"] for row in selected])
                ),
            }
        )
    dominated_by = {}
    nondominated = []
    for candidate in candidates:
        dominators = [
            other["checkpoint_update"]
            for other in candidates
            if other is not candidate
            and all(float(other[key]) <= float(candidate[key]) for key in PARETO_FIELDS)
            and any(float(other[key]) < float(candidate[key]) for key in PARETO_FIELDS)
        ]
        if dominators:
            dominated_by[str(candidate["checkpoint_update"])] = dominators
        else:
            nondominated.append(candidate["checkpoint_update"])
    return {
        "direction": "all declared objectives minimized independently; no scalar score",
        "objectives": list(PARETO_FIELDS),
        "candidates": candidates,
        "nondominated_checkpoints": nondominated,
        "dominated_by": dominated_by,
        "latest_checkpoint_selected_automatically": False,
    }


def classify_replication(
    block_summaries: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    pareto: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    numerical = all(
        bool(row["finite_coordinates"])
        and bool(row["biological_mask_exact"])
        and bool(row["padded_zeros_exact"])
        and bool(row["deterministic_replay"])
        and bool(row["o3_compatible_derived_geometry"])
        and float(row["centroid_max_abs_angstrom"]) <= float(config["metrics"]["centroid_max_abs_tolerance_angstrom"])
        and float(row["distance_diagonal_error_angstrom"]) == 0
        and float(row["distance_symmetry_error_angstrom"])
        <= float(config["metrics"]["distance_symmetry_tolerance_angstrom"])
        and bool(row["triangle_valid_by_euclidean_construction"])
        and float(row["sampled_maximum_triangle_violation_angstrom"])
        <= float(config["metrics"]["triangle_tolerance_angstrom"])
        and float(row["centered_gram_negative_eigenmass_fraction"])
        <= float(config["metrics"]["gram_negative_eigenmass_tolerance"])
        for row in rows
    )
    if not numerical:
        return "numerical_or_replay_failure"
    failures = {key: int(value["adjacent_failure_count"]) for key, value in block_summaries.items()}
    if not failures or sum(int(value["count"]) for value in block_summaries.values()) != len(rows):
        return "inconclusive_requires_review"
    if sum(failures.values()) == 0:
        return "sampling_replication_verified_all_records"
    decision = config["decision"]
    if sum(failures.values()) <= int(decision["isolated_tail_max_total_failures"]) and max(failures.values()) <= int(
        decision["isolated_tail_max_failures_per_block"]
    ):
        return "sampling_replication_verified_except_isolated_tail"
    systematic = float(decision["systematic_failure_fraction"])
    rates = {
        key: value["adjacent_failure_count"] / value["count"]
        for key, value in block_summaries.items()
        if key.startswith("1000:")
    }
    if rates.get("1000:500", 0.0) >= systematic and all(
        rate < systematic for key, rate in rates.items() if key != "1000:500"
    ):
        return "systematic_n500_sampling_deficiency"
    if sum(rate >= systematic for rate in rates.values()) >= 2:
        return "broader_length_dependent_sampling_deficiency"
    if len(pareto.get("nondominated_checkpoints", [])) > 1:
        return "checkpoint_tradeoff_requires_review"
    return "inconclusive_requires_review"


def _memory(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "cuda_allocated_mib": None,
            "cuda_reserved_mib": None,
            "peak_cuda_allocated_mib": None,
            "peak_cuda_reserved_mib": None,
        }
    torch.cuda.synchronize(device)
    return {
        "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def _enforce_memory(device: torch.device, config: Mapping[str, Any]) -> dict[str, float | None]:
    memory = _memory(device)
    if device.type == "cuda" and (
        float(memory["peak_cuda_allocated_mib"]) > float(config["memory"]["maximum_cuda_allocated_mib"])
        or float(memory["peak_cuda_reserved_mib"]) > float(config["memory"]["maximum_cuda_reserved_mib"])
    ):
        raise MemoryError(f"E007 Phase-3G CUDA memory envelope exceeded: {memory}")
    return memory


def _checkpoint_model(item: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    payload = torch.load(item["path"], map_location="cpu", weights_only=False)
    if int(payload.get("optimizer_update", -1)) != int(item["optimizer_update"]):
        raise ValueError("E007 Phase-3G checkpoint update contradiction")
    if payload.get("configuration_sha256") != config["phase3f"]["config_sha256"]:
        raise ValueError("E007 Phase-3G checkpoint configuration contradiction")
    phase3f_config = yaml.safe_load(Path(config["phase3f"]["config_path"]).read_text())
    model = EquivariantPairCoordinateUNet(**phase3f_config["model"])
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3G model parameter-count contradiction")
    model.load_state_dict(payload["model"])
    model.requires_grad_(False).eval().to(device)
    del payload
    return model


def _sample_block(
    *,
    checkpoint: Mapping[str, Any],
    length: int,
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    reference: Mapping[str, float],
    config: Mapping[str, Any],
    staging: Path,
    device: torch.device,
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    rows = []
    sample_dir, temporary_dir = _prepare_uncommitted_sample_block(
        staging,
        checkpoint=int(checkpoint["optimizer_update"]),
        length=length,
    )
    seeds = [row for row in paired_seed_records(config) if int(row["length"]) == length]
    for identity in seeds:
        result = diffusion.sample(model, length=length, seed=int(identity["seed"]), device=device)
        replay = diffusion.sample(model, length=length, seed=int(identity["seed"]), device=device)
        coordinates = result["coordinates"].detach().cpu()
        replay_coordinates = replay["coordinates"].detach().cpu()
        deterministic = bool(torch.equal(coordinates, replay_coordinates))
        row, _ = coordinate_metrics(
            coordinates,
            checkpoint=int(checkpoint["optimizer_update"]),
            length=length,
            sample_index=int(identity["sample_index"]),
            seed=int(identity["seed"]),
            reference=reference,
            config=config,
        )
        noise = initial_noise_provenance(length, int(identity["seed"]), device)
        artifact = temporary_dir / f"sample-{int(identity['sample_index']):03d}.npz"
        final_artifact = sample_dir / artifact.name
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            coordinates=coordinates.numpy(),
            distance_matrix=result["distance_matrix"].detach().cpu().numpy(),
        )
        temporary.replace(artifact)
        row.update(
            {
                "deterministic_replay": deterministic,
                "artifact_path": final_artifact.relative_to(staging).as_posix(),
                "artifact_sha256": sha256_file(artifact),
                "noise_identity_sha256": identity["noise_identity_sha256"],
                "initial_noise_tensor_sha256": noise["tensor_sha256"],
                "reverse_draw_provenance_sha256": identity["reverse_draw_provenance_sha256"],
            }
        )
        rows.append(row)
        _enforce_memory(device, config)
        if progress is not None:
            progress(len(rows))
    temporary_dir.replace(sample_dir)
    return rows


def _block_path(staging: Path, checkpoint: int, length: int) -> Path:
    return staging / "blocks" / f"step-{checkpoint:04d}-length-{length}.parquet"


def _prepare_uncommitted_sample_block(staging: Path, *, checkpoint: int, length: int) -> tuple[Path, Path]:
    checkpoint_dir = staging / "samples" / f"step-{checkpoint:04d}"
    sample_dir = checkpoint_dir / f"length-{length}"
    temporary_dir = checkpoint_dir / f".length-{length}.inprogress"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for uncommitted in (temporary_dir, sample_dir):
        if uncommitted.exists():
            shutil.rmtree(uncommitted)
    temporary_dir.mkdir()
    return sample_dir, temporary_dir


def _completed_blocks(staging: Path) -> dict[str, dict[str, Any]]:
    path = staging / "block_journal.json"
    return json.loads(path.read_text()) if path.exists() else {}


def verify_completed_blocks(
    staging: Path,
    journal: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    permitted = {
        f"{int(checkpoint['optimizer_update'])}:{int(length)}"
        for checkpoint in config["checkpoints"]
        for length in config["lengths"]
    }
    for key, record in journal.items():
        if key not in permitted:
            raise ValueError(f"E007 Phase-3G journal contains an unknown block: {key}")
        relative = Path(str(record["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"E007 Phase-3G journal block path escapes staging: {relative}")
        path = staging / relative
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"E007 Phase-3G completed block hash contradiction: {key}")
        parquet = pq.ParquetFile(path)
        expected_rows = int(config["samples_per_length"])
        if int(record["row_count"]) != expected_rows or parquet.metadata.num_rows != expected_rows:
            raise ValueError(f"E007 Phase-3G completed block row-count contradiction: {key}")


def _inventory(staging: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
    return [
        {"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name
    ]


def validate_count_conservation(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, int]:
    expected_per_block = int(config["samples_per_length"])
    counts = Counter((int(row["checkpoint_update"]), int(row["length"])) for row in rows)
    expected = {
        (int(checkpoint["optimizer_update"]), int(length)): expected_per_block
        for checkpoint in config["checkpoints"]
        for length in config["lengths"]
    }
    if counts != expected:
        raise ValueError(f"E007 Phase-3G count conservation contradiction: {dict(counts)} != {expected}")
    return {
        "block_count": len(counts),
        "sample_count": len(rows),
        "samples_per_block": expected_per_block,
    }


def validate_paired_provenance(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["length"]), int(row["sample_index"])), []).append(row)
    expected_checkpoints = {int(item["optimizer_update"]) for item in config["checkpoints"]}
    for identity, records in grouped.items():
        if {int(row["checkpoint_update"]) for row in records} != expected_checkpoints:
            raise ValueError(f"E007 Phase-3G paired checkpoint membership contradiction: {identity}")
        for field in (
            "seed",
            "noise_identity_sha256",
            "initial_noise_tensor_sha256",
            "reverse_draw_provenance_sha256",
        ):
            if len({row[field] for row in records}) != 1:
                raise ValueError(f"E007 Phase-3G paired provenance contradiction: {identity} {field}")
    expected_identities = len(config["lengths"]) * int(config["samples_per_length"])
    if len(grouped) != expected_identities:
        raise ValueError("E007 Phase-3G paired provenance identity count contradiction")
    compact = [
        {
            "length": identity[0],
            "sample_index": identity[1],
            "seed": records[0]["seed"],
            "noise_identity_sha256": records[0]["noise_identity_sha256"],
            "initial_noise_tensor_sha256": records[0]["initial_noise_tensor_sha256"],
            "reverse_draw_provenance_sha256": records[0]["reverse_draw_provenance_sha256"],
        }
        for identity, records in sorted(grouped.items())
    ]
    return {
        "paired_identity_count": len(compact),
        "checkpoint_count_per_identity": len(expected_checkpoints),
        "paired_provenance_sha256": _canonical_sha(compact),
        "records": compact,
    }


def run_sampling_replication(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Execute inference-only sampling; checkpoints and source evidence remain immutable."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    prerequisites_before = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists():
        raise FileExistsError(f"E007 Phase-3G output already exists: {output}")
    if resume:
        if not staging.is_dir():
            raise FileNotFoundError("E007 Phase-3G resume requires an existing staging directory")
    elif staging.exists():
        raise FileExistsError(f"E007 Phase-3G staging already exists: {staging}")
    else:
        staging.mkdir(parents=True)
        (staging / "blocks").mkdir()
    heartbeat_path = staging / "heartbeat.json"
    started = time.monotonic()

    def heartbeat(status: str, **values: Any) -> None:
        _atomic_json(
            heartbeat_path,
            {"status": status, "updated_utc": _utc_now(), **values, **NON_AUTHORIZING},
        )

    journal = _completed_blocks(staging)
    verify_completed_blocks(staging, journal, config)
    heartbeat("initializing", completed_blocks=len(journal), total_blocks=15)
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3G configured audit requires CUDA")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        references, reference_provenance = _reference_distributions(config)
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        with coordinate_model_execution_context(config["numerics"], device) as backend:
            for checkpoint in config["checkpoints"]:
                pending = [
                    length
                    for length in map(int, config["lengths"])
                    if f"{int(checkpoint['optimizer_update'])}:{length}" not in journal
                ]
                if not pending:
                    continue
                model = _checkpoint_model(checkpoint, config, device)
                for length in pending:
                    key = f"{int(checkpoint['optimizer_update'])}:{length}"
                    completed_before = sum(int(item["row_count"]) for item in journal.values())

                    def block_progress(
                        completed_in_block: int,
                        *,
                        prior_samples: int = completed_before,
                        checkpoint_update: int = int(checkpoint["optimizer_update"]),
                        current_length: int = length,
                    ) -> None:
                        completed_samples = prior_samples + completed_in_block
                        elapsed = time.monotonic() - started
                        heartbeat(
                            "running",
                            checkpoint_update=checkpoint_update,
                            length=current_length,
                            completed_samples=completed_samples,
                            total_samples=480,
                            memory=_memory(device),
                            elapsed_seconds=elapsed,
                            eta_seconds=elapsed / max(completed_samples, 1) * (480 - completed_samples),
                        )

                    rows = _sample_block(
                        checkpoint=checkpoint,
                        length=length,
                        model=model,
                        diffusion=diffusion,
                        reference=references[length],
                        config=config,
                        staging=staging,
                        device=device,
                        progress=block_progress,
                    )
                    block_path = _block_path(staging, int(checkpoint["optimizer_update"]), length)
                    _atomic_parquet(block_path, rows, config["publication"]["parquet_compression"])
                    journal[key] = {
                        "checkpoint_update": int(checkpoint["optimizer_update"]),
                        "length": length,
                        "row_count": len(rows),
                        "path": block_path.relative_to(staging).as_posix(),
                        "sha256": sha256_file(block_path),
                    }
                    _atomic_json(staging / "block_journal.json", journal)
                    completed_samples = sum(int(item["row_count"]) for item in journal.values())
                    elapsed = time.monotonic() - started
                    total_samples = 480
                    heartbeat(
                        "running",
                        checkpoint_update=int(checkpoint["optimizer_update"]),
                        length=length,
                        completed_samples=completed_samples,
                        total_samples=total_samples,
                        memory=_memory(device),
                        elapsed_seconds=elapsed,
                        eta_seconds=elapsed / max(completed_samples, 1) * (total_samples - completed_samples),
                    )
                del model
                torch.cuda.empty_cache()
            if set(journal) != {
                f"{int(checkpoint['optimizer_update'])}:{int(length)}"
                for checkpoint in config["checkpoints"]
                for length in config["lengths"]
            }:
                raise ValueError("E007 Phase-3G completed block set is incomplete")
            rows = []
            for key, record in sorted(journal.items()):
                path = staging / record["path"]
                if sha256_file(path) != record["sha256"]:
                    raise ValueError(f"E007 Phase-3G completed block hash contradiction: {key}")
                table = pq.read_table(path)
                if table.num_rows != int(config["samples_per_length"]):
                    raise ValueError(f"E007 Phase-3G completed block row-count contradiction: {key}")
                rows.extend(table.to_pylist())
            count_conservation = validate_count_conservation(rows, config)
            paired_provenance = validate_paired_provenance(rows, config)
            block_summaries = {
                key: summarize_block(
                    [
                        row
                        for row in rows
                        if int(row["checkpoint_update"]) == int(key.split(":")[0])
                        and int(row["length"]) == int(key.split(":")[1])
                    ],
                    config,
                )
                for key in sorted(journal)
            }
            checkpoint_summaries = {}
            for checkpoint in config["checkpoints"]:
                update = int(checkpoint["optimizer_update"])
                selected = [row for row in rows if int(row["checkpoint_update"]) == update]
                checkpoint_summaries[str(update)] = {
                    **summarize_block(selected, config),
                    "duplicate_detection": duplicate_summary(selected, config),
                    "lengths": list(map(int, config["lengths"])),
                }
            pareto = checkpoint_pareto(rows, config)
            classification = classify_replication(block_summaries, rows, pareto, config)
            per_sample_path = staging / "per_sample_metrics.parquet"
            _atomic_parquet(per_sample_path, rows, config["publication"]["parquet_compression"])
            sampling_manifest = {
                "seed_records": paired_seed_records(config),
                "seed_records_sha256": _canonical_sha(paired_seed_records(config)),
                "block_journal": journal,
                "reference_panel": reference_provenance,
                "sample_count": len(rows),
                "count_conservation": count_conservation,
                "paired_noise_and_draw_provenance": paired_provenance,
            }
            _atomic_json(staging / "sampling_manifest.json", sampling_manifest)
            _atomic_json(staging / "checkpoint_pareto.json", pareto)
        prerequisites_after = verify_prerequisites(config)
        if prerequisites_after != prerequisites_before:
            raise ValueError("E007 Phase-3G protected inputs changed")
        inventory = _inventory(staging)
        _atomic_json(
            staging / "artifact_inventory.json",
            {"artifacts": inventory, "aggregate_sha256": _canonical_sha(inventory)},
        )
        report = {
            "status": "completed_non_authorizing",
            "version": VERSION,
            "classification": classification,
            "original_phase3f_gate_revised": False,
            "original_adjacent_error_limit_angstrom": 1.0,
            "sample_count": len(rows),
            "block_summaries": block_summaries,
            "checkpoint_summaries": checkpoint_summaries,
            "checkpoint_pareto": pareto,
            "protected_inputs_unchanged": True,
            "numerical_backend": backend,
            "memory": _memory(device),
            "elapsed_seconds": time.monotonic() - started,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "source_report_sha256": config["phase3f"]["report_sha256"],
            "source_protocol_sha256": config["phase3f"]["protocol_sha256"],
            "source_aggregate_fingerprint": config["phase3f"]["aggregate_fingerprint"],
            "report_sha256": sha256_file(staging / "report.json"),
            "per_sample_metrics_sha256": sha256_file(per_sample_path),
            "sampling_manifest_sha256": sha256_file(staging / "sampling_manifest.json"),
            "checkpoint_pareto_sha256": sha256_file(staging / "checkpoint_pareto.json"),
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            "completed_utc": _utc_now(),
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        heartbeat("completed", completed_samples=480, total_samples=480, report_sha256=protocol["report_sha256"])
        staging.replace(output)
        return {
            "status": report["status"],
            "classification": classification,
            "output_dir": str(output),
            **NON_AUTHORIZING,
        }
    except KeyboardInterrupt as error:
        heartbeat(
            "interrupted",
            error_type=type(error).__name__,
            error_message="SIGINT",
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
    except MemoryError as error:
        heartbeat(
            "memory_limit_exceeded",
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
    except BaseException as error:
        heartbeat(
            "failed",
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
