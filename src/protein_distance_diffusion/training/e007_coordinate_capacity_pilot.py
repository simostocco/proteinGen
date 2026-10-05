"""Bounded synthetic-only capacity scaling pilot for E007 Phase 3D."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    centered_coordinate_noise,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_objective_pilot import (
    NON_AUTHORIZING,
    _atomic_json,
    _canonical_hash,
    _deterministic_gzip_text,
    _gradient_norms,
    _matched_quantiles,
    _parameter_change_groups,
)
from protein_distance_diffusion.training.e007_coordinate_smoke import (
    _evaluation_metrics,
    _joint_polymer_quality,
    _matrix_geometry,
    _paired_diffusion_batch,
    _sampling_geometry_valid,
    _sha256_file,
    _tensor_sha256,
    _trained_contract_checks,
    build_polymer_panels,
    require_finite_training_state,
)

PILOT_VERSION = "e007_coordinate_capacity_pilot_v1"
CAPACITIES = ("small", "medium", "production")
CLASSIFICATIONS = (
    "small_capacity_sufficient",
    "medium_capacity_required",
    "production_capacity_required",
    "capacity_improves_but_not_all_seeds",
    "no_capacity_benefit",
    "capacity_scaling_regresses_quality",
    "invalid_execution",
)
FORBIDDEN_OBJECTIVE_FIELDS = (
    "timestep_weight_bins",
    "x0_geometry_auxiliary_coefficients",
    "auxiliary_scale_bounds",
)


def uniform_coordinate_v_loss(
    prediction: torch.Tensor, target: torch.Tensor, residue_mask: torch.Tensor
) -> torch.Tensor:
    """Original Phase-3B-v2 valid-coordinate coordinate-v MSE."""
    if prediction.shape != target.shape or residue_mask.shape != prediction.shape[:2]:
        raise ValueError("E007 Phase-3D coordinate-v objective shapes disagree")
    valid = residue_mask[..., None].expand_as(target).to(prediction.dtype)
    return ((prediction - target).square() * valid).sum() / valid.sum().clamp_min(1)


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != PILOT_VERSION:
        raise ValueError("E007 Phase-3D configuration version contradiction")
    if tuple(config.get("capacities", ())) != CAPACITIES:
        raise ValueError("E007 Phase-3D capacities changed")
    if list(map(int, config.get("seeds", ()))) != [7301, 7302, 7303]:
        raise ValueError("E007 Phase-3D seeds changed")
    if list(map(int, config.get("lengths", ()))) != [16, 24, 31, 48]:
        raise ValueError("E007 Phase-3D lengths changed")
    if int(config.get("optimizer_updates_per_capacity_seed", 0)) != 1500:
        raise ValueError("E007 Phase-3D requires exactly 1,500 successful updates")
    if list(map(int, config.get("evaluation_updates", ()))) != [0, 100, 300, 600, 1000, 1500]:
        raise ValueError("E007 Phase-3D evaluation schedule changed")
    if int(config.get("train_replicates_per_family_length", 0)) != 8:
        raise ValueError("E007 Phase-3D requires 128 balanced training structures")
    if int(config.get("heldout_replicates_per_family_length", 0)) != 4:
        raise ValueError("E007 Phase-3D requires 64 balanced held-out structures")
    if int(config.get("sampling_replicates_per_length", 0)) < 8:
        raise ValueError("E007 Phase-3D requires at least eight samples per length")
    if any(field in config for field in FORBIDDEN_OBJECTIVE_FIELDS):
        raise ValueError("E007 Phase-3D must not contain Phase-3C objective corrections")
    if config.get("mixed_precision") is not False or int(config.get("batch_size", 0)) != 1:
        raise ValueError("E007 Phase-3D frozen execution contract changed")
    return config


def _verify_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    records = {
        "phase3b_v2_report": (
            Path(config["phase3b_v2"]["report_path"]),
            config["phase3b_v2"]["report_sha256"],
        ),
        "phase3b_v2_protocol": (
            Path(config["phase3b_v2"]["protocol_path"]),
            config["phase3b_v2"]["protocol_sha256"],
        ),
        "phase3c_config": (Path(config["phase3c"]["config_path"]), config["phase3c"]["config_sha256"]),
        "phase3c_report": (Path(config["phase3c"]["report_path"]), config["phase3c"]["report_sha256"]),
        "phase3c_protocol": (
            Path(config["phase3c"]["protocol_path"]),
            config["phase3c"]["protocol_sha256"],
        ),
        "coordinate_model_contract": (
            Path(config["coordinate_model_contract"]["path"]),
            config["coordinate_model_contract"]["sha256"],
        ),
    }
    hashes = {name: _sha256_file(path) for name, (path, _) in records.items()}
    for name, (_, expected) in records.items():
        if hashes[name] != expected:
            raise ValueError(f"E007 Phase-3D prerequisite hash contradiction: {name}")
    phase3b_report = json.loads(records["phase3b_v2_report"][0].read_text())
    phase3b_protocol = json.loads(records["phase3b_v2_protocol"][0].read_text())
    phase3c_report = json.loads(records["phase3c_report"][0].read_text())
    phase3c_protocol = json.loads(records["phase3c_protocol"][0].read_text())
    model_contract = json.loads(records["coordinate_model_contract"][0].read_text())
    if phase3b_report.get("status") != "completed" or phase3b_protocol.get("status") != "completed":
        raise ValueError("E007 Phase-3B-v2 evidence is incomplete")
    retained_thresholds = {
        key: value
        for key, value in phase3b_report["decision_thresholds"].items()
        if key != "maximum_optimizer_updates_per_seed"
    }
    observed_thresholds = {key: config["decision_thresholds"].get(key) for key in retained_thresholds}
    if observed_thresholds != retained_thresholds:
        raise ValueError("E007 Phase-3D changed a retained Phase-3B-v2 scientific threshold")
    required_phase3c = {
        "status": "completed",
        "classification": "objective_correction_regresses_quality",
        "protected_inputs_unchanged": True,
    }
    for field, expected in required_phase3c.items():
        if phase3c_report.get(field) != expected:
            raise ValueError(f"E007 Phase-3C evidence contradiction: {field}")
    if phase3c_report.get("pairing_evidence", {}).get("passed") is not True:
        raise ValueError("E007 Phase-3C pairing did not pass")
    if phase3c_report.get("selection_evidence", {}).get("eligible") != {
        "high_noise_balanced_v": False,
        "high_noise_balanced_v_plus_x0_geometry": False,
    }:
        raise ValueError("E007 Phase-3C corrected-arm eligibility contradiction")
    successful = sum(int(row.get("successful_updates", 0)) for row in phase3c_report.get("arm_seed_results", []))
    if successful != 9000:
        raise ValueError("E007 Phase-3C successful-update count contradiction")
    if phase3c_protocol.get("status") != "completed" or phase3c_protocol.get("authorizes_training") is not False:
        raise ValueError("E007 Phase-3C protocol is not completed non-authorizing evidence")
    if model_contract.get("status") != "phase3a_architecture_contract_verified_non_authorizing":
        raise ValueError("E007 coordinate-model contract status contradiction")
    generator_path = Path(model_contract["configuration"]["generator_path"])
    generator_hash = _sha256_file(generator_path)
    if generator_hash != model_contract["configuration"]["generator_sha256"]:
        raise ValueError("E007 production generator configuration hash contradiction")
    production = yaml.safe_load(generator_path.read_text())["model"]
    if config["models"]["production"] != production:
        raise ValueError("E007 Phase-3D production architecture differs from its pinned contract")
    return {
        "hashes": {**hashes, "production_generator_config": generator_hash},
        "phase3b_report": phase3b_report,
        "phase3c_report": phase3c_report,
        "model_contract": model_contract,
    }


def _parameter_counts(config: dict[str, Any]) -> dict[str, int]:
    counts = {
        name: sum(
            parameter.numel() for parameter in EquivariantPairCoordinateUNet(**config["models"][name]).parameters()
        )
        for name in CAPACITIES
    }
    expected = {name: int(value) for name, value in config["expected_parameter_counts"].items()}
    if counts != expected:
        raise ValueError(f"E007 Phase-3D parameter-count contradiction: observed={counts}, expected={expected}")
    low, high = map(int, config["medium_parameter_range"])
    if not low <= counts["medium"] <= high:
        raise ValueError("E007 Phase-3D medium capacity is outside its declared range")
    return counts


def _architecture_contract(config: dict[str, Any]) -> dict[str, Any]:
    models = config["models"]
    declared = set(config["capacity_parameters"])
    all_fields = set().union(*(model.keys() for model in models.values()))
    invariant = sorted(all_fields - declared)
    for field in invariant:
        values = {json.dumps(models[name].get(field), sort_keys=True) for name in CAPACITIES}
        if len(values) != 1:
            raise ValueError(f"E007 Phase-3D undeclared architecture difference: {field}")
    differences = sorted(
        field for field in declared if len({json.dumps(models[name].get(field)) for name in CAPACITIES}) > 1
    )
    return {
        "implementation": "EquivariantPairCoordinateUNet",
        "invariant_fields": invariant,
        "declared_capacity_fields": sorted(declared),
        "observed_differing_fields": differences,
        "model_input_contract": [
            "noisy_coordinates",
            "timesteps",
            "lengths",
            "residue_mask",
            "chain_continuity_mask",
        ],
        "prediction_parameterization": "coordinate_v",
        "sequence_inputs": False,
        "clean_coordinate_feature_inputs": False,
    }


def _build_frozen_panels(config: dict[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train, heldout, metadata = build_polymer_panels(config)
    published = json.loads(json.dumps(metadata, sort_keys=True))
    for field, expected in config["expected_panel_hashes"].items():
        if published.get(field) != expected:
            raise ValueError(f"E007 Phase-3D frozen panel hash contradiction: {field}")
    if len(train) != 128 or len(heldout) != 64:
        raise ValueError("E007 Phase-3D panel cardinality contradiction")
    if set(published["family_counts"]["train"].values()) != {32}:
        raise ValueError("E007 Phase-3D training families are not balanced")
    if set(published["length_counts"]["heldout"].values()) != {16}:
        raise ValueError("E007 Phase-3D held-out lengths are not balanced")
    return train, heldout, published


def plan_capacity_pilot(config_path: str | Path) -> dict[str, Any]:
    """Build a read-only Phase-3D plan without model execution or output writes."""
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3D output already exists: {output} or {staging}")
    prerequisites = _verify_prerequisites(config)
    counts = _parameter_counts(config)
    _, _, panels = _build_frozen_panels(config)
    architecture = _architecture_contract(config)
    samples_per_checkpoint = len(config["sampling_lengths"]) * int(config["sampling_replicates_per_length"])
    return {
        "status": "planned_non_authorizing",
        "version": PILOT_VERSION,
        "output_dir": str(output),
        "capacities": list(CAPACITIES),
        "parameter_counts": counts,
        "architecture_contract": architecture,
        "objective": {
            "name": "uniform_valid_coordinate_v_mse",
            "uniform_discrete_timestep_sampling": True,
            "timestep_weighting": False,
            "x0_auxiliaries": False,
        },
        "panel_contract": panels,
        "train_sample_count": 128,
        "heldout_sample_count": 64,
        "sampling_lengths": list(map(int, config["sampling_lengths"])),
        "sampling_replicates_per_length": int(config["sampling_replicates_per_length"]),
        "unconditional_samples_per_capacity_seed_checkpoint": samples_per_checkpoint,
        "unconditional_samples_per_capacity_seed": samples_per_checkpoint * len(config["evaluation_updates"]),
        "successful_updates_per_capacity_seed": int(config["optimizer_updates_per_capacity_seed"]),
        "total_planned_successful_updates": 3 * 3 * int(config["optimizer_updates_per_capacity_seed"]),
        "evaluation_updates": list(map(int, config["evaluation_updates"])),
        "isolated_spawned_processes": True,
        "prerequisite_hashes": prerequisites["hashes"],
        "config_sha256": _sha256_file(Path(config_path)),
        "optimizer_created": False,
        "forward_executed": False,
        "backward_executed": False,
        "sampling_executed": False,
        "real_data_loaded": False,
        "pretrained_weights_loaded": False,
        "synthetic_only": True,
        **NON_AUTHORIZING,
    }


def _sample_seed(config: dict[str, Any], seed: int, update: int, length_index: int, replicate: int) -> int:
    return int(config["sampling_seed"]) + seed * 100_000 + update * 100 + length_index * 10 + replicate


def sampling_identity(config: dict[str, Any], seed: int, update: int) -> dict[str, Any]:
    records = []
    for length_index, length in enumerate(map(int, config["sampling_lengths"])):
        for replicate in range(int(config["sampling_replicates_per_length"])):
            draw_seed = _sample_seed(config, seed, update, length_index, replicate)
            generator = torch.Generator().manual_seed(draw_seed)
            mask = torch.ones((1, length), dtype=torch.bool)
            initial = centered_coordinate_noise(torch.empty((1, length, 3)), mask, generator=generator)
            records.append((length, replicate, draw_seed, _tensor_sha256(initial)))
    return {"records": records, "sha256": _canonical_hash(records), "count": len(records)}


def _target_distributions(panel: list[Any], lengths: list[int]) -> dict[str, Any]:
    output = {}
    all_rows = []
    for length in lengths:
        rows = [_matrix_geometry(sample.coordinates) for sample in panel if sample.length == length]
        all_rows.extend(rows)
        output[str(length)] = {
            "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in rows])),
            "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in rows])),
            "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in rows])),
        }
    output["global"] = {
        "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in all_rows])),
        "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in all_rows])),
        "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in all_rows])),
    }
    return output


def _sample_row(
    coordinates: torch.Tensor, *, length: int, replicate: int, seed: int
) -> tuple[dict[str, Any], torch.Tensor]:
    distances = coordinates_to_distance_matrix(coordinates, diagnostic_float64=True)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    nonneighbor = torch.triu(torch.ones_like(upper), diagonal=2)
    pair_values = distances[upper]
    adjacent = torch.diagonal(distances, offset=1)
    neighbor_counts = ((distances < 8.0) & nonneighbor).sum(dim=1).double()
    fingerprint = _matched_quantiles(pair_values, [index / 20 for index in range(21)]).cpu()
    row = {
        "length": length,
        "replicate": replicate,
        "seed": seed,
        **_matrix_geometry(coordinates),
        "pair_distance_quantiles": [float(value) for value in _matched_quantiles(pair_values, [0.1, 0.5, 0.9])],
        "adjacent_distance_quantiles": [float(value) for value in _matched_quantiles(adjacent, [0.1, 0.5, 0.9])],
        "contact_density_6a": float((distances[nonneighbor] < 6.0).double().mean()),
        "contact_density_8a": float((distances[nonneighbor] < 8.0).double().mean()),
        "contact_density_10a": float((distances[nonneighbor] < 10.0).double().mean()),
        "neighborhood_count_mean_8a": float(neighbor_counts.mean()),
        "neighborhood_count_std_8a": float(neighbor_counts.std(unbiased=False)),
    }
    return row, fingerprint


def _duplicate_summary(rows: list[dict[str, Any]], fingerprints: list[torch.Tensor]) -> dict[str, Any]:
    exact, near, pairs = 0, 0, 0
    examples = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left]["length"] != rows[right]["length"]:
                continue
            pairs += 1
            error = float(torch.sqrt(((fingerprints[left] - fingerprints[right]) ** 2).mean()))
            exact += int(error == 0.0)
            near += int(error < 0.05)
            if error < 0.05 and len(examples) < 20:
                examples.append({"left": left, "right": right, "fingerprint_rmse": error})
    return {
        "comparable_pair_count": pairs,
        "exact_duplicate_count": exact,
        "near_duplicate_count": near,
        "exact_duplicate_fraction": exact / max(pairs, 1),
        "near_duplicate_fraction": near / max(pairs, 1),
        "bounded_near_duplicate_examples": examples,
    }


def _bootstrap_mean_ci(values: list[float], *, seed: int, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("E007 Phase-3D bootstrap requires observations")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = array[rng.integers(0, len(array), len(array))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _sampling_metric_summary(rows: list[dict[str, Any]], *, seed: int, replicates: int) -> dict[str, dict[str, Any]]:
    fields = (
        "adjacent_distance_mean",
        "adjacent_distance_std",
        "radius_of_gyration",
        "clash_fraction",
        "contact_density_6a",
        "contact_density_8a",
        "contact_density_10a",
        "neighborhood_count_mean_8a",
        "neighborhood_count_std_8a",
        "symmetry_error",
        "diagonal_error",
        "maximum_triangle_violation",
        "centred_gram_negative_eigenmass_fraction",
        "rank3_residual_fraction",
        "rank3_reconstruction_error",
    )
    summary = {}
    for index, field in enumerate(fields):
        values = [float(row[field]) for row in rows]
        summary[field] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "bootstrap_mean_ci_95": _bootstrap_mean_ci(
                values,
                seed=seed + index,
                replicates=replicates,
            ),
        }
    summary["finite_coordinate_fraction"] = {
        "mean": float(np.mean([bool(row["finite_coordinates"]) for row in rows])),
        "median": float(np.median([bool(row["finite_coordinates"]) for row in rows])),
        "bootstrap_mean_ci_95": _bootstrap_mean_ci(
            [float(bool(row["finite_coordinates"])) for row in rows],
            seed=seed + len(fields),
            replicates=replicates,
        ),
    }
    return summary


def _summarize_sampling(
    rows: list[dict[str, Any]], fingerprints: list[torch.Tensor], targets: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    by_length = {}
    for length in map(int, config["sampling_lengths"]):
        selected_indices = [index for index, row in enumerate(rows) if row["length"] == length]
        selected = [rows[index] for index in selected_indices]
        selected_fingerprints = [fingerprints[index] for index in selected_indices]
        target = targets[str(length)]
        by_length[str(length)] = {
            "count": len(selected),
            "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in selected])),
            "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in selected])),
            "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in selected])),
            "adjacent_absolute_error": abs(
                float(np.mean([row["adjacent_distance_mean"] for row in selected])) - target["adjacent_distance_mean"]
            ),
            "radius_absolute_error": abs(
                float(np.mean([row["radius_of_gyration"] for row in selected])) - target["radius_of_gyration_mean"]
            ),
            "pair_distance_median": float(np.median([row["pair_distance_quantiles"][1] for row in selected])),
            "contact_density_8a_mean": float(np.mean([row["contact_density_8a"] for row in selected])),
            "neighborhood_count_mean_8a": float(np.mean([row["neighborhood_count_mean_8a"] for row in selected])),
            "radius_mean_ci_95": _bootstrap_mean_ci(
                [row["radius_of_gyration"] for row in selected],
                seed=int(config["bootstrap_seed"]) + length,
                replicates=int(config["bootstrap_replicates"]),
            ),
            "metric_distributions": _sampling_metric_summary(
                selected,
                seed=int(config["bootstrap_seed"]) + length * 100,
                replicates=int(config["bootstrap_replicates"]),
            ),
            "diversity": _duplicate_summary(selected, selected_fingerprints),
        }
    target = targets["global"]
    global_summary = {
        "count": len(rows),
        "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in rows])),
        "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in rows])),
        "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in rows])),
    }
    global_summary["adjacent_absolute_error"] = abs(
        global_summary["adjacent_distance_mean"] - target["adjacent_distance_mean"]
    )
    global_summary["radius_absolute_error"] = abs(
        global_summary["radius_of_gyration_mean"] - target["radius_of_gyration_mean"]
    )
    global_summary["clash_fraction_excess"] = max(
        0.0, global_summary["clash_fraction_mean"] - target["clash_fraction_mean"]
    )
    global_summary["metric_distributions"] = _sampling_metric_summary(
        rows,
        seed=int(config["bootstrap_seed"]),
        replicates=int(config["bootstrap_replicates"]),
    )
    global_summary.update(_duplicate_summary(rows, fingerprints))
    return {"global": global_summary, "by_length": by_length}


def _write_sampling_artifact(path: Path, coordinates: list[torch.Tensor], rows: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        arrays = {f"sample_{index:03d}": value.detach().cpu().numpy() for index, value in enumerate(coordinates)}
        arrays["metadata_json"] = np.asarray(json.dumps(rows, sort_keys=True))
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return _sha256_file(path)


def _sampling_panel(
    model: EquivariantPairCoordinateUNet,
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    heldout: list[Any],
    *,
    seed: int,
    update: int,
    device: torch.device,
    artifact_path: Path,
) -> dict[str, Any]:
    rows, fingerprints, coordinate_records = [], [], []
    initial_records = []
    model.eval()
    for length_index, length in enumerate(map(int, config["sampling_lengths"])):
        for replicate in range(int(config["sampling_replicates_per_length"])):
            draw_seed = _sample_seed(config, seed, update, length_index, replicate)
            generator = torch.Generator(device=device).manual_seed(draw_seed)
            residue_mask = torch.ones((1, length), dtype=torch.bool, device=device)
            initial = centered_coordinate_noise(
                torch.empty((1, length, 3), device=device), residue_mask, generator=generator
            )
            initial_records.append((length, replicate, draw_seed, _tensor_sha256(initial)))
            sampled = diffusion.sample(model, length=length, seed=draw_seed, device=device)
            coordinates = sampled["coordinates"][0] * float(config["coordinate_scale_angstrom"])
            row, fingerprint = _sample_row(coordinates, length=length, replicate=replicate, seed=draw_seed)
            rows.append(row)
            fingerprints.append(fingerprint)
            coordinate_records.append(coordinates)
    targets = _target_distributions(heldout, list(map(int, config["sampling_lengths"])))
    summary = _summarize_sampling(rows, fingerprints, targets, config)
    geometry_valid = _sampling_geometry_valid(rows, float(config["decision_thresholds"]["euclidean_scaled_tolerance"]))
    joint = _joint_polymer_quality(rows, targets["global"], config["decision_thresholds"])
    artifact_hash = _write_sampling_artifact(artifact_path, coordinate_records, rows)
    return {
        "rows": rows,
        "summary": summary,
        "target_distributions": targets,
        "joint_polymer_quality": joint,
        "sampling_geometry_valid": geometry_valid,
        "initial_state_sha256": _canonical_hash(initial_records),
        "stochastic_draws_sha256": _canonical_hash([]),
        "artifact": {"path": artifact_path.name, "sha256": artifact_hash, "sample_count": len(rows)},
    }


def _evaluate(
    model: EquivariantPairCoordinateUNet,
    train: list[Any],
    heldout: list[Any],
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    *,
    seed: int,
    update: int,
    device: torch.device,
    directory: Path,
) -> dict[str, Any]:
    result = {
        "train": _evaluation_metrics(
            model,
            train,
            diffusion,
            scale=float(config["coordinate_scale_angstrom"]),
            corruption_seed=seed + 40000,
            device=device,
        ),
        "heldout": _evaluation_metrics(
            model,
            heldout,
            diffusion,
            scale=float(config["coordinate_scale_angstrom"]),
            corruption_seed=seed + 50000,
            device=device,
        ),
        "train_timestep_bins": {},
        "heldout_timestep_bins": {},
    }
    for bin_record in config["timestep_evaluation_bins"]:
        timestep = int(round(float(bin_record["fraction"]) * (diffusion.timesteps - 1)))
        name = str(bin_record["name"])
        for panel_name, panel, offset in (("train", train, 60000), ("heldout", heldout, 70000)):
            metrics = _evaluation_metrics(
                model,
                panel,
                diffusion,
                scale=float(config["coordinate_scale_angstrom"]),
                corruption_seed=seed + offset,
                device=device,
                forced_timestep=timestep,
            )
            metrics["timestep"] = timestep
            result[f"{panel_name}_timestep_bins"][name] = metrics
    artifact = directory / f"sampling_update_{update:04d}.npz"
    result["unconditional_sampling"] = _sampling_panel(
        model,
        diffusion,
        config,
        heldout,
        seed=seed,
        update=update,
        device=device,
        artifact_path=artifact,
    )
    return result


def _externalize_details(directory: Path, evaluations: dict[str, Any]) -> dict[str, Any]:
    path = directory / "evaluation_details.jsonl.gz"
    count = 0
    with _deterministic_gzip_text(path) as handle:
        for update in sorted(evaluations, key=int):
            evaluation = evaluations[update]
            records = {"train": evaluation["train"], "heldout": evaluation["heldout"]}
            for panel in ("train", "heldout"):
                records.update(
                    {
                        f"{panel}_timestep_bin:{name}": value
                        for name, value in evaluation[f"{panel}_timestep_bins"].items()
                    }
                )
            for scope, record in records.items():
                rows = record.pop("per_sample")
                record["per_sample_count"] = len(rows)
                record["per_sample_sha256"] = _canonical_hash(rows)
                for row in rows:
                    handle.write(json.dumps({"update": int(update), "scope": scope, **row}, sort_keys=True) + "\n")
                    count += 1
            sample_rows = evaluation["unconditional_sampling"].pop("rows")
            evaluation["unconditional_sampling"]["per_sample_count"] = len(sample_rows)
            evaluation["unconditional_sampling"]["per_sample_sha256"] = _canonical_hash(sample_rows)
            for row in sample_rows:
                handle.write(
                    json.dumps({"update": int(update), "scope": "unconditional", **row}, sort_keys=True) + "\n"
                )
                count += 1
    return {"path": path.name, "row_count": count, "sha256": _sha256_file(path)}


def _run_capacity_seed(config_path: str, capacity: str, seed: int, directory_path: str) -> None:
    config = _load_config(config_path)
    prerequisites = _verify_prerequisites(config)
    train, heldout, panels = _build_frozen_panels(config)
    directory = Path(directory_path)
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(
        directory / "heartbeat.json",
        {
            "status": "running",
            "capacity": capacity,
            "seed": seed,
            "successful_optimizer_updates": 0,
            **NON_AUTHORIZING,
        },
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = EquivariantPairCoordinateUNet(**config["models"][capacity]).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    evaluation_updates = set(map(int, config["evaluation_updates"]))
    evaluations = {}
    training_identity = {"sample_order": [], "timesteps": [], "coordinate_noise": [], "targets": []}
    gradient_observed = {"full_model": False, "pair_grid_unet_trunk": False, "coefficient_head": False}
    latest_gradients = {name: 0.0 for name in gradient_observed}
    successful = 0
    evaluations["0"] = _evaluate(
        model, train, heldout, diffusion, config, seed=seed, update=0, device=device, directory=directory
    )
    trajectory_path = directory / "trajectory.jsonl.gz"
    with _deterministic_gzip_text(trajectory_path) as metrics:
        for attempted in range(1, int(config["optimizer_updates_per_capacity_seed"]) + 1):
            model.train()
            sample = train[(attempted - 1) % len(train)]
            batch, diffused = _paired_diffusion_batch(
                sample,
                diffusion,
                scale=float(config["coordinate_scale_angstrom"]),
                draw_seed=seed * 1_000_000 + attempted,
                rotate=True,
                device=device,
            )
            training_identity["sample_order"].append(sample.sample_id)
            training_identity["timesteps"].append(int(diffused.timesteps.item()))
            training_identity["coordinate_noise"].append(_tensor_sha256(diffused.coordinate_noise))
            training_identity["targets"].append(_tensor_sha256(diffused.coordinate_v_target))
            prediction = model(
                diffused.noisy_coordinates,
                diffused.timesteps,
                batch["lengths"],
                batch["residue_mask"],
                batch["continuity"],
            )["v_prediction"]
            loss = uniform_coordinate_v_loss(prediction, diffused.coordinate_v_target, batch["residue_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            named = require_finite_training_state(loss, model, seed=seed, update=attempted)
            latest_gradients = _gradient_norms(named)
            for name, value in latest_gradients.items():
                gradient_observed[name] |= value > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            successful += 1
            if attempted % int(config["metrics_frequency"]) == 0 or attempted in evaluation_updates:
                metrics.write(
                    json.dumps(
                        {
                            "capacity": capacity,
                            "seed": seed,
                            "successful_update": successful,
                            "sample_id": sample.sample_id,
                            "timestep": int(diffused.timesteps.item()),
                            "coordinate_v_loss": float(loss.detach()),
                            "gradient_norms": latest_gradients,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics.flush()
            if attempted in evaluation_updates:
                evaluations[str(attempted)] = _evaluate(
                    model,
                    train,
                    heldout,
                    diffusion,
                    config,
                    seed=seed,
                    update=attempted,
                    device=device,
                    directory=directory,
                )
                _atomic_json(
                    directory / "heartbeat.json",
                    {
                        "status": "running",
                        "capacity": capacity,
                        "seed": seed,
                        "successful_optimizer_updates": successful,
                        "latest_evaluation_update": attempted,
                        "updated_utc": datetime.now(UTC).isoformat(),
                        **NON_AUTHORIZING,
                    },
                )
    details = _externalize_details(directory, evaluations)
    final = evaluations[str(successful)]
    before = evaluations["0"]

    def relative(first: float, last: float) -> float:
        return (first - last) / max(abs(first), 1e-12)

    elapsed = time.monotonic() - started
    pairing = {name: _canonical_hash(values) for name, values in training_identity.items()}
    pairing.update(
        {
            "panel": _canonical_hash(panels),
            "evaluation_corruptions": _canonical_hash(
                [
                    (
                        update,
                        panel,
                        evaluations[str(update)][panel]["noisy_coordinate_sha256"],
                        evaluations[str(update)][panel]["coordinate_v_target_sha256"],
                    )
                    for update in config["evaluation_updates"]
                    for panel in ("train", "heldout")
                ]
            ),
            "sampling_initial_states": _canonical_hash(
                [
                    evaluations[str(update)]["unconditional_sampling"]["initial_state_sha256"]
                    for update in config["evaluation_updates"]
                ]
            ),
            "sampling_stochastic_draws": _canonical_hash([]),
        }
    )
    result = {
        "capacity": capacity,
        "seed": seed,
        "successful_updates": successful,
        "optimizer_overflows": 0,
        "finite_losses_and_gradients": True,
        "gradient_norms": latest_gradients,
        "nonzero_gradient_observed": gradient_observed,
        "parameter_change_by_group": _parameter_change_groups(model, initial),
        "evaluations": evaluations,
        "evaluation_details": details,
        "relative_improvements": {
            "train_coordinate_v_mse": relative(
                before["train"]["means"]["coordinate_v_mse"], final["train"]["means"]["coordinate_v_mse"]
            ),
            "heldout_coordinate_v_mse": relative(
                before["heldout"]["means"]["coordinate_v_mse"],
                final["heldout"]["means"]["coordinate_v_mse"],
            ),
            "heldout_pair_distance_rmse": relative(
                before["heldout"]["means"]["pair_distance_rmse_angstrom"],
                final["heldout"]["means"]["pair_distance_rmse_angstrom"],
            ),
        },
        "sampling_geometry_valid": final["unconditional_sampling"]["sampling_geometry_valid"],
        "joint_polymer_quality": final["unconditional_sampling"]["joint_polymer_quality"],
        "trained_contract_checks": _trained_contract_checks(
            model, tolerance=float(config["decision_thresholds"]["equivariance_atol"]), device=device
        ),
        "pairing_hashes": pairing,
        "wall_time_seconds": elapsed,
        "samples_per_second": successful / max(elapsed, 1e-12),
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "pretrained_weights_loaded": False,
        "source_hashes": prerequisites["hashes"],
        **NON_AUTHORIZING,
    }
    checkpoint = directory / "final_synthetic_checkpoint.pt"
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "version": PILOT_VERSION,
            "capacity": capacity,
            "seed": seed,
            "model": model.state_dict(),
            "successful_optimizer_updates": successful,
            "synthetic_only": True,
            **NON_AUTHORIZING,
        },
        temporary,
    )
    temporary.replace(checkpoint)
    result["checkpoint"] = {"path": checkpoint.name, "sha256": _sha256_file(checkpoint)}
    result["trajectory_sha256"] = _sha256_file(trajectory_path)
    _atomic_json(directory / "result.json", result)
    _atomic_json(
        directory / "heartbeat.json",
        {
            "status": "completed",
            "capacity": capacity,
            "seed": seed,
            "successful_optimizer_updates": successful,
            "result_sha256": _sha256_file(directory / "result.json"),
            "completed_utc": datetime.now(UTC).isoformat(),
            **NON_AUTHORIZING,
        },
    )


def _worker(config_path: str, capacity: str, seed: int, directory: str) -> None:
    try:
        _run_capacity_seed(config_path, capacity, seed, directory)
    except BaseException as error:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            path / "failure.json",
            {
                "capacity": capacity,
                "seed": seed,
                "type": type(error).__name__,
                "message": str(error),
                **NON_AUTHORIZING,
            },
        )
        _atomic_json(
            path / "heartbeat.json",
            {
                "status": "failed",
                "capacity": capacity,
                "seed": seed,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "failed_utc": datetime.now(UTC).isoformat(),
                **NON_AUTHORIZING,
            },
        )
        raise


def _run_isolated(config_path: Path, capacity: str, seed: int, directory: Path) -> dict[str, Any]:
    process = mp.get_context("spawn").Process(target=_worker, args=(str(config_path), capacity, seed, str(directory)))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"E007 Phase-3D child failed: capacity={capacity}, seed={seed}, exit={process.exitcode}")
    return json.loads((directory / "result.json").read_text())


def verify_pairing(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "panel",
        "sample_order",
        "timesteps",
        "coordinate_noise",
        "targets",
        "evaluation_corruptions",
        "sampling_initial_states",
        "sampling_stochastic_draws",
    )
    by_seed = {}
    for seed in sorted({int(row["seed"]) for row in results}):
        rows = [row for row in results if int(row["seed"]) == seed]
        if {row["capacity"] for row in rows} != set(CAPACITIES):
            raise ValueError(f"E007 Phase-3D missing capacity for seed {seed}")
        evidence = {}
        for field in fields:
            values = {row["pairing_hashes"][field] for row in rows}
            if len(values) != 1:
                raise ValueError(f"E007 Phase-3D pairing contradiction: seed={seed}, field={field}")
            evidence[field] = values.pop()
        by_seed[str(seed)] = evidence
    return {"passed": True, "by_seed": by_seed}


def _capacity_eligible(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> bool:
    for row in rows:
        sampling = row["evaluations"]["1500"]["unconditional_sampling"]
        per_length = sampling["summary"]["by_length"]
        if not (
            row["successful_updates"] == 1500
            and row["finite_losses_and_gradients"]
            and row["optimizer_overflows"] == 0
            and all(row["nonzero_gradient_observed"].values())
            and all(value > 0 for value in row["parameter_change_by_group"].values())
            and all(row["trained_contract_checks"].values())
            and row["sampling_geometry_valid"]
            and row["joint_polymer_quality"]["passed"]
            and sampling["summary"]["global"]["near_duplicate_fraction"]
            <= float(thresholds["maximum_near_duplicate_fraction"])
            and row["relative_improvements"]["train_coordinate_v_mse"]
            >= float(thresholds["minimum_train_v_mse_relative_improvement"])
            and row["relative_improvements"]["heldout_coordinate_v_mse"]
            >= float(thresholds["minimum_heldout_v_mse_relative_improvement_per_seed"])
            and row["relative_improvements"]["heldout_pair_distance_rmse"]
            >= float(thresholds["minimum_heldout_pair_rmse_relative_improvement_per_seed"])
            and all(record["count"] >= 8 for record in per_length.values())
        ):
            return False
    return True


def _per_length_regression(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], tolerance: float
) -> dict[str, Any]:
    failures = []
    changes = {}
    for seed in sorted(row["seed"] for row in baseline):
        left = next(row for row in baseline if row["seed"] == seed)
        right = next(row for row in candidate if row["seed"] == seed)
        for length in left["evaluations"]["1500"]["unconditional_sampling"]["summary"]["by_length"]:
            left_metrics = left["evaluations"]["1500"]["unconditional_sampling"]["summary"]["by_length"][length]
            right_metrics = right["evaluations"]["1500"]["unconditional_sampling"]["summary"]["by_length"][length]
            for metric in ("adjacent_absolute_error", "radius_absolute_error", "clash_fraction_mean"):
                change = (right_metrics[metric] - left_metrics[metric]) / max(abs(left_metrics[metric]), 1e-12)
                key = f"{seed}:N{length}:{metric}"
                changes[key] = change
                if change > tolerance:
                    failures.append(key)
    return {"passed": not failures, "failures": failures, "relative_changes": changes}


def _capacity_vector(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "heldout_v_mse": float(
            np.mean([row["evaluations"]["1500"]["heldout"]["means"]["coordinate_v_mse"] for row in rows])
        ),
        "heldout_pair_rmse": float(
            np.mean([row["evaluations"]["1500"]["heldout"]["means"]["pair_distance_rmse_angstrom"] for row in rows])
        ),
        "joint_error": float(np.mean([row["joint_polymer_quality"]["joint_error"] for row in rows])),
        "duplicate_fraction": float(
            np.mean(
                [
                    row["evaluations"]["1500"]["unconditional_sampling"]["summary"]["global"]["near_duplicate_fraction"]
                    for row in rows
                ]
            )
        ),
    }


def _paired_pareto_evidence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    thresholds: dict[str, Any],
    *,
    seed: int = 17301,
    replicates: int = 5000,
) -> dict[str, Any]:
    """Test predeclared paired, lower-is-better capacity improvements."""
    baseline_by_seed = {int(row["seed"]): row for row in baseline}
    candidate_by_seed = {int(row["seed"]): row for row in candidate}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise ValueError("E007 Phase-3D Pareto comparison is not seed-paired")

    def metrics(row: dict[str, Any]) -> dict[str, float]:
        final = row["evaluations"]["1500"]
        sampling = final["unconditional_sampling"]
        return {
            "heldout_v_mse": float(final["heldout"]["means"]["coordinate_v_mse"]),
            "heldout_pair_rmse": float(final["heldout"]["means"]["pair_distance_rmse_angstrom"]),
            "joint_error": float(row["joint_polymer_quality"]["joint_error"]),
            "duplicate_fraction": float(sampling["summary"]["global"]["near_duplicate_fraction"]),
        }

    improvements: dict[str, list[float]] = {name: [] for name in metrics(baseline[0])}
    for paired_seed in sorted(baseline_by_seed):
        left = metrics(baseline_by_seed[paired_seed])
        right = metrics(candidate_by_seed[paired_seed])
        for name in improvements:
            improvements[name].append((left[name] - right[name]) / max(abs(left[name]), 1e-12))
    rng = np.random.default_rng(seed)
    summaries = {}
    for metric_index, (name, values) in enumerate(improvements.items()):
        array = np.asarray(values, dtype=np.float64)
        bootstrap = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            selection = rng.integers(0, len(array), len(array))
            bootstrap[index] = array[selection].mean()
        summaries[name] = {
            "paired_relative_improvements": values,
            "mean_relative_improvement": float(array.mean()),
            "bootstrap_ci_95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "bootstrap_stream_index": metric_index,
        }
    material = float(thresholds["material_pareto_relative_improvement"])
    lower_bound = float(thresholds["material_pareto_bootstrap_lower_bound"])
    no_mean_regression = all(row["mean_relative_improvement"] >= 0.0 for row in summaries.values())
    supported_metrics = [
        name
        for name, row in summaries.items()
        if row["mean_relative_improvement"] >= material and row["bootstrap_ci_95"][0] > lower_bound
    ]
    return {
        "passed": no_mean_regression and bool(supported_metrics),
        "no_mean_regression": no_mean_regression,
        "material_supported_metrics": supported_metrics,
        "metrics": summaries,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
    }


def classify_capacity(results: list[dict[str, Any]], thresholds: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    grouped = {name: [row for row in results if row["capacity"] == name] for name in CAPACITIES}
    if any(len(rows) != 3 for rows in grouped.values()) or any(
        not row.get("finite_losses_and_gradients", False) for row in results
    ):
        return "invalid_execution", {}
    eligible = {name: _capacity_eligible(rows, thresholds) for name, rows in grouped.items()}
    tolerance = float(thresholds["maximum_per_length_relative_regression"])
    regressions = {
        "medium_vs_small": _per_length_regression(grouped["small"], grouped["medium"], tolerance),
        "production_vs_small": _per_length_regression(grouped["small"], grouped["production"], tolerance),
        "production_vs_medium": _per_length_regression(grouped["medium"], grouped["production"], tolerance),
    }
    eligible["medium"] = eligible["medium"] and regressions["medium_vs_small"]["passed"]
    eligible["production"] = (
        eligible["production"]
        and regressions["production_vs_small"]["passed"]
        and regressions["production_vs_medium"]["passed"]
    )
    vectors = {name: _capacity_vector(rows) for name, rows in grouped.items()}
    pareto = {
        "medium_vs_small": _paired_pareto_evidence(grouped["small"], grouped["medium"], thresholds),
        "production_vs_small": _paired_pareto_evidence(grouped["small"], grouped["production"], thresholds),
        "production_vs_medium": _paired_pareto_evidence(grouped["medium"], grouped["production"], thresholds),
    }
    evidence = {
        "eligible": eligible,
        "per_length_regression": regressions,
        "pareto_vectors": vectors,
        "paired_bootstrap_pareto": pareto,
    }
    eligible_names = [name for name in CAPACITIES if eligible[name]]
    if eligible_names:
        selected = eligible_names[0]
        for candidate in eligible_names[1:]:
            comparison = f"{candidate}_vs_{selected}"
            if pareto[comparison]["passed"]:
                selected = candidate
        evidence["selected_capacity"] = selected
        evidence["selection_rule"] = (
            "smallest eligible unless a larger eligible capacity has supported material Pareto gain"
        )
        classifications = {
            "small": "small_capacity_sufficient",
            "medium": "medium_capacity_required",
            "production": "production_capacity_required",
        }
        return classifications[selected], evidence
    if any(record["failures"] for record in regressions.values()):
        return "capacity_scaling_regresses_quality", evidence
    passed_seeds = {name: sum(_capacity_eligible([row], thresholds) for row in rows) for name, rows in grouped.items()}
    evidence["passing_seed_counts"] = passed_seeds
    if any(value > 0 for value in passed_seeds.values()):
        return "capacity_improves_but_not_all_seeds", evidence
    return "no_capacity_benefit", evidence


def run_capacity_pilot(config_path: str | Path) -> dict[str, Any]:
    """Run all capacity/seed children and atomically publish non-authorizing evidence."""
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    plan = plan_capacity_pilot(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    before = _verify_prerequisites(config)["hashes"]
    started = datetime.now(UTC).isoformat()
    _atomic_json(
        staging / "heartbeat.json",
        {"status": "running", "stage": "capacity_seed_children", "started_utc": started, **NON_AUTHORIZING},
    )
    try:
        results = []
        for capacity in CAPACITIES:
            for seed in map(int, config["seeds"]):
                results.append(
                    _run_isolated(config_path, capacity, seed, staging / "capacities" / capacity / f"seed_{seed}")
                )
                _atomic_json(
                    staging / "heartbeat.json",
                    {
                        "status": "running",
                        "stage": "capacity_seed_children",
                        "completed_children": len(results),
                        "total_children": len(CAPACITIES) * len(config["seeds"]),
                        "latest_capacity": capacity,
                        "latest_seed": seed,
                        "updated_utc": datetime.now(UTC).isoformat(),
                        **NON_AUTHORIZING,
                    },
                )
        pairing = verify_pairing(results)
        classification, selection = classify_capacity(results, config["decision_thresholds"])
        after = _verify_prerequisites(config)["hashes"]
        if before != after:
            raise ValueError("E007 Phase-3D protected inputs changed during execution")
        report = {
            **plan,
            "status": "completed",
            "classification": classification,
            "pairing_evidence": pairing,
            "selection_evidence": selection,
            "capacity_seed_results": results,
            "successful_optimizer_updates": sum(row["successful_updates"] for row in results),
            "protected_hashes_before": before,
            "protected_hashes_after": after,
            "protected_inputs_unchanged": True,
            "optimizer_created": True,
            "forward_executed": True,
            "backward_executed": True,
            "sampling_executed": True,
            "completed_utc": datetime.now(UTC).isoformat(),
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": PILOT_VERSION,
            "status": "completed",
            "classification": classification,
            "report_sha256": _sha256_file(staging / "report.json"),
            "successful_optimizer_updates": report["successful_optimizer_updates"],
            "pairing_evidence": pairing,
            "protected_inputs_unchanged": True,
            "synthetic_only": True,
            "non_production": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "classification": classification,
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
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "failed_utc": datetime.now(UTC).isoformat(),
                **NON_AUTHORIZING,
            },
        )
        raise
