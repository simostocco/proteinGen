"""Bounded paired synthetic objective-correction pilot for E007 Phase 3C."""

from __future__ import annotations

import gzip
import io
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    centered_coordinate_noise,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_losses import (
    ObjectiveCorrectionWeights,
    coordinate_objective_correction_losses,
)
from protein_distance_diffusion.training.e007_coordinate_smoke import (
    _canonical_hash,
    _evaluation_metrics,
    _joint_polymer_quality,
    _matrix_geometry,
    _paired_diffusion_batch,
    _parameter_change,
    _sampling_geometry_valid,
    _sha256_file,
    _target_distribution,
    _tensor_sha256,
    _trained_contract_checks,
    build_polymer_panels,
    require_finite_training_state,
)

PILOT_VERSION = "e007_coordinate_objective_correction_pilot_v1"
ARMS = (
    "uniform_v_control",
    "high_noise_balanced_v",
    "high_noise_balanced_v_plus_x0_geometry",
)
CLASSIFICATIONS = (
    "objective_correction_verified",
    "high_noise_weighting_sufficient",
    "x0_geometry_auxiliaries_required",
    "correction_improves_but_not_all_seeds",
    "no_objective_correction_benefit",
    "objective_correction_regresses_quality",
    "invalid_execution",
)
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    """Write gzip text with a fixed header timestamp for reproducible hashes."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                yield text


def _externalize_evaluation_rows(directory: Path, evaluations: dict[str, Any]) -> dict[str, Any]:
    path = directory / "evaluation_details.jsonl.gz"
    row_count = 0
    with _deterministic_gzip_text(path) as handle:
        for update in sorted(evaluations, key=int):
            records = {
                "train": evaluations[update]["train"],
                "heldout": evaluations[update]["heldout"],
                **{
                    f"heldout_timestep_bin:{name}": value
                    for name, value in evaluations[update]["heldout_timestep_bins"].items()
                },
            }
            for scope, record in records.items():
                rows = record.pop("per_sample")
                record["per_sample_count"] = len(rows)
                record["per_sample_sha256"] = _canonical_hash(rows)
                for row in rows:
                    handle.write(json.dumps({"update": int(update), "scope": scope, **row}, sort_keys=True) + "\n")
                    row_count += 1
    return {"path": path.name, "row_count": row_count, "sha256": _sha256_file(path)}


def _state_hash(model: torch.nn.Module) -> str:
    return _canonical_hash([(name, _tensor_sha256(value)) for name, value in sorted(model.state_dict().items())])


def normalized_timestep_weights(
    diffusion_steps: int, bins: list[dict[str, Any]]
) -> tuple[torch.Tensor, list[dict[str, Any]], str]:
    """Expand and exactly normalize an explicit monotonic timestep schedule."""
    if diffusion_steps < 2 or not bins:
        raise ValueError("E007 Phase-3C timestep weighting is empty")
    raw = torch.full((diffusion_steps,), float("nan"), dtype=torch.float64)
    expanded = []
    previous_weight = -math.inf
    expected_start = 0
    for record in bins:
        start, end = int(record["start"]), int(record["end"])
        weight = float(record["unnormalized_weight"])
        if start != expected_start or end < start or end >= diffusion_steps:
            raise ValueError("E007 Phase-3C timestep bins must cover the schedule contiguously")
        if not math.isfinite(weight) or weight <= 0 or weight < previous_weight:
            raise ValueError("E007 Phase-3C timestep weights must be finite, positive, and monotonic")
        raw[start : end + 1] = weight
        expanded.append({"name": str(record["name"]), "start": start, "end": end, "unnormalized_weight": weight})
        expected_start = end + 1
        previous_weight = weight
    if expected_start != diffusion_steps or not bool(torch.isfinite(raw).all()):
        raise ValueError("E007 Phase-3C timestep bins do not cover every timestep")
    weights = raw / raw.mean()
    # Make the discrete float64 table sum exactly to the number of timesteps.
    weights[-1] += diffusion_steps - weights.sum()
    if abs(float(weights.mean()) - 1.0) > 1e-12:
        raise ValueError("E007 Phase-3C normalized timestep weights do not have mean one")
    table = [
        {"timestep": index, "weight": float(weight), "unnormalized_weight": float(raw[index])}
        for index, weight in enumerate(weights)
    ]
    return weights, expanded, _canonical_hash(table)


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != PILOT_VERSION:
        raise ValueError("E007 Phase-3C configuration version contradiction")
    if tuple(config.get("arms", ())) != ARMS:
        raise ValueError("E007 Phase-3C arm contract changed")
    if list(map(int, config.get("seeds", ()))) != [7301, 7302, 7303]:
        raise ValueError("E007 Phase-3C seeds changed")
    if list(map(int, config.get("lengths", ()))) != [16, 24, 31, 48]:
        raise ValueError("E007 Phase-3C frozen lengths changed")
    if int(config.get("optimizer_updates_per_arm_seed", 0)) != 1000:
        raise ValueError("E007 Phase-3C requires exactly 1,000 updates per arm and seed")
    if list(map(int, config.get("evaluation_updates", ()))) != [0, 100, 300, 600, 1000]:
        raise ValueError("E007 Phase-3C evaluation schedule changed")
    if int(config.get("batch_size", 0)) != 1:
        raise ValueError("E007 Phase-3C frozen pilot uses physical batch size one")
    normalized_timestep_weights(int(config["diffusion_steps"]), config["timestep_weight_bins"])
    for value in config["x0_geometry_auxiliary_coefficients"].values():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("E007 Phase-3C auxiliary coefficients must be finite and nonnegative")
    return config


def _verify_phase3b_v2(config: dict[str, Any]) -> dict[str, Any]:
    pinned = config["phase3b_v2"]
    paths = {
        "config": Path(pinned["config_path"]),
        "report": Path(pinned["report_path"]),
        "protocol": Path(pinned["protocol_path"]),
        "coordinate_model_contract": Path(pinned["coordinate_model_contract_path"]),
    }
    expected = {
        "config": pinned["config_sha256"],
        "report": pinned["report_sha256"],
        "protocol": pinned["protocol_sha256"],
        "coordinate_model_contract": pinned["coordinate_model_contract_sha256"],
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    if observed != expected:
        raise ValueError(f"E007 Phase-3B-v2 prerequisite hash contradiction: {observed}")
    report = json.loads(paths["report"].read_text())
    protocol = json.loads(paths["protocol"].read_text())
    required = {
        "status": "completed",
        "classification": "denoising_learned_but_sampling_not_learned",
        "successful_optimizer_updates": 3000,
        "seeds": [7301, 7302, 7303],
        "protected_inputs_unchanged": True,
    }
    for field, value in required.items():
        if report.get(field) != value:
            raise ValueError(f"E007 Phase-3B-v2 evidence contradiction: {field}")
    if protocol.get("status") != "completed" or protocol.get("authorizes_training") is not False:
        raise ValueError("E007 Phase-3B-v2 protocol is not completed non-authorizing evidence")
    if report.get("oracle_sampler_contract", {}).get("passed") is not True:
        raise ValueError("E007 Phase-3B-v2 oracle contract did not pass")
    if any(seed_report.get("sampling_geometry_valid") is not True for seed_report in report["seed_reports"]):
        raise ValueError("E007 Phase-3B-v2 numerical geometry did not pass")
    if config["decision_thresholds"] != report["decision_thresholds"]:
        raise ValueError("E007 Phase-3C changed a Phase-3B-v2 decision threshold")
    return {"hashes": observed, "report": report, "protocol": protocol, "paths": paths}


def _protected_hashes(config: dict[str, Any]) -> dict[str, str]:
    prerequisite = _verify_phase3b_v2(config)
    paths = prerequisite["paths"]
    phase3a = config["phase3a"]
    extra = {
        "phase3a_generator_config": Path(phase3a["generator_config_path"]),
        "e004_checkpoint": Path(phase3a["e004_checkpoint_path"]),
    }
    return {
        **{f"phase3b_v2_{name}": _sha256_file(path) for name, path in paths.items()},
        **{name: _sha256_file(path) for name, path in extra.items()},
    }


def _verify_panels(config: dict[str, Any], evidence: dict[str, Any]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    train, heldout, metadata = build_polymer_panels(config)
    # JSON publication stringifies integer dictionary keys such as lengths.
    published_metadata = json.loads(json.dumps(metadata, sort_keys=True))
    if published_metadata != evidence["report"]["polymer_panels"]:
        raise ValueError("E007 Phase-3C panel identity differs from frozen Phase-3B-v2 panels")
    return train, heldout, published_metadata


def _weight_table_payload(config: dict[str, Any]) -> dict[str, Any]:
    weights, bins, digest = normalized_timestep_weights(int(config["diffusion_steps"]), config["timestep_weight_bins"])
    return {
        "bins": bins,
        "table": [float(value) for value in weights],
        "mean": 1.0,
        "sha256": digest,
    }


def plan_objective_correction_pilot(config_path: str | Path) -> dict[str, Any]:
    """Return a read-only plan without creating models' execution state or outputs."""
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3C output already exists: {output} or {staging}")
    evidence = _verify_phase3b_v2(config)
    _, _, panels = _verify_panels(config, evidence)
    model = EquivariantPairCoordinateUNet(**config["smoke_model"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    if parameter_count != int(config["expected_model_parameter_count"]):
        raise ValueError("E007 Phase-3C model parameter count contradiction")
    weight_table = _weight_table_payload(config)
    return {
        "status": "planned_non_authorizing",
        "version": PILOT_VERSION,
        "output_dir": str(output),
        "arms": list(ARMS),
        "seeds": list(map(int, config["seeds"])),
        "lengths": list(map(int, config["lengths"])),
        "model_parameter_count": parameter_count,
        "successful_optimizer_updates_per_arm_seed": int(config["optimizer_updates_per_arm_seed"]),
        "total_planned_successful_optimizer_updates": len(ARMS)
        * len(config["seeds"])
        * int(config["optimizer_updates_per_arm_seed"]),
        "evaluation_updates": list(map(int, config["evaluation_updates"])),
        "unconditional_sampling_at_every_evaluation": True,
        "train_sample_count": sum(panels["length_counts"]["train"].values()),
        "heldout_sample_count": sum(panels["length_counts"]["heldout"].values()),
        "frozen_panel_hashes": panels,
        "timestep_weight_schedule": weight_table,
        "auxiliary_coefficients": dict(config["x0_geometry_auxiliary_coefficients"]),
        "auxiliary_scale_bounds": dict(config["auxiliary_scale_bounds"]),
        "phase3b_v2_prerequisite_hashes": evidence["hashes"],
        "decision_thresholds": dict(config["decision_thresholds"]),
        "config_sha256": _sha256_file(Path(config_path)),
        "isolated_processes": True,
        "optimizer_created": False,
        "forward_executed": False,
        "backward_executed": False,
        "sampling_executed": False,
        "real_data_loaded": False,
        "synthetic_only": True,
        **NON_AUTHORIZING,
    }


def _objective_weights(config: dict[str, Any]) -> ObjectiveCorrectionWeights:
    values = config["x0_geometry_auxiliary_coefficients"]
    return ObjectiveCorrectionWeights(
        pair_distance=float(values["pair_distance"]),
        adjacent_distance=float(values["adjacent_distance"]),
        radius_of_gyration=float(values["radius_of_gyration"]),
        steric_clash=float(values["steric_clash"]),
    )


def _losses_for_arm(
    arm: str,
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    reconstructed: torch.Tensor,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    timestep_weights: torch.Tensor,
    residue_mask: torch.Tensor,
    continuity: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    table = torch.ones_like(timestep_weights) if arm == "uniform_v_control" else timestep_weights
    auxiliaries = (
        _objective_weights(config) if arm.endswith("plus_x0_geometry") else ObjectiveCorrectionWeights(0, 0, 0, 0)
    )
    return coordinate_objective_correction_losses(
        v_prediction=prediction,
        v_target=target,
        predicted_clean_coordinates=reconstructed,
        clean_coordinates=clean,
        timesteps=timesteps,
        timestep_weights=table,
        residue_mask=residue_mask,
        chain_continuity_mask=continuity,
        auxiliary_weights=auxiliaries,
        clash_distance_normalized=float(config["steric_clash_distance_angstrom"])
        / float(config["coordinate_scale_angstrom"]),
    )


def _scalar_losses(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach()) for name, value in losses.items() if value.ndim == 0}


def verify_auxiliary_scale_bound(losses: dict[str, torch.Tensor], bounds: dict[str, Any]) -> dict[str, Any]:
    primary = float(losses["weighted_coordinate_v"].detach())
    weighted = {
        name.removeprefix("weighted_"): float(value.detach())
        for name, value in losses.items()
        if name.startswith("weighted_") and name not in {"weighted_coordinate_v", "weighted_auxiliary_total"}
    }
    ratios = {name: value / max(primary, 1e-12) for name, value in weighted.items()}
    total_ratio = float(losses["weighted_auxiliary_total"].detach()) / max(primary, 1e-12)
    tolerance = 1e-7
    passed = all(value <= float(bounds["maximum_each_to_primary_ratio"]) + tolerance for value in ratios.values()) and (
        total_ratio <= float(bounds["maximum_total_to_primary_ratio"]) + tolerance
    )
    result = {
        "weighted_v_loss": primary,
        "raw_components": {
            name.removeprefix("raw_"): float(value.detach())
            for name, value in losses.items()
            if name.startswith("raw_")
        },
        "weighted_components": weighted,
        "component_to_primary_ratios": ratios,
        "total_auxiliary_to_primary_ratio": total_ratio,
        "bounds": dict(bounds),
        "passed": passed,
    }
    if not passed:
        raise ValueError(f"E007 Phase-3C auxiliary scale bound failed: {result}")
    return result


def _sampling_initial_state_hash(config: dict[str, Any], seed_offset: int) -> str:
    hashes = []
    for index, length in enumerate(map(int, config["sampling_lengths"])):
        seed = int(config["sampling_seed"]) + seed_offset + index
        generator = torch.Generator().manual_seed(seed)
        mask = torch.ones((1, length), dtype=torch.bool)
        initial = centered_coordinate_noise(torch.empty((1, length, 3)), mask, generator=generator)
        hashes.append((length, seed, _tensor_sha256(initial)))
    return _canonical_hash(hashes)


def _matched_quantiles(values: torch.Tensor, probabilities: list[float]) -> torch.Tensor:
    """Compute quantiles without changing the input dtype or device."""
    if values.numel() == 0:
        raise ValueError("E007 Phase-3C quantiles require at least one valid value")
    if not probabilities:
        raise ValueError("E007 Phase-3C quantile probabilities are empty")
    return torch.quantile(values, values.new_tensor(probabilities))


def _sample_panel(
    model: EquivariantPairCoordinateUNet,
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    *,
    seed_offset: int,
    device: torch.device,
) -> dict[str, Any]:
    rows, fingerprints = [], []
    model.eval()
    for index, length in enumerate(map(int, config["sampling_lengths"])):
        seed = int(config["sampling_seed"]) + seed_offset + index
        sampled = diffusion.sample(model, length=length, seed=seed, device=device)
        coordinates = sampled["coordinates"][0] * float(config["coordinate_scale_angstrom"])
        distances = coordinates_to_distance_matrix(coordinates, diagnostic_float64=True)
        upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        nonneighbor = torch.triu(torch.ones_like(upper), diagonal=2)
        pair_values = distances[upper]
        adjacent = torch.diagonal(distances, offset=1)
        contact_density = {
            f"contact_density_{threshold}a": float((distances[nonneighbor] < threshold).double().mean())
            for threshold in (6.0, 8.0, 10.0)
        }
        fingerprint = _matched_quantiles(pair_values, [index / 20 for index in range(21)])
        fingerprints.append(fingerprint.cpu())
        rows.append(
            {
                "length": length,
                "seed": seed,
                **_matrix_geometry(coordinates),
                "adjacent_distance_quantiles": [
                    float(value) for value in _matched_quantiles(adjacent, [0.1, 0.5, 0.9])
                ],
                "pair_distance_quantiles": [float(value) for value in _matched_quantiles(pair_values, [0.1, 0.5, 0.9])],
                **contact_density,
            }
        )
    near_duplicate_pairs = 0
    pair_count = 0
    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            pair_count += 1
            near_duplicate_pairs += int(
                float(torch.sqrt(((fingerprints[left] - fingerprints[right]) ** 2).mean())) < 0.05
            )
    return {
        "rows": rows,
        "initial_state_sha256": _sampling_initial_state_hash(config, seed_offset),
        "stochastic_draws_sha256": _canonical_hash([]),
        "summary": {
            "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in rows])),
            "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in rows])),
            "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in rows])),
            "contact_density_8a_mean": float(np.mean([row["contact_density_8.0a"] for row in rows])),
            "near_duplicate_fraction": near_duplicate_pairs / max(pair_count, 1),
            "unique_shape_fraction": (len(rows) - near_duplicate_pairs) / max(len(rows), 1),
        },
    }


def _evaluation_with_losses(
    model: EquivariantPairCoordinateUNet,
    panel: list[Any],
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    arm: str,
    *,
    corruption_seed: int,
    device: torch.device,
    forced_timestep: int | None = None,
) -> dict[str, Any]:
    base = _evaluation_metrics(
        model,
        panel,
        diffusion,
        scale=float(config["coordinate_scale_angstrom"]),
        corruption_seed=corruption_seed,
        device=device,
        forced_timestep=forced_timestep,
    )
    timestep_weights, _, _ = normalized_timestep_weights(int(config["diffusion_steps"]), config["timestep_weight_bins"])
    component_rows = []
    model.eval()
    with torch.no_grad():
        for index, sample in enumerate(panel):
            batch, diffused = _paired_diffusion_batch(
                sample,
                diffusion,
                scale=float(config["coordinate_scale_angstrom"]),
                draw_seed=corruption_seed + index * 997,
                rotate=False,
                device=device,
                forced_timestep=forced_timestep,
            )
            prediction = model(
                diffused.noisy_coordinates,
                diffused.timesteps,
                batch["lengths"],
                batch["residue_mask"],
                batch["continuity"],
            )["v_prediction"]
            reconstructed = diffusion.reconstruct_x0(
                diffused.noisy_coordinates, diffused.timesteps, prediction, batch["residue_mask"]
            )
            component_rows.append(
                _scalar_losses(
                    _losses_for_arm(
                        arm,
                        prediction=prediction,
                        target=diffused.coordinate_v_target,
                        reconstructed=reconstructed,
                        clean=batch["coordinates"],
                        timesteps=diffused.timesteps,
                        timestep_weights=timestep_weights,
                        residue_mask=batch["residue_mask"],
                        continuity=batch["continuity"],
                        config=config,
                    )
                )
            )
    base["loss_components"] = {
        name: float(np.mean([row[name] for row in component_rows])) for name in component_rows[0]
    }
    return base


def _gradient_norms(named: dict[str, torch.Tensor]) -> dict[str, float]:
    groups = {
        "full_model": list(named.values()),
        "coefficient_head": [value for name, value in named.items() if "coefficient_head" in name],
        "pair_grid_unet_trunk": [
            value for name, value in named.items() if "pair_trunk" in name and "coefficient_head" not in name
        ],
    }
    return {
        name: math.sqrt(sum(float(value.detach().double().square().sum()) for value in values))
        for name, values in groups.items()
    }


def _parameter_change_groups(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> dict[str, float]:
    groups = {"full_model": 0.0, "pair_grid_unet_trunk": 0.0, "coefficient_head": 0.0}
    for name, value in model.state_dict().items():
        if not value.is_floating_point():
            continue
        squared = float((value.detach().cpu() - initial[name]).double().square().sum())
        groups["full_model"] += squared
        if "coefficient_head" in name:
            groups["coefficient_head"] += squared
        elif "pair_trunk" in name:
            groups["pair_grid_unet_trunk"] += squared
    return {name: math.sqrt(value) for name, value in groups.items()}


def _calibrate_auxiliaries(
    model: EquivariantPairCoordinateUNet,
    sample: Any,
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    batch, diffused = _paired_diffusion_batch(
        sample,
        diffusion,
        scale=float(config["coordinate_scale_angstrom"]),
        draw_seed=seed * 1_000_000 + int(config["calibration_draw_offset"]),
        rotate=True,
        device=device,
    )
    with torch.no_grad():
        prediction = model(
            diffused.noisy_coordinates,
            diffused.timesteps,
            batch["lengths"],
            batch["residue_mask"],
            batch["continuity"],
        )["v_prediction"]
        reconstructed = diffusion.reconstruct_x0(
            diffused.noisy_coordinates, diffused.timesteps, prediction, batch["residue_mask"]
        )
        table, _, _ = normalized_timestep_weights(int(config["diffusion_steps"]), config["timestep_weight_bins"])
        losses = _losses_for_arm(
            "high_noise_balanced_v_plus_x0_geometry",
            prediction=prediction,
            target=diffused.coordinate_v_target,
            reconstructed=reconstructed,
            clean=batch["coordinates"],
            timesteps=diffused.timesteps,
            timestep_weights=table,
            residue_mask=batch["residue_mask"],
            continuity=batch["continuity"],
            config=config,
        )
    result = verify_auxiliary_scale_bound(losses, config["auxiliary_scale_bounds"])
    result.update({"sample_id": sample.sample_id, "timestep": int(diffused.timesteps.item()), "optimized": False})
    return result


def _run_arm_seed(config_path: str, arm: str, seed: int, arm_directory: str) -> None:
    config = _load_config(config_path)
    evidence = _verify_phase3b_v2(config)
    train, heldout, panel_metadata = _verify_panels(config, evidence)
    directory = Path(arm_directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = EquivariantPairCoordinateUNet(**config["smoke_model"]).to(device)
    initialization_hash = _state_hash(model)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
    timestep_weights, _, table_hash = normalized_timestep_weights(
        int(config["diffusion_steps"]), config["timestep_weight_bins"]
    )
    calibration = (
        _calibrate_auxiliaries(model, train[0], diffusion, config, seed, device)
        if arm.endswith("plus_x0_geometry")
        else None
    )
    evaluations: dict[str, Any] = {}
    evaluation_updates = set(map(int, config["evaluation_updates"]))
    target_distribution = _target_distribution(heldout, set(map(int, config["sampling_lengths"])))
    training_identity = {"sample_order": [], "timesteps": [], "coordinate_noise": [], "targets": []}
    latest_gradients = {"full_model": 0.0, "pair_grid_unet_trunk": 0.0, "coefficient_head": 0.0}
    observed_nonzero = {name: False for name in latest_gradients}
    successful = 0

    def evaluate(update: int) -> None:
        panels = {}
        for panel_name, panel, offset in (
            ("train", train, 40000),
            ("heldout", heldout, 50000),
        ):
            panels[panel_name] = _evaluation_with_losses(
                model,
                panel,
                diffusion,
                config,
                arm,
                corruption_seed=seed + offset,
                device=device,
            )
        bins = {}
        for record in config["timestep_evaluation_bins"]:
            timestep = int(round(float(record["fraction"]) * (diffusion.timesteps - 1)))
            bins[str(record["name"])] = _evaluation_with_losses(
                model,
                heldout,
                diffusion,
                config,
                arm,
                corruption_seed=seed + 60000,
                device=device,
                forced_timestep=timestep,
            )
            bins[str(record["name"])]["timestep"] = timestep
        sampling = _sample_panel(model, diffusion, config, seed_offset=seed * 10, device=device)
        sampling["joint_polymer_quality"] = _joint_polymer_quality(
            sampling["rows"], target_distribution, config["decision_thresholds"]
        )
        sampling["sampling_geometry_valid"] = _sampling_geometry_valid(
            sampling["rows"], float(config["decision_thresholds"]["euclidean_scaled_tolerance"])
        )
        evaluations[str(update)] = {
            **panels,
            "heldout_timestep_bins": bins,
            "unconditional_sampling": sampling,
        }

    evaluate(0)
    trajectory_path = directory / "trajectory.jsonl.gz"
    with _deterministic_gzip_text(trajectory_path) as metrics:
        for attempted in range(1, int(config["optimizer_updates_per_arm_seed"]) + 1):
            model.train()
            sample = train[(attempted - 1) % len(train)]
            draw_seed = seed * 1_000_000 + attempted
            batch, diffused = _paired_diffusion_batch(
                sample,
                diffusion,
                scale=float(config["coordinate_scale_angstrom"]),
                draw_seed=draw_seed,
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
            reconstructed = diffusion.reconstruct_x0(
                diffused.noisy_coordinates, diffused.timesteps, prediction, batch["residue_mask"]
            )
            losses = _losses_for_arm(
                arm,
                prediction=prediction,
                target=diffused.coordinate_v_target,
                reconstructed=reconstructed,
                clean=batch["coordinates"],
                timesteps=diffused.timesteps,
                timestep_weights=timestep_weights,
                residue_mask=batch["residue_mask"],
                continuity=batch["continuity"],
                config=config,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            named = require_finite_training_state(losses["total"], model, seed=seed, update=attempted)
            latest_gradients = _gradient_norms(named)
            for name, value in latest_gradients.items():
                observed_nonzero[name] |= value > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            successful += 1
            if attempted % int(config["metrics_frequency"]) == 0 or attempted in evaluation_updates:
                metrics.write(
                    json.dumps(
                        {
                            "arm": arm,
                            "seed": seed,
                            "successful_update": successful,
                            "sample_id": sample.sample_id,
                            "timestep": int(diffused.timesteps.item()),
                            "losses": _scalar_losses(losses),
                            "gradient_norms": latest_gradients,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics.flush()
            if attempted in evaluation_updates:
                evaluate(attempted)
    final = evaluations[str(successful)]
    initial = evaluations["0"]
    final_sampling = final["unconditional_sampling"]
    final_contract = _trained_contract_checks(
        model, tolerance=float(config["decision_thresholds"]["equivariance_atol"]), device=device
    )
    identity_hashes = {name: _canonical_hash(values) for name, values in training_identity.items()}
    evaluation_corruptions = []
    for update in map(str, config["evaluation_updates"]):
        for panel_name in ("train", "heldout"):
            record = evaluations[update][panel_name]
            evaluation_corruptions.append(
                (
                    update,
                    panel_name,
                    record["corruption_identity_sha256"],
                    record["noisy_coordinate_sha256"],
                    record["coordinate_v_target_sha256"],
                )
            )
    identity_hashes.update(
        {
            "initialization": initialization_hash,
            "evaluation_panel": panel_metadata["sample_id_sha256"]["heldout"],
            "evaluation_corruptions": _canonical_hash(evaluation_corruptions),
            "sampling_initial_states": final_sampling["initial_state_sha256"],
            "sampling_stochastic_draws": final_sampling["stochastic_draws_sha256"],
        }
    )
    initial_heldout = initial["heldout"]["means"]
    final_heldout = final["heldout"]["means"]
    evaluation_details = _externalize_evaluation_rows(directory, evaluations)

    def relative(before: float, after: float) -> float:
        return (before - after) / max(abs(before), 1e-12)

    result = {
        "arm": arm,
        "seed": seed,
        "successful_updates": successful,
        "finite_losses_and_gradients": True,
        "gradient_norms": latest_gradients,
        "nonzero_gradient_observed": observed_nonzero,
        "parameter_l2_change": _parameter_change(model, initial_state),
        "parameter_change_by_group": _parameter_change_groups(model, initial_state),
        "initialization_sha256": initialization_hash,
        "pairing_hashes": identity_hashes,
        "timestep_weight_table_sha256": table_hash,
        "auxiliary_scale_calibration": calibration,
        "evaluations": evaluations,
        "evaluation_details": evaluation_details,
        "relative_improvements": {
            "heldout_coordinate_v_mse": relative(
                initial_heldout["coordinate_v_mse"], final_heldout["coordinate_v_mse"]
            ),
            "heldout_pair_distance_rmse": relative(
                initial_heldout["pair_distance_rmse_angstrom"], final_heldout["pair_distance_rmse_angstrom"]
            ),
        },
        "sampling_geometry_valid": final_sampling["sampling_geometry_valid"],
        "joint_polymer_quality": final_sampling["joint_polymer_quality"],
        "trained_contract_checks": final_contract,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        **NON_AUTHORIZING,
    }
    checkpoint = directory / "final_synthetic_checkpoint.pt"
    torch.save(
        {
            "version": PILOT_VERSION,
            "arm": arm,
            "seed": seed,
            "model": model.state_dict(),
            "successful_optimizer_updates": successful,
            "synthetic_only": True,
            **NON_AUTHORIZING,
        },
        checkpoint,
    )
    result["checkpoint"] = {"path": checkpoint.name, "sha256": _sha256_file(checkpoint)}
    result["trajectory_sha256"] = _sha256_file(trajectory_path)
    _atomic_json(directory / "result.json", result)


def _arm_worker(config_path: str, arm: str, seed: int, directory: str) -> None:
    try:
        _run_arm_seed(config_path, arm, seed, directory)
    except BaseException as error:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            path / "failure.json",
            {"type": type(error).__name__, "message": str(error), "arm": arm, "seed": seed, **NON_AUTHORIZING},
        )
        raise


def _run_isolated(config_path: Path, arm: str, seed: int, directory: Path) -> dict[str, Any]:
    context = mp.get_context("spawn")
    process = context.Process(target=_arm_worker, args=(str(config_path), arm, seed, str(directory)))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"E007 Phase-3C isolated arm failed: arm={arm}, seed={seed}, exit={process.exitcode}")
    return json.loads((directory / "result.json").read_text())


def verify_pairing(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Require exact per-seed pairing across all three independent arms."""
    fields = (
        "initialization",
        "sample_order",
        "timesteps",
        "coordinate_noise",
        "targets",
        "evaluation_panel",
        "evaluation_corruptions",
        "sampling_initial_states",
        "sampling_stochastic_draws",
    )
    evidence = {}
    for seed in sorted({int(result["seed"]) for result in results}):
        rows = [result for result in results if int(result["seed"]) == seed]
        if {result["arm"] for result in rows} != set(ARMS):
            raise ValueError(f"E007 Phase-3C missing paired arm for seed {seed}")
        seed_evidence = {}
        for field in fields:
            values = {result["pairing_hashes"][field] for result in rows}
            if len(values) != 1:
                raise ValueError(f"E007 Phase-3C pairing contradiction: seed={seed}, field={field}")
            seed_evidence[field] = values.pop()
        evidence[str(seed)] = seed_evidence
    return {"passed": True, "by_seed": evidence}


def _arm_passes_v2_thresholds(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> bool:
    for row in rows:
        final = row["evaluations"]["1000"]
        initial = row["evaluations"]["0"]
        train_before = initial["train"]["means"]["coordinate_v_mse"]
        train_after = final["train"]["means"]["coordinate_v_mse"]
        train_improvement = (train_before - train_after) / max(abs(train_before), 1e-12)
        if not (
            row["successful_updates"] == 1000
            and row["finite_losses_and_gradients"]
            and all(row["nonzero_gradient_observed"].values())
            and row["parameter_l2_change"] > 0
            and all(row["trained_contract_checks"].values())
            and row["sampling_geometry_valid"]
            and row["joint_polymer_quality"]["passed"]
            and train_improvement >= float(thresholds["minimum_train_v_mse_relative_improvement"])
            and row["relative_improvements"]["heldout_coordinate_v_mse"]
            >= float(thresholds["minimum_heldout_v_mse_relative_improvement_per_seed"])
            and row["relative_improvements"]["heldout_pair_distance_rmse"]
            >= float(thresholds["minimum_heldout_pair_rmse_relative_improvement_per_seed"])
        ):
            return False
    return True


def _regression_check(
    control: list[dict[str, Any]], candidate: list[dict[str, Any]], tolerances: dict[str, Any]
) -> dict[str, Any]:
    failures = []
    metrics = (
        (
            "heldout_pair_distance_rmse",
            lambda row: row["evaluations"]["1000"]["heldout"]["means"]["pair_distance_rmse_angstrom"],
        ),
        (
            "heldout_adjacent_distance_rmse",
            lambda row: row["evaluations"]["1000"]["heldout"]["means"]["adjacent_distance_rmse_angstrom"],
        ),
        (
            "sampling_clash_fraction",
            lambda row: row["evaluations"]["1000"]["unconditional_sampling"]["summary"]["clash_fraction_mean"],
        ),
        (
            "sampling_near_duplicate_fraction",
            lambda row: row["evaluations"]["1000"]["unconditional_sampling"]["summary"]["near_duplicate_fraction"],
        ),
    )
    details = {}
    for seed in sorted(row["seed"] for row in control):
        left = next(row for row in control if row["seed"] == seed)
        right = next(row for row in candidate if row["seed"] == seed)
        for name, getter in metrics:
            baseline, observed = getter(left), getter(right)
            relative_regression = (observed - baseline) / max(abs(baseline), 1e-12)
            details[f"{seed}:{name}"] = relative_regression
            if relative_regression > float(tolerances["maximum_relative_regression"]):
                failures.append(f"{seed}:{name}")
        for contract in ("proper_rotation_equivariance", "padding_invariance", "padding_exact_zero"):
            if left["trained_contract_checks"][contract] and not right["trained_contract_checks"][contract]:
                failures.append(f"{seed}:{contract}")
    return {"passed": not failures, "failures": failures, "relative_regressions": details}


def _pareto_vector(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Build the predeclared update-1000 lower-is-better scientific vector."""
    values: dict[str, list[float]] = {
        "heldout_denoising": [],
        "high_very_high_x0_geometry": [],
        "unconditional_radius_error": [],
        "unconditional_adjacent_error": [],
        "unconditional_clash_excess": [],
        "joint_polymer_error": [],
        "near_duplicate_fraction": [],
    }
    for row in rows:
        final = row["evaluations"]["1000"]
        bins = final["heldout_timestep_bins"]
        values["heldout_denoising"].append(final["heldout"]["means"]["coordinate_v_mse"])
        values["high_very_high_x0_geometry"].append(
            float(
                np.mean(
                    [
                        bins[name]["means"]["aligned_x0_coordinate_mse_angstrom2"]
                        for name in ("high_noise", "very_high_noise")
                    ]
                )
            )
        )
        quality = row["joint_polymer_quality"]
        values["unconditional_radius_error"].append(quality["target_normalized_radius_error"])
        values["unconditional_adjacent_error"].append(quality["target_normalized_adjacent_error"])
        values["unconditional_clash_excess"].append(quality["clash_fraction_excess"])
        values["joint_polymer_error"].append(quality["joint_error"])
        values["near_duplicate_fraction"].append(final["unconditional_sampling"]["summary"]["near_duplicate_fraction"])
    return {name: float(np.mean(metric_values)) for name, metric_values in values.items()}


def classify_objective_correction(
    results: list[dict[str, Any]], thresholds: dict[str, Any], tolerances: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Apply the predeclared all-seed gate and conservative Pareto policy."""
    grouped = {arm: [row for row in results if row["arm"] == arm] for arm in ARMS}
    if any(len(rows) != 3 for rows in grouped.values()) or any(
        not row.get("finite_losses_and_gradients", False) for row in results
    ):
        return "invalid_execution", {}
    passes = {arm: _arm_passes_v2_thresholds(rows, thresholds) for arm, rows in grouped.items()}
    regressions = {arm: _regression_check(grouped["uniform_v_control"], grouped[arm], tolerances) for arm in ARMS[1:]}
    eligible = {arm: passes[arm] and regressions[arm]["passed"] for arm in ARMS[1:]}
    pareto_vectors = {arm: _pareto_vector(grouped[arm]) for arm in ARMS[1:]}
    evidence = {
        "passes_unchanged_v2_thresholds": passes,
        "control_regression_checks": regressions,
        "eligible": eligible,
        "pareto_metrics_lower_is_better": pareto_vectors,
    }
    if all(eligible.values()):
        weighted = pareto_vectors["high_noise_balanced_v"]
        auxiliary = pareto_vectors["high_noise_balanced_v_plus_x0_geometry"]
        weighted_dominates = all(weighted[name] <= auxiliary[name] for name in weighted) and any(
            weighted[name] < auxiliary[name] for name in weighted
        )
        auxiliary_dominates = all(auxiliary[name] <= weighted[name] for name in weighted) and any(
            auxiliary[name] < weighted[name] for name in weighted
        )
        evidence["pareto_dominance"] = {
            "high_noise_balanced_v_dominates": weighted_dominates,
            "high_noise_balanced_v_plus_x0_geometry_dominates": auxiliary_dominates,
        }
        if weighted_dominates:
            return "high_noise_weighting_sufficient", evidence
        if auxiliary_dominates:
            return "x0_geometry_auxiliaries_required", evidence
        return "objective_correction_verified", evidence
    if eligible["high_noise_balanced_v"]:
        return "high_noise_weighting_sufficient", evidence
    if eligible["high_noise_balanced_v_plus_x0_geometry"]:
        return "x0_geometry_auxiliaries_required", evidence
    if any(regressions[arm]["failures"] for arm in ARMS[1:]):
        return "objective_correction_regresses_quality", evidence
    partial = any(any(row["joint_polymer_quality"]["passed"] for row in grouped[arm]) for arm in ARMS[1:])
    return ("correction_improves_but_not_all_seeds" if partial else "no_objective_correction_benefit"), evidence


def _bootstrap_comparisons(results: list[dict[str, Any]], seed: int, replicates: int) -> dict[str, Any]:
    control = {row["seed"]: row for row in results if row["arm"] == "uniform_v_control"}
    rng = np.random.default_rng(seed)
    output = {}
    extractors = {
        "heldout_coordinate_v_mse": lambda row: row["evaluations"]["1000"]["heldout"]["means"]["coordinate_v_mse"],
        "high_noise_x0_mse": lambda row: row["evaluations"]["1000"]["heldout_timestep_bins"]["very_high_noise"][
            "means"
        ]["aligned_x0_coordinate_mse_angstrom2"],
        "unconditional_radius_error": lambda row: row["joint_polymer_quality"]["target_normalized_radius_error"],
        "unconditional_joint_error": lambda row: row["joint_polymer_quality"]["joint_error"],
    }
    for arm in ARMS[1:]:
        candidate = {row["seed"]: row for row in results if row["arm"] == arm}
        output[arm] = {}
        for name, getter in extractors.items():
            differences = np.array([getter(control[value]) - getter(candidate[value]) for value in sorted(control)])
            bootstrap = np.array(
                [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(replicates)]
            )
            output[arm][name] = {
                "positive_means_candidate_is_lower_better": True,
                "mean_improvement": float(differences.mean()),
                "ci_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
                "per_seed": {
                    str(value): float(getter(control[value]) - getter(candidate[value])) for value in sorted(control)
                },
            }
    return output


def run_objective_correction_pilot(config_path: str | Path) -> dict[str, Any]:
    """Run nine isolated arm/seed jobs and atomically publish the non-authorizing pilot."""
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    plan = plan_objective_correction_pilot(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = datetime.now(UTC).isoformat()
    before = _protected_hashes(config)
    _atomic_json(
        staging / "heartbeat.json",
        {"status": "running", "stage": "isolated_arms", "started_utc": started, **NON_AUTHORIZING},
    )
    try:
        results = []
        for arm in ARMS:
            for seed in map(int, config["seeds"]):
                results.append(_run_isolated(config_path, arm, seed, staging / "arms" / arm / f"seed_{seed}"))
        pairing = verify_pairing(results)
        classification, selection = classify_objective_correction(
            results, config["decision_thresholds"], config["control_regression_tolerances"]
        )
        after = _protected_hashes(config)
        if before != after:
            raise ValueError("E007 protected inputs changed during Phase-3C execution")
        comparisons = _bootstrap_comparisons(
            results, int(config["bootstrap_seed"]), int(config["bootstrap_replicates"])
        )
        report = {
            **plan,
            "status": "completed",
            "classification": classification,
            "pairing_evidence": pairing,
            "arm_seed_results": results,
            "selection_evidence": selection,
            "paired_control_comparisons": comparisons,
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
        scalar_hashes = {
            str(path.relative_to(staging)): _sha256_file(path)
            for path in sorted(staging.glob("arms/*/seed_*/*.jsonl.gz"))
        }
        protocol = {
            "version": PILOT_VERSION,
            "status": "completed",
            "classification": classification,
            "report_sha256": _sha256_file(staging / "report.json"),
            "scalar_trajectory_hashes": scalar_hashes,
            "timestep_weight_table_sha256": plan["timestep_weight_schedule"]["sha256"],
            "pairing_evidence": pairing,
            "synthetic_only": True,
            "non_production": True,
            "protected_inputs_unchanged": True,
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


def remove_failed_test_staging(path: Path) -> None:
    """Test-only cleanup; production execution preserves failed staging."""
    shutil.rmtree(path)
