"""Bounded real-data coordinate-learning pilot for E007 Phase 3F."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import random
import resource
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.data.rich_geometry import authorize_rich_geometry_dataset
from protein_distance_diffusion.models.coordinate_equivariance import (
    coordinate_backend_policy,
    coordinate_model_execution_context,
    equivariance_criterion,
    equivariance_metrics,
)
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file, verify_metadata
from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import (
    _parameter_sha256,
    prepare_coordinate_batch,
)

VERSION = "e007_coordinate_real_pilot_v1"
EXPECTED_PARAMETER_COUNT = 7_586_505
EXPECTED_SCALE = 12.22820347644835
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "mandatory_scientific_review": True,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase-3F configuration version contradiction")
    if int(payload.get("successful_optimizer_updates", 0)) != 1000:
        raise ValueError("E007 Phase-3F must stop after 1,000 successful updates")
    if payload.get("evaluation_updates") != [0, 100, 250, 500, 750, 1000]:
        raise ValueError("E007 Phase-3F evaluation schedule changed")
    if payload.get("sampling_updates") != [0, 250, 500, 750, 1000]:
        raise ValueError("E007 Phase-3F sampling schedule changed")
    if payload.get("checkpoint_updates") != [250, 500, 750, 1000]:
        raise ValueError("E007 Phase-3F checkpoint schedule changed")
    objective = payload.get("objective", {})
    if objective != {
        "name": "uniform_valid_coordinate_v_mse",
        "timestep_sampling": "uniform_discrete",
        "timestep_weighting": "none",
        "auxiliary_losses": [],
    }:
        raise ValueError("E007 Phase-3F objective contract changed")
    if float(payload.get("coordinate_scale_angstrom", 0)) != EXPECTED_SCALE:
        raise ValueError("E007 Phase-3F coordinate normalization changed")
    if int(payload.get("expected_parameter_count", 0)) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3F parameter-count contract changed")
    optimizer = payload.get("optimizer", {})
    if (
        optimizer.get("family") != "AdamW"
        or float(optimizer.get("learning_rate", 0)) != 3e-4
        or float(optimizer.get("weight_decay", -1)) != 0.01
        or optimizer.get("scheduler") != "constant"
    ):
        raise ValueError("E007 Phase-3F validated optimizer contract changed")
    coordinate_backend_policy(payload.get("numerics"))
    _validate_regimes(payload["batch_regimes"])
    return payload


def _verify_file(path: str | Path, expected: str, label: str) -> str:
    observed = sha256_file(Path(path))
    if observed != expected:
        raise ValueError(f"E007 Phase-3F prerequisite hash contradiction: {label}")
    return observed


def verify_pilot_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    """Verify pinned metadata and reports without scanning coordinate payloads."""
    files: dict[str, tuple[str, str]] = {}
    for section_name in ("phase3e_c1", "phase3e_b"):
        section = config[section_name]
        for name in ("report", "protocol"):
            files[f"{section_name}_{name}"] = (section[f"{name}_path"], section[f"{name}_sha256"])
    section = config["phase3e_c1"]
    files["phase3e_c1_artifact_inventory"] = (
        section["artifact_inventory_path"],
        section["artifact_inventory_sha256"],
    )
    files["identity_30_clean_validation_manifest"] = (
        config["clean_validation"]["manifest_path"],
        config["clean_validation"]["manifest_sha256"],
    )
    files["production_generator_config"] = (
        config["production_generator"]["config_path"],
        config["production_generator"]["config_sha256"],
    )
    files["phase3e_b_config"] = (
        config["phase3e_b"]["config_path"],
        config["phase3e_b"]["config_sha256"],
    )
    normalization = config["coordinate_normalization"]
    for name in ("artifact", "report", "protocol"):
        files[f"coordinate_normalization_{name}"] = (
            normalization[f"{name}_path"],
            normalization[f"{name}_sha256"],
        )
    hashes = {name: _verify_file(path, expected, name) for name, (path, expected) in files.items()}
    report = json.loads(Path(config["phase3e_c1"]["report_path"]).read_text())
    protocol = json.loads(Path(config["phase3e_c1"]["protocol_path"]).read_text())
    if report.get("status") != "completed_non_authorizing" or protocol.get("status") != "completed_non_authorizing":
        raise ValueError("E007 Phase-3E-C.1 corrected publication is incomplete")
    counts = report.get("accepted_population_counts", {})
    if counts != {"combined": 253293, "train": 230132, "validation": 23161}:
        raise ValueError("E007 Phase-3E-C.1 accepted population contradiction")
    identity = report.get("homology_thresholds", {}).get("identity_30", {})
    if identity.get("clean_validation_count") != 21583:
        raise ValueError("E007 identity-30 clean-validation count contradiction")
    if identity.get("clean_validation_sha256") != config["clean_validation"]["manifest_sha256"]:
        raise ValueError("E007 identity-30 clean-validation identity contradiction")
    if any(bool(report.get(key)) for key in NON_AUTHORIZING if key.startswith("authorizes_")):
        raise ValueError("E007 Phase-3E-C.1 unexpectedly authorizes training")
    smoke = json.loads(Path(config["phase3e_b"]["report_path"]).read_text())
    if (
        smoke.get("status") != "completed_non_authorizing"
        or smoke.get("model_evidence", {}).get("parameter_count") != EXPECTED_PARAMETER_COUNT
    ):
        raise ValueError("E007 Phase-3E-B production smoke contradiction")
    generator = yaml.safe_load(Path(config["production_generator"]["config_path"]).read_text())
    if generator["model"] != config["model"] or int(generator["diffusion_steps"]) != 500:
        raise ValueError("E007 Phase-3F production architecture or diffusion schedule changed")
    metadata = verify_metadata(config["dataset"])
    return {"hashes": {**hashes, **{f"dataset_{k}": v for k, v in metadata["hashes"].items()}}, "metadata": metadata}


def _validate_regimes(regimes: list[dict[str, Any]]) -> None:
    observed = [(int(x["maximum_length"]), int(x["physical_batch_size"])) for x in regimes]
    if observed != [(64, 4), (128, 2), (256, 1), (384, 1), (500, 1)]:
        raise ValueError("E007 Phase-3F physical-batch regimes changed")
    if any(int(x.get("accumulation_steps", 0)) != 1 for x in regimes):
        raise ValueError("E007 Phase-3F does not require gradient accumulation")


def _memory_preflight(config: dict[str, Any]) -> dict[str, Any]:
    memory = config["memory"]
    checks = {
        "allocated": float(memory["smoke_peak_cuda_allocated_mib"]) <= float(memory["maximum_cuda_allocated_mib"]),
        "reserved": float(memory["smoke_peak_cuda_reserved_mib"]) <= float(memory["maximum_cuda_reserved_mib"]),
    }
    if not all(checks.values()):
        raise ValueError("E007 Phase-3F estimated memory exceeds the validated GPU envelope")
    return {"passed": True, "checks": checks, **memory}


def plan_real_coordinate_pilot(config_path: str | Path) -> dict[str, Any]:
    """Plan without scanning shards, constructing the model, optimizer, or CUDA state."""
    config = _load_config(config_path)
    prerequisites = verify_pilot_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3F output already exists: {output} or {staging}")
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(Path(config_path)),
        "output_dir": str(output),
        "successful_optimizer_updates": 1000,
        "evaluation_updates": config["evaluation_updates"],
        "sampling_updates": config["sampling_updates"],
        "checkpoint_updates": config["checkpoint_updates"],
        "accepted_train_count": config["dataset"]["accepted_train_count"],
        "clean_validation_count": config["clean_validation"]["accepted_validation_count"],
        "clean_validation_policy": "identity_30_no_accepted_train_cluster_member",
        "batch_regimes": config["batch_regimes"],
        "objective": config["objective"],
        "model_contract": {
            "name": "EquivariantPairCoordinateUNet",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "initialization": "deterministic_random_from_scratch",
            "sequence_inputs": False,
            "pretrained_weights": False,
        },
        "memory_preflight": _memory_preflight(config),
        "prerequisite_hashes": prerequisites["hashes"],
        "coordinate_payloads_scanned": False,
        "model_created": False,
        "optimizer_created": False,
        "cuda_tensors_allocated": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def length_stratum(length: int, strata: list[dict[str, Any]]) -> str:
    matches = [x["name"] for x in strata if int(x["minimum"]) <= length <= int(x["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 Phase-3F length has no unique stratum: {length}")
    return str(matches[0])


def update_stratum(update: int, strata: list[dict[str, Any]]) -> str:
    if update < 1:
        raise ValueError("optimizer update must be positive")
    return str(strata[(update - 1) % len(strata)]["name"])


def _regime_for_stratum(name: str, config: dict[str, Any]) -> dict[str, Any]:
    maximum = next(int(x["maximum"]) for x in config["length_strata"] if x["name"] == name)
    return next(x for x in config["batch_regimes"] if int(x["maximum_length"]) == maximum)


def planned_batch_accounting(config: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    samples: dict[str, int] = defaultdict(int)
    for update in range(1, int(config["successful_optimizer_updates"]) + 1):
        name = update_stratum(update, config["length_strata"])
        counts[name] += 1
        samples[name] += int(_regime_for_stratum(name, config)["physical_batch_size"])
    return {"optimizer_updates_by_stratum": dict(counts), "planned_samples_by_stratum": dict(samples)}


def _rank(seed: int, purpose: str, sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{purpose}|{sample_id}".encode()).digest(), "big")


def _bounded_ranked_rows(
    rows: Any,
    *,
    strata: list[dict[str, Any]],
    capacities: dict[str, int],
    seed: int,
    purpose: str,
    permitted_ids: set[str] | None = None,
    expected_eligible_count: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    seen: set[str] = set()
    eligible_count = 0
    for row in rows:
        if not row["accepted_contiguous_single_chain"]:
            continue
        sample_id = str(row["sample_id"])
        if permitted_ids is not None and sample_id not in permitted_ids:
            continue
        if sample_id in seen:
            raise ValueError(f"duplicate E007 sample ID: {sample_id}")
        seen.add(sample_id)
        eligible_count += 1
        name = length_stratum(int(row["sequence_length"]), strata)
        capacity = capacities.get(name, 0)
        if capacity <= 0:
            continue
        ranking = _rank(seed, purpose, sample_id)
        item = (-ranking, sample_id, row)
        heap = heaps[name]
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    result = {name: [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))] for name, heap in heaps.items()}
    shortages = {
        name: capacities[name] - len(result.get(name, []))
        for name in capacities
        if len(result.get(name, [])) < capacities[name]
    }
    if shortages:
        raise ValueError(f"E007 Phase-3F panel underfill: {shortages}")
    if expected_eligible_count is not None and eligible_count != expected_eligible_count:
        raise ValueError(
            f"E007 Phase-3F accepted population contradiction: {eligible_count} != {expected_eligible_count}"
        )
    return result


def _clean_validation_ids(config: dict[str, Any]) -> set[str]:
    table = pq.read_table(
        config["clean_validation"]["manifest_path"], columns=["sample_id", "split", "coordinate_accepted"]
    )
    values = table.to_pylist()
    ids = {str(x["sample_id"]) for x in values if x["split"] == "validation" and x["coordinate_accepted"] is True}
    if len(values) != int(config["clean_validation"]["accepted_validation_count"]) or len(ids) != len(values):
        raise ValueError("E007 identity-30 clean-validation membership contradiction")
    return ids


def _authorize(config: dict[str, Any]):
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["root"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
        protected_input_relocations=dataset.get("protected_input_relocations"),
    )


def _select_rows(config: dict[str, Any], authorization: Any) -> dict[str, Any]:
    accounting = planned_batch_accounting(config)
    train_capacities = {
        name: accounting["planned_samples_by_stratum"][name] + int(config["train_evaluation_samples_per_stratum"])
        for name in accounting["planned_samples_by_stratum"]
    }
    train = _bounded_ranked_rows(
        E007CoordinateDataset(authorization, split="train"),
        strata=config["length_strata"],
        capacities=train_capacities,
        seed=int(config["seed"]),
        purpose="train_schedule_and_evaluation",
        expected_eligible_count=int(config["dataset"]["accepted_train_count"]),
    )
    clean_ids = _clean_validation_ids(config)
    validation = _bounded_ranked_rows(
        E007CoordinateDataset(authorization, split="validation"),
        strata=config["length_strata"],
        capacities={x["name"]: int(config["validation_samples_per_stratum"]) for x in config["length_strata"]},
        seed=int(config["seed"]),
        purpose="identity_30_clean_validation_panel",
        permitted_ids=clean_ids,
        expected_eligible_count=int(config["clean_validation"]["accepted_validation_count"]),
    )
    train_ids = {row["sample_id"] for values in train.values() for row in values}
    validation_ids = {row["sample_id"] for values in validation.values() for row in values}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(f"E007 Phase-3F train/validation panel overlap: {sorted(overlap)[:20]}")
    return {"train": train, "validation": validation, "accounting": accounting, "clean_ids": clean_ids}


def _scheduler_factor(update: int, config: dict[str, Any]) -> float:
    del update
    if config["optimizer"]["scheduler"] != "constant":
        raise ValueError("E007 Phase-3F scheduler contract changed")
    return 1.0


def uniform_coordinate_v_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask[..., None].expand_as(prediction)
    if not bool(valid.any()):
        raise ValueError("E007 coordinate-v loss has no valid coordinates")
    return (prediction[valid] - target[valid]).square().mean()


def make_uniform_training_corruption(
    prepared: dict[str, Any],
    diffusion: CoordinateVPDiffusion,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    clean = prepared["coordinates"].to(device)
    mask = prepared["residue_mask"].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    timesteps = torch.randint(
        diffusion.timesteps,
        (clean.shape[0],),
        generator=generator,
        device=device,
    )
    diffused = diffusion.make_training_batch(clean, mask, timesteps=timesteps, generator=generator)
    return {"batch": diffused, "timesteps": timesteps, "mask": mask}


def _gradient_evidence(model: torch.nn.Module) -> dict[str, Any]:
    missing = []
    nonfinite = []
    group_squared = {"pair_grid_unet_trunk": 0.0, "coefficient_head": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        group = "coefficient_head" if "pair_trunk.coefficient_head" in name else "pair_grid_unet_trunk"
        group_squared[group] += float(parameter.grad.detach().double().square().sum().cpu())
    norms = {name: math.sqrt(value) for name, value in group_squared.items()}
    if missing or nonfinite or any(value <= 0 for value in norms.values()):
        raise FloatingPointError(
            f"E007 Phase-3F gradient coverage failed: missing={missing[:20]}, nonfinite={nonfinite[:20]}, norms={norms}"
        )
    return {"group_norms": norms, "global_norm": math.sqrt(sum(x * x for x in norms.values()))}


def _physical_metrics(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    predicted = predicted[mask].double() * EXPECTED_SCALE
    target = target[mask].double() * EXPECTED_SCALE
    pred_dist = torch.cdist(predicted, predicted)
    true_dist = torch.cdist(target, target)
    length = predicted.shape[0]
    off_diagonal = ~torch.eye(length, dtype=torch.bool, device=predicted.device)
    adjacent = torch.arange(length - 1, device=predicted.device)
    pred_adjacent = pred_dist[adjacent, adjacent + 1] if length > 1 else pred_dist.new_empty(0)
    true_adjacent = true_dist[adjacent, adjacent + 1] if length > 1 else true_dist.new_empty(0)
    non_neighbor = torch.ones_like(off_diagonal)
    non_neighbor.fill_diagonal_(False)
    if length > 1:
        non_neighbor[adjacent, adjacent + 1] = False
        non_neighbor[adjacent + 1, adjacent] = False
    pred_rg = predicted.square().sum(dim=-1).mean().sqrt()
    true_rg = target.square().sum(dim=-1).mean().sqrt()
    return {
        "x0_coordinate_rmse_angstrom": float((predicted - target).square().mean().sqrt().cpu()),
        "pair_distance_rmse_angstrom": float(
            (pred_dist[off_diagonal] - true_dist[off_diagonal]).square().mean().sqrt().cpu()
        ),
        "adjacent_distance_error_angstrom": float((pred_adjacent - true_adjacent).abs().mean().cpu())
        if length > 1
        else 0.0,
        "radius_of_gyration_error_angstrom": float((pred_rg - true_rg).abs().cpu()),
        "radius_of_gyration_relative_error": float(((pred_rg - true_rg).abs() / true_rg.clamp_min(1e-8)).cpu()),
        "clash_fraction": float((pred_dist[non_neighbor] < 3.0).double().mean().cpu())
        if bool(non_neighbor.any())
        else 0.0,
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


@torch.no_grad()
def _equivariance_check(
    model: torch.nn.Module,
    row: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prepared = prepare_coordinate_batch([row], EXPECTED_SCALE, int(config["expected_downsample_factor"]))
    clean = prepared["coordinates"].to(device)
    mask = prepared["residue_mask"].to(device)
    continuity = prepared["chain_continuity_mask"].to(device)
    lengths = prepared["lengths"].to(device)
    timestep = torch.tensor([251], device=device)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 8_000_000)
    corruption = CoordinateVPDiffusion(int(config["diffusion_steps"])).make_training_batch(
        clean,
        mask,
        timesteps=timestep,
        generator=generator,
    )
    rotation_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 8_000_001)
    matrix = torch.randn((3, 3), generator=rotation_generator, device=device)
    rotation, triangular = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(triangular)).masked_fill(torch.diag(triangular) == 0, 1)
    rotation = rotation * signs
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    reference = model(corruption.noisy_coordinates, timestep, lengths, mask, continuity)
    transformed_coordinates = corruption.noisy_coordinates @ rotation
    transformed = model(transformed_coordinates, timestep, lengths, mask, continuity)
    metrics = equivariance_metrics(
        reference=reference,
        transformed=transformed,
        transformation=rotation,
        reference_coordinates=corruption.noisy_coordinates,
        transformed_coordinates=transformed_coordinates,
        residue_mask=mask,
    )
    criterion = equivariance_criterion(
        metrics,
        absolute_tolerance=5e-5,
        relative_l2_tolerance=5e-5,
        coefficient_tolerance=5e-5,
    )
    if not criterion["passed"]:
        raise ValueError(f"E007 Phase-3F strict O(3) equivariance failed: {metrics}")
    return {"metrics": metrics, "criterion": criterion}


@torch.no_grad()
def _evaluate_panel(
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    rows: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    device: torch.device,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    model.eval()
    all_records: list[dict[str, Any]] = []
    for stratum_index, specification in enumerate(config["length_strata"]):
        name = specification["name"]
        for row_index, row in enumerate(rows[name]):
            prepared = prepare_coordinate_batch([row], EXPECTED_SCALE, int(config["expected_downsample_factor"]))
            for bin_index, bin_spec in enumerate(config["timestep_evaluation_bins"]):
                timestep = int(bin_spec["timestep"])
                generator = torch.Generator(device=device).manual_seed(
                    int(config["seed"]) + seed_offset + stratum_index * 100000 + row_index * 100 + bin_index
                )
                coordinates = prepared["coordinates"].to(device)
                mask = prepared["residue_mask"].to(device)
                times = torch.tensor([timestep], device=device)
                diffused = diffusion.make_training_batch(coordinates, mask, timesteps=times, generator=generator)
                output = model(
                    diffused.noisy_coordinates,
                    times,
                    prepared["lengths"].to(device),
                    mask,
                    prepared["chain_continuity_mask"].to(device),
                )["v_prediction"]
                loss = uniform_coordinate_v_mse(output, diffused.coordinate_v_target, mask)
                reconstructed = diffusion.reconstruct_x0(diffused.noisy_coordinates, times, output, mask)
                physical = _physical_metrics(reconstructed[0], coordinates[0], mask[0])
                all_records.append(
                    {
                        "sample_id": row["sample_id"],
                        "length_stratum": name,
                        "timestep_bin": bin_spec["name"],
                        "coordinate_v_mse": float(loss.cpu()),
                        **physical,
                    }
                )
    metric_keys = [key for key in all_records[0] if key not in {"sample_id", "length_stratum", "timestep_bin"}]

    def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
        return {key: float(np.mean([x[key] for x in items])) for key in metric_keys}

    return {
        "global": aggregate(all_records),
        "by_length_stratum": {
            x["name"]: aggregate([r for r in all_records if r["length_stratum"] == x["name"]])
            for x in config["length_strata"]
        },
        "by_timestep_bin": {
            x["name"]: aggregate([r for r in all_records if r["timestep_bin"] == x["name"]])
            for x in config["timestep_evaluation_bins"]
        },
        "record_count": len(all_records),
        "equivariance": _equivariance_check(
            model,
            rows[config["length_strata"][0]["name"]][0],
            config,
            device,
        ),
    }


def _sampling_metrics(coordinates: torch.Tensor) -> dict[str, float | bool]:
    values = coordinates[0].double() * EXPECTED_SCALE
    distances = coordinates_to_distance_matrix(values, diagnostic_float64=True)
    length = values.shape[0]
    adjacent = distances.diagonal(offset=1)
    indices = torch.arange(length, device=values.device)
    non_neighbor = (indices[:, None] - indices[None, :]).abs() > 1
    rg = values.square().sum(dim=-1).mean().sqrt()
    centered_gram = -0.5 * (
        distances.square()
        - distances.square().mean(dim=0, keepdim=True)
        - distances.square().mean(dim=1, keepdim=True)
        + distances.square().mean()
    )
    eigenvalues = torch.linalg.eigvalsh(centered_gram)
    negative_mass = eigenvalues.clamp_max(0).abs().sum() / eigenvalues.abs().sum().clamp_min(1e-12)
    return {
        "finite": bool(torch.isfinite(values).all()),
        "centering_max_abs_angstrom": float(values.mean(dim=0).abs().max().cpu()),
        "adjacent_distance_mean_angstrom": float(adjacent.mean().cpu()),
        "radius_of_gyration_angstrom": float(rg.cpu()),
        "non_neighbor_clash_fraction": float((distances[non_neighbor] < 3.0).double().mean().cpu()),
        "contact_density": float((distances[non_neighbor] < 8.0).double().mean().cpu()),
        "distance_symmetry_error": float((distances - distances.T).abs().max().cpu()),
        "distance_diagonal_error": float(distances.diagonal().abs().max().cpu()),
        "negative_gram_eigenvalue_mass_fraction": float(negative_mass.cpu()),
        "strict_euclidean_valid": bool(
            torch.isfinite(distances).all()
            and (distances - distances.T).abs().max() <= 1e-8
            and distances.diagonal().abs().max() == 0
            and negative_mass <= 1e-8
        ),
        "masking_exact": True,
    }


def _real_reference_distributions(
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    references = {}
    for name, values in rows.items():
        records = []
        for row in values:
            coordinates = center_coordinates(row["coordinates"][None], row["residue_mask"][None])
            records.append(_sampling_metrics(coordinates / EXPECTED_SCALE))
        references[name] = {
            key: float(np.mean([float(record[key]) for record in records]))
            for key in (
                "adjacent_distance_mean_angstrom",
                "radius_of_gyration_angstrom",
                "non_neighbor_clash_fraction",
                "contact_density",
            )
        }
    return references


@torch.no_grad()
def _sample_panel(
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    device: torch.device,
    output: Path,
    update: int,
    references: dict[str, dict[str, float]],
) -> dict[str, Any]:
    model.eval()
    records = []
    sample_dir = output / "samples" / f"update-{update:04d}"
    sample_dir.mkdir(parents=True, exist_ok=False)
    for stratum_index, stratum in enumerate(config["length_strata"]):
        length = int(stratum["maximum"])
        for sample_index in range(int(config["sampling_samples_per_stratum"])):
            seed = int(config["sampling_seed"]) + stratum_index * 100 + sample_index
            result = diffusion.sample(model, length=length, seed=seed, device=device)
            coordinates = result["coordinates"].detach().cpu()
            distances = result["distance_matrix"].detach().cpu()
            path = sample_dir / f"{stratum['name']}-{sample_index}.npz"
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, coordinates=coordinates.numpy(), distance_matrix=distances.numpy())
            temporary.replace(path)
            metrics = _sampling_metrics(coordinates)
            reference = references[stratum["name"]]
            records.append(
                {
                    "length_stratum": stratum["name"],
                    "length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "artifact_path": str(path),
                    "artifact_sha256": sha256_file(path),
                    **metrics,
                    "reference_distribution": reference,
                    "adjacent_reference_error_angstrom": abs(
                        float(metrics["adjacent_distance_mean_angstrom"]) - reference["adjacent_distance_mean_angstrom"]
                    ),
                    "radius_of_gyration_reference_relative_error": abs(
                        float(metrics["radius_of_gyration_angstrom"]) - reference["radius_of_gyration_angstrom"]
                    )
                    / max(reference["radius_of_gyration_angstrom"], 1e-8),
                    "clash_reference_error": abs(
                        float(metrics["non_neighbor_clash_fraction"]) - reference["non_neighbor_clash_fraction"]
                    ),
                    "contact_density_reference_error": abs(
                        float(metrics["contact_density"]) - reference["contact_density"]
                    ),
                }
            )
    vectors = []
    for record in records:
        data = np.load(record["artifact_path"])
        matrix = data["distance_matrix"][0]
        vectors.append((record["length"], matrix[np.triu_indices(record["length"], 1)]))
    near_duplicate_pairs = 0
    compared_pairs = 0
    for index, (length, first) in enumerate(vectors):
        for other_length, second in vectors[index + 1 :]:
            if length != other_length:
                continue
            compared_pairs += 1
            near_duplicate_pairs += float(np.sqrt(np.mean((first - second) ** 2))) < 1e-3
    first = records[0]
    replay = diffusion.sample(model, length=int(first["length"]), seed=int(first["seed"]), device=device)
    replay_coordinates = replay["coordinates"].detach().cpu().numpy()
    original_coordinates = np.load(first["artifact_path"])["coordinates"]
    deterministic_replay = bool(np.array_equal(replay_coordinates, original_coordinates))
    if not deterministic_replay:
        raise ValueError("E007 Phase-3F unconditional sampling replay is nondeterministic")
    return {
        "records": records,
        "sample_count": len(records),
        "fixed_noise_identity_sha256": _canonical_sha([(r["length"], r["seed"]) for r in records]),
        "near_duplicate_pairs": near_duplicate_pairs,
        "compared_pairs": compared_pairs,
        "deterministic_replay": deterministic_replay,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    update: int,
    samples_processed: int,
    valid_residues_processed: int,
    config_sha256: str,
    protected_hashes_sha256: str,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration_sha256": config_sha256,
        "protected_hashes_sha256": protected_hashes_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": _rng_state(),
        "optimizer_update": update,
        "sampler_cursor": update,
        "samples_processed": samples_processed,
        "valid_residues_processed": valid_residues_processed,
        "successful_optimizer_boundary": True,
        **NON_AUTHORIZING,
    }


def _load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config_sha256: str,
    protected_hashes_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("configuration_sha256") != config_sha256
        or payload.get("protected_hashes_sha256") != protected_hashes_sha256
    ):
        raise ValueError("E007 Phase-3F resume identity contradiction")
    if not payload.get("successful_optimizer_boundary"):
        raise ValueError("E007 Phase-3F resume checkpoint is not at a successful optimizer boundary")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng_state"])
    return payload


def classify_pilot(
    evaluations: dict[int, dict[str, Any]], sampling: dict[int, dict[str, Any]], config: dict[str, Any]
) -> str:
    if 0 not in evaluations or 1000 not in evaluations or 0 not in sampling or 1000 not in sampling:
        return "inconclusive_requires_review"
    initial = evaluations[0]["validation"]
    final = evaluations[1000]["validation"]
    threshold = config["classification_thresholds"]
    improvement = 1 - final["global"]["coordinate_v_mse"] / initial["global"]["coordinate_v_mse"]
    high = (
        1
        - final["by_timestep_bin"]["high_noise"]["coordinate_v_mse"]
        / initial["by_timestep_bin"]["high_noise"]["coordinate_v_mse"]
    )
    very_high = (
        1
        - final["by_timestep_bin"]["very_high_noise"]["coordinate_v_mse"]
        / initial["by_timestep_bin"]["very_high_noise"]["coordinate_v_mse"]
    )
    length_regressions = [
        final["by_length_stratum"][x["name"]]["coordinate_v_mse"]
        / initial["by_length_stratum"][x["name"]]["coordinate_v_mse"]
        - 1
        for x in config["length_strata"]
    ]
    denoising = (
        improvement >= float(threshold["minimum_validation_v_mse_relative_improvement"])
        and high >= float(threshold["minimum_high_noise_relative_improvement"])
        and very_high >= float(threshold["minimum_very_high_noise_relative_improvement"])
        and max(length_regressions) <= float(threshold["maximum_length_stratum_relative_regression"])
    )

    def quality(records: list[dict[str, Any]]) -> float:
        return float(
            np.mean(
                [
                    x.get("adjacent_reference_error_angstrom", abs(x["adjacent_distance_mean_angstrom"] - 3.8))
                    + x.get("radius_of_gyration_reference_relative_error", 0.0)
                    + x.get("clash_reference_error", x["non_neighbor_clash_fraction"])
                    + x.get("contact_density_reference_error", 0.0)
                    for x in records
                ]
            )
        )

    initial_quality = quality(sampling[0]["records"])
    final_quality = quality(sampling[1000]["records"])
    sampling_improvement = 1 - final_quality / max(initial_quality, 1e-12)
    strict_checks = (
        sampling[0].get("deterministic_replay", True)
        and sampling[1000].get("deterministic_replay", True)
        and all(
            record.get("finite", True)
            and record.get("strict_euclidean_valid", True)
            and record.get("masking_exact", True)
            and record.get("centering_max_abs_angstrom", 0.0) <= 5e-5 * EXPECTED_SCALE
            for record in sampling[1000]["records"]
        )
    )
    if not strict_checks:
        return "numerical_or_memory_failure"
    geometry = all(
        x.get("adjacent_reference_error_angstrom", abs(x["adjacent_distance_mean_angstrom"] - 3.8))
        <= float(threshold["maximum_adjacent_distance_error_angstrom"])
        and x.get("radius_of_gyration_reference_relative_error", 0.0)
        <= float(threshold["maximum_radius_of_gyration_relative_error"])
        and x["non_neighbor_clash_fraction"] <= float(threshold["maximum_non_neighbor_clash_fraction"])
        and x.get("strict_euclidean_valid", True)
        and x.get("finite", True)
        for x in sampling[1000]["records"]
    )
    sampling_learned = (
        sampling_improvement >= float(threshold["minimum_sampling_quality_relative_improvement"]) and geometry
    )
    if denoising and sampling_learned:
        return "real_data_learning_and_sampling_verified"
    if denoising:
        return "denoising_learned_but_sampling_not_learned"
    if improvement > 0 and any(
        value > float(threshold["maximum_length_stratum_relative_regression"]) for value in length_regressions
    ):
        return "length_limited_learning"
    return "insufficient_real_data_learning"


def _memory(device: torch.device) -> dict[str, float | None]:
    current = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "peak_rss_mib": current,
        "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
    }


def run_real_coordinate_pilot(config_path: str | Path, *, resume_from: str | None = None) -> dict[str, Any]:
    """Run the bounded pilot; never authorize continuation beyond update 1,000."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    prerequisites = verify_pilot_prerequisites(config)
    _memory_preflight(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or (staging.exists() and resume_from is None):
        raise FileExistsError(f"E007 Phase-3F output already exists: {output} or {staging}")
    if resume_from is not None and not staging.is_dir():
        raise FileNotFoundError("E007 Phase-3F resume requires the existing staging directory")
    if resume_from is None:
        staging.mkdir(parents=True)
        (staging / "checkpoints").mkdir()
    config_sha = sha256_file(config_path)
    protected_sha = _canonical_sha(prerequisites["hashes"])
    start_time = time.monotonic()
    heartbeat_path = staging / "heartbeat.json"

    def heartbeat(status: str, **values: Any) -> None:
        _atomic_json(heartbeat_path, {"status": status, "updated_utc": _utc_now(), **values, **NON_AUTHORIZING})

    heartbeat("initializing", optimizer_update=0)
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3F configured pilot requires CUDA")
        device = torch.device("cuda")
        authorization = _authorize(config)
        protected_before = {
            **prerequisites["hashes"],
            "dataset_shards": _canonical_sha(authorization.observed_shard_hashes),
        }
        selected = _select_rows(config, authorization)
        panel_manifest = {
            split: {name: [row["sample_id"] for row in rows] for name, rows in values.items()}
            for split, values in (("train", selected["train"]), ("validation", selected["validation"]))
        }
        _atomic_json(staging / "panel_manifest.json", panel_manifest)
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        random.seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
        with coordinate_model_execution_context(config["numerics"], device) as backend:
            model = EquivariantPairCoordinateUNet(**config["model"]).to(device)
            if sum(x.numel() for x in model.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise ValueError("E007 Phase-3F model parameter-count contradiction")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["optimizer"]["learning_rate"]),
                weight_decay=float(config["optimizer"]["weight_decay"]),
                betas=tuple(config["optimizer"]["betas"]),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _scheduler_factor(step, config))
            diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
            update = samples_processed = valid_residues_processed = 0
            if resume_from is not None:
                state = _load_checkpoint(
                    Path(resume_from),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config_sha256=config_sha,
                    protected_hashes_sha256=protected_sha,
                )
                update = int(state["optimizer_update"])
                samples_processed = int(state["samples_processed"])
                valid_residues_processed = int(state["valid_residues_processed"])
            train_cursor = {name: 0 for name in selected["train"]}
            for completed in range(1, update + 1):
                name = update_stratum(completed, config["length_strata"])
                train_cursor[name] += int(_regime_for_stratum(name, config)["physical_batch_size"])
            evaluations: dict[int, dict[str, Any]] = {}
            samplings: dict[int, dict[str, Any]] = {}
            metrics_path = staging / "metrics.jsonl"
            evaluation_path = staging / "evaluations.json"
            sampling_path = staging / "sampling.json"
            comparator_path = staging / "checkpoint_comparators.json"
            if evaluation_path.exists():
                evaluations = {int(k): v for k, v in json.loads(evaluation_path.read_text()).items()}
            if sampling_path.exists():
                samplings = {int(k): v for k, v in json.loads(sampling_path.read_text()).items()}
            references = _real_reference_distributions(selected["validation"])

            def evaluate(current: int) -> None:
                train_eval = {
                    name: rows[-int(config["train_evaluation_samples_per_stratum"]) :]
                    for name, rows in selected["train"].items()
                }
                evaluations[current] = {
                    "train": _evaluate_panel(model, diffusion, train_eval, config, device, seed_offset=1000000),
                    "validation": _evaluate_panel(
                        model, diffusion, selected["validation"], config, device, seed_offset=2000000
                    ),
                }
                _atomic_json(evaluation_path, {str(k): v for k, v in evaluations.items()})
                if current in config["sampling_updates"]:
                    samplings[current] = _sample_panel(
                        model,
                        diffusion,
                        config,
                        device,
                        staging,
                        current,
                        references,
                    )
                    _atomic_json(sampling_path, {str(k): v for k, v in samplings.items()})

            if update == 0:
                evaluate(0)
            while update < int(config["successful_optimizer_updates"]):
                next_update = update + 1
                name = update_stratum(next_update, config["length_strata"])
                regime = _regime_for_stratum(name, config)
                size = int(regime["physical_batch_size"])
                start = train_cursor[name]
                rows = selected["train"][name][start : start + size]
                if len(rows) != size:
                    raise RuntimeError("E007 Phase-3F deterministic training stream exhausted")
                train_cursor[name] += size
                prepared = prepare_coordinate_batch(rows, EXPECTED_SCALE, int(config["expected_downsample_factor"]))
                corruption = make_uniform_training_corruption(
                    prepared, diffusion, seed=int(config["seed"]) + next_update, device=device
                )
                optimizer.zero_grad(set_to_none=True)
                model.train()
                prediction = model(
                    corruption["batch"].noisy_coordinates,
                    corruption["batch"].timesteps,
                    prepared["lengths"].to(device),
                    corruption["mask"],
                    prepared["chain_continuity_mask"].to(device),
                )["v_prediction"]
                if torch.count_nonzero(prediction[~corruption["mask"]]) != 0:
                    raise ValueError("E007 Phase-3F model produced nonzero padded coordinates")
                loss = uniform_coordinate_v_mse(prediction, corruption["batch"].coordinate_v_target, corruption["mask"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("E007 Phase-3F non-finite loss")
                loss.backward()
                gradients = _gradient_evidence(model)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                update = next_update
                samples_processed += size
                valid_residues_processed += int(prepared["lengths"].sum())
                memory = _memory(device)
                if memory["cuda_allocated_mib"] > float(config["memory"]["maximum_cuda_allocated_mib"]) or memory[
                    "cuda_reserved_mib"
                ] > float(config["memory"]["maximum_cuda_reserved_mib"]):
                    raise MemoryError(f"E007 Phase-3F CUDA envelope exceeded: {memory}")
                metric = {
                    "optimizer_update": update,
                    "samples_processed": samples_processed,
                    "valid_residues_processed": valid_residues_processed,
                    "diffusion_timesteps": corruption["timesteps"].detach().cpu().tolist(),
                    "coordinate_v_mse": float(loss.detach().cpu()),
                    "gradient_norm": gradients["global_norm"],
                    "gradient_group_norms": gradients["group_norms"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - start_time,
                    "eta_seconds": (time.monotonic() - start_time) / update * (1000 - update),
                    "nonfinite_count": 0,
                    "overflow_count": 0,
                    **memory,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(metric, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                heartbeat(
                    "running",
                    optimizer_update=update,
                    samples_processed=samples_processed,
                    valid_residues_processed=valid_residues_processed,
                    memory=memory,
                )
                if update in config["evaluation_updates"]:
                    evaluate(update)
                if update in config["checkpoint_updates"]:
                    checkpoint = staging / "checkpoints" / f"step-{update:04d}.pt"
                    _atomic_torch(
                        checkpoint,
                        checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            update=update,
                            samples_processed=samples_processed,
                            valid_residues_processed=valid_residues_processed,
                            config_sha256=config_sha,
                            protected_hashes_sha256=protected_sha,
                        ),
                    )
                    _atomic_json(
                        checkpoint.with_suffix(".json"),
                        {
                            "optimizer_update": update,
                            "sha256": sha256_file(checkpoint),
                            "validation_coordinate_v_mse": evaluations[update]["validation"]["global"][
                                "coordinate_v_mse"
                            ],
                            **NON_AUTHORIZING,
                        },
                    )
                    checkpoint_records = [
                        {
                            "optimizer_update": candidate,
                            "checkpoint_path": str(staging / "checkpoints" / f"step-{candidate:04d}.pt"),
                            "validation_coordinate_v_mse": evaluations[candidate]["validation"]["global"][
                                "coordinate_v_mse"
                            ],
                            "sampling_quality_score": float(
                                np.mean(
                                    [
                                        record["adjacent_reference_error_angstrom"]
                                        + record["radius_of_gyration_reference_relative_error"]
                                        + record["clash_reference_error"]
                                        + record["contact_density_reference_error"]
                                        for record in samplings[candidate]["records"]
                                    ]
                                )
                            ),
                        }
                        for candidate in config["checkpoint_updates"]
                        if candidate <= update
                    ]
                    denoising = min(checkpoint_records, key=lambda item: item["validation_coordinate_v_mse"])
                    sampling_comparator = min(checkpoint_records, key=lambda item: item["sampling_quality_score"])
                    _atomic_json(
                        comparator_path,
                        {
                            "validation_denoising_comparator": denoising,
                            "sampling_quality_comparator": sampling_comparator,
                            "comparators_are_non_definitive": True,
                            **NON_AUTHORIZING,
                        },
                    )
            classification = classify_pilot(evaluations, samplings, config)
            prerequisites_after = verify_pilot_prerequisites(config)
            authorization_after = _authorize(config)
            protected_after = {
                **prerequisites_after["hashes"],
                "dataset_shards": _canonical_sha(authorization_after.observed_shard_hashes),
            }
            if protected_after != protected_before:
                raise ValueError("E007 Phase-3F protected inputs changed")
            report = {
                "status": "completed_mandatory_scientific_review_pause",
                "version": VERSION,
                "classification": classification,
                "optimizer_updates": update,
                "samples_processed": samples_processed,
                "valid_residues_processed": valid_residues_processed,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "model_parameter_sha256": _parameter_sha256(model),
                "evaluations": evaluations,
                "sampling": samplings,
                "panel_manifest_sha256": sha256_file(staging / "panel_manifest.json"),
                "protected_inputs_unchanged": True,
                "protected_input_hashes": protected_after,
                "numerical_backend": backend,
                "memory": _memory(device),
                "training_performed": True,
                **NON_AUTHORIZING,
            }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": config_sha,
            "report_sha256": sha256_file(staging / "report.json"),
            "optimizer_updates": 1000,
            "mandatory_scientific_review": True,
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        if sha256_file(staging / "report.json") == sha256_file(staging / "protocol.json"):
            raise ValueError("E007 Phase-3F report/protocol publication collision")
        heartbeat(
            "completed_mandatory_scientific_review_pause",
            optimizer_update=1000,
            report_sha256=sha256_file(staging / "report.json"),
        )
        staging.replace(output)
        return {
            "status": report["status"],
            "classification": report["classification"],
            "output_dir": str(output),
            **NON_AUTHORIZING,
        }
    except KeyboardInterrupt as error:
        heartbeat("interrupted", error_type=type(error).__name__, error_message="SIGINT")
        raise
    except MemoryError as error:
        heartbeat(
            "memory_limit_exceeded",
            classification="numerical_or_memory_failure",
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
        )
        raise
    except BaseException as error:
        heartbeat("failed", error_type=type(error).__name__, error_message=str(error)[:2000])
        raise
