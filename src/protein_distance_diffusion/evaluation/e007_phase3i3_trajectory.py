"""Bounded, non-authorizing Phase 3I.3 reverse-trajectory audit."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from protein_distance_diffusion.evaluation.e007_geometry_generator_capability import geometry_metrics
from protein_distance_diffusion.training.e007_local_backbone_repair import CudaMemoryTelemetry

VERSION = "e007_phase3i3_trajectory_audit_v1"
ARMS = ("original", "v_only", "v_plus_local")
REPRESENTATIONS = ("x_t", "v_hat", "x0_hat", "epsilon", "next_state")
MILESTONES = (499, 425, 375, 250, 150, 75, 25, 0)
LOCAL_DECISION_METRICS = {
    "adjacent_distance_rmse_to_3_8_angstrom": 1,
    "adjacent_distance_violation_fraction": 1,
    "i_plus_2_distance_error_to_6_2": 1,
    "i_plus_3_distance_error_to_8_0": 1,
    "bond_angle_error_to_110_degrees": 1,
    "clash_fraction": 1,
    "discontinuity_fraction": 1,
    "locally_valid_residue_fraction": -1,
}
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_additional_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_downstream_generation": False,
    "authorizes_sampler_correction": False,
    "authorizes_pilot_execution": False,
    "authorizes_phase3j": False,
    "optimizer_created": False,
    "backward_performed": False,
    "parameter_mutation": False,
}
CATEGORIES = {
    "denoiser_improvement_preserved_by_sampler",
    "sampler_transition_erases_denoiser_improvement",
    "cumulative_reverse_process_drift",
    "local_objective_model_regression",
    "representation_constraint_failure",
    "sampler_algebra_contradiction",
    "inconclusive",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if config.get("version") != VERSION:
        raise ValueError("Phase 3I.3 version mismatch")
    panel = config["panel"]
    if panel["lengths"] != [64, 128, 256, 384, 500] or panel["samples_per_length"] != 4:
        raise ValueError("Phase 3I.3 paired panel changed")
    if tuple(panel["milestones"]) != MILESTONES or config["diffusion_steps"] != 500:
        raise ValueError("Phase 3I.3 production timestep contract changed")
    if tuple(config["checkpoints"]) != ARMS:
        raise ValueError("Phase 3I.3 checkpoint arms changed")
    if config["runtime"]["estimated_model_forwards"] != 30000:
        raise ValueError("Phase 3I.3 forward count changed")
    if config["runtime"]["estimated_wall_seconds"] != 2400 or config["runtime"]["maximum_wall_seconds"] != 7200:
        raise ValueError("Phase 3I.3 wall-time envelope changed")
    if config["memory"] != {
        "maximum_rss_mib": 4096,
        "maximum_cuda_allocated_mib": 6144,
        "maximum_cuda_reserved_mib": 7680,
    }:
        raise ValueError("Phase 3I.3 memory envelope changed")
    if config["publication"]["all_authorization_fields_false"] is not True:
        raise ValueError("Phase 3I.3 authorization contract changed")
    return config


def identities(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"length": length, "sample_index": index, "seed": config["panel"]["seed_base"] + length * 100 + index}
        for length in config["panel"]["lengths"]
        for index in range(4)
    ]


def units(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"arm": arm, **identity} for arm in ARMS for identity in identities(config)]


def plan(config_path: str | Path) -> dict[str, Any]:
    """Parse static configuration only; do not inspect data or create output."""
    config = load_config(config_path)
    return {
        "status": "planned_read_only_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha(Path(config_path)),
        "final_path": config["output_dir"],
        "staging_path": config["staging_dir"],
        "models": 3,
        "samples_per_model": 20,
        "milestones_per_sample": 8,
        "representations_per_milestone": list(REPRESENTATIONS),
        "estimated_metric_records": 2400,
        "estimated_model_forwards": 30000,
        "estimated_wall_seconds": config["runtime"]["estimated_wall_seconds"],
        "estimate_basis": config["runtime"]["estimate_basis"],
        "maximum_wall_seconds": config["runtime"]["maximum_wall_seconds"],
        "memory_limits_mib": config["memory"],
        "coordinate_scan_performed": False,
        "checkpoint_loaded": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def validate_contract(config_path: str | Path) -> dict[str, Any]:
    """Read and hash only protected files and static configuration."""
    config = load_config(config_path)
    inventory = Path(config["protected_inventory"])
    if sha(inventory) != config["protected_inventory_sha256"]:
        raise ValueError("protected inventory hash mismatch")
    rows = json.loads(inventory.read_text())
    if len(rows) != 25 or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("protected inventory cardinality or uniqueness mismatch")
    hashes = {}
    for row in rows:
        path = Path(row["path"])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or sha(path) != row["sha256"]
        ):
            raise ValueError(f"protected artifact hash mismatch: {path}")
        hashes[row["path"]] = row["sha256"]
    for arm, path in config["checkpoints"].items():
        if path not in hashes or not path.endswith("step-09000.pt" if arm == "original" else "step-00500.pt"):
            raise ValueError(f"unprotected checkpoint: {arm}")
    pilot_root = Path(config["checkpoints"]["v_only"]).parents[2]
    actual_pilot_files = {path.as_posix() for path in pilot_root.rglob("*") if path.is_file()}
    expected_pilot_files = {row["path"] for row in rows if Path(row["path"]).is_relative_to(pilot_root)}
    if actual_pilot_files != expected_pilot_files:
        raise ValueError("protected pilot artifact inventory does not cover the completed output")
    for name in ("source_config", "relocation_config"):
        if sha(Path(config[name])) != config[f"{name}_sha256"]:
            raise ValueError(f"protected configuration hash mismatch: {name}")
    from protein_distance_diffusion.data.rich_geometry import validate_protected_input_relocations

    relocation = yaml.safe_load(Path(config["relocation_config"]).read_text())
    validate_protected_input_relocations(relocation["dataset"]["protected_input_relocations"])
    source = yaml.safe_load(Path(config["source_config"]).read_text())
    if not source.get("model") or relocation["diffusion_steps"] != 500:
        raise ValueError("source model or sampler configuration mismatch")
    report = json.loads((pilot_root / "report.json").read_text())
    protocol = json.loads((pilot_root / "protocol.json").read_text())
    if (
        report.get("status") != "completed_non_authorizing_bounded_pilot"
        or protocol.get("status") != report["status"]
        or any(
            report.get(key) is not False or protocol.get(key) is not False
            for key in ("authorizes_training", "authorizes_phase3j")
        )
    ):
        raise ValueError("protected pilot completion or authorization mismatch")
    return {
        "status": "contract_validated_read_only_non_authorizing",
        "protected_artifact_count": len(rows),
        "protected_inventory_sha256": sha(inventory),
        "checkpoint_hashes": {arm: hashes[path] for arm, path in config["checkpoints"].items()},
        "relocation_policy_validated": True,
        "coordinate_scan_performed": False,
        "checkpoint_loaded": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def verify_algebra() -> dict[str, bool]:
    """Independent CPU checks before any checkpoint is loaded."""
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import (
        CoordinateVPDiffusion,
        center_coordinates,
        centered_coordinate_noise,
    )

    diffusion = CoordinateVPDiffusion(500)
    mask = torch.tensor([[True, True, True, True, False, False]])
    clean = center_coordinates(torch.arange(18, dtype=torch.float64).reshape(1, 6, 3), mask)
    noise = centered_coordinate_noise(clean, mask, generator=torch.Generator().manual_seed(19))
    seeded = centered_coordinate_noise(clean, mask, generator=torch.Generator().manual_seed(19))
    checks = {"deterministic_seeded_equality": bool(torch.equal(noise, seeded))}
    for step in (0, 1, 25, 499):
        t = torch.tensor([step])
        alpha, sigma = diffusion.alpha_sigma(t, clean)
        x = alpha * clean + sigma * noise
        v = alpha * noise - sigma * clean
        expected_x0 = alpha * x - sigma * v
        expected_eps = sigma * x + alpha * v
        next_state, x0, eps = diffusion.deterministic_reverse_step(x, t, v, mask)
        prefix = f"t{step}_"
        checks[prefix + "forward_round_trip"] = bool(
            torch.allclose(alpha * x0 + sigma * eps, x, atol=1e-6)
            and torch.allclose(alpha * eps - sigma * x0, v, atol=1e-6)
        )
        checks[prefix + "x0_from_v"] = bool(torch.allclose(x0, clean, atol=1e-6))
        checks[prefix + "epsilon_from_v"] = bool(torch.allclose(eps, noise, atol=1e-6))
        checks[prefix + "production_reconstruction"] = bool(
            torch.allclose(x0, expected_x0, atol=1e-6) and torch.allclose(eps, expected_eps, atol=1e-6)
        )
        previous_alpha, previous_sigma = diffusion.alpha_sigma(torch.tensor([max(step - 1, 0)]), clean)
        expected_next = clean if step == 0 else previous_alpha * clean + previous_sigma * noise
        checks[prefix + "timestep_boundary"] = bool(torch.allclose(next_state, expected_next, atol=1e-6))
        checks[prefix + "mask_center_padding"] = bool(
            torch.count_nonzero(next_state[:, 4:]) == 0
            and torch.allclose(next_state[:, :4].mean(1), torch.zeros(1, 3, dtype=clean.dtype), atol=1e-6)
        )
        altered = x.clone()
        altered[:, 4:] = 1e8
        altered_next = diffusion.deterministic_reverse_step(altered, t, v, mask)[0]
        checks[prefix + "no_padded_leakage"] = bool(torch.allclose(next_state, altered_next, atol=1e-6))
    if not all(checks.values()):
        raise ValueError(f"sampler algebra contradiction: {[key for key, value in checks.items() if not value]}")
    return checks


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unit_path(index: int) -> Path:
    return Path("units") / f"unit-{index:03d}.json"


def _read_journal(staging: Path, work: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = staging / "journal.jsonl"
    records = []
    if not path.exists():
        return records
    journal_text = path.read_text()
    if journal_text and not journal_text.endswith("\n"):
        raise ValueError("truncated trajectory journal")
    for index, line in enumerate(journal_text.splitlines()):
        if index >= len(work):
            raise ValueError("trajectory journal exceeds work plan")
        record = json.loads(line)
        artifact = staging / _unit_path(index)
        if (
            record.get("index") != index
            or record.get("unit") != work[index]
            or record.get("path") != _unit_path(index).as_posix()
            or not artifact.is_file()
            or sha(artifact) != record.get("sha256")
        ):
            raise ValueError(f"corrupt committed trajectory unit {index}")
        rows = json.loads(artifact.read_text())
        expected_rows = {(step, representation) for step in MILESTONES for representation in REPRESENTATIONS}
        observed_rows = {(row.get("timestep"), row.get("representation")) for row in rows}
        if (
            len(rows) != 40
            or observed_rows != expected_rows
            or any(
                row.get(key) != work[index][key] for row in rows for key in ("arm", "seed", "length", "sample_index")
            )
        ):
            raise ValueError(f"corrupt trajectory row identity {index}")
        if any(
            not isinstance(row.get("coordinates_normalized"), list)
            or len(row["coordinates_normalized"]) != work[index]["length"]
            or any(not isinstance(point, list) or len(point) != 3 for point in row["coordinates_normalized"])
            for row in rows
        ):
            raise ValueError(f"corrupt trajectory coordinate shape {index}")
        records.append(record)
    return records


def _commit(staging: Path, index: int, unit: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    artifact = staging / _unit_path(index)
    _atomic_json(artifact, rows)
    record = {"index": index, "unit": unit, "path": _unit_path(index).as_posix(), "sha256": sha(artifact)}
    with (staging / "journal.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return record


def progress(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {arm: sum(record["unit"]["arm"] == arm for record in records) for arm in ARMS}
    return {
        "completed_models": sum(count == 20 for count in counts.values()),
        "completed_samples": len(records),
        "completed_milestones": len(records) * 8,
        "completed_model_forwards": len(records) * 500,
        "forward_weighted_percent": 100 * len(records) / 60,
        "completed_samples_by_model": counts,
    }


def monitor(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    staging = Path(config["staging_dir"])
    final = Path(config["output_dir"])
    root = final if final.exists() else staging
    if not root.exists():
        return {"status": "not_started", **progress([]), **NON_AUTHORIZING}
    records = _read_journal(root, units(config))
    return {"status": "complete" if root == final else "in_progress", **progress(records), **NON_AUTHORIZING}


def _metrics(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    physical = values.astype(np.float64) * config["coordinate_scale_angstrom"]
    if not np.isfinite(physical).all():
        return {"finite_coordinate_rate": float(np.isfinite(physical).all()), "completion_rate": 0.0}
    row, arrays = geometry_metrics(
        physical, source="generated", sample_id="trajectory", length=len(physical), config=config
    )
    adjacency = arrays["adjacent"]
    error = np.abs(adjacency - 3.8)
    valid_bond = error <= config["metrics"]["adjacent_tolerance_angstrom"]
    runs = np.diff(np.r_[False, ~valid_bond, False].astype(int))
    starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    residue_valid = np.r_[valid_bond[0], valid_bond[:-1] & valid_bond[1:], valid_bond[-1]]
    pair = np.linalg.norm(physical[:, None] - physical[None, :], axis=-1)
    upper = pair[np.triu_indices(len(physical), 1)]
    torsion = arrays["signed_pseudo_dihedral_radians"]
    torsion = torsion[np.isfinite(torsion)]
    row.update(
        {
            "adjacent_distance_violation_fraction": float(np.mean(~valid_bond)),
            "adjacent_absolute_error_p50": float(np.quantile(error, 0.5)),
            "adjacent_absolute_error_p90": float(np.quantile(error, 0.9)),
            "adjacent_absolute_error_p99": float(np.quantile(error, 0.99)),
            "i_plus_2_distance_error_to_6_2": float(np.mean(np.abs(arrays["distance_i_plus_2"] - 6.2))),
            "i_plus_3_distance_error_to_8_0": float(np.mean(np.abs(arrays["distance_i_plus_3"] - 8.0))),
            "bond_angle_error_to_110_degrees": float(
                np.nanmean(np.abs(np.degrees(arrays["bond_angle_radians"]) - 110))
            ),
            "longest_invalid_contiguous_segment": int(
                max((end - start + 1 for start, end in zip(starts, ends, strict=True)), default=0)
            ),
            "locally_valid_residue_fraction": float(residue_valid.mean()),
            "locally_valid_bond_fraction": float(valid_bond.mean()),
            "pair_distance_p10": float(np.quantile(upper, 0.1)),
            "pair_distance_p50": float(np.quantile(upper, 0.5)),
            "pair_distance_p90": float(np.quantile(upper, 0.9)),
            "signed_pseudo_dihedral_p50": float(np.quantile(torsion, 0.5)) if len(torsion) else None,
            "signed_pseudo_dihedral_p90": float(np.quantile(torsion, 0.9)) if len(torsion) else None,
            "finite_coordinate_rate": 1.0,
            "completion_rate": 1.0,
        }
    )
    return {
        key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
        for key, value in row.items()
    }


def _representation_record(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    serialized = [[float(value) if np.isfinite(value) else None for value in point] for point in values]
    return {
        "coordinates_normalized": serialized,
        "coordinate_scale_angstrom": config["coordinate_scale_angstrom"],
        **_metrics(values, config),
    }


def _load_model(config: dict[str, Any], arm: str, device: Any) -> Any:
    import torch

    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet

    source = yaml.safe_load(Path(config["source_config"]).read_text())
    model = EquivariantPairCoordinateUNet(**source["model"])
    checkpoint = torch.load(config["checkpoints"][arm], map_location="cpu", weights_only=False)
    expected_update = 9000 if arm == "original" else 500
    if int(checkpoint.get("optimizer_update", -1)) != expected_update:
        raise ValueError(f"checkpoint update mismatch: {arm}")
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    model.requires_grad_(False).eval().to(device)
    return model


def _cleanup_cuda(cuda: Any, telemetry: CudaMemoryTelemetry) -> dict[str, float]:
    gc.collect()
    cuda.empty_cache()
    return telemetry.end_phase()


def _parameter_fingerprints(model: Any) -> list[bytes]:
    return [hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).digest() for parameter in model.parameters()]


def _trajectory(
    model: Any, diffusion: Any, unit: dict[str, Any], config: dict[str, Any], device: Any
) -> list[dict[str, Any]]:
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise

    length, seed = unit["length"], unit["seed"]
    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    continuity = torch.ones((1, length - 1), dtype=torch.bool, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    x = centered_coordinate_noise(
        torch.empty((1, length, 3), device=device), mask, generator=torch.Generator(device=device).manual_seed(seed)
    )
    rows = []
    with torch.inference_mode():
        for step in range(499, -1, -1):
            t = torch.tensor([step], dtype=torch.long, device=device)
            v = model(x, t, lengths, mask, continuity)["v_prediction"]
            next_state, x0, epsilon = diffusion.deterministic_reverse_step(x, t, v, mask)
            if step in MILESTONES:
                for representation, tensor in zip(REPRESENTATIONS, (x, v, x0, epsilon, next_state), strict=True):
                    values = tensor[0].detach().cpu().numpy()
                    rows.append(
                        {
                            **unit,
                            "timestep": step,
                            "representation": representation,
                            **_representation_record(values, config),
                        }
                    )
            x = next_state
    return rows


def paired_bootstrap(rows: list[dict[str, Any]], *, seed: int, replicates: int) -> list[dict[str, Any]]:
    by_key = {(r["length"], r["sample_index"], r["arm"], r["timestep"], r["representation"]): r for r in rows}
    output = []
    comparisons = (("v_only", "original"), ("v_plus_local", "original"), ("v_plus_local", "v_only"))
    excluded = {
        "length",
        "requested_length",
        "sample_index",
        "seed",
        "timestep",
        "coordinate_scale_angstrom",
        "finite_coordinates",
        "padding_checked",
        "padding_exact_zero",
    }
    fields = tuple(
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    for length in (None, 64, 128, 256, 384, 500):
        selected = [
            key
            for key in sorted({(r["length"], r["sample_index"]) for r in rows})
            if length is None or key[0] == length
        ]
        for t in MILESTONES:
            for representation in REPRESENTATIONS:
                for left, right in comparisons:
                    for field in fields:
                        pairs = [
                            (
                                by_key[(*key, left, t, representation)].get(field),
                                by_key[(*key, right, t, representation)].get(field),
                            )
                            for key in selected
                        ]
                        if any(a is None or b is None for a, b in pairs):
                            continue
                        differences = np.asarray([a - b for a, b in pairs], dtype=np.float64)
                        if not np.isfinite(differences).all():
                            continue
                        rng = np.random.default_rng(seed + (0 if length is None else length) + t + len(output))
                        draws = rng.integers(0, len(differences), (replicates, len(differences)))
                        means = differences[draws].mean(axis=1)
                        output.append(
                            {
                                "length": length,
                                "timestep": t,
                                "representation": representation,
                                "comparison": f"{left}_minus_{right}",
                                "metric": field,
                                "paired_mean_difference": float(differences.mean()),
                                "ci95_low": float(np.quantile(means, 0.025)),
                                "ci95_high": float(np.quantile(means, 0.975)),
                            }
                        )
    return output


def sign_changes(
    comparisons: list[dict[str, Any]],
    length: int | None = None,
    metric: str = "adjacent_distance_rmse_to_3_8_angstrom",
) -> dict[str, Any]:
    if metric not in LOCAL_DECISION_METRICS:
        raise ValueError("unsupported local transition metric")
    direction = LOCAL_DECISION_METRICS[metric]
    selected = [
        r
        for r in comparisons
        if r["length"] == length and r["comparison"] == "v_plus_local_minus_v_only" and r["metric"] == metric
    ]
    by_representation = {
        rep: {r["timestep"]: direction * r["paired_mean_difference"] for r in selected if r["representation"] == rep}
        for rep in ("x0_hat", "next_state")
    }
    if any(set(values) != set(MILESTONES) for values in by_representation.values()):
        return {
            "metric": metric,
            "first_advantage_disappears": None,
            "first_sign_change": None,
            "deterioration_location": "inconclusive",
        }
    first_advantage = next((t for t in MILESTONES if by_representation["x0_hat"][t] < 0), None)
    if first_advantage is None:
        x0_rows = [r for r in selected if r["representation"] == "x0_hat"]
        return {
            "metric": metric,
            "first_advantage_milestone": None,
            "first_advantage_disappears": None,
            "first_sign_change": None,
            "deterioration_location": "local_objective_model_regression"
            if all((r["ci95_low"] if direction == 1 else -r["ci95_high"]) > 0 for r in x0_rows)
            else "inconclusive",
        }
    active_milestones = MILESTONES[MILESTONES.index(first_advantage) :]
    disappear = {
        rep: next((t for t in active_milestones if by_representation[rep][t] >= 0), None)
        for rep in ("x0_hat", "next_state")
    }
    sign = {
        rep: next((t for t in active_milestones if by_representation[rep][t] > 0), None)
        for rep in ("x0_hat", "next_state")
    }
    first_disappearance = next((t for t in active_milestones if t in disappear.values()), None)
    first_sign = next((t for t in active_milestones if t in sign.values()), None)
    location = (
        ("x0_hat" if first_disappearance == disappear["x0_hat"] else "sampler_transition")
        if first_disappearance is not None
        else "none"
    )
    return {
        "metric": metric,
        "first_advantage_milestone": first_advantage,
        "first_advantage_disappears": first_disappearance,
        "first_sign_change": first_sign,
        "first_disappearance_by_representation": disappear,
        "first_sign_change_by_representation": sign,
        "deterioration_location": location,
    }


def audit(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    import torch

    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    config = load_config(config_path)
    staging, final = Path(config["staging_dir"]), Path(config["output_dir"])
    if final.exists() or (staging.exists() and not resume) or (resume and not staging.exists()):
        raise FileExistsError("Phase 3I.3 output state conflicts with requested run mode")
    protected_before = validate_contract(config_path)
    algebra = verify_algebra()
    work = units(config)
    contract = {
        "configuration_sha256": sha(Path(config_path)),
        "inventory_sha256": protected_before["protected_inventory_sha256"],
        "units": work,
        "algebra": algebra,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3I.3 requires configured CUDA")
    if resume:
        if json.loads((staging / "contract.json").read_text()) != contract:
            raise ValueError("resume contract mismatch")
        records = _read_journal(staging, work)
        for path in (staging / "units").glob("unit-*.json"):
            if path not in {staging / record["path"] for record in records}:
                path.unlink()
    else:
        staging.mkdir(parents=True)
        _atomic_json(staging / "contract.json", contract)
        records = []
    device = torch.device("cuda")
    telemetry = CudaMemoryTelemetry(torch.cuda, device)
    diffusion = CoordinateVPDiffusion(500)
    started = time.monotonic()
    last_memory = telemetry.snapshot()
    with coordinate_model_execution_context(config["numerics"], device):
        for arm in ARMS:
            pending = [
                (index, unit) for index, unit in enumerate(work[len(records) :], len(records)) if unit["arm"] == arm
            ]
            if not pending:
                continue
            model = _load_model(config, arm, device)
            try:
                parameter_hashes = _parameter_fingerprints(model)
                for index, unit in pending:
                    rows = _trajectory(model, diffusion, unit, config, device)
                    if len(rows) != 40:
                        raise ValueError("incomplete milestone trajectory")
                    if parameter_hashes != _parameter_fingerprints(model):
                        raise ValueError("model parameter mutation detected before trajectory commit")
                    memory = telemetry.end_phase()
                    last_memory = memory
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    if (
                        rss > config["memory"]["maximum_rss_mib"]
                        or memory["run_peak_cuda_allocated_mib"] > config["memory"]["maximum_cuda_allocated_mib"]
                        or memory["run_peak_cuda_reserved_mib"] > config["memory"]["maximum_cuda_reserved_mib"]
                    ):
                        raise MemoryError("Phase 3I.3 memory envelope exceeded")
                    records.append(_commit(staging, index, unit, rows))
                    _atomic_json(
                        staging / "heartbeat.json",
                        {
                            "status": "running",
                            **progress(records),
                            "memory": memory,
                            "peak_rss_mib": rss,
                            **NON_AUTHORIZING,
                        },
                    )
                    if time.monotonic() - started > config["runtime"]["maximum_wall_seconds"]:
                        raise TimeoutError("Phase 3I.3 wall time exceeded")
                if parameter_hashes != _parameter_fingerprints(model):
                    raise ValueError("model parameter mutation detected")
            finally:
                del model
                last_memory = _cleanup_cuda(torch.cuda, telemetry)
    all_rows = [row for record in records for row in json.loads((staging / record["path"]).read_text())]
    comparisons = paired_bootstrap(
        all_rows, seed=config["publication"]["bootstrap_seed"], replicates=config["publication"]["bootstrap_replicates"]
    )
    transition = sign_changes(comparisons)
    transitions_by_length = {str(length): sign_changes(comparisons, length) for length in config["panel"]["lengths"]}
    transitions_by_local_metric = {
        metric: {
            "overall": sign_changes(comparisons, metric=metric),
            "by_length": {
                str(length): sign_changes(comparisons, length, metric) for length in config["panel"]["lengths"]
            },
        }
        for metric in LOCAL_DECISION_METRICS
    }
    categories = []
    if transition["deterioration_location"] == "local_objective_model_regression":
        categories.append("local_objective_model_regression")
    elif transition["deterioration_location"] == "inconclusive":
        categories.append("inconclusive")
    elif transition["first_advantage_disappears"] is None:
        categories.append("denoiser_improvement_preserved_by_sampler")
    elif transition["deterioration_location"] == "sampler_transition":
        categories.append("sampler_transition_erases_denoiser_improvement")
    else:
        categories.append("cumulative_reverse_process_drift")
    final_candidate_rows = [
        row
        for row in all_rows
        if row["arm"] == "v_plus_local" and row["timestep"] == 0 and row["representation"] == "next_state"
    ]
    if any(row.get("locally_valid_residue_fraction", 0) < 0.5 for row in final_candidate_rows):
        categories.append("representation_constraint_failure")
    if not set(categories) <= CATEGORIES:
        raise ValueError("unknown Phase 3I.3 decision category")
    if validate_contract(config_path)["checkpoint_hashes"] != protected_before["checkpoint_hashes"]:
        raise ValueError("protected checkpoint changed during audit")
    _atomic_json(staging / "per_sample_trajectories.json", all_rows)
    _atomic_json(staging / "paired_comparisons.json", comparisons)
    _atomic_json(
        staging / "protocol.json",
        {
            "version": VERSION,
            "status": "completed_non_authorizing",
            "configuration_sha256": sha(Path(config_path)),
            "protected_inventory_sha256": protected_before["protected_inventory_sha256"],
            "checkpoint_hashes": protected_before["checkpoint_hashes"],
            "production_sampler": "CoordinateVPDiffusion.deterministic_reverse_step",
            "milestones": list(MILESTONES),
            "representations": list(REPRESENTATIONS),
            "protected_artifacts_unchanged": True,
            "no_scalar_composite_score": True,
            **NON_AUTHORIZING,
        },
    )
    _atomic_json(
        staging / "report.json",
        {
            "status": "completed_non_authorizing",
            "categories": categories,
            "transition": transition,
            "transitions_by_length": transitions_by_length,
            "transitions_by_local_metric": transitions_by_local_metric,
            "decision_basis_metric": "adjacent_distance_rmse_to_3_8_angstrom",
            "no_scalar_composite_score": True,
            "protected_artifacts_unchanged": True,
            "memory_telemetry": last_memory,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "failure_examples": sorted(
                final_candidate_rows,
                key=lambda r: r.get("adjacent_distance_rmse_to_3_8_angstrom", -1),
                reverse=True,
            )[: config["publication"]["maximum_failure_examples"]],
            **progress(records),
            **NON_AUTHORIZING,
        },
    )
    _atomic_json(staging / "heartbeat.json", {"status": "complete", **progress(records), **NON_AUTHORIZING})
    inventory = [
        {"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]
    _atomic_json(staging / "artifact_inventory.json", inventory)
    os.replace(staging, final)
    return json.loads((final / "report.json").read_text())
