"""Bounded Phase 3I.4 v2 calibration/holdout workflow and read-only contracts.

Sampling is reachable only through the explicit calibration or holdout CLI
modes. Plan, contract-validation, and monitor functions perform read-only work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

VERSION = "e007_phase3i4_sampler_correction_v2"
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
    "authorizes_larger_validation": False,
    "optimizer_created": False,
    "backward_performed": False,
    "parameter_mutation": False,
}
PROTECTED = {
    "report.json": "7a653ec269230e4a71e05b94802058fa13d5c4d6efd15bde0d6adbc264036fb2",
    "paired_comparisons.json": "b57f2bb433715644b937cdd14d45774cdb3a6606924889eabe25919fd4828ff7",
    "per_sample_trajectories.json": "5f4874c24ad76898e6da6da5e6217155dc178ff0eeeac01a26585479a29d4954",
    "protocol.json": "9680f9c21e5fd99a7b166ef2b12743dc8d08a2375d415ecf90ccf36e74d47e7e",
    "contract.json": "834eb3e21251330eea56278996f195e3912841de75c9f5f81c62d824e37ea17d",
    "heartbeat.json": "8cfb44b5b9674a23a5c2522a76f6a0c62620a72f54db5b8e5fea7ddd6a8e4533",
    "artifact_inventory.json": "10b3d5b9169139a9c7207523c8427af381c62123f7d02a135cfa4520be46da38",
    "journal.jsonl": "ea6c228dac688cc1e0b579d763909f1f9330a2855045f5ab11f2035486f361e2",
}
EXECUTION_LOG = Path("logs/e007_denoiser_sampler_trajectory_audit_v1.log")
EXECUTION_LOG_SHA256 = "6a13dff5c8c3291b4030bf1aa9bb8ec22154f9c6cd57dafe7fe25aa0629439e1"
BASELINE_TOLERANCE_SHA256 = "349221fd5c23b58a81374ded1ab7ba857b0c04d3c549feb36a47df11adf10d87"
PHASES = ("calibration", "holdout")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_execution_log(path: str | Path) -> str:
    target = Path(path)
    if target != EXECUTION_LOG:
        raise ValueError(f"protected external execution log path mismatch: {target}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"protected external execution log missing: {target}")
    if sha(target) != EXECUTION_LOG_SHA256:
        raise ValueError(f"protected external execution log hash mismatch: {target}")
    return EXECUTION_LOG_SHA256


def load_config(path: str | Path) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation import e007_phase3i4_performance_v2 as kernels

    cfg = yaml.safe_load(Path(path).read_text())
    if cfg.get("version") != VERSION:
        raise ValueError("v2 config required")
    protocol_path = Path(cfg["protocol_file"])
    manifest_path = Path(cfg["panel_candidate_manifest"])
    if sha(protocol_path) != cfg["protocol_sha256"] or sha(manifest_path) != cfg["panel_candidate_sha256"]:
        raise ValueError("v2 prospective contract hash mismatch")
    protocol = json.loads(protocol_path.read_text())
    baseline = json.loads(Path("configs/e007_phase3i4_protocol_v1.json").read_text())
    if (
        protocol.get("version") != VERSION
        or protocol.get("selection") != baseline.get("selection")
        or protocol.get("authorization") != baseline.get("authorization")
    ):
        raise ValueError("v2 prospective gates or authorization changed")
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["calibration_panel"]) != 10 or len(manifest["candidates"]) != 7:
        raise ValueError("v2 fresh calibration/candidate contract mismatch")
    if manifest["sparse_timestep_sha256"] != kernels.schedule_hashes():
        raise ValueError("v2 sparse correction schedule mismatch")
    reuse_path = Path(cfg["reuse_manifest"])
    if sha(reuse_path) != cfg["reuse_manifest_sha256"]:
        raise ValueError("v2 reuse manifest hash mismatch")
    reuse = json.loads(reuse_path.read_text())
    if reuse.get("accepted") is not False or reuse.get("reused_unit_count") != 0:
        raise ValueError("v1 reuse is prohibited")
    if cfg["authorization"] != NON_AUTHORIZING:
        raise ValueError("authorization contract changed")
    for phase in PHASES:
        for kind in ("output", "staging"):
            path_value = Path(cfg[f"{phase}_{kind}_dir"])
            if "_v1" in path_value.name or path_value.name.endswith("v1.inprogress"):
                raise ValueError("v1 staging/output path refused")
    cfg["baseline_tolerances"] = json.loads(Path("configs/e007_phase3i4_baseline_tolerances_v1.json").read_text())[
        "tolerances"
    ]
    return cfg


def _panel(config: dict[str, Any], name: str, count: int, offset: int) -> list[dict[str, Any]]:
    result = []
    for length in config["lengths"]:
        for index in range(count):
            seed = config["panel"]["seed_base"] + offset + length * 100 + index
            result.append({"identity": f"{name}-n{length}-{index:02d}", "length": length, "index": index, "seed": seed})
    return result


def panels(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(Path(config["panel_candidate_manifest"]).read_text())
    return manifest["calibration_panel"], manifest["prospective_holdout_panel"]


def calibration_panel(config: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(Path(config["panel_candidate_manifest"]).read_text())["calibration_panel"]


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    return json.loads(Path(config["panel_candidate_manifest"]).read_text())["candidates"]


def paired_initial_noise(
    seed: int, length: int, *, device: Any = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Construct one centered noise tensor to share across each corrected/control pair."""
    from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise

    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    reference = torch.empty((1, length, 3), dtype=dtype, device=device)
    return centered_coordinate_noise(reference, mask, generator=generator)


def schedule_fraction(timestep: int, start_timestep: int, signal_fraction: float) -> float:
    """Ramp deterministically from weak correction at start to full at t=0."""
    if not 0 <= timestep <= start_timestep or start_timestep <= 0:
        return 0.0
    progress = (start_timestep - timestep) / start_timestep
    return float(signal_fraction) * progress


def select_policy(comparisons: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Lexicographic, fail-closed selection; requires calibration rows only."""
    if any(row.get("panel") != "calibration" for row in comparisons):
        raise ValueError("holdout data access during selection")
    passing = []
    for candidate in candidates(config)[1:]:
        rows = [r for r in comparisons if r["candidate"] == candidate["name"]]
        if len(rows) != len(config["lengths"]) or {r["length"] for r in rows} != set(config["lengths"]):
            continue
        if any(r.get("completion_fraction") != 1.0 or r.get("finite_fraction") != 1.0 for r in rows):
            continue
        if any(not r.get("all_primary_gates_pass", False) or not r.get("all_safeguards_pass", False) for r in rows):
            continue
        if any(r.get("primary_ci_direction") != "improved" for r in rows):
            continue
        passing.append(candidate)
    if not passing:
        return {"selected": None, "status": "fail_closed_no_passing_candidate", **NON_AUTHORIZING}
    if len(passing) != 1:
        return {"selected": None, "status": "fail_closed_ambiguous", **NON_AUTHORIZING}
    return {"selected": passing[0], "status": "selected_for_holdout_only", **NON_AUTHORIZING}


def validate_contract(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    protected = {}
    # Read-only contract validation uses the v1 protected Phase 3I.3 evidence only.
    base = Path(cfg["phase3i3_dir"])
    for name, expected in PROTECTED.items():
        path = base / name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError(f"protected Phase 3I.3 artifact mismatch: {path}")
        protected[name] = expected
    validate_execution_log(cfg["execution_log"])
    if (
        sha(Path(cfg["checkpoint"])) != cfg["checkpoint_sha256"]
        or sha(Path(cfg["source_config"])) != cfg["source_config_sha256"]
    ):
        raise ValueError("checkpoint/source configuration hash mismatch")
    return {
        "status": "contract_validated_read_only_non_authorizing",
        "protected_hashes": protected,
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "checkpoint_loaded": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    cal, holdout = panels(cfg)
    calibration_forwards = len(cal) * len(candidates(cfg)) * cfg["timesteps"]
    holdout_forwards = len(holdout) * 2 * cfg["timesteps"]
    return {
        "status": "planned_read_only_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha(Path(config_path)),
        "calibration_identities": len(cal),
        "holdout_identities": len(holdout),
        "candidate_count": len(candidates(cfg)),
        "calibration_forward_count": calibration_forwards,
        "holdout_forward_count": holdout_forwards,
        "planned_forward_count": calibration_forwards + holdout_forwards,
        "calibration_wall_time_bound_seconds": cfg["runtime"]["calibration_max_wall_seconds"],
        "holdout_wall_time_bound_seconds": cfg["runtime"]["holdout_max_wall_seconds"],
        "wall_time_bound_seconds": cfg["runtime"]["maximum_wall_seconds"],
        "final_path": cfg["calibration_output_dir"],
        "calibration_output_path": cfg["calibration_output_dir"],
        "calibration_staging_path": cfg["calibration_staging_dir"],
        "holdout_output_path": cfg["holdout_output_dir"],
        "holdout_staging_path": cfg["holdout_staging_dir"],
        "coordinate_scan_performed": False,
        "checkpoint_loaded": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "output_created": False,
        **NON_AUTHORIZING,
    }


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def phase_paths(config: dict[str, Any], phase: str) -> tuple[Path, Path]:
    if phase not in PHASES:
        raise ValueError("unknown Phase 3I.4 phase")
    return Path(config[f"{phase}_staging_dir"]), Path(config[f"{phase}_output_dir"])


def work_units(config: dict[str, Any], phase: str, selected: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    calibration, holdout = panels(config)
    if phase == "calibration":
        return [
            {"phase": phase, "candidate": candidate["name"], "candidate_sha256": canonical_sha(candidate), **identity}
            for candidate in candidates(config)
            for identity in calibration
        ]
    if phase != "holdout" or selected is None:
        raise ValueError("holdout units require a verified selected policy")
    selected_candidate = selected["selected"]
    if not isinstance(selected_candidate, dict) or selected_candidate not in candidates(config):
        raise ValueError("invalid selected policy")
    control = candidates(config)[0]
    return [
        {"phase": phase, "candidate": candidate["name"], "candidate_sha256": canonical_sha(candidate), **identity}
        for candidate in (control, selected_candidate)
        for identity in holdout
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unit_relative(index: int) -> Path:
    return Path("trajectories") / f"trajectory-{index:04d}.json"


def _validate_trajectory_payload(unit: dict[str, Any], payload: dict[str, Any], contract: dict[str, Any]) -> None:
    if payload.get("unit") != unit or payload.get("contract_sha256") != canonical_sha(contract):
        raise ValueError("trajectory identity or contract corruption")
    milestone_rows = payload.get("milestones")
    expected_steps = set((499, 425, 375, 250, 150, 75, 25, 0))
    expected_representations = {"x_t", "v_hat", "x0_hat", "x0_guided", "next_state"}
    observed = {(row.get("timestep"), row.get("representation")) for row in milestone_rows or []}
    expected = {(step, rep) for step in expected_steps for rep in expected_representations}
    if len(milestone_rows or []) != len(expected) or observed != expected:
        raise ValueError("incomplete committed candidate/sample trajectory")
    for row in milestone_rows:
        coords = row.get("coordinates_normalized")
        if (
            not isinstance(coords, list)
            or len(coords) != unit["length"]
            or any(
                not isinstance(point, list)
                or len(point) != 3
                or any(value is None or not math.isfinite(float(value)) for value in point)
                for point in coords
            )
        ):
            raise ValueError("trajectory has invalid or nonfinite coordinate payload")
    final = payload.get("final_metrics")
    if not isinstance(final, dict) or final.get("finite_coordinate_rate") != 1.0 or final.get("completion_rate") != 1.0:
        raise ValueError("trajectory final metrics are incomplete or nonfinite")


def _journal_rows(staging: Path, work: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    journal = staging / "journal.jsonl"
    if journal.is_symlink():
        raise ValueError("journal symlink rejected")
    if not journal.exists():
        return []
    raw = journal.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        raise ValueError("truncated journal")
    records = []
    for index, line in enumerate(raw.splitlines()):
        if index >= len(work):
            raise ValueError("journal exceeds strict work prefix")
        record = json.loads(line)
        expected_rel = _unit_relative(index).as_posix()
        artifact = (staging / expected_rel).resolve()
        if Path(record.get("path", "")).as_posix() != expected_rel:
            raise ValueError("journal path escape or unexpected path")
        if not artifact.is_relative_to(staging.resolve()) or (staging / expected_rel).is_symlink():
            raise ValueError("journal path escape or symlink")
        if record.get("index") != index or record.get("unit") != work[index]:
            raise ValueError("journal is not a strict work-plan prefix")
        if record.get("contract_sha256") != canonical_sha(contract):
            raise ValueError("journal contract mismatch")
        if not artifact.is_file() or sha(artifact) != record.get("sha256"):
            raise ValueError("trajectory hash mismatch")
        rows = json.loads(artifact.read_text(encoding="utf-8"))
        _validate_trajectory_payload(work[index], rows, contract)
        records.append(record)
    committed = {record["path"] for record in records}
    for path in (staging / "trajectories").glob("trajectory-*.json") if (staging / "trajectories").exists() else ():
        if path.relative_to(staging).as_posix() not in committed:
            raise ValueError("uncommitted trajectory file present")
    if list(staging.rglob("*.tmp")):
        raise ValueError("temporary file found outside an atomic trajectory boundary")
    return records


def _commit_trajectory(
    staging: Path, index: int, unit: dict[str, Any], contract: dict[str, Any], trajectory: dict[str, Any]
) -> dict[str, Any]:
    _validate_trajectory_payload(
        unit, {"unit": unit, "contract_sha256": canonical_sha(contract), **trajectory}, contract
    )
    relative = _unit_relative(index)
    payload = {"unit": unit, "contract_sha256": canonical_sha(contract), **trajectory, **NON_AUTHORIZING}
    path = staging / relative
    _atomic_json(path, payload)
    record = {
        "index": index,
        "unit": unit,
        "path": relative.as_posix(),
        "sha256": sha(path),
        "contract_sha256": canonical_sha(contract),
    }
    with (staging / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(staging, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def _phase_contract(
    config_path: str | Path, config: dict[str, Any], phase: str, work: list[dict[str, Any]], selection: Any = None
) -> dict[str, Any]:
    protected = validate_contract(config_path)
    source = {
        "phase": phase,
        "config_sha256": sha(Path(config_path)),
        "checkpoint_sha256": protected["checkpoint_sha256"],
        "protected_hashes": protected["protected_hashes"],
        "work_sha256": canonical_sha(work),
        "numerical_policy": config["numerics"],
        "rng_contract": "torch_generator_per_identity_seed_shared_across_candidate_pair_v1",
        "selected_policy_sha256": canonical_sha(selection) if selection is not None else None,
    }
    return source


def monitor(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    candidates_by_phase: dict[str, Any] = {}
    for phase in PHASES:
        staging, final = phase_paths(config, phase)
        root = final if final.exists() else staging
        if not root.exists():
            planned_count = 70 if phase == "calibration" else 40
            candidates_by_phase[phase] = {
                "status": "not_started",
                "completed_trajectories": 0,
                "planned_trajectories": planned_count,
                "completed_forwards": 0,
                "planned_forwards": planned_count * config["timesteps"],
            }
            continue
        heartbeat_path = root / "heartbeat.json"
        heartbeat = json.loads(heartbeat_path.read_text()) if heartbeat_path.is_file() else {}
        if not heartbeat:
            heartbeat = {
                "status": "corrupt_monitor_state",
                "completed_trajectories": 0,
                "planned_trajectories": 0,
                "completed_forwards": 0,
                "planned_forwards": 0,
            }
        candidates_by_phase[phase] = heartbeat
    active = next(
        (phase for phase in PHASES if candidates_by_phase[phase].get("status") in {"running", "paused_wall_bound"}),
        None,
    )
    if active:
        observed = candidates_by_phase[active]
    elif candidates_by_phase["holdout"].get("status") != "not_started":
        active, observed = "holdout", candidates_by_phase["holdout"]
    else:
        active, observed = "calibration", candidates_by_phase["calibration"]
    planned = int(observed.get("planned_forwards", 0))
    completed = int(observed.get("completed_forwards", 0))
    return {
        "phase": active or observed.get("phase"),
        "candidate": observed.get("candidate"),
        "completed_trajectories": observed.get("completed_trajectories", 0),
        "planned_trajectories": observed.get("planned_trajectories", 0),
        "completed_forwards": completed,
        "planned_forwards": planned,
        "forward_weighted_percent": 100.0 * completed / planned if planned else 0.0,
        "selected_policy": observed.get("selected_policy"),
        "terminal_status": observed.get("status", "not_started"),
        **NON_AUTHORIZING,
    }


def _candidate_for_unit(config: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    candidate = next((row for row in candidates(config) if row["name"] == unit["candidate"]), None)
    if candidate is None or canonical_sha(candidate) != unit["candidate_sha256"]:
        raise ValueError("candidate identity/hash mismatch")
    return candidate


def _sample_trajectory(model: Any, diffusion: Any, config: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    from protein_distance_diffusion.evaluation import e007_phase3i4_performance_v2 as kernels
    from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import _representation_record

    candidate = _candidate_for_unit(config, unit)
    length, device = unit["length"], next(model.parameters()).device
    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    continuity = torch.ones((1, length - 1), dtype=torch.bool, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    x = paired_initial_noise(unit["seed"], length, device=device)
    start = candidate["start_timestep"]
    schedule = set(kernels.sparse_timesteps(start)) if start is not None else set()
    if start is not None:
        kernels.prepare_cache(length, x.dtype, device, starts=(start,))
    milestones = set(config["milestones"])
    rows = []
    with torch.inference_mode():
        for step in range(config["timesteps"] - 1, -1, -1):
            t = torch.tensor([step], dtype=torch.long, device=device)
            v_hat = model(x, t, lengths, mask, continuity)["v_prediction"]
            x0_hat = diffusion.reconstruct_x0(x, t, v_hat, mask)
            if candidate["family"] == "control" or step not in schedule or step == start:
                next_state, _, _ = diffusion.deterministic_reverse_step(x, t, v_hat, mask)
                guided_x0 = x0_hat
            else:
                x0_ang = x0_hat * config["coordinate_scale_angstrom"]
                projected = (
                    kernels.project_x0(
                        x0_ang,
                        mask,
                        family=candidate["family"],
                        iterations=candidate["iterations"],
                        displacement_cap=candidate["cap_angstrom"],
                        scheduled_clash=candidate["family"] == "composite",
                    )
                    / config["coordinate_scale_angstrom"]
                )
                frac = schedule_fraction(step, start, config["grid"]["signal_fraction"])
                guided_x0 = (1.0 - frac) * x0_hat + frac * projected
                alpha, sigma = diffusion.alpha_sigma(t, x)
                guided_v = (alpha * x - guided_x0) / sigma.clamp_min(1e-8)
                next_state, _, _ = diffusion.deterministic_reverse_step(x, t, guided_v, mask)
            if step in milestones:
                for rep, tensor in (
                    ("x_t", x),
                    ("v_hat", v_hat),
                    ("x0_hat", x0_hat),
                    ("x0_guided", guided_x0),
                    ("next_state", next_state),
                ):
                    rows.append(
                        {
                            "timestep": step,
                            "representation": rep,
                            **_representation_record(tensor[0].detach().cpu().numpy(), config),
                        }
                    )
            x = next_state
    final_metrics = next(row for row in rows if row["timestep"] == 0 and row["representation"] == "next_state")
    return {"milestones": rows, "final_metrics": final_metrics}


def _load_calibration_rows(final: Path, work: list[dict[str, Any]], contract: dict[str, Any]) -> list[dict[str, Any]]:
    records = _journal_rows(final, work, contract)
    if len(records) != len(work):
        raise ValueError("calibration publication is incomplete")
    trajectories = []
    for record in records:
        trajectories.append(json.loads((final / record["path"]).read_text()))
    return trajectories


def _paired_candidate_comparisons(config: dict[str, Any], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = {}
    for trajectory in trajectories:
        unit = trajectory["unit"]
        final[(unit["candidate"], unit["identity"])] = trajectory["final_metrics"]
    lower = (
        "adjacent_distance_rmse_to_3_8_angstrom",
        "adjacent_distance_violation_fraction",
        "i_plus_2_distance_error_to_6_2",
        "i_plus_3_distance_error_to_8_0",
        "bond_angle_error_to_110_degrees",
        "clash_fraction",
        "discontinuity_fraction",
    )
    primary_directions = {
        "adjacent_distance_rmse_to_3_8_angstrom": -1,
        "adjacent_distance_violation_fraction": -1,
        "locally_valid_residue_fraction": 1,
        "locally_valid_bond_fraction": 1,
    }
    tolerance_by_metric = config["baseline_tolerances"]
    global_fields = tuple(tolerance_by_metric["global_geometry_relative_by_metric"])
    result = []
    cal = calibration_panel(config)
    for candidate in candidates(config)[1:]:
        for length in config["lengths"]:
            identities = [row for row in cal if row["length"] == length]
            if len(identities) != 2:
                raise ValueError("calibration selection panel cardinality mismatch")
            paired = [
                (final[(candidate["name"], row["identity"])], final[("control", row["identity"])]) for row in identities
            ]
            directions_pass = True
            primary_intervals = {}
            for metric, direction in primary_directions.items():
                diffs = np.asarray([(a[metric] - b[metric]) * direction for a, b in paired], dtype=float)
                interval = _bootstrap_difference(
                    list(diffs),
                    [0.0] * len(diffs),
                    config["selection_gates"]["bootstrap_seed"] + length + len(result) + len(primary_intervals),
                    config["selection_gates"]["bootstrap_replicates"],
                )
                primary_intervals[metric] = interval
                if interval["ci95_high"] >= 0:
                    directions_pass = False
            safeguards_pass = True
            safeguard_intervals = {}
            safeguard_fields = tuple(
                dict.fromkeys((*lower, *global_fields, *tolerance_by_metric["chirality_absolute_by_metric"]))
            )
            for metric_index, metric in enumerate(safeguard_fields):
                left_values = [a.get(metric) for a, _ in paired]
                right_values = [b.get(metric) for _, b in paired]
                if (
                    any(value is None for value in (*left_values, *right_values))
                    or not np.isfinite([*left_values, *right_values]).all()
                ):
                    safeguards_pass = False
                    continue
                interval = _bootstrap_difference(
                    [float(value) for value in left_values],
                    [float(value) for value in right_values],
                    config["selection_gates"]["bootstrap_seed"] + length + 1000 + len(result) + metric_index,
                    config["selection_gates"]["bootstrap_replicates"],
                )
                safeguard_intervals[metric] = interval
                if metric in lower:
                    tolerance = tolerance_by_metric[metric]
                    # Primary metrics additionally require a confidence interval wholly in the improving direction.
                    gate_pass = interval["ci95_high"] <= tolerance
                    if metric in primary_directions:
                        gate_pass &= interval["ci95_high"] < 0
                elif metric in global_fields:
                    tolerance = (
                        abs(float(np.mean(right_values)))
                        * tolerance_by_metric["global_geometry_relative_by_metric"][metric]
                    )
                    gate_pass = interval["ci95_low"] >= -tolerance and interval["ci95_high"] <= tolerance
                else:
                    tolerance = tolerance_by_metric["chirality_absolute_by_metric"][metric]
                    gate_pass = interval["ci95_low"] >= -tolerance and interval["ci95_high"] <= tolerance
                safeguards_pass &= bool(gate_pass)
            # Preserve cross-sample diversity using pairwise C-alpha distance fingerprints.
            cand_div = _pairwise_distance_diversity([a["coordinates_normalized"] for a, _ in paired], config)
            ctrl_div = _pairwise_distance_diversity([b["coordinates_normalized"] for _, b in paired], config)
            diversity_pass = cand_div >= ctrl_div * (1 - tolerance_by_metric["diversity_relative"])
            result.append(
                {
                    "panel": "calibration",
                    "candidate": candidate["name"],
                    "length": length,
                    "completion_fraction": 1.0
                    if all(a["completion_rate"] == 1 and b["completion_rate"] == 1 for a, b in paired)
                    else 0.0,
                    "finite_fraction": 1.0
                    if all(a["finite_coordinate_rate"] == 1 and b["finite_coordinate_rate"] == 1 for a, b in paired)
                    else 0.0,
                    "all_primary_gates_pass": directions_pass,
                    "all_safeguards_pass": safeguards_pass and diversity_pass,
                    "primary_ci_direction": "improved" if directions_pass else "uncertain_or_worse",
                    "paired_bootstrap_primary_metrics": primary_intervals,
                    "paired_bootstrap_safeguards": safeguard_intervals,
                    "diversity_ratio": float(cand_div / ctrl_div) if ctrl_div else None,
                    **NON_AUTHORIZING,
                }
            )
    return result


def _pairwise_distance_diversity(coordinates: list[Any], config: dict[str, Any]) -> float:
    values = _pairwise_distance_diversity_values(coordinates, config)
    return float(np.mean(values)) if values else 0.0


def _pairwise_distance_diversity_values(coordinates: list[Any], config: dict[str, Any]) -> list[float]:
    if len(coordinates) < 2:
        return []
    physical = [np.asarray(item, dtype=float) * config["coordinate_scale_angstrom"] for item in coordinates]
    distance_matrices = [np.linalg.norm(row[:, None] - row[None, :], axis=-1) for row in physical]
    upper = np.triu_indices(len(physical[0]), 1)
    return [
        float(np.sqrt(np.mean((distance_matrices[i][upper] - distance_matrices[j][upper]) ** 2)))
        for i in range(len(distance_matrices))
        for j in range(i + 1, len(distance_matrices))
    ]


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    }


def _write_heartbeat(
    root: Path,
    phase: str,
    work: list[dict[str, Any]],
    committed: int,
    status: str,
    selected: Any = None,
    elapsed_seconds: float = 0.0,
) -> None:
    unit = work[committed - 1] if committed else (work[0] if work else {})
    _atomic_json(
        root / "heartbeat.json",
        {
            "phase": phase,
            "candidate": unit.get("candidate"),
            "completed_trajectories": committed,
            "planned_trajectories": len(work),
            "completed_forwards": committed * 500,
            "planned_forwards": len(work) * 500,
            "forward_weighted_percent": 100 * committed / len(work) if work else 0.0,
            "selected_policy": selected,
            "elapsed_seconds": elapsed_seconds,
            "status": status,
            **NON_AUTHORIZING,
        },
    )


def _bootstrap_difference(left: list[float], right: list[float], seed: int, replicates: int) -> dict[str, float]:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not len(a):
        raise ValueError("paired bootstrap requires equally sized nonempty arrays")
    delta = a - b
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(delta), (replicates, len(delta)))
    means = delta[draw].mean(axis=1)
    return {
        "paired_mean_difference": float(delta.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _calibration_publication(
    config: dict[str, Any], staging: Path, work: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    trajectories = _load_calibration_rows(staging, work, contract)
    comparisons = _paired_candidate_comparisons(config, trajectories)
    selection = select_policy(comparisons, config)
    _atomic_json(staging / "calibration_comparisons.json", comparisons)
    _atomic_json(
        staging / "failure_examples.json",
        [row for row in comparisons if not (row["all_primary_gates_pass"] and row["all_safeguards_pass"])][:20],
    )
    _atomic_json(
        staging / "selected_policy.json",
        {
            "status": selection["status"],
            "selected": selection["selected"],
            "selection_sha256": canonical_sha(selection),
            "candidate_grid_sha256": canonical_sha(candidates(config)),
            "calibration_panel_sha256": canonical_sha(panels(config)[0]),
            "holdout_identity_hash_only": canonical_sha(panels(config)[1]),
            "tolerance_evidence_sha256": BASELINE_TOLERANCE_SHA256,
            "config_sha256": contract["config_sha256"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "holdout_accessed": False,
            "thresholds_modified": False,
            **NON_AUTHORIZING,
        },
    )
    summary = {
        "phase": "calibration",
        "status": selection["status"],
        "trajectory_count": len(work),
        "forward_count": len(work) * config["timesteps"],
        "selected_policy": selection["selected"],
        "decision_categories": [] if selection["selected"] else ["inconclusive"],
        "scalar_composite_score_used": False,
        "holdout_accessed": False,
        "authorizes_larger_validation": False,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", summary)
    inventory = _file_inventory(staging)
    _atomic_json(staging / "artifact_inventory.json", inventory)
    return summary


def _verify_calibration(
    config_path: str | Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    _stage, final = phase_paths(config, "calibration")
    if not final.is_dir() or final.is_symlink():
        raise ValueError("holdout requires completed calibration publication")
    inventory_path = final / "artifact_inventory.json"
    if inventory_path.is_symlink():
        raise ValueError("calibration inventory symlink rejected")
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("calibration artifact inventory invalid")
    for relative, expected_hash in inventory.items():
        candidate_path = Path(relative)
        path = (final / candidate_path).resolve()
        if candidate_path.is_absolute() or ".." in candidate_path.parts or not path.is_relative_to(final.resolve()):
            raise ValueError("calibration inventory path escape")
        if (final / candidate_path).is_symlink() or not path.is_file() or sha(path) != expected_hash:
            raise ValueError(f"calibration artifact hash mismatch: {relative}")
    actual_files = {p.relative_to(final).as_posix() for p in final.rglob("*") if p.is_file()}
    if actual_files != set(inventory) | {"artifact_inventory.json"}:
        raise ValueError("calibration inventory has missing or unlisted files")
    report = json.loads((final / "report.json").read_text())
    record = json.loads((final / "selected_policy.json").read_text())
    if report.get("phase") != "calibration" or report.get("status") != "selected_for_holdout_only":
        raise ValueError("calibration did not uniquely select a passing policy")
    if record.get("status") != "selected_for_holdout_only" or record.get("holdout_accessed") is not False:
        raise ValueError("selected-policy record failed closed")
    if record.get("config_sha256") != sha(Path(config_path)) or record.get("candidate_grid_sha256") != canonical_sha(
        candidates(config)
    ):
        raise ValueError("calibration policy/config/grid mismatch")
    if record.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("calibration selected-policy checkpoint mismatch")
    if record.get("tolerance_evidence_sha256") != BASELINE_TOLERANCE_SHA256:
        raise ValueError("calibration tolerance evidence changed")
    if record.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("calibration selected-policy checkpoint mismatch")
    if record.get("calibration_panel_sha256") != canonical_sha(panels(config)[0]):
        raise ValueError("calibration panel identity mismatch")
    if record.get("holdout_identity_hash_only") != canonical_sha(panels(config)[1]):
        raise ValueError("predeclared holdout identity hash mismatch")
    selection_payload = {"selected": record["selected"], "status": record["status"], **NON_AUTHORIZING}
    if record.get("selection_sha256") != canonical_sha(selection_payload):
        raise ValueError("selected-policy hash mismatch")
    calibration_contract = json.loads((final / "contract.json").read_text())
    units = work_units(config, "calibration")
    if calibration_contract != _phase_contract(config_path, config, "calibration", units):
        raise ValueError("calibration contract/config/checkpoint mismatch")
    trajectories = _load_calibration_rows(final, units, calibration_contract)
    comparisons = json.loads((final / "calibration_comparisons.json").read_text())
    rederived_comparisons = _paired_candidate_comparisons(config, trajectories)
    if rederived_comparisons != comparisons:
        raise ValueError("calibration comparisons do not match committed trajectories")
    reproduced = select_policy(rederived_comparisons, config)
    if reproduced["status"] != record["status"] or reproduced["selected"] != record["selected"]:
        raise ValueError("calibration selection cannot be reproduced")
    return record, trajectories, calibration_contract


def _holdout_publication(
    config: dict[str, Any],
    staging: Path,
    work: list[dict[str, Any]],
    contract: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    records = _journal_rows(staging, work, contract)
    trajectories = [json.loads((staging / row["path"]).read_text()) for row in records]
    by_pair = {(row["unit"]["candidate"], row["unit"]["identity"]): row["final_metrics"] for row in trajectories}
    control = candidates(config)[0]
    chosen = selection["selected"]
    length_summaries = []
    primary = (
        "adjacent_distance_rmse_to_3_8_angstrom",
        "adjacent_distance_violation_fraction",
        "locally_valid_residue_fraction",
        "locally_valid_bond_fraction",
    )
    safeguard_losses = (
        "i_plus_2_distance_error_to_6_2",
        "i_plus_3_distance_error_to_8_0",
        "bond_angle_error_to_110_degrees",
        "clash_fraction",
        "discontinuity_fraction",
    )
    baseline_tol = config["baseline_tolerances"]
    global_metrics = tuple(baseline_tol["global_geometry_relative_by_metric"])
    chirality_metrics = tuple(baseline_tol["chirality_absolute_by_metric"])
    geometry_safeguards_pass = True
    for length in config["lengths"]:
        identities = [row for row in panels(config)[1] if row["length"] == length]
        for metric_index, metric in enumerate((*primary, *safeguard_losses, *global_metrics, *chirality_metrics)):
            a = [by_pair[(chosen["name"], row["identity"])][metric] for row in identities]
            b = [by_pair[(control["name"], row["identity"])][metric] for row in identities]
            stats = _bootstrap_difference(
                a,
                b,
                config["selection_gates"]["bootstrap_seed"] + length + metric_index,
                config["selection_gates"]["bootstrap_replicates"],
            )
            summary = {
                "length": length,
                "metric": metric,
                **stats,
            }
            length_summaries.append(summary)
            if metric in primary:
                continue
            if metric in safeguard_losses:
                tol = baseline_tol[metric]
                geometry_safeguards_pass &= stats["ci95_high"] <= tol
            elif metric in global_metrics:
                tol = abs(float(np.mean(b))) * baseline_tol["global_geometry_relative_by_metric"][metric]
                geometry_safeguards_pass &= stats["ci95_low"] >= -tol and stats["ci95_high"] <= tol
            else:
                tol = baseline_tol["chirality_absolute_by_metric"][metric]
                geometry_safeguards_pass &= stats["ci95_low"] >= -tol and stats["ci95_high"] <= tol
    finite = all(row["final_metrics"].get("finite_coordinate_rate") == 1 for row in trajectories)
    completed = len(records) == len(work)
    improved = all(
        next(row for row in length_summaries if row["length"] == length and row["metric"] == metric)["ci95_high"] < 0
        for length in config["lengths"]
        for metric in ("adjacent_distance_rmse_to_3_8_angstrom", "adjacent_distance_violation_fraction")
    ) and all(
        next(row for row in length_summaries if row["length"] == length and row["metric"] == metric)["ci95_low"] > 0
        for length in config["lengths"]
        for metric in ("locally_valid_residue_fraction", "locally_valid_bond_fraction")
    )
    diversity_by_length = {}
    diversity_intervals = {}
    diversity_safeguards_pass = True
    diversity_collapse = False
    for length in config["lengths"]:
        identities = [row for row in panels(config)[1] if row["length"] == length]
        candidates_metrics = [by_pair[(chosen["name"], row["identity"])] for row in identities]
        controls_metrics = [by_pair[(control["name"], row["identity"])] for row in identities]
        corrected_diversity_values = _pairwise_distance_diversity_values(
            [row["coordinates_normalized"] for row in candidates_metrics], config
        )
        control_diversity_values = _pairwise_distance_diversity_values(
            [row["coordinates_normalized"] for row in controls_metrics], config
        )
        corrected_diversity = float(np.mean(corrected_diversity_values))
        control_diversity = float(np.mean(control_diversity_values))
        ratio = corrected_diversity / control_diversity if control_diversity else 1.0
        diversity_by_length[length] = ratio
        diversity_stats = _bootstrap_difference(
            corrected_diversity_values,
            control_diversity_values,
            config["selection_gates"]["bootstrap_seed"] + length + 9000,
            config["selection_gates"]["bootstrap_replicates"],
        )
        diversity_intervals[length] = diversity_stats
        material_diversity_loss = control_diversity * baseline_tol["diversity_relative"]
        diversity_safeguards_pass &= diversity_stats["ci95_low"] >= -material_diversity_loss
        diversity_collapse |= diversity_stats["ci95_high"] < -material_diversity_loss
    safeguards_pass = geometry_safeguards_pass and diversity_safeguards_pass
    status = "prospective_holdout_complete" if completed and finite else "correction_instability"
    if not completed or not finite:
        category = "correction_instability"
    elif diversity_collapse:
        category = "diversity_collapse"
    elif improved and safeguards_pass:
        category = (
            "composite_projection_required" if chosen["family"] == "composite" else "sampler_correction_supported"
        )
    elif improved and not geometry_safeguards_pass:
        category = "global_geometry_degradation"
    elif improved and not diversity_safeguards_pass:
        category = "inconclusive"
    elif chosen["family"] == "bond":
        category = "bond_projection_insufficient"
    else:
        category = "representation_redesign_required"
    report = {
        "phase": "holdout",
        "status": status,
        "decision_categories": [category],
        "selected_policy": chosen,
        "holdout_trajectory_count": len(records),
        "forward_count": len(records) * config["timesteps"],
        "finite_completion": bool(completed and finite),
        "paired_bootstrap_per_length": length_summaries,
        "diversity_ratio_by_length": diversity_by_length,
        "paired_bootstrap_diversity_by_length": diversity_intervals,
        "all_safeguards_pass": bool(safeguards_pass),
        "scalar_composite_score_used": False,
        "selection_changed": False,
        "thresholds_changed": False,
        "authorizes_larger_validation": False,
        "failure_examples": sorted(
            (row["final_metrics"] | {"unit": row["unit"]} for row in trajectories),
            key=lambda row: row.get("adjacent_distance_rmse_to_3_8_angstrom", float("inf")),
            reverse=True,
        )[:10],
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "prospective_holdout_comparison.json", report)
    _atomic_json(
        staging / "per_trajectory_metrics.json",
        [row["final_metrics"] | {"unit": row["unit"]} | NON_AUTHORIZING for row in trajectories],
    )
    inventory = _file_inventory(staging)
    _atomic_json(staging / "artifact_inventory.json", inventory)
    return report


def _initialize_stage(
    staging: Path,
    phase: str,
    config: dict[str, Any],
    config_path: str | Path,
    work: list[dict[str, Any]],
    contract: dict[str, Any],
    selected: dict[str, Any] | None,
) -> None:
    staging.mkdir(parents=True)
    _atomic_json(staging / "contract.json", contract)
    _atomic_json(staging / "config_snapshot.json", config)
    calibration_panel, holdout_panel = panels(config)
    _atomic_json(staging / "active_panel.json", calibration_panel if phase == "calibration" else holdout_panel)
    candidate_manifest = [row | {"canonical_sha256": canonical_sha(row)} for row in candidates(config)]
    _atomic_json(staging / "candidate_definitions.json", candidate_manifest)
    _atomic_json(
        staging / "active_panel_hash.json",
        {
            "phase": phase,
            "active_panel_sha256": canonical_sha(calibration_panel if phase == "calibration" else holdout_panel),
        },
    )
    if phase == "calibration":
        _atomic_json(staging / "holdout_identity_hash.json", {"sha256": canonical_sha(holdout_panel)})
        _atomic_json(staging / "protocol.json", json.loads(Path(config["protocol_file"]).read_text()))
    else:
        _atomic_json(
            staging / "calibration_verification.json",
            {
                "calibration_inventory": json.loads(
                    (Path(config["calibration_output_dir"]) / "artifact_inventory.json").read_text()
                ),
                "selected_policy": selected,
            },
        )
    _write_heartbeat(staging, phase, work, 0, "running", selected.get("selected") if selected else None)


def execute(config_path: str | Path, phase: str, *, resume: bool = False) -> dict[str, Any]:
    """Execute one bounded phase; callers explicitly opt in with CLI run modes."""
    import resource

    # The scientific phase limit applies to each invocation.  The heartbeat's
    # elapsed_seconds is cumulative telemetry and must never consume a later
    # invocation's fresh wall-clock allowance.
    invocation_started = time.monotonic()

    config = load_config(config_path)
    if phase not in PHASES:
        raise ValueError("unknown execution phase")
    selected = None
    if phase == "holdout":
        selected, _calibration_rows, _calibration_contract = _verify_calibration(config_path, config)
    staging, final = phase_paths(config, phase)
    if resume:
        if final.exists() or not staging.is_dir() or staging.is_symlink():
            raise ValueError("resume requires an unfinished staging publication")
    elif staging.exists() or final.exists():
        raise FileExistsError("phase output already exists; use --resume only for an unfinished stage")
    work = work_units(config, phase, selected)
    contract = _phase_contract(config_path, config, phase, work, selected)
    if resume:
        if (staging / "contract.json").is_symlink() or (staging / "heartbeat.json").is_symlink():
            raise ValueError("resume contract/heartbeat symlink rejected")
        existing_contract = json.loads((staging / "contract.json").read_text())
        if existing_contract != contract:
            raise ValueError("resume contract/config/checkpoint/numerics/RNG/panel mismatch")
        heartbeat = json.loads((staging / "heartbeat.json").read_text())
        if heartbeat.get("status") not in {"running", "paused_wall_bound"}:
            raise ValueError("resume refuses failed-closed or completed state")
        records = _journal_rows(staging, work, contract)
        elapsed_before = float(heartbeat.get("elapsed_seconds", 0.0))
    else:
        _initialize_stage(staging, phase, config, config_path, work, contract, selected)
        records = []
        elapsed_before = 0.0
    if len(records) == len(work):
        if not (staging / "memory_telemetry.json").exists():
            _atomic_json(
                staging / "memory_telemetry.json", {"status": "resume_finalized_at_verified_trajectory_boundary"}
            )
        return _publish_phase(config, phase, staging, final, work, contract, selected, records, elapsed_before)
    import torch

    # Delay runtime/model imports until lifecycle and journal validation passes.
    from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import _load_model, _parameter_fingerprints
    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
    from protein_distance_diffusion.training.e007_local_backbone_repair import CudaMemoryTelemetry

    if not torch.cuda.is_available():
        raise RuntimeError("future Phase 3I.4 sampling requires CUDA; preflight remains CPU/read-only")
    device = torch.device("cuda")
    cuda_telemetry = CudaMemoryTelemetry(torch.cuda, device)
    model_config = dict(config)
    model_config["checkpoints"] = {"v_only": config["checkpoint"]}
    model = _load_model(model_config, "v_only", device)
    parameters_before = _parameter_fingerprints(model)
    diffusion = CoordinateVPDiffusion(config["timesteps"])
    try:
        with coordinate_model_execution_context(config["numerics"], device):
            for index in range(len(records), len(work)):
                invocation_elapsed = time.monotonic() - invocation_started
                cumulative_elapsed = elapsed_before + invocation_elapsed
                if invocation_elapsed > config["runtime"][f"{phase}_max_wall_seconds"]:
                    _write_heartbeat(
                        staging,
                        phase,
                        work,
                        index,
                        "paused_wall_bound",
                        selected.get("selected") if selected else None,
                        cumulative_elapsed,
                    )
                    return {
                        "status": "paused_wall_bound",
                        "phase": phase,
                        "completed_trajectories": index,
                        **NON_AUTHORIZING,
                    }
                unit = work[index]
                trajectory = _sample_trajectory(model, diffusion, config, unit)
                if parameters_before != _parameter_fingerprints(model):
                    raise RuntimeError("frozen model parameter mutation detected")
                record = _commit_trajectory(staging, index, unit, contract, trajectory)
                records.append(record)
                memory = cuda_telemetry.snapshot()
                rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if (
                    rss_mib > config["memory_mib"]["rss"]
                    or memory["run_peak_cuda_allocated_mib"] > config["memory_mib"]["cuda_allocated"]
                    or memory["run_peak_cuda_reserved_mib"] > config["memory_mib"]["cuda_reserved"]
                ):
                    raise MemoryError("Phase 3I.4 memory envelope exceeded")
                _write_heartbeat(
                    staging,
                    phase,
                    work,
                    len(records),
                    "running",
                    selected.get("selected") if selected else None,
                    elapsed_before + time.monotonic() - invocation_started,
                )
        if parameters_before != _parameter_fingerprints(model):
            raise RuntimeError("frozen model parameter mutation detected at phase end")
        protected_after = validate_contract(config_path)
        if (
            protected_after["checkpoint_sha256"] != contract["checkpoint_sha256"]
            or protected_after["protected_hashes"] != contract["protected_hashes"]
        ):
            raise RuntimeError("protected Phase 3I.3 artifacts changed during execution")
    except Exception:
        _write_heartbeat(
            staging,
            phase,
            work,
            len(records),
            "failed_closed",
            selected.get("selected") if selected else None,
            elapsed_before + time.monotonic() - invocation_started,
        )
        raise
    from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import _cleanup_cuda

    memory_peak = cuda_telemetry.snapshot()
    del model
    _cleanup = _cleanup_cuda(torch.cuda, cuda_telemetry)
    _atomic_json(staging / "memory_telemetry.json", {"peak": memory_peak, "cleanup": _cleanup})
    return _publish_phase(
        config,
        phase,
        staging,
        final,
        work,
        contract,
        selected,
        records,
        elapsed_before + time.monotonic() - invocation_started,
    )


def resume(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    active = [phase for phase in PHASES if phase_paths(config, phase)[0].exists()]
    if len(active) != 1:
        raise ValueError("resume requires exactly one active phase staging path")
    return execute(config_path, active[0], resume=True)


def _publish_phase(
    config: dict[str, Any],
    phase: str,
    staging: Path,
    final: Path,
    work: list[dict[str, Any]],
    contract: dict[str, Any],
    selected: dict[str, Any] | None,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if phase == "calibration":
        summary = _calibration_publication(config, staging, work, contract)
        terminal = summary["status"]
        selected_policy = summary["selected_policy"]
    else:
        if selected is None:
            raise ValueError("holdout selection missing at publication")
        summary = _holdout_publication(config, staging, work, contract, selected)
        terminal = summary["status"]
        selected_policy = selected["selected"]
    _write_heartbeat(staging, phase, work, len(records), terminal, selected_policy, elapsed_seconds)
    _atomic_json(staging / "artifact_inventory.json", _file_inventory(staging))
    if final.exists():
        raise FileExistsError("final path appeared during publication")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return summary
