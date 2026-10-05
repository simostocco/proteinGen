"""Isolated, device-aware E006 production calibration v2."""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.rich_geometry import RichGeometryDataset, collate_rich_geometry
from protein_distance_diffusion.models.rich_codesign import E006_ARCHITECTURE_VERSION
from protein_distance_diffusion.training.codesign import masked_sequence_inputs
from protein_distance_diffusion.training.rich_codesign_production import (
    LENGTH_REGIMES,
    _amp_context,
    _assert_finite_model,
    _atomic_json,
    _authorization,
    _canonical_hash,
    _dataset_identity,
    _memory,
    _memory_guard,
    _model,
    _move,
    _protected_hashes,
    _sha256,
    _stage_trainable,
    _utc_now,
    collate_sequence_pretraining,
    nearest_length_indices,
)
from protein_distance_diffusion.training.rich_codesign_smoke import _forward

CALIBRATION_V2_VERSION = "e006_production_calibration_v2"
STAGES = ("sequence-pretrain", "joint-train")
MIB = 1024**2


def validate_calibration_v2_config(config: dict[str, Any]) -> None:
    calibration = config.get("calibration", {})
    if calibration.get("version") != CALIBRATION_V2_VERSION:
        raise ValueError("E006 calibration-v2 version contradiction")
    occupancy = float(calibration.get("maximum_total_device_occupancy_fraction", 0))
    reserve = float(calibration.get("minimum_remaining_vram_mib", 0))
    if not math.isfinite(occupancy) or not 0 < occupancy <= 0.90:
        raise ValueError("E006 calibration-v2 occupancy ceiling must be in (0, 0.90]")
    if not math.isfinite(reserve) or reserve < 768:
        raise ValueError("E006 calibration-v2 remaining VRAM must be at least 768 MiB")
    stages = calibration.get("stages", [])
    if tuple(item.get("stage") for item in stages) != STAGES:
        raise ValueError(f"E006 calibration-v2 stages must be {STAGES}")
    for stage in stages:
        regimes = stage.get("regimes", [])
        if tuple(int(item.get("target_length", -1)) for item in regimes) != LENGTH_REGIMES:
            raise ValueError(f"E006 calibration-v2 must cover length regimes {LENGTH_REGIMES}")
        for regime in regimes:
            target_budget = int(regime.get("target_effective_token_budget", 0))
            candidates = regime.get("candidates", [])
            if target_budget < 1 or not candidates:
                raise ValueError("E006 calibration-v2 regimes require a token budget and candidates")
            if any(
                int(item.get("physical_batch_size", 0)) < 1 or int(item.get("accumulation_steps", 0)) < 1
                for item in candidates
            ):
                raise ValueError("E006 calibration-v2 batch and accumulation sizes must be positive")


def cuda_device_memory_snapshot(device: torch.device) -> dict[str, float]:
    """Measure allocator and device-wide CUDA memory after synchronization."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E006 calibration-v2 device telemetry requires CUDA")
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    device_used = total_bytes - free_bytes
    return {
        "total_vram_mib": total_bytes / MIB,
        "free_vram_mib": free_bytes / MIB,
        "device_used_vram_mib": device_used / MIB,
        "external_baseline_usage_mib": max(device_used - reserved, 0) / MIB,
        "torch_allocated_mib": allocated / MIB,
        "torch_reserved_mib": reserved / MIB,
    }


def infer_peak_device_memory(
    before: dict[str, float],
    *,
    peak_allocated_mib: float,
    peak_reserved_mib: float,
) -> dict[str, float]:
    total = float(before["total_vram_mib"])
    external = float(before["external_baseline_usage_mib"])
    peak_torch = max(float(peak_allocated_mib), float(peak_reserved_mib))
    peak_total = min(total, external + peak_torch)
    return {
        "inferred_peak_total_device_occupancy_mib": peak_total,
        "inferred_peak_total_device_occupancy_fraction": peak_total / total,
        "remaining_free_memory_estimate_mib": max(total - peak_total, 0.0),
    }


def evaluate_case_safety(
    case: dict[str, Any],
    *,
    maximum_occupancy_fraction: float = 0.90,
    minimum_remaining_mib: float = 768.0,
    require_cuda: bool = True,
) -> tuple[bool, list[str]]:
    reasons = []
    if case.get("status") in {"cuda_oom", "oom"}:
        reasons.append("handled_oom")
    if case.get("finite") is not True:
        reasons.append("nonfinite_or_unverified")
    if case.get("numerical_status") != "equivalent":
        reasons.append("numerical_equivalence_failed")
    keys = (
        "inferred_peak_total_device_occupancy_fraction",
        "remaining_free_memory_estimate_mib",
    )
    if require_cuda and any(case.get(key) is None for key in keys):
        reasons.append("missing_device_memory_telemetry")
        return False, reasons
    try:
        occupancy = float(case["inferred_peak_total_device_occupancy_fraction"])
        remaining = float(case["remaining_free_memory_estimate_mib"])
    except (KeyError, TypeError, ValueError):
        return False, [*reasons, "invalid_device_memory_telemetry"]
    if not math.isfinite(occupancy) or not math.isfinite(remaining):
        reasons.append("nonfinite_device_memory_telemetry")
    else:
        if occupancy > maximum_occupancy_fraction:
            reasons.append("total_device_occupancy_above_maximum")
        if remaining < minimum_remaining_mib:
            reasons.append("remaining_vram_below_minimum")
    return not reasons, reasons


def compare_numerical_outputs(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    names: Iterable[str],
    atol: float,
    rtol: float,
) -> tuple[bool, dict[str, dict[str, float | bool]]]:
    """Compare scientific outputs with finite, shape-aware tensor checks."""
    diagnostics = {}
    equivalent = True
    for name in names:
        left = reference[name].detach().float()
        right = candidate[name].detach().float()
        same_shape = left.shape == right.shape
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        maximum_absolute_error = float((left - right).abs().max()) if same_shape and left.numel() else 0.0
        matches = same_shape and finite and torch.allclose(left, right, atol=atol, rtol=rtol)
        diagnostics[name] = {
            "same_shape": same_shape,
            "finite": finite,
            "maximum_absolute_error": maximum_absolute_error,
            "equivalent": bool(matches),
        }
        equivalent &= bool(matches)
    return equivalent, diagnostics


def _candidate_identity(case: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(case["stage"]),
        int(case["target_length"]),
        int(case["physical_batch_size"]),
        int(case["accumulation_steps"]),
    )


def select_v2_recommendations(
    cases: Iterable[dict[str, Any]],
    stage_configs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select by microstep reduction then throughput, never memory occupancy."""
    case_list = [dict(item) for item in cases]
    recommendations = []
    decisions: dict[tuple[str, int, int, int], str] = {}
    for stage_config in stage_configs:
        stage = str(stage_config["stage"])
        for regime in stage_config["regimes"]:
            length = int(regime["target_length"])
            target = int(regime["target_effective_token_budget"])
            group = [item for item in case_list if item["stage"] == stage and int(item["target_length"]) == length]
            eligible = []
            for item in group:
                safe, reasons = evaluate_case_safety(item)
                if not safe:
                    decisions[_candidate_identity(item)] = ",".join(reasons)
                elif int(item["effective_token_budget"]) != target:
                    decisions[_candidate_identity(item)] = "effective_token_budget_not_preserved"
                else:
                    eligible.append(item)
            if not eligible:
                raise ValueError(f"No safe equivalent E006 calibration-v2 case for {stage} length {length}")
            minimum_microsteps = min(int(item["accumulation_steps"]) for item in eligible)
            finalists = [item for item in eligible if int(item["accumulation_steps"]) == minimum_microsteps]
            selected = max(
                finalists, key=lambda item: (float(item["tokens_per_second"]), -int(item["physical_batch_size"]))
            )
            for item in eligible:
                decisions[_candidate_identity(item)] = (
                    "selected" if item is selected else "no_microstep_or_throughput_advantage"
                )
            recommendations.append(
                {
                    "stage": stage,
                    "maximum_length": length,
                    "physical_batch_size": int(selected["physical_batch_size"]),
                    "accumulation_steps": int(selected["accumulation_steps"]),
                    "effective_token_budget": int(selected["effective_token_budget"]),
                    "maximum_pair_elements": int(selected["pair_elements"]) if stage == "joint-train" else None,
                    "peak_total_device_occupancy_mib": float(selected["inferred_peak_total_device_occupancy_mib"]),
                    "peak_total_device_occupancy_fraction": float(
                        selected["inferred_peak_total_device_occupancy_fraction"]
                    ),
                    "peak_cuda_allocated_mib": float(selected["peak_cuda_allocated_mib"]),
                    "peak_cuda_reserved_mib": float(selected["peak_cuda_reserved_mib"]),
                    "remaining_free_memory_estimate_mib": float(selected["remaining_free_memory_estimate_mib"]),
                    "samples_per_second": float(selected["samples_per_second"]),
                    "tokens_per_second": float(selected["tokens_per_second"]),
                    "numerical_status": str(selected["numerical_status"]),
                }
            )
    annotated = [
        {**item, "selection_decision": decisions.get(_candidate_identity(item), "not_considered")} for item in case_list
    ]
    return recommendations, annotated


def _stage_a_case(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(config.get("device", "cuda"))
    before = cuda_device_memory_snapshot(device)
    authorization = _authorization(config)
    dataset = RichGeometryDataset(authorization, split="train")
    count = int(candidate["physical_batch_size"])
    target = int(candidate["target_length"])
    indices = nearest_length_indices(dataset, target_length=target, count=count, seed=int(config["seed"]) + target)
    rows = [dataset[index] for index in indices]
    # This collator is intentionally O(N) and has no geometry/pair-feature fields.
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in collate_sequence_pretraining(rows).items()
    }
    if any("pair" in key or "geometry" in key for key in batch):
        raise RuntimeError("E006 Stage-A calibration constructed forbidden pair features")
    model = _model(config).to(device)
    _stage_trainable(model, "sequence-pretrain")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(config["optimizer"]["learning_rate"]))
    inputs, masked = masked_sequence_inputs(
        batch["sequence_token_ids"],
        batch["residue_mask"],
        mask_token_id=1,
        probability=float(config["objective"]["mask_fraction"]),
        seed=int(config["seed"]),
        step=target,
    )
    tolerance = config["calibration"].get("numerical_equivalence", {})
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=False):
            reference_logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
        with _amp_context(device, config)[0]:
            candidate_logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
    equivalent, numerical = compare_numerical_outputs(
        {"sequence_logits": reference_logits[masked]},
        {"sequence_logits": candidate_logits[masked]},
        names=("sequence_logits",),
        atol=float(tolerance.get("absolute_tolerance", 0.05)),
        rtol=float(tolerance.get("relative_tolerance", 0.05)),
    )
    del reference_logits, candidate_logits
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = None
    for _ in range(int(candidate["accumulation_steps"])):
        with _amp_context(device, config)[0]:
            logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
            loss = F.cross_entropy(logits[masked].float(), batch["sequence_token_ids"][masked])
        (loss / int(candidate["accumulation_steps"])).backward()
    assert loss is not None
    gradients = [parameter.grad for parameter in trainable]
    finite = bool(
        torch.isfinite(loss) and gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
    )
    optimizer.step()
    _assert_finite_model(model)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / MIB
    peak_reserved = torch.cuda.max_memory_reserved(device) / MIB
    inferred = infer_peak_device_memory(before, peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved)
    tokens = int(batch["residue_mask"].sum()) * int(candidate["accumulation_steps"])
    return {
        **candidate,
        "status": "passed" if finite else "nonfinite",
        "finite": finite,
        "numerical_status": "equivalent" if equivalent else "failed",
        "numerical_equivalence": numerical,
        "pair_features_constructed": False,
        "actual_lengths": batch["lengths"].tolist(),
        "pair_elements": 0,
        "effective_token_budget": target * count * int(candidate["accumulation_steps"]),
        "loss": float(loss.detach()),
        "free_vram_before_mib": before["free_vram_mib"],
        "total_vram_mib": before["total_vram_mib"],
        "external_baseline_usage_mib": before["external_baseline_usage_mib"],
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
        **inferred,
        **_memory(device),
        "samples_per_second": count * int(candidate["accumulation_steps"]) / elapsed,
        "tokens_per_second": tokens / elapsed,
    }


def _stage_b_case(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(config.get("device", "cuda"))
    before = cuda_device_memory_snapshot(device)
    authorization = _authorization(config)
    target = int(candidate["target_length"])
    count = int(candidate["physical_batch_size"])
    dataset = RichGeometryDataset(authorization, split="train")
    indices = nearest_length_indices(dataset, target_length=target, count=count, seed=int(config["seed"]) + target)
    rows = [dataset[index] for index in indices]
    model = _model(config).to(device)
    batch = _move(collate_rich_geometry(rows, maximum_pair_elements=count * 512 * 512), device)
    tolerance = config["calibration"].get("numerical_equivalence", {})
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=False):
            reference_outputs, _, _ = _forward(
                model,
                batch,
                config=config,
                step=target,
                mode="learned_geometry_gating",
            )
        with _amp_context(device, config)[0]:
            candidate_outputs, _, _ = _forward(
                model,
                batch,
                config=config,
                step=target,
                mode="learned_geometry_gating",
            )
    equivalent, numerical = compare_numerical_outputs(
        reference_outputs,
        candidate_outputs,
        names=("sequence_logits", "geometry_prediction"),
        atol=float(tolerance.get("absolute_tolerance", 0.05)),
        rtol=float(tolerance.get("relative_tolerance", 0.05)),
    )
    del reference_outputs, candidate_outputs
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["learning_rate"]))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    losses = None
    for accumulation_index in range(int(candidate["accumulation_steps"])):
        with _amp_context(device, config)[0]:
            _, losses, _ = _forward(
                model,
                batch,
                config=config,
                step=accumulation_index,
                mode="learned_geometry_gating",
            )
        (losses["total"] / int(candidate["accumulation_steps"])).backward()
    assert losses is not None
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(
        all(math.isfinite(float(value.detach())) for value in losses.values())
        and gradients
        and all(torch.isfinite(value).all() for value in gradients)
    )
    optimizer.step()
    _assert_finite_model(model)
    final_memory = _memory_guard(config, device)
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    peak_reserved = float(final_memory["peak_cuda_reserved_mib"])
    peak_allocated = float(final_memory["peak_cuda_allocated_mib"])
    inferred = infer_peak_device_memory(before, peak_allocated_mib=peak_allocated, peak_reserved_mib=peak_reserved)
    tokens = int(batch["residue_mask"].sum()) * int(candidate["accumulation_steps"])
    return {
        **candidate,
        "stage": "joint-train",
        "status": "passed" if finite else "nonfinite",
        "finite": finite,
        "numerical_status": "equivalent" if equivalent else "failed",
        "numerical_equivalence": numerical,
        "actual_lengths": batch["lengths"].tolist(),
        "pair_elements": int(count * batch["distance_matrices"].shape[-1] ** 2),
        "effective_token_budget": target * count * int(candidate["accumulation_steps"]),
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "free_vram_before_mib": before["free_vram_mib"],
        "total_vram_mib": before["total_vram_mib"],
        "external_baseline_usage_mib": before["external_baseline_usage_mib"],
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
        **inferred,
        **_memory(device),
        "samples_per_second": count * int(candidate["accumulation_steps"]) / elapsed,
        "tokens_per_second": tokens / elapsed,
    }


def _calibration_child(config_path: str, candidate: dict[str, Any], result_path: str) -> None:
    output = Path(result_path)
    try:
        config = load_yaml(config_path)
        torch.manual_seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]))
        result = (
            _stage_a_case(config, candidate)
            if candidate["stage"] == "sequence-pretrain"
            else _stage_b_case(config, candidate)
        )
    except torch.OutOfMemoryError as error:
        result = {
            **candidate,
            "status": "cuda_oom",
            "finite": False,
            "numerical_status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
        }
    except BaseException as error:
        result = {
            **candidate,
            "status": "failed",
            "finite": False,
            "numerical_status": "failed",
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
        }
    _atomic_json(output, result)


def run_case_isolated(
    config_path: str | Path,
    candidate: dict[str, Any],
    result_path: str | Path,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run exactly one candidate in a fresh spawn child."""
    output = Path(result_path)
    context = mp.get_context("spawn")
    process = context.Process(target=_calibration_child, args=(str(config_path), candidate, str(output)))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        result = {**candidate, "status": "timeout", "finite": False, "numerical_status": "failed"}
        _atomic_json(output, result)
        return result
    if not output.exists():
        raise RuntimeError(f"E006 calibration-v2 child exited {process.exitcode} without a case report")
    import json

    return json.loads(output.read_text())


def run_calibration_v2(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_calibration_v2_config(config)
    calibration = config["calibration"]
    output = Path(calibration["output_report"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E006 calibration-v2 report: {output}")
    authorization = _authorization(config)
    identity = _dataset_identity(authorization)
    protected_before = _protected_hashes(authorization)
    heartbeat = output.with_name("calibration_heartbeat.json")
    started = _utc_now()
    _atomic_json(heartbeat, {"status": "running", "version": CALIBRATION_V2_VERSION, "started_utc": started})
    cases = []
    case_directory = output.parent / ".calibration_v2_cases"
    case_directory.mkdir(parents=True, exist_ok=True)
    try:
        for stage_config in calibration["stages"]:
            stage = str(stage_config["stage"])
            for regime in stage_config["regimes"]:
                stopped = False
                target = int(regime["target_length"])
                for candidate_config in sorted(regime["candidates"], key=lambda item: int(item["physical_batch_size"])):
                    candidate = {
                        "stage": stage,
                        "target_length": target,
                        "physical_batch_size": int(candidate_config["physical_batch_size"]),
                        "accumulation_steps": int(candidate_config["accumulation_steps"]),
                    }
                    if stopped:
                        cases.append(
                            {
                                **candidate,
                                "status": "not_run_after_safety_failure",
                                "finite": False,
                                "numerical_status": "not_run",
                            }
                        )
                        continue
                    case_path = case_directory / ("-".join(map(str, _candidate_identity(candidate))) + ".json")
                    result = run_case_isolated(
                        config_path,
                        candidate,
                        case_path,
                        timeout_seconds=float(calibration.get("case_timeout_seconds", 3600)),
                    )
                    safe, reasons = evaluate_case_safety(
                        result,
                        maximum_occupancy_fraction=float(calibration["maximum_total_device_occupancy_fraction"]),
                        minimum_remaining_mib=float(calibration["minimum_remaining_vram_mib"]),
                    )
                    result.update(memory_safe=safe, memory_safety_reasons=reasons)
                    cases.append(result)
                    stopped = result.get("status") == "cuda_oom" or any(
                        reason in {"total_device_occupancy_above_maximum", "remaining_vram_below_minimum"}
                        for reason in reasons
                    )
                    _atomic_json(
                        heartbeat,
                        {
                            "status": "running",
                            "version": CALIBRATION_V2_VERSION,
                            "completed_cases": len(cases),
                            "current_case": candidate,
                            "timestamp_utc": _utc_now(),
                        },
                    )
        recommendations, annotated = select_v2_recommendations(cases, calibration["stages"])
        after = _authorization(config)
        identity_after = _dataset_identity(after)
        protected_after = _protected_hashes(after)
        if identity_after != identity or protected_after != protected_before:
            raise RuntimeError("E006 protected dataset changed during calibration-v2")
        report = {
            "status": "completed",
            "version": CALIBRATION_V2_VERSION,
            "architecture_version": E006_ARCHITECTURE_VERSION,
            "configuration_sha256": _canonical_hash(config),
            "dataset_identity": identity,
            "dataset_identity_after": identity_after,
            "protected_inputs_unchanged": True,
            "protected_input_hashes_before": protected_before,
            "protected_input_hashes_after": protected_after,
            "maximum_total_device_occupancy_fraction": float(calibration["maximum_total_device_occupancy_fraction"]),
            "minimum_remaining_vram_mib": float(calibration["minimum_remaining_vram_mib"]),
            "case_isolation": "spawn_child_process_per_case",
            "cases": annotated,
            "recommendations": recommendations,
            "authorizes_training": False,
            "training_performed": False,
            "production_checkpoints_written": 0,
            "optimizer_state_retained": False,
            "started_utc": started,
            "completed_utc": _utc_now(),
        }
        _atomic_json(output, report)
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "version": CALIBRATION_V2_VERSION,
                "completed_utc": report["completed_utc"],
                "report_path": str(output),
                "report_sha256": _sha256(output),
            },
        )
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "version": CALIBRATION_V2_VERSION,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "cases": cases,
            "authorizes_training": False,
            "training_performed": False,
            "completed_utc": _utc_now(),
        }
        _atomic_json(output, failure)
        _atomic_json(heartbeat, {**failure, "report_sha256": _sha256(output)})
        raise
