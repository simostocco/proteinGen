"""Production E005-Large launch gate and single-arm training contracts."""

from __future__ import annotations

import json
import math
import random
import signal
import time
from collections import defaultdict
from datetime import UTC, datetime
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryDataset
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.codesign import E005_ARCHITECTURE_VERSION
from protein_distance_diffusion.training.capacity_benchmark import _select_nearest_real_row
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.codesign import (
    _configuration_sha256,
    _parameter_groups,
    _sha256_file,
    _synthetic_items,
)
from protein_distance_diffusion.training.codesign_pilot import (
    DEFAULT_PAIR_BUDGET,
    VALIDATION_MODES,
    PilotInterrupted,
    _active_parameter_groups,
    _atomic_json,
    _dataset_identity,
    _forward_loss,
    _gradient_state,
    _group_gradient_metrics,
    _item_for_row,
    _memory,
    _model,
    _prepare_batch,
    _restore_gradients,
    _scan_bounded_rows,
    _synthetic_rows,
    accumulation_for_length,
)
from protein_distance_diffusion.training.trainer import _restore_rng_state, _rng_state

LARGE_ARCHITECTURE_VERSION = "e005_large_sequence_geometry_codesign_v1"
LARGE_SEQUENCE_SHAPE = (8, 384, 12, 1536)
LARGE_CURRICULUM = ((128, 14_000), (256, 14_000), (500, 7_000))
LARGE_TOTAL_UPDATES = 35_000
LARGE_PARAMETER_COUNTS = {
    "geometry_branch": 7_582_833,
    "sequence_transformer": 14_388_480,
    "token_embeddings_output_head": 16_918,
    "sequence_to_geometry_feedback": 385,
    "geometry_to_sequence_feedback": 2_304,
    "gates": 590_977,
    "total_model": 22_581_897,
}
TRAINING_MODE = "learned_geometry_gating"
MEMORY_TELEMETRY_KEYS = (
    "peak_rss_mib",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "current_rss_mib",
    "cuda_allocated_mib",
    "cuda_reserved_mib",
)


def large_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Return an exact, disjoint E005-Large parameter accounting."""
    modules = {
        "geometry_branch": (model.geometry_model,),
        "sequence_transformer": (model.position_embedding, model.sequence_encoder, model.sequence_norm),
        "token_embeddings_output_head": (model.token_embedding, model.sequence_logits),
        "sequence_to_geometry_feedback": (model.sequence_to_pair,),
        "geometry_to_sequence_feedback": (model.geometry_to_sequence, model.return_geometry_to_sequence),
        "gates": (model.geometry_to_sequence_gate, model.sequence_to_pair_gate, model.return_geometry_gate),
    }
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assigned: set[int] = set()
    counts: dict[str, int] = {}
    for name, values in modules.items():
        parameters = [parameter for module in values for parameter in module.parameters()]
        overlap = assigned & {id(parameter) for parameter in parameters}
        if overlap:
            raise RuntimeError(f"E005-Large parameter groups overlap in {name}")
        counts[name] = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        assigned.update(id(parameter) for parameter in parameters if parameter.requires_grad)
    if assigned != trainable:
        raise RuntimeError("E005-Large parameter accounting does not cover the trainable model")
    counts["total_model"] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def warmup_cosine_multiplier(step: int, *, total_updates: int, warmup_updates: int) -> float:
    """Linear warmup followed by cosine decay, indexed by completed updates."""
    if total_updates < 1 or not 0 <= warmup_updates < total_updates:
        raise ValueError("warmup_updates must be in [0, total_updates)")
    if warmup_updates and step < warmup_updates:
        return (step + 1) / warmup_updates
    progress = min(max((step - warmup_updates) / (total_updates - warmup_updates), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def validation_steps(stage_updates: tuple[int, ...], frequency: int) -> tuple[int, ...]:
    """Return preregistered validation positions including zero and boundaries."""
    if not stage_updates or any(value < 1 for value in stage_updates) or frequency < 1:
        raise ValueError("Validation cadence requires positive stages and frequency")
    total = sum(stage_updates)
    boundaries = set(np.cumsum(stage_updates).tolist())
    return tuple(sorted({0, total, *boundaries, *range(frequency, total + 1, frequency)}))


def checkpoint_kind(step: int, stage_updates: tuple[int, ...], frequency: int) -> tuple[bool, bool]:
    """Return whether latest and permanent checkpoints are due."""
    boundaries = set(np.cumsum(stage_updates).tolist())
    total = sum(stage_updates)
    return step > 0 and step % frequency == 0, step in boundaries or step == total


def validate_memory_telemetry(
    memory: dict[str, Any],
    *,
    cuda: bool,
    limits: dict[str, float],
    strict: bool,
) -> dict[str, float | None]:
    """Validate the canonical memory schema and configured upper bounds."""
    missing = [key for key in MEMORY_TELEMETRY_KEYS if key not in memory]
    if missing:
        raise ValueError(f"missing memory telemetry: {', '.join(missing)}")
    normalized: dict[str, float | None] = {}
    cuda_keys = {key for key in MEMORY_TELEMETRY_KEYS if "cuda" in key}
    for key in MEMORY_TELEMETRY_KEYS:
        value = memory[key]
        if value is None and key in cuda_keys and not cuda:
            normalized[key] = None
            continue
        if value is None:
            raise ValueError(f"missing {'CUDA ' if key in cuda_keys else ''}memory telemetry: {key}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid memory telemetry: {key}={value!r}") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"non-finite or negative memory telemetry: {key}={value!r}")
        normalized[key] = numeric
    for key, limit in limits.items():
        if key not in MEMORY_TELEMETRY_KEYS:
            raise ValueError(f"unknown memory telemetry limit: {key}")
        value = normalized[key]
        if value is None:
            raise ValueError(f"cannot apply memory limit without telemetry: {key}")
        crossed = value >= float(limit) if strict else value > float(limit)
        if crossed:
            relation = "below" if strict else "at most"
            raise MemoryError(f"memory criterion failed: {key}={value} must be {relation} {float(limit)}")
    return normalized


def enforce_memory_limits(
    memory: dict[str, float | None], *, max_rss_mib: int, max_cuda_memory_mib: int, cuda: bool
) -> None:
    """Fail before continued optimization when a configured memory ceiling is crossed."""
    limits = {"current_rss_mib": float(max_rss_mib)}
    if cuda:
        limits.update(
            {
                "cuda_allocated_mib": float(max_cuda_memory_mib),
                "cuda_reserved_mib": float(max_cuda_memory_mib),
            }
        )
    validate_memory_telemetry(memory, cuda=cuda, limits=limits, strict=False)


def validate_resume_checkpoint(saved: dict[str, Any], *, config_hash: str, dataset_hash: str, plan_hash: str) -> None:
    """Reject checkpoints from any different config, dataset, plan, or architecture."""
    expected = {
        "version": "e005_large_checkpoint_v1",
        "architecture_version": LARGE_ARCHITECTURE_VERSION,
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
        "plan_sha256": plan_hash,
    }
    contradictions = [name for name, value in expected.items() if saved.get(name) != value]
    if contradictions:
        raise ValueError(f"Incompatible E005-Large resume checkpoint: {', '.join(contradictions)}")


def validate_large_config(config: dict[str, Any], *, synthetic: bool = False) -> None:
    """Validate immutable scientific and operational E005-Large contracts."""
    model = config.get("model", {})
    shape = (
        int(model.get("sequence_layers", -1)),
        int(model.get("sequence_hidden_dim", -1)),
        int(model.get("sequence_heads", -1)),
        int(model.get("sequence_feedforward_dim", -1)),
    )
    if shape != LARGE_SEQUENCE_SHAPE:
        raise ValueError(f"E005-Large sequence shape must be {LARGE_SEQUENCE_SHAPE}, got {shape}")
    production = config.get("production", {})
    stages = tuple(
        (int(item["maximum_length"]), int(item["optimizer_updates"])) for item in production.get("curriculum", [])
    )
    if not synthetic and stages != LARGE_CURRICULUM:
        raise ValueError(f"E005-Large curriculum must be {LARGE_CURRICULUM}")
    if synthetic and (not stages or any(length < 1 or updates < 1 for length, updates in stages)):
        raise ValueError("Synthetic E005-Large stages must be positive")
    budget = tuple(
        (int(item["maximum_padded_length"]), int(item["microbatches"]))
        for item in production.get("pair_budget_accumulation", [])
    )
    if budget != DEFAULT_PAIR_BUDGET:
        raise ValueError("E005-Large must retain the validated pair-budget schedule")
    required = {
        "training_mode": TRAINING_MODE,
        "physical_batch_size": 1,
        "activation_checkpointing": True,
        "max_rss_mib": 6144,
        "max_cuda_memory_mib": 8192,
        "checkpoint_frequency": 250,
        "validation_frequency": 2500,
        "validation_panel_size": 256,
        "validation_corruption_seeds": 3,
    }
    mismatches = [key for key, expected in required.items() if production.get(key) != expected]
    if mismatches and not synthetic:
        raise ValueError(f"Invalid E005-Large production settings: {', '.join(mismatches)}")
    if str(config.get("amp_dtype")) != "float16" or config.get("mixed_precision") is not True:
        raise ValueError("E005-Large requires CUDA float16 AMP")
    if not synthetic and str(config.get("device")) != "cuda":
        raise ValueError("E005-Large production requires CUDA")
    if not 0 <= float(config.get("conditioning", {}).get("dropout_probability", -1)) <= 1:
        raise ValueError("conditioning dropout_probability must be in [0, 1]")
    optimizer = config.get("optimizer", {})
    if optimizer.get("name") != "AdamW" or float(optimizer.get("learning_rate", 0)) <= 0:
        raise ValueError("E005-Large requires AdamW with an explicit positive learning rate")
    if float(optimizer.get("weight_decay", -1)) < 0 or float(optimizer.get("gradient_clip_norm", 0)) <= 0:
        raise ValueError("E005-Large optimizer bounds are invalid")
    total = sum(updates for _, updates in stages)
    warmup = int(optimizer.get("warmup_updates", -1))
    warmup_cosine_multiplier(0, total_updates=total, warmup_updates=warmup)
    if tuple(config.get("validation", {}).get("modes", ())) != VALIDATION_MODES:
        raise ValueError("E005-Large validation must retain all three conditioning modes")


def _large_plan(config: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
    production = config["production"]
    if synthetic:
        train_rows = _synthetic_rows({**config, "pilot": production}, validation=False)
        validation_rows = _synthetic_rows({**config, "pilot": production}, validation=True)
    else:
        directory = Path(config["dataset"]["directory"])
        train_rows = []
        validation_rows = _scan_bounded_rows(
            directory / str(config["dataset"]["validation_dataset"]),
            maximum_length=500,
            seed=int(config["seed"]) + 1,
            capacity=int(production["validation_panel_size"]),
        )
    overlap = {row["sample_id"] for row in train_rows} & {row["sample_id"] for row in validation_rows}
    if overlap:
        raise ValueError(f"Train/validation leakage in E005-Large panel: {sorted(overlap)[0]}")
    core = {
        "version": "e005_large_plan_v1",
        "seed": int(config["seed"]),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "curriculum": production["curriculum"],
    }
    core["sha256"] = _configuration_sha256(core)
    return core


def _stage_for_step(config: dict[str, Any], step: int) -> tuple[int, dict[str, int]]:
    cursor = 0
    for index, stage in enumerate(config["production"]["curriculum"]):
        cursor += int(stage["optimizer_updates"])
        if step < cursor:
            return index, stage
    return len(config["production"]["curriculum"]) - 1, config["production"]["curriculum"][-1]


def _microbatch_rows(config: dict[str, Any], plan: dict[str, Any], step: int) -> list[tuple[dict[str, Any], int]]:
    stage_index, stage = _stage_for_step(config, step)
    eligible = [row for row in plan["train_rows"] if int(row["sequence_length"]) <= int(stage["maximum_length"])]
    if not eligible:
        raise ValueError(f"No rows for E005-Large curriculum stage {stage_index + 1}")
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        strata[(row["length_bin"], row["experimental_method"], row["pairing_classification"])].append(row)
    groups = [sorted(rows, key=lambda row: row["sample_id"]) for _, rows in sorted(strata.items())]
    anchor_group = groups[step % len(groups)]
    anchor = anchor_group[(step // len(groups)) % len(anchor_group)]
    budget = tuple(
        (int(item["maximum_padded_length"]), int(item["microbatches"]))
        for item in config["production"]["pair_budget_accumulation"]
    )
    count = accumulation_for_length(int(anchor["sequence_length"]), budget)
    same_bin = [row for row in eligible if row["length_bin"] == anchor["length_bin"]]
    return [
        (same_bin[(step * count + offset) % len(same_bin)], int(config["seed"]) + step * 1_000_003 + offset)
        for offset in range(count)
    ]


def _coprime_stride(length: int, seed: int) -> int:
    stride = max(1, (seed * 2 + 1) % length)
    while gcd(stride, length) != 1:
        stride = (stride + 2) % length or 1
    return stride


def _full_corpus_microbatch_rows(
    config: dict[str, Any], dataset: SequenceGeometryDataset, step: int
) -> list[tuple[dict[str, Any], int]]:
    """Select deterministic curriculum rows without retaining corpus metadata."""
    _, stage = _stage_for_step(config, step)
    maximum_length = int(stage["maximum_length"])
    target_bins = [maximum for maximum in (64, 128, 256, 384, 500) if maximum <= maximum_length]
    target_bin = target_bins[step % len(target_bins)]
    lower = 1 if target_bin == 64 else target_bins[target_bins.index(target_bin) - 1] + 1
    size = len(dataset)
    start = (int(config["seed"]) + step * 1_000_003) % size
    stride = _coprime_stride(size, int(config["seed"]) + step)
    anchor: dict[str, Any] | None = None
    for offset in range(size):
        row = dataset.row_metadata((start + offset * stride) % size)
        length = int(row["sequence_length"])
        if lower <= length <= target_bin and str(row["practical_training_eligibility"]) != "excluded_or_unresolved":
            row["experimental_method"] = str(row.get("experimental_method") or row.get("method") or "unknown")
            row["pairing_classification"] = str(
                row.get("pairing_classification") or row.get("v3_pairing_classification") or "unknown"
            )
            row["length_bin"] = f"le_{target_bin}"
            anchor = row
            break
    if anchor is None:
        raise ValueError(f"No eligible full-corpus row in length cohort {lower}-{target_bin}")
    budget = tuple(
        (int(item["maximum_padded_length"]), int(item["microbatches"]))
        for item in config["production"]["pair_budget_accumulation"]
    )
    count = accumulation_for_length(int(anchor["sequence_length"]), budget)
    rows = []
    for microbatch in range(count):
        # A distinct deterministic permutation start avoids repeatedly loading one chain.
        candidate_start = (start + microbatch * 104_729) % size
        selected = None
        for offset in range(size):
            row = dataset.row_metadata((candidate_start + offset * stride) % size)
            length = int(row["sequence_length"])
            if lower <= length <= target_bin and str(row["practical_training_eligibility"]) != "excluded_or_unresolved":
                row["experimental_method"] = str(row.get("experimental_method") or row.get("method") or "unknown")
                row["pairing_classification"] = str(
                    row.get("pairing_classification") or row.get("v3_pairing_classification") or "unknown"
                )
                row["length_bin"] = f"le_{target_bin}"
                selected = row
                break
        if selected is None:
            raise ValueError(f"No eligible full-corpus row in length cohort {lower}-{target_bin}")
        rows.append((selected, int(config["seed"]) + step * 1_000_003 + microbatch))
    return rows


def _assert_finite_parameters(model: torch.nn.Module) -> None:
    invalid = [name for name, parameter in model.named_parameters() if not torch.isfinite(parameter).all()]
    if invalid:
        raise FloatingPointError(f"nonfinite_parameters:{','.join(invalid[:10])}")


def _adamw_parameter_change_norms(
    optimizer: torch.optim.AdamW,
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> dict[str, float]:
    """Recover exact last-step AdamW deltas without retaining a model copy."""
    values: dict[int, float] = {}
    with torch.no_grad():
        for optimizer_group in optimizer.param_groups:
            beta1, beta2 = optimizer_group["betas"]
            learning_rate = float(optimizer_group["lr"])
            weight_decay = float(optimizer_group["weight_decay"])
            decay = 1.0 - learning_rate * weight_decay
            for parameter in optimizer_group["params"]:
                state = optimizer.state.get(parameter, {})
                if not state or "exp_avg" not in state:
                    values[id(parameter)] = 0.0
                    continue
                update_index = float(state["step"])
                bias_one = 1.0 - beta1**update_index
                bias_two = 1.0 - beta2**update_index
                variance = state.get("max_exp_avg_sq", state["exp_avg_sq"])
                denominator = variance.sqrt() / math.sqrt(bias_two) + float(optimizer_group["eps"])
                adaptive = (learning_rate / bias_one) * state["exp_avg"] / denominator
                previous = (parameter.detach() + adaptive) / decay
                values[id(parameter)] = float((parameter.detach() - previous).float().square().sum())
    return {
        name: sum(values[id(parameter)] for _, parameter in parameters) ** 0.5 for name, parameters in groups.items()
    }


def _gate_passes(gates: dict[str, dict[str, float]]) -> bool:
    return all(0.01 < values["mean"] < 0.99 and values["saturated_fraction"] < 0.95 for values in gates.values())


def _launch_gate_memory_limits(config: dict[str, Any]) -> dict[str, float]:
    values = config["launch_gate"]
    return {
        "peak_rss_mib": float(values["maximum_peak_rss_mib"]),
        "peak_cuda_allocated_mib": float(values["maximum_peak_cuda_allocated_mib"]),
        "peak_cuda_reserved_mib": float(values["maximum_peak_cuda_reserved_mib"]),
    }


def _validate_launch_gate_criteria(report: dict[str, Any], *, config: dict[str, Any], cuda: bool) -> None:
    report["memory"] = validate_memory_telemetry(
        report.get("memory", {}),
        cuda=cuda,
        limits=_launch_gate_memory_limits(config) if cuda else {"peak_rss_mib": 6144.0},
        strict=True,
    )
    losses = report.get("losses")
    if not isinstance(losses, dict) or not losses or any(not math.isfinite(float(value)) for value in losses.values()):
        raise ValueError("launch-gate losses are missing or non-finite")
    if report.get("optimizer_step_completed") is not True:
        raise ValueError("launch-gate optimizer step did not complete")
    if report.get("gradient_checks", {}).get("all_active_parameter_groups_nonzero") is not True:
        raise ValueError("launch-gate active gradient checks failed")
    if report.get("gate_checks", {}).get("valid_nonsaturated") is not True:
        raise ValueError("launch-gate learned-gate checks failed")
    before = report.get("dataset_before_sha256")
    after = report.get("dataset_after_sha256")
    if not before or not after or before != after or report.get("dataset_inputs_unchanged") is not True:
        raise ValueError("launch-gate dataset preservation check failed")


def _publish_launch_gate_evaluation(
    destination: Path,
    report: dict[str, Any],
    *,
    config: dict[str, Any],
    cuda: bool,
) -> dict[str, Any]:
    """Evaluate post-workload criteria and always publish their outcome atomically."""
    try:
        _validate_launch_gate_criteria(report, config=config, cuda=cuda)
    except BaseException as error:
        report.update(
            status="failed",
            failure_reason=str(error),
            error_type=type(error).__name__,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        _atomic_json(destination, report)
        raise
    report.update(status="passed", failure_reason=None, completed_utc=datetime.now(UTC).isoformat())
    _atomic_json(destination, report)
    return report


def verify_launch_gate_report(path: str | Path, *, config: dict[str, Any], dataset_sha256: str) -> dict[str, Any]:
    """Verify that a real N=500 gate attests this exact config and dataset."""
    report = json.loads(Path(path).read_text())
    expected = {
        "status": "passed",
        "architecture_version": LARGE_ARCHITECTURE_VERSION,
        "config_sha256": _configuration_sha256(config),
        "dataset_sha256": dataset_sha256,
        "requested_length": 500,
        "actual_length": 500,
        "optimizer_step_completed": True,
        "dataset_inputs_unchanged": True,
    }
    contradictions = [key for key, value in expected.items() if report.get(key) != value]
    if contradictions:
        raise ValueError(f"E005-Large launch gate contradiction: {', '.join(contradictions)}")
    _validate_launch_gate_criteria(report, config=config, cuda=True)
    return report


def run_large_launch_gate(
    config: dict[str, Any],
    *,
    report_path: str | Path,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Run exactly one full-width E005-Large update and atomically attest it."""
    validate_large_config(config, synthetic=synthetic)
    destination = Path(report_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite launch-gate report: {destination}")
    identity = _dataset_identity(config, synthetic=synthetic)
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E005-Large launch gate requires available CUDA")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    context: dict[str, Any] = {
        "status": "running",
        "failure_reason": None,
        "architecture_version": LARGE_ARCHITECTURE_VERSION,
        "config_sha256": _configuration_sha256(config),
        "dataset_sha256": identity["sha256"],
        "dataset_before_sha256": identity["sha256"],
        "dataset_after_sha256": None,
        "dataset_inputs_unchanged": False,
        "device": str(device),
        "requested_length": 500,
        "actual_length": None,
        "padded_length": None,
        "sample_id": None,
        "optimizer_step_completed": False,
        "losses": None,
        "gradient_checks": {
            "norms": None,
            "all_active_parameter_groups_nonzero": False,
        },
        "gate_checks": {"statistics": None, "valid_nonsaturated": False},
        "memory": None,
    }
    try:
        if synthetic:
            item = _synthetic_items([500], int(config["seed"]))[0]
            sample_id = item["sample_id"]
        else:
            selected, _ = _select_nearest_real_row(config, 500, int(config["seed"]) + 500)
            row = selected.to_pylist()[0]
            if int(row["sequence_length"]) != 500:
                raise ValueError("E005-Large launch gate requires an eligible sample of exact length 500")
            item = _item_for_row(config, row, synthetic=False, validation=False)
            sample_id = row["sample_id"]
        model = _model(config).to(device)
        model.train()
        counts = large_parameter_counts(model)
        if counts != LARGE_PARAMETER_COUNTS:
            raise RuntimeError(f"E005-Large parameter-count contradiction: {counts}")
        batch = _prepare_batch(config, model, item, device)
        actual_length = int(batch["lengths"].item())
        if actual_length != 500:
            raise ValueError(f"E005-Large launch-gate sample length is {actual_length}, expected 500")
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["diffusion"]["steps"]))).to(device)
        losses, _, gates = _forward_loss(
            config,
            model,
            diffusion,
            batch,
            mode=TRAINING_MODE,
            stochastic_seed=int(config["seed"]) + 5_000_005,
            dropout_probability=0.0,
            activation_checkpointing=True,
            amp_enabled=device.type == "cuda",
            amp_dtype=torch.float16,
        )
        context["losses"] = {key: float(value.detach()) for key, value in losses.items()}
        context["gate_checks"] = {"statistics": gates, "valid_nonsaturated": _gate_passes(gates)}
        if not all(torch.isfinite(value) for value in losses.values()):
            raise FloatingPointError("nonfinite_launch_gate_loss")
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        gradients = _group_gradient_metrics(_parameter_groups(model), TRAINING_MODE)
        active_nonzero = all(float(gradients[name] or 0) > 0 for name in _active_parameter_groups(TRAINING_MODE))
        context["gradient_checks"] = {
            "norms": gradients,
            "all_active_parameter_groups_nonzero": active_nonzero,
        }
        if not active_nonzero:
            raise RuntimeError("missing_or_zero_active_group_gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimizer_config["gradient_clip_norm"]))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise RuntimeError("launch_gate_amp_step_skipped")
        context["optimizer_step_completed"] = True
        _assert_finite_parameters(model)
        memory = _memory(device)
        context["memory"] = memory
        identity_after = _dataset_identity(config, synthetic=synthetic)
        context.update(
            dataset_after_sha256=identity_after["sha256"],
            dataset_inputs_unchanged=identity_after == identity,
            actual_length=actual_length,
            padded_length=int(batch["clean"].shape[-1]),
            sample_id=sample_id,
            parameter_counts=counts,
            elapsed_seconds=time.monotonic() - started,
        )
        return _publish_launch_gate_evaluation(destination, context, config=config, cuda=device.type == "cuda")
    except BaseException as error:
        if context["memory"] is None:
            context["memory"] = _memory(device)
        if context["dataset_after_sha256"] is None:
            try:
                identity_after = _dataset_identity(config, synthetic=synthetic)
                context["dataset_after_sha256"] = identity_after["sha256"]
                context["dataset_inputs_unchanged"] = identity_after == identity
            except BaseException as identity_error:
                context["dataset_after_error"] = f"{type(identity_error).__name__}: {identity_error}"
        context.update(
            status="failed",
            failure_reason=str(error),
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
            completed_utc=datetime.now(UTC).isoformat(),
        )
        _atomic_json(destination, context)
        raise


def _heartbeat(path: Path, status: str, **values: Any) -> None:
    _atomic_json(path, {"status": status, "updated_utc": datetime.now(UTC).isoformat(), **values})


def _save_training_checkpoint(
    path: Path,
    *,
    config_hash: str,
    dataset_hash: str,
    plan_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    optimizer_step: int,
    microstep: int,
    stage: int,
    accumulated_pair_tokens: int,
    metrics_offset: int,
    validation_offset: int,
    completed_validations: list[int],
    best_validation: float,
) -> None:
    save_checkpoint(
        path,
        {
            "version": "e005_large_checkpoint_v1",
            "architecture_version": LARGE_ARCHITECTURE_VERSION,
            "base_architecture_version": E005_ARCHITECTURE_VERSION,
            "config_sha256": config_hash,
            "dataset_sha256": dataset_hash,
            "plan_sha256": plan_hash,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "rng_state": _rng_state(),
            "sampler_cursor": optimizer_step,
            "curriculum_stage": stage,
            "microstep": microstep,
            "optimizer_step": optimizer_step,
            "accumulated_pair_token_count": accumulated_pair_tokens,
            "gradients": _gradient_state(model),
            "metrics_offset": metrics_offset,
            "validation_offset": validation_offset,
            "completed_validations": completed_validations,
            "best_validation": best_validation,
        },
    )


def _validation_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"record_count": len(records), "by_mode": {}, "strata": []}
    for mode in VALIDATION_MODES:
        values = [record for record in records if record["mode"] == mode]
        macro = {
            key: float(np.mean([record[key] for record in values]))
            for key in ("sequence_loss", "sequence_accuracy", "geometry_loss", "consistency_loss")
        }
        macro["sequence_perplexity"] = float(math.exp(min(macro["sequence_loss"], 27.0)))
        token_total = sum(record["token_count"] for record in values)
        pair_total = sum(record["pair_count"] for record in values)
        result["by_mode"][mode] = {
            "macro": macro,
            "token_weighted": {
                "sequence_cross_entropy": sum(record["sequence_loss"] * record["token_count"] for record in values)
                / token_total,
                "sequence_accuracy": sum(record["sequence_accuracy"] * record["token_count"] for record in values)
                / token_total,
            },
            "pair_weighted": {
                key: sum(record[key] * record["pair_count"] for record in values) / pair_total
                for key in ("geometry_loss", "consistency_loss")
            },
            "gate_statistics": {
                gate: {
                    statistic: float(np.mean([record["gates"][gate][statistic] for record in values]))
                    for statistic in ("mean", "standard_deviation", "saturated_fraction")
                }
                for gate in (
                    "geometry_to_sequence",
                    "sequence_to_geometry",
                    "return_geometry_to_sequence",
                )
            },
        }
    learned = result["by_mode"][TRAINING_MODE]["macro"]
    result["paired_mode_differences"] = {
        mode: {
            key: result["by_mode"][mode]["macro"][key] - learned[key]
            for key in ("sequence_loss", "sequence_accuracy", "geometry_loss", "consistency_loss")
        }
        for mode in VALIDATION_MODES
        if mode != TRAINING_MODE
    }
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record["mode"],
                record["length_bin"],
                record["experimental_method"],
                record["pairing_classification"],
            )
        ].append(record)
    result["strata"] = [
        {
            "mode": key[0],
            "length_bin": key[1],
            "experimental_method": key[2],
            "pairing_classification": key[3],
            "record_count": len(values),
            "macro": {
                metric: float(np.mean([record[metric] for record in values]))
                for metric in ("sequence_loss", "sequence_accuracy", "geometry_loss", "consistency_loss")
            },
        }
        for key, values in sorted(grouped.items())
    ]
    return result


def evaluate_large(
    config: dict[str, Any],
    model: torch.nn.Module,
    plan: dict[str, Any],
    *,
    device: torch.device,
    synthetic: bool,
    optimizer_step: int,
) -> dict[str, Any]:
    """Evaluate the fixed panel in all modes and deterministic corruption seeds."""
    model.eval()
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["diffusion"]["steps"]))).to(device)
    records = []
    seed_count = int(config["production"]["validation_corruption_seeds"])
    with torch.no_grad():
        for row_index, row in enumerate(plan["validation_rows"]):
            item = _item_for_row(config, row, synthetic=synthetic, validation=True)
            batch = _prepare_batch(config, model, item, device)
            length = int(batch["lengths"].item())
            for seed_index in range(seed_count):
                stochastic_seed = int(config["seed"]) + 90_000_001 + row_index * seed_count + seed_index
                for mode in VALIDATION_MODES:
                    losses, diagnostics, gates = _forward_loss(
                        config,
                        model,
                        diffusion,
                        batch,
                        mode=mode,
                        stochastic_seed=stochastic_seed,
                        dropout_probability=0.0,
                        activation_checkpointing=False,
                        amp_enabled=bool(config["mixed_precision"] and device.type == "cuda"),
                        amp_dtype=torch.float16,
                    )
                    records.append(
                        {
                            "sample_id": row["sample_id"],
                            "mode": mode,
                            "seed_index": seed_index,
                            "length": length,
                            "length_bin": row["length_bin"],
                            "experimental_method": row["experimental_method"],
                            "pairing_classification": row["pairing_classification"],
                            "token_count": diagnostics["masked_token_count"],
                            "pair_count": diagnostics["valid_pair_count"],
                            "sequence_loss": float(losses["sequence"]),
                            "sequence_accuracy": diagnostics["sequence_accuracy"],
                            "geometry_loss": float(losses["geometry"]),
                            "consistency_loss": float(losses["consistency"]),
                            "gates": gates,
                        }
                    )
    model.train()
    return {"optimizer_step": optimizer_step, "records": records, **_validation_aggregate(records)}


def _training_aggregate(path: Path) -> list[dict[str, Any]]:
    aggregates: dict[tuple[int, str, str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["record_type"] != "microbatch":
                continue
            key = (
                int(record["stage"]),
                record["length_bin"],
                record["experimental_method"],
                record["pairing_classification"],
            )
            aggregate = aggregates[key]
            aggregate["samples"] += 1
            aggregate["pair_tokens"] += int(record["pair_tokens"])
            aggregate["dropout"] += int(record["conditioning_dropped"])
            aggregate["accuracy"] += float(record["sequence_accuracy"])
            for name, value in record["losses"].items():
                aggregate[f"loss_{name}"] += float(value)
    output = []
    for key, values in sorted(aggregates.items()):
        count = int(values["samples"])
        output.append(
            {
                "arm": TRAINING_MODE,
                "stage": key[0],
                "length_bin": key[1],
                "experimental_method": key[2],
                "pairing_classification": key[3],
                "samples": count,
                "pair_tokens": int(values["pair_tokens"]),
                "mean_losses": {
                    name: values[f"loss_{name}"] / count for name in ("total", "sequence", "geometry", "consistency")
                },
                "mean_sequence_accuracy": values["accuracy"] / count,
                "conditioning_dropout_frequency": values["dropout"] / count,
            }
        )
    return output


def run_large_training(
    config: dict[str, Any],
    *,
    launch_gate_report: str | Path,
    output_dir: str | Path | None = None,
    resume: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Run the gated, resumable, single-arm E005-Large production schedule."""
    validate_large_config(config, synthetic=synthetic)
    destination = Path(output_dir or config["output_dir"])
    summary_path = destination / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
        raise FileExistsError(f"Refusing to overwrite completed E005-Large run: {destination}")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"E005-Large output is non-empty; use --resume: {destination}")
    heartbeat_path = destination / "heartbeat.json"
    identity = _dataset_identity(config, synthetic=synthetic)
    verify_launch_gate_report(launch_gate_report, config=config, dataset_sha256=identity["sha256"])
    destination.mkdir(parents=True, exist_ok=True)
    config_hash = _configuration_sha256(config)
    plan_path = destination / "training_plan.json"
    if resume:
        plan = json.loads(plan_path.read_text())
    else:
        plan = _large_plan(config, synthetic=synthetic)
        _atomic_json(plan_path, plan)
    if plan["sha256"] != _configuration_sha256({key: value for key, value in plan.items() if key != "sha256"}):
        raise ValueError("E005-Large training plan hash contradiction")
    training_dataset = None
    if not synthetic:
        training_dataset = SequenceGeometryDataset(
            Path(config["dataset"]["directory"]) / str(config["dataset"]["train_dataset"]),
            mode="geometry_conditioned",
            seed=int(config["seed"]),
            include_metadata=True,
        )
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E005-Large requires available CUDA")
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    model = _model(config).to(device)
    counts = large_parameter_counts(model)
    if not synthetic and counts != LARGE_PARAMETER_COUNTS:
        raise RuntimeError(f"E005-Large parameter-count contradiction: {counts}")
    groups = _parameter_groups(model)
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    stages = tuple(int(item["optimizer_updates"]) for item in config["production"]["curriculum"])
    total_updates = sum(stages)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_multiplier(
            step,
            total_updates=total_updates,
            warmup_updates=int(optimizer_config["warmup_updates"]),
        ),
    )
    amp_enabled = bool(config["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["diffusion"]["steps"]))).to(device)
    latest = destination / "checkpoints" / "latest.pt"
    metrics_path = destination / "metrics.jsonl.inprogress"
    validation_path = destination / "validation.jsonl.inprogress"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = microstep = accumulated_pairs = 0
    completed_validations: list[int] = []
    best_validation = math.inf
    skipped_updates = 0
    consecutive_skipped_updates = 0
    if resume:
        saved = load_checkpoint(latest, map_location=device)
        validate_resume_checkpoint(
            saved,
            config_hash=config_hash,
            dataset_hash=identity["sha256"],
            plan_hash=plan["sha256"],
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["grad_scaler"])
        _restore_rng_state(saved.get("rng_state"))
        _restore_gradients(model, saved.get("gradients", {}), device)
        step = int(saved["optimizer_step"])
        microstep = int(saved["microstep"])
        accumulated_pairs = int(saved["accumulated_pair_token_count"])
        completed_validations = list(saved.get("completed_validations", []))
        best_validation = float(saved.get("best_validation", math.inf))
        for path, offset_name in (
            (metrics_path, "metrics_offset"),
            (validation_path, "validation_offset"),
        ):
            with path.open("ab") as handle:
                handle.truncate(int(saved.get(offset_name, 0)))
    due_validation = set(validation_steps(stages, int(config["production"]["validation_frequency"])))
    interrupted = [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: interrupted.__setitem__(0, True))
    started = time.monotonic()
    try:
        if step in due_validation and step not in completed_validations:
            result = evaluate_large(config, model, plan, device=device, synthetic=synthetic, optimizer_step=step)
            with validation_path.open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
            completed_validations.append(step)
            best_validation = float(result["by_mode"][TRAINING_MODE]["macro"]["sequence_loss"])
            stage_index, _ = _stage_for_step(config, step)
            _save_training_checkpoint(
                latest,
                config_hash=config_hash,
                dataset_hash=identity["sha256"],
                plan_hash=plan["sha256"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=step,
                microstep=microstep,
                stage=stage_index + 1,
                accumulated_pair_tokens=accumulated_pairs,
                metrics_offset=metrics_path.stat().st_size if metrics_path.exists() else 0,
                validation_offset=validation_path.stat().st_size,
                completed_validations=completed_validations,
                best_validation=best_validation,
            )
            save_checkpoint(destination / "checkpoints" / "best.pt", load_checkpoint(latest))
        while step < total_updates:
            stage_index, _ = _stage_for_step(config, step)
            microbatches = (
                _microbatch_rows(config, plan, step)
                if synthetic
                else _full_corpus_microbatch_rows(config, training_dataset, step)
            )
            if microstep == 0:
                optimizer.zero_grad(set_to_none=True)
                accumulated_pairs = 0
            conditioning_present = False
            for microbatch_index, (row, stochastic_seed) in enumerate(microbatches):
                if microbatch_index < microstep:
                    continue
                if interrupted[0]:
                    raise PilotInterrupted("SIGINT requested")
                item = _item_for_row(config, row, synthetic=synthetic, validation=False)
                batch = _prepare_batch(config, model, item, device)
                losses, diagnostics, gates = _forward_loss(
                    config,
                    model,
                    diffusion,
                    batch,
                    mode=TRAINING_MODE,
                    stochastic_seed=stochastic_seed,
                    dropout_probability=float(config["conditioning"]["dropout_probability"]),
                    activation_checkpointing=True,
                    amp_enabled=amp_enabled,
                    amp_dtype=torch.float16,
                )
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("nonfinite_loss")
                scaler.scale(losses["total"] / len(microbatches)).backward()
                length = int(batch["lengths"].item())
                pair_tokens = length * length
                accumulated_pairs += pair_tokens
                conditioning_present = conditioning_present or not diagnostics["conditioning_dropped"]
                record = {
                    "record_type": "microbatch",
                    "optimizer_step": step,
                    "microstep": microbatch_index,
                    "stage": stage_index + 1,
                    "sample_id": row["sample_id"],
                    "length_bin": row["length_bin"],
                    "experimental_method": row["experimental_method"],
                    "pairing_classification": row["pairing_classification"],
                    "losses": {key: float(value.detach()) for key, value in losses.items()},
                    "sequence_accuracy": diagnostics["sequence_accuracy"],
                    "conditioning_dropped": diagnostics["conditioning_dropped"],
                    "gate_statistics": gates,
                    "pair_tokens": pair_tokens,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                microstep = microbatch_index + 1
            scaler.unscale_(optimizer)
            scale_before = scaler.get_scale()
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
            )
            if not gradients_finite and not amp_enabled:
                raise FloatingPointError("nonfinite_gradient")
            if gradients_finite:
                gradient_norms = _group_gradient_metrics(groups, TRAINING_MODE)
                always_required = {"sequence_branch", "geometry_branch"}
                feedback_required = _active_parameter_groups(TRAINING_MODE) - always_required
                required = always_required | (feedback_required if conditioning_present else set())
                if any(float(gradient_norms[name] or 0) <= 0 for name in required):
                    raise RuntimeError("missing_or_zero_active_group_gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimizer_config["gradient_clip_norm"]))
            else:
                gradient_norms = {name: None for name in groups}
            scaler.step(optimizer)
            scaler.update()
            skipped = not gradients_finite or scaler.get_scale() < scale_before
            if skipped:
                skipped_updates += 1
                consecutive_skipped_updates += 1
                if consecutive_skipped_updates > int(config["production"]["maximum_consecutive_skipped_updates"]):
                    raise FloatingPointError("maximum_consecutive_amp_skipped_updates_exceeded")
            else:
                consecutive_skipped_updates = 0
                step += 1
                scheduler.step()
                _assert_finite_parameters(model)
            parameter_changes = (
                {name: 0.0 for name in groups} if skipped else _adamw_parameter_change_norms(optimizer, groups)
            )
            microstep = 0
            memory = _memory(device)
            enforce_memory_limits(
                memory,
                max_rss_mib=int(config["production"]["max_rss_mib"]),
                max_cuda_memory_mib=int(config["production"]["max_cuda_memory_mib"]),
                cuda=device.type == "cuda",
            )
            with metrics_path.open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "record_type": "optimizer_update",
                            "optimizer_step": step,
                            "stage": stage_index + 1,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "gradient_norms": gradient_norms,
                            "parameter_change_norms": parameter_changes,
                            "amp_skipped": skipped,
                            "accumulated_pair_tokens": accumulated_pairs,
                            "samples": len(microbatches),
                            "memory": memory,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            latest_due, permanent_due = checkpoint_kind(step, stages, int(config["production"]["checkpoint_frequency"]))
            if latest_due or permanent_due:
                _save_training_checkpoint(
                    latest,
                    config_hash=config_hash,
                    dataset_hash=identity["sha256"],
                    plan_hash=plan["sha256"],
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    optimizer_step=step,
                    microstep=microstep,
                    stage=stage_index + 1,
                    accumulated_pair_tokens=accumulated_pairs,
                    metrics_offset=metrics_path.stat().st_size,
                    validation_offset=validation_path.stat().st_size if validation_path.exists() else 0,
                    completed_validations=completed_validations,
                    best_validation=best_validation,
                )
                if permanent_due:
                    permanent = destination / "checkpoints" / f"stage_{stage_index + 1}_step_{step}.pt"
                    save_checkpoint(permanent, load_checkpoint(latest))
            if step in due_validation and step not in completed_validations:
                identity_during = _dataset_identity(config, synthetic=synthetic)
                if identity_during != identity:
                    raise RuntimeError("dataset_mutation_detected")
                result = evaluate_large(config, model, plan, device=device, synthetic=synthetic, optimizer_step=step)
                with validation_path.open("a") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
                completed_validations.append(step)
                criterion = float(result["by_mode"][TRAINING_MODE]["macro"]["sequence_loss"])
                if criterion < best_validation:
                    best_validation = criterion
                    is_best = True
                else:
                    is_best = False
                _save_training_checkpoint(
                    latest,
                    config_hash=config_hash,
                    dataset_hash=identity["sha256"],
                    plan_hash=plan["sha256"],
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    optimizer_step=step,
                    microstep=microstep,
                    stage=stage_index + 1,
                    accumulated_pair_tokens=accumulated_pairs,
                    metrics_offset=metrics_path.stat().st_size,
                    validation_offset=validation_path.stat().st_size,
                    completed_validations=completed_validations,
                    best_validation=best_validation,
                )
                if is_best:
                    save_checkpoint(destination / "checkpoints" / "best.pt", load_checkpoint(latest))
            _heartbeat(
                heartbeat_path,
                "running",
                optimizer_step=step,
                curriculum_stage=stage_index + 1,
                memory=memory,
                skipped_updates=skipped_updates,
            )
        identity_before_publication = _dataset_identity(config, synthetic=synthetic)
        if identity_before_publication != identity:
            raise RuntimeError("dataset_mutation_detected")
        final_checkpoint = destination / "checkpoints" / "final.pt"
        save_checkpoint(final_checkpoint, load_checkpoint(latest))
        metrics_final = destination / "metrics.jsonl"
        validation_final = destination / "validation.jsonl"
        metrics_path.replace(metrics_final)
        validation_path.replace(validation_final)
        identity_after = _dataset_identity(config, synthetic=synthetic)
        if identity_after != identity:
            raise RuntimeError("dataset_mutation_detected")
        summary = {
            "status": "completed",
            "architecture_version": LARGE_ARCHITECTURE_VERSION,
            "training_mode": TRAINING_MODE,
            "optimizer_steps": step,
            "parameter_counts": counts,
            "dataset_identity_before": identity,
            "dataset_identity_after": identity_after,
            "dataset_inputs_unchanged": True,
            "validation_steps": completed_validations,
            "best_validation_criterion": config["validation"]["selection_criterion"],
            "best_validation_value": best_validation,
            "amp_skipped_steps": skipped_updates,
            "stratified_training_metrics": _training_aggregate(metrics_final),
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": _sha256_file(final_checkpoint),
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(summary_path, summary)
        _heartbeat(
            heartbeat_path,
            "completed",
            optimizer_step=step,
            curriculum_stage=len(stages),
            summary_path=str(summary_path),
            summary_sha256=_sha256_file(summary_path),
        )
        return summary
    except BaseException as error:
        status = (
            "interrupted"
            if isinstance(error, PilotInterrupted)
            else ("memory_limit_exceeded" if isinstance(error, MemoryError) else "failed")
        )
        if model is not None:
            stage_index, _ = _stage_for_step(config, min(step, total_updates - 1))
            _save_training_checkpoint(
                latest,
                config_hash=config_hash,
                dataset_hash=identity["sha256"],
                plan_hash=plan["sha256"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                optimizer_step=step,
                microstep=microstep,
                stage=stage_index + 1,
                accumulated_pair_tokens=accumulated_pairs,
                metrics_offset=metrics_path.stat().st_size if metrics_path.exists() else 0,
                validation_offset=validation_path.stat().st_size if validation_path.exists() else 0,
                completed_validations=completed_validations,
                best_validation=best_validation,
            )
        _heartbeat(
            heartbeat_path,
            status,
            error_type=type(error).__name__,
            error=str(error),
            latest_resumable_checkpoint=str(latest),
            optimizer_step=step,
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)
