"""Monitored, review-gated production support for E006 Stage-A v5."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.stage_a_context import (
    CANONICAL_TOKEN_COUNT,
    CANONICAL_TOKEN_START,
    canonical_corrupted_cross_entropy,
    context_corruption,
)

MONITOR_VERSION = "e006_stage_a_context_monitor_v1"
MARGIN_SIGN_CONVENTION = "normal_ce_minus_counterfactual_ce; more negative favors ordered context"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_v5_pretraining_gates(config: dict[str, Any]) -> dict[str, Any]:
    """Verify the three non-authorizing artifacts required before v5 training."""
    required = ("synthetic_context_smoke", "real_loader_smoke", "comparison_pilot")
    specifications = config.get("gate_artifacts", {})
    if set(specifications) != set(required):
        raise ValueError(f"E006 Stage-A v5 startup gates must be exactly {required}")
    verified = {}
    for name in required:
        record = specifications[name]
        path = Path(record.get("path", ""))
        expected = record.get("sha256")
        if not expected or not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"E006 Stage-A v5 launch-gate hash contradiction: {name}")
        payload = json.loads(path.read_text())
        if payload.get("status") not in set(record.get("acceptable_statuses", ["completed"])):
            raise ValueError(f"E006 Stage-A v5 launch gate did not pass: {name}")
        if payload.get("authorizes_training") is not False or payload.get("authorizes_joint_training") is not False:
            raise ValueError(f"E006 Stage-A v5 prerequisite must be non-authorizing: {name}")
        if name != "comparison_pilot" and payload.get("gates", {}).get("passed") is not True:
            raise ValueError(f"E006 Stage-A v5 smoke gate did not pass: {name}")
        if name == "comparison_pilot":
            recommendation = payload.get("recommendation", {}).get("recommendation")
            if recommendation != "v4_warm_start":
                raise ValueError("E006 Stage-A v5 comparison pilot did not recommend v4_warm_start")
        verified[name] = {"path": str(path), "sha256": expected, "status": payload["status"]}
    return {
        "required": True,
        "passed": True,
        "artifacts": verified,
        "post_training_context_diagnostic_required": True,
        "authorizes_joint_training": False,
    }


def load_warm_start_weights_only(model: torch.nn.Module, initialization: dict[str, Any]) -> dict[str, Any]:
    if initialization.get("mode") != "checkpoint_weights_only":
        raise ValueError("E006 Stage-A v5 production requires checkpoint_weights_only initialization")
    path = Path(initialization.get("checkpoint_path", ""))
    expected = initialization.get("checkpoint_sha256")
    if not expected or not path.is_file() or _sha256(path) != expected:
        raise ValueError("E006 Stage-A v5 warm-start checkpoint hash contradiction")
    payload = load_checkpoint(path, map_location="cpu")
    model.load_state_dict(payload["model"])
    return {
        "mode": "checkpoint_weights_only",
        "checkpoint_path": str(path),
        "checkpoint_sha256": expected,
        "source_optimizer_step_ignored": int(payload.get("optimizer_step", -1)),
        "model_weights_loaded": True,
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
        "rng_restored": False,
        "cursor_restored": False,
        "training_progress_restored": False,
    }


def contextual_monitoring_steps(total_updates: int, pass_end_steps: Sequence[int]) -> list[int]:
    if total_updates < 1:
        raise ValueError("E006 contextual monitoring requires positive total updates")
    scheduled = {0, 250, 500, 1000, total_updates, *range(1500, total_updates + 1, 500), *pass_end_steps}
    return sorted(step for step in scheduled if 0 <= step <= total_updates)


def select_monitoring_panel(
    dataset: Any,
    *,
    count: int,
    seed: int,
    maximum_length: int,
    excluded_sample_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = []
    for index, sample_id, length in dataset.iter_metadata():
        if 0 < int(length) <= maximum_length and sample_id not in excluded_sample_ids:
            rank = hashlib.sha256(f"{seed}:context-monitor:validation:{sample_id}".encode()).hexdigest()
            candidates.append((rank, str(sample_id), int(index), int(length)))
    selected = sorted(candidates)[:count]
    if len(selected) != count:
        raise ValueError(f"E006 contextual monitoring panel has {len(selected)} rows, expected {count}")
    ids = [item[1] for item in selected]
    if len(set(ids)) != count or set(ids).intersection(excluded_sample_ids):
        raise ValueError("E006 contextual monitoring panel identity/disjointness contradiction")
    rows = [dataset[item[2]] for item in selected]
    return rows, {
        "version": MONITOR_VERSION,
        "sample_count": count,
        "unique_sample_count": len(set(ids)),
        "sample_ids": ids,
        "sample_id_sha256": _canonical_hash(ids),
        "seed": seed,
        "maximum_length": maximum_length,
        "disjoint_from_final_diagnostic_reservation": True,
        "excluded_final_diagnostic_sample_count": len(excluded_sample_ids),
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
    }


def training_unigram(dataset: Any, *, smoothing: float = 1.0) -> dict[str, Any]:
    counts = np.zeros(CANONICAL_TOKEN_COUNT, dtype=np.int64)
    samples = 0
    for _, _, token_ids in dataset.iter_token_ids():
        values = np.asarray(token_ids, dtype=np.int64) - CANONICAL_TOKEN_START
        if (values < 0).any() or (values >= CANONICAL_TOKEN_COUNT).any():
            raise ValueError("E006 contextual monitor unigram encountered noncanonical tokens")
        counts += np.bincount(values, minlength=CANONICAL_TOKEN_COUNT)
        samples += 1
    probabilities = (counts.astype(np.float64) + smoothing) / (counts.sum() + smoothing * CANONICAL_TOKEN_COUNT)
    return {
        "sample_count": samples,
        "token_count": int(counts.sum()),
        "token_counts": counts.tolist(),
        "probabilities": probabilities.tolist(),
        "smoothing": smoothing,
        "sha256": _canonical_hash(counts.tolist()),
    }


def _collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    side = max(len(row["sequence"]) for row in rows)
    tokens = torch.zeros((len(rows), side), dtype=torch.long)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row["token_ids"], dtype=torch.long)
        tokens[index, : len(values)] = values
        mask[index, : len(values)] = True
    return {"tokens": tokens, "mask": mask}


def classify_context_health(
    *,
    step: int,
    finite: bool,
    unigram_improvement: float,
    shuffle_margin: float,
    null_margin: float,
    grace_steps: int = 2500,
    minimum_improvement: float = 0.0,
    maximum_margin: float = 0.0,
    ce_change_from_initial: float = 0.0,
) -> str:
    if not finite:
        return "failing"
    contextual = (
        unigram_improvement > minimum_improvement
        and shuffle_margin < maximum_margin
        and null_margin < maximum_margin
        and ce_change_from_initial <= 0
    )
    if contextual:
        return "healthy"
    return "warning" if step < grace_steps else "failing"


def classify_context_health_v6(
    *,
    finite: bool,
    shuffle_margin: float,
    ce_change_from_initial: float,
    provisional_margin_threshold: float = -0.005,
) -> str:
    """Apply the conservative v6 monitor policy; this never verifies learning."""
    if not finite:
        return "failing"
    if shuffle_margin <= provisional_margin_threshold and ce_change_from_initial < 0:
        return "provisional_healthy"
    return "warning"


def evaluate_context_monitor(
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    *,
    step: int,
    seed: int,
    mask_fraction: float,
    unigram: dict[str, Any],
    device: torch.device,
    initial_normal_cross_entropy: float | None = None,
    health_policy: str = "v5",
) -> dict[str, Any]:
    totals = {name: {"nll": 0.0, "correct": 0, "tokens": 0} for name in ("normal", "shuffled", "null")}
    probabilities = torch.tensor(unigram["probabilities"], dtype=torch.float64)
    unigram_nll = 0.0
    model.eval()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(rows), 32)):
            batch = _collate(rows[start : start + 32])
            targets = batch["tokens"].to(device)
            residue_mask = batch["mask"].to(device)
            corruption = context_corruption(
                targets,
                residue_mask,
                mask_token_id=1,
                probability=mask_fraction,
                seed=seed,
                step=batch_index,
            )
            null = torch.where(residue_mask, torch.ones_like(targets), targets)
            selected = corruption.corrupted_mask
            selected_targets = targets[selected]
            unigram_nll += float(-torch.log(probabilities[selected_targets.cpu() - CANONICAL_TOKEN_START]).sum())
            for name, inputs in (
                ("normal", corruption.inputs),
                ("shuffled", corruption.shuffled_inputs),
                ("null", null),
            ):
                logits = model.forward_sequence_pretraining(inputs, residue_mask)
                loss = canonical_corrupted_cross_entropy(logits, targets, selected, residue_mask)
                predictions = logits[selected, 2:22].argmax(-1) + 2
                count = int(selected.sum())
                totals[name]["nll"] += float(loss) * count
                totals[name]["correct"] += int((predictions == selected_targets).sum())
                totals[name]["tokens"] += count
    model.train()
    conditions = {
        name: {
            "canonical_cross_entropy": value["nll"] / value["tokens"],
            "top1_accuracy": value["correct"] / value["tokens"],
        }
        for name, value in totals.items()
    }
    count = totals["normal"]["tokens"]
    normal = conditions["normal"]["canonical_cross_entropy"]
    shuffle_margin = normal - conditions["shuffled"]["canonical_cross_entropy"]
    null_margin = normal - conditions["null"]["canonical_cross_entropy"]
    unigram_ce = unigram_nll / count
    finite = all(math.isfinite(item["canonical_cross_entropy"]) for item in conditions.values())
    ce_change = 0.0 if initial_normal_cross_entropy is None else normal - initial_normal_cross_entropy
    if health_policy == "v6":
        health = classify_context_health_v6(
            finite=finite,
            shuffle_margin=shuffle_margin,
            ce_change_from_initial=ce_change,
        )
    elif health_policy == "v5":
        health = classify_context_health(
            step=step,
            finite=finite,
            unigram_improvement=unigram_ce - normal,
            shuffle_margin=shuffle_margin,
            null_margin=null_margin,
            ce_change_from_initial=ce_change,
        )
    else:
        raise ValueError(f"Unknown E006 contextual health policy: {health_policy}")
    return {
        "record_type": "contextual_monitoring",
        "version": MONITOR_VERSION,
        "optimizer_step": step,
        "mask_fraction": mask_fraction,
        "canonical_corrupted_token_count": count,
        "conditions": conditions,
        "normal_minus_shuffled_ce": shuffle_margin,
        "normal_minus_null_ce": null_margin,
        "margin_sign_convention": MARGIN_SIGN_CONVENTION,
        "uniform_cross_entropy": math.log(CANONICAL_TOKEN_COUNT),
        "training_unigram_cross_entropy": unigram_ce,
        "improvement_over_training_unigram": unigram_ce - normal,
        "normal_ce_change_from_initial": ce_change,
        "finite": finite,
        "health": health,
        "health_policy": health_policy,
        "contextual_learning_verified": False,
        "constructs_rich_pair_features": False,
    }


def contextual_selection_key(record: dict[str, Any]) -> tuple[float, ...]:
    ranks = {"healthy": 0.0, "provisional_healthy": 0.5, "warning": 1.0, "failing": 2.0}
    return (
        ranks[record["health"]],
        float(record["normal_minus_shuffled_ce"]),
        float(record["normal_minus_null_ce"]),
        float(record["conditions"]["normal"]["canonical_cross_entropy"]),
    )


def append_rolling_training_metric(path: str | Path, record: dict[str, Any]) -> None:
    """Durably append compact scalar training evidence without logits or batches."""
    forbidden = {name for name in record if "logit" in name.lower() or name in {"batch", "inputs", "targets"}}
    if forbidden:
        raise ValueError(f"E006 rolling metric contains forbidden retained payloads: {sorted(forbidden)}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_review_decision(
    path: str | Path,
    expected_sha256: str,
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    optimizer_step: int,
) -> dict[str, Any]:
    decision_path = Path(path)
    if not expected_sha256 or not decision_path.is_file() or _sha256(decision_path) != expected_sha256:
        raise ValueError("E006 scientific-review decision hash contradiction")
    payload = json.loads(decision_path.read_text())
    expected = {
        "status": "completed",
        "decision": "approve_continue",
        "config_sha256": config_sha256,
        "review_checkpoint_sha256": checkpoint_sha256,
        "optimizer_step": optimizer_step,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    contradictions = [name for name, value in expected.items() if payload.get(name) != value]
    if contradictions:
        raise ValueError(f"E006 scientific-review decision contradiction: {contradictions}")
    return payload


def trajectory_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip() and json.loads(line).get("record_type") == "contextual_monitoring"
    ]


def format_trajectory(path: str | Path) -> str:
    header = "step normal_ce shuffled_ce null_ce shuffle_margin null_margin unigram_gain top1 health lr overflows"
    lines = [header]
    for row in trajectory_rows(path):
        conditions = row["conditions"]
        lines.append(
            f"{row['optimizer_step']} {conditions['normal']['canonical_cross_entropy']:.6f} "
            f"{conditions['shuffled']['canonical_cross_entropy']:.6f} "
            f"{conditions['null']['canonical_cross_entropy']:.6f} "
            f"{row['normal_minus_shuffled_ce']:.6f} {row['normal_minus_null_ce']:.6f} "
            f"{row['improvement_over_training_unigram']:.6f} {conditions['normal']['top1_accuracy']:.6f} "
            f"{row['health']} {row.get('learning_rate', float('nan')):.8g} "
            f"{row.get('amp_overflows_total', 0)}"
        )
    return "\n".join(lines)
