"""Bounded non-authorizing evidence harnesses for the E006 Stage-A v5 objective."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import queue as queue_module
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.evaluation.e006_stage_a_context import SequenceOnlyRichDataset
from protein_distance_diffusion.models.rich_codesign import E006RichGeometryCoDesign
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.rich_codesign_production import (
    _authorization,
    _model,
    _protected_hashes,
    _scheduler_multiplier,
    collate_sequence_pretraining,
    validate_phase3_config,
)
from protein_distance_diffusion.training.rich_codesign_smoke import _memory
from protein_distance_diffusion.training.stage_a_context import (
    canonical_corrupted_cross_entropy,
    context_corruption,
    contextual_stage_a_loss,
    visible_shuffle_inputs,
)

SYNTHETIC_SMOKE_VERSION = "e006_stage_a_context_v5_synthetic_smoke_v2"
LOADER_SMOKE_VERSION = "e006_stage_a_context_v5_loader_smoke_v1"
SEQUENCE_PARAMETER_PREFIXES = (
    "token_embedding.",
    "position_embedding.",
    "sequence_layers.",
    "sequence_norm.",
    "sequence_output.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def contextual_toy(split: str) -> dict[str, torch.Tensor | list[str]]:
    """Build paired contexts whose bag of visible residues cannot identify the target."""
    if split not in {"train", "held_out"}:
        raise ValueError(f"Unknown E006 contextual toy split: {split}")
    targets = (2, 3, 4, 5)
    outer = (10, 11, 12, 13) if split == "train" else (14, 15, 16, 17)
    rows = []
    identities = []
    for first in targets:
        for second in targets:
            if first == second:
                continue
            rows.append([outer[0], outer[1], first, first, second, outer[2], outer[3]])
            identities.append(f"{split}:{first}:{second}")
    values = torch.tensor(rows, dtype=torch.long)
    residue_mask = torch.ones_like(values, dtype=torch.bool)
    corrupted = torch.zeros_like(residue_mask)
    corrupted[:, 3] = True
    inputs = values.clone()
    inputs[corrupted] = 1
    return {
        "sample_ids": identities,
        "targets": values,
        "inputs": inputs,
        "residue_mask": residue_mask,
        "corrupted_mask": corrupted,
    }


def toy_contract() -> dict[str, Any]:
    train = contextual_toy("train")
    held_out = contextual_toy("held_out")
    train_targets = train["targets"][train["corrupted_mask"]].tolist()
    held_targets = held_out["targets"][held_out["corrupted_mask"]].tolist()
    train_marginal = Counter(train_targets)
    held_marginal = Counter(held_targets)
    train_bags = {
        tuple(sorted(row[mask].tolist())) for row, mask in zip(train["inputs"], train["residue_mask"], strict=True)
    }
    paired_ambiguous_bags = 0
    for bag in train_bags:
        outcomes = {
            int(target[corrupted].item())
            for row, target, corrupted in zip(train["inputs"], train["targets"], train["corrupted_mask"], strict=True)
            if tuple(sorted(row.tolist())) == bag
        }
        paired_ambiguous_bags += len(outcomes) > 1
    return {
        "train_target_counts": {str(key): value for key, value in sorted(train_marginal.items())},
        "held_out_target_counts": {str(key): value for key, value in sorted(held_marginal.items())},
        "marginals_equal": train_marginal == held_marginal and len(set(train_marginal.values())) == 1,
        "train_held_out_contexts_disjoint": not set(train["sample_ids"]) & set(held_out["sample_ids"]),
        "bag_ambiguous_context_count": paired_ambiguous_bags,
        "target_leakage_absent": bool((train["inputs"][train["corrupted_mask"]] == 1).all()),
    }


def _tiny_model(seed: int) -> E006RichGeometryCoDesign:
    torch.manual_seed(seed)
    return E006RichGeometryCoDesign(
        sequence_hidden_dim=32,
        sequence_layers=2,
        sequence_heads=4,
        sequence_feedforward_dim=64,
        sequence_dropout=0.0,
        max_length=16,
        rich_hidden_dim=16,
        fusion_layers=(0,),
        minimum_fusion_capacity_ratio=0.0,
        minimum_fusion_parameters=0,
        geometry_model={
            "base_channels": 8,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "group_norm_groups": 4,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 32,
            "length_embedding_dim": 32,
            "max_length": 16,
        },
    )


def _parameter_digest(parameters: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in parameters:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _condition_metrics(
    model: E006RichGeometryCoDesign,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    shuffle_seed: int,
) -> dict[str, Any]:
    targets = batch["targets"]
    residue_mask = batch["residue_mask"]
    corrupted = batch["corrupted_mask"]
    normal = batch["inputs"]
    shuffled = visible_shuffle_inputs(
        normal,
        corrupted,
        residue_mask,
        seed=shuffle_seed,
        step=0,
    )
    null = torch.where(residue_mask, torch.ones_like(targets), targets)
    result = {}
    with torch.inference_mode():
        for name, inputs in (("normal", normal), ("shuffled", shuffled), ("null", null)):
            logits = model.forward_sequence_pretraining(inputs, residue_mask)
            loss = canonical_corrupted_cross_entropy(logits, targets, corrupted, residue_mask)
            canonical = logits[corrupted, 2:22]
            predictions = canonical.argmax(-1) + 2
            result[name] = {
                "cross_entropy": float(loss),
                "top1_accuracy": float((predictions == targets[corrupted]).float().mean()),
                "corrupted_token_count": int(corrupted.sum()),
                "finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
            }
    counts = Counter(targets[corrupted].tolist())
    probabilities = torch.tensor([counts.get(token, 0) / int(corrupted.sum()) for token in range(2, 22)])
    target_indices = targets[corrupted] - 2
    unigram_ce = float(-probabilities[target_indices].clamp_min(1e-12).log().mean())
    result["uniform"] = {"cross_entropy": math.log(20.0), "corrupted_token_count": int(corrupted.sum())}
    result["unigram"] = {"cross_entropy": unigram_ce, "corrupted_token_count": int(corrupted.sum())}
    result["normal_to_shuffled_margin"] = result["shuffled"]["cross_entropy"] - result["normal"]["cross_entropy"]
    result["normal_to_null_margin"] = result["null"]["cross_entropy"] - result["normal"]["cross_entropy"]
    return result


def _padding_error(model: E006RichGeometryCoDesign, batch: dict[str, Any]) -> float:
    inputs = batch["inputs"]
    mask = batch["residue_mask"]
    with torch.inference_mode():
        reference = model.forward_sequence_pretraining(inputs, mask)
        padded_inputs = torch.nn.functional.pad(inputs, (0, 3))
        padded_mask = torch.nn.functional.pad(mask, (0, 3))
        padded = model.forward_sequence_pretraining(padded_inputs, padded_mask)
    return float((reference - padded[:, : inputs.shape[1]]).abs().max())


def plan_synthetic_smoke(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("version") != SYNTHETIC_SMOKE_VERSION:
        raise ValueError("E006 Stage-A synthetic-smoke-v2 version contradiction")
    contract = toy_contract()
    if not all(
        (
            contract["marginals_equal"],
            contract["train_held_out_contexts_disjoint"],
            contract["target_leakage_absent"],
        )
    ):
        raise ValueError("E006 contextual toy contract is invalid")
    return {
        "status": "planned",
        "version": SYNTHETIC_SMOKE_VERSION,
        "toy_contract": contract,
        "seeds": [int(value) for value in config["seeds"]],
        "optimizer_updates_per_seed": int(config["optimizer_updates_per_seed"]),
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
        "authorizes_training": False,
        "output_dir": config["output_dir"],
    }


def run_synthetic_smoke(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    plan = plan_synthetic_smoke(config_path)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 Stage-A synthetic smoke output exists: {output}")
    train = contextual_toy("train")
    held_out = contextual_toy("held_out")
    runs = []
    for seed in plan["seeds"]:
        model = _tiny_model(seed)
        sequence_parameters = [
            parameter for name, parameter in model.named_parameters() if name.startswith(SEQUENCE_PARAMETER_PREFIXES)
        ]
        geometry_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(SEQUENCE_PARAMETER_PREFIXES)
        ]
        sequence_before = _parameter_digest(sequence_parameters)
        geometry_before = _parameter_digest(geometry_parameters)
        initial_train = _condition_metrics(model.eval(), train, shuffle_seed=seed + 1)
        initial_held_out = _condition_metrics(model, held_out, shuffle_seed=seed + 2)
        optimizer = torch.optim.Adam(sequence_parameters, lr=float(config["learning_rate"]))
        components = []
        finite_gradients = True
        active_penalties = 0
        model.train()
        for step in range(int(config["optimizer_updates_per_seed"])):
            shuffled = visible_shuffle_inputs(
                train["inputs"], train["corrupted_mask"], train["residue_mask"], seed=seed, step=step
            )
            normal_logits = model.forward_sequence_pretraining(train["inputs"], train["residue_mask"])
            with torch.no_grad():
                shuffled_logits = model.forward_sequence_pretraining(shuffled, train["residue_mask"])
            losses = contextual_stage_a_loss(
                normal_logits,
                shuffled_logits,
                train["targets"],
                train["corrupted_mask"],
                train["residue_mask"],
                contrast_weight=float(config["context_contrast_weight"]),
                contrast_margin_nats=float(config["required_context_margin_nats"]),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            finite_gradients &= all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in sequence_parameters
            )
            optimizer.step()
            active_penalties += float(losses["context_contrast"]) > 0
            components.append(
                {
                    "canonical_corrupted_position_ce": float(losses["sequence"].detach()),
                    "shuffle_margin_penalty": float(losses["context_contrast_weighted"].detach()),
                    "total_objective": float(losses["total"].detach()),
                }
            )
        final_train = _condition_metrics(model.eval(), train, shuffle_seed=seed + 1)
        final_held_out = _condition_metrics(model, held_out, shuffle_seed=seed + 2)
        runs.append(
            {
                "seed": seed,
                "initial": {"train": initial_train, "held_out": initial_held_out},
                "final": {"train": final_train, "held_out": final_held_out},
                "objective_components_initial": components[0],
                "objective_components_final": components[-1],
                "active_margin_penalty_batch_fraction": active_penalties / len(components),
                "finite_losses": all(math.isfinite(value) for record in components for value in record.values()),
                "finite_gradients": finite_gradients,
                "sequence_parameters_changed": _parameter_digest(sequence_parameters) != sequence_before,
                "geometry_and_fusion_parameters_unchanged": _parameter_digest(geometry_parameters) == geometry_before,
                "padding_max_absolute_error": _padding_error(model, held_out),
            }
        )
    required_margin = float(config["required_context_margin_nats"])
    minimum_unigram_improvement = float(config["minimum_unigram_improvement_nats"])
    shuffle_margins = [item["final"]["held_out"]["normal_to_shuffled_margin"] for item in runs]
    null_margins = [item["final"]["held_out"]["normal_to_null_margin"] for item in runs]
    unigram_improvements = [
        item["final"]["held_out"]["unigram"]["cross_entropy"] - item["final"]["held_out"]["normal"]["cross_entropy"]
        for item in runs
    ]
    gates = {
        "finite_losses_and_gradients": all(item["finite_losses"] and item["finite_gradients"] for item in runs),
        "held_out_normal_beats_unigram": min(unigram_improvements) >= minimum_unigram_improvement,
        "held_out_shuffle_margin": min(shuffle_margins) >= required_margin,
        "held_out_null_margin": min(null_margins) >= required_margin,
        "repeated_seed_shuffle_effect_excludes_zero": min(shuffle_margins) > 0,
        "repeated_seed_null_effect_excludes_zero": min(null_margins) > 0,
        "target_leakage_absent": plan["toy_contract"]["target_leakage_absent"],
        "padding_invariant": max(item["padding_max_absolute_error"] for item in runs)
        <= float(config["padding_tolerance"]),
        "sequence_parameters_changed": all(item["sequence_parameters_changed"] for item in runs),
        "geometry_and_fusion_unused": all(item["geometry_and_fusion_parameters_unchanged"] for item in runs),
    }
    gates["passed"] = all(gates.values())
    report = {
        **plan,
        "status": "completed" if gates["passed"] else "failed",
        "runs": runs,
        "repeated_seed_evidence": {
            "shuffle_margins": shuffle_margins,
            "null_margins": null_margins,
            "unigram_improvements": unigram_improvements,
            "minimum_shuffle_margin": min(shuffle_margins),
            "minimum_null_margin": min(null_margins),
        },
        "gates": gates,
        "synthetic_optimization_performed": True,
        "real_data_training_performed": False,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    _atomic_json(output / "report.json", report)
    return report


def _verify_file(path: str, expected: str, label: str) -> Path:
    value = Path(path)
    if not value.is_file() or _sha256(value) != expected:
        raise ValueError(f"E006 Stage-A loader-smoke {label} hash contradiction")
    return value


def verify_smoke_prerequisites(specification: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Verify passed, non-authorizing smoke reports before the paired pilot."""
    prerequisites = {}
    for name in ("synthetic_smoke", "real_loader_smoke"):
        record = specification.get("prerequisites", {}).get(name) or {}
        if not record.get("path") or not record.get("sha256"):
            raise ValueError(f"E006 Stage-A v5 comparison prerequisite is not pinned: {name}")
        artifact = Path(record["path"])
        if not artifact.is_file() or _sha256(artifact) != record["sha256"]:
            raise ValueError(f"E006 Stage-A v5 comparison prerequisite hash contradiction: {name}")
        report = json.loads(artifact.read_text())
        if report.get("status") != "completed" or report.get("gates", {}).get("passed") is not True:
            raise ValueError(f"E006 Stage-A v5 comparison prerequisite did not pass: {name}")
        if report.get("authorizes_training") is not False or report.get("authorizes_joint_training") is not False:
            raise ValueError(f"E006 Stage-A v5 comparison prerequisite must be non-authorizing: {name}")
        prerequisites[name] = {"path": str(artifact), "sha256": record["sha256"]}
    return prerequisites


def _bounded_rows(
    dataset: SequenceOnlyRichDataset,
    *,
    count: int,
    seed: int,
    maximum_length: int,
) -> list[dict[str, Any]]:
    candidates = []
    for index, sample_id, length in dataset.iter_metadata():
        if length <= maximum_length:
            rank = hashlib.sha256(f"{seed}:{dataset.split}:{sample_id}".encode()).hexdigest()
            candidates.append((rank, sample_id, index))
    selected = sorted(candidates)[:count]
    if len(selected) != count:
        raise ValueError(f"E006 Stage-A loader smoke lacks {count} eligible {dataset.split} rows")
    return [dataset[index] for _, _, index in selected]


def plan_loader_smoke(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("version") != LOADER_SMOKE_VERSION:
        raise ValueError("E006 Stage-A loader-smoke version contradiction")
    base_path = _verify_file(
        config["base_training_config"], config["base_training_config_sha256"], "base configuration"
    )
    base = load_yaml(base_path)
    validate_phase3_config(base, mode="sequence-pretrain", synthetic=True)
    checkpoint = _verify_file(config["checkpoint_path"], config["checkpoint_sha256"], "checkpoint")
    authorization = _authorization(base)
    train = SequenceOnlyRichDataset(authorization, split="train")
    validation = SequenceOnlyRichDataset(authorization, split="validation")
    train_rows = _bounded_rows(
        train,
        count=int(config["train_sample_count"]),
        seed=int(config["seed"]),
        maximum_length=int(config["maximum_length"]),
    )
    validation_rows = _bounded_rows(
        validation,
        count=int(config["validation_sample_count"]),
        seed=int(config["seed"]) + 1,
        maximum_length=int(config["maximum_length"]),
    )
    return {
        "status": "planned",
        "version": LOADER_SMOKE_VERSION,
        "base_training_config": str(base_path),
        "base_training_config_sha256": config["base_training_config_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "train_sample_ids": [row["sample_id"] for row in train_rows],
        "validation_sample_ids": [row["sample_id"] for row in validation_rows],
        "projected_columns": list(SequenceOnlyRichDataset.columns),
        "token_contract": {
            "pad_token_id": 0,
            "mask_token_id": 1,
            "canonical_token_ids": list(range(2, 22)),
            "loss_positions": "corrupted_valid_only",
        },
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
        "authorizes_training": False,
        "output_dir": config["output_dir"],
        "protected_input_hashes": {
            **_protected_hashes(authorization),
            str(base_path): config["base_training_config_sha256"],
            str(checkpoint): config["checkpoint_sha256"],
        },
    }


def run_loader_smoke(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    plan = plan_loader_smoke(config_path)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 Stage-A loader-smoke output exists: {output}")
    base = load_yaml(config["base_training_config"])
    authorization = _authorization(base)
    train = SequenceOnlyRichDataset(authorization, split="train")
    validation = SequenceOnlyRichDataset(authorization, split="validation")
    train_rows = _bounded_rows(
        train,
        count=int(config["train_sample_count"]),
        seed=int(config["seed"]),
        maximum_length=int(config["maximum_length"]),
    )
    validation_rows = _bounded_rows(
        validation,
        count=int(config["validation_sample_count"]),
        seed=int(config["seed"]) + 1,
        maximum_length=int(config["maximum_length"]),
    )
    model = _model(base).eval()
    checkpoint = load_checkpoint(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    condition_records = []
    shape_records = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        batch = collate_sequence_pretraining(rows)
        corruption = context_corruption(
            batch["sequence_token_ids"],
            batch["residue_mask"],
            mask_token_id=1,
            probability=float(base["objective"]["mask_fraction"]),
            seed=int(config["seed"]),
            step=0 if split == "train" else 1,
        )
        null = torch.where(
            batch["residue_mask"], torch.ones_like(batch["sequence_token_ids"]), batch["sequence_token_ids"]
        )
        corrupted_inputs = corruption.inputs[corruption.corrupted_mask]
        if not torch.equal(corrupted_inputs, torch.ones_like(corrupted_inputs)):
            raise ValueError("E006 loader smoke exposed a corrupted target")
        if (corruption.corrupted_mask & ~batch["residue_mask"]).any():
            raise ValueError("E006 loader smoke corrupted padding")
        logits_by_condition = {}
        with torch.inference_mode():
            for name, inputs in (
                ("normal", corruption.inputs),
                ("shuffled", corruption.shuffled_inputs),
                ("null", null),
            ):
                logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
                loss = canonical_corrupted_cross_entropy(
                    logits,
                    batch["sequence_token_ids"],
                    corruption.corrupted_mask,
                    batch["residue_mask"],
                )
                logits_by_condition[name] = logits
                condition_records.append(
                    {
                        "split": split,
                        "condition": name,
                        "cross_entropy": float(loss),
                        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(loss)),
                        "canonical_corrupted_token_count": int(corruption.corrupted_mask.sum()),
                    }
                )
        shape_records.append(
            {
                "split": split,
                "token_shape": list(batch["sequence_token_ids"].shape),
                "residue_mask_shape": list(batch["residue_mask"].shape),
                "valid_token_count": int(batch["residue_mask"].sum()),
                "padding_token_count": int((~batch["residue_mask"]).sum()),
                "corrupted_token_count": int(corruption.corrupted_mask.sum()),
                "normal_shuffled_inputs_distinct": not torch.equal(corruption.inputs, corruption.shuffled_inputs),
                "normal_null_inputs_distinct": not torch.equal(corruption.inputs, null),
                "normal_shuffled_logits_distinct": not torch.equal(
                    logits_by_condition["normal"], logits_by_condition["shuffled"]
                ),
                "normal_null_logits_distinct": not torch.equal(
                    logits_by_condition["normal"], logits_by_condition["null"]
                ),
            }
        )
    protected_after = {
        **_protected_hashes(_authorization(base)),
        config["base_training_config"]: _sha256(Path(config["base_training_config"])),
        config["checkpoint_path"]: _sha256(Path(config["checkpoint_path"])),
    }
    memory = _memory(torch.device("cpu"))
    gates = {
        "finite_forward_and_loss": all(item["finite"] for item in condition_records),
        "paired_conditions_distinct": all(
            item["normal_shuffled_inputs_distinct"]
            and item["normal_null_inputs_distinct"]
            and item["normal_shuffled_logits_distinct"]
            and item["normal_null_logits_distinct"]
            for item in shape_records
        ),
        "canonical_corrupted_tokens_present": all(item["corrupted_token_count"] > 0 for item in shape_records),
        "pad_and_mask_excluded_from_targets": True,
        "protected_inputs_unchanged": protected_after == plan["protected_input_hashes"],
        "zero_optimizer_updates": True,
        "no_geometry_features_constructed": True,
        "memory_within_limit": float(memory["peak_rss_mib"]) <= float(config["maximum_rss_mib"]),
    }
    gates["passed"] = all(gates.values())
    report = {
        **plan,
        "status": "completed" if gates["passed"] else "failed",
        "condition_metrics": condition_records,
        "tensor_and_mask_checks": shape_records,
        "memory": memory,
        "gates": gates,
        "protected_input_hashes_after": protected_after,
        "protected_inputs_unchanged": protected_after == plan["protected_input_hashes"],
        "optimizer_updates": 0,
        "optimizer_created": False,
        "real_data_training_performed": False,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    _atomic_json(output / "report.json", report)
    return report


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sequence_parameters(model: E006RichGeometryCoDesign) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, value) for name, value in model.named_parameters() if name.startswith(SEQUENCE_PARAMETER_PREFIXES)]


def initialize_comparison_arm(
    model: E006RichGeometryCoDesign,
    arm: dict[str, Any],
) -> dict[str, Any]:
    """Apply only model weights for warm start; all training state is deliberately excluded."""
    initialization = str(arm["initialization"])
    if initialization == "scratch":
        return {
            "initialization": "scratch",
            "checkpoint_loaded": False,
            "model_weights_loaded": False,
            "optimizer_restored": False,
            "scheduler_restored": False,
            "scaler_restored": False,
            "rng_restored": False,
            "cursor_restored": False,
            "training_progress_restored": False,
        }
    if initialization != "checkpoint_weights_only":
        raise ValueError(f"Unknown E006 comparison initialization: {initialization}")
    checkpoint_path = _verify_file(arm["checkpoint_path"], arm["checkpoint_sha256"], "warm-start checkpoint")
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return {
        "initialization": initialization,
        "checkpoint_loaded": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": arm["checkpoint_sha256"],
        "model_weights_loaded": True,
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
        "rng_restored": False,
        "cursor_restored": False,
        "training_progress_restored": False,
        "source_optimizer_step_ignored": int(checkpoint.get("optimizer_step", -1)),
    }


def _comparison_panels(
    base: dict[str, Any], specification: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    authorization = _authorization(base)
    train_dataset = SequenceOnlyRichDataset(authorization, split="train")
    validation_dataset = SequenceOnlyRichDataset(authorization, split="validation")
    train_rows = _bounded_rows(
        train_dataset,
        count=int(specification["sample_count"]),
        seed=int(specification["seed"]),
        maximum_length=int(specification["maximum_length"]),
    )
    validation_rows = _bounded_rows(
        validation_dataset,
        count=int(specification["validation_sample_count"]),
        seed=int(specification["seed"]) + 1,
        maximum_length=int(specification["maximum_length"]),
    )
    train_ids = [str(row["sample_id"]) for row in train_rows]
    validation_ids = [str(row["sample_id"]) for row in validation_rows]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("E006 comparison panel contains duplicate sample IDs")
    if set(train_ids) & set(validation_ids):
        raise ValueError("E006 comparison train and validation panels overlap")
    identities = {
        "train_sample_count": len(train_ids),
        "validation_sample_count": len(validation_ids),
        "train_sample_ids": train_ids,
        "validation_sample_ids": validation_ids,
        "train_sample_id_sha256": _canonical_hash(train_ids),
        "validation_sample_id_sha256": _canonical_hash(validation_ids),
        "panels_disjoint": True,
        "maximum_length": int(specification["maximum_length"]),
    }
    return authorization, train_rows, validation_rows, identities


def comparison_preflight(config_path: Path) -> dict[str, Any]:
    specification = load_yaml(config_path)
    specification_hash = _sha256(config_path)
    if specification.get("version") != "e006_stage_a_context_v5_comparison_pilot_v1":
        raise ValueError("E006 Stage-A v5 comparison-pilot version contradiction")
    base_path = Path(specification["base_training_config"])
    expected_base_hash = specification.get("base_training_config_sha256")
    if not expected_base_hash or not base_path.is_file() or _sha256(base_path) != expected_base_hash:
        raise ValueError("E006 Stage-A v5 comparison base-configuration hash contradiction")
    base = load_yaml(base_path)
    validate_phase3_config(base, mode="sequence-pretrain", synthetic=True)
    if base["objective"].get("version") != "e006_stage_a_context_objective_v5":
        raise ValueError("E006 Stage-A comparison requires the v5 contextual objective")
    exact_values = {
        "sample_count": 512,
        "validation_sample_count": 256,
        "optimizer_updates": 250,
        "maximum_length": 128,
    }
    contradictions = [name for name, value in exact_values.items() if int(specification.get(name, -1)) != value]
    if contradictions:
        raise ValueError(f"E006 Stage-A v5 comparison bounded-contract contradiction: {contradictions}")
    if int(specification["scheduler_total_updates"]) < int(base["optimizer"]["warmup_updates"]) + 1:
        raise ValueError("E006 Stage-A v5 comparison scheduler horizon is shorter than warmup")
    if int(specification["physical_batch_size"]) < 1 or int(specification["evaluation_batch_size"]) < 1:
        raise ValueError("E006 Stage-A v5 comparison batch sizes must be positive")
    if specification.get("authorization") != {
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }:
        raise ValueError("E006 Stage-A v5 comparison must be explicitly non-authorizing")
    if specification.get("selection") != {
        "recommendation_policy": "held_out_contextual_pareto",
        "raw_cross_entropy_alone_forbidden": True,
        "fallback": "inconclusive",
    }:
        raise ValueError("E006 Stage-A v5 comparison selection-policy contradiction")
    arms = specification.get("arms", [])
    if [item.get("name") for item in arms] != ["scratch", "v4_warm_start"]:
        raise ValueError("E006 Stage-A v5 comparison requires scratch and v4 warm-start arms")
    if arms[0].get("initialization") != "scratch" or arms[1].get("initialization") != "checkpoint_weights_only":
        raise ValueError("E006 Stage-A v5 comparison arm initialization contradiction")
    _verify_file(arms[1]["checkpoint_path"], arms[1]["checkpoint_sha256"], "warm-start checkpoint")
    prerequisites = verify_smoke_prerequisites(specification)
    paired = specification.get("paired_controls", {})
    if not paired or not all(value is True for value in paired.values()):
        raise ValueError("E006 Stage-A v5 comparison controls must all be paired")
    authorization, _, _, panel_identities = _comparison_panels(base, specification)
    protected = {
        **_protected_hashes(authorization),
        str(config_path): specification_hash,
        str(base_path): expected_base_hash,
        arms[1]["checkpoint_path"]: arms[1]["checkpoint_sha256"],
        **{record["path"]: record["sha256"] for record in prerequisites.values()},
    }
    return {
        "status": "planned",
        "version": specification["version"],
        "comparison_config_path": str(config_path),
        "comparison_config_sha256": specification_hash,
        "base_training_config": str(base_path),
        "base_training_config_sha256": expected_base_hash,
        "objective_version": base["objective"]["version"],
        "arms": arms,
        "paired_controls": paired,
        "verified_prerequisites": prerequisites,
        "optimizer_updates": int(specification["optimizer_updates"]),
        "scheduler_total_updates": int(specification["scheduler_total_updates"]),
        "physical_batch_size": int(specification["physical_batch_size"]),
        "evaluation_batch_size": int(specification["evaluation_batch_size"]),
        "objective": {
            "version": base["objective"]["version"],
            "canonical_support": "token_ids_2_through_21",
            "loss_positions": "corrupted_valid_positions_only",
            "mask_token_id": 1,
            "mask_fraction": float(base["objective"]["mask_fraction"]),
            "context_contrast_weight": float(base["objective"]["context_contrast_weight"]),
            "context_contrast_margin_nats": float(base["objective"]["context_contrast_margin_nats"]),
        },
        "panel_identities": panel_identities,
        "protected_input_hashes": protected,
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "output_dir": specification["output_dir"],
    }


def _move_sequence_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}


def _evaluate_comparison_panel(
    model: E006RichGeometryCoDesign,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    mask_probability: float,
    contrast_weight: float,
    contrast_margin_nats: float,
    device: torch.device,
) -> dict[str, Any]:
    totals = {name: {"loss": 0.0, "correct": 0, "tokens": 0} for name in ("normal", "shuffled", "null")}
    target_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    model.eval()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(rows), batch_size)):
            batch = _move_sequence_batch(collate_sequence_pretraining(rows[start : start + batch_size]), device)
            corruption = context_corruption(
                batch["sequence_token_ids"],
                batch["residue_mask"],
                mask_token_id=1,
                probability=mask_probability,
                seed=seed,
                step=batch_index,
            )
            null = torch.where(
                batch["residue_mask"],
                torch.ones_like(batch["sequence_token_ids"]),
                batch["sequence_token_ids"],
            )
            targets = batch["sequence_token_ids"]
            selected = corruption.corrupted_mask
            target_digest.update(targets[selected].detach().cpu().contiguous().numpy().tobytes())
            mask_digest.update(selected.detach().cpu().contiguous().numpy().tobytes())
            for name, inputs in (
                ("normal", corruption.inputs),
                ("shuffled", corruption.shuffled_inputs),
                ("null", null),
            ):
                logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
                loss = canonical_corrupted_cross_entropy(logits, targets, selected, batch["residue_mask"])
                token_count = int(selected.sum())
                predictions = logits[selected, 2:22].argmax(-1) + 2
                totals[name]["loss"] += float(loss) * token_count
                totals[name]["correct"] += int((predictions == targets[selected]).sum())
                totals[name]["tokens"] += token_count
    conditions = {
        name: {
            "canonical_cross_entropy": value["loss"] / value["tokens"],
            "top1_accuracy": value["correct"] / value["tokens"],
            "canonical_corrupted_token_count": value["tokens"],
        }
        for name, value in totals.items()
    }
    normal = conditions["normal"]["canonical_cross_entropy"]
    shuffled = conditions["shuffled"]["canonical_cross_entropy"]
    context_contrast = max(normal - shuffled + contrast_margin_nats, 0.0)
    conditions["margin_sign_convention"] = "normal_ce_minus_counterfactual_ce; negative favors ordered context"
    conditions["normal_minus_shuffled_ce"] = normal - shuffled
    conditions["normal_minus_null_ce"] = normal - conditions["null"]["canonical_cross_entropy"]
    return {
        "conditions": conditions,
        "objective_components": {
            "canonical_corrupted_position_ce": normal,
            "context_contrast": context_contrast,
            "weighted_context_contrast": context_contrast * contrast_weight,
            "total": normal + context_contrast * contrast_weight,
        },
        "target_sha256": target_digest.hexdigest(),
        "corruption_mask_sha256": mask_digest.hexdigest(),
    }


def _enforce_comparison_memory(
    device: torch.device,
    specification: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, float | None]:
    memory = _memory(device)
    if float(memory["peak_rss_mib"]) > float(specification["maximum_rss_mib"]):
        raise MemoryError(
            "E006 comparison pilot exceeded RSS limit: "
            f"{memory['peak_rss_mib']:.3f} MiB > {float(specification['maximum_rss_mib']):.3f} MiB"
        )
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        memory["peak_cuda_allocated_mib"] = allocated
        memory["peak_cuda_reserved_mib"] = reserved
        limits = base["memory"]
        if allocated > float(limits["maximum_cuda_allocated_mib"]):
            raise MemoryError("E006 comparison pilot exceeded CUDA allocated-memory limit")
        if reserved > float(limits["maximum_cuda_reserved_mib"]):
            raise MemoryError("E006 comparison pilot exceeded CUDA reserved-memory limit")
    return memory


def _comparison_arm_worker(config_path: str, arm_name: str, queue: Any) -> None:
    try:
        specification = load_yaml(config_path)
        base = load_yaml(specification["base_training_config"])
        arm = next(item for item in specification["arms"] if item["name"] == arm_name)
        authorization, train_rows, validation_rows, panel_identities = _comparison_panels(base, specification)
        seed = int(specification["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = _model(base)
        initialization = initialize_comparison_arm(model, arm)
        sequence_named = _sequence_parameters(model)
        sequence_parameters = [value for _, value in sequence_named]
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(SEQUENCE_PARAMETER_PREFIXES))
        before = {name: value.detach().cpu().clone() for name, value in sequence_named}
        frozen_before = _parameter_digest(
            [value for name, value in model.named_parameters() if not name.startswith(SEQUENCE_PARAMETER_PREFIXES)]
        )
        device = torch.device(specification.get("device", base.get("device", "cpu")))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("E006 comparison pilot requested unavailable CUDA")
        model.to(device)
        optimizer_config = base["optimizer"]
        optimizer = torch.optim.AdamW(
            sequence_parameters,
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        updates = int(specification["optimizer_updates"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _scheduler_multiplier(
                step,
                int(optimizer_config["warmup_updates"]),
                int(specification["scheduler_total_updates"]),
            ),
        )
        amp_enabled = bool(base["mixed_precision"]["enabled"]) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        batch_size = int(specification["physical_batch_size"])
        mask_probability = float(base["objective"]["mask_fraction"])
        initial = _evaluate_comparison_panel(
            model,
            validation_rows,
            batch_size=int(specification["evaluation_batch_size"]),
            seed=seed + 100_000,
            mask_probability=mask_probability,
            contrast_weight=float(base["objective"]["context_contrast_weight"]),
            contrast_margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
            device=device,
        )
        model.train()
        successful_updates = attempts = overflows = 0
        training_target_digest = hashlib.sha256()
        training_mask_digest = hashlib.sha256()
        objective_sums = {
            "canonical_ce": 0.0,
            "context_contrast": 0.0,
            "weighted_context_contrast": 0.0,
            "total": 0.0,
        }
        started = time.monotonic()
        while successful_updates < updates:
            start = (successful_updates * batch_size) % len(train_rows)
            selected_rows = [train_rows[(start + offset) % len(train_rows)] for offset in range(batch_size)]
            batch = _move_sequence_batch(collate_sequence_pretraining(selected_rows), device)
            corruption = context_corruption(
                batch["sequence_token_ids"],
                batch["residue_mask"],
                mask_token_id=1,
                probability=mask_probability,
                seed=seed,
                step=successful_updates,
            )
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            )
            with autocast:
                normal_logits = model.forward_sequence_pretraining(corruption.inputs, batch["residue_mask"])
                with torch.no_grad():
                    shuffled_logits = model.forward_sequence_pretraining(
                        corruption.shuffled_inputs, batch["residue_mask"]
                    )
                losses = contextual_stage_a_loss(
                    normal_logits,
                    shuffled_logits,
                    batch["sequence_token_ids"],
                    corruption.corrupted_mask,
                    batch["residue_mask"],
                    contrast_weight=float(base["objective"]["context_contrast_weight"]),
                    contrast_margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
                )
            if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
                raise FloatingPointError("E006 comparison pilot produced a non-finite loss")
            scale_before = float(scaler.get_scale()) if amp_enabled else 1.0
            scaler.scale(losses["total"]).backward()
            if amp_enabled:
                scaler.unscale_(optimizer)
            missing_gradients = [name for name, parameter in sequence_named if parameter.grad is None]
            if missing_gradients:
                raise FloatingPointError(f"E006 comparison pilot missing active gradients: {missing_gradients[:20]}")
            if not amp_enabled and any(
                not bool(torch.isfinite(parameter.grad).all()) for _, parameter in sequence_named
            ):
                raise FloatingPointError("E006 comparison pilot produced non-finite gradients")
            torch.nn.utils.clip_grad_norm_(sequence_parameters, float(optimizer_config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            overflow = amp_enabled and float(scaler.get_scale()) < scale_before
            attempts += 1
            if overflow:
                overflows += 1
                if overflows > int(specification["maximum_amp_overflows"]):
                    raise FloatingPointError("E006 comparison pilot exceeded its AMP-overflow bound")
                continue
            scheduler.step()
            successful_updates += 1
            training_target_digest.update(
                batch["sequence_token_ids"][corruption.corrupted_mask].detach().cpu().contiguous().numpy().tobytes()
            )
            training_mask_digest.update(corruption.corrupted_mask.detach().cpu().contiguous().numpy().tobytes())
            objective_sums["canonical_ce"] += float(losses["sequence"].detach())
            objective_sums["context_contrast"] += float(losses["context_contrast"].detach())
            objective_sums["weighted_context_contrast"] += float(losses["context_contrast_weighted"].detach())
            objective_sums["total"] += float(losses["total"].detach())
            _enforce_comparison_memory(device, specification, base)
        final = _evaluate_comparison_panel(
            model,
            validation_rows,
            batch_size=int(specification["evaluation_batch_size"]),
            seed=seed + 100_000,
            mask_probability=mask_probability,
            contrast_weight=float(base["objective"]["context_contrast_weight"]),
            contrast_margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
            device=device,
        )
        change_squared = baseline_squared = 0.0
        trunk_change_squared = trunk_baseline_squared = 0.0
        for name, value in sequence_named:
            current = value.detach().cpu()
            change_squared += float((current - before[name]).square().sum())
            baseline_squared += float(before[name].square().sum())
            if name.startswith(("position_embedding.", "sequence_layers.", "sequence_norm.")):
                trunk_change_squared += float((current - before[name]).square().sum())
                trunk_baseline_squared += float(before[name].square().sum())
        frozen_after = _parameter_digest(
            [value for name, value in model.named_parameters() if not name.startswith(SEQUENCE_PARAMETER_PREFIXES)]
        )
        memory = _enforce_comparison_memory(device, specification, base)
        result = {
            "status": "completed",
            "arm": arm_name,
            "process_id": os.getpid(),
            "initialization": initialization,
            "panel_identities": panel_identities,
            "initial_evaluation": initial,
            "final_evaluation": final,
            "objective_components_mean": {name: value / updates for name, value in objective_sums.items()},
            "successful_optimizer_updates": successful_updates,
            "optimizer_attempts": attempts,
            "amp_overflows": overflows,
            "finite_losses_and_gradients": True,
            "training_target_sha256": training_target_digest.hexdigest(),
            "training_corruption_mask_sha256": training_mask_digest.hexdigest(),
            "sequence_branch_relative_parameter_change": math.sqrt(change_squared)
            / max(math.sqrt(baseline_squared), 1e-12),
            "sequence_trunk_relative_parameter_change": math.sqrt(trunk_change_squared)
            / max(math.sqrt(trunk_baseline_squared), 1e-12),
            "frozen_geometry_and_fusion_parameters_unchanged": frozen_before == frozen_after,
            "constructs_rich_pair_features": False,
            "feature_complexity": "O(N)",
            "process_isolation": "spawned_dedicated_child",
            "optimizer_settings": {
                "name": "AdamW",
                "learning_rate": float(optimizer_config["learning_rate"]),
                "weight_decay": float(optimizer_config["weight_decay"]),
                "gradient_clip_norm": float(optimizer_config["gradient_clip_norm"]),
                "warmup_updates": int(optimizer_config["warmup_updates"]),
                "scheduler_total_updates": int(specification["scheduler_total_updates"]),
                "amp_enabled": amp_enabled,
            },
            "memory": memory,
            "elapsed_seconds": time.monotonic() - started,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "protected_dataset_hashes": _protected_hashes(authorization),
        }
        queue.put({"ok": True, "result": result})
    except BaseException as error:
        queue.put({"ok": False, "error_type": type(error).__name__, "error": str(error)[:2000]})


def _recommend_arm(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {item["arm"]: item for item in results}
    scratch = by_name["scratch"]["final_evaluation"]["conditions"]
    warm = by_name["v4_warm_start"]["final_evaluation"]["conditions"]

    def contextual_dominates(first: dict[str, Any], second: dict[str, Any]) -> bool:
        return (
            first["normal"]["canonical_cross_entropy"] <= second["normal"]["canonical_cross_entropy"]
            and first["normal_minus_shuffled_ce"] <= second["normal_minus_shuffled_ce"]
            and first["normal_minus_null_ce"] <= second["normal_minus_null_ce"]
            and first["normal"]["top1_accuracy"] >= second["normal"]["top1_accuracy"]
        )

    if contextual_dominates(warm, scratch) and warm != scratch:
        recommendation = "v4_warm_start"
    elif contextual_dominates(scratch, warm) and warm != scratch:
        recommendation = "scratch"
    else:
        recommendation = "inconclusive"
    return {
        "recommendation": recommendation,
        "policy": "Pareto dominance over held-out normal CE, normal-minus-shuffled/null CE, and top-1 accuracy",
        "raw_ce_alone_used": False,
        "sign_convention": "normal_ce_minus_counterfactual_ce; more negative is better",
    }


def _paired_control_checks(results: list[dict[str, Any]], optimizer_updates: int) -> dict[str, bool]:
    if len(results) != 2:
        raise ValueError("E006 comparison requires exactly two completed arms")
    first, second = results
    return {
        "sample_order_identical": first["panel_identities"] == second["panel_identities"],
        "training_targets_identical": first["training_target_sha256"] == second["training_target_sha256"],
        "training_corruption_masks_identical": first["training_corruption_mask_sha256"]
        == second["training_corruption_mask_sha256"],
        "evaluation_targets_identical": first["final_evaluation"]["target_sha256"]
        == second["final_evaluation"]["target_sha256"],
        "evaluation_corruption_masks_identical": first["final_evaluation"]["corruption_mask_sha256"]
        == second["final_evaluation"]["corruption_mask_sha256"],
        "isolated_processes": all(item["process_isolation"] == "spawned_dedicated_child" for item in results),
        "finite_losses_and_gradients": all(item["finite_losses_and_gradients"] for item in results),
        "exact_successful_updates": all(item["successful_optimizer_updates"] == optimizer_updates for item in results),
        "no_geometry_or_pair_features": all(
            not item["constructs_rich_pair_features"] and item["frozen_geometry_and_fusion_parameters_unchanged"]
            for item in results
        ),
    }


def comparison_protocol(report_path: Path, report_sha256: str, recommendation: str) -> dict[str, Any]:
    return {
        "status": "completed",
        "version": "e006_stage_a_context_v5_comparison_pilot_v1",
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "recommendation": recommendation,
        "protected_inputs_unchanged": True,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }


def run_comparison_pilot(config_path: Path) -> dict[str, Any]:
    specification = load_yaml(config_path)
    output = Path(specification["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 comparison-pilot output already exists: {output}")
    preflight = comparison_preflight(config_path)
    output.mkdir(parents=True)
    heartbeat = output / "heartbeat.json"
    protocol_path = output / "protocol.json"
    report_path = output / "report.json"
    _atomic_json(
        heartbeat,
        {
            "status": "running",
            "version": specification["version"],
            "completed_arms": [],
            "authorizes_training": False,
        },
    )
    results = []
    try:
        context = multiprocessing.get_context("spawn")
        for arm in ("scratch", "v4_warm_start"):
            result_queue = context.Queue(maxsize=1)
            process = context.Process(target=_comparison_arm_worker, args=(str(config_path), arm, result_queue))
            process.start()
            message = None
            while process.is_alive() and message is None:
                try:
                    message = result_queue.get(timeout=1.0)
                except queue_module.Empty:
                    pass
            process.join()
            if message is None:
                try:
                    message = result_queue.get_nowait()
                except queue_module.Empty as error:
                    raise RuntimeError(
                        f"E006 comparison arm exited without a result: {arm}; exit={process.exitcode}"
                    ) from error
            if process.exitcode != 0 or not message.get("ok"):
                raise RuntimeError(
                    f"E006 comparison arm failed: {arm}: {message.get('error_type')}: {message.get('error')}"
                )
            results.append(message["result"])
            _atomic_json(
                heartbeat,
                {
                    "status": "running",
                    "version": specification["version"],
                    "completed_arms": [item["arm"] for item in results],
                    "authorizes_training": False,
                },
            )
        paired = _paired_control_checks(results, int(specification["optimizer_updates"]))
        if not all(paired.values()):
            raise ValueError(f"E006 comparison paired-control contradiction: {paired}")
        base = load_yaml(specification["base_training_config"])
        authorization = _authorization(base)
        protected_after = {
            **_protected_hashes(authorization),
            str(config_path): _sha256(config_path),
            specification["base_training_config"]: _sha256(Path(specification["base_training_config"])),
            specification["arms"][1]["checkpoint_path"]: _sha256(Path(specification["arms"][1]["checkpoint_path"])),
            **{record["path"]: _sha256(Path(record["path"])) for record in specification["prerequisites"].values()},
        }
        if protected_after != preflight["protected_input_hashes"]:
            raise RuntimeError("E006 comparison pilot protected inputs changed")
        report = {
            **preflight,
            "status": "completed",
            "arms": results,
            "paired_control_checks": paired,
            "recommendation": _recommend_arm(results),
            "protected_input_hashes_after": protected_after,
            "protected_inputs_unchanged": True,
            "training_scope": "bounded_non_authorizing_comparison_pilot",
            "definitive_training_performed": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
        }
        _atomic_json(report_path, report)
        protocol = comparison_protocol(
            report_path,
            _sha256(report_path),
            report["recommendation"]["recommendation"],
        )
        _atomic_json(protocol_path, protocol)
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "version": specification["version"],
                "completed_arms": [item["arm"] for item in results],
                "report_sha256": protocol["report_sha256"],
                "authorizes_training": False,
            },
        )
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "version": specification["version"],
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "completed_arms": [item["arm"] for item in results],
            "authorizes_training": False,
            "authorizes_joint_training": False,
        }
        _atomic_json(protocol_path, failure)
        _atomic_json(heartbeat, failure)
        raise
