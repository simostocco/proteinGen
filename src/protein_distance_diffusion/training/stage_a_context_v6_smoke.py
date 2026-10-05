"""Non-authorizing smoke and bounded correction pilot for Stage-A objective v6."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_production import (
    _model,
    _protected_hashes,
    collate_sequence_pretraining,
)
from protein_distance_diffusion.training.rich_codesign_smoke import _memory
from protein_distance_diffusion.training.stage_a_context import (
    STAGE_A_CONTEXT_OBJECTIVE_V6,
    context_corruption,
    contextual_stage_a_loss,
    contextual_stage_a_loss_v6,
    paired_dropout_forwards,
    visible_shuffle_inputs,
)
from protein_distance_diffusion.training.stage_a_context_smoke import (
    SEQUENCE_PARAMETER_PREFIXES,
    _comparison_panels,
    _condition_metrics,
    _evaluate_comparison_panel,
    _move_sequence_batch,
    _sequence_parameters,
    _sha256,
    _tiny_model,
    contextual_toy,
    initialize_comparison_arm,
    toy_contract,
)

SMOKE_VERSION = "e006_stage_a_context_v6_synthetic_smoke_v1"
PILOT_VERSION = "e006_stage_a_context_v6_correction_pilot_v1"
OBJECTIVE_ARMS = ("v6_per_sample_paired_dropout", "v5_batch_mean_control")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _v6_tiny_model(seed: int) -> torch.nn.Module:
    model = _tiny_model(seed)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.1
    return model


def _verify_parity_report(config: dict[str, Any]) -> dict[str, str]:
    record = config.get("parity_audit") or {}
    path = Path(record.get("path", ""))
    expected = record.get("sha256")
    if not expected or not path.is_file() or _sha256(path) != expected:
        raise ValueError("E006 Stage-A v6 parity-audit hash contradiction")
    report = json.loads(path.read_text())
    if report.get("status") != "completed" or report.get("authorizes_training") is not False:
        raise ValueError("E006 Stage-A v6 parity audit is not completed and non-authorizing")
    return {"path": str(path), "sha256": str(expected)}


def _objective_step(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    objective: str,
    seed: int,
    step: int,
    mask_fraction: float,
    contrast_weight: float,
    margin_nats: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    corruption = context_corruption(
        batch["sequence_token_ids"],
        batch["residue_mask"],
        mask_token_id=1,
        probability=mask_fraction,
        seed=seed,
        step=step,
    )
    losses = _paired_objective_loss(
        model,
        batch["sequence_token_ids"],
        corruption.inputs,
        corruption.shuffled_inputs,
        corruption.corrupted_mask,
        batch["residue_mask"],
        objective=objective,
        contrast_weight=contrast_weight,
        margin_nats=margin_nats,
    )
    return losses, batch["sequence_token_ids"][corruption.corrupted_mask], corruption.corrupted_mask


def _paired_objective_loss(
    model: torch.nn.Module,
    targets: torch.Tensor,
    normal_inputs: torch.Tensor,
    shuffled_inputs: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    objective: str,
    contrast_weight: float,
    margin_nats: float,
) -> dict[str, torch.Tensor]:
    if objective == OBJECTIVE_ARMS[0]:
        normal, shuffled, paired = paired_dropout_forwards(
            model,
            normal_inputs,
            shuffled_inputs,
            residue_mask,
        )
        losses = contextual_stage_a_loss_v6(
            normal,
            shuffled,
            targets,
            corrupted_mask,
            residue_mask,
            contrast_weight=contrast_weight,
            contrast_margin_nats=margin_nats,
            paired_dropout_evidence=paired,
        )
    elif objective == OBJECTIVE_ARMS[1]:
        normal = model.forward_sequence_pretraining(normal_inputs, residue_mask)
        with torch.no_grad():
            shuffled = model.forward_sequence_pretraining(shuffled_inputs, residue_mask)
        losses = contextual_stage_a_loss(
            normal,
            shuffled,
            targets,
            corrupted_mask,
            residue_mask,
            contrast_weight=contrast_weight,
            contrast_margin_nats=margin_nats,
        )
        losses["context_active_hinge_sample_fraction"] = (losses["context_contrast"] > 0).float()
        losses["paired_dropout_verified"] = losses["sequence"].new_zeros(())
    else:
        raise ValueError(f"Unknown E006 Stage-A correction objective arm: {objective}")
    return losses


def plan_synthetic_smoke_v6(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("version") != SMOKE_VERSION:
        raise ValueError("E006 Stage-A v6 smoke version contradiction")
    parity = _verify_parity_report(config)
    contract = toy_contract()
    if not contract["marginals_equal"] or not contract["target_leakage_absent"]:
        raise ValueError("E006 Stage-A v6 toy contract is invalid")
    return {
        "status": "planned",
        "version": SMOKE_VERSION,
        "objective_arms": list(OBJECTIVE_ARMS),
        "parity_audit": parity,
        "toy_contract": contract,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "output_dir": config["output_dir"],
    }


def run_synthetic_smoke_v6(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    plan = plan_synthetic_smoke_v6(config_path)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 Stage-A v6 smoke output exists: {output}")
    train = contextual_toy("train")
    held_out = contextual_toy("held_out")
    rows = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        template = _v6_tiny_model(seed)
        initial_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
        for objective in OBJECTIVE_ARMS:
            model = _v6_tiny_model(seed)
            model.load_state_dict(initial_state)
            parameters = [
                value for name, value in model.named_parameters() if name.startswith(SEQUENCE_PARAMETER_PREFIXES)
            ]
            optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
            initial = _condition_metrics(model.eval(), held_out, shuffle_seed=seed + 10)
            finite = True
            active = []
            paired = []
            model.train()
            for step in range(int(config["optimizer_updates_per_seed"])):
                shuffled_inputs = visible_shuffle_inputs(
                    train["inputs"],
                    train["corrupted_mask"],
                    train["residue_mask"],
                    seed=seed,
                    step=step,
                )
                losses = _paired_objective_loss(
                    model,
                    train["targets"],
                    train["inputs"],
                    shuffled_inputs,
                    train["corrupted_mask"],
                    train["residue_mask"],
                    objective=objective,
                    contrast_weight=float(config["context_contrast_weight"]),
                    margin_nats=float(config["context_contrast_margin_nats"]),
                )
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                finite &= bool(torch.isfinite(losses["total"])) and all(
                    value.grad is not None and torch.isfinite(value.grad).all() for value in parameters
                )
                optimizer.step()
                active.append(float(losses["context_active_hinge_sample_fraction"].detach()))
                paired.append(float(losses["paired_dropout_verified"].detach()))
            final = _condition_metrics(model.eval(), held_out, shuffle_seed=seed + 10)
            rows.append(
                {
                    "seed": seed,
                    "objective": objective,
                    "initial": initial,
                    "final": final,
                    "finite_losses_and_gradients": finite,
                    "mean_active_hinge_sample_fraction": float(np.mean(active)),
                    "paired_dropout_verified_fraction": float(np.mean(paired)),
                }
            )
    gates = {
        "finite_losses_and_gradients": all(item["finite_losses_and_gradients"] for item in rows),
        "v6_paired_dropout_verified": all(
            item["paired_dropout_verified_fraction"] == 1.0 for item in rows if item["objective"] == OBJECTIVE_ARMS[0]
        ),
        "matched_initial_metrics": all(
            next(item for item in rows if item["seed"] == seed and item["objective"] == OBJECTIVE_ARMS[0])["initial"]
            == next(item for item in rows if item["seed"] == seed and item["objective"] == OBJECTIVE_ARMS[1])["initial"]
            for seed in map(int, config["seeds"])
        ),
        "target_leakage_absent": plan["toy_contract"]["target_leakage_absent"],
        "v6_held_out_normal_beats_unigram": all(
            item["final"]["normal"]["cross_entropy"]
            <= item["final"]["unigram"]["cross_entropy"] - float(config["minimum_unigram_improvement_nats"])
            for item in rows
            if item["objective"] == OBJECTIVE_ARMS[0]
        ),
        "v6_held_out_shuffle_margin": all(
            item["final"]["normal_to_shuffled_margin"] >= float(config["minimum_counterfactual_margin_nats"])
            for item in rows
            if item["objective"] == OBJECTIVE_ARMS[0]
        ),
        "v6_held_out_null_margin": all(
            item["final"]["normal_to_null_margin"] >= float(config["minimum_counterfactual_margin_nats"])
            for item in rows
            if item["objective"] == OBJECTIVE_ARMS[0]
        ),
        "non_authorizing": True,
    }
    gates["passed"] = all(gates.values())
    report = {
        **plan,
        "status": "completed" if gates["passed"] else "failed",
        "runs": rows,
        "gates": gates,
        "training_scope": "synthetic_only",
        "real_data_training_performed": False,
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "protocol.json",
        {
            "status": report["status"],
            "version": SMOKE_VERSION,
            "report_sha256": _sha256(output / "report.json"),
            "authorizes_training": False,
            "authorizes_joint_training": False,
        },
    )
    return report


def comparison_preflight_v6(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if config.get("version") != PILOT_VERSION:
        raise ValueError("E006 Stage-A v6 correction-pilot version contradiction")
    parity = _verify_parity_report(config)
    base_path = Path(config["base_training_config"])
    if _sha256(base_path) != config.get("base_training_config_sha256"):
        raise ValueError("E006 Stage-A v6 pilot base-config hash contradiction")
    base = load_yaml(base_path)
    if base["objective"].get("version") != STAGE_A_CONTEXT_OBJECTIVE_V6:
        raise ValueError("E006 Stage-A v6 pilot requires the v6 base objective")
    smoke = config.get("synthetic_smoke") or {}
    smoke_path = Path(smoke.get("path", ""))
    if not smoke.get("sha256") or not smoke_path.is_file() or _sha256(smoke_path) != smoke["sha256"]:
        raise ValueError("E006 Stage-A v6 pilot synthetic-smoke hash contradiction")
    smoke_report = json.loads(smoke_path.read_text())
    if smoke_report.get("status") != "completed" or smoke_report.get("gates", {}).get("passed") is not True:
        raise ValueError("E006 Stage-A v6 pilot synthetic smoke did not pass")
    exact = {"sample_count": 512, "validation_sample_count": 256, "optimizer_updates": 250, "maximum_length": 128}
    contradictions = [name for name, value in exact.items() if int(config.get(name, -1)) != value]
    if contradictions:
        raise ValueError(f"E006 Stage-A v6 pilot bounded contract contradiction: {contradictions}")
    authorization, train, validation, identities = _comparison_panels(base, config)
    return {
        "status": "planned",
        "version": PILOT_VERSION,
        "objective_arms": list(OBJECTIVE_ARMS),
        "parity_audit": parity,
        "synthetic_smoke": smoke,
        "panel_identities": identities,
        "protected_input_hashes": {
            **_protected_hashes(authorization),
            str(base_path): config["base_training_config_sha256"],
            str(smoke_path): smoke["sha256"],
            parity["path"]: parity["sha256"],
        },
        "train_rows": train,
        "validation_rows": validation,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "output_dir": config["output_dir"],
    }


def _run_pilot_arm(
    base: dict[str, Any],
    config: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    objective: str,
) -> dict[str, Any]:
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = _model(base)
    initialization = initialize_comparison_arm(model, config["initialization"])
    named = _sequence_parameters(model)
    parameters = [value for _, value in named]
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith(SEQUENCE_PARAMETER_PREFIXES))
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E006 Stage-A v6 pilot requested unavailable CUDA")
    model.to(device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(base["optimizer"]["learning_rate"]),
        weight_decay=float(base["optimizer"]["weight_decay"]),
    )
    evaluation_args = {
        "batch_size": int(config["evaluation_batch_size"]),
        "seed": seed + 100_000,
        "mask_probability": float(base["objective"]["mask_fraction"]),
        "contrast_weight": float(base["objective"]["context_contrast_weight"]),
        "contrast_margin_nats": float(base["objective"]["context_contrast_margin_nats"]),
        "device": device,
    }
    initial = _evaluate_comparison_panel(model, validation_rows, **evaluation_args)
    initial["active_hinge_sample_fraction"] = _heldout_active_hinge_fraction(
        model,
        validation_rows,
        batch_size=int(config["evaluation_batch_size"]),
        seed=seed + 100_000,
        mask_fraction=float(base["objective"]["mask_fraction"]),
        contrast_weight=float(base["objective"]["context_contrast_weight"]),
        margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
        device=device,
    )
    target_hash = hashlib.sha256()
    mask_hash = hashlib.sha256()
    active = []
    finite = True
    batch_size = int(config["physical_batch_size"])
    updates = int(config["optimizer_updates"])
    model.train()
    for step in range(updates):
        start = step * batch_size % len(train_rows)
        rows = [train_rows[(start + offset) % len(train_rows)] for offset in range(batch_size)]
        batch = _move_sequence_batch(collate_sequence_pretraining(rows), device)
        optimizer.zero_grad(set_to_none=True)
        losses, selected_targets, selected_mask = _objective_step(
            model,
            batch,
            objective=objective,
            seed=seed,
            step=step,
            mask_fraction=float(base["objective"]["mask_fraction"]),
            contrast_weight=float(base["objective"]["context_contrast_weight"]),
            margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
        )
        losses["total"].backward()
        finite &= bool(torch.isfinite(losses["total"])) and all(
            value.grad is not None and torch.isfinite(value.grad).all() for value in parameters
        )
        torch.nn.utils.clip_grad_norm_(parameters, float(base["optimizer"]["gradient_clip_norm"]))
        optimizer.step()
        target_hash.update(selected_targets.detach().cpu().contiguous().numpy().tobytes())
        mask_hash.update(selected_mask.detach().cpu().contiguous().numpy().tobytes())
        active.append(float(losses["context_active_hinge_sample_fraction"].detach()))
        if step % 10 == 0:
            memory = _memory(device)
            if float(memory["peak_rss_mib"]) > float(config["maximum_rss_mib"]):
                raise MemoryError("E006 Stage-A v6 correction pilot exceeded its RSS limit")
            if device.type == "cuda" and (
                float(memory["peak_cuda_allocated_mib"] or 0) > float(config["maximum_cuda_allocated_mib"])
                or float(memory["peak_cuda_reserved_mib"] or 0) > float(config["maximum_cuda_reserved_mib"])
            ):
                raise MemoryError("E006 Stage-A v6 correction pilot exceeded its CUDA-memory limit")
    final = _evaluate_comparison_panel(model, validation_rows, **evaluation_args)
    final["active_hinge_sample_fraction"] = _heldout_active_hinge_fraction(
        model,
        validation_rows,
        batch_size=int(config["evaluation_batch_size"]),
        seed=seed + 100_000,
        mask_fraction=float(base["objective"]["mask_fraction"]),
        contrast_weight=float(base["objective"]["context_contrast_weight"]),
        margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
        device=device,
    )
    return {
        "objective": objective,
        "initialization": initialization,
        "initial_evaluation": initial,
        "final_evaluation": final,
        "successful_optimizer_updates": updates,
        "training_target_sha256": target_hash.hexdigest(),
        "training_corruption_mask_sha256": mask_hash.hexdigest(),
        "active_hinge_sample_fraction": float(np.mean(active)),
        "finite_losses_and_gradients": finite,
        "memory": _memory(device),
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }


def _heldout_active_hinge_fraction(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    mask_fraction: float,
    contrast_weight: float,
    margin_nats: float,
    device: torch.device,
) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(rows), batch_size)):
            batch = _move_sequence_batch(collate_sequence_pretraining(rows[start : start + batch_size]), device)
            corruption = context_corruption(
                batch["sequence_token_ids"],
                batch["residue_mask"],
                mask_token_id=1,
                probability=mask_fraction,
                seed=seed,
                step=batch_index,
            )
            losses = _paired_objective_loss(
                model,
                batch["sequence_token_ids"],
                corruption.inputs,
                corruption.shuffled_inputs,
                corruption.corrupted_mask,
                batch["residue_mask"],
                objective=OBJECTIVE_ARMS[0],
                contrast_weight=contrast_weight,
                margin_nats=margin_nats,
            )
            values.extend([float(losses["context_active_hinge_sample_fraction"])] * len(batch["sample_ids"]))
    model.train()
    return float(np.mean(values))


def run_comparison_pilot_v6(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 Stage-A v6 correction-pilot output exists: {output}")
    plan = comparison_preflight_v6(config_path)
    train_rows = plan.pop("train_rows")
    validation_rows = plan.pop("validation_rows")
    base = load_yaml(config["base_training_config"])
    results = [_run_pilot_arm(base, config, train_rows, validation_rows, objective) for objective in OBJECTIVE_ARMS]
    paired = {
        "training_targets_identical": results[0]["training_target_sha256"] == results[1]["training_target_sha256"],
        "corruption_masks_identical": results[0]["training_corruption_mask_sha256"]
        == results[1]["training_corruption_mask_sha256"],
        "evaluation_targets_identical": results[0]["final_evaluation"]["target_sha256"]
        == results[1]["final_evaluation"]["target_sha256"],
        "finite_gradients": all(item["finite_losses_and_gradients"] for item in results),
        "exact_updates": all(item["successful_optimizer_updates"] == 250 for item in results),
    }
    gates = {**paired, "passed": all(paired.values())}
    report = {
        **plan,
        "status": "completed" if gates["passed"] else "failed",
        "arms": results,
        "gates": gates,
        "comparison_metrics": [
            "held_out_normal_ce",
            "normal_minus_shuffled_ce",
            "normal_minus_null_ce",
            "top1_accuracy",
            "active_hinge_sample_fraction",
            "finite_gradient_behavior",
        ],
        "recommendation": "evidence_only_no_automatic_production_authorization",
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "protocol.json",
        {
            "status": report["status"],
            "version": PILOT_VERSION,
            "report_sha256": _sha256(output / "report.json"),
            "authorizes_training": False,
            "authorizes_joint_training": False,
        },
    )
    return report
