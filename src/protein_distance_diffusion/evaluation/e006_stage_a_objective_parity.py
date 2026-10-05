"""Read-only objective-parity audit for paused E006 Stage-A v5 training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.rich_codesign_production import (
    _authorization,
    _context_monitoring_resources,
    _model,
    _training_rows,
)
from protein_distance_diffusion.training.stage_a_context import (
    CANONICAL_TOKEN_COUNT,
    CANONICAL_TOKEN_START,
    context_corruption,
)

AUDIT_VERSION = "e006_stage_a_v5_objective_parity_audit_v1"
MARGIN_SIGN_CONVENTION = "normal_ce_minus_shuffled_ce; negative favors ordered context"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def published_artifacts(output: str | Path) -> list[str]:
    """List a published audit only when its output directory exists."""
    root = Path(output)
    if not root.is_dir():
        return []
    return [str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    side = max(len(row["sequence"]) for row in rows)
    tokens = torch.zeros((len(rows), side), dtype=torch.long)
    residue_mask = torch.zeros_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        values = torch.as_tensor(row["token_ids"], dtype=torch.long)
        tokens[index, : values.numel()] = values
        residue_mask[index, : values.numel()] = True
    return {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "tokens": tokens,
        "residue_mask": residue_mask,
    }


def validate_paired_context(
    targets: torch.Tensor,
    normal_inputs: torch.Tensor,
    shuffled_inputs: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    mask_token_id: int = 1,
) -> dict[str, Any]:
    """Prove that paired branches differ only by visible-token order."""
    if not all(
        value.shape == targets.shape for value in (normal_inputs, shuffled_inputs, corrupted_mask, residue_mask)
    ):
        raise ValueError("E006 parity tensors have contradictory shapes")
    selected = corrupted_mask.bool() & residue_mask.bool()
    if not selected.any() or (corrupted_mask.bool() & ~residue_mask.bool()).any():
        raise ValueError("E006 parity corrupted-position mask is invalid")
    expected_masks = torch.full_like(normal_inputs[selected], mask_token_id)
    if not torch.equal(normal_inputs[selected], expected_masks) or not torch.equal(
        shuffled_inputs[selected], expected_masks
    ):
        raise ValueError("E006 parity shuffle moved or exposed MASK positions")
    if not torch.equal(normal_inputs[~residue_mask.bool()], targets[~residue_mask.bool()]) or not torch.equal(
        shuffled_inputs[~residue_mask.bool()], targets[~residue_mask.bool()]
    ):
        raise ValueError("E006 parity shuffle changed padding positions")
    visible = residue_mask.bool() & ~corrupted_mask.bool()
    for row in range(targets.shape[0]):
        if not torch.equal(
            torch.sort(normal_inputs[row, visible[row]]).values,
            torch.sort(shuffled_inputs[row, visible[row]]).values,
        ):
            raise ValueError("E006 parity shuffle changed the visible-token multiset")
    canonical_targets = targets[selected]
    if ((canonical_targets < CANONICAL_TOKEN_START) | (canonical_targets >= 22)).any():
        raise ValueError("E006 parity targets are not canonical")
    if torch.equal(normal_inputs[selected], canonical_targets):
        raise ValueError("E006 parity corrupted targets are exposed")
    return {
        "paired_corrupted_masks_identical": True,
        "paired_targets_identical": True,
        "paired_padding_masks_identical": True,
        "mask_positions_fixed": True,
        "padding_positions_fixed": True,
        "visible_token_multisets_preserved": True,
        "target_leakage_absent": True,
        "canonical_corrupted_token_count": int(selected.sum()),
        "valid_token_count": int(residue_mask.sum()),
    }


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = corrupted_mask.bool() & residue_mask.bool()
    canonical_targets = targets[selected].long() - CANONICAL_TOKEN_START
    canonical_logits = logits[selected, CANONICAL_TOKEN_START : CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT]
    return F.cross_entropy(canonical_logits.float(), canonical_targets, reduction="none"), selected


def reduction_diagnostics(
    normal_ce: torch.Tensor,
    shuffled_ce: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    sample_count: int,
    margin_nats: float,
) -> dict[str, Any]:
    """Compare the production batch hinge with less cancellation-prone reductions."""
    if normal_ce.shape != shuffled_ce.shape or normal_ce.ndim != 1:
        raise ValueError("E006 parity token CE arrays have contradictory shapes")
    token_signed = normal_ce - shuffled_ce + margin_nats
    per_sample_normal = []
    per_sample_shuffled = []
    per_sample_hinges = []
    for index in range(sample_count):
        selected = sample_indices == index
        if not selected.any():
            continue
        normal = normal_ce[selected].mean()
        shuffled = shuffled_ce[selected].mean()
        per_sample_normal.append(normal)
        per_sample_shuffled.append(shuffled)
        per_sample_hinges.append(F.relu(normal - shuffled + margin_nats))
    current = F.relu(normal_ce.mean() - shuffled_ce.mean() + margin_nats)
    sample_hinge = torch.stack(per_sample_hinges).mean()
    token_hinge = F.relu(token_signed).mean()
    positive = token_signed > 0
    negative = token_signed < 0
    return {
        "current_hinge_after_batch_token_mean": float(current),
        "hinge_after_batch_mean": float(current),
        "mean_of_per_sample_hinges": float(sample_hinge),
        "mean_of_per_token_hinges": float(token_hinge),
        "active_hinge_token_fraction": float(positive.float().mean()),
        "active_hinge_sample_fraction": float((torch.stack(per_sample_hinges) > 0).float().mean()),
        "easy_token_fraction": float(negative.float().mean()),
        "mixed_sign_token_evidence": bool(positive.any() and negative.any()),
        "easy_examples_cancel_hard_examples": bool(float(current) + 1e-8 < float(token_hinge)),
        "normal_ce": float(normal_ce.mean()),
        "shuffled_ce": float(shuffled_ce.mean()),
        "normal_minus_shuffled_ce": float(normal_ce.mean() - shuffled_ce.mean()),
        "margin_sign_convention": MARGIN_SIGN_CONVENTION,
        "per_sample_normal_ce": [float(value) for value in per_sample_normal],
        "per_sample_shuffled_ce": [float(value) for value in per_sample_shuffled],
        "per_sample_hinge": [float(value) for value in per_sample_hinges],
    }


def _rng_snapshot(device: torch.device) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    return cpu, cuda


def _rng_restore(state: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def evaluate_frozen_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    seed: int,
    corruption_step: int,
    mask_fraction: float,
    margin_nats: float,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one frozen paired batch without gradients or state mutation."""
    targets = batch["tokens"].to(device)
    residue_mask = batch["residue_mask"].to(device)
    corruption = context_corruption(
        targets,
        residue_mask,
        mask_token_id=1,
        probability=mask_fraction,
        seed=seed,
        step=corruption_step,
    )
    invariants = validate_paired_context(
        targets,
        corruption.inputs,
        corruption.shuffled_inputs,
        corruption.corrupted_mask,
        residue_mask,
    )

    def amp_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )

    path_logits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        model.eval()
        monitor_normal = model.forward_sequence_pretraining(corruption.inputs, residue_mask)
        monitor_shuffled = model.forward_sequence_pretraining(corruption.shuffled_inputs, residue_mask)
        path_logits["monitor_eval_fp32"] = (monitor_normal, monitor_shuffled)

        production_fp32_normal = model.forward_sequence_pretraining(corruption.inputs, residue_mask)
        production_fp32_shuffled = model.forward_sequence_pretraining(corruption.shuffled_inputs, residue_mask)
        path_logits["production_eval_fp32"] = (production_fp32_normal, production_fp32_shuffled)

        with amp_context():
            eval_normal = model.forward_sequence_pretraining(corruption.inputs, residue_mask)
            eval_shuffled = model.forward_sequence_pretraining(corruption.shuffled_inputs, residue_mask)
        path_logits["production_eval_amp"] = (eval_normal, eval_shuffled)

        model.train()
        torch.manual_seed(seed + corruption_step * 101)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + corruption_step * 101)
        with amp_context():
            train_normal = model.forward_sequence_pretraining(corruption.inputs, residue_mask)
            train_shuffled = model.forward_sequence_pretraining(corruption.shuffled_inputs, residue_mask)
        path_logits["production_train_sequential_dropout"] = (train_normal, train_shuffled)

        torch.manual_seed(seed + corruption_step * 101)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + corruption_step * 101)
        paired_state = _rng_snapshot(device)
        with amp_context():
            paired_normal = model.forward_sequence_pretraining(corruption.inputs, residue_mask)
            _rng_restore(paired_state)
            paired_shuffled = model.forward_sequence_pretraining(corruption.shuffled_inputs, residue_mask)
        path_logits["train_paired_dropout"] = (paired_normal, paired_shuffled)
        model.eval()

    token_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    paths: dict[str, Any] = {}
    selected_coordinates = torch.nonzero(corruption.corrupted_mask & residue_mask, as_tuple=False)
    sample_indices = selected_coordinates[:, 0]
    for path_name, (normal_logits, shuffled_logits) in path_logits.items():
        normal_ce, selected = token_cross_entropy(normal_logits, targets, corruption.corrupted_mask, residue_mask)
        shuffled_ce, _ = token_cross_entropy(shuffled_logits, targets, corruption.corrupted_mask, residue_mask)
        reductions = reduction_diagnostics(
            normal_ce,
            shuffled_ce,
            sample_indices,
            sample_count=len(batch["sample_ids"]),
            margin_nats=margin_nats,
        )
        paths[path_name] = {key: value for key, value in reductions.items() if not key.startswith("per_sample_")}
        for token_index, ((sample_index, residue_index), normal, shuffled) in enumerate(
            zip(selected_coordinates.tolist(), normal_ce.tolist(), shuffled_ce.tolist(), strict=True)
        ):
            token_rows.append(
                {
                    "path": path_name,
                    "sample_id": batch["sample_ids"][sample_index],
                    "sample_index": sample_index,
                    "residue_index": residue_index,
                    "target_token": int(targets[selected][token_index]),
                    "normal_ce": normal,
                    "shuffled_ce": shuffled,
                    "normal_minus_shuffled_ce": normal - shuffled,
                    "hinge": max(normal - shuffled + margin_nats, 0.0),
                }
            )
        for sample_index, sample_id in enumerate(batch["sample_ids"]):
            sample_rows.append(
                {
                    "path": path_name,
                    "sample_id": sample_id,
                    "normal_ce": reductions["per_sample_normal_ce"][sample_index],
                    "shuffled_ce": reductions["per_sample_shuffled_ce"][sample_index],
                    "hinge": reductions["per_sample_hinge"][sample_index],
                }
            )
    return (
        {
            "invariants": invariants,
            "paths": paths,
            "dropout_disabled_parity": {
                "production_monitor_max_abs_normal_logit_difference": float(
                    (production_fp32_normal - monitor_normal).abs().max()
                ),
                "production_monitor_max_abs_shuffled_logit_difference": float(
                    (production_fp32_shuffled - monitor_shuffled).abs().max()
                ),
                "identical_computation_confirmed": bool(
                    torch.equal(production_fp32_normal, monitor_normal)
                    and torch.equal(production_fp32_shuffled, monitor_shuffled)
                ),
                "production_amp_max_abs_normal_logit_difference": float((eval_normal - monitor_normal).abs().max()),
                "production_amp_max_abs_shuffled_logit_difference": float(
                    (eval_shuffled - monitor_shuffled).abs().max()
                ),
            },
            "train_eval_max_abs_normal_logit_difference": float(
                (path_logits["production_train_sequential_dropout"][0] - monitor_normal).abs().max()
            ),
            "train_eval_max_abs_shuffled_logit_difference": float(
                (path_logits["production_train_sequential_dropout"][1] - monitor_shuffled).abs().max()
            ),
        },
        token_rows,
        sample_rows,
    )


def classify_root_cause(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if any(not all(record["invariants"].values()) for record in records):
        return {"classification": "paired_input_or_target_divergence", "first_divergent_computation": "paired inputs"}
    dropout_difference = max(record["train_eval_max_abs_normal_logit_difference"] for record in records)
    amp_difference = max(
        record["dropout_disabled_parity"]["production_amp_max_abs_normal_logit_difference"] for record in records
    )
    cancellation = any(
        path["easy_examples_cancel_hard_examples"] for record in records for path in record["paths"].values()
    )
    if dropout_difference > 0:
        primary = "train_eval_stochastic_mode_divergence"
        first = "model mode and independent dropout draws before CE reduction"
    elif cancellation:
        primary = "batch_mean_hinge_cancellation"
        first = "hinge reduction order"
    else:
        primary = "panel_generalization_or_weight_effect"
        first = "checkpoint response across distinct panels"
    return {
        "classification": primary,
        "first_divergent_computation": first,
        "independent_dropout_is_present": dropout_difference > 0,
        "mixed_precision_difference_is_present": amp_difference > 0,
        "batch_mean_hinge_cancellation_is_present": cancellation,
        "recommended_correction": (
            "Make the paired objective comparison deterministic with shared stochastic state or eval-mode "
            "counterfactual evaluation, then choose a hinge reduction only after reviewing token/sample evidence."
            if dropout_difference > 0
            else "Review the exported token/sample reduction evidence before changing the objective."
        ),
        "objective_change_implemented": False,
    }


def _checkpoint_position(payload: dict[str, Any]) -> dict[str, int]:
    return {
        name: int(payload.get(name, 0))
        for name in ("optimizer_step", "microstep", "data_cursor", "dataset_pass", "processed_valid_tokens")
    }


def run_objective_parity_audit(config_path: str | Path) -> dict[str, Any]:
    """Run the bounded real-data audit. This function never creates an optimizer."""
    audit_config_path = Path(config_path)
    config = load_yaml(audit_config_path)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 objective-parity output already exists: {output}")
    base_path = Path(config["base_training_config"])
    if _sha256(base_path) != config["base_training_config_sha256"]:
        raise ValueError("E006 objective-parity base configuration hash contradiction")
    base = load_yaml(base_path)
    paused = Path(base["training"]["output_dir"])
    before = _tree_hashes(paused)
    protocol = json.loads((paused / "protocol.json").read_text())
    if protocol.get("status") != "paused_for_scientific_review":
        raise ValueError("E006 objective-parity audit requires a paused scientific-review run")

    checkpoints = []
    for record in config["checkpoints"]:
        path = Path(record["path"])
        if _sha256(path) != record["sha256"]:
            raise ValueError(f"E006 objective-parity checkpoint hash contradiction: {record['name']}")
        checkpoints.append((record, load_checkpoint(path, map_location="cpu")))

    authorization = _authorization(base)
    train = _training_rows(base, authorization, split="train", synthetic=False)
    batch_size = int(config["audit"]["training_batch_size"])
    batch_count = int(config["audit"]["training_batch_count"])
    cursor = _checkpoint_position(checkpoints[-1][1])["data_cursor"]
    start = max(0, cursor - batch_size * batch_count)
    training_rows = [train[index] for index in range(start, min(start + batch_size * batch_count, len(train)))]
    monitor_rows, monitor_identity, _ = _context_monitoring_resources(
        base,
        authorization,
        include_unigram=False,
    )
    expected_monitor_size = int(config["audit"]["monitoring_panel_size"])
    if len(monitor_rows) != expected_monitor_size:
        raise ValueError(
            f"E006 objective-parity monitoring panel has {len(monitor_rows)} rows, expected {expected_monitor_size}"
        )
    train_ids = {row["sample_id"] for row in training_rows}
    if train_ids.intersection(row["sample_id"] for row in monitor_rows):
        raise ValueError("E006 objective-parity train/monitor panels overlap")

    device = torch.device(config.get("device", base.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E006 objective-parity audit requested unavailable CUDA")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    _atomic_json(temporary / "heartbeat.json", {"status": "running", "started_utc": datetime.now(UTC).isoformat()})
    token_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    records = []
    try:
        for checkpoint_record, payload in checkpoints:
            model = _model(base)
            model.load_state_dict(payload["model"])
            model.to(device)
            for panel_name, rows, size in (
                ("training", training_rows, batch_size),
                ("monitoring", monitor_rows, int(config["audit"]["monitoring_batch_size"])),
            ):
                for batch_number, offset in enumerate(range(0, len(rows), size)):
                    batch = _collate(rows[offset : offset + size])
                    evidence, tokens, samples = evaluate_frozen_batch(
                        model,
                        batch,
                        seed=(
                            int(base["seed"])
                            if panel_name == "training"
                            else int(base["contextual_monitoring"]["seed"])
                        ),
                        corruption_step=(
                            int(checkpoints[-1][1].get("microstep", 0)) + batch_number
                            if panel_name == "training"
                            else batch_number
                        ),
                        mask_fraction=float(base["objective"]["mask_fraction"]),
                        margin_nats=float(base["objective"]["context_contrast_margin_nats"]),
                        device=device,
                        use_amp=bool(config["audit"].get("compare_production_amp", True)),
                    )
                    prefix = {
                        "checkpoint": checkpoint_record["name"],
                        "optimizer_step": int(payload.get("optimizer_step", 0)),
                        "panel": panel_name,
                        "batch_number": batch_number,
                    }
                    records.append({**prefix, **evidence})
                    token_rows.extend({**prefix, **row} for row in tokens)
                    sample_rows.extend({**prefix, **row} for row in samples)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pq.write_table(pa.Table.from_pylist(token_rows), temporary / "per_token_ce.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(sample_rows), temporary / "per_sample_ce.parquet", compression="zstd")
        after = _tree_hashes(paused)
        if before != after:
            raise RuntimeError("E006 objective-parity audit mutated the paused training output")
        report = {
            "version": AUDIT_VERSION,
            "status": "completed",
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "training_performed": False,
            "optimizer_updates": 0,
            "base_training_config": str(base_path),
            "base_training_config_sha256": config["base_training_config_sha256"],
            "checkpoints": [
                {
                    **record,
                    "position": _checkpoint_position(payload),
                }
                for record, payload in checkpoints
            ],
            "panels": {
                "training": {
                    "sample_count": len(training_rows),
                    "sample_id_sha256": _canonical_hash([row["sample_id"] for row in training_rows]),
                    "cursor_window_start": start,
                },
                "monitoring": monitor_identity,
                "disjoint": True,
            },
            "paired_comparisons": records,
            "root_cause": classify_root_cause(records),
            "exports": {
                "per_token_ce": "per_token_ce.parquet",
                "per_sample_ce": "per_sample_ce.parquet",
                "token_row_count": len(token_rows),
                "sample_row_count": len(sample_rows),
            },
            "paused_output_hashes_before_sha256": _canonical_hash(before),
            "paused_output_hashes_after_sha256": _canonical_hash(after),
            "protected_inputs_unchanged": True,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(temporary / "report.json", report)
        protocol_payload = {
            "version": AUDIT_VERSION,
            "status": "completed",
            "report_sha256": _sha256(temporary / "report.json"),
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "training_performed": False,
            "optimizer_updates": 0,
            "protected_inputs_unchanged": True,
        }
        _atomic_json(temporary / "protocol.json", protocol_payload)
        _atomic_json(
            temporary / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": report["completed_utc"],
                "report_path": str(output / "report.json"),
                "report_sha256": protocol_payload["report_sha256"],
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
