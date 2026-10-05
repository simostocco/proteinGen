"""Authorized staged E006 training and bounded calibration infrastructure."""

from __future__ import annotations

import hashlib
import io
import json
import math
import operator
import os
import random
import signal
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.rich_geometry import (
    RICH_FEATURE_VERSION,
    RichDatasetAuthorization,
    RichGeometryDataset,
    authorize_rich_geometry_dataset,
    collate_rich_geometry,
    deterministic_length_bucket_sample,
)
from protein_distance_diffusion.models.rich_codesign import (
    E006_ARCHITECTURE_VERSION,
    E006RichGeometryCoDesign,
)
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.codesign import masked_sequence_inputs
from protein_distance_diffusion.training.rich_checkpoint_policy import (
    finalize_best_checkpoint,
    publish_best_checkpoint,
    publish_immutable_checkpoint,
    publish_recovery_checkpoint,
    require_storage_preflight,
    storage_preflight,
    verify_recovery_checkpoint,
)
from protein_distance_diffusion.training.rich_codesign_selection import verify_production_selection
from protein_distance_diffusion.training.rich_codesign_smoke import (
    _forward,
    _memory,
    _metrics,
    _model,
    _move,
    _parameter_groups,
    _sha256,
    _synthetic_row,
    stratified_metrics,
)
from protein_distance_diffusion.training.stage_a_context import (
    STAGE_A_CONTEXT_OBJECTIVE_V6,
    STAGE_A_CONTEXT_OBJECTIVE_VERSION,
    canonical_corrupted_cross_entropy,
    context_corruption,
    contextual_stage_a_loss,
    contextual_stage_a_loss_v6,
    paired_dropout_forwards,
)
from protein_distance_diffusion.training.stage_a_context_production import (
    append_rolling_training_metric,
    contextual_monitoring_steps,
    contextual_selection_key,
    evaluate_context_monitor,
    load_warm_start_weights_only,
    select_monitoring_panel,
    training_unigram,
    verify_review_decision,
    verify_v5_pretraining_gates,
)
from protein_distance_diffusion.training.trainer import _restore_rng_state, _rng_state

PHASE3_PROTOCOL_VERSION = "e006_staged_training_v1"
CALIBRATION_VERSION = "e006_production_calibration_v1"
STAGE_A_CHECKPOINT_VERSION = "e006_sequence_pretrain_checkpoint_v1"
STAGE_B_CHECKPOINT_VERSION = "e006_joint_training_checkpoint_v1"
LENGTH_REGIMES = (128, 256, 384, 500)
TRAINING_MODES = ("sequence-pretrain", "joint-train")
MAXIMUM_GRADIENT_DIAGNOSTIC_NAMES = 20
CONTINUATION_PROTOCOL_VERSION = "e006_stage_a_audited_continuation_v1"


class E006GradientError(FloatingPointError):
    """Gradient failure carrying bounded machine-readable evidence."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class TrainingInterrupted(RuntimeError):
    """Raised after a resumable checkpoint is safely published."""


@dataclass(frozen=True)
class BatchRecommendation:
    maximum_length: int
    physical_batch_size: int
    accumulation_steps: int
    maximum_pair_elements: int
    effective_token_budget: int | None = None
    constructs_pair_features: bool = True


class _LazySelectedRows(Sequence[dict[str, Any]]):
    def __init__(
        self,
        dataset: RichGeometryDataset,
        indices: list[int],
        lengths: list[int],
        sample_ids: list[str],
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.lengths = tuple(lengths)
        self.sample_ids = tuple(sample_ids)
        if not (len(self.indices) == len(self.lengths) == len(self.sample_ids)):
            raise ValueError("E006 selected-row length metadata contradiction")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("E006 selected-row indices must be scalar integers, not booleans")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise TypeError(
                f"E006 selected-row indices must be scalar integers; received {type(index).__name__}"
            ) from error
        return self.dataset[self.indices[index]]


def build_validation_panel(
    validation_rows: Sequence[dict[str, Any]],
    panel_size: int,
) -> list[dict[str, Any]]:
    """Materialize a bounded deterministic panel using scalar dataset access only."""
    if isinstance(panel_size, bool) or not isinstance(panel_size, int) or panel_size < 1:
        raise ValueError("E006 validation panel size must be a positive integer")
    return [validation_rows[index] for index in range(min(panel_size, len(validation_rows)))]


def _validation_panel_preflight(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    panel_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel = build_validation_panel(validation_rows, panel_size)
    sample_ids = [str(row["sample_id"]) for row in panel]
    unique_ids = set(sample_ids)
    if len(panel) != panel_size:
        raise ValueError(f"E006 validation panel has {len(panel)} rows, expected {panel_size}")
    if len(unique_ids) != panel_size:
        raise ValueError("E006 validation panel sample IDs are not unique")
    train_ids = (
        set(train_rows.sample_ids)
        if isinstance(train_rows, _LazySelectedRows)
        else {str(row["sample_id"]) for row in train_rows}
    )
    if train_ids.intersection(unique_ids):
        raise ValueError("E006 training and validation panel membership overlap")
    return panel, {
        "requested_sample_count": panel_size,
        "sample_count": len(panel),
        "unique_sample_count": len(unique_ids),
        "sample_id_sha256": _canonical_hash(sample_ids),
        "scalar_indexed": True,
        "train_validation_disjoint": True,
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _torch_state_hash(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _directory_fingerprint(path: str | Path) -> dict[str, str]:
    root = Path(path)
    return {
        str(item.relative_to(root)): _sha256(item)
        for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    }


def _training_config_hash(config: dict[str, Any]) -> str:
    scientific = json.loads(json.dumps(config))
    scientific.get("training", {}).pop("interrupt_after_optimizer_steps", None)
    return _canonical_hash(scientific)


def _scientific_continuation_config(config: dict[str, Any]) -> dict[str, Any]:
    scientific = json.loads(json.dumps(config))
    scientific.pop("experiment", None)
    scientific.pop("continuation", None)
    training = scientific.get("training", {})
    training.pop("output_dir", None)
    for name in (
        "maximum_total_amp_overflows",
        "maximum_consecutive_amp_overflows",
        "maximum_overflow_diagnostic_examples",
        "interrupt_after_optimizer_steps",
    ):
        training.pop(name, None)
    return scientific


def verify_continuation_source(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a historical recovery boundary for explicit cross-run continuation."""
    continuation = config.get("continuation")
    if not isinstance(continuation, dict) or continuation.get("mode") != CONTINUATION_PROTOCOL_VERSION:
        raise ValueError("E006 audited continuation configuration is missing or invalid")
    source_output = Path(continuation["source_output_dir"])
    checkpoint = Path(continuation["source_checkpoint_path"])
    metadata_path = Path(continuation["source_checkpoint_metadata_path"])
    manifest_path = Path(continuation["source_checkpoint_manifest_path"])
    validation_path = Path(continuation["source_validation_journal_path"])
    source_config_path = Path(continuation["source_config_path"])
    best_path = Path(continuation["source_best_checkpoint_path"])
    best_metadata_path = Path(continuation["source_best_metadata_path"])
    expected_files = {
        checkpoint: continuation["source_checkpoint_sha256"],
        metadata_path: continuation["source_checkpoint_metadata_sha256"],
        manifest_path: continuation["source_checkpoint_manifest_sha256"],
        validation_path: continuation["source_validation_journal_sha256"],
        source_config_path: continuation["source_config_file_sha256"],
        best_path: continuation["source_best_checkpoint_sha256"],
        best_metadata_path: continuation["source_best_metadata_sha256"],
    }
    for path, expected_sha in expected_files.items():
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"E006 continuation source hash contradiction: {path}")
    if checkpoint.parent.parent != source_output or metadata_path.parent.parent != source_output:
        raise ValueError("E006 continuation checkpoint is outside its declared source output")
    metadata = json.loads(metadata_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if metadata.get("sha256") != expected_files[checkpoint]:
        raise ValueError("E006 continuation latest metadata SHA contradiction")
    rolling = manifest.get("rolling_latest", {})
    if rolling.get("sha256") != expected_files[checkpoint] or rolling.get("path") != str(checkpoint):
        raise ValueError("E006 continuation checkpoint manifest contradiction")
    verify_recovery_checkpoint(checkpoint.parent)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    source_config = load_yaml(source_config_path)
    source_config_hash = _training_config_hash(source_config)
    if payload.get("config_sha256") != source_config_hash:
        raise ValueError("E006 continuation source configuration hash contradiction")
    if _scientific_continuation_config(source_config) != _scientific_continuation_config(config):
        raise ValueError("E006 continuation scientific configuration mismatch")
    expected_state = {
        "optimizer_step": int(continuation["source_optimizer_step"]),
        "microstep": int(continuation["source_microstep"]),
        "data_cursor": int(continuation["source_data_cursor"]),
        "dataset_pass": int(continuation["source_dataset_pass"]),
        "processed_valid_tokens": int(continuation["source_processed_valid_tokens"]),
    }
    contradictions = [name for name, expected in expected_state.items() if payload.get(name) != expected]
    for name in ("optimizer_step", "dataset_pass", "processed_valid_tokens"):
        if metadata.get(name) != expected_state[name] or rolling.get(name) != expected_state[name]:
            contradictions.append(f"metadata_{name}")
    accumulation = payload.get("accumulation_state", {})
    sampler = payload.get("sampler_state", {})
    if accumulation.get("at_optimizer_boundary") is not True or accumulation.get("microbatches_accumulated") != 0:
        contradictions.append("accumulation_state")
    if accumulation.get("microstep") != payload.get("microstep"):
        contradictions.append("accumulation_microstep")
    if sampler.get("data_cursor") != payload.get("data_cursor") or sampler.get("dataset_pass") != payload.get(
        "dataset_pass"
    ):
        contradictions.append("sampler_state")
    scheduler = payload.get("scheduler", {})
    if (
        scheduler.get("last_epoch") != payload.get("optimizer_step")
        or scheduler.get("_step_count") != int(payload.get("optimizer_step", -1)) + 1
    ):
        contradictions.append("scheduler_state")
    if not payload.get("optimizer", {}).get("param_groups") or not isinstance(payload.get("scaler"), dict):
        contradictions.append("optimizer_or_scaler_state")
    if set(payload.get("rng_state", {})) != {"python", "numpy", "torch", "cuda"}:
        contradictions.append("rng_state")
    scaler_scale = payload.get("scaler", {}).get("scale")
    if scaler_scale is not None and (not math.isfinite(float(scaler_scale)) or float(scaler_scale) <= 0):
        contradictions.append("scaler_state")
    if not payload.get("model"):
        contradictions.append("model_state")
    if payload.get("status") != "recovery_only" or payload.get("authorizes_training") is not False:
        contradictions.append("source_authorization")
    if contradictions:
        raise ValueError(f"E006 continuation source state contradiction: {', '.join(contradictions)}")
    validations = [json.loads(line) for line in validation_path.read_text().splitlines() if line.strip()]
    validation_steps = [int(item["optimizer_step"]) for item in validations]
    expected_validation_steps = [int(value) for value in continuation["source_validation_steps"]]
    if validation_steps != expected_validation_steps:
        raise ValueError("E006 continuation validation history contradiction")
    best_metadata = json.loads(best_metadata_path.read_text())
    if int(best_metadata.get("optimizer_step", -1)) != int(continuation["source_best_optimizer_step"]) or float(
        best_metadata.get("validation_sequence_cross_entropy", math.inf)
    ) != float(continuation["source_best_sequence_cross_entropy"]):
        raise ValueError("E006 continuation best-incumbent contradiction")
    protocol = {
        "version": CONTINUATION_PROTOCOL_VERSION,
        "status": "source_verified",
        "source_output_dir": str(source_output),
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": expected_files[checkpoint],
        "source_configuration_path": str(source_config_path),
        "source_configuration_file_sha256": expected_files[source_config_path],
        "source_configuration_sha256": source_config_hash,
        "destination_configuration_sha256": _training_config_hash(config),
        "scientific_settings_equal": True,
        "scientific_settings_sha256": _canonical_hash(_scientific_continuation_config(config)),
        "implementation_difference": "gradient_and_amp_overflow_control_only",
        "imported_state": expected_state,
        "source_amp_scale": (float(payload["scaler"]["scale"]) if payload["scaler"].get("scale") is not None else None),
        "clean_accumulation_boundary": True,
        "discarded_uncheckpointed_optimizer_updates": int(continuation["discarded_uncheckpointed_updates"]),
        "validation_history": {
            "path": str(validation_path),
            "sha256": expected_files[validation_path],
            "optimizer_steps": validation_steps,
        },
        "imported_best_incumbent": {
            "optimizer_step": int(continuation["source_best_optimizer_step"]),
            "validation_sequence_cross_entropy": float(continuation["source_best_sequence_cross_entropy"]),
            "checkpoint_path": str(best_path),
            "checkpoint_sha256": expected_files[best_path],
        },
        "next_scheduled_validation_step": int(continuation["next_scheduled_validation_step"]),
        "dataset_pass_end_steps": [int(value) for value in continuation["dataset_pass_end_steps"]],
        "total_target_optimizer_steps": int(continuation["total_target_optimizer_steps"]),
        "source_artifact_fingerprints": _directory_fingerprint(source_output),
        "authorizes_training": False,
        "authorizes_joint_training": False,
    }
    return payload, protocol


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _authorization(config: dict[str, Any]) -> RichDatasetAuthorization:
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["directory"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
    )


def _dataset_identity(authorization: RichDatasetAuthorization | None) -> str:
    if authorization is None:
        return "synthetic-e006-rich-v1"
    return _canonical_hash(
        {
            "protocol": authorization.protocol_sha256,
            "schema": authorization.schema_sha256,
            "vocabulary": authorization.vocabulary_sha256,
            "normalization": authorization.normalization_sha256,
            "inventory": authorization.shard_inventory_sha256,
            "shards": authorization.observed_shard_hashes,
        }
    )


def _protected_hashes(authorization: RichDatasetAuthorization | None) -> dict[str, str]:
    if authorization is None:
        return {}
    return {
        "protocol.json": authorization.protocol_sha256,
        "schema.json": authorization.schema_sha256,
        "vocabulary.json": authorization.vocabulary_sha256,
        "normalization.json": authorization.normalization_sha256,
        "shard_hashes.sha256": authorization.shard_inventory_sha256,
        **authorization.observed_shard_hashes,
    }


def validate_phase3_config(config: dict[str, Any], *, mode: str, synthetic: bool = False) -> None:
    if mode not in {"calibrate", *TRAINING_MODES}:
        raise ValueError(f"Unknown E006 Phase-3 mode: {mode}")
    if config.get("architecture_version") != E006_ARCHITECTURE_VERSION:
        raise ValueError("E006 Phase-3 architecture version contradiction")
    if config.get("feature_version") != RICH_FEATURE_VERSION:
        raise ValueError("E006 Phase-3 feature version contradiction")
    if int(config.get("seed", -1)) < 0:
        raise ValueError("E006 Phase-3 seed must be nonnegative")
    limits = config.get("memory", {})
    if float(limits.get("maximum_rss_mib", 0)) <= 0 or float(limits.get("maximum_cuda_reserved_mib", 0)) <= 0:
        raise ValueError("E006 Phase-3 memory limits must be positive")
    if float(limits.get("maximum_cuda_allocated_mib", 0)) <= 0:
        raise ValueError("E006 Phase-3 CUDA allocated-memory limit must be positive")
    if not 0 <= float(config.get("objective", {}).get("conditioning_dropout_probability", 0)) <= 1:
        raise ValueError("E006 conditioning dropout must be in [0, 1]")
    objective = config.get("objective", {})
    objective_version = objective.get("version")
    contextual_versions = (STAGE_A_CONTEXT_OBJECTIVE_VERSION, STAGE_A_CONTEXT_OBJECTIVE_V6)
    if objective_version not in (None, *contextual_versions):
        raise ValueError(f"Unsupported E006 Stage-A objective version: {objective_version}")
    if objective_version in contextual_versions:
        if objective.get("loss_positions") != "corrupted_canonical_only":
            raise ValueError("E006 Stage-A v5 must train only on corrupted canonical positions")
        if objective.get("corruption_state") != "explicit_mask_token":
            raise ValueError("E006 Stage-A v5 requires an explicit mask-token corruption state")
        if float(objective.get("context_contrast_weight", -1)) < 0:
            raise ValueError("E006 Stage-A v5 context contrast weight must be nonnegative")
        if float(objective.get("context_contrast_margin_nats", -1)) < 0:
            raise ValueError("E006 Stage-A v5 context contrast margin must be nonnegative")
        if objective_version == STAGE_A_CONTEXT_OBJECTIVE_V6 and (
            objective.get("context_hinge_reduction") != "equal_weight_per_sample"
            or objective.get("dropout_pairing") != "cpu_and_all_cuda_rng_states"
        ):
            raise ValueError("E006 Stage-A v6 objective reduction/dropout-pairing contract is invalid")
        if (
            mode == "sequence-pretrain"
            and not synthetic
            and config.get("production_monitoring_version")
            in {"e006_stage_a_v5_monitored_training_v1", "e006_stage_a_v6_monitored_training_v1"}
        ):
            initialization = config.get("initialization", {})
            if initialization.get("mode") != "checkpoint_weights_only":
                raise ValueError("E006 Stage-A v5 production requires weights-only initialization")
            if not initialization.get("checkpoint_path") or not initialization.get("checkpoint_sha256"):
                raise ValueError("E006 Stage-A v5 warm-start checkpoint is not hash-pinned")
            monitoring = config.get("contextual_monitoring", {})
            if int(monitoring.get("panel_size", 0)) < 1 or int(monitoring.get("maximum_length", 0)) > 128:
                raise ValueError("E006 Stage-A v5 contextual monitoring panel is invalid")
            if int(monitoring.get("rolling_metric_frequency", 0)) > 50:
                raise ValueError("E006 Stage-A v5 rolling metrics must be written at least every 50 updates")
            if config.get("training", {}).get("review_pause_steps") != [2500]:
                raise ValueError("E006 Stage-A v5 requires the immutable step-2500 review pause")
            post = config.get("post_training_context_diagnostic", {})
            if (
                post.get("required_for_stage_b") is not True
                or post.get("required_for_definitive_evaluation") is not True
            ):
                raise ValueError("E006 Stage-A v5 requires a post-training contextual diagnostic")
    if mode == "calibrate":
        if config.get("calibration", {}).get("version") == "e006_production_calibration_v2":
            from protein_distance_diffusion.training.rich_codesign_calibration_v2 import (
                validate_calibration_v2_config,
            )

            validate_calibration_v2_config(config)
            return
        regimes = config.get("calibration", {}).get("regimes", [])
        if tuple(int(item["target_length"]) for item in regimes) != LENGTH_REGIMES:
            raise ValueError(f"E006 calibration must cover length regimes {LENGTH_REGIMES}")
        for regime in regimes:
            candidates = regime.get("candidates", [])
            if not candidates or any(
                int(item.get("physical_batch_size", 0)) < 1 or int(item.get("accumulation_steps", 0)) < 1
                for item in candidates
            ):
                raise ValueError("E006 calibration candidates must have positive batch and accumulation sizes")
        return
    training = config.get("training", {})
    if int(training.get("dataset_passes", 0)) < 1:
        raise ValueError("E006 training requires at least one complete dataset pass")
    if int(training.get("recovery_checkpoint_frequency", 0)) < 1 or int(training.get("validation_frequency", 0)) < 1:
        raise ValueError("E006 recovery-checkpoint and validation frequencies must be positive")
    for name in (
        "immutable_checkpoint_on_validation",
        "immutable_checkpoint_on_pass_end",
        "maintain_best_checkpoint",
    ):
        if training.get(name) is not True:
            raise ValueError(f"E006 production checkpoint policy requires {name}=true")
    if int(training.get("estimated_checkpoint_bytes", 0)) < 1:
        raise ValueError("E006 estimated checkpoint bytes must be positive")
    if float(training.get("minimum_free_disk_gib", 0)) < 25:
        raise ValueError("E006 minimum free-disk reserve must be at least 25 GiB")
    if "checkpoint_frequency" in training:
        raise ValueError("E006 checkpoint_frequency is ambiguous; use recovery_checkpoint_frequency")
    for name in ("maximum_total_amp_overflows", "maximum_consecutive_amp_overflows"):
        if int(training.get(name, 20 if "consecutive" in name else 1000)) < 1:
            raise ValueError(f"E006 {name} must be a positive integer")
    if int(training.get("maximum_overflow_diagnostic_examples", 20)) < 1:
        raise ValueError("E006 maximum_overflow_diagnostic_examples must be positive")
    optimizer = config.get("optimizer", {})
    if optimizer.get("name") != "AdamW" or float(optimizer.get("learning_rate", 0)) <= 0:
        raise ValueError("E006 Phase-3 requires AdamW and a positive learning rate")
    if float(optimizer.get("gradient_clip_norm", 0)) <= 0:
        raise ValueError("E006 gradient clipping must be positive")
    if not synthetic and not config.get("calibration", {}).get("report_sha256"):
        raise ValueError("E006 production training requires a completed calibration report SHA-256")
    if not synthetic:
        calibration = config.get("calibration", {})
        if not calibration.get("production_selection_path") or not calibration.get("production_selection_sha256"):
            raise ValueError("E006 production training requires a pinned production-selection artifact")
        regimes = config.get("batching", {}).get("regimes", [])
        if tuple(int(item.get("maximum_length", -1)) for item in regimes) != LENGTH_REGIMES:
            raise ValueError(f"E006 production batching must cover length regimes {LENGTH_REGIMES}")
        if any(
            int(item.get("physical_batch_size", 0)) < 1
            or int(item.get("accumulation_steps", 0)) < 1
            or int(item.get("effective_token_budget", 0)) < 1
            for item in regimes
        ):
            raise ValueError("E006 production batch budgets must be positive")
        if config["batching"].get("loss_normalization") != "valid_tokens_and_valid_pairs":
            raise ValueError("E006 production losses must retain valid-token and valid-pair normalization")
        if config["training"].get("progress_units") != [
            "optimizer_steps",
            "processed_valid_tokens",
            "dataset_passes",
        ]:
            raise ValueError("E006 production progress units must include steps, valid tokens, and dataset passes")
    if mode == "joint-train" and not config.get("stage_a", {}).get("checkpoint_sha256"):
        raise ValueError("E006 joint training requires an authorized Stage-A checkpoint SHA-256")
    if mode == "joint-train" and not synthetic and config.get("stage_a", {}).get("context_diagnostic_required", False):
        stage_a = config["stage_a"]
        if not stage_a.get("context_diagnostic_path") or not stage_a.get("context_diagnostic_sha256"):
            raise ValueError("E006 joint training requires a pinned Stage-A contextual diagnostic")
        if not stage_a.get("acceptable_context_classifications"):
            raise ValueError("E006 joint training requires acceptable contextual classifications")


def verify_stage_a_v5_launch_gates(config: dict[str, Any]) -> dict[str, Any]:
    """Verify every non-authorizing prerequisite before definitive v5 training."""
    objective_version = config.get("objective", {}).get("version")
    if objective_version not in {STAGE_A_CONTEXT_OBJECTIVE_VERSION, STAGE_A_CONTEXT_OBJECTIVE_V6}:
        return {"required": False, "passed": True, "artifacts": {}}
    if config.get("production_monitoring_version") == "e006_stage_a_v5_monitored_training_v1":
        return verify_v5_pretraining_gates(config)
    specifications = config.get("gate_artifacts", {})
    if objective_version == STAGE_A_CONTEXT_OBJECTIVE_V6:
        required = ("parity_audit", "synthetic_context_smoke", "comparison_pilot")
        verified = {}
        for name in required:
            record = specifications.get(name) or {}
            if not record.get("path") or not record.get("sha256"):
                raise ValueError(f"E006 Stage-A v6 launch gate is not pinned: {name}")
            path = Path(record["path"])
            if not path.is_file() or _sha256(path) != record["sha256"]:
                raise ValueError(f"E006 Stage-A v6 launch-gate hash contradiction: {name}")
            payload = json.loads(path.read_text())
            if payload.get("status") != "completed" or payload.get("authorizes_training") is not False:
                raise ValueError(f"E006 Stage-A v6 prerequisite did not pass: {name}")
            if name != "parity_audit" and payload.get("gates", {}).get("passed") is not True:
                raise ValueError(f"E006 Stage-A v6 evidence gate did not pass: {name}")
            verified[name] = {"path": str(path), "sha256": record["sha256"], "status": payload["status"]}
        return {"required": True, "passed": True, "artifacts": verified}
    required = ("synthetic_context_smoke", "real_loader_smoke", "comparison_pilot", "context_diagnostic")
    verified = {}
    for name in required:
        record = specifications.get(name) or {}
        if not record.get("path") or not record.get("sha256"):
            raise ValueError(f"E006 Stage-A v5 launch gate is not pinned: {name}")
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"E006 Stage-A v5 launch-gate hash contradiction: {name}")
        payload = json.loads(path.read_text())
        if payload.get("status") not in set(record.get("acceptable_statuses", ["completed", "passed"])):
            raise ValueError(f"E006 Stage-A v5 launch gate did not pass: {name}")
        if payload.get("authorizes_training") is not False:
            raise ValueError(f"E006 Stage-A v5 prerequisite must be non-authorizing: {name}")
        if name == "context_diagnostic" and payload.get("classification") != "contextual_learning_verified":
            raise ValueError("E006 Stage-A v5 contextual diagnostic has not verified contextual learning")
        verified[name] = {"path": str(path), "sha256": record["sha256"], "status": payload["status"]}
    return {"required": True, "passed": True, "artifacts": verified}


def _assert_finite_model(model: torch.nn.Module) -> None:
    invalid = [name for name, value in model.named_parameters() if not torch.isfinite(value).all()]
    if invalid:
        raise FloatingPointError(f"E006 non-finite parameters: {', '.join(invalid[:10])}")


def _assert_finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError(f"E006 non-finite optimizer state: {name}")


def _expected_active_parameter_names(model: E006RichGeometryCoDesign, stage: str) -> set[str]:
    if stage == "joint-train":
        return {name for name, _ in model.named_parameters()}
    if stage != "sequence-pretrain":
        raise ValueError(f"Unknown E006 training stage: {stage}")
    prefixes = (
        "token_embedding.",
        "position_embedding.",
        "sequence_layers.",
        "sequence_norm.",
        "sequence_output.",
    )
    return {name for name, _ in model.named_parameters() if name.startswith(prefixes)}


def _gradient_evidence(
    model: E006RichGeometryCoDesign,
    *,
    stage: str,
    limit: int = MAXIMUM_GRADIENT_DIAGNOSTIC_NAMES,
) -> dict[str, Any]:
    expected = _expected_active_parameter_names(model, stage)
    named = dict(model.named_parameters())
    missing = sorted(name for name in expected if not named[name].requires_grad or named[name].grad is None)
    nonfinite = sorted(
        name for name in expected if named[name].grad is not None and not torch.isfinite(named[name].grad).all()
    )
    inactive_with_grad = sorted(
        name for name, parameter in named.items() if name not in expected and parameter.grad is not None
    )
    return {
        "expected_active_parameter_count": len(expected),
        "missing_active_gradient_count": len(missing),
        "missing_active_gradient_names": missing[:limit],
        "missing_active_gradient_names_truncated": len(missing) > limit,
        "nonfinite_gradient_count": len(nonfinite),
        "nonfinite_gradient_names": nonfinite[:limit],
        "nonfinite_gradient_names_truncated": len(nonfinite) > limit,
        "inactive_parameter_with_gradient_count": len(inactive_with_grad),
        "inactive_parameter_with_gradient_names": inactive_with_grad[:limit],
    }


def _require_finite_training_loss(total: torch.Tensor, batch_evidence: dict[str, Any]) -> None:
    if not torch.isfinite(total):
        raise E006GradientError(
            "E006 non-finite training loss",
            {**batch_evidence, "failure_kind": "nonfinite_loss", "total_loss": float(total.detach())},
        )


def _optimizer_boundary_update(
    *,
    model: E006RichGeometryCoDesign,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    amp_enabled: bool,
    stage: str,
    gradient_clip_norm: float,
    amp_overflows_total: int,
    amp_overflows_consecutive: int,
    maximum_total_amp_overflows: int,
    maximum_consecutive_amp_overflows: int,
    diagnostic_limit: int = MAXIMUM_GRADIENT_DIAGNOSTIC_NAMES,
) -> dict[str, Any]:
    """Validate stage participation and apply one AMP-aware optimizer boundary."""
    old_scale = float(scaler.get_scale()) if amp_enabled else None
    if amp_enabled:
        scaler.unscale_(optimizer)
    evidence = _gradient_evidence(model, stage=stage, limit=diagnostic_limit)
    if evidence["missing_active_gradient_count"]:
        raise E006GradientError("E006 missing gradients on active parameters", evidence)
    nonfinite = bool(evidence["nonfinite_gradient_count"])
    if nonfinite and not amp_enabled:
        raise E006GradientError("E006 non-finite gradients with AMP disabled", evidence)

    gradient_norms = {}
    for name, group in _parameter_groups(model).items():
        values = [parameter.grad for parameter in group if parameter.requires_grad and parameter.grad is not None]
        gradient_norms[name] = (
            math.sqrt(sum(float(value.detach().float().square().sum()) for value in values))
            if values and all(torch.isfinite(value).all() for value in values)
            else None
        )
    if nonfinite:
        grad_norm_value = math.inf
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            gradient_clip_norm,
            error_if_nonfinite=False,
        )
        grad_norm_value = float(grad_norm.detach().float().cpu())
    if amp_enabled:
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        skipped = new_scale < float(old_scale)
        if nonfinite and not skipped:
            raise E006GradientError(
                "E006 non-finite gradients were not classified as an AMP overflow",
                {**evidence, "amp_scale_before": old_scale, "amp_scale_after": new_scale},
            )
    else:
        optimizer.step()
        new_scale = None
        skipped = False
    optimizer.zero_grad(set_to_none=True)

    if skipped:
        amp_overflows_total += 1
        amp_overflows_consecutive += 1
    else:
        amp_overflows_consecutive = 0
    return {
        "update_skipped": skipped,
        "amp_scale_before": old_scale,
        "amp_scale_after": new_scale,
        "amp_overflows_total": amp_overflows_total,
        "amp_overflows_consecutive": amp_overflows_consecutive,
        "overflow_limit_exceeded": (
            amp_overflows_total > maximum_total_amp_overflows
            or amp_overflows_consecutive > maximum_consecutive_amp_overflows
        ),
        "gradient_norm_preclip": grad_norm_value,
        "gradient_norms": gradient_norms,
        "gradient_evidence": evidence,
    }


def _memory_guard(config: dict[str, Any], device: torch.device) -> dict[str, float | None]:
    memory = _memory(device)
    limits = config["memory"]
    if float(memory["peak_rss_mib"] or 0) > float(limits["maximum_rss_mib"]):
        raise MemoryError("E006 host RSS limit exceeded")
    if (
        device.type == "cuda"
        and max(
            float(memory["peak_cuda_allocated_mib"] or 0) / float(limits["maximum_cuda_allocated_mib"]),
            float(memory["peak_cuda_reserved_mib"] or 0) / float(limits["maximum_cuda_reserved_mib"]),
        )
        > 1.0
    ):
        raise MemoryError("E006 CUDA allocated or reserved memory limit exceeded")
    return memory


def select_calibration_recommendations(
    cases: Iterable[dict[str, Any]],
    *,
    total_vram_mib: float,
    safety_headroom_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    """Select the largest finite case below the required VRAM envelope."""
    if not 0 < safety_headroom_fraction < 1 or not math.isfinite(total_vram_mib):
        raise ValueError("Invalid E006 calibration headroom inputs")
    ceiling = total_vram_mib * (1.0 - safety_headroom_fraction)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(int(case["target_length"]), []).append(case)
    recommendations = []
    for length in LENGTH_REGIMES:
        candidates = [
            item
            for item in grouped.get(length, [])
            if item.get("status") == "passed"
            and item.get("finite") is True
            and float(item["peak_cuda_reserved_mib"]) <= ceiling
        ]
        if not candidates:
            raise ValueError(f"No safe E006 calibration case for length {length}")
        selected = max(
            candidates,
            key=lambda item: (
                int(item["physical_batch_size"]),
                int(item["accumulation_steps"]),
                -float(item["peak_cuda_reserved_mib"]),
            ),
        )
        recommendations.append(
            {
                "maximum_length": length,
                "physical_batch_size": int(selected["physical_batch_size"]),
                "accumulation_steps": int(selected["accumulation_steps"]),
                "maximum_pair_elements": int(selected["pair_elements"]),
                "effective_token_budget": (
                    length * int(selected["physical_batch_size"]) * int(selected["accumulation_steps"])
                ),
                "safety_headroom_fraction": safety_headroom_fraction,
            }
        )
    return recommendations


def execute_calibration_regime(
    candidates: Iterable[dict[str, int]],
    executor: Callable[[dict[str, int]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run bounded candidates and stop larger batches after a handled CUDA OOM."""
    results = []
    for candidate in sorted(candidates, key=lambda item: int(item["physical_batch_size"])):
        try:
            result = executor(candidate)
        except torch.OutOfMemoryError as error:
            result = {**candidate, "status": "cuda_oom", "finite": False, "error": str(error)[:500]}
            results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        results.append({**candidate, **result})
    return results


def nearest_length_indices(
    dataset: RichGeometryDataset,
    *,
    target_length: int,
    count: int,
    seed: int,
) -> list[int]:
    """Select deterministic valid proteins nearest a calibration length."""
    retained: list[tuple[int, str, int]] = []
    for index, sample_id, length in dataset.iter_metadata():
        rank = hashlib.sha256(f"{seed}:{dataset.split}:{sample_id}".encode()).hexdigest()
        retained.append((abs(length - target_length), rank, index))
        retained.sort()
        del retained[count:]
    if len(retained) != count:
        raise ValueError(f"E006 calibration requested {count} rows but found {len(retained)}")
    return [index for _, _, index in retained]


def recommendation_for_length(
    length: int,
    recommendations: Iterable[dict[str, Any]],
    *,
    stage: str | None = None,
) -> BatchRecommendation:
    values = [
        item for item in recommendations if item.get("stage") is None or item.get("stage") == (stage or "joint-train")
    ]
    for item in sorted(values, key=lambda value: int(value["maximum_length"])):
        if length <= int(item["maximum_length"]):
            return BatchRecommendation(
                int(item["maximum_length"]),
                int(item["physical_batch_size"]),
                int(item["accumulation_steps"]),
                int(item.get("maximum_pair_elements") or 0),
                int(item["effective_token_budget"]) if item.get("effective_token_budget") is not None else None,
                item.get("stage") != "sequence-pretrain",
            )
    raise ValueError(f"No calibrated E006 batch recommendation covers length {length}")


def validate_batch_budget(
    lengths: Iterable[int],
    *,
    recommendation: BatchRecommendation,
    maximum_residues: int,
    maximum_pair_elements: int,
) -> dict[str, int | float]:
    values = tuple(int(value) for value in lengths)
    if not values or len(values) > recommendation.physical_batch_size:
        raise MemoryError("E006 calibrated physical batch-size budget exceeded")
    if max(values) > recommendation.maximum_length:
        raise MemoryError("E006 batch exceeds its calibrated length regime")
    side = math.ceil(max(values) / 8) * 8
    residues = sum(values)
    pairs = len(values) * side * side if recommendation.constructs_pair_features else 0
    if residues > maximum_residues:
        raise MemoryError("E006 residue budget exceeded")
    if recommendation.constructs_pair_features and (
        pairs > maximum_pair_elements or pairs > recommendation.maximum_pair_elements
    ):
        raise MemoryError("E006 residue or pair-element budget exceeded")
    return {
        "physical_batch_size": len(values),
        "effective_token_count": residues,
        "pair_elements": pairs,
        "padding_fraction": 1.0 - residues / (len(values) * side),
        "accumulation_count": recommendation.accumulation_steps,
    }


def verify_calibration_report(path: str | Path, expected_sha256: str, *, dataset_identity: str) -> dict[str, Any]:
    report_path = Path(path)
    if _sha256(report_path) != expected_sha256:
        raise ValueError("E006 calibration report SHA-256 contradiction")
    report = json.loads(report_path.read_text())
    version = report.get("version")
    expected = {
        "status": "completed",
        "architecture_version": E006_ARCHITECTURE_VERSION,
        "dataset_identity": dataset_identity,
        "dataset_identity_after": dataset_identity,
        "protected_inputs_unchanged": True,
        "authorizes_training": False,
    }
    contradictions = [name for name, value in expected.items() if report.get(name) != value]
    recommendations = report.get("recommendations") or []
    if version == CALIBRATION_VERSION:
        invalid_recommendations = tuple(
            int(item.get("maximum_length", -1)) for item in recommendations
        ) != LENGTH_REGIMES or any(float(item.get("safety_headroom_fraction", 0)) < 0.15 for item in recommendations)
    elif version == "e006_production_calibration_v2":
        expected_pairs = {
            (stage, length) for stage in ("sequence-pretrain", "joint-train") for length in LENGTH_REGIMES
        }
        observed_pairs = {(item.get("stage"), int(item.get("maximum_length", -1))) for item in recommendations}
        invalid_recommendations = (
            observed_pairs != expected_pairs
            or float(report.get("maximum_total_device_occupancy_fraction", 1)) > 0.90
            or float(report.get("minimum_remaining_vram_mib", 0)) < 768
            or any(
                item.get("numerical_status") != "equivalent"
                or float(item.get("peak_total_device_occupancy_fraction", 1)) > 0.90
                or float(item.get("remaining_free_memory_estimate_mib", 0)) < 768
                for item in recommendations
            )
        )
    else:
        invalid_recommendations = True
        contradictions.append("version")
    if contradictions or invalid_recommendations:
        raise ValueError(f"E006 calibration report contradiction: {', '.join(contradictions)}")
    return report


def verify_stage_a_checkpoint(
    path: str | Path,
    expected_sha256: str,
    *,
    dataset_identity: str,
    calibration_sha256: str,
    production_selection_sha256: str | None = None,
    require_contextual_checkpoint: bool = False,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    expected_name = "best_context_verified.pt" if require_contextual_checkpoint else "best.pt"
    if checkpoint_path.name != expected_name:
        raise ValueError(f"E006 Stage-A authorization requires checkpoints/{expected_name}; latest.pt is recovery-only")
    if _sha256(checkpoint_path) != expected_sha256:
        raise ValueError("E006 Stage-A checkpoint SHA-256 contradiction")
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    expected = {
        "version": STAGE_A_CHECKPOINT_VERSION,
        "stage": "sequence-pretrain",
        "architecture_version": E006_ARCHITECTURE_VERSION,
        "dataset_identity": dataset_identity,
        "calibration_sha256": calibration_sha256,
        "authorizes_joint_training": True,
        "authorizes_training": True,
        "status": "completed",
        "checkpoint_role": "validation_selected_best",
        "independently_verified_best": True,
    }
    if require_contextual_checkpoint:
        expected.update(
            checkpoint_role="contextual_diagnostic_selected_best",
            independently_verified_contextual=True,
            authorizes_definitive_evaluation=True,
        )
    contradictions = [name for name, value in expected.items() if payload.get(name) != value]
    if production_selection_sha256 is not None and payload.get("production_selection_sha256") != (
        production_selection_sha256
    ):
        contradictions.append("production_selection_sha256")
    source_path = payload.get("source_immutable_checkpoint")
    source_sha = payload.get("source_immutable_checkpoint_sha256")
    if not source_path or not source_sha or not Path(source_path).is_file() or _sha256(source_path) != source_sha:
        contradictions.append("source_immutable_checkpoint")
    if contradictions:
        raise ValueError(f"E006 Stage-A checkpoint authorization contradiction: {', '.join(contradictions)}")
    return payload


def verify_stage_a_context_gate(
    path: str | Path,
    expected_sha256: str,
    *,
    checkpoint_sha256: str,
    dataset_protocol_sha256: str,
    acceptable_classifications: Sequence[str],
) -> dict[str, Any]:
    """Verify the separate non-authorizing scientific gate for Stage B."""
    report_path = Path(path)
    if not expected_sha256 or not report_path.is_file() or _sha256(report_path) != expected_sha256:
        raise ValueError("E006 Stage-A contextual diagnostic SHA-256 contradiction")
    report = json.loads(report_path.read_text())
    expected = {
        "status": "completed",
        "version": "e006_stage_a_context_diagnostic_v1",
        "checkpoint_sha256": checkpoint_sha256,
        "dataset_protocol_sha256": dataset_protocol_sha256,
        "protected_inputs_unchanged": True,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_evaluation": False,
    }
    contradictions = [name for name, value in expected.items() if report.get(name) != value]
    if report.get("classification") not in set(acceptable_classifications):
        contradictions.append("classification")
    scientific_gate = report.get("scientific_gate", {})
    if (
        scientific_gate.get("classification") != report.get("classification")
        or scientific_gate.get("acceptable_for_stage_b") is not True
    ):
        contradictions.append("scientific_gate")
    if contradictions:
        raise ValueError(f"E006 Stage-A contextual diagnostic contradiction: {', '.join(contradictions)}")
    return report


def verify_checkpoint_artifact(
    path: str | Path,
    expected_sha256: str,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Verify a completed Stage-A or Stage-B checkpoint against authoritative inputs."""
    authorization = _authorization(config)
    dataset_identity = _dataset_identity(authorization)
    calibration_sha = str(config["calibration"]["report_sha256"])
    verify_calibration_report(config["calibration"]["report_path"], calibration_sha, dataset_identity=dataset_identity)
    if _sha256(path) != expected_sha256:
        raise ValueError("E006 checkpoint SHA-256 contradiction")
    payload = load_checkpoint(path, map_location="cpu")
    if payload.get("version") == STAGE_A_CHECKPOINT_VERSION:
        return verify_stage_a_checkpoint(
            path,
            expected_sha256,
            dataset_identity=dataset_identity,
            calibration_sha256=calibration_sha,
            production_selection_sha256=str(config["calibration"]["production_selection_sha256"]),
            require_contextual_checkpoint=(payload.get("checkpoint_role") == "contextual_diagnostic_selected_best"),
        )
    expected = {
        "version": STAGE_B_CHECKPOINT_VERSION,
        "stage": "joint-train",
        "status": "completed",
        "architecture_version": E006_ARCHITECTURE_VERSION,
        "dataset_identity": dataset_identity,
        "calibration_sha256": calibration_sha,
        "production_selection_sha256": str(config["calibration"]["production_selection_sha256"]),
        "authorizes_definitive_evaluation": True,
    }
    contradictions = [name for name, value in expected.items() if payload.get(name) != value]
    if contradictions:
        raise ValueError(f"E006 Stage-B checkpoint contradiction: {', '.join(contradictions)}")
    return payload


def select_best_checkpoint(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("status", "completed") == "completed"]
    if not completed:
        raise ValueError("E006 validation journal has no completed checkpoint")
    return min(completed, key=lambda item: (float(item["sequence_cross_entropy"]), int(item["optimizer_step"])))


def _scheduler_multiplier(step: int, warmup: int, total: int) -> float:
    if total < 1 or not 0 <= warmup < total:
        raise ValueError("E006 scheduler warmup must be in [0, total updates)")
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _amp_context(device: torch.device, config: dict[str, Any]):
    enabled = bool(config.get("mixed_precision", {}).get("enabled", False)) and device.type == "cuda"
    dtype_name = str(config.get("mixed_precision", {}).get("dtype", "float16"))
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported E006 mixed-precision dtype: {dtype_name}")
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled), enabled


def _stage_trainable(model: E006RichGeometryCoDesign, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "joint-train")
    if stage == "sequence-pretrain":
        for module in (
            model.token_embedding,
            model.position_embedding,
            model.sequence_layers,
            model.sequence_norm,
            model.sequence_output,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def _synthetic_rows(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    lengths = config.get("synthetic", {}).get("lengths", [8, 12])
    return [_synthetic_row(f"synthetic-{split}-{index}", int(length), split) for index, length in enumerate(lengths)]


def collate_sequence_pretraining(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad only sequence tensors; Stage A must not construct O(N^2) geometry."""
    if not rows:
        raise ValueError("E006 Stage-A collation requires at least one row")
    lengths = [len(row["sequence"]) for row in rows]
    side = max(lengths)
    tokens = torch.zeros((len(rows), side), dtype=torch.long)
    mask = torch.zeros((len(rows), side), dtype=torch.bool)
    for index, (row, length) in enumerate(zip(rows, lengths, strict=True)):
        values = torch.as_tensor(row["token_ids"], dtype=torch.long)
        if values.shape != (length,):
            raise ValueError(f"E006 Stage-A token/sequence length contradiction: {row['sample_id']}")
        tokens[index, :length] = values
        mask[index, :length] = True
    return {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "sequence_token_ids": tokens,
        "residue_mask": mask,
        "lengths": torch.as_tensor(lengths, dtype=torch.long),
    }


def _validation(
    model: E006RichGeometryCoDesign,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    *,
    stage: str,
    step: int,
) -> dict[str, Any]:
    model.eval()
    records = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            if stage == "sequence-pretrain":
                batch = _move(collate_sequence_pretraining([row]), device)
                if config["objective"].get("version") in {
                    STAGE_A_CONTEXT_OBJECTIVE_VERSION,
                    STAGE_A_CONTEXT_OBJECTIVE_V6,
                }:
                    corruption = context_corruption(
                        batch["sequence_token_ids"],
                        batch["residue_mask"],
                        mask_token_id=1,
                        probability=float(config["objective"]["mask_fraction"]),
                        seed=int(config["seed"]) + 10_000,
                        step=index,
                    )
                    tokens, masked = corruption.inputs, corruption.corrupted_mask
                    logits = model.forward_sequence_pretraining(tokens, batch["residue_mask"])
                    loss = canonical_corrupted_cross_entropy(
                        logits,
                        batch["sequence_token_ids"],
                        masked,
                        batch["residue_mask"],
                    )
                else:
                    tokens, masked = masked_sequence_inputs(
                        batch["sequence_token_ids"],
                        batch["residue_mask"],
                        mask_token_id=1,
                        probability=float(config["objective"]["mask_fraction"]),
                        seed=int(config["seed"]) + 10_000,
                        step=index,
                    )
                    logits = model.forward_sequence_pretraining(tokens, batch["residue_mask"])
                    loss = F.cross_entropy(logits[masked].float(), batch["sequence_token_ids"][masked])
                masked_logits = logits[masked]
                targets = batch["sequence_token_ids"][masked]
                top = masked_logits.topk(5, dim=-1).indices
                length = len(row["sequence"])
                records.append(
                    {
                        "mask_fraction": float(config["objective"]["mask_fraction"]),
                        "geometry_corruption_level": 0.0,
                        "diffusion_timestep": 0,
                        "protein_length": length,
                        "experimental_method": str(row.get("experimental_method") or "unknown"),
                        "pseudo_cb_coverage": sum(value == 0 for value in row["cb_source"]) / length,
                        "frame_coverage": sum(row["local_frame_valid"]) / length,
                        "torsion_coverage": sum(sum(row[f"{name}_mask"]) for name in ("phi", "psi", "omega"))
                        / (3 * length),
                        "conditioning_mode": "sequence_only",
                        "sequence_cross_entropy": float(loss),
                        "perplexity": float(torch.exp(loss)),
                        "top1_accuracy": float((top[:, :1] == targets[:, None]).any(-1).float().mean()),
                        "top3_accuracy": float((top[:, :3] == targets[:, None]).any(-1).float().mean()),
                        "top5_accuracy": float((top == targets[:, None]).any(-1).float().mean()),
                        "geometry_loss": 0.0,
                        "consistency_loss": 0.0,
                        "fusion_gate_mean": 0.0,
                        "fusion_gate_std": 0.0,
                        "fusion_gate_saturated_fraction": 0.0,
                        "sequence_to_geometry_gate_mean": 0.0,
                    }
                )
            else:
                batch = _move(collate_rich_geometry([row]), device)
                for mode in ("sequence_only", "learned_geometry_gating", "forced_geometry_conditioning"):
                    outputs, losses, metadata = _forward(model, batch, config=config, step=step + index, mode=mode)
                    records.append(_metrics(outputs, losses, batch, metadata, mode))
    model.train()
    sequence_ce = float(np.mean([item["sequence_cross_entropy"] for item in records]))
    return {
        "optimizer_step": step,
        "sequence_cross_entropy": sequence_ce,
        "perplexity": math.exp(min(sequence_ce, 50.0)),
        "records": records,
        "stratified": stratified_metrics(records),
    }


def _checkpoint_payload(
    *,
    stage: str,
    status: str,
    model: E006RichGeometryCoDesign,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config_hash: str,
    dataset_identity: str,
    calibration_sha256: str,
    optimizer_step: int,
    microstep: int,
    data_cursor: int,
    dataset_pass: int,
    best_sequence_ce: float,
    production_selection_sha256: str = "",
    processed_valid_tokens: int = 0,
    at_optimizer_boundary: bool = True,
    microbatches_accumulated: int = 0,
    amp_overflows_total: int = 0,
    amp_overflows_consecutive: int = 0,
    overflow_diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": STAGE_A_CHECKPOINT_VERSION if stage == "sequence-pretrain" else STAGE_B_CHECKPOINT_VERSION,
        "stage": stage,
        "status": status,
        "architecture_version": E006_ARCHITECTURE_VERSION,
        "feature_version": RICH_FEATURE_VERSION,
        "config_sha256": config_hash,
        "dataset_identity": dataset_identity,
        "calibration_sha256": calibration_sha256,
        "production_selection_sha256": production_selection_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": _rng_state(),
        "optimizer_step": optimizer_step,
        "microstep": microstep,
        "data_cursor": data_cursor,
        "dataset_pass": dataset_pass,
        "best_sequence_cross_entropy": best_sequence_ce,
        "processed_valid_tokens": processed_valid_tokens,
        "amp_overflows_total": amp_overflows_total,
        "amp_overflows_consecutive": amp_overflows_consecutive,
        "overflow_diagnostics": list(overflow_diagnostics or []),
        "sampler_state": {
            "data_cursor": data_cursor,
            "dataset_pass": dataset_pass,
            "ordering": "deterministic_length_bucket_hash_order",
        },
        "accumulation_state": {
            "microstep": microstep,
            "microbatches_accumulated": microbatches_accumulated,
            "at_optimizer_boundary": at_optimizer_boundary,
        },
        "authorizes_joint_training": stage == "sequence-pretrain" and status == "completed",
        "authorizes_training": stage == "sequence-pretrain" and status == "completed",
        "authorizes_definitive_evaluation": stage == "joint-train" and status == "completed",
    }


def validate_resume_checkpoint(
    payload: dict[str, Any],
    *,
    stage: str,
    config_hash: str,
    dataset_identity: str,
    calibration_sha256: str,
    production_selection_sha256: str | None = None,
) -> None:
    expected = {
        "version": STAGE_A_CHECKPOINT_VERSION if stage == "sequence-pretrain" else STAGE_B_CHECKPOINT_VERSION,
        "stage": stage,
        "architecture_version": E006_ARCHITECTURE_VERSION,
        "config_sha256": config_hash,
        "dataset_identity": dataset_identity,
        "calibration_sha256": calibration_sha256,
    }
    contradictions = [name for name, value in expected.items() if payload.get(name) != value]
    if production_selection_sha256 is not None and payload.get("production_selection_sha256") != (
        production_selection_sha256
    ):
        contradictions.append("production_selection_sha256")
    if contradictions:
        raise ValueError(f"E006 resume checkpoint contradiction: {', '.join(contradictions)}")


def plan_phase3(
    config: dict[str, Any],
    *,
    mode: str,
    synthetic: bool = False,
    _allow_continuation: bool = False,
) -> dict[str, Any]:
    if config.get("continuation") and not _allow_continuation:
        raise ValueError("E006 continuation configuration requires explicit continuation planning")
    validate_phase3_config(config, mode=mode, synthetic=synthetic)
    authorization = None if synthetic else _authorization(config)
    selection_sha = None
    recommendations = None
    if synthetic and mode in TRAINING_MODES:
        recommendations = [
            {
                "maximum_length": 500,
                "physical_batch_size": 1,
                "accumulation_steps": 1,
                "effective_token_budget": 500,
                "maximum_pair_elements": 512 * 512,
            }
        ]
    if not synthetic and mode in TRAINING_MODES:
        calibration_sha = str(config["calibration"]["report_sha256"])
        verify_calibration_report(
            config["calibration"]["report_path"],
            calibration_sha,
            dataset_identity=_dataset_identity(authorization),
        )
        selection_sha = str(config["calibration"]["production_selection_sha256"])
        selection = verify_production_selection(
            config["calibration"]["production_selection_path"],
            selection_sha,
            calibration_path=config["calibration"]["report_path"],
            expected_calibration_sha256=calibration_sha,
            stage=mode,
            configured_regimes=config["batching"]["regimes"],
        )
        recommendations = selection["recommendations"]
    model = _model(config)
    checkpoint_storage = None
    validation_panel_preflight = None
    contextual_monitoring_panel = None
    launch_gates = None
    warm_start_provenance = None
    output_directory_exists = Path(config.get("training", {}).get("output_dir", "")).exists()
    if mode in TRAINING_MODES:
        rows = _training_rows(config, authorization, split="train", synthetic=synthetic)
        validation_rows = _training_rows(config, authorization, split="validation", synthetic=synthetic)
        _, validation_panel_preflight = _validation_panel_preflight(
            rows,
            validation_rows,
            int(config["validation"]["panel_size"]),
        )
        lengths = list(rows.lengths) if isinstance(rows, _LazySelectedRows) else [len(row["sequence"]) for row in rows]
        checkpoint_storage = storage_preflight(
            lengths=lengths,
            regimes=config["batching"].get("regimes") or recommendations or [],
            dataset_passes=int(config["training"]["dataset_passes"]),
            validation_frequency=int(config["training"]["validation_frequency"]),
            recovery_frequency=int(config["training"]["recovery_checkpoint_frequency"]),
            estimated_checkpoint_bytes=int(config["training"]["estimated_checkpoint_bytes"]),
            output_directory=config["training"]["output_dir"],
            minimum_free_disk_gib=float(config["training"]["minimum_free_disk_gib"]),
            review_pause_steps=config["training"].get("review_pause_steps", []),
            maintain_contextual_best=(
                config.get("production_monitoring_version")
                in {"e006_stage_a_v5_monitored_training_v1", "e006_stage_a_v6_monitored_training_v1"}
            ),
        )
        if (
            mode == "sequence-pretrain"
            and not synthetic
            and config.get("production_monitoring_version")
            in {"e006_stage_a_v5_monitored_training_v1", "e006_stage_a_v6_monitored_training_v1"}
        ):
            launch_gates = verify_stage_a_v5_launch_gates(config)
            initialization = config["initialization"]
            checkpoint_path = Path(initialization["checkpoint_path"])
            if _sha256(checkpoint_path) != initialization["checkpoint_sha256"]:
                raise ValueError("E006 Stage-A v5 warm-start checkpoint hash contradiction")
            warm_start_provenance = {
                "mode": initialization["mode"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": initialization["checkpoint_sha256"],
                "restores_training_state": False,
            }
            _, contextual_monitoring_panel, _ = _context_monitoring_resources(
                config,
                authorization,
                include_unigram=False,
            )
    return {
        "status": (
            "blocked_existing_output"
            if output_directory_exists
            else (
                "blocked_insufficient_disk"
                if checkpoint_storage is not None and not checkpoint_storage["storage_safe"]
                else "planned"
            )
        ),
        "version": PHASE3_PROTOCOL_VERSION,
        "mode": mode,
        "architecture_version": model.architecture_version,
        "feature_version": RICH_FEATURE_VERSION,
        "configuration_sha256": _canonical_hash(config),
        "dataset_identity": _dataset_identity(authorization),
        "production_selection_sha256": selection_sha,
        "parameter_counts": model.parameter_counts(),
        "checkpoint_storage_preflight": checkpoint_storage,
        "output_directory": str(config.get("training", {}).get("output_dir", "")),
        "output_directory_exists": output_directory_exists,
        "validation_panel_preflight": validation_panel_preflight,
        "warm_start_provenance": warm_start_provenance,
        "pretraining_launch_gates": launch_gates,
        "contextual_monitoring_panel": contextual_monitoring_panel,
        "contextual_monitoring_schedule": (
            contextual_monitoring_steps(
                int(checkpoint_storage["total_optimizer_updates"]),
                checkpoint_storage["dataset_pass_end_steps"],
            )
            if contextual_monitoring_panel is not None
            else None
        ),
        "scientific_review_pause_steps": config.get("training", {}).get("review_pause_steps", []),
        "post_training_context_diagnostic": config.get("post_training_context_diagnostic"),
        "authorizes_joint_training_before_post_training_diagnostic": False,
        "authorizes_definitive_evaluation_before_post_training_diagnostic": False,
        "authorizes_training": False,
        "training_performed": False,
    }


def plan_continuation(config: dict[str, Any]) -> dict[str, Any]:
    """Verify and describe an audited Stage-A migration without publishing artifacts."""
    _, migration = verify_continuation_source(config)
    plan = plan_phase3(
        config,
        mode="sequence-pretrain",
        synthetic=False,
        _allow_continuation=True,
    )
    imported_step = int(migration["imported_state"]["optimizer_step"])
    storage = plan["checkpoint_storage_preflight"]
    scheduled = storage["scheduled_validation_steps"]
    next_validation = min(step for step in scheduled if step > imported_step)
    if (
        int(migration["total_target_optimizer_steps"]) != int(storage["total_optimizer_updates"])
        or migration["dataset_pass_end_steps"] != storage["dataset_pass_end_steps"]
        or int(migration["next_scheduled_validation_step"]) != next_validation
    ):
        raise ValueError("E006 continuation schedule contradicts the production plan")
    return {
        **plan,
        "mode": "audited-continuation-plan",
        "continuation": migration,
        "starts_from_optimizer_step": imported_step,
        "starts_from_scratch": False,
        "discarded_uncheckpointed_optimizer_updates": 13,
        "next_scheduled_validation_step": next_validation,
        "remaining_immutable_checkpoint_steps": [
            step for step in plan["checkpoint_storage_preflight"]["immutable_checkpoint_steps"] if step > imported_step
        ],
        "authorizes_training": False,
        "training_performed": False,
    }


def _real_calibration_case(
    config: dict[str, Any],
    authorization: RichDatasetAuthorization,
    candidate: dict[str, int],
) -> dict[str, Any]:
    device = torch.device(config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E006 real production calibration requires CUDA")
    target = int(candidate["target_length"])
    dataset = RichGeometryDataset(authorization, split="train")
    indices = nearest_length_indices(
        dataset,
        target_length=target,
        count=int(candidate["physical_batch_size"]),
        seed=int(config["seed"]) + target,
    )
    rows = [dataset[index] for index in indices]
    model = _model(config).to(device).train()
    batch = _move(collate_rich_geometry(rows), device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["learning_rate"]))
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    forward_memory = None
    backward_memory = None
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
        if forward_memory is None:
            forward_memory = _memory(device)
        (losses["total"] / int(candidate["accumulation_steps"])).backward()
        backward_memory = _memory(device)
    assert losses is not None and forward_memory is not None and backward_memory is not None
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or any(not torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("E006 calibration produced missing or non-finite gradients")
    optimizer.step()
    _assert_finite_model(model)
    final_memory = _memory_guard(config, device)
    elapsed = time.monotonic() - started
    return {
        **candidate,
        "status": "passed",
        "finite": all(math.isfinite(float(value.detach())) for value in losses.values()),
        "actual_lengths": batch["lengths"].tolist(),
        "pair_elements": int(len(rows) * batch["distance_matrices"].shape[-1] ** 2),
        "padding_fraction": 1.0 - float(batch["residue_mask"].sum()) / batch["residue_mask"].numel(),
        "forward_memory": forward_memory,
        "forward_backward_memory": backward_memory,
        "optimizer_step_memory": final_memory,
        "peak_cuda_reserved_mib": final_memory["peak_cuda_reserved_mib"],
        "host_rss_mib": final_memory["current_rss_mib"],
        "samples_per_second": len(rows) / elapsed,
        "tokens_per_second": int(batch["residue_mask"].sum()) / elapsed,
    }


def run_calibration(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("calibration", {}).get("version") == "e006_production_calibration_v2":
        from protein_distance_diffusion.training.rich_codesign_calibration_v2 import run_calibration_v2

        return run_calibration_v2(config_path)
    validate_phase3_config(config, mode="calibrate")
    output = Path(config["calibration"]["output_report"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E006 calibration report: {output}")
    authorization = _authorization(config)
    identity = _dataset_identity(authorization)
    protected_before = _protected_hashes(authorization)
    heartbeat = output.with_name("calibration_heartbeat.json")
    _atomic_json(heartbeat, {"status": "running", "started_utc": _utc_now()})
    all_cases = []
    try:
        for regime in config["calibration"]["regimes"]:
            target = int(regime["target_length"])
            candidates = [
                {
                    "target_length": target,
                    "physical_batch_size": int(item["physical_batch_size"]),
                    "accumulation_steps": int(item["accumulation_steps"]),
                }
                for item in regime["candidates"]
            ]
            all_cases.extend(
                execute_calibration_regime(
                    candidates,
                    lambda candidate: _real_calibration_case(config, authorization, candidate),
                )
            )
            _atomic_json(
                heartbeat,
                {
                    "status": "running",
                    "completed_regimes": len({int(item["target_length"]) for item in all_cases}),
                    "total_regimes": len(LENGTH_REGIMES),
                    "percentage": 100 * len({int(item["target_length"]) for item in all_cases}) / len(LENGTH_REGIMES),
                    "timestamp_utc": _utc_now(),
                },
            )
        total_vram = torch.cuda.get_device_properties(torch.device(config["device"])).total_memory / 2**20
        recommendations = select_calibration_recommendations(all_cases, total_vram_mib=total_vram)
        authorization_after = _authorization(config)
        identity_after = _dataset_identity(authorization_after)
        protected_after = _protected_hashes(authorization_after)
        if identity_after != identity or protected_after != protected_before:
            raise RuntimeError("E006 protected dataset changed during calibration")
        report = {
            "status": "completed",
            "version": CALIBRATION_VERSION,
            "architecture_version": E006_ARCHITECTURE_VERSION,
            "configuration_sha256": _canonical_hash(config),
            "dataset_identity": identity,
            "dataset_identity_after": identity_after,
            "protected_inputs_unchanged": True,
            "protected_input_hashes_before": protected_before,
            "protected_input_hashes_after": protected_after,
            "cases": all_cases,
            "recommendations": recommendations,
            "total_vram_mib": total_vram,
            "required_headroom_fraction": 0.15,
            "authorizes_training": False,
            "authorizes_checkpoint_initialization": False,
            "completed_utc": _utc_now(),
        }
        _atomic_json(output, report)
        _atomic_json(
            heartbeat,
            {"status": "completed", "completed_utc": report["completed_utc"], "report_sha256": _sha256(output)},
        )
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "version": CALIBRATION_VERSION,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "cases": all_cases,
            "authorizes_training": False,
            "completed_utc": _utc_now(),
        }
        _atomic_json(output, failure)
        _atomic_json(heartbeat, failure)
        raise


def _training_rows(
    config: dict[str, Any], authorization: RichDatasetAuthorization | None, *, split: str, synthetic: bool
) -> Sequence[dict[str, Any]]:
    if synthetic:
        return _synthetic_rows(config, split)
    if authorization is None:
        raise RuntimeError("E006 production rows require dataset authorization")
    dataset = RichGeometryDataset(authorization, split=split)
    requested = (
        int(config["validation"]["panel_size"])
        if split == "validation"
        else int(config["training"].get("bounded_panel_size", len(dataset)))
    )
    indices = deterministic_length_bucket_sample(
        dataset,
        count=min(requested, len(dataset)),
        seed=int(config["seed"]) + (1 if split == "validation" else 0),
        maximum_length=int(config["training"].get("maximum_length", 500)),
    )
    selected = set(indices)
    ordered = []
    for index, sample_id, length in dataset.iter_metadata():
        if index in selected:
            boundary = next(value for value in (64, 128, 256, 384, 500) if length <= value)
            rank = hashlib.sha256(f"{config['seed']}:{split}:{sample_id}".encode()).hexdigest()
            ordered.append((boundary, rank, index, length, sample_id))
    ordered.sort()
    return _LazySelectedRows(
        dataset,
        [index for _, _, index, _, _ in ordered],
        [length for _, _, _, length, _ in ordered],
        [sample_id for _, _, _, _, sample_id in ordered],
    )


def _context_monitoring_resources(
    config: dict[str, Any],
    authorization: RichDatasetAuthorization,
    *,
    include_unigram: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    from protein_distance_diffusion.evaluation.e006_stage_a_context import (
        SequenceOnlyRichDataset,
        select_panels,
    )

    settings = config["contextual_monitoring"]
    reservation = settings["final_diagnostic_reservation"]
    validation = SequenceOnlyRichDataset(authorization, split="validation")
    reserved_panels, reserved_identity = select_panels(
        validation,
        original_size=int(reservation["original_size"]),
        independent_size=int(reservation["independent_size"]),
        training_seed=int(reservation["training_seed"]),
        diagnostic_seed=int(reservation["diagnostic_seed"]),
        minimum_per_nonempty_bucket=int(reservation["minimum_per_nonempty_bucket"]),
        maximum_length=int(reservation["maximum_length"]),
        expected_original_sample_id_sha256=str(reservation["expected_original_sample_id_sha256"]),
    )
    reserved_ids = {member.sample_id for members in reserved_panels.values() for member in members}
    monitor_rows, monitor_identity = select_monitoring_panel(
        validation,
        count=int(settings["panel_size"]),
        seed=int(settings["seed"]),
        maximum_length=int(settings["maximum_length"]),
        excluded_sample_ids=reserved_ids,
    )
    monitor_identity["final_diagnostic_reservation"] = reserved_identity["panels"]
    unigram = None
    if include_unigram:
        unigram = training_unigram(SequenceOnlyRichDataset(authorization, split="train"))
    return monitor_rows, monitor_identity, unigram


def _selected_valid_token_count(rows: Sequence[dict[str, Any]]) -> int:
    if isinstance(rows, _LazySelectedRows):
        return sum(rows.lengths)
    return sum(len(row["sequence"]) for row in rows)


def _planned_optimizer_updates(
    lengths: Sequence[int],
    recommendations: Iterable[dict[str, Any]],
    *,
    stage: str,
    dataset_passes: int,
) -> int:
    updates = 0
    cursor = 0
    while cursor < len(lengths):
        recommendation = recommendation_for_length(lengths[cursor], recommendations, stage=stage)
        consumed = 0
        for offset in range(recommendation.physical_batch_size):
            if cursor + offset >= len(lengths) or lengths[cursor + offset] > recommendation.maximum_length:
                break
            consumed += 1
        if consumed < 1:
            raise ValueError("E006 production batch plan made no progress")
        cursor += consumed
        updates += 1
    return updates * dataset_passes


def _run_training_stage(
    config_path: str | Path,
    *,
    mode: str,
    resume: bool = False,
    synthetic: bool = False,
    continuation: bool = False,
    review_decision_path: str | Path | None = None,
    expected_review_decision_sha256: str | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_phase3_config(config, mode=mode, synthetic=synthetic)
    monitored_v5 = (
        mode == "sequence-pretrain"
        and not synthetic
        and config.get("production_monitoring_version")
        in {"e006_stage_a_v5_monitored_training_v1", "e006_stage_a_v6_monitored_training_v1"}
    )
    launch_gate_evidence = None
    if mode == "sequence-pretrain" and not synthetic:
        launch_gate_evidence = verify_stage_a_v5_launch_gates(config)
    if continuation and (mode != "sequence-pretrain" or resume):
        raise ValueError("E006 audited continuation is a Stage-A migration-only mode")
    if config.get("continuation") and not (continuation or resume):
        raise ValueError("E006 continuation configuration requires explicit continuation mode")
    continuation_source = verify_continuation_source(config) if continuation else None
    output = Path(config["training"]["output_dir"])
    summary_path = output / "protocol.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
        raise FileExistsError(f"Refusing to overwrite completed E006 stage: {output}")
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"E006 stage output is non-empty; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    authorization = None if synthetic else _authorization(config)
    dataset_identity = _dataset_identity(authorization)
    protected_before = _protected_hashes(authorization)
    config_hash = _training_config_hash(config)
    if synthetic:
        calibration = {
            "recommendations": [
                {
                    "maximum_length": 500,
                    "physical_batch_size": 1,
                    "accumulation_steps": int(config["training"].get("synthetic_accumulation_steps", 1)),
                    "maximum_pair_elements": 512 * 512,
                }
            ]
        }
        calibration_sha = "synthetic-calibration-v1"
        production_selection_sha = "synthetic-production-selection-v1"
    else:
        calibration_sha = str(config["calibration"]["report_sha256"])
        calibration = verify_calibration_report(
            config["calibration"]["report_path"], calibration_sha, dataset_identity=dataset_identity
        )
        production_selection_sha = str(config["calibration"]["production_selection_sha256"])
        selection = verify_production_selection(
            config["calibration"]["production_selection_path"],
            production_selection_sha,
            calibration_path=config["calibration"]["report_path"],
            expected_calibration_sha256=calibration_sha,
            stage=mode,
            configured_regimes=config["batching"]["regimes"],
        )
        calibration["recommendations"] = selection["recommendations"]
    if not resume and not continuation:
        torch.manual_seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        random.seed(int(config["seed"]))
    model = _model(config)
    warm_start_provenance = None
    if monitored_v5:
        if not resume:
            warm_start_provenance = load_warm_start_weights_only(model, config["initialization"])
        else:
            initialization = config["initialization"]
            path = Path(initialization["checkpoint_path"])
            if _sha256(path) != initialization["checkpoint_sha256"]:
                raise ValueError("E006 Stage-A v5 warm-start checkpoint hash contradiction")
            warm_start_provenance = {
                "mode": "checkpoint_weights_only",
                "checkpoint_path": str(path),
                "checkpoint_sha256": initialization["checkpoint_sha256"],
                "resume_uses_recovery_state_instead_of_reloading_source": True,
            }
    if mode == "joint-train":
        if not synthetic and config["stage_a"].get("context_diagnostic_required", False):
            verify_stage_a_context_gate(
                config["stage_a"]["context_diagnostic_path"],
                config["stage_a"]["context_diagnostic_sha256"],
                checkpoint_sha256=config["stage_a"]["checkpoint_sha256"],
                dataset_protocol_sha256=config["dataset"]["protocol_sha256"],
                acceptable_classifications=config["stage_a"]["acceptable_context_classifications"],
            )
        stage_a = verify_stage_a_checkpoint(
            config["stage_a"]["checkpoint_path"],
            config["stage_a"]["checkpoint_sha256"],
            dataset_identity=dataset_identity,
            calibration_sha256=calibration_sha,
            production_selection_sha256=production_selection_sha,
            require_contextual_checkpoint=bool(config["stage_a"].get("requires_contextual_checkpoint", False)),
        )
        model.load_state_dict(stage_a["model"])
    _stage_trainable(model, mode)
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E006 production training requested unavailable CUDA")
    model.to(device).train()
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    train_rows = _training_rows(config, authorization, split="train", synthetic=synthetic)
    validation_rows = _training_rows(config, authorization, split="validation", synthetic=synthetic)
    validation_panel, _ = _validation_panel_preflight(
        train_rows,
        validation_rows,
        int(config["validation"]["panel_size"]),
    )
    contextual_monitor_panel = None
    contextual_monitor_identity = None
    unigram_baseline = None
    if monitored_v5:
        assert authorization is not None
        contextual_monitor_panel, contextual_monitor_identity, unigram_baseline = _context_monitoring_resources(
            config,
            authorization,
            include_unigram=True,
        )
    passes = int(config["training"]["dataset_passes"])
    target_microsteps = passes * len(train_rows)
    target_valid_tokens = passes * _selected_valid_token_count(train_rows)
    row_lengths = (
        train_rows.lengths
        if isinstance(train_rows, _LazySelectedRows)
        else tuple(len(row["sequence"]) for row in train_rows)
    )
    planned_updates = _planned_optimizer_updates(
        row_lengths,
        calibration["recommendations"],
        stage=mode,
        dataset_passes=passes,
    )
    total_updates = int(config["training"].get("maximum_optimizer_updates") or planned_updates)
    checkpoint_storage = storage_preflight(
        lengths=list(row_lengths),
        regimes=config["batching"].get("regimes") or calibration["recommendations"],
        dataset_passes=passes,
        validation_frequency=int(config["training"]["validation_frequency"]),
        recovery_frequency=int(config["training"]["recovery_checkpoint_frequency"]),
        estimated_checkpoint_bytes=int(config["training"]["estimated_checkpoint_bytes"]),
        output_directory=output,
        minimum_free_disk_gib=float(config["training"]["minimum_free_disk_gib"]),
        review_pause_steps=config["training"].get("review_pause_steps", []),
        maintain_contextual_best=monitored_v5,
    )
    if continuation_source is not None:
        migration_schedule = continuation_source[1]
        imported_step = int(migration_schedule["imported_state"]["optimizer_step"])
        remaining_validations = [
            step for step in checkpoint_storage["scheduled_validation_steps"] if step > imported_step
        ]
        if (
            int(migration_schedule["total_target_optimizer_steps"]) != total_updates
            or migration_schedule["dataset_pass_end_steps"] != checkpoint_storage["dataset_pass_end_steps"]
            or not remaining_validations
            or int(migration_schedule["next_scheduled_validation_step"]) != remaining_validations[0]
        ):
            raise ValueError("E006 continuation schedule contradicts the restored training state")
    require_storage_preflight(checkpoint_storage)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_multiplier(step, int(optimizer_config["warmup_updates"]), total_updates),
    )
    amp_enabled = bool(config.get("mixed_precision", {}).get("enabled", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint = output / "checkpoints" / "latest.pt"
    optimizer_step = microstep = data_cursor = dataset_pass = processed_valid_tokens = 0
    best_sequence_ce = math.inf
    amp_overflows_total = 0
    amp_overflows_consecutive = 0
    overflow_diagnostics: list[dict[str, Any]] = []
    review_decision = None
    if resume:
        verify_recovery_checkpoint(checkpoint.parent)
        saved = load_checkpoint(checkpoint, map_location=device)
        validate_resume_checkpoint(
            saved,
            stage=mode,
            config_hash=config_hash,
            dataset_identity=dataset_identity,
            calibration_sha256=calibration_sha,
            production_selection_sha256=production_selection_sha,
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        _restore_rng_state(saved.get("rng_state"))
        optimizer_step = int(saved["optimizer_step"])
        microstep = int(saved["microstep"])
        data_cursor = int(saved["data_cursor"])
        dataset_pass = int(saved["dataset_pass"])
        processed_valid_tokens = int(saved.get("processed_valid_tokens", 0))
        best_sequence_ce = float(saved["best_sequence_cross_entropy"])
        amp_overflows_total = int(saved.get("amp_overflows_total", 0))
        amp_overflows_consecutive = int(saved.get("amp_overflows_consecutive", 0))
        overflow_diagnostics = list(saved.get("overflow_diagnostics", []))
        review_decision = saved.get("scientific_review_decision")
        prior_protocol = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        if monitored_v5 and prior_protocol.get("status") == "paused_for_scientific_review":
            if optimizer_step not in set(config["training"]["review_pause_steps"]):
                raise ValueError("E006 scientific-review pause step contradicts recovery state")
            if review_decision_path is None or expected_review_decision_sha256 is None:
                raise ValueError("E006 resume from scientific-review pause requires a hash-pinned review decision")
            verified_decision = verify_review_decision(
                review_decision_path,
                expected_review_decision_sha256,
                config_sha256=config_hash,
                checkpoint_sha256=prior_protocol["immutable_review_checkpoint"]["sha256"],
                optimizer_step=optimizer_step,
            )
            review_decision = {
                **verified_decision,
                "path": str(review_decision_path),
                "sha256": expected_review_decision_sha256,
            }
        elif review_decision is not None:
            verify_review_decision(
                review_decision["path"],
                review_decision["sha256"],
                config_sha256=config_hash,
                checkpoint_sha256=review_decision["review_checkpoint_sha256"],
                optimizer_step=int(review_decision["optimizer_step"]),
            )
        elif review_decision_path is not None or expected_review_decision_sha256 is not None:
            raise ValueError("E006 review decision supplied outside a scientific-review pause")
        elif monitored_v5 and optimizer_step >= min(config["training"]["review_pause_steps"]):
            raise ValueError("E006 recovery state at or beyond review pause lacks an approved review decision")
    elif continuation:
        assert continuation_source is not None
        saved, _ = continuation_source
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        _restore_rng_state(saved.get("rng_state"))
        optimizer_step = int(saved["optimizer_step"])
        microstep = int(saved["microstep"])
        data_cursor = int(saved["data_cursor"])
        dataset_pass = int(saved["dataset_pass"])
        processed_valid_tokens = int(saved["processed_valid_tokens"])
        best_sequence_ce = float(saved["best_sequence_cross_entropy"])
        amp_overflows_total = int(saved.get("amp_overflows_total", 0))
        amp_overflows_consecutive = int(saved.get("amp_overflows_consecutive", 0))
        overflow_diagnostics = list(saved.get("overflow_diagnostics", []))
    interrupted = [False]
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    heartbeat = output / "heartbeat.json"
    metrics_path = output / "metrics.jsonl"
    validation_path = output / "validation_journal.jsonl"
    contextual_monitoring_path = output / "contextual_monitoring.jsonl"
    checkpoint_manifest_path = output / "checkpoint_manifest.json"
    existing_manifest = (
        json.loads(checkpoint_manifest_path.read_text()) if resume and checkpoint_manifest_path.exists() else {}
    )
    checkpoint_manifest: list[dict[str, Any]] = existing_manifest.get("checkpoints", [])
    best_record = (
        json.loads((checkpoint.parent / "best.json").read_text())
        if resume and (checkpoint.parent / "best.json").exists()
        else None
    )
    latest_record = (
        json.loads((checkpoint.parent / "latest.json").read_text())
        if resume and (checkpoint.parent / "latest.json").exists()
        else None
    )
    best_context_record = (
        json.loads((checkpoint.parent / "best_context.json").read_text())
        if resume and (checkpoint.parent / "best_context.json").exists()
        else None
    )
    existing_context_records = (
        [json.loads(line) for line in contextual_monitoring_path.read_text().splitlines() if line.strip()]
        if resume and contextual_monitoring_path.exists()
        else []
    )
    monitored_steps = {
        int(record["optimizer_step"])
        for record in existing_context_records
        if record.get("record_type") == "contextual_monitoring"
    }
    monitoring_schedule = (
        contextual_monitoring_steps(total_updates, checkpoint_storage["dataset_pass_end_steps"]) if monitored_v5 else []
    )
    started = time.monotonic()
    at_optimizer_boundary = True
    microbatches_accumulated = 0
    latest_batch_evidence: dict[str, Any] | None = None
    migration_record: dict[str, Any] | None = existing_manifest.get("migration")

    def checkpoint_payload() -> dict[str, Any]:
        payload = _checkpoint_payload(
            stage=mode,
            status="recovery_only",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config_hash=config_hash,
            dataset_identity=dataset_identity,
            calibration_sha256=calibration_sha,
            production_selection_sha256=production_selection_sha,
            optimizer_step=optimizer_step,
            microstep=microstep,
            data_cursor=data_cursor,
            dataset_pass=dataset_pass,
            best_sequence_ce=best_sequence_ce,
            processed_valid_tokens=processed_valid_tokens,
            at_optimizer_boundary=at_optimizer_boundary,
            microbatches_accumulated=microbatches_accumulated,
            amp_overflows_total=amp_overflows_total,
            amp_overflows_consecutive=amp_overflows_consecutive,
            overflow_diagnostics=overflow_diagnostics,
        )
        if config.get("continuation"):
            payload["migration_source"] = {
                "role": "audited_migration_source",
                "checkpoint_path": str(config["continuation"]["source_checkpoint_path"]),
                "checkpoint_sha256": str(config["continuation"]["source_checkpoint_sha256"]),
                "optimizer_step": int(config["continuation"]["source_optimizer_step"]),
                "authorizes_training": False,
            }
        if monitored_v5:
            payload["warm_start_provenance"] = warm_start_provenance
            payload["contextual_monitoring_panel"] = contextual_monitor_identity
            payload["best_context_record"] = best_context_record
            payload["post_training_context_diagnostic_required"] = True
            payload["scientific_review_decision"] = review_decision
            payload["authorizes_joint_training"] = False
            payload["authorizes_training"] = False
            payload["authorizes_definitive_evaluation"] = False
        return payload

    def publish_manifest() -> None:
        _atomic_json(
            checkpoint_manifest_path,
            {
                "version": PHASE3_PROTOCOL_VERSION,
                "checkpoint_policy_version": "e006_production_checkpoint_policy_v1",
                "checkpoints": checkpoint_manifest,
                "rolling_latest": latest_record,
                "best": best_record,
                "best_context": best_context_record,
                "migration": migration_record,
            },
        )

    def immutable_for_step(payload: dict[str, Any], reasons: set[str]) -> dict[str, Any]:
        existing = next(
            (item for item in checkpoint_manifest if int(item["optimizer_step"]) == optimizer_step),
            None,
        )
        record = publish_immutable_checkpoint(
            checkpoint.parent,
            payload,
            reasons=reasons,
            existing_record=existing,
        )
        if existing is None:
            checkpoint_manifest.append(record)
        else:
            checkpoint_manifest[checkpoint_manifest.index(existing)] = record
        checkpoint_manifest.sort(key=lambda item: int(item["optimizer_step"]))
        return record

    def publish_context_monitor() -> dict[str, Any]:
        nonlocal best_context_record
        if not monitored_v5 or contextual_monitor_panel is None or unigram_baseline is None:
            raise RuntimeError("E006 contextual monitor resources are unavailable")
        optimizer_state_before = _torch_state_hash(optimizer.state_dict())
        scheduler_state_before = _torch_state_hash(scheduler.state_dict())
        record = evaluate_context_monitor(
            model,
            contextual_monitor_panel,
            step=optimizer_step,
            seed=int(config["contextual_monitoring"]["seed"]),
            mask_fraction=float(config["contextual_monitoring"]["mask_fraction"]),
            unigram=unigram_baseline,
            device=device,
            initial_normal_cross_entropy=(
                existing_context_records[0]["conditions"]["normal"]["canonical_cross_entropy"]
                if existing_context_records
                else None
            ),
            health_policy=("v6" if config["objective"].get("version") == STAGE_A_CONTEXT_OBJECTIVE_V6 else "v5"),
        )
        existing_context_records.append(record)
        optimizer_changed = optimizer_state_before != _torch_state_hash(optimizer.state_dict())
        scheduler_changed = scheduler_state_before != _torch_state_hash(scheduler.state_dict())
        if optimizer_changed or scheduler_changed:
            raise RuntimeError("E006 contextual monitoring mutated optimizer or scheduler state")
        record.update(
            learning_rate=float(scheduler.get_last_lr()[0]),
            amp_scale=float(scaler.get_scale()) if amp_enabled else None,
            amp_overflows_total=amp_overflows_total,
            amp_overflows_consecutive=amp_overflows_consecutive,
            processed_valid_tokens=processed_valid_tokens,
            data_cursor=data_cursor,
            dataset_pass=dataset_pass,
            memory=_memory_guard(config, device),
            panel_sample_id_sha256=contextual_monitor_identity["sample_id_sha256"],
        )
        _append_jsonl(contextual_monitoring_path, record)
        monitored_steps.add(optimizer_step)
        candidate_key = contextual_selection_key(record)
        incumbent_key = tuple(best_context_record["selection_key"]) if best_context_record is not None else None
        if incumbent_key is None or candidate_key < incumbent_key:
            selected = checkpoint_payload()
            selected.update(
                checkpoint_role="contextual_monitor_selected_best",
                selected_contextual_monitor_record=record,
                contextual_selection_policy=(
                    "lexicographic health, normal-minus-shuffled CE, normal-minus-null CE, normal CE"
                ),
                authorizes_training=False,
                authorizes_joint_training=False,
                authorizes_definitive_evaluation=False,
            )
            path = checkpoint.parent / "best_context.pt"
            save_checkpoint(path, selected)
            best_context_record = {
                "role": "contextual_monitor_selected_best",
                "path": str(path),
                "sha256": _sha256(path),
                "optimizer_step": optimizer_step,
                "selection_key": list(candidate_key),
                "health": record["health"],
                "normal_cross_entropy": record["conditions"]["normal"]["canonical_cross_entropy"],
                "normal_minus_shuffled_ce": record["normal_minus_shuffled_ce"],
                "normal_minus_null_ce": record["normal_minus_null_ce"],
                "authorizes_training": False,
                "authorizes_joint_training": False,
                "authorizes_definitive_evaluation": False,
            }
            _atomic_json(checkpoint.parent / "best_context.json", best_context_record)
        return record

    try:
        if continuation:
            assert continuation_source is not None
            _, migration_record = verify_continuation_source(config)
            source = config["continuation"]
            latest_record = publish_recovery_checkpoint(checkpoint.parent, checkpoint_payload())
            source_best = load_checkpoint(source["source_best_checkpoint_path"], map_location="cpu")
            source_best.update(
                config_sha256=config_hash,
                checkpoint_role="imported_validation_incumbent",
                migration_source_checkpoint=str(source["source_best_checkpoint_path"]),
                migration_source_checkpoint_sha256=str(source["source_best_checkpoint_sha256"]),
                authorizes_joint_training=False,
                authorizes_training=False,
                authorizes_definitive_evaluation=False,
                status="imported_validation_incumbent",
            )
            destination_best = checkpoint.parent / "best.pt"
            save_checkpoint(destination_best, source_best)
            source_best_metadata = json.loads(Path(source["source_best_metadata_path"]).read_text())
            best_record = {
                **source_best_metadata,
                "role": "imported_validation_incumbent",
                "path": str(destination_best),
                "sha256": _sha256(destination_best),
                "authorizes_training": False,
                "migration_source_checkpoint": str(source["source_best_checkpoint_path"]),
                "migration_source_checkpoint_sha256": str(source["source_best_checkpoint_sha256"]),
            }
            _atomic_json(checkpoint.parent / "best.json", best_record)
            _append_jsonl(
                validation_path,
                {
                    "record_type": "immutable_validation_history_reference",
                    **migration_record["validation_history"],
                    "best_incumbent": migration_record["imported_best_incumbent"],
                },
            )
            migration_record.update(
                status="startup_verified",
                startup_verification_passed=True,
                destination_output_dir=str(output),
                destination_recovery_checkpoint=str(checkpoint),
                destination_recovery_checkpoint_sha256=latest_record["sha256"],
                destination_artifact_fingerprints=_directory_fingerprint(output),
                authorizes_training=False,
                authorizes_joint_training=False,
                completed_utc=_utc_now(),
            )
            _atomic_json(output / "migration_protocol.json", migration_record)
            publish_manifest()
        if optimizer_step == 0:
            initial_validation = _validation(
                model,
                validation_panel,
                config,
                device,
                stage=mode,
                step=0,
            )
            _append_jsonl(validation_path, initial_validation)
            best_sequence_ce = initial_validation["sequence_cross_entropy"]
            payload = checkpoint_payload()
            immutable = immutable_for_step(payload, {"validation", "best_selection", "initial_validation"})
            best_record = publish_best_checkpoint(
                checkpoint.parent,
                immutable,
                payload,
                validation_sequence_cross_entropy=best_sequence_ce,
            )
            if monitored_v5 and 0 not in monitored_steps:
                publish_context_monitor()
            publish_manifest()
        while optimizer_step < total_updates and data_cursor < target_microsteps:
            if interrupted[0]:
                raise TrainingInterrupted("SIGINT requested")
            previous_dataset_pass = dataset_pass
            row = train_rows[data_cursor % len(train_rows)]
            recommendation = recommendation_for_length(
                len(row["sequence"]),
                calibration["recommendations"],
                stage=mode,
            )
            physical_rows = []
            for offset in range(recommendation.physical_batch_size):
                candidate = train_rows[(data_cursor + offset) % len(train_rows)]
                if len(candidate["sequence"]) > recommendation.maximum_length:
                    break
                physical_rows.append(candidate)
            budget = validate_batch_budget(
                [len(item["sequence"]) for item in physical_rows],
                recommendation=recommendation,
                maximum_residues=int(
                    config["batching"].get("maximum_residues")
                    or recommendation.maximum_length * recommendation.physical_batch_size
                ),
                maximum_pair_elements=int(
                    config["batching"].get("maximum_pair_elements") or recommendation.maximum_pair_elements
                ),
            )
            at_optimizer_boundary = False
            microbatches_accumulated = 0
            optimizer.zero_grad(set_to_none=True)
            step_losses = []
            skipped = False
            remaining = min(
                target_microsteps - data_cursor,
                len(train_rows) - (data_cursor % len(train_rows)),
            )
            microbatches = []
            consumed = 0
            for _ in range(recommendation.accumulation_steps):
                selected_rows = []
                for offset in range(recommendation.physical_batch_size):
                    if consumed + offset >= remaining:
                        break
                    candidate = train_rows[(data_cursor + consumed + offset) % len(train_rows)]
                    if len(candidate["sequence"]) > recommendation.maximum_length:
                        break
                    selected_rows.append(candidate)
                if not selected_rows:
                    break
                microbatches.append(selected_rows)
                consumed += len(selected_rows)
            accumulation_steps = len(microbatches)
            step_lengths = [[len(item["sequence"]) for item in rows] for rows in microbatches]
            for lengths in step_lengths:
                validate_batch_budget(
                    lengths,
                    recommendation=recommendation,
                    maximum_residues=int(
                        config["batching"].get("maximum_residues")
                        or recommendation.maximum_length * recommendation.physical_batch_size
                    ),
                    maximum_pair_elements=int(
                        config["batching"].get("maximum_pair_elements") or recommendation.maximum_pair_elements
                    ),
                )
            effective_tokens = sum(sum(lengths) for lengths in step_lengths)
            padded_tokens = sum(len(lengths) * (math.ceil(max(lengths) / 8) * 8) for lengths in step_lengths)
            budget.update(
                physical_batch_size=max(map(len, step_lengths)),
                effective_token_count=effective_tokens,
                pair_elements=sum(len(lengths) * (math.ceil(max(lengths) / 8) * 8) ** 2 for lengths in step_lengths),
                padding_fraction=1.0 - effective_tokens / padded_tokens,
                accumulation_count=accumulation_steps,
            )
            diagnostic_limit = int(config["training"].get("maximum_overflow_diagnostic_examples", 20))
            latest_batch_evidence = {
                "optimizer_step_before": optimizer_step,
                "microstep_before": microstep,
                "dataset_pass_before": dataset_pass,
                "data_cursor_before": data_cursor,
                "length_bucket": recommendation.maximum_length,
                "effective_token_count": effective_tokens,
                "sample_count": consumed,
                "sample_ids": [str(item["sample_id"]) for rows in microbatches for item in rows][:diagnostic_limit],
                "sample_ids_truncated": consumed > diagnostic_limit,
                "sample_lengths": [len(item["sequence"]) for rows in microbatches for item in rows][:diagnostic_limit],
                "amp_scale_before": float(scaler.get_scale()) if amp_enabled else None,
            }
            for selected_rows in microbatches:
                batch = _move(
                    collate_sequence_pretraining(selected_rows)
                    if mode == "sequence-pretrain"
                    else collate_rich_geometry(selected_rows),
                    device,
                )
                context, _ = _amp_context(device, config)
                with context:
                    if mode == "sequence-pretrain":
                        objective_version = config["objective"].get("version")
                        if objective_version in {STAGE_A_CONTEXT_OBJECTIVE_VERSION, STAGE_A_CONTEXT_OBJECTIVE_V6}:
                            corruption = context_corruption(
                                batch["sequence_token_ids"],
                                batch["residue_mask"],
                                mask_token_id=1,
                                probability=float(config["objective"]["mask_fraction"]),
                                seed=int(config["seed"]),
                                step=microstep,
                            )
                            inputs, masked = corruption.inputs, corruption.corrupted_mask
                        else:
                            inputs, masked = masked_sequence_inputs(
                                batch["sequence_token_ids"],
                                batch["residue_mask"],
                                mask_token_id=1,
                                probability=float(config["objective"]["mask_fraction"]),
                                seed=int(config["seed"]),
                                step=microstep,
                            )
                        if objective_version == STAGE_A_CONTEXT_OBJECTIVE_V6:
                            logits, shuffled_logits, paired_evidence = paired_dropout_forwards(
                                model,
                                inputs,
                                corruption.shuffled_inputs,
                                batch["residue_mask"],
                            )
                        else:
                            logits = model.forward_sequence_pretraining(inputs, batch["residue_mask"])
                        if not torch.isfinite(logits).all():
                            raise E006GradientError(
                                "E006 non-finite Stage-A activations",
                                {**latest_batch_evidence, "failure_kind": "nonfinite_activation"},
                            )
                        if objective_version == STAGE_A_CONTEXT_OBJECTIVE_VERSION:
                            with torch.no_grad():
                                shuffled_logits = model.forward_sequence_pretraining(
                                    corruption.shuffled_inputs,
                                    batch["residue_mask"],
                                )
                            losses = contextual_stage_a_loss(
                                logits,
                                shuffled_logits,
                                batch["sequence_token_ids"],
                                masked,
                                batch["residue_mask"],
                                contrast_weight=float(config["objective"]["context_contrast_weight"]),
                                contrast_margin_nats=float(config["objective"]["context_contrast_margin_nats"]),
                            )
                            total = losses["total"]
                        elif objective_version == STAGE_A_CONTEXT_OBJECTIVE_V6:
                            losses = contextual_stage_a_loss_v6(
                                logits,
                                shuffled_logits,
                                batch["sequence_token_ids"],
                                masked,
                                batch["residue_mask"],
                                contrast_weight=float(config["objective"]["context_contrast_weight"]),
                                contrast_margin_nats=float(config["objective"]["context_contrast_margin_nats"]),
                                paired_dropout_evidence=paired_evidence,
                            )
                            total = losses["total"]
                        else:
                            total = F.cross_entropy(logits[masked].float(), batch["sequence_token_ids"][masked])
                            losses = {
                                "sequence": total,
                                "geometry": total.new_zeros(()),
                                "consistency": total.new_zeros(()),
                                "total": total,
                            }
                    else:
                        _, losses, _ = _forward(
                            model,
                            batch,
                            config=config,
                            step=optimizer_step,
                            mode="learned_geometry_gating",
                        )
                        total = losses["total"]
                latest_batch_evidence["loss_components"] = {
                    name: float(value.detach()) for name, value in losses.items()
                }
                latest_batch_evidence["forward_loss_finite"] = bool(torch.isfinite(total))
                _require_finite_training_loss(total, latest_batch_evidence)
                scaler.scale(total / accumulation_steps).backward()
                step_losses.append({name: float(value.detach()) for name, value in losses.items()})
                microstep += 1
                microbatches_accumulated += 1
            step_result = _optimizer_boundary_update(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                stage=mode,
                gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
                amp_overflows_total=amp_overflows_total,
                amp_overflows_consecutive=amp_overflows_consecutive,
                maximum_total_amp_overflows=int(config["training"].get("maximum_total_amp_overflows", 1000)),
                maximum_consecutive_amp_overflows=int(config["training"].get("maximum_consecutive_amp_overflows", 20)),
                diagnostic_limit=diagnostic_limit,
            )
            skipped = bool(step_result["update_skipped"])
            amp_overflows_total = int(step_result["amp_overflows_total"])
            amp_overflows_consecutive = int(step_result["amp_overflows_consecutive"])
            gradient_norms = step_result["gradient_norms"]
            latest_batch_evidence.update(
                {
                    "forward_loss_finite": True,
                    "loss_components": {
                        name: float(np.mean([item[name] for item in step_losses])) for name in step_losses[0]
                    },
                    "amp_scale_after": step_result["amp_scale_after"],
                    "amp_overflow_detected": skipped,
                    "gradient_evidence": step_result["gradient_evidence"],
                }
            )
            if skipped:
                overflow_diagnostics.append(dict(latest_batch_evidence))
                del overflow_diagnostics[:-diagnostic_limit]
            if not skipped:
                scheduler.step()
                optimizer_step += 1
            data_cursor += consumed
            processed_valid_tokens += effective_tokens
            dataset_pass = data_cursor // len(train_rows)
            at_optimizer_boundary = True
            microbatches_accumulated = 0
            if step_result["overflow_limit_exceeded"]:
                raise E006GradientError(
                    "E006 configured AMP overflow limit exceeded",
                    {
                        **latest_batch_evidence,
                        "failure_kind": "amp_overflow_limit",
                        "amp_overflows_total": amp_overflows_total,
                        "amp_overflows_consecutive": amp_overflows_consecutive,
                    },
                )
            _assert_finite_model(model)
            _assert_finite_optimizer(optimizer)
            memory = _memory_guard(config, device)
            metric = {
                "record_type": "training_update",
                "optimizer_step": optimizer_step,
                "microstep": microstep,
                "dataset_pass": dataset_pass,
                "processed_valid_tokens": processed_valid_tokens,
                "target_valid_tokens": target_valid_tokens,
                "valid_token_progress_fraction": processed_valid_tokens / max(target_valid_tokens, 1),
                "tokens_per_optimizer_step_by_length_bucket": {str(recommendation.maximum_length): effective_tokens},
                "loss_normalization": config["batching"].get(
                    "loss_normalization",
                    "valid_tokens_and_valid_pairs",
                ),
                **budget,
                "losses": {name: float(np.mean([item[name] for item in step_losses])) for name in step_losses[0]},
                "canonical_cross_entropy": float(np.mean([item["sequence"] for item in step_losses])),
                "context_contrast": float(np.mean([item.get("context_contrast", 0.0) for item in step_losses])),
                "weighted_context_contrast": float(
                    np.mean([item.get("context_contrast_weighted", 0.0) for item in step_losses])
                ),
                "total_loss": float(np.mean([item["total"] for item in step_losses])),
                "normal_minus_shuffled_batch_margin": float(
                    -np.mean([item.get("context_gap", 0.0) for item in step_losses])
                ),
                "gradient_norm": step_result["gradient_norm_preclip"],
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "amp_update_skipped": skipped,
                "amp_scale_before": step_result["amp_scale_before"],
                "amp_scale_after": step_result["amp_scale_after"],
                "amp_overflows_total": amp_overflows_total,
                "amp_overflows_consecutive": amp_overflows_consecutive,
                "batch_evidence": latest_batch_evidence,
                "gradient_norms": gradient_norms,
                "memory": memory,
            }
            append_rolling_training_metric(metrics_path, metric)
            normal_completion = data_cursor >= target_microsteps or optimizer_step >= total_updates
            validation_frequency = int(config["training"]["validation_frequency"])
            validation_due = normal_completion or (
                optimizer_step > 0 and optimizer_step % validation_frequency == 0 and not skipped
            )
            validation = None
            if validation_due:
                validation = _validation(
                    model,
                    validation_panel,
                    config,
                    device,
                    stage=mode,
                    step=optimizer_step,
                )
                _append_jsonl(validation_path, validation)
            contextual_record = None
            contextual_due = (
                monitored_v5
                and not skipped
                and (optimizer_step in monitoring_schedule or dataset_pass > previous_dataset_pass or normal_completion)
            )
            if contextual_due and optimizer_step not in monitored_steps:
                contextual_record = publish_context_monitor()
            immutable_reasons = set()
            if validation_due and config["training"]["immutable_checkpoint_on_validation"]:
                immutable_reasons.add("validation")
            if dataset_pass > previous_dataset_pass and config["training"]["immutable_checkpoint_on_pass_end"]:
                immutable_reasons.add("dataset_pass_end")
            if normal_completion:
                immutable_reasons.add("normal_completion")
            review_pause_due = (
                monitored_v5
                and not skipped
                and optimizer_step in set(config["training"]["review_pause_steps"])
                and review_decision is None
            )
            if review_pause_due:
                immutable_reasons.add("scientific_review_pause")
            payload = checkpoint_payload()
            immutable = immutable_for_step(payload, immutable_reasons) if immutable_reasons else None
            if validation is not None and validation["sequence_cross_entropy"] < best_sequence_ce:
                if immutable is None:
                    immutable = immutable_for_step(payload, {"best_selection"})
                else:
                    immutable = immutable_for_step(payload, {"best_selection"})
                best_sequence_ce = validation["sequence_cross_entropy"]
                payload = checkpoint_payload()
                best_record = publish_best_checkpoint(
                    checkpoint.parent,
                    immutable,
                    payload,
                    validation_sequence_cross_entropy=best_sequence_ce,
                )
            recovery_due = normal_completion or (
                optimizer_step > 0
                and optimizer_step % int(config["training"]["recovery_checkpoint_frequency"]) == 0
                and not skipped
            )
            if recovery_due:
                latest_record = publish_recovery_checkpoint(checkpoint.parent, checkpoint_payload())
            if immutable_reasons or recovery_due or validation is not None:
                publish_manifest()
            optimizer_progress = optimizer_step / total_updates
            token_progress = processed_valid_tokens / max(target_valid_tokens, 1)
            dataset_progress = data_cursor / target_microsteps
            progress = max(optimizer_progress, token_progress, dataset_progress)
            elapsed = time.monotonic() - started
            _atomic_json(
                heartbeat,
                {
                    "status": "running",
                    "stage": mode,
                    "optimizer_step": optimizer_step,
                    "microstep": microstep,
                    "data_cursor": data_cursor,
                    "dataset_pass": dataset_pass,
                    "processed_valid_tokens": processed_valid_tokens,
                    "target_valid_tokens": target_valid_tokens,
                    "optimizer_step_progress_fraction": optimizer_progress,
                    "valid_token_progress_fraction": token_progress,
                    "dataset_pass_progress_fraction": dataset_progress,
                    "percentage": progress * 100,
                    "eta_seconds": elapsed * (1 - progress) / max(progress, 1e-12),
                    "memory": memory,
                    "amp_scale": float(scaler.get_scale()) if amp_enabled else None,
                    "amp_overflows_total": amp_overflows_total,
                    "amp_overflows_consecutive": amp_overflows_consecutive,
                    "latest_batch_evidence": latest_batch_evidence,
                    "timestamp_utc": _utc_now(),
                },
            )
            if review_pause_due:
                if latest_record is None or int(latest_record["optimizer_step"]) != optimizer_step:
                    latest_record = publish_recovery_checkpoint(checkpoint.parent, checkpoint_payload())
                publish_manifest()
                pause = {
                    "status": "paused_for_scientific_review",
                    "version": PHASE3_PROTOCOL_VERSION,
                    "stage": mode,
                    "optimizer_step": optimizer_step,
                    "microstep": microstep,
                    "data_cursor": data_cursor,
                    "dataset_pass": dataset_pass,
                    "processed_valid_tokens": processed_valid_tokens,
                    "checkpoint_path": latest_record["path"],
                    "checkpoint_sha256": latest_record["sha256"],
                    "immutable_review_checkpoint": immutable,
                    "contextual_monitoring_record": contextual_record,
                    "configuration_sha256": config_hash,
                    "resumable": True,
                    "review_decision_required": True,
                    "review_decision_contract": {
                        "decision": "approve_continue",
                        "config_sha256": config_hash,
                        "review_checkpoint_sha256": immutable["sha256"],
                        "optimizer_step": optimizer_step,
                        "authorizes_training": False,
                        "authorizes_joint_training": False,
                    },
                    "authorizes_training": False,
                    "authorizes_joint_training": False,
                    "authorizes_definitive_evaluation": False,
                    "timestamp_utc": _utc_now(),
                }
                _atomic_json(summary_path, pause)
                _atomic_json(heartbeat, pause)
                return pause
            interrupt_after = config["training"].get("interrupt_after_optimizer_steps")
            if interrupt_after is not None and optimizer_step == int(interrupt_after):
                raise TrainingInterrupted("Configured synthetic interruption")
        if latest_record is None:
            latest_record = publish_recovery_checkpoint(checkpoint.parent, checkpoint_payload())
        best_record = finalize_best_checkpoint(
            checkpoint.parent,
            stage=mode,
            authorize=not monitored_v5,
        )
        selected_path = checkpoint.parent / "best.pt"
        checkpoint_sha = best_record["sha256"]
        publish_manifest()
        authorization_after = None if synthetic else _authorization(config)
        protected_after = _protected_hashes(authorization_after)
        if _dataset_identity(authorization_after) != dataset_identity or protected_after != protected_before:
            raise RuntimeError("E006 protected dataset changed during training")
        continuation_source_unchanged = None
        if migration_record is not None:
            source_after = _directory_fingerprint(migration_record["source_output_dir"])
            continuation_source_unchanged = source_after == migration_record["source_artifact_fingerprints"]
            if not continuation_source_unchanged:
                raise RuntimeError("E006 continuation source artifacts changed during training")
        report = {
            "status": "completed",
            "version": PHASE3_PROTOCOL_VERSION,
            "stage": mode,
            "architecture_version": E006_ARCHITECTURE_VERSION,
            "feature_version": RICH_FEATURE_VERSION,
            "configuration_sha256": config_hash,
            "dataset_identity": dataset_identity,
            "calibration_sha256": calibration_sha,
            "production_selection_sha256": production_selection_sha,
            "protected_input_hashes_before": protected_before,
            "protected_input_hashes_after": protected_after,
            "protected_inputs_unchanged": True,
            "continuation_lineage": migration_record,
            "continuation_source_unchanged": continuation_source_unchanged,
            "warm_start_provenance": warm_start_provenance,
            "pretraining_launch_gates": launch_gate_evidence,
            "contextual_monitoring_panel": contextual_monitor_identity,
            "contextual_monitoring_schedule": monitoring_schedule,
            "contextual_monitoring_path": str(contextual_monitoring_path) if monitored_v5 else None,
            "contextual_monitoring_sha256": (
                _sha256(contextual_monitoring_path) if monitored_v5 and contextual_monitoring_path.exists() else None
            ),
            "best_context_checkpoint": best_context_record,
            "review_decision": review_decision,
            "post_training_context_diagnostic": config.get("post_training_context_diagnostic"),
            "optimizer_steps": optimizer_step,
            "microstep": microstep,
            "data_cursor": data_cursor,
            "processed_valid_tokens": processed_valid_tokens,
            "target_valid_tokens": target_valid_tokens,
            "scheduler_progress": {
                "primary_unit": "optimizer_steps",
                "optimizer_steps": optimizer_step,
                "processed_valid_tokens": processed_valid_tokens,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "learning_rate_scaled_for_batch_size": False,
            },
            "dataset_passes_completed": dataset_pass,
            "amp_scale": float(scaler.get_scale()) if amp_enabled else None,
            "amp_overflows_total": amp_overflows_total,
            "amp_overflows_consecutive": amp_overflows_consecutive,
            "overflow_diagnostics": overflow_diagnostics,
            "best_sequence_cross_entropy": best_sequence_ce,
            "best_checkpoint_criterion": "validation_sequence_cross_entropy",
            "checkpoint_path": str(selected_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_manifest": checkpoint_manifest,
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_path),
            "checkpoint_storage_preflight": checkpoint_storage,
            "metrics_path": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "validation_journal_path": str(validation_path),
            "validation_journal_sha256": _sha256(validation_path),
            "authorizes_joint_training": mode == "sequence-pretrain" and not monitored_v5,
            "authorizes_training": mode == "sequence-pretrain" and not monitored_v5,
            "authorizes_definitive_evaluation": mode == "joint-train" and not monitored_v5,
            "memory": _memory(device),
            "completed_utc": _utc_now(),
        }
        _atomic_json(summary_path, report)
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "completed_utc": report["completed_utc"],
                "protocol_sha256": _sha256(summary_path),
                "authorizes_training": report["authorizes_training"],
                "authorizes_joint_training": report["authorizes_joint_training"],
                "authorizes_definitive_evaluation": report["authorizes_definitive_evaluation"],
            },
        )
        return report
    except BaseException as error:
        recovery_write_error = None
        if "model" in locals():
            if at_optimizer_boundary:
                try:
                    latest_record = publish_recovery_checkpoint(checkpoint.parent, checkpoint_payload())
                    publish_manifest()
                except BaseException as recovery_error:
                    recovery_write_error = f"{type(recovery_error).__name__}: {str(recovery_error)[:500]}"
            else:
                recovery_write_error = "inflight_accumulation_preserved_previous_valid_recovery_checkpoint"
        failure_timestamp = _utc_now()
        failure_diagnostics = getattr(error, "diagnostics", None)
        try:
            failure_memory = _memory(device)
        except BaseException:
            failure_memory = None
        failure = {
            "status": "interrupted" if isinstance(error, TrainingInterrupted) else "failed",
            "version": PHASE3_PROTOCOL_VERSION,
            "stage": mode,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "optimizer_step": optimizer_step,
            "microstep": microstep,
            "data_cursor": data_cursor,
            "dataset_pass": dataset_pass,
            "processed_valid_tokens": processed_valid_tokens,
            "target_valid_tokens": target_valid_tokens,
            "at_optimizer_boundary": at_optimizer_boundary,
            "microbatches_accumulated": microbatches_accumulated,
            "amp_scale": float(scaler.get_scale()) if amp_enabled else None,
            "amp_overflows_total": amp_overflows_total,
            "amp_overflows_consecutive": amp_overflows_consecutive,
            "overflow_diagnostics": overflow_diagnostics,
            "latest_batch_evidence": latest_batch_evidence,
            "failure_diagnostics": failure_diagnostics,
            "memory": failure_memory,
            "latest_checkpoint": str(checkpoint) if checkpoint.exists() else None,
            "latest_checkpoint_metadata": latest_record,
            "recovery_checkpoint_write_error": recovery_write_error,
            "authorizes_joint_training": False,
            "authorizes_training": False,
            "authorizes_definitive_evaluation": False,
            "timestamp_utc": failure_timestamp,
            "completed_utc": failure_timestamp,
        }
        _atomic_json(summary_path, failure)
        _atomic_json(heartbeat, failure)
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def run_training_stage(
    config_path: str | Path,
    *,
    mode: str,
    resume: bool = False,
    synthetic: bool = False,
    continuation: bool = False,
    review_decision_path: str | Path | None = None,
    expected_review_decision_sha256: str | None = None,
) -> dict[str, Any]:
    """Run one stage and publish a terminal non-authorizing record for preflight failures."""
    try:
        return _run_training_stage(
            config_path,
            mode=mode,
            resume=resume,
            synthetic=synthetic,
            continuation=continuation,
            review_decision_path=review_decision_path,
            expected_review_decision_sha256=expected_review_decision_sha256,
        )
    except BaseException as error:
        try:
            config = load_yaml(config_path)
            output = Path(config.get("training", {}).get("output_dir", ""))
            if (
                config.get("production_monitoring_version")
                in {"e006_stage_a_v5_monitored_training_v1", "e006_stage_a_v6_monitored_training_v1"}
                and not output.exists()
            ):
                raise
            if continuation and not output.exists():
                raise
            if output:
                protocol_path = output / "protocol.json"
                completed = (
                    protocol_path.exists() and json.loads(protocol_path.read_text()).get("status") == "completed"
                )
                if not completed and not protocol_path.exists():
                    failure = {
                        "status": "interrupted" if isinstance(error, TrainingInterrupted) else "failed",
                        "version": PHASE3_PROTOCOL_VERSION,
                        "stage": mode,
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                        "authorizes_training": False,
                        "authorizes_joint_training": False,
                        "authorizes_definitive_evaluation": False,
                        "completed_utc": _utc_now(),
                    }
                    _atomic_json(protocol_path, failure)
                    _atomic_json(output / "heartbeat.json", failure)
        except BaseException:
            pass
        raise


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.suffix == ".pt":
        payload = load_checkpoint(artifact, map_location="cpu")
        return {
            "path": str(artifact),
            "sha256": _sha256(artifact),
            **{key: payload.get(key) for key in ("version", "stage", "status", "optimizer_step")},
        }
    return {"path": str(artifact), "sha256": _sha256(artifact), "payload": json.loads(artifact.read_text())}
