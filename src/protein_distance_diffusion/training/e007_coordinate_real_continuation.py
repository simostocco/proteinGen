"""Exact-state continuation of the completed E007 Phase-3F pilot."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file
from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import _parameter_sha256
from protein_distance_diffusion.training.e007_coordinate_real_pilot import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_SCALE,
    NON_AUTHORIZING,
    _atomic_json,
    _atomic_torch,
    _authorize,
    _evaluate_panel,
    _gradient_evidence,
    _real_reference_distributions,
    _regime_for_stratum,
    _restore_rng,
    _rng_state,
    _sample_panel,
    _scheduler_factor,
    _select_rows,
    make_uniform_training_corruption,
    planned_batch_accounting,
    prepare_coordinate_batch,
    uniform_coordinate_v_mse,
    update_stratum,
    verify_pilot_prerequisites,
)
from protein_distance_diffusion.training.e007_coordinate_real_pilot import (
    _load_config as _load_phase3f_config,
)

VERSION = "e007_coordinate_real_continuation_to_10000_v1"
SOURCE_VERSION = "e007_coordinate_real_pilot_v1"
REQUIRED_STATE_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "rng_state",
    "optimizer_update",
    "sampler_cursor",
    "samples_processed",
    "valid_residues_processed",
    "successful_optimizer_boundary",
}
LONG_STRATUM = "385-500"
MEDIUM_LONG_STRATUM = "257-384"
MAXIMUM_IDLE_ALLOCATED_MIB = 512.0
ALLOCATOR_POLICY = "pre_and_post_385_500_v2"
DIAGNOSTIC_PADDED_LENGTH_SEQUENCE = (
    64,
    96,
    176,
    264,
    448,
    64,
    128,
    208,
    312,
    488,
    64,
    128,
    192,
    264,
    400,
    56,
    96,
    144,
    352,
    448,
    64,
    104,
    248,
    360,
    472,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _directory_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{path.stat().st_size}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def published_paths(value: Any, staging: Path, output: Path) -> Any:
    if isinstance(value, dict):
        return {key: published_paths(item, staging, output) for key, item in value.items()}
    if isinstance(value, list):
        return [published_paths(item, staging, output) for item in value]
    if isinstance(value, str) and (value == str(staging) or value.startswith(f"{staging}{os.sep}")):
        return f"{output}{value[len(str(staging)) :]}"
    return value


def _cuda_memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "current_cuda_allocated_mib": None,
            "current_cuda_reserved_mib": None,
            "phase_peak_cuda_allocated_mib": None,
            "phase_peak_cuda_reserved_mib": None,
            "active_cuda_mib": None,
            "inactive_split_cuda_mib": None,
            "unclassified_reserved_cuda_mib": None,
            "cuda_segment_count": None,
        }
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    active_bytes = int(stats.get("active_bytes.all.current", 0))
    inactive_split_bytes = int(stats.get("inactive_split_bytes.all.current", 0))
    reserved_bytes = int(stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved(device)))
    return {
        "current_cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
        "current_cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
        "phase_peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "phase_peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "active_cuda_mib": active_bytes / 2**20,
        "inactive_split_cuda_mib": inactive_split_bytes / 2**20,
        "unclassified_reserved_cuda_mib": max(reserved_bytes - active_bytes - inactive_split_bytes, 0) / 2**20,
        "cuda_segment_count": int(stats.get("segment.all.current", 0)),
    }


def allocator_cleanup(
    device: torch.device,
    *,
    active_computation: bool,
    padded_length: int,
    event_type: str = "unspecified",
    optimizer_update: int | None = None,
    length_stratum: str | None = None,
) -> dict[str, Any]:
    """Release inactive CUDA cache without touching model, optimizer, or RNG state."""
    if active_computation:
        raise RuntimeError("E007 allocator cleanup is forbidden during active computation")
    if device.type != "cuda":
        raise ValueError("E007 allocator cleanup requires CUDA")
    before = _cuda_memory_snapshot(device)
    gc.collect()
    torch.cuda.empty_cache()
    after = _cuda_memory_snapshot(device)
    return {
        "event_type": event_type,
        "optimizer_update": optimizer_update,
        "length_stratum": length_stratum,
        "padded_length": int(padded_length),
        "before": before,
        "after": after,
        "allocated_mib_released": float(before["current_cuda_allocated_mib"])
        - float(after["current_cuda_allocated_mib"]),
        "reserved_mib_released": float(before["current_cuda_reserved_mib"]) - float(after["current_cuda_reserved_mib"]),
    }


def post_cleanup_reserved_ceiling_mib(config: dict[str, Any]) -> float:
    allocated = float(config["memory"]["maximum_cuda_allocated_mib"])
    reserved = float(config["memory"]["maximum_cuda_reserved_mib"])
    ceiling = reserved - allocated
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("E007 CUDA limits do not provide positive post-clean reserved headroom")
    return ceiling


def cleanup_reservations_show_monotonic_growth(values: list[float]) -> bool:
    """Detect sustained growth while permitting stable non-releasable segments."""
    if len(values) < 3:
        return False
    recent = values[-3:]
    return all(second > first for first, second in zip(recent, recent[1:], strict=False))


def assess_allocator_cleanup(
    event: dict[str, Any],
    config: dict[str, Any],
    *,
    prior_post_cleanup_reserved_mib: list[float],
) -> dict[str, Any]:
    """Attach portable cleanup-health evidence without raising."""
    assessed = copy.deepcopy(event)
    after_allocated = float(assessed["after"]["current_cuda_allocated_mib"])
    after_reserved = float(assessed["after"]["current_cuda_reserved_mib"])
    reserved_ceiling = post_cleanup_reserved_ceiling_mib(config)
    failures = []
    if not np.isfinite(after_allocated):
        failures.append("post_cleanup_allocated_nonfinite")
    elif after_allocated > MAXIMUM_IDLE_ALLOCATED_MIB:
        failures.append("post_cleanup_allocated_above_persistent_state_ceiling")
    if not np.isfinite(after_reserved):
        failures.append("post_cleanup_reserved_nonfinite")
    elif after_reserved > reserved_ceiling:
        failures.append("post_cleanup_reserved_above_derived_headroom_ceiling")
    if assessed.get("captured_memory_gate_passed") is False:
        failures.append("phase_memory_gate_failed")
    post_reservations = [*prior_post_cleanup_reserved_mib, after_reserved]
    if assessed.get("event_type") == "post_update" and cleanup_reservations_show_monotonic_growth(post_reservations):
        failures.append("post_cleanup_reserved_monotonic_growth")
    assessed.update(
        {
            "event_status": "failed" if failures else "passed",
            "post_cleanup_allocated_ceiling_mib": MAXIMUM_IDLE_ALLOCATED_MIB,
            "post_cleanup_reserved_ceiling_mib": reserved_ceiling,
            "post_cleanup_reserved_headroom_mib": reserved_ceiling - after_reserved,
            "post_cleanup_reservation_history_mib": post_reservations[-10:],
            "failure_reason": ";".join(failures) if failures else None,
        }
    )
    return assessed


def publish_and_enforce_allocator_cleanup(
    path: Path,
    event: dict[str, Any],
    config: dict[str, Any],
    *,
    prior_post_cleanup_reserved_mib: list[float],
) -> dict[str, Any]:
    """Durably publish cleanup evidence before enforcing its health contract."""
    assessed = assess_allocator_cleanup(
        event,
        config,
        prior_post_cleanup_reserved_mib=prior_post_cleanup_reserved_mib,
    )
    _append_jsonl_fsync(path, assessed)
    if assessed["event_status"] != "passed":
        raise MemoryError(f"E007 allocator cleanup health check failed: {assessed['failure_reason']}")
    return assessed


def allocator_cleanup_actions(policy: str, length_stratum: str) -> tuple[bool, bool]:
    """Return pre/post cleanup actions without making canonical split names ambiguous."""
    if policy == ALLOCATOR_POLICY:
        is_long = length_stratum == LONG_STRATUM
        return is_long, is_long
    if policy == "pre_medium_and_long":
        return length_stratum in {MEDIUM_LONG_STRATUM, LONG_STRATUM}, False
    raise ValueError(f"Unknown E007 allocator lifecycle policy: {policy}")


def memory_gate_passes(memory: dict[str, float | None], config: dict[str, Any]) -> bool:
    return (
        float(memory["phase_peak_cuda_allocated_mib"]) <= float(config["memory"]["maximum_cuda_allocated_mib"])
        and float(memory["phase_peak_cuda_reserved_mib"]) <= float(config["memory"]["maximum_cuda_reserved_mib"])
        and float(memory["current_cuda_reserved_mib"]) <= float(config["memory"]["maximum_cuda_reserved_mib"])
    )


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_jsonl_fsync(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def diagnose_resume_boundary(payload: dict[str, Any], metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    update = int(payload["optimizer_update"])
    if int(payload["sampler_cursor"]) != update or not payload["successful_optimizer_boundary"]:
        raise ValueError("E007 continuation checkpoint is not an atomic optimizer boundary")
    optimizer_steps = {
        int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
        for value in payload["optimizer"]["state"].values()
        if "step" in value
    }
    if optimizer_steps != {update}:
        raise ValueError(f"E007 continuation optimizer-state step contradiction: {sorted(optimizer_steps)}")
    scheduler = payload["scheduler"]
    if int(scheduler.get("last_epoch", -1)) != update or int(scheduler.get("_step_count", -1)) != update + 1:
        raise ValueError("E007 continuation scheduler state contradicts optimizer update")
    if any(not bool(torch.isfinite(value).all()) for value in payload["model"].values() if value.is_floating_point()):
        raise ValueError("E007 continuation checkpoint contains non-finite model state")
    metric_updates = [int(row["global_optimizer_update"]) for row in metric_rows]
    if metric_updates != sorted(set(metric_updates)):
        raise ValueError("E007 continuation metrics have duplicate or unordered optimizer updates")
    last_metric = metric_updates[-1] if metric_updates else 1000
    gap = list(range(last_metric + 1, update + 1))
    if len(gap) > 1:
        raise ValueError(f"E007 continuation has an unbounded metric/checkpoint gap: {gap[:20]}")
    last_metric_row = metric_rows[-1] if metric_rows and last_metric == update else None
    if (
        payload.get("last_committed_metric_update") is not None
        and int(payload["last_committed_metric_update"]) != last_metric
    ):
        raise ValueError("E007 continuation checkpoint/metric update contradiction")
    if payload.get("last_memory_gate_passed") is not None and last_metric_row is not None:
        if bool(payload["last_memory_gate_passed"]) != bool(last_metric_row["memory_gate_passed"]):
            raise ValueError("E007 continuation checkpoint/metric memory-gate contradiction")
    return {
        "atomic_optimizer_boundary": True,
        "optimizer_update": update,
        "sampler_cursor": int(payload["sampler_cursor"]),
        "samples_processed": int(payload["samples_processed"]),
        "valid_residues_processed": int(payload["valid_residues_processed"]),
        "optimizer_state_steps": sorted(optimizer_steps),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "last_metric_update": last_metric,
        "last_memory_gate_passed": None if last_metric_row is None else bool(last_metric_row["memory_gate_passed"]),
        "missing_metric_updates": gap,
        "next_optimizer_update": update + 1,
        "metric_reconstruction_permitted": False,
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 continuation configuration version contradiction")
    source = config.get("source", {})
    if int(source.get("start_global_step", -1)) != 1000:
        raise ValueError("E007 continuation must start at global step 1000")
    if int(config.get("stop_global_step", -1)) != 10000:
        raise ValueError("E007 continuation must stop at global step 10000")
    if int(config.get("additional_successful_updates", -1)) != 9000:
        raise ValueError("E007 continuation must perform 9000 additional successful updates")
    if int(config.get("mandatory_scientific_review_at_global_step", -1)) != 10000:
        raise ValueError("E007 continuation review boundary changed")
    if config.get("sampling_monitor_updates") != [2500, 5000, 7500, 10000]:
        raise ValueError("E007 continuation sampling-monitor schedule changed")
    monitor = config.get("sampling_monitor", {})
    if monitor.get("lengths") != [64, 128, 256, 384, 500] or int(monitor.get("samples_per_length", 0)) != 2:
        raise ValueError("E007 continuation sampling-monitor panel changed")
    for key, expected in {
        "recovery_checkpoint_frequency": 500,
        "immutable_checkpoint_frequency": 1000,
        "validation_frequency": 1000,
    }.items():
        if int(config.get(key, 0)) != expected:
            raise ValueError(f"E007 continuation {key} changed")
    if not config.get("failure_policy", {}).get("sampling_quality_is_not_an_automatic_stop"):
        raise ValueError("E007 continuation sampling monitor cannot be an automatic stop")
    if any(bool(config.get("authorization", {}).get(key)) for key in NON_AUTHORIZING if key.startswith("authorizes_")):
        raise ValueError("E007 continuation configuration unexpectedly authorizes training")
    return config


def continuation_schedule(config: dict[str, Any]) -> dict[str, list[int]]:
    start = int(config["source"]["start_global_step"])
    stop = int(config["stop_global_step"])
    recovery = list(
        range(
            start + int(config["recovery_checkpoint_frequency"]), stop + 1, int(config["recovery_checkpoint_frequency"])
        )
    )
    validation = list(range(start + int(config["validation_frequency"]), stop + 1, int(config["validation_frequency"])))
    immutable = sorted(set(validation) | set(config["sampling_monitor_updates"]))
    return {
        "recovery": recovery,
        "immutable": immutable,
        "validation": validation,
        "sampling": list(config["sampling_monitor_updates"]),
    }


def monitor_seed_records(config: dict[str, Any]) -> list[dict[str, int | str]]:
    seed = int(config["sampling_monitor"]["seed"])
    records = []
    for length_index, length in enumerate(config["sampling_monitor"]["lengths"]):
        for sample_index in range(int(config["sampling_monitor"]["samples_per_length"])):
            sample_seed = seed + length_index * 100 + sample_index
            records.append(
                {
                    "length": int(length),
                    "sample_index": sample_index,
                    "seed": sample_seed,
                    "noise_identity_sha256": hashlib.sha256(
                        f"{length}|{sample_index}|{sample_seed}".encode()
                    ).hexdigest(),
                }
            )
    return records


def _verify_source_files(config: dict[str, Any]) -> dict[str, str]:
    source = config["source"]
    paths = {
        "config": Path(source["config_path"]),
        "report": Path(source["output_dir"]) / "report.json",
        "protocol": Path(source["output_dir"]) / "protocol.json",
        "panel_manifest": Path(source["output_dir"]) / "panel_manifest.json",
        "evaluations": Path(source["output_dir"]) / "evaluations.json",
        "checkpoint": Path(source["checkpoint_path"]),
    }
    expected = {
        "config": source["config_sha256"],
        "report": source["report_sha256"],
        "protocol": source["protocol_sha256"],
        "panel_manifest": source["panel_manifest_sha256"],
        "evaluations": source["evaluations_sha256"],
        "checkpoint": source["checkpoint_sha256"],
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, observed in hashes.items():
        if observed != expected[name]:
            raise ValueError(f"E007 continuation source {name} hash contradiction")
    report = json.loads(paths["report"].read_text())
    protocol = json.loads(paths["protocol"].read_text())
    if int(report.get("optimizer_updates", -1)) != 1000 or int(protocol.get("optimizer_updates", -1)) != 1000:
        raise ValueError("E007 continuation source is not the completed step-1000 pilot")
    return hashes


def inspect_checkpoint(path: str | Path, config: dict[str, Any], *, continuation: bool = False) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    missing = sorted(REQUIRED_STATE_KEYS - payload.keys())
    if missing:
        raise ValueError(f"E007 continuation checkpoint is partial; missing {missing}")
    if not payload["successful_optimizer_boundary"]:
        raise ValueError("E007 continuation checkpoint is not at an optimizer boundary")
    if int(payload["optimizer_update"]) != int(payload["sampler_cursor"]):
        raise ValueError("E007 continuation checkpoint cursor contradicts optimizer step")
    if not isinstance(payload["rng_state"], dict) or not {"python", "numpy", "torch"}.issubset(payload["rng_state"]):
        raise ValueError("E007 continuation checkpoint has incomplete RNG state")
    if continuation:
        if payload.get("version") != VERSION or payload.get("continuation_configuration_sha256") != config["_sha256"]:
            raise ValueError("E007 continuation recovery checkpoint identity contradiction")
        if payload.get("source_checkpoint_sha256") != config["source"]["checkpoint_sha256"]:
            raise ValueError("E007 continuation recovery checkpoint source contradiction")
    else:
        source = config["source"]
        if payload.get("version") != SOURCE_VERSION or payload.get("configuration_sha256") != source["config_sha256"]:
            raise ValueError("E007 continuation source checkpoint identity contradiction")
        expected = {
            "optimizer_update": source["start_global_step"],
            "sampler_cursor": source["start_global_step"],
            "samples_processed": source["samples_processed"],
            "valid_residues_processed": source["valid_residues_processed"],
        }
        for key, value in expected.items():
            if int(payload[key]) != int(value):
                raise ValueError(f"E007 continuation source checkpoint {key} contradiction")
    return payload


def verify_source_checkpoint_protection(payload: dict[str, Any], prerequisites: dict[str, Any]) -> None:
    expected = _canonical_sha(prerequisites["hashes"])
    if payload.get("protected_hashes_sha256") != expected:
        raise ValueError("E007 continuation source checkpoint protected-input identity contradiction")


def continuation_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    update: int,
    samples_processed: int,
    valid_residues_processed: int,
    config: dict[str, Any],
    protected_hashes_sha256: str,
    last_committed_metric_update: int | None = None,
    last_memory_gate_passed: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "version": VERSION,
        "continuation_configuration_sha256": config["_sha256"],
        "source_checkpoint_sha256": config["source"]["checkpoint_sha256"],
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
    if last_committed_metric_update is not None:
        payload["last_committed_metric_update"] = int(last_committed_metric_update)
    if last_memory_gate_passed is not None:
        payload["last_memory_gate_passed"] = bool(last_memory_gate_passed)
    return payload


def restore_full_state(
    payload: dict[str, Any], model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any
) -> None:
    missing = sorted(REQUIRED_STATE_KEYS - payload.keys())
    if missing:
        raise ValueError(f"E007 continuation refuses partial state restoration; missing {missing}")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng_state"])


def verify_stream_prefix(
    source_panel: dict[str, Any], selected: dict[str, Any], source_config: dict[str, Any]
) -> dict[str, str]:
    source_counts = planned_batch_accounting(source_config)["planned_samples_by_stratum"]
    hashes = {}
    for name, count in source_counts.items():
        expected = list(source_panel["train"][name])[: int(count)]
        observed = [str(row["sample_id"]) for row in selected["train"][name]][: int(count)]
        if observed != expected:
            raise ValueError(f"E007 continuation training-stream prefix contradiction: {name}")
        hashes[name] = _canonical_sha(observed)
    observed_validation = {
        name: [str(row["sample_id"]) for row in rows] for name, rows in selected["validation"].items()
    }
    if observed_validation != source_panel["validation"]:
        raise ValueError("E007 continuation validation-panel contradiction")
    return hashes


def _source_scientific_config(config: dict[str, Any], *, extended: bool) -> dict[str, Any]:
    source = _load_phase3f_config(config["source"]["config_path"])
    if extended:
        source = copy.deepcopy(source)
        source["successful_optimizer_updates"] = int(config["stop_global_step"])
        source["sampling_seed"] = int(config["sampling_monitor"]["seed"])
        source["sampling_samples_per_stratum"] = int(config["sampling_monitor"]["samples_per_length"])
    return source


def plan_coordinate_continuation(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = _load_config(path)
    config["_sha256"] = sha256_file(path)
    source_hashes = _verify_source_files(config)
    source_config = _source_scientific_config(config, extended=False)
    prerequisites = verify_pilot_prerequisites(source_config)
    checkpoint = inspect_checkpoint(config["source"]["checkpoint_path"], config)
    verify_source_checkpoint_protection(checkpoint, prerequisites)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists():
        raise FileExistsError(f"E007 continuation output already exists: {output}")
    resume_diagnosis = None
    if staging.exists():
        latest = staging / "checkpoints" / "latest.pt"
        if not latest.is_file():
            raise FileNotFoundError("E007 continuation staging exists without a recovery checkpoint")
        recovery = inspect_checkpoint(latest, config, continuation=True)
        resume_diagnosis = diagnose_resume_boundary(recovery, _read_metric_rows(staging / "metrics.jsonl"))
        del recovery
        gc.collect()
    schedule = continuation_schedule(config)
    accounting_config = _source_scientific_config(config, extended=True)
    accounting = planned_batch_accounting(accounting_config)
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": config["_sha256"],
        "output_dir": str(output),
        "source_hashes": source_hashes,
        "source_state": {
            key: int(checkpoint[key])
            for key in ("optimizer_update", "sampler_cursor", "samples_processed", "valid_residues_processed")
        },
        "start_global_step": 1000,
        "stop_global_step": 10000,
        "additional_successful_updates": 9000,
        "schedule": schedule,
        "full_stream_accounting": accounting,
        "batch_regimes": source_config["batch_regimes"],
        "model_contract": {"name": "EquivariantPairCoordinateUNet", "parameter_count": EXPECTED_PARAMETER_COUNT},
        "objective": source_config["objective"],
        "coordinate_scale_angstrom": EXPECTED_SCALE,
        "sampling_monitor": {
            **config["sampling_monitor"],
            "record_count_per_checkpoint": 10,
            "fixed_noise_identity_sha256": _canonical_sha(monitor_seed_records(config)),
            "descriptive_only": True,
        },
        "source_prerequisite_hashes": prerequisites["hashes"],
        "checkpoint_restore": "full_state_only",
        "existing_staging_resume_diagnosis": resume_diagnosis,
        "model_created": False,
        "optimizer_created": False,
        "cuda_tensors_allocated": False,
        "training_performed": False,
        **NON_AUTHORIZING,
    }


def _sampling_objectives(sampling: dict[str, Any]) -> dict[str, float]:
    records = sampling["records"]
    return {
        key: float(np.mean([float(row[key]) for row in records]))
        for key in (
            "adjacent_reference_error_angstrom",
            "radius_of_gyration_reference_relative_error",
            "clash_reference_error",
            "contact_density_reference_error",
        )
    }


def nondominated_sampling(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    objectives = (
        "adjacent_reference_error_angstrom",
        "radius_of_gyration_reference_relative_error",
        "clash_reference_error",
        "contact_density_reference_error",
    )
    result = []
    for candidate in records:
        dominated = any(
            all(other[key] <= candidate[key] for key in objectives)
            and any(other[key] < candidate[key] for key in objectives)
            for other in records
            if other is not candidate
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda row: row["optimizer_update"])


def _save_recovery(path: Path, payload: dict[str, Any]) -> None:
    _atomic_torch(path, payload)
    _atomic_json(
        path.with_suffix(".json"),
        {"optimizer_update": payload["optimizer_update"], "sha256": sha256_file(path), **NON_AUTHORIZING},
    )


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def _gradient_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise FloatingPointError(f"E007 diagnostic missing gradient: {name}")
        digest.update(name.encode())
        digest.update(parameter.grad.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().contiguous().cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        else:
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _diagnostic_stratum(padded_length: int) -> str:
    if padded_length <= 64:
        return "20-64"
    if padded_length <= 128:
        return "65-128"
    if padded_length <= 256:
        return "129-256"
    if padded_length <= 384:
        return MEDIUM_LONG_STRATUM
    return LONG_STRATUM


def _diagnostic_sequence(repeated_cycles: int) -> list[int]:
    repeated_tail = [64, 104, 248, 360, 472]
    return [*DIAGNOSTIC_PADDED_LENGTH_SEQUENCE, *(repeated_tail * repeated_cycles)]


def _post_cleanup_reservations(events: list[dict[str, Any]]) -> list[float]:
    return [
        float(event["after"]["current_cuda_reserved_mib"])
        for event in events
        if event.get("event_type") == "post_update"
    ]


def _diagnostic_case(config: dict[str, Any], *, policy: str, repeated_cycles: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("E007 allocator diagnostic requires CUDA")
    if policy not in {ALLOCATOR_POLICY, "pre_medium_and_long"}:
        raise ValueError(f"Unknown E007 allocator diagnostic policy: {policy}")
    device = torch.device("cuda")
    source_config = _source_scientific_config(config, extended=False)
    staging = Path(config["output_dir"]).with_name(f".{Path(config['output_dir']).name}.inprogress")
    checkpoint_path = staging / "checkpoints" / "latest.pt"
    checkpoint = inspect_checkpoint(checkpoint_path, config, continuation=True)
    records = []
    cleanup_events = []
    process_peak_allocated = 0.0
    process_peak_reserved = 0.0
    with coordinate_model_execution_context(source_config["numerics"], device):
        model = EquivariantPairCoordinateUNet(**source_config["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(source_config["optimizer"]["learning_rate"]),
            weight_decay=float(source_config["optimizer"]["weight_decay"]),
            betas=tuple(source_config["optimizer"]["betas"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _scheduler_factor(step, source_config))
        restore_full_state(checkpoint, model, optimizer, scheduler)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        parameter_sha_before = _parameter_sha256(model)
        optimizer_sha_before = _state_sha256(optimizer.state_dict())
        cpu_rng_before = _tensor_sha256(torch.get_rng_state())
        cuda_rng_before = [_tensor_sha256(value) for value in torch.cuda.get_rng_state_all()]
        diffusion = CoordinateVPDiffusion(int(source_config["diffusion_steps"]))
        sequence = _diagnostic_sequence(repeated_cycles)
        for sequence_index, padded_length in enumerate(sequence):
            model.zero_grad(set_to_none=True)
            stratum = _diagnostic_stratum(padded_length)
            batch_size = int(_regime_for_stratum(stratum, source_config)["physical_batch_size"])
            pre_cleanup, post_cleanup = allocator_cleanup_actions(policy, stratum)
            if pre_cleanup:
                event = allocator_cleanup(
                    device,
                    active_computation=False,
                    padded_length=padded_length,
                    event_type="pre_update",
                    optimizer_update=sequence_index,
                    length_stratum=stratum,
                )
                event = assess_allocator_cleanup(
                    event,
                    config,
                    prior_post_cleanup_reserved_mib=_post_cleanup_reservations(cleanup_events),
                )
                cleanup_events.append(event)
                if event["event_status"] != "passed":
                    break
            torch.cuda.reset_peak_memory_stats(device)
            rows = []
            for batch_index in range(batch_size):
                generator = torch.Generator(device="cpu").manual_seed(880000 + sequence_index * 100 + batch_index)
                coordinates = torch.randn((padded_length, 3), generator=generator, dtype=torch.float32)
                coordinates = (coordinates - coordinates.mean(dim=0, keepdim=True)) * 3.0
                rows.append(
                    {
                        "sample_id": f"diagnostic-{sequence_index}-{batch_index}",
                        "sequence_length": padded_length,
                        "accepted_contiguous_single_chain": True,
                        "coordinates": coordinates,
                        "residue_mask": torch.ones(padded_length, dtype=torch.bool),
                        "chain_continuity_mask": torch.ones(padded_length - 1, dtype=torch.bool),
                    }
                )
            prepared = prepare_coordinate_batch(rows, EXPECTED_SCALE, int(source_config["expected_downsample_factor"]))
            corruption = make_uniform_training_corruption(
                prepared,
                diffusion,
                seed=990000 + sequence_index,
                device=device,
            )
            model.train()
            prediction = model(
                corruption["batch"].noisy_coordinates,
                corruption["batch"].timesteps,
                prepared["lengths"].to(device),
                corruption["mask"],
                prepared["chain_continuity_mask"].to(device),
            )["v_prediction"]
            loss = uniform_coordinate_v_mse(prediction, corruption["batch"].coordinate_v_target, corruption["mask"])
            loss.backward()
            gradients = _gradient_evidence(model)
            gradient_sha256 = _gradient_sha256(model)
            loss_value = float(loss.detach().cpu())
            phase_memory = _cuda_memory_snapshot(device)
            process_peak_allocated = max(process_peak_allocated, float(phase_memory["phase_peak_cuda_allocated_mib"]))
            process_peak_reserved = max(process_peak_reserved, float(phase_memory["phase_peak_cuda_reserved_mib"]))
            memory_gate_passed = memory_gate_passes(phase_memory, config)
            record = {
                "sequence_index": sequence_index,
                "padded_length": padded_length,
                "length_stratum": stratum,
                "physical_batch_size": batch_size,
                "loss": loss_value,
                "gradient_sha256": gradient_sha256,
                "gradient_norm": gradients["global_norm"],
                "pre_cleanup_applied": pre_cleanup,
                "post_cleanup_applied": False,
                "memory_gate_passed": memory_gate_passed,
                **phase_memory,
            }
            model.zero_grad(set_to_none=True)
            del prediction, loss, corruption, prepared, rows, gradients
            gc.collect()
            if post_cleanup:
                event = allocator_cleanup(
                    device,
                    active_computation=False,
                    padded_length=padded_length,
                    event_type="post_update",
                    optimizer_update=sequence_index,
                    length_stratum=stratum,
                )
                event["captured_phase_peak_cuda_allocated_mib"] = phase_memory["phase_peak_cuda_allocated_mib"]
                event["captured_phase_peak_cuda_reserved_mib"] = phase_memory["phase_peak_cuda_reserved_mib"]
                event["captured_memory_gate_passed"] = memory_gate_passed
                event = assess_allocator_cleanup(
                    event,
                    config,
                    prior_post_cleanup_reserved_mib=_post_cleanup_reservations(cleanup_events),
                )
                cleanup_events.append(event)
                record["post_cleanup_applied"] = True
                record["post_cleanup_current_cuda_reserved_mib"] = event["after"]["current_cuda_reserved_mib"]
                record["post_cleanup_event_status"] = event["event_status"]
            records.append(record)
            if not memory_gate_passed or (post_cleanup and event["event_status"] != "passed"):
                break
        parameter_sha_after = _parameter_sha256(model)
        optimizer_sha_after = _state_sha256(optimizer.state_dict())
        cpu_rng_after = _tensor_sha256(torch.get_rng_state())
        cuda_rng_after = [_tensor_sha256(value) for value in torch.cuda.get_rng_state_all()]
    del model, optimizer, scheduler, diffusion
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "policy": policy,
        "padded_length_sequence": sequence,
        "repeated_cycles": repeated_cycles,
        "completed_records": len(records),
        "records": records,
        "cleanup_events": cleanup_events,
        "process_peak_cuda_allocated_mib": process_peak_allocated,
        "process_peak_cuda_reserved_mib": process_peak_reserved,
        "parameter_sha256_before": parameter_sha_before,
        "parameter_sha256_after": parameter_sha_after,
        "parameters_unchanged": parameter_sha_before == parameter_sha_after,
        "optimizer_sha256_before": optimizer_sha_before,
        "optimizer_sha256_after": optimizer_sha_after,
        "optimizer_unchanged": optimizer_sha_before == optimizer_sha_after,
        "cpu_rng_unchanged": cpu_rng_before == cpu_rng_after,
        "cuda_rng_unchanged": cuda_rng_before == cuda_rng_after,
        "optimizer_created": True,
        "optimizer_updates": 0,
    }


def run_allocator_diagnostic(
    config_path: str | Path,
    *,
    report_path: str | Path,
    cycles: int = 2,
) -> dict[str, Any]:
    """Run a bounded, non-optimizing CUDA allocator diagnostic."""
    path = Path(config_path)
    config = _load_config(path)
    config["_sha256"] = sha256_file(path)
    _verify_source_files(config)
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError(f"E007 allocator diagnostic report already exists: {report_path}")
    if cycles < 2:
        raise ValueError("E007 allocator diagnostic requires at least two repeated lifecycle cycles")
    protected = {
        "staging": _directory_fingerprint(
            Path(config["output_dir"]).with_name(f".{Path(config['output_dir']).name}.inprogress")
        ),
        "source_checkpoint": sha256_file(Path(config["source"]["checkpoint_path"])),
        "recovery_checkpoint": sha256_file(
            Path(config["output_dir"]).with_name(f".{Path(config['output_dir']).name}.inprogress")
            / "checkpoints"
            / "latest.pt"
        ),
    }
    post_long = _diagnostic_case(config, policy=ALLOCATOR_POLICY, repeated_cycles=cycles)
    pre_both = _diagnostic_case(config, policy="pre_medium_and_long", repeated_cycles=cycles)
    common = min(len(post_long["records"]), len(pre_both["records"]))
    equivalent = all(
        post_long["records"][index]["loss"] == pre_both["records"][index]["loss"]
        and post_long["records"][index]["gradient_sha256"] == pre_both["records"][index]["gradient_sha256"]
        for index in range(common)
    )
    protected_after = {
        "staging": _directory_fingerprint(
            Path(config["output_dir"]).with_name(f".{Path(config['output_dir']).name}.inprogress")
        ),
        "source_checkpoint": sha256_file(Path(config["source"]["checkpoint_path"])),
        "recovery_checkpoint": sha256_file(
            Path(config["output_dir"]).with_name(f".{Path(config['output_dir']).name}.inprogress")
            / "checkpoints"
            / "latest.pt"
        ),
    }
    allocated_limit = float(config["memory"]["maximum_cuda_allocated_mib"])
    reserved_limit = float(config["memory"]["maximum_cuda_reserved_mib"])

    def safe(case: dict[str, Any]) -> bool:
        return (
            case["completed_records"] == len(case["padded_length_sequence"])
            and case["process_peak_cuda_allocated_mib"] <= allocated_limit
            and case["process_peak_cuda_reserved_mib"] <= reserved_limit
            and case["parameters_unchanged"]
            and case["optimizer_unchanged"]
            and case["cpu_rng_unchanged"]
            and case["cuda_rng_unchanged"]
            and all(event.get("event_status") == "passed" for event in case["cleanup_events"])
        )

    post_long_safe = safe(post_long)
    pre_both_safe = safe(pre_both)
    selected_policy = ALLOCATOR_POLICY if post_long_safe else "pre_medium_and_long" if pre_both_safe else None
    cleanup_within_allocated = post_long["process_peak_cuda_allocated_mib"] <= allocated_limit
    cleanup_within_reserved = post_long["process_peak_cuda_reserved_mib"] <= reserved_limit
    passed = equivalent and selected_policy is not None and protected == protected_after
    report = {
        "status": "completed_non_authorizing" if passed else "failed",
        "version": "e007_continuation_allocator_diagnostic_v3",
        "configuration_sha256": config["_sha256"],
        "repeated_cycles": cycles,
        "selected_policy": selected_policy,
        "selection_rationale": (
            "least_intrusive_safe_policy"
            if selected_policy == ALLOCATOR_POLICY
            else "preferred_policy_unsafe_conservative_fallback_selected"
            if selected_policy is not None
            else "no_policy_satisfied_memory_and_equivalence_contracts"
        ),
        "post_long_cleanup": post_long,
        "pre_medium_and_long_cleanup": pre_both,
        "post_long_cleanup_safe": post_long_safe,
        "pre_medium_and_long_cleanup_safe": pre_both_safe,
        "post_long_cleanup_reserved_trajectory_mib": _post_cleanup_reservations(post_long["cleanup_events"]),
        "post_long_cleanup_maximum_reserved_mib": max(
            _post_cleanup_reservations(post_long["cleanup_events"]), default=0.0
        ),
        "derived_post_cleanup_reserved_ceiling_mib": post_cleanup_reserved_ceiling_mib(config),
        "fragmentation_statistics": {
            "active_cuda_mib": "torch.cuda.memory_stats active_bytes.all.current",
            "inactive_split_cuda_mib": "torch.cuda.memory_stats inactive_split_bytes.all.current",
            "cuda_segment_count": "torch.cuda.memory_stats segment.all.current",
            "unclassified_reserved_cuda_mib": (
                "reserved minus active minus inactive-split bytes; closest available proxy, "
                "not asserted to be a live tensor leak"
            ),
        },
        "selected_reserved_safety_margin_mib": None
        if selected_policy is None
        else reserved_limit
        - (
            post_long["process_peak_cuda_reserved_mib"]
            if selected_policy == ALLOCATOR_POLICY
            else pre_both["process_peak_cuda_reserved_mib"]
        ),
        "common_record_count": common,
        "loss_and_gradient_equivalence": equivalent,
        "cleanup_within_allocated_limit": cleanup_within_allocated,
        "cleanup_within_reserved_limit": cleanup_within_reserved,
        "protected_inputs_unchanged": protected == protected_after,
        "protected_hashes_before": protected,
        "protected_hashes_after": protected_after,
        "training_performed": False,
        "optimizer_created": True,
        "optimizer_updates": 0,
        **NON_AUTHORIZING,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(report_path, report)
    return report


def run_coordinate_continuation(config_path: str | Path, *, resume_from: str | None = None) -> dict[str, Any]:
    """Run exact-state continuation. This function is intentionally never called by plan-only."""
    path = Path(config_path)
    config = _load_config(path)
    config["_sha256"] = sha256_file(path)
    source_hashes = _verify_source_files(config)
    source_config = _source_scientific_config(config, extended=False)
    prerequisites = verify_pilot_prerequisites(source_config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or (staging.exists() and resume_from is None):
        raise FileExistsError(f"E007 continuation output already exists: {output} or {staging}")
    if resume_from is not None and not staging.is_dir():
        raise FileNotFoundError("E007 continuation resume requires its existing staging directory")
    if resume_from is None:
        staging.mkdir(parents=True)
        (staging / "checkpoints").mkdir()
    heartbeat_path = staging / "heartbeat.json"
    schedule = continuation_schedule(config)
    start_time = time.monotonic()

    def heartbeat(status: str, **values: Any) -> None:
        _atomic_json(heartbeat_path, {"status": status, "updated_utc": _utc_now(), **values, **NON_AUTHORIZING})

    if resume_from is None:
        heartbeat("initializing", global_optimizer_update=1000)
    model = optimizer = scheduler = None
    update = 1000
    samples_processed = int(config["source"]["samples_processed"])
    valid_residues_processed = int(config["source"]["valid_residues_processed"])
    consistent_boundary = False
    last_committed_metric_update = 1000
    last_memory_gate_passed: bool | None = None
    protected_before: dict[str, Any] | None = None
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 continuation requires configured CUDA execution")
        device = torch.device("cuda")
        authorization = _authorize(source_config)
        protected_before = {
            **prerequisites["hashes"],
            "dataset_shards": _canonical_sha(authorization.observed_shard_hashes),
            **source_hashes,
        }
        extended_config = _source_scientific_config(config, extended=True)
        selected = _select_rows(extended_config, authorization)
        source_panel = json.loads((Path(config["source"]["output_dir"]) / "panel_manifest.json").read_text())
        prefix_hashes = verify_stream_prefix(source_panel, selected, source_config)
        train_evaluation = {
            name: [
                next(row for row in selected["train"][name] if row["sample_id"] == sample_id)
                for sample_id in source_panel["train"][name][
                    -int(source_config["train_evaluation_samples_per_stratum"]) :
                ]
            ]
            for name in source_panel["train"]
        }
        panel = {
            "training_prefix_sha256_by_stratum": prefix_hashes,
            "validation": source_panel["validation"],
            "train_evaluation": {name: [row["sample_id"] for row in rows] for name, rows in train_evaluation.items()},
        }
        _atomic_json(staging / "panel_manifest.json", panel)
        torch.manual_seed(int(source_config["seed"]))
        np.random.seed(int(source_config["seed"]))
        random.seed(int(source_config["seed"]))
        torch.cuda.manual_seed_all(int(source_config["seed"]))
        torch.cuda.reset_peak_memory_stats(device)
        with coordinate_model_execution_context(source_config["numerics"], device) as backend:
            model = EquivariantPairCoordinateUNet(**source_config["model"]).to(device)
            if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETER_COUNT:
                raise ValueError("E007 continuation model parameter-count contradiction")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(source_config["optimizer"]["learning_rate"]),
                weight_decay=float(source_config["optimizer"]["weight_decay"]),
                betas=tuple(source_config["optimizer"]["betas"]),
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: _scheduler_factor(step, source_config)
            )
            diffusion = CoordinateVPDiffusion(int(source_config["diffusion_steps"]))
            if resume_from is None:
                state = inspect_checkpoint(config["source"]["checkpoint_path"], config)
                verify_source_checkpoint_protection(state, prerequisites)
            else:
                state = inspect_checkpoint(resume_from, config, continuation=True)
                if state.get("protected_hashes_sha256") != _canonical_sha(protected_before):
                    raise ValueError("E007 continuation recovery protected-input contradiction")
            metric_rows = _read_metric_rows(staging / "metrics.jsonl")
            resume_diagnosis = diagnose_resume_boundary(state, metric_rows)
            if resume_from is not None:
                heartbeat(
                    "resuming_validated_boundary",
                    global_optimizer_update=int(state["optimizer_update"]),
                    missing_metric_updates=resume_diagnosis["missing_metric_updates"],
                )
            restore_full_state(state, model, optimizer, scheduler)
            update = int(state["optimizer_update"])
            samples_processed = int(state["samples_processed"])
            valid_residues_processed = int(state["valid_residues_processed"])
            last_committed_metric_update = int(resume_diagnosis["last_metric_update"])
            last_memory_gate_passed = resume_diagnosis["last_memory_gate_passed"]
            del state
            gc.collect()
            if device.type == "cuda" and resume_from is None:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
            consistent_boundary = True
            train_cursor = {name: 0 for name in selected["train"]}
            for completed in range(1, update + 1):
                name = update_stratum(completed, source_config["length_strata"])
                train_cursor[name] += int(_regime_for_stratum(name, source_config)["physical_batch_size"])
            evaluations: dict[int, Any] = {}
            sampling: dict[int, Any] = {}
            evaluation_path = staging / "evaluations.json"
            sampling_path = staging / "sampling.json"
            if evaluation_path.exists():
                evaluations = {int(k): v for k, v in json.loads(evaluation_path.read_text()).items()}
            else:
                source_evaluations = json.loads((Path(config["source"]["output_dir"]) / "evaluations.json").read_text())
                evaluations = {1000: source_evaluations["1000"]}
            if sampling_path.exists():
                sampling = {int(k): v for k, v in json.loads(sampling_path.read_text()).items()}
            references = _real_reference_distributions(selected["validation"])
            metrics_path = staging / "metrics.jsonl"
            failed_updates = 0
            cleanup_path = staging / "allocator_cleanup.jsonl"
            cleanup_events: list[dict[str, Any]] = _read_metric_rows(cleanup_path)
            process_peak_allocated_mib = 0.0
            process_peak_reserved_mib = 0.0
            active_computation = False

            if resume_from is not None:
                resume_cleanup = allocator_cleanup(
                    device,
                    active_computation=False,
                    padded_length=0,
                    event_type="resume_before_next_update",
                    optimizer_update=update,
                    length_stratum="resume_boundary",
                )
                resume_cleanup = publish_and_enforce_allocator_cleanup(
                    cleanup_path,
                    resume_cleanup,
                    config,
                    prior_post_cleanup_reserved_mib=_post_cleanup_reservations(cleanup_events),
                )
                cleanup_events.append(resume_cleanup)
                if resume_diagnosis["last_memory_gate_passed"] is False:
                    acknowledgement = {
                        "event": "prior_failed_memory_gate_acknowledged",
                        "optimizer_update": update,
                        "next_optimizer_update": update + 1,
                        "checkpoint_sha256": sha256_file(Path(resume_from)),
                        "metric_memory_gate_passed": False,
                        "allocator_policy": ALLOCATOR_POLICY,
                        **NON_AUTHORIZING,
                    }
                    recovery_path = staging / "recovery_events.jsonl"
                    existing_events = _read_metric_rows(recovery_path)
                    if not any(
                        item.get("event") == acknowledgement["event"]
                        and int(item.get("optimizer_update", -1)) == update
                        for item in existing_events
                    ):
                        _append_jsonl_fsync(recovery_path, acknowledgement)

            if resume_from is not None and resume_diagnosis["missing_metric_updates"]:
                missing_update = int(resume_diagnosis["missing_metric_updates"][0])
                name = update_stratum(missing_update, source_config["length_strata"])
                size = int(_regime_for_stratum(name, source_config)["physical_batch_size"])
                start = train_cursor[name] - size
                missing_rows = selected["train"][name][start : start + size]
                previous_metric = metric_rows[-1]
                if samples_processed - int(
                    previous_metric["samples_processed"]
                ) != size or valid_residues_processed - int(previous_metric["valid_residues_processed"]) != sum(
                    int(row["sequence_length"]) for row in missing_rows
                ):
                    raise ValueError("E007 update-1020 checkpoint counters contradict the deterministic stream")
                event = {
                    "event": "metric_publication_gap",
                    "checkpoint_sha256": sha256_file(Path(resume_from)),
                    "optimizer_update": missing_update,
                    "reason": "post_update_memory_gate_preceded_metric_commit",
                    "metric_reconstructed": False,
                    "next_optimizer_update": update + 1,
                    "sample_ids": [str(row["sample_id"]) for row in missing_rows],
                    **NON_AUTHORIZING,
                }
                recovery_path = staging / "recovery_events.jsonl"
                existing_events = _read_metric_rows(recovery_path)
                if not any(item.get("checkpoint_sha256") == event["checkpoint_sha256"] for item in existing_events):
                    _append_jsonl_fsync(recovery_path, event)
                    _append_jsonl_fsync(
                        staging / "trajectory_gaps.jsonl",
                        {
                            "optimizer_update": missing_update,
                            "metric_status": "unavailable_not_fabricated",
                            "resume_checkpoint_sha256": event["checkpoint_sha256"],
                        },
                    )

            def publish_scheduled_artifacts(current: int) -> None:
                if current in schedule["validation"] and current not in evaluations:
                    evaluations[current] = {
                        "train": _evaluate_panel(
                            model, diffusion, train_evaluation, source_config, device, seed_offset=1000000
                        ),
                        "validation": _evaluate_panel(
                            model, diffusion, selected["validation"], source_config, device, seed_offset=2000000
                        ),
                    }
                    _atomic_json(evaluation_path, {str(key): value for key, value in evaluations.items()})
                if current in schedule["sampling"] and current not in sampling:
                    partial_sample_dir = staging / "samples" / f"update-{current:04d}"
                    if partial_sample_dir.exists():
                        shutil.rmtree(partial_sample_dir)
                    sampling[current] = _sample_panel(
                        model, diffusion, extended_config, device, staging, current, references
                    )
                    _atomic_json(sampling_path, {str(key): value for key, value in sampling.items()})
                payload = None
                if current in set(schedule["recovery"]) | set(schedule["immutable"]):
                    payload = continuation_checkpoint_payload(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        update=current,
                        samples_processed=samples_processed,
                        valid_residues_processed=valid_residues_processed,
                        config=config,
                        protected_hashes_sha256=_canonical_sha(protected_before),
                        last_committed_metric_update=last_committed_metric_update,
                        last_memory_gate_passed=last_memory_gate_passed,
                    )
                    if current in schedule["recovery"]:
                        _save_recovery(staging / "checkpoints" / "latest.pt", payload)
                if current in schedule["immutable"]:
                    immutable = staging / "checkpoints" / f"step-{current:05d}.pt"
                    if payload is None:
                        raise RuntimeError("E007 continuation missing scheduled checkpoint payload")
                    if not immutable.exists():
                        _atomic_torch(immutable, payload)
                        _atomic_json(
                            immutable.with_suffix(".json"),
                            {
                                "optimizer_update": current,
                                "sha256": sha256_file(immutable),
                                "validation_coordinate_v_mse": evaluations.get(current, {})
                                .get("validation", {})
                                .get("global", {})
                                .get("coordinate_v_mse"),
                                **NON_AUTHORIZING,
                            },
                        )
                if current not in schedule["validation"] and current not in schedule["sampling"]:
                    return
                candidates = [
                    {
                        "optimizer_update": candidate,
                        "validation_coordinate_v_mse": values["validation"]["global"]["coordinate_v_mse"],
                        "checkpoint_path": str(
                            Path(config["source"]["checkpoint_path"])
                            if candidate == 1000
                            else staging / "checkpoints" / f"step-{candidate:05d}.pt"
                        ),
                    }
                    for candidate, values in evaluations.items()
                    if (candidate == 1000 or (staging / "checkpoints" / f"step-{candidate:05d}.pt").exists())
                ]
                if candidates:
                    best = min(candidates, key=lambda row: row["validation_coordinate_v_mse"])
                    best_source = Path(best["checkpoint_path"])
                    _atomic_copy(best_source, staging / "checkpoints" / "best-denoising.pt")
                    _atomic_json(
                        staging / "checkpoints" / "best-denoising.json",
                        {
                            **best,
                            "checkpoint_sha256": sha256_file(best_source),
                            "alias_sha256": sha256_file(staging / "checkpoints" / "best-denoising.pt"),
                            "criterion": "minimum_validation_coordinate_v_mse",
                            "scalar_sampling_score_used": False,
                            **NON_AUTHORIZING,
                        },
                    )
                if sampling:
                    records = [
                        {
                            "optimizer_update": candidate,
                            **_sampling_objectives(values),
                            "checkpoint_path": str(staging / "checkpoints" / f"step-{candidate:05d}.pt"),
                        }
                        for candidate, values in sampling.items()
                    ]
                    nondominated = nondominated_sampling(records)
                    pareto_dir = staging / "checkpoints" / "best-sampling-pareto"
                    pareto_dir.mkdir(exist_ok=True)
                    retained_names = set()
                    for candidate in nondominated:
                        checkpoint_source = Path(candidate["checkpoint_path"])
                        destination = pareto_dir / checkpoint_source.name
                        _atomic_copy(checkpoint_source, destination)
                        candidate["alias_path"] = str(destination)
                        candidate["alias_sha256"] = sha256_file(destination)
                        retained_names.add(destination.name)
                    for stale in pareto_dir.glob("step-*.pt"):
                        if stale.name not in retained_names:
                            stale.unlink()
                    _atomic_json(
                        staging / "checkpoints" / "best-sampling-pareto.json",
                        {
                            "objectives": list(_sampling_objectives(next(iter(sampling.values())))),
                            "nondominated": nondominated,
                            "scalar_score_used": False,
                            **NON_AUTHORIZING,
                        },
                    )

            if resume_from is not None:
                publish_scheduled_artifacts(update)
            while update < int(config["stop_global_step"]):
                next_update = update + 1
                name = update_stratum(next_update, source_config["length_strata"])
                regime = _regime_for_stratum(name, source_config)
                size = int(regime["physical_batch_size"])
                start = train_cursor[name]
                rows = selected["train"][name][start : start + size]
                if len(rows) != size:
                    raise RuntimeError("E007 continuation deterministic stream exhausted")
                optimizer.zero_grad(set_to_none=True)
                padded_length = (
                    (
                        max(int(row["sequence_length"]) for row in rows)
                        + int(source_config["expected_downsample_factor"])
                        - 1
                    )
                    // int(source_config["expected_downsample_factor"])
                    * int(source_config["expected_downsample_factor"])
                )
                cleanup_event = None
                pre_cleanup_required, post_cleanup_required = allocator_cleanup_actions(ALLOCATOR_POLICY, name)
                if pre_cleanup_required:
                    cleanup_event = allocator_cleanup(
                        device,
                        active_computation=active_computation,
                        padded_length=padded_length,
                        event_type="pre_update",
                        optimizer_update=next_update,
                        length_stratum=name,
                    )
                    cleanup_event = publish_and_enforce_allocator_cleanup(
                        cleanup_path,
                        cleanup_event,
                        config,
                        prior_post_cleanup_reserved_mib=_post_cleanup_reservations(cleanup_events),
                    )
                    cleanup_events.append(cleanup_event)
                torch.cuda.reset_peak_memory_stats(device)
                prepared = prepare_coordinate_batch(
                    rows, EXPECTED_SCALE, int(source_config["expected_downsample_factor"])
                )
                corruption = make_uniform_training_corruption(
                    prepared, diffusion, seed=int(source_config["seed"]) + next_update, device=device
                )
                model.train()
                consistent_boundary = False
                active_computation = True
                prediction = model(
                    corruption["batch"].noisy_coordinates,
                    corruption["batch"].timesteps,
                    prepared["lengths"].to(device),
                    corruption["mask"],
                    prepared["chain_continuity_mask"].to(device),
                )["v_prediction"]
                loss = uniform_coordinate_v_mse(prediction, corruption["batch"].coordinate_v_target, corruption["mask"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("E007 continuation non-finite loss")
                loss.backward()
                active_computation = False
                gradients = _gradient_evidence(model)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(source_config["optimizer"]["gradient_clip_norm"]), error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise FloatingPointError("E007 continuation non-finite model parameter")
                update = next_update
                train_cursor[name] += size
                samples_processed += size
                valid_residues_processed += int(prepared["lengths"].sum())
                consistent_boundary = True
                failed_updates = 0
                phase_memory = _cuda_memory_snapshot(device)
                process_peak_allocated_mib = max(
                    process_peak_allocated_mib, float(phase_memory["phase_peak_cuda_allocated_mib"])
                )
                process_peak_reserved_mib = max(
                    process_peak_reserved_mib, float(phase_memory["phase_peak_cuda_reserved_mib"])
                )
                memory_gate_passed = memory_gate_passes(phase_memory, config)
                metric = {
                    "global_optimizer_update": update,
                    "samples_processed": samples_processed,
                    "valid_residues_processed": valid_residues_processed,
                    "length_stratum": name,
                    "coordinate_v_mse": float(loss.detach().cpu()),
                    "gradient_norm": gradients["global_norm"],
                    "gradient_group_norms": gradients["group_norms"],
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - start_time,
                    "failed_update_count": failed_updates,
                    "padded_length": padded_length,
                    "allocator_policy": ALLOCATOR_POLICY,
                    "pre_update_allocator_cleanup_applied": cleanup_event is not None,
                    "post_update_allocator_cleanup_planned": post_cleanup_required,
                    "allocator_cleanup_event_count": len(cleanup_events),
                    "memory_gate_passed": memory_gate_passed,
                    "cuda_allocated_mib": phase_memory["current_cuda_allocated_mib"],
                    "cuda_reserved_mib": phase_memory["current_cuda_reserved_mib"],
                    "phase_peak_cuda_allocated_mib": phase_memory["phase_peak_cuda_allocated_mib"],
                    "phase_peak_cuda_reserved_mib": phase_memory["phase_peak_cuda_reserved_mib"],
                    "peak_cuda_allocated_mib": process_peak_allocated_mib,
                    "peak_cuda_reserved_mib": process_peak_reserved_mib,
                }
                _append_jsonl_fsync(metrics_path, metric)
                last_committed_metric_update = update
                last_memory_gate_passed = memory_gate_passed
                optimizer.zero_grad(set_to_none=True)
                del prediction, loss, corruption, prepared, gradients, rows
                if post_cleanup_required:
                    post_cleanup_event = allocator_cleanup(
                        device,
                        active_computation=active_computation,
                        padded_length=padded_length,
                        event_type="post_update",
                        optimizer_update=update,
                        length_stratum=name,
                    )
                    post_cleanup_event["captured_phase_peak_cuda_allocated_mib"] = phase_memory[
                        "phase_peak_cuda_allocated_mib"
                    ]
                    post_cleanup_event["captured_phase_peak_cuda_reserved_mib"] = phase_memory[
                        "phase_peak_cuda_reserved_mib"
                    ]
                    post_cleanup_event["captured_memory_gate_passed"] = memory_gate_passed
                    post_cleanup_event = publish_and_enforce_allocator_cleanup(
                        cleanup_path,
                        post_cleanup_event,
                        config,
                        prior_post_cleanup_reserved_mib=_post_cleanup_reservations(cleanup_events),
                    )
                    cleanup_events.append(post_cleanup_event)
                if not memory_gate_passed:
                    raise MemoryError(f"E007 continuation CUDA envelope exceeded: {phase_memory}")
                heartbeat(
                    "running",
                    global_optimizer_update=update,
                    samples_processed=samples_processed,
                    valid_residues_processed=valid_residues_processed,
                    memory=metric,
                )
                publish_scheduled_artifacts(update)
            protected_after = {
                **verify_pilot_prerequisites(source_config)["hashes"],
                "dataset_shards": _canonical_sha(_authorize(source_config).observed_shard_hashes),
                **_verify_source_files(config),
            }
            if protected_after != protected_before:
                raise ValueError("E007 continuation protected inputs changed")
            sampling = published_paths(sampling, staging, output)
            if sampling:
                _atomic_json(sampling_path, {str(key): value for key, value in sampling.items()})
            best_denoising = published_paths(
                json.loads((staging / "checkpoints" / "best-denoising.json").read_text()), staging, output
            )
            best_sampling_pareto = published_paths(
                json.loads((staging / "checkpoints" / "best-sampling-pareto.json").read_text()), staging, output
            )
            _atomic_json(staging / "checkpoints" / "best-denoising.json", best_denoising)
            _atomic_json(staging / "checkpoints" / "best-sampling-pareto.json", best_sampling_pareto)
            report = {
                "status": "completed_mandatory_scientific_review_pause",
                "version": VERSION,
                "start_global_step": 1000,
                "global_optimizer_update": update,
                "additional_successful_updates": update - 1000,
                "samples_processed": samples_processed,
                "valid_residues_processed": valid_residues_processed,
                "parameter_count": EXPECTED_PARAMETER_COUNT,
                "model_parameter_sha256": _parameter_sha256(model),
                "evaluations": evaluations,
                "sampling": sampling,
                "best_denoising": best_denoising,
                "best_sampling_pareto": best_sampling_pareto,
                "protected_inputs_unchanged": True,
                "protected_input_hashes": protected_after,
                "numerical_backend": backend,
                "memory": {
                    **_cuda_memory_snapshot(device),
                    "process_peak_cuda_allocated_mib": process_peak_allocated_mib,
                    "process_peak_cuda_reserved_mib": process_peak_reserved_mib,
                    "allocator_cleanup_event_count": len(cleanup_events),
                },
                "mandatory_scientific_review": True,
                **NON_AUTHORIZING,
            }
        _atomic_json(staging / "report.json", report)
        _atomic_json(
            staging / "protocol.json",
            {
                "status": report["status"],
                "version": VERSION,
                "configuration_sha256": config["_sha256"],
                "source_checkpoint_sha256": config["source"]["checkpoint_sha256"],
                "report_sha256": sha256_file(staging / "report.json"),
                "global_optimizer_update": update,
                "protected_inputs_unchanged": True,
                **NON_AUTHORIZING,
            },
        )
        heartbeat(
            "completed_mandatory_scientific_review_pause",
            global_optimizer_update=update,
            report_sha256=sha256_file(staging / "report.json"),
        )
        staging.replace(output)
        return {"status": report["status"], "output_dir": str(output), **NON_AUTHORIZING}
    except BaseException as error:
        if (
            consistent_boundary
            and protected_before is not None
            and last_committed_metric_update >= update
            and model is not None
            and optimizer is not None
            and scheduler is not None
        ):
            try:
                payload = continuation_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    update=update,
                    samples_processed=samples_processed,
                    valid_residues_processed=valid_residues_processed,
                    config=config,
                    protected_hashes_sha256=_canonical_sha(protected_before),
                    last_committed_metric_update=last_committed_metric_update,
                    last_memory_gate_passed=last_memory_gate_passed,
                )
                _save_recovery(staging / "checkpoints" / "latest.pt", payload)
            except BaseException:
                pass
        status = (
            "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "memory_limit_exceeded"
            if isinstance(error, MemoryError)
            else "failed"
        )
        heartbeat(
            status,
            global_optimizer_update=update,
            samples_processed=samples_processed,
            valid_residues_processed=valid_residues_processed,
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            latest_recovery_checkpoint=str(staging / "checkpoints" / "latest.pt")
            if (staging / "checkpoints" / "latest.pt").exists()
            else None,
        )
        raise
