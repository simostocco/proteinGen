"""Bounded checkpoint publication and storage planning for E006."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_POLICY_VERSION = "e006_production_checkpoint_policy_v1"
GIB = 1024**3


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def publish_recovery_checkpoint(
    checkpoint_directory: str | Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Transactionally replace the sole rolling checkpoint and its metadata."""
    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "latest.pt"
    metadata = directory / "latest.json"
    temporary_checkpoint = directory / f".latest.{os.getpid()}.tmp"
    temporary_metadata = directory / f".latest-meta.{os.getpid()}.tmp"
    backup_checkpoint = directory / f".latest.{os.getpid()}.backup"
    backup_metadata = directory / f".latest-meta.{os.getpid()}.backup"
    torch.save(payload, temporary_checkpoint)
    record = {
        "version": CHECKPOINT_POLICY_VERSION,
        "role": "rolling_recovery",
        "path": str(checkpoint),
        "sha256": _sha256(temporary_checkpoint),
        "optimizer_step": int(payload["optimizer_step"]),
        "processed_valid_tokens": int(payload.get("processed_valid_tokens", 0)),
        "dataset_pass": int(payload["dataset_pass"]),
        "authorizes_training": False,
    }
    temporary_metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    try:
        if checkpoint.exists():
            checkpoint.replace(backup_checkpoint)
        if metadata.exists():
            metadata.replace(backup_metadata)
        temporary_checkpoint.replace(checkpoint)
        temporary_metadata.replace(metadata)
    except BaseException:
        if checkpoint.exists():
            checkpoint.unlink()
        if metadata.exists():
            metadata.unlink()
        if backup_checkpoint.exists():
            backup_checkpoint.replace(checkpoint)
        if backup_metadata.exists():
            backup_metadata.replace(metadata)
        raise
    finally:
        for path in (temporary_checkpoint, temporary_metadata, backup_checkpoint, backup_metadata):
            if path.exists():
                path.unlink()
    return record


def verify_recovery_checkpoint(checkpoint_directory: str | Path) -> dict[str, Any]:
    directory = Path(checkpoint_directory)
    checkpoint = directory / "latest.pt"
    metadata = directory / "latest.json"
    if not checkpoint.is_file() or not metadata.is_file():
        raise ValueError("E006 rolling recovery checkpoint or metadata is missing")
    record = json.loads(metadata.read_text())
    if (
        record.get("version") != CHECKPOINT_POLICY_VERSION
        or record.get("role") != "rolling_recovery"
        or record.get("authorizes_training") is not False
        or record.get("sha256") != _sha256(checkpoint)
    ):
        raise ValueError("E006 rolling recovery checkpoint metadata contradiction")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("status") != "recovery_only"
        or payload.get("authorizes_training") is not False
        or payload.get("authorizes_joint_training") is not False
        or payload.get("accumulation_state", {}).get("at_optimizer_boundary") is not True
        or payload.get("accumulation_state", {}).get("microbatches_accumulated") != 0
        or payload.get("sampler_state", {}).get("data_cursor") != payload.get("data_cursor")
    ):
        raise ValueError("E006 rolling recovery payload is not an exact optimizer-boundary state")
    return record


def publish_immutable_checkpoint(
    checkpoint_directory: str | Path,
    payload: dict[str, Any],
    *,
    reasons: set[str],
    existing_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(checkpoint_directory)
    step = int(payload["optimizer_step"])
    path = directory / f"step-{step:09d}.pt"
    if path.exists():
        if existing_record is None or existing_record.get("sha256") != _sha256(path):
            raise ValueError(f"E006 immutable checkpoint contradiction at step {step}")
        return {**existing_record, "reasons": sorted(set(existing_record.get("reasons", [])) | reasons)}
    from protein_distance_diffusion.training.checkpointing import save_checkpoint

    save_checkpoint(path, payload)
    return {
        "role": "immutable_scientific",
        "optimizer_step": step,
        "processed_valid_tokens": int(payload.get("processed_valid_tokens", 0)),
        "dataset_pass": int(payload["dataset_pass"]),
        "path": str(path),
        "sha256": _sha256(path),
        "reasons": sorted(reasons),
    }


def publish_best_checkpoint(
    checkpoint_directory: str | Path,
    immutable_record: dict[str, Any],
    payload: dict[str, Any],
    *,
    validation_sequence_cross_entropy: float,
) -> dict[str, Any]:
    source = Path(immutable_record["path"])
    if _sha256(source) != immutable_record["sha256"]:
        raise ValueError("E006 best-checkpoint immutable source hash contradiction")
    selected = {
        **payload,
        "checkpoint_role": "validation_selected_best",
        "source_immutable_checkpoint": str(source),
        "source_immutable_checkpoint_sha256": immutable_record["sha256"],
        "selected_validation_sequence_cross_entropy": float(validation_sequence_cross_entropy),
        "authorizes_joint_training": False,
        "authorizes_training": False,
        "authorizes_definitive_evaluation": False,
    }
    from protein_distance_diffusion.training.checkpointing import save_checkpoint

    path = Path(checkpoint_directory) / "best.pt"
    save_checkpoint(path, selected)
    record = {
        "version": CHECKPOINT_POLICY_VERSION,
        "role": "validation_selected_best",
        "path": str(path),
        "sha256": _sha256(path),
        "source_immutable_checkpoint": str(source),
        "source_immutable_checkpoint_sha256": immutable_record["sha256"],
        "optimizer_step": int(payload["optimizer_step"]),
        "processed_valid_tokens": int(payload.get("processed_valid_tokens", 0)),
        "dataset_pass": int(payload["dataset_pass"]),
        "validation_sequence_cross_entropy": float(validation_sequence_cross_entropy),
    }
    _atomic_json(Path(checkpoint_directory) / "best.json", record)
    return record


def finalize_best_checkpoint(
    checkpoint_directory: str | Path,
    *,
    stage: str,
    authorize: bool = True,
) -> dict[str, Any]:
    directory = Path(checkpoint_directory)
    path = directory / "best.pt"
    metadata_path = directory / "best.json"
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("E006 validation-selected best checkpoint is missing")
    record = json.loads(metadata_path.read_text())
    source = Path(record["source_immutable_checkpoint"])
    if record.get("source_immutable_checkpoint_sha256") != _sha256(source):
        raise ValueError("E006 best checkpoint source is not independently verifiable")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("source_immutable_checkpoint_sha256") != record["source_immutable_checkpoint_sha256"]:
        raise ValueError("E006 best checkpoint provenance contradiction")
    payload.update(
        status="completed",
        checkpoint_role="validation_selected_best",
        independently_verified_best=True,
        authorizes_joint_training=authorize and stage == "sequence-pretrain",
        authorizes_training=authorize and stage == "sequence-pretrain",
        authorizes_definitive_evaluation=authorize and stage == "joint-train",
    )
    from protein_distance_diffusion.training.checkpointing import save_checkpoint

    save_checkpoint(path, payload)
    record.update(
        sha256=_sha256(path),
        role="validation_selected_best",
        status="completed",
        independently_verified=True,
        authorizes_joint_training=authorize and stage == "sequence-pretrain",
        authorizes_training=authorize and stage == "sequence-pretrain",
        authorizes_definitive_evaluation=authorize and stage == "joint-train",
    )
    _atomic_json(metadata_path, record)
    return record


def length_bucket_counts(lengths: list[int] | tuple[int, ...]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for length in lengths:
        boundary = next((value for value in (128, 256, 384, 500) if length <= value), None)
        if boundary is None:
            raise ValueError(f"E006 length exceeds production regimes: {length}")
        counts[boundary] += 1
    return {boundary: counts[boundary] for boundary in (128, 256, 384, 500)}


def storage_preflight(
    *,
    lengths: list[int] | tuple[int, ...],
    regimes: list[dict[str, Any]],
    dataset_passes: int,
    validation_frequency: int,
    recovery_frequency: int,
    estimated_checkpoint_bytes: int,
    output_directory: str | Path,
    minimum_free_disk_gib: float,
    review_pause_steps: Sequence[int] = (),
    maintain_contextual_best: bool = False,
    free_disk_bytes: int | None = None,
) -> dict[str, Any]:
    counts = length_bucket_counts(lengths)
    ordered_regimes = sorted(regimes, key=lambda item: int(item["maximum_length"]))

    def regime_for(length: int) -> dict[str, Any]:
        try:
            return next(item for item in ordered_regimes if length <= int(item["maximum_length"]))
        except StopIteration as error:
            raise ValueError(f"No E006 checkpoint-plan batch regime covers length {length}") from error

    updates_per_bucket = {
        str(length): math.ceil(count / int(regime_for(length)["physical_batch_size"]))
        for length, count in counts.items()
    }
    updates_per_pass = sum(updates_per_bucket.values())
    total_updates = updates_per_pass * dataset_passes
    validation_steps = set(range(validation_frequency, total_updates + 1, validation_frequency))
    pass_end_steps = {updates_per_pass * index for index in range(1, dataset_passes + 1)}
    review_steps = {int(step) for step in review_pause_steps if 0 < int(step) <= total_updates}
    immutable_steps = {0, total_updates, *validation_steps, *pass_end_steps, *review_steps}
    immutable_count = len(immutable_steps)
    persistent_checkpoint_count = immutable_count + 2 + int(maintain_contextual_best)
    projected_bytes = persistent_checkpoint_count * estimated_checkpoint_bytes
    output = Path(output_directory)
    probe = output
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    available = int(free_disk_bytes if free_disk_bytes is not None else shutil.disk_usage(probe).free)
    reserve_bytes = int(minimum_free_disk_gib * GIB)
    safe = available - projected_bytes >= reserve_bytes
    return {
        "version": CHECKPOINT_POLICY_VERSION,
        "sample_count_per_pass": len(lengths),
        "updates_per_length_bucket_per_pass": updates_per_bucket,
        "optimizer_updates_per_pass": updates_per_pass,
        "total_optimizer_updates": total_updates,
        "dataset_passes": dataset_passes,
        "rolling_recovery_checkpoint_frequency": recovery_frequency,
        "scheduled_validation_steps": sorted(validation_steps),
        "dataset_pass_end_steps": sorted(pass_end_steps),
        "scientific_review_pause_steps": sorted(review_steps),
        "immutable_checkpoint_steps": sorted(immutable_steps),
        "expected_immutable_checkpoint_count": immutable_count,
        "expected_persistent_checkpoint_file_count": persistent_checkpoint_count,
        "maintains_contextual_best_checkpoint": maintain_contextual_best,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "projected_checkpoint_storage_bytes": projected_bytes,
        "projected_checkpoint_storage_gib": projected_bytes / GIB,
        "current_free_disk_bytes": available,
        "current_free_disk_gib": available / GIB,
        "minimum_free_disk_reserve_gib": minimum_free_disk_gib,
        "projected_remaining_disk_gib": (available - projected_bytes) / GIB,
        "storage_safe": safe,
    }


def require_storage_preflight(preflight: dict[str, Any]) -> None:
    if preflight.get("storage_safe") is not True:
        raise OSError(
            "E006 projected checkpoint storage violates the minimum free-space reserve: "
            f"projected={preflight['projected_checkpoint_storage_gib']:.3f} GiB, "
            f"free={preflight['current_free_disk_gib']:.3f} GiB, "
            f"reserve={preflight['minimum_free_disk_reserve_gib']:.3f} GiB"
        )
