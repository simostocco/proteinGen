from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_checkpoint_policy import (
    finalize_best_checkpoint,
    publish_best_checkpoint,
    publish_immutable_checkpoint,
    publish_recovery_checkpoint,
    require_storage_preflight,
    storage_preflight,
    verify_recovery_checkpoint,
)
from protein_distance_diffusion.training.rich_codesign_production import verify_stage_a_checkpoint


def _payload(step: int, *, dataset_pass: int = 0) -> dict:
    return {
        "version": "e006_sequence_pretrain_checkpoint_v1",
        "stage": "sequence-pretrain",
        "status": "recovery_only",
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "architecture_version": "e006_rich_geometry_codesign_v1",
        "dataset_identity": "dataset",
        "calibration_sha256": "calibration",
        "production_selection_sha256": "selection",
        "optimizer_step": step,
        "processed_valid_tokens": step * 128,
        "dataset_pass": dataset_pass,
        "microstep": step,
        "data_cursor": step,
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {"state": {}},
        "scheduler": {"last_epoch": step},
        "scaler": {},
        "rng_state": None,
        "sampler_state": {"data_cursor": step},
        "accumulation_state": {"microbatches_accumulated": 0, "at_optimizer_boundary": True},
    }


def test_atomic_rolling_replacement_and_constant_file_count(tmp_path: Path) -> None:
    first = publish_recovery_checkpoint(tmp_path, _payload(250))
    second = publish_recovery_checkpoint(tmp_path, _payload(500))
    assert first["sha256"] != second["sha256"]
    assert verify_recovery_checkpoint(tmp_path)["optimizer_step"] == 500
    assert [path.name for path in tmp_path.glob("latest*.pt")] == ["latest.pt"]
    restored = torch.load(tmp_path / "latest.pt", map_location="cpu", weights_only=False)
    assert restored["optimizer_step"] == 500
    assert restored["sampler_state"]["data_cursor"] == 500
    assert restored["accumulation_state"]["at_optimizer_boundary"] is True


def test_failed_rolling_write_preserves_previous_checkpoint(tmp_path: Path, monkeypatch) -> None:
    publish_recovery_checkpoint(tmp_path, _payload(250))
    before_checkpoint = hashlib.sha256((tmp_path / "latest.pt").read_bytes()).hexdigest()
    before_metadata = (tmp_path / "latest.json").read_bytes()

    def fail(*_args, **_kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="synthetic disk failure"):
        publish_recovery_checkpoint(tmp_path, _payload(500))
    assert hashlib.sha256((tmp_path / "latest.pt").read_bytes()).hexdigest() == before_checkpoint
    assert (tmp_path / "latest.json").read_bytes() == before_metadata


def test_immutable_deduplication_and_best_authorization(tmp_path: Path) -> None:
    payload = _payload(2500, dataset_pass=1)
    immutable = publish_immutable_checkpoint(tmp_path, payload, reasons={"validation"})
    merged = publish_immutable_checkpoint(
        tmp_path,
        payload,
        reasons={"dataset_pass_end", "best_selection"},
        existing_record=immutable,
    )
    assert list(tmp_path.glob("step-*.pt")) == [tmp_path / "step-000002500.pt"]
    assert merged["reasons"] == ["best_selection", "dataset_pass_end", "validation"]
    best = publish_best_checkpoint(
        tmp_path,
        merged,
        payload,
        validation_sequence_cross_entropy=0.75,
    )
    assert best["source_immutable_checkpoint_sha256"] == immutable["sha256"]
    finalized = finalize_best_checkpoint(tmp_path, stage="sequence-pretrain")
    verified = verify_stage_a_checkpoint(
        tmp_path / "best.pt",
        finalized["sha256"],
        dataset_identity="dataset",
        calibration_sha256="calibration",
        production_selection_sha256="selection",
    )
    assert verified["selected_validation_sequence_cross_entropy"] == 0.75


def test_latest_checkpoint_can_never_authorize_stage_b(tmp_path: Path) -> None:
    latest = publish_recovery_checkpoint(tmp_path, _payload(250))
    with pytest.raises(ValueError, match="best.pt"):
        verify_stage_a_checkpoint(
            tmp_path / "latest.pt",
            latest["sha256"],
            dataset_identity="dataset",
            calibration_sha256="calibration",
            production_selection_sha256="selection",
        )


def test_storage_projection_and_insufficient_disk_rejection(tmp_path: Path) -> None:
    regimes = [
        {"maximum_length": 128, "physical_batch_size": 4},
        {"maximum_length": 256, "physical_batch_size": 2},
        {"maximum_length": 384, "physical_batch_size": 1},
        {"maximum_length": 500, "physical_batch_size": 1},
    ]
    report = storage_preflight(
        lengths=[128] * 8 + [256] * 4 + [384, 500],
        regimes=regimes,
        dataset_passes=2,
        validation_frequency=3,
        recovery_frequency=2,
        estimated_checkpoint_bytes=1024,
        output_directory=tmp_path,
        minimum_free_disk_gib=25,
        free_disk_bytes=30 * 1024**3,
    )
    assert report["updates_per_length_bucket_per_pass"] == {"128": 2, "256": 2, "384": 1, "500": 1}
    assert report["optimizer_updates_per_pass"] == 6
    assert report["total_optimizer_updates"] == 12
    assert report["immutable_checkpoint_steps"] == [0, 3, 6, 9, 12]
    assert report["expected_immutable_checkpoint_count"] == 5
    require_storage_preflight(report)
    report["current_free_disk_bytes"] = 25 * 1024**3
    report["current_free_disk_gib"] = 25.0
    report["storage_safe"] = False
    with pytest.raises(OSError, match="minimum free-space reserve"):
        require_storage_preflight(report)


def test_stage_a_and_stage_b_checkpoint_policy_consistency() -> None:
    keys = {
        "recovery_checkpoint_frequency",
        "validation_frequency",
        "immutable_checkpoint_on_validation",
        "immutable_checkpoint_on_pass_end",
        "maintain_best_checkpoint",
        "estimated_checkpoint_bytes",
        "minimum_free_disk_gib",
    }
    stage_a = load_yaml("configs/e006_rich_geometry_sequence_pretrain.yaml")["training"]
    stage_b = load_yaml("configs/e006_rich_geometry_joint_train.yaml")["training"]
    assert {key: stage_a[key] for key in keys} == {key: stage_b[key] for key in keys}
    assert "checkpoint_frequency" not in stage_a
    assert "checkpoint_frequency" not in stage_b
    assert stage_b["minimum_free_disk_gib"] == 25
