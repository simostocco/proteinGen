from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.codesign import _configuration_sha256
from protein_distance_diffusion.training.codesign_large import (
    LARGE_CURRICULUM,
    LARGE_PARAMETER_COUNTS,
    LARGE_SEQUENCE_SHAPE,
    MEMORY_TELEMETRY_KEYS,
    _publish_launch_gate_evaluation,
    checkpoint_kind,
    enforce_memory_limits,
    large_parameter_counts,
    run_large_training,
    validate_large_config,
    validate_memory_telemetry,
    validate_resume_checkpoint,
    validation_steps,
    verify_launch_gate_report,
    warmup_cosine_multiplier,
)
from protein_distance_diffusion.training.codesign_pilot import DEFAULT_PAIR_BUDGET, _model


def _config() -> dict:
    return load_yaml("configs/e005_sequence_geometry_codesign_large.yaml")


def _passing_gate(config: dict, dataset_hash: str) -> dict:
    return {
        "status": "passed",
        "device": "cuda",
        "architecture_version": "e005_large_sequence_geometry_codesign_v1",
        "config_sha256": _configuration_sha256(config),
        "dataset_sha256": dataset_hash,
        "requested_length": 500,
        "actual_length": 500,
        "optimizer_step_completed": True,
        "dataset_inputs_unchanged": True,
        "dataset_before_sha256": dataset_hash,
        "dataset_after_sha256": dataset_hash,
        "losses": {"total": 1.0, "sequence": 0.5, "geometry": 0.5, "consistency": 0.0},
        "gradient_checks": {"norms": {"sequence_branch": 1.0}, "all_active_parameter_groups_nonzero": True},
        "gate_checks": {"statistics": {}, "valid_nonsaturated": True},
        "memory": {
            "peak_cuda_allocated_mib": 7000.0,
            "peak_cuda_reserved_mib": 8100.0,
            "peak_rss_mib": 6000.0,
            "current_rss_mib": 5900.0,
            "cuda_allocated_mib": 6800.0,
            "cuda_reserved_mib": 8000.0,
        },
    }


def test_large_dimensions_and_exact_parameter_accounting() -> None:
    config = _config()
    model = _model(config)
    shape = (
        config["model"]["sequence_layers"],
        config["model"]["sequence_hidden_dim"],
        config["model"]["sequence_heads"],
        config["model"]["sequence_feedforward_dim"],
    )

    assert shape == LARGE_SEQUENCE_SHAPE
    assert model.sequence_encoder.layers[0].norm_first is True
    assert large_parameter_counts(model) == LARGE_PARAMETER_COUNTS


def test_definitive_schedule_and_accumulation_contracts() -> None:
    config = _config()
    validate_large_config(config)
    stages = tuple((item["maximum_length"], item["optimizer_updates"]) for item in config["production"]["curriculum"])
    budget = tuple(
        (item["maximum_padded_length"], item["microbatches"])
        for item in config["production"]["pair_budget_accumulation"]
    )

    assert stages == LARGE_CURRICULUM
    assert sum(value for _, value in stages) == 35_000
    assert budget == DEFAULT_PAIR_BUDGET
    assert validation_steps((14_000, 14_000, 7_000), 2500) == (
        0,
        2500,
        5000,
        7500,
        10000,
        12500,
        14000,
        15000,
        17500,
        20000,
        22500,
        25000,
        27500,
        28000,
        30000,
        32500,
        35000,
    )


def test_warmup_cosine_and_checkpoint_cadence() -> None:
    assert warmup_cosine_multiplier(0, total_updates=35_000, warmup_updates=1000) == pytest.approx(0.001)
    assert warmup_cosine_multiplier(999, total_updates=35_000, warmup_updates=1000) == pytest.approx(1.0)
    assert warmup_cosine_multiplier(35_000, total_updates=35_000, warmup_updates=1000) == pytest.approx(0.0)
    assert checkpoint_kind(250, (14_000, 14_000, 7_000), 250) == (True, False)
    assert checkpoint_kind(14_000, (14_000, 14_000, 7_000), 250) == (True, True)
    assert checkpoint_kind(35_000, (14_000, 14_000, 7_000), 250) == (True, True)


def test_launch_gate_rejects_wrong_identity_and_memory(tmp_path: Path) -> None:
    config = _config()
    report = _passing_gate(config, "dataset-hash")
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report))
    assert verify_launch_gate_report(path, config=config, dataset_sha256="dataset-hash") == report

    report["dataset_sha256"] = "wrong"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="dataset_sha256"):
        verify_launch_gate_report(path, config=config, dataset_sha256="dataset-hash")

    report = _passing_gate(config, "dataset-hash")
    report["memory"]["peak_cuda_allocated_mib"] = 7168
    path.write_text(json.dumps(report))
    with pytest.raises(MemoryError, match="must be below"):
        verify_launch_gate_report(path, config=config, dataset_sha256="dataset-hash")


@pytest.mark.parametrize("missing", ["peak_cuda_allocated_mib", "peak_cuda_reserved_mib"])
def test_cuda_memory_telemetry_requires_every_canonical_peak(missing: str) -> None:
    memory = dict(_passing_gate(_config(), "dataset")["memory"])
    memory.pop(missing)
    with pytest.raises(ValueError, match=missing):
        validate_memory_telemetry(memory, cuda=True, limits={}, strict=True)


def test_memory_telemetry_rejects_nonfinite_and_strict_thresholds() -> None:
    memory = dict(_passing_gate(_config(), "dataset")["memory"])
    assert set(memory) == set(MEMORY_TELEMETRY_KEYS)
    memory["peak_cuda_allocated_mib"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_memory_telemetry(memory, cuda=True, limits={}, strict=True)

    memory = dict(_passing_gate(_config(), "dataset")["memory"])
    memory["peak_cuda_allocated_mib"] = 7168.0
    with pytest.raises(MemoryError, match="must be below"):
        validate_memory_telemetry(
            memory,
            cuda=True,
            limits={"peak_cuda_allocated_mib": 7168.0},
            strict=True,
        )
    memory["peak_cuda_allocated_mib"] = 7168.01
    with pytest.raises(MemoryError, match="must be below"):
        validate_memory_telemetry(
            memory,
            cuda=True,
            limits={"peak_cuda_allocated_mib": 7168.0},
            strict=True,
        )


def test_cpu_memory_telemetry_explicitly_allows_null_cuda_values() -> None:
    memory = {
        "peak_rss_mib": 900.0,
        "peak_cuda_allocated_mib": None,
        "peak_cuda_reserved_mib": None,
        "current_rss_mib": 850.0,
        "cuda_allocated_mib": None,
        "cuda_reserved_mib": None,
    }
    assert (
        validate_memory_telemetry(
            memory,
            cuda=False,
            limits={"peak_rss_mib": 6144.0},
            strict=True,
        )
        == memory
    )


def test_failed_post_workload_criteria_publish_complete_atomic_report(tmp_path: Path) -> None:
    config = _config()
    report = _passing_gate(config, "dataset-hash")
    report["memory"].pop("peak_cuda_reserved_mib")
    destination = tmp_path / "gate.json"

    with pytest.raises(ValueError, match="peak_cuda_reserved_mib"):
        _publish_launch_gate_evaluation(destination, report, config=config, cuda=True)

    published = json.loads(destination.read_text())
    assert published["status"] == "failed"
    assert "peak_cuda_reserved_mib" in published["failure_reason"]
    assert published["device"] == "cuda"
    assert published["losses"]
    assert published["gradient_checks"]["all_active_parameter_groups_nonzero"] is True
    assert published["gate_checks"]["valid_nonsaturated"] is True
    assert published["dataset_before_sha256"] == published["dataset_after_sha256"]


@pytest.mark.parametrize("status", ["failed", "incomplete", "running"])
def test_nonpassing_reports_never_authorize_training(tmp_path: Path, status: str) -> None:
    config = _config()
    report = _passing_gate(config, "dataset-hash")
    report["status"] = status
    path = tmp_path / f"{status}.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="status"):
        verify_launch_gate_report(path, config=config, dataset_sha256="dataset-hash")


def test_training_does_not_publish_state_before_launch_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["device"] = "cpu"
    report = _passing_gate(config, "dataset-hash")
    report["status"] = "failed"
    gate_path = tmp_path / "failed-gate.json"
    gate_path.write_text(json.dumps(report))
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        "protein_distance_diffusion.training.codesign_large._dataset_identity",
        lambda config, synthetic: {"kind": "synthetic", "sha256": "dataset-hash"},
    )

    with pytest.raises(ValueError, match="status"):
        run_large_training(
            config,
            launch_gate_report=gate_path,
            output_dir=output,
            synthetic=True,
        )

    assert not output.exists()


def test_config_rejects_architecture_and_validation_drift() -> None:
    config = _config()
    config["model"]["sequence_layers"] = 7
    with pytest.raises(ValueError, match="sequence shape"):
        validate_large_config(config)


def test_resume_contract_and_memory_failures() -> None:
    saved = {
        "version": "e005_large_checkpoint_v1",
        "architecture_version": "e005_large_sequence_geometry_codesign_v1",
        "config_sha256": "config",
        "dataset_sha256": "dataset",
        "plan_sha256": "plan",
        "sampler_cursor": 17,
        "microstep": 3,
        "accumulated_pair_token_count": 4096,
        "rng_state": {"torch": torch.get_rng_state()},
    }
    validate_resume_checkpoint(saved, config_hash="config", dataset_hash="dataset", plan_hash="plan")
    saved["dataset_sha256"] = "changed"
    with pytest.raises(ValueError, match="dataset_sha256"):
        validate_resume_checkpoint(saved, config_hash="config", dataset_hash="dataset", plan_hash="plan")

    memory = {
        "current_rss_mib": 6144.1,
        "peak_rss_mib": 6144.1,
        "peak_cuda_allocated_mib": None,
        "peak_cuda_reserved_mib": None,
        "cuda_allocated_mib": 0.0,
        "cuda_reserved_mib": 0.0,
    }
    with pytest.raises(MemoryError, match="current_rss_mib"):
        enforce_memory_limits(memory, max_rss_mib=6144, max_cuda_memory_mib=8192, cuda=False)

    memory.update(
        current_rss_mib=1000.0,
        peak_cuda_allocated_mib=7000.0,
        peak_cuda_reserved_mib=8192.1,
        cuda_allocated_mib=7000.0,
        cuda_reserved_mib=8192.1,
    )
    with pytest.raises(MemoryError, match="cuda_reserved_mib"):
        enforce_memory_limits(memory, max_rss_mib=6144, max_cuda_memory_mib=8192, cuda=True)

    config = _config()
    config["production"]["validation_frequency"] = 2501
    with pytest.raises(ValueError, match="production settings"):
        validate_large_config(config)


def test_all_large_parameters_are_trainable_and_finite() -> None:
    model = _model(_config())
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
