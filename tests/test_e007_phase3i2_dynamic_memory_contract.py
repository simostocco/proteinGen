from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from protein_distance_diffusion.training import e007_local_backbone_repair as repair

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "configs/e007_local_backbone_repair_dynamic_stability_preflight_v2.yaml"


def test_production_memory_smoke_startup_resolves_reviewed_model_contract(monkeypatch) -> None:
    config = repair.load_config(PRODUCTION)
    resolved = repair._resolve_reviewed_coordinate_source(config)
    assert resolved["source"]["model"]
    assert resolved["source"]["model"]["base_channels"] == 24
    assert resolved["source"]["model"]["max_length"] == 500
    assert resolved["checkpoint_path"].name == "step-09000.pt"
    assert (
        resolved["source"]["dataset"]["protected_input_relocations"] == config["dataset"]["protected_input_relocations"]
    )

    # Exercise the same run entrypoint and model-construction expression while
    # replacing only expensive panel, checkpoint and CUDA operations.
    calls: dict[str, object] = {}
    monkeypatch.setattr(repair, "validate_dynamic_memory_smoke_contract", lambda _path: {"status": "validated"})
    monkeypatch.setattr(
        repair, "_resolve_reviewed_coordinate_source", lambda _config, **_kw: {**resolved, "checkpoint": {"model": {}}}
    )
    monkeypatch.setattr(repair, "load_coefficient_table", lambda _path: {})
    monkeypatch.setattr(repair, "zero_update_drift_audit", lambda *a, **k: {"pass": True})
    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    monkeypatch.setattr(localization, "verify_prerequisites", lambda *_a, **_k: None)
    monkeypatch.setattr(localization, "select_validation_panel", lambda *_a: ([], None))
    monkeypatch.setattr(
        localization,
        "reconstruct_authoritative_panel",
        lambda *_a: ([{"selection": {"target_length": 500}, "canonical_row": {"sample_id": "500-sample"}}], None),
    )

    # Keep the actual constructor call observable without allocating model weights.
    class Model:
        def __init__(self, **kwargs):
            calls["model_contract"] = kwargs

        def to(self, _device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, _state):
            pass

        def parameters(self):
            return []

    class Optimizer:
        def __init__(self, *_a, **_k):
            pass

        def zero_grad(self, **_k):
            pass

    class Scheduler:
        def __init__(self, *_a, **_k):
            pass

    class Device:
        type = "cuda"

    monkeypatch.setattr(torch, "device", lambda _name: Device())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_a: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_a: 1)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_a: 1)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda *_a: {"active_bytes.all.current": 1})
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_a: type("DeviceProperties", (), {"total_memory": 8151 * 2**20})()
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_a: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_a: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_a: None)
    monkeypatch.setattr(
        repair, "_resolve_reviewed_coordinate_source", lambda _config, **_kw: {**resolved, "checkpoint": {"model": {}}}
    )
    monkeypatch.setattr(
        "protein_distance_diffusion.models.equivariant_pair_coordinate_unet.EquivariantPairCoordinateUNet", Model
    )
    monkeypatch.setattr(
        "protein_distance_diffusion.training.coordinate_diffusion.CoordinateVPDiffusion", lambda *_a: object()
    )
    monkeypatch.setattr(torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", Scheduler)
    monkeypatch.setattr(repair, "coordinate_model_execution_context", None, raising=False)
    # The context manager is imported inside run_dynamic_memory_smoke.
    from contextlib import nullcontext

    import protein_distance_diffusion.models.coordinate_equivariance as equivariance

    monkeypatch.setattr(equivariance, "coordinate_model_execution_context", lambda *_a: nullcontext())
    import resource

    monkeypatch.setattr(resource, "getrusage", lambda *_a: type("Usage", (), {"ru_maxrss": 1})())
    config["dynamic_preflight"]["audit_sample_ids_by_length"]["500"] = "500-sample"
    monkeypatch.setattr(repair, "load_config", lambda _path: config)
    monkeypatch.setattr(
        repair, "publish_dynamic_memory_smoke_result", lambda result, _path: calls.setdefault("published", result)
    )
    repair.run_dynamic_memory_smoke(PRODUCTION, repetitions=3)
    assert calls["model_contract"] == resolved["source"]["model"]
    assert calls["published"]["repetitions"] == 3


def test_reviewed_source_missing_model_block_fails_closed(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "reviewed.yaml"
    source.write_text(
        yaml.safe_dump({"optimizer": {"learning_rate": 1, "weight_decay": 0, "betas": [0.9, 0.99]}, "dataset": {}})
    )
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"checkpoint")
    audit_source = {
        "model_source": {
            "phase3f_config_path": str(source),
            "phase3f_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    }

    class Localization:
        @staticmethod
        def load_config(_path):
            return audit_source

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    monkeypatch.setattr(localization, "load_config", Localization.load_config)
    monkeypatch.setattr(repair, "load_config", lambda _path: {"dataset": {"protected_input_relocations": []}})
    monkeypatch.setattr(repair, "_phase3f_dataset_config", lambda _config: {})
    config = {
        "dataset_source_config": "unused",
        "selected_checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "dataset": {"protected_input_relocations": []},
    }
    with pytest.raises(ValueError, match="missing required keys.*model"):
        repair._resolve_reviewed_coordinate_source(config)


def test_successful_memory_smoke_publishes_to_smoke_path_atomically(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: publication\n")
    schedule = tmp_path / "schedule.json"
    schedule.write_text("{}\n")
    source = tmp_path / "preserved.log"
    source.write_text("preserved smoke evidence\n")
    result = {
        "status": "passed",
        "checkpoint_sha256": "a" * 64,
        "repetitions": 3,
        "optimizer_updates": 0,
        "measurements": [
            {
                "repetition": index,
                "audit_state_restored": True,
                "optimizer_updates": 0,
                "current_cuda_allocated_mib": 93.0,
                "current_cuda_reserved_mib": 96.0,
                "peak_cuda_allocated_mib": 5813.0,
                "peak_cuda_reserved_mib": 6286.0,
                "peak_rss_mib": 2416.0,
            }
            for index in range(1, 4)
        ],
        **repair.NON_AUTHORIZING,
    }
    config = {
        "dynamic_preflight": {"version": "e007_phase3i2_dynamic_stability_preflight_v2"},
        "selected_checkpoint": {"sha256": result["checkpoint_sha256"]},
        "memory": {
            "maximum_rss_mib": 4096,
            "maximum_cuda_allocated_mib": 6144,
            "maximum_cuda_reserved_mib": 7680,
        },
        "local_objective": {"coefficient_table_path": str(schedule)},
    }
    monkeypatch.setattr(repair, "load_config", lambda _path: config)
    parent = tmp_path / "reports/experiments/E007_matrix_sequence_cogeneration"
    parent.mkdir(parents=True)
    published = repair.publish_dynamic_memory_smoke_result(result, config_path, log_path=source)
    output = parent / "local_backbone_repair_dynamic_memory_smoke_v2"
    assert published["record_type"] == "retrospective_preserved_log"
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "protocol.json",
        "artifact_inventory.json",
        "heartbeat.json",
    }
    assert not (parent / ".local_backbone_repair_dynamic_memory_smoke_v2.inprogress").exists()
    assert json.loads((output / "protocol.json").read_text())["report_sha256"] == repair._sha256_file(
        output / "report.json"
    )
    with pytest.raises(FileExistsError):
        repair.publish_dynamic_memory_smoke_result(result, config_path, log_path=source)
