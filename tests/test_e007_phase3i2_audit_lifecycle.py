"""Regression coverage for the production multi-cell zero-update drift audit."""

from __future__ import annotations

import gc
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from protein_distance_diffusion.training import e007_local_backbone_repair as repair


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, noisy, timestep, lengths, mask, continuity):
        return {"v_prediction": noisy * self.weight}


class TinyDiffusion:
    def __init__(self, fail_on: int | None = None) -> None:
        self.calls = 0
        self.fail_on = fail_on

    def make_training_batch(self, clean, mask, *, timesteps, generator):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("cell failure")
        return SimpleNamespace(noisy_coordinates=clean + 0.1, coordinate_v_target=torch.zeros_like(clean))

    def reconstruct_x0(self, noisy, timestep, prediction, mask):
        return prediction


def _setup(monkeypatch):
    loader = importlib.import_module("protein_distance_diffusion.training.e007_coordinate_real_loader_smoke")
    pilot = importlib.import_module("protein_distance_diffusion.training.e007_coordinate_real_pilot")

    monkeypatch.setattr(repair, "verify_protected_evidence", lambda config: {"input": "unchanged"})
    monkeypatch.setattr(
        loader,
        "prepare_coordinate_batch",
        lambda rows, scale, factor: {
            "coordinates": torch.full((1, rows[0]["length"], 3), rows[0]["value"]),
            "residue_mask": torch.ones(1, rows[0]["length"], dtype=torch.bool),
            "chain_continuity_mask": torch.ones(1, rows[0]["length"] - 1, dtype=torch.bool),
            "lengths": torch.tensor([rows[0]["length"]]),
        },
    )
    monkeypatch.setattr(
        pilot, "uniform_coordinate_v_mse", lambda prediction, target, mask: (prediction - target).square().mean()
    )
    monkeypatch.setattr(
        repair,
        "local_backbone_losses",
        lambda predicted, clean, mask, continuity, settings, per_structure: {
            name: (predicted.square().mean() * 0.001).reshape(1) for name in repair.LOCAL_TERMS
        },
    )
    monkeypatch.setattr(repair, "coefficient_tensor", lambda table, timestep, device: torch.ones(1, 6))
    cleanup = []
    monkeypatch.setattr(gc, "collect", lambda: cleanup.append("collected"))
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = {
        "local_objective": {},
        "coordinate_scale_angstrom": 1.0,
        "dataset_source_config": "configs/e007_denoiser_sampler_localization_v1.yaml",
        "dynamic_preflight": {"version": "test"},
    }
    panel = [
        (64, {"sample_id": "a", "length": 64, "value": 1.0}),
        (128, {"sample_id": "b", "length": 128, "value": 2.0}),
        (500, {"sample_id": "c", "length": 500, "value": 3.0}),
        (64, {"sample_id": "a", "length": 64, "value": 1.0}),
    ]
    return model, optimizer, scheduler, config, panel, cleanup


def test_production_audit_reuses_no_cell_tensors_and_restores_state(monkeypatch):
    model, optimizer, scheduler, config, panel, cleanup = _setup(monkeypatch)
    model.train()
    diffusion = TinyDiffusion()
    result = repair.zero_update_drift_audit(
        model,
        optimizer,
        scheduler,
        diffusion,
        panel,
        config,
        {"sha256": "table"},
        torch.device("cpu"),
        update=0,
        data_cursor=7,
        audit_timesteps=[25, 250, 499],
    )
    assert diffusion.calls == 12
    assert len(cleanup) == 12
    assert [(r["length"], r["sample_id"]) for r in result["records"][::3]] == [
        (64, "a"),
        (128, "b"),
        (500, "c"),
        (64, "a"),
    ]
    assert [r["timestep"] for r in result["records"][:3]] == [25, 250, 499]
    assert all(
        set(r) >= {"finite_counts", "individual_ratios", "loss_values", "coefficient_row"} for r in result["records"]
    )
    assert result["state_hashes_before"]["model"] == result["state_hashes_after"]["model"]
    assert all(
        result["state_hashes_before"][key] == result["state_hashes_after"][key] for key in result["state_hashes_after"]
    )
    assert result["pass"] is True
    assert model.training is True
    json.dumps(result, allow_nan=False)

    def has_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(has_tensor(item) for item in value.values())
        if isinstance(value, list):
            return any(has_tensor(item) for item in value)
        return False

    assert not has_tensor(result)


def test_cell_exception_runs_cleanup_and_restores_training_state(monkeypatch):
    model, optimizer, scheduler, config, panel, cleanup = _setup(monkeypatch)
    cuda_cleanup = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: cuda_cleanup.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cuda_cleanup.append("empty_cache"))

    class CpuDeviceWithCudaCleanup(str):
        type = "cuda"

    model.train()
    model.weight.grad = torch.tensor(5.0)
    with pytest.raises(RuntimeError, match="cell failure"):
        repair.zero_update_drift_audit(
            model,
            optimizer,
            scheduler,
            TinyDiffusion(fail_on=2),
            panel,
            config,
            {"sha256": "table"},
            CpuDeviceWithCudaCleanup("cpu"),
            update=0,
            data_cursor=7,
            audit_timesteps=[25, 250, 499],
        )
    assert len(cleanup) == 2
    assert cuda_cleanup == ["synchronize", "empty_cache"] * 2
    assert model.training is True


def test_v4_contract_pins_passed_smoke_and_failed_v3_evidence(tmp_path):
    from pathlib import Path

    import yaml

    path = Path("configs/e007_local_backbone_repair_dynamic_stability_preflight_v4.yaml")
    validated = repair.validate_dynamic_preflight_contract(path)
    assert validated["status"] == "validated_read_only"
    assert validated["staging_output_exists"] is True
    config = yaml.safe_load(path.read_text())
    config["dynamic_preflight"]["lifecycle_evidence"]["failed_v3_log"]["sha256"] = "0" * 64
    corrupted = tmp_path / "corrupted.yaml"
    corrupted.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="v4 failed_v3_log hash mismatch"):
        repair.validate_dynamic_preflight_contract(corrupted)


def test_v5_warning_is_recorded_and_hard_ceiling_fails(monkeypatch):
    model, optimizer, scheduler, config, panel, _ = _setup(monkeypatch)
    config["dynamic_preflight"] = {
        "version": "e007_phase3i2_dynamic_stability_preflight_v5",
        "combined_auxiliary_warning_threshold": 0.20,
        "maximum_combined_auxiliary_v_ratio": 0.25,
    }
    ratio = [0.20]

    def cell(*args, **kwargs):
        return {
            "update": 5,
            "sample_id": "2e19_A",
            "length": 64,
            "timestep": 425,
            "finite_counts": {"v": {"all_finite": True}},
            "individual_ratios": {name: 0.1 for name in repair.LOCAL_TERMS},
            "combined_auxiliary_ratio": ratio[0],
            "total_v_ratio": 0.9288470670534813,
            "loss_values": {"v": 0.1, "raw": {"x": 0.1}, "weighted": {"x": 0.1}},
        }

    monkeypatch.setattr(repair, "_zero_update_drift_cell", cell)
    for value, expected_warnings, expected_pass in (
        (0.20, 0, True),
        (0.23559981839253288, 1, True),
        (0.25, 1, True),
        (0.2500001, 1, False),
    ):
        ratio[0] = value
        result = repair.zero_update_drift_audit(
            model,
            optimizer,
            scheduler,
            TinyDiffusion(),
            panel[:1],
            config,
            {"sha256": "table"},
            torch.device("cpu"),
            update=5,
            data_cursor=5,
            audit_timesteps=[425],
        )
        assert result["warning_count"] == expected_warnings
        assert result["pass"] is expected_pass
        if expected_warnings:
            assert result["warnings"][0]["sample_id"] == "2e19_A"
            assert result["warnings"][0]["classification"] == (
                "accepted_dynamic_warning_below_hard_ceiling" if expected_pass else "hard_ceiling_exceeded"
            )


def test_v5_contract_is_read_only_and_pins_failed_v4_evidence(tmp_path):
    from pathlib import Path

    import yaml

    path = Path("configs/e007_local_backbone_repair_dynamic_stability_preflight_v5.yaml")
    validated = repair.validate_dynamic_preflight_contract(path)
    assert validated["status"] == "validated_read_only"
    assert validated["staging_output_exists"] is True
    assert validated["maximum_combined_auxiliary_v_ratio"] == 0.25
    config = yaml.safe_load(path.read_text())
    config["dynamic_preflight"]["reviewed_v4_warning"]["sha256"] = "0" * 64
    corrupted = tmp_path / "corrupted.yaml"
    corrupted.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="v5 reviewed warning pin mismatch"):
        repair.validate_dynamic_preflight_contract(corrupted)
