from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.training.e007_local_backbone_repair import (
    _atomic_torch,
    _bounded_clash_pairs,
    _calibration_ready,
    _reconcile_metrics_to_checkpoint,
    _recovery_payload,
    _restore_rng_state,
    _valid_offset_mask,
    _validate_recovery_state,
    apply_local_guidance,
    gradient_profile,
    load_config,
    local_backbone_losses,
    local_guidance_energy,
    paired_bootstrap_improvement,
    paired_update_identity,
    plan_local_backbone_repair,
    run_gradient_calibration,
    run_local_backbone_pilot,
    run_validate_calibration_panel,
    weighted_local_objective,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e007_local_backbone_repair_v1.yaml"


def _settings() -> dict[str, float | int]:
    return {
        "distance_normalizer_angstrom": 3.8,
        "discontinuity_threshold_angstrom": 4.5,
        "clash_threshold_angstrom": 3.0,
        "clash_minimum_sequence_separation": 3,
        "maximum_clash_pairs_per_structure": 32,
        "smooth_tail_beta": 4.0,
    }


def _chain(batch: int = 2, length: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinate = torch.zeros((batch, length, 3), dtype=torch.float64)
    coordinate[..., 0] = torch.arange(length, dtype=torch.float64) * (3.8 / 12.22820347644835)
    coordinate[..., 1] = 0.05 * torch.sin(torch.arange(length, dtype=torch.float64))
    mask = torch.ones((batch, length), dtype=torch.bool)
    continuity = torch.ones((batch, length - 1), dtype=torch.bool)
    return coordinate, mask, continuity


def _rotation() -> torch.Tensor:
    return torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def test_local_losses_are_rigid_transform_invariant_and_report_denominators() -> None:
    native, mask, continuity = _chain()
    predicted = native.clone()
    predicted[:, 3, 1] += 0.1
    baseline = local_backbone_losses(predicted, native, mask, continuity, _settings())
    rotation = _rotation()
    translation = torch.tensor([2.0, -3.0, 4.0], dtype=torch.float64)
    transformed = local_backbone_losses(
        predicted @ rotation.T + translation,
        native @ rotation.T + translation,
        mask,
        continuity,
        _settings(),
    )
    for name in ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "discontinuity", "clash"):
        torch.testing.assert_close(baseline[name], transformed[name], atol=1e-12, rtol=1e-10)
        assert torch.isfinite(baseline[name])
    assert set(baseline["denominators"]) == {
        "adjacent",
        "i_plus_2",
        "i_plus_3",
        "bond_angle_cosine",
        "discontinuity",
        "clash",
    }


def test_masks_padding_and_chain_breaks_are_excluded() -> None:
    native, mask, continuity = _chain(batch=1)
    mask[:, 6:] = False
    continuity[:, 3] = False
    changed = native.clone()
    changed[:, 4:] += 1000
    changed[:, 6:] = torch.nan
    losses = local_backbone_losses(changed, native, mask, continuity, _settings())
    assert float(losses["adjacent"]) == pytest.approx(0.0, abs=1e-20)
    assert float(losses["i_plus_2"]) == pytest.approx(0.0, abs=1e-20)
    assert float(losses["i_plus_3"]) == pytest.approx(0.0, abs=1e-20)
    assert torch.isfinite(losses["clash"])
    offset_two = _valid_offset_mask(mask, continuity, 2)
    assert not bool(offset_two[0, 2]) and not bool(offset_two[0, 3])


def test_empty_masks_have_finite_zero_losses_and_gradients() -> None:
    predicted = torch.zeros((1, 4, 3), requires_grad=True)
    native = torch.zeros_like(predicted)
    mask = torch.zeros((1, 4), dtype=torch.bool)
    continuity = torch.zeros((1, 3), dtype=torch.bool)
    losses = local_backbone_losses(predicted, native, mask, continuity, _settings())
    total, weighted = weighted_local_objective(losses, {name: 1.0 for name in weighted_terms()})
    total.backward()
    assert float(total.detach()) == 0.0
    assert all(float(value.detach()) == 0.0 for value in weighted.values())
    assert predicted.grad is not None and bool(torch.isfinite(predicted.grad).all())


def weighted_terms() -> tuple[str, ...]:
    return ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "discontinuity", "clash")


def test_clash_pair_enumeration_is_deterministic_and_bounded() -> None:
    pairs = _bounded_clash_pairs(500, 3, 4096)
    assert len(pairs) == 4096
    assert pairs == _bounded_clash_pairs(500, 3, 4096)
    assert all(right - left >= 3 for left, right in pairs)


def test_reference_free_guidance_is_equivariant_bounded_and_zeroes_padding() -> None:
    coordinates, mask, continuity = _chain(batch=1)
    mask[:, -2:] = False
    coordinates[:, -2:] = 9
    settings = {
        "coordinate_scale_angstrom": 12.22820347644835,
        "correction_steps": 1,
        "maximum_displacement_angstrom": 0.25,
        "adjacent_target_angstrom": 3.8,
        "i_plus_2_target_angstrom": 6.2,
        "i_plus_3_target_angstrom": 8.0,
        "bond_angle_degrees": 110.0,
        "discontinuity_threshold_angstrom": 4.5,
        "clash_threshold_angstrom": 3.0,
        "maximum_clash_pairs_per_structure": 32,
    }
    corrected, diagnostics = apply_local_guidance(coordinates, mask, continuity, settings, strength=0.0025)
    rotation = _rotation()
    transformed = coordinates @ rotation.T + torch.tensor([3.0, 2.0, -1.0], dtype=torch.float64)
    transformed[:, -2:] = 0
    corrected_transformed, _ = apply_local_guidance(transformed, mask, continuity, settings, strength=0.0025)
    expected = corrected @ rotation.T
    expected = expected - (expected * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
    expected = expected * mask[..., None]
    torch.testing.assert_close(corrected_transformed, expected, atol=1e-9, rtol=1e-8)
    assert torch.count_nonzero(corrected[:, -2:]) == 0
    assert diagnostics["maximum_normalized_displacement"] <= 0.25 / 12.22820347644835 + 1e-12
    assert local_guidance_energy(coordinates, mask, continuity, settings).ndim == 0


def test_gradient_profile_reports_each_term_without_mutating_gradients() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    v_loss = parameter.square().sum()
    local = {name: ((index + 1) * parameter).square().mean() for index, name in enumerate(weighted_terms())}
    profile = gradient_profile(v_loss, local, [parameter])
    assert set(profile["terms"]) == set(weighted_terms())
    assert parameter.grad is None
    assert profile["v_finite_gradient_coverage"] == 1.0
    assert all(record["finite_gradient_coverage"] == 1.0 for record in profile["terms"].values())


def test_paired_identity_is_arm_independent_and_order_sensitive() -> None:
    first = paired_update_identity(3914001, 12, ["a", "b"])
    assert first == paired_update_identity(3914001, 12, ["a", "b"])
    assert first["sha256"] != paired_update_identity(3914001, 12, ["b", "a"])["sha256"]


def test_plan_is_side_effect_free_and_blocks_uncalibrated_pilot(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "final")
    config["calibration_output_dir"] = str(tmp_path / "calibration")
    local = tmp_path / "config.yaml"
    local.write_text(yaml.safe_dump(config, sort_keys=False))
    plan = plan_local_backbone_repair(local)
    assert plan["pilot_startup_ready"] is False
    assert plan["model_created"] is False
    assert plan["dataset_scanned"] is False
    assert not (tmp_path / "final").exists()
    assert not (tmp_path / "calibration").exists()
    with pytest.raises(ValueError, match="reviewed hash-pinned gradient calibration"):
        run_local_backbone_pilot(local)


def test_calibration_pin_must_match_selected_set_and_hash(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "selected_coefficient_set": "very_conservative",
                "authorizes_training": False,
            }
        )
    )
    config["local_objective"]["selected_coefficient_set"] = "very_conservative"
    config["calibration"].update(
        {
            "reviewed": True,
            "report_path": str(report),
            "report_sha256": __import__("hashlib").sha256(report.read_bytes()).hexdigest(),
        }
    )
    assert _calibration_ready(config, raise_on_failure=True)
    config["calibration"]["report_sha256"] = "0" * 64
    assert not _calibration_ready(config, raise_on_failure=False)


def test_configuration_forbids_chirality_and_preserves_historical_gate() -> None:
    config = load_config(CONFIG)
    assert "chirality" not in json.dumps(config["local_objective"])
    assert config["selected_checkpoint"]["optimizer_update"] == 9000


def test_calibration_startup_reconstructs_compact_records_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual Phase-3I.2 startup with the exact compact selection schema."""
    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    canonical = {
        "sample_id": "sample-64",
        "sequence_length": 64,
        "coordinates": torch.zeros((64, 3)),
        "residue_mask": torch.ones(64, dtype=torch.bool),
        "chain_continuity_mask": torch.ones(63, dtype=torch.bool),
        "accepted_contiguous_single_chain": True,
    }
    compact = {
        "sample_id": "sample-64",
        "target_length": 64,
        "actual_length": 64,
        "dataset_shard_path": "validation/shard.parquet",
        "shard_row_index": 7,
    }
    config = {
        "calibration_output_dir": str(tmp_path / "calibration"),
        "dataset_source_config": "source.yaml",
        "device": "cuda",
        "calibration": {"lengths": [64], "timesteps": [25]},
        "diffusion_steps": 500,
        "coordinate_scale_angstrom": 12.22820347644835,
        "local_objective": {"candidate_coefficient_sets": {}},
    }
    calibration_config = tmp_path / "config.yaml"
    calibration_config.write_text("stub")
    calls: list[str] = []
    monkeypatch.setattr("protein_distance_diffusion.training.e007_local_backbone_repair.load_config", lambda _p: config)
    monkeypatch.setattr(
        "protein_distance_diffusion.training.e007_local_backbone_repair.verify_protected_evidence", lambda _c: {}
    )
    monkeypatch.setattr(localization, "load_config", lambda _p: {"panel": {"lengths": [64], "samples_per_length": 1}})
    monkeypatch.setattr(localization, "verify_prerequisites", lambda _c, full: {"hashes": {}, "full": full})
    monkeypatch.setattr(localization, "select_validation_panel", lambda _c: ([compact], {"record_sha256": "selection"}))

    def reconstruct(_source: object, rows: list[dict[str, object]]):
        assert rows == [compact]
        calls.append("reconstruct")
        return (
            [
                {
                    "selection": compact,
                    "canonical_row": canonical,
                    "provenance": {
                        "source_relocation_resolution": "recorded_path",
                        "npz_relocation_resolution": "recorded_path",
                    },
                }
            ],
            {"accepted_row_count": 1, "identity_sha256": "canonical"},
        )

    monkeypatch.setattr(localization, "reconstruct_authoritative_panel", reconstruct)
    payload = run_validate_calibration_panel(calibration_config)
    assert payload["selected_row_count"] == payload["accepted_row_count"] == 1
    assert payload["counts_by_requested_length"] == {"64": 1}
    assert payload["canonical_panel_identity_sha256"] == "canonical"
    assert payload["model_created"] is False and payload["output_created"] is False
    assert not (tmp_path / "calibration").exists()

    def prepared(_source: object, row: dict[str, object]):
        assert set(canonical) <= set(row)
        assert "coordinates" not in compact
        calls.append("prepared")
        return {
            "coordinates": torch.zeros((1, 64, 3)),
            "residue_mask": torch.ones((1, 64), dtype=torch.bool),
            "chain_continuity_mask": torch.ones((1, 63), dtype=torch.bool),
            "lengths": torch.tensor([64]),
            "sample_ids": ["sample-64"],
        }

    monkeypatch.setattr(localization, "_prepared_reference", prepared)

    class PreparedBoundaryReached(Exception):
        pass

    class ModelStub:
        def requires_grad_(self, _required: bool) -> ModelStub:
            return self

        def train(self) -> ModelStub:
            return self

    def load_model(*_args: object) -> ModelStub:
        assert calls[-1] == "prepared"
        calls.append("model")
        return ModelStub()

    def stop_after_prepare(source: object, row: dict[str, object]):
        prepared(source, row)
        raise PreparedBoundaryReached

    monkeypatch.setattr(localization, "_load_model", load_model)
    monkeypatch.setattr(localization, "_prepared_reference", stop_after_prepare)
    import torch as torch_module

    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_module.cuda, "reset_peak_memory_stats", lambda *_a: None)
    with pytest.raises(PreparedBoundaryReached):
        run_gradient_calibration(calibration_config)
    assert calls[-2:] == ["reconstruct", "prepared"]
    assert "model" not in calls
    assert not (tmp_path / ".calibration.inprogress").exists()


def test_calibration_validation_failure_precedes_all_runtime_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch as torch_module

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    config_path = tmp_path / "config.yaml"
    config_path.write_text("unused")
    monkeypatch.setattr(
        "protein_distance_diffusion.training.e007_local_backbone_repair.validate_calibration_panel",
        lambda _path: (_ for _ in ()).throw(ValueError("authoritative panel rejection")),
    )
    monkeypatch.setattr(
        localization,
        "_load_model",
        lambda *_args: (_ for _ in ()).throw(AssertionError("checkpoint/model boundary crossed")),
    )
    monkeypatch.setattr(
        torch_module.cuda,
        "is_available",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA boundary crossed")),
    )
    with pytest.raises(ValueError, match="authoritative panel rejection"):
        run_gradient_calibration(config_path)
    assert not (tmp_path / ".calibration.inprogress").exists()
    assert not (tmp_path / "calibration").exists()


def test_paired_bootstrap_sign_and_determinism() -> None:
    first = paired_bootstrap_improvement([3.0, 4.0, 5.0], [2.0, 3.0, 4.0], seed=9, replicates=100)
    second = paired_bootstrap_improvement([3.0, 4.0, 5.0], [2.0, 3.0, 4.0], seed=9, replicates=100)
    assert first == second
    assert first["mean_improvement"] == 1.0
    assert first["ci_95"][0] > 0


def test_recovery_identity_rejects_arm_or_schedule_mismatch(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    local = tmp_path / "config.yaml"
    local.write_text(yaml.safe_dump(config, sort_keys=False))
    protected = {"a": "b"}
    state = {
        "version": "e007_local_backbone_repair_v1",
        "configuration_sha256": __import__("hashlib").sha256(local.read_bytes()).hexdigest(),
        "protected_hashes_sha256": __import__("hashlib")
        .sha256(json.dumps(protected, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest(),
        "arm": "matched_v_only",
        "paired_schedule_sha256": "schedule",
        "successful_optimizer_boundary": True,
    }
    _validate_recovery_state(state, local, protected, "matched_v_only", "schedule")
    with pytest.raises(ValueError, match="recovery identity contradiction"):
        _validate_recovery_state(state, local, protected, "matched_v_plus_local", "schedule")


def test_recovery_checkpoint_reproduces_uninterrupted_update(tmp_path: Path) -> None:
    torch.manual_seed(41)
    uninterrupted = torch.nn.Linear(3, 2)
    resumed = torch.nn.Linear(3, 2)
    resumed.load_state_dict(uninterrupted.state_dict())
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    optimizer.zero_grad(set_to_none=True)
    uninterrupted(inputs).square().mean().backward()
    optimizer.step()
    scheduler.step()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("version: test\n")
    protected = {"input": "hash"}
    payload = _recovery_payload(
        uninterrupted,
        optimizer,
        scheduler,
        1,
        4,
        12,
        config_path,
        protected,
        "matched_v_only",
        "schedule",
    )
    checkpoint = tmp_path / "latest.pt"
    _atomic_torch(checkpoint, payload)

    optimizer.zero_grad(set_to_none=True)
    uninterrupted(inputs).square().mean().backward()
    optimizer.step()
    scheduler.step()

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resumed.load_state_dict(loaded["model"])
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    resumed_scheduler.load_state_dict(loaded["scheduler"])
    _restore_rng_state(loaded["rng_state"])
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed(inputs).square().mean().backward()
    resumed_optimizer.step()
    resumed_scheduler.step()
    for expected, observed in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(expected, observed, atol=0, rtol=0)


def test_resume_discards_only_trailing_uncommitted_metric(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    for update in (1, 2, 3):
        with path.open("a") as stream:
            stream.write(json.dumps({"optimizer_update": update}) + "\n")
    result = _reconcile_metrics_to_checkpoint(path, 2)
    assert result == {"retained": 2, "discarded_uncommitted": 1}
    assert [json.loads(line)["optimizer_update"] for line in path.read_text().splitlines()] == [1, 2]
    with path.open("a") as stream:
        stream.write(json.dumps({"optimizer_update": 4}) + "\n")
    with pytest.raises(ValueError, match="duplicated or out of order"):
        _reconcile_metrics_to_checkpoint(path, 2)
