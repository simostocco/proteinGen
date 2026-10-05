"""Static and CPU-only contracts for the bounded Phase 3I.3 audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from protein_distance_diffusion.evaluation import e007_phase3i3_trajectory as audit

CONFIG = Path("configs/e007_denoiser_sampler_trajectory_audit_v1.yaml")


def test_exact_production_configuration_and_paired_identities() -> None:
    config = audit.load_config(CONFIG)
    work = audit.units(config)
    assert len(work) == 60
    assert [unit["seed"] for unit in work[:20]] == [unit["seed"] for unit in work[20:40]]
    assert [unit["seed"] for unit in work[:20]] == [unit["seed"] for unit in work[40:]]
    assert tuple(config["panel"]["milestones"]) == audit.MILESTONES
    assert set(audit.REPRESENTATIONS) == {"x_t", "v_hat", "x0_hat", "epsilon", "next_state"}
    assert config["runtime"]["estimated_model_forwards"] == len(work) * 500
    assert config["runtime"]["estimated_wall_seconds"] == 2400
    assert config["runtime"]["maximum_wall_seconds"] == 7200


def test_independent_sampler_algebra_and_boundaries() -> None:
    assert all(audit.verify_algebra().values())


def test_five_representations_remain_separate() -> None:
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates

    diffusion = CoordinateVPDiffusion(500)
    mask = torch.ones((1, 4), dtype=torch.bool)
    x = center_coordinates(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3), mask)
    v = center_coordinates(torch.arange(12, dtype=torch.float32).reshape(1, 4, 3).square(), mask)
    next_state, x0, epsilon = diffusion.deterministic_reverse_step(x, torch.tensor([250]), v, mask)
    values = dict(zip(audit.REPRESENTATIONS, (x, v, x0, epsilon, next_state), strict=True))
    assert len(values) == 5
    assert not torch.equal(values["x_t"], values["v_hat"])
    assert not torch.equal(values["x0_hat"], values["epsilon"])
    assert not torch.equal(values["x0_hat"], values["next_state"])


def test_plan_is_side_effect_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = audit.load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "final")
    config["staging_dir"] = str(tmp_path / "stage")
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(audit, "validate_contract", lambda _: pytest.fail("plan validated protected files"))
    result = audit.plan(path)
    assert result["checkpoint_loaded"] is False
    assert result["cuda_initialized"] is False
    assert list(tmp_path.iterdir()) == [path]


def test_contract_validation_is_read_only(tmp_path: Path) -> None:
    import yaml

    config = audit.load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "final")
    config["staging_dir"] = str(tmp_path / "stage")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    output = Path(config["output_dir"])
    staging = Path(config["staging_dir"])
    assert not output.exists() and not staging.exists()
    result = audit.validate_contract(path)
    assert result["protected_artifact_count"] == 25
    assert result["checkpoint_loaded"] is False
    assert result["cuda_initialized"] is False
    assert not output.exists() and not staging.exists()


def test_atomic_resume_and_corruption_rejection(tmp_path: Path) -> None:
    config = audit.load_config(CONFIG)
    work = audit.units(config)
    stage = tmp_path / "stage"
    rows = [
        {
            "arm": work[0]["arm"],
            "seed": work[0]["seed"],
            "length": work[0]["length"],
            "sample_index": work[0]["sample_index"],
            "representation": rep,
            "timestep": step,
            "coordinates_normalized": [[0.0, 0.0, 0.0] for _ in range(work[0]["length"])],
        }
        for step in audit.MILESTONES
        for rep in audit.REPRESENTATIONS
    ]
    audit._commit(stage, 0, work[0], rows)
    assert len(audit._read_journal(stage, work)) == 1
    (stage / audit._unit_path(0)).write_text("corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        audit._read_journal(stage, work)


def test_sign_change_and_non_authorizing_publication() -> None:
    rows = []
    for step in audit.MILESTONES:
        for representation in ("x0_hat", "next_state"):
            rows.append(
                {
                    "length": None,
                    "comparison": "v_plus_local_minus_v_only",
                    "metric": "adjacent_distance_rmse_to_3_8_angstrom",
                    "timestep": step,
                    "representation": representation,
                    "paired_mean_difference": -0.1 if step > 375 else 0.1,
                    "ci95_low": -0.2 if step > 375 else 0.05,
                }
            )
    result = audit.sign_changes(rows)
    assert result["first_advantage_disappears"] == 375
    assert result["first_sign_change"] == 375
    assert all(value is False for value in audit.NON_AUTHORIZING.values())


def test_sampler_transition_can_flip_sign_before_x0() -> None:
    rows = []
    for step in audit.MILESTONES:
        for representation in ("x0_hat", "next_state"):
            value = 0.1 if representation == "next_state" and step <= 425 else -0.1
            rows.append(
                {
                    "length": None,
                    "comparison": "v_plus_local_minus_v_only",
                    "metric": "adjacent_distance_rmse_to_3_8_angstrom",
                    "timestep": step,
                    "representation": representation,
                    "paired_mean_difference": value,
                    "ci95_low": value - 0.01,
                }
            )
    result = audit.sign_changes(rows)
    assert result["first_sign_change"] == 425
    assert result["deterioration_location"] == "sampler_transition"


def test_transient_x0_advantage_is_tracked_from_first_occurrence() -> None:
    rows = []
    for step in audit.MILESTONES:
        for representation in ("x0_hat", "next_state"):
            value = 0.1 if step in (499, 425, 25, 0) else -0.1
            rows.append(
                {
                    "length": None,
                    "comparison": "v_plus_local_minus_v_only",
                    "metric": "adjacent_distance_rmse_to_3_8_angstrom",
                    "timestep": step,
                    "representation": representation,
                    "paired_mean_difference": value,
                    "ci95_low": value - 0.01,
                }
            )
    result = audit.sign_changes(rows)
    assert result["first_advantage_milestone"] == 375
    assert result["first_advantage_disappears"] == 25


def test_higher_is_better_local_validity_uses_correct_sign() -> None:
    rows = []
    for step in audit.MILESTONES:
        for representation in ("x0_hat", "next_state"):
            value = 0.1 if step > 250 else -0.1
            rows.append(
                {
                    "length": None,
                    "comparison": "v_plus_local_minus_v_only",
                    "metric": "locally_valid_residue_fraction",
                    "timestep": step,
                    "representation": representation,
                    "paired_mean_difference": value,
                    "ci95_low": value - 0.01,
                    "ci95_high": value + 0.01,
                }
            )
    result = audit.sign_changes(rows, metric="locally_valid_residue_fraction")
    assert result["first_advantage_milestone"] == 499
    assert result["first_sign_change"] == 250


def test_local_metrics_are_graded() -> None:
    config = audit.load_config(CONFIG)
    index = np.arange(64)
    points = (
        np.stack((index * 3.8, 0.1 * np.sin(index), 0.1 * np.cos(index)), axis=-1) / config["coordinate_scale_angstrom"]
    )
    row = audit._representation_record(points, config)
    assert len(row["coordinates_normalized"]) == 64
    assert len(row["coordinates_normalized"][0]) == 3
    assert row["coordinate_scale_angstrom"] == config["coordinate_scale_angstrom"]
    assert row["adjacent_distance_violation_fraction"] == 0
    assert row["locally_valid_residue_fraction"] == 1
    assert row["longest_invalid_contiguous_segment"] == 0


def test_nonfinite_coordinate_record_is_serializable_and_incomplete() -> None:
    config = audit.load_config(CONFIG)
    values = np.zeros((64, 3), dtype=np.float64)
    values[3, 2] = np.nan
    row = audit._representation_record(values, config)
    assert row["coordinates_normalized"][3][2] is None
    assert row["finite_coordinate_rate"] == 0
    assert row["completion_rate"] == 0
    json.dumps(row, allow_nan=False)


def test_monitor_absent_is_read_only(tmp_path: Path) -> None:
    import yaml

    config = audit.load_config(CONFIG)
    config["output_dir"] = str(tmp_path / "final")
    config["staging_dir"] = str(tmp_path / "stage")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    result = audit.monitor(path)
    assert result["completed_model_forwards"] == 0
    assert not (tmp_path / "stage").exists()


def test_cleanup_preserves_aggregated_phase_telemetry() -> None:
    calls = []

    class FakeCuda:
        def empty_cache(self) -> None:
            calls.append("empty_cache")

    class FakeTelemetry:
        def end_phase(self) -> dict[str, float]:
            calls.append("end_phase")
            return {"run_peak_cuda_allocated_mib": 5900.0, "run_peak_cuda_reserved_mib": 6500.0}

    result = audit._cleanup_cuda(FakeCuda(), FakeTelemetry())
    assert calls == ["empty_cache", "end_phase"]
    assert result["run_peak_cuda_reserved_mib"] == 6500.0


def test_parameter_mutation_detection_without_model_forward() -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    baseline = audit._parameter_fingerprints(model)
    assert audit._parameter_fingerprints(model) == baseline
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert audit._parameter_fingerprints(model) != baseline
