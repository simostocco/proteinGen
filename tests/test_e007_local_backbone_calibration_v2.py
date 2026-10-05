from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from protein_distance_diffusion.training.e007_local_backbone_calibration_v2 import (
    LENGTHS,
    TIMESTEPS,
    candidate_weights,
    cosine_vp_schedule,
    evaluate_gates,
    load_config,
    plan_only,
    schedule_weight,
    summarize,
    validate_config,
    validate_panel,
    worst_cells,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e007_local_backbone_repair_calibration_v2.yaml"
V2_REPORT = ROOT / (
    "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v2/report.json"
)


def test_schedule_weights_are_finite_monotone_and_independent_of_length() -> None:
    schedule = cosine_vp_schedule(500)
    weights = [schedule_weight(schedule[t]) for t in TIMESTEPS]
    assert all(0 < value <= 1 for value in weights)
    assert weights == sorted(weights, reverse=True)
    assert schedule_weight(schedule[25]) > 0.9
    assert schedule_weight(schedule[499]) < schedule_weight(schedule[425])
    config = load_config(CONFIG)
    independent = [
        candidate_weights(config, 425)["schedule_tapered"]
        for _length in LENGTHS
        for _sample_id in ("panel-1", "panel-2", "panel-3", "panel-4")
    ]
    assert all(weights == independent[0] for weights in independent)


def test_schedule_endpoint_numerics_and_range() -> None:
    assert schedule_weight(1 - 1e-15) == pytest.approx(1.0)
    assert schedule_weight(1e-15) > 0
    with pytest.raises(ValueError):
        schedule_weight(0)


def test_gate_evaluation_and_fail_closed_multiple_passes() -> None:
    item = {
        "total_to_v_gradient_ratio": 1.0,
        "terms": {
            "a": {
                "weighted_gradient_ratio": 0.1,
                "eligible_count": 1,
                "weighted_gradient_norm": 0.1,
                "raw_gradient_finite": True,
                "weighted_gradient_finite": True,
            }
        },
    }
    # Legacy float32 telemetry rounded a fully finite mean below 1.0; exact flags/counts gate instead.
    profile = {
        "v_finite_gradient_coverage": 0.9999999403953552,
        "terms": {"x": {"finite_gradient_coverage": 0.9999999403953552}},
    }
    rows = [
        {
            "raw_and_gradient_finite": True,
            "gradient_profile": profile,
            "candidate_coefficients": {"a": item, "b": copy.deepcopy(item)},
        }
    ]
    assert evaluate_gates(rows, ["a", "b"])["selected_set"] is None
    rows[0]["candidate_coefficients"]["b"]["total_to_v_gradient_ratio"] = 1.31
    assert evaluate_gates(rows, ["a", "b"])["selected_set"] == "a"


def test_percentile_aggregation_and_worst_cell_identity() -> None:
    stats = summarize([1, 2, 3, 4, 100])
    assert stats["minimum"] == 1
    assert stats["median"] == 3
    assert stats["p90"] == pytest.approx(61.6)
    rows = [
        {
            "identity": {"length": 64, "timestep": 25, "sample_id": "a"},
            "candidate_coefficients": {"x": {"total_to_v_gradient_ratio": 1.0}},
        },
        {
            "identity": {"length": 128, "timestep": 499, "sample_id": "b"},
            "candidate_coefficients": {"x": {"total_to_v_gradient_ratio": 1.2}},
        },
    ]
    assert worst_cells(rows, "x")["maximum_total_ratio"]["sample_id"] == "b"


def test_panel_contract_and_plan_only_side_effect_free(tmp_path: Path) -> None:
    before = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    plan = plan_only(CONFIG)
    assert plan["structure_timestep_cells"] == plan["forward_count"] == 80
    assert plan["staging_created"] is plan["checkpoint_loaded"] is plan["cuda_initialized"] is False
    assert plan["vp_parameterization"]["x0_from_velocity_jacobian"] == "d(x0_hat)/d(v_hat) = -sigma_t I"
    assert plan["v_mse_coefficient"] == 1.0
    taper_values = []
    for timestep in TIMESTEPS:
        detail = plan["taper_schedule"][str(timestep)]
        assert 0 <= detail["signal_fraction_taper"] <= 1
        assert detail["coefficient_multipliers"]["very_conservative"] == 1.0
        assert detail["coefficient_multipliers"]["schedule_tapered"] == detail["signal_fraction_taper"]
        taper_values.append(detail["signal_fraction_taper"])
    assert taper_values == sorted(taper_values, reverse=True)
    assert (
        hashlib.sha256(V2_REPORT.read_bytes()).hexdigest()
        == "8e95fc94bad1633ca2e5af7880f6a1b1aa7b49af670ae10795f5e50965078b70"
    )
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == before


def test_exact_production_yaml_passes_shared_validator() -> None:
    config = load_config(CONFIG)
    assert validate_config(config) == config
    assert config["calibration"]["selection_seed"] == 3911001


def test_plan_only_rejects_missing_seed_and_each_required_leaf(tmp_path: Path) -> None:
    base = yaml.safe_load(CONFIG.read_text())
    from protein_distance_diffusion.training.e007_local_backbone_calibration_v2 import REQUIRED_PATHS

    for dotted in REQUIRED_PATHS:
        altered = copy.deepcopy(base)
        parent = altered
        parts = dotted.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        del parent[parts[-1]]
        candidate = tmp_path / "bad.yaml"
        candidate.write_text(yaml.safe_dump(altered, sort_keys=False))
        with pytest.raises(ValueError, match=dotted.replace(".", r"\.")):
            plan_only(candidate)


@pytest.mark.parametrize("seed", [True, "3911001", -1, 2**32, 1.5])
def test_selection_seed_type_and_range_are_strict(seed: object) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["calibration"]["selection_seed"] = seed
    with pytest.raises(ValueError, match="calibration.selection_seed"):
        validate_config(config)


def test_real_panel_validation_is_read_only_and_builds_expected_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("model/checkpoint/CUDA path touched")

    for name in ("_load_model",):
        monkeypatch.setattr(localization, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    report, panel = validate_panel(CONFIG)
    assert len(panel) == report["unique_selected_structures"] == 20
    assert report["structure_timestep_cells"] == 80
    assert report["selection_seed"] == 3911001
    assert set(report["counts_by_length"].values()) == {4}
    assert all(
        report[key] is False
        for key in (
            "model_created",
            "checkpoint_loaded",
            "cuda_initialized",
            "staging_created",
            "optimizer_created",
            "forward_pass",
            "backward_pass",
        )
    )
    assert (
        hashlib.sha256(V2_REPORT.read_bytes()).hexdigest()
        == "8e95fc94bad1633ca2e5af7880f6a1b1aa7b49af670ae10795f5e50965078b70"
    )


def test_v1_config_preserved() -> None:
    v1 = ROOT / "configs/e007_local_backbone_repair_v1.yaml"
    assert (
        hashlib.sha256(v1.read_bytes()).hexdigest()
        == "f07bfa7ab69409749396805422ef8a87b51f0736edf32d8d88dc69845bc21cf0"
    )
    report = (
        ROOT / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v1/report.json"
    )
    assert (
        hashlib.sha256(report.read_bytes()).hexdigest()
        == "1ed084323d95ac6245aca3a17506a92dbabbd07a6993caab590304a576ca9144"
    )
