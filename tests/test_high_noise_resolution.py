"""Observational fronts and frozen bounded diagnostic contracts."""

import ast
from pathlib import Path

import numpy as np
import pytest

from protein_distance_diffusion.evaluation.high_noise_resolution import (
    ENDPOINTS,
    GRID,
    HIGH,
    LOW_MID,
    MARGINAL_GRID,
    collapse_onset,
    gaussian_seed,
    marginal_status,
    noisy_scale_status,
    prefix_steps,
    resolution_front,
    schedule_values,
)
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import d_path, noise_seed


def values(energy=(-0.2, -0.1), mmd=(-0.1, -0.01), sw=(-0.1, 0.1)):
    return [
        dict(metric=k, model_minus_noop_ci=v)
        for k, v in [("energy", energy), ("mmd2", mmd), ("sliced_wasserstein", sw)]
    ]


def test_dense_grid_and_exact_prefix_indexing():
    assert tuple(sorted(set(GRID))) == GRID
    assert set(HIGH + LOW_MID) == set(GRID)
    assert set(MARGINAL_GRID).issubset(GRID)
    steps = prefix_steps()
    assert len(steps) == 99 and steps[0] == 499 and steps[-1] == 401
    assert set(ENDPOINTS) - {400} <= set(steps)
    assert 0 not in steps


def test_snr_uses_frozen_schedule():
    diffusion = CoordinateVPDiffusion(500)
    previous = np.inf
    for t in GRID:
        row = schedule_values(diffusion, t)
        assert row["alpha"] ** 2 + row["sigma"] ** 2 == pytest.approx(1)
        assert row["snr"] == pytest.approx(row["alpha"] ** 2 / row["sigma"] ** 2)
        assert row["log_snr"] == pytest.approx(np.log(row["snr"]))
        assert row["snr"] < previous
        previous = row["snr"]
    with pytest.raises(ValueError):
        schedule_values(diffusion, 500)


def test_frozen_bootstrap_significance_and_scale_guard():
    assert marginal_status(values(), 1, 1) == "resolved_positive"
    assert marginal_status(values(energy=(-0.1, 0)), 1, 1) == "unresolved"
    assert marginal_status(values(mmd=(-0.1, 0)), 1, 1) == "unresolved"
    assert marginal_status(values(), 0.79, 1) == "unresolved"
    assert marginal_status(values(energy=(0.1, 0.2), mmd=(0.01, 0.1)), 1, 1) == "resolved_negative"


def test_resolution_front_distinguishes_isolated_crossings():
    rows = [
        dict(timestep=t, resolution_status=s)
        for t, s in [
            (5, "resolved_positive"),
            (50, "resolved_positive"),
            (250, "unresolved"),
            (350, "resolved_positive"),
        ]
    ]
    front = resolution_front(rows)
    assert front["highest_resolved_timestep"] == 350
    assert front["contiguous_sampled_prefix_front"] == 50 and front["nonmonotone"]
    assert resolution_front([])["highest_resolved_timestep"] is None


def test_direct_clean_onset_is_separate_from_historical_collapse():
    rows = [
        dict(timestep=t, direct_rg_median=m, direct_collapse_fraction=f)
        for t, m, f in [(400, 0.7, 0.1), (440, 0.49, 0.4), (450, 0.2, 0.6)]
    ]
    out = collapse_onset(rows)
    assert out["first_median_rg_below_half"] == 440
    assert out["first_historical_collapse_fraction_over_half"] == 450


def test_noisy_collapse_does_not_use_direct_estimate():
    assert noisy_scale_status(0.99, 0.98) == dict(rg_below_08=False, trace_below_08=False)
    assert noisy_scale_status(0.79, 0.9)["rg_below_08"]
    assert noisy_scale_status(0.8, 0.8) == dict(rg_below_08=False, trace_below_08=False)


def test_length_fronts_are_not_pooled():
    short = resolution_front([dict(timestep=400, resolution_status="resolved_positive")])
    long = resolution_front([dict(timestep=400, resolution_status="unresolved")])
    assert short["highest_resolved_timestep"] == 400 and long["highest_resolved_timestep"] is None


def test_gaussian_prefix_exact_previous_seeds():
    for i, length in enumerate([20, 128, 256, 384, 500]):
        expected = noise_seed("e014-marginal-gaussian-v1", f"sample-{i}-N{length}", 499)
        assert gaussian_seed(i, length, noise_seed) == expected


def test_no_target_in_model_call():
    tree = ast.parse(Path("scripts/diagnose_e014_high_noise.py").read_text())
    modelcalls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            isinstance(n.func, ast.Name)
            and n.func.id == "model"
            or isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "model"
            and n.func.attr == "telemetry"
        )
    ]
    assert len(modelcalls) == 3
    for call in modelcalls:
        assert len(call.args) == 4 and not call.keywords
        assert ast.unparse(call.args[1]) == "t"
        assert ast.unparse(call.args[2]) == "lengths"
        assert ast.unparse(call.args[3]) == "mask"
        assert "clean" not in ast.unparse(call.args[0])


def test_generated_path_requires_d():
    assert str(d_path("/mnt/d/Simone/proteinGen/e014_artifacts/e014/high_noise_75pct_v1")).startswith("/mnt/d/")
    with pytest.raises(ValueError):
        d_path("/tmp/high_noise_75pct_v1")


def test_oracle_sensitivity_control_never_loads_or_calls_a_model():
    source = Path("scripts/calibrate_e014_high_noise_resolution.py").read_text()
    tree = ast.parse(source)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "torch.load" not in calls
    assert "model" not in calls
    assert "E014CapacityDenoiser" not in source
    assert "diffusion.deterministic_reverse_step" in calls
    assert "oracle_targets_never_enter_model_inference_or_rollout=True" in source
