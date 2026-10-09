"""Focused tests of the observational paired-path diagnostic."""

import pytest
import torch

from protein_distance_diffusion.evaluation.reverse_field_diagnostic import (
    distribution,
    geometry_metrics,
    paired_state,
    reverse_identity,
    rollout_timesteps,
    select_rollout_subset,
    vector_metrics,
)
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
)
from protein_distance_diffusion.training.e014_runtime import d_path, sha


def test_true_v_identity_and_same_epsilon_with_padding():
    diffusion = CoordinateVPDiffusion(500)
    generator = torch.Generator().manual_seed(123)
    mask = torch.arange(20)[None] < torch.tensor([20, 13])[:, None]
    clean = center_coordinates(torch.randn(2, 20, 3, generator=generator), mask)
    epsilon = center_coordinates(torch.randn(2, 20, 3, generator=generator), mask)
    original = epsilon.clone()
    for value in (5, 50, 250, 400, 490):
        t = torch.full((2,), value)
        assert reverse_identity(diffusion, clean, epsilon, t, mask) < 5e-6
        current = paired_state(diffusion, clean, epsilon, t, mask)
        alpha, sigma = diffusion.alpha_sigma(t, clean)
        assert torch.allclose((current - alpha * clean) / sigma, epsilon, atol=2e-6)
        assert torch.equal(current[~mask], torch.zeros_like(current[~mask]))
        assert torch.allclose(current.sum(1), torch.zeros(2, 3), atol=3e-6)
    assert torch.equal(epsilon, original)


def test_noop_and_zero_v_are_different():
    diffusion = CoordinateVPDiffusion(500)
    mask = torch.ones(1, 20, dtype=torch.bool)
    current = center_coordinates(torch.arange(60).float().reshape(1, 20, 3), mask)
    noop = current.clone()
    previous, _, _ = diffusion.deterministic_reverse_step(current, torch.tensor([490]), torch.zeros_like(current), mask)
    assert torch.equal(noop, current)
    assert not torch.equal(previous, noop)
    assert (noop - current).square().mean() == 0


def test_vector_metrics_degenerate_and_normal():
    mask = torch.tensor([[True, True, False]])
    a = torch.tensor([[[1.0, 0, 0], [-1, 0, 0], [999, 0, 0]]])
    result = vector_metrics(a, 2 * a, mask, "v")
    assert result["v_cosine"] == pytest.approx(1)
    assert result["v_norm_ratio"] == pytest.approx(0.5)
    result = vector_metrics(torch.zeros_like(a), torch.zeros_like(a), mask, "v")
    assert result["v_cosine"] is None and result["v_norm_ratio"] is None
    result = vector_metrics(torch.zeros_like(a), a, mask, "v")
    assert result["v_norm_ratio"] == 0 and result["v_log_norm_ratio"] is None


def test_geometry_matches_oracle_not_clean_valid_ranges():
    mask = torch.ones(1, 20, dtype=torch.bool)
    a = center_coordinates(torch.randn(1, 20, 3), mask)
    result = geometry_metrics(a, a, mask, 12.22820347644835)
    assert result["raw_mse"] == result["L1_angstrom2"] == result["pair_distance_mse_angstrom2"] == 0
    assert result["rg_ratio"] == 1
    assert result["aligned_rmsd_angstrom"] < 1e-10


def test_exact_rollout_indexing():
    steps = rollout_timesteps()
    assert len(steps) == 50 and steps[0] == 490 and steps[-1] == 441
    assert [s - 1 for s in steps][-1] == 440
    with pytest.raises(ValueError):
        rollout_timesteps(440, 490)


def test_metric_blind_subset_deterministic():
    rows = [dict(sample_id=f"id{i}", length_stratum="a") for i in range(100)]
    assert select_rollout_subset(rows, ["a"]) == select_rollout_subset(list(reversed(rows)), ["a"])
    assert len({r["sample_id"] for r in select_rollout_subset(rows, ["a"])}) == 20


def test_checkpoint_and_model_not_mutated(tmp_path):
    model = torch.nn.Linear(3, 3).eval().requires_grad_(False)
    path = d_path(tmp_path / "frozen.pt")
    torch.save(model.state_dict(), path)
    digest = sha(path)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model(torch.ones(20, 3))
    assert sha(path) == digest
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_d_only_paths():
    assert str(d_path("/mnt/d/Simone/proteinGen/e014_artifacts/e014/reverse_field_50pct_v1")).startswith("/mnt/d/")
    for path in ("/tmp/experiment", "/home/experiment", "/mnt/c/experiment"):
        with pytest.raises(ValueError):
            d_path(path)


def test_boolean_summary_quantiles():
    result = distribution([True, False, True])
    assert result["mean"] == pytest.approx(2 / 3)
    assert result["denominator"] == 3
