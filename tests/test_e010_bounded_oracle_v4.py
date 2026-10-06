import copy

import pytest
import torch

from protein_distance_diffusion.models.e010_hybrid_local import local_representation
from protein_distance_diffusion.training.e010_bounded_oracle_v4 import (
    initialize,
    metric_row,
    projected_stationarity,
    solve,
    trajectory,
)
from protein_distance_diffusion.training.e010_recurrent_capacity import objective_components


def example(n=12):
    t = torch.arange(n, dtype=torch.float64)
    pg = torch.stack((2.0 * t, torch.sin(t), torch.cos(t)), -1)[None].float().double()
    return {"pg": pg, "source": pg.clone(), "target": pg.clone(), "mask": torch.ones(1, n, dtype=torch.bool)}


SETTINGS = dict(
    learning_rate=1.0,
    history_size=20,
    maximum_iterations=30,
    stationarity_interval=10,
    relative_objective_tolerance=1e-10,
    coordinate_tolerance_angstrom=1e-6,
    projected_gradient_tolerance=0.001,
    stable_iterations_required=10,
)


def test_zero_parity_deterministic_initialization_no_neural_parameters():
    b = example()
    rng = torch.get_rng_state().clone()
    z = initialize(b["pg"])
    assert type(z) is torch.Tensor and z.is_leaf
    assert z.numel() == 4 * b["pg"].numel()
    assert torch.equal(z, initialize(b["pg"]))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p, b["pg"]) for p in trajectory(b["pg"], b["mask"], z)["states"])


@pytest.mark.parametrize("magnitude", [1.0, 1e6])
def test_bounds_endpoints_frames_recomputed(magnitude):
    b = example()
    z = initialize(b["pg"])
    with torch.no_grad():
        z[..., 0] = magnitude
        z[..., 1] = magnitude / 2
    tr = trajectory(b["pg"], b["mask"], z)
    for t, s in enumerate(tr["steps"]):
        assert torch.equal(s["frame"], local_representation(tr["states"][t], b["mask"])["frame"])
        assert (s["bounded_local"].norm(dim=-1) < 0.04).all()
        assert s["delta"].norm(dim=-1).max() <= 0.04000000001
        assert torch.count_nonzero(s["delta"][:, [0, -1]]) == 0
    assert not torch.equal(tr["steps"][0]["frame"], tr["steps"][1]["frame"])
    assert (tr["prediction"] - b["pg"]).norm(dim=-1).max() <= 0.160001
    a = torch.cdist(tr["prediction"], tr["prediction"])
    c = torch.cdist(b["pg"], b["pg"])
    assert (a - c).abs().max() <= 0.320002


def test_objective_v3_parity_and_no_global_input_gradient():
    b = example()
    b["pg"].requires_grad_()
    b["target"] = b["target"] + 0.02
    z = initialize(b["pg"])
    tr = trajectory(b["pg"], b["mask"], z)
    ls = objective_components(tr["prediction"], b, examples_total=60, quartets_total=13029)
    assert torch.equal(ls["total"], ls["local"] + 16.8 * ls["cartesian"] + 2 * ls["chiral"])
    ls["total"].backward()
    assert b["pg"].grad is None
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0


def test_independence_deterministic_solution_and_convergence_bookkeeping():
    b = example()
    original = copy.deepcopy(b)
    tr, log = solve(b, examples_total=60, quartets_total=13029, settings=SETTINGS)
    second, again = solve(b, examples_total=60, quartets_total=13029, settings=SETTINGS)
    assert log["converged"] and log["iterations"] == 10
    assert log["projected_gradient_residual"] == 0
    assert torch.equal(tr["prediction"], second["prediction"])
    assert log["history"] == again["history"]
    assert all(torch.equal(b[k], original[k]) for k in b)
    other = example()
    other["target"][:, 3, 0] += 0.1
    solve(other, examples_total=60, quartets_total=13029, settings={**SETTINGS, "maximum_iterations": 2})
    assert all(torch.equal(b[k], original[k]) for k in b)


def test_iteration_limit_is_not_convergence():
    b = example()
    b["target"][:, 3, 0] += 0.1
    _, log = solve(b, examples_total=60, quartets_total=13029, settings={**SETTINGS, "maximum_iterations": 1})
    assert log["iterations"] == 1 and not log["converged"]
    assert log["termination"] == "maximum_iterations"


def test_projected_gradient_does_not_confuse_radial_saturation_with_convergence():
    b = example()
    z = initialize(b["pg"])
    with torch.no_grad():
        z[..., 0] = 1e8
    tr = trajectory(b["pg"], b["mask"], z)
    loss = sum(s["bounded_local"][..., 1].sum() for s in tr["steps"])
    assert projected_stationarity(loss, tr) > 0.5


def test_metric_units_and_assessability():
    b = example()
    p = b["pg"] + torch.tensor([0.03, 0.04, 0.0], dtype=torch.float64)
    row = metric_row(p, b, {"sample_id": "synthetic", "condition": 50, "stratum": "20-64"})
    assert row["displacement_rms"] == pytest.approx(0.05)
    assert row["raw_cartesian"] == pytest.approx(0.0025 / 3 * 1.00001, rel=2e-5)
    assert row["aligned_rmsd"] < 1e-12
    assert row["continuous_chiral_loss"] < 1e-25


def test_independent_objective_contributions_preserve_full_panel_reductions():
    one = example()
    two = example()
    two["target"][:, 4, 0] += 0.2
    total = {k: torch.cat((one[k], two[k]), 0) for k in one}
    full = objective_components(total["pg"], total, examples_total=60, quartets_total=13029)["total"]
    independent = sum(
        objective_components(b["pg"], b, examples_total=60, quartets_total=13029)["total"] for b in [one, two]
    )
    assert torch.allclose(full, independent, atol=1e-15, rtol=1e-12)


def test_optimistic_pair_bound_is_not_an_oracle_loss():
    from scripts.run_e010_phase4d_bounded_oracle_v4 import optimistic_local_bound

    b = example()
    b["target"][:, 3:7, 0] += 0.2
    bounds = optimistic_local_bound(b)
    tr = trajectory(b["pg"], b["mask"], initialize(b["pg"]))
    row = metric_row(tr["prediction"], b, {"sample_id": "synthetic", "condition": 450, "stratum": "20-64"})
    assert bounds["mean_local_rmse_lower_bound"] <= row["mean_local_rmse"]
    for k in ["1", "2", "3"]:
        assert bounds["offset_rmse_lower_bounds"][k] <= row["local_rmse"][k]
