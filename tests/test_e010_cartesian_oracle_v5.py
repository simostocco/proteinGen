import copy

import pytest
import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import local_representation
from protein_distance_diffusion.training import e010_bounded_oracle_v4 as v4
from protein_distance_diffusion.training import e010_cartesian_oracle_v5 as v5
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import objective_components
from scripts.run_e010_phase4d_cartesian_oracle_v5 import CONFIG, V4_CONFIG, canonical, setup


def example(n=12):
    t = torch.arange(n, dtype=torch.float64)
    x = torch.stack((2 * t, torch.sin(t), torch.cos(t)), -1)[None].float().double()
    return {"pg": x, "source": x.clone(), "target": x.clone(), "mask": torch.ones(1, n, dtype=torch.bool)}


def test_zero_init_exact_parity_no_neural_parameters():
    b = example()
    state = torch.get_rng_state().clone()
    z = v5.initialize(b["pg"])
    assert type(z) is torch.Tensor and z.is_leaf
    assert not isinstance(z, torch.nn.Parameter)
    assert all(torch.equal(p, b["pg"]) for p in v5.trajectory(b["pg"], b["mask"], z)["states"])
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e100])
def test_radial_step_cumulative_pair_bounds(scale):
    b = example()
    z = v5.initialize(b["pg"])
    with torch.no_grad():
        z[..., 0] = scale
        z[..., 1] = scale / 2
    tr = v5.trajectory(b["pg"], b["mask"], z)
    assert all((s["delta"].norm(dim=-1) < 0.04).all() for s in tr["steps"])
    # The mathematical .16 bound retains v4's coordinate-addition tolerance.
    assert (tr["prediction"] - b["pg"]).norm(dim=-1).max() <= 0.160001
    assert (torch.cdist(tr["prediction"], tr["prediction"]) - torch.cdist(b["pg"], b["pg"])).abs().max() <= 0.320002


def test_eligibility_recomputed_endpoints_degeneracy_padding():
    b = example()
    z = v5.initialize(b["pg"])
    with torch.no_grad():
        z[..., 0] = 1
        z[:, :, 4, 1] = 2
    tr = v5.trajectory(b["pg"], b["mask"], z)
    for t, s in enumerate(tr["steps"]):
        assert torch.equal(s["eligible"], local_representation(tr["states"][t], b["mask"])["eligible"])
        assert torch.count_nonzero(s["delta"][:, [0, -1]]) == 0
    x = torch.stack((torch.arange(8, dtype=torch.float64), torch.zeros(8), torch.zeros(8)), -1)[None]
    m = torch.ones(1, 8, dtype=torch.bool)
    with torch.no_grad():
        raw = v5.initialize(x) + 1
        assert all(torch.count_nonzero(s["delta"]) == 0 for s in v5.trajectory(x, m, raw)["steps"])
    p = torch.cat((b["pg"], torch.zeros(1, 3, 3, dtype=torch.float64)), 1)
    mask = torch.cat((b["mask"], torch.zeros(1, 3, dtype=torch.bool)), 1)
    zz = torch.cat((z, torch.zeros(4, 1, 3, 3, dtype=torch.float64)), 2)
    assert torch.equal(v5.trajectory(p, mask, zz)["prediction"][:, :12], tr["prediction"])


def test_same_free_correction_balls_at_matching_trajectory():
    b = example()
    local = v4.initialize(b["pg"])
    with torch.no_grad():
        local[..., 0] = 0.7
        local[..., 1] = 0.2
    a = v4.trajectory(b["pg"], b["mask"], local)
    cart = torch.stack([torch.einsum("bnij,bnj->bni", s["frame"], local[t]) for t, s in enumerate(a["steps"])])
    z = v5.trajectory(b["pg"], b["mask"], cart)
    for p, q in zip(a["states"], z["states"], strict=True):
        assert torch.allclose(p, q, atol=1e-12, rtol=0)


def test_solver_code_is_identical_globals_isolated_optimizer_config_identical():
    assert v5.solve.__code__ is v4.solve.__code__
    assert v5.solve.__globals__ is not v4.solve.__globals__
    assert v4.solve.__globals__["trajectory"] is v4.trajectory
    assert v5.solve.__globals__["trajectory"] is v5.trajectory
    assert v5.projected_stationarity is v4.projected_stationarity
    assert setup()["optimizer"] == yaml.safe_load(V4_CONFIG.read_text())["optimizer"]
    assert yaml.safe_load(CONFIG.read_text())["objective"] == yaml.safe_load(V4_CONFIG.read_text())["objective"]


def test_objective_exact_parity_and_no_pg_mutation_gradient():
    b = example()
    original = copy.deepcopy(b)
    b["pg"].requires_grad_()
    z = v5.initialize(b["pg"])
    p = v5.trajectory(b["pg"], b["mask"], z)["prediction"]
    ls = objective_components(p, b, examples_total=60, quartets_total=13029)
    old = v4.solve.__globals__["objective_components"](p, b, examples_total=60, quartets_total=13029)
    assert all(torch.equal(ls[k], old[k]) for k in ls)
    (p.square().sum()).backward()
    assert b["pg"].grad is None
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0
    assert all(torch.equal(b[k], original[k]) for k in b)


def test_metric_function_and_units_parity():
    assert v5.metric_row is v4.metric_row and v5.summarize is v4.summarize
    b = example()
    p = b["pg"] + torch.tensor([0.03, 0.04, 0.0], dtype=torch.float64)
    r = v5.metric_row(p, b, {"sample_id": "synthetic", "condition": 50, "stratum": "20-64"})
    assert r["displacement_rms"] == pytest.approx(0.05)
    assert r["raw_cartesian"] == pytest.approx(0.0025 / 3 * 1.00001, rel=2e-5)


def test_convergence_bookkeeping_and_reproduction():
    settings = setup()["optimizer"]
    b = example()
    x, a = v5.solve(b, examples_total=60, quartets_total=13029, settings=settings)
    y, c = v5.solve(b, examples_total=60, quartets_total=13029, settings=settings)
    assert a["converged"] and a["iterations"] == 10
    assert a["history"] == c["history"] and a["projected_gradient_residual"] == 0
    assert torch.equal(x["prediction"], y["prediction"])
    assert canonical({"optimizer": a}) == canonical({"optimizer": c})
    _, old = v4.solve(b, examples_total=60, quartets_total=13029, settings=settings)
    assert a["history"] == old["history"]


def test_paired_counts_and_insufficient_convergence_classification():
    old = [{"optimizer": {"converged": v}} for v in [False, True, True, False]]
    new = [{"optimizer": {"converged": v}} for v in [True, False, True, False]]
    assert set(v5.convergence_pairs(old, new).values()) == {1}
    g = {"50": 6, "250": 1, "450": 0.1}
    s = {k: True for k in g}
    assert v5.classify(47, 20, g, s) == "CART-O5"
    assert v5.classify(60, 17, g, s) == "CART-O5"
    assert v5.classify(48, 18, g, s) == "CART-O2"
    assert v5.classify(60, 20, {**g, "450": 5}, s) == "CART-O1"


def test_protected_hash_integrity(tmp_path):
    p = tmp_path / "protected.txt"
    p.write_text("immutable")
    pins = {str(p): file_hash(p)}
    assert_file_pins(pins)
    p.write_text("mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        assert_file_pins(pins)
