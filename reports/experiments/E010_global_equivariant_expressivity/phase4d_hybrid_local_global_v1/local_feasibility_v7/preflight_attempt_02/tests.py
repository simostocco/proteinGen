from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.optimize import NonlinearConstraint

from protein_distance_diffusion.training import e010_local_feasibility_v7 as v7
from protein_distance_diffusion.training.e010_phase4d_objective_v2 import freeze_chirality
from protein_distance_diffusion.training.e010_precision_kkt_v6 import directional_checks

CONFIG = Path(__file__).resolve().parents[1] / "configs/e010_phase4d_local_feasibility_v7.yaml"


def geometry():
    t = torch.arange(9, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.2 * torch.sin(1.7 * t), 0.3 * torch.cos(1.2 * t), 0.1 * torch.sin(2.3 * t)], -1)[None]
    return dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, 9, dtype=torch.bool))


def test_zero_parity_bounds_and_endpoint_semantics():
    b = geometry()
    zeros = torch.zeros(4, *b["pg"].shape, dtype=torch.float64)
    tr = v7.physical_trajectory(b["pg"], b["mask"], zeros)
    for p in tr["states"]:
        torch.testing.assert_close(p, b["pg"], atol=0, rtol=0)
    tr = v7.physical_trajectory(b["pg"], b["mask"], torch.full_like(zeros, 1e5))
    assert max(float(s["delta"].norm(dim=-1).max()) for s in tr["steps"]) < 0.04
    assert float((tr["prediction"] - b["pg"]).norm(dim=-1).max()) <= 0.160001
    torch.testing.assert_close(tr["prediction"][:, [0, -1]], b["pg"][:, [0, -1]], atol=0, rtol=0)


def test_float64_only_terms():
    b = geometry()
    frozen = freeze_chirality(b["pg"], b["target"], b["mask"])
    assert all(v.dtype == torch.float64 for v in v7.terms(b["pg"], b, frozen).values())
    with pytest.raises(ValueError, match="float64"):
        v7.terms(b["pg"].float(), b, frozen)


def test_kabsch_value_proper_rotation_and_reflection():
    b = geometry()
    p = b["pg"]
    r = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
    assert v7.aligned_rmsd(p @ r + 5, p, b["mask"]) < 1e-12
    reflected = p * torch.tensor([-1.0, 1, 1], dtype=torch.float64)
    assert v7.aligned_rmsd(reflected, p, b["mask"]) > 0.01


def test_proper_kabsch_gradient_finite_differences():
    b = geometry()
    p = b["pg"].clone().requires_grad_()

    def fun(x):
        return v7.aligned_rmsd(x, b["target"], b["mask"])

    gradient = torch.autograd.grad(fun(p), p)[0]
    checks = directional_checks(fun, p.detach(), gradient, [1e-6, 1e-5, 1e-4])
    assert all(any(c["passed"] for c in checks if c["direction"] == j) for j in range(3))


def test_chiral_constraint_baseline_and_signed_objective():
    b = geometry()
    oracle = v7.Oracle(b, "B")
    z = np.zeros(np.prod(oracle.shape), dtype=np.float64)
    np.testing.assert_allclose(oracle.cfun(z), [1 / 1.01 - 1, 0], atol=1e-15)
    assert oracle.baseline["chiral"] > 0


@pytest.mark.parametrize("arm", ["A", "B"])
def test_local_only_no_hidden_weights_no_neural_mutation(arm, monkeypatch):
    b = geometry()
    before = b["pg"].clone()
    oracle = v7.Oracle(b, arm)
    monkeypatch.setattr(
        v7, "signed_status", lambda *args: (_ for _ in ()).throw(AssertionError("binary telemetry in gradient"))
    )
    z = np.linspace(-0.1, 0.1, np.prod(oracle.shape), dtype=np.float64)
    f, g = oracle.fun(z)
    point = oracle.point(z)
    assert f == float((point.values["local"] / oracle.baseline["local"]).detach())
    assert g.dtype == np.float64 and np.isfinite(g).all()
    torch.testing.assert_close(b["pg"], before, atol=0, rtol=0)
    assert b["pg"].grad is None


def test_exact_objective_and_constraint_hvp():
    oracle = v7.Oracle(geometry(), "B")
    z = np.linspace(-0.1, 0.1, np.prod(oracle.shape), dtype=np.float64)
    d = np.sin(np.arange(len(z)) + 1)
    d /= np.linalg.norm(d)
    h = oracle.hess(z) @ d
    numerical = (oracle.fun(z + 1e-5 * d)[1] - oracle.fun(z - 1e-5 * d)[1]) / 2e-5
    np.testing.assert_allclose(h, numerical, rtol=1e-5, atol=1e-9)
    hc = oracle.chess(z, [0.1, 0.3]) @ d
    numerical = (
        np.array([0.1, 0.3]) @ oracle.cjac(z + 1e-5 * d) - np.array([0.1, 0.3]) @ oracle.cjac(z - 1e-5 * d)
    ) / 2e-5
    np.testing.assert_allclose(hc, numerical, rtol=1e-5, atol=1e-9)


def test_zero_state_exact_hessian_and_same_radial_map():
    from protein_distance_diffusion.training.e010_precision_kkt_v6 import physical_trajectory

    oracle = v7.Oracle(geometry(), "B")
    z = np.zeros(np.prod(oracle.shape), dtype=np.float64)
    d = np.sin(np.arange(len(z)) + 1)
    assert np.isfinite(oracle.hess(z) @ d).all()
    assert np.isfinite(oracle.chess(z, [0.1, 0.3]) @ d).all()
    b = geometry()
    vector = torch.linspace(-2.0, 2.0, 108, dtype=torch.float64).reshape(4, 1, 9, 3)
    old = physical_trajectory(b["pg"], b["mask"], vector)
    new = v7.physical_trajectory(b["pg"], b["mask"], vector)
    for a, c in zip(old["states"], new["states"], strict=True):
        torch.testing.assert_close(a, c, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("center,feasible,expected", [(0.5, True, 0.5), (2.0, True, 1.0), (0.0, False, 0.0)])
def test_constrained_synthetic_feasible_active_infeasible(center, feasible, expected):
    cfg = yaml.safe_load(CONFIG.read_text())

    def fun(x):
        return ((x[0] - center) ** 2, np.array([2 * (x[0] - center)]))

    def hess(x):
        return np.array([[2.0]])

    cfun = (lambda x: np.array([x[0] ** 2 - 1])) if feasible else (lambda x: np.array([x[0] ** 2 + 1]))
    constraint = NonlinearConstraint(
        cfun, [-np.inf], [0.0], jac=lambda x: np.array([[2 * x[0]]]), hess=lambda x, v: np.array([[2 * v[0]]])
    )
    r = v7.constrained_minimize(fun, np.zeros(1, dtype=np.float64), hess, [constraint], cfg["solver"])
    if feasible:
        assert r.success and cfun(r.x)[0] <= cfg["constraints"]["normalized_feasibility_tolerance"]
        assert abs(r.x[0] - expected) < 1e-4
    else:
        assert not r.success or cfun(r.x)[0] > cfg["constraints"]["normalized_feasibility_tolerance"]


def test_metrics_zero_path_and_units():
    b = geometry()
    r = v7.metric_row(b["pg"], b, dict(sample_id="synthetic", condition=50, stratum="20-64"))
    assert r["step_correction_max"] == r["displacement_max"] == r["path_length_max"] == 0
    assert abs(sum(r["local_rmse"].values()) / 3 - r["mean_local_rmse"]) < 1e-15
