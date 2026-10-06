import inspect

import numpy as np
import pytest
import torch

from protein_distance_diffusion.training import e010_direct_correction_v11 as direct
from protein_distance_diffusion.training import e010_strict_scientific_v10 as old
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_strict_scientific_v10 import setup


def test_zero_parity_no_radial_call_and_input_isolation(monkeypatch):
    b = synthetic(12)
    before = {k: v.clone() for k, v in b.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("radial map executed")

    monkeypatch.setattr(direct.v9.v8, "physical_trajectory", forbidden)
    oracle = direct.Oracle(b)
    z = np.zeros(np.prod(oracle.shape))
    assert oracle.fun(z)[0] == 1
    assert all(torch.equal(p, b["pg"]) for p in oracle.point(z).tr["states"])
    assert not isinstance(oracle, torch.nn.Module)
    for k in b:
        assert torch.equal(b[k], before[k]) and b[k].grad is None
    assert "sqrt" not in inspect.getsource(direct.trajectory)
    with pytest.raises(ValueError, match="float64"):
        direct.Oracle({k: v.float() if v.is_floating_point() else v for k, v in b.items()})


def test_linear_scaling_condition_one_closed_bound_and_cumulative():
    b = synthetic(12)
    z = torch.zeros((8, *b["pg"].shape), dtype=torch.float64)
    z[..., 0] = 1
    tr = direct.trajectory(b["pg"], b["mask"], z)
    for s in tr["steps"]:
        assert (s["delta"].norm(dim=-1) <= 0.04).all()
        assert (s["delta"][~s["eligible"]] == 0).all()
        torch.testing.assert_close(s["delta"][s["eligible"]], (0.04 * z[0])[s["eligible"]], atol=0, rtol=0)
    assert (tr["prediction"] - b["pg"]).norm(dim=-1).max() <= 0.320000000001
    assert sum(s["delta"].norm(dim=-1) for s in tr["steps"]).max() <= 0.32
    assert np.linalg.cond(0.04 * np.eye(3)) == 1
    j = torch.autograd.functional.jacobian(lambda x: 0.04 * x, torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(j, 0.04 * torch.eye(3, dtype=torch.float64), atol=0, rtol=0)


def test_ball_value_gradient_hessian_and_hvp():
    z = np.sin(np.arange(18) + 1) * 0.1
    d = np.cos(np.arange(18) + 1)
    q = torch.tensor(z, requires_grad=True)
    vals = 1 - q.reshape(-1, 3).square().sum(-1)
    j = torch.autograd.functional.jacobian(lambda x: 1 - x.reshape(-1, 3).square().sum(-1), q)
    np.testing.assert_array_equal(direct.Oracle.ball_values(z), vals.detach().numpy())
    np.testing.assert_array_equal(direct.Oracle.ball_jacobian(z).toarray(), j.detach().numpy())
    mu = np.arange(6) + 0.2
    h = torch.autograd.functional.hessian(
        lambda x: ((1 - x.reshape(-1, 3).square().sum(-1)) * torch.tensor(mu)).sum(), q
    )
    np.testing.assert_array_equal(direct.Oracle.ball_hessian(z, mu).toarray(), h.detach().numpy())
    np.testing.assert_allclose(
        (direct.Oracle.ball_values(z + 1e-5 * d) - direct.Oracle.ball_values(z - 1e-5 * d)) / 2e-5,
        direct.Oracle.ball_jacobian(z) @ d,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        (mu @ direct.Oracle.ball_jacobian(z + 1e-5 * d) - mu @ direct.Oracle.ball_jacobian(z - 1e-5 * d)) / 2e-5,
        direct.Oracle.ball_hessian(z, mu) @ d,
        atol=1e-10,
    )


def test_interior_pipeline_and_scientific_constraints_parity():
    b = synthetic(12)
    o, historical = direct.Oracle(b), direct.v9.Oracle(b)
    z = np.sin(np.arange(np.prod(o.shape)) + 1) * 0.05
    d = 0.04 * torch.from_numpy(z).reshape(o.shape)
    v = d / (1 - d.square().sum(-1, keepdim=True) / 0.04**2).sqrt()
    p = o.point(z)
    h = historical.point((v / 0.04).numpy().reshape(-1))
    for x, y in zip(p.tr["states"], h.tr["states"], strict=True):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    for k in p.values:
        torch.testing.assert_close(p.values[k], h.values[k], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(p.constraints, h.constraints, atol=1e-12, rtol=1e-12)
    assert o.quartets.telemetry(p.tr["prediction"], 1e-5) == historical.quartets.telemetry(h.tr["prediction"], 1e-5)


def test_scientific_jacobian_and_objective_and_lagrangian_hvp():
    o = direct.Oracle(synthetic(12))
    z = np.sin(np.arange(np.prod(o.shape)) + 1) * 0.05
    d = np.cos(np.arange(len(z)) + 1)
    d /= np.linalg.norm(d)
    p = o.point(z)
    exact = torch.stack([torch.autograd.grad(c, p.z, retain_graph=True)[0] for c in p.constraints]).detach().numpy()
    np.testing.assert_allclose(o.cjac(z).toarray(), exact, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        (o.cfun(z + 1e-5 * d) - o.cfun(z - 1e-5 * d)) / 2e-5, o.cjac(z) @ d, atol=1e-8, rtol=1e-5
    )
    np.testing.assert_allclose(
        (o.fun(z + 1e-5 * d)[0] - o.fun(z - 1e-5 * d)[0]) / 2e-5, o.fun(z)[1] @ d, atol=1e-8, rtol=1e-5
    )
    np.testing.assert_allclose(
        (o.fun(z + 1e-5 * d)[1] - o.fun(z - 1e-5 * d)[1]) / 2e-5, o.hess(z) @ d, atol=1e-8, rtol=1e-5
    )
    mu = np.linspace(0.1, 0.3, len(o.cfun(z)))
    np.testing.assert_allclose(
        (mu @ o.cjac(z + 1e-5 * d) - mu @ o.cjac(z - 1e-5 * d)) / 2e-5, o.chess(z, mu) @ d, atol=1e-8, rtol=1e-5
    )


def test_physical_certificate_and_shadow_parity():
    _, cfg, limits = setup()
    b = synthetic(12)
    delta = torch.sin(torch.arange(8 * b["pg"].numel(), dtype=torch.float64)).reshape(8, *b["pg"].shape) * 0.001
    mu = np.zeros(len(direct.v9.QuartetConstraints(b)(b["pg"])) + 2)
    assert direct.certificate(b, delta, mu, cfg, limits) == old.certificate(b, delta, mu, cfg, limits)
    assert direct.v9.QuartetConstraints is old.v9.QuartetConstraints
    assert direct.v9.v8.normalized_constraints is old.v9.v8.normalized_constraints
    assert direct.v9.v8.terms is old.v9.v8.terms
