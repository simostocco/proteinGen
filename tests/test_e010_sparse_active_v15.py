import inspect

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from protein_distance_diffusion.training import e010_sparse_active_v15 as s
from scripts.recover_e010_conditioning_v9b import synthetic


def test_projection_exact_and_minimal():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(10000, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x *= np.nextafter(1.0, 2.0)
    p, info = s.project_z(x)
    assert (np.linalg.norm(0.04 * p, axis=1) <= 0.04).all()
    assert (np.sum(p * p, axis=1) <= 1).all()
    assert info["maximum_inward_guard_angstrom"] <= 8 * np.spacing(0.04)
    q, _ = s.project_z(p)
    assert np.array_equal(p, q)
    assert np.array_equal(s.project_z(x * 0.5)[0], x * 0.5)


def test_qp_analytic_solution():
    d, mu, log = s.sparse_qp(np.array([-2.0, 1.0]), csr_matrix([[1.0, 0.0]]), np.array([0.5]), 1.0)
    np.testing.assert_allclose(d, [0.5, -1.0], atol=1e-9)
    np.testing.assert_allclose(mu, [1.5], atol=1e-9)
    assert log["success"] and log["dense_quadratic_arrays"] == 0


def test_qp_empty_and_inconsistent():
    d, mu, log = s.sparse_qp(np.ones(3), csr_matrix((0, 3)), np.empty(0), 2.0)
    np.testing.assert_equal(d, -0.5)
    assert log["success"] and not len(mu)
    assert not s.sparse_qp(np.ones(3), csr_matrix((1, 3)), np.array([-1.0]), 2.0)[2]["success"]


def test_no_dense_quadratic_path():
    source = inspect.getsource(s.sparse_qp)
    assert "scaled.T @ mu" in source
    assert "a @ a.T" not in source and "np.eye" not in source
    with pytest.raises(TypeError):
        s.sparse_qp(np.ones(3), np.zeros((1, 3)), np.zeros(1), 1.0)


def test_direct_zero_sparse_jacobian_and_active_set():
    o = s.direct.Oracle(synthetic(12))
    x = np.zeros(np.prod(o.shape))
    assert s.feasible(o, x)
    assert all(torch.equal(p, o.b["pg"]) for p in o.point(x).tr["states"]) if hasattr(o, "b") else s.feasible(o, x)
    a, rhs, ids, balls, f = s.active_set(o, x, o.cfun(x), o.cjac(x))
    assert isinstance(a, csr_matrix) and len(balls) == 0
    assert f["continuous_chirality"] == 1


def test_feasible_line_search():
    o = s.direct.Oracle(synthetic(12))
    x = np.zeros(np.prod(o.shape))
    f, g = o.fun(x)
    a, rhs, *_ = s.active_set(o, x, o.cfun(x), o.cjac(x))
    d, mu, qp = s.sparse_qp(g, a, rhs, 0.01)
    assert qp["success"]
    d /= max(1, np.linalg.norm(d.reshape(-1, 3), axis=1).max())
    p, log = s.line_search(o, x, float(f), g, d)
    assert p is not None and s.feasible(o, p) and log["objective_change"] < 0


def test_fixed_settings_and_mapping():
    assert s.SETTINGS["outer_maxiter"] == 2000
    assert s.direct.S == 0.04 and s.direct.K == 8
    assert "radial(" not in inspect.getsource(s)


def test_raw_state_saved_before_telemetry_exception(monkeypatch):
    from scripts import run_e010_direct_correction_v11 as h

    _, physical, limits, _ = h.setup()
    calls = []
    monkeypatch.setattr(
        s, "sparse_qp", lambda g, a, r, h: (np.zeros_like(g), np.zeros(len(r)), dict(success=False, reason="test"))
    )
    monkeypatch.setattr(s, "certify", lambda *args: (_ for _ in ()).throw(RuntimeError("telemetry")))
    with pytest.raises(RuntimeError, match="telemetry"):
        s.solve(synthetic(12), physical, limits, lambda x, meta: calls.append((x.copy(), meta)))
    assert len(calls) == 1 and np.isfinite(calls[0][0]).all()


def test_historical_derivatives_and_no_mutation():
    from scripts import run_e010_direct_correction_v11 as h

    b = synthetic(12)
    original = {k: v.clone() for k, v in b.items() if isinstance(v, torch.Tensor)}
    assert h.validate(b)["zero_exact"]
    for k, v in original.items():
        assert torch.equal(b[k], v)
