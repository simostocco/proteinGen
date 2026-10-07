"""Independent allocation and derivative checks; no scientific optimizer invocation."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from protein_distance_diffusion.training import e010_sparse_active_v15 as solver
from scripts.recover_e010_conditioning_v9b import synthetic


def test_length500_local_qp_forbids_dense_square_allocations(monkeypatch):
    n, m = 12000, 4000
    a = csr_matrix((np.ones(m), (np.arange(m), 3 * np.arange(m))), shape=(m, n))
    g = np.tile([-2.0, 1.0, 0.0], m)
    for name in ("zeros", "empty", "ones", "full"):
        original = getattr(np, name)

        def guarded(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple) and len(shape) == 2 and shape[0] == shape[1] and shape[0] > 128:
                raise AssertionError("Dense quadratic allocation")
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, name, guarded)
    d, mu, log = solver.sparse_qp(g, a, np.full(m, 0.25), 1.0)
    assert log["success"] and log["dense_quadratic_arrays"] == 0
    np.testing.assert_allclose(d.reshape(-1, 3), np.tile([0.25, -1.0, 0.0], (m, 1)), atol=1e-8)
    assert log["estimated_sparse_working_bytes"] < 4 * 1024**2


@pytest.mark.parametrize("phase", [0.0, 0.7, 1.3])
def test_sparse_jacobian_matches_frozen_float64_finite_difference(phase):
    o = solver.direct.Oracle(synthetic(12))
    x = np.sin(np.arange(np.prod(o.shape)) + 1) * 0.05
    direction = np.sin(np.arange(len(x)) + 1 + phase)
    direction /= np.linalg.norm(direction)
    j = o.cjac(x)
    expected = j @ direction
    eps = 1e-6 / 0.04
    measured = (o.cfun(x + eps * direction) - o.cfun(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(measured, expected, atol=1e-8, rtol=1e-5)


def test_projection_rejects_nonfinite_and_lower_precision():
    with pytest.raises(ValueError):
        solver.project_z(np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError):
        solver.project_z(np.array([np.nan, 0.0, 0.0]))
