"""Synthetic physical conditioning/KKT verification, no panel or model."""

import numpy as np
import pytest
import torch

from protein_distance_diffusion.training import e010_conditioning_v9b as a
from scripts.recover_e010_conditioning_v9b import synthetic


def test_svd_rank_dependencies():
    out = a.spectrum(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]))
    assert out["rank"] == 2 and out["nearly_parallel_pairs"] == [[0, 1]]
    assert out["condition"] == pytest.approx(np.sqrt(5))


@pytest.mark.parametrize("g,expected", [([-1.0, 2.0], [0.0, -2.0]), ([1.0, 2.0], [-1.0, -2.0])])
def test_tangent_cone_and_nonnegative_multipliers(g, expected):
    d, mu = a.cone_projection(g, [[1.0, 0.0]])
    np.testing.assert_allclose(d, expected, atol=1e-14)
    assert (mu >= 0).all() and d[0] <= 0


def test_row_scaling_preserves_projection():
    g = np.array([-3.0, 1.0])
    d1, m1 = a.cone_projection(g, [[1.0, 0.0]])
    d2, m2 = a.cone_projection(g, [[7.0, 0.0]])
    np.testing.assert_allclose(d1, d2)
    assert m1[0] == pytest.approx(7 * m2[0])


def test_ball_kkt_outward_and_inward():
    d = np.array([[[[0.04, 0.0, 0.0]]]])
    eligible = np.ones(d.shape[:-1], bool)
    j = np.zeros((1, 3))
    c = np.array([-1.0])
    mu = np.zeros(1)
    out = a.components(d, eligible, np.array([-2.0, 0.0, 0.0]), j, c, mu)
    assert out["physical_normalized_stationarity_max"] == pytest.approx(0.0)
    assert out["ball_multipliers"][0] == pytest.approx(0.08)
    inward = a.components(d, eligible, np.array([2.0, 0.0, 0.0]), j, c, mu)
    assert inward["physical_normalized_stationarity_max"] == pytest.approx(0.08)
    assert inward["normalized_feasible_radial_max"] == pytest.approx(0.08)


def test_interior_kkt_and_fixed_invalid():
    d = np.zeros((1, 1, 2, 3))
    valid = np.array([[[True, False]]])
    result = a.components(d, valid, np.ones(6), np.zeros((1, 6)), np.array([-1.0]), np.zeros(1))
    assert result["physical_normalized_stationarity_l2"] == pytest.approx(0.04 * np.sqrt(3))


def test_active_extraction_and_ball_normals():
    d = np.array([[[[0.04, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
    valid = np.ones(d.shape[:-1], bool)
    mat, ids, balls = a.active_system(d, valid, [-1e-6, -1.0], np.ones((2, 6)))
    assert ids.tolist() == [0] and balls.tolist() == [0] and mat.shape == (2, 6)
    assert mat[1, 0] == 1.0


def test_radial_eigenvalues():
    v = torch.tensor([0.12, -0.08, 0.03], dtype=torch.float64)
    j = torch.autograd.functional.jacobian(lambda x: x / (1 + x.square().sum() / 0.04**2).sqrt(), v)
    stats = a.radial(v.numpy())
    eig = torch.linalg.eigvalsh(j).numpy()
    np.testing.assert_allclose(eig, [stats["radial"], stats["tangential"], stats["tangential"]], rtol=1e-12)


def test_feasible_shadow_bound():
    x = np.array([[[[1.0, -2.0, 3.0]]]])
    d = a.project_balls(x)
    assert np.linalg.norm(d, axis=-1).max() <= 0.04
    assert np.linalg.norm(8 * d, axis=-1).max() <= 0.320000000001


def test_physical_gradients_and_fd_lagrangian():
    b = synthetic(12)
    delta = torch.zeros((8, 1, 12, 3), dtype=torch.float64)
    d, f, c, tr, o, g, j = a.physical_jacobians(b, delta)
    assert d.dtype == torch.float64
    torch.testing.assert_close(tr["prediction"], b["pg"], atol=0, rtol=0)
    mu = np.linspace(0.001, 0.002, len(c))
    checks = a.finite_checks(b, delta, g, j, mu, [0, 1, 2], [1e-6, 1e-5], [0.0])
    assert all(row["passed"] for row in checks)
    assert not any(key == "local_model" for key in b)
    torch.testing.assert_close(b["pg"], synthetic(12)["pg"], atol=0, rtol=0)
