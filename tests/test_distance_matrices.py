"""Distance-matrix tests."""

from __future__ import annotations

import numpy as np

from protein_distance_diffusion.data.preprocess import compute_distance_matrix


def test_known_distance_matrix() -> None:
    """A 3-4-5 coordinate set produces known C-alpha distances in angstrom."""
    coords = np.asarray([[0, 0, 0], [3, 4, 0], [3, 0, 0]], dtype=np.float32)
    d = compute_distance_matrix(coords)
    expected = np.asarray([[0, 5, 3], [5, 0, 4], [3, 4, 0]], dtype=np.float32)
    np.testing.assert_allclose(d, expected)
    assert d.dtype == np.float32
    assert np.allclose(d, d.T)
