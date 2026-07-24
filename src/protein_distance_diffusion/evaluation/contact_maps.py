"""Derived contact-map utilities for visualization and diagnostics."""

from __future__ import annotations

import numpy as np


def binary_contact_map(
    distance_matrix: np.ndarray,
    *,
    threshold_angstrom: float = 8.0,
    exclude_diagonal: bool = True,
    exclude_near_diagonal: int = 0,
) -> np.ndarray:
    """Create a binary contact map from a continuous distance matrix.

    Args:
        distance_matrix: Continuous C-alpha distance matrix in angstrom with shape [N, N].
        threshold_angstrom: Contact threshold in angstrom.
        exclude_diagonal: Whether to zero diagonal contacts.
        exclude_near_diagonal: Also zero entries with `|i-j| <= exclude_near_diagonal`.

    Returns:
        Boolean contact map with shape [N, N].
    """
    d = np.asarray(distance_matrix, dtype=np.float32)
    contacts = d < np.float32(threshold_angstrom)
    idx = np.arange(d.shape[0])
    if exclude_diagonal:
        contacts[idx, idx] = False
    if exclude_near_diagonal > 0:
        near = np.abs(idx[:, None] - idx[None, :]) <= exclude_near_diagonal
        contacts[near] = False
    return contacts


def offdiagonal_pair_values(distance_matrix: np.ndarray, *, exclude_near_diagonal: int = 0) -> np.ndarray:
    """Return non-diagonal pair distances for statistics.

    Args:
        distance_matrix: Continuous distance matrix in angstrom with shape [N, N].
        exclude_near_diagonal: Exclude pairs with `|i-j| <= exclude_near_diagonal`.

    Returns:
        One-dimensional float32 array of pairwise distances.
    """
    d = np.asarray(distance_matrix, dtype=np.float32)
    idx = np.arange(d.shape[0])
    mask = np.triu(np.ones(d.shape, dtype=bool), k=1)
    if exclude_near_diagonal > 0:
        mask &= np.abs(idx[:, None] - idx[None, :]) > exclude_near_diagonal
    return d[mask]
