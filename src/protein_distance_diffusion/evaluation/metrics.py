"""Distance-map evaluation metrics that do not require coordinate generation."""

from __future__ import annotations

import numpy as np


def basic_identity_metrics(matrix: np.ndarray, *, eps: float = 1e-8) -> dict[str, float | bool]:
    """Compute symmetry, diagonal, negativity, and finite-value diagnostics.

    Args:
        matrix: Distance matrix in angstrom with shape [N, N].
        eps: Numerical denominator floor.

    Returns:
        Dictionary of scalar diagnostics.
    """
    d = np.asarray(matrix, dtype=np.float64)
    offdiag = ~np.eye(d.shape[0], dtype=bool)
    return {
        "symmetry_error": float(np.linalg.norm(d - d.T) / max(np.linalg.norm(d), eps)),
        "max_abs_diagonal": float(np.max(np.abs(np.diag(d)))),
        "negative_offdiag_fraction": float(np.mean(d[offdiag] < 0.0)),
        "finite": bool(np.isfinite(d).all()),
    }


def local_chain_statistics(matrix: np.ndarray, separations: tuple[int, ...] = (1, 2, 3)) -> dict[str, float]:
    """Summarize distances at selected sequence separations.

    Args:
        matrix: Distance matrix [N, N] in angstrom.
        separations: Sequence separations |i-j| to summarize.

    Returns:
        Mean and standard deviation metrics.
    """
    d = np.asarray(matrix, dtype=np.float64)
    out: dict[str, float] = {}
    for sep in separations:
        vals = np.diag(d, k=sep)
        out[f"sep_{sep}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"sep_{sep}_std"] = float(vals.std()) if vals.size else float("nan")
    return out


def triangle_violation_metrics(matrix: np.ndarray, *, num_triplets: int = 1024, seed: int = 42) -> dict[str, float]:
    """Sample triangle-inequality violations for diagnostics only.

    Args:
        matrix: Distance matrix [N, N] in angstrom.
        num_triplets: Number of sampled triplets.
        seed: RNG seed.

    Returns:
        Fraction, mean, and max violation magnitudes.
    """
    d = np.asarray(matrix, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    if n < 3:
        return {"triangle_violation_fraction": 0.0, "triangle_violation_mean": 0.0, "triangle_violation_max": 0.0}
    triples = rng.integers(0, n, size=(num_triplets, 3))
    violations = np.maximum(
        0.0,
        d[triples[:, 0], triples[:, 1]] - d[triples[:, 0], triples[:, 2]] - d[triples[:, 2], triples[:, 1]],
    )
    return {
        "triangle_violation_fraction": float(np.mean(violations > 0.0)),
        "triangle_violation_mean": float(violations.mean()),
        "triangle_violation_max": float(violations.max()),
    }


def edm_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    """Compute Gram-matrix eigenvalue diagnostics for a generated distance matrix.

    Args:
        matrix: Distance matrix [N, N] in angstrom.

    Returns:
        Negative eigenvalue mass and rank-three residual diagnostics.
    """
    d = np.asarray(matrix, dtype=np.float64)
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * j @ (d * d) @ j
    eig = np.linalg.eigvalsh(gram)
    neg = eig[eig < 0]
    total = np.sum(np.abs(eig)) + 1e-8
    pos = np.sort(eig[eig > 0])[::-1]
    residual = max(float(np.sum(pos[3:]) / total), 0.0) if pos.size > 3 else 0.0
    return {
        "negative_eigenvalue_mass": float(np.sum(np.abs(neg))),
        "negative_eigenvalue_mass_fraction": float(np.sum(np.abs(neg)) / total),
        "rank3_residual_energy_fraction": residual,
    }
