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


def contact_fractions(matrix: np.ndarray) -> dict[str, float]:
    """Return off-diagonal contact fractions at common Angstrom thresholds."""
    d = np.asarray(matrix, dtype=np.float64)
    offdiag = ~np.eye(d.shape[0], dtype=bool)
    values = d[offdiag]
    return {f"contact_fraction_{threshold}A": float(np.mean(values <= threshold)) for threshold in (6, 8, 10)}


def generated_matrix_report(matrix: np.ndarray, *, scale: float) -> dict[str, object]:
    """Compute physical plausibility diagnostics for one generated normalized matrix."""
    raw = np.asarray(matrix, dtype=np.float64)
    physical = raw * float(scale)
    offdiag = ~np.eye(physical.shape[0], dtype=bool)
    report: dict[str, object] = {
        "raw_normalized_min": float(np.nanmin(raw)),
        "raw_normalized_max": float(np.nanmax(raw)),
        "physical_distance_min_angstrom": float(np.nanmin(physical)),
        "physical_distance_max_angstrom": float(np.nanmax(physical)),
        "negative_distance_fraction": float(np.mean(physical[offdiag] < 0.0)),
        "nonfinite_fraction": float(np.mean(~np.isfinite(physical))),
    }
    report.update(basic_identity_metrics(physical))
    report.update(local_chain_statistics(physical, separations=(1,)))
    report["adjacent_residue_distance_mean"] = report.pop("sep_1_mean")
    report["adjacent_residue_distance_std"] = report.pop("sep_1_std")
    report.update(contact_fractions(physical))
    report.update(triangle_violation_metrics(physical, num_triplets=1024, seed=42))
    report.update(edm_diagnostics(physical))
    centered = np.eye(physical.shape[0]) - np.ones_like(physical) / physical.shape[0]
    gram = -0.5 * centered @ (physical * physical) @ centered
    report["effective_embedding_rank"] = int(np.sum(np.linalg.eigvalsh(gram) > 1e-6))
    report["physically_plausible"] = bool(
        report["finite"]
        and report["negative_distance_fraction"] == 0.0
        and report["physical_distance_max_angstrom"] < 2000.0
        and report["max_abs_diagonal"] < 1e-5
        and report["symmetry_error"] < 1e-5
        and report["negative_eigenvalue_mass_fraction"] < 0.05
        and report["triangle_violation_fraction"] < 0.05
    )
    return report
