"""Deterministic, mask-aware quality diagnostics for distance matrices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MatrixQualityConfig:
    """Numerical settings for one-candidate matrix assessment."""

    contact_thresholds_angstrom: tuple[float, ...] = (6.0, 8.0, 10.0)
    contact_min_sequence_separation: int = 1
    adjacent_target_angstrom: float = 3.8
    adjacent_tolerance_angstrom: float = 0.5
    triangle_tolerance_angstrom: float = 1e-5
    triangle_exact_max_length: int = 64
    triangle_sample_count: int = 4096
    triangle_seed: int = 42
    eigen_exact_max_length: int = 512
    eigen_sample_size: int = 256
    eigen_seed: int = 42
    eigenvalue_tolerance: float = 1e-6


def _matrix_array(matrix: Any) -> np.ndarray:
    value = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.asarray(matrix)
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"distance matrix must be square [N, N], got {value.shape}")
    return value.copy()


def _mask_array(pair_mask: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if pair_mask is None:
        return np.ones(shape, dtype=bool)
    value = pair_mask.detach().cpu().numpy() if hasattr(pair_mask, "detach") else np.asarray(pair_mask)
    value = np.asarray(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape:
        raise ValueError(f"pair mask must match matrix shape {shape}, got {value.shape}")
    mask = value.astype(bool, copy=True)
    if not np.array_equal(mask, mask.T):
        raise ValueError("pair mask must be symmetric")
    return mask


def _valid_residue_indices(mask: np.ndarray, sequence_length: int | None) -> np.ndarray:
    n = mask.shape[0]
    if sequence_length is not None:
        length = int(sequence_length)
        if length < 0 or length > n:
            raise ValueError(f"sequence_length must be between 0 and {n}, got {length}")
        expected = np.zeros_like(mask)
        expected[:length, :length] = True
        if np.any(mask & ~expected):
            raise ValueError("pair mask includes positions beyond sequence_length")
        indices = np.arange(length, dtype=np.int64)
    else:
        indices = np.flatnonzero(np.diag(mask) | np.any(mask, axis=0) | np.any(mask, axis=1))
    if indices.size and not np.all(mask[np.ix_(indices, indices)]):
        raise ValueError("valid residues must define a complete biological pair mask")
    if np.any(mask) and not indices.size:
        raise ValueError("pair mask contains pairs but no valid residues")
    expected = np.zeros_like(mask)
    expected[np.ix_(indices, indices)] = True
    if not np.array_equal(mask, expected):
        raise ValueError("pair mask must be exactly the square mask of valid residues")
    return indices


def _triangle_metrics(matrix: np.ndarray, config: MatrixQualityConfig) -> dict[str, Any]:
    n = matrix.shape[0]
    available = n * (n - 1) * (n - 2) // 6
    if n < 3:
        return {
            "triangle_calculation": "not_applicable",
            "triangle_triplet_count": 0,
            "triangle_inequality_count": 0,
            "triangle_violation_count": 0,
            "triangle_violation_fraction": 0.0,
            "triangle_violation_mean_angstrom": 0.0,
            "triangle_violation_max_angstrom": 0.0,
        }
    if n <= config.triangle_exact_max_length or available <= config.triangle_sample_count:
        triples = np.asarray(list(combinations(range(n), 3)), dtype=np.int64)
        mode = "exact"
    else:
        rng = np.random.default_rng(config.triangle_seed)
        chosen: set[tuple[int, int, int]] = set()
        target = min(int(config.triangle_sample_count), available)
        while len(chosen) < target:
            chosen.add(tuple(sorted(rng.choice(n, size=3, replace=False).tolist())))
        triples = np.asarray(sorted(chosen), dtype=np.int64)
        mode = "sampled_without_replacement"
    sides = np.stack(
        [
            matrix[triples[:, 0], triples[:, 1]],
            matrix[triples[:, 0], triples[:, 2]],
            matrix[triples[:, 1], triples[:, 2]],
        ],
        axis=1,
    )
    violations = np.maximum(sides - (np.sum(sides, axis=1, keepdims=True) - sides), 0.0)
    flat = violations.reshape(-1)
    violating = flat > config.triangle_tolerance_angstrom
    return {
        "triangle_calculation": mode,
        "triangle_triplet_count": int(triples.shape[0]),
        "triangle_inequality_count": int(flat.size),
        "triangle_violation_count": int(np.sum(violating)),
        "triangle_violation_fraction": float(np.mean(violating)),
        "triangle_violation_mean_angstrom": float(np.mean(flat)),
        "triangle_violation_max_angstrom": float(np.max(flat)),
    }


def _edm_metrics(matrix: np.ndarray, config: MatrixQualityConfig) -> dict[str, Any]:
    n = matrix.shape[0]
    if n == 0:
        return {
            "edm_calculation": "not_applicable",
            "edm_evaluated_length": 0,
            "euclidean_gram_valid": True,
            "negative_eigenvalue_count": 0,
            "negative_eigenmass": 0.0,
            "negative_eigenmass_fraction": 0.0,
            "rank3_reconstruction_rmse_angstrom": 0.0,
            "rank3_residual_energy_fraction": 0.0,
            "reconstructed_coordinates_are_experimental": False,
        }
    if n <= config.eigen_exact_max_length:
        selected = np.arange(n)
        mode = "exact"
    else:
        size = min(config.eigen_sample_size, n)
        rng = np.random.default_rng(config.eigen_seed)
        selected = np.sort(rng.choice(n, size=size, replace=False))
        mode = "deterministic_principal_submatrix"
    d = 0.5 * (matrix[np.ix_(selected, selected)] + matrix[np.ix_(selected, selected)].T)
    np.fill_diagonal(d, 0.0)
    m = d.shape[0]
    centering = np.eye(m) - np.ones((m, m), dtype=np.float64) / m
    gram = -0.5 * centering @ np.square(d) @ centering
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    materially_negative = eigenvalues < -config.eigenvalue_tolerance * scale
    negative_mass = float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0])))
    total_mass = max(float(np.sum(np.abs(eigenvalues))), np.finfo(np.float64).eps)
    order = np.argsort(eigenvalues)[::-1]
    positive_order = order[eigenvalues[order] > 0.0]
    top3 = positive_order[:3]
    coordinates = np.zeros((m, 3), dtype=np.float64)
    for axis, index in enumerate(top3):
        coordinates[:, axis] = eigenvectors[:, index] * np.sqrt(eigenvalues[index])
    reconstructed = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    upper = np.triu(np.ones_like(d, dtype=bool), k=1)
    rmse = float(np.sqrt(np.mean(np.square(reconstructed[upper] - d[upper])))) if np.any(upper) else 0.0
    positive = np.clip(eigenvalues, 0.0, None)
    positive_energy = max(float(np.sum(np.square(positive))), np.finfo(np.float64).eps)
    residual = float(np.sum(np.square(positive[order[3:]])) / positive_energy) if m > 3 else 0.0
    return {
        "edm_calculation": mode,
        "edm_evaluated_length": int(m),
        "euclidean_gram_valid": not bool(np.any(materially_negative)),
        "negative_eigenvalue_count": int(np.sum(materially_negative)),
        "negative_eigenmass": negative_mass,
        "negative_eigenmass_fraction": negative_mass / total_mass,
        "rank3_reconstruction_rmse_angstrom": rmse,
        "rank3_residual_energy_fraction": residual,
        "reconstructed_coordinates_are_experimental": False,
    }


def assess_distance_matrix(
    matrix: Any,
    *,
    pair_mask: Any | None = None,
    sequence_length: int | None = None,
    candidate_id: str | None = None,
    config: MatrixQualityConfig | None = None,
) -> dict[str, Any]:
    """Assess one individual candidate without mutating or averaging it."""
    settings = config or MatrixQualityConfig()
    d_full = _matrix_array(matrix)
    mask = _mask_array(pair_mask, d_full.shape)
    valid = _valid_residue_indices(mask, sequence_length)
    d = d_full[np.ix_(valid, valid)]
    finite = np.isfinite(d)
    report: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_count": 1,
        "candidate_aggregation": "none",
        "sequence_length": int(valid.size),
        "valid_residue_count": int(valid.size),
        "valid_pair_count": int(valid.size * (valid.size - 1) // 2),
        "valid_matrix_cell_count": int(valid.size * valid.size),
        "finite": bool(finite.all()),
        "nonfinite_count": int(np.size(finite) - np.sum(finite)),
        "quality_score": None,
    }
    if not report["finite"]:
        report.update(
            {
                "metrics_available": False,
                "unavailable_reason": "nonfinite_distance_values",
                "triangle_calculation": "unavailable",
                "edm_calculation": "unavailable",
            }
        )
        return report
    n = d.shape[0]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    offdiag_values = d[upper]
    diagonal = np.diag(d)
    symmetry = np.abs(d - d.T)
    adjacent = np.diag(d, k=1)
    adjacent_error = adjacent - settings.adjacent_target_angstrom
    report.update(
        {
            "metrics_available": True,
            "symmetry_error_max_angstrom": float(np.max(symmetry)) if symmetry.size else 0.0,
            "symmetry_error_mean_angstrom": float(np.mean(symmetry)) if symmetry.size else 0.0,
            "diagonal_error_max_angstrom": float(np.max(np.abs(diagonal))) if diagonal.size else 0.0,
            "diagonal_error_mean_angstrom": float(np.mean(np.abs(diagonal))) if diagonal.size else 0.0,
            "negative_distance_count": int(np.sum(offdiag_values < 0.0)),
            "negative_distance_fraction": float(np.mean(offdiag_values < 0.0)) if offdiag_values.size else 0.0,
            "adjacent_pair_count": int(adjacent.size),
            "adjacent_residue_distance_mean_angstrom": float(np.mean(adjacent)) if adjacent.size else None,
            "adjacent_residue_distance_rmse_angstrom": (
                float(np.sqrt(np.mean(np.square(adjacent_error)))) if adjacent.size else None
            ),
            "adjacent_residue_plausible_fraction": (
                float(np.mean(np.abs(adjacent_error) <= settings.adjacent_tolerance_angstrom))
                if adjacent.size
                else None
            ),
        }
    )
    residue_index = np.arange(n)
    separation = np.abs(residue_index[:, None] - residue_index[None, :])
    contact_pairs = upper & (separation >= settings.contact_min_sequence_separation)
    contact_values = d[contact_pairs]
    report["contact_pair_count"] = int(contact_values.size)
    report["contact_density"] = {
        f"{threshold:g}": float(np.mean(contact_values <= threshold)) if contact_values.size else None
        for threshold in settings.contact_thresholds_angstrom
    }
    report.update(_triangle_metrics(d, settings))
    report.update(_edm_metrics(d, settings))
    return report


def assess_distance_matrices(
    matrices: Any,
    *,
    pair_masks: Any | None = None,
    sequence_lengths: list[int] | tuple[int, ...] | np.ndarray | None = None,
    candidate_ids: list[str] | tuple[str, ...] | None = None,
    config: MatrixQualityConfig | None = None,
) -> list[dict[str, Any]]:
    """Assess a batch as separate candidates; no cross-candidate reduction occurs."""
    values = matrices.detach().cpu().numpy() if hasattr(matrices, "detach") else np.asarray(matrices)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError(f"batched matrices must have shape [B, N, N] or [B, 1, N, N], got {values.shape}")
    masks = None
    if pair_masks is not None:
        masks = pair_masks.detach().cpu().numpy() if hasattr(pair_masks, "detach") else np.asarray(pair_masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.shape != values.shape:
            raise ValueError(f"batched pair masks must match matrix shape {values.shape}, got {masks.shape}")
    batch = values.shape[0]
    if sequence_lengths is not None and len(sequence_lengths) != batch:
        raise ValueError("sequence_lengths must contain one value per candidate")
    if candidate_ids is not None and len(candidate_ids) != batch:
        raise ValueError("candidate_ids must contain one identity per candidate")
    return [
        assess_distance_matrix(
            values[index],
            pair_mask=None if masks is None else masks[index],
            sequence_length=None if sequence_lengths is None else int(sequence_lengths[index]),
            candidate_id=None if candidate_ids is None else str(candidate_ids[index]),
            config=config,
        )
        for index in range(batch)
    ]
