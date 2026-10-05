"""Distance-map repairability and C-alpha trace diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(dst)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_replace_path(path: str | Path) -> Path:
    """Return a temporary sibling path for atomic dataframe/figure writes."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return dst.with_name(f".{dst.name}.tmp")


def symmetrize_zero_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetrized copy with zero diagonal."""
    d = np.asarray(matrix, dtype=np.float64).copy()
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d


@dataclass(frozen=True)
class MDSProjection:
    """Rank-3 classical-MDS projection result."""

    coordinates: np.ndarray
    projected_distances: np.ndarray
    eigenvalues: np.ndarray
    negative_eigenvalue_mass_fraction: float
    rank3_residual_energy_fraction: float


def classical_mds_rank3_projection(matrix: np.ndarray) -> MDSProjection:
    """Project a distance matrix to rank-3 coordinates with classical MDS.

    This is a rank-3 classical-MDS projection, not a claim of nearest-EDM
    optimality.
    """
    d = symmetrize_zero_diagonal(matrix)
    n = d.shape[0]
    centering = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / float(max(n, 1))
    gram = -0.5 * centering @ np.square(d) @ centering
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues[:3], a_min=0.0, a_max=None)
    coords = eigenvectors[:, :3] * np.sqrt(positive)[None, :]
    projected = pairwise_distances(coords)
    abs_energy = float(np.sum(np.abs(eigenvalues)))
    positive_energy = float(np.sum(np.square(np.clip(eigenvalues, a_min=0.0, a_max=None))))
    negative = eigenvalues[eigenvalues < 0.0]
    positive_tail = np.clip(eigenvalues[3:], a_min=0.0, a_max=None)
    return MDSProjection(
        coordinates=coords,
        projected_distances=projected,
        eigenvalues=eigenvalues,
        negative_eigenvalue_mass_fraction=float(np.sum(np.abs(negative)) / max(abs_energy, 1e-12)),
        rank3_residual_energy_fraction=float(np.sum(np.square(positive_tail)) / max(positive_energy, 1e-12)),
    )


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Return Euclidean pairwise distances for coordinates."""
    x = np.asarray(coords, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def upper_pair_mask(length: int, *, min_separation: int = 1) -> np.ndarray:
    """Return upper-triangular pair mask with sequence-separation filtering."""
    idx = np.arange(int(length))
    return np.triu(np.ones((int(length), int(length)), dtype=bool), k=1) & (
        np.abs(idx[:, None] - idx[None, :]) >= int(min_separation)
    )


def contact_metrics(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    threshold: float,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute binary contact precision, recall, F1 and Jaccard."""
    target_contacts = (np.asarray(target) <= float(threshold)) & mask
    predicted_contacts = (np.asarray(predicted) <= float(threshold)) & mask
    tp = int(np.logical_and(target_contacts, predicted_contacts).sum())
    fp = int(np.logical_and(~target_contacts, predicted_contacts).sum())
    fn = int(np.logical_and(target_contacts, ~predicted_contacts).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    jaccard = tp / max(tp + fp + fn, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
    }


def radius_of_gyration(coords: np.ndarray) -> float:
    """Return radius of gyration for coordinates."""
    x = np.asarray(coords, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    centered = x - x.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def distance_radius_of_gyration(matrix: np.ndarray) -> float:
    """Infer radius of gyration from a Euclidean distance matrix."""
    d2 = np.square(symmetrize_zero_diagonal(matrix))
    return float(np.sqrt(np.maximum(d2.sum() / (2.0 * max(d2.shape[0], 1) ** 2), 0.0)))


def repairability_metrics(matrix: np.ndarray, *, min_short_sep: int = 3, long_range_sep: int = 24) -> dict[str, float]:
    """Compute rank-3 projection repairability metrics for one matrix."""
    target = symmetrize_zero_diagonal(matrix)
    projection = classical_mds_rank3_projection(target)
    projected = projection.projected_distances
    mask = upper_pair_mask(target.shape[0], min_separation=1)
    target_values = target[mask]
    projected_values = projected[mask]
    delta = projected_values - target_values
    abs_delta = np.abs(delta)
    rmse = float(np.sqrt(np.mean(delta * delta))) if delta.size else float("nan")
    stress = float(np.linalg.norm(delta) / max(np.linalg.norm(target_values), 1e-12)) if delta.size else float("nan")
    idx = np.arange(target.shape[0])
    sep = np.abs(idx[:, None] - idx[None, :])
    short = mask & (sep >= int(min_short_sep)) & (sep < int(long_range_sep))
    long = mask & (sep >= int(long_range_sep))
    out = {
        "relative_frobenius_stress": stress,
        "offdiagonal_rmse_angstrom": rmse,
        "offdiagonal_mae_angstrom": float(np.mean(abs_delta)) if delta.size else float("nan"),
        "rmse_over_mean_target_distance": (
            rmse / max(float(np.mean(target_values)), 1e-12) if delta.size else float("nan")
        ),
        "abs_change_q50": float(np.percentile(abs_delta, 50)) if delta.size else float("nan"),
        "abs_change_q90": float(np.percentile(abs_delta, 90)) if delta.size else float("nan"),
        "abs_change_q95": float(np.percentile(abs_delta, 95)) if delta.size else float("nan"),
        "abs_change_q99": float(np.percentile(abs_delta, 99)) if delta.size else float("nan"),
        "fraction_changed_gt_1A": float(np.mean(abs_delta > 1.0)) if delta.size else float("nan"),
        "fraction_changed_gt_2A": float(np.mean(abs_delta > 2.0)) if delta.size else float("nan"),
        "fraction_changed_gt_4A": float(np.mean(abs_delta > 4.0)) if delta.size else float("nan"),
        "short_range_projection_rmse": _masked_rmse(target, projected, short),
        "long_range_projection_rmse": _masked_rmse(target, projected, long),
        "radius_of_gyration_change": radius_of_gyration(projection.coordinates) - distance_radius_of_gyration(target),
        "negative_eigenvalue_mass_fraction_before_projection": projection.negative_eigenvalue_mass_fraction,
        "rank3_residual_energy_fraction_before_projection": projection.rank3_residual_energy_fraction,
    }
    for threshold in (6.0, 8.0, 10.0):
        metrics = contact_metrics(target, projected, threshold=threshold, mask=mask)
        for name, value in metrics.items():
            out[f"contact_{int(threshold)}A_{name}"] = value
    return out


def _masked_rmse(target: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return float("nan")
    delta = np.asarray(predicted)[mask] - np.asarray(target)[mask]
    return float(np.sqrt(np.mean(delta * delta)))


def trace_metrics(coords: np.ndarray, *, clash_threshold: float = 3.0) -> dict[str, float]:
    """Compute projected C-alpha trace plausibility metrics."""
    x = np.asarray(coords, dtype=np.float64)
    n = x.shape[0]
    d = pairwise_distances(x)
    adjacent = np.diag(d, k=1)
    sep2 = np.diag(d, k=2)
    angles = virtual_bond_angles(x)
    dihedrals = virtual_dihedrals(x)
    non_neighbor = upper_pair_mask(n, min_separation=2)
    contacts8 = (d <= 8.0) & upper_pair_mask(n, min_separation=4)
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return {
        "ca_adjacent_distance_mean": float(np.mean(adjacent)) if adjacent.size else float("nan"),
        "ca_adjacent_distance_std": float(np.std(adjacent)) if adjacent.size else float("nan"),
        "ca_adjacent_rmse_to_3p8A": (
            float(np.sqrt(np.mean(np.square(adjacent - 3.8)))) if adjacent.size else float("nan")
        ),
        "sep2_distance_mean": float(np.mean(sep2)) if sep2.size else float("nan"),
        "sep2_distance_std": float(np.std(sep2)) if sep2.size else float("nan"),
        "virtual_bond_angle_mean_degrees": float(np.mean(angles)) if angles.size else float("nan"),
        "virtual_bond_angle_std_degrees": float(np.std(angles)) if angles.size else float("nan"),
        "virtual_dihedral_mean_degrees": float(np.mean(dihedrals)) if dihedrals.size else float("nan"),
        "virtual_dihedral_std_degrees": float(np.std(dihedrals)) if dihedrals.size else float("nan"),
        "non_neighbor_clash_fraction_lt_3A": float(np.mean(d[non_neighbor] < float(clash_threshold)))
        if non_neighbor.any()
        else float("nan"),
        "radius_of_gyration": radius_of_gyration(x),
        "long_range_contact_fraction": float(np.mean(d[upper_pair_mask(n, min_separation=24)] <= 8.0))
        if upper_pair_mask(n, min_separation=24).any()
        else float("nan"),
        "relative_contact_order": float(np.mean(seq_sep[contacts8] / max(n, 1))) if contacts8.any() else 0.0,
        "maximum_chain_discontinuity": float(np.max(adjacent)) if adjacent.size else float("nan"),
    }


def virtual_bond_angles(coords: np.ndarray) -> np.ndarray:
    """Return virtual C-alpha bond angles in degrees."""
    x = np.asarray(coords, dtype=np.float64)
    if x.shape[0] < 3:
        return np.empty(0, dtype=np.float64)
    v1 = x[:-2] - x[1:-1]
    v2 = x[2:] - x[1:-1]
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cosines = np.sum(v1 * v2, axis=1) / np.maximum(denom, 1e-12)
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def virtual_dihedrals(coords: np.ndarray) -> np.ndarray:
    """Return virtual C-alpha dihedrals in degrees.

    Distance matrices do not determine absolute chirality; reflected traces have
    opposite signed dihedral angles but identical distances.
    """
    x = np.asarray(coords, dtype=np.float64)
    if x.shape[0] < 4:
        return np.empty(0, dtype=np.float64)
    out = []
    for p0, p1, p2, p3 in zip(x[:-3], x[1:-2], x[2:-1], x[3:], strict=True):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / max(np.linalg.norm(b1), 1e-12)
        v = b0 - np.dot(b0, b1_norm) * b1_norm
        w = b2 - np.dot(b2, b1_norm) * b1_norm
        x_val = np.dot(v, w)
        y_val = np.dot(np.cross(b1_norm, v), w)
        out.append(np.degrees(np.arctan2(y_val, x_val)))
    return np.asarray(out, dtype=np.float64)


def corrupt_distance_matrix(matrix: np.ndarray, *, noise_angstrom: float, seed: int) -> np.ndarray:
    """Add deterministic symmetric Gaussian distance noise and clamp negatives."""
    d = symmetrize_zero_diagonal(matrix)
    rng = np.random.default_rng(int(seed))
    upper = rng.normal(0.0, float(noise_angstrom), size=d.shape)
    upper = np.triu(upper, k=1)
    noisy = d + upper + upper.T
    noisy = np.maximum(noisy, 0.0)
    np.fill_diagonal(noisy, 0.0)
    return noisy


def kabsch_rmsd(reference: np.ndarray, mobile: np.ndarray, *, allow_reflection: bool = True) -> float:
    """Return optimal rigid-alignment RMSD, optionally reflection invariant."""
    ref = np.asarray(reference, dtype=np.float64) - np.asarray(reference, dtype=np.float64).mean(axis=0, keepdims=True)
    mob = np.asarray(mobile, dtype=np.float64) - np.asarray(mobile, dtype=np.float64).mean(axis=0, keepdims=True)
    covariance = mob.T @ ref
    u, _, vt = np.linalg.svd(covariance)
    candidates = []
    for sign in [1.0, -1.0] if allow_reflection else [1.0]:
        correction = np.eye(3)
        correction[-1, -1] = sign if np.linalg.det(u @ vt) < 0.0 else 1.0
        rotated = mob @ (u @ correction @ vt)
        candidates.append(float(np.sqrt(np.mean(np.sum(np.square(rotated - ref), axis=1)))))
    return min(candidates)


def export_ca_pdb(path: str | Path, coords: np.ndarray, *, sample_id: str, residue_start: int = 1) -> None:
    """Export reconstructed C-alpha-only coordinates as a labelled PDB file."""
    lines = [
        "REMARK Reconstructed C-alpha trace from distance-map rank-3 classical-MDS projection",
        "REMARK Not an experimentally resolved structure and not a full-atom protein",
        f"REMARK sample_id {sample_id}",
    ]
    for index, (x, y, z) in enumerate(np.asarray(coords, dtype=np.float64), start=int(residue_start)):
        serial = index - int(residue_start) + 1
        lines.append(f"ATOM  {serial:5d}  CA  ALA A{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
    lines.append("END")
    atomic_write_text(path, "\n".join(lines) + "\n")


def replace_dataframe(path: str | Path, writer: Any) -> None:
    """Atomically write a dataframe using the provided writer callback."""
    dst = Path(path)
    tmp = atomic_replace_path(dst)
    writer(tmp)
    os.replace(tmp, dst)
