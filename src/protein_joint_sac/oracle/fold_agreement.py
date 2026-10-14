"""Completed-state fixed-correspondence comparisons, never online actor features."""

import torch
from ..contracts import GeneratedProtein, OracleResult, OracleStatus


def compare_completed(protein, result, mask=None):
    if (
        type(protein) != GeneratedProtein
        or type(result) != OracleResult
        or result.status != OracleStatus.SUCCEEDED
        or result.sequence_sha256 != protein.sequence_sha256
    ):
        raise ValueError("completed evaluated matching sequence required")
    if result.predicted_ca is None:
        raise ValueError("missing predictor coordinate mapping")
    X = protein.state.ca.detach()
    Y = torch.tensor(result.predicted_ca, dtype=torch.float64)
    if Y.shape != X.shape or not torch.isfinite(Y).all():
        raise ValueError("full ordered position correspondence required; do not align cropped sequences silently")
    valid = torch.ones(protein.state.N, dtype=torch.bool) if mask is None else mask
    if valid.shape != (protein.state.N,) or valid.dtype != torch.bool or valid.sum() < 3:
        raise ValueError("aligned validity and at least3 coordinates")
    x = X[valid]
    y = Y[valid]
    xc = x - x.mean(0)
    yc = y - y.mean(0)
    u, _, vh = torch.linalg.svd(xc.T @ yc)
    D = torch.eye(3, dtype=x.dtype)
    D[-1, -1] = torch.det(u @ vh)
    rot = u @ D @ vh
    err = torch.linalg.vector_norm(xc @ rot - yc, dim=-1)
    N = protein.state.N
    d0 = max(0.5, 1.24 * (N - 15) ** (1 / 3) - 1.8)
    dx = torch.cdist(x, x)
    dy = torch.cdist(y, y)
    positions = torch.arange(N)[valid]
    pair = torch.triu(torch.ones((len(x), len(x)), dtype=torch.bool), diagonal=1) & (
        (positions[:, None] - positions[None, :]).abs() >= 12
    )
    ctX = dx <= 8
    ctY = dy <= 8
    union = ((ctX | ctY) & pair).sum()
    return {
        "full_length_denominator": N,
        "coordinate_coverage": float(valid.sum() / N),
        "fixed_mapping_Kabsch_RMSD_A": float(err.square().mean().sqrt()),
        "fixed_mapping_Kabsch_TM_similarity": float((1 / (1 + (err / d0).square())).sum() / N),
        "optimized_TM_score": None,
        "TM_limit": "fixed-correspondence Kabsch similarity, not an optimized TM-align/US-align score",
        "long_range_distance_RMSE_A": float((dx[pair] - dy[pair]).square().mean().sqrt()) if pair.any() else None,
        "long_range_contact_Jaccard": float(((ctX & ctY) & pair).sum() / union) if union else None,
        "candidate_geometry_sha256": protein.geometry_sha256,
        "oracle_output_sha256": result.predictor_output_sha256,
        "usage_role": "COMPLETED_EVALUATION_ONLY",
        "evidence_type": result.evidence_type,
    }
