"""Equal-protein Cartesian and endpoint-masked local losses for E010 Phase 4C."""

from __future__ import annotations

import torch


def phase4c_losses(prediction, source, target, mask, *, local_weight: float = 0.0):
    """Return Å² objectives; lambda=0 preserves the exact historical reduction.

    Each local offset is normalized separately within each protein, then averaged
    over proteins. A protein with no eligible endpoints contributes zero, matching
    the frozen diagnostic's clamped denominator. No alignment is performed.
    """
    if prediction.shape != target.shape or source.shape != target.shape or target.shape[-1] != 3:
        raise ValueError("coordinate tensors must share shape [batch, length, 3]")
    if mask.shape != target.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean [batch, length]")
    if local_weight < 0 or not torch.isfinite(torch.tensor(local_weight)):
        raise ValueError("local weight must be finite and nonnegative")
    if not mask.any(dim=1).all():
        raise ValueError("each protein must have valid residues")
    sq = (prediction.float() - target).square().mean(-1)
    per = (sq * mask).sum(1) / mask.sum(1).clamp_min(1)
    delta = (prediction.float() - source).square().mean(-1)
    reg = (delta * mask).sum(1) / mask.sum(1).clamp_min(1)
    cart = (per + 1e-5 * reg).mean()
    local = []
    for k in (1, 2, 3):
        valid = mask[:, k:] & mask[:, :-k]
        error = (
            torch.linalg.vector_norm(prediction[:, k:] - prediction[:, :-k], dim=-1)
            - torch.linalg.vector_norm(target[:, k:] - target[:, :-k], dim=-1)
        ).square()
        local.append(((error * valid).sum(1) / valid.sum(1).clamp_min(1)).mean())
    mean_local = sum(local) / 3
    # Preserve lambda=0's graph and floating-point result exactly.
    total = cart if local_weight == 0 else cart + local_weight * mean_local
    return {
        "total": total,
        "cartesian": cart,
        "coordinate_mse": per.mean(),
        "residual_mse": reg.mean(),
        "local_1": local[0],
        "local_2": local[1],
        "local_3": local[2],
        "local_mean": mean_local,
    }
