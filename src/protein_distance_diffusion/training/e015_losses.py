"""TRAIN-only geometry objectives. No target appears in the inference API."""

import torch
from torch.nn import functional as F

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry

SCALE = 12.22820347644835


def per_protein_losses(prediction, target, mask, tau=0.1, beta_sign=0.1):
    prediction, target = prediction.float(), target.float()
    rows = []
    for p, q, m in zip(prediction, target, mask, strict=True):
        p, q = p[m], q[m]
        if len(p) < 4:
            raise ValueError("Geometry losses require at least four residues")
        distance = []
        for k in (1, 2, 3):
            distance.append(((p[k:] - p[:-k]).norm(dim=-1) - (q[k:] - q[:-k]).norm(dim=-1)).square().mean())
        pf, _, _ = local_geometry(p)
        tf, _, _ = local_geometry(q)
        valid = (pf[:, 29] > 0) & (tf[:, 29] > 0)
        denominator = valid.sum().clamp_min(1)
        torsion = ((pf[:, 25:27] - tf[:, 25:27]).square().mean(-1) * valid).sum() / denominator
        difference = pf[:, 27] - tf[:, 27]
        continuous = F.huber_loss(difference, torch.zeros_like(difference), reduction="none")
        sign = F.softplus(-pf[:, 27] * tf[:, 27] / tau)
        chirality = ((continuous + beta_sign * sign) * valid).sum() / denominator
        ids = torch.arange(len(p), device=p.device)
        pairs = (ids[:, None] - ids[None, :]).abs() > 3
        distances = (p[:, None] - p[None, :]).norm(dim=-1)
        # Frozen project definition: 3 Angstrom, chain separation >3.
        clash = ((3 / SCALE - distances).clamp_min(0).square() * pairs).sum() / pairs.sum().clamp_min(1)
        rows.append(torch.stack([*distance, torsion, chirality, clash]))
    values = torch.stack(rows)
    return {
        "L1": values[:, 0],
        "L2": values[:, 1],
        "L3": values[:, 2],
        "distance": values[:, :3].mean(1),
        "torsion": values[:, 3],
        "chirality": values[:, 4],
        "clash": values[:, 5],
    }


def objectives(vhat, vtarget, xhat, x0, mask, weights, gates, coefficients):
    per_v = ((vhat.float() - vtarget.float()).square() * mask[..., None]).sum((1, 2)) / (3 * mask.sum(1))
    losses = per_protein_losses(xhat, x0, mask)
    structural = sum(
        (losses[name] * gates[name] * coefficients[name]).mean()
        for name in ("distance", "torsion", "chirality", "clash")
    )
    return (per_v * weights).mean(), structural, losses


def global_scale_regularizer(prediction, target, mask):
    """Optional future equal-protein log-scale penalty; disabled in v1."""
    weights = mask[..., None].float()
    p = (prediction.float().square() * weights).sum((1, 2)) / mask.sum(1)
    q = (target.float().square() * weights).sum((1, 2)) / mask.sum(1)
    return (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log()).square().mean()
