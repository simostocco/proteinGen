"""Equal-protein v loss; geometry auxiliaries remain disabled by default."""

import torch


def coordinate_v_loss(prediction, target, mask):
    per_protein = ((prediction - target).square() * mask[..., None]).sum((1, 2)) / (3 * mask.sum(1))
    return per_protein.mean()


def local_distance_loss(prediction, target, mask):
    losses = []
    for k in (1, 2, 3):
        valid = mask[:, k:] & mask[:, :-k]
        error = (
            (prediction[:, k:] - prediction[:, :-k]).norm(dim=-1) - (target[:, k:] - target[:, :-k]).norm(dim=-1)
        ).square()
        losses.append(((error * valid).sum(1) / valid.sum(1).clamp_min(1)).mean())
    return torch.stack(losses).mean()


def low_noise_geometry_auxiliaries(prediction, target, mask, timesteps, diffusion_steps=500):
    """Future extension only: no coefficients or activation in the v-only pilot.

    Return equal-protein low-noise-weighted local, torsion and steric terms.
    Local reference geometry is permitted only in this training loss function.
    Degenerate torsions do not receive fabricated supervision.
    """
    from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry

    terms = []
    for z in range(len(prediction)):
        p, q = prediction[z, mask[z]], target[z, mask[z]]
        pm = torch.ones(1, len(p), dtype=torch.bool, device=p.device)
        local = local_distance_loss(p[None], q[None], pm)
        pf, _, _ = local_geometry(p)
        tf, _, _ = local_geometry(q)
        valid = (pf[:, 29] > 0) & (tf[:, 29] > 0)
        chiral = ((pf[:, 25:28] - tf[:, 25:28]).square().sum(-1) * valid).sum() / valid.sum().clamp_min(1)
        # Future loss assumes normalized coordinates; 3 A / frozen scale.
        ids = torch.arange(len(p), device=p.device)
        pairs = (ids[:, None] - ids[None]).abs() > 3
        distances = torch.linalg.vector_norm(p[:, None] - p[None], dim=-1)
        clash = (((3 / 12.22820347644835 - distances).clamp_min(0).square()) * pairs).sum() / pairs.sum().clamp_min(1)
        weight = (1 - timesteps[z].to(p.dtype) / (diffusion_steps - 1)).clamp(0, 1).square()
        terms.append(weight * torch.stack([local, chiral, clash]))
    values = torch.stack(terms).mean(0)
    return dict(local=values[0], chiral=values[1], clash=values[2])
