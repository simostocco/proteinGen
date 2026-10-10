"""TRAIN-only on-policy states with mathematically consistent implied targets."""

import torch

from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates


@torch.no_grad()
def induced_state(ema, diffusion, xt, t, x0, lengths, mask, steps):
    if steps not in (1, 2, 3) or (t < steps + 1).any():
        raise ValueError("Induced states require 1-3 nonterminal steps")
    xprev = None
    for _ in range(steps):
        available = torch.full((len(xt),), xprev is not None, dtype=torch.bool, device=xt.device)
        v = ema(xt, t, lengths, mask, xprev, available)
        xt, xprev, _ = diffusion.deterministic_reverse_step(xt, t, v, mask)
        t = t - 1
    alpha, sigma = diffusion.alpha_sigma(t, xt)
    epsilon = center_coordinates((xt - alpha * x0) / sigma, mask)
    target = center_coordinates(alpha * epsilon - sigma * x0, mask)
    return xt.detach(), t, target.detach()
