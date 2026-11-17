"""Model-generated detached context; no clean-target argument."""

import torch


@torch.no_grad()
def make_self_condition(model, diffusion, xt, timesteps, lengths, mask, available):
    if not available:
        return torch.zeros_like(xt), torch.zeros(len(xt), dtype=torch.bool, device=xt.device)
    v = model(xt, timesteps, lengths, mask)
    xprev = diffusion.reconstruct_x0(xt, timesteps, v, mask)
    return xprev.detach(), torch.ones(len(xt), dtype=torch.bool, device=xt.device)
