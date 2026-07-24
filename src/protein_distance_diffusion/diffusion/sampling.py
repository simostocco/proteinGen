"""Reverse diffusion sampling for generated distance matrices."""

from __future__ import annotations

import torch

from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    project_symmetric_zero_diagonal,
)


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    *,
    lengths: torch.Tensor,
    pair_mask: torch.Tensor,
    sequence_separation: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate normalized distance matrices with ancestral DDPM sampling.

    Args:
        model: Epsilon-prediction model.
        diffusion: Gaussian diffusion schedule.
        lengths: Residue lengths [B].
        pair_mask: Valid pair masks [B, 1, L, L].
        sequence_separation: Sequence separation channel [B, 1, L, L].
        device: Target device.
        generator: Optional random generator.

    Returns:
        Normalized generated matrices [B, 1, L, L].
    """
    x = torch.randn(pair_mask.shape, generator=generator, device=device)
    x = project_symmetric_zero_diagonal(x, pair_mask.to(device))
    diffusion.to(device)
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((x.shape[0],), step, dtype=torch.long, device=device)
        eps = model(x, t, lengths.to(device), sequence_separation.to(device), pair_mask.to(device))
        beta = diffusion.betas[step]
        alpha = diffusion.alphas[step]
        alpha_bar = diffusion.alphas_cumprod[step]
        mean = (x - beta / (1.0 - alpha_bar).sqrt() * eps) / alpha.sqrt()
        if step > 0:
            noise = torch.randn(x.shape, generator=generator, device=device)
            x = mean + beta.sqrt() * project_symmetric_zero_diagonal(noise, pair_mask.to(device))
        else:
            x = mean
        x = project_symmetric_zero_diagonal(x, pair_mask.to(device))
    return x
