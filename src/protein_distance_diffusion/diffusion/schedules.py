"""Noise schedules for DDPM diffusion."""

from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
    """Create a cosine DDPM beta schedule.

    Args:
        timesteps: Number of diffusion steps T.
        s: Small offset from Nichol and Dhariwal's cosine schedule.

    Returns:
        float32 tensor with shape [T].
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0001, 0.9999).float()
