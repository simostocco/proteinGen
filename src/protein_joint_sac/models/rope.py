"""Ordered neural feature-pair RoPE, unrelated to Cartesian 3D rotations."""

import torch


def apply_rope(x, positions, base=10000.0):
    if x.shape[-1] % 2:
        raise ValueError("even RoPE head dimension required")
    inv = base ** (-torch.arange(0, x.shape[-1], 2, dtype=x.dtype, device=x.device) / x.shape[-1])
    ang = positions.to(x.dtype)[:, None] * inv
    c = ang.cos()
    s = ang.sin()
    a = x[..., ::2]
    b = x[..., 1::2]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)
