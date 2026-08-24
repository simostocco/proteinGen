"""Rotary position and continuous length embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class ContinuousLengthEmbedding(nn.Module):
    """Embed requested sequence length as log(N) / log(max_length)."""

    def __init__(self, d_model: int, *, max_length: int) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.net = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        values = torch.log(lengths.float().clamp_min(1.0)) / math.log(float(self.max_length))
        return self.net(values[:, None])


class RotaryEmbedding(nn.Module):
    """RoPE frequencies for causal self-attention heads."""

    def __init__(self, head_dim: int, *, max_length: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q/k tensors shaped [B, heads, L, head_dim]."""
        length = q.shape[-2]
        cos = self.cos[:length][None, None, :, :]
        sin = self.sin[:length][None, None, :, :]
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)
