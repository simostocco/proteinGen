"""Conditioning embeddings for diffusion timestep and protein length."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP.

    Args:
        dim: Output embedding dimension.

    Inputs:
        t: int64 or float tensor with shape [B].

    Outputs:
        float32 tensor with shape [B, dim].
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            t: Timestep tensor [B].

        Returns:
            Embedding tensor [B, dim].
        """
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class LengthEmbedding(nn.Module):
    """Protein-length conditioning embedding.

    Args:
        dim: Output embedding dimension.
        max_length: Maximum configured residue length used in log normalization.

    Inputs:
        lengths: int64 tensor [B].

    Outputs:
        float32 tensor [B, dim].
    """

    def __init__(self, dim: int, *, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        """Embed residue lengths.

        Args:
            lengths: Residue counts [B].

        Returns:
            Embedding tensor [B, dim].
        """
        x = torch.log(lengths.float().clamp_min(1.0)) / math.log(float(self.max_length))
        return self.net(x[:, None])
