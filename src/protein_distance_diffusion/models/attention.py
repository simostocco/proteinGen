"""Bottleneck self-attention with padding masks."""

from __future__ import annotations

import torch
from torch import nn


class BottleneckSelfAttention(nn.Module):
    """Transformer-style self-attention over spatial bottleneck tokens.

    Args:
        channels: Channel dimension C.
        heads: Number of attention heads.

    Inputs:
        x: Feature map [B, C, H, W].
        valid_mask: Boolean mask [B, 1, H, W] where true tokens may participate.

    Outputs:
        Feature map [B, C, H, W] with invalid positions zeroed.
    """

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4), nn.SiLU(), nn.Linear(channels * 4, channels))

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention.

        Args:
            x: Feature map [B, C, H, W].
            valid_mask: Boolean validity mask [B, 1, H, W].

        Returns:
            Masked attended feature map [B, C, H, W].
        """
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        valid = valid_mask.flatten(1).bool()
        # MultiheadAttention masks invalid keys/values. Invalid queries are zeroed afterwards.
        attended, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), key_padding_mask=~valid)
        tokens = tokens + attended.masked_fill(~valid[:, :, None], 0.0)
        tokens = tokens + self.ff(self.norm2(tokens)).masked_fill(~valid[:, :, None], 0.0)
        out = tokens.transpose(1, 2).reshape(b, c, h, w)
        return out * valid_mask.float()


def downsample_pair_mask(pair_mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Downsample pair masks to bottleneck resolution with nearest-valid pooling.

    Args:
        pair_mask: Boolean pair mask [B, 1, L, L].
        shape: Target `(H, W)`.

    Returns:
        Boolean mask [B, 1, H, W].
    """
    pooled = torch.nn.functional.adaptive_max_pool2d(pair_mask.float(), shape)
    return pooled > 0.5
