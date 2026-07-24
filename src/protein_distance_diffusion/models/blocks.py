"""U-Net convolutional building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


def valid_group_count(channels: int, requested: int) -> int:
    """Return a GroupNorm group count that divides channels.

    Args:
        channels: Channel count.
        requested: Preferred group count.

    Returns:
        Divisor of channels.
    """
    return math.gcd(channels, requested)


class ResidualBlock(nn.Module):
    """Conditioned residual block for 2D distance maps.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        cond_dim: Conditioning embedding dimension [B, cond_dim].
        groups: Preferred GroupNorm groups.
        dropout: Dropout probability.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, *, groups: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(valid_group_count(in_channels, groups), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels)
        self.norm2 = nn.GroupNorm(valid_group_count(out_channels, groups), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Feature map [B, C, H, W].
            cond: Conditioning embedding [B, cond_dim].

        Returns:
            Feature map [B, out_channels, H, W].
        """
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    """Stride-2 convolutional downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample [B, C, H, W] to [B, C, H/2, W/2]."""
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest-neighbor upsampling followed by a convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample [B, C, H, W] to [B, C, 2H, 2W]."""
        return self.conv(torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"))
