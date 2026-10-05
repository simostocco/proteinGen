"""Reusable conditioned pair-grid U-Net trunk for invariant scalar features."""

from __future__ import annotations

import torch
from torch import nn

from protein_distance_diffusion.models.attention import (
    BottleneckSelfAttention,
    SymmetricAxialAttentionBlock,
    SymmetricTriangleMultiplicativeUpdate,
    downsample_pair_mask,
)
from protein_distance_diffusion.models.blocks import Downsample, ResidualBlock, Upsample
from protein_distance_diffusion.models.embeddings import LengthEmbedding, SinusoidalTimeEmbedding


class PairGridUNetTrunk(nn.Module):
    """E004-shaped U-Net mapping invariant pair channels to scalar coefficients.

    This is intentionally separate from :class:`DistanceUNet`: E004 module names,
    state dictionaries, and checkpoint loading remain unchanged.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        base_channels: int = 24,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        residual_blocks_per_level: int = 2,
        dropout: float = 0.0,
        group_norm_groups: int = 8,
        attention_heads: int = 4,
        use_bottleneck_attention: bool = True,
        use_pre_bottleneck_axial_attention: bool = True,
        axial_attention_heads: int = 4,
        axial_attention_dropout: float = 0.0,
        axial_attention_chunk_size: int | None = 128,
        use_pre_bottleneck_triangle_multiplication: bool = True,
        triangle_hidden_channels: int = 32,
        triangle_dropout: float = 0.0,
        triangle_chunk_size: int | None = 16,
        time_embedding_dim: int = 256,
        length_embedding_dim: int = 256,
        max_length: int = 500,
    ) -> None:
        super().__init__()
        if time_embedding_dim != length_embedding_dim:
            raise ValueError("time_embedding_dim and length_embedding_dim must match")
        if len(channel_multipliers) < 1:
            raise ValueError("channel_multipliers must not be empty")
        if use_pre_bottleneck_axial_attention and len(channel_multipliers) < 2:
            raise ValueError("pre-bottleneck axial attention requires at least two levels")
        if use_pre_bottleneck_axial_attention and residual_blocks_per_level < 2:
            raise ValueError("pre-bottleneck axial attention requires at least two residual blocks")
        if use_pre_bottleneck_triangle_multiplication and not use_pre_bottleneck_axial_attention:
            raise ValueError("triangle multiplication requires pre-bottleneck axial attention")

        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.length_embedding = LengthEmbedding(length_embedding_dim, max_length=max_length)
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.input = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.pre_bottleneck_triangle = nn.ModuleDict()
        pre_level = len(channels) - 2
        pre_block = residual_blocks_per_level - 1
        in_channels = channels[0]
        skip_channels: list[int] = []
        for level, out_channels in enumerate(channels):
            blocks = nn.ModuleList()
            for block_index in range(residual_blocks_per_level):
                axial = (
                    use_pre_bottleneck_axial_attention
                    and level == pre_level
                    and block_index == pre_block
                    and in_channels == out_channels
                )
                if axial:
                    blocks.append(
                        SymmetricAxialAttentionBlock(
                            out_channels,
                            time_embedding_dim,
                            heads=axial_attention_heads,
                            groups=group_norm_groups,
                            dropout=axial_attention_dropout,
                            chunk_size=axial_attention_chunk_size,
                        )
                    )
                else:
                    blocks.append(
                        ResidualBlock(
                            in_channels,
                            out_channels,
                            time_embedding_dim,
                            groups=group_norm_groups,
                            dropout=dropout,
                        )
                    )
                in_channels = out_channels
                skip_channels.append(out_channels)
            self.down_blocks.append(blocks)
            if use_pre_bottleneck_triangle_multiplication and level == pre_level:
                self.pre_bottleneck_triangle[str(level)] = SymmetricTriangleMultiplicativeUpdate(
                    in_channels,
                    hidden_channels=triangle_hidden_channels,
                    groups=group_norm_groups,
                    dropout=triangle_dropout,
                    chunk_size=triangle_chunk_size,
                )
            if level != len(channels) - 1:
                self.downsamples.append(Downsample(in_channels))

        self.mid1 = ResidualBlock(
            in_channels, in_channels, time_embedding_dim, groups=group_norm_groups, dropout=dropout
        )
        self.attention = BottleneckSelfAttention(in_channels, attention_heads) if use_bottleneck_attention else None
        self.mid2 = ResidualBlock(
            in_channels, in_channels, time_embedding_dim, groups=group_norm_groups, dropout=dropout
        )
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level, out_channels in reversed(list(enumerate(channels))):
            blocks = nn.ModuleList()
            for _ in range(residual_blocks_per_level):
                skip = skip_channels.pop()
                blocks.append(
                    ResidualBlock(
                        in_channels + skip,
                        out_channels,
                        time_embedding_dim,
                        groups=group_norm_groups,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(in_channels))
        self.coefficient_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        pair_features: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return unconstrained scalar pair coefficients with shape ``[B,1,N,N]``."""
        if pair_features.shape[-1] % self.downsample_factor:
            raise ValueError("pair-grid side must be divisible by the U-Net downsampling factor")
        condition = self.time_embedding(timesteps) + self.length_embedding(lengths)
        x = self.input(pair_features)
        skips: list[torch.Tensor] = []
        downsample_index = 0
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, SymmetricAxialAttentionBlock):
                    x = block(x, condition, downsample_pair_mask(pair_mask, x.shape[-2:]))
                else:
                    x = block(x, condition)
                skips.append(x)
            if str(level) in self.pre_bottleneck_triangle:
                x = self.pre_bottleneck_triangle[str(level)](x, downsample_pair_mask(pair_mask, x.shape[-2:]))
                skips[-1] = x
            if level != len(self.down_blocks) - 1:
                x = self.downsamples[downsample_index](x)
                downsample_index += 1
        x = self.mid1(x, condition)
        if self.attention is not None:
            x = self.attention(x, downsample_pair_mask(pair_mask, x.shape[-2:]))
        x = self.mid2(x, condition)
        upsample_index = 0
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = skips.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = block(torch.cat((x, skip), dim=1), condition)
            if level != len(self.up_blocks) - 1:
                x = self.upsamples[upsample_index](x)
                upsample_index += 1
        return self.coefficient_head(x)
