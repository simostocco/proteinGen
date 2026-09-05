"""Conditional 2D U-Net for protein distance-map diffusion."""

from __future__ import annotations

import torch
from torch import nn

from protein_distance_diffusion.diffusion.gaussian import project_symmetric_zero_diagonal
from protein_distance_diffusion.models.attention import (
    BottleneckSelfAttention,
    SymmetricAxialAttentionBlock,
    SymmetricTriangleMultiplicativeUpdate,
    downsample_pair_mask,
)
from protein_distance_diffusion.models.blocks import Downsample, ResidualBlock, Upsample
from protein_distance_diffusion.models.embeddings import LengthEmbedding, SinusoidalTimeEmbedding


class DistanceUNet(nn.Module):
    """Matrix-size-independent 2D U-Net with timestep and length conditioning.

    Args:
        input_channels: Input channels: noisy matrix, sequence separation, pair mask.
        output_channels: Output channels, normally 1 for epsilon prediction.
        base_channels: First hidden channel count.
        channel_multipliers: Per-level channel multipliers.
        residual_blocks_per_level: Number of residual blocks at each level.
        dropout: Residual-block dropout probability.
        group_norm_groups: Preferred GroupNorm groups.
        attention_heads: Bottleneck attention heads.
        use_bottleneck_attention: Whether to apply masked self-attention at the bottleneck.
        use_pre_bottleneck_axial_attention: Replace the last residual block one encoder level above
            the bottleneck with symmetry-preserving axial attention.
        axial_attention_heads: Number of axial attention heads. Defaults to `attention_heads`.
        axial_attention_dropout: Axial attention and feed-forward dropout probability.
        axial_attention_chunk_size: Optional number of row/column sequences per attention chunk.
        use_pre_bottleneck_triangle_multiplication: Add one symmetric triangle-multiplicative
            update after pre-bottleneck axial attention and before bottleneck downsampling.
        triangle_hidden_channels: Hidden width for triangle path-composition factors.
        triangle_dropout: Triangle output dropout probability.
        triangle_chunk_size: Number of hidden triangle channels per contraction chunk.
        time_embedding_dim: Conditioning dimension for timesteps.
        length_embedding_dim: Conditioning dimension for lengths. Must match time embedding.
        max_length: Maximum configured protein length for length normalization.
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        output_channels: int = 1,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        residual_blocks_per_level: int = 2,
        dropout: float = 0.0,
        group_norm_groups: int = 8,
        attention_heads: int = 8,
        use_bottleneck_attention: bool = True,
        use_pre_bottleneck_axial_attention: bool = False,
        axial_attention_heads: int | None = None,
        axial_attention_dropout: float | None = None,
        axial_attention_chunk_size: int | None = None,
        use_pre_bottleneck_triangle_multiplication: bool = False,
        triangle_hidden_channels: int = 32,
        triangle_dropout: float = 0.0,
        triangle_chunk_size: int | None = 16,
        time_embedding_dim: int = 256,
        length_embedding_dim: int = 256,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        if time_embedding_dim != length_embedding_dim:
            raise ValueError("time_embedding_dim and length_embedding_dim must match")
        if use_pre_bottleneck_axial_attention and len(channel_multipliers) < 2:
            raise ValueError("use_pre_bottleneck_axial_attention requires at least two U-Net levels")
        if use_pre_bottleneck_axial_attention and residual_blocks_per_level < 2:
            raise ValueError("use_pre_bottleneck_axial_attention requires residual_blocks_per_level >= 2")
        if use_pre_bottleneck_triangle_multiplication and not use_pre_bottleneck_axial_attention:
            raise ValueError("use_pre_bottleneck_triangle_multiplication requires pre-bottleneck axial attention")
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        self.use_pre_bottleneck_axial_attention = bool(use_pre_bottleneck_axial_attention)
        self.use_pre_bottleneck_triangle_multiplication = bool(use_pre_bottleneck_triangle_multiplication)
        self.pre_bottleneck_axial_level = len(channel_multipliers) - 2
        self.pre_bottleneck_axial_block_index = residual_blocks_per_level - 1
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.length_embedding = LengthEmbedding(length_embedding_dim, max_length=max_length)
        cond_dim = time_embedding_dim
        channels = [base_channels * mult for mult in channel_multipliers]
        self.input = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.pre_bottleneck_triangle = nn.ModuleDict()
        in_ch = channels[0]
        self.skip_channels: list[int] = []
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for block_index in range(residual_blocks_per_level):
                use_axial_here = (
                    self.use_pre_bottleneck_axial_attention
                    and level == self.pre_bottleneck_axial_level
                    and block_index == self.pre_bottleneck_axial_block_index
                    and level >= 0
                    and residual_blocks_per_level >= 2
                    and in_ch == out_ch
                )
                if use_axial_here:
                    blocks.append(
                        SymmetricAxialAttentionBlock(
                            out_ch,
                            cond_dim,
                            heads=axial_attention_heads or attention_heads,
                            groups=group_norm_groups,
                            dropout=dropout if axial_attention_dropout is None else float(axial_attention_dropout),
                            chunk_size=axial_attention_chunk_size,
                        )
                    )
                else:
                    blocks.append(ResidualBlock(in_ch, out_ch, cond_dim, groups=group_norm_groups, dropout=dropout))
                in_ch = out_ch
                self.skip_channels.append(out_ch)
            self.down_blocks.append(blocks)
            if self.use_pre_bottleneck_triangle_multiplication and level == self.pre_bottleneck_axial_level:
                self.pre_bottleneck_triangle[str(level)] = SymmetricTriangleMultiplicativeUpdate(
                    in_ch,
                    hidden_channels=triangle_hidden_channels,
                    groups=group_norm_groups,
                    dropout=triangle_dropout,
                    chunk_size=triangle_chunk_size,
                )
            if level != len(channels) - 1:
                self.downsamples.append(Downsample(in_ch))
        self.mid1 = ResidualBlock(in_ch, in_ch, cond_dim, groups=group_norm_groups, dropout=dropout)
        self.attention = BottleneckSelfAttention(in_ch, attention_heads) if use_bottleneck_attention else None
        self.mid2 = ResidualBlock(in_ch, in_ch, cond_dim, groups=group_norm_groups, dropout=dropout)
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level, out_ch in reversed(list(enumerate(channels))):
            blocks = nn.ModuleList()
            for _ in range(residual_blocks_per_level):
                skip_ch = self.skip_channels.pop()
                blocks.append(
                    ResidualBlock(in_ch + skip_ch, out_ch, cond_dim, groups=group_norm_groups, dropout=dropout)
                )
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(Upsample(in_ch))
        self.output = nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_distance: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        sequence_separation: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict symmetric epsilon.

        Args:
            noisy_distance: Noisy normalized matrices [B, 1, L, L].
            timesteps: Diffusion timesteps [B].
            lengths: Residue lengths [B].
            sequence_separation: Sequence-separation channel [B, 1, L, L].
            pair_mask: Boolean pair mask [B, 1, L, L].

        Returns:
            Epsilon prediction [B, 1, L, L], symmetrized, zero-diagonal, and masked.
        """
        if noisy_distance.shape[-1] % self.downsample_factor != 0:
            raise ValueError(f"Padded side {noisy_distance.shape[-1]} must be divisible by {self.downsample_factor}")
        cond = self.time_embedding(timesteps) + self.length_embedding(lengths)
        x = self.input(torch.cat([noisy_distance, sequence_separation, pair_mask.float()], dim=1))
        skips: list[torch.Tensor] = []
        downsample_idx = 0
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, SymmetricAxialAttentionBlock):
                    x = block(x, cond, downsample_pair_mask(pair_mask, x.shape[-2:]))
                else:
                    x = block(x, cond)
                skips.append(x)
            if str(level) in self.pre_bottleneck_triangle:
                x = self.pre_bottleneck_triangle[str(level)](x, downsample_pair_mask(pair_mask, x.shape[-2:]))
                skips[-1] = x
            if level != len(self.down_blocks) - 1:
                x = self.downsamples[downsample_idx](x)
                downsample_idx += 1
        x = self.mid1(x, cond)
        if self.attention is not None:
            x = self.attention(x, downsample_pair_mask(pair_mask, x.shape[-2:]))
        x = self.mid2(x, cond)
        upsample_idx = 0
        for level, blocks in enumerate(self.up_blocks):
            for block in blocks:
                skip = skips.pop()
                if skip.shape[-2:] != x.shape[-2:]:
                    x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
                x = block(torch.cat([x, skip], dim=1), cond)
            if level != len(self.up_blocks) - 1:
                x = self.upsamples[upsample_idx](x)
                upsample_idx += 1
        return project_symmetric_zero_diagonal(self.output(x), pair_mask)
