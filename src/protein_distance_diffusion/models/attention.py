"""Bottleneck self-attention with padding masks."""

from __future__ import annotations

import torch
from torch import nn

from protein_distance_diffusion.models.blocks import valid_group_count


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


class SymmetricAxialAttentionBlock(nn.Module):
    """Transpose-equivariant axial attention block for square distance-map features.

    The same multi-head attention module is applied across rows and columns, so
    the row/column projections are shared. Attention is axial: it never builds a
    full ``[B, heads, H*W, H*W]`` attention matrix.
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        *,
        heads: int,
        groups: int,
        dropout: float,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by axial attention heads")
        self.channels = int(channels)
        self.heads = int(heads)
        self.chunk_size = None if chunk_size is None or chunk_size <= 0 else int(chunk_size)
        self.norm1 = nn.GroupNorm(valid_group_count(channels, groups), channels)
        self.cond = nn.Linear(cond_dim, channels)
        self.token_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.GroupNorm(valid_group_count(channels, groups), channels)
        self.ff = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels * 4, channels, kernel_size=1),
        )

    @staticmethod
    def _masked_group_norm(norm: nn.GroupNorm, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply GroupNorm using only valid spatial positions in each sample."""
        b, c, h, w = x.shape
        groups = int(norm.num_groups)
        channels_per_group = c // groups
        x_grouped = x.reshape(b, groups, channels_per_group, h, w)
        mask = valid_mask[:, :, None].to(dtype=x.dtype)
        count = (mask.sum(dim=(2, 3, 4), keepdim=True) * channels_per_group).clamp_min(1.0)
        mean = (x_grouped * mask).sum(dim=(2, 3, 4), keepdim=True) / count
        variance = ((x_grouped - mean).square() * mask).sum(dim=(2, 3, 4), keepdim=True) / count
        normalized = (x_grouped - mean) * torch.rsqrt(variance + norm.eps)
        normalized = normalized.reshape(b, c, h, w)
        if norm.affine:
            normalized = normalized * norm.weight[None, :, None, None] + norm.bias[None, :, None, None]
        return normalized * valid_mask.to(dtype=x.dtype)

    def _attend_axis(self, x: torch.Tensor, valid_mask: torch.Tensor, *, axis: str) -> torch.Tensor:
        """Apply shared attention along one spatial axis."""
        b, c, h, w = x.shape
        if axis == "row":
            tokens = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
            valid = valid_mask[:, 0].reshape(b * h, w).bool()
            shape = (b, h, w, c)
            restore = "row"
        elif axis == "column":
            tokens = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
            valid = valid_mask[:, 0].transpose(1, 2).reshape(b * w, h).bool()
            shape = (b, w, h, c)
            restore = "column"
        else:
            raise ValueError("axis must be 'row' or 'column'")

        out = torch.zeros_like(tokens)
        active = valid.any(dim=1)
        if active.any():
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            chunks = active_indices.split(self.chunk_size or len(active_indices))
            for indices in chunks:
                chunk_tokens = self.token_norm(tokens.index_select(0, indices))
                chunk_valid = valid.index_select(0, indices)
                attended, _ = self.attn(
                    chunk_tokens,
                    chunk_tokens,
                    chunk_tokens,
                    key_padding_mask=~chunk_valid,
                    need_weights=False,
                )
                attended = attended.masked_fill(~chunk_valid[:, :, None], 0.0)
                attended = attended.to(device=out.device, dtype=out.dtype)
                out.index_copy_(0, indices, attended)
        if restore == "row":
            return out.reshape(shape).permute(0, 3, 1, 2)
        return out.reshape(shape).permute(0, 3, 2, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Apply conditioned row and column attention to ``x``.

        Args:
            x: Feature map ``[B, C, H, W]``.
            cond: Time-plus-length conditioning ``[B, cond_dim]``.
            valid_mask: Boolean mask ``[B, 1, H, W]``.

        Returns:
            Feature map with padded positions forced to zero.
        """
        valid = valid_mask.bool()
        h = self._masked_group_norm(self.norm1, x, valid)
        h = h + self.cond(cond)[:, :, None, None]
        h = h * valid.to(dtype=h.dtype)
        out = x + self._attend_axis(h, valid, axis="row")
        out = out * valid.to(dtype=out.dtype)
        out = out + self._attend_axis(self._masked_group_norm(self.norm1, out, valid), valid, axis="column")
        out = out * valid.to(dtype=out.dtype)
        out = out + self.ff(torch.nn.functional.silu(self._masked_group_norm(self.norm2, out, valid))).masked_fill(
            ~valid, 0.0
        )
        if out.shape[-1] == out.shape[-2]:
            out = 0.5 * (out + out.transpose(-1, -2))
        return out * valid.to(dtype=out.dtype)
