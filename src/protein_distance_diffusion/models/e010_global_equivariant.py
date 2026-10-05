"""Global positional equivariant Cartesian residual model for E010 diagnostics."""

from __future__ import annotations

import math

import torch
from torch import nn


def invariant_position_features(mask: torch.Tensor, *, scale: float = 500.0) -> torch.Tensor:
    """Invariant sequence-position features for right-padded variable-length batches."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, padded_length]")
    valid = mask.bool()
    lengths = valid.sum(dim=1, keepdim=True)
    if torch.any(lengths == 0):
        raise ValueError("each batch member must contain at least one valid residue")
    pos = (valid.long().cumsum(dim=1) - 1).clamp_min(0).to(torch.float32)
    denominator = (lengths - 1).clamp_min(1).to(pos.dtype)
    normalized = pos / denominator
    distance_n = pos / scale
    distance_c = (lengths.to(pos.dtype) - 1 - pos).clamp_min(0) / scale
    frequencies = torch.tensor((1, 2, 4, 8, 16, 32, 64, 128), device=pos.device, dtype=pos.dtype)
    phase = 2.0 * math.pi * normalized[..., None] * frequencies
    fourier = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
    features = torch.cat((normalized[..., None], distance_n[..., None], distance_c[..., None], fourier), dim=-1)
    return features * valid[..., None].to(features.dtype)


class _VectorChannelNorm(nn.Module):
    """Normalize vector channels using rotation-invariant scalar statistics."""

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        # [B,N,C,3]. Both channel centering and the aggregate squared norm are equivariant/invariant.
        centered = vector - vector.mean(dim=2, keepdim=True)
        rms = (centered.square().sum(dim=(2, 3), keepdim=True) / vector.shape[2]).clamp_min(1e-8).sqrt()
        return centered / rms * self.scale[None, None, :, None]


class _GlobalEquivariantBlock(nn.Module):
    def __init__(self, width: int, heads: int, vector_channels: int, radial_features: int = 16):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.vector_channels = vector_channels
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.scalar_out = nn.Linear(width, width, bias=False)
        self.radial_bias = nn.Linear(radial_features, heads, bias=False)
        self.sequence_bias = nn.Linear(4, heads, bias=False)
        self.scalar_norm_1 = nn.LayerNorm(width)
        self.scalar_ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))
        self.scalar_norm_2 = nn.LayerNorm(width)
        self.node_vector_gate = nn.Linear(width, vector_channels, bias=False)
        self.radial_vector_gate = nn.Linear(radial_features, vector_channels, bias=False)
        self.sequence_vector_gate = nn.Linear(4, vector_channels, bias=False)
        self.vector_mix = nn.Linear(vector_channels, vector_channels, bias=False)
        self.vector_update_gate = nn.Linear(width, vector_channels)
        self.vector_norm = _VectorChannelNorm(vector_channels)

    def forward(self, scalar, vector, coords, mask, radial, sequence, unit_vectors):
        batch, nodes, _ = scalar.shape
        q = self.query(scalar).view(batch, nodes, self.heads, self.head_width).transpose(1, 2)
        k = self.key(scalar).view(batch, nodes, self.heads, self.head_width).transpose(1, 2)
        v = self.value(scalar).view(batch, nodes, self.heads, self.head_width).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_width)
        logits = logits + self.radial_bias(radial).permute(0, 3, 1, 2)
        logits = logits + self.sequence_bias(sequence).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        scalar_aggregate = torch.matmul(attention, v).transpose(1, 2).reshape(batch, nodes, self.width)
        valid = mask[..., None].to(scalar.dtype)
        scalar = self.scalar_norm_1(scalar + self.scalar_out(scalar_aggregate)) * valid
        scalar = self.scalar_norm_2(scalar + self.scalar_ff(scalar)) * valid

        node_gate = self.node_vector_gate(scalar)
        pair_gate = node_gate[:, :, None, :] + node_gate[:, None, :, :]
        pair_gate = pair_gate + self.radial_vector_gate(radial) + self.sequence_vector_gate(sequence)
        pair_gate = torch.tanh(pair_gate)
        attention_mean = attention.mean(dim=1)
        vector_aggregate = (
            attention_mean[..., None, None] * pair_gate[..., None] * unit_vectors[:, :, :, None, :]
        ).sum(dim=2)
        vector_mixed = self.vector_mix(vector_aggregate.transpose(-1, -2)).transpose(-1, -2)
        gate = torch.sigmoid(self.vector_update_gate(scalar))[..., None]
        vector = self.vector_norm(vector + gate * vector_mixed)
        vector = vector * valid[..., None]
        return scalar, vector


class GlobalEquivariantResidual(nn.Module):
    """All-residue attention model whose Cartesian updates are relative-vector sums."""

    def __init__(
        self,
        width: int = 192,
        layers: int = 4,
        heads: int = 4,
        vector_channels: int = 32,
        max_length: int = 500,
        sigma_distance: float = 2.0,
    ):
        super().__init__()
        if width < 32 or layers < 1 or vector_channels < 4 or max_length < 2 or sigma_distance <= 0:
            raise ValueError("invalid E010 model dimensions or distance scale")
        self.max_length = max_length
        self.sigma_distance = sigma_distance
        self.input = nn.Linear(19, width)
        self.blocks = nn.ModuleList(_GlobalEquivariantBlock(width, heads, vector_channels) for _ in range(layers))
        self.vector_out = nn.Parameter(torch.empty(vector_channels))
        nn.init.normal_(self.vector_out, std=0.02)
        centers = torch.linspace(0.0, 32.0, 16)
        self.register_buffer("radial_centers", centers, persistent=False)

    def _pair_features(self, coords: torch.Tensor, mask: torch.Tensor):
        pair = coords[:, None, :, :] - coords[:, :, None, :]
        distance = torch.linalg.vector_norm(pair, dim=-1).clamp_min(1e-6)
        centers = self.radial_centers.to(device=coords.device, dtype=coords.dtype)
        radial = torch.exp(-((distance[..., None] - centers) / self.sigma_distance).square())

        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pos = (mask.long().cumsum(dim=1) - 1).clamp_min(0).to(coords.dtype)
        separation = (pos[:, :, None] - pos[:, None, :]).abs()
        sequence = torch.stack(
            (
                separation / max(self.max_length - 1, 1),
                torch.log1p(separation) / math.log1p(self.max_length - 1),
                (separation == 1).to(coords.dtype),
                (separation == 2).to(coords.dtype),
            ),
            dim=-1,
        )
        sequence = sequence * (mask[:, :, None] & mask[:, None, :])[..., None]
        unit_vectors = pair / distance[..., None]
        unit_vectors = unit_vectors * (mask[:, :, None] & mask[:, None, :])[..., None]
        # Silence an unused local in static analyzers while documenting the valid-length dependency.
        del lengths
        return radial, sequence, unit_vectors

    def forward(self, coords: torch.Tensor, mask: torch.Tensor):
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape [batch, padded_length, 3]")
        if mask.shape != coords.shape[:2]:
            raise ValueError("mask must have shape [batch, padded_length]")
        if torch.any(mask.bool().sum(dim=1) == 0):
            raise ValueError("each batch member must contain at least one valid residue")
        valid = mask.bool()
        pos_features = invariant_position_features(valid, scale=float(self.max_length)).to(coords.dtype)
        scalar = self.input(pos_features) * valid[..., None].to(coords.dtype)
        radial, sequence, unit_vectors = self._pair_features(coords, valid)
        vector = coords.new_zeros((*coords.shape[:2], self.vector_out.numel(), 3))
        for block in self.blocks:
            scalar, vector = block(scalar, vector, coords, valid, radial, sequence, unit_vectors)
        delta = torch.einsum("c,bncd->bnd", self.vector_out, vector)
        delta = delta * valid[..., None].to(delta.dtype)
        prediction = coords + delta
        return {"prediction": prediction, "delta": delta}
