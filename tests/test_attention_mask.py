"""Bottleneck attention mask tests."""

from __future__ import annotations

import torch

from protein_distance_diffusion.models.attention import BottleneckSelfAttention


def test_attention_zeroes_invalid_tokens() -> None:
    """Padded bottleneck tokens are zeroed after attention."""
    attn = BottleneckSelfAttention(8, 2)
    x = torch.randn(1, 8, 4, 4)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    mask[:, :, :2, :2] = True
    out = attn(x, mask)
    assert out.shape == x.shape
    assert out[:, :, 2:, :].abs().sum() == 0
