"""U-Net shape tests."""

from __future__ import annotations

import torch

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.models.unet import DistanceUNet


def tiny_model() -> DistanceUNet:
    """Return a small model for CPU tests."""
    return DistanceUNet(
        base_channels=8,
        channel_multipliers=(1, 2),
        residual_blocks_per_level=1,
        attention_heads=2,
        time_embedding_dim=32,
        length_embedding_dim=32,
        max_length=32,
    )


def test_unet_multiple_sizes_symmetric_output_and_gradients() -> None:
    """One model instance runs on several padded matrix sizes."""
    model = tiny_model()
    for side in (8, 12, 16):
        lengths = torch.tensor([side - 1])
        mask = make_pair_mask(lengths, side)
        sep = make_sequence_separation(lengths, side)
        x = torch.randn(1, 1, side, side, requires_grad=True)
        out = model(x, torch.tensor([1]), lengths, sep, mask)
        assert out.shape == x.shape
        assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)
        assert torch.diagonal(out, dim1=-1, dim2=-2).abs().sum() == 0
        out.square().mean().backward()
