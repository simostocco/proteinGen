"""Symmetric diffusion tests."""

from __future__ import annotations

import torch

from protein_distance_diffusion.data.collate import make_pair_mask
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    masked_upper_triangular_loss,
    per_protein_masked_upper_triangular_loss,
    sample_symmetric_noise,
)
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule


def test_symmetric_noise_zero_diagonal_and_mask() -> None:
    """Noise is mirrored, diagonal-free, and zero outside padding."""
    lengths = torch.tensor([4, 2])
    mask = make_pair_mask(lengths, 4)
    noise = sample_symmetric_noise((2, 1, 4, 4), mask, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(noise, noise.transpose(-1, -2))
    assert torch.allclose(torch.diagonal(noise, dim1=-1, dim2=-2), torch.zeros(2, 1, 4))
    assert noise[1, 0, 2:, :].abs().sum() == 0


def test_forward_noising_and_loss_shapes() -> None:
    """Forward noising preserves representation constraints and loss is scalar."""
    lengths = torch.tensor([4])
    mask = make_pair_mask(lengths, 4)
    diffusion = GaussianDiffusion(cosine_beta_schedule(8))
    x = torch.ones(1, 1, 4, 4) * mask.float()
    noisy, eps = diffusion.q_sample(x, torch.tensor([3]), mask, generator=torch.Generator().manual_seed(2))
    assert noisy.shape == x.shape
    assert torch.allclose(noisy, noisy.transpose(-1, -2))
    assert torch.diagonal(noisy, dim1=-1, dim2=-2).abs().sum() == 0
    assert masked_upper_triangular_loss(eps, eps, mask).item() == 0.0


def test_weighted_loss_combines_per_protein_losses_not_cells() -> None:
    """Sample weights combine already-normalized per-protein losses."""
    lengths = torch.tensor([3, 3])
    mask = make_pair_mask(lengths, 3)
    target = torch.zeros(2, 1, 3, 3)
    prediction = torch.zeros(2, 1, 3, 3)
    prediction[0, 0, 0, 1] = 1.0
    prediction[0, 0, 1, 0] = 1.0
    prediction[1, 0, 0, 1] = 3.0
    prediction[1, 0, 1, 0] = 3.0
    per_protein = per_protein_masked_upper_triangular_loss(target, prediction, mask)
    assert torch.allclose(per_protein, torch.tensor([1.0 / 3.0, 3.0]))
    weighted = masked_upper_triangular_loss(target, prediction, mask, sample_weight=torch.tensor([3.0, 1.0]))
    assert weighted.item() == torch.tensor((3.0 * (1.0 / 3.0) + 3.0) / 4.0).item()
