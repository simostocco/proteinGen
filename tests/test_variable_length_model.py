"""Variable-length model and batching invariants."""

from __future__ import annotations

import torch

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.data.samplers import MatrixBudgetBatchSampler
from protein_distance_diffusion.diffusion.gaussian import masked_upper_triangular_loss
from protein_distance_diffusion.models.unet import DistanceUNet


def test_matrix_budget_sampler_visits_every_sample_once_without_length_classes() -> None:
    """Matrix-budget batching changes padding efficiency, not sampling probability."""
    lengths = [17, 129, 287, 500]
    sampler = MatrixBudgetBatchSampler(lengths, batch_matrix_budget=500 * 500 + 17 * 17, shuffle=False)
    flat = [idx for batch in sampler for idx in batch]
    assert sorted(flat) == [0, 1, 2, 3]


def test_lengths_17_129_287_500_use_same_model_and_masked_objective() -> None:
    """Mixed protein lengths pass through one model branch and one per-sample normalized loss."""
    lengths = torch.tensor([17, 129, 287, 500], dtype=torch.long)
    side = 500
    pair_mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    noisy = torch.randn((len(lengths), 1, side, side)) * pair_mask.float()
    target = torch.randn_like(noisy) * pair_mask.float()
    timesteps = torch.tensor([0, 10, 100, 250], dtype=torch.long)
    model = DistanceUNet(
        base_channels=1,
        channel_multipliers=(1,),
        residual_blocks_per_level=1,
        group_norm_groups=1,
        attention_heads=1,
        use_bottleneck_attention=False,
        time_embedding_dim=8,
        length_embedding_dim=8,
        max_length=500,
    )
    prediction = model(noisy, timesteps, lengths, sep, pair_mask)
    loss = masked_upper_triangular_loss(target, prediction, pair_mask)
    assert prediction.shape == noisy.shape
    assert torch.isfinite(loss)
    assert model.length_embedding.max_length == 500
    assert torch.allclose(sep[0, 0, 0, 16], torch.tensor(1.0))
    assert torch.allclose(sep[3, 0, 0, 499], torch.tensor(1.0))
