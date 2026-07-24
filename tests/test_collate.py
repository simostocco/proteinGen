"""Batch collation tests."""

from __future__ import annotations

import torch

from protein_distance_diffusion.data.collate import collate_distance_maps


def test_mixed_lengths_collate_masks_and_separation() -> None:
    """Mixed lengths are padded, masked, and receive computed separation maps."""
    items = [
        {"sample_id": "a", "length": 3, "distance_matrix": torch.ones(3, 3)},
        {"sample_id": "b", "length": 5, "distance_matrix": torch.ones(5, 5) * 2},
    ]
    batch = collate_distance_maps(items, downsample_stages=2)
    assert batch["distance_matrices"].shape == (2, 1, 8, 8)
    assert batch["pair_masks"][0, 0, 2, 2]
    assert not batch["pair_masks"][0, 0, 3, 3]
    assert batch["distance_matrices"][0, 0, 3:, :].sum() == 0
    assert torch.isclose(batch["sequence_separation"][0, 0, 0, 2], torch.tensor(1.0))
    assert batch["sequence_separation"][0, 0, 3, 3] == 0
