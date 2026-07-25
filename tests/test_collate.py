"""Batch collation tests."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

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


def test_standard_dataloader_mixed_lengths_dynamic_padding_and_masks() -> None:
    """Standard PyTorch DataLoader batches mixed lengths through the variable-size collate."""
    items = [
        {"sample_id": "n17", "length": 17, "distance_matrix": torch.ones(17, 17)},
        {"sample_id": "n29", "length": 29, "distance_matrix": torch.ones(29, 29) * 2},
    ]
    loader = DataLoader(
        items,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda batch: collate_distance_maps(batch, downsample_stages=3),
    )
    batch = next(iter(loader))
    assert batch["distance_matrices"].shape == (2, 1, 32, 32)
    assert batch["lengths"].tolist() == [17, 29]
    assert batch["pair_masks"][0, 0, 16, 16]
    assert not batch["pair_masks"][0, 0, 17, 17]
    assert batch["distance_matrices"][0, 0, 17:, :].sum() == 0
    assert batch["distance_matrices"][1, 0, :29, :29].sum() > batch["distance_matrices"][0, 0, :17, :17].sum()
