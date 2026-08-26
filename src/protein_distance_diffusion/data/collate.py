"""Batch collation for variable-size distance matrices."""

from __future__ import annotations

from typing import Any

import torch


def padded_side(length: int, *, downsample_stages: int = 3) -> int:
    """Round a length up to the U-Net spatial downsampling factor."""
    if downsample_stages < 0:
        raise ValueError("downsample_stages must be non-negative")
    factor = 2 ** int(downsample_stages)
    return ((int(length) + factor - 1) // factor) * factor


def make_pair_mask(lengths: torch.Tensor, side: int) -> torch.Tensor:
    """Return a biological-pair mask with shape ``(B, 1, side, side)``."""
    lengths = lengths.to(dtype=torch.long)
    idx = torch.arange(int(side), device=lengths.device)
    valid = idx[None, :] < lengths[:, None]
    return (valid[:, None, :, None] & valid[:, None, None, :]).bool()


def make_sequence_separation(lengths: torch.Tensor, side: int) -> torch.Tensor:
    """Return ``abs(i-j)/max(N-1,1)`` with padded entries zeroed."""
    lengths = lengths.to(dtype=torch.long)
    idx = torch.arange(int(side), device=lengths.device)
    sep = (idx[None, :] - idx[:, None]).abs().to(dtype=torch.float32)
    denom = torch.clamp(lengths - 1, min=1).to(dtype=torch.float32)
    sep = sep[None, None, :, :] / denom[:, None, None, None]
    return sep * make_pair_mask(lengths, side).to(dtype=torch.float32)


def collate_distance_maps(items: list[dict[str, Any]], *, downsample_stages: int = 3) -> dict[str, Any]:
    """Pad distance matrices dynamically and attach model conditioning channels."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([int(item["length"]) for item in items], dtype=torch.long)
    side = padded_side(int(lengths.max()), downsample_stages=downsample_stages)
    matrices = torch.zeros((len(items), 1, side, side), dtype=torch.float32)
    for idx, item in enumerate(items):
        matrix = torch.as_tensor(item["distance_matrix"], dtype=torch.float32)
        n = int(lengths[idx])
        if matrix.shape != (n, n):
            raise ValueError(f"distance_matrix for {item.get('sample_id', idx)} has shape {tuple(matrix.shape)}")
        matrices[idx, 0, :n, :n] = matrix
    pair_masks = make_pair_mask(lengths, side)
    separation = make_sequence_separation(lengths, side)
    matrices = matrices * pair_masks.to(dtype=torch.float32)
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "lengths": lengths,
        "sample_weights": torch.tensor([float(item.get("sample_weight", 1.0)) for item in items], dtype=torch.float32),
        "distance_matrices": matrices,
        "sequence_separation": separation,
        "pair_masks": pair_masks,
    }
