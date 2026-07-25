"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically save a PyTorch checkpoint.

    Args:
        path: Destination `.pt` file.
        payload: Checkpoint payload.

    Returns:
        None.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(dst)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint.

    Args:
        path: Checkpoint path.
        map_location: Torch map location.

    Returns:
        Checkpoint payload.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
