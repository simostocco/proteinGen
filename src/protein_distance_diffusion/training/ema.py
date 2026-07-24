"""Exponential moving average for model parameters."""

from __future__ import annotations

import copy

import torch


class EMA:
    """Maintain an exponential moving average copy of a model.

    Args:
        model: Source model.
        decay: EMA decay in [0, 1).
    """

    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters from a source model."""
        for ema_p, src_p in zip(self.module.parameters(), model.parameters(), strict=True):
            ema_p.mul_(self.decay).add_(src_p, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA module state."""
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load EMA module state."""
        self.module.load_state_dict(state)
