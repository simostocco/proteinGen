"""Frozen-prior geometry conditioning for E007 Phase 4C."""

from __future__ import annotations

import torch
from torch import nn

CONDITIONER_VERSION = "e007_frozen_prior_pair_distance_conditioner_v1"


class InvariantGeometryConditioner(nn.Module):
    """Encode masked C-alpha distances without orientation or translation."""

    def __init__(
        self,
        *,
        rbf_bins: int = 32,
        hidden_width: int = 256,
        shared_output_width: int = 1024,
        maximum_distance: float = 32.0,
    ) -> None:
        super().__init__()
        if min(rbf_bins, hidden_width, shared_output_width) < 1 or maximum_distance <= 0:
            raise ValueError("E007 Phase 4C conditioner dimensions must be positive")
        self.rbf_bins = rbf_bins
        self.hidden_width = hidden_width
        self.shared_output_width = shared_output_width
        self.maximum_distance = maximum_distance
        self.register_buffer("rbf_centers", torch.linspace(0, maximum_distance, rbf_bins))
        self.pair_encoder = nn.Sequential(
            nn.Linear(rbf_bins, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
        )
        self.conditioning_adapter = nn.Sequential(
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, shared_output_width),
        )
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        coordinates: torch.Tensor,
        residue_mask: torch.Tensor,
        *,
        null_geometry: bool = False,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("E007 Phase 4C coordinates must have shape [B,N,3]")
        if residue_mask.shape != coordinates.shape[:2]:
            raise ValueError("E007 Phase 4C residue mask shape contradiction")
        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
        if null_geometry:
            pooled = coordinates.new_zeros((*coordinates.shape[:2], self.rbf_bins))
        else:
            distances = torch.cdist(coordinates.float(), coordinates.float())
            widths = self.maximum_distance / max(self.rbf_bins - 1, 1)
            rbf = torch.exp(-((distances[..., None] - self.rbf_centers) / widths).square())
            rbf = rbf * pair_mask[..., None]
            pooled = rbf.sum(dim=2) / pair_mask.sum(dim=2).clamp_min(1)[..., None]
        hidden = self.pair_encoder(pooled.to(coordinates.dtype))
        output = self.conditioning_adapter(hidden) * torch.sigmoid(self.gate_logit)
        return output * residue_mask[..., None]

    def for_prior_width(self, conditioning: torch.Tensor, width: int) -> torch.Tensor:
        """Use a fixed coordinate projection so trainable budgets remain identical."""
        if not 1 <= width <= self.shared_output_width:
            raise ValueError("E007 Phase 4C prior width exceeds shared conditioning width")
        return conditioning[..., :width]


def conditioner_parameter_count(model: InvariantGeometryConditioner) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
