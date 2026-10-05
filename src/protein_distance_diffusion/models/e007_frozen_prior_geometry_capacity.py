"""Capacity-controlled frozen-ProGen2 geometry conditioners for E007 Phase 4C.1."""

from __future__ import annotations

import torch
from torch import nn

from protein_distance_diffusion.models.e007_frozen_prior_geometry import (
    InvariantGeometryConditioner,
)

CAPACITY_CONDITIONER_VERSION = "e007_frozen_progen2_geometry_capacity_v1"
MEDIUM_CONDITIONER_VERSION = "e007_pair_message_conditioner_medium_v1"


class PairToResidueMessageBlock(nn.Module):
    """Update residue states using masked messages from every paired residue."""

    def __init__(self, *, pair_width: int, residue_width: int) -> None:
        super().__init__()
        self.pair_to_residue = nn.Linear(pair_width, residue_width)
        self.update = nn.Sequential(
            nn.LayerNorm(2 * residue_width),
            nn.Linear(2 * residue_width, residue_width),
            nn.GELU(),
            nn.Linear(residue_width, residue_width),
        )
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        residue: torch.Tensor,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.pair_to_residue(pair) + residue[:, None, :, :]
        messages = messages * pair_mask[..., None]
        denominator = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregated = messages.sum(dim=2) / denominator
        update = self.update(torch.cat((residue, aggregated), dim=-1))
        result = residue + self.residual_scale * update
        return result * residue_mask[..., None]


class MediumInvariantGeometryConditioner(nn.Module):
    """O(3)-invariant pair encoder with iterative residue message passing."""

    def __init__(
        self,
        *,
        rbf_bins: int = 32,
        separation_bins: int = 65,
        separation_width: int = 32,
        pair_width: int = 192,
        residue_width: int = 384,
        shared_output_width: int = 1024,
        message_blocks: int = 4,
        maximum_distance: float = 32.0,
        initial_gate_logit: float = -4.0,
        injection_depths: tuple[int, int, int] = (0, 5, 11),
    ) -> None:
        super().__init__()
        dimensions = (
            rbf_bins,
            separation_bins,
            separation_width,
            pair_width,
            residue_width,
            shared_output_width,
            message_blocks,
        )
        if min(dimensions) < 1 or maximum_distance <= 0:
            raise ValueError("E007 Phase 4C.1 conditioner dimensions must be positive")
        if message_blocks < 4:
            raise ValueError("E007 Phase 4C.1 medium conditioner requires at least four message blocks")
        injection_depths = tuple(injection_depths)
        if len(injection_depths) != 3 or tuple(sorted(set(injection_depths))) != injection_depths:
            raise ValueError("E007 Phase 4C.1 injection depths must be three unique sorted blocks")
        self.rbf_bins = rbf_bins
        self.separation_bins = separation_bins
        self.shared_output_width = shared_output_width
        self.maximum_distance = maximum_distance
        self.injection_depths = injection_depths
        self.register_buffer("rbf_centers", torch.linspace(0, maximum_distance, rbf_bins))
        self.separation_embedding = nn.Embedding(separation_bins, separation_width)
        self.pair_encoder = nn.Sequential(
            nn.Linear(rbf_bins + separation_width + 1, pair_width),
            nn.GELU(),
            nn.LayerNorm(pair_width),
        )
        self.residue_input = nn.Sequential(
            nn.Linear(pair_width, residue_width),
            nn.LayerNorm(residue_width),
        )
        self.message_blocks = nn.ModuleList(
            [
                PairToResidueMessageBlock(pair_width=pair_width, residue_width=residue_width)
                for _ in range(message_blocks)
            ]
        )
        self.output_adapter = nn.Sequential(
            nn.Linear(residue_width, shared_output_width),
            nn.GELU(),
            nn.Linear(shared_output_width, shared_output_width),
        )
        self.injection_gate_logits = nn.Parameter(torch.full((3,), initial_gate_logit))

    @staticmethod
    def _continuity_pair_mask(residue_mask: torch.Tensor, continuity_mask: torch.Tensor) -> torch.Tensor:
        batch, length = residue_mask.shape
        if continuity_mask.shape != (batch, max(length - 1, 0)):
            raise ValueError("E007 Phase 4C.1 continuity mask shape contradiction")
        if length == 0:
            return residue_mask[:, :, None] & residue_mask[:, None, :]
        breaks = torch.zeros((batch, length), dtype=torch.long, device=residue_mask.device)
        if length > 1:
            breaks[:, 1:] = (~continuity_mask).long()
        segments = torch.cumsum(breaks, dim=1)
        return (segments[:, :, None] == segments[:, None, :]) & residue_mask[:, :, None] & residue_mask[:, None, :]

    def forward(
        self,
        coordinates: torch.Tensor,
        residue_mask: torch.Tensor,
        continuity_mask: torch.Tensor,
        *,
        null_geometry: bool = False,
    ) -> torch.Tensor:
        if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
            raise ValueError("E007 Phase 4C.1 coordinates must have shape [B,N,3]")
        if residue_mask.shape != coordinates.shape[:2]:
            raise ValueError("E007 Phase 4C.1 residue mask shape contradiction")
        pair_mask = self._continuity_pair_mask(residue_mask, continuity_mask)
        batch, length = residue_mask.shape
        if null_geometry:
            positions = torch.arange(length, device=coordinates.device)
            separation = (positions[:, None] - positions[None, :]).abs().clamp_max(self.separation_bins - 1)
            separation_features = self.separation_embedding(separation)[None].expand(batch, -1, -1, -1) * 0
            rbf = coordinates.new_zeros((batch, length, length, self.rbf_bins))
            continuity = pair_mask[..., None].to(coordinates.dtype) * 0
            features = torch.cat((rbf, separation_features, continuity), dim=-1)
        else:
            distances = torch.cdist(coordinates.float(), coordinates.float())
            width = self.maximum_distance / max(self.rbf_bins - 1, 1)
            rbf = torch.exp(-((distances[..., None] - self.rbf_centers) / width).square())
            positions = torch.arange(length, device=coordinates.device)
            separation = (positions[:, None] - positions[None, :]).abs().clamp_max(self.separation_bins - 1)
            separation_features = self.separation_embedding(separation)[None].expand(batch, -1, -1, -1)
            continuity = pair_mask[..., None].to(coordinates.dtype)
            features = torch.cat((rbf, separation_features, continuity), dim=-1)
            features = features * pair_mask[..., None]
        pair = self.pair_encoder(features.to(self.rbf_centers.dtype)) * pair_mask[..., None]
        pooled = pair.sum(dim=2) / pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        residue = self.residue_input(pooled) * residue_mask[..., None]
        for block in self.message_blocks:
            residue = block(residue, pair, pair_mask, residue_mask)
        return self.output_adapter(residue) * residue_mask[..., None]

    def gated_conditioning(self, conditioning: torch.Tensor, injection_index: int) -> torch.Tensor:
        if not 0 <= injection_index < len(self.injection_gate_logits):
            raise IndexError("E007 Phase 4C.1 injection index is invalid")
        return conditioning * torch.sigmoid(self.injection_gate_logits[injection_index])

    def gate_values(self) -> list[float]:
        return torch.sigmoid(self.injection_gate_logits.detach()).cpu().tolist()


def build_capacity_conditioner(capacity: str, settings: dict[str, object]) -> nn.Module:
    if capacity == "small":
        return InvariantGeometryConditioner(**settings)
    if capacity == "medium":
        values = dict(settings)
        values["injection_depths"] = tuple(values["injection_depths"])
        return MediumInvariantGeometryConditioner(**values)
    raise ValueError(f"E007 Phase 4C.1 unsupported capacity: {capacity}")


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
