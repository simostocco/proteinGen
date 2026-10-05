"""O(3)-equivariant coordinate diffusion model with an invariant pair U-Net."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from protein_distance_diffusion.models.pair_grid_unet_trunk import PairGridUNetTrunk


def pair_masks(residue_mask: torch.Tensor, chain_continuity_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return biological pair and same-contiguous-segment masks.

    The biological mask is exactly the outer product of valid residue masks.
    Continuity is an independent model feature and never silently removes pairs.
    """
    valid = residue_mask.bool()
    biological = valid[:, :, None] & valid[:, None, :]
    if chain_continuity_mask.shape != (valid.shape[0], max(valid.shape[1] - 1, 0)):
        raise ValueError("chain_continuity_mask shape contradicts residue_mask")
    segment = torch.zeros_like(valid, dtype=torch.long)
    if valid.shape[1] > 1:
        breaks = ~chain_continuity_mask.bool()
        segment[:, 1:] = torch.cumsum(breaks.long(), dim=1)
    continuous = biological & (segment[:, :, None] == segment[:, None, :])
    return biological, continuous


class EquivariantPairCoordinateUNet(nn.Module):
    """Predict coordinate-v vectors through invariant pair coefficients.

    Scalar coefficients depend only on invariant distances, sequence separation,
    masks, timestep, and length. Lifting those coefficients along pair unit
    vectors makes the output equivariant under every orthogonal transformation,
    including reflections. This C-alpha-only contract makes no chirality claim.
    """

    def __init__(
        self,
        *,
        rbf_bins: int = 16,
        rbf_min_distance: float = 0.0,
        rbf_max_distance: float = 5.0,
        lifting_epsilon: float = 1e-6,
        **trunk_config: Any,
    ) -> None:
        super().__init__()
        if rbf_bins < 2 or rbf_max_distance <= rbf_min_distance:
            raise ValueError("invalid radial-basis configuration")
        self.rbf_bins = int(rbf_bins)
        self.lifting_epsilon = float(lifting_epsilon)
        centers = torch.linspace(rbf_min_distance, rbf_max_distance, rbf_bins)
        spacing = float(centers[1] - centers[0])
        self.register_buffer("rbf_centers", centers, persistent=True)
        self.register_buffer("rbf_gamma", torch.tensor(1.0 / max(spacing * spacing, 1e-12)), persistent=True)
        self.pair_trunk = PairGridUNetTrunk(input_channels=rbf_bins + 4, **trunk_config)
        self.downsample_factor = self.pair_trunk.downsample_factor

    def _single(
        self,
        coordinates: torch.Tensor,
        timestep: torch.Tensor,
        length: torch.Tensor,
        residue_mask: torch.Tensor,
        continuity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = coordinates.shape[1]
        padded = math.ceil(n / self.downsample_factor) * self.downsample_factor
        pad = padded - n
        coordinates = torch.nn.functional.pad(coordinates, (0, 0, 0, pad))
        residue_mask = torch.nn.functional.pad(residue_mask, (0, pad), value=False)
        continuity = torch.nn.functional.pad(continuity, (0, pad), value=False)
        biological, continuous = pair_masks(residue_mask, continuity[:, : padded - 1])
        relative = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distances = torch.linalg.vector_norm(relative, dim=-1)
        rbf = torch.exp(
            -self.rbf_gamma.to(distances.dtype)
            * (distances[:, None] - self.rbf_centers.to(distances.dtype)[None, :, None, None]).square()
        )
        indices = torch.arange(padded, device=coordinates.device)
        separation = (indices[:, None] - indices[None, :]).abs().to(coordinates.dtype)
        separation = separation / length.to(coordinates.dtype).sub(1).clamp_min(1)[:, None, None]
        adjacent = (indices[:, None] - indices[None, :]).abs().eq(1)
        features = torch.cat(
            (
                rbf,
                separation[:, None],
                adjacent.to(coordinates.dtype)[None, None].expand(coordinates.shape[0], -1, -1, -1),
                continuous.to(coordinates.dtype)[:, None],
                biological.to(coordinates.dtype)[:, None],
            ),
            dim=1,
        )
        features = features * biological[:, None].to(features.dtype)
        raw = self.pair_trunk(features, timestep, length, biological[:, None])
        coefficients = 0.5 * (raw + raw.transpose(-1, -2))
        diagonal = torch.eye(padded, dtype=torch.bool, device=coordinates.device)[None, None]
        coefficients = coefficients.masked_fill(diagonal | ~biological[:, None], 0.0)
        units = relative / (distances[..., None] + self.lifting_epsilon)
        degree = biological.sum(dim=-1).sub(residue_mask.long()).clamp_min(1).to(coordinates.dtype)
        vectors = (coefficients[:, 0, :, :, None] * units).sum(dim=2) / degree.sqrt()[..., None]
        vectors = vectors * residue_mask[..., None]
        centroid = vectors.sum(dim=1, keepdim=True) / residue_mask.sum(dim=1).clamp_min(1)[:, None, None]
        vectors = (vectors - centroid) * residue_mask[..., None]
        return vectors[:, :n], coefficients[:, :, :n, :n], biological[:, :n, :n]

    def forward(
        self,
        noisy_coordinates: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        residue_mask: torch.Tensor,
        chain_continuity_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return coordinate-v predictions and diagnostic scalar pair coefficients."""
        if noisy_coordinates.ndim != 3 or noisy_coordinates.shape[-1] != 3:
            raise ValueError("noisy_coordinates must have shape [B,N,3]")
        batch, side, _ = noisy_coordinates.shape
        if residue_mask.shape != (batch, side) or lengths.shape != (batch,) or timesteps.shape != (batch,):
            raise ValueError("coordinate batch shape contradiction")
        if not torch.equal(residue_mask.sum(dim=1).to(lengths.dtype), lengths):
            raise ValueError("lengths must equal valid residue-mask counts")
        predictions: list[torch.Tensor] = []
        coefficients: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index in range(batch):
            length = int(lengths[index].item())
            prediction, coefficient, mask = self._single(
                noisy_coordinates[index : index + 1, :length],
                timesteps[index : index + 1],
                lengths[index : index + 1],
                residue_mask[index : index + 1, :length],
                chain_continuity_mask[index : index + 1, : max(length - 1, 0)],
            )
            predictions.append(torch.nn.functional.pad(prediction, (0, 0, 0, side - length)))
            coefficients.append(torch.nn.functional.pad(coefficient, (0, side - length, 0, side - length)))
            masks.append(torch.nn.functional.pad(mask, (0, side - length, 0, side - length)))
        return {
            "v_prediction": torch.cat(predictions),
            "pair_coefficients": torch.cat(coefficients),
            "pair_mask": torch.cat(masks),
        }
