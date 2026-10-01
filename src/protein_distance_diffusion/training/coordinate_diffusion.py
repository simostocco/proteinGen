"""Centered coordinate VP diffusion and synthetic sampling for E007."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule


class _ZeroSafeSqrt(torch.autograd.Function):
    """Square root with the exact subgradient zero at an exact zero input."""

    @staticmethod
    def forward(ctx, squared: torch.Tensor) -> torch.Tensor:
        result = torch.sqrt(squared)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        (result,) = ctx.saved_tensors
        derivative = torch.where(result > 0, 0.5 / result, torch.zeros_like(result))
        return (gradient * derivative,)


def coordinates_to_distance_matrix(
    coordinates: torch.Tensor,
    residue_mask: torch.Tensor | None = None,
    *,
    diagnostic_float64: bool = False,
) -> torch.Tensor:
    """Construct exact-diagonal Euclidean distances from coordinates.

    Half and bfloat16 inputs are promoted to float32. Read-only scientific
    diagnostics may request float64. No epsilon is introduced into distances.
    """
    if coordinates.ndim not in {2, 3} or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape [N,3] or [B,N,3]")
    unbatched = coordinates.ndim == 2
    values = coordinates[None] if unbatched else coordinates
    dtype = (
        torch.float64
        if diagnostic_float64
        else (torch.float32 if values.dtype in {torch.float16, torch.bfloat16} else values.dtype)
    )
    values = values.to(dtype)
    differences = values[:, :, None, :] - values[:, None, :, :]
    squared = differences.square().sum(dim=-1).clamp_min(0)
    distances = _ZeroSafeSqrt.apply(squared)
    distances = 0.5 * (distances + distances.transpose(-1, -2))
    diagonal = torch.eye(distances.shape[-1], dtype=torch.bool, device=distances.device)[None]
    distances = distances.masked_fill(diagonal, 0.0)
    if residue_mask is not None:
        mask = residue_mask[None] if residue_mask.ndim == 1 else residue_mask
        if mask.shape != distances.shape[:2]:
            raise ValueError("residue_mask shape contradicts coordinates")
        pair_mask = mask.bool()[:, :, None] & mask.bool()[:, None, :]
        distances = distances * pair_mask.to(distances.dtype)
    return distances[0] if unbatched else distances


def center_coordinates(coordinates: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """Center valid coordinates independently per protein and zero padding."""
    mask = residue_mask.to(coordinates.dtype)[..., None]
    centroid = (coordinates * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return (coordinates - centroid) * mask


def centered_coordinate_noise(
    coordinates: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample Gaussian coordinate noise with zero valid-residue centroid."""
    noise = torch.randn(
        coordinates.shape,
        dtype=coordinates.dtype,
        device=coordinates.device,
        generator=generator,
    )
    return center_coordinates(noise, residue_mask)


@dataclass(frozen=True)
class CoordinateDiffusionBatch:
    noisy_coordinates: torch.Tensor
    coordinate_noise: torch.Tensor
    coordinate_v_target: torch.Tensor
    timesteps: torch.Tensor


class _CoordinateModel(Protocol):
    def __call__(
        self,
        noisy_coordinates: torch.Tensor,
        timesteps: torch.Tensor,
        lengths: torch.Tensor,
        residue_mask: torch.Tensor,
        chain_continuity_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...


class CoordinateVPDiffusion:
    """Cosine-schedule centred VP diffusion with coordinate v-prediction."""

    def __init__(self, timesteps: int = 500) -> None:
        if timesteps < 2:
            raise ValueError("coordinate diffusion requires at least two timesteps")
        betas = cosine_beta_schedule(timesteps)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @property
    def timesteps(self) -> int:
        return int(self.betas.numel())

    def alpha_sigma(self, timestep: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alphas_cumprod.to(reference.device, torch.float32)[timestep]
        shape = (timestep.shape[0],) + (1,) * (reference.ndim - 1)
        return alpha_bar.sqrt().view(shape), (1.0 - alpha_bar).sqrt().view(shape)

    def make_training_batch(
        self,
        clean_coordinates: torch.Tensor,
        residue_mask: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> CoordinateDiffusionBatch:
        clean = center_coordinates(clean_coordinates, residue_mask)
        if timesteps is None:
            timesteps = torch.randint(
                self.timesteps,
                (clean.shape[0],),
                device=clean.device,
                generator=generator,
            )
        noise = centered_coordinate_noise(clean, residue_mask, generator=generator)
        alpha, sigma = self.alpha_sigma(timesteps, clean)
        noisy = center_coordinates(alpha * clean + sigma * noise, residue_mask)
        target = self.training_target(clean, noise, timesteps, residue_mask)
        return CoordinateDiffusionBatch(noisy, noise, target, timesteps)

    def training_target(
        self,
        clean_coordinates: torch.Tensor,
        epsilon: torch.Tensor,
        timesteps: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Construct E007's centered coordinate velocity target in float32."""
        clean = center_coordinates(clean_coordinates.float(), residue_mask)
        noise = center_coordinates(epsilon.float(), residue_mask)
        alpha, sigma = self.alpha_sigma(timesteps, clean)
        return center_coordinates(alpha * noise - sigma * clean, residue_mask)

    def reconstruct_x0(
        self,
        noisy_coordinates: torch.Tensor,
        timesteps: torch.Tensor,
        v_prediction: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(timesteps, noisy_coordinates)
        return center_coordinates(alpha * noisy_coordinates - sigma * v_prediction, residue_mask)

    def deterministic_reverse_step(
        self,
        noisy_coordinates: torch.Tensor,
        timesteps: torch.Tensor,
        v_prediction: torch.Tensor,
        residue_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one deterministic DDIM-style reverse update."""
        alpha, sigma = self.alpha_sigma(timesteps, noisy_coordinates)
        x0 = center_coordinates(alpha * noisy_coordinates - sigma * v_prediction, residue_mask)
        epsilon = center_coordinates(sigma * noisy_coordinates + alpha * v_prediction, residue_mask)
        previous_timesteps = (timesteps - 1).clamp_min(0)
        previous_alpha, previous_sigma = self.alpha_sigma(previous_timesteps, noisy_coordinates)
        previous = previous_alpha * x0 + previous_sigma * epsilon
        terminal = (timesteps == 0).view((-1,) + (1,) * (noisy_coordinates.ndim - 1))
        previous = torch.where(terminal, x0, previous)
        return center_coordinates(previous, residue_mask), x0, epsilon

    @torch.no_grad()
    def sample(
        self,
        model: _CoordinateModel,
        *,
        length: int,
        seed: int,
        device: torch.device | str = "cpu",
        return_trajectory: bool = False,
    ) -> dict[str, object]:
        """Run deterministic DDIM-style synthetic sampling from centered noise."""
        if length < 2:
            raise ValueError("requested coordinate sample length must be at least two")
        device = torch.device(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        residue_mask = torch.ones((1, length), dtype=torch.bool, device=device)
        pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
        continuity = torch.ones((1, length - 1), dtype=torch.bool, device=device)
        lengths = torch.tensor([length], dtype=torch.long, device=device)
        coordinates = centered_coordinate_noise(
            torch.empty((1, length, 3), device=device), residue_mask, generator=generator
        )
        trajectory = [coordinates.detach().clone()] if return_trajectory else []
        for step in range(self.timesteps - 1, -1, -1):
            timestep = torch.tensor([step], dtype=torch.long, device=device)
            prediction = model(coordinates, timestep, lengths, residue_mask, continuity)["v_prediction"]
            coordinates, _, _ = self.deterministic_reverse_step(coordinates, timestep, prediction, residue_mask)
            if return_trajectory:
                trajectory.append(coordinates.detach().clone())
        distances = coordinates_to_distance_matrix(coordinates, residue_mask)
        return {
            "coordinates": coordinates,
            "distance_matrix": distances,
            "residue_mask": residue_mask,
            "pair_mask": pair_mask,
            "metadata": {
                "seed": seed,
                "requested_length": length,
                "diffusion_steps": self.timesteps,
                "parameterization": "coordinate_v",
                "trained_weights_required": False,
            },
            **({"trajectory": trajectory} if return_trajectory else {}),
        }
