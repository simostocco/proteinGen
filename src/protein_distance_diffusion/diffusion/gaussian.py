"""Gaussian diffusion utilities for symmetric distance matrices."""

from __future__ import annotations

import torch


def project_symmetric_zero_diagonal(x: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """Project matrices to symmetric, zero-diagonal, masked representation.

    Args:
        x: Tensor with shape [B, 1, L, L].
        pair_mask: Boolean tensor with shape [B, 1, L, L].

    Returns:
        Projected tensor with shape [B, 1, L, L].
    """
    y = 0.5 * (x + x.transpose(-1, -2))
    diag = torch.eye(y.shape[-1], dtype=torch.bool, device=y.device)[None, None]
    y = y.masked_fill(diag, 0.0)
    return y * pair_mask.float()


def sample_symmetric_noise(
    shape: tuple[int, int, int, int],
    pair_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return symmetric zero-diagonal Gaussian noise.

    Args:
        shape: Expected input/output shape [B, 1, L, L].
        pair_mask: Boolean valid pair mask with shape [B, 1, L, L].
        generator: Optional torch random generator.

    Returns:
        Gaussian noise with shape [B, 1, L, L], mirrored from sampled upper-triangular entries.
    """
    if len(shape) != 4 or shape[1] != 1 or shape[-1] != shape[-2]:
        raise ValueError(f"Expected shape [B, 1, L, L], got {shape}")
    upper = torch.triu(torch.ones(shape[-2:], dtype=torch.bool, device=pair_mask.device), diagonal=1)
    sampled = torch.randn(shape, generator=generator, device=pair_mask.device)
    noise = sampled.masked_fill(~upper[None, None], 0.0)
    noise = noise + noise.transpose(-1, -2)
    return noise * pair_mask.float()


class GaussianDiffusion:
    """DDPM forward process and epsilon-prediction loss.

    Args:
        betas: Noise schedule tensor with shape [T].
    """

    def __init__(self, betas: torch.Tensor) -> None:
        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @property
    def timesteps(self) -> int:
        """Return number of diffusion steps."""
        return int(self.betas.numel())

    def to(self, device: torch.device | str) -> GaussianDiffusion:
        """Move schedule tensors to a device.

        Args:
            device: Target device.

        Returns:
            Self.
        """
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffuse clean matrices at per-sample timesteps.

        Args:
            x_start: Clean normalized matrices [B, 1, L, L].
            t: int64 timestep tensor [B].
            pair_mask: Valid pair mask [B, 1, L, L].
            noise: Optional symmetric noise [B, 1, L, L].
            generator: Optional random generator used if `noise` is omitted.

        Returns:
            Tuple `(x_t, epsilon)` with both tensors [B, 1, L, L].
        """
        eps = (
            noise if noise is not None else sample_symmetric_noise(tuple(x_start.shape), pair_mask, generator=generator)
        )
        a = self.alphas_cumprod.to(x_start.device)[t].view(-1, 1, 1, 1)
        x_t = a.sqrt() * x_start + (1.0 - a).sqrt() * eps
        return project_symmetric_zero_diagonal(x_t, pair_mask), eps


def masked_upper_triangular_loss(
    epsilon: torch.Tensor, epsilon_hat: torch.Tensor, pair_mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-protein masked upper-triangular epsilon MSE.

    Args:
        epsilon: Target noise [B, 1, L, L].
        epsilon_hat: Predicted noise [B, 1, L, L].
        pair_mask: Valid pair mask [B, 1, L, L].

    Returns:
        Scalar loss averaged across proteins, after normalizing each protein by its own
        valid upper-triangular pair count so long proteins do not dominate by O(N^2).
    """
    matrix_size = epsilon.shape[-1]
    upper = torch.triu(
        torch.ones((matrix_size, matrix_size), dtype=torch.bool, device=epsilon.device),
        diagonal=1,
    )
    valid = pair_mask.bool() & upper[None, None]
    sq = (epsilon - epsilon_hat).pow(2) * valid.float()
    denom = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    return (sq.flatten(1).sum(dim=1) / denom).mean()
