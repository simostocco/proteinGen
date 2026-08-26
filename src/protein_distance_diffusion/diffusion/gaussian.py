"""Gaussian diffusion utilities for symmetric distance matrices."""

from __future__ import annotations

from enum import Enum

import torch


class PredictionType(str, Enum):  # noqa: UP042
    """Supported diffusion model output parameterizations."""

    EPSILON = "epsilon"
    V = "v"


def validate_prediction_type(value: str | PredictionType) -> PredictionType:
    """Validate and normalize the diffusion prediction parameterization."""
    if isinstance(value, PredictionType):
        return value
    try:
        return PredictionType(str(value))
    except ValueError as exc:
        raise ValueError("prediction_parameterization must be 'epsilon' or 'v'") from exc


def prediction_parameterization_from_config(config: dict) -> PredictionType:
    """Return canonical prediction parameterization from config/checkpoint metadata.

    Existing checkpoints used `prediction_type`; checkpoints without either field are
    interpreted as epsilon for backward compatibility.
    """
    canonical = config.get("prediction_parameterization")
    legacy = config.get("prediction_type")
    if canonical is not None and legacy is not None and str(canonical) != str(legacy):
        raise ValueError(
            "Conflicting prediction parameterization fields: "
            f"prediction_parameterization={canonical!r}, prediction_type={legacy!r}"
        )
    return validate_prediction_type(canonical if canonical is not None else legacy if legacy is not None else "epsilon")


def ensure_config_matches_checkpoint_parameterization(
    *,
    config: dict,
    checkpoint_config: dict,
) -> PredictionType:
    """Validate that runtime config agrees with checkpoint parameterization."""
    config_value = prediction_parameterization_from_config(config)
    checkpoint_value = prediction_parameterization_from_config(checkpoint_config)
    if config_value is not checkpoint_value:
        raise ValueError(
            "Config/checkpoint prediction parameterization mismatch: "
            f"config={config_value.value}, checkpoint={checkpoint_value.value}"
        )
    return checkpoint_value


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
        previous = torch.cat([torch.ones(1, dtype=self.alphas_cumprod.dtype), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_prev = previous
        self.posterior_variance = self.betas * (1.0 - previous) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = self.posterior_variance.clamp_min(0.0)

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
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
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

    def predict_x0_from_epsilon(self, x_t: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Reconstruct x_0 from x_t and exact/model-predicted epsilon."""
        alpha_bar = self.alphas_cumprod.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        return (x_t.float() - (1.0 - alpha_bar).sqrt() * epsilon.float()) / alpha_bar.sqrt()

    def v_target(self, x_start: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        """Compute v = sqrt(alpha_bar) * epsilon - sqrt(1-alpha_bar) * x0."""
        alpha_bar = self.alphas_cumprod.to(x_start.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * epsilon.float() - (1.0 - alpha_bar).sqrt() * x_start.float()

    def training_target(
        self,
        *,
        x_start: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor,
        prediction_type: str | PredictionType,
    ) -> torch.Tensor:
        """Return the training target for the requested output parameterization."""
        parameterization = validate_prediction_type(prediction_type)
        if parameterization is PredictionType.EPSILON:
            return epsilon.float()
        return self.v_target(x_start, t, epsilon)

    def predict_x0_epsilon_from_model_output(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        prediction_type: str | PredictionType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert model output to `(x0_hat, epsilon_hat)` in float32."""
        parameterization = validate_prediction_type(prediction_type)
        alpha_bar = self.alphas_cumprod.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        x_t = x_t.float()
        model_output = model_output.float()
        if parameterization is PredictionType.EPSILON:
            epsilon_hat = model_output
            x0_hat = self.predict_x0_from_epsilon(x_t, t, epsilon_hat)
            return x0_hat, epsilon_hat
        x0_hat = alpha_bar.sqrt() * x_t - (1.0 - alpha_bar).sqrt() * model_output
        epsilon_hat = (1.0 - alpha_bar).sqrt() * x_t + alpha_bar.sqrt() * model_output
        return x0_hat, epsilon_hat

    def posterior_mean_from_x0_epsilon(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDPM q(x_{t-1} | x_t, x0_hat) posterior mean."""
        beta_t = self.betas.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alphas_cumprod.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alphas_cumprod_prev.to(x_t.device, dtype=torch.float32)[t].view(-1, 1, 1, 1)
        coef_x0 = beta_t * alpha_bar_prev.sqrt() / (1.0 - alpha_bar_t)
        coef_xt = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return coef_x0 * x0_hat.float() + coef_xt * x_t.float()


def per_protein_masked_upper_triangular_loss(
    epsilon: torch.Tensor, epsilon_hat: torch.Tensor, pair_mask: torch.Tensor
) -> torch.Tensor:
    """Compute per-protein masked upper-triangular epsilon MSE values.

    Args:
        epsilon: Target noise [B, 1, L, L].
        epsilon_hat: Predicted noise [B, 1, L, L].
        pair_mask: Valid pair mask [B, 1, L, L].

    Returns:
        One scalar per protein after normalizing by its own valid upper-triangular pair count.
    """
    matrix_size = epsilon.shape[-1]
    upper = torch.triu(
        torch.ones((matrix_size, matrix_size), dtype=torch.bool, device=epsilon.device),
        diagonal=1,
    )
    valid = pair_mask.bool() & upper[None, None]
    sq = (epsilon - epsilon_hat).pow(2) * valid.float()
    denom = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    return sq.flatten(1).sum(dim=1) / denom


def masked_upper_triangular_loss(
    epsilon: torch.Tensor,
    epsilon_hat: torch.Tensor,
    pair_mask: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute masked upper-triangular epsilon MSE, optionally weighted per protein."""
    per_protein = per_protein_masked_upper_triangular_loss(epsilon, epsilon_hat, pair_mask)
    if sample_weight is None:
        return per_protein.mean()
    weights = sample_weight.to(device=per_protein.device, dtype=per_protein.dtype).flatten()
    if weights.numel() != per_protein.numel():
        raise ValueError("sample_weight must contain one value per protein")
    return (weights * per_protein).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
