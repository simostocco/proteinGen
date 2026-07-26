"""Reverse diffusion sampling for generated distance matrices."""

from __future__ import annotations

from typing import Any

import torch

from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    PredictionType,
    project_symmetric_zero_diagonal,
    validate_prediction_type,
)


def tensor_stats(x: torch.Tensor, pair_mask: torch.Tensor) -> dict[str, float]:
    """Return compact masked tensor diagnostics."""
    values = x.detach().float()[pair_mask.bool()]
    if values.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(values.min().cpu()),
        "max": float(values.max().cpu()),
        "mean": float(values.mean().cpu()),
        "std": float(values.std(unbiased=False).cpu()),
    }


def projection_diagnostics(x: torch.Tensor, pair_mask: torch.Tensor) -> dict[str, float]:
    """Return symmetry, diagonal, and finite diagnostics for a generated state."""
    masked = x.detach().float() * pair_mask.float()
    denom = masked.abs().amax().clamp_min(1e-8)
    return {
        "fraction_nonfinite": float((~torch.isfinite(masked)).float().mean().cpu()),
        "symmetry_error": float(((masked - masked.transpose(-1, -2)).abs().amax() / denom).cpu()),
        "diagonal_abs_max": float(torch.diagonal(masked, dim1=-1, dim2=-2).abs().amax().cpu()),
    }


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    *,
    lengths: torch.Tensor,
    pair_mask: torch.Tensor,
    sequence_separation: torch.Tensor,
    device: torch.device,
    generator: torch.Generator | None = None,
    prediction_type: str | PredictionType = PredictionType.EPSILON,
    trace_every: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]]:
    """Generate normalized distance matrices with ancestral DDPM sampling.

    Args:
        model: Epsilon-prediction model.
        diffusion: Gaussian diffusion schedule.
        lengths: Residue lengths [B].
        pair_mask: Valid pair masks [B, 1, L, L].
        sequence_separation: Sequence separation channel [B, 1, L, L].
        device: Target device.
        generator: Optional random generator.
        prediction_type: Model output parameterization. Currently only epsilon.
        trace_every: Optional reverse-step interval for compact diagnostics.

    Returns:
        Normalized generated matrices, or `(matrices, trace)` when tracing is enabled.
    """
    parameterization = validate_prediction_type(prediction_type)
    if parameterization is not PredictionType.EPSILON:
        raise ValueError("Only epsilon prediction is supported")

    mask = pair_mask.to(device)
    lengths = lengths.to(device)
    sequence_separation = sequence_separation.to(device)
    x = torch.randn(mask.shape, generator=generator, device=device, dtype=torch.float32)
    x = project_symmetric_zero_diagonal(x, mask)
    diffusion.to(device)
    trace: list[dict[str, Any]] = []

    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((x.shape[0],), step, dtype=torch.long, device=device)
        eps = model(x.float(), t, lengths, sequence_separation.float(), mask).float()
        eps = project_symmetric_zero_diagonal(eps, mask)
        x0_hat = project_symmetric_zero_diagonal(diffusion.predict_x0_from_epsilon(x, t, eps), mask)
        posterior_mean = project_symmetric_zero_diagonal(
            diffusion.posterior_mean_from_x0_epsilon(x_t=x, t=t, x0_hat=x0_hat),
            mask,
        )

        if trace_every is not None and (step % trace_every == 0 or step == diffusion.timesteps - 1):
            row: dict[str, Any] = {"timestep": int(step)}
            row["x_t"] = tensor_stats(x, mask)
            row["epsilon_prediction"] = tensor_stats(eps, mask)
            row["x0_hat"] = tensor_stats(x0_hat, mask)
            row["posterior_mean"] = tensor_stats(posterior_mean, mask)
            row.update(projection_diagnostics(x, mask))
            trace.append(row)

        if step > 0:
            variance = diffusion.posterior_variance.to(device, dtype=torch.float32)[step]
            noise = torch.randn(x.shape, generator=generator, device=device, dtype=torch.float32)
            noise = project_symmetric_zero_diagonal(noise, mask)
            x = posterior_mean + variance.sqrt() * noise
        else:
            x = posterior_mean
        x = project_symmetric_zero_diagonal(x.float(), mask)

    return (x, trace) if trace_every is not None else x
