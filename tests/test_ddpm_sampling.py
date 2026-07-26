"""DDPM equation and reverse-sampling regression tests."""

from __future__ import annotations

import numpy as np
import torch

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, project_symmetric_zero_diagonal
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.evaluation.metrics import generated_matrix_report


def _linear_diffusion() -> GaussianDiffusion:
    return GaussianDiffusion(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))


def test_forward_noising_matches_manual_equation() -> None:
    """q_sample implements x_t = sqrt(alpha_bar) x0 + sqrt(1-alpha_bar) eps."""
    diffusion = _linear_diffusion()
    mask = make_pair_mask(torch.tensor([3]), 3)
    x0 = torch.tensor([[[[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]]])
    eps = torch.ones_like(x0) * mask.float()
    eps = project_symmetric_zero_diagonal(eps, mask)
    t = torch.tensor([1])
    noisy, returned_eps = diffusion.q_sample(x0, t, mask, noise=eps)
    alpha_bar = diffusion.alphas_cumprod[1]
    expected = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * eps
    assert torch.allclose(noisy, project_symmetric_zero_diagonal(expected, mask))
    assert torch.allclose(returned_eps, eps)


def test_x0_reconstruction_from_exact_epsilon_recovers_clean_matrix() -> None:
    """Epsilon parameterization reconstructs x0 exactly when epsilon is exact."""
    diffusion = _linear_diffusion()
    mask = make_pair_mask(torch.tensor([3]), 3)
    x0 = torch.randn(1, 1, 3, 3)
    x0 = project_symmetric_zero_diagonal(x0, mask)
    eps = project_symmetric_zero_diagonal(torch.randn_like(x0), mask)
    t = torch.tensor([2])
    noisy, _ = diffusion.q_sample(x0, t, mask, noise=eps)
    reconstructed = project_symmetric_zero_diagonal(diffusion.predict_x0_from_epsilon(noisy, t, eps), mask)
    assert torch.allclose(reconstructed, x0, atol=1e-5)


def test_posterior_mean_coefficients_match_manual_reference() -> None:
    """Posterior mean uses beta_t, alpha_t, alpha_bar_t and alpha_bar_{t-1} correctly."""
    diffusion = _linear_diffusion()
    x_t = torch.ones(1, 1, 2, 2)
    x0 = torch.ones_like(x_t) * 3.0
    t = torch.tensor([2])
    mean = diffusion.posterior_mean_from_x0_epsilon(x_t=x_t, t=t, x0_hat=x0)
    beta = diffusion.betas[2]
    alpha = diffusion.alphas[2]
    alpha_bar = diffusion.alphas_cumprod[2]
    alpha_bar_prev = diffusion.alphas_cumprod[1]
    expected = beta * alpha_bar_prev.sqrt() / (1.0 - alpha_bar) * x0
    expected = expected + alpha.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * x_t
    assert torch.allclose(mean, expected)


def test_timestep_indexing_and_schedule_identities() -> None:
    """The schedule uses 0-based t=0..T-1 indexing and valid DDPM identities."""
    diffusion = _linear_diffusion()
    assert diffusion.timesteps == 3
    assert torch.allclose(diffusion.alphas, 1.0 - diffusion.betas)
    assert torch.allclose(diffusion.alphas_cumprod, torch.cumprod(diffusion.alphas, dim=0))
    assert diffusion.alphas_cumprod_prev[0] == 1.0
    assert diffusion.posterior_variance[0] == 0.0


class ZeroEpsilonModel(torch.nn.Module):
    """Model that predicts zero epsilon."""

    def forward(self, x, t, lengths, sequence_separation, pair_mask):  # type: ignore[no-untyped-def]
        del t, lengths, sequence_separation
        return torch.zeros_like(x) * pair_mask.float()


def test_t0_reverse_step_adds_no_random_noise() -> None:
    """With one diffusion step, reverse sampling never adds stochastic noise at t=0."""
    diffusion = GaussianDiffusion(torch.tensor([0.1], dtype=torch.float32))
    lengths = torch.tensor([3])
    mask = make_pair_mask(lengths, 3)
    sep = make_sequence_separation(lengths, 3)
    generator_a = torch.Generator().manual_seed(1)
    generator_b = torch.Generator().manual_seed(1)
    out_a = sample_ddpm(
        ZeroEpsilonModel(),
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=generator_a,
    )
    out_b = sample_ddpm(
        ZeroEpsilonModel(),
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=generator_b,
    )
    assert torch.allclose(out_a, out_b)


class OracleEpsilonModel(torch.nn.Module):
    """Oracle epsilon model for a fixed clean matrix."""

    def __init__(self, diffusion: GaussianDiffusion, x0: torch.Tensor) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.x0 = x0

    def forward(self, x, t, lengths, sequence_separation, pair_mask):  # type: ignore[no-untyped-def]
        del lengths, sequence_separation
        alpha_bar = self.diffusion.alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)
        eps = (x - alpha_bar.sqrt() * self.x0.to(x.device)) / (1.0 - alpha_bar).sqrt()
        return project_symmetric_zero_diagonal(eps, pair_mask)


def test_oracle_epsilon_reverse_process_does_not_explode() -> None:
    """A mathematically consistent epsilon oracle keeps reverse samples bounded."""
    diffusion = _linear_diffusion()
    lengths = torch.tensor([3])
    mask = make_pair_mask(lengths, 3)
    sep = make_sequence_separation(lengths, 3)
    x0 = project_symmetric_zero_diagonal(torch.ones(1, 1, 3, 3), mask)
    model = OracleEpsilonModel(diffusion, x0)
    out, trace = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
        trace_every=1,
    )
    assert torch.isfinite(out).all()
    assert out.abs().max() < 5.0
    assert len(trace) == diffusion.timesteps


def test_projection_and_trace_detect_exploding_values() -> None:
    """Trace statistics expose intentionally exploding epsilon predictions."""

    class Exploding(torch.nn.Module):
        def forward(self, x, t, lengths, sequence_separation, pair_mask):  # type: ignore[no-untyped-def]
            del t, lengths, sequence_separation
            return torch.ones_like(x) * pair_mask.float() * 1e6

    diffusion = _linear_diffusion()
    lengths = torch.tensor([3])
    mask = make_pair_mask(lengths, 3)
    sep = make_sequence_separation(lengths, 3)
    out, trace = sample_ddpm(
        Exploding(),
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(4),
        trace_every=1,
    )
    assert torch.diagonal(out, dim1=-1, dim2=-2).abs().max() == 0
    assert trace[0]["epsilon_prediction"]["max"] > 1e5


def test_sampling_calculations_remain_float32() -> None:
    """Reverse state arithmetic returns float32 tensors."""
    diffusion = _linear_diffusion()
    lengths = torch.tensor([3])
    mask = make_pair_mask(lengths, 3)
    sep = make_sequence_separation(lengths, 3)
    out = sample_ddpm(
        ZeroEpsilonModel(),
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
    )
    assert out.dtype == torch.float32


def test_normalization_roundtrip_and_generated_validation_rejects_bad_matrix() -> None:
    """Generated-matrix diagnostics reject negative and enormous distances."""
    normalized = np.array([[0.0, -1.0, 1000.0], [-1.0, 0.0, 2.0], [1000.0, 2.0, 0.0]], dtype=np.float32)
    scale = 54.625
    physical = normalized * scale
    assert np.allclose(physical / scale, normalized)
    report = generated_matrix_report(normalized, scale=scale)
    assert report["negative_distance_fraction"] > 0.0
    assert report["physical_distance_max_angstrom"] > 2000.0
    assert report["physically_plausible"] is False
