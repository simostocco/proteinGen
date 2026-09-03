"""Tests for stochastic EDM spectral auxiliary loss."""

from __future__ import annotations

import pytest
import torch

from protein_distance_diffusion.data.collate import make_pair_mask
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, masked_upper_triangular_loss
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.training.physical_auxiliary import (
    AdjacentAuxiliaryLossConfig,
    AdjacentAuxiliaryLossResult,
    PhysicalAuxiliaryLossConfig,
    adjacent_auxiliary_config_from_mapping,
    adjacent_auxiliary_weight,
    adjacent_chain_smooth_l1_loss,
    deterministic_subset_indices,
    physical_auxiliary_config_from_mapping,
    physical_auxiliary_weight,
    stochastic_edm_spectral_loss,
)


def _distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coords.float(), coords.float()).unsqueeze(0).unsqueeze(0)


def _loss(
    matrix: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    subset_size: int = 8,
    subsets_per_sample: int = 1,
    sample_ids: list[str] | None = None,
) -> torch.Tensor:
    lengths = lengths if lengths is not None else torch.tensor([matrix.shape[-1]])
    mask = make_pair_mask(lengths, matrix.shape[-1]).to(matrix.device)
    result = stochastic_edm_spectral_loss(
        x0_hat_normalized=matrix,
        pair_mask=mask,
        lengths=lengths.to(matrix.device),
        normalization_scale=1.0,
        config=PhysicalAuxiliaryLossConfig(
            enabled=True,
            subset_size=subset_size,
            subsets_per_sample=subsets_per_sample,
            seed=123,
        ),
        sample_ids=sample_ids or ["sample"],
        optimizer_step=5,
        microbatch=2,
    )
    return result.loss


def test_valid_3d_edm_has_near_zero_loss() -> None:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
        ]
    )
    assert float(_loss(_distance_matrix(coords), subset_size=6)) < 1e-10


def test_non_edm_matrix_has_negative_spectrum_penalty() -> None:
    matrix = (
        torch.tensor(
            [
                [0.0, 1.0, 1.0, 4.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [4.0, 1.0, 1.0, 0.0],
            ]
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    mask = make_pair_mask(torch.tensor([4]), 4)
    result = stochastic_edm_spectral_loss(
        x0_hat_normalized=matrix,
        pair_mask=mask,
        lengths=torch.tensor([4]),
        normalization_scale=1.0,
        config=PhysicalAuxiliaryLossConfig(enabled=True, subset_size=4),
    )
    assert float(result.negative_loss) > 0.01


def test_valid_4d_edm_has_rank_tail_penalty() -> None:
    coords = torch.eye(5)
    matrix = _distance_matrix(coords)
    mask = make_pair_mask(torch.tensor([5]), 5)
    result = stochastic_edm_spectral_loss(
        x0_hat_normalized=matrix,
        pair_mask=mask,
        lengths=torch.tensor([5]),
        normalization_scale=1.0,
        config=PhysicalAuxiliaryLossConfig(enabled=True, subset_size=5),
    )
    assert float(result.negative_loss) < 1e-8
    assert float(result.rank3_loss) > 0.01


def test_masking_and_padding_invariance() -> None:
    coords = torch.randn(6, 3)
    base = _distance_matrix(coords)
    padded = torch.zeros(1, 1, 10, 10)
    padded[:, :, :6, :6] = base
    padded[:, :, 6:, :] = 999.0
    padded[:, :, :, 6:] = 999.0
    lengths = torch.tensor([6])
    assert torch.allclose(
        _loss(base, lengths=lengths, subset_size=6),
        _loss(padded, lengths=lengths, subset_size=6),
        atol=1e-10,
        rtol=0.0,
    )


def test_variable_length_batch_and_diagnostics() -> None:
    matrices = torch.zeros(2, 1, 7, 7)
    matrices[0, :, :5, :5] = _distance_matrix(torch.randn(5, 3))
    matrices[1, :, :7, :7] = _distance_matrix(torch.randn(7, 3))
    lengths = torch.tensor([5, 7])
    result = stochastic_edm_spectral_loss(
        x0_hat_normalized=matrices,
        pair_mask=make_pair_mask(lengths, 7),
        lengths=lengths,
        normalization_scale=1.0,
        config=PhysicalAuxiliaryLossConfig(enabled=True, subset_size=6, subsets_per_sample=2),
        sample_ids=["a", "b"],
    )
    assert torch.isfinite(result.loss)
    assert result.eligible_fraction == 1.0
    assert result.subset_count == 4
    assert result.mean_subset_size == 5.5


def test_deterministic_subset_sampling_without_global_rng_side_effects() -> None:
    torch.manual_seed(99)
    before = torch.get_rng_state()
    first = deterministic_subset_indices(
        length=20,
        subset_size=8,
        base_seed=7,
        sample_id="x",
        optimizer_step=3,
        microbatch=1,
        subset_index=0,
        device=torch.device("cpu"),
    )
    second = deterministic_subset_indices(
        length=20,
        subset_size=8,
        base_seed=7,
        sample_id="x",
        optimizer_step=3,
        microbatch=1,
        subset_index=0,
        device=torch.device("cpu"),
    )
    assert torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), before)


def test_symmetry_and_transpose_invariance() -> None:
    matrix = torch.rand(1, 1, 8, 8)
    matrix = matrix + 0.25 * matrix.transpose(-1, -2)
    loss_a = _loss(matrix, subset_size=8)
    loss_b = _loss(matrix.transpose(-1, -2), subset_size=8)
    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)


def test_finite_gradients_cpu_float32() -> None:
    matrix = _distance_matrix(torch.randn(6, 3)).requires_grad_(True)
    loss = _loss(matrix, subset_size=6)
    loss.backward()
    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()


def test_cpu_bfloat16_autocast_when_supported() -> None:
    if not torch.backends.cpu.get_cpu_capability():
        pytest.skip("CPU capability metadata unavailable")
    matrix = _distance_matrix(torch.randn(6, 3)).requires_grad_(True)
    try:
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            loss = _loss(matrix, subset_size=6)
    except RuntimeError as exc:
        pytest.skip(f"CPU bfloat16 autocast does not support this operation: {exc}")
    loss.backward()
    assert torch.isfinite(matrix.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_float16_autocast() -> None:
    matrix = _distance_matrix(torch.randn(8, 3, device="cuda")).requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = _loss(matrix, lengths=torch.tensor([8], device="cuda"), subset_size=8)
    loss.backward()
    assert torch.isfinite(matrix.grad).all()


def test_config_validation_and_weight_warmup() -> None:
    cfg = physical_auxiliary_config_from_mapping(
        {
            "physical_auxiliary_loss_enabled": True,
            "physical_auxiliary_loss_weight": 0.2,
            "physical_auxiliary_loss_warmup_steps": 4,
            "edm_subset_size": 5,
        }
    )
    assert physical_auxiliary_weight(cfg, 0) == pytest.approx(0.05)
    assert physical_auxiliary_weight(cfg, 3) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="edm_subset_size"):
        physical_auxiliary_config_from_mapping({"edm_subset_size": 3})
    with pytest.raises(ValueError, match="physical_auxiliary_loss_weight"):
        physical_auxiliary_config_from_mapping({"physical_auxiliary_loss_weight": -1.0})


def test_weight_zero_preserves_diffusion_objective_and_gradients() -> None:
    torch.manual_seed(11)
    clean = _distance_matrix(torch.randn(6, 3)).requires_grad_(True)
    mask = make_pair_mask(torch.tensor([6]), 6)
    diffusion = GaussianDiffusion(cosine_beta_schedule(4))
    t = torch.tensor([2])
    noisy, eps = diffusion.q_sample(clean, t, mask, noise=torch.randn_like(clean))
    prediction = torch.randn_like(noisy, requires_grad=True)
    target = diffusion.training_target(x_start=clean, t=t, epsilon=eps, prediction_type="v")
    diffusion_loss = masked_upper_triangular_loss(target, prediction, mask)
    x0_hat, _ = diffusion.predict_x0_epsilon_from_model_output(
        x_t=noisy,
        t=t,
        model_output=prediction,
        prediction_type="v",
    )
    auxiliary = stochastic_edm_spectral_loss(
        x0_hat_normalized=x0_hat,
        pair_mask=mask,
        lengths=torch.tensor([6]),
        normalization_scale=10.0,
        config=PhysicalAuxiliaryLossConfig(enabled=True, weight=0.0, subset_size=6),
    )
    total = diffusion_loss + 0.0 * auxiliary.loss
    grad_diffusion = torch.autograd.grad(diffusion_loss, prediction, retain_graph=True)[0]
    grad_total = torch.autograd.grad(total, prediction)[0]
    assert torch.allclose(grad_total, grad_diffusion, atol=0.0, rtol=0.0)


def _adjacent_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> AdjacentAuxiliaryLossResult:
    lengths = lengths if lengths is not None else torch.tensor([predicted.shape[-1]])
    mask = mask if mask is not None else make_pair_mask(lengths, predicted.shape[-1]).to(predicted.device)
    return adjacent_chain_smooth_l1_loss(
        x0_hat_normalized=predicted,
        clean_normalized=target,
        pair_mask=mask,
        lengths=lengths.to(predicted.device),
        normalization_scale=1.0,
        config=AdjacentAuxiliaryLossConfig(enabled=True, huber_beta_angstrom=0.25),
    )


def test_adjacent_loss_zero_when_predicted_matches_target() -> None:
    matrix = _distance_matrix(torch.randn(6, 3))
    result = _adjacent_loss(matrix, matrix)
    assert torch.allclose(result.loss, torch.tensor(0.0))
    assert result.eligible_pair_count == 5
    assert result.eligible_fraction == 1.0


def test_adjacent_loss_positive_for_adjacent_perturbation_and_counts_once() -> None:
    target = _distance_matrix(torch.randn(5, 3))
    predicted = target.clone()
    predicted[:, :, 1, 2] += 1.0
    predicted[:, :, 2, 1] += 1.0
    result = _adjacent_loss(predicted, target)
    expected_one_bond = 1.0 - 0.5 * 0.25
    assert result.loss.item() == pytest.approx(expected_one_bond / 4.0)
    assert result.eligible_pair_count == 4


def test_adjacent_loss_ignores_non_adjacent_perturbation() -> None:
    target = _distance_matrix(torch.randn(6, 3))
    predicted = target.clone()
    predicted[:, :, 0, 3] += 10.0
    predicted[:, :, 3, 0] += 10.0
    assert torch.allclose(_adjacent_loss(predicted, target).loss, torch.tensor(0.0))


def test_adjacent_loss_padding_invariance_and_variable_lengths() -> None:
    base = torch.zeros(2, 1, 7, 7)
    base[0, :, :4, :4] = _distance_matrix(torch.randn(4, 3))
    base[1, :, :7, :7] = _distance_matrix(torch.randn(7, 3))
    predicted = base.clone()
    predicted[:, :, 6:, :] = 999.0
    predicted[:, :, :, 6:] = 999.0
    lengths = torch.tensor([4, 6])
    assert torch.allclose(_adjacent_loss(predicted, base, lengths=lengths).loss, torch.tensor(0.0))


def test_adjacent_loss_excludes_masked_adjacent_pairs() -> None:
    target = _distance_matrix(torch.randn(5, 3))
    predicted = target.clone()
    predicted[:, :, 1, 2] += 1.0
    predicted[:, :, 2, 1] += 1.0
    mask = make_pair_mask(torch.tensor([5]), 5)
    mask[:, :, 1, 2] = False
    mask[:, :, 2, 1] = False
    result = _adjacent_loss(predicted, target, mask=mask)
    assert torch.allclose(result.loss, torch.tensor(0.0))
    assert result.eligible_pair_count == 3
    assert result.eligible_fraction == 0.75


def test_adjacent_loss_safe_zero_when_no_pair_is_eligible() -> None:
    matrix = _distance_matrix(torch.randn(1, 3)).requires_grad_(True)
    result = _adjacent_loss(matrix, matrix, lengths=torch.tensor([1]))
    result.loss.backward()
    assert torch.allclose(result.loss.detach(), torch.tensor(0.0))
    assert matrix.grad is not None
    assert torch.count_nonzero(matrix.grad) == 0


def test_adjacent_loss_finite_gradients_cpu_float32() -> None:
    target = _distance_matrix(torch.randn(6, 3))
    predicted = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    result = _adjacent_loss(predicted, target)
    result.loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_adjacent_loss_cpu_bfloat16_autocast_when_supported() -> None:
    target = _distance_matrix(torch.randn(6, 3))
    predicted = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    try:
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            loss = _adjacent_loss(predicted, target).loss
    except RuntimeError as exc:
        pytest.skip(f"CPU bfloat16 autocast does not support this operation: {exc}")
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_adjacent_loss_cuda_float16_autocast() -> None:
    target = _distance_matrix(torch.randn(8, 3, device="cuda"))
    predicted = (target + 0.1 * torch.randn_like(target)).requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        loss = _adjacent_loss(predicted, target, lengths=torch.tensor([8], device="cuda")).loss
    loss.backward()
    assert torch.isfinite(predicted.grad).all()


def test_adjacent_config_validation_and_weight_warmup() -> None:
    cfg = adjacent_auxiliary_config_from_mapping(
        {
            "adjacent_auxiliary_loss_enabled": True,
            "adjacent_auxiliary_loss_weight": 0.3,
            "adjacent_auxiliary_loss_warmup_steps": 3,
            "adjacent_auxiliary_huber_beta_angstrom": 0.25,
        }
    )
    assert adjacent_auxiliary_weight(cfg, 0) == pytest.approx(0.1)
    assert adjacent_auxiliary_weight(cfg, 2) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="adjacent_auxiliary_loss_weight"):
        adjacent_auxiliary_config_from_mapping({"adjacent_auxiliary_loss_weight": -1.0})
    with pytest.raises(ValueError, match="adjacent_auxiliary_huber_beta_angstrom"):
        adjacent_auxiliary_config_from_mapping({"adjacent_auxiliary_huber_beta_angstrom": 0.0})


def test_adjacent_weight_zero_preserves_e002_objective_and_gradients() -> None:
    torch.manual_seed(17)
    clean = _distance_matrix(torch.randn(6, 3))
    mask = make_pair_mask(torch.tensor([6]), 6)
    diffusion = GaussianDiffusion(cosine_beta_schedule(4))
    t = torch.tensor([2])
    noisy, eps = diffusion.q_sample(clean, t, mask, noise=torch.randn_like(clean))
    prediction = torch.randn_like(noisy, requires_grad=True)
    target = diffusion.training_target(x_start=clean, t=t, epsilon=eps, prediction_type="v")
    diffusion_loss = masked_upper_triangular_loss(target, prediction, mask)
    x0_hat, _ = diffusion.predict_x0_epsilon_from_model_output(
        x_t=noisy,
        t=t,
        model_output=prediction,
        prediction_type="v",
    )
    spectral = stochastic_edm_spectral_loss(
        x0_hat_normalized=x0_hat,
        pair_mask=mask,
        lengths=torch.tensor([6]),
        normalization_scale=10.0,
        config=PhysicalAuxiliaryLossConfig(enabled=True, subset_size=6),
    )
    adjacent = adjacent_chain_smooth_l1_loss(
        x0_hat_normalized=x0_hat,
        clean_normalized=clean,
        pair_mask=mask,
        lengths=torch.tensor([6]),
        normalization_scale=10.0,
        config=AdjacentAuxiliaryLossConfig(enabled=True, weight=0.0),
    )
    e002 = diffusion_loss + 0.01 * spectral.loss
    e003_zero = e002 + 0.0 * adjacent.loss
    grad_e002 = torch.autograd.grad(e002, prediction, retain_graph=True)[0]
    grad_e003_zero = torch.autograd.grad(e003_zero, prediction)[0]
    assert torch.allclose(grad_e003_zero, grad_e002, atol=0.0, rtol=0.0)
