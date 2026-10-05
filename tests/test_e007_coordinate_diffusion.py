from __future__ import annotations

import pytest
import torch

from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
    centered_coordinate_noise,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_losses import coordinate_diffusion_losses


def test_coordinate_noise_and_v_algebra() -> None:
    diffusion = CoordinateVPDiffusion(8)
    mask = torch.tensor([[True, True, True, False]])
    clean = center_coordinates(torch.randn(1, 4, 3), mask)
    batch = diffusion.make_training_batch(
        clean,
        mask,
        timesteps=torch.tensor([3]),
        generator=torch.Generator().manual_seed(4),
    )
    assert torch.allclose(batch.coordinate_noise[:, :3].sum(dim=1), torch.zeros((1, 3)), atol=1e-6)
    assert torch.allclose(batch.noisy_coordinates[:, :3].sum(dim=1), torch.zeros((1, 3)), atol=1e-6)
    reconstructed = diffusion.reconstruct_x0(batch.noisy_coordinates, batch.timesteps, batch.coordinate_v_target, mask)
    assert torch.allclose(reconstructed, clean, atol=2e-6)


def test_centered_noise_is_seeded_and_padding_is_zero() -> None:
    mask = torch.tensor([[True, True, False]])
    empty = torch.empty((1, 3, 3))
    left = centered_coordinate_noise(empty, mask, generator=torch.Generator().manual_seed(9))
    right = centered_coordinate_noise(empty, mask, generator=torch.Generator().manual_seed(9))
    assert torch.equal(left, right)
    assert torch.count_nonzero(left[:, 2:]) == 0


class _RadialModel:
    def __call__(self, coordinates, timesteps, lengths, residue_mask, continuity):
        return {"v_prediction": center_coordinates(coordinates * 0.01, residue_mask)}


def test_sampling_is_seeded_centered_and_returns_euclidean_matrices() -> None:
    diffusion = CoordinateVPDiffusion(4)
    first = diffusion.sample(_RadialModel(), length=9, seed=3)
    repeated = diffusion.sample(_RadialModel(), length=9, seed=3)
    other = diffusion.sample(_RadialModel(), length=9, seed=4)
    coordinates = first["coordinates"]
    distances = first["distance_matrix"]
    assert torch.equal(coordinates, repeated["coordinates"])
    assert not torch.equal(coordinates, other["coordinates"])
    assert torch.allclose(coordinates.mean(dim=1), torch.zeros((1, 3)), atol=1e-6)
    assert torch.isfinite(distances).all()
    assert torch.allclose(distances, distances.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(distances, dim1=-2, dim2=-1)) == 0
    assert torch.all(distances[:, :, None, :] <= distances[:, :, :, None] + distances[:, None, :, :] + 1e-5)
    squared = distances[0].square()
    gram = -0.5 * (squared - squared.mean(0)[None] - squared.mean(1)[:, None] + squared.mean())
    eigenvalues = torch.linalg.eigvalsh(gram)
    assert int((eigenvalues > eigenvalues.abs().max() * 1e-5).sum()) <= 3


def test_losses_are_finite_reported_and_ignore_padding() -> None:
    mask = torch.tensor([[True, True, True, False]])
    continuity = torch.tensor([[True, True, False]])
    clean = center_coordinates(torch.randn(1, 4, 3), mask)
    predicted = clean + 0.05 * center_coordinates(torch.randn(1, 4, 3), mask)
    target_v = torch.randn(1, 4, 3) * mask[..., None]
    prediction_v = target_v.clone()
    losses = coordinate_diffusion_losses(
        v_prediction=prediction_v,
        v_target=target_v,
        clean_coordinates=clean,
        predicted_clean_coordinates=predicted,
        residue_mask=mask,
        chain_continuity_mask=continuity,
    )
    assert set(losses) == {
        "total",
        "coordinate_v",
        "adjacent_distance_huber",
        "stratified_pair_distance",
        "soft_contact",
        "steric_clash",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["coordinate_v"] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_canonical_distance_matrix_promotes_and_zeroes_diagonal(dtype: torch.dtype) -> None:
    coordinates = (torch.randn((2, 31, 3)) * 10 + 1000).to(dtype)
    mask = torch.ones((2, 31), dtype=torch.bool)
    mask[0, 27:] = False
    distances = coordinates_to_distance_matrix(coordinates, mask)
    assert distances.dtype == torch.float32
    assert torch.count_nonzero(torch.diagonal(distances, dim1=-2, dim2=-1)) == 0
    assert torch.allclose(distances, distances.transpose(-1, -2), atol=0, rtol=0)
    assert torch.count_nonzero(distances[0, 27:]) == 0
    assert torch.count_nonzero(distances[0, :, 27:]) == 0


def test_canonical_distance_matrix_handles_coincident_translation_and_rotation() -> None:
    base = torch.tensor([[0.0, 0.0, 0.0], [1e-7, 0.0, 0.0], [2.0, 3.0, 4.0]], dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), dtype=torch.float64))
    translated = base @ rotation + 10_000.0
    left = coordinates_to_distance_matrix(base, diagnostic_float64=True)
    right = coordinates_to_distance_matrix(translated.double(), diagnostic_float64=True)
    assert torch.allclose(left, right, atol=1e-10, rtol=1e-10)
    assert left[0, 1] == pytest.approx(1e-7, rel=1e-6)
    assert torch.count_nonzero(torch.diagonal(right)) == 0


def test_canonical_distance_matrix_preserves_gradients() -> None:
    coordinates = torch.randn((1, 9, 3), requires_grad=True)
    mask = torch.ones((1, 9), dtype=torch.bool)
    distances = coordinates_to_distance_matrix(coordinates, mask)
    torch.triu(distances, diagonal=1).sum().backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert torch.count_nonzero(coordinates.grad) > 0


def test_old_cdist_diagonal_failure_is_corrected() -> None:
    torch.manual_seed(1)
    coordinates = torch.randn((1, 31, 3)) * 10 + 1000
    old = torch.cdist(coordinates, coordinates)
    corrected = coordinates_to_distance_matrix(coordinates)
    assert float(torch.diagonal(old, dim1=-2, dim2=-1).max()) >= 0.1
    assert float(torch.diagonal(corrected, dim1=-2, dim2=-1).max()) == 0.0
    assert torch.allclose(corrected, corrected.transpose(-1, -2), atol=0, rtol=0)
