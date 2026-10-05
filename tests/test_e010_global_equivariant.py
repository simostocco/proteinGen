import pytest
import torch

from protein_distance_diffusion.models.e010_global_equivariant import (
    GlobalEquivariantResidual,
    invariant_position_features,
)


def _small_model(seed=13):
    torch.manual_seed(seed)
    return GlobalEquivariantResidual(width=48, layers=2, heads=4, vector_channels=8, max_length=64)


def test_masking_zeroes_invalid_residuals_and_position_features():
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    coords = torch.randn(1, 6, 3)
    model = _small_model()
    result = model(coords, mask)
    assert torch.equal(result["delta"][~mask], torch.zeros_like(result["delta"][~mask]))
    assert torch.equal(result["prediction"][~mask], coords[~mask])
    features = invariant_position_features(mask)
    assert torch.equal(features[~mask], torch.zeros_like(features[~mask]))
    assert torch.allclose(features[0, :4, 0], torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))


@pytest.mark.parametrize("determinant", [1, -1])
def test_global_model_is_e3_equivariant_including_reflections(determinant):
    model = _small_model().eval()
    coords = torch.randn(1, 9, 3)
    mask = torch.ones(1, 9, dtype=torch.bool)
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) * determinant < 0:
        q[:, -1] *= -1
    shift = torch.tensor([1.5, -3.0, 0.25])
    with torch.no_grad():
        p = model(coords, mask)["prediction"]
        transformed = model(coords @ q + shift, mask)["prediction"]
    assert torch.allclose(transformed, p @ q + shift, atol=2e-5, rtol=2e-5)


def test_right_padding_matches_independent_variable_length_call():
    model = _small_model().eval()
    short = torch.randn(1, 5, 3)
    long = torch.randn(1, 8, 3)
    batch = torch.zeros(2, 8, 3)
    batch[0, :5] = short
    batch[1] = long
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0], [1] * 8], dtype=torch.bool)
    with torch.no_grad():
        alone = model(short, torch.ones(1, 5, dtype=torch.bool))["prediction"]
        together = model(batch, mask)["prediction"]
    assert torch.allclose(alone[0], together[0, :5], atol=2e-5, rtol=2e-5)
    assert torch.allclose(model(batch, mask)["delta"][0, 5:], torch.zeros(3, 3))


def test_global_model_has_finite_parameter_gradients_and_required_capacity():
    model = GlobalEquivariantResidual()
    count = sum(p.numel() for p in model.parameters())
    assert 1_000_000 <= count <= 3_000_000
    coords = torch.randn(1, 7, 3, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    result = model(coords, mask)
    loss = result["prediction"][mask].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert torch.equal(coords.grad[~mask], torch.zeros_like(coords.grad[~mask]))
