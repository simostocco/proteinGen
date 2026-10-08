import inspect

import pytest
import torch

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import (
    E013UnconditionalEquivariantDenoiser,
    VectorBlock,
    edges,
    local_geometry,
    radial_basis,
)
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss

torch.set_num_threads(1)


def inputs(n=20):
    return torch.randn(1, n, 3), torch.tensor([5]), torch.tensor([n]), torch.ones(1, n, dtype=torch.bool)


@pytest.mark.parametrize("n", [2, 3, 20, 500])
def test_shapes_finite_backward(n):
    model = E013UnconditionalEquivariantDenoiser()
    args = inputs(n)
    y = model(*args)
    assert y.shape == (1, n, 3) and torch.isfinite(y).all()
    assert y.mean(1).abs().max() < 1e-6
    y.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_symmetry_padding_parity():
    torch.manual_seed(13)
    model = E013UnconditionalEquivariantDenoiser().eval()
    x, t, length, mask = inputs()
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(r) < 0:
        r[:, 0] *= -1
    with torch.no_grad():
        y = model(x, t, length, mask)
        rotated = model(x @ r + 2, t, length, mask)
        torch.testing.assert_close(rotated, y @ r, atol=5e-5, rtol=5e-5)
        padded = torch.cat([x, torch.full((1, 9, 3), float("nan"))], 1)
        pm = torch.cat([mask, torch.zeros(1, 9, dtype=torch.bool)], 1)
        yp = model(padded, t, length, pm)
        torch.testing.assert_close(yp[:, :20], y, atol=0, rtol=0)
        assert yp[:, 20:].count_nonzero() == 0
        reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
        reflected = model(x @ reflection, t, length, mask)
        assert (reflected - y @ reflection).norm() > 1e-6
    assert list(inspect.signature(model.forward).parameters) == [
        "noisy_coordinates",
        "timesteps",
        "lengths",
        "residue_mask",
    ]
    with pytest.raises(ValueError):
        model(x, t, length - 1, mask)


def test_edges_rbf():
    x = torch.arange(20.0).reshape(-1, 1).expand(-1, 3)
    i, j, spatial = edges(x, k=2)
    assert len(torch.unique(i * 20 + j)) == len(i)
    assert ((i * 20 + j)[1:] > (i * 20 + j)[:-1]).all()
    for k in (1, 2, 3):
        assert ((j - i) == k).sum() == 20 - k
    for a, b in zip(edges(x, k=2), (i, j, spatial), strict=True):
        assert torch.equal(a, b)
    assert spatial.sum() == 40
    d = torch.linspace(0, 6, 100)
    centers = torch.linspace(0, 5, 16)
    gamma = torch.tensor(1.0 / float(centers[1] - centers[0]) ** 2)
    torch.testing.assert_close(
        radial_basis(d), torch.exp(-gamma * (d[:, None] - centers).square()), atol=1e-7, rtol=1e-6
    )


def test_local_frame_torsion_degeneracy():
    x = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0], [1.0, 1, 1], [2.0, 1, 1]])
    features, frame, reliability = local_geometry(x)
    assert features.shape == (5, 32)
    torch.testing.assert_close(frame[1].T @ frame[1], torch.eye(3))
    assert features[1, 25] == 1  # sin torsion +pi/2
    assert features[1, 26] == 0 and features[1, 27] == 1
    reflected, _, _ = local_geometry(x * torch.tensor([-1.0, 1, 1]))
    torch.testing.assert_close(reflected[:, 27], -features[:, 27])
    degenerate = torch.zeros(20, 3, requires_grad=True)
    f, frame, reliability = local_geometry(degenerate)
    assert torch.isfinite(f).all() and frame.count_nonzero() == 0 and reliability.count_nonzero() == 0
    model = E013UnconditionalEquivariantDenoiser()
    y = model(degenerate[None], torch.tensor([5]), torch.tensor([20]), torch.ones(1, 20, dtype=torch.bool))
    y.sum().backward()
    assert torch.isfinite(degenerate.grad).all()


def test_block_equivariance_and_diffusion_identity():
    x = torch.randn(20, 3)
    h, v, time = torch.randn(20, 128), torch.randn(20, 32, 3), torch.randn(64)
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    block = VectorBlock()
    graph = edges(x)
    ha, va = block(h, v, x, time, graph)
    hb, vb = block(h, v @ r, x @ r, time, graph)
    torch.testing.assert_close(ha, hb, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(va @ r, vb, atol=5e-5, rtol=5e-5)
    diffusion = CoordinateVPDiffusion(500)
    clean, _, lengths, mask = inputs()
    t = torch.tensor([250])
    batch = diffusion.make_training_batch(clean, mask, timesteps=t)
    a, s = diffusion.alpha_sigma(t, clean)
    torch.testing.assert_close(
        batch.coordinate_v_target, a * batch.coordinate_noise - s * center_coordinates(clean, mask)
    )
    torch.testing.assert_close(
        diffusion.reconstruct_x0(batch.noisy_coordinates, t, batch.coordinate_v_target, mask),
        center_coordinates(clean, mask),
    )
    assert coordinate_v_loss(batch.coordinate_v_target, batch.coordinate_v_target, mask) == 0
