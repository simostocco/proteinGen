import torch

from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import E015MultiscaleParityDenoiser
from protein_distance_diffusion.training.e015_ema import make_ema


def small_model():
    torch.manual_seed(15)
    return E015MultiscaleParityDenoiser(24, 8, 4, 4, 2, 1, 1, k_current=5, k_self=4).eval()


def inputs():
    torch.manual_seed(123)
    x = torch.randn(1, 25, 3).cumsum(1)
    return x, torch.tensor([5]), torch.tensor([25]), torch.ones(1, 25, dtype=torch.bool)


def test_typed_trunk_full_o3_and_translation():
    model = small_model()
    x, t, _, _ = inputs()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    for det in (1, -1):
        q[:, 0] *= det * torch.det(q)
        original = model.trunk_single(x[0], t[0], x[0] * 0.8)[0]
        transformed = model.trunk_single(x[0] @ q.T + 7, t[0], x[0] * 0.8 @ q.T - 3)[0]
        assert torch.allclose(transformed.h, original.h, atol=2e-5, rtol=2e-5)
        assert torch.allclose(transformed.p, original.p * det, atol=2e-5, rtol=2e-5)
        assert torch.allclose(transformed.v, original.v @ q.T, atol=2e-5, rtol=2e-5)
        assert torch.allclose(transformed.a, original.a @ q.T * det, atol=2e-5, rtol=2e-5)


def test_complete_raw_ema_selfcondition_so3_translation():
    model = small_model()
    x, t, n, m = inputs()
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    r[:, 0] *= torch.det(r)
    for candidate in (model, make_ema(model)):
        for sc in (False, True):
            context = x * 0.8 if sc else None
            available = torch.tensor([sc])
            result = candidate(x, t, n, m, context, available)
            changed = candidate(x @ r.T + 12, t, n, m, None if context is None else context @ r.T - 4, available)
            assert torch.allclose(changed, result @ r.T, atol=2e-5, rtol=2e-5)
            assert result.mean(1).abs().max() < 1e-6


def test_enabled_gap_and_disabled_full_reflection():
    model = small_model()
    x, t, n, m = inputs()
    q = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    with torch.no_grad():
        model.chiral_a.weight.fill_(0.1)
        model.local[-1].weight.fill_(0.1)
    gap = (model(x @ q.T, t, n, m) - model(x, t, n, m) @ q.T).norm()
    assert gap > 1e-4
    model.chiral_enabled = False
    assert torch.allclose(model(x @ q.T, t, n, m), model(x, t, n, m) @ q.T, atol=2e-5, rtol=2e-5)


def test_small_chiral_initialization_allows_gradient():
    model = small_model()
    x, t, n, m = inputs()
    model(x, t, n, m).square().mean().backward()
    assert model.chiral_a.weight.grad.norm() > 0
    assert model.local[-1].weight.grad.norm() > 0
