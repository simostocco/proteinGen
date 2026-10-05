import math

import torch

from protein_distance_diffusion.models.e009_bayesian_refiner import (
    BayesianSE3Refiner,
    GeometryPrior,
    normalized_refiner_losses,
)


def _trace(n=18):
    t = torch.arange(n, dtype=torch.float32)
    return torch.stack((2 * torch.cos(t * 0.7), 2 * torch.sin(t * 0.7), t * 0.8), -1)[None]


def test_se3_equivariant_mean_and_invariant_scale():
    torch.manual_seed(4)
    model = BayesianSE3Refiner(width=32, layers=6, max_length=64).eval()
    x = _trace()
    mask = torch.ones((1, x.shape[1]), dtype=torch.bool)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    q[:, 0] *= torch.linalg.det(q)
    shift = torch.tensor([3.0, -2.0, 1.0])
    a = model(x, mask)
    b = model(x @ q + shift, mask)
    assert torch.allclose(b["mean"], a["mean"] @ q + shift, atol=2e-5)
    assert torch.allclose(b["sigma"], a["sigma"], atol=2e-5)


def test_prior_log_prob_normalized_and_torsion_periodic():
    p = GeometryPrior(3)
    xs = torch.linspace(-math.pi, math.pi, 2001)[:-1]
    lp = p.log_prob(torch.full_like(xs, 3.82), torch.full_like(xs, 1.8), xs)["torsion"]
    assert torch.allclose(
        torch.logsumexp(lp, 0) - math.log(len(xs)) + math.log(2 * math.pi), torch.tensor(0.0), atol=0.02
    )
    one = p.log_prob(torch.tensor([3.7]), torch.tensor([1.4]), torch.tensor([0.2]))["torsion"]
    two = p.log_prob(torch.tensor([3.7]), torch.tensor([1.4]), torch.tensor([0.2 + 2 * math.pi]))["torsion"]
    assert torch.allclose(one, two, atol=1e-5)


def test_mask_variable_length_and_reparameterized_gradients():
    model = BayesianSE3Refiner(width=32, layers=6, max_length=64)
    prior = GeometryPrior(3)
    x = torch.zeros((2, 20, 3))
    x[0] = _trace(20)[0]
    x[1, :13] = _trace(13)[0]
    mask = torch.zeros((2, 20), dtype=torch.bool)
    mask[0] = True
    mask[1, :13] = True
    out = model(x, mask)
    assert (out["sigma"] >= 0.03).all() and (out["sigma"] <= 3).all()
    target = x.detach().clone().requires_grad_(False)
    losses = normalized_refiner_losses(out, target, prior, mask)
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(prior.parameters())
    )
    assert model.scale.weight.grad is not None


def test_scale_bounds_and_overlength_rejected():
    model = BayesianSE3Refiner(width=32, layers=6, max_length=16)
    out = model(_trace(16), torch.ones((1, 16), dtype=torch.bool))
    assert bool((out["sigma"] > 0.03).all() and (out["sigma"] < 3).all())
    try:
        model(_trace(17), torch.ones((1, 17), dtype=torch.bool))
    except ValueError as exc:
        assert "max_length" in str(exc)
    else:
        raise AssertionError("length overflow accepted")
