import torch

from protein_distance_diffusion.training.e015_ema import make_ema, update_ema
from protein_distance_diffusion.training.e015_snr import SNRBinSampler, frozen_inverse_gradient_weights


def test_timestep_rng_exact_resume_and_coverage():
    sampler = SNRBinSampler()
    assert sorted(torch.cat(sampler.bins).tolist()) == list(range(500))
    sampler.sample(100)
    state = sampler.state_dict()
    expected = sampler.sample(101)
    resumed = SNRBinSampler()
    resumed.load_state_dict(state)
    actual = resumed.sample(101)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))
    assert torch.equal(sampler.histogram, resumed.histogram)
    assert resumed.timestep_histogram.sum() == 201


def test_bounded_weights_normalized_without_breaking_clip():
    weights = frozen_inverse_gradient_weights([1e-9, 1e9, 1, 2, 3, 4, 5, 6])
    assert weights.min() >= 0.5 and weights.max() <= 2
    assert abs(weights.mean().item() - 1) < 1e-7


def test_fp32_ema_arithmetic():
    model = torch.nn.Linear(2, 3).half()
    ema = make_ema(model)
    previous = ema.weight.clone()
    with torch.no_grad():
        model.weight.add_(1)
    update_ema(ema, model, 0.9)
    assert ema.weight.dtype == torch.float32 and not ema.weight.requires_grad
    assert torch.allclose(ema.weight, previous * 0.9 + model.weight.float() * 0.1)
