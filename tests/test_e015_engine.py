import copy

import torch

from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import E015MultiscaleParityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates
from protein_distance_diffusion.training.e015_engine import Engine


def small_engine():
    cfg = dict(
        seed=15,
        diffusion_steps=500,
        model={"self_conditioning": True},
        optimizer=dict(learning_rate=0.0002, weight_decay=0.01, gradient_clip=1.0),
        features=dict(rollout_states=True, geometry_losses=True, snr_weighting=True, pcgrad=True, ema=True),
        ema_decay=0.9999,
        rollout_probability=0.1,
        rollout_warmup_exposures=115066,
        self_conditioning_probability=0.5,
    )
    calibration = dict(
        midpoints=dict(distance=0.0, torsion=0.0, chirality=0.0, clash=0.0),
        gate_width_logsnr=1.0,
        v_weights=[1.0] * 8,
        coefficients=dict(distance=0.15, torsion=0.1, chirality=0.1, clash=0.05),
    )
    model = E015MultiscaleParityDenoiser(24, 8, 4, 4, 1, 1, 1, k_current=4, k_self=4)
    return Engine(cfg, calibration, "cpu", model)


def update(engine, x, mask, force=False):
    engine.begin()
    result = engine.microbatch(
        x, mask.sum(1), mask, 0, 1.0, force_sc=True if force else None, force_rollout=True if force else None
    )
    gradient = engine.finish()
    return result, gradient


def test_integrated_step_and_exact_resume():
    torch.manual_seed(5)
    engine = small_engine()
    mask = torch.ones(2, 12, dtype=torch.bool)
    x = center_coordinates(torch.randn(2, 12, 3), mask)
    result, gradient = update(engine, x, mask, True)
    assert result["self_conditioned"] and result["rollout"] and result["rollout_steps"] in (1, 2, 3)
    assert torch.isfinite(torch.tensor(gradient["combined_preclip_norm"]))
    state = copy.deepcopy(
        dict(
            model=engine.model.state_dict(),
            ema=engine.ema.state_dict(),
            optimizer=engine.optimizer.state_dict(),
            rng=engine.rng_state(),
        )
    )
    expected = update(engine, x, mask)
    resumed = small_engine()
    resumed.model.load_state_dict(state["model"])
    resumed.ema.load_state_dict(state["ema"])
    resumed.optimizer.load_state_dict(state["optimizer"])
    resumed.restore_rng(state["rng"])
    actual = update(resumed, x, mask)
    assert expected == actual
    for a, b in zip(engine.model.parameters(), resumed.model.parameters(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(engine.ema.parameters(), resumed.ema.parameters(), strict=True):
        assert torch.equal(a, b)
