import numpy as np
import torch

from protein_distance_diffusion.training import e010_strict_scientific_v10 as v10
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_strict_scientific_v10 import V9B, setup


def test_frozen_stationarity_formula():
    _, cfg, limits = setup()
    assert v10.threshold(cfg, 80) == 4.472135954999578e-5
    assert v10.threshold(cfg, 240) == 2.581988897471611e-5
    assert limits["normalized_complementarity_max"] == 1e-6
    assert limits["dual_negativity_max"] == 1e-8


def test_recovered_control_physical_contract():
    _, cfg, limits = setup()
    for n in [12, 32]:
        saved = np.load(V9B / f"recovered_length_{n}.npz")
        b = synthetic(n)
        delta = torch.from_numpy(saved["delta"].copy())
        screen = v10.physical_screen(b, delta, saved["multipliers"], cfg)
        cert = v10.certificate(b, delta, saved["multipliers"], cfg, limits)
        assert screen["candidate"] and cert["converged"] and all(cert["gates"].values())
        np.testing.assert_allclose(
            screen["physical_normalized_stationarity_l2"],
            cert["physical"]["physical_normalized_stationarity_l2"],
            atol=1e-14,
        )
        assert cert["derivative_uncertainty_normalized"] <= cert["stationarity_threshold"] / 10


def test_zero_is_not_stationary_and_frozen_inputs():
    _, cfg, _ = setup()
    b = synthetic(12)
    o = v10.v9.Oracle(b)
    delta = torch.zeros(o.shape, dtype=torch.float64)
    screen = v10.physical_screen(b, delta, np.zeros(len(o.quartets(b["pg"])) + 2), cfg)
    assert not screen["candidate"] and screen["feasible"]
    torch.testing.assert_close(b["pg"], synthetic(12)["pg"], atol=0, rtol=0)
