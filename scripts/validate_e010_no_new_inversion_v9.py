"""Read-only exact evaluator/constraint validation for all fixed Pg baselines."""

import numpy as np
import torch

from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from protein_distance_diffusion.training.e010_precision_kkt_v6 import directional_checks
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_no_new_inversion_v9 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache, write

cfg = setup()
cache, manifest = load_cache()
records = []
for i in range(60):
    b = {k: v.double() if v.is_floating_point() else v for k, v in batch(cache, [i], "cpu").items()}
    oracle = v9.Oracle(b)
    zero = np.zeros(np.prod(oracle.shape), dtype=np.float64)
    assert (oracle.cfun(zero)[2:] <= 0).all()
    q = oracle.quartets.telemetry(b["pg"], cfg["constraints"]["active_tolerance"])
    assert q["new_inversions"] == q["assessability_lost"] == 0
    j = oracle.cjac(zero)
    errors = []
    for phase in (0.0, 0.7, 1.3):
        d = np.sin(np.arange(len(zero)) + 1 + phase)
        d /= np.linalg.norm(d)
        expected = j @ d
        trials = []
        for epsilon_angstrom in cfg["gradient_validation"]["epsilons_angstrom"]:
            epsilon = epsilon_angstrom / 0.04
            fd = (oracle.cfun(zero + epsilon * d) - oracle.cfun(zero - epsilon * d)) / (2 * epsilon)
            scaled = np.abs(fd - expected) / (1e-8 + 1e-5 * np.abs(expected))
            trials.append(
                dict(epsilon_angstrom=epsilon_angstrom, epsilon_z=epsilon, maximum_scaled_error=float(scaled.max()))
            )
        assert any(t["maximum_scaled_error"] <= 1 for t in trials), f"baseline {i} constraint gradient failed"
        errors.append(trials)
    d = np.sin(np.arange(len(zero)) + 1)
    assert np.isfinite(oracle.hess(zero) @ d).all()
    assert np.isfinite(oracle.chess(zero, np.full(len(oracle.cfun(zero)), 0.1)) @ d).all()
    p = b["pg"].clone().requires_grad_()

    def aligned(x, data=b):
        return v9.v8.aligned_rmsd(x, data["target"], data["mask"])

    g = torch.autograd.grad(aligned(p), p)[0]
    checks = directional_checks(aligned, p.detach(), g, cfg["gradient_validation"]["epsilons_angstrom"])
    assert all(any(c["passed"] for c in checks if c["direction"] == k) for k in range(3))
    records.append(
        dict(
            index=i,
            record=cache["records"][i],
            baseline_quartets=q,
            constraint_directional_scaled_errors=errors,
            aligned_checks=checks,
            zero_hvp_finite=True,
        )
    )
    print(
        i,
        q["correct"],
        q["inverted"],
        max(min(t["maximum_scaled_error"] for t in trials) for trials in errors),
        flush=True,
    )
assert_file_pins(manifest["protected_input_sha256"])
write(
    OUT / "baseline_constraint_preflight.json",
    dict(
        all_passed=True,
        examples=60,
        constraint_directions=180,
        aligned_directions=180,
        assessable_quartets=sum(r["baseline_quartets"]["assessable"] for r in records),
        records=records,
        protected_hashes_verified=True,
        scientific_optimization_launched=False,
    ),
)
