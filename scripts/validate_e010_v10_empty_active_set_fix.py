"""Validate a prepared telemetry-only patch in memory, without scientific solves."""

import inspect
import json

import numpy as np
import torch

from protein_distance_diffusion.training import e010_strict_scientific_v10 as v10
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_strict_scientific_v10 import OUT, V9B, setup, write


def main():
    _, cfg, limits = setup()
    old = inspect.getsource(v10.certificate)
    new = old.replace(
        "maximum_linearized_cone_violation=float(np.max(a @ direction)),",
        "maximum_linearized_cone_violation=float(np.max(a @ direction)) if len(a) else 0.0,",
    )
    assert old != new
    namespace = vars(v10).copy()
    exec(compile(new, "<prepared-empty-active-set-telemetry-fix>", "exec"), namespace)
    fixed_certificate = namespace["certificate"]
    b = synthetic(12)
    zero = torch.zeros((8, *b["pg"].shape), dtype=torch.float64, requires_grad=True)
    _, c, _, _ = v10.audit.physical(b, zero)
    gradient = torch.autograd.grad(c[1], zero)[0]
    delta = (-0.001 * gradient / gradient.norm(dim=-1).max()).detach()
    _, _, constraints, tr, _, g, j = v10.audit.physical_jacobians(b, delta)
    a, ids, balls = v10.audit.active_system(
        delta.numpy(),
        tr["eligible"].numpy(),
        constraints.detach().numpy(),
        j,
        cfg["active_normalized_slack"],
        cfg["boundary_relative_tolerance"],
    )
    assert len(a) == len(ids) == len(balls) == 0
    original_failed = False
    try:
        v10.certificate(b, delta, np.zeros(len(constraints)), cfg, limits)
    except ValueError as exc:
        assert "zero-size array" in str(exc)
        original_failed = True
    assert original_failed
    certificate = fixed_certificate(b, delta, np.zeros(len(constraints)), cfg, limits)
    assert certificate["maximum_linearized_cone_violation"] == 0.0
    assert not certificate["converged"]
    np.testing.assert_allclose(certificate["projected_feasible_gradient_norm"], np.linalg.norm(0.04 * g))
    controls = []
    for n in [12, 32]:
        saved = np.load(V9B / f"recovered_length_{n}.npz")
        args = synthetic(n), torch.from_numpy(saved["delta"].copy()), saved["multipliers"], cfg, limits
        original = v10.certificate(*args)
        fixed = fixed_certificate(*args)
        assert json.loads(json.dumps(original)) == json.loads(json.dumps(fixed))
        controls.append(dict(length=n, bitwise_canonical_certificate_parity=True))
    write(
        OUT / "prepared_empty_active_set_fix_validation.json",
        dict(
            empty_active_set_original_error_reproduced=True,
            empty_active_set_telemetry_zero=True,
            empty_active_set_not_silently_accepted=True,
            nonempty_controls=controls,
            patch_applied_to_worktree=False,
            scientific_optimization_launched=False,
            recovery_launched=False,
            cuda_used=False,
        ),
    )
    print("Prepared telemetry-only fix validated: empty active set and exact nonempty-control parity")


if __name__ == "__main__":
    main()
