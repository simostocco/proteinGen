import subprocess

import numpy as np

from scripts.recover_e010_strict_scientific_v10 import BASE, INDICES, MODULE, NEW_LINE, OLD_LINE, original


def test_exact_single_telemetry_guard():
    old = subprocess.check_output(["git", "show", f"{BASE}:{MODULE}"], text=True)
    assert (original.ROOT / MODULE).read_text() == old.replace(OLD_LINE, NEW_LINE)
    assert len(INDICES) == len(set(INDICES)) == 17


def test_empty_active_set_guard_does_not_accept_stationarity():
    import torch

    from scripts.recover_e010_conditioning_v9b import synthetic

    _, config, limits = original.setup()
    b = synthetic(12)
    zero = torch.zeros((8, *b["pg"].shape), dtype=torch.float64, requires_grad=True)
    _, c, _, _ = original.v10.audit.physical(b, zero)
    gradient = torch.autograd.grad(c[1], zero)[0]
    delta = (-0.001 * gradient / gradient.norm(dim=-1).max()).detach()
    cert = original.v10.certificate(b, delta, np.zeros(len(c)), config, limits)
    assert cert["active_inequality_indices"] == [] and cert["active_balls"] == 0
    assert cert["maximum_linearized_cone_violation"] == 0
    assert not cert["converged"]
