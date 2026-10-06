import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protein_distance_diffusion.training import e010_local_feasibility_v7 as v7
from protein_distance_diffusion.training import e010_local_feasibility_v8 as v8


def geometry(n=12):
    t = torch.arange(n, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
    return dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))


def test_only_k_changes_solver_and_objective():
    root = Path(__file__).resolve().parents[1] / "configs"
    a = yaml.safe_load((root / "e010_phase4d_local_feasibility_v7.yaml").read_text())
    b = yaml.safe_load((root / "e010_phase4d_local_feasibility_v8.yaml").read_text())
    for k in ["solver", "constraints", "convergence", "gradient_validation", "objective", "objective_normalization"]:
        assert a[k] == b[k]
    assert a["recurrence"]["K"] == 4 and b["recurrence"]["K"] == 8
    assert a["recurrence"]["s_max_angstrom"] == b["recurrence"]["s_max_angstrom"] == 0.04


def test_zero_init_eight_states_and_bounds():
    b = geometry()
    z = torch.zeros(8, *b["pg"].shape, dtype=torch.float64)
    tr = v8.physical_trajectory(b["pg"], b["mask"], z)
    assert len(tr["states"]) == 9
    assert all(torch.equal(p, b["pg"]) for p in tr["states"])
    tr = v8.physical_trajectory(b["pg"], b["mask"], torch.full_like(z, 100))
    assert all(s["delta"].norm(dim=-1).max() < 0.04 for s in tr["steps"])
    final = tr["prediction"]
    mask = b["mask"]
    assert (final - b["pg"]).norm(dim=-1).max() <= 0.320001
    assert (torch.cdist(final[mask], final[mask]) - torch.cdist(b["pg"][mask], b["pg"][mask])).abs().max() <= 0.640002
    for t, p in enumerate(tr["states"]):
        row = v8.metric_row(p, b, dict(sample_id="synthetic", condition=50, stratum="20-64"), tr, t)
        assert row["path_length_max"] <= t * 0.04 + 1e-12
        assert row["displacement_max"] <= row["path_length_max"] + 1e-12
        assert row["step_correction_max"] <= 0.04
        assert row["frame_eligible"] == 10


@pytest.mark.parametrize("arm", ["A", "B"])
def test_constraint_parity_gradient_and_no_mutation(arm):
    b = geometry()
    before = b["pg"].clone()
    old = v7.Oracle(b, arm)
    new = v8.Oracle(b, arm)
    z = np.zeros(math.prod(new.shape), dtype=np.float64)
    assert new.shape[0] == 8
    np.testing.assert_array_equal(new.cfun(z), old.cfun(np.zeros(math.prod(old.shape))))
    f, g = new.fun(z)
    assert f == 1 and g.dtype == np.float64 and np.isfinite(g).all()
    d = np.sin(np.arange(len(z)) + 1)
    d /= np.linalg.norm(d)
    derivative = np.dot(g, d)
    fd = (new.fun(z + 1e-5 * d)[0] - new.fun(z - 1e-5 * d)[0]) / 2e-5
    assert abs(fd - derivative) <= 1e-8
    assert torch.equal(before, b["pg"]) and b["pg"].grad is None
    assert np.isfinite(new.hess(z) @ d).all()
    if arm == "B":
        assert np.isfinite(new.chess(z, [0.2, 0.3]) @ d).all()


def test_consecutive_cosines_and_saturation():
    b = geometry()
    z = torch.ones(8, *b["pg"].shape, dtype=torch.float64)
    tr = v8.physical_trajectory(b["pg"], b["mask"], z)
    x = v8.correction_telemetry(tr)
    assert len(x["consecutive_cosines"]) == 7 and len(x["steps"]) == 8
    for c in x["consecutive_cosines"]:
        assert c["defined"] == 10
        np.testing.assert_allclose(c["values"], 1, atol=1e-14)
    assert all(s["saturation"]["0.99"] == 1 for s in x["steps"])
    z[1::2] *= -1
    x = v8.correction_telemetry(v8.physical_trajectory(b["pg"], b["mask"], z))
    assert all(np.allclose(c["values"], -1) for c in x["consecutive_cosines"])
    x = v8.correction_telemetry(v8.physical_trajectory(b["pg"], b["mask"], z * 0))
    assert all(c["defined"] == 0 and c["values"] == [] for c in x["consecutive_cosines"])
