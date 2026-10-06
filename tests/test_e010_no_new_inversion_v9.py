import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import EPS, PSEUDOSCALAR_INDEX, local_representation
from protein_distance_diffusion.training import e010_local_feasibility_v8 as v8
from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from protein_distance_diffusion.training.e010_phase4d_diagnostic import signed_status


def geometry(n=12):
    t = torch.arange(n, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
    return dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))


def test_threshold_is_weakest_float64_and_baseline_feasible():
    assert v9.THRESHOLD > EPS and np.nextafter(v9.THRESHOLD, -np.inf) == EPS
    b = geometry()
    policy = v9.QuartetConstraints(b)
    assert bool((policy(b["pg"]) <= 0).all())
    q, _, _ = v9.quantities(b["pg"])
    torch.testing.assert_close(
        q, local_representation(b["pg"], b["mask"])["features"][:, 1:-2, PSEUDOSCALAR_INDEX], atol=0, rtol=0
    )
    assert policy.telemetry(b["pg"], 1e-5)["assessability_lost"] == 0


def test_sign_crossing_violates_and_inverted_quartets_free_to_repair():
    b = geometry()
    b["target"] = b["pg"] * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    policy = v9.QuartetConstraints(b)
    assert policy.counts["correct"] == 0 and policy.counts["inverted"] == 9
    repaired = b["target"]
    assert bool((policy(repaired) <= 0).all())
    assert policy.telemetry(repaired, 1e-5)["repaired_inversions"] == 9
    b["target"] = b["pg"].clone()
    policy = v9.QuartetConstraints(b)
    assert policy.counts["correct"] == 9 and policy.counts["inverted"] == 0
    reflected = b["pg"] * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    assert (policy(reflected)[:9] > 0).all()
    assert policy.telemetry(reflected, 1e-5)["new_inversions"] == 9


def test_assessability_cannot_hide_sign_or_collapsed_bonds():
    b = geometry()
    policy = v9.QuartetConstraints(b)
    collapsed = b["pg"].clone()
    collapsed[:, 2] = collapsed[:, 1]
    assert (policy(collapsed) > 0).any()
    p = b["pg"].clone()
    p[:, :, 2] = 0
    assert (policy(p) > 0).any()
    assert policy.telemetry(p, 1e-5)["assessability_lost"] > 0


def test_finite_difference_sparse_jacobian_exact_chain_and_hessian():
    oracle = v9.Oracle(geometry())
    z = np.zeros(math.prod(oracle.shape), dtype=np.float64)
    d = np.sin(np.arange(len(z)) + 1)
    d /= np.linalg.norm(d)
    j = oracle.cjac(z)
    finite = (oracle.cfun(z + 1e-5 * d) - oracle.cfun(z - 1e-5 * d)) / 2e-5
    np.testing.assert_allclose(j @ d, finite, atol=1e-8, rtol=1e-5)
    p = oracle.point(z)
    exact = torch.stack([torch.autograd.grad(c, p.z, retain_graph=True)[0] for c in p.constraints]).detach().numpy()
    np.testing.assert_allclose(j.toarray(), exact, atol=1e-12, rtol=1e-12)
    mu = np.linspace(0.1, 0.3, len(finite))
    h = oracle.chess(z, mu) @ d
    finite = (mu @ oracle.cjac(z + 1e-5 * d) - mu @ oracle.cjac(z - 1e-5 * d)) / 2e-5
    np.testing.assert_allclose(h, finite, atol=1e-7, rtol=1e-5)


def test_correctness_implies_discrete_gate_for_deterministic_candidates():
    b = geometry()
    policy = v9.QuartetConstraints(b)
    for scale in (0.0, 0.01, 0.05, 0.2, 0.5):
        v = torch.sin(torch.arange(8 * b["pg"].numel(), dtype=torch.float64)).reshape(8, *b["pg"].shape) * scale
        p = v8.physical_trajectory(b["pg"], b["mask"], v)["prediction"]
        if (policy(p) <= 0).all():
            a, inv = signed_status(p, b["target"], b["mask"])
            old, oldinv = signed_status(b["pg"], b["target"], b["mask"])
            assert (a & old).sum() == old.sum()
            assert inv.sum() <= oldinv.sum()
            assert not (inv & old & ~oldinv).any()


def test_exact_old_constraints_zero_parity_and_no_mutation():
    b = geometry()
    before = {k: v.clone() for k, v in b.items()}
    oracle = v9.Oracle(b)
    old = v8.Oracle(b, "B")
    z = np.zeros(math.prod(oracle.shape), dtype=np.float64)
    np.testing.assert_array_equal(oracle.cfun(z)[:2], old.cfun(z))
    assert all(torch.equal(p, b["pg"]) for p in oracle.point(z).tr["states"])
    assert oracle.fun(z)[0] == 1
    for k, v in b.items():
        assert torch.equal(v, before[k]) and v.grad is None
    with pytest.raises(ValueError, match="float64"):
        v9.Oracle({k: v.float() if v.is_floating_point() else v for k, v in b.items()})


def test_numerical_contract_only_sparse_storage_changes():
    root = Path(__file__).resolve().parents[1] / "configs"
    a = yaml.safe_load((root / "e010_phase4d_local_feasibility_v8.yaml").read_text())
    b = yaml.safe_load((root / "e010_phase4d_no_new_inversion_v9.yaml").read_text())
    assert a["recurrence"] == b["recurrence"]
    assert {k: v for k, v in a["solver"].items() if k not in ("sparse_jacobian", "maxiter")} == {
        k: v for k, v in b["solver"].items() if k not in ("sparse_jacobian", "maxiter")
    }
    assert a["constraints"] == b["constraints"]
