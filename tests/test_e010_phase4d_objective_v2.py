import json
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from protein_distance_diffusion.models.e010_hybrid_local import (
    PSEUDOSCALAR_INDEX,
    FrozenGlobalLocal,
    LocalGeometryBranch,
    local_representation,
)
from protein_distance_diffusion.training.e010_phase4d_diagnostic import (
    displaced_copy,
    dot,
    file_hash,
    signed_status,
    state_hash,
)
from protein_distance_diffusion.training.e010_phase4d_objective_v2 import (
    BETA,
    chiral_sum,
    combined_gradient,
    common_descent_interval,
    direction_rms_slope,
    freeze_chirality,
    select_gamma,
    signed_triple,
    smooth_bounded_local,
)


def geometry():
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0]]], dtype=torch.float64
    )
    m = torch.ones(x.shape[:2], dtype=torch.bool)
    return x, m


def test_beta_combined_arithmetic_and_real_local_cartesian_descent():
    assert BETA == 16.8
    a, b, c = [torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 4.0])], [torch.tensor([5.0, 6.0])]
    torch.testing.assert_close(combined_gradient(a, b, c, 2)[0], a[0] + 16.8 * b[0] + 2 * c[0])
    path = (
        Path(__file__).resolve().parents[1]
        / "reports/experiments/E010_global_equivariant_expressivity"
        / "phase4d_hybrid_local_global_v1/calibration_diagnostic_v1/results.json"
    )
    audit = json.loads(path.read_text())["initial_variants"][0]
    ll = audit["local_gradient_norm"] ** 2
    cc = audit["cartesian_gradient_norm"] ** 2
    lc = -audit["predicted_cartesian_derivative"]
    assert ll + BETA * lc > 0 and lc + BETA * cc > 0


def test_signed_loss_matches_evaluator_and_inversions():
    x, m = geometry()
    q = signed_triple(x, m)
    rep = local_representation(x, m)
    torch.testing.assert_close(q, rep["features"][:, 1:-2, PSEUDOSCALAR_INDEX])
    reflected = x * torch.tensor([-1.0, 1.0, 1.0], dtype=x.dtype)
    f = freeze_chirality(x, x, m)
    assert chiral_sum(x, m, f) == 0
    torch.testing.assert_close(chiral_sum(reflected, m, f), 4 * q.square().sum())
    eligible, inverted = signed_status(reflected, x, m)
    assert inverted.sum() == eligible.sum() == 2


def test_rotation_translation_and_reflection_conventions():
    x, m = geometry()
    q, _ = torch.linalg.qr(torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, -1.0]], dtype=x.dtype))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    torch.testing.assert_close(signed_triple(x @ q.T + 17, m), signed_triple(x, m), atol=1e-12, rtol=1e-12)
    reflected = x.clone()
    reflected[:, :, 0] *= -1
    torch.testing.assert_close(signed_triple(reflected, m), -signed_triple(x, m))


def test_eligibility_frozen_target_detached_and_no_predicted_mask_escape():
    x, m = geometry()
    target = x.clone().requires_grad_()
    f = freeze_chirality(x, target, m)
    assert f["eligible"].sum() == 2 and not f["q_target"].requires_grad
    p = torch.zeros_like(x).requires_grad_()
    loss = chiral_sum(p, m, f)
    assert loss > 0 and f["eligible"].sum() == 2
    loss.backward()
    assert torch.isfinite(p.grad).all() and target.grad is None
    planar = x.clone()
    planar[:, :, 2] = 0
    assert freeze_chirality(x, planar, m)["eligible"].sum() == 0
    assert chiral_sum(x, m, freeze_chirality(x, planar, m)) == 0


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_short_chain_safety(n):
    x = torch.zeros(1, n, 3, requires_grad=True)
    m = torch.ones(1, n, dtype=torch.bool)
    f = freeze_chirality(x.detach(), x.detach(), m)
    loss = chiral_sum(x, m, f)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_padding_does_not_add_quartets():
    x, m = geometry()
    padded = torch.cat([x, torch.full((1, 4, 3), float("nan"), dtype=x.dtype)], 1)
    mask = torch.cat([m, torch.zeros(1, 4, dtype=torch.bool)], 1)
    f = freeze_chirality(padded, padded, mask)
    assert f["eligible"].sum() == 2
    assert chiral_sum(padded, mask, f) == 0


def test_feasible_interval_selection_and_impossible_classification():
    g = {
        "local": [torch.tensor([1.0, 0.0])],
        "cartesian": [torch.tensor([0.0, 1.0])],
        "chiral": [torch.tensor([-0.1, 1.0])],
    }
    interval = common_descent_interval(g)
    assert interval["feasible"] and select_gamma(interval, g) == 0
    bad = {
        "local": [torch.tensor([1.0, 0.0])],
        "cartesian": [torch.tensor([-1.0, 0.0])],
        "chiral": [torch.tensor([0.0, 1.0])],
    }
    interval = common_descent_interval(bad)
    assert not interval["feasible"] and select_gamma(interval, bad) is None
    g = {
        "local": [torch.tensor([1.0, 0.0])],
        "cartesian": [torch.tensor([0.0, 1.0])],
        "chiral": [torch.tensor([0.0, -1.0])],
    }
    assert not common_descent_interval(g)["feasible"]


def test_positive_gamma_constraint_rule_without_metric_search():
    # Beta forces y=-0.168, z=16.8; chirality needs gamma>0.168, cart descent allows gamma<1680.168.
    g = {
        "local": [torch.tensor([1.0, 0.0, 0.0])],
        "cartesian": [torch.tensor([0.0, -0.01, 1.0])],
        "chiral": [torch.tensor([0.0, 1.0, 0.0])],
    }
    interval = common_descent_interval(g)
    gamma = select_gamma(interval, g)
    assert interval["feasible"] and gamma == pytest.approx(0.2)
    total = combined_gradient(g["local"], g["cartesian"], g["chiral"], gamma)
    assert all(dot(v, total) > 0 for v in g.values())


def test_correction_rms_mapping_independent_copy_and_gradient_isolation():
    torch.manual_seed(41047)
    model = FrozenGlobalLocal(
        GlobalEquivariantResidual(width=32, layers=1, heads=4, vector_channels=4),
        LocalGeometryBranch(width=16, blocks=1),
    ).double()
    x, m = geometry()
    out = model(x, m)
    pg = out["global_prediction"]
    f = freeze_chirality(pg, x, m)
    loss = chiral_sum(out["prediction"], m, f)
    gs = torch.autograd.grad(loss, tuple(model.local.parameters()))
    direction = [-g for g in gs]
    panel = [{"pg": pg, "mask": m}]
    pin = state_hash(model.local)
    slope = direction_rms_slope(model.local, direction, panel)
    assert slope > 0
    alpha = 0.1 / slope
    candidate = displaced_copy(model.local, direction, alpha)
    result = candidate(pg, m)
    actual = ((result["delta"].square().sum(-1) * m).sum() / m.sum()).sqrt()
    torch.testing.assert_close(actual, torch.tensor(0.1, dtype=x.dtype), atol=1e-12, rtol=1e-12)
    assert state_hash(model.local) == pin
    chiral_sum(result["prediction"], m, f).backward()
    assert all(p.grad is None and not p.requires_grad for p in model.global_model.parameters())
    assert candidate.head.weight.grad.abs().sum() > 0


def test_smooth_bound_zero_jacobian_equivariance_and_norm():
    u = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-100.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    s = 0.3
    bounded = smooth_bounded_local(u, s)
    assert torch.equal(bounded[0], u[0]) and (bounded.norm(dim=-1) <= s).all()
    jac = torch.autograd.functional.jacobian(lambda v: smooth_bounded_local(v, s), u[0])
    torch.testing.assert_close(jac, torch.eye(3, dtype=u.dtype))
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=u.dtype)
    torch.testing.assert_close(smooth_bounded_local(u @ rotation, s), bounded @ rotation)
    with pytest.raises(ValueError):
        smooth_bounded_local(u, 0)


def test_historical_and_v1_integrity_against_record():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "reports/experiments/E010_global_equivariant_expressivity"
        / "phase4d_hybrid_local_global_v1/calibration_diagnostic_v1/validation.json"
    )
    data = json.loads(path.read_text())
    for relative, pin in data["diagnostic_files_sha256"].items():
        assert file_hash(root / relative) == pin


def test_smooth_bound_large_float32_finite():
    u = torch.tensor([[1e30, -1e30, 1e30]], dtype=torch.float32)
    bounded = smooth_bounded_local(u, 0.05)
    assert bounded.dtype == u.dtype and torch.isfinite(bounded).all()
    assert bounded.norm() <= 0.05 + 1e-8


def test_target_eligibility_loss_matches_pooled_quartet_reduction():
    x, m = geometry()
    f = freeze_chirality(x, x, m)
    p = x.clone()
    p[0, 2, 0] += 0.1
    expected = (signed_triple(p, m) - signed_triple(x, m)).square()[f["eligible"]].mean()
    torch.testing.assert_close(chiral_sum(p, m, f) / f["eligible"].sum(), expected)


def test_saturation_commutes_with_frame_cartesian_rotation():
    x, m = geometry()
    f = local_representation(x, m)["frame"]
    u = torch.randn(1, 5, 3, dtype=x.dtype)
    r = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype)
    delta = torch.einsum("bnij,bnj->bni", f, smooth_bounded_local(u, 0.05))
    moved_f = local_representation(x @ r.T + 3, m)["frame"]
    moved = torch.einsum("bnij,bnj->bni", moved_f, smooth_bounded_local(u, 0.05))
    torch.testing.assert_close(moved, delta @ r.T)


def test_actual_v2_gamma_and_scale_contract_when_record_available():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "reports/experiments/E010_global_equivariant_expressivity"
        / "phase4d_hybrid_local_global_v1/objective_v2_diagnostic/results.json"
    )
    if not path.exists():
        pytest.skip("untracked v2 diagnostic record not available")
    data = json.loads(path.read_text())
    audit = data["audit"]
    interval = audit["feasible_gamma_interval"]
    assert interval["lower"] < audit["selected_gamma"] == 2 < interval["upper"]
    assert all(v < 0 for v in audit["predicted_derivatives"].values())
    assert data["global_gradient_isolation"] and data["protected_inputs_unchanged"]
    assert not data["training_launched"] and not data["cuda_used"]
    for point in data["finite_points"]:
        actual = point["metrics"]["overall"]["displacement_rms_equal_example"]
        assert actual == pytest.approx(point["target_correction_rms"], abs=1e-8)
        assert point["metrics"]["overall"]["chiral_loss_eligible"] == 13029
