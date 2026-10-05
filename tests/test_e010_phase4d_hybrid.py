import pytest
import torch

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from protein_distance_diffusion.models.e010_hybrid_local import (
    FEATURE_DIM,
    FrozenGlobalLocal,
    LocalGeometryBranch,
    local_representation,
)
from protein_distance_diffusion.training.e010_phase4d import (
    aggregate_metrics,
    distance_diversity,
    example_metrics,
    hybrid_losses,
)


def fixture():
    torch.manual_seed(14)
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    m = torch.tensor([[True] * 9, [True] * 6 + [False] * 3])
    return x, m


def rotation():
    q, _ = torch.linalg.qr(torch.tensor([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [2.0, 1.0, -1.0]], dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_frames_and_features_rigid_behavior():
    x, m = fixture()
    a = local_representation(x, m)
    q = rotation()
    b = local_representation(x @ q.T + 23, m)
    torch.testing.assert_close(a["features"], b["features"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(b["frame"], q @ a["frame"], atol=1e-12, rtol=1e-12)
    f = a["frame"][a["eligible"]]
    torch.testing.assert_close(f.transpose(-1, -2) @ f, torch.eye(3, dtype=x.dtype).expand_as(f))
    torch.testing.assert_close(torch.det(f), torch.ones(f.shape[0], dtype=x.dtype))
    assert a["features"].shape == (2, 9, FEATURE_DIM)
    assert a["eligible"].sum() == 11
    assert not a["eligible"][:, 0].any() and not a["eligible"][0, -1]
    assert not a["eligible"][1, 5:].any()


def test_reflection_parity():
    x, m = fixture()
    q = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=x.dtype))
    a, b = local_representation(x, m), local_representation(x @ q.T, m)
    parity = torch.ones(FEATURE_DIM, dtype=x.dtype)
    parity[14:30:3] = -1  # neighbor local z
    parity[34] = parity[36] = -1  # sin(torsion), pseudoscalar
    torch.testing.assert_close(b["features"], a["features"] * parity)
    torch.testing.assert_close(b["frame"], q @ a["frame"] @ torch.diag(torch.tensor([1.0, 1.0, -1.0], dtype=x.dtype)))


def test_signed_geometry_known_convention():
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]], dtype=torch.float64)
    r = local_representation(x, torch.ones(1, 4, dtype=torch.bool))
    assert r["features"][0, 1, 34] == 1
    assert r["features"][0, 1, 35] == 0
    assert r["features"][0, 1, 36] == 1
    assert r["torsion_assessable"].sum() == 1


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7])
def test_short_degenerate_all_padding_and_near_collinear(n):
    x = torch.zeros(2, n, 3)
    if n:
        x[0, :, 0] = torch.arange(n)
        x[0, :, 1] = torch.arange(n).square() * 1e-9
    m = torch.ones(2, n, dtype=torch.bool)
    m[1] = False
    r = LocalGeometryBranch()(x, m)
    assert torch.isfinite(r["features"]).all()
    assert torch.isfinite(r["prediction"]).all()
    assert not r["eligible"].any()
    assert torch.count_nonzero(r["delta"]) == 0
    assert r["degenerate"].sum() == max(n - 2, 0)


def test_padding_invariance_and_availability():
    x, m = fixture()
    branch = LocalGeometryBranch().double()
    torch.nn.init.normal_(branch.head.weight, std=0.01)
    a = branch(x, m)
    xx = x.clone()
    xx[~m] = float("nan")
    b = branch(xx, m)
    torch.testing.assert_close(a["delta"], b["delta"], atol=0, rtol=0)
    one = branch(x[1:2, :6], m[1:2, :6])
    torch.testing.assert_close(one["delta"], a["delta"][1:2, :6], atol=1e-14, rtol=1e-14)
    assert a["features"][0, 1, 6:12].tolist() == [0, 0, 1, 1, 1, 1]
    assert (a["features"][~a["eligible"]] == 0).all()


def test_nonzero_output_equivariance():
    x, m = fixture()
    b = LocalGeometryBranch().double()
    torch.nn.init.normal_(b.head.weight, std=0.01)
    a = b(x, m)
    q = rotation()
    moved = b(x @ q.T + 7, m)
    torch.testing.assert_close(moved["delta"], a["delta"] @ q.T, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(moved["prediction"], a["prediction"] @ q.T + 7, atol=1e-12, rtol=1e-12)
    assert a["delta"].abs().sum() > 0
    assert torch.count_nonzero(a["delta"][~a["eligible"]]) == 0


def test_zero_init_and_gradient_isolation():
    x, m = fixture()
    x = x.float().requires_grad_()
    global_model = GlobalEquivariantResidual(width=32, layers=1, heads=4, vector_channels=4)
    model = FrozenGlobalLocal(global_model)
    result = model(x, m)
    assert torch.equal(result["prediction"], result["global_prediction"])
    assert torch.count_nonzero(result["delta"]) == 0
    hybrid_losses(result["prediction"], result["global_prediction"], x.detach(), x.detach() + 0.3, m)[
        "total"
    ].backward()
    assert all(not p.requires_grad and p.grad is None for p in global_model.parameters())
    grads = [p.grad for p in model.local.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0
    assert model.local.head.weight.grad.abs().sum() > 0
    assert x.grad is None
    model.train()
    assert not global_model.training


def test_parameter_count():
    assert sum(p.numel() for p in LocalGeometryBranch().parameters()) == 927875
    assert (
        sum(p.numel() for p in GlobalEquivariantResidual(width=416, layers=6, heads=8, vector_channels=64).parameters())
        == 12844352
    )


def test_guard_and_endpoint_losses():
    x, m = fixture()
    p = (x + 0.8).requires_grad_()
    y = x + 0.2
    got = hybrid_losses(p, x, x, y, m, beta=2, rho=0.1, delta_cart=0.01)
    expected = torch.relu(got["cartesian"] - 1.01 * got["global_cartesian"])
    assert torch.equal(got["guard"], expected) and expected > 0
    torch.testing.assert_close(got["displacement"], torch.tensor(1.92, dtype=x.dtype))
    torch.testing.assert_close(got["total"], got["local_mean"] + 2 * expected + 0.1 * got["displacement"])
    zero = hybrid_losses(x, x, x, y, m)
    assert zero["guard"] == 0 and zero["displacement"] == 0
    for k in (1, 2, 3):
        valid = m[:, k:] & m[:, :-k]
        err = ((p[:, k:] - p[:, :-k]).norm(dim=-1) - (y[:, k:] - y[:, :-k]).norm(dim=-1)).square()
        torch.testing.assert_close(got[f"local_{k}"], ((err * valid).sum(1) / valid.sum(1)).mean())
    with pytest.raises(ValueError):
        hybrid_losses(p, x, x, y, m, beta=-1)


def test_metrics_and_grouping():
    x, m = fixture()
    rows = example_metrics(x, x, x, x, m)
    assert all(r["aligned_rmsd"] < 1e-12 and r["chirality_inversions"] == 0 for r in rows)
    for i, r in enumerate(rows):
        r.update(sample_id=str(i), condition=50, stratum="20-64")
    agg = aggregate_metrics(rows)
    assert agg["overall"]["frame_eligible"] == 11 and agg["overall"]["all_finite"]
    assert distance_diversity(x[:1].expand(2, -1, -1), m[:1].expand(2, -1)) == 0


def test_holes_rejected():
    x, m = fixture()
    m[0, 3] = False
    with pytest.raises(ValueError, match="right-padded"):
        local_representation(x, m)


def test_reflected_metrics_count_inversions_and_assessability():
    x, m = fixture()
    reflected = x * torch.tensor([-1.0, 1.0, 1.0], dtype=x.dtype)
    rows = example_metrics(reflected, x, x, x, m)
    assert all(r["chirality_assessable"] > 0 for r in rows)
    assert all(r["chirality_inversions"] == r["chirality_assessable"] for r in rows)
    assert all(r["aligned_rmsd"] > 0 for r in rows)


def test_guard_detaches_baseline_and_active_guard_gradient():
    x, m = fixture()
    pg = x.clone().requires_grad_()
    p = (x + 1).requires_grad_()
    loss = hybrid_losses(p, pg, x, x, m)
    loss["guard"].backward()
    assert pg.grad is None and p.grad[m].abs().sum() > 0
    assert torch.count_nonzero(p.grad[~m]) == 0


def test_actual_historical_checkpoint_zero_parity_and_gradients():
    from pathlib import Path

    from protein_distance_diffusion.models.e010_hybrid_local import load_frozen_hybrid

    path = Path(
        "/home/simostocco/proteinGen/reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_training_v1.final/latest.pt"
    )
    if not path.exists():
        pytest.skip("historical untracked checkpoint unavailable")
    model = load_frozen_hybrid(path)
    x, m = fixture()
    x = x.float()
    out = model(x, m)
    with torch.no_grad():
        historical = model.global_model(x, m)["prediction"]
    assert torch.equal(out["prediction"], historical)
    assert torch.count_nonzero(out["delta"]) == 0
    hybrid_losses(out["prediction"], out["global_prediction"], x, x, m)["total"].backward()
    assert all(p.grad is None for p in model.global_model.parameters())
    assert torch.isfinite(model.local.head.weight.grad).all()
    assert model.local.head.weight.grad.abs().sum() > 0


def test_panel_selection_deterministic_and_stratified():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "phase4d_prep_test", Path(__file__).resolve().parents[1] / "scripts/prepare_e010_phase4d.py"
    )
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    rows = [
        dict(sample_id=f"{s}_{i}", length=50, stratum=s)
        for s in ("20-64", "65-128", "129-256", "257-384", "385-500")
        for i in range(8)
    ]
    a = prep.select_panel(rows, "fixed")
    assert a == prep.select_panel(list(reversed(rows)), "fixed")
    assert len(a) == len({r["sample_id"] for r in a}) == 20
    assert all(sum(r["stratum"] == s for r in a) == 4 for s in {r["stratum"] for r in rows})


def test_hybrid_empty_rows_bypass_historical_forward():
    x = torch.zeros(2, 3, 3)
    m = torch.zeros(2, 3, dtype=torch.bool)
    model = FrozenGlobalLocal(GlobalEquivariantResidual(width=32, layers=1, heads=4, vector_channels=4))
    out = model(x, m)
    assert torch.equal(out["prediction"], x) and torch.count_nonzero(out["delta"]) == 0


def test_nonfinite_evaluation_reports_status_before_alignment():
    x, m = fixture()
    x[0, 1, 0] = float("nan")
    rows = example_metrics(x, x, x, x, m)
    assert rows[0]["finite"] is False
    for i, r in enumerate(rows):
        r.update(sample_id=str(i), condition=50, stratum="20-64")
    assert aggregate_metrics(rows)["overall"]["all_finite"] is False
