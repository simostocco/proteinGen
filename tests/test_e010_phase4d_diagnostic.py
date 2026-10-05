import copy

import pytest
import torch

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from protein_distance_diffusion.models.e010_hybrid_local import FrozenGlobalLocal, LocalGeometryBranch
from protein_distance_diffusion.training.e010_phase4d import hybrid_losses
from protein_distance_diffusion.training.e010_phase4d_diagnostic import (
    aggregate_rows,
    assert_file_pins,
    combine,
    displaced_copy,
    evaluate_point,
    file_hash,
    guard_status,
    norm,
    run_diagnostic,
    state_hash,
)


def fixtures():
    torch.manual_seed(41047)
    branch = LocalGeometryBranch(width=16, blocks=1)
    pg = torch.randn(1, 8, 3)
    y = torch.randn(1, 8, 3)
    item = dict(
        pg=pg,
        source=pg.clone(),
        target=y,
        mask=torch.ones(1, 8, dtype=torch.bool),
        sample_id="fixture",
        condition=50,
        stratum="20-64",
    )
    protocol = {
        "variants": {"A": {"beta": 0.0, "rho": 0.0}, "B": {"beta": 1.0, "rho": 0.0}, "C": {"beta": 1.0, "rho": 0.01}},
        "alpha": [0.0, 0.0001, 0.001, 0.01],
        "delta_cart": [0.0, 0.005, 0.01],
    }
    return branch, [item], protocol


def test_independent_displacements_are_not_cumulative_and_do_not_alias():
    base, panel, _ = fixtures()
    _, g = evaluate_point(base, panel)
    d = [-t for t in g["local_mean"]]
    pin = state_hash(base)
    a = displaced_copy(base, d, 0.001)
    b = displaced_copy(base, d, 0.01)
    expected = copy.deepcopy(base)
    with torch.no_grad():
        for p, v in zip(expected.parameters(), d, strict=True):
            p.add_(v, alpha=0.01)
        a.head.bias.add_(123)
    assert state_hash(base) == pin
    assert state_hash(b) == state_hash(expected)
    assert a.head.bias.data_ptr() != base.head.bias.data_ptr() != b.head.bias.data_ptr()


def test_complete_diagnostic_is_deterministic():
    base, panel, protocol = fixtures()
    a = run_diagnostic(base, panel, protocol)
    b = run_diagnostic(base, panel, protocol)
    assert a == b
    assert len(a["variant_records"]) == 36
    assert a["all_initial_directions_equal"] and a["baseline_local_unchanged"]
    assert a["optimizer_steps"] == 0
    assert all(p["metrics"]["overall"]["all_finite"] for p in a["points"])
    assert all(v["initial_guard"]["value"] == 0 for v in a["initial_variants"])


def test_no_e010_gradient_leakage_after_finite_displacement():
    base, panel, _ = fixtures()
    model = FrozenGlobalLocal(GlobalEquivariantResidual(width=32, layers=1, heads=4, vector_channels=4), base)
    x = panel[0]["source"].clone().requires_grad_()
    m = panel[0]["mask"]
    global_pin = state_hash(model.global_model)
    pg = model(x, m)["global_prediction"]
    panel[0]["pg"] = pg
    _, g = evaluate_point(base, panel)
    local = displaced_copy(base, [-t for t in g["local_mean"]], 0.001)
    out = local(pg, m)
    hybrid_losses(out["prediction"], pg, x.detach(), panel[0]["target"], m)["total"].backward()
    assert all(p.grad is None and not p.requires_grad for p in model.global_model.parameters())
    assert state_hash(model.global_model) == global_pin and x.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in local.parameters())


@pytest.mark.parametrize("tol", [0.0, 0.005, 0.01])
def test_guard_threshold_strict_activation_and_zero_boundary_derivative(tol):
    baseline = 20.0
    threshold = (1 + tol) * baseline
    assert not guard_status(threshold, baseline, tol)["active"]
    assert not guard_status(threshold - 1e-5, baseline, tol)["active"]
    assert guard_status(threshold + 1e-5, baseline, tol)["active"]
    excess = torch.tensor(0.0, requires_grad=True)
    torch.relu(excess).backward()
    assert excess.grad == 0


def test_guard_uses_full_panel_reduction_not_mean_rectified_microbatches():
    baseline = 10.0
    carts = [12.0, 8.0]
    full = guard_status(sum(carts) / 2, baseline, 0.0)["value"]
    micro = sum(guard_status(c, baseline, 0.0)["value"] for c in carts) / 2
    assert full == 0 and micro == 1


def test_displacement_penalty_quadratic_growth_and_gradient_nonzero():
    base, panel, _ = fixtures()
    _, g = evaluate_point(base, panel)
    d = [-t for t in g["local_mean"]]
    initial, gi = evaluate_point(base, panel)
    a, ga = evaluate_point(displaced_copy(base, d, 0.001), panel)
    b, gb = evaluate_point(displaced_copy(base, d, 0.002), panel)
    assert initial["objective"]["displacement"] == 0 and norm(gi["displacement"]) == 0
    assert norm(ga["displacement"]) > 0 and norm(gb["displacement"]) > norm(ga["displacement"])
    assert b["objective"]["displacement"] == pytest.approx(4 * a["objective"]["displacement"], rel=2e-5)
    assert 0.01 * norm(ga["displacement"]) > 0


def test_input_pins_checked_without_checkpoint_write(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"pinned fixture")
    pins = {path: file_hash(path)}
    assert_file_pins(pins)
    base, panel, protocol = fixtures()
    run_diagnostic(base, panel, protocol)
    assert path.read_bytes() == b"pinned fixture"
    assert_file_pins(pins)
    path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="mutated"):
        assert_file_pins(pins)


def test_metric_units_and_distinct_rmse_reductions():
    base, panel, _ = fixtures()
    point, _ = evaluate_point(base, panel)
    rows = copy.deepcopy(point["per_example"])
    rows.append(copy.deepcopy(rows[0]))
    rows[1]["sample_id"] = "second"
    for r, v in zip(rows, (1.0, 9.0), strict=True):
        r["local_mse"] = {str(k): v for k in (1, 2, 3)}
        r["local_rmse"] = {str(k): v**0.5 for k in (1, 2, 3)}
        r["mean_local_rmse"] = v**0.5
        r["displacement_mean_square"] = v
        r["displacement_rms"] = v**0.5
    s = aggregate_rows(rows)["overall"]
    assert s["local_mse"]["1"] == 5
    assert s["local_rmse"]["1"] == 2
    assert s["local_rmse_from_mean_mse"]["1"] == pytest.approx(5**0.5)
    assert s["displacement_rms_equal_example"] == pytest.approx(5**0.5)


def test_initial_combinations_identical_and_nonzero_finite_penalty_changes_gradient():
    base, panel, _ = fixtures()
    _, g = evaluate_point(base, panel)
    initial = combine(g["local_mean"], g["cartesian"], g["displacement"], beta=1, rho=0.01, active=False)
    assert all(torch.equal(a, b) for a, b in zip(initial, g["local_mean"], strict=True))
    with pytest.raises(ValueError):
        displaced_copy(base, initial, float("nan"))


def test_streaming_components_match_full_panel_objective_and_active_guard_gradient():
    base, panel, _ = fixtures()
    second = copy.deepcopy(panel[0])
    second["sample_id"] = "second"
    second["target"] = second["target"] + 0.5
    panel.append(second)
    _, g0 = evaluate_point(base, panel)
    branch = displaced_copy(base, [-t for t in g0["local_mean"]], 0.001)
    point, g = evaluate_point(branch, panel)
    tensors = {key: torch.cat([r[key] for r in panel]) for key in ("pg", "source", "target", "mask")}
    out = branch(tensors["pg"], tensors["mask"])
    losses = hybrid_losses(
        out["prediction"], tensors["pg"], tensors["source"], tensors["target"], tensors["mask"], delta_cart=0.0
    )
    expected = torch.autograd.grad(losses["total"], tuple(branch.parameters()))
    status = guard_status(point["objective"]["cartesian"], point["objective"]["global_cartesian"], 0.0)
    actual = combine(g["local_mean"], g["cartesian"], g["displacement"], beta=1.0, rho=0.01, active=status["active"])
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_diagnostic_never_calls_optimizer_step(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optimizer step forbidden")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    monkeypatch.setattr(torch.optim.SGD, "step", forbidden)
    base, panel, protocol = fixtures()
    assert run_diagnostic(base, panel, protocol)["optimizer_steps"] == 0
