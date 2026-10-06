import numpy as np
import pytest

from protein_distance_diffusion.training import e010_fixed_state_audit_v12 as audit
from scripts import audit_e010_fixed_state_v12 as runner


def test_radial_tangential_units_and_outward_sign():
    d = np.array([[0.04, 0, 0], [0, 0.02, 0]])
    g = np.array([[-3.0, 4.0, 0], [0, 2.0, 5.0]])
    p = audit.split_gradient(d, g, [True, True])
    np.testing.assert_array_equal(p["outward"], [3, 0])
    np.testing.assert_array_equal(p["inward"], [0, 2])
    np.testing.assert_array_equal(p["tangent"], [[0, 4, 0], [0, 0, 5]])
    s = audit.decomposition_summary(p, [True, False])
    assert s["frozen_normalized"]["outward_descent_radial_l2"] == 0.12
    assert s["frozen_normalized"]["tangential_l2"] == 0.16


def test_svd_rank_dependencies_and_physical_units():
    a = np.array([[1.0, 0, 0], [0, 2, 0], [1, 0, 0]])
    info = audit.jacobian_geometry(a)
    assert info["frozen_normalized"]["rank"] == 2
    np.testing.assert_allclose(info["physical_singular_values_per_angstrom"], np.linalg.svd(a, compute_uv=False) / 0.04)
    assert audit.jacobian_geometry(np.empty((0, 3)))["frozen_normalized"]["rank"] == 0


def test_tangent_qp_respects_inequalities_and_small_block_radius():
    g = np.array([-1.0, 2, 0])
    a = np.array([[1.0, 0, 0]])
    direction, q = audit.tangent_diagnostic(g, a, (1, 1, 3), {"shadow_steps_angstrom": [1e-6, 1e-5, 1e-4]})
    np.testing.assert_allclose(direction, [[[0, -1, 0]]], atol=1e-15)
    assert q["maximum_linearized_violation"] <= 1e-15
    assert q["stationarity"] < 1e-15
    assert q["predicted_normalized_decrease_by_scale"]["0.0001"] == 0.0002


def test_history_sampling_is_not_an_invented_iteration_trajectory():
    rows = [
        dict(
            iteration=i,
            normalized_objective=2 - 0.001 * i,
            optimality=0.1,
            constraint_violation=0,
            trust_radius=1,
            barrier_parameter=1e-7,
        )
        for i in range(10, 101, 10)
    ]
    log = dict(iterations=100, history=rows, physical_screens=[])
    w = audit.history_windows(log, [25])["25"]
    assert w["normalized_objective"]["snapshots"] == 3
    assert w["normalized_objective"]["first_iteration"] == 80
    assert w["normalized_objective"]["slope_per_iteration"] == pytest.approx(-0.001)
    assert not w["physical_stationarity"]["available"]
    log.update(multipliers=[], ball_constraint_multipliers=[])
    assert not audit.telemetry(log)["constraint_penalty"]["available"]


def test_optimizer_guard_allows_diagnostic_nnls_only():
    from scipy.optimize import minimize

    with audit.forbid_nonlinear_optimization():
        import scipy.optimize

        with pytest.raises(AssertionError, match="forbids"):
            scipy.optimize.minimize(lambda x: x * x, [0.0])
        with pytest.raises(AssertionError, match="forbids"):
            audit.direct.solve(None, None)
        audit.tangent_diagnostic(np.ones(3), np.empty((0, 3)), (1, 1, 3), {"shadow_steps_angstrom": [1e-6]})
    assert scipy.optimize.minimize is minimize


def test_saved_control_hash_metric_certificate_and_shadow_reproduction():
    runner.setup()
    with audit.forbid_nonlinear_optimization():
        b, row, d, mu, path = runner.load_fixed(9)
        before = runner.file_hash(path)
        result = audit.local_analysis(
            b,
            d,
            mu,
            row["optimizer"],
            runner.historical.CONTRACT["physical_config"],
            runner.historical.CONTRACT["physical_limits"],
        )
    assert (
        runner.json.loads(runner.json.dumps(result["physical_certificate"])) == row["optimizer"]["physical_certificate"]
    )
    assert runner.file_hash(path) == before
    assert not result["material_feasible_descent"]
    assert result["tangent_QP"]["maximum_linearized_violation"] <= 1e-12


def test_fixed_state_hash_mismatch_stops(monkeypatch):
    runner.setup()
    real = runner.file_hash
    monkeypatch.setattr(runner, "file_hash", lambda p: "bad" if str(p).endswith("example_09.npz") else real(p))
    with pytest.raises(AssertionError, match="Fixed state hash mismatch"):
        runner.load_fixed(9)


def test_grouping_exact_historical_labels():
    runner.setup()
    rows = [runner.json.loads((runner.historical.OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    assert [sum(runner.group(r) == g for r in rows) for g in "ABC"] == [4, 55, 1]
