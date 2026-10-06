import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protein_distance_diffusion.training import e010_slsqp_v14 as sqp
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_strict_scientific_v10 import setup


def test_runner_setup_matches_committed_physical_limits():
    from scripts.run_e010_slsqp_v14 import setup as runner_setup

    own, physical, limits = runner_setup()
    _, expected_physical, expected_limits = setup()
    assert own["solver"] == sqp.SETTINGS
    assert physical == expected_physical and limits == expected_limits


def test_frozen_settings_and_analytic_vector_constraints():
    assert sqp.SETTINGS == dict(method="SLSQP", maxiter=2000, ftol=1e-12, disp=False)
    source = inspect.getsource(sqp.solve)
    assert "jac=True" in source and "hess=" not in source
    o = sqp.direct.Oracle(synthetic(12))
    x = np.zeros(np.prod(o.shape))
    dictionaries = sqp.Constraints(o).dictionaries()
    assert len(dictionaries) == 2 and all(d["type"] == "ineq" and callable(d["jac"]) for d in dictionaries)
    np.testing.assert_array_equal(dictionaries[0]["fun"](x), -o.cfun(x))
    np.testing.assert_array_equal(dictionaries[1]["fun"](x), np.ones(len(x) // 3))


def test_all_vector_constraint_jacobians_and_objective_float64():
    o = sqp.direct.Oracle(synthetic(12))
    x = np.sin(np.arange(np.prod(o.shape)) + 1) * 0.05
    d = np.cos(np.arange(len(x)) + 1)
    d /= np.linalg.norm(d)
    for c in sqp.Constraints(o).dictionaries():
        j = c["jac"](x)
        assert j.dtype == np.float64
        np.testing.assert_allclose(
            (c["fun"](x + 1e-5 * d) - c["fun"](x - 1e-5 * d)) / 2e-5, j @ d, atol=1e-8, rtol=1e-5
        )
    np.testing.assert_allclose(
        (o.fun(x + 1e-5 * d)[0] - o.fun(x - 1e-5 * d)[0]) / 2e-5, o.fun(x)[1] @ d, atol=1e-8, rtol=1e-5
    )


def test_zero_no_radial_model_or_input_mutation(monkeypatch):
    b = synthetic(12)
    before = {k: v.clone() for k, v in b.items()}

    def forbidden(*args, **kwargs):
        raise AssertionError("radial map/model must not execute")

    monkeypatch.setattr(sqp.direct.v9.v8, "physical_trajectory", forbidden)
    o = sqp.direct.Oracle(b)
    x = np.zeros(np.prod(o.shape))
    assert all(torch.equal(p, b["pg"]) for p in o.point(x).tr["states"])
    assert not isinstance(o, torch.nn.Module)
    assert all(torch.equal(v, before[k]) and v.grad is None for k, v in b.items())


def test_exact_installed_workspace_formula_and_memory_gate():
    from scipy.optimize import _slsqp_py

    source = inspect.getsource(_slsqp_py._minimize_slsqp)
    assert "n*(n+1)//2 + 3*m*n - (m + 5*n + 7)*meq + 9*m + 8*n*n + 35*n + meq*meq + 28" in source
    out = sqp.required_arrays(12000, 5495)
    assert out["buffer_elements"] == 1422295483
    assert out["required_array_bytes"] == 11905883864
    policy = dict(minimum_reserve_bytes=2**31, reserve_fraction_available_ram=0.25)
    assert not sqp.ram_gate(out, 12 * 2**30, policy)["resource_feasible"]
    assert sqp.ram_gate(out, 20 * 2**30, policy)["resource_feasible"]


def test_empty_active_set_and_multiplier_reconstruction():
    d, mu = sqp.audit.cone_projection(np.array([1.0, 2.0, 3.0]), np.zeros((0, 3)))
    np.testing.assert_array_equal(d, [-1.0, -2.0, -3.0])
    assert len(mu) == 0
    d, mu = sqp.audit.cone_projection(np.array([-1.0, 2.0, 0.0]), np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(d, [0.0, -2.0, 0.0])
    np.testing.assert_array_equal(mu, [1.0])
    assert sqp.optional(SimpleNamespace(), "multipliers", list) is None


def test_completed_failed_solver_saved_before_telemetry_exception(monkeypatch, tmp_path):
    from scripts.run_e010_slsqp_v14 import save_raw

    _, cfg, limits = setup()
    b = synthetic(12)

    def mocked(fun, x, **kwargs):
        assert kwargs["method"] == "SLSQP" and kwargs["jac"] is True
        return SimpleNamespace(x=x, fun=1.0, nit=2000, status=9, success=False, message="iteration cap", nfev=1)

    monkeypatch.setattr(sqp, "minimize", mocked)

    def telemetry_failure(*args):
        raise RuntimeError("injected optional telemetry failure")

    monkeypatch.setattr(sqp, "reconstruct", telemetry_failure)
    path = tmp_path / "saved.npz"
    with pytest.raises(RuntimeError, match="injected"):
        sqp.solve(b, cfg, limits, lambda r, t: save_raw(path, r, t))
    assert path.exists() and path.with_suffix(".json").exists()
    with np.load(path) as saved:
        assert (saved["z"] == 0).all()
    with pytest.raises(FileExistsError):
        save_raw(path, mocked(None, np.zeros(2), method="SLSQP", jac=True), 0)


def test_fixed_physical_certificate_and_shadow_reconstruction():
    _, cfg, limits = setup()
    b = synthetic(12)
    delta = torch.sin(torch.arange(8 * b["pg"].numel(), dtype=torch.float64)).reshape(8, *b["pg"].shape) * 0.001
    mu, record = sqp.reconstruct(b, delta, cfg)
    cert = sqp.direct.certificate(b, delta, mu, cfg, limits)
    assert cert == sqp.direct.v10.certificate(b, delta, mu, cfg, limits)
    assert cert["projected_feasible_gradient_norm"] == record["projected_feasible_gradient_norm"]
    assert [s["step_angstrom"] for s in cert["shadow_steps"]] == [1e-6, 1e-5, 1e-4]


def test_classification_frozen_both_major_thresholds():
    from scripts.report_e010_slsqp_v14 import classify

    assert classify(60, 0, 6, True) == "SQP-A"
    assert classify(60, 0, 4, True) == "SQP-D"
    assert classify(30, 25, 8, True) == "SQP-B"
    assert classify(29, 25, 8, True) == "SQP-C"
    assert classify(30, 26, 8, True) == "SQP-C"
    assert classify(0, 0, 0, False, resource_ok=False) == "SQP-E"
    assert classify(0, 0, 0, False, integrity_ok=False) == "SQP-F"


def test_real_slsqp_determinism_without_numerical_jacobians(monkeypatch):
    import copy

    from scipy.optimize import _slsqp_py

    def forbidden(*args, **kwargs):
        raise AssertionError("Numerical differentiation inside SLSQP is forbidden")

    monkeypatch.setattr(_slsqp_py, "approx_derivative", forbidden)
    _, cfg, limits = setup()
    saved = []
    answers = [sqp.solve(synthetic(6), cfg, limits, lambda r, t: saved.append(r.x.copy())) for _ in range(2)]
    np.testing.assert_array_equal(saved[0], saved[1])
    first, second = [copy.deepcopy(answer[1]) for answer in answers]
    for log in [first, second]:
        log.pop("runtime_seconds")
        log.pop("process_peak_rss_bytes")
    assert first == second
