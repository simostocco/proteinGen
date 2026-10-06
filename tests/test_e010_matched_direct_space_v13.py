import copy
import inspect
import json

import pytest

from scripts import run_e010_matched_direct_space_v13 as runner


def test_only_iteration_cap_changes_and_frozen_physical_contract():
    cfg, physical, limits, own = runner.setup()
    previous = json.loads((runner.HISTORICAL / "execution_contract.json").read_text())
    assert cfg["solver"]["maxiter"] == 2000
    restored = copy.deepcopy(cfg)
    restored["solver"]["maxiter"] = 1000
    assert restored == previous["solver_config"]
    assert physical == previous["physical_config"] and limits == previous["physical_limits"]
    assert own["initialization"] == "exact_zero" and own["solver_invocations_per_example"] == 1


def test_reuses_original_solver_without_new_parameterization():
    from protein_distance_diffusion.training import e010_direct_correction_v11 as direct

    assert runner.historical.direct.solve is direct.solve
    source = inspect.getsource(runner.example)
    assert "historical.example(i, mode)" in source
    assert "direct.solve" not in source
    assert "Once-only V13 panel cannot be restarted" in inspect.getsource(runner.execute)


def test_classification_uses_inherited_descriptive_materiality():
    from scripts.report_e010_matched_direct_space_v13 import classify

    assert classify(60, 0, 6, []) == "MATCH-A"
    assert classify(10, 55, 8, []) == "MATCH-B"
    assert classify(4, 49, 8, []) == "MATCH-B"
    assert classify(4, 55, 8, []) == "MATCH-C"
    assert classify(60, 0, 4, []) == "MATCH-B"
    assert classify(60, 0, 8, [1]) == "MATCH-D"


def test_zero_state_and_historical_derivatives_without_solver(monkeypatch):
    from scripts.recover_e010_conditioning_v9b import synthetic

    runner.setup()

    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer in implementation validation")

    monkeypatch.setattr(runner.historical.direct, "solve", forbidden)
    check = runner.historical.validate(synthetic(12))
    assert check["zero_exact"] and check["baseline_exact_feasible"] and check["interior_parity"]
    assert all(r["passed"] for r in check["directions"])


def test_attempt_marker_blocks_second_solver_invocation(tmp_path, monkeypatch):
    historical = runner.historical
    monkeypatch.setattr(historical, "OUT", tmp_path)
    monkeypatch.setattr(historical, "data", lambda i: None)
    monkeypatch.setattr(historical, "CONTRACT", dict(solver_config={}, physical_config={}, physical_limits={}))
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    (attempts / "example_00.json").write_text("{}")

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing attempt must block solver")

    monkeypatch.setattr(historical.direct, "solve", forbidden)
    with pytest.raises(FileExistsError):
        historical.example(0, "run")
