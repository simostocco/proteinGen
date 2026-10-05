import pytest

from scripts import run_e009_bayesian_refiner as e009


def test_v4_authorizes_only_length64_stage(monkeypatch):
    cfg = {
        "version": "e009_bayesian_geometry_refiner_v4",
        "authorization": {
            "run_fitting": False,
            "run_cuda": False,
            "run_overfits": False,
            "run_pilot": False,
        },
        "length_64_overfit_authorized": True,
    }
    calls = []
    monkeypatch.setattr(
        e009, "_overfit64", lambda config, resume=False: calls.append("length64") or {"status": "prepared"}
    )

    assert e009.run(cfg, "overfit-length-64")["status"] == "prepared"
    assert calls == ["length64"]
    with pytest.raises(PermissionError):
        e009.run(cfg, "fit-prior")
    with pytest.raises(PermissionError):
        e009.run(cfg, "cuda-smoke")
    with pytest.raises(ValueError):
        e009.run(cfg, "pilot")


def test_v3_does_not_authorize_length64_by_default():
    cfg = {
        "version": "e009_bayesian_geometry_refiner_v3",
        "authorization": {"run_fitting": False, "run_cuda": False, "run_overfits": False, "run_pilot": False},
    }
    with pytest.raises(PermissionError):
        e009.run(cfg, "overfit-length-64")
