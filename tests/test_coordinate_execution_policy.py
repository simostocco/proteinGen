"""Concurrency control cannot mask scientific source changes."""

import subprocess
from pathlib import Path

import pytest
from coordinate_execution_policy import FROZEN_RUNNER_SHA, canonical_runner, sequence_status


def test_exact_frozen_runner_recovery():
    source = Path("scripts/run_e014_scaleup.py").read_text()
    expected = subprocess.check_output(
        ["git", "show", "f08cfcc704feaa8c4514090fa9b539ff2018c5e8:scripts/run_e014_scaleup.py"], text=True
    )
    assert canonical_runner(source) == expected
    assert FROZEN_RUNNER_SHA == "3063e842f6cd93f0fdcc8a86d1bcd70b0598b51386e01c8dad0da06dd4608601"
    with pytest.raises(ValueError, match="scientific runner"):
        canonical_runner(source.replace('lr=cfg["optimizer"]["learning_rate"]', "lr=0.01"))
    with pytest.raises(ValueError):
        canonical_runner(source + "\n# unaudited change\n")


def test_telemetry_preserves_degradation(monkeypatch):
    status = dict(persistent_material_degradation=True, phase_changed=True, phase="active", timestamp_unix=0)
    compare = lambda *_: status  # noqa: E731
    assert sequence_status(compare, {"baseline": True}, {}, True) is status
    assert sequence_status(compare, {"baseline": True}, {}, False) is status
