"""Read-only authorization checks for the reviewed Phase-3I.2 pilot v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from protein_distance_diffusion.training import e007_local_backbone_repair as repair

CONFIG = Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v4.yaml")
DECISION = Path(
    "reports/experiments/E007_matrix_sequence_cogeneration/"
    "local_backbone_repair_phase3i2_lifecycle_smoke_reviewed_pilot_v4_decision_v1.json"
)


def test_v4_is_reviewed_and_only_authorizes_bounded_pilot() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    decision = json.loads(DECISION.read_text())
    assert decision["immutable"] is True
    assert decision["review_status"] == "reviewed"
    assert decision["authorizes_pilot_execution"] is True
    assert decision["pilot_completion_non_authorizing_pending_scientific_review"] is True
    assert all(value is False for value in decision["downstream_authorizations"].values())
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == decision["pilot_configuration_sha256"]
    assert config["authorization"]["pilot_authorized"] is True
    assert all(value is False for key, value in config["authorization"].items() if key != "pilot_authorized")
    for item in decision["lifecycle_smoke_evidence"].values():
        assert repair._sha256_file(Path(item["path"])) == item["sha256"]
    output = Path(config["output_dir"])
    assert output.is_dir()
    report = json.loads((output / "report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert report["status"] == protocol["status"] == "completed_non_authorizing_bounded_pilot"
    assert report["authorizes_training"] is False
    assert protocol["authorizes_phase3j"] is False
    assert not output.with_name(f".{output.name}.inprogress").exists()


def test_changed_smoke_evidence_refuses_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    original = repair._sha256_file
    monkeypatch.setattr(
        repair,
        "_sha256_file",
        lambda path: "0" * 64 if Path(path).name == "smoke_contract.json" else original(path),
    )
    with pytest.raises(ValueError, match="lifecycle smoke evidence changed"):
        repair.validate_pilot_contract(CONFIG)


def test_missing_smoke_evidence_refuses_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    original = repair._sha256_file

    def sha256_or_missing(path: Path) -> str:
        if Path(path).name == "smoke_contract.json":
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(repair, "_sha256_file", sha256_or_missing)
    with pytest.raises(FileNotFoundError):
        repair.validate_pilot_contract(CONFIG)


def test_changed_decision_refuses_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    original = repair._sha256_file
    monkeypatch.setattr(
        repair,
        "_sha256_file",
        lambda path: "0" * 64 if Path(path) == DECISION else original(path),
    )
    with pytest.raises(ValueError, match="reviewed pilot-v4 decision changed"):
        repair.validate_pilot_contract(CONFIG)


def test_missing_reviewed_decision_refuses_contract(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["reviewed_v6_decision_path"] = str(tmp_path / "absent_decision.json")
    copied = tmp_path / "unreviewed.yaml"
    copied.write_text(yaml.safe_dump(config))
    with pytest.raises(FileNotFoundError):
        repair.validate_pilot_contract(copied)


def test_previous_staging_cannot_be_reused(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["output_dir"] = (
        "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_pilot_phase3i2_reviewed_v6_v2"
    )
    copied = tmp_path / "reused.yaml"
    copied.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="pilot-v4 output path mismatch"):
        repair.validate_pilot_contract(copied)
