"""Plan-only provenance tests for the E007 matrix-generator audit."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_e007_matrix_generator.py"
SPEC = importlib.util.spec_from_file_location("evaluate_e007_matrix_generator", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_fixture(tmp_path: Path) -> Path:
    files = {}
    for name in ("checkpoint", "training_config", "normalization", "train_manifest", "validation_manifest"):
        path = tmp_path / name
        path.write_text('{"mode":"scale","scale":2.0}' if name == "normalization" else name, encoding="utf-8")
        files[name] = path
    protocol = tmp_path / "protocol.json"
    payload = {
        "status": "completed",
        "checkpoint_path": "checkpoint",
        "checkpoint_sha256": MODULE.sha256_file(files["checkpoint"]),
        "training_config_path": "training_config",
        "training_config_sha256": MODULE.sha256_file(files["training_config"]),
        "normalization_path": "normalization",
        "normalization_sha256": MODULE.sha256_file(files["normalization"]),
        "reference_manifest_path": "validation_manifest",
        "reference_manifest_sha256": MODULE.sha256_file(files["validation_manifest"]),
    }
    protocol.write_text(json.dumps(payload), encoding="utf-8")
    records = {"generator_protocol": protocol, "generator_checkpoint": files["checkpoint"]}
    records.update(
        {
            "generator_training_config": files["training_config"],
            "normalization": files["normalization"],
            "train_manifest": files["train_manifest"],
            "validation_manifest": files["validation_manifest"],
        }
    )
    config = {
        "version": MODULE.PLAN_VERSION,
        "experiment_id": "test",
        "output_dir": "output",
        "generator": {
            "checkpoint_path": "checkpoint",
            "checkpoint_sha256": MODULE.sha256_file(files["checkpoint"]),
            "training_config_path": "training_config",
            "training_config_sha256": MODULE.sha256_file(files["training_config"]),
            "normalization_path": "normalization",
            "normalization_sha256": MODULE.sha256_file(files["normalization"]),
        },
        "dataset": {
            "train_manifest_sha256": MODULE.sha256_file(files["train_manifest"]),
            "validation_manifest_path": "validation_manifest",
            "validation_manifest_sha256": MODULE.sha256_file(files["validation_manifest"]),
        },
        "matrix_representation": {"normalization": {"mode": "scale", "scale_angstrom": 2.0}},
        "audit": {
            "lengths": [3],
            "generated_counts_by_length": {3: 2},
            "real_counts_by_length": {3: 2},
            "panels": ["real_validation_matrices", "independently_generated_candidate_matrices"],
            "corruption_controls": {"asymmetry": {"delta_angstrom": 1.0}},
            "missing_valid_pair_control_supported": False,
            "scientific_threshold_policy": "reference_quantile_warnings_only",
        },
        "fatal_contract_thresholds": {"require_finite": True},
        "bounds": {
            "maximum_rss_mib": 64,
            "process_candidates_individually": True,
            "prohibit_cross_candidate_averaging": True,
        },
        "protected_inputs": {
            key: {"path": path.name, "sha256": MODULE.sha256_file(path)} for key, path in records.items()
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_plan_is_verified_and_non_authorizing(tmp_path: Path) -> None:
    plan = MODULE.build_plan(_write_fixture(tmp_path), repository_root=tmp_path)
    assert plan["protocol_contract_verified"]
    assert not plan["authorizes_training"]
    assert not plan["authorizes_joint_training"]
    assert not plan["real_audit_executed"]
    assert not plan["cross_candidate_averaging"]


def test_existing_output_is_refused(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    (tmp_path / "output").mkdir()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        MODULE.build_plan(config, repository_root=tmp_path)


def test_protected_input_contradiction_is_refused(tmp_path: Path) -> None:
    config = _write_fixture(tmp_path)
    (tmp_path / "checkpoint").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        MODULE.build_plan(config, repository_root=tmp_path)


def test_cli_modes_are_required_and_mutually_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--config", "config.yaml"])
    with pytest.raises(SystemExit):
        MODULE.parse_args()
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--config", "config.yaml", "--plan-only", "--evaluate"])
    with pytest.raises(SystemExit):
        MODULE.parse_args()
