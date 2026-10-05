from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_production import (
    _planned_optimizer_updates,
    recommendation_for_length,
    validate_batch_budget,
)
from protein_distance_diffusion.training.rich_codesign_selection import (
    calibration_case_is_production_safe,
    verify_production_selection,
)

CALIBRATION = Path("reports/experiments/E006_rich_geometry_codesign/phase3_calibration_v2/report.json")
CALIBRATION_SHA = "47c9d9548ccdc037df15e5e12879b09e1a45ec9ed9dcf987192ccee5d6cf2c2a"
SELECTION = Path("reports/experiments/E006_rich_geometry_codesign/phase3_production_batch_selection_v1.json")
SELECTION_SHA = "eee314b655cb0614635dd8f152964bf2732876b921483c51e6a1000a8f5a6e31"


def _verify(config_path: str) -> dict:
    config = load_yaml(config_path)
    stage = "sequence-pretrain" if "sequence_pretrain" in config_path else "joint-train"
    return verify_production_selection(
        SELECTION,
        SELECTION_SHA,
        calibration_path=CALIBRATION,
        expected_calibration_sha256=CALIBRATION_SHA,
        stage=stage,
        configured_regimes=config["batching"]["regimes"],
    )


def test_immutable_report_and_selection_hashes_and_exact_budgets() -> None:
    assert hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() == CALIBRATION_SHA
    assert hashlib.sha256(SELECTION.read_bytes()).hexdigest() == SELECTION_SHA
    stage_a = _verify("configs/e006_rich_geometry_sequence_pretrain.yaml")["recommendations"]
    stage_b = _verify("configs/e006_rich_geometry_joint_train.yaml")["recommendations"]
    assert [
        (row["physical_batch_size"], row["accumulation_steps"], row["effective_token_budget"]) for row in stage_a
    ] == [
        (32, 1, 4096),
        (16, 1, 4096),
        (8, 1, 3072),
        (8, 1, 4000),
    ]
    assert [
        (
            row["physical_batch_size"],
            row["accumulation_steps"],
            row["effective_token_budget"],
            row["maximum_pair_elements"],
        )
        for row in stage_b
    ] == [
        (6, 1, 768, 98304),
        (4, 1, 1024, 262144),
        (2, 1, 768, 294912),
        (1, 1, 500, 254016),
    ]


def test_numerical_equivalence_does_not_override_memory_gates() -> None:
    report = json.loads(CALIBRATION.read_text())
    unsafe = next(
        row
        for row in report["cases"]
        if row["stage"] == "joint-train" and row["target_length"] == 500 and row["physical_batch_size"] == 2
    )
    assert unsafe["numerical_status"] == "equivalent"
    assert unsafe["remaining_free_memory_estimate_mib"] == 193.0
    assert calibration_case_is_production_safe(unsafe) is False


def test_selection_and_calibration_hash_tampering_fail() -> None:
    config = load_yaml("configs/e006_rich_geometry_joint_train.yaml")
    kwargs = {
        "calibration_path": CALIBRATION,
        "expected_calibration_sha256": CALIBRATION_SHA,
        "stage": "joint-train",
        "configured_regimes": config["batching"]["regimes"],
    }
    with pytest.raises(ValueError, match="production-selection SHA-256"):
        verify_production_selection(SELECTION, "0" * 64, **kwargs)
    with pytest.raises(ValueError, match="calibration-v2 SHA-256"):
        verify_production_selection(
            SELECTION,
            SELECTION_SHA,
            **{**kwargs, "expected_calibration_sha256": "0" * 64},
        )


def test_relabeling_unsafe_case_as_selected_still_fails(tmp_path: Path) -> None:
    selection = json.loads(SELECTION.read_text())
    unsafe = selection["rejected_cases"][0]
    unsafe["production_safe"] = True
    selection["selected_cases"][-1] = unsafe
    path = tmp_path / "tampered-selection.json"
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = load_yaml("configs/e006_rich_geometry_joint_train.yaml")
    with pytest.raises(ValueError, match="production-selection contradiction"):
        verify_production_selection(
            path,
            digest,
            calibration_path=CALIBRATION,
            expected_calibration_sha256=CALIBRATION_SHA,
            stage="joint-train",
            configured_regimes=config["batching"]["regimes"],
        )


def test_stage_specific_batch_planning_and_pair_budget_semantics() -> None:
    stage_a = _verify("configs/e006_rich_geometry_sequence_pretrain.yaml")["recommendations"]
    stage_b = _verify("configs/e006_rich_geometry_joint_train.yaml")["recommendations"]
    sequence = recommendation_for_length(128, stage_a, stage="sequence-pretrain")
    joint = recommendation_for_length(500, stage_b, stage="joint-train")
    assert sequence.constructs_pair_features is False
    assert joint.constructs_pair_features is True
    assert (
        validate_batch_budget(
            [128] * 32,
            recommendation=sequence,
            maximum_residues=4096,
            maximum_pair_elements=0,
        )["pair_elements"]
        == 0
    )
    assert (
        _planned_optimizer_updates(
            [128] * 32 + [256] * 16 + [384] * 8 + [500] * 8,
            stage_a,
            stage="sequence-pretrain",
            dataset_passes=3,
        )
        == 12
    )
