from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_calibration_v2 import (
    CALIBRATION_V2_VERSION,
    compare_numerical_outputs,
    evaluate_case_safety,
    infer_peak_device_memory,
    run_case_isolated,
    select_v2_recommendations,
    validate_calibration_v2_config,
)
from protein_distance_diffusion.training.rich_codesign_production import (
    _canonical_hash,
    collate_sequence_pretraining,
    validate_phase3_config,
)
from protein_distance_diffusion.training.rich_codesign_smoke import _synthetic_row

V1_REPORT = Path("reports/experiments/E006_rich_geometry_codesign/phase3_calibration_v1/report.json")
V1_SHA256 = "a4d68b10cd252fd267b17e9b8faf86ff59d7778cea652ffafafe8dc94eccf572"


def _case(
    stage: str,
    length: int,
    batch: int,
    accumulation: int,
    throughput: float,
    *,
    occupancy: float = 0.8,
    remaining: float = 1000,
) -> dict:
    return {
        "stage": stage,
        "target_length": length,
        "physical_batch_size": batch,
        "accumulation_steps": accumulation,
        "effective_token_budget": length * batch * accumulation,
        "pair_elements": 0 if stage == "sequence-pretrain" else batch * length**2,
        "status": "passed",
        "finite": True,
        "numerical_status": "equivalent",
        "inferred_peak_total_device_occupancy_mib": occupancy * 8000,
        "inferred_peak_total_device_occupancy_fraction": occupancy,
        "remaining_free_memory_estimate_mib": remaining,
        "peak_cuda_allocated_mib": 3000,
        "peak_cuda_reserved_mib": 3500,
        "samples_per_second": throughput / length,
        "tokens_per_second": throughput,
    }


def test_external_usage_is_included_in_total_device_peak() -> None:
    result = infer_peak_device_memory(
        {"total_vram_mib": 8000, "external_baseline_usage_mib": 1200},
        peak_allocated_mib=5000,
        peak_reserved_mib=5500,
    )
    assert result["inferred_peak_total_device_occupancy_mib"] == 6700
    assert result["remaining_free_memory_estimate_mib"] == 1300


@pytest.mark.parametrize(
    ("occupancy", "remaining", "safe", "reason"),
    [
        (0.90, 768, True, None),
        (0.90001, 768, False, "total_device_occupancy_above_maximum"),
        (0.80, 767.99, False, "remaining_vram_below_minimum"),
    ],
)
def test_dual_memory_safety_contract(occupancy, remaining, safe, reason) -> None:
    case = _case("joint-train", 128, 1, 4, 100, occupancy=occupancy, remaining=remaining)
    passed, reasons = evaluate_case_safety(case)
    assert passed is safe
    if reason:
        assert reason in reasons


def test_oom_and_nonfinite_telemetry_are_never_safe() -> None:
    case = _case("joint-train", 128, 1, 4, 100)
    case.update(status="cuda_oom", numerical_status="failed")
    assert evaluate_case_safety(case)[0] is False
    case.update(status="passed", numerical_status="equivalent", inferred_peak_total_device_occupancy_fraction=math.nan)
    assert evaluate_case_safety(case)[0] is False


def test_numerical_equivalence_is_scientific_not_just_finiteness() -> None:
    import torch

    reference = {"prediction": torch.tensor([1.0, 2.0])}
    close = {"prediction": torch.tensor([1.001, 1.999])}
    changed = {"prediction": torch.tensor([1.0, 3.0])}
    assert compare_numerical_outputs(reference, close, names=("prediction",), atol=0.01, rtol=0.0)[0]
    equivalent, diagnostics = compare_numerical_outputs(
        reference,
        changed,
        names=("prediction",),
        atol=0.01,
        rtol=0.0,
    )
    assert equivalent is False
    assert diagnostics["prediction"]["maximum_absolute_error"] == 1.0


def test_stage_a_collation_never_constructs_pair_features() -> None:
    batch = collate_sequence_pretraining([_synthetic_row("a", 12)])
    assert set(batch) == {"sample_ids", "sequence_token_ids", "residue_mask", "lengths"}
    assert not any("pair" in key or "geometry" in key for key in batch)


def test_selection_reduces_microsteps_then_uses_throughput_not_vram() -> None:
    stage = {
        "stage": "sequence-pretrain",
        "regimes": [{"target_length": 128, "target_effective_token_budget": 512, "candidates": []}],
    }
    cases = [
        _case("sequence-pretrain", 128, 1, 4, 100, occupancy=0.5),
        _case("sequence-pretrain", 128, 2, 2, 90, occupancy=0.85),
        _case("sequence-pretrain", 128, 4, 1, 80, occupancy=0.88),
    ]
    recommendations, annotated = select_v2_recommendations(cases, [stage])
    assert recommendations[0]["physical_batch_size"] == 4
    assert next(item for item in annotated if item["physical_batch_size"] == 4)["selection_decision"] == "selected"

    equal_microsteps = [
        _case("sequence-pretrain", 128, 4, 1, 100, occupancy=0.88),
        _case("sequence-pretrain", 128, 4, 1, 120, occupancy=0.60),
    ]
    # Candidate identity must be unique in production; this direct policy test
    # demonstrates that occupancy is absent from the ranking key.
    recommendations, _ = select_v2_recommendations(equal_microsteps, [stage])
    assert recommendations[0]["tokens_per_second"] == 120


def test_v2_config_and_v1_artifact_are_immutable() -> None:
    config = load_yaml("configs/e006_rich_geometry_production_calibration_v2.yaml")
    validate_calibration_v2_config(config)
    validate_phase3_config(config, mode="calibrate")
    assert config["calibration"]["version"] == CALIBRATION_V2_VERSION
    import hashlib

    assert hashlib.sha256(V1_REPORT.read_bytes()).hexdigest() == V1_SHA256
    assert "phase3_calibration_v1" not in config["calibration"]["output_report"]


def test_case_definitions_retain_v1_and_add_only_bounded_joint_cases() -> None:
    v1 = load_yaml("configs/e006_rich_geometry_production_calibration.yaml")
    v2 = load_yaml("configs/e006_rich_geometry_production_calibration_v2.yaml")
    joint = next(item for item in v2["calibration"]["stages"] if item["stage"] == "joint-train")
    by_length = {
        item["target_length"]: {(c["physical_batch_size"], c["accumulation_steps"]) for c in item["candidates"]}
        for item in joint["regimes"]
    }
    for regime in v1["calibration"]["regimes"]:
        assert {(c["physical_batch_size"], c["accumulation_steps"]) for c in regime["candidates"]} <= by_length[
            regime["target_length"]
        ]
    assert by_length[128] == {(1, 4), (2, 2), (4, 1), (6, 1)}
    assert by_length[256] == {(1, 2), (2, 1), (3, 1), (4, 1)}
    assert by_length[384] == {(1, 1), (2, 1)}
    assert by_length[500] == {(1, 1), (2, 1)}


def test_configuration_hash_is_stable_json_semantics() -> None:
    config = load_yaml("configs/e006_rich_geometry_production_calibration_v2.yaml")
    assert _canonical_hash(config) == _canonical_hash(json.loads(json.dumps(config)))


def test_case_execution_uses_a_child_and_publishes_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"seed": 1, "device": "cpu"}))
    result_path = tmp_path / "case.json"
    result = run_case_isolated(
        config_path,
        {
            "stage": "sequence-pretrain",
            "target_length": 128,
            "physical_batch_size": 1,
            "accumulation_steps": 1,
        },
        result_path,
        timeout_seconds=30,
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result_path.exists()
