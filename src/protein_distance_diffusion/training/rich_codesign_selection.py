"""Immutable production-batch selection verification for E006."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SELECTION_VERSION = "e006_production_batch_selection_v1"
CALIBRATION_VERSION = "e006_production_calibration_v2"
LENGTH_REGIMES = (128, 256, 384, 500)
SELECTION_CASE_FIELDS = (
    "stage",
    "target_length",
    "physical_batch_size",
    "accumulation_steps",
    "effective_token_budget",
    "maximum_pair_elements",
    "numerical_status",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "peak_total_device_occupancy_fraction",
    "peak_total_device_occupancy_mib",
    "remaining_free_memory_estimate_mib",
    "samples_per_second",
    "tokens_per_second",
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_identity(case: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(case["stage"]),
        int(case["target_length"]),
        int(case["physical_batch_size"]),
        int(case["accumulation_steps"]),
    )


def _calibration_projection(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": case["stage"],
        "target_length": case["target_length"],
        "physical_batch_size": case["physical_batch_size"],
        "accumulation_steps": case["accumulation_steps"],
        "effective_token_budget": case["effective_token_budget"],
        "maximum_pair_elements": case["pair_elements"] or None,
        "numerical_status": case["numerical_status"],
        "peak_cuda_allocated_mib": case["peak_cuda_allocated_mib"],
        "peak_cuda_reserved_mib": case["peak_cuda_reserved_mib"],
        "peak_total_device_occupancy_fraction": case["inferred_peak_total_device_occupancy_fraction"],
        "peak_total_device_occupancy_mib": case["inferred_peak_total_device_occupancy_mib"],
        "remaining_free_memory_estimate_mib": case["remaining_free_memory_estimate_mib"],
        "samples_per_second": case["samples_per_second"],
        "tokens_per_second": case["tokens_per_second"],
    }


def _selection_projection(case: dict[str, Any]) -> dict[str, Any]:
    return {name: case[name] for name in SELECTION_CASE_FIELDS}


def calibration_case_is_production_safe(
    case: dict[str, Any],
    *,
    maximum_occupancy_fraction: float = 0.90,
    minimum_remaining_mib: float = 768.0,
) -> bool:
    """Distinguish numerical correctness from the complete memory contract."""
    values = (
        case.get("inferred_peak_total_device_occupancy_fraction"),
        case.get("remaining_free_memory_estimate_mib"),
    )
    try:
        occupancy, remaining = (float(value) for value in values)
    except (TypeError, ValueError):
        return False
    return bool(
        case.get("status") == "passed"
        and case.get("finite") is True
        and case.get("numerical_status") == "equivalent"
        and math.isfinite(occupancy)
        and math.isfinite(remaining)
        and occupancy <= maximum_occupancy_fraction
        and remaining >= minimum_remaining_mib
    )


def verify_production_selection(
    selection_path: str | Path,
    expected_selection_sha256: str,
    *,
    calibration_path: str | Path,
    expected_calibration_sha256: str,
    stage: str,
    configured_regimes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify selection, source calibration cases, safety, and configured budgets."""
    if stage not in {"sequence-pretrain", "joint-train"}:
        raise ValueError(f"Unknown E006 production-selection stage: {stage}")
    if _sha256(selection_path) != expected_selection_sha256:
        raise ValueError("E006 production-selection SHA-256 contradiction")
    if _sha256(calibration_path) != expected_calibration_sha256:
        raise ValueError("E006 calibration-v2 SHA-256 contradiction")
    selection = json.loads(Path(selection_path).read_text())
    calibration = json.loads(Path(calibration_path).read_text())
    expected = {
        "status": "completed",
        "version": SELECTION_VERSION,
        "calibration_report_path": str(calibration_path),
        "calibration_report_sha256": expected_calibration_sha256,
        "authorizes_training": False,
        "training_performed": False,
    }
    contradictions = [name for name, value in expected.items() if selection.get(name) != value]
    if calibration.get("status") != "completed" or calibration.get("version") != CALIBRATION_VERSION:
        contradictions.append("calibration_status_or_version")
    memory_policy = selection.get("memory_policy", {})
    maximum_occupancy = float(memory_policy.get("maximum_total_device_occupancy_fraction", 0))
    minimum_remaining = float(memory_policy.get("minimum_remaining_vram_mib", 0))
    if maximum_occupancy != 0.90 or minimum_remaining != 768.0:
        contradictions.append("memory_policy")
    calibration_cases = {_case_identity(case): case for case in calibration.get("cases", [])}
    if len(calibration_cases) != len(calibration.get("cases", [])):
        contradictions.append("duplicate_calibration_case")
    selected = selection.get("selected_cases", [])
    rejected = selection.get("rejected_cases", [])
    selected_identities = [_case_identity(record) for record in selected]
    rejected_identities = [_case_identity(record) for record in rejected]
    if len(selected_identities) != len(set(selected_identities)):
        contradictions.append("duplicate_selected_case")
    if len(rejected_identities) != len(set(rejected_identities)):
        contradictions.append("duplicate_rejected_case")
    if set(selected_identities) & set(rejected_identities):
        contradictions.append("selected_rejected_overlap")
    for record in [*selected, *rejected]:
        source = calibration_cases.get(_case_identity(record))
        if source is None or _selection_projection(record) != _calibration_projection(source):
            contradictions.append(f"case_evidence:{_case_identity(record)}")
            continue
        safe = calibration_case_is_production_safe(
            source,
            maximum_occupancy_fraction=maximum_occupancy,
            minimum_remaining_mib=minimum_remaining,
        )
        if record.get("production_safe") is not safe:
            contradictions.append(f"production_safety:{_case_identity(record)}")
    selected_stage = [record for record in selected if record.get("stage") == stage]
    if tuple(int(record["target_length"]) for record in selected_stage) != LENGTH_REGIMES:
        contradictions.append("selected_length_regimes")
    if any(record.get("production_safe") is not True for record in selected_stage):
        contradictions.append("unsafe_selected_case")
    expected_regimes = [
        {
            "maximum_length": record["target_length"],
            "physical_batch_size": record["physical_batch_size"],
            "accumulation_steps": record["accumulation_steps"],
            "effective_token_budget": record["effective_token_budget"],
            "maximum_pair_elements": record["maximum_pair_elements"],
        }
        for record in selected_stage
    ]
    if configured_regimes != expected_regimes:
        contradictions.append("configured_batch_regimes")
    unsafe_500 = [record for record in rejected if _case_identity(record) == ("joint-train", 500, 2, 1)]
    if len(unsafe_500) != 1 or unsafe_500[0].get("production_safe") is not False:
        contradictions.append("length_500_batch_2_rejection")
    if contradictions:
        raise ValueError(f"E006 production-selection contradiction: {', '.join(contradictions)}")
    return {
        **selection,
        "recommendations": [
            {
                "stage": stage,
                "maximum_length": record["target_length"],
                "physical_batch_size": record["physical_batch_size"],
                "accumulation_steps": record["accumulation_steps"],
                "effective_token_budget": record["effective_token_budget"],
                "maximum_pair_elements": record["maximum_pair_elements"],
            }
            for record in selected_stage
        ],
    }
