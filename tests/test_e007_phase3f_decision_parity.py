from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.evaluation.e007_phase3f_decision_parity import (
    corrected_classification,
    decision_audit,
    pareto_analysis,
)
from protein_distance_diffusion.training.e007_coordinate_real_pilot import _sampling_metrics

SOURCE = Path("reports/experiments/E007_matrix_sequence_cogeneration/coordinate_real_pilot_v1")


def _completed_evidence() -> tuple[dict, dict, list[dict], dict, dict, dict]:
    report = json.loads((SOURCE / "report.json").read_text())
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    metrics = [json.loads(line) for line in (SOURCE / "metrics.jsonl").read_text().splitlines()]
    evaluations = {int(key): value for key, value in json.loads((SOURCE / "evaluations.json").read_text()).items()}
    samplings = {int(key): value for key, value in json.loads((SOURCE / "sampling.json").read_text()).items()}
    config = yaml.safe_load(Path("configs/e007_coordinate_real_pilot_v1.yaml").read_text())
    return report, protocol, metrics, evaluations, samplings, config


def _audit(**changes: object) -> dict:
    report, protocol, metrics, evaluations, samplings, config = _completed_evidence()
    report.update(changes.get("report", {}))
    config.update(changes.get("config", {}))
    checkpoint = {
        "parameter_change_observed_between_first_and_last_checkpoint": True,
    }
    return decision_audit(
        report,
        protocol,
        metrics,
        evaluations,
        samplings,
        config,
        {"all_canonical_diagonals_exact_zero": True},
        checkpoint,
    )


def test_completed_report_reproduces_legacy_failure_and_corrected_decision() -> None:
    result = _audit()
    assert result["published_classification"] == "numerical_or_memory_failure"
    assert result["legacy_classification"] == "numerical_or_memory_failure"
    assert result["corrected_classification"] == "denoising_learned_but_sampling_not_learned"
    assert result["classifier_defect_confirmed"] is True
    diagonal = next(row for row in result["decision_table"] if row["gate_name"] == "reported_cdist_diagonal_exact_zero")
    assert diagonal["passed"] is False
    canonical = next(
        row for row in result["decision_table"] if row["gate_name"] == "canonical_distance_diagonal_exact_zero"
    )
    assert canonical["passed"] is True


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"numerical_checks_pass": False, "denoising_checks_pass": True, "sampling_checks_pass": True},
            "numerical_or_memory_failure",
        ),
        (
            {"numerical_checks_pass": True, "denoising_checks_pass": True, "sampling_checks_pass": True},
            "real_data_learning_and_sampling_verified",
        ),
        (
            {"numerical_checks_pass": True, "denoising_checks_pass": True, "sampling_checks_pass": False},
            "denoising_learned_but_sampling_not_learned",
        ),
        (
            {"numerical_checks_pass": True, "denoising_checks_pass": False, "sampling_checks_pass": False},
            "length_limited_learning",
        ),
    ],
)
def test_corrected_classification_branches(kwargs: dict, expected: str) -> None:
    assert (
        corrected_classification(
            **kwargs,
            global_improvement=0.1,
            maximum_length_regression=0.06,
            maximum_allowed_length_regression=0.05,
        )
        == expected
    )
    if expected == "length_limited_learning":
        assert (
            corrected_classification(
                **kwargs,
                global_improvement=0.1,
                maximum_length_regression=0.05,
                maximum_allowed_length_regression=0.05,
            )
            == "insufficient_real_data_learning"
        )


def test_memory_is_compared_as_peak_mib_with_inclusive_boundary() -> None:
    report, protocol, metrics, evaluations, samplings, config = _completed_evidence()
    report = copy.deepcopy(report)
    config = copy.deepcopy(config)
    report["memory"]["peak_cuda_allocated_mib"] = 6144.0
    report["memory"]["peak_cuda_reserved_mib"] = 7680.0
    result = decision_audit(
        report,
        protocol,
        metrics,
        evaluations,
        samplings,
        config,
        {"all_canonical_diagonals_exact_zero": True},
        {"parameter_change_observed_between_first_and_last_checkpoint": True},
    )
    gates = {row["gate_name"]: row for row in result["decision_table"]}
    assert gates["cuda_peak_allocated_mib"]["passed"] is True
    assert gates["cuda_peak_reserved_mib"]["passed"] is True
    report["memory"]["peak_cuda_allocated_mib"] = 6144.0001
    result = decision_audit(
        report,
        protocol,
        metrics,
        evaluations,
        samplings,
        config,
        {"all_canonical_diagonals_exact_zero": True},
        {"parameter_change_observed_between_first_and_last_checkpoint": True},
    )
    gate = next(row for row in result["decision_table"] if row["gate_name"] == "cuda_peak_allocated_mib")
    assert gate["passed"] is False


def test_missing_numerical_field_refuses_instead_of_defaulting_true() -> None:
    report, protocol, metrics, evaluations, samplings, config = _completed_evidence()
    del samplings[1000]["records"][0]["finite"]
    with pytest.raises(ValueError, match=r"missing sampling record\.finite"):
        decision_audit(
            report,
            protocol,
            metrics,
            evaluations,
            samplings,
            config,
            {"all_canonical_diagonals_exact_zero": True},
            {"parameter_change_observed_between_first_and_last_checkpoint": True},
        )


def test_sampling_metrics_use_canonical_exact_diagonal() -> None:
    coordinates = torch.tensor([[[0.1234567, 1.234567, 2.345678], [3.456789, 4.567891, 5.678912]]], dtype=torch.float32)
    metrics = _sampling_metrics(coordinates)
    assert metrics["distance_diagonal_error"] == 0.0
    assert metrics["distance_symmetry_error"] == 0.0
    assert metrics["strict_euclidean_valid"] is True


def test_pareto_analysis_uses_no_scalar_and_retains_separate_candidates() -> None:
    _, _, _, evaluations, samplings, _ = _completed_evidence()
    result = pareto_analysis(evaluations, samplings)
    assert result["best_denoising_candidate"] == 1000
    assert result["best_sampling_candidates"]
    assert "scalar" in result["best_sampling_selection"]
    assert set(result["combined_frontier"]).issubset({250, 500, 750, 1000})
    assert result["initialization_baseline_update"] == 0
    assert result["initialization_baseline_is_checkpoint"] is False
    assert "sampling_quality_score" not in {key for row in result["candidates"] for key in row}
