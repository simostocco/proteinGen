"""Read-only decision-parity audit for the completed E007 Phase-3F pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from protein_distance_diffusion.training.coordinate_diffusion import coordinates_to_distance_matrix
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file
from protein_distance_diffusion.training.e007_coordinate_real_pilot import EXPECTED_SCALE, classify_pilot

VERSION = "e007_phase3f_decision_parity_v1"
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "dataset_modified": False,
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"E007 Phase-3F decision evidence missing {context}.{key}")
    return mapping[key]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _gate(
    name: str,
    observed: Any,
    required: str,
    passed: bool,
    source: str,
    branch: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "gate_name": name,
        "observed_value": observed,
        "required_value": required,
        "passed": bool(passed),
        "report_path": source,
        "classification_branch_affected": branch,
        **extra,
    }


def _quality(records: Iterable[Mapping[str, Any]]) -> float:
    values = []
    for record in records:
        values.append(
            float(_require(record, "adjacent_reference_error_angstrom", "sampling record"))
            + float(_require(record, "radius_of_gyration_reference_relative_error", "sampling record"))
            + float(_require(record, "clash_reference_error", "sampling record"))
            + float(_require(record, "contact_density_reference_error", "sampling record"))
        )
    if not values:
        raise ValueError("E007 Phase-3F sampling records are empty")
    return float(np.mean(values))


def _relative_improvement(initial: float, final: float) -> float:
    if not _finite_number(initial) or not _finite_number(final) or initial <= 0:
        raise ValueError("E007 Phase-3F relative-improvement inputs are invalid")
    return 1.0 - final / initial


def _canonical_sample_checks(source_dir: Path, samplings: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for update, payload in sorted(samplings.items()):
        for record in _require(payload, "records", f"sampling[{update}]"):
            artifact = Path(_require(record, "artifact_path", "sampling record"))
            if not artifact.is_absolute() and not artifact.exists():
                artifact = source_dir / "samples" / f"update-{update:04d}" / artifact.name
            if not artifact.is_file():
                raise FileNotFoundError(f"E007 Phase-3F sample artifact is missing: {artifact}")
            with np.load(artifact) as archive:
                coordinates = np.asarray(archive["coordinates"])
                stored = np.asarray(archive["distance_matrix"])
            tensor = torch.from_numpy(coordinates)
            canonical = coordinates_to_distance_matrix(tensor * EXPECTED_SCALE, diagnostic_float64=True)
            if canonical.ndim == 3:
                canonical = canonical[0]
            canonical_diagonal = float(canonical.diagonal().abs().max())
            canonical_symmetry = float((canonical - canonical.T).abs().max())
            stored_diagonal = float(np.max(np.abs(np.diagonal(stored, axis1=-2, axis2=-1))))
            records.append(
                {
                    "optimizer_update": update,
                    "length_stratum": record["length_stratum"],
                    "sample_index": record["sample_index"],
                    "artifact_path": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "reported_cdist_diagonal_error": float(record["distance_diagonal_error"]),
                    "stored_distance_diagonal_error": stored_diagonal,
                    "canonical_distance_diagonal_error": canonical_diagonal,
                    "canonical_distance_symmetry_error": canonical_symmetry,
                }
            )
    return {
        "record_count": len(records),
        "all_canonical_diagonals_exact_zero": all(x["canonical_distance_diagonal_error"] == 0 for x in records),
        "all_canonical_symmetric": all(x["canonical_distance_symmetry_error"] <= 1e-8 for x in records),
        "reported_cdist_diagonal_error_range": [
            min(x["reported_cdist_diagonal_error"] for x in records),
            max(x["reported_cdist_diagonal_error"] for x in records),
        ],
        "records": records,
    }


def _means(records: list[Mapping[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    return {key: float(np.mean([float(_require(row, key, "sampling record")) for row in records])) for key in keys}


def _dominates(left: Mapping[str, float], right: Mapping[str, float], keys: list[str]) -> bool:
    return all(left[key] <= right[key] for key in keys) and any(left[key] < right[key] for key in keys)


def corrected_classification(
    *,
    numerical_checks_pass: bool,
    denoising_checks_pass: bool,
    sampling_checks_pass: bool,
    global_improvement: float,
    maximum_length_regression: float,
    maximum_allowed_length_regression: float,
) -> str:
    """Apply the Phase-3F branches after corrected numerical validation."""
    if not numerical_checks_pass:
        return "numerical_or_memory_failure"
    if denoising_checks_pass and sampling_checks_pass:
        return "real_data_learning_and_sampling_verified"
    if denoising_checks_pass:
        return "denoising_learned_but_sampling_not_learned"
    if global_improvement > 0 and maximum_length_regression > maximum_allowed_length_regression:
        return "length_limited_learning"
    return "insufficient_real_data_learning"


def _model_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_parameter_evidence(source_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted((source_dir / "checkpoints").glob("step-*.pt")):
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"E007 Phase-3F checkpoint metadata hash contradiction: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        update = int(_require(payload, "optimizer_update", f"checkpoint {path.name}"))
        if update != int(_require(metadata, "optimizer_update", f"checkpoint metadata {path.name}")):
            raise ValueError(f"E007 Phase-3F checkpoint update contradiction: {path}")
        records.append(
            {
                "optimizer_update": update,
                "path": str(path),
                "checkpoint_sha256": metadata["sha256"],
                "model_parameter_sha256": _model_state_sha256(_require(payload, "model", f"checkpoint {path.name}")),
                "successful_optimizer_boundary": payload.get("successful_optimizer_boundary") is True,
            }
        )
        del payload
    if not records:
        raise ValueError("E007 Phase-3F checkpoint parameter evidence is absent")
    return {
        "records": records,
        "parameter_change_observed_between_first_and_last_checkpoint": (
            records[0]["model_parameter_sha256"] != records[-1]["model_parameter_sha256"]
        ),
        "update_zero_parameter_hash_available": False,
        "interpretation": (
            "Parameter change is directly verified from update 250 through update 1000; "
            "no update-0 checkpoint was published."
        ),
    }


def pareto_analysis(
    evaluations: Mapping[int, Mapping[str, Any]], samplings: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    # Update zero is an initialization baseline, not a durable checkpoint.
    updates = sorted(update for update in set(evaluations) & set(samplings) if update > 0)
    denoising_keys = [
        "validation_coordinate_v_mse",
        "validation_x0_coordinate_rmse_angstrom",
        "validation_pair_distance_rmse_angstrom",
        "validation_adjacent_distance_error_angstrom",
        "validation_radius_of_gyration_error_angstrom",
        "validation_clash_fraction",
        "validation_high_noise_coordinate_v_mse",
        "validation_very_high_noise_coordinate_v_mse",
    ]
    sampling_keys = [
        "sampling_adjacent_reference_error_angstrom",
        "sampling_radius_of_gyration_reference_relative_error",
        "sampling_clash_reference_error",
        "sampling_contact_density_reference_error",
    ]
    candidates: list[dict[str, Any]] = []
    for update in updates:
        validation = evaluations[update]["validation"]
        global_metrics = validation["global"]
        sampling_means = _means(
            samplings[update]["records"],
            [
                "adjacent_reference_error_angstrom",
                "radius_of_gyration_reference_relative_error",
                "clash_reference_error",
                "contact_density_reference_error",
                "non_neighbor_clash_fraction",
            ],
        )
        candidates.append(
            {
                "optimizer_update": update,
                "validation_coordinate_v_mse": global_metrics["coordinate_v_mse"],
                "validation_x0_coordinate_rmse_angstrom": global_metrics["x0_coordinate_rmse_angstrom"],
                "validation_pair_distance_rmse_angstrom": global_metrics["pair_distance_rmse_angstrom"],
                "validation_adjacent_distance_error_angstrom": global_metrics["adjacent_distance_error_angstrom"],
                "validation_radius_of_gyration_error_angstrom": global_metrics["radius_of_gyration_error_angstrom"],
                "validation_clash_fraction": global_metrics["clash_fraction"],
                "validation_high_noise_coordinate_v_mse": validation["by_timestep_bin"]["high_noise"][
                    "coordinate_v_mse"
                ],
                "validation_very_high_noise_coordinate_v_mse": validation["by_timestep_bin"]["very_high_noise"][
                    "coordinate_v_mse"
                ],
                "sampling_adjacent_reference_error_angstrom": sampling_means["adjacent_reference_error_angstrom"],
                "sampling_radius_of_gyration_reference_relative_error": sampling_means[
                    "radius_of_gyration_reference_relative_error"
                ],
                "sampling_clash_reference_error": sampling_means["clash_reference_error"],
                "sampling_contact_density_reference_error": sampling_means["contact_density_reference_error"],
                "sampling_non_neighbor_clash_fraction": sampling_means["non_neighbor_clash_fraction"],
            }
        )

    def frontier(keys: list[str]) -> tuple[list[int], dict[str, list[int]]]:
        nondominated = []
        dominated_by: dict[str, list[int]] = {}
        for candidate in candidates:
            dominators = [
                other["optimizer_update"]
                for other in candidates
                if other is not candidate and _dominates(other, candidate, keys)
            ]
            if dominators:
                dominated_by[str(candidate["optimizer_update"])] = dominators
            else:
                nondominated.append(candidate["optimizer_update"])
        return nondominated, dominated_by

    denoising_frontier, denoising_dominated = frontier(denoising_keys)
    sampling_frontier, sampling_dominated = frontier(sampling_keys)
    combined_frontier, combined_dominated = frontier(denoising_keys + sampling_keys)
    return {
        "direction": "all declared objectives are minimized independently; no scalar score is used",
        "initialization_baseline_update": 0,
        "initialization_baseline_is_checkpoint": False,
        "candidates": candidates,
        "denoising_objectives": denoising_keys,
        "sampling_objectives": sampling_keys,
        "denoising_frontier": denoising_frontier,
        "denoising_dominated_by": denoising_dominated,
        "sampling_frontier": sampling_frontier,
        "sampling_dominated_by": sampling_dominated,
        "combined_frontier": combined_frontier,
        "combined_dominated_by": combined_dominated,
        "best_denoising_candidate": min(candidates, key=lambda x: x["validation_coordinate_v_mse"])["optimizer_update"],
        "best_sampling_candidates": sampling_frontier,
        "best_sampling_selection": "set-valued Pareto result; no undeclared scalar tie-breaker",
    }


def decision_audit(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    metrics: list[Mapping[str, Any]],
    evaluations: Mapping[int, Mapping[str, Any]],
    samplings: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    canonical_checks: Mapping[str, Any],
    checkpoint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_report = "report.json"
    source_metrics = "metrics.jsonl"
    source_evaluations = "evaluations.json"
    source_sampling = "sampling.json"
    threshold = _require(config, "classification_thresholds", "config")
    memory_limits = _require(config, "memory", "config")
    if 0 not in evaluations or 1000 not in evaluations or 0 not in samplings or 1000 not in samplings:
        raise ValueError("E007 Phase-3F required update-0/update-1000 evidence is missing")
    initial = evaluations[0]["validation"]
    final = evaluations[1000]["validation"]
    final_records = _require(samplings[1000], "records", "sampling[1000]")
    finite_losses = bool(metrics) and all(
        _finite_number(_require(row, "coordinate_v_mse", "metric")) for row in metrics
    )
    finite_gradients = bool(metrics) and all(
        _finite_number(_require(row, "gradient_norm", "metric"))
        and int(_require(row, "nonfinite_count", "metric")) == 0
        for row in metrics
    )
    gradient_coverage = bool(metrics) and all(
        all(
            _finite_number(value) and float(value) > 0
            for value in _require(row, "gradient_group_norms", "metric").values()
        )
        for row in metrics
    )
    global_improvement = _relative_improvement(
        initial["global"]["coordinate_v_mse"], final["global"]["coordinate_v_mse"]
    )
    high_improvement = _relative_improvement(
        initial["by_timestep_bin"]["high_noise"]["coordinate_v_mse"],
        final["by_timestep_bin"]["high_noise"]["coordinate_v_mse"],
    )
    very_high_improvement = _relative_improvement(
        initial["by_timestep_bin"]["very_high_noise"]["coordinate_v_mse"],
        final["by_timestep_bin"]["very_high_noise"]["coordinate_v_mse"],
    )
    length_regressions = {
        item["name"]: final["by_length_stratum"][item["name"]]["coordinate_v_mse"]
        / initial["by_length_stratum"][item["name"]]["coordinate_v_mse"]
        - 1
        for item in config["length_strata"]
    }
    sampling_improvement = _relative_improvement(_quality(samplings[0]["records"]), _quality(final_records))
    backend = _require(report, "numerical_backend", "report")
    during = _require(backend, "during", "numerical_backend")
    policy = _require(backend, "policy", "numerical_backend")
    backend_matches = all(during.get(key) == policy.get(key) for key in policy if key != "equivariance_policy")
    equivariance = [
        evaluations[1000][split]["equivariance"]["criterion"]["passed"] for split in ("train", "validation")
    ]
    memory = _require(report, "memory", "report")
    allocated = float(_require(memory, "peak_cuda_allocated_mib", "report.memory"))
    reserved = float(_require(memory, "peak_cuda_reserved_mib", "report.memory"))
    final_finite = all(bool(_require(row, "finite", "sampling record")) for row in final_records)
    masking = all(bool(_require(row, "masking_exact", "sampling record")) for row in final_records)
    centering_limit = 5e-5 * EXPECTED_SCALE
    final_centering = max(
        float(_require(row, "centering_max_abs_angstrom", "sampling record")) for row in final_records
    )
    final_symmetry = max(float(_require(row, "distance_symmetry_error", "sampling record")) for row in final_records)
    final_reported_diagonal = max(
        float(_require(row, "distance_diagonal_error", "sampling record")) for row in final_records
    )
    final_negative_mass = max(
        float(_require(row, "negative_gram_eigenvalue_mass_fraction", "sampling record")) for row in final_records
    )
    geometry_failures = [
        {
            "length_stratum": row["length_stratum"],
            "sample_index": row["sample_index"],
            "failed": [
                name
                for name, passed in (
                    (
                        "adjacent_distance",
                        float(row["adjacent_reference_error_angstrom"])
                        <= float(threshold["maximum_adjacent_distance_error_angstrom"]),
                    ),
                    (
                        "radius_of_gyration",
                        float(row["radius_of_gyration_reference_relative_error"])
                        <= float(threshold["maximum_radius_of_gyration_relative_error"]),
                    ),
                    (
                        "clash_fraction",
                        float(row["non_neighbor_clash_fraction"])
                        <= float(threshold["maximum_non_neighbor_clash_fraction"]),
                    ),
                )
                if not passed
            ],
        }
        for row in final_records
    ]
    geometry_failures = [row for row in geometry_failures if row["failed"]]
    gates = [
        _gate("finite_training_losses", finite_losses, "all 1,000 finite", finite_losses, source_metrics, "numerical"),
        _gate("finite_gradients", finite_gradients, "all finite", finite_gradients, source_metrics, "numerical"),
        _gate(
            "gradient_coverage",
            gradient_coverage,
            "all active groups nonzero",
            gradient_coverage,
            source_metrics,
            "numerical",
        ),
        _gate(
            "parameter_change",
            None
            if checkpoint_evidence is None
            else checkpoint_evidence["parameter_change_observed_between_first_and_last_checkpoint"],
            "true across published checkpoints",
            checkpoint_evidence is not None
            and bool(checkpoint_evidence["parameter_change_observed_between_first_and_last_checkpoint"]),
            "checkpoints/step-0250.pt + checkpoints/step-1000.pt",
            "diagnostic; not used by legacy classifier",
            update_zero_hash_available=False,
        ),
        _gate(
            "optimizer_updates",
            report.get("optimizer_updates"),
            "1000",
            report.get("optimizer_updates") == 1000,
            source_report,
            "completion",
        ),
        _gate(
            "cuda_peak_allocated_mib",
            allocated,
            f"<= {memory_limits['maximum_cuda_allocated_mib']} MiB",
            allocated <= float(memory_limits["maximum_cuda_allocated_mib"]),
            source_report,
            "memory",
        ),
        _gate(
            "cuda_peak_reserved_mib",
            reserved,
            f"<= {memory_limits['maximum_cuda_reserved_mib']} MiB",
            reserved <= float(memory_limits["maximum_cuda_reserved_mib"]),
            source_report,
            "memory",
        ),
        _gate("backend_policy_active", during, "matches strict policy", backend_matches, source_report, "numerical"),
        _gate(
            "backend_restored",
            backend.get("restored"),
            "true",
            backend.get("restored") is True,
            source_report,
            "numerical",
        ),
        _gate(
            "o3_equivariance",
            equivariance,
            "train and validation true",
            all(equivariance),
            source_evaluations,
            "numerical",
        ),
        _gate(
            "sampling_finite",
            final_finite,
            "all final samples true",
            final_finite,
            source_sampling,
            "legacy strict_checks",
        ),
        _gate(
            "sampling_masking_exact",
            masking,
            "all final samples true",
            masking,
            source_sampling,
            "legacy strict_checks",
        ),
        _gate(
            "sampling_centering",
            final_centering,
            f"<= {centering_limit} Angstrom",
            final_centering <= centering_limit,
            source_sampling,
            "legacy strict_checks",
        ),
        _gate("distance_symmetry", final_symmetry, "<= 1e-8", final_symmetry <= 1e-8, source_sampling, "numerical"),
        _gate(
            "reported_cdist_diagonal_exact_zero",
            final_reported_diagonal,
            "== 0",
            final_reported_diagonal == 0,
            source_sampling,
            "legacy strict_checks",
            defect="raw torch.cdist float32 self-distance residual",
        ),
        _gate(
            "canonical_distance_diagonal_exact_zero",
            canonical_checks["all_canonical_diagonals_exact_zero"],
            "true",
            bool(canonical_checks["all_canonical_diagonals_exact_zero"]),
            "sample NPZ coordinates",
            "corrected strict checks",
        ),
        _gate(
            "negative_gram_eigenvalue_mass",
            final_negative_mass,
            "<= 1e-8",
            final_negative_mass <= 1e-8,
            source_sampling,
            "numerical",
        ),
        _gate(
            "global_denoising_improvement",
            global_improvement,
            f">= {threshold['minimum_validation_v_mse_relative_improvement']}",
            global_improvement >= float(threshold["minimum_validation_v_mse_relative_improvement"]),
            source_evaluations,
            "denoising",
        ),
        _gate(
            "high_noise_improvement",
            high_improvement,
            f">= {threshold['minimum_high_noise_relative_improvement']}",
            high_improvement >= float(threshold["minimum_high_noise_relative_improvement"]),
            source_evaluations,
            "denoising",
        ),
        _gate(
            "very_high_noise_improvement",
            very_high_improvement,
            f">= {threshold['minimum_very_high_noise_relative_improvement']}",
            very_high_improvement >= float(threshold["minimum_very_high_noise_relative_improvement"]),
            source_evaluations,
            "denoising",
        ),
        _gate(
            "maximum_length_stratum_regression",
            max(length_regressions.values()),
            f"<= {threshold['maximum_length_stratum_relative_regression']}",
            max(length_regressions.values()) <= float(threshold["maximum_length_stratum_relative_regression"]),
            source_evaluations,
            "denoising",
            by_length_stratum=length_regressions,
        ),
        _gate(
            "sampling_quality_improvement",
            sampling_improvement,
            f">= {threshold['minimum_sampling_quality_relative_improvement']}",
            sampling_improvement >= float(threshold["minimum_sampling_quality_relative_improvement"]),
            source_sampling,
            "sampling",
        ),
        _gate(
            "sampling_geometry_all_records",
            len(geometry_failures),
            "0 failed final records",
            not geometry_failures,
            source_sampling,
            "sampling",
            bounded_failures=geometry_failures[:20],
        ),
        _gate(
            "protected_inputs_unchanged",
            report.get("protected_inputs_unchanged"),
            "true",
            report.get("protected_inputs_unchanged") is True and protocol.get("protected_inputs_unchanged") is True,
            "report.json + protocol.json",
            "integrity",
        ),
    ]
    corrected_numerical = all(
        gate["passed"]
        for gate in gates
        if gate["gate_name"]
        in {
            "finite_training_losses",
            "finite_gradients",
            "gradient_coverage",
            "optimizer_updates",
            "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib",
            "backend_policy_active",
            "backend_restored",
            "o3_equivariance",
            "sampling_finite",
            "sampling_masking_exact",
            "sampling_centering",
            "distance_symmetry",
            "canonical_distance_diagonal_exact_zero",
            "negative_gram_eigenvalue_mass",
            "protected_inputs_unchanged",
        }
    )
    denoising = all(
        next(gate["passed"] for gate in gates if gate["gate_name"] == name)
        for name in (
            "global_denoising_improvement",
            "high_noise_improvement",
            "very_high_noise_improvement",
            "maximum_length_stratum_regression",
        )
    )
    sampling_learned = (
        next(g["passed"] for g in gates if g["gate_name"] == "sampling_quality_improvement") and not geometry_failures
    )
    corrected = corrected_classification(
        numerical_checks_pass=corrected_numerical,
        denoising_checks_pass=denoising,
        sampling_checks_pass=sampling_learned,
        global_improvement=global_improvement,
        maximum_length_regression=max(length_regressions.values()),
        maximum_allowed_length_regression=float(threshold["maximum_length_stratum_relative_regression"]),
    )
    legacy = classify_pilot(dict(evaluations), dict(samplings), dict(config))
    return {
        "legacy_classification": legacy,
        "published_classification": report.get("classification"),
        "corrected_classification": corrected,
        "classifier_defect_confirmed": legacy == "numerical_or_memory_failure" and corrected != legacy,
        "root_cause": (
            "raw torch.cdist self-distances were required to equal exact zero instead of using the canonical "
            "exact-diagonal constructor"
        ),
        "decision_table": gates,
        "geometry_failure_count": len(geometry_failures),
        "geometry_failures": geometry_failures,
        "strict_checks_scope": (
            "legacy classifier checks replay at updates 0 and 1000, but final-sample strict fields only at update 1000"
        ),
        "memory_unit_audit": (
            "observed and configured values are both MiB; peak values are used during execution and in this audit"
        ),
        "missing_field_policy": "required evidence is refused; optional defaults are not treated as passing",
        "scientific_conclusion": {
            "denoising_learned": denoising,
            "unconditional_sampling_improved": sampling_improvement > 0,
            "unconditional_sampling_verified": sampling_learned,
            "length_specific_denoising_failure": any(value > 0 for value in length_regressions.values()),
            "very_high_noise_relative_improvement": very_high_improvement,
            "very_high_noise_remains_deficient": final["by_timestep_bin"]["very_high_noise"]["coordinate_v_mse"]
            == max(x["coordinate_v_mse"] for x in final["by_timestep_bin"].values()),
            "longer_training_authorized": False,
            "interpretation": (
                "1,000 updates establish denoising and strong sampling improvement, but not the all-sample geometry "
                "gate or evidence needed to authorize longer training."
            ),
        },
    }


def _inventory(directory: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(directory).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def publish_decision_audit(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    if config.get("version") != VERSION:
        raise ValueError("E007 Phase-3F decision-audit configuration version contradiction")
    source = Path(config["source_dir"])
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E007 decision-correction output exists: {output}")
    report_path = source / "report.json"
    protocol_path = source / "protocol.json"
    if sha256_file(report_path) != config["expected_report_sha256"]:
        raise ValueError("E007 Phase-3F source report hash contradiction")
    if sha256_file(protocol_path) != config["expected_protocol_sha256"]:
        raise ValueError("E007 Phase-3F source protocol hash contradiction")
    pilot_config_path = Path(config["pilot_config_path"])
    if sha256_file(pilot_config_path) != config["expected_configuration_sha256"]:
        raise ValueError("E007 Phase-3F pilot configuration hash contradiction")
    source_inventory_before = _inventory(source)
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    metrics = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines() if line.strip()]
    evaluations = {int(key): value for key, value in json.loads((source / "evaluations.json").read_text()).items()}
    samplings = {int(key): value for key, value in json.loads((source / "sampling.json").read_text()).items()}
    pilot_config = yaml.safe_load(pilot_config_path.read_text())
    canonical_checks = _canonical_sample_checks(source, samplings)
    checkpoint_evidence = _checkpoint_parameter_evidence(source)
    audit = decision_audit(
        report,
        protocol,
        metrics,
        evaluations,
        samplings,
        pilot_config,
        canonical_checks,
        checkpoint_evidence,
    )
    pareto = pareto_analysis(evaluations, samplings)
    source_inventory_after = _inventory(source)
    if source_inventory_after != source_inventory_before:
        raise ValueError("E007 Phase-3F source artifacts changed during decision audit")
    staging = output.with_name(f".{output.name}.inprogress")
    if staging.exists():
        raise FileExistsError(f"E007 decision-correction staging exists: {staging}")
    staging.mkdir(parents=True)
    inventory_payload = {
        "source_directory": str(source),
        "source_artifacts": source_inventory_before,
        "aggregate_sha256": _canonical_sha(source_inventory_before),
    }
    _atomic_json(staging / "artifact_inventory.json", inventory_payload)
    _atomic_json(staging / "decision_table.json", {"gates": audit["decision_table"]})
    _atomic_json(staging / "checkpoint_pareto.json", pareto)
    correction_report = {
        "status": "completed_read_only_decision_correction",
        "version": VERSION,
        "source_report_sha256": config["expected_report_sha256"],
        "source_protocol_sha256": config["expected_protocol_sha256"],
        "source_inventory_sha256": inventory_payload["aggregate_sha256"],
        "original_measurements_revised": False,
        "source_artifacts_unchanged": True,
        "canonical_distance_recheck": canonical_checks,
        "checkpoint_parameter_evidence": checkpoint_evidence,
        "decision_audit": audit,
        "checkpoint_pareto": pareto,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "report.json", correction_report)
    correction_protocol = {
        "status": correction_report["status"],
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "source_report_sha256": config["expected_report_sha256"],
        "source_protocol_sha256": config["expected_protocol_sha256"],
        "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
        "decision_table_sha256": sha256_file(staging / "decision_table.json"),
        "checkpoint_pareto_sha256": sha256_file(staging / "checkpoint_pareto.json"),
        "report_sha256": sha256_file(staging / "report.json"),
        "completed_utc": _utc_now(),
        "source_artifacts_unchanged": True,
        **NON_AUTHORIZING,
    }
    _atomic_json(staging / "protocol.json", correction_protocol)
    _atomic_json(
        staging / "heartbeat.json",
        {
            "status": "completed",
            "completed_utc": correction_protocol["completed_utc"],
            "report_sha256": correction_protocol["report_sha256"],
            **NON_AUTHORIZING,
        },
    )
    staging.replace(output)
    if _inventory(source) != source_inventory_before:
        raise ValueError("E007 Phase-3F source artifacts changed after publication")
    return {
        "status": correction_report["status"],
        "output_dir": str(output),
        "corrected_classification": audit["corrected_classification"],
        "source_artifacts_unchanged": True,
        **NON_AUTHORIZING,
    }
