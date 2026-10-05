"""Read-only E007 repair-feasibility analysis for immutable matrix candidates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from protein_distance_diffusion.evaluation.distance_matrix_quality import (
    MatrixQualityConfig,
    assess_distance_matrix,
)
from protein_distance_diffusion.evaluation.e007_matrix_audit import load_candidate_npz
from protein_distance_diffusion.evaluation.repairability import (
    contact_metrics,
    distance_radius_of_gyration,
    pairwise_distances,
    radius_of_gyration,
    upper_pair_mask,
)

PLAN_VERSION = "e007_matrix_repair_audit_plan_v1"
AUDIT_VERSION = "e007_matrix_repair_audit_v1"


@dataclass(frozen=True)
class Rank3Projection:
    """Deterministic rank-three PSD projection and its spectral accounting."""

    coordinates: np.ndarray
    distance_matrix: np.ndarray
    diagnostics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verified_file(root: Path, record: dict[str, Any], label: str) -> Path:
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else root / raw
    if not path.is_file():
        raise FileNotFoundError(f"E007 repair {label} is missing: {path}")
    observed = sha256_file(path)
    if observed != str(record["sha256"]).lower():
        raise ValueError(f"E007 repair {label} SHA-256 contradiction: {path}")
    return path


def build_repair_plan(config_path: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    """Verify all transitive inputs and return a non-mutating repair plan."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("version") != PLAN_VERSION:
        raise ValueError(f"Unsupported E007 repair plan version: {config.get('version')!r}")
    output = root / str(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 repair output or in-progress directory already exists: {output}")

    source = config["source_audit"]
    report_path = _verified_file(
        root, {"path": source["report_path"], "sha256": source["report_sha256"]}, "source report"
    )
    protocol_path = _verified_file(
        root, {"path": source["protocol_path"], "sha256": source["protocol_sha256"]}, "source protocol"
    )
    manifest_path = _verified_file(
        root,
        {"path": source["candidate_manifest_path"], "sha256": source["candidate_manifest_sha256"]},
        "candidate manifest",
    )
    _verified_file(
        root,
        {"path": source["matrix_metrics_path"], "sha256": source["matrix_metrics_sha256"]},
        "matrix metrics",
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if protocol.get("status") != source["required_status"]:
        raise ValueError("E007 source matrix audit is not completed")
    if protocol.get("authorizes_training") or protocol.get("training_performed"):
        raise ValueError("E007 source matrix audit has invalid authorization/training semantics")
    if report.get("counts", {}).get("generated_candidates") != int(source["required_candidate_count"]):
        raise ValueError("E007 source matrix-audit candidate count contradicts the repair contract")
    payload_hashes = protocol.get("published_payload_hashes", {})
    if payload_hashes.get("candidate_manifest.jsonl") != source["candidate_manifest_sha256"]:
        raise ValueError("E007 source protocol candidate-manifest hash contradiction")
    if payload_hashes.get("matrix_metrics.jsonl") != source["matrix_metrics_sha256"]:
        raise ValueError("E007 source protocol matrix-metrics hash contradiction")

    candidates = _read_jsonl(manifest_path)
    if len(candidates) != int(source["required_candidate_count"]):
        raise ValueError("E007 repair candidate manifest has the wrong row count")
    if len(candidates) > int(config["bounds"]["maximum_candidate_count"]):
        raise ValueError("E007 repair candidate count exceeds the configured bound")
    if not bool(config["bounds"]["process_candidates_individually"]):
        raise ValueError("E007 repair requires individual candidate processing")
    source_directory = root / str(source["directory"])
    source_resolved = source_directory.resolve()
    identities: set[str] = set()
    inventory = []
    length_counts: dict[str, int] = {}
    for row in candidates:
        candidate_id = str(row["candidate_id"])
        if candidate_id in identities:
            raise ValueError(f"Duplicate E007 repair candidate identity: {candidate_id}")
        identities.add(candidate_id)
        relative = Path(str(row["candidate_artifact_path"]))
        path = (source_directory / relative).resolve()
        try:
            path.relative_to(source_resolved)
        except ValueError as exc:
            raise ValueError(f"E007 candidate path escapes the source audit: {relative}") from exc
        expected = str(row["candidate_artifact_sha256"])
        if payload_hashes.get(str(relative)) != expected or sha256_file(path) != expected:
            raise ValueError(f"E007 candidate artifact hash contradiction: {candidate_id}")
        loaded = load_candidate_npz(path)
        if loaded["candidate_id"] != candidate_id or loaded["sampling_seed"] != int(row["sampling_seed"]):
            raise ValueError(f"E007 candidate NPZ provenance contradiction: {candidate_id}")
        length = int(row["requested_length"])
        if loaded["requested_length"] != length or loaded["actual_valid_length"] != length:
            raise ValueError(f"E007 candidate length provenance contradiction: {candidate_id}")
        if loaded["physical_matrix_angstrom"].shape != (length, length):
            raise ValueError(f"E007 candidate physical-matrix shape contradiction: {candidate_id}")
        length_counts[str(length)] = length_counts.get(str(length), 0) + 1
        inventory.append(
            {
                "candidate_id": candidate_id,
                "requested_length": length,
                "sampling_seed": int(row["sampling_seed"]),
                "path": str(relative),
                "sha256": expected,
            }
        )
    if sorted(map(int, length_counts)) != sorted(map(int, source["required_lengths"])):
        raise ValueError("E007 repair candidate lengths contradict the configured strata")
    protected = {
        name: {
            "path": str(record["path"]),
            "sha256": sha256_file(_verified_file(root, record, name.replace("_", " "))),
        }
        for name, record in config["protected_inputs"].items()
    }
    return {
        "version": PLAN_VERSION,
        "status": "planned",
        "mode": "plan_only",
        "output_dir": str(config["output_dir"]),
        "output_directory_absent": True,
        "source_audit": {
            "directory": str(source["directory"]),
            "report_sha256": source["report_sha256"],
            "protocol_sha256": source["protocol_sha256"],
            "candidate_manifest_sha256": source["candidate_manifest_sha256"],
            "matrix_metrics_sha256": source["matrix_metrics_sha256"],
        },
        "candidate_count": len(inventory),
        "candidate_counts_by_length": length_counts,
        "candidate_inventory_sha256": _json_hash(inventory),
        "candidates": inventory,
        "repair_methods": ["rank3_psd_projection", "constrained_coordinate_repair"],
        "decision_policy": config["decision_policy"],
        "bounds": config["bounds"],
        "native_or_reference_geometry_used_for_repair": False,
        "independent_candidates_averaged": False,
        "protected_inputs": protected,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_sequence_conditioning": False,
        "real_evaluation_executed": False,
    }


def rank3_psd_projection(matrix: np.ndarray) -> Rank3Projection:
    """Project a complete distance matrix to a deterministic rank-three PSD EDM."""
    original = np.asarray(matrix, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError(f"rank3 projection requires a square matrix, got {original.shape}")
    if not np.isfinite(original).all():
        raise ValueError("rank3 projection requires finite distances")
    asymmetry = np.abs(original - original.T)
    distances = 0.5 * (original + original.T)
    original_diagonal_max = float(np.max(np.abs(np.diag(distances)))) if distances.size else 0.0
    np.fill_diagonal(distances, 0.0)
    n = distances.shape[0]
    centering = np.eye(n) - np.ones((n, n), dtype=np.float64) / max(n, 1)
    gram = -0.5 * centering @ np.square(distances) @ centering
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    nonnegative = np.clip(eigenvalues, 0.0, None)
    coordinates = np.zeros((n, 3), dtype=np.float64)
    retained = min(3, n)
    coordinates[:, :retained] = eigenvectors[:, :retained] * np.sqrt(nonnegative[:retained])[None, :]
    coordinates -= coordinates.mean(axis=0, keepdims=True)
    repaired = pairwise_distances(coordinates)
    upper = upper_pair_mask(n)
    delta = repaired[upper] - original[upper]
    abs_mass = max(float(np.sum(np.abs(eigenvalues))), np.finfo(np.float64).eps)
    positive_mass = max(float(np.sum(nonnegative)), np.finfo(np.float64).eps)
    diagnostics = {
        "original_asymmetry_max_angstrom": float(np.max(asymmetry)) if asymmetry.size else 0.0,
        "original_diagonal_max_angstrom": original_diagonal_max,
        "negative_eigenvalue_count": int(np.sum(eigenvalues < 0.0)),
        "discarded_negative_eigenmass": float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0]))),
        "discarded_negative_eigenmass_fraction": float(np.sum(np.abs(eigenvalues[eigenvalues < 0.0])) / abs_mass),
        "discarded_positive_mass_beyond_rank3": float(np.sum(nonnegative[3:])),
        "discarded_positive_mass_beyond_rank3_fraction": float(np.sum(nonnegative[3:]) / positive_mass),
        "reconstruction_rmse_angstrom": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "reconstruction_mae_angstrom": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "coordinates_are_mathematical_reconstruction": True,
        "chirality_claimed": False,
    }
    return Rank3Projection(coordinates, repaired, diagnostics)


def _pair_categories(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = raw.shape[0]
    i, j = np.triu_indices(n, k=1)
    separation = j - i
    categories = {
        "adjacent": separation == 1,
        "local_sequence_separation_2_to_4": (separation >= 2) & (separation <= 4),
        "raw_contact_at_or_below_10_angstrom": (separation > 4) & (raw[i, j] <= 10.0),
    }
    assigned = np.logical_or.reduce(list(categories.values()))
    categories["other_pairs"] = ~assigned
    return i, j, categories


def _loss_and_gradient(
    coordinates: np.ndarray,
    raw: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    settings = config["constrained_coordinate_repair"]
    i, j, categories = _pair_categories(raw)
    difference = coordinates[i] - coordinates[j]
    distances = np.linalg.norm(difference, axis=1)
    safe = np.maximum(distances, 1e-8)
    residual = distances - raw[i, j]
    delta = float(settings["huber_delta_angstrom"])
    abs_residual = np.abs(residual)
    huber = np.where(abs_residual <= delta, 0.5 * residual**2, delta * (abs_residual - 0.5 * delta))
    huber_derivative = np.where(abs_residual <= delta, residual, delta * np.sign(residual))
    pair_weights = settings["pair_weighting"]
    active_weight_sum = sum(float(pair_weights[name]) for name, mask in categories.items() if np.any(mask))
    if active_weight_sum <= 0.0:
        raise ValueError("E007 repair requires a positive weight for at least one populated pair stratum")
    category_weight_sum = active_weight_sum if bool(pair_weights["normalize_to_mean_one"]) else 1.0
    fit_loss = 0.0
    fit_derivative = np.zeros_like(residual)
    category_losses: dict[str, float] = {}
    for name, mask in categories.items():
        if not np.any(mask):
            category_losses[name] = 0.0
            continue
        normalized_weight = float(pair_weights[name]) / category_weight_sum
        category_losses[name] = float(np.mean(huber[mask]))
        fit_loss += normalized_weight * category_losses[name]
        fit_derivative[mask] = normalized_weight * huber_derivative[mask] / int(np.sum(mask))
    fit_weight = float(settings["distance_fit_weight"])
    gradient = np.zeros_like(coordinates)
    pair_gradient = (fit_weight * fit_derivative / safe)[:, None] * difference
    np.add.at(gradient, i, pair_gradient)
    np.add.at(gradient, j, -pair_gradient)

    adjacent_difference = coordinates[1:] - coordinates[:-1]
    adjacent_distance = np.linalg.norm(adjacent_difference, axis=1)
    adjacent_residual = adjacent_distance - float(settings["adjacent_target_angstrom"])
    adjacent_loss = float(np.mean(adjacent_residual**2)) if adjacent_residual.size else 0.0
    if adjacent_residual.size:
        derivative = 2.0 * adjacent_residual / adjacent_residual.size
        adjacent_gradient = (derivative / np.maximum(adjacent_distance, 1e-8))[:, None] * adjacent_difference
        adjacent_gradient *= float(settings["adjacent_penalty_weight"])
        gradient[:-1] -= adjacent_gradient
        gradient[1:] += adjacent_gradient

    non_neighbour = (j - i) >= 2
    clash_violation = np.maximum(float(settings["clash_threshold_angstrom"]) - distances[non_neighbour], 0.0)
    clash_loss = float(np.mean(clash_violation**2)) if clash_violation.size else 0.0
    active_indices = np.flatnonzero(non_neighbour)[clash_violation > 0.0]
    if active_indices.size:
        derivative = -2.0 * clash_violation[clash_violation > 0.0] / max(clash_violation.size, 1)
        clash_gradient = (derivative / safe[active_indices])[:, None] * difference[active_indices]
        clash_gradient *= float(settings["clash_penalty_weight"])
        np.add.at(gradient, i[active_indices], clash_gradient)
        np.add.at(gradient, j[active_indices], -clash_gradient)

    centroid = coordinates.mean(axis=0)
    centering_loss = float(np.mean(centroid**2))
    centering_weight = float(settings["centering_regularization_weight"])
    gradient += centering_weight * 2.0 * centroid[None, :] / (3.0 * coordinates.shape[0])
    total = (
        fit_weight * fit_loss
        + float(settings["adjacent_penalty_weight"]) * adjacent_loss
        + float(settings["clash_penalty_weight"]) * clash_loss
        + centering_weight * centering_loss
    )
    return {
        "total": float(total),
        "robust_distance_fit": float(fit_loss),
        "adjacent_bond": adjacent_loss,
        "steric_clash": clash_loss,
        "centering": centering_loss,
        "fit_category_losses": category_losses,
    }, gradient


def constrained_coordinate_repair(
    raw_matrix: np.ndarray,
    initial_coordinates: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Run deterministic manual-Adam coordinate refinement with analytic gradients."""
    raw = np.asarray(raw_matrix, dtype=np.float64)
    coordinates = np.asarray(initial_coordinates, dtype=np.float64).copy()
    if raw.shape != (coordinates.shape[0], coordinates.shape[0]) or coordinates.shape[1] != 3:
        raise ValueError("E007 constrained repair input shape contradiction")
    settings = config["constrained_coordinate_repair"]
    initial_losses, initial_gradient = _loss_and_gradient(coordinates, raw, config)
    moment = np.zeros_like(coordinates)
    variance = np.zeros_like(coordinates)
    best_loss = initial_losses["total"]
    stale = 0
    started = time.monotonic()
    termination = "maximum_iterations"
    iterations = 0
    for step in range(1, int(settings["maximum_iterations"]) + 1):
        losses, gradient = _loss_and_gradient(coordinates, raw, config)
        gradient_norm = float(np.linalg.norm(gradient))
        if not math.isfinite(losses["total"]) or not np.isfinite(gradient).all():
            raise FloatingPointError("E007 constrained coordinate repair became non-finite")
        if gradient_norm <= float(settings["gradient_norm_tolerance"]):
            termination = "gradient_norm_tolerance"
            iterations = step - 1
            break
        clip = float(settings["gradient_clip_norm"])
        if gradient_norm > clip:
            gradient *= clip / gradient_norm
        beta1, beta2 = float(settings["adam_beta1"]), float(settings["adam_beta2"])
        moment = beta1 * moment + (1.0 - beta1) * gradient
        variance = beta2 * variance + (1.0 - beta2) * np.square(gradient)
        corrected_moment = moment / (1.0 - beta1**step)
        corrected_variance = variance / (1.0 - beta2**step)
        coordinates -= (
            float(settings["learning_rate"])
            * corrected_moment
            / (np.sqrt(corrected_variance) + float(settings["adam_epsilon"]))
        )
        coordinates -= coordinates.mean(axis=0, keepdims=True)
        iterations = step
        relative_improvement = (best_loss - losses["total"]) / max(abs(best_loss), 1e-12)
        if relative_improvement > float(settings["convergence_relative_tolerance"]):
            best_loss, stale = losses["total"], 0
        else:
            stale += 1
        if stale >= int(settings["convergence_patience"]):
            termination = "relative_improvement_patience"
            break
        if time.monotonic() - started > float(settings["maximum_seconds_per_candidate"]):
            termination = "runtime_limit"
            break
    final_losses, final_gradient = _loss_and_gradient(coordinates, raw, config)
    stored_coordinates = coordinates.astype(np.float32)
    matrix = pairwise_distances(stored_coordinates).astype(np.float32)
    return (
        stored_coordinates,
        matrix,
        {
            "algorithm": "manual_adam_analytic_coordinate_gradient_v1",
            "deterministic": True,
            "iterations": iterations,
            "converged": termination not in {"maximum_iterations", "runtime_limit"},
            "termination_reason": termination,
            "elapsed_seconds": float(time.monotonic() - started),
            "initial_component_losses": initial_losses,
            "final_component_losses": final_losses,
            "initial_gradient_norm": float(np.linalg.norm(initial_gradient)),
            "final_gradient_norm": float(np.linalg.norm(final_gradient)),
            "model_optimizer_created": False,
            "model_optimizer_updates": 0,
            "native_or_reference_geometry_used": False,
            "chirality_claimed": False,
        },
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 1.0
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def topology_preservation_metrics(
    raw_matrix: np.ndarray,
    repaired_matrix: np.ndarray,
    *,
    repaired_coordinates: np.ndarray | None,
    contact_thresholds: list[float],
    neighbourhood_size: int,
    clash_threshold: float,
    adjacent_target: float,
    adjacent_tolerance: float,
) -> dict[str, Any]:
    """Measure topology and local-chain preservation against the same raw candidate."""
    raw, repaired = np.asarray(raw_matrix, dtype=np.float64), np.asarray(repaired_matrix, dtype=np.float64)
    n = raw.shape[0]
    mask = upper_pair_mask(n)
    raw_values, repaired_values = raw[mask], repaired[mask]
    delta = repaired_values - raw_values
    contacts: dict[str, dict[str, float]] = {}
    for threshold in contact_thresholds:
        contacts[f"{threshold:g}"] = contact_metrics(raw, repaired, threshold=threshold, mask=mask)
    k = min(int(neighbourhood_size), max(n - 1, 0))
    retention = []
    for index in range(n):
        raw_order = np.argsort(raw[index], kind="mergesort")
        repaired_order = np.argsort(repaired[index], kind="mergesort")
        raw_neighbours = [value for value in raw_order if value != index][:k]
        repaired_neighbours = [value for value in repaired_order if value != index][:k]
        retention.append(len(set(raw_neighbours) & set(repaired_neighbours)) / max(k, 1))
    adjacent = np.diag(repaired, k=1)
    non_neighbour_mask = upper_pair_mask(n, min_separation=2)
    clashes = repaired[non_neighbour_mask] < float(clash_threshold)
    raw_radius = distance_radius_of_gyration(raw)
    coordinate_radius = radius_of_gyration(repaired_coordinates) if repaired_coordinates is not None else None
    return {
        "pairwise_distance_rmse_angstrom": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "pairwise_distance_mae_angstrom": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "relative_frobenius_distortion": float(np.linalg.norm(delta) / max(np.linalg.norm(raw_values), 1e-12)),
        "distance_spearman_correlation": _spearman(raw_values, repaired_values),
        "contact_metrics": contacts,
        "neighbourhood_size": k,
        "neighbourhood_retention_mean": float(np.mean(retention)) if retention else 1.0,
        "neighbourhood_retention_median": float(np.median(retention)) if retention else 1.0,
        "neighbourhood_retention_minimum": float(np.min(retention)) if retention else 1.0,
        "adjacent_distance_mean_angstrom": float(np.mean(adjacent)) if adjacent.size else None,
        "adjacent_distance_rmse_angstrom": (
            float(np.sqrt(np.mean(np.square(adjacent - adjacent_target)))) if adjacent.size else None
        ),
        "adjacent_distance_plausible_fraction": (
            float(np.mean(np.abs(adjacent - adjacent_target) <= adjacent_tolerance)) if adjacent.size else None
        ),
        "non_neighbour_clash_count": int(np.sum(clashes)),
        "non_neighbour_pair_count": int(clashes.size),
        "non_neighbour_clash_fraction": float(np.mean(clashes)) if clashes.size else 0.0,
        "raw_distance_inferred_radius_of_gyration": raw_radius,
        "repaired_coordinate_radius_of_gyration": coordinate_radius,
        "radius_of_gyration_change": coordinate_radius - raw_radius if coordinate_radius is not None else None,
    }


def _quality(matrix: np.ndarray, candidate_id: str, config: dict[str, Any]) -> dict[str, Any]:
    n = matrix.shape[0]
    return assess_distance_matrix(
        matrix,
        pair_mask=np.ones((n, n), dtype=bool),
        sequence_length=n,
        candidate_id=candidate_id,
        config=MatrixQualityConfig(
            triangle_exact_max_length=64,
            triangle_sample_count=4096,
            triangle_seed=7307,
            eigen_exact_max_length=500,
        ),
    )


def _get_path(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


AGGREGATE_METRICS = (
    "quality.valid_residue_count",
    "quality.valid_pair_count",
    "quality.valid_matrix_cell_count",
    "quality.symmetry_error_max_angstrom",
    "quality.diagonal_error_max_angstrom",
    "quality.negative_distance_count",
    "quality.triangle_violation_fraction",
    "quality.triangle_violation_max_angstrom",
    "quality.negative_eigenmass_fraction",
    "quality.rank3_residual_energy_fraction",
    "quality.rank3_reconstruction_rmse_angstrom",
    "quality.adjacent_residue_distance_mean_angstrom",
    "quality.adjacent_residue_distance_rmse_angstrom",
    "quality.adjacent_residue_plausible_fraction",
    "quality.contact_density.6",
    "quality.contact_density.8",
    "quality.contact_density.10",
    "topology.pairwise_distance_rmse_angstrom",
    "topology.pairwise_distance_mae_angstrom",
    "topology.relative_frobenius_distortion",
    "topology.distance_spearman_correlation",
    "topology.contact_metrics.6.precision",
    "topology.contact_metrics.6.recall",
    "topology.contact_metrics.6.f1",
    "topology.contact_metrics.6.jaccard",
    "topology.contact_metrics.8.precision",
    "topology.contact_metrics.8.recall",
    "topology.contact_metrics.8.f1",
    "topology.contact_metrics.8.jaccard",
    "topology.contact_metrics.10.precision",
    "topology.contact_metrics.10.recall",
    "topology.contact_metrics.10.f1",
    "topology.contact_metrics.10.jaccard",
    "topology.neighbourhood_retention_mean",
    "topology.neighbourhood_retention_median",
    "topology.neighbourhood_retention_minimum",
    "topology.adjacent_distance_mean_angstrom",
    "topology.adjacent_distance_rmse_angstrom",
    "topology.adjacent_distance_plausible_fraction",
    "topology.non_neighbour_clash_count",
    "topology.non_neighbour_clash_fraction",
    "topology.raw_distance_inferred_radius_of_gyration",
    "topology.repaired_coordinate_radius_of_gyration",
    "topology.radius_of_gyration_change",
)


def _bootstrap(values: np.ndarray, iterations: int, seed: int) -> list[float] | None:
    if values.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=values.size, replace=True))) for _ in range(iterations)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_repair_metrics(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate candidate versions and immutable real-reference context by length."""
    output = []
    versions = sorted({str(row["version"]) for row in rows})
    lengths: list[int | None] = [None, *sorted({int(row["requested_length"]) for row in rows})]
    for version in versions:
        for length in lengths:
            selected = [
                row
                for row in rows
                if row["version"] == version and (length is None or int(row["requested_length"]) == length)
            ]
            for metric in AGGREGATE_METRICS:
                values = np.asarray(
                    [float(value) for row in selected if (value := _get_path(row, metric)) is not None],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                output.append(
                    {
                        "version": version,
                        "requested_length": length,
                        "metric": metric,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "standard_deviation": float(np.std(values)),
                        "quantile_05": float(np.quantile(values, 0.05)),
                        "quantile_25": float(np.quantile(values, 0.25)),
                        "quantile_75": float(np.quantile(values, 0.75)),
                        "quantile_95": float(np.quantile(values, 0.95)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                        "mean_bootstrap_ci_95": _bootstrap(
                            values,
                            int(config["aggregation"]["bootstrap_iterations"]),
                            int(config["aggregation"]["bootstrap_seed"]) + len(output),
                        ),
                    }
                )
    return output


def _passes(row: dict[str, Any], config: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    hard, topology = (
        config["decision_policy"]["hard_physical_requirements"],
        config["decision_policy"]["topology_preservation_requirements"],
    )
    q, t = row["quality"], row["topology"]
    failures = []
    checks = {
        "finite": bool(q["finite"]) if hard["require_finite"] else True,
        "symmetric": q["symmetry_error_max_angstrom"] <= hard["maximum_symmetry_error_angstrom"],
        "zero_diagonal": q["diagonal_error_max_angstrom"] <= hard["maximum_diagonal_error_angstrom"],
        "nonnegative": q["negative_distance_count"] <= hard["maximum_negative_distance_count"],
        "triangle": q["triangle_violation_fraction"] <= hard["maximum_triangle_violation_fraction"],
        "edm": q["negative_eigenmass_fraction"] <= hard["maximum_negative_eigenmass_fraction"],
        "rank3": q["rank3_residual_energy_fraction"] <= hard["maximum_rank3_residual_energy_fraction"],
        "adjacent_rmse": t["adjacent_distance_rmse_angstrom"] <= hard["maximum_adjacent_rmse_angstrom"],
        "adjacent_plausibility": (
            t["adjacent_distance_plausible_fraction"] >= hard["minimum_adjacent_plausible_fraction"]
        ),
        "clashes": t["non_neighbour_clash_fraction"] <= hard["maximum_non_neighbour_clash_fraction"],
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    topology_checks = {
        "pairwise_rmse": t["pairwise_distance_rmse_angstrom"] <= topology["maximum_pairwise_rmse_angstrom"],
        "relative_distortion": (
            t["relative_frobenius_distortion"] <= topology["maximum_relative_frobenius_distortion"]
        ),
        "spearman": t["distance_spearman_correlation"] >= topology["minimum_distance_spearman_correlation"],
        "contact_f1_8A": t["contact_metrics"]["8"]["f1"] >= topology["minimum_contact_f1_at_8_angstrom"],
        "neighbourhood_retention": (
            t["neighbourhood_retention_mean"] >= topology["minimum_neighbourhood_retention_mean"]
        ),
    }
    failures.extend(name for name, passed in topology_checks.items() if not passed)
    return all(checks.values()), all(topology_checks.values()), failures


def classify_repair(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared decision table without tuning against outcomes."""
    minimum = float(config["decision_policy"]["minimum_candidate_pass_fraction_per_length"])
    fractions: dict[str, dict[str, dict[str, float]]] = {}
    for version in ("rank3_psd_projection", "constrained_coordinate_repair"):
        fractions[version] = {}
        for length in sorted({int(row["requested_length"]) for row in rows if row["version"] == version}):
            selected = [row for row in rows if row["version"] == version and int(row["requested_length"]) == length]
            physical = np.mean([bool(row["physical_requirements_passed"]) for row in selected])
            topology = np.mean([bool(row["topology_requirements_passed"]) for row in selected])
            joint = np.mean(
                [bool(row["physical_requirements_passed"] and row["topology_requirements_passed"]) for row in selected]
            )
            fractions[version][str(length)] = {
                "physical_pass_fraction": float(physical),
                "topology_pass_fraction": float(topology),
                "joint_pass_fraction": float(joint),
            }
    rank_all = all(value["joint_pass_fraction"] >= minimum for value in fractions["rank3_psd_projection"].values())
    constrained_all = all(
        value["joint_pass_fraction"] >= minimum for value in fractions["constrained_coordinate_repair"].values()
    )
    any_joint = any(
        value["joint_pass_fraction"] >= minimum for method in fractions.values() for value in method.values()
    )
    any_physical = any(
        value["physical_pass_fraction"] >= minimum for method in fractions.values() for value in method.values()
    )
    if constrained_all:
        classification = "repair_preserves_topology_and_restores_geometry"
    elif rank_all:
        classification = "rank3_projection_only_is_acceptable"
    elif any_joint:
        classification = "repair_is_length_limited"
    elif any_physical:
        classification = "repair_destroys_learned_topology"
    else:
        classification = "inconclusive_requires_scientific_review"
    return {"classification": classification, "minimum_pass_fraction": minimum, "pass_fractions": fractions}


def _memory_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_repair_audit(
    config_path: str | Path,
    *,
    plan: dict[str, Any],
    repository_root: str | Path = ".",
) -> Path:
    """Evaluate immutable candidates and atomically publish repair artifacts."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = root / str(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 repair output or in-progress directory already exists: {output}")
    protected_before = {name: sha256_file(root / record["path"]) for name, record in plan["protected_inputs"].items()}
    source_base = root / str(config["source_audit"]["directory"])
    candidate_hashes_before = {
        record["candidate_id"]: sha256_file(source_base / record["path"]) for record in plan["candidates"]
    }
    source_report = json.loads((root / config["source_audit"]["report_path"]).read_text(encoding="utf-8"))
    source_metrics = _read_jsonl(root / config["source_audit"]["matrix_metrics_path"])
    real_rows = [
        {
            "candidate_id": row["candidate_id"],
            "requested_length": row["requested_length"],
            "version": "real_validation_reference",
            "quality": row["assessment"],
            "topology": {},
        }
        for row in source_metrics
        if row["panel"] == "real_validation"
    ]
    started = _utc_now()
    staging.mkdir(parents=True)
    _atomic_json(staging / "heartbeat.json", {"status": "running", "stage": "repair", "started_utc": started})
    rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    try:
        for index, record in enumerate(plan["candidates"]):
            source_path = source_base / record["path"]
            if sha256_file(source_path) != record["sha256"]:
                raise ValueError(f"E007 candidate changed before repair: {record['candidate_id']}")
            loaded = load_candidate_npz(source_path)
            raw = loaded["physical_matrix_angstrom"]
            expected_length = int(record["requested_length"])
            if raw.shape != (expected_length, expected_length):
                raise ValueError(f"E007 candidate matrix shape contradiction: {record['candidate_id']}")
            projection = rank3_psd_projection(raw)
            repaired_coordinates, repaired_matrix, solver = constrained_coordinate_repair(
                raw, projection.coordinates, config
            )
            projection_coordinates = projection.coordinates.astype(np.float32)
            projection_matrix = pairwise_distances(projection_coordinates).astype(np.float32)
            versions = (
                ("raw_generated", raw, None, {}),
                ("rank3_psd_projection", projection_matrix, projection_coordinates, projection.diagnostics),
                ("constrained_coordinate_repair", repaired_matrix, repaired_coordinates, solver),
            )
            for version, matrix, coordinates, method_diagnostics in versions:
                topology = topology_preservation_metrics(
                    raw,
                    matrix,
                    repaired_coordinates=coordinates,
                    contact_thresholds=config["topology_evaluation"]["contact_thresholds_angstrom"],
                    neighbourhood_size=int(config["topology_evaluation"]["neighbourhood_size"]),
                    clash_threshold=float(config["constrained_coordinate_repair"]["clash_threshold_angstrom"]),
                    adjacent_target=float(config["constrained_coordinate_repair"]["adjacent_target_angstrom"]),
                    adjacent_tolerance=float(
                        config["constrained_coordinate_repair"]["adjacent_plausibility_tolerance_angstrom"]
                    ),
                )
                row = {
                    "candidate_id": record["candidate_id"],
                    "requested_length": int(record["requested_length"]),
                    "version": version,
                    "source_candidate_path": record["path"],
                    "source_candidate_sha256": record["sha256"],
                    "quality": _quality(matrix, f"{record['candidate_id']}::{version}", config),
                    "topology": topology,
                    "method_diagnostics": method_diagnostics,
                }
                if version != "raw_generated":
                    physical, topology_pass, failures = _passes(row, config)
                    row["physical_requirements_passed"] = physical
                    row["topology_requirements_passed"] = topology_pass
                    row["decision_failures"] = failures
                rows.append(row)
            relative = Path("repaired") / f"{record['candidate_id']}.npz"
            metadata = {
                "candidate_id": record["candidate_id"],
                "source_candidate_path": record["path"],
                "source_candidate_sha256": record["sha256"],
                "rank3_method": "rank3_psd_projection",
                "constrained_method": "manual_adam_analytic_coordinate_gradient_v1",
                "native_or_reference_geometry_used": False,
                "chirality_claimed": False,
            }
            _atomic_npz(
                staging / relative,
                candidate_id=np.asarray(record["candidate_id"]),
                source_candidate_path=np.asarray(record["path"]),
                source_candidate_sha256=np.asarray(record["sha256"]),
                rank3_coordinates=projection_coordinates,
                rank3_distance_matrix_angstrom=projection_matrix,
                constrained_coordinates=repaired_coordinates,
                constrained_distance_matrix_angstrom=repaired_matrix,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            manifest.append(
                metadata
                | {
                    "requested_length": int(record["requested_length"]),
                    "repaired_artifact_path": str(relative),
                    "repaired_artifact_sha256": sha256_file(staging / relative),
                }
            )
            if _memory_mib() > float(config["bounds"]["maximum_rss_mib"]):
                raise MemoryError(
                    f"E007 repair RSS limit exceeded at {record['candidate_id']}: {_memory_mib():.1f} MiB"
                )
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "running",
                    "stage": "repair",
                    "started_utc": started,
                    "processed_candidates": index + 1,
                    "total_candidates": len(plan["candidates"]),
                    "last_candidate_id": record["candidate_id"],
                    "peak_rss_mib": _memory_mib(),
                },
            )
        decision = classify_repair(rows, config)
        aggregates = aggregate_repair_metrics([*rows, *real_rows], config)
        _atomic_jsonl(staging / "repair_manifest.jsonl", manifest)
        _atomic_jsonl(staging / "matrix_metrics.jsonl", rows)
        protected_after = {
            name: sha256_file(root / record["path"]) for name, record in plan["protected_inputs"].items()
        }
        if protected_before != protected_after:
            raise RuntimeError("E007 repair protected inputs changed during execution")
        candidate_hashes_after = {
            record["candidate_id"]: sha256_file(source_base / record["path"]) for record in plan["candidates"]
        }
        if candidate_hashes_before != candidate_hashes_after:
            raise RuntimeError("E007 repair source candidates changed during execution")
        payload_hashes = {
            str(path.relative_to(staging)): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "heartbeat.json"
        }
        report = {
            "version": AUDIT_VERSION,
            "status": "completed_requires_scientific_review",
            "decision": decision,
            "source_audit": plan["source_audit"],
            "source_audit_scientific_context": {
                "classification": source_report["scientific_review_classification"],
                "recommendation": source_report["recommendation"],
            },
            "configuration_path": str(config_file.relative_to(root)),
            "configuration_sha256": sha256_file(config_file),
            "candidate_count": len(plan["candidates"]),
            "candidate_counts_by_length": plan["candidate_counts_by_length"],
            "repair_methods": plan["repair_methods"],
            "decision_policy": config["decision_policy"],
            "aggregates": aggregates,
            "real_reference_context_count": len(real_rows),
            "native_or_reference_geometry_used_for_repair": False,
            "independent_candidates_averaged": False,
            "chirality_claimed": False,
            "protected_inputs_before": protected_before,
            "protected_inputs_after": protected_after,
            "protected_inputs_unchanged": True,
            "source_candidate_inventory_sha256_before": _json_hash(candidate_hashes_before),
            "source_candidate_inventory_sha256_after": _json_hash(candidate_hashes_after),
            "source_candidates_unchanged": True,
            "published_payload_hashes": payload_hashes,
            "peak_rss_mib": _memory_mib(),
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "backward_performed": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
        }
        _atomic_json(staging / "report.json", report)
        completed = _utc_now()
        protocol = {
            "version": AUDIT_VERSION,
            "status": "completed",
            "started_utc": started,
            "completed_utc": completed,
            "git_commit": _git_commit(root),
            "configuration_sha256": sha256_file(config_file),
            "source_report_sha256": config["source_audit"]["report_sha256"],
            "source_protocol_sha256": config["source_audit"]["protocol_sha256"],
            "report_sha256": sha256_file(staging / "report.json"),
            "training_performed": False,
            "optimizer_created": False,
            "optimizer_updates": 0,
            "backward_performed": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": completed,
                "processed_candidates": len(plan["candidates"]),
                "report_sha256": sha256_file(staging / "report.json"),
                "protocol_sha256": sha256_file(staging / "protocol.json"),
            },
        )
        staging.replace(output)
        return output
    except BaseException as exc:
        if staging.exists():
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "status": "failed",
                    "failed_utc": _utc_now(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "completed_output_published": False,
                },
            )
        raise
