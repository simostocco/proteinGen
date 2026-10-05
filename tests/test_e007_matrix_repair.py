"""Synthetic and provenance tests for the E007 Phase-2B repair audit."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from protein_distance_diffusion.evaluation.e007_matrix_repair import (
    PLAN_VERSION,
    _loss_and_gradient,
    build_repair_plan,
    constrained_coordinate_repair,
    rank3_psd_projection,
    sha256_file,
    topology_preservation_metrics,
)
from protein_distance_diffusion.evaluation.repairability import pairwise_distances


def _repair_config(*, iterations: int = 160) -> dict:
    return {
        "constrained_coordinate_repair": {
            "maximum_iterations": iterations,
            "maximum_seconds_per_candidate": 10,
            "learning_rate": 0.02,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "gradient_clip_norm": 100.0,
            "convergence_relative_tolerance": 1e-9,
            "convergence_patience": iterations + 1,
            "gradient_norm_tolerance": 1e-10,
            "huber_delta_angstrom": 1.0,
            "distance_fit_weight": 1.0,
            "adjacent_penalty_weight": 6.0,
            "clash_penalty_weight": 4.0,
            "centering_regularization_weight": 1e-6,
            "adjacent_target_angstrom": 3.8,
            "adjacent_plausibility_tolerance_angstrom": 0.5,
            "clash_threshold_angstrom": 3.0,
            "pair_weighting": {
                "adjacent": 4.0,
                "local_sequence_separation_2_to_4": 2.0,
                "raw_contact_at_or_below_10_angstrom": 2.0,
                "other_pairs": 1.0,
                "normalize_to_mean_one": True,
            },
        }
    }


def _helix_coordinates(length: int) -> np.ndarray:
    index = np.arange(length, dtype=np.float64)
    return np.stack([1.8 * np.cos(index), 1.8 * np.sin(index), 3.3 * index], axis=1)


def test_rank3_projection_preserves_exact_euclidean_matrix() -> None:
    matrix = pairwise_distances(_helix_coordinates(12))
    result = rank3_psd_projection(matrix)
    assert np.allclose(result.distance_matrix, matrix, atol=1e-6)
    assert result.diagnostics["reconstruction_rmse_angstrom"] < 1e-6
    assert result.coordinates.shape == (12, 3)


def test_rank3_projection_repairs_negative_gram_spectrum() -> None:
    matrix = pairwise_distances(_helix_coordinates(8))
    matrix[0, 2] = matrix[2, 0] = matrix[0, 1] + matrix[1, 2] + 4.0
    result = rank3_psd_projection(matrix)
    assert result.diagnostics["discarded_negative_eigenmass"] > 0.0
    reprojection = rank3_psd_projection(result.distance_matrix)
    assert reprojection.diagnostics["reconstruction_rmse_angstrom"] < 1e-6


def test_constrained_repair_improves_broken_adjacent_geometry() -> None:
    raw = pairwise_distances(_helix_coordinates(10))
    for index in range(9):
        raw[index, index + 1] = raw[index + 1, index] = 7.0
    projected = rank3_psd_projection(raw)
    initial = np.sqrt(np.mean((np.diag(projected.distance_matrix, 1) - 3.8) ** 2))
    _, repaired, diagnostics = constrained_coordinate_repair(raw, projected.coordinates, _repair_config())
    final = np.sqrt(np.mean((np.diag(repaired, 1) - 3.8) ** 2))
    assert final < initial
    assert diagnostics["final_component_losses"]["total"] < diagnostics["initial_component_losses"]["total"]


def test_constrained_repair_reduces_severe_non_neighbour_clash_penalty() -> None:
    raw = pairwise_distances(_helix_coordinates(9))
    initial = _helix_coordinates(9)
    initial[6] = initial[1] + np.array([0.05, 0.0, 0.0])
    initial_loss, _ = _loss_and_gradient(initial, raw, _repair_config())
    _, _, diagnostics = constrained_coordinate_repair(raw, initial, _repair_config())
    assert diagnostics["final_component_losses"]["steric_clash"] < initial_loss["steric_clash"]


def _topology(raw: np.ndarray, repaired: np.ndarray, coords: np.ndarray | None) -> dict:
    return topology_preservation_metrics(
        raw,
        repaired,
        repaired_coordinates=coords,
        contact_thresholds=[6.0, 8.0, 10.0],
        neighbourhood_size=3,
        clash_threshold=3.0,
        adjacent_target=3.8,
        adjacent_tolerance=0.5,
    )


def test_topology_metrics_detect_residue_permutation_and_contact_changes() -> None:
    coordinates = _helix_coordinates(8)
    raw = pairwise_distances(coordinates)
    permutation = np.array([0, 3, 2, 1, 4, 7, 6, 5])
    permuted = raw[np.ix_(permutation, permutation)]
    unchanged = _topology(raw, raw, coordinates)
    changed = _topology(raw, permuted, coordinates[permutation])
    assert unchanged["distance_spearman_correlation"] == pytest.approx(1.0)
    assert unchanged["contact_metrics"]["8"]["precision"] == pytest.approx(1.0)
    assert unchanged["contact_metrics"]["8"]["recall"] == pytest.approx(1.0)
    assert unchanged["contact_metrics"]["8"]["jaccard"] == pytest.approx(1.0)
    assert changed["pairwise_distance_rmse_angstrom"] > 0.0
    assert changed["neighbourhood_retention_mean"] < 1.0


@pytest.mark.parametrize("length", [6, 11, 17])
def test_repair_is_deterministic_at_multiple_lengths(length: int) -> None:
    raw = pairwise_distances(_helix_coordinates(length))
    raw[0, -1] += 1.0
    raw[-1, 0] = raw[0, -1]
    projected = rank3_psd_projection(raw)
    first = constrained_coordinate_repair(raw, projected.coordinates, _repair_config(iterations=20))
    second = constrained_coordinate_repair(raw, projected.coordinates, _repair_config(iterations=20))
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2]["termination_reason"] == second[2]["termination_reason"]


def _write_plan_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source_audit"
    candidates = source / "candidates"
    candidates.mkdir(parents=True)
    candidate_id = "candidate_000"
    length = 6
    matrix = pairwise_distances(_helix_coordinates(length)).astype(np.float32)
    candidate_path = candidates / f"{candidate_id}.npz"
    metadata = {"candidate_id": candidate_id}
    with candidate_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_id=np.asarray(candidate_id),
            requested_length=np.asarray(length),
            actual_valid_length=np.asarray(length),
            sampling_seed=np.asarray(17),
            normalized_matrix=matrix / 53.775,
            physical_matrix_angstrom=matrix,
            pair_mask=np.ones((length, length), dtype=bool),
            metadata=np.asarray(json.dumps(metadata)),
        )
    relative = f"candidates/{candidate_id}.npz"
    candidate_hash = sha256_file(candidate_path)
    manifest = source / "candidate_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "candidate_artifact_path": relative,
                "candidate_artifact_sha256": candidate_hash,
                "requested_length": length,
                "sampling_seed": 17,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = source / "matrix_metrics.jsonl"
    metrics.write_text("", encoding="utf-8")
    report = source / "report.json"
    report.write_text(json.dumps({"counts": {"generated_candidates": 1}}), encoding="utf-8")
    protocol = source / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "status": "completed",
                "authorizes_training": False,
                "training_performed": False,
                "published_payload_hashes": {
                    "candidate_manifest.jsonl": sha256_file(manifest),
                    "matrix_metrics.jsonl": sha256_file(metrics),
                    relative: candidate_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    protected = tmp_path / "checkpoint.pt"
    protected.write_bytes(b"immutable")
    config = {
        "version": PLAN_VERSION,
        "output_dir": "repair_output",
        "source_audit": {
            "directory": "source_audit",
            "report_path": "source_audit/report.json",
            "report_sha256": sha256_file(report),
            "protocol_path": "source_audit/protocol.json",
            "protocol_sha256": sha256_file(protocol),
            "candidate_manifest_path": "source_audit/candidate_manifest.jsonl",
            "candidate_manifest_sha256": sha256_file(manifest),
            "matrix_metrics_path": "source_audit/matrix_metrics.jsonl",
            "matrix_metrics_sha256": sha256_file(metrics),
            "required_status": "completed",
            "required_candidate_count": 1,
            "required_lengths": [length],
        },
        "protected_inputs": {"checkpoint": {"path": "checkpoint.pt", "sha256": sha256_file(protected)}},
        "decision_policy": {"version": "test"},
        "bounds": {"maximum_candidate_count": 1, "process_candidates_individually": True},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, candidate_path


def test_plan_hash_pins_candidates_and_preserves_original_bytes(tmp_path: Path) -> None:
    config, candidate = _write_plan_fixture(tmp_path)
    before = sha256_file(candidate)
    plan = build_repair_plan(config, repository_root=tmp_path)
    assert plan["candidate_count"] == 1
    assert not plan["native_or_reference_geometry_used_for_repair"]
    assert sha256_file(candidate) == before


def test_plan_refuses_changed_candidate_hash(tmp_path: Path) -> None:
    config, candidate = _write_plan_fixture(tmp_path)
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="candidate artifact hash contradiction"):
        build_repair_plan(config, repository_root=tmp_path)


def test_plan_refuses_existing_output(tmp_path: Path) -> None:
    config, _ = _write_plan_fixture(tmp_path)
    (tmp_path / "repair_output").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_repair_plan(config, repository_root=tmp_path)


def test_repair_api_has_no_native_or_reference_geometry_input() -> None:
    parameters = set(inspect.signature(constrained_coordinate_repair).parameters)
    assert parameters == {"raw_matrix", "initial_coordinates", "config"}
