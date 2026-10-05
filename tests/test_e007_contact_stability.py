"""Synthetic tests for E007 Phase-2C contact stability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from protein_distance_diffusion.evaluation.e007_contact_stability import (
    _aggregate_candidate_metrics,
    ambiguity_exclusion_metrics,
    analyze_candidate_pair,
    build_contact_stability_plan,
    contact_confusion,
    neighbourhood_metrics,
    run_contact_stability_audit,
    sha256_file,
    threshold_sweep_metrics,
)

ROOT = Path(__file__).parents[1]


def _config() -> dict:
    config = yaml.safe_load((ROOT / "configs/e007_contact_stability_audit.yaml").read_text())
    config["contact_thresholds"]["start_angstrom"] = 5.0
    config["contact_thresholds"]["stop_angstrom"] = 12.0
    config["contact_thresholds"]["step_angstrom"] = 0.5
    return config


def _line_matrix(length: int, spacing: float = 3.8) -> np.ndarray:
    coordinates = np.stack([np.arange(length, dtype=np.float64) * spacing, np.zeros(length), np.zeros(length)], axis=1)
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)


def test_exact_topology_preservation() -> None:
    matrix = _line_matrix(30)
    row, thresholds, residues, _ = analyze_candidate_pair(
        matrix,
        matrix.copy(),
        candidate_id="exact",
        requested_length=30,
        method="constrained_coordinate_repair",
        config=_config(),
    )
    assert row["contact_8A"]["f1"] == pytest.approx(1.0)
    assert row["continuous"]["pairwise_distance_rmse_angstrom"] == 0.0
    assert row["neighbourhood"]["8"]["mean"] == pytest.approx(1.0)
    assert len([item for item in thresholds if item["metric_scope"] == "threshold_sweep"]) == 15
    assert residues


def test_near_threshold_flip_disappears_after_ambiguity_exclusion() -> None:
    raw = np.full((4, 4), 20.0)
    repaired = raw.copy()
    np.fill_diagonal(raw, 0.0)
    np.fill_diagonal(repaired, 0.0)
    raw[0, 1] = raw[1, 0] = 7.9
    repaired[0, 1] = repaired[1, 0] = 8.1
    raw[2, 3] = raw[3, 2] = repaired[2, 3] = repaired[3, 2] = 6.0
    hard = contact_confusion(raw, repaired, threshold=8.0)
    excluded = ambiguity_exclusion_metrics(raw, repaired, threshold=8.0, half_widths=[0.25])[0]
    assert hard["false_negative"] == 1
    assert excluded["false_negative"] == 0
    assert excluded["f1"] == pytest.approx(1.0)


def test_confident_contact_loss_remains_after_exclusion() -> None:
    raw = np.full((3, 3), 20.0)
    repaired = raw.copy()
    np.fill_diagonal(raw, 0.0)
    np.fill_diagonal(repaired, 0.0)
    raw[0, 2] = raw[2, 0] = 6.0
    repaired[0, 2] = repaired[2, 0] = 10.0
    result = ambiguity_exclusion_metrics(raw, repaired, threshold=8.0, half_widths=[1.0])[0]
    assert result["false_negative"] == 1
    assert result["recall"] == 0.0


def test_false_positives_and_false_negatives_are_separate() -> None:
    raw = np.full((4, 4), 12.0)
    repaired = raw.copy()
    np.fill_diagonal(raw, 0.0)
    np.fill_diagonal(repaired, 0.0)
    raw[0, 1] = raw[1, 0] = 6.0
    repaired[0, 1] = repaired[1, 0] = 10.0
    repaired[2, 3] = repaired[3, 2] = 6.0
    result = contact_confusion(raw, repaired, threshold=8.0)
    assert result["false_positive"] == 1
    assert result["false_negative"] == 1
    assert result["true_positive"] == 0


def test_exact_absence_of_contacts_is_perfect_preservation() -> None:
    raw = np.full((4, 4), 20.0)
    np.fill_diagonal(raw, 0.0)
    result = contact_confusion(raw, raw, threshold=8.0)
    assert result["contact_prevalence"] == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["jaccard"] == 1.0


def test_long_range_contact_loss_is_stratified() -> None:
    raw = np.full((30, 30), 20.0)
    repaired = raw.copy()
    np.fill_diagonal(raw, 0.0)
    np.fill_diagonal(repaired, 0.0)
    raw[0, 24] = raw[24, 0] = 6.0
    repaired[0, 24] = repaired[24, 0] = 10.0
    _, thresholds, _, _ = analyze_candidate_pair(
        raw,
        repaired,
        candidate_id="long",
        requested_length=30,
        method="rank3_psd_projection",
        config=_config(),
    )
    long_range = next(row for row in thresholds if row.get("band") == "separation_24_or_more")
    assert long_range["false_negative"] == 1
    assert long_range["recall"] == 0.0


def test_nearest_neighbour_retention_excludes_self_and_detects_change() -> None:
    raw = _line_matrix(12)
    permutation = np.array([0, 5, 2, 3, 4, 1, 6, 7, 8, 9, 10, 11])
    repaired = raw[np.ix_(permutation, permutation)]
    rows, summary, _ = neighbourhood_metrics(raw, repaired, k_values=[4, 8, 16, 32], retention_threshold=0.7)
    assert summary["4"]["mean"] < 1.0
    assert summary["32"]["effective_k"] == 11
    assert all(row["retained_neighbour_count"] <= row["effective_k"] for row in rows)


def test_threshold_sweep_is_deterministic() -> None:
    raw = _line_matrix(10)
    repaired = raw.copy()
    repaired[0, 2] = repaired[2, 0] = 9.0
    thresholds = [5.0 + 0.5 * index for index in range(15)]
    assert threshold_sweep_metrics(raw, repaired, thresholds) == threshold_sweep_metrics(raw, repaired, thresholds)


def test_candidate_and_length_aggregation_remains_stratified() -> None:
    config = _config()
    rows = []
    for length in (30, 40):
        matrix = _line_matrix(length)
        for method in ("rank3_psd_projection", "constrained_coordinate_repair"):
            row, _, _, _ = analyze_candidate_pair(
                matrix,
                matrix,
                candidate_id=f"sample_{length}_{method}",
                requested_length=length,
                method=method,
                config=config,
            )
            rows.append(row)
    aggregates = _aggregate_candidate_metrics(rows)
    lengths = {row["requested_length"] for row in aggregates}
    assert lengths == {None, 30, 40}


def _write_npz(path: Path, candidate_id: str, matrix: np.ndarray) -> None:
    length = matrix.shape[0]
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_id=np.asarray(candidate_id),
            requested_length=np.asarray(length),
            actual_valid_length=np.asarray(length),
            sampling_seed=np.asarray(7),
            normalized_matrix=(matrix / 53.775).astype(np.float32),
            physical_matrix_angstrom=matrix.astype(np.float32),
            pair_mask=np.ones(matrix.shape, dtype=bool),
            metadata=np.asarray(json.dumps({"candidate_id": candidate_id})),
        )


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    phase2 = tmp_path / "phase2"
    phase2b = tmp_path / "phase2b"
    (phase2 / "candidates").mkdir(parents=True)
    (phase2b / "repaired").mkdir(parents=True)
    candidate_id = "candidate_30"
    matrix = _line_matrix(30).astype(np.float32)
    raw_path = phase2 / "candidates" / f"{candidate_id}.npz"
    _write_npz(raw_path, candidate_id, matrix)
    raw_relative = f"candidates/{candidate_id}.npz"
    raw_sha = sha256_file(raw_path)
    raw_manifest = phase2 / "candidate_manifest.jsonl"
    raw_manifest.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "candidate_artifact_path": raw_relative,
                "candidate_artifact_sha256": raw_sha,
                "requested_length": 30,
            }
        )
        + "\n"
    )
    phase2_report = phase2 / "report.json"
    phase2_report.write_text(json.dumps({"status": "completed_requires_scientific_review"}))
    phase2_protocol = phase2 / "protocol.json"
    phase2_protocol.write_text(
        json.dumps(
            {
                "status": "completed",
                "report_sha256": sha256_file(phase2_report),
                "authorizes_training": False,
                "authorizes_joint_training": False,
                "published_payload_hashes": {
                    "candidate_manifest.jsonl": sha256_file(raw_manifest),
                    raw_relative: raw_sha,
                },
            }
        )
    )
    repaired_path = phase2b / "repaired" / f"{candidate_id}.npz"
    with repaired_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_id=np.asarray(candidate_id),
            source_candidate_path=np.asarray(raw_relative),
            source_candidate_sha256=np.asarray(raw_sha),
            rank3_coordinates=np.zeros((30, 3), dtype=np.float32),
            rank3_distance_matrix_angstrom=matrix,
            constrained_coordinates=np.zeros((30, 3), dtype=np.float32),
            constrained_distance_matrix_angstrom=matrix,
            metadata=np.asarray("{}"),
        )
    repaired_relative = f"repaired/{candidate_id}.npz"
    repaired_sha = sha256_file(repaired_path)
    repair_manifest = phase2b / "repair_manifest.jsonl"
    repair_manifest.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "requested_length": 30,
                "source_candidate_path": raw_relative,
                "source_candidate_sha256": raw_sha,
                "repaired_artifact_path": repaired_relative,
                "repaired_artifact_sha256": repaired_sha,
            }
        )
        + "\n"
    )
    quality = {"negative_eigenmass_fraction": 0.1, "rank3_reconstruction_rmse_angstrom": 1.0}
    topology = {"pairwise_distance_rmse_angstrom": 0.0}
    metric_rows = [
        {
            "candidate_id": candidate_id,
            "requested_length": 30,
            "version": "raw_generated",
            "quality": quality,
            "topology": topology,
            "method_diagnostics": {},
        },
        {
            "candidate_id": candidate_id,
            "requested_length": 30,
            "version": "rank3_psd_projection",
            "quality": quality,
            "topology": topology,
            "method_diagnostics": {},
        },
        {
            "candidate_id": candidate_id,
            "requested_length": 30,
            "version": "constrained_coordinate_repair",
            "quality": quality,
            "topology": topology,
            "method_diagnostics": {
                "termination_reason": "maximum_iterations",
                "iterations": 10,
                "final_gradient_norm": 0.1,
                "final_component_losses": {
                    "total": 1.0,
                    "robust_distance_fit": 0.5,
                    "adjacent_bond": 0.2,
                    "steric_clash": 0.1,
                },
            },
        },
    ]
    metrics = phase2b / "matrix_metrics.jsonl"
    metrics.write_text("".join(json.dumps(row) + "\n" for row in metric_rows))
    phase2b_report = phase2b / "report.json"
    phase2b_report.write_text(
        json.dumps(
            {
                "decision": {"classification": "repair_is_length_limited"},
                "authorizes_training": False,
                "authorizes_joint_training": False,
                "published_payload_hashes": {
                    "matrix_metrics.jsonl": sha256_file(metrics),
                    "repair_manifest.jsonl": sha256_file(repair_manifest),
                    repaired_relative: repaired_sha,
                },
            }
        )
    )
    phase2b_protocol = phase2b / "protocol.json"
    phase2b_protocol.write_text(
        json.dumps(
            {
                "status": "completed",
                "report_sha256": sha256_file(phase2b_report),
                "authorizes_training": False,
                "authorizes_joint_training": False,
            }
        )
    )
    protected = {}
    for name in ("e004", "e006", "phase2_config", "phase2b_config"):
        path = tmp_path / name
        path.write_text(name)
        protected[name] = {"path": path.name, "sha256": sha256_file(path)}
    config = _config()
    config.update(
        {
            "output_dir": "output",
            "required_candidate_count": 1,
            "required_lengths": [30],
            "phase2_source": {
                "directory": "phase2",
                "report_path": "phase2/report.json",
                "report_sha256": sha256_file(phase2_report),
                "protocol_path": "phase2/protocol.json",
                "protocol_sha256": sha256_file(phase2_protocol),
                "candidate_manifest_path": "phase2/candidate_manifest.jsonl",
                "candidate_manifest_sha256": sha256_file(raw_manifest),
            },
            "phase2b_source": {
                "directory": "phase2b",
                "report_path": "phase2b/report.json",
                "report_sha256": sha256_file(phase2b_report),
                "protocol_path": "phase2b/protocol.json",
                "protocol_sha256": sha256_file(phase2b_protocol),
                "matrix_metrics_path": "phase2b/matrix_metrics.jsonl",
                "matrix_metrics_sha256": sha256_file(metrics),
                "repair_manifest_path": "phase2b/repair_manifest.jsonl",
                "repair_manifest_sha256": sha256_file(repair_manifest),
                "required_decision": "repair_is_length_limited",
            },
            "protected_inputs": protected,
        }
    )
    config["bounds"]["maximum_candidate_count"] = 1
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, raw_path, repaired_path


def test_changed_hash_is_refused_and_sources_are_immutable(tmp_path: Path) -> None:
    config, raw, repaired = _write_fixture(tmp_path)
    before = (sha256_file(raw), sha256_file(repaired))
    plan = build_contact_stability_plan(config, repository_root=tmp_path)
    assert plan["candidate_count"] == 1
    assert (sha256_file(raw), sha256_file(repaired)) == before
    raw.write_bytes(raw.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="raw candidate hash contradiction"):
        build_contact_stability_plan(config, repository_root=tmp_path)


def test_synthetic_execution_is_non_authorizing_and_atomic(tmp_path: Path) -> None:
    config, raw, repaired = _write_fixture(tmp_path)
    before = (sha256_file(raw), sha256_file(repaired))
    plan = build_contact_stability_plan(config, repository_root=tmp_path)
    output = run_contact_stability_audit(config, plan=plan, repository_root=tmp_path)
    protocol = json.loads((output / "protocol.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert protocol["status"] == "completed"
    assert not protocol["authorizes_training"]
    assert not protocol["authorizes_joint_training"]
    assert not protocol["authorizes_sequence_conditioning"]
    assert not protocol["optimizer_created"]
    assert protocol["optimizer_updates"] == 0
    assert not protocol["backward_performed"]
    assert not protocol["matrices_modified"]
    assert not protocol["candidates_averaged"]
    assert report["protected_inputs_unchanged"]
    assert pq.read_table(output / "per_candidate_metrics.parquet").num_rows == 2
    assert (sha256_file(raw), sha256_file(repaired)) == before


def test_existing_output_is_refused(tmp_path: Path) -> None:
    config, _, _ = _write_fixture(tmp_path)
    (tmp_path / "output").mkdir()
    with pytest.raises(FileExistsError, match="exists"):
        build_contact_stability_plan(config, repository_root=tmp_path)
