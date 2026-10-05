"""Synthetic execution-contract tests for the E007 matrix-generator audit."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from protein_distance_diffusion.evaluation import e007_matrix_audit as audit
from protein_distance_diffusion.evaluation.distance_matrix_quality import assess_distance_matrix


def _distance_matrix(length: int) -> np.ndarray:
    coordinates = np.stack([np.arange(length, dtype=float) * 3.8, np.zeros(length), np.zeros(length)], axis=1)
    return np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1).astype(np.float32)


def test_seed_schedule_candidate_identity_and_no_averaging() -> None:
    first = audit.candidate_seed_schedule({8: 2, 4: 2}, 17)
    second = audit.candidate_seed_schedule({4: 2, 8: 2}, 17)
    assert first == second
    identities = [
        audit.generated_candidate_id(length, index, seed)
        for length, seeds in first.items()
        for index, seed in enumerate(seeds)
    ]
    assert len(identities) == len(set(identities)) == 4
    assert "mean" not in inspect.getsource(audit.generated_candidate_id)


def test_real_panel_is_deterministic_and_scan_order_independent(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": f"sample_{index}",
            "length": 4 if index % 2 == 0 else 8,
            "path": f"matrix_{index}.npz",
            "pdb_id": "1abc",
            "chain_id": "A",
            "model_number": 1,
            "split": "validation",
        }
        for index in range(20)
    ]
    first_path, second_path = tmp_path / "first.parquet", tmp_path / "second.parquet"
    pq.write_table(pa.Table.from_pylist(rows), first_path, row_group_size=3)
    pq.write_table(pa.Table.from_pylist(list(reversed(rows))), second_path, row_group_size=7)
    first, first_record = audit.select_real_reference_panel(first_path, counts_by_length={4: 3, 8: 2}, seed=9)
    second, second_record = audit.select_real_reference_panel(second_path, counts_by_length={4: 3, 8: 2}, seed=9)
    assert [row["sample_id"] for row in first] == [row["sample_id"] for row in second]
    assert first_record["sample_id_sha256"] == second_record["sample_id_sha256"]
    assert len({row["sample_id"] for row in first}) == 5


def test_two_axis_permutation_preserves_euclidean_validity_and_changes_chain_order() -> None:
    matrix = _distance_matrix(8)
    permuted, provenance = audit.residue_permuted_control(matrix, seed=42)
    permutation = np.asarray(provenance["permutation"])
    assert np.array_equal(permuted, matrix[np.ix_(permutation, permutation)])
    assert assess_distance_matrix(permuted)["euclidean_gram_valid"]
    assert provenance["generic_non_euclidean_control"] is False
    assert not np.allclose(np.diag(permuted, k=1), np.diag(matrix, k=1))


@pytest.mark.parametrize(
    ("name", "settings", "expected_metric"),
    [
        ("asymmetry", {"delta_angstrom": 2.0}, "symmetry_error_max_angstrom"),
        ("nonzero_diagonal", {"value_angstrom": 1.0}, "diagonal_error_max_angstrom"),
        ("negative_distance", {"value_angstrom": -1.0}, "negative_distance_count"),
        ("local_chain_disruption", {"adjacent_delta_angstrom": 8.0}, "adjacent_residue_distance_rmse_angstrom"),
        ("triangle_violation", {"excess_angstrom": 2.0}, "triangle_violation_fraction"),
        ("non_euclidean_four_point", {"scale_angstrom": 4.0}, "negative_eigenmass_fraction"),
    ],
)
def test_explicit_corruptions_are_separate_and_provenanced(
    name: str, settings: dict[str, float], expected_metric: str
) -> None:
    corrupted, provenance = audit.explicit_corruption(_distance_matrix(8), corruption_type=name, settings=settings)
    report = assess_distance_matrix(corrupted)
    assert provenance["control_type"] == name
    assert provenance["settings"] == settings
    assert float(report[expected_metric]) > 0.0


def test_candidate_npz_lossless_round_trip(tmp_path: Path) -> None:
    matrix = _distance_matrix(5)
    mask = np.ones((5, 5), dtype=bool)
    path = tmp_path / "candidate.npz"
    audit._atomic_npz(
        path,
        candidate_id=np.asarray("candidate"),
        requested_length=np.asarray(5),
        actual_valid_length=np.asarray(5),
        sampling_seed=np.asarray(12),
        normalized_matrix=matrix / 53.775,
        physical_matrix_angstrom=matrix,
        pair_mask=mask,
        metadata=np.asarray(json.dumps({"source": "synthetic"})),
    )
    loaded = audit.load_candidate_npz(path)
    assert loaded["candidate_id"] == "candidate"
    assert loaded["sampling_seed"] == 12
    assert np.array_equal(loaded["physical_matrix_angstrom"], matrix)
    assert np.array_equal(loaded["pair_mask"], mask)


def test_aggregation_is_length_stratified_and_paired() -> None:
    config = {
        "bounds": {"triangle_exact_max_length": 64, "triangle_sample_count": 8, "eigen_exact_max_length": 64},
        "fatal_contract_thresholds": {
            "require_finite": True,
            "maximum_symmetry_error_angstrom": 1e-5,
            "maximum_diagonal_error_angstrom": 1e-5,
            "maximum_negative_distance_count": 0,
        },
    }
    rows = []
    for length in (4, 8):
        source = f"real::{length}"
        rows.append(
            audit._assessed_row(
                _distance_matrix(length),
                candidate_id=source,
                panel="real_validation",
                requested_length=length,
                metric_seed=1,
                config=config,
            )
        )
        corrupted, provenance = audit.explicit_corruption(
            _distance_matrix(length), corruption_type="nonzero_diagonal", settings={"value_angstrom": 1.0}
        )
        rows.append(
            audit._assessed_row(
                corrupted,
                candidate_id=f"control::{length}",
                panel="corruption_nonzero_diagonal",
                requested_length=length,
                metric_seed=1,
                config=config,
                source_matrix_id=source,
                provenance=provenance,
            )
        )
    aggregates = audit.aggregate_metrics(rows, bootstrap_iterations=10, bootstrap_seed=2)
    assert {row["requested_length"] for row in aggregates} >= {None, 4, 8}
    paired = audit.paired_control_statistics(rows)
    diagonal = next(row for row in paired if row["metric"] == "diagonal_error_max_angstrom")
    assert diagonal["mean_difference"] == pytest.approx(1.0)
    comparisons = audit.aggregate_panel_comparisons(aggregates)
    assert any(row["right_panel"] == "corruption_nonzero_diagonal" for row in comparisons)


def test_fatal_failures_and_reference_warnings_remain_separate() -> None:
    thresholds = {
        "require_finite": True,
        "maximum_symmetry_error_angstrom": 1e-5,
        "maximum_diagonal_error_angstrom": 1e-5,
        "maximum_negative_distance_count": 0,
    }
    fatal = assess_distance_matrix(np.array([[1.0, -1.0], [-1.0, 0.0]]))
    reasons = audit.fatal_validity_reasons(fatal, thresholds)
    assert "diagonal_contract_violation" in reasons
    assert "negative_distance_contract_violation" in reasons


def _synthetic_execution_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    protected = tmp_path / "protected.txt"
    protected.write_text("immutable", encoding="utf-8")
    matrix_path = tmp_path / "real.npz"
    np.savez_compressed(matrix_path, distance_matrix=_distance_matrix(4))
    output = tmp_path / "audit"
    config = {
        "output_dir": str(output),
        "generator": {
            "checkpoint_path": "checkpoint.pt",
            "checkpoint_sha256": "checkpoint-sha",
            "training_config_path": "training.yaml",
            "training_config_sha256": "training-config-sha",
            "diffusion_steps": 2,
            "prediction_parameterization": "v",
        },
        "dataset": {"validation_manifest_path": "validation.parquet", "validation_manifest_sha256": "manifest"},
        "matrix_representation": {"normalization": {"scale_angstrom": 53.775}},
        "audit": {
            "generated_master_seed": 3,
            "generated_counts_by_length": {4: 2},
            "real_counts_by_length": {4: 1},
            "panel_selection_seed": 4,
            "corruption_seed": 5,
            "bootstrap_seed": 6,
            "bootstrap_iterations": 5,
            "reference_warning_quantiles": [0.01, 0.99],
            "scientific_threshold_policy": "reference_quantile_warnings_only",
            "corruption_controls": {
                "asymmetry": {"delta_angstrom": 2.0},
                "nonzero_diagonal": {"value_angstrom": 1.0},
                "negative_distance": {"value_angstrom": -1.0},
                "local_chain_disruption": {"adjacent_delta_angstrom": 8.0},
                "triangle_violation": {"excess_angstrom": 2.0},
                "non_euclidean_four_point": {"scale_angstrom": 4.0},
            },
        },
        "fatal_contract_thresholds": {
            "require_finite": True,
            "maximum_symmetry_error_angstrom": 1e-5,
            "maximum_diagonal_error_angstrom": 1e-5,
            "maximum_negative_distance_count": 0,
        },
        "bounds": {
            "runtime_device": "cpu",
            "maximum_rss_mib": 100000,
            "maximum_cuda_allocated_mib": 1,
            "maximum_cuda_reserved_mib": 1,
            "triangle_exact_max_length": 64,
            "triangle_sample_count": 8,
            "eigen_exact_max_length": 64,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    plan = {
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "generator": config["generator"],
        "dataset": config["dataset"],
        "matrix_representation": config["matrix_representation"],
        "protected_inputs": {"fixture": {"path": protected.name, "sha256": audit.sha256_file(protected)}},
    }
    return config_path, plan, matrix_path


def test_synthetic_execution_publishes_atomically_and_is_non_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, plan, matrix_path = _synthetic_execution_fixture(tmp_path)
    record = {
        "sample_id": "real",
        "length": 4,
        "requested_length": 4,
        "path": str(matrix_path),
        "pdb_id": "1abc",
        "chain_id": "A",
        "model_number": 1,
        "split": "validation",
    }
    monkeypatch.setattr(
        audit,
        "select_real_reference_panel",
        lambda *args, **kwargs: ([record], {"sample_ids": ["real"]}),
    )
    monkeypatch.setattr(audit, "_checkpoint_model", lambda *args, **kwargs: (object(), object(), {}))

    def sample_candidate(**kwargs):
        del kwargs
        matrix = _distance_matrix(4)
        return matrix / 53.775, matrix, np.ones((4, 4), dtype=bool)

    output = audit.run_matrix_generator_audit(
        config_path, plan=plan, repository_root=tmp_path, sample_candidate=sample_candidate
    )
    protocol = json.loads((output / "protocol.json").read_text())
    report = json.loads((output / "report.json").read_text())
    assert protocol["status"] == "completed"
    assert not protocol["training_performed"]
    assert not protocol["backward_performed"]
    assert not protocol["optimizer_created"]
    assert not protocol["authorizes_training"]
    assert report["counts"]["generated_candidates"] == 2
    assert report["counts"]["by_panel"]["generated"] == 2
    assert report["scientific_review_classification"] == "completed_requires_scientific_review"
    assert report["protected_inputs_unchanged"]
    assert (tmp_path / "protected.txt").read_text(encoding="utf-8") == "immutable"
    assert not output.with_name(f".{output.name}.inprogress").exists()


def test_failed_execution_never_publishes_completed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path, plan, matrix_path = _synthetic_execution_fixture(tmp_path)
    record = {
        "sample_id": "real",
        "length": 4,
        "requested_length": 4,
        "path": str(matrix_path),
        "pdb_id": "1abc",
        "chain_id": "A",
        "model_number": 1,
        "split": "validation",
    }
    monkeypatch.setattr(audit, "select_real_reference_panel", lambda *args, **kwargs: ([record], {}))
    monkeypatch.setattr(audit, "_checkpoint_model", lambda *args, **kwargs: (object(), object(), {}))

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        audit.run_matrix_generator_audit(config_path, plan=plan, repository_root=tmp_path, sample_candidate=fail)
    output = tmp_path / "audit"
    assert not output.exists()
    heartbeat = json.loads((tmp_path / ".audit.inprogress" / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert not heartbeat["completed_output_published"]
