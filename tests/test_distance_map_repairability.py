"""Tests for E004 distance-map repairability helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protein_distance_diffusion.evaluation.repairability import (
    classical_mds_rank3_projection,
    corrupt_distance_matrix,
    export_ca_pdb,
    kabsch_rmsd,
    pairwise_distances,
    repairability_metrics,
    sha256_file,
    trace_metrics,
)


def _coords(length: int = 8) -> np.ndarray:
    x = np.arange(length, dtype=np.float64)
    return np.stack([3.8 * x, np.sin(x), np.cos(x / 2.0)], axis=1)


def test_exact_euclidean_distance_matrix_projects_with_near_zero_error() -> None:
    matrix = pairwise_distances(_coords())
    projection = classical_mds_rank3_projection(matrix)
    assert projection.projected_distances == pytest.approx(matrix, abs=1e-8)
    metrics = repairability_metrics(matrix)
    assert metrics["offdiagonal_rmse_angstrom"] < 1e-8


def test_kabsch_alignment_accepts_rigid_transform_and_reflection() -> None:
    coords = _coords()
    transformed = coords[:, [1, 0, 2]] * np.array([1.0, -1.0, 1.0])
    assert kabsch_rmsd(coords, transformed, allow_reflection=True) < 1e-10


def test_non_euclidean_matrix_has_positive_projection_error() -> None:
    matrix = pairwise_distances(_coords())
    matrix[0, 5] = matrix[5, 0] = 80.0
    metrics = repairability_metrics(matrix)
    assert metrics["offdiagonal_rmse_angstrom"] > 1.0
    assert metrics["negative_eigenvalue_mass_fraction_before_projection"] > 0.0


def test_padding_entries_can_be_excluded_by_slicing_valid_region() -> None:
    matrix = pairwise_distances(_coords(6))
    padded = np.pad(matrix, ((0, 4), (0, 4)), constant_values=999.0)
    np.fill_diagonal(padded, 0.0)
    assert repairability_metrics(padded[:6, :6])["offdiagonal_rmse_angstrom"] == pytest.approx(
        repairability_metrics(matrix)["offdiagonal_rmse_angstrom"]
    )


def test_trace_metrics_report_chain_plausibility() -> None:
    metrics = trace_metrics(_coords(10))
    assert metrics["ca_adjacent_distance_mean"] > 3.0
    assert metrics["maximum_chain_discontinuity"] < 5.0
    assert np.isfinite(metrics["virtual_bond_angle_mean_degrees"])


def test_corruption_is_deterministic_and_symmetric() -> None:
    matrix = pairwise_distances(_coords())
    left = corrupt_distance_matrix(matrix, noise_angstrom=1.0, seed=7)
    right = corrupt_distance_matrix(matrix, noise_angstrom=1.0, seed=7)
    assert np.array_equal(left, right)
    assert np.allclose(left, left.T)
    assert np.allclose(np.diag(left), 0.0)


def test_pdb_export_has_one_ordered_ca_atom_per_residue(tmp_path: Path) -> None:
    path = tmp_path / "trace.pdb"
    export_ca_pdb(path, _coords(4), sample_id="sample")
    atoms = [line for line in path.read_text().splitlines() if line.startswith("ATOM")]
    assert len(atoms) == 4
    assert atoms[0][22:26].strip() == "1"
    assert atoms[-1][22:26].strip() == "4"


def test_synthetic_smoke_preserves_raw_hashes(tmp_path: Path) -> None:
    from scripts.analyze_distance_map_repairability import run_synthetic_smoke

    protocol_path = run_synthetic_smoke(tmp_path / "smoke")
    protocol = json.loads(protocol_path.read_text())
    assert protocol["raw_input_hashes_preserved"] is True
    assert Path(protocol_path.parent / "per_sample_repairability.parquet").exists()
    for path, digest in protocol["raw_input_hashes_before"].items():
        assert sha256_file(path) == digest


def _write_completed_candidate_protocol(root: Path, *, checkpoint_sha256: str) -> Path:
    generated = root / "generated" / "N0004"
    generated.mkdir(parents=True)
    matrix = pairwise_distances(_coords(4))
    np.savez(
        generated / "sample.npz",
        sample_id=np.array("N0004_i00000_seed1"),
        requested_length=np.array(4),
        sample_index=np.array(0),
        seed=np.array(1),
        physical_distance_matrix_angstrom=matrix,
    )
    protocol_path = root / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": checkpoint_sha256,
                "sample_counts_by_length": {"4": 1},
            }
        )
    )
    return protocol_path


def _write_evaluation_bank(root: Path, *, lengths: list[int], count_per_length: int, checkpoint_sha256: str) -> Path:
    counts = {}
    for requested_length in lengths:
        generated = root / "generated" / f"N{requested_length:04d}"
        generated.mkdir(parents=True)
        counts[str(requested_length)] = count_per_length
        for index in range(count_per_length):
            matrix = pairwise_distances(_coords(requested_length) + index * 0.01)
            np.savez(
                generated / f"sample_{index}.npz",
                sample_id=np.array(f"N{requested_length:04d}_i{index:05d}_seed{index}"),
                requested_length=np.array(requested_length),
                sample_index=np.array(index),
                seed=np.array(index),
                physical_distance_matrix_angstrom=matrix,
            )
    protocol_path = root / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": checkpoint_sha256,
                "sample_counts_by_length": counts,
            }
        )
    )
    return protocol_path


def _write_real_controls(root: Path, *, lengths: list[int], count_per_length: int) -> Path:
    rows = []
    samples = root / "real_samples"
    samples.mkdir()
    for requested_length in lengths:
        for index in range(count_per_length):
            actual_length = requested_length + (1 if requested_length == max(lengths) else 0)
            path = samples / f"real_N{requested_length}_{index}.npz"
            np.savez(path, distance_matrix=pairwise_distances(_coords(actual_length)))
            rows.append(
                {
                    "sample_id": f"real_N{requested_length}_{index}",
                    "path": str(path),
                    "requested_length": requested_length,
                    "length": actual_length,
                    "seed": index,
                }
            )
    manifest = root / "real_controls.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    return manifest


def test_definitive_checkpoint_provenance_is_validated_before_derived_output(tmp_path: Path) -> None:
    from scripts.analyze_distance_map_repairability import run_repairability_analysis

    definitive_hash = "59db27a3dbecbc199cb20065e1263ad0cec12b3553cca1a263429f890b38ea86"
    preflight_hash = "ecd1b34780e74bf2f367bf2ced884e57e8a6ec3233f6b75f29c798cdbca80d01"
    candidate_dir = tmp_path / "candidate"
    protocol_path = _write_completed_candidate_protocol(candidate_dir, checkpoint_sha256=definitive_hash)
    sample_path = candidate_dir / "generated" / "N0004" / "sample.npz"
    raw_hashes = {protocol_path: sha256_file(protocol_path), sample_path: sha256_file(sample_path)}

    output_dir = tmp_path / "mismatched_output"
    with pytest.raises(ValueError, match="candidate checkpoint SHA-256 mismatch"):
        run_repairability_analysis(
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            expected_candidate_checkpoint_sha256=preflight_hash,
        )
    assert not output_dir.exists()
    assert {path: sha256_file(path) for path in raw_hashes} == raw_hashes

    matching_output = tmp_path / "matching_output"
    protocol = json.loads(
        run_repairability_analysis(
            candidate_dir=candidate_dir,
            output_dir=matching_output,
            expected_candidate_checkpoint_sha256=definitive_hash,
        ).read_text()
    )
    assert protocol["candidate_checkpoint_sha256"] == definitive_hash


def test_real_controls_are_limited_independently_by_requested_length(tmp_path: Path) -> None:
    from scripts.analyze_distance_map_repairability import run_repairability_analysis

    lengths = [4, 5]
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    _write_evaluation_bank(candidate, lengths=lengths, count_per_length=3, checkpoint_sha256="candidate")
    _write_evaluation_bank(baseline, lengths=lengths, count_per_length=3, checkpoint_sha256="baseline")
    real_manifest = _write_real_controls(tmp_path, lengths=lengths, count_per_length=3)

    protocol_path = run_repairability_analysis(
        candidate_dir=candidate,
        baseline_dir=baseline,
        real_manifest=real_manifest,
        output_dir=tmp_path / "output",
        expected_candidate_checkpoint_sha256="candidate",
        limit_per_length=2,
    )
    output_dir = protocol_path.parent
    per_sample = pd.read_parquet(output_dir / "per_sample_repairability.parquet")
    counts = per_sample.groupby(["source_type", "requested_length"]).size().to_dict()
    assert counts == {
        ("E002", 4): 2,
        ("E002", 5): 2,
        ("E004", 4): 2,
        ("E004", 5): 2,
        ("real_control", 4): 2,
        ("real_control", 5): 2,
    }
    assert set(per_sample["source_type"]) == {"E002", "E004", "real_control"}
    assert set(per_sample["requested_length"]) == set(lengths)
    long_real = per_sample[(per_sample["source_type"] == "real_control") & (per_sample["requested_length"] == 5)]
    assert set(long_real["actual_length"]) == {6}

    corruption = pd.read_csv(output_dir / "corruption_recovery_by_length.csv")
    assert len(corruption) == len(lengths) * 5
    assert set(map(tuple, corruption[["requested_length", "noise_angstrom"]].to_numpy())) == {
        (length, noise) for length in lengths for noise in (0.25, 0.5, 1.0, 2.0, 4.0)
    }

    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "completed"
    assert protocol["candidate_status"] == "completed"
    assert protocol["expected_candidate_checkpoint_sha256"] == "candidate"
    assert protocol["raw_inputs_unchanged"] is True
    assert protocol["raw_input_hashes_preserved"] is True
    assert protocol["runtime_seconds"] >= 0.0
    assert protocol["sample_counts_by_source_and_requested_length"] == {
        "E002": {"4": 2, "5": 2},
        "E004": {"4": 2, "5": 2},
        "real_control": {"4": 2, "5": 2},
    }


def test_missing_real_cohort_fails_before_output_mutation(tmp_path: Path) -> None:
    from scripts.analyze_distance_map_repairability import run_repairability_analysis

    candidate = tmp_path / "candidate"
    protocol_path = _write_evaluation_bank(candidate, lengths=[4, 5], count_per_length=2, checkpoint_sha256="candidate")
    real_manifest = _write_real_controls(tmp_path, lengths=[4], count_per_length=2)
    sample_path = candidate / "generated" / "N0004" / "sample_0.npz"
    raw_hashes = {protocol_path: sha256_file(protocol_path), sample_path: sha256_file(sample_path)}

    output_dir = tmp_path / "existing_pilot"
    output_dir.mkdir()
    sentinel = output_dir / "completed_pilot.txt"
    sentinel.write_text("preserve me")
    sentinel_hash = sha256_file(sentinel)

    with pytest.raises(ValueError, match="Insufficient real controls for requested_length=5"):
        run_repairability_analysis(
            candidate_dir=candidate,
            real_manifest=real_manifest,
            output_dir=output_dir,
            expected_candidate_checkpoint_sha256="candidate",
            limit_per_length=2,
            restart=True,
        )
    assert sha256_file(sentinel) == sentinel_hash
    assert {path: sha256_file(path) for path in raw_hashes} == raw_hashes


def test_generated_count_zero_warning_is_data_not_missing(tmp_path: Path) -> None:
    path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["sample_id"]).to_parquet(path, index=False)
    assert pd.read_parquet(path).empty
