from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from protein_distance_diffusion.data.sequence_geometry import (
    DATASET_MODES,
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryVocabulary,
)
from protein_distance_diffusion.evaluation.sequence_readiness import sha256_file
from scripts.smoke_sequence_geometry_pairing import CORRUPTION_CONFIGS, DATASETS, run_smoke


def _write_npz(path: Path, sequence: str) -> None:
    coordinates = np.arange(len(sequence), dtype=np.float32)[:, None] * np.array([[3.8, 0.0, 0.0]], dtype=np.float32)
    matrix = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1).astype(np.float32)
    np.savez_compressed(path, sequence=np.asarray(sequence), distance_matrix=matrix)


def _fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "pairing"
    dataset_dir.mkdir()
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mode": "scale", "scale": 50.0}))
    normalization_hash = sha256_file(normalization)
    lengths = (20, 65, 129, 257, 385)
    methods = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR")
    classifications = (
        "verified_sequence_geometry_pair",
        "unique_sequence_pair_coordinate_unavailable",
        "residue_identity_provenance_incomplete",
    )
    rows = []
    for split in ("train", "validation"):
        for index, length in enumerate(lengths):
            sample_id = f"{split}-{length}"
            sequence = ("ACDEFGHIKLMNPQRSTVWY" * ((length + 19) // 20))[:length]
            matrix_path = tmp_path / f"{sample_id}.npz"
            _write_npz(matrix_path, sequence)
            rows.append(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "sequence_length": length,
                    "matrix_length": length,
                    "actual_length": length,
                    "matrix_path": str(matrix_path),
                    "matrix_sha256": sha256_file(matrix_path),
                    "schema_version": PAIRING_SCHEMA_VERSION,
                    "practical_training_eligibility": "conditionally_verified_pair",
                    "experimental_method": methods[index % len(methods)],
                    "pairing_classification": classifications[index % len(classifications)],
                    "normalization_file": str(normalization),
                    "normalization_sha256": normalization_hash,
                    "eligible_for_training": True,
                    "original_split": split,
                }
            )
    excluded_matrix = tmp_path / "excluded.npz"
    excluded_sequence = "ACDEFGHIKLMNPQRSTVWY"
    _write_npz(excluded_matrix, excluded_sequence)
    rows.append(
        {
            "sample_id": "excluded-20",
            "sequence": excluded_sequence,
            "sequence_length": 20,
            "matrix_length": 20,
            "actual_length": 20,
            "matrix_path": str(excluded_matrix),
            "matrix_sha256": sha256_file(excluded_matrix),
            "schema_version": PAIRING_SCHEMA_VERSION,
            "practical_training_eligibility": "excluded_or_unresolved",
            "experimental_method": "X-RAY DIFFRACTION",
            "pairing_classification": "unresolved_ambiguity",
            "normalization_file": str(normalization),
            "normalization_sha256": normalization_hash,
            "eligible_for_training": False,
            "original_split": "excluded",
        }
    )
    all_rows = pd.DataFrame(rows)
    subsets = {
        "all_pairs": all_rows,
        "eligible_train": all_rows[all_rows["original_split"] == "train"],
        "eligible_validation": all_rows[all_rows["original_split"] == "validation"],
        "excluded_pairs": all_rows[~all_rows["eligible_for_training"]],
        "pairing_ineligible": all_rows[~all_rows["eligible_for_training"]],
        "pairing_eligible_original_split_excluded": all_rows.iloc[0:0],
    }
    partitions = []
    schema_columns = None
    for name in DATASETS:
        directory = dataset_dir / f"{name}.parquet"
        directory.mkdir()
        frame = subsets[name]
        if frame.empty:
            continue
        path = directory / "part-000000.parquet"
        frame.to_parquet(path, index=False)
        current_schema = {field.name: str(field.type) for field in pq.ParquetFile(path).schema_arrow}
        schema_columns = schema_columns or current_schema
        assert current_schema == schema_columns
        partitions.append(
            {
                "dataset": name,
                "partition_index": 0,
                "path": str(path.relative_to(dataset_dir)),
                "row_count": len(frame),
                "sha256": sha256_file(path),
            }
        )
    vocabulary = SequenceGeometryVocabulary().as_dict()
    (dataset_dir / "vocabulary.json").write_text(json.dumps(vocabulary))
    (dataset_dir / "schema.json").write_text(
        json.dumps(
            {
                "schema_version": PAIRING_SCHEMA_VERSION,
                "partitioned_parquet": True,
                "columns": schema_columns,
            }
        )
    )
    input_hashes = {str(normalization): normalization_hash}
    protocol = {
        "status": "completed",
        "schema_version": PAIRING_SCHEMA_VERSION,
        "input_hashes_preserved": True,
        "input_hashes": input_hashes,
        "partitions": partitions,
        "validated_membership_counts": {
            "all_pair_count": 11,
            "eligible_train_count": 5,
            "eligible_validation_count": 5,
            "derived_dataset_excluded_count": 1,
        },
    }
    (dataset_dir / "protocol.json").write_text(json.dumps(protocol))
    (dataset_dir / "input_hashes.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in input_hashes.items())
    )
    return dataset_dir


def test_bounded_read_only_pairing_smoke(tmp_path: Path) -> None:
    dataset_dir = _fixture(tmp_path)
    scientific_paths = sorted(dataset_dir.glob("*.parquet/part-*.parquet"))
    before = {path: sha256_file(path) for path in scientific_paths}
    report_path = tmp_path / "report.json"

    report = run_smoke(
        dataset_dir=dataset_dir,
        report_path=report_path,
        samples_per_split=8,
        seed=31,
        batch_size=3,
        max_memory_mib=2048,
    )

    assert report["status"] == "passed"
    assert report["selected_sample_count"] == 10
    assert report["bounded_npz_sample_count"] == 10
    assert report["membership_counts"] == {
        "all_pairs": 11,
        "eligible_train": 5,
        "eligible_validation": 5,
        "excluded_pairs": 1,
    }
    assert set(report["selection"]["train"]["length_bins"]) == {
        "20-64",
        "65-128",
        "129-256",
        "257-384",
        "385-500",
    }
    assert set(report["loader"]["supported_modes"]) == DATASET_MODES
    assert set(report["loader"]["corruption_results"]) == set(CORRUPTION_CONFIGS)
    assert report["loader"]["corruption_results"]["additive_symmetric_noise"]["seed_changed_output"]
    assert report["loader"]["pad_id"] != report["loader"]["mask_id"]
    assert report["peak_rss_mib"] <= 2048
    assert report["dataset_writes_performed"] == 0
    assert {path: sha256_file(path) for path in scientific_paths} == before
    assert json.loads(report_path.read_text()) == report
