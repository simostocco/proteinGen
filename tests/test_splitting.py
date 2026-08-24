"""Split and deduplication tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from protein_distance_diffusion.data.clustering import deduplicate_sequences, load_mmseqs_clusters
from protein_distance_diffusion.data.splitting import (
    assert_no_leakage,
    assign_clusters_to_splits,
    build_splits_from_files,
)
from protein_distance_diffusion.data.statistics import compute_scale_statistics


def test_exact_sequence_deduplication_is_deterministic(synthetic_manifest: Path) -> None:
    """Exact duplicates keep the best-resolution sample with stable tie-breaks."""
    frame = pd.read_parquet(synthetic_manifest)
    retained, duplicates = deduplicate_sequences(frame)
    assert set(retained["sample_id"]) == {"s1", "s3"}
    assert "s2" in duplicates.loc[duplicates["representative_sample_id"] == "s1", "duplicate_sample_ids"].iloc[0]


def test_cluster_assignment_no_leakage() -> None:
    """Whole clusters remain in one split."""
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "cluster_id": ["x", "x", "y", "z"],
            "length": [50, 52, 80, 120],
            "sequence": ["AA", "AB", "CC", "DD"],
            "sequence_hash": ["h1", "h2", "h3", "h4"],
            "pdb_id": ["1", "2", "3", "4"],
            "chain_id": ["A", "A", "A", "A"],
        }
    )
    split = assign_clusters_to_splits(frame, seed=7)
    assert_no_leakage(split)
    assert split.groupby("cluster_id")["split"].nunique().max() == 1


def test_leakage_detection_rejects_overlap() -> None:
    """Exact sequence hashes cannot cross split boundaries."""
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "cluster_id": ["x", "y"],
            "sequence_hash": ["same", "same"],
            "pdb_id": ["1", "2"],
            "chain_id": ["A", "A"],
            "split": ["train", "test"],
        }
    )
    with pytest.raises(ValueError, match="Split leakage detected"):
        assert_no_leakage(frame)


def test_leakage_detection_survives_optimized_python(tmp_path: Path) -> None:
    """Leakage checks use explicit exceptions, so `python -O` cannot remove them."""
    script = (
        "import pandas as pd\n"
        "from protein_distance_diffusion.data.splitting import assert_no_leakage\n"
        "frame = pd.DataFrame({"
        "'sample_id':['a','b'], 'cluster_id':['x','y'], 'sequence_hash':['same','same'], "
        "'pdb_id':['1','2'], 'split':['train','test']})\n"
        "assert_no_leakage(frame)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Split leakage detected" in completed.stderr


def test_load_mmseqs_clusters_accepts_headerless_and_historical_header(tmp_path: Path) -> None:
    """Cluster TSV loading keeps headerless output canonical while accepting old headers."""
    headerless = tmp_path / "headerless.tsv"
    headerless.write_text("c1\ts1\nc2\ts2\n")
    historical = tmp_path / "historical.tsv"
    historical.write_text("representative_id\tmember_id\nc1\ts1\n")

    loaded = load_mmseqs_clusters(headerless)
    assert loaded.to_dict(orient="records") == [
        {"cluster_id": "c1", "sample_id": "s1"},
        {"cluster_id": "c2", "sample_id": "s2"},
    ]
    assert load_mmseqs_clusters(historical).to_dict(orient="records") == [{"cluster_id": "c1", "sample_id": "s1"}]


def test_load_mmseqs_clusters_rejects_malformed_and_contradictory_rows(tmp_path: Path) -> None:
    """Malformed cluster TSVs fail loudly instead of creating ambiguous groups."""
    malformed = tmp_path / "malformed.tsv"
    malformed.write_text("c1\ts1\textra\n")
    with pytest.raises(ValueError, match="exactly two columns"):
        load_mmseqs_clusters(malformed)

    contradictory = tmp_path / "contradictory.tsv"
    contradictory.write_text("c1\ts1\nc2\ts1\n")
    with pytest.raises(ValueError, match="contradictory duplicate"):
        load_mmseqs_clusters(contradictory)


def test_build_splits_and_train_only_statistics(tmp_path: Path, synthetic_manifest: Path) -> None:
    """Split manifests contain all samples and statistics read only train manifest."""
    clusters = tmp_path / "clusters.tsv"
    clusters.write_text("c1\ts1\nc2\ts2\nc3\ts3\n")
    split = build_splits_from_files(synthetic_manifest, clusters, tmp_path / "splits", seed=1)
    assert set(split["sample_id"]) == {"s1", "s2", "s3"}
    stats = compute_scale_statistics(tmp_path / "splits" / "train.parquet")
    assert stats["mode"] == "scale"
    assert stats["scale"] > 0
