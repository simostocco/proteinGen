"""End-to-end split workflow tests with cached mocked MMseqs2 output."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from protein_distance_diffusion.data.splitting import run_split_workflow


def test_split_workflow_dedup_clusters_components_and_audit(tmp_path: Path) -> None:
    """The split workflow deduplicates, normalizes clusters, groups leakage links, and audits overlaps."""
    manifest = tmp_path / "manifest.parquet"
    rows = [
        {
            "sample_id": "s1",
            "pdb_id": "P1",
            "chain_id": "A",
            "sequence": "AAAA",
            "length": 4,
            "path": "s1.npz",
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.0,
        },
        {
            "sample_id": "s2",
            "pdb_id": "P2",
            "chain_id": "A",
            "sequence": "AAAA",
            "length": 4,
            "path": "s2.npz",
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 2.0,
        },
        {
            "sample_id": "s3",
            "pdb_id": "P3",
            "chain_id": "A",
            "sequence": "BBBB",
            "length": 4,
            "path": "s3.npz",
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.5,
        },
        {
            "sample_id": "s4",
            "pdb_id": "P3",
            "chain_id": "B",
            "sequence": "CCCC",
            "length": 4,
            "path": "s4.npz",
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.5,
        },
        {
            "sample_id": "s5",
            "pdb_id": "P5",
            "chain_id": "A",
            "sequence": "DDDD",
            "length": 4,
            "path": "s5.npz",
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.5,
        },
    ]
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    prefix = tmp_path / "mmseqs" / "clusters"
    prefix.parent.mkdir()
    raw_cluster = Path(f"{prefix}_cluster.tsv")
    raw_cluster.write_text("c1\ts1\nc1\ts3\nc5\ts5\n")
    result = run_split_workflow(
        manifest_path=manifest,
        deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
        deduplication_report_path=tmp_path / "deduplication_report.json",
        fasta_path=tmp_path / "retained.fasta",
        mmseqs_output_prefix=prefix,
        mmseqs_tmp_dir=tmp_path / "tmp",
        cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
        output_dir=tmp_path / "splits",
        seed=42,
        fractions=(0.6, 0.2, 0.2),
        force=False,
    )
    assert result["original_count"] == 5
    assert result["retained_count"] == 4
    assert result["removed_count"] == 1
    retained = pd.read_parquet(tmp_path / "manifest_deduplicated.parquet")
    assert set(retained["sample_id"]) == {"s1", "s3", "s4", "s5"}
    clusters = pd.read_csv(tmp_path / "mmseqs_clusters.tsv", sep="\t")
    assert set(clusters["sample_id"]) == {"s1", "s3", "s4", "s5"}
    assert clusters.loc[clusters["sample_id"] == "s4", "cluster_id"].iloc[0] == "s4"
    split = pd.read_parquet(tmp_path / "splits" / "split_assignments.parquet")
    assert split.groupby("cluster_id")["split"].nunique().max() == 1
    assert split.groupby("pdb_id")["split"].nunique().max() == 1
    audit = json.loads((tmp_path / "splits" / "leakage_audit.json").read_text())
    assert all(check["overlap_count"] == 0 for check in audit["leakage_checks"].values())
    report = json.loads((tmp_path / "deduplication_report.json").read_text())
    assert sorted(report["duplicate_group_sizes"], reverse=True) == [2, 1, 1, 1]
