"""End-to-end split workflow tests with cached mocked MMseqs2 output."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import protein_distance_diffusion.data.splitting as splitting_module
from protein_distance_diffusion.data.clustering import (
    build_mmseqs_easy_cluster_command,
    completed_cache_metadata,
)
from protein_distance_diffusion.data.splitting import (
    assign_groups_to_splits,
    expected_mmseqs_cache_metadata,
    file_sha256,
    mmseqs_cache_metadata_path,
    run_split_workflow,
)


def _write_cache_metadata(
    *,
    manifest: Path,
    fasta: Path,
    prefix: Path,
    tmp_dir: Path,
    retained_sequences: dict[str, str],
    cluster_tsv: str,
    minimum_sequence_length: int | None,
    version: str = "test-mmseqs",
    mmseqs: str = "mmseqs",
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    cov_mode: int = 0,
    threads: int | None = None,
    split_memory_limit: str | None = None,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fasta.write_text(
        "".join(f">{sample_id}\n{retained_sequences[sample_id]}\n" for sample_id in sorted(retained_sequences))
    )
    Path(f"{prefix}_cluster.tsv").write_text(cluster_tsv)
    command = build_mmseqs_easy_cluster_command(
        fasta,
        prefix,
        tmp_dir,
        mmseqs=mmseqs,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
        split_memory_limit=split_memory_limit,
    )
    metadata = expected_mmseqs_cache_metadata(
        fasta_path=fasta,
        mmseqs_command=command,
        minimum_sequence_length=minimum_sequence_length,
        retained_sequence_count=len(retained_sequences),
        mmseqs_executable=mmseqs,
        mmseqs_version_value=version,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
        split_memory_limit=split_memory_limit,
        input_manifest_sha256=file_sha256(manifest),
    )
    mmseqs_cache_metadata_path(prefix).write_text(json.dumps(completed_cache_metadata(**metadata)))


def test_split_workflow_dedup_clusters_components_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The split workflow deduplicates, normalizes clusters, groups leakage links, and audits overlaps."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
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
    _write_cache_metadata(
        manifest=manifest,
        fasta=tmp_path / "retained.fasta",
        prefix=prefix,
        tmp_dir=tmp_path / "tmp",
        retained_sequences={"s1": "AAAA", "s3": "BBBB", "s4": "CCCC", "s5": "DDDD"},
        cluster_tsv="c1\ts1\nc1\ts3\ns4\ts4\nc5\ts5\n",
        minimum_sequence_length=1,
        threads=2,
        split_memory_limit="4G",
    )
    result = run_split_workflow(
        manifest_path=manifest,
        deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
        deduplication_report_path=tmp_path / "deduplication_report.json",
        fasta_path=tmp_path / "retained.fasta",
        mmseqs_output_prefix=prefix,
        mmseqs_tmp_dir=tmp_path / "tmp",
        cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
        output_dir=tmp_path / "splits",
        mmseqs_threads=2,
        mmseqs_cov_mode=0,
        mmseqs_split_memory_limit="4G",
        seed=42,
        fractions=(0.6, 0.2, 0.2),
        minimum_sequence_length=1,
        force=False,
    )
    assert result["original_count"] == 5
    assert result["retained_count"] == 4
    assert result["removed_count"] == 1
    retained = pd.read_parquet(tmp_path / "manifest_deduplicated.parquet")
    assert set(retained["sample_id"]) == {"s1", "s3", "s4", "s5"}
    clusters = pd.read_csv(tmp_path / "mmseqs_clusters.tsv", sep="\t", header=None, names=["cluster_id", "sample_id"])
    assert set(clusters["sample_id"]) == {"s1", "s3", "s4", "s5"}
    assert clusters.loc[clusters["sample_id"] == "s4", "cluster_id"].iloc[0] == "s4"
    split = pd.read_parquet(tmp_path / "splits" / "split_assignments.parquet")
    assert split.groupby("cluster_id")["split"].nunique().max() == 1
    assert split.groupby("pdb_id")["split"].nunique().max() == 1
    audit = json.loads((tmp_path / "splits" / "leakage_audit.json").read_text())
    assert all(check["overlap_count"] == 0 for check in audit["leakage_checks"].values())
    report = json.loads((tmp_path / "deduplication_report.json").read_text())
    assert sorted(report["duplicate_group_sizes"], reverse=True) == [2, 1, 1, 1]
    metadata = json.loads((tmp_path / "splits" / "workflow_metadata.json").read_text())
    assert "--threads" in metadata["mmseqs_command"]
    assert "2" in metadata["mmseqs_command"]
    assert "--split-memory-limit" in metadata["mmseqs_command"]
    assert "4G" in metadata["mmseqs_command"]
    cache_metadata = json.loads(mmseqs_cache_metadata_path(prefix).read_text())
    assert cache_metadata["completed"] is True
    assert cache_metadata["retained_sequence_count"] == 4


def test_split_workflow_filters_minimum_sequence_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Samples shorter than minimum_sequence_length are removed before deduplication and FASTA export."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "short",
                "pdb_id": "P1",
                "chain_id": "A",
                "sequence": "A" * 19,
                "length": 19,
                "path": "short.npz",
            },
            {
                "sample_id": "edge",
                "pdb_id": "P2",
                "chain_id": "A",
                "sequence": "C" * 20,
                "length": 20,
                "path": "edge.npz",
            },
            {
                "sample_id": "long",
                "pdb_id": "P3",
                "chain_id": "A",
                "sequence": "D" * 21,
                "length": 21,
                "path": "long.npz",
            },
        ]
    ).to_parquet(manifest, index=False)
    prefix = tmp_path / "mmseqs" / "clusters"
    _write_cache_metadata(
        manifest=manifest,
        fasta=tmp_path / "retained.fasta",
        prefix=prefix,
        tmp_dir=tmp_path / "tmp",
        retained_sequences={"edge": "C" * 20, "long": "D" * 21},
        cluster_tsv="c1\tedge\nc2\tlong\n",
        minimum_sequence_length=20,
    )

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
        fractions=(0.8, 0.1, 0.1),
        minimum_sequence_length=20,
        force=False,
    )

    assert result["original_count"] == 3
    assert result["length_filtered_count"] == 2
    assert result["length_rejected_count"] == 1
    assert result["retained_count"] == 2
    retained = pd.read_parquet(tmp_path / "manifest_deduplicated.parquet")
    assert set(retained["sample_id"]) == {"edge", "long"}
    fasta = (tmp_path / "retained.fasta").read_text()
    assert ">short" not in fasta
    metadata = json.loads((tmp_path / "splits" / "workflow_metadata.json").read_text())
    assert metadata["minimum_sequence_length"] == 20
    assert metadata["length_rejected_count"] == 1


def test_mmseqs_cache_metadata_records_filtered_fasta_and_command(tmp_path: Path) -> None:
    """New MMseqs cache metadata includes FASTA hash, command and minimum length."""
    fasta = tmp_path / "retained.fasta"
    fasta.write_text(">s1\n" + "A" * 20 + "\n")
    prefix = tmp_path / "mmseqs" / "clusters"
    prefix.parent.mkdir()
    command = build_mmseqs_easy_cluster_command(
        fasta,
        prefix,
        tmp_path / "tmp",
        threads=2,
        split_memory_limit="4G",
    )
    metadata = expected_mmseqs_cache_metadata(
        fasta_path=fasta,
        mmseqs_command=command,
        minimum_sequence_length=20,
    )
    assert metadata["fasta_sha256"] == file_sha256(fasta)
    assert metadata["mmseqs_command"] == command
    assert metadata["minimum_sequence_length"] == 20
    assert mmseqs_cache_metadata_path(prefix).name == "clusters_cluster.metadata.json"


def test_split_workflow_rejects_missing_cache_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-existing cluster TSV is not trusted without completed matching metadata."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20}]).to_parquet(
        manifest, index=False
    )
    prefix = tmp_path / "mmseqs" / "clusters"
    prefix.parent.mkdir()
    Path(f"{prefix}_cluster.tsv").write_text("c1\ts1\n")

    with pytest.raises(RuntimeError, match="metadata is missing"):
        run_split_workflow(
            manifest_path=manifest,
            deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
            deduplication_report_path=tmp_path / "deduplication_report.json",
            fasta_path=tmp_path / "retained.fasta",
            mmseqs_output_prefix=prefix,
            mmseqs_tmp_dir=tmp_path / "tmp",
            cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
            output_dir=tmp_path / "splits",
        )


def test_split_workflow_rejects_stale_cache_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing split inputs invalidates a cached MMseqs cluster TSV."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20}]).to_parquet(
        manifest, index=False
    )
    prefix = tmp_path / "mmseqs" / "clusters"
    _write_cache_metadata(
        manifest=manifest,
        fasta=tmp_path / "retained.fasta",
        prefix=prefix,
        tmp_dir=tmp_path / "tmp",
        retained_sequences={"s1": "A" * 20},
        cluster_tsv="c1\ts1\n",
        minimum_sequence_length=20,
        min_seq_id=0.30,
    )

    with pytest.raises(RuntimeError, match="metadata does not match"):
        run_split_workflow(
            manifest_path=manifest,
            deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
            deduplication_report_path=tmp_path / "deduplication_report.json",
            fasta_path=tmp_path / "retained.fasta",
            mmseqs_output_prefix=prefix,
            mmseqs_tmp_dir=tmp_path / "tmp",
            cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
            output_dir=tmp_path / "splits",
            sequence_identity_threshold=0.40,
        )


def test_split_workflow_force_recomputes_and_writes_completed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force bypasses stale cache metadata and writes completed metadata only after MMseqs output exists."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20}]).to_parquet(
        manifest, index=False
    )
    prefix = tmp_path / "mmseqs" / "clusters"
    prefix.parent.mkdir()

    def fake_run(command: list[str], *, log_path: str | Path | None = None) -> None:
        assert command[0:2] == ["mmseqs", "easy-cluster"]
        Path(f"{prefix}_cluster.tsv").write_text("c1\ts1\n")

    monkeypatch.setattr(splitting_module, "run_mmseqs_easy_cluster", fake_run)
    run_split_workflow(
        manifest_path=manifest,
        deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
        deduplication_report_path=tmp_path / "deduplication_report.json",
        fasta_path=tmp_path / "retained.fasta",
        mmseqs_output_prefix=prefix,
        mmseqs_tmp_dir=tmp_path / "tmp",
        cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
        output_dir=tmp_path / "splits",
        force=True,
    )

    metadata = json.loads(mmseqs_cache_metadata_path(prefix).read_text())
    assert metadata["completed"] is True
    assert metadata["mmseqs_version"] == "test-mmseqs"


def test_failed_mmseqs_does_not_write_completed_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed MMseqs runs leave no completed metadata behind."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20}]).to_parquet(
        manifest, index=False
    )
    prefix = tmp_path / "mmseqs" / "clusters"

    def fail_run(command: list[str], *, log_path: str | Path | None = None) -> None:
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(splitting_module, "run_mmseqs_easy_cluster", fail_run)
    with pytest.raises(subprocess.CalledProcessError):
        run_split_workflow(
            manifest_path=manifest,
            deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
            deduplication_report_path=tmp_path / "deduplication_report.json",
            fasta_path=tmp_path / "retained.fasta",
            mmseqs_output_prefix=prefix,
            mmseqs_tmp_dir=tmp_path / "tmp",
            cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
            output_dir=tmp_path / "splits",
            force=True,
        )
    assert not mmseqs_cache_metadata_path(prefix).exists()


def test_split_workflow_rejects_unknown_and_missing_member_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cluster TSV member IDs must exactly cover retained sample IDs."""
    monkeypatch.setattr(splitting_module, "mmseqs_version", lambda _: "test-mmseqs")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        [
            {"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20},
            {"sample_id": "s2", "pdb_id": "P2", "chain_id": "A", "sequence": "C" * 20, "length": 20},
        ]
    ).to_parquet(manifest, index=False)
    prefix = tmp_path / "mmseqs" / "clusters"
    _write_cache_metadata(
        manifest=manifest,
        fasta=tmp_path / "retained.fasta",
        prefix=prefix,
        tmp_dir=tmp_path / "tmp",
        retained_sequences={"s1": "A" * 20, "s2": "C" * 20},
        cluster_tsv="c1\ts1\nc9\tghost\n",
        minimum_sequence_length=20,
    )

    with pytest.raises(ValueError, match="unknown sample IDs"):
        run_split_workflow(
            manifest_path=manifest,
            deduplicated_manifest_path=tmp_path / "manifest_deduplicated.parquet",
            deduplication_report_path=tmp_path / "deduplication_report.json",
            fasta_path=tmp_path / "retained.fasta",
            mmseqs_output_prefix=prefix,
            mmseqs_tmp_dir=tmp_path / "tmp",
            cluster_assignments_path=tmp_path / "mmseqs_clusters.tsv",
            output_dir=tmp_path / "splits",
        )


def test_split_workflow_rejects_or_warns_for_unsupported_resume_args(tmp_path: Path) -> None:
    """Unsupported split workflow arguments fail or warn explicitly."""
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame([{"sample_id": "s1", "pdb_id": "P1", "chain_id": "A", "sequence": "A" * 20, "length": 20}]).to_parquet(
        manifest, index=False
    )
    kwargs = {
        "manifest_path": manifest,
        "deduplicated_manifest_path": tmp_path / "manifest_deduplicated.parquet",
        "deduplication_report_path": tmp_path / "deduplication_report.json",
        "fasta_path": tmp_path / "retained.fasta",
        "mmseqs_output_prefix": tmp_path / "mmseqs" / "clusters",
        "mmseqs_tmp_dir": tmp_path / "tmp",
        "cluster_assignments_path": tmp_path / "mmseqs_clusters.tsv",
        "output_dir": tmp_path / "splits",
        "dry_run": True,
    }

    with pytest.raises(NotImplementedError, match="external_group_file"):
        run_split_workflow(**kwargs, external_group_file=tmp_path / "groups.tsv")
    with pytest.warns(RuntimeWarning, match="not currently resumable"):
        run_split_workflow(**kwargs, state_db_path=tmp_path / "state.sqlite", checkpoint_every=10)


def test_mmseqs_command_uses_configured_resources(tmp_path: Path) -> None:
    """MMseqs2 receives configured threads, cov mode and split memory limit."""
    cmd = build_mmseqs_easy_cluster_command(
        tmp_path / "input.fasta",
        tmp_path / "out" / "clusters",
        tmp_path / "tmp",
        mmseqs="mmseqs",
        min_seq_id=0.3,
        coverage=0.8,
        cov_mode=0,
        threads=2,
        split_memory_limit="4G",
    )
    assert cmd[0:2] == ["mmseqs", "easy-cluster"]
    assert cmd[cmd.index("--threads") : cmd.index("--threads") + 2] == ["--threads", "2"]
    assert cmd[cmd.index("--cov-mode") : cmd.index("--cov-mode") + 2] == ["--cov-mode", "0"]
    memory_idx = cmd.index("--split-memory-limit")
    assert cmd[memory_idx : memory_idx + 2] == ["--split-memory-limit", "4G"]


def test_split_assignment_balances_total_counts_not_method_or_length_bins() -> None:
    """Split assignment keeps groups intact and optimizes only total sample count."""
    rows = [
        {
            "sample_id": f"x{idx}",
            "split_group_id": "big_xray_short",
            "experimental_method": "X-RAY DIFFRACTION",
            "length": 17,
        }
        for idx in range(8)
    ]
    rows.extend(
        [
            {
                "sample_id": "em_long_1",
                "split_group_id": "em_long_1",
                "experimental_method": "ELECTRON MICROSCOPY",
                "length": 500,
            },
            {
                "sample_id": "em_long_2",
                "split_group_id": "em_long_2",
                "experimental_method": "ELECTRON MICROSCOPY",
                "length": 500,
            },
        ]
    )
    split = assign_groups_to_splits(
        pd.DataFrame(rows),
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
        seed=1,
    )
    assert split.groupby("split_group_id")["split"].nunique().max() == 1
    assert sorted(split["split"].value_counts().tolist(), reverse=True) == [8, 1, 1]
