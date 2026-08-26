"""Leakage-safe split assignment workflow."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.data.clustering import (
    add_sequence_hashes,
    build_mmseqs_easy_cluster_command,
    completed_cache_metadata,
    deduplicate_sequences,
    deduplication_report,
    load_mmseqs_clusters,
    mmseqs_version,
    run_mmseqs_easy_cluster,
    write_fasta,
)
from protein_distance_diffusion.data.preprocess import load_manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mmseqs_cache_metadata_path(output_prefix: str | Path) -> Path:
    return Path(f"{output_prefix}_cluster.metadata.json")


def expected_mmseqs_cache_metadata(
    *,
    fasta_path: str | Path,
    mmseqs_command: list[str],
    minimum_sequence_length: int | None,
    retained_sequence_count: int | None = None,
    mmseqs_executable: str | None = None,
    mmseqs_version_value: str | None = None,
    min_seq_id: float | None = None,
    coverage: float | None = None,
    cov_mode: int | None = None,
    threads: int | None = None,
    split_memory_limit: str | None = None,
    input_manifest_sha256: str | None = None,
    retention_mode: str | None = "exact_sequence_representative",
    collapse_within_pdb_exact_duplicates: bool | None = False,
    exact_sequence_weight_exponent: float | None = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "retained_fasta_path": str(Path(fasta_path)),
        "fasta_sha256": file_sha256(fasta_path),
        "retained_sequence_count": retained_sequence_count,
        "mmseqs_executable": mmseqs_executable,
        "mmseqs_version": mmseqs_version_value,
        "mmseqs_command": [str(part) for part in mmseqs_command],
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "cov_mode": cov_mode,
        "threads": threads,
        "split_memory_limit": split_memory_limit,
        "minimum_sequence_length": minimum_sequence_length,
        "input_manifest_sha256": input_manifest_sha256,
        "retention_mode": retention_mode,
        "collapse_within_pdb_exact_duplicates": collapse_within_pdb_exact_duplicates,
        "exact_sequence_weight_exponent": exact_sequence_weight_exponent,
    }


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _with_split_groups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    ids = [str(value) for value in out["sample_id"]]
    uf = _UnionFind(ids)
    for column in ("cluster_id", "pdb_id", "sequence_hash"):
        if column not in out:
            continue
        for _, group in out.groupby(column, dropna=False):
            values = [str(value) for value in group["sample_id"]]
            for value in values[1:]:
                uf.union(values[0], value)
    out["split_group_id"] = [uf.find(str(value)) for value in out["sample_id"]]
    return out


def assign_groups_to_splits(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign whole ``split_group_id`` groups while balancing total counts."""
    total_fraction = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total_fraction, 1.0):
        raise ValueError("split fractions must sum to 1.0")
    out = frame.copy()
    if "split_group_id" not in out:
        out = _with_split_groups(out)
    groups = out.groupby("split_group_id").size().reset_index(name="size")
    groups["_rand"] = np.random.default_rng(seed).random(len(groups))
    groups = groups.sort_values(["size", "_rand"], ascending=[False, True])
    targets = {
        "train": train_fraction * len(out),
        "validation": validation_fraction * len(out),
        "test": test_fraction * len(out),
    }
    counts = {key: 0 for key in targets}
    assignments: dict[str, str] = {}
    for row in groups.itertuples(index=False):
        split = min(targets, key=lambda key: (counts[key] - targets[key], counts[key]))
        assignments[str(row.split_group_id)] = split
        counts[split] += int(row.size)
    out["split"] = out["split_group_id"].astype(str).map(assignments)
    return out


def assign_clusters_to_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Assign leakage-connected cluster/PDB/sequence groups to splits."""
    grouped = _with_split_groups(frame)
    return assign_groups_to_splits(
        grouped,
        train_fraction=fractions[0],
        validation_fraction=fractions[1],
        test_fraction=fractions[2],
        seed=seed,
    )


def _overlaps(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame:
        return []
    overlaps = []
    for value, group in frame.groupby(column, dropna=False):
        splits = sorted({str(item) for item in group["split"]})
        if len(splits) > 1:
            overlaps.append({"value": str(value), "splits": splits, "sample_ids": sorted(map(str, group["sample_id"]))})
    return overlaps


def leakage_audit(frame: pd.DataFrame) -> dict[str, Any]:
    checks = {}
    for column in ("cluster_id", "pdb_id", "sample_id", "sequence_hash"):
        values = _overlaps(frame, column)
        checks[column] = {"overlap_count": len(values), "overlaps": values[:50]}
    return {"leakage_checks": checks}


def assert_no_leakage(frame: pd.DataFrame) -> None:
    audit = leakage_audit(frame)
    offenders = {key: value for key, value in audit["leakage_checks"].items() if value["overlap_count"]}
    if offenders:
        raise ValueError(f"Split leakage detected: {offenders}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _write_split_manifests(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "split_assignments.parquet", index=False)
    for split in ("train", "validation", "test"):
        frame[frame["split"] == split].to_parquet(output_dir / f"{split}.parquet", index=False)


def _representatives_per_sequence(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    if "resolution_angstrom" in data:
        resolution = pd.to_numeric(data["resolution_angstrom"], errors="coerce")
    else:
        resolution = pd.Series(float("inf"), index=data.index)
    data["_resolution_sort"] = resolution.fillna(float("inf"))
    data = data.sort_values(["sequence_hash", "_resolution_sort", "sample_id"]).reset_index(drop=True)
    return data.groupby("sequence_hash", sort=False).head(1).drop(columns=["_resolution_sort"]).reset_index(drop=True)


def _write_all_structure_split_manifests(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "split_assignments.parquet", index=False)
    frame[frame["split"] == "train"].to_parquet(output_dir / "train.parquet", index=False)
    for split in ("validation", "test"):
        split_frame = frame[frame["split"] == split].reset_index(drop=True)
        split_frame.to_parquet(output_dir / f"{split}_all_structures.parquet", index=False)
        _representatives_per_sequence(split_frame).to_parquet(output_dir / f"{split}.parquet", index=False)


def _collapse_within_pdb_exact_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    if "resolution_angstrom" in data:
        resolution = pd.to_numeric(data["resolution_angstrom"], errors="coerce")
    else:
        resolution = pd.Series(float("inf"), index=data.index)
    data["_resolution_sort"] = resolution.fillna(float("inf"))
    data = data.sort_values(["pdb_id", "sequence_hash", "_resolution_sort", "sample_id"]).reset_index(drop=True)
    retained_rows = []
    duplicate_rows = []
    for (pdb_id, sequence_hash), group in data.groupby(["pdb_id", "sequence_hash"], sort=False):
        representative = group.iloc[0].drop(labels=["_resolution_sort"]).to_dict()
        retained_rows.append(representative)
        duplicate_rows.append(
            {
                "pdb_id": str(pdb_id),
                "sequence_hash": str(sequence_hash),
                "representative_sample_id": str(representative["sample_id"]),
                "duplicate_sample_ids": [str(value) for value in group["sample_id"].iloc[1:].tolist()],
                "group_size": int(len(group)),
            }
        )
    return pd.DataFrame(retained_rows), pd.DataFrame(duplicate_rows)


def _attach_sequence_weights(frame: pd.DataFrame, exponent: float) -> pd.DataFrame:
    out = frame.copy()
    counts = out.groupby("sequence_hash")["sample_id"].transform("count").astype(int)
    out["exact_sequence_count"] = counts
    out["sample_weight"] = counts.astype(float).pow(-float(exponent))
    return out


def _split_report(frame: pd.DataFrame, *, raw_count: int, retained_count: int) -> dict[str, Any]:
    cluster_sizes = frame.groupby("cluster_id").size() if "cluster_id" in frame else pd.Series(dtype=int)
    sequence_sizes = frame.groupby("sequence_hash").size() if "sequence_hash" in frame else pd.Series(dtype=int)
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        split_frame = frame[frame["split"] == split]
        split_counts[split] = {
            "samples": int(len(split_frame)),
            "unique_sequences": int(split_frame["sequence_hash"].nunique()) if "sequence_hash" in split_frame else 0,
        }
    train = frame[frame["split"] == "train"]
    return {
        "raw_sample_count": int(raw_count),
        "unique_exact_sequence_count": int(frame["sequence_hash"].nunique()) if "sequence_hash" in frame else 0,
        "retained_after_within_pdb_collapse_count": int(retained_count),
        "structures_per_exact_sequence": {
            "min": int(sequence_sizes.min()) if not sequence_sizes.empty else 0,
            "max": int(sequence_sizes.max()) if not sequence_sizes.empty else 0,
            "mean": float(sequence_sizes.mean()) if not sequence_sizes.empty else 0.0,
        },
        "mmseqs_cluster_count": int(frame["cluster_id"].nunique()) if "cluster_id" in frame else 0,
        "mmseqs_cluster_size_distribution": {
            "min": int(cluster_sizes.min()) if not cluster_sizes.empty else 0,
            "max": int(cluster_sizes.max()) if not cluster_sizes.empty else 0,
            "mean": float(cluster_sizes.mean()) if not cluster_sizes.empty else 0.0,
        },
        "splits": split_counts,
        "effective_total_training_weight": float(train.get("sample_weight", pd.Series(dtype=float)).sum()),
        **leakage_audit(frame),
    }


def build_splits_from_files(
    manifest_path: str | Path,
    cluster_assignments_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> pd.DataFrame:
    """Build split manifests from an existing manifest and cluster TSV."""
    frame = load_manifest(manifest_path)
    if "sequence_hash" not in frame:
        from protein_distance_diffusion.data.clustering import add_sequence_hashes

        frame = add_sequence_hashes(frame)
    clusters = load_mmseqs_clusters(cluster_assignments_path)
    merged = frame.merge(clusters, on="sample_id", how="left")
    merged["cluster_id"] = merged["cluster_id"].fillna(merged["sample_id"]).astype(str)
    split = assign_clusters_to_splits(merged, seed=seed, fractions=fractions)
    assert_no_leakage(split)
    out = Path(output_dir)
    _write_split_manifests(split, out)
    _atomic_json(out / "leakage_audit.json", leakage_audit(split))
    return split


def _load_cache_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError("MMseqs cluster cache metadata is missing; rerun with --force")
    metadata = json.loads(path.read_text())
    if metadata.get("completed") is not True:
        raise RuntimeError("MMseqs cluster cache metadata is incomplete; rerun with --force")
    return metadata


def _validate_cache_or_raise(raw_cluster: Path, metadata_path: Path, expected: dict[str, Any]) -> None:
    if not raw_cluster.exists():
        return
    existing = _load_cache_metadata(metadata_path)
    comparable = {key: value for key, value in existing.items() if key != "completed_utc"}
    expected_completed = completed_cache_metadata(**expected)
    expected_comparable = {key: value for key, value in expected_completed.items() if key != "completed_utc"}
    if comparable != expected_comparable:
        raise RuntimeError(
            "MMseqs cluster cache metadata does not match current inputs/configuration; rerun with --force"
        )


def _validate_cluster_ids(clusters: pd.DataFrame, retained: pd.DataFrame) -> None:
    retained_ids = set(retained["sample_id"].astype(str))
    cluster_ids = set(clusters["sample_id"].astype(str))
    unknown = sorted(cluster_ids - retained_ids)
    missing = sorted(retained_ids - cluster_ids)
    if unknown:
        raise ValueError(f"MMseqs cluster TSV contains unknown sample IDs: {', '.join(unknown[:5])}")
    if missing:
        raise ValueError(f"MMseqs cluster TSV is missing retained sample IDs: {', '.join(missing[:5])}")


def run_split_workflow(
    *,
    manifest_path: str | Path,
    deduplicated_manifest_path: str | Path,
    deduplication_report_path: str | Path,
    fasta_path: str | Path,
    mmseqs_output_prefix: str | Path,
    mmseqs_tmp_dir: str | Path,
    cluster_assignments_path: str | Path,
    output_dir: str | Path,
    mmseqs: str = "mmseqs",
    sequence_identity_threshold: float = 0.30,
    alignment_coverage_threshold: float = 0.80,
    mmseqs_threads: int | None = None,
    mmseqs_cov_mode: int = 0,
    mmseqs_split_memory_limit: str | None = None,
    mmseqs_remove_tmp_files: bool = False,
    mmseqs_log_path: str | Path | None = None,
    state_db_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    minimum_sequence_length: int | None = 20,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    external_group_file: str | Path | None = None,
    retention_mode: str = "exact_sequence_representative",
    collapse_within_pdb_exact_duplicates: bool = False,
    exact_sequence_weight_exponent: float = 1.0,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run deduplication, FASTA export, MMseqs clustering, split assignment and audit."""
    if retention_mode not in {"exact_sequence_representative", "all_structures_group_safe"}:
        raise ValueError("retention_mode must be one of: exact_sequence_representative, all_structures_group_safe")
    if external_group_file is not None:
        raise NotImplementedError("external_group_file grouping is not implemented for split workflow")
    if state_db_path is not None:
        warnings.warn(
            "state_db_path is ignored; split workflow is not currently resumable",
            RuntimeWarning,
            stacklevel=2,
        )
    if checkpoint_every is not None:
        warnings.warn(
            "checkpoint_every is ignored; split workflow is not currently resumable",
            RuntimeWarning,
            stacklevel=2,
        )
    frame = load_manifest(manifest_path)
    input_manifest_sha256 = file_sha256(manifest_path)
    original_count = len(frame)
    if minimum_sequence_length is not None:
        frame = frame[pd.to_numeric(frame["length"], errors="coerce") >= int(minimum_sequence_length)].copy()
    length_filtered_count = len(frame)
    if retention_mode == "exact_sequence_representative":
        retained, duplicates = deduplicate_sequences(frame)
        clustering_input = retained
    else:
        frame = add_sequence_hashes(frame)
        if collapse_within_pdb_exact_duplicates:
            retained, duplicates = _collapse_within_pdb_exact_duplicates(frame)
        else:
            retained = frame.sort_values("sample_id").reset_index(drop=True)
            duplicates = pd.DataFrame(
                [
                    {
                        "representative_sample_id": str(row.sample_id),
                        "sequence_hash": str(row.sequence_hash),
                        "duplicate_sample_ids": [],
                        "group_size": 1,
                    }
                    for row in retained.itertuples(index=False)
                ]
            )
        retained = _attach_sequence_weights(retained, exact_sequence_weight_exponent)
        clustering_input = _representatives_per_sequence(retained)
    if dry_run:
        return {
            "original_count": original_count,
            "length_filtered_count": length_filtered_count,
            "retained_count": len(retained),
            "unique_sequence_count": (
                int(clustering_input["sequence_hash"].nunique())
                if "sequence_hash" in clustering_input
                else len(clustering_input)
            ),
            "retention_mode": retention_mode,
        }
    Path(deduplicated_manifest_path).parent.mkdir(parents=True, exist_ok=True)
    retained.to_parquet(deduplicated_manifest_path, index=False)
    _atomic_json(
        Path(deduplication_report_path),
        deduplication_report(duplicates, original_count=length_filtered_count, retained_count=len(retained)),
    )
    write_fasta(clustering_input, fasta_path)
    command = build_mmseqs_easy_cluster_command(
        fasta_path,
        mmseqs_output_prefix,
        mmseqs_tmp_dir,
        mmseqs=mmseqs,
        min_seq_id=sequence_identity_threshold,
        coverage=alignment_coverage_threshold,
        cov_mode=mmseqs_cov_mode,
        threads=mmseqs_threads,
        split_memory_limit=mmseqs_split_memory_limit,
        remove_tmp_files=mmseqs_remove_tmp_files,
    )
    raw_cluster = Path(f"{mmseqs_output_prefix}_cluster.tsv")
    meta_path = mmseqs_cache_metadata_path(mmseqs_output_prefix)
    version = mmseqs_version(mmseqs)
    metadata = expected_mmseqs_cache_metadata(
        fasta_path=fasta_path,
        mmseqs_command=command,
        minimum_sequence_length=minimum_sequence_length,
        retained_sequence_count=len(clustering_input),
        mmseqs_executable=mmseqs,
        mmseqs_version_value=version,
        min_seq_id=sequence_identity_threshold,
        coverage=alignment_coverage_threshold,
        cov_mode=mmseqs_cov_mode,
        threads=mmseqs_threads,
        split_memory_limit=mmseqs_split_memory_limit,
        input_manifest_sha256=input_manifest_sha256,
        retention_mode=retention_mode,
        collapse_within_pdb_exact_duplicates=collapse_within_pdb_exact_duplicates,
        exact_sequence_weight_exponent=exact_sequence_weight_exponent,
    )
    if raw_cluster.exists() and not force:
        _validate_cache_or_raise(raw_cluster, meta_path, metadata)
    if force or not raw_cluster.exists():
        run_mmseqs_easy_cluster(command, log_path=mmseqs_log_path)
        if not raw_cluster.exists():
            raise RuntimeError("MMseqs completed but expected cluster TSV was not created")
        _atomic_json(meta_path, completed_cache_metadata(**metadata))
    clusters = (
        load_mmseqs_clusters(raw_cluster) if raw_cluster.exists() else pd.DataFrame(columns=["cluster_id", "sample_id"])
    )
    _validate_cluster_ids(clusters, clustering_input)
    if retention_mode == "exact_sequence_representative":
        clusters = retained[["sample_id"]].merge(clusters, on="sample_id", how="left")
        clusters["cluster_id"] = clusters["cluster_id"].fillna(clusters["sample_id"]).astype(str)
    else:
        unique_clusters = clustering_input[["sample_id", "sequence_hash"]].merge(clusters, on="sample_id", how="left")
        unique_clusters["cluster_id"] = unique_clusters["cluster_id"].fillna(unique_clusters["sample_id"]).astype(str)
        cluster_by_hash = unique_clusters.set_index("sequence_hash")["cluster_id"]
        clusters = retained[["sample_id", "sequence_hash"]].copy()
        clusters["cluster_id"] = clusters["sequence_hash"].map(cluster_by_hash).astype(str)
        clusters = clusters[["sample_id", "cluster_id"]]
    clusters = clusters[["cluster_id", "sample_id"]]
    Path(cluster_assignments_path).parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(cluster_assignments_path, sep="\t", index=False, header=False)
    if retention_mode == "exact_sequence_representative":
        split = build_splits_from_files(
            deduplicated_manifest_path, cluster_assignments_path, output_dir, seed=seed, fractions=fractions
        )
    else:
        merged = retained.merge(clusters, on="sample_id", how="left")
        split = assign_clusters_to_splits(merged, seed=seed, fractions=fractions)
        assert_no_leakage(split)
        out = Path(output_dir)
        _write_all_structure_split_manifests(split, out)
        _atomic_json(out / "leakage_audit.json", leakage_audit(split))
        _atomic_json(
            out / "split_report.json",
            _split_report(split, raw_count=original_count, retained_count=len(retained)),
        )
    split_sizes = {key: int(value) for key, value in split["split"].value_counts().sort_index().items()}
    workflow_metadata = {
        **metadata,
        "completed": True,
        "original_count": int(original_count),
        "length_filtered_count": int(length_filtered_count),
        "length_rejected_count": int(original_count - length_filtered_count),
        "retained_count": int(len(retained)),
        "unique_sequence_count": (
            int(clustering_input["sequence_hash"].nunique())
            if "sequence_hash" in clustering_input
            else int(len(clustering_input))
        ),
        "cluster_count": int(split["cluster_id"].nunique()),
        "retention_mode": retention_mode,
        "collapse_within_pdb_exact_duplicates": bool(collapse_within_pdb_exact_duplicates),
        "exact_sequence_weight_exponent": float(exact_sequence_weight_exponent),
        "split_sizes": split_sizes,
    }
    _atomic_json(Path(output_dir) / "workflow_metadata.json", workflow_metadata)
    return workflow_metadata | {"removed_count": int(length_filtered_count - len(retained))}
