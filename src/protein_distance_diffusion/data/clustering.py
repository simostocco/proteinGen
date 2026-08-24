"""Sequence deduplication and MMseqs2 integration helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def sequence_hash(sequence: str) -> str:
    """Hash an exact amino-acid sequence."""
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def add_sequence_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a ``sequence_hash`` column."""
    out = frame.copy()
    out["sequence_hash"] = out["sequence"].astype(str).map(sequence_hash)
    return out


def deduplicate_sequences(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one representative per exact sequence using resolution then stable IDs."""
    data = add_sequence_hashes(frame)
    sort_cols = ["sequence_hash"]
    if "resolution_angstrom" in data:
        resolution = pd.to_numeric(data["resolution_angstrom"], errors="coerce")
    else:
        resolution = pd.Series(float("inf"), index=data.index)
    data["_resolution_sort"] = resolution.fillna(float("inf"))
    data = data.sort_values(sort_cols + ["_resolution_sort", "sample_id"]).reset_index(drop=True)
    retained_rows = []
    duplicate_rows = []
    for _, group in data.groupby("sequence_hash", sort=False):
        rep = group.iloc[0].drop(labels=["_resolution_sort"]).to_dict()
        retained_rows.append(rep)
        duplicate_rows.append(
            {
                "representative_sample_id": rep["sample_id"],
                "sequence_hash": rep["sequence_hash"],
                "duplicate_sample_ids": [str(value) for value in group["sample_id"].iloc[1:].tolist()],
                "group_size": int(len(group)),
            }
        )
    return pd.DataFrame(retained_rows), pd.DataFrame(duplicate_rows)


def write_fasta(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write sample sequences to FASTA."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in frame.sort_values("sample_id").itertuples(index=False):
        lines.extend([f">{row.sample_id}", str(row.sequence)])
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))
    return dst


def build_mmseqs_easy_cluster_command(
    fasta_path: str | Path,
    output_prefix: str | Path,
    tmp_dir: str | Path,
    *,
    mmseqs: str = "mmseqs",
    min_seq_id: float = 0.30,
    coverage: float = 0.80,
    cov_mode: int = 0,
    threads: int | None = None,
    split_memory_limit: str | None = None,
    remove_tmp_files: bool = False,
) -> list[str]:
    """Build an ``mmseqs easy-cluster`` command."""
    cmd = [
        mmseqs,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
    ]
    if threads is not None:
        cmd.extend(["--threads", str(int(threads))])
    if split_memory_limit is not None:
        cmd.extend(["--split-memory-limit", str(split_memory_limit)])
    if remove_tmp_files:
        cmd.extend(["--remove-tmp-files", "1"])
    return cmd


def run_mmseqs_easy_cluster(command: list[str], *, log_path: str | Path | None = None) -> None:
    """Run MMseqs2 and optionally capture stdout/stderr."""
    if log_path is None:
        subprocess.run(command, check=True)
        return
    dst = Path(log_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def load_mmseqs_clusters(path: str | Path) -> pd.DataFrame:
    """Load MMseqs cluster TSV as ``cluster_id,sample_id``."""
    frame = pd.read_csv(path, sep="\t", header=None, names=["cluster_id", "sample_id"], usecols=[0, 1])
    return frame.astype({"cluster_id": str, "sample_id": str})


def deduplication_report(duplicates: pd.DataFrame, *, original_count: int, retained_count: int) -> dict[str, Any]:
    """Return a JSON-serializable exact-deduplication report."""
    return {
        "original_count": int(original_count),
        "retained_count": int(retained_count),
        "removed_count": int(original_count - retained_count),
        "duplicate_group_sizes": [int(value) for value in duplicates.get("group_size", [])],
        "duplicates": duplicates.to_dict(orient="records"),
    }
