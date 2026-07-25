#!/usr/bin/env python3
"""Summarize processed distance-map datasets with resumable matrix reads."""

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from protein_distance_diffusion.data.clustering import add_sequence_hashes
from protein_distance_diffusion.data.preprocess import load_manifest


@dataclass(frozen=True)
class SummaryOptions:
    """Runtime controls for resumable dataset summarization."""

    workers: int = 2
    resume: bool = True
    restart: bool = False
    checkpoint_every: int = 10_000
    show_progress: bool = True
    interrupt_after_completed: int | None = None


@dataclass(frozen=True)
class DistanceTask:
    """One processed sample to read."""

    sample_id: str
    path: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 Python 3.10 test env


def _atomic_write_text(path: str | Path, text: str) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    tmp.write_text(text)
    tmp.replace(dst)


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _maybe_load_json(path: str | Path | None) -> dict[str, Any]:
    """Load optional JSON metadata."""
    if path is None:
        return {}
    src = Path(path)
    return json.loads(src.read_text()) if src.exists() else {}


def _default_preprocessing_summary_path(manifest: str | Path) -> Path:
    """Return the conventional preprocessing summary path next to a manifest."""
    return Path(manifest).parent / "preprocess_summary.json"


def _resolution_summary(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize resolution values with clear unavailable numeric fields."""
    empty = {
        "count": 0,
        "mean": None,
        "std": None,
        "min": None,
        "25%": None,
        "50%": None,
        "75%": None,
        "max": None,
    }
    if "resolution_angstrom" not in frame:
        return empty
    values = pd.to_numeric(frame["resolution_angstrom"], errors="coerce").dropna()
    if values.empty:
        return empty
    desc = values.describe()
    return {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not pd.isna(desc["std"]) else None,
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _settings_hash(*, contact_thresholds: list[float], exclude_near_diagonal: int) -> str:
    payload = json.dumps(
        {
            "contact_thresholds": [float(value) for value in contact_thresholds],
            "exclude_near_diagonal": int(exclude_near_diagonal),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_hash(frame: pd.DataFrame) -> str:
    columns = [column for column in ["sample_id", "path", "sequence", "length"] if column in frame]
    stable = frame[columns].astype(str).sort_values(columns).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(stable, index=False).values.tobytes())
    return digest.hexdigest()


def manifest_only_summary(
    frame: pd.DataFrame,
    *,
    split_manifest: str | Path | None = None,
    cluster_assignments: str | Path | None = None,
    preprocessing_summary: str | Path | None = None,
) -> dict[str, Any]:
    """Compute statistics that do not require opening sample `.npz` files."""
    frame = add_sequence_hashes(frame)
    split_counts: dict[str, int] = {}
    if split_manifest:
        split_frame = load_manifest(split_manifest)
        if "split" in split_frame:
            split_counts = {str(k): int(v) for k, v in split_frame["split"].value_counts().to_dict().items()}
    clusters = None
    if cluster_assignments:
        clusters = pd.read_csv(cluster_assignments, sep="\t")
    aa_counts: Counter[str] = Counter()
    for sequence in frame["sequence"].astype(str):
        aa_counts.update(sequence)
    missing_metadata = {
        column: int(frame[column].isna().sum())
        for column in ["experimental_method", "resolution_angstrom"]
        if column in frame
    }
    return {
        "accepted_chains": int(len(frame)),
        "unique_sequences": int(frame["sequence_hash"].nunique()),
        "mmseqs2_clusters": int(clusters.iloc[:, 0].nunique()) if clusters is not None else None,
        "split_counts": split_counts,
        "length_distribution": {
            "min": int(frame["length"].min()) if len(frame) else 0,
            "max": int(frame["length"].max()) if len(frame) else 0,
            "mean": float(frame["length"].mean()) if len(frame) else 0.0,
        },
        "experimental_method_distribution": frame.get("experimental_method", pd.Series(dtype=str))
        .value_counts()
        .to_dict()
        if "experimental_method" in frame
        else {},
        "resolution_distribution": _resolution_summary(frame),
        "amino_acid_frequency": dict(sorted(aa_counts.items())),
        "missing_metadata_counts": missing_metadata,
        "preprocessing_summary": _maybe_load_json(preprocessing_summary),
    }


def _valid_upper_mask(matrix_shape: tuple[int, int], *, exclude_near_diagonal: int) -> np.ndarray:
    idx = np.arange(matrix_shape[0])
    valid = np.triu(np.ones(matrix_shape, dtype=bool), k=1)
    if exclude_near_diagonal > 0:
        valid &= np.abs(idx[:, None] - idx[None, :]) > exclude_near_diagonal
    return valid


def distance_worker(task: DistanceTask, thresholds: list[float], exclude_near_diagonal: int) -> dict[str, Any]:
    """Read one sample and return compact distance/contact aggregates."""
    try:
        data = np.load(task.path, allow_pickle=False)
        matrix = data["distance_matrix"].astype(np.float64)
        valid = _valid_upper_mask(matrix.shape, exclude_near_diagonal=exclude_near_diagonal)
        values = matrix[valid]
        if values.size == 0:
            distance_count = 0
            distance_sum = 0.0
            distance_sumsq = 0.0
            distance_min = math.nan
            distance_max = math.nan
        else:
            distance_count = int(values.size)
            distance_sum = float(values.sum())
            distance_sumsq = float(np.square(values).sum())
            distance_min = float(values.min())
            distance_max = float(values.max())
        contact_counts = {str(threshold): int((values < threshold).sum()) for threshold in thresholds}
        return {
            "sample_id": task.sample_id,
            "status": "completed",
            "distance_count": distance_count,
            "distance_sum": distance_sum,
            "distance_sumsq": distance_sumsq,
            "distance_min": distance_min,
            "distance_max": distance_max,
            "contact_counts": contact_counts,
            "error_message": "",
        }
    except Exception as exc:
        return {
            "sample_id": task.sample_id,
            "status": "failed",
            "distance_count": 0,
            "distance_sum": 0.0,
            "distance_sumsq": 0.0,
            "distance_min": math.nan,
            "distance_max": math.nan,
            "contact_counts": {str(threshold): 0 for threshold in thresholds},
            "error_message": str(exc),
        }


def init_db(conn: sqlite3.Connection) -> None:
    """Create resumable summary tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sample_aggregates (
            sample_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            status TEXT NOT NULL,
            manifest_hash TEXT NOT NULL,
            settings_hash TEXT NOT NULL,
            distance_count INTEGER NOT NULL,
            distance_sum REAL NOT NULL,
            distance_sumsq REAL NOT NULL,
            distance_min REAL,
            distance_max REAL,
            contact_counts_json TEXT NOT NULL,
            error_message TEXT NOT NULL DEFAULT '',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            updated_utc TEXT NOT NULL
        );
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear summary state."""
    conn.executescript("DELETE FROM sample_aggregates; DELETE FROM run_metadata;")
    conn.commit()


def validate_or_set_hashes(conn: sqlite3.Connection, *, manifest_hash: str, settings_hash: str, restart: bool) -> None:
    """Validate resumable state against manifest and settings hashes."""
    saved_manifest = conn.execute("SELECT value FROM run_metadata WHERE key='manifest_hash'").fetchone()
    saved_settings = conn.execute("SELECT value FROM run_metadata WHERE key='settings_hash'").fetchone()
    if not restart and saved_manifest is not None and saved_manifest[0] != manifest_hash:
        raise ValueError(
            "Refusing to resume summary because the manifest hash differs. Use --restart to discard state."
        )
    if not restart and saved_settings is not None and saved_settings[0] != settings_hash:
        raise ValueError("Refusing to resume summary because the summary settings hash differs. Use --restart.")
    conn.execute("INSERT OR REPLACE INTO run_metadata(key, value) VALUES('manifest_hash', ?)", (manifest_hash,))
    conn.execute("INSERT OR REPLACE INTO run_metadata(key, value) VALUES('settings_hash', ?)", (settings_hash,))
    conn.commit()


def record_aggregate(
    conn: sqlite3.Connection,
    result: dict[str, Any],
    *,
    path: str,
    manifest_hash: str,
    settings_hash: str,
) -> None:
    """Persist one compact per-sample aggregate."""
    row = conn.execute(
        "SELECT attempt_count FROM sample_aggregates WHERE sample_id=?",
        (result["sample_id"],),
    ).fetchone()
    attempt_count = int(row[0]) + 1 if row is not None else 1
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sample_aggregates(
                sample_id, path, status, manifest_hash, settings_hash, distance_count,
                distance_sum, distance_sumsq, distance_min, distance_max,
                contact_counts_json, error_message, attempt_count, updated_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result["sample_id"],
                path,
                result["status"],
                manifest_hash,
                settings_hash,
                int(result["distance_count"]),
                float(result["distance_sum"]),
                float(result["distance_sumsq"]),
                None if math.isnan(float(result["distance_min"])) else float(result["distance_min"]),
                None if math.isnan(float(result["distance_max"])) else float(result["distance_max"]),
                json.dumps(result["contact_counts"], sort_keys=True),
                result["error_message"],
                attempt_count,
                _utc_now(),
            ),
        )


def pending_tasks(conn: sqlite3.Connection, frame: pd.DataFrame, *, resume: bool) -> list[DistanceTask]:
    """Return samples whose distance aggregates still need reading."""
    tasks = []
    for row in frame[["sample_id", "path"]].itertuples(index=False):
        sample_id = str(row.sample_id)
        if resume:
            saved = conn.execute(
                "SELECT status FROM sample_aggregates WHERE sample_id=?",
                (sample_id,),
            ).fetchone()
            if saved is not None and saved[0] == "completed":
                continue
        tasks.append(DistanceTask(sample_id=sample_id, path=str(row.path)))
    return tasks


def completed_aggregate_count(conn: sqlite3.Connection) -> int:
    """Count completed per-sample distance aggregates already in state."""
    return int(conn.execute("SELECT COUNT(*) FROM sample_aggregates WHERE status='completed'").fetchone()[0])


def aggregate_distance_summary(conn: sqlite3.Connection, thresholds: list[float]) -> dict[str, Any]:
    """Aggregate compact per-sample rows into dataset-level distance/contact stats."""
    rows = conn.execute(
        """
        SELECT status, distance_count, distance_sum, distance_sumsq, distance_min,
               distance_max, contact_counts_json
        FROM sample_aggregates
        ORDER BY sample_id
        """
    ).fetchall()
    total_count = 0
    total_sum = 0.0
    total_sumsq = 0.0
    mins = []
    maxes = []
    contact_counts = {str(threshold): 0 for threshold in thresholds}
    missing_or_corrupt = 0
    completed = 0
    for status, count, sum_value, sumsq, min_value, max_value, contact_json in rows:
        if status != "completed":
            missing_or_corrupt += 1
            continue
        completed += 1
        count = int(count)
        total_count += count
        total_sum += float(sum_value)
        total_sumsq += float(sumsq)
        if min_value is not None:
            mins.append(float(min_value))
        if max_value is not None:
            maxes.append(float(max_value))
        sample_contacts = json.loads(contact_json)
        for threshold in contact_counts:
            contact_counts[threshold] += int(sample_contacts.get(threshold, 0))
    if total_count == 0:
        distance_distribution = {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        contact_fraction = {threshold: None for threshold in contact_counts}
    else:
        mean = total_sum / total_count
        variance = max(total_sumsq / total_count - mean * mean, 0.0)
        distance_distribution = {
            "count": total_count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(mins) if mins else None,
            "max": max(maxes) if maxes else None,
        }
        contact_fraction = {
            threshold: float(count / total_count) for threshold, count in sorted(contact_counts.items())
        }
    return {
        "distance_distribution": distance_distribution,
        "contact_fraction_mean": contact_fraction,
        "distance_samples_completed": completed,
        "missing_or_corrupt_samples": missing_or_corrupt,
    }


def write_summary_snapshot(
    conn: sqlite3.Connection,
    *,
    manifest_summary: dict[str, Any],
    thresholds: list[float],
    output_dir: str | Path,
    partial: bool,
) -> Path:
    """Write an atomic full or partial JSON summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("dataset_summary.partial.json" if partial else "dataset_summary.json")
    summary = dict(manifest_summary)
    summary.update(aggregate_distance_summary(conn, thresholds))
    summary["partial"] = partial
    summary["updated_utc"] = _utc_now()
    _atomic_write_json(path, summary)
    return path


def run_distance_phase(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    *,
    manifest_hash: str,
    settings_hash: str,
    thresholds: list[float],
    exclude_near_diagonal: int,
    output_dir: str | Path,
    manifest_summary: dict[str, Any],
    options: SummaryOptions,
) -> bool:
    """Run bounded parallel distance/contact aggregation."""
    tasks = pending_tasks(conn, frame, resume=options.resume)
    task_paths = {task.sample_id: task.path for task in tasks}
    already_completed = completed_aggregate_count(conn) if options.resume else 0
    completed = 0
    missing_or_corrupt = 0
    max_in_flight = max(1, options.workers * 2)
    next_index = 0
    bar = tqdm(
        total=len(frame),
        initial=already_completed,
        desc="reading distance matrices",
        unit="sample",
        dynamic_ncols=True,
        disable=not options.show_progress,
    )
    interrupted = False
    try:
        if options.workers == 1:
            for task in tasks:
                result = distance_worker(task, thresholds, exclude_near_diagonal)
                record_aggregate(
                    conn,
                    result,
                    path=task.path,
                    manifest_hash=manifest_hash,
                    settings_hash=settings_hash,
                )
                completed += 1
                missing_or_corrupt += 1 if result["status"] != "completed" else 0
                bar.update(1)
                bar.set_postfix({"missing/corrupt": missing_or_corrupt})
                if completed % options.checkpoint_every == 0:
                    write_summary_snapshot(
                        conn,
                        manifest_summary=manifest_summary,
                        thresholds=thresholds,
                        output_dir=output_dir,
                        partial=True,
                    )
                if options.interrupt_after_completed is not None and completed >= options.interrupt_after_completed:
                    raise KeyboardInterrupt
        else:
            executor = ProcessPoolExecutor(max_workers=options.workers)
            in_flight: set[Future] = set()
            try:
                while next_index < len(tasks) or in_flight:
                    while next_index < len(tasks) and len(in_flight) < max_in_flight:
                        task = tasks[next_index]
                        in_flight.add(executor.submit(distance_worker, task, thresholds, exclude_near_diagonal))
                        next_index += 1
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        record_aggregate(
                            conn,
                            result,
                            path=task_paths[str(result["sample_id"])],
                            manifest_hash=manifest_hash,
                            settings_hash=settings_hash,
                        )
                        completed += 1
                        missing_or_corrupt += 1 if result["status"] != "completed" else 0
                        bar.update(1)
                        bar.set_postfix({"missing/corrupt": missing_or_corrupt})
                        if completed % options.checkpoint_every == 0:
                            write_summary_snapshot(
                                conn,
                                manifest_summary=manifest_summary,
                                thresholds=thresholds,
                                output_dir=output_dir,
                                partial=True,
                            )
                        if (
                            options.interrupt_after_completed is not None
                            and completed >= options.interrupt_after_completed
                        ):
                            raise KeyboardInterrupt
            except KeyboardInterrupt:
                for future in in_flight:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
        write_summary_snapshot(
            conn,
            manifest_summary=manifest_summary,
            thresholds=thresholds,
            output_dir=output_dir,
            partial=True,
        )
        print(f"Interrupted after summarizing {completed} sample(s).", file=sys.stderr)
        print(
            "Resume with: "
            f"python scripts/summarize_dataset.py --manifest {frame.attrs['manifest_path']} "
            f"--output-dir {output_dir} --workers {options.workers} --resume",
            file=sys.stderr,
        )
    finally:
        bar.close()
    if interrupted:
        return False
    return True


def summarize_manifest(
    manifest: str | Path,
    *,
    split_manifest: str | Path | None = None,
    cluster_assignments: str | Path | None = None,
    preprocessing_summary: str | Path | None = None,
    contact_thresholds: list[float] | None = None,
    exclude_near_diagonal: int = 0,
    output_dir: str | Path | None = None,
    workers: int = 2,
    resume: bool = True,
    restart: bool = False,
    checkpoint_every: int = 10_000,
    show_progress: bool = False,
    interrupt_after_completed: int | None = None,
) -> dict[str, Any]:
    """Build machine-readable dataset summary statistics."""
    thresholds = contact_thresholds or [6.0, 8.0, 10.0]
    if workers <= 0:
        raise ValueError("--workers must be positive")
    if checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    with tqdm(total=1, desc="loading manifest", disable=not show_progress, dynamic_ncols=True) as bar:
        frame = load_manifest(manifest)
        frame.attrs["manifest_path"] = str(manifest)
        bar.update(1)
    summary_path = preprocessing_summary or _default_preprocessing_summary_path(manifest)
    with tqdm(total=1, desc="manifest statistics", disable=not show_progress, dynamic_ncols=True) as bar:
        manifest_summary = manifest_only_summary(
            frame,
            split_manifest=split_manifest,
            cluster_assignments=cluster_assignments,
            preprocessing_summary=summary_path,
        )
        bar.update(1)
    out = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="proteinGen_summary_"))
    out.mkdir(parents=True, exist_ok=True)
    state_db = out / "dataset_summary_state.sqlite"
    manifest_digest = _manifest_hash(frame)
    settings_digest = _settings_hash(contact_thresholds=thresholds, exclude_near_diagonal=exclude_near_diagonal)
    conn = sqlite3.connect(state_db)
    try:
        init_db(conn)
        if restart:
            reset_db(conn)
        validate_or_set_hashes(conn, manifest_hash=manifest_digest, settings_hash=settings_digest, restart=restart)
        completed_all = run_distance_phase(
            conn,
            frame,
            manifest_hash=manifest_digest,
            settings_hash=settings_digest,
            thresholds=thresholds,
            exclude_near_diagonal=exclude_near_diagonal,
            output_dir=out,
            manifest_summary=manifest_summary,
            options=SummaryOptions(
                workers=workers,
                resume=resume,
                restart=restart,
                checkpoint_every=checkpoint_every,
                show_progress=show_progress,
                interrupt_after_completed=interrupt_after_completed,
            ),
        )
        if not completed_all:
            partial_summary = dict(manifest_summary)
            partial_summary.update(aggregate_distance_summary(conn, thresholds))
            partial_summary["partial"] = True
            return partial_summary
        with tqdm(total=1, desc="aggregating results", disable=not show_progress, dynamic_ncols=True) as bar:
            summary = dict(manifest_summary)
            summary.update(aggregate_distance_summary(conn, thresholds))
            summary["partial"] = False
            summary["state_db_path"] = str(state_db)
            summary["updated_utc"] = _utc_now()
            bar.update(1)
        with tqdm(total=1, desc="writing outputs", disable=not show_progress, dynamic_ncols=True) as bar:
            _atomic_write_json(out / "dataset_summary.json", summary)
            bar.update(1)
        return summary
    finally:
        conn.close()


def write_plots(manifest: str | Path, output_dir: str | Path) -> None:
    """Write human-readable length and resolution plots."""
    frame = load_manifest(manifest)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(frame["length"], bins=30, color="#1b9e77", edgecolor="white")
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Accepted chains")
    fig.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=140)
    plt.close(fig)
    if "resolution_angstrom" in frame and frame["resolution_angstrom"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(frame["resolution_angstrom"].dropna(), bins=30, color="#7570b3", edgecolor="white")
        ax.set_xlabel("Resolution (A)")
        ax.set_ylabel("Accepted chains")
        fig.tight_layout()
        fig.savefig(out / "resolution_distribution.png", dpi=140)
        plt.close(fig)


def main() -> None:
    """Run dataset summary generation."""
    parser = argparse.ArgumentParser(description="Summarize processed protein distance-map samples.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--cluster-assignments", type=Path, default=None)
    parser.add_argument("--preprocessing-summary", type=Path, default=None)
    parser.add_argument("--contact-thresholds", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    parser.add_argument("--exclude-near-diagonal", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the summary state database. Enabled by default.",
    )
    parser.add_argument("--restart", action="store_true", help="Discard previous summary state.")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    args = parser.parse_args()
    start = time.time()
    summary = summarize_manifest(
        args.manifest,
        split_manifest=args.split_manifest,
        cluster_assignments=args.cluster_assignments,
        preprocessing_summary=args.preprocessing_summary,
        contact_thresholds=args.contact_thresholds,
        exclude_near_diagonal=args.exclude_near_diagonal,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
        restart=args.restart,
        checkpoint_every=args.checkpoint_every,
        show_progress=True,
    )
    if summary.get("partial"):
        raise SystemExit(130)
    write_plots(args.manifest, args.output_dir)
    print(f"Wrote dataset summary to {args.output_dir} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
