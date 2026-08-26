#!/usr/bin/env python3
"""Preprocess local structure files into distance-map samples with resumable state."""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.constants import DEFAULT_RESIDUE_MAPPINGS
from protein_distance_diffusion.data.preprocess import save_processed_sample, write_manifest
from protein_distance_diffusion.data.structure_parser import (
    iter_supported_structure_files,
    parse_structure_file_with_rejections,
)

DETERMINISTIC_REJECTION_STATUS = "rejected"
SUCCESS_STATUS = "completed"
FAILURE_STATUS = "failed"


@dataclass(frozen=True)
class PreprocessOptions:
    """Runtime preprocessing controls."""

    workers: int
    resume: bool
    restart: bool
    retry_failures: bool
    checkpoint_every: int
    show_progress: bool = True
    interrupt_after_completed: int | None = None


@dataclass(frozen=True)
class WorkerConfig:
    """Pickleable worker preprocessing config."""

    backend: str
    samples_dir: str
    min_length: int | None
    max_length: int
    chain_id: str | None
    residue_mappings: dict[str, str]
    allowed_methods: list[str] | None
    max_xray_resolution_angstrom: float | None
    max_cryoem_resolution_angstrom: float | None
    missing_calpha_policy: str
    max_terminal_trim_fraction: float | None


@dataclass(frozen=True)
class SourceRecord:
    """One source file and its immutable-ish file metadata."""

    path: str
    size: int
    mtime_ns: int


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


def _parse_optional_min_length(value: object) -> int | None:
    """Parse `min_length` from YAML, where null disables the lower bound."""
    if value is None:
        return None
    min_length = int(value)
    if min_length <= 0:
        raise ValueError("min_length must be a positive integer or null")
    return min_length


def _parse_optional_float(value: object) -> float | None:
    """Parse an optional float threshold from YAML."""
    return None if value is None else float(value)


def _parse_optional_fraction(value: object, *, name: str) -> float | None:
    """Parse an optional fraction constrained to [0, 1]."""
    if value is None:
        return None
    fraction = float(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive or null")
    return fraction


def _parse_max_length(value: object) -> int:
    """Parse required positive maximum sequence length from YAML."""
    max_length = int(value)
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    return max_length


def _parse_missing_calpha_policy(value: object) -> str:
    policy = str(value or "reject")
    if policy not in {"reject", "trim_terminal"}:
        raise ValueError("missing_calpha_policy must be one of: reject, trim_terminal")
    return policy


def normalized_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the behavior-affecting preprocessing config in canonical form."""
    return {
        "source_dir": str(Path(cfg["source_dir"])),
        "samples_dir": str(Path(cfg["samples_dir"])),
        "backend": cfg.get("backend", "auto"),
        "chain_id": cfg.get("chain_id"),
        "min_length": _parse_optional_min_length(cfg.get("min_length", 40)),
        "max_length": _parse_max_length(cfg["max_length"]),
        "allowed_methods": cfg.get("allowed_methods"),
        "max_xray_resolution_angstrom": _parse_optional_float(
            cfg.get("max_xray_resolution_angstrom", cfg.get("xray_resolution_max"))
        ),
        "max_cryoem_resolution_angstrom": _parse_optional_float(
            cfg.get("max_cryoem_resolution_angstrom", cfg.get("em_resolution_max"))
        ),
        "missing_calpha_policy": _parse_missing_calpha_policy(cfg.get("missing_calpha_policy", "reject")),
        "max_terminal_trim_fraction": _parse_optional_fraction(
            cfg.get("max_terminal_trim_fraction"),
            name="max_terminal_trim_fraction",
        ),
        "residue_mappings": {**DEFAULT_RESIDUE_MAPPINGS, **cfg.get("residue_mappings", {})},
    }


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash the behavior-affecting preprocessing configuration."""
    payload = json.dumps(normalized_config(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def worker_config_from_cfg(cfg: dict[str, Any]) -> WorkerConfig:
    """Build the compact worker configuration."""
    normalized = normalized_config(cfg)
    return WorkerConfig(
        backend=str(normalized["backend"]),
        samples_dir=str(normalized["samples_dir"]),
        min_length=normalized["min_length"],  # type: ignore[arg-type]
        max_length=int(normalized["max_length"]),
        chain_id=normalized["chain_id"],  # type: ignore[arg-type]
        residue_mappings=normalized["residue_mappings"],  # type: ignore[arg-type]
        allowed_methods=normalized["allowed_methods"],  # type: ignore[arg-type]
        max_xray_resolution_angstrom=normalized["max_xray_resolution_angstrom"],  # type: ignore[arg-type]
        max_cryoem_resolution_angstrom=normalized["max_cryoem_resolution_angstrom"],  # type: ignore[arg-type]
        missing_calpha_policy=str(normalized["missing_calpha_policy"]),
        max_terminal_trim_fraction=normalized["max_terminal_trim_fraction"],  # type: ignore[arg-type]
    )


def default_state_db_path(cfg: dict[str, Any]) -> Path:
    """Return the default SQLite state DB path next to the manifest."""
    return Path(cfg.get("state_db_path") or Path(cfg["manifest_path"]).with_suffix(".preprocess_state.sqlite"))


def source_record(path: Path) -> SourceRecord:
    """Build a source record from the filesystem."""
    stat = path.stat()
    return SourceRecord(path=str(path), size=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


def init_db(conn: sqlite3.Connection) -> None:
    """Create preprocessing state tables."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_files (
            source_path TEXT PRIMARY KEY,
            source_size INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            status TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            accepted_chain_count INTEGER NOT NULL DEFAULT 0,
            rejection_info TEXT NOT NULL DEFAULT '[]',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            first_seen_utc TEXT NOT NULL,
            started_utc TEXT,
            finished_utc TEXT,
            error_message TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS manifest_rows (
            sample_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            row_json TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            FOREIGN KEY(source_path) REFERENCES source_files(source_path)
        );
        CREATE INDEX IF NOT EXISTS idx_manifest_rows_source_path ON manifest_rows(source_path);
        """
    )
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    """Clear all resumable preprocessing state."""
    conn.executescript(
        """
        DELETE FROM manifest_rows;
        DELETE FROM source_files;
        DELETE FROM run_metadata;
        """
    )
    conn.commit()


def validate_or_set_config_hash(conn: sqlite3.Connection, current_hash: str, *, restart: bool) -> None:
    """Ensure a resume uses the same preprocessing config hash."""
    row = conn.execute("SELECT value FROM run_metadata WHERE key='config_hash'").fetchone()
    if row is not None and row[0] != current_hash and not restart:
        raise ValueError(
            "Refusing to resume preprocessing because the config hash differs. "
            "Use --restart to discard the previous state."
        )
    conn.execute(
        "INSERT OR REPLACE INTO run_metadata(key, value) VALUES('config_hash', ?)",
        (current_hash,),
    )
    conn.commit()


def classify_pending_sources(
    conn: sqlite3.Connection,
    sources: list[SourceRecord],
    *,
    current_hash: str,
    resume: bool,
    retry_failures: bool,
) -> list[SourceRecord]:
    """Return sources that should be processed for this run."""
    if not resume:
        return sources
    pending = []
    for record in sources:
        row = conn.execute(
            """
            SELECT status, source_size, source_mtime_ns, config_hash
            FROM source_files
            WHERE source_path=?
            """,
            (record.path,),
        ).fetchone()
        if row is None:
            pending.append(record)
            continue
        status, size, mtime_ns, saved_hash = row
        same_source = int(size) == record.size and int(mtime_ns) == record.mtime_ns and saved_hash == current_hash
        if not same_source:
            pending.append(record)
        elif status == SUCCESS_STATUS:
            continue
        elif status == DETERMINISTIC_REJECTION_STATUS:
            continue
        elif status == FAILURE_STATUS and retry_failures:
            pending.append(record)
        elif status == FAILURE_STATUS:
            continue
        else:
            pending.append(record)
    return pending


def worker_process_source(record: SourceRecord, worker_cfg: WorkerConfig) -> dict[str, Any]:
    """Parse one source file and atomically write sample files."""
    try:
        samples, structural_rejections = parse_structure_file_with_rejections(
            record.path,
            backend=worker_cfg.backend,
            min_length=worker_cfg.min_length,
            max_length=worker_cfg.max_length,
            chain_id=worker_cfg.chain_id,
            residue_mappings=worker_cfg.residue_mappings,
            allowed_methods=worker_cfg.allowed_methods,
            max_xray_resolution_angstrom=worker_cfg.max_xray_resolution_angstrom,
            max_cryoem_resolution_angstrom=worker_cfg.max_cryoem_resolution_angstrom,
            missing_calpha_policy=worker_cfg.missing_calpha_policy,
            max_terminal_trim_fraction=worker_cfg.max_terminal_trim_fraction,
        )
    except (IndexError, TypeError, AttributeError, AssertionError):
        raise
    except Exception as exc:
        return {
            "source_path": record.path,
            "source_size": record.size,
            "source_mtime_ns": record.mtime_ns,
            "status": FAILURE_STATUS,
            "manifest_rows": [],
            "rejections": [{"source_file": record.path, "reason": "parse_error", "message": str(exc)}],
            "error_message": str(exc),
        }
    rejections = [rejection.to_dict() for rejection in structural_rejections]
    if not samples:
        if not rejections:
            rejections = [
                {
                    "source_file": record.path,
                    "reason": "no_accepted_samples",
                    "message": (
                        "No chains passed structure metadata, residue, or length filters "
                        f"(including max_length={worker_cfg.max_length})."
                    ),
                }
            ]
        status = DETERMINISTIC_REJECTION_STATUS
    else:
        status = SUCCESS_STATUS
    rows = [save_processed_sample(sample, worker_cfg.samples_dir) for sample in samples]
    return {
        "source_path": record.path,
        "source_size": record.size,
        "source_mtime_ns": record.mtime_ns,
        "status": status,
        "manifest_rows": rows,
        "rejections": rejections,
        "error_message": "",
    }


def record_result(conn: sqlite3.Connection, result: dict[str, Any], *, current_hash: str) -> None:
    """Persist one worker result in SQLite."""
    source_path = str(result["source_path"])
    now = _utc_now()
    previous = conn.execute(
        "SELECT attempt_count, first_seen_utc FROM source_files WHERE source_path=?",
        (source_path,),
    )
    row = previous.fetchone()
    attempt_count = int(row[0]) + 1 if row is not None else 1
    first_seen = str(row[1]) if row is not None else now
    rejections = result["rejections"]
    manifest_rows = result["manifest_rows"]
    with conn:
        conn.execute("DELETE FROM manifest_rows WHERE source_path=?", (source_path,))
        conn.execute(
            """
            INSERT OR REPLACE INTO source_files(
                source_path, source_size, source_mtime_ns, status, config_hash,
                accepted_chain_count, rejection_info, attempt_count, first_seen_utc,
                started_utc, finished_utc, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_path,
                int(result["source_size"]),
                int(result["source_mtime_ns"]),
                str(result["status"]),
                current_hash,
                len(manifest_rows),
                json.dumps(rejections, sort_keys=True),
                attempt_count,
                first_seen,
                now,
                now,
                str(result.get("error_message", "")),
            ),
        )
        for manifest_row in manifest_rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO manifest_rows(sample_id, source_path, row_json, config_hash)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(manifest_row["sample_id"]),
                    source_path,
                    json.dumps(manifest_row, sort_keys=True),
                    current_hash,
                ),
            )


def load_manifest_rows_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Load deterministic manifest rows from SQLite."""
    rows = conn.execute("SELECT row_json FROM manifest_rows ORDER BY sample_id").fetchall()
    return [json.loads(row[0]) for row in rows]


def load_rejections_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Load deterministic chain/file rejections from SQLite."""
    rows = conn.execute(
        "SELECT rejection_info FROM source_files WHERE rejection_info != '[]' ORDER BY source_path"
    ).fetchall()
    rejections: list[dict[str, Any]] = []
    for row in rows:
        rejections.extend(json.loads(row[0]))
    return rejections


def build_summary(conn: sqlite3.Connection, cfg: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    """Build a deterministic preprocessing summary from SQLite state."""
    manifest_rows = load_manifest_rows_from_db(conn)
    rejections = load_rejections_from_db(conn)
    status_counts = {
        str(status): int(count)
        for status, count in conn.execute("SELECT status, COUNT(*) FROM source_files GROUP BY status").fetchall()
    }
    counts: dict[str, int] = {}
    for rejection in rejections:
        reason = str(rejection["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return {
        "partial": partial,
        "accepted_samples": len(manifest_rows),
        "accepted_files": int(status_counts.get(SUCCESS_STATUS, 0)),
        "deterministically_rejected_files": int(status_counts.get(DETERMINISTIC_REJECTION_STATUS, 0)),
        "failed_files": int(status_counts.get(FAILURE_STATUS, 0)),
        "rejected_files": len({str(rejection["source_file"]) for rejection in rejections}),
        "rejected_records": len(rejections),
        "rejection_counts": counts,
        "status_counts": status_counts,
        "rejections": rejections,
        "backend": cfg.get("backend", "auto"),
        "source_dir": str(Path(cfg["source_dir"])),
        "state_db_path": str(default_state_db_path(cfg)),
        "updated_utc": _utc_now(),
    }


def write_snapshots(conn: sqlite3.Connection, cfg: dict[str, Any], *, partial: bool) -> None:
    """Write manifest and summary snapshots atomically."""
    manifest_path = Path(cfg["manifest_path"])
    summary_path = Path(cfg.get("summary_path", manifest_path.with_suffix(".preprocess_summary.json")))
    if partial:
        manifest_path = manifest_path.with_name(f"{manifest_path.stem}.partial{manifest_path.suffix}")
        summary_path = summary_path.with_name("preprocess_summary.partial.json")
    rows = load_manifest_rows_from_db(conn)
    tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    write_manifest(rows, tmp_manifest)
    tmp_manifest.replace(manifest_path)
    _atomic_write_json(summary_path, build_summary(conn, cfg, partial=partial))


def _record_future_result(
    future: Future,
    conn: sqlite3.Connection,
    *,
    current_hash: str,
) -> tuple[int, int, int, int]:
    result = future.result()
    record_result(conn, result, current_hash=current_hash)
    accepted = len(result["manifest_rows"])
    rejected = len(result["rejections"])
    failed = 1 if result["status"] == FAILURE_STATUS else 0
    return 1, accepted, rejected, failed


def run_preprocessing(config_path: str | Path, options: PreprocessOptions) -> Path:
    """Run resumable preprocessing and return the final manifest path."""
    if options.workers <= 0:
        raise ValueError("--workers must be positive")
    if options.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    cfg = load_yaml(config_path)
    current_hash = config_hash(cfg)
    state_db = default_state_db_path(cfg)
    state_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_db)
    try:
        init_db(conn)
        if options.restart:
            reset_db(conn)
        validate_or_set_config_hash(conn, current_hash, restart=options.restart)
        sources = [source_record(path) for path in iter_supported_structure_files(cfg["source_dir"])]
        pending = classify_pending_sources(
            conn,
            sources,
            current_hash=current_hash,
            resume=options.resume,
            retry_failures=options.retry_failures,
        )
        worker_cfg = worker_config_from_cfg(cfg)
        processed = 0
        accepted = 0
        rejected = 0
        failed = 0
        max_in_flight = max(1, options.workers * 2)
        next_index = 0
        bar = tqdm(
            total=len(pending),
            desc="Preprocessing structures",
            unit="file",
            dynamic_ncols=True,
            disable=not options.show_progress,
        )
        interrupted = False
        try:
            if options.workers == 1:
                for record in pending:
                    result = worker_process_source(record, worker_cfg)
                    record_result(conn, result, current_hash=current_hash)
                    processed += 1
                    accepted += len(result["manifest_rows"])
                    rejected += len(result["rejections"])
                    failed += 1 if result["status"] == FAILURE_STATUS else 0
                    bar.update(1)
                    bar.set_postfix(
                        {
                            "accepted": accepted,
                            "rejected": rejected,
                            "failures": failed,
                            "files/s": f"{bar.format_dict.get('rate') or 0.0:.2f}",
                        }
                    )
                    if processed % options.checkpoint_every == 0:
                        write_snapshots(conn, cfg, partial=True)
                    if options.interrupt_after_completed is not None and processed >= options.interrupt_after_completed:
                        raise KeyboardInterrupt
            else:
                executor = ProcessPoolExecutor(max_workers=options.workers)
                in_flight: set[Future] = set()
                try:
                    while next_index < len(pending) or in_flight:
                        while next_index < len(pending) and len(in_flight) < max_in_flight:
                            in_flight.add(executor.submit(worker_process_source, pending[next_index], worker_cfg))
                            next_index += 1
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            done_count, accepted_count, rejected_count, failed_count = _record_future_result(
                                future,
                                conn,
                                current_hash=current_hash,
                            )
                            processed += done_count
                            accepted += accepted_count
                            rejected += rejected_count
                            failed += failed_count
                            bar.update(done_count)
                            bar.set_postfix(
                                {
                                    "accepted": accepted,
                                    "rejected": rejected,
                                    "failures": failed,
                                    "files/s": f"{bar.format_dict.get('rate') or 0.0:.2f}",
                                }
                            )
                            if processed % options.checkpoint_every == 0:
                                write_snapshots(conn, cfg, partial=True)
                            if (
                                options.interrupt_after_completed is not None
                                and processed >= options.interrupt_after_completed
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
            write_snapshots(conn, cfg, partial=True)
            print(f"Interrupted after processing {processed} source file(s).", file=sys.stderr)
            print(f"State database: {state_db}", file=sys.stderr)
            print(
                "Resume with: "
                f"python scripts/preprocess_pdb.py --config {config_path} --workers {options.workers} --resume",
                file=sys.stderr,
            )
        finally:
            bar.close()
        if interrupted:
            return Path(cfg["manifest_path"])
        write_snapshots(conn, cfg, partial=False)
        write_snapshots(conn, cfg, partial=True)
        return Path(cfg["manifest_path"])
    finally:
        conn.close()


def main() -> None:
    """Run preprocessing."""
    parser = argparse.ArgumentParser(description="Parse structure files and save C-alpha distance matrices.")
    parser.add_argument("--config", required=True, type=Path, help="YAML preprocessing config.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for structure parsing.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from the SQLite processing-state database. Enabled by default.",
    )
    parser.add_argument("--restart", action="store_true", help="Discard previous processing state.")
    parser.add_argument("--retry-failures", action="store_true", help="Retry technical failures from prior runs.")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Write partial snapshots every N files.")
    args = parser.parse_args()
    start = time.time()
    manifest = run_preprocessing(
        args.config,
        PreprocessOptions(
            workers=args.workers,
            resume=args.resume,
            restart=args.restart,
            retry_failures=args.retry_failures,
            checkpoint_every=args.checkpoint_every,
        ),
    )
    print(f"Wrote preprocessing outputs for {args.config} to {manifest} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
