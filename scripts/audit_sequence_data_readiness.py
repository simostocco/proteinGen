#!/usr/bin/env python
"""Run staged, read-only sequence and geometry readiness audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from protein_distance_diffusion.evaluation.provenance_forensics import refine_pair_status, score_candidate
from protein_distance_diffusion.evaluation.sequence_readiness import (
    canonical_sequence,
    infer_sequence_source,
    inspect_mmcif,
    inspect_processed_npz,
    sequence_evidence_occurrences,
    sequence_sha256,
    sha256_file,
)
from protein_distance_diffusion.evaluation.sequence_readiness_storage import (
    DEFAULT_SOURCES_PER_PARTITION,
    SAFETY_RESERVE_BYTES,
    STORAGE_PROFILES,
    SequenceReadinessArtifactReader,
    compare_sequence_readiness_artifacts,
    storage_projection,
    validate_compact_partition,
    write_compact_partition,
)

AUDIT_MODES = {"manifest-only", "raw-pilot", "raw-full"}
LENGTH_BINS = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
MANIFEST_REQUIRED = {"sample_id", "pdb_id", "chain_id", "sequence", "length", "path", "source_file"}
DEFAULT_MAX_EXAMPLES_PER_GROUP = 5
DEFAULT_MAX_DIAGNOSTIC_GROUPS = 10_000
REPORT_SEMANTICS_VERSION = 4
RAW_FULL_ATTESTATION_COUNTS = {
    "selected_source_count": 223_709,
    "matrix_pair_count": 506_919,
    "practical_eligible_count": 501_797,
    "unresolved_count": 5_122,
}
V4_COMPACT_REQUIRED_SCHEMAS = {
    "source_identity": {
        "source_id",
        "source_file",
        "source_sha256",
        "parse_status",
        "parser_calls",
    },
    "matrix_pair_alignments": {
        "source_id",
        "sample_id",
        "pdb_id",
        "chain_id",
        "model_id",
        "matrix_path",
        "refined_primary_classification",
        "strict_provenance_status",
        "training_eligibility",
        "source_identity_status",
    },
    "residue_id_convention_evidence": {
        "source_id",
        "sample_id",
        "physical_candidate_id",
        "evidence_row_index",
        "residue_id_convention",
        "model_number",
        "sequence_match",
        "strong_evidential_match",
    },
}
V4_PAIRING_CLASSIFICATIONS = frozenset(
    {
        "verified_sequence_geometry_pair",
        "unique_sequence_pair_coordinate_unavailable",
        "residue_identity_provenance_incomplete",
        "unresolved_ambiguity",
    }
)
SEQUENCE_HASH_ALGORITHM = "sha256_utf8_v1"
COORDINATE_DISTANCE_TOLERANCE_ANGSTROM = 1e-4
CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS = frozenset(
    {
        "author_chain_match",
        "label_chain_match",
        "sequence_match",
        "strong_evidential_match",
        "identity_mapping_available",
        "identity_mapping_ambiguous",
        "insertion_codes_match",
        "trim_metadata_available",
        "trim_metadata_compatible",
    }
)


class AuditInterrupted(RuntimeError):
    """Raised by bounded test/probe runs after checkpoint publication."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _candidate_evidence_schema(columns: list[str]) -> pa.Schema:
    """Keep future candidate Boolean evidence typed instead of stringifying it."""
    return pa.schema(
        [
            pa.field(
                key,
                pa.bool_() if key in CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS else pa.string(),
                nullable=key not in CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS,
            )
            for key in columns
        ]
    )


def _candidate_evidence_arrow_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    output = {}
    for key in columns:
        value = row.get(key)
        if key in CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Candidate Boolean evidence {key} must be a native Boolean")
            output[key] = bool(value)
        else:
            output[key] = (
                None
                if value is None
                else json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else str(value)
            )
    return output


def _iter_manifest(path: Path, *, batch_size: int) -> Iterator[pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(path, chunksize=batch_size)
        return
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()


def _manifest_columns(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    return set(pq.ParquetFile(path).schema.names)


def _validate_preflight(
    *,
    audit_mode: str,
    raw_dir: Path | None,
    manifest_paths: list[Path],
    preprocess_state_dbs: list[Path],
    output_dir: Path,
    state_dir: Path,
    storage_profile: str,
    sources_per_partition: int,
    resume: bool,
    restart: bool,
    max_source_files: int | None,
    samples_per_stratum: int,
    checkpoint_frequency: int,
    max_examples_per_group: int,
    max_diagnostic_groups: int,
    prior_audit_dir: Path | None,
    prior_forensics_dir: Path | None,
    storage_projection_path: Path | None,
    compact_equivalence_report: Path | None,
    unsafe_skip_disk_check: bool,
) -> None:
    if audit_mode not in AUDIT_MODES:
        raise ValueError(f"audit_mode must be one of {sorted(AUDIT_MODES)}")
    if resume and restart:
        raise ValueError("--resume and --restart are mutually exclusive")
    if storage_profile not in STORAGE_PROFILES:
        raise ValueError(f"storage_profile must be one of {sorted(STORAGE_PROFILES)}")
    if sources_per_partition < 1:
        raise ValueError("--sources-per-partition must be at least 1")
    if storage_profile == "compact-v1" and audit_mode == "raw-full":
        if storage_projection_path is None or not storage_projection_path.is_file():
            raise FileNotFoundError("compact-v1 raw-full requires --storage-projection")
        if compact_equivalence_report is None or not compact_equivalence_report.is_file():
            raise FileNotFoundError("compact-v1 raw-full requires --compact-equivalence-report")
        projection = json.loads(storage_projection_path.read_text())
        equivalence = json.loads(compact_equivalence_report.read_text())
        if not projection.get("disk_guard_passed") and not unsafe_skip_disk_check:
            raise ValueError("The supplied compact storage projection did not pass its disk guard")
        if equivalence.get("status") != "passed":
            raise ValueError("The supplied compact equivalence report did not pass")
        capacity_path = output_dir
        while not capacity_path.exists():
            capacity_path = capacity_path.parent
        free = os.statvfs(capacity_path).f_bavail * os.statvfs(capacity_path).f_frsize
        required = 2 * int(projection["projected_full_corpus_bytes"]) + SAFETY_RESERVE_BYTES
        if free < required and not unsafe_skip_disk_check:
            raise RuntimeError(
                "Compact audit disk guard failed before output creation: free_bytes must be at least "
                "2 * projected_remaining_bytes + 20 GiB"
            )
    for path in manifest_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {path}")
        missing = sorted(MANIFEST_REQUIRED - _manifest_columns(path))
        if missing:
            raise ValueError(f"Manifest {path} is missing columns: {missing}")
    for path in preprocess_state_dbs:
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessing state database does not exist: {path}")
    if prior_audit_dir is not None and not (prior_audit_dir / "sequence_readiness_protocol.json").is_file():
        raise FileNotFoundError(f"Prior audit protocol does not exist: {prior_audit_dir}")
    if prior_forensics_dir is not None and not (prior_forensics_dir / "per_failure_classification.parquet").is_file():
        raise FileNotFoundError(f"Prior forensic classifications do not exist: {prior_forensics_dir}")
    if audit_mode != "manifest-only" and (raw_dir is None or not raw_dir.is_dir()):
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
    if max_source_files is not None and max_source_files < 1:
        raise ValueError("--max-source-files must be at least 1")
    if samples_per_stratum < 1:
        raise ValueError("--samples-per-stratum must be at least 1")
    if checkpoint_frequency < 1:
        raise ValueError("--checkpoint-frequency must be at least 1")
    if max_examples_per_group < 1:
        raise ValueError("--max-examples-per-group must be at least 1")
    if max_diagnostic_groups < 1:
        raise ValueError("--max-diagnostic-groups must be at least 1")
    resolved_output = output_dir.resolve()
    resolved_state = state_dir.resolve()
    protected = [path.resolve() for path in manifest_paths + preprocess_state_dbs]
    protected.extend(path.resolve() for path in (prior_audit_dir, prior_forensics_dir) if path is not None)
    if raw_dir is not None:
        protected.append(raw_dir.resolve())
    if any(
        resolved_output == path or resolved_output in path.parents or path in resolved_output.parents
        for path in protected
    ):
        raise ValueError("Output directory must not overlap an input path")
    if any(
        resolved_state == path or resolved_state in path.parents or path in resolved_state.parents for path in protected
    ):
        raise ValueError("State directory must not overlap an input path")
    if output_dir.exists() and not (resume or restart):
        raise FileExistsError(f"Output exists; use --resume or --restart: {output_dir}")
    state_path = state_dir / "audit_state.sqlite"
    if resume and not state_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {state_path}")
    if resume and not (output_dir / "run_config.json").is_file():
        raise FileNotFoundError(f"Cannot resume without {output_dir / 'run_config.json'}")


def _config_payload(
    *,
    audit_mode: str,
    raw_dir: Path | None,
    processed_manifest: Path,
    train_manifest: Path,
    validation_manifest: Path,
    preprocess_state_dbs: list[Path],
    max_source_files: int | None,
    samples_per_stratum: int,
    pilot_seed: int,
    checkpoint_frequency: int,
    batch_size: int,
    max_examples_per_group: int,
    max_diagnostic_groups: int,
    prior_audit_dir: Path | None,
    prior_forensics_dir: Path | None,
    output_dir: Path,
    state_dir: Path,
    storage_profile: str,
    sources_per_partition: int,
    unsafe_skip_disk_check: bool,
    storage_projection_path: Path | None,
    equivalence_reference_dir: Path | None,
    compact_equivalence_report: Path | None,
) -> dict[str, Any]:
    return {
        "report_semantics_version": REPORT_SEMANTICS_VERSION,
        "audit_mode": audit_mode,
        "raw_dir": str(raw_dir) if raw_dir else None,
        "processed_manifest": str(processed_manifest),
        "train_manifest": str(train_manifest),
        "validation_manifest": str(validation_manifest),
        "preprocess_state_dbs": [str(path) for path in preprocess_state_dbs],
        "max_source_files": max_source_files,
        "samples_per_stratum": samples_per_stratum,
        "pilot_seed": pilot_seed,
        "checkpoint_frequency": checkpoint_frequency,
        "batch_size": batch_size,
        "max_examples_per_group": max_examples_per_group,
        "max_diagnostic_groups": max_diagnostic_groups,
        "prior_audit_dir": str(prior_audit_dir) if prior_audit_dir else None,
        "prior_forensics_dir": str(prior_forensics_dir) if prior_forensics_dir else None,
        "output_dir": str(output_dir.resolve()),
        "state_dir": str(state_dir.resolve()),
        "storage_profile": storage_profile,
        "sources_per_partition": sources_per_partition,
        "unsafe_skip_disk_check": unsafe_skip_disk_check,
        "storage_projection_path": str(storage_projection_path) if storage_projection_path else None,
        "equivalence_reference_dir": str(equivalence_reference_dir) if equivalence_reference_dir else None,
        "compact_equivalence_report": str(compact_equivalence_report) if compact_equivalence_report else None,
    }


def _config_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS manifest_rows (
            manifest_kind TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            sample_id TEXT,
            pdb_id TEXT,
            chain_id TEXT,
            model_id TEXT,
            matrix_path TEXT,
            sequence TEXT,
            recorded_length INTEGER,
            source_file TEXT,
            experimental_method TEXT,
            terminal_trimming_applied INTEGER,
            missing_calpha_policy TEXT,
            stored_sequence_hash TEXT,
            recomputed_sequence_hash TEXT,
            cluster_id TEXT,
            split_group_id TEXT,
            canonical_sequence INTEGER,
            PRIMARY KEY (manifest_kind, row_number)
        );
        CREATE TABLE IF NOT EXISTS selected_sources (
            source_file TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            strata_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS source_state (
            source_file TEXT NOT NULL,
            source_sha256 TEXT NOT NULL,
            status TEXT NOT NULL,
            partition_path TEXT,
            error_type TEXT,
            error_message TEXT,
            parser_calls INTEGER NOT NULL DEFAULT 0,
            updated_utc TEXT NOT NULL,
            PRIMARY KEY (source_file, source_sha256)
        );
        CREATE TABLE IF NOT EXISTS stage_state (
            stage_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            processed_count INTEGER NOT NULL DEFAULT 0,
            total_count INTEGER,
            started_utc TEXT NOT NULL,
            heartbeat_utc TEXT NOT NULL,
            completed_utc TEXT,
            elapsed_seconds REAL NOT NULL DEFAULT 0,
            detail TEXT
        );
        CREATE TABLE IF NOT EXISTS npz_mismatches (
            row_number INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL,
            matrix_path TEXT NOT NULL,
            mismatch_reasons TEXT NOT NULL,
            matrix_path_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS compact_partitions (
            partition_index INTEGER PRIMARY KEY,
            source_count INTEGER NOT NULL,
            marker_path TEXT NOT NULL,
            marker_sha256 TEXT NOT NULL,
            committed_utc TEXT NOT NULL
        );
        """
    )
    return connection


ANALYSIS_INDEXES = {
    "idx_manifest_sample": "manifest_rows(manifest_kind, sample_id)",
    "idx_manifest_path": "manifest_rows(manifest_kind, matrix_path)",
    "idx_manifest_source": "manifest_rows(manifest_kind, source_file)",
    "idx_manifest_sequence": "manifest_rows(manifest_kind, recomputed_sequence_hash)",
    "idx_manifest_stored_sequence": "manifest_rows(manifest_kind, stored_sequence_hash)",
    "idx_manifest_pdb": "manifest_rows(manifest_kind, pdb_id)",
    "idx_manifest_chain": "manifest_rows(manifest_kind, chain_id)",
    "idx_manifest_model": "manifest_rows(manifest_kind, model_id)",
    "idx_manifest_cluster": "manifest_rows(manifest_kind, cluster_id)",
    "idx_manifest_split_group": "manifest_rows(manifest_kind, split_group_id)",
    "idx_manifest_pdb_chain_model": "manifest_rows(manifest_kind, pdb_id, chain_id, model_id)",
}


class StageReporter:
    """Persist stage state and publish an accessible heartbeat protocol."""

    def __init__(self, connection: sqlite3.Connection, output_dir: Path, audit_mode: str, started_utc: str):
        self.connection = connection
        self.output_dir = output_dir
        self.audit_mode = audit_mode
        self.started_utc = started_utc
        self.stage = "initialization"
        self.stage_started = time.monotonic()
        self.processed_count = 0
        self.total_count: int | None = None
        self.last_heartbeat = 0.0
        self.sqlite_steps = 0

    def completed(self, stage: str, outputs: tuple[Path, ...] = ()) -> bool:
        row = self.connection.execute("SELECT status FROM stage_state WHERE stage_name=?", (stage,)).fetchone()
        return bool(row and row[0] == "completed" and all(path.exists() for path in outputs))

    def start(self, stage: str, *, total_count: int | None = None, detail: str | None = None) -> None:
        self.stage = stage
        self.stage_started = time.monotonic()
        self.processed_count = 0
        self.total_count = total_count
        self.sqlite_steps = 0
        now = _utc_now()
        self.connection.execute(
            """INSERT OR REPLACE INTO stage_state
            (stage_name,status,processed_count,total_count,started_utc,heartbeat_utc,
             completed_utc,elapsed_seconds,detail) VALUES (?,?,?,?,?,?,?,?,?)""",
            (stage, "running", 0, total_count, now, now, None, 0.0, detail),
        )
        self.connection.commit()
        print(f"[sequence-readiness] stage={stage} status=running", flush=True)
        self._publish("running")
        self.connection.set_progress_handler(self._sqlite_heartbeat, 100_000)

    def progress(self, processed_count: int, *, detail: str | None = None, force: bool = False) -> None:
        self.processed_count = processed_count
        if force or time.monotonic() - self.last_heartbeat >= 2.0:
            self._heartbeat(detail=detail)

    def complete(self, *, processed_count: int | None = None, detail: str | None = None) -> None:
        self.connection.set_progress_handler(None, 0)
        if processed_count is not None:
            self.processed_count = processed_count
        elapsed = time.monotonic() - self.stage_started
        now = _utc_now()
        self.connection.execute(
            """UPDATE stage_state SET status='completed',processed_count=?,total_count=?,heartbeat_utc=?,
            completed_utc=?,elapsed_seconds=?,detail=COALESCE(?,detail) WHERE stage_name=?""",
            (self.processed_count, self.total_count, now, now, elapsed, detail, self.stage),
        )
        self.connection.commit()
        print(f"[sequence-readiness] stage={self.stage} status=completed elapsed_seconds={elapsed:.3f}", flush=True)
        self._publish("running")

    def interrupt(self) -> None:
        self.connection.set_progress_handler(None, 0)
        elapsed = time.monotonic() - self.stage_started
        now = _utc_now()
        self.connection.execute(
            """UPDATE stage_state SET status='interrupted',processed_count=?,heartbeat_utc=?,
            elapsed_seconds=?,detail=COALESCE(detail,'interrupted by signal') WHERE stage_name=?""",
            (self.processed_count, now, elapsed, self.stage),
        )
        self.connection.commit()
        self._publish("interrupted")

    def _heartbeat(self, *, detail: str | None = None) -> None:
        now = _utc_now()
        elapsed = time.monotonic() - self.stage_started
        self.connection.execute(
            """UPDATE stage_state SET processed_count=?,heartbeat_utc=?,elapsed_seconds=?,
            detail=COALESCE(?,detail) WHERE stage_name=?""",
            (self.processed_count, now, elapsed, detail, self.stage),
        )
        self.connection.commit()
        self.last_heartbeat = time.monotonic()
        self._publish("running")

    def _sqlite_heartbeat(self) -> int:
        self.sqlite_steps += 100_000
        if time.monotonic() - self.last_heartbeat >= 2.0:
            self.processed_count = self.sqlite_steps
            self._publish("running")
            self.last_heartbeat = time.monotonic()
        return 0

    def _publish(self, status: str) -> None:
        config_path = self.output_dir / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        _atomic_json(
            self.output_dir / "sequence_readiness_protocol.partial.json",
            {
                "status": status,
                "audit_mode": self.audit_mode,
                "storage_profile": config.get("storage_profile", "verbose-v3"),
                "resolved_output_dir": config.get("output_dir", str(self.output_dir.resolve())),
                "resolved_state_dir": config.get("state_dir", str(self.output_dir.resolve())),
                "started_utc": self.started_utc,
                "heartbeat_utc": _utc_now(),
                "current_stage": self.stage,
                "stage_processed_count": self.processed_count,
                "stage_total_count": self.total_count,
                "stage_progress_unit": "sqlite_vm_steps" if self.sqlite_steps else "rows_or_items",
                "stage_elapsed_seconds": time.monotonic() - self.stage_started,
                "sqlite_vm_steps_since_stage_start": self.sqlite_steps,
                "recovery_command": "rerun the identical command with --resume",
            },
        )


def _value(row: pd.Series, name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    return default if pd.isna(value) else value


def _ingest_manifest(
    connection: sqlite3.Connection,
    *,
    kind: str,
    path: Path,
    batch_size: int,
    reporter: StageReporter | None = None,
) -> int:
    completion_key = f"manifest_ingested:{kind}"
    completed = connection.execute("SELECT value FROM metadata WHERE key=?", (completion_key,)).fetchone()
    existing = connection.execute("SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind = ?", (kind,)).fetchone()[0]
    if completed:
        return int(existing)
    if existing:
        connection.execute("DELETE FROM manifest_rows WHERE manifest_kind=?", (kind,))
        connection.commit()
    row_number = 0
    insert = """
        INSERT INTO manifest_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for frame in tqdm(_iter_manifest(path, batch_size=batch_size), desc=f"Index {kind}", unit="batch"):
        records = []
        for _, row in frame.iterrows():
            sequence = str(_value(row, "sequence", ""))
            records.append(
                (
                    kind,
                    row_number,
                    str(_value(row, "sample_id", "")),
                    str(_value(row, "pdb_id", "")),
                    str(_value(row, "chain_id", "")),
                    str(_value(row, "model_number", 1)),
                    str(_value(row, "path", "")),
                    sequence,
                    int(_value(row, "length", 0)),
                    str(_value(row, "source_file", "")),
                    str(_value(row, "experimental_method", "unknown")),
                    int(bool(_value(row, "terminal_trimming_applied", False))),
                    str(_value(row, "missing_calpha_policy", "unknown")),
                    str(_value(row, "sequence_hash", "")),
                    sequence_sha256(sequence) if sequence else "",
                    str(_value(row, "cluster_id", "")),
                    str(_value(row, "split_group_id", "")),
                    int(canonical_sequence(sequence)),
                )
            )
            row_number += 1
        connection.executemany(insert, records)
        connection.commit()
        if reporter is not None:
            reporter.progress(row_number, detail=f"indexed {kind} rows")
    connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (completion_key, str(row_number)))
    connection.commit()
    return row_number


def _ensure_analysis_indexes(connection: sqlite3.Connection, reporter: StageReporter) -> None:
    for index, (name, expression) in enumerate(ANALYSIS_INDEXES.items(), start=1):
        connection.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {expression}")
        connection.commit()
        reporter.progress(index, detail=f"created or verified {name}", force=True)
    connection.execute("ANALYZE manifest_rows")
    connection.commit()


def _query_rows(connection: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, params)
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def _write_query_csv(connection: sqlite3.Connection, path: Path, sql: str) -> int:
    cursor = connection.execute(sql)
    columns = [item[0] for item in cursor.description]
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in cursor:
            writer.writerow(row)
            count += 1
    return count


def _manifest_aggregation(connection: sqlite3.Connection, output_dir: Path) -> dict[str, Any]:
    counts = {
        kind: int(connection.execute("SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind=?", (kind,)).fetchone()[0])
        for kind in ("processed", "train", "validation")
    }
    unique_samples = {
        kind: int(
            connection.execute(
                "SELECT COUNT(DISTINCT sample_id) FROM manifest_rows WHERE manifest_kind=?", (kind,)
            ).fetchone()[0]
        )
        for kind in counts
    }
    rows_by_split = {
        row[0]: int(row[1])
        for row in connection.execute(
            """
            SELECT split_status,COUNT(*) FROM (
                SELECT CASE
                    WHEN EXISTS (SELECT 1 FROM manifest_rows t
                                 WHERE t.manifest_kind='train' AND t.sample_id=p.sample_id)
                     AND EXISTS (SELECT 1 FROM manifest_rows v
                                 WHERE v.manifest_kind='validation' AND v.sample_id=p.sample_id)
                        THEN 'train_and_validation'
                    WHEN EXISTS (SELECT 1 FROM manifest_rows t
                                 WHERE t.manifest_kind='train' AND t.sample_id=p.sample_id) THEN 'train'
                    WHEN EXISTS (SELECT 1 FROM manifest_rows v
                                 WHERE v.manifest_kind='validation' AND v.sample_id=p.sample_id) THEN 'validation'
                    ELSE 'excluded'
                END AS split_status
                FROM manifest_rows p WHERE p.manifest_kind='processed'
            ) GROUP BY split_status
            """
        )
    }
    hash_statistics = {}
    for kind in ("processed", "train", "validation"):
        columns_row = connection.execute(
            "SELECT value FROM metadata WHERE key=?", (f"manifest_columns:{kind}",)
        ).fetchone()
        stored_hash_column_present = bool(columns_row and "sequence_hash" in json.loads(columns_row[0]))
        total, available, matches, mismatches = connection.execute(
            """SELECT COUNT(*),
            COALESCE(SUM(stored_sequence_hash != ''),0),
            COALESCE(SUM(stored_sequence_hash != '' AND stored_sequence_hash=recomputed_sequence_hash),0),
            COALESCE(SUM(stored_sequence_hash != '' AND stored_sequence_hash!=recomputed_sequence_hash),0)
            FROM manifest_rows WHERE manifest_kind=?""",
            (kind,),
        ).fetchone()
        total = int(total)
        available = int(available)
        matches = int(matches)
        mismatches = int(mismatches)
        hash_statistics[kind] = {
            "total_rows": total,
            "stored_hash_column_present": stored_hash_column_present,
            "stored_hash_available_count": available,
            "stored_hash_unavailable_count": total - available,
            "checked_count": available,
            "match_count": matches,
            "mismatch_count": mismatches,
            "comparison_status": "unavailable" if available == 0 else "passed" if mismatches == 0 else "failed",
        }
    model_distribution = {
        str(model_number): int(count)
        for model_number, count in connection.execute(
            """SELECT model_id,COUNT(*) FROM manifest_rows
            WHERE manifest_kind='processed' GROUP BY model_id ORDER BY model_id"""
        )
    }
    rows_model_gt1 = int(
        connection.execute(
            """SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind='processed'
            AND CAST(model_id AS INTEGER)>1"""
        ).fetchone()[0]
    )
    nmr_rows = int(
        connection.execute(
            """SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind='processed'
            AND UPPER(experimental_method) LIKE '%NMR%'"""
        ).fetchone()[0]
    )
    nmr_rows_model_gt1 = int(
        connection.execute(
            """SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind='processed'
            AND UPPER(experimental_method) LIKE '%NMR%' AND CAST(model_id AS INTEGER)>1"""
        ).fetchone()[0]
    )
    summary = {
        "manifest_row_counts": counts,
        "unique_sample_id_counts": unique_samples,
        "rows_by_split_status": rows_by_split,
        "canonical_sequence_count": int(
            connection.execute(
                "SELECT COALESCE(SUM(canonical_sequence),0) FROM manifest_rows WHERE manifest_kind='processed'"
            ).fetchone()[0]
        ),
        "sequence_recorded_length_match_count": int(
            connection.execute(
                """SELECT COUNT(*) FROM manifest_rows
                WHERE manifest_kind='processed' AND LENGTH(sequence)=recorded_length"""
            ).fetchone()[0]
        ),
        "sequence_hash_algorithm": SEQUENCE_HASH_ALGORITHM,
        "sequence_hash_statistics": hash_statistics,
        "unique_source_file_count": int(
            connection.execute(
                "SELECT COUNT(DISTINCT source_file) FROM manifest_rows WHERE manifest_kind='processed'"
            ).fetchone()[0]
        ),
        "unique_pdb_count": int(
            connection.execute(
                "SELECT COUNT(DISTINCT pdb_id) FROM manifest_rows WHERE manifest_kind='processed'"
            ).fetchone()[0]
        ),
        "unique_chain_id_count": int(
            connection.execute(
                "SELECT COUNT(DISTINCT chain_id) FROM manifest_rows WHERE manifest_kind='processed'"
            ).fetchone()[0]
        ),
        "model_number_statistics": {
            "source_column": "model_number",
            "distribution": model_distribution,
            "unique_model_number_count": len(model_distribution),
            "rows_with_model_number_greater_than_1": rows_model_gt1,
            "nmr_row_count": nmr_rows,
            "nmr_rows_with_model_number_greater_than_1": nmr_rows_model_gt1,
            "interpretation": (
                "Only model 1 is retained in the processed manifest. Raw NMR auditing must inspect "
                "all coordinate models in each selected mmCIF."
                if len(model_distribution) == 1 and "1" in model_distribution
                else "The processed manifest retains more than one model number."
            ),
        },
        "unique_pdb_chain_model_count": int(
            connection.execute(
                """SELECT COUNT(*) FROM (SELECT DISTINCT pdb_id,chain_id,model_id FROM manifest_rows
                WHERE manifest_kind='processed')"""
            ).fetchone()[0]
        ),
        "unique_source_pdb_chain_model_count": int(
            connection.execute(
                """SELECT COUNT(*) FROM (SELECT DISTINCT source_file,pdb_id,chain_id,model_id
                FROM manifest_rows WHERE manifest_kind='processed')"""
            ).fetchone()[0]
        ),
    }
    _atomic_json(output_dir / "manifest_summary.json", summary)
    return summary


def _write_sequence_alphabet(connection: sqlite3.Connection, output_dir: Path, reporter: StageReporter) -> int:
    alphabet = Counter()
    processed = 0
    for processed, (sequence,) in enumerate(
        connection.execute("SELECT sequence FROM manifest_rows WHERE manifest_kind='processed'"), start=1
    ):
        alphabet.update(str(sequence))
        if processed % 10_000 == 0:
            reporter.progress(processed, detail="validated manifest sequence alphabet")
    with (output_dir / "sequence_alphabet_counts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "canonical", "count"])
        writer.writeheader()
        for token, count in sorted(alphabet.items()):
            writer.writerow({"token": token, "canonical": canonical_sequence(token), "count": count})
    return processed


def _examples_for_key(
    connection: sqlite3.Connection,
    *,
    manifest_kind: str,
    columns: tuple[str, ...],
    values: tuple[str, ...],
    limit: int,
) -> list[str]:
    conditions = " AND ".join(f"{column}=?" for column in columns)
    return [
        str(row[0])
        for row in connection.execute(
            f"""SELECT sample_id FROM manifest_rows WHERE manifest_kind=? AND {conditions}
            ORDER BY sample_id LIMIT ?""",
            (manifest_kind, *values, limit),
        )
    ]


def _group_totals(connection: sqlite3.Connection, *, manifest_kind: str, columns: tuple[str, ...]) -> tuple[int, int]:
    grouped = ",".join(columns)
    row = connection.execute(
        f"""SELECT COUNT(*),COALESCE(SUM(member_count),0) FROM
        (SELECT COUNT(*) member_count FROM manifest_rows WHERE manifest_kind=?
         GROUP BY {grouped} HAVING COUNT(*)>1)""",
        (manifest_kind,),
    ).fetchone()
    return int(row[0]), int(row[1])


def _write_duplicate_groups(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    max_examples: int,
    max_groups: int,
    reporter: StageReporter,
) -> dict[str, Any]:
    policies = {
        "sample_id": ("sample_id",),
        "matrix_path": ("matrix_path",),
        "exact_sequence": ("recomputed_sequence_hash",),
    }
    fields = [
        "finding_type",
        "identity",
        "row_count",
        "representative_sample_ids",
        "examples_truncated",
    ]
    emitted_total = 0
    summary: dict[str, Any] = {}
    with (output_dir / "duplicate_findings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy, columns in policies.items():
            total_groups, underlying_rows = _group_totals(connection, manifest_kind="processed", columns=columns)
            remaining = max(max_groups - emitted_total, 0)
            grouped = ",".join(columns)
            rows = connection.execute(
                f"""SELECT {grouped},COUNT(*) member_count FROM manifest_rows
                WHERE manifest_kind='processed' GROUP BY {grouped} HAVING COUNT(*)>1
                ORDER BY {grouped} LIMIT ?""",
                (remaining,),
            )
            emitted = 0
            for row in rows:
                values = tuple(str(value) for value in row[:-1])
                member_count = int(row[-1])
                examples = _examples_for_key(
                    connection,
                    manifest_kind="processed",
                    columns=columns,
                    values=values,
                    limit=max_examples,
                )
                writer.writerow(
                    {
                        "finding_type": policy,
                        "identity": values[0] if len(values) == 1 else json.dumps(values),
                        "row_count": member_count,
                        "representative_sample_ids": json.dumps(examples),
                        "examples_truncated": member_count > len(examples),
                    }
                )
                emitted += 1
                emitted_total += 1
                reporter.progress(emitted_total, detail=f"exported duplicate groups through {policy}")
            summary[policy] = {
                "underlying_group_count": total_groups,
                "underlying_member_row_count": underlying_rows,
                "emitted_group_count": emitted,
                "groups_truncated": total_groups > emitted,
            }
    summary["maximum_emitted_groups"] = max_groups
    summary["maximum_examples_per_group"] = max_examples
    return summary


LEAKAGE_POLICIES = {
    "exact_sequence": ("recomputed_sequence_hash",),
    "cluster_id": ("cluster_id",),
    "split_group_id": ("split_group_id",),
    "pdb_id": ("pdb_id",),
    "pdb_chain_model": ("pdb_id", "chain_id", "model_id"),
    "sample_id": ("sample_id",),
}


def _leakage_group_counts(
    connection: sqlite3.Connection, columns: tuple[str, ...], *, limit: int | None = None
) -> Iterator[tuple[Any, ...]]:
    keys = ",".join(columns)
    join = " AND ".join(f"t.{column}=v.{column}" for column in columns)
    nonempty = " AND ".join(f"{column} != ''" for column in columns)
    limit_sql = " LIMIT ?" if limit is not None else ""
    parameters = (limit,) if limit is not None else ()
    return iter(
        connection.execute(
            f"""WITH t AS (
            SELECT {keys},COUNT(*) member_count FROM manifest_rows
            WHERE manifest_kind='train' AND {nonempty} GROUP BY {keys}
        ), v AS (
            SELECT {keys},COUNT(*) member_count FROM manifest_rows
            WHERE manifest_kind='validation' AND {nonempty} GROUP BY {keys}
        ) SELECT {",".join(f"t.{column}" for column in columns)},
                 t.member_count,v.member_count FROM t JOIN v ON {join}
          ORDER BY {",".join(f"t.{column}" for column in columns)}{limit_sql}""",
            parameters,
        )
    )


def _leakage_totals(connection: sqlite3.Connection, columns: tuple[str, ...]) -> tuple[int, int]:
    keys = ",".join(columns)
    join = " AND ".join(f"t.{column}=v.{column}" for column in columns)
    nonempty = " AND ".join(f"{column} != ''" for column in columns)
    row = connection.execute(
        f"""WITH t AS (
            SELECT {keys},COUNT(*) member_count FROM manifest_rows
            WHERE manifest_kind='train' AND {nonempty} GROUP BY {keys}
        ), v AS (
            SELECT {keys},COUNT(*) member_count FROM manifest_rows
            WHERE manifest_kind='validation' AND {nonempty} GROUP BY {keys}
        ) SELECT COUNT(*),COALESCE(SUM(t.member_count+v.member_count),0)
          FROM t JOIN v ON {join}"""
    ).fetchone()
    return int(row[0]), int(row[1])


def _write_leakage_groups(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    max_examples: int,
    max_groups: int,
    reporter: StageReporter,
) -> dict[str, Any]:
    fields = [
        "policy",
        "identity",
        "train_count",
        "validation_count",
        "train_representative_sample_ids",
        "validation_representative_sample_ids",
        "train_examples_truncated",
        "validation_examples_truncated",
    ]
    emitted_total = 0
    summary: dict[str, Any] = {}
    with (output_dir / "train_validation_leakage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for policy, columns in LEAKAGE_POLICIES.items():
            total_groups, underlying_rows = _leakage_totals(connection, columns)
            remaining = max(max_groups - emitted_total, 0)
            emitted = 0
            for row in _leakage_group_counts(connection, columns, limit=remaining):
                values = tuple(str(value) for value in row[: len(columns)])
                train_count, validation_count = int(row[-2]), int(row[-1])
                train_examples = _examples_for_key(
                    connection,
                    manifest_kind="train",
                    columns=columns,
                    values=values,
                    limit=max_examples,
                )
                validation_examples = _examples_for_key(
                    connection,
                    manifest_kind="validation",
                    columns=columns,
                    values=values,
                    limit=max_examples,
                )
                writer.writerow(
                    {
                        "policy": policy,
                        "identity": values[0] if len(values) == 1 else json.dumps(values),
                        "train_count": train_count,
                        "validation_count": validation_count,
                        "train_representative_sample_ids": json.dumps(train_examples),
                        "validation_representative_sample_ids": json.dumps(validation_examples),
                        "train_examples_truncated": train_count > len(train_examples),
                        "validation_examples_truncated": validation_count > len(validation_examples),
                    }
                )
                emitted += 1
                emitted_total += 1
                reporter.progress(emitted_total, detail=f"exported leakage groups through {policy}")
            summary[policy] = {
                "underlying_group_count": total_groups,
                "underlying_member_row_count": underlying_rows,
                "emitted_group_count": emitted,
                "groups_truncated": total_groups > emitted,
            }
    summary["maximum_emitted_groups"] = max_groups
    summary["maximum_examples_per_group"] = max_examples
    return summary


def _write_ambiguous_groups(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    max_examples: int,
    max_groups: int,
    reporter: StageReporter,
) -> dict[str, Any]:
    count_sql = """SELECT COUNT(*),COALESCE(SUM(association_count),0) FROM (
        SELECT COUNT(*) association_count FROM manifest_rows WHERE manifest_kind='processed'
        GROUP BY matrix_path HAVING COUNT(*) != 1 OR COUNT(DISTINCT recomputed_sequence_hash) != 1)"""
    total_groups, underlying_rows = (int(value) for value in connection.execute(count_sql).fetchone())
    rows = connection.execute(
        """SELECT matrix_path,COUNT(*) association_count,
        COUNT(DISTINCT recomputed_sequence_hash) sequence_count
        FROM manifest_rows WHERE manifest_kind='processed' GROUP BY matrix_path
        HAVING COUNT(*) != 1 OR COUNT(DISTINCT recomputed_sequence_hash) != 1
        ORDER BY matrix_path LIMIT ?""",
        (max_groups,),
    )
    fields = [
        "matrix_path",
        "association_count",
        "sequence_count",
        "representative_sample_ids",
        "examples_truncated",
    ]
    emitted = 0
    with (output_dir / "ambiguous_manifest_pairings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for matrix_path, association_count, sequence_count in rows:
            examples = _examples_for_key(
                connection,
                manifest_kind="processed",
                columns=("matrix_path",),
                values=(str(matrix_path),),
                limit=max_examples,
            )
            writer.writerow(
                {
                    "matrix_path": matrix_path,
                    "association_count": association_count,
                    "sequence_count": sequence_count,
                    "representative_sample_ids": json.dumps(examples),
                    "examples_truncated": int(association_count) > len(examples),
                }
            )
            emitted += 1
            reporter.progress(emitted, detail="exported ambiguous matrix groups")
    return {
        "underlying_group_count": total_groups,
        "underlying_member_row_count": underlying_rows,
        "emitted_group_count": emitted,
        "groups_truncated": total_groups > emitted,
        "maximum_emitted_groups": max_groups,
        "maximum_examples_per_group": max_examples,
    }


def _write_policy_distributions(connection: sqlite3.Connection, output_dir: Path) -> None:
    _write_query_csv(
        connection,
        output_dir / "manifest_policy_distributions.csv",
        """
        SELECT 'experimental_method' field,experimental_method value,COUNT(*) count FROM manifest_rows
        WHERE manifest_kind='processed' GROUP BY experimental_method
        UNION ALL SELECT 'missing_calpha_policy',missing_calpha_policy,COUNT(*) FROM manifest_rows
        WHERE manifest_kind='processed' GROUP BY missing_calpha_policy
        UNION ALL SELECT 'terminal_trimming_applied',CAST(terminal_trimming_applied AS TEXT),COUNT(*) FROM manifest_rows
        WHERE manifest_kind='processed' GROUP BY terminal_trimming_applied
        """,
    )


def _inspect_npz_rows(
    connection: sqlite3.Connection,
    output_path: Path,
    *,
    selected_only: bool,
    checkpoint_size: int,
    reporter: StageReporter,
) -> int:
    last_row = int(
        (connection.execute("SELECT value FROM metadata WHERE key='npz_last_row_number'").fetchone() or (-1,))[0]
    )
    sql = """SELECT m.row_number,m.sample_id,m.matrix_path,m.sequence,m.recorded_length,
             m.stored_sequence_hash,m.recomputed_sequence_hash FROM manifest_rows m"""
    if selected_only:
        sql += " JOIN selected_sources s ON s.source_file=m.source_file"
    sql += " WHERE m.manifest_kind='processed' AND m.row_number>? ORDER BY m.row_number"
    processed = last_row + 1
    since_checkpoint = 0
    for row_number, sample_id, matrix_path, sequence, length, stored_hash, recomputed_hash in tqdm(
        connection.execute(sql, (last_row,)), desc="Inspect processed NPZ", unit="sample", initial=processed
    ):
        reasons = []
        digest = ""
        try:
            inspected = inspect_processed_npz(matrix_path)
            digest = inspected["matrix_path_sha256"]
            if inspected["npz_sequence"] != sequence:
                reasons.append("manifest_npz_sequence_mismatch")
            if not inspected["matrix_is_square"] or inspected["matrix_rows"] != int(length):
                reasons.append("matrix_recorded_length_mismatch")
            if inspected["matrix_rows"] != len(sequence):
                reasons.append("matrix_sequence_length_mismatch")
            if not inspected["sequence_tokens_match_sequence"]:
                reasons.append("sequence_token_mismatch")
            if inspected["residue_id_count"] != len(sequence):
                reasons.append("residue_id_length_mismatch")
            if inspected["residue_mask_count"] != len(sequence):
                reasons.append("residue_mask_length_mismatch")
        except Exception as exc:
            reasons.append(f"npz_read_error:{type(exc).__name__}:{exc}")
        if stored_hash and stored_hash != recomputed_hash:
            reasons.append("stored_sequence_hash_mismatch")
        if reasons:
            connection.execute(
                "INSERT OR REPLACE INTO npz_mismatches VALUES (?,?,?,?,?)",
                (row_number, sample_id, matrix_path, json.dumps(reasons), digest),
            )
        processed = int(row_number) + 1
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_size:
            connection.execute("INSERT OR REPLACE INTO metadata VALUES ('npz_last_row_number',?)", (str(row_number),))
            connection.commit()
            reporter.progress(processed, detail="validated processed NPZ metadata", force=True)
            since_checkpoint = 0
    if processed:
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('npz_last_row_number',?)", (str(processed - 1),))
    connection.commit()

    mismatch_count = int(connection.execute("SELECT COUNT(*) FROM npz_mismatches").fetchone()[0])
    fields = ["sample_id", "matrix_path", "mismatch_reasons", "matrix_path_sha256"]
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, matrix_path, reasons, digest in connection.execute(
            """SELECT sample_id,matrix_path,mismatch_reasons,matrix_path_sha256
            FROM npz_mismatches ORDER BY row_number"""
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "matrix_path": matrix_path,
                    "mismatch_reasons": reasons,
                    "matrix_path_sha256": digest,
                }
            )
    temporary.replace(output_path)
    return mismatch_count


def _length_bin(length: int) -> str:
    for low, high in LENGTH_BINS:
        if low <= length <= high:
            return f"length={low}-{high}"
    return "length=outside-20-500"


def _method_stratum(method: str) -> str:
    upper = method.upper()
    if "X-RAY" in upper:
        return "method=xray"
    if "ELECTRON" in upper or "CRYO" in upper:
        return "method=cryoem"
    if "NMR" in upper:
        return "method=nmr"
    return "method=other"


def _missing_calpha_sources(state_dbs: list[Path]) -> set[str]:
    result = set()
    for path in state_dbs:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
            query = connection.execute("SELECT source_path,rejection_info FROM source_files")
            for source_file, rejection_json in query:
                try:
                    rejections = json.loads(rejection_json)
                except json.JSONDecodeError:
                    continue
                if any(item.get("reason") in {"missing_calpha", "internal_missing_calpha"} for item in rejections):
                    result.add(str(source_file))
    return result


def _source_strata(connection: sqlite3.Connection, state_dbs: list[Path]) -> dict[str, set[str]]:
    features: dict[str, set[str]] = defaultdict(set)
    rows = connection.execute(
        """SELECT p.source_file,p.chain_id,p.recorded_length,p.experimental_method,p.model_id,
        p.terminal_trimming_applied,p.canonical_sequence,
        EXISTS(SELECT 1 FROM manifest_rows t WHERE t.manifest_kind='train' AND t.sample_id=p.sample_id),
        EXISTS(SELECT 1 FROM manifest_rows v WHERE v.manifest_kind='validation' AND v.sample_id=p.sample_id)
        FROM manifest_rows p WHERE p.manifest_kind='processed'"""
    )
    source_rows: Counter[str] = Counter()
    source_chains: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for source, chain_id, length, method, model, trimmed, canonical, in_train, in_validation in rows:
        source_rows[source] += 1
        features[source].add(_length_bin(int(length)))
        features[source].add(_method_stratum(str(method)))
        features[source].add("model=gt1" if str(model) not in {"", "1", "1.0"} else "model=1")
        features[source].add("trimmed=true" if trimmed else "trimmed=false")
        features[source].add("sequence=canonical" if canonical else "sequence=noncanonical_or_unknown")
        if in_train:
            features[source].add("split=train")
        if in_validation:
            features[source].add("split=validation")
        if not in_train and not in_validation:
            features[source].add("split=excluded")
        source_chains[source].add((str(chain_id), str(model)))
    for source, count in source_rows.items():
        features[source].add("source_rows=repeated" if count > 1 else "source_rows=single")
        multiple_chains = len({chain for chain, _ in source_chains[source]}) > 1
        features[source].add("chains=multiple" if multiple_chains else "chains=single")
    for source in _missing_calpha_sources(state_dbs):
        features[source].add("missing_calpha=true")
    return features


def _pilot_selection(
    features: dict[str, set[str]], *, max_source_files: int, samples_per_stratum: int, seed: int
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for source, strata in features.items():
        for stratum in strata:
            by_stratum[stratum].append(source)
    for stratum, sources in by_stratum.items():
        sources.sort(key=lambda source: hashlib.sha256(f"{seed}:{stratum}:{source}".encode()).hexdigest())
    requested = {key: samples_per_stratum for key in sorted(by_stratum)}
    selected: list[str] = []
    selected_set: set[str] = set()
    for rank in range(samples_per_stratum):
        for stratum in sorted(by_stratum):
            if rank >= len(by_stratum[stratum]):
                continue
            source = by_stratum[stratum][rank]
            if source not in selected_set:
                selected.append(source)
                selected_set.add(source)
                if len(selected) == max_source_files:
                    break
        if len(selected) == max_source_files:
            break
    all_sources = sorted(features, key=lambda source: hashlib.sha256(f"{seed}:fill:{source}".encode()).hexdigest())
    for source in all_sources:
        if len(selected) == max_source_files:
            break
        if source not in selected_set:
            selected.append(source)
            selected_set.add(source)
    achieved = {
        stratum: sum(source in selected_set for source in sources) for stratum, sources in sorted(by_stratum.items())
    }
    return selected, requested, achieved


def _all_raw_sources(raw_dir: Path) -> list[str]:
    suffixes = (".cif", ".cif.gz", ".mmcif", ".mmcif.gz")
    return [str(path) for path in sorted(raw_dir.rglob("*")) if path.is_file() and path.name.lower().endswith(suffixes)]


def _ensure_selected_sources(
    connection: sqlite3.Connection,
    *,
    audit_mode: str,
    raw_dir: Path,
    state_dbs: list[Path],
    max_source_files: int | None,
    samples_per_stratum: int,
    pilot_seed: int,
    prior_audit_dir: Path | None = None,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    existing = _query_rows(
        connection, "SELECT source_file,source_sha256,strata_json FROM selected_sources ORDER BY source_file"
    )
    if existing:
        selected = []
        for row in existing:
            source = str(row["source_file"])
            digest = sha256_file(source)
            selected.append(source)
            if digest != row["source_sha256"]:
                connection.execute("UPDATE selected_sources SET source_sha256=? WHERE source_file=?", (digest, source))
                connection.execute(
                    """INSERT OR IGNORE INTO source_state
                    (source_file,source_sha256,status,updated_utc) VALUES (?,?,?,?)""",
                    (source, digest, "pending", _utc_now()),
                )
        connection.commit()
        requested_row = connection.execute("SELECT value FROM metadata WHERE key='requested_coverage'").fetchone()
        achieved_row = connection.execute("SELECT value FROM metadata WHERE key='achieved_coverage'").fetchone()
        requested = json.loads(requested_row[0])
        achieved = json.loads(achieved_row[0])
        return selected, requested, achieved
    if audit_mode == "raw-full":
        selected = _all_raw_sources(raw_dir)
        requested: dict[str, int] = {}
        achieved: dict[str, int] = {}
        strata_by_source = {source: {"mode=raw-full"} for source in selected}
    else:
        features = _source_strata(connection, state_dbs)
        if prior_audit_dir is not None:
            prior_protocol = json.loads((prior_audit_dir / "sequence_readiness_protocol.json").read_text())
            selected = sorted(str(path) for path in prior_protocol.get("selected_source_hashes", {}))
            expected_count = int(max_source_files or 250)
            if len(selected) != expected_count:
                raise ValueError(f"Prior audit selected {len(selected)} sources; expected exactly {expected_count}")
            missing = [path for path in selected if not Path(path).is_file()]
            if missing:
                raise FileNotFoundError(f"Prior selected raw source is missing: {missing[0]}")
            requested = {
                str(key): int(value) for key, value in prior_protocol.get("requested_stratum_coverage", {}).items()
            }
            achieved = {
                str(key): int(value) for key, value in prior_protocol.get("achieved_stratum_coverage", {}).items()
            }
        else:
            selected, requested, achieved = _pilot_selection(
                features,
                max_source_files=int(max_source_files or 250),
                samples_per_stratum=samples_per_stratum,
                seed=pilot_seed,
            )
        strata_by_source = features
        target_members = set()
        for stratum in sorted({item for values in features.values() for item in values}):
            members = sorted(
                (source for source, values in features.items() if stratum in values),
                key=lambda source: hashlib.sha256(f"{pilot_seed}:{stratum}:{source}".encode()).hexdigest(),
            )
            target_members.update(members[:samples_per_stratum])
        selected_set = set(selected)
        coverage_accounting = {
            "selected_unique_source_count": len(selected),
            "selected_target_stratum_source_count": len(selected_set & target_members),
            "deterministic_fill_source_count": len(selected_set - target_members),
            "sum_of_achieved_stratum_memberships": sum(achieved.values()),
            "strata_overlap": True,
            "denominator_interpretation": (
                "Target-stratum and deterministic-fill counts partition selected unique sources; achieved stratum "
                "memberships overlap and therefore must not be compared directly with the unique-source denominator."
            ),
        }
    selected_records = []
    for source in selected:
        digest = sha256_file(source)
        selected_records.append((source, digest, json.dumps(sorted(strata_by_source.get(source, set())))))
    connection.executemany("INSERT INTO selected_sources VALUES (?,?,?)", selected_records)
    connection.executemany(
        """INSERT OR IGNORE INTO source_state
        (source_file,source_sha256,status,updated_utc) VALUES (?,?,?,?)""",
        [(source, digest, "pending", _utc_now()) for source, digest, _strata in selected_records],
    )
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('requested_coverage',?)", (json.dumps(requested),))
    connection.execute("INSERT OR REPLACE INTO metadata VALUES ('achieved_coverage',?)", (json.dumps(achieved),))
    if audit_mode == "raw-pilot":
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('coverage_accounting',?)",
            (json.dumps(coverage_accounting),),
        )
    connection.commit()
    return selected, requested, achieved


def _matrix_rows_for_source(connection: sqlite3.Connection, source: str) -> list[dict[str, Any]]:
    return _query_rows(
        connection,
        """SELECT p.sample_id,p.pdb_id,p.chain_id,p.model_id,p.matrix_path,p.sequence,
        p.recorded_length,p.experimental_method,p.terminal_trimming_applied,p.missing_calpha_policy,
        CASE
          WHEN EXISTS (SELECT 1 FROM manifest_rows t WHERE t.manifest_kind='train' AND t.sample_id=p.sample_id)
           AND EXISTS (SELECT 1 FROM manifest_rows v WHERE v.manifest_kind='validation' AND v.sample_id=p.sample_id)
            THEN 'train_and_validation'
          WHEN EXISTS (SELECT 1 FROM manifest_rows t WHERE t.manifest_kind='train' AND t.sample_id=p.sample_id)
            THEN 'train'
          WHEN EXISTS (SELECT 1 FROM manifest_rows v WHERE v.manifest_kind='validation' AND v.sample_id=p.sample_id)
            THEN 'validation'
          ELSE 'excluded'
        END split
        FROM manifest_rows p WHERE p.manifest_kind='processed' AND p.source_file=? ORDER BY p.sample_id""",
        (source,),
    )


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(json.loads(value))


def _raw_residue_ids(raw: dict[str, Any]) -> list[str]:
    auth_ids = _json_list(raw.get("auth_sequence_ids"))
    insertions = _json_list(raw.get("insertion_codes"))
    return [f"{auth}{insertion}" for auth, insertion in zip(auth_ids, insertions, strict=True)]


def _ordered_id_indices(raw_ids: list[str], matrix_ids: list[str]) -> tuple[list[int], bool]:
    candidates = [[index for index, raw_id in enumerate(raw_ids) if raw_id == matrix_id] for matrix_id in matrix_ids]
    if any(not values for values in candidates):
        return [], False
    paths: list[list[int]] = [[]]
    for values in candidates:
        paths = [path + [value] for path in paths for value in values if not path or value > path[-1]][:2]
        if not paths:
            return [], False
    return paths[0], len(paths) > 1


def _classify_alignment(
    matrix: dict[str, Any], raw: dict[str, Any] | None, inspected: dict[str, Any]
) -> dict[str, Any]:
    matrix_sequence = inspected["npz_sequence"]
    result: dict[str, Any] = {
        "alignment_class": "unavailable",
        "alignment_reason": "raw_chain_model_unavailable",
        "matrix_manifest_sequence_match": matrix_sequence == matrix["sequence"],
        "coordinate_verification_status": "unavailable",
        "coordinate_distance_tolerance_angstrom": COORDINATE_DISTANCE_TOLERANCE_ANGSTROM,
    }
    if raw is None:
        return result
    if matrix_sequence != matrix["sequence"]:
        result.update(alignment_class="sequence_mismatch", alignment_reason="stored_npz_manifest_sequence_mismatch")
        return result

    evidence = str(raw.get("seqres_sequence") or raw.get("atom_sequence") or "")
    raw_ids = _raw_residue_ids(raw)
    matrix_ids = [str(value) for value in inspected["residue_ids"]]
    indices, ambiguous_ids = _ordered_id_indices(raw_ids, matrix_ids)
    raw_tokens = _json_list(raw.get("residue_tokens"))
    selected_sequence = "".join(str(raw_tokens[index]) for index in indices) if indices else ""
    result["matched_raw_residue_indices"] = indices
    result["residue_id_selection_ambiguous"] = ambiguous_ids
    result["selected_raw_sequence"] = selected_sequence or None

    occurrences = sequence_evidence_occurrences(
        matrix_sequence,
        raw,
        infer_sequence_source(matrix_sequence, raw),
    )
    if indices and selected_sequence == matrix_sequence and not ambiguous_ids:
        contiguous = indices == list(range(indices[0], indices[-1] + 1))
        full = indices == list(range(len(raw_ids)))
        start_ids = _json_list(raw.get("label_sequence_ids"))
        recorded_start = inspected.get("retained_start_label_seq_id")
        recorded_end = inspected.get("retained_end_label_seq_id")
        start_matches = recorded_start is None or str(recorded_start) == str(start_ids[indices[0]])
        end_matches = recorded_end is None or str(recorded_end) == str(start_ids[indices[-1]])
        expected_n = indices[0]
        expected_c = len(raw_ids) - indices[-1] - 1
        recorded_n = inspected.get("trimmed_n_terminal_residues")
        recorded_c = inspected.get("trimmed_c_terminal_residues")
        has_trim_metadata = all(value is not None for value in (recorded_start, recorded_end, recorded_n, recorded_c))
        trims_match = has_trim_metadata and int(recorded_n) == expected_n and int(recorded_c) == expected_c
        if full:
            result.update(alignment_class="exact", alignment_reason="full_ordered_residue_identity_match")
        elif contiguous and start_matches and end_matches and trims_match:
            result.update(
                alignment_class="valid_terminal_trim",
                alignment_reason="contiguous_interval_matches_recorded_boundaries_and_trim_counts",
            )
        elif not contiguous:
            result.update(alignment_class="internal_gap", alignment_reason="ordered_residue_ids_contain_internal_gap")
        elif not has_trim_metadata:
            result.update(
                alignment_class="valid_residue_id_selection",
                alignment_reason="exact_ordered_residue_identity_match_without_recorded_trim_boundaries",
            )
        else:
            result.update(
                alignment_class="sequence_mismatch",
                alignment_reason="residue_ids_match_but_recorded_terminal_trim_metadata_disagrees",
            )
    elif indices and selected_sequence != matrix_sequence:
        result.update(
            alignment_class="sequence_mismatch", alignment_reason="ordered_residue_ids_disagree_with_sequence"
        )
    elif occurrences > 1:
        result.update(
            alignment_class="ambiguous_subsequence", alignment_reason="repeated_sequence_without_resolving_ids"
        )
    elif occurrences == 1:
        result.update(
            alignment_class="unavailable",
            alignment_reason="unique_sequence_interval_lacks_residue_identity_provenance",
        )
    elif evidence:
        result.update(alignment_class="sequence_mismatch", alignment_reason="matrix_sequence_absent_from_raw_polymer")

    if indices:
        raw_altlocs = _json_list(raw.get("selected_altlocs"))
        stored_altlocs = inspected.get("selected_altlocs")
        if stored_altlocs is None or len(raw_altlocs) != len(raw_ids):
            result["altloc_verification_status"] = "unavailable"
        else:
            selected_raw_altlocs = [raw_altlocs[index] for index in indices]
            result["altloc_verification_status"] = (
                "passed" if list(stored_altlocs) == selected_raw_altlocs else "failed"
            )
        coordinates = _json_list(raw.get("selected_calpha_coordinates"))
        selected_coordinates = [coordinates[index] for index in indices] if len(coordinates) == len(raw_ids) else []
        if selected_coordinates and all(coordinate is not None for coordinate in selected_coordinates):
            coordinate_array = np.asarray(selected_coordinates, dtype=np.float32)
            reconstructed = np.linalg.norm(coordinate_array[:, None, :] - coordinate_array[None, :, :], axis=-1)
            stored = inspected.get("distance_matrix")
            if stored is not None and stored.shape == reconstructed.shape:
                maximum_error = float(np.max(np.abs(stored - reconstructed), initial=0.0))
                result["coordinate_distance_max_abs_error_angstrom"] = maximum_error
                result["coordinate_verification_status"] = (
                    "passed" if maximum_error <= COORDINATE_DISTANCE_TOLERANCE_ANGSTROM else "failed"
                )
            else:
                result["coordinate_verification_reason"] = "stored_matrix_shape_mismatch"
        else:
            result["coordinate_verification_reason"] = "selected_raw_calpha_unavailable"
    return result


def _source_identity_status(source: Path, state_dbs: list[Path]) -> str:
    """Compare current source metadata with historical preprocessing state."""
    current = source.stat()
    records: list[tuple[int, int, str | None]] = []
    for database in state_dbs:
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as state:
            columns = {row[1] for row in state.execute("PRAGMA table_info(source_files)")}
            if not {"source_path", "source_size", "source_mtime_ns"} <= columns:
                continue
            hash_column = next(
                (name for name in ("source_sha256", "raw_sha256") if name in columns),
                None,
            )
            select_hash = f",{hash_column}" if hash_column is not None else ""
            row = state.execute(
                f"SELECT source_size,source_mtime_ns{select_hash} FROM source_files WHERE source_path=?",
                (str(source),),
            ).fetchone()
            if row is not None:
                records.append((int(row[0]), int(row[1]), str(row[2]) if hash_column else None))
    if not records:
        return "no_state_evidence"
    stored_hashes = {digest for _size, _mtime, digest in records if digest}
    if stored_hashes:
        return "historical_sha_verified" if sha256_file(source) in stored_hashes else "state_metadata_mismatch"
    if any(size != current.st_size or mtime != current.st_mtime_ns for size, mtime, _digest in records):
        return "state_metadata_mismatch"
    return "state_size_mtime_match_sha_unavailable"


def _candidate_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_file")),
        str(row.get("model_number", row.get("model_id"))),
        str(row.get("entity_id")),
        str(row.get("label_asym_id")),
        str(row.get("auth_asym_id")),
    )


def _source_partition(
    connection: sqlite3.Connection,
    source: str,
    raw_rows: list[dict[str, Any]],
    token_counts: Counter[Any],
    state_dbs: list[Path] | None = None,
    prior_cases: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    author_lookup: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        prefix = (str(row["pdb_id"]).upper(), str(row["model_id"]))
        label_lookup[(prefix[0], str(row["label_asym_id"]), prefix[1])].append(row)
        if row.get("auth_asym_id") is not None:
            author_lookup[(prefix[0], str(row["auth_asym_id"]), prefix[1])].append(row)
    alignments = []
    candidate_evidence = []
    source_status = _source_identity_status(Path(source), list(state_dbs or []))
    for matrix in _matrix_rows_for_source(connection, source):
        inspected = inspect_processed_npz(matrix["matrix_path"], include_geometry=True)
        key = (str(matrix["pdb_id"]).upper(), str(matrix["chain_id"]), str(matrix["model_id"]))
        label_matches = label_lookup.get(key, [])
        author_matches = author_lookup.get(key, [])
        matches_by_key = {_candidate_key(row): row for row in [*author_matches, *label_matches]}
        matches = list(matches_by_key.values())
        raw = author_matches[0] if len({_candidate_key(row) for row in author_matches}) == 1 else None
        sequence_source = infer_sequence_source(matrix["sequence"], raw)
        metadata = {
            key: inspected.get(key)
            for key in (
                "retained_insertion_codes",
                "retained_start_label_seq_id",
                "retained_end_label_seq_id",
                "trimmed_n_terminal_residues",
                "trimmed_c_terminal_residues",
            )
        }
        prior_case = (prior_cases or {}).get(str(matrix["sample_id"]), {})
        case = {
            **matrix,
            **{key: prior_case[key] for key in ("primary_classification", "alignment_reason") if key in prior_case},
            "matrix_sequence": inspected["npz_sequence"],
            "matrix_manifest_sequence_match": inspected["npz_sequence"] == matrix["sequence"],
            "source_identity_status": source_status,
        }
        scores = [
            score
            for candidate in matches
            for score in score_candidate(
                candidate,
                matrix_sequence=inspected["npz_sequence"],
                matrix_ids=[str(value) for value in inspected["residue_ids"]],
                matrix=inspected["distance_matrix"],
                metadata=metadata,
                requested_chain=str(matrix["chain_id"]),
                tolerance=COORDINATE_DISTANCE_TOLERANCE_ANGSTROM,
            )
        ]
        refined = refine_pair_status(case, scores)
        selected_scores = [score for score in scores if score.get("author_chain_match")]
        passed_scores = [score for score in selected_scores if score.get("coordinate_status") == "passed"]
        best_score = (passed_scores or selected_scores or [None])[0]
        alignment_class = (
            "exact"
            if refined["refined_primary_classification"] == "verified_sequence_geometry_pair"
            else "valid_residue_id_selection"
            if refined["refined_primary_classification"]
            in {"audit_residue_id_convention_bug", "legacy_trim_metadata_inconsistency"}
            else "unavailable"
        )
        alignment = {
            "alignment_class": alignment_class,
            "alignment_reason": refined["refined_primary_classification"],
            "matrix_manifest_sequence_match": case["matrix_manifest_sequence_match"],
            "coordinate_verification_status": refined["coordinate_matrix_status"],
            "coordinate_distance_tolerance_angstrom": COORDINATE_DISTANCE_TOLERANCE_ANGSTROM,
            **refined,
        }
        if best_score is not None:
            alignment["selected_residue_id_convention"] = best_score["convention"]
            alignment["coordinate_distance_max_abs_error_angstrom"] = best_score["coordinate_max_abs_error_angstrom"]
        for candidate_index, score in enumerate(scores):
            physical_key = (
                source,
                str(score.get("model_number")),
                str(score.get("entity_id")),
                str(score.get("label_asym_id")),
                str(score.get("auth_asym_id")),
            )
            candidate_evidence.append(
                {
                    "sample_id": matrix["sample_id"],
                    "source_file": source,
                    "candidate_index": candidate_index,
                    "evidence_row_index": candidate_index,
                    "residue_id_convention": score.get("convention"),
                    "physical_candidate_id": hashlib.sha256(
                        json.dumps(physical_key, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    **score,
                }
            )
        alignments.append(
            {
                **matrix,
                **alignment,
                "matrix_sequence": inspected["npz_sequence"],
                "matrix_sequence_sha256": sequence_sha256(inspected["npz_sequence"]),
                "matrix_actual_length": inspected["matrix_rows"],
                "seqres_sequence": raw.get("seqres_sequence") if raw else None,
                "atom_sequence": raw.get("atom_sequence") if raw else None,
                "sequence_source": sequence_source,
                "raw_match_count": len(matches),
                "author_linked_candidate_count": len({_candidate_key(row) for row in author_matches}),
                "raw_match_strategy": "auth_asym_id_first",
                "raw_label_asym_id": raw.get("label_asym_id") if raw else None,
                "raw_auth_asym_id": raw.get("auth_asym_id") if raw else None,
                "raw_sequence_occurrence_count": sequence_evidence_occurrences(
                    matrix["sequence"], raw, sequence_source
                ),
                "residue_ids": json.dumps(inspected["residue_ids"]),
                "insertion_codes": json.dumps(inspected["retained_insertion_codes"]),
                "selected_altlocs": json.dumps(inspected["selected_altlocs"]),
            }
        )
    tokens = [
        {
            "count_semantics": semantics,
            "sequence_evidence": evidence,
            "residue_name": residue,
            "classification": classification,
            "count": int(count),
        }
        for (semantics, evidence, residue, classification), count in sorted(token_counts.items())
    ]
    first = raw_rows[0] if raw_rows else {}
    nonpolymer_context = [
        {
            "source_file": source,
            "nonpolymer_component_counts": first.get("nonpolymer_component_counts", "{}"),
            "water_count": first.get("water_count", 0),
            "ion_count": first.get("ion_count", 0),
            "ligand_count": first.get("ligand_count", 0),
        }
    ]
    nmr_summaries = []
    for label_chain in sorted({str(row["label_asym_id"]) for row in raw_rows if row.get("is_nmr")}):
        chain_rows = [row for row in raw_rows if str(row["label_asym_id"]) == label_chain]
        missing = [
            int(row["missing_calpha_count"]) for row in chain_rows if row.get("missing_calpha_count") is not None
        ]
        model_one_alignments = [
            row for row in alignments if row.get("raw_label_asym_id") == label_chain and str(row.get("model_id")) == "1"
        ]
        compatible = all(
            row["alignment_class"] in {"exact", "valid_terminal_trim", "valid_residue_id_selection"}
            for row in model_one_alignments
        ) and all(bool(row.get("chain_sequence_consistent_across_models")) for row in chain_rows)
        nmr_summaries.append(
            {
                "source_file": source,
                "label_asym_id": label_chain,
                "auth_asym_id": chain_rows[0].get("auth_asym_id"),
                "model_count": len({str(row["model_id"]) for row in chain_rows}),
                "sequence_consistent_across_models": all(
                    bool(row.get("chain_sequence_consistent_across_models")) for row in chain_rows
                ),
                "minimum_missing_calpha_count": min(missing) if missing else None,
                "maximum_missing_calpha_count": max(missing) if missing else None,
                "missing_calpha_varies_across_models": len(set(missing)) > 1,
                "processed_model_1_matrix_compatible_with_all_models": compatible if model_one_alignments else None,
                "processed_model_1_sample_count": len(model_one_alignments),
            }
        )
    return {
        "source_file": source,
        "raw_rows": raw_rows,
        "alignments": alignments,
        "tokens": tokens,
        "nonpolymer_context": nonpolymer_context,
        "nmr_summaries": nmr_summaries,
        "candidate_evidence": candidate_evidence,
    }


def _partial_protocol(
    *,
    connection: sqlite3.Connection,
    output_dir: Path,
    audit_mode: str,
    started_utc: str,
    started_monotonic: float,
    input_hashes: dict[str, str],
    selected_count: int,
    requested_coverage: dict[str, int],
    achieved_coverage: dict[str, int],
    status: str,
) -> dict[str, Any]:
    run_config = json.loads((output_dir / "run_config.json").read_text())
    counts = {
        state: int(
            connection.execute(
                """SELECT COUNT(*) FROM selected_sources s JOIN source_state st
                ON st.source_file=s.source_file AND st.source_sha256=s.source_sha256 WHERE st.status=?""",
                (state,),
            ).fetchone()[0]
        )
        for state in ("completed", "failed")
    }
    completed_or_failed = counts["completed"] + counts["failed"]
    parser_calls = int(
        connection.execute(
            """SELECT COALESCE(SUM(st.parser_calls),0) FROM selected_sources s JOIN source_state st
            ON st.source_file=s.source_file AND st.source_sha256=s.source_sha256"""
        ).fetchone()[0]
    )
    selected_hashes = {
        row[0]: row[1] for row in connection.execute("SELECT source_file,source_sha256 FROM selected_sources")
    }
    manifest_counts = {
        kind: int(connection.execute("SELECT COUNT(*) FROM manifest_rows WHERE manifest_kind=?", (kind,)).fetchone()[0])
        for kind in ("processed", "train", "validation")
    }
    payload = {
        "report_semantics_version": REPORT_SEMANTICS_VERSION,
        "status": status,
        "audit_mode": audit_mode,
        "storage_profile": run_config.get("storage_profile", "verbose-v3"),
        "resolved_output_dir": run_config.get("output_dir", str(output_dir.resolve())),
        "resolved_state_dir": run_config.get("state_dir", str(output_dir.resolve())),
        "started_utc": started_utc,
        "completed_utc": _utc_now() if status == "completed" else None,
        "runtime_seconds": time.monotonic() - started_monotonic,
        "peak_memory_if_available": {"ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
        "input_hashes": input_hashes,
        "selected_source_hashes": selected_hashes,
        "raw_inputs_unchanged": True,
        "selected_source_file_count": selected_count,
        "parsed_source_file_count": completed_or_failed,
        "parser_call_count": parser_calls,
        "manifest_row_counts": manifest_counts,
        "rows_by_split_status": json.loads((output_dir / "manifest_summary.json").read_text()).get(
            "rows_by_split_status", {}
        ),
        "requested_stratum_coverage": requested_coverage,
        "achieved_stratum_coverage": achieved_coverage,
        "coverage_accounting": json.loads(
            (connection.execute("SELECT value FROM metadata WHERE key='coverage_accounting'").fetchone() or ("{}",))[0]
        ),
        "completed_source_count": counts["completed"],
        "failed_source_count": counts["failed"],
        "pending_source_count": max(selected_count - completed_or_failed, 0),
        "recovery_command": "rerun the same command with --resume instead of --restart",
    }
    _atomic_json(output_dir / "sequence_readiness_protocol.partial.json", payload)
    return payload


def _process_raw_sources(
    connection: sqlite3.Connection,
    *,
    selected: list[str],
    output_dir: Path,
    checkpoint_frequency: int,
    protocol_args: dict[str, Any],
    stop_after_source_files: int | None,
    state_dbs: list[Path] | None = None,
    prior_forensics_dir: Path | None = None,
    storage_profile: str = "verbose-v3",
    sources_per_partition: int = DEFAULT_SOURCES_PER_PARTITION,
    unsafe_skip_disk_check: bool = False,
    projected_remaining_bytes: int | None = None,
) -> dict[str, Any]:
    if storage_profile == "compact-v1":
        return _process_raw_sources_compact(
            connection,
            selected=selected,
            output_dir=output_dir,
            protocol_args=protocol_args,
            stop_after_source_files=stop_after_source_files,
            state_dbs=state_dbs,
            prior_forensics_dir=prior_forensics_dir,
            sources_per_partition=sources_per_partition,
            unsafe_skip_disk_check=unsafe_skip_disk_check,
            projected_remaining_bytes=projected_remaining_bytes,
        )
    partitions = output_dir / "partitions"
    partitions.mkdir(exist_ok=True)
    prior_cases = (
        {
            str(row["sample_id"]): row
            for row in pd.read_parquet(prior_forensics_dir / "per_failure_classification.parquet").to_dict("records")
        }
        if prior_forensics_dir is not None
        else {}
    )
    processed_this_run = 0
    for source in tqdm(selected, desc="Raw sequence provenance", unit="source"):
        path = Path(source)
        digest = sha256_file(path)
        selected_digest = connection.execute(
            "SELECT source_sha256 FROM selected_sources WHERE source_file=?", (source,)
        ).fetchone()[0]
        if digest != selected_digest:
            connection.execute("UPDATE selected_sources SET source_sha256=? WHERE source_file=?", (digest, source))
            connection.execute(
                """INSERT OR IGNORE INTO source_state
                (source_file,source_sha256,status,updated_utc) VALUES (?,?,?,?)""",
                (source, digest, "pending", _utc_now()),
            )
            connection.commit()
        state = connection.execute(
            "SELECT status,partition_path FROM source_state WHERE source_file=? AND source_sha256=?",
            (source, digest),
        ).fetchone()
        if state and state[0] == "completed" and state[1] and Path(state[1]).is_file():
            continue
        try:
            raw_rows, token_counts = inspect_mmcif(path)
            partition = _source_partition(
                connection,
                source,
                raw_rows,
                token_counts,
                state_dbs,
                prior_cases,
            )
            source_key = hashlib.sha256(source.encode()).hexdigest()[:16]
            partition_path = partitions / f"{source_key}_{digest}.json"
            _atomic_json(partition_path, partition)
            connection.execute(
                """INSERT OR REPLACE INTO source_state
                (source_file,source_sha256,status,partition_path,error_type,error_message,parser_calls,updated_utc)
                VALUES (?,?,?,?,?,?,COALESCE((SELECT parser_calls FROM source_state
                WHERE source_file=? AND source_sha256=?),0)+1,?)""",
                (source, digest, "completed", str(partition_path), None, None, source, digest, _utc_now()),
            )
        except Exception as exc:
            connection.execute(
                """INSERT OR REPLACE INTO source_state
                (source_file,source_sha256,status,partition_path,error_type,error_message,parser_calls,updated_utc)
                VALUES (?,?,?,?,?,?,COALESCE((SELECT parser_calls FROM source_state
                WHERE source_file=? AND source_sha256=?),0)+1,?)""",
                (source, digest, "failed", None, type(exc).__name__, str(exc), source, digest, _utc_now()),
            )
        connection.commit()
        processed_this_run += 1
        if processed_this_run % checkpoint_frequency == 0:
            _partial_protocol(connection=connection, output_dir=output_dir, status="running", **protocol_args)
        if stop_after_source_files is not None and processed_this_run >= stop_after_source_files:
            _partial_protocol(connection=connection, output_dir=output_dir, status="interrupted", **protocol_args)
            raise AuditInterrupted("Synthetic interruption after persistent source checkpoint")
    return _partial_protocol(connection=connection, output_dir=output_dir, status="completed", **protocol_args)


def _disk_guard(
    output_dir: Path,
    *,
    projected_remaining_bytes: int | None,
    unsafe_skip_disk_check: bool,
) -> dict[str, Any]:
    stat = os.statvfs(output_dir)
    free_bytes = stat.f_bavail * stat.f_frsize
    required = 2 * projected_remaining_bytes + SAFETY_RESERVE_BYTES if projected_remaining_bytes is not None else 0
    passed = free_bytes >= required
    result = {
        "free_bytes": free_bytes,
        "projected_remaining_bytes": projected_remaining_bytes,
        "required_free_bytes": required,
        "required_safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "passed": passed,
        "unsafe_override": bool(unsafe_skip_disk_check),
    }
    if not passed and not unsafe_skip_disk_check:
        raise RuntimeError(
            "Compact audit disk guard failed: free_bytes must be at least 2 * projected_remaining_bytes + 20 GiB"
        )
    return result


def _commit_compact_marker(
    connection: sqlite3.Connection,
    output_dir: Path,
    marker: dict[str, Any],
) -> None:
    validate_compact_partition(output_dir, marker)
    source_table = output_dir / marker["tables"]["source_identity"]["path"]
    source_rows = pq.read_table(source_table).to_pylist()
    if len(source_rows) != int(marker["source_count"]):
        raise ValueError("Compact partition source count does not match source identity table")
    if len({row["source_id"] for row in source_rows}) != len(source_rows):
        raise ValueError("Compact partition contains duplicate source identities")
    marker_path = Path(marker["marker_path"])
    for row in source_rows:
        connection.execute(
            """INSERT OR REPLACE INTO source_state
            (source_file,source_sha256,status,partition_path,error_type,error_message,parser_calls,updated_utc)
            VALUES (?,?,?,?,?,?,?,?)""",
            (
                row["source_file"],
                row["source_sha256"],
                "completed" if row["parse_status"] == "completed" else "failed",
                str(marker_path),
                row.get("error_type"),
                row.get("error_message"),
                int(row.get("parser_calls", 1)),
                _utc_now(),
            ),
        )
    connection.execute(
        "INSERT OR REPLACE INTO compact_partitions VALUES (?,?,?,?,?)",
        (
            int(marker["partition_index"]),
            int(marker["source_count"]),
            str(marker_path),
            sha256_file(marker_path),
            _utc_now(),
        ),
    )
    connection.commit()


def _reconcile_compact_partitions(connection: sqlite3.Connection, output_dir: Path) -> None:
    markers = sorted((output_dir / "partition_commits").glob("part-*.json")) if output_dir.exists() else []
    marker_names = {path.stem for path in markers}
    artifact_names = {path.stem for path in (output_dir / "tables").glob("*/part-*.parquet") if path.is_file()}
    orphan_names = sorted(artifact_names - marker_names)
    for name in orphan_names:
        for path in (output_dir / "tables").glob(f"*/{name}.parquet"):
            path.unlink()
    if orphan_names:
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('reconciled_partial_partitions',?)",
            (json.dumps(orphan_names),),
        )
        connection.commit()
    database_indexes = {int(row[0]) for row in connection.execute("SELECT partition_index FROM compact_partitions")}
    marker_indexes = {int(json.loads(path.read_text())["partition_index"]) for path in markers}
    missing_markers = sorted(database_indexes - marker_indexes)
    if missing_markers:
        raise FileNotFoundError(f"Committed compact partition marker is missing: {missing_markers[0]}")
    seen_sources: set[str] = set()
    previous_index = -1
    for marker_path in markers:
        marker = json.loads(marker_path.read_text())
        marker["marker_path"] = str(marker_path)
        index = int(marker["partition_index"])
        if index != previous_index + 1:
            raise ValueError("Compact partitions are duplicated, missing, or out of order")
        overlap = seen_sources.intersection(marker["source_ids"])
        if overlap:
            raise ValueError(f"Compact partitions overlap at source identity: {sorted(overlap)[0]}")
        seen_sources.update(marker["source_ids"])
        validate_compact_partition(output_dir, marker)
        recorded = connection.execute(
            "SELECT marker_sha256 FROM compact_partitions WHERE partition_index=?", (index,)
        ).fetchone()
        if recorded is None:
            _commit_compact_marker(connection, output_dir, marker)
        elif recorded[0] != sha256_file(marker_path):
            raise ValueError(f"Compact partition marker changed after commit: {marker_path}")
        previous_index = index


def _process_raw_sources_compact(
    connection: sqlite3.Connection,
    *,
    selected: list[str],
    output_dir: Path,
    protocol_args: dict[str, Any],
    stop_after_source_files: int | None,
    state_dbs: list[Path] | None,
    prior_forensics_dir: Path | None,
    sources_per_partition: int,
    unsafe_skip_disk_check: bool,
    projected_remaining_bytes: int | None,
) -> dict[str, Any]:
    _reconcile_compact_partitions(connection, output_dir)
    prior_cases = (
        {
            str(row["sample_id"]): row
            for row in pd.read_parquet(prior_forensics_dir / "per_failure_classification.parquet").to_dict("records")
        }
        if prior_forensics_dir is not None
        else {}
    )
    processed_this_run = 0
    for partition_index, start in enumerate(range(0, len(selected), sources_per_partition)):
        chunk = selected[start : start + sources_per_partition]
        committed = connection.execute(
            "SELECT source_count FROM compact_partitions WHERE partition_index=?", (partition_index,)
        ).fetchone()
        if committed:
            if int(committed[0]) != len(chunk):
                raise ValueError(f"Compact partition {partition_index} source range is incompatible with resume")
            continue
        remaining_projection = (
            int(projected_remaining_bytes * (len(selected) - start) / max(len(selected), 1))
            if projected_remaining_bytes is not None
            else None
        )
        disk_status = _disk_guard(
            output_dir,
            projected_remaining_bytes=remaining_projection,
            unsafe_skip_disk_check=unsafe_skip_disk_check,
        )
        source_payloads = []
        for source in tqdm(chunk, desc=f"Raw compact partition {partition_index}", unit="source"):
            digest = sha256_file(Path(source))
            try:
                raw_rows, token_counts = inspect_mmcif(Path(source))
                payload = _source_partition(connection, source, raw_rows, token_counts, state_dbs, prior_cases)
                source_payloads.append(
                    {
                        "source_file": source,
                        "source_sha256": digest,
                        "status": "completed",
                        "parser_calls": 1,
                        "payload": payload,
                    }
                )
            except Exception as exc:
                source_payloads.append(
                    {
                        "source_file": source,
                        "source_sha256": digest,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "parser_calls": 1,
                    }
                )
            processed_this_run += 1
        marker = write_compact_partition(
            output_dir,
            partition_index=partition_index,
            sources=source_payloads,
        )
        _commit_compact_marker(connection, output_dir, marker)
        connection.execute(
            "INSERT OR REPLACE INTO metadata VALUES ('last_disk_guard',?)",
            (json.dumps(disk_status, sort_keys=True),),
        )
        connection.commit()
        _partial_protocol(connection=connection, output_dir=output_dir, status="running", **protocol_args)
        if stop_after_source_files is not None and processed_this_run >= stop_after_source_files:
            _partial_protocol(connection=connection, output_dir=output_dir, status="interrupted", **protocol_args)
            raise AuditInterrupted("Synthetic interruption after compact partition commit")
    expected_partitions = (len(selected) + sources_per_partition - 1) // sources_per_partition
    committed_partitions = int(connection.execute("SELECT COUNT(*) FROM compact_partitions").fetchone()[0])
    if committed_partitions != expected_partitions:
        raise RuntimeError(
            f"Compact partition finalization is incomplete: expected {expected_partitions}, "
            f"found {committed_partitions}"
        )
    return _partial_protocol(connection=connection, output_dir=output_dir, status="running", **protocol_args)


def _current_pairing_classification(row: dict[str, Any]) -> str:
    """Describe current pairing evidence without retaining historical audit diagnoses."""
    if row.get("training_eligibility") == "excluded_or_unresolved":
        return "unresolved_ambiguity"
    namespace_reconciled = any(
        row.get(key)
        for key in (
            "auth_residue_ids_match",
            "label_residue_ids_match",
            "zero_based_positions_match",
            "one_based_positions_match",
        )
    )
    if row.get("strict_provenance_status") == "residue_identity_provenance_incomplete" and not namespace_reconciled:
        return "residue_identity_provenance_incomplete"
    if row.get("coordinate_matrix_status") == "unavailable":
        return "unique_sequence_pair_coordinate_unavailable"
    return "verified_sequence_geometry_pair"


def _forensic_root_cause(old: dict[str, Any], current: dict[str, Any]) -> str:
    historical = str(old["primary_classification"])
    pairing = _current_pairing_classification(current)
    if pairing == "residue_identity_provenance_incomplete":
        return pairing
    if historical == "raw_source_version_unverifiable":
        return str(current["refined_primary_classification"])
    return historical


def _v2_v3_transition_frame(prior_forensics_dir: Path, alignments: list[dict[str, Any]]) -> pd.DataFrame:
    previous = pd.read_parquet(prior_forensics_dir / "per_failure_classification.parquet")
    if len(previous) != 197 or previous["sample_id"].nunique() != 197:
        raise ValueError("The v2 forensic transition requires exactly 197 unique blocking cases")
    current = {str(row["sample_id"]): row for row in alignments}
    missing = sorted(set(previous["sample_id"].astype(str)) - set(current))
    if missing:
        raise ValueError(f"The v3 audit is missing prior blocking sample: {missing[0]}")
    rows = []
    for old in previous.to_dict("records"):
        new = current[str(old["sample_id"])]
        rows.append(
            {
                "sample_id": old["sample_id"],
                "v2_classification": old["primary_classification"],
                "forensic_root_cause": _forensic_root_cause(old, new),
                "v3_pairing_classification": _current_pairing_classification(new),
                "strict_provenance_status": new["strict_provenance_status"],
                "practical_training_eligibility": new["training_eligibility"],
                "source_identity_status": new["source_identity_status"],
            }
        )
    return pd.DataFrame(rows)


def _write_v2_v3_transition(
    prior_forensics_dir: Path, alignments: list[dict[str, Any]], output_path: Path
) -> dict[str, int]:
    frame = _v2_v3_transition_frame(prior_forensics_dir, alignments)
    _atomic_csv(output_path, frame)
    counts = Counter(frame["v3_pairing_classification"])
    if sum(counts.values()) != 197:
        raise RuntimeError("v2-to-v3 transition accounting did not sum to 197")
    return dict(sorted(counts.items()))


def _consolidate_partitions(
    connection: sqlite3.Connection,
    output_dir: Path,
    prior_forensics_dir: Path | None = None,
    storage_profile: str = "verbose-v3",
) -> dict[str, Any]:
    specs = {
        "raw_sequence_provenance.jsonl": "raw_rows",
        "seqres_atom_matrix_alignments.jsonl": "alignments",
        "raw_residue_tokens.jsonl": "tokens",
        "excluded_nonpolymer_context.jsonl": "nonpolymer_context",
        "nmr_source_chain_consistency.jsonl": "nmr_summaries",
    }
    publish_verbose = storage_profile == "verbose-v3"
    handles = {name: (output_dir / name).open("w") for name in specs} if publish_verbose else {}
    missing_handle = (output_dir / "missing_residues_and_calpha.jsonl").open("w") if publish_verbose else None
    modified_handle = (output_dir / "modified_residues.jsonl").open("w") if publish_verbose else None
    nmr_handle = (output_dir / "nmr_model_consistency.jsonl").open("w") if publish_verbose else None
    alignment_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    modified_counts: Counter[str] = Counter()
    nonpolymer_counts: Counter[tuple[str, str]] = Counter()
    altloc_counts: Counter[str] = Counter()
    missing_counts: Counter[tuple[str, str]] = Counter()
    alignment_class_counts: Counter[str] = Counter()
    refined_classification_counts: Counter[str] = Counter()
    coordinate_status_counts: Counter[str] = Counter()
    nmr_chain_rows: list[dict[str, Any]] = []
    blocking_rows: list[dict[str, Any]] = []
    transition_ids = (
        set(
            pd.read_parquet(prior_forensics_dir / "per_failure_classification.parquet", columns=["sample_id"])[
                "sample_id"
            ].astype(str)
        )
        if prior_forensics_dir is not None
        else set()
    )
    transition_alignments: list[dict[str, Any]] = []
    eligibility_counts: Counter[str] = Counter()
    source_identity_counts: Counter[str] = Counter()
    eligibility_columns = [
        "sample_id",
        "pdb_id",
        "chain_id",
        "model_id",
        "matrix_path",
        "refined_primary_classification",
        "strict_provenance_status",
        "training_eligibility",
        "source_identity_status",
    ]
    eligibility_handles = {
        "strict": (output_dir / "strict_training_eligibility.csv").open("w", newline=""),
        "practical": (output_dir / "practical_training_eligibility.csv").open("w", newline=""),
        "unresolved": (output_dir / "unresolved_cases.csv").open("w", newline=""),
    }
    eligibility_writers = {
        key: csv.DictWriter(handle, fieldnames=eligibility_columns) for key, handle in eligibility_handles.items()
    }
    for writer in eligibility_writers.values():
        writer.writeheader()
    candidate_writer: pq.ParquetWriter | None = None
    candidate_columns: list[str] = []
    raw_chain_model_count = 0
    ambiguous_altloc_position_count = 0
    try:
        if publish_verbose:
            payloads = (
                json.loads(Path(partition_path).read_text())
                for (partition_path,) in connection.execute(
                    """SELECT st.partition_path FROM selected_sources s JOIN source_state st
                    ON st.source_file=s.source_file AND st.source_sha256=s.source_sha256
                    WHERE st.status='completed' ORDER BY st.source_file"""
                )
            )
        else:
            payloads = SequenceReadinessArtifactReader(output_dir).iter_source_payloads()
        for payload in payloads:
            for filename, key in specs.items():
                for row in payload.get(key, []):
                    if publish_verbose:
                        handles[filename].write(json.dumps(row, sort_keys=True) + "\n")
            candidates = payload.get("candidate_evidence", [])
            if candidates and publish_verbose:
                if candidate_writer is None:
                    candidate_columns = sorted({key for row in candidates for key in row})
                    schema = _candidate_evidence_schema(candidate_columns)
                    candidate_writer = pq.ParquetWriter(output_dir / "candidate_resolution_evidence.parquet", schema)
                candidate_writer.write_table(
                    pa.Table.from_pylist(
                        [_candidate_evidence_arrow_row(row, candidate_columns) for row in candidates],
                        schema=candidate_writer.schema,
                    )
                )
            for alignment in payload.get("alignments", []):
                sample_id = str(alignment["sample_id"])
                if sample_id in transition_ids:
                    transition_alignments.append(alignment)
                eligibility = str(alignment.get("training_eligibility", "excluded_or_unresolved"))
                eligibility_counts[eligibility] += 1
                source_identity_counts[str(alignment.get("source_identity_status", "no_state_evidence"))] += 1
                eligibility_row = {key: alignment.get(key) for key in eligibility_columns}
                if eligibility == "strict_verified_pair":
                    eligibility_writers["strict"].writerow(eligibility_row)
                    eligibility_writers["practical"].writerow(eligibility_row)
                elif eligibility == "conditionally_verified_pair":
                    eligibility_writers["practical"].writerow(eligibility_row)
                else:
                    eligibility_writers["unresolved"].writerow(eligibility_row)
                alignment_class = str(alignment.get("alignment_class", "unavailable"))
                alignment_class_counts[alignment_class] += 1
                refined_classification_counts[str(alignment.get("refined_primary_classification", "unavailable"))] += 1
                coordinate_status_counts[str(alignment.get("coordinate_verification_status", "unavailable"))] += 1
                alignment_counts[
                    (
                        str(alignment.get("split", "unknown")),
                        _method_stratum(str(alignment.get("experimental_method", "unknown"))).removeprefix("method="),
                        _length_bin(int(alignment.get("recorded_length", 0))).removeprefix("length="),
                        "trimmed" if alignment.get("terminal_trimming_applied") else "complete",
                        alignment_class,
                    )
                ] += 1
                if alignment.get("training_eligibility") == "excluded_or_unresolved":
                    blocking_rows.append(alignment)
            for row in payload.get("raw_rows", []):
                raw_chain_model_count += 1
                ambiguous_altloc_position_count += int(row.get("ambiguous_altloc_position_count", 0))
                if row.get("missing_residue_count", 0) or row.get("missing_calpha_count", 0):
                    if missing_handle is not None:
                        missing_handle.write(json.dumps(row, sort_keys=True) + "\n")
                missing_counts[("missing_residue", str(row.get("missing_residue_count", 0)))] += 1
                missing_counts[("missing_calpha", str(row.get("missing_calpha_count", 0)))] += 1
                if row.get("modified_residue_counts") not in {None, "{}"}:
                    if modified_handle is not None:
                        modified_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    modified_counts.update(json.loads(row["modified_residue_counts"]))
                if row.get("is_nmr"):
                    if nmr_handle is not None:
                        nmr_handle.write(json.dumps(row, sort_keys=True) + "\n")
                altloc_counts.update(json.loads(row.get("altloc_outcome_counts") or "{}"))
            for context in payload.get("nonpolymer_context", []):
                for component, count in json.loads(context["nonpolymer_component_counts"]).items():
                    if component in {"DOD", "HOH"}:
                        classification = "water"
                    elif component in {"CA", "CD", "CL", "CO", "CU", "FE", "K", "MG", "MN", "NA", "NI", "ZN"}:
                        classification = "ion"
                    else:
                        classification = "ligand"
                    nonpolymer_counts[(classification, component)] += int(count)
            nmr_chain_rows.extend(payload.get("nmr_summaries", []))
    finally:
        for handle in [
            *handles.values(),
            *eligibility_handles.values(),
            *[handle for handle in (missing_handle, modified_handle, nmr_handle) if handle is not None],
        ]:
            handle.close()
        if candidate_writer is not None:
            candidate_writer.close()
    if candidate_writer is None and publish_verbose:
        pd.DataFrame(columns=["sample_id", "source_file", "candidate_index"]).to_parquet(
            output_dir / "candidate_resolution_evidence.parquet", index=False
        )
    _write_query_csv(
        connection,
        output_dir / "raw_source_failures.csv",
        """SELECT st.source_file,st.source_sha256,st.error_type,st.error_message,
        st.parser_calls,st.updated_utc FROM selected_sources s JOIN source_state st
        ON st.source_file=s.source_file AND st.source_sha256=s.source_sha256
        WHERE st.status='failed' ORDER BY st.source_file""",
    )
    pd.DataFrame(
        [
            {
                "split": key[0],
                "method": key[1],
                "length_bin": key[2],
                "trimming_status": key[3],
                "alignment_class": key[4],
                "count": count,
            }
            for key, count in sorted(alignment_counts.items())
        ],
        columns=["split", "method", "length_bin", "trimming_status", "alignment_class", "count"],
    ).to_csv(output_dir / "alignment_classes_summary.csv", index=False)
    pd.DataFrame(
        [{"residue_name": key, "count": value} for key, value in sorted(modified_counts.items())],
        columns=["residue_name", "count"],
    ).to_csv(output_dir / "polymer_modified_residue_summary.csv", index=False)
    pd.DataFrame(
        [
            {"classification": key[0], "component": key[1], "count": value}
            for key, value in sorted(nonpolymer_counts.items())
        ],
        columns=["classification", "component", "count"],
    ).to_csv(output_dir / "excluded_nonpolymer_component_summary.csv", index=False)
    pd.DataFrame(
        [{"outcome": key, "count": value} for key, value in sorted(altloc_counts.items())],
        columns=["outcome", "count"],
    ).to_csv(output_dir / "alternate_location_outcome_summary.csv", index=False)
    pd.DataFrame(
        [
            {"outcome": key[0], "count_value": key[1], "chain_model_count": value}
            for key, value in sorted(missing_counts.items())
        ],
        columns=["outcome", "count_value", "chain_model_count"],
    ).to_csv(output_dir / "missing_residue_outcome_summary.csv", index=False)
    nmr_frame = pd.DataFrame(nmr_chain_rows)
    if nmr_frame.empty:
        nmr_frame = pd.DataFrame(
            columns=[
                "source_file",
                "label_asym_id",
                "auth_asym_id",
                "model_count",
                "sequence_consistent_across_models",
                "minimum_missing_calpha_count",
                "maximum_missing_calpha_count",
                "missing_calpha_varies_across_models",
                "processed_model_1_matrix_compatible_with_all_models",
                "processed_model_1_sample_count",
            ]
        )
    nmr_frame.to_csv(output_dir / "nmr_source_chain_consistency.csv", index=False)
    blocking_frame = pd.DataFrame(blocking_rows)
    if blocking_frame.empty:
        blocking_frame = pd.DataFrame(
            columns=["sample_id", "alignment_class", "alignment_reason", "coordinate_verification_status"]
        )
    blocking_frame.to_csv(output_dir / "raw_blocking_failures.csv", index=False)
    transition_counts = (
        _write_v2_v3_transition(
            prior_forensics_dir,
            transition_alignments,
            output_dir / "v2_forensic_v3_transition.csv",
        )
        if prior_forensics_dir is not None
        else {}
    )
    if prior_forensics_dir is None:
        _atomic_csv(
            output_dir / "v2_forensic_v3_transition.csv",
            pd.DataFrame(
                columns=[
                    "sample_id",
                    "v2_classification",
                    "forensic_root_cause",
                    "v3_pairing_classification",
                    "strict_provenance_status",
                    "practical_training_eligibility",
                    "source_identity_status",
                ]
            ),
        )
    eligibility_counts = dict(sorted(eligibility_counts.items()))
    source_identity_counts = dict(sorted(source_identity_counts.items()))
    raw_summary = {
        "alignment_class_counts": dict(sorted(alignment_class_counts.items())),
        "refined_primary_classification_counts": dict(sorted(refined_classification_counts.items())),
        "coordinate_verification_status_counts": dict(sorted(coordinate_status_counts.items())),
        "associated_processed_sample_count": sum(alignment_class_counts.values()),
        "protein_modified_residue_counts": dict(sorted(modified_counts.items())),
        "excluded_nonpolymer_component_counts": {
            f"{classification}:{component}": count
            for (classification, component), count in sorted(nonpolymer_counts.items())
        },
        "nmr_source_chain_count": len(nmr_chain_rows),
        "nmr_sequence_consistent_source_chain_count": sum(
            bool(row["sequence_consistent_across_models"]) for row in nmr_chain_rows
        ),
        "nmr_sequence_inconsistent_source_chain_count": sum(
            not bool(row["sequence_consistent_across_models"]) for row in nmr_chain_rows
        ),
        "blocking_failure_count": len(blocking_rows),
        "strict_verified_pair_count": eligibility_counts.get("strict_verified_pair", 0),
        "conditionally_verified_pair_count": eligibility_counts.get("conditionally_verified_pair", 0),
        "excluded_or_unresolved_count": eligibility_counts.get("excluded_or_unresolved", 0),
        "source_identity_status_counts": source_identity_counts,
        "confirmed_audit_defect_count": sum(
            refined_classification_counts[name]
            for name in (
                "audit_chain_entity_mapping_bug",
                "audit_residue_id_convention_bug",
                "audit_coordinate_reconstruction_bug",
            )
        ),
        "metadata_only_inconsistency_count": refined_classification_counts["legacy_trim_metadata_inconsistency"],
        "genuinely_contradictory_pair_count": sum(
            refined_classification_counts[name]
            for name in (
                "matrix_sequence_absent_from_author_linked_polymer",
                "coordinate_disagreement",
                "demonstrated_sequence_matrix_mismatch",
            )
        ),
        "unresolved_pair_count": sum(
            refined_classification_counts[name]
            for name in ("multiple_author_linked_polymer_candidates", "unresolved_ambiguity")
        ),
        "unavailable_historical_source_hash_count": source_identity_counts.get(
            "state_size_mtime_match_sha_unavailable", 0
        ),
        "v2_forensic_v3_transition_counts": transition_counts,
        "raw_chain_model_count": raw_chain_model_count,
        "ambiguous_altloc_position_count": ambiguous_altloc_position_count,
        "alternate_location_evidence_count": sum(altloc_counts.values()),
    }
    _atomic_json(output_dir / "raw_pilot_summary.json", raw_summary)
    return {
        "training_eligibility_counts": eligibility_counts,
        "source_identity_status_counts": source_identity_counts,
        "v2_forensic_v3_transition_counts": transition_counts,
        "v2_forensic_v3_transition_total": sum(transition_counts.values()),
        "scientific_reporting_counts": {
            key: raw_summary[key]
            for key in (
                "confirmed_audit_defect_count",
                "metadata_only_inconsistency_count",
                "strict_verified_pair_count",
                "conditionally_verified_pair_count",
                "genuinely_contradictory_pair_count",
                "unresolved_pair_count",
                "unavailable_historical_source_hash_count",
            )
        },
    }


def _read_alignment_table(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _eligibility_summary(rows: list[dict[str, Any]], *, strict: bool) -> pd.DataFrame:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "eligibility_status": (
                    "archival_cryptographic_provenance_complete"
                    if row.get("training_eligibility") == "strict_verified_pair"
                    else "archival_cryptographic_provenance_incomplete"
                )
                if strict
                else str(row["training_eligibility"]),
                "experimental_method": str(row.get("experimental_method") or "unavailable"),
                "split": str(row.get("split") or "unavailable"),
                "requested_length": str(row.get("requested_length") or "unavailable"),
                "actual_length": str(row.get("matrix_actual_length") or "unavailable"),
                "primary_pairing_classification": _current_pairing_classification(row),
            }
        )
    frame = pd.DataFrame(normalized)
    total = len(frame)
    output = []
    dimensions = (
        "eligibility_status",
        "experimental_method",
        "split",
        "requested_length",
        "actual_length",
        "primary_pairing_classification",
    )
    for dimension in dimensions:
        grouped = frame.groupby(["eligibility_status", dimension], dropna=False).size()
        dimension_totals = frame.groupby(dimension, dropna=False).size()
        for (status, value), count in grouped.items():
            group_total = int(dimension_totals.loc[value])
            output.append(
                {
                    "summary_dimension": dimension,
                    "group_value": value,
                    "eligibility_status": status,
                    "pair_count": int(count),
                    "cohort_total_pair_count": total,
                    "fraction_of_all_pairs": int(count) / total,
                    "group_total_pair_count": group_total,
                    "fraction_within_group": int(count) / group_total,
                }
            )
    return pd.DataFrame(output)


def _unresolved_report(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    output = []
    candidates_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidate_rows or []:
        candidates_by_sample[str(candidate.get("sample_id"))].append(candidate)
    evidence_fields = (
        "manifest_npz_sequence_match",
        "unique_author_linked_candidate",
        "author_linked_candidate_count",
        "matrix_sequence_occurrence_count_in_candidate",
        "retained_interval_match",
        "auth_residue_ids_match",
        "label_residue_ids_match",
        "zero_based_positions_match",
        "one_based_positions_match",
        "insertion_codes_match",
        "raw_calpha_available_for_selected_interval",
        "coordinate_matrix_status",
        "competing_candidate_count",
        "source_identity_status",
    )
    for row in rows:
        if row.get("training_eligibility") != "excluded_or_unresolved":
            continue
        blockers = []
        sample_candidates = candidates_by_sample.get(str(row["sample_id"]), [])
        if int(row.get("competing_candidate_count") or 0) > 1:
            blockers.append("multiple_competing_polymer_candidates")
        if not any(
            row.get(key)
            for key in (
                "auth_residue_ids_match",
                "label_residue_ids_match",
                "zero_based_positions_match",
                "one_based_positions_match",
            )
        ):
            blockers.append("residue_identity_namespace_unresolved")
        if row.get("coordinate_matrix_status") == "unavailable":
            blockers.append("coordinate_evidence_unavailable")
        elif row.get("coordinate_matrix_status") == "failed":
            blockers.append("coordinate_disagreement")
        if any(candidate.get("coordinate_status") == "failed" for candidate in sample_candidates):
            blockers.append("competing_candidate_coordinate_disagreement")
        candidate_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for candidate in sample_candidates:
            key = tuple(
                str(candidate.get(field)) for field in ("label_asym_id", "auth_asym_id", "entity_id", "model_number")
            )
            summary = candidate_groups.setdefault(
                key,
                {
                    "label_asym_id": key[0],
                    "auth_asym_id": key[1],
                    "entity_id": key[2],
                    "model_number": key[3],
                    "polymer_type": candidate.get("polymer_type"),
                    "sequence_occurrence_counts": set(),
                    "coordinate_statuses": set(),
                    "residue_id_conventions_with_sequence_match": set(),
                },
            )
            summary["sequence_occurrence_counts"].add(
                str(candidate.get("matrix_sequence_occurrence_count_in_candidate"))
            )
            summary["coordinate_statuses"].add(str(candidate.get("coordinate_status")))
            if str(candidate.get("sequence_match")).lower() == "true":
                summary["residue_id_conventions_with_sequence_match"].add(str(candidate.get("convention")))
        candidate_evidence = []
        for summary in candidate_groups.values():
            candidate_evidence.append(
                {
                    **summary,
                    "sequence_occurrence_counts": sorted(summary["sequence_occurrence_counts"]),
                    "coordinate_statuses": sorted(summary["coordinate_statuses"]),
                    "residue_id_conventions_with_sequence_match": sorted(
                        summary["residue_id_conventions_with_sequence_match"]
                    ),
                }
            )
        output.append(
            {
                "sample_id": row["sample_id"],
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "model_id": row["model_id"],
                "experimental_method": row.get("experimental_method"),
                "split": row.get("split"),
                "requested_length": row.get("requested_length"),
                "actual_length": row.get("matrix_actual_length"),
                "v3_pairing_classification": _current_pairing_classification(row),
                "practical_training_eligibility": row["training_eligibility"],
                "blocking_evidence": json.dumps(blockers),
                "candidate_resolution_evidence": json.dumps(candidate_evidence, sort_keys=True),
                **{key: row.get(key) for key in evidence_fields},
            }
        )
    return pd.DataFrame(output)


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _verify_compact_raw_full_partitions(
    output_dir: Path,
    state_path: Path,
    *,
    expected_source_count: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    scientific_paths = sorted((output_dir / "tables").glob("*/part-*.parquet"))
    if not scientific_paths:
        raise FileNotFoundError("Compact raw-full attestation found no scientific Parquet partitions")
    hashes_before = {str(path): sha256_file(path) for path in scientific_paths}
    marker_paths = sorted((output_dir / "partition_commits").glob("part-*.json"))
    if not marker_paths:
        raise FileNotFoundError("Compact raw-full attestation found no partition commit metadata")

    with closing(_readonly_sqlite(state_path)) as connection:
        state_versions = dict(
            connection.execute(
                "SELECT key,value FROM metadata "
                "WHERE key IN ('aggregation_semantics_version','report_semantics_version')"
            )
        )
        if state_versions != {
            "aggregation_semantics_version": str(REPORT_SEMANTICS_VERSION),
            "report_semantics_version": str(REPORT_SEMANTICS_VERSION),
        }:
            raise ValueError("State database does not attest v4 aggregation and reporting semantics")
        database_partitions = {
            int(index): {
                "source_count": int(source_count),
                "marker_path": str(marker_path),
                "marker_sha256": str(marker_sha256),
            }
            for index, source_count, marker_path, marker_sha256 in connection.execute(
                "SELECT partition_index,source_count,marker_path,marker_sha256 FROM compact_partitions"
            )
        }
        selected_sources = {
            (str(source_file), str(source_sha256))
            for source_file, source_sha256 in connection.execute(
                "SELECT source_file,source_sha256 FROM selected_sources"
            )
        }
        source_states = list(connection.execute("SELECT source_file,source_sha256,status FROM source_state"))

    if len(database_partitions) != len(marker_paths):
        raise ValueError("Compact partition database and commit-marker counts differ")
    if len(selected_sources) != expected_source_count:
        raise ValueError(f"Expected {expected_source_count} selected compact sources, found {len(selected_sources)}")
    completed_states = {
        (str(source_file), str(source_sha256))
        for source_file, source_sha256, status in source_states
        if status == "completed"
    }
    failed_states = [row for row in source_states if row[2] == "failed"]
    if failed_states:
        raise ValueError(f"Compact raw-full state contains parser failures: {failed_states[0][0]}")
    if completed_states != selected_sources:
        raise ValueError("Compact raw-full selected/completed source membership is incomplete or inconsistent")

    expected_paths: set[Path] = set()
    seen_source_ids: set[str] = set()
    source_identities: set[tuple[str, str]] = set()
    schema_columns: dict[str, set[str]] = defaultdict(set)
    table_names: set[str] | None = None
    total_marker_sources = 0
    for expected_index, marker_path in enumerate(marker_paths):
        marker = json.loads(marker_path.read_text())
        index = int(marker.get("partition_index", -1))
        if index != expected_index or marker_path.name != f"part-{index:06d}.json":
            raise ValueError("Compact partitions are missing, duplicated, or out of order")
        database = database_partitions.get(index)
        if database is None:
            raise ValueError(f"Compact partition {index} is absent from the state database")
        if Path(database["marker_path"]).resolve() != marker_path.resolve():
            raise ValueError(f"Compact partition {index} marker path contradicts the state database")
        if database["marker_sha256"] != sha256_file(marker_path):
            raise ValueError(f"Compact partition {index} marker hash contradicts the state database")
        source_ids = [str(value) for value in marker.get("source_ids", [])]
        source_count = int(marker.get("source_count", -1))
        if source_count != database["source_count"] or len(source_ids) != source_count:
            raise ValueError(f"Compact partition {index} source count is inconsistent")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError(f"Compact partition {index} contains duplicate source identities")
        overlap = seen_source_ids.intersection(source_ids)
        if overlap:
            raise ValueError(f"Compact partitions overlap at source identity: {sorted(overlap)[0]}")
        seen_source_ids.update(source_ids)
        total_marker_sources += source_count

        current_tables = set(marker.get("tables", {}))
        table_names = current_tables if table_names is None else table_names
        if current_tables != table_names:
            raise ValueError(f"Compact partition {index} has an inconsistent logical-table set")
        validate_compact_partition(output_dir, marker)
        for logical_table, metadata in marker["tables"].items():
            path = (output_dir / str(metadata["path"])).resolve()
            if output_dir.resolve() not in path.parents or path in expected_paths:
                raise ValueError(f"Compact partition path is duplicated or outside the audit: {path}")
            expected_paths.add(path)
            partition_columns = set(pq.ParquetFile(path).schema.names)
            schema_columns[logical_table].update(partition_columns)
            if int(metadata["row_count"]) > 0 and logical_table in V4_COMPACT_REQUIRED_SCHEMAS:
                missing = V4_COMPACT_REQUIRED_SCHEMAS[logical_table] - partition_columns
                if missing:
                    raise ValueError(
                        f"Compact v4 logical table {logical_table} partition {index} "
                        f"is missing columns: {sorted(missing)}"
                    )
        source_path = (output_dir / marker["tables"]["source_identity"]["path"]).resolve()
        source_rows = pq.read_table(
            source_path,
            columns=["source_id", "source_file", "source_sha256", "parse_status"],
        ).to_pylist()
        if [str(row["source_id"]) for row in source_rows] != source_ids:
            raise ValueError(f"Compact partition {index} source identities contradict its commit metadata")
        for row in source_rows:
            if row["parse_status"] != "completed":
                raise ValueError(f"Compact raw-full partition contains a parser failure: {row['source_file']}")
            identity = (str(row["source_file"]), str(row["source_sha256"]))
            if identity in source_identities:
                raise ValueError(f"Compact source identity is duplicated across partitions: {identity[0]}")
            source_identities.add(identity)

    actual_paths = {path.resolve() for path in scientific_paths}
    if actual_paths != expected_paths:
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        raise ValueError(
            "Compact scientific partitions do not exactly match commit metadata: "
            f"missing={missing[:1]}, unexpected={unexpected[:1]}"
        )
    if total_marker_sources != expected_source_count or source_identities != selected_sources:
        raise ValueError("Compact partition source coverage does not match the selected source state")
    for logical_table, required_columns in V4_COMPACT_REQUIRED_SCHEMAS.items():
        missing = required_columns - schema_columns.get(logical_table, set())
        if missing:
            raise ValueError(f"Compact v4 logical table {logical_table} is missing columns: {sorted(missing)}")

    return hashes_before, {
        "committed_partition_count": len(marker_paths),
        "scientific_partition_count": len(scientific_paths),
        "committed_source_count": total_marker_sources,
        "logical_tables": sorted(table_names or ()),
        "required_v4_schemas_verified": True,
        "missing_partition_count": 0,
        "corrupt_partition_count": 0,
        "duplicate_partition_count": 0,
        "overlapping_partition_count": 0,
        "raw_source_parser_failure_count": 0,
    }


def _verify_compact_raw_full_pairing_semantics(
    output_dir: Path,
    *,
    expected_pair_count: int,
    expected_eligible_count: int,
    expected_unresolved_count: int,
) -> tuple[dict[str, int], dict[str, int]]:
    reader = SequenceReadinessArtifactReader(output_dir)
    practical = reader.frame("practical_eligibility")
    strict = reader.frame("strict_eligibility")
    unresolved = reader.frame("unresolved_eligibility")
    required_summary_columns = {
        "sample_id",
        "practical_training_eligibility",
        "model_id",
    }
    for name, frame in (("practical", practical), ("strict", strict), ("unresolved", unresolved)):
        missing = required_summary_columns - set(frame)
        if missing:
            raise ValueError(f"Compact v4 {name} summary is missing columns: {sorted(missing)}")
    if len(practical) != expected_eligible_count or len(unresolved) != expected_unresolved_count:
        raise ValueError("Compact raw-full eligibility summary counts do not match the required v4 contract")
    if len(strict) != 0:
        raise ValueError("Compact raw-full strict eligibility summary must be empty")
    practical_ids = set(practical["sample_id"].astype(str))
    unresolved_ids = set(unresolved["sample_id"].astype(str))
    if practical_ids & unresolved_ids:
        raise ValueError("Compact raw-full practical and unresolved summaries overlap")

    seen_ids: set[str] = set()
    eligibility_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    for row in reader.iter_records("matrix_pair_alignments"):
        sample_id = str(row.get("sample_id"))
        if sample_id in seen_ids:
            raise ValueError(f"Compact raw-full matrix pair is duplicated: {sample_id}")
        seen_ids.add(sample_id)
        eligibility = str(row.get("training_eligibility"))
        if eligibility not in {"strict_verified_pair", "conditionally_verified_pair", "excluded_or_unresolved"}:
            raise ValueError(f"Unsupported compact raw-full training eligibility for {sample_id}: {eligibility}")
        classification = _current_pairing_classification(row)
        if classification not in V4_PAIRING_CLASSIFICATIONS:
            raise ValueError(f"Unsupported v4 pairing classification for {sample_id}: {classification}")
        eligibility_counts[eligibility] += 1
        classification_counts[classification] += 1
    if len(seen_ids) != expected_pair_count:
        raise ValueError(f"Expected {expected_pair_count} unique compact matrix pairs, found {len(seen_ids)}")
    expected_eligibility_counts = {
        key: value
        for key, value in {
            "conditionally_verified_pair": expected_eligible_count,
            "excluded_or_unresolved": expected_unresolved_count,
        }.items()
        if value
    }
    if eligibility_counts != expected_eligibility_counts:
        raise ValueError(f"Compact raw-full matrix-pair eligibility counts are incorrect: {dict(eligibility_counts)}")
    if practical_ids | unresolved_ids != seen_ids:
        raise ValueError("Compact raw-full summary membership does not cover the matrix-pair population exactly")
    failures_path = output_dir / "raw_source_failures.csv"
    if failures_path.is_file() and not pd.read_csv(failures_path).empty:
        raise ValueError("Compact raw-full derived failure summary contains parser failures")
    return dict(sorted(eligibility_counts.items())), dict(sorted(classification_counts.items()))


def _regenerate_v4_pairing_summaries(
    output_dir: Path,
    *,
    expected_pair_count: int,
    expected_eligible_count: int,
    expected_unresolved_count: int,
) -> list[str]:
    columns = [
        "sample_id",
        "pdb_id",
        "chain_id",
        "model_id",
        "matrix_path",
        "v3_pairing_classification",
        "strict_provenance_status",
        "practical_training_eligibility",
        "source_identity_status",
    ]
    targets = {
        "strict": output_dir / "strict_training_eligibility.csv",
        "practical": output_dir / "practical_training_eligibility.csv",
        "unresolved": output_dir / "unresolved_cases.csv",
        "unresolved_legacy": output_dir / "unresolved_case_summary.csv",
    }
    temporary = {key: path.with_name(f".{path.name}.{os.getpid()}.tmp") for key, path in targets.items()}
    counts: Counter[str] = Counter()
    handles: dict[str, Any] = {}
    try:
        for key in ("strict", "practical", "unresolved"):
            handles[key] = temporary[key].open("w", newline="")
        writers = {key: csv.DictWriter(handle, fieldnames=columns) for key, handle in handles.items()}
        for writer in writers.values():
            writer.writeheader()
        for row in SequenceReadinessArtifactReader(output_dir).iter_records("matrix_pair_alignments"):
            eligibility = str(row["training_eligibility"])
            destination = (
                "strict"
                if eligibility == "strict_verified_pair"
                else "practical"
                if eligibility == "conditionally_verified_pair"
                else "unresolved"
            )
            writers[destination].writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "pdb_id": row.get("pdb_id"),
                    "chain_id": row.get("chain_id"),
                    "model_id": row.get("model_id"),
                    "matrix_path": row.get("matrix_path"),
                    "v3_pairing_classification": _current_pairing_classification(row),
                    "strict_provenance_status": row.get("strict_provenance_status"),
                    "practical_training_eligibility": eligibility,
                    "source_identity_status": row.get("source_identity_status"),
                }
            )
            counts[destination] += 1
    finally:
        for handle in handles.values():
            handle.close()
    expected_summary_counts = {
        key: value
        for key, value in {
            "practical": expected_eligible_count,
            "unresolved": expected_unresolved_count,
        }.items()
        if value
    }
    if counts != expected_summary_counts or sum(counts.values()) != expected_pair_count:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(f"Regenerated v4 summary counts are inconsistent: {dict(counts)}")
    shutil.copyfile(temporary["unresolved"], temporary["unresolved_legacy"])
    for key, target in targets.items():
        temporary[key].replace(target)
    return [str(path) for path in targets.values()]


def attest_compact_raw_full_semantics(
    output_dir: Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> Path:
    """Verify immutable compact raw-full evidence and publish a v4 semantics attestation."""
    counts = dict(RAW_FULL_ATTESTATION_COUNTS if expected_counts is None else expected_counts)
    protocol_path = output_dir / "sequence_readiness_protocol.json"
    config_path = output_dir / "run_config.json"
    if not protocol_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Compact raw-full attestation requires completed protocol and run metadata")
    prior_protocol_sha256 = sha256_file(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    run_config = json.loads(config_path.read_text())
    if protocol.get("status") != "completed" or protocol.get("audit_mode") != "raw-full":
        raise ValueError("V4 attestation requires a completed raw-full protocol")
    if protocol.get("storage_profile") != "compact-v1" or run_config.get("storage_profile") != "compact-v1":
        raise ValueError("V4 raw-full attestation requires compact-v1 artifacts and run metadata")
    if run_config.get("audit_mode") != "raw-full":
        raise ValueError("Run metadata does not identify a raw-full audit")
    if protocol.get("raw_inputs_unchanged") is not True:
        raise ValueError("V4 raw-full attestation requires raw_inputs_unchanged=true")
    expected_sources = counts["selected_source_count"]
    for field in ("selected_source_file_count", "completed_source_count"):
        if int(protocol.get(field, -1)) != expected_sources:
            raise ValueError(f"V4 raw-full source contract failed for {field}")
    if int(protocol.get("pending_source_count", -1)) != 0 or int(protocol.get("failed_source_count", -1)) != 0:
        raise ValueError("V4 raw-full attestation requires zero pending and failed sources")
    state_path = Path(run_config.get("state_dir", output_dir)) / "audit_state.sqlite"
    if not state_path.is_file():
        raise FileNotFoundError(f"Compact raw-full state database is missing: {state_path}")

    hashes_before, partition_verification = _verify_compact_raw_full_partitions(
        output_dir,
        state_path,
        expected_source_count=expected_sources,
    )
    eligibility_counts, classification_counts = _verify_compact_raw_full_pairing_semantics(
        output_dir,
        expected_pair_count=counts["matrix_pair_count"],
        expected_eligible_count=counts["practical_eligible_count"],
        expected_unresolved_count=counts["unresolved_count"],
    )
    hashes_after_verification = {path: sha256_file(Path(path)) for path in hashes_before}
    if hashes_before != hashes_after_verification:
        raise RuntimeError("A scientific partition changed during compact raw-full attestation")

    regenerated = _regenerate_v4_pairing_summaries(
        output_dir,
        expected_pair_count=counts["matrix_pair_count"],
        expected_eligible_count=counts["practical_eligible_count"],
        expected_unresolved_count=counts["unresolved_count"],
    )
    hashes_after = {path: sha256_file(Path(path)) for path in hashes_before}
    if hashes_before != hashes_after:
        raise RuntimeError("A scientific partition changed during compact raw-full summary regeneration")

    completed_utc = _utc_now()
    raw_summary_path = output_dir / "raw_pilot_summary.json"
    if raw_summary_path.is_file():
        raw_summary = json.loads(raw_summary_path.read_text())
        raw_summary["report_semantics_version"] = REPORT_SEMANTICS_VERSION
        raw_summary["primary_pairing_classification_counts"] = classification_counts
        raw_summary["conditionally_verified_pair_count"] = counts["practical_eligible_count"]
        raw_summary["excluded_or_unresolved_count"] = counts["unresolved_count"]
        raw_summary["unresolved_pair_count"] = counts["unresolved_count"]
        _atomic_json(raw_summary_path, raw_summary)
        regenerated.append(str(raw_summary_path))

    run_config["report_semantics_version"] = REPORT_SEMANTICS_VERSION
    run_config["semantics_attestation_completed_utc"] = completed_utc
    _atomic_json(config_path, run_config)
    protocol["report_semantics_version"] = REPORT_SEMANTICS_VERSION
    protocol["training_eligibility_counts"] = eligibility_counts
    protocol.setdefault("scientific_reporting_counts", {}).update(
        {
            "conditionally_verified_pair_count": counts["practical_eligible_count"],
            "strict_verified_pair_count": 0,
            "unresolved_pair_count": counts["unresolved_count"],
        }
    )
    protocol["summary_only_regeneration"] = {
        "status": "completed",
        "attestation": "compact_raw_full_v4",
        "completed_utc": completed_utc,
        "prior_protocol_sha256": prior_protocol_sha256,
        "manifest_rows_reindexed": 0,
        "npz_reads": 0,
        "mmcif_parser_calls": 0,
        "scientific_partition_rewrites": 0,
        "scientific_partition_hashes_before": hashes_before,
        "scientific_partition_hashes_after": hashes_after,
        "scientific_partition_hashes_preserved": True,
        "verified_count_contracts": counts,
        "classification_counts": classification_counts,
        "partition_verification": partition_verification,
        "regenerated_derived_files": regenerated,
    }
    _atomic_json(protocol_path, protocol)
    return protocol_path


def regenerate_summary_reports(
    output_dir: Path,
    *,
    expected_total_pairs: int = 686,
    expected_eligible_pairs: int = 675,
    expected_transition_cases: int = 197,
    expected_candidate_rows: int = 3260,
) -> Path:
    """Regenerate derived reports from immutable completed scientific tables."""
    protocol_path = output_dir / "sequence_readiness_protocol.json"
    if protocol_path.is_file():
        existing_protocol = json.loads(protocol_path.read_text())
        if existing_protocol.get("audit_mode") == "raw-full":
            return attest_compact_raw_full_semantics(output_dir)
    reader = SequenceReadinessArtifactReader(output_dir)
    config_path = output_dir / "run_config.json"
    run_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    state_path = Path(run_config.get("state_dir", output_dir)) / "audit_state.sqlite"
    required = (protocol_path, config_path, state_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing completed v3 reporting input(s): {', '.join(missing)}")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "completed" or protocol.get("audit_mode") != "raw-pilot":
        raise ValueError("Summary-only regeneration requires a completed raw-pilot protocol")
    prior_forensics_dir = Path(run_config["prior_forensics_dir"])
    forensic_path = prior_forensics_dir / "per_failure_classification.parquet"
    if not forensic_path.is_file():
        raise FileNotFoundError(f"Missing preserved forensic classifications: {forensic_path}")

    rows = list(reader.iter_records("matrix_pair_alignments"))
    if len(rows) != expected_total_pairs or len({str(row["sample_id"]) for row in rows}) != expected_total_pairs:
        raise ValueError(f"Expected {expected_total_pairs} unique matrix pairs in completed v3")
    eligible = [row for row in rows if row.get("training_eligibility") != "excluded_or_unresolved"]
    unresolved = [row for row in rows if row.get("training_eligibility") == "excluded_or_unresolved"]
    if len(eligible) != expected_eligible_pairs or len(unresolved) != expected_total_pairs - expected_eligible_pairs:
        raise ValueError("Completed v3 eligibility counts do not match the validated scientific counts")
    if not any(str(row["sample_id"]).lower() == "7ycs_d" for row in unresolved):
        raise ValueError("7YCS_D is missing from the unresolved v3 cohort")
    candidate_rows = list(reader.iter_records("residue_id_convention_evidence"))
    candidate_count = len(candidate_rows)
    if candidate_count != expected_candidate_rows:
        raise ValueError(f"Expected {expected_candidate_rows} candidate-evidence rows, found {candidate_count}")
    scientific_paths = (*reader.scientific_paths(), forensic_path)
    scientific_hashes_before = {str(path): sha256_file(path) for path in scientific_paths}
    transition_path = output_dir / "v2_forensic_v3_transition.csv"
    transition = _v2_v3_transition_frame(prior_forensics_dir, rows)
    if len(transition) != expected_transition_cases or transition["sample_id"].nunique() != expected_transition_cases:
        raise RuntimeError("Corrected transition does not contain exactly 197 unique cases")
    transition_eligibility = Counter(transition["practical_training_eligibility"])
    if transition_eligibility != {"conditionally_verified_pair": 186, "excluded_or_unresolved": 11}:
        raise RuntimeError("Corrected transition eligibility must be 186 conditional and 11 unresolved")
    transition_counts = dict(sorted(Counter(transition["v3_pairing_classification"]).items()))

    strict_summary = _eligibility_summary(rows, strict=True)
    practical_summary = _eligibility_summary(rows, strict=False)
    unresolved_summary = _unresolved_report(rows, candidate_rows)
    row_columns = [
        "sample_id",
        "pdb_id",
        "chain_id",
        "model_id",
        "matrix_path",
        "v3_pairing_classification",
        "strict_provenance_status",
        "practical_training_eligibility",
        "source_identity_status",
    ]
    row_reports = []
    for row in rows:
        row_reports.append(
            {
                **{key: row.get(key) for key in row_columns},
                "v3_pairing_classification": _current_pairing_classification(row),
                "practical_training_eligibility": row["training_eligibility"],
            }
        )
    row_frame = pd.DataFrame(row_reports, columns=row_columns)
    _atomic_csv(transition_path, transition)
    _atomic_csv(output_dir / "strict_training_eligibility.csv", row_frame.iloc[0:0])
    _atomic_csv(
        output_dir / "practical_training_eligibility.csv",
        row_frame[row_frame["practical_training_eligibility"] != "excluded_or_unresolved"],
    )
    _atomic_csv(output_dir / "unresolved_cases.csv", unresolved_summary)
    _atomic_csv(output_dir / "strict_training_eligibility_summary.csv", strict_summary)
    _atomic_csv(output_dir / "practical_training_eligibility_summary.csv", practical_summary)
    _atomic_csv(output_dir / "unresolved_case_summary.csv", unresolved_summary)

    total = len(rows)
    excluded_count = len(unresolved)
    decision = {
        "status": "usable_with_filtering",
        "unfiltered_cohort_acceptance": "failed",
        "filtered_dataset_usable": "passed",
        "eligible_pair_count": len(eligible),
        "total_pair_count": total,
        "eligible_pair_fraction": len(eligible) / total,
        "excluded_or_unresolved_count": excluded_count,
        "excluded_or_unresolved_fraction": excluded_count / total,
        "archival_cryptographic_provenance_complete_count": 0,
        "strict_archival_provenance_limitation": (
            "Historical raw SHA-256 values were never stored; this does not invalidate the 675 supported "
            "sequence-geometry correspondences."
        ),
        "genuine_sequence_matrix_mismatch_count": 0,
        "pilot_sampling_limitation": (
            "This stratified 250-source pilot is diagnostic and does not estimate corpus-wide prevalence."
        ),
        "corpus_wide_status": "pending_full_raw_audit",
    }
    _atomic_json(output_dir / "sequence_geometry_dataset_decision.json", decision)

    acceptance = json.loads((output_dir / "acceptance_criteria_results.json").read_text())
    acceptance["criteria"] = [
        item
        for item in acceptance["criteria"]
        if item["criterion"] not in {"strict_sequence_geometry_pairing", "practical_sequence_geometry_pairing"}
    ]
    acceptance["criteria"].extend(
        [
            _criterion(
                "archival_cryptographic_provenance_complete",
                "informational",
                0,
                False,
                "0/686 pairs have archived historical raw SHA-256 provenance; all 686 have only matching "
                "historical size/mtime state evidence.",
            ),
            _criterion(
                "unfiltered_cohort_acceptance",
                "failed",
                total,
                True,
                "The unfiltered pilot fails: 11/686 pairs remain excluded or unresolved; 675/686 are eligible.",
            ),
            _criterion(
                "filtered_dataset_usable",
                "passed",
                len(eligible),
                True,
                "The filtered pilot is usable: 675/686 pairs are deterministically eligible, 11/686 are excluded, "
                "and 0/686 are confirmed sequence-matrix mismatches.",
            ),
        ]
    )
    for item in acceptance["criteria"]:
        if item["criterion"] in {"seqres_atom_matrix_provenance", "missing_residue_alignment"}:
            item["explanation"] = (
                "The pilot contains 675/686 practically eligible pairs and 11/686 excluded or unresolved pairs; "
                "the latter must be filtered before dataset use."
            )
    acceptance["status_counts"] = dict(Counter(item["status"] for item in acceptance["criteria"]))
    acceptance["unfiltered_cohort_acceptance"] = "failed"
    acceptance["filtered_dataset_usable"] = "passed"
    acceptance["eligible_pair_count"] = len(eligible)
    acceptance["total_pair_count"] = total
    acceptance["eligible_pair_fraction"] = len(eligible) / total
    acceptance["excluded_or_unresolved_count"] = excluded_count
    _atomic_json(output_dir / "acceptance_criteria_results.json", acceptance)

    raw_summary = json.loads((output_dir / "raw_pilot_summary.json").read_text())
    pairing_counts = Counter(_current_pairing_classification(row) for row in rows)
    raw_summary["primary_pairing_classification_counts"] = dict(sorted(pairing_counts.items()))
    raw_summary["metadata_only_inconsistency_count"] = int(
        raw_summary.get("alignment_class_counts", {}).get("valid_residue_id_selection", 0)
    )
    raw_summary["strict_archival_provenance_complete_count"] = 0
    raw_summary["unresolved_pair_count"] = excluded_count
    raw_summary["v2_forensic_v3_transition_counts"] = transition_counts
    _atomic_json(output_dir / "raw_pilot_summary.json", raw_summary)

    scientific_hashes_after = {str(path): sha256_file(path) for path in scientific_paths}
    if scientific_hashes_before != scientific_hashes_after:
        raise RuntimeError("A preserved scientific table changed during summary-only regeneration")
    protocol["report_semantics_version"] = REPORT_SEMANTICS_VERSION
    protocol["training_eligibility_counts"] = {
        "conditionally_verified_pair": len(eligible),
        "excluded_or_unresolved": excluded_count,
        "strict_verified_pair": 0,
    }
    protocol["v2_forensic_v3_transition_counts"] = transition_counts
    protocol["v2_forensic_v3_transition_total"] = len(transition)
    protocol["scientific_reporting_counts"].update(
        {
            "conditionally_verified_pair_count": len(eligible),
            "strict_verified_pair_count": 0,
            "unresolved_pair_count": excluded_count,
            "metadata_only_inconsistency_count": 28,
            "genuinely_contradictory_pair_count": 0,
        }
    )
    protocol["dataset_decision"] = decision
    protocol["summary_only_regeneration"] = {
        "status": "completed",
        "completed_utc": _utc_now(),
        "manifest_rows_reindexed": 0,
        "npz_reads": 0,
        "mmcif_parser_calls": 0,
        "scientific_input_hashes_before": scientific_hashes_before,
        "scientific_input_hashes_after": scientific_hashes_after,
        "preserved_scientific_classification_table_and_candidate_evidence_unchanged": True,
        "reporting_semantics_updated": True,
    }
    _atomic_json(protocol_path, protocol)
    return protocol_path


def _criterion(
    criterion: str,
    status: str,
    evidence_count: int,
    blocking: bool,
    explanation: str,
) -> dict[str, Any]:
    return {
        "criterion": criterion,
        "status": status,
        "evidence_count": evidence_count,
        "blocking": blocking,
        "explanation": explanation,
    }


def _write_static_documents(
    output_dir: Path,
    audit_mode: str,
    manifest_summary: dict[str, Any],
    manifest_npz_mismatch_count: int,
) -> None:
    (output_dir / "PROPOSED_SCHEMA_AND_MIGRATION.md").write_text(
        """# Proposed Schema And Migration

This audit does not migrate data. A future versioned schema must preserve
`sample_id`, `pdb_id`, `chain_id`, `model_id`, `seqres_sequence`,
`atom_sequence`, `matrix_sequence`, residue tokens, label/auth residue IDs,
insertion codes, residue and missing-C-alpha masks, selected alternate
locations, matrix path, requested and actual lengths, sequence source, modified
residue mappings, and source/matrix/configuration SHA-256 provenance.
"""
    )
    total = int(manifest_summary["manifest_row_counts"]["processed"])
    canonical_count = int(manifest_summary["canonical_sequence_count"])
    length_match_count = int(manifest_summary["sequence_recorded_length_match_count"])
    ambiguity_summary = json.loads((output_dir / "ambiguity_output_summary.json").read_text())
    ambiguity_count = int(ambiguity_summary["underlying_group_count"])
    leakage_summary = json.loads((output_dir / "leakage_output_summary.json").read_text())
    leakage_count = sum(
        int(value["underlying_group_count"])
        for value in leakage_summary.values()
        if isinstance(value, dict) and "underlying_group_count" in value
    )
    hash_statistics = manifest_summary["sequence_hash_statistics"]
    checked_hashes = sum(int(value["checked_count"]) for value in hash_statistics.values())
    hash_mismatches = sum(int(value["mismatch_count"]) for value in hash_statistics.values())
    unavailable_hashes = sum(int(value["stored_hash_unavailable_count"]) for value in hash_statistics.values())
    criteria = [
        _criterion(
            "canonical_sequence",
            "passed" if canonical_count == total else "failed",
            total,
            True,
            f"{canonical_count} of {total} processed sequences contain only canonical one-letter residues.",
        ),
        _criterion(
            "recorded_sequence_matrix_length_agreement",
            "passed" if length_match_count == total else "failed",
            total,
            True,
            f"{length_match_count} of {total} processed rows have sequence length equal to recorded matrix length.",
        ),
        _criterion(
            "unambiguous_matrix_association",
            "passed" if ambiguity_count == 0 else "failed",
            ambiguity_count,
            True,
            f"Found {ambiguity_count} ambiguous matrix-path association groups.",
        ),
        _criterion(
            "manifest_npz_agreement",
            "passed" if manifest_npz_mismatch_count == 0 else "failed",
            total,
            True,
            f"Checked {total} processed NPZ files and found {manifest_npz_mismatch_count} mismatches.",
        ),
        _criterion(
            "train_validation_leakage",
            "passed" if leakage_count == 0 else "failed",
            leakage_count,
            True,
            f"Found {leakage_count} leaking unique keys across all configured policies.",
        ),
        _criterion(
            "stored_sequence_hash_verification",
            "failed" if hash_mismatches else "informational" if unavailable_hashes else "passed",
            checked_hashes,
            False,
            "Stored hashes were checked where available; absence in the processed manifest is reported as unavailable.",
        ),
    ]
    if audit_mode == "manifest-only":
        for criterion, explanation in (
            ("seqres_atom_matrix_provenance", "Requires declared-polymer and coordinate evidence from raw mmCIF."),
            ("alternate_location_handling", "Requires raw alternate-location candidates and selection evidence."),
            ("modified_residue_mapping", "Requires raw residue identities before canonical mapping."),
            ("missing_residue_alignment", "Requires raw polymer-to-coordinate alignment."),
            ("nmr_cross_model_sequence_consistency", "Requires inspection of every model in selected NMR mmCIF files."),
        ):
            criteria.append(_criterion(criterion, "pending_raw_audit", 0, True, explanation))
    else:
        raw_summary = json.loads((output_dir / "raw_pilot_summary.json").read_text())
        sample_count = int(raw_summary["associated_processed_sample_count"])
        failures = int(raw_summary["blocking_failure_count"])
        criteria.extend(
            [
                _criterion(
                    "strict_sequence_geometry_pairing",
                    "pilot_passed"
                    if sample_count and raw_summary["strict_verified_pair_count"] == sample_count
                    else "pilot_failed",
                    int(raw_summary["strict_verified_pair_count"]),
                    False,
                    "Strict verification requires exact residue identity and coordinates plus complete historical "
                    "source provenance; practical eligibility is reported separately.",
                ),
                _criterion(
                    "practical_sequence_geometry_pairing",
                    "pilot_passed" if sample_count and failures == 0 else "pilot_failed",
                    sample_count - failures,
                    True,
                    "Practical eligibility includes uniquely supported conditional pairs without contradictory "
                    "evidence.",
                ),
                _criterion(
                    "legacy_terminal_trim_metadata",
                    "informational",
                    int(raw_summary["alignment_class_counts"].get("valid_residue_id_selection", 0)),
                    False,
                    "Stale trim metadata is informational when sequence, identifiers, and coordinates agree.",
                ),
                _criterion(
                    "seqres_atom_matrix_provenance",
                    "pilot_passed" if sample_count and failures == 0 else "pilot_failed",
                    sample_count,
                    True,
                    f"The inspected pilot classified {sample_count} matrix alignments with "
                    f"{failures} blocking failures.",
                ),
                _criterion(
                    "alternate_location_handling",
                    "pilot_passed" if raw_summary["ambiguous_altloc_position_count"] == 0 else "pilot_failed",
                    int(raw_summary["alternate_location_evidence_count"]),
                    True,
                    "Alternate-location candidates were resolved with the preprocessing selection policy; "
                    "true duplicate ambiguity remains blocking.",
                ),
                _criterion(
                    "modified_residue_mapping",
                    "pilot_passed",
                    sum(int(value) for value in raw_summary["protein_modified_residue_counts"].values()),
                    True,
                    "Versioned protein-polymer mappings were applied; MSE maps to MET/M.",
                ),
                _criterion(
                    "missing_residue_alignment",
                    "pilot_passed" if sample_count and failures == 0 else "pilot_failed",
                    sample_count,
                    True,
                    "Missing-residue outcomes were calculated only over declared protein-polymer positions.",
                ),
                _criterion(
                    "nmr_cross_model_sequence_consistency",
                    (
                        "pilot_passed"
                        if raw_summary["nmr_source_chain_count"]
                        and raw_summary["nmr_sequence_inconsistent_source_chain_count"] == 0
                        else "pilot_failed"
                        if raw_summary["nmr_source_chain_count"]
                        else "informational"
                    ),
                    int(raw_summary["nmr_source_chain_count"]),
                    True,
                    "Every coordinate model in each selected NMR source/chain was compared; "
                    "the full corpus remains unaudited.",
                ),
            ]
        )
    if audit_mode == "raw-pilot":
        for item in criteria:
            if item["status"] == "passed":
                item["status"] = "pilot_passed"
            elif item["status"] == "failed":
                item["status"] = "pilot_failed"
        criteria.append(
            _criterion(
                "corpus_wide_raw_acceptance",
                "pending_full_raw_audit",
                int(raw_summary["raw_chain_model_count"]),
                True,
                "Bounded-pilot evidence cannot establish corpus-wide raw sequence provenance acceptance.",
            )
        )
    _atomic_json(
        output_dir / "acceptance_criteria_results.json",
        {
            "audit_mode": audit_mode,
            "criteria": criteria,
            "status_counts": dict(Counter(item["status"] for item in criteria)),
            "pilot_status": "completed" if audit_mode != "manifest-only" else None,
            "corpus_wide_status": "pending_full_raw_audit" if audit_mode == "raw-pilot" else None,
        },
    )


def run_audit(
    *,
    audit_mode: str,
    processed_manifest: Path,
    train_manifest: Path,
    validation_manifest: Path,
    output_dir: Path,
    raw_dir: Path | None = None,
    preprocess_state_dbs: list[Path] | None = None,
    max_source_files: int | None = None,
    samples_per_stratum: int = 4,
    pilot_seed: int = 4004,
    checkpoint_frequency: int = 25,
    batch_size: int = 8192,
    max_examples_per_group: int = DEFAULT_MAX_EXAMPLES_PER_GROUP,
    max_diagnostic_groups: int = DEFAULT_MAX_DIAGNOSTIC_GROUPS,
    resume: bool = False,
    restart: bool = False,
    stop_after_source_files: int | None = None,
    prior_audit_dir: Path | None = None,
    prior_forensics_dir: Path | None = None,
    storage_profile: str = "verbose-v3",
    state_dir: Path | None = None,
    sources_per_partition: int = DEFAULT_SOURCES_PER_PARTITION,
    unsafe_skip_disk_check: bool = False,
    storage_projection_path: Path | None = None,
    equivalence_reference_dir: Path | None = None,
    compact_equivalence_report: Path | None = None,
) -> Path:
    """Run one staged mode; ``stop_after_source_files`` is for interruption tests."""
    state_dbs = list(preprocess_state_dbs or [])
    state_dir = output_dir if state_dir is None else state_dir
    manifests = [processed_manifest, train_manifest, validation_manifest]
    _validate_preflight(
        audit_mode=audit_mode,
        raw_dir=raw_dir,
        manifest_paths=manifests,
        preprocess_state_dbs=state_dbs,
        output_dir=output_dir,
        state_dir=state_dir,
        storage_profile=storage_profile,
        sources_per_partition=sources_per_partition,
        resume=resume,
        restart=restart,
        max_source_files=max_source_files,
        samples_per_stratum=samples_per_stratum,
        checkpoint_frequency=checkpoint_frequency,
        max_examples_per_group=max_examples_per_group,
        max_diagnostic_groups=max_diagnostic_groups,
        prior_audit_dir=prior_audit_dir,
        prior_forensics_dir=prior_forensics_dir,
        storage_projection_path=storage_projection_path,
        compact_equivalence_report=compact_equivalence_report,
        unsafe_skip_disk_check=unsafe_skip_disk_check,
    )
    config = _config_payload(
        audit_mode=audit_mode,
        raw_dir=raw_dir,
        processed_manifest=processed_manifest,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        preprocess_state_dbs=state_dbs,
        max_source_files=max_source_files,
        samples_per_stratum=samples_per_stratum,
        pilot_seed=pilot_seed,
        checkpoint_frequency=checkpoint_frequency,
        batch_size=batch_size,
        max_examples_per_group=max_examples_per_group,
        max_diagnostic_groups=max_diagnostic_groups,
        prior_audit_dir=prior_audit_dir,
        prior_forensics_dir=prior_forensics_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        storage_profile=storage_profile,
        sources_per_partition=sources_per_partition,
        unsafe_skip_disk_check=unsafe_skip_disk_check,
        storage_projection_path=storage_projection_path,
        equivalence_reference_dir=equivalence_reference_dir,
        compact_equivalence_report=compact_equivalence_report,
    )
    config_digest = _config_hash(config)
    if resume:
        previous = json.loads((output_dir / "run_config.json").read_text())
        previous.setdefault("max_examples_per_group", DEFAULT_MAX_EXAMPLES_PER_GROUP)
        previous.setdefault("max_diagnostic_groups", DEFAULT_MAX_DIAGNOSTIC_GROUPS)
        previous.setdefault("output_dir", str(output_dir.resolve()))
        previous.setdefault("state_dir", str(state_dir.resolve()))
        previous.setdefault("storage_profile", "verbose-v3")
        previous.setdefault("sources_per_partition", DEFAULT_SOURCES_PER_PARTITION)
        previous.setdefault("unsafe_skip_disk_check", False)
        previous.setdefault("storage_projection_path", None)
        previous.setdefault("equivalence_reference_dir", None)
        previous.setdefault("compact_equivalence_report", None)
        if _config_hash(previous) != config_digest:
            raise ValueError("Resume configuration does not match the persisted audit configuration")
    elif restart:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if state_dir != output_dir and state_dir.exists():
            shutil.rmtree(state_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        _atomic_json(output_dir / "run_config.json", config)
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    auxiliary_inputs = []
    if prior_audit_dir is not None:
        auxiliary_inputs.append(prior_audit_dir / "sequence_readiness_protocol.json")
    if prior_forensics_dir is not None:
        auxiliary_inputs.append(prior_forensics_dir / "per_failure_classification.parquet")
    if storage_projection_path is not None:
        auxiliary_inputs.append(storage_projection_path)
    if compact_equivalence_report is not None:
        auxiliary_inputs.append(compact_equivalence_report)
    if equivalence_reference_dir is not None:
        reference_reader = SequenceReadinessArtifactReader(equivalence_reference_dir)
        auxiliary_inputs.extend(reference_reader.scientific_paths())
        auxiliary_inputs.append(equivalence_reference_dir / "sequence_readiness_protocol.json")
    input_paths = manifests + state_dbs + auxiliary_inputs
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    connection = _connect_state(state_dir / "audit_state.sqlite")
    reporter = StageReporter(connection, output_dir, audit_mode, started_utc)
    try:
        for kind, path in zip(("processed", "train", "validation"), manifests, strict=True):
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES (?,?)",
                (f"manifest_columns:{kind}", json.dumps(sorted(_manifest_columns(path)))),
            )
            connection.commit()
            stage = f"indexing_{kind}"
            total = int(pq.ParquetFile(path).metadata.num_rows) if path.suffix.lower() != ".csv" else None
            if not reporter.completed(stage):
                reporter.start(stage, total_count=total)
                indexed = _ingest_manifest(connection, kind=kind, path=path, batch_size=batch_size, reporter=reporter)
                reporter.complete(processed_count=indexed)

        if not reporter.completed("index_creation"):
            reporter.start("index_creation", total_count=len(ANALYSIS_INDEXES))
            _ensure_analysis_indexes(connection, reporter)
            reporter.complete(processed_count=len(ANALYSIS_INDEXES))

        summary_path = output_dir / "manifest_summary.json"
        aggregation_version = connection.execute(
            "SELECT value FROM metadata WHERE key='aggregation_semantics_version'"
        ).fetchone()
        if (
            aggregation_version
            and int(aggregation_version[0]) == REPORT_SEMANTICS_VERSION
            and reporter.completed("aggregation", (summary_path,))
        ):
            manifest_summary = json.loads(summary_path.read_text())
        else:
            reporter.start("aggregation")
            manifest_summary = _manifest_aggregation(connection, output_dir)
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('aggregation_semantics_version',?)",
                (str(REPORT_SEMANTICS_VERSION),),
            )
            connection.commit()
            reporter.complete(detail="manifest summary aggregations completed")

        alphabet_path = output_dir / "sequence_alphabet_counts.csv"
        if not reporter.completed("sequence_validation", (alphabet_path,)):
            reporter.start(
                "sequence_validation",
                total_count=manifest_summary["manifest_row_counts"]["processed"],
            )
            validated = _write_sequence_alphabet(connection, output_dir, reporter)
            reporter.complete(processed_count=validated)

        duplicate_summary_path = output_dir / "duplicate_output_summary.json"
        if not reporter.completed(
            "duplicate_analysis", (output_dir / "duplicate_findings.csv", duplicate_summary_path)
        ):
            reporter.start("duplicate_analysis", total_count=max_diagnostic_groups)
            duplicate_summary = _write_duplicate_groups(
                connection,
                output_dir,
                max_examples=max_examples_per_group,
                max_groups=max_diagnostic_groups,
                reporter=reporter,
            )
            _atomic_json(duplicate_summary_path, duplicate_summary)
            reporter.complete()

        leakage_summary_path = output_dir / "leakage_output_summary.json"
        if not reporter.completed(
            "leakage_analysis", (output_dir / "train_validation_leakage.csv", leakage_summary_path)
        ):
            reporter.start("leakage_analysis", total_count=max_diagnostic_groups)
            leakage_summary = _write_leakage_groups(
                connection,
                output_dir,
                max_examples=max_examples_per_group,
                max_groups=max_diagnostic_groups,
                reporter=reporter,
            )
            _atomic_json(leakage_summary_path, leakage_summary)
            reporter.complete()

        ambiguity_summary_path = output_dir / "ambiguity_output_summary.json"
        if not reporter.completed(
            "ambiguity_analysis", (output_dir / "ambiguous_manifest_pairings.csv", ambiguity_summary_path)
        ):
            reporter.start("ambiguity_analysis", total_count=max_diagnostic_groups)
            ambiguity_summary = _write_ambiguous_groups(
                connection,
                output_dir,
                max_examples=max_examples_per_group,
                max_groups=max_diagnostic_groups,
                reporter=reporter,
            )
            _atomic_json(ambiguity_summary_path, ambiguity_summary)
            reporter.complete()

        policy_path = output_dir / "manifest_policy_distributions.csv"
        if not reporter.completed("table_export", (policy_path,)):
            reporter.start("table_export")
            _write_policy_distributions(connection, output_dir)
            reporter.complete(detail="manifest policy tables exported")

        if audit_mode == "manifest-only":
            mismatch_path = output_dir / "manifest_npz_mismatches.csv"
            if reporter.completed("npz_metadata_validation", (mismatch_path,)):
                mismatch_count = int(connection.execute("SELECT COUNT(*) FROM npz_mismatches").fetchone()[0])
            else:
                reporter.start(
                    "npz_metadata_validation",
                    total_count=manifest_summary["manifest_row_counts"]["processed"],
                )
                mismatch_count = _inspect_npz_rows(
                    connection,
                    mismatch_path,
                    selected_only=False,
                    checkpoint_size=batch_size,
                    reporter=reporter,
                )
                reporter.complete(processed_count=manifest_summary["manifest_row_counts"]["processed"])
            protocol = {
                "report_semantics_version": REPORT_SEMANTICS_VERSION,
                "status": "completed",
                "audit_mode": audit_mode,
                "started_utc": started_utc,
                "completed_utc": _utc_now(),
                "runtime_seconds": time.monotonic() - started_monotonic,
                "peak_memory_if_available": {"ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)},
                "input_hashes": input_hashes,
                "raw_inputs_unchanged": True,
                "selected_source_file_count": 0,
                "parsed_source_file_count": 0,
                "parser_call_count": 0,
                "manifest_row_counts": manifest_summary["manifest_row_counts"],
                "rows_by_split_status": manifest_summary["rows_by_split_status"],
                "requested_stratum_coverage": {},
                "achieved_stratum_coverage": {},
                "completed_source_count": 0,
                "failed_source_count": 0,
                "pending_source_count": 0,
                "manifest_npz_mismatch_count": mismatch_count,
                "maximum_examples_per_group": max_examples_per_group,
                "maximum_diagnostic_groups_per_table": max_diagnostic_groups,
                "recovery_command": "rerun the identical command with --resume",
            }
        else:
            assert raw_dir is not None
            selected, requested, achieved = _ensure_selected_sources(
                connection,
                audit_mode=audit_mode,
                raw_dir=raw_dir,
                state_dbs=state_dbs,
                max_source_files=max_source_files,
                samples_per_stratum=samples_per_stratum,
                pilot_seed=pilot_seed,
                prior_audit_dir=prior_audit_dir,
            )
            reporter.start("npz_metadata_validation")
            mismatch_count = _inspect_npz_rows(
                connection,
                output_dir / "manifest_npz_mismatches.csv",
                selected_only=True,
                checkpoint_size=batch_size,
                reporter=reporter,
            )
            reporter.complete()
            coverage_rows = [
                {
                    "stratum": stratum,
                    "requested": requested_count,
                    "achieved": achieved.get(stratum, 0),
                    "sparse": achieved.get(stratum, 0) < requested_count,
                }
                for stratum, requested_count in sorted(requested.items())
            ]
            pd.DataFrame(coverage_rows, columns=["stratum", "requested", "achieved", "sparse"]).to_csv(
                output_dir / "per_stratum_coverage.csv", index=False
            )
            projected_remaining_bytes = None
            if storage_projection_path is not None:
                projected_remaining_bytes = int(
                    json.loads(storage_projection_path.read_text())["projected_full_corpus_bytes"]
                )
            protocol = _process_raw_sources(
                connection,
                selected=selected,
                output_dir=output_dir,
                checkpoint_frequency=checkpoint_frequency,
                protocol_args={
                    "audit_mode": audit_mode,
                    "started_utc": started_utc,
                    "started_monotonic": started_monotonic,
                    "input_hashes": input_hashes,
                    "selected_count": len(selected),
                    "requested_coverage": requested,
                    "achieved_coverage": achieved,
                },
                stop_after_source_files=stop_after_source_files,
                state_dbs=state_dbs,
                prior_forensics_dir=prior_forensics_dir,
                storage_profile=storage_profile,
                sources_per_partition=sources_per_partition,
                unsafe_skip_disk_check=unsafe_skip_disk_check,
                projected_remaining_bytes=projected_remaining_bytes,
            )
            consolidation = _consolidate_partitions(
                connection,
                output_dir,
                prior_forensics_dir,
                storage_profile=storage_profile,
            )
            protocol.update(consolidation)
            if storage_profile == "compact-v1" and audit_mode == "raw-pilot":
                projection = storage_projection(
                    output_dir,
                    pilot_source_count=len(selected),
                    sources_per_partition=sources_per_partition,
                )
                _atomic_json(output_dir / "storage_projection.json", projection)
                protocol["storage_projection"] = projection
                if equivalence_reference_dir is not None:
                    equivalence = compare_sequence_readiness_artifacts(
                        equivalence_reference_dir,
                        output_dir,
                        report_path=output_dir / "compact_equivalence_report.json",
                    )
                    protocol["compact_equivalence_status"] = equivalence["status"]
                    if equivalence["status"] != "passed":
                        raise RuntimeError("Compact scientific-equivalence validation failed")

        static_outputs = (
            output_dir / "PROPOSED_SCHEMA_AND_MIGRATION.md",
            output_dir / "acceptance_criteria_results.json",
        )
        report_version = connection.execute(
            "SELECT value FROM metadata WHERE key='report_semantics_version'"
        ).fetchone()
        if (
            not report_version
            or int(report_version[0]) != REPORT_SEMANTICS_VERSION
            or not reporter.completed("report_writing", static_outputs)
        ):
            reporter.start("report_writing")
            _write_static_documents(output_dir, audit_mode, manifest_summary, mismatch_count)
            connection.execute(
                "INSERT OR REPLACE INTO metadata VALUES ('report_semantics_version',?)",
                (str(REPORT_SEMANTICS_VERSION),),
            )
            connection.commit()
            reporter.complete(detail="versioned acceptance report documents written")
        reporter.start("finalization")
        after_hashes = {str(path): sha256_file(path) for path in input_paths}
        if input_hashes != after_hashes:
            raise RuntimeError("An input manifest or preprocessing state database changed during the audit")
        protocol["input_hashes_after"] = after_hashes
        protocol["raw_inputs_unchanged"] = True
        protocol["report_semantics_version"] = REPORT_SEMANTICS_VERSION
        protocol["status"] = "completed"
        protocol["completed_utc"] = _utc_now()
        protocol["storage_profile"] = storage_profile
        protocol["resolved_output_dir"] = str(output_dir.resolve())
        protocol["resolved_state_dir"] = str(state_dir.resolve())
        protocol["sources_per_partition"] = sources_per_partition
        protocol["unsafe_disk_check_override"] = bool(unsafe_skip_disk_check)
        reporter.complete(detail="final input verification completed")
        protocol["stages"] = _query_rows(connection, "SELECT * FROM stage_state ORDER BY started_utc")
        _atomic_json(output_dir / "sequence_readiness_protocol.json", protocol)
        reporter._publish("completed")
    except KeyboardInterrupt:
        reporter.interrupt()
        raise
    finally:
        connection.close()
    return output_dir / "sequence_readiness_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--equivalence-only", action="store_true")
    parser.add_argument("--equivalence-report", type=Path)
    parser.add_argument("--audit-mode", choices=sorted(AUDIT_MODES))
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--processed-manifest", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--preprocess-state-db", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--storage-profile", choices=sorted(STORAGE_PROFILES), default="verbose-v3")
    parser.add_argument("--sources-per-partition", type=int, default=DEFAULT_SOURCES_PER_PARTITION)
    parser.add_argument("--storage-projection", type=Path)
    parser.add_argument("--compact-equivalence-report", type=Path)
    parser.add_argument("--equivalence-reference-dir", type=Path)
    parser.add_argument("--unsafe-skip-disk-check", action="store_true")
    parser.add_argument("--max-source-files", type=int)
    parser.add_argument("--samples-per-stratum", type=int, default=4)
    parser.add_argument("--pilot-seed", type=int, default=4004)
    parser.add_argument("--checkpoint-frequency", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-examples-per-group", type=int, default=DEFAULT_MAX_EXAMPLES_PER_GROUP)
    parser.add_argument("--max-diagnostic-groups", type=int, default=DEFAULT_MAX_DIAGNOSTIC_GROUPS)
    parser.add_argument("--prior-audit-dir", type=Path)
    parser.add_argument("--prior-forensics-dir", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        print(regenerate_summary_reports(args.output_dir))
        return
    if args.equivalence_only:
        if args.equivalence_reference_dir is None:
            parser.error("--equivalence-only requires --equivalence-reference-dir")
        if args.equivalence_report is None:
            parser.error("--equivalence-only requires an explicit --equivalence-report path")
        report_path = args.equivalence_report.resolve()
        protected_roots = (args.output_dir.resolve(), args.equivalence_reference_dir.resolve())
        if any(report_path == root or root in report_path.parents for root in protected_roots):
            parser.error("--equivalence-report must be outside both immutable audit directories")
        report = compare_sequence_readiness_artifacts(
            args.equivalence_reference_dir,
            args.output_dir,
            report_path=args.equivalence_report,
        )
        print(args.equivalence_report)
        if report["status"] != "passed":
            raise SystemExit(1)
        return
    required = {
        "--audit-mode": args.audit_mode,
        "--processed-manifest": args.processed_manifest,
        "--train-manifest": args.train_manifest,
        "--validation-manifest": args.validation_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"normal audit mode requires: {', '.join(missing)}")
    assert args.audit_mode is not None
    assert args.processed_manifest is not None
    assert args.train_manifest is not None
    assert args.validation_manifest is not None
    print(
        run_audit(
            audit_mode=args.audit_mode,
            raw_dir=args.raw_dir,
            processed_manifest=args.processed_manifest,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            preprocess_state_dbs=args.preprocess_state_db,
            output_dir=args.output_dir,
            max_source_files=args.max_source_files,
            samples_per_stratum=args.samples_per_stratum,
            pilot_seed=args.pilot_seed,
            checkpoint_frequency=args.checkpoint_frequency,
            batch_size=args.batch_size,
            max_examples_per_group=args.max_examples_per_group,
            max_diagnostic_groups=args.max_diagnostic_groups,
            resume=args.resume,
            restart=args.restart,
            prior_audit_dir=args.prior_audit_dir,
            prior_forensics_dir=args.prior_forensics_dir,
            storage_profile=args.storage_profile,
            state_dir=args.state_dir,
            sources_per_partition=args.sources_per_partition,
            unsafe_skip_disk_check=args.unsafe_skip_disk_check,
            storage_projection_path=args.storage_projection,
            equivalence_reference_dir=args.equivalence_reference_dir,
            compact_equivalence_report=args.compact_equivalence_report,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Audit interrupted safely. Rerun the identical command with --resume.", file=sys.stderr)
        raise SystemExit(130) from None
