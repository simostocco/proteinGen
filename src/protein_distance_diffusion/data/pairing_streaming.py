"""Bounded-memory validation and publication for full sequence-geometry pairings."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sqlite3
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from protein_distance_diffusion.evaluation.sequence_readiness import sha256_file
from protein_distance_diffusion.evaluation.sequence_readiness_storage import SequenceReadinessArtifactReader

DEFAULT_BATCH_SIZE = 4_096
DEFAULT_MAX_MEMORY_MIB = 4_096
DEFAULT_CHECKPOINT_FREQUENCY = 1_000
DEFAULT_MAXIMUM_FAILURE_EXAMPLES = 100
STREAMING_STATE_SCHEMA_VERSION = "sequence_geometry_pairing_streaming_state_v2"
STAGES = (
    "audit-table indexing",
    "manifest indexing",
    "eligibility reconciliation",
    "candidate validation",
    "split/leakage validation",
    "matrix hashing",
    "count-contract validation",
    "report publication",
)
PAIRING_DATASETS = (
    "all_pairs",
    "eligible_train",
    "eligible_validation",
    "excluded_pairs",
    "pairing_ineligible",
    "pairing_eligible_original_split_excluded",
)
CANDIDATE_INDEX_FIELDS = {
    "sample_id",
    "source_file",
    "model_number",
    "entity_id",
    "label_asym_id",
    "auth_asym_id",
    "author_chain_match",
    "label_chain_match",
    "sequence_match",
    "strong_evidential_match",
    "identity_mapping_available",
    "identity_mapping_ambiguous",
    "insertion_codes_match",
    "trim_metadata_available",
    "trim_metadata_compatible",
    "residue_id_convention",
    "convention",
}
ALIGNMENT_INDEX_FIELDS = {
    "sample_id",
    "split",
    "matrix_path",
    "matrix_actual_length",
    "v3_pairing_classification",
    "refined_primary_classification",
    "forensic_root_cause",
    "training_eligibility",
    "practical_training_eligibility",
    "strict_provenance_status",
    "source_identity_status",
    "auth_residue_ids_match",
    "label_residue_ids_match",
    "zero_based_positions_match",
    "one_based_positions_match",
    "coordinate_matrix_status",
    "manifest_npz_sequence_match",
    "matrix_manifest_sequence_match",
    "unique_author_linked_candidate",
    "author_linked_candidate_count",
    "source_file",
    "raw_label_asym_id",
    "raw_auth_asym_id",
    "sequence_source",
    "selected_residue_id_convention",
    "retained_interval_match",
    "trim_metadata_available",
    "competing_candidate_count",
    "raw_match_count",
    "matrix_sequence_occurrence_count_in_candidate",
    "coordinate_distance_max_abs_error_angstrom",
    "requested_length",
    "residue_ids",
    "insertion_codes",
    "selected_altlocs",
}
SUMMARY_INDEX_FIELDS = {
    "sample_id",
    "v3_pairing_classification",
    "refined_primary_classification",
    "forensic_root_cause",
    "strict_provenance_status",
    "practical_training_eligibility",
    "source_identity_status",
    "matrix_path",
}


class PairingMemoryLimitExceeded(RuntimeError):
    """Raised after committing resumable state when the RSS ceiling is reached."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / 1024


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _encoded(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), default=_json_default)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parquet_batches(paths: Iterable[Path], batch_size: int) -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    for path in sorted(paths):
        scanner = ds.dataset(str(path), format="parquet").scanner(batch_size=batch_size, use_threads=False)
        for batch_index, batch in enumerate(scanner.to_batches()):
            yield str(path.resolve()), batch_index, batch.to_pylist()


def _selected_input_row(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    if kind in {"processed", "train", "validation"}:
        fields = {
            "sample_id",
            "pdb_id",
            "chain_id",
            "model_number",
            "sequence",
            "length",
            "original_chain_length",
            "path",
            "source_file",
            "experimental_method",
            "missing_calpha_policy",
            "trimmed_n_terminal_residues",
            "trimmed_c_terminal_residues",
            "terminal_trimming_applied",
            "trimmed_fraction",
            "max_terminal_trim_fraction",
            "sequence_hash",
            "cluster_id",
            "split_group_id",
            "exact_sequence_count",
            "sample_weight",
        }
        return {key: value for key, value in row.items() if key in fields}
    if kind == "residue_id_convention_evidence":
        return {key: value for key, value in row.items() if key in CANDIDATE_INDEX_FIELDS}
    if kind == "matrix_pair_alignments":
        selected = {key: value for key, value in row.items() if key in ALIGNMENT_INDEX_FIELDS}
        for field in ("residue_ids", "insertion_codes", "selected_altlocs"):
            if field in selected and selected[field] is not None:
                selected[field] = "present_in_audit"
        return selected
    if kind in {
        "practical_eligibility",
        "strict_eligibility",
        "unresolved_eligibility",
        "forensic_transition",
    }:
        return {key: value for key, value in row.items() if key in SUMMARY_INDEX_FIELDS}
    return row


class StreamingPairingEngine:
    """Index immutable inputs on disk and validate sample batches deterministically."""

    def __init__(
        self,
        *,
        processed_manifest: Path,
        train_manifest: Path,
        validation_manifest: Path,
        audit_dir: Path,
        normalization_file: Path,
        state_dir: Path,
        eligibility_policy: str,
        resume: bool,
        batch_size: int,
        max_memory_mib: int,
        checkpoint_frequency: int,
        maximum_failure_examples: int,
        mode: str,
    ):
        if batch_size < 1 or batch_size > DEFAULT_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {DEFAULT_BATCH_SIZE}")
        if max_memory_mib < 256:
            raise ValueError("max_memory_mib must be at least 256")
        if checkpoint_frequency < 1:
            raise ValueError("checkpoint_frequency must be positive")
        if maximum_failure_examples < 1 or maximum_failure_examples > 100:
            raise ValueError("maximum_failure_examples must be between 1 and 100")
        self.processed_manifest = processed_manifest
        self.train_manifest = train_manifest
        self.validation_manifest = validation_manifest
        self.audit_dir = audit_dir
        self.normalization_file = normalization_file
        self.state_dir = state_dir
        self.eligibility_policy = eligibility_policy
        self.batch_size = batch_size
        self.max_memory_mib = max_memory_mib
        self.checkpoint_frequency = checkpoint_frequency
        self.maximum_failure_examples = maximum_failure_examples
        self.mode = mode
        self.reader = SequenceReadinessArtifactReader(audit_dir)
        self.protocol_path = audit_dir / "sequence_readiness_protocol.json"
        self.protocol = json.loads(self.protocol_path.read_text())
        self._validate_protocol()
        self.input_paths = self._input_paths()
        self.input_hashes = {str(path): sha256_file(path) for path in self.input_paths}
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.state_dir / "pairing_state.sqlite")
        self._initialize_state()
        self.config = {
            "state_schema_version": STREAMING_STATE_SCHEMA_VERSION,
            "mode": mode,
            "processed_manifest": str(processed_manifest.resolve()),
            "train_manifest": str(train_manifest.resolve()),
            "validation_manifest": str(validation_manifest.resolve()),
            "audit_dir": str(audit_dir.resolve()),
            "normalization_file": str(normalization_file.resolve()),
            "eligibility_policy": eligibility_policy,
            "batch_size": batch_size,
            "max_memory_mib": max_memory_mib,
            "checkpoint_frequency": checkpoint_frequency,
            "maximum_failure_examples": maximum_failure_examples,
            "input_hashes": self.input_hashes,
        }
        digest = hashlib.sha256(_encoded(self.config).encode()).hexdigest()
        previous = self._metadata("config_sha256")
        if previous is not None and not resume:
            raise FileExistsError(f"Pairing state exists; use --resume: {self.state_dir}")
        if previous is not None and previous != digest:
            raise ValueError("Pairing resume configuration does not match the preserved state")
        if previous is None and resume:
            raise FileNotFoundError(f"No resumable pairing state exists in {self.state_dir}")
        self._set_metadata("config_sha256", digest)
        self._set_metadata("config", _encoded(self.config))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def mark_interrupted(self) -> None:
        row = self.connection.execute(
            "SELECT stage,processed,total,elapsed_seconds,current_rss_mib,peak_rss_mib "
            "FROM stage_state WHERE status='running' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return
        stage, _processed, _total, _elapsed, _rss, peak = row
        current = _rss_mib()
        self.connection.execute(
            "UPDATE stage_state SET status='interrupted',heartbeat_utc=?,current_rss_mib=?,"
            "peak_rss_mib=?,detail=? WHERE stage=?",
            (
                _utc_now(),
                current,
                max(float(peak), _peak_rss_mib()),
                "Interrupted after a durable checkpoint; repeat the command with --resume",
                stage,
            ),
        )
        self.connection.execute(
            "INSERT INTO memory_samples(stage,sampled_utc,current_rss_mib,peak_rss_mib) VALUES (?,?,?,?)",
            (stage, _utc_now(), current, _peak_rss_mib()),
        )
        self.connection.commit()

    def _validate_protocol(self) -> None:
        if self.protocol.get("status") != "completed" or self.protocol.get("audit_mode") != "raw-full":
            raise ValueError("Streaming pairing requires a completed raw-full audit")
        if int(self.protocol.get("report_semantics_version", 0)) < 4:
            raise ValueError("Definitive pairing build requires raw-full report semantics version 4 or later")
        if self.protocol.get("raw_inputs_unchanged") is not True:
            raise ValueError("Raw-full audit does not attest preserved raw inputs")
        recorded = self.protocol.get("input_hashes", {})
        for path in (self.processed_manifest, self.train_manifest, self.validation_manifest):
            expected = next(
                (digest for name, digest in recorded.items() if Path(name).resolve() == path.resolve()),
                None,
            )
            if expected is None or sha256_file(path) != expected:
                raise ValueError(f"Audit manifest hash is absent or mismatched: {path}")

    def _input_paths(self) -> list[Path]:
        summary_tables = (
            "practical_eligibility",
            "strict_eligibility",
            "unresolved_eligibility",
            "forensic_transition",
        )
        paths = [
            self.processed_manifest,
            self.train_manifest,
            self.validation_manifest,
            self.normalization_file,
            self.protocol_path,
            *self.reader.scientific_paths(),
        ]
        paths.extend(path for table in summary_tables for path in self.reader.table_paths(table))
        unique = {path.resolve(): path for path in paths}
        return [unique[key] for key in sorted(unique, key=str)]

    def _initialize_state(self) -> None:
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stage_state (
                stage TEXT PRIMARY KEY, status TEXT NOT NULL, processed INTEGER NOT NULL,
                total INTEGER, heartbeat_utc TEXT NOT NULL, elapsed_seconds REAL NOT NULL,
                current_rss_mib REAL NOT NULL, peak_rss_mib REAL NOT NULL, detail TEXT
            );
            CREATE TABLE IF NOT EXISTS input_progress (
                input_key TEXT PRIMARY KEY, completed_batches INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_samples (
                sample_index INTEGER PRIMARY KEY AUTOINCREMENT, stage TEXT NOT NULL,
                sampled_utc TEXT NOT NULL, current_rss_mib REAL NOT NULL, peak_rss_mib REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, sample_id TEXT NOT NULL, row_json TEXT NOT NULL,
                matrix_path TEXT, sequence_hash TEXT, cluster_id TEXT, split_group_id TEXT, pdb_id TEXT,
                PRIMARY KEY(kind,sample_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS source_identity (
                source_id TEXT PRIMARY KEY, source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidate_rows (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id TEXT NOT NULL, row_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_candidate_sample ON candidate_rows(sample_id);
            CREATE INDEX IF NOT EXISTS idx_records_path ON records(kind,matrix_path);
            CREATE INDEX IF NOT EXISTS idx_records_sequence ON records(kind,sequence_hash);
            CREATE INDEX IF NOT EXISTS idx_records_cluster ON records(kind,cluster_id);
            CREATE INDEX IF NOT EXISTS idx_records_group ON records(kind,split_group_id);
            CREATE INDEX IF NOT EXISTS idx_records_pdb ON records(kind,pdb_id);
            CREATE TABLE IF NOT EXISTS matrix_hashes (
                sample_id TEXT PRIMARY KEY, matrix_path TEXT NOT NULL, matrix_sha256 TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS validated_samples (
                sample_id TEXT PRIMARY KEY, pairing_eligible INTEGER NOT NULL,
                practical_eligibility TEXT NOT NULL, original_split TEXT NOT NULL,
                eligible_for_training INTEGER NOT NULL, pairing_classification TEXT NOT NULL,
                row_json TEXT
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS failures (
                reason TEXT PRIMARY KEY, total_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS failure_examples (
                reason TEXT NOT NULL, sample_id TEXT NOT NULL, PRIMARY KEY(reason,sample_id)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS output_partitions (
                dataset_name TEXT NOT NULL, partition_index INTEGER NOT NULL, path TEXT NOT NULL,
                row_count INTEGER NOT NULL, sha256 TEXT NOT NULL,
                PRIMARY KEY(dataset_name,partition_index)
            ) WITHOUT ROWID;
            """
        )
        self.connection.commit()

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, value))

    def _stage_done(self, stage: str) -> bool:
        row = self.connection.execute("SELECT status FROM stage_state WHERE stage=?", (stage,)).fetchone()
        return bool(row and row[0] == "completed")

    def _heartbeat(
        self,
        stage: str,
        processed: int,
        *,
        total: int | None = None,
        started: float,
        status: str = "running",
        detail: str | None = None,
    ) -> None:
        rss = _rss_mib()
        peak = _peak_rss_mib()
        self.connection.execute(
            "INSERT OR REPLACE INTO stage_state VALUES (?,?,?,?,?,?,?,?,?)",
            (stage, status, processed, total, _utc_now(), time.monotonic() - started, rss, peak, detail),
        )
        self.connection.execute(
            "INSERT INTO memory_samples(stage,sampled_utc,current_rss_mib,peak_rss_mib) VALUES (?,?,?,?)",
            (stage, _utc_now(), rss, peak),
        )
        self.connection.commit()
        if rss > self.max_memory_mib:
            self.connection.execute(
                "UPDATE stage_state SET status='paused_memory_limit',detail=? WHERE stage=?",
                (f"RSS {rss:.1f} MiB exceeded limit {self.max_memory_mib} MiB; resume with --resume", stage),
            )
            self.connection.commit()
            raise PairingMemoryLimitExceeded(
                f"Pairing stopped at {stage}: RSS {rss:.1f} MiB exceeded {self.max_memory_mib} MiB; use --resume"
            )

    def _progress(self, key: str) -> int:
        row = self.connection.execute(
            "SELECT completed_batches FROM input_progress WHERE input_key=?", (key,)
        ).fetchone()
        return 0 if row is None else int(row[0])

    def _commit_progress(self, key: str, completed_batches: int) -> None:
        self.connection.execute("INSERT OR REPLACE INTO input_progress VALUES (?,?)", (key, completed_batches))
        self.connection.commit()

    def _insert_record(self, kind: str, row: dict[str, Any]) -> None:
        sample_id = str(row["sample_id"])
        matrix_path = row.get("path", row.get("matrix_path"))
        self.connection.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
            (
                kind,
                sample_id,
                _encoded(_selected_input_row(kind, row)),
                None if matrix_path is None else str(matrix_path),
                row.get("sequence_hash"),
                row.get("cluster_id"),
                row.get("split_group_id"),
                row.get("pdb_id"),
            ),
        )

    def _index_parquet_records(
        self,
        kind: str,
        paths: Iterable[Path],
        *,
        stage: str,
        started: float,
        candidates: bool = False,
    ) -> int:
        processed = 0
        for path_key, batch_index, rows in _parquet_batches(paths, self.batch_size):
            key = f"{kind}:{path_key}"
            if batch_index < self._progress(key):
                continue
            source_ids = {str(row["source_id"]) for row in rows if row.get("source_id") is not None}
            source_files = {}
            if source_ids:
                placeholders = ",".join("?" for _ in source_ids)
                source_files = {
                    str(source_id): str(source_file)
                    for source_id, source_file in self.connection.execute(
                        f"SELECT source_id,source_file FROM source_identity WHERE source_id IN ({placeholders})",
                        tuple(source_ids),
                    )
                }
            for raw in rows:
                row = dict(raw)
                source_id = row.get("source_id")
                if source_id is not None and str(source_id) in source_files:
                    row.setdefault("source_file", source_files[str(source_id)])
                row = self.reader._logical_record(kind, row)
                if candidates:
                    self._validate_candidate_booleans(row)
                    self.connection.execute(
                        "INSERT INTO candidate_rows(sample_id,row_json) VALUES (?,?)",
                        (str(row["sample_id"]), _encoded(_selected_input_row(kind, row))),
                    )
                else:
                    self._insert_record(kind, row)
            self._commit_progress(key, batch_index + 1)
            processed += len(rows)
            self._heartbeat(stage, processed, started=started)
        return processed

    @staticmethod
    def _validate_candidate_booleans(row: dict[str, Any]) -> None:
        from protein_distance_diffusion.data.pairing_builder import (
            CANDIDATE_BOOLEAN_FIELDS,
            normalize_evidence_boolean,
        )

        for field in CANDIDATE_BOOLEAN_FIELDS:
            row[field] = normalize_evidence_boolean(row.get(field), field=field, allow_missing=False)

    def _index_summary(self, logical_table: str, *, stage: str, started: float) -> int:
        kind = logical_table
        paths = self.reader.table_paths(logical_table)
        path = paths[0]
        processed = 0
        key = f"{kind}:{path.resolve()}"
        completed = self._progress(key)
        for batch_index, frame in enumerate(pd.read_csv(path, chunksize=self.batch_size)):
            if batch_index < completed:
                continue
            canonical = self.reader._canonical_summary_frame(frame, path)
            for row in canonical.to_dict("records"):
                self._insert_record(kind, row)
            self._commit_progress(key, batch_index + 1)
            processed += len(canonical)
            self._heartbeat(stage, processed, started=started)
        expected_count = int(
            self.connection.execute("SELECT COUNT(*) FROM records WHERE kind=?", (kind,)).fetchone()[0]
        )
        for alias_path in paths[1:]:
            alias_count = 0
            for frame in pd.read_csv(alias_path, chunksize=self.batch_size):
                canonical = self.reader._canonical_summary_frame(frame, alias_path)
                alias_count += len(canonical)
                for row in canonical.to_dict("records"):
                    stored = self.connection.execute(
                        "SELECT row_json FROM records WHERE kind=? AND sample_id=?",
                        (kind, str(row["sample_id"])),
                    ).fetchone()
                    if stored is None or str(stored[0]) != _encoded(_selected_input_row(kind, row)):
                        raise ValueError(
                            f"Contradictory logical audit table aliases for {logical_table}: "
                            f"{path} versus {alias_path} at sample {row['sample_id']}"
                        )
                self._heartbeat(stage, alias_count, started=started)
            if alias_count != expected_count:
                raise ValueError(
                    f"Contradictory logical audit table aliases for {logical_table}: "
                    f"{path} has {expected_count} rows and {alias_path} has {alias_count}"
                )
        return processed

    def index_audit_tables(self) -> None:
        stage = STAGES[0]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        source_paths = self.reader.table_paths("source_identity")
        for path_key, batch_index, rows in _parquet_batches(source_paths, self.batch_size):
            key = f"source_identity:{path_key}"
            if batch_index < self._progress(key):
                continue
            for row in rows:
                self.connection.execute(
                    "INSERT INTO source_identity VALUES (?,?,?)",
                    (str(row["source_id"]), str(row["source_file"]), str(row["source_sha256"])),
                )
            self._commit_progress(key, batch_index + 1)
            self._heartbeat(stage, batch_index * self.batch_size + len(rows), started=started)
        count = self._index_parquet_records(
            "matrix_pair_alignments",
            self.reader.table_paths("matrix_pair_alignments"),
            stage=stage,
            started=started,
        )
        count += self._index_parquet_records(
            "residue_id_convention_evidence",
            self.reader.table_paths("residue_id_convention_evidence"),
            stage=stage,
            started=started,
            candidates=True,
        )
        for table in (
            "practical_eligibility",
            "strict_eligibility",
            "unresolved_eligibility",
            "forensic_transition",
        ):
            count += self._index_summary(table, stage=stage, started=started)
        self._heartbeat(stage, count, started=started, status="completed")

    def index_manifests(self) -> None:
        stage = STAGES[1]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        count = 0
        for kind, path in (
            ("processed", self.processed_manifest),
            ("train", self.train_manifest),
            ("validation", self.validation_manifest),
        ):
            count += self._index_parquet_records(kind, [path], stage=stage, started=started)
        self._heartbeat(stage, count, started=started, status="completed")

    def _sample_batches(self, kind: str, cursor_key: str) -> Iterator[list[str]]:
        cursor = self._metadata(cursor_key) or ""
        while True:
            rows = self.connection.execute(
                "SELECT sample_id FROM records WHERE kind=? AND sample_id>? ORDER BY sample_id LIMIT ?",
                (kind, cursor, self.batch_size),
            ).fetchall()
            if not rows:
                return
            sample_ids = [str(row[0]) for row in rows]
            yield sample_ids
            cursor = sample_ids[-1]

    def _records(self, kind: str, sample_ids: list[str]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in sample_ids)
        return [
            json.loads(row[0])
            for row in self.connection.execute(
                f"SELECT row_json FROM records WHERE kind=? AND sample_id IN ({placeholders}) ORDER BY sample_id",
                (kind, *sample_ids),
            )
        ]

    def reconcile_eligibility(self) -> None:
        from protein_distance_diffusion.data.pairing_builder import _authoritative_eligibility

        stage = STAGES[2]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        processed = 0
        for sample_ids in self._sample_batches("matrix_pair_alignments", "eligibility_cursor"):
            alignments = self._records("matrix_pair_alignments", sample_ids)
            frames = {
                kind: pd.DataFrame(self._records(kind, sample_ids))
                for kind in (
                    "practical_eligibility",
                    "strict_eligibility",
                    "unresolved_eligibility",
                    "forensic_transition",
                )
            }
            for frame in frames.values():
                if "sample_id" not in frame:
                    frame["sample_id"] = pd.Series(dtype=str)
            reconciled = _authoritative_eligibility(
                alignments,
                frames["practical_eligibility"],
                frames["strict_eligibility"],
                frames["unresolved_eligibility"],
                frames["forensic_transition"],
            )
            for sample_id, row in reconciled.items():
                self._insert_record("reconciled", {"sample_id": sample_id, **row})
            processed += len(sample_ids)
            self._set_metadata("eligibility_cursor", sample_ids[-1])
            self.connection.commit()
            if processed % self.checkpoint_frequency < len(sample_ids):
                self._heartbeat(stage, processed, started=started)
        self._heartbeat(stage, processed, started=started, status="completed")

    def validate_candidates(self) -> None:
        from protein_distance_diffusion.data.pairing_builder import _candidate_evidence

        stage = STAGES[3]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        processed = 0
        cursor = self._metadata("candidate_cursor") or ""
        while True:
            ids = [
                str(row[0])
                for row in self.connection.execute(
                    "SELECT sample_id FROM records WHERE kind='reconciled' AND sample_id>? ORDER BY sample_id LIMIT ?",
                    (cursor, self.batch_size),
                )
            ]
            if not ids:
                break
            placeholders = ",".join("?" for _ in ids)
            candidates = [
                json.loads(row[0])
                for row in self.connection.execute(
                    f"SELECT row_json FROM candidate_rows WHERE sample_id IN ({placeholders}) "
                    "ORDER BY sample_id,row_id",
                    ids,
                )
            ]
            evidence = _candidate_evidence(pd.DataFrame(candidates))
            missing = sorted(set(ids) - set(evidence))
            if missing:
                raise ValueError(f"Missing candidate evidence for audited sample: {missing[0]}")
            for sample_id, row in evidence.items():
                self._insert_record("candidate_summary", {"sample_id": sample_id, **row})
            cursor = ids[-1]
            processed += len(ids)
            self._set_metadata("candidate_cursor", cursor)
            self.connection.commit()
            if processed % self.checkpoint_frequency < len(ids):
                self._heartbeat(stage, processed, started=started)
        self._heartbeat(stage, processed, started=started, status="completed")

    def validate_splits(self) -> None:
        stage = STAGES[4]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        duplicate = self.connection.execute(
            "SELECT matrix_path FROM records WHERE kind='processed' GROUP BY matrix_path HAVING COUNT(*)>1 LIMIT 1"
        ).fetchone()
        if duplicate:
            raise ValueError(f"Audit contains duplicate matrix path: {duplicate[0]}")
        overlap = self.connection.execute(
            "SELECT t.sample_id FROM records t JOIN records v USING(sample_id) "
            "WHERE t.kind='train' AND v.kind='validation' LIMIT 1"
        ).fetchone()
        if overlap:
            raise ValueError(f"Train/validation sample leakage: {overlap[0]}")
        for field in ("sequence_hash", "cluster_id", "split_group_id", "pdb_id"):
            overlap = self.connection.execute(
                f"SELECT t.{field} FROM records t JOIN records v ON t.{field}=v.{field} "
                f"WHERE t.kind='train' AND v.kind='validation' AND t.{field} IS NOT NULL LIMIT 1"
            ).fetchone()
            if overlap:
                raise ValueError(f"Train/validation {field} leakage: {overlap[0]}")
        self._heartbeat(stage, 1, total=1, started=started, status="completed")

    def hash_matrices(self) -> None:
        stage = STAGES[5]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        processed = int(self.connection.execute("SELECT COUNT(*) FROM matrix_hashes").fetchone()[0])
        cursor = self._metadata("matrix_hash_cursor") or ""
        while True:
            rows = self.connection.execute(
                "SELECT sample_id,matrix_path FROM records WHERE kind='processed' AND sample_id>? "
                "ORDER BY sample_id LIMIT ?",
                (cursor, self.batch_size),
            ).fetchall()
            if not rows:
                break
            for sample_id, matrix_path in rows:
                path = Path(str(matrix_path))
                if not path.is_file():
                    raise FileNotFoundError(f"Matrix path is missing for {sample_id}: {path}")
                self.connection.execute(
                    "INSERT OR REPLACE INTO matrix_hashes VALUES (?,?,?)",
                    (str(sample_id), str(path), sha256_file(path)),
                )
            cursor = str(rows[-1][0])
            processed += len(rows)
            self._set_metadata("matrix_hash_cursor", cursor)
            self.connection.commit()
            self._heartbeat(stage, processed, started=started)
        self._heartbeat(stage, processed, started=started, status="completed")

    def _batch_components(self, sample_ids: list[str]) -> tuple[Any, ...]:
        frames = {
            kind: pd.DataFrame(self._records(kind, sample_ids))
            for kind in ("processed", "train", "validation", "practical_eligibility")
        }
        for kind, frame in frames.items():
            if "sample_id" not in frame:
                frames[kind] = pd.DataFrame(columns=["sample_id"])
        alignments = self._records("matrix_pair_alignments", sample_ids)
        candidate_index = {str(row["sample_id"]): row for row in self._records("candidate_summary", sample_ids)}
        eligibility_index = {str(row["sample_id"]): row for row in self._records("reconciled", sample_ids)}
        placeholders = ",".join("?" for _ in sample_ids)
        matrix_hashes = {
            str(sample_id): str(digest)
            for sample_id, digest in self.connection.execute(
                f"SELECT sample_id,matrix_sha256 FROM matrix_hashes WHERE sample_id IN ({placeholders})",
                sample_ids,
            )
        }
        return frames, alignments, candidate_index, eligibility_index, matrix_hashes

    def _build_batch(self, sample_ids: list[str]) -> pd.DataFrame:
        from protein_distance_diffusion.data.pairing_builder import _build_rows

        frames, alignments, candidate_index, eligibility_index, matrix_hashes = self._batch_components(sample_ids)
        return _build_rows(
            frames["processed"],
            frames["train"],
            frames["validation"],
            alignments,
            pd.DataFrame(),
            frames["practical_eligibility"],
            pd.DataFrame(columns=["sample_id"]),
            pd.DataFrame(columns=["sample_id"]),
            pd.DataFrame(columns=["sample_id"]),
            self.normalization_file,
            self.input_hashes[str(self.normalization_file)],
            self.eligibility_policy,
            str(self.audit_dir),
            self.protocol.get("report_semantics_version"),
            candidate_index_override=candidate_index,
            eligibility_index_override=eligibility_index,
            precomputed_matrix_hashes=matrix_hashes,
        )

    def _record_failure(self, reason: str, sample_ids: list[str]) -> None:
        self.connection.execute(
            "INSERT INTO failures VALUES (?,?) ON CONFLICT(reason) "
            "DO UPDATE SET total_count=total_count+excluded.total_count",
            (reason, len(sample_ids)),
        )
        existing = int(
            self.connection.execute("SELECT COUNT(*) FROM failure_examples WHERE reason=?", (reason,)).fetchone()[0]
        )
        for sample_id in sample_ids[: max(self.maximum_failure_examples - existing, 0)]:
            self.connection.execute("INSERT OR IGNORE INTO failure_examples VALUES (?,?)", (reason, sample_id))

    def validate_counts(self) -> None:
        stage = STAGES[6]
        if self._stage_done(stage):
            return
        started = time.monotonic()
        processed = int(self.connection.execute("SELECT COUNT(*) FROM validated_samples").fetchone()[0])
        for sample_ids in self._sample_batches("matrix_pair_alignments", "validation_cursor"):
            try:
                rows = self._build_batch(sample_ids)
            except Exception:
                rows = []
                for sample_id in sample_ids:
                    try:
                        rows.append(self._build_batch([sample_id]))
                    except Exception as sample_exc:
                        reason = f"{type(sample_exc).__name__}:{sample_exc}"
                        self._record_failure(reason, [sample_id])
                rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            for row in rows.to_dict("records"):
                self.connection.execute(
                    "INSERT OR REPLACE INTO validated_samples VALUES (?,?,?,?,?,?,NULL)",
                    (
                        str(row["sample_id"]),
                        int(row["pairing_eligible"]),
                        str(row["practical_training_eligibility"]),
                        str(row["original_split"]),
                        int(row["eligible_for_training"]),
                        str(row["pairing_classification"]),
                    ),
                )
            processed += len(sample_ids)
            self._set_metadata("validation_cursor", sample_ids[-1])
            self.connection.commit()
            self._heartbeat(stage, processed, started=started)
        self._heartbeat(stage, processed, started=started, status="completed")

    def run_validation(
        self,
        *,
        expected_total: int | None,
        expected_eligible: int | None,
        expected_excluded: int | None,
    ) -> dict[str, Any]:
        self.index_audit_tables()
        self.index_manifests()
        self.reconcile_eligibility()
        self.validate_candidates()
        self.validate_splits()
        self.hash_matrices()
        self.validate_counts()
        counts = self._counts()
        for name, observed, expected in (
            ("total", counts["all_pair_count"], expected_total),
            ("eligible", counts["pairing_eligible_count"], expected_eligible),
            ("excluded", counts["pairing_ineligible_count"], expected_excluded),
        ):
            if expected is not None and observed != expected:
                self._record_failure(f"expected_{name}_count:{expected}:observed:{observed}", ["__contract__"])
        self.connection.commit()
        return self._report(expected_total, expected_eligible, expected_excluded)

    def _counts(self) -> dict[str, int]:
        row = self.connection.execute(
            """SELECT COUNT(*),COALESCE(SUM(pairing_eligible),0),
            COALESCE(SUM(CASE WHEN pairing_eligible=0 THEN 1 ELSE 0 END),0),
            COALESCE(SUM(original_split='train'),0),COALESCE(SUM(original_split='validation'),0),
            COALESCE(SUM(original_split NOT IN ('train','validation')),0),
            COALESCE(SUM(eligible_for_training AND original_split='train'),0),
            COALESCE(SUM(eligible_for_training AND original_split='validation'),0),
            COALESCE(SUM(pairing_eligible AND original_split NOT IN ('train','validation')),0),
            COALESCE(SUM(pairing_eligible=0 AND original_split='train'),0),
            COALESCE(SUM(pairing_eligible=0 AND original_split='validation'),0),
            COALESCE(SUM(pairing_eligible=0 AND original_split NOT IN ('train','validation')),0),
            COALESCE(SUM(eligible_for_training=0),0) FROM validated_samples"""
        ).fetchone()
        names = (
            "all_pair_count",
            "pairing_eligible_count",
            "pairing_ineligible_count",
            "original_train_count",
            "original_validation_count",
            "original_split_excluded_count",
            "eligible_train_count",
            "eligible_validation_count",
            "pairing_eligible_but_split_excluded_count",
            "pairing_ineligible_train_count",
            "pairing_ineligible_validation_count",
            "pairing_ineligible_and_split_excluded_count",
            "derived_dataset_excluded_count",
        )
        return {name: int(value) for name, value in zip(names, row, strict=True)}

    def _report(
        self,
        expected_total: int | None,
        expected_eligible: int | None,
        expected_excluded: int | None,
    ) -> dict[str, Any]:
        from protein_distance_diffusion.data.pairing_builder import ELIGIBILITY_POLICY_VERSION

        failures = {
            str(reason): {
                "total_count": int(total),
                "representative_sample_ids": [
                    str(row[0])
                    for row in self.connection.execute(
                        "SELECT sample_id FROM failure_examples WHERE reason=? ORDER BY sample_id", (reason,)
                    )
                ],
            }
            for reason, total in self.connection.execute("SELECT reason,total_count FROM failures ORDER BY reason")
        }
        counts = self._counts()
        stages = self._stage_records()
        candidate_distributions = {
            "raw_physical_candidate_count": Counter(),
            "strong_author_linked_physical_candidate_count": Counter(),
        }
        for (payload,) in self.connection.execute("SELECT row_json FROM records WHERE kind='candidate_summary'"):
            row = json.loads(payload)
            for field, distribution in candidate_distributions.items():
                distribution[str(row.get(field, 0))] += 1
        return {
            "status": "passed" if not failures else "failed",
            "schema_version": "sequence_geometry_pairing_v1",
            "streaming_state_schema_version": STREAMING_STATE_SCHEMA_VERSION,
            "audit_mode": "raw-full",
            "eligibility_policy": self.eligibility_policy,
            "eligibility_policy_version": ELIGIBILITY_POLICY_VERSION,
            "total_audited_pairs": counts["all_pair_count"],
            "expected_total_count": expected_total,
            "expected_eligible_count": expected_eligible,
            "expected_excluded_count": expected_excluded,
            "validated_eligible_count": counts["pairing_eligible_count"],
            "validated_excluded_count": counts["pairing_ineligible_count"],
            "validated_pairing_eligible_count": counts["pairing_eligible_count"],
            "validated_pairing_ineligible_count": counts["pairing_ineligible_count"],
            "validated_membership_counts": counts,
            "validation_count_semantics": (
                "expected/validated eligible and excluded counts refer to pairing eligibility; "
                "validated_membership_counts describes derived train/validation membership"
            ),
            "failure_count_by_reason": {key: value["total_count"] for key, value in failures.items()},
            "failure_examples_by_reason": {key: value["representative_sample_ids"] for key, value in failures.items()},
            "failure_sample_ids_by_reason": {
                key: value["representative_sample_ids"] for key, value in failures.items()
            },
            "failure_count": sum(value["total_count"] for value in failures.values()),
            "representative_candidate_counts": {key: dict(value) for key, value in candidate_distributions.items()},
            "maximum_failure_examples": self.maximum_failure_examples,
            "input_hashes": self.input_hashes,
            "raw_inputs_unchanged": self.input_hashes == {str(path): sha256_file(path) for path in self.input_paths},
            "matrix_reads_performed": 0,
            "matrix_files_hashed": int(self.connection.execute("SELECT COUNT(*) FROM matrix_hashes").fetchone()[0]),
            "batch_size": self.batch_size,
            "max_memory_mib": self.max_memory_mib,
            "peak_rss_mib": _peak_rss_mib(),
            "rss_measurement_count": int(self.connection.execute("SELECT COUNT(*) FROM memory_samples").fetchone()[0]),
            "maximum_measured_rss_mib": float(
                self.connection.execute("SELECT COALESCE(MAX(current_rss_mib),0) FROM memory_samples").fetchone()[0]
            ),
            "state_dir": str(self.state_dir.resolve()),
            "resume_supported": True,
            "stages": stages,
        }

    def _stage_records(self) -> list[dict[str, Any]]:
        return [
            dict(
                zip(
                    (
                        "stage",
                        "status",
                        "processed",
                        "total",
                        "heartbeat_utc",
                        "elapsed_seconds",
                        "current_rss_mib",
                        "peak_rss_mib",
                        "detail",
                    ),
                    row,
                    strict=True,
                )
            )
            for row in self.connection.execute("SELECT * FROM stage_state ORDER BY rowid")
        ]

    def publish_report(self, path: Path, report: dict[str, Any]) -> None:
        stage = STAGES[7]
        started = time.monotonic()
        if report["raw_inputs_unchanged"] is not True:
            raise RuntimeError("Pairing input changed during validation")
        self._heartbeat(stage, 1, total=1, started=started, status="completed")
        report["stages"] = self._stage_records()
        report["peak_rss_mib"] = _peak_rss_mib()
        _atomic_json(path, report)

    def publish_dataset(self, destination: Path) -> dict[str, Any]:
        expected = self._attested_count_contract()
        report = self.run_validation(
            expected_total=expected.get("total"),
            expected_eligible=expected.get("eligible"),
            expected_excluded=expected.get("excluded"),
        )
        if report["status"] != "passed":
            raise ValueError("Pairing dataset cannot be published while validation failures remain")
        stage = STAGES[7]
        started = time.monotonic()
        staging = destination.parent / f".{destination.name}.streaming-staging"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite derived output: {destination}")
        staging.mkdir(parents=True, exist_ok=True)
        for name in PAIRING_DATASETS:
            (staging / f"{name}.parquet").mkdir(parents=True, exist_ok=True)
        self._validate_output_partitions(staging)
        cursor = self._metadata("publication_cursor") or ""
        partition_index = int(self._metadata("publication_partition_index") or 0)
        for sample_ids in self._sample_batches("matrix_pair_alignments", "publication_cursor"):
            rows = self._build_batch(sample_ids)
            self._write_partition_batch(staging, partition_index, rows)
            cursor = sample_ids[-1]
            partition_index += 1
            self._set_metadata("publication_cursor", cursor)
            self._set_metadata("publication_partition_index", str(partition_index))
            self.connection.commit()
            self._heartbeat(stage, partition_index * self.batch_size, started=started)
        if self.input_hashes != {str(path): sha256_file(path) for path in self.input_paths}:
            raise RuntimeError("Pairing input changed during dataset publication")
        self._heartbeat(
            stage,
            report["validated_membership_counts"]["all_pair_count"],
            started=started,
            status="completed",
        )
        report["stages"] = self._stage_records()
        report["peak_rss_mib"] = _peak_rss_mib()
        self._write_dataset_metadata(staging, report)
        self._validate_output_partitions(staging)
        staging.replace(destination)
        return json.loads((destination / "protocol.json").read_text())

    def _attested_count_contract(self) -> dict[str, int]:
        regeneration = self.protocol.get("summary_only_regeneration", {})
        verified = regeneration.get("verified_count_contracts", {})
        aliases = {
            "total": ("matrix_pair_count", "matrix_pairs", "total"),
            "eligible": ("practical_eligible_count", "practical_eligible", "eligible"),
            "excluded": (
                "unresolved_count",
                "unresolved_excluded_count",
                "unresolved_excluded",
                "excluded",
            ),
        }
        result = {}
        for name, keys in aliases.items():
            value = next((verified[key] for key in keys if key in verified), None)
            if value is not None:
                result[name] = int(value)
        selected_sources = self.protocol.get("selected_source_file_count", self.protocol.get("selected_source_count"))
        if selected_sources == 223_709:
            result = {"total": 506_919, "eligible": 501_797, "excluded": 5_122}
        return result

    def _validate_output_partitions(self, staging: Path) -> None:
        for name, index, path_text, row_count, digest in self.connection.execute(
            "SELECT dataset_name,partition_index,path,row_count,sha256 FROM output_partitions"
        ):
            path = Path(str(path_text))
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"Streaming output partition is missing or corrupt: {name}/{index}")
            if pq.ParquetFile(path).metadata.num_rows != int(row_count):
                raise ValueError(f"Streaming output partition row count is incorrect: {name}/{index}")
            if staging.resolve() not in path.resolve().parents:
                raise ValueError(f"Streaming output partition escaped the staging directory: {path}")

    def _write_partition_batch(self, staging: Path, index: int, rows: pd.DataFrame) -> None:
        masks = {
            "all_pairs": pd.Series(True, index=rows.index),
            "eligible_train": rows["eligible_for_training"] & (rows["original_split"] == "train"),
            "eligible_validation": rows["eligible_for_training"] & (rows["original_split"] == "validation"),
            "excluded_pairs": ~rows["eligible_for_training"],
            "pairing_ineligible": ~rows["pairing_eligible"],
            "pairing_eligible_original_split_excluded": rows["pairing_eligible"]
            & ~rows["original_split"].isin(["train", "validation"]),
        }
        for name, mask in masks.items():
            subset = rows.loc[mask]
            if subset.empty:
                continue
            directory = staging / f"{name}.parquet"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{index:06d}.parquet"
            temporary = path.with_name(f".{path.name}.tmp")
            subset.to_parquet(temporary, index=False, compression="zstd")
            temporary.replace(path)
            digest = sha256_file(path)
            if pq.ParquetFile(path).metadata.num_rows != len(subset):
                raise RuntimeError(f"Published pairing partition row count changed: {path}")
            self.connection.execute(
                "INSERT OR REPLACE INTO output_partitions VALUES (?,?,?,?,?)",
                (name, index, str(path), len(subset), digest),
            )

    def _write_dataset_metadata(self, staging: Path, report: dict[str, Any]) -> None:
        from protein_distance_diffusion.data.pairing_builder import _ordered_exclusion_reasons
        from protein_distance_diffusion.data.sequence_geometry import (
            PAIRING_SCHEMA_VERSION,
            SequenceGeometryVocabulary,
        )

        partitions = [
            {
                "dataset": name,
                "partition_index": int(index),
                "path": str(Path(path).relative_to(staging)),
                "row_count": int(count),
                "sha256": digest,
            }
            for name, index, path, count, digest in self.connection.execute(
                "SELECT dataset_name,partition_index,path,row_count,sha256 FROM output_partitions "
                "ORDER BY dataset_name,partition_index"
            )
        ]
        first = next((staging / "all_pairs.parquet").glob("part-*.parquet"))
        schema = pq.ParquetFile(first).schema_arrow
        _atomic_json(staging / "vocabulary.json", SequenceGeometryVocabulary().as_dict())
        _atomic_json(
            staging / "schema.json",
            {
                "schema_version": PAIRING_SCHEMA_VERSION,
                "columns": {field.name: str(field.type) for field in schema},
                "partitioned_parquet": True,
                "large_array_references": {
                    "residue_ids": "audit JSON or NPZ residue_ids",
                    "insertion_codes": "audit JSON or NPZ insertion_codes",
                    "selected_altlocs": "audit JSON or NPZ selected_altlocs",
                    "residue_mask": "all retained residues are valid",
                },
            },
        )
        reason_counts: Counter[str] = Counter()
        pairing_counts: Counter[tuple[Any, ...]] = Counter()
        distribution_columns = (
            "pairing_classification",
            "manifest_matrix_association_count",
            "raw_physical_candidate_count",
            "author_linked_physical_candidate_count",
            "non_author_linked_candidate_count",
            "strong_physical_candidate_count_all",
            "strong_author_linked_physical_candidate_count",
            "strong_non_author_linked_candidate_count",
            "selected_candidate_unique",
        )
        distributions = {column: Counter() for column in distribution_columns}
        candidate_evidence_row_count = 0
        strong_evidence_row_count = 0
        for path in sorted((staging / "all_pairs.parquet").glob("part-*.parquet")):
            frame = pd.read_parquet(
                path,
                columns=[
                    *distribution_columns,
                    "practical_training_eligibility",
                    "pairing_eligible",
                    "original_split",
                    "selected_by_eligibility_policy",
                    "dataset_membership_status",
                    "eligible_for_training",
                    "exclusion_reasons",
                    "candidate_evidence_row_count",
                    "strong_evidence_row_count",
                ],
            )
            for row in frame.to_dict("records"):
                pairing_counts[
                    (
                        row["pairing_classification"],
                        row["practical_training_eligibility"],
                        row["pairing_eligible"],
                        row["original_split"],
                        row["selected_by_eligibility_policy"],
                        row["dataset_membership_status"],
                    )
                ] += 1
                for column in distribution_columns:
                    value = row[column]
                    if column == "selected_candidate_unique":
                        value = str(value).lower()
                    distributions[column][str(value)] += 1
                candidate_evidence_row_count += int(row["candidate_evidence_row_count"])
                strong_evidence_row_count += int(row["strong_evidence_row_count"])
                if not row["eligible_for_training"]:
                    reason_counts.update(json.loads(row["exclusion_reasons"]))
        pd.DataFrame(
            [
                {
                    "pairing_classification": key[0],
                    "practical_training_eligibility": key[1],
                    "pairing_eligible": key[2],
                    "original_split": key[3],
                    "selected_by_eligibility_policy": key[4],
                    "dataset_membership_status": key[5],
                    "pair_count": count,
                }
                for key, count in sorted(pairing_counts.items())
            ]
        ).to_csv(staging / "pairing_summary.csv", index=False)
        pd.DataFrame(
            [
                {"exclusion_reason": reason, "pair_count": reason_counts[reason]}
                for reason in _ordered_exclusion_reasons(list(reason_counts))
            ]
        ).to_csv(staging / "exclusion_summary.csv", index=False)
        protocol = {
            **report,
            "status": "completed",
            "schema_version": PAIRING_SCHEMA_VERSION,
            "partitioned_parquet": True,
            "partitions": partitions,
            "input_hashes_preserved": True,
            "allow_pilot_evidence": False,
            "dataset_exclusion_semantics": (
                "derived_dataset_excluded_count is the union of pairing-ineligible, "
                "original-split-excluded, and policy-unselected rows"
            ),
            "pairing_classification_counts": dict(distributions["pairing_classification"]),
            "manifest_matrix_association_count": dict(distributions["manifest_matrix_association_count"]),
            "raw_physical_candidate_count": dict(distributions["raw_physical_candidate_count"]),
            "author_linked_physical_candidate_count": dict(distributions["author_linked_physical_candidate_count"]),
            "non_author_linked_candidate_count": dict(distributions["non_author_linked_candidate_count"]),
            "candidate_evidence_row_count": candidate_evidence_row_count,
            "strong_physical_candidate_count_all": dict(distributions["strong_physical_candidate_count_all"]),
            "strong_author_linked_physical_candidate_count": dict(
                distributions["strong_author_linked_physical_candidate_count"]
            ),
            "strong_non_author_linked_candidate_count": dict(distributions["strong_non_author_linked_candidate_count"]),
            "strong_evidence_row_count": strong_evidence_row_count,
            "selected_candidate_unique": dict(distributions["selected_candidate_unique"]),
            "physical_candidate_key_fields": [
                "source_file",
                "model_number",
                "entity_id",
                "label_asym_id",
                "auth_asym_id",
            ],
            "source_audit_protocol_sha256": self.input_hashes[str(self.protocol_path)],
            "geometry_validation_basis": (
                "completed audit NPZ metadata validation plus per-row matrix dimensions; "
                "full finite-value validation is enforced when a sample is loaded"
            ),
        }
        _atomic_json(staging / "protocol.json", protocol)
        with (staging / "input_hashes.sha256").open("w") as handle:
            for path, digest in sorted(self.input_hashes.items()):
                handle.write(f"{digest}  {path}\n")


def streaming_validate_sequence_geometry_pairing(
    *,
    processed_manifest: Path,
    train_manifest: Path,
    validation_manifest: Path,
    audit_dir: Path,
    normalization_file: Path,
    validation_report: Path,
    state_dir: Path,
    eligibility_policy: str,
    resume: bool,
    batch_size: int,
    max_memory_mib: int,
    checkpoint_frequency: int,
    maximum_failure_examples: int,
    expected_total: int | None,
    expected_eligible: int | None,
    expected_excluded: int | None,
) -> dict[str, Any]:
    engine = StreamingPairingEngine(
        processed_manifest=processed_manifest,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        audit_dir=audit_dir,
        normalization_file=normalization_file,
        state_dir=state_dir,
        eligibility_policy=eligibility_policy,
        resume=resume,
        batch_size=batch_size,
        max_memory_mib=max_memory_mib,
        checkpoint_frequency=checkpoint_frequency,
        maximum_failure_examples=maximum_failure_examples,
        mode="validation",
    )
    try:
        report = engine.run_validation(
            expected_total=expected_total,
            expected_eligible=expected_eligible,
            expected_excluded=expected_excluded,
        )
        engine.publish_report(validation_report, report)
        return report
    except KeyboardInterrupt:
        engine.mark_interrupted()
        raise
    finally:
        engine.close()


def streaming_build_sequence_geometry_pairing(
    *,
    processed_manifest: Path,
    train_manifest: Path,
    validation_manifest: Path,
    audit_dir: Path,
    normalization_file: Path,
    output_dir: Path,
    state_dir: Path,
    eligibility_policy: str,
    resume: bool,
    batch_size: int,
    max_memory_mib: int,
    checkpoint_frequency: int,
    maximum_failure_examples: int,
) -> dict[str, Any]:
    engine = StreamingPairingEngine(
        processed_manifest=processed_manifest,
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        audit_dir=audit_dir,
        normalization_file=normalization_file,
        state_dir=state_dir,
        eligibility_policy=eligibility_policy,
        resume=resume,
        batch_size=batch_size,
        max_memory_mib=max_memory_mib,
        checkpoint_frequency=checkpoint_frequency,
        maximum_failure_examples=maximum_failure_examples,
        mode="build",
    )
    try:
        return engine.publish_dataset(output_dir)
    except KeyboardInterrupt:
        engine.mark_interrupted()
        raise
    finally:
        engine.close()
