"""Versioned storage and readers for sequence-readiness raw evidence."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STORAGE_PROFILES = frozenset({"verbose-v3", "compact-v1"})
DEFAULT_SOURCES_PER_PARTITION = 1_000
FULL_CORPUS_SOURCE_COUNT = 223_709
SAFETY_RESERVE_BYTES = 20 * 1024**3
CANDIDATE_BOOLEAN_COLUMNS = frozenset(
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

COMPACT_TABLE_KEYS = {
    "protein_chain_models": "raw_rows",
    "matrix_pair_alignments": "alignments",
    "residue_id_convention_evidence": "candidate_evidence",
    "residue_tokens": "tokens",
    "excluded_nonpolymer_summaries": "nonpolymer_context",
    "nmr_cross_model_consistency": "nmr_summaries",
}

LEGACY_PATHS = {
    "protein_chain_models": "raw_sequence_provenance.jsonl",
    "matrix_pair_alignments": "seqres_atom_matrix_alignments.jsonl",
    "residue_id_convention_evidence": "candidate_resolution_evidence.parquet",
    "residue_tokens": "raw_residue_tokens.jsonl",
    "excluded_nonpolymer_summaries": "excluded_nonpolymer_context.jsonl",
    "nmr_cross_model_consistency": "nmr_source_chain_consistency.jsonl",
    "modified_residue_evidence": "modified_residues.jsonl",
    "missing_calpha_outcomes": "missing_residues_and_calpha.jsonl",
    "blocking_failures": "raw_blocking_failures.csv",
}
SUMMARY_TABLE_PATHS = {
    "practical_eligibility": ("practical_training_eligibility.csv",),
    "strict_eligibility": ("strict_training_eligibility.csv",),
    "unresolved_eligibility": ("unresolved_case_summary.csv", "unresolved_cases.csv"),
    "forensic_transition": ("v2_forensic_v3_transition.csv",),
}
SUMMARY_COLUMN_ALIASES = {
    "training_eligibility": "practical_training_eligibility",
    "model_number": "model_id",
}


def _canonical_identifier(value: Any) -> str | None:
    if value is None or (isinstance(value, numbers.Real) and math.isnan(float(value))):
        return None
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


LEGACY_TABLES_WITH_SOURCE_FILE = frozenset(
    {
        "protein_chain_models",
        "residue_id_convention_evidence",
        "excluded_nonpolymer_summaries",
        "nmr_cross_model_consistency",
        "modified_residue_evidence",
        "missing_calpha_outcomes",
    }
)
LOGICAL_IDENTIFIER_FIELDS = frozenset(
    {
        "sample_id",
        "pdb_id",
        "entity_id",
        "label_asym_id",
        "auth_asym_id",
        "model_id",
        "model_number",
        "candidate_index",
        "evidence_row_index",
        "physical_candidate_id",
        "residue_id_convention",
    }
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalized_row(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        **{key: _jsonable_scalar(value) for key, value in row.items() if key not in {"source_file", "source_sha256"}},
    }


def _normalized_candidate_row(row: dict[str, Any], source_id: str) -> dict[str, Any]:
    normalized = _normalized_row(row, source_id)
    output = {}
    for key, value in normalized.items():
        if key in CANDIDATE_BOOLEAN_COLUMNS:
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                output[key] = value.lower() == "true"
            elif isinstance(value, bool):
                output[key] = value
            else:
                raise ValueError(f"Candidate evidence {key} must be Boolean")
        elif key == "source_id" or value is None:
            output[key] = value
        else:
            output[key] = str(value)
    return output


def compact_tables_for_sources(sources: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Normalize source payloads into compact logical tables."""
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sources:
        source = str(item["source_file"])
        source_id = hashlib.sha256(source.encode("utf-8")).hexdigest()
        tables["source_identity"].append(
            {
                "source_id": source_id,
                "source_file": source,
                "source_sha256": str(item["source_sha256"]),
                "parse_status": str(item["status"]),
                "error_type": item.get("error_type"),
                "error_message": item.get("error_message"),
                "parser_calls": int(item.get("parser_calls", 1)),
            }
        )
        payload = item.get("payload") or {}
        for table, key in COMPACT_TABLE_KEYS.items():
            normalizer = _normalized_candidate_row if table == "residue_id_convention_evidence" else _normalized_row
            tables[table].extend(normalizer(row, source_id) for row in payload.get(key, []))
        raw_rows = payload.get("raw_rows", [])
        tables["modified_residue_evidence"].extend(
            _normalized_row(row, source_id)
            for row in raw_rows
            if row.get("modified_residue_counts") not in {None, "{}"}
        )
        tables["missing_calpha_outcomes"].extend(
            _normalized_row(row, source_id)
            for row in raw_rows
            if row.get("missing_residue_count", 0) or row.get("missing_calpha_count", 0)
        )
        tables["blocking_failures"].extend(
            _normalized_row(row, source_id)
            for row in payload.get("alignments", [])
            if row.get("training_eligibility") == "excluded_or_unresolved"
        )
        physical = {}
        for row in payload.get("candidate_evidence", []):
            candidate_id = str(row.get("physical_candidate_id", ""))
            if candidate_id and candidate_id not in physical:
                physical[candidate_id] = {
                    key: row.get(key)
                    for key in (
                        "physical_candidate_id",
                        "sample_id",
                        "model_number",
                        "entity_id",
                        "label_asym_id",
                        "auth_asym_id",
                        "author_chain_match",
                        "label_chain_match",
                    )
                }
        tables["physical_candidates"].extend(_normalized_row(row, source_id) for row in physical.values())
    return dict(tables)


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({"source_id": pa.array([], type=pa.string())})
    columns = sorted({key for row in rows for key in row})
    normalized = [{key: row.get(key) for key in columns} for row in rows]
    return pa.Table.from_pylist(normalized)


def write_compact_partition(
    output_dir: Path,
    *,
    partition_index: int,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Atomically publish one source-range partition and its commit marker."""
    tables = compact_tables_for_sources(sources)
    table_metadata = {}
    table_names = {
        "source_identity",
        *COMPACT_TABLE_KEYS,
        "physical_candidates",
        "modified_residue_evidence",
        "missing_calpha_outcomes",
        "blocking_failures",
    }
    for table_name in sorted(table_names):
        directory = output_dir / "tables" / table_name
        directory.mkdir(parents=True, exist_ok=True)
        final = directory / f"part-{partition_index:06d}.parquet"
        temporary = directory / f".{final.name}.{os.getpid()}.tmp"
        table = _table_from_rows(tables.get(table_name, []))
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        temporary.replace(final)
        table_metadata[table_name] = {
            "path": str(final.relative_to(output_dir)),
            "row_count": table.num_rows,
            "sha256": sha256_path(final),
            "bytes": final.stat().st_size,
        }
    marker = {
        "storage_profile": "compact-v1",
        "partition_index": partition_index,
        "source_count": len(sources),
        "source_ids": [row["source_id"] for row in tables["source_identity"]],
        "tables": table_metadata,
    }
    marker_dir = output_dir / "partition_commits"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"part-{partition_index:06d}.json"
    temporary = marker_dir / f".{marker_path.name}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker_path)
    marker["marker_path"] = str(marker_path)
    return marker


def validate_compact_partition(output_dir: Path, marker: dict[str, Any]) -> None:
    seen_paths = set()
    for metadata in marker["tables"].values():
        path = output_dir / metadata["path"]
        if path in seen_paths:
            raise ValueError(f"Duplicated compact partition artifact: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing compact partition artifact: {path}")
        if sha256_path(path) != metadata["sha256"]:
            raise ValueError(f"Corrupt compact partition artifact: {path}")
        if pq.ParquetFile(path).metadata.num_rows != int(metadata["row_count"]):
            raise ValueError(f"Compact partition row-count mismatch: {path}")


class SequenceReadinessArtifactReader:
    """Expose common logical records from verbose-v3 and compact-v1 audits."""

    def __init__(self, audit_dir: str | Path):
        self.audit_dir = Path(audit_dir)
        config_path = self.audit_dir / "run_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        self.storage_profile = str(config.get("storage_profile", "verbose-v3"))
        if self.storage_profile not in STORAGE_PROFILES:
            raise ValueError(f"Unsupported sequence-readiness storage profile: {self.storage_profile}")

    def iter_records(self, logical_table: str) -> Iterator[dict[str, Any]]:
        if logical_table in SUMMARY_TABLE_PATHS:
            yield from self._summary_records(logical_table)
            return
        if self.storage_profile == "verbose-v3":
            path = self.audit_dir / LEGACY_PATHS[logical_table]
            if path.suffix == ".parquet":
                for batch in pq.ParquetFile(path).iter_batches():
                    for row in batch.to_pylist():
                        yield self._logical_record(logical_table, row)
            elif path.suffix == ".csv":
                for row in pd.read_csv(path).to_dict("records"):
                    yield self._logical_record(logical_table, row)
            else:
                with path.open() as handle:
                    for line in handle:
                        if line.strip():
                            yield self._logical_record(logical_table, json.loads(line))
            return
        source_identities = self._source_identities()
        directory = self.audit_dir / "tables" / logical_table
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("part-*.parquet")):
            for batch in pq.ParquetFile(path).iter_batches():
                for row in batch.to_pylist():
                    source_id = str(row.pop("source_id"))
                    identity = source_identities[source_id]
                    if logical_table in LEGACY_TABLES_WITH_SOURCE_FILE:
                        row.setdefault("source_file", identity["source_file"])
                    if logical_table == "protein_chain_models":
                        row.setdefault("source_sha256", identity["source_sha256"])
                    yield self._logical_record(logical_table, row)

    def frame(self, logical_table: str) -> pd.DataFrame:
        if logical_table in SUMMARY_TABLE_PATHS:
            return self._summary_frame(logical_table)
        return pd.DataFrame(self.iter_records(logical_table))

    def table_paths(self, logical_table: str) -> list[Path]:
        if logical_table in SUMMARY_TABLE_PATHS:
            paths = [self.audit_dir / name for name in SUMMARY_TABLE_PATHS[logical_table]]
            existing = [path for path in paths if path.is_file()]
            if not existing:
                alternatives = ", ".join(str(path) for path in paths)
                raise FileNotFoundError(f"Missing logical audit table {logical_table}; expected one of: {alternatives}")
            return existing
        if self.storage_profile == "verbose-v3":
            path = self.audit_dir / LEGACY_PATHS[logical_table]
            if not path.is_file():
                raise FileNotFoundError(f"Missing logical audit table {logical_table}: {path}")
            return [path]
        paths = sorted((self.audit_dir / "tables" / logical_table).glob("part-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"Missing compact logical audit table {logical_table}")
        return paths

    def scientific_paths(self) -> list[Path]:
        if self.storage_profile == "verbose-v3":
            return [self.audit_dir / path for path in LEGACY_PATHS.values() if (self.audit_dir / path).is_file()]
        return sorted((self.audit_dir / "tables").glob("*/*.parquet"))

    def _source_identities(self) -> dict[str, dict[str, str]]:
        identities = {}
        directory = self.audit_dir / "tables" / "source_identity"
        for path in sorted(directory.glob("part-*.parquet")):
            columns = ["source_id", "source_file", "source_sha256"]
            for row in pq.read_table(path, columns=columns).to_pylist():
                source_id = str(row["source_id"])
                if source_id in identities:
                    raise ValueError(f"Duplicate compact source identity: {source_id}")
                identities[source_id] = {
                    "source_file": str(row["source_file"]),
                    "source_sha256": str(row["source_sha256"]),
                }
        return identities

    def _summary_records(self, logical_table: str) -> Iterator[dict[str, Any]]:
        yield from self._summary_frame(logical_table).to_dict("records")

    def _summary_frame(self, logical_table: str) -> pd.DataFrame:
        paths = self.table_paths(logical_table)
        frames = [self._canonical_summary_frame(pd.read_csv(path), path) for path in paths]
        reference = frames[0]
        for path, candidate in zip(paths[1:], frames[1:], strict=True):
            self._validate_summary_equality(logical_table, paths[0], reference, path, candidate)
        return reference

    @staticmethod
    def _canonical_summary_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
        frame = frame.copy()
        for alias, canonical in SUMMARY_COLUMN_ALIASES.items():
            if alias not in frame:
                continue
            if canonical in frame:
                conflicts = []
                for index, (canonical_value, alias_value) in enumerate(
                    zip(frame[canonical], frame[alias], strict=True)
                ):
                    left = _canonical_identifier(canonical_value)
                    right = _canonical_identifier(alias_value)
                    if left is not None and right is not None and left != right:
                        conflicts.append(index)
                if conflicts:
                    sample = frame.iloc[conflicts[0]].get("sample_id", conflicts[0])
                    raise ValueError(
                        f"Contradictory audit aliases in {path} for sample {sample}: {alias} versus {canonical}"
                    )
                frame[canonical] = frame[canonical].where(frame[canonical].notna(), frame[alias])
            else:
                frame[canonical] = frame[alias]
            frame = frame.drop(columns=alias)
        if "sample_id" not in frame:
            raise ValueError(f"Logical audit table has no sample_id column: {path}")
        frame["sample_id"] = frame["sample_id"].astype(str)
        if frame["sample_id"].duplicated().any():
            duplicate = frame.loc[frame["sample_id"].duplicated(keep=False), "sample_id"].iloc[0]
            raise ValueError(f"Logical audit table contains duplicate sample_id in {path}: {duplicate}")
        if "model_id" in frame:
            frame["model_id"] = frame["model_id"].map(_canonical_identifier)
        return frame.sort_values("sample_id", kind="stable").reset_index(drop=True)

    @staticmethod
    def _validate_summary_equality(
        logical_table: str,
        reference_path: Path,
        reference: pd.DataFrame,
        candidate_path: Path,
        candidate: pd.DataFrame,
    ) -> None:
        columns = sorted(set(reference) | set(candidate))

        def normalized_rows(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
            output = {}
            for row in frame.to_dict("records"):
                normalized = {}
                for column in columns:
                    value = row.get(column)
                    if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
                        value = None
                    normalized[column] = _normalized_evidence(value)
                output[str(row["sample_id"])] = normalized
            return output

        reference_rows = normalized_rows(reference)
        candidate_rows = normalized_rows(candidate)
        if reference_rows != candidate_rows:
            all_ids = sorted(set(reference_rows) | set(candidate_rows))
            sample = next(
                sample_id for sample_id in all_ids if reference_rows.get(sample_id) != candidate_rows.get(sample_id)
            )
            raise ValueError(
                f"Contradictory {logical_table} audit tables for sample {sample}: "
                f"{reference_path} versus {candidate_path}"
            )

    @staticmethod
    def _logical_record(logical_table: str, row: dict[str, Any]) -> dict[str, Any]:
        row = dict(row)
        row.pop("source_id", None)
        for field in LOGICAL_IDENTIFIER_FIELDS & row.keys():
            row[field] = _canonical_identifier(row[field])
        for alias, canonical in SUMMARY_COLUMN_ALIASES.items():
            if alias not in row:
                continue
            alias_value = row[alias]
            canonical_value = row.get(canonical)
            left = _canonical_identifier(canonical_value) if canonical == "model_id" else canonical_value
            right = _canonical_identifier(alias_value) if canonical == "model_id" else alias_value
            if canonical in row and left is not None and right is not None and str(left) != str(right):
                raise ValueError(f"Contradictory logical audit aliases: {alias} versus {canonical}")
            if canonical not in row or canonical_value is None:
                row[canonical] = right
        if logical_table != "residue_id_convention_evidence":
            return row
        for field in CANDIDATE_BOOLEAN_COLUMNS:
            value = row.get(field)
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                row[field] = value.lower() == "true"
        convention = row.get("residue_id_convention", row.get("convention"))
        row["residue_id_convention"] = convention
        row.setdefault("convention", convention)
        evidence_index = row.get("evidence_row_index", row.get("candidate_index"))
        row["evidence_row_index"] = str(evidence_index) if evidence_index is not None else None
        row["candidate_index"] = str(row["candidate_index"]) if row.get("candidate_index") is not None else None
        if not row.get("physical_candidate_id"):
            physical_key = (
                str(row.get("source_file")),
                str(row.get("model_number")),
                str(row.get("entity_id")),
                str(row.get("label_asym_id")),
                str(row.get("auth_asym_id")),
            )
            row["physical_candidate_id"] = hashlib.sha256(
                json.dumps(physical_key, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return row

    def iter_source_payloads(self) -> Iterator[dict[str, Any]]:
        if self.storage_profile == "verbose-v3":
            raise ValueError("Source payload iteration is only used by compact-v1 consolidation")
        for source_partition in sorted((self.audit_dir / "tables" / "source_identity").glob("part-*.parquet")):
            source_rows = pq.read_table(source_partition).to_pylist()
            paths = {str(row["source_id"]): str(row["source_file"]) for row in source_rows}
            grouped: dict[str, dict[str, Any]] = {source: {} for source in paths.values()}
            for table_name, payload_key in COMPACT_TABLE_KEYS.items():
                path = self.audit_dir / "tables" / table_name / source_partition.name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing compact logical partition: {path}")
                for row in pq.read_table(path).to_pylist():
                    source = paths[str(row.pop("source_id"))]
                    if table_name in LEGACY_TABLES_WITH_SOURCE_FILE:
                        row.setdefault("source_file", source)
                    grouped[source].setdefault(payload_key, []).append(row)
            for source in sorted(grouped):
                yield {"source_file": source, **grouped[source]}


def storage_projection(
    output_dir: Path,
    *,
    pilot_source_count: int,
    full_source_count: int = FULL_CORPUS_SOURCE_COUNT,
    sources_per_partition: int = DEFAULT_SOURCES_PER_PARTITION,
) -> dict[str, Any]:
    bytes_by_table = {
        path.name: sum(file.stat().st_size for file in path.glob("part-*.parquet"))
        for path in sorted((output_dir / "tables").iterdir())
        if path.is_dir()
    }
    total = sum(bytes_by_table.values())
    per_source = total / max(pilot_source_count, 1)
    projected = int(per_source * full_source_count)
    free = os.statvfs(output_dir).f_bavail * os.statvfs(output_dir).f_frsize
    required = 2 * projected + SAFETY_RESERVE_BYTES
    return {
        "pilot_source_count": pilot_source_count,
        "bytes_by_logical_table": bytes_by_table,
        "total_pilot_bytes": total,
        "bytes_per_source": per_source,
        "projected_full_corpus_bytes": projected,
        "projected_remaining_bytes": projected,
        "full_corpus_source_count": full_source_count,
        "expected_partition_count": (full_source_count + sources_per_partition - 1) // sources_per_partition,
        "current_free_bytes": free,
        "required_safety_reserve_bytes": SAFETY_RESERVE_BYTES,
        "required_free_bytes": required,
        "disk_guard_passed": free >= required,
    }


def _normalized_evidence(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return _normalized_evidence(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        return {key: _normalized_evidence(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalized_evidence(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return value
    return value


def _logical_hash(rows: list[dict[str, Any]]) -> str:
    encoded = [json.dumps(_normalized_evidence(row), sort_keys=True, separators=(",", ":")) for row in rows]
    return hashlib.sha256("\n".join(sorted(encoded)).encode("utf-8")).hexdigest()


def _evidence_equal(
    left: Any,
    right: Any,
    *,
    tolerance: float = 1e-4,
    numeric_tolerant: bool = False,
) -> bool:
    left = _normalized_evidence(left)
    right = _normalized_evidence(right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _evidence_equal(
                left[key],
                right[key],
                tolerance=tolerance,
                numeric_tolerant=numeric_tolerant or "coordinate" in key or "angstrom" in key,
            )
            for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _evidence_equal(a, b, tolerance=tolerance, numeric_tolerant=numeric_tolerant)
            for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
            if numeric_tolerant
            else left == right and type(left) is type(right)
        )
    return left == right


def _sorted_logical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_fields = (
        "source_file",
        "sample_id",
        "physical_candidate_id",
        "candidate_index",
        "evidence_row_index",
        "pdb_id",
        "model_id",
        "label_asym_id",
        "auth_asym_id",
    )
    return sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in identity_fields))


LOGICAL_PRIMARY_KEYS = {
    "protein_chain_models": (
        "source_file",
        "source_sha256",
        "pdb_id",
        "entity_id",
        "label_asym_id",
        "auth_asym_id",
        "model_id",
    ),
    "residue_id_convention_evidence": (
        "sample_id",
        "physical_candidate_id",
        "residue_id_convention",
        "evidence_row_index",
    ),
}
_MISSING = object()


def _type_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "null"
    return type(value).__name__


def _logical_schema(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    columns = sorted({key for row in rows for key in row})
    return {column: sorted({_type_name(row[column]) for row in rows if column in row}) for column in columns}


def _raw_schema(reader: SequenceReadinessArtifactReader, logical_table: str) -> dict[str, str]:
    if reader.storage_profile == "compact-v1":
        files = sorted((reader.audit_dir / "tables" / logical_table).glob("part-*.parquet"))
        schemas = [pq.ParquetFile(path).schema_arrow for path in files]
        columns = sorted({field.name for schema in schemas for field in schema})
        return {
            column: " | ".join(sorted({str(schema.field(column).type) for schema in schemas if column in schema.names}))
            for column in columns
        }
    path = reader.audit_dir / LEGACY_PATHS[logical_table]
    if path.suffix == ".parquet":
        schema = pq.ParquetFile(path).schema_arrow
        return {field.name: str(field.type) for field in schema}
    rows = list(reader.iter_records(logical_table))
    return {column: " | ".join(types) for column, types in _logical_schema(rows).items()}


def _key_value(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return str(value)


def _index_by_primary_key(
    rows: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> tuple[dict[tuple[str | None, ...], dict[str, Any]], int, int]:
    index = {}
    duplicate_keys = set()
    duplicate_rows = 0
    for row in rows:
        key = tuple(_key_value(row.get(field)) for field in key_fields)
        if key in index:
            duplicate_keys.add(key)
            duplicate_rows += 1
        else:
            index[key] = row
    return index, len(duplicate_keys), duplicate_rows


def _is_null_representation(value: Any) -> bool:
    return value is _MISSING or value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def _decoded(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_decoded(item) for item in value]
    return _normalized_evidence(value)


def _classify_field_difference(field: str, reference: Any, candidate: Any) -> tuple[str, bool]:
    both_nan = (
        isinstance(reference, float)
        and isinstance(candidate, float)
        and math.isnan(reference)
        and math.isnan(candidate)
    )
    if (
        reference is candidate
        or both_nan
        or (
            reference is not _MISSING
            and candidate is not _MISSING
            and not isinstance(reference, (dict, list, tuple))
            and type(reference) is type(candidate)
            and reference == candidate
        )
    ):
        return "equal", True
    if reference is _MISSING or candidate is _MISSING:
        if _is_null_representation(reference) and _is_null_representation(candidate):
            return "null_normalization", True
        return "missing_field", False
    if _is_null_representation(reference) or _is_null_representation(candidate):
        if _is_null_representation(reference) and _is_null_representation(candidate):
            return "null_normalization", True
        return "genuinely_different_value", False
    left = _decoded(reference)
    right = _decoded(candidate)
    if left == right:
        if type(reference) is not type(candidate):
            return "type_normalization", True
        return "equal", True
    if isinstance(left, list) and isinstance(right, list):
        return "list_order_normalization", False
    coordinate = "coordinate" in field or "angstrom" in field
    if coordinate and isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-4):
            return "numeric_tolerance", True
    return "genuinely_different_value", False


def _keyed_table_diagnostics(
    table: str,
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    max_examples: int,
) -> dict[str, Any]:
    key_fields = LOGICAL_PRIMARY_KEYS[table]
    reference_index, reference_duplicate_keys, reference_duplicate_rows = _index_by_primary_key(
        reference_rows, key_fields
    )
    candidate_index, candidate_duplicate_keys, candidate_duplicate_rows = _index_by_primary_key(
        candidate_rows, key_fields
    )
    reference_keys = set(reference_index)
    candidate_keys = set(candidate_index)
    missing_keys = sorted(reference_keys - candidate_keys, key=str)
    extra_keys = sorted(candidate_keys - reference_keys, key=str)
    mismatch_counts: defaultdict[str, int] = defaultdict(int)
    category_counts: defaultdict[str, int] = defaultdict(int)
    representatives = []

    def add_example(category: str, key: tuple[Any, ...], field: str | None, left: Any, right: Any) -> None:
        if len(representatives) >= max_examples:
            return
        representatives.append(
            {
                "category": category,
                "primary_key": dict(zip(key_fields, key, strict=True)),
                "field": field,
                "reference_value": None if left is _MISSING else left,
                "candidate_value": None if right is _MISSING else right,
            }
        )

    for key in missing_keys:
        category_counts["incorrect_row_association"] += 1
        add_example("incorrect_row_association", key, None, reference_index[key], _MISSING)
    for key in extra_keys:
        category_counts["incorrect_row_association"] += 1
        add_example("incorrect_row_association", key, None, _MISSING, candidate_index[key])
    for key in sorted(reference_keys & candidate_keys, key=str):
        reference_row = reference_index[key]
        candidate_row = candidate_index[key]
        for field in sorted(set(reference_row) | set(candidate_row)):
            left = reference_row.get(field, _MISSING)
            right = candidate_row.get(field, _MISSING)
            category, equivalent = _classify_field_difference(field, left, right)
            if category == "equal":
                continue
            category_counts[category] += 1
            if not equivalent:
                mismatch_counts[field] += 1
            add_example(category, key, field, left, right)
    failed = bool(missing_keys or extra_keys or mismatch_counts or reference_duplicate_keys or candidate_duplicate_keys)
    return {
        "primary_key": list(key_fields),
        "reference_duplicate_primary_key_count": reference_duplicate_keys,
        "reference_duplicate_row_count": reference_duplicate_rows,
        "candidate_duplicate_primary_key_count": candidate_duplicate_keys,
        "candidate_duplicate_row_count": candidate_duplicate_rows,
        "missing_logical_row_count": len(missing_keys),
        "extra_logical_row_count": len(extra_keys),
        "per_column_mismatch_counts": dict(sorted(mismatch_counts.items())),
        "difference_category_counts": dict(sorted(category_counts.items())),
        "representative_differences": representatives,
        "representative_limit": max_examples,
        "representatives_truncated": sum(category_counts.values()) > len(representatives),
        "equal": not failed,
    }


def compare_sequence_readiness_artifacts(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    report_path: str | Path,
    max_representative_mismatches: int = 20,
) -> dict[str, Any]:
    """Compare compact evidence with a legacy audit after deterministic normalization."""
    reference = SequenceReadinessArtifactReader(reference_dir)
    candidate = SequenceReadinessArtifactReader(candidate_dir)
    tables = sorted(COMPACT_TABLE_KEYS)
    results = {}
    discrepancies = []
    for table in tables:
        reference_rows = list(reference.iter_records(table))
        candidate_rows = list(candidate.iter_records(table))
        reference_hash = _logical_hash(reference_rows)
        candidate_hash = _logical_hash(candidate_rows)
        reference_columns = set().union(*(row.keys() for row in reference_rows)) if reference_rows else set()
        candidate_columns = set().union(*(row.keys() for row in candidate_rows)) if candidate_rows else set()
        if table in LOGICAL_PRIMARY_KEYS:
            diagnostics = _keyed_table_diagnostics(
                table,
                reference_rows,
                candidate_rows,
                max_examples=max_representative_mismatches,
            )
            equal = diagnostics["equal"]
        else:
            sorted_reference = _sorted_logical_rows(reference_rows)
            sorted_candidate = _sorted_logical_rows(candidate_rows)
            equal = len(reference_rows) == len(candidate_rows) and all(
                _evidence_equal(left, right) for left, right in zip(sorted_reference, sorted_candidate, strict=True)
            )
            diagnostics = {
                "primary_key": None,
                "reference_duplicate_primary_key_count": None,
                "reference_duplicate_row_count": None,
                "candidate_duplicate_primary_key_count": None,
                "candidate_duplicate_row_count": None,
                "missing_logical_row_count": max(len(reference_rows) - len(candidate_rows), 0),
                "extra_logical_row_count": max(len(candidate_rows) - len(reference_rows), 0),
                "per_column_mismatch_counts": {},
                "difference_category_counts": {},
                "representative_differences": [],
                "representative_limit": max_representative_mismatches,
                "representatives_truncated": False,
            }
        results[table] = {
            "reference_row_count": len(reference_rows),
            "candidate_row_count": len(candidate_rows),
            "reference_logical_sha256": reference_hash,
            "candidate_logical_sha256": candidate_hash,
            "equal": equal,
            "numeric_tolerance_angstrom": 1e-4,
            "reference_physical_schema": _raw_schema(reference, table),
            "candidate_physical_schema": _raw_schema(candidate, table),
            "reference_logical_schema": _logical_schema(reference_rows),
            "candidate_logical_schema": _logical_schema(candidate_rows),
            "reference_only_logical_columns": sorted(reference_columns - candidate_columns),
            "candidate_only_logical_columns": sorted(candidate_columns - reference_columns),
            **diagnostics,
        }
        if not equal:
            discrepancies.append(table)
    alignments = list(candidate.iter_records("matrix_pair_alignments"))
    classifications = defaultdict(int)
    eligible = 0
    for row in alignments:
        status = str(row.get("training_eligibility"))
        eligible += status != "excluded_or_unresolved"
        if status == "excluded_or_unresolved":
            classification = "unresolved_ambiguity"
        elif row.get("strict_provenance_status") == "residue_identity_provenance_incomplete" and not any(
            row.get(key)
            for key in (
                "auth_residue_ids_match",
                "label_residue_ids_match",
                "zero_based_positions_match",
                "one_based_positions_match",
            )
        ):
            classification = "residue_identity_provenance_incomplete"
        elif row.get("coordinate_matrix_status") == "unavailable":
            classification = "unique_sequence_pair_coordinate_unavailable"
        else:
            classification = "verified_sequence_geometry_pair"
        classifications[classification] += 1
    candidate_count = sum(1 for _ in candidate.iter_records("residue_id_convention_evidence"))
    contracts = {
        "matrix_pair_count": len(alignments),
        "pairing_eligible_count": eligible,
        "pairing_ineligible_count": len(alignments) - eligible,
        "classification_counts": dict(sorted(classifications.items())),
        "candidate_evidence_row_count": candidate_count,
    }
    expected = {
        "matrix_pair_count": 686,
        "pairing_eligible_count": 675,
        "pairing_ineligible_count": 11,
        "classification_counts": {
            "residue_identity_provenance_incomplete": 4,
            "unique_sequence_pair_coordinate_unavailable": 36,
            "unresolved_ambiguity": 11,
            "verified_sequence_geometry_pair": 635,
        },
        "candidate_evidence_row_count": 3260,
    }
    passed = not discrepancies and contracts == expected
    report = {
        "status": "passed" if passed else "failed",
        "reference_storage_profile": reference.storage_profile,
        "candidate_storage_profile": candidate.storage_profile,
        "table_results": results,
        "discrepant_tables": discrepancies,
        "observed_contracts": contracts,
        "expected_contracts": expected,
        "maximum_representative_mismatches_per_table": max_representative_mismatches,
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return report
