"""Read-only train/validation homology audit for E007 Phase 3E-C."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import coordinate_acceptance_reasons
from protein_distance_diffusion.data.rich_geometry import authorize_rich_geometry_dataset
from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryVocabulary
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file, verify_metadata

AUDIT_VERSION = "e007_split_homology_audit_v1"
PROJECTED_COLUMNS = (
    "sample_id",
    "split",
    "source_path",
    "sequence",
    "token_ids",
    "ca_mask",
    "chain_continuity_mask",
    "chain_break_mask",
)
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "dataset_modified": False,
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
}
MMSEQS_ENVIRONMENT = {
    "LC_ALL": "C",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MMSEQS_NUM_THREADS": "1",
}
_PDB_ID = re.compile(r"(?i)(?<![A-Za-z0-9])([0-9][A-Za-z0-9]{3})(?![A-Za-z0-9])")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != AUDIT_VERSION:
        raise ValueError("E007 Phase-3E-C configuration version contradiction")
    thresholds = payload.get("identity_thresholds")
    if thresholds != [0.9, 0.7, 0.5, 0.3]:
        raise ValueError("E007 Phase-3E-C identity-threshold contract changed")
    if float(payload.get("coverage", 0)) != 0.8 or int(payload.get("coverage_mode", -1)) != 0:
        raise ValueError("E007 Phase-3E-C coverage contract changed")
    if int(payload.get("threads", 0)) != 1:
        raise ValueError("E007 Phase-3E-C deterministic MMseqs2 execution requires one thread")
    if int(payload.get("maximum_examples_per_group", 0)) < 1:
        raise ValueError("E007 Phase-3E-C requires bounded positive examples")
    if int(payload.get("minimum_clean_validation_per_length_stratum", 0)) < 1:
        raise ValueError("E007 Phase-3E-C clean-panel minimum must be positive")
    return payload


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3E-C prerequisite hash contradiction: {label}")
    return observed


def _verify_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    forward = config["phase3e_b"]
    directory = Path(forward["directory"])
    names = {
        "phase3e_b_report": ("report.json", "report_sha256"),
        "phase3e_b_protocol": ("protocol.json", "protocol_sha256"),
        "phase3e_b_manifest": ("selected_panel_manifest.json", "manifest_sha256"),
        "phase3e_b_detailed_metrics": ("detailed_metrics.jsonl.gz", "detailed_metrics_sha256"),
    }
    hashes = {
        label: _verify_hash(directory / filename, forward[key], label) for label, (filename, key) in names.items()
    }
    phase3e_a = config["phase3e_a"]
    normalization_directory = Path(phase3e_a["directory"])
    for label, filename, key in (
        ("phase3e_a_normalization", "normalization.json", "normalization_sha256"),
        ("phase3e_a_report", "report.json", "report_sha256"),
        ("phase3e_a_protocol", "protocol.json", "protocol_sha256"),
        ("phase3e_a_length_statistics", "length_stratified_statistics.json", "length_statistics_sha256"),
        ("phase3e_a_per_sample_audit", "per_sample_audit.jsonl.gz", "per_sample_audit_sha256"),
    ):
        hashes[label] = _verify_hash(normalization_directory / filename, phase3e_a[key], label)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if report.get("status") != "completed_non_authorizing" or protocol.get("status") != "completed_non_authorizing":
        raise ValueError("E007 Phase-3E-B prerequisite is incomplete")
    if report.get("optimizer_updates") != 0 or report.get("dataset_modified") is not False:
        raise ValueError("E007 Phase-3E-B prerequisite contract contradiction")
    normalization = json.loads((normalization_directory / "normalization.json").read_text())
    if normalization.get("version") != phase3e_a["required_version"]:
        raise ValueError("E007 Phase-3E-A normalization version contradiction")
    metadata = verify_metadata(config["dataset"])
    normalization_report = json.loads((normalization_directory / "report.json").read_text())
    panel_counts = report.get("panel_evidence", {}).get("leakage", {}).get("within_split", {})
    population = {
        split: {
            "candidate": int(panel_counts.get(split, {}).get("candidate_count", -1)),
            "accepted": int(panel_counts.get(split, {}).get("accepted_count", -1)),
            "rejected": int(panel_counts.get(split, {}).get("rejected_count", -1)),
        }
        for split in ("train", "validation")
    }
    expected_population = config.get("expected_population_counts")
    if population != expected_population:
        raise ValueError(f"E007 Phase-3E-C pinned accepted-population contradiction: {population}")
    if population["train"]["accepted"] != int(normalization.get("accepted_sample_count", -1)):
        raise ValueError("E007 Phase-3E-A/3E-B accepted training count contradiction")
    statistics = normalization_report.get("statistics", {})
    if any(
        population["train"][key] != int(statistics.get(f"{key}_sample_count", -1))
        for key in ("candidate", "accepted", "rejected")
    ):
        raise ValueError("E007 Phase-3E-A training population counts contradict Phase-3E-B")
    for split in ("train", "validation"):
        if population[split]["candidate"] != int(metadata["eligible_split_counts"][split]):
            raise ValueError(f"E007 Phase-1/Phase-3E-B candidate count contradiction: {split}")
    return {
        "hashes": {**hashes, **{f"dataset_{key}": value for key, value in metadata["hashes"].items()}},
        "dataset": metadata,
        "population_counts": population,
        "phase3e_a_train_accepted_sample_id_sha256": str(statistics["fitted_sample_id_sha256"]),
    }


def detect_mmseqs(executable: str) -> dict[str, Any]:
    path = shutil.which(executable)
    if path is None:
        return {"available": False, "requested_executable": executable, "path": None, "version": None}
    completed = subprocess.run([path, "version"], check=True, capture_output=True, text=True)
    version = completed.stdout.strip() or completed.stderr.strip()
    return {"available": True, "requested_executable": executable, "path": path, "version": version}


def mmseqs_command(
    executable: str,
    fasta: str | Path,
    prefix: str | Path,
    temporary: str | Path,
    *,
    identity: float,
    coverage: float,
    coverage_mode: int,
    sensitivity: float,
    threads: int,
) -> list[str]:
    return [
        executable,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(temporary),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(coverage_mode),
        "--cluster-mode",
        "0",
        "-s",
        str(sensitivity),
        "--threads",
        str(threads),
        "--remove-tmp-files",
        "0",
    ]


def _planned_commands(config: dict[str, Any], executable: str) -> list[dict[str, Any]]:
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    fasta = staging / "combined_sequences.fasta"
    commands = []
    for identity in config["identity_thresholds"]:
        label = f"identity_{round(identity * 100):02d}"
        root = staging / "mmseqs" / label
        commands.append(
            {
                "identity": identity,
                "coverage": config["coverage"],
                "label": label,
                "command": mmseqs_command(
                    executable,
                    fasta,
                    root / "clusters",
                    root / "tmp",
                    identity=identity,
                    coverage=config["coverage"],
                    coverage_mode=config["coverage_mode"],
                    sensitivity=config["sensitivity"],
                    threads=config["threads"],
                ),
                "environment": dict(MMSEQS_ENVIRONMENT),
            }
        )
    return commands


def plan_split_homology_audit(config_path: str | Path) -> dict[str, Any]:
    """Validate metadata and commands without opening scientific shards."""
    path = Path(config_path)
    config = _load_config(path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3E-C output already exists: {output} or {staging}")
    prerequisites = _verify_prerequisites(config)
    mmseqs = detect_mmseqs(config["mmseqs_executable"])
    counts = prerequisites["population_counts"]
    return {
        "status": "planned_non_authorizing",
        "version": AUDIT_VERSION,
        "configuration_sha256": sha256_file(path),
        "output_dir": str(output),
        "projected_columns": list(PROJECTED_COLUMNS),
        "sequence_records_scanned": False,
        "mmseqs_executed": False,
        "mmseqs": mmseqs,
        "clustering_status": "planned" if mmseqs["available"] else "refused_mmseqs2_unavailable",
        "population_counts": counts,
        "planned_sequence_counts": {
            "train": int(counts["train"]["accepted"]),
            "validation": int(counts["validation"]["accepted"]),
            "combined": int(counts["train"]["accepted"]) + int(counts["validation"]["accepted"]),
        },
        "commands": _planned_commands(config, mmseqs["path"] or config["mmseqs_executable"]),
        "prerequisite_hashes": prerequisites["hashes"],
        **NON_AUTHORIZING,
    }


def reconstruct_canonical_sequence(token_ids: Any, vocabulary: SequenceGeometryVocabulary) -> str:
    if not isinstance(token_ids, (list, tuple)):
        try:
            token_ids = token_ids.as_py()
        except AttributeError as error:
            raise ValueError("E007 sequence token IDs are not an array") from error
    values = [int(value) for value in token_ids]
    if not values:
        raise ValueError("E007 sequence token IDs are empty")
    if any(
        value in (vocabulary.pad_id, vocabulary.mask_id) or not 2 <= value < len(vocabulary.tokens) for value in values
    ):
        raise ValueError("E007 sequence token IDs contain PAD, MASK, or noncanonical values")
    return vocabulary.decode(values)


def exact_sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def extract_pdb_id(sample_id: str, source_path: str) -> str:
    source_matches = _PDB_ID.findall(source_path.replace("_", "/").replace("-", "/"))
    sample_matches = _PDB_ID.findall(sample_id.replace("_", "/").replace("-", "/"))
    source = source_matches[-1].lower() if source_matches else None
    sample = sample_matches[0].lower() if sample_matches else None
    if source and sample and source != sample:
        raise ValueError(f"E007 PDB identity contradiction for {sample_id}: {source} != {sample}")
    result = source or sample
    if result is None:
        raise ValueError(f"E007 cannot extract PDB entry identity for {sample_id}")
    return result


def length_stratum(length: int, strata: list[dict[str, Any]]) -> str:
    matches = [str(item["name"]) for item in strata if int(item["minimum"]) <= length <= int(item["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 sequence length has no unique stratum: {length}")
    return matches[0]


def require_population_counts(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    if observed != expected:
        raise ValueError(f"E007 indexed accepted-population contradiction: {observed} != {expected}")


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE sequences(
          sample_id TEXT PRIMARY KEY, split TEXT NOT NULL, pdb_id TEXT NOT NULL,
          length INTEGER NOT NULL, length_stratum TEXT NOT NULL,
          sequence_hash TEXT NOT NULL, sequence TEXT NOT NULL,
          source_path TEXT, dataset_shard_path TEXT, shard_row_index INTEGER
        );
        CREATE TABLE rejections(
          sample_id TEXT PRIMARY KEY, split TEXT NOT NULL, length_stratum TEXT NOT NULL,
          rejection_reasons TEXT NOT NULL, dataset_shard_path TEXT NOT NULL,
          shard_row_index INTEGER NOT NULL
        );
        CREATE INDEX sequences_hash ON sequences(sequence_hash,split);
        CREATE INDEX sequences_pdb ON sequences(pdb_id,split);
        CREATE INDEX sequences_split_stratum ON sequences(split,length_stratum);
        """
    )
    return connection


def _scientific_shards(config: dict[str, Any]) -> Iterator[tuple[str, Path]]:
    root = Path(config["dataset"]["root"])
    protocol = json.loads((root / "protocol.json").read_text())
    for record in protocol["shards"]:
        split = str(record["dataset"])
        if split in {"train", "validation"}:
            yield split, root / str(record["path"])


def _index_sequences(config: dict[str, Any], connection: sqlite3.Connection, fasta: Path) -> dict[str, Any]:
    vocabulary = SequenceGeometryVocabulary()
    candidate_counts = {"train": 0, "validation": 0}
    accepted_counts = {"train": 0, "validation": 0}
    rejected_counts = {"train": 0, "validation": 0}
    rejection_reason_counts: dict[str, dict[str, int]] = {"train": {}, "validation": {}}
    root = Path(config["dataset"]["root"])
    for physical_split, path in _scientific_shards(config):
        parquet = pq.ParquetFile(path)
        missing = sorted(set(PROJECTED_COLUMNS) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"E007 homology shard lacks columns: {missing}")
        shard_row_index = 0
        relative_path = str(path.relative_to(root))
        for batch in parquet.iter_batches(columns=list(PROJECTED_COLUMNS), batch_size=int(config["batch_size"])):
            columns = batch.to_pydict()
            accepted_records = []
            rejected_records = []
            for local_index, values in enumerate(zip(*(columns[name] for name in PROJECTED_COLUMNS), strict=True)):
                sample_id, split, source_path, stored_sequence, token_ids, ca_mask, continuity, breaks = values
                sample_id = str(sample_id)
                split = str(split)
                if split != physical_split:
                    raise ValueError(f"E007 physical/row split contradiction: {sample_id}")
                candidate_counts[split] += 1
                sequence = reconstruct_canonical_sequence(token_ids, vocabulary)
                if sequence != str(stored_sequence):
                    raise ValueError(f"E007 reconstructed/stored sequence contradiction: {sample_id}")
                stratum = length_stratum(len(sequence), config["length_strata"])
                reasons = coordinate_acceptance_reasons(
                    sequence_length=len(sequence),
                    ca_mask=ca_mask,
                    chain_continuity_mask=continuity,
                    chain_break_mask=breaks,
                )
                row_index = shard_row_index + local_index
                if reasons:
                    rejected_counts[split] += 1
                    for reason in reasons:
                        rejection_reason_counts[split][reason] = rejection_reason_counts[split].get(reason, 0) + 1
                    rejected_records.append((sample_id, split, stratum, json.dumps(reasons), relative_path, row_index))
                    continue
                accepted_counts[split] += 1
                pdb_id = extract_pdb_id(sample_id, str(source_path))
                digest = exact_sequence_hash(sequence)
                accepted_records.append(
                    (
                        sample_id,
                        split,
                        pdb_id,
                        len(sequence),
                        stratum,
                        digest,
                        sequence,
                        str(source_path),
                        relative_path,
                        row_index,
                    )
                )
            connection.executemany("INSERT INTO sequences VALUES (?,?,?,?,?,?,?,?,?,?)", accepted_records)
            connection.executemany("INSERT INTO rejections VALUES (?,?,?,?,?,?)", rejected_records)
            connection.commit()
            shard_row_index += batch.num_rows
    sequence_digest = hashlib.sha256()
    with fasta.open("w", encoding="ascii", newline="\n") as handle:
        for sample_id, split, digest, sequence in connection.execute(
            "SELECT sample_id,split,sequence_hash,sequence FROM sequences ORDER BY sample_id"
        ):
            handle.write(f">{sample_id}\n{sequence}\n")
            sequence_digest.update(f"{split}\0{sample_id}\0{digest}\n".encode())
    accepted_id_hashes = {
        split: _canonical_hash(
            [
                row[0]
                for row in connection.execute(
                    "SELECT sample_id FROM sequences WHERE split=? ORDER BY sample_id", (split,)
                )
            ]
        )
        for split in ("train", "validation")
    }
    combined_accepted_ids = [row[0] for row in connection.execute("SELECT sample_id FROM sequences ORDER BY sample_id")]
    rejected_records = [
        {"sample_id": row[0], "split": row[1], "rejection_reasons": json.loads(row[2])}
        for row in connection.execute(
            "SELECT sample_id,split,rejection_reasons FROM rejections ORDER BY split,sample_id"
        )
    ]
    return {
        "population_counts": {
            split: {
                "candidate": candidate_counts[split],
                "accepted": accepted_counts[split],
                "rejected": rejected_counts[split],
            }
            for split in ("train", "validation")
        },
        "rejection_reason_counts": rejection_reason_counts,
        "accepted_sample_id_sha256": accepted_id_hashes,
        "combined_accepted_sample_id_sha256": _canonical_hash(combined_accepted_ids),
        "rejected_sample_id_and_reason_sha256": _canonical_hash(rejected_records),
        "fasta_record_count": sum(accepted_counts.values()),
        "sequence_record_sha256": sequence_digest.hexdigest(),
        "fasta_sha256": sha256_file(fasta),
    }


def _parquet_from_query(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[Any, ...],
    path: Path,
    schema: pa.Schema,
    *,
    batch_size: int,
) -> tuple[int, str]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    cursor = connection.execute(query, parameters)
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    total = 0
    try:
        while rows := cursor.fetchmany(batch_size):
            arrays = list(zip(*rows, strict=True))
            converted = [
                [bool(value) for value in values] if pa.types.is_boolean(field.type) else values
                for values, field in zip(arrays, schema, strict=True)
            ]
            table = pa.Table.from_arrays(
                [pa.array(values, type=field.type) for values, field in zip(converted, schema, strict=True)],
                schema=schema,
            )
            writer.write_table(table)
            total += len(rows)
    finally:
        writer.close()
    temporary.replace(path)
    return total, sha256_file(path)


def _clean_manifest(connection: sqlite3.Connection, key: str, path: Path, batch_size: int) -> dict[str, Any]:
    if key not in {"sequence_hash", "pdb_id"}:
        raise ValueError(f"unsupported exact exclusion key: {key}")
    query = f"""
      SELECT v.sample_id,v.split,1,v.pdb_id,v.length,v.length_stratum,v.sequence_hash,
             v.source_path,v.dataset_shard_path,v.shard_row_index
      FROM sequences v
      WHERE v.split='validation' AND NOT EXISTS(
        SELECT 1 FROM sequences t WHERE t.split='train' AND t.{key}=v.{key}
      ) ORDER BY v.sample_id
    """
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("coordinate_accepted", pa.bool_()),
            ("pdb_id", pa.string()),
            ("length", pa.int32()),
            ("length_stratum", pa.string()),
            ("sequence_hash", pa.string()),
            ("source_path", pa.string()),
            ("dataset_shard_path", pa.string()),
            ("shard_row_index", pa.int64()),
        ]
    )
    count, digest = _parquet_from_query(connection, query, (), path, schema, batch_size=batch_size)
    total = connection.execute("SELECT COUNT(*) FROM sequences WHERE split='validation'").fetchone()[0]
    return {"retained": count, "excluded": total - count, "retained_fraction": count / total, "sha256": digest}


def _group_effect(
    connection: sqlite3.Connection,
    key: str,
    maximum_examples: int,
    strata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if key not in {"sequence_hash", "pdb_id"}:
        raise ValueError(key)
    cross = connection.execute(
        f"""SELECT COUNT(*) FROM (
          SELECT {key} FROM sequences GROUP BY {key}
          HAVING SUM(split='train')>0 AND SUM(split='validation')>0
        )"""
    ).fetchone()[0]
    affected = connection.execute(
        f"""SELECT COUNT(*) FROM sequences v WHERE split='validation' AND EXISTS(
          SELECT 1 FROM sequences t WHERE t.split='train' AND t.{key}=v.{key})"""
    ).fetchone()[0]
    total = connection.execute("SELECT COUNT(*) FROM sequences WHERE split='validation'").fetchone()[0]
    examples = [
        row[0]
        for row in connection.execute(
            f"""SELECT {key} FROM sequences GROUP BY {key}
            HAVING SUM(split='train')>0 AND SUM(split='validation')>0 ORDER BY {key} LIMIT ?""",
            (maximum_examples,),
        )
    ]
    by_length = {}
    for item in strata or []:
        name = str(item["name"])
        stratum_total, stratum_affected = connection.execute(
            f"""SELECT COUNT(*),SUM(EXISTS(
              SELECT 1 FROM sequences t WHERE t.split='train' AND t.{key}=v.{key}
            )) FROM sequences v WHERE v.split='validation' AND v.length_stratum=?""",
            (name,),
        ).fetchone()
        stratum_affected = int(stratum_affected or 0)
        by_length[name] = {
            "validation_samples": int(stratum_total),
            "validation_samples_affected": stratum_affected,
            "validation_fraction_affected": stratum_affected / stratum_total if stratum_total else None,
        }
    return {
        "cross_split_group_count": int(cross),
        "validation_samples_affected": int(affected),
        "validation_fraction_affected": affected / total,
        "bounded_examples": examples,
        "examples_truncated": cross > maximum_examples,
        "by_length_stratum": by_length,
    }


def _run_mmseqs(command: list[str], log: Path) -> None:
    environment = os.environ.copy()
    environment.update(MMSEQS_ENVIRONMENT)
    with log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=environment)


def _load_assignments(connection: sqlite3.Connection, path: Path) -> None:
    connection.execute("DROP TABLE IF EXISTS assignments")
    connection.execute("CREATE TABLE assignments(sample_id TEXT PRIMARY KEY,cluster_id TEXT NOT NULL)")
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            values = line.rstrip("\n").split("\t")
            if len(values) != 2 or not all(values):
                raise ValueError(f"invalid MMseqs2 assignment at line {line_number}")
            try:
                connection.execute("INSERT INTO assignments VALUES (?,?)", (values[1], values[0]))
            except sqlite3.IntegrityError as error:
                raise ValueError(f"duplicate MMseqs2 assignment: {values[1]}") from error
    connection.commit()
    expected = connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0]
    assigned = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    unknown = connection.execute(
        "SELECT COUNT(*) FROM assignments a LEFT JOIN sequences s USING(sample_id) WHERE s.sample_id IS NULL"
    ).fetchone()[0]
    missing = connection.execute(
        "SELECT COUNT(*) FROM sequences s LEFT JOIN assignments a USING(sample_id) WHERE a.sample_id IS NULL"
    ).fetchone()[0]
    if assigned != expected or unknown or missing:
        raise ValueError(
            f"MMseqs2 assignment membership contradiction: expected={expected}, assigned={assigned}, "
            f"unknown={unknown}, missing={missing}"
        )
    connection.execute("CREATE INDEX assignments_cluster ON assignments(cluster_id)")


def _cluster_outputs(
    connection: sqlite3.Connection,
    directory: Path,
    strata: list[dict[str, Any]],
    maximum_examples: int,
    batch_size: int,
    minimum_clean_per_stratum: int,
) -> dict[str, Any]:
    connection.execute("DROP TABLE IF EXISTS cluster_split")
    connection.execute(
        """CREATE TEMP TABLE cluster_split AS
        SELECT a.cluster_id,SUM(s.split='train') train_count,SUM(s.split='validation') validation_count
        FROM assignments a JOIN sequences s USING(sample_id) GROUP BY a.cluster_id"""
    )
    connection.execute("CREATE INDEX cluster_split_id ON cluster_split(cluster_id)")
    counts = connection.execute(
        """SELECT COUNT(*),SUM(train_count>0 AND validation_count=0),
        SUM(train_count=0 AND validation_count>0),SUM(train_count>0 AND validation_count>0)
        FROM cluster_split"""
    ).fetchone()
    total_validation = connection.execute("SELECT COUNT(*) FROM sequences WHERE split='validation'").fetchone()[0]
    affected = connection.execute(
        """SELECT COUNT(*) FROM sequences s JOIN assignments a USING(sample_id)
        JOIN cluster_split c USING(cluster_id) WHERE s.split='validation' AND c.train_count>0"""
    ).fetchone()[0]
    by_length = {}
    for item in strata:
        name = str(item["name"])
        total, leaking = connection.execute(
            """SELECT COUNT(*),SUM(c.train_count>0) FROM sequences s JOIN assignments a USING(sample_id)
            JOIN cluster_split c USING(cluster_id) WHERE s.split='validation' AND s.length_stratum=?""",
            (name,),
        ).fetchone()
        leaking = int(leaking or 0)
        by_length[name] = {
            "validation_samples": int(total),
            "validation_samples_affected": leaking,
            "validation_fraction_affected": leaking / total if total else None,
        }
    examples = [
        {"cluster_id": row[0], "train_count": row[1], "validation_count": row[2]}
        for row in connection.execute(
            "SELECT cluster_id,train_count,validation_count FROM cluster_split "
            "WHERE train_count>0 AND validation_count>0 ORDER BY cluster_id LIMIT ?",
            (maximum_examples,),
        )
    ]
    assignment_schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("cluster_id", pa.string()),
            ("length", pa.int32()),
            ("length_stratum", pa.string()),
            ("sequence_hash", pa.string()),
            ("dataset_shard_path", pa.string()),
            ("shard_row_index", pa.int64()),
        ]
    )
    assignment_count, assignment_hash = _parquet_from_query(
        connection,
        """SELECT s.sample_id,s.split,a.cluster_id,s.length,s.length_stratum,s.sequence_hash,
                   s.dataset_shard_path,s.shard_row_index
        FROM sequences s JOIN assignments a USING(sample_id) ORDER BY s.sample_id""",
        (),
        directory / "cluster_assignments.parquet",
        assignment_schema,
        batch_size=batch_size,
    )
    clean_schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("split", pa.string()),
            ("coordinate_accepted", pa.bool_()),
            ("pdb_id", pa.string()),
            ("length", pa.int32()),
            ("length_stratum", pa.string()),
            ("sequence_hash", pa.string()),
            ("cluster_id", pa.string()),
            ("source_path", pa.string()),
            ("dataset_shard_path", pa.string()),
            ("shard_row_index", pa.int64()),
        ]
    )
    clean_count, clean_hash = _parquet_from_query(
        connection,
        """SELECT s.sample_id,s.split,1,s.pdb_id,s.length,s.length_stratum,s.sequence_hash,a.cluster_id,
                   s.source_path,s.dataset_shard_path,s.shard_row_index
        FROM sequences s JOIN assignments a USING(sample_id) JOIN cluster_split c USING(cluster_id)
        WHERE s.split='validation' AND c.train_count=0 ORDER BY s.sample_id""",
        (),
        directory / "clean_validation_manifest.parquet",
        clean_schema,
        batch_size=batch_size,
    )
    clean_by_length = {
        str(item["name"]): connection.execute(
            """SELECT COUNT(*) FROM sequences s JOIN assignments a USING(sample_id)
            JOIN cluster_split c USING(cluster_id)
            WHERE s.split='validation' AND c.train_count=0 AND s.length_stratum=?""",
            (str(item["name"]),),
        ).fetchone()[0]
        for item in strata
    }
    shortages = {
        key: {"available": int(value), "required_minimum": minimum_clean_per_stratum}
        for key, value in clean_by_length.items()
        if value < minimum_clean_per_stratum
    }
    return {
        "cluster_count": int(counts[0]),
        "train_only_clusters": int(counts[1] or 0),
        "validation_only_clusters": int(counts[2] or 0),
        "cross_split_clusters": int(counts[3] or 0),
        "validation_samples_in_cross_split_clusters": int(affected),
        "validation_fraction_affected": affected / total_validation,
        "by_length_stratum": by_length,
        "bounded_cross_split_examples": examples,
        "examples_truncated": int(counts[3] or 0) > maximum_examples,
        "assignment_count": assignment_count,
        "assignment_sha256": assignment_hash,
        "clean_validation_count": clean_count,
        "clean_validation_sha256": clean_hash,
        "clean_validation_by_length_stratum": {key: int(value) for key, value in clean_by_length.items()},
        "empty_clean_length_strata": sorted(key for key, value in clean_by_length.items() if not value),
        "clean_validation_length_stratum_shortages": shortages,
        "clean_validation_minimum_per_length_stratum": minimum_clean_per_stratum,
    }


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"heartbeat.json", "protocol.json", "report.json", "audit.sqlite"}
    }


def _protocol_payload(report: dict[str, Any], report_sha256: str) -> dict[str, Any]:
    """Build a compact execution contract without duplicating the scientific report."""
    return {
        "status": report["status"],
        "version": report["version"],
        "configuration_sha256": report["configuration_sha256"],
        "report_sha256": report_sha256,
        "accepted_population_policy": "contiguous_single_chain_complete_calpha_v1",
        "population_counts": report["indexed_sequences"]["population_counts"],
        "accepted_sample_id_sha256": report["indexed_sequences"]["accepted_sample_id_sha256"],
        "rejected_sample_id_and_reason_sha256": report["indexed_sequences"]["rejected_sample_id_and_reason_sha256"],
        "fasta_sha256": report["indexed_sequences"]["fasta_sha256"],
        "mmseqs": report["mmseqs"],
        "mmseqs_environment": report["mmseqs_environment"],
        "protected_input_hashes": report["protected_input_hashes"],
        "protected_scientific_shards_unchanged": report["protected_scientific_shards_unchanged"],
        "artifact_hashes": report["artifact_hashes"],
        **NON_AUTHORIZING,
    }


def audit_split_homology(config_path: str | Path) -> dict[str, Any]:
    """Execute the immutable, sequence-only split audit."""
    started = time.monotonic()
    started_utc = _utc_now()
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3E-C output already exists: {output} or {staging}")
    prerequisites = _verify_prerequisites(config)
    mmseqs = detect_mmseqs(config["mmseqs_executable"])
    if not mmseqs["available"]:
        raise RuntimeError("E007 Phase-3E-C clustering refused: MMseqs2 is unavailable")
    authorization = authorize_rich_geometry_dataset(
        config["dataset"]["root"],
        expected_protocol_sha256=config["dataset"]["protocol_sha256"],
        expected_schema_sha256=config["dataset"]["schema_sha256"],
        expected_vocabulary_sha256=config["dataset"]["vocabulary_sha256"],
        expected_normalization_sha256=config["dataset"]["normalization_sha256"],
        expected_shard_inventory_sha256=config["dataset"]["shard_inventory_sha256"],
    )
    staging.mkdir(parents=True)
    heartbeat_path = staging / "heartbeat.json"

    def heartbeat(stage: str, status: str = "running", **extra: Any) -> None:
        _atomic_json(
            heartbeat_path,
            {
                "version": AUDIT_VERSION,
                "status": status,
                "stage": stage,
                "updated_utc": _utc_now(),
                **extra,
                **NON_AUTHORIZING,
            },
        )

    connection: sqlite3.Connection | None = None
    try:
        heartbeat("sequence_indexing")
        connection = _database(staging / "audit.sqlite")
        indexed = _index_sequences(config, connection, staging / "combined_sequences.fasta")
        expected = prerequisites["population_counts"]
        require_population_counts(indexed["population_counts"], expected)
        expected_fasta_records = sum(int(expected[split]["accepted"]) for split in ("train", "validation"))
        if indexed["fasta_record_count"] != expected_fasta_records:
            raise ValueError(
                f"E007 accepted FASTA count contradiction: {indexed['fasta_record_count']} != {expected_fasta_records}"
            )
        if indexed["accepted_sample_id_sha256"]["train"] != prerequisites["phase3e_a_train_accepted_sample_id_sha256"]:
            raise ValueError("E007 accepted training sample-ID hash contradicts Phase-3E-A")
        heartbeat("exact_sequence_and_pdb_analysis", population_counts=indexed["population_counts"])
        exact = _group_effect(
            connection,
            "sequence_hash",
            int(config["maximum_examples_per_group"]),
            config["length_strata"],
        )
        pdb = _group_effect(
            connection,
            "pdb_id",
            int(config["maximum_examples_per_group"]),
            config["length_strata"],
        )
        exact["clean_manifest"] = _clean_manifest(
            connection,
            "sequence_hash",
            staging / "clean_validation_exact_sequence.parquet",
            int(config["batch_size"]),
        )
        pdb["clean_manifest"] = _clean_manifest(
            connection,
            "pdb_id",
            staging / "clean_validation_same_pdb.parquet",
            int(config["batch_size"]),
        )
        threshold_results = {}
        commands = _planned_commands(config, mmseqs["path"])
        for position, command_record in enumerate(commands, start=1):
            label = command_record["label"]
            heartbeat("mmseqs_clustering", threshold=label, threshold_index=position)
            directory = staging / "mmseqs" / label
            directory.mkdir(parents=True)
            command = command_record["command"]
            _run_mmseqs(command, directory / "mmseqs.log")
            assignments = Path(f"{directory / 'clusters'}_cluster.tsv")
            if not assignments.is_file():
                raise FileNotFoundError(f"MMseqs2 did not publish assignments: {assignments}")
            _load_assignments(connection, assignments)
            result = _cluster_outputs(
                connection,
                directory,
                config["length_strata"],
                int(config["maximum_examples_per_group"]),
                int(config["batch_size"]),
                int(config["minimum_clean_validation_per_length_stratum"]),
            )
            result.update(command_record)
            result["raw_assignment_sha256"] = sha256_file(assignments)
            threshold_results[label] = result
        heartbeat("protected_input_verification")
        after = authorize_rich_geometry_dataset(
            config["dataset"]["root"],
            expected_protocol_sha256=config["dataset"]["protocol_sha256"],
            expected_schema_sha256=config["dataset"]["schema_sha256"],
            expected_vocabulary_sha256=config["dataset"]["vocabulary_sha256"],
            expected_normalization_sha256=config["dataset"]["normalization_sha256"],
            expected_shard_inventory_sha256=config["dataset"]["shard_inventory_sha256"],
        )
        if authorization.observed_shard_hashes != after.observed_shard_hashes:
            raise ValueError("E007 protected scientific shard hashes changed during audit")
        if connection is not None:
            connection.close()
            connection = None
        (staging / "audit.sqlite").unlink()
        elapsed = time.monotonic() - started
        payload = {
            "status": "completed_non_authorizing",
            "version": AUDIT_VERSION,
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "configuration_sha256": sha256_file(config_path),
            "projected_columns": list(PROJECTED_COLUMNS),
            "sequence_use": "split_audit_only_never_model_features",
            "indexed_sequences": indexed,
            "exact_sequence": exact,
            "same_pdb_entry": pdb,
            "sequence_clustering": threshold_results,
            "mmseqs": mmseqs,
            "mmseqs_environment": dict(MMSEQS_ENVIRONMENT),
            "protected_input_hashes": prerequisites["hashes"],
            "protected_scientific_shards_unchanged": True,
            **NON_AUTHORIZING,
        }
        payload["artifact_hashes"] = _artifact_hashes(staging)
        _atomic_json(staging / "report.json", payload)
        protocol = _protocol_payload(payload, sha256_file(staging / "report.json"))
        _atomic_json(staging / "protocol.json", protocol)
        if sha256_file(staging / "report.json") == sha256_file(staging / "protocol.json"):
            raise ValueError("E007 Phase-3E-C report/protocol publication collision")
        heartbeat("publication", status="completed", report_sha256=sha256_file(staging / "report.json"))
        staging.replace(output)
        return {**payload, "output_dir": str(output)}
    except BaseException as error:
        if connection is not None:
            connection.close()
        heartbeat("failed", status="failed", error_type=type(error).__name__, error_message=str(error)[:1000])
        raise
