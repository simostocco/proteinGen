"""Publication-only correction for the completed E007 Phase 3E-C audit."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from protein_distance_diffusion.evaluation.e007_split_homology import (
    MMSEQS_ENVIRONMENT,
    NON_AUTHORIZING,
    _verify_prerequisites,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file

PUBLICATION_VERSION = "e007_split_homology_publication_correction_v1"
THRESHOLDS = ("identity_90", "identity_70", "identity_50", "identity_30")
EXCLUDED_INVENTORY_NAMES = {"report.json", "protocol.json", "heartbeat.json", "artifact_inventory.json"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != PUBLICATION_VERSION:
        raise ValueError("E007 Phase-3E-C.1 configuration version contradiction")
    if payload.get("thresholds") != list(THRESHOLDS):
        raise ValueError("E007 Phase-3E-C.1 threshold contract changed")
    return payload


def _source(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    source = Path(config["source_output_dir"])
    report = source / "report.json"
    protocol = source / "protocol.json"
    expected = config["source_publication"]
    report_hash = sha256_file(report)
    protocol_hash = sha256_file(protocol)
    if report_hash != expected["report_sha256"] or protocol_hash != expected["protocol_sha256"]:
        raise ValueError("E007 Phase-3E-C source publication hash contradiction")
    if report.read_bytes() != protocol.read_bytes():
        raise ValueError("E007 Phase-3E-C source no longer exhibits the pinned publication defect")
    payload = json.loads(report.read_text())
    if payload.get("status") != "completed_non_authorizing":
        raise ValueError("E007 Phase-3E-C source calculation is incomplete")
    if payload.get("configuration_sha256") != expected["configuration_sha256"]:
        raise ValueError("E007 Phase-3E-C source configuration hash contradiction")
    return source, payload


def _durable_paths() -> tuple[str, ...]:
    paths = ["clean_validation_exact_sequence.parquet", "clean_validation_same_pdb.parquet"]
    for threshold in THRESHOLDS:
        paths.extend(
            (
                f"mmseqs/{threshold}/cluster_assignments.parquet",
                f"mmseqs/{threshold}/clean_validation_manifest.parquet",
            )
        )
    return tuple(paths)


def _verify_source_artifacts(source: Path, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    recorded = payload.get("artifact_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("E007 Phase-3E-C source artifact hashes are unavailable")
    inventory = {}
    for relative in _durable_paths():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"E007 Phase-3E-C durable artifact is missing: {relative}")
        digest = sha256_file(path)
        if recorded.get(relative) != digest:
            raise ValueError(f"E007 Phase-3E-C durable artifact hash contradiction: {relative}")
        inventory[relative] = {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
    return inventory


def _manifest_ids(path: Path, *, expected_count: int) -> tuple[set[str], dict[str, int]]:
    parquet = pq.ParquetFile(path)
    required = {"sample_id", "split", "coordinate_accepted", "length_stratum"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"E007 clean manifest lacks required columns: {missing}")
    identifiers: set[str] = set()
    by_length: dict[str, int] = {}
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=4096):
        values = batch.to_pydict()
        for sample_id, split, accepted, stratum in zip(
            values["sample_id"],
            values["split"],
            values["coordinate_accepted"],
            values["length_stratum"],
            strict=True,
        ):
            if str(split) != "validation" or accepted is not True:
                raise ValueError(f"E007 clean manifest contains a non-accepted validation row: {sample_id}")
            sample_id = str(sample_id)
            if sample_id in identifiers:
                raise ValueError(f"E007 clean manifest contains duplicate sample ID: {sample_id}")
            identifiers.add(sample_id)
            key = str(stratum)
            by_length[key] = by_length.get(key, 0) + 1
    if len(identifiers) != expected_count or parquet.metadata.num_rows != expected_count:
        raise ValueError(
            f"E007 clean manifest count contradiction: ids={len(identifiers)}, "
            f"rows={parquet.metadata.num_rows}, expected={expected_count}"
        )
    return identifiers, dict(sorted(by_length.items()))


def _reproduce_threshold(
    source: Path,
    threshold: str,
    expected: dict[str, Any],
    source_summary: dict[str, Any],
    accepted_population: dict[str, int],
) -> dict[str, Any]:
    assignments = source / "mmseqs" / threshold / "cluster_assignments.parquet"
    clean = source / "mmseqs" / threshold / "clean_validation_manifest.parquet"
    with tempfile.TemporaryDirectory(prefix=f"e007-{threshold}-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "assignments.sqlite")
        connection.executescript(
            """
            CREATE TABLE assignments(
              sample_id TEXT PRIMARY KEY, split TEXT NOT NULL, cluster_id TEXT NOT NULL,
              length_stratum TEXT NOT NULL
            );
            CREATE INDEX assignments_cluster ON assignments(cluster_id,split);
            """
        )
        parquet = pq.ParquetFile(assignments)
        required = {"sample_id", "split", "cluster_id", "length_stratum"}
        missing = sorted(required - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"E007 assignment table lacks required columns: {missing}")
        split_counts = {"train": 0, "validation": 0}
        for batch in parquet.iter_batches(columns=sorted(required), batch_size=4096):
            values = batch.to_pydict()
            rows = []
            for sample_id, split, cluster_id, stratum in zip(
                values["sample_id"],
                values["split"],
                values["cluster_id"],
                values["length_stratum"],
                strict=True,
            ):
                split = str(split)
                if split not in split_counts:
                    raise ValueError(f"E007 assignment has invalid split: {split}")
                split_counts[split] += 1
                rows.append((str(sample_id), split, str(cluster_id), str(stratum)))
            try:
                connection.executemany("INSERT INTO assignments VALUES (?,?,?,?)", rows)
            except sqlite3.IntegrityError as error:
                raise ValueError("E007 assignment contains duplicate sample IDs") from error
        connection.commit()
        expected_total = sum(accepted_population.values())
        if split_counts != accepted_population or parquet.metadata.num_rows != expected_total:
            raise ValueError(f"E007 {threshold} assignment population contradiction: {split_counts}")
        connection.execute(
            """CREATE TEMP TABLE clusters AS
            SELECT cluster_id,SUM(split='train') train_count,SUM(split='validation') validation_count
            FROM assignments GROUP BY cluster_id"""
        )
        counts = connection.execute(
            """SELECT COUNT(*),SUM(train_count>0 AND validation_count=0),
            SUM(train_count=0 AND validation_count>0),SUM(train_count>0 AND validation_count>0)
            FROM clusters"""
        ).fetchone()
        affected = connection.execute(
            """SELECT COUNT(*) FROM assignments a JOIN clusters c USING(cluster_id)
            WHERE a.split='validation' AND c.train_count>0"""
        ).fetchone()[0]
        clean_ids = {
            row[0]
            for row in connection.execute(
                """SELECT sample_id FROM assignments a JOIN clusters c USING(cluster_id)
                WHERE a.split='validation' AND c.train_count=0"""
            )
        }
        clean_file_ids, clean_by_length = _manifest_ids(clean, expected_count=len(clean_ids))
        if clean_ids != clean_file_ids:
            raise ValueError(f"E007 {threshold} clean manifest contradicts assignment recomputation")
        affected_by_length = {
            row[0]: {"validation_samples": int(row[1]), "affected": int(row[2] or 0)}
            for row in connection.execute(
                """SELECT a.length_stratum,COUNT(*),SUM(c.train_count>0)
                FROM assignments a JOIN clusters c USING(cluster_id)
                WHERE a.split='validation' GROUP BY a.length_stratum ORDER BY a.length_stratum"""
            )
        }
        connection.close()
    reproduced = {
        "assignment_count": sum(accepted_population.values()),
        "cluster_count": int(counts[0]),
        "train_only_clusters": int(counts[1] or 0),
        "validation_only_clusters": int(counts[2] or 0),
        "cross_split_clusters": int(counts[3] or 0),
        "affected_validation_samples": int(affected),
        "affected_validation_fraction": affected / accepted_population["validation"],
        "clean_validation_count": len(clean_ids),
        "clean_validation_by_length_stratum": clean_by_length,
        "affected_by_length_stratum": affected_by_length,
        "assignment_sha256": sha256_file(assignments),
        "clean_validation_sha256": sha256_file(clean),
    }
    gates = {
        "cross_split_clusters": reproduced["cross_split_clusters"],
        "affected_validation_samples": reproduced["affected_validation_samples"],
        "clean_validation_count": reproduced["clean_validation_count"],
    }
    if gates != expected:
        raise ValueError(f"E007 {threshold} expected scientific summary contradiction: {gates} != {expected}")
    source_gates = {
        "cross_split_clusters": int(source_summary["cross_split_clusters"]),
        "affected_validation_samples": int(source_summary["validation_samples_in_cross_split_clusters"]),
        "clean_validation_count": int(source_summary["clean_validation_count"]),
    }
    if gates != source_gates:
        raise ValueError(f"E007 {threshold} source-summary reproduction failed")
    if reproduced["assignment_sha256"] != source_summary["assignment_sha256"]:
        raise ValueError(f"E007 {threshold} assignment hash contradicts source summary")
    if reproduced["clean_validation_sha256"] != source_summary["clean_validation_sha256"]:
        raise ValueError(f"E007 {threshold} clean-manifest hash contradicts source summary")
    return reproduced


def _verify_exact_result(
    source: Path,
    payload: dict[str, Any],
    key: str,
    relative: str,
    expected_hash: str,
    expected_validation_count: int,
) -> dict[str, Any]:
    result = payload[key]
    if (
        int(result["validation_samples_affected"]) != 0
        or int(result["clean_manifest"]["retained"]) != expected_validation_count
    ):
        raise ValueError(f"E007 {key} source-result contradiction")
    path = source / relative
    identifiers, by_length = _manifest_ids(path, expected_count=expected_validation_count)
    digest = sha256_file(path)
    if digest != expected_hash or digest != result["clean_manifest"]["sha256"]:
        raise ValueError(f"E007 {key} clean-manifest hash contradiction")
    return {
        "affected_validation_samples": 0,
        "affected_validation_fraction": 0.0,
        "clean_validation_count": len(identifiers),
        "clean_validation_by_length_stratum": by_length,
        "clean_validation_sha256": digest,
    }


def _copy_durable(source: Path, staging: Path, inventory: dict[str, dict[str, Any]]) -> None:
    for relative, record in inventory.items():
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(source / relative, temporary)
        if sha256_file(temporary) != record["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"E007 copied durable artifact hash contradiction: {relative}")
        temporary.replace(destination)


def _publication_inventory(staging: Path) -> dict[str, Any]:
    records = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED_INVENTORY_NAMES:
            continue
        relative = path.relative_to(staging)
        if "tmp" in relative.parts or path.name.startswith("."):
            continue
        records.append(
            {
                "path": str(relative),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "version": PUBLICATION_VERSION,
        "artifact_count": len(records),
        "artifacts": records,
        "aggregate_inventory_sha256": _canonical_hash(records),
        "exclusions": {
            "self_referential_publication_files": sorted(EXCLUDED_INVENTORY_NAMES),
            "staging_paths": True,
            "transient_mmseqs_tmp_database_files": True,
        },
    }


def _verify_all(config: dict[str, Any]) -> dict[str, Any]:
    source, source_payload = _source(config)
    source_inventory = _verify_source_artifacts(source, source_payload)
    prerequisite_config_path = Path(config["source_audit_config_path"])
    if sha256_file(prerequisite_config_path) != config["source_audit_config_sha256"]:
        raise ValueError("E007 Phase-3E-C source audit configuration hash contradiction")
    source_config = yaml.safe_load(prerequisite_config_path.read_text())
    prerequisite = _verify_prerequisites(source_config)
    accepted_record = config["expected_population_counts"]["accepted"]
    accepted_population = {
        "train": int(accepted_record["train"]),
        "validation": int(accepted_record["validation"]),
    }
    if int(accepted_record["combined"]) != sum(accepted_population.values()):
        raise ValueError("E007 Phase-3E-C.1 accepted population total contradiction")
    exact = _verify_exact_result(
        source,
        source_payload,
        "exact_sequence",
        "clean_validation_exact_sequence.parquet",
        config["expected_exact_clean_sha256"],
        int(accepted_population["validation"]),
    )
    same_pdb = _verify_exact_result(
        source,
        source_payload,
        "same_pdb_entry",
        "clean_validation_same_pdb.parquet",
        config["expected_same_pdb_clean_sha256"],
        int(accepted_population["validation"]),
    )
    thresholds = {
        threshold: _reproduce_threshold(
            source,
            threshold,
            config["expected_threshold_summaries"][threshold],
            source_payload["sequence_clustering"][threshold],
            accepted_population,
        )
        for threshold in THRESHOLDS
    }
    return {
        "source": source,
        "source_payload": source_payload,
        "source_inventory": source_inventory,
        "prerequisite": prerequisite,
        "exact_sequence": exact,
        "same_pdb_entry": same_pdb,
        "thresholds": thresholds,
    }


def plan_publication_correction(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = _load_config(path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3E-C.1 output already exists: {output} or {staging}")
    source, payload = _source(config)
    source_inventory = _verify_source_artifacts(source, payload)
    return {
        "status": "planned_publication_correction_non_authorizing",
        "version": PUBLICATION_VERSION,
        "configuration_sha256": sha256_file(path),
        "source_output_dir": str(source),
        "output_dir": str(output),
        "source_publication_defect": "report_and_protocol_byte_identical",
        "source_durable_artifact_count": len(source_inventory),
        "mmseqs_executed": False,
        "scientific_artifacts_scanned": False,
        **NON_AUTHORIZING,
    }


def publish_correction(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = _load_config(path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3E-C.1 output already exists: {output} or {staging}")
    verified = _verify_all(config)
    staging.mkdir(parents=True)
    try:
        _atomic_json(
            staging / "heartbeat.json",
            {"status": "publishing", "version": PUBLICATION_VERSION, "updated_utc": _utc_now(), **NON_AUTHORIZING},
        )
        _copy_durable(verified["source"], staging, verified["source_inventory"])
        inventory = _publication_inventory(staging)
        _atomic_json(staging / "artifact_inventory.json", inventory)
        source_payload = verified["source_payload"]
        report = {
            "status": "completed_non_authorizing",
            "version": PUBLICATION_VERSION,
            "publication_correction_only": True,
            "scientific_calculation_reused": True,
            "mmseqs_rerun": False,
            "accepted_population_counts": config["expected_population_counts"]["accepted"],
            "rejected_population_counts": config["expected_population_counts"]["rejected"],
            "exact_sequence": verified["exact_sequence"],
            "same_pdb_entry": verified["same_pdb_entry"],
            "homology_thresholds": verified["thresholds"],
            "recommendation": {
                "model_selection_and_evaluation_candidate": "identity_30_clean_validation_manifest",
                "manifest_path": "mmseqs/identity_30/clean_validation_manifest.parquet",
                "rationale": "most conservative prospective 30%-identity, 80%-coverage clean panel",
                "authorizes_training": False,
            },
            "protected_input_verification": {
                "verified": True,
                "prerequisite_hashes": verified["prerequisite"]["hashes"],
                "source_protected_scientific_shards_unchanged": bool(
                    source_payload["protected_scientific_shards_unchanged"]
                ),
            },
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            **NON_AUTHORIZING,
        }
        protocol = {
            "status": "completed_non_authorizing",
            "version": PUBLICATION_VERSION,
            "publication_mode": "read_only_republication_from_verified_phase3e_c_artifacts",
            "configuration_sha256": sha256_file(path),
            "source_phase3e_c": {
                "path": str(verified["source"]),
                "report_sha256": config["source_publication"]["report_sha256"],
                "protocol_sha256": config["source_publication"]["protocol_sha256"],
                "configuration_sha256": config["source_publication"]["configuration_sha256"],
                "publication_defect": "report_and_protocol_byte_identical",
                "durable_artifact_hashes": {
                    key: value["sha256"] for key, value in verified["source_inventory"].items()
                },
            },
            "accepted_population_policy": "contiguous_single_chain_complete_calpha_v1",
            "accepted_population_counts": config["expected_population_counts"]["accepted"],
            "mmseqs": {
                "path": source_payload["mmseqs"]["path"],
                "version": source_payload["mmseqs"]["version"],
                "thresholds": [0.9, 0.7, 0.5, 0.3],
                "coverage": 0.8,
                "coverage_mode": 0,
                "cluster_mode": 0,
                "sensitivity": 7.5,
                "threads": 1,
                "environment": dict(MMSEQS_ENVIRONMENT),
                "rerun": False,
            },
            "prerequisite_hashes": verified["prerequisite"]["hashes"],
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        _atomic_json(staging / "protocol.json", protocol)
        report_hash = sha256_file(staging / "report.json")
        protocol_hash = sha256_file(staging / "protocol.json")
        identical_bytes = (staging / "report.json").read_bytes() == (staging / "protocol.json").read_bytes()
        if report_hash == protocol_hash or identical_bytes:
            raise ValueError("E007 corrected report and protocol are not distinct")
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "version": PUBLICATION_VERSION,
                "completed_utc": _utc_now(),
                "report_sha256": report_hash,
                "protocol_sha256": protocol_hash,
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return {
            "status": "completed_non_authorizing",
            "output_dir": str(output),
            "report_sha256": report_hash,
            "protocol_sha256": protocol_hash,
            "artifact_inventory_sha256": sha256_file(output / "artifact_inventory.json"),
            **NON_AUTHORIZING,
        }
    except BaseException:
        if staging.exists():
            _atomic_json(
                staging / "heartbeat.json",
                {"status": "failed", "version": PUBLICATION_VERSION, "updated_utc": _utc_now(), **NON_AUTHORIZING},
            )
        raise
