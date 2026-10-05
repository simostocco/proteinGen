#!/usr/bin/env python
"""Diagnose bounded raw-pilot sequence/matrix provenance failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from protein_distance_diffusion.evaluation.provenance_forensics import (
    CLASSIFICATIONS,
    classify_case,
    inspect_npz_forensics,
    score_candidate,
)
from protein_distance_diffusion.evaluation.sequence_readiness import inspect_mmcif, sha256_file

DEFAULT_AUDIT_DIR = Path("reports/sequence_data_readiness_raw_pilot_250_v2")
DEFAULT_OUTPUT_DIR = Path("reports/sequence_data_readiness_raw_pilot_250_v2_forensics")
DEFAULT_STATE_DBS = (
    Path("data/full/processed/preprocess_state.sqlite"),
    Path("data/full/processed_recovery/preprocess_state.sqlite"),
)
EXPECTED_FAILURE_COUNT = 197
REPRESENTATIVE_IDS = {"1ej7_l", "3jbt_b", "4cdy_a", "4oig_d", "6k43_f", "6n8o_g"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp{path.suffix}")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_controls(path: Path, per_method: int) -> list[dict[str, Any]]:
    candidates = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("coordinate_verification_status") != "passed":
                continue
            if row.get("alignment_class") not in {"exact", "valid_terminal_trim", "valid_residue_id_selection"}:
                continue
            candidates.append(row)
    selected = []
    methods: Counter[str] = Counter()
    for row in sorted(candidates, key=lambda item: hashlib.sha256(str(item["sample_id"]).encode()).hexdigest()):
        method = str(row.get("experimental_method", "unknown"))
        if methods[method] < per_method:
            selected.append(row)
            methods[method] += 1
    return selected


def _npz_source_and_metadata(path: Path) -> tuple[str, dict[str, Any]]:
    inspected = inspect_npz_forensics(path)
    source = inspected["metadata"].get("source_file")
    if not source:
        raise ValueError(f"NPZ does not record metadata.source_file: {path}")
    return str(source), inspected["metadata"]


def _preflight(
    audit_dir: Path,
    output_dir: Path,
    state_dbs: tuple[Path, ...],
    *,
    resume: bool,
    expected_failure_count: int,
    controls_per_method: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    failures_path = audit_dir / "raw_blocking_failures.csv"
    alignments_path = audit_dir / "seqres_atom_matrix_alignments.jsonl"
    protocol_path = audit_dir / "sequence_readiness_protocol.json"
    run_config_path = audit_dir / "run_config.json"
    required = (failures_path, alignments_path, protocol_path, run_config_path, *state_dbs)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing forensic input(s): {', '.join(missing)}")
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Forensic output already exists; use --resume: {output_dir}")
    if resume and not (output_dir / "forensic_state.sqlite").is_file():
        raise FileNotFoundError("--resume requires an existing forensic_state.sqlite")

    with failures_path.open(newline="") as handle:
        failures = list(csv.DictReader(handle))
    if len(failures) != expected_failure_count:
        raise ValueError(f"Expected {expected_failure_count} blocking rows, found {len(failures)}")
    if len({row["sample_id"] for row in failures}) != len(failures):
        raise ValueError("Blocking input contains duplicate sample_id values")
    controls = _select_controls(alignments_path, controls_per_method)
    consumed = [failures_path, alignments_path, protocol_path, run_config_path, *state_dbs]
    for row in [*failures, *controls]:
        matrix_path = Path(row["matrix_path"])
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Missing processed NPZ: {matrix_path}")
        source, _metadata = _npz_source_and_metadata(matrix_path)
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing raw source: {source_path}")
        row["source_file"] = source
        consumed.extend((matrix_path, source_path))
    unique_inputs = sorted(set(consumed), key=str)
    return failures, controls, {str(path): sha256_file(path) for path in unique_inputs}


def _connect_output_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS cases (
            sample_id TEXT PRIMARY KEY,
            is_control INTEGER NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT,
            updated_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            sample_id TEXT NOT NULL,
            candidate_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(sample_id, candidate_index)
        );
        """
    )
    return connection


def _state_evidence(source: Path, sample_id: str, state_dbs: tuple[Path, ...]) -> dict[str, Any]:
    current = source.stat()
    matches = []
    for database in state_dbs:
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
            schema = {row[1] for row in connection.execute("PRAGMA table_info(source_files)")}
            if not {"source_path", "source_size", "source_mtime_ns", "config_hash"} <= schema:
                raise ValueError(f"Unexpected preprocessing-state schema: {database}")
            source_row = connection.execute(
                """SELECT source_size,source_mtime_ns,status,config_hash,attempt_count
                FROM source_files WHERE source_path=?""",
                (str(source),),
            ).fetchone()
            manifest_row = connection.execute(
                "SELECT row_json,config_hash FROM manifest_rows WHERE sample_id=?", (sample_id,)
            ).fetchone()
        if source_row or manifest_row:
            matches.append(
                {
                    "database": str(database),
                    "source_size": source_row[0] if source_row else None,
                    "source_mtime_ns": source_row[1] if source_row else None,
                    "source_status": source_row[2] if source_row else None,
                    "source_config_hash": source_row[3] if source_row else None,
                    "attempt_count": source_row[4] if source_row else None,
                    "sample_manifest_present": manifest_row is not None,
                    "sample_config_hash": manifest_row[1] if manifest_row else None,
                    "size_matches_current": bool(source_row and int(source_row[0]) == current.st_size),
                    "mtime_matches_current": bool(source_row and int(source_row[1]) == current.st_mtime_ns),
                    "historical_source_sha256_available": False,
                }
            )
    relevant = [row for row in matches if row["sample_manifest_present"]] or matches
    if any(not row["size_matches_current"] or not row["mtime_matches_current"] for row in relevant):
        status = "state_identity_changed"
    elif relevant:
        status = "state_size_mtime_match_sha_unavailable"
    else:
        status = "no_preprocessing_state_record"
    return {
        "status": status,
        "records": matches,
        "current_size": current.st_size,
        "current_mtime_ns": current.st_mtime_ns,
    }


def _state_database_provenance(state_dbs: tuple[Path, ...]) -> dict[str, Any]:
    result = {}
    for database in state_dbs:
        with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            result[str(database)] = {
                "tables": {
                    table: [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
                    for table in sorted(tables)
                },
                "indexes": [
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL ORDER BY name"
                    )
                ],
                "run_metadata": dict(connection.execute("SELECT key,value FROM run_metadata")),
                "historical_source_sha256_stored": False,
            }
    return result


def _serialization_tolerance(failures: list[dict[str, Any]], controls: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for row in [*failures, *controls]:
        if row.get("coordinate_verification_status") == "passed":
            try:
                errors.append(float(row.get("coordinate_distance_max_abs_error_angstrom") or 0.0))
            except ValueError:
                continue
    maximum = max(errors, default=0.0)
    return {
        "known_good_sample_count": len(errors),
        "known_good_max_abs_error_angstrom": maximum,
        "selected_tolerance_angstrom": max(1e-4, maximum * 10.0),
        "rule": "max(1e-4 Angstrom, 10 * maximum observed known-good serialization error)",
    }


def _process_case(
    row: dict[str, Any],
    *,
    raw_rows: list[dict[str, Any]],
    state_dbs: tuple[Path, ...],
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    npz = inspect_npz_forensics(row["matrix_path"])
    metadata = npz["metadata"]
    model_number = str(metadata.get("model_number", row.get("model_id", "1")))
    candidates = [
        raw
        for raw in raw_rows
        if str(raw.get("model_id")) == model_number
        and (
            str(raw.get("auth_asym_id")) == str(npz["chain_id"])
            or str(raw.get("label_asym_id")) == str(npz["chain_id"])
        )
    ]
    scores = []
    for raw in candidates:
        scores.extend(
            score_candidate(
                raw,
                matrix_sequence=npz["sequence"],
                matrix_ids=npz["residue_ids"],
                matrix=npz["distance_matrix"],
                metadata=metadata,
                requested_chain=npz["chain_id"],
                tolerance=tolerance,
            )
        )
    state = _state_evidence(Path(row["source_file"]), row["sample_id"], state_dbs)
    primary, secondary, selected, selection_reason = classify_case(
        original=row,
        scores=scores,
        source_identity_status=state["status"],
        npz_internal_coordinate_status=npz["coordinate_matrix_status"],
    )
    result = {
        **row,
        "primary_classification": primary,
        "secondary_findings": json.dumps(secondary),
        "candidate_count": len(candidates),
        "candidate_score_count": len(scores),
        "candidate_selection_reason": selection_reason,
        "selected_label_asym_id": selected.get("label_asym_id") if selected else None,
        "selected_auth_asym_id": selected.get("auth_asym_id") if selected else None,
        "selected_entity_id": selected.get("entity_id") if selected else None,
        "selected_residue_id_convention": selected.get("convention") if selected else None,
        "selected_coordinate_rmse_angstrom": selected.get("coordinate_rmse_angstrom") if selected else None,
        "selected_coordinate_max_abs_error_angstrom": (
            selected.get("coordinate_max_abs_error_angstrom") if selected else None
        ),
        "source_identity_status": state["status"],
        "source_state_evidence": json.dumps(state["records"], sort_keys=True),
        "npz_distance_matrix_semantics": npz["distance_matrix_semantics"],
        "npz_internal_coordinate_status": npz["coordinate_matrix_status"],
        "npz_internal_coordinate_rmse_angstrom": npz["coordinate_matrix_rmse_angstrom"],
        "npz_internal_coordinate_max_abs_error_angstrom": npz["coordinate_matrix_max_abs_error_angstrom"],
    }
    return result, scores


def _write_readme(output_dir: Path, protocol: dict[str, Any]) -> None:
    counts = "\n".join(f"- `{key}`: {value}" for key, value in sorted(protocol["counts_by_classification"].items()))
    (output_dir / "README.md").write_text(
        f"""# Sequence Provenance Failure Forensics

This bounded investigation classifies the {protocol["failure_count"]} blocking
rows from the 250-source v2 pilot and a small deterministic passing-control
set. Frequencies are clustered within stratified source files and are **not**
corpus prevalence estimates.

## Primary Classifications

{counts}

Candidate rows retain every protein label/auth/entity/model mapping and every
explicit residue-ID convention. A sample is reassigned only when exactly one
author-chain-supported candidate has independent sequence/residue or coordinate
support. Minimum coordinate error alone is never a selection rule.

`distance_matrix` is checked against the NPZ's physical C-alpha coordinates
before raw reconstruction, preventing normalized/physical confusion. Raw file
size and nanosecond mtime are compared with read-only preprocessing state, but
historical source SHA-256 was not stored; this limitation is reported
explicitly.
"""
    )


def _finalize(connection: sqlite3.Connection, output_dir: Path, protocol: dict[str, Any]) -> None:
    cases = [json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM cases WHERE is_control=0")]
    controls = [json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM cases WHERE is_control=1")]
    candidates = [json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM candidates")]
    _atomic_frame(pd.DataFrame(cases), output_dir / "per_failure_classification.parquet")
    _atomic_frame(pd.DataFrame(candidates), output_dir / "candidate_chain_entity_mappings.parquet")
    classification = pd.DataFrame(
        Counter(row["primary_classification"] for row in cases).items(), columns=["classification", "count"]
    ).sort_values("classification")
    _atomic_frame(classification, output_dir / "classification_summary.csv")
    coordinate_rows = [
        {
            "classification": name,
            "count": len(values),
            "coordinate_available_count": sum(row["selected_coordinate_rmse_angstrom"] is not None for row in values),
            "median_rmse_angstrom": pd.Series(
                [
                    row["selected_coordinate_rmse_angstrom"]
                    for row in values
                    if row["selected_coordinate_rmse_angstrom"] is not None
                ]
            ).median(),
            "maximum_abs_error_angstrom": max(
                (
                    row["selected_coordinate_max_abs_error_angstrom"]
                    for row in values
                    if row["selected_coordinate_max_abs_error_angstrom"] is not None
                ),
                default=None,
            ),
        }
        for name in sorted(CLASSIFICATIONS)
        if (values := [row for row in cases if row["primary_classification"] == name])
    ]
    _atomic_frame(pd.DataFrame(coordinate_rows), output_dir / "coordinate_error_summary.csv")
    convention = Counter(
        (
            row["convention"],
            "passing_control" if row.get("is_passing_control") else "blocking_case",
            str(bool(row["strong_evidential_match"])).lower(),
        )
        for row in candidates
    )
    _atomic_frame(
        pd.DataFrame(
            [
                {
                    "residue_id_convention": key[0],
                    "cohort": key[1],
                    "strong_evidential_match": key[2],
                    "count": value,
                }
                for key, value in sorted(convention.items())
            ]
        ),
        output_dir / "residue_id_convention_summary.csv",
    )
    source_versions = Counter((row["source_identity_status"], row["primary_classification"]) for row in cases)
    _atomic_frame(
        pd.DataFrame(
            [
                {"source_identity_status": key[0], "classification": key[1], "count": value}
                for key, value in sorted(source_versions.items())
            ]
        ),
        output_dir / "raw_source_version_summary.csv",
    )
    large_error = max(cases, key=lambda row: float(row.get("coordinate_distance_max_abs_error_angstrom") or -1))
    representative_ids = set(REPRESENTATIVE_IDS) | {str(large_error["sample_id"]).lower()}
    three_jbt = next(
        (
            row
            for row in cases
            if str(row["sample_id"]).lower().startswith("3jbt_") and str(row["sample_id"]).lower() != "3jbt_b"
        ),
        None,
    )
    representatives = [row for row in cases if str(row["sample_id"]).lower() in representative_ids]
    if three_jbt:
        representatives.append(three_jbt)
    representatives.extend(controls)
    temporary = output_dir / f".representative_case_reports.jsonl.{os.getpid()}.tmp"
    with temporary.open("w") as handle:
        for row in representatives:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output_dir / "representative_case_reports.jsonl")
    _write_readme(output_dir, protocol)


def run_forensics(
    *,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    state_dbs: tuple[Path, ...] = DEFAULT_STATE_DBS,
    resume: bool = False,
    expected_failure_count: int = EXPECTED_FAILURE_COUNT,
    controls_per_method: int = 1,
    stop_after: int | None = None,
) -> Path:
    """Run the bounded forensic investigation; ``stop_after`` supports interruption tests."""
    failures, controls, hashes_before = _preflight(
        audit_dir,
        output_dir,
        state_dbs,
        resume=resume,
        expected_failure_count=expected_failure_count,
        controls_per_method=controls_per_method,
    )
    if resume:
        previous_protocol = json.loads((output_dir / "forensic_protocol.incomplete.json").read_text())
        persisted_hashes = previous_protocol.get("input_hashes_before")
        if persisted_hashes != hashes_before:
            raise ValueError("Forensic inputs changed since the interrupted run; refusing to resume")
    tolerance = _serialization_tolerance(failures, controls)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol = {
        "status": "running",
        "started_utc": _utc_now(),
        "runtime_seconds": 0.0,
        "failure_count": len(failures),
        "passing_control_count": len(controls),
        "serialization_tolerance": tolerance,
        "input_hashes_before": hashes_before,
        "raw_inputs_unchanged": None,
        "pilot_frequency_interpretation": (
            "Frequencies are clustered within 250 stratified sources and are not corpus prevalence estimates."
        ),
        "run_configuration": {
            "audit_dir": str(audit_dir),
            "output_dir": str(output_dir),
            "state_databases": [str(path) for path in state_dbs],
            "expected_failure_count": expected_failure_count,
            "passing_controls_per_method": controls_per_method,
        },
        "code_provenance": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "script_sha256": sha256_file(Path(__file__)),
            "forensics_module_sha256": sha256_file(
                Path("src/protein_distance_diffusion/evaluation/provenance_forensics.py")
            ),
            "raw_parser_module_sha256": sha256_file(
                Path("src/protein_distance_diffusion/evaluation/raw_sequence_readiness.py")
            ),
            "historical_mmcif_parser_sha256": sha256_file(Path("src/protein_distance_diffusion/data/mmcif_parser.py")),
            "historical_preprocess_script_sha256": sha256_file(Path("scripts/preprocess_pdb.py")),
            "audit_run_config_sha256": sha256_file(audit_dir / "run_config.json"),
            "audit_protocol_sha256": sha256_file(audit_dir / "sequence_readiness_protocol.json"),
        },
        "preprocessing_state_provenance": _state_database_provenance(state_dbs),
    }
    _atomic_json(output_dir / "forensic_protocol.incomplete.json", protocol)
    connection = _connect_output_state(output_dir / "forensic_state.sqlite")
    raw_cache: dict[str, list[dict[str, Any]]] = {}
    processed = 0
    try:
        for is_control, row in tqdm(
            [(False, item) for item in failures] + [(True, item) for item in controls],
            desc="Sequence provenance forensics",
            unit="sample",
        ):
            done = connection.execute("SELECT status FROM cases WHERE sample_id=?", (row["sample_id"],)).fetchone()
            if done and done[0] == "completed":
                continue
            source = row["source_file"]
            if source not in raw_cache:
                raw_cache[source], _counts = inspect_mmcif(source)
            result, scores = _process_case(
                row,
                raw_rows=raw_cache[source],
                state_dbs=state_dbs,
                tolerance=float(tolerance["selected_tolerance_angstrom"]),
            )
            if is_control:
                result["primary_classification"] = "passing_control"
            with connection:
                connection.execute("DELETE FROM candidates WHERE sample_id=?", (row["sample_id"],))
                connection.executemany(
                    "INSERT INTO candidates VALUES (?,?,?)",
                    [
                        (
                            row["sample_id"],
                            index,
                            json.dumps(
                                {**score, "sample_id": row["sample_id"], "is_passing_control": bool(is_control)},
                                sort_keys=True,
                            ),
                        )
                        for index, score in enumerate(scores)
                    ],
                )
                connection.execute(
                    "INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?)",
                    (row["sample_id"], int(is_control), "completed", json.dumps(result, sort_keys=True), _utc_now()),
                )
            processed += 1
            protocol.update(
                runtime_seconds=time.monotonic() - started,
                processed_count=processed,
                heartbeat_utc=_utc_now(),
            )
            _atomic_json(output_dir / "forensic_protocol.incomplete.json", protocol)
            if stop_after is not None and processed >= stop_after:
                raise KeyboardInterrupt

        hashes_after = {path: sha256_file(path) for path in hashes_before}
        if hashes_after != hashes_before:
            raise RuntimeError("A forensic input changed during analysis")
        case_rows = [
            json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM cases WHERE is_control=0")
        ]
        protocol.update(
            status="completed",
            completed_utc=_utc_now(),
            runtime_seconds=time.monotonic() - started,
            processed_count=len(case_rows) + len(controls),
            counts_by_classification=dict(sorted(Counter(row["primary_classification"] for row in case_rows).items())),
            counts_by_experimental_method=dict(
                sorted(Counter(row.get("experimental_method", "unknown") for row in case_rows).items())
            ),
            counts_by_split=dict(sorted(Counter(row.get("split", "unknown") for row in case_rows).items())),
            input_hashes_after=hashes_after,
            raw_inputs_unchanged=True,
        )
        _finalize(connection, output_dir, protocol)
        _atomic_json(output_dir / "forensic_protocol.json", protocol)
        (output_dir / "forensic_protocol.incomplete.json").unlink()
    except KeyboardInterrupt:
        protocol.update(status="interrupted", runtime_seconds=time.monotonic() - started, heartbeat_utc=_utc_now())
        protocol["resume_command"] = "rerun the identical command with --resume"
        _atomic_json(output_dir / "forensic_protocol.incomplete.json", protocol)
        raise
    finally:
        connection.close()
    return output_dir / "forensic_protocol.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preprocess-state-db", type=Path, action="append", dest="state_dbs")
    parser.add_argument("--expected-failure-count", type=int, default=EXPECTED_FAILURE_COUNT)
    parser.add_argument("--passing-controls-per-method", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    state_dbs = tuple(args.state_dbs) if args.state_dbs else DEFAULT_STATE_DBS
    protocol = run_forensics(
        audit_dir=args.audit_dir,
        output_dir=args.output_dir,
        state_dbs=state_dbs,
        resume=args.resume,
        expected_failure_count=args.expected_failure_count,
        controls_per_method=args.passing_controls_per_method,
    )
    print(f"Wrote forensic protocol: {protocol}")


if __name__ == "__main__":
    main()
