"""Synthetic tests for bounded sequence-provenance forensics."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protein_distance_diffusion.data.preprocess import compute_distance_matrix
from protein_distance_diffusion.evaluation.provenance_forensics import (
    classify_case,
    convention_indices,
    inspect_npz_forensics,
    refine_pair_status,
    score_candidate,
    select_unique_candidate,
)
from protein_distance_diffusion.evaluation.sequence_readiness import sha256_file


def _raw(
    *,
    label: str = "X",
    author: str = "A",
    sequence: str = "ACG",
    auth_ids: tuple[str, ...] = ("10", "11", "12"),
    label_ids: tuple[str, ...] = ("1", "2", "3"),
    coordinate_scale: float = 3.8,
) -> dict:
    coordinates = [[index * coordinate_scale, 0.0, 0.0] for index in range(len(sequence))]
    return {
        "label_asym_id": label,
        "auth_asym_id": author,
        "entity_id": f"entity-{label}",
        "polymer_type": "polypeptide(L)",
        "model_id": "1",
        "atom_sequence": sequence,
        "seqres_sequence": sequence,
        "entity_poly_sequence": sequence,
        "residue_tokens": json.dumps(list(sequence)),
        "auth_sequence_ids": json.dumps(list(auth_ids)),
        "label_sequence_ids": json.dumps(list(label_ids)),
        "insertion_codes": json.dumps([""] * len(sequence)),
        "selected_calpha_coordinates": json.dumps(coordinates),
    }


def _matrix(sequence: str = "ACG", scale: float = 3.8) -> np.ndarray:
    coordinates = np.asarray([[index * scale, 0.0, 0.0] for index in range(len(sequence))], dtype=np.float32)
    return compute_distance_matrix(coordinates)


def _scores(raw: dict, *, ids: list[str] | None = None, metadata: dict | None = None, matrix=None):
    return score_candidate(
        raw,
        matrix_sequence="ACG",
        matrix_ids=ids or ["10", "11", "12"],
        matrix=_matrix() if matrix is None else matrix,
        metadata=metadata or {},
        requested_chain="A",
        tolerance=1e-4,
    )


def test_multiple_label_chains_select_only_unique_author_supported_candidate() -> None:
    correct = _scores(_raw(label="X"))
    wrong = _scores(_raw(label="Y", sequence="GGG", coordinate_scale=20.0))
    selected, reason = select_unique_candidate([*correct, *wrong])
    assert selected is not None
    assert selected["label_asym_id"] == "X"
    assert reason == "unique_author_chain_evidential_match"


def test_equally_plausible_author_chain_candidates_are_not_selected() -> None:
    selected, reason = select_unique_candidate([*_scores(_raw(label="X")), *_scores(_raw(label="Y"))])
    assert selected is None
    assert reason == "multiple_author_chain_candidates_remain_evidentially_plausible"


def test_label_auth_and_positional_conventions_are_distinct() -> None:
    raw = _raw()
    assert convention_indices(raw, ["10", "11", "12"], "auth_seq_id_insertion")[0] == [0, 1, 2]
    assert convention_indices(raw, ["1", "2", "3"], "label_seq_id")[0] == [0, 1, 2]
    assert convention_indices(raw, ["0", "1", "2"], "position_zero_based")[0] == [0, 1, 2]
    assert convention_indices(raw, ["1", "2", "3"], "position_one_based")[0] == [0, 1, 2]


def test_insertion_codes_are_part_of_author_residue_identity() -> None:
    raw = _raw()
    raw["insertion_codes"] = json.dumps(["", "A", ""])
    indices, ambiguous = convention_indices(raw, ["10", "11A", "12"], "auth_seq_id_insertion")
    assert indices == [0, 1, 2]
    assert ambiguous is False


def test_terminal_trim_and_stale_trim_metadata_are_separate() -> None:
    raw = _raw(sequence="TACGG", auth_ids=("9", "10", "11", "12", "13"), label_ids=("1", "2", "3", "4", "5"))
    raw["selected_calpha_coordinates"] = json.dumps([[-3.8, 0, 0], [0, 0, 0], [3.8, 0, 0], [7.6, 0, 0], [11.4, 0, 0]])
    metadata = {
        "retained_start_label_seq_id": 2,
        "retained_end_label_seq_id": 4,
        "trimmed_n_terminal_residues": 1,
        "trimmed_c_terminal_residues": 1,
    }
    scores = _scores(raw, metadata=metadata)
    auth = next(row for row in scores if row["convention"] == "auth_seq_id_insertion")
    assert auth["trim_metadata_compatible"] is True
    original = {
        "raw_label_asym_id": "X",
        "alignment_reason": "residue_ids_match_but_recorded_terminal_trim_metadata_disagrees",
        "terminal_trimming_applied": True,
    }
    classification = classify_case(
        original=original,
        scores=scores,
        source_identity_status="state_size_mtime_match_sha_unavailable",
        npz_internal_coordinate_status="passed",
    )[0]
    assert classification == "legacy_trim_metadata_inconsistency"


def test_wrong_polymer_produces_large_coordinate_error_without_tolerance_relaxation() -> None:
    wrong = _scores(_raw(coordinate_scale=40.0))
    auth = next(row for row in wrong if row["convention"] == "auth_seq_id_insertion")
    assert auth["coordinate_status"] == "failed"
    assert auth["coordinate_max_abs_error_angstrom"] > 50.0


def test_npz_physical_matrix_is_distinguished_from_normalized_matrix(tmp_path: Path) -> None:
    coordinates = np.asarray([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]], dtype=np.float32)
    path = tmp_path / "normalized.npz"
    np.savez(
        path,
        pdb_id=np.asarray("TEST"),
        chain_id=np.asarray("A"),
        sequence=np.asarray("ACG"),
        residue_ids=np.asarray(["10", "11", "12"]),
        ca_coordinates=coordinates,
        distance_matrix=compute_distance_matrix(coordinates) / 10.0,
        metadata=np.asarray(json.dumps({"source_file": "source.cif", "model_number": 1})),
    )
    result = inspect_npz_forensics(path)
    assert result["coordinate_matrix_status"] == "failed"
    assert result["distance_matrix_semantics"] == "physical_calpha_angstrom"


def test_changed_raw_source_state_is_detected(tmp_path: Path) -> None:
    from scripts.diagnose_sequence_provenance_failures import _state_evidence

    source = tmp_path / "source.cif"
    source.write_text("new")
    database = tmp_path / "state.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE source_files (
                source_path TEXT PRIMARY KEY, source_size INTEGER, source_mtime_ns INTEGER,
                status TEXT, config_hash TEXT, attempt_count INTEGER
            );
            CREATE TABLE manifest_rows (sample_id TEXT PRIMARY KEY, row_json TEXT, config_hash TEXT);
            CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        connection.execute(
            "INSERT INTO source_files VALUES (?,?,?,?,?,?)", (str(source), 999, 1, "completed", "config", 1)
        )
        connection.execute("INSERT INTO manifest_rows VALUES (?,?,?)", ("test_A", "{}", "config"))
    evidence = _state_evidence(source, "test_A", (database,))
    assert evidence["status"] == "state_identity_changed"


def test_invalid_preflight_does_not_create_output(tmp_path: Path) -> None:
    from scripts.diagnose_sequence_provenance_failures import _preflight

    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        _preflight(
            tmp_path / "missing",
            output,
            (tmp_path / "missing.sqlite",),
            resume=False,
            expected_failure_count=1,
            controls_per_method=1,
        )
    assert not output.exists()


def test_scoring_does_not_mutate_raw_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source.cif"
    source.write_text("immutable")
    before = sha256_file(source)
    _scores(_raw())
    assert sha256_file(source) == before


def test_exact_geometry_with_historical_identity_gap_is_conditional() -> None:
    result = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "alignment_reason": "unique_sequence_interval_lacks_residue_identity_provenance",
            "source_identity_status": "state_size_mtime_match_sha_unavailable",
        },
        _scores(_raw()),
    )
    assert result["matrix_sequence_occurrence_count_in_candidate"] == 1
    assert result["coordinate_matrix_match"] is True
    assert result["strict_provenance_status"] == "residue_identity_provenance_incomplete"
    assert result["training_eligibility"] == "conditionally_verified_pair"


def test_missing_raw_calpha_is_missing_evidence_not_sequence_failure() -> None:
    raw = _raw()
    raw["selected_calpha_coordinates"] = json.dumps([None, None, None])
    result = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "state_size_mtime_match_sha_unavailable",
        },
        _scores(raw),
    )
    assert result["coordinate_matrix_status"] == "unavailable"
    assert result["refined_primary_classification"] == "unique_sequence_pair_coordinate_unavailable"
    assert result["training_eligibility"] == "conditionally_verified_pair"


def test_multiple_author_candidates_remain_excluded_even_without_historical_sha() -> None:
    scores = [*_scores(_raw(label="X")), *_scores(_raw(label="Y"))]
    result = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "state_size_mtime_match_sha_unavailable",
        },
        scores,
    )
    assert result["refined_primary_classification"] == "multiple_author_linked_polymer_candidates"
    assert result["training_eligibility"] == "excluded_or_unresolved"


def test_coordinate_disagreement_with_label_collision_is_not_force_resolved() -> None:
    author = _scores(_raw(label="X", coordinate_scale=40.0))
    label_collision = _scores(_raw(label="A", author="B", sequence="GGG", coordinate_scale=20.0))
    result = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "state_metadata_mismatch",
        },
        [*author, *label_collision],
    )
    assert result["coordinate_matrix_status"] == "failed"
    assert result["refined_primary_classification"] == "unresolved_ambiguity"
    assert result["training_eligibility"] == "excluded_or_unresolved"


def test_source_sha_availability_changes_strictness_not_pair_classification() -> None:
    case = {"matrix_sequence": "ACG", "matrix_manifest_sequence_match": True}
    without_sha = refine_pair_status(
        {**case, "source_identity_status": "state_size_mtime_match_sha_unavailable"}, _scores(_raw())
    )
    with_sha = refine_pair_status({**case, "source_identity_status": "historical_sha_verified"}, _scores(_raw()))
    assert without_sha["refined_primary_classification"] == "verified_sequence_geometry_pair"
    assert with_sha["refined_primary_classification"] == "verified_sequence_geometry_pair"
    assert without_sha["training_eligibility"] == "conditionally_verified_pair"
    assert with_sha["training_eligibility"] == "strict_verified_pair"


def test_zero_based_residue_ids_and_stale_trim_metadata_are_reconciled() -> None:
    zero_based = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "historical_sha_verified",
        },
        _scores(_raw(), ids=["0", "1", "2"]),
    )
    assert zero_based["zero_based_positions_match"] is True
    assert zero_based["training_eligibility"] == "strict_verified_pair"

    label_namespace = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "state_size_mtime_match_sha_unavailable",
        },
        _scores(_raw(), ids=["1", "2", "3"]),
    )
    assert label_namespace["refined_primary_classification"] == "residue_id_namespace_mismatch"
    assert label_namespace["training_eligibility"] == "conditionally_verified_pair"

    stale = refine_pair_status(
        {
            "matrix_sequence": "ACG",
            "matrix_manifest_sequence_match": True,
            "source_identity_status": "historical_sha_verified",
        },
        _scores(
            _raw(),
            metadata={
                "retained_start_label_seq_id": 99,
                "retained_end_label_seq_id": 100,
                "trimmed_n_terminal_residues": 4,
                "trimmed_c_terminal_residues": 5,
            },
        ),
    )
    assert stale["refined_primary_classification"] == "legacy_trim_metadata_inconsistency"
    assert stale["training_eligibility"] == "strict_verified_pair"


def test_bounded_run_interrupts_resumes_and_preserves_inputs(tmp_path: Path) -> None:
    from scripts.diagnose_sequence_provenance_failures import run_forensics

    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    raw = tmp_path / "raw.cif"
    raw.write_text((Path(__file__).parent / "fixtures" / "two_residue_xray.cif").read_text())
    coordinates = np.asarray([[0, 0, 0], [3, 4, 0]], dtype=np.float32)
    npz = tmp_path / "sample.npz"
    np.savez(
        npz,
        pdb_id=np.asarray("XRAY"),
        chain_id=np.asarray("A"),
        sequence=np.asarray("GA"),
        residue_ids=np.asarray(["1", "2"]),
        ca_coordinates=coordinates,
        distance_matrix=compute_distance_matrix(coordinates),
        metadata=np.asarray(json.dumps({"source_file": str(raw), "model_number": 1})),
    )
    failure = {
        "sample_id": "xray_A",
        "pdb_id": "XRAY",
        "chain_id": "A",
        "model_id": "1",
        "matrix_path": str(npz),
        "experimental_method": "X-RAY DIFFRACTION",
        "split": "train",
        "alignment_class": "sequence_mismatch",
        "alignment_reason": "ordered_residue_ids_disagree_with_sequence",
        "raw_label_asym_id": "B",
        "terminal_trimming_applied": "False",
        "coordinate_verification_status": "failed",
        "coordinate_distance_max_abs_error_angstrom": "50",
    }
    pd.DataFrame([failure]).to_csv(audit_dir / "raw_blocking_failures.csv", index=False)
    (audit_dir / "seqres_atom_matrix_alignments.jsonl").write_text("")
    (audit_dir / "sequence_readiness_protocol.json").write_text("{}\n")
    (audit_dir / "run_config.json").write_text("{}\n")
    state = tmp_path / "preprocess.sqlite"
    stat = raw.stat()
    with sqlite3.connect(state) as connection:
        connection.executescript(
            """
            CREATE TABLE source_files (
                source_path TEXT PRIMARY KEY, source_size INTEGER, source_mtime_ns INTEGER,
                status TEXT, config_hash TEXT, attempt_count INTEGER
            );
            CREATE TABLE manifest_rows (sample_id TEXT PRIMARY KEY, row_json TEXT, config_hash TEXT);
            CREATE TABLE run_metadata (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        connection.execute(
            "INSERT INTO source_files VALUES (?,?,?,?,?,?)",
            (str(raw), stat.st_size, stat.st_mtime_ns, "completed", "config", 1),
        )
        connection.execute("INSERT INTO manifest_rows VALUES (?,?,?)", ("xray_A", "{}", "config"))
        connection.execute("INSERT INTO run_metadata VALUES (?,?)", ("config_hash", "config"))
    input_hashes = {path: sha256_file(path) for path in (raw, npz, state)}
    output = tmp_path / "forensics"
    with pytest.raises(KeyboardInterrupt):
        run_forensics(
            audit_dir=audit_dir,
            output_dir=output,
            state_dbs=(state,),
            expected_failure_count=1,
            controls_per_method=0,
            stop_after=1,
        )
    assert not (output / "forensic_protocol.json").exists()
    assert json.loads((output / "forensic_protocol.incomplete.json").read_text())["status"] == "interrupted"
    protocol_path = run_forensics(
        audit_dir=audit_dir,
        output_dir=output,
        state_dbs=(state,),
        expected_failure_count=1,
        controls_per_method=0,
        resume=True,
    )
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "completed"
    assert protocol["raw_inputs_unchanged"] is True
    assert protocol["counts_by_classification"] == {"audit_chain_entity_mapping_bug": 1}
    for filename in (
        "per_failure_classification.parquet",
        "candidate_chain_entity_mappings.parquet",
        "classification_summary.csv",
        "coordinate_error_summary.csv",
        "residue_id_convention_summary.csv",
        "raw_source_version_summary.csv",
        "representative_case_reports.jsonl",
        "README.md",
    ):
        assert (output / filename).is_file()
    assert {path: sha256_file(path) for path in input_hashes} == input_hashes
