"""Tests for the analysis-only sequence-data readiness audit."""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protein_distance_diffusion.constants import AA_TO_TOKEN
from protein_distance_diffusion.evaluation.sequence_readiness import (
    infer_sequence_source,
    inspect_mmcif,
    inspect_processed_npz,
    sequence_sha256,
    sha256_file,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _write_entity_aware_cif(path: Path) -> None:
    path.write_text(
        """data_ENTITY
_exptl.method 'X-RAY DIFFRACTION'
loop_
_entity_poly.entity_id
_entity_poly.type
1 'polypeptide(L)'
5 polydeoxyribonucleotide
6 polyribonucleotide
#
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
W 2
I 3
G 4
D 5
R 6
#
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 1 ALA 10 X .
A 1 2 MSE 11 X .
A 1 3 GLY 12 X .
D 5 1 DA 1 D .
R 6 1 U 1 R .
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
ATOM 1 CA . ALA A 1 1 ? 0 0 0 1.0 10 X 1
HETATM 2 CA A MSE A 1 2 ? 3.8 0 0 0.5 11 X 1
HETATM 3 CA B MSE A 1 2 ? 9.0 0 0 0.4 11 X 1
ATOM 4 CA . GLY A 1 3 ? 7.6 0 0 1.0 12 X 1
HETATM 5 O . HOH W 2 . ? 1 1 1 1.0 1 X 1
HETATM 6 ZN . ZN I 3 . ? 2 2 2 1.0 1 X 1
HETATM 7 C1 . NAG G 4 . ? 3 3 3 1.0 1 X 1
ATOM 8 P . DA D 5 1 ? 4 4 4 1.0 1 D 1
ATOM 9 P . U R 6 1 ? 5 5 5 1.0 1 R 1
#
"""
    )


def _write_processed_npz(path: Path, *, sequence: str, source_file: Path, sample_id: str = "xray_A") -> None:
    length = len(sequence)
    coordinates = np.stack([np.arange(length) * 5.0, np.zeros(length), np.zeros(length)], axis=1)
    matrix = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1).astype(np.float32)
    metadata = {
        "source_file": str(source_file),
        "model_number": 1,
        "retained_insertion_codes": [""] * length,
        "selected_altlocs": [None] * length,
    }
    np.savez(
        path,
        sample_id=np.asarray(sample_id),
        pdb_id=np.asarray("XRAY"),
        chain_id=np.asarray("A"),
        sequence=np.asarray(sequence),
        sequence_tokens=np.asarray([AA_TO_TOKEN[token] for token in sequence]),
        residue_ids=np.asarray([str(index + 1) for index in range(length)]),
        residue_mask=np.ones(length, dtype=bool),
        ca_coordinates=coordinates.astype(np.float32),
        distance_matrix=matrix,
        metadata=np.asarray(json.dumps(metadata)),
    )


def test_real_gemmi_raw_inventory_and_sequence_source() -> None:
    rows, tokens = inspect_mmcif(FIXTURES / "two_residue_xray.cif")
    assert len(rows) == 1
    assert rows[0]["atom_sequence"] == "GA"
    assert rows[0]["scheme_sequence"] is None
    assert rows[0]["missing_calpha_count"] == 0
    assert infer_sequence_source("GA", rows[0]) == "ATOM-derived"
    assert tokens[("biological_residue", "coordinate_model", "GLY", "canonical")] == 1
    assert tokens[("biological_residue", "coordinate_model", "ALA", "canonical")] == 1


def test_sequence_source_prefers_scheme_and_preserves_unknown() -> None:
    assert infer_sequence_source("AM", {"scheme_sequence": "GAMV", "atom_sequence": "AM"}) == "SEQRES-derived"
    assert infer_sequence_source("GA", {"scheme_sequence": None, "atom_sequence": "GA"}) == "ATOM-derived"
    assert infer_sequence_source("GA", None) == "unknown"


def test_processed_npz_reports_sequence_matrix_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    _write_processed_npz(path, sequence="GA", source_file=FIXTURES / "two_residue_xray.cif")
    result = inspect_processed_npz(path)
    assert result["npz_sequence"] == "GA"
    assert result["matrix_is_square"] is True
    assert result["matrix_rows"] == 2
    assert result["residue_mask_true_count"] == 2
    assert result["sequence_tokens_match_sequence"] is True
    assert result["coordinate_count"] == 2
    assert result["retained_insertion_codes"] == ["", ""]


def test_entity_aware_parser_retains_mse_and_excludes_nonprotein_components(tmp_path: Path) -> None:
    path = tmp_path / "entity.cif"
    _write_entity_aware_cif(path)
    rows, counts = inspect_mmcif(path)
    assert [(row["label_asym_id"], row["auth_asym_id"]) for row in rows] == [("A", "X")]
    assert rows[0]["seqres_sequence"] == "AMG"
    assert rows[0]["atom_sequence"] == "AMG"
    assert json.loads(rows[0]["modified_residue_counts"]) == {"MSE": 1}
    assert json.loads(rows[0]["unknown_residue_counts"]) == {}
    context = json.loads(rows[0]["nonpolymer_component_counts"])
    assert {"HOH", "ZN", "NAG", "DA", "U"} <= set(context)
    assert rows[0]["missing_residue_count"] == 0
    assert rows[0]["missing_calpha_count"] == 0
    assert rows[0]["multiple_altloc_candidate_position_count"] == 1
    assert json.loads(rows[0]["selected_altlocs"])[1] == "A"
    assert counts[("biological_residue", "declared_polymer", "MSE", "modified")] == 1
    assert all(key[2] not in {"HOH", "ZN", "NAG", "DA", "U"} for key in counts)


def test_alignment_taxonomy_uses_ids_trim_metadata_and_coordinates(tmp_path: Path) -> None:
    import scripts.audit_sequence_data_readiness as audit

    matrix_path = tmp_path / "trimmed.npz"
    source = tmp_path / "source.cif"
    source.write_text("placeholder")
    coordinates = np.asarray([[3.8, 0, 0], [7.6, 0, 0]], dtype=np.float32)
    metadata = {
        "source_file": str(source),
        "model_number": 1,
        "retained_start_label_seq_id": 2,
        "retained_end_label_seq_id": 3,
        "trimmed_n_terminal_residues": 1,
        "trimmed_c_terminal_residues": 1,
        "terminal_trimming_applied": True,
        "retained_insertion_codes": ["", ""],
        "selected_altlocs": [None, None],
    }
    np.savez(
        matrix_path,
        sequence=np.asarray("CG"),
        sequence_tokens=np.asarray([AA_TO_TOKEN["C"], AA_TO_TOKEN["G"]]),
        residue_ids=np.asarray(["11", "12"]),
        residue_mask=np.ones(2, dtype=bool),
        ca_coordinates=coordinates,
        distance_matrix=np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1),
        metadata=np.asarray(json.dumps(metadata)),
    )
    inspected = inspect_processed_npz(matrix_path, include_geometry=True)
    raw = {
        "seqres_sequence": "ACGT",
        "atom_sequence": "CG",
        "auth_sequence_ids": json.dumps(["10", "11", "12", "13"]),
        "label_sequence_ids": json.dumps(["1", "2", "3", "4"]),
        "insertion_codes": json.dumps(["", "", "", ""]),
        "residue_tokens": json.dumps(list("ACGT")),
        "selected_calpha_coordinates": json.dumps([None, [3.8, 0, 0], [7.6, 0, 0], None]),
    }
    result = audit._classify_alignment({"sequence": "CG"}, raw, inspected)
    assert result["alignment_class"] == "valid_terminal_trim"
    assert result["coordinate_verification_status"] == "passed"
    assert result["coordinate_distance_max_abs_error_angstrom"] <= 1e-4

    repeated = {**raw, "seqres_sequence": "ACGTCG", "residue_tokens": json.dumps(list("ACGTCG"))}
    assert (
        audit._classify_alignment({"sequence": "CG"}, repeated, inspected)["alignment_class"] == "valid_terminal_trim"
    )
    unresolved_repeated = {
        **repeated,
        "auth_sequence_ids": "[]",
        "label_sequence_ids": "[]",
        "insertion_codes": "[]",
    }
    assert (
        audit._classify_alignment({"sequence": "CG"}, unresolved_repeated, inspected)["alignment_class"]
        == "ambiguous_subsequence"
    )
    internal_npz = {**inspected, "residue_ids": ["10", "12"], "npz_sequence": "AG"}
    assert audit._classify_alignment({"sequence": "AG"}, raw, internal_npz)["alignment_class"] == "internal_gap"


def _audit_inputs(tmp_path: Path, *, repeated_source: bool = False) -> dict[str, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / "xray.cif"
    shutil.copyfile(FIXTURES / "two_residue_xray.cif", raw_path)
    sample_path = tmp_path / "sample.npz"
    _write_processed_npz(sample_path, sequence="GA", source_file=raw_path)
    row = {
        "sample_id": "xray_A",
        "pdb_id": "XRAY",
        "chain_id": "A",
        "model_number": 1,
        "sequence": "GA",
        "length": 2,
        "path": str(sample_path),
        "source_file": str(raw_path),
        "sequence_hash": sequence_sha256("GA"),
        "cluster_id": "cluster-A",
        "split_group_id": "group-A",
        "experimental_method": "X-RAY DIFFRACTION",
        "terminal_trimming_applied": False,
        "missing_calpha_policy": "reject",
    }
    processed_rows = [row]
    if repeated_source:
        second_path = tmp_path / "sample_2.npz"
        _write_processed_npz(second_path, sequence="GA", source_file=raw_path, sample_id="xray_B")
        processed_rows.append({**row, "sample_id": "xray_B", "path": str(second_path), "chain_id": "B"})
    processed_manifest = tmp_path / "processed.parquet"
    train_manifest = tmp_path / "train.parquet"
    validation_manifest = tmp_path / "validation.parquet"
    pd.DataFrame(processed_rows).drop(columns=["sequence_hash"]).to_parquet(processed_manifest, index=False)
    pd.DataFrame([row]).to_parquet(train_manifest, index=False)
    pd.DataFrame([{**row, "sample_id": "validation-copy"}]).to_parquet(validation_manifest, index=False)
    return {
        "raw_dir": raw_dir,
        "raw_path": raw_path,
        "sample_path": sample_path,
        "processed_manifest": processed_manifest,
        "train_manifest": train_manifest,
        "validation_manifest": validation_manifest,
    }


def test_manifest_only_makes_zero_raw_parser_calls_and_counts_excluded(tmp_path: Path, monkeypatch) -> None:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path, repeated_source=True)
    monkeypatch.setattr(audit, "inspect_mmcif", lambda _path: pytest.fail("raw parser called"))
    protocol_path = audit.run_audit(
        audit_mode="manifest-only",
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "manifest_audit",
        batch_size=1,
    )
    protocol = json.loads(protocol_path.read_text())
    assert protocol["audit_mode"] == "manifest-only"
    assert protocol["parser_call_count"] == 0
    assert protocol["manifest_row_counts"] == {"processed": 2, "train": 1, "validation": 1}
    assert protocol["rows_by_split_status"] == {"excluded": 1, "train": 1}
    summary = json.loads((protocol_path.parent / "manifest_summary.json").read_text())
    assert summary["sequence_hash_statistics"]["processed"] == {
        "total_rows": 2,
        "stored_hash_column_present": False,
        "stored_hash_available_count": 0,
        "stored_hash_unavailable_count": 2,
        "checked_count": 0,
        "match_count": 0,
        "mismatch_count": 0,
        "comparison_status": "unavailable",
    }
    assert summary["sequence_hash_statistics"]["train"]["comparison_status"] == "passed"
    assert summary["sequence_hash_statistics"]["validation"]["comparison_status"] == "passed"
    assert summary["model_number_statistics"]["source_column"] == "model_number"
    assert summary["model_number_statistics"]["distribution"] == {"1": 2}
    assert summary["model_number_statistics"]["rows_with_model_number_greater_than_1"] == 0
    criteria = json.loads((protocol_path.parent / "acceptance_criteria_results.json").read_text())["criteria"]
    statuses = {item["criterion"]: item["status"] for item in criteria}
    assert statuses["canonical_sequence"] == "passed"
    assert statuses["manifest_npz_agreement"] == "passed"
    assert statuses["seqres_atom_matrix_provenance"] == "pending_raw_audit"
    assert statuses["nmr_cross_model_sequence_consistency"] == "pending_raw_audit"
    leakage = pd.read_csv(protocol_path.parent / "train_validation_leakage.csv")
    assert {"exact_sequence", "cluster_id", "split_group_id", "pdb_id"} <= set(leakage["policy"])


def test_available_incorrect_sequence_hash_fails(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    validation = pd.read_parquet(inputs["validation_manifest"])
    validation["sequence_hash"] = "incorrect"
    validation.to_parquet(inputs["validation_manifest"], index=False)
    protocol_path = run_audit(
        audit_mode="manifest-only",
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "incorrect_hash",
    )
    summary = json.loads((protocol_path.parent / "manifest_summary.json").read_text())
    validation_stats = summary["sequence_hash_statistics"]["validation"]
    assert validation_stats["checked_count"] == 1
    assert validation_stats["match_count"] == 0
    assert validation_stats["mismatch_count"] == 1
    assert validation_stats["comparison_status"] == "failed"
    criteria = json.loads((protocol_path.parent / "acceptance_criteria_results.json").read_text())["criteria"]
    hash_criterion = next(item for item in criteria if item["criterion"] == "stored_sequence_hash_verification")
    assert hash_criterion["status"] == "failed"


def test_model_number_statistics_use_manifest_model_number(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    processed = pd.read_parquet(inputs["processed_manifest"])
    processed["model_number"] = 2
    processed["experimental_method"] = "SOLUTION NMR"
    processed.to_parquet(inputs["processed_manifest"], index=False)
    protocol_path = run_audit(
        audit_mode="manifest-only",
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "model_statistics",
    )
    model_stats = json.loads((protocol_path.parent / "manifest_summary.json").read_text())["model_number_statistics"]
    assert model_stats["distribution"] == {"2": 1}
    assert model_stats["rows_with_model_number_greater_than_1"] == 1
    assert model_stats["nmr_row_count"] == 1
    assert model_stats["nmr_rows_with_model_number_greater_than_1"] == 1


def test_manifest_only_interruption_is_resumable(tmp_path: Path, monkeypatch) -> None:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path, repeated_source=True)
    output = tmp_path / "manifest_resume"
    original = audit.inspect_processed_npz
    calls = 0

    def interrupt_once(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt
        return original(path)

    monkeypatch.setattr(audit, "inspect_processed_npz", interrupt_once)
    kwargs = {
        "audit_mode": "manifest-only",
        "processed_manifest": inputs["processed_manifest"],
        "train_manifest": inputs["train_manifest"],
        "validation_manifest": inputs["validation_manifest"],
        "output_dir": output,
        "batch_size": 1,
    }
    with pytest.raises(KeyboardInterrupt):
        audit.run_audit(**kwargs)
    assert not (output / "sequence_readiness_protocol.json").exists()
    partial = json.loads((output / "sequence_readiness_protocol.partial.json").read_text())
    assert partial["status"] == "interrupted"
    assert partial["current_stage"] == "npz_metadata_validation"

    monkeypatch.setattr(audit, "inspect_processed_npz", original)
    protocol_path = audit.run_audit(**kwargs, resume=True)
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "completed"
    assert protocol["parser_call_count"] == 0
    with sqlite3.connect(output / "audit_state.sqlite") as connection:
        assert (
            connection.execute("SELECT status FROM stage_state WHERE stage_name='npz_metadata_validation'").fetchone()[
                0
            ]
            == "completed"
        )


def test_large_group_analysis_is_linear_in_unique_groups(tmp_path: Path) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output = tmp_path / "performance"
    output.mkdir()
    connection = audit._connect_state(output / "audit_state.sqlite")
    insert = "INSERT INTO manifest_rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

    def rows(kind: str, count: int):
        for index in range(count):
            yield (
                kind,
                index,
                f"{kind}-{index}",
                "SAME",
                "A",
                "1",
                "same-path.npz",
                "GA",
                2,
                "same-source.cif",
                "X-RAY DIFFRACTION",
                0,
                "reject",
                "same-hash",
                "same-hash",
                "same-cluster",
                "same-split-group",
                1,
            )

    connection.executemany(insert, rows("processed", 100_000))
    connection.executemany(insert, rows("train", 10_000))
    connection.executemany(insert, rows("validation", 10_000))
    connection.commit()
    reporter = audit.StageReporter(connection, output, "manifest-only", audit._utc_now())
    started = time.monotonic()
    reporter.start("index_creation", total_count=len(audit.ANALYSIS_INDEXES))
    audit._ensure_analysis_indexes(connection, reporter)
    reporter.complete(processed_count=len(audit.ANALYSIS_INDEXES))
    summary = audit._manifest_aggregation(connection, output)
    duplicate_summary = audit._write_duplicate_groups(
        connection, output, max_examples=3, max_groups=1, reporter=reporter
    )
    leakage_summary = audit._write_leakage_groups(connection, output, max_examples=3, max_groups=1, reporter=reporter)
    ambiguity_summary = audit._write_ambiguous_groups(
        connection, output, max_examples=3, max_groups=100, reporter=reporter
    )
    elapsed = time.monotonic() - started
    connection.close()

    duplicates = pd.read_csv(output / "duplicate_findings.csv")
    leakage = pd.read_csv(output / "train_validation_leakage.csv")
    ambiguity = pd.read_csv(output / "ambiguous_manifest_pairings.csv")
    assert summary["rows_by_split_status"] == {"excluded": 100_000}
    assert len(duplicates) == 1
    assert len(leakage) == 1
    assert len(ambiguity) == 1
    assert duplicate_summary["exact_sequence"]["underlying_member_row_count"] == 100_000
    assert duplicate_summary["exact_sequence"]["groups_truncated"] is True
    assert leakage_summary["exact_sequence"]["underlying_group_count"] == 1
    assert ambiguity_summary["underlying_member_row_count"] == 100_000
    assert duplicates["representative_sample_ids"].map(lambda value: len(json.loads(value))).max() == 3
    assert elapsed < 30.0


def test_raw_pilot_parses_repeated_source_once_and_reports_sparse_strata(tmp_path: Path, monkeypatch) -> None:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path, repeated_source=True)
    calls = []
    original = audit.inspect_mmcif

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(audit, "inspect_mmcif", counted)
    protocol_path = audit.run_audit(
        audit_mode="raw-pilot",
        raw_dir=inputs["raw_dir"],
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "pilot",
        max_source_files=1,
        samples_per_stratum=3,
        pilot_seed=7,
        checkpoint_frequency=1,
    )
    protocol = json.loads(protocol_path.read_text())
    assert calls == [str(inputs["raw_path"])]
    assert protocol["selected_source_file_count"] == 1
    assert protocol["parsed_source_file_count"] == 1
    assert protocol["parser_call_count"] == 1
    coverage = pd.read_csv(protocol_path.parent / "per_stratum_coverage.csv")
    assert coverage["sparse"].any()
    assert len(list((protocol_path.parent / "partitions").glob("*.json"))) == 1
    assert protocol["coverage_accounting"]["selected_unique_source_count"] == 1
    assert protocol["coverage_accounting"]["strata_overlap"] is True
    acceptance = json.loads((protocol_path.parent / "acceptance_criteria_results.json").read_text())
    raw_criteria = {row["criterion"]: row for row in acceptance["criteria"] if row["status"].startswith("pilot_")}
    assert raw_criteria["seqres_atom_matrix_provenance"]["evidence_count"] == 2
    assert acceptance["corpus_wide_status"] == "pending_full_raw_audit"
    assert {row["status"] for row in acceptance["criteria"]} <= {
        "pilot_passed",
        "pilot_failed",
        "pending_full_raw_audit",
        "informational",
    }


def test_raw_nmr_pilot_inspects_models_absent_from_processed_manifest(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    nmr_text = (FIXTURES / "two_residue_nmr.cif").read_text()
    atom_lines = [line for line in nmr_text.splitlines() if line.startswith("ATOM ")]
    model_two = []
    for offset, line in enumerate(atom_lines, start=7):
        fields = line.split()
        fields[1] = str(offset)
        fields[-1] = "2"
        model_two.append(" ".join(fields))
    inputs["raw_path"].write_text(nmr_text.rstrip().removesuffix("#").rstrip() + "\n" + "\n".join(model_two) + "\n#\n")
    protocol_path = run_audit(
        audit_mode="raw-pilot",
        raw_dir=inputs["raw_dir"],
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "nmr_pilot",
        max_source_files=1,
    )
    provenance = [
        json.loads(line) for line in (protocol_path.parent / "raw_sequence_provenance.jsonl").read_text().splitlines()
    ]
    assert {row["model_id"] for row in provenance} == {"1", "2"}
    assert all(row["is_nmr"] for row in provenance)
    nmr_summary = pd.read_csv(protocol_path.parent / "nmr_source_chain_consistency.csv")
    assert len(nmr_summary) == 1
    assert nmr_summary.iloc[0]["model_count"] == 2
    assert bool(nmr_summary.iloc[0]["sequence_consistent_across_models"])


def test_pilot_sampling_is_deterministic_and_source_unique() -> None:
    from scripts.audit_sequence_data_readiness import _pilot_selection

    features = {
        "a.cif": {"split=train", "method=xray"},
        "b.cif": {"split=train", "method=nmr"},
        "c.cif": {"split=validation", "method=xray"},
    }
    first = _pilot_selection(features, max_source_files=2, samples_per_stratum=2, seed=11)
    second = _pilot_selection(features, max_source_files=2, samples_per_stratum=2, seed=11)
    assert first == second
    assert len(first[0]) == len(set(first[0])) == 2
    assert first[1]["method=nmr"] == 2
    assert first[2]["method=nmr"] == 1


def test_raw_full_interruption_resume_and_changed_hash(tmp_path: Path, monkeypatch) -> None:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path)
    second_raw = inputs["raw_dir"] / "second.cif"
    shutil.copyfile(FIXTURES / "two_residue_xray.cif", second_raw)
    calls = []
    original = audit.inspect_mmcif

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(audit, "inspect_mmcif", counted)
    kwargs = {
        "audit_mode": "raw-full",
        "raw_dir": inputs["raw_dir"],
        "processed_manifest": inputs["processed_manifest"],
        "train_manifest": inputs["train_manifest"],
        "validation_manifest": inputs["validation_manifest"],
        "output_dir": tmp_path / "raw_full",
        "checkpoint_frequency": 1,
    }
    with pytest.raises(audit.AuditInterrupted):
        audit.run_audit(**kwargs, stop_after_source_files=1)
    partial = json.loads((kwargs["output_dir"] / "sequence_readiness_protocol.partial.json").read_text())
    assert partial["completed_source_count"] == 1
    assert partial["pending_source_count"] == 1

    protocol_path = audit.run_audit(**kwargs, resume=True)
    assert len(calls) == 2
    assert len(list((protocol_path.parent / "partitions").glob("*.json"))) == 2
    protocol = json.loads(protocol_path.read_text())
    assert protocol["completed_source_count"] == 2
    assert protocol["pending_source_count"] == 0

    inputs["raw_path"].write_text(inputs["raw_path"].read_text() + "\n")
    audit.run_audit(**kwargs, resume=True)
    assert calls.count(str(inputs["raw_path"])) == 2
    with sqlite3.connect(kwargs["output_dir"] / "audit_state.sqlite") as connection:
        versions = connection.execute(
            "SELECT COUNT(*) FROM source_state WHERE source_file=?", (str(inputs["raw_path"]),)
        ).fetchone()[0]
    assert versions == 2


def test_failed_preflight_preserves_existing_output(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("unchanged")
    digest = sha256_file(sentinel)
    with pytest.raises(FileNotFoundError):
        run_audit(
            audit_mode="raw-pilot",
            raw_dir=tmp_path / "missing_raw",
            processed_manifest=inputs["processed_manifest"],
            train_manifest=inputs["train_manifest"],
            validation_manifest=inputs["validation_manifest"],
            output_dir=output,
            restart=True,
            max_source_files=1,
        )
    assert sha256_file(sentinel) == digest


def test_source_identity_metadata_status_is_orthogonal(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import _source_identity_status

    source = tmp_path / "source.cif"
    source.write_text("source")
    state_path = tmp_path / "state.sqlite"
    stat = source.stat()
    with sqlite3.connect(state_path) as connection:
        connection.execute("CREATE TABLE source_files (source_path TEXT, source_size INTEGER, source_mtime_ns INTEGER)")
        connection.execute(
            "INSERT INTO source_files VALUES (?,?,?)",
            (str(source), stat.st_size, stat.st_mtime_ns),
        )
    assert _source_identity_status(source, [state_path]) == "state_size_mtime_match_sha_unavailable"
    source.write_text("changed")
    assert _source_identity_status(source, [state_path]) == "state_metadata_mismatch"
    assert _source_identity_status(source, []) == "no_state_evidence"

    hashed_state = tmp_path / "hashed_state.sqlite"
    current = source.stat()
    with sqlite3.connect(hashed_state) as connection:
        connection.execute(
            """CREATE TABLE source_files (
            source_path TEXT, source_size INTEGER, source_mtime_ns INTEGER, source_sha256 TEXT)"""
        )
        connection.execute(
            "INSERT INTO source_files VALUES (?,?,?,?)",
            (str(source), current.st_size, current.st_mtime_ns, sha256_file(source)),
        )
    assert _source_identity_status(source, [hashed_state]) == "historical_sha_verified"


def test_source_partition_prefers_requested_author_chain_and_retains_label_collision(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.audit_sequence_data_readiness as audit

    source = tmp_path / "source.cif"
    source.write_text("source")
    matrix = {
        "sample_id": "test_A",
        "pdb_id": "TEST",
        "chain_id": "A",
        "model_id": "1",
        "matrix_path": str(tmp_path / "sample.npz"),
        "sequence": "ACG",
        "recorded_length": 3,
        "experimental_method": "X-RAY DIFFRACTION",
        "terminal_trimming_applied": 0,
        "missing_calpha_policy": "reject",
        "split": "train",
    }
    inspected = {
        "npz_sequence": "ACG",
        "residue_ids": ["10", "11", "12"],
        "distance_matrix": np.asarray([[0, 3.8, 7.6], [3.8, 0, 3.8], [7.6, 3.8, 0]], dtype=np.float32),
        "matrix_rows": 3,
        "retained_insertion_codes": ["", "", ""],
        "selected_altlocs": None,
        "retained_start_label_seq_id": None,
        "retained_end_label_seq_id": None,
        "trimmed_n_terminal_residues": None,
        "trimmed_c_terminal_residues": None,
    }
    correct = {
        "pdb_id": "TEST",
        "model_id": "1",
        "label_asym_id": "X",
        "auth_asym_id": "A",
        "entity_id": "1",
        "polymer_type": "polypeptide(L)",
        "residue_tokens": json.dumps(list("ACG")),
        "auth_sequence_ids": json.dumps(["10", "11", "12"]),
        "label_sequence_ids": json.dumps(["1", "2", "3"]),
        "insertion_codes": json.dumps(["", "", ""]),
        "selected_calpha_coordinates": json.dumps([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]]),
        "atom_sequence": "ACG",
        "seqres_sequence": "ACG",
    }
    collision = {
        **correct,
        "label_asym_id": "A",
        "auth_asym_id": "B",
        "entity_id": "2",
        "residue_tokens": json.dumps(list("GGG")),
        "atom_sequence": "GGG",
        "seqres_sequence": "GGG",
    }
    monkeypatch.setattr(audit, "_matrix_rows_for_source", lambda _connection, _source: [matrix])
    monkeypatch.setattr(audit, "inspect_processed_npz", lambda _path, include_geometry: inspected)
    partition = audit._source_partition(sqlite3.connect(":memory:"), str(source), [collision, correct], Counter())
    alignment = partition["alignments"][0]
    assert alignment["raw_match_strategy"] == "auth_asym_id_first"
    assert alignment["raw_label_asym_id"] == "X"
    assert alignment["refined_primary_classification"] == "verified_sequence_geometry_pair"
    assert {row["label_asym_id"] for row in partition["candidate_evidence"]} == {"A", "X"}


def test_v2_v3_transition_requires_and_accounts_for_197_cases(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import _write_v2_v3_transition

    prior = tmp_path / "forensics"
    prior.mkdir()
    old = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(197)],
            "primary_classification": ["raw_source_version_unverifiable"] * 197,
        }
    )
    old.to_parquet(prior / "per_failure_classification.parquet", index=False)
    current = [
        {
            "sample_id": f"sample-{index}",
            "refined_primary_classification": "unique_sequence_pair_coordinate_unavailable",
            "strict_provenance_status": "coordinate_or_residue_provenance_incomplete",
            "training_eligibility": "conditionally_verified_pair",
            "source_identity_status": "state_size_mtime_match_sha_unavailable",
            "coordinate_matrix_status": "unavailable",
        }
        for index in range(197)
    ]
    counts = _write_v2_v3_transition(prior, current, tmp_path / "transition.csv")
    assert counts == {"unique_sequence_pair_coordinate_unavailable": 197}
    transition = pd.read_csv(tmp_path / "transition.csv")
    assert len(transition) == 197
    assert {
        "v2_classification",
        "forensic_root_cause",
        "v3_pairing_classification",
        "practical_training_eligibility",
    } <= set(transition.columns)


def test_current_pairing_accepts_empirically_reconciled_zero_based_ids() -> None:
    from scripts.audit_sequence_data_readiness import _current_pairing_classification

    assert (
        _current_pairing_classification(
            {
                "training_eligibility": "conditionally_verified_pair",
                "strict_provenance_status": "residue_identity_provenance_incomplete",
                "zero_based_positions_match": True,
                "coordinate_matrix_status": "passed",
            }
        )
        == "verified_sequence_geometry_pair"
    )


def test_summary_only_regeneration_accounts_for_validated_v3_without_scientific_reads(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output = tmp_path / "pilot_v3"
    output.mkdir()
    prior = tmp_path / "forensics"
    prior.mkdir()
    special = ["1ej7_L", "3b5k_A", "3b5k_B", "5oty_A"]
    eligible_ids = [*special, *[f"eligible-{index}" for index in range(182)]]
    unresolved_ids = ["7ycs_D", *[f"unresolved-{index}" for index in range(10)]]
    remaining_ids = [f"remaining-{index}" for index in range(489)]
    sample_ids = [*eligible_ids, *unresolved_ids, *remaining_ids]
    rows = []
    for index, sample_id in enumerate(sample_ids):
        is_unresolved = sample_id in unresolved_ids
        is_special = sample_id in special
        rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": sample_id.split("_")[0].upper(),
                "chain_id": sample_id.split("_")[-1],
                "model_id": "1",
                "matrix_path": f"unused/{sample_id}.npz",
                "experimental_method": "X-RAY DIFFRACTION" if index % 2 else "SOLUTION NMR",
                "split": ("train", "validation", "excluded")[index % 3],
                "recorded_length": 64 + index % 5,
                "matrix_actual_length": 64 + index % 5,
                "refined_primary_classification": (
                    "unresolved_ambiguity" if is_unresolved or is_special else "verified_sequence_geometry_pair"
                ),
                "strict_provenance_status": (
                    "contradictory_or_unresolved"
                    if is_unresolved
                    else "residue_identity_provenance_incomplete"
                    if is_special
                    else "historical_source_sha_unavailable"
                ),
                "training_eligibility": ("excluded_or_unresolved" if is_unresolved else "conditionally_verified_pair"),
                "source_identity_status": "state_size_mtime_match_sha_unavailable",
                "manifest_npz_sequence_match": True,
                "unique_author_linked_candidate": True,
                "author_linked_candidate_count": 1,
                "matrix_sequence_occurrence_count_in_candidate": 1,
                "retained_interval_match": not is_unresolved,
                "auth_residue_ids_match": not (is_unresolved or is_special),
                "label_residue_ids_match": False,
                "zero_based_positions_match": False,
                "one_based_positions_match": False,
                "insertion_codes_match": True,
                "raw_calpha_available_for_selected_interval": not is_unresolved,
                "coordinate_matrix_match": not is_unresolved,
                "coordinate_matrix_status": "unavailable" if is_unresolved else "passed",
                "competing_candidate_count": 2 if is_unresolved else 1,
            }
        )
    (output / "seqres_atom_matrix_alignments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    pd.DataFrame({"candidate": range(3260)}).to_parquet(output / "candidate_resolution_evidence.parquet", index=False)
    for name in ("raw_sequence_provenance.jsonl", "raw_residue_tokens.jsonl", "nmr_model_consistency.jsonl"):
        (output / name).write_text("\n")
    with sqlite3.connect(output / "audit_state.sqlite"):
        pass
    pd.DataFrame(
        {
            "sample_id": sample_ids[:197],
            "primary_classification": ["raw_source_version_unverifiable"] * 197,
        }
    ).to_parquet(prior / "per_failure_classification.parquet", index=False)
    (output / "run_config.json").write_text(json.dumps({"prior_forensics_dir": str(prior)}))
    (output / "sequence_readiness_protocol.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "audit_mode": "raw-pilot",
                "scientific_reporting_counts": {},
            }
        )
    )
    (output / "acceptance_criteria_results.json").write_text(
        json.dumps(
            {
                "audit_mode": "raw-pilot",
                "criteria": [
                    {
                        "criterion": "practical_sequence_geometry_pairing",
                        "status": "pilot_failed",
                        "evidence_count": 675,
                        "blocking": True,
                        "explanation": "old",
                    }
                ],
            }
        )
    )
    (output / "raw_pilot_summary.json").write_text(
        json.dumps({"alignment_class_counts": {"valid_residue_id_selection": 28}})
    )
    scientific = [
        output / "seqres_atom_matrix_alignments.jsonl",
        output / "candidate_resolution_evidence.parquet",
        output / "raw_sequence_provenance.jsonl",
        output / "raw_residue_tokens.jsonl",
        output / "nmr_model_consistency.jsonl",
        prior / "per_failure_classification.parquet",
    ]
    before = {path: sha256_file(path) for path in scientific}
    monkeypatch.setattr(audit, "inspect_mmcif", lambda *_args, **_kwargs: pytest.fail("raw parser called"))
    monkeypatch.setattr(audit, "inspect_processed_npz", lambda *_args, **_kwargs: pytest.fail("NPZ read"))
    monkeypatch.setattr(audit, "_ingest_manifest", lambda *_args, **_kwargs: pytest.fail("manifest reindexed"))

    protocol_path = audit.regenerate_summary_reports(output)

    assert {path: sha256_file(path) for path in scientific} == before
    practical = pd.read_csv(output / "practical_training_eligibility_summary.csv")
    for dimension in practical["summary_dimension"].unique():
        assert practical.loc[practical["summary_dimension"] == dimension, "pair_count"].sum() == 686
    strict = pd.read_csv(output / "strict_training_eligibility_summary.csv")
    for dimension in strict["summary_dimension"].unique():
        assert strict.loc[strict["summary_dimension"] == dimension, "pair_count"].sum() == 686
    assert len(pd.read_csv(output / "practical_training_eligibility.csv")) == 675
    unresolved = pd.read_csv(output / "unresolved_case_summary.csv")
    assert len(unresolved) == 11
    assert "7ycs_D" in set(unresolved["sample_id"])
    transition = pd.read_csv(output / "v2_forensic_v3_transition.csv")
    assert len(transition) == transition["sample_id"].nunique() == 197
    assert set(transition.loc[transition["sample_id"].isin(special), "v3_pairing_classification"]) == {
        "residue_identity_provenance_incomplete"
    }
    decision = json.loads((output / "sequence_geometry_dataset_decision.json").read_text())
    assert decision["unfiltered_cohort_acceptance"] == "failed"
    assert decision["filtered_dataset_usable"] == "passed"
    assert decision["eligible_pair_count"] == 675
    assert decision["genuine_sequence_matrix_mismatch_count"] == 0
    acceptance = json.loads((output / "acceptance_criteria_results.json").read_text())
    assert acceptance["unfiltered_cohort_acceptance"] == "failed"
    assert acceptance["filtered_dataset_usable"] == "passed"
    assert "675/686" in next(
        item["explanation"] for item in acceptance["criteria"] if item["criterion"] == "filtered_dataset_usable"
    )
    protocol = json.loads(protocol_path.read_text())
    assert protocol["summary_only_regeneration"]["mmcif_parser_calls"] == 0
    assert protocol["summary_only_regeneration"]["npz_reads"] == 0
    assert protocol["summary_only_regeneration"]["manifest_rows_reindexed"] == 0


def test_synthetic_sequence_readiness_audit_is_read_only(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    declared_sequence = """loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_ins_code
_pdbx_poly_seq_scheme.mon_id
A A 1 1 ? GLY
A A 2 2 ? ALA
A A 3 3 ? THR
#
"""
    raw_text = inputs["raw_path"].read_text()
    inputs["raw_path"].write_text(raw_text.replace("loop_\n_atom_site.", declared_sequence + "loop_\n_atom_site."))
    input_paths = [
        inputs["raw_path"],
        inputs["sample_path"],
        inputs["processed_manifest"],
        inputs["train_manifest"],
        inputs["validation_manifest"],
    ]
    before = {path: sha256_file(path) for path in input_paths}

    protocol_path = run_audit(
        audit_mode="raw-full",
        raw_dir=inputs["raw_dir"],
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=tmp_path / "audit",
    )
    protocol = json.loads(protocol_path.read_text())
    assert protocol["raw_inputs_unchanged"] is True
    alignment_path = protocol_path.parent / "seqres_atom_matrix_alignments.jsonl"
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    assert alignments[0]["atom_sequence"] == "GA"
    assert alignments[0]["matrix_sequence"] == "GA"
    assert alignments[0]["seqres_sequence"] == "GAT"
    assert len({alignments[0][key] for key in ("seqres_sequence", "atom_sequence", "matrix_sequence")}) == 2
    assert (protocol_path.parent / "PROPOSED_SCHEMA_AND_MIGRATION.md").exists()
    assert {path: sha256_file(path) for path in input_paths} == before


def test_compact_raw_audit_separates_state_and_artifacts(tmp_path: Path) -> None:
    from protein_distance_diffusion.evaluation.sequence_readiness_storage import (
        SequenceReadinessArtifactReader,
    )
    from scripts.audit_sequence_data_readiness import run_audit

    inputs = _audit_inputs(tmp_path)
    output = tmp_path / "published"
    state = tmp_path / "state"
    protocol_path = run_audit(
        audit_mode="raw-pilot",
        raw_dir=inputs["raw_dir"],
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=output,
        state_dir=state,
        storage_profile="compact-v1",
        sources_per_partition=1,
        max_source_files=1,
    )

    protocol = json.loads(protocol_path.read_text())
    assert protocol["report_semantics_version"] == 4
    assert json.loads((output / "run_config.json").read_text())["report_semantics_version"] == 4
    assert protocol["storage_profile"] == "compact-v1"
    assert protocol["resolved_output_dir"] == str(output.resolve())
    assert protocol["resolved_state_dir"] == str(state.resolve())
    assert (state / "audit_state.sqlite").is_file()
    assert set(path.name for path in state.iterdir()) <= {
        "audit_state.sqlite",
        "audit_state.sqlite-shm",
        "audit_state.sqlite-wal",
    }
    assert not (output / "audit_state.sqlite").exists()
    assert not (output / "partitions").exists()
    assert not (output / "seqres_atom_matrix_alignments.jsonl").exists()
    reader = SequenceReadinessArtifactReader(output)
    assert len(list(reader.iter_records("matrix_pair_alignments"))) == 1
    projection = json.loads((output / "storage_projection.json").read_text())
    assert projection["pilot_source_count"] == 1


def _compact_raw_full_attestation_fixture(tmp_path: Path) -> tuple[Path, dict[str, int]]:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path)
    output = tmp_path / "compact-raw-full"
    state = tmp_path / "compact-raw-full-state"
    protocol_path = audit.run_audit(
        audit_mode="raw-pilot",
        raw_dir=inputs["raw_dir"],
        processed_manifest=inputs["processed_manifest"],
        train_manifest=inputs["train_manifest"],
        validation_manifest=inputs["validation_manifest"],
        output_dir=output,
        state_dir=state,
        storage_profile="compact-v1",
        sources_per_partition=1,
        max_source_files=1,
    )
    protocol = json.loads(protocol_path.read_text())
    protocol["audit_mode"] = "raw-full"
    protocol.pop("report_semantics_version")
    protocol_path.write_text(json.dumps(protocol))
    run_config_path = output / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["audit_mode"] = "raw-full"
    run_config.pop("report_semantics_version")
    run_config_path.write_text(json.dumps(run_config))
    expected = {
        "selected_source_count": 1,
        "matrix_pair_count": 1,
        "practical_eligible_count": 1,
        "unresolved_count": 0,
    }
    return output, expected


def test_compact_raw_full_summary_only_attests_v4_without_scientific_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output, expected = _compact_raw_full_attestation_fixture(tmp_path)
    scientific = sorted((output / "tables").glob("*/part-*.parquet"))
    before = {path: sha256_file(path) for path in scientific}
    prior_protocol_sha256 = sha256_file(output / "sequence_readiness_protocol.json")
    monkeypatch.setattr(audit, "inspect_mmcif", lambda *_args, **_kwargs: pytest.fail("raw parser called"))
    monkeypatch.setattr(audit, "inspect_processed_npz", lambda *_args, **_kwargs: pytest.fail("NPZ read"))
    monkeypatch.setattr(audit, "_ingest_manifest", lambda *_args, **_kwargs: pytest.fail("manifest reindexed"))
    monkeypatch.setattr(audit, "RAW_FULL_ATTESTATION_COUNTS", expected)

    protocol_path = audit.regenerate_summary_reports(output)

    protocol = json.loads(protocol_path.read_text())
    regeneration = protocol["summary_only_regeneration"]
    assert protocol["report_semantics_version"] == 4
    assert regeneration["prior_protocol_sha256"] == prior_protocol_sha256
    assert regeneration["verified_count_contracts"] == expected
    assert regeneration["mmcif_parser_calls"] == 0
    assert regeneration["npz_reads"] == 0
    assert regeneration["manifest_rows_reindexed"] == 0
    assert regeneration["scientific_partition_rewrites"] == 0
    assert regeneration["scientific_partition_hashes_preserved"] is True
    assert {path: sha256_file(path) for path in scientific} == before
    assert json.loads((output / "run_config.json").read_text())["report_semantics_version"] == 4
    assert list(pd.read_csv(output / "practical_training_eligibility.csv")) == [
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


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_compact_raw_full_attestation_rejects_missing_or_corrupt_partition(tmp_path: Path, failure: str) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output, expected = _compact_raw_full_attestation_fixture(tmp_path)
    protocol_path = output / "sequence_readiness_protocol.json"
    prior_protocol = protocol_path.read_bytes()
    partition = sorted((output / "tables").glob("*/part-*.parquet"))[0]
    if failure == "missing":
        partition.unlink()
    else:
        partition.write_bytes(partition.read_bytes() + b"corrupt")

    with pytest.raises((FileNotFoundError, ValueError), match="[Cc]ompact|[Cc]orrupt|[Mm]issing"):
        audit.attest_compact_raw_full_semantics(output, expected_counts=expected)

    assert protocol_path.read_bytes() == prior_protocol


def test_compact_raw_full_attestation_rejects_wrong_counts_before_metadata_write(tmp_path: Path) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output, expected = _compact_raw_full_attestation_fixture(tmp_path)
    protocol_path = output / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["completed_source_count"] = 2
    protocol_path.write_text(json.dumps(protocol))
    prior_protocol = protocol_path.read_bytes()

    with pytest.raises(ValueError, match="source contract failed"):
        audit.attest_compact_raw_full_semantics(output, expected_counts=expected)

    assert protocol_path.read_bytes() == prior_protocol


def test_compact_raw_pilot_cannot_receive_raw_full_attestation(tmp_path: Path) -> None:
    import scripts.audit_sequence_data_readiness as audit

    output, expected = _compact_raw_full_attestation_fixture(tmp_path)
    protocol_path = output / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["audit_mode"] = "raw-pilot"
    protocol_path.write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="completed raw-full protocol"):
        audit.attest_compact_raw_full_semantics(output, expected_counts=expected)


def test_compact_raw_audit_resumes_without_reparsing_committed_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.audit_sequence_data_readiness as audit

    inputs = _audit_inputs(tmp_path)
    second_raw = inputs["raw_dir"] / "second.cif"
    shutil.copyfile(FIXTURES / "two_residue_xray.cif", second_raw)
    calls = []
    original = audit.inspect_mmcif

    def counted(path):
        calls.append(str(path))
        return original(path)

    monkeypatch.setattr(audit, "inspect_mmcif", counted)
    kwargs = {
        "audit_mode": "raw-full",
        "raw_dir": inputs["raw_dir"],
        "processed_manifest": inputs["processed_manifest"],
        "train_manifest": inputs["train_manifest"],
        "validation_manifest": inputs["validation_manifest"],
        "output_dir": tmp_path / "compact-output",
        "state_dir": tmp_path / "compact-state",
        "storage_profile": "compact-v1",
        "sources_per_partition": 1,
        "unsafe_skip_disk_check": True,
    }
    projection = tmp_path / "projection.json"
    projection.write_text(json.dumps({"disk_guard_passed": True, "projected_full_corpus_bytes": 1}))
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text(json.dumps({"status": "passed"}))
    kwargs.update(storage_projection_path=projection, compact_equivalence_report=equivalence)

    with pytest.raises(audit.AuditInterrupted):
        audit.run_audit(**kwargs, stop_after_source_files=1)
    assert len(calls) == 1
    assert not (kwargs["output_dir"] / "sequence_readiness_protocol.json").exists()

    protocol_path = audit.run_audit(**kwargs, resume=True)

    assert len(calls) == 2
    assert json.loads(protocol_path.read_text())["status"] == "completed"
    assert len(list((kwargs["output_dir"] / "partition_commits").glob("*.json"))) == 2
