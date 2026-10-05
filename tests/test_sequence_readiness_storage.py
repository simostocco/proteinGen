"""Regression tests for compact sequence-readiness storage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from protein_distance_diffusion.evaluation.sequence_readiness_storage import (
    SequenceReadinessArtifactReader,
    compare_sequence_readiness_artifacts,
    storage_projection,
    validate_compact_partition,
    write_compact_partition,
)


def _source(index: int) -> dict[str, object]:
    source = f"/raw/{index:04d}.cif.gz"
    return {
        "source_file": source,
        "source_sha256": f"{index:064x}",
        "status": "completed",
        "payload": {
            "raw_rows": [
                {
                    "source_file": source,
                    "pdb_id": f"P{index}",
                    "model_id": "1",
                    "label_asym_id": "A",
                    "modified_residue_counts": "{}",
                    "missing_residue_count": 0,
                    "missing_calpha_count": 0,
                }
            ],
            "alignments": [
                {
                    "source_file": source,
                    "sample_id": f"P{index}_A",
                    "matrix_sequence": "ACG",
                    "training_eligibility": "conditionally_verified_pair",
                }
            ],
            "candidate_evidence": [
                {
                    "source_file": source,
                    "sample_id": f"P{index}_A",
                    "physical_candidate_id": f"candidate-{index}",
                    "model_number": "1",
                    "entity_id": "1",
                    "label_asym_id": "A",
                    "auth_asym_id": "A",
                    "author_chain_match": True,
                    "label_chain_match": True,
                    "coordinate_max_abs_error_angstrom": 0.00001,
                }
            ],
            "tokens": [],
            "nonpolymer_context": [],
            "nmr_summaries": [],
        },
    }


def test_compact_partitions_are_typed_compressed_and_not_per_source(tmp_path: Path) -> None:
    sources = [_source(index) for index in range(100)]
    for partition_index, start in enumerate(range(0, len(sources), 25)):
        write_compact_partition(
            tmp_path,
            partition_index=partition_index,
            sources=sources[start : start + 25],
        )

    source_files = sorted((tmp_path / "tables" / "source_identity").glob("*.parquet"))
    assert len(source_files) == 4
    assert len(list(tmp_path.rglob("*.parquet"))) < len(sources)
    parquet = pq.ParquetFile(source_files[0])
    assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"
    assert parquet.schema_arrow.field("parser_calls").type.bit_width == 64
    (tmp_path / "run_config.json").write_text(json.dumps({"storage_profile": "compact-v1"}))
    reader = SequenceReadinessArtifactReader(tmp_path)
    assert len(list(reader.iter_records("matrix_pair_alignments"))) == 100


def test_compact_partition_validation_detects_corruption(tmp_path: Path) -> None:
    marker = write_compact_partition(tmp_path, partition_index=0, sources=[_source(0)])
    validate_compact_partition(tmp_path, marker)
    path = tmp_path / marker["tables"]["matrix_pair_alignments"]["path"]
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="Corrupt compact partition"):
        validate_compact_partition(tmp_path, marker)


def test_finalized_partition_reconciles_after_sqlite_commit_interruption(tmp_path: Path) -> None:
    import scripts.audit_sequence_data_readiness as audit

    marker = write_compact_partition(tmp_path, partition_index=0, sources=[_source(0)])
    orphan = tmp_path / "tables" / "matrix_pair_alignments" / "part-000001.parquet"
    orphan.write_bytes((tmp_path / marker["tables"]["matrix_pair_alignments"]["path"]).read_bytes())
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    connection = audit._connect_state(state_dir / "audit_state.sqlite")
    try:
        assert connection.execute("SELECT COUNT(*) FROM compact_partitions").fetchone()[0] == 0
        audit._reconcile_compact_partitions(connection, tmp_path)
        assert not orphan.exists()
        assert json.loads(
            connection.execute("SELECT value FROM metadata WHERE key='reconciled_partial_partitions'").fetchone()[0]
        ) == ["part-000001"]
        assert connection.execute("SELECT COUNT(*) FROM compact_partitions").fetchone()[0] == 1
        assert connection.execute("SELECT status FROM source_state").fetchone()[0] == "completed"
        audit._reconcile_compact_partitions(connection, tmp_path)
        assert connection.execute("SELECT COUNT(*) FROM compact_partitions").fetchone()[0] == 1
    finally:
        connection.close()
    assert Path(marker["marker_path"]).is_file()


def test_storage_projection_has_required_guard_fields(tmp_path: Path) -> None:
    write_compact_partition(tmp_path, partition_index=0, sources=[_source(0), _source(1)])
    projection = storage_projection(tmp_path, pilot_source_count=2, full_source_count=10, sources_per_partition=4)
    assert projection["expected_partition_count"] == 3
    assert projection["required_free_bytes"] == (
        2 * projection["projected_remaining_bytes"] + projection["required_safety_reserve_bytes"]
    )


def test_disk_guard_blocks_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.audit_sequence_data_readiness as audit

    class TinyDisk:
        f_bavail = 1
        f_frsize = 1

    monkeypatch.setattr(audit.os, "statvfs", lambda _path: TinyDisk())
    with pytest.raises(RuntimeError, match="disk guard failed"):
        audit._disk_guard(tmp_path, projected_remaining_bytes=10, unsafe_skip_disk_check=False)
    result = audit._disk_guard(tmp_path, projected_remaining_bytes=10, unsafe_skip_disk_check=True)
    assert result == {
        "free_bytes": 1,
        "projected_remaining_bytes": 10,
        "required_free_bytes": 2 * 10 + 20 * 1024**3,
        "required_safety_reserve_bytes": 20 * 1024**3,
        "passed": False,
        "unsafe_override": True,
    }


def test_logical_normalization_categories_remain_scientifically_strict() -> None:
    import protein_distance_diffusion.evaluation.sequence_readiness_storage as storage

    normalized = storage.SequenceReadinessArtifactReader._logical_record(
        "residue_id_convention_evidence",
        {
            "source_file": "/raw/test.cif",
            "sample_id": "sample",
            "model_number": 1,
            "entity_id": 2,
            "label_asym_id": "A",
            "auth_asym_id": "B",
            "candidate_index": 3,
            "convention": "auth_seq_id_insertion",
            "author_chain_match": "False",
        },
    )
    assert normalized["model_number"] == "1"
    assert normalized["entity_id"] == "2"
    assert normalized["author_chain_match"] is False
    assert normalized["residue_id_convention"] == "auth_seq_id_insertion"
    assert normalized["evidence_row_index"] == "3"
    assert len(normalized["physical_candidate_id"]) == 64

    assert storage._classify_field_difference("optional", None, "") == ("null_normalization", True)
    assert storage._classify_field_difference("tokens", '["A","C"]', ("A", "C")) == (
        "type_normalization",
        True,
    )
    assert storage._classify_field_difference("residue_ids", '["1","2"]', '["2","1"]') == (
        "list_order_normalization",
        False,
    )
    assert storage._classify_field_difference("coordinate_rmse_angstrom", 1.0, 1.00005) == (
        "numeric_tolerance",
        True,
    )
    assert storage._classify_field_difference("sequence", "ACG", "AGC") == (
        "genuinely_different_value",
        False,
    )


def test_summary_reader_normalizes_filename_and_column_aliases(tmp_path: Path) -> None:
    legacy = pd.DataFrame(
        [
            {
                "sample_id": "sample-A",
                "v3_pairing_classification": "unresolved_ambiguity",
                "refined_primary_classification": "historical_missing_calpha_selection_unreproducible",
                "practical_training_eligibility": "excluded_or_unresolved",
                "model_id": "1",
            }
        ]
    )
    compact = legacy.rename(
        columns={
            "practical_training_eligibility": "training_eligibility",
            "model_id": "model_number",
        }
    )
    legacy.to_csv(tmp_path / "unresolved_case_summary.csv", index=False)
    compact.to_csv(tmp_path / "unresolved_cases.csv", index=False)

    frame = SequenceReadinessArtifactReader(tmp_path).frame("unresolved_eligibility")

    assert frame.to_dict("records") == legacy.to_dict("records")


def test_reader_does_not_alias_forensic_classification_to_pairing_classification() -> None:
    row = SequenceReadinessArtifactReader._logical_record(
        "matrix_pair_alignments",
        {
            "sample_id": "10af_A",
            "refined_primary_classification": "historical_missing_calpha_selection_unreproducible",
        },
    )

    assert row["refined_primary_classification"] == ("historical_missing_calpha_selection_unreproducible")
    assert "v3_pairing_classification" not in row


def test_summary_reader_preserves_columns_for_empty_csv(tmp_path: Path) -> None:
    pd.DataFrame(columns=["sample_id", "training_eligibility", "model_number"]).to_csv(
        tmp_path / "unresolved_cases.csv", index=False
    )

    frame = SequenceReadinessArtifactReader(tmp_path).frame("unresolved_eligibility")

    assert frame.empty
    assert list(frame) == ["sample_id", "practical_training_eligibility", "model_id"]


def test_compact_equivalence_enforces_validated_pilot_contract(tmp_path: Path) -> None:
    reference = tmp_path / "verbose"
    compact = tmp_path / "compact"
    reference.mkdir()
    alignments = []
    classes = [
        (635, "historical_source_sha_verified", "passed", "conditionally_verified_pair"),
        (36, "historical_source_sha_verified", "unavailable", "conditionally_verified_pair"),
        (4, "residue_identity_provenance_incomplete", "passed", "conditionally_verified_pair"),
        (11, "contradictory_or_unresolved", "unavailable", "excluded_or_unresolved"),
    ]
    source = "/raw/cohort.cif.gz"
    index = 0
    for count, provenance, coordinate, eligibility in classes:
        for _ in range(count):
            alignments.append(
                {
                    "sample_id": f"sample-{index}",
                    "strict_provenance_status": provenance,
                    "coordinate_matrix_status": coordinate,
                    "training_eligibility": eligibility,
                    "auth_residue_ids_match": provenance != "residue_identity_provenance_incomplete",
                }
            )
            index += 1
    candidates = [
        {
            "source_file": source,
            "sample_id": f"sample-{index % 686}",
            "candidate_index": str(index),
            "physical_candidate_id": f"candidate-{index}",
            "author_chain_match": "True",
            "selected_sequence": "ACG",
        }
        for index in range(3260)
    ]
    (reference / "seqres_atom_matrix_alignments.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in alignments)
    )
    pd.DataFrame(candidates).to_parquet(reference / "candidate_resolution_evidence.parquet", index=False)
    protein_row = {
        "source_file": source,
        "source_sha256": "0" * 64,
        "pdb_id": "TEST",
        "entity_id": "1",
        "label_asym_id": "A",
        "auth_asym_id": "A",
        "model_id": "1",
        "residue_tokens": '["A","C","G"]',
    }
    (reference / "raw_sequence_provenance.jsonl").write_text(json.dumps(protein_row) + "\n")
    for filename in (
        "raw_residue_tokens.jsonl",
        "excluded_nonpolymer_context.jsonl",
        "nmr_source_chain_consistency.jsonl",
    ):
        (reference / filename).write_text("")
    write_compact_partition(
        compact,
        partition_index=0,
        sources=[
            {
                "source_file": source,
                "source_sha256": "0" * 64,
                "status": "completed",
                "payload": {
                    "raw_rows": [protein_row],
                    "alignments": alignments,
                    "candidate_evidence": candidates,
                    "tokens": [],
                    "nonpolymer_context": [],
                    "nmr_summaries": [],
                },
            }
        ],
    )
    (compact / "run_config.json").write_text(json.dumps({"storage_profile": "compact-v1"}))

    report = compare_sequence_readiness_artifacts(
        reference,
        compact,
        report_path=compact / "compact_equivalence_report.json",
    )

    assert report["status"] == "passed"
    assert report["observed_contracts"]["matrix_pair_count"] == 686
    assert report["observed_contracts"]["pairing_eligible_count"] == 675
    assert report["table_results"]["protein_chain_models"]["equal"] is True
    candidate_result = report["table_results"]["residue_id_convention_evidence"]
    assert candidate_result["equal"] is True
    assert candidate_result["primary_key"] == [
        "sample_id",
        "physical_candidate_id",
        "residue_id_convention",
        "evidence_row_index",
    ]
    assert candidate_result["reference_duplicate_primary_key_count"] == 0

    candidate_path = compact / "tables" / "residue_id_convention_evidence" / "part-000000.parquet"
    changed = pd.read_parquet(candidate_path)
    changed.loc[0, "selected_sequence"] = "GCA"
    changed.to_parquet(candidate_path, index=False)
    failed = compare_sequence_readiness_artifacts(
        reference,
        compact,
        report_path=compact / "scientific_difference.json",
    )
    mismatch = failed["table_results"]["residue_id_convention_evidence"]
    assert failed["status"] == "failed"
    assert mismatch["per_column_mismatch_counts"] == {"selected_sequence": 1}
    assert any(
        item["category"] == "genuinely_different_value" and item["field"] == "selected_sequence"
        for item in mismatch["representative_differences"]
    )
    assert len(mismatch["representative_differences"]) <= 20
