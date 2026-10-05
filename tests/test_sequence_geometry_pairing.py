from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.data.pairing_builder import (
    FORENSIC_PAIRING_COMPATIBILITY,
    _membership_counts,
    _pairing_classification,
    _validate_forensic_pairing_relationship,
    build_sequence_geometry_pairing,
    normalize_evidence_boolean,
    validate_sequence_geometry_pairing,
)
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    AdditiveSymmetricNoise,
    ContactDeletion,
    LongRangePairMask,
    LowRankDistanceDistortion,
    SequenceGeometryDataset,
    SequenceGeometryVocabulary,
    build_geometry_corruption,
    collate_sequence_geometry,
)
from protein_distance_diffusion.evaluation.sequence_readiness import sequence_sha256, sha256_file


def _write_npz(path: Path, sequence: str) -> None:
    positions = np.arange(len(sequence), dtype=np.float32)[:, None] * np.array([[3.8, 0.0, 0.0]])
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1).astype(np.float32)
    np.savez_compressed(
        path,
        sequence=np.asarray(sequence),
        distance_matrix=distances,
        residue_ids=np.arange(1, len(sequence) + 1),
        insertion_codes=np.asarray([""] * len(sequence)),
    )


def _fixture(tmp_path: Path, *, audit_mode: str = "raw-pilot") -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    sequences = ("ACDE", "FGHIK", "LMNPQR", "STVWYAC")
    sample_ids = ("1aaa_A", "2bbb_A", "3ccc_A", "4ddd_A")
    processed_rows = []
    for sample_id, sequence in zip(sample_ids, sequences, strict=True):
        matrix_path = tmp_path / f"{sample_id}.npz"
        _write_npz(matrix_path, sequence)
        processed_rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": sample_id[:4].upper(),
                "chain_id": "A",
                "model_number": 1,
                "sequence": sequence,
                "length": len(sequence),
                "original_chain_length": len(sequence),
                "path": str(matrix_path),
                "source_file": str(tmp_path / f"{sample_id[:4]}.cif.gz"),
                "experimental_method": "X-RAY DIFFRACTION",
                "missing_calpha_policy": "reject",
                "trimmed_n_terminal_residues": 0,
                "trimmed_c_terminal_residues": 0,
                "terminal_trimming_applied": False,
                "trimmed_fraction": 0.0,
                "max_terminal_trim_fraction": None,
            }
        )
    processed = pd.DataFrame(processed_rows)
    train_ids = sample_ids[:2]
    validation_ids = sample_ids[2:]

    def split_frame(ids: tuple[str, ...]) -> pd.DataFrame:
        rows = []
        for sample_id in ids:
            row = processed.loc[processed["sample_id"] == sample_id].iloc[0].to_dict()
            row.update(
                {
                    "sequence_hash": sequence_sha256(row["sequence"]),
                    "cluster_id": f"cluster-{sample_id}",
                    "split_group_id": f"group-{sample_id}",
                    "exact_sequence_count": 1,
                    "sample_weight": 1.0,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    paths = {
        "processed": tmp_path / "processed.parquet",
        "train": tmp_path / "train.parquet",
        "validation": tmp_path / "validation.parquet",
        "normalization": tmp_path / "normalization.json",
        "audit": tmp_path / "audit",
    }
    processed.to_parquet(paths["processed"], index=False)
    split_frame(train_ids).to_parquet(paths["train"], index=False)
    split_frame(validation_ids).to_parquet(paths["validation"], index=False)
    paths["normalization"].write_text(json.dumps({"mean": 9.0, "std": 4.0}))
    paths["audit"].mkdir()
    classes = (
        "strict",
        "coordinate_unavailable",
        "residue_incomplete",
        "unresolved",
    )
    alignments = []
    candidates = []
    for row, kind in zip(processed_rows, classes, strict=True):
        unresolved = kind == "unresolved"
        alignments.append(
            {
                "sample_id": row["sample_id"],
                "pdb_id": row["pdb_id"],
                "chain_id": row["chain_id"],
                "model_id": "1",
                "sequence": row["sequence"],
                "matrix_actual_length": row["length"],
                "manifest_npz_sequence_match": True,
                "raw_match_count": 2 if unresolved else 1,
                "training_eligibility": (
                    "excluded_or_unresolved"
                    if unresolved
                    else "strict_verified_pair"
                    if kind == "strict"
                    else "conditionally_verified_pair"
                ),
                "strict_provenance_status": (
                    "contradictory_or_unresolved"
                    if unresolved
                    else "residue_identity_provenance_incomplete"
                    if kind == "residue_incomplete"
                    else "historical_source_sha_verified"
                ),
                "source_identity_status": "historical_sha_verified",
                "auth_residue_ids_match": kind != "residue_incomplete" and not unresolved,
                "label_residue_ids_match": False,
                "zero_based_positions_match": False,
                "one_based_positions_match": False,
                "coordinate_matrix_status": ("unavailable" if kind == "coordinate_unavailable" else "passed"),
                "coordinate_distance_max_abs_error_angstrom": (None if kind == "coordinate_unavailable" else 0.0),
                "unique_author_linked_candidate": not unresolved,
                "author_linked_candidate_count": 0 if unresolved else 1,
                "competing_candidate_count": 2 if unresolved else 1,
                "matrix_sequence_occurrence_count_in_candidate": 1,
                "selected_residue_id_convention": "auth_seq_id_insertion",
                "sequence_source": "ATOM-derived",
                "residue_ids": json.dumps(list(range(1, row["length"] + 1))),
                "insertion_codes": json.dumps([""] * row["length"]),
                "selected_altlocs": json.dumps([None] * row["length"]),
                "trim_metadata_available": True,
                "retained_interval_match": True,
            }
        )
        candidates.append(
            {
                "sample_id": row["sample_id"],
                "candidate_index": 0,
                "convention": "auth_seq_id_insertion",
                "author_chain_match": str(not unresolved),
                "label_chain_match": str(not unresolved),
                "sequence_match": str(not unresolved),
                "strong_evidential_match": str(not unresolved),
                "identity_mapping_available": str(not unresolved),
                "identity_mapping_ambiguous": "False",
                "insertion_codes_match": "True",
                "trim_metadata_available": "True",
                "trim_metadata_compatible": str(not unresolved),
                "label_asym_id": "L",
                "auth_asym_id": "A",
                "entity_id": "1",
                "model_number": "1",
                "source_file": row["source_file"],
            }
        )
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in alignments))
    pd.DataFrame(candidates).to_parquet(paths["audit"] / "candidate_resolution_evidence.parquet", index=False)
    eligibility_rows = [
        {
            "sample_id": row["sample_id"],
            "pdb_id": row["pdb_id"],
            "chain_id": row["chain_id"],
            "model_id": row["model_id"],
            "matrix_path": processed_rows[index]["path"],
            "v3_pairing_classification": (
                "unresolved_ambiguity"
                if row["training_eligibility"] == "excluded_or_unresolved"
                else "residue_identity_provenance_incomplete"
                if row["strict_provenance_status"] == "residue_identity_provenance_incomplete"
                else "unique_sequence_pair_coordinate_unavailable"
                if row["coordinate_matrix_status"] == "unavailable"
                else "verified_sequence_geometry_pair"
            ),
            "strict_provenance_status": row["strict_provenance_status"],
            "practical_training_eligibility": row["training_eligibility"],
            "source_identity_status": row["source_identity_status"],
        }
        for index, row in enumerate(alignments)
    ]
    practical_frame = pd.DataFrame(eligibility_rows[:3])
    practical_frame.to_csv(paths["audit"] / "practical_training_eligibility.csv", index=False)
    practical_frame.iloc[:1].to_csv(paths["audit"] / "strict_training_eligibility.csv", index=False)
    pd.DataFrame(eligibility_rows[3:]).drop(columns="strict_provenance_status").to_csv(
        paths["audit"] / "unresolved_case_summary.csv", index=False
    )
    pd.DataFrame(eligibility_rows).drop(columns="matrix_path").to_csv(
        paths["audit"] / "v2_forensic_v3_transition.csv", index=False
    )
    input_hashes = {str(paths[name]): sha256_file(paths[name]) for name in ("processed", "train", "validation")}
    (paths["audit"] / "sequence_readiness_protocol.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "audit_mode": audit_mode,
                "report_semantics_version": 4,
                "input_hashes": input_hashes,
            }
        )
    )
    return paths


def _build(paths: dict[str, Path], tmp_path: Path, policy: str, **kwargs: object) -> Path:
    output = tmp_path / f"output-{policy}"
    build_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        output_dir=output,
        eligibility_policy=policy,
        allow_pilot_evidence=True,
        **kwargs,
    )
    return output


def _refresh_manifest_hashes(paths: dict[str, Path]) -> None:
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for name in ("processed", "train", "validation"):
        protocol["input_hashes"][str(paths[name])] = sha256_file(paths[name])
    protocol_path.write_text(json.dumps(protocol))


def _convert_audit_to_compact(paths: dict[str, Path]) -> None:
    from protein_distance_diffusion.evaluation.sequence_readiness_storage import write_compact_partition

    audit_dir = paths["audit"]
    alignments = [
        json.loads(line)
        for line in (audit_dir / "seqres_atom_matrix_alignments.jsonl").read_text().splitlines()
        if line.strip()
    ]
    candidates = pd.read_parquet(audit_dir / "candidate_resolution_evidence.parquet").to_dict("records")
    processed = pd.read_parquet(paths["processed"]).set_index("sample_id")
    source_payloads = []
    for sample_id in processed.index:
        source = str(processed.loc[sample_id, "source_file"])
        source_payloads.append(
            {
                "source_file": source,
                "source_sha256": "0" * 64,
                "status": "completed",
                "payload": {
                    "raw_rows": [],
                    "alignments": [row for row in alignments if row["sample_id"] == sample_id],
                    "candidate_evidence": [row for row in candidates if row["sample_id"] == sample_id],
                    "tokens": [],
                    "nonpolymer_context": [],
                    "nmr_summaries": [],
                },
            }
        )
    write_compact_partition(audit_dir, partition_index=0, sources=source_payloads)
    (audit_dir / "run_config.json").write_text(json.dumps({"storage_profile": "compact-v1"}))
    (audit_dir / "seqres_atom_matrix_alignments.jsonl").unlink()
    (audit_dir / "candidate_resolution_evidence.parquet").unlink()
    compact_aliases = {
        "practical_training_eligibility": "training_eligibility",
        "model_id": "model_number",
    }
    for filename in ("practical_training_eligibility.csv", "strict_training_eligibility.csv"):
        path = audit_dir / filename
        pd.read_csv(path).rename(columns=compact_aliases).to_csv(path, index=False)
    unresolved_path = audit_dir / "unresolved_case_summary.csv"
    unresolved = pd.read_csv(unresolved_path).rename(columns=compact_aliases)
    unresolved.to_csv(audit_dir / "unresolved_cases.csv", index=False)
    unresolved_path.unlink()


def _move_sample_to_split(paths: dict[str, Path], sample_id: str, split: str) -> None:
    train = pd.read_parquet(paths["train"])
    validation = pd.read_parquet(paths["validation"])
    rows = pd.concat(
        [train[train["sample_id"] == sample_id], validation[validation["sample_id"] == sample_id]],
        ignore_index=True,
    )
    train = train[train["sample_id"] != sample_id]
    validation = validation[validation["sample_id"] != sample_id]
    if split == "train":
        train = pd.concat([train, rows], ignore_index=True)
    elif split == "validation":
        validation = pd.concat([validation, rows], ignore_index=True)
    elif split != "excluded":
        raise ValueError(split)
    train.to_parquet(paths["train"], index=False)
    validation.to_parquet(paths["validation"], index=False)
    _refresh_manifest_hashes(paths)


@pytest.mark.parametrize(
    ("policy", "eligible_count"),
    (("strict", 1), ("practical", 3), ("all_with_status", 3)),
)
def test_builder_policies_schema_and_unresolved_exclusion(tmp_path: Path, policy: str, eligible_count: int) -> None:
    paths = _fixture(tmp_path)
    output = _build(paths, tmp_path, policy)
    rows = pd.read_parquet(output / "all_pairs.parquet")
    excluded = pd.read_parquet(output / "excluded_pairs.parquet")
    protocol = json.loads((output / "protocol.json").read_text())
    assert set(rows["schema_version"]) == {PAIRING_SCHEMA_VERSION}
    assert {"residue_ids", "insertion_codes", "selected_altlocs", "residue_mask"} <= set(rows)
    assert rows["eligible_for_training"].sum() == eligible_count
    assert "4ddd_A" in set(excluded["sample_id"])
    assert not rows.loc[rows["sample_id"] == "4ddd_A", "eligible_for_training"].item()
    assert protocol["input_hashes_preserved"] is True
    assert (output / "schema.json").is_file()
    assert (output / "vocabulary.json").is_file()


def test_builder_reads_compact_sequence_readiness_artifacts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, audit_mode="raw-full")
    _convert_audit_to_compact(paths)

    output = _build(paths, tmp_path / "compact", "practical")

    assert len(pd.read_parquet(output / "all_pairs.parquet")) == 4
    assert len(pd.read_parquet(output / "eligible_train.parquet")) == 2
    excluded = pd.read_parquet(output / "excluded_pairs.parquet")
    assert excluded.loc[excluded["sample_id"] == "4ddd_A", "pairing_classification"].item() == ("unresolved_ambiguity")


def test_10af_forensic_root_cause_is_compatible_with_current_pairing_classification() -> None:
    alignment = {
        "sample_id": "10af_A",
        "training_eligibility": "conditionally_verified_pair",
        "strict_provenance_status": "coordinate_or_residue_provenance_incomplete",
        "coordinate_matrix_status": "unavailable",
        "refined_primary_classification": "historical_missing_calpha_selection_unreproducible",
    }
    pairing_classification = _pairing_classification(alignment)

    assert pairing_classification == "unique_sequence_pair_coordinate_unavailable"
    _validate_forensic_pairing_relationship(
        alignment["refined_primary_classification"],
        pairing_classification,
        sample_id="10af_A",
    )


def test_legacy_and_streaming_builders_preserve_forensic_classification_domain(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, audit_mode="raw-full")
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    alignments[1]["refined_primary_classification"] = "historical_missing_calpha_selection_unreproducible"
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in alignments))

    legacy_output = _build(paths, tmp_path / "legacy", "practical")
    legacy_row = pd.read_parquet(legacy_output / "all_pairs.parquet").set_index("sample_id").loc["2bbb_A"]
    assert legacy_row["pairing_classification"] == "unique_sequence_pair_coordinate_unavailable"
    assert legacy_row["forensic_root_cause"] == ("historical_missing_calpha_selection_unreproducible")

    _convert_audit_to_compact(paths)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["raw_inputs_unchanged"] = True
    protocol_path.write_text(json.dumps(protocol))
    streaming_output = tmp_path / "streaming-output"
    build_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        output_dir=streaming_output,
        eligibility_policy="practical",
        state_dir=tmp_path / "streaming-state",
        batch_size=2,
    )
    streaming_row = pd.read_parquet(streaming_output / "all_pairs.parquet").set_index("sample_id").loc["2bbb_A"]
    assert streaming_row["pairing_classification"] == legacy_row["pairing_classification"]
    assert streaming_row["forensic_root_cause"] == legacy_row["forensic_root_cause"]


RAW_FULL_FORENSIC_PAIRING_COMBINATIONS = (
    ("coordinate_disagreement", "unresolved_ambiguity"),
    ("demonstrated_sequence_matrix_mismatch", "unresolved_ambiguity"),
    (
        "historical_missing_calpha_selection_unreproducible",
        "unique_sequence_pair_coordinate_unavailable",
    ),
    ("historical_missing_calpha_selection_unreproducible", "unresolved_ambiguity"),
    ("legacy_trim_metadata_inconsistency", "verified_sequence_geometry_pair"),
    ("matrix_sequence_absent_from_author_linked_polymer", "unresolved_ambiguity"),
    ("residue_id_namespace_mismatch", "unique_sequence_pair_coordinate_unavailable"),
    ("residue_id_namespace_mismatch", "unresolved_ambiguity"),
    ("residue_id_namespace_mismatch", "verified_sequence_geometry_pair"),
    ("unresolved_ambiguity", "residue_identity_provenance_incomplete"),
    ("unresolved_ambiguity", "unresolved_ambiguity"),
    ("verified_sequence_geometry_pair", "verified_sequence_geometry_pair"),
)


@pytest.mark.parametrize(("forensic_root_cause", "pairing_classification"), RAW_FULL_FORENSIC_PAIRING_COMBINATIONS)
def test_completed_raw_full_forensic_pairing_combinations_are_compatible(
    forensic_root_cause: str, pairing_classification: str
) -> None:
    assert pairing_classification in FORENSIC_PAIRING_COMPATIBILITY[forensic_root_cause]
    _validate_forensic_pairing_relationship(
        forensic_root_cause,
        pairing_classification,
        sample_id="corpus-example",
    )


def test_incompatible_known_forensic_pairing_combination_is_rejected() -> None:
    with pytest.raises(ValueError, match="Incompatible forensic root cause"):
        _validate_forensic_pairing_relationship(
            "demonstrated_sequence_matrix_mismatch",
            "verified_sequence_geometry_pair",
            sample_id="contradiction",
        )


def test_genuine_pairing_classification_contradiction_is_still_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    practical_path = paths["audit"] / "practical_training_eligibility.csv"
    practical = pd.read_csv(practical_path)
    practical.loc[practical["sample_id"] == "1aaa_A", "v3_pairing_classification"] = (
        "unique_sequence_pair_coordinate_unavailable"
    )
    practical.to_csv(practical_path, index=False)

    with pytest.raises(ValueError, match="v3_pairing_classification contradiction"):
        _build(paths, tmp_path, "practical")


def test_full_compact_validation_streams_and_resumes_without_pandas_table_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, audit_mode="raw-full")
    _convert_audit_to_compact(paths)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["raw_inputs_unchanged"] = True
    protocol_path.write_text(json.dumps(protocol))
    state = tmp_path / "streaming-state"
    report_path = tmp_path / "streaming-report.json"
    import protein_distance_diffusion.data.pairing_streaming as streaming

    real_rss = streaming._rss_mib
    monkeypatch.setattr(streaming, "_rss_mib", lambda: 5000.0)
    with pytest.raises(streaming.PairingMemoryLimitExceeded, match="use --resume"):
        validate_sequence_geometry_pairing(
            processed_manifest=paths["processed"],
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            audit_dir=paths["audit"],
            normalization_file=paths["normalization"],
            validation_report=report_path,
            eligibility_policy="practical",
            expected_total=4,
            expected_eligible=3,
            expected_excluded=1,
            state_dir=state,
            batch_size=2,
            max_memory_mib=4096,
        )
    assert not report_path.exists()

    monkeypatch.setattr(streaming, "_rss_mib", real_rss)
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("whole table read"))
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("matrix payload read"))
    report = validate_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        validation_report=report_path,
        eligibility_policy="practical",
        expected_total=4,
        expected_eligible=3,
        expected_excluded=1,
        state_dir=state,
        resume=True,
        batch_size=2,
        max_memory_mib=4096,
    )

    assert report["status"] == "passed"
    assert report["matrix_reads_performed"] == 0
    assert report["matrix_files_hashed"] == 4
    assert report["validated_pairing_eligible_count"] == 3
    assert report["validated_pairing_ineligible_count"] == 1
    assert max((len(values) for values in report["failure_examples_by_reason"].values()), default=0) <= 100
    assert {row["stage"] for row in report["stages"]} == set(streaming.STAGES)


def test_full_compact_keyboard_interrupt_is_checkpointed_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, audit_mode="raw-full")
    _convert_audit_to_compact(paths)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["raw_inputs_unchanged"] = True
    protocol_path.write_text(json.dumps(protocol))
    state = tmp_path / "interrupt-state"
    report_path = tmp_path / "interrupt-report.json"
    import protein_distance_diffusion.data.pairing_streaming as streaming

    original = streaming.StreamingPairingEngine.index_manifests

    def interrupt(engine: streaming.StreamingPairingEngine) -> None:
        engine._heartbeat(streaming.STAGES[1], 0, started=time.monotonic())
        raise KeyboardInterrupt

    monkeypatch.setattr(streaming.StreamingPairingEngine, "index_manifests", interrupt)
    with pytest.raises(KeyboardInterrupt):
        validate_sequence_geometry_pairing(
            processed_manifest=paths["processed"],
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            audit_dir=paths["audit"],
            normalization_file=paths["normalization"],
            validation_report=report_path,
            eligibility_policy="practical",
            state_dir=state,
            batch_size=2,
        )
    assert not report_path.exists()
    with sqlite3.connect(state / "pairing_state.sqlite") as connection:
        status = connection.execute("SELECT status FROM stage_state WHERE stage=?", (streaming.STAGES[1],)).fetchone()[
            0
        ]
    assert status == "interrupted"

    monkeypatch.setattr(streaming.StreamingPairingEngine, "index_manifests", original)
    report = validate_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        validation_report=report_path,
        eligibility_policy="practical",
        expected_total=4,
        expected_eligible=3,
        expected_excluded=1,
        state_dir=state,
        resume=True,
        batch_size=2,
    )
    assert report["status"] == "passed"


def test_full_compact_build_writes_deterministic_bounded_parquet_partitions(tmp_path: Path) -> None:
    paths = _fixture(tmp_path / "inputs", audit_mode="raw-full")
    _convert_audit_to_compact(paths)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["raw_inputs_unchanged"] = True
    protocol_path.write_text(json.dumps(protocol))
    outputs = []
    protocols = []
    for run in range(2):
        output = tmp_path / f"streaming-output-{run}"
        protocols.append(
            build_sequence_geometry_pairing(
                processed_manifest=paths["processed"],
                train_manifest=paths["train"],
                validation_manifest=paths["validation"],
                audit_dir=paths["audit"],
                normalization_file=paths["normalization"],
                output_dir=output,
                eligibility_policy="practical",
                state_dir=tmp_path / f"build-state-{run}",
                batch_size=2,
            )
        )
        outputs.append(pd.read_parquet(output / "all_pairs.parquet").sort_values("sample_id").reset_index(drop=True))
        assert (
            max(
                pq.ParquetFile(path).metadata.num_rows for path in (output / "all_pairs.parquet").glob("part-*.parquet")
            )
            <= 2
        )

    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_like=True)
    assert protocols[0]["partitioned_parquet"] is True
    assert protocols[0]["validated_membership_counts"] == protocols[1]["validated_membership_counts"]
    assert protocols[0]["pairing_classification_counts"] == {
        "residue_identity_provenance_incomplete": 1,
        "unique_sequence_pair_coordinate_unavailable": 1,
        "unresolved_ambiguity": 1,
        "verified_sequence_geometry_pair": 1,
    }
    assert protocols[0]["candidate_evidence_row_count"] == 4


def test_arrow_scanner_keeps_100k_row_batches_bounded(tmp_path: Path) -> None:
    from protein_distance_diffusion.data.pairing_streaming import _parquet_batches

    path = tmp_path / "large.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": pa.array([f"sample-{index:06d}" for index in range(100_000)]),
                "value": pa.array(range(100_000), type=pa.int64()),
            }
        ),
        path,
        row_group_size=10_000,
    )

    sizes = [len(rows) for _, _, rows in _parquet_batches([path], 4_096)]

    assert sum(sizes) == 100_000
    assert max(sizes) <= 4_096


def test_builder_rejects_contradictory_unresolved_filename_aliases_before_output(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, audit_mode="raw-full")
    _convert_audit_to_compact(paths)
    audit_dir = paths["audit"]
    protocol_path = audit_dir / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["raw_inputs_unchanged"] = True
    protocol_path.write_text(json.dumps(protocol))
    compact_path = audit_dir / "unresolved_cases.csv"
    unresolved = pd.read_csv(compact_path).rename(
        columns={
            "training_eligibility": "practical_training_eligibility",
            "model_number": "model_id",
        }
    )
    unresolved.loc[0, "v3_pairing_classification"] = "verified_sequence_geometry_pair"
    unresolved.to_csv(audit_dir / "unresolved_case_summary.csv", index=False)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="Contradictory logical audit table aliases"):
        build_sequence_geometry_pairing(
            processed_manifest=paths["processed"],
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            audit_dir=audit_dir,
            normalization_file=paths["normalization"],
            output_dir=output,
            eligibility_policy="practical",
            allow_pilot_evidence=True,
        )

    assert not output.exists()


def test_pairing_valid_original_split_excluded_has_explicit_reason(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _move_sample_to_split(paths, "3ccc_A", "excluded")
    output = _build(paths, tmp_path, "practical")
    row = pd.read_parquet(output / "excluded_pairs.parquet").set_index("sample_id").loc["3ccc_A"]
    assert row["pairing_eligible"]
    assert row["practical_training_eligibility"] == "conditionally_verified_pair"
    assert row["selected_by_eligibility_policy"]
    assert row["original_split"] == "excluded"
    assert not row["eligible_for_training"]
    assert row["dataset_membership_status"] == "excluded"
    assert json.loads(row["exclusion_reasons"]) == ["original_split_excluded"]
    diagnostic = pd.read_parquet(output / "pairing_eligible_original_split_excluded.parquet")
    assert "3ccc_A" in set(diagnostic["sample_id"])


def test_unresolved_train_and_excluded_reason_semantics(tmp_path: Path) -> None:
    train_paths = _fixture(tmp_path / "train-case")
    _move_sample_to_split(train_paths, "4ddd_A", "train")
    train_output = _build(train_paths, tmp_path / "train-case", "practical")
    train_row = pd.read_parquet(train_output / "excluded_pairs.parquet").set_index("sample_id").loc["4ddd_A"]
    assert json.loads(train_row["exclusion_reasons"]) == ["unresolved_ambiguity"]
    assert not train_row["pairing_eligible"]
    assert not train_row["eligible_for_training"]

    excluded_paths = _fixture(tmp_path / "excluded-case")
    _move_sample_to_split(excluded_paths, "4ddd_A", "excluded")
    excluded_output = _build(excluded_paths, tmp_path / "excluded-case", "practical")
    excluded_row = pd.read_parquet(excluded_output / "excluded_pairs.parquet").set_index("sample_id").loc["4ddd_A"]
    assert json.loads(excluded_row["exclusion_reasons"]) == [
        "original_split_excluded",
        "unresolved_ambiguity",
    ]


def test_excluded_union_has_no_empty_reasons_and_protocol_counts_are_separate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _move_sample_to_split(paths, "3ccc_A", "excluded")
    output = _build(paths, tmp_path, "practical")
    excluded = pd.read_parquet(output / "excluded_pairs.parquet")
    protocol = json.loads((output / "protocol.json").read_text())
    assert all(json.loads(value) for value in excluded["exclusion_reasons"])
    assert protocol["pairing_eligible_count"] == 3
    assert protocol["pairing_ineligible_count"] == 1
    assert protocol["original_train_count"] == 2
    assert protocol["original_validation_count"] == 1
    assert protocol["original_split_excluded_count"] == 1
    assert protocol["eligible_train_count"] == 2
    assert protocol["eligible_validation_count"] == 0
    assert protocol["pairing_eligible_but_split_excluded_count"] == 1
    assert protocol["pairing_ineligible_validation_count"] == 1
    assert protocol["derived_dataset_excluded_count"] == 2
    assert "excluded_pair_count" not in protocol


def test_exact_pilot_membership_count_contract() -> None:
    rows = pd.DataFrame(
        [
            *({"pairing_eligible": True, "original_split": "train", "eligible_for_training": True},) * 415,
            *({"pairing_eligible": False, "original_split": "train", "eligible_for_training": False},) * 7,
            *({"pairing_eligible": True, "original_split": "validation", "eligible_for_training": True},) * 35,
            *({"pairing_eligible": False, "original_split": "validation", "eligible_for_training": False},),
            *({"pairing_eligible": True, "original_split": "excluded", "eligible_for_training": False},) * 225,
            *({"pairing_eligible": False, "original_split": "excluded", "eligible_for_training": False},) * 3,
        ]
    )
    assert _membership_counts(rows) == {
        "all_pair_count": 686,
        "pairing_eligible_count": 675,
        "pairing_ineligible_count": 11,
        "original_train_count": 422,
        "original_validation_count": 36,
        "original_split_excluded_count": 228,
        "eligible_train_count": 415,
        "eligible_validation_count": 35,
        "pairing_eligible_but_split_excluded_count": 225,
        "pairing_ineligible_train_count": 7,
        "pairing_ineligible_validation_count": 1,
        "pairing_ineligible_and_split_excluded_count": 3,
        "derived_dataset_excluded_count": 236,
    }


def test_pilot_requires_explicit_permission_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="allow-pilot-evidence"):
        build_sequence_geometry_pairing(
            processed_manifest=paths["processed"],
            train_manifest=paths["train"],
            validation_manifest=paths["validation"],
            audit_dir=paths["audit"],
            normalization_file=paths["normalization"],
            output_dir=output,
            eligibility_policy="practical",
            state_dir=tmp_path / "contradiction-state",
        )
    assert not output.exists()


def test_builder_refuses_to_overwrite_existing_preview(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = _build(paths, tmp_path, "practical")
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(paths, tmp_path, "practical")
    assert sentinel.read_text() == "preserve"


def test_eleven_unresolved_pilot_pairs_are_excluded(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    processed = pd.read_parquet(paths["processed"])
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    candidates_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidates_path)
    unresolved_path = paths["audit"] / "unresolved_case_summary.csv"
    unresolved_rows = pd.read_csv(unresolved_path).to_dict("records")
    template_manifest = processed.iloc[-1].to_dict()
    template_alignment = alignments[-1]
    template_candidate = candidates.iloc[-1].to_dict()
    for index in range(10):
        sample_id = f"5{index:03d}_A"
        sequence = "ACDEFGH"
        matrix_path = tmp_path / f"{sample_id}.npz"
        _write_npz(matrix_path, sequence)
        manifest = dict(template_manifest)
        manifest.update(
            sample_id=sample_id,
            pdb_id=sample_id[:4].upper(),
            sequence=sequence,
            length=len(sequence),
            original_chain_length=len(sequence),
            path=str(matrix_path),
        )
        processed.loc[len(processed)] = manifest
        alignment = dict(template_alignment)
        alignment.update(
            sample_id=sample_id,
            pdb_id=sample_id[:4].upper(),
            sequence=sequence,
            matrix_actual_length=len(sequence),
        )
        alignments.append(alignment)
        candidate = dict(template_candidate)
        candidate["sample_id"] = sample_id
        candidate["source_file"] = manifest["source_file"]
        candidates.loc[len(candidates)] = candidate
        unresolved_rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": manifest["pdb_id"],
                "chain_id": "A",
                "model_id": 1,
                "matrix_path": str(matrix_path),
                "v3_pairing_classification": "unresolved_ambiguity",
                "strict_provenance_status": alignment["strict_provenance_status"],
                "practical_training_eligibility": "excluded_or_unresolved",
                "source_identity_status": alignment["source_identity_status"],
            }
        )
    processed.to_parquet(paths["processed"], index=False)
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in alignments))
    candidates.to_parquet(candidates_path, index=False)
    pd.DataFrame(unresolved_rows).to_csv(unresolved_path, index=False)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["input_hashes"][str(paths["processed"])] = sha256_file(paths["processed"])
    protocol_path.write_text(json.dumps(protocol))
    output = _build(paths, tmp_path, "practical")
    excluded = pd.read_parquet(output / "excluded_pairs.parquet")
    assert (excluded["pairing_classification"] == "unresolved_ambiguity").sum() == 11


def _make_two_candidate_case(
    paths: dict[str, Path],
    *,
    second_strong: bool,
    second_author_linked: bool = False,
    candidate_identities: tuple[tuple[str, str], tuple[str, str]] = (("A", "A"), ("C", "H")),
) -> str:
    sample_id = "1aaa_A"
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    alignments[0].update(
        raw_match_count=2,
        training_eligibility="conditionally_verified_pair",
        strict_provenance_status="historical_source_sha_unavailable",
        unique_author_linked_candidate=True,
        author_linked_candidate_count=1,
    )
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in alignments))
    practical_path = paths["audit"] / "practical_training_eligibility.csv"
    practical = pd.read_csv(practical_path)
    practical.loc[practical["sample_id"] == sample_id, "practical_training_eligibility"] = "conditionally_verified_pair"
    practical.loc[practical["sample_id"] == sample_id, "strict_provenance_status"] = "historical_source_sha_unavailable"
    practical.to_csv(practical_path, index=False)
    transition_path = paths["audit"] / "v2_forensic_v3_transition.csv"
    transition = pd.read_csv(transition_path)
    transition.loc[transition["sample_id"] == sample_id, "practical_training_eligibility"] = (
        "conditionally_verified_pair"
    )
    transition.loc[transition["sample_id"] == sample_id, "strict_provenance_status"] = (
        "historical_source_sha_unavailable"
    )
    transition.to_csv(transition_path, index=False)
    strict_path = paths["audit"] / "strict_training_eligibility.csv"
    strict = pd.read_csv(strict_path)
    strict[strict["sample_id"] != sample_id].to_csv(strict_path, index=False)
    candidates_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidates_path)
    template = candidates[candidates["sample_id"] == sample_id].iloc[0].to_dict()
    other = candidates[candidates["sample_id"] != sample_id]
    rows = []
    for physical_index, identity in enumerate(("supported", "competing")):
        for convention_index, convention in enumerate(
            ("auth_seq_id_insertion", "label_seq_id", "position_zero_based", "position_one_based")
        ):
            row = dict(template)
            row.update(
                entity_id=str(physical_index + 1),
                label_asym_id=candidate_identities[physical_index][0],
                auth_asym_id=candidate_identities[physical_index][1],
                convention=convention,
                author_chain_match=str(physical_index == 0 or second_author_linked),
                strong_evidential_match=str(convention_index == 0 and (physical_index == 0 or second_strong)),
                sequence_match=str(convention_index == 0 and (identity == "supported" or second_strong)),
                coordinate_status="passed" if convention_index == 0 and identity == "supported" else "unavailable",
                coordinate_rmse_angstrom=0.0 if convention_index == 0 and identity == "supported" else np.nan,
                coordinate_max_abs_error_angstrom=(
                    0.0 if convention_index == 0 and identity == "supported" else np.nan
                ),
                trim_metadata_compatible=str(convention_index == 0 and identity == "supported"),
            )
            rows.append(row)
    pd.concat([other, pd.DataFrame(rows)], ignore_index=True).to_parquet(candidates_path, index=False)
    return sample_id


def _make_unresolved_two_candidate_case(paths: dict[str, Path]) -> str:
    sample_id = "4ddd_A"
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    alignments = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    for alignment in alignments:
        if alignment["sample_id"] == sample_id:
            alignment.update(
                raw_match_count=2,
                unique_author_linked_candidate=True,
                author_linked_candidate_count=1,
                coordinate_matrix_status="unavailable",
            )
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in alignments))
    candidates_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidates_path)
    template = candidates[candidates["sample_id"] == sample_id].iloc[0].to_dict()
    other = candidates[candidates["sample_id"] != sample_id]
    rows = []
    for physical_index in range(2):
        for convention in ("auth_seq_id_insertion", "label_seq_id", "position_zero_based", "position_one_based"):
            row = dict(template)
            row.update(
                entity_id=str(physical_index + 1),
                label_asym_id="C" if physical_index == 0 else "B",
                auth_asym_id="A" if physical_index == 0 else "C",
                convention=convention,
                author_chain_match=str(physical_index == 0),
                strong_evidential_match="False",
            )
            rows.append(row)
    pd.concat([other, pd.DataFrame(rows)], ignore_index=True).to_parquet(candidates_path, index=False)
    return sample_id


def _make_one_candidate_three_strong_conventions(paths: dict[str, Path]) -> str:
    sample_id = "1aaa_A"
    candidates_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidates_path)
    template = candidates[candidates["sample_id"] == sample_id].iloc[0].to_dict()
    other = candidates[candidates["sample_id"] != sample_id]
    rows = []
    for index, convention in enumerate(
        ("auth_seq_id_insertion", "label_seq_id", "position_zero_based", "position_one_based")
    ):
        row = dict(template)
        strong = index != 2
        row.update(
            candidate_index=index,
            convention=convention,
            sequence_match=str(strong),
            strong_evidential_match=str(strong),
            identity_mapping_available=str(strong),
        )
        rows.append(row)
    pd.concat([other, pd.DataFrame(rows)], ignore_index=True).to_parquet(candidates_path, index=False)
    return sample_id


def test_two_physical_candidates_and_eight_convention_rows_can_select_one(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sample_id = _make_two_candidate_case(paths, second_strong=False)
    output = _build(paths, tmp_path, "practical")
    row = pd.read_parquet(output / "all_pairs.parquet").set_index("sample_id").loc[sample_id]
    protocol = json.loads((output / "protocol.json").read_text())
    assert row["manifest_matrix_association_count"] == 1
    assert row["raw_polymer_candidate_count"] == 2
    assert row["raw_physical_candidate_count"] == 2
    assert row["candidate_evidence_row_count"] == 8
    assert row["strong_physical_candidate_count_all"] == 1
    assert row["strong_author_linked_physical_candidate_count"] == 1
    assert row["selected_candidate_unique"]
    assert row["eligible_for_training"]
    assert protocol["candidate_evidence_row_count"] == 11
    assert protocol["raw_physical_candidate_count"]["2"] == 1
    assert protocol["selected_candidate_unique"]["true"] == 3
    assert protocol["selected_candidate_unique"]["false"] == 1


def test_non_author_linked_strong_decoy_does_not_break_uniqueness(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sample_id = _make_two_candidate_case(
        paths,
        second_strong=True,
        candidate_identities=(("C", "D"), ("D", "E")),
    )
    _move_sample_to_split(paths, sample_id, "excluded")
    output = _build(paths, tmp_path, "practical")
    row = pd.read_parquet(output / "all_pairs.parquet").set_index("sample_id").loc[sample_id]
    assert row["raw_physical_candidate_count"] == 2
    assert row["author_linked_physical_candidate_count"] == 1
    assert row["non_author_linked_candidate_count"] == 1
    assert row["strong_physical_candidate_count_all"] == 2
    assert row["strong_author_linked_physical_candidate_count"] == 1
    assert row["strong_non_author_linked_candidate_count"] == 1
    assert row["label_asym_id"] == "C"
    assert row["auth_asym_id"] == "D"
    assert row["pairing_eligible"]
    assert not row["eligible_for_training"]
    assert json.loads(row["exclusion_reasons"]) == ["original_split_excluded"]
    rejected = json.loads(row["rejected_candidate_evidence"])
    assert rejected == [
        {
            "auth_asym_id": "E",
            "entity_id": "2",
            "has_strong_evidence": True,
            "label_asym_id": "D",
            "model_number": "1",
            "rejection_reason": "author_chain_mismatch",
            "source_file": str(tmp_path / "1aaa.cif.gz"),
        }
    ]


def test_one_physical_candidate_with_three_strong_conventions_is_eligible(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sample_id = _make_one_candidate_three_strong_conventions(paths)
    output = _build(paths, tmp_path, "practical")
    row = pd.read_parquet(output / "all_pairs.parquet").set_index("sample_id").loc[sample_id]
    assert row["candidate_evidence_row_count"] == 4
    assert row["strong_evidence_row_count"] == 3
    assert row["raw_physical_candidate_count"] == 1
    assert row["strong_physical_candidate_count_all"] == 1
    assert row["strong_author_linked_physical_candidate_count"] == 1
    assert json.loads(row["supported_residue_id_conventions"]) == [
        "auth_seq_id_insertion",
        "label_seq_id",
        "position_one_based",
    ]
    assert row["eligible_for_training"]


def test_two_strong_physical_candidates_fail_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _make_two_candidate_case(paths, second_strong=True, second_author_linked=True)
    output = tmp_path / "output-practical"
    with pytest.raises(ValueError, match="Multiple strong author-linked candidates"):
        _build(paths, tmp_path, "practical")
    assert not output.exists()


def test_zero_author_linked_candidates_fail_for_eligible_pair(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidates.loc[candidates["sample_id"] == "1aaa_A", "author_chain_match"] = "False"
    candidates.to_parquet(candidate_path, index=False)
    with pytest.raises(ValueError, match="No author-linked candidate"):
        _build(paths, tmp_path, "practical")
    assert not (tmp_path / "output-practical").exists()


def test_validation_only_accumulates_failures_without_dataset_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidates.loc[candidates["sample_id"] == "1aaa_A", "strong_evidential_match"] = "yes"
    candidates.loc[candidates["sample_id"] == "2bbb_A", "author_chain_match"] = "False "
    candidates.to_parquet(candidate_path, index=False)
    report_path = tmp_path / "validation.json"
    report = validate_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        validation_report=report_path,
        eligibility_policy="practical",
        allow_pilot_evidence=True,
        expected_total=4,
        expected_eligible=3,
        expected_excluded=1,
    )
    failed_samples = {
        sample_id for sample_ids in report["failure_sample_ids_by_reason"].values() for sample_id in sample_ids
    }
    assert report["status"] == "failed"
    assert {"1aaa_A", "2bbb_A"} <= failed_samples
    assert report["matrix_reads_performed"] == 0
    assert report["raw_inputs_unchanged"] is True
    assert json.loads(report_path.read_text()) == report
    assert not (tmp_path / "output-practical").exists()


def test_validation_only_success_report_matches_expected_contract(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report_path = tmp_path / "validation.json"
    report = validate_sequence_geometry_pairing(
        processed_manifest=paths["processed"],
        train_manifest=paths["train"],
        validation_manifest=paths["validation"],
        audit_dir=paths["audit"],
        normalization_file=paths["normalization"],
        validation_report=report_path,
        eligibility_policy="practical",
        allow_pilot_evidence=True,
        expected_total=4,
        expected_eligible=3,
        expected_excluded=1,
    )
    assert report["status"] == "passed"
    assert report["total_audited_pairs"] == 4
    assert report["validated_eligible_count"] == 3
    assert report["validated_excluded_count"] == 1
    assert report["failure_sample_ids_by_reason"] == {}
    assert report["raw_inputs_unchanged"] is True
    assert report["matrix_reads_performed"] == 0
    assert not (tmp_path / "output-practical").exists()


def test_validation_only_cli_returns_nonzero_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.build_sequence_geometry_pairing_manifest as cli

    paths = _fixture(tmp_path)
    candidate_path = paths["audit"] / "candidate_resolution_evidence.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidates.loc[candidates["sample_id"] == "1aaa_A", "sequence_match"] = "yes"
    candidates.to_parquet(candidate_path, index=False)
    report_path = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_sequence_geometry_pairing_manifest.py",
            "--processed-manifest",
            str(paths["processed"]),
            "--train-manifest",
            str(paths["train"]),
            "--validation-manifest",
            str(paths["validation"]),
            "--audit-dir",
            str(paths["audit"]),
            "--normalization-file",
            str(paths["normalization"]),
            "--validate-only",
            "--validation-report",
            str(report_path),
            "--eligibility-policy",
            "practical",
            "--allow-pilot-evidence",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 1
    assert json.loads(report_path.read_text())["status"] == "failed"


@pytest.mark.parametrize("missing_value", [None, pd.NA, np.nan])
def test_missing_strict_status_values_do_not_create_contradictions(tmp_path: Path, missing_value: object) -> None:
    paths = _fixture(tmp_path)
    unresolved_path = paths["audit"] / "unresolved_case_summary.csv"
    unresolved = pd.read_csv(unresolved_path)
    unresolved["strict_provenance_status"] = missing_value
    unresolved.to_csv(unresolved_path, index=False)
    output = _build(paths, tmp_path, "practical")
    row = pd.read_parquet(output / "excluded_pairs.parquet").set_index("sample_id").loc["4ddd_A"]
    assert row["strict_provenance_status"] == "contradictory_or_unresolved"


def test_identical_explicit_strict_status_passes(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    unresolved_path = paths["audit"] / "unresolved_case_summary.csv"
    unresolved = pd.read_csv(unresolved_path)
    unresolved["strict_provenance_status"] = "contradictory_or_unresolved"
    unresolved.to_csv(unresolved_path, index=False)
    assert _build(paths, tmp_path, "practical").is_dir()


def test_different_explicit_strict_status_fails_atomically(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    unresolved_path = paths["audit"] / "unresolved_case_summary.csv"
    unresolved = pd.read_csv(unresolved_path)
    unresolved["strict_provenance_status"] = "historical_source_sha_verified"
    unresolved.to_csv(unresolved_path, index=False)
    with pytest.raises(ValueError, match="strict_provenance_status contradiction"):
        _build(paths, tmp_path, "practical")
    assert not (tmp_path / "output-practical").exists()


def test_unresolved_two_candidate_evidence_remains_excluded(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sample_id = _make_unresolved_two_candidate_case(paths)
    output = _build(paths, tmp_path, "practical")
    excluded = pd.read_parquet(output / "excluded_pairs.parquet").set_index("sample_id")
    row = excluded.loc[sample_id]
    assert row["pairing_classification"] == "unresolved_ambiguity"
    assert row["strict_provenance_status"] == "contradictory_or_unresolved"
    assert row["practical_training_eligibility"] == "excluded_or_unresolved"
    assert row["raw_physical_candidate_count"] == 2
    assert row["candidate_evidence_row_count"] == 8
    assert sample_id not in set(pd.read_parquet(output / "eligible_train.parquet")["sample_id"])
    assert sample_id not in set(pd.read_parquet(output / "eligible_validation.parquet")["sample_id"])


def test_one_sample_with_two_manifest_paths_fails_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    processed = pd.read_parquet(paths["processed"])
    duplicate = processed.iloc[0].copy()
    duplicate["path"] = str(tmp_path / "other.npz")
    processed.loc[len(processed)] = duplicate
    processed.to_parquet(paths["processed"], index=False)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["input_hashes"][str(paths["processed"])] = sha256_file(paths["processed"])
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _build(paths, tmp_path, "practical")
    assert not (tmp_path / "output-practical").exists()


def test_two_samples_sharing_matrix_path_fail_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    processed = pd.read_parquet(paths["processed"])
    processed.loc[1, "path"] = processed.loc[0, "path"]
    processed.to_parquet(paths["processed"], index=False)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["input_hashes"][str(paths["processed"])] = sha256_file(paths["processed"])
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="duplicate path"):
        _build(paths, tmp_path, "practical")
    assert not (tmp_path / "output-practical").exists()


def test_unresolved_sample_in_eligible_table_is_integrity_failure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    practical_path = paths["audit"] / "practical_training_eligibility.csv"
    unresolved_path = paths["audit"] / "unresolved_case_summary.csv"
    practical = pd.read_csv(practical_path)
    unresolved = pd.read_csv(unresolved_path)
    practical = pd.concat([practical, unresolved], ignore_index=True)
    unresolved.iloc[0:0].to_csv(unresolved_path, index=False)
    practical.to_csv(practical_path, index=False)
    with pytest.raises(ValueError, match="Unresolved sample appears"):
        _build(paths, tmp_path, "practical")


def test_eligible_and_unresolved_membership_overlap_is_integrity_failure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    practical_path = paths["audit"] / "practical_training_eligibility.csv"
    unresolved = pd.read_csv(paths["audit"] / "unresolved_case_summary.csv")
    practical = pd.read_csv(practical_path)
    pd.concat([practical, unresolved], ignore_index=True).to_csv(practical_path, index=False)
    with pytest.raises(ValueError, match="both eligible and unresolved"):
        _build(paths, tmp_path, "practical")
    assert not (tmp_path / "output-practical").exists()


def test_manifest_npz_mismatch_fails_before_publication(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    alignment_path = paths["audit"] / "seqres_atom_matrix_alignments.jsonl"
    rows = [json.loads(line) for line in alignment_path.read_text().splitlines()]
    rows[0]["manifest_npz_sequence_match"] = False
    alignment_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="manifest_npz_sequence_mismatch"):
        _build(paths, tmp_path, "practical")
    assert not output.exists()


def test_split_leakage_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    validation = pd.read_parquet(paths["validation"])
    train = pd.read_parquet(paths["train"])
    validation.loc[0, "cluster_id"] = train.loc[0, "cluster_id"]
    validation.to_parquet(paths["validation"], index=False)
    protocol_path = paths["audit"] / "sequence_readiness_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["input_hashes"][str(paths["validation"])] = sha256_file(paths["validation"])
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="cluster_id leakage"):
        _build(paths, tmp_path, "practical")


def test_vocabulary_round_trip_and_strict_unknown_rejection() -> None:
    vocabulary = SequenceGeometryVocabulary()
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    assert vocabulary.decode(vocabulary.encode(sequence)) == sequence
    assert vocabulary.pad_id != vocabulary.mask_id
    with pytest.raises(ValueError, match="noncanonical"):
        vocabulary.encode("ACX")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
        (1, True),
        (0, False),
        (np.int64(1), True),
        (np.int64(0), False),
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
        ("1", True),
        ("0", False),
    ],
)
def test_strict_candidate_boolean_normalization(value: object, expected: bool) -> None:
    assert normalize_evidence_boolean(value, field="evidence", allow_missing=False) is expected


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, ""])
def test_candidate_boolean_missing_policy(value: object) -> None:
    assert normalize_evidence_boolean(value, field="evidence", allow_missing=True) is None
    with pytest.raises(ValueError, match="is required"):
        normalize_evidence_boolean(value, field="evidence", allow_missing=False)


@pytest.mark.parametrize("value", ["yes", "False ", 2, -1, 0.0, "arbitrary"])
def test_candidate_boolean_rejects_unrecognized_non_null_values(value: object) -> None:
    with pytest.raises(ValueError, match="Invalid Boolean evidence"):
        normalize_evidence_boolean(value, field="evidence", allow_missing=False)


def test_future_candidate_writer_uses_native_boolean_columns(tmp_path: Path) -> None:
    from scripts.audit_sequence_data_readiness import (
        CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS,
        _candidate_evidence_arrow_row,
        _candidate_evidence_schema,
    )

    columns = ["sample_id", *sorted(CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS)]
    schema = _candidate_evidence_schema(columns)
    row = {"sample_id": "1cq0_A", **{field: field != "identity_mapping_ambiguous" for field in columns[1:]}}
    table = pa.Table.from_pylist([_candidate_evidence_arrow_row(row, columns)], schema=schema)
    path = tmp_path / "candidate.parquet"
    pq.write_table(table, path)
    result = pd.read_parquet(path)
    assert all(str(result[field].dtype) == "bool" for field in CANDIDATE_EVIDENCE_BOOLEAN_COLUMNS)
    malformed = dict(row, strong_evidential_match="False")
    with pytest.raises(ValueError, match="must be a native Boolean"):
        _candidate_evidence_arrow_row(malformed, columns)


def test_candidate_index_and_convention_do_not_define_physical_identity() -> None:
    from scripts.audit_sequence_data_readiness import _candidate_key

    first = {
        "source_file": "1cq0.cif.gz",
        "model_number": 1,
        "entity_id": 1,
        "label_asym_id": "A",
        "auth_asym_id": "A",
        "candidate_index": 0,
        "convention": "auth_seq_id_insertion",
    }
    second = {**first, "candidate_index": 3, "convention": "position_one_based"}
    assert _candidate_key(first) == _candidate_key(second)


def test_dataset_modes_dropout_and_mixed_length_collation(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = _build(paths, tmp_path, "practical")
    manifest = output / "eligible_train.parquet"
    sequence_only = SequenceGeometryDataset(manifest, mode="sequence_only")
    item = sequence_only[0]
    assert item["geometry_availability_flag"]
    assert not item["geometry_conditioning_flag"]
    assert not item["pair_mask"].any()
    conditioned = SequenceGeometryDataset(manifest, mode="geometry_conditioned")
    assert conditioned[0]["geometry_conditioning_flag"]
    assert conditioned[0]["pair_mask"].all()
    dropout = SequenceGeometryDataset(
        manifest,
        mode="geometry_conditioned_with_dropout",
        conditioning_dropout_probability=1.0,
        seed=17,
    )
    first = dropout[0]
    second = dropout[0]
    assert torch.equal(first["distance_matrix"], second["distance_matrix"])
    assert not first["geometry_conditioning_flag"]
    batch = collate_sequence_geometry([conditioned[0], conditioned[1]])
    assert batch["sequence_token_ids"].shape == (2, 5)
    assert batch["distance_matrices"].shape == (2, 5, 5)
    assert not batch["pair_mask"][0, 4].any()
    assert batch["sequence_mask"].sum().item() == 9


@pytest.mark.parametrize(
    "corruption",
    (
        AdditiveSymmetricNoise(0.2),
        LongRangePairMask(2, 0.5),
        ContactDeletion(8.0, 0.5),
        LowRankDistanceDistortion(2, 0.5),
    ),
)
def test_corruptions_preserve_symmetry_diagonal_and_masks(tmp_path: Path, corruption: object) -> None:
    paths = _fixture(tmp_path)
    output = _build(paths, tmp_path, "practical")
    dataset = SequenceGeometryDataset(
        output / "eligible_train.parquet",
        mode="corrupted_real_geometry",
        corruption=corruption,
        seed=91,
    )
    first = dataset[0]
    second = dataset[0]
    matrix = first["distance_matrix"]
    assert torch.equal(matrix, second["distance_matrix"])
    assert torch.allclose(matrix, matrix.T)
    assert torch.equal(torch.diag(matrix), torch.zeros(len(matrix)))
    assert torch.equal(first["pair_mask"], first["pair_mask"].T)
    assert torch.equal(matrix[~first["pair_mask"]], torch.zeros_like(matrix[~first["pair_mask"]]))


def test_input_files_are_unchanged(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    inputs = [paths[name] for name in ("processed", "train", "validation", "normalization")]
    before = {path: sha256_file(path) for path in inputs}
    _build(paths, tmp_path, "practical")
    assert {path: sha256_file(path) for path in inputs} == before


def test_configured_corruption_mixture_is_supported() -> None:
    corruption = build_geometry_corruption(
        {
            "type": "mixture",
            "transforms": [
                {"type": "additive_symmetric_noise", "standard_deviation": 0.1},
                {"type": "long_range_pair_mask", "minimum_separation": 2, "mask_probability": 0.2},
            ],
        }
    )
    assert corruption is not None
    with pytest.raises(ValueError, match="non-empty transforms"):
        build_geometry_corruption({"type": "mixture", "transforms": []})
