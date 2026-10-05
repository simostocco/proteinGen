from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization
from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryVocabulary
from protein_distance_diffusion.evaluation import e007_split_homology as audit


def _config(tmp_path: Path) -> dict:
    payload = yaml.safe_load(Path("configs/e007_split_homology_audit_v1.yaml").read_text())
    payload["output_dir"] = str(tmp_path / "output")
    payload["dataset"]["root"] = str(tmp_path / "dataset")
    return payload


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _authorization(root: Path, counts: dict[str, int]) -> RichDatasetAuthorization:
    return RichDatasetAuthorization(
        root=root,
        protocol_sha256="1" * 64,
        schema_sha256="2" * 64,
        vocabulary_sha256="3" * 64,
        normalization_sha256="4" * 64,
        shard_inventory_sha256="5" * 64,
        split_counts=counts,
        observed_shard_hashes={"train/part.parquet": "6" * 64},
    )


def _make_database(tmp_path: Path) -> sqlite3.Connection:
    return audit._database(tmp_path / "audit.sqlite")


def _insert(connection: sqlite3.Connection, rows: list[tuple[str, str, str, str]]) -> None:
    records = []
    for sample_id, split, pdb_id, sequence in rows:
        records.append(
            (
                sample_id,
                split,
                pdb_id,
                len(sequence),
                audit.length_stratum(
                    len(sequence),
                    [
                        {"name": "short", "minimum": 1, "maximum": 4},
                        {"name": "long", "minimum": 5, "maximum": 8},
                    ],
                ),
                audit.exact_sequence_hash(sequence),
                sequence,
            )
        )
    connection.executemany(
        """INSERT INTO sequences(
        sample_id,split,pdb_id,length,length_stratum,sequence_hash,sequence
        ) VALUES (?,?,?,?,?,?,?)""",
        records,
    )
    connection.commit()


def test_canonical_sequence_reconstruction_and_exact_hash() -> None:
    vocabulary = SequenceGeometryVocabulary()
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    tokens = vocabulary.encode(sequence)
    assert audit.reconstruct_canonical_sequence(tokens, vocabulary) == sequence
    assert audit.reconstruct_canonical_sequence(pa.array([tokens])[0], vocabulary) == sequence
    assert audit.exact_sequence_hash(sequence) == hashlib.sha256(sequence.encode("ascii")).hexdigest()
    for invalid in ([0, 2], [1, 2], [22], []):
        with pytest.raises(ValueError):
            audit.reconstruct_canonical_sequence(invalid, vocabulary)


@pytest.mark.parametrize(
    ("sample_id", "source_path", "expected"),
    [
        ("1ABC_A", "/raw/mmcif/ab/1abc.cif.gz", "1abc"),
        ("2xyz_B_model1", "/raw/2XYZ.ent", "2xyz"),
        ("3def-A", "", "3def"),
    ],
)
def test_pdb_entry_extraction(sample_id: str, source_path: str, expected: str) -> None:
    assert audit.extract_pdb_id(sample_id, source_path) == expected
    with pytest.raises(ValueError, match="contradiction"):
        audit.extract_pdb_id("1abc_A", "/raw/2def.cif.gz")


def test_known_exact_and_pdb_leakage_and_clean_manifests(tmp_path: Path) -> None:
    connection = _make_database(tmp_path)
    _insert(
        connection,
        [
            ("1abc_A", "train", "1abc", "AAAA"),
            ("1abc_B", "validation", "1abc", "CCCC"),
            ("2def_A", "validation", "2def", "AAAA"),
            ("3ghi_A", "validation", "3ghi", "DDDDD"),
        ],
    )
    strata = [
        {"name": "short", "minimum": 1, "maximum": 4},
        {"name": "long", "minimum": 5, "maximum": 8},
    ]
    exact = audit._group_effect(connection, "sequence_hash", 10, strata)
    pdb = audit._group_effect(connection, "pdb_id", 10, strata)
    assert exact["cross_split_group_count"] == 1
    assert exact["validation_samples_affected"] == 1
    assert pdb["cross_split_group_count"] == 1
    assert pdb["validation_samples_affected"] == 1
    exact_manifest = audit._clean_manifest(connection, "sequence_hash", tmp_path / "exact.parquet", 2)
    pdb_manifest = audit._clean_manifest(connection, "pdb_id", tmp_path / "pdb.parquet", 2)
    assert exact_manifest["retained"] == pdb_manifest["retained"] == 2
    assert pq.read_table(tmp_path / "exact.parquet")["sample_id"].to_pylist() == ["1abc_B", "3ghi_A"]
    connection.close()


def test_synthetic_cluster_leakage_clean_manifest_and_length_strata(tmp_path: Path) -> None:
    connection = _make_database(tmp_path)
    _insert(
        connection,
        [
            ("1abc_A", "train", "1abc", "AAAA"),
            ("2abc_A", "train", "2abc", "CCCCC"),
            ("3abc_A", "validation", "3abc", "DDDD"),
            ("4abc_A", "validation", "4abc", "EEEEEE"),
        ],
    )
    assignments = tmp_path / "assignments.tsv"
    assignments.write_text("c1\t1abc_A\nc2\t2abc_A\nc1\t3abc_A\nc3\t4abc_A\n")
    audit._load_assignments(connection, assignments)
    directory = tmp_path / "cluster"
    directory.mkdir()
    result = audit._cluster_outputs(
        connection,
        directory,
        [
            {"name": "short", "minimum": 1, "maximum": 4},
            {"name": "long", "minimum": 5, "maximum": 8},
        ],
        10,
        2,
        1,
    )
    assert result["cross_split_clusters"] == 1
    assert result["validation_samples_in_cross_split_clusters"] == 1
    assert result["clean_validation_count"] == 1
    assert result["by_length_stratum"]["short"]["validation_samples_affected"] == 1
    assert result["by_length_stratum"]["long"]["validation_samples_affected"] == 0
    assert pq.read_table(directory / "clean_validation_manifest.parquet")["sample_id"].to_pylist() == ["4abc_A"]
    connection.close()


def test_assignment_validation_rejects_missing_and_duplicate_members(tmp_path: Path) -> None:
    connection = _make_database(tmp_path)
    _insert(connection, [("1abc_A", "train", "1abc", "AAAA"), ("2abc_A", "validation", "2abc", "CCCC")])
    incomplete = tmp_path / "incomplete.tsv"
    incomplete.write_text("c1\t1abc_A\n")
    with pytest.raises(ValueError, match="membership contradiction"):
        audit._load_assignments(connection, incomplete)
    duplicate = tmp_path / "duplicate.tsv"
    duplicate.write_text("c1\t1abc_A\nc2\t1abc_A\nc3\t2abc_A\n")
    with pytest.raises(ValueError, match="duplicate MMseqs2 assignment"):
        audit._load_assignments(connection, duplicate)
    connection.close()


def test_deterministic_fasta_and_record_hash_ignore_fragment_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocabulary = SequenceGeometryVocabulary()
    rows = {
        "a": [
            {
                "sample_id": "2def_A",
                "split": "validation",
                "source_path": "2def.cif",
                "sequence": "CCCC",
                "token_ids": vocabulary.encode("CCCC"),
                "ca_mask": [True] * 4,
                "chain_continuity_mask": [True] * 3,
                "chain_break_mask": [False] * 3,
            }
        ],
        "b": [
            {
                "sample_id": "1abc_A",
                "split": "train",
                "source_path": "1abc.cif",
                "sequence": "AAAA",
                "token_ids": vocabulary.encode("AAAA"),
                "ca_mask": [True] * 4,
                "chain_continuity_mask": [True] * 3,
                "chain_break_mask": [False] * 3,
            }
        ],
    }
    files = {}
    dataset = Path(_config(tmp_path)["dataset"]["root"])
    dataset.mkdir()
    for name, values in rows.items():
        path = dataset / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(values), path)
        files[name] = path
    config = _config(tmp_path)
    config["length_strata"] = [{"name": "all", "minimum": 1, "maximum": 8}]

    def run(order: list[str], suffix: str) -> tuple[dict, str]:
        monkeypatch.setattr(
            audit, "_scientific_shards", lambda _: iter((rows[name][0]["split"], files[name]) for name in order)
        )
        connection = audit._database(tmp_path / f"{suffix}.sqlite")
        fasta = tmp_path / f"{suffix}.fasta"
        result = audit._index_sequences(config, connection, fasta)
        connection.close()
        return result, fasta.read_text()

    first, first_fasta = run(["a", "b"], "first")
    second, second_fasta = run(["b", "a"], "second")
    assert first == second
    assert first_fasta == second_fasta
    assert first_fasta.startswith(">1abc_A")


def test_rejected_rows_never_enter_fasta_leakage_or_clean_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocabulary = SequenceGeometryVocabulary()

    def row(sample_id: str, split: str, sequence: str, *, accepted: bool) -> dict:
        continuity = [True] * (len(sequence) - 1)
        if not accepted:
            continuity[0] = False
        return {
            "sample_id": sample_id,
            "split": split,
            "source_path": f"{sample_id[:4]}.cif",
            "sequence": sequence,
            "token_ids": vocabulary.encode(sequence),
            "ca_mask": [True] * len(sequence),
            "chain_continuity_mask": continuity,
            "chain_break_mask": [not value for value in continuity],
        }

    train_rows = [
        row("1abc_A", "train", "AAAA", accepted=True),
        row("2def_A", "train", "CCCC", accepted=False),
    ]
    validation_rows = [
        row("3ghi_A", "validation", "AAAA", accepted=True),
        row("4jkl_A", "validation", "CCCC", accepted=True),
        row("5mno_A", "validation", "DDDD", accepted=False),
    ]
    paths = {}
    dataset = Path(_config(tmp_path)["dataset"]["root"])
    dataset.mkdir()
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        path = dataset / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths[split] = path
    monkeypatch.setattr(
        audit,
        "_scientific_shards",
        lambda _: iter((("train", paths["train"]), ("validation", paths["validation"]))),
    )
    config = _config(tmp_path)
    config["length_strata"] = [{"name": "all", "minimum": 1, "maximum": 8}]
    connection = audit._database(tmp_path / "population.sqlite")
    fasta = tmp_path / "accepted.fasta"
    indexed = audit._index_sequences(config, connection, fasta)
    assert indexed["population_counts"] == {
        "train": {"candidate": 2, "accepted": 1, "rejected": 1},
        "validation": {"candidate": 3, "accepted": 2, "rejected": 1},
    }
    assert indexed["rejection_reason_counts"] == {
        "train": {"chain_break": 1},
        "validation": {"chain_break": 1},
    }
    text = fasta.read_text()
    assert set(line[1:] for line in text.splitlines() if line.startswith(">")) == {
        "1abc_A",
        "3ghi_A",
        "4jkl_A",
    }
    exact = audit._group_effect(connection, "sequence_hash", 10)
    assert exact["validation_samples_affected"] == 1
    exact_clean = audit._clean_manifest(connection, "sequence_hash", tmp_path / "exact-clean.parquet", 2)
    assert exact_clean["retained"] == 1
    assert pq.read_table(tmp_path / "exact-clean.parquet")["sample_id"].to_pylist() == ["4jkl_A"]
    assignments = tmp_path / "accepted-clusters.tsv"
    assignments.write_text("c1\t1abc_A\nc1\t3ghi_A\nc2\t4jkl_A\n")
    audit._load_assignments(connection, assignments)
    cluster = tmp_path / "cluster"
    cluster.mkdir()
    result = audit._cluster_outputs(
        connection,
        cluster,
        config["length_strata"],
        10,
        2,
        1,
    )
    assert result["assignment_count"] == 3
    assert result["validation_samples_in_cross_split_clusters"] == 1
    assert pq.read_table(cluster / "clean_validation_manifest.parquet")["sample_id"].to_pylist() == ["4jkl_A"]
    connection.close()


def test_expected_accepted_population_counts_are_enforced() -> None:
    expected = {
        "train": {"candidate": 2, "accepted": 1, "rejected": 1},
        "validation": {"candidate": 3, "accepted": 2, "rejected": 1},
    }
    audit.require_population_counts(copy.deepcopy(expected), expected)
    wrong = copy.deepcopy(expected)
    wrong["validation"]["accepted"] += 1
    with pytest.raises(ValueError, match="accepted-population contradiction"):
        audit.require_population_counts(wrong, expected)


def test_missing_mmseqs_refuses_before_output_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    path = _write_config(tmp_path, config)
    monkeypatch.setattr(audit, "_verify_prerequisites", lambda _: {"hashes": {}, "dataset": {}})
    monkeypatch.setattr(
        audit,
        "detect_mmseqs",
        lambda _: {"available": False, "requested_executable": "missing", "path": None, "version": None},
    )
    with pytest.raises(RuntimeError, match="MMseqs2 is unavailable"):
        audit.audit_split_homology(path)
    assert not Path(config["output_dir"]).exists()
    assert not Path(config["output_dir"]).with_name(".output.inprogress").exists()


def test_plan_is_metadata_only_and_does_not_create_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    path = _write_config(tmp_path, config)
    monkeypatch.setattr(
        audit,
        "_verify_prerequisites",
        lambda _: {
            "hashes": {"protected": "a" * 64},
            "dataset": {"eligible_split_counts": {"train": 7, "validation": 3}},
            "population_counts": {
                "train": {"candidate": 7, "accepted": 6, "rejected": 1},
                "validation": {"candidate": 3, "accepted": 2, "rejected": 1},
            },
        },
    )
    monkeypatch.setattr(
        audit,
        "detect_mmseqs",
        lambda _: {"available": True, "requested_executable": "mmseqs", "path": "/usr/bin/mmseqs", "version": "test"},
    )
    monkeypatch.setattr(pq, "ParquetFile", lambda *_args, **_kwargs: pytest.fail("plan opened a Parquet shard"))
    result = audit.plan_split_homology_audit(path)
    assert result["planned_sequence_counts"] == {"train": 6, "validation": 2, "combined": 8}
    assert result["population_counts"]["train"] == {"candidate": 7, "accepted": 6, "rejected": 1}
    assert result["sequence_records_scanned"] is False
    assert result["mmseqs_executed"] is False
    assert result["model_created"] is result["optimizer_created"] is False
    assert not Path(config["output_dir"]).exists()


def test_atomic_publication_non_authorization_and_protected_input_preservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    dataset = Path(config["dataset"]["root"])
    dataset.mkdir()
    protected = dataset / "protected.txt"
    protected.write_text("immutable")
    protected_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    path = _write_config(tmp_path, config)
    monkeypatch.setattr(
        audit,
        "_verify_prerequisites",
        lambda _: {
            "hashes": {"protected": protected_hash},
            "dataset": {"eligible_split_counts": {"train": 1, "validation": 1}},
            "population_counts": {
                "train": {"candidate": 1, "accepted": 1, "rejected": 0},
                "validation": {"candidate": 1, "accepted": 1, "rejected": 0},
            },
            "phase3e_a_train_accepted_sample_id_sha256": audit._canonical_hash(["1abc_A"]),
        },
    )
    authorization = _authorization(dataset, {"train": 1, "validation": 1})
    monkeypatch.setattr(audit, "authorize_rich_geometry_dataset", lambda *_args, **_kwargs: authorization)
    monkeypatch.setattr(
        audit,
        "detect_mmseqs",
        lambda _: {"available": True, "requested_executable": "mmseqs", "path": "/usr/bin/mmseqs", "version": "test"},
    )

    def fake_index(_config: dict, connection: sqlite3.Connection, fasta: Path) -> dict:
        _insert(
            connection,
            [("1abc_A", "train", "1abc", "AAAA"), ("2def_A", "validation", "2def", "CCCC")],
        )
        fasta.write_text(">1abc_A\nAAAA\n>2def_A\nCCCC\n")
        return {
            "population_counts": {
                "train": {"candidate": 1, "accepted": 1, "rejected": 0},
                "validation": {"candidate": 1, "accepted": 1, "rejected": 0},
            },
            "rejection_reason_counts": {"train": {}, "validation": {}},
            "accepted_sample_id_sha256": {
                "train": audit._canonical_hash(["1abc_A"]),
                "validation": audit._canonical_hash(["2def_A"]),
            },
            "combined_accepted_sample_id_sha256": audit._canonical_hash(["1abc_A", "2def_A"]),
            "rejected_sample_id_and_reason_sha256": audit._canonical_hash([]),
            "fasta_record_count": 2,
            "sequence_record_sha256": "a" * 64,
            "fasta_sha256": audit.sha256_file(fasta),
        }

    def fake_mmseqs(command: list[str], _log: Path) -> None:
        prefix = Path(command[3])
        Path(f"{prefix}_cluster.tsv").write_text("1abc_A\t1abc_A\n2def_A\t2def_A\n")

    monkeypatch.setattr(audit, "_index_sequences", fake_index)
    monkeypatch.setattr(audit, "_run_mmseqs", fake_mmseqs)
    result = audit.audit_split_homology(path)
    output = Path(config["output_dir"])
    assert result["status"] == "completed_non_authorizing"
    assert output.is_dir() and not output.with_name(".output.inprogress").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["optimizer_updates"] == 0
    assert all(report[key] is False for key in audit.NON_AUTHORIZING if key != "optimizer_updates")
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == protected_hash
    with pytest.raises(FileExistsError):
        audit.audit_split_homology(path)


def test_mmseqs_command_declares_deterministic_scientific_contract() -> None:
    command = audit.mmseqs_command(
        "/usr/bin/mmseqs",
        "input.fasta",
        "clusters",
        "tmp",
        identity=0.3,
        coverage=0.8,
        coverage_mode=0,
        sensitivity=7.5,
        threads=1,
    )
    assert command[:2] == ["/usr/bin/mmseqs", "easy-cluster"]
    assert command[command.index("--min-seq-id") + 1] == "0.3"
    assert command[command.index("-c") + 1] == "0.8"
    assert command[command.index("--cov-mode") + 1] == "0"
    assert command[command.index("--threads") + 1] == "1"
    assert audit.MMSEQS_ENVIRONMENT["OMP_NUM_THREADS"] == "1"


def test_configuration_hash_is_deterministic(tmp_path: Path) -> None:
    payload = _config(tmp_path)
    first = _write_config(tmp_path, payload)
    first_hash = audit.sha256_file(first)
    second_payload = copy.deepcopy(payload)
    second = tmp_path / "second.yaml"
    second.write_text(yaml.safe_dump(second_payload, sort_keys=False))
    assert audit.sha256_file(second) == first_hash


def test_report_and_protocol_payloads_are_distinct() -> None:
    report = {
        "status": "completed_non_authorizing",
        "version": audit.AUDIT_VERSION,
        "configuration_sha256": "a" * 64,
        "indexed_sequences": {
            "population_counts": {"train": {"accepted": 1}, "validation": {"accepted": 1}},
            "accepted_sample_id_sha256": {"train": "b" * 64, "validation": "c" * 64},
            "rejected_sample_id_and_reason_sha256": "d" * 64,
            "fasta_sha256": "e" * 64,
        },
        "mmseqs": {"version": "test"},
        "mmseqs_environment": {"OMP_NUM_THREADS": "1"},
        "protected_input_hashes": {"dataset": "f" * 64},
        "protected_scientific_shards_unchanged": True,
        "artifact_hashes": {"assignment.parquet": "1" * 64},
        "sequence_clustering": {"identity_30": {"cross_split_clusters": 1}},
        **audit.NON_AUTHORIZING,
    }
    protocol = audit._protocol_payload(report, "2" * 64)
    report_bytes = json.dumps(report, sort_keys=True).encode()
    protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
    assert report_bytes != protocol_bytes
    assert hashlib.sha256(report_bytes).hexdigest() != hashlib.sha256(protocol_bytes).hexdigest()
    assert "sequence_clustering" not in protocol
