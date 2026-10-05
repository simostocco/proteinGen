from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from protein_distance_diffusion.evaluation import e007_split_homology_publication as publication


def _config(tmp_path: Path) -> dict:
    config = yaml.safe_load(Path("configs/e007_split_homology_publication_correction_v1.yaml").read_text())
    config["source_output_dir"] = str(tmp_path / "source")
    config["output_dir"] = str(tmp_path / "corrected")
    config["source_audit_config_path"] = str(tmp_path / "source-config.yaml")
    config["expected_population_counts"] = {
        "accepted": {"train": 2, "validation": 2, "combined": 4},
        "rejected": {"train": 1, "validation": 1, "combined": 2},
    }
    return config


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "publication.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _assignment_and_clean(tmp_path: Path) -> tuple[Path, Path]:
    assignments = tmp_path / "assignments.parquet"
    clean = tmp_path / "clean.parquet"
    rows = [
        {"sample_id": "t1", "split": "train", "cluster_id": "c1", "length_stratum": "short"},
        {"sample_id": "t2", "split": "train", "cluster_id": "c2", "length_stratum": "long"},
        {"sample_id": "v1", "split": "validation", "cluster_id": "c1", "length_stratum": "short"},
        {"sample_id": "v2", "split": "validation", "cluster_id": "c3", "length_stratum": "long"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), assignments)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "v2",
                    "split": "validation",
                    "coordinate_accepted": True,
                    "length_stratum": "long",
                }
            ]
        ),
        clean,
    )
    return assignments, clean


def test_threshold_summary_and_clean_manifest_are_reproduced(tmp_path: Path) -> None:
    assignments, clean = _assignment_and_clean(tmp_path)
    directory = tmp_path / "mmseqs" / "identity_90"
    directory.mkdir(parents=True)
    assignments.replace(directory / "cluster_assignments.parquet")
    clean.replace(directory / "clean_validation_manifest.parquet")
    source_summary = {
        "cross_split_clusters": 1,
        "validation_samples_in_cross_split_clusters": 1,
        "clean_validation_count": 1,
        "assignment_sha256": publication.sha256_file(directory / "cluster_assignments.parquet"),
        "clean_validation_sha256": publication.sha256_file(directory / "clean_validation_manifest.parquet"),
    }
    result = publication._reproduce_threshold(
        tmp_path,
        "identity_90",
        {"cross_split_clusters": 1, "affected_validation_samples": 1, "clean_validation_count": 1},
        source_summary,
        {"train": 2, "validation": 2},
    )
    assert result["assignment_count"] == 4
    assert result["cross_split_clusters"] == 1
    assert result["affected_validation_samples"] == 1
    assert result["clean_validation_count"] == 1
    assert result["clean_validation_by_length_stratum"] == {"long": 1}


def test_clean_manifest_reproduction_rejects_wrong_membership(tmp_path: Path) -> None:
    assignments, clean = _assignment_and_clean(tmp_path)
    directory = tmp_path / "mmseqs" / "identity_90"
    directory.mkdir(parents=True)
    assignments.replace(directory / "cluster_assignments.parquet")
    table = pq.read_table(clean).to_pylist()
    table[0]["sample_id"] = "v1"
    pq.write_table(pa.Table.from_pylist(table), directory / "clean_validation_manifest.parquet")
    summary = {
        "cross_split_clusters": 1,
        "validation_samples_in_cross_split_clusters": 1,
        "clean_validation_count": 1,
        "assignment_sha256": publication.sha256_file(directory / "cluster_assignments.parquet"),
        "clean_validation_sha256": publication.sha256_file(directory / "clean_validation_manifest.parquet"),
    }
    with pytest.raises(ValueError, match="clean manifest contradicts"):
        publication._reproduce_threshold(
            tmp_path,
            "identity_90",
            {"cross_split_clusters": 1, "affected_validation_samples": 1, "clean_validation_count": 1},
            summary,
            {"train": 2, "validation": 2},
        )


def test_count_conservation_refuses_missing_assignment(tmp_path: Path) -> None:
    assignments, clean = _assignment_and_clean(tmp_path)
    directory = tmp_path / "mmseqs" / "identity_90"
    directory.mkdir(parents=True)
    rows = pq.read_table(assignments).to_pylist()[:-1]
    pq.write_table(pa.Table.from_pylist(rows), directory / "cluster_assignments.parquet")
    clean.replace(directory / "clean_validation_manifest.parquet")
    with pytest.raises(ValueError, match="assignment population contradiction"):
        publication._reproduce_threshold(
            tmp_path,
            "identity_90",
            {"cross_split_clusters": 1, "affected_validation_samples": 1, "clean_validation_count": 1},
            {},
            {"train": 2, "validation": 2},
        )


def test_inventory_excludes_publication_and_transient_files(tmp_path: Path) -> None:
    (tmp_path / "durable.parquet").write_bytes(b"durable")
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "protocol.json").write_text("{}")
    (tmp_path / "heartbeat.json").write_text("{}")
    (tmp_path / "artifact_inventory.json").write_text("{}")
    transient = tmp_path / "mmseqs" / "identity_30" / "tmp"
    transient.mkdir(parents=True)
    (transient / "database").write_bytes(b"transient")
    inventory = publication._publication_inventory(tmp_path)
    assert [record["path"] for record in inventory["artifacts"]] == ["durable.parquet"]
    assert inventory["aggregate_inventory_sha256"] == publication._canonical_hash(inventory["artifacts"])


def test_source_artifact_hash_refusal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    relative = publication._durable_paths()[0]
    path = source / relative
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="hash contradiction"):
        publication._verify_source_artifacts(source, {"artifact_hashes": {relative: "0" * 64}})


def test_atomic_publication_separates_report_protocol_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config_path = _write_config(tmp_path, config)
    source = Path(config["source_output_dir"])
    source.mkdir()
    source_file = source / "durable.parquet"
    source_file.write_bytes(b"immutable-science")
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    source_payload = {
        "mmseqs": {"path": "/usr/bin/mmseqs", "version": "test"},
        "protected_scientific_shards_unchanged": True,
    }
    verified = {
        "source": source,
        "source_payload": source_payload,
        "source_inventory": {
            "durable.parquet": {
                "path": "durable.parquet",
                "size_bytes": source_file.stat().st_size,
                "sha256": source_hash,
            }
        },
        "prerequisite": {"hashes": {"dataset": "a" * 64}},
        "exact_sequence": {"affected_validation_samples": 0},
        "same_pdb_entry": {"affected_validation_samples": 0},
        "thresholds": {threshold: {"assignment_count": 4} for threshold in publication.THRESHOLDS},
    }
    monkeypatch.setattr(publication, "_verify_all", lambda _: copy.deepcopy(verified))
    result = publication.publish_correction(config_path)
    output = Path(config["output_dir"])
    assert result["status"] == "completed_non_authorizing"
    report = output / "report.json"
    protocol = output / "protocol.json"
    assert report.read_bytes() != protocol.read_bytes()
    assert publication.sha256_file(report) != publication.sha256_file(protocol)
    assert json.loads(report.read_text())["homology_thresholds"]
    assert "homology_thresholds" not in json.loads(protocol.read_text())
    inventory = json.loads((output / "artifact_inventory.json").read_text())
    assert [record["path"] for record in inventory["artifacts"]] == ["durable.parquet"]
    assert hashlib.sha256(source_file.read_bytes()).hexdigest() == source_hash
    assert not output.with_name(".corrected.inprogress").exists()
    with pytest.raises(FileExistsError):
        publication.publish_correction(config_path)


def test_plan_is_read_only_and_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    path = _write_config(tmp_path, config)
    source = Path(config["source_output_dir"])
    source.mkdir()
    payload = {"status": "completed_non_authorizing", "configuration_sha256": "c" * 64, "artifact_hashes": {}}
    data = json.dumps(payload).encode()
    (source / "report.json").write_bytes(data)
    (source / "protocol.json").write_bytes(data)
    digest = publication.sha256_file(source / "report.json")
    config["source_publication"] = {
        "report_sha256": digest,
        "protocol_sha256": digest,
        "configuration_sha256": "c" * 64,
    }
    path = _write_config(tmp_path, config)
    monkeypatch.setattr(publication, "_durable_paths", lambda: ())
    result = publication.plan_publication_correction(path)
    assert result["mmseqs_executed"] is False
    assert result["scientific_artifacts_scanned"] is False
    assert not Path(config["output_dir"]).exists()
    Path(config["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        publication.plan_publication_correction(path)


def test_repository_source_publication_defect_is_pinned_and_immutable() -> None:
    config = yaml.safe_load(Path("configs/e007_split_homology_publication_correction_v1.yaml").read_text())
    source = Path(config["source_output_dir"])
    report = source / "report.json"
    protocol = source / "protocol.json"
    assert report.read_bytes() == protocol.read_bytes()
    assert publication.sha256_file(report) == config["source_publication"]["report_sha256"]
    assert report.stat().st_size == protocol.stat().st_size == 108_268
