from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryVocabulary,
)
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.codesign_pilot import (
    DEFAULT_PAIR_BUDGET,
    TRAINING_ARMS,
    VALIDATION_MODES,
    PilotInterrupted,
    _dataset_identity,
    _sha256_file,
    accumulation_for_length,
    build_paired_plan,
    curriculum_stages,
    finalize_completed_heartbeat,
    run_bounded_pilot,
)


def _config() -> dict:
    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    config["pilot"] = {
        "activation_checkpointing": True,
        "max_rss_mib": 4096,
        "max_cuda_memory_mib": 8192,
        "checkpoint_frequency": 1,
        "maximum_optimizer_updates": 2,
        "validation_panel_size": 2,
        "synthetic_lengths": [8, 12],
        "curriculum": [
            {"maximum_length": 8, "optimizer_updates": 1},
            {"maximum_length": 16, "optimizer_updates": 1},
        ],
        "pair_budget_accumulation": [{"maximum_padded_length": 64, "microbatches": 2}],
    }
    return config


def _model_hash(path: Path) -> dict[str, torch.Tensor]:
    return load_checkpoint(path)["model"]


def _partitioned_dataset(tmp_path: Path) -> tuple[dict, Path, Path]:
    directory = tmp_path / "absolute dataset directory with spaces"
    train_directory = directory / "eligible_train.parquet"
    validation_directory = directory / "eligible_validation.parquet"
    train_directory.mkdir(parents=True)
    validation_directory.mkdir()
    train_partition = train_directory / "part-000000.parquet"
    validation_partition = validation_directory / "part-000000.parquet"
    pq.write_table(pa.table({"sample_id": [f"train-{index}" for index in range(2295)]}), train_partition)
    pq.write_table(pa.table({"sample_id": ["validation-0"]}), validation_partition)
    records = [
        {
            "dataset": "eligible_train",
            "partition_index": 0,
            "path": "eligible_train.parquet/part-000000.parquet",
            "row_count": 2295,
            "sha256": _sha256_file(train_partition),
        },
        {
            "dataset": "eligible_validation",
            "partition_index": 0,
            "path": "eligible_validation.parquet/part-000000.parquet",
            "row_count": 1,
            "sha256": _sha256_file(validation_partition),
        },
    ]
    protocol = {
        "status": "completed",
        "schema_version": PAIRING_SCHEMA_VERSION,
        "input_hashes_preserved": True,
        "failure_count": 0,
        "validated_membership_counts": {"eligible_train_count": 2295, "eligible_validation_count": 1},
        "partitions": records,
    }
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "schema.json").write_text(json.dumps({"schema_version": PAIRING_SCHEMA_VERSION}))
    (directory / "vocabulary.json").write_text(json.dumps(SequenceGeometryVocabulary().as_dict()))
    (directory / "input_hashes.sha256").write_text("upstream input inventory\n")
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"mode": "scale", "scale": 50.0}))
    config = {
        "dataset": {
            "directory": str(directory.resolve()),
            "train_dataset": "eligible_train.parquet",
            "validation_dataset": "eligible_validation.parquet",
            "immutable": True,
        },
        "normalization_file": str(normalization),
    }
    return config, directory, train_partition


def _rewrite_protocol(directory: Path, mutate) -> None:
    path = directory / "protocol.json"
    protocol = json.loads(path.read_text())
    mutate(protocol)
    path.write_text(json.dumps(protocol))


def test_pair_budget_and_curriculum_contracts() -> None:
    assert [accumulation_for_length(value, DEFAULT_PAIR_BUDGET) for value in (64, 65, 128, 129, 500)] == [
        16,
        4,
        4,
        1,
        1,
    ]
    assert accumulation_for_length(504, DEFAULT_PAIR_BUDGET) == 1
    stages = curriculum_stages(_config())
    assert [(stage.maximum_length, stage.optimizer_updates) for stage in stages] == [(8, 1), (16, 1)]


def test_dataset_identity_verifies_exact_relative_partition_under_absolute_path_with_spaces(tmp_path: Path) -> None:
    config, directory, train_partition = _partitioned_dataset(tmp_path)
    expected = json.loads((directory / "protocol.json").read_text())["partitions"][0]

    identity = _dataset_identity(config, synthetic=False)

    assert expected["path"] == "eligible_train.parquet/part-000000.parquet"
    assert expected["row_count"] == 2295
    assert expected["sha256"] == _sha256_file(train_partition)
    assert identity["files"][str(train_partition.resolve())] == expected["sha256"]


def test_dataset_identity_rejects_altered_partition_bytes(tmp_path: Path) -> None:
    config, _, train_partition = _partitioned_dataset(tmp_path)
    pq.write_table(pa.table({"sample_id": [f"changed-{index}" for index in range(2295)]}), train_partition)
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        _dataset_identity(config, synthetic=False)


def test_dataset_identity_verifies_non_loader_partition_records_too(tmp_path: Path) -> None:
    config, directory, _ = _partitioned_dataset(tmp_path)
    other_directory = directory / "all_pairs.parquet"
    other_directory.mkdir()
    other_partition = other_directory / "part-000000.parquet"
    pq.write_table(pa.table({"sample_id": ["all-pair"]}), other_partition)

    def add_other(protocol: dict) -> None:
        protocol["partitions"].append(
            {
                "dataset": "all_pairs",
                "partition_index": 0,
                "path": "all_pairs.parquet/part-000000.parquet",
                "row_count": 1,
                "sha256": _sha256_file(other_partition),
            }
        )

    _rewrite_protocol(directory, add_other)
    assert _dataset_identity(config, synthetic=False)["files"][str(other_partition.resolve())] == _sha256_file(
        other_partition
    )
    _rewrite_protocol(directory, lambda value: value["partitions"][-1].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        _dataset_identity(config, synthetic=False)


def test_dataset_identity_rejects_incorrect_sha_and_row_count(tmp_path: Path) -> None:
    config, directory, _ = _partitioned_dataset(tmp_path)
    _rewrite_protocol(directory, lambda value: value["partitions"][0].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="SHA-256 contradiction"):
        _dataset_identity(config, synthetic=False)

    config, directory, _ = _partitioned_dataset(tmp_path / "row-count")
    _rewrite_protocol(directory, lambda value: value["partitions"][0].update(row_count=2294))
    with pytest.raises(ValueError, match="row-count contradiction"):
        _dataset_identity(config, synthetic=False)


def test_dataset_identity_rejects_duplicate_and_missing_partition_records(tmp_path: Path) -> None:
    config, directory, _ = _partitioned_dataset(tmp_path)
    _rewrite_protocol(directory, lambda value: value["partitions"].append(dict(value["partitions"][0])))
    with pytest.raises(ValueError, match="Duplicate pairing dataset partition path"):
        _dataset_identity(config, synthetic=False)

    config, directory, _ = _partitioned_dataset(tmp_path / "missing")
    _rewrite_protocol(directory, lambda value: value["partitions"].pop())
    with pytest.raises(ValueError, match="missing from protocol"):
        _dataset_identity(config, synthetic=False)


def test_dataset_identity_rejects_partition_path_traversal(tmp_path: Path) -> None:
    config, directory, _ = _partitioned_dataset(tmp_path)
    outside = tmp_path / "outside.parquet"
    pq.write_table(pa.table({"sample_id": ["outside"]}), outside)

    def add_escape(protocol: dict) -> None:
        protocol["partitions"].append(
            {
                "dataset": "other",
                "partition_index": 0,
                "path": "../outside.parquet",
                "row_count": 1,
                "sha256": _sha256_file(outside),
            }
        )

    _rewrite_protocol(directory, add_escape)
    with pytest.raises(ValueError, match="escapes dataset directory"):
        _dataset_identity(config, synthetic=False)


def test_dataset_identity_never_materializes_full_parquet_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, _ = _partitioned_dataset(tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("pandas materialization used"))
    monkeypatch.setattr(pq, "read_table", lambda *args, **kwargs: pytest.fail("Arrow table materialization used"))

    identity = _dataset_identity(config, synthetic=False)

    assert identity["kind"] == "sequence_geometry_pairing_v1"


def test_paired_plan_shares_order_seeds_and_transitions() -> None:
    plan = build_paired_plan(_config(), synthetic=True)
    assert [item["stage_index"] for item in plan["updates"]] == [0, 1]
    assert [item["stage_maximum_length"] for item in plan["updates"]] == [8, 16]
    assert plan == build_paired_plan(_config(), synthetic=True)
    assert all(len(item["microbatches"]) == 2 for item in plan["updates"])


def test_synthetic_pilot_uses_shared_initialization_and_all_validation_modes(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    report = run_bounded_pilot(_config(), output_dir=output, synthetic=True)
    assert report["status"] == "completed"
    assert [item["arm"] for item in report["training_arms"]] == list(TRAINING_ARMS)
    assert {item["initialization_sha256"] for item in report["training_arms"]} == {
        report["shared_initialization_sha256"]
    }
    assert {item["validation_mode"] for item in report["validation_records"]} == set(VALIDATION_MODES)
    assert report["dataset_inputs_unchanged"] is True
    heartbeat = json.loads((output / "heartbeat.json").read_text())
    assert heartbeat["status"] == "completed"
    assert heartbeat["completed_utc"] == report["completed_utc"]
    assert heartbeat["final_optimizer_step"] == 2
    assert heartbeat["final_arm"] == "learned_geometry_gating"
    assert heartbeat["final_stage"] == 2
    assert heartbeat["optimizer_step"] == 2
    assert heartbeat["arm"] == "learned_geometry_gating"
    assert heartbeat["curriculum_stage"] == 2
    assert heartbeat["summary_path"] == str(output / "summary.json")
    assert heartbeat["summary_sha256"] == _sha256_file(output / "summary.json")
    assert (output / "metrics" / "sequence_only.jsonl").is_file()
    assert (output / "metrics" / "learned_geometry_gating.jsonl").is_file()
    assert (output / "metrics" / "learned_validation.jsonl").is_file()
    assert not list(output.rglob("*.inprogress"))
    paired_records = []
    for arm in TRAINING_ARMS:
        with (output / "metrics" / f"{arm}.jsonl").open() as handle:
            paired_records.append(
                [
                    {
                        key: record[key]
                        for key in (
                            "sample_id",
                            "stochastic_seed",
                            "timestep",
                            "masked_sequence_sha256",
                            "masked_token_mask_sha256",
                            "geometry_noise_sha256",
                        )
                    }
                    for record in map(json.loads, handle)
                    if record["record_type"] == "microbatch"
                ]
            )
    assert paired_records[0] == paired_records[1]
    with pytest.raises(FileExistsError, match="Refusing to overwrite completed"):
        run_bounded_pilot(_config(), output_dir=output, synthetic=True)


def test_interrupted_pilot_resumes_to_identical_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    reference = tmp_path / "reference"
    run_bounded_pilot(config, output_dir=reference, synthetic=True)

    import protein_distance_diffusion.training.codesign_pilot as pilot_module

    original_guard = pilot_module._guard_memory
    calls = 0

    def interrupt_after_first_microbatch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise PilotInterrupted("injected SIGINT")
        return original_guard(*args, **kwargs)

    interrupted = tmp_path / "interrupted"
    monkeypatch.setattr(pilot_module, "_guard_memory", interrupt_after_first_microbatch)
    with pytest.raises(PilotInterrupted, match="injected"):
        run_bounded_pilot(config, output_dir=interrupted, synthetic=True)
    assert json.loads((interrupted / "incomplete.json").read_text())["status"] == "interrupted"
    interrupted_heartbeat = json.loads((interrupted / "heartbeat.json").read_text())
    assert interrupted_heartbeat["status"] == "interrupted"
    assert interrupted_heartbeat["error_type"] == "PilotInterrupted"
    assert interrupted_heartbeat["latest_resumable_checkpoint"].endswith("sequence_only_latest.pt")
    assert not (interrupted / "metrics" / "sequence_only.jsonl").exists()
    assert (interrupted / "metrics" / "sequence_only.jsonl.inprogress").exists()

    monkeypatch.setattr(pilot_module, "_guard_memory", original_guard)
    resumed = run_bounded_pilot(config, output_dir=interrupted, synthetic=True, resume=True)
    assert resumed["status"] == "completed"
    assert json.loads((interrupted / "heartbeat.json").read_text())["status"] == "completed"
    for arm in TRAINING_ARMS:
        expected = _model_hash(reference / "checkpoints" / f"{arm}_latest.pt")
        actual = _model_hash(interrupted / "checkpoints" / f"{arm}_latest.pt")
        assert expected.keys() == actual.keys()
        assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_memory_failure_publishes_only_incomplete_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import protein_distance_diffusion.training.codesign_pilot as pilot_module

    monkeypatch.setattr(pilot_module, "_rss_mib", lambda: 5000.0)
    monkeypatch.setattr(pilot_module, "_peak_rss_mib", lambda: 5000.0)
    output = tmp_path / "memory-failure"
    with pytest.raises(MemoryError, match="current=5000.0 MiB"):
        run_bounded_pilot(_config(), output_dir=output, synthetic=True)
    incomplete = json.loads((output / "incomplete.json").read_text())
    assert incomplete["status"] == "memory_limit_exceeded"
    heartbeat = json.loads((output / "heartbeat.json").read_text())
    assert heartbeat["status"] == "memory_limit_exceeded"
    assert heartbeat["error_type"] == "MemoryError"
    assert heartbeat["latest_resumable_checkpoint"].endswith("sequence_only_latest.pt")
    assert not (output / "summary.json").exists()
    assert not list(output.glob("metrics/*.jsonl"))


def test_exception_finalizes_failed_heartbeat_with_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import protein_distance_diffusion.training.codesign_pilot as pilot_module

    def fail_validation(*args, **kwargs):
        raise RuntimeError("injected validation failure")

    monkeypatch.setattr(pilot_module, "_validate_checkpoints", fail_validation)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected validation failure"):
        run_bounded_pilot(_config(), output_dir=output, synthetic=True)
    heartbeat = json.loads((output / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["error_type"] == "RuntimeError"
    checkpoint = Path(heartbeat["latest_resumable_checkpoint"])
    assert checkpoint.is_file()
    assert heartbeat["latest_resumable_checkpoint_sha256"] == _sha256_file(checkpoint)


def test_completed_summary_is_authoritative_and_read_only_repair_changes_only_heartbeat(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "stale-heartbeat"
    run_bounded_pilot(config, output_dir=output, synthetic=True)
    stale = {"status": "running", "optimizer_step": 1}
    (output / "heartbeat.json").write_text(json.dumps(stale))
    with pytest.raises(FileExistsError, match="Refusing to overwrite completed"):
        run_bounded_pilot(config, output_dir=output, synthetic=True, resume=True)
    assert json.loads((output / "heartbeat.json").read_text()) == stale
    before = {
        str(path.relative_to(output)): _sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "heartbeat.json"
    }

    heartbeat = finalize_completed_heartbeat(config, output_dir=output, synthetic=True)

    after = {
        str(path.relative_to(output)): _sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "heartbeat.json"
    }
    assert before == after
    assert heartbeat["status"] == "completed"
    assert heartbeat["repair_finalization"] is True
    assert heartbeat["arm"] == "learned_geometry_gating"
    assert heartbeat["curriculum_stage"] == 2
    assert heartbeat["optimizer_step"] == 2
    assert heartbeat["summary_sha256"] == _sha256_file(output / "summary.json")
    assert set(heartbeat["checkpoint_sha256"]) == set(TRAINING_ARMS)


def test_repair_fills_incomplete_completed_position_and_is_idempotent(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "incomplete-completed"
    run_bounded_pilot(config, output_dir=output, synthetic=True)
    heartbeat_path = output / "heartbeat.json"
    incomplete = json.loads(heartbeat_path.read_text())
    for field in ("arm", "curriculum_stage", "optimizer_step"):
        incomplete[field] = None
    heartbeat_path.write_text(json.dumps(incomplete))

    repaired = finalize_completed_heartbeat(config, output_dir=output, synthetic=True)

    assert repaired["status"] == "completed"
    assert repaired["arm"] == "learned_geometry_gating"
    assert repaired["curriculum_stage"] == 2
    assert repaired["optimizer_step"] == 2
    first_hash = _sha256_file(heartbeat_path)
    first_mtime = heartbeat_path.stat().st_mtime_ns

    repeated = finalize_completed_heartbeat(config, output_dir=output, synthetic=True)

    assert repeated == repaired
    assert _sha256_file(heartbeat_path) == first_hash
    assert heartbeat_path.stat().st_mtime_ns == first_mtime


def test_fully_valid_normal_heartbeat_is_an_idempotent_repair_noop(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "already-valid"
    run_bounded_pilot(config, output_dir=output, synthetic=True)
    heartbeat_path = output / "heartbeat.json"
    original = json.loads(heartbeat_path.read_text())
    original_hash = _sha256_file(heartbeat_path)
    original_mtime = heartbeat_path.stat().st_mtime_ns

    result = finalize_completed_heartbeat(config, output_dir=output, synthetic=True)

    assert result == original
    assert _sha256_file(heartbeat_path) == original_hash
    assert heartbeat_path.stat().st_mtime_ns == original_mtime


def test_heartbeat_repair_rejects_checkpoint_hash_change_without_touching_heartbeat(tmp_path: Path) -> None:
    config = _config()
    output = tmp_path / "checkpoint-change"
    run_bounded_pilot(config, output_dir=output, synthetic=True)
    stale = {"status": "running"}
    (output / "heartbeat.json").write_text(json.dumps(stale))
    checkpoint = output / "checkpoints" / "sequence_only_latest.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        finalize_completed_heartbeat(config, output_dir=output, synthetic=True)

    assert json.loads((output / "heartbeat.json").read_text()) == stale
