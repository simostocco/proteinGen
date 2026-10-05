from __future__ import annotations

import copy
import hashlib
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import protein_distance_diffusion.training.e007_coordinate_real_continuation as continuation

CONFIG_PATH = Path("configs/e007_coordinate_real_continuation_to_10000_v1.yaml")


def _config() -> dict:
    config = continuation._load_config(CONFIG_PATH)
    config["_sha256"] = continuation.sha256_file(CONFIG_PATH)
    return config


def _step1045_recovery_fixture(tmp_path: Path, config: dict | None = None) -> tuple[Path, dict]:
    config = copy.deepcopy(config or _config())
    staging = tmp_path / ".continuation.inprogress"
    checkpoints = staging / "checkpoints"
    checkpoints.mkdir(parents=True)
    payload = {
        "version": continuation.VERSION,
        "continuation_configuration_sha256": config["_sha256"],
        "source_checkpoint_sha256": config["source"]["checkpoint_sha256"],
        "protected_hashes_sha256": "fixture-protected-hashes",
        "model": torch.nn.Linear(2, 2).state_dict(),
        "optimizer": {"state": {0: {"step": torch.tensor(1045)}}, "param_groups": []},
        "scheduler": {"last_epoch": 1045, "_step_count": 1046},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "optimizer_update": 1045,
        "sampler_cursor": 1045,
        "samples_processed": 1881,
        "valid_residues_processed": 275394,
        "successful_optimizer_boundary": True,
        "last_committed_metric_update": 1045,
        "last_memory_gate_passed": True,
        **continuation.NON_AUTHORIZING,
    }
    checkpoint = checkpoints / "latest.pt"
    continuation._atomic_torch(checkpoint, payload)
    continuation._atomic_json(
        checkpoints / "latest.json",
        {
            "optimizer_update": 1045,
            "sha256": continuation.sha256_file(checkpoint),
            "recovery_only": True,
        },
    )
    continuation._append_jsonl_fsync(
        staging / "metrics.jsonl",
        {"global_optimizer_update": 1045, "memory_gate_passed": True},
    )
    continuation._atomic_json(
        staging / "heartbeat.json",
        {"status": "failed", "optimizer_update": 1045, "resumable": True},
    )
    continuation._append_jsonl_fsync(
        staging / "allocator_cleanup.jsonl",
        {"optimizer_update": 1045, "event_status": "passed"},
    )
    continuation._atomic_json(staging / "panel_manifest.json", {"train": {}, "validation": {}})
    return staging, config


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: object) -> tuple[float, float]:
    random_value = random.random()
    numpy_value = float(np.random.random())
    x = torch.randn(4, 3)
    optimizer.zero_grad(set_to_none=True)
    model(x).square().mean().backward()
    optimizer.step()
    scheduler.step()
    return random_value, numpy_value


def test_configuration_preserves_exact_phase3f_scientific_contract() -> None:
    config = _config()
    source = continuation._source_scientific_config(config, extended=False)
    extended = continuation._source_scientific_config(config, extended=True)
    for key in (
        "model",
        "objective",
        "optimizer",
        "numerics",
        "batch_regimes",
        "length_strata",
        "diffusion_steps",
        "coordinate_scale_angstrom",
        "dataset",
        "clean_validation",
    ):
        assert extended[key] == source[key]
    assert source["expected_parameter_count"] == 7_586_505
    assert extended["successful_optimizer_updates"] == 10_000


def test_global_step_and_checkpoint_schedules_are_exact() -> None:
    schedule = continuation.continuation_schedule(_config())
    assert schedule["recovery"] == list(range(1500, 10001, 500))
    assert schedule["validation"] == list(range(2000, 10001, 1000))
    assert schedule["sampling"] == [2500, 5000, 7500, 10000]
    assert schedule["immutable"] == [2000, 2500, 3000, 4000, 5000, 6000, 7000, 7500, 8000, 9000, 10000]
    assert 1000 not in schedule["recovery"]


def test_source_step1000_evaluation_is_hash_pinned() -> None:
    config = _config()
    assert continuation._verify_source_files(config)["evaluations"] == config["source"]["evaluations_sha256"]


def test_fixed_sampling_noise_is_checkpoint_independent_and_has_ten_records() -> None:
    records = continuation.monitor_seed_records(_config())
    assert len(records) == 10
    assert len({row["seed"] for row in records}) == 10
    assert len({row["noise_identity_sha256"] for row in records}) == 10
    assert continuation.monitor_seed_records(_config()) == records
    for update in _config()["sampling_monitor_updates"]:
        assert [(update, row["length"], row["sample_index"], row["seed"]) for row in records] == [
            (update, row["length"], row["sample_index"], row["seed"]) for row in records
        ]


def test_training_prefix_and_fixed_validation_are_verified() -> None:
    source_config = continuation._source_scientific_config(_config(), extended=False)
    counts = continuation.planned_batch_accounting(source_config)["planned_samples_by_stratum"]
    source_panel = {"train": {}, "validation": {}}
    selected = {"train": {}, "validation": {}}
    for name, count in counts.items():
        source_panel["train"][name] = [f"{name}-{index}" for index in range(count + 8)]
        source_panel["validation"][name] = [f"v-{name}-{index}" for index in range(16)]
        selected["train"][name] = [
            {"sample_id": value}
            for value in source_panel["train"][name] + [f"new-{name}-{index}" for index in range(20)]
        ]
        selected["validation"][name] = [{"sample_id": value} for value in source_panel["validation"][name]]
    assert set(continuation.verify_stream_prefix(source_panel, selected, source_config)) == set(counts)
    selected["train"]["20-64"][0]["sample_id"] = "changed"
    with pytest.raises(ValueError, match="training-stream prefix"):
        continuation.verify_stream_prefix(source_panel, selected, source_config)


def test_partial_state_and_weights_only_loading_are_refused() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    with pytest.raises(ValueError, match="partial state restoration"):
        continuation.restore_full_state({"model": model.state_dict()}, model, optimizer, scheduler)


def test_source_checkpoint_protected_identity_is_strict() -> None:
    payload = {"protected_hashes_sha256": continuation._canonical_sha({"one": "hash"})}
    continuation.verify_source_checkpoint_protection(payload, {"hashes": {"one": "hash"}})
    with pytest.raises(ValueError, match="protected-input identity"):
        continuation.verify_source_checkpoint_protection(payload, {"hashes": {"one": "changed"}})


def test_full_state_resume_matches_uninterrupted_execution(tmp_path: Path) -> None:
    config = _config()
    protected = "protected"
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    uninterrupted = torch.nn.Linear(3, 2)
    interrupted = copy.deepcopy(uninterrupted)
    optimizer_a = torch.optim.AdamW(uninterrupted.parameters(), lr=0.01)
    optimizer_b = torch.optim.AdamW(interrupted.parameters(), lr=0.01)
    scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lambda _: 1.0)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lambda _: 1.0)

    _step(uninterrupted, optimizer_a, scheduler_a)
    payload = continuation.continuation_checkpoint_payload(
        model=uninterrupted,
        optimizer=optimizer_a,
        scheduler=scheduler_a,
        update=1001,
        samples_processed=1804,
        valid_residues_processed=263700,
        config=config,
        protected_hashes_sha256=protected,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    expected_random = _step(uninterrupted, optimizer_a, scheduler_a)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    continuation.restore_full_state(payload, interrupted, optimizer_b, scheduler_b)
    observed_random = _step(interrupted, optimizer_b, scheduler_b)
    assert observed_random == expected_random
    for first, second in zip(uninterrupted.parameters(), interrupted.parameters(), strict=True):
        assert torch.equal(first, second)
    assert optimizer_a.state_dict()["state"].keys() == optimizer_b.state_dict()["state"].keys()
    assert scheduler_a.state_dict() == scheduler_b.state_dict()


def test_sampling_pareto_has_no_scalar_score() -> None:
    records = [
        {
            "optimizer_update": 2500,
            "adjacent_reference_error_angstrom": 0.4,
            "radius_of_gyration_reference_relative_error": 0.2,
            "clash_reference_error": 0.1,
            "contact_density_reference_error": 0.2,
        },
        {
            "optimizer_update": 5000,
            "adjacent_reference_error_angstrom": 0.3,
            "radius_of_gyration_reference_relative_error": 0.3,
            "clash_reference_error": 0.1,
            "contact_density_reference_error": 0.1,
        },
        {
            "optimizer_update": 7500,
            "adjacent_reference_error_angstrom": 0.6,
            "radius_of_gyration_reference_relative_error": 0.4,
            "clash_reference_error": 0.2,
            "contact_density_reference_error": 0.3,
        },
    ]
    result = continuation.nondominated_sampling(records)
    assert [row["optimizer_update"] for row in result] == [2500, 5000]
    assert all("score" not in row for row in result)


def test_atomic_staging_paths_are_rewritten_for_final_publication(tmp_path: Path) -> None:
    staging = tmp_path / ".run.inprogress"
    output = tmp_path / "run"
    payload = {
        "artifact_path": str(staging / "samples" / "sample.npz"),
        "nested": [str(staging / "checkpoints" / "step.pt"), "unrelated"],
    }
    published = continuation.published_paths(payload, staging, output)
    assert published == {
        "artifact_path": str(output / "samples" / "sample.npz"),
        "nested": [str(output / "checkpoints" / "step.pt"), "unrelated"],
    }


def test_source_checkpoint_and_protected_inputs_are_pinned_read_only() -> None:
    config = _config()
    before = continuation._verify_source_files(config)
    payload = continuation.inspect_checkpoint(config["source"]["checkpoint_path"], config)
    after = continuation._verify_source_files(config)
    assert before == after
    assert payload["optimizer_update"] == payload["sampler_cursor"] == 1000
    assert payload["samples_processed"] == 1800
    assert payload["valid_residues_processed"] == 263579


def test_plan_is_model_free_and_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    config["output_dir"] = str(tmp_path / "future")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(
        continuation, "EquivariantPairCoordinateUNet", lambda **_: (_ for _ in ()).throw(AssertionError())
    )
    plan = continuation.plan_coordinate_continuation(path)
    assert plan["model_created"] is False
    assert plan["optimizer_created"] is False
    assert not Path(config["output_dir"]).exists()
    Path(config["output_dir"]).mkdir()
    with pytest.raises(FileExistsError):
        continuation.plan_coordinate_continuation(path)


def test_configuration_hash_changes_on_unrelated_tampering(tmp_path: Path) -> None:
    original = yaml.safe_load(CONFIG_PATH.read_text())
    changed = copy.deepcopy(original)
    changed["memory"]["maximum_cuda_allocated_mib"] += 1
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(original))
    second.write_text(yaml.safe_dump(changed))
    assert hashlib.sha256(first.read_bytes()).hexdigest() != hashlib.sha256(second.read_bytes()).hexdigest()


def test_failed_update1045_checkpoint_is_atomic_and_next_update_is_1046(tmp_path: Path) -> None:
    staging, config = _step1045_recovery_fixture(tmp_path)
    checkpoint = staging / "checkpoints/latest.pt"
    metadata = json.loads((staging / "checkpoints/latest.json").read_text())
    assert metadata["sha256"] == continuation.sha256_file(checkpoint)
    assert not list((staging / "checkpoints").glob("*.tmp"))
    payload = continuation.inspect_checkpoint(checkpoint, config, continuation=True)
    diagnosis = continuation.diagnose_resume_boundary(
        payload,
        continuation._read_metric_rows(staging / "metrics.jsonl"),
    )
    assert diagnosis["atomic_optimizer_boundary"] is True
    assert diagnosis["optimizer_state_steps"] == [1045]
    assert diagnosis["scheduler_last_epoch"] == 1045
    assert diagnosis["missing_metric_updates"] == []
    assert diagnosis["last_memory_gate_passed"] is True
    assert diagnosis["metric_reconstruction_permitted"] is False
    assert diagnosis["next_optimizer_update"] == 1046
    assert diagnosis["samples_processed"] == 1881
    assert diagnosis["valid_residues_processed"] == 275394


def test_protected_failed_staging_hashes_are_unchanged(tmp_path: Path) -> None:
    staging, config = _step1045_recovery_fixture(tmp_path)
    before = continuation._directory_fingerprint(staging)
    payload = continuation.inspect_checkpoint(staging / "checkpoints/latest.pt", config, continuation=True)
    continuation.diagnose_resume_boundary(payload, continuation._read_metric_rows(staging / "metrics.jsonl"))
    assert continuation._directory_fingerprint(staging) == before
    assert (staging / "heartbeat.json").is_file()


def test_allocator_cleanup_releases_reserved_cache_without_consuming_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        [
            {
                "current_cuda_allocated_mib": 180.0,
                "current_cuda_reserved_mib": 7046.0,
                "phase_peak_cuda_allocated_mib": 4459.0,
                "phase_peak_cuda_reserved_mib": 7046.0,
            },
            {
                "current_cuda_allocated_mib": 180.0,
                "current_cuda_reserved_mib": 340.0,
                "phase_peak_cuda_allocated_mib": 180.0,
                "phase_peak_cuda_reserved_mib": 340.0,
            },
        ]
    )
    calls = []
    monkeypatch.setattr(continuation, "_cuda_memory_snapshot", lambda _device: next(snapshots))
    monkeypatch.setattr(continuation.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    rng_before = torch.get_rng_state().clone()
    event = continuation.allocator_cleanup(
        torch.device("cuda"),
        active_computation=False,
        padded_length=504,
    )
    assert event["reserved_mib_released"] == pytest.approx(6706.0)
    assert event["padded_length"] == 504
    assert calls == ["gc", "empty_cache"]
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_allocator_cleanup_is_forbidden_during_live_computation() -> None:
    with pytest.raises(RuntimeError, match="forbidden during active computation"):
        continuation.allocator_cleanup(
            torch.device("cuda"),
            active_computation=True,
            padded_length=500,
        )


def test_cleanup_boundary_does_not_change_next_update_or_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(77)
    reference = torch.nn.Linear(3, 2)
    cleaned = copy.deepcopy(reference)
    optimizer_reference = torch.optim.AdamW(reference.parameters(), lr=0.01)
    optimizer_cleaned = torch.optim.AdamW(cleaned.parameters(), lr=0.01)
    scheduler_reference = torch.optim.lr_scheduler.LambdaLR(optimizer_reference, lambda _: 1.0)
    scheduler_cleaned = torch.optim.lr_scheduler.LambdaLR(optimizer_cleaned, lambda _: 1.0)
    torch_state = torch.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    expected = _step(reference, optimizer_reference, scheduler_reference)
    torch.set_rng_state(torch_state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    snapshots = iter(
        [
            {
                "current_cuda_allocated_mib": 100.0,
                "current_cuda_reserved_mib": 6000.0,
                "phase_peak_cuda_allocated_mib": 100.0,
                "phase_peak_cuda_reserved_mib": 6000.0,
            },
            {
                "current_cuda_allocated_mib": 100.0,
                "current_cuda_reserved_mib": 120.0,
                "phase_peak_cuda_allocated_mib": 100.0,
                "phase_peak_cuda_reserved_mib": 120.0,
            },
        ]
    )
    monkeypatch.setattr(continuation, "_cuda_memory_snapshot", lambda _device: next(snapshots))
    monkeypatch.setattr(continuation.gc, "collect", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    continuation.allocator_cleanup(torch.device("cuda"), active_computation=False, padded_length=500)
    observed = _step(cleaned, optimizer_cleaned, scheduler_cleaned)
    assert observed == expected
    for first, second in zip(reference.parameters(), cleaned.parameters(), strict=True):
        assert torch.equal(first, second)
    assert scheduler_reference.state_dict() == scheduler_cleaned.state_dict()
    for key in optimizer_reference.state_dict()["state"]:
        for field, value in optimizer_reference.state_dict()["state"][key].items():
            other = optimizer_cleaned.state_dict()["state"][key][field]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_memory_limits_and_transaction_order_are_unchanged() -> None:
    config = _config()
    assert config["memory"] == {
        "maximum_cuda_allocated_mib": 6144,
        "maximum_cuda_reserved_mib": 7680,
    }
    source = inspect.getsource(continuation.run_coordinate_continuation)
    assert source.index("_append_jsonl_fsync(metrics_path, metric)") < source.index(
        'raise MemoryError(f"E007 continuation CUDA envelope exceeded: {phase_memory}")'
    )
    assert source.index("_append_jsonl_fsync(metrics_path, metric)") < source.index('event_type="post_update"')
    assert source.index("del prediction, loss, corruption, prepared, gradients, rows") < source.index(
        'event_type="post_update"'
    )
    assert source.index("last_committed_metric_update >= update") > source.index("except BaseException as error")


def test_allocator_lifecycle_policy_cleans_before_and_after_only_long_updates() -> None:
    assert continuation.allocator_cleanup_actions(continuation.ALLOCATOR_POLICY, "385-500") == (
        True,
        True,
    )
    assert continuation.allocator_cleanup_actions(continuation.ALLOCATOR_POLICY, "257-384") == (
        False,
        False,
    )
    assert continuation.allocator_cleanup_actions("pre_medium_and_long", "257-384") == (
        True,
        False,
    )
    assert continuation.allocator_cleanup_actions("pre_medium_and_long", "385-500") == (
        True,
        False,
    )


def test_exact_second_failure_padded_sequence_and_repeated_cycles() -> None:
    assert list(continuation.DIAGNOSTIC_PADDED_LENGTH_SEQUENCE) == [
        64,
        96,
        176,
        264,
        448,
        64,
        128,
        208,
        312,
        488,
        64,
        128,
        192,
        264,
        400,
        56,
        96,
        144,
        352,
        448,
        64,
        104,
        248,
        360,
        472,
    ]
    sequence = continuation._diagnostic_sequence(2)
    assert sequence[-10:] == [64, 104, 248, 360, 472] * 2


def test_captured_phase_peak_fails_even_if_post_cleanup_is_small() -> None:
    config = _config()
    memory = {
        "current_cuda_allocated_mib": 185.0,
        "current_cuda_reserved_mib": 200.0,
        "phase_peak_cuda_allocated_mib": 5000.0,
        "phase_peak_cuda_reserved_mib": 7680.0001,
    }
    assert continuation.memory_gate_passes(memory, config) is False
    memory["phase_peak_cuda_reserved_mib"] = 7680.0
    assert continuation.memory_gate_passes(memory, config) is True


def _cleanup_event(
    reserved_mib: float,
    *,
    allocated_mib: float = 151.0,
    phase_gate_passed: bool = True,
) -> dict:
    snapshot = {
        "current_cuda_allocated_mib": allocated_mib,
        "current_cuda_reserved_mib": reserved_mib,
        "phase_peak_cuda_allocated_mib": 4942.0,
        "phase_peak_cuda_reserved_mib": 6462.0,
    }
    return {
        "event_type": "post_update",
        "optimizer_update": 1045,
        "length_stratum": "385-500",
        "padded_length": 472,
        "before": snapshot,
        "after": snapshot,
        "allocated_mib_released": 0.0,
        "reserved_mib_released": 5150.0,
        "captured_memory_gate_passed": phase_gate_passed,
    }


def test_post_cleanup_ceiling_is_derived_and_inclusive() -> None:
    config = _config()
    assert continuation.post_cleanup_reserved_ceiling_mib(config) == 1536.0
    observed = continuation.assess_allocator_cleanup(_cleanup_event(1312.0), config, prior_post_cleanup_reserved_mib=[])
    boundary = continuation.assess_allocator_cleanup(_cleanup_event(1536.0), config, prior_post_cleanup_reserved_mib=[])
    above = continuation.assess_allocator_cleanup(_cleanup_event(1536.01), config, prior_post_cleanup_reserved_mib=[])
    assert observed["event_status"] == "passed"
    assert observed["post_cleanup_reserved_headroom_mib"] == 224.0
    assert boundary["event_status"] == "passed"
    assert above["event_status"] == "failed"
    assert "post_cleanup_reserved_above_derived_headroom_ceiling" in above["failure_reason"]


def test_cleanup_health_checks_persistent_allocation_phase_gate_and_growth() -> None:
    config = _config()
    allocated = continuation.assess_allocator_cleanup(
        _cleanup_event(200.0, allocated_mib=512.01),
        config,
        prior_post_cleanup_reserved_mib=[],
    )
    phase = continuation.assess_allocator_cleanup(
        _cleanup_event(200.0, phase_gate_passed=False),
        config,
        prior_post_cleanup_reserved_mib=[],
    )
    growing = continuation.assess_allocator_cleanup(
        _cleanup_event(600.0),
        config,
        prior_post_cleanup_reserved_mib=[200.0, 400.0],
    )
    stable = continuation.assess_allocator_cleanup(
        _cleanup_event(1312.0),
        config,
        prior_post_cleanup_reserved_mib=[1312.0, 1312.0],
    )
    assert "post_cleanup_allocated_above_persistent_state_ceiling" in allocated["failure_reason"]
    assert "phase_memory_gate_failed" in phase["failure_reason"]
    assert "post_cleanup_reserved_monotonic_growth" in growing["failure_reason"]
    assert stable["event_status"] == "passed"


def test_failed_cleanup_event_is_fsynced_before_enforcement(tmp_path: Path) -> None:
    path = tmp_path / "cleanup.jsonl"
    with pytest.raises(MemoryError, match="reserved_above_derived_headroom"):
        continuation.publish_and_enforce_allocator_cleanup(
            path,
            _cleanup_event(1600.0),
            _config(),
            prior_post_cleanup_reserved_mib=[],
        )
    rows = continuation._read_metric_rows(path)
    assert len(rows) == 1
    assert rows[0]["event_status"] == "failed"
    assert rows[0]["failure_reason"] == "post_cleanup_reserved_above_derived_headroom_ceiling"


def test_checkpoint_metric_memory_gate_consistency_is_validated(tmp_path: Path) -> None:
    staging, _fixture_config = _step1045_recovery_fixture(tmp_path)
    payload = torch.load(staging / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    rows = continuation._read_metric_rows(staging / "metrics.jsonl")
    payload["last_committed_metric_update"] = 1045
    payload["last_memory_gate_passed"] = True
    assert continuation.diagnose_resume_boundary(payload, rows)["next_optimizer_update"] == 1046
    payload["last_memory_gate_passed"] = False
    with pytest.raises(ValueError, match="checkpoint/metric memory-gate contradiction"):
        continuation.diagnose_resume_boundary(payload, rows)


def test_checkpoint_loading_is_explicitly_cpu_first(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    observed = {}
    payload = {
        "version": "e007_coordinate_real_pilot_v1",
        "configuration_sha256": config["source"]["config_sha256"],
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "rng_state": {"python": (), "numpy": (), "torch": torch.zeros(1, dtype=torch.uint8)},
        "optimizer_update": 1000,
        "sampler_cursor": 1000,
        "samples_processed": 1800,
        "valid_residues_processed": 263579,
        "successful_optimizer_boundary": True,
    }

    def fake_load(path: Path, **kwargs: object) -> dict:
        observed.update(kwargs)
        return payload

    monkeypatch.setattr(torch, "load", fake_load)
    assert continuation.inspect_checkpoint("checkpoint.pt", config)["optimizer_update"] == 1000
    assert observed == {"map_location": "cpu", "weights_only": False}


def test_plan_reads_existing_staging_without_mutation(tmp_path: Path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["output_dir"] = str(tmp_path / "continuation")
    config_path = tmp_path / "continuation.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config = continuation._load_config(config_path)
    config["_sha256"] = continuation.sha256_file(config_path)
    fixture, _fixture_config = _step1045_recovery_fixture(tmp_path / "fixture", config)
    staging = Path(raw["output_dir"]).with_name(f".{Path(raw['output_dir']).name}.inprogress")
    fixture.rename(staging)
    before = continuation._directory_fingerprint(staging)
    plan = continuation.plan_coordinate_continuation(config_path)
    assert continuation._directory_fingerprint(staging) == before
    assert plan["existing_staging_resume_diagnosis"]["next_optimizer_update"] == 1046
    assert plan["existing_staging_resume_diagnosis"]["missing_metric_updates"] == []
    assert plan["existing_staging_resume_diagnosis"]["last_memory_gate_passed"] is True
    staging.rename(Path(raw["output_dir"]))
    completed_before = continuation._directory_fingerprint(Path(raw["output_dir"]))
    with pytest.raises(FileExistsError, match="output already exists"):
        continuation.plan_coordinate_continuation(config_path)
    assert continuation._directory_fingerprint(Path(raw["output_dir"])) == completed_before


def test_missing_metric_gap_refuses_more_than_one_update(tmp_path: Path) -> None:
    staging, _fixture_config = _step1045_recovery_fixture(tmp_path)
    payload = torch.load(staging / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    rows = [{"global_optimizer_update": 1043, "memory_gate_passed": True}]
    with pytest.raises(ValueError, match="unbounded metric/checkpoint gap"):
        continuation.diagnose_resume_boundary(payload, rows)


def test_diagnostic_contract_has_no_optimizer_updates() -> None:
    source = inspect.getsource(continuation._diagnostic_case)
    assert "optimizer.step" not in source
    assert '"optimizer_created": True' in source
    assert '"optimizer_updates": 0' in source


def test_completed_cuda_allocator_diagnostic_is_equivalent_and_bounded() -> None:
    path = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/continuation_allocator_diagnostic_v1/report.json"
    )
    assert continuation.sha256_file(path) == "b3c1185bb7a476cd6910d59680c41ee87f23faf2bc65b3004efff076cc6f6ee9"
    report = json.loads(path.read_text())
    assert report["status"] == "completed_non_authorizing"
    assert report["loss_and_gradient_equivalence"] is True
    assert report["cleanup_within_allocated_limit"] is True
    assert report["cleanup_within_reserved_limit"] is True
    assert report["cleanup"]["completed_records"] == 30
    assert report["cleanup"]["optimizer_updates"] == 0
    assert report["cleanup"]["parameters_unchanged"] is True
    assert report["cleanup"]["cpu_rng_unchanged"] is True
    assert report["cleanup"]["cuda_rng_unchanged"] is True
    assert report["cleanup"]["process_peak_cuda_allocated_mib"] == pytest.approx(5234.15478515625)
    assert report["cleanup"]["process_peak_cuda_reserved_mib"] == pytest.approx(6316.0)
    assert len(report["cleanup"]["cleanup_events"]) == 6
    assert all(event["after"]["current_cuda_reserved_mib"] == 122.0 for event in report["cleanup"]["cleanup_events"])
    assert report["protected_inputs_unchanged"] is True


def test_second_cuda_allocator_diagnostic_selects_post_long_cleanup() -> None:
    path = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/continuation_allocator_diagnostic_v2/report.json"
    )
    assert continuation.sha256_file(path) == "d110befb60ef65d4a4a87eec9258cf31f1d151eafc569e5577e85d92a78ddf00"
    report = json.loads(path.read_text())
    assert report["status"] == "completed_non_authorizing"
    assert report["selected_policy"] == continuation.ALLOCATOR_POLICY
    assert report["selection_rationale"] == "least_intrusive_safe_policy"
    assert report["common_record_count"] == 34
    assert report["loss_and_gradient_equivalence"] is True
    assert report["selected_reserved_safety_margin_mib"] == pytest.approx(1178.0)
    preferred = report["post_long_cleanup"]
    alternative = report["pre_medium_and_long_cleanup"]
    assert preferred["process_peak_cuda_allocated_mib"] == pytest.approx(5079.6728515625)
    assert preferred["process_peak_cuda_reserved_mib"] == pytest.approx(6502.0)
    assert alternative["process_peak_cuda_reserved_mib"] == pytest.approx(6596.0)
    assert preferred["optimizer_updates"] == 0
    assert preferred["parameters_unchanged"] is True
    assert preferred["optimizer_unchanged"] is True
    assert preferred["cpu_rng_unchanged"] is True
    assert preferred["cuda_rng_unchanged"] is True
    post_events = [event for event in preferred["cleanup_events"] if event["event_type"] == "post_update"]
    assert post_events
    assert all(event["after"]["current_cuda_reserved_mib"] == 204.0 for event in post_events)
    assert report["protected_inputs_unchanged"] is True


def test_cleanup_headroom_cuda_diagnostic_is_stable_and_non_authorizing() -> None:
    path = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/continuation_allocator_diagnostic_v3/report.json"
    )
    assert continuation.sha256_file(path) == "747cbd916eedcf5e8187950a62f56a68f58cbc93f6096bf395a5d7e6f11183a8"
    report = json.loads(path.read_text())
    assert report["status"] == "completed_non_authorizing"
    assert report["selected_policy"] == continuation.ALLOCATOR_POLICY
    assert report["derived_post_cleanup_reserved_ceiling_mib"] == 1536.0
    assert report["post_long_cleanup_reserved_trajectory_mib"] == [204.0] * 7
    assert report["post_long_cleanup_maximum_reserved_mib"] == 204.0
    assert report["loss_and_gradient_equivalence"] is True
    assert report["protected_inputs_unchanged"] is True
    preferred = report["post_long_cleanup"]
    assert preferred["completed_records"] == 35
    assert preferred["optimizer_updates"] == 0
    assert preferred["parameters_unchanged"] is True
    assert preferred["optimizer_unchanged"] is True
    assert preferred["cpu_rng_unchanged"] is True
    assert preferred["cuda_rng_unchanged"] is True
    post_events = [event for event in preferred["cleanup_events"] if event["event_type"] == "post_update"]
    assert all(event["event_status"] == "passed" for event in post_events)
    assert all(event["post_cleanup_reserved_headroom_mib"] == 1332.0 for event in post_events)
    assert all(event["after"]["cuda_segment_count"] == 22 for event in post_events)


def test_update1046_schedule_and_cursor_do_not_repeat_update1045() -> None:
    source = continuation._source_scientific_config(_config(), extended=False)
    assert continuation.update_stratum(1044, source["length_strata"]) == "257-384"
    assert continuation.update_stratum(1045, source["length_strata"]) == "385-500"
    assert continuation.update_stratum(1046, source["length_strata"]) == "20-64"
    cursors = {item["name"]: 0 for item in source["length_strata"]}
    for update in range(1, 1046):
        name = continuation.update_stratum(update, source["length_strata"])
        cursors[name] += int(continuation._regime_for_stratum(name, source)["physical_batch_size"])
    assert cursors == {
        "20-64": 836,
        "65-128": 418,
        "129-256": 209,
        "257-384": 209,
        "385-500": 209,
    }
