import json

import pytest

from scripts import run_e010_phase4a_training as lifecycle
from scripts import run_e010_phase4a_training_v2 as lifecycle_v2


def test_journal_accepts_only_contiguous_hash_valid_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "STAGING", tmp_path)
    monkeypatch.setattr(lifecycle, "JOURNAL", tmp_path / "journal.jsonl")
    lifecycle.append_event({"global_update": 1, "checkpoint_sha256": "a" * 64})
    lifecycle.append_event({"global_update": 2, "checkpoint_sha256": "b" * 64})
    assert [x["global_update"] for x in lifecycle.journal_events()] == [1, 2]


def test_journal_rejects_tampering_and_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "JOURNAL", tmp_path / "journal.jsonl")
    lifecycle.append_event({"global_update": 1})
    row = json.loads(lifecycle.JOURNAL.read_text())
    row["global_update"] = 2
    lifecycle.JOURNAL.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        lifecycle.journal_events()
    lifecycle.JOURNAL.write_bytes(b'{"global_update":')
    with pytest.raises(ValueError, match="truncated"):
        lifecycle.journal_events()


def test_recovery_promotes_journal_committed_pending_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "STAGING", tmp_path)
    monkeypatch.setattr(lifecycle, "JOURNAL", tmp_path / "journal.jsonl")
    state = {"global_update": 1, "schedule_cursor": 1, "optimizer": {"step": 1}, "scaler": {}, "rng": "complete"}
    pending = tmp_path / "pending.pt"
    lifecycle.save_checkpoint(pending, state)
    digest = lifecycle.file_sha(pending)
    lifecycle.append_event({"global_update": 1, "checkpoint_sha256": digest})
    recovered = lifecycle.recover_state(lifecycle.journal_events(), {"global_update": 0})
    assert recovered["optimizer"] == {"step": 1}
    assert lifecycle.file_sha(tmp_path / "latest.pt") == digest


def test_predeclared_boundaries_are_fixed():
    assert lifecycle.EXPOSURE_BOUNDARIES == (5, 10, 20, 35, 50)


def test_inconsistent_v1_contract_fails_closed():
    with pytest.raises(ValueError, match="sha256 pin mismatch for input_validation_sha256"):
        lifecycle.validate_contract()


@pytest.mark.parametrize(
    "label", ["input_validation", "corruption_cache_manifest", "preparation_manifest", "update_0_checkpoint"]
)
def test_changed_pinned_contract_bytes_are_rejected(tmp_path, label):
    path = tmp_path / f"{label}.bin"
    path.write_bytes(b"pinned original bytes")
    expected = lifecycle.file_sha(path)
    path.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match=f"sha256 pin mismatch for {label}"):
        lifecycle.verify_file_pin(path, expected, label)


def test_read_only_contract_checks_do_not_create_training_outputs():
    assert not lifecycle.STAGING.exists()
    assert not lifecycle.FINAL.exists()


def test_reconciled_v2_preparation_pins_remain_read_only_after_publication():
    # The original assertion described an interrupted 115-update staging run.
    # That run has since completed and published; verify immutable input pins,
    # rather than treating publication as a failure or authorizing a new resume.
    prep_path = lifecycle_v2.OUT / "preparation_manifest.json"
    before = lifecycle_v2.file_sha(prep_path)
    prep = lifecycle_v2.load_json(prep_path)
    for name, filename in (
        ("phase4a_plan_sha256", "phase4a_plan.json"),
        ("input_validation_sha256", "input_validation.json"),
        ("cache_manifest_sha256", "corruption_cache_manifest.json"),
        ("cache_validation_sha256", "cache_validation.json"),
        ("development_baseline_sha256", "development_corrupted_baseline.json"),
    ):
        lifecycle_v2.verify_file_pin(lifecycle_v2.OUT / filename, prep[name], name)
    lifecycle_v2.verify_file_pin(lifecycle_v2.CONFIG, prep["config_sha256"], "config")
    assert prep["training_identity_count"] == 2048
    assert prep["development_identity_count"] == 320
    assert lifecycle_v2.file_sha(prep_path) == before


def test_completed_final_monitor_does_not_require_staging_journal(tmp_path, monkeypatch):
    out = tmp_path / "run"
    final = out / "final"
    staging = out / "staging"
    final.mkdir(parents=True)
    (final / "training_metrics.json").write_text(
        json.dumps({"status": "completed", "authorizes_downstream": False, "training_updates": 1150})
    )
    monkeypatch.setattr(lifecycle_v2, "ROOT", tmp_path)
    monkeypatch.setattr(lifecycle_v2, "OUT", out)
    monkeypatch.setattr(lifecycle_v2, "FINAL", final)
    monkeypatch.setattr(lifecycle_v2, "STAGING", staging)
    monkeypatch.setattr(lifecycle_v2, "JOURNAL", staging / "journal.jsonl")
    monkeypatch.setattr(
        lifecycle_v2,
        "journal_events",
        lambda path=None: [{"checkpoint_sha256": "ok"}] * 1149 + [{"checkpoint_sha256": "ok"}],
    )
    monkeypatch.setattr(lifecycle_v2, "validate_boundary_files", lambda path: None)
    monkeypatch.setattr(lifecycle_v2, "file_sha", lambda path: "ok")
    monkeypatch.setattr(lifecycle_v2.torch, "load", lambda *a, **k: {"global_update": 1150, "schedule_cursor": 1150})
    result = lifecycle_v2.monitor()
    assert result["status"] == "completed"
    assert result["metrics"]["status"] == "completed"
    assert result["validated_journal_updates"] == 1150
    assert not staging.exists()


def test_final_without_scientific_review_is_monitorable_for_posthoc_recovery(tmp_path, monkeypatch):
    out = tmp_path / "run"
    final = out / "final"
    staging = out / "staging"
    final.mkdir(parents=True)
    (final / "training_metrics.json").write_text(
        json.dumps({"status": "completed", "phase4b_authorized": False, "training_updates": 1150})
    )
    monkeypatch.setattr(lifecycle_v2, "ROOT", tmp_path)
    monkeypatch.setattr(lifecycle_v2, "OUT", out)
    monkeypatch.setattr(lifecycle_v2, "FINAL", final)
    monkeypatch.setattr(lifecycle_v2, "STAGING", staging)
    monkeypatch.setattr(lifecycle_v2, "JOURNAL", staging / "journal.jsonl")
    monkeypatch.setattr(
        lifecycle_v2,
        "journal_events",
        lambda path=None: [{"checkpoint_sha256": "ok"}] * 1149 + [{"checkpoint_sha256": "ok"}],
    )
    monkeypatch.setattr(lifecycle_v2, "validate_boundary_files", lambda path: None)
    monkeypatch.setattr(lifecycle_v2, "file_sha", lambda path: "ok")
    monkeypatch.setattr(lifecycle_v2.torch, "load", lambda *a, **k: {"global_update": 1150, "schedule_cursor": 1150})
    result = lifecycle_v2.monitor()
    assert result["status"] == "completed"
    assert not (out / "phase4a_v2_scientific_review_v1").exists()
    assert not staging.exists()


def test_monitor_fails_closed_when_final_and_staging_both_exist(tmp_path, monkeypatch):
    out = tmp_path / "run"
    final = out / "final"
    staging = out / "staging"
    final.mkdir(parents=True)
    staging.mkdir()
    monkeypatch.setattr(lifecycle_v2, "ROOT", tmp_path)
    monkeypatch.setattr(lifecycle_v2, "OUT", out)
    monkeypatch.setattr(lifecycle_v2, "FINAL", final)
    monkeypatch.setattr(lifecycle_v2, "STAGING", staging)
    with pytest.raises(ValueError, match="both final and staging"):
        lifecycle_v2.monitor()


def test_plan_only_dispatch_does_not_enter_execution(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lifecycle_v2, "ROOT", tmp_path)
    staging, final = tmp_path / "staging", tmp_path / "final"
    monkeypatch.setattr(lifecycle_v2, "STAGING", staging)
    monkeypatch.setattr(lifecycle_v2, "FINAL", final)
    monkeypatch.setattr(lifecycle_v2, "validate_contract", lambda *args: {"status": "valid"})
    monkeypatch.setattr(lifecycle_v2, "run", lambda **kwargs: pytest.fail("plan-only entered training"))
    monkeypatch.setattr("sys.argv", ["run_e010_phase4a_training_v2.py", "--plan-only"])
    lifecycle_v2.main()
    assert '"status": "plan_only"' in capsys.readouterr().out
    assert not lifecycle_v2.STAGING.exists()
    assert not lifecycle_v2.FINAL.exists()


def _mock_full_state(update, exposures):
    return {
        "global_update": update,
        "schedule_cursor": update,
        "model": {"w": update},
        "optimizer": {"step": update},
        "scheduler": None,
        "scaler": {},
        "python_rng_state": ("mock",),
        "numpy_rng_state": ("mock",),
        "torch_cpu_rng_state": __import__("torch").tensor([update]),
        "torch_cuda_rng_state": [],
        "sampler_rng_state": ("mock",),
        "identity_exposures": dict(exposures),
        "training_schedule": ["a", "b", "c"],
    }


def _publish_mock_boundary(tmp_path, update=1):
    state = _mock_full_state(update, {"a": 1, "b": 1})
    latest = tmp_path / "latest.pt"
    lifecycle_v2.save_checkpoint(latest, state)
    record = {"path": latest, "sha256": lifecycle_v2.file_sha(latest)}
    rows = [
        {"sample_id": "dev1", "predicted_coordinates": [1, 2, 3]},
        {"sample_id": "dev2", "predicted_coordinates": [4, 5, 6]},
    ]
    result = lifecycle_v2.publish_boundary_record(
        1,
        record,
        state,
        lambda rows=rows: rows,
        lambda dev, base, cfg: {"complete": len(dev) == 2},
        [{}, {}],
        [],
        {},
        tmp_path,
    )
    return result, state, latest


def test_first_exposure_boundary_publishes_checkpoint_then_complete_evaluation(tmp_path):
    result, state, latest = _publish_mock_boundary(tmp_path)
    checkpoint = tmp_path / "checkpoint_at_exposure_01.pt"
    assert lifecycle_v2.file_sha(checkpoint) == result["checkpoint_sha256"]
    assert lifecycle_v2.load_json(tmp_path / "development_exposure_01.json") == result
    assert result["global_update"] == state["global_update"] == 1
    assert latest.exists()


@pytest.mark.parametrize("interrupt_after_publish", [False, True])
def test_boundary_interruptions_resume_without_replaying_updates(tmp_path, interrupt_after_publish):
    import torch

    exposures = {"a": 1, "b": 1}
    state = _mock_full_state(1, exposures)
    latest = tmp_path / "latest.pt"
    lifecycle_v2.save_checkpoint(latest, state)
    record = {"path": latest, "sha256": lifecycle_v2.file_sha(latest)}
    if interrupt_after_publish:
        _publish_mock_boundary(tmp_path)
    # A missing boundary is an uncommitted evaluation; resume publishes from update 1.
    path = tmp_path / "development_exposure_01.json"
    if not path.exists():
        rows = [{"sample_id": "d1"}, {"sample_id": "d2"}]
        lifecycle_v2.publish_boundary_record(
            1, record, state, lambda rows=rows: rows, lambda d, b, c: {"complete": True}, [{}, {}], [], {}, tmp_path
        )
    recovered = torch.load(latest, map_location="cpu", weights_only=False)
    assert recovered["global_update"] == recovered["schedule_cursor"] == 1
    assert recovered["identity_exposures"] == exposures
    assert path.exists()


def test_boundary_rejects_checkpoint_corruption_and_partial_boundary(tmp_path):
    result, state, latest = _publish_mock_boundary(tmp_path)
    checkpoint = tmp_path / "checkpoint_at_exposure_01.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="corrupt development boundary"):
        lifecycle_v2.validate_boundary_files(tmp_path)
    checkpoint.write_bytes(latest.read_bytes())
    boundary_path = tmp_path / "development_exposure_01.json"
    boundary_path.write_text('{"exposure":1,"global_update":1,"checkpoint_sha256":"x"}')
    with pytest.raises(ValueError, match="corrupt development boundary"):
        lifecycle_v2.validate_boundary_files(tmp_path)


def test_corrupt_checkpoint_and_journal_are_rejected_and_mock_lifecycle_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle_v2, "STAGING", tmp_path)
    monkeypatch.setattr(lifecycle_v2, "JOURNAL", tmp_path / "journal.jsonl")
    base = _mock_full_state(0, {"a": 0, "b": 0})
    state = _mock_full_state(1, {"a": 1, "b": 1})
    pending = tmp_path / "pending.pt"
    lifecycle_v2.save_checkpoint(pending, state)
    digest = lifecycle_v2.file_sha(pending)
    lifecycle_v2.append_event({"global_update": 1, "checkpoint_sha256": digest})
    recovered = lifecycle_v2.recover_state(lifecycle_v2.journal_events(), base)
    assert recovered["global_update"] == 1
    lifecycle_v2.validate_boundary_files(tmp_path)
    rows = [{"sample_id": "d1"}, {"sample_id": "d2"}]
    lifecycle_v2.publish_boundary_record(
        1,
        {"path": tmp_path / "latest.pt", "sha256": digest},
        recovered,
        lambda rows=rows: rows,
        lambda d, b, c: {"gate_adjudication": "mock_only"},
        [{}, {}],
        [],
        {},
        tmp_path,
    )
    assert (
        lifecycle_v2.load_json(tmp_path / "development_exposure_01.json")["metrics"]["gate_adjudication"] == "mock_only"
    )
    # A changed checkpoint is never accepted as the journal's recovery point.
    (tmp_path / "latest.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="no matching committed or pending checkpoint"):
        lifecycle_v2.recover_state(lifecycle_v2.journal_events(), base)
    lifecycle_v2.JOURNAL.write_bytes(lifecycle_v2.JOURNAL.read_bytes() + b"{partial")
    with pytest.raises(ValueError, match="truncated"):
        lifecycle_v2.journal_events()


def test_small_mocked_lifecycle_reaches_first_boundary_and_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle_v2, "STAGING", tmp_path)
    monkeypatch.setattr(lifecycle_v2, "JOURNAL", tmp_path / "journal.jsonl")
    exposures = {"train_a": 0, "train_b": 0}
    base = _mock_full_state(0, exposures)
    # A two update mock schedule reaches exposure one from a fresh update-zero state.
    for update, sid in ((1, "train_a"), (2, "train_b")):
        exposures[sid] += 1
        state = _mock_full_state(update, exposures)
        pending = tmp_path / "pending.pt"
        lifecycle_v2.save_checkpoint(pending, state)
        digest = lifecycle_v2.file_sha(pending)
        lifecycle_v2.append_event({"global_update": update, "checkpoint_sha256": digest})
        __import__("os").replace(pending, tmp_path / "latest.pt")
        if update == 1:
            rows = [{"sample_id": "dev_a"}, {"sample_id": "dev_b"}]
            lifecycle_v2.publish_boundary_record(
                1,
                {"path": tmp_path / "latest.pt", "sha256": digest},
                state,
                lambda rows=rows: rows,
                lambda d, b, c: {"mock_gate": "complete"},
                [{}, {}],
                [],
                {},
                tmp_path,
            )
    events = lifecycle_v2.journal_events()
    recovered = lifecycle_v2.recover_state(events, base)
    lifecycle_v2.validate_boundary_files(tmp_path)
    assert recovered["global_update"] == 2 and events[-1]["global_update"] == 2
    assert recovered["identity_exposures"] == {"train_a": 1, "train_b": 1}
    assert lifecycle_v2.load_json(tmp_path / "development_exposure_01.json")["evaluation_complete"] is True
    with pytest.raises(ValueError, match="sha256 pin mismatch for input_validation_sha256"):
        lifecycle.validate_contract()
    assert not lifecycle.STAGING.exists()
    assert not lifecycle.FINAL.exists()
