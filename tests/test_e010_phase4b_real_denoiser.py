from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/prepare_cache.py"
SPEC = importlib.util.spec_from_file_location("phase4b_prepare_cache", MODULE)
pc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pc)
RUNNER_PATH = MODULE.with_name("runner.py")
RUNNER_SPEC = importlib.util.spec_from_file_location("phase4b_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def test_v_prediction_reconstructs_clean_coordinates_and_zeros_padding():
    rng = np.random.default_rng(19)
    clean = rng.normal(size=(9, 3)).astype(np.float32)
    clean -= clean.mean(axis=0, keepdims=True)
    noise = rng.normal(size=(9, 3)).astype(np.float32)
    noise -= noise.mean(axis=0, keepdims=True)
    alpha_bar = 0.37
    alpha, sigma = np.sqrt(alpha_bar), np.sqrt(1 - alpha_bar)
    xt = alpha * clean + sigma * noise
    v = alpha * noise - sigma * clean
    got = pc.reconstruct_x0(xt, v, alpha_bar, np.ones(9, dtype=bool))
    np.testing.assert_allclose(got, clean, atol=2e-7)
    mask = np.array([True] * 7 + [False] * 2)
    padded = np.vstack([xt[:7], np.zeros((2, 3), dtype=np.float32)])
    vp = np.vstack([v[:7], np.zeros((2, 3), dtype=np.float32)])
    out = pc.reconstruct_x0(padded, vp, alpha_bar, mask)
    assert np.count_nonzero(out[~mask]) == 0


def test_forward_noising_is_deterministic_and_centered():
    rng = np.random.default_rng(23)
    clean = rng.normal(size=(12, 3)).astype(np.float32)
    clean -= clean.mean(0, keepdims=True)
    noise = rng.normal(size=clean.shape).astype(np.float32)
    noise -= noise.mean(0, keepdims=True)
    mask = np.ones(12, dtype=bool)
    first = pc.forward_noise(clean, noise, 0.61, mask)
    second = pc.forward_noise(clean, noise.copy(), 0.61, mask)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(0), 0, atol=1e-7)


def test_actual_e007_v_target_roundtrip_at_pinned_timesteps_with_padding():
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates

    diffusion = CoordinateVPDiffusion(500)
    clean = torch.tensor([
        [[1.0, 2.0, -1.0], [2.5, -1.0, 0.5], [-0.5, 3.0, 1.0], [0, 0, 0], [0, 0, 0]],
        [[-1.0, 0.5, 2.0], [1.0, -0.5, -2.0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    clean = center_coordinates(clean, mask)
    epsilon = torch.tensor([
        [[0.5, -0.2, 0.7], [-0.1, 0.8, -0.4], [-0.4, -0.6, -0.3], [0, 0, 0], [0, 0, 0]],
        [[0.3, 0.7, -0.2], [-0.3, -0.7, 0.2], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=torch.float32)
    epsilon = center_coordinates(epsilon, mask)
    for timestep in (50, 250, 450):
        times = torch.full((2,), timestep, dtype=torch.long)
        alpha, sigma = diffusion.alpha_sigma(times, clean)
        xt = center_coordinates(alpha * clean + sigma * epsilon, mask)
        # This is the exact centered coordinate v target in E007's
        # CoordinateVPDiffusion.make_training_batch implementation.
        target_v = diffusion.training_target(clean, epsilon, times, mask)
        recovered = diffusion.reconstruct_x0(xt, times, target_v, mask)
        torch.testing.assert_close(recovered[mask], clean[mask], rtol=0, atol=2e-6)
        assert torch.count_nonzero(recovered[~mask]).item() == 0


def test_frozen_imperfect_prediction_is_accepted_and_reports_rmse():
    import torch

    mask = torch.tensor([[True, True, False]])
    x0 = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0, 0, 0]]])
    x0_hat = torch.tensor([[[0.75, 0.0, 0.0], [-0.75, 0.0, 0.0], [0, 0, 0]]])
    class DeterministicFrozenMock(torch.nn.Module):
        def forward(self, noisy, _times, _lengths, _mask, _continuity):
            return {"v_prediction": torch.zeros_like(noisy)}

    model = DeterministicFrozenMock().eval()
    noisy = torch.ones_like(x0)
    with torch.inference_mode():
        first = model(noisy, None, None, mask, None)["v_prediction"]
        second = model(noisy, None, None, mask, None)["v_prediction"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    rmse = pc.validate_frozen_prediction(first, x0_hat, x0, mask)
    assert rmse == pytest.approx((0.0625 / 3) ** 0.5)
    assert rmse > 0


def test_e007_parameterization_fails_closed_on_checkpoint_or_config_contradiction():
    assert pc.validate_diffusion_parameterization(
        {"prediction_parameterization": "centered_coordinate_v"}, {"arm": "v_only"}
    ) == "centered_coordinate_v"
    with pytest.raises(ValueError, match="parameterization mismatch"):
        pc.validate_diffusion_parameterization({"prediction_parameterization": "epsilon"}, {"arm": "v_only"})
    with pytest.raises(ValueError, match="arm is not v_only"):
        pc.validate_diffusion_parameterization({"prediction_parameterization": "centered_coordinate_v"}, {"arm": "epsilon_only"})


def test_real_schedule_reuses_exact_identities_and_is_timestep_balanced():
    train, dev = pc.build_records()
    assert len(train) == 98280
    assert len({row["sample_id"] for row in train}) == 16384
    assert len(dev) == 960
    assert len({row["sample_id"] for row in dev}) == 320
    assert not ({row["sample_id"] for row in train} & {row["sample_id"] for row in dev})
    config = pc.config()
    for stratum in pc.STRATA:
        counts = [sum(row["timestep"] == t and row["stratum"] == stratum for row in train)
                  for t in config["diffusion"]["timesteps"]]
        assert counts == [6552, 6552, 6552]
    train2, dev2 = pc.build_records()
    assert [(r["sample_id"], r["seed"], r["timestep"]) for r in train] == [(r["sample_id"], r["seed"], r["timestep"]) for r in train2]
    assert [(r["sample_id"], r["seed"], r["timestep"]) for r in dev] == [(r["sample_id"], r["seed"], r["timestep"]) for r in dev2]


def test_cache_writer_schema_reader_monitor_and_fail_closed(tmp_path: Path, monkeypatch):
    """Use the exact manifest, sidecar, and NPZ layout written by build_cache."""
    cache = tmp_path / "cache.final"
    cache.mkdir()
    contract = {"training_schedule_sha256": "schedule"}
    cfg = {"cache": {"final_dir": cache.name},
           "diffusion": {"checkpoint_sha256": "checkpoint", "timesteps": [50, 250, 450]},
           "data": {"train_example_count": 2, "development_example_count": 1}}
    monkeypatch.setattr(pc, "config", lambda: cfg)
    monkeypatch.setattr(pc, "validate_contract", lambda: contract)
    monkeypatch.setattr(runner.pc, "config", lambda: cfg)
    monkeypatch.setattr(runner.pc, "validate_contract", lambda: contract)
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "STAGE", tmp_path / "training.staging")
    monkeypatch.setattr(runner, "FINAL", tmp_path / "training.final")

    rows, targets, predictions = [], [], []
    for i, (split, timestep) in enumerate((("train", 50), ("train", 250), ("development", 450))):
        target = (np.arange(60, dtype=np.float32).reshape(20, 3) + i).copy()
        prediction = target + np.float32(0.25)
        row = {"split": split, "sample_id": f"identity_{i}", "length": 20,
               "stratum": "20-64", "source_path": f"source_{i}.npz",
               "source_sha256": "a" * 64, "seed": i + 100,
               "condition_index": i, "timestep": timestep,
               "target_sha256": pc.digest(np.ascontiguousarray(target, dtype="<f4").tobytes()),
               "prediction_sha256": pc.digest(np.ascontiguousarray(prediction, dtype="<f4").tobytes()),
               "mask_all_valid": True, "mask_sha256": pc.digest(np.ones(20, dtype=np.uint8).tobytes()),
               "denoiser_error_rmse": 0.25, "normalized_x0_hat_rmse": 0.25}
        if split == "train":
            row.update(schedule_index=i, schedule_update=1, microbatch_position=i)
        rows.append(row)
        targets.append(target)
        predictions.append(prediction)
    archive = cache / "shard_00000.npz"
    with archive.open("wb") as handle:
        np.savez_compressed(handle, target=np.concatenate(targets), prediction=np.concatenate(predictions),
            offsets=np.array([0, 20, 40, 60], dtype=np.int64),
            records_json=np.frombuffer(json.dumps(rows, sort_keys=True).encode(), dtype=np.uint8))
    entry = {"shard": archive.name, "archive_sha256": pc.sha256(archive),
             "record_count": 3, "first_record": 0, "records": rows}
    pc.atomic_json(archive.with_suffix(".json"), entry)
    summary = {stratum: {str(t): {"count": int(stratum == "20-64"),
        "mean_rmse": 0.25 if stratum == "20-64" else None,
        "median_rmse": 0.25 if stratum == "20-64" else None,
        "max_rmse": 0.25 if stratum == "20-64" else None}
        for t in cfg["diffusion"]["timesteps"]} for stratum in pc.STRATA}
    manifest = {"schema": "e010_phase4b_real_denoiser_cache_v1",
                "contract_sha256": pc.digest(pc.canonical(contract)),
                "diffusion_checkpoint_sha256": "checkpoint", "record_count": 3,
                "training_schedule_sha256": "schedule", "training_record_count": 2,
                "development_record_count": 1, "shards": [entry],
                "error_summary_by_stratum_timestep": summary, "authorization": pc.AUTH,
                "prospective_accessed": False}
    pc.atomic_json(cache / "manifest.json", manifest)

    loaded, actual = runner._load_cache()
    assert actual == manifest
    assert [row["split"] for row in loaded] == ["train", "train", "development"]
    assert len(loaded) == 3
    for i, row in enumerate(loaded):
        np.testing.assert_array_equal(row["target"], targets[i])
        np.testing.assert_array_equal(row["prediction"], predictions[i])
    assert runner.monitor()["status"] == "cache_complete_awaiting_zero_shot"
    runner.STAGE.mkdir()
    zero_archive = runner.STAGE / "zero_shot_predictions.npz"
    with zero_archive.open("wb") as handle:
        np.savez_compressed(handle, prediction=predictions[-1],
            offsets=np.array([0, 20], dtype=np.int64),
            records_json=np.frombuffer(json.dumps([{"sample_id": "identity_2"}]).encode(), dtype=np.uint8))
    cache_sha256 = pc.digest(pc.canonical(manifest))
    zero_report = {"schema": "e010_phase4b_zero_shot_v1", "cache_sha256": cache_sha256,
                   "prediction_archive_sha256": pc.sha256(zero_archive),
                   "training_started": False, "authorization": pc.AUTH}
    pc.atomic_json(runner.STAGE / "zero_shot.json", zero_report)
    pc.atomic_json(runner.STAGE / "zero_shot_complete.json", {
        "schema": "e010_phase4b_zero_shot_commit_v1",
        "report_sha256": pc.sha256(runner.STAGE / "zero_shot.json"),
        "prediction_archive_sha256": pc.sha256(zero_archive),
        "cache_sha256": cache_sha256, "development_condition_count": 1,
        "training_started": False, "authorization": pc.AUTH})
    assert runner.monitor()["status"] == "zero_shot_complete_awaiting_execution"
    runner.FINAL.mkdir()
    with pytest.raises(ValueError, match="lacks results.json"):
        runner.monitor()
    runner.FINAL.rmdir()

    bad = json.loads(json.dumps(manifest))
    bad["shards"][0]["shard"] = "missing.npz"
    with pytest.raises(ValueError, match="name/order mismatch"):
        pc.verify_cache(cache, bad)
    archive.rename(cache / "shard_00000.missing")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        runner._load_cache()


def test_artifact_pin_rejects_wrong_expected_checkpoint(tmp_path: Path):
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"authentic test bytes")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pc.assert_sha256(artifact, "0" * 64, "test checkpoint")


def test_zero_shot_is_immutable_and_json_scalars_are_native():
    import torch

    class MockRefiner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, coords, mask):
            return {"prediction": coords * self.weight}

    model = MockRefiner()
    before = model.weight.detach().clone()
    out = runner.zero_shot_prediction(model, torch.ones((1, 4, 3)), torch.ones((1, 4), dtype=torch.bool))
    torch.testing.assert_close(out, torch.full((1, 4, 3), 2.0))
    torch.testing.assert_close(model.weight, before)
    assert model.weight.grad is None
    native = runner._native({"count": np.int64(2), "rate": np.float32(.25), "ok": np.bool_(True)})
    assert native == {"count": 2, "rate": .25, "ok": True}


def test_resume_requires_full_journal_matched_state():
    journal = b'{"global_update":1}\n'
    state = {"global_update": 1, "journal_prefix_sha256": __import__("hashlib").sha256(journal).hexdigest(),
        "cache_manifest_sha256": "cache", "model": {}, "optimizer": {}, "scheduler": None,
        "scaler": {}, "python_rng": 1, "numpy_rng": 2, "torch_rng": 3, "cuda_rng": []}
    assert runner.resume_identity_valid(state, journal, "cache")
    assert not runner.resume_identity_valid(state, journal, "wrong-cache")
    assert not runner.resume_identity_valid(state, journal + b"{}\n", "cache")


def test_publication_refuses_existing_final_directory(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    marker = final / "keep.txt"
    marker.write_text("unchanged")
    source = tmp_path / "staging"
    source.mkdir()
    (source / "new.txt").write_text("new")
    with pytest.raises(FileExistsError):
        pc.publish_directory_once(source, final)
    assert marker.read_text() == "unchanged"


def _mock_state(update, cache="cache"):
    import random

    import torch
    return {"schema": "e010_phase4b_exact_state_v1", "global_update": update,
            "cache_manifest_sha256": cache, "model": {"weight": torch.tensor([float(update)])},
            "optimizer": {"step": update}, "scheduler": None, "scaler": {},
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": [], "boundaries": {},
            "authorization": pc.AUTH}


def _mock_event(update):
    return {"global_update": update, "mean_loss": 0.25, "cache_manifest_sha256": "cache"}


@pytest.mark.parametrize("empty_journal", [False, True])
def test_fresh_and_failed_first_attempt_restart_from_zero(tmp_path, empty_journal):
    if empty_journal:
        (tmp_path / "journal.jsonl").write_bytes(b"")
    (tmp_path / "pending.pt.tmp").write_bytes(b"interrupted serialization")
    (tmp_path / "journal.jsonl.tmp").write_bytes(b"unpublished row")
    state, raw, status = runner._prepare_execution(tmp_path, "cache", resume=False)
    assert state is None and raw == b"" and status == "zero_committed_updates"
    with pytest.raises(ValueError, match="zero committed updates require --execute"):
        runner._prepare_execution(tmp_path, "cache", resume=True)
    runner._commit_update(tmp_path, _mock_state(1), _mock_event(1), "cache")
    assert runner._validated_journal(tmp_path, "cache")[0] == 1


def test_first_commit_and_exact_second_update_rng_continuation(tmp_path):
    import random

    import torch
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    event1 = _mock_event(1)
    runner._commit_update(tmp_path, _mock_state(1), event1, "cache")
    one_row = pc.canonical(event1) + b"\n"
    assert (tmp_path / "journal.jsonl").read_bytes() == one_row
    state, raw, status = runner._prepare_execution(tmp_path, "cache", resume=True, recover=True)
    assert raw == one_row and status == "consistent"
    assert state["journal_prefix_sha256"] == pc.digest(one_row)
    expected = (random.random(), np.random.random(), torch.rand(3))
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); torch.set_rng_state(state["torch_rng"])
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    runner._commit_update(tmp_path, _mock_state(2), _mock_event(2), "cache")
    state2, raw2, _, status2 = runner._inspect_transaction(tmp_path, "cache")
    assert raw2 == one_row + pc.canonical(_mock_event(2)) + b"\n"
    assert state2["global_update"] == state2["optimizer"]["step"] == 2
    assert state2["journal_prefix_sha256"] == pc.digest(raw2) and status2 == "consistent"
    torch.testing.assert_close(state2["torch_rng"], torch.get_rng_state())
    with pytest.raises(ValueError, match="use --resume"):
        runner._prepare_execution(tmp_path, "cache", resume=False)


@pytest.mark.parametrize("update", [1, 2])
@pytest.mark.parametrize("crash_at", ["checkpoint", "journal", "latest"])
def test_transaction_crashes_are_detected_and_recovered(tmp_path, monkeypatch, update, crash_at):
    if update == 2:
        runner._commit_update(tmp_path, _mock_state(1), _mock_event(1), "cache")
    replace = runner.os.replace
    def crash(source, target):
        names = {"checkpoint": "pending.pt", "journal": "journal.jsonl", "latest": "latest.pt"}
        if Path(target).name == names[crash_at]:
            raise OSError("simulated crash")
        replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(runner.os, "replace", crash)
        with pytest.raises(OSError, match="simulated crash"):
            runner._commit_update(tmp_path, _mock_state(update), _mock_event(update), "cache")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    state, raw, _, status = runner._inspect_transaction(tmp_path, "cache")
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}  # monitor inspection is read-only
    committed = update - 1 if crash_at == "checkpoint" else update
    assert (state["global_update"] if state else 0) == committed
    if committed == 0:
        assert status == "zero_committed_updates"
        runner._prepare_execution(tmp_path, "cache", resume=False)
    else:
        if crash_at != "checkpoint":
            assert status == ("checkpoint_committed_journal_pending" if crash_at == "journal" else "journal_committed_latest_pending")
        restored, published, _ = runner._prepare_execution(tmp_path, "cache", resume=True, recover=True)
        assert restored["global_update"] == committed
        assert (tmp_path / "journal.jsonl").read_bytes() == published == raw
        assert runner._validated_journal(tmp_path, "cache")[0] == committed
        assert not (tmp_path / "pending.pt").exists()
        # Recovery is idempotent and retains the exact model/optimizer/RNG payload.
        again, _, _ = runner._prepare_execution(tmp_path, "cache", resume=True, recover=True)
        assert again["optimizer"] == restored["optimizer"]
        import torch
        torch.testing.assert_close(again["model"]["weight"], restored["model"]["weight"])
        torch.testing.assert_close(again["torch_rng"], restored["torch_rng"])


def test_recovery_fails_closed_on_disagreement(tmp_path):
    runner._commit_update(tmp_path, _mock_state(1), _mock_event(1), "cache")
    (tmp_path / "journal.jsonl").write_bytes(pc.canonical(_mock_event(2)) + b"\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner._prepare_execution(tmp_path, "cache", resume=False)


def test_monitor_zero_and_pending_transactions(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir(); (cache / "manifest.json").write_text('{}\n')
    manifest = {"record_count": 3, "development_record_count": 1}
    cache_sha = pc.digest(pc.canonical(manifest))
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "STAGE", tmp_path / "stage")
    monkeypatch.setattr(runner, "FINAL", tmp_path / "final")
    monkeypatch.setattr(runner.pc, "config", lambda: {"cache": {"final_dir": "cache"}})
    monkeypatch.setattr(runner.pc, "load_cache_manifest", lambda _p: manifest)
    monkeypatch.setattr(runner.pc, "verify_cache", lambda *_a: None)
    monkeypatch.setattr(runner, "_validated_zero_shot", lambda *_a: True)
    runner.STAGE.mkdir()
    result = runner.monitor()
    assert result["committed_updates"] == result["journal_updates"] == 0
    assert result["next_action"] == "--execute"
    assert result["cache_manifest_sha256"] == pc.sha256(cache / "manifest.json")
    assert result["cache_manifest_canonical_sha256"] == cache_sha
    (runner.STAGE / "journal.jsonl").write_bytes(b"")
    assert runner.monitor()["committed_updates"] == 0
    state = _mock_state(1, cache_sha)
    event = {**_mock_event(1), "cache_manifest_sha256": cache_sha}
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_publish_journal", lambda *_a: (_ for _ in ()).throw(OSError("crash")))
        with pytest.raises(OSError):
            runner._commit_update(runner.STAGE, state, event, cache_sha)
    result = runner.monitor()
    assert result["committed_updates"] == 1 and result["journal_updates"] == 0
    assert result["next_action"] == "--resume"
    assert not (runner.STAGE / "latest.pt").exists()


def test_zero_shot_byte_pin_is_explicit_and_fail_closed():
    runner._validate_manifest_byte_pin({"cache_sha256": "canonical"}, {}, "raw")
    runner._validate_manifest_byte_pin({"cache_manifest_file_sha256": "raw"},
                                      {"cache_manifest_file_sha256": "raw"}, "raw")
    with pytest.raises(ValueError, match="byte pin mismatch"):
        runner._validate_manifest_byte_pin({"cache_manifest_file_sha256": "other"},
                                          {"cache_manifest_file_sha256": "other"}, "raw")
    with pytest.raises(ValueError, match="byte pin mismatch"):
        runner._validate_manifest_byte_pin({"cache_manifest_file_sha256": "raw"}, {}, "raw")


@pytest.mark.parametrize("committed", [False, True])
def test_execute_restart_guard_runs_before_any_cuda_or_model_work(tmp_path, monkeypatch, committed):
    import torch

    stage = tmp_path / "stage"; stage.mkdir()
    cache = tmp_path / "cache"; cache.mkdir()
    manifest = {"test": "manifest"}
    (cache / "manifest.json").write_bytes(pc.canonical(manifest))
    cache_sha = pc.digest(pc.canonical(manifest))
    zero = {"cache_sha256": cache_sha, "prediction_archive_sha256": "prediction"}
    pc.atomic_json(stage / "zero_shot.json", zero)
    pc.atomic_json(stage / "zero_shot_complete.json", {
        "schema": "e010_phase4b_zero_shot_commit_v1", "report_sha256": pc.sha256(stage / "zero_shot.json"),
        "prediction_archive_sha256": "prediction", "development_condition_count": 960,
        "training_started": False, "authorization": pc.AUTH, "cache_sha256": cache_sha})
    monkeypatch.setattr(runner, "STAGE", stage)
    monkeypatch.setattr(runner, "HERE", tmp_path)
    monkeypatch.setattr(runner, "FINAL", tmp_path / "final")
    monkeypatch.setattr(runner, "REVIEW", tmp_path / "review")
    monkeypatch.setattr(runner.pc, "config", lambda: {"cache": {"final_dir": "cache"}})
    monkeypatch.setattr(runner.pc, "load_cache_manifest", lambda *_a: manifest)
    monkeypatch.setattr(runner.pc, "verify_cache", lambda *_a: None)
    monkeypatch.setattr(runner, "_model", lambda *_a, **_kw: pytest.fail("must not construct a model"))
    if committed:
        runner._commit_update(stage, _mock_state(1, cache_sha),
                              {**_mock_event(1), "cache_manifest_sha256": cache_sha}, cache_sha)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("must reject execute before CUDA"))
        with pytest.raises(ValueError, match="use --resume"):
            runner.execute(resume=False)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            runner.execute(resume=False)
        with pytest.raises(ValueError, match="zero committed updates require --execute"):
            runner.execute(resume=True)
        assert not (stage / "journal.jsonl").exists()
        assert not torch.cuda.is_initialized()
