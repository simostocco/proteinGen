from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import subprocess
import sys

import pytest
import torch
import yaml

from protein_distance_diffusion.training import e007_phase3i5_sampler_unroll as phase3i5
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e007_phase3i5_contract import plan, validate_contract


class LinearDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.15))

    def forward(self, x, timestep, lengths, mask, continuity):
        return {"v_prediction": self.scale * x}


def test_transition_unroll_is_differentiable_and_masked():
    torch.manual_seed(7)
    model = LinearDenoiser()
    clean = torch.randn(1, 6, 3)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    continuity = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    clean = clean * mask[..., None]
    noisy = clean + torch.randn_like(clean) * 0.3
    terms = phase3i5.sampler_unroll_objective(
        model,
        CoordinateVPDiffusion(12),
        noisy,
        torch.tensor([8]),
        torch.tensor([4]),
        mask,
        continuity,
        clean,
        12.22820347644835,
    )
    assert all(torch.isfinite(value) for value in terms.values())
    grad = torch.autograd.grad(terms["post_transition_x0_geometry"], model.scale)[0]
    assert torch.isfinite(grad) and grad.abs() > 0
    assert terms["immediate_x0_geometry"].ndim == 0


def test_rss_policy_uses_current_usage_and_keeps_peak_distinct():
    snapshot = {
        "rss_current_mib": 6000.0,
        "rss_peak_mib": 7000.0,
        "current_allocated_mib": 100.0,
        "current_reserved_mib": 200.0,
        "peak_allocated_mib": 300.0,
        "peak_reserved_mib": 400.0,
        "device_capacity_mib": 8000.0,
    }
    phase3i5.validate_memory_telemetry(snapshot, {"rss": 6144, "allocated": 6144, "reserved": 7680})
    with pytest.raises(MemoryError, match="rss_current_mib"):
        phase3i5.validate_memory_telemetry(
            {**snapshot, "rss_current_mib": 6145.0}, {"rss": 6144, "allocated": 6144, "reserved": 7680}
        )
    # Historical peak may exceed the cap; the current resident set is the cap input.
    phase3i5.validate_memory_telemetry(
        {**snapshot, "rss_current_mib": 1000.0, "rss_peak_mib": 9000.0},
        {"rss": 6144, "allocated": 6144, "reserved": 7680},
    )


def _rss_boundary(boundary, current, *, tensors=100, trajectories=20, handles=12, peak=None):
    return {
        "boundary": boundary,
        "rss_current_mib": current,
        "rss_peak_mib": peak if peak is not None else current,
        "live_tensor_objects": tensors,
        "trajectory_record_count": trajectories,
        "expected_trajectory_record_count": trajectories,
        "open_file_handles": handles,
    }


def test_rss_trend_passes_observed_warmup_then_steady_state_values():
    phase3i5.validate_boundary_rss_trend(
        [_rss_boundary(0, 2986.672), _rss_boundary(10, 3969.902), _rss_boundary(25, 4087.281)]
    )


def test_large_warmup_increase_alone_does_not_fail():
    phase3i5.validate_boundary_rss_trend([_rss_boundary(0, 1000), _rss_boundary(10, 4000)])


def test_post_warmup_rss_growth_over_tolerance_fails():
    with pytest.raises(MemoryError, match="delta=256.001"):
        phase3i5.validate_boundary_rss_trend([_rss_boundary(10, 4000), _rss_boundary(25, 4256.001)])


def test_rss_trend_rejects_absolute_current_rss_cap_excess():
    snapshot = {
        "rss_current_mib": 6144.001,
        "rss_peak_mib": 6144.001,
        "current_allocated_mib": 1.0,
        "current_reserved_mib": 1.0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
        "device_capacity_mib": 8192.0,
    }
    with pytest.raises(MemoryError, match="rss_current_mib"):
        phase3i5.validate_memory_telemetry(snapshot, {"rss": 6144, "allocated": 6144, "reserved": 7680})


@pytest.mark.parametrize("field", ["live_tensor_objects", "trajectory_record_count", "open_file_handles"])
def test_tracked_resource_accumulation_fails(field):
    reference = _rss_boundary(10, 4000)
    final = _rss_boundary(25, 4010)
    final[field] += 1
    with pytest.raises(MemoryError, match=f"tracked resource accumulation: {field}"):
        phase3i5.validate_boundary_rss_trend([reference, final])


def test_checkpointed_unroll_preserves_rng_and_gradient_equivalence():
    class RandomDenoiser(LinearDenoiser):
        def forward(self, x, timestep, lengths, mask, continuity):
            return {"v_prediction": self.scale * x + torch.rand_like(x) * 0.01}

    torch.manual_seed(82)
    clean = torch.randn(1, 6, 3)
    noisy = clean + torch.randn_like(clean) * 0.2
    args = (
        CoordinateVPDiffusion(12),
        noisy,
        torch.tensor([8]),
        torch.tensor([6]),
        torch.ones(1, 6, dtype=torch.bool),
        torch.ones(1, 5, dtype=torch.bool),
        clean,
        12.22820347644835,
    )
    model_a, model_b = RandomDenoiser(), RandomDenoiser()
    model_b.load_state_dict(model_a.state_dict())
    torch.manual_seed(702)
    loss_a = phase3i5.sampler_unroll_objective(model_a, *args)["total"]
    grad_a = torch.autograd.grad(loss_a, model_a.scale)[0]
    torch.manual_seed(702)
    loss_b = phase3i5.sampler_unroll_objective(model_b, *args, activation_checkpointing=True)["total"]
    grad_b = torch.autograd.grad(loss_b, model_b.scale)[0]
    assert torch.allclose(loss_a, loss_b, atol=1e-7, rtol=1e-6)
    assert torch.allclose(grad_a, grad_b, atol=1e-6, rtol=1e-5)


def test_scaler_state_is_included_in_resume_round_trip():
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model = LinearDenoiser()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    payload = phase3i5.make_resume_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        update=0,
        data_cursor=0,
        update_identity={"update": 0},
        protected_hashes={"checkpoint": "pinned"},
    )
    payload["state_hash"] = phase3i5.state_hash({k: v for k, v in payload.items() if k != "state_hash"})
    phase3i5.verify_resume_state(payload, {"checkpoint": "pinned"})
    resumed = torch.amp.GradScaler("cuda", enabled=False)
    resumed.load_state_dict(payload["scaler"])
    assert resumed.state_dict() == scaler.state_dict()


def test_panel_contract_proves_validation_calibration_and_holdout_separation():
    result = validate_contract("configs/e007_phase3i5_sampler_unroll_v1.yaml")
    panel = json.loads(open("configs/e007_phase3i5_development_panel_v1.json").read())
    ids = {row["sample_id"] for row in panel["records"]}
    assert result["development_panel_size"] == 10
    assert {row["target_length"] for row in panel["records"]} == {64, 128, 256, 384, 500}
    assert not ids.intersection(panel["calibration_exclusion"]["identities"])
    assert panel["training_exclusion"]["disjoint_by_split"] is True
    assert panel["holdout_exclusion"]["excluded_namespace"] == "holdout-n"


def test_plan_has_complete_workload_and_no_extension_or_side_effects():
    report = plan("configs/e007_phase3i5_sampler_unroll_v1.yaml")
    assert report["workload_counts"]["training_forwards_by_arm"] == {
        "continued_v_only": 25,
        "v_plus_sampler_unroll": 50,
    }
    assert report["workload_counts"]["one_step_evaluation_forwards"] == 60
    assert report["workload_counts"]["production_sampling_forwards"] == 20_000
    assert report["workload_counts"]["total_forwards"] == 20_210
    assert report["workload_counts"]["total_backward_calls"] == 200
    assert report["post_smoke_empirical_runtime_projection_seconds"] is None
    assert report["output_created"] is False and report["cuda_initialized"] is False
    assert "maximum_updates" not in yaml.safe_load(open("configs/e007_phase3i5_sampler_unroll_v1.yaml"))
    assert list(phase3i5.updates_to_boundary(0, 10)) == list(range(1, 11))
    assert list(phase3i5.updates_to_boundary(10, 25)) == list(range(11, 26))
    assert list(phase3i5.updates_to_boundary(25, 25)) == []
    with pytest.raises(ValueError):
        phase3i5.updates_to_boundary(0, 50)


def test_reviewed_v5_contract_and_plan_are_non_authorizing():
    contract = validate_contract("configs/e007_phase3i5_sampler_unroll_v5_reviewed.yaml")
    report = plan("configs/e007_phase3i5_sampler_unroll_v5_reviewed.yaml")
    assert contract["status"] == "read_only_contract_validated"
    assert report["status"] == "planned_non_authorizing"
    assert report["cuda_initialized"] is False
    assert report["output_created"] is False
    assert report["output_dir"].endswith("phase3i5_sampler_unroll_v5_reviewed")


def test_plan_cli_does_not_import_torch_or_create_outputs():
    blocker = (
        "import importlib.abc,runpy,sys;"
        "class_blocker=type('Blocker',(importlib.abc.MetaPathFinder,),"
        "{'find_spec':lambda self,fullname,path=None,target=None: "
        "(_ for _ in ()).throw(AssertionError('torch imported')) if fullname.startswith('torch') else None});"
        "sys.meta_path.insert(0,class_blocker());"
        "sys.argv=['scripts/run_e007_phase3i5_sampler_unroll.py','--plan-only'];"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    subprocess.run([sys.executable, "-c", blocker], check=True, capture_output=True, text=True)
    assert not phase3i5.Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/phase3i5_sampler_unroll_v1"
    ).exists()


def test_matched_arm_corruptions_and_resume_cursor_checks():
    identity = phase3i5.matched_update_identity(7, 3916505)
    assert identity == phase3i5.matched_update_identity(7, 3916505)
    assert identity["identity_sha256"] == hashlib.sha256(b"3916505:phase3i5-example:7").hexdigest()
    clean = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    diffusion = CoordinateVPDiffusion(500)
    gen_a = torch.Generator().manual_seed(identity["corruption_seed"])
    gen_b = torch.Generator().manual_seed(identity["corruption_seed"])
    times = torch.tensor([identity["timestep"]])
    a = diffusion.make_training_batch(clean, mask, timesteps=times, generator=gen_a)
    b = diffusion.make_training_batch(clean, mask, timesteps=times, generator=gen_b)
    assert torch.equal(a.noisy_coordinates, b.noisy_coordinates)
    assert torch.equal(a.coordinate_v_target, b.coordinate_v_target)
    phase3i5.validate_resume_cursor(10, 10, [0], "paired_boundary")  # recover update-10 evaluation
    phase3i5.validate_resume_cursor(25, 25, [0, 10], "paired_boundary")  # recover update-25 evaluation
    with pytest.raises(ValueError):
        phase3i5.validate_resume_cursor(11, 11, [0], "paired_boundary")


def test_interrupt_resume_is_byte_equivalent_for_metrics_and_state(tmp_path):
    def train(interrupt: bool) -> tuple[bytes, str]:
        torch.manual_seed(991)
        model = LinearDenoiser()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        records = []
        for update in range(1, 5):
            torch.manual_seed(1000 + update)
            x = torch.randn(1, 4, 3)
            optimizer.zero_grad(set_to_none=True)
            loss = (model.scale * x).square().mean()
            loss.backward()
            optimizer.step()
            records.append({"update": update, "loss": float(loss.detach())})
            if interrupt and update == 2:
                state = {
                    "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                    "scheduler": {},
                    "scaler": {},
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": None,
                    "update": update,
                    "data_cursor": update,
                    "update_identity": {"update": update},
                    "protected_hashes": {"checkpoint": "pinned"},
                }
                state["state_hash"] = phase3i5.state_hash(state)
                path = tmp_path / "resume.pt"
                torch.save(state, path)
                state = torch.load(path, weights_only=False)
                phase3i5.verify_resume_state(state, {"checkpoint": "pinned"})
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                torch.set_rng_state(state["cpu_rng"])
        return json.dumps(records, sort_keys=True, separators=(",", ":")).encode(), phase3i5.state_hash(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        )

    uninterrupted = train(False)
    resumed = train(True)
    assert resumed == uninterrupted


def test_memory_telemetry_rejects_invalid_and_over_limit_values():
    snapshot = {
        "rss_current_mib": 10.0,
        "rss_peak_mib": 10.0,
        "current_allocated_mib": 10.0,
        "current_reserved_mib": 20.0,
        "peak_allocated_mib": 30.0,
        "peak_reserved_mib": 40.0,
        "device_capacity_mib": 8192.0,
    }
    phase3i5.validate_memory_telemetry(snapshot, {"rss": 100, "allocated": 100, "reserved": 100})
    with pytest.raises(MemoryError):
        phase3i5.validate_memory_telemetry(
            {**snapshot, "peak_reserved_mib": 101}, {"rss": 100, "allocated": 100, "reserved": 100}
        )
    with pytest.raises(ValueError):
        phase3i5.validate_memory_telemetry(
            {**snapshot, "current_allocated_mib": float("nan")}, {"rss": 100, "allocated": 100, "reserved": 100}
        )
    phase3i5.validate_memory_telemetry(
        {**snapshot, "rss_current_mib": 99.0, "rss_peak_mib": 9000.0},
        {"rss": 100, "allocated": 100, "reserved": 100},
    )


def test_direct_phase_peak_aggregation_uses_maximum_and_never_sum():
    final = {"peak_allocated_mib": 4.0, "peak_reserved_mib": 8.0}
    phases = [
        {"peak_allocated_mib": 20.0, "peak_reserved_mib": 30.0},
        {"peak_allocated_mib": 25.0, "peak_reserved_mib": 35.0},
    ]
    result = phase3i5.aggregate_direct_phase_peaks(phases, final)
    assert result["peak_allocated_mib"] == 25.0
    assert result["peak_reserved_mib"] == 35.0


def test_v2_contract_is_fresh_non_authorizing_and_keeps_two_transitions():
    cfg = yaml.safe_load(open("configs/e007_phase3i5_sampler_unroll_v2.yaml"))
    result = validate_contract("configs/e007_phase3i5_sampler_unroll_v2.yaml")
    assert cfg["unroll_transitions"] == 2 and cfg["batch_size"] == 1
    assert cfg["numerics"]["activation_checkpointing"] is True
    assert cfg["numerics"]["checkpoint_preserve_rng_state"] is True
    assert cfg["memory_limit_mib"]["allocated"] == 6144
    assert result["authorization"]["authorizes_training"] is False
    planned = plan("configs/e007_phase3i5_sampler_unroll_v2.yaml")
    assert planned["workload_counts"]["audit_forwards_by_arm"] == {
        "continued_v_only": 25,
        "v_plus_sampler_unroll": 50,
    }
    assert planned["workload_counts"]["total_forwards"] == 20210
    assert planned["cuda_initialized"] is False and planned["output_created"] is False


def test_incident_record_preserves_v1_config_and_failure_hashes():
    incident = json.loads(
        open(
            "reports/experiments/E007_matrix_sequence_cogeneration/phase3i5_incidents/"
            "e007_phase3i5_memory_incident_v1.json"
        ).read()
    )
    assert incident["immutable_v1_config"]["sha256"] == phase3i5.file_sha256(
        "configs/e007_phase3i5_sampler_unroll_v1.yaml"
    )
    for item in incident["smoke_directory_inventory"]:
        assert phase3i5.file_sha256(item["path"]) == item["sha256"]
    assert incident["observation"]["peak_allocated_mib"] == 8136.74462890625
    assert incident["optimizer_steps"]["control_completed"] is True
    assert incident["optimizer_steps"]["two_transition_unroll_completed"] is True


def test_smoke_dispatch_is_real_lifecycle_entrypoint(monkeypatch):
    called = {}

    def fake_smoke(path, config, validation):
        called["path"] = path
        called["config"] = config
        return {"status": "mocked_disposable_smoke"}

    monkeypatch.setattr(phase3i5, "run_cuda_smoke", fake_smoke)
    result = phase3i5.execute("configs/e007_phase3i5_sampler_unroll_v1.yaml", "cuda-memory-smoke")
    assert result["status"] == "mocked_disposable_smoke"
    assert called["config"]["smoke_output_dir"].endswith("phase3i5_cuda_smoke_v1")


def test_failure_is_recorded_and_smoke_publication_is_separate(tmp_path, monkeypatch):
    config = yaml.safe_load(open("configs/e007_phase3i5_sampler_unroll_v1.yaml"))
    config["output_dir"] = str(tmp_path / "scientific-final")
    config["staging_dir"] = str(tmp_path / "scientific-staging")
    config["smoke_output_dir"] = str(tmp_path / "smoke-final")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        phase3i5.execute(config_path, "execute")
    failure = json.loads((tmp_path / "scientific-staging" / "failure.json").read_text())
    assert failure["status"] == "failed_closed"
    assert not (tmp_path / "scientific-final").exists()


def test_full_cpu_mock_execution_stops_at_25_and_publishes_review_gate(tmp_path, monkeypatch):
    base = yaml.safe_load(open("configs/e007_phase3i5_sampler_unroll_v1.yaml"))
    base["output_dir"] = str(tmp_path / "final")
    base["staging_dir"] = str(tmp_path / "stage")
    base["smoke_output_dir"] = str(tmp_path / "smoke")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(base, sort_keys=False))
    model = LinearDenoiser()

    class FakeOptimizer:
        def __init__(self, parameters):
            self.parameters = list(parameters)
            self.steps = 0

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            self.steps = state["steps"]

        def zero_grad(self, set_to_none=True):
            for parameter in self.parameters:
                parameter.grad = None

        def step(self):
            with torch.no_grad():
                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-0.01)
            self.steps += 1

    class FakeScheduler:
        def __init__(self):
            self.steps = 0

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            self.steps = state["steps"]

        def step(self):
            self.steps += 1

    optimizer = FakeOptimizer(model.parameters())
    scheduler = FakeScheduler()
    source_checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
    }
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [torch.zeros(1, dtype=torch.uint8)])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda *_: None)
    monkeypatch.setattr(
        phase3i5,
        "_memory_snapshot",
        lambda _device: {
            "rss_peak_mib": 20.0,
            "rss_current_mib": 20.0,
            "live_tensor_objects": 50,
            "current_allocated_mib": 10.0,
            "current_reserved_mib": 20.0,
            "peak_allocated_mib": 30.0,
            "peak_reserved_mib": 40.0,
            "device_capacity_mib": 8192.0,
            "phase_peak_allocated_mib": 30.0,
            "phase_peak_reserved_mib": 40.0,
            "run_wide_peak_allocated_mib": 30.0,
            "run_wide_peak_reserved_mib": 40.0,
        },
    )
    monkeypatch.setattr(
        phase3i5,
        "_resolve_data",
        lambda _cfg: (type("Auth", (), {"observed_shard_hashes": {"shard": "sha"}})(), [], []),
    )
    panel = json.loads(open("configs/e007_phase3i5_development_panel_v1.json").read())
    panel_rows = {
        item["sample_id"]: {"sample_id": item["sample_id"], "sequence_length": item["observed_length"]}
        for item in panel["records"]
    }
    monkeypatch.setattr(phase3i5, "_load_rows_by_id", lambda _dataset, wanted: {key: panel_rows[key] for key in wanted})
    monkeypatch.setattr(phase3i5, "_select_training_rows", lambda *_: [{"sample_id": f"train-{i}"} for i in range(25)])
    monkeypatch.setattr(phase3i5, "_prepare", lambda *_: {})
    monkeypatch.setattr(phase3i5, "_load_model_optimizer", lambda *_: (model, optimizer, scheduler, source_checkpoint))
    train_calls = {"continued_v_only": 0, "v_plus_sampler_unroll": 0}

    def fake_train_step(active_model, active_optimizer, _diffusion, _prepared, _t, _seed, _source, arm, **_kwargs):
        train_calls[arm] += 1
        active_optimizer.zero_grad(set_to_none=True)
        loss = active_model.scale.square()
        loss.backward()
        active_optimizer.step()
        return (
            {"v_mse": 1.0, "immediate_x0_geometry": 0.1, "post_transition_x0_geometry": 0.1, "total": 1.004},
            {"v_mse": 0.2},
            2 if arm == "continued_v_only" else 4,
            3 if arm == "continued_v_only" else 5,
        )

    monkeypatch.setattr(phase3i5, "_train_step", fake_train_step)
    eval_record = {
        "i_plus_1_distance_rmse_angstrom": 0.5,
        "i_plus_2_distance_rmse_angstrom": 0.5,
        "i_plus_3_distance_rmse_angstrom": 0.5,
        "valid_bond_fraction": 0.5,
        "valid_residue_fraction": 0.5,
        "discontinuity_fraction": 0.1,
        "v_mse": 1.0,
        "radius_of_gyration_error_angstrom": 0.2,
    }
    monkeypatch.setattr(
        phase3i5,
        "_evaluate_arm",
        lambda _m, _d, rows, _s, _dev, boundary: [
            {"identity": key, "boundary": boundary, "observed_length": value["sequence_length"], **eval_record}
            for key, value in sorted(rows.items())
        ],
    )
    sample_record = {
        "adjacent_distance_rmse_angstrom": 0.5,
        "valid_bond_fraction": 0.5,
        "radius_of_gyration_angstrom": 10.0,
        "contact_density_at_8a": 0.2,
        "within_length_diversity_descriptor_rmse": 1.0,
    }
    monkeypatch.setattr(
        phase3i5,
        "_sample_arm",
        lambda _m, _d, _dev, rows: [{"identity": row["identity"], **sample_record} for row in rows],
    )
    monkeypatch.setattr(
        phase3i5, "coordinate_model_execution_context", lambda *_args, **_kwargs: contextlib.nullcontext({})
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    result = phase3i5.execute(config_path, "execute")
    assert train_calls == {"continued_v_only": 25, "v_plus_sampler_unroll": 25}
    assert result["status"] == "completed_non_authorizing_awaiting_checkpoint25_review"
    assert sorted(result["evaluations"]) == ["0", "10", "25"]
    assert sorted(result["production_sampler_evaluations"]) == ["0", "25"]
    assert result["workload_counts"]["total_forwards"] == 20_210
    assert result["workload_counts"]["backward_calls"]["component_gradient_audits"] == 150
    assert result["workload_counts"]["backward_calls"]["total_objective_backward"] == 50
    assert json.loads((tmp_path / "final" / "checkpoint25_review.json").read_text())["extension_authorized"] is False
    assert not (tmp_path / "stage").exists()


def test_atomic_publication(tmp_path):
    staging, output = tmp_path / "stage", tmp_path / "final"
    staging.mkdir()
    (staging / "complete.json").write_text("{}")
    phase3i5.atomic_publish(staging, output)
    assert (output / "complete.json").is_file() and not staging.exists()
    with pytest.raises(FileExistsError):
        phase3i5.atomic_publish(output, output)
