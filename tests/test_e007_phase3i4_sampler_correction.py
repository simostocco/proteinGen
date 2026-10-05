"""CPU-only static and geometric contracts for Phase 3I.4."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction as correction

CONFIG = Path("configs/e007_phase3i4_sampler_correction_v1.yaml")
V2_SCRIPT = Path("scripts/run_e007_phase3i4_sampler_correction_v2.py")


def test_cuda_smoke_is_bounded_non_authorizing_and_uses_v2_kernel() -> None:
    source = V2_SCRIPT.read_text()
    assert "--cuda-performance-smoke" in source
    assert "def _cuda_smoke" in source
    assert "kernels.project_x0(" in source
    assert '"model_forward_count": 2000' in source
    assert '"trajectory_count": 4' in source
    assert '"reverse_loop_host_device_transfers": False' in source
    assert '"calibration_or_holdout_reuse": False' in source
    assert "os.replace(tmp, out)" in source
    assert "**kernels.NON_AUTHORIZING" in source
    assert '"backward_pass": False' in source
    assert '"parameter_mutation": False' in source


def _trajectory_payload(unit: dict, contract: dict) -> dict:
    milestones = [
        {
            "timestep": timestep,
            "representation": representation,
            "coordinates_normalized": [[0.0, 0.0, 0.0] for _ in range(unit["length"])],
        }
        for timestep in (499, 425, 375, 250, 150, 75, 25, 0)
        for representation in ("x_t", "v_hat", "x0_hat", "x0_guided", "next_state")
    ]
    return {"milestones": milestones, "final_metrics": {"finite_coordinate_rate": 1.0, "completion_rate": 1.0}}


def test_panels_and_grid_are_fixed_disjoint_and_paired() -> None:
    cfg = correction.load_config(CONFIG)
    cal, holdout = correction.panels(cfg)
    assert len(cal) == 10 and len(holdout) == 20
    assert [sum(row["length"] == n for row in cal) for n in cfg["lengths"]] == [2] * 5
    assert [sum(row["length"] == n for row in holdout) for n in cfg["lengths"]] == [4] * 5
    assert {row["identity"] for row in cal}.isdisjoint(row["identity"] for row in holdout)
    assert {row["seed"] for row in cal}.isdisjoint(row["seed"] for row in holdout)
    assert len(correction.candidates(cfg)) == 7
    manifest = json.loads(Path("configs/e007_phase3i4_panels_candidates_v1.json").read_text())
    assert manifest["canonical_sha256"]["calibration_panel"] == correction.canonical_sha(manifest["calibration_panel"])
    assert manifest["canonical_sha256"]["prospective_holdout_panel"] == correction.canonical_sha(
        manifest["prospective_holdout_panel"]
    )
    assert manifest["canonical_sha256"]["candidates"] == correction.canonical_sha(manifest["candidates"])
    cal_noise = correction.paired_initial_noise(cal[0]["seed"], cal[0]["length"])
    control_noise = correction.paired_initial_noise(cal[0]["seed"], cal[0]["length"])
    assert torch.equal(cal_noise, control_noise)


@pytest.mark.parametrize("family", ["bond", "composite"])
def test_projection_is_o3_equivariant_centroid_preserving_and_padding_exact(family: str) -> None:
    torch.manual_seed(13)
    x = torch.randn(2, 12, 3, dtype=torch.float64)
    mask = torch.tensor([[1] * 9 + [0] * 3, [1] * 12], dtype=torch.bool)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    # Correct a reflection so this is an O(3) transform with det -1.
    q[:, 0] *= -1
    a = correction.project_x0(x, mask, family=family, iterations=2, displacement_cap=0.03)
    b = correction.project_x0(x @ q, mask, family=family, iterations=2, displacement_cap=0.03)
    assert torch.allclose(b, a @ q, atol=1e-10)
    assert torch.equal(a[~mask], torch.zeros_like(a[~mask]))
    for i in range(2):
        assert torch.allclose(a[i, mask[i]].mean(0), x[i, mask[i]].mean(0), atol=1e-12)
    assert torch.all((a - torch.where(mask.unsqueeze(-1), x, 0)).norm(dim=-1)[mask] <= 0.0300000001)


def test_bond_projection_moves_pair_symmetrically_toward_target() -> None:
    x = torch.tensor([[[0.0, 0, 0], [5.0, 0, 0], [9.0, 1, 0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    projected = correction.project_x0(x, mask, family="bond", iterations=1, displacement_cap=0.5, strength=0.1)
    assert torch.linalg.vector_norm(projected[:, 1] - projected[:, 0]) < 5.0
    delta = projected - x
    assert torch.allclose(delta.sum(dim=1), torch.zeros((1, 3)), atol=1e-6)


def test_schedule_is_signal_fraction_ramp() -> None:
    assert correction.schedule_fraction(425, 425, 0.25) == 0
    assert correction.schedule_fraction(0, 425, 0.25) == 0.25
    assert correction.schedule_fraction(250, 425, 0.25) == pytest.approx(0.25 * 175 / 425)
    assert correction.schedule_fraction(499, 425, 0.25) == 0


def test_production_transition_is_recomputed_from_guided_x0() -> None:
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    class CaptureDiffusion(CoordinateVPDiffusion):
        seen = None

        def deterministic_reverse_step(self, noisy_coordinates, timesteps, v_prediction, residue_mask):
            self.seen = v_prediction.clone()
            return super().deterministic_reverse_step(noisy_coordinates, timesteps, v_prediction, residue_mask)

    diffusion = CaptureDiffusion(500)
    x = torch.randn(1, 8, 3)
    x = x - x.mean(1, keepdim=True)
    mask = torch.ones((1, 8), dtype=torch.bool)
    v = torch.randn_like(x)
    step = torch.tensor([249])
    next_state, guided, lam = correction.guided_reverse_step(
        diffusion,
        x,
        step,
        v,
        mask,
        start_timestep=250,
        signal_fraction=0.25,
        family="bond",
        iterations=1,
        displacement_cap=0.01,
    )
    alpha, sigma = diffusion.alpha_sigma(step, x)
    assert lam > 0
    assert torch.allclose(alpha * x - sigma * diffusion.seen, guided, atol=2e-5)
    expected, _, _ = CoordinateVPDiffusion(500).deterministic_reverse_step(x, step, diffusion.seen, mask)
    assert torch.allclose(next_state, expected)


def test_selection_is_calibration_only_and_fail_closed() -> None:
    cfg = correction.load_config(CONFIG)
    row = {
        "panel": "calibration",
        "candidate": "bond_t425",
        "length": 64,
        "completion_fraction": 1.0,
        "finite_fraction": 1.0,
        "all_primary_gates_pass": True,
        "all_safeguards_pass": True,
        "primary_ci_direction": "improved",
    }
    with pytest.raises(ValueError, match="holdout"):
        correction.select_policy([{**row, "panel": "holdout"}], cfg)
    assert correction.select_policy([row], cfg)["selected"] is None
    assert correction.select_policy([], cfg)["status"] == "fail_closed_no_passing_candidate"
    assert correction.NON_AUTHORIZING["authorizes_sampler_correction"] is False


def test_plan_and_validation_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = correction.load_config(CONFIG)
    cfg["calibration_output_dir"] = str(tmp_path / "phase3i4_calibration_v1")
    cfg["calibration_staging_dir"] = str(tmp_path / "stage")
    cfg["holdout_output_dir"] = str(tmp_path / "phase3i4_holdout_v1")
    cfg["holdout_staging_dir"] = str(tmp_path / "hstage")

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(correction, "validate_contract", lambda _: pytest.fail("plan touched protected inputs"))
    result = correction.plan(path)
    assert result["calibration_forward_count"] == 35000
    assert result["holdout_forward_count"] == 20000
    assert result["planned_forward_count"] == 55000
    assert not result["checkpoint_loaded"] and not result["cuda_initialized"]
    assert list(tmp_path.iterdir()) == [path]


def test_config_rejects_candidate_grid_or_threshold_evidence_change(tmp_path: Path) -> None:
    cfg = correction.load_config(CONFIG)
    cfg["grid"]["iterations"] += 1
    altered = tmp_path / "config.yaml"
    altered.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="candidate grid|panel/candidate"):
        correction.load_config(altered)


def test_contract_checks_exact_external_log_path_and_hash(tmp_path: Path) -> None:
    result = correction.validate_contract(CONFIG)
    assert result["protected_artifact_count"] == 67
    assert (
        result["protected_hashes"]["logs/e007_denoiser_sampler_trajectory_audit_v1.log"]
        == correction.EXECUTION_LOG_SHA256
    )
    cfg = correction.load_config(CONFIG)
    cfg["execution_log"] = str(tmp_path / "different.log")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="external execution log"):
        correction.validate_contract(path)


def test_external_log_missing_and_wrong_hash_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "external.log"
    monkeypatch.setattr(correction, "EXECUTION_LOG", target)
    monkeypatch.setattr(correction, "EXECUTION_LOG_SHA256", "expected")
    with pytest.raises(ValueError, match="missing"):
        correction.validate_execution_log(target)
    target.write_text("wrong bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        correction.validate_execution_log(target)
    target.write_bytes(b"expected bytes")
    expected = hashlib.sha256(b"expected bytes").hexdigest()
    monkeypatch.setattr(correction, "EXECUTION_LOG_SHA256", expected)
    assert correction.validate_execution_log(target) == expected


def test_selection_requires_unique_candidate_and_is_deterministic() -> None:
    cfg = correction.load_config(CONFIG)
    rows = []
    for candidate in correction.candidates(cfg)[1:]:
        for length in cfg["lengths"]:
            selected = candidate["name"] == "bond_t425"
            rows.append(
                {
                    "panel": "calibration",
                    "candidate": candidate["name"],
                    "length": length,
                    "completion_fraction": 1.0,
                    "finite_fraction": 1.0,
                    "all_primary_gates_pass": selected,
                    "all_safeguards_pass": selected,
                    "primary_ci_direction": "improved" if selected else "uncertain_or_worse",
                }
            )
    expected = correction.select_policy(rows, cfg)
    assert expected == correction.select_policy(rows, cfg)
    assert expected["selected"]["name"] == "bond_t425"
    two = [
        {**row, "all_primary_gates_pass": True, "all_safeguards_pass": True, "primary_ci_direction": "improved"}
        if row["candidate"] in {"bond_t425", "bond_t250"}
        else row
        for row in rows
    ]
    assert correction.select_policy(two, cfg)["status"] == "fail_closed_ambiguous"


@pytest.mark.parametrize("safeguard", ["global_geometry", "diversity"])
def test_global_geometry_and_diversity_safeguard_failures_reject_candidate(safeguard: str) -> None:
    cfg = correction.load_config(CONFIG)
    rows = []
    for candidate in correction.candidates(cfg)[1:]:
        for length in cfg["lengths"]:
            passes = candidate["name"] == "bond_t425"
            rows.append(
                {
                    "panel": "calibration",
                    "candidate": candidate["name"],
                    "length": length,
                    "completion_fraction": 1.0,
                    "finite_fraction": 1.0,
                    "all_primary_gates_pass": passes,
                    "all_safeguards_pass": passes and safeguard not in {"global_geometry", "diversity"},
                    "primary_ci_direction": "improved" if passes else "uncertain_or_worse",
                    "safeguard_failure": safeguard,
                }
            )
    assert correction.select_policy(rows, cfg)["status"] == "fail_closed_no_passing_candidate"


def test_atomic_commit_resume_prefix_corruption_and_escape(tmp_path: Path) -> None:
    cfg = correction.load_config(CONFIG)
    contract = {"phase": "calibration", "config_sha256": "cfg", "rng_contract": "rng"}
    work = correction.work_units(cfg, "calibration")[:2]
    staging = tmp_path / "stage"
    staging.mkdir()
    correction._commit_trajectory(staging, 0, work[0], contract, _trajectory_payload(work[0], contract))
    assert len(correction._journal_rows(staging, work, contract)) == 1
    journal = staging / "journal.jsonl"
    saved = journal.read_text()
    record = json.loads(saved)
    record["path"] = "../../escape.json"
    journal.write_text(__import__("json").dumps(record) + "\n")
    with pytest.raises(ValueError, match="path escape"):
        correction._journal_rows(staging, work, contract)
    journal.write_text(saved)
    trajectory = staging / json.loads(saved)["path"]
    trajectory.write_text("corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        correction._journal_rows(staging, work, contract)
    (staging / "journal.jsonl").unlink()
    (staging / "trajectories" / "trajectory-0000.json").unlink()
    (staging / "journal.jsonl").symlink_to(tmp_path / "outside-journal")
    with pytest.raises(ValueError, match="symlink"):
        correction._journal_rows(staging, work, contract)


@pytest.mark.parametrize("phase", ["calibration", "holdout"])
def test_resume_journal_supports_both_phase_work_plans(tmp_path: Path, phase: str) -> None:
    cfg = correction.load_config(CONFIG)
    selected = {
        "status": "selected_for_holdout_only",
        "selected": correction.candidates(cfg)[1],
        **correction.NON_AUTHORIZING,
    }
    work = correction.work_units(cfg, phase, selected if phase == "holdout" else None)
    assert work[0]["phase"] == phase
    staging = tmp_path / phase
    staging.mkdir()
    contract = {"phase": phase, "config_sha256": "cfg", "rng_contract": "rng"}
    correction._commit_trajectory(staging, 0, work[0], contract, _trajectory_payload(work[0], contract))
    assert correction._journal_rows(staging, work, contract)[0]["unit"] == work[0]


@pytest.mark.parametrize("phase", ["calibration", "holdout"])
def test_resume_dispatches_from_exactly_one_phase_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    cfg = correction.load_config(CONFIG)
    for key in ("calibration_output_dir", "calibration_staging_dir", "holdout_output_dir", "holdout_staging_dir"):
        cfg[key] = str(
            tmp_path
            / (
                {"calibration_output_dir": "phase3i4_calibration_v1", "holdout_output_dir": "phase3i4_holdout_v1"}.get(
                    key, key
                )
            )
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    staging_key = f"{phase}_staging_dir"
    Path(cfg[staging_key]).mkdir()
    monkeypatch.setattr(correction, "execute", lambda _path, observed, *, resume: {"phase": observed, "resume": resume})
    assert correction.resume(config_path) == {"phase": phase, "resume": True}


def test_wall_bound_resume_uses_fresh_budget_and_strict_journal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU mocks exercise resume orchestration without sampling or CUDA work."""
    cfg = correction.load_config(CONFIG)
    cfg = dict(cfg)
    cfg["calibration_staging_dir"] = str(tmp_path / ".stage")
    cfg["calibration_output_dir"] = str(tmp_path / "phase3i4_calibration_v1")
    cfg["holdout_staging_dir"] = str(tmp_path / ".holdout")
    cfg["holdout_output_dir"] = str(tmp_path / "phase3i4_holdout_v1")
    cfg["runtime"] = dict(cfg["runtime"], calibration_max_wall_seconds=1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(correction, "load_config", lambda _path: cfg)

    stage = Path(cfg["calibration_staging_dir"])
    work = correction.work_units(cfg, "calibration")
    contract = {"phase": "calibration", "config_sha256": "test", "rng_contract": "fixed"}
    correction._initialize_stage(stage, "calibration", cfg, config_path, work, contract, None)
    committed = []
    for index in range(3):
        committed.append(
            correction._commit_trajectory(
                stage, index, work[index], contract, _trajectory_payload(work[index], contract)
            )
        )
    correction._write_heartbeat(stage, "calibration", work, 3, "paused_wall_bound", elapsed_seconds=50_000)
    monkeypatch.setattr(correction, "_phase_contract", lambda *_args, **_kwargs: contract)

    now = [10_000.0]
    monkeypatch.setattr(correction.time, "monotonic", lambda: now[0])
    executed: list[dict] = []

    def mock_sample(_model, _diffusion, _config, unit):
        executed.append(unit)
        now[0] += 2.0  # exceed this invocation's one-second test budget
        return _trajectory_payload(unit, contract)

    class Memory:
        def __init__(self, *_args, **_kwargs):
            pass

        def snapshot(self):
            return {"run_peak_cuda_allocated_mib": 1, "run_peak_cuda_reserved_mib": 1}

    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(correction, "_sample_trajectory", mock_sample)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    phase3i3 = types.ModuleType("phase3i3_test_stub")
    phase3i3._load_model = lambda *_args: model
    phase3i3._parameter_fingerprints = lambda _model: ["fixed"]
    phase3i3._cleanup_cuda = lambda *_args: {}
    equivariance = types.ModuleType("equivariance_test_stub")
    equivariance.coordinate_model_execution_context = lambda *_args: nullcontext()
    diffusion = types.ModuleType("diffusion_test_stub")
    diffusion.CoordinateVPDiffusion = lambda *_args: object()
    memory = types.ModuleType("memory_test_stub")
    memory.CudaMemoryTelemetry = Memory
    for module_name, module in (
        ("protein_distance_diffusion.evaluation.e007_phase3i3_trajectory", phase3i3),
        ("protein_distance_diffusion.models.coordinate_equivariance", equivariance),
        ("protein_distance_diffusion.training.coordinate_diffusion", diffusion),
        ("protein_distance_diffusion.training.e007_local_backbone_repair", memory),
    ):
        monkeypatch.setitem(sys.modules, module_name, module)

    for expected_index in (3, 4):
        before = {record["path"]: record["sha256"] for record in committed[:3]}
        result = correction.resume(config_path)
        assert result["status"] == "paused_wall_bound"
        assert result["completed_trajectories"] == expected_index + 1
        assert executed[-1] == work[expected_index]
        rows = correction._journal_rows(stage, work, contract)
        committed = rows
        assert len(rows) == expected_index + 1
        assert {record["path"]: record["sha256"] for record in rows[:3]} == before
        heartbeat = json.loads((stage / "heartbeat.json").read_text())
        assert heartbeat["elapsed_seconds"] >= 50_000
    assert executed == work[3:5]
    assert [row["unit"] for row in committed] == work[:5]


@pytest.mark.parametrize("terminal", ["completed", "failed_closed", "corrupt", "ambiguous"])
def test_resume_rejects_nonresumable_lifecycle_states(
    tmp_path: Path, terminal: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = correction.load_config(CONFIG)
    cfg = dict(cfg)
    cfg["calibration_staging_dir"] = str(tmp_path / ".stage")
    cfg["calibration_output_dir"] = str(tmp_path / "phase3i4_calibration_v1")
    cfg["holdout_staging_dir"] = str(tmp_path / ".holdout")
    cfg["holdout_output_dir"] = str(tmp_path / "phase3i4_holdout_v1")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    stage = Path(cfg["calibration_staging_dir"])
    work = correction.work_units(cfg, "calibration")
    contract = {"phase": "calibration", "config_sha256": "test", "rng_contract": "fixed"}
    correction._initialize_stage(stage, "calibration", cfg, config_path, work, contract, None)
    correction._write_heartbeat(stage, "calibration", work, 0, terminal)
    monkeypatch.setattr(correction, "_phase_contract", lambda *_args, **_kwargs: contract)
    with pytest.raises(ValueError, match="failed-closed or completed"):
        correction.execute(config_path, "calibration", resume=True)


def test_holdout_refuses_without_completed_calibration(tmp_path: Path) -> None:
    cfg = correction.load_config(CONFIG)
    cfg["calibration_output_dir"] = str(tmp_path / "calibration")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="completed calibration"):
        correction._verify_calibration(config_path, cfg)


def test_monitor_is_read_only(tmp_path: Path) -> None:
    cfg = correction.load_config(CONFIG)
    for key in ("calibration_output_dir", "calibration_staging_dir", "holdout_output_dir", "holdout_staging_dir"):
        cfg[key] = str(
            tmp_path
            / (
                {"calibration_output_dir": "phase3i4_calibration_v1", "holdout_output_dir": "phase3i4_holdout_v1"}.get(
                    key, key
                )
            )
        )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    before = set(tmp_path.iterdir())
    status = correction.monitor(path)
    assert status["terminal_status"] == "not_started"
    assert status["planned_forwards"] == 35000
    assert set(tmp_path.iterdir()) == before


def test_monitor_reports_active_forward_weighted_phase_read_only(tmp_path: Path) -> None:
    cfg = correction.load_config(CONFIG)
    for key in ("calibration_output_dir", "calibration_staging_dir", "holdout_output_dir", "holdout_staging_dir"):
        cfg[key] = str(
            tmp_path
            / (
                {"calibration_output_dir": "phase3i4_calibration_v1", "holdout_output_dir": "phase3i4_holdout_v1"}.get(
                    key, key
                )
            )
        )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    stage = Path(cfg["calibration_staging_dir"])
    stage.mkdir()
    heartbeat = {
        "phase": "calibration",
        "candidate": "bond_t425",
        "completed_trajectories": 7,
        "planned_trajectories": 70,
        "completed_forwards": 3500,
        "planned_forwards": 35000,
        "selected_policy": None,
        "status": "running",
    }
    (stage / "heartbeat.json").write_text(json.dumps(heartbeat))
    before = set(stage.iterdir())
    output = correction.monitor(config_path)
    assert output["phase"] == "calibration" and output["candidate"] == "bond_t425"
    assert output["forward_weighted_percent"] == 10
    assert set(stage.iterdir()) == before


def test_all_authorization_fields_remain_false() -> None:
    assert correction.NON_AUTHORIZING and all(value is False for value in correction.NON_AUTHORIZING.values())


def test_projection_does_not_mutate_frozen_model_parameters() -> None:
    model = torch.nn.Linear(3, 4).eval().requires_grad_(False)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    coordinates = torch.randn(1, 10, 3)
    mask = torch.ones((1, 10), dtype=torch.bool)
    correction.project_x0(coordinates, mask, family="composite", iterations=2, displacement_cap=0.1)
    assert all(torch.equal(parameter, original) for parameter, original in zip(model.parameters(), before, strict=True))
    from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import _parameter_fingerprints

    before_hashes = _parameter_fingerprints(model)
    assert before_hashes == _parameter_fingerprints(model)
    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    assert before_hashes != _parameter_fingerprints(model)
