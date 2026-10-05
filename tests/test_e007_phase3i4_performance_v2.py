from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.evaluation import e007_phase3i4_performance_v2 as v2


def _ref_angle(x: torch.Tensor, mask: torch.Tensor, strength: float, target: float) -> torch.Tensor:
    x1 = x.detach().clone().requires_grad_(True)
    left = x1[:, :-2] - x1[:, 1:-1]
    right = x1[:, 2:] - x1[:, 1:-1]
    u = left / torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1e-8)
    w = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-8)
    theta = torch.acos((u * w).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
    valid = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    loss = torch.where(valid, 0.5 * (theta - target).square(), 0.0).sum()
    grad = torch.autograd.grad(loss, x1)[0]
    delta = -strength * grad
    m = mask.unsqueeze(-1)
    delta = delta - (delta * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    return torch.where(m, delta, 0.0)


def test_analytic_angle_matches_autograd_reference() -> None:
    torch.manual_seed(22)
    x = torch.randn(2, 9, 3, dtype=torch.float64)
    mask = torch.tensor([[1] * 7 + [0] * 2, [1] * 9], dtype=torch.bool)
    expected = _ref_angle(x, mask, 0.07, 1.9198621771937625)
    actual = v2.angle_delta(x, mask, 0.07)
    assert torch.allclose(actual, expected, atol=1e-9, rtol=1e-8)
    assert not actual.requires_grad and actual.grad_fn is None


@pytest.mark.parametrize("family", ["bond", "composite"])
def test_kernels_are_o3_equivariant_centroid_preserving_masked_and_capped(family: str) -> None:
    torch.manual_seed(10)
    x = torch.randn(2, 20, 3, dtype=torch.float64)
    mask = torch.tensor([[1] * 13 + [0] * 7, [1] * 20], dtype=torch.bool)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q[:, 0] *= -1
    a = v2.project_x0(x, mask, family=family, iterations=2, displacement_cap=0.04)
    b = v2.project_x0(x @ q, mask, family=family, iterations=2, displacement_cap=0.04)
    assert torch.allclose(b, a @ q, atol=1e-9)
    assert torch.equal(a[~mask], torch.zeros_like(a[~mask]))
    for i in range(2):
        assert torch.allclose(a[i, mask[i]].mean(0), x[i, mask[i]].mean(0), atol=1e-11)
    motion = (a - torch.where(mask.unsqueeze(-1), x, 0)).norm(dim=-1)
    assert torch.all(motion[mask] <= 0.0400000001)
    assert not a.requires_grad and a.grad_fn is None


def test_cached_and_uncached_topology_are_equal() -> None:
    torch.manual_seed(5)
    x = torch.randn(1, 33, 3)
    m = torch.ones(1, 33, dtype=torch.bool)
    cached = v2.project_x0(x, m, family="composite", iterations=2, displacement_cap=0.1)
    v2.clear_kernel_cache()
    uncached = v2.project_x0(x, m, family="composite", iterations=2, displacement_cap=0.1)
    assert torch.equal(cached, uncached)


def test_exact_sparse_schedule_and_hashes() -> None:
    expected = {
        425: list(range(425, 152, -16)) + list(range(150, 50, -8)) + list(range(50, 0, -2)) + [0],
        250: list(range(250, 152, -16)) + list(range(150, 50, -8)) + list(range(50, 0, -2)) + [0],
        150: list(range(150, 50, -8)) + list(range(50, 0, -2)) + [0],
    }
    assert {n: v2.sparse_timesteps(n) for n in expected} == expected
    for n, values in expected.items():
        assert (
            v2.schedule_hashes()[str(n)]
            == hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
        )


def test_correction_source_has_no_transfer_or_autograd_operations() -> None:
    source = Path("src/protein_distance_diffusion/evaluation/e007_phase3i4_performance_v2.py").read_text()
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not ({"cpu", "cuda", "item", "numpy", "backward", "autograd", "to"} & names)
    assert "torch.no_grad()" in source


def test_reuse_manifest_acceptance_and_rejection(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    (root / "trajectories").mkdir(parents=True)
    candidates = [{"candidate": f"c{i}", "candidate_sha256": f"h{i}"} for i in range(4)]
    identities = [{"identity": f"id{i}", "index": i, "length": 64, "seed": 100 + i} for i in range(10)]
    rows = []
    for c in candidates:
        for identity in identities:
            index = len(rows)
            rel = f"trajectories/trajectory-{index:04d}.json"
            payload = f"hashed evidence {index}".encode()
            (root / rel).write_bytes(payload)
            unit = {"phase": "calibration", **c, **identity}
            rows.append({"unit": unit, "path": rel, "sha256": hashlib.sha256(payload).hexdigest()})
    (root / "journal.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    base = {"expected_reusable_candidates": candidates, "calibration_panel": identities}
    accepted = v2.verify_reuse(root, {**base, "v1_sampler_sha256": "sampler-hash"})
    assert accepted["accepted"] and len(accepted["units"]) == 40
    rejected = v2.verify_reuse(root, {**base, "v1_sampler_sha256": None})
    assert not rejected["accepted"] and "implementation hash" in rejected["reason"]


def test_holdout_stays_untouched_and_authorizations_false() -> None:
    assert all(value is False for value in v2.NON_AUTHORIZING.values())
    source = Path("src/protein_distance_diffusion/evaluation/e007_phase3i4_performance_v2.py").read_text()
    assert "holdout" not in source


def test_v2_production_cli_dispatches_lifecycle_modes(monkeypatch, capsys) -> None:
    import sys
    import types
    from pathlib import Path

    import yaml

    from scripts import run_e007_phase3i4_sampler_correction_v2 as cli

    calls = []
    fake = types.SimpleNamespace(
        validate_contract=lambda path: calls.append(("validate", path)) or {"status": "ok"},
        monitor=lambda path: calls.append(("monitor", path)) or {"status": "read_only"},
        execute=lambda path, phase: calls.append((phase, path)) or {"status": "dispatched"},
        resume=lambda path: calls.append(("resume", path)) or {"status": "resumed"},
        plan=lambda path: calls.append(("plan", path)) or {"status": "planned"},
    )
    monkeypatch.setattr(cli, "lifecycle", fake)
    monkeypatch.setattr(cli, "_validate_v2_metadata", lambda cfg: {"cuda_performance_smoke_report_sha256": "pinned"})
    monkeypatch.setattr(yaml, "safe_load", lambda _text: {"version": "e007_phase3i4_sampler_correction_v2"})
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: "version: mocked")
    for flag, expected in (
        ("--calibrate", "calibration"),
        ("--holdout", "holdout"),
        ("--monitor", "monitor"),
        ("--resume", "resume"),
    ):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["run_e007_phase3i4_sampler_correction_v2.py", flag])
        cli.main()
        assert calls == [(expected, "configs/e007_phase3i4_sampler_correction_v2.yaml")]
        capsys.readouterr()


def test_v2_lifecycle_uses_optimized_sparse_kernels_and_fresh_manifest() -> None:
    from pathlib import Path

    source = Path("src/protein_distance_diffusion/evaluation/e007_phase3i4_sampler_correction_v2.py").read_text()
    sample = source[source.index("def _sample_trajectory(") : source.index("def _load_calibration_rows(")]
    assert "kernels.project_x0(" in sample
    assert "kernels.sparse_timesteps(start)" in sample
    assert "guided_reverse_step(" not in sample
    assert "reused_unit_count" in source
    assert "reuse is prohibited" in source
    assert 'len(manifest["calibration_panel"]) != 10' in source


def test_v2_holdout_refuses_without_completed_selected_calibration(tmp_path: Path) -> None:
    import yaml

    from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction_v2 as lifecycle

    config = yaml.safe_load(Path("configs/e007_phase3i4_sampler_correction_v2.yaml").read_text())
    config["calibration_output_dir"] = str(tmp_path / "missing-calibration-v2")
    config["calibration_staging_dir"] = str(tmp_path / ".missing-calibration-v2.inprogress")
    config["holdout_output_dir"] = str(tmp_path / "missing-holdout-v2")
    config["holdout_staging_dir"] = str(tmp_path / ".missing-holdout-v2.inprogress")
    isolated_config = tmp_path / "isolated_config.yaml"
    isolated_config.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="holdout requires completed calibration publication"):
        lifecycle.execute(isolated_config, "holdout")


def test_v2_monitor_is_read_only_and_paths_are_versioned(tmp_path: Path) -> None:
    import yaml

    from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction_v2 as lifecycle

    config = yaml.safe_load(Path("configs/e007_phase3i4_sampler_correction_v2.yaml").read_text())
    for phase in lifecycle.PHASES:
        config[f"{phase}_output_dir"] = str(tmp_path / f"{phase}_v2")
        config[f"{phase}_staging_dir"] = str(tmp_path / f".{phase}_v2.inprogress")
    isolated_config = tmp_path / "isolated_config.yaml"
    isolated_config.write_text(yaml.safe_dump(config, sort_keys=False))
    cfg = lifecycle.load_config(isolated_config)
    paths = [lifecycle.phase_paths(cfg, phase) for phase in lifecycle.PHASES]
    assert all("_v2" in staging.name and "_v2" in final.name for staging, final in paths)
    assert all(not path.exists() for pair in paths for path in pair)
    before = [(path, path.exists()) for pair in paths for path in pair]
    state = lifecycle.monitor(isolated_config)
    assert state["terminal_status"] == "not_started"
    assert before == [(path, path.exists()) for pair in paths for path in pair]


def test_v2_execution_refuses_existing_publication_and_incompatible_staging(tmp_path: Path) -> None:
    import yaml

    from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction_v2 as lifecycle

    config = yaml.safe_load(Path("configs/e007_phase3i4_sampler_correction_v2.yaml").read_text())
    final = tmp_path / "calibration_v2"
    stage = tmp_path / ".calibration_v2.inprogress"
    config["calibration_output_dir"] = str(final)
    config["calibration_staging_dir"] = str(stage)
    config_path = tmp_path / "isolated_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    final.mkdir()
    marker = final / "preserve.txt"
    marker.write_text("existing publication")
    with pytest.raises(FileExistsError, match="phase output already exists"):
        lifecycle.execute(config_path, "calibration")
    assert marker.read_text() == "existing publication"

    final.rename(tmp_path / "prior_calibration_v2")
    stage.mkdir()
    (stage / "contract.json").write_text("{}")
    (stage / "heartbeat.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="resume contract/config/checkpoint/numerics/RNG/panel mismatch"):
        lifecycle.execute(config_path, "calibration", resume=True)


def test_v2_journal_requires_atomic_strict_work_prefix(tmp_path: Path) -> None:
    from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction_v2 as lifecycle

    stage = tmp_path / ".phase3i4_calibration_v2.inprogress"
    stage.mkdir()
    unit = {
        "phase": "calibration",
        "candidate": "control",
        "candidate_sha256": "candidate-hash",
        "identity": "cal-n64-00",
        "length": 64,
        "index": 0,
        "seed": 1,
    }
    contract = {"phase": "calibration", "config_sha256": "config-hash"}
    metrics = {"finite_coordinate_rate": 1.0, "completion_rate": 1.0}
    coords = [[0.0, 0.0, 0.0] for _ in range(unit["length"])]
    milestones = [
        {"timestep": timestep, "representation": representation, "coordinates_normalized": coords}
        for timestep in (499, 425, 375, 250, 150, 75, 25, 0)
        for representation in ("x_t", "v_hat", "x0_hat", "x0_guided", "next_state")
    ]
    trajectory = {"milestones": milestones, "final_metrics": metrics}
    lifecycle._commit_trajectory(stage, 0, unit, contract, trajectory)
    assert len(lifecycle._journal_rows(stage, [unit], contract)) == 1
    extra = stage / "trajectories" / "trajectory-0001.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="uncommitted trajectory"):
        lifecycle._journal_rows(stage, [unit], contract)


def test_v2_resume_preserves_paused_prefix_and_fresh_deadline_contract() -> None:
    from pathlib import Path

    source = Path("src/protein_distance_diffusion/evaluation/e007_phase3i4_sampler_correction_v2.py").read_text()
    execute = source[source.index("def execute(") : source.index("def resume(")]
    assert '"paused_wall_bound"' in execute
    assert '"running", "paused_wall_bound"' in execute
    assert "invocation_started = time.monotonic()" in execute
    assert "invocation_elapsed = time.monotonic() - invocation_started" in execute
    assert "elapsed_before + invocation_elapsed" in execute
    assert '"failed_closed"' in execute
    assert "_publish_phase(" in execute
