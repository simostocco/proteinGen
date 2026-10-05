from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from protein_distance_diffusion.training.local_geometry import phase4c_losses

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4c_local_auxiliary_v1/runner.py"
SPEC = importlib.util.spec_from_file_location("phase4c_test_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def example():
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(2, 8, 3, generator=gen)
    y = torch.randn(2, 8, 3, generator=gen)
    p = torch.randn(2, 8, 3, generator=gen).requires_grad_(True)
    mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    return p, x, y, mask


def frozen_diagnostic(p, x, y, mask):
    # Exact validated diagnostic equations, independent reference implementation.
    sq = (p.float() - y).square().mean(-1)
    per = (sq * mask).sum(1) / mask.sum(1).clamp_min(1)
    reg = (((p.float() - x).square().mean(-1)) * mask).sum(1) / mask.sum(1).clamp_min(1)
    cart = (per + 1e-5 * reg).mean()
    local = []
    for k in (1, 2, 3):
        valid = mask[:, k:] & mask[:, :-k]
        err = (
            torch.linalg.vector_norm(p[:, k:] - p[:, :-k], dim=-1)
            - torch.linalg.vector_norm(y[:, k:] - y[:, :-k], dim=-1)
        ).square()
        local.append(((err * valid).sum(1) / valid.sum(1).clamp_min(1)).mean())
    return cart, local


def test_lambda_zero_exactly_reproduces_historical_value_and_gradient():
    p, x, y, mask = example()
    got = phase4c_losses(p, x, y, mask)
    expected, _ = frozen_diagnostic(p, x, y, mask)
    assert torch.equal(got["total"], expected)
    a = torch.autograd.grad(got["total"], p, retain_graph=True)[0]
    b = torch.autograd.grad(expected, p)[0]
    assert torch.equal(a, b)


def test_auxiliary_exactly_matches_validated_reference_and_has_gradients():
    p, x, y, mask = example()
    got = phase4c_losses(p, x, y, mask, local_weight=runner.LAMBDA)
    cart, local = frozen_diagnostic(p, x, y, mask)
    for k in (1, 2, 3):
        assert torch.equal(got[f"local_{k}"], local[k - 1])
    assert torch.equal(got["total"], cart + runner.LAMBDA * sum(local) / 3)
    grad = torch.autograd.grad(got["local_mean"], p)[0]
    assert torch.isfinite(grad).all() and grad[mask].abs().sum() > 0
    assert torch.count_nonzero(grad[~mask]) == 0


def test_padding_changes_do_not_affect_losses_or_valid_gradients():
    p, x, y, mask = example()
    before = phase4c_losses(p, x, y, mask, local_weight=runner.LAMBDA)
    grad = torch.autograd.grad(before["total"], p)[0]
    q, xx, yy = p.detach().clone(), x.clone(), y.clone()
    q[~mask], xx[~mask], yy[~mask] = 9000.0, -3000.0, 2000.0
    q.requires_grad_(True)
    after = phase4c_losses(q, xx, yy, mask, local_weight=runner.LAMBDA)
    for key in before:
        assert torch.equal(before[key], after[key])
    assert torch.equal(grad, torch.autograd.grad(after["total"], q)[0])


def test_noncontiguous_endpoint_masks_exclude_only_invalid_relationships():
    p, x, y, mask = example()
    mask[0, 3] = False
    got = phase4c_losses(p, x, y, mask)
    _, local = frozen_diagnostic(p, x, y, mask)
    for k in (1, 2, 3):
        assert torch.equal(got[f"local_{k}"], local[k - 1])


def test_local_units_scale_quadratically_and_are_rigid_invariant():
    p, x, y, mask = example()
    before = phase4c_losses(p, x, y, mask)
    scaled = phase4c_losses(2 * p, 2 * x, 2 * y, mask)
    for key in before:
        torch.testing.assert_close(scaled[key], 4 * before[key])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    moved = phase4c_losses(p @ rotation + 12, x @ rotation + 12, y @ rotation + 12, mask)
    for key in ("local_1", "local_2", "local_3", "local_mean"):
        torch.testing.assert_close(moved[key], before[key], atol=2e-6, rtol=2e-6)


def test_local_rmse_matches_numpy_endpoint_evaluator():
    p, x, y, mask = example()
    got = phase4c_losses(p, x, y, mask)
    for k in (1, 2, 3):
        mses = []
        for i in range(2):
            valid = (mask[i, k:] & mask[i, :-k]).numpy()
            pp, yy = p.detach().numpy()[i], y.numpy()[i]
            err = np.linalg.norm(pp[k:] - pp[:-k], axis=-1) - np.linalg.norm(yy[k:] - yy[:-k], axis=-1)
            mses.append(np.mean(err[valid] ** 2))
        assert float(got[f"local_{k}"].detach()) == pytest.approx(np.mean(mses), abs=5e-7)


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_chains_have_finite_zero_empty_offsets(length):
    p = torch.ones(1, length, 3, requires_grad=True)
    got = phase4c_losses(p, p.detach(), p.detach(), torch.ones(1, length, dtype=torch.bool))
    assert got["local_mean"] == 0
    assert torch.isfinite(torch.autograd.grad(got["local_mean"], p)[0]).all()


def test_both_arms_receive_identical_pinned_batches_noise_and_masks():
    rows = [
        {"length": n, "prediction": np.full((n, 3), i, np.float32), "target": np.full((n, 3), i + 2, np.float32)}
        for i, n in enumerate((3, 7, 5))
    ]
    a = runner.batch_tensors(rows, "cpu")
    b = runner.batch_tensors(copy.deepcopy(rows), "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_schedule_preserves_cached_order_and_does_not_regenerate_noise():
    records = []
    for u in range(1, 365):
        for s in runner.pc.STRATA:
            for pos in reversed(range(18)):
                records.append(
                    {
                        "split": "train",
                        "schedule_update": u,
                        "stratum": s,
                        "microbatch_position": pos,
                        "timestep": (50, 250, 450)[pos % 3],
                    }
                )
    queues, metadata = runner.schedule_records(records)
    assert len(metadata) == 32760
    assert [r["microbatch_position"] for r in queues[1][runner.pc.STRATA[0]]] == list(range(18))
    assert records[0]["microbatch_position"] == 17
    with pytest.raises(ValueError, match="18-example"):
        runner.schedule_records(records[:-1])


def test_resume_and_update_zero_preserve_identical_optimizer_state(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    # Small fixture construction only; production historical state is never stepped here.
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    import random

    state = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [],
    }
    expected = runner.tensors_hash(state["optimizer"])
    hashes = []
    for _ in range(2):
        other = torch.nn.Linear(3, 3)
        op = torch.optim.AdamW(other.parameters(), lr=0.01)
        runner.restore_state(other, op, state)
        assert runner.tensors_hash(op.state_dict()) == expected
        assert op.param_groups[0]["lr"] == 0.0003
        hashes.append(runner.tensors_hash(other.state_dict()))
    assert hashes[0] == hashes[1] == runner.tensors_hash(state["model"])


def test_checkpoint_roundtrip_preserves_full_resume_state(tmp_path):
    value = {
        "model": {"w": torch.randn(4)},
        "optimizer": {"state": {1: {"step": torch.tensor(1092.0), "exp_avg": torch.randn(4)}}},
        "scheduler": None,
        "continuation_update": 91,
    }
    path = tmp_path / "state.pt"
    runner.atomic_checkpoint(path, value)
    loaded = torch.load(path, weights_only=False)
    assert runner.tensors_hash(value) == runner.tensors_hash(loaded)
    assert loaded["continuation_update"] == 91 and not path.with_suffix(".tmp.pt").exists()


def test_paired_local_bootstrap_uses_identity_not_condition():
    def metric(value):
        return {
            "per_condition": [
                {"sample_id": sid, "local_distance_rmse": {str(k): value for k in (1, 2, 3)}}
                for sid in ("a", "b")
                for _ in range(3)
            ]
        }

    result = runner.paired_local(metric(2), metric(1))
    assert result["identity_count"] == 2
    assert result["mean_paired_percentage_improvement"] == 0.5
    assert result["bootstrap_ci95"] == [0.5, 0.5]
