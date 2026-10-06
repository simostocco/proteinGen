"""Continuation batching, restart and protected-state invariants."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from protein_sequence_generation.e012_continuation import (
    SOURCE_SHA256,
    ContinuationScheduler,
    IdentitySampler,
    continuation_lr,
    partition64,
    routed_microbatches,
    weighted_loss,
)
from protein_sequence_generation.metrics import sequence_cross_entropy

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/pilot_v1"


@pytest.mark.parametrize("u", [2000, 2001, 2050, 2100, 5000, 7500, 10000])
def test_restart_schedule(u):
    ck_lr = 0.0
    expected = 0.0003 * (u - 2000) / 100 if u <= 2100 else 0.0003 * (1 + math.cos(math.pi * (u - 2100) / 7900)) / 2
    assert continuation_lr(u, ck_lr) == pytest.approx(expected, abs=1e-15)
    assert continuation_lr(2000, 1e-10) == 1e-10


def test_scheduler_state_resume_and_reject_old():
    p = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.AdamW([p], lr=0.0003)
    p.square().backward()
    opt.step()
    before = opt.state[p]["exp_avg"].clone()
    sched = ContinuationScheduler(opt, 0.0)
    sched.set_update(5000)
    state = sched.state_dict()
    again = ContinuationScheduler(opt, 0.0)
    again.load_state_dict(state)
    assert again.global_update == 5000
    assert opt.param_groups[0]["lr"] == continuation_lr(5000, 0.0)
    assert torch.equal(opt.state[p]["exp_avg"], before)
    with pytest.raises((ValueError, KeyError)):
        again.load_state_dict({"last_epoch": 2000})


@pytest.mark.parametrize("capacity", [8, 16, 24, 32, 40, 48, 56, 64])
def test_exact64_partition(capacity):
    pieces = partition64(capacity)
    assert sum(pieces) == 64 and max(pieces) <= capacity


def test_unequal_microbatches_gradient_equal_protein():
    torch.manual_seed(2)
    logits = torch.randn(64, 7, 24, requires_grad=True)
    targets = torch.randint(4, 24, (64, 7))
    mask = torch.arange(7)[None, :] < torch.arange(64)[:, None] % 7 + 1
    full = sequence_cross_entropy(logits, targets, mask)
    grad = torch.autograd.grad(full, logits, retain_graph=True)[0]
    split = sum(
        weighted_loss(sequence_cross_entropy(logits[a:b], targets[a:b], mask[a:b]), b - a)
        for a, b in [(0, 40), (40, 64)]
    )
    grad2 = torch.autograd.grad(split, logits)[0]
    torch.testing.assert_close(full, split)
    torch.testing.assert_close(grad, grad2)
    padded = logits.detach().clone()
    padded[~mask] = 100.0
    torch.testing.assert_close(sequence_cross_entropy(padded, targets, mask), full.detach())


@pytest.mark.parametrize(
    "length,expected", [(64, 0), (65, 1), (128, 1), (129, 2), (256, 2), (257, 3), (384, 3), (385, 4), (500, 4)]
)
def test_length_routing_preserves_membership(length, expected):
    plan = {str(i): {"partition": partition64([64, 56, 48, 40, 32][i])} for i in range(5)}
    rows = [{"sample_id": str(i), "length": 20 if i < 63 else length} for i in range(64)]
    micro = routed_microbatches(rows, plan)
    assert [len(x) for x in micro] == plan[str(expected)]["partition"]
    assert sorted(r["sample_id"] for group in micro for r in group) == sorted(r["sample_id"] for r in rows)
    assert micro == routed_microbatches(rows, plan)


def test_sampler_continues_exact_historical_cursor_and_resume():
    order = np.random.default_rng(12012).permutation(231743)
    sampler = IdentitySampler(order, 128000)
    assert sampler.take64() == order[128000:128064].tolist()
    clone = IdentitySampler.from_state(sampler.state_dict())
    assert clone.take64() == sampler.take64()
    edge = IdentitySampler(order, 231730)
    result = edge.take64()
    assert result[:13] == order[231730:].tolist()
    assert result[13:] == np.random.default_rng(12013).permutation(231743)[:51].tolist()
    assert edge.epoch == 1 and edge.cursor == 51


def test_historical_checkpoint_contract_and_adam():
    path = ROOT / "outputs/e012_causal_rope_sequence/pilot_v1/checkpoint-02000.pt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["successful_updates"] == 2000 and ck["processed_proteins"] == 128000
    assert ck["data_cursor"] == 128000 and len(ck["data_order"]) == 231743
    assert sum(t.numel() for t in ck["model"].values()) == 15533952
    assert all(
        float(s["step"]) == 2000 and s["exp_avg"].shape == s["exp_avg_sq"].shape
        for s in ck["optimizer"]["state"].values()
    )
    assert any(s["exp_avg"].abs().sum() > 0 for s in ck["optimizer"]["state"].values())
    assert ck["optimizer"]["param_groups"][0]["betas"] == (0.9, 0.95)
    assert ck["optimizer"]["param_groups"][0]["weight_decay"] == 0.01
    assert ck["scheduler"]["last_epoch"] == 2000
    contract = json.loads((REPORT / "contract.json").read_text())
    for name, digest in contract["protected_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    # Frozen gates and shuffle/windows are protected source, not copied or redefined.
    assert "src/protein_sequence_generation/e012.py" in contract["protected_hashes"]


def test_cpu_oracle_reproduce_is_not_cuda_owner(monkeypatch):
    from types import SimpleNamespace

    from scripts import run_e012_continuation as runner

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="999999 python scripts/run_e010_phase4d_cartesian_oracle_v5.py reproduce\n"
        ),
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "is_bf16_supported", lambda: True)
    assert "reproduce" in runner.gpu_guard()
