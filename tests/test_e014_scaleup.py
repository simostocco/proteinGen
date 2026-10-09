import numpy as np
import pytest
import torch

from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import (
    E013UnconditionalEquivariantDenoiser,
)
from protein_distance_diffusion.models.e014_capacity import LADDER, E014CapacityDenoiser
from protein_distance_diffusion.training.e014_runtime import (
    atomic_checkpoint,
    canonical_hash,
    d_path,
    epoch_schedule,
    load_checkpoint,
    noise_seed,
    restore_rng,
    rng_state,
    select_panels,
)

torch.set_num_threads(1)


def test_small_exact_regression():
    torch.manual_seed(1414)
    old = E013UnconditionalEquivariantDenoiser()
    torch.manual_seed(1414)
    new = E014CapacityDenoiser()
    assert sum(p.numel() for p in new.parameters()) == 980794
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    x = torch.randn(1, 20, 3)
    args = x, torch.tensor([50]), torch.tensor([20]), torch.ones(1, 20, dtype=torch.bool)
    torch.testing.assert_close(old(*args), new(*args), atol=0, rtol=0)


def test_ladder_counts():
    assert [sum(p.numel() for p in E014CapacityDenoiser(**config).parameters()) for config in LADDER.values()] == [
        2378155,
        5119069,
        7867597,
    ]


def test_epoch_coverage_resume():
    lengths = np.array([20, 30, 64, 65, 128, 200, 256, 300, 384, 400, 500] * 7)
    schedule = epoch_schedule(lengths, [4, 2, 1, 1, 1], 14014, 1)
    assert schedule == epoch_schedule(lengths, [4, 2, 1, 1, 1], 14014, 1)
    assert schedule != epoch_schedule(lengths, [4, 2, 1, 1, 1], 14014, 2)
    exposures = np.zeros(len(lengths), dtype=np.int32)
    position = len(schedule) // 2
    for batch in schedule[:position]:
        exposures[batch] += 1
    saved = exposures.copy()
    for batch in schedule[position:]:
        saved[batch] += 1
    assert saved.min() == saved.max() == 1
    for batch in epoch_schedule(lengths, [4, 2, 1, 1, 1], 14014, 2):
        saved[batch] += 1
    assert saved.min() == saved.max() == 2


def test_atomic_resume_refusals(tmp_path):
    contract = {"config": "a", "data": "b", "normalization": "c"}
    payload = dict(
        contract=contract, position=3, model={"w": torch.arange(4.0)}, exposures=np.array([1, 0]), optimizer={}, rng={}
    )
    path = atomic_checkpoint(tmp_path, "test", payload)
    assert not path.with_suffix(".pt.partial").exists()
    restored = load_checkpoint(tmp_path, contract)
    assert restored["position"] == 3
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, dict(contract, data="changed"))
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, contract)


def test_d_paths_and_metric_blind_panels():
    for path in ("/tmp/e014", "/mnt/c/e014", "/home/simostocco/e014"):
        with pytest.raises(ValueError):
            d_path(path)
    rows = [dict(sample_id=f"{length}_{i}", length=length) for length in (20, 65, 129, 257, 385) for i in range(10)]
    dev, final = select_panels(rows, 2, 4)
    assert (dev, final) == select_panels(list(reversed(rows)), 2, 4)
    assert not {r["sample_id"] for r in dev} & {r["sample_id"] for r in final}
    assert canonical_hash(dev) == canonical_hash(select_panels(rows, 2, 4)[0])
    assert noise_seed("dev", "id", 490) == noise_seed("dev", "id", 490)


def test_optimizer_rng_exact_resume(tmp_path):
    torch.manual_seed(14014)
    model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    contract = {"config": "fixed", "data": "fixed", "normalization": "fixed"}

    def step(m, opt):
        x = torch.randn(4, 3)
        opt.zero_grad(set_to_none=True)
        m(x).square().mean().backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    for _ in range(3):
        step(model, optimizer)
    atomic_checkpoint(
        tmp_path,
        "mid",
        dict(
            contract=contract,
            position=3,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            exposures=np.array([1, 0]),
            rng=rng_state(),
        ),
    )
    for _ in range(3):
        step(model, optimizer)
    restored = load_checkpoint(tmp_path, contract)
    copy = torch.nn.Linear(3, 3)
    copy.load_state_dict(restored["model"])
    opt = torch.optim.AdamW(copy.parameters(), lr=0.0002, weight_decay=0.01)
    opt.load_state_dict(restored["optimizer"])
    restore_rng(restored["rng"])
    for _ in range(3):
        step(copy, opt)
    for a, b in zip(model.parameters(), copy.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_rolling_checkpoint_pointer_atomicity(tmp_path, monkeypatch):
    from protein_distance_diffusion.training import e014_runtime

    contract = {"config": "fixed"}
    payload = dict(contract=contract, position=1, model={"w": torch.ones(1)}, exposures=np.array([1]))
    atomic_checkpoint(tmp_path, "recovery_0", payload, rolling=True)
    original_replace = e014_runtime.os.replace

    def fail_pointer(src, dest):
        if str(dest).endswith("LATEST.json"):
            raise OSError("injected pointer commit failure")
        return original_replace(src, dest)

    monkeypatch.setattr(e014_runtime.os, "replace", fail_pointer)
    changed = dict(payload, position=2, exposures=np.array([2]))
    with pytest.raises(OSError):
        atomic_checkpoint(tmp_path, "recovery_0", changed, rolling=True)
    assert load_checkpoint(tmp_path, contract)["position"] == 1


@pytest.mark.parametrize("n", [20, 500])
def test_large_symmetry_telemetry_backward(n):
    torch.manual_seed(14014)
    model = E014CapacityDenoiser(**LADDER["large"])
    x = torch.randn(1, n, 3)
    t = torch.tensor([50])
    lengths = torch.tensor([n])
    mask = torch.ones(1, n, dtype=torch.bool)
    y = model(x, t, lengths, mask)
    y.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    torch.testing.assert_close(model(x @ rotation + 2, t, lengths, mask), y @ rotation, atol=5e-5, rtol=5e-5)
    prediction, components = model.telemetry(x, t, lengths, mask)
    c = components[0]
    assert c["global_v"].shape == c["local_effective"].shape == (n, 3)
    assert c["gate"].shape == c["reliability"].shape == (n,)
    torch.testing.assert_close(c["global_v"] + c["local_effective"], prediction[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(prediction, y, atol=0, rtol=0)
    mirror = torch.tensor([-1.0, 1, 1])
    assert (model(x * mirror, t, lengths, mask) - y * mirror).norm() > 1e-6
    padded = torch.cat([x, torch.full((1, 3, 3), float("nan"))], 1)
    pm = torch.cat([mask, torch.zeros(1, 3, dtype=torch.bool)], 1)
    torch.testing.assert_close(model(padded, t, lengths, pm)[:, :n], y, atol=0, rtol=0)
    degenerate = model(torch.zeros_like(x), t, lengths, mask)
    assert torch.isfinite(degenerate).all()
