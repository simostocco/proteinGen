import copy

import pytest
import torch

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from protein_distance_diffusion.models.e010_hybrid_local import LocalGeometryBranch, local_representation
from protein_distance_diffusion.models.e010_recurrent_local import COUNTS, S_MAX, K, RecurrentLocalRefiner
from protein_distance_diffusion.training.e010_phase4d_diagnostic import state_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import (
    BETA,
    GAMMA,
    matched_order,
    objective_components,
    restore_checkpoint,
    save_checkpoint,
    select_capacity,
)


def example():
    torch.manual_seed(9)
    x = torch.randn(2, 8, 3)
    mask = torch.tensor([[True] * 8, [True] * 5 + [False] * 3])
    return x, mask


@pytest.mark.parametrize("variant", ["S", "M", "L"])
def test_capacity_count_architecture_feature_zero_init_parity(variant):
    x, m = example()
    model = RecurrentLocalRefiner(variant, activation_checkpoint=False)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[variant]
    assert model.input.in_features == 41 and model.head.out_features == 3
    assert len(model.blocks) == {"S": 4, "M": 6, "L": 8}[variant]
    out = model(x, m)
    assert len(out["steps"]) == K and len(out["states"]) == 5
    assert all(torch.equal(p, x) for p in out["states"])
    ref = local_representation(x, m)
    for step in out["steps"]:
        assert torch.equal(step["features"], ref["features"])
        assert len(model.blocks[0].messages) == 6
    if variant == "S":
        assert set(model.state_dict()) == set(LocalGeometryBranch().state_dict())


@pytest.mark.parametrize("variant", ["S", "M", "L"])
def test_shared_weights_recompute_bound_and_recurrent_gradients(variant):
    x, m = example()
    model = RecurrentLocalRefiner(variant, activation_checkpoint=False)
    torch.nn.init.normal_(model.head.weight, std=0.03)
    before = tuple(id(p) for p in model.parameters())
    out = model(x, m)
    assert before == tuple(id(p) for p in model.parameters())
    assert not torch.equal(out["steps"][0]["features"], out["steps"][1]["features"])
    assert all((s["bounded_local"].norm(dim=-1) < S_MAX).all() for s in out["steps"])
    assert out["delta"].norm(dim=-1).max() <= 0.160001
    assert torch.count_nonzero(out["delta"][~m]) == 0
    for t, step in enumerate(out["steps"]):
        rep = local_representation(out["states"][t], m)
        torch.testing.assert_close(step["features"], rep["features"])
        step["prediction"].retain_grad()
    out["prediction"].square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.input.weight.grad.abs().sum() > 0 and model.head.weight.grad.abs().sum() > 0
    assert all(step["prediction"].grad is not None for step in out["steps"])
    initial_distance = torch.cdist(x, x)
    final_distance = torch.cdist(out["prediction"].detach(), out["prediction"].detach())
    assert (final_distance - initial_distance).abs().max() <= 0.320002


@pytest.mark.parametrize("variant", ["S", "M", "L"])
def test_rotation_reflection_and_padding(variant):
    x, m = example()
    x = x.double()
    model = RecurrentLocalRefiner(variant, activation_checkpoint=False).double()
    torch.nn.init.normal_(model.head.weight, std=0.01)
    r = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype)
    out = model(x, m)
    moved = model(x @ r.T + 7, m)
    torch.testing.assert_close(moved["prediction"], out["prediction"] @ r.T + 7, atol=1e-12, rtol=1e-12)
    xx = x.clone()
    xx[~m] = 9000
    torch.testing.assert_close(model(xx, m)["prediction"][m], out["prediction"][m], atol=0, rtol=0)
    flipped = x * torch.tensor([-1.0, 1.0, 1.0], dtype=x.dtype)
    a = local_representation(x, m)
    b = local_representation(flipped, m)
    torch.testing.assert_close(a["features"][..., 36], -b["features"][..., 36])


@pytest.mark.parametrize("variant", ["S", "M", "L"])
def test_frozen_global_isolation_and_exact_objective(variant):
    x, m = example()
    global_model = GlobalEquivariantResidual(width=32, layers=1, heads=4, vector_channels=4).requires_grad_(False)
    with torch.no_grad():
        pg = global_model(x, m)["prediction"]
    refiner = RecurrentLocalRefiner(variant, activation_checkpoint=False)
    out = refiner(pg, m)
    b = {"pg": pg, "source": x, "target": x + 0.2, "mask": m}
    from protein_distance_diffusion.training.e010_phase4d_objective_v2 import freeze_chirality

    count = int(freeze_chirality(pg, b["target"], m)["eligible"].sum())
    losses = objective_components(out["prediction"], b, examples_total=2, quartets_total=count)
    assert BETA == 16.8 and GAMMA == 2
    torch.testing.assert_close(losses["total"], losses["local"] + 16.8 * losses["cartesian"] + 2 * losses["chiral"])
    losses["total"].backward()
    assert all(p.grad is None for p in global_model.parameters())
    assert refiner.head.weight.grad.abs().sum() > 0


def test_matched_schedule_and_selection():
    records = [{"sample_id": s, "condition": c} for s in ("b", "a") for c in (450, 50, 250)]
    order = matched_order(records)
    assert [(records[i]["sample_id"], records[i]["condition"]) for i in order] == [
        (s, c) for s in ("a", "b") for c in (50, 250, 450)
    ]
    arms = {
        n: {"updates": 500, "gates": {"all_pass": True, "local_improvement_fraction": 0.1}} for n in ("S", "M", "L")
    }
    assert select_capacity(arms)["selected"] == "S"
    arms["S"]["gates"]["all_pass"] = False
    assert select_capacity(arms)["classification"] == "CAP2"
    arms["M"]["gates"]["all_pass"] = False
    assert select_capacity(arms)["classification"] == "CAP3"
    arms["L"]["gates"]["all_pass"] = False
    assert select_capacity(arms)["classification"] == "CAP5"
    arms["M"]["gates"]["local_improvement_fraction"] = 0.2
    arms["L"]["gates"]["local_improvement_fraction"] = 0.3
    assert select_capacity(arms)["classification"] == "CAP4"
    arms["L"] = {"updates": 0, "status": "resource_infeasible"}
    assert select_capacity(arms)["classification"] == "CAP6"
    arms["S"]["gates"]["all_pass"] = True
    assert select_capacity(arms)["classification"] == "CAP1"


@pytest.mark.parametrize("variant", ["S", "M", "L"])
def test_checkpoint_resume_exact_next_update(tmp_path, variant):
    x, m = example()
    model = RecurrentLocalRefiner(variant, activation_checkpoint=False)
    op = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0)
    for _ in range(2):
        op.zero_grad()
        model(x, m)["prediction"].square().mean().backward()
        op.step()
    file = tmp_path / "resume.pt"
    save_checkpoint(file, model, op, 2, "contract", "cache", [])
    other = RecurrentLocalRefiner(variant, activation_checkpoint=False)
    oo = torch.optim.AdamW(other.parameters(), lr=9)
    state = restore_checkpoint(file, other, oo, "contract", "cache")
    assert state["successful_updates"] == 2
    for mod, opt in ((model, op), (other, oo)):
        opt.zero_grad()
        mod(x, m)["prediction"].square().mean().backward()
        opt.step()
    assert state_hash(model) == state_hash(other)
    with pytest.raises(ValueError):
        restore_checkpoint(file, other, oo, "changed", "cache")


def test_activation_checkpoint_matches_eager_gradient():
    x, m = example()
    torch.manual_seed(5)
    a = RecurrentLocalRefiner("S", activation_checkpoint=False)
    torch.nn.init.normal_(a.head.weight, std=0.01)
    b = copy.deepcopy(a)
    b.activation_checkpoint = True
    for model in (a, b):
        model(x, m)["prediction"].square().sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)


def test_final_state_mechanistic_evaluator_integration():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_e010_phase4d_objective_v2.py"
    spec = importlib.util.spec_from_file_location("v3_mechanistic_test", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    from protein_distance_diffusion.training.e010_phase4d_objective_v2 import freeze_chirality

    x, m = example()
    x = x[:1]
    m = m[:1]
    target = x + 0.2
    panel = [
        {
            "pg": x,
            "source": x,
            "target": target,
            "mask": m,
            "sample_id": "fixture",
            "condition": 50,
            "stratum": "20-64",
            "frozen": freeze_chirality(x, target, m),
        }
    ]
    model = RecurrentLocalRefiner("S", activation_checkpoint=False)
    point, grad = evaluator.evaluate(model, panel)
    assert set(grad) == {"local", "cartesian", "chiral"}
    assert point["metrics"]["overall"]["all_finite"]
    assert sum(g.abs().sum() for g in grad["cartesian"]) > 0
