"""Matched final-state objective, telemetry, gates and resume for Phase 4D v3."""

from pathlib import Path

import torch

from .e010_phase4d import hybrid_losses
from .e010_phase4d_objective_v2 import chiral_sum, freeze_chirality

BETA = 16.8
GAMMA = 2.0
BOUNDARIES = (0, 25, 50, 100, 250, 500)


def matched_order(records):
    return sorted(range(len(records)), key=lambda i: (records[i]["sample_id"], records[i]["condition"]))


def batch(cache, indices, device):
    n = max(int(cache["lengths"][i]) for i in indices)
    ids = torch.tensor(indices, dtype=torch.long)
    return {k: cache[k].index_select(0, ids)[:, :n].to(device) for k in ("pg", "source", "target", "mask")}


def objective_components(pred, b, *, examples_total, quartets_total):
    frozen = freeze_chirality(b["pg"], b["target"], b["mask"])
    loss = hybrid_losses(pred, b["pg"], b["source"], b["target"], b["mask"])
    local = loss["local_mean"] * pred.shape[0] / examples_total
    cart = loss["cartesian"] * pred.shape[0] / examples_total
    chiral = chiral_sum(pred, b["mask"], frozen) / quartets_total
    return {"local": local, "cartesian": cart, "chiral": chiral, "total": local + BETA * cart + GAMMA * chiral}


def gates(final, baseline):
    f = final["metrics"]["overall"]
    b = baseline["metrics"]["overall"]
    change = 1 - f["mean_local_rmse"] / b["mean_local_rmse"]
    results = {
        "1_local_capacity": change >= 0.05,
        "2_offsets": all(f["local_rmse"][k] < b["local_rmse"][k] for k in ("1", "2", "3")),
        "3_conditions": all(
            v["mean_local_rmse"] < baseline["metrics"]["by_condition"][k]["mean_local_rmse"]
            for k, v in final["metrics"]["by_condition"].items()
        ),
        "4_cartesian": f["aligned_rmsd"] <= 1.01 * b["aligned_rmsd"],
        "5_chirality": f["continuous_chiral_loss"] <= b["continuous_chiral_loss"]
        and f["chirality_inversions"] <= b["chirality_inversions"]
        and f["chirality_assessable"] == b["chirality_assessable"]
        and f["chirality_assessability_lost"] == 0,
        "6_architecture": f["all_finite"]
        and not f["any_collapse"]
        and final["step_max"] <= 0.040001
        and f["displacement_max"] <= 0.160001
        and final["frame_assessability_preserved"],
        "7_length": all(
            v["mean_local_rmse"] < baseline["metrics"]["by_stratum"][k]["mean_local_rmse"]
            and v["aligned_rmsd"] <= 1.01 * baseline["metrics"]["by_stratum"][k]["aligned_rmsd"]
            for k, v in final["metrics"]["by_stratum"].items()
        ),
    }
    return {**results, "all_pass": all(results.values()), "local_improvement_fraction": change}


def select_capacity(arms):
    # Resource-infeasible arms cannot imply a completed no-benefit curve.
    for name in ("S", "M", "L"):
        if name not in arms or (arms[name].get("updates") != 500 and arms[name].get("status") != "resource_infeasible"):
            return {"selected": None, "classification": "CAP6", "reason": "capacity curve incomplete"}
    for name, classification in (("S", "CAP1"), ("M", "CAP2"), ("L", "CAP3")):
        if arms[name].get("gates", {}).get("all_pass", False):
            return {"selected": name, "classification": classification}
    if any(arms[n].get("status") == "resource_infeasible" for n in ("S", "M", "L")):
        return {
            "selected": None,
            "classification": "CAP6",
            "reason": "no feasible arm passed; incomplete capacity curve",
        }
    gains = [arms[n]["gates"]["local_improvement_fraction"] for n in ("S", "M", "L")]
    monotonic = gains[1] > gains[0] + 0.001 and gains[2] > gains[1] + 0.001
    return {"selected": None, "classification": "CAP4" if monotonic else "CAP5"}


def save_checkpoint(path, model, optimizer, update, contract_sha, cache_sha, history, statistics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "e010_recurrent_capacity_resume_v3",
        "variant": model.variant,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "successful_updates": update,
        "contract_sha256": contract_sha,
        "cache_sha256": cache_sha,
        "history": history,
        "statistics": statistics or {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    }
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)


def restore_checkpoint(path, model, optimizer, contract_sha, cache_sha):
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if (
        state["variant"] != model.variant
        or state["contract_sha256"] != contract_sha
        or state["cache_sha256"] != cache_sha
    ):
        raise ValueError("resume provenance mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    return state
