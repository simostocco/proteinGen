"""Independent coordinate oracle, with no neural/global model construction."""

import time

import torch

from ..models.e010_hybrid_local import local_representation
from .e010_phase4d import example_metrics, hybrid_losses
from .e010_phase4d_diagnostic import aggregate_rows, signed_status
from .e010_phase4d_objective_v2 import chiral_sum, freeze_chirality, smooth_bounded_local
from .e010_recurrent_capacity import objective_components

K = 4
S_MAX = 0.04


def initialize(pg):
    """Dimensionless leaf variables; no nn.Module, Parameter, or RNG."""
    return torch.zeros((K, *pg.shape), dtype=pg.dtype, device=pg.device, requires_grad=True)


def trajectory(pg, mask, variables):
    if variables.shape != (K, *pg.shape):
        raise ValueError("expected four independent correction fields")
    current = pg.detach()
    states, steps = [current], []
    for t in range(K):
        rep = local_representation(current, mask)
        u = smooth_bounded_local(S_MAX * variables[t], S_MAX)
        limit = torch.nextafter(u.new_tensor(S_MAX), u.new_tensor(0.0))
        size = u.norm(dim=-1, keepdim=True)
        u = u * torch.minimum(torch.ones_like(size), limit / size.clamp_min(1e-12))
        u = u * rep["eligible"][..., None]
        delta = torch.einsum("bnij,bnj->bni", rep["frame"], u)
        current = current + delta
        steps.append({"bounded_local": u, "delta": delta, "eligible": rep["eligible"], "frame": rep["frame"]})
        states.append(current)
    return {"prediction": current, "states": states, "steps": steps}


def projected_stationarity(loss, tr, absolute_tolerance=1e-7):
    """Scale-free feasible-ball gradient mapping; includes downstream frame derivatives.

    Retained intermediate bounded vectors are independent perturbation points for
    this diagnostic. A step s_max/max||g|| makes the largest trial gradient step
    one radius; report maximum projected residual divided by that radius.
    """
    vectors = [s["bounded_local"] for s in tr["steps"]]
    gradients = torch.autograd.grad(loss, vectors)
    scale = max(float(g.norm(dim=-1).max()) for g in gradients)
    if scale <= absolute_tolerance:
        return 0.0
    worst = 0.0
    for u, g, step in zip(vectors, gradients, tr["steps"], strict=True):
        candidate = u.detach() - (S_MAX / scale) * g.detach()
        candidate = candidate * torch.minimum(
            torch.ones_like(candidate[..., :1]), S_MAX / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        )
        residual = (u.detach() - candidate).norm(dim=-1)
        residual = torch.where(step["eligible"], residual, 0)
        worst = max(worst, float(residual.max()) / S_MAX)
    return worst


def solve(b, *, examples_total, quartets_total, settings):
    if b["pg"].shape[0] != 1:
        raise ValueError("oracle examples must be independent")
    b = {k: v.detach() for k, v in b.items()}
    z = initialize(b["pg"])
    op = torch.optim.LBFGS(
        [z],
        lr=settings["learning_rate"],
        max_iter=1,
        max_eval=25,
        history_size=settings["history_size"],
        line_search_fn="strong_wolfe",
        tolerance_grad=0.0,
        tolerance_change=0.0,
    )
    evaluations = 0
    history = []
    start = time.perf_counter()

    def loss_of(tr):
        return objective_components(tr["prediction"], b, examples_total=examples_total, quartets_total=quartets_total)[
            "total"
        ]

    def closure():
        nonlocal evaluations
        op.zero_grad(set_to_none=True)
        value = loss_of(trajectory(b["pg"], b["mask"], z))
        if not torch.isfinite(value):
            raise RuntimeError("nonfinite oracle objective")
        value.backward()
        if not torch.isfinite(z.grad).all():
            raise RuntimeError("nonfinite oracle gradient")
        evaluations += 1
        return value

    with torch.no_grad():
        previous = b["pg"].clone()
        previous_loss = float(loss_of(trajectory(b["pg"], b["mask"], z)))
    stable = 0
    converged = False
    residual = None
    for iteration in range(1, settings["maximum_iterations"] + 1):
        op.step(closure)
        tr = trajectory(b["pg"], b["mask"], z)
        value = loss_of(tr)
        now = float(value.detach())
        movement = float((tr["prediction"].detach() - previous).norm(dim=-1).max())
        relative = abs(now - previous_loss) / max(1.0, abs(previous_loss))
        stable = (
            stable + 1
            if relative <= settings["relative_objective_tolerance"]
            and movement <= settings["coordinate_tolerance_angstrom"]
            else 0
        )
        if iteration % settings["stationarity_interval"] == 0 or iteration == settings["maximum_iterations"]:
            residual = projected_stationarity(value, tr, settings.get("absolute_gradient_tolerance", 1e-7))
            history.append(
                {
                    "iteration": iteration,
                    "objective_contribution": now,
                    "relative_objective_change": relative,
                    "coordinate_change_max_angstrom": movement,
                    "projected_gradient_residual": residual,
                    "stable_iterations": stable,
                    "closure_evaluations": evaluations,
                }
            )
            converged = (
                stable >= settings["stable_iterations_required"]
                and residual <= settings["projected_gradient_tolerance"]
            )
        previous = tr["prediction"].detach().clone()
        previous_loss = now
        if converged:
            break
    with torch.no_grad():
        final = trajectory(b["pg"], b["mask"], z)
    return final, {
        "converged": converged,
        "iterations": iteration,
        "closure_evaluations": evaluations,
        "projected_gradient_residual": residual,
        "stable_iterations": stable,
        "runtime_seconds": time.perf_counter() - start,
        "history": history,
        "termination": "registered_convergence" if converged else "maximum_iterations",
    }


@torch.no_grad()
def metric_row(prediction, b, record, *, step_delta=None):
    pg, y, mask = b["pg"], b["target"], b["mask"]
    row = example_metrics(prediction, pg, b["source"], y, mask)[0]
    raw = hybrid_losses(prediction, pg, b["source"], y, mask)
    a, inv = signed_status(prediction, y, mask)
    ba, bi = signed_status(pg, y, mask)
    common = a & ba
    rep, br = local_representation(prediction, mask), local_representation(pg, mask)
    frozen = freeze_chirality(pg, y, mask)
    delta = torch.zeros_like(pg) if step_delta is None else step_delta
    row.update(record)
    row.update(
        local_mse={str(k): float(raw[f"local_{k}"]) for k in (1, 2, 3)},
        displacement_mean_square=float(raw["displacement"]),
        chirality_common_assessable=int(common.sum()),
        chirality_common_inversions=int((inv & common).sum()),
        chirality_common_baseline_inversions=int((bi & common).sum()),
        chirality_assessability_lost=int((ba & ~a).sum()),
        chirality_assessability_gained=int((a & ~ba).sum()),
        input_frame_eligible=int(br["eligible"].sum()),
        input_frame_degenerate=int(br["degenerate"].sum()),
        chiral_error_sum=float(chiral_sum(prediction, mask, frozen)),
        chiral_loss_eligible=int(frozen["eligible"].sum()),
        step_correction_rms=float(delta[mask].square().sum(-1).mean().sqrt()),
        step_correction_max=float(delta[mask].norm(dim=-1).max()),
        frame_assessability_preserved=bool(torch.equal(rep["eligible"], br["eligible"])),
    )
    row["continuous_chiral_loss"] = row["chiral_error_sum"] / max(1, row["chiral_loss_eligible"])
    return row


def summarize(rows):
    result = aggregate_rows(rows)
    groups = [(result["overall"], rows)]
    groups += [
        (result["by_condition"][k], [r for r in rows if str(r["condition"]) == k]) for k in result["by_condition"]
    ]
    groups += [(result["by_stratum"][k], [r for r in rows if r["stratum"] == k]) for k in result["by_stratum"]]
    for summary, group in groups:
        count = sum(r["chiral_loss_eligible"] for r in group)
        summary.update(
            continuous_chiral_loss=sum(r["chiral_error_sum"] for r in group) / max(1, count),
            step_correction_rms=sum(r["step_correction_rms"] for r in group) / len(group),
            step_correction_max=max(r["step_correction_max"] for r in group),
            frame_assessability_preserved=all(r["frame_assessability_preserved"] for r in group),
        )
    return result
