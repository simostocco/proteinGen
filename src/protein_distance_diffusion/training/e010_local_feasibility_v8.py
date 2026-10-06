"""K=8-only intervention over the immutable V7 float64 oracle."""

import math
import time

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint

from .e010_local_feasibility_v7 import Oracle as HistoricalOracle
from .e010_local_feasibility_v7 import (
    aligned_rmsd,
    constrained_minimize,
    metric_row,
    normalized_constraints,
    physical_trajectory,
    stationarity,
    terms,
)


class Oracle(HistoricalOracle):
    def __init__(self, b, arm):
        super().__init__(b, arm)
        self.shape = (8, *b["pg"].shape)


def solve(b, arm, cfg):
    oracle = Oracle(b, arm)
    constraints = (
        [NonlinearConstraint(oracle.cfun, [-np.inf, -np.inf], [0.0, 0.0], jac=oracle.cjac, hess=oracle.chess)]
        if arm == "B"
        else []
    )
    history = []
    started = time.perf_counter()

    def callback(x, state):
        if state.nit == 1 or state.nit % cfg["history_interval"] == 0:
            p = oracle.point(x)
            history.append(
                dict(
                    iteration=int(state.nit),
                    local_objective=float(p.values["local"].detach()),
                    normalized_objective=float(state.fun),
                    optimality=float(state.optimality),
                    constraint_violation=float(state.constr_violation),
                    constraints=oracle.cfun(x).tolist(),
                    trust_radius=float(state.tr_radius),
                    barrier_parameter=float(getattr(state, "barrier_parameter", 0)),
                )
            )

    result = constrained_minimize(
        oracle.fun,
        np.zeros(math.prod(oracle.shape), dtype=np.float64),
        oracle.hess,
        constraints,
        cfg["solver"],
        callback,
    )
    p = oracle.point(result.x)
    mu = result.v[0].tolist() if arm == "B" else [0.0, 0.0]
    residual = stationarity(p.tr, b, oracle.baseline, oracle.frozen, mu, cfg)
    excess = p.constraints.detach().numpy()
    feasible = bool(np.max(excess) <= cfg["constraints"]["normalized_feasibility_tolerance"]) if arm == "B" else True
    complementarity = max(abs(m * c) for m, c in zip(mu, excess, strict=True))
    converged = bool(
        result.success
        and result.optimality <= cfg["solver"]["gtol"]
        and feasible
        and residual["normalized_ball_kkt_max"] <= cfg["convergence"]["normalized_ball_kkt_max"]
        and residual["projected_ball_mapping_max"] <= cfg["convergence"]["projected_ball_mapping_max"]
        and min(mu) >= -cfg["convergence"]["dual_feasibility_tolerance"]
        and complementarity <= cfg["convergence"]["complementarity_tolerance"]
    )
    return (
        p.tr,
        dict(
            converged=converged,
            scipy_success=bool(result.success),
            status=int(result.status),
            message=str(result.message),
            iterations=int(result.nit),
            function_evaluations=int(result.nfev),
            cg_iterations=int(result.cg_niter),
            optimality=float(result.optimality),
            initial_local=float(oracle.baseline["local"]),
            final_local=float(p.values["local"].detach()),
            constraints=excess.tolist(),
            constraint_feasible=feasible,
            multipliers=mu,
            complementarity=complementarity,
            active_constraints=[bool(abs(c) <= cfg["constraints"]["active_tolerance"]) for c in excess],
            stationarity=residual,
            history=history,
            runtime_seconds=time.perf_counter() - started,
        ),
        result.x.copy(),
    )


@torch.no_grad()
def correction_telemetry(tr):
    steps = []
    cosines = []
    for t, s in enumerate(tr["steps"]):
        n = s["delta"].norm(dim=-1)
        valid = s["eligible"]
        steps.append(
            dict(
                step=t + 1,
                eligible=int(valid.sum()),
                saturation={
                    str(k): float((n[valid] >= k * 0.04).double().mean()) if valid.any() else 0.0
                    for k in (0.9, 0.95, 0.99)
                },
            )
        )
        if t:
            prev = tr["steps"][t - 1]
            pn = prev["delta"].norm(dim=-1)
            available = valid & prev["eligible"] & (n > 1e-12) & (pn > 1e-12)
            c = (s["delta"] * prev["delta"]).sum(-1)[available] / (n[available] * pn[available])
            cosines.append(
                dict(previous_step=t, next_step=t + 1, defined=int(available.sum()), values=c.clamp(-1, 1).tolist())
            )
    return dict(steps=steps, consecutive_cosines=cosines)


__all__ = [
    "Oracle",
    "solve",
    "correction_telemetry",
    "metric_row",
    "physical_trajectory",
    "aligned_rmsd",
    "terms",
    "normalized_constraints",
]
