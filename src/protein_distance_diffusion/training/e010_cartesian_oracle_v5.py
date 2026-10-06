"""Cartesian variable cross-check; historical v4 solver and telemetry unchanged."""

from types import FunctionType

import torch

from ..models.e010_hybrid_local import local_representation
from . import e010_bounded_oracle_v4 as v4
from .e010_phase4d_objective_v2 import smooth_bounded_local

K = v4.K
S_MAX = v4.S_MAX
initialize = v4.initialize
metric_row = v4.metric_row
summarize = v4.summarize
projected_stationarity = v4.projected_stationarity


def trajectory(pg, mask, variables):
    """Physical v=.04*z, preserving v4's dimensionless optimizer variable scale.

    Cartesian variables consume no frame or local features. The reviewed
    representation routine is reused solely to obtain the identical Boolean
    eligibility state, with no frame/feature derivatives in this trajectory.
    bounded_local is a compatibility key for the unchanged projected-residual
    function: its value here is the bounded Cartesian correction tensor.
    """
    if variables.shape != (K, *pg.shape):
        raise ValueError("expected four independent Cartesian correction fields")
    current = pg.detach()
    states, steps = [current], []
    for t in range(K):
        with torch.no_grad():
            rep = local_representation(current, mask)
        delta = smooth_bounded_local(S_MAX * variables[t], S_MAX)
        limit = torch.nextafter(delta.new_tensor(S_MAX), delta.new_tensor(0.0))
        size = delta.norm(dim=-1, keepdim=True)
        delta = delta * torch.minimum(torch.ones_like(size), limit / size.clamp_min(1e-12))
        delta = delta * rep["eligible"][..., None]
        current = current + delta
        steps.append(
            {
                "bounded_local": delta,
                "bounded_cartesian": delta,
                "delta": delta,
                "eligible": rep["eligible"],
                "frame": rep["frame"],
            }
        )
        states.append(current)
    return {"prediction": current, "states": states, "steps": steps}


# Clone the globals dictionary rather than mutating the historical module.
# Exactly the same code object supplies initialization, optimizer, closure,
# convergence checks, stopping and history. Only trajectory's binding changes.
solve = FunctionType(v4.solve.__code__, {**v4.solve.__globals__, "trajectory": trajectory}, name="solve")
solve.__kwdefaults__ = v4.solve.__kwdefaults__


def contribution(row):
    return (
        sum(row["local_mse"].values()) / 3 / 60 + 16.8 * row["raw_cartesian"] / 60 + 2 * row["chiral_error_sum"] / 13029
    )


def convergence_pairs(old, new):
    counts = {
        k: 0 for k in ("nonconverged_to_converged", "converged_to_nonconverged", "converged_both", "nonconverged_both")
    }
    for a, b in zip(old, new, strict=True):
        x, y = a["optimizer"]["converged"], b["optimizer"]["converged"]
        key = (
            "converged_both"
            if x and y
            else "nonconverged_both"
            if not x and not y
            else "converged_to_nonconverged"
            if x
            else "nonconverged_to_converged"
        )
        counts[key] += 1
    return counts


def classify(converged, high_converged, gains, safety):
    # Frozen before any panel execution: >=80% overall, >=90% at primary 450.
    if converged < 48 or high_converged < 18:
        return "CART-O5"
    if gains["450"] >= 5:
        return "CART-O1" if safety["450"] else "CART-O3"
    if all(gains[c] >= 5 and safety[c] for c in ("50", "250")):
        return "CART-O4"
    return "CART-O2" if safety["450"] else "CART-O5"
