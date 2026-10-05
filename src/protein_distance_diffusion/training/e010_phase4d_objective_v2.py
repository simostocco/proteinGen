"""Signed-quartet objective and analytic common-descent calibration; no training."""

import math

import torch

from ..models.e010_hybrid_local import EPS, local_representation
from .e010_phase4d_diagnostic import dot, norm, signed_status

BETA = 16.8


def signed_triple(coords, mask):
    """Differentiable signed q indexed by central i=1..N-3 (quartet i-1..i+2).

    Unlike feature extraction, do not gate q by prediction assessability: frozen
    loss eligibility prevents prediction collapse from removing loss terms.
    """
    x = torch.where(mask[..., None], coords, 0)
    a = x[:, 1:-2] - x[:, :-3]
    b = x[:, 2:-1] - x[:, 1:-2]
    c = x[:, 3:] - x[:, 2:-1]
    denom = a.norm(dim=-1) * b.norm(dim=-1) * c.norm(dim=-1)
    return (torch.cross(a, b, dim=-1) * c).sum(-1) / denom.clamp_min(EPS**3)


@torch.no_grad()
def freeze_chirality(pg, target, mask):
    eligible, _ = signed_status(pg, target, mask)
    return {"eligible": eligible[:, 1:-2].detach(), "q_target": signed_triple(target, mask).detach()}


def chiral_sum(prediction, mask, frozen):
    q = signed_triple(prediction, mask)
    error = (q - frozen["q_target"]).square()
    return torch.where(frozen["eligible"], error, 0).sum()


def combined_gradient(local, cart, chiral, gamma):
    return [a + BETA * b + gamma * c for a, b, c in zip(local, cart, chiral, strict=True)]


def gradient_audit(gradients):
    names = list(gradients)
    dots = {a: {b: float(dot(gradients[a], gradients[b])) for b in names} for a in names}
    norms = {a: norm(gradients[a]) for a in names}
    cos = {a: {b: dots[a][b] / (norms[a] * norms[b]) if norms[a] * norms[b] else None for b in names} for a in names}
    return {"norms": norms, "dot_products": dots, "cosines": cos}


def common_descent_interval(gradients):
    """Solve gi dot (gl+beta gc+gamma gh)>0, gamma>=0, using float64 dots."""
    base = [a + BETA * b for a, b in zip(gradients["local"], gradients["cartesian"], strict=True)]
    gh = gradients["chiral"]
    lower, upper = 0.0, math.inf
    constraints = {}
    impossible = False
    for name, g in gradients.items():
        intercept = float(dot(g, base))
        slope = float(dot(g, gh))
        constraints[name] = {"intercept": intercept, "slope": slope, "inequality": "intercept + slope*gamma > 0"}
        if slope > 0:
            lower = max(lower, -intercept / slope)
        elif slope < 0:
            upper = min(upper, -intercept / slope)
        elif intercept <= 0:
            impossible = True
    zero_feasible = all(v["intercept"] > 0 for v in constraints.values())
    feasible = not impossible and (upper > lower or (zero_feasible and upper >= 0)) and upper > 0
    return {
        "feasible": feasible,
        "lower": lower,
        "upper": None if math.isinf(upper) else upper,
        "zero_feasible": zero_feasible,
        "lower_open": not (lower == 0 and zero_feasible),
        "upper_open": True,
        "constraints": constraints,
    }


def select_gamma(interval, gradients):
    if not interval["feasible"]:
        return None
    if interval["zero_feasible"]:
        return 0.0
    lo = interval["lower"]
    hi = interval["upper"] if interval["upper"] is not None else math.inf
    base = [a + BETA * b for a, b in zip(gradients["local"], gradients["cartesian"], strict=True)]
    anchor = 0.01 * norm(base) / norm(gradients["chiral"]) if lo == 0 else lo
    margin = 1e-6 * max(1.0, lo)
    floor = max(lo + margin, anchor if lo == 0 else lo + margin)
    candidates = sorted(
        m * 10.0**p
        for p in range(math.floor(math.log10(floor)) - 1, math.floor(math.log10(floor)) + 3)
        for m in (1, 2, 5)
    )
    values = (
        [v for v in candidates if v > lo + margin and v < hi - 1e-6 * max(1.0, abs(hi))]
        if math.isfinite(hi)
        else [v for v in candidates if v > lo + margin]
    )
    gamma = values[0] if values else (lo + hi) / 2 if math.isfinite(hi) else floor * 2
    total = combined_gradient(gradients["local"], gradients["cartesian"], gradients["chiral"], gamma)
    if not all(float(dot(g, total)) > 0 for g in gradients.values()):
        raise ValueError("numerically unsafe gamma; common-descent verification failed")
    return gamma


def direction_rms_slope(branch, direction, panel):
    """Exact head-only directional coordinate RMS; no extra alpha evaluation."""
    named = dict(zip((n for n, _ in branch.named_parameters()), direction, strict=True))
    if any(torch.count_nonzero(d) for name, d in named.items() if not name.startswith("head.")):
        raise ValueError("scale mapping requires proved head-only initialization direction")
    total = 0.0
    with torch.no_grad():
        for item in panel:
            rep = local_representation(item["pg"], item["mask"])
            h = branch.input(rep["features"]) * rep["eligible"][..., None]
            for block in branch.blocks:
                h = block(h, rep["eligible"])
            u = torch.nn.functional.linear(h, named["head.weight"], named["head.bias"]) * rep["eligible"][..., None]
            delta = torch.einsum("bnij,bnj->bni", rep["frame"], u)
            per = (delta.square().sum(-1) * item["mask"]).sum() / item["mask"].sum()
            total += float(per) / len(panel)
    return math.sqrt(total)


def smooth_bounded_local(u, s_max):
    """Prepared utility only; unattached to the retained hybrid architecture.

    Isotropic radial saturation commutes with orthogonal transforms, preserves
    exact zero, and has identity Jacobian at zero; ||output|| < s_max.
    """
    if not math.isfinite(s_max) or s_max <= 0:
        raise ValueError("s_max must be finite and positive")
    # Float64 norm avoids overflow for large finite float32 predictions.
    work = u.double()
    denominator = torch.hypot(torch.ones_like(work[..., :1]), work.norm(dim=-1, keepdim=True) / s_max)
    return (work / denominator).to(u.dtype)
