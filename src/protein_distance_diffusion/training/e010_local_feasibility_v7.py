"""Float64 local-only objective with optional explicit nonlinear safety constraints."""

import math
import time

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint, minimize
from scipy.sparse.linalg import LinearOperator

from ..models.e010_hybrid_local import local_representation
from .e010_phase4d_diagnostic import signed_status
from .e010_phase4d_objective_v2 import chiral_sum, freeze_chirality
from .e010_precision_kkt_v6 import correction_trajectory, kkt, project_ball


def physical_trajectory(pg, mask, variables):
    """Same map; squared-norm algebra keeps its zero-state Hessian well defined."""
    require64(pg, variables)
    current = pg.detach()
    states, steps = [current], []
    for v in variables:
        with torch.no_grad():
            valid = local_representation(current, mask)["eligible"]
        delta = v / (1 + v.square().sum(-1, keepdim=True) / 0.04**2).sqrt()
        limit = torch.nextafter(delta.new_tensor(0.04), delta.new_tensor(0.0))
        size = delta.square().sum(-1, keepdim=True).clamp_min(1e-24).sqrt()
        delta = delta * torch.minimum(torch.ones_like(size), limit / size)
        delta = delta * valid[..., None]
        current = current + delta
        states.append(current)
        steps.append(dict(delta=delta, eligible=valid))
    return dict(prediction=current, states=states, steps=steps)


def require64(*values):
    if any(v.dtype != torch.float64 or v.device.type != "cpu" for v in values):
        raise ValueError("V7 requires CPU float64 coordinates/variables")


def aligned_rmsd(prediction, target, mask):
    """Same proper row-vector Kabsch convention as Phase 4D, fully differentiable."""
    require64(prediction, target)
    p, y = prediction[mask], target[mask]
    pc, yc = p - p.mean(0), y - y.mean(0)
    u, _, vh = torch.linalg.svd(pc.T @ yc)
    d = torch.diag(torch.stack([p.new_tensor(1.0), p.new_tensor(1.0), torch.linalg.det(u @ vh)]))
    return (pc @ (u @ d @ vh) - yc).norm() / math.sqrt(len(p))


def terms(prediction, b, frozen):
    require64(prediction, b["target"], b["source"])
    mask = b["mask"]
    p = torch.where(mask[..., None], prediction, 0)
    y = torch.where(mask[..., None], b["target"], 0)
    s = torch.where(mask[..., None], b["source"], 0)
    local = []
    for offset in (1, 2, 3):
        valid = mask[:, offset:] & mask[:, :-offset]
        error = ((p[:, offset:] - p[:, :-offset]).norm(dim=-1) - (y[:, offset:] - y[:, :-offset]).norm(dim=-1)).square()
        local.append(((error * valid).sum(1) / valid.sum(1).clamp_min(1)).mean())
    cart = (
        (((p - y).square().mean(-1) + 1e-5 * (p - s).square().mean(-1)) * mask).sum(1) / mask.sum(1).clamp_min(1)
    ).mean()
    count = frozen["eligible"].sum().clamp_min(1)
    chiral = chiral_sum(p, mask, frozen) / count
    return dict(
        local=sum(local) / 3,
        cartesian=cart,
        chiral=chiral,
        aligned=aligned_rmsd(p, y, mask),
        **{f"local_{i}": v for i, v in enumerate(local, 1)},
    )


def normalized_constraints(values, baseline):
    return torch.stack(
        [values["aligned"] / (1.01 * baseline["aligned"]) - 1, values["chiral"] / baseline["chiral"] - 1]
    )


def options(settings):
    keys = (
        "maxiter",
        "gtol",
        "xtol",
        "barrier_tol",
        "initial_tr_radius",
        "initial_constr_penalty",
        "initial_barrier_parameter",
        "initial_barrier_tolerance",
        "sparse_jacobian",
        "verbose",
    )
    return {k: settings[k] for k in keys}


def constrained_minimize(fun, x0, hess, constraints, settings, callback=None):
    return minimize(
        fun,
        x0,
        method="trust-constr",
        jac=True,
        hess=hess,
        constraints=constraints,
        callback=callback,
        options=options(settings),
    )


class Point:
    def __init__(self, x, oracle):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.z = torch.from_numpy(self.x.copy()).requires_grad_()
        self.tr = physical_trajectory(oracle.b["pg"], oracle.b["mask"], 0.04 * self.z.reshape(oracle.shape))
        self.values = terms(self.tr["prediction"], oracle.b, oracle.frozen)
        self.objective = self.values["local"] / oracle.baseline["local"]
        self.constraints = normalized_constraints(self.values, oracle.baseline)
        self.gf = torch.autograd.grad(self.objective, self.z, create_graph=True, retain_graph=True)[0]
        self.gc = (
            [torch.autograd.grad(c, self.z, create_graph=True, retain_graph=True)[0] for c in self.constraints]
            if oracle.arm == "B"
            else None
        )

    def hessian(self, gradient):
        def matvec(vector):
            d = torch.as_tensor(np.asarray(vector).reshape(-1).copy(), dtype=torch.float64)
            h = torch.autograd.grad((gradient * d).sum(), self.z, retain_graph=True)[0]
            if not torch.isfinite(h).all():
                raise RuntimeError("nonfinite exact Hessian-vector product")
            return h.detach().numpy().copy()

        return LinearOperator((len(self.x), len(self.x)), matvec=matvec, dtype=np.float64)


class Oracle:
    def __init__(self, b, arm):
        if arm not in ("A", "B"):
            raise ValueError("unknown oracle arm")
        require64(b["pg"], b["target"], b["source"])
        self.b = {k: v.detach() for k, v in b.items()}
        self.arm = arm
        self.shape = (4, *b["pg"].shape)
        self.frozen = freeze_chirality(b["pg"], b["target"], b["mask"])
        with torch.no_grad():
            self.baseline = terms(b["pg"], b, self.frozen)
        if any(self.baseline[k] <= 0 for k in ("local", "aligned", "chiral")):
            raise ValueError("baseline normalization must be positive; no silent fallback")
        self.last = None

    def point(self, x):
        if self.last is None or not np.array_equal(x, self.last.x):
            self.last = Point(x, self)
        return self.last

    def fun(self, x):
        p = self.point(x)
        return float(p.objective.detach()), p.gf.detach().numpy().copy()

    def hess(self, x):
        p = self.point(x)
        return p.hessian(p.gf)

    def cfun(self, x):
        return self.point(x).constraints.detach().numpy().copy()

    def cjac(self, x):
        return torch.stack(self.point(x).gc).detach().numpy().copy()

    def chess(self, x, multipliers):
        p = self.point(x)
        return p.hessian(sum(float(m) * g for m, g in zip(multipliers, p.gc, strict=True)))


def stationarity(tr, b, baseline, frozen, multipliers, cfg):
    corrections = torch.stack([s["delta"].detach() for s in tr["steps"]]).requires_grad_()
    direct = correction_trajectory(b["pg"], b["mask"], corrections)
    values = terms(direct["prediction"], b, frozen)
    f = values["local"] / baseline["local"]
    c = normalized_constraints(values, baseline)
    gf = torch.autograd.grad(f, corrections, retain_graph=True)[0]
    lag = f + sum(float(mu) * ci for mu, ci in zip(multipliers, c, strict=True))
    gl = torch.autograd.grad(lag, corrections)[0]
    valid = direct["eligible"]
    info = kkt(corrections.detach(), gl, valid, cfg["convergence"]["boundary_relative_tolerance"])
    scale = float(gl[valid].norm(dim=-1).max())
    trial = project_ball(corrections.detach() - (0.04 / max(scale, 1e-300)) * gl)
    mapping = float(((corrections.detach() - trial).norm(dim=-1) * valid).max()) / 0.04 if scale > 1e-12 else 0.0
    residual = float(info["residual_norm"].max()) / max(scale, 1e-300)
    return dict(
        local_gradient_norm=float(gf.norm()),
        lagrangian_correction_gradient_norm=float(gl.norm()),
        normalized_ball_kkt_max=residual,
        projected_ball_mapping_max=mapping,
        maximum_tangential_residual=float(info["tangent_residual"].max()),
        maximum_feasible_radial_residual=float(info["feasible_radial_residual"].max()),
        boundary_active_fraction=float(info["active"][valid].double().mean()),
    )


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
def metric_row(prediction, b, record, tr=None, t=0):
    frozen = freeze_chirality(b["pg"], b["target"], b["mask"])
    values = terms(prediction, b, frozen)
    mask = b["mask"]
    a, inv = signed_status(prediction, b["target"], mask)
    ba, bi = signed_status(b["pg"], b["target"], mask)
    rep = local_representation(prediction, mask)
    delta = (prediction - b["pg"])[mask].norm(dim=-1)
    p = prediction[mask]
    pc = p - p.mean(0)
    step = tr["steps"][t - 1]["delta"] if tr is not None and t else torch.zeros_like(prediction)
    path = (
        sum(s["delta"].norm(dim=-1) for s in tr["steps"][:t])
        if tr is not None and t
        else torch.zeros_like(mask, dtype=torch.float64)
    )
    mses = {str(k): float(values[f"local_{k}"]) for k in (1, 2, 3)}
    return dict(
        record,
        local_mse=mses,
        local_rmse={k: math.sqrt(v) for k, v in mses.items()},
        mean_local_rmse=sum(math.sqrt(v) for v in mses.values()) / 3,
        raw_cartesian=float(values["cartesian"]),
        aligned_rmsd=float(values["aligned"]),
        continuous_chiral_loss=float(values["chiral"]),
        chiral_error_sum=float(chiral_sum(prediction, mask, frozen)),
        chiral_loss_eligible=int(frozen["eligible"].sum()),
        chirality_inversions=int(inv.sum()),
        chirality_assessable=int(a.sum()),
        chirality_assessability_lost=int((ba & ~a).sum()),
        chirality_assessability_gained=int((a & ~ba).sum()),
        chirality_common_assessable=int((a & ba).sum()),
        chirality_common_inversions=int((inv & a & ba).sum()),
        chirality_common_baseline_inversions=int((bi & a & ba).sum()),
        frame_eligible=int(rep["eligible"].sum()),
        frame_degenerate=int(rep["degenerate"].sum()),
        frame_interior=int(rep["interior"].sum()),
        frame_assessability_preserved=bool(
            torch.equal(rep["eligible"], local_representation(b["pg"], mask)["eligible"])
        ),
        displacement_rms=float(delta.square().mean().sqrt()),
        displacement_max=float(delta.max()),
        step_correction_rms=float(step[mask].norm(dim=-1).square().mean().sqrt()),
        step_correction_max=float(step[mask].norm(dim=-1).max()),
        path_length_rms=float(path[mask].square().mean().sqrt()),
        path_length_max=float(path[mask].max()),
        finite=bool(torch.isfinite(prediction[mask]).all()),
        radius_gyration=float(pc.norm(dim=-1).square().mean().sqrt()),
        collapse=bool(pc.norm(dim=-1).square().mean().sqrt() < 1e-3),
    )
