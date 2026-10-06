"""Exact historical chirality gates as float64 endpoint inequalities; no training."""

import math
import time

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint
from scipy.sparse import block_diag, csr_matrix, hstack, vstack

from ..models.e010_hybrid_local import EPS
from . import e010_local_feasibility_v7 as v7
from . import e010_local_feasibility_v8 as v8
from .e010_phase4d_diagnostic import signed_status
from .e010_phase4d_objective_v2 import signed_triple
from .e010_precision_kkt_v6 import correction_trajectory, kkt, project_ball

THRESHOLD = float(np.nextafter(np.float64(EPS), np.float64(np.inf)))


def quantities(p):
    """Raw q and frame-assessability arithmetic used by the evaluator."""
    a, b, c = p[:, 1:-2] - p[:, :-3], p[:, 2:-1] - p[:, 1:-2], p[:, 3:] - p[:, 2:-1]
    an, bn, cn = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
    q = (torch.cross(a, b, dim=-1) * c).sum(-1) / (an * bn * cn).clamp_min(EPS**3)
    e1 = b / bn.clamp_min(EPS)[..., None]
    v = a - (a * e1).sum(-1, keepdim=True) * e1
    turn = v.norm(dim=-1) / an.clamp_min(EPS)
    bonds = (p[:, 1:] - p[:, :-1]).norm(dim=-1)
    return q, turn, bonds


class QuartetConstraints:
    def __init__(self, b):
        self.b = b
        self.assess, self.inverted = signed_status(b["pg"], b["target"], b["mask"])
        self.assess = self.assess[:, 1:-2]
        self.inverted = self.inverted[:, 1:-2]
        self.correct = self.assess & ~self.inverted
        self.sign = signed_triple(b["target"], b["mask"]).sign().detach()
        self.q0, self.turn0, self.bonds0 = (v.detach() for v in quantities(b["pg"]))
        # Fixed panel has all N-3 quartets assessable; reject any unsupported case.
        if not bool(self.assess.all()):
            raise ValueError("Panel quartet coverage changed; do not silently omit newly assessable quartets")
        self.counts = dict(
            assessable=int(self.assess.sum()), correct=int(self.correct.sum()), inverted=int(self.inverted.sum())
        )
        self.slices = {}
        pos = 0
        for name, count in [
            ("signed", self.counts["correct"]),
            ("unsigned_assessability", self.counts["inverted"]),
            ("frame_assessability", self.counts["assessable"]),
            ("bond_assessability", self.bonds0.numel()),
        ]:
            self.slices[name] = (pos, pos + count)
            pos += count

    def __call__(self, p):
        q, turn, bonds = quantities(p)
        return torch.cat(
            [
                (THRESHOLD - self.sign * q)[self.correct] / self.q0[self.correct].abs(),
                (THRESHOLD**2 - q.square())[self.inverted] / self.q0[self.inverted].square(),
                (THRESHOLD - turn)[self.assess] / self.turn0[self.assess],
                ((THRESHOLD - bonds) / self.bonds0).reshape(-1),
            ]
        )

    @torch.no_grad()
    def telemetry(self, p, tolerance):
        q, turn, bonds = quantities(p)
        assess, inv = signed_status(p, self.b["target"], self.b["mask"])
        assess = assess[:, 1:-2]
        inv = inv[:, 1:-2]
        c = self(p).detach()
        lo, hi = self.slices["signed"]
        signed = (self.sign * q)[self.correct]
        return dict(
            **self.counts,
            active_signed=int((c[lo:hi].abs() <= tolerance).sum()),
            minimum_signed_quantity=float(signed.min()) if len(signed) else None,
            minimum_signed_margin=float((signed - EPS).min()) if len(signed) else None,
            new_inversions=int((inv & self.correct).sum()),
            repaired_inversions=int((self.inverted & assess & ~inv).sum()),
            assessability_lost=int((self.assess & ~assess).sum()),
            minimum_bond_margin=float((bonds - EPS).min()),
            minimum_frame_margin=float((turn[self.assess] - EPS).min()),
            minimum_absolute_q_margin=float((q[self.assess].abs() - EPS).min()),
            exact_extra_feasible=bool((c <= 0).all()),
            constraint_slices=self.slices,
        )


class Point(v7.Point):
    def __init__(self, x, oracle):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.z = torch.from_numpy(self.x.copy()).requires_grad_()
        self.tr = v8.physical_trajectory(oracle.b["pg"], oracle.b["mask"], 0.04 * self.z.reshape(oracle.shape))
        self.values = v8.terms(self.tr["prediction"], oracle.b, oracle.frozen)
        self.objective = self.values["local"] / oracle.baseline["local"]
        self.constraints = torch.cat(
            [v8.normalized_constraints(self.values, oracle.baseline), oracle.quartets(self.tr["prediction"])]
        )
        self.gf = torch.autograd.grad(self.objective, self.z, create_graph=True, retain_graph=True)[0]


class Oracle(v8.Oracle):
    def __init__(self, b):
        super().__init__(b, "B")
        self.quartets = QuartetConstraints(self.b)

    def point(self, x):
        if self.last is None or not np.array_equal(x, self.last.x):
            self.last = Point(x, self)
        return self.last

    def cjac(self, x):
        p = self.point(x)
        first = csr_matrix(
            torch.stack([torch.autograd.grad(c, p.z, retain_graph=True)[0] for c in p.constraints[:2]]).detach().numpy()
        )
        # Exact coordinate Jacobian, followed by the exact block-local radial Jacobian.
        j = torch.func.jacrev(self.quartets)(p.tr["prediction"].detach()).reshape(len(p.constraints) - 2, -1).numpy()
        z = p.z.detach().reshape(-1, 3)

        def radial(t):
            v = 0.04 * t
            d = v / (1 + v.square().sum() / 0.04**2).sqrt()
            limit = torch.nextafter(d.new_tensor(0.04), d.new_tensor(0.0))
            size = d.square().sum().clamp_min(1e-24).sqrt()
            return d * torch.minimum(d.new_tensor(1.0), limit / size)

        blocks = torch.func.vmap(torch.func.jacrev(radial))(z).numpy()
        eligible = torch.stack([s["eligible"] for s in p.tr["steps"]]).reshape(8, -1).numpy()
        blocks = blocks.reshape(8, -1, 3, 3) * eligible[:, :, None, None]
        chain = hstack(
            [block_diag([csr_matrix(block) for block in step], format="csr") for step in blocks], format="csr"
        )
        answer = vstack([first, csr_matrix(j) @ chain], format="csr")
        if not np.isfinite(answer.data).all():
            raise RuntimeError("Nonfinite exact sparse constraint Jacobian")
        return answer

    def chess(self, x, multipliers):
        p = self.point(x)
        mu = torch.as_tensor(np.asarray(multipliers).copy(), dtype=torch.float64)
        g = torch.autograd.grad((p.constraints * mu).sum(), p.z, create_graph=True, retain_graph=True)[0]
        return p.hessian(g)


def stationarity(tr, oracle, multipliers, cfg):
    delta = torch.stack([s["delta"].detach() for s in tr["steps"]]).requires_grad_()
    direct = correction_trajectory(oracle.b["pg"], oracle.b["mask"], delta)
    values = v8.terms(direct["prediction"], oracle.b, oracle.frozen)
    f = values["local"] / oracle.baseline["local"]
    c = torch.cat([v8.normalized_constraints(values, oracle.baseline), oracle.quartets(direct["prediction"])])
    gf = torch.autograd.grad(f, delta, retain_graph=True)[0]
    gl = torch.autograd.grad(f + (c * torch.tensor(multipliers, dtype=torch.float64)).sum(), delta)[0]
    valid = direct["eligible"]
    info = kkt(delta.detach(), gl, valid, cfg["convergence"]["boundary_relative_tolerance"])
    scale = float(gl[valid].norm(dim=-1).max())
    trial = project_ball(delta.detach() - (0.04 / max(scale, 1e-300)) * gl)
    return dict(
        local_gradient_norm=float(gf.norm()),
        lagrangian_correction_gradient_norm=float(gl.norm()),
        normalized_ball_kkt_max=float(info["residual_norm"].max()) / max(scale, 1e-300),
        projected_ball_mapping_max=float(((delta.detach() - trial).norm(dim=-1) * valid).max()) / 0.04
        if scale > 1e-12
        else 0.0,
        boundary_active_fraction=float(info["active"][valid].double().mean()),
    )


def solve(b, cfg):
    oracle = Oracle(b)
    history = []
    started = time.perf_counter()

    def callback(x, state):
        if state.nit == 1 or state.nit % cfg["history_interval"] == 0:
            history.append(
                dict(
                    iteration=int(state.nit),
                    normalized_objective=float(state.fun),
                    optimality=float(state.optimality),
                    constraint_violation=float(state.constr_violation),
                    trust_radius=float(state.tr_radius),
                    barrier_parameter=float(state.barrier_parameter),
                )
            )

    constraints = [NonlinearConstraint(oracle.cfun, -np.inf, 0.0, jac=oracle.cjac, hess=oracle.chess)]
    result = v7.constrained_minimize(
        oracle.fun, np.zeros(math.prod(oracle.shape)), oracle.hess, constraints, cfg["solver"], callback
    )
    p = oracle.point(result.x)
    mu = result.v[0].tolist()
    c = oracle.cfun(result.x)
    telemetry = oracle.quartets.telemetry(p.tr["prediction"], cfg["constraints"]["active_tolerance"])
    feasible = bool(
        max(c[:2]) <= cfg["constraints"]["normalized_feasibility_tolerance"]
        and telemetry["exact_extra_feasible"]
        and telemetry["new_inversions"] == 0
        and telemetry["assessability_lost"] == 0
    )
    residual = stationarity(p.tr, oracle, mu, cfg)
    comp = float(np.max(np.abs(c * np.asarray(mu))))
    converged = bool(
        result.success
        and result.optimality <= cfg["solver"]["gtol"]
        and feasible
        and residual["normalized_ball_kkt_max"] <= 0.001
        and residual["projected_ball_mapping_max"] <= 0.001
        and min(mu) >= -1e-8
        and comp <= 1e-6
    )
    log = dict(
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
        constraints=c.tolist(),
        constraint_feasible=feasible,
        multipliers=mu,
        complementarity=comp,
        stationarity=residual,
        quartets=telemetry,
        history=history,
        runtime_seconds=time.perf_counter() - started,
    )
    return p.tr, log, result.x.copy()


__all__ = ["Oracle", "QuartetConstraints", "THRESHOLD", "solve", "quantities"]
