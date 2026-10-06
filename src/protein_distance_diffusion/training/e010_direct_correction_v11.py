"""Direct linearly scaled Cartesian balls; unchanged V9 science/V9B physical KKT."""

import math
import time

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint
from scipy.sparse import csr_matrix, diags, hstack, vstack

from . import e010_no_new_inversion_v9 as v9
from . import e010_strict_scientific_v10 as v10
from .e010_precision_kkt_v6 import correction_trajectory

S = 0.04
K = 8


def trajectory(pg, mask, z):
    v9.v7.require64(pg, z)
    if z.shape != (K, *pg.shape):
        raise ValueError("V11 requires exactly eight Cartesian correction steps")
    direct = correction_trajectory(pg, mask, S * z)
    direct["steps"] = [
        dict(delta=S * zi * valid[..., None], eligible=valid) for zi, valid in zip(z, direct["eligible"], strict=True)
    ]
    return direct


class Point(v9.v7.Point):
    def __init__(self, x, oracle):
        self.x = np.asarray(x, dtype=np.float64).copy()
        self.z = torch.from_numpy(self.x.copy()).requires_grad_()
        self.tr = trajectory(oracle.b["pg"], oracle.b["mask"], self.z.reshape(oracle.shape))
        self.values = v9.v8.terms(self.tr["prediction"], oracle.b, oracle.frozen)
        self.objective = self.values["local"] / oracle.baseline["local"]
        self.constraints = torch.cat(
            [v9.v8.normalized_constraints(self.values, oracle.baseline), oracle.quartets(self.tr["prediction"])]
        )
        self.gf = torch.autograd.grad(self.objective, self.z, create_graph=True, retain_graph=True)[0]


class Oracle(v9.v7.Oracle):
    def __init__(self, b):
        super().__init__(b, "B")
        self.shape = (K, *b["pg"].shape)
        self.quartets = v9.QuartetConstraints(self.b)

    def point(self, x):
        if self.last is None or not np.array_equal(x, self.last.x):
            self.last = Point(x, self)
        return self.last

    def cjac(self, x):
        p = self.point(x)
        first = csr_matrix(
            torch.stack([torch.autograd.grad(c, p.z, retain_graph=True)[0] for c in p.constraints[:2]]).detach().numpy()
        )
        j = torch.func.jacrev(self.quartets)(p.tr["prediction"].detach()).reshape(len(p.constraints) - 2, -1).numpy()
        chain = hstack(
            [diags(np.repeat(step.numpy().reshape(-1) * S, 3), format="csr") for step in p.tr["eligible"]], format="csr"
        )
        answer = vstack([first, csr_matrix(j) @ chain], format="csr")
        if not np.isfinite(answer.data).all():
            raise RuntimeError("Nonfinite direct-space scientific constraint Jacobian")
        return answer

    def chess(self, x, multipliers):
        p = self.point(x)
        mu = torch.as_tensor(np.asarray(multipliers).copy(), dtype=torch.float64)
        g = torch.autograd.grad((p.constraints * mu).sum(), p.z, create_graph=True, retain_graph=True)[0]
        return p.hessian(g)

    @staticmethod
    def ball_values(x):
        return 1 - (np.asarray(x).reshape(-1, 3) ** 2).sum(-1)

    @staticmethod
    def ball_jacobian(x):
        x = np.asarray(x)
        return csr_matrix(
            (-2 * x, (np.repeat(np.arange(len(x) // 3), 3), np.arange(len(x)))), shape=(len(x) // 3, len(x))
        )

    @staticmethod
    def ball_hessian(x, multipliers):
        return diags(np.repeat(-2 * np.asarray(multipliers), 3), format="csr")


def closed_ball_primal(cert, delta):
    """Only the explicitly authorized interior-to-closed-ball boundary bookkeeping."""
    cv = cert["quartets"]
    return bool(
        cert["physical"]["normalized_primal"] <= 1e-8
        and cv["exact_extra_feasible"]
        and not cv["new_inversions"]
        and not cv["assessability_lost"]
        and np.linalg.norm(delta.numpy(), axis=-1).max() <= S
    )


def certificate(b, delta, mu, config, contract):
    cert = v10.certificate(b, delta, mu, config, contract)
    # All interior certificates are bitwise identical. Exact boundary points only
    # replace the old representational strict-radius check with the intended <=.
    if np.linalg.norm(delta.numpy(), axis=-1).max() == S:
        cert["gates"]["primal_feasibility"] = closed_ball_primal(cert, delta)
        cert["converged"] = all(cert["gates"].values())
    return cert


def solve(b, cfg, config, contract):
    oracle = Oracle(b)
    history, screens, accepted = [], [], {}
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
            point = oracle.point(x)
            delta = torch.stack([s["delta"].detach() for s in point.tr["steps"]])
            mu = np.asarray(state.v[0])
            screen = v10.physical_screen(b, delta, mu, config)
            screens.append(dict(iteration=int(state.nit), **screen))
            if screen["candidate"]:
                cert = certificate(b, delta, mu, config, contract)
                if cert["converged"]:
                    accepted.update(certificate=cert, x=np.asarray(x).copy())
                    return True
        return False

    result = v9.v7.constrained_minimize(
        oracle.fun,
        np.zeros(math.prod(oracle.shape)),
        oracle.hess,
        [
            NonlinearConstraint(oracle.cfun, -np.inf, 0.0, jac=oracle.cjac, hess=oracle.chess),
            NonlinearConstraint(oracle.ball_values, 0.0, np.inf, jac=oracle.ball_jacobian, hess=oracle.ball_hessian),
        ],
        cfg["solver"],
        callback,
    )
    point = oracle.point(result.x)
    delta = torch.stack([s["delta"].detach() for s in point.tr["steps"]])
    mu = np.asarray(result.v[0])
    cert = (
        accepted.get("certificate")
        if "x" in accepted and np.array_equal(result.x, accepted["x"])
        else certificate(b, delta, mu, config, contract)
    )
    ball_values = oracle.ball_values(result.x)
    ball_mu = -np.asarray(result.v[1])  # lower-bound convention -> nonnegative c<=0 multiplier
    log = dict(
        converged=cert["converged"],
        iterations=int(result.nit),
        scipy_status=int(result.status),
        scipy_success=bool(result.success),
        message=str(result.message),
        optimality=float(result.optimality),
        constraint_violation=float(result.constr_violation),
        barrier_parameter=float(result.barrier_parameter),
        barrier_tolerance=float(result.barrier_tolerance),
        trust_radius=float(result.tr_radius),
        cg_stop_cond=int(result.cg_stop_cond),
        function_evaluations=int(result.nfev),
        cg_iterations=int(result.cg_niter),
        multipliers=mu.tolist(),
        constraints=oracle.cfun(result.x).tolist(),
        ball_constraint_values=ball_values.tolist(),
        ball_constraint_multipliers=ball_mu.tolist(),
        solver_ball_complementarity=float(np.max(np.abs(ball_values * ball_mu))),
        solver_ball_dual_negativity=max(0.0, float(-ball_mu.min())),
        history=history,
        physical_screens=screens,
        physical_certificate=cert,
        runtime_seconds=time.perf_counter() - started,
    )
    return point.tr, log, result.x.copy()
