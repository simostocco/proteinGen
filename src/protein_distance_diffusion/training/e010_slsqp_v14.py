"""SLSQP solver intervention; V11 geometry and V9B certificates are unchanged."""

import math
import resource
import time

import numpy as np
import torch
from scipy.optimize import minimize

from . import e010_conditioning_v9b as audit
from . import e010_direct_correction_v11 as direct

SETTINGS = dict(method="SLSQP", maxiter=2000, ftol=1e-12, disp=False)


def required_arrays(n, m):
    """Exact installed SciPy 1.18.1 allocation formula, meq=0 and mieq>0."""
    if n <= 0 or m <= 0:
        raise ValueError("Positive variable and inequality counts required")
    buffer = n * (n + 1) // 2 + 3 * m * n + 9 * m + 8 * n * n + 35 * n + 28
    return dict(
        variables=n,
        inequalities=m,
        buffer_elements=buffer,
        buffer_bytes=8 * buffer,
        dense_normals_bytes=8 * m * n,
        required_array_bytes=8 * (buffer + m * n),
        other_allocations_included=False,
    )


def ram_gate(required, available, policy):
    reserve = max(policy["minimum_reserve_bytes"], int(available * policy["reserve_fraction_available_ram"]))
    budget = max(0, available - reserve)
    return dict(
        available_ram_bytes=available,
        reserved_bytes=reserve,
        solver_array_budget_bytes=budget,
        resource_feasible=required["required_array_bytes"] <= budget,
        swap_used_for_budget=False,
    )


class Constraints:
    """Two vector inequalities >=0; scientific values/Jacobians exactly negated."""

    def __init__(self, oracle):
        self.oracle = oracle

    def science(self, x):
        return -self.oracle.cfun(x)

    def science_jac(self, x):
        return -self.oracle.cjac(x).toarray()

    def dictionaries(self):
        return [
            dict(type="ineq", fun=self.science, jac=self.science_jac),
            dict(type="ineq", fun=self.oracle.ball_values, jac=lambda x: self.oracle.ball_jacobian(x).toarray()),
        ]


def reconstruct(b, delta, config):
    """Frozen physical active-system/NNLS multiplier reconstruction, independently of SLSQP."""
    _, _, c, tr, _, g, j = audit.physical_jacobians(b, delta)
    cv = c.detach().numpy()
    a, ids, balls = audit.active_system(
        delta.numpy(),
        tr["eligible"].numpy(),
        cv,
        j,
        config["active_normalized_slack"],
        config["boundary_relative_tolerance"],
    )
    direction, multipliers = audit.cone_projection(g * direct.S, a)
    mu = np.zeros(len(cv), dtype=np.float64)
    mu[ids] = multipliers[: len(ids)]
    return mu, dict(
        active_scientific_indices=ids.tolist(),
        active_balls=len(balls),
        active_multipliers=multipliers.tolist(),
        projected_feasible_gradient_norm=float(np.linalg.norm(direction)),
        empty_active_set=len(a) == 0,
    )


def optional(result, name, transform):
    value = getattr(result, name, None)
    return None if value is None else transform(value)


def solve(b, config, limits, save_raw, settings=None):
    """Persist completed solver state BEFORE certificate/optional telemetry can fail."""
    settings = SETTINGS if settings is None else settings
    if settings != SETTINGS:
        raise ValueError("V14 SLSQP settings are frozen")
    oracle = direct.Oracle(b)
    constraints = Constraints(oracle)
    history = []
    iterations = 0
    start = time.perf_counter()

    def callback(x):
        nonlocal iterations
        iterations += 1
        if iterations == 1 or iterations % 10 == 0:
            history.append(dict(iteration=iterations, normalized_objective=float(oracle.fun(x)[0])))

    result = minimize(
        oracle.fun,
        np.zeros(math.prod(oracle.shape), dtype=np.float64),
        jac=True,
        method=settings["method"],
        constraints=constraints.dictionaries(),
        callback=callback,
        options={k: v for k, v in settings.items() if k != "method"},
    )
    elapsed = time.perf_counter() - start
    # The raw optimizer result is serialized without constructing new telemetry.
    save_raw(result, elapsed)
    point = oracle.point(result.x)
    tr = point.tr
    delta = torch.stack([s["delta"].detach() for s in tr["steps"]])
    mu, reconstruction = reconstruct(b, delta, config)
    cert = direct.certificate(b, delta, mu, config, limits)
    solver_mu = optional(result, "multipliers", lambda x: np.asarray(x).tolist())
    science_count = len(oracle.cfun(result.x))
    agreement = None
    if solver_mu is not None and len(solver_mu) == science_count + len(result.x) // 3:
        returned = np.asarray(solver_mu[:science_count])
        agreement = dict(
            scientific_l2_difference=float(np.linalg.norm(returned - mu)),
            scientific_max_absolute_difference=float(np.max(np.abs(returned - mu))),
            solver_scientific_multiplier_norm=float(np.linalg.norm(returned)),
            reconstructed_scientific_multiplier_norm=float(np.linalg.norm(mu)),
            convention="SLSQP positive >=0 multipliers correspond to historical c<=0 positive multipliers",
        )
    log = dict(
        converged=cert["converged"],
        iterations=int(result.nit),
        scipy_status=int(result.status),
        scipy_success=bool(result.success),
        message=str(result.message),
        function_evaluations=int(result.nfev),
        jacobian_evaluations=optional(result, "njev", int),
        multipliers=mu.tolist(),
        solver_reported_multipliers=solver_mu,
        multiplier_reconstruction=reconstruction,
        multiplier_agreement=agreement,
        constraints=oracle.cfun(result.x).tolist(),
        ball_constraint_values=oracle.ball_values(result.x).tolist(),
        physical_certificate=cert,
        history=history,
        runtime_seconds=elapsed,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        cpu_threads=1,
        hessian_vector_products_used=False,
        optimality=None,
        barrier_parameter=None,
        barrier_tolerance=None,
        trust_radius=None,
        cg_stop_cond=None,
        cg_iterations=None,
    )
    return tr, log, np.asarray(result.x).copy()
