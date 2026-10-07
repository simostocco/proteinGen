"""Sparse feasible SQP; direct V11 geometry and frozen V9B certification."""

import math
import resource
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, vstack

from . import e010_direct_correction_v11 as direct
from . import e010_slsqp_v14 as historical_certificate

SETTINGS = dict(
    outer_maxiter=2000,
    qp_maxiter=500,
    qp_gtol=1e-11,
    qp_ftol=1e-15,
    qp_primal_tolerance=1e-8,
    qp_projected_gradient_tolerance=1e-8,
    curvature_initial=0.01,
    curvature_min=1e-6,
    curvature_max=1000.0,
    active_science_slack=1e-5,
    active_ball_relative=1e-4,
    science_slack_fraction=0.5,
    inward_linearization_margin=1e-12,
    armijo=1e-4,
    backtrack_factor=0.5,
    max_backtracks=50,
    maximum_direction_vector_norm=1.0,
    physical_check_interval=20,
)


def project_z(x):
    """Leave feasible vectors bitwise unchanged; first feasible uniform inward ULP guard."""
    x = np.asarray(x)
    if x.dtype != np.float64 or not np.isfinite(x).all():
        raise ValueError("Projection requires finite float64 variables")
    original = x.reshape(-1, 3)
    projected = original.copy()
    norms = np.linalg.norm(direct.S * original, axis=1)
    znorms = np.linalg.norm(original, axis=1)
    outside = (norms > direct.S) | (np.sum(original**2, axis=1) > 1)
    scales = np.minimum(direct.S / np.maximum(norms, 1e-300), 1 / np.maximum(znorms, 1e-300))
    projected[outside] *= scales[outside, None]
    rounded = projected.copy()
    beta = np.ones(len(projected))
    guard_steps = np.zeros(len(projected), dtype=int)
    for _ in range(128):
        bad = (np.linalg.norm(direct.S * projected, axis=1) > direct.S) | (np.sum(projected**2, axis=1) > 1)
        if not bad.any():
            break
        beta[bad] = np.nextafter(beta[bad], 0.0)
        guard_steps[bad] += 1
        projected[bad] = rounded[bad] * beta[bad, None]
    else:
        raise RuntimeError("Minimal representable inward projection failed")
    return projected.reshape(x.shape), dict(
        projected_vectors=int(outside.sum()),
        guarded_vectors=int((guard_steps > 0).sum()),
        maximum_inward_guard_angstrom=float(np.linalg.norm(direct.S * (projected - rounded), axis=1).max()),
        maximum_guard_factor_ulps=int(guard_steps.max()),
        maximum_physical_projection_angstrom=float(np.linalg.norm(direct.S * (projected - original), axis=1).max()),
    )


def sparse_qp(g, a, rhs, curvature, settings=SETTINGS):
    """Convex diagonal primal QP via sparse matrix-free nonnegative dual; no Gram matrix."""
    if not isinstance(a, csr_matrix):
        raise TypeError("Local QP must receive CSR constraint Jacobians")
    g, rhs = np.asarray(g, float), np.asarray(rhs, float)
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("Positive definite curvature is mandatory")
    norms = np.sqrt(a.multiply(a).sum(axis=1)).A1
    zero = norms == 0
    if (rhs[zero] < -settings["qp_primal_tolerance"]).any():
        return np.zeros_like(g), np.zeros(len(rhs)), dict(success=False, reason="inconsistent_zero_row", iterations=0)
    keep = ~zero
    scaled = a[keep].multiply(1 / norms[keep, None]).tocsr()
    r = rhs[keep] / norms[keep]
    if not len(r):
        return (
            -g / curvature,
            np.zeros(len(rhs)),
            dict(
                success=True,
                reason="empty_active_set",
                iterations=0,
                primal_violation=0.0,
                projected_gradient=0.0,
            ),
        )

    def dual(mu):
        lag = g + scaled.T @ mu
        return float(0.5 * (lag @ lag) / curvature + r @ mu), np.asarray(scaled @ lag / curvature + r)

    result = minimize(
        dual,
        np.zeros(len(r)),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(r),
        options=dict(
            maxiter=settings["qp_maxiter"], gtol=settings["qp_gtol"], ftol=settings["qp_ftol"], maxcor=10, maxls=50
        ),
    )
    mu = np.zeros(len(rhs))
    mu[keep] = result.x / norms[keep]
    step = -(g + a.T @ mu) / curvature
    _, dual_grad = dual(result.x)
    residual = result.x - np.maximum(result.x - dual_grad, 0)
    violation = max(0.0, float(np.max(a @ step - rhs)))
    residual_max = float(np.max(np.abs(residual)))
    valid = np.isfinite(step).all() and violation <= settings["qp_primal_tolerance"]
    valid = valid and residual_max <= settings["qp_projected_gradient_tolerance"]
    return (
        step,
        mu,
        dict(
            success=bool(valid),
            scipy_success=bool(result.success),
            reason=str(result.message),
            iterations=int(result.nit),
            evaluations=int(result.nfev),
            primal_violation=violation,
            projected_gradient=residual_max,
            constraints=len(rhs),
            variables=len(g),
            jacobian_nnz=int(a.nnz),
            curvature_nnz=len(g),
            dense_quadratic_arrays=0,
            estimated_sparse_working_bytes=int(
                3 * (a.data.nbytes + a.indices.nbytes + a.indptr.nbytes) + 8 * (5 * len(g) + 35 * len(rhs))
            ),
        ),
    )


def active_set(oracle, x, c, j, settings=SETTINGS):
    ids = np.flatnonzero(c >= -settings["active_science_slack"])
    vectors = x.reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)
    eligible = np.asarray(oracle.point(x).tr["eligible"]).reshape(-1)
    balls = np.flatnonzero(eligible & (norms >= 1 - settings["active_ball_relative"]))
    rows = np.repeat(np.arange(len(balls)), 3)
    cols = (3 * balls[:, None] + np.arange(3)).reshape(-1)
    values = (vectors[balls] / np.maximum(norms[balls, None], 1e-300)).reshape(-1)
    bj = csr_matrix((values, (rows, cols)), shape=(len(balls), len(x)))
    a = vstack([j[ids], bj], format="csr")
    rhs = np.r_[
        -settings["science_slack_fraction"] * c[ids] - settings["inward_linearization_margin"],
        1 - norms[balls],
    ]
    families = dict(aligned=int(0 in ids), continuous_chirality=int(1 in ids), correction_balls=len(balls))
    for name, (lo, hi) in oracle.quartets.slices.items():
        families[name] = int(((ids >= lo + 2) & (ids < hi + 2)).sum())
    return a, rhs, ids, balls, families


def feasible(oracle, x):
    c = oracle.cfun(x)
    tr = oracle.point(x).tr
    delta = torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy()
    return bool(
        np.isfinite(c).all()
        and np.isfinite(x).all()
        and max(c[:2]) <= 1e-8
        and (c[2:] <= 0).all()
        and np.linalg.norm(delta, axis=-1).max() <= direct.S
        and (np.sum(x.reshape(-1, 3) ** 2, axis=1) <= 1).all()
    )


def line_search(oracle, x, f, g, direction, settings=SETTINGS):
    alpha = 1.0
    largest_guard = 0.0
    for count in range(settings["max_backtracks"] + 1):
        candidate, guard = project_z(x + alpha * direction)
        largest_guard = max(largest_guard, guard["maximum_inward_guard_angstrom"])
        displacement = candidate - x
        derivative = float(g @ displacement)
        if derivative < 0 and feasible(oracle, candidate):
            value = float(oracle.fun(candidate)[0])
            if value < f and value <= f + settings["armijo"] * derivative:
                return candidate, dict(
                    accepted=True,
                    alpha=alpha,
                    backtracks=count,
                    predicted_change=derivative,
                    objective_change=value - f,
                    maximum_inward_guard_angstrom=largest_guard,
                )
        alpha *= settings["backtrack_factor"]
    return None, dict(
        accepted=False, backtracks=settings["max_backtracks"], maximum_inward_guard_angstrom=largest_guard
    )


def certify(b, oracle, x, physical, limits):
    tr = oracle.point(x).tr
    delta = torch.stack([s["delta"].detach() for s in tr["steps"]])
    mu, reconstruction = historical_certificate.reconstruct(b, delta, physical)
    return direct.certificate(b, delta, mu, physical, limits), mu, reconstruction


def solve(b, physical, limits, save_raw, settings=SETTINGS, smoke_steps=None):
    if settings != SETTINGS:
        raise ValueError("V15 settings must be preregistered")
    oracle = direct.Oracle(b)
    x = np.zeros(math.prod(oracle.shape), dtype=np.float64)
    if not feasible(oracle, x):
        raise ValueError("Zero baseline is not exactly feasible")
    history, qps, screens = [], [], []
    curvature = settings["curvature_initial"]
    accepted = backtracks = failures = 0
    largest_guard = 0.0
    start = time.perf_counter()
    reason = "outer_iteration_cap"
    previous_x = previous_g = None
    iteration = 0
    try:
        if smoke_steps not in (None, 25):
            raise ValueError("Only the preregistered 25-step resource smoke is permitted")
        for iteration in range(1, (smoke_steps or settings["outer_maxiter"]) + 1):
            f, g = oracle.fun(x)
            c, j = oracle.cfun(x), oracle.cjac(x)
            if previous_x is not None:
                s, y = x - previous_x, g - previous_g
                ss, sy = float(s @ s), float(s @ y)
                if ss > 0 and sy > 0:
                    curvature = float(np.clip(sy / ss, settings["curvature_min"], settings["curvature_max"]))
            a, rhs, ids, balls, families = active_set(oracle, x, c, j)
            d, multipliers, qp = sparse_qp(g, a, rhs, curvature)
            qp.update(outer_iteration=iteration, active_families=families, full_jacobian_nnz=int(j.nnz))
            qps.append(qp)
            mu = np.zeros(len(c))
            mu[ids] = multipliers[: len(ids)]
            if iteration == 1 or iteration % settings["physical_check_interval"] == 0 or not qp["success"]:
                delta = torch.stack([s["delta"].detach() for s in oracle.point(x).tr["steps"]])
                screen = direct.v10.physical_screen(b, delta, mu, physical)
                screens.append(dict(iteration=iteration, **screen))
                numerical = (
                    screen["physical_normalized_stationarity_l2"] <= screen["stationarity_threshold"]
                    and screen["normalized_complementarity"] <= limits["normalized_complementarity_max"]
                    and screen["normalized_dual"] <= limits["dual_negativity_max"]
                )
                if numerical and feasible(oracle, x):
                    cert, _, _ = certify(b, oracle, x, physical, limits)
                    if cert["converged"]:
                        reason = "physical_contract_pass"
                        break
            if not qp["success"]:
                failures += 1
                reason = "local_QP_failure"
                break
            size = np.linalg.norm(d.reshape(-1, 3), axis=1).max()
            if size > settings["maximum_direction_vector_norm"]:
                d *= settings["maximum_direction_vector_norm"] / size
            candidate, trial = line_search(oracle, x, float(f), g, d)
            backtracks += trial["backtracks"]
            largest_guard = max(largest_guard, trial["maximum_inward_guard_angstrom"])
            if iteration == 1 or iteration % 10 == 0 or candidate is None:
                history.append(
                    dict(
                        iteration=iteration,
                        normalized_objective=float(f),
                        curvature=curvature,
                        active_families=families,
                        line_search=trial,
                    )
                )
            if candidate is None:
                reason = "line_search_failure"
                break
            previous_x, previous_g = x.copy(), g.copy()
            x = candidate
            accepted += 1
    except Exception as exc:
        reason = f"solver_exception:{type(exc).__name__}:{exc}"
    elapsed = time.perf_counter() - start
    # Persist last accepted feasible variables BEFORE optional telemetry/certificate work.
    save_raw(x.copy(), dict(iterations=iteration, reason=reason, accepted_iterates=accepted, runtime_seconds=elapsed))
    cert, mu, reconstruction = certify(b, oracle, x, physical, limits)
    tr = oracle.point(x).tr
    log = dict(
        converged=cert["converged"],
        iterations=iteration,
        termination=reason,
        accepted_iterates=accepted,
        all_accepted_iterates_feasible=True,
        total_QP_solves=len(qps),
        QP_failures=failures,
        line_search_backtracks=backtracks,
        maximum_inward_guard_angstrom=largest_guard,
        history=history,
        qp_history=qps,
        physical_screens=screens,
        physical_certificate=cert,
        multipliers=mu.tolist(),
        multiplier_reconstruction=reconstruction,
        constraints=oracle.cfun(x).tolist(),
        ball_constraint_values=oracle.ball_values(x).tolist(),
        runtime_seconds=elapsed,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        variable_dimension=len(x),
        scientific_constraint_dimension=len(oracle.cfun(x)),
        total_constraint_dimension=len(oracle.cfun(x)) + len(x) // 3,
        Jacobian_nnz=int(oracle.cjac(x).nnz),
        QP_curvature_nnz=len(x),
        dense_quadratic_arrays=0,
        backend="installed_SciPy_L-BFGS-B_nonnegative_dual_sparse_products",
        cpu_threads=1,
        neural_training_launched=False,
        cuda_used=False,
    )
    return tr, log, x.copy()
