"""V9 geometry and solver with the committed V9B physical convergence certificate."""

import math
import time

import numpy as np
import torch
from scipy.optimize import NonlinearConstraint

from . import e010_conditioning_v9b as audit
from . import e010_no_new_inversion_v9 as v9


def threshold(config, count):
    return config["material_normalized_local_decrease"] / (
        math.sqrt(count) * max(config["shadow_steps_angstrom"]) / config["s_max_angstrom"]
    )


def physical_screen(b, delta, mu, config):
    d = delta.detach().clone().requires_grad_()
    f, c, tr, o = audit.physical(b, d)
    g = torch.autograd.grad(f, d, retain_graph=True)[0].detach().numpy().reshape(-1)
    lag = torch.autograd.grad(f + (c * torch.from_numpy(np.asarray(mu))).sum(), d)[0].detach().numpy().reshape(-1)
    cv = c.detach().numpy()
    eligible = tr["eligible"].numpy()
    # A single exact VJP screens candidates without materializing the full Jacobian.
    info = audit.components(
        d.detach().numpy(), eligible, g, (lag - g)[None], np.zeros(1), np.ones(1), config["boundary_relative_tolerance"]
    )
    info["normalized_complementarity"] = max(info["normalized_complementarity"], float(np.abs(mu * cv).max()))
    info["normalized_dual"] = max(info["normalized_dual"], max(0.0, float(-np.min(mu))))
    telemetry = o.quartets.telemetry(tr["prediction"], config["active_normalized_slack"])
    count = int(eligible.sum())
    limit = threshold(config, count)
    feasible = bool(
        max(cv[:2]) <= 1e-8
        and (cv[2:] <= 0).all()
        and telemetry["new_inversions"] == 0
        and telemetry["assessability_lost"] == 0
        and np.linalg.norm(d.detach().numpy(), axis=-1).max() < 0.04
    )
    numerical = (
        info["physical_normalized_stationarity_l2"] <= limit
        and info["normalized_complementarity"] <= 1e-6
        and info["normalized_dual"] <= 1e-8
    )
    return dict(
        candidate=bool(feasible and numerical),
        feasible=feasible,
        stationarity_threshold=limit,
        eligible_vectors=count,
        physical_normalized_stationarity_l2=info["physical_normalized_stationarity_l2"],
        physical_normalized_stationarity_max=info["physical_normalized_stationarity_max"],
        normalized_complementarity=info["normalized_complementarity"],
        normalized_dual=info["normalized_dual"],
    )


def extra_directions(b, delta, g, j, mu, ids, directions, config):
    records = []
    for name, direction in directions.items():
        norm = np.linalg.norm(direction)
        if norm == 0:
            records.append(dict(direction=name, exact_zero=True, passed=True, uncertainty_normalized=0.0))
            continue
        direction = direction / norm
        expected = np.r_[g @ direction, j[ids] @ direction, (g + j.T @ mu) @ direction]
        trials = []
        for eps in config["finite_difference_eps_angstrom"]:
            fp, cp, *_ = audit.physical(b, delta + torch.from_numpy((eps * direction).reshape(delta.shape)))
            fm, cm, *_ = audit.physical(b, delta - torch.from_numpy((eps * direction).reshape(delta.shape)))
            fp, cp, fm, cm = [x.detach().numpy() for x in (fp, cp, fm, cm)]
            fd = np.r_[(fp - fm) / (2 * eps), (cp[ids] - cm[ids]) / (2 * eps), ((fp - fm) + mu @ (cp - cm)) / (2 * eps)]
            error = np.abs(fd - expected)
            trials.append(
                dict(
                    epsilon_angstrom=eps,
                    maximum_absolute_error=float(error.max()),
                    stationarity_absolute_error=float(error[[0, -1]].max()),
                    passed=bool(
                        (
                            error
                            <= config["finite_difference_absolute_tolerance"]
                            + config["finite_difference_relative_tolerance"] * np.abs(expected)
                        ).all()
                    ),
                )
            )
        records.append(
            dict(
                direction=name,
                exact_zero=False,
                trials=trials,
                passed=any(t["passed"] for t in trials),
                uncertainty_normalized=0.04 * min(t["stationarity_absolute_error"] for t in trials),
            )
        )
    return records


def certificate(b, delta, mu, config, contract):
    d, f, c, tr, o, g, j = audit.physical_jacobians(b, delta)
    cv = c.detach().numpy()
    eligible = tr["eligible"].numpy()
    info = audit.components(delta.numpy(), eligible, g, j, cv, mu, config["boundary_relative_tolerance"])
    a, ids, balls = audit.active_system(
        delta.numpy(), eligible, cv, j, config["active_normalized_slack"], config["boundary_relative_tolerance"]
    )
    direction, multipliers = audit.cone_projection(g * 0.04, a)
    count = int(eligible.sum())
    limit = threshold(config, count)
    checks = audit.finite_checks(
        b,
        delta,
        g,
        j,
        mu,
        ids,
        config["finite_difference_eps_angstrom"],
        config["finite_difference_directions"],
        config["finite_difference_absolute_tolerance"],
        config["finite_difference_relative_tolerance"],
    )
    directions = {
        f"fixed_phase_{phase}": np.sin(np.arange(delta.numel()) + 1 + phase)
        for phase in config["finite_difference_directions"]
    }
    directions.update(projected_feasible_gradient=direction, physical_lagrangian_gradient=g + j.T @ mu)
    additional = extra_directions(b, delta, g, j, mu, ids, directions, config)
    uncertainty = max(r["uncertainty_normalized"] for r in additional)
    derivative_pass = (
        all(any(r["passed"] for r in checks if r["phase"] == phase) for phase in config["finite_difference_directions"])
        and all(r["passed"] for r in additional)
        and uncertainty <= limit / 10
    )
    shadow_direction = direction.reshape(delta.shape).copy()
    size = np.linalg.norm(shadow_direction, axis=-1).max()
    if size:
        shadow_direction /= size
    shadows = []
    for step in config["shadow_steps_angstrom"]:
        candidate = audit.project_balls(delta.numpy() + step * shadow_direction)
        ff, cc, tt, oo = audit.physical(b, torch.from_numpy(candidate))
        tele = oo.quartets.telemetry(tt["prediction"], config["active_normalized_slack"])
        vals = cc.detach().numpy()
        feasible = bool(
            max(vals[:2]) <= 1e-8
            and (vals[2:] <= 0).all()
            and not tele["new_inversions"]
            and not tele["assessability_lost"]
            and np.linalg.norm(candidate, axis=-1).max() < 0.04
        )
        decrease = float(f.detach() - ff.detach())
        shadows.append(
            dict(
                step_angstrom=step,
                normalized_loss_decrease=decrease,
                feasible=feasible,
                material=feasible and decrease >= config["material_normalized_local_decrease"],
            )
        )
    telemetry = o.quartets.telemetry(tr["prediction"], config["active_normalized_slack"])
    feasibility = bool(
        max(cv[:2]) <= 1e-8
        and (cv[2:] <= 0).all()
        and not telemetry["new_inversions"]
        and not telemetry["assessability_lost"]
        and np.linalg.norm(delta.numpy(), axis=-1).max() < 0.04
    )
    gates = dict(
        primal_feasibility=feasibility,
        stationarity=info["physical_normalized_stationarity_l2"] <= limit,
        complementarity=info["normalized_complementarity"] <= contract["normalized_complementarity_max"],
        dual_feasibility=info["normalized_dual"] <= contract["dual_negativity_max"],
        derivatives=bool(derivative_pass),
        no_material_shadow_descent=not any(s["material"] for s in shadows),
    )
    info.pop("residual")
    info["ball_multipliers"] = info["ball_multipliers"].tolist()
    return dict(
        converged=all(gates.values()),
        gates=gates,
        physical=info,
        stationarity_threshold=limit,
        eligible_vectors=count,
        derivative_uncertainty_normalized=uncertainty,
        directional_checks=checks,
        certificate_direction_checks=additional,
        shadow_steps=shadows,
        projected_feasible_gradient_norm=float(np.linalg.norm(direction)),
        maximum_linearized_cone_violation=float(np.max(a @ direction)),
        active_inequality_indices=ids.tolist(),
        active_balls=len(balls),
        quartets=telemetry,
        active_multiplier_reconstruction=multipliers.tolist(),
    )


def solve(b, cfg, config, contract):
    oracle = v9.Oracle(b)
    history = []
    screens = []
    accepted = {}
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
            p = oracle.point(x)
            delta = torch.stack([s["delta"].detach() for s in p.tr["steps"]])
            mu = np.asarray(state.v[0])
            screen = physical_screen(b, delta, mu, config)
            screens.append(dict(iteration=int(state.nit), **screen))
            if screen["candidate"]:
                cert = certificate(b, delta, mu, config, contract)
                if cert["converged"]:
                    accepted["certificate"] = cert
                    accepted["x"] = np.asarray(x).copy()
                    return True
        return False

    result = v9.v7.constrained_minimize(
        oracle.fun,
        np.zeros(math.prod(oracle.shape)),
        oracle.hess,
        [NonlinearConstraint(oracle.cfun, -np.inf, 0.0, jac=oracle.cjac, hess=oracle.chess)],
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
        history=history,
        physical_screens=screens,
        physical_certificate=cert,
        runtime_seconds=time.perf_counter() - started,
    )
    return point.tr, log, result.x.copy()
