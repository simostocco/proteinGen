"""Fixed endpoint diagnostics only. NNLS tangent-cone QP is the sole solve."""

import contextlib
from unittest.mock import patch

import numpy as np
import torch

from . import e010_conditioning_v9b as physical
from . import e010_direct_correction_v11 as direct


@contextlib.contextmanager
def forbid_nonlinear_optimization():
    def forbidden(*args, **kwargs):
        raise AssertionError("V12 forbids nonlinear optimization and model training")

    with contextlib.ExitStack() as stack:
        for name in [
            "scipy.optimize.minimize",
            "protein_distance_diffusion.training.e010_local_feasibility_v7.constrained_minimize",
            "protein_distance_diffusion.training.e010_direct_correction_v11.solve",
            "protein_distance_diffusion.training.e010_no_new_inversion_v9.solve",
            "protein_distance_diffusion.training.e010_strict_scientific_v10.solve",
            "torch.optim.LBFGS.step",
            "torch.optim.Adam.step",
            "torch.optim.SGD.step",
        ]:
            stack.enter_context(patch(name, forbidden))
        yield


def distribution(values):
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(a):
        return dict(count=0, minimum=None, median=None, maximum=None, mean=None, quantiles=[])
    return dict(
        count=len(a),
        minimum=float(a.min()),
        median=float(np.median(a)),
        maximum=float(a.max()),
        mean=float(a.mean()),
        quantiles=np.quantile(a, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist(),
    )


def split_gradient(delta, gradient, eligible):
    d = np.asarray(delta).reshape(-1, 3)
    g = np.asarray(gradient).reshape(-1, 3)
    valid = np.asarray(eligible).reshape(-1)
    radius = np.linalg.norm(d, axis=-1)
    unit = d / np.maximum(radius[:, None], 1e-300)
    radial = np.sum(g * unit, axis=-1)
    tangent = g - radial[:, None] * unit
    outward = np.maximum(-radial, 0)  # negative gradient points outward
    inward = np.maximum(radial, 0)
    return dict(unit=unit, radius=radius, radial=radial, tangent=tangent, outward=outward, inward=inward, valid=valid)


def decomposition_summary(parts, subset):
    subset = np.asarray(subset).reshape(-1) & parts["valid"]
    out, inc, tan = (parts[k][subset] for k in ["outward", "inward", "tangent"])
    norms = dict(
        outward_descent_radial_l2=float(np.linalg.norm(out)),
        inward_descent_radial_l2=float(np.linalg.norm(inc)),
        tangential_l2=float(np.linalg.norm(tan)),
    )
    return dict(
        vectors=int(subset.sum()),
        raw_per_angstrom=norms,
        frozen_normalized={k: 0.04 * v for k, v in norms.items()},
        outward_distribution=distribution(0.04 * out),
        inward_distribution=distribution(0.04 * inc),
        tangential_distribution=distribution(0.04 * np.linalg.norm(tan, axis=-1)),
    )


def jacobian_geometry(a):
    info = physical.spectrum(a)
    unit = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-300)
    companion = physical.spectrum(unit)
    return dict(
        frozen_normalized=info,
        physical_singular_values_per_angstrom=[v / 0.04 for v in info["singular_values"]],
        row_unit_diagnostic=companion,
        row_scaling_is_analysis_only=True,
    )


def tangent_diagnostic(g, a, shape, config):
    projected, multipliers = physical.cone_projection(0.04 * np.asarray(g), a)
    blocks = projected.reshape(shape)
    size = np.linalg.norm(blocks, axis=-1).max()
    direction = blocks / size if size else np.zeros_like(blocks)
    cone = a @ projected
    return direction, dict(
        method="frozen_V9B_nonnegative_least_squares_dual_of_Euclidean_cone_projection",
        problem="min_h 0.5*||h+0.04*g||^2 subject to A*h<=0; scale candidate to frozen block radius",
        trust_radius_angstrom=max(config["shadow_steps_angstrom"]),
        approximate_block_radius_linear_minimum=True,
        projected_feasible_gradient_norm=float(np.linalg.norm(projected)),
        maximum_linearized_violation=float(cone.max()) if len(cone) else 0.0,
        dual_negativity=max(0.0, float(-multipliers.min())) if len(multipliers) else 0.0,
        complementarity=float(np.max(np.abs(multipliers * cone))) if len(cone) else 0.0,
        stationarity=float(np.linalg.norm(projected + 0.04 * np.asarray(g) + a.T @ multipliers)),
        active_nonnegative_multipliers=multipliers.tolist(),
        predicted_normalized_decrease_by_scale={
            str(eps): float(-eps * np.asarray(g) @ direction.reshape(-1)) for eps in config["shadow_steps_angstrom"]
        },
    )


def shadows(b, delta, direction, objective, baseline_local, config):
    results = []
    for eps in config["shadow_steps_angstrom"]:
        candidate = physical.project_balls(delta.numpy() + eps * direction)
        f, c, tr, oracle = physical.physical(b, torch.from_numpy(candidate))
        vals = c.detach().numpy()
        q = oracle.quartets.telemetry(tr["prediction"], config["active_normalized_slack"])
        feasible = bool(
            max(vals[:2]) <= 1e-8
            and (vals[2:] <= 0).all()
            and not q["new_inversions"]
            and not q["assessability_lost"]
            and np.linalg.norm(candidate, axis=-1).max() < 0.04
        )
        decrease = float(objective.detach() - f.detach())
        results.append(
            dict(
                step_angstrom=eps,
                normalized_loss_decrease=decrease,
                feasible=feasible,
                material=feasible and decrease >= config["material_normalized_local_decrease"],
                raw_local_mse_decrease=decrease * baseline_local,
                aligned_constraint=float(vals[0]),
                chiral_constraint=float(vals[1]),
                maximum_strict_residual=float(vals[2:].max()),
                new_inversions=q["new_inversions"],
                assessability_lost=q["assessability_lost"],
                realized_maximum_perturbation_angstrom=float(np.linalg.norm(candidate - delta.numpy(), axis=-1).max()),
            )
        )
    return results


def history_windows(log, windows):
    history = {r["iteration"]: r for r in log["history"]}
    screens = {r["iteration"]: r for r in log["physical_screens"]}
    end = log["iterations"]
    fields = {
        "normalized_objective": ("history", "normalized_objective"),
        "physical_stationarity": ("screens", "physical_normalized_stationarity_l2"),
        "complementarity": ("screens", "normalized_complementarity"),
        "dual_negativity": ("screens", "normalized_dual"),
        "primal_solver_violation": ("history", "constraint_violation"),
        "trust_radius": ("history", "trust_radius"),
        "barrier_parameter": ("history", "barrier_parameter"),
        "raw_optimality": ("history", "optimality"),
    }
    results = {}
    for window in windows:
        row = {}
        for name, (source, key) in fields.items():
            values = history if source == "history" else screens
            points = [(t, r[key]) for t, r in sorted(values.items()) if end - window < t <= end]
            if len(points) < 2:
                row[name] = dict(available=False, snapshots=len(points))
                continue
            x, y = np.asarray(points, dtype=float).T
            slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            row[name] = dict(
                available=True,
                snapshots=len(points),
                first_iteration=int(x[0]),
                last_iteration=int(x[-1]),
                first=float(y[0]),
                last=float(y[-1]),
                change=float(y[-1] - y[0]),
                slope_per_iteration=slope,
                relative_change=float((y[-1] - y[0]) / abs(y[0])) if y[0] else None,
                minimum=float(y.min()),
                maximum=float(y.max()),
            )
        results[str(window)] = row
    return results


def telemetry(log):
    saved_keys = [
        "iterations",
        "scipy_status",
        "message",
        "optimality",
        "constraint_violation",
        "barrier_parameter",
        "barrier_tolerance",
        "trust_radius",
        "cg_iterations",
        "cg_stop_cond",
        "function_evaluations",
        "runtime_seconds",
        "solver_ball_complementarity",
        "solver_ball_dual_negativity",
    ]
    result = {k: dict(available=k in log, value=log.get(k)) for k in saved_keys}
    for k in [
        "constraint_penalty",
        "gradient_evaluations",
        "hessian_evaluations",
        "internal_slack_variables",
        "per_iteration_multipliers",
        "actual_accepted_step_norms",
    ]:
        result[k] = dict(available=False, value=None, reason="not serialized by historical V11 wrapper")
    result["scientific_multipliers"] = dict(
        available=True,
        count=len(log["multipliers"]),
        values_source="historical optimizer.multipliers",
        summary=distribution(log["multipliers"]),
    )
    result["ball_multipliers"] = dict(
        available=True,
        count=len(log["ball_constraint_multipliers"]),
        values_source="historical optimizer.ball_constraint_multipliers",
        summary=distribution(log["ball_constraint_multipliers"]),
    )
    return result


def local_analysis(b, delta, mu, log, config, limits):
    cert = direct.certificate(b, delta, mu, config, limits)
    d, f, c, tr, o, g, j = physical.physical_jacobians(b, delta)
    cv = c.detach().numpy()
    eligible = tr["eligible"].numpy()
    a, ids, balls = physical.active_system(
        delta.numpy(), eligible, cv, j, config["active_normalized_slack"], config["boundary_relative_tolerance"]
    )
    reconstructed = physical.components(delta.numpy(), eligible, g, j, cv, mu, config["boundary_relative_tolerance"])
    for key in cert["physical"]:
        if key == "ball_multipliers":
            np.testing.assert_array_equal(reconstructed[key], cert["physical"][key])
        else:
            assert reconstructed[key] == cert["physical"][key], key
    direction, qp = tangent_diagnostic(g, a, delta.shape, config)
    assert qp["projected_feasible_gradient_norm"] == cert["projected_feasible_gradient_norm"]
    trial = shadows(b, delta, direction, f, float(o.baseline["local"]), config)
    for actual, expected in zip(trial, cert["shadow_steps"], strict=True):
        assert {k: actual[k] for k in expected} == expected
    parts = split_gradient(delta.numpy(), g, eligible)
    lag_parts = split_gradient(delta.numpy(), g + j.T @ mu, eligible)
    active = np.zeros(parts["valid"].shape, dtype=bool)
    active[balls] = True
    counts = {"correction_balls": len(balls), "aligned_RMSD": int(0 in ids), "continuous_chirality": int(1 in ids)}
    for name, (lo, hi) in o.quartets.slices.items():
        counts[name] = int(((ids >= lo + 2) & (ids < hi + 2)).sum())
    gradient_info = {
        name: decomposition_summary(parts, subset)
        for name, subset in [("all_eligible", parts["valid"]), ("active_balls", active), ("interior", ~active)]
    }
    lag_info = {
        name: decomposition_summary(lag_parts, subset)
        for name, subset in [("all_eligible", parts["valid"]), ("active_balls", active), ("interior", ~active)]
    }
    unit = parts["unit"].reshape(delta.shape)
    dr = (direction * unit).sum(-1)[..., None] * unit
    dt = direction - dr
    contribution = (-g.reshape(delta.shape) * direction).sum(-1)
    radial_contribution = (-g.reshape(delta.shape) * dr).sum(-1)
    tangential_contribution = (-g.reshape(delta.shape) * dt).sum(-1)
    active_shape = active.reshape(eligible.shape)
    position = []
    step = []
    active_rows = []
    n = b["pg"].shape[1]
    for i in range(n):
        position.append(
            dict(
                residue_index=i,
                normalized_position=i / max(n - 1, 1),
                predicted_decrease_per_angstrom=float(contribution[:, 0, i].sum()),
                radial_contribution=float(radial_contribution[:, 0, i].sum()),
                tangential_contribution=float(tangential_contribution[:, 0, i].sum()),
                active_ball_vectors=int(active_shape[:, 0, i].sum()),
            )
        )
    for t in range(8):
        step.append(
            dict(
                step=t + 1,
                predicted_decrease_per_angstrom=float(contribution[t].sum()),
                active_contribution=float(contribution[t][active_shape[t]].sum()),
                interior_contribution=float(contribution[t][~active_shape[t]].sum()),
                radial_contribution=float(radial_contribution[t].sum()),
                tangential_contribution=float(tangential_contribution[t].sum()),
            )
        )
    for idx in balls:
        t, _, i = np.unravel_index(idx, eligible.shape)
        active_rows.append(
            dict(
                step=int(t) + 1,
                residue_index=int(i),
                normalized_position=int(i) / max(n - 1, 1),
                radial_gradient_per_angstrom=float(parts["radial"][idx]),
                tangential_gradient_norm_per_angstrom=float(np.linalg.norm(parts["tangent"][idx])),
                frozen_ball_multiplier=float(reconstructed["ball_multipliers"][idx]),
            )
        )
    valid = parts["valid"]
    ball_slack = np.asarray(log["ball_constraint_values"])
    ball_mu = np.asarray(log["ball_constraint_multipliers"])
    centered = ball_slack * ball_mu - log["barrier_parameter"]
    # Equivalent optimizer gradient, including exact quadratic-ball multipliers.
    z = delta.numpy().reshape(-1) / 0.04
    equivalent_lag = 0.04 * (g + j.T @ mu) + 2 * np.repeat(ball_mu, 3) * z
    raw_optimality = float(np.max(np.abs(equivalent_lag)))
    np.testing.assert_allclose(raw_optimality, log["optimality"], atol=1e-12, rtol=1e-8)
    correlation = None
    if len(balls) > 2:
        x = ball_mu[balls]
        y = np.linalg.norm(parts["tangent"][balls], axis=-1) * 0.04
        if x.std() > 0 and y.std() > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
    return dict(
        physical_certificate=cert,
        normalized_objective=float(f.detach()),
        raw_local_mse=float(f.detach() * o.baseline["local"]),
        objective_gradient_norm_per_angstrom=float(np.linalg.norm(g)),
        raw_local_gradient_norm=float(np.linalg.norm(g) * o.baseline["local"]),
        active_counts=counts,
        active_geometry=jacobian_geometry(a),
        gradient_decomposition=gradient_info,
        lagrangian_decomposition=lag_info,
        active_ball_rows=active_rows,
        tangent_QP=qp,
        shadow_steps=trial,
        material_feasible_descent=any(s["material"] for s in trial),
        direction_distribution=dict(
            radial_l2=float(np.linalg.norm(dr)),
            tangential_l2=float(np.linalg.norm(dt)),
            active_l2=float(np.linalg.norm(direction[active_shape])),
            interior_l2=float(np.linalg.norm(direction[~active_shape])),
            total_l2=float(np.linalg.norm(direction)),
        ),
        descent_by_step=step,
        descent_by_residue=position,
        boundary_geometry=dict(
            saturation=distribution(parts["radius"][valid] / 0.04),
            fraction_at_least_99pct=float((parts["radius"][valid] >= 0.99 * 0.04).mean()),
            exact_frozen_active_fraction=float(active[valid].mean()),
        ),
        barrier_geometry=dict(
            eligible_ball_slack=distribution(ball_slack[valid]),
            eligible_solver_ball_multiplier=distribution(ball_mu[valid]),
            active_solver_ball_multiplier=distribution(ball_mu[balls]),
            active_multiplier_vs_local_tangent_correlation=correlation,
            implied_true_slack_centering=distribution(centered[valid]),
            inferred_centering_is_not_saved_internal_slack=True,
            barrier_to_configured_terminal_ratio=log["barrier_parameter"] / 1e-12,
        ),
        optimizer_optimality_recomputed=raw_optimality,
    )
