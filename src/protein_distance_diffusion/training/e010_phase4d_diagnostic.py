"""Independent finite parameter displacements for frozen Phase 4D calibration.

No optimizer, parameter fitting, checkpoint continuation, or device selection.
"""

import copy
import hashlib
import math
from pathlib import Path

import torch

from ..models.e010_hybrid_local import PSEUDOSCALAR_INDEX, local_representation
from .e010_phase4d import aggregate_metrics, example_metrics, hybrid_losses

COMPONENTS = ("local_mean", "cartesian", "displacement")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_file_pins(pins):
    for path, expected in pins.items():
        if file_hash(path) != expected:
            raise ValueError(f"input mutated or hash mismatch: {path}")


def state_hash(module):
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def dot(a, b):
    return sum((x.double() * y.double()).sum() for x, y in zip(a, b, strict=True))


def norm(grads):
    return float(dot(grads, grads).sqrt())


def combine(local, cart, disp, *, beta, rho, active):
    return [a + beta * float(active) * b + rho * c for a, b, c in zip(local, cart, disp, strict=True)]


def guard_status(cartesian, baseline, tolerance):
    excess = cartesian - (1 + tolerance) * baseline
    return {"active": excess > 0, "value": max(excess, 0.0), "threshold": (1 + tolerance) * baseline}


def module_norms(model, grads):
    groups = {}
    for (name, _), g in zip(model.named_parameters(), grads, strict=True):
        parts = name.split(".")
        group = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        groups.setdefault(group, []).append(g)
    return {k: norm(v) for k, v in groups.items()}


def displaced_copy(base, direction, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    candidate = copy.deepcopy(base)
    with torch.no_grad():
        for p, d in zip(candidate.parameters(), direction, strict=True):
            if p.shape != d.shape or not torch.isfinite(d).all():
                raise ValueError("invalid displacement direction")
            p.add_(d, alpha=alpha)
    return candidate


def signed_status(p, y, mask):
    rp, ry = local_representation(p, mask), local_representation(y, mask)
    ps = rp["features"][..., PSEUDOSCALAR_INDEX]
    ys = ry["features"][..., PSEUDOSCALAR_INDEX]
    assess = rp["pseudoscalar_assessable"] & ry["pseudoscalar_assessable"] & (ps.abs() > 1e-6) & (ys.abs() > 1e-6)
    return assess, (ps * ys < 0) & assess


def aggregate_rows(rows):
    summary = aggregate_metrics(rows)
    groups = [(summary["overall"], rows)]
    groups.extend(
        (summary["by_condition"][str(c)], [r for r in rows if r["condition"] == c])
        for c in sorted({r["condition"] for r in rows})
    )
    groups.extend(
        (summary["by_stratum"][s], [r for r in rows if r["stratum"] == s]) for s in sorted({r["stratum"] for r in rows})
    )
    for s, group in groups:
        if not s["all_finite"]:
            continue
        s["local_mse"] = {str(k): sum(r["local_mse"][str(k)] for r in group) / len(group) for k in (1, 2, 3)}
        s["local_rmse_from_mean_mse"] = {k: math.sqrt(v) for k, v in s["local_mse"].items()}
        s["local_mean_mse"] = sum(s["local_mse"].values()) / 3
        s["mean_local_rmse_from_mean_mse"] = sum(s["local_rmse_from_mean_mse"].values()) / 3
        s["displacement_mean_square"] = sum(r["displacement_mean_square"] for r in group) / len(group)
        # Historical summary averages per-example RMS. Add exact objective RMS too.
        s["displacement_rms_equal_example"] = math.sqrt(s["displacement_mean_square"])
        for name in (
            "chirality_common_assessable",
            "chirality_common_inversions",
            "chirality_common_baseline_inversions",
            "chirality_assessability_lost",
            "chirality_assessability_gained",
            "input_frame_eligible",
            "input_frame_degenerate",
        ):
            s[name] = sum(r[name] for r in group)
        s["chirality_inversion_rate"] = (
            s["chirality_inversions"] / s["chirality_assessable"] if s["chirality_assessable"] else None
        )
        s["correction_gate"] = None
    return summary


def evaluate_point(branch, panel):
    """Streaming equal-example gradients; effective-panel guard formed afterward."""
    params = tuple(branch.parameters())
    gradients = {k: [torch.zeros_like(p) for p in params] for k in COMPONENTS}
    objective = {k: 0.0 for k in (*COMPONENTS, "global_cartesian")}
    rows = []
    n = len(panel)
    if n == 0:
        raise ValueError("empty diagnostic panel")
    for item in panel:
        pg, x, y, m = (item[k] for k in ("pg", "source", "target", "mask"))
        out = branch(pg, m)
        losses = hybrid_losses(out["prediction"], pg, x, y, m)
        for key in objective:
            objective[key] += float(losses[key].detach()) / n
        for key in COMPONENTS:
            gg = torch.autograd.grad(losses[key], params, retain_graph=True, allow_unused=True)
            for accum, g in zip(gradients[key], gg, strict=True):
                if g is not None:
                    accum.add_(g / n)
        with torch.no_grad():
            row = example_metrics(out["prediction"], pg, x, y, m)[0]
            row.update({k: item[k] for k in ("sample_id", "condition", "stratum")})
            if row["finite"]:
                row["local_mse"] = {str(k): float(losses[f"local_{k}"]) for k in (1, 2, 3)}
                row["displacement_mean_square"] = float(losses["displacement"])
                point_assess, point_inv = signed_status(out["prediction"], y, m)
                base_assess, base_inv = signed_status(pg, y, m)
                common = point_assess & base_assess
                row.update(
                    chirality_common_assessable=int(common.sum()),
                    chirality_common_inversions=int((point_inv & common).sum()),
                    chirality_common_baseline_inversions=int((base_inv & common).sum()),
                    chirality_assessability_lost=int((base_assess & ~point_assess).sum()),
                    chirality_assessability_gained=int((point_assess & ~base_assess).sum()),
                    input_frame_eligible=int(out["eligible"].sum()),
                    input_frame_degenerate=int(out["degenerate"].sum()),
                    correction_gate=None,
                )
            rows.append(row)
    if not all(torch.isfinite(g).all() for gg in gradients.values() for g in gg):
        raise ValueError("nonfinite diagnostic gradient")
    return {"objective": objective, "metrics": aggregate_rows(rows), "per_example": rows}, gradients


def changes(point, baseline):
    return {
        "local_improvement_fraction": 1 - point["mean_local_rmse"] / baseline["mean_local_rmse"],
        "local_offset_change_fraction": {
            k: point["local_rmse"][k] / baseline["local_rmse"][k] - 1 for k in ("1", "2", "3")
        },
        "raw_cartesian_change": point["raw_cartesian"] - baseline["raw_cartesian"],
        "raw_cartesian_change_fraction": point["raw_cartesian"] / baseline["raw_cartesian"] - 1,
        "aligned_rmsd_change_fraction": point["aligned_rmsd"] / baseline["aligned_rmsd"] - 1,
        "chirality_inversion_count_change": point["chirality_inversions"] - baseline["chirality_inversions"],
        "chirality_assessability_count_change": point["chirality_assessable"] - baseline["chirality_assessable"],
        "chirality_inversion_rate_change": point["chirality_inversion_rate"] - baseline["chirality_inversion_rate"],
    }


def primary_criteria(point, baseline):
    diff = changes(point, baseline)
    checks = {
        "mean_local_lower": diff["local_improvement_fraction"] > 0,
        "all_offsets_non_harmful": all(v <= 0 for v in diff["local_offset_change_fraction"].values()),
        "aligned_degradation_le_1pct": diff["aligned_rmsd_change_fraction"] <= 0.01,
        "chirality_non_worsening": diff["chirality_inversion_rate_change"] <= 0,
        "displacement_rms_le_1A": point["displacement_rms_equal_example"] <= 1,
        "displacement_max_le_3A": point["displacement_max"] <= 3,
        "all_finite": point["all_finite"],
    }
    return {**checks, "simultaneous_pass": all(checks.values())}


def run_diagnostic(base, panel, protocol, progress=None):
    """Prove initial objective directions identical, evaluate independent copies."""
    base_pin = state_hash(base)
    initial, gg = evaluate_point(base, panel)
    gl, gc, gd = (gg[k] for k in COMPONENTS)
    direction = [-g for g in gl]
    initial_variants = []
    for variant, coeff in protocol["variants"].items():
        for tol in protocol["delta_cart"]:
            guard = guard_status(initial["objective"]["cartesian"], initial["objective"]["global_cartesian"], tol)
            total = combine(gl, gc, gd, **coeff, active=guard["active"])
            if not all(torch.equal(a, b) for a, b in zip(total, gl, strict=True)):
                raise AssertionError("initial directions differ; reuse disallowed")
            initial_variants.append(
                {
                    "variant": variant,
                    "delta_cart": tol,
                    "total_gradient_norm": norm(total),
                    "local_gradient_norm": norm(gl),
                    "cartesian_gradient_norm": norm(gc),
                    "guard_gradient_norm": norm(gc) if guard["active"] else 0.0,
                    "displacement_gradient_norm": norm(gd),
                    "raw_local_cartesian_cosine": float(dot(gl, gc)) / (norm(gl) * norm(gc)),
                    "per_module": {
                        "total": module_norms(base, total),
                        "local": module_norms(base, gl),
                        "cartesian": module_norms(base, gc),
                    },
                    "direction": "negative_raw_total_gradient",
                    "predicted_local_derivative": float(dot(gl, direction)),
                    "predicted_cartesian_derivative": float(dot(gc, direction)),
                    "initial_guard": guard,
                }
            )
    points, records = [], []
    for alpha in protocol["alpha"]:
        candidate = displaced_copy(base, direction, alpha)
        point, g = evaluate_point(candidate, panel) if alpha else (initial, gg)
        point = copy.deepcopy(point)
        point.update(
            alpha=alpha,
            component_gradient_norms={k: norm(v) for k, v in g.items()},
            component_per_module={k: module_norms(candidate, v) for k, v in g.items()},
            displacement_penalty_gradient_norm=0.01 * norm(g["displacement"]),
            displacement_penalty_to_local_gradient_norm_ratio=0.01 * norm(g["displacement"]) / norm(g["local_mean"]),
        )
        point["change"] = changes(point["metrics"]["overall"], initial["metrics"]["overall"])
        point["primary_criteria"] = primary_criteria(point["metrics"]["overall"], initial["metrics"]["overall"])
        points.append(point)
        for variant, coeff in protocol["variants"].items():
            for tol in protocol["delta_cart"]:
                guard = guard_status(point["objective"]["cartesian"], point["objective"]["global_cartesian"], tol)
                total = combine(g["local_mean"], g["cartesian"], g["displacement"], **coeff, active=guard["active"])
                records.append(
                    {
                        "variant": variant,
                        "delta_cart": tol,
                        "alpha": alpha,
                        "point_index": len(points) - 1,
                        "objective": point["objective"]["local_mean"]
                        + coeff["beta"] * guard["value"]
                        + coeff["rho"] * point["objective"]["displacement"],
                        "guard": guard,
                        "guard_in_objective": bool(coeff["beta"]),
                        "total_gradient_norm": norm(total),
                        "guard_gradient_norm": norm(g["cartesian"]) if guard["active"] else 0.0,
                        "penalty_gradient_norm": coeff["rho"] * norm(g["displacement"]),
                        "total_directional_derivative": float(dot(total, direction)),
                    }
                )
        if state_hash(base) != base_pin:
            raise AssertionError("baseline local branch mutated")
        if progress:
            progress(
                f"alpha={alpha}: mean-local={point['metrics']['overall']['mean_local_rmse']:.6f}, "
                f"RMS displacement={point['metrics']['overall']['displacement_rms_equal_example']:.6f} A"
            )
    activation = []
    for tol in protocol["delta_cart"]:
        active = [
            p
            for p in points
            if guard_status(p["objective"]["cartesian"], p["objective"]["global_cartesian"], tol)["active"]
        ]
        p = active[0] if active else None
        activation.append(
            {
                "delta_cart": tol,
                "first_sampled_alpha": p["alpha"] if p else None,
                "activation_bracket": None if p is None else [points[max(0, points.index(p) - 1)]["alpha"], p["alpha"]],
                "local_improvement_fraction_at_activation": p["change"]["local_improvement_fraction"] if p else None,
                "raw_cartesian_cost_at_activation": p["change"]["raw_cartesian_change"] if p else None,
                "raw_cartesian_cost_fraction_at_activation": p["change"]["raw_cartesian_change_fraction"]
                if p
                else None,
                "displacement_rms_at_activation": p["metrics"]["overall"]["displacement_rms_equal_example"]
                if p
                else None,
            }
        )
    return {
        "initial_variants": initial_variants,
        "all_initial_directions_equal": True,
        "points": points,
        "variant_records": records,
        "guard_activation": activation,
        "baseline_local_unchanged": state_hash(base) == base_pin,
        "optimizer_steps": 0,
    }
