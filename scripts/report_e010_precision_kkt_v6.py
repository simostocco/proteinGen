#!/usr/bin/env python3
"""Aggregate fixed V6 records without changing any solver or audit tolerances."""

import json

import torch

from scripts import audit_e010_phase4d_precision_kkt_v6 as run
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def main():
    cfg = run.contract()
    ids = cfg["stalled_indices"] + cfg["control_indices"]
    rows = [json.loads((run.OUT / "examples" / f"example_{i:02d}.json").read_text()) for i in ids]
    stalled = [r for r in rows if r["stalled"]]
    replay_path = run.OUT / "precision_replay_result.json"
    replay = json.loads(replay_path.read_text()) if replay_path.exists() else None
    failures = sum(len(r["finite_differences"]["failed_directions"]) for r in rows)
    chain_failures = sum(
        r["chain_rule_relative_error"] > cfg["finite_difference"]["relative_error_tolerance"] for r in rows
    )
    material_shadow = [r["index"] for r in stalled if any(s["material_decrease"] for s in r["shadows"])]
    material_gradient = [
        r["index"]
        for r in rows
        if r["gradient_relative_difference"] > cfg["precision_gradient_relative_materiality"]
        or r["gradient_cosine"] < cfg["precision_gradient_cosine_materiality"]
    ]
    conditioning = [
        r["index"]
        for r in stalled
        if r["index"] in material_shadow
        and r["kkt_normalized_max"] > cfg["near_kkt_normalized_max_tolerance"]
        and r["gradient_norms"]["float64"] < r["correction_gradient_norm"]
    ]
    near_kkt = [r["index"] for r in stalled if r["kkt_normalized_max"] <= cfg["near_kkt_normalized_max_tolerance"]]
    classification = (
        "PREC-D"
        if failures or chain_failures
        else "PREC-B"
        if conditioning
        else "PREC-A"
        if len(near_kkt) == 16 and not material_shadow
        else "PREC-C"
        if material_gradient and replay and replay["converged"] > 0
        else "PREC-E"
    )
    groups = {
        **{name: [r for r in rows if r["stalled"] == flag] for name, flag in [("stalled", True), ("control", False)]},
        **{
            f"{name}_condition_{c}": [r for r in rows if r["stalled"] == flag and r["record"]["condition"] == c]
            for name, flag in [("stalled", True), ("control", False)]
            for c in (50, 250, 450)
        },
    }
    radial_groups = {}
    for name, group in groups.items():
        bystep = []
        for t in range(4):
            arrays = {
                k: []
                for k in (
                    "v_norm",
                    "actual_delta_norm",
                    "saturation",
                    "a",
                    "radial_eigenvalue",
                    "tangential_eigenvalue",
                    "condition_number",
                    "g_v_norm",
                    "g_delta_norm",
                    "radial_gradient",
                    "tangent_gradient_norm",
                    "feasible_radial_residual",
                    "kkt_residual",
                )
            }
            for r in group:
                n = len(r["variable_columns"]["eligible"]) // 4
                eligible = r["variable_columns"]["eligible"][t * n : (t + 1) * n]
                for k in arrays:
                    arrays[k] += [
                        x for x, e in zip(r["variable_columns"][k][t * n : (t + 1) * n], eligible, strict=True) if e
                    ]
            bystep.append(
                {k: run.distribution(torch.tensor(values, dtype=torch.float64)) for k, values in arrays.items()}
            )
        radial_groups[name] = bystep
    fd = [x for r in rows for space in ("v", "delta") for x in r["finite_differences"][space]]
    result = dict(
        classification=classification,
        audited_stalled=cfg["stalled_indices"],
        controls=cfg["control_indices"],
        failed_fd_directions=failures,
        failed_chain_checks=chain_failures,
        material_shadow_examples=material_shadow,
        material_gradient_precision_examples=material_gradient,
        near_kkt_stalled_examples=near_kkt,
        objective_precision_material_examples=[r["index"] for r in rows if r["objective_precision_material"]],
        maximum_objective_absolute_difference=max(abs(r["objective_difference"]) for r in rows),
        maximum_objective_relative_difference=max(r["objective_relative_difference"] for r in rows),
        maximum_gradient_relative_difference=max(r["gradient_relative_difference"] for r in rows),
        minimum_gradient_cosine=min(r["gradient_cosine"] for r in rows),
        fd_relative_error=run.distribution(torch.tensor([r["relative_error"] for r in fd], dtype=torch.float64)),
        fd_absolute_error=run.distribution(torch.tensor([r["absolute_error"] for r in fd], dtype=torch.float64)),
        radial_by_group_step=radial_groups,
        optional_precision_replay=replay,
        historical_condition_450_local_gain_pct=0.06631446765544835,
        cuda_used=False,
        neural_training_launched=False,
    )
    write(run.OUT / "audit_result.json", result)
    lines = [
        "# E010 Phase 4D v6 precision and correction-space KKT audit",
        "",
        f"Classification: **{classification}**. Historical v5 CART-O5 remains unchanged.",
        "",
        "All sixteen historical stalls and six pre-registered strongly-converged controls were recovered by the "
        "exact historical solver with mixed historical arithmetic. Final/state hashes, metrics, convergence status, "
        "iterations, closure evaluations and complete recorded history matched exactly. Recovery is the explicitly "
        "authorized state-reconstruction exception, not a replacement scientific result. Recovered tensors remain "
        "outside Git.",
        "",
        "Fixed scientific settings: K=4, s_max=.04 Å, beta=16.8, gamma=2, original panel/Pg/targets/masks, current "
        "geometric correction eligibility and frozen chiral quartets. No neural model or CUDA. Numerical tolerances, "
        "examples, shadow scales and classification rules were frozen before recovery/audit.",
        "",
        "For delta=a*v, J=a*I-a³*v*v^T/s²: radial eigenvalue a³, tangential a, condition number a^-2. "
        "Physical variables are v=.04*z. The correction-space gradient is taken before the radial Jacobian, "
        "and the analytic chain is tested against autograd.",
        "",
        "At active balls, lambda=max(0,-g·delta/(2||delta||²)); residual g+2*lambda*delta retains tangential and "
        "feasible inward descent. Interior residual is g. Near-boundary tolerance is 1e-6 relative; normalized KKT "
        "threshold .001. This is distinct from the historical v-space gradient and projected finite-step mapping.",
        "",
        f"Maximum absolute/relative objective precision differences: "
        f"{result['maximum_objective_absolute_difference']:.9g}/{result['maximum_objective_relative_difference']:.9g}. "
        f"Material objective cases: {result['objective_precision_material_examples']}. "
        f"Maximum relative gradient difference {result['maximum_gradient_relative_difference']:.9g}; "
        f"minimum cosine {result['minimum_gradient_cosine']:.12g}. Material gradient cases {material_gradient}.",
        "",
        f"Finite-difference maximum/median relative error: {result['fd_relative_error']['maximum']:.9g}/"
        f"{result['fd_relative_error']['median']:.9g}. All individual scales/errors are retained; validation requires "
        f"at least one passing epsilon per fixed direction/space under mixed absolute+relative tolerances. "
        f"Failed directions: {failures}; failed analytic chain checks: {chain_failures}.",
        "",
        f"Near-KKT stalled examples: {near_kkt}. Material single shadow-step decreases: {material_shadow}. "
        "Shadow perturbations are not iterated and keep the original correction balls; "
        "eligibility changes invalidate comparisons.",
        "",
        "## Per-example audit",
        "",
        "| Index | Identity | Condition | Group | g_v norm | g_delta norm | Normalized KKT max | Active fraction "
        "| Median kappa | Median radial eigenvalue | Relative gradient precision difference | Shadow material |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        r["index"],
                        r["record"]["sample_id"],
                        r["record"]["condition"],
                        "stall" if r["stalled"] else "control",
                        f"{r['gradient_norms']['float64']:.9g}",
                        f"{r['correction_gradient_norm']:.9g}",
                        f"{r['kkt_normalized_max']:.9g}",
                        f"{r['boundary_active_fraction']:.9g}",
                        f"{r['radial_summary']['condition_number']['median']:.9g}",
                        f"{r['radial_summary']['radial_eigenvalue']['median']:.9g}",
                        f"{r['gradient_relative_difference']:.9g}",
                        r["index"] in material_shadow,
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Group/step distributions",
        "",
        "Each row is one recurrent step, with pooled eligible-residue statistics. Full quantiles for all requested "
        "quantities, per-variable columns (including fixed endpoints), metric differences and shadow results "
        "are retained in JSON.",
        "",
        "| Group | Step | Median kappa | Max kappa | Median radial eigenvalue | Median tangential eigenvalue "
        "| Median saturation | Max KKT residual |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g, steps in radial_groups.items():
        for t, s in enumerate(steps):
            lines.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        [
                            g,
                            t,
                            f"{s['condition_number']['median']:.9g}",
                            f"{s['condition_number']['maximum']:.9g}",
                            f"{s['radial_eigenvalue']['median']:.9g}",
                            f"{s['tangential_eigenvalue']['median']:.9g}",
                            f"{s['saturation']['median']:.12g}",
                            f"{s['kkt_residual']['maximum']:.9g}",
                        ],
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Optional precision replay",
        "",
        json.dumps(replay, indent=2) if replay else "Not executed. No optimizer is part of the fixed-state audit.",
        "",
        "## Scientific limit",
        "",
        "Even a fully converged minimum of L_local+16.8*L_cart+2*L_chiral does not determine maximum local repair "
        "subject to Cartesian/chirality safety. Weighted-sum optimization and constrained local feasibility are "
        "different questions. Condition-450 historical gain remains 0.066314%; this audit cannot alone establish "
        "correction-budget insufficiency.",
        "",
    ]
    (run.OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "radial_by_group_step"}, indent=2))


if __name__ == "__main__":
    main()
