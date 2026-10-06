#!/usr/bin/env python3
"""Read-only aggregation of the preregistered two-arm V7 panel."""

import json
from statistics import median

from protein_distance_diffusion.training.e010_phase4d import aggregate_metrics
from scripts.run_e010_local_feasibility_v7 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def summarize(rows):
    result = aggregate_metrics(rows)
    groups = [(result["overall"], rows)]
    groups += [
        (result["by_condition"][c], [r for r in rows if str(r["condition"]) == c]) for c in result["by_condition"]
    ]
    groups += [(result["by_stratum"][s], [r for r in rows if r["stratum"] == s]) for s in result["by_stratum"]]
    for summary, group in groups:
        count = sum(r["chiral_loss_eligible"] for r in group)
        summary.update(
            continuous_chiral_loss=sum(r["chiral_error_sum"] for r in group) / max(1, count),
            **{k: sum(r[k] for r in group) / len(group) for k in ["step_correction_rms", "path_length_rms"]},
            **{k: max(r[k] for r in group) for k in ["step_correction_max", "path_length_max"]},
            chirality_assessability_lost=sum(r["chirality_assessability_lost"] for r in group),
            chirality_assessability_gained=sum(r["chirality_assessability_gained"] for r in group),
            frame_assessability_preserved=all(r["frame_assessability_preserved"] for r in group),
        )
    return result


def percent(final, baseline, key):
    return 100 * (final[key] / baseline[key] - 1)


def main():
    cfg = setup()
    rows = {
        arm: [json.loads((OUT / arm / f"example_{i:02d}.json").read_text()) for i in range(60)] for arm in ("A", "B")
    }
    assert all(a["baseline"] == b["baseline"] for a, b in zip(rows["A"], rows["B"], strict=True))
    baseline = summarize([r["baseline"] for r in rows["A"]])
    arms = {}
    for arm, records in rows.items():
        final = summarize([r["oracle"] for r in records])
        gains = {
            c: -percent(final["by_condition"][c], baseline["by_condition"][c], "mean_local_rmse")
            for c in ["50", "250", "450"]
        }
        safe = [
            r["optimizer"]["constraint_feasible"]
            and r["oracle"]["chirality_inversions"] <= r["baseline"]["chirality_inversions"]
            and r["oracle"]["chirality_assessability_lost"] == 0
            for r in records
        ]
        numeric = sum(r["optimizer"]["converged"] for r in records)
        high = sum(r["optimizer"]["converged"] for r in records if r["record"]["condition"] == 450)
        strong = (
            numeric >= cfg["convergence"]["strong_overall_min_per_arm"]
            and high >= cfg["convergence"]["strong_condition_450_min_per_arm"]
        )
        by_condition = {}
        for c in (50, 250, 450):
            group = [r for r in records if r["record"]["condition"] == c]
            by_condition[str(c)] = dict(
                converged=sum(r["optimizer"]["converged"] for r in group),
                examples=len(group),
                constraint_feasible=sum(r["optimizer"]["constraint_feasible"] for r in group),
                inversion_gate_pass=sum(
                    r["oracle"]["chirality_inversions"] <= r["baseline"]["chirality_inversions"] for r in group
                ),
                assessability_gate_pass=sum(r["oracle"]["chirality_assessability_lost"] == 0 for r in group),
                aligned_active=sum(r["optimizer"]["active_constraints"][0] for r in group),
                chiral_active=sum(r["optimizer"]["active_constraints"][1] for r in group),
            )
        iterations = [r["optimizer"]["iterations"] for r in records]
        arms[arm] = dict(
            metrics=final,
            condition_gains_pct=gains,
            convergence=numeric,
            strong_convergence=strong,
            by_condition=by_condition,
            all_final_gates=all(safe),
            final_gate_fail_indices=[r["index"] for r, s in zip(records, safe, strict=True) if not s],
            iterations_min_median_max=[min(iterations), median(iterations), max(iterations)],
            boundary_saturation_fraction_mean=sum(
                r["optimizer"]["stationarity"]["boundary_active_fraction"] for r in records
            )
            / 60,
            maximum_normalized_ball_kkt=max(r["optimizer"]["stationarity"]["normalized_ball_kkt_max"] for r in records),
            maximum_projected_ball_mapping=max(
                r["optimizer"]["stationarity"]["projected_ball_mapping_max"] for r in records
            ),
            maximum_constraint_excess=[max(r["optimizer"]["constraints"][j] for r in records) for j in range(2)],
            total_runtime_seconds=sum(r["optimizer"]["runtime_seconds"] for r in records),
            states=[summarize([r["states"][t] for r in records]) for t in range(5)],
        )
    if not all(a["strong_convergence"] for a in arms.values()):
        classification = "FEAS-E"
    elif arms["A"]["condition_gains_pct"]["450"] < 5:
        classification = "FEAS-C"
    elif arms["B"]["condition_gains_pct"]["450"] >= 5 and arms["B"]["all_final_gates"]:
        classification = "FEAS-A"
    else:
        classification = "FEAS-B"
    parent = OUT.parent
    v3 = json.loads((parent / "recurrent_capacity_v3/capacity_result.json").read_text())
    v4 = json.loads((parent / "bounded_oracle_v4/oracle_result.json").read_text())
    v5 = json.loads((parent / "cartesian_oracle_v5/oracle_result.json").read_text())
    v6 = json.loads((parent / "precision_kkt_v6/precision_comparison_summary.json").read_text())
    comparison = {
        c: dict(
            neural={
                a: 100
                * (
                    1
                    - v3["arms"][a]["final_metrics"]["by_condition"][c]["mean_local_rmse"]
                    / v3["arms"][a]["baseline_metrics"]["by_condition"][c]["mean_local_rmse"]
                )
                for a in ["S", "M", "L"]
            },
            weighted_v4=v4["condition_local_gain_pct"][c],
            weighted_v5=v5["condition_local_gain_pct"][c],
            weighted_v6_assembled=v6["assembled_panel_condition_gains_pct"][c],
            local_only_v7=arms["A"]["condition_gains_pct"][c],
            safe_local_v7=arms["B"]["condition_gains_pct"][c],
        )
        for c in ["50", "250", "450"]
    }
    result = dict(
        classification=classification,
        baseline=baseline,
        arms=arms,
        historical_comparison=comparison,
        binary_inversions_are_final_gate_only=True,
        no_beta_gamma_in_objective=True,
        cuda_used=False,
        neural_training_launched=False,
    )
    write(OUT / "feasibility_result.json", result)
    lines = [
        "# E010 Phase 4D v7 float64 local-feasibility oracle",
        "",
        f"Classification: **{classification}**.",
        "",
        "Historical v6 and earlier records remain immutable. Both arms start from zero on the same60 frozen examples. "
        "K=4, per-step radius=.04 Å. Primary objective is local distance MSE only, "
        "with constant baseline normalization. "
        "Arm B adds aligned-RMSD<=1.01*baseline and continuous-chiral<=baseline inequalities. Binary inversions and "
        "assessability are final per-example scientific gates, never gradient terms.",
        "",
        "Deterministic SciPy1.18.1 trust-constr, CPU float64, exact matrix-free objective/constraint Hessians, "
        "max1000 iterations, gtol/xtol/barrier_tol1e-12 and initial barrier1e-7. All180 baseline aligned-gradient "
        "validations and60 zero-state Hessian checks passed. The initial norm-expression Hessian defect stopped "
        "before any example result; source/config/trace are archived and the same mathematical map was corrected "
        "without changing settings. No coefficient, bound, data or per-example tuning.",
        "",
        "## Condition results",
        "",
        "| Condition | Arm | Converged | Local gain % | i+1/i+2/i+3 gains % | Aligned change % | Raw cart change % | "
        "Chiral loss change % | Inversion change | Continuous-feasible | Inversion gates | Assessability gates |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in ["50", "250", "450"]:
        for arm in ["A", "B"]:
            b = baseline["by_condition"][c]
            f = arms[arm]["metrics"]["by_condition"][c]
            conv = arms[arm]["by_condition"][c]
            lines.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        [
                            c,
                            arm,
                            f"{conv['converged']}/20",
                            f"{arms[arm]['condition_gains_pct'][c]:.6f}",
                            "/".join(
                                f"{100 * (1 - f['local_rmse'][k] / b['local_rmse'][k]):.6f}" for k in ["1", "2", "3"]
                            ),
                            *[
                                f"{percent(f, b, k):.6f}"
                                for k in ["aligned_rmsd", "raw_cartesian", "continuous_chiral_loss"]
                            ],
                            f["chirality_inversions"] - b["chirality_inversions"],
                            conv["constraint_feasible"] if arm == "B" else "not constrained",
                            conv["inversion_gate_pass"],
                            conv["assessability_gate_pass"],
                        ],
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Length strata",
        "",
        "| Stratum | Arm | Local gain % | Aligned change % | Raw cart change % | Chiral loss change % | "
        "Inversion change | Net RMS/max Å | Path RMS/max Å |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in baseline["by_stratum"]:
        for arm in ["A", "B"]:
            b = baseline["by_stratum"][s]
            f = arms[arm]["metrics"]["by_stratum"][s]
            lines.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        [
                            s,
                            arm,
                            *[
                                f"{(-1 if k == 'mean_local_rmse' else 1) * percent(f, b, k):.6f}"
                                for k in ["mean_local_rmse", "aligned_rmsd", "raw_cartesian", "continuous_chiral_loss"]
                            ],
                            f["chirality_inversions"] - b["chirality_inversions"],
                            f"{f['displacement_rms']:.6f}/{f['displacement_max']:.9f}",
                            f"{f['path_length_rms']:.6f}/{f['path_length_max']:.9f}",
                        ],
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Solver, safety and correction diagnostics",
        "",
        "| Arm | Converged | Iterations min/median/max | Max ball KKT | Max projected residual | "
        "Mean saturation | Cartesian/chiral max excess | Cartesian/chiral active | Net RMS/max Å | Path RMS/max Å |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, a in arms.items():
        f = a["metrics"]["overall"]
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        arm,
                        f"{a['convergence']}/60",
                        "/".join(map(str, a["iterations_min_median_max"])),
                        f"{a['maximum_normalized_ball_kkt']:.6g}",
                        f"{a['maximum_projected_ball_mapping']:.6g}",
                        f"{a['boundary_saturation_fraction_mean']:.6g}",
                        "/".join(f"{x:.6g}" for x in a["maximum_constraint_excess"]),
                        "/".join(
                            str(sum(g[k] for g in a["by_condition"].values()))
                            for k in ["aligned_active", "chiral_active"]
                        ),
                        f"{f['displacement_rms']:.6f}/{f['displacement_max']:.9f}",
                        f"{f['path_length_rms']:.6f}/{f['path_length_max']:.9f}",
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Historical comparison (local gain %)",
        "",
        "| Condition | Neural S/M/L | Weighted v4 | Weighted v5 | Weighted v6 assembled | V7 A | V7 B |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c, r in comparison.items():
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        c,
                        "/".join(f"{x:.6f}" for x in r["neural"].values()),
                        *[
                            f"{r[k]:.6f}"
                            for k in [
                                "weighted_v4",
                                "weighted_v5",
                                "weighted_v6_assembled",
                                "local_only_v7",
                                "safe_local_v7",
                            ]
                        ],
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        "V6 assembled values combine16 precision replays and44 historical outputs; they are not a uniform "
        "60-example precision rerun. Comparisons are descriptive and never select V7 settings.",
        "",
        "## Per-example final bookkeeping",
        "",
        "| Arm | Index | Identity | Condition | Converged | Iterations | Local gain % | Constraint residuals | "
        "Multipliers | KKT max | Inversion change | Final gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, records in rows.items():
        for r in records:
            o = r["optimizer"]
            f = r["oracle"]
            b = r["baseline"]
            gate = (
                o["constraint_feasible"]
                and f["chirality_inversions"] <= b["chirality_inversions"]
                and f["chirality_assessability_lost"] == 0
            )
            lines.append(
                "| "
                + " | ".join(
                    map(
                        str,
                        [
                            arm,
                            r["index"],
                            r["record"]["sample_id"],
                            r["record"]["condition"],
                            o["converged"],
                            o["iterations"],
                            f"{-percent(f, b, 'mean_local_rmse'):.6f}",
                            "/".join(f"{x:.6g}" for x in o["constraints"]),
                            "/".join(f"{x:.6g}" for x in o["multipliers"]),
                            f"{o['stationarity']['normalized_ball_kkt_max']:.6g}",
                            f["chirality_inversions"] - b["chirality_inversions"],
                            gate,
                        ],
                    )
                )
                + " |"
            )
    lines += [
        "",
        "Full P0-P4 telemetry, histories, coordinate/variable digests and all baseline/final metric values "
        "are in per-example JSON. Feasibility obeys the frozen normalized1e-8 tolerance. Solver success and "
        "physical convergence are recorded separately; failures cannot imply budget insufficiency. Nonconvex "
        "stationarity is not a global-optimality certificate. A binary gate failure does not prove that every "
        "possible safety-feasible correction would invert more quartets.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        classification, {a: (r["convergence"], r["condition_gains_pct"], r["all_final_gates"]) for a, r in arms.items()}
    )


if __name__ == "__main__":
    main()
