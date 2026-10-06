"""Read-only full-panel K=8 aggregation, safety gates and comparison to V7."""

import json
from statistics import median

import numpy as np
import torch

from protein_distance_diffusion.training import e010_local_feasibility_v8 as v8
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.run_e010_local_feasibility_v8 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache, write


def directions(records):
    groups = {"overall": records}
    groups.update({str(c): [r for r in records if r["record"]["condition"] == c] for c in (50, 250, 450)})
    output = {}
    for key, group in groups.items():
        steps = []
        cos = []
        for t in range(len(group[0]["correction_direction_telemetry"]["steps"])):
            rows = [r["correction_direction_telemetry"]["steps"][t] for r in group]
            count = sum(r["eligible"] for r in rows)
            steps.append(
                dict(
                    step=t + 1,
                    eligible=count,
                    saturation={
                        k: sum(r["eligible"] * r["saturation"][k] for r in rows) / max(1, count)
                        for k in ("0.9", "0.95", "0.99")
                    },
                )
            )
            if t:
                values = [
                    v
                    for r in group
                    for v in r["correction_direction_telemetry"]["consecutive_cosines"][t - 1]["values"]
                ]
                a = np.asarray(values, dtype=np.float64)
                cos.append(
                    dict(
                        previous_step=t,
                        next_step=t + 1,
                        defined=len(a),
                        mean=float(a.mean()) if len(a) else None,
                        quantiles=np.quantile(a, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist() if len(a) else [],
                        negative_fraction=float((a < 0).mean()) if len(a) else None,
                    )
                )
        output[key] = dict(steps=steps, consecutive_cosines=cos)
    return output


def main():
    cfg = setup()
    cache, _ = load_cache()
    rows = {a: [json.loads((OUT / a / f"example_{i:02d}.json").read_text()) for i in range(60)] for a in "AB"}
    baseline = summarize([r["baseline"] for r in rows["A"]])
    historic = OUT.parent / "local_feasibility_v7"
    old = json.loads((historic / "feasibility_result.json").read_text())
    arms = {}
    for a, records in rows.items():
        assert all(
            r["baseline"] == oldr["baseline"]
            for r, oldr in zip(
                records,
                [json.loads((historic / a / f"example_{i:02d}.json").read_text()) for i in range(60)],
                strict=True,
            )
        )
        metrics = summarize([r["oracle"] for r in records])
        bycondition = {}
        for c in (50, 250, 450):
            group = [r for r in records if r["record"]["condition"] == c]
            bycondition[str(c)] = dict(
                converged=sum(r["optimizer"]["converged"] for r in group),
                feasible=sum(r["optimizer"]["constraint_feasible"] for r in group),
                inversion_gate_pass=sum(
                    r["oracle"]["chirality_inversions"] <= r["baseline"]["chirality_inversions"] for r in group
                ),
                assessability_gate_pass=sum(r["oracle"]["chirality_assessability_lost"] == 0 for r in group),
                aligned_active=sum(r["optimizer"]["active_constraints"][0] for r in group) if a == "B" else 0,
                chiral_active=sum(r["optimizer"]["active_constraints"][1] for r in group) if a == "B" else 0,
            )
        gains = {
            c: -percent(metrics["by_condition"][c], baseline["by_condition"][c], "mean_local_rmse")
            for c in ("50", "250", "450")
        }
        n = sum(r["optimizer"]["converged"] for r in records)
        invfail = [
            r["index"] for r in records if r["oracle"]["chirality_inversions"] > r["baseline"]["chirality_inversions"]
        ]
        arms[a] = dict(
            metrics=metrics,
            condition_gains_pct=gains,
            condition_delta_vs_v7_pp={c: gains[c] - old["arms"][a]["condition_gains_pct"][c] for c in gains},
            convergence=n,
            by_condition=bycondition,
            strong_convergence=n >= 48 and bycondition["450"]["converged"] >= 18,
            continuous_feasible=all(r["optimizer"]["constraint_feasible"] for r in records),
            inversion_fail_indices=invfail,
            all_scientific_gates=not invfail
            and all(
                r["oracle"]["chirality_assessability_lost"] == 0 and r["optimizer"]["constraint_feasible"]
                for r in records
            ),
            iterations_min_median_max=[
                min(r["optimizer"]["iterations"] for r in records),
                median(r["optimizer"]["iterations"] for r in records),
                max(r["optimizer"]["iterations"] for r in records),
            ],
            states=[summarize([r["states"][t] for r in records]) for t in range(9)],
            directions=directions(records),
            max_constraint_excess=[max(r["optimizer"]["constraints"][j] for r in records) for j in range(2)],
            max_ball_kkt=max(r["optimizer"]["stationarity"]["normalized_ball_kkt_max"] for r in records),
            temporary_losses=[
                dict(index=r["index"], record=r["record"], losses=r["temporary_assessability_losses"])
                for r in records
                if any(r["temporary_assessability_losses"])
            ],
        )
        prior = []
        for i, r in enumerate(records):
            saved = torch.load(historic / "untracked_states" / a / f"example_{i:02d}.pt", weights_only=True)
            b = {k: v.double() if v.is_floating_point() else v for k, v in batch(cache, [i], "cpu").items()}
            tr = v8.physical_trajectory(b["pg"], b["mask"], 0.04 * saved["z"].reshape(4, *b["pg"].shape))
            prior.append(dict(record=r["record"], correction_direction_telemetry=v8.correction_telemetry(tr)))
        arms[a]["v7_directions"] = directions(prior)
    A, B = arms["A"], arms["B"]
    if not A["strong_convergence"] or not B["strong_convergence"]:
        classification = "K8-E"
    elif B["condition_gains_pct"]["450"] >= 5 and B["all_scientific_gates"]:
        classification = "K8-A"
    elif A["condition_gains_pct"]["450"] >= 5:
        classification = "K8-D"
    elif B["condition_delta_vs_v7_pp"]["450"] >= cfg["material_gain_delta_percentage_points"]:
        classification = "K8-B"
    else:
        classification = "K8-C"
    result = dict(
        classification=classification,
        classification_binary_gate=cfg["classification_binary_gate"],
        baseline=baseline,
        arms=arms,
        historical_v7_condition_gains={a: old["arms"][a]["condition_gains_pct"] for a in "AB"},
        cuda_used=False,
        neural_training_launched=False,
        no_beta_gamma=True,
    )
    write(OUT / "feasibility_result.json", result)
    lines = [
        "# E010 Phase4D V8 K=8 local-feasibility oracle",
        "",
        f"Classification: **{classification}**. Strict historical per-example binary inversion gate retained.",
        "",
        "Same CPU float64 trust-constr settings as V7, max1000, exact HVPs. Only K changes4->8"
        "; radius remains.04 Å. No warm starts, weighted objectives, restarts or neural traini"
        "ng.",
        "",
        "| Condition | Arm | Convergence | V7 gain % | V8 gain % | Delta K pp | Offsets1/2/3 g"
        "ain % | Aligned change % | Raw Cartesian change % | Continuous chiral change % | Inve"
        "rsion change | Inversion gate passes | Active aligned/chiral |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in ("50", "250", "450"):
        for a in "AB":
            q = arms[a]
            f = q["metrics"]["by_condition"][c]
            b = baseline["by_condition"][c]
            g = q["by_condition"][c]
            vals = [
                c,
                a,
                f"{g['converged']}/20",
                f"{old['arms'][a]['condition_gains_pct'][c]:.6f}",
                f"{q['condition_gains_pct'][c]:.6f}",
                f"{q['condition_delta_vs_v7_pp'][c]:.6f}",
                "/".join(f"{100 * (1 - f['local_rmse'][k] / b['local_rmse'][k]):.6f}" for k in ("1", "2", "3")),
                f"{percent(f, b, 'aligned_rmsd'):.6f}",
                f"{percent(f, b, 'raw_cartesian'):.6f}",
                f"{percent(f, b, 'continuous_chiral_loss'):.6f}",
                f["chirality_inversions"] - b["chirality_inversions"],
                g["inversion_gate_pass"],
                f"{g['aligned_active']}/{g['chiral_active']}",
            ]
            lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += [
        "",
        "## Length strata",
        "",
        "| Stratum | Arm | Local gain % | Aligned change % | Chiral change % | Inversion chang"
        "e | Net RMS/max Å | Path RMS/max Å |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in ("20-64", "65-128", "129-256", "257-384", "385-500"):
        for a in "AB":
            f = arms[a]["metrics"]["by_stratum"][s]
            b = baseline["by_stratum"][s]
            vals = [
                s,
                a,
                f"{-percent(f, b, 'mean_local_rmse'):.6f}",
                f"{percent(f, b, 'aligned_rmsd'):.6f}",
                f"{percent(f, b, 'continuous_chiral_loss'):.6f}",
                f["chirality_inversions"] - b["chirality_inversions"],
                f"{f['displacement_rms']:.9f}/{f['displacement_max']:.9f}",
                f"{f['path_length_rms']:.9f}/{f['path_length_max']:.9f}",
            ]
            lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += [
        "",
        "## Every intermediate state and step",
        "",
        "All scalar metrics for P0-P8, by overall/condition/stratum, are in feasibility_result"
        ".json (arms.*.states). Every individual baseline/state, optimizer history, multiplier"
        "s, constraint residuals, inversion change and coordinate/variable digest is recorded "
        "in A/B/example_*.json.",
        "",
        "| Arm | State | Local RMSE | Raw cart | Aligned RMSD | Chiral loss | Inversions | Ass"
        "essable | Step RMS/max Å | Net RMS/max Å | Path RMS/max Å | Eligible/degenerate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in "AB":
        for t, groups in enumerate(arms[a]["states"]):
            f = groups["overall"]
            vals = [
                a,
                t,
                f["mean_local_rmse"],
                f["raw_cartesian"],
                f["aligned_rmsd"],
                f["continuous_chiral_loss"],
                f["chirality_inversions"],
                f["chirality_assessable"],
                f"{f['step_correction_rms']:.9f}/{f['step_correction_max']:.9f}",
                f"{f['displacement_rms']:.9f}/{f['displacement_max']:.9f}",
                f"{f['path_length_rms']:.9f}/{f['path_length_max']:.9f}",
                f"{f['frame_eligible']}/{f['frame_degenerate']}",
            ]
            lines.append("| " + " | ".join(map(str, vals)) + " |")
    lines += [
        "",
        "## Direction alignment and saturation",
        "",
        "Cosine values are pooled over both-eligible nonzero consecutive vectors. JSON reports"
        " count, mean, negative fraction and quantiles[0,.05,.25,.5,.75,.95,1] for each adjace"
        "nt step overall and by condition. Saturation uses eligible-count weighting at >=90/95"
        "/99%; the identical thresholds are recomputed on V7 frozen states.",
        "",
        "| Arm | Step | Saturation90/95/99% | Previous-step cosine mean/min/max |",
        "| --- | --- | --- | --- |",
    ]
    for a in "AB":
        q = arms[a]["directions"]["overall"]
        for t, s in enumerate(q["steps"]):
            c = q["consecutive_cosines"][t - 1] if t else None
            lines.append(
                f"| {a} | {t + 1} | "
                + "/".join(f"{100 * s['saturation'][k]:.6f}" for k in ("0.9", "0.95", "0.99"))
                + " | "
                + (f"{c['mean']:.9f}/{c['quantiles'][0]:.9f}/{c['quantiles'][-1]:.9f}" if c else "undefined")
                + " |"
            )
    lines += [
        "",
        "Solver success is distinct from physical KKT convergence; failed examples remain reco"
        "rded. Global optimality is not certified for these nonconvex problems. Continuous chi"
        "rality feasibility does not establish per-example binary inversion safety. No increas"
        "e of radius is justified merely by saturation.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines).rstrip() + "\n")
    print(
        classification,
        {a: (q["convergence"], q["condition_gains_pct"], q["inversion_fail_indices"]) for a, q in arms.items()},
    )


if __name__ == "__main__":
    main()
