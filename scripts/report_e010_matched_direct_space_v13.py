"""Full-panel matched-budget comparison, unchanged metrics and physical gates."""

import json
from collections import Counter

import numpy as np

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts import run_e010_matched_direct_space_v13 as runner
from scripts.report_e010_direct_correction_v11 import span
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.report_e010_local_feasibility_v8 import directions
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write
from scripts.summarize_e010_strict_scientific_v10 import group_summary

OUT = runner.OUT


def classify(count, material, gain, failures):
    if failures:
        return "MATCH-D"
    if count == 60 and gain >= 5:
        return "MATCH-A"
    if count >= 10 or material <= 49:
        return "MATCH-B"
    return "MATCH-C"


def numerical(rows):
    result = {
        k: span([r["optimizer"]["physical_certificate"]["physical"][name] for r in rows])
        for k, name in [
            ("stationarity", "physical_normalized_stationarity_l2"),
            ("complementarity", "normalized_complementarity"),
            ("dual_negativity", "normalized_dual"),
        ]
    }
    result.update(
        {
            k: span([r["optimizer"][k] for r in rows])
            for k in ["iterations", "optimality", "barrier_parameter", "barrier_tolerance", "trust_radius"]
        }
    )
    result["barrier_counts"] = dict(Counter(str(r["optimizer"]["barrier_parameter"]) for r in rows))
    result["solver_status_counts"] = dict(Counter(str(r["optimizer"]["scipy_status"]) for r in rows))
    return result


def main():
    runner.frozen()
    contract = runner.CONTRACT
    reproduction = json.loads((OUT / "reproduction_complete.json").read_text())
    assert len(reproduction["results"]) == 60 and all(r["reproduced"] for r in reproduction["results"])
    assert len(list((OUT / "attempts").glob("*.json"))) == 60
    rows = [json.loads((OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    prior = [json.loads((runner.HISTORICAL / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    previous = json.loads((runner.HISTORICAL / "result.json").read_text())
    baseline, final = (summarize([r[k] for r in rows]) for k in ["baseline", "oracle"])
    count = sum(r["optimizer"]["converged"] for r in rows)
    material = [
        r["index"] for r in rows if any(s["material"] for s in r["optimizer"]["physical_certificate"]["shadow_steps"])
    ]
    failures = [
        r["index"]
        for r in rows
        if r["quartet_states"][-1]["new_inversions"]
        or r["quartet_states"][-1]["assessability_lost"]
        or r["oracle"]["chirality_inversions"] > r["baseline"]["chirality_inversions"]
        or not r["optimizer"]["physical_certificate"]["gates"]["primal_feasibility"]
        or not all(s["finite"] and not s["collapse"] and s["frame_assessability_preserved"] for s in r["states"])
    ]
    conditions = {}
    for c in ["50", "250", "450"]:
        b, f = baseline["by_condition"][c], final["by_condition"][c]
        group = [r for r in rows if str(r["record"]["condition"]) == c]
        gain = -percent(f, b, "mean_local_rmse")
        conditions[c] = dict(
            local_gain_pct=gain,
            v11_local_gain_pct=previous["conditions"][c]["local_gain_pct"],
            gain_delta_percentage_points=gain - previous["conditions"][c]["local_gain_pct"],
            offset_gains_pct={k: 100 * (1 - f["local_rmse"][k] / b["local_rmse"][k]) for k in ["1", "2", "3"]},
            raw_cartesian_change_pct=percent(f, b, "raw_cartesian"),
            aligned_change_pct=percent(f, b, "aligned_rmsd"),
            chiral_change_pct=percent(f, b, "continuous_chiral_loss"),
            baseline_inversions=b["chirality_inversions"],
            final_inversions=f["chirality_inversions"],
            repaired_inversions=sum(r["quartet_states"][-1]["repaired_inversions"] for r in group),
            new_inversions=sum(r["quartet_states"][-1]["new_inversions"] for r in group),
            material_shadow_count=sum(r["index"] in material for r in group),
            numerical=numerical(group),
            **group_summary(group),
        )
    strata = {}
    for name, b in baseline["by_stratum"].items():
        f = final["by_stratum"][name]
        group = [r for r in rows if r["record"]["stratum"] == name]
        strata[name] = dict(
            local_gain_pct=-percent(f, b, "mean_local_rmse"),
            aligned_change_pct=percent(f, b, "aligned_rmsd"),
            numerical=numerical(group),
            **group_summary(group),
        )
    paired = []
    for row, old in zip(rows, prior, strict=True):
        assert row["record"] == old["record"] and row["baseline"] == old["baseline"]
        runner.prefix_match(row["index"])
        a, b = (r["optimizer"]["physical_certificate"]["physical"] for r in [row, old])
        with np.load(OUT / "untracked_states" / f"example_{row['index']:02d}.npz") as current:
            with np.load(runner.HISTORICAL / "untracked_states" / f"example_{row['index']:02d}.npz") as past:
                rms = float(np.sqrt(np.mean(np.sum((current["states"][-1] - past["states"][-1]) ** 2, axis=-1))))
        paired.append(
            dict(
                index=row["index"],
                record=row["record"],
                convergence_v11=old["optimizer"]["converged"],
                convergence_v13=row["optimizer"]["converged"],
                material_v11=any(s["material"] for s in old["optimizer"]["physical_certificate"]["shadow_steps"]),
                material_v13=row["index"] in material,
                stationarity_v11=b["physical_normalized_stationarity_l2"],
                stationarity_v13=a["physical_normalized_stationarity_l2"],
                complementarity_v11=b["normalized_complementarity"],
                complementarity_v13=a["normalized_complementarity"],
                normalized_local_objective_v11=old["optimizer"]["history"][-1]["normalized_objective"],
                normalized_local_objective_v13=row["optimizer"]["history"][-1]["normalized_objective"],
                coordinate_RMS_difference_angstrom=rms,
            )
        )
    classification = classify(count, len(material), conditions["450"]["local_gain_pct"], failures)
    result = dict(
        classification=classification,
        certified_sufficient=classification == "MATCH-A",
        baseline=baseline,
        final=final,
        conditions=conditions,
        length_strata=strata,
        physical_convergence=count,
        v11_physical_convergence=4,
        material_shadow_examples=material,
        v11_material_shadow_count=55,
        numerical=numerical(rows),
        v11_numerical=numerical(prior),
        overall=group_summary(rows),
        paired_examples=paired,
        per_example_safety_failures=failures,
        nonconverged=[
            dict(index=r["index"], record=r["record"], gates=r["optimizer"]["physical_certificate"]["gates"])
            for r in rows
            if not r["optimizer"]["converged"]
        ],
        trajectories=[summarize([r["states"][t] for r in rows]) for t in range(9)],
        correction_directions=directions(rows),
        panel_solver_invocations=60,
        outcome_based_reruns=0,
        prefix_validation_exact=True,
        only_change=dict(maxiter_v11=1000, maxiter_v13=2000),
        cuda_used=False,
        neural_training_launched=False,
        follow_on_launched=False,
    )
    write(OUT / "result.json", result)
    lines = [
        "# E010 V13 matched 2000-iteration direct-space repeat",
        "",
        f"Classification: **{classification}**. Physical convergence **{count}/60**, V11 **4/60**.",
        f"Material-shadow cases **{len(material)}**, V11 **55/60**. "
        f"Complete feasibility certified: **{result['certified_sufficient']}**.",
        "",
        "| Condition | V11 local gain % | V13 local gain % | Aligned change % | Chiral change % | Convergence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, q in conditions.items():
        lines.append(
            f"| {c} | {q['v11_local_gain_pct']:.6f} | {q['local_gain_pct']:.6f} | "
            f"{q['aligned_change_pct']:.6f} | {q['chiral_change_pct']:.6f} | {q['convergence']}/20 |"
        )
    lines += [
        "",
        "The only settings change is maxiter=1000 to 2000. Original V11 solver, Hessian-vector products, "
        "initialization, inputs, constraints and V9B physical gates are unchanged.",
        "Every example ran exactly once from zero. All metrics and certificates independently reproduced "
        "from private saved states without optimization. Historical solver-history prefixes match exactly.",
        "No example is filtered; nonconverged endpoint metrics are descriptive "
        "and do not certify full-panel feasibility.",
        "See result.json and B records for numerical distributions, paired comparisons, "
        "safety and trajectory telemetry.",
        "No CUDA, neural training, changed objective, solver or geometric budget, or automatic follow-on.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    print(classification, count, len(material), {c: q["local_gain_pct"] for c, q in conditions.items()}, flush=True)


if __name__ == "__main__":
    main()
