"""Read-only V10 panel aggregation and predetermined classification."""

import json
from statistics import median

from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.report_e010_local_feasibility_v8 import directions
from scripts.run_e010_strict_scientific_v10 import OUT, V9OUT, write


def main():
    rows = [json.loads((OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    assert json.loads((OUT / "reproduction_complete.json").read_text())["protected_hashes_verified"]
    baseline = summarize([r["baseline"] for r in rows])
    final = summarize([r["oracle"] for r in rows])
    old = json.loads((V9OUT.parent / "local_feasibility_v8/feasibility_result.json").read_text())
    conditions = {}
    for c in ["50", "250", "450"]:
        group = [r for r in rows if str(r["record"]["condition"]) == c]
        b = baseline["by_condition"][c]
        f = final["by_condition"][c]
        conditions[c] = dict(
            local_gain_pct=-percent(f, b, "mean_local_rmse"),
            offset_gains_pct={k: 100 * (1 - f["local_rmse"][k] / b["local_rmse"][k]) for k in ["1", "2", "3"]},
            aligned_change_pct=percent(f, b, "aligned_rmsd"),
            raw_cartesian_change_pct=percent(f, b, "raw_cartesian"),
            chiral_change_pct=percent(f, b, "continuous_chiral_loss"),
            converged=sum(r["optimizer"]["converged"] for r in group),
            baseline_inversions=b["chirality_inversions"],
            final_inversions=f["chirality_inversions"],
            repaired_inversions=sum(r["quartet_states"][-1]["repaired_inversions"] for r in group),
            new_inversions=sum(r["quartet_states"][-1]["new_inversions"] for r in group),
            active_signed=sum(r["quartet_states"][-1]["active_signed"] for r in group),
            minimum_signed_margin=min(r["quartet_states"][-1]["minimum_signed_margin"] for r in group),
            v8_gain_pct=old["arms"]["B"]["condition_gains_pct"][c],
        )
    nonconverged = [
        dict(index=r["index"], record=r["record"], gates=r["optimizer"]["physical_certificate"]["gates"])
        for r in rows
        if not r["optimizer"]["converged"]
    ]
    failures = [
        r["index"]
        for r in rows
        if r["oracle"]["chirality_inversions"] > r["baseline"]["chirality_inversions"]
        or r["quartet_states"][-1]["new_inversions"]
    ]
    assesslost = sum(r["oracle"]["chirality_assessability_lost"] for r in rows)
    if nonconverged:
        classification = "V10-D"
    elif failures or assesslost:
        classification = "V10-C"
    elif conditions["450"]["local_gain_pct"] >= 5:
        classification = "V10-A"
    else:
        classification = "V10-B"
    ranges = {}
    for key, extract in {
        "physical_stationarity_l2": lambda r: r["optimizer"]["physical_certificate"]["physical"][
            "physical_normalized_stationarity_l2"
        ],
        "raw_optimality": lambda r: r["optimizer"]["optimality"],
        "complementarity": lambda r: r["optimizer"]["physical_certificate"]["physical"]["normalized_complementarity"],
        "dual_negativity": lambda r: r["optimizer"]["physical_certificate"]["physical"]["normalized_dual"],
        "iterations": lambda r: r["optimizer"]["iterations"],
    }.items():
        a = [extract(r) for r in rows]
        ranges[key] = dict(minimum=min(a), median=median(a), maximum=max(a))
    result = dict(
        classification=classification,
        full_historical_target_passes=classification == "V10-A",
        successful_physical_convergence=60 - len(nonconverged),
        nonconverged=nonconverged,
        baseline=baseline,
        final=final,
        conditions=conditions,
        per_example_inversion_failures=failures,
        assessability_lost=assesslost,
        numerical_ranges=ranges,
        trajectories=[summarize([r["states"][t] for r in rows]) for t in range(9)],
        directions=directions(rows),
        shadow_material_failures=[
            r["index"]
            for r in rows
            if any(s["material"] for s in r["optimizer"]["physical_certificate"]["shadow_steps"])
        ],
        cuda_used=False,
        neural_training_launched=False,
    )
    write(OUT / "result.json", result)
    lines = [
        "# E010 Phase 4D V10 strict scientific panel",
        "",
        f"Classification: **{classification}**. Physical convergence: {60 - len(nonconverged)}/60.",
        "",
        "Same K8/.04 Å, zero initialization, exact V9 local-only objective and strict constraints. V9B physical contract used; raw optimality is telemetry.",  # noqa: E501
        "",
        "| Condition | Local gain % | V8 Arm B gain % | Offset gains 1/2/3 % | Aligned change % | Chiral change % | Inversions baseline/final | Converged |",  # noqa: E501
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for c, q in conditions.items():
        lines.append(
            f"| {c} | {q['local_gain_pct']:.6f} | {q['v8_gain_pct']:.6f} | {' / '.join(f'{v:.6f}' for v in q['offset_gains_pct'].values())} | {q['aligned_change_pct']:.6f} | {q['chiral_change_pct']:.6f} | {q['baseline_inversions']}/{q['final_inversions']} | {q['converged']}/20 |"  # noqa: E501
        )
    lines += [
        "",
        "All final states, correction trajectories, constraints, certificate checks and inversion transitions are recorded per example. No failures are filtered.",  # noqa: E501
        "",
        f"Nonconverged: `{json.dumps(nonconverged)}`.",
        f"Per-example inversion failures: `{failures}`. Assessability lost: {assesslost}.",
        "",
        f"Numerical ranges: `{json.dumps(ranges)}`.",
        "",
        "Independent reproduction recomputed every metric and physical certificate from hashed saved variables without a second optimization. Historical source and input hashes were checked.",  # noqa: E501
        "",
        "CPU float64 only. CUDA: NO. Neural training: NO. No follow-on experiment launched.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(classification, conditions, flush=True)


if __name__ == "__main__":
    main()
