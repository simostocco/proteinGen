"""Matched V11/V10 aggregation and preregistered classification; no optimization."""

import json
from collections import Counter
from statistics import median

import numpy as np

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.report_e010_local_feasibility_v8 import directions
from scripts.run_e010_direct_correction_v11 import OUT, frozen, old
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write
from scripts.summarize_e010_strict_scientific_v10 import group_summary


def span(values):
    return dict(minimum=min(values), median=median(values), maximum=max(values))


def main():
    frozen()
    contract = json.loads((OUT / "execution_contract.json").read_text())
    reproduction = json.loads((OUT / "reproduction_complete.json").read_text())
    assert len(reproduction["results"]) == 60 and all(r["reproduced"] for r in reproduction["results"])
    assert len(list((OUT / "attempts").glob("*.json"))) == 60
    rows = [json.loads((OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    previous_root = old.OUT / "technical_recovery"
    previous = json.loads((previous_root / "result.json").read_text())
    original_rows = [json.loads((previous_root / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    baseline, final = (summarize([r[k] for r in rows]) for k in ["baseline", "oracle"])
    conditions = {}
    for condition in ["50", "250", "450"]:
        b, f = baseline["by_condition"][condition], final["by_condition"][condition]
        group = [r for r in rows if str(r["record"]["condition"]) == condition]
        conditions[condition] = dict(
            local_gain_pct=-percent(f, b, "mean_local_rmse"),
            offset_gains_pct={k: 100 * (1 - f["local_rmse"][k] / b["local_rmse"][k]) for k in ["1", "2", "3"]},
            raw_cartesian_change_pct=percent(f, b, "raw_cartesian"),
            aligned_change_pct=percent(f, b, "aligned_rmsd"),
            chiral_change_pct=percent(f, b, "continuous_chiral_loss"),
            baseline_inversions=b["chirality_inversions"],
            final_inversions=f["chirality_inversions"],
            repaired_inversions=sum(r["quartet_states"][-1]["repaired_inversions"] for r in group),
            new_inversions=sum(r["quartet_states"][-1]["new_inversions"] for r in group),
            v10_local_gain_pct=previous["conditions"][condition]["local_gain_pct"],
            **group_summary(group),
        )
    count = sum(r["optimizer"]["converged"] for r in rows)
    material = [
        r["index"] for r in rows if any(s["material"] for s in r["optimizer"]["physical_certificate"]["shadow_steps"])
    ]
    failures = [
        r["index"]
        for r in rows
        if r["quartet_states"][-1]["new_inversions"]
        or r["oracle"]["chirality_inversions"] > r["baseline"]["chirality_inversions"]
        or r["oracle"]["chirality_assessability_lost"]
        or not r["optimizer"]["physical_certificate"]["gates"]["primal_feasibility"]
    ]
    if count == 60 and conditions["450"]["local_gain_pct"] >= 5 and not failures:
        classification = "DIRECT-A"
    elif count == 60 and conditions["450"]["local_gain_pct"] < 5 and not failures:
        classification = "DIRECT-C"
    elif count >= 18 or len(material) <= 29:
        classification = "DIRECT-B"
    else:
        classification = "DIRECT-D"
    paired = []
    for row, prior in zip(rows, original_rows, strict=True):
        assert row["record"] == prior["record"]
        assert row["baseline"] == prior["baseline"]
        if row["optimizer"]["converged"] and prior["optimizer"]["converged"]:
            a = np.load(OUT / "untracked_states" / f"example_{row['index']:02d}.npz")
            historical_path = (
                old.OUT
                / ("technical_recovery/untracked_states" if prior.get("state_provenance") else "untracked_states")
                / f"example_{row['index']:02d}.npz"
            )
            with np.load(historical_path) as b:
                coordinate_rms = float(np.sqrt(np.mean(np.sum((a["states"][-1] - b["states"][-1]) ** 2, axis=-1))))
            paired.append(
                dict(
                    index=row["index"],
                    coordinate_rms_angstrom=coordinate_rms,
                    local_mse_change={
                        k: row["oracle"]["local_mse"][k] - prior["oracle"]["local_mse"][k] for k in ["1", "2", "3"]
                    },
                    inversions_v11=row["oracle"]["chirality_inversions"],
                    inversions_v10=prior["oracle"]["chirality_inversions"],
                    path_length_rms_v11=row["oracle"]["path_length_rms"],
                    path_length_rms_v10=prior["oracle"]["path_length_rms"],
                    active_constraints_v11=row["optimizer"]["physical_certificate"]["active_inequality_indices"],
                    active_constraints_v10=prior["optimizer"]["physical_certificate"]["active_inequality_indices"],
                )
            )
            a.close()
    numerical = {
        k: span([r["optimizer"]["physical_certificate"]["physical"][name] for r in rows])
        for k, name in [
            ("physical_stationarity", "physical_normalized_stationarity_l2"),
            ("complementarity", "normalized_complementarity"),
            ("dual_negativity", "normalized_dual"),
        ]
    }
    numerical.update(
        raw_optimality=span([r["optimizer"]["optimality"] for r in rows]),
        iterations=span([r["optimizer"]["iterations"] for r in rows]),
        solver_status=dict(Counter(str(r["optimizer"]["scipy_status"]) for r in rows)),
    )
    strata = {}
    for name, b in baseline["by_stratum"].items():
        f = final["by_stratum"][name]
        strata[name] = dict(
            local_gain_pct=-percent(f, b, "mean_local_rmse"),
            aligned_change_pct=percent(f, b, "aligned_rmsd"),
            **group_summary([r for r in rows if r["record"]["stratum"] == name]),
        )
    result = dict(
        classification=classification,
        certified_sufficient=classification == "DIRECT-A",
        baseline=baseline,
        final=final,
        conditions=conditions,
        length_strata=strata,
        numerical=numerical,
        overall=group_summary(rows),
        physical_convergence=count,
        v10_physical_convergence=12,
        material_shadow_examples=material,
        v10_material_shadow_examples=35,
        per_example_safety_failures=failures,
        nonconverged=[
            dict(index=r["index"], record=r["record"], gates=r["optimizer"]["physical_certificate"]["gates"])
            for r in rows
            if not r["optimizer"]["converged"]
        ],
        trajectories=[summarize([r["states"][t] for r in rows]) for t in range(9)],
        correction_directions=directions(rows),
        converged_both=paired,
        scientific_settings_changed=False,
        parameterization_changed=True,
        requested_cap_change_2000_to_1000=True,
        cuda_used=False,
        neural_training_launched=False,
        follow_on_launched=False,
    )
    write(OUT / "result.json", result)
    lines = [
        "# E010 V11 direct correction-space strict-chirality oracle",
        "",
        f"Classification: **{classification}**. Physical convergence: **{count}/60**, historical V10 **12/60**.",
        f"Material-shadow cases: **{len(material)}**, historical **35**. "
        f"Certification: **{result['certified_sufficient']}**.",
        "",
        "| Condition | V11 local gain % | V10 gain % | Offset gains 1/2/3 % | "
        "Aligned change % | Chiral change % | Convergence |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for c, q in conditions.items():
        lines.append(
            f"| {c} | {q['local_gain_pct']:.6f} | {q['v10_local_gain_pct']:.6f} | "
            f"{' / '.join(f'{v:.6f}' for v in q['offset_gains_pct'].values())} | "
            f"{q['aligned_change_pct']:.6f} | {q['chiral_change_pct']:.6f} | {q['convergence']}/20 |"
        )
    lines += [
        "",
        "Nonconverged endpoints are descriptive; no examples are filtered. "
        "All60 metrics and physical certificates independently reproduced from saved variables without reoptimization.",
        "",
        "See result.json and per-example B records for safety, assessability, margins, active balls/constraints, "
        "numerical ranges, trajectories and paired solutions.",
        "",
        "Exactly one run per example, zero initialization, CPU float64. "
        "No radial map, no enlarged radius, no neural training, no CUDA.",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    print(classification, count, len(material), {c: q["local_gain_pct"] for c, q in conditions.items()}, flush=True)


if __name__ == "__main__":
    main()
