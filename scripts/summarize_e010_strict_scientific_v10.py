"""Supplementary read-only handoff statistics for the completed V10 panel."""

import json
from collections import Counter
from statistics import median

from scripts.report_e010_local_feasibility_v7 import percent
from scripts.run_e010_strict_scientific_v10 import OUT, write


def span(values):
    return dict(minimum=min(values), median=median(values), maximum=max(values))


def group_summary(rows):
    final_quartets = [r["quartet_states"][-1] for r in rows]
    certificates = [r["optimizer"]["physical_certificate"] for r in rows]
    active = Counter()
    for row, cert in zip(rows, certificates, strict=True):
        ids = set(cert["active_inequality_indices"])
        for name, (start, stop) in row["quartet_states"][-1]["constraint_slices"].items():
            active[name] += len(ids.intersection(range(start + 2, stop + 2)))
        active["aligned"] += 0 in ids
        active["continuous_chirality"] += 1 in ids
        active["correction_balls"] += cert["active_balls"]
    return dict(
        examples=len(rows),
        convergence=sum(r["optimizer"]["converged"] for r in rows),
        failed_gates=dict(Counter(k for c in certificates for k, passed in c["gates"].items() if not passed)),
        baseline_correct=sum(r["quartet_states"][0]["correct"] for r in rows),
        baseline_inverted=sum(r["quartet_states"][0]["inverted"] for r in rows),
        baseline_assessable=sum(r["quartet_states"][0]["assessable"] for r in rows),
        final_assessable=sum(q["assessable"] for q in final_quartets),
        temporary_assessability_lost=sum(q["assessability_lost"] for r in rows for q in r["quartet_states"]),
        active_constraints=dict(active),
        minimum_margins={
            k: min(q[k] for q in final_quartets)
            for k in [
                "minimum_signed_margin",
                "minimum_absolute_q_margin",
                "minimum_bond_margin",
                "minimum_frame_margin",
            ]
        },
        maximum_constraint_residuals=dict(
            aligned=max(r["optimizer"]["constraints"][0] for r in rows),
            continuous_chirality=max(r["optimizer"]["constraints"][1] for r in rows),
            strict_extra=max(max(r["optimizer"]["constraints"][2:]) for r in rows),
        ),
        stationarity_threshold=span([c["stationarity_threshold"] for c in certificates]),
        projected_feasible_gradient=span([c["projected_feasible_gradient_norm"] for c in certificates]),
        derivative_uncertainty=span([c["derivative_uncertainty_normalized"] for c in certificates]),
        shadow_steps=dict(
            evaluations=sum(len(c["shadow_steps"]) for c in certificates),
            feasible=sum(s["feasible"] for c in certificates for s in c["shadow_steps"]),
            feasible_material=sum(s["material"] for c in certificates for s in c["shadow_steps"]),
            maximum_feasible_normalized_decrease=max(
                [s["normalized_loss_decrease"] for c in certificates for s in c["shadow_steps"] if s["feasible"]],
                default=None,
            ),
        ),
        runtime_seconds=span([r["optimizer"]["runtime_seconds"] for r in rows]),
        all_finite=all(s["finite"] for r in rows for s in r["states"]),
        no_collapse=all(not s["collapse"] for r in rows for s in r["states"]),
        frame_assessability_preserved=all(s["frame_assessability_preserved"] for r in rows for s in r["states"]),
        maximum_step_correction_angstrom=max(s["step_correction_max"] for r in rows for s in r["states"]),
        maximum_net_displacement_angstrom=max(s["displacement_max"] for r in rows for s in r["states"]),
        maximum_path_length_angstrom=max(s["path_length_max"] for r in rows for s in r["states"]),
    )


def main():
    assert json.loads((OUT / "reproduction_complete.json").read_text())["protected_hashes_verified"]
    rows = [json.loads((OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    for row in rows:
        assert row["optimizer"]["converged"] == row["optimizer"]["physical_certificate"]["converged"]
        assert json.loads((OUT / "reproduction" / f"example_{row['index']:02d}.json").read_text())[
            "physical_certificate_exact"
        ]
    result = json.loads((OUT / "result.json").read_text())
    strata = {}
    for name, baseline in result["baseline"]["by_stratum"].items():
        final = result["final"]["by_stratum"][name]
        strata[name] = dict(
            local_gain_pct=-percent(final, baseline, "mean_local_rmse"),
            offset_gains_pct={
                k: 100 * (1 - final["local_rmse"][k] / baseline["local_rmse"][k]) for k in ["1", "2", "3"]
            },
            aligned_change_pct=percent(final, baseline, "aligned_rmsd"),
            raw_cartesian_change_pct=percent(final, baseline, "raw_cartesian"),
            chiral_change_pct=percent(final, baseline, "continuous_chiral_loss"),
            **group_summary([r for r in rows if r["record"]["stratum"] == name]),
        )
    write(
        OUT / "handoff_statistics.json",
        dict(
            interpretation=("All examples retained; nonconverged endpoints are descriptive feasible witnesses, "
                "not accepted oracle optima."),
            overall=group_summary(rows),
            conditions={
                str(c): group_summary([r for r in rows if r["record"]["condition"] == c]) for c in [50, 250, 450]
            },
            length_strata=strata,
            scientific_settings_changed=False,
            follow_on_launched=False,
        ),
    )


if __name__ == "__main__":
    main()
