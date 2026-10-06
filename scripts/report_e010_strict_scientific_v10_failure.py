"""Publish the original failed V10 attempt without fabricating missing states."""

import argparse
import json

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.report_e010_local_feasibility_v8 import directions
from scripts.run_e010_strict_scientific_v10 import OUT, write
from scripts.summarize_e010_strict_scientific_v10 import group_summary, span


def main(verify=False):
    contract = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    recovery = json.loads((OUT / "original_partial_reproduction_complete.json").read_text())
    assert recovery["protected_hashes_verified"] and not recovery["full_panel_complete"]
    partial = json.loads((OUT / "original_partial_summary.json").read_text())
    rows = [json.loads(f.read_text()) for f in sorted((OUT / "B").glob("example_*.json"))]
    assert len(rows) == 43 == recovery["saved_examples"]
    for row in rows:
        verified = json.loads((OUT / "reproduction" / f"example_{row['index']:02d}.json").read_text())
        assert verified["physical_certificate_exact"] and verified["metrics_exact"] and verified["hashes_verified"]
        assert row["optimizer"]["converged"] == row["optimizer"]["physical_certificate"]["converged"]
    old = json.loads((OUT.parent / "local_feasibility_v8/feasibility_result.json").read_text())
    numerical = {}
    for name, extract in {
        "physical_stationarity_l2": lambda r: r["optimizer"]["physical_certificate"]["physical"][
            "physical_normalized_stationarity_l2"
        ],
        "raw_optimality": lambda r: r["optimizer"]["optimality"],
        "complementarity": lambda r: r["optimizer"]["physical_certificate"]["physical"]["normalized_complementarity"],
        "dual_negativity": lambda r: r["optimizer"]["physical_certificate"]["physical"]["normalized_dual"],
    }.items():
        numerical[name] = span([extract(r) for r in rows])
    strata = {}
    for name, b in partial["baseline_saved_subset"]["by_stratum"].items():
        f = partial["final_saved_subset"]["by_stratum"][name]
        strata[name] = dict(
            saved_subset_local_gain_pct=-percent(f, b, "mean_local_rmse"),
            saved_subset_aligned_change_pct=percent(f, b, "aligned_rmsd"),
            **group_summary([r for r in rows if r["record"]["stratum"] == name]),
        )
    result = dict(
        **partial,
        full_historical_target_passes=False,
        scientific_adjudication="V10-D: original execution failed; missing states and failed physical certificates.",
        all60_submitted=True,
        full_panel_convergence_count=None,
        known_physical_passes=12,
        known_nonconverged_saved=31,
        final_state_unavailable=17,
        numerical_ranges_saved_subset=numerical,
        length_strata_saved_subset=strata,
        trajectories_saved_subset=[summarize([r["states"][t] for r in rows]) for t in range(9)],
        directions_saved_subset=directions(rows),
        historical_v8_full_panel_gains_pct=old["arms"]["B"]["condition_gains_pct"],
        v8_comparison_valid_on_full_panel=False,
        protected_hashes_verified=True,
        telemetry_patch_applied=False,
        optimizer_settings_changed=False,
        cuda_used=False,
        neural_training_launched=False,
        next_action=(
            "Authorize a separate technical replay of the 17 missing states with the validated telemetry guard."
        ),
    )
    if verify:
        assert json.loads((OUT / "result.json").read_text()) == result
    else:
        write(OUT / "result.json", result)
    lines = [
        "# E010 Phase 4D V10 — original execution failure",
        "",
        "**V10-D — numerically inconclusive.** All 60 examples were submitted under the frozen contract. "
        "The original process exited with a telemetry exception; 43 final states were saved, 17 were not.",
        "",
        "Of the 43 saved states, 12 pass the complete physical contract and 31 fail. "
        "All 43 final metrics, trajectories, inversion transitions and certificates independently reproduce exactly. "
        "Protected source/input hashes remain unchanged.",
        "",
        "## Execution defect",
        "",
        "The newly added `maximum_linearized_cone_violation` telemetry uses `np.max(a @ direction)` even when "
        "the active constraint matrix has zero rows. This raises a ValueError after certificate calculations and "
        "before serialization. The first worker exception is preserved in `original_execution_failure.log`; "
        "the parent waits for its submitted workers before propagating that exception. "
        "The traceback does not identify every missing example's individual exception. "
        "No unavailable final state, metric, or convergence classification is inferred.",
        "",
        "A one-line empty-set guard is prepared in `prepared_empty_active_set.patch`, but is **not applied**. "
        "In-memory regression validation reproduces the original empty-set error, "
        "returns zero telemetry after the fix, "
        "does not silently accept the nonstationary synthetic point, and preserves the length-12/32 control "
        "certificates exactly. No recovery optimization has run.",
        "",
        "## Available-state telemetry — not the full panel",
        "",
        "Missing states prevent full-panel condition gains and direct V8 comparisons. "
        "The following numbers describe only paired saved endpoints, with no substitution for missing data.",
        "",
        "| Condition | Saved / 20 | Saved-subset local gain % | Offset gains 1/2/3 % | "
        "Aligned change % | Chiral change % |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for c, q in partial["conditions"].items():
        offsets = " / ".join(f"{v:.6f}" for v in q["saved_subset_offset_gains_pct"].values())
        lines.append(
            f"| {c} | {q['saved_examples']} | {q['saved_subset_local_gain_pct']:.6f} | {offsets} | "
            f"{q['saved_subset_aligned_change_pct']:.6f} | {q['saved_subset_chiral_change_pct']:.6f} |"
        )
    lines += [
        "",
        "All saved endpoints pass final primal safety, create no new inversions, preserve assessability, "
        "and remain finite. This does not establish safety for the 17 unavailable endpoints or physical convergence "
        "for the 31 failed certificates.",
        "",
        "Missing examples: " + json.dumps(partial["missing_examples"]) + ".",
        "",
        "Saved nonconverged examples: " + json.dumps(partial["nonconverged_saved"]) + ".",
        "",
        "Numerical ranges (saved subset): " + json.dumps(numerical) + ".",
        "",
        "Full P0–P8 telemetry, strict margins/activity, length summaries, shadows, saturation, path length and "
        "direction cosines for available states are in `result.json` and the per-example records. "
        "Full-panel inversions, final metrics and target adjudication are unavailable.",
        "",
        "Historical V8 full-panel gains (not a matched comparison): 19.2207%, 8.4448%, 8.5719% for 50/250/450. "
        "No bound-insufficiency or chirality-trade-off conclusion follows from this failed numerical attempt.",
        "",
        "Validation: 143 CPU tests passed before execution; empty-active-set regression and nonempty-control parity "
        "passed afterward without changing scientific source. K8/.04 Å, local-only objective, strict V9 constraints "
        "and V9B physical thresholds are unchanged. CUDA: NO. Neural training: NO.",
        "",
        "Recommended next action: authorize a separate technical replay of the 17 missing states with the validated "
        "telemetry guard. The 43 saved endpoints remain immutable; no failed saved solve is retried.",
    ]
    markdown = "\n".join(lines) + "\n"
    if verify:
        assert (OUT / "RESULTS.md").read_text() == markdown
    else:
        (OUT / "RESULTS.md").write_text(markdown)
    print("V10-D original failure record: 43 reproduced states, 12 physical passes, 17 unavailable states")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    main(parser.parse_args().verify)
