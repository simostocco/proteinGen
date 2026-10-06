"""Descriptive V12 evidence signatures and contribution summaries; no state changes."""

import json

from protein_distance_diffusion.training import e010_fixed_state_audit_v12 as audit
from scripts import audit_e010_fixed_state_v12 as runner
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def publish_equal(path, value):
    if path.exists():
        assert path.read_text() == json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        write(path, value)


def main():
    runner.setup()
    protocol = json.loads((runner.OUT / "preregistration.json").read_text())
    runner.assert_file_pins(protocol["protected_sha256"])
    rows = [json.loads((runner.OUT / "cases" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    non = [r for r in rows if r["group"] != "A"]
    progress = [
        r["index"]
        for r in non
        if r["evidence_flags"]["meaningful_objective_progress_last100"]
        and not r["evidence_flags"]["trust_radius_at_or_below_historical_xtol"]
    ]
    tangent = [r["index"] for r in non if r["evidence_flags"]["tangential_direction_majority"]]
    by_step = []
    for t in range(8):
        by_step.append(
            dict(
                step=t + 1,
                **{
                    key: sum(r["descent_by_step"][t][key] for r in non)
                    for key in [
                        "predicted_decrease_per_angstrom",
                        "radial_contribution",
                        "tangential_contribution",
                        "active_contribution",
                        "interior_contribution",
                    ]
                },
            )
        )
    bins = []
    for bin_index in range(10):
        low = bin_index / 10
        points = [
            p for r in non for p in r["descent_by_residue"] if min(9, int(p["normalized_position"] * 10)) == bin_index
        ]
        bins.append(
            dict(
                low=float(low),
                high=float(low + 0.1),
                residues=len(points),
                predicted_decrease_per_angstrom=sum(p["predicted_decrease_per_angstrom"] for p in points),
                radial_contribution=sum(p["radial_contribution"] for p in points),
                tangential_contribution=sum(p["tangential_contribution"] for p in points),
            )
        )
    families = {name: [] for name in rows[0]["active_counts"]}
    solver_ball = []
    implied_centering = []
    saved_telemetry_inventory = []
    for r in rows:
        old = json.loads((runner.historical.OUT / "B" / f"example_{r['index']:02d}.json").read_text())["optimizer"]
        saved_telemetry_inventory.append(
            dict(
                index=r["index"],
                scalar_fields={k: v for k, v in old.items() if isinstance(v, (str, int, float, bool)) or v is None},
                structured_fields={
                    k: dict(available=True, source=f"V11/B/example_{r['index']:02d}.json optimizer.{k}")
                    for k, v in old.items()
                    if isinstance(v, (list, dict))
                },
            )
        )
        ids = r["physical_certificate"]["active_inequality_indices"]
        for idx in ids:
            if idx == 0:
                name = "aligned_RMSD"
            elif idx == 1:
                name = "continuous_chirality"
            else:
                name = next(
                    k
                    for k, (lo, hi) in r["physical_certificate"]["quartets"]["constraint_slices"].items()
                    if lo + 2 <= idx < hi + 2
                )
            families[name].append(old["multipliers"][idx])
        families["correction_balls"].extend(p["frozen_ball_multiplier"] for p in r["active_ball_rows"])
        n = len(r["descent_by_residue"])
        solver_ball.extend(
            old["ball_constraint_multipliers"][(p["step"] - 1) * n + p["residue_index"]] for p in r["active_ball_rows"]
        )
        implied_centering.append(r["barrier_geometry"]["implied_true_slack_centering"])
    result = dict(
        classification="NUM-A",
        classification_is_evidence_supported_not_a_causal_proof=True,
        reason=(
            "Late objective progress, noncollapsed radii, full-rank physical active sets "
            "and overwhelmingly radial interior descent support testing the iteration budget first."
        ),
        evidence_signatures=dict(
            denominator_nonconverged=56,
            iteration_progress_indices=progress,
            iteration_progress_count=len(progress),
            iteration_progress_percent=100 * len(progress) / 56,
            strong_tangential_barrier_pattern_indices=tangent,
            strong_tangential_barrier_pattern_count=len(tangent),
            strong_tangential_barrier_pattern_percent=100 * len(tangent) / 56,
            unresolved_indices=[r["index"] for r in non if r["index"] not in progress],
            note=(
                "These signatures do not causally allocate cases. Missing internal slack and multiplier "
                "histories prevent a definitive efficiency diagnosis."
            ),
        ),
        nonconverged_descent_by_step=by_step,
        nonconverged_position_bins=bins,
        active_multiplier_by_family={k: audit.distribution(v) for k, v in families.items()},
        solver_quadratic_ball_multiplier_active=audit.distribution(solver_ball),
        true_slack_centering_proxy_extrema={
            k: audit.distribution([r[k] for r in implied_centering]) for k in ["minimum", "maximum", "median"]
        },
        matched_2000_iteration_experiment_justified=True,
        different_solver_justified_as_primary_next_experiment=False,
        continuation_convergence_guaranteed=False,
        recommendation=(
            "Repeat V11 from zero with exactly 2000 iterations and otherwise identical settings; "
            "retain the exact frozen physical contract."
        ),
        historical_V11_classification="DIRECT-D",
        historical_endpoints_unchanged=True,
        nonlinear_optimizer_invocations=0,
        cuda_used=False,
        neural_training_launched=False,
    )
    publish_equal(runner.OUT / "telemetry_inventory.json", saved_telemetry_inventory)
    publish_equal(runner.OUT / "interpretation.json", result)
    runner.assert_file_pins(protocol["protected_sha256"])


if __name__ == "__main__":
    main()
