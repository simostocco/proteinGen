"""Aggregate fixed-state numerical evidence without invoking any optimizer."""

import json

import numpy as np

from protein_distance_diffusion.training import e010_fixed_state_audit_v12 as audit
from scripts import audit_e010_fixed_state_v12 as runner
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def get(row, path):
    for part in path.split("/"):
        row = row[part]
    return row


def aggregate(rows):
    paths = {
        "stationarity": "physical_certificate/physical/physical_normalized_stationarity_l2",
        "complementarity": "physical_certificate/physical/normalized_complementarity",
        "dual_negativity": "physical_certificate/physical/normalized_dual",
        "projected_feasible_gradient": "tangent_QP/projected_feasible_gradient_norm",
        "trust_radius": "telemetry/trust_radius/value",
        "barrier_parameter": "telemetry/barrier_parameter/value",
        "raw_optimality": "telemetry/optimality/value",
        "iterations": "telemetry/iterations/value",
        "active_Jacobian_condition": "active_geometry/frozen_normalized/condition",
        "row_unit_Jacobian_condition": "active_geometry/row_unit_diagnostic/condition",
        "active_Jacobian_rank": "active_geometry/frozen_normalized/rank",
        "largest_singular_value": "active_geometry/frozen_normalized/largest",
        "smallest_nonzero_singular_value": "active_geometry/frozen_normalized/smallest_nonzero",
        "outward_radial": "gradient_decomposition/active_balls/frozen_normalized/outward_descent_radial_l2",
        "inward_radial": "gradient_decomposition/active_balls/frozen_normalized/inward_descent_radial_l2",
        "tangential": "gradient_decomposition/active_balls/frozen_normalized/tangential_l2",
        "eligible_outward_radial": "gradient_decomposition/all_eligible/frozen_normalized/outward_descent_radial_l2",
        "eligible_tangential": "gradient_decomposition/all_eligible/frozen_normalized/tangential_l2",
        "tangential_direction_power_fraction": "evidence_flags/tangential_direction_power_fraction",
        "saturation_fraction_99pct": "boundary_geometry/fraction_at_least_99pct",
        "frozen_active_fraction": "boundary_geometry/exact_frozen_active_fraction",
        "QP_linearized_violation": "tangent_QP/maximum_linearized_violation",
    }
    stats = {
        name: audit.distribution([get(r, path) for r in rows if get(r, path) is not None])
        for name, path in paths.items()
    }
    families = sorted({k for r in rows for k in r["active_counts"]})
    flags = [k for k, v in rows[0]["evidence_flags"].items() if isinstance(v, bool)] if rows else []
    windows = {}
    for w in ["25", "50", "100", "250"]:
        windows[w] = {}
        for name in [
            "normalized_objective",
            "physical_stationarity",
            "complementarity",
            "trust_radius",
            "barrier_parameter",
            "raw_optimality",
        ]:
            values = [r["history_windows"][w][name] for r in rows if r["history_windows"][w][name]["available"]]
            windows[w][name] = {
                k: audit.distribution([v[k] for v in values if v[k] is not None])
                for k in ["change", "relative_change", "slope_per_iteration"]
            }
    shadow = []
    for eps in [1e-6, 1e-5, 1e-4]:
        values = [s for r in rows for s in r["shadow_steps"] if s["step_angstrom"] == eps]
        shadow.append(
            dict(
                scale_angstrom=eps,
                feasible=sum(v["feasible"] for v in values),
                material=sum(v["material"] for v in values),
                normalized_decrease=audit.distribution([v["normalized_loss_decrease"] for v in values]),
                feasible_decrease=audit.distribution([v["normalized_loss_decrease"] for v in values if v["feasible"]]),
            )
        )
    return dict(
        count=len(rows),
        group_counts={g: sum(r["group"] == g for r in rows) for g in "ABC"},
        material_shadow_count=sum(r["material_feasible_descent"] for r in rows),
        statistics=stats,
        active_counts={k: sum(r["active_counts"].get(k, 0) for r in rows) for k in families},
        near_parallel_pairs=sum(len(r["active_geometry"]["frozen_normalized"]["nearly_parallel_pairs"]) for r in rows),
        rank_deficient_cases=sum(
            r["active_geometry"]["frozen_normalized"]["rank"] < r["active_geometry"]["frozen_normalized"]["rows"]
            for r in rows
        ),
        evidence_counts={k: sum(r["evidence_flags"][k] for r in rows) for k in flags},
        history_windows=windows,
        shadows=shadow,
        step_descent=[
            dict(
                step=t + 1,
                **{
                    k: audit.distribution([r["descent_by_step"][t][k] for r in rows])
                    for k in [
                        "predicted_decrease_per_angstrom",
                        "active_contribution",
                        "interior_contribution",
                        "radial_contribution",
                        "tangential_contribution",
                    ]
                },
            )
            for t in range(8)
        ],
        ball_multiplier=audit.distribution([v["frozen_ball_multiplier"] for r in rows for v in r["active_ball_rows"]]),
        solver_ball_multiplier_extrema=audit.distribution(
            [r["barrier_geometry"]["eligible_solver_ball_multiplier"]["maximum"] for r in rows]
        ),
        barrier_terminal_ratio=audit.distribution(
            [r["barrier_geometry"]["barrier_to_configured_terminal_ratio"] for r in rows]
        ),
    )


def historical_comparison():
    root = runner.historical.OUT.parent / "strict_constraint_conditioning_v9b"
    cases = json.loads((root / "audit.json").read_text())["cases"]
    evidence = json.loads((root / "recovery_evidence_length_32.json").read_text())
    evidence.setdefault("physical_screens", [])
    windows = audit.history_windows(evidence, [25, 50, 100, 250])
    recovered = json.loads((root / "recovery.json").read_text())
    for r in recovered:
        path = root / f"recovered_length_{r['length']}.npz"
        assert runner.file_hash(path) == r["state_file_sha256"]
        with np.load(path) as saved:
            for key, sha in r["array_sha256"].items():
                assert runner.hashlib.sha256(saved[key].tobytes()).hexdigest() == sha
    return dict(
        fixed_historical_cases=cases,
        length32_iterations=evidence["iterations"],
        length32_history_windows=windows,
        final_metrics_and_hashes_loaded_without_replay=True,
        physical_stationarity_history_available=False,
        multiplier_history_available=False,
        caution=(
            "Historical radial length-32 convergence is descriptive evidence, not a convergence-iteration prediction."
        ),
    )


def main():
    runner.setup()
    contract = json.loads((runner.OUT / "preregistration.json").read_text())
    runner.assert_file_pins(contract["protected_sha256"])
    rows = [json.loads((runner.OUT / "cases" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    result = dict(
        overall=aggregate(rows),
        groups={g: aggregate([r for r in rows if r["group"] == g]) for g in "ABC"},
        conditions={str(c): aggregate([r for r in rows if r["record"]["condition"] == c]) for c in [50, 250, 450]},
        lengths={
            s: aggregate([r for r in rows if r["record"]["stratum"] == s])
            for s in ["20-64", "65-128", "129-256", "257-384", "385-500"]
        },
        historical_length32=historical_comparison(),
        unavailable=[
            "internal slack variables",
            "constraint penalty",
            "per-iteration multipliers",
            "actual accepted step norms",
            "historical radial physical KKT iteration history",
        ],
        nonlinear_optimizer_invocations=0,
        cuda_used=False,
        neural_training_launched=False,
    )
    write(runner.OUT / "audit_aggregate.json", result)
    runner.assert_file_pins(contract["protected_sha256"])


if __name__ == "__main__":
    main()
