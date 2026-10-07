"""Read-only publication of V15 preflight and once-only panel records."""

import json
import sys
from collections import Counter
from statistics import median

from scripts import run_e010_sparse_active_v15 as run
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def span(values):
    return dict(minimum=min(values), median=median(values), maximum=max(values))


def verify_metrics():
    run.frozen()
    import numpy as np

    results = []
    for i in range(60):
        expected = json.loads((run.OUT / "B" / f"example_{i:02d}.json").read_text())
        original = json.loads((run.OUT / f"panel_{i:02d}.json").read_text())
        path = run.OUT / "untracked_states" / f"panel_{i:02d}_certified.npz"
        assert run.file_hash(path) == original["state_file_sha256"]
        with np.load(path) as state:
            x = state["z"].copy()
            saved_states = state["states"].copy()
        tr = run.solver.direct.Oracle(run.historical.data(i)).point(x).tr
        assert np.array_equal(saved_states, run.torch.stack([p.detach() for p in tr["states"]]).numpy())
        rebuilt = run.historical.build(i, tr, original["optimizer"], x)
        assert json.loads(json.dumps(rebuilt)) == expected
        assert expected["optimizer"]["physical_certificate"] == original["optimizer"]["physical_certificate"]
        results.append(
            dict(
                index=i,
                all_metrics_exact=True,
                coordinate_hashes_exact=True,
                certificate_independently_reconstructed_in_execution_driver=True,
                convergence_classification_exact=True,
                protected_hashes_verified=True,
            )
        )
    write(run.OUT / "reproduction_complete.json", dict(results=results, optimizer_invocations=0))


def main():
    c = run.frozen()
    conclusion = json.loads((run.OUT / "conclusion.json").read_text())
    preflights = [json.loads((run.OUT / f"preflight_{n}.json").read_text()) for n in [12, 32, 500]]
    result = dict(
        conclusion=conclusion,
        preflight_summary={},
        historical_V13=dict(
            physical_convergence=6,
            material_shadow=51,
            condition_gains_pct={"50": 19.1694, "250": 8.3859, "450": 8.5007},
        ),
        solver_config=c["config"],
        scientific_panel=conclusion["full_panel_launched"],
    )
    for r in preflights:
        o = r["optimizer"]
        qp = o["qp_history"]
        result["preflight_summary"][r["name"]] = dict(
            converged=o["converged"],
            iterations=o["iterations"],
            termination=o["termination"],
            accepted_iterates=o["accepted_iterates"],
            objective_decreased=r["objective_decreased"],
            primal_feasible=r["exact_primal_feasible"],
            total_QP_solves=o["total_QP_solves"],
            QP_failures=o["QP_failures"],
            line_search_backtracks=o["line_search_backtracks"],
            runtime_seconds=o["runtime_seconds"],
            peak_RSS_GiB=o["process_peak_rss_bytes"] / 1024**3,
            maximum_inward_guard_angstrom=o["maximum_inward_guard_angstrom"],
            variables=o["variable_dimension"],
            constraints=o["total_constraint_dimension"],
            jacobian_nnz=o["Jacobian_nnz"],
            QP_nnz_max=max(q.get("jacobian_nnz", 0) + o["QP_curvature_nnz"] for q in qp),
            sparse_working_bytes_max=max(q.get("estimated_sparse_working_bytes", 0) for q in qp),
            active_set_final=qp[-1]["active_families"],
            physical=o["physical_certificate"]["physical"],
            gates=o["physical_certificate"]["gates"],
            material_shadow=sum(s["material"] for s in o["physical_certificate"]["shadow_steps"]),
            projected_feasible_gradient=o["multiplier_reconstruction"]["projected_feasible_gradient_norm"],
            new_inversions=r["quartet_telemetry"]["new_inversions"],
            assessability_lost=r["quartet_telemetry"]["assessability_lost"],
        )
    if conclusion["full_panel_launched"]:
        rows = [json.loads((run.OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
        baseline, final = (summarize([r[k] for r in rows]) for k in ["baseline", "oracle"])
        result.update(baseline=baseline, final=final, conditions={}, length_strata={})
        for condition in ["50", "250", "450"]:
            b, f = baseline["by_condition"][condition], final["by_condition"][condition]
            group = [r for r in rows if str(r["record"]["condition"]) == condition]
            result["conditions"][condition] = dict(
                local_gain_pct=-percent(f, b, "mean_local_rmse"),
                offset_gains_pct={k: 100 * (1 - f["local_rmse"][k] / b["local_rmse"][k]) for k in ["1", "2", "3"]},
                raw_cartesian_change_pct=percent(f, b, "raw_cartesian"),
                aligned_change_pct=percent(f, b, "aligned_rmsd"),
                chiral_change_pct=percent(f, b, "continuous_chiral_loss"),
                baseline_inversions=b["chirality_inversions"],
                final_inversions=f["chirality_inversions"],
                new_inversions=sum(r["quartet_states"][-1]["new_inversions"] for r in group),
                repaired_inversions=sum(r["quartet_states"][-1]["repaired_inversions"] for r in group),
            )
        for stratum, b in baseline["by_stratum"].items():
            f = final["by_stratum"][stratum]
            result["length_strata"][stratum] = dict(
                local_gain_pct=-percent(f, b, "mean_local_rmse"), aligned_change_pct=percent(f, b, "aligned_rmsd")
            )
        logs = [r["optimizer"] for r in rows]
        result["numerical"] = {
            key: span([o["physical_certificate"]["physical"][key] for o in logs])
            for key in ["physical_normalized_stationarity_l2", "normalized_complementarity", "normalized_dual"]
        }
        result["resources"] = dict(
            iterations=span([o["iterations"] for o in logs]),
            total_runtime_seconds=sum(o["runtime_seconds"] for o in logs),
            peak_RSS_GiB=max(o["process_peak_rss_bytes"] for o in logs) / 1024**3,
            total_QP_solves=sum(o["total_QP_solves"] for o in logs),
            QP_failures=sum(o["QP_failures"] for o in logs),
            line_search_backtracks=sum(o["line_search_backtracks"] for o in logs),
            maximum_inward_guard_angstrom=max(o["maximum_inward_guard_angstrom"] for o in logs),
            jacobian_nnz=span([o["Jacobian_nnz"] for o in logs]),
            projected_feasible_gradient=span(
                [o["multiplier_reconstruction"]["projected_feasible_gradient_norm"] for o in logs]
            ),
        )
        result["trajectories"] = [summarize([r["states"][t] for r in rows]) for t in range(9)]
        from scripts.report_e010_local_feasibility_v8 import directions

        result["correction_directions"] = directions(rows)
        result["termination_counts"] = dict(Counter(o["termination"] for o in logs))
        shadows = [s for o in logs for s in o["physical_certificate"]["shadow_steps"]]
        result["frozen_shadow_trial_telemetry"] = dict(
            total=len(shadows),
            feasible=sum(s["feasible"] for s in shadows),
            infeasible=sum(not s["feasible"] for s in shadows),
            material=sum(s["material"] for s in shadows),
            infeasible_with_normalized_decrease_above_materiality=sum(
                not s["feasible"]
                and s["normalized_loss_decrease"] >= c["physical_config"]["material_normalized_local_decrease"]
                for s in shadows
            ),
            protocol_unchanged=True,
            interpretation="An infeasible shadow is not evidence that feasible descent is absent.",
        )
        result["active_set_statistics"] = {
            family: span([o["qp_history"][-1]["active_families"][family] for o in logs])
            for family in logs[0]["qp_history"][-1]["active_families"]
        }
        result["QP_diagnostics"] = dict(
            nnz=span([q.get("jacobian_nnz", 0) + o["QP_curvature_nnz"] for o in logs for q in o["qp_history"]]),
            estimated_sparse_working_bytes=span(
                [q.get("estimated_sparse_working_bytes", 0) for o in logs for q in o["qp_history"]]
            ),
            rejected_final_QP_residuals=[
                dict(index=r["index"], last_QP=r["optimizer"]["qp_history"][-1])
                for r in rows
                if r["optimizer"]["termination"] == "local_QP_failure"
            ],
        )
        result["state_serialization_certificate_intervals_seconds"] = span(
            [
                (run.OUT / "untracked_states" / f"panel_{i:02d}_certified.npz").stat().st_mtime
                - (run.OUT / "untracked_states" / f"panel_{i:02d}.npz").stat().st_mtime
                for i in range(60)
            ]
        )
        result["runtime_note"] = (
            "Solver runtime includes periodic certificate checks but excludes final certification. "
            "Serialization intervals describe final certificate overhead using filesystem timestamps. "
            "Peak RSS is cumulative process high water, including preflight reproduction, not isolated per-example RSS."
        )
        compact = []
        for row in rows:
            item = dict(row)
            item["optimizer"] = {
                k: v
                for k, v in row["optimizer"].items()
                if k
                not in (
                    "history",
                    "qp_history",
                    "physical_screens",
                    "multipliers",
                    "constraints",
                    "ball_constraint_values",
                    "multiplier_reconstruction",
                )
            }
            item["raw_record_sha256"] = run.file_hash(run.OUT / f"panel_{row['index']:02d}.json")
            item["state_file_sha256"] = run.file_hash(
                run.OUT / "untracked_states" / f"panel_{row['index']:02d}_certified.npz"
            )
            compact.append(item)
        write(run.OUT / "per_example_records.json", compact)
    write(run.OUT / "result.json", result)


if __name__ == "__main__":
    run.torch.set_num_threads(1)
    verify_metrics() if sys.argv[1:] == ["verify"] else main()
