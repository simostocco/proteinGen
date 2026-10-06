"""Derive V9B classification and prospective sensitivity contract from frozen audit."""

import json
import math

from scripts.recover_e010_conditioning_v9b import OUT

if __name__ == "__main__":
    audit = json.loads((OUT / "audit.json").read_text())
    summary = []
    for c in audit["cases"]:
        m = len(c["physical_solver_multipliers"]["ball_multipliers"]) - 16  # eight steps, two fixed endpoints
        threshold = 1e-6 / (math.sqrt(m) * 1e-4 / 0.04)
        k = c["physical_solver_multipliers"]
        assert k["normalized_primal"] == 0
        assert k["normalized_dual"] <= 1e-8
        assert k["normalized_complementarity"] <= 1e-6
        assert k["physical_normalized_stationarity_l2"] <= threshold
        assert not any(s["material"] for s in c["shadow_steps"])
        assert all(s["feasible"] for s in c["shadow_steps"])
        assert all(r["passed"] for r in c["finite_difference_checks"])
        summary.append(
            dict(
                length=c["length"],
                eligible_vectors=m,
                stationarity_l2_sensitivity_threshold=threshold,
                physical_normalized_stationarity_l2=k["physical_normalized_stationarity_l2"],
                best_shadow_normalized_decrease=max(s["normalized_loss_decrease"] for s in c["shadow_steps"]),
            )
        )
    result = dict(
        classification="COND-A",
        meaning="V9 synthetic preflight is practically stationary under frozen physical sensitivity checks",
        cases=summary,
        meaningful_feasible_descent_found=False,
        future_contract=dict(
            status="proposal_only_not_applied_to_scientific_panel",
            primal=(
                "exact nonpositive V9 extra constraints, global normalized safety residual <=1e-8, "
                "exact radial bounds and discrete gates"
            ),
            stationarity=(
                "physical normalized L2 residual <= 1e-6/(sqrt(M)*1e-4/0.04), "
                "with derivative uncertainty tenfold smaller"
            ),
            dual_negativity_max=1e-8,
            normalized_complementarity_max=1e-6,
            shadow="no feasible normalized local-loss decrease >=1e-6 at any frozen shadow scale",
        ),
        limitation="Local numerical stationarity only, not global optimality or scientific-panel feasibility",
        recommended_next_action="Prepare V10 with a preregistered physically normalized KKT termination contract",
        scientific_panel_launched=False,
        cuda_used=False,
        neural_training_launched=False,
    )
    (OUT / "conclusion.json").write_text(json.dumps(result, indent=2) + "\n")
