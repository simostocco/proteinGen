"""Publish SLSQP numerical results or the preregistered resource stop."""

import json

import numpy as np
import torch

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_slsqp_v14 as runner
from scripts.report_e010_direct_correction_v11 import span
from scripts.report_e010_local_feasibility_v7 import percent, summarize
from scripts.report_e010_local_feasibility_v8 import directions
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write
from scripts.summarize_e010_strict_scientific_v10 import group_summary


def classify(count, material, gain, safe, resource_ok=True, integrity_ok=True):
    if not integrity_ok:
        return "SQP-F"
    if not resource_ok:
        return "SQP-E"
    if count == 60 and safe:
        return "SQP-A" if gain >= 5 else "SQP-D"
    return "SQP-B" if count >= 30 and material <= 25 else "SQP-C"


def reproduce_preflight(pre):
    evidence = []
    for row in pre["records"]:
        n = row["length"]
        b = runner.synthetic(n)
        path = runner.OUT / "untracked_states" / f"preflight_{n}_certified.npz"
        assert file_hash(path) == row["state_file_sha256"]
        with np.load(path) as saved:
            z = saved["z"].copy()
            tr = runner.sqp.direct.trajectory(b["pg"], b["mask"], torch.from_numpy(z).reshape(8, *b["pg"].shape))
            delta = torch.stack([s["delta"].detach() for s in tr["steps"]])
            assert np.array_equal(torch.stack(tr["states"]).detach().numpy(), saved["states"])
            assert np.array_equal(delta.numpy(), saved["delta"])
            mu, rec = runner.sqp.reconstruct(b, delta, runner.CONTRACT["physical_config"])
            assert np.array_equal(mu, saved["multipliers"])
            cert = runner.sqp.direct.certificate(
                b, delta, mu, runner.CONTRACT["physical_config"], runner.CONTRACT["physical_limits"]
            )
        assert json.loads(json.dumps(cert)) == row["optimizer"]["physical_certificate"]
        assert json.loads(json.dumps(rec)) == row["optimizer"]["multiplier_reconstruction"]
        evidence.append(dict(length=n, state_and_certificate_exact=True, optimizer_invocations=0))
    return evidence


def main():
    runner.frozen()
    out = runner.OUT
    pre = json.loads((out / "preflight_complete.json").read_text())
    proof = reproduce_preflight(pre)
    reference = json.loads((runner.historical.OUT.parent / "matched_direct_space_v13" / "result.json").read_text())
    base = dict(
        solver="scipy.optimize.minimize(method=SLSQP)",
        settings=runner.sqp.SETTINGS,
        runtime=runner.CONTRACT["runtime"],
        preflight=pre,
        preflight_reproduction=proof,
        reference_v13=dict(
            physical_convergence=6,
            material_shadow_count=51,
            conditions={k: v["local_gain_pct"] for k, v in reference["conditions"].items()},
        ),
        cuda_used=False,
        neural_training_launched=False,
        follow_on_launched=False,
        objective_constraints_and_physical_contract_unchanged=True,
        historical_hessian_vector_products_used=False,
    )
    if not pre["panel_permitted"]:
        assert not list((out / "attempts").glob("*.json"))
        result = dict(
            **base,
            classification="SQP-E",
            scientific_panel_launched=False,
            panel_solver_invocations=0,
            physical_convergence=None,
            material_shadow_count=None,
            conditions=None,
            certified_sufficient=False,
            no_scientific_interpretation=True,
        )
        body = [
            "# E010 V14 SLSQP benchmark",
            "",
            "**SQP-E — solver resource/scaling failure.**",
            "The scientific panel was not launched. No scientific feasibility interpretation is made.",
            "",
            "SLSQP uses dense QP storage. The frozen length-500 resource preflight requires "
            f"{pre['resource_smoke']['required_allocations']['required_array_bytes'] / 2**30:.6f} GiB "
            "for its installed mandatory workspace and constraint normals alone. "
            "The preregistered allocation budget is "
            f"{pre['resource_smoke']['solver_array_budget_bytes'] / 2**30:.6f} GiB, "
            "after reserving RAM for the framework/OS. The resource gate refused the dense allocation.",
            "Full length-500 geometry, constraints and derivative plumbing were validated "
            "without reducing K or length.",
            "",
        ]
    else:
        complete = json.loads((out / "reproduce_complete.json").read_text())
        assert len(complete["results"]) == 60 and all(r["reproduced"] for r in complete["results"])
        rows = [json.loads((out / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
        baseline, final = [summarize([r[key] for r in rows]) for key in ["baseline", "oracle"]]
        count = sum(r["optimizer"]["converged"] for r in rows)
        material = [
            r["index"]
            for r in rows
            if any(s["material"] for s in r["optimizer"]["physical_certificate"]["shadow_steps"])
        ]
        failures = [
            r["index"]
            for r in rows
            if not r["optimizer"]["physical_certificate"]["gates"]["primal_feasibility"]
            or r["quartet_states"][-1]["new_inversions"]
            or r["quartet_states"][-1]["assessability_lost"]
            or not all(s["finite"] and not s["collapse"] and s["frame_assessability_preserved"] for s in r["states"])
        ]
        conditions = {}
        for c in ["50", "250", "450"]:
            b, f = baseline["by_condition"][c], final["by_condition"][c]
            group = [r for r in rows if str(r["record"]["condition"]) == c]
            conditions[c] = dict(
                local_gain_pct=-percent(f, b, "mean_local_rmse"),
                offset_gains_pct={k: 100 * (1 - f["local_rmse"][k] / b["local_rmse"][k]) for k in ["1", "2", "3"]},
                aligned_change_pct=percent(f, b, "aligned_rmsd"),
                chiral_change_pct=percent(f, b, "continuous_chiral_loss"),
                raw_cartesian_change_pct=percent(f, b, "raw_cartesian"),
                **group_summary(group),
            )
        label = classify(count, len(material), conditions["450"]["local_gain_pct"], not failures)
        result = dict(
            **base,
            classification=label,
            scientific_panel_launched=True,
            panel_solver_invocations=60,
            physical_convergence=count,
            material_shadow_count=len(material),
            material_shadow_examples=material,
            baseline=baseline,
            final=final,
            conditions=conditions,
            safety_failures=failures,
            numerical={
                k: span([r["optimizer"][k] for r in rows])
                for k in ["iterations", "runtime_seconds", "process_peak_rss_bytes"]
            },
            overall=group_summary(rows),
            certified_sufficient=label == "SQP-A",
            trajectories=[summarize([r["states"][t] for r in rows]) for t in range(9)],
            correction_directions=directions(rows),
        )
        body = [
            "# E010 V14 SLSQP benchmark",
            "",
            f"**{label}**; physical convergence {count}/60, material-shadow {len(material)}/60.",
            "",
        ]
    body += [
        "| Preflight length | Physical convergence | SLSQP success | Iterations | Wall seconds | Peak RSS GiB |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in pre["records"]:
        opt = row["optimizer"]
        body.append(
            f"| {row['length']} | {opt['converged']} | {opt['scipy_success']} | {opt['iterations']} | "
            f"{opt['runtime_seconds']:.3f} | {opt['process_peak_rss_bytes'] / 2**30:.3f} |"
        )
    body += [
        "",
        "Fixed SLSQP configuration: CPU float64, one arithmetic thread, maxiter=2000, ftol=1e-12, "
        "analytic objective and vector constraint Jacobians. No additional constraint scaling. "
        "SLSQP does not use the historical exact Hessian-vector products. Its success flag is telemetry only.",
        "All protected hashes and independently reconstructed preflight certificates passed. "
        "Historical V11/V12/V13 remain unchanged. No CUDA, neural training or follow-on solver was launched.",
    ]
    assert_file_pins(runner.CONTRACT["protected_sha256"])
    assert_file_pins(runner.CONTRACT["protected_input_sha256"])
    write(out / "result.json", result)
    (out / "RESULTS.md").write_text("\n".join(body) + "\n")
    write(
        out / "publication_verification.json",
        dict(
            protected_hashes_verified=True,
            fixed_preflight_reproduction_exact=True,
            verification_optimizer_invocations=0,
            classification=result["classification"],
            publication_sha256={k: file_hash(out / k) for k in ["result.json", "RESULTS.md"]},
        ),
    )
    print(result["classification"], "panel launched", result["scientific_panel_launched"], flush=True)


if __name__ == "__main__":
    main()
