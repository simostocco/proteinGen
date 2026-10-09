"""Publish factual live/final E014 status; no conclusions before full exposure."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha


def summary_table(summary, dimension, fields):
    """Bounded scientific tables from persisted equal-protein summaries."""
    rows = [r for r in summary["groups"] if r["dimension"] == dimension]
    text = "| " + " | ".join([dimension, "protein conditions", *fields]) + " |\n"
    text += "|" + "---|" * (2 + len(fields)) + "\n"
    for row in rows:
        values = []
        for name in fields:
            value = row.get(name)
            values.append(f"{value:.6g}" if isinstance(value, (float, int)) else str(value))
        text += "| " + " | ".join([str(row["value"]), str(row["protein_condition_denominator"]), *values]) + " |\n"
    return text + "\n"


def boundary_curves(run, curves):
    text = "| epoch | updates | training v-MSE | development v-MSE | L1 | L2 | L3 | torsion MSE | t490 collapse |\n"
    text += "|---|---|---|---|---|---|---|---|---|\n"
    for curve in curves:
        epoch = curve["epoch"]
        if epoch == 0:
            label = "initialization_development"
        else:
            completed = int(epoch)
            label = (
                f"epoch_{completed}_fraction_1.00_development"
                if abs(epoch - completed) < 1e-12
                else (f"epoch_{completed + 1}_fraction_{epoch - completed:.2f}_development")
            )
        path = run / "evaluation" / f"{label}.summary.json"
        # Physical batches may cross the threshold by a few identities. The
        # immutable checkpoint name provides the actual boundary label.
        if not path.exists() and curve.get("checkpoint"):
            path = run / "evaluation" / (Path(curve["checkpoint"]).stem + "_development.summary.json")
        if not path.exists() and curve.get("resumed_pending_boundary"):
            path = run / "evaluation" / (curve["resumed_pending_boundary"] + "_development.summary.json")
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        allrow = next(r for r in summary["groups"] if r["dimension"] == "all")
        high = next(r for r in summary["groups"] if r["dimension"] == "timestep" and r["value"] == 490)
        values = [curve.get("training_v_mse"), curve["development_v_mse"]]
        values += [allrow.get(f"offset_{k}_mse_angstrom2_equal_protein_mean") for k in (1, 2, 3)]
        values += [allrow.get("torsion_mse_equal_protein_mean")]
        text += f"| {epoch:.6g} | {curve['updates']} | "
        text += " | ".join("—" if v is None else f"{v:.6g}" for v in values)
        text += f" | {high['collapse_count']}/{high['protein_condition_denominator']} |\n"
    return text + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    run = ROOT / "e014"
    heartbeat = json.loads((run / "HEARTBEAT.json").read_text()) if (run / "HEARTBEAT.json").exists() else {}
    population = json.loads((ROOT / "population/population.json").read_text())
    capacity = json.loads((ROOT / "capacity_selection.json").read_text())
    final = json.loads((run / "FINAL_REPORT.json").read_text()) if (run / "FINAL_REPORT.json").exists() else None
    if final and final["completed_epochs"] < 2:
        raise ValueError("final architectural adjudication forbidden before two complete epochs")
    if final:
        subprocess.run([sys.executable, "-B", "scripts/report_e014_effective_gates.py"], check=True)
    text = "\n## Execution status (automatically refreshed)\n\n"
    text += f"Updated unix timestamp: {time.time():.3f}. Phase: **{heartbeat.get('phase', 'not_started')}**.\n\n"
    text += f"TRAIN population: {population['train_count']}; sample-ID hash `{population['train_sample_id_hash']}`. "
    text += f"Selected model: {capacity['selected']['parameters']:,} parameters, {capacity['selected']['config']}.\n\n"
    text += (
        f"Optimizer updates: {heartbeat.get('update', 0)}; "
        f"protein exposures: {heartbeat.get('total_protein_exposures', 0)}; "
    )
    text += f"exposure min/max: {heartbeat.get('exposure_min', 0)}/{heartbeat.get('exposure_max', 0)}.\n\n"
    if final:
        text += f"Complete epochs: **{final['completed_epochs']}**; wall seconds: {final['wall_seconds']:.2f}; "
        text += f"GPU training seconds: {final['gpu_training_seconds']:.2f}; plateau: {final['plateau_triggered']}.\n\n"
        text += "Final predeclared decision:\n\n```json\n" + json.dumps(final["decision"], indent=2) + "\n```\n\n"
        text += "Training/development curves (local distances in Å²):\n\n"
        text += boundary_curves(run, final["curves"])
        text += "Final prospective noise-level metrics (counts retain denominators):\n\n"
        text += summary_table(
            final["final_prospective"],
            "timestep",
            [
                "v_mse_equal_protein_mean",
                "local_mse_angstrom2_equal_protein_mean",
                "torsion_mse_equal_protein_mean",
                "radius_ratio_equal_protein_mean",
                "collapse_count",
                "triple_inversion_count_sum",
                "triple_sign_assessed_sum",
            ],
        )
        text += "Final prospective length metrics:\n\n"
        text += summary_table(
            final["final_prospective"],
            "length",
            [
                "v_mse_equal_protein_mean",
                "offset_1_mse_angstrom2_equal_protein_mean",
                "offset_2_mse_angstrom2_equal_protein_mean",
                "offset_3_mse_angstrom2_equal_protein_mean",
                "torsion_mse_equal_protein_mean",
                "radius_ratio_equal_protein_mean",
                "collapse_count",
            ],
        )
        text += "Read-only global/local component telemetry by timestep:\n\n"
        text += summary_table(
            final["final_prospective"],
            "timestep",
            [
                "global_output_norm_equal_protein_mean",
                "local_effective_output_norm_equal_protein_mean",
                "local_global_norm_ratio_equal_protein_mean",
                "global_local_cosine_equal_protein_mean",
                "local_gate_mean_equal_protein_mean",
                "frame_reliability_mean_equal_protein_mean",
                "frame_assessable_count_sum",
                "frame_residue_count_sum",
            ],
        )
        text += (
            "Full curves and noise/length/geometry/head telemetry: `e014/FINAL_REPORT.json` "
            "and `e014/evaluation/*.summary.json`.\n"
        )
        text += "Exact combined-gate mean/median supplement: `e014/EFFECTIVE_GATE_TELEMETRY.json`. "
        text += (
            "It uses checkpoint time/gate weights and paired current noisy geometry; no training state is changed.\n"
        )
        next_experiment = {
            "A": "Bounded sampler-exposure diagnostic at matched timesteps; no full generation campaign yet.",
            "B": "Matched local-branch ablation on the same full-data coordinate-v contract.",
            "C": "Training-only capacity utilization and optimization audit before sampler work.",
            "D": "Separate low-noise geometry-refinement objective study, preserving the global prior.",
            "E": "High-noise/global-prior diagnostic with component telemetry before reverse trajectories.",
        }.get(final["decision"]["case"])
        text += f"\nRecommended next experiment: {next_experiment}\n"
    else:
        text += (
            "**The required minimum two complete TRAIN epochs has not yet completed. "
            "No architectural conclusion is available.** "
        )
        text += (
            "E013’s v-MSE/geometry divergence remains an open question. "
            "No sampler exposure or de novo sampling is authorized at this stage.\n"
        )
        curve_path = run / "CURVES.json"
        if curve_path.exists():
            text += "\nAvailable measured boundaries (early diagnostics only):\n\n"
            text += boundary_curves(run, json.loads(curve_path.read_text()))
    doc = Path("docs/e014_unconditional_coordinate_v_scaleup.md")
    existing = doc.read_text().split("\n## Execution status (automatically refreshed)")[0]
    doc.write_text((existing + text).rstrip() + "\n")
    status = dict(
        heartbeat=heartbeat,
        final_available=bool(final),
        final=final,
        architecture_conclusion_available=bool(final and final["completed_epochs"] >= 2),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        document_sha256=sha(doc),
        all_artifacts_on_d=True,
    )
    atomic_json(ROOT / "STATUS.json", status)
    if args.commit:
        if final:
            readme = Path("README.md")
            lines = readme.read_text().splitlines()
            for z, line in enumerate(lines):
                if line.startswith("| [E014]"):
                    lines[z] = (
                        "| [E014](docs/e014_unconditional_coordinate_v_scaleup.md) | "
                        "Full-scale E013 capacity/data/duration test? | "
                        f"{final['completed_epochs']} complete epochs, {final['optimizer_updates']} updates; "
                        f"predeclared case {final['decision'].get('case')}. | "
                        f"Stage D authorized: {final['decision']['stage_d_authorized']}; not run automatically. |"
                    )
            readme.write_text("\n".join(lines) + "\n")
            timeline = Path("docs/experiment_timeline.md")
            timeline.write_text(
                timeline.read_text() + f"\nE014 full-scale completion: {final['completed_epochs']} epochs, "
                f"{final['total_protein_exposures']} exposures; case {final['decision'].get('case')}. "
                "See the dedicated E014 record. Historical E013 remains a bounded pilot.\n"
            )
        subprocess.run(["git", "add", str(doc), "README.md", "docs/experiment_timeline.md"], check=True)
        subprocess.run(["git", "commit", "-m", "docs(e014): record full-scale execution status"], check=True)
        status["final_record_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status["worktree_status"] = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        atomic_json(ROOT / "STATUS.json", status)
        if final:
            frozen = json.loads(Path("/mnt/d/Simone/proteinGen/e013_artifacts/completion_manifest.json").read_text())
            preserved = all(sha(path) == record["sha256"] for path, record in frozen["artifact_hashes"].items())
            if not preserved:
                raise ValueError("historical E013 artifact preservation check failed")
            atomic_json(
                ROOT / "completion_manifest.json",
                dict(
                    branch=status["branch"],
                    final_record_commit=status["final_record_commit"],
                    training_source_commit=final["git_sha"],
                    config_hash=final["contract"]["config_hash"],
                    train_population_hash=final["contract"]["data_hash"],
                    normalization_hash=final["contract"]["normalization_hash"],
                    completed_epochs=final["completed_epochs"],
                    decision=final["decision"],
                    final_report_hash=sha(run / "FINAL_REPORT.json"),
                    document_hash=sha(doc),
                    all_generated_artifacts_on_d=True,
                    amino_acid_input=False,
                    target_derived_inference_features=False,
                    historical_e013_artifacts_preserved=preserved,
                    sequence_job_modified_or_stopped=False,
                    stage_d_run=False,
                    de_novo_sampling_run=False,
                    matched_e007_control_run=False,
                ),
            )
    print(text, flush=True)


if __name__ == "__main__":
    main()
