"""Publish factual live/final E014 status; no conclusions before full exposure."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import argparse
import json
import subprocess
import time
from pathlib import Path
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha


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
        text += (
            "Full curves and noise/length/geometry/head telemetry: `e014/FINAL_REPORT.json` "
            "and `e014/evaluation/*.summary.json`.\n"
        )
    else:
        text += (
            "**The required minimum two complete TRAIN epochs has not yet completed. "
            "No architectural conclusion is available.** "
        )
        text += (
            "E013’s v-MSE/geometry divergence remains an open question. "
            "No sampler exposure or de novo sampling is authorized at this stage.\n"
        )
    doc = Path("docs/e014_unconditional_coordinate_v_scaleup.md")
    existing = doc.read_text().split("\n## Execution status (automatically refreshed)")[0]
    doc.write_text((existing + text).rstrip() + "\n")
    atomic_json(
        ROOT / "STATUS.json",
        dict(
            heartbeat=heartbeat,
            final_available=bool(final),
            final=final,
            architecture_conclusion_available=bool(final and final["completed_epochs"] >= 2),
            branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            document_sha256=sha(doc),
            all_artifacts_on_d=True,
        ),
    )
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
    print(text, flush=True)


if __name__ == "__main__":
    main()
