"""Bounded scalar/footer validation; no model or checkpoint load."""

import hashlib
import json
import os
from pathlib import Path


def main(out):
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.nice(19)
    import numpy as np
    import pyarrow.parquet as pq

    out = Path(out)
    required = [
        "REPORT.md",
        "INFERENCE_GUIDE.md",
        "MANIFEST.json",
        "verdict.json",
        "checkpoint_selection.json",
        "checkpoint_inventory.json",
        "panel_definitions.json",
        "generalization_bootstrap.json",
        "generation_distribution_metrics.json",
        "diversity_novelty_metrics.json",
        "resource_monitor.jsonl",
        "recovery_resume_proof.json",
    ]
    assert all((out / name).is_file() for name in required)
    counts = {p.name: pq.ParquetFile(p).metadata.num_rows for p in out.glob("*.parquet")}
    assert counts["absolute_reconstruction_metrics.parquet"] == 10000
    assert counts["generation_manifest.parquet"] == counts["generation_metrics.parquet"] == 25
    assert counts["trajectory_metrics.parquet"] == 325
    assert pq.ParquetFile(out / "numerical/heldout500_metrics.parquet").metadata.num_rows == 5000
    typed = pq.read_table(out / "typed_state_statistics.parquet", use_threads=False).to_pylist()
    stages = sorted({q["stage"] for q in typed})
    assert len(stages) == 19
    assert min(q["finite_fraction"] for q in typed) == 1
    assert all(q["vector_norm_mean"] is not None for q in typed if q["type"] in ("V", "A"))
    generation = pq.read_table(out / "generation_metrics.parquet", use_threads=False).to_pylist()
    assert all(q["finite_fraction"] == 1 for q in generation)
    manifest = pq.read_table(out / "generation_manifest.parquet", use_threads=False).to_pylist()
    expected_frames = {f"t{t}" for t in (499, 475, 450, 425, 400, 350, 300, 250, 200, 150, 100, 50, 0)}
    for sample in manifest:
        folder = Path(sample["path"])
        with np.load(folder / "sparse_trajectory.npz") as latest, np.load(sample["initial_gaussian_source"]) as old:
            assert set(latest.files) == expected_frames
            assert np.array_equal(latest["t499"], old["t499"])
            assert hashlib.sha256(old["t499"].tobytes()).hexdigest() == sample["initial_gaussian_sha256"]
            assert all(np.isfinite(latest[name]).all() for name in latest.files)
    bootstrap = json.loads((out / "generalization_bootstrap.json").read_text())
    assert bootstrap["draws"] >= 5000 and all(q["draws"] >= 5000 for q in bootstrap["rows"])
    heads = pq.read_table(out / "head_contributions.parquet", use_threads=False).to_pylist()
    ablations = [
        q for q in heads if q["checkpoint"] == "latest_u10382" and q["weights"] == "raw" and q["split"] == "heldout"
    ]
    assert len(ablations) == 500
    ablation_summary = {}
    for t in (5, 50, 250, 400, 490):
        rows = [q for q in ablations if q["timestep"] == t]
        ablation_summary[t] = {
            metric: sum(q["chiral_only_disabled"][metric] - q["enabled"][metric] for q in rows) / len(rows)
            for metric in (
                "v_mse",
                "local_rmse_angstrom",
                "torsion_angular_mae_radians",
                "triple_inversion_fraction",
                "clash_fraction",
            )
        }
    gif_sizes = {p.name: p.stat().st_size for p in (out / "trajectory_gifs").glob("*.gif")}
    assert len(gif_sizes) == 3 and max(gif_sizes.values()) <= 25 * 1024**2
    assert len(list((out / "samples").glob("*/*/quality_card.png"))) == 25
    assert len(list((out / "same_seed_comparison").glob("*_final.png"))) == 20
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    assert total <= 6 * 1024**3
    result = dict(
        status="PASSED",
        counts=counts,
        typed_stages=stages,
        minimum_typed_finite_fraction=1,
        missing_vector_norms=0,
        generation_finite_fraction=1,
        exact_saved_initial_states_verified=len(manifest),
        gif_bytes=gif_sizes,
        total_bytes=total,
        chiral_disabled_minus_enabled=ablation_summary,
    )
    with (out / "logs/final_artifact_validation.json").open("w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
