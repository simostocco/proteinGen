"""Known-correct marginal-shift sensitivity control, never an E014 inference path.

True v is used ONLY by this isolated oracle control, as in the identity check.
No model/checkpoint is loaded, no model rollout is corrected, and no gate changes.
"""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import time
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import analyze_e014_marginal_reverse as marginal
from diagnose_e014_high_noise import OUT, population
from collect_e014_marginal_reverse import FAMILIES
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    MARGINAL_GRID,
    STRATA,
    marginal_status,
    resolution_front,
)
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha


def main():
    control = OUT / "oracle_sensitivity"
    control.mkdir(exist_ok=False)
    atomic_json(
        control / "CONTROL_POLICY.json",
        dict(
            created_unix=time.time(),
            purpose="calibrate finite-sample marginal-step test sensitivity using mathematically correct true-v transition",
            model_loaded=False,
            model_calls=0,
            oracle_targets_never_enter_model_inference_or_rollout=True,
            primary_front_rule_unchanged=True,
            reference_source_sha256=sha(OUT / "shared_reference_descriptors.parquet"),
            marginal_metric_source_sha256=sha(
                "src/protein_distance_diffusion/evaluation/marginal_distribution_diagnostic.py"
            ),
        ),
    )
    _, panel, _ = population()
    cache = CoordinateCache(ROOT)
    diffusion = CoordinateVPDiffusion(500)
    scale = 12.22820347644835
    rows = pq.read_table(OUT / "shared_reference_descriptors.parquet", use_threads=False).to_pylist()
    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(panel):
            clean, _, mask = coordinate_batch([row], cache, scale)
            for t in MARGINAL_GRID:
                ts = torch.tensor([t], device="cuda")
                g = torch.Generator(device="cuda").manual_seed(noise_seed(FAMILIES["source"], row["sample_id"], t))
                batch = diffusion.make_training_batch(clean, mask, timesteps=ts, generator=g)
                previous, _, _ = diffusion.deterministic_reverse_step(
                    batch.noisy_coordinates, ts, batch.coordinate_v_target, mask
                )
                rows.append(
                    dict(
                        sample_id=row["sample_id"],
                        length=row["length"],
                        length_stratum=row["length_stratum"],
                        timestep=t,
                        population="model",
                        descriptor=descriptor(previous[0].cpu().numpy()).tolist(),
                    )
                )
            if (i + 1) % 100 == 0:
                print(f"oracle sensitivity reference {i + 1}/500", flush=True)
    pq.write_table(pa.Table.from_pylist(rows), control / "oracle.descriptors.parquet", compression="zstd")
    marginal.OUT = control
    results = marginal.analyze_phase("one_step", "oracle.descriptors.parquet")
    fronts = []
    for group in results:
        group.pop("_model_draws", None)
        raw = group["raw_descriptor_distributions"]
        rg = raw["model"]["rg"]["mean"] / raw["ref_b"]["rg"]["mean"]
        trace = raw["model"]["trace"]["mean"] / raw["ref_b"]["trace"]["mean"]
        group["resolution_status"] = marginal_status(group["metrics"], rg, trace)
        group["population"] = "known-correct oracle marginal; not E014 model"
    for s in (*STRATA, "pooled"):
        fronts.append(dict(length_stratum=s, **resolution_front([r for r in results if r["length_stratum"] == s])))
    atomic_json(
        control / "metrics.json",
        dict(
            groups=results,
            resolution_fronts=fronts,
            elapsed_seconds=time.time() - start,
            model_calls=0,
            not_an_inference_correction=True,
            no_gate_change=True,
        ),
    )
    print("oracle marginal-shift sensitivity certified; no E014 model called", flush=True)


if __name__ == "__main__":
    main()
