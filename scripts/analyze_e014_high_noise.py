"""CPU-only frozen high-noise curves/fronts; no training or sampler mutation."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import argparse
import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import analyze_e014_marginal_reverse as marginal
from diagnose_e014_high_noise import OUT
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    GRID,
    HIGH,
    LOW_MID,
    LABELS,
    STRATA,
    collapse_onset,
    marginal_status,
    noisy_scale_status,
    resolution_front,
)
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import correlation, distribution
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.evaluation.high_noise_resolution import schedule_values
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha

marginal.OUT = OUT
assert OUT.is_relative_to(ROOT)


def paired_summary(rows):
    result = dict(
        identity_count=len(rows),
        finite_count=sum(r["finite"] for r in rows),
        direct_collapse_count=sum(r["collapsed"] for r in rows),
        direct_collapse_fraction=sum(r["collapsed"] for r in rows) / len(rows),
    )
    for field in rows[0]:
        vals = [r[field] for r in rows if isinstance(r[field], (float, int)) and r[field] is not None]
        if field not in ["length", "timestep", "finite", "collapsed"] and vals:
            result[field] = distribution(vals)
            if field.endswith("cosine"):
                result[field + "_positive_fraction"] = dict(
                    count=sum(v > 0 for v in vals), denominator=len(vals), fraction=sum(v > 0 for v in vals) / len(vals)
                )
    result.update(
        direct_rg_median=result["radius_ratio"]["median"],
        direct_rg_mean=result["radius_ratio"]["mean"],
        step_gain_of_means=1 - result["model_mse"]["mean"] / result["noop_mse"]["mean"],
    )
    for field in [
        "torsion_assessed",
        "torsion_potential_count",
        "triple_sign_assessed",
        "triple_inversion_count",
        "frame_assessable_count",
        "frame_residue_count",
    ]:
        result[field + "_sum"] = sum(r[field] for r in rows)
    result["triple_inversion_rate_pooled"] = (
        result["triple_inversion_count_sum"] / result["triple_sign_assessed_sum"]
        if result["triple_sign_assessed_sum"]
        else None
    )
    result["frame_assessability_pooled"] = result["frame_assessable_count_sum"] / result["frame_residue_count_sum"]
    return result


def load_paired(label):
    return pq.read_table(OUT / f"{label}.paired.parquet", use_threads=False).to_pylist()


def analyze_checkpoint(label):
    assert (OUT / f"{label}.INFERENCE_COMPLETE.json").exists()
    start = time.time()
    target = OUT / f"{label}.ANALYSIS_COMPLETE.json"
    if target.exists():
        return json.loads(target.read_text())
    rows = load_paired(label)
    assert len(rows) == 500 * len(GRID)
    summaries = []
    cors = []
    for t in GRID:
        for stratum in (*STRATA, "pooled"):
            chosen = [r for r in rows if r["timestep"] == t and (stratum == "pooled" or r["length_stratum"] == stratum)]
            assert len(chosen) == (500 if stratum == "pooled" else 100)
            summary = paired_summary(chosen)
            summary.update(
                checkpoint=label, timestep=t, length_stratum=stratum, **schedule_values(CoordinateVPDiffusion(500), t)
            )
            # Paired MSE gain uncertainty, independent from marginal-front rule.
            seed = noise_seed("e014-high-noise-paired-bootstrap-v1", stratum, t)
            rng = np.random.default_rng(seed)
            model = np.array([r["model_mse"] for r in chosen])
            noop = np.array([r["noop_mse"] for r in chosen])
            sizes = [100] * 5 if stratum == "pooled" else [100]
            bounds = np.cumsum([0, *sizes])
            draws = []
            for _ in range(500):
                idx = np.concatenate(
                    [rng.integers(lo, hi, size=hi - lo) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]
                )
                draws.append(1 - model[idx].mean() / noop[idx].mean())
            summary["step_gain_ci"] = np.quantile(draws, [0.025, 0.975]).tolist()
            summaries.append(summary)
            if t in HIGH:
                fields = [
                    "radius_ratio",
                    "v_norm_ratio",
                    "v_cosine",
                    "step_gain",
                    "global_output_norm",
                    "local_global_norm_ratio",
                ]
                for i, a in enumerate(fields):
                    for b in fields[i + 1 :]:
                        values = [(r[a], r[b]) for r in chosen if r[a] is not None and r[b] is not None]
                        cors.append(
                            dict(
                                checkpoint=label,
                                timestep=t,
                                length_stratum=stratum,
                                a=a,
                                b=b,
                                **correlation(*zip(*values, strict=True)),
                            )
                        )
    groups = marginal.analyze_phase("one_step", f"{label}.descriptors.parquet")
    for g in groups:
        g.pop("_model_draws", None)
        g["checkpoint"] = label
        raw = g["raw_descriptor_distributions"]
        rg = raw["model"]["rg"]["mean"] / raw["ref_b"]["rg"]["mean"]
        trace = raw["model"]["trace"]["mean"] / raw["ref_b"]["trace"]["mean"]
        g.update(
            rg_mean_ratio=rg,
            trace_mean_ratio=trace,
            resolution_status=marginal_status(g["metrics"], rg, trace),
            **schedule_values(CoordinateVPDiffusion(500), g["timestep"]),
        )
    rollouts = []
    if label in LABELS[-2:]:
        rollouts = marginal.analyze_phase("gaussian_prefix", f"{label}.gaussian_descriptors.parquet")
        for g in rollouts:
            g.pop("_model_draws", None)
            g["checkpoint"] = label
            raw = g["raw_descriptor_distributions"]
            rg = raw["model"]["rg"]["mean"] / raw["ref_b"]["rg"]["mean"]
            trace = raw["model"]["trace"]["mean"] / raw["ref_b"]["trace"]["mean"]
            g.update(
                rg_mean_ratio=rg,
                trace_mean_ratio=trace,
                noisy_scale_status=noisy_scale_status(rg, trace),
                **schedule_values(CoordinateVPDiffusion(500), g["timestep"]),
            )
    result = dict(
        checkpoint=label,
        paired_summaries=summaries,
        marginal_groups=groups,
        gaussian_groups=rollouts,
        per_protein_correlations=cors,
        analysis_seconds=time.time() - start,
        source_paired_sha256=sha(OUT / f"{label}.paired.parquet"),
        source_descriptor_sha256=sha(OUT / f"{label}.descriptors.parquet"),
        protocol_sha256=sha(OUT / "PROTOCOL.json"),
    )
    atomic_json(target, result)
    print(f"analysis certified {label}", flush=True)
    return result


def geometry_fronts():
    rows = {label: load_paired(label) for label in LABELS}
    maps = {label: {(r["sample_id"], r["timestep"]): r for r in data} for label, data in rows.items()}
    output = []
    comparisons = list(zip(LABELS[:-1], LABELS[1:], strict=True))
    comparisons.extend((LABELS[0], label) for label in LABELS[2:])
    for old, new in comparisons:
        for stratum in (*STRATA, "pooled"):
            grid = []
            for t in LOW_MID:
                chosen = sorted(
                    [
                        r
                        for r in rows[new]
                        if r["timestep"] == t and (stratum == "pooled" or r["length_stratum"] == stratum)
                    ],
                    key=lambda r: (STRATA.index(r["length_stratum"]), r["sample_id"]),
                )
                delta = np.array(
                    [r["local_mse_angstrom2"] - maps[old][r["sample_id"], t]["local_mse_angstrom2"] for r in chosen]
                )
                sizes = [100] * 5 if stratum == "pooled" else [100]
                bounds = np.cumsum([0, *sizes])
                rng = np.random.default_rng(noise_seed("e014-high-noise-geometry-bootstrap-v1", stratum, t))
                means = []
                for _ in range(500):
                    idx = np.concatenate(
                        [rng.integers(lo, hi, size=hi - lo) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]
                    )
                    means.append(delta[idx].mean())
                ci = np.quantile(means, [0.025, 0.975]).tolist()
                grid.append(
                    dict(
                        timestep=t,
                        mean_delta=float(delta.mean()),
                        ci=ci,
                        resolution_status="resolved_positive"
                        if ci[1] < 0
                        else "resolved_negative"
                        if ci[0] > 0
                        else "unresolved",
                    )
                )
            output.append(dict(old=old, new=new, length_stratum=stratum, grid=grid, **resolution_front(grid)))
    return output


def finalize():
    checkpoints = [json.loads((OUT / f"{label}.ANALYSIS_COMPLETE.json").read_text()) for label in LABELS]
    assert json.loads((OUT / "HEARTBEAT.json").read_text())["phase"] == "inference_complete"
    for c in checkpoints:
        assert c["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
        assert c["source_paired_sha256"] == sha(OUT / f"{c['checkpoint']}.paired.parquet")
    paired = [g for c in checkpoints for g in c["paired_summaries"]]
    marginals = [g for c in checkpoints for g in c["marginal_groups"]]
    gaussian = [g for c in checkpoints for g in c["gaussian_groups"]]
    fronts = []
    onsets = []
    aggregate_correlations = []
    for label in LABELS:
        for stratum in (*STRATA, "pooled"):
            ms = [g for g in marginals if g["checkpoint"] == label and g["length_stratum"] == stratum]
            front = resolution_front(ms)
            t = front["highest_resolved_timestep"]
            front.update(
                checkpoint=label,
                length_stratum=stratum,
                snr_at_front=next((g["snr"] for g in ms if g["timestep"] == t), None),
                log_snr_at_front=next((g["log_snr"] for g in ms if g["timestep"] == t), None),
            )
            fronts.append(front)
            ps = [g for g in paired if g["checkpoint"] == label and g["length_stratum"] == stratum]
            onsets.append(dict(checkpoint=label, length_stratum=stratum, **collapse_onset(ps)))
            # Distribution energy improvement is a population statistic, not an individual error.
            population_rows = []
            for t in HIGH:
                p = next(g for g in ps if g["timestep"] == t)
                m = next(g for g in ms if g["timestep"] == t)
                e = next(r for r in m["metrics"] if r["metric"] == "energy")
                population_rows.append(
                    dict(
                        timestep=t,
                        rg=p["direct_rg_mean"],
                        v_norm=p["v_norm_ratio"]["mean"],
                        v_cosine=p["v_cosine"]["mean"],
                        step_gain=p["step_gain_of_means"],
                        marginal_energy_improvement=e["noop_ref"] - e["model_ref"],
                        global_norm=p["global_output_norm"]["mean"],
                        local_global=p["local_global_norm_ratio"]["mean"],
                    )
                )
            names = list(population_rows[0])
            names.remove("timestep")
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    aggregate_correlations.append(
                        dict(
                            checkpoint=label,
                            length_stratum=stratum,
                            a=a,
                            b=b,
                            level="across15timesteps; SNR-confounded ecological association",
                            **correlation([r[a] for r in population_rows], [r[b] for r in population_rows]),
                        )
                    )
    geometry = geometry_fronts()
    resolution = dict(
        marginal_fronts=fronts,
        direct_clean_onsets=onsets,
        geometry_improvement_fronts=geometry,
        protocol_sha256=sha(OUT / "PROTOCOL.json"),
        not_scientific_gates=True,
    )
    atomic_json(OUT / "resolution_front.json", resolution)
    result = dict(
        paired=paired,
        marginal=marginals,
        gaussian=gaussian,
        resolution=resolution,
        per_protein_correlations=[r for c in checkpoints for r in c["per_protein_correlations"]],
        across_timestep_correlations=aggregate_correlations,
        final_adjudication=False,
    )
    atomic_json(OUT / "metrics.json", result)
    flat = []
    for g in marginals + gaussian:
        for m in g["metrics"]:
            flat.append(
                dict(
                    checkpoint=g["checkpoint"],
                    phase=g["phase"],
                    timestep=g["timestep"],
                    length_stratum=g["length_stratum"],
                    sample_count=g["sample_count"],
                    rg_mean_ratio=g["rg_mean_ratio"],
                    trace_mean_ratio=g["trace_mean_ratio"],
                    resolution_status=g.get("resolution_status"),
                    **m,
                )
            )
    pq.write_table(
        pa.Table.from_pylist([r for r in flat if r["phase"] == "one_step"]),
        OUT / "high_noise_grid.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist([r for r in flat if r["phase"] == "gaussian_prefix"]),
        OUT / "gaussian_prefix_metrics.parquet",
        compression="zstd",
    )
    geometry_rows = [
        dict(old=g["old"], new=g["new"], length_stratum=g["length_stratum"], **row)
        for g in geometry
        for row in g["grid"]
    ]
    pq.write_table(pa.Table.from_pylist(geometry_rows), OUT / "geometry_front_comparisons.parquet", compression="zstd")
    curves = []
    for group in paired:
        if group["timestep"] not in LOW_MID:
            continue
        row = {
            k: group[k]
            for k in (
                "checkpoint",
                "timestep",
                "length_stratum",
                "identity_count",
                "alpha",
                "sigma",
                "snr",
                "log_snr",
                "triple_inversion_rate_pooled",
                "torsion_assessed_sum",
                "triple_sign_assessed_sum",
                "triple_inversion_count_sum",
            )
        }
        for field in (
            "local_mse_angstrom2",
            "offset_1_mse_angstrom2",
            "offset_2_mse_angstrom2",
            "offset_3_mse_angstrom2",
            "torsion_mse",
            "torsion_angular_mae_radians",
            "triple_product_mse",
        ):
            row[field] = group.get(field, {}).get("mean")
            row[field + "_protein_denominator"] = group.get(field, {}).get("denominator", 0)
        curves.append(row)
    pq.write_table(pa.Table.from_pylist(curves), OUT / "low_mid_geometry_grid.parquet", compression="zstd")
    atomic_json(
        OUT / "ANALYSIS_CERTIFICATION.json",
        dict(
            condition_count=len(marginals) + len(gaussian),
            bootstrap_replicates=500,
            maximum_permutation_error=max(g["permutation_max_error"] for g in marginals + gaussian),
            protocol_sha256=sha(OUT / "PROTOCOL.json"),
        ),
    )
    print("high-noise grids and fronts certified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["checkpoint", "finalize"])
    parser.add_argument("--label", choices=LABELS)
    args = parser.parse_args()
    if args.phase == "checkpoint":
        if not args.label:
            parser.error("--label required")
        analyze_checkpoint(args.label)
    else:
        finalize()
