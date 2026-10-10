"""Paired, protein-weighted analysis of the frozen historical comparison."""

# ruff: noqa: I001, E501, F403, F405
from e007_e014_comparison_protocol import *
import argparse
import time
import analyze_e014_marginal_reverse as marginal
from analyze_e014_high_noise import paired_summary
from protein_distance_diffusion.evaluation.high_noise_resolution import schedule_values, noisy_scale_status
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    ReferenceStandardizer,
    compare_populations,
)


def analyze(label):
    target = OUT / f"{label}.ANALYSIS_COMPLETE.json"
    if target.exists():
        return
    assert (OUT / f"{label}.INFERENCE_COMPLETE.json").exists()
    marginal.OUT = OUT
    groups = {}
    for phase, suffix in (("one_step", "descriptors"), ("gaussian_prefix", "gaussian_descriptors")):
        groups[phase] = marginal.analyze_phase(phase, f"{label}.{suffix}.parquet")
        for g in groups[phase]:
            g["model_bootstrap_draws"] = g.pop("_model_draws")
            raw = g["raw_descriptor_distributions"]
            rg = raw["model"]["rg"]["mean"] / raw["ref_b"]["rg"]["mean"]
            trace = raw["model"]["trace"]["mean"] / raw["ref_b"]["trace"]["mean"]
            g.update(
                checkpoint=label,
                rg_mean_ratio=rg,
                trace_mean_ratio=trace,
                coordinate_rms_ratio=float(np.sqrt(trace)),
                **schedule_values(CoordinateVPDiffusion(500), g["timestep"]),
            )
            if phase == "gaussian_prefix":
                g["noisy_scale_status"] = noisy_scale_status(rg, trace)
    raw = pq.read_table(OUT / f"{label}.paired.parquet", use_threads=False).to_pylist()
    summaries = [
        dict(
            checkpoint=label,
            timestep=t,
            length_stratum=s,
            **paired_summary([r for r in raw if r["timestep"] == t and (s == "pooled" or r["length_stratum"] == s)]),
        )
        for t in GRID
        for s in (*STRATA, "pooled")
    ]
    write(
        target.name,
        dict(
            checkpoint=label,
            paired_summaries=summaries,
            marginal_groups=groups["one_step"],
            gaussian_groups=groups["gaussian_prefix"],
            source_paired_sha256=sha(OUT / f"{label}.paired.parquet"),
            source_descriptor_sha256=sha(OUT / f"{label}.descriptors.parquet"),
        ),
    )


def source(label):
    return OUT if label in E007_LABELS else E014


def primary_distribution(label, t):
    cached = OUT / f"{label}.primary_gaussian_{t}.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        assert result["bootstrap_replicates"] == 5000
        return result
    rows = pq.read_table(source(label) / f"{label}.gaussian_descriptors.parquet", use_threads=False).to_pylist()
    pieces, norms = {}, {}
    for s in STRATA:
        pieces[s] = {
            p: np.array(
                [
                    r["descriptor"]
                    for r in sorted(rows, key=lambda r: r["sample_id"])
                    if r["timestep"] == t and r["length_stratum"] == s and r["population"] == p
                ]
            )
            for p in ("ref_a", "ref_b", "noop", "model")
        }
        norms[s] = ReferenceStandardizer(pieces[s]["ref_a"], pieces[s]["ref_b"])
    columns = sorted(set.intersection(*(set(n.columns) for n in norms.values())))
    raw = {p: np.concatenate([pieces[s][p] for s in STRATA]) for p in pieces[STRATA[0]]}
    normalized = {
        p: np.concatenate(
            [
                ReferenceStandardizer(pieces[s]["ref_a"], pieces[s]["ref_b"], columns).transform(pieces[s][p])
                for s in STRATA
            ]
        )
        for p in raw
    }
    result = compare_populations(
        normalized,
        raw,
        columns,
        noise_seed("e014-marginal-bootstrap-v1", "rollout|pooled", t),
        [20] * 5,
        replicates=5000,
    )
    write(f"{label}.primary_gaussian_{t}.json", result)
    return result


def distribution_difference(a, b, metric):
    aa = next(r for r in a["metrics"] if r["metric"] == metric)
    bb = next(r for r in b["metrics"] if r["metric"] == metric)
    da = a.get("model_bootstrap_draws", a.get("_model_draws"))[metric]
    db = b.get("model_bootstrap_draws", b.get("_model_draws"))[metric]
    assert len(da) == len(db)
    d = np.array(db) - np.array(da)
    relative = np.array(db)[np.array(da) != 0] / np.array(da)[np.array(da) != 0] - 1
    assert np.isclose(aa["ref_ref"], bb["ref_ref"], atol=1e-10)
    return dict(
        e007=aa["model_ref"],
        e014=bb["model_ref"],
        difference=bb["model_ref"] - aa["model_ref"],
        relative_difference=bb["model_ref"] / aa["model_ref"] - 1 if aa["model_ref"] else None,
        ci95=np.quantile(d, [0.025, 0.975]).tolist(),
        relative_ci95=np.quantile(relative, [0.025, 0.975]).tolist() if len(relative) else None,
        ci_familywise=np.quantile(d, [0.05 / (2 * len(PRIMARY)), 1 - 0.05 / (2 * len(PRIMARY))]).tolist(),
        bootstrap_replicates=len(d),
        ref_ref=aa["ref_ref"],
        noop_ref=aa["noop_ref"],
        e007_minus_noop=aa["model_ref"] - aa["noop_ref"],
        e014_minus_noop=bb["model_ref"] - bb["noop_ref"],
        e007_minus_noop_ci95=aa["model_minus_noop_ci"],
        e014_minus_noop_ci95=bb["model_minus_noop_ci"],
    )


def finish():
    labels = (*E007_LABELS, *E014_LABELS)
    records = {
        label: pq.read_table(source(label) / f"{label}.paired.parquet", use_threads=False).to_pylist()
        for label in labels
    }
    analyses = {label: json.loads((source(label) / f"{label}.ANALYSIS_COMPLETE.json").read_text()) for label in labels}
    panel, _ = population()
    ids = [r["sample_id"] for r in panel]
    strata = [r["length_stratum"] for r in panel]
    by = {label: {(r["sample_id"], r["timestep"]): r for r in rr} for label, rr in records.items()}
    for label in labels:
        assert set(by[label]) == {(i, t) for i in ids for t in GRID}
    allcore = [
        dict(checkpoint=label, **{k: v for k, v in r.items() if k != "checkpoint"})
        for label in labels
        for r in records[label]
        if r["timestep"] in CORE
    ]
    table("core_metrics.parquet", allcore)
    geomkeys = ("offset_1_mse_angstrom2", "offset_2_mse_angstrom2", "offset_3_mse_angstrom2", "local_mse_angstrom2")
    chirkeys = (
        "torsion_mse",
        "torsion_angular_mae_radians",
        "triple_product_mse",
        "triple_inversion_fraction",
        "triple_inversion_count",
        "triple_sign_assessed",
        "torsion_assessed",
        "torsion_potential_count",
    )
    basekeys = ("checkpoint", "sample_id", "length", "length_stratum", "timestep")
    table("geometry_metrics.parquet", [{k: r.get(k) for k in (*basekeys, *geomkeys)} for r in allcore])
    table("chirality_metrics.parquet", [{k: r.get(k) for k in (*basekeys, *chirkeys)} for r in allcore])
    table(
        "high_noise_grid.parquet",
        [
            dict(checkpoint=label, **{k: v for k, v in r.items() if k != "checkpoint"})
            for label in labels
            for r in records[label]
            if r["timestep"] in HIGH
        ],
    )
    efficiency = []
    for label in labels:
        chosen = [r for r in records[label] if r["timestep"] in CORE]
        parameters = 7586505 if label in E007_LABELS else 5119069
        mse = float(np.mean([r["v_mse"] for r in chosen]))
        baseline = float(np.mean([r["v_target_norm"] ** 2 / (3 * r["length"]) for r in chosen]))
        quality = 1 - mse / baseline
        efficiency.append(
            dict(
                checkpoint=label,
                parameters=parameters,
                v_mse=mse,
                zero_v_baseline_mse=baseline,
                fractional_v_mse_reduction=quality,
                fractional_v_mse_reduction_per_million_parameters=quality / (parameters / 1e6),
                local_mse=float(np.mean([r["local_mse_angstrom2"] for r in chosen])),
                error_per_million_parameters=mse / (parameters / 1e6),
            )
        )
    write(
        "parameter_efficiency.json",
        dict(
            records=efficiency,
            interpretation="Descriptive arithmetic only; training is unmatched. Fractional MSE reduction relative to zero-v is divided by parameter millions. Error/parameter is also recorded but is not an efficiency ranking or scaling law.",
        ),
    )
    primary, secondary = [], []
    metrics = (
        "v_mse",
        "v_cosine",
        "v_norm_ratio",
        "x0_mse_angstrom2",
        "aligned_rmsd_angstrom",
        "radius_ratio",
        "covariance_trace_ratio",
        *geomkeys,
        *chirkeys[:4],
        "step_cosine",
        "step_norm_ratio",
        "step_gain",
        "collapsed",
    )
    for a in E007_LABELS:
        for b in E014_LABELS:
            for t in GRID:
                for s in (*STRATA, "pooled"):
                    indices = [i for i, r in enumerate(panel) if s == "pooled" or r["length_stratum"] == s]
                    for metric in metrics:
                        av = [by[a][ids[i], t].get(metric) for i in indices]
                        bv = [by[b][ids[i], t].get(metric) for i in indices]
                        est = paired_estimate(av, bv, [strata[i] for i in indices], replicates=500)
                        est.pop("draws", None)
                        secondary.append(
                            dict(
                                e007_checkpoint=a, e014_checkpoint=b, timestep=t, length_stratum=s, metric=metric, **est
                            )
                        )
            for name, metric, ts in (
                ("aggregate_v_mse", "v_mse", CORE),
                ("aggregate_local_mse", "local_mse_angstrom2", CORE),
                ("torsion_angular_400", "torsion_angular_mae_radians", (400,)),
                ("inversion_rate_400", "triple_inversion_fraction", (400,)),
            ):
                if a != E007_LABELS[0] or b != E014_LABELS[0]:
                    continue
                av = [np.mean([by[a][i, t][metric] for t in ts]) for i in ids]
                bv = [np.mean([by[b][i, t][metric] for t in ts]) for i in ids]
                est = paired_estimate(av, bv, strata)
                est.pop("draws", None)
                primary.append(dict(metric=name, **est))
    gain_rows = []
    for a in E007_LABELS:
        for b in E014_LABELS:
            for t in HIGH:
                for s in (*STRATA, "pooled"):
                    chosen = [i for i, r in enumerate(panel) if s == "pooled" or r["length_stratum"] == s]
                    ns = [strata[i] for i in chosen]
                    ix = bootstrap_indices(ns, replicates=500)
                    nop = np.array([by[a][ids[i], t]["noop_mse"] for i in chosen])
                    ma = np.array([by[a][ids[i], t]["model_mse"] for i in chosen])
                    mb = np.array([by[b][ids[i], t]["model_mse"] for i in chosen])
                    ga = 1 - ma[ix].mean(1) / nop[ix].mean(1)
                    gb = 1 - mb[ix].mean(1) / nop[ix].mean(1)
                    gain_rows.append(
                        dict(
                            e007_checkpoint=a,
                            e014_checkpoint=b,
                            timestep=t,
                            length_stratum=s,
                            e007=1 - float(ma.mean() / nop.mean()),
                            e014=1 - float(mb.mean() / nop.mean()),
                            difference=float((ma.mean() - mb.mean()) / nop.mean()),
                            ci95=np.quantile(gb - ga, [0.025, 0.975]).tolist(),
                            e007_ci95=np.quantile(ga, [0.025, 0.975]).tolist(),
                            e014_ci95=np.quantile(gb, [0.025, 0.975]).tolist(),
                            bootstrap_replicates=500,
                            metric="paired_step_gain_of_means",
                        )
                    )
    table("high_noise_step_gain_intervals.parquet", gain_rows)
    write("paired_summaries.json", {label: analyses[label]["paired_summaries"] for label in labels})
    input_checks = []
    # The no-op paired target and Gaussian held initial state are model-free:
    # equality verifies actual frozen inputs, beyond checking seed formulas.
    a0 = by[E007_LABELS[0]]
    for label in labels[1:]:
        maximum = max(abs(a0[key]["noop_mse"] - by[label][key]["noop_mse"]) for key in a0)
        assert maximum == 0, (label, maximum)
        rows = pq.read_table(source(label) / f"{label}.gaussian_descriptors.parquet", use_threads=False).to_pylist()
        base = pq.read_table(OUT / f"{E007_LABELS[0]}.gaussian_descriptors.parquet", use_threads=False).to_pylist()

        def key(r):
            return r["sample_id"], r["timestep"], r["population"]

        frozen = {key(r): r["descriptor"] for r in base if r["population"] != "model"}
        error = max(
            float(np.nanmax(np.abs(np.array(r["descriptor"]) - frozen[key(r)])))
            for r in rows
            if r["population"] != "model"
        )
        assert error == 0, (label, error)
        input_checks.append(
            dict(
                checkpoint=label,
                paired_noop_max_difference=maximum,
                gaussian_init_and_independent_references_max_descriptor_difference=error,
            )
        )
    write(
        "MATCHED_EVALUATION_INPUT_CHECKS.json",
        dict(checks=input_checks, panel_hash=PANEL_SHA, subset_hash=SUBSET_SHA, matched_training=False),
    )
    distributions = []
    flat = {"marginal_groups": [], "gaussian_groups": []}
    for kind in flat:
        for a in E007_LABELS:
            for b in E014_LABELS:
                ag = {(g["timestep"], g["length_stratum"]): g for g in analyses[a][kind]}
                for bg in analyses[b][kind]:
                    ga = ag[bg["timestep"], bg["length_stratum"]]
                    assert ga["reference_preprocessing"] == bg["reference_preprocessing"]
                    for m in ga["metrics"]:
                        distributions.append(
                            dict(
                                phase=kind,
                                e007_checkpoint=a,
                                e014_checkpoint=b,
                                timestep=bg["timestep"],
                                length_stratum=bg["length_stratum"],
                                metric=m["metric"],
                                **distribution_difference(ga, bg, m["metric"]),
                            )
                        )
        for label in labels:
            for g in analyses[label][kind]:
                raw = g["raw_descriptor_distributions"]
                descriptor_means = {
                    f"{pop}_{name}_protein_mean": values["mean"]
                    for pop in ("model", "ref_b")
                    for name, values in raw[pop].items()
                }
                descriptor_stds = {
                    f"{pop}_{name}_between_protein_std": values["std"]
                    for pop in ("model", "ref_b")
                    for name, values in raw[pop].items()
                }
                for m in g["metrics"]:
                    flat[kind].append(
                        dict(
                            checkpoint=label,
                            timestep=g["timestep"],
                            length_stratum=g["length_stratum"],
                            sample_count=g["sample_count"],
                            rg_ratio=g["rg_mean_ratio"],
                            covariance_trace_ratio=g["trace_mean_ratio"],
                            coordinate_rms_ratio=float(np.sqrt(g["trace_mean_ratio"])),
                            pair_mean_ratio=raw["model"]["pair_mean"]["mean"] / raw["ref_b"]["pair_mean"]["mean"],
                            finite=True,
                            noisy_state_collapse=g["rg_mean_ratio"] < 0.8 or g["trace_mean_ratio"] < 0.8,
                            **descriptor_means,
                            **descriptor_stds,
                            **m,
                        )
                    )
    for t in (400, 350):
        a, b = [primary_distribution(label, t) for label in (E007_LABELS[0], E014_LABELS[0])]
        primary.append(dict(metric=f"gaussian_energy_{t}", **distribution_difference(a, b, "energy")))
    table("marginal_metrics.parquet", flat["marginal_groups"])
    table("gaussian_prefix_metrics.parquet", flat["gaussian_groups"])
    table("length_strata.parquet", secondary)
    write(
        "bootstrap_intervals.json",
        dict(
            primary=primary,
            secondary=secondary,
            distribution_comparisons=distributions,
            direction="E014 minus E007",
            causal_classification="C5",
            familywise_interval_coverage=1 - 0.05 / 6,
            assumptions="Paired protein bootstrap preserves length strata; equal protein weights. Distribution fits and projections held fixed; finite panel/checkpoint conditional inference. Training remains unmatched.",
        ),
    )
    write(
        "COMPARISON_COMPLETE.json",
        dict(
            primary_classification="C5",
            architecture_attribution_allowed=False,
            primary=primary,
            selected=list(labels),
            all_noisy_states_noncollapsed=not any(r["noisy_state_collapse"] for r in flat["gaussian_groups"]),
            artifacts_on_D=True,
            full_sampling=False,
            training=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["analyze", "watch", "finish"])
    parser.add_argument("--checkpoint", choices=E007_LABELS)
    args = parser.parse_args()
    if args.phase == "watch":
        for label in E007_LABELS:
            while not (OUT / f"{label}.INFERENCE_COMPLETE.json").exists():
                time.sleep(10)
            analyze(label)
        finish()
    elif args.phase == "analyze":
        analyze(args.checkpoint)
    else:
        finish()
