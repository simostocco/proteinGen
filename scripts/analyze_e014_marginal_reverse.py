"""CPU-only marginal comparisons, protein bootstrap and observational reporting."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import json
import argparse
import time
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collect_e014_marginal_reverse import OUT, STRATA
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    NAMES,
    ReferenceStandardizer,
    compare_populations,
)
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha

assert OUT.is_relative_to(ROOT)


def distribution(values):
    values = np.asarray(values)
    return dict(
        count=len(values),
        mean=float(values.mean()),
        std=float(values.std()),
        quantiles=np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
    )


def analyze_phase(phase, filename):
    rows = pq.read_table(OUT / filename, use_threads=False).to_pylist()
    assert len({r["sample_id"] for r in rows}) == (500 if phase == "one_step" else 100)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["timestep"], row["length_stratum"], row["population"])].append(row)
    results = []
    for timestep in sorted({r["timestep"] for r in rows}):
        raw_by_stratum = {}
        normalizers = {}
        for stratum in STRATA:
            raw = {}
            for label in ("ref_a", "ref_b", "noop", "model"):
                chosen = sorted(grouped[(timestep, stratum, label)], key=lambda r: r["sample_id"])
                assert len(chosen) == (100 if phase == "one_step" else 20)
                raw[label] = np.asarray([r["descriptor"] for r in chosen], float)
            raw_by_stratum[stratum] = raw
            normalizers[stratum] = ReferenceStandardizer(raw["ref_a"], raw["ref_b"])
        common = sorted(set.intersection(*(set(n.columns) for n in normalizers.values())))
        for stratum in (*STRATA, "pooled"):
            if stratum == "pooled":
                raw = {
                    label: np.concatenate([raw_by_stratum[s][label] for s in STRATA])
                    for label in ("ref_a", "ref_b", "noop", "model")
                }
                normalized = {
                    label: np.concatenate(
                        [
                            ReferenceStandardizer(
                                raw_by_stratum[s]["ref_a"], raw_by_stratum[s]["ref_b"], common
                            ).transform(raw_by_stratum[s][label])
                            for s in STRATA
                        ]
                    )
                    for label in raw
                }
                columns = common
                sizes = [100 if phase == "one_step" else 20] * 5
            else:
                raw = raw_by_stratum[stratum]
                norm = normalizers[stratum]
                normalized = {label: norm.transform(values) for label, values in raw.items()}
                columns = norm.columns
                sizes = [len(raw["model"])]
            # Shared bootstrap draws for teacher/gaussian at matching endpoints.
            seed = noise_seed(
                "e014-marginal-bootstrap-v1", f"{'one_step' if phase == 'one_step' else 'rollout'}|{stratum}", timestep
            )
            result = compare_populations(normalized, raw, columns, seed, sizes, replicates=500)
            result.update(
                phase=phase,
                timestep=timestep,
                length_stratum=stratum,
                raw_descriptor_distributions={
                    label: {
                        name: distribution(values[:, i])
                        for i, name in enumerate(NAMES)
                        if np.isfinite(values[:, i]).all()
                    }
                    for label, values in raw.items()
                },
                reference_preprocessing={
                    s: dict(
                        columns=normalizers[s].columns.tolist(),
                        location=normalizers[s].location.tolist(),
                        scale=normalizers[s].scale.tolist(),
                    )
                    for s in STRATA
                }
                if stratum == "pooled"
                else dict(
                    columns=normalizers[stratum].columns.tolist(),
                    location=normalizers[stratum].location.tolist(),
                    scale=normalizers[stratum].scale.tolist(),
                ),
            )
            results.append(result)
        print(f"CPU distribution analysis {phase} t={timestep} complete", flush=True)
        atomic_json(OUT / "ANALYSIS_HEARTBEAT.json", dict(phase=phase, timestep=timestep, timestamp_unix=time.time()))
    return results


def metric(group, name):
    return next(r for r in group["metrics"] if r["metric"] == name)


def render(results, gaps, elapsed):
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    subset = json.loads((OUT / "subset_identity.json").read_text())
    oracle = json.loads((OUT / "oracle_identity.json").read_text())
    pooled = [r for r in results if r["length_stratum"] == "pooled"]
    lines = [
        "# E014 frozen 50%-epoch marginal reverse-distribution diagnostic — non-adjudicating",
        "",
        f"Checkpoint `{identity['path']}` SHA256 `{identity['sha256']}`: epoch1 / update and cursor90,413 / exposures115,067. Panel500 (100 per length bin), SHA256 `{subset['panel_sha256']}`; subset100 (20 per bin), SHA256 `{subset['subset_sha256']}`.",
        "",
        f"True-v identity: maximum normalized absolute error {oracle['max_abs_error']:.9g}, tolerance5e-6, {oracle['conditions']} bounded conditions. No main metric uses paired coordinates or identities. All distribution distances compare populations without matching pairs; reference ordering permutation checks passed.",
        "",
        "## Frozen protocol",
        "",
        "34 invariant descriptors: Rg; descending covariance eigenvalues, trace, largest/smallest eigenvalue fractions; scalar component RMS; unordered all-pair distance mean/std/five quantiles; distance mean/std for chain separations1/2/3/4–8/9–32/>32; residue-norm mean/std/five quantiles. No clean-protein cutoffs, sequence, target-derived inference features, torsion or chirality features. Distances/norms use normalized units (one unit=12.22820347644835 Å); covariance uses squared units. The descriptor is invariant under rotations/translations and also reflections, so it cannot assess chirality or the full joint coordinate distribution.",
        "",
        "Independent source/ref-A/ref-B seed namespaces are frozen in subset_identity.json. References use the same clean identity/length population with independent noise. Only reference B fits robust median/IQR scaling; reference A/B determine finite, nonconstant columns. Missing >32-separation channels are excluded where unavailable. Pooled comparisons concatenate separately reference-standardized strata on their common columns; every protein has equal weight.",
        "",
        "Energy distance and MMD² use empirical biased V-statistics (nonnegative); energy Euclidean distance is divided by sqrt(descriptor dimension). MMD RBF bandwidth is median positive distance among reference A/B only. Sliced W1 uses64 deterministic unit projections. Rg W1/KS use raw normalized radius; covariance-spectrum W1 averages standardized eig1/eig2/eig3/trace/anisotropy1/anisotropy3 distributions. Descriptor mean error is RMS of mean differences; covariance error is Frobenius difference divided by reference covariance norm.",
        "",
        "500 deterministic protein-unit bootstrap resamples per condition; pooled resampling preserves length strata. Shared identity indices preserve the matched length/clean-mixture design across independent noise populations; distances themselves never match identity pairs. Reference preprocessing/kernel/projections are fixed during bootstrap, so intervals are conditional on those fits and this checkpoint/panel. No multiplicity adjustment: interpretation is exploratory. Calibrated gain is undefined if no-op does not exceed the ref-ref floor. Gain intervals can be unstable near zero excess distance; valid replicate counts are retained.",
        "",
        "A condition is descriptively improving/worsening if ≥3/5 primary model-minus-noop95% intervals exclude zero in that direction and none exclude zero in the opposite direction; otherwise unresolved. This is not an E014 scientific gate.",
        "",
        "## One-step marginal comparisons (pooled after stratum standardization)",
        "",
    ]
    for group in pooled:
        if group["phase"] != "one_step":
            continue
        lines += [
            f"### t={group['timestep']} → {group['timestep'] - 1}: {group['status']}",
            "",
            "|Metric|Ref–ref|No-op–ref|Model–ref|Calibrated gain|Model–ref95% CI|Model minus no-op95% CI|",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for row in group["metrics"]:
            gain = "undefined" if row["gain"] is None else f"{row['gain']:.2%}"

            def interval(x):
                return f"[{x[0]:.6g}, {x[1]:.6g}]"

            lines.append(
                f"|{row['metric']}|{row['ref_ref']:.7g}|{row['noop_ref']:.7g}|{row['model_ref']:.7g}|{gain}|{interval(row['model_ref_ci'])}|{interval(row['model_minus_noop_ci'])}|"
            )
    lines += [
        "",
        "## Radius, covariance spectrum and separation-distance means",
        "",
        "All full means/std/quantiles for each descriptor/population/timestep/stratum are in metrics.json. Below, radius means/std are in normalized coordinate units; multiply by frozen scale for Å.",
        "",
        "|One-step input t|Reference Rg mean/std|No-op Rg mean/std|Model Rg mean/std|Model/ref mean Rg|Model/ref covariance trace|Model/ref adjacent mean|Model/ref i+2 mean|Model/ref i+3 mean|",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in pooled:
        if group["phase"] != "one_step":
            continue
        raw = group["raw_descriptor_distributions"]

        def fmt(label, current_raw=raw):
            return f"{current_raw[label]['rg']['mean']:.7f}/{current_raw[label]['rg']['std']:.7f}"

        def ratio(key, current_raw=raw):
            return current_raw["model"][key]["mean"] / current_raw["ref_b"][key]["mean"]

        lines.append(
            f"|{group['timestep']}|{fmt('ref_b')}|{fmt('noop')}|{fmt('model')}|{ratio('rg'):.6f}|{ratio('trace'):.6f}|{ratio('sep_1_1_mean'):.6f}|{ratio('sep_2_2_mean'):.6f}|{ratio('sep_3_3_mean'):.6f}|"
        )
    lines += [
        "",
        "## Length-stratum results",
        "",
        "Each one-step stratum has100 proteins; each rollout/prefix stratum only20, limiting power. These are not fixed-N laws.",
        "",
        "|t|Stratum|Status|Energy ref-ref / noop / model|Energy gain|SW ref-ref / noop / model|",
        "|---:|---|---|---|---:|---|",
    ]
    for group in results:
        if group["phase"] != "one_step" or group["length_stratum"] == "pooled":
            continue
        e, s = metric(group, "energy"), metric(group, "sliced_wasserstein")
        gain = "undefined" if e["gain"] is None else f"{e['gain']:.2%}"
        lines.append(
            f"|{group['timestep']}|{group['length_stratum']}|{group['status']}|{e['ref_ref']:.6g} / {e['noop_ref']:.6g} / {e['model_ref']:.6g}|{gain}|{s['ref_ref']:.6g} / {s['noop_ref']:.6g} / {s['model_ref']:.6g}|"
        )
    lines += [
        "",
        "## Separate marginal rollout and Gaussian prefix",
        "",
        "Teacher rollout starts from real q490 and performs50 reverse steps to440. Gaussian prefix starts at499 from the exact centered Gaussian constructor and performs59 steps to440. References at each endpoint use independent noise; no-op holds each branch's own initial population unchanged. Neither branch reaches t=0. Descriptor matching is not full de novo protein validity.",
        "",
        "|Branch|Endpoint|N|Status vs held initial population|Energy ref-ref / noop / model|MMD² ref-ref / noop / model|SW ref-ref / noop / model|Mean Rg model/ref|",
        "|---|---:|---:|---|---|---|---|---:|",
    ]
    for group in pooled:
        if group["phase"] == "one_step":
            continue

        def fmt(key, current_group=group):
            return " / ".join(
                f"{metric(current_group, key)[field]:.7g}" for field in ("ref_ref", "noop_ref", "model_ref")
            )

        raw = group["raw_descriptor_distributions"]
        lines.append(
            f"|{group['phase']}|{group['timestep']}|{group['sample_count']}|{group['status']}|{fmt('energy')}|{fmt('mmd2')}|{fmt('sliced_wasserstein')}|{raw['model']['rg']['mean'] / raw['ref_b']['rg']['mean']:.6f}|"
        )
    lines += [
        "",
        "### Population exposure difference: Gaussian prefix minus teacher rollout",
        "",
        "Positive means Gaussian prefix is farther from the same independent reference marginal. Both branches use identical reference populations, standardization and bootstrap draws at matching endpoints.",
        "",
        "|Endpoint|Metric|Dteacher|Dgen|Dgen−Dteacher|Difference95% CI|",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in gaps:
        if row["length_stratum"] == "pooled":
            lines.append(
                f"|{row['timestep']}|{row['metric']}|{row['teacher']:.7g}|{row['gaussian']:.7g}|{row['difference']:.7g}|[{row['difference_ci'][0]:.6g}, {row['difference_ci'][1]:.6g}]|"
            )
    lines += [
        "",
        "## Execution and interpretation limits",
        "",
        f"Inference wall time {identity['elapsed_seconds']:.1f}s; CPU analysis invocation {elapsed:.1f}s. The one-step cache records {json.loads((OUT / 'one_step_analysis.json').read_text())['elapsed_seconds']:.1f}s (included in this invocation when freshly computed). Inference peak allocated/reserved {identity['peak_allocated_bytes'] / 2**20:.1f}/{identity['peak_reserved_bytes'] / 2**20:.1f}MiB. CPU threads1; unchanged FP32 CUDA model, no AMP/compile. Active training advanced from {identity['training_before']['total_protein_exposures']:,} to {identity['training_after']['total_protein_exposures']:,} exposures during inference. Model tensors/checkpoint hash verified unchanged. Both E014 and corrected E012 screening were untouched; bounded contention was accepted.",
        "",
        "No full sampler to t=0, E007 comparator/training, scientific gate change, objective change or model change. All generated artifacts/cache/temp stayed on D:. Independent-reference finite-sample uncertainty and descriptor limitations prevent equating a near-floor distance with a verified full unconditional distribution. The minimum two TRAIN epochs remains pending: NOT final E014 adjudication.",
        "",
        "Interpretation is appended after reviewing these frozen distribution-only results.",
        "",
    ]
    with (OUT / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines))


def main():
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one-step-only",
        action="store_true",
        help="CPU analysis of atomically completed one-step descriptors while rollout inference continues",
    )
    early = parser.parse_args().one_step_only
    cachepath = OUT / "one_step_analysis.json"
    cache_identity = dict(
        source_sha256=sha(OUT / "one_step_descriptors.parquet"),
        helper_sha256=sha("src/protein_distance_diffusion/evaluation/marginal_distribution_diagnostic.py"),
        definition_sha256=sha(OUT / "descriptor_definition.json"),
        replicates=500,
    )
    if cachepath.exists():
        cached = json.loads(cachepath.read_text())
        assert cached["identity"] == cache_identity
        initial = cached["groups"]
    else:
        initial = analyze_phase("one_step", "one_step_descriptors.parquet")
        for group in initial:
            group.pop("_model_draws", None)
        atomic_json(cachepath, dict(identity=cache_identity, groups=initial, elapsed_seconds=time.time() - start))
    if early:
        print("one-step marginal CPU analysis certified", flush=True)
        return
    assert json.loads((OUT / "HEARTBEAT.json").read_text())["phase"] == "inference_complete"
    results = list(initial)
    for phase, filename in (
        ("teacher_rollout", "teacher_rollout_descriptors.parquet"),
        ("gaussian_prefix", "gaussian_prefix_descriptors.parquet"),
    ):
        results.extend(analyze_phase(phase, filename))
    keyed = {(r["phase"], r["timestep"], r["length_stratum"]): r for r in results}
    gaps = []
    for s in (490, 480, 465, 440):
        for stratum in (*STRATA, "pooled"):
            teacher, gen = keyed[("teacher_rollout", s, stratum)], keyed[("gaussian_prefix", s, stratum)]
            for name in teacher["_model_draws"]:
                a, b = metric(teacher, name), metric(gen, name)
                delta = np.asarray(gen["_model_draws"][name]) - np.asarray(teacher["_model_draws"][name])
                gaps.append(
                    dict(
                        timestep=s,
                        length_stratum=stratum,
                        metric=name,
                        teacher=a["model_ref"],
                        gaussian=b["model_ref"],
                        difference=b["model_ref"] - a["model_ref"],
                        difference_ci=np.quantile(delta, [0.025, 0.975]).tolist(),
                    )
                )
    for group in results:
        group.pop("_model_draws", None)
    atomic_json(
        OUT / "metrics.json",
        dict(groups=results, population_exposure_gap=gaps, analysis_elapsed_seconds=time.time() - start),
    )
    flat = [
        dict(
            phase=g["phase"],
            timestep=g["timestep"],
            length_stratum=g["length_stratum"],
            sample_count=g["sample_count"],
            status=g["status"],
            **m,
        )
        for g in results
        for m in g["metrics"]
    ]
    for phase, filename in (
        ("one_step", "one_step_distribution_metrics.parquet"),
        ("rollout", "rollout_distribution_metrics.parquet"),
    ):
        selected = [r for r in flat if (r["phase"] == "one_step") == (phase == "one_step")]
        pq.write_table(pa.Table.from_pylist(selected), OUT / filename, compression="zstd")
    atomic_json(
        OUT / "bootstrap_intervals.json",
        dict(replicates=500, condition_count=len(results), conditions=flat, population_exposure_gap=gaps),
    )
    render(results, gaps, time.time() - start)
    atomic_json(
        OUT / "EVIDENCE_HASHES.json",
        {str(p): sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "EVIDENCE_HASHES.json"},
    )
    print("marginal distribution analysis complete", flush=True)


if __name__ == "__main__":
    main()
