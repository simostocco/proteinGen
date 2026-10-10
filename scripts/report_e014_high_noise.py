"""Read-only report/plots from certified high-noise diagnostics, not live weights."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import json
import subprocess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diagnose_e014_high_noise import OUT
from protein_distance_diffusion.evaluation.high_noise_resolution import GRID, HIGH, LOW_MID, ENDPOINTS, LABELS
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha

assert OUT.is_relative_to(ROOT)
ALIASES = {LABELS[0]: "init", LABELS[1]: "25%", LABELS[2]: "50%", LABELS[3]: "75%"}
COLORS = {LABELS[0]: "gray", LABELS[1]: "tab:blue", LABELS[2]: "tab:orange", LABELS[3]: "tab:green"}


def metric(group, name):
    return next(m for m in group["metrics"] if m["metric"] == name)


def main():
    data = json.loads((OUT / "metrics.json").read_text())
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    fronts = data["resolution"]
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    certificate = json.loads((OUT / "ANALYSIS_CERTIFICATION.json").read_text())
    paired = {(g["checkpoint"], g["timestep"], g["length_stratum"]): g for g in data["paired"]}
    marginal = {(g["checkpoint"], g["timestep"], g["length_stratum"]): g for g in data["marginal"]}
    gaussian = {(g["checkpoint"], g["timestep"], g["length_stratum"]): g for g in data["gaussian"]}
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)

    def curve(filename, field, label, thresholds=()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        for ax, timesteps in zip(axes, (GRID, HIGH), strict=True):
            for cp in LABELS:
                gs = [paired[cp, t, "pooled"] for t in timesteps]
                values = [g[field]["mean"] if isinstance(g[field], dict) else g[field] for g in gs]
                ax.plot(
                    timesteps,
                    values,
                    "o-",
                    markersize=3,
                    label=ALIASES[cp],
                    color=COLORS[cp],
                    alpha=0.6 if cp == LABELS[0] else 1,
                )
                if field == "step_gain_of_means":
                    ax.fill_between(
                        timesteps,
                        [g["step_gain_ci"][0] for g in gs],
                        [g["step_gain_ci"][1] for g in gs],
                        alpha=0.1,
                        color=COLORS[cp],
                    )
                if field == "radius_ratio" and cp != LABELS[0]:
                    ax.plot(timesteps, [g[field]["median"] for g in gs], ":", color=COLORS[cp], alpha=0.7)
            for threshold in thresholds:
                ax.axhline(threshold, color="gray", ls="--", lw=1)
            ax.set_xlabel("Noise timestep")
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            ticktimes = [t for t in (5, 100, 250, 350, 400, 440, 470, 490, 498) if t >= min(timesteps)]
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            top.set_xticks(ticktimes)
            top.set_xticklabels([f"{paired[LABELS[-1], t, 'pooled']['log_snr']:.1f}" for t in ticktimes], fontsize=7)
            top.set_xlabel("log-SNR (frozen 500-step VP)", fontsize=9)
        fig.suptitle("E014 frozen checkpoints: observational paired-field curve")
        fig.savefig(plots / filename, dpi=180)
        plt.close(fig)

    curve("v_cosine_vs_t.png", "v_cosine", "Paired v cosine")
    curve("v_norm_vs_t.png", "v_norm_ratio", "Paired predicted / target v norm", (1,))
    curve("reverse_gain_vs_t.png", "step_gain_of_means", "Paired one-step gain over no-op", (0,))
    curve("step_cosine_vs_t.png", "step_cosine", "Paired incremental-step cosine")
    curve("step_norm_vs_t.png", "step_norm_ratio", "Paired incremental-step norm ratio", (1,))
    curve("direct_x0_rg_vs_t.png", "radius_ratio", "Direct X0 Rg ratio (solid mean, dotted median)", (0.5, 0.25))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, times in zip(axes, (protocol["marginal_grid"], HIGH), strict=True):
        for cp in LABELS:
            ms = [metric(marginal[cp, t, "pooled"], "energy") for t in times]
            ax.plot(
                times,
                [m["model_ref"] - m["noop_ref"] for m in ms],
                "o-",
                markersize=3,
                color=COLORS[cp],
                label=ALIASES[cp],
            )
            ax.fill_between(
                times,
                [m["model_minus_noop_ci"][0] for m in ms],
                [m["model_minus_noop_ci"][1] for m in ms],
                alpha=0.1,
                color=COLORS[cp],
            )
        ax.axhline(0, color="gray", ls="--")
        ax.set_xlabel("Noise timestep")
        ax.set_ylabel("Energy model minus no-op (negative improves)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Marginal transition uncertainty; 500 protein bootstrap replicates")
    fig.savefig(plots / "marginal_gain_vs_t.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for cp in LABELS[-2:]:
        gs = [gaussian[cp, t, "pooled"] for t in ENDPOINTS]
        for ax, field in zip(axes, ("rg_mean_ratio", "trace_mean_ratio"), strict=True):
            ax.plot(ENDPOINTS, [g[field] for g in gs], "o-", color=COLORS[cp], label=ALIASES[cp])
            ax.axhline(0.8, color="gray", ls="--")
            ax.axhline(1, color="gray", ls=":")
            ax.set_ylabel("Mean Rg ratio" if field == "rg_mean_ratio" else "Mean covariance trace ratio")
            ax.set_xlabel("Gaussian prefix endpoint")
            ax.grid(alpha=0.2)
            ax.invert_xaxis()
            ax.legend()
    fig.suptitle("Actual noisy population scale: no direct-X0 collapse criterion")
    fig.savefig(plots / "gaussian_rollout_scale_vs_t.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for offset, ax in enumerate(axes, 1):
        for cp in LABELS:
            ax.plot(
                LOW_MID,
                [paired[cp, t, "pooled"][f"offset_{offset}_mse_angstrom2"]["mean"] for t in LOW_MID],
                "o-",
                markersize=3,
                color=COLORS[cp],
                label=ALIASES[cp],
            )
        ax.set_xlabel("Noise timestep")
        ax.set_ylabel(f"L{offset} distance MSE (Å²)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Direct clean-estimate local geometry; not noisy-state validity")
    fig.savefig(plots / "local_geometry_vs_t.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for cp in LABELS:
        gs = [paired[cp, t, "pooled"] for t in LOW_MID]
        axes[0].plot(LOW_MID, [g["torsion_mse"]["mean"] for g in gs], "o-", color=COLORS[cp], label=ALIASES[cp])
        axes[1].plot(
            LOW_MID, [g["triple_inversion_rate_pooled"] for g in gs], "o-", color=COLORS[cp], label=ALIASES[cp]
        )
    for ax in axes:
        ax.set_xlabel("Noise timestep")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Torsion sin/cos sum-of-squares error")
    axes[1].set_ylabel("Triple-product inversion / jointly assessed count")
    fig.suptitle("Direct clean-estimate torsion/chirality; denominators in metric tables")
    fig.savefig(plots / "torsion_chirality_vs_t.png", dpi=180)
    plt.close(fig)
    lines = [
        "# E014 frozen75% high-noise-resolution diagnostic — interim only",
        "",
        f"Exact75% checkpoint: `{identity['checkpoints'][-1]['path']}`, SHA256 `{identity['checkpoints'][-1]['sha256']}`. Update/cursor135,676, exposures172,599, offset0. All four frozen model/config/population/normalization/source identities and deterministic schedule exposure-prefix accounting pass. The live model was never used.",
        "",
        f"Panel500/100 per stratum, SHA256 `{protocol['panel_hash']}`; Gaussian subset100/20 per stratum, SHA256 `{protocol['subset_hash']}`. Paired field/clean geometry retains `e014-paired-validation-v1`; marginal source/refA/refB and Gaussian seeds retain the previous `e014-marginal-*` protocol. Each protocol is paired consistently across checkpoints; they are not mixed into one comparison.",
        "",
        "## Frozen descriptive rules",
        "",
        protocol["front_rule"],
        protocol["negative_rule"],
        "",
        "Highest useful sampled t is not necessarily a monotone boundary. The contiguous sampled-prefix front and isolated/nonmonotone crossings are reported. There are no multiplicity-adjusted hypothesis claims. Reference-only preprocessing and independent reference populations are exactly shared across checkpoints at everyt/stratum, not fitted to model outputs. Existing34-descriptor definitions, energy/MMD²/slicedW1 metric code and500 protein-unit bootstrap draws remain unchanged. High-noise scale comparisons use q_t, not clean-protein cutoffs. Biased V-statistic bootstrap bounds near a finite-sample floor can be shifted from point estimates; nonsignificance is not equivalence. These descriptive fronts do not alter E014 gates.",
        "",
        "## Marginal resolution fronts",
        "",
        "|Checkpoint|Stratum|Highest resolvedt|SNR|Contiguous sampled prefix|Nonmonotone|Resolved timesteps|",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for f in fronts["marginal_fronts"]:
        lines.append(
            f"|{ALIASES[f['checkpoint']]}|{f['length_stratum']}|{f['highest_resolved_timestep']}|{f['snr_at_front']}|{f['contiguous_sampled_prefix_front']}|{f['nonmonotone']}|{f['resolved_timesteps']}|"
        )
    lines += [
        "",
        "## Direct clean-estimate collapse onset",
        "",
        "Per-protein historical collapse remains Rg(X0hat)/Rg(X0)<0.25. The median<0.5 threshold and collapsed fraction>50% are descriptive and separate. Onsets are first sampled crossings, not interpolated exact limits.",
        "",
        "|Checkpoint|Stratum|First median Rg<0.5|First historical collapse fraction>50%|",
        "|---|---|---:|---:|",
    ]
    for f in fronts["direct_clean_onsets"]:
        lines.append(
            f"|{ALIASES[f['checkpoint']]}|{f['length_stratum']}|{f['first_median_rg_below_half']}|{f['first_historical_collapse_fraction_over_half']}|"
        )
    lines += [
        "",
        "## Dense paired-field and direct-estimator curves",
        "",
        "|Checkpoint|t|alpha|sigma|SNR|vMSE|Direct Rg mean/median|Collapsed/N|v cosine|v norm ratio|Step gain [95%]|Step cosine/norm ratio|",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|",
    ]
    for cp in LABELS:
        for t in GRID:
            g = paired[cp, t, "pooled"]
            lines.append(
                f"|{ALIASES[cp]}|{t}|{g['alpha']:.7g}|{g['sigma']:.7g}|{g['snr']:.7g}|{g['v_mse']['mean']:.7g}|{g['direct_rg_mean']:.5f}/{g['direct_rg_median']:.5f}|{g['direct_collapse_count']}/{g['identity_count']}|{g['v_cosine']['mean']:.5f}|{g['v_norm_ratio']['mean']:.5f}|{g['step_gain_of_means']:.5f} [{g['step_gain_ci'][0]:.5f},{g['step_gain_ci'][1]:.5f}]|{g['step_cosine']['mean']:.5f}/{g['step_norm_ratio']['mean']:.5f}|"
            )
    lines += [
        "",
        "All per-protein quantiles/positive-cosine fractions, coordinate-norm and covariance-trace ratios, model/no-op next-state MSE, and paired assessability denominators are in metrics.json and checkpoint-specific paired parquet files. Paired next-state comparisons are deviations from the shared-noise deterministic reference path, not the only valid unconditional trajectory.",
        "",
        "## Marginal one-step comparisons",
        "",
        "|Checkpoint|t|Energy floor/noop/model|MMD² floor/noop/model|SW floor/noop/model|Energy difference95%|Resolution status|",
        "|---|---:|---|---|---|---|---|",
    ]
    for cp in LABELS:
        for t in protocol["marginal_grid"]:
            g = marginal[cp, t, "pooled"]
            values = []
            for name in ["energy", "mmd2", "sliced_wasserstein"]:
                m = metric(g, name)
                values.append("/".join(f"{m[k]:.7g}" for k in ["ref_ref", "noop_ref", "model_ref"]))
            ci = metric(g, "energy")["model_minus_noop_ci"]
            lines.append(f"|{ALIASES[cp]}|{t}|{'|'.join(values)}|[{ci[0]:.7g},{ci[1]:.7g}]|{g['resolution_status']}|")
    lines += [
        "",
        "Rg W1/KS, covariance-spectrum discrepancy, descriptor mean/covariance differences, all intervals/gains and length-stratum results are included in high_noise_grid.parquet and metrics.json. Gains are undefined if no-op does not exceed the realized reference floor.",
        "",
        "## Gaussian prefix: actual noisy-state distribution",
        "",
        "|Checkpoint|Endpoint|Energy floor/noop/model|MMD² floor/noop/model|SW floor/noop/model|Mean Rg ratio|Mean trace ratio|Rg/trace<0.8|",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for cp in LABELS[-2:]:
        for t in ENDPOINTS:
            g = gaussian[cp, t, "pooled"]
            values = []
            for name in ["energy", "mmd2", "sliced_wasserstein"]:
                m = metric(g, name)
                values.append("/".join(f"{m[k]:.7g}" for k in ["ref_ref", "noop_ref", "model_ref"]))
            lines.append(
                f"|{ALIASES[cp]}|{t}|{'|'.join(values)}|{g['rg_mean_ratio']:.7f}|{g['trace_mean_ratio']:.7f}|{g['noisy_scale_status']}|"
            )
    lines += [
        "",
        "Gaussian initialization is the exact centered standard-normal constructor at499, with matching requested lengths and identical frozen seeds across50/75. Rollouts run99 updates only, ending400. Descriptor populations use independent real forward-noised references, never paired-coordinate MSE. Coordinate component RMS equals Rg/sqrt(3), so its population mean ratio equals the Rg mean ratio. Every descriptor raw mean/std/quantile is retained; per-stratum Ns20 limits power.",
        "",
        "## Local geometry and torsion/chirality",
        "",
        "|Checkpoint|t|L1/L2/L3 (Å²)|Mean local MSE|Torsion sin/cos error|Angular MAE(rad)|Triple error|Inversions/assessed|",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for cp in LABELS:
        for t in LOW_MID:
            g = paired[cp, t, "pooled"]
            ls = "/".join(f"{g[f'offset_{k}_mse_angstrom2']['mean']:.5g}" for k in [1, 2, 3])
            lines.append(
                f"|{ALIASES[cp]}|{t}|{ls}|{g['local_mse_angstrom2']['mean']:.6g}|{g['torsion_mse']['mean']:.6g}|{g['torsion_angular_mae_radians']['mean']:.6g}|{g['triple_product_mse']['mean']:.6g}|{g['triple_inversion_count_sum']}/{g['triple_sign_assessed_sum']}|"
            )
    lines += [
        "",
        "### Geometry improvement fronts",
        "",
        "These measure change in local distance MSE, not proof of acceptable absolute geometry. Common-init and adjacent-checkpoint baselines are distinguished; intervals resample matched proteins within length strata.",
        "",
        "|Old→new|Stratum|Highest improvedt|Contiguous prefix|Nonmonotone|",
        "|---|---|---:|---:|---|",
    ]
    for f in fronts["geometry_improvement_fronts"]:
        lines.append(
            f"|{ALIASES[f['old']]}→{ALIASES[f['new']]}|{f['length_stratum']}|{f['highest_resolved_timestep']}|{f['contiguous_sampled_prefix_front']}|{f['nonmonotone']}|"
        )
    lines += [
        "",
        "## Global/local branch telemetry",
        "",
        "|Checkpoint|t|Global norm|Effective local norm|Local/global ratio|Global/local cosine|Gate mean/median|Reliability mean/median|Frame assessed/residues|",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for cp in LABELS[1:]:
        for t in HIGH:
            g = paired[cp, t, "pooled"]
            lines.append(
                f"|{ALIASES[cp]}|{t}|{g['global_output_norm']['mean']:.6g}|{g['local_effective_output_norm']['mean']:.6g}|{g['local_global_norm_ratio']['mean']:.6g}|{g['global_local_cosine']['mean']:.6g}|{g['local_gate_mean']['mean']:.5f}/{g['local_gate_median']['mean']:.5f}|{g['frame_reliability_mean']['mean']:.5f}/{g['frame_reliability_median']['mean']:.5f}|{g['frame_assessable_count_sum']}/{g['frame_residue_count_sum']}|"
            )
    lines += [
        "",
        "Full per-protein Pearson/Spearman correlations are in metrics.json. Marginal energy improvement is a population quantity; correlations involving it are across15high-noise timesteps per checkpoint/stratum and are explicitly SNR-confounded ecological associations. No individual paired reconstruction error is substituted for marginal error, and no correlation establishes causality.",
        "",
        "## Execution and certification",
        "",
        f"True-v identity max {json.loads((OUT / 'oracle_identity.json').read_text())['max_abs_error']:.9g} vs5e-6,450conditions. Final permutation max {certificate['maximum_permutation_error']:.9g} vs1e-10. {certificate['condition_count']} marginal/rollout conditions ×500protein bootstrap replicates. Inference {identity['elapsed_seconds']:.1f}s; peak allocated/reserved {identity['peak_allocated_bytes'] / 2**20:.1f}/{identity['peak_reserved_bytes'] / 2**20:.1f}MiB; CPU1thread/FP32, no AMP/compile. Training advanced {identity['training_before']['total_protein_exposures']:,}→{identity['training_after']['total_protein_exposures']:,} exposures during inference. Checkpoints and model state verified unchanged. Active training and corrected E012 screening were untouched; bounded contention is accepted, not a stopping condition.",
        "",
        "Independent reference populations and shared preprocessing are certified separately. Historical one-step replay passes; one Gaussian-prefix descriptor exceeds the frozen replay tolerance, detailed in the numerical audit below. The descriptor omits chirality/topology/full joint structure and contains redundant scale channels. No t=0 sample, matched E007 training, gate change, architecture/objective/optimizer/data change occurred. Every generated artifact/temp/cache stays on D:. This frozen 75%-of-epoch-1 observation is NOT final E014 adjudication, regardless of subsequent live training progress.",
        "",
        f"Reporting code commit: `{subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}` (final certification records final documentation commit).",
        "",
        "Scientific interpretation is appended after reviewing the frozen results.",
        "",
    ]
    path = OUT / "REPORT.md"
    with path.open("x") as f:
        f.write("\n".join(lines))
    atomic_json(
        OUT / "REPORT_GENERATION.json",
        dict(
            metric_sha256=sha(OUT / "metrics.json"),
            report_sha256=sha(path),
            plots={str(p): sha(p) for p in plots.iterdir()},
        ),
    )
    print("frozen high-noise report and plots rendered", flush=True)


if __name__ == "__main__":
    main()
