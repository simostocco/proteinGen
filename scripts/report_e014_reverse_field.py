"""Render completed observational reverse-field artifacts without model calls."""

# ruff: noqa: E501, I001
from e014_bootstrap import ROOT
import json
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protein_distance_diffusion.training.e014_runtime import atomic_json, sha

OUT = ROOT / "e014/reverse_field_50pct_v1"


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    rollout = json.loads((OUT / "rollout_summary.json").read_text())
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    plan = json.loads((OUT / "FROZEN_PLAN.json").read_text())
    assert json.loads((OUT / "HEARTBEAT.json").read_text())["phase"] == "complete"
    # Preserve raw output. Initial-state rows have no executed input transition;
    # expose that explicitly rather than carrying the raw 491 marker as a step.
    raw_csv = OUT / "rollout_per_step.csv"
    raw_digest = sha(raw_csv)
    with raw_csv.open() as stream:
        indexed = list(csv.DictReader(stream))
    assert len(indexed) == 5100
    for row in indexed:
        if int(row["timestep"]) == 490:
            assert row["input_timestep"] in ("491", "") and float(row["raw_mse"]) == 0
            row["input_timestep"] = ""
        else:
            assert int(row["input_timestep"]) == int(row["timestep"]) + 1
    with (OUT / "rollout_per_step_indexed.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(indexed[0]))
        writer.writeheader()
        writer.writerows(indexed)
    assert sha(raw_csv) == raw_digest
    atomic_json(
        OUT / "ROLLOUT_INDEXING.json",
        dict(
            raw_csv_sha256=raw_digest,
            indexed_csv_sha256=sha(OUT / "rollout_per_step_indexed.csv"),
            initial_state_timestep=490,
            initial_state_has_no_input_transition=True,
            actual_input_timesteps=list(range(490, 440, -1)),
            conditions=5100,
            note="Raw initial-state input marker 491 is not an executed reverse step. Indexed CSV leaves it empty; all scientific metrics are unchanged.",
        ),
    )

    def mean(group, field):
        return group[field]["mean"]

    lines = [
        "# E014 frozen 50%-epoch reverse-field diagnostic — non-adjudicating",
        "",
        f"Checkpoint: `{identity['path']}`; SHA256 `{identity['sha256']}`. Epoch 1, update/cursor 90,413, exposures 115,067 (first complete batch crossing nominal 115,066).",
        "",
        f"Frozen development panel: 500 homology-clean VALIDATION identities, 100 per length stratum; identity SHA256 `{plan['panel_sha256']}`. Rollout subset: 100 identities, 20 per stratum; SHA256 `{rollout['subset_hash']}`. Selection was SHA256 ranking under namespace `e014-reverse-field-50pct-v1`, frozen before metrics.",
        "",
        "Training and corrected E012 screening continued concurrently. No training process, config, optimizer, scientific target, gate or historical artifact was changed. Only frozen weights were evaluated. No full de novo sampling or E007 evaluation/training was run.",
        "",
        f"True-v reverse identity: all 2,500 real conditions passed; maximum absolute normalized-coordinate error {metrics['true_v_identity_max_abs_error']:.9g}, frozen tolerance {plan['reverse_identity_max_abs_tolerance']:.9g}. Replay of the original per-example CUDA paired-noise v-MSE differed by at most {metrics['paired_noise_replay_max_v_mse_error']:.9g}, tolerance {plan['frozen_v_mse_replay_max_abs_tolerance']:.9g}.",
        "",
        "## Definitions and reference limits",
        "",
        "Reverse steps use the unchanged `CoordinateVPDiffusion.deterministic_reverse_step`, including masking and centering. Oracle predecessor uses `alpha_sigma` and the identical epsilon. Noise is regenerated with the frozen CUDA FP32 seed `noise_seed('e014-paired-validation-v1', sample_id, t)`. The five fixed one-step timesteps retain their original timestep-specific epsilon. The rollout starts with the t=490 epsilon and reuses it throughout its paired reference path.",
        "",
        "Primary errors are equal-protein raw centered coordinate-component MSE in normalized units. Multiply by 12.22820347644835² for Å². Relative gain is 1 minus the ratio of equal-protein model/no-op mean errors; per-protein gains and distributions are also retained. No-op is exactly unchanged X_t, not zero v. Cosines and norms flatten only valid components; nonassessable zero-vector cases are marked rather than fabricated. Geometry errors compare noisy prediction to the paired noisy predecessor, never to clean-protein validity ranges. Distance errors are Å²; Kabsch RMSD is a secondary proper-rotation comparison. Direct X0 collapse retains the historical Rg ratio <0.25 definition.",
        "",
        "The paired forward-noise path is one deterministic reference, not the unique valid generative trajectory. At extreme noise the individual clean structure can be unidentifiable; an MSE-trained conditional-mean v field need not recover each reference's v*. These comparisons cannot by themselves establish whether the learned marginal generative field is correct or whether full generation works.",
        "",
        "## One-step reverse field",
        "",
        "|t|Model MSE|No-op MSE|Relative gain|Step cosine|Step norm ratio|v cosine|v norm ratio|Direct X0 collapse|Direct X0 Rg ratio|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, group in metrics["by_timestep"].items():
        lines.append(
            f"|{t}|{mean(group, 'model_mse'):.8g}|{mean(group, 'noop_mse'):.8g}|{group['relative_gain_of_equal_protein_means']:.2%}|{mean(group, 'step_cosine'):.5f}|{mean(group, 'step_norm_ratio'):.5f}|{mean(group, 'v_cosine'):.5f}|{mean(group, 'v_norm_ratio'):.5f}|{group['direct_collapse_count']}/{group['denominator']}|{mean(group, 'direct_rg_ratio'):.5f}|"
        )
    for field in ("step_cosine", "step_norm_ratio", "v_cosine", "v_norm_ratio"):
        lines += [
            "",
            f"### {field} distribution",
            "",
            "|t|N|Mean|p05|p25|Median|p75|p95|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for t, group in metrics["by_timestep"].items():
            d = group[field]
            lines.append(
                f"|{t}|{d['denominator']}|"
                + "|".join(f"{d[k]:.6f}" for k in ("mean", "p05", "p25", "median", "p75", "p95"))
                + "|"
            )
    lines += [
        "",
        "### Direction fractions and v magnitudes",
        "",
        "|t|Step cos >0 / >.5 / >.9|v cos >0 / >.5 / >.9|Mean v norm: prediction / oracle|",
        "|---:|---|---|---:|",
    ]
    for t, group in metrics["by_timestep"].items():

        def fractions(field, current_group=group):
            d = current_group[field + "_positive_fractions"]
            return " / ".join(f"{d[str(k)]['count']}/{d['denominator']}" for k in (0, 0.5, 0.9))

        lines.append(
            f"|{t}|{fractions('step_cosine')}|{fractions('v_cosine')}|{mean(group, 'v_predicted_norm'):.6f} / {mean(group, 'v_target_norm'):.6f}|"
        )
    lines += [
        "",
        "### Noisy predecessor geometry",
        "",
        "|t|L1|L2|L3|All-pair distance MSE|Mean Rg ratio|Proper-Kabsch RMSD (Å)|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t, group in metrics["by_timestep"].items():
        lines.append(
            f"|{t}|"
            + "|".join(
                f"{mean(group, k):.8g}"
                for k in (
                    "L1_angstrom2",
                    "L2_angstrom2",
                    "L3_angstrom2",
                    "pair_distance_mse_angstrom2",
                    "rg_ratio",
                    "aligned_rmsd_angstrom",
                )
            )
            + "|"
        )
    lines += [
        "",
        "### Length summaries (all five timesteps)",
        "",
        "Each length stratum contains 100 identities / 500 protein-timestep conditions. Full timestep-by-length distributions are in metrics.json.",
        "",
        "|Length|Model MSE|No-op MSE|Gain|Step cosine|Step norm ratio|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for length, group in metrics["by_length"].items():
        lines.append(
            f"|{length}|{mean(group, 'model_mse'):.8g}|{mean(group, 'noop_mse'):.8g}|{group['relative_gain_of_equal_protein_means']:.2%}|{mean(group, 'step_cosine'):.5f}|{mean(group, 'step_norm_ratio'):.5f}|"
        )
    lines += [
        "",
        "### t=490 direct radius correlations",
        "",
        "All 500 direct estimates are collapsed, so binary collapse correlations are undefined. The following compare continuous Rg ratio with field calibration, over 500 identities.",
        "",
        "|Variable|Pearson|Spearman|",
        "|---|---:|---:|",
    ]
    for key, value in metrics["t490_correlations"].items():
        lines.append(f"|{key}|{value['pearson']:.6f}|{value['spearman']:.6f}|")
    lines += [
        "",
        "## Bounded 490→440 rollout",
        "",
        "Rollout inputs are exactly 490,...,441 (50 steps); outputs are 489,...,440. Teacher-forced and rollout steps are evaluated at each identical input timestep; both next states are compared with the same paired oracle predecessor. The exposure gap includes accumulated departure of the rollout input, and is not a pure isolated field-error attribution. `rollout_per_step_indexed.csv` marks the initial t=490 state with an empty input transition; the preserved raw CSV used a 491 initial-state marker, which was never executed. ROLLOUT_INDEXING.json records this metadata-only clarification and both hashes.",
        "",
        "|Output t|Steps|Raw MSE|Teacher-forced next-step MSE|Exposure gap|Ratio of mean errors|Rg ratio|Aligned RMSD (Å)|L1 / L2 / L3 (Å²)|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in (490, 489, 485, 480, 475, 465, 450, 440):
        group = rollout["by_output_timestep"][str(s)]
        ratio = f"{mean(group, 'rollout_mse') / (mean(group, 'forward_mse') + 1e-12):.3f}" if s < 490 else "—"
        lines.append(
            f"|{s}|{490 - s}|{mean(group, 'raw_mse'):.8g}|{mean(group, 'forward_mse'):.8g}|{mean(group, 'exposure_gap'):.8g}|{ratio}|{mean(group, 'rg_ratio'):.6f}|{mean(group, 'aligned_rmsd_angstrom'):.6f}|"
            + " / ".join(f"{mean(group, k):.6f}" for k in ("L1_angstrom2", "L2_angstrom2", "L3_angstrom2"))
            + "|"
        )
    outputs = list(range(489, 439, -1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = [490 - s for s in outputs]
    for key, label in (("raw_mse", "Model-owned path deviation"), ("forward_mse", "Teacher-forced next-step error")):
        axes[0].plot(steps, [mean(rollout["by_output_timestep"][str(s)], key) for s in outputs], label=label)
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Reverse steps from t=490", ylabel="Normalized coordinate-component MSE")
    axes[0].legend()
    axes[1].plot(steps, [mean(rollout["by_output_timestep"][str(s)], "rg_ratio") for s in outputs])
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set(xlabel="Reverse steps from t=490", ylabel="Rg rollout / paired noisy reference")
    fig.tight_layout()
    fig.savefig(OUT / "rollout_curve.png", dpi=150)
    fig.savefig(OUT / "rollout_curve.svg")
    plt.close(fig)
    lines += [
        "",
        "![Bounded rollout metrics](rollout_curve.png)",
        "",
        "## Execution record",
        "",
        f"Wall duration including summary-writer recovery: {identity['elapsed_seconds']:.1f} s. Continuation/rollout-process peak allocated/reserved: {identity['peak_allocated_bytes'] / 2**20:.1f}/{identity['peak_reserved_bytes'] / 2**20:.1f} MiB. The preceding one-step process peaked at 133.1/356.0 MiB (observed at its 500-identity heartbeat). CPU threads=1; CUDA FP32, TF32 disabled, no AMP or compile. GPU evaluation was chosen to preserve the exact frozen CUDA noise and inference kernel contract; bounded concurrent GPU contention was explicitly allowed.",
        "",
        "The first attempt completed and preserved the 2,500-row one-step CSV, then its summary writer raised a NumPy boolean-quantile TypeError. A float conversion and regression test fixed reporting only. `--continue-diagnostic` reverified checkpoint/data/protocol identities and hashes and reused that complete CSV unchanged before running rollouts. It never resumed or modified training. CONTINUATION_RECORD.json records the preserved CSV/identity hashes. Nine focused tests and lint pass.",
        "",
    ]
    before, after = identity["training_before"], identity["training_after"]
    elapsed = after["timestamp_unix"] - before["timestamp_unix"]
    lines += [
        f"Training heartbeat advanced from update {before['update']:,} / exposure {before['total_protein_exposures']:,} to update {after['update']:,} / exposure {after['total_protein_exposures']:,}, phase `{after['phase']}`. Counter-derived recent rate during this window: {(after['total_protein_exposures'] - before['total_protein_exposures']) / elapsed:.3f} proteins/s and {(after['update'] - before['update']) / elapsed:.3f} updates/s. This is a contention observation, not a controlled before/after throughput experiment. GPU snapshots before/after (utilization, total memory, power): `{identity['gpu_before']}` / `{identity['gpu_after']}`.",
        "",
        "Checkpoint hash and all model state tensors were verified unchanged after evaluation. All generated artifacts, temporary files and caches are D-resident. No sequence identity or target-derived feature entered the model; clean references were used exclusively to construct corruption and compute observational metrics. Existing E007/E010/E013/E014 evidence was preserved.",
        "",
        "## Interpretation",
        "",
        "Descriptive categories **R4 + R5 on the paired-reference diagnostic**: t=490 direction and magnitude are weak (mean step cosine 0.08617, norm ratio 0.09923; v cosine 0.08551, norm ratio 0.09952), with only 1.60% gain over no-op. Only 20/500 step cosines exceed 0.5 and 0/500 exceed 0.9. Lower/mid noise is substantially stronger: t=5/50/250/400 gains are 54.30%/70.58%/77.90%/54.88%. This does not identify an architectural failure, inadequate optimization or irreducible reference uncertainty as the cause.",
        "",
        "The answer has two parts. Universal direct X0 collapse (500/500, mean Rg ratio 0.11231) is not accompanied by radius collapse of these short noisy trajectories: their mean Rg ratio is 1.00067/1.00155/1.00242 after 10/25/50 steps, and all 5,100 states are finite. But the incremental paired-oracle direction/magnitude are not healthy enough to claim R1: trajectory MSE grows 0.00075640 → 0.00486720 → 0.02045440 while the corresponding teacher-forced next-step errors decrease 7.1653e-6 → 6.2945e-6 → 4.8665e-6. Exposure gaps grow 0.00074923 → 0.00486091 → 0.02044953. This is accumulated reference-path drift, not proof of a separate sampler-specific instability; extreme-noise teacher-forced displacement was already weak, so strict R3 is not established.",
        "",
        "Small raw next-state MSE alone is insufficient: neighboring diffusion states differ very little. Under the unchanged linear deterministic reverse helper, next-state prediction error scales with v error and the small adjacent noise-angle change. The no-op comparison prevents mistaking a small timestep for a well-calibrated incremental field.",
        "",
        "Continuous direct radius ratio correlates strongly with v norm calibration (Pearson/Spearman 0.98242/0.98223), but weakly with v direction (0.16920/0.18545). Association is not a causal branch attribution. At near-pure noise, conditional-mean denoising need not recover an individual unidentifiable clean reference or its paired displacement; these results therefore do not establish that the marginal unconditional reverse field is wrong. Full de novo quality remains untested.",
        "",
        "This is NOT final E014 adjudication. The required minimum two complete TRAIN epochs remains pending. No existing scientific gate, stopping rule, objective or subsequent sampler/de novo authorization was changed. Continue the frozen training protocol.",
        "",
    ]
    (OUT / "REPORT.md").open("x").write("\n".join(lines))
    atomic_json(
        OUT / "EVIDENCE_HASHES.json",
        {
            str(path): sha(path)
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "EVIDENCE_HASHES.json"
        },
    )


if __name__ == "__main__":
    main()
