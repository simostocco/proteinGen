"""Standalone figures, historical training audit and transparent C5 report."""

# ruff: noqa: I001, E501, F403, F405
from e007_e014_comparison_protocol import *
from collections import defaultdict

ALIASES = dict(
    zip(
        (*E007_LABELS, *E014_LABELS),
        ("E007 10k", "E007 9k historical selection", "E014 epoch 2", "E014 newest"),
        strict=True,
    )
)


def compute_audit():
    data = json.loads((OUT / "compute_comparison.json").read_text())
    data["complexity"] = (
        "E007 dense N² pair states; axial attention/triangle cubic compute; full bottleneck attention over (ceil(N/8))² pair-grid tokens has O((ceil(N/8))⁴) score storage/compute in the historical need_weights=True implementation. E014 O(N²) graph-construction storage / O(N² log N) sorting, O((k+6)N) sparse messages, O((ceil(N/8))²) coarse residue global graph. Neither total architecture is linear in N."
    )
    for r in data["results"]:
        r["e007_bottleneck_pair_tokens"] = ((r["N"] + 7) // 8) ** 2
        r["e007_bottleneck_token_pairs"] = ((r["N"] + 7) // 8) ** 4
        if "median_seconds" in r:
            r["synthetic_proteins_per_second"] = 1 / r["median_seconds"]
    for n in (64, 128, 256, 500):
        digests = {r["input_sha256"] for r in data["results"] if r["N"] == n and "input_sha256" in r}
        assert len(digests) == 1, (n, digests)
    data["matched_input_hashes_verified"] = True
    data["source_complexity_hashes"] = {
        name: sha(name)
        for name in (
            "src/protein_distance_diffusion/models/attention.py",
            "src/protein_distance_diffusion/models/pair_grid_unet_trunk.py",
            "src/protein_distance_diffusion/models/e013_unconditional_equivariant_denoiser.py",
        )
    }
    write("compute_comparison.json", data)
    n500 = next(
        r
        for r in data["results"]
        if r["checkpoint"] == E007_LABELS[0] and r["N"] == 500 and r["phase"] == "forward_backward"
    )
    write(
        "RESOURCE_ASSESSMENT.json",
        dict(
            fresh_N500_probe=n500,
            frozen_comfortable_reserved_bytes=5 * 2**30,
            comfortable=not n500.get("OOM", False) and n500["peak_reserved_bytes"] <= 5 * 2**30,
            actual_training_fits_certified=False,
            reason="Disposable-copy backward only; no Adam states/optimizer update or full training lifecycle. Frozen primary-final-report and resource-approval prerequisites remain unmet.",
            approved=False,
            training_launched=False,
        ),
    )


def training_audit():
    historical = [json.loads(line) for line in (HIST / "metrics.jsonl").read_text().splitlines()]
    byupdate = {r["global_optimizer_update"]: r for r in historical if r["global_optimizer_update"] <= 10000}
    bins = defaultdict(list)
    for r in byupdate.values():
        bins[(r["global_optimizer_update"] - 1) // 500].append(r)
    curve = []
    for group in sorted(bins):
        rows = sorted(bins[group], key=lambda r: r["global_optimizer_update"])
        counts = [dict(zip(STRATA, (4, 2, 1, 1, 1), strict=True))[r["length_stratum"]] for r in rows]
        # Retry/recovery may reset elapsed timers. Report only positive adjacent
        # intervals, without interpreting evaluation-inclusive time as GPU time.
        deltas = [
            r["elapsed_seconds"] - byupdate[r["global_optimizer_update"] - 1]["elapsed_seconds"]
            for r in rows
            if r["global_optimizer_update"] - 1 in byupdate
        ]
        curve.append(
            dict(
                model="E007",
                first_update=rows[0]["global_optimizer_update"],
                last_update=rows[-1]["global_optimizer_update"],
                exposures_end=rows[-1]["samples_processed"],
                logged_v_loss=float(np.average([r["coordinate_v_mse"] for r in rows], weights=counts)),
                updates=len(rows),
                positive_logged_wall_seconds=float(sum(t for t in deltas if t > 0)),
            )
        )
    prior = pq.read_table(E014 / "training_curve.parquet", use_threads=False).to_pylist()
    exposure = 0
    for r in prior:
        exposure += r["exposures"]
        curve.append(
            dict(
                model="E014",
                first_update=r["first_update"],
                last_update=r["last_update"],
                exposures_end=exposure,
                logged_v_loss=r["training_v_loss"],
                updates=r["updates"],
            )
        )
    table("training_efficiency.parquet", curve)
    records = []
    for label in E014_LABELS:
        cp = next(
            r for r in json.loads((E014 / "checkpoint_inventory.json").read_text())["selected"] if r["label"] == label
        )
        state = torch.load(cp.get("immutable_copy", cp["path"]), map_location="cpu", weights_only=False)
        records.append(
            dict(
                checkpoint=label,
                elapsed_seconds=state["elapsed_seconds"],
                gpu_seconds=state["gpu_seconds"],
                exposures=cp["exposures"],
                observed_proteins_per_wall_second=cp["exposures"] / state["elapsed_seconds"],
                scope="Cumulative trainer wall time includes validation/checkpoint/contention; CUDA-event time counts training GPU operations; not equal-hardware controlled throughput.",
            )
        )
        del state
    write(
        "training_efficiency.json",
        dict(
            matched_training_curves_exist=False,
            architecture_learning_speed_identifiable=False,
            reason="No common-exposure frozen validation checkpoints. Different streams, learning rates, exposure ranges and stopping rules prevent a learning-speed attribution.",
            e014_compute=records,
            e007_logged_updates=len(byupdate),
            e007_missing_logged_updates=sorted(set(range(1001, 10001)) - set(byupdate)),
            e007_total_training_gpu_seconds=None,
            e007_log_elapsed_max=max(r["elapsed_seconds"] for r in historical),
            e007_time_note="Historical cumulative elapsed logs include evaluation and allocator cleanup; timer/recovery resets prevent a certified total training GPU compute estimate.",
        ),
    )


def plots():
    import os

    assert str(Path(os.environ["MPLCONFIGDIR"]).resolve()).startswith(str(OUT))
    assert os.access(os.environ["MPLCONFIGDIR"], os.W_OK)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = OUT / "plots"
    destination.mkdir(exist_ok=True)
    raw = pq.read_table(OUT / "core_metrics.parquet", use_threads=False).to_pylist()
    high = pq.read_table(OUT / "high_noise_grid.parquet", use_threads=False).to_pylist()
    gauss = pq.read_table(OUT / "gaussian_prefix_metrics.parquet", use_threads=False).to_pylist()
    specifications = [
        ("v_mse_by_t", "v_mse", "Coordinate-v MSE"),
        ("local_mse_by_t", "local_mse_angstrom2", "Local distance MSE (Å²)"),
        ("torsion_by_t", "torsion_angular_mae_radians", "Torsion angular MAE (rad)"),
        ("inversion_rate_by_t", "triple_inversion_fraction", "Equal-protein inversion rate"),
        ("high_noise_v_cosine", "v_cosine", "Paired v cosine"),
        ("high_noise_step_gain", "step_gain", "Equal-protein paired step gain"),
    ]
    for filename, metric, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        rows = high if filename.startswith("high_noise") else raw
        times = sorted({r["timestep"] for r in rows})
        for label, title in ALIASES.items():
            ax.plot(
                times,
                [
                    np.mean(
                        [
                            r[metric]
                            for r in rows
                            if r["checkpoint"] == label and r["timestep"] == t and r[metric] is not None
                        ]
                    )
                    for t in times
                ],
                "o-",
                label=title,
                markersize=3,
            )
        ax.set(xlabel="Timestep", ylabel=ylabel, title="Matched evaluation; unmatched training (C5)")
        if metric == "step_gain":
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(destination / f"{filename}.png", dpi=180)
        plt.close(fig)
    for filename, metric, ylabel in (
        ("gaussian_prefix_energy", "model_ref", "34-descriptor energy distance"),
        ("gaussian_prefix_rg", "rg_ratio", "Noisy-state/reference Rg ratio"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for label, title in ALIASES.items():
            chosen = sorted(
                [
                    r
                    for r in gauss
                    if r["checkpoint"] == label and r["length_stratum"] == "pooled" and r["metric"] == "energy"
                ],
                key=lambda r: r["timestep"],
                reverse=True,
            )
            ax.plot([r["timestep"] for r in chosen], [r[metric] for r in chosen], "o-", label=title)
            if metric == "model_ref":
                ax.fill_between(
                    [r["timestep"] for r in chosen],
                    [r["model_ref_ci"][0] for r in chosen],
                    [r["model_ref_ci"][1] for r in chosen],
                    alpha=0.12,
                )
        if metric == "rg_ratio":
            ax.axhline(0.8, color="red", linestyle="--", label="Descriptive collapse threshold")
            ax.axhline(1, color="grey", linestyle=":", label="Reference scale")
        else:
            baseline = sorted(
                [
                    r
                    for r in gauss
                    if r["checkpoint"] == E007_LABELS[0] and r["length_stratum"] == "pooled" and r["metric"] == "energy"
                ],
                key=lambda r: r["timestep"],
                reverse=True,
            )
            ax.plot(
                [r["timestep"] for r in baseline],
                [r["ref_ref"] for r in baseline],
                "k:",
                label="Reference/reference floor",
            )
            ax.plot([r["timestep"] for r in baseline], [r["noop_ref"] for r in baseline], "k--", label="Held Gaussian")
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set(
            xlabel="Endpoint of identical Gaussian start at t=499",
            ylabel=ylabel,
            title="Bounded sampling only; minimum endpoint 350",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.savefig(destination / f"{filename}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("v_mse", "local_mse_angstrom2", "triple_inversion_fraction"), strict=True):
        for label, title in ALIASES.items():
            ax.plot(
                STRATA,
                [
                    np.mean(
                        [
                            r[metric]
                            for r in raw
                            if r["checkpoint"] == label and r["length_stratum"] == s and r[metric] is not None
                        ]
                    )
                    for s in STRATA
                ],
                "o-",
                label=title,
            )
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(destination / "performance_vs_length.png", dpi=180)
    plt.close(fig)


def mdtable(headers, rows):
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.7g}"
        if isinstance(x, list):
            return "[" + ", ".join(fmt(y) for y in x) + "]"
        return str(x)

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(fmt(x) for x in row) + " |" for row in rows],
        ]
    )


def report():
    data = json.loads((OUT / "bootstrap_intervals.json").read_text())
    summaries = json.loads((OUT / "paired_summaries.json").read_text())
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text())
    compute = json.loads((OUT / "compute_comparison.json").read_text())
    replay = json.loads((OUT / "replay_numerics.json").read_text())
    efficiency = json.loads((OUT / "parameter_efficiency.json").read_text())["records"]
    gaussian = pq.read_table(OUT / "gaussian_prefix_metrics.parquet", use_threads=False).to_pylist()
    marginal_rows = pq.read_table(OUT / "marginal_metrics.parquet", use_threads=False).to_pylist()

    def p(label, t, s="pooled"):
        return next(r for r in summaries[label] if r["timestep"] == t and r["length_stratum"] == s)

    lines = [
        "# E007 versus E014: frozen evaluation, historical training confound",
        "",
        "Primary classification **C5**. The existing E007 model is **M2 for architectural attribution**, and usable as a historical comparator. The models share the diffusion formulation, source data universe and normalization, but E007 has only 18,000 training exposures versus E014 epoch 2's 460,264. Matching evaluation tensors cannot correct this 25.57-fold training difference. No architecture or objective-dominance causal claim is allowed.",
        "",
        "The historical comparison shows a tradeoff: E014 epoch 2 reduces aggregate v-MSE by 9.78%, while increasing aggregate local-distance MSE by 35.18%. E014 improves low-noise geometry and torsion, but E007 has better local-distance geometry at t=250/400/490. Both approach chance-level chirality at mid/high noise. The primary t=400 inversion and torsion comparisons do not establish an E014 advantage after familywise uncertainty is considered. Gaussian-prefix energy favors E014 at t=400 and E007 at t=350 in point estimates, with wide paired intervals crossing zero at both endpoints. Both transport Gaussian noise toward the noisy-protein marginal without observed population-scale collapse; this does not establish full protein generation.",
        "",
        "The historical E007 fixed 10,000-update mandatory-review endpoint was selected before new inference. The previously published 9,000-update choice is retained as a sensitivity comparator. E014 epoch 2 is the formal minimum-exposure primary boundary; newest partial epoch 3 is secondary. None was selected using this comparison's performance.",
        "",
        "All model evaluation was read-only. Disposable-copy backward benchmarks did not create an optimizer or change model tensors. No retraining, sequence conditioning, target-derived inference input, full t=0 sampling or historical-report edits occurred. Outputs and runtime caches are on D:.",
        "",
        "## Checkpoint provenance",
        "",
        mdtable(
            ["checkpoint", "SHA256", "parameters", "updates", "exposures", "complete full-data epochs"],
            [
                [r["label"], r["sha256"], r["parameters"], r["updates"], r["exposures"], r["complete_full_data_epochs"]]
                for r in inventory["selected"]
            ],
        ),
        "",
        "E007 architecture: EquivariantPairCoordinateUNet, 16 RBF distance channels, unsigned chain separation, adjacency/continuity/pair masks, timestep+length conditioning, scalar pair U-Net with attention/triangle operations and final equivariant vector lift. Its O(3) reflection-inclusive representation has no persistent vector hidden state or parity-sensitive local head. E014: 256 residue scalars, 64 persistent vector channels, six sparse polymer+kNN blocks, three coarse global blocks and a parity-sensitive local branch of width 256. Both use raw weights, no EMA.",
        "",
        "Both use 12.22820347644835 Å normalization and identical 500-step cosine VP coordinate-v targets. Same eligible 230,132 TRAIN identities and homology-clean validation universe; different actual training exposure streams and validation/stopping rules. E007 AdamW learning rate 0.0003 versus E014 0.0002, weight decay 0.01 and common optimizer family. Historical training code commit was not recorded; it remains unknown rather than inferred from current Git history. Source/config/protected-input hashes are inventoried.",
        "",
        "Reconstructed E007 10k trained on 18,000 distinct identities (7.8216% of eligible TRAIN), with 8,000/4,000/2,000/2,000/2,000 exposures by increasing length stratum. E014 completed two passes over all 230,132 identities. Frozen rank-stream reconstruction matches the historical 1,000-update prefix and its persisted hashes; the actual identity coverage is recorded in historical_train_population.json. Training loss aggregation also differs: E007 pools valid coordinates within each batch, whereas E014 first averages within each protein. The shared v-target does not make these weighted objectives identical.",
        "",
        "## Statistical design",
        "",
        "Frozen 500 validation identities, 100 each in 20–64/65–128/129–256/257–384/385–500; panel hash 3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf. Frozen Gaussian subset 100/20 per stratum, hash c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0. Noise namespaces, schedule, normalization, masks and Gaussian initial states are identical. Model-free paired no-op values and Gaussian held-state/reference descriptors are checked across all comparators.",
        "",
        "Primary family has six predeclared statistics. Scalar metrics average timesteps within protein, then weight proteins equally; inversion primary is mean protein-level fraction, with pooled counts separately. Paired length-stratified bootstrap uses 5,000 draws and 95% intervals, plus Bonferroni 99.1667% intervals across six primary tests. Secondary scalar/distribution summaries use 500 draws and are descriptive. Distribution preprocessing/kernel/projections are reference-only and fixed during bootstrap; intervals are conditional on this panel and fit. Paired indices do not turn population distance into coordinate-paired distances.",
        "",
        mdtable(
            ["primary statistic", "E007 10k", "E014 epoch 2", "E014−E007", "95% paired CI", "99.1667% familywise CI"],
            [
                [r["metric"], r["e007"], r["e014"], r["difference"], r["ci95"], r["ci_familywise"]]
                for r in data["primary"]
            ],
        ),
        "",
        "## Core denoising and direct-X0 diagnostic",
        "",
        mdtable(
            [
                "checkpoint",
                "aggregate v-MSE",
                "aggregate local MSE (Å²)",
                "fractional v-MSE reduction/parameter million",
            ],
            [
                [
                    ALIASES[r["checkpoint"]],
                    r["v_mse"],
                    r["local_mse"],
                    r["fractional_v_mse_reduction_per_million_parameters"],
                ]
                for r in efficiency
            ],
        ),
        "",
    ]
    for field, title in (
        ("v_mse", "Coordinate-v MSE"),
        ("v_cosine", "v cosine"),
        ("v_norm_ratio", "v norm ratio"),
        ("x0_mse_angstrom2", "Centered reconstruction component MSE (Å²)"),
        ("aligned_rmsd_angstrom", "Proper Kabsch RMSD (Å)"),
        ("radius_ratio", "Direct X0 Rg ratio"),
        ("covariance_trace_ratio", "Direct X0 trace ratio"),
    ):
        lines += [
            f"### {title}",
            "",
            mdtable(
                ["checkpoint", *map(str, CORE)],
                [[ALIASES[label], *[p(label, t)[field]["mean"] for t in CORE]] for label in ALIASES],
            ),
            "",
        ]
    lines += [
        "Direct X0 collapse is diagnostic of the clean conditional-mean estimator. It is not collapse of the actual noisy rollout state.",
        "",
        mdtable(
            ["checkpoint", "t", "direct collapse/500"],
            [[ALIASES[label], t, p(label, t)["direct_collapse_count"]] for label in ALIASES for t in CORE],
        ),
        "",
        "## Local geometry and chirality",
        "",
    ]
    for field, title in (
        ("offset_1_mse_angstrom2", "L1 distance MSE (Å²)"),
        ("offset_2_mse_angstrom2", "L2 distance MSE (Å²)"),
        ("offset_3_mse_angstrom2", "L3 distance MSE (Å²)"),
        ("local_mse_angstrom2", "Mean local MSE (Å²)"),
        ("torsion_mse", "Torsion sin/cos sum-square error"),
        ("torsion_angular_mae_radians", "Torsion angular MAE (rad)"),
        ("triple_product_mse", "Normalized signed triple-product MSE"),
    ):
        lines += [
            f"### {title}",
            "",
            mdtable(
                ["checkpoint", *map(str, CORE)],
                [[ALIASES[label], *[p(label, t)[field]["mean"] for t in CORE]] for label in ALIASES],
            ),
            "",
        ]
    lines += [
        mdtable(
            ["checkpoint", "t", "torsions assessed/potential", "inversions/sign assessed", "pooled inversion rate"],
            [
                [
                    ALIASES[label],
                    t,
                    f"{p(label, t)['torsion_assessed_sum']}/{p(label, t)['torsion_potential_count_sum']}",
                    f"{p(label, t)['triple_inversion_count_sum']}/{p(label, t)['triple_sign_assessed_sum']}",
                    p(label, t)["triple_inversion_rate_pooled"],
                ]
                for label in ALIASES
                for t in CORE
            ],
        ),
        "",
        "## Dense high-noise field",
        "",
        "Paired gain is 1−mean(model next-step MSE)/mean(no-op MSE). Individual protein gain and paired bootstrap deltas are also retained; the two aggregates need not agree. Paired direction is not proof of a unique unconditional reverse trajectory.",
        "",
        "Epoch-2 E014 has better paired gain at extreme noise, especially t=492–496, while its v cosine is similar to E007 there. Its smaller prediction norm suggests improved amplitude calibration rather than evidence of a consistently extended directional-information front. The newest E014 checkpoint is nonmonotonic on this grid. Neither these paired gains nor unresolved marginal effects establish a new high-noise generative resolution boundary.",
        "",
        mdtable(
            [
                "checkpoint",
                "t",
                "alpha",
                "sigma",
                "SNR",
                "logSNR",
                "v cosine",
                "v norm ratio",
                "paired step gain",
                "step cosine",
                "step norm ratio",
            ],
            [
                [
                    ALIASES[label],
                    t,
                    *[p(label, t)[k]["mean"] for k in ("alpha", "sigma", "snr", "log_snr", "v_cosine", "v_norm_ratio")],
                    p(label, t)["step_gain_of_means"],
                    *[p(label, t)[k]["mean"] for k in ("step_cosine", "step_norm_ratio")],
                ]
                for label in ALIASES
                for t in HIGH
            ],
        ),
        "",
        "## Marginal reverse effects",
        "",
        "Validated 34 reflection-invariant descriptors; q_t source→model→t−1 versus independent q_(t−1). Reference A/B floor, no-op baseline, raw model-minus-no-op effects and E014-minus-E007 paired intervals are retained. The earlier true-v positive control resolved only to t=50; high-noise nonsignificance is not equivalence. These descriptors do not measure chirality, topology or the full joint coordinate law.",
        "",
        mdtable(
            [
                "t",
                "metric",
                "ref-ref floor",
                "no-op/ref",
                "E007 10k/ref",
                "E014 epoch 2/ref",
                "E014−E007",
                "95% paired CI",
                "E007−no-op",
                "E014−no-op",
            ],
            [
                [
                    r["timestep"],
                    r["metric"],
                    r["ref_ref"],
                    r["noop_ref"],
                    r["e007"],
                    r["e014"],
                    r["difference"],
                    r["ci95"],
                    r["e007_minus_noop"],
                    r["e014_minus_noop"],
                ]
                for r in data["distribution_comparisons"]
                if r["phase"] == "marginal_groups"
                and r["length_stratum"] == "pooled"
                and r["e007_checkpoint"] == E007_LABELS[0]
                and r["e014_checkpoint"] == E014_LABELS[0]
                and r["metric"] in ("energy", "mmd2", "sliced_wasserstein", "rg_wasserstein", "covariance_spectrum_w1")
            ],
        ),
        "",
        "## Gaussian-prefix transport and actual noisy-state collapse",
        "",
        "Both models start at t=499 from identical centered Gaussian states. Endpoints 490/480/470/460/450/440/420/400/350 only. Descriptive population collapse is Rg/reference<0.8 OR covariance-trace/reference<0.8, checked pooled and in each stratum. Coordinate RMS ratio is sqrt(trace ratio). Scale retention alone does not establish protein-generation success.",
        "",
        mdtable(
            [
                "checkpoint",
                "endpoint",
                "energy",
                "95% CI",
                "Rg ratio",
                "trace ratio",
                "RMS ratio",
                "pair mean ratio",
                "finite",
                "scale collapse",
            ],
            [
                [
                    ALIASES[r["checkpoint"]],
                    r["timestep"],
                    r["model_ref"],
                    r["model_ref_ci"],
                    r["rg_ratio"],
                    r["covariance_trace_ratio"],
                    r["coordinate_rms_ratio"],
                    r["pair_mean_ratio"],
                    r["finite"],
                    r["noisy_state_collapse"],
                ]
                for r in gaussian
                if r["length_stratum"] == "pooled" and r["metric"] == "energy"
            ],
        ),
        "",
        mdtable(
            ["endpoint", "metric", "E007 10k", "E014 epoch 2", "E014−E007", "95% paired CI"],
            [
                [r["timestep"], r["metric"], r["e007"], r["e014"], r["difference"], r["ci95"]]
                for r in data["distribution_comparisons"]
                if r["phase"] == "gaussian_groups"
                and r["length_stratum"] == "pooled"
                and r["e007_checkpoint"] == E007_LABELS[0]
                and r["e014_checkpoint"] == E014_LABELS[0]
                and r["metric"] in ("energy", "mmd2", "sliced_wasserstein", "rg_wasserstein", "covariance_spectrum_w1")
            ],
        ),
        "",
        "## Length dependence",
        "",
        mdtable(
            [
                "checkpoint",
                "stratum",
                "t",
                "v-MSE",
                "L1",
                "L2",
                "L3",
                "torsion angular",
                "pooled inversion rate",
                "direct collapse/100",
            ],
            [
                [
                    ALIASES[label],
                    s,
                    t,
                    *[
                        p(label, t, s)[k]["mean"]
                        for k in (
                            "v_mse",
                            "offset_1_mse_angstrom2",
                            "offset_2_mse_angstrom2",
                            "offset_3_mse_angstrom2",
                            "torsion_angular_mae_radians",
                        )
                    ],
                    p(label, t, s)["triple_inversion_rate_pooled"],
                    p(label, t, s)["direct_collapse_count"],
                ]
                for label in ALIASES
                for s in STRATA
                for t in CORE
            ],
        ),
        "",
        "Full length-specific Gaussian/marginal metrics, all pair summaries and paired intervals are in the corresponding parquets and bootstrap_intervals.json. Pooled inversion counts are residue-weighted diagnostics; primary/length scalar estimates remain equal-protein weighted.",
        "",
        "## Parameter and computational fairness",
        "",
        "E014 has 32.53% fewer parameters than E007. Fresh cost benchmarks use identical N=64/128/256/500 inputs, batch 1, t=250, FP32, one CPU thread, TF32/AMP/compile off, three warmups and five measured repeats. Forward+backward runs on disposable copies, without Adam states or optimizer updates. Peak reserved memory depends on allocator behavior; these timings are device/host conditions, not a scaling law. Other sequence work was left untouched and may contribute host contention.",
        "",
        mdtable(
            ["checkpoint", "N", "phase", "median ms", "p95 ms", "allocated GiB", "reserved GiB"],
            [
                [
                    ALIASES[r["checkpoint"]],
                    r["N"],
                    r["phase"],
                    r.get("median_seconds", float("nan")) * 1000,
                    r.get("p95_seconds", float("nan")) * 1000,
                    r.get("peak_allocated_bytes", 0) / 2**30,
                    r.get("peak_reserved_bytes", 0) / 2**30,
                ]
                for r in compute["results"]
            ],
        ),
        "",
        compute["complexity"],
        "",
        "Descriptive parameter-normalized view: divide a reported error/distance by 7.586505M (E007) or 5.119069M (E014); this arithmetic is explicitly not an efficiency ranking or scaling-law estimate. Lower error per parameter is ambiguous. Raw performance and measured memory/latency are the scientifically useful comparisons.",
        "",
        "## Training efficiency and matched-control decision",
        "",
        "No matched-exposure validation curves exist. Historical loss-versus-exposure records are saved in training_efficiency.parquet; learning faster per exposure cannot be established because training schedules, exposure ranges and validation designs differ. training_efficiency.json records available wall/GPU time and distinguishes missing E007 total GPU compute from E014 checkpoint-recorded CUDA-event time.",
        "",
        "The predeclared strict E007 control is prepared with unchanged representation, exact 230,132-identity TRAIN schedule, shared normalization/diffusion/v-target/batches/AdamW settings/cadence, and two-complete-epoch 460,264-exposure minimum. Parameter count remains 7,586,505. MATCHED_CONTROL_PREPARATION.json records this specification. The frozen runner requires e014/FINAL_REPORT.json and MATCHED_CONTROL_RESOURCE_APPROVED.json; both are absent. Minimum-exposure adjudication does not satisfy the separate frozen final-stop trigger. The prior E007 N=500 reserved-memory probe also exceeded the 5 GiB comfortable-training envelope. No gate was fabricated or E014 changed; no control training was launched.",
        "",
        "## Interpretation and sampling gate",
        "",
        "C5 is the primary scientific outcome regardless of the descriptive performance deltas: unmatched exposure/data scheduling prevents attributing differences to architecture. Similar high-noise torsion/inversion failures would support a common geometry/information bottleneck as a hypothesis, not prove diffusion-objective causality. A parity-sensitive branch has a measurable advantage only where the paired chirality evidence supports it; ~50% inversion at high noise does not establish useful chirality.",
        "",
        "E014 epoch 2 and newest remain Case E; full unconditional protein-generation success is unestablished. The unchanged Case A-only StageD gate remains closed. This comparison does not authorize full t=0 sampling. A fully matched control remains necessary for architecture attribution once frozen prerequisites are satisfied.",
        "",
        "## Numerical repeatability",
        "",
        "E014's preserved descriptor repeatability envelope is ~8.96e−6 against unchanged 5e−6 tolerance; exact replay remains uncertified. This archived ten-repeat test concerns the epoch-1 50% checkpoint at 6nof_A/440, and does not certify epoch 2 or newest. Its measured population-distance numerical effects were below 3.12e−8, far below the primary effects and bootstrap uncertainty. Original replay record and hash are embedded without alteration. Current E007 ten-repeat prefixes at 6nof_A/440 are reported separately; current repeatability is not exact historical certification because no matching historical prefix record exists.",
        "",
        mdtable(
            [
                "E007 checkpoint",
                "coordinate envelope",
                "descriptor envelope",
                "tolerance",
                "current repeatability certified",
                "initial states identical",
            ],
            [
                [
                    ALIASES[r["checkpoint"]],
                    r["coordinate_envelope"],
                    r["descriptor_envelope"],
                    replay["tolerance"],
                    r["repeatability_certified"],
                    r["initial_identical"],
                ]
                for r in replay["e007"]
            ],
        ),
        "",
        "Scientific effect sizes and uncertainty must be interpreted separately from exact replay certification. No tolerance was relaxed.",
        "",
        "## Artifacts and preservation",
        "",
        "All requested parquets, checkpoint_inventory.json, matching_assessment.json, compute_comparison.json, bootstrap_intervals.json, replay_numerics.json and nine figures are resident under this D: root. Protected historical inputs and sequence main ref are verified at closure. New comparison code/tests/documentation are separate from E014/E007 source and reports.",
        "",
    ]
    # Keep raw tables exhaustive; concise evidence-based interpretation is added
    # after inspecting computed results, without altering historical documents.
    (OUT / "REPORT.md").write_text("\n".join(lines))
    write(
        "REPORT_COUNTS.json",
        dict(
            gaussian_rows=len(gaussian),
            marginal_rows=len(marginal_rows),
            primary=len(data["primary"]),
            secondary=len(data["secondary"]),
        ),
    )


if __name__ == "__main__":
    compute_audit()
    training_audit()
    plots()
    report()
