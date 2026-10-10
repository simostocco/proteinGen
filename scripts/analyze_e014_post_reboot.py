"""Scientific comparisons from frozen post-reboot inference and historical records."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

import analyze_e014_marginal_reverse as marginal
from analyze_e014_high_noise import paired_summary
from evaluate_e014_post_reboot import OLD, OUT, SELECTED
from protein_distance_diffusion.evaluation.e014_evaluation import adjudicate_final, aggregate
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    GRID,
    HIGH,
    STRATA,
    collapse_onset,
    marginal_status,
    noisy_scale_status,
    resolution_front,
    schedule_values,
)
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha

CORE = (5, 50, 250, 400, 490)
EARLY = ("initialization", "epoch_1_fraction_0.25", "epoch_1_fraction_0.50", "epoch_1_fraction_0.75")
ALL = (*EARLY, *SELECTED)
ALIASES = dict(zip(ALL, ("init", "25%", "50%", "75%", "epoch 1", "epoch 2", "newest"), strict=True))


def write_table(name, rows):
    path = OUT / name
    partial = path.with_name(path.name + ".partial")
    pq.write_table(pa.Table.from_pylist(rows), partial, compression="zstd")
    assert pq.read_metadata(partial).num_rows == len(rows)
    partial.replace(path)


def analyze(label):
    assert label in SELECTED
    target = OUT / f"{label}.ANALYSIS_COMPLETE.json"
    if target.exists():
        return
    assert (OUT / f"{label}.INFERENCE_COMPLETE.json").exists()
    marginal.OUT = OUT
    groups = marginal.analyze_phase("one_step", f"{label}.descriptors.parquet")
    gaussian = marginal.analyze_phase("gaussian_prefix", f"{label}.gaussian_descriptors.parquet")
    for phase, rows in (("one_step", groups), ("gaussian_prefix", gaussian)):
        for g in rows:
            g["model_bootstrap_draws"] = g.pop("_model_draws")
            raw = g["raw_descriptor_distributions"]
            rg = raw["model"]["rg"]["mean"] / raw["ref_b"]["rg"]["mean"]
            trace = raw["model"]["trace"]["mean"] / raw["ref_b"]["trace"]["mean"]
            g.update(
                checkpoint=label,
                rg_mean_ratio=rg,
                trace_mean_ratio=trace,
                **schedule_values(CoordinateVPDiffusion(500), g["timestep"]),
            )
            if phase == "one_step":
                g["resolution_status"] = marginal_status(g["metrics"], rg, trace)
            else:
                g["noisy_scale_status"] = noisy_scale_status(rg, trace)
    raw = pq.read_table(OUT / f"{label}.paired.parquet", use_threads=False).to_pylist()
    paired = []
    for t in GRID:
        for stratum in (*STRATA, "pooled"):
            chosen = [r for r in raw if r["timestep"] == t and (stratum == "pooled" or r["length_stratum"] == stratum)]
            paired.append(dict(checkpoint=label, timestep=t, length_stratum=stratum, **paired_summary(chosen)))
    atomic_json(
        target,
        dict(
            checkpoint=label,
            paired_summaries=paired,
            marginal_groups=groups,
            gaussian_groups=gaussian,
            source_paired_sha256=sha(OUT / f"{label}.paired.parquet"),
            source_descriptor_sha256=sha(OUT / f"{label}.descriptors.parquet"),
        ),
    )


def training_audit(latest_update):
    # Keep the last logged occurrence of each update and discard uncommitted tail.
    committed = {}
    malformed = []
    tail = 0
    tail_exposures = 0
    last_logged_update = 0
    paths = sorted((ROOT / "e014").glob("*TRAINING*.jsonl"))
    for path in paths:
        with path.open() as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    malformed.append(dict(path=str(path), line=line_no))
                    continue
                if r["update"] <= latest_update:
                    committed[r["update"]] = r
                else:
                    tail += 1
                    tail_exposures += r["identity_count"]
                last_logged_update = max(last_logged_update, r["update"])
    assert set(committed) == set(range(1, latest_update + 1))
    newest = json.loads((OUT / "checkpoint_inventory.json").read_text())["newest"]
    assert sum(r["identity_count"] for r in committed.values()) == newest["exposures"]
    assert sum(r["clipped"] for r in committed.values()) == round(newest["clipping_fraction"] * latest_update)
    bins = defaultdict(list)
    telemetry = []
    for u in sorted(committed):
        r = committed[u]
        bins[(r["epoch"], (u - 1) // 5000)].append(r)
        for branch, values in r.get("module_statistics", {}).items():
            telemetry.append(dict(update=u, epoch=r["epoch"], branch=branch, **values))
    summary = []
    for (epoch, _), rows in bins.items():
        weights = np.array([r["identity_count"] for r in rows])
        summary.append(
            dict(
                epoch=epoch,
                first_update=rows[0]["update"],
                last_update=rows[-1]["update"],
                updates=len(rows),
                exposures=int(weights.sum()),
                training_v_loss=float(np.average([r["loss"] for r in rows], weights=weights)),
                gradient_norm_mean=float(np.mean([r["gradient_norm"] for r in rows])),
                gradient_norm_p95=float(np.quantile([r["gradient_norm"] for r in rows], 0.95)),
                clipping_fraction=float(np.mean([r["clipped"] for r in rows])),
            )
        )
    result = dict(
        committed_updates=len(committed),
        logged_uncommitted_records=tail,
        logged_uncommitted_exposures=tail_exposures,
        last_logged_update=last_logged_update,
        malformed_lines=malformed,
        bins=summary,
        training_geometry_logged=False,
        explanation="Training objective logs v-loss/gradient/clipping; structural training metrics were not logged. Frozen validation geometry is reported at every durable quarter boundary.",
    )
    write_table("training_curve.parquet", summary)
    write_table("optimizer_telemetry.parquet", telemetry)
    atomic_json(OUT / "training_audit.json", result)
    return result


def finish():
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text())
    prior_hashes = json.loads((OLD / "FINAL_EVIDENCE_HASHES.json").read_text())["files"]
    reused_files = [f"{label}.{suffix}" for label in EARLY for suffix in ("paired.parquet", "ANALYSIS_COMPLETE.json")]
    reused_files += ["shared_reference_descriptors.parquet", "PROTOCOL.json"]
    for name in reused_files:
        assert sha(OLD / name) == prior_hashes[name]["sha256"], name
    atomic_json(
        OUT / "HISTORICAL_EVIDENCE_VERIFICATION.json",
        dict(files=reused_files, matches_pre_shutdown_evidence_hashes=True),
    )
    results = [
        json.loads(((OLD if label in EARLY else OUT) / f"{label}.ANALYSIS_COMPLETE.json").read_text()) for label in ALL
    ]
    raw_by_label = {
        label: pq.read_table(
            (OLD if label in EARLY else OUT) / f"{label}.paired.parquet", use_threads=False
        ).to_pylist()
        for label in ALL
    }
    core_replay = []
    for label in SELECTED[:2]:
        historical = json.loads((ROOT / f"e014/evaluation/{label}_development.metrics.json").read_text())
        fresh = {(r["sample_id"], r["timestep"]): r for r in raw_by_label[label]}
        for field in (
            "v_mse",
            "x0_mse_angstrom2",
            "aligned_rmsd_angstrom",
            "local_mse_angstrom2",
            "torsion_mse",
            "torsion_angular_mae_radians",
            "triple_product_mse",
            "triple_inversion_fraction",
            "radius_ratio",
            "clash_fraction",
            "local_global_norm_ratio",
            "local_gate_mean",
            "frame_reliability_mean",
        ):
            valid = [r for r in historical if r.get(field) is not None]
            diffs = [abs(fresh[r["sample_id"], r["timestep"]][field] - r[field]) for r in valid]
            ix = int(np.argmax(diffs))
            angstrom = "angstrom" in field
            passes = (
                all(
                    np.isclose(fresh[r["sample_id"], r["timestep"]][field], r[field], atol=1e-4, rtol=1e-5)
                    for r in valid
                )
                if angstrom
                else max(diffs) <= 5e-6
            )
            core_replay.append(
                dict(
                    checkpoint=label,
                    metric=field,
                    maximum_absolute_error=max(diffs),
                    mean_absolute_error=float(np.mean(diffs)),
                    maximum_timestep_mean_difference=max(
                        abs(
                            float(
                                np.mean(
                                    [
                                        fresh[r["sample_id"], r["timestep"]][field] - r[field]
                                        for r in valid
                                        if r["timestep"] == t
                                    ]
                                )
                            )
                        )
                        for t in CORE
                    ),
                    maximum_at=dict(sample_id=valid[ix]["sample_id"], timestep=valid[ix]["timestep"]),
                    passed=bool(passes),
                    tolerance="angstrom atol1e-4/rtol1e-5" if angstrom else "absolute5e-6; unchanged",
                    conditions=len(valid),
                )
            )
    replay_path = OUT / "replay_numerics.json"
    replay = json.loads(replay_path.read_text())
    replay.update(
        core_metric_replay=core_replay,
        historical_outlier_behavior_reproduced=any(
            a > 5e-6 and b <= 5e-6
            for a, b in zip(replay["old_replay_errors"], replay["outlier_replay_errors"], strict=True)
        ),
        nondeterminism_observed=True,
        exact_replay_certified=False,
    )
    atomic_json(replay_path, replay)
    # Include every training boundary in the denoising/structure learning curves.
    curve_rows = []
    for path in sorted((ROOT / "e014/evaluation").glob("*development.metrics.json")):
        label = path.name.removesuffix("_development.metrics.json")
        if label in ALL:
            rows = [r for r in raw_by_label[label] if r["timestep"] in CORE]
        else:
            rows = json.loads(path.read_text())
        cp = next(r for r in inventory["checkpoints"] if r["label"] == label)
        for r in rows:
            curve_rows.append(
                dict(
                    checkpoint=label,
                    updates=cp["updates"],
                    exposures=cp["exposures"],
                    **{k: v for k, v in r.items() if k != "checkpoint"},
                )
            )
    cp = inventory["newest"]
    curve_rows.extend(
        dict(updates=cp["updates"], exposures=cp["exposures"], **r)
        for r in raw_by_label[SELECTED[-1]]
        if r["timestep"] in CORE
    )
    # PyArrow infers columns from the first row; use the union explicitly so
    # newly collected cosine/norm fields are retained for historical curves.
    keys = sorted(set().union(*(r.keys() for r in curve_rows)))
    curve_rows = [{k: r.get(k) for k in keys} for r in curve_rows]
    write_table("core_metrics.parquet", curve_rows)
    dense = [r for label in ALL for r in raw_by_label[label] if r["timestep"] in HIGH]
    write_table("high_noise_grid.parquet", dense)
    branch_keys = [
        "checkpoint",
        "sample_id",
        "length",
        "length_stratum",
        "timestep",
        "global_output_norm",
        "local_effective_output_norm",
        "local_global_norm_ratio",
        "global_local_cosine",
        "local_gate_mean",
        "local_gate_median",
        "frame_reliability_mean",
        "frame_reliability_median",
        "frame_assessable_count",
        "frame_residue_count",
        "frame_assessable_fraction",
    ]
    write_table(
        "branch_telemetry.parquet", [{k: r[k] for k in branch_keys} for label in ALL for r in raw_by_label[label]]
    )
    paired = [g for c in results for g in c["paired_summaries"]]
    marginals = [g for c in results for g in c["marginal_groups"]]
    gaussian = [g for c in results for g in c["gaussian_groups"]]
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
                    rg_model_mean=g["raw_descriptor_distributions"]["model"]["rg"]["mean"],
                    rg_reference_mean=g["raw_descriptor_distributions"]["ref_b"]["rg"]["mean"],
                    covariance_trace_model_mean=g["raw_descriptor_distributions"]["model"]["trace"]["mean"],
                    covariance_trace_reference_mean=g["raw_descriptor_distributions"]["ref_b"]["trace"]["mean"],
                    pair_distance_model_mean=g["raw_descriptor_distributions"]["model"]["pair_mean"]["mean"],
                    pair_distance_reference_mean=g["raw_descriptor_distributions"]["ref_b"]["pair_mean"]["mean"],
                    resolution_status=g.get("resolution_status"),
                    model_minus_noop=m["model_ref"] - m["noop_ref"],
                    **m,
                )
            )
    write_table("marginal_metrics.parquet", [r for r in flat if r["phase"] == "one_step"])
    write_table("gaussian_prefix_metrics.parquet", [r for r in flat if r["phase"] == "gaussian_prefix"])
    # Shared reference preprocessing must be exactly invariant across checkpoints.
    for phase, gs in (("one_step", marginals), ("gaussian", gaussian)):
        seen = {}
        for g in gs:
            key = g["timestep"], g["length_stratum"]
            current = [
                g["reference_preprocessing"],
                g["bandwidth"],
                g["descriptor_columns"],
                g["raw_descriptor_distributions"]["ref_a"],
                g["raw_descriptor_distributions"]["ref_b"],
            ]
            if key in seen:
                assert current == seen[key], (phase, key)
            seen[key] = current
    initial = dict(groups=aggregate([r for r in raw_by_label[EARLY[0]] if r["timestep"] in CORE]))
    decisions = {
        label: adjudicate_final(
            initial, dict(groups=aggregate([r for r in raw_by_label[label] if r["timestep"] in CORE])), 2, cfg
        )
        for label in SELECTED[1:]
    }
    geometry = []
    for old, new in zip((EARLY[-1], *SELECTED[:-1]), SELECTED, strict=True):
        oldmap = {(r["sample_id"], r["timestep"]): r for r in raw_by_label[old]}
        for t in CORE:
            for stratum in (*STRATA, "pooled"):
                rows = sorted(
                    [
                        r
                        for r in raw_by_label[new]
                        if r["timestep"] == t and (stratum == "pooled" or r["length_stratum"] == stratum)
                    ],
                    key=lambda r: (STRATA.index(r["length_stratum"]), r["sample_id"]),
                )
                for field in (
                    "offset_1_mse_angstrom2",
                    "offset_2_mse_angstrom2",
                    "offset_3_mse_angstrom2",
                    "local_mse_angstrom2",
                ):
                    delta = np.array([r[field] - oldmap[r["sample_id"], t][field] for r in rows])
                    rng = np.random.default_rng(noise_seed("e014-high-noise-geometry-bootstrap-v1", stratum, t))
                    sizes = [100] * 5 if stratum == "pooled" else [100]
                    bounds = np.cumsum([0, *sizes])
                    draws = [
                        float(
                            delta[
                                np.concatenate(
                                    [
                                        rng.integers(lo, hi, size=hi - lo)
                                        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
                                    ]
                                )
                            ].mean()
                        )
                        for _ in range(500)
                    ]
                    oldmean = float(np.mean([oldmap[r["sample_id"], t][field] for r in rows]))
                    geometry.append(
                        dict(
                            old=old,
                            new=new,
                            timestep=t,
                            length_stratum=stratum,
                            metric=field,
                            old_mean=oldmean,
                            new_mean=oldmean + float(delta.mean()),
                            delta=float(delta.mean()),
                            relative_improvement=-float(delta.mean()) / oldmean,
                            ci=np.quantile(draws, [0.025, 0.975]).tolist(),
                        )
                    )
    write_table("geometry_epoch_deltas.parquet", geometry)
    onsets = [
        dict(
            checkpoint=label,
            length_stratum=s,
            **collapse_onset([g for g in paired if g["checkpoint"] == label and g["length_stratum"] == s]),
        )
        for label in ALL
        for s in (*STRATA, "pooled")
    ]
    fronts = [
        dict(
            checkpoint=label,
            length_stratum=s,
            **resolution_front([g for g in marginals if g["checkpoint"] == label and g["length_stratum"] == s]),
        )
        for label in ALL
        for s in (*STRATA, "pooled")
    ]
    effects = []
    for t in HIGH:
        for name in ("energy", "mmd2", "sliced_wasserstein", "rg_wasserstein", "covariance_spectrum_w1"):
            ms = [
                next(
                    m
                    for g in marginals
                    if g["checkpoint"] == label and g["timestep"] == t and g["length_stratum"] == "pooled"
                    for m in g["metrics"]
                    if m["metric"] == name
                )
                for label in (EARLY[2], EARLY[3], *SELECTED)
            ]
            values = [m["model_ref"] - m["noop_ref"] for m in ms]
            effects.append(
                dict(
                    timestep=t,
                    metric=name,
                    checkpoints=[EARLY[2], EARLY[3], *SELECTED],
                    model_minus_noop=values,
                    ci=[m["model_minus_noop_ci"] for m in ms],
                    decreases_each_epoch=all(a >= b for a, b in zip(values[-3:-1], values[-2:], strict=True)),
                )
            )
    epoch_effect_deltas = []
    for phase, groups in (("one_step", marginals), ("gaussian_prefix", gaussian)):
        selected_groups = {
            (g["checkpoint"], g["timestep"], g["length_stratum"]): g for g in groups if g["checkpoint"] in SELECTED
        }
        for old, new in zip(SELECTED[:-1], SELECTED[1:], strict=True):
            for (label, t, stratum), newer in selected_groups.items():
                if label != new:
                    continue
                older = selected_groups[old, t, stratum]
                assert older["reference_preprocessing"] == newer["reference_preprocessing"]
                for metric in newer["model_bootstrap_draws"]:
                    delta = np.array(newer["model_bootstrap_draws"][metric]) - np.array(
                        older["model_bootstrap_draws"][metric]
                    )
                    old_point = next(m["model_ref"] for m in older["metrics"] if m["metric"] == metric)
                    new_point = next(m["model_ref"] for m in newer["metrics"] if m["metric"] == metric)
                    epoch_effect_deltas.append(
                        dict(
                            phase=phase,
                            old=old,
                            new=new,
                            timestep=t,
                            length_stratum=stratum,
                            metric=metric,
                            model_distance_delta=new_point - old_point,
                            ci=np.quantile(delta, [0.025, 0.975]).tolist(),
                            bootstraps=500,
                            pairing="identical frozen identities/reference fits/bootstrap seeds across checkpoints",
                        )
                    )
    write_table("distribution_epoch_deltas.parquet", epoch_effect_deltas)
    all_core_groups = []
    for label in sorted({r["checkpoint"] for r in curve_rows}):
        all_core_groups.extend(
            dict(checkpoint=label, **g) for g in aggregate([r for r in curve_rows if r["checkpoint"] == label])
        )
    training = training_audit(cp["updates"])
    parameter_norms = []
    for checkpoint in inventory["selected"]:
        state = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        model = E014CapacityDenoiser(**checkpoint["contract"]["capacity_config"]).eval()
        model.load_state_dict(state["model"], strict=True)
        norms = defaultdict(float)
        for name, parameter in model.named_parameters():
            group = (
                "output_heads"
                if name.startswith(("head.", "local_head."))
                else "local_branch"
                if name.startswith(("local_", "gate."))
                else "trunk"
            )
            norms[group] += float(parameter.detach().double().square().sum())
        parameter_norms.append(
            dict(
                checkpoint=checkpoint["label"],
                updates=checkpoint["updates"],
                branch_l2={group: np.sqrt(value) for group, value in norms.items()},
                total_l2=np.sqrt(sum(norms.values())),
                accumulation="float64 CPU diagnostic from immutable checkpoint",
            )
        )
        del state, model
    for r in inventory["checkpoints"]:
        if r.get("eligible"):
            assert sha(r["path"]) == r["sha256"]
    result = dict(
        decisions=decisions,
        core_groups=all_core_groups,
        paired=paired,
        marginal=marginals,
        gaussian=gaussian,
        geometry_epoch_deltas=geometry,
        direct_onsets=onsets,
        marginal_fronts=fronts,
        high_noise_effect_trends=effects,
        distribution_epoch_deltas=epoch_effect_deltas,
        prospective=dict(
            evaluated=False,
            population_size=2000,
            population_hash="373cb7c0b2f0f7b64fdd5d47d742cf7f07b8b0ddbc0da8683032375cd8dec25b",
            reason="Frozen final trigger (two consecutive complete-epoch improvements <1%, or epoch4) has not occurred; epoch1->2 improvement is1.67465%. Current epoch3 is incomplete.",
        ),
        training=training,
        checkpoint_parameter_norms=parameter_norms,
        exact_shared_reference_preprocessing=True,
        all_checkpoint_hashes_unchanged=True,
        sequence_branch_untouched=True,
        training_resumed=False,
        full_sampling_run=False,
    )
    atomic_json(OUT / "epoch_comparison.json", result)
    print(json.dumps(decisions), flush=True)
    report(result, inventory)


def report(data, inventory):
    def p(label, t, s="pooled"):
        return next(g for g in data["paired"] if (g["checkpoint"], g["timestep"], g["length_stratum"]) == (label, t, s))

    def mean(g, k):
        return g[k]["mean"]

    def table(headers, rows):
        return "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(str(x) for x in r) + " |" for r in rows],
            ]
        )

    lines = [
        "# E014 post-minimum-exposure scientific evaluation",
        "",
        "E014 has learned a useful low/mid-noise reverse field, but the frozen success criteria are not met after two complete epochs. The epoch-2 and newest-checkpoint adjudications are both Case E: low-noise local geometry is acceptable while high-noise direct reconstruction remains problematic. This is a partial scientific success, not evidence of completed unconditional protein generation.",
        "",
        "Coordinate-v loss and local geometry improve substantially relative to initialization. Across trained checkpoints, however, geometry is nonmonotonic: the temporary 75%-epoch recovery is followed by regression at epoch 1 and epoch 2, then recovery at the newest boundary. Mid/high-noise torsion and chirality remain poorly resolved. Paired high-noise direction weakens with SNR, and the marginal diagnostic has insufficient power to establish a consistent extension to lower SNR.",
        "",
        "At extreme noise, a conditional-mean clean estimator can contract while an unconditional reverse distribution remains stable. Neither small paired v norm/cosine nor direct-X0 collapse alone proves generative failure. Case E records the unchanged original interpretation criteria; the independent Gaussian-prefix and marginal-distribution evidence is required to assess actual noisy-state behavior.",
        "",
        "The exact Gaussian-start prefixes test actual noisy states separately from the direct-X0 diagnostic. Their full descriptor distributions, uncertainty and reference-relative scales are reported below; preserved Rg alone does not establish the full coordinate law. These 34 descriptors do not test generated chirality or topology.",
        "",
        "READ-ONLY model/data evaluation. Training remained stopped; Gaussian sampling ended at350. No full de novo or E007 control was run. All new artifacts are under this D:-resident evaluation root. Previous interim results remain unchanged.",
        "",
        "## Durable boundaries",
        "",
    ]
    lines.append(
        table(
            ["checkpoint", "SHA256", "updates", "TRAIN exposures", "complete epochs"],
            [
                [r["label"], r["sha256"], r["updates"], r["exposures"], r["completed_epochs"]]
                for r in inventory["selected"]
            ],
        )
    )
    lines += [
        "",
        "All checkpoint payloads were inventoried, strict-loaded and checked against the frozen capacity/config/population/source contract, regenerated schedule, per-identity exposures, optimizer moments/steps and RNG serialization. LATEST SHA256 matches independently. Older boundaries without independent persisted digests are explicitly marked in the inventory; computing a fresh hash is not independent proof of their historical bytes.",
        "",
        "## Frozen development results",
        "",
        "500 identities,100 per length stratum; panel3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf. Paired noises and independent marginal seeds are unchanged. Gaussian subset100 identities/20 per stratum, c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0.",
        "",
    ]
    for field, title in [
        ("v_mse", "Coordinate-v MSE"),
        ("x0_mse_angstrom2", "Centered X0 component MSE (Å²)"),
        ("aligned_rmsd_angstrom", "Proper Kabsch RMSD (Å)"),
        ("v_norm_ratio", "Predicted/target v norm"),
        ("v_cosine", "Paired v cosine"),
        ("offset_1_mse_angstrom2", "L1 distance MSE (Å²)"),
        ("offset_2_mse_angstrom2", "L2 distance MSE (Å²)"),
        ("offset_3_mse_angstrom2", "L3 distance MSE (Å²)"),
        ("local_mse_angstrom2", "Mean local MSE (Å²)"),
        ("torsion_mse", "Pseudo-torsion sin/cos sum-square error"),
        ("torsion_angular_mae_radians", "Pseudo-torsion angular MAE (radians)"),
        ("triple_product_mse", "Normalized signed triple-product MSE"),
        ("radius_ratio", "Direct X0 mean Rg ratio"),
        ("covariance_trace_ratio", "Direct X0 covariance trace ratio"),
    ]:
        lines += [
            f"### {title}",
            "",
            table(
                ["checkpoint", *map(str, CORE)],
                [[ALIASES[label], *[f"{mean(p(label, t), field):.7g}" for t in CORE]] for label in ALL],
            ),
            "",
        ]
    lines += [
        "### Chirality counts and direct collapse",
        "",
        table(
            [
                "checkpoint",
                "t",
                "torsions assessed/potential",
                "inversions/sign assessed",
                "inversion rate",
                "direct collapse/500",
            ],
            [
                [
                    ALIASES[label],
                    t,
                    f"{p(label, t)['torsion_assessed_sum']}/{p(label, t)['torsion_potential_count_sum']}",
                    f"{p(label, t)['triple_inversion_count_sum']}/{p(label, t)['triple_sign_assessed_sum']}",
                    f"{p(label, t)['triple_inversion_rate_pooled']:.6g}",
                    p(label, t)["direct_collapse_count"],
                ]
                for label in ALL
                for t in CORE
            ],
        ),
        "",
        "### Direct X0 onset (diagnostic)",
        "",
        table(
            ["checkpoint", "median Rg<0.5 onset", ">50% direct collapse onset"],
            [
                [
                    ALIASES[g["checkpoint"]],
                    g["first_median_rg_below_half"],
                    g["first_historical_collapse_fraction_over_half"],
                ]
                for g in data["direct_onsets"]
                if g["length_stratum"] == "pooled"
            ],
        ),
        "",
        "### Incremental local geometry",
        "",
        table(
            ["comparison", "t", "metric", "old", "new", "relative improvement", "paired95% delta CI"],
            [
                [
                    f"{ALIASES[g['old']]}→{ALIASES[g['new']]}",
                    g["timestep"],
                    g["metric"],
                    f"{g['old_mean']:.6g}",
                    f"{g['new_mean']:.6g}",
                    f"{100 * g['relative_improvement']:.3f}%",
                    g["ci"],
                ]
                for g in data["geometry_epoch_deltas"]
                if g["length_stratum"] == "pooled"
            ],
        ),
        "",
    ]
    lines += [
        "## Dense high-noise field",
        "",
        "Paired next-step gain compares with the same clean/original-epsilon path. It is a diagnostic of paired direction and scale, not identification of a unique unconditional trajectory.",
        "",
        table(
            [
                "checkpoint",
                "t",
                "alpha",
                "sigma",
                "SNR",
                "logSNR",
                "v cosine",
                "v norm ratio",
                "step gain",
                "step cosine",
                "step norm ratio",
            ],
            [
                [
                    ALIASES[label],
                    t,
                    *[
                        f"{schedule_values(CoordinateVPDiffusion(500), t)[k]:.5g}"
                        for k in ("alpha", "sigma", "snr", "log_snr")
                    ],
                    f"{mean(p(label, t), 'v_cosine'):.5g}",
                    f"{mean(p(label, t), 'v_norm_ratio'):.5g}",
                    f"{100 * p(label, t)['step_gain_of_means']:.4g}%",
                    f"{mean(p(label, t), 'step_cosine'):.5g}",
                    f"{mean(p(label, t), 'step_norm_ratio'):.5g}",
                ]
                for label in (EARLY[2], EARLY[3], *SELECTED)
                for t in HIGH
            ],
        ),
        "",
    ]
    lines += [
        "## Marginal reverse-distribution effects",
        "",
        "Unchanged34-descriptor diagnostic; independent q(t−1) reference populations; reference-only median/IQR fits;500 protein-unit, length-stratified bootstraps;64 sliced projections. Conditional uncertainty, exploratory multiple comparisons, biased finite-sample V-statistic. Small transitions at near-Gaussian noise have limited power; nonsignificance is not equivalence. The historical true-v positive control resolved only to50. Raw effect is D_model−D_noop (negative is improvement).",
        "",
    ]
    lines.append(
        table(
            ["checkpoint", "t", "metric", "ref-ref floor", "no-op/ref", "model/ref", "effect", "95% effect CI"],
            [
                [
                    ALIASES[g["checkpoint"]],
                    g["timestep"],
                    m["metric"],
                    f"{m['ref_ref']:.6g}",
                    f"{m['noop_ref']:.6g}",
                    f"{m['model_ref']:.6g}",
                    f"{m['model_ref'] - m['noop_ref']:.6g}",
                    m["model_minus_noop_ci"],
                ]
                for g in data["marginal"]
                if g["length_stratum"] == "pooled" and g["checkpoint"] in (EARLY[2], EARLY[3], *SELECTED)
                for m in g["metrics"]
                if m["metric"] in ("energy", "mmd2", "sliced_wasserstein", "rg_wasserstein", "covariance_spectrum_w1")
            ],
        )
    )
    lines += [
        "",
        "## Gaussian generative prefixes and actual noisy states",
        "",
        "Exact centered-Gaussian X499 initialization; matched length strata and frozen seeds. Reference populations are forward-noised real q_t at each endpoint. Descriptive population noisy collapse is Rg ratio<0.8 OR covariance-trace ratio<0.8; check pooled and every length stratum. This is separate from direct-X0 collapse. Per-protein matched-reference scale comparisons are descriptive because unconditional states are not paired targets.",
        "",
        table(
            [
                "checkpoint",
                "endpoint",
                "energy",
                "MMD²",
                "slicedW1",
                "Rg ratio",
                "trace ratio",
                "pair mean ratio",
                "finite",
                "scale collapse",
            ],
            [
                [
                    ALIASES[g["checkpoint"]],
                    g["timestep"],
                    *[
                        f"{next(m['model_ref'] for m in g['metrics'] if m['metric'] == name):.6g}"
                        for name in ("energy", "mmd2", "sliced_wasserstein")
                    ],
                    f"{g['rg_mean_ratio']:.6g}",
                    f"{g['trace_mean_ratio']:.6g}",
                    f"{g['raw_descriptor_distributions']['model']['pair_mean']['mean'] / g['raw_descriptor_distributions']['ref_b']['pair_mean']['mean']:.6g}",
                    True,
                    g["noisy_scale_status"],
                ]
                for g in data["gaussian"]
                if g["length_stratum"] == "pooled"
            ],
        ),
        "",
    ]
    lines += [
        "## Branch telemetry",
        "",
        table(
            [
                "checkpoint",
                "t",
                "global norm",
                "effective local norm",
                "local/global",
                "cosine",
                "gate",
                "reliability",
                "frame assessed/residues",
            ],
            [
                [
                    ALIASES[label],
                    t,
                    *[
                        f"{mean(p(label, t), k):.6g}"
                        for k in (
                            "global_output_norm",
                            "local_effective_output_norm",
                            "local_global_norm_ratio",
                            "global_local_cosine",
                            "local_gate_mean",
                            "frame_reliability_mean",
                        )
                    ],
                    f"{p(label, t)['frame_assessable_count_sum']}/{p(label, t)['frame_residue_count_sum']}",
                ]
                for label in ALL
                for t in CORE
            ],
        ),
        "",
        "## Length-stratified development results",
        "",
        table(
            [
                "checkpoint",
                "stratum",
                "t",
                "v-MSE",
                "L1",
                "L2",
                "L3",
                "torsion",
                "inversions/assessed",
                "direct collapse/100",
                "Rg ratio",
            ],
            [
                [
                    ALIASES[label],
                    s,
                    t,
                    *[
                        f"{mean(p(label, t, s), k):.6g}"
                        for k in (
                            "v_mse",
                            "offset_1_mse_angstrom2",
                            "offset_2_mse_angstrom2",
                            "offset_3_mse_angstrom2",
                            "torsion_mse",
                        )
                    ],
                    f"{p(label, t, s)['triple_inversion_count_sum']}/{p(label, t, s)['triple_sign_assessed_sum']}",
                    p(label, t, s)["direct_collapse_count"],
                    f"{mean(p(label, t, s), 'radius_ratio'):.6g}",
                ]
                for label in SELECTED
                for s in STRATA
                for t in CORE
            ],
        ),
        "",
    ]
    lines += [
        "## Training, plateau and frozen adjudication",
        "",
        "All logged updates are checked against the durable update boundary; uncommitted tail is excluded. Training v-loss, gradient/clipping, per-branch parameter and optimizer-step norms are saved in training_curve.parquet and optimizer_telemetry.parquet. Training geometry/torsion/chirality were not logged; validation boundary records supply those curves. The epoch2 verdict is retained independently of the later rolling checkpoint.",
        "",
        table(
            [
                "boundary",
                "cumulative training v-loss",
                "current epoch v-loss",
                "cumulative clipping rate",
                "parameter L2",
            ],
            [
                [
                    ALIASES[r["label"]],
                    r["training_v_loss"],
                    r["epoch_training_v_loss"],
                    r["clipping_fraction"],
                    next(x["total_l2"] for x in data["checkpoint_parameter_norms"] if x["checkpoint"] == r["label"]),
                ]
                for r in inventory["selected"]
            ],
        ),
        "",
        "Training loss is still declining across complete epochs and the partial third epoch, with short-bin fluctuations. Balanced fixed-timestep development v-MSE improves 1.67465% from epoch 1 to epoch 2, exceeding the frozen 1% plateau criterion. No two-consecutive-complete-epoch plateau or four-epoch stop has occurred. Geometry has regressed and recovered rather than improving monotonically. Training and development v-loss use different timestep/length weighting, so their absolute gap is not an overfitting estimate.",
        "",
        f"Uncommitted logged tail: {data['training']['logged_uncommitted_records']} updates / {data['training']['logged_uncommitted_exposures']} exposures, ending at update {data['training']['last_logged_update']}; excluded from durable exposure and training conclusions.",
        "",
        "```json",
        json.dumps(data["decisions"], indent=2),
        "```",
        "",
        "## Prospective validation",
        "",
        data["prospective"]["reason"],
        "The frozen2,000-identity prospective panel remains uninspected. Meeting minimum exposure permits scientific adjudication but is not the runner's final-validation trigger.",
        "",
        "## Next-experiment decision",
        "",
        "Stage D / full t=0 sampling is not authorized by the unchanged frozen protocol: only Case A opens that gate, and both epoch 2 and newest remain Case E. The blocking criterion is high-noise direct-estimator geometry; this does not establish actual noisy-state collapse. A revised sampling gate would require an explicitly authorized protocol change. No training restart is implied by this evaluation.",
        "",
        "A matched full-scale E007 control is unnecessary to assign the within-E014 Case E record, but remains necessary for architecture-specific attribution. Recommend it if the next scientific objective is to compare coordinate-v learning with the matched distance-based control; historical E007/E013 results are not matched evidence. No E007 process was launched.",
        "",
        "## Numerical replay",
        "",
        "See replay_numerics.json for repeated6nof_A/440 prefixes and the unchanged5e−6 tolerance. Exact historical replay certification is separate from scientific summaries; the historical8.81764e−6 outlier remains preserved.",
        "",
        "## Artifacts and scope",
        "",
        "checkpoint_inventory.json; epoch_comparison.json; core_metrics.parquet; high_noise_grid.parquet; marginal_metrics.parquet; gaussian_prefix_metrics.parquet; branch_telemetry.parquet; geometry_epoch_deltas.parquet; training_curve.parquet; optimizer_telemetry.parquet; replay_numerics.json. All scientific outputs and retained artifacts are on D:. No sequence branch files or historical model/training sources were edited. No trainer was started.",
        "",
    ]
    if (OUT / "STORAGE_AUDIT.json").exists():
        lines += [
            "Storage exception: the first plotting attempt encountered a sandbox permission error on the original D cache path, and Matplotlib temporarily created a font cache under /tmp. Plotting was interrupted; the cache was relocated to the writable D evaluation root, its /tmp directory was removed, and plots were regenerated with an explicitly checked writable D cache. Scientific outputs never left D, but strict no-transient-/tmp compliance cannot be certified. See STORAGE_AUDIT.json.",
            "",
        ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["analyze", "finish"])
    parser.add_argument("--checkpoint", choices=SELECTED)
    args = parser.parse_args()
    analyze(args.checkpoint) if args.phase == "analyze" else finish()
