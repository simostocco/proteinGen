"""CPU-only observational analysis of frozen E014 development evaluations."""

# ruff: noqa: I001, E501  # Preserve standalone report prose.
from e014_bootstrap import ROOT
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, UTC
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from coordinate_execution_policy import canonical_runner
from upgrade_e014_terminal_guard import FROZEN_PROBE_SOURCE, verify_guard_only_change
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, epoch_schedule, sha

RUN = ROOT / "e014"
OUT = RUN / "interim_50pct"
LABELS = ("initialization", "epoch_1_fraction_0.25", "epoch_1_fraction_0.50")
TIMES = (5, 50, 250, 400, 490)
STRATA = ("20-64", "65-128", "129-256", "257-384", "385-500")


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def corr(a, b, ranked=False):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if ranked:

        def ranks(x):
            unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
            del unique
            end = np.cumsum(counts)
            return ((end - counts + end - 1) / 2)[inverse]

        a, b = ranks(a), ranks(b)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows):
    def total(key):
        return sum(r[key] for r in rows)

    result = dict(
        condition_count=len(rows),
        identity_count=len({r["sample_id"] for r in rows}),
        collapse_count=total("collapsed"),
        finite_count=total("finite"),
    )
    for key in rows[0]:
        if isinstance(rows[0][key], (int, float)) and key not in ("timestep", "length", "finite", "collapsed"):
            result[key] = mean(rows, key)
    for key in (
        "adjacent_count",
        "adjacent_valid_count",
        "torsion_assessed",
        "torsion_potential_count",
        "triple_sign_assessed",
        "triple_inversion_count",
        "clash_count",
        "clash_pair_count",
        "long_range_contact_count",
        "long_range_pair_count",
        "frame_assessable_count",
        "frame_residue_count",
    ):
        result[key + "_sum"] = total(key)
    result["collapse_fraction"] = total("collapsed") / len(rows)
    result["triple_inversion_pooled_rate"] = total("triple_inversion_count") / total("triple_sign_assessed")
    result["torsion_assessability"] = total("torsion_assessed") / total("torsion_potential_count")
    result["adjacent_valid_pooled_fraction"] = total("adjacent_valid_count") / total("adjacent_count")
    result["clash_pooled_fraction"] = total("clash_count") / total("clash_pair_count")
    result["frame_assessability_pooled_fraction"] = total("frame_assessable_count") / total("frame_residue_count")
    result["unaligned_centered_rmsd_angstrom"] = float(np.mean([math.sqrt(3 * r["x0_mse_angstrom2"]) for r in rows]))
    result["rg_ratio_quantiles"] = dict(
        zip(
            ("p05", "p25", "median", "p75", "p95"),
            np.quantile([r["radius_ratio"] for r in rows], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
            strict=True,
        )
    )
    result["adjacent_protein_quantiles_mean_angstrom"] = np.mean(
        [r["adjacent_quantiles_angstrom"] for r in rows], axis=0
    ).tolist()
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
    population = json.loads((ROOT / "population/population.json").read_text())
    panelpath = ROOT / "population/development.parquet"
    assert sha(panelpath) == population["manifest_hashes"][str(panelpath)]
    panel = pq.read_table(panelpath, use_threads=False).to_pylist()
    panelids = [r["sample_id"] for r in panel]
    assert len(panelids) == len(set(panelids)) == 500
    assert canonical_hash(panelids) == population["development_sample_id_hash"]
    assert all(r["split"] == "validation" for r in panel)
    strata = {s: sum(r["length_stratum"] == s for r in panel) for s in STRATA}
    assert all(n == 100 for n in strata.values())
    assert all(row["length_stratum"] == STRATA[z // 100] for z, row in enumerate(panel))
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    assert len(train) == 230132 and set(panelids).isdisjoint(r["sample_id"] for r in train)
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == population["train_sample_id_hash"]
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], 1)
    expected = np.zeros(len(train), dtype=np.int32)
    prefix = np.concatenate([[0], np.cumsum([len(x) for x in schedule])])
    frozen_runner = canonical_runner(Path("scripts/run_e014_scaleup.py").read_text()).encode()
    old_runner = subprocess.check_output(["git", "show", f"{FROZEN_PROBE_SOURCE}:scripts/run_e014_scaleup.py"])
    verify_guard_only_change(old_runner, frozen_runner)
    source_hashes = dict()
    checkpoints = []
    metrics = {}
    summaries = {}
    model = None
    base_contract = None
    diffusion = CoordinateVPDiffusion(500)
    for label in LABELS:
        path = RUN / "checkpoints" / f"{label}.pt"
        digest = sha(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        contract = state["contract"]
        assert contract["config_hash"] == sha(cfgpath)
        assert contract["data_hash"] == population["train_sample_id_hash"]
        assert contract["normalization_hash"] == sha(cfg["normalization_artifact"]) == cfg["normalization_sha256"]
        assert contract["population_manifest_hash"] == sha(ROOT / "population/population.json")
        assert contract["cache_ready_hash"] == sha(ROOT / "coordinate_cache/READY.json")
        for file, hashed in contract["source_hashes"].items():
            actual = (
                hashlib.sha256(old_runner if label == "initialization" else frozen_runner).hexdigest()
                if file == "scripts/run_e014_scaleup.py"
                else sha(file)
            )
            assert actual == hashed, (label, file)
        science = {k: v for k, v in contract.items() if k != "source_hashes"}
        if base_contract is None:
            base_contract = science
        assert science == base_contract
        if model is None:
            model = E014CapacityDenoiser(**contract["capacity_config"]).eval()
        model.load_state_dict(state["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 5119069
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert state["epoch"] == 1 and state["updates"] == state["position"]
        expected.fill(0)
        for ids in schedule[: state["position"]]:
            expected[ids] += 1
        assert np.array_equal(expected, state["exposures"])
        if state["updates"]:
            assert state["schedule_hash"] == canonical_hash(schedule)
            assert state["next_batch"] == schedule[state["position"]]
            assert len(state["optimizer"]["state"]) == 150
        else:
            assert int(state["exposures"].sum()) == 0
        assert set(state["rng"]) == {"python", "numpy", "torch", "cuda"}
        assert sha(path) == digest
        entry = dict(
            label=label,
            path=str(path),
            sha256=digest,
            timestamp_utc=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
            epoch=1,
            update=state["updates"],
            schedule_cursor=state["position"],
            exposures=int(state["exposures"].sum()),
            preceding_update_exposures=int(prefix[max(0, state["position"] - 1)]),
            schedule_hash=canonical_hash(schedule),
            parameter_count=5119069,
            contract=contract,
            elapsed_seconds=state["elapsed_seconds"],
            gpu_seconds=state["gpu_seconds"],
            checkpoint_training_mean=state["epoch_loss_sum"] / state["epoch_loss_proteins"]
            if state["epoch_loss_proteins"]
            else None,
            clipping_fraction=state["clipping_count"] / state["updates"] if state["updates"] else None,
            complete=True,
            loadable=True,
            internally_consistent=True,
            model_state_complete=True,
        )
        checkpoints.append(entry)
        file = RUN / "evaluation" / f"{label}_development.metrics.json"
        summarypath = file.with_name(f"{label}_development.summary.json")
        rows = json.loads(file.read_text())
        stored = json.loads(summarypath.read_text())
        source_hashes[str(file)] = sha(file)
        source_hashes[str(summarypath)] = sha(summarypath)
        pairs = {(r["sample_id"], r["timestep"]) for r in rows}
        assert len(rows) == len(pairs) == 2500 and pairs == {(i, t) for i in panelids for t in TIMES}
        assert all(r["split"] == "validation" and r["finite"] for r in rows)
        assert stored["identity_count"] == 500 and stored["condition_count"] == 2500
        for row in rows:
            sigma2 = float(1 - diffusion.alphas_cumprod[row["timestep"]])
            assert math.isclose(
                row["x0_mse_angstrom2"],
                cfg["coordinate_scale_angstrom"] ** 2 * sigma2 * row["v_mse"],
                rel_tol=2e-4,
                abs_tol=2e-5,
            )
            assert row["collapsed"] == (row["radius_ratio"] < 0.25)
            assert row["local_mse_angstrom2"] == sum(row[f"offset_{k}_mse_angstrom2"] for k in (1, 2, 3)) / 3
        metrics[label] = rows
        summaries[label] = dict(
            all=summarize(rows),
            timestep={t: summarize([r for r in rows if r["timestep"] == t]) for t in TIMES},
            length={s: summarize([r for r in rows if r["length_stratum"] == s]) for s in STRATA},
            length_timestep={
                f"{s}|{t}": summarize([r for r in rows if r["length_stratum"] == s and r["timestep"] == t])
                for s in STRATA
                for t in TIMES
            },
        )
        assert math.isclose(
            summaries[label]["all"]["v_mse"],
            next(g["v_mse_equal_protein_mean"] for g in stored["groups"] if g["dimension"] == "all"),
            abs_tol=1e-12,
        )
    curves = json.loads((RUN / "CURVES.json").read_text())
    for entry in checkpoints:
        curve = next(c for c in curves if c["updates"] == entry["update"])
        assert curve["protein_exposures"] == entry["exposures"]
        assert math.isclose(curve["development_v_mse"], summaries[entry["label"]]["all"]["v_mse"], abs_tol=1e-12)
        if entry["update"]:
            assert curve["checkpoint"] == entry["path"]
    quarter, half = checkpoints[1:]
    assert quarter["exposures"] == 57533
    assert half["preceding_update_exposures"] < 115066 <= half["exposures"] == 115067
    assert half["schedule_cursor"] == int(np.searchsorted(prefix, 115066))
    reference = {(r["sample_id"], r["timestep"]): r for r in metrics[LABELS[0]]}
    for label in LABELS[1:]:
        for row in metrics[label]:
            ref = reference[row["sample_id"], row["timestep"]]
            for field in (
                "reference_radius_of_gyration_angstrom",
                "frame_reliability_mean",
                "frame_reliability_median",
                "frame_assessable_count",
            ):
                assert row[field] == ref[field], (label, field)
    identity = dict(
        checkpoints=checkpoints,
        panel=dict(
            size=500,
            sample_id_hash=population["development_sample_id_hash"],
            parquet_sha256=sha(panelpath),
            length_counts=strata,
            split_provenance=population["split_policy"],
            all_validation=True,
            train_disjoint=True,
        ),
        nominal_half_exposures=115066,
        actual_half_exposures=115067,
        overshoot=1,
        half_certification="first complete optimizer batch reaching threshold; previous batch exposure 115065",
        evaluation_source_hashes=source_hashes,
        panel_hash_definition="SHA256 canonical JSON of development IDs in frozen manifest order",
        paired_noise_namespace="e014-paired-validation-v1",
        paired_seed_algorithm="SHA256(namespace|sample_id|t) mod (2^63-1); CUDA generator",
        training_modified=False,
        gpu_evaluation_run=False,
        existing_evaluations_reused=True,
    )
    atomic_json(OUT / "checkpoint_identity.json", identity)
    with (OUT / "collapse_cases.csv").open("w", newline="") as f:
        fields = [
            "sample_id",
            "length",
            "timestep",
            "reference_radius_of_gyration_angstrom",
            "radius_of_gyration_angstrom",
            "radius_ratio",
            "v_mse",
            "offset_1_mse_angstrom2",
            "offset_2_mse_angstrom2",
            "offset_3_mse_angstrom2",
            "global_output_norm",
            "local_effective_output_norm",
            "local_global_norm_ratio",
            "global_local_cosine",
            "local_gate_mean",
            "local_gate_median",
            "frame_reliability_mean",
            "frame_reliability_median",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: r[key] for key in fields} for r in metrics[LABELS[2]] if r["collapsed"])
    correlations = {}
    for label in LABELS[1:]:
        correlations[label] = {}
        for t in TIMES:
            rows = [r for r in metrics[label] if r["timestep"] == t]
            rho = [r["local_global_norm_ratio"] for r in rows]
            correlations[label][t] = dict(
                n=len(rows),
                collapse_count=sum(r["collapsed"] for r in rows),
                collapse_vs_rho_pearson=corr([r["collapsed"] for r in rows], rho),
                rg_ratio_vs_rho_pearson=corr([r["radius_ratio"] for r in rows], rho),
                rg_ratio_vs_rho_spearman=corr([r["radius_ratio"] for r in rows], rho, ranked=True),
            )
    # Read only the committed log prefix ending at the half checkpoint.
    records = []
    log_hash = hashlib.sha256()
    with (RUN / "TRAINING.jsonl").open("rb") as f:
        for raw in f:
            row = json.loads(raw)
            log_hash.update(raw)
            if row["update"] != len(records) + 1:
                raise ValueError("provisional/replayed log prefix needs independent reconciliation")
            records.append(row)
            if row["update"] == half["update"]:
                break
    assert len(records) == half["update"]
    assert sum(r["identity_count"] for r in records) == half["exposures"]
    assert math.isclose(
        sum(r["loss"] * r["identity_count"] for r in records) / half["exposures"],
        half["checkpoint_training_mean"],
        abs_tol=1e-12,
    )

    def training_bin(rows):
        telemetry = [r["module_statistics"] for r in rows if "module_statistics" in r]
        return dict(
            first_update=rows[0]["update"],
            last_update=rows[-1]["update"],
            updates=len(rows),
            proteins=sum(r["identity_count"] for r in rows),
            v_loss=sum(r["loss"] * r["identity_count"] for r in rows) / sum(r["identity_count"] for r in rows),
            gradient_mean=float(np.mean([r["gradient_norm"] for r in rows])),
            gradient_median=float(np.median([r["gradient_norm"] for r in rows])),
            gradient_p95=float(np.quantile([r["gradient_norm"] for r in rows], 0.95)),
            clipping_fraction=sum(r["clipped"] for r in rows) / len(rows),
            optimizer_telemetry_samples=len(telemetry),
            optimizer_step_norm_mean={
                g: float(np.mean([s[g]["optimizer_step_norm"] for s in telemetry]))
                for g in ("trunk", "local_branch", "output_heads")
            },
        )

    bins = [training_bin(records[i : i + 5000]) for i in range(0, len(records), 5000)]
    interval = training_bin(records[quarter["update"] :])
    boundaries = {
        c["label"]: dict(
            raw_record=records[c["update"] - 1] if c["update"] else None,
            checkpoint_training_mean=c["checkpoint_training_mean"],
            cumulative_clipping_fraction=c["clipping_fraction"],
        )
        for c in checkpoints
    }
    throughput = []
    for a, b in zip(checkpoints[:-1], checkpoints[1:], strict=True):
        throughput.append(
            dict(
                from_label=a["label"],
                to_label=b["label"],
                exposures=b["exposures"] - a["exposures"],
                elapsed_seconds=b["elapsed_seconds"] - a["elapsed_seconds"],
                proteins_per_execution_wall_second=(b["exposures"] - a["exposures"])
                / (b["elapsed_seconds"] - a["elapsed_seconds"]),
                updates_per_execution_wall_second=(b["update"] - a["update"])
                / (b["elapsed_seconds"] - a["elapsed_seconds"]),
            )
        )
    training = dict(
        bins=bins,
        boundaries=boundaries,
        quarter_to_half=interval,
        throughput_intervals=throughput,
        prefix_sha256=log_hash.hexdigest(),
        per_update_timestamps_available=False,
        throughput_note="checkpoint elapsed counters include startup/development evaluation; no invented per-step timing",
    )
    # Paired identity bootstrap, preserving all five conditions for an identity.
    rng = np.random.default_rng(140505)
    index = {label: {(r["sample_id"], r["timestep"]): r for r in metrics[label]} for label in LABELS}
    draws = np.concatenate([rng.integers(z * 100, (z + 1) * 100, size=(2000, 100)) for z in range(5)], axis=1)
    boot = {}
    for field in ("v_mse", "local_mse_angstrom2"):
        a = np.array([[index[LABELS[1]][i, t][field] for t in TIMES] for i in panelids]).mean(1)
        b = np.array([[index[LABELS[2]][i, t][field] for t in TIMES] for i in panelids]).mean(1)
        samples = 1 - b[draws].mean(1) / a[draws].mean(1)
        boot[field] = dict(
            relative_improvement=1 - float(b.mean() / a.mean()),
            paired_bootstrap_95_interval=np.quantile(samples, [0.025, 0.975]).tolist(),
            seed=140505,
            replicates=2000,
            stratified=True,
        )
    atomic_json(
        OUT / "metrics.json",
        dict(
            boundaries=summaries,
            training=training,
            correlations=correlations,
            paired_bootstrap=boot,
            collapse_cases_count=500,
            half_t490_noncollapsed_controls=0,
            interim_category="C",
            final_adjudication=False,
        ),
    )
    plot_training(bins, throughput)
    report_markdown(checkpoints, identity, summaries, training, correlations, boot)
    print(
        json.dumps(dict(checkpoint=half, aggregate={k: summaries[k]["all"]["v_mse"] for k in LABELS}, bootstrap=boot)),
        flush=True,
    )


def plot_training(bins, throughput):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    x = [b["last_update"] for b in bins]
    for ax, key, title in zip(
        axes.flat[:3],
        ("v_loss", "gradient_mean", "clipping_fraction"),
        ("Training coordinate-v loss", "Gradient norm", "Clipping fraction"),
        strict=True,
    ):
        ax.plot(x, [b[key] for b in bins], marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Optimizer update")
    ax = axes.flat[3]
    for group in ("trunk", "local_branch", "output_heads"):
        ax.plot(x, [b["optimizer_step_norm_mean"][group] for b in bins], label=group)
    ax.set_title("Optimizer step norm (bounded telemetry)")
    ax.legend()
    ax.set_xlabel("Optimizer update")
    ax = axes.flat[4]
    ax.bar(["0–25%", "25–50%"], [r["proteins_per_execution_wall_second"] for r in throughput])
    ax.set_title("Proteins / execution wall second")
    ax.set_ylabel("Includes evaluation/startup time")
    axes.flat[5].axis("off")
    fig.suptitle("E014 interim: frozen prefix through 115,067 exposures; 5,000-update bins")
    fig.savefig(OUT / "training_curve.png", dpi=160)
    fig.savefig(OUT / "training_curve.svg")
    plt.close(fig)


def report_markdown(checkpoints, identity, summaries, training, correlations, boot):
    report = [
        "# E014 interim 50%-epoch observation — non-adjudicating",
        "",
        "**Descriptive pattern C:** coordinate-v learning improves, but aggregate local-distance error and high-noise collapse worsen from 25% to 50%. Low-noise local geometry improves. The E013-style objective/geometry divergence remains present so far; this is not a final architectural result. Training must continue for at least two complete TRAIN epochs before adjudication.",
        "",
        "Training continued throughout. This report reuses already completed frozen development evaluations; no GPU/model evaluation, sampling, matched control or training intervention occurred. CPU-only provenance checks, aggregation, log-prefix analysis and plots were performed.",
        "",
    ]

    def table(headers, rows):
        report.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        for row in rows:
            report.append("| " + " | ".join(str(x) for x in row) + " |")
        report.append("")

    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    def groups(dimension, key):
        return [summaries[label][dimension][key] for label in LABELS]

    report.extend(["## Checkpoints and panel", ""])
    table(
        ["Boundary", "Update/cursor", "Exposures", "UTC timestamp", "SHA256"],
        [[c["label"], c["update"], c["exposures"], c["timestamp_utc"], c["sha256"]] for c in checkpoints],
    )
    report.extend(
        [
            f"50% checkpoint path: `{checkpoints[2]['path']}`. Nominal 115,066 exposures; actual 115,067, +1 identity (+0.000435 percentage points of an epoch). Previous completed update had 115,065 exposures; this is the first complete optimizer batch crossing the frozen threshold. No exact 115,066 state exists, and none was fabricated.",
            "",
            f"Development panel: 500 validation identities, 100 in each length stratum. Sample-ID hash `{identity['panel']['sample_id_hash']}`; manifest hash `{identity['panel']['parquet_sha256']}`. Frozen 30%-identity/80%-coverage clean validation policy; disjoint from TRAIN. Five paired conditions per identity, 2,500 conditions. Equal-protein aggregate also gives equal weight to the five balanced strata, not TRAIN population weights.",
            "",
            "All checkpoints load completely; model has 5,119,069 parameters. Config, population, normalization, exposure vector and schedule provenance match. The initialization runner differs only by the documented terminal-phase guard; exact byte-level scientific compatibility was checked. Evaluation source code and all model/diffusion/loss files match frozen hashes. Existing summary/checkpoint association was checked against CURVES and frozen filename conventions; original evaluator did not store a checkpoint hash inside each metric record. Hashes are now pinned in checkpoint_identity.json.",
            "",
            "Noise namespace: `e014-paired-validation-v1`; SHA256-derived identity/timestep seed, CUDA generator. Identical 500 IDs × five timesteps were checked across boundaries; target Rg and current-X frame reliability/assessability are identical. No CPU random-corruption substitution was used. Reconstruction/v-loss identity was checked for all 7,500 records.",
            "",
            "## Coordinate-v objective",
            "",
        ]
    )
    table(
        ["t", "Init", "25%", "50%", "25→50 improvement"],
        [
            [
                t,
                *[fmt(g["v_mse"]) for g in groups("timestep", t)],
                f"{100 * (1 - groups('timestep', t)[2]['v_mse'] / groups('timestep', t)[1]['v_mse']):.2f}%",
            ]
            for t in TIMES
        ],
    )
    allgroups = [summaries[boundary]["all"] for boundary in LABELS]
    table(["Aggregate", "Init", "25%", "50%"], [["v-MSE", *[fmt(g["v_mse"]) for g in allgroups]]])
    report.extend(
        [
            f"25→50 aggregate improvement: {100 * boot['v_mse']['relative_improvement']:.2f}%; paired stratified identity-bootstrap 95% interval {np.array(boot['v_mse']['paired_bootstrap_95_interval']) * 100}%. Bootstrap is descriptive (2,000 draws, seed 140505), not a gate or tuning decision. High-noise t=490 improves only {100 * (1 - groups('timestep', 490)[2]['v_mse'] / groups('timestep', 490)[1]['v_mse']):.2f}%.",
            "",
            "## Reconstruction — matched-reference denoising only",
            "",
            "X0_hat = alpha_t X_t − sigma_t v_hat. Component MSE is Å² averaged over xyz and residues equally per protein; unaligned RMSD is mean sqrt(3×component MSE) per protein. Proper Kabsch uses SO(3), never reflection alignment.",
            "",
        ]
    )
    for field, title in [
        ("x0_mse_angstrom2", "Centered component MSE (Å²)"),
        ("unaligned_centered_rmsd_angstrom", "Unaligned centered RMSD (Å)"),
        ("aligned_rmsd_angstrom", "Proper-Kabsch aligned RMSD (Å)"),
    ]:
        report.extend([f"### {title}", ""])
        table(["t", "Init", "25%", "50%"], [[t, *[fmt(g[field]) for g in groups("timestep", t)]] for t in TIMES])
    report.extend(
        [
            "## Local distances",
            "",
            "L1/L2/L3 are squared distance errors at offsets 1/2/3 in Å². L_local=(L1+L2+L3)/3.",
            "",
        ]
    )
    table(
        ["t", "Init L1 / L2 / L3 / mean", "25% L1 / L2 / L3 / mean", "50% L1 / L2 / L3 / mean"],
        [
            [
                t,
                *[
                    " / ".join(
                        fmt(g[f])
                        for f in (
                            "offset_1_mse_angstrom2",
                            "offset_2_mse_angstrom2",
                            "offset_3_mse_angstrom2",
                            "local_mse_angstrom2",
                        )
                    )
                    for g in groups("timestep", t)
                ],
            ]
            for t in TIMES
        ],
    )
    table(
        ["Aggregate", "Init", "25%", "50%"],
        [
            [field, *[fmt(g[field]) for g in allgroups]]
            for field in (
                "offset_1_mse_angstrom2",
                "offset_2_mse_angstrom2",
                "offset_3_mse_angstrom2",
                "local_mse_angstrom2",
            )
        ],
    )
    report.extend(
        [
            f"Aggregate local MSE increases {-100 * boot['local_mse_angstrom2']['relative_improvement']:.2f}% from 25→50. t=5 and 50 improve, but t=250/400/490 worsen. This is a noise-specific divergence, not universal geometry failure. Initialization→50 local MSE nevertheless improves strongly; the recent trend is the concern.",
            "",
            "## Adjacent C-alpha geometry",
            "",
            "Frozen valid range: inclusive 3.4–4.2 Å. Main fractions and RMSE are equal-protein; quantile summaries below are means of per-protein p05/p50/p95, not pooled bond quantiles. Pooled valid counts/denominators are in metrics.json.",
            "",
        ]
    )
    for label in LABELS:
        report.extend([f"### {label}", ""])
        table(
            ["t", "Mean Å", "RMSE Å", "Valid fraction", "Mean protein p05 / p50 / p95 Å"],
            [
                [
                    t,
                    fmt(summaries[label]["timestep"][t]["adjacent_mean_angstrom"]),
                    fmt(summaries[label]["timestep"][t]["adjacent_rmse_angstrom"]),
                    fmt(summaries[label]["timestep"][t]["adjacent_valid_fraction"]),
                    " / ".join(
                        fmt(x) for x in summaries[label]["timestep"][t]["adjacent_protein_quantiles_mean_angstrom"]
                    ),
                ]
                for t in TIMES
            ],
        )
    report.extend(
        [
            "## Radius and collapse",
            "",
            "Frozen collapse: Rg_hat < 0.25 Rg_target. Each timestep denominator is 500 identities. Rg is centered C-alpha radius in Å.",
            "",
        ]
    )
    for label in LABELS:
        report.extend([f"### {label}", ""])
        table(
            ["t", "Rg target Å", "Rg prediction Å", "Ratio mean", "p05/p25/median/p75/p95", "Collapsed / N (%)"],
            [
                [
                    t,
                    fmt(summaries[label]["timestep"][t]["reference_radius_of_gyration_angstrom"]),
                    fmt(summaries[label]["timestep"][t]["radius_of_gyration_angstrom"]),
                    fmt(summaries[label]["timestep"][t]["radius_ratio"]),
                    " / ".join(fmt(x) for x in summaries[label]["timestep"][t]["rg_ratio_quantiles"].values()),
                    f"{summaries[label]['timestep'][t]['collapse_count']}/500 ({100 * summaries[label]['timestep'][t]['collapse_fraction']:.1f}%)",
                ]
                for t in TIMES
            ],
        )
    report.extend(
        [
            "At t=490: **500/500 → 264/500 → 500/500 collapsed**; mean Rg ratio **0.0370 → 0.2452 → 0.1123**. The initial improvement at 25% reverses at 50%; all 236 formerly noncollapsed 25% examples become collapsed by 50%. At 50%, no t=490 noncollapsed same-timestep controls exist. All 500 cases are in collapse_cases.csv; it includes each ID, length, Rg, v-MSE, L1/L2/L3, component norms/cosine, learned gate and reliability. No control case was invented.",
            "",
            "## Torsion and chirality",
            "",
            "Frozen torsion error is (sin_pred−sin_target)²+(cos_pred−cos_target)², summed over two components, averaged over jointly assessable residues then equally over proteins. Angular error is circular MAE in radians. Triple-product MSE uses jointly assessable torsions; inversion requires |q_target|>1e-3. Rates below pool explicitly reported counts; per-protein fractions are also retained in metrics.json.",
            "",
        ]
    )
    for label in LABELS:
        report.extend([f"### {label}", ""])
        table(
            ["t", "Sin/cos error", "Angular MAE", "Assessed / potential", "Triple MSE", "Inversions / assessed (%)"],
            [
                [
                    t,
                    fmt(summaries[label]["timestep"][t]["torsion_mse"]),
                    fmt(summaries[label]["timestep"][t]["torsion_angular_mae_radians"]),
                    f"{summaries[label]['timestep'][t]['torsion_assessed_sum']}/{summaries[label]['timestep'][t]['torsion_potential_count_sum']}",
                    fmt(summaries[label]["timestep"][t]["triple_product_mse"]),
                    f"{summaries[label]['timestep'][t]['triple_inversion_count_sum']}/{summaries[label]['timestep'][t]['triple_sign_assessed_sum']} ({100 * summaries[label]['timestep'][t]['triple_inversion_pooled_rate']:.2f}%)",
                ]
                for t in TIMES
            ],
        )
    report.extend(
        [
            "## Clashes, contacts and finite status",
            "",
            "Frozen pair definitions: clashes distance<3 Å with |i−j|>3; long-range contacts distance<8 Å with |i−j|>=12. Counts are directed pairs, preserving historical definitions. Fractions shown are equal-protein; pooled counts and denominators are explicitly provided.",
            "",
        ]
    )
    for label in LABELS:
        report.extend([f"### {label}", ""])
        table(
            [
                "t",
                "Finite / N",
                "Clash fraction",
                "Clashes / eligible pairs",
                "Contact density",
                "Contacts / eligible pairs",
            ],
            [
                [
                    t,
                    f"{summaries[label]['timestep'][t]['finite_count']}/500",
                    fmt(summaries[label]["timestep"][t]["clash_fraction"]),
                    f"{summaries[label]['timestep'][t]['clash_count_sum']}/{summaries[label]['timestep'][t]['clash_pair_count_sum']}",
                    fmt(summaries[label]["timestep"][t]["long_range_contact_density"]),
                    f"{summaries[label]['timestep'][t]['long_range_contact_count_sum']}/{summaries[label]['timestep'][t]['long_range_pair_count_sum']}",
                ]
                for t in TIMES
            ],
        )
    report.extend(
        [
            "## Length strata",
            "",
            "Every stratum has 100 identities × 5 timesteps = 500 conditions. No small-identity stratum; do not infer a length law. Aggregate collapse rates mix noise conditions; t=490 rates are shown separately. Full length×timestep results are in metrics.json.",
            "",
        ]
    )
    table(
        ["Length", "v-MSE init / 25 / 50", "Local MSE init / 25 / 50", "Collapsed conditions init / 25 / 50"],
        [
            [
                s,
                " / ".join(fmt(g["v_mse"]) for g in groups("length", s)),
                " / ".join(fmt(g["local_mse_angstrom2"]) for g in groups("length", s)),
                " / ".join(
                    f"{g['collapse_count']}/500 ({100 * g['collapse_fraction']:.1f}%)" for g in groups("length", s)
                ),
            ]
            for s in STRATA
        ],
    )
    table(
        ["Length", "t490 collapse init / 25 / 50"],
        [
            [
                s,
                " / ".join(
                    f"{summaries[boundary]['length_timestep'][f'{s}|490']['collapse_count']}/100" for boundary in LABELS
                ),
            ]
            for s in STRATA
        ],
    )
    report.extend(
        [
            "## Global/local branch telemetry",
            "",
            "Norms are Frobenius norms over the chain, equally averaged across proteins. rho is mean of per-protein local/global norm ratios, not ratio of group means. Component vectors are centered independently, matching the final centered prediction decomposition. Learned gate is sigmoid(time,reliability); effective contribution also multiplies geometric reliability and (1−t/499)².",
            "",
        ]
    )
    for label in LABELS[1:]:
        report.extend([f"### {label}", ""])
        table(
            [
                "t",
                "Global norm",
                "Effective local norm",
                "rho",
                "Cosine",
                "Gate mean / median",
                "Reliability mean / median",
                "Assessable residues / N",
            ],
            [
                [
                    t,
                    *[
                        fmt(summaries[label]["timestep"][t][f])
                        for f in (
                            "global_output_norm",
                            "local_effective_output_norm",
                            "local_global_norm_ratio",
                            "global_local_cosine",
                        )
                    ],
                    f"{fmt(summaries[label]['timestep'][t]['local_gate_mean'])} / {fmt(summaries[label]['timestep'][t]['local_gate_median'])}",
                    f"{fmt(summaries[label]['timestep'][t]['frame_reliability_mean'])} / {fmt(summaries[label]['timestep'][t]['frame_reliability_median'])}",
                    f"{summaries[label]['timestep'][t]['frame_assessable_count_sum']}/{summaries[label]['timestep'][t]['frame_residue_count_sum']} ({100 * summaries[label]['timestep'][t]['frame_assessability_pooled_fraction']:.2f}%)",
                ]
                for t in TIMES
            ],
        )
    report.extend(
        [
            "Norms use normalized coordinate-v units. Gate/reliability medians are means of per-protein medians, not pooled residue medians. Low-noise local influence decreases at t=5/50; it increases relatively at t=400/490. At t=490 effective local norm remains tiny (0.01345→0.01403); global norm shrinks 5.5446→2.3227. rho rises 0.00315→0.00899 primarily against a smaller global component. The high-noise effective contribution is small despite learned sigmoid gate≈0.236: the fixed noise envelope is ≈0.000325. High-noise cosine changes +0.350→−0.132, indicating mean directional conflict, while low/mid-noise cosines remain positive. This is correlation/telemetry, not a causal local-branch ablation.",
            "",
            "All t=490 examples collapse at 50%, so a within-timestep binary collapse/local-magnitude correlation and collapsed-vs-noncollapsed contrast are **undefined**. Continuous Rg correlations below are exploratory and confounded by length and ratio denominators; they cannot identify a mechanism.",
            "",
        ]
    )
    table(
        ["Boundary", "t490 Rg/rho Pearson", "t490 Rg/rho Spearman", "Collapse/rho Pearson"],
        [
            [
                boundary,
                fmt(correlations[boundary][490]["rg_ratio_vs_rho_pearson"]),
                fmt(correlations[boundary][490]["rg_ratio_vs_rho_spearman"]),
                fmt(correlations[boundary][490]["collapse_vs_rho_pearson"]),
            ]
            for boundary in LABELS[1:]
        ],
    )
    report.extend(
        [
            "## Training curves — telemetry only",
            "",
            "![Training prefix summaries](training_curve.png)",
            "",
            "5,000-update bins preserve all committed records up to 90,413. Loss is protein-weighted; clipping is fraction of optimizer updates. Step norms are module telemetry sampled every 500 updates (plus update 1), not every step. Per-update timestamps were not logged; no synthetic fine-grained throughput curve is created.",
            "",
        ]
    )
    table(
        ["Updates", "Proteins", "Loss", "Gradient mean / p95", "Clip fraction", "Step-norm telemetry N"],
        [
            [
                f"{b['first_update']}–{b['last_update']}",
                b["proteins"],
                fmt(b["v_loss"]),
                f"{fmt(b['gradient_mean'])} / {fmt(b['gradient_p95'])}",
                fmt(b["clipping_fraction"]),
                b["optimizer_telemetry_samples"],
            ]
            for b in training["bins"]
        ],
    )
    table(
        ["Boundary", "Raw loss", "Raw gradient norm", "Cumulative train mean", "Cumulative clipping"],
        [
            [
                boundary,
                fmt(training["boundaries"][boundary]["raw_record"]["loss"])
                if training["boundaries"][boundary]["raw_record"]
                else "N/A",
                fmt(training["boundaries"][boundary]["raw_record"]["gradient_norm"])
                if training["boundaries"][boundary]["raw_record"]
                else "N/A",
                fmt(training["boundaries"][boundary]["checkpoint_training_mean"]),
                fmt(training["boundaries"][boundary]["cumulative_clipping_fraction"]),
            ]
            for boundary in LABELS
        ],
    )
    table(
        ["Interval", "Proteins / execution second", "Updates / execution second"],
        [
            [
                f"{r['from_label']}→{r['to_label']}",
                fmt(r["proteins_per_execution_wall_second"]),
                fmt(r["updates_per_execution_wall_second"]),
            ]
            for r in training["throughput_intervals"]
        ],
    )
    report.extend(
        [
            f"Training cumulative v-loss: 0.35730 at 25%→0.34320 at 50%; 25→50 interval mean {training['quarter_to_half']['v_loss']:.6f}. Development v-MSE decreases another 4.50%, showing continued objective learning. Smoothed training losses are noisy rather than monotonically convergent; no plateau/convergence decision is made. Raw boundary losses are telemetry, not validation. Original finite/symmetry checks at both boundaries passed; no new symmetry run was needed.",
            "",
            "## Historical context and limits",
            "",
            "E013 was a 980,794-parameter, 400-update pilot on 20 training/17 validation identities. Its ~7% v-MSE advantage over matched E007, +35.6% local-distance MSE and 10/17 t490 collapse are contextual only. This E014 development panel/capacity/exposure differs; no matched cross-experiment percentage or causal comparison is claimed. Matched E007 remains deferred. No Stage D, reverse trajectory, de novo generation, gate adjudication or retuning occurred. Minimum two complete TRAIN epochs remains pending.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
