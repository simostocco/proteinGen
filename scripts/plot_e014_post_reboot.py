"""D-resident scientific learning curves from completed checkpoint evaluations."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT  # noqa: F401

import json
import os

from evaluate_e014_post_reboot import OUT, SELECTED

import numpy as np
import pyarrow.parquet as pq

from analyze_e014_post_reboot import ALL, ALIASES, CORE
from protein_distance_diffusion.evaluation.high_noise_resolution import HIGH, STRATA


def main():
    # Configure and verify the D cache before Matplotlib can choose a fallback.
    plot_cache = OUT / "runtime_cache/matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(plot_cache)
    os.environ["TMPDIR"] = str(OUT / "runtime_cache")
    assert os.access(plot_cache, os.W_OK), "Refuse a non-D cache fallback"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((OUT / "epoch_comparison.json").read_text())
    raw = pq.read_table(OUT / "core_metrics.parquet", use_threads=False).to_pylist()
    plots = OUT / "plots"
    plots.mkdir(exist_ok=True)
    order = sorted({(r["exposures"], r["checkpoint"]) for r in raw})
    quantities = [
        ("v_mse", "Coordinate-v MSE"),
        ("local_mse_angstrom2", "Mean local distance MSE (Å²)"),
        ("torsion_mse", "Torsion sin/cos sum-square error"),
        ("triple_product_mse", "Normalized signed triple MSE"),
        ("radius_ratio", "Direct X0 Rg ratio"),
        ("collapsed", "Direct X0 collapse fraction"),
    ]
    for field, ylabel in quantities:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for t in CORE:
            for ax, subset in zip(axes, (order, order[1:]), strict=True):
                y = [
                    np.mean([r[field] for r in raw if r["checkpoint"] == label and r["timestep"] == t])
                    for _, label in subset
                ]
                ax.plot([e / 230132 for e, _ in subset], y, "o-", label=f"t={t}", markersize=4)
        for ax in axes:
            ax.set_xlabel("Durable TRAIN exposures /230,132")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes[0].set_title("All boundaries including initialization")
        axes[1].set_title("Trained checkpoints")
        fig.savefig(plots / f"learning_{field}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(17, 10), layout="constrained")
    for j, t in enumerate(CORE):
        for i, field in enumerate(("offset_1_mse_angstrom2", "offset_2_mse_angstrom2", "offset_3_mse_angstrom2")):
            ax = axes[i, j]
            for s in STRATA:
                y = [
                    np.mean(
                        [
                            r[field]
                            for r in raw
                            if r["checkpoint"] == label and r["timestep"] == t and r["length_stratum"] == s
                        ]
                    )
                    for _, label in order[1:]
                ]
                ax.plot([e / 230132 for e, _ in order[1:]], y, "o-", label=s, markersize=2)
            ax.set_title(f"L{i + 1},t={t}")
            ax.set_xlabel("TRAIN exposure equivalents")
            ax.set_ylabel("Distance MSE (Å²)")
            ax.grid(alpha=0.2)
            if (i, j) == (0, 0):
                ax.legend(fontsize=7)
    fig.savefig(plots / "length_stratified_local_geometry.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    fields = [
        ("v_cosine", "v cosine"),
        ("v_norm_ratio", "v norm ratio"),
        ("step_gain_of_means", "Paired step gain"),
        ("step_cosine", "Step cosine"),
        ("step_norm_ratio", "Step norm ratio"),
        ("local_global_norm_ratio", "Local/global branch norm"),
    ]
    for ax, (field, ylabel) in zip(axes.flat, fields, strict=True):
        for label in ALL[2:]:
            rows = sorted(
                [
                    g
                    for g in data["paired"]
                    if g["checkpoint"] == label and g["length_stratum"] == "pooled" and g["timestep"] in HIGH
                ],
                key=lambda g: g["timestep"],
            )
            x = [
                next(
                    g["log_snr"]
                    for g in data["marginal"]
                    if g["checkpoint"] == label and g["timestep"] == r["timestep"] and g["length_stratum"] == "pooled"
                )
                for r in rows
            ]
            y = [g[field]["mean"] if isinstance(g[field], dict) else g[field] for g in rows]
            ax.plot(x, y, "o-", markersize=3, label=ALIASES[label])
        ax.set_xlabel("log-SNR")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(plots / "dense_high_noise_field.png", dpi=180)
    plt.close(fig)
    for phase, key in (("one_step", "marginal"), ("gaussian_prefix", "gaussian")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
        names = ("energy", "mmd2", "sliced_wasserstein", "rg_wasserstein", "covariance_spectrum_w1")
        for ax, name in zip(list(axes.flat)[:5], names, strict=True):
            for label in ALL[2:]:
                rows = sorted(
                    [
                        g
                        for g in data[key]
                        if g["checkpoint"] == label
                        and g["length_stratum"] == "pooled"
                        and (g["timestep"] in HIGH if phase == "one_step" else True)
                    ],
                    key=lambda g: g["timestep"],
                )
                if not rows:
                    continue
                ms = [next(m for m in g["metrics"] if m["metric"] == name) for g in rows]
                x = [g["timestep"] for g in rows]
                y = [m["model_ref"] - m["noop_ref"] for m in ms]
                ax.plot(x, y, "o-", label=ALIASES[label], markersize=3)
                ax.fill_between(
                    x, [m["model_minus_noop_ci"][0] for m in ms], [m["model_minus_noop_ci"][1] for m in ms], alpha=0.08
                )
            ax.axhline(0, ls="--", color="gray")
            ax.set_xlabel("Timestep")
            ax.set_ylabel(f"{name}: D_model−D_noop")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        ax = axes.flat[5]
        for label in SELECTED:
            rows = sorted(
                [g for g in data[key] if g["checkpoint"] == label and g["length_stratum"] == "pooled"],
                key=lambda g: g["timestep"],
            )
            ax.plot([g["timestep"] for g in rows], [g["rg_mean_ratio"] for g in rows], "o-", label=ALIASES[label])
            ax.plot([g["timestep"] for g in rows], [g["trace_mean_ratio"] for g in rows], ":", alpha=0.7)
        ax.axhline(0.8, color="gray", ls="--")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Rg (solid) / trace (dotted) reference ratio")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        fig.suptitle(
            "Independent marginal transitions" if phase == "one_step" else "Centered-Gaussian prefixes; stop350"
        )
        fig.savefig(plots / f"{phase}_effects.png", dpi=180)
        plt.close(fig)
    training = data["training"]["bins"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, field, ylabel in zip(
        axes,
        ("training_v_loss", "gradient_norm_mean", "clipping_fraction"),
        ("Training v-loss", "Raw gradient norm", "Clipping fraction"),
        strict=True,
    ):
        ax.plot([g["last_update"] for g in training], [g[field] for g in training])
        ax.set_xlabel("Durable optimizer update")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    fig.savefig(plots / "training_telemetry.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
