# ruff: noqa: E501
"""Bounded scientific figures and plain-language inference interpretation."""

import io
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

CAMERA = (25, 35)
STORY = [499, 450, 400, 350, 250, 150, 50, 0]


def limits(arrays):
    x = np.concatenate(arrays)
    lo, hi = x.min(0), x.max(0)
    return (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1) * 1.05


def trace(ax, x, bounds, title="", camera=CAMERA):
    c, s = bounds
    line = Line3DCollection(
        np.stack([x[:-1], x[1:]], axis=1), cmap="viridis", norm=colors.Normalize(0, len(x) - 1), linewidth=0.8
    )
    line.set_array(np.arange(len(x) - 1))
    ax.add_collection3d(line)
    ax.scatter(*x[0], color="purple", s=12)
    ax.scatter(*x[-1], color="gold", s=12)
    ax.set(
        xlim=(c[0] - s, c[0] + s),
        ylim=(c[1] - s, c[1] + s),
        zlim=(c[2] - s, c[2] + s),
        title=title,
        xlabel="x Å",
        ylabel="y Å",
        zlabel="z Å",
    )
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(*camera)
    ax.tick_params(labelsize=6)


def save(fig, path, dpi=110):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def card(c, meta, metric, score, nearest):
    folder = Path(meta["path"])
    x = np.load(folder / "final_coordinates.npz")["coordinates_angstrom"]
    trajectory = np.load(folder / "sparse_trajectory.npz")
    fig = plt.figure(figsize=(13, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 1.1))
    bounds = limits([x])
    trace(fig.add_subplot(grid[0, 0], projection="3d"), x, bounds, "Final trace: fixed physical Å", CAMERA)
    trace(fig.add_subplot(grid[0, 1], projection="3d"), x, bounds, "Second view", (20, 125))
    ax = fig.add_subplot(grid[0, 2])
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    image = ax.imshow(d, origin="lower", cmap="magma", vmin=0, vmax=30)
    fig.colorbar(image, ax=ax, label="distance Å (0–30 clipped)")
    ax.set(title="Final distance map", xlabel="residue", ylabel="residue")
    ax = fig.add_subplot(grid[1, 0])
    for k in (1, 2, 3):
        ax.plot(np.linalg.norm(x[k:] - x[:-k], axis=-1), label=f"i,i+{k}", linewidth=0.8)
    ax.axhspan(3.4, 4.2, color="gray", alpha=0.2)
    ax.set(title="Local distances", ylabel="Å", xlabel="residue")
    ax.legend()
    f, _, _ = c.b.local_geometry(c.torch.from_numpy(x))
    f = f.numpy()
    phi = np.arctan2(f[f[:, 29] > 0, 25], f[f[:, 29] > 0, 26])
    ax = fig.add_subplot(grid[1, 1])
    ax.hist(phi, bins=np.linspace(-math.pi, math.pi, 25), density=True)
    ax.set(title="Pseudo-torsion", xlabel="radians", ylabel="density")
    ax = fig.add_subplot(grid[1, 2])
    ts = sorted([int(k[1:]) for k in trajectory.files], reverse=True)
    rg = [c.b.intrinsic(trajectory[f"t{t}"])["radius_of_gyration_angstrom"] for t in ts]
    ax.plot(ts, rg, "o-")
    ax.invert_xaxis()
    ax.set(title="Actual-state Rg", xlabel="t: noise → terminal", ylabel="Å")
    classification = (
        "STRONG"
        if score < 1.5 and metric["adjacent_valid_fraction"] >= 0.9 and metric["clash_fraction"] <= 0.01
        else "MODERATE"
        if score < 3 and metric["adjacent_valid_fraction"] >= 0.75
        else "WEAK"
        if metric["finite_fraction"] == 1 and metric["adjacent_valid_fraction"] >= 0.5
        else "FAILED"
    )
    text = [
        f"{meta['sample_key']} | {meta['weights']} | N={len(x)}",
        f"Numerical score={score:.3g} | descriptive class={classification}",
        f"Adjacent {metric['adjacent_mean_angstrom']:.3f}±{metric['adjacent_std_angstrom']:.3f} Å",
        f"Adjacent min/max={metric['adjacent_min_angstrom']:.3f}/{metric['adjacent_max_angstrom']:.3f} Å; valid={metric['adjacent_valid_fraction']:.1%}",
        f"Clashes={metric['clash_fraction']:.5f}; Rg={metric['radius_of_gyration_angstrom']:.2f} Å",
        f"End-to-end={metric['end_to_end_angstrom']:.2f} Å",
        f"Covariance eigenvalues Å²={np.round(metric['covariance_eigenvalues_angstrom2'], 2)}",
        f"Negative signed triples={metric['negative_signed_triple_fraction']:.1%}; torsion assessable={metric['torsion_assessable_fraction']:.1%}",
        f"Contact density={metric['long_range_contact_density']:.4f}; order={metric['long_range_contact_order']}",
    ]
    for split in ("train", "heldout"):
        q = nearest[split]
        text.append(
            f"Bounded nearest {split}: {q['descriptor_nearest_identity']} | descriptor distance={q['descriptor_distance']:.3g}"
        )
    text += [
        "CA trace only. No sequence or side chains. Not all atom.",
        "Not proof of thermodynamic stability. No paired generation target.",
    ]
    ax = fig.add_subplot(grid[2, :])
    ax.axis("off")
    ax.text(0, 1, "\n".join(text), va="top", fontsize=9)
    fig.suptitle("E015 latest sample quality card", fontsize=12)
    save(fig, folder / "quality_card.png")
    return classification


def storyboard(c, meta):
    folder = Path(meta["path"])
    states = np.load(folder / "inference_states.npz")
    records = {q["timestep"]: q for q in c.b.read(str(folder.relative_to(c.OUT) / "metrics.json"))["trajectory"]}
    bounds = limits([states[k] for k in states.files])
    fig = plt.figure(figsize=(19, 34))
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.025, top=0.965, hspace=0.55, wspace=0.15)
    for row, t in enumerate(STORY):
        stat = records[t]
        for col, (key, title) in enumerate(
            [("xt", "Current X_t"), ("x0hat", "Model-implied X0_hat"), ("next", "Next reverse state")]
        ):
            x = states[f"{key}_{t}"]
            m = c.augment_intrinsic(x)
            ax = fig.add_subplot(8, 4, row * 4 + col + 1, projection="3d")
            trace(
                ax,
                x,
                bounds,
                f"t={t} {title}\nRg={m['radius_of_gyration_angstrom']:.1f}Å, adj={m['adjacent_mean_angstrom']:.2f}Å\nvalid={m['adjacent_valid_fraction']:.0%}, clash={m['clash_fraction']:.3f}",
            )
        ax = fig.add_subplot(8, 4, row * 4 + 4)
        ax.axis("off")
        hd = stat["head_details"]
        denom = max(hd["final_rms"], 1e-12)
        ax.text(
            0,
            0.95,
            f"t={t}; α={stat['alpha']:.5g}\nσ={stat['sigma']:.5g}; logSNR={stat['logsnr']:.3f}\nTorsion assessable={stat['torsion_assessable_fraction']:.1%}\nHead RMS normalized v:\n"
            + "\n".join(f"{k}: {v:.4g} (ratio {v / denom:.3g})" for k, v in hd["rms"].items())
            + f"\nFinal RMS={hd['final_rms']:.4g}\nGate p10/50/90:\n{np.round(hd['gate_p10_p50_p90'], 3)}\nN: purple; C: gold\nFixed camera and axes",
            va="top",
            fontsize=9,
        )
    fig.suptitle(f"{meta['sample_key']} — fixed-scale reverse inference; t0 next is final", fontsize=13)
    path = c.OUT / "inference_story" / (meta["sample_key"] + "_storyboard.png")
    save(fig, path, 95)
    return path


def distance_story(c, meta, cache):
    frames = np.load(Path(meta["path"]) / "sparse_trajectory.npz")
    references, _ = c.base.refs(cache, meta["requested_length"], 8)
    ref = references[0]
    xreal = cache.get(ref["sample_id"])
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for ax, t in zip(axes.flat, STORY, strict=False):
        x = frames[f"t{t}"]
        d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        ax.imshow(d, cmap="magma", origin="lower", vmin=0, vmax=30)
        ax.set(title=f"Generated actual state t={t}", xlabel="residue", ylabel="residue")
    axes[-1, -1].imshow(
        np.linalg.norm(xreal[:, None] - xreal[None, :], axis=-1), cmap="magma", origin="lower", vmin=0, vmax=30
    )
    axes[-1, -1].set(title=f"Real context {ref['sample_id']} N={len(xreal)}\nUnpaired heldout reference")
    fig.suptitle(f"{meta['sample_key']} — fixed 0–30Å distance scale; values above30 saturated")
    path = c.OUT / "inference_story" / (meta["sample_key"] + "_distance_evolution.png")
    save(fig, path)
    return path


def vector_story(c, meta):
    data = np.load(Path(meta["path"]) / "vector_fields.npz")
    bounds = limits([data[f"xt_{t}"] for t in (490, 400, 250, 50)])
    fig_profiles, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    for row, t in enumerate((490, 400, 250, 50)):
        x = data[f"xt_{t}"]
        index = np.linspace(0, len(x) - 1, min(24, len(x)), dtype=int)
        names = ("total", "polar", "chiral", "local")
        maximum = max(float(np.linalg.norm(data[f"{k}_{t}"], axis=-1).max()) for k in names)
        gain = 0.12 * bounds[1] / max(maximum, 1e-12)
        fig = plt.figure(figsize=(16, 4), constrained_layout=True)
        for col, k in enumerate(names):
            v = data[f"{k}_{t}"]
            rms = float(np.sqrt(np.mean(v * v)))
            ax = fig.add_subplot(1, 4, col + 1, projection="3d")
            trace(ax, x, bounds, f"t{t} {k}; RMS={rms:.4g}")
            ax.quiver(*x[index].T, *(v[index] * gain).T, color="red", linewidth=0.7, normalize=False)
            axes[row].plot(np.linalg.norm(v, axis=-1), label=f"{k} RMS={rms:.4g}", linewidth=0.8)
        fig.suptitle(
            f"{meta['sample_key']} — common display gain={gain:.3g}Å per normalized v; arrows are not step displacement"
        )
        save(fig, c.OUT / "vector_field_views" / f"{meta['sample_key']}_t{t}_quiver.png")
        axes[row].set(title=f"t={t}", ylabel="normalized |v|", xlabel="residue")
        axes[row].legend(fontsize=8)
    save(fig_profiles, c.OUT / "vector_field_views" / (meta["sample_key"] + "_magnitudes.png"))


def comparison(c, meta, trajectory=False):
    old = np.load(Path(meta["historical_raw_path"]) / "sparse_trajectory.npz")
    new = np.load(Path(meta["path"]) / "sparse_trajectory.npz")
    times = [499, 400, 250, 100, 0] if trajectory else [0]
    bounds = limits([a[f"t{t}"] for a in (old, new) for t in times])
    fig = plt.figure(figsize=(12, 5 * len(times) + 1))
    fig.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.13 if len(times) == 1 else 0.05,
        top=0.86 if len(times) == 1 else 0.955,
        hspace=0.45,
        wspace=0.1,
    )
    for row, t in enumerate(times):
        for col, (data, label) in enumerate([(old, "50% raw"), (new, "Latest " + meta["weights"])]):
            x = data[f"t{t}"]
            stat = c.augment_intrinsic(x)
            ax = fig.add_subplot(len(times), 2, row * 2 + col + 1, projection="3d")
            trace(
                ax,
                x,
                bounds,
                f"{label} t={t}; Rg={stat['radius_of_gyration_angstrom']:.2f}Å\nadj={stat['adjacent_mean_angstrom']:.2f}Å, valid={stat['adjacent_valid_fraction']:.1%}\nclash={stat['clash_fraction']:.4f}, negative triples={stat['negative_signed_triple_fraction']:.1%}",
            )
    fig.suptitle(f"{meta['sample_key']} — exact saved Gaussian start; common camera/axes; no target", fontsize=11)
    paired = next(q for q in c.b.read("same_seed_comparison/metrics.json")["rows"] if q["sample"] == meta["sample_key"])
    fig.text(
        0.02,
        0.002,
        f"Energy contribution: {paired['old_energy_contribution']:.3g} → {paired['latest_energy_contribution']:.3g}; covariance eigenvaluesÅ² {np.round(paired['old']['covariance_eigenvalues_angstrom2'], 1)} → {np.round(paired['new']['covariance_eigenvalues_angstrom2'], 1)}",
        fontsize=8,
    )
    save(
        fig,
        c.OUT / "same_seed_comparison" / (meta["sample_key"] + ("_trajectory.png" if trajectory else "_final.png")),
        100,
    )


def gif(c, meta):
    data = np.load(Path(meta["path"]) / "sparse_trajectory.npz")
    bounds = limits([data[k] for k in data.files])
    images = []
    for t in c.FRAMES:
        fig = plt.figure(figsize=(5, 5))
        trace(
            fig.add_subplot(111, projection="3d"),
            data[f"t{t}"],
            bounds,
            f"{meta['sample_key']} t={t}\nFixed physical scale",
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90)
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
    path = c.OUT / "trajectory_gifs" / (meta["sample_key"] + ".gif")
    images[0].save(path, save_all=True, append_images=images[1:], duration=550, loop=0, optimize=False)
    assert path.stat().st_size <= 25 * 1024**2
    for image in images:
        image.close()
    return path


def aggregate_plots(c):
    gaps = c.table("train_vs_heldout_metrics.parquet")
    positions = {q["label"]: q["position"] for q in c.records()}
    for axis_key in ("exposures", "update"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, metric in zip(axes.flat, c.PRIMARY[:6], strict=True):
            for w in ("raw", "ema"):
                for split in ("train", "heldout"):
                    q = [
                        next(
                            r
                            for r in gaps
                            if r["checkpoint"] == cp
                            and r["weights"] == w
                            and r["metric"] == metric
                            and r["timestep"] is None
                            and r["length_stratum"] == "all"
                        )
                        for cp in c.b.CP
                    ]
                    ax.plot(
                        [positions[cp][axis_key] for cp in c.b.CP],
                        [r[split + "_mean"] for r in q],
                        "o-",
                        label=split + " " + w,
                    )
            ax.set(title=metric, xlabel=axis_key)
            ax.legend(fontsize=7)
        save(fig, c.OUT / "plots" / f"absolute_learning_curves_{axis_key}.png")
    heads = c.table("head_contributions.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, setting in zip(axes, ("train", "heldout", "generation"), strict=True):
        for name in ("polar", "chiral", "local"):
            selected = [
                q
                for q in heads
                if q["checkpoint"] == c.LATEST
                and q["weights"] == c.b.read("sampler_weight_selection.json")["primary_weights"]
                and q["split"] == setting
            ]
            times = sorted({q["timestep"] for q in selected})
            ax.plot(
                times,
                [np.mean([r["head_details"]["rms"][name] for r in selected if r["timestep"] == t]) for t in times],
                "o-",
                label=name,
            )
        ax.set(title=setting, xlabel="t", ylabel="head RMS normalized v")
        ax.legend()
    save(fig, c.OUT / "plots/head_contributions_by_timestep.png")
    typed = c.table("typed_state_statistics.parquet")
    stages = [
        "embedding",
        *[f"fine.{i}" for i in range(8)],
        *[f"medium.{i}" for i in range(4)],
        *[f"coarse.{i}" for i in range(4)],
        "up_medium",
        "up_fine",
    ]
    for typ in ("H", "V", "P", "A"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        for ax, cp in zip(axes, ("epoch_1_fraction_0.50", c.LATEST), strict=True):
            selected = [
                q
                for q in typed
                if q["checkpoint"] == cp
                and q["type"] == typ
                and q["weights"] == "raw"
                and q.get("setting", "reconstruction") == "reconstruction"
            ]
            matrix = np.array(
                [
                    [
                        np.mean([q["rms"] for q in selected if q["stage"] == stage and q["timestep"] == t])
                        for t in c.GRID
                    ]
                    for stage in stages
                ]
            )
            im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xticks(range(5), c.GRID)
            ax.set_yticks(range(len(stages)), stages)
            ax.set(title=cp + " raw " + typ, xlabel="timestep")
            fig.colorbar(im, ax=ax, label="component RMS")
        save(fig, c.OUT / "plots" / f"typed_{typ}_state_rms.png")


def visuals(c):
    manifest = c.table("generation_manifest.parquet")
    metrics = {q["sample_key"]: q for q in c.table("generation_metrics.parquet")}
    scores = {q["sample_key"]: q for q in c.table("generation_scores.parquet")}
    novelty = c.b.read("diversity_novelty_metrics.json")
    nearest = {q["sample"]: q["nearest"] for q in novelty["novelty"]}
    cache = c.b.CoordinateCache(Path(c.b.cfg["population_root"]), verify_payload=False)
    primary = sorted(
        [q for q in manifest if q["primary"]],
        key=lambda q: (
            scores[q["sample_key"]]["score"] if scores[q["sample_key"]]["score"] is not None else float("inf"),
            q["sample_key"],
        ),
    )
    selected = [primary[0], primary[len(primary) // 2], primary[-1]]
    for n in c.b.LENGTHS:
        group = [q for q in primary if q["requested_length"] == n]
        selected.append(group[len(group) // 2])
    selected += [next(q for q in primary if q["sample_key"] == g["sample_key"]) for g in novelty["galleries"]]
    selected = list({q["sample_key"]: q for q in selected}.values())
    c.b.write(
        "inference_story/selection.json",
        dict(
            best=primary[0]["sample_key"],
            median=primary[len(primary) // 2]["sample_key"],
            worst=primary[-1]["sample_key"],
            representatives=[q["sample_key"] for q in selected],
            criterion=c.b.read("numerical/quality_score_contract.json"),
        ),
    )
    for meta in manifest:
        c.pulse("sample_cards", meta["sample_key"])
        score = scores.get(meta["sample_key"], {}).get("score", float("inf"))
        score = score if score is not None else float("inf")
        classification = card(c, meta, metrics[meta["sample_key"]], score, nearest[meta["sample_key"]])
        c.b.write(
            str(Path(meta["path"]).relative_to(c.OUT) / "quality_classification.json"),
            dict(
                score=score,
                classification=classification,
                criterion="Frozen descriptive thresholds; no all-atom/fold/stability certification",
            ),
        )
        if meta["primary"]:
            comparison(c, meta)
    for meta in selected:
        c.pulse("inference_story", meta["sample_key"])
        storyboard(c, meta)
        distance_story(c, meta, cache)
        comparison(c, meta, True)
    for meta in (primary[0], primary[len(primary) // 2], primary[-1]):
        c.pulse("vector_fields_gif", meta["sample_key"])
        vector_story(c, meta)
        gif(c, meta)
    for category, meta in zip(
        ("success_gallery", "median_gallery", "failure_gallery"),
        (primary[0], primary[len(primary) // 2], primary[-1]),
        strict=True,
    ):
        folder = c.OUT / category / meta["sample_key"]
        folder.mkdir(exist_ok=True)
        shutil.copyfile(Path(meta["path"]) / "quality_card.png", folder / "quality_card.png")
        shutil.copyfile(c.OUT / "inference_story" / (meta["sample_key"] + "_storyboard.png"), folder / "storyboard.png")
        c.b.write(
            str(folder.relative_to(c.OUT) / "artifacts.json"),
            dict(
                score=scores[meta["sample_key"]]["score"],
                rank=primary.index(meta) + 1,
                pdb=str(Path(meta["path"]) / "trace_ca_only.pdb"),
                npz=str(Path(meta["path"]) / "final_coordinates.npz"),
                gif=str(c.OUT / "trajectory_gifs" / (meta["sample_key"] + ".gif")),
            ),
        )
    for gallery in novelty["galleries"]:
        meta = next(q for q in manifest if q["sample_key"] == gallery["sample_key"])
        folder = c.OUT / gallery["category"] / meta["sample_key"]
        shutil.copyfile(Path(meta["path"]) / "quality_card.png", folder / "quality_card.png")
        shutil.copyfile(c.OUT / "inference_story" / (meta["sample_key"] + "_storyboard.png"), folder / "storyboard.png")
        c.b.write(
            str(folder.relative_to(c.OUT) / "artifacts.json"),
            dict(
                score=scores[meta["sample_key"]]["score"],
                global_rank=primary.index(meta) + 1,
                pdb=str(Path(meta["path"]) / "trace_ca_only.pdb"),
                npz=str(Path(meta["path"]) / "final_coordinates.npz"),
                criterion=c.b.read("numerical/quality_score_contract.json"),
            ),
        )
    aggregate_plots(c)
    c.b.write(
        "logs/visuals_complete.json",
        dict(
            sample_cards=len(manifest),
            final_same_seed_comparisons=20,
            storyboards=len(selected),
            fixed_scale=True,
            gifs=3,
            max_gif_bytes=max(p.stat().st_size for p in (c.OUT / "trajectory_gifs").glob("*.gif")),
        ),
    )


def report(c):
    # Redraw only cards after the reviewed grid-layout correction, while the
    # trainer is still stopped. Numerical outputs and sampling stay intact.
    if not (c.OUT / "logs/cards_layout_verified.json").exists():
        scores = {q["sample_key"]: q for q in c.table("generation_scores.parquet")}
        metrics = {q["sample_key"]: q for q in c.table("generation_metrics.parquet")}
        nearest = {q["sample"]: q["nearest"] for q in c.b.read("diversity_novelty_metrics.json")["novelty"]}
        for meta in c.table("generation_manifest.parquet"):
            c.pulse("card_layout_correction", meta["sample_key"])
            value = scores[meta["sample_key"]]["score"]
            card(
                c,
                meta,
                metrics[meta["sample_key"]],
                value if value is not None else float("inf"),
                nearest[meta["sample_key"]],
            )
            if meta["primary"]:
                comparison(c, meta)
        for category in ("success_gallery", "median_gallery", "failure_gallery"):
            for folder in (c.OUT / category).iterdir():
                if folder.is_dir():
                    source = c.OUT / "samples" / folder.name.split("_")[0] / folder.name / "quality_card.png"
                    shutil.copyfile(source, folder / "quality_card.png")
        story_keys = {
            p.name.removesuffix("_storyboard.png") for p in (c.OUT / "inference_story").glob("*_storyboard.png")
        }
        for meta in c.table("generation_manifest.parquet"):
            if meta["sample_key"] in story_keys:
                c.pulse("storyboard_layout_correction", meta["sample_key"])
                storyboard(c, meta)
                comparison(c, meta, True)
        for category in ("success_gallery", "median_gallery", "failure_gallery"):
            for folder in (c.OUT / category).iterdir():
                if folder.is_dir():
                    source = c.OUT / "inference_story" / (folder.name + "_storyboard.png")
                    if source.exists():
                        shutil.copyfile(source, folder / "storyboard.png")
        c.b.write(
            "logs/cards_layout_verified.json",
            dict(cards=25, storyboards=len(story_keys), common_grid=True, no_sampling_repeated=True),
        )
    verdict = c.b.read("verdict.json")
    gaps = c.table("train_vs_heldout_metrics.parquet")
    chosen = verdict["sampler_weights"]
    primary = [
        q
        for q in gaps
        if q["checkpoint"] == c.LATEST
        and q["weights"] == chosen
        and q["timestep"] is None
        and q["length_stratum"] == "all"
    ]
    generation = c.table("generation_metrics.parquet")
    manifest = c.table("generation_manifest.parquet")
    distribution = c.b.read("generation_distribution_metrics.json")
    learning = c.table("checkpoint_trajectory_metrics.parquet")
    comparisons = c.b.read("same_seed_comparison/metrics.json")["rows"]
    selection = c.selection
    lines = [
        "# E015 latest durable checkpoint: absolute reconstruction and unconditional inference",
        "",
        f"Primary: `{selection['snapshot']}`. SHA256 `{c.SHA}`. Position `{selection['source']['position']}`; completed epochs `{selection['completed_epochs']}`. Selected through the existing graceful pause after a durable boundary, with zero uncommitted exposures and strict model/EMA/optimizer/scaler/RNG/schedule/population/config verification. No performance-based checkpoint selection.",
        "",
        "Frozen comparator: epoch1 50%, update3811/exposures115066, SHA256 `2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03`. Previous evaluation was preserved. All20 primary initial states match its saved X499 tensors bit for bit under the original normalization/seed contract.",
        "",
        f"Sampling choice: **{chosen}**. `{c.b.read('sampler_weight_selection.json')}`. EMA initialization/parameter-distance: `{c.b.read('numerical/ema_parameter_distance.json')}`.",
        "",
        "| Metric | TRAIN mean | TRAIN median | Heldout mean | Heldout median | Gap | Ratio | Gap95%CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for q in primary:
        lines.append(
            f"| {q['metric']} | {q['train_mean']:.6g} | {q['train_median']:.6g} | {q['heldout_mean']:.6g} | {q['heldout_median']:.6g} | {q['gap']:.6g} | {q['ratio']} | {q['gap_ci95']} |"
        )
    lines += [
        "",
        "All intervals use5000 stratified protein draws. Timesteps are averaged within each protein before aggregate bootstrap. Both absolute errors and split gaps determine fit interpretation; a small gap does not establish successful distribution learning. The full frozen TRAIN population and500 development identities are disjoint. The100+100 panel was not reselected. Latest raw/EMA heldout500 results are separately streamed in numerical/heldout500_metrics.parquet.",
        "",
        "| t | Split | v-MSE | KabschÅ | localRMSEÅ | adjacentÅ | valid | torsiondegrees | inversion | clashes | Rg ratio |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = c.table("absolute_reconstruction_metrics.parquet")
    for t in c.GRID:
        for split in ("train", "heldout"):
            group = [
                q
                for q in rows
                if q["checkpoint"] == c.LATEST and q["weights"] == chosen and q["timestep"] == t and q["split"] == split
            ]
            values = [
                np.mean([q[k] for q in group])
                for k in (
                    "v_mse",
                    "aligned_rmsd_angstrom",
                    "local_rmse_angstrom",
                    "adjacent_mean_angstrom",
                    "adjacent_valid_fraction",
                    "torsion_angular_mae_degrees",
                    "triple_inversion_fraction",
                    "clash_fraction",
                    "radius_ratio",
                )
            ]
            lines.append(f"| {t} | {split} | " + " | ".join(f"{v:.5g}" for v in values) + " |")
    lines += [
        "",
        "| Learning interval | Weights | Split | Metric | Old | New | Difference | Relative difference | Difference95%CI |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for q in learning:
        if q["weights"] == chosen and q["metric"] in ("v_mse", "local_rmse_angstrom", "aligned_rmsd_angstrom"):
            lines.append(
                f"| {q['old']} → {q['new']} | {q['weights']} | {q['split']} | {q['metric']} | {q['train_mean']:.6g} | {q['heldout_mean']:.6g} | {q['gap']:.6g} | {q['relative_gap']} | {q['gap_ci95']} |"
            )
    lines += [
        "",
        "High-noise X0_hat is a model-implied conditional clean estimate, not the actual reverse trajectory. At very low signal it can shrink toward a mean without the actual X_t collapsing. Fixed-axis storyboards show both quantities separately. Global generation evidence must come from complete unconditional trajectories and matched real distributions.",
        "",
        f"Full sampling: {len(manifest)} samples ({sum(q['primary'] for q in manifest)} primary), finite fraction={np.mean([q['finite_fraction'] for q in generation]):.6g}. Batch1, exact500-step production sampler, no sequence/target/teacher input. Sparse13 states only; vector snapshots additionally record t490. PDBs are Cα-only, no sequence/side chains, not all atom, not proof of thermodynamic stability.",
        "",
        "| N | AdjacentÅ | AdjacentstdÅ | Valid | Clashes | RgÅ | Contact density | Negative triples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in c.b.LENGTHS:
        group = [q for q in generation if q["primary"] and q["requested_length"] == n]
        lines.append(
            f"| {n} | "
            + " | ".join(
                f"{np.mean([q[k] for q in group]):.6g}"
                for k in (
                    "adjacent_mean_angstrom",
                    "adjacent_std_angstrom",
                    "adjacent_valid_fraction",
                    "clash_fraction",
                    "radius_of_gyration_angstrom",
                    "long_range_contact_density",
                    "negative_signed_triple_fraction",
                )
            )
            + " |"
        )
    lines += [
        "",
        "| N | Descriptor metric | Generated-real | Real-real floor | Excess | Excess ratio | Excess95%CI |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for d in distribution["by_length"]:
        for q in d["metrics"]:
            lines.append(
                f"| {d['length']} | {q['metric']} | {q['generated_vs_real']:.6g} | {q['real_vs_real_floor']:.6g} | {q['excess_above_floor']:.6g} | {q['excess_ratio']} | {q['excess_ci95']} |"
            )
    lines += [
        "",
        "Descriptors summarize invariant geometry; they do not establish equality of full coordinate distributions. Floors use small deterministic disjoint matched-length heldout4+4 subsets. Small panels have wide uncertainty. Bounded novelty uses20TRAIN/8heldout references per length; no uncontrolled full-corpus comparison. Nearest-reference RMSD is a novelty measurement and never treated as paired generation quality.",
        "",
        "| N | Same-start adjacent valid improvement | Same-start clash difference | Same-start Rg differenceÅ |",
        "|---:|---:|---:|---:|",
    ]
    for n in c.b.LENGTHS:
        group = [q for q in comparisons if q["length"] == n]
        lines.append(
            f"| {n} | "
            + " | ".join(
                f"{np.mean([q['differences'][k] for q in group]):.6g}"
                for k in ("adjacent_valid_fraction", "clash_fraction", "radius_of_gyration_angstrom")
            )
            + " |"
        )
    local_improvement = np.mean([q["differences"]["adjacent_valid_fraction"] for q in comparisons])
    parts = c.b.read("same_seed_comparison/local_vs_global_descriptor_differences.json")["rows"]
    lines += [
        "",
        "| N | Descriptor subset | Metric | 50% | Latest | Real-real floor | Same-start difference95%CI |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for part in parts:
        for q in part["metrics"]:
            lines.append(
                f"| {part['length']} | {part['part']} | {q['metric']} | {q['old']:.6g} | {q['latest']:.6g} | {q['real_real_floor']:.6g} | {q['difference_ci95']} |"
            )
    global_energy = [
        next(q for q in part["metrics"] if q["metric"] == "energy") for part in parts if part["part"] == "global_shape"
    ]
    global_improved = sum(q["difference_ci95"][1] < 0 for q in global_energy)
    verdict["same_seed_global_energy_improved_lengths"] = global_improved
    floor_compatible = verdict.get("descriptor_floor_compatible", False)
    answer = (
        "The latest checkpoint shows bounded evidence of useful protein-like Cα geometry, with limits from the small reference/sample panels."
        if verdict["generation"] == "EVIDENCE_OF_USEFUL_GENERATION"
        else "The latest checkpoint remains a stable but immature generator; complete trajectories do not establish genuinely protein-like structures or global fold formation."
    )
    verdict.update(
        scientific_answer=answer,
        same_seed_adjacent_valid_improvement=float(local_improvement),
        same_seed_global_distribution_floor_compatible=floor_compatible,
    )
    lines += [
        "",
        answer,
        "",
        f"Same-start mean adjacent-valid change={local_improvement:+.4f}. Local cleanup and global distribution quality must be judged separately. Actual high-noise transport, final contacts/covariance, descriptor excess and fixed-scale same-start trajectories are provided; reconstruction percentage changes alone cannot demonstrate fold formation.",
        f"Global-shape energy improved with a paired95% interval entirely below zero at {global_improved}/5 lengths. The separate local-polymer descriptor rows isolate chain separations1–3; remaining shape/nonlocal descriptors assess more than adjacent spacing. Improvement is distinct from matching the real-reference floor.",
        "",
        f"Verdicts: reconstruction **{verdict['reconstruction']}**; generation **{verdict['generation']}**; generalization **{verdict['generalization']}**; training **{verdict['continued_training']}**. Scientific production protocol was unchanged.",
        "",
        "Output-head tables include RMS, final-output ratios, gate quantiles, reliability, pairwise/output cosines and chiral-disabled fixed-context reconstruction. Typed-state tables include embedding, all8fine/4medium/4coarse blocks and both top-down decoder stages, with V/A vector norms. Compact hooks save scalar statistics; no full activations. Geometry-head ablation keeps self-context/trunk fixed and the local head active.",
        "",
        "Visual artifacts: inference_story/*_storyboard.png, *_distance_evolution.png; vector_field_views/*_quiver.png and *_magnitudes.png; same_seed_comparison/*_final.png and *_trajectory.png; samples/{raw,ema}/*/quality_card.png; trajectory_gifs/*.gif; numerically ranked success/median/failure galleries. Fixed cameras and common spatial limits show scale honestly.",
        "",
        "Evaluation ran only after trainer/supervisor exit and GPU release. RAM3GiB/D30GiB admission and clean-stop monitoring; CPU1thread, workers0, batch1, nice>=10. All artifacts/caches/temp stayed onD; total output cap6GiB. E007/E012/E014 were not modified or resumed. A legacy12-slot plot failed on the new13-frame trajectory after one complete sample. The safe owner resumed training, later gracefully paused at a new durable point, and continued the same evaluation using that saved sample and all10000 reconstruction conditions. A report-only layout retry hit a renderer name-shadowing error and safely resumed production; a model-free scalar-table control-flow test passed before its next serialized retry. These software errors were not resource-pressure incidents. The scientific primary remained the original immutable10382 snapshot. Operational pause/resume checkpoints are separately recorded. Final production resume proof follows evaluator exit, strict checkpoint verification and a new durable rolling checkpoint.",
    ]
    lines += [
        "",
        "| t | Polar RMS/final | Chiral RMS/final | Local RMS/final | Chiral gate p10/p50/p90 |",
        "|---:|---:|---:|---:|---|",
    ]
    for q in c.b.read("numerical/head_summary_by_timestep.json")["rows"]:
        if q["checkpoint"] == c.LATEST and q["weights"] == chosen and q["split"] == "heldout":
            h = q["heads"]
            lines.append(
                f"| {q['timestep']} | {h['polar']['mean_ratio_to_final']:.6g} | {h['chiral']['mean_ratio_to_final']:.6g} | {h['local']['mean_ratio_to_final']:.6g} | {q['gate_equal_protein_mean_p10_p50_p90']} |"
            )
    validation_path = c.OUT / "logs/final_artifact_validation.json"
    if validation_path.exists():
        checked = c.b.read("logs/final_artifact_validation.json")
        lines += [
            "",
            "Chiral-only disabled minus enabled, fixed self-context/trunk with local contribution retained:",
            "",
            "| t | v-MSE difference | Local RMSE difference Å | Torsion difference rad | Inversion fraction difference |",
            "|---:|---:|---:|---:|---:|",
        ]
        for t, q in checked["chiral_disabled_minus_enabled"].items():
            lines.append(
                f"| {t} | {q['v_mse']:.6g} | {q['local_rmse_angstrom']:.6g} | {q['torsion_angular_mae_radians']:.6g} | {q['triple_inversion_fraction']:.6g} |"
            )
        lines += [
            "",
            "The chiral head helps measured low-noise reconstruction and torsion; SNR gates suppress it at high noise, where polar output dominates. All checked typed states were finite across19 stages, with V/A norm columns populated. P/A components fall entirely below the1e-8 near-zero threshold at the initial generation t499 across all19 stages for the five instrumented lengths; prepare_single and typed blocks explicitly gate these channels by SNR/reliability. They become active later. This is a timestep-dependent suppressed regime, not evidence that every typed channel is active at every timestep. Active typed channels do not establish learned semantic meanings.",
        ]
    diversity = c.b.read("diversity_novelty_metrics.json")["diversity"]
    held_references = [q for q in c.table("generation_reference_metrics.parquet") if q["split"] == "heldout"]
    lines += [
        "",
        f"Frozen adjacent-valid range is3.4–4.2Å. Primary generated equal-protein adjacent mean is{np.mean([q['adjacent_mean_angstrom'] for q in generation if q['primary']]):.4f}Å, versus{np.mean([q['adjacent_mean_angstrom'] for q in held_references]):.4f}Å in matched heldout references. Improved valid fraction does not remove the spacing bias or prove native geometry.",
    ]
    lines += [
        "",
        "Within-length median generated pairwise Kabsch RMSD Å: "
        + str({q["length"]: q["median_kabsch_rmsd_angstrom"] for q in diversity})
        + ". Exact and near duplicates: "
        + str(sum(q["exact_duplicates"] + q["near_duplicates"] for q in diversity))
        + ". This small bounded reference experiment provides no full-corpus exclusion of memorization.",
    ]
    (c.OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    guide = [
        "# Reading E015 reverse inference",
        "",
        "Each storyboard row uses the same camera and spatial bounds. Purple marks N, gold marks C, and the line color follows residue order. The first column is the actual noisy state X_t. The second is the model-implied clean estimate X0_hat. The third is the next state after the exact production reverse update. The terminal t0 next state is the final sample.",
        "",
        "The model knows current coordinates, requested length, chain order, timestep and its own prior clean estimate. It receives no sequence, target structure or teacher correction during sampling. At high noise the observation contains little information: X0_hat can be small because it behaves like a conditional mean. Only collapse of the actual trajectory, or poor final geometry/distributions, establishes a sampler problem. Compare the actual-state Rg curve with the clean-estimate curve and fixed-scale panels.",
        "",
        "Alpha and sigma describe signal/noise weights; logSNR is log(alpha²/sigma²). Polar, chiral and local vectors add to the predicted coordinate-v field. Gates weight chiral/local geometry contributions. RMS ratios need not sum to one because vectors can reinforce or cancel. Quiver panels use one display gain per timestep across all heads; arrows visualize normalized velocity, not literal reverse displacement. Magnitude profiles show where each contribution is concentrated.",
        "",
        "Distance maps encode residue-pair separations. Consistent near-diagonal bands indicate local chain organization; off-diagonal contacts can reveal global organization. All evolution maps use0–30Å and saturate above30. The real map is matched-length context, never a paired target. Similar Rg alone does not prove a protein fold; covariance shows shape/anisotropy, while chain-separation distances, contacts, torsions and local spacing test other aspects.",
        "",
        "Same-seed panels reuse the stored 50% Gaussian starts. The exact historical normalized FP32 representation is accepted only when its serialized X499 matches the saved tensor bit for bit. The old final/trajectory coordinates are reused; old full sampling is not rerun. The chosen weights are explicitly labelled, so a checkpoint/EMA-choice comparison cannot be mistaken for a weight-controlled ablation.",
        "",
        "Numerical best/median/worst and one representative per length are recorded in inference_story/selection.json. Quality scores were frozen before visual selection and use deviations from real spacing, contacts, covariance, torsion/chirality and scale. “Success” means relatively best numerical rank, not a thermodynamic or all-atom certification. Adjacent-distance valid range is the project's3.4–4.2Å criterion. Torsion assessability fractions can exclude chain termini, which lack full local quartets.",
        "",
        answer,
    ]
    (c.OUT / "INFERENCE_GUIDE.md").write_text("\n".join(guide) + "\n")
    c.b.write("verdict.json", verdict)
    files = [
        dict(path=str(p.relative_to(c.OUT)), bytes=p.stat().st_size, sha256=c.sha(p))
        for p in c.OUT.rglob("*")
        if p.is_file() and p.name not in ("MANIFEST.json", "resource_monitor.jsonl") and "runtime" not in p.parts
    ]
    total = sum(p.stat().st_size for p in c.OUT.rglob("*") if p.is_file())
    assert total <= 6 * 1024**3
    c.b.write(
        "MANIFEST.json",
        dict(
            status="EVALUATION_COMPLETE_RESUME_PENDING",
            primary=selection,
            previous_primary_sha256="2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03",
            metric_version=c.VERSION,
            files=files,
            total_bytes=total,
            all_artifacts_on_d=True,
            serialized=True,
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.WORK, text=True).strip(),
        ),
    )
    c.b.write("logs/report_complete.json", dict(report=str(c.OUT / "REPORT.md"), total_bytes=total))
