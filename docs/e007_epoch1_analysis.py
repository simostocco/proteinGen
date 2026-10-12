"""CPU-only matched panels, frozen ranking, bootstrap, visuals and external panel."""

import os, pathlib, json, sys, ast, types, math, hashlib

C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
E = pathlib.Path(
    "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/evaluations/latest_complete_u10382_4951d838_v1"
)
EXT = E.parent.parent / "external_designability_loop_v1"
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import e007_epoch1_mission as mission

mission.helper().host(True)
import e007_comparison_metrics as b

np = b.np
cache = b.CoordinateCache(pathlib.Path(b.cfg["population_root"]), verify_payload=False)
core = pathlib.Path(__file__).with_name("e015_latest_core.py").read_text()
node = next(n for n in ast.parse(core).body if isinstance(n, ast.FunctionDef) and n.name == "score_vector")
scope = {"np": np}
exec(compile(ast.Module(body=[node], type_ignores=[]), "frozen_quality_function", "exec"), scope)
score_vector = scope["score_vector"]
floor = np.array([0.1, 0.1, 0.02, 0.001, 1, 0.03, 0.03, 0.03, 0.005, 0.02, 1, 1, 1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])


def rows(name, root=C):
    return b.pq.read_table(root / name, use_threads=False).to_pylist()


def refs(n):
    dev = rows("population/development.parquet", pathlib.Path(b.cfg["population_root"]))
    rr = sorted(dev, key=lambda r: (abs(r["length"] - n), b.key("e015-generation-reference-v1", r)))[:8]
    return [dict(b.intrinsic(cache.get(r["sample_id"])), sample_id=r["sample_id"], length=r["length"]) for r in rr]


def analysis():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = rows("e007_reconstruction.parquet")
    old = rows("absolute_reconstruction_metrics.parquet", E)
    metrics = [
        "v_mse",
        "aligned_rmsd_angstrom",
        "local_rmse_angstrom",
        "torsion_angular_mae_radians",
        "triple_inversion_fraction",
        "clash_fraction",
        "radius_ratio",
    ]
    summaries = []
    rng = np.random.default_rng(15015)
    for t in b.GRID:
        for metric in metrics:
            sets = {
                split: [r for r in rec if r["timestep"] == t and r["split"] == split] for split in ["train", "heldout"]
            }
            by = {split: {r["pair_id"]: r for r in rs} for split, rs in sets.items()}
            pairs = [
                key
                for key in sorted(set(by["train"]) & set(by["heldout"]))
                if all(
                    isinstance(by[split][key].get(metric), (float, int)) and math.isfinite(by[split][key][metric])
                    for split in ["train", "heldout"]
                )
            ]
            if not pairs:
                summaries.append(
                    dict(
                        timestep=t,
                        metric=metric,
                        assessable_pairs=0,
                        train_mean=None,
                        heldout_mean=None,
                        gap=None,
                        gap_ci95=None,
                        reason="No jointly assessable paired proteins",
                    )
                )
                continue
            a = np.array([by["train"][key][metric] for key in pairs])
            h = np.array([by["heldout"][key][metric] for key in pairs])
            ad = np.zeros(5000)
            hd = np.zeros(5000)
            for st in b.STRATA:
                ix = [j for j, key in enumerate(pairs) if by["train"][key]["length_stratum"] == st]
                if ix:
                    draw = rng.choice(ix, (5000, len(ix)), replace=True)
                    ad += a[draw].sum(1) / len(pairs)
                    hd += h[draw].sum(1) / len(pairs)
            gap = hd - ad
            summaries.append(
                dict(
                    timestep=t,
                    metric=metric,
                    train_mean=float(a.mean()),
                    train_median=float(np.median(a)),
                    heldout_mean=float(h.mean()),
                    heldout_median=float(np.median(h)),
                    gap=float(h.mean() - a.mean()),
                    ratio=float(h.mean() / a.mean()) if a.mean() != 0 else None,
                    gap_ci95=np.quantile(gap, [0.025, 0.975]).tolist(),
                    draws=5000,
                    aggregation="stratified paired equal-protein bootstrap, shared noise family",
                    assessable_pairs=len(pairs),
                )
            )
    b.write(
        "E007_RECONSTRUCTION.json",
        dict(
            status="COMPLETE",
            conditions=len(rec),
            rows=summaries,
            comparison_panel_sha256=b.sha(E / "panel_definitions.json"),
            noise="Identical CPU Gaussian family; native shared coordinate scale and500step schedule",
            EMA="NOT_APPLICABLE",
            chiral_heads="NOT_APPLICABLE",
        ),
    )
    gen = rows("generation_metrics.parquet")
    rank = []
    distribution = []
    diversity = []
    novelty = []
    from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import ks, w1

    serial = pathlib.Path(__file__).with_name("e015_serial_evaluation.py").read_text()
    pn = next(n for n in ast.parse(serial).body if isinstance(n, ast.FunctionDef) and n.name == "population_metrics")
    pscope = dict(np=np, ks=ks, w1=w1)
    exec(compile(ast.Module(body=[pn], type_ignores=[]), "frozen_descriptor_bootstrap", "exec"), pscope)
    population_metrics = pscope["population_metrics"]
    for n in b.LENGTHS:
        rr = refs(n)
        real = np.array([score_vector(r) for r in rr])
        center = np.median(real, 0)
        scale = np.maximum((np.quantile(real, 0.75, axis=0) - np.quantile(real, 0.25, axis=0)) / 1.349, floor)
        gs = [r for r in gen if r["requested_length"] == n]
        for r in gs:
            rank.append(
                dict(
                    sample_key=r["sample_key"],
                    length=n,
                    score=float(np.sqrt(np.mean(((score_vector(r) - center) / scale) ** 2))),
                    metrics=r,
                    reference_ids=[x["sample_id"] for x in rr],
                )
            )
        raw_a = np.array([r["descriptor"] for r in rr[::2]])
        raw_b = np.array([r["descriptor"] for r in rr[1::2]])
        standard = b.ReferenceStandardizer(raw_a, raw_b)
        a = standard.transform(raw_a)
        bb = standard.transform(raw_b)
        h = b.bandwidth(a, bb)
        p = b.projections(len(standard.columns))
        spectrum = [
            i
            for i, c in enumerate(standard.columns)
            if b.NAMES[c] in ["eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3"]
        ]
        raw = np.array([r["descriptor"] for r in gs])
        gc = b.Comparison(standard.transform(raw), bb, raw, raw_b, h, p, spectrum)
        fc = b.Comparison(a, bb, raw_a, raw_b, h, p, spectrum)
        gm = gc.metrics()
        fm = fc.metrics()
        boots = {k: [] for k in gm}
        floorboots = {k: [] for k in fm}
        brng = np.random.default_rng(15015)
        for _ in range(5000):
            ig = brng.integers(0, 4, 4)
            ia = brng.integers(0, 4, 4)
            ib = brng.integers(0, 4, 4)
            g = population_metrics(gc, ig, ib)
            f = population_metrics(fc, ia, ib)
            for k in gm:
                boots[k].append(g[k])
                floorboots[k].append(f[k])
        coord = {
            r["sample_key"]: np.load(C / "e007_samples" / r["sample_key"] / "final_coordinates.npz")["coordinates"]
            for r in gs
        }
        pairs = []
        for i, r in enumerate(gs):
            for q in gs[i + 1 :]:
                x, z = coord[r["sample_key"]], coord[q["sample_key"]]
                dm = np.linalg.norm(x[:, None] - x[None, :], axis=-1) - np.linalg.norm(z[:, None] - z[None, :], axis=-1)
                pairs.append(
                    dict(
                        a=r["sample_key"],
                        b=q["sample_key"],
                        kabsch_rmsd_angstrom=b.aligned_rmsd(x, z),
                        distance_matrix_rmse_angstrom=float(np.sqrt(np.mean(dm**2))),
                        descriptor_distance=float(
                            b.distances(
                                standard.transform(np.array([r["descriptor"]])),
                                standard.transform(np.array([q["descriptor"]])),
                            )[0, 0]
                        ),
                    )
                )
        diversity.append(
            dict(
                length=n,
                pairs=pairs,
                exact_duplicates=sum(x["kabsch_rmsd_angstrom"] < 1e-6 for x in pairs),
                median_pairwise_kabsch_rmsd_angstrom=float(np.median([x["kabsch_rmsd_angstrom"] for x in pairs])),
            )
        )
        train = rows("population/train.parquet", pathlib.Path(b.cfg["population_root"]))
        tr = sorted(train, key=lambda r: (abs(r["length"] - n), b.key("e015-generation-novelty-v1", r)))[:20]
        trstats = [dict(b.intrinsic(cache.get(q["sample_id"])), sample_id=q["sample_id"]) for q in tr]
        for r in gs:
            nearest = {}
            for split, ref in [("train", trstats), ("heldout", rr)]:
                ds = b.distances(
                    standard.transform(np.array([r["descriptor"]])),
                    standard.transform(np.array([q["descriptor"] for q in ref])),
                )[0]
                j = int(np.argmin(ds))
                nearest[split] = dict(
                    identity=ref[j]["sample_id"],
                    descriptor_distance=float(ds[j]),
                    reference_count=len(ref),
                    reference_population_root=b.cfg["population_root"],
                )
            novelty.append(
                dict(
                    sample=r["sample_key"],
                    nearest=nearest,
                    scope="Bounded sampled novelty; not a fullTRAINindex or proof against memorization",
                )
            )

        distribution.append(
            dict(
                length=n,
                generated_vs_real=gm,
                real_real_floor=fm,
                excess={k: gm[k] - fm[k] for k in gm},
                excess_ratio={k: gm[k] / fm[k] if fm[k] else None for k in gm},
                reference_ids=[x["sample_id"] for x in rr],
                bootstrap95={
                    k: dict(
                        generated=np.quantile(boots[k], [0.025, 0.975]).tolist(),
                        real_floor=np.quantile(floorboots[k], [0.025, 0.975]).tolist(),
                        excess=np.quantile(np.asarray(boots[k]) - np.asarray(floorboots[k]), [0.025, 0.975]).tolist(),
                    )
                    for k in gm
                },
                draws=5000,
            )
        )
    b.write("diversity_novelty_metrics.json", dict(diversity=diversity, novelty=novelty))
    rank.sort(key=lambda x: (x["score"], x["sample_key"]))
    b.write(
        "quality_ranking.json",
        dict(
            rows=rank,
            definition="Exact E01520-scalar matched-real median/IQR quality score and resolution floors; no visual selection",
        ),
    )
    b.write(
        "generation_distribution_metrics.json",
        dict(rows=distribution, scope="Bounded descriptor comparison, not equality of full coordinate distributions"),
    )
    # Fixed camera/scale storyboards and sample cards; selection solely frozen numeric score.
    selected = [rank[0], rank[len(rank) // 2], rank[-1]]
    for i, r in enumerate(rank):
        folder = C / "e007_samples" / r["sample_key"]
        xyz = np.load(folder / "final_coordinates.npz")["coordinates"]
        trajectory = np.load(folder / "sparse_trajectory.npz")
        direct = np.load(folder / "direct_x0.npz")
        fig = plt.figure(figsize=(12, 7), layout="constrained")
        lim = max(float(np.abs(xyz).max()), 1)
        for j, (el, az) in enumerate([(20, 35), (20, 125)], 1):
            ax = fig.add_subplot(2, 3, j, projection="3d")
            ax.plot(*xyz.T, lw=1)
            ax.scatter(*xyz.T, c=np.arange(len(xyz)), cmap="viridis", s=3)
            ax.scatter(*xyz[[0, -1]].T, c=["blue", "red"], s=25)
            ax.view_init(el, az)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), title="Cα only; no sequence")
        ax = fig.add_subplot(2, 3, 3)
        ax.imshow(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1), vmin=0, vmax=30)
        ax.set_title("Distances Å; fixed0–30")
        ax = fig.add_subplot(2, 3, 4)
        for k in [1, 2, 3]:
            ax.plot(np.linalg.norm(xyz[k:] - xyz[:-k], axis=-1), label=f"i+{k}")
        ax.legend()
        ax.set_ylabel("Distance Å")
        ax = fig.add_subplot(2, 3, 5)
        ts = [int(k[1:]) for k in trajectory.files]
        ax.plot(ts, [b.intrinsic(trajectory[f"t{t}"])["radius_of_gyration_angstrom"] for t in ts])
        ax.invert_xaxis()
        ax.set_ylabel("Rg Å")
        ax = fig.add_subplot(2, 3, 6)
        ax.axis("off")
        ax.text(
            0,
            1,
            f"Rank {i + 1}/20; frozen score {r['score']:.3f}\nAdjacent {r['metrics']['adjacent_mean_angstrom']:.3f} Å\nValid {r['metrics']['adjacent_valid_fraction']:.1%}\nClash {r['metrics']['clash_fraction']:.4f}\nComputational geometry only",
            va="top",
        )
        fig.savefig(folder / "quality_card.png", dpi=130)
        plt.close(fig)
        if r not in selected:
            continue
        category = ["best", "median", "failure"][selected.index(r)]
        dest = C / "e007_galleries" / category
        dest.mkdir(exist_ok=True)
        states = [499, 450, 400, 350, 250, 150, 50, 0]
        lim = max(np.abs(trajectory[f"t{t}"]).max() for t in states)
        lim = max(lim, max(np.abs(direct[f"t{t}"]).max() for t in states))
        fig = plt.figure(figsize=(9, 24))
        for j, t in enumerate(states):
            for col, collection in enumerate([trajectory, direct]):
                x = collection[f"t{t}"]
                ax = fig.add_subplot(8, 2, j * 2 + col + 1, projection="3d")
                ax.plot(*x.T, lw=0.7)
                ax.scatter(*x.T, c=np.arange(len(x)), cmap="viridis", s=2)
                ax.scatter(*x[[0, -1]].T, c=["blue", "red"], s=20)
                ax.view_init(20, 35)
                ax.set(
                    xlim=(-lim, lim),
                    ylim=(-lim, lim),
                    zlim=(-lim, lim),
                    title=f"t={t}; {'X_t' if col == 0 else 'direct X0'}; Rg={b.intrinsic(x)['radius_of_gyration_angstrom']:.2f}Å",
                )
        fig.savefig(dest / "fixed_scale_storyboard.png", dpi=100)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axs, ["v_mse", "aligned_rmsd_angstrom", "local_rmse_angstrom"]):
        ax.plot(
            b.GRID,
            [
                next(q["heldout_mean"] for q in summaries if q["timestep"] == t and q["metric"] == metric)
                for t in b.GRID
            ],
            label="E007epoch1",
        )
        for cp in ["epoch_1_fraction_1.00", "latest_u10382"]:
            rs = [r for r in old if r["checkpoint"] == cp and r["weights"] == "raw" and r["split"] == "heldout"]
            if rs:
                ax.plot(
                    b.GRID, [np.mean([r[metric] for r in rs if r["timestep"] == t]) for t in b.GRID], label="E015 " + cp
                )
        ax.set(xlabel="timestep", ylabel=metric)
        ax.legend()
    fig.savefig(C / "comparison_plots/reconstruction.png", dpi=140)
    plt.close(fig)
    b.write(
        "E007_GENERATION.json",
        dict(
            status="COMPLETE",
            samples=20,
            finite_rate=float(np.mean([r["finite_fraction"] == 1 for r in gen])),
            distribution=distribution,
            rank_path=str(C / "quality_ranking.json"),
            external_designability="PENDING; finite geometry is not designability",
        ),
    )


def external_panel():
    import shutil

    out = C / "e007_inverse_folding"
    out.mkdir(exist_ok=True)
    for name in [
        "sanitized_backbones",
        "logs",
        "sequences",
        "cache",
        "proteinmpnn",
        "colabfold",
        "alignments",
        "foldseek",
        "candidates",
    ]:
        (out / name).mkdir(exist_ok=True)
    if not (out / "tools").exists():
        (out / "tools").symlink_to(EXT / "tools", target_is_directory=True)
    shutil.copyfile(EXT / "toolchain_manifest.json", out / "toolchain_manifest.json")
    ranking = json.loads((C / "quality_ranking.json").read_text())["rows"]
    base = json.loads((EXT / "backbone_panel.json").read_text())
    cp = json.loads((C / "E007_EPOCH1_CHECKPOINT.json").read_text())
    panel = []
    picks = [
        (min([r for r in ranking if r["length"] == n], key=lambda r: r["score"]), "latest_best") for n in b.LENGTHS
    ] + [(ranking[len(ranking) // 2], "latest_median"), (ranking[-1], "latest_failure")]
    for r, category in picks:
        bid = "E007_" + category + "_" + r["sample_key"]
        origin = C / "e007_samples" / r["sample_key"] / "final_ca.pdb"
        target = out / "sanitized_backbones" / (bid + ".pdb")
        shutil.copyfile(origin, target)
        panel.append(
            dict(
                backbone_id=bid,
                source_category=category,
                length=r["length"],
                seed=r["metrics"]["seed"],
                checkpoint_sha256=cp["sha256"],
                original_pdb=str(origin),
                sanitized_pdb=str(target),
                backbone_sha256=b.sha(target),
                numerical_score=r["score"],
                original_metrics=r["metrics"],
            )
        )
    panel += [r for r in base["backbones"] if r["source_category"] == "real_heldout"]
    for r in panel:
        assert b.sha(r["sanitized_pdb"]) == r["backbone_sha256"]
    mission.write(
        out / "backbone_panel.json",
        dict(
            model="E007_epoch1",
            backbones=panel,
            positive_controls_reused_hash_matched=True,
            protocol_source=str(EXT / "assessment_protocol.json"),
            protocol_sha256=b.sha(EXT / "assessment_protocol.json"),
        ),
    )
    controls = {
        k: v
        for k, v in json.loads((EXT / "colabfold/jobs.json").read_text()).items()
        if v["sequence"]["source_category"] == "real_heldout"
    }
    for v in controls.values():
        assert v["status"] == "COMPLETE" and all(b.sha(p) == h for p, h in v["files"].items())
    existing = json.loads((out / "colabfold/jobs.json").read_text()) if (out / "colabfold/jobs.json").exists() else {}
    existing.update(controls)
    mission.write(out / "colabfold/jobs.json", existing)
    print("E007 external panel frozen", len(panel), "backbones; existing controls reused")


def matched_external_panel():
    import shutil

    root = C / "e015_epoch1_context"
    out = C / "e015_epoch1_inverse_folding"
    out.mkdir(exist_ok=True)
    for name in [
        "sanitized_backbones",
        "logs",
        "sequences",
        "cache",
        "proteinmpnn",
        "colabfold",
        "alignments",
        "foldseek",
        "candidates",
    ]:
        (out / name).mkdir(exist_ok=True)
    if not (out / "tools").exists():
        (out / "tools").symlink_to(EXT / "tools", target_is_directory=True)
    shutil.copyfile(EXT / "toolchain_manifest.json", out / "toolchain_manifest.json")
    ranking = []
    for row in rows("generation_metrics.parquet", root):
        rr = refs(row["requested_length"])
        real = np.array([score_vector(r) for r in rr])
        center = np.median(real, 0)
        scale = np.maximum((np.quantile(real, 0.75, axis=0) - np.quantile(real, 0.25, axis=0)) / 1.349, floor)
        ranking.append(
            dict(
                sample_key=row["sample_key"],
                length=row["requested_length"],
                score=float(np.sqrt(np.mean(((score_vector(row) - center) / scale) ** 2))),
                metrics=row,
            )
        )
    ranking.sort(key=lambda r: (r["score"], r["sample_key"]))
    b.write(
        "e015_epoch1_quality_ranking.json",
        dict(
            rows=ranking,
            definition="Identical frozen E01520-scalar reference ranking and resolution floors; selected before external predictions",
        ),
    )
    cp = json.loads((root / "MANIFEST.json").read_text())
    base = json.loads((EXT / "backbone_panel.json").read_text())
    panel = []
    picks = [
        (min([r for r in ranking if r["length"] == n], key=lambda r: r["score"]), "latest_best") for n in b.LENGTHS
    ] + [(ranking[len(ranking) // 2], "latest_median"), (ranking[-1], "latest_failure")]
    for r, category in picks:
        bid = "E015_epoch1_" + category + "_" + r["sample_key"]
        origin = root / "samples" / r["sample_key"] / "final_ca.pdb"
        target = out / "sanitized_backbones" / (bid + ".pdb")
        shutil.copyfile(origin, target)
        panel.append(
            dict(
                backbone_id=bid,
                source_category=category,
                length=r["length"],
                seed=r["metrics"]["seed"],
                checkpoint_sha256=cp["sha256"],
                original_pdb=str(origin),
                sanitized_pdb=str(target),
                backbone_sha256=b.sha(target),
                numerical_score=r["score"],
                original_metrics=r["metrics"],
            )
        )
    panel += [r for r in base["backbones"] if r["source_category"] == "real_heldout"]
    for r in panel:
        assert b.sha(r["sanitized_pdb"]) == r["backbone_sha256"]
    mission.write(
        out / "backbone_panel.json",
        dict(
            model="E015_epoch1",
            model_checkpoint=cp,
            backbones=panel,
            positive_controls_reused_hash_matched=True,
            protocol_sha256=b.sha(EXT / "assessment_protocol.json"),
        ),
    )
    controls = {
        k: v
        for k, v in json.loads((EXT / "colabfold/jobs.json").read_text()).items()
        if v["sequence"]["source_category"] == "real_heldout"
    }
    for v in controls.values():
        assert v["status"] == "COMPLETE" and all(b.sha(p) == h for p, h in v["files"].items())
    existing = json.loads((out / "colabfold/jobs.json").read_text()) if (out / "colabfold/jobs.json").exists() else {}
    existing.update(controls)
    mission.write(out / "colabfold/jobs.json", existing)


if __name__ == "__main__":
    {"analysis": analysis, "external-panel": external_panel, "matched-external-panel": matched_external_panel}[
        sys.argv[1]
    ]()
