"""Compact exposure-matched and later-context comparisons; never resumes training."""

import os, pathlib, json, math, time, ast

C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
E = pathlib.Path(
    "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/evaluations/latest_complete_u10382_4951d838_v1"
)
X = E.parent.parent / "external_designability_loop_v1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import e007_epoch1_mission as mission

mission.helper().host(True)
import e007_comparison_metrics as b

np = b.np
read = lambda p: json.loads(p.read_text())


def table(p):
    return b.pq.read_table(p, use_threads=False).to_pylist() if p.exists() else []


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0, c - w), min(1, c + w)]


r7 = table(C / "e007_reconstruction.parquet")
r15 = table(E / "absolute_reconstruction_metrics.parquet")
cp7 = read(C / "E007_EPOCH1_CHECKPOINT.json")
pending = read(C / "COMPUTATION_STATUS.json")["pending"]
rng = np.random.default_rng(151502)
metrics = [
    "v_mse",
    "aligned_rmsd_angstrom",
    "local_rmse_angstrom",
    "torsion_angular_mae_radians",
    "triple_inversion_fraction",
    "clash_fraction",
    "rg_discrepancy",
]
comparison = []
for label in ["epoch_1_fraction_1.00", "latest_u10382"]:
    for t in b.GRID:
        a = {r["sample_id"]: r for r in r7 if r["timestep"] == t and r["split"] == "heldout"}
        v = {
            r["sample_id"]: r
            for r in r15
            if r["checkpoint"] == label and r["weights"] == "raw" and r["timestep"] == t and r["split"] == "heldout"
        }
        assert set(a) == set(v), "Mismatched fixed held-out panel"
        for metric in metrics:
            ids = [
                i
                for i in sorted(a)
                if isinstance(a[i].get(metric), (int, float))
                and isinstance(v[i].get(metric), (int, float))
                and math.isfinite(a[i][metric])
                and math.isfinite(v[i][metric])
            ]
            aa = np.array([a[i][metric] for i in ids])
            bb = np.array([v[i][metric] for i in ids])
            draw = np.zeros(5000)
            if not ids:
                continue
            for st in b.STRATA:
                ix = [j for j, i in enumerate(ids) if a[i]["length_stratum"] == st]
                if ix:
                    draw += (bb - aa)[rng.choice(ix, (5000, len(ix)), replace=True)].sum(1) / len(ids)
            comparison.append(
                dict(
                    E015_checkpoint=label,
                    timestep=t,
                    metric=metric,
                    E007_mean=float(aa.mean()),
                    E015_mean=float(bb.mean()),
                    E015_minus_E007=float((bb - aa).mean()),
                    difference_ci95=np.quantile(draw, [0.025, 0.975]).tolist(),
                    paired_proteins=len(ids),
                    bootstrap_draws=5000,
                    scope="Exploratory paired protein bootstrap; no architecture-only causal claim or multiplicity-adjusted significance claim",
                )
            )
b.write("matched_reconstruction_comparison.json", dict(rows=comparison))
g7 = table(C / "generation_metrics.parquet")
gm = table(C / "e015_epoch1_context/generation_metrics.parquet")
gl = [r for r in table(E / "generation_metrics.parquet") if r["primary"]]
design7 = read(C / "E007_EXTERNAL_DESIGNABILITY.json")
design15 = table(X / "designability_summary.parquet")
controls15 = read(X / "controls_summary.json")
for result in design7["rows"]:
    result["D_0_7_wilson95_descriptive"] = wilson(result["strong_count"], result["selected_denominator"])
for result in design15:
    result["D_0_7_wilson95_descriptive"] = wilson(result["strong_count"], result["selected_denominator"])
b.write(
    "designability_uncertainty.json",
    dict(
        E007=design7["rows"],
        E015=design15,
        qualification="Descriptive small-sample Wilson intervals; sequence designs within a backbone are correlated and selected, not random independent biological experiments",
    ),
)
paired = []
for label, gs in [("E015_epoch1_exposure_matched", gm), ("E015_u10382_later_context", gl)]:
    lookup = {(r["requested_length"], r["seed"]): r for r in gs}
    for metric in [
        "adjacent_valid_fraction",
        "adjacent_mean_angstrom",
        "clash_fraction",
        "radius_of_gyration_angstrom",
        "long_range_contact_density",
        "long_range_contact_order",
    ]:
        for n in b.LENGTHS:
            cases = [(r, lookup[(n, r["seed"])]) for r in g7 if r["requested_length"] == n and (n, r["seed"]) in lookup]
            values = [
                (a[metric], z[metric]) for a, z in cases if a.get(metric) is not None and z.get(metric) is not None
            ]
            if not values:
                continue
            aa, bb = np.array(values).T
            draw = (bb - aa)[rng.integers(0, len(aa), (5000, len(aa)))].mean(1)
            paired.append(
                dict(
                    comparison=label,
                    length=n,
                    metric=metric,
                    E007_mean=float(aa.mean()),
                    E015_mean=float(bb.mean()),
                    E015_minus_E007=float((bb - aa).mean()),
                    difference_ci95=np.quantile(draw, [0.025, 0.975]).tolist(),
                    paired_seeds=len(aa),
                    interpretation="Four-seed exploratory panel; no broad significance claim",
                )
            )
b.write("paired_generation_comparison.json", dict(rows=paired))
# Shared camera and spatial limits; all twenty paired final structures, plus numeric galleries.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if gm:
    mindex = {
        (r["requested_length"], r["seed"]): r for r in table(C / "e015_epoch1_context/generation_manifest.parquet")
    }
    for r in g7:
        n, seed = r["requested_length"], r["seed"]
        q = mindex[(n, seed)]
        folder = C / "e007_samples" / r["sample_key"]
        meta = read(folder / "COMPLETE.json")
        assert meta["initial_gaussian_sha256"] == q["initial_gaussian_sha256"]
        a = np.load(folder / "final_coordinates.npz")["coordinates"]
        z = np.load(pathlib.Path(q["path"]) / "final_coordinates.npz")["coordinates_angstrom"]
        lim = max(1, np.abs(a).max(), np.abs(z).max())
        fig = plt.figure(figsize=(10, 5))
        for j, (xyz, title) in enumerate([(a, "E007 epoch1"), (z, "E015 epoch1")], 1):
            ax = fig.add_subplot(1, 2, j, projection="3d")
            ax.plot(*xyz.T, lw=1)
            ax.scatter(*xyz.T, c=np.arange(len(xyz)), cmap="viridis", s=5)
            ax.scatter(*xyz[[0, -1]].T, c=["blue", "red"], s=30)
            ax.text(*xyz[0], "N")
            ax.text(*xyz[-1], "C")
            ax.view_init(20, 35)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), title=title + "; Cα only")
        fig.suptitle(f"N={n}; seed={seed}; identical Gaussian and fixed scale")
        fig.savefig(C / "comparison_galleries" / (r["sample_key"] + ".png"), dpi=130)
        plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
for ax, metric in zip(axs, ["adjacent_mean_angstrom", "adjacent_valid_fraction", "clash_fraction"]):
    for label, gs in [("E007 epoch1", g7), ("E015 epoch1", gm), ("E015 u10382; unmatched", gl)]:
        if gs:
            ax.plot(
                b.LENGTHS,
                [np.mean([r[metric] for r in gs if r["requested_length"] == n]) for n in b.LENGTHS],
                marker="o",
                label=label,
            )
    if metric == "adjacent_mean_angstrom":
        ax.axhline(3.8, color="green", ls="--", label="~3.8Å physical context")
    ax.set(xlabel="Requested length", ylabel=metric)
    ax.legend(fontsize=7)
fig.savefig(C / "comparison_plots/generation_geometry.png", dpi=140)
plt.close(fig)
low = [r for r in r7 if r["split"] == "heldout" and r["timestep"] == 5]
finite = all(r["finite"] for r in r7)
adj = float(np.mean([r["adjacent_valid_fraction"] for r in g7]))
rate = float(np.mean([r["finite_fraction"] == 1 for r in g7]))
recon = (
    "promising"
    if finite
    and np.mean([r["aligned_rmsd_angstrom"] for r in low]) < 1
    and np.mean([r["local_rmse_angstrom"] for r in low]) < 0.5
    else "weak"
    if finite
    else "unstable"
)
generation = "collapsed" if rate < 1 else "bounded but weak" if adj < 0.9 else "improving but immature"
matched_design = (
    read(C / "E015_MATCHED_EXTERNAL_DESIGNABILITY.json")
    if (C / "E015_MATCHED_EXTERNAL_DESIGNABILITY.json").exists()
    else None
)
matched_pairs = []
if matched_design:
    for n in b.LENGTHS:
        a = next(r for r in design7["rows"] if r["source_category"] == "latest_best" and r["length"] == n)
        z = next(r for r in matched_design["rows"] if r["source_category"] == "latest_best" and r["length"] == n)
        matched_pairs.append(
            dict(
                length=n,
                E007=a,
                E015=z,
                D07_difference_E015_minus_E007=z["D_0_7"] - a["D_0_7"]
                if z["D_0_7"] is not None and a["D_0_7"] is not None
                else None,
                scope="Best-of-four independently selected by frozen geometry score; length paired, winner seeds may differ",
            )
        )
b.write(
    "matched_designability_comparison.json",
    dict(
        rows=matched_pairs,
        qualification="Five length cases; no broad significance claim",
        E015_matched_available=matched_design is not None,
    ),
)
d15 = read(X / "verdict.json").get("designability", "INCONCLUSIVE")
d7 = design7["verdict"]
practical = (
    "generate-and-filter justified"
    if any(d in ["STRONG_EVIDENCE", "PROMISING_BEST_TAIL"] for d in [d7, d15])
    else "further training needed"
    if all(d not in ["INCONCLUSIVE"] for d in [d7, d15])
    else "inconclusive"
)
verdict = dict(
    E007_reconstruction=recon,
    E007_unconditional_generation=generation,
    E007_designability=d7,
    E015_designability=d15,
    matched_epoch1_comparison=(
        "mixed"
        if matched_design and any(p["D07_difference_E015_minus_E007"] is not None for p in matched_pairs)
        else "inconclusive"
    ),
    practical_de_novo_design=practical,
    qualification="Exposure-matched reconstruction/generation and, where the positive-control gate permits, designability are separately reported from the314,256-exposure E015 context. Equal exposure does not isolate architecture. Generation maturity labels are descriptive, not a paired earlier-E007 improvement claim.",
    pending=pending,
)
b.write("verdict.json", verdict)


def md(title, label):
    rows = [r for r in comparison if r["E015_checkpoint"] == label]
    text = (
        f"# {title}\n\nE007:230,132 exposures;7,586,505 parameters. E015 "
        + (
            "epoch1:230,132 exposures"
            if label == "epoch_1_fraction_1.00"
            else "u10382:314,256 exposures; not exposure-matched"
        )
        + ";33,128,139 parameters.\n\n"
    )
    text += "Equal exposure does not isolate architecture: physical batches, optimizer update count, BF16 training, self-conditioning, auxiliary objectives, PCGrad, rollout states and EMA differ. Comparison inference uses native500-step coordinate-v schedules, common12.22820347644835Å scale and identical saved Gaussian starts.\n\n|t|Metric|E007|E015|E015−E007|95% paired interval|\n|---:|---|---:|---:|---:|---|\n"
    for r in rows:
        text += f"|{r['timestep']}|{r['metric']}|{r['E007_mean']:.5g}|{r['E015_mean']:.5g}|{r['E015_minus_E007']:.5g}|{r['difference_ci95']}|\n"
    text += "\nSmall TRAIN–held-out gaps do not establish useful generalization when absolute errors are poor. High-noise direct-X0 shrinkage alone is not sampler collapse: inspect actual-state trajectories and full unconditional samples. No RMSD to arbitrary real proteins is used as unconditional generation quality.\n\nDetailed geometry, paired seed intervals, descriptor floors and designability denominators are in the JSON/parquet files. Twenty generated structures and five best-of-four candidates form a bounded panel; best-tail external yield does not estimate the entire generated distribution. Untested backbones are unknown, not successes or failures.\n"
    return text


(C / "MATCHED_EPOCH1_COMPARISON.md").write_text(md("Exposure-matched epoch-one comparison", "epoch_1_fraction_1.00"))
(C / "LATER_E015_CONTEXT_COMPARISON.md").write_text(md("Later E015 practical-capability context", "latest_u10382"))
text = (
    "# Computational comparison verdict\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in verdict.items() if k not in ["pending"])
    + "\n\n"
)
text += (
    "Positive controls: E007 "
    + design7["controls"]["status"]
    + "; E015 "
    + controls15["status"]
    + ". Failed controls make model designability inconclusive, never evidence of non-designability.\n\n"
)
text += "ProteinMPNN scores and AlphaFold confidence are supporting evidence; neither replaces target TM-score/coverage/RMSD agreement. A low PDB similarity without successful refolding is not novelty. These are computational results, not experimental stability, solubility or functionality.\n\n"
text += "Reconstruction: E007_RECONSTRUCTION.json. Generation: E007_GENERATION.json and generation_distribution_metrics.json. Designability: E007_EXTERNAL_DESIGNABILITY.json, e007_inverse_folding/designability_summary.parquet, and the referenced E015 external root. Nearest confirmed PDB hits and novelty tables remain separate. Galleries:e007_galleries/, comparison_galleries/, and e007_inverse_folding/candidates/.\n"
if pending:
    text += "\nPending external/resource-limited stages:" + json.dumps(pending) + "\n"
(C / "FINAL_VERDICT.md").write_text(text)
(C / "RECOVERY_REPORT.md").write_text(
    (C / "RECOVERY_REPORT.md").read_text()
    + "\nEpoch1 named checkpoint and exact epoch2 next operation verified in E007_EPOCH1_CHECKPOINT.json. Neither training job was restarted after evaluation.\n"
)
files = {
    str(p.relative_to(C)): b.sha(p)
    for p in C.iterdir()
    if p.is_file() and p.name not in ["resource_monitor.jsonl", "progress.json", "MANIFEST.json"]
}
b.write(
    "MANIFEST.json",
    dict(
        status="PARTIAL_EXTERNAL_LIMITATIONS" if pending else "COMPLETE_BOUNDED_COMPARISON",
        checkpoint=cp7,
        files=files,
        E015_references=dict(
            latest_evaluation=str(E),
            external=str(X),
            exposures=314256,
            sha256="4951d8387ebd2bce5273be9580dd26989f067e770fd124bc6bb1c12f5da8d32f",
        ),
        matched_E015_generation=read(C / "e015_epoch1_context/MANIFEST.json") if gm else None,
        no_automatic_training_resume=True,
        all_artifacts_on_d=True,
        claims="Computational bounded-panel assessment; not experimental validation. Matched designability availability is recorded explicitly in matched_designability_comparison.json.",
    ),
)
b.write(
    "progress.json",
    dict(
        stage="REPORT_COMPLETE_WITH_LIMITATIONS" if pending else "REPORT_COMPLETE",
        pending=pending,
        E007_epoch1_paused=True,
        E015_training_paused=True,
        automatic_epoch2=False,
    ),
)
print(text)
