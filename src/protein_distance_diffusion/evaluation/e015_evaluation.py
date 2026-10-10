"""Frozen-panel E015 diagnostics. Prefixes stop at350; prospective stays shut."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import aggregate, coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    HIGH,
    MARGINAL_GRID,
    STRATA,
    gaussian_seed,
    schedule_values,
)
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    NAMES,
    Comparison,
    ReferenceStandardizer,
    bandwidth,
    compare_populations,
    descriptor,
    projections,
)
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import (
    paired_state,
    select_rollout_subset,
    vector_metrics,
)
from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import canonical_hash, noise_seed
from protein_distance_diffusion.training.e015_runtime import e007_active, status
from protein_distance_diffusion.training.e015_self_conditioning import make_self_condition

ENDPOINTS = (490, 480, 470, 460, 450, 440, 420, 400, 350)
E014_ROOT = Path("/mnt/d/Simone/proteinGen/.e014_worktree/e014_artifacts/e014/post_reboot_full_evaluation_v1")


def table(runtime, name, rows):
    path = runtime.path(name)
    partial = runtime.path(name + ".partial")
    pq.write_table(pa.Table.from_pylist(rows), partial, compression="zstd")
    if pq.read_metadata(partial).num_rows != len(rows):
        raise ValueError("Parquet verification failed")
    partial.replace(path)


@torch.no_grad()
def prediction(model, diffusion, x, t, n, m):
    previous, available = make_self_condition(model, diffusion, x, t, n, m, model.self_conditioning_enabled)
    return model.with_components(x, t, n, m, previous, available)


def noise_batch(diffusion, x, m, s, family, sample_id):
    generator = torch.Generator(device=x.device).manual_seed(noise_seed(family, sample_id, s))
    return diffusion.make_training_batch(x, m, timesteps=torch.tensor([s], device=x.device), generator=generator)


def descriptor_row(row, t, label, x):
    return dict(
        sample_id=row["sample_id"],
        length=row["length"],
        length_stratum=row["length_stratum"],
        timestep=t,
        population=label,
        descriptor=descriptor(x[0].detach().cpu().numpy()).tolist(),
    )


def distribution_report(rows):
    outputs = []
    for timestep in sorted({r["timestep"] for r in rows}):
        labels = {
            label: sorted(
                [r for r in rows if r["timestep"] == timestep and r["population"] == label],
                key=lambda r: (STRATA.index(r["length_stratum"]), r["sample_id"]),
            )
            for label in ("model", "noop", "ref_a", "ref_b")
        }
        identities = [r["sample_id"] for r in labels["model"]]
        if any([r["sample_id"] for r in values] != identities for values in labels.values()):
            raise ValueError("Unpaired descriptor identities")
        raw = {label: np.array([r["descriptor"] for r in values]) for label, values in labels.items()}
        standardizer = ReferenceStandardizer(raw["ref_a"], raw["ref_b"])
        populations = {label: standardizer.transform(values) for label, values in raw.items()}
        sizes = [sum(r["length_stratum"] == s for r in labels["model"]) for s in STRATA]
        report = compare_populations(populations, raw, standardizer.columns, 15015, sizes, replicates=500)
        report.pop("_model_draws", None)
        report.update(
            timestep=timestep,
            rg_ratio=float(np.mean(raw["model"][:, 0] / raw["ref_b"][:, 0])),
            covariance_trace_ratio=float(np.mean(raw["model"][:, 4] / raw["ref_b"][:, 4])),
            pair_distance_mean_ratio=float(np.mean(raw["model"][:, 8] / raw["ref_b"][:, 8])),
        )
        report["actual_noisy_state_collapse"] = report["rg_ratio"] < 0.8 or report["covariance_trace_ratio"] < 0.8
        outputs.append(report)
    return outputs


def paired_scalar_comparison(new, old, metric, timesteps):
    def proteins(rows):
        grouped = {}
        for row in rows:
            if row["timestep"] not in timesteps:
                continue
            value = row.get(metric)
            if value is not None:
                grouped.setdefault(row["sample_id"], []).append(float(value))
        return {sample: np.mean(values) for sample, values in grouped.items()}

    a, b = proteins(old), proteins(new)
    identities = sorted(set(a) & set(b))
    strata = {r["sample_id"]: r["length_stratum"] for r in new}
    delta = np.array([b[i] - a[i] for i in identities])
    generator = np.random.default_rng(15015)
    groups = [np.flatnonzero(np.array([strata[i] for i in identities]) == s) for s in STRATA]
    draws = np.zeros(5000)
    for group in groups:
        if len(group):
            draws += delta[generator.choice(group, (5000, len(group)), replace=True)].sum(1) / len(delta)
    new_draws = np.zeros(5000)
    for group in groups:
        if len(group):
            values = np.array([b[i] for i in identities])
            new_draws += values[generator.choice(group, (5000, len(group)), replace=True)].sum(1) / len(identities)
    return dict(
        e015_ci95=np.quantile(new_draws, [0.025, 0.975]).tolist(),
        metric=metric,
        timesteps=list(timesteps),
        old=float(np.mean([a[i] for i in identities])),
        e015=float(np.mean([b[i] for i in identities])),
        difference=float(delta.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        paired_proteins=len(identities),
        replicates=5000,
    )


def compare_e014(runtime, base, label, metrics):
    results = {}
    core = (5, 50, 250, 400, 490)
    for comparator in ("epoch_2_fraction_1.00", "recovery_0"):
        path = E014_ROOT / f"{comparator}.paired.parquet"
        if not path.exists():
            raise RuntimeError("Frozen E014 comparator metrics missing")
        old = pq.read_table(path, use_threads=False).to_pylist()
        for row in old:
            row["mean_local_mse_angstrom2"] = sum(row[f"offset_{k}_mse_angstrom2"] for k in (1, 2, 3)) / 3
        family = [("v_mse", core), ("mean_local_mse_angstrom2", core)] + [
            (f"offset_{k}_mse_angstrom2", core) for k in (1, 2, 3)
        ]
        family += [
            ("torsion_angular_mae_radians", (250,)),
            ("torsion_angular_mae_radians", (400,)),
            ("triple_inversion_fraction", (250,)),
            ("triple_inversion_fraction", (400,)),
        ]
        values = [paired_scalar_comparison(metrics, old, metric, times) for metric, times in family]
        results[comparator] = values
    runtime.write(base + "/paired_e014_comparison.json", results)
    return results


@torch.no_grad()
def evaluate_boundary(runtime, engine, folder, label, full=False):
    base = f"{folder}/evaluation/{label}"
    completion = runtime.path(base + "/COMPLETE.json")
    if completion.exists():
        return json.loads(completion.read_text())
    engine.model.eval()
    engine.ema.eval()
    summaries = {}
    primary_metrics = []
    for kind, model in (("ema", engine.ema), ("raw", engine.model)):
        target = runtime.path(base + f"/{kind}.core.parquet")
        if target.exists():
            metrics = pq.read_table(target, use_threads=False).to_pylist()
        else:
            metrics = []
            grid = sorted(set(runtime.cfg["noise_timesteps"]) | (set(HIGH) if full and kind == "ema" else set()))
            for i, row in enumerate(runtime.dev):
                if e007_active(runtime.cfg):
                    raise RuntimeError("E007 became active; stop E015 evaluation")
                x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
                for s in grid:
                    t = torch.tensor([s], device="cuda")
                    batch = noise_batch(engine.diffusion, x, m, s, "e014-paired-validation-v1", row["sample_id"])
                    v, components = prediction(model, engine.diffusion, batch.noisy_coordinates, t, n, m)
                    current, xhat, _ = engine.diffusion.deterministic_reverse_step(batch.noisy_coordinates, t, v, m)
                    oracle = paired_state(engine.diffusion, x, batch.coordinate_noise, t - 1, m)
                    result = extended_metrics(
                        xhat[0],
                        x[0],
                        v[0],
                        batch.coordinate_v_target[0],
                        runtime.cfg["coordinate_scale_angstrom"],
                        components[0],
                    )
                    result.update(vector_metrics(v, batch.coordinate_v_target, m, "v"))
                    result.update(
                        vector_metrics(current - batch.noisy_coordinates, oracle - batch.noisy_coordinates, m, "step")
                    )
                    noop_mse = float((batch.noisy_coordinates - oracle).square()[m].mean())
                    result["mean_local_mse_angstrom2"] = sum(result[f"offset_{k}_mse_angstrom2"] for k in (1, 2, 3)) / 3
                    result.update(
                        timestep=s,
                        sample_id=row["sample_id"],
                        length=row["length"],
                        length_stratum=row["length_stratum"],
                        step_gain=1 - float((current - oracle).square()[m].mean()) / noop_mse,
                        chiral_output_norm=float(components[0]["chiral_effective"].norm()),
                        **schedule_values(engine.diffusion, s),
                    )
                    if not result["finite"]:
                        raise FloatingPointError("Nonfinite E015 development output")
                    metrics.append(result)
                if (i + 1) % 25 == 0:
                    runtime.write("HEARTBEAT.json", status("evaluation", label=label, kind=kind, identities=i + 1))
            table(runtime, base + f"/{kind}.core.parquet", metrics)
        summaries[kind] = aggregate([r for r in metrics if r["timestep"] in runtime.cfg["noise_timesteps"]])
        runtime.write(base + f"/{kind}.summary.json", summaries[kind])
        if kind == "ema":
            primary_metrics = metrics
    comparisons = compare_e014(runtime, base, label, primary_metrics)
    prefixes = []
    if full:
        # Marginal references use independent existing noise families, never the source noise.
        marginal_path = runtime.path(base + "/marginal_descriptors.parquet")
        if not marginal_path.exists():
            marginal = []
            for i, row in enumerate(runtime.dev):
                x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
                for s in MARGINAL_GRID:
                    t = torch.tensor([s], device="cuda")
                    source = noise_batch(
                        engine.diffusion, x, m, s, "e014-marginal-source-v1", row["sample_id"]
                    ).noisy_coordinates
                    v, _ = prediction(engine.ema, engine.diffusion, source, t, n, m)
                    previous, _, _ = engine.diffusion.deterministic_reverse_step(source, t, v, m)
                    populations = {"model": previous, "noop": source}
                    for key, family in [("ref_a", "e014-marginal-ref-a-v1"), ("ref_b", "e014-marginal-ref-b-v1")]:
                        populations[key] = noise_batch(
                            engine.diffusion, x, m, s - 1, family, row["sample_id"]
                        ).noisy_coordinates
                    for key, value in populations.items():
                        marginal.append(descriptor_row(row, s, key, value))
                if (i + 1) % 25 == 0:
                    runtime.write("HEARTBEAT.json", status("marginal", label=label, identities=i + 1))
            table(runtime, base + "/marginal_descriptors.parquet", marginal)
        marginal = pq.read_table(marginal_path, use_threads=False).to_pylist()
        runtime.write(base + "/marginal_metrics.json", distribution_report(marginal))
        subset = select_rollout_subset(runtime.dev, STRATA)
        if canonical_hash([r["sample_id"] for r in subset]) != runtime.cfg["gaussian_subset_hash"]:
            raise ValueError("Gaussian subset changed")
        prefix_path = runtime.path(base + "/gaussian_descriptors.parquet")
        if not prefix_path.exists():
            prefixes = []
            for i, row in enumerate(subset):
                x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
                generator = torch.Generator(device="cuda").manual_seed(gaussian_seed(i, row["length"], noise_seed))
                current = centered_coordinate_noise(torch.zeros_like(x), m, generator=generator)
                initial = current.clone()
                context = None
                for s in range(499, 349, -1):
                    if s in ENDPOINTS:
                        populations = {"model": current, "noop": initial}
                        for key, family in [("ref_a", "e014-marginal-ref-a-v1"), ("ref_b", "e014-marginal-ref-b-v1")]:
                            populations[key] = noise_batch(
                                engine.diffusion, x, m, s, family, row["sample_id"]
                            ).noisy_coordinates
                        for key, value in populations.items():
                            prefixes.append(descriptor_row(row, s, key, value))
                    if s == 350:
                        break
                    t = torch.tensor([s], device="cuda")
                    available = torch.tensor([context is not None], device="cuda")
                    v = engine.ema(current, t, n, m, context, available)
                    current, context, _ = engine.diffusion.deterministic_reverse_step(current, t, v, m)
                    if not torch.isfinite(current).all():
                        raise FloatingPointError("Nonfinite noisy rollout state")
                    if e007_active(runtime.cfg):
                        raise RuntimeError("E007 became active; stop E015 prefix")
                if (i + 1) % 5 == 0:
                    runtime.write("HEARTBEAT.json", status("gaussian_prefix", label=label, identities=i + 1))
            table(runtime, base + "/gaussian_descriptors.parquet", prefixes)
        prefixes = pq.read_table(prefix_path, use_threads=False).to_pylist()
        runtime.write(base + "/gaussian_metrics.json", distribution_report(prefixes))
        compare_prefix_e014(runtime, base, prefixes)
        if label.startswith("epoch_2_"):
            sampling_recommendation(runtime, base, primary_metrics, comparisons)
    engine.model.train()
    report = dict(
        label=label,
        summaries=summaries,
        paired_e014=comparisons,
        full_diagnostics=full,
        prospective_opened=False,
        full_t0_sampling=False,
        inference_self_conditioning=(
            "one model-only preliminary pass for paired states; previous generated estimate for prefixes"
        ),
    )
    runtime.write(base + "/COMPLETE.json", report)
    return report


def compare_prefix_e014(runtime, base, new):
    reports = []
    for comparator in ("epoch_2_fraction_1.00", "recovery_0"):
        old = pq.read_table(E014_ROOT / f"{comparator}.gaussian_descriptors.parquet", use_threads=False).to_pylist()
        for t in (400, 350):
            group = {}
            for label, rows in [("new", new), ("old", old)]:
                group[label] = {
                    pop: sorted(
                        [r for r in rows if r["timestep"] == t and r["population"] == pop],
                        key=lambda r: (STRATA.index(r["length_stratum"]), r["sample_id"]),
                    )
                    for pop in ("model", "ref_a", "ref_b")
                }
            identities = [r["sample_id"] for r in group["new"]["model"]]
            if any([r["sample_id"] for r in g] != identities for v in group.values() for g in v.values()):
                raise ValueError("Unpaired Gaussian populations")
            raw = {key: np.array([r["descriptor"] for r in group["new"][key]]) for key in ("ref_a", "ref_b")}
            standardizer = ReferenceStandardizer(raw["ref_a"], raw["ref_b"])
            a, b = standardizer.transform(raw["ref_a"]), standardizer.transform(raw["ref_b"])
            h = bandwidth(a, b)
            p = projections(a.shape[1])
            spectrum = [
                i
                for i, c in enumerate(standardizer.columns)
                if NAMES[c] in ("eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3")
            ]
            comparisons = {}
            for label in ("new", "old"):
                values = np.array([r["descriptor"] for r in group[label]["model"]])
                comparisons[label] = Comparison(standardizer.transform(values), b, values, raw["ref_b"], h, p, spectrum)
            rng = np.random.default_rng(15015)
            draws = []
            strata = [r["length_stratum"] for r in group["new"]["model"]]
            groups = [np.flatnonzero(np.array(strata) == s) for s in STRATA]
            for _ in range(5000):
                indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
                draws.append(
                    comparisons["new"].metrics(indices)["energy"] - comparisons["old"].metrics(indices)["energy"]
                )
            delta = comparisons["new"].metrics()["energy"] - comparisons["old"].metrics()["energy"]
            reports.append(
                dict(
                    comparator=comparator,
                    timestep=t,
                    difference=delta,
                    ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                    replicates=5000,
                )
            )
    runtime.write(base + "/paired_gaussian_e014.json", reports)


def continue_after_epoch(runtime, folder, epoch):
    # Joint development gate; no choice of checkpoint based on prospective data.
    current = f"{folder}/evaluation/epoch_{epoch}_fraction_1.00"
    previous = f"{folder}/evaluation/epoch_{epoch - 1}_fraction_1.00"
    new = json.loads(runtime.path(current + "/ema.summary.json").read_text())
    old = json.loads(runtime.path(previous + "/ema.summary.json").read_text())

    def overall(rows):
        return next(r for r in rows if r["dimension"] == "all")

    n, o = overall(new), overall(old)
    fields = [
        "v_mse",
        "offset_1_mse_angstrom2",
        "offset_2_mse_angstrom2",
        "offset_3_mse_angstrom2",
        "torsion_angular_mae_radians",
        "triple_inversion_fraction",
    ]
    changes = {field: n[field + "_equal_protein_mean"] / o[field + "_equal_protein_mean"] - 1 for field in fields}
    gaussian = json.loads(runtime.path(current + "/gaussian_metrics.json").read_text())
    safe = not any(r["actual_noisy_state_collapse"] for r in gaussian)
    improve_v = changes["v_mse"] < -runtime.cfg["continuation_relative_improvement"]
    improve_geometry = np.mean([changes[f] for f in fields[1:4]]) < -runtime.cfg["continuation_relative_improvement"]
    no_regression = all(v < 0.05 for k, v in changes.items() if k != "v_mse")
    proceed = safe and improve_v and improve_geometry and no_regression
    runtime.write(
        current + "/CONTINUATION.json",
        dict(
            continue_training=proceed,
            changes=changes,
            noisy_state_safe=safe,
            rule="joint >1% v/local improvement; no >5% structural regression; no noisy-state collapse",
            full_sampling_launched=False,
        ),
    )
    return proceed


def sampling_recommendation(runtime, base, metrics, comparisons):
    """Owner-only future experiment recommendation; never launch sampling."""
    gaussian = json.loads(runtime.path(base + "/gaussian_metrics.json").read_text())
    energies = json.loads(runtime.path(base + "/paired_gaussian_e014.json").read_text())
    local_improved = all(
        next(r for r in values if r["metric"] == "mean_local_mse_angstrom2")["ci95"][1] < 0
        and next(r for r in values if r["metric"] == "mean_local_mse_angstrom2")["e015"]
            <=.9*next(r for r in values if r["metric"] == "mean_local_mse_angstrom2")["old"]
        for values in comparisons.values()
    )
    chiral_improved = all(
        r["ci95"][1] < 0 and r["e015_ci95"][1] < 0.5
        for values in comparisons.values()
        for r in values
        if r["metric"] == "triple_inversion_fraction"
    )
    torsion_improved = all(
        r["ci95"][1] < 0
        for values in comparisons.values()
        for r in values
        if r["metric"] == "torsion_angular_mae_radians"
    )
    transport_improved = any(all(r["ci95"][1] < 0 for r in energies if r["timestep"] == t) for t in (400, 350))
    safe = not any(r["actual_noisy_state_collapse"] for r in gaussian)
    # Every length stratum must have no material (>5%) mean local regression.
    old = pq.read_table(E014_ROOT / "epoch_2_fraction_1.00.paired.parquet", use_threads=False).to_pylist()
    length_status = {}
    for stratum in STRATA:

        def local(rows, stratum=stratum):
            subset = [r for r in rows if r["length_stratum"] == stratum and r["timestep"] in (5, 50, 250, 400, 490)]
            return np.mean([sum(r[f"offset_{k}_mse_angstrom2"] for k in (1, 2, 3)) / 3 for r in subset])

        length_status[stratum] = bool(local(metrics) <= 1.05 * local(old))
    recommendation = (
        local_improved
        and chiral_improved
        and torsion_improved
        and transport_improved
        and safe
        and all(length_status.values())
    )
    runtime.write(
        base + "/FULL_SAMPLING_GATE.json",
        dict(
            recommended=bool(recommendation),
            local_improved=local_improved,
            chirality_below_chance_and_improved=chiral_improved,
            torsion_improved=torsion_improved,
            transport_improved=transport_improved,
            noisy_state_safe=safe,
            length_strata_safe=length_status,
            full_sampling_launched=False,
            requires_owner_authorization=True,
        ),
    )
