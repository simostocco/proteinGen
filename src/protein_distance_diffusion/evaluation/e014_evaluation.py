"""Fixed paired-noise diagnostics and read-only head telemetry for E014."""

import json
from collections import defaultdict

import numpy as np
import torch

from protein_distance_diffusion.evaluation.e013_one_step_benchmark import diagnostics
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed


def coordinate_batch(rows, cache, scale, device="cuda"):
    lengths = torch.tensor([r["length"] for r in rows], device=device)
    n = int(lengths.max())
    x = torch.zeros(len(rows), n, 3, device=device)
    mask = torch.arange(n, device=device)[None] < lengths[:, None]
    for i, row in enumerate(rows):
        x[i, : row["length"]] = torch.from_numpy(cache.get(row["sample_id"])).to(device) / scale
    return center_coordinates(x, mask), lengths, mask


def extended_metrics(xhat, x0, vhat, vstar, scale, component):
    metrics = diagnostics(xhat, x0, vhat, vstar, scale)
    n = len(x0)
    bonds = (xhat[1:] - xhat[:-1]).norm(dim=-1) * scale
    pf, _, _ = local_geometry(xhat)
    tf, _, _ = local_geometry(x0)
    assess = (pf[:, 29] > 0) & (tf[:, 29] > 0)
    sign_assess = assess & (tf[:, 27].abs() > 1e-3)
    phi_p = torch.atan2(pf[:, 25], pf[:, 26])
    phi_t = torch.atan2(tf[:, 25], tf[:, 26])
    angular = torch.atan2((phi_p - phi_t).sin(), (phi_p - phi_t).cos()).abs()
    ids = torch.arange(n, device=x0.device)
    sep = (ids[:, None] - ids[None]).abs()
    distances = torch.cdist(xhat * scale, xhat * scale)
    global_v, local_v = component["global_v"], component["local_effective"]
    gn, ln = global_v.norm(), local_v.norm()
    metrics.update(
        adjacent_count=n - 1,
        adjacent_valid_count=int(((bonds >= 3.4) & (bonds <= 4.2)).sum()),
        adjacent_valid_fraction=float(((bonds >= 3.4) & (bonds <= 4.2)).float().mean()),
        adjacent_rmse_angstrom=metrics["offset_1_mse_angstrom2"] ** 0.5,
        torsion_potential_count=max(n - 3, 0),
        torsion_angular_mae_radians=float(angular[assess].mean()) if assess.any() else None,
        triple_inversion_count=int(((pf[:, 27] * tf[:, 27] < 0) & sign_assess).sum()),
        triple_product_mse=float((pf[:, 27] - tf[:, 27]).square()[assess].mean()) if assess.any() else None,
        reference_radius_of_gyration_angstrom=float((x0 * scale).square().sum(-1).mean().sqrt()),
        clash_pair_count=int((sep > 3).sum()),
        clash_count=int(((distances < 3) & (sep > 3)).sum()),
        long_range_pair_count=int((sep >= 12).sum()),
        long_range_contact_count=int(((distances < 8) & (sep >= 12)).sum()),
        global_output_norm=float(gn),
        local_effective_output_norm=float(ln),
        local_global_norm_ratio=float(ln / (gn + 1e-8)),
        global_local_cosine=float((global_v * local_v).sum() / (gn * ln + 1e-8)),
        local_gate_mean=float(component["gate"].mean()),
        local_gate_median=float(component["gate"].median()),
        frame_reliability_mean=float(component["reliability"].mean()),
        frame_reliability_median=float(component["reliability"].median()),
        frame_assessable_count=int(component["assessable"].sum()),
        frame_residue_count=n,
        frame_assessable_fraction=float(component["assessable"].float().mean()),
    )
    return metrics


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[("all", "all")].append(row)
        groups[("timestep", row["timestep"])].append(row)
        groups[("length", row["length_stratum"])].append(row)
        groups[("length_timestep", f"{row['length_stratum']}|{row['timestep']}")].append(row)
        if "split" in row:
            groups[("split", row["split"])].append(row)
            groups[("split_timestep", f"{row['split']}|{row['timestep']}")].append(row)
    output = []
    for (dimension, value), values in groups.items():
        summary = dict(
            dimension=dimension,
            value=value,
            protein_condition_denominator=len(values),
            collapse_count=sum(r["collapsed"] for r in values),
            finite_count=sum(r["finite"] for r in values),
        )
        summary["collapse_fraction"] = summary["collapse_count"] / len(values)
        summary["rg_ratio_quantiles"] = [
            float(x) for x in np.quantile([r["radius_ratio"] for r in values], [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
        ]
        exclude = {"timestep", "length", "length_stratum", "sample_id", "finite", "collapsed", "historical_split"}
        for field in values[0]:
            numbers = [r[field] for r in values if r[field] is not None and isinstance(r[field], (int, float))]
            if field not in exclude and numbers:
                if field.endswith("_count") or field.endswith("_assessed"):
                    summary[field + "_sum"] = sum(numbers)
                else:
                    summary[field + "_equal_protein_mean"] = sum(numbers) / len(numbers)
                    summary[field + "_protein_denominator"] = len(numbers)
        output.append(summary)
    return output


@torch.no_grad()
def evaluate(model, diffusion, rows, cache, cfg, directory, name, historical=False):
    path = directory / f"{name}.summary.json"
    if path.exists():
        return json.loads(path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    metrics = []
    for position, row in enumerate(rows):
        x, lengths, mask = coordinate_batch([row], cache, cfg["coordinate_scale_angstrom"])
        for timestep in cfg["noise_timesteps"]:
            seed = (
                1300000 + row["historical_index"] * 500 + timestep
                if historical
                else noise_seed("e014-paired-validation-v1", row["sample_id"], timestep)
            )
            g = torch.Generator(device=x.device).manual_seed(seed)
            t = torch.tensor([timestep], device=x.device)
            batch = diffusion.make_training_batch(x, mask, timesteps=t, generator=g)
            prediction, components = model.telemetry(batch.noisy_coordinates, t, lengths, mask)
            xhat = diffusion.reconstruct_x0(batch.noisy_coordinates, t, prediction, mask)
            result = extended_metrics(
                xhat[0],
                x[0],
                prediction[0],
                batch.coordinate_v_target[0],
                cfg["coordinate_scale_angstrom"],
                components[0],
            )
            result.update(
                sample_id=row["sample_id"],
                length=row["length"],
                length_stratum=row["length_stratum"],
                timestep=timestep,
                split=row["split"],
            )
            if not result["finite"]:
                raise ValueError("nonfinite evaluation: hard stop")
            metrics.append(result)
        if (position + 1) % 100 == 0:
            print(f"evaluation {name}: {position + 1}/{len(rows)} identities", flush=True)
    summary = dict(
        name=name,
        identity_count=len(rows),
        condition_count=len(metrics),
        groups=aggregate(metrics),
        paired_noise="historical E013 exact indices/seeds" if historical else "SHA256(namespace|sample_id|t)",
        training_loss_unchanged=True,
    )
    atomic_json(directory / f"{name}.metrics.json", metrics)
    atomic_json(path, summary)
    return summary


def primary_v(summary):
    return next(r["v_mse_equal_protein_mean"] for r in summary["groups"] if r["dimension"] == "all")


def adjudicate_final(initial, final, completed_epochs, cfg):
    if completed_epochs < cfg["minimum_epochs"]:
        return dict(status="insufficient_complete_exposure", architectural_conclusion=None, stage_d_authorized=False)
    end = {r["value"]: r for r in final["groups"] if r["dimension"] == "timestep"}
    limit = cfg["scientific_adjudication"]
    v_improvement = 1 - primary_v(final) / primary_v(initial)
    local_improvement = 1 - next(
        r["local_mse_angstrom2_equal_protein_mean"] for r in final["groups"] if r["dimension"] == "all"
    ) / next(r["local_mse_angstrom2_equal_protein_mean"] for r in initial["groups"] if r["dimension"] == "all")
    high = end[490]
    low = end[5]
    low_good = low["adjacent_valid_fraction_equal_protein_mean"] >= limit["acceptable_adjacent_valid_fraction"]
    high_good = (
        high["collapse_fraction"] <= limit["max_high_noise_collapse_fraction"]
        and high["clash_fraction_equal_protein_mean"] <= limit["max_high_noise_clash_fraction"]
    )
    improving = v_improvement >= limit["substantially_improved_fraction"]
    if not improving:
        case = "C"
    elif high_good and not low_good:
        case = "D"
    elif low_good and not high_good:
        case = "E"
    elif high_good and low_good and local_improvement >= limit["substantially_improved_fraction"]:
        case = "A"
    else:
        case = "B"
    return dict(
        status="completed_minimum_exposure",
        case=case,
        v_improvement_fraction=v_improvement,
        local_improvement_fraction=local_improvement,
        high_noise_collapse_count=high["collapse_count"],
        high_noise_collapse_denominator=high["protein_condition_denominator"],
        high_noise_acceptable=high_good,
        low_noise_local_acceptable=low_good,
        stage_d_authorized=case == "A",
        comparator="paired E014 initialization; historical E007/E013 context is non-matched",
        v_geometry_divergence_persists=improving and not (high_good and low_good and local_improvement >= 0.1),
        v_local_distance_divergence_persists=improving and local_improvement < 0.1,
    )
