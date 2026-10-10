"""Metric-blind TRAIN-only gate and gradient calibration; immutable output."""

import argparse
import hashlib
import json

import e015_bootstrap
import numpy as np
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import E015MultiscaleParityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import canonical_hash, noise_seed
from protein_distance_diffusion.training.e015_losses import per_protein_losses
from protein_distance_diffusion.training.e015_runtime import GPULease, Runtime, e007_active
from protein_distance_diffusion.training.e015_snr import SNRBinSampler, frozen_inverse_gradient_weights

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def calibration_panel(runtime):
    strata = ("20-64", "65-128", "129-256", "257-384", "385-500")
    result = []
    for stratum in strata:
        group = [r for r in runtime.train if r["length_stratum"] == stratum]
        group.sort(key=lambda r: hashlib.sha256(("e015-calibration-v1|" + r["sample_id"]).encode()).digest())
        result.extend(group[: runtime.cfg["calibration"]["per_stratum"]])
    return result


def correlation(a, b):
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    denominator = a.norm() * b.norm()
    return float((a * b).sum() / denominator) if denominator > 1e-12 else None


def circular_correlation(a, b):
    ca = torch.atan2(a.sin().mean(), a.cos().mean())
    cb = torch.atan2(b.sin().mean(), b.cos().mean())
    return correlation((a - ca).sin(), (b - cb).sin())


@torch.no_grad()
def reliability(runtime):
    path = runtime.path("calibration/reliability.json")
    if path.exists():
        return json.loads(path.read_text())
    panel = calibration_panel(runtime)
    diffusion = CoordinateVPDiffusion()
    results = []
    for s in runtime.cfg["calibration"]["noise_grid"]:
        values = []
        for row in panel:
            x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"], device="cpu")
            generator = torch.Generator().manual_seed(noise_seed("e015-train-gate-v1", row["sample_id"], s))
            batch = diffusion.make_training_batch(x, m, timesteps=torch.tensor([s]), generator=generator)
            p, q = batch.noisy_coordinates[0], x[0]
            pf, _, reliable = local_geometry(p)
            tf, _, _ = local_geometry(q)
            distances = [correlation((p[k:] - p[:-k]).norm(dim=-1), (q[k:] - q[:-k]).norm(dim=-1)) for k in (1, 2, 3)]
            valid = (pf[:, 29] > 0) & (tf[:, 29] > 0)
            torsion = (
                circular_correlation(
                    torch.atan2(pf[valid, 25], pf[valid, 26]), torch.atan2(tf[valid, 25], tf[valid, 26])
                )
                if valid.sum() > 3
                else None
            )
            assess = valid & (tf[:, 27].abs() > 1e-3)
            signs = float((pf[assess, 27] * tf[assess, 27] > 0).float().mean()) if assess.any() else None
            values.append(
                dict(
                    distance=np.mean([d for d in distances if d is not None]),
                    torsion=torsion,
                    chirality=signs,
                    frame_reliability=float(reliable.mean()),
                    frame_assessability=float((pf[:, 28] > 0).float().mean()),
                    distance_offsets=distances,
                )
            )
        result = dict(
            timestep=s, logsnr=float(torch.log(diffusion.alphas_cumprod[s]) - torch.log1p(-diffusion.alphas_cumprod[s]))
        )
        for name in ("distance", "torsion", "chirality", "frame_reliability", "frame_assessability"):
            observed = [v[name] for v in values if v[name] is not None and np.isfinite(v[name])]
            result[name] = float(np.mean(observed)) if observed else None
        result["per_protein"] = values
        results.append(result)
        print("TRAIN reliability", s, {k: v for k, v in result.items() if k != "per_protein"}, flush=True)
    midpoints = {}
    for term, threshold in [("distance", 0.25), ("torsion", 0.15), ("chirality", 0.60)]:
        good = [r for r in results if r[term] is not None and r[term] > threshold]
        if not good:
            raise RuntimeError(f"TRAIN-only {term} reliability threshold never reached")
        midpoints[term] = min(r["logsnr"] for r in good)
    midpoints["clash"] = midpoints["distance"]
    output = dict(
        panel=[r["sample_id"] for r in panel],
        panel_hash=canonical_hash([r["sample_id"] for r in panel]),
        population_hash=runtime.cfg["population_hash"],
        results=results,
        midpoints=midpoints,
        gate_width_logsnr=1.0,
        thresholds={"distance": 0.25, "torsion": 0.15, "chirality": 0.60},
        distance_rule="mean of per-protein Pearson correlations for offsets 1/2/3",
        torsion_rule="Jammalamadaka-style centered circular sine correlation, equal proteins",
        clash_rule="same distance reliability midpoint; frozen 3 Angstrom clash definition",
        validation_read=False,
    )
    runtime.write("calibration/reliability.json", output)
    return output


def gradients(runtime):
    path = runtime.path("calibration/frozen.json")
    if path.exists():
        return json.loads(path.read_text())
    if e007_active(runtime.cfg):
        raise RuntimeError("E007 active; defer CUDA calibration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    lease = GPULease(runtime)
    reliable = reliability(runtime)
    torch.manual_seed(runtime.cfg["seed"])
    model = E015MultiscaleParityDenoiser(**runtime.cfg["model"]).cuda().train()
    model.parity_midpoint.fill_(reliable["midpoints"]["chirality"])
    diffusion = CoordinateVPDiffusion()
    sampler = SNRBinSampler()
    # CPU/GPU feasibility selected from short TRAIN strata, not validation.
    short = [r for r in calibration_panel(runtime) if r["length"] <= 64]
    measurements = []

    def measure(row, t, term):
        x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
        generator = torch.Generator(device="cuda").manual_seed(
            noise_seed("e015-train-gradient-v1", row["sample_id"], t)
        )
        times = torch.tensor([t], device="cuda")
        batch = diffusion.make_training_batch(x, m, timesteps=times, generator=generator)
        v = model(batch.noisy_coordinates, times, n, m)
        vloss = ((v - batch.coordinate_v_target).square() * m[..., None]).sum() / (3 * m.sum())
        if term == "v":
            loss = vloss
        else:
            xhat = diffusion.reconstruct_x0(batch.noisy_coordinates, times, v, m)
            loss = per_protein_losses(xhat, x, m)[term].mean()
        grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
        norm = sum(g.float().square().sum() for g in grads if g is not None).sqrt()
        value = float(norm)
        if not np.isfinite(value) or (value <= 0 and term == "v"):
            raise RuntimeError(f"Nonpositive/nonfinite TRAIN gradient for {term}")
        return value, float(loss.detach())

    norms = []
    for b, bin_times in enumerate(sampler.bins):
        t = int(bin_times[len(bin_times) // 2])
        group = []
        for row in short[: runtime.cfg["calibration"]["gradient_examples_per_bin"]]:
            norm, loss = measure(row, t, "v")
            group.append(norm)
            measurements.append(dict(bin=b, timestep=t, term="v", norm=norm, loss=loss, sample_id=row["sample_id"]))
        norms.append(float(np.mean(group)))
    weights = frozen_inverse_gradient_weights(norms)
    coefficients = {}
    for term, fraction in runtime.cfg["calibration"]["relative_gradient_targets"].items():
        # Closest grid point with sigmoid gate >= 0.5, away from t=0 degeneracy.
        grid = [r for r in reliable["results"] if r["logsnr"] >= reliable["midpoints"][term] and r["timestep"] >= 5]
        if not grid:
            raise RuntimeError(f"No calibrated active TRAIN region for {term}")
        point = max(grid, key=lambda r: r["timestep"])
        t = point["timestep"]
        gate = torch.sigmoid(torch.tensor(point["logsnr"] - reliable["midpoints"][term]))
        panel = calibration_panel(runtime)
        selected = [
            row
            for stratum in ("20-64", "65-128", "129-256", "257-384", "385-500")
            for row in [r for r in panel if r["length_stratum"] == stratum][
                : runtime.cfg["calibration"]["gradient_examples_per_bin"]
            ]
        ]
        vnorms, snorms = [], []
        bin_id = next(b for b, ts in enumerate(sampler.bins) if t in ts.tolist())
        for row in selected:
            vn, _ = measure(row, t, "v")
            sn, loss = measure(row, t, term)
            vnorms.append(vn)
            snorms.append(sn)
            measurements.append(
                dict(
                    term=term,
                    timestep=t,
                    v_norm=vn,
                    struct_norm=sn,
                    gate=float(gate),
                    loss=loss,
                    sample_id=row["sample_id"],
                )
            )
        mean_sn = float(np.mean(snorms))
        if mean_sn <= 0:
            raise RuntimeError(f"TRAIN panel cannot identify a positive initial gradient for {term}")
        coefficient = fraction * float(np.mean(vnorms)) * float(weights[bin_id]) / (mean_sn * float(gate))
        coefficients[term] = min(runtime.cfg["calibration"]["coefficient_cap"], max(1e-6, coefficient))
    output = dict(
        reliability_hash=canonical_hash(reliable),
        midpoints=reliable["midpoints"],
        gate_width_logsnr=1.0,
        snr_bins=[b.tolist() for b in sampler.bins],
        gradient_norms=norms,
        v_weights=weights.tolist(),
        coefficients=coefficients,
        measurements=measurements,
        parameter_count=sum(p.numel() for p in model.parameters()),
        calibration_precision="FP32",
        validation_read=False,
        population_hash=runtime.cfg["population_hash"],
    )
    runtime.write("calibration/frozen.json", output)
    lease.stream.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["reliability", "gradients"])
    args = parser.parse_args()
    runtime = Runtime(CONFIG)
    if args.action == "reliability":
        reliability(runtime)
    else:
        gradients(runtime)


if __name__ == "__main__":
    main()
