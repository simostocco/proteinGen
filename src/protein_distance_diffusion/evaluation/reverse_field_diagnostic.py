"""Read-only metrics for a paired deterministic forward-noise reference path.

The reference is not the unique generative reverse trajectory. Reverse mathematics
come exclusively from CoordinateVPDiffusion; no sampler is reimplemented here.
"""

import hashlib

import numpy as np
import torch

from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates


def paired_state(diffusion, clean, epsilon, timestep, mask):
    alpha, sigma = diffusion.alpha_sigma(timestep, clean)
    return center_coordinates(alpha * clean + sigma * epsilon, mask)


def reverse_identity(diffusion, clean, epsilon, timestep, mask, tolerance=5e-6):
    current = paired_state(diffusion, clean, epsilon, timestep, mask)
    target_v = diffusion.training_target(clean, epsilon, timestep, mask)
    previous, _, _ = diffusion.deterministic_reverse_step(current, timestep, target_v, mask)
    oracle = paired_state(diffusion, clean, epsilon, timestep - 1, mask)
    error = float((previous - oracle).abs().max())
    if not torch.isfinite(previous).all() or error > tolerance:
        raise ValueError(f"true-v reverse identity failed: {error} > {tolerance}")
    return error


def vector_metrics(predicted, target, mask, prefix):
    a, b = predicted[mask].double().flatten(), target[mask].double().flatten()
    na, nb = a.norm(), b.norm()
    assessable = bool((na > 1e-12) & (nb > 1e-12))
    ratio = float(na / nb) if nb > 1e-12 else None
    return {
        f"{prefix}_cosine": float((a * b).sum() / (na * nb)) if assessable else None,
        f"{prefix}_norm_ratio": ratio,
        f"{prefix}_log_norm_ratio": float(np.log(ratio)) if ratio is not None and ratio > 0 else None,
        f"{prefix}_predicted_norm": float(na),
        f"{prefix}_target_norm": float(nb),
        f"{prefix}_cosine_assessable": assessable,
    }


def geometry_metrics(predicted, target, mask, scale):
    a, b = predicted[mask].double() * scale, target[mask].double() * scale
    a, b = a - a.mean(0), b - b.mean(0)
    ra, rb = a.square().sum(-1).mean().sqrt(), b.square().sum(-1).mean().sqrt()
    u, _, vh = torch.linalg.svd(a.T @ b)
    sign = torch.linalg.det(u @ vh)
    correction = torch.diag(torch.stack([sign.new_tensor(1), sign.new_tensor(1), sign]))
    result = dict(
        raw_mse=float((a - b).square().mean()) / scale**2,
        aligned_rmsd_angstrom=float((a @ (u @ correction @ vh) - b).square().sum(-1).mean().sqrt()),
        rg_predicted_angstrom=float(ra),
        rg_reference_angstrom=float(rb),
        rg_ratio=float(ra / rb) if rb > 1e-12 else None,
        finite=bool(torch.isfinite(predicted).all()),
    )
    for k in (1, 2, 3):
        da, db = (a[k:] - a[:-k]).norm(dim=-1), (b[k:] - b[:-k]).norm(dim=-1)
        result[f"L{k}_angstrom2"] = float((da - db).square().mean())
    result["pair_distance_mse_angstrom2"] = float((torch.pdist(a) - torch.pdist(b)).square().mean())
    return result


def select_rollout_subset(rows, strata, per_stratum=20):
    chosen = []
    for stratum in strata:
        eligible = [r for r in rows if r["length_stratum"] == stratum]
        eligible.sort(
            key=lambda r: hashlib.sha256(f"e014-reverse-field-50pct-v1|{r['sample_id']}".encode()).hexdigest()
        )
        if len(eligible) < per_stratum:
            raise ValueError("insufficient frozen development identities")
        chosen.extend(eligible[:per_stratum])
    return chosen


def rollout_timesteps(start=490, stop=440):
    if not 0 < stop < start:
        raise ValueError("rollout must have strictly descending positive timesteps")
    return list(range(start, stop, -1))


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"denominator": 0}
    return dict(
        denominator=len(values),
        mean=float(np.mean(values)),
        **dict(
            zip(
                ("p05", "p25", "median", "p75", "p95"),
                np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                strict=True,
            )
        ),
    )


def summarize(rows):
    result = {"denominator": len(rows), "identity_count": len({r["sample_id"] for r in rows})}
    for key in rows[0]:
        if key in ("sample_id", "length", "length_stratum", "timestep", "input_timestep"):
            continue
        values = [r[key] for r in rows if isinstance(r[key], (float, int))]
        if values:
            result[key] = distribution(values)
        if key.endswith("cosine"):
            valid = [r[key] for r in rows if r[key] is not None]
            result[key + "_positive_fractions"] = {
                "denominator": len(valid),
                **{
                    str(t): dict(
                        count=sum(v > t for v in valid),
                        fraction=sum(v > t for v in valid) / len(valid) if valid else None,
                    )
                    for t in (0, 0.5, 0.9)
                },
            }
    if "model_mse" in rows[0]:
        result["relative_gain_of_equal_protein_means"] = 1 - np.mean([r["model_mse"] for r in rows]) / np.mean(
            [r["noop_mse"] for r in rows]
        )
    if "direct_collapsed" in rows[0]:
        result["direct_collapse_count"] = sum(r["direct_collapsed"] for r in rows)
    return result


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else None

    def ranks(x):
        _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        end = np.cumsum(counts)
        return ((end - counts + end - 1) / 2)[inverse]

    return dict(denominator=len(a), pearson=pearson(a, b), spearman=pearson(ranks(a), ranks(b)))
