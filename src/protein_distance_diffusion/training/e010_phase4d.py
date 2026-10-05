"""Phase 4D objective and per-example telemetry; no optimizer or training loop."""

import torch

from ..models.e010_hybrid_local import PSEUDOSCALAR_INDEX, local_representation
from .local_geometry import phase4c_losses


def hybrid_losses(prediction, pg, source, target, mask, *, beta=1.0, rho=0.01, delta_cart=0.01):
    if any(v < 0 or not torch.isfinite(torch.tensor(v)) for v in (beta, rho, delta_cart)):
        raise ValueError("objective coefficients must be finite and nonnegative")

    # Sanitize padding before historical arithmetic (including NaN padding).
    def clean(t):
        return torch.where(mask[..., None], t, 0)

    p, g, s, y = map(clean, (prediction, pg.detach(), source, target))
    local = phase4c_losses(p, s, y, mask)
    baseline = phase4c_losses(g, s, y, mask)["cartesian"].detach()
    guard = torch.relu(local["cartesian"] - (1 + delta_cart) * baseline)
    disp = (((p - g).square().sum(-1) * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    return {
        **local,
        "global_cartesian": baseline,
        "guard": guard,
        "displacement": disp,
        "total": local["local_mean"] + beta * guard + rho * disp,
    }


@torch.no_grad()
def example_metrics(prediction, pg, source, target, mask):
    """Per-protein rows for explicit identity/condition/stratum aggregation.

    Alignment forbids reflections. Chirality compares normalized signed triple
    products on jointly assessable, nonplanar windows; report denominators.
    Empty proteins are deliberately rejected by the historical loss contract.
    """
    rows = []
    for p, g, s, y, m in zip(prediction, pg, source, target, mask, strict=True):
        if not torch.isfinite(p[m]).all():
            rows.append({"finite": False, "metrics_unavailable": "nonfinite valid coordinates"})
            continue
        losses = hybrid_losses(p[None], g[None], s[None], y[None], m[None])
        pp, yy = p[m].double(), y[m].double()
        pc, yc = pp - pp.mean(0), yy - yy.mean(0)
        u, _, vh = torch.linalg.svd(pc.T @ yc)
        d = torch.eye(3, dtype=pc.dtype, device=pc.device)
        d[-1, -1] = torch.linalg.det(u @ vh)
        aligned = pc @ (u @ d @ vh)
        rep, truth = local_representation(p[None], m[None]), local_representation(y[None], m[None])
        ps, ts = rep["features"][0, :, PSEUDOSCALAR_INDEX], truth["features"][0, :, PSEUDOSCALAR_INDEX]
        joint = (
            rep["pseudoscalar_assessable"][0]
            & truth["pseudoscalar_assessable"][0]
            & (ps.abs() > 1e-6)
            & (ts.abs() > 1e-6)
        )
        disp = (p - g)[m].norm(dim=-1)
        rows.append(
            {
                "aligned_rmsd": float((aligned - yc).square().sum(-1).mean().sqrt()),
                "raw_cartesian": float(losses["cartesian"]),
                "local_rmse": {str(k): float(losses[f"local_{k}"].sqrt()) for k in (1, 2, 3)},
                "mean_local_rmse": sum(float(losses[f"local_{k}"].sqrt()) for k in (1, 2, 3)) / 3,
                "chirality_inversions": int(((ps * ts < 0) & joint).sum()),
                "chirality_assessable": int(joint.sum()),
                "frame_eligible": int(rep["eligible"].sum()),
                "frame_degenerate": int(rep["degenerate"].sum()),
                "frame_interior": int(rep["interior"].sum()),
                "displacement_rms": float(disp.square().mean().sqrt()),
                "displacement_max": float(disp.max()),
                "finite": bool(torch.isfinite(pp).all()),
                "radius_gyration": float(pc.square().sum(-1).mean().sqrt()),
                "collapse": bool(pc.square().sum(-1).mean().sqrt() < 1e-3),
            }
        )
    return rows


def aggregate_metrics(rows):
    """Rows must carry sample_id, condition, stratum. Pairwise distance diversity
    is computed separately from coordinates during later panel evaluation.
    """

    def summarize(group):
        if not all(r["finite"] for r in group):
            return {
                "examples": len(group),
                "all_finite": False,
                "nonfinite_examples": sum(not r["finite"] for r in group),
                "metrics_unavailable": True,
            }

        def mean(key):
            return sum(r[key] for r in group) / len(group)

        return {
            "examples": len(group),
            "identities": len({r["sample_id"] for r in group}),
            **{
                k: mean(k)
                for k in ("aligned_rmsd", "raw_cartesian", "mean_local_rmse", "displacement_rms", "radius_gyration")
            },
            "local_rmse": {k: sum(r["local_rmse"][k] for r in group) / len(group) for k in ("1", "2", "3")},
            **{
                k: sum(r[k] for r in group)
                for k in (
                    "chirality_inversions",
                    "chirality_assessable",
                    "frame_eligible",
                    "frame_degenerate",
                    "frame_interior",
                )
            },
            "all_finite": all(r["finite"] for r in group),
            "any_collapse": any(r["collapse"] for r in group),
            "displacement_max": max(r["displacement_max"] for r in group),
        }

    if not rows:
        raise ValueError("no metric rows")
    return {
        "overall": summarize(rows),
        "by_condition": {
            str(v): summarize([r for r in rows if r["condition"] == v]) for v in sorted({r["condition"] for r in rows})
        },
        "by_stratum": {
            v: summarize([r for r in rows if r["stratum"] == v]) for v in sorted({r["stratum"] for r in rows})
        },
    }


@torch.no_grad()
def distance_diversity(predictions, masks):
    """Same-length group: mean pairwise RMS difference between distance matrices."""
    signatures = [torch.cdist(p[m], p[m]) for p, m in zip(predictions, masks, strict=True)]
    if len({s.shape for s in signatures}) != 1:
        raise ValueError("diversity requires matched lengths")
    values = [(a - b).square().mean().sqrt() for i, a in enumerate(signatures) for b in signatures[i + 1 :]]
    return None if not values else float(torch.stack(values).mean())
