#!/usr/bin/env python3
"""Read-only E009 objective audit and bounded paired length-64 diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from protein_distance_diffusion.models.e009_bayesian_refiner import (
    normalized_refiner_losses,
)
from scripts import run_e009_bayesian_refiner as e009

ROOT = Path("reports/experiments/E009_bayesian_geometry_refiner")
CFG = Path("configs/e009_bayesian_refiner_v4_execution.yaml")
OUT = ROOT / "objective_diagnostic_v1"
ARM_CONFIG = {
    "schema": "e009_paired_objective_protocol_v1",
    "authorizes_downstream": False,
    "max_updates_per_arm": 1000,
    "success_gate_aligned_posterior_mean_rmse_angstrom": 0.10,
    "evaluation_schedule": [0, 10, 50, 100, 250, 500, 1000],
    "arm_a": {
        "coordinate_nll": 1.0,
        "log_sigma_squared_regularizer": 0.0001,
        "regularizer_reduction": "mean over eligible residues",
        "empirical_geometry_prior_terms": False,
    },
    "arm_b": {
        "coordinate_nll": 1.0,
        "bond_prior_nll": 0.1,
        "angle_prior_nll": 0.1,
        "torsion_prior_nll": 0.1,
        "long_range_pair_nll": 0.2,
        "contact_nll": 0.2,
        "radius_of_gyration_nll": 0.2,
        "chirality_nll": 0.1,
        "posterior_kl": 0.01,
        "normalization": (
            "coordinate mean over xyz scalars multiplied by 3 to per-residue; each local "
            "NLL mean over eligible elements; radius one value per structure; KL mean "
            "per residue; explicit coefficients"
        ),
    },
    "optimizer": {"name": "AdamW", "learning_rate": 0.0002, "weight_decay": 0.01, "gradient_clip_norm": 1.0},
    "initialization_seed": 8015,
    "paired_stochastic_seed_rule": (
        "torch RNG reset to seed + update before each arm update; same initialization state and fixed corruption"
    ),
}


def _tensor_norm(grads):
    return math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads if g is not None))


def _cos(a, b):
    dot = sum(
        float((x.detach().double() * y.detach().double()).sum())
        for x, y in zip(a, b, strict=True)
        if x is not None and y is not None
    )
    na, nb = _tensor_norm(a), _tensor_norm(b)
    return dot / (na * nb) if na and nb else None


def _grads(loss, params, retain_graph=True):
    raw = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, raw, strict=True)]


def _flatgrad_norm(grads):
    return _tensor_norm(grads)


def _losses(post, target, prior, mask):
    # Keep the historical implementation intact and expose all individual geometry terms.
    return normalized_refiner_losses(
        post,
        target,
        prior,
        mask,
        {
            "geometry_nll": 0.0,
            "long_range_pair_nll": 0.0,
            "contact_nll": 0.0,
            "radius_of_gyration_nll": 0.0,
            "chirality_nll": 0.0,
            "posterior_kl": 0.0,
        },
    )


def _component_audit(model, prior, d, device, seed, checkpoint=None):
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("schema") != "e009_overfit64_resume_v1" or state.get("step") != 1000:
            raise ValueError("saved failed run checkpoint is not the expected update-1000 E009 state")
        model.load_state_dict(state["model"], strict=True)
    target, coarse, mask = (d[k].to(device)[None] for k in ("target", "coarse", "mask"))
    torch.manual_seed(seed + (1000 if checkpoint else 0))
    prior.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    post = model(coarse, mask)
    losses = _losses(post, target, prior, mask)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [
        "coordinate_nll",
        "bond_prior_nll",
        "angle_prior_nll",
        "torsion_prior_nll",
        "long_range_pair_nll",
        "contact_nll",
        "radius_of_gyration_nll",
        "chirality_nll",
        "posterior_kl",
    ]
    coeff = {
        "coordinate_nll": 1.0,
        "bond_prior_nll": 0.1 / 3,
        "angle_prior_nll": 0.1 / 3,
        "torsion_prior_nll": 0.1 / 3,
        "long_range_pair_nll": 0.2,
        "contact_nll": 0.2,
        "radius_of_gyration_nll": 0.2,
        "chirality_nll": 0.1,
        "posterior_kl": 0.01,
    }
    # Each named raw term is already per its documented element type, except the historical
    # geometry aggregate. Audit its constituent terms independently.
    weighted = {n: losses[n] * coeff[n] for n in names}
    total = sum(weighted.values())
    complete = _grads(total, params, retain_graph=True)
    coordinate = _grads(losses["coordinate_nll"], params, retain_graph=True)
    table = {}
    for n in names:
        g = _grads(losses[n], params, retain_graph=True)
        table[n] = {
            "eligible_element_count": _eligible_counts(mask)[n],
            "reduction": _reductions()[n],
            "raw_scalar_loss": float(losses[n].detach()),
            "configured_coefficient": coeff[n],
            "weighted_contribution": float(weighted[n].detach()),
            "parameter_gradient_l2_norm": _flatgrad_norm(g),
            "cosine_with_coordinate_reconstruction_gradient": _cos(g, coordinate),
            "cosine_with_complete_objective_gradient": _cos(g, complete),
        }
    total_norm = _flatgrad_norm(complete)
    return {
        "components": table,
        "complete_objective_gradient_l2_norm": total_norm,
        "coordinate_gradient_l2_norm": _flatgrad_norm(coordinate),
        "total_gradient_norm_before_clipping": total_norm,
        "total_gradient_norm_after_clipping": min(total_norm, 1.0),
        "saved_run_clipped_updates_fraction": None,
    }


def _eligible_counts(mask):
    n = int(mask.sum())
    b = int((mask[:, 1:] * mask[:, :-1]).sum())
    a = int((mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).sum())
    t = int((mask[:, 3:] * mask[:, 2:-1] * mask[:, 1:-2] * mask[:, :-3]).sum())
    pairs = int(
        (
            mask[:, :, None]
            * mask[:, None, :]
            * (
                torch.arange(mask.shape[1], device=mask.device)[:, None]
                - torch.arange(mask.shape[1], device=mask.device)[None, :]
            )
            .abs()
            .ge(8)
        ).sum()
    )
    return {
        "coordinate_nll": n * 3,
        "bond_prior_nll": b,
        "angle_prior_nll": a,
        "torsion_prior_nll": t,
        "long_range_pair_nll": pairs,
        "contact_nll": pairs,
        "radius_of_gyration_nll": 1,
        "chirality_nll": int((mask[:, 3:] * mask[:, :-3]).sum()),
        "posterior_kl": n,
    }


def _reductions():
    return {
        "coordinate_nll": "mean over coordinate scalars (3 per eligible residue)",
        "bond_prior_nll": "mean over eligible bonds",
        "angle_prior_nll": "mean over eligible angles",
        "torsion_prior_nll": "mean over eligible torsions",
        "long_range_pair_nll": "mean over eligible ordered pairs",
        "contact_nll": "mean over eligible ordered pairs",
        "radius_of_gyration_nll": "mean over structures",
        "chirality_nll": "mean over selected eligible quadruplets",
        "posterior_kl": "mean over eligible residues",
    }


def _normalization_review():
    return {
        "existing_objective": {
            "compatible_reductions": False,
            "reason": (
                "the existing weighted sum combines coordinate-scalar means, local element "
                "means, pair means, a structure mean, and a residue mean without converting "
                "them to a shared declared unit"
            ),
            "fail_closed": True,
        },
        "arm_b": {
            "compatible_reductions": True,
            "explicit_conversions": {
                "coordinate_nll": "multiply xyz-scalar mean by 3 to obtain per-residue mean",
                "bond_angle_torsion_pair_contact_chirality": (
                    "mean within eligible element family, then declared coefficient"
                ),
                "radius_of_gyration_nll": "one scalar per structure, then declared coefficient",
                "posterior_kl": "mean over eligible residues, then declared coefficient",
            },
        },
    }


def _validate_normalized_arm_protocol():
    required = {
        "coordinate_nll",
        "bond_prior_nll",
        "angle_prior_nll",
        "torsion_prior_nll",
        "long_range_pair_nll",
        "contact_nll",
        "radius_of_gyration_nll",
        "chirality_nll",
        "posterior_kl",
    }
    explicit = {
        "coordinate_nll": "mean over xyz scalars multiplied by 3, then per-residue mean",
        "bond_prior_nll": "mean over eligible bonds",
        "angle_prior_nll": "mean over eligible angles",
        "torsion_prior_nll": "mean over eligible torsions",
        "long_range_pair_nll": "mean over eligible ordered pairs",
        "contact_nll": "mean over eligible ordered pairs",
        "radius_of_gyration_nll": "one value per structure",
        "chirality_nll": "mean over eligible quadruplets",
        "posterior_kl": "mean over eligible residues",
    }
    if set(explicit) != required or not ARM_CONFIG["arm_b"].get("normalization"):
        raise ValueError("Arm B combines objective reductions without complete explicit normalization")
    if not required.issubset(ARM_CONFIG["arm_b"]):
        raise ValueError("Arm B objective is missing an explicit coefficient for a normalized component")
    return explicit


def _run_arm(name, model, prior, datum, cfg, device):
    target, coarse, mask = (datum[k].to(device)[None] for k in ("target", "coarse", "mask"))
    opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    schedule = ARM_CONFIG["evaluation_schedule"]
    rec, logs = [], []
    for step in range(1001):
        if step in schedule:
            torch.manual_seed(8015 + step + 771)
            with torch.no_grad():
                post = model(coarse, mask)
                rmse = e009._kabsch_rmse(post["mean"][0], target[0])
                vals = {k: float(v) for k, v in _losses(post, target, prior, mask).items()}
            rec.append(
                {"update": step, "aligned_posterior_mean_rmse_angstrom": rmse, "historical_component_values": vals}
            )
            if rmse <= 0.10:
                break
        if step == 1000:
            break
        torch.manual_seed(8015 + step)
        opt.zero_grad(set_to_none=True)
        post = model(coarse, mask)
        loss_terms = _losses(post, target, prior, mask)
        if name == "A":
            reg = post["log_sigma"].square().mean()
            objective = 3.0 * loss_terms["coordinate_nll"] + 0.0001 * reg
        else:
            objective = (
                3.0 * loss_terms["coordinate_nll"]
                + 0.1
                / 3
                * (loss_terms["bond_prior_nll"] + loss_terms["angle_prior_nll"] + loss_terms["torsion_prior_nll"])
                + 0.2
                * (loss_terms["long_range_pair_nll"] + loss_terms["contact_nll"] + loss_terms["radius_of_gyration_nll"])
                + 0.1 * loss_terms["chirality_nll"]
                + 0.01 * loss_terms["posterior_kl"]
            )
        objective.backward()
        params = [p for p in model.parameters() if p.grad is not None]
        pre = _tensor_norm([p.grad for p in params])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        postclip = _tensor_norm([p.grad for p in params])
        before = [p.detach().clone() for p in params]
        opt.step()
        stepnorm = _tensor_norm([p.detach() - old for p, old in zip(params, before, strict=True)])
        logs.append(
            {
                "update": step + 1,
                "unclipped_gradient_norm": pre,
                "clipped_gradient_norm": postclip,
                "clipped": pre > 1.0,
                "effective_optimizer_step_l2_norm": stepnorm,
            }
        )
    return {
        "arm": name,
        "updates": logs[-1]["update"] if logs else 0,
        "records": rec,
        "gradient_updates": logs,
        "clipping_fraction": sum(x["clipped"] for x in logs) / len(logs) if logs else 0,
        "passed": bool(rec[-1]["aligned_posterior_mean_rmse_angstrom"] <= 0.10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    cfg = e009.load_config(CFG)
    cache = Path(cfg["corruption"]["fixed_cache"])
    with __import__("numpy").load(cache, allow_pickle=False) as z:
        d = {
            "target": torch.from_numpy(z["target"].copy()),
            "coarse": torch.from_numpy(z["coarse"].copy()),
            "mask": torch.from_numpy(z["mask"].copy()).bool(),
        }
        _meta = json.loads(str(z["metadata"].item()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prior, _ = e009._prior_from_artifact(cfg, "cpu", use_case="prototype")
    prior = prior.to(device)
    torch.manual_seed(cfg["seed"])
    initial = e009._model(cfg, device)
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / ("objective_audit.json" if args.audit_only else "paired_result.json")
    if result_path.exists():
        raise FileExistsError(result_path)
    init_audit = _component_audit(initial, prior, d, device, cfg["seed"])
    run = json.loads(Path(cfg["overfit_result"]).read_text())
    logs = run["gradient_norms"]
    init_audit["saved_run_clipped_updates_fraction"] = sum(bool(x["clipped"]) for x in logs) / len(logs)
    ckpt_model = e009._model(cfg, device)
    ckpt_audit = _component_audit(ckpt_model, prior, d, device, cfg["seed"], cfg["overfit_checkpoint"])
    ckpt_audit["saved_run_clipped_updates_fraction"] = init_audit["saved_run_clipped_updates_fraction"]
    if args.audit_only:
        payload = {
            "schema": "e009_objective_gradient_audit_v1",
            "authorizes_downstream": False,
            "failed_run_preserved_read_only": True,
            "cache_sha256": e009.sha256(cache),
            "prior_sha256": e009.sha256(cfg["prior_artifact"]),
            "failed_run_result_sha256": e009.sha256(cfg["overfit_result"]),
            "failed_run_checkpoint_sha256": e009.sha256(cfg["overfit_checkpoint"]),
            "saved_run_status": run["status"],
            "saved_run_updates": run["updates"],
            "device": device,
            "normalization_findings": _reductions(),
            "normalization_compatibility": _normalization_review(),
            "initialization": init_audit,
            "update_1000": ckpt_audit,
        }
        e009.atomic_json(result_path, payload)
        print(result_path)
        return
    normalization_contract = _validate_normalized_arm_protocol()
    protocol_path = out / "paired_protocol.json"
    if not protocol_path.exists():
        protocol = copy.deepcopy(ARM_CONFIG)
        protocol["arm_b"]["explicit_component_normalizations"] = normalization_contract
        e009.atomic_json(protocol_path, protocol)
    else:
        raise FileExistsError(protocol_path)
    arms = []
    for name in ("A", "B"):
        torch.manual_seed(cfg["seed"])
        model = e009._model(cfg, device)
        arms.append(_run_arm(name, model, prior, d, cfg, device))
    passed = {a["arm"]: a["passed"] for a in arms}
    interpretation = (
        "empirical-prior/objective conflict"
        if passed == {"A": True, "B": False}
        else "objective normalization repaired prototype"
        if all(passed.values())
        else "model/input representation lacks memorization capacity"
        if not any(passed.values())
        else "inspect optimization or posterior parameterization"
    )
    e009.atomic_json(
        result_path,
        {
            "schema": "e009_paired_objective_diagnostic_v1",
            "authorizes_downstream": False,
            "interpretation": interpretation,
            "device": device,
            "protocol_sha256": e009.sha256(protocol_path),
            "cache_sha256": e009.sha256(cache),
            "prior_sha256": e009.sha256(cfg["prior_artifact"]),
            "arms": arms,
        },
    )
    print(result_path)


if __name__ == "__main__":
    main()
