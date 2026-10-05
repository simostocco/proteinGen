#!/usr/bin/env python3
"""Non-authorizing E008 failure diagnostics; never resumes or enters the pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.models.e008_kinematic_decoder import (
    BackboneKinematicDecoder,
    _kabsch_align,
    cartesian_to_internal,
    decoder_parameter_count,
    geometry_native_losses,
    internal_to_cartesian,
)
from scripts.run_e008_geometry_native_decoder import (
    _authorization,
    _corrupt,
    _scan_indices,
    _sha,
    _stratified_tiny_pool,
    _structure_classes,
    _verify_manifest_provenance,
)


def load_config(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    required = {
        "version",
        "source_config",
        "source_run_dir",
        "source_config_sha256",
        "source_checkpoint_sha256",
        "source_metrics_sha256",
        "source_tiny_overfit_sha256",
        "source_panel_sha256",
        "source_run_manifest_sha256",
        "source_class_manifest_sha256",
        "output_dir",
        "lengths",
        "seed",
        "updates",
        "evaluation_updates",
        "gate",
        "authorization",
    }
    if not isinstance(cfg, dict) or required - cfg.keys():
        missing = sorted(required - cfg.keys()) if isinstance(cfg, dict) else sorted(required)
        raise ValueError(f"diagnostic config missing keys: {missing}")
    if cfg["version"] != "e008_failure_diagnostic_v1" or cfg["updates"] != 1000:
        raise ValueError("unsupported E008 diagnostic config")
    run = Path(cfg["source_run_dir"])
    pins = (
        (cfg["source_config"], cfg["source_config_sha256"]),
        (run / "checkpoint.pt", cfg["source_checkpoint_sha256"]),
        (run / "training_metrics.jsonl", cfg["source_metrics_sha256"]),
        (run / "tiny_overfit_result.json", cfg["source_tiny_overfit_sha256"]),
    )
    for path, expected in pins:
        if _sha(path) != expected:
            raise ValueError(f"pinned E008 evidence hash mismatch: {path}")
    source = yaml.safe_load(Path(cfg["source_config"]).read_text())
    from scripts.run_e008_geometry_native_decoder import _validate_config_schema

    _validate_config_schema(source)
    if _sha(run / "panel_manifest.json") != cfg["source_panel_sha256"]:
        raise ValueError("source panel manifest hash mismatch")
    if _sha(run / "run_manifest.json") != cfg["source_run_manifest_sha256"]:
        raise ValueError("source run manifest hash mismatch")
    if _sha(source["real_structures"]["structure_class_manifest"]) != cfg["source_class_manifest_sha256"]:
        raise ValueError("source class manifest hash mismatch")
    cfg["source"] = source
    return cfg


def _aligned_errors(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = _kabsch_align(pred - pred.mean(0), target - target.mean(0))
    return torch.linalg.vector_norm(p - (target - target.mean(0)), dim=-1)


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    err = _aligned_errors(pred, target).detach().cpu().numpy()
    n = len(target)
    slope = float(np.polyfit(np.arange(n), err, 1)[0])
    bonds = torch.linalg.vector_norm(pred[1:] - pred[:-1], dim=-1)
    truth_bonds = torch.linalg.vector_norm(target[1:] - target[:-1], dim=-1)
    result = {
        "aligned_coordinate_rmse_angstrom": float(np.mean(err**2) ** 0.5),
        "adjacent_bond_rmse_angstrom": float(((bonds - truth_bonds) ** 2).mean().sqrt()),
        "i_plus_2_error_angstrom": float(
            (
                torch.linalg.vector_norm(pred[2:] - pred[:-2], dim=-1)
                - torch.linalg.vector_norm(target[2:] - target[:-2], dim=-1)
            )
            .square()
            .mean()
            .sqrt()
        ),
        "i_plus_3_error_angstrom": float(
            (
                torch.linalg.vector_norm(pred[3:] - pred[:-3], dim=-1)
                - torch.linalg.vector_norm(target[3:] - target[:-3], dim=-1)
            )
            .square()
            .mean()
            .sqrt()
        ),
        "per_residue_aligned_error_angstrom": err.tolist(),
        "normalized_position_quartile_rmse_angstrom": [
            float(np.mean(err[a:b] ** 2) ** 0.5)
            for a, b in ((0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n))
        ],
        "error_linear_slope_angstrom_per_residue": slope,
    }
    _, _, pt = cartesian_to_internal(pred)
    _, _, tt = cartesian_to_internal(target)
    result["chirality_agreement"] = float((torch.cos(pt - tt) > 0).float().mean())
    return result


def _dataset(cfg: dict, device: torch.device):
    source = cfg["source"]
    labels = _structure_classes(source["real_structures"]["structure_class_manifest"])
    auth = _authorization(source)
    _verify_manifest_provenance(auth, labels)
    dataset = E007CoordinateDataset(auth, split="train")
    candidates = _scan_indices(dataset, labels, {"globular", "unknown"})
    panel = json.loads((Path(cfg["source_run_dir"]) / "panel_manifest.json").read_text())
    allowed = set(panel["training_sample_ids"])
    candidates = [r for r in candidates if r[1] in allowed]
    tiny = _stratified_tiny_pool(candidates, source["seed"] + 3)
    calibration = source["corruption_calibration"]
    evals = json.loads(Path(calibration["observed_error_summary"]).read_text())["500"]["denoising"]["by_length_stratum"]
    strata = ((64, "20-64"), (128, "65-128"), (256, "129-256"), (384, "257-384"), (500, "385-500"))
    sigmas = {
        n: float(
            np.clip(
                evals[s]["adjacent_distance_error_angstrom"] / math.sqrt(6),
                calibration["minimum_sigma_angstrom"],
                calibration["maximum_sigma_angstrom"],
            )
        )
        for n, s in strata
    }
    gen = torch.Generator(device=device).manual_seed(source["seed"] + 11)
    rows = []
    for ix, sid, n, *_ in tiny:
        row = dataset[ix]
        target = row["coordinates"].to(device).float()
        sigma = sigmas[min(sigmas, key=lambda x: abs(x - n))]
        rows.append((sid, target, _corrupt(target, sigma, gen)))
    return rows


def evaluate_existing(cfg: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic execution requires CUDA")
    device = torch.device("cuda")
    rows = _dataset(cfg, device)
    state = torch.load(Path(cfg["source_run_dir"]) / "checkpoint.pt", map_location=device, weights_only=False)
    model = BackboneKinematicDecoder(**cfg["source"]["decoder"]).to(device).eval()
    model.load_state_dict(state["decoder"], strict=True)
    results = []
    with torch.no_grad():
        for sid, target, coarse in rows:
            mask = torch.ones((1, len(target)), device=device, dtype=torch.bool)
            decoded = model(coarse[None], mask)["coordinates"][0]
            bonds, angles, torsions = cartesian_to_internal(coarse)
            fixed = internal_to_cartesian(coarse[:3], angles, torsions, bond_length=3.8)
            _, ta, tt = cartesian_to_internal(target)
            oracle = internal_to_cartesian(
                target[:3], ta, tt, bond_lengths=torch.linalg.vector_norm(target[1:] - target[:-1], dim=-1)
            )
            torch.manual_seed(cfg["seed"])
            untrained = (
                BackboneKinematicDecoder(**cfg["source"]["decoder"])
                .to(device)
                .eval()(coarse[None], mask)["coordinates"][0]
            )
            results.append(
                {
                    "sample_id": sid,
                    "length": len(target),
                    "coarse": _metrics(coarse, target),
                    "fixed_bond": _metrics(fixed, target),
                    "untrained_decoder": _metrics(untrained, target),
                    "update_500_decoder": _metrics(decoded, target),
                    "oracle_roundtrip": _metrics(oracle, target),
                }
            )
    buckets = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
    by_length = {}
    for lo, hi in buckets:
        members = [r for r in results if lo <= r["length"] <= hi]
        by_length[f"{lo}-{hi}"] = {
            "structure_count": len(members),
            "update_500_mean_aligned_rmse_angstrom": (
                float(np.mean([r["update_500_decoder"]["aligned_coordinate_rmse_angstrom"] for r in members]))
                if members
                else None
            ),
        }
    out = {
        "status": "completed_non_authorizing",
        "checkpoint_sha256": cfg["source_checkpoint_sha256"],
        "update_500_by_length_stratum": by_length,
        "structures": results,
        "authorizes_training": False,
    }
    dest = Path(cfg["output_dir"]) / "existing_checkpoint_evaluation.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def overfit(cfg: dict, length: int) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic execution requires CUDA")
    device = torch.device("cuda")
    candidates = _dataset(cfg, device)
    pool = [r for r in candidates if len(r[1]) == length]
    if not pool:
        raise ValueError(f"no fixed overfit input at length {length}")
    sid, target, coarse = pool[0]
    target_bytes = target.detach().cpu().numpy().tobytes()
    coarse_bytes = coarse.detach().cpu().numpy().tobytes()
    torch.manual_seed(cfg["seed"] + length)
    model = BackboneKinematicDecoder(**cfg["source"]["decoder"]).to(device).train()
    frozen_prior = True
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["source"]["optimizer"]["learning_rate"],
        weight_decay=cfg["source"]["optimizer"]["weight_decay"],
    )
    mask = torch.ones((1, length), device=device, dtype=torch.bool)
    tb = target[None]
    cb = coarse[None]
    points = set(cfg["evaluation_updates"])
    records = []
    for step in range(cfg["updates"] + 1):
        if step in points:
            model.eval()
            with torch.no_grad():
                pred = model(cb, mask)["coordinates"][0]
                rec = {"update": step, **_metrics(pred, target), "finite": True}
            records.append(rec)
            model.train()
        if step == cfg["updates"]:
            break
        optim.zero_grad(set_to_none=True)
        pred = model(cb, mask)["coordinates"]
        ls = geometry_native_losses(pred, tb, cb, mask)
        w = cfg["source"]["loss_weights"]
        loss = sum(
            w[k] * ls[lk]
            for k, lk in (
                ("internal_angle_torsion", "internal"),
                ("kabsch_coordinate", "kabsch_coordinate"),
                ("i_plus_2", "i_plus_2"),
                ("i_plus_3", "i_plus_3"),
                ("long_range_pair", "long_range_pair"),
                ("contact_map", "contact_map"),
                ("chirality", "chirality"),
                ("radius_of_gyration", "radius_of_gyration"),
            )
        )
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("non-finite or missing decoder gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["source"]["optimizer"]["gradient_clip_norm"])
        optim.step()
        if (
            target.detach().cpu().numpy().tobytes() != target_bytes
            or coarse.detach().cpu().numpy().tobytes() != coarse_bytes
        ):
            raise RuntimeError("fixed diagnostic input mutated")
    result = {
        "status": "completed_non_authorizing",
        "length": length,
        "sample_id": sid,
        "fixed_corruption_sha256": hashlib.sha256(coarse_bytes).hexdigest(),
        "target_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "decoder_parameter_count": decoder_parameter_count(model),
        "frozen_generator": frozen_prior,
        "batch_size": 1,
        "records": records,
        "success": records[-1]["aligned_coordinate_rmse_angstrom"]
        <= cfg["gate"]["aligned_coordinate_rmse_angstrom_max"]
        and records[-1]["adjacent_bond_rmse_angstrom"] <= cfg["gate"]["adjacent_bond_rmse_angstrom_max"],
        "authorizes_2000_update_pilot": False,
    }
    dest = Path(cfg["output_dir"]) / f"overfit_length_{length}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/e008_failure_diagnostic_v1.yaml")
    g = p.add_mutually_exclusive_group(required=True)
    for x in ("plan-only", "evaluate-existing", "overfit-length-64", "overfit-length-500"):
        g.add_argument("--" + x, action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.plan_only:
        result = {
            "status": "plan_only_non_authorizing",
            "config_sha256": _sha(a.config),
            "fresh_paths": [
                cfg["output_dir"],
                str(Path(cfg["output_dir"]) / "existing_checkpoint_evaluation.json"),
                str(Path(cfg["output_dir"]) / "overfit_length_64.json"),
                str(Path(cfg["output_dir"]) / "overfit_length_500.json"),
            ],
            "commands": [
                f"python scripts/diagnose_e008_failure.py --config {a.config} --evaluate-existing",
                f"python scripts/diagnose_e008_failure.py --config {a.config} --overfit-length-64",
                f"python scripts/diagnose_e008_failure.py --config {a.config} --overfit-length-500",
            ],
            "estimated_runtime": "~10-30 min per GPU diagnostic; hardware dependent",
            "authorizes_2000_update_pilot": False,
        }
    elif a.evaluate_existing:
        result = evaluate_existing(cfg)
    else:
        result = overfit(cfg, 64 if a.overfit_length_64 else 500)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
