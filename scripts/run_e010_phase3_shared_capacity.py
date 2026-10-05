#!/usr/bin/env python3
"""Non-authorizing E010 capacity-only Phase 3 smokes and training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# E009's legacy helper imports datetime.UTC; retain Phase 2's metric implementation
# while supporting the CUDA host's Python 3.10 runtime.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protein_distance_diffusion.models.e010_global_equivariant import (  # noqa: E402 - CLI path bootstrap.
    GlobalEquivariantResidual,  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
)
from scripts.run_e010_phase2 import (  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
    _load_pairs,
    _metrics,
    file_sha,
)

CONFIG = ROOT / "configs/e010_phase3_shared_capacity.yaml"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_inputs(cfg):
    protocol_path = ROOT / cfg["source_phase2_protocol"]
    baseline_path = ROOT / cfg["source_phase2_baseline"]
    manifest_path = ROOT / cfg["source_phase2_manifest"]
    checkpoint_path = ROOT / cfg["source_phase2_update6400_checkpoint"]
    baseline_sha = file_sha(baseline_path)
    if baseline_sha != cfg["source_result_sha256"]:
        raise ValueError("existing Phase 2 v2 baseline result hash mismatch; refusing Phase 3")
    baseline = json.loads(baseline_path.read_text())
    if (
        baseline.get("status") != "completed"
        or baseline.get("start_update") != 2000
        or baseline.get("end_update") != 6400
    ):
        raise ValueError("Phase 2 v2 baseline is not the expected completed 2000-to-6400 run")
    manifest = json.loads(manifest_path.read_text())
    rec = next((x for x in manifest["checkpoints"] if x["update"] == 6400), None)
    checkpoint_sha = file_sha(checkpoint_path)
    if not rec or rec["sha256"] != checkpoint_sha:
        raise ValueError("update-6400 checkpoint does not match the existing manifest")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("panel_count") != 32 or protocol.get("authorizes_downstream") is not False:
        raise ValueError("Phase 2 fixed panel protocol is missing, changed, or authorizing")
    # Validate all fixed panel archive and tensor pins before a smoke or training run.
    archive_pins = []
    for item in protocol["panel"]:
        path = ROOT / item["archive"]
        observed = file_sha(path)
        if observed != item["archive_sha256"]:
            raise ValueError(f"Phase 2 corruption archive hash mismatch: {item['sample_id']}")
        with np.load(path, allow_pickle=False) as z:
            hashes = {
                "target_sha256": sha_bytes(np.ascontiguousarray(z["target"].astype("<f4", copy=False)).tobytes()),
                "corruption_sha256": sha_bytes(np.ascontiguousarray(z["coarse"].astype("<f4", copy=False)).tobytes()),
                "mask_sha256": sha_bytes(np.ascontiguousarray(z["mask"].astype(np.bool_, copy=False)).tobytes()),
            }
            meta = json.loads(str(z["metadata"].item()))
            if any(hashes[k] != item[k] or meta[k] != item[k] for k in hashes):
                raise ValueError(f"Phase 2 corruption tensor hash mismatch: {item['sample_id']}")
            if meta["split"] != "train" or meta["sample_id"] != item["sample_id"]:
                raise ValueError(f"Phase 2 panel identity/split mismatch: {item['sample_id']}")
        archive_pins.append(
            {
                "sample_id": item["sample_id"],
                "length": item["length"],
                "stratum": item["stratum"],
                "archive_sha256": observed,
                **hashes,
            }
        )
    baseline_end = next((r for r in baseline["records"] if r.get("update") == 6400), None)
    if baseline_end is None:
        raise ValueError("Phase 2 baseline lacks update-6400 evaluation; refusing rerun")
    v1_result = (
        ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase2_v1_prepared/training_result.json"
    )
    if json.loads(v1_result.read_text()).get("status") != "failed":
        raise ValueError("Phase 2 v1 failed status was not preserved")
    return (
        protocol,
        baseline_end,
        {
            "authorizes_downstream": False,
            "v1_status_preserved": "failed_under_predeclared_2000_update_gate",
            "v1_result_sha256": file_sha(v1_result),
            "phase2_protocol_sha256": file_sha(protocol_path),
            "phase2_v2_baseline_sha256": baseline_sha,
            "phase2_v2_manifest_sha256": file_sha(manifest_path),
            "phase2_v2_update6400_checkpoint_sha256": checkpoint_sha,
            "model_source_sha256": file_sha(ROOT / cfg["source_model"]),
            "phase2_panel_count": len(archive_pins),
            "phase2_corruption_pins": archive_pins,
            "baseline_update6400": {
                k: baseline_end[k]
                for k in (
                    "mean_aligned_rmse_angstrom",
                    "median_aligned_rmse_angstrom",
                    "maximum_aligned_rmse_angstrom",
                    "error_vs_length_slope_angstrom_per_residue",
                    "non_finite_outputs",
                )
            },
        },
    )


def device_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_gradients(model):
    vals = [p.grad for p in model.parameters() if p.grad is not None]
    return bool(vals) and all(bool(torch.isfinite(g).all()) for g in vals)


def smoke_variant(name, spec, cfg, device):
    device_seed(cfg["seed"])
    model = GlobalEquivariantResidual(**spec["model"]).to(device)
    count = sum(p.numel() for p in model.parameters())
    n = cfg["smoke"]["length"]
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 500)
    coords = torch.randn((1, n, 3), generator=gen, device=device)
    target = coords + 0.35 * torch.randn((1, n, 3), generator=gen, device=device)
    mask = torch.ones((1, n), dtype=torch.bool, device=device)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    out = model(coords, mask)
    finite_outputs = bool(torch.isfinite(out["prediction"]).all() and torch.isfinite(out["delta"]).all())
    loss = (out["prediction"] - target).square().mean() + 1e-5 * out["delta"].square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    finite_grads = finite_gradients(model)
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    alloc_limit = cfg["smoke"]["memory_allocated_limit_mib"] * 1024 * 1024
    reserv_limit = cfg["smoke"]["memory_reserved_limit_mib"] * 1024 * 1024
    passed = finite_outputs and finite_grads and allocated < alloc_limit and reserved < reserv_limit
    record = {
        "variant": name,
        "model": spec["model"],
        "parameter_count": count,
        "length": n,
        "finite_outputs": finite_outputs,
        "finite_gradients": finite_grads,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "allocated_limit_bytes": alloc_limit,
        "reserved_limit_bytes": reserv_limit,
        "loss": float(loss.detach().item()),
        "runtime_seconds": runtime,
        "device": torch.cuda.get_device_name(device),
        "status": "passed" if passed else "failed",
    }
    del model, coords, target, mask, out, loss
    torch.cuda.empty_cache()
    return record


def make_boundary(
    model,
    pairs,
    protocol,
    device,
    update,
    exposure,
    loss_history,
    grads,
    steps,
    clipped,
    started,
    count,
    baseline_metrics,
):
    rec = _metrics(model, pairs, device, update)
    by_id = {x["sample_id"]: x for x in rec["per_structure"]}
    strata = {}
    for item in protocol["panel"]:
        strata.setdefault(item["stratum"], []).append(by_id[item["sample_id"]]["aligned_rmse_angstrom"])
    rec["by_length_stratum"] = {
        name: {
            "count": len(v),
            "mean_aligned_rmse_angstrom": float(np.mean(v)),
            "median_aligned_rmse_angstrom": float(np.median(v)),
            "maximum_aligned_rmse_angstrom": float(np.max(v)),
        }
        for name, v in sorted(strata.items())
    }
    rec["per_identity_trajectory_point"] = [
        {
            "sample_id": x["sample_id"],
            "length": x["length"],
            "aligned_rmse_angstrom": x["aligned_rmse_angstrom"],
            "finite": x["finite"],
            "geometry_telemetry": x["geometry_telemetry"],
        }
        for x in rec["per_structure"]
    ]
    n = len(loss_history)
    tail_start = max(0, n - math.ceil(0.20 * update)) if update else 0
    xs = np.arange(tail_start + 1, n + 1, dtype=float)
    tail = np.asarray(loss_history[tail_start:], dtype=float)
    rec["training_loss_last_20_percent"] = {
        "first_update": tail_start + 1 if update else None,
        "last_update": update,
        "slope_loss_per_update": float(np.polyfit(xs, tail, 1)[0]) if len(tail) > 1 else None,
        "first_loss": float(tail[0]) if len(tail) else None,
        "last_loss": float(tail[-1]) if len(tail) else None,
    }
    rec["optimization_norms_since_previous_boundary"] = {
        "updates": len(grads),
        "gradient_l2_mean": float(np.mean(grads)) if grads else None,
        "gradient_l2_max": float(np.max(grads)) if grads else None,
        "optimizer_step_l2_mean": float(np.mean(steps)) if steps else None,
        "optimizer_step_l2_max": float(np.max(steps)) if steps else None,
        "clipping_fraction": float(np.mean(clipped)) if clipped else None,
    }
    exposures = list(exposure.values())
    rec["identity_exposures"] = {
        "minimum": min(exposures),
        "mean": float(np.mean(exposures)),
        "maximum": max(exposures),
        "per_identity": dict(exposure),
    }
    rec["elapsed_runtime_seconds"] = time.perf_counter() - started
    rec["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    rec["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    if update == 6400:
        added_m = (count - 1_836_640) / 1_000_000
        rec["improvement_per_million_added_parameters"] = {
            "baseline_phase2_v2_mean_rmse_angstrom": baseline_metrics["mean_aligned_rmse_angstrom"],
            "variant_mean_rmse_improvement_angstrom": baseline_metrics["mean_aligned_rmse_angstrom"]
            - rec["mean_aligned_rmse_angstrom"],
            "added_parameters_millions_vs_phase2_baseline": added_m,
            "mean_rmse_improvement_angstrom_per_million_added_parameters": (
                baseline_metrics["mean_aligned_rmse_angstrom"] - rec["mean_aligned_rmse_angstrom"]
            )
            / added_m,
        }
    return rec


def train_variant(name, spec, cfg, protocol, baseline_metrics, device, out):
    output = out / f"{name}_training_result.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Phase 3 result: {output}")
    smoke_path = out / "cuda_smokes.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("run and record both capacity CUDA smokes before training")
    smokes = json.loads(smoke_path.read_text())
    smoke = next((x for x in smokes["variants"] if x["variant"] == name), None)
    if not smoke or smoke["status"] != "passed":
        raise RuntimeError(f"{name} did not pass the length-500 CUDA smoke; refusing training")
    # Ensure no input artifact changed between smoke and training.
    protocol2, _, audit = verify_inputs(cfg)
    if protocol2 != protocol:
        raise ValueError("Phase 2 panel changed between smoke and training")
    device_seed(cfg["seed"])
    model = GlobalEquivariantResidual(**spec["model"]).to(device)
    count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
    pairs = _load_pairs(ROOT / cfg["source_phase2_protocol"].rsplit("/", 1)[0])
    ids = [p["metadata"]["sample_id"] for p in pairs]
    exposures = {sid: 0 for sid in ids}
    order = list(range(len(pairs)))
    rng = random.Random(cfg["seed"])
    rng.shuffle(order)
    boundaries = set(cfg["training"]["evaluation_updates"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    loss_history, records = [], []
    grad_window, step_window, clip_window = [], [], []
    records.append(
        make_boundary(
            model,
            pairs,
            protocol,
            device,
            0,
            exposures,
            loss_history,
            grad_window,
            step_window,
            clip_window,
            started,
            count,
            baseline_metrics,
        )
    )
    for step in range(cfg["training"]["updates"]):
        if step % len(order) == 0 and step > 0:
            rng.shuffle(order)
        idx = order[step % len(order)]
        pair = pairs[idx]
        target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
        opt.zero_grad(set_to_none=True)
        result = model(coarse[None], mask[None])
        valid = mask[:, None].to(target.dtype)
        coord = ((result["prediction"][0] - target).square() * valid).sum() / (3 * mask.sum())
        residual = (result["delta"][0].square() * valid).sum() / (3 * mask.sum())
        loss = coord + 1e-5 * residual
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{name}: non-finite loss at update {step + 1}")
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_max_norm"]).item()
        )
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"{name}: non-finite gradient norm at update {step + 1}")
        clipped = grad_norm > cfg["training"]["gradient_clip_max_norm"]
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        parts = [(p.detach() - old).square().sum() for old, p in zip(before, model.parameters(), strict=True)]
        step_norm = float(torch.stack(parts).sum().sqrt().item())
        del before, parts
        update = step + 1
        exposures[pair["metadata"]["sample_id"]] += 1
        loss_history.append(float(loss.detach().item()))
        grad_window.append(grad_norm)
        step_window.append(step_norm)
        clip_window.append(clipped)
        if update in boundaries:
            rec = make_boundary(
                model,
                pairs,
                protocol,
                device,
                update,
                exposures,
                loss_history,
                grad_window,
                step_window,
                clip_window,
                started,
                count,
                baseline_metrics,
            )
            records.append(rec)
            grad_window, step_window, clip_window = [], [], []
    if any(x != 200 for x in exposures.values()):
        raise ValueError(f"{name}: expected exactly 200 exposures per identity")
    gates = cfg["acceptance"]
    final = records[-1]
    passed = (
        final["mean_aligned_rmse_angstrom"] <= gates["mean_aligned_rmse_angstrom_max"]
        and final["median_aligned_rmse_angstrom"] <= gates["median_aligned_rmse_angstrom_max"]
        and final["maximum_aligned_rmse_angstrom"] <= gates["maximum_aligned_rmse_angstrom_max"]
        and final["error_vs_length_slope_angstrom_per_residue"]
        <= gates["error_vs_length_slope_angstrom_per_residue_max"]
        and final["non_finite_outputs"] == 0
    )
    result = {
        "schema": "e010_phase3_capacity_training_result_v1",
        "authorizes_downstream": False,
        "variant": name,
        "status": "passed" if passed else "failed",
        "parameter_count": count,
        "updates": cfg["training"]["updates"],
        "runtime_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
        "records": records,
        "training_loss_by_update": loss_history,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "objective": cfg["training"]["objective"],
        "optimizer": "AdamW lr=3e-4 weight_decay=0; no scheduler",
        "uniform_identity_schedule": "seeded random permutation reshuffled at each 32-update epoch",
        "acceptance": gates,
        "input_audit": audit,
        "pilot_prepared": False,
        "restrictions": cfg["restrictions"],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    torch.save(
        {
            "model": model.state_dict(),
            "variant": name,
            "parameter_count": count,
            "config_sha256": file_sha(CONFIG),
            "panel_protocol_sha256": file_sha(ROOT / cfg["source_phase2_protocol"]),
        },
        out / f"{name}_model_final.pt",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-all", action="store_true")
    modes.add_argument("--train", choices=("medium", "large"))
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    out = ROOT / cfg["output_dir"]
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requires CUDA; no training artifacts were written")
    protocol, baseline_end, input_audit = verify_inputs(cfg)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.smoke_all:
        out.mkdir(parents=True, exist_ok=True)
        output = out / "cuda_smokes.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite smoke results: {output}")
        variants = [smoke_variant(name, cfg["variants"][name], cfg, device) for name in ("medium", "large")]
        result = {
            "schema": "e010_phase3_capacity_cuda_smokes_v1",
            "authorizes_downstream": False,
            "input_audit": input_audit,
            "variants": variants,
            "training_authorized_by_smoke": {x["variant"]: x["status"] == "passed" for x in variants},
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2))
    else:
        if not out.is_dir():
            raise FileNotFoundError("run --smoke-all before training any capacity variant")
        result = train_variant(args.train, cfg["variants"][args.train], cfg, protocol, baseline_end, device, out)
        print(
            json.dumps(
                {
                    "variant": args.train,
                    "status": result["status"],
                    "parameter_count": result["parameter_count"],
                    "runtime_seconds": result["runtime_seconds"],
                    "final_metrics": result["records"][-1],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
