#!/usr/bin/env python3
"""Prepare and execute the isolated E010 multi-structure coordinate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from scripts.run_e009_bayesian_refiner import _calibrated_sigma, _kabsch_rmse, _make_corruption
from scripts.run_e009_bayesian_refiner import load_config as load_e009

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e010_global_equivariant_phase2_v1.yaml"
E009_CONFIG = ROOT / "configs/e009_bayesian_refiner_v4_execution.yaml"


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x: np.ndarray) -> str:
    return digest_bytes(np.ascontiguousarray(x.astype("<f4", copy=False)).tobytes())


def _panel_exclusions() -> tuple[set[str], dict[str, list[str]]]:
    excluded: set[str] = {"10px_18"}
    sources: dict[str, list[str]] = {"e010_v1": ["10px_18"]}
    markers = ("development", "prospective", "validation", "test", "holdout")
    for path in (ROOT / "reports/experiments").rglob("*.json"):
        if not any(x in path.name.lower() for x in ("panel", "manifest", "selection", "holdout")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        found: set[str] = set()

        def visit(node, exclusion_context=False, found=found):
            if isinstance(node, dict):
                for key, value in node.items():
                    key_lower = str(key).lower()
                    context = exclusion_context or any(marker in key_lower for marker in markers)
                    visit(value, context)
            elif isinstance(node, list):
                if exclusion_context:
                    found.update(str(x) for x in node if isinstance(x, str) and "_" in x)
                for item in node:
                    if isinstance(item, (dict, list)):
                        visit(item, exclusion_context)
            elif exclusion_context and isinstance(node, str) and "_" in node:
                found.add(node)

        visit(data)
        if found:
            excluded.update(found)
            sources[str(path.relative_to(ROOT))] = sorted(found)
    for path in (ROOT / "reports/experiments").rglob("*.csv"):
        if "manifest" not in path.name.lower():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "sample_id" not in frame.columns:
            continue
        context_cols = [c for c in frame.columns if any(m in c.lower() for m in markers)]
        if not context_cols:
            # E008 prototype manifest lists identities selected for its prior pilot.
            if "selection_class" not in frame.columns:
                continue
            mask = frame["selection_class"].astype(str).str.contains("prototype", case=False, na=False)
        else:
            mask = pd.Series(False, index=frame.index)
            for col in context_cols:
                mask |= (
                    frame[col]
                    .astype(str)
                    .str.lower()
                    .isin(("true", "1", "yes", "development", "prospective", "validation", "test", "holdout"))
                )
        ids = set(frame.loc[mask, "sample_id"].dropna().astype(str))
        if ids:
            excluded.update(ids)
            sources[str(path.relative_to(ROOT))] = sorted(ids)
    return excluded, sources


def prepare(cfg: dict) -> dict:
    out = ROOT / cfg["output_dir"]
    if out.exists():
        raise FileExistsError(f"Phase 2 output already exists; refusing overwrite: {out}")
    split_path = ROOT / cfg["panel"]["source_split"]
    frame = pd.read_parquet(split_path)
    frame = frame[(frame["split"] == "train") & frame["length"].between(1, 500)].copy()
    excluded, raw_exclusion_sources = _panel_exclusions()
    excluded.intersection_update(set(frame.sample_id.astype(str)))
    exclusion_sources = {
        name: {"identity_count": len(ids), "identity_sha256": digest_bytes("\n".join(sorted(ids)).encode())}
        for name, ids in raw_exclusion_sources.items()
    }
    frame = frame[~frame.sample_id.astype(str).isin(excluded)].copy()
    frame["selection_rank"] = frame.sample_id.map(lambda sid: digest_bytes(f"e010_phase2_v1|{sid}".encode()))
    chosen = []
    candidate_rejections = []
    for stratum in cfg["panel"]["strata"]:
        candidates = frame[frame.length.between(stratum["min"], stratum["max"])].sort_values(
            ["selection_rank", "sample_id"]
        )
        accepted = 0
        for _, row in candidates.iterrows():
            source = (ROOT / str(row.path)).resolve()
            try:
                with np.load(source, allow_pickle=False) as z:
                    coords = np.asarray(z["ca_coordinates"], dtype=np.float32)
                    mask = np.asarray(z["residue_mask"], dtype=np.bool_)
                    stored_id = str(z["sample_id"].item())
                n = int(row.length)
                if (
                    stored_id != str(row.sample_id)
                    or coords.shape != (n, 3)
                    or mask.shape != (n,)
                    or not mask.all()
                    or not np.isfinite(coords).all()
                ):
                    raise ValueError("source identity, shape, mask, or finiteness mismatch")
            except Exception as exc:
                candidate_rejections.append({"sample_id": str(row.sample_id), "reason": str(exc)})
                continue
            chosen.append((stratum, row))
            accepted += 1
            if accepted == stratum["count"]:
                break
        if accepted < stratum["count"]:
            raise ValueError(f"insufficient eligible train-only structures for {stratum['name']}: selected {accepted}")
    e009 = load_e009(E009_CONFIG)
    out.mkdir(parents=True)
    records = []
    for panel_rank, (stratum, row) in enumerate(chosen):
        source = (ROOT / str(row.path)).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as z:
            coords = np.asarray(z["ca_coordinates"], dtype=np.float32)
            mask = np.asarray(z["residue_mask"], dtype=np.bool_)
        n = int(row.length)
        if coords.shape != (n, 3) or mask.shape != (n,) or not mask.all() or not np.isfinite(coords).all():
            raise ValueError(f"invalid complete CA tensor for {row.sample_id}")
        target = np.ascontiguousarray(coords - coords.mean(0, keepdims=True), dtype=np.float32)
        sigma = _calibrated_sigma(e009, n)
        seed = 8015 + 900000 + 1009 * panel_rank
        coarse = _make_corruption(torch.from_numpy(target), sigma, seed).numpy().astype(np.float32, copy=False)
        mask = np.ones((n,), dtype=np.bool_)
        archive = out / f"pair_{panel_rank:02d}_{row.sample_id}.npz"
        metadata = {
            "schema": "e010_phase2_fixed_corruption_v1",
            "sample_id": str(row.sample_id),
            "length": n,
            "split": "train",
            "stratum": stratum["name"],
            "panel_rank": panel_rank,
            "source_npz": str(source.relative_to(ROOT)),
            "source_npz_sha256": file_sha(source),
            "calibration_summary_sha256": file_sha(ROOT / e009["corruption"]["calibration_summary"]),
            "coordinate_noise_sigma_angstrom": sigma,
            "seed": seed,
            "construction": "E009 independent centered Gaussian noise plus centered cumulative drift (0.12 sigma RMS)",
            "target_sha256": tensor_sha(target),
            "corruption_sha256": tensor_sha(coarse),
            "mask_sha256": digest_bytes(np.ascontiguousarray(mask).tobytes()),
        }
        with archive.open("wb") as f:
            np.savez_compressed(
                f, target=target, coarse=coarse, mask=mask, metadata=np.asarray(json.dumps(metadata, sort_keys=True))
            )
        records.append({**metadata, "archive": str(archive.relative_to(ROOT)), "archive_sha256": file_sha(archive)})
    protocol = {
        "schema": "e010_phase2_protocol_v1",
        "authorizes_downstream": False,
        "config_sha256": file_sha(CONFIG),
        "source_split": str(split_path.relative_to(ROOT)),
        "source_split_sha256": file_sha(split_path),
        "selection_algorithm": cfg["panel"]["ranking"],
        "exclusion_sources": exclusion_sources,
        "excluded_identity_count": len(excluded),
        "excluded_train_identity_sha256": digest_bytes("\n".join(sorted(excluded)).encode()),
        "panel_count": len(records),
        "panel": records,
        "ranked_candidate_rejections_before_selection": candidate_rejections,
        "fixed_boundaries": cfg["training"]["evaluation_updates"],
        "acceptance": cfg["acceptance"],
        "objective": "masked xyz coordinate MSE + 1e-5 masked residual L2",
        "restrictions": {
            k: cfg["training"][k]
            for k in (
                "sampling",
                "fixed_bond_constraints",
                "e009_geometry_priors",
                "local_geometry_losses",
                "prospective_panel",
            )
        },
        "v1_artifacts_modified": False,
        "conditional_supervised_pilot": (
            "prepare only after Phase 2 passes; frozen E007 denoiser outputs on "
            "corrupted real structures paired with known targets; do not execute "
            "automatically"
        ),
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol


def _new_model(cfg, device):
    return GlobalEquivariantResidual(**cfg["model"]).to(device)


def cuda_lifecycle(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA lifecycle is pending: torch.cuda.is_available() is false")
    device = torch.device("cuda")
    records = []
    for n in cfg["cuda_lifecycle"]["lengths"]:
        torch.manual_seed(cfg["seed"] + n)
        torch.cuda.manual_seed_all(cfg["seed"] + n)
        model = _new_model(cfg, device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=0.0)
        coords = torch.randn((1, n, 3), device=device)
        target = coords + 0.35 * torch.randn_like(coords)
        mask = torch.ones((1, n), device=device, dtype=torch.bool)
        samples = []
        for step in range(cfg["cuda_lifecycle"]["warmup_steps"] + cfg["cuda_lifecycle"]["measured_steps"]):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            result = model(coords, mask)
            torch.cuda.synchronize(device)
            tf = time.perf_counter()
            loss = (result["prediction"] - target).square().mean() + 1e-5 * result["delta"].square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            tb = time.perf_counter()
            if step >= cfg["cuda_lifecycle"]["warmup_steps"]:
                samples.append(
                    {"forward_seconds": tf - t0, "backward_seconds": tb - tf, "forward_backward_seconds": tb - t0}
                )
        records.append(
            {
                "length": n,
                "batch_size": 1,
                "steps": samples,
                "mean_forward_seconds": float(np.mean([x["forward_seconds"] for x in samples])),
                "mean_backward_seconds": float(np.mean([x["backward_seconds"] for x in samples])),
                "mean_forward_backward_seconds": float(np.mean([x["forward_backward_seconds"] for x in samples])),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "allocated_bytes_after_step": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes_after_step": int(torch.cuda.memory_reserved(device)),
                "device": torch.cuda.get_device_name(device),
            }
        )
        del model, opt, coords, target, mask
        torch.cuda.empty_cache()
    return {"schema": "e010_phase2_cuda_lifecycle_v1", "authorizes_downstream": False, "records": records}


def _load_pairs(out):
    protocol = json.loads((out / "protocol.json").read_text())
    pairs = []
    for item in protocol["panel"]:
        with np.load(ROOT / item["archive"], allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            if (
                meta["target_sha256"] != item["target_sha256"]
                or file_sha(ROOT / item["archive"]) != item["archive_sha256"]
            ):
                raise ValueError(f"pair archive hash mismatch: {item['sample_id']}")
            pairs.append(
                {
                    "target": torch.from_numpy(z["target"].copy()),
                    "coarse": torch.from_numpy(z["coarse"].copy()),
                    "mask": torch.from_numpy(z["mask"].copy()).bool(),
                    "metadata": meta,
                }
            )
    return pairs


def _metrics(model, pairs, device, update):
    results = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
            prediction = model(coarse[None], mask[None])["prediction"][0]
            finite = bool(torch.isfinite(prediction).all())
            rmse = float(_kabsch_rmse(prediction[mask], target[mask])) if finite else float("inf")
            geo = None
            if finite:
                geo = {}
                for sep in (1, 2, 3):
                    pd = torch.linalg.vector_norm(prediction[sep:] - prediction[:-sep], dim=-1)
                    td = torch.linalg.vector_norm(target[sep:] - target[:-sep], dim=-1)
                    geo[f"i_plus_{sep}_distance_rmse_angstrom"] = float((pd - td).square().mean().sqrt())
                v1, v2, v3 = (
                    prediction[1:-2] - prediction[:-3],
                    prediction[2:-1] - prediction[1:-2],
                    prediction[3:] - prediction[2:-1],
                )
                t1, t2, t3 = target[1:-2] - target[:-3], target[2:-1] - target[1:-2], target[3:] - target[2:-1]
                pv = (torch.cross(v1, v2, dim=-1) * v3).sum(-1)
                tv = (torch.cross(t1, t2, dim=-1) * t3).sum(-1)
                geo["chirality_inversions"] = int((pv * tv < 0).sum())
            results.append(
                {
                    "sample_id": pair["metadata"]["sample_id"],
                    "length": len(target),
                    "aligned_rmse_angstrom": rmse,
                    "finite": finite,
                    "geometry_telemetry": geo,
                }
            )
    values = np.asarray([r["aligned_rmse_angstrom"] for r in results])
    lengths = np.asarray([r["length"] for r in results], dtype=float)
    slope = float(np.polyfit(lengths, values, 1)[0])
    return {
        "update": update,
        "per_structure": results,
        "mean_aligned_rmse_angstrom": float(np.mean(values)),
        "median_aligned_rmse_angstrom": float(np.median(values)),
        "maximum_aligned_rmse_angstrom": float(np.max(values)),
        "non_finite_outputs": int(sum(not r["finite"] for r in results)),
        "error_vs_length_slope_angstrom_per_residue": slope,
        "chirality_inversions_total": int(
            sum(r["geometry_telemetry"]["chirality_inversions"] for r in results if r["finite"])
        ),
        "local_distance_rmse_mean_angstrom": {
            f"i_plus_{k}": float(
                np.mean([r["geometry_telemetry"][f"i_plus_{k}_distance_rmse_angstrom"] for r in results if r["finite"]])
            )
            for k in (1, 2, 3)
        },
    }


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training is pending: torch.cuda.is_available() is false")
    out = ROOT / cfg["output_dir"]
    if (out / "training_result.json").exists():
        raise FileExistsError("training_result.json already exists; refusing overwrite")
    pairs = _load_pairs(out)
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = _new_model(cfg, device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"]
    )
    boundaries = set(cfg["training"]["evaluation_updates"])
    order = list(range(len(pairs)))
    rng = random.Random(cfg["seed"])
    rng.shuffle(order)
    records = [_metrics(model, pairs, device, 0)]
    started = time.perf_counter()
    for step in range(cfg["training"]["max_updates"]):
        idx = order[step % len(order)]
        if step % len(order) == 0 and step > 0:
            rng.shuffle(order)
        pair = pairs[idx]
        target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
        opt.zero_grad(set_to_none=True)
        result = model(coarse[None], mask[None])
        valid = mask[:, None].to(target.dtype)
        coord = ((result["prediction"][0] - target).square() * valid).sum() / (3 * mask.sum())
        residual = (result["delta"][0].square() * valid).sum() / (3 * mask.sum())
        loss = coord + 1e-5 * residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {step}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"non-finite gradient at update {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_max_norm"])
        opt.step()
        update = step + 1
        if update in boundaries:
            records.append(_metrics(model, pairs, device, update))
    final = records[-1]
    gates = cfg["acceptance"]
    passed = (
        final["mean_aligned_rmse_angstrom"] <= gates["mean_aligned_rmse_angstrom_max"]
        and final["median_aligned_rmse_angstrom"] <= gates["median_aligned_rmse_angstrom_max"]
        and final["maximum_aligned_rmse_angstrom"] <= gates["maximum_aligned_rmse_angstrom_max"]
        and final["non_finite_outputs"] == 0
        and final["error_vs_length_slope_angstrom_per_residue"]
        <= gates["positive_error_vs_length_slope_angstrom_per_residue_max"]
    )
    result = {
        "schema": "e010_phase2_training_result_v1",
        "authorizes_downstream": False,
        "status": "passed" if passed else "failed",
        "evaluation_split": "training_panel_only",
        "updates": cfg["training"]["max_updates"],
        "runtime_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "objective": "masked xyz coordinate MSE + 1e-5 masked residual L2",
        "optimizer": "AdamW lr=3e-4 weight_decay=0",
        "fixed_bond_constraints": False,
        "e009_geometry_priors": False,
        "local_geometry_losses": False,
        "sampling": False,
        "prospective_panel": False,
        "records": records,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "held_out_improvement_demonstrated": False,
        "conditional_supervised_pilot_prepared": False,
    }
    temp = out / "training_result.json.tmp"
    temp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temp, out / "training_result.json")
    torch.save(
        {
            "model": model.state_dict(),
            "config_sha256": file_sha(CONFIG),
            "panel_sha256": file_sha(out / "protocol.json"),
        },
        out / "phase2_model.pt",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cuda-lifecycle", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.cuda_lifecycle, args.train)) != 1:
        parser.error("select exactly one mode")
    cfg = yaml.safe_load(CONFIG.read_text())
    if args.prepare:
        result = prepare(cfg)
    elif args.cuda_lifecycle:
        out = ROOT / cfg["output_dir"]
        if not torch.cuda.is_available():
            result = {
                "schema": "e010_phase2_cuda_lifecycle_v1",
                "authorizes_downstream": False,
                "status": "pending_cuda_unavailable",
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": False,
                "records": [],
            }
        else:
            result = cuda_lifecycle(cfg)
        (out / "cuda_lifecycle.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        out = ROOT / cfg["output_dir"]
        if not torch.cuda.is_available():
            result = {
                "schema": "e010_phase2_training_result_v1",
                "authorizes_downstream": False,
                "status": "pending_cuda_unavailable",
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": False,
                "updates": 0,
                "records": [],
                "conditional_supervised_pilot_prepared": False,
            }
            (out / "training_status.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        else:
            result = train(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
