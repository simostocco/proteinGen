#!/usr/bin/env python3
"""Prepare the bounded E010 Phase 4A pilot; intentionally has no training mode."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protein_distance_diffusion.models.e010_global_equivariant import (  # noqa: E402 - CLI path bootstrap.
    GlobalEquivariantResidual,  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
)
from scripts.run_e010_phase2 import (  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
    _calibrated_sigma,
    _make_corruption,
)

CONFIG = ROOT / "configs/e010_phase4a_supervised_generalization.yaml"
PHASE2_DEVICE_SEED = 8015
EXPOSURE_BOUNDARIES = (5, 10, 20, 35, 50)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha_np(value: np.ndarray, dtype="<f4") -> str:
    return sha_bytes(np.ascontiguousarray(value.astype(dtype, copy=False)).tobytes())


def tensor_sha_torch(value: torch.Tensor) -> str:
    return sha_bytes(value.detach().contiguous().cpu().numpy().tobytes())


def canonical_sha(value) -> str:
    def normalize(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            return {"__tensor__": True, "dtype": str(t.dtype), "shape": list(t.shape), "sha256": tensor_sha_torch(t)}
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            return {
                "__ndarray__": True,
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "sha256": sha_bytes(a.tobytes()),
            }
        if isinstance(x, dict):
            return {str(k): normalize(v) for k, v in sorted(x.items(), key=lambda p: str(p[0]))}
        if isinstance(x, (list, tuple)):
            return [normalize(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return {"__repr__": repr(x), "__type__": type(x).__qualname__}

    raw = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha_bytes(raw)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prepared Phase 4A artifact: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def replace_json(path: Path, value):
    """Atomically advance a new Phase 4A planning artifact through its stages."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_config():
    return yaml.safe_load(CONFIG.read_text())


def phase3_pins(cfg):
    closeout_path = ROOT / cfg["phase3_closeout"]
    selection_path = ROOT / cfg["phase3_selection"]
    result_path = ROOT / cfg["phase3_large_result"]
    model_path = ROOT / cfg["phase3_large_model"]
    closeout = json.loads(closeout_path.read_text())
    selection = json.loads(selection_path.read_text())
    result = json.loads(result_path.read_text())
    if closeout.get("status") != "passed_shared_capacity_selection" or closeout.get("selected_variant") != "large":
        raise ValueError("Phase 3 is not closed with the expected Large selection")
    if (
        selection.get("selected_variant") != "large"
        or selection.get("variants", {}).get("large", {}).get("status") != "passed"
    ):
        raise ValueError("Phase 3 selection record does not pass for Large")
    if result.get("parameter_count") != 12844352 or result.get("status") != "passed":
        raise ValueError("selected Phase 3 Large result/config mismatch")
    if closeout.get("selected_training_result_sha256") != file_sha(result_path):
        raise ValueError("Phase 3 closeout result hash mismatch")
    if closeout.get("selected_final_model_sha256") != file_sha(model_path):
        raise ValueError("Phase 3 closeout model hash mismatch")
    return {
        "phase3_closeout_sha256": file_sha(closeout_path),
        "phase3_selection_sha256": file_sha(selection_path),
        "phase3_large_result_sha256": file_sha(result_path),
        "phase3_large_model_sha256": file_sha(model_path),
        "selected_variant": "large",
        "parameter_count": result["parameter_count"],
        "phase3_training_weights_loaded": False,
    }


def read_authorized_split(path: Path, expected_split: str):
    # Only the explicitly permitted train and validation parquet files are passed here.
    table = pq.read_table(path, columns=["sample_id", "length", "path", "split"])
    columns = table.to_pydict()
    rows = []
    seen = set()
    for sid, length, source_path, split in zip(
        columns["sample_id"], columns["length"], columns["path"], columns["split"], strict=True
    ):
        sid = str(sid)
        if sid in seen:
            raise ValueError(f"duplicate {expected_split} identity in authorized split file: {sid}")
        seen.add(sid)
        if str(split) != expected_split:
            raise ValueError(f"authorized split file label mismatch for {sid}: {split}")
        if 20 <= int(length) <= 500:
            rows.append(
                {"sample_id": sid, "length": int(length), "source_path": str(source_path), "split": expected_split}
            )
    return rows


def rank_row(namespace: str, split_name: str, sid: str) -> str:
    return sha_bytes(f"{namespace}|{split_name}|{sid}".encode())


def plan_only(cfg):
    out = ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = ROOT / cfg["train_split"], ROOT / cfg["development_split"]
    train_rows = read_authorized_split(train_path, "train")
    dev_rows = read_authorized_split(dev_path, "validation")
    train_ids, dev_ids = {x["sample_id"] for x in train_rows}, {x["sample_id"] for x in dev_rows}
    overlap = train_ids & dev_ids
    if overlap:
        raise ValueError(f"training/development identities overlap ({len(overlap)})")
    train_count_cfg = cfg["selection"]["training_counts"]
    dev_per = cfg["selection"]["development_count_per_stratum"]
    candidate_train, candidate_dev, counts = {}, {}, {}
    for stratum in cfg["selection"]["strata"]:
        lo, hi, name = stratum["min"], stratum["max"], stratum["name"]
        train_candidates = [r for r in train_rows if lo <= r["length"] <= hi]
        dev_candidates = [r for r in dev_rows if lo <= r["length"] <= hi]
        for r in train_candidates:
            r["selection_rank_sha256"] = rank_row(cfg["selection"]["namespace"], "train", r["sample_id"])
        for r in dev_candidates:
            r["selection_rank_sha256"] = rank_row(cfg["selection"]["namespace"], "development", r["sample_id"])
        train_candidates.sort(key=lambda r: (r["selection_rank_sha256"], r["sample_id"]))
        dev_candidates.sort(key=lambda r: (r["selection_rank_sha256"], r["sample_id"]))
        n_train = int(train_count_cfg[name])
        if len(train_candidates) < n_train or len(dev_candidates) < dev_per:
            raise ValueError(f"insufficient authorized identities in stratum {name}")
        candidate_train[name] = [{**row, "stratum": name} for row in train_candidates]
        candidate_dev[name] = [{**row, "stratum": name} for row in dev_candidates]
        counts[name] = {
            "available_train": len(train_candidates),
            "selected_train": n_train,
            "available_development": len(dev_candidates),
            "selected_development": dev_per,
        }
    candidate_pool = {
        "schema": "e010_phase4a_ranked_candidate_pool_v1",
        "authorizes_downstream": False,
        "selection_namespace": cfg["selection"]["namespace"],
        "ranking": cfg["selection"]["rank"],
        "train": candidate_train,
        "development": candidate_dev,
        "prospective_split_accessed": False,
    }
    candidate_path = out / "ranked_candidate_pool.json"
    if candidate_path.exists():
        raise FileExistsError(f"refusing to overwrite ranked candidate pool: {candidate_path}")
    write_json(candidate_path, candidate_pool)
    # Whole-file hashes pin only the two permitted split files.
    plan = {
        "schema": "e010_phase4a_plan_v1",
        "authorizes_downstream": False,
        "phase3": phase3_pins(cfg),
        "selection_namespace": cfg["selection"]["namespace"],
        "ranking": cfg["selection"]["rank"],
        "planning_status": "metadata_ranked_candidates_pending_readonly_validation",
        "candidate_pool_path": str(candidate_path.relative_to(ROOT)),
        "candidate_pool_sha256": file_sha(candidate_path),
        "authorized_split_files": {
            "train": {"path": cfg["train_split"], "sha256": file_sha(train_path), "label": "train"},
            "development": {"path": cfg["development_split"], "sha256": file_sha(dev_path), "label": "validation"},
        },
        "requested_stratum_counts": counts,
        "train_count": 2048,
        "development_count": 320,
        "train_development_identity_overlap": 0,
        "prospective_split_accessed": False,
        "training_started": False,
    }
    replace_json(out / "phase4a_plan.json", plan)
    return {
        "plan_path": str((out / "phase4a_plan.json").relative_to(ROOT)),
        "train_count": 2048,
        "development_count": 320,
        "stratum_counts": counts,
        "candidate_pool_sha256": file_sha(candidate_path),
    }


def load_plan(cfg):
    p = ROOT / cfg["output_dir"] / "phase4a_plan.json"
    plan = json.loads(p.read_text())
    if plan.get("authorizes_downstream") is not False or plan.get("prospective_split_accessed") is not False:
        raise ValueError("invalid or authorizing Phase 4A plan")
    if plan.get("planning_status") != "validated_metadata_order_selection":
        raise ValueError("read-only input validation has not finalized the Phase 4A identities")
    if len(plan["training"]) != 2048 or len(plan["development"]) != 320:
        raise ValueError("unexpected Phase 4A panel sizes")
    return plan


def load_source_arrays(row):
    source = (ROOT / row["source_path"]).resolve()
    if not source.is_file() or ROOT.resolve() not in source.parents:
        raise FileNotFoundError(f"authorized source structure missing or outside root: {row['sample_id']}")
    with np.load(source, allow_pickle=False) as z:
        coords = np.asarray(z["ca_coordinates"], dtype=np.float32)
        mask = np.asarray(z["residue_mask"], dtype=np.bool_)
        stored_id = str(z["sample_id"].item())
    n = row["length"]
    if (
        stored_id != row["sample_id"]
        or coords.shape != (n, 3)
        or mask.shape != (n,)
        or not mask.all()
        or not np.isfinite(coords).all()
    ):
        raise ValueError(f"invalid structure arrays for authorized identity {row['sample_id']}")
    return source, coords, mask


def verify_phase2_corruption_process(cfg):
    protocol = json.loads((ROOT / cfg["phase2_protocol"]).read_text())
    e009 = yaml.safe_load((ROOT / cfg["e009_corruption_config"]).read_text())
    verified = []
    for item in protocol["panel"]:
        path = ROOT / item["archive"]
        if file_sha(path) != item["archive_sha256"]:
            raise ValueError(f"Phase 2 reference corruption archive changed: {item['sample_id']}")
        with np.load(path, allow_pickle=False) as z:
            target = torch.from_numpy(z["target"].copy())
            stored = z["coarse"].astype("<f4", copy=False)
        sigma = _calibrated_sigma(e009, int(item["length"]))
        rebuilt = _make_corruption(target, sigma, int(item["seed"])).numpy().astype("<f4", copy=False)
        rebuilt_sha = tensor_sha_np(rebuilt)
        if rebuilt_sha != item["corruption_sha256"] or tensor_sha_np(stored) != rebuilt_sha:
            raise ValueError(f"current corruption construction does not reproduce Phase 2: {item['sample_id']}")
        verified.append(
            {
                "sample_id": item["sample_id"],
                "archive_sha256": item["archive_sha256"],
                "corruption_sha256": rebuilt_sha,
                "sigma": sigma,
                "seed": int(item["seed"]),
            }
        )
    return {
        "reference_protocol_sha256": file_sha(ROOT / cfg["phase2_protocol"]),
        "e009_corruption_config_sha256": file_sha(ROOT / cfg["e009_corruption_config"]),
        "calibration_summary_sha256": file_sha(ROOT / e009["corruption"]["calibration_summary"]),
        "phase2_reference_count": len(verified),
        "exact_reconstruction_verified": True,
        "references": verified,
    }


def validate_inputs(cfg):
    out = ROOT / cfg["output_dir"]
    plan_path = out / "phase4a_plan.json"
    plan = json.loads(plan_path.read_text())
    pool_path = ROOT / plan["candidate_pool_path"]
    if file_sha(pool_path) != plan["candidate_pool_sha256"]:
        raise ValueError("ranked authorized candidate pool hash mismatch")
    pool = json.loads(pool_path.read_text())
    for name, path_key in (("train", "train_split"), ("development", "development_split")):
        expected = plan["authorized_split_files"][name]["sha256"]
        if file_sha(ROOT / cfg[path_key]) != expected:
            raise ValueError(f"authorized {name} split source changed after planning")
    seen = {"train": set(), "development": set()}
    accepted_rows = {"train": [], "development": []}
    records, rejections = [], []
    for split_key, expected_split in (("train", "train"), ("development", "validation")):
        required_total = 2048 if split_key == "train" else 320
        for stratum in cfg["selection"]["strata"]:
            name = stratum["name"]
            required = (
                int(cfg["selection"]["training_counts"][name])
                if split_key == "train"
                else int(cfg["selection"]["development_count_per_stratum"])
            )
            accepted_in_stratum = 0
            for row in pool[split_key][name]:
                if accepted_in_stratum == required:
                    break
                try:
                    _, coords, mask = load_source_arrays(row)
                except (OSError, ValueError, KeyError, EOFError) as exc:
                    rejections.append(
                        {
                            "split": split_key,
                            "sample_id": row["sample_id"],
                            "stratum": name,
                            "source_path": row["source_path"],
                            "reason": str(exc),
                        }
                    )
                    continue
                sid = row["sample_id"]
                if sid in seen[split_key]:
                    raise ValueError(f"duplicate selected identity {sid}")
                seen[split_key].add(sid)
                accepted = {**row, "split": expected_split}
                accepted_rows[split_key].append(accepted)
                accepted_in_stratum += 1
                records.append(
                    {
                        "split": split_key,
                        "sample_id": sid,
                        "length": row["length"],
                        "stratum": name,
                        "source_path": row["source_path"],
                        "source_sha256": file_sha(ROOT / row["source_path"]),
                        "target_sha256": tensor_sha_np(coords - coords.mean(0, keepdims=True)),
                        "mask_sha256": tensor_sha_np(mask, dtype=np.bool_),
                    }
                )
            if accepted_in_stratum != required:
                raise ValueError(
                    f"only {accepted_in_stratum}/{required} valid identities in {split_key} stratum {name}"
                )
        if len(accepted_rows[split_key]) != required_total:
            raise ValueError(f"selected {len(accepted_rows[split_key])}, expected {required_total}, for {split_key}")
    if seen["train"] & seen["development"]:
        raise ValueError("selected source payload identities overlap")
    corruption_check = verify_phase2_corruption_process(cfg)
    original_plan_sha = file_sha(plan_path)
    plan.update(
        {
            "planning_status": "validated_metadata_order_selection",
            "training": accepted_rows["train"],
            "development": accepted_rows["development"],
            "train_development_identity_overlap": 0,
            "selected_train_count": len(accepted_rows["train"]),
            "selected_development_count": len(accepted_rows["development"]),
            "readonly_rejected_candidate_count": len(rejections),
            "readonly_rejected_candidate_sha256": sha_bytes(
                json.dumps(rejections, sort_keys=True, separators=(",", ":")).encode()
            ),
            "metadata_plan_sha256": original_plan_sha,
        }
    )
    replace_json(plan_path, plan)
    report = {
        "schema": "e010_phase4a_readonly_input_validation_v1",
        "authorizes_downstream": False,
        "plan_sha256": file_sha(plan_path),
        "metadata_plan_sha256": original_plan_sha,
        "train_count": len(seen["train"]),
        "development_count": len(seen["development"]),
        "identity_overlap": 0,
        "source_payloads_finite_and_complete": True,
        "phase2_corruption_reconstruction": corruption_check,
        "readonly_rejected_candidate_count": len(rejections),
        "readonly_rejected_candidates": rejections,
        "selected_source_pins": records,
        "prospective_split_accessed": False,
        "training_started": False,
    }
    write_json(out / "input_validation.json", report)
    return {
        "status": "validated",
        "train_count": report["train_count"],
        "development_count": report["development_count"],
        "phase2_corruption_reconstruction_verified": True,
    }


def collate_synthetic(batch_size: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(41041 + batch_size)
    coords = torch.randn((batch_size, 500, 3), generator=gen, device=device)
    target = coords + 0.35 * torch.randn((batch_size, 500, 3), generator=gen, device=device)
    mask = torch.ones((batch_size, 500), dtype=torch.bool, device=device)
    return coords, target, mask


def smoke_one(batch_size: int, cfg, device):
    spec = {k: cfg["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")}
    torch.manual_seed(cfg["model"]["seed"])
    torch.cuda.manual_seed_all(cfg["model"]["seed"])
    model = GlobalEquivariantResidual(**spec).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training_plan"]["learning_rate"], weight_decay=cfg["training_plan"]["weight_decay"]
    )
    coords, target, mask = collate_synthetic(batch_size, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    status, error = "passed", None
    try:
        opt.zero_grad(set_to_none=True)
        result = model(coords, mask)
        outputs_finite = bool(torch.isfinite(result["prediction"]).all() and torch.isfinite(result["delta"]).all())
        per_item = (result["prediction"] - target).square().mean((1, 2)) + 1e-5 * result["delta"].square().mean((1, 2))
        loss = per_item.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        gradients_finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        opt.step()
        torch.cuda.synchronize(device)
        if not outputs_finite or not gradients_finite:
            status = "failed_finite_check"
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize(device)
        status, error = "out_of_memory", str(exc).splitlines()[0][:300]
        torch.cuda.empty_cache()
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    runtime = time.perf_counter() - start
    device_info = torch.cuda.get_device_properties(device)
    rec = {
        "batch_size": batch_size,
        "status": status,
        "error": error,
        "peak_cuda_allocated_bytes": peak_alloc,
        "peak_cuda_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": int(device_info.total_memory),
        "runtime_seconds": runtime,
        "finite_outputs": status == "passed",
        "finite_gradients": status == "passed",
        "smoke_includes_adamw_step": True,
    }
    del model, opt, coords, target, mask
    if "result" in locals():
        del result
    if "loss" in locals():
        del loss
    if "grads" in locals():
        del grads
    torch.cuda.empty_cache()
    return rec


def smoke_batches(cfg):
    out = ROOT / cfg["output_dir"]
    plan = load_plan(cfg)
    validation = json.loads((out / "input_validation.json").read_text())
    if validation.get("plan_sha256") != file_sha(out / "phase4a_plan.json") or plan.get("phase3") != phase3_pins(cfg):
        raise ValueError("Phase 4A panel or selected Phase 3 source pins changed before CUDA smoke")
    if validation.get("phase2_corruption_reconstruction", {}).get("exact_reconstruction_verified") is not True:
        raise ValueError("read-only input validation must pass before CUDA smoke")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4A batch smoke requires CUDA")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    records = []
    max_size = 0
    # Probe every integer up to 64 using exponential growth and a binary search
    # between the largest pass and first failure, stopping at CUDA OOM.
    _probes = [1]
    failure = None
    current = 1
    while current < 64:
        current *= 2
        rec = smoke_one(current, cfg, device)
        records.append(rec)
        if rec["status"] != "passed":
            failure = current
            break
        max_size = current
        if current == 1:
            max_size = 1
    if failure is not None:
        lo, hi = max_size + 1, failure - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            rec = smoke_one(mid, cfg, device)
            records.append(rec)
            if rec["status"] == "passed":
                max_size = mid
                lo = mid + 1
            else:
                hi = mid - 1
    elif max_size < 64:
        max_size = max((r["batch_size"] for r in records if r["status"] == "passed"), default=0)
    else:
        max_size = 64
    if max_size < 1:
        raise RuntimeError("Large architecture did not pass batch-size-1 length-500 smoke")
    result = {
        "schema": "e010_phase4a_length500_batch_smoke_v1",
        "authorizes_downstream": False,
        "phase4a_plan_sha256": file_sha(out / "phase4a_plan.json"),
        "input_validation_sha256": file_sha(out / "input_validation.json"),
        "model": cfg["model"],
        "largest_passing_batch_size": max_size,
        "microbatch_size": max_size,
        "effective_batch_size": 5 * max_size,
        "gradient_accumulation_microbatches": 5,
        "trials": records,
        "training_started": False,
        "prospective_split_accessed": False,
    }
    write_json(out / "batch_size_smoke.json", result)
    return {
        "largest_passing_batch_size": max_size,
        "effective_batch_size": 5 * max_size,
        "trial_count": len(records),
        "all_required_checks_passed": True,
    }


def npy_bytes(array):
    buff = io.BytesIO()
    np.lib.format.write_array(buff, np.asanyarray(array), allow_pickle=False)
    return buff.getvalue()


def deterministic_npz(path: Path, arrays: dict[str, np.ndarray]):
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            info.flag_bits = 0
            zf.writestr(info, npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def build_cache(cfg):
    out = ROOT / cfg["output_dir"]
    plan = load_plan(cfg)
    validation = json.loads((out / "input_validation.json").read_text())
    smoke = json.loads((out / "batch_size_smoke.json").read_text())
    if validation.get("plan_sha256") != file_sha(out / "phase4a_plan.json") or plan.get("phase3") != phase3_pins(cfg):
        raise ValueError("Phase 4A panel or selected Phase 3 source pins changed before cache construction")
    if smoke.get("phase4a_plan_sha256") != file_sha(out / "phase4a_plan.json"):
        raise ValueError("batch smoke belongs to a different Phase 4A panel")
    if not smoke.get("largest_passing_batch_size"):
        raise ValueError("passing batch smoke required before cache construction")
    source_pins = {(r["split"], r["sample_id"]): r for r in validation["selected_source_pins"]}
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=False)
    e009 = yaml.safe_load((ROOT / cfg["e009_corruption_config"]).read_text())
    calibration_sha = file_sha(ROOT / e009["corruption"]["calibration_summary"])
    all_rows = [("train", r) for r in plan["training"]] + [("development", r) for r in plan["development"]]
    manifest_rows = []
    cache_rank = 0
    for split_name, row in all_rows:
        pinned = source_pins[(split_name, row["sample_id"])]
        source, coords, mask = load_source_arrays(row)
        source_sha = file_sha(source)
        if source_sha != pinned["source_sha256"]:
            raise ValueError(f"source changed since read-only validation: {row['sample_id']}")
        target = np.ascontiguousarray(coords - coords.mean(0, keepdims=True), dtype=np.float32)
        sigma = _calibrated_sigma(e009, row["length"])
        seed = 908015 + 1009 * cache_rank
        coarse = _make_corruption(torch.from_numpy(target), sigma, seed).numpy().astype(np.float32, copy=False)
        target_sha, coarse_sha = tensor_sha_np(target), tensor_sha_np(coarse)
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
        mask_sha = tensor_sha_np(mask, dtype=np.bool_)
        metadata = {
            "schema": "e010_phase4a_fixed_corruption_v1",
            "sample_id": row["sample_id"],
            "split": split_name,
            "source_split_label": "train" if split_name == "train" else "validation",
            "stratum": row["stratum"],
            "length": row["length"],
            "cache_rank": cache_rank,
            "selection_rank_sha256": row["selection_rank_sha256"],
            "source_path": row["source_path"],
            "source_sha256": source_sha,
            "calibration_summary_sha256": calibration_sha,
            "coordinate_noise_sigma_angstrom": sigma,
            "seed": seed,
            "construction": (
                "Phase 2/E009 fixed corruption: independent noise centered by axis plus "
                "centered cumulative drift at 0.12 sigma RMS"
            ),
            "target_sha256": target_sha,
            "corruption_sha256": coarse_sha,
            "mask_sha256": mask_sha,
        }
        archive = cache_dir / f"{split_name}_{cache_rank:05d}_{row['sample_id']}.npz"
        deterministic_npz(
            archive,
            {
                "target": target,
                "coarse": coarse,
                "mask": mask,
                "metadata": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            },
        )
        manifest_rows.append(
            {**metadata, "archive": str(archive.relative_to(ROOT)), "archive_sha256": file_sha(archive)}
        )
        cache_rank += 1
    manifest = {
        "schema": "e010_phase4a_corruption_cache_manifest_v1",
        "authorizes_downstream": False,
        "deterministic_archive_format": "NPZ with fixed ZIP timestamps and sorted member names",
        "training_count": 2048,
        "development_count": 320,
        "total": len(manifest_rows),
        "phase4a_plan_sha256": file_sha(out / "phase4a_plan.json"),
        "input_validation_sha256": file_sha(out / "input_validation.json"),
        "batch_size_smoke_sha256": file_sha(out / "batch_size_smoke.json"),
        "entries": manifest_rows,
    }
    write_json(out / "corruption_cache_manifest.json", manifest)
    return {
        "cache_count": len(manifest_rows),
        "training_count": 2048,
        "development_count": 320,
        "cache_manifest_sha256": file_sha(out / "corruption_cache_manifest.json"),
        "deterministic_npz": True,
    }


def load_cache_entry(entry):
    path = ROOT / entry["archive"]
    if file_sha(path) != entry["archive_sha256"]:
        raise ValueError(f"cached corruption archive hash mismatch: {entry['sample_id']}")
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        target, coarse, mask = z["target"].copy(), z["coarse"].copy(), z["mask"].copy()
    if (
        meta["sample_id"] != entry["sample_id"]
        or tensor_sha_np(target) != entry["target_sha256"]
        or tensor_sha_np(coarse) != entry["corruption_sha256"]
        or tensor_sha_np(mask, np.bool_) != entry["mask_sha256"]
    ):
        raise ValueError(f"cache tensor hash mismatch: {entry['sample_id']}")
    return target, coarse, mask


def verify_cache(cfg):
    out = ROOT / cfg["output_dir"]
    manifest_path = out / "corruption_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("authorizes_downstream") is not False or manifest.get("total") != 2368:
        raise ValueError("unexpected or authorizing Phase 4A corruption cache manifest")
    checked = []
    for entry in manifest["entries"]:
        target, coarse, mask = load_cache_entry(entry)
        if not np.isfinite(target).all() or not np.isfinite(coarse).all() or not mask.all():
            raise ValueError(f"non-finite or incomplete cache arrays: {entry['sample_id']}")
        checked.append(
            {
                "sample_id": entry["sample_id"],
                "split": entry["split"],
                "archive_sha256": entry["archive_sha256"],
                "target_sha256": entry["target_sha256"],
                "corruption_sha256": entry["corruption_sha256"],
                "mask_sha256": entry["mask_sha256"],
            }
        )
    result = {
        "schema": "e010_phase4a_cache_validation_v1",
        "authorizes_downstream": False,
        "cache_manifest_sha256": file_sha(manifest_path),
        "count": len(checked),
        "training_count": sum(x["split"] == "train" for x in checked),
        "development_count": sum(x["split"] == "development" for x in checked),
        "all_archive_and_tensor_hashes_verified": True,
        "all_arrays_finite_and_complete": True,
        "entries": checked,
        "prospective_split_accessed": False,
        "training_started": False,
    }
    write_json(out / "cache_validation.json", result)
    return {
        "count": result["count"],
        "training_count": result["training_count"],
        "development_count": result["development_count"],
        "verified": True,
    }


def kabsch_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    p, t = pred - pred.mean(0, keepdim=True), target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.T @ t)
    eye = torch.eye(3, device=p.device, dtype=p.dtype)
    eye[-1, -1] = torch.det(u @ vh)
    aligned = p @ (u @ eye @ vh)
    return float((aligned - t).square().sum(-1).mean().sqrt().item())


def geometry_metrics(pred: torch.Tensor, target: torch.Tensor):
    out = {}
    for sep in (1, 2, 3):
        pd = torch.linalg.vector_norm(pred[sep:] - pred[:-sep], dim=-1)
        td = torch.linalg.vector_norm(target[sep:] - target[:-sep], dim=-1)
        out[f"i_plus_{sep}_distance_rmse_angstrom"] = float((pd - td).square().mean().sqrt())
    a, b, c = pred[1:-2] - pred[:-3], pred[2:-1] - pred[1:-2], pred[3:] - pred[2:-1]
    x, y, z = target[1:-2] - target[:-3], target[2:-1] - target[1:-2], target[3:] - target[2:-1]
    pv, tv = (torch.cross(a, b, dim=-1) * c).sum(-1), (torch.cross(x, y, dim=-1) * z).sum(-1)
    out["chirality_inversions"] = int((pv * tv < 0).sum())
    out["chirality_triplets"] = int(pv.numel())
    out["chirality_inversion_rate"] = float((pv * tv < 0).float().mean()) if pv.numel() else 0.0
    out["target_radius_gyration_angstrom"] = float(torch.sqrt((target - target.mean(0)).square().sum(-1).mean()))
    out["prediction_radius_gyration_angstrom"] = float(torch.sqrt((pred - pred.mean(0)).square().sum(-1).mean()))
    return out


def bootstrap_mean_ci(values, reps=10000, seed=41042):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for start in range(0, reps, 256):
        n = min(256, reps - start)
        indices = rng.integers(0, len(arr), size=(n, len(arr)))
        means[start : start + n] = arr[indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_percentile": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "replicates": reps,
        "unit": "development_identity",
    }


def development_baseline(cfg):
    out = ROOT / cfg["output_dir"]
    load_plan(cfg)
    if not (out / "cache_validation.json").is_file():
        raise FileNotFoundError("validate all deterministic cache entries before baseline preparation")
    manifest = json.loads((out / "corruption_cache_manifest.json").read_text())
    dev_entries = [e for e in manifest["entries"] if e["split"] == "development"]
    rows, targets, predictions = [], [], []
    for entry in dev_entries:
        target_np, coarse_np, _ = load_cache_entry(entry)
        target, coarse = torch.from_numpy(target_np), torch.from_numpy(coarse_np)
        rows.append(entry)
        targets.append(target)
        predictions.append(coarse)
    rmse = np.asarray([kabsch_rmse(p, t) for p, t in zip(predictions, targets, strict=True)])
    lengths = np.asarray([e["length"] for e in rows], dtype=float)
    slopes = float(np.polyfit(lengths, rmse, 1)[0])
    per_identity = []
    for entry, pred, target, error in zip(rows, predictions, targets, rmse, strict=True):
        geo = geometry_metrics(pred, target)
        per_identity.append(
            {
                "sample_id": entry["sample_id"],
                "length": entry["length"],
                "stratum": entry["stratum"],
                "corrupted_input_aligned_rmse_angstrom": float(error),
                "geometry_telemetry": geo,
                "paired_change_after_training": None,
            }
        )
    strata = {}
    for s in cfg["selection"]["strata"]:
        subset = [r for r in per_identity if r["stratum"] == s["name"]]
        strata[s["name"]] = {
            "count": len(subset),
            "mean_aligned_rmse_angstrom": float(np.mean([r["corrupted_input_aligned_rmse_angstrom"] for r in subset])),
            "mean_rmse_bootstrap_ci95": bootstrap_mean_ci(
                [r["corrupted_input_aligned_rmse_angstrom"] for r in subset], seed=41042 + len(strata)
            ),
        }
    all_geo = [r["geometry_telemetry"] for r in per_identity]
    total_trips = sum(x["chirality_triplets"] for x in all_geo)
    total_inv = sum(x["chirality_inversions"] for x in all_geo)
    local_sep = {
        f"i_plus_{k}": float(np.mean([x[f"i_plus_{k}_distance_rmse_angstrom"] for x in all_geo])) for k in (1, 2, 3)
    }
    ratios = [
        x["prediction_radius_gyration_angstrom"] / max(x["target_radius_gyration_angstrom"], 1e-8) for x in all_geo
    ]
    collapsed = [
        r["sample_id"]
        for r, x, ratio in zip(per_identity, all_geo, ratios, strict=True)
        if x["prediction_radius_gyration_angstrom"] < 1.0 or ratio < 0.5
    ]
    diversity_by_stratum = {}
    for s in cfg["selection"]["strata"]:
        indexes = [i for i, r in enumerate(per_identity) if r["stratum"] == s["name"]]
        rg_pred = np.asarray([all_geo[i]["prediction_radius_gyration_angstrom"] for i in indexes])
        rg_target = np.asarray([all_geo[i]["target_radius_gyration_angstrom"] for i in indexes])
        denominator = float(rg_target.std(ddof=0))
        ratio = float(rg_pred.std(ddof=0) / denominator) if denominator > 1e-12 else None
        diversity_by_stratum[s["name"]] = {
            "predicted_radius_gyration_sd": float(rg_pred.std(ddof=0)),
            "target_radius_gyration_sd": denominator,
            "sd_ratio_prediction_to_target": ratio,
            "collapse_indicator": bool(ratio is not None and ratio < 0.5),
        }
    result = {
        "schema": "e010_phase4a_development_corrupted_baseline_v1",
        "authorizes_downstream": False,
        "split": "development_only",
        "count": len(rows),
        "mean_aligned_rmse_angstrom": float(rmse.mean()),
        "median_aligned_rmse_angstrom": float(np.median(rmse)),
        "maximum_aligned_rmse_angstrom": float(rmse.max()),
        "mean_rmse_bootstrap_ci95": bootstrap_mean_ci(rmse),
        "error_vs_length_slope_angstrom_per_residue": slopes,
        "by_length_stratum": strata,
        "mean_local_distance_rmse_angstrom": local_sep,
        "mean_i_plus_1_i_plus_2_i_plus_3_distance_rmse_angstrom": float(np.mean(list(local_sep.values()))),
        "chirality_inversions": total_inv,
        "chirality_triplets": total_trips,
        "chirality_inversion_rate": total_inv / total_trips,
        "coordinate_collapse_identities": collapsed,
        "coordinate_collapse_count": len(collapsed),
        "diversity_collapse_by_stratum": diversity_by_stratum,
        "per_identity_baseline_and_paired_change_slots": per_identity,
        "post_training_paired_changes_available": False,
        "development_mean_rmse_checkpoint_selection_only": True,
        "prospective_split_accessed": False,
        "training_started": False,
    }
    write_json(out / "development_corrupted_baseline.json", result)
    return {
        "development_count": len(rows),
        "corrupted_mean_rmse_angstrom": float(rmse.mean()),
        "slope": slopes,
        "chirality_rate": total_inv / total_trips,
    }


def prepare_resume(cfg):
    out = ROOT / cfg["output_dir"]
    plan = load_plan(cfg)
    validation = json.loads((out / "input_validation.json").read_text())
    smoke = json.loads((out / "batch_size_smoke.json").read_text())
    cache_manifest_path = out / "corruption_cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    cache_validation_path = out / "cache_validation.json"
    cache_validation = json.loads(cache_validation_path.read_text())
    if (
        cache_validation.get("all_archive_and_tensor_hashes_verified") is not True
        or cache_validation.get("count") != 2368
    ):
        raise ValueError("all cache hashes must be verified before exact resume preparation")
    baseline_path = out / "development_corrupted_baseline.json"
    if not baseline_path.is_file():
        development_baseline(cfg)
    _baseline = json.loads(baseline_path.read_text())
    if (
        validation.get("plan_sha256") != file_sha(out / "phase4a_plan.json")
        or smoke.get("phase4a_plan_sha256") != file_sha(out / "phase4a_plan.json")
        or cache_manifest.get("phase4a_plan_sha256") != file_sha(out / "phase4a_plan.json")
        or plan.get("phase3") != phase3_pins(cfg)
    ):
        raise ValueError("Phase 4A input pins changed before update-0 resume preparation")
    micro = int(smoke["largest_passing_batch_size"])
    strata = [s["name"] for s in cfg["selection"]["strata"]]
    train_by_stratum = {name: [r for r in plan["training"] if r["stratum"] == name] for name in strata}
    sampler_rng = random.Random(cfg["statistics"]["seed"])
    schedule = []
    steps_per_epoch = max(math.ceil(len(v) / micro) for v in train_by_stratum.values())
    for epoch in range(1, cfg["training_plan"]["epochs"] + 1):
        buckets = {}
        for name in strata:
            bucket = [r["sample_id"] for r in train_by_stratum[name]]
            sampler_rng.shuffle(bucket)
            buckets[name] = [bucket[i : i + micro] for i in range(0, len(bucket), micro)]
        for step in range(steps_per_epoch):
            group = {name: buckets[name][step] if step < len(buckets[name]) else [] for name in strata}
            schedule.append(
                {
                    "epoch": epoch,
                    "step_in_epoch": step + 1,
                    "stratum_microbatches": group,
                    "effective_batch_sample_count": sum(len(v) for v in group.values()),
                }
            )
    exposure_counts = {r["sample_id"]: 0 for r in plan["training"]}
    for epoch in range(1, cfg["training_plan"]["epochs"] + 1):
        for item in schedule[(epoch - 1) * steps_per_epoch : epoch * steps_per_epoch]:
            for ids in item["stratum_microbatches"].values():
                for sid in ids:
                    exposure_counts[sid] += 1
    if set(exposure_counts.values()) != {cfg["training_plan"]["exposures_per_identity"]}:
        raise ValueError("prepared schedule does not give every training identity exactly 50 exposures")
    effective_counts = [x["effective_batch_sample_count"] for x in schedule]
    if any(not all(x["stratum_microbatches"][s] for s in strata) for x in schedule):
        raise ValueError("a planned accumulated update is missing one or more length strata")
    seed = cfg["model"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("exact update-0 resume state requires the CUDA execution environment")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    model_spec = {
        k: cfg["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")
    }
    model = GlobalEquivariantResidual(**model_spec).to(device)
    count = sum(p.numel() for p in model.parameters())
    if count != 12844352:
        raise ValueError(f"fresh Phase 4A model parameter count mismatch: {count}")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training_plan"]["learning_rate"], weight_decay=cfg["training_plan"]["weight_decay"]
    )
    model_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ckpt = {
        "schema": "e010_phase4a_exact_resume_state_v1",
        "global_update": 0,
        "epoch": 0,
        "model": model_cpu,
        "optimizer": opt.state_dict(),
        "scheduler": None,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": [v.cpu() for v in torch.cuda.get_rng_state_all()],
        "sampler_rng_state": sampler_rng.getstate(),
        "training_schedule": schedule,
        "schedule_cursor": 0,
        "identity_exposures": {r["sample_id"]: 0 for r in plan["training"]},
        "planned_final_exposures": exposure_counts,
        "microbatch_size": micro,
        "effective_batch_size_nominal": 5 * micro,
        "effective_batch_size_min": min(effective_counts),
        "effective_batch_size_max": max(effective_counts),
        "gradient_accumulation_microbatches": len(strata),
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_updates": len(schedule),
        "evaluate_after_exposures": list(EXPOSURE_BOUNDARIES),
        "plan_sha256": file_sha(out / "phase4a_plan.json"),
        "input_validation_sha256": file_sha(out / "input_validation.json"),
        "cache_manifest_sha256": file_sha(cache_manifest_path),
        "batch_size_smoke_sha256": file_sha(out / "batch_size_smoke.json"),
        "cache_validation_sha256": file_sha(cache_validation_path),
        "development_baseline_sha256": file_sha(baseline_path),
        "determinism": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "training_started": False,
        "authorizes_downstream": False,
    }
    state_hashes = {
        "model_sha256": canonical_sha(ckpt["model"]),
        "optimizer_sha256": canonical_sha(ckpt["optimizer"]),
        "python_rng_sha256": canonical_sha(ckpt["python_rng_state"]),
        "numpy_rng_sha256": canonical_sha(ckpt["numpy_rng_state"]),
        "torch_cpu_rng_sha256": tensor_sha_torch(ckpt["torch_cpu_rng_state"]),
        "torch_cuda_rng_sha256": canonical_sha(ckpt["torch_cuda_rng_state"]),
        "sampler_rng_sha256": canonical_sha(ckpt["sampler_rng_state"]),
        "schedule_sha256": sha_bytes(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()),
        "planned_exposures_sha256": sha_bytes(
            json.dumps(exposure_counts, sort_keys=True, separators=(",", ":")).encode()
        ),
    }
    ckpt["state_hashes"] = state_hashes
    resume_path = out / "resume_update_0000.pt"
    if resume_path.exists():
        raise FileExistsError(f"refusing to overwrite exact resume state: {resume_path}")
    torch.save(ckpt, resume_path)
    loaded = torch.load(resume_path, map_location="cpu", weights_only=False)
    if loaded["state_hashes"] != state_hashes or canonical_sha(loaded["model"]) != state_hashes["model_sha256"]:
        raise ValueError("saved update-0 state did not pass exact hash reload validation")
    manifest = {
        "schema": "e010_phase4a_resume_manifest_v1",
        "authorizes_downstream": False,
        "resume_path": str(resume_path.relative_to(ROOT)),
        "resume_sha256": file_sha(resume_path),
        "cache_validation_sha256": file_sha(cache_validation_path),
        "global_update": 0,
        "parameter_count": count,
        "fresh_seed": seed,
        "model_state_is_phase3_weights": False,
        "state_hashes": state_hashes,
        "optimizer": "AdamW lr=3e-4 weight_decay=0, empty fresh state at update 0",
        "scheduler": None,
        "sampler_state_complete": True,
        "all_rng_states_complete": True,
        "schedule_sha256": state_hashes["schedule_sha256"],
        "planned_optimizer_updates": len(schedule),
        "sample_exposures_total": sum(exposure_counts.values()),
        "per_identity_exposure_count": 50,
        "microbatch_size": micro,
        "effective_batch_size_nominal": 5 * micro,
        "effective_batch_size_min": min(effective_counts),
        "effective_batch_size_max": max(effective_counts),
        "gradient_accumulation_microbatches": len(strata),
        "training_started": False,
        "prospective_split_accessed": False,
        "pilot_integration_prepared": False,
    }
    write_json(out / "resume_manifest.json", manifest)
    training_protocol = {
        "schema": "e010_phase4a_training_protocol_v1",
        "authorizes_downstream": False,
        "checkpoint_selection": (
            "lowest development mean aligned RMSE only at the five predeclared exposure "
            "boundaries; first boundary wins ties"
        ),
        "exposure_boundaries": list(EXPOSURE_BOUNDARIES),
        "schedule_sha256": state_hashes["schedule_sha256"],
        "optimizer_updates_per_exposure": steps_per_epoch,
        "optimizer_updates_total": len(schedule),
        "effective_batch_size_nominal": 5 * micro,
        "microbatch_size": micro,
        "gradient_accumulation": (
            "one backward pass per length stratum; gradients accumulated across all five "
            "strata and divided by actual effective sample count before each optimizer "
            "step"
        ),
        "loss": (
            "per-structure masked xyz MSE + 1e-5 masked residual L2; average structures equally, independent of length"
        ),
        "bootstrap": cfg["statistics"],
        "acceptance": cfg["acceptance"],
        "collapse_indicators": cfg["collapse_indicators"],
        "development_corrupted_baseline_sha256": file_sha(baseline_path),
        "cache_validation_sha256": file_sha(cache_validation_path),
        "training_started": False,
        "prospective_split_accessed": False,
    }
    write_json(out / "training_protocol.json", training_protocol)
    return {
        "resume_path": str(resume_path.relative_to(ROOT)),
        "resume_sha256": manifest["resume_sha256"],
        "planned_optimizer_updates": len(schedule),
        "microbatch_size": micro,
        "effective_batch_size_nominal": 5 * micro,
        "sample_exposures_total": sum(exposure_counts.values()),
        "training_started": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-only", action="store_true")
    group.add_argument("--validate-inputs", action="store_true")
    group.add_argument("--smoke-batches", action="store_true")
    group.add_argument("--build-cache", action="store_true")
    group.add_argument("--verify-cache", action="store_true")
    group.add_argument("--development-baseline", action="store_true")
    group.add_argument("--prepare-resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if args.plan_only:
        result = plan_only(cfg)
    elif args.validate_inputs:
        result = validate_inputs(cfg)
    elif args.smoke_batches:
        result = smoke_batches(cfg)
    elif args.build_cache:
        result = build_cache(cfg)
    elif args.verify_cache:
        result = verify_cache(cfg)
    elif args.development_baseline:
        result = development_baseline(cfg)
    else:
        result = prepare_resume(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
