#!/usr/bin/env python3
"""Exact-replay-gated E010 Phase 2 continuation (non-authorizing)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from scripts.run_e010_phase2 import _load_pairs, _metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e010_global_equivariant_phase2_v2.yaml"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    x = value.detach().contiguous().cpu()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def serialized_sha(value) -> str:
    def normalize(item):
        if isinstance(item, torch.Tensor):
            t = item.detach().contiguous().cpu()
            return {
                "__tensor__": True,
                "dtype": str(t.dtype),
                "shape": list(t.shape),
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        if isinstance(item, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(item.dtype),
                "shape": list(item.shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(item).tobytes()).hexdigest(),
            }
        if isinstance(item, dict):
            return {str(k): normalize(v) for k, v in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (list, tuple)):
            return [normalize(x) for x in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return {"__repr__": repr(item), "__type__": type(item).__qualname__}

    raw = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def state_hashes(checkpoint: dict) -> dict:
    return {
        "model_state_sha256": serialized_sha(checkpoint["model"]),
        "optimizer_state_sha256": serialized_sha(checkpoint["optimizer"]),
        "python_rng_state_sha256": serialized_sha(checkpoint["python_rng_state"]),
        "numpy_rng_state_sha256": serialized_sha(checkpoint["numpy_rng_state"]),
        "torch_cpu_rng_state_sha256": tensor_sha(checkpoint["torch_cpu_rng_state"]),
        "torch_cuda_rng_state_sha256": serialized_sha(checkpoint["torch_cuda_rng_state"]),
        "identity_order_sha256": hashlib.sha256(
            json.dumps(checkpoint["identity_order"], separators=(",", ":")).encode()
        ).hexdigest(),
        "continuation_uniform_tie_order_sha256": hashlib.sha256(
            "\n".join(checkpoint["continuation_uniform_tie_order"]).encode()
        ).hexdigest(),
        "exposures_sha256": hashlib.sha256(
            json.dumps(checkpoint["identity_exposures"], sort_keys=True).encode()
        ).hexdigest(),
        "training_loss_history_sha256": hashlib.sha256(
            np.asarray(checkpoint["training_loss_history"], dtype="<f8").tobytes()
        ).hexdigest(),
    }


def capture_state(model, opt, rng, order, exposures, losses, update, uniform_tie_order):
    return {
        "global_update": int(update),
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": None,
        "python_rng_state": rng.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": [x.cpu() for x in torch.cuda.get_rng_state_all()],
        "identity_order": list(order),
        "identity_exposures": dict(exposures),
        "continuation_uniform_tie_order": list(uniform_tie_order),
        "training_loss_history": list(losses),
        "determinism": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def restore_rng(checkpoint, rng):
    rng.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])


def source_audit(cfg, out):
    _v1_out = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase2_v1_prepared"
    config = ROOT / cfg["source_config"]
    protocol_path = ROOT / cfg["source_protocol"]
    result_path = ROOT / cfg["source_result"]
    checkpoint_path = ROOT / cfg["source_model_checkpoint"]
    protocol = json.loads(protocol_path.read_text())
    result = json.loads(result_path.read_text())
    if (
        protocol.get("authorizes_downstream") is not False
        or result.get("status") != "failed"
        or result.get("updates") != 2000
    ):
        raise ValueError("v1 source record is not the immutable failed update-2000 result")
    source_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if source_ckpt.get("config_sha256") != file_sha(config) or source_ckpt.get("panel_sha256") != file_sha(
        protocol_path
    ):
        raise ValueError("v1 checkpoint config/panel hash does not match its pinned source")
    pins = []
    for item in protocol["panel"]:
        path = ROOT / item["archive"]
        observed_sha = file_sha(path)
        if observed_sha != item["archive_sha256"]:
            raise ValueError(f"panel/corruption archive hash mismatch: {item['sample_id']}")
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"].item()))
            for field in ("target_sha256", "corruption_sha256", "mask_sha256"):
                if metadata[field] != item[field]:
                    raise ValueError(f"panel/corruption tensor hash mismatch: {item['sample_id']} {field}")
            expected_hashes = {
                "target_sha256": hashlib.sha256(
                    np.ascontiguousarray(z["target"].astype("<f4", copy=False)).tobytes()
                ).hexdigest(),
                "corruption_sha256": hashlib.sha256(
                    np.ascontiguousarray(z["coarse"].astype("<f4", copy=False)).tobytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(z["mask"].astype(np.bool_, copy=False)).tobytes()
                ).hexdigest(),
            }
            if any(expected_hashes[k] != item[k] for k in expected_hashes):
                raise ValueError(f"panel/corruption byte hash mismatch: {item['sample_id']}")
        pins.append(
            {
                "sample_id": item["sample_id"],
                "archive_sha256": observed_sha,
                "target_sha256": item["target_sha256"],
                "corruption_sha256": item["corruption_sha256"],
                "mask_sha256": item["mask_sha256"],
            }
        )
    return {
        "schema": "e010_phase2_v2_input_audit_v1",
        "authorizes_downstream": False,
        "v1_status_preserved": "failed_under_predeclared_2000_update_gate",
        "v1_result_sha256": file_sha(result_path),
        "v1_config_sha256": file_sha(config),
        "v1_protocol_sha256": file_sha(protocol_path),
        "source_model_checkpoint_sha256": file_sha(checkpoint_path),
        "source_checkpoint_keys": sorted(source_ckpt),
        "source_checkpoint_contains_optimizer_state": "optimizer" in source_ckpt,
        "source_checkpoint_contains_rng_state": any("rng" in k.lower() for k in source_ckpt),
        "model_parameter_tensor_count": len(source_ckpt["model"]),
        "panel_count": len(pins),
        "panel": pins,
        "panel_archive_set_sha256": hashlib.sha256("\n".join(x["archive_sha256"] for x in pins).encode()).hexdigest(),
        "source_checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }


def summarize_boundary(model, pairs, device, update, exposures, loss_history, opt_stats):
    record = _metrics(model, pairs, device, update)
    lengths = np.asarray([x["length"] for x in record["per_structure"]])
    errors = np.asarray([x["aligned_rmse_angstrom"] for x in record["per_structure"]])
    by_stratum = {}
    for lo, hi, label in (
        (20, 64, "20-64"),
        (65, 128, "65-128"),
        (129, 256, "129-256"),
        (257, 384, "257-384"),
        (385, 500, "385-500"),
    ):
        vals = errors[(lengths >= lo) & (lengths <= hi)]
        if vals.size:
            by_stratum[label] = {
                "count": int(vals.size),
                "mean_aligned_rmse_angstrom": float(vals.mean()),
                "median_aligned_rmse_angstrom": float(np.median(vals)),
                "maximum_aligned_rmse_angstrom": float(vals.max()),
            }
    n = len(loss_history)
    tail_start = max(0, n - math.ceil(0.20 * update))
    xs = np.arange(tail_start + 1, n + 1, dtype=float)
    tail = np.asarray(loss_history[tail_start:], dtype=float)
    loss_slope = float(np.polyfit(xs, tail, 1)[0]) if len(tail) > 1 else None
    stats = opt_stats
    record.update(
        {
            "identity_exposures": {
                "minimum": min(exposures.values()),
                "mean": float(np.mean(list(exposures.values()))),
                "maximum": max(exposures.values()),
                "per_identity": exposures,
            },
            "by_length_stratum": by_stratum,
            "training_loss_last_20_percent": {
                "first_update": tail_start + 1,
                "last_update": update,
                "slope_loss_per_update": loss_slope,
                "first_loss": float(tail[0]),
                "last_loss": float(tail[-1]),
            },
            "gradient_and_optimizer_step_norms": {
                "window_start_update": stats["window_start_update"],
                "window_end_update": update,
                "updates": len(stats["gradient_norms"]),
                "gradient_l2_mean": float(np.mean(stats["gradient_norms"])) if stats["gradient_norms"] else None,
                "gradient_l2_max": float(np.max(stats["gradient_norms"])) if stats["gradient_norms"] else None,
                "optimizer_step_l2_mean": float(np.mean(stats["step_norms"])) if stats["step_norms"] else None,
                "optimizer_step_l2_max": float(np.max(stats["step_norms"])) if stats["step_norms"] else None,
                "clip_fraction": float(np.mean(stats["clipped"])) if stats["clipped"] else None,
            },
        }
    )
    return record


def make_checkpoint(model, opt, rng, order, exposures, losses, update, cfg_sha, protocol_sha, uniform_tie_order):
    ckpt = capture_state(model, opt, rng, order, exposures, losses, update, uniform_tie_order)
    ckpt["config_sha256"] = cfg_sha
    ckpt["panel_protocol_sha256"] = protocol_sha
    ckpt["state_hashes"] = state_hashes(ckpt)
    return ckpt


def save_checkpoint(out, model, opt, rng, order, exposures, losses, update, cfg_sha, protocol_sha, uniform_tie_order):
    ckpt = make_checkpoint(model, opt, rng, order, exposures, losses, update, cfg_sha, protocol_sha, uniform_tie_order)
    path = out / f"checkpoint_update_{update:04d}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    observed_sha = file_sha(path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if loaded["state_hashes"] != state_hashes(loaded):
        raise ValueError(f"saved checkpoint internal state hash mismatch at {update}")
    return path, observed_sha, loaded


def _compare_metric_records(replayed, stored):
    names = (
        "mean_aligned_rmse_angstrom",
        "median_aligned_rmse_angstrom",
        "maximum_aligned_rmse_angstrom",
        "error_vs_length_slope_angstrom_per_residue",
    )
    diffs = {}
    for new, old in zip(replayed, stored, strict=True):
        for name in names:
            delta = abs(float(new[name]) - float(old[name]))
            diffs[f"update_{new['update']}_{name}"] = delta
            if delta > 1e-6:
                raise ValueError(f"exact replay metric mismatch at update {new['update']}: {name} delta={delta}")
        by_new = {x["sample_id"]: x["aligned_rmse_angstrom"] for x in new["per_structure"]}
        by_old = {x["sample_id"]: x["aligned_rmse_angstrom"] for x in old["per_structure"]}
        if by_new.keys() != by_old.keys() or any(abs(by_new[k] - by_old[k]) > 1e-6 for k in by_new):
            raise ValueError(f"exact replay per-identity mismatch at update {new['update']}")
    return diffs


def execute(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("v2 continuation requires CUDA; refusing CPU resume")
    out = ROOT / cfg["output_dir"]
    if out.exists():
        raise FileExistsError(f"v2 output exists; refusing overwrite: {out}")
    out.mkdir(parents=True)
    torch.backends.cudnn.benchmark = False
    audit = source_audit(cfg, out)
    (out / "input_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["source_checkpoint_contains_optimizer_state"] or audit["source_checkpoint_contains_rng_state"]:
        raise ValueError("unexpected source state availability; use a dedicated exact-state loader")
    source_checkpoint = torch.load(ROOT / cfg["source_model_checkpoint"], map_location="cpu", weights_only=False)
    protocol_path = ROOT / cfg["source_protocol"]
    json.loads(protocol_path.read_text())
    source_result = json.loads((ROOT / cfg["source_result"]).read_text())
    v1_train_cfg = yaml.safe_load((ROOT / cfg["source_config"]).read_text())
    pairs = _load_pairs(ROOT / v1_train_cfg["output_dir"])
    device = torch.device("cuda")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = GlobalEquivariantResidual(**cfg["model"]).to(device)
    if len(model.state_dict()) != audit["model_parameter_tensor_count"]:
        raise ValueError("architecture parameter tensor topology differs from source checkpoint")
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    rng = random.Random(cfg["seed"])
    order = list(range(len(pairs)))
    rng.shuffle(order)
    exposures = {x["metadata"]["sample_id"]: 0 for x in pairs}
    losses, replayed, _step_records = [], [], []
    evaluate = {0, 100, 250, 500, 1000, 1500, 2000}
    stats = {"window_start_update": 0, "gradient_norms": [], "step_norms": [], "clipped": []}
    for step in range(2000):
        if step in evaluate:
            replayed.append(_metrics(model, pairs, device, step))
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
            raise FloatingPointError(f"non-finite replay loss at update {step}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"non-finite replay gradient at update {step}")
        pre = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_max_norm"])
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        step_norm = math.sqrt(
            sum(
                float((p.detach() - old).double().square().sum())
                for p, old in zip(model.parameters(), before, strict=True)
            )
        )
        losses.append(float(loss.detach()))
        exposures[pair["metadata"]["sample_id"]] += 1
        stats["gradient_norms"].append(pre)
        stats["step_norms"].append(step_norm)
        stats["clipped"].append(pre > cfg["optimizer"]["gradient_clip_max_norm"])
    replayed.append(_metrics(model, pairs, device, 2000))
    old_records = source_result["records"]
    metric_diffs = _compare_metric_records(replayed, old_records)
    expected = source_checkpoint["model"]
    observed = model.state_dict()
    if expected.keys() != observed.keys() or any(
        not torch.equal(expected[k].cpu(), observed[k].detach().cpu()) for k in expected
    ):
        raise ValueError(
            "update-2000 exact replay did not reproduce the saved model checkpoint bit-for-bit; continuation refused"
        )
    replay_model_sha = serialized_sha({k: v.detach().cpu() for k, v in observed.items()})
    old_model_sha = serialized_sha(expected)
    if replay_model_sha != old_model_sha:
        raise ValueError("update-2000 model state hash differs from source checkpoint")
    # The replayed optimizer/scheduler/sampler and all RNG states now define the exact
    # update-2000 state missing from the original weights-only file.
    tie_order = sorted(
        (x["metadata"]["sample_id"] for x in pairs),
        key=lambda sid: hashlib.sha256(f"e010_phase2_v2_uniform|{sid}".encode()).hexdigest(),
    )
    start_path, start_sha, start_state = save_checkpoint(
        out, model, opt, rng, order, exposures, losses, 2000, file_sha(CONFIG), file_sha(protocol_path), tie_order
    )
    start_audit = {
        "status": "exact_replay_verified",
        "saved_source_model_tensor_match": True,
        "source_model_state_sha256": old_model_sha,
        "replayed_model_state_sha256": replay_model_sha,
        "source_metric_max_absolute_delta": max(metric_diffs.values()),
        "source_metric_deltas": metric_diffs,
        "reconstructed_complete_state_hashes": start_state["state_hashes"],
        "update_2000_checkpoint": start_path.name,
        "update_2000_checkpoint_sha256": start_sha,
        "optimizer": cfg["optimizer"],
        "scheduler_state": None,
        "rng_and_uniform_identity_schedule_reconstructed_by_exact_replay": True,
    }
    (out / "exact_replay_audit.json").write_text(json.dumps(start_audit, indent=2, sort_keys=True) + "\n")
    # Exercise the serialized resume path and verify state hashes before update 2001.
    loaded = torch.load(start_path, map_location="cpu", weights_only=False)
    if file_sha(start_path) != start_sha or loaded["state_hashes"] != state_hashes(loaded):
        raise ValueError("update-2000 checkpoint or state hash changed before continuation")
    model.load_state_dict(loaded["model"])
    opt.load_state_dict(loaded["optimizer"])
    order = list(loaded["identity_order"])
    exposures = dict(loaded["identity_exposures"])
    losses = list(loaded["training_loss_history"])
    tie_order = list(loaded["continuation_uniform_tie_order"])
    pair_by_id = {x["metadata"]["sample_id"]: x for x in pairs}
    restore_rng(loaded, rng)
    boundaries = set(cfg["training"]["evaluation_updates"])
    records = [summarize_boundary(model, pairs, device, 2000, exposures, losses, stats)]
    checkpoint_manifest = [
        {
            "update": 2000,
            "path": str(start_path.relative_to(ROOT)),
            "sha256": start_sha,
            "state_hashes": loaded["state_hashes"],
        }
    ]
    for global_step in range(2000, cfg["training"]["continuation_end_update"]):
        min_exposure = min(exposures.values())
        identity = next(sid for sid in tie_order if exposures[sid] == min_exposure)
        pair = pair_by_id[identity]
        target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
        opt.zero_grad(set_to_none=True)
        result = model(coarse[None], mask[None])
        valid = mask[:, None].to(target.dtype)
        coord = ((result["prediction"][0] - target).square() * valid).sum() / (3 * mask.sum())
        residual = (result["delta"][0].square() * valid).sum() / (3 * mask.sum())
        loss = coord + 1e-5 * residual
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {global_step}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"non-finite gradient at update {global_step}")
        pre = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["gradient_clip_max_norm"])
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        step_norm = math.sqrt(
            sum(
                float((p.detach() - old).double().square().sum())
                for p, old in zip(model.parameters(), before, strict=True)
            )
        )
        update = global_step + 1
        losses.append(float(loss.detach()))
        exposures[pair["metadata"]["sample_id"]] += 1
        stats["gradient_norms"].append(pre)
        stats["step_norms"].append(step_norm)
        stats["clipped"].append(pre > cfg["optimizer"]["gradient_clip_max_norm"])
        if update in boundaries:
            records.append(summarize_boundary(model, pairs, device, update, exposures, losses, stats))
            cp_path, cp_sha, cp = save_checkpoint(
                out,
                model,
                opt,
                rng,
                order,
                exposures,
                losses,
                update,
                file_sha(CONFIG),
                file_sha(protocol_path),
                tie_order,
            )
            checkpoint_manifest.append(
                {
                    "update": update,
                    "path": str(cp_path.relative_to(ROOT)),
                    "sha256": cp_sha,
                    "state_hashes": cp["state_hashes"],
                }
            )
            stats = {"window_start_update": update, "gradient_norms": [], "step_norms": [], "clipped": []}
    # Fixed, uniform exposure totals are a hard continuation invariant.
    for record in records[1:]:
        expected_exposure = record["update"] // len(pairs)
        ex = record["identity_exposures"]
        if (ex["minimum"], ex["mean"], ex["maximum"]) != (
            expected_exposure,
            float(expected_exposure),
            expected_exposure,
        ):
            raise ValueError(f"identity exposure schedule is not exactly uniform at {record['update']}")
    gates = cfg["acceptance"]
    for record in records[1:]:
        record["acceptance_gates"] = {
            "mean": record["mean_aligned_rmse_angstrom"] <= gates["mean_aligned_rmse_angstrom_max"],
            "median": record["median_aligned_rmse_angstrom"] <= gates["median_aligned_rmse_angstrom_max"],
            "maximum": record["maximum_aligned_rmse_angstrom"] <= gates["maximum_aligned_rmse_angstrom_max"],
            "length_slope": record["error_vs_length_slope_angstrom_per_residue"]
            <= gates["error_vs_length_slope_angstrom_per_residue_max"],
            "finite": record["non_finite_outputs"] == 0,
        }
    payload = {
        "schema": "e010_phase2_v2_continuation_result_v1",
        "authorizes_downstream": False,
        "status": "completed",
        "v1_status_preserved": "failed_under_predeclared_2000_update_gate",
        "start_update": 2000,
        "end_update": 6400,
        "exact_replay_verified": True,
        "evaluation_split": "same 32-identity training panel only",
        "updates": 4400,
        "objective": "masked xyz-coordinate MSE plus 1e-5 masked residual L2",
        "model_architecture": cfg["model"],
        "optimizer": cfg["optimizer"],
        "identity_schedule": (
            "replay v1 schedule exactly through update 2000; then deterministic "
            "least-exposed-first round robin with SHA-256 tie order, yielding exact "
            "uniform cumulative exposure at each fixed boundary"
        ),
        "no_geometry_losses": True,
        "no_priors": True,
        "no_sequence_features": True,
        "no_architecture_changes": True,
        "no_sampling": True,
        "no_held_out_data": True,
        "checkpoint_manifest": checkpoint_manifest,
        "records": records,
        "held_out_improvement_demonstrated": False,
        "conditional_supervised_pilot_prepared": False,
    }
    tmp = out / "continuation_result.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out / "continuation_result.json")
    (out / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "schema": "e010_phase2_v2_checkpoint_manifest_v1",
                "authorizes_downstream": False,
                "checkpoints": checkpoint_manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    cfg = yaml.safe_load(CONFIG.read_text())
    print(json.dumps(execute(cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
