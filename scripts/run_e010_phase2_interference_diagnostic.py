#!/usr/bin/env python3
"Focused, non-authorizing E010 Phase 2 interference diagnostic."

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e010_phase2_interference_diagnostic_v1.yaml"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tensor_sha(value):
    t = value.detach().contiguous().cpu()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def serialized_sha(value):
    def norm(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            return {"__tensor__": True, "dtype": str(t.dtype), "shape": list(t.shape), "sha256": tensor_sha(t)}
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            return {
                "__ndarray__": True,
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            }
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda p: str(p[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return {"repr": repr(x), "type": type(x).__qualname__}

    raw = json.dumps(norm(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def state_hashes(ckpt):
    return {
        "model_state_sha256": serialized_sha(ckpt["model"]),
        "optimizer_state_sha256": serialized_sha(ckpt["optimizer"]),
        "python_rng_state_sha256": serialized_sha(ckpt["python_rng_state"]),
        "numpy_rng_state_sha256": serialized_sha(ckpt["numpy_rng_state"]),
        "torch_cpu_rng_state_sha256": tensor_sha(ckpt["torch_cpu_rng_state"]),
        "torch_cuda_rng_state_sha256": serialized_sha(ckpt["torch_cuda_rng_state"]),
        "identity_order_sha256": hashlib.sha256(
            json.dumps(ckpt["identity_order"], separators=(",", ":")).encode()
        ).hexdigest(),
        "continuation_uniform_tie_order_sha256": hashlib.sha256(
            "\n".join(ckpt["continuation_uniform_tie_order"]).encode()
        ).hexdigest(),
        "exposures_sha256": hashlib.sha256(json.dumps(ckpt["identity_exposures"], sort_keys=True).encode()).hexdigest(),
        "training_loss_history_sha256": hashlib.sha256(
            np.asarray(ckpt["training_loss_history"], dtype="<f8").tobytes()
        ).hexdigest(),
    }


def kabsch_rmse(pred, target):
    p = pred - pred.mean(0, keepdim=True)
    q = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.T @ q)
    d = torch.det(u @ vh)
    fix = torch.eye(3, dtype=p.dtype, device=p.device)
    fix[-1, -1] = d
    aligned = p @ (u @ fix @ vh)
    return (aligned - q).square().sum(-1).mean().sqrt()


def load_pairs(protocol):
    pairs = []
    pins = []
    for item in protocol["panel"]:
        path = ROOT / item["archive"]
        observed = file_sha(path)
        if observed != item["archive_sha256"]:
            raise ValueError(f"corruption archive SHA mismatch: {item['sample_id']}")
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            arrays = {
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
            if any(arrays[k] != item[k] or meta[k] != item[k] for k in arrays):
                raise ValueError(f"corruption tensor SHA mismatch: {item['sample_id']}")
            if meta["sample_id"] != item["sample_id"] or meta["length"] != item["length"] or meta["split"] != "train":
                raise ValueError(f"corruption identity metadata mismatch: {item['sample_id']}")
            target = torch.from_numpy(z["target"].copy())
            coarse = torch.from_numpy(z["coarse"].copy())
            mask = torch.from_numpy(z["mask"].copy()).bool()
        pairs.append({"target": target, "coarse": coarse, "mask": mask, "metadata": meta})
        pins.append(
            {
                "sample_id": item["sample_id"],
                "length": item["length"],
                "stratum": item["stratum"],
                "archive": item["archive"],
                "archive_sha256": observed,
                **arrays,
            }
        )
    return pairs, pins


def verify_inputs(cfg):
    path = ROOT / cfg["source_checkpoint"]
    manifest_path = ROOT / cfg["source_v2_manifest"]
    protocol_path = ROOT / cfg["source_protocol"]
    v2cfg_path = ROOT / cfg["source_config"]
    checkpoint_sha = file_sha(path)
    manifest = json.loads(manifest_path.read_text())
    manifest_record = next((x for x in manifest["checkpoints"] if x["update"] == 6400), None)
    if not manifest_record or manifest_record["sha256"] != checkpoint_sha:
        raise ValueError("update-6400 checkpoint file SHA does not match manifest")
    if file_sha(ROOT / cfg["source_v2_result"]) != "d0a73ec846930f145697f0afc568a3738a07cc51d77448215825e7905b4391d7":
        raise ValueError("v2 continuation result hash changed")
    v2_result = json.loads((ROOT / cfg["source_v2_result"]).read_text())
    if (
        v2_result.get("updates") != 4400
        or v2_result.get("status") != "completed"
        or v2_result.get("authorizes_downstream") is not False
    ):
        raise ValueError("v2 continuation record is not the expected non-authorizing continuation from update 2000")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("panel_count") != 32 or protocol.get("authorizes_downstream") is not False:
        raise ValueError("v1 panel protocol is not the expected fixed non-authorizing 32-identity panel")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    observed_hashes = state_hashes(ckpt)
    if (
        ckpt.get("global_update") != 6400
        or observed_hashes != manifest_record["state_hashes"]
        or ckpt.get("state_hashes") != observed_hashes
    ):
        raise ValueError("update-6400 checkpoint internal model/optimizer/RNG/schedule state hash mismatch")
    if ckpt.get("config_sha256") != file_sha(v2cfg_path) or ckpt.get("panel_protocol_sha256") != file_sha(
        protocol_path
    ):
        raise ValueError("checkpoint source configuration or panel protocol pin mismatch")
    if ckpt.get("scheduler") is not None:
        raise ValueError("expected unchanged no-scheduler checkpoint")
    if len(ckpt["model"]) != len(
        GlobalEquivariantResidual(**yaml.safe_load(v2cfg_path.read_text())["model"]).state_dict()
    ):
        raise ValueError("checkpoint model tensor topology mismatch")
    pairs, pair_pins = load_pairs(protocol)
    v1_result = json.loads(
        (
            ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase2_v1_prepared/training_result.json"
        ).read_text()
    )
    if v1_result.get("status") != "failed" or v1_result.get("updates") != 2000:
        raise ValueError("Phase 2 v1 failed gate record missing or changed")
    return (
        ckpt,
        protocol,
        pairs,
        {
            "authorizes_downstream": False,
            "v1_status_preserved": "failed_under_predeclared_2000_update_gate",
            "v1_result_sha256": file_sha(
                ROOT
                / "reports/experiments/E010_global_equivariant_expressivity/phase2_v1_prepared/training_result.json"
            ),
            "v1_protocol_sha256": file_sha(protocol_path),
            "v2_config_sha256": file_sha(v2cfg_path),
            "v2_manifest_sha256": file_sha(manifest_path),
            "v2_result_sha256": file_sha(ROOT / cfg["source_v2_result"]),
            "v2_adjudication_sha256": file_sha(ROOT / cfg["source_v2_adjudication"]),
            "checkpoint_path": cfg["source_checkpoint"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_update": ckpt["global_update"],
            "checkpoint_state_hashes": observed_hashes,
            "model_source_sha256": file_sha(ROOT / cfg["source_model"]),
            "panel_count": len(pairs),
            "panel_corruption_pins": pair_pins,
        },
    )


def choose_representatives(cfg, pairs):
    result = []
    ns = cfg["selection"]["namespace"]
    for stratum in cfg["selection"]["strata"]:
        candidates = []
        for p in pairs:
            m = p["metadata"]
            if stratum["min"] <= m["length"] <= stratum["max"]:
                rank = hashlib.sha256(f"{ns}|{m['sample_id']}".encode()).hexdigest()
                candidates.append((rank, m["sample_id"], p))
        if not candidates:
            raise ValueError(f"no training panel identity in stratum {stratum['name']}")
        candidates.sort(key=lambda x: (x[0], x[1]))
        rank, sid, pair = candidates[0]
        result.append(
            {
                "stratum": stratum["name"],
                "sample_id": sid,
                "length": pair["metadata"]["length"],
                "metadata_rank_sha256": rank,
                "candidate_count": len(candidates),
                "candidate_ranks": [{"sample_id": x[1], "rank_sha256": x[0]} for x in candidates],
                "pair": pair,
            }
        )
    return result


def metric(model, pair, device, update, loss_value=None):
    model.eval()
    with torch.no_grad():
        target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
        out = model(coarse[None], mask[None])
        pred = out["prediction"][0]
        finite = bool(torch.isfinite(pred).all())
        rmse = float(kabsch_rmse(pred[mask], target[mask])) if finite else float("inf")
        geo = None
        if finite:
            geo = {}
            for sep in (1, 2, 3):
                pd = torch.linalg.vector_norm(pred[sep:] - pred[:-sep], dim=-1)
                td = torch.linalg.vector_norm(target[sep:] - target[:-sep], dim=-1)
                geo[f"i_plus_{sep}_distance_rmse_angstrom"] = float((pd - td).square().mean().sqrt())
            a, b, c = pred[1:-2] - pred[:-3], pred[2:-1] - pred[1:-2], pred[3:] - pred[2:-1]
            x, y, z = target[1:-2] - target[:-3], target[2:-1] - target[1:-2], target[3:] - target[2:-1]
            pv = (torch.cross(a, b, dim=-1) * c).sum(-1)
            tv = (torch.cross(x, y, dim=-1) * z).sum(-1)
            geo["chirality_inversions"] = int((pv * tv < 0).sum())
    row = {"update": update, "aligned_rmse_angstrom": rmse, "finite": finite, "geometry_telemetry": geo}
    if loss_value is not None:
        row["training_loss"] = float(loss_value)
    return row


def loss_for(model, pair, device, coordinate_only=False):
    target, coarse, mask = (pair[k].to(device) for k in ("target", "coarse", "mask"))
    out = model(coarse[None], mask[None])
    valid = mask[:, None].to(target.dtype)
    coord = ((out["prediction"][0] - target).square() * valid).sum() / (3 * mask.sum())
    if coordinate_only:
        return coord
    residual = (out["delta"][0].square() * valid).sum() / (3 * mask.sum())
    return coord + 1e-5 * residual


def new_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(), lr=cfg["arms"]["learning_rate"], weight_decay=cfg["arms"]["weight_decay"]
    )


def parameter_norm_delta(before, model):
    parts = []
    for old, p in zip(before, model.parameters(), strict=True):
        parts.append((p.detach() - old).square().sum())
    return float(torch.stack(parts).sum().sqrt().item())


def train_arm(kind, rep, cfg, checkpoint, v2cfg, device):
    pair, sid, length = rep["pair"], rep["sample_id"], rep["length"]
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    model = GlobalEquivariantResidual(**v2cfg["model"]).to(device)
    opt = new_optimizer(model, cfg)
    if kind == "checkpoint_specialization":
        model.load_state_dict(checkpoint["model"], strict=True)
        opt.load_state_dict(checkpoint["optimizer"])
    start_hash = serialized_sha(model.state_dict())
    if kind == "checkpoint_specialization" and start_hash != checkpoint["state_hashes"]["model_state_sha256"]:
        raise ValueError("specialization model did not exactly load update-6400 weights")
    _random_state_before = torch.get_rng_state().clone()
    if kind == "checkpoint_specialization":
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_cpu_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state"])
    max_updates = (
        cfg["arms"]["fresh_isolated_max_updates"]
        if kind == "fresh_isolated"
        else cfg["arms"]["checkpoint_specialization_max_updates"]
    )
    telemetry_updates = set(cfg["arms"]["telemetry_updates"])
    telemetry_updates = {u for u in telemetry_updates if u <= max_updates} | {0, max_updates}
    trajectory, rmse_trajectory, gradient_norms, step_norms, clipped = [], [], [], [], []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    row = metric(model, pair, device, 0)
    trajectory.append(row)
    rmse_trajectory.append(
        {"update": 0, "aligned_rmse_angstrom": row["aligned_rmse_angstrom"], "finite": row["finite"]}
    )
    if row["aligned_rmse_angstrom"] <= cfg["arms"]["early_success_aligned_rmse_angstrom"]:
        first_passage = 0
    else:
        first_passage = None
    for update in range(1, max_updates + 1):
        if first_passage is not None:
            break
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_for(model, pair, device)
        loss.backward()
        gradnorm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["arms"]["gradient_clip_max_norm"]).item()
        )
        was_clipped = gradnorm > cfg["arms"]["gradient_clip_max_norm"]
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        stepnorm = parameter_norm_delta(before, model)
        del before
        gradient_norms.append(gradnorm)
        step_norms.append(stepnorm)
        clipped.append(was_clipped)
        torch.cuda.synchronize(device)
        row = metric(model, pair, device, update, float(loss.detach().item()))
        rmse_trajectory.append(
            {"update": update, "aligned_rmse_angstrom": row["aligned_rmse_angstrom"], "finite": row["finite"]}
        )
        if (
            update in telemetry_updates
            or row["aligned_rmse_angstrom"] <= cfg["arms"]["early_success_aligned_rmse_angstrom"]
        ):
            trajectory.append(row)
        if row["aligned_rmse_angstrom"] <= cfg["arms"]["early_success_aligned_rmse_angstrom"]:
            first_passage = update
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    final_update = trajectory[-1]["update"]
    return {
        "kind": kind,
        "sample_id": sid,
        "length": length,
        "source_checkpoint_update": 6400 if kind == "checkpoint_specialization" else None,
        "maximum_updates": max_updates,
        "updates_executed": final_update,
        "success_threshold_aligned_rmse_angstrom": cfg["arms"]["early_success_aligned_rmse_angstrom"],
        "first_passage_update": first_passage,
        "final_aligned_rmse_angstrom": trajectory[-1]["aligned_rmse_angstrom"],
        "trajectory": trajectory,
        "rmse_trajectory": rmse_trajectory,
        "gradient_l2_norms": gradient_norms,
        "optimizer_step_l2_norms": step_norms,
        "clipped_updates": int(sum(clipped)),
        "clipping_fraction": float(np.mean(clipped)) if clipped else 0.0,
        "runtime_seconds": runtime,
        "device": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "start_model_state_sha256": start_hash,
        "checkpoint_weights_exact_at_start": kind != "checkpoint_specialization"
        or start_hash == checkpoint["state_hashes"]["model_state_sha256"],
    }


def gradient_audit(checkpoint, pairs, protocol, v2cfg, device):
    model = GlobalEquivariantResidual(**v2cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    before_model = serialized_sha(model.state_dict())
    before_optimizer = serialized_sha(checkpoint["optimizer"])
    grads, norms, losses = [], [], []
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for pair in pairs:
        model.zero_grad(set_to_none=True)
        loss = loss_for(model, pair, device, coordinate_only=True)
        gs = torch.autograd.grad(loss, params, allow_unused=True)
        flat = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).detach().reshape(-1).float().cpu()
                for p, g in zip(params, gs, strict=True)
            ]
        )
        grads.append(flat)
        norms.append(float(torch.linalg.vector_norm(flat)))
        losses.append(float(loss.detach().item()))
        del loss, gs, flat
    torch.cuda.synchronize(device)
    matrix = torch.stack(grads)
    unit = torch.nn.functional.normalize(matrix, dim=1, eps=1e-30)
    cosine = unit @ unit.T
    mean_grad = matrix.mean(0)
    mean_norm = float(torch.linalg.vector_norm(mean_grad))
    mean_unit = torch.nn.functional.normalize(mean_grad, dim=0, eps=1e-30)
    cos_mean = unit @ mean_unit
    ids = [p["metadata"]["sample_id"] for p in pairs]
    lengths = np.asarray([p["metadata"]["length"] for p in pairs], dtype=float)
    strata = [p["metadata"]["stratum"] for p in pairs]
    pair_rows = []
    groups = {}
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            val = float(cosine[i, j])
            key = (
                f"within:{strata[i]}"
                if strata[i] == strata[j]
                else f"between:{'|'.join(sorted((strata[i], strata[j])))}"
            )
            groups.setdefault(key, []).append(val)
            pair_rows.append(val)

    def summarize(vals):
        a = np.asarray(vals, dtype=float)
        return {
            "count": len(vals),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "minimum": float(a.min()),
            "maximum": float(a.max()),
            "negative_fraction": float((a < 0).mean()),
        }

    group_summaries = {k: summarize(v) for k, v in sorted(groups.items())}
    norms_np = np.asarray(norms)
    projections = (matrix @ mean_grad) / max(mean_norm**2, 1e-30)
    records = []
    for i, _pair in enumerate(pairs):
        records.append(
            {
                "sample_id": ids[i],
                "length": int(lengths[i]),
                "stratum": strata[i],
                "coordinate_mse": losses[i],
                "gradient_l2_norm": norms[i],
                "cosine_with_mean_joint_gradient": float(cos_mean[i]),
                "signed_projection_on_mean_gradient": float(projections[i]),
                "gradient_norm_share": float(norms[i] / max(norms_np.sum(), 1e-30)),
            }
        )
    slope_norm = float(np.polyfit(lengths, norms_np, 1)[0])
    projection_np = projections.numpy()
    slope_projection = float(np.polyfit(lengths, projection_np, 1)[0])
    after_model = serialized_sha(model.state_dict())
    after_optimizer = serialized_sha(checkpoint["optimizer"])
    if before_model != after_model or before_optimizer != after_optimizer:
        raise ValueError("read-only audit mutated model or optimizer state")
    return {
        "loss_component": "masked_xyz_coordinate_mse_only",
        "optimizer_steps": 0,
        "identity_order": ids,
        "gradient_dimension": int(matrix.shape[1]),
        "gradient_matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest(),
        "gradient_norm_by_identity": records,
        "cosine_matrix": cosine.tolist(),
        "cosine_matrix_identity_order": ids,
        "cosine_summaries": {"all_unordered_pairs": summarize(pair_rows), **group_summaries},
        "negative_gradient_pair_fraction": float((np.asarray(pair_rows) < 0).mean()),
        "mean_joint_gradient_l2_norm": mean_norm,
        "gradient_norm_vs_length_slope_per_residue": slope_norm,
        "signed_projection_vs_length_slope_per_residue": slope_projection,
        "gradient_norm_length_correlation": float(np.corrcoef(lengths, norms_np)[0, 1]),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "runtime_seconds": time.perf_counter() - started,
        "model_state_unchanged": before_model == after_model,
        "optimizer_state_unchanged": before_optimizer == after_optimizer,
    }


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    out = ROOT / cfg["output_dir"]
    if out.exists():
        raise FileExistsError(f"refusing overwrite of diagnostic output: {out}")
    if not torch.cuda.is_available():
        raise RuntimeError("focused diagnostic requires CUDA; no outputs were created")
    v2cfg = yaml.safe_load((ROOT / cfg["source_config"]).read_text())
    checkpoint, protocol, pairs, input_audit = verify_inputs(cfg)
    out.mkdir(parents=True)
    representatives = choose_representatives(cfg, pairs)
    pins = {x["sample_id"]: x for x in input_audit["panel_corruption_pins"]}
    for rep in representatives:
        rep["archive_sha256"] = pins[rep["sample_id"]]["archive_sha256"]
        rep.pop("pair")
    # Select again to retain the verified in-memory tensors for execution.
    reps = choose_representatives(cfg, pairs)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    audit = gradient_audit(checkpoint, pairs, protocol, v2cfg, device)
    arms = []
    for rep in reps:
        arms.append(train_arm("fresh_isolated", rep, cfg, checkpoint, v2cfg, device))
        arms.append(train_arm("checkpoint_specialization", rep, cfg, checkpoint, v2cfg, device))
    if file_sha(ROOT / cfg["source_checkpoint"]) != input_audit["checkpoint_sha256"]:
        raise ValueError("source update-6400 checkpoint changed during diagnostic")
    result = {
        "schema": "e010_phase2_interference_diagnostic_result_v1",
        "authorizes_downstream": False,
        "scope": "training_panel_diagnostic_only",
        "input_audit": input_audit,
        "selection": {"method": cfg["selection"]["method"], "representatives": representatives},
        "gradient_audit": audit,
        "isolated_runs": arms,
        "restrictions": {
            "coordinate_only": True,
            "geometry_losses": False,
            "priors": False,
            "sequence_features": False,
            "architecture_changes": False,
            "sampling": False,
            "held_out_data": False,
            "joint_run_continued": False,
            "supervised_pilot_prepared": False,
        },
        "interpretation_rules": {
            "fresh_isolated_failure_increasing_with_length": "long-length architectural expressivity failure",
            "isolated_pass_plus_quick_specialization_pass": "shared-capacity or gradient-interference failure",
            "negative_cross_stratum_gradient_cosines": (
                "consider stratified mini-batches or gradient accumulation before architecture changes"
            ),
            "compatible_gradients_plus_isolated_pass_and_joint_plateau": "increase shared model capacity",
            "long_isolated_pass_only_after_many_updates": "use length-aware exposure schedules",
            "quick_specialization_definition_updates": cfg["arms"]["quick_specialization_updates"],
        },
    }
    (out / "input_audit.json").write_text(json.dumps(input_audit, indent=2, sort_keys=True) + "\n")
    (out / "diagnostic_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "representatives": representatives,
                "gradient_summary": audit["cosine_summaries"]["all_unordered_pairs"],
                "arms": [
                    {
                        k: x[k]
                        for k in (
                            "kind",
                            "sample_id",
                            "length",
                            "updates_executed",
                            "first_passage_update",
                            "final_aligned_rmse_angstrom",
                        )
                    }
                    for x in arms
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
