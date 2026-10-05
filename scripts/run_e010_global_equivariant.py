#!/usr/bin/env python3
"""Isolated, bounded E010 Cartesian expressivity diagnostic lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
from scripts.run_e009_bayesian_refiner import _kabsch_rmse

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/e010_global_equivariant_expressivity_v1.yaml"
EXPECTED_CONFIG_KEYS = {
    "schema",
    "version",
    "authorizes_downstream",
    "initialization_seed",
    "repeats",
    "inputs",
    "output_dir",
    "oracle",
    "model_stage",
    "cuda_smoke",
}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_hash(array: np.ndarray, kind: str) -> str:
    dtype = {"float": "<f4", "bool": "|b1"}[kind]
    raw = np.ascontiguousarray(array.astype(dtype, copy=False)).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) != EXPECTED_CONFIG_KEYS:
        raise ValueError("E010 config key/schema mismatch")
    if cfg["schema"] != "e010_global_equivariant_expressivity_config_v1" or cfg["version"] != "v1":
        raise ValueError("unsupported E010 config version")
    if cfg["authorizes_downstream"] is not False or cfg["repeats"] != 1:
        raise ValueError("E010 must remain non-authorizing and single-repeat")
    inp = cfg["inputs"]
    if inp["length"] != 64 or inp["sample_id"] != "10px_18":
        raise ValueError("E010 requires the pinned E009 v4 length-64 input")
    if cfg["oracle"]["evaluation_schedule"] != [0, 1, 10, 50, 100, 250, 500, 1000]:
        raise ValueError("oracle evaluation schedule changed")
    if cfg["model_stage"]["evaluation_schedule"] != [0, 10, 50, 100, 250, 500, 1000]:
        raise ValueError("model evaluation schedule changed")
    if cfg["oracle"]["max_updates"] != 1000 or cfg["model_stage"]["max_updates"] != 1000:
        raise ValueError("E010 update bound changed")
    if cfg["oracle"]["aligned_rmse_gate_angstrom"] != 0.01 or cfg["model_stage"]["aligned_rmse_gate_angstrom"] != 0.10:
        raise ValueError("E010 success gates changed")
    if cfg["model_stage"]["gradient_clip_max_norm"] != 5.0 or cfg["model_stage"]["residual_l2_coefficient"] > 1e-4:
        raise ValueError("E010 normalized clipping or residual regularization policy changed")
    if cfg["model_stage"]["geometry_telemetry_thresholds"] != {
        "maximum_i_plus_1_i_plus_2_i_plus_3_distance_rmse_angstrom": 1.0,
        "maximum_chirality_inversions": 0,
    }:
        raise ValueError("descriptive geometry telemetry interpretation thresholds changed")
    return cfg


def _input_paths(cfg: dict) -> tuple[Path, Path]:
    inp = cfg["inputs"]
    return REPO / inp["fixed_corruption_cache"], REPO / inp["evaluator_path"]


def validate_inputs(cfg: dict) -> dict:
    cache_path, evaluator_path = _input_paths(cfg)
    inp = cfg["inputs"]
    if not cache_path.is_file() or sha256(cache_path) != inp["cache_sha256"]:
        raise ValueError("pinned E009 fixed-corruption cache missing or changed")
    if not evaluator_path.is_file() or sha256(evaluator_path) != inp["evaluator_sha256"]:
        raise ValueError("pinned E009 aligned-RMSE evaluator source missing or changed")
    with np.load(cache_path, allow_pickle=False) as z:
        target = np.array(z["target"], copy=True)
        coarse = np.array(z["coarse"], copy=True)
        mask = np.array(z["mask"], copy=True).astype(bool, copy=False)
        metadata = json.loads(str(z["metadata"].item()))
    if (
        metadata.get("schema") != "e009_fixed_corruption_v1"
        or metadata.get("sample_id") != inp["sample_id"]
        or metadata.get("length") != 64
    ):
        raise ValueError("fixed-corruption metadata does not match the E010 pin")
    if target.shape != (64, 3) or coarse.shape != target.shape or mask.shape != (64,) or not mask.any():
        raise ValueError("fixed-corruption tensors have invalid shapes or mask")
    if not np.isfinite(target).all() or not np.isfinite(coarse).all():
        raise ValueError("fixed-corruption tensors contain non-finite coordinates")
    observed = {
        "target_sha256": _tensor_hash(target, "float"),
        "corruption_sha256": _tensor_hash(coarse, "float"),
        "mask_sha256": _tensor_hash(mask, "bool"),
    }
    for name, digest in observed.items():
        if digest != inp[name]:
            raise ValueError(f"fixed E009 tensor pin mismatch: {name}")
    return {
        "cache_path": str(cache_path.relative_to(REPO)),
        "cache_sha256": inp["cache_sha256"],
        "sample_id": metadata["sample_id"],
        "target_sha256": observed["target_sha256"],
        "corruption_sha256": observed["corruption_sha256"],
        "mask_sha256": observed["mask_sha256"],
        "evaluator_path": str(evaluator_path.relative_to(REPO)),
        "evaluator_sha256": inp["evaluator_sha256"],
        "input_validation": "passed_read_only",
    }


def _read_datum(cfg: dict, device: torch.device | str = "cpu") -> dict:
    validate_inputs(cfg)
    cache_path, _ = _input_paths(cfg)
    with np.load(cache_path, allow_pickle=False) as z:
        return {
            "target": torch.from_numpy(np.array(z["target"], copy=True)).to(device=device, dtype=torch.float32),
            "corrupt": torch.from_numpy(np.array(z["coarse"], copy=True)).to(device=device, dtype=torch.float32),
            "mask": torch.from_numpy(np.array(z["mask"], copy=True)).to(device=device, dtype=torch.bool),
            "metadata": json.loads(str(z["metadata"].item())),
        }


def _new_model(cfg: dict, device: torch.device | str) -> GlobalEquivariantResidual:
    m = cfg["model_stage"]["model"]
    return GlobalEquivariantResidual(**m).to(device)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def plan(cfg: dict, cfg_path: Path) -> dict:
    pins = validate_inputs(cfg)
    torch.manual_seed(cfg["initialization_seed"])
    model = _new_model(cfg, "cpu")
    params = _parameter_count(model)
    if not 1_000_000 <= params <= 3_000_000:
        raise ValueError(f"E010 model parameter count outside required 1–3M range: {params}")
    return {
        "schema": "e010_plan_v1",
        "authorizes_downstream": False,
        "config_path": str(cfg_path.relative_to(REPO)),
        "config_sha256": sha256(cfg_path),
        "inputs": pins,
        "model_parameter_count": params,
        "planned_stages": [
            "read-only validation",
            "bounded CUDA smoke",
            "free-coordinate oracle",
            "global-model overfit only if oracle passes",
        ],
        "future_commands": commands(),
        "outputs": ["protocol.json", "oracle_result.json", "model_result.json", "report.json"],
        "checkpoint_policy": (
            "no resume or checkpoints are implemented; stage updates are bounded, "
            "results publish atomically at completion"
        ),
    }


def commands() -> list[str]:
    return [
        (
            "PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config "
            "configs/e010_global_equivariant_expressivity_v1.yaml --plan-only"
        ),
        (
            "PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config "
            "configs/e010_global_equivariant_expressivity_v1.yaml --validate-only"
        ),
        (
            "PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config "
            "configs/e010_global_equivariant_expressivity_v1.yaml --cuda-smoke"
        ),
        (
            "PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config "
            "configs/e010_global_equivariant_expressivity_v1.yaml --oracle"
        ),
        (
            "PYTHONPATH=src:. python scripts/run_e010_global_equivariant.py --config "
            "configs/e010_global_equivariant_expressivity_v1.yaml --model-overfit"
        ),
    ]


def _rotation(device: torch.device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn((3, 3), generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) < 0:
        q[:, -1] *= -1
    shift = torch.tensor((2.5, -1.25, 4.0))
    return q.to(device), shift.to(device)


@torch.no_grad()
def equivariance_error(model: GlobalEquivariantResidual, coords: torch.Tensor, mask: torch.Tensor, seed: int) -> dict:
    was_training = model.training
    model.eval()
    q, shift = _rotation(coords.device, seed)
    base = model(coords, mask)["prediction"]
    transformed = model(coords @ q + shift, mask)["prediction"]
    expected = base @ q + shift
    valid = mask.bool()[..., None].expand_as(base)
    error = (transformed - expected)[valid]
    if was_training:
        model.train()
    return {"max_abs_angstrom": float(error.abs().max()), "rmse_angstrom": float(error.square().mean().sqrt())}


def _unaligned_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    err2 = (pred - target).square().sum(dim=-1)
    return float((err2[mask.bool()].mean()).sqrt())


def _geometry_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask.bool()
    out = {}
    for sep in (1, 2, 3):
        pair_valid = valid[sep:] & valid[:-sep]
        pd = torch.linalg.vector_norm(pred[sep:] - pred[:-sep], dim=-1)
        td = torch.linalg.vector_norm(target[sep:] - target[:-sep], dim=-1)
        out[f"i_plus_{sep}_distance_rmse_angstrom"] = float((pd[pair_valid] - td[pair_valid]).square().mean().sqrt())
    v1 = pred[1:-2] - pred[:-3]
    v2 = pred[2:-1] - pred[1:-2]
    v3 = pred[3:] - pred[2:-1]
    pvol = (torch.cross(v1, v2, dim=-1) * v3).sum(-1)
    t1 = target[1:-2] - target[:-3]
    t2 = target[2:-1] - target[1:-2]
    t3 = target[3:] - target[2:-1]
    tvol = (torch.cross(t1, t2, dim=-1) * t3).sum(-1)
    qmask = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
    out["chirality_inversions"] = int(((pvol * tvol < 0) & qmask).sum())
    out["eligible_chirality_quadruplets"] = int(qmask.sum())
    return out


def _evaluate_model(model: GlobalEquivariantResidual, datum: dict, step: int, seed: int) -> dict:
    with torch.no_grad():
        prediction = model(datum["corrupt"][None], datum["mask"][None])["prediction"][0]
    target, mask = datum["target"], datum["mask"]
    # This is the exact E009 v4 evaluator function, pinned by source hash in the config.
    aligned = float(_kabsch_rmse(prediction[mask], target[mask]))
    unaligned = _unaligned_rmse(prediction, target, mask)
    errors = torch.linalg.vector_norm(prediction - target, dim=-1)[mask]
    idx = torch.arange(errors.numel(), device=errors.device, dtype=torch.float32)
    slope = float(torch.linalg.lstsq(torch.stack((idx, torch.ones_like(idx)), dim=-1), errors).solution[0])
    quantiles = torch.quantile(errors, torch.tensor((0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), device=errors.device))
    return {
        "update": step,
        "aligned_coordinate_rmse_angstrom": aligned,
        "unaligned_coordinate_rmse_angstrom": unaligned,
        "per_residue_error_angstrom": errors.cpu().tolist(),
        "per_residue_error_distribution_angstrom": {
            "min_p10_p25_median_p75_p90_max": [float(x) for x in quantiles],
            "mean": float(errors.mean()),
            "std": float(errors.std(unbiased=False)),
        },
        "error_vs_residue_index_slope_angstrom_per_residue": slope,
        "geometry_telemetry": _geometry_metrics(prediction, target, mask),
        "finite_prediction": bool(torch.isfinite(prediction).all()),
        "equivariance_error": equivariance_error(model, datum["corrupt"][None], datum["mask"][None], seed + 991),
    }


def _memory_snapshot(device: torch.device) -> dict:
    cpu_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    result = {"peak_process_cpu_rss_bytes": cpu_bytes}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update(
            {
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    else:
        result.update({"peak_cuda_allocated_bytes": None, "peak_cuda_reserved_bytes": None})
    return result


def _publish_stage(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"E010 artifact already exists; refusing overwrite: {path}")
    _atomic_json(path, payload)


def _publish_report(cfg: dict, oracle: dict, model_result: dict | None, inputs: dict) -> None:
    passed_oracle = oracle["status"] == "passed"
    if not passed_oracle:
        interpretation = "evaluation_or_optimization_pipeline_failure"
    elif model_result is None:
        raise ValueError("oracle passed; model result is required before report publication")
    elif model_result["status"] == "passed" and not model_result["geometry_acceptable"]:
        interpretation = (
            "global_model_passes_with_poor_geometry_telemetry_add_learned_prior_only_as_weak_regularizer_later"
        )
    elif model_result["status"] == "passed":
        interpretation = "E009_local_representation_and_objective_were_inadequate_proceed_toward_global_cartesian_model"
    else:
        interpretation = "representation_or_equivariant_architecture_still_lacks_capacity"
    payload = {
        "schema": "e010_global_equivariant_expressivity_report_v1",
        "authorizes_downstream": False,
        "inputs": inputs,
        "oracle_status": oracle["status"],
        "model_status": None if model_result is None else model_result["status"],
        "interpretation": interpretation,
        "predeclared_gates": {
            "oracle_aligned_rmse_angstrom": 0.01,
            "model_aligned_rmse_angstrom": 0.10,
            "model_geometry_telemetry_is_descriptive": True,
        },
        "oracle_result": "oracle_result.json",
        "model_result": "model_result.json" if model_result is not None else None,
        "lifecycle": "non-authorizing; no E009 artifacts modified; no checkpoint/resume implemented",
    }
    _publish_stage(REPO / cfg["output_dir"] / "report.json", payload)


def run_oracle(cfg: dict, cfg_path: Path, device: torch.device) -> dict:
    inputs = validate_inputs(cfg)
    out = REPO / cfg["output_dir"]
    result_path = out / "oracle_result.json"
    if result_path.exists():
        raise FileExistsError(f"E010 oracle result already exists: {result_path}")
    datum = _read_datum(cfg, device)
    torch.manual_seed(cfg["initialization_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["initialization_seed"])
        torch.cuda.reset_peak_memory_stats(device)
    delta = torch.nn.Parameter(torch.zeros_like(datum["corrupt"])[None])
    opt = torch.optim.Adam([delta], lr=cfg["oracle"]["learning_rate"])
    mask3 = datum["mask"][None, :, None].to(datum["target"].dtype)
    schedule = cfg["oracle"]["evaluation_schedule"]
    records, updates = [], 0
    started = time.perf_counter()
    finite_loss, finite_grad = True, True
    for step in range(cfg["oracle"]["max_updates"] + 1):
        if step in schedule:
            prediction = datum["corrupt"][None] + delta * mask3
            rmse = float(_kabsch_rmse(prediction[0, datum["mask"]], datum["target"][datum["mask"]]))
            records.append(
                {
                    "update": step,
                    "aligned_coordinate_rmse_angstrom": rmse,
                    "target_frame_coordinate_mse": float(
                        ((prediction - datum["target"][None]).square() * mask3).sum() / (3 * datum["mask"].sum())
                    ),
                    "finite_prediction": bool(torch.isfinite(prediction).all()),
                }
            )
            updates = step
            if rmse <= cfg["oracle"]["aligned_rmse_gate_angstrom"]:
                break
        if step == cfg["oracle"]["max_updates"]:
            break
        opt.zero_grad(set_to_none=True)
        prediction = datum["corrupt"][None] + delta * mask3
        loss = (((prediction - datum["target"][None]).square()) * mask3).sum() / (3 * datum["mask"].sum())
        finite_loss = finite_loss and bool(torch.isfinite(loss))
        if not finite_loss:
            break
        loss.backward()
        finite_grad = finite_grad and bool(delta.grad is not None and torch.isfinite(delta.grad).all())
        if not finite_grad:
            break
        opt.step()
    status = (
        "passed"
        if records
        and records[-1]["aligned_coordinate_rmse_angstrom"] <= cfg["oracle"]["aligned_rmse_gate_angstrom"]
        and finite_loss
        and finite_grad
        else "failed"
    )
    failure_class = None if status == "passed" else "evaluation_or_optimization_pipeline_failure"
    result = {
        "schema": "e010_free_coordinate_oracle_result_v1",
        "authorizes_downstream": False,
        "status": status,
        "failure_class": failure_class,
        "input_pins": inputs,
        "config_sha256": sha256(cfg_path),
        "seed": cfg["initialization_seed"],
        "optimizer": {"name": "Adam", "learning_rate": cfg["oracle"]["learning_rate"]},
        "objective": "masked target-frame Cartesian MSE only; one learnable delta [1,N,3]",
        "updates": updates,
        "runtime_seconds": time.perf_counter() - started,
        "finite_loss": finite_loss,
        "finite_gradients": finite_grad,
        "records": records,
        "memory": _memory_snapshot(device),
    }
    _publish_stage(result_path, result)
    if status == "failed":
        _publish_report(cfg, result, None, inputs)
    return result


def _model_geometry_acceptable(record: dict, cfg: dict) -> bool:
    geo = record["geometry_telemetry"]
    # Descriptive thresholds are predeclared solely to select the interpretation branch.
    thresholds = cfg["model_stage"]["geometry_telemetry_thresholds"]
    return (
        geo["chirality_inversions"] <= thresholds["maximum_chirality_inversions"]
        and max(geo[f"i_plus_{k}_distance_rmse_angstrom"] for k in (1, 2, 3))
        <= thresholds["maximum_i_plus_1_i_plus_2_i_plus_3_distance_rmse_angstrom"]
    )


def run_model_overfit(cfg: dict, cfg_path: Path, device: torch.device) -> dict:
    out = REPO / cfg["output_dir"]
    oracle_path = out / "oracle_result.json"
    result_path = out / "model_result.json"
    if not oracle_path.is_file():
        raise FileNotFoundError("model-overfit is gated on a completed E010 oracle result")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    if oracle.get("authorizes_downstream") is not False or oracle.get("status") != "passed":
        raise RuntimeError("oracle did not pass; E010 model stage is fail-closed")
    if oracle.get("config_sha256") != sha256(cfg_path):
        raise ValueError("oracle config identity differs from the current E010 config")
    if result_path.exists():
        raise FileExistsError(f"E010 model result already exists: {result_path}")
    inputs = validate_inputs(cfg)
    datum = _read_datum(cfg, device)
    seed = cfg["initialization_seed"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = _new_model(cfg, device)
    count = _parameter_count(model)
    if not 1_000_000 <= count <= 3_000_000:
        raise ValueError(f"model parameter count outside required range: {count}")
    settings = cfg["model_stage"]
    opt = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    schedule = settings["evaluation_schedule"]
    records, update_log, finite_loss, finite_grad = [], [], True, True
    started = time.perf_counter()
    updates = 0
    for step in range(settings["max_updates"] + 1):
        if step in schedule:
            record = _evaluate_model(model, datum, step, seed)
            records.append(record)
            updates = step
            if record["aligned_coordinate_rmse_angstrom"] <= settings["aligned_rmse_gate_angstrom"]:
                break
        if step == settings["max_updates"]:
            break
        opt.zero_grad(set_to_none=True)
        result = model(datum["corrupt"][None], datum["mask"][None])
        delta = result["delta"][0]
        valid = datum["mask"][:, None].to(delta.dtype)
        coordinate_mse = (((result["prediction"][0] - datum["target"]).square()) * valid).sum() / (
            3 * datum["mask"].sum()
        )
        residual_penalty = (delta.square() * valid).sum() / (3 * datum["mask"].sum())
        loss = coordinate_mse + settings["residual_l2_coefficient"] * residual_penalty
        finite_loss = finite_loss and bool(torch.isfinite(loss))
        if not finite_loss:
            break
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        pre = math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients))
        this_finite_grad = all(bool(torch.isfinite(g).all()) for g in gradients)
        finite_grad = finite_grad and this_finite_grad
        if not this_finite_grad:
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_max_norm"])
        post = math.sqrt(
            sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)
        )
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        step_norm = math.sqrt(
            sum(
                float((p.detach() - old).double().square().sum())
                for p, old in zip(model.parameters(), before, strict=True)
            )
        )
        update_log.append(
            {
                "update": step + 1,
                "unclipped_gradient_l2_norm": pre,
                "clipped_gradient_l2_norm": post,
                "clip_activated": pre > settings["gradient_clip_max_norm"],
                "effective_optimizer_step_l2_norm": step_norm,
                "finite_loss": bool(torch.isfinite(loss)),
                "finite_gradients": this_finite_grad,
            }
        )
    status = (
        "passed"
        if records
        and records[-1]["aligned_coordinate_rmse_angstrom"] <= settings["aligned_rmse_gate_angstrom"]
        and finite_loss
        and finite_grad
        else "failed"
    )
    clipping_fraction = sum(x["clip_activated"] for x in update_log) / len(update_log) if update_log else 0.0
    geometry_ok = _model_geometry_acceptable(records[-1], cfg) if records else False
    result = {
        "schema": "e010_global_model_result_v1",
        "authorizes_downstream": False,
        "status": status,
        "input_pins": inputs,
        "config_sha256": sha256(cfg_path),
        "seed": seed,
        "device": str(device),
        "parameter_count": count,
        "repeats": 1,
        "objective": {
            "coordinate_mse": "mean over eligible xyz scalars",
            "residual_l2_coefficient": settings["residual_l2_coefficient"],
            "geometry_prior_or_geometry_loss": False,
            "fixed_bond_length": False,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": settings["learning_rate"],
            "weight_decay": settings["weight_decay"],
        },
        "gradient_clipping": {
            "policy": settings["clipping_policy"],
            "max_norm": settings["gradient_clip_max_norm"],
            "fraction_activated": clipping_fraction,
            "updates": len(update_log),
        },
        "updates": updates,
        "runtime_seconds": time.perf_counter() - started,
        "finite_loss": finite_loss,
        "finite_gradients": finite_grad,
        "records": records,
        "per_update_optimization": update_log,
        "geometry_acceptable": geometry_ok,
        "memory": _memory_snapshot(device),
    }
    _publish_stage(result_path, result)
    _publish_report(cfg, oracle, result, inputs)
    return result


def run_cuda_smoke(cfg: dict, cfg_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-smoke requires an available CUDA device")
    device = torch.device("cuda")
    inputs = validate_inputs(cfg)
    datum = _read_datum(cfg, device)
    seed = cfg["initialization_seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    model = _new_model(cfg, device)
    params = _parameter_count(model)
    if not 1_000_000 <= params <= 3_000_000:
        raise ValueError(f"model parameter count outside required range: {params}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["model_stage"]["learning_rate"])
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    result = model(datum["corrupt"][None], datum["mask"][None])
    valid = datum["mask"][None, :, None].to(result["prediction"].dtype)
    loss = (((result["prediction"] - datum["target"][None]).square()) * valid).sum() / (3 * datum["mask"].sum())
    loss.backward()
    pre = math.sqrt(
        sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["model_stage"]["gradient_clip_max_norm"])
    post = math.sqrt(
        sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)
    )
    opt.step()
    torch.cuda.synchronize(device)
    eq_error = equivariance_error(model, datum["corrupt"][None], datum["mask"][None], seed + 19)
    payload = {
        "schema": "e010_cuda_smoke_v1",
        "authorizes_downstream": False,
        "status": "passed"
        if torch.isfinite(loss)
        and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        and eq_error["max_abs_angstrom"] <= 1e-4
        else "failed",
        "config_sha256": sha256(cfg_path),
        "input_pins": inputs,
        "device": torch.cuda.get_device_name(device),
        "parameter_count": params,
        "optimizer_steps": 1,
        "finite_loss": bool(torch.isfinite(loss)),
        "gradient_norm_before_clip": pre,
        "gradient_norm_after_clip": post,
        "equivariance_error": eq_error,
        "runtime_seconds": time.perf_counter() - started,
        "memory": _memory_snapshot(device),
    }
    _publish_stage(REPO / cfg["output_dir"] / "cuda_smoke_result.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan-only", "validate-only", "cuda-smoke", "oracle", "model-overfit"):
        modes.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO / args.config
    cfg = load_config(cfg_path)
    if args.plan_only:
        print(json.dumps(plan(cfg, cfg_path), indent=2, sort_keys=True))
        return
    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema": "e010_read_only_validation_v1",
                    "authorizes_downstream": False,
                    "config_sha256": sha256(cfg_path),
                    "inputs": validate_inputs(cfg),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.cuda_smoke:
        result = run_cuda_smoke(cfg, cfg_path)
    elif args.oracle:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = run_oracle(cfg, cfg_path, device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result = run_model_overfit(cfg, cfg_path, device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
