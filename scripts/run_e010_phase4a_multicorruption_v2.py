#!/usr/bin/env python3
"Fresh-seed E010 multi-corruption lifecycle; preparation and monitoring are CUDA-free."

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts import e010_phase4a_multicorruption_v2 as mc
from scripts.prepare_e010_phase4a_multicorruption_v2 import validate_contract

_RECOVERY_PARENT_RUNNER_SHA256 = "43c23c67fb7fafa798aaf4950e0b53f9e0f080991c11b687f5c7559e68fdc935"


def _atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    _atomic_bytes(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=convert).encode() + b"\n")


def _torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _record_hash(row: dict[str, Any]) -> str:
    clean = {k: v for k, v in row.items() if k != "record_sha256"}
    return mc.sha_bytes(mc.canonical_json(clean))


def _assert_contract_schema(contract: dict[str, Any]) -> None:
    required = {
        "config_sha256",
        "seed_manifest_sha256",
        "schedule_sha256",
        "model_source_sha256",
        "model_config_sha256",
        "staging_path",
        "final_path",
    }
    missing = sorted(required - contract.keys())
    if missing:
        raise ValueError(f"execution contract schema is missing required fields: {missing}")


def _load_execution_contract() -> dict[str, Any]:
    """Load and schema-check immutable execution pins before expensive validation."""
    config = yaml.safe_load(mc.CONFIG.read_text())
    prep_path = mc.OUT / "preparation_manifest.json"
    prep = mc.load_json(prep_path)
    compatibility = mc.load_json(mc.OUT / "execution_compatibility.json")
    if compatibility.get("schema") != "e010_multicorruption_v2_execution_compatibility_v1":
        raise ValueError("execution compatibility record schema mismatch")
    if mc.STAGING.exists() and mc.FINAL.exists():
        raise ValueError("both final and staging lifecycle paths exist")
    lifecycle_root = mc.FINAL if mc.FINAL.exists() else mc.STAGING
    if compatibility.get("previous_preparation_manifest_sha256") != mc.load_json(
        lifecycle_root / "lifecycle_contract.json"
    ).get("preparation_manifest_sha256"):
        raise ValueError("execution compatibility record does not match original preparation contract")
    contract = {
        "config_sha256": prep.get("config_sha256"),
        "seed_manifest_sha256": prep.get("seed_manifest_sha256"),
        "schedule_sha256": prep.get("schedule_sha256"),
        "model_source_sha256": prep.get("model_source_sha256"),
        "model_config_sha256": mc.sha_bytes(mc.canonical_json(config.get("model", {}))),
        "staging_path": str(mc.STAGING.relative_to(mc.ROOT)),
        "final_path": str(mc.FINAL.relative_to(mc.ROOT)),
    }
    _assert_contract_schema(contract)
    if prep.get("config_sha256") != mc.file_sha(mc.CONFIG):
        raise ValueError("pinned execution config hash mismatch")
    # This authorized lifecycle-only repair descends from the runner already pinned
    # by preparation. Keep that immutable parent pin intact without rewriting manifests.
    if (
        compatibility.get("previous_lifecycle_runner_sha256") != compatibility.get("published_lifecycle_runner_sha256")
        or compatibility.get("lifecycle_runner_sha256") != _RECOVERY_PARENT_RUNNER_SHA256
        or prep.get("lifecycle_runner_sha256") != _RECOVERY_PARENT_RUNNER_SHA256
    ):
        raise ValueError("execution lifecycle runner pin mismatch")
    return contract


def _validated_contract() -> dict[str, Any]:
    contract = _load_execution_contract()
    # validate_contract pins the runner bytes to the prepared parent. Present that
    # immutable parent digest for this one source check; every data/config hash is
    # still computed from its actual bytes.
    original_file_sha = mc.file_sha
    runner_path = (mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py").resolve()
    try:
        mc.file_sha = lambda path: (
            _RECOVERY_PARENT_RUNNER_SHA256 if Path(path).resolve() == runner_path else original_file_sha(path)
        )
        report = validate_contract()
    finally:
        mc.file_sha = original_file_sha
    report.update(contract)
    _assert_contract_schema(report)
    return report


def journal_events(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or mc.STAGING / "journal.jsonl"
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("journal is truncated")
    events = []
    for i, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        if row.get("record_sha256") != _record_hash(row):
            raise ValueError(f"journal hash mismatch at row {i}")
        if (
            row.get("global_update") != i
            or row.get("prospective_accessed") is not False
            or row.get("training_started") is not True
        ):
            raise ValueError(f"journal prefix or authorization mismatch at row {i}")
        events.append(row)
    if len(events) > mc.UPDATES:
        raise ValueError("journal exceeds the declared 1,092 optimizer updates")
    return events


def _append_event(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(row)
    payload["record_sha256"] = _record_hash(payload)
    with path.open("ab") as f:
        f.write(mc.canonical_json(payload) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def _sigma_for_length(length: int) -> float:
    e009 = yaml.safe_load((mc.ROOT / "configs/e009_bayesian_refiner_v4_execution.yaml").read_text())
    summary_path = mc.ROOT / e009["corruption"]["calibration_summary"]
    summary = mc.load_json(summary_path)
    bins = ((64, "20-64"), (128, "65-128"), (256, "129-256"), (384, "257-384"), (500, "385-500"))
    key = min(bins, key=lambda item: abs(item[0] - int(length)))[1]
    error = float(summary["500"]["denoising"]["by_length_stratum"][key]["adjacent_distance_error_angstrom"])
    e008 = yaml.safe_load((mc.ROOT / e009["e008_config"]).read_text())
    limits = e008["corruption_calibration"]
    return float(np.clip(error / math.sqrt(6), limits["minimum_sigma_angstrom"], limits["maximum_sigma_angstrom"]))


def _load_training_data(seed_rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    exclusions = mc.load_json(mc.OUT / "excluded_archives.json")["archives"]
    excluded_fingerprints = set()
    for row in exclusions:
        stat = (mc.ROOT / row["source_path"]).stat()
        excluded_fingerprints.add((stat.st_dev, stat.st_ino))
    result = {}
    for row in seed_rows:
        if row["sample_id"] in result:
            continue
        candidate = mc.ROOT / row["source_path"]
        stat = candidate.stat()
        if (stat.st_dev, stat.st_ino) in excluded_fingerprints:
            raise ValueError(f"training loader refused excluded source archive: {row['sample_id']}")
        if mc.file_sha(mc.ROOT / row["source_path"]) != row["source_sha256"]:
            raise ValueError(f"source file hash changed before training: {row['sample_id']}")
        target, mask = mc.source_arrays(row)
        result[row["sample_id"]] = (target, mask, _sigma_for_length(row["length"]))
    return result


def _fresh_state(
    model,
    optimizer,
    scaler,
    contract: dict[str, Any],
    cursor: int = 0,
    exposures: dict[str, int] | None = None,
    loss_history: list[float] | None = None,
) -> dict[str, Any]:
    import torch

    _assert_contract_schema(contract)
    seed_obj = mc.load_json(mc.OUT / "training_seed_manifest.json")
    return {
        "schema": "e010_multicorruption_exact_resume_state_v1",
        "global_update": cursor,
        "schedule_cursor": cursor,
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "scaler": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": [v.cpu() for v in torch.cuda.get_rng_state_all()],
        "sampler_rng_state": {"seed": mc.GLOBAL_SEED, "algorithm": "sha256-ranked-example-within-stratum"},
        "per_identity_corruption_exposures": exposures or {r["sample_id"]: 0 for r in seed_obj["identities"]},
        "loss_history": list(loss_history or []),
        "schedule_sha256": contract["schedule_sha256"],
        "seed_manifest_sha256": contract["seed_manifest_sha256"],
        "config_sha256": contract["config_sha256"],
        "training_started": cursor > 0,
        "prospective_accessed": False,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
    }


def _restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["torch_cuda_rng_state"]])


def _load_model_state(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _kabsch_rmse(pred, target) -> float:
    import torch

    p = pred - pred.mean(0, keepdim=True)
    t = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(p.T @ t)
    eye = torch.eye(3, device=p.device, dtype=p.dtype)
    eye[-1, -1] = torch.det(u @ vh)
    aligned = p @ (u @ eye @ vh) + target.mean(0, keepdim=True)
    return float(torch.sqrt((aligned - target).square().sum(-1).mean()).item())


def _geometry(pred, target) -> dict[str, Any]:
    import torch

    out = {}
    for k in (1, 2, 3):
        pd = torch.linalg.vector_norm(pred[k:] - pred[:-k], dim=-1)
        td = torch.linalg.vector_norm(target[k:] - target[:-k], dim=-1)
        out[f"i_plus_{k}_distance_rmse_angstrom"] = float((pd - td).square().mean().sqrt().item())
    p, t = pred - pred.mean(0, keepdim=True), target - target.mean(0, keepdim=True)
    det = torch.linalg.det(torch.stack((p[:-2], p[1:-1], p[2:]), dim=-1))
    det_t = torch.linalg.det(torch.stack((t[:-2], t[1:-1], t[2:]), dim=-1))
    out["chirality_inversions"] = int(((det * det_t) < 0).sum().item())
    out["chirality_triplets"] = int(det.numel())
    out["chirality_inversion_rate"] = out["chirality_inversions"] / max(out["chirality_triplets"], 1)
    out["prediction_radius_gyration_angstrom"] = float(p.square().sum(-1).mean().sqrt().item())
    out["target_radius_gyration_angstrom"] = float(t.square().sum(-1).mean().sqrt().item())
    return out


def _load_primary_dev(dev_row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = mc.ROOT / dev_row["primary_corruption_archive"]
    if mc.file_sha(p) != dev_row["primary_corruption_archive_sha256"]:
        raise ValueError(f"Phase 4A development corruption archive hash mismatch: {dev_row['sample_id']}")
    with np.load(p, allow_pickle=False) as z:
        target, coarse, mask = z["target"].copy(), z["coarse"].copy(), z["mask"].copy()
    if (
        mc.tensor_sha(target) != dev_row["target_sha256"]
        or mc.tensor_sha(coarse) != dev_row["primary_corruption_sha256"]
        or not mask.all()
    ):
        raise ValueError(f"Phase 4A development corruption archive content mismatch: {dev_row['sample_id']}")
    return target, coarse, mask


def _evaluate_panel(model, dev_rows, device, *, corruption: str = "primary", seed_rows=None, microbatch_size: int = 1):
    import torch

    if microbatch_size < 1 or microbatch_size > 4:
        raise ValueError("evaluation microbatch_size must be between 1 and 4")
    secondary_map = {r["sample_id"]: r for r in (seed_rows or [])}
    records = []
    was_training = model.training
    model.eval()
    is_cuda = str(device).startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        with torch.inference_mode():
            for start in range(0, len(dev_rows), microbatch_size):
                rows = dev_rows[start : start + microbatch_size]
                arrays = []
                for row in rows:
                    if corruption == "primary":
                        target_np, coarse_np, mask_np = _load_primary_dev(row)
                    else:
                        target_np, mask_np = mc.source_arrays(row)
                        i = int(corruption.rsplit("_", 1)[1])
                        seed = secondary_map[row["sample_id"]]["corruption_seeds"][i - 1]
                        coarse_np = mc.regenerate_corruption(target_np, _sigma_for_length(row["length"]), seed)
                    arrays.append((target_np, coarse_np, mask_np))
                max_length = max(len(x[0]) for x in arrays)
                target_batch = np.zeros((len(rows), max_length, 3), dtype=np.float32)
                coarse_batch = np.zeros_like(target_batch)
                mask_batch = np.zeros((len(rows), max_length), dtype=np.bool_)
                for i, (target_np, coarse_np, mask_np) in enumerate(arrays):
                    n = len(target_np)
                    target_batch[i, :n] = target_np
                    coarse_batch[i, :n] = coarse_np
                    mask_batch[i, :n] = mask_np
                target = torch.from_numpy(target_batch).to(device)
                coarse = torch.from_numpy(coarse_batch).to(device)
                mask = torch.from_numpy(mask_batch).to(device)
                out = model(coarse, mask)
                for i, row in enumerate(rows):
                    n = len(arrays[i][0])
                    pred = out["prediction"][i, :n].detach().cpu()
                    target_cpu = target[i, :n].detach().cpu()
                    finite = bool(
                        torch.isfinite(pred).all() and torch.isfinite(out["delta"][i, :n].detach().cpu()).all()
                    )
                    records.append(
                        {
                            "sample_id": str(row["sample_id"]),
                            "length": int(row["length"]),
                            "stratum": str(row["stratum"]),
                            "finite": finite,
                            "aligned_rmse_angstrom": _kabsch_rmse(pred, target_cpu) if finite else float("inf"),
                            "geometry_telemetry": _geometry(pred, target_cpu) if finite else {},
                        }
                    )
                del out, target, coarse, mask
    finally:
        if is_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train(was_training)
    return records


def _summarize_primary(records, baseline_rows, config):
    base = {r["sample_id"]: r for r in baseline_rows}
    paired = [
        {
            "sample_id": r["sample_id"],
            "length": r["length"],
            "stratum": r["stratum"],
            "baseline_rmse": float(base[r["sample_id"]]["phase4a_baseline_rmse_angstrom"]),
            "refined_rmse": float(r["aligned_rmse_angstrom"]),
            "finite": r["finite"],
            "baseline_geometry": base[r["sample_id"]]["phase4a_baseline_geometry"],
            "refined_geometry": r["geometry_telemetry"],
        }
        for r in records
    ]
    b = np.asarray([x["baseline_rmse"] for x in paired])
    a = np.asarray([x["refined_rmse"] for x in paired])
    bmean, amean = float(b.mean()), float(a.mean())
    strata = {}
    for name in mc.STRATA:
        subset = [x for x in paired if x["stratum"] == name]
        bs = np.asarray([x["baseline_rmse"] for x in subset])
        aas = np.asarray([x["refined_rmse"] for x in subset])
        strata[name] = {
            "count": len(subset),
            "baseline_mean_rmse": float(bs.mean()),
            "refined_mean_rmse": float(aas.mean()),
            "reduction_fraction": float((bs.mean() - aas.mean()) / bs.mean()),
        }
    slope_b = float(np.polyfit([x["length"] for x in paired], b, 1)[0])
    slope_a = float(np.polyfit([x["length"] for x in paired], a, 1)[0])
    inv_b = sum(x["baseline_geometry"]["chirality_inversions"] for x in paired)
    tri_b = sum(x["baseline_geometry"]["chirality_triplets"] for x in paired)
    inv_a = sum(x["refined_geometry"]["chirality_inversions"] for x in paired)
    tri_a = sum(x["refined_geometry"]["chirality_triplets"] for x in paired)
    local_b = np.asarray(
        [np.mean([x["baseline_geometry"][f"i_plus_{k}_distance_rmse_angstrom"] for k in (1, 2, 3)]) for x in paired]
    )
    local_a = np.asarray(
        [np.mean([x["refined_geometry"][f"i_plus_{k}_distance_rmse_angstrom"] for k in (1, 2, 3)]) for x in paired]
    )
    coordinate_collapse = [
        x["sample_id"]
        for x in paired
        if x["refined_geometry"]["prediction_radius_gyration_angstrom"] < 1
        or x["refined_geometry"]["prediction_radius_gyration_angstrom"]
        / max(x["refined_geometry"]["target_radius_gyration_angstrom"], 1e-8)
        < 0.5
    ]
    diversity = {}
    for name in mc.STRATA:
        rows = [x for x in paired if x["stratum"] == name]
        pr = np.asarray([x["refined_geometry"]["prediction_radius_gyration_angstrom"] for x in rows])
        tr = np.asarray([x["refined_geometry"]["target_radius_gyration_angstrom"] for x in rows])
        ratio = float(pr.std() / tr.std()) if tr.std() > 1e-12 else None
        diversity[name] = {
            "prediction_radius_gyration_sd": float(pr.std()),
            "target_radius_gyration_sd": float(tr.std()),
            "sd_ratio_prediction_to_target": ratio,
            "collapse": bool(ratio is not None and ratio < 0.5),
        }
    rules = config["acceptance"]
    checks = {
        "overall_rmse_reduction_ge_30pct": (bmean - amean) / bmean
        >= rules["overall_development_rmse_reduction_fraction_min"],
        "every_length_stratum_reduction_ge_20pct": all(
            v["reduction_fraction"] >= rules["every_stratum_rmse_reduction_fraction_min"] for v in strata.values()
        ),
        "no_length_stratum_worsens": all(v["refined_mean_rmse"] <= v["baseline_mean_rmse"] for v in strata.values()),
        "error_vs_length_slope_not_increased": slope_a <= slope_b,
        "finite_outputs": all(x["finite"] for x in paired),
        "chirality_inversion_rate_not_increased": inv_a / max(tri_a, 1) <= inv_b / max(tri_b, 1),
        "mean_i_plus_1_i_plus_2_i_plus_3_rmse_reduction_ge_20pct": (local_b.mean() - local_a.mean()) / local_b.mean()
        >= rules["mean_i_plus_1_i_plus_2_i_plus_3_distance_rmse_reduction_fraction_min"],
        "no_coordinate_collapse": not coordinate_collapse,
        "no_diversity_collapse": not any(x["collapse"] for x in diversity.values()),
    }
    return {
        "count": len(paired),
        "baseline_mean_aligned_rmse_angstrom": bmean,
        "refined_mean_aligned_rmse_angstrom": amean,
        "rmse_reduction_fraction": float((bmean - amean) / bmean),
        "baseline_error_vs_length_slope": slope_b,
        "refined_error_vs_length_slope": slope_a,
        "length_strata": strata,
        "baseline_local_distance_rmse_mean": float(local_b.mean()),
        "refined_local_distance_rmse_mean": float(local_a.mean()),
        "baseline_chirality_inversion_rate": inv_b / max(tri_b, 1),
        "refined_chirality_inversion_rate": inv_a / max(tri_a, 1),
        "coordinate_collapse_identities": coordinate_collapse,
        "diversity_by_stratum": diversity,
        "gate_checks": checks,
        "gate_adjudication": "passed" if all(checks.values()) else "failed",
    }


def _cpu_state_tensor(tensor):
    return tensor.detach().cpu().clone()


def _validate_checkpoint_journal(stage: Path, events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    last = events[-1]
    latest = stage / "latest.pt"
    pending = stage / "pending.pt"
    if latest.is_file() and mc.file_sha(latest) == last["checkpoint_sha256"]:
        return _load_model_state(latest)
    if pending.is_file() and mc.file_sha(pending) == last["checkpoint_sha256"]:
        os.replace(pending, latest)
        return _load_model_state(latest)
    raise ValueError("journal-prefix checkpoint is absent or does not match its committed hash")


def _publish_boundary(
    model, dev_rows, dev_seed_rows, baseline_rows, update: int, checkpoint_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    stage = mc.STAGING
    checkpoint_hash = mc.file_sha(checkpoint_path)
    boundary_path = stage / f"development_update_{update:04d}.json"
    if boundary_path.exists():
        try:
            existing = mc.load_json(boundary_path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            existing = None
        if isinstance(existing, dict):
            if existing.get("checkpoint_sha256") not in (None, checkpoint_hash):
                raise ValueError(f"existing boundary checkpoint hash mismatch at update {update}")
            primary = existing.get("primary_panel", {})
            secondary = existing.get("secondary_corruption_robustness_descriptive_only", {})
            complete = (
                existing.get("schema") == "e010_multicorruption_development_boundary_v1"
                and existing.get("optimizer_updates") == update
                and existing.get("checkpoint_sha256") == checkpoint_hash
                and isinstance(primary, dict)
                and isinstance(secondary, dict)
                and isinstance(primary.get("metrics"), dict)
                and len(primary.get("per_identity", [])) == len(dev_rows)
                and all(
                    isinstance(secondary.get(str(i)), dict)
                    and len(secondary[str(i)].get("per_identity", [])) == len(dev_rows)
                    for i in (1, 2)
                )
            )
            if complete:
                return existing
    primary = _evaluate_panel(model, dev_rows, "cuda", corruption="primary")
    metrics = _summarize_primary(primary, baseline_rows, config)
    # Descriptive secondary fixed-seed panels never enter checkpoint selection or gates.
    secondary = {}
    for idx in (1, 2):
        recs = _evaluate_panel(model, dev_rows, "cuda", corruption=f"secondary_{idx}", seed_rows=dev_seed_rows)
        secondary[str(idx)] = {
            "corruption_index": idx,
            "identity_count": len(recs),
            "mean_aligned_rmse_angstrom": float(np.mean([r["aligned_rmse_angstrom"] for r in recs])),
            "finite_outputs": all(r["finite"] for r in recs),
            "per_identity": recs,
        }
    result = {
        "schema": "e010_multicorruption_development_boundary_v1",
        "optimizer_updates": update,
        "training_examples_seen": update * mc.EFFECTIVE_BATCH,
        "checkpoint_sha256": checkpoint_hash,
        "primary_panel": {
            "source": "original_phase4a_fixed_corruption_development_panel",
            "metrics": metrics,
            "per_identity": primary,
        },
        "secondary_corruption_robustness_descriptive_only": secondary,
        "secondary_panels_used_for_selection": False,
        "secondary_panels_used_for_gates": False,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
        "prospective_accessed": False,
    }
    _atomic_json(boundary_path, result)
    return result


def _recover_missing_boundary(cursor: int, model, dev_rows, dev_seed_rows, baseline_rows, config) -> None:
    """Publish the committed boundary record for an already committed cursor."""
    if cursor not in {x[0] for x in mc.BOUNDARIES}:
        return
    checkpoint = mc.STAGING / f"checkpoint_update_{cursor:04d}.pt"
    latest = mc.STAGING / "latest.pt"
    if not checkpoint.exists():
        shutil.copy2(latest, checkpoint)
    elif mc.file_sha(checkpoint) != mc.file_sha(latest):
        raise ValueError(f"existing boundary checkpoint hash mismatch at update {cursor}")
    _publish_boundary(model, dev_rows, dev_seed_rows, baseline_rows, cursor, checkpoint, config)


def _schedule_from_cursor(schedule, cursor: int):
    """Return only not-yet-committed optimizer updates (cursor is a count)."""
    return schedule[cursor:]


def _cuda_smoke() -> dict[str, Any]:
    """Bounded one-batch/one-step check; never called by preparation or validation."""
    import torch

    from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual

    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-smoke requires an available CUDA device")
    config = yaml.safe_load(mc.CONFIG.read_text())
    torch.manual_seed(int(config["model"]["seed"]))
    torch.cuda.manual_seed_all(int(config["model"]["seed"]))
    spec = {
        k: config["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")
    }
    model = GlobalEquivariantResidual(**spec).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    x = torch.randn((1, 500, 3), device="cuda")
    target = torch.randn_like(x)
    mask = torch.ones((1, 500), dtype=torch.bool, device="cuda")
    out = model(x, mask)
    loss = (out["prediction"] - target).square().mean() + 1e-5 * out["delta"].square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite = bool(
        torch.isfinite(out["prediction"]).all()
        and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    )
    if finite:
        optimizer.step()
    if not finite:
        raise FloatingPointError("CUDA smoke found non-finite outputs or gradients")
    return {
        "status": "passed",
        "batch_size": 1,
        "length": 500,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_started": False,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
    }


def execute(*, resume: bool = False) -> dict[str, Any]:
    contract = _validated_contract()
    if contract.get("status") != "valid_read_only_contract":
        raise RuntimeError(f"execution blocked by read-only contract: {contract.get('status')}")
    import torch

    from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual

    config = yaml.safe_load(mc.CONFIG.read_text())
    if mc.FINAL.exists() or mc.REVIEW.exists():
        raise FileExistsError("final/review artifacts already exist; lifecycle will not overwrite them")
    if not mc.STAGING.is_dir():
        raise FileNotFoundError("prepared staging path is absent; run --plan-only first")
    if resume and not (mc.STAGING / "latest.pt").exists():
        raise FileNotFoundError("resume requested but no committed checkpoint exists")
    if not resume and (mc.STAGING / "journal.jsonl").exists():
        raise FileExistsError("staging journal exists; use --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("scientific E010 training requires CUDA; use --cuda-smoke for the separate bounded smoke")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = int(config["model"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev_rows = mc.load_json(mc.OUT / "development_baseline_reference.json")["identities"]
    seed_obj = mc.load_json(mc.OUT / "training_seed_manifest.json")
    seed_rows = seed_obj["identities"]
    dev_seed_rows = mc.load_json(mc.OUT / "development_secondary_seed_manifest.json")["identities"]
    baseline_rows = dev_rows
    schedule = mc.build_schedule(seed_rows, schedule_seed=int(config["global_seed"]))
    spec = {
        k: config["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")
    }
    model = GlobalEquivariantResidual(**spec).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    events = journal_events() if resume else []
    if resume:
        state = _validate_checkpoint_journal(mc.STAGING, events)
        if state is None:
            raise ValueError("resume staging has no committed optimizer update")
        if state.get("global_update") != len(events) or state.get("schedule_cursor") != len(events):
            raise ValueError("resume checkpoint update cursor does not match the validated journal prefix")
        if (
            state.get("schedule_sha256") != contract["schedule_sha256"]
            or state.get("seed_manifest_sha256") != contract["seed_manifest_sha256"]
            or state.get("config_sha256") != contract["config_sha256"]
        ):
            raise ValueError("resume checkpoint pins do not match prepared schedule/seed manifest")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            raise ValueError("unexpected scheduler state for the declared scheduler:none protocol")
        scaler.load_state_dict(state["scaler"])
        _restore_rng(state)
        cursor = int(state["global_update"])
        exposures = dict(state["per_identity_corruption_exposures"])
        loss_history = list(state["loss_history"])
    else:
        cursor = 0
        exposures = {r["sample_id"]: 0 for r in seed_rows}
        loss_history = []
    training_data = _load_training_data(seed_rows)
    seed_by_id = {r["sample_id"]: r for r in seed_rows}
    started = time.time()
    model.train()
    try:
        _recover_missing_boundary(cursor, model, dev_rows, dev_seed_rows, baseline_rows, config)
        for item in _schedule_from_cursor(schedule, cursor):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for stratum in mc.STRATA:
                examples = item["stratum_microbatches"][stratum]
                if len(examples) != mc.MICROBATCH:
                    raise ValueError("optimizer update lacks exactly 18 examples from a stratum")
                batch_rows = []
                for example in examples:
                    row = seed_by_id[example["sample_id"]]
                    target_np, mask_np, sigma = training_data[example["sample_id"]]
                    coarse_np = mc.regenerate_corruption(target_np, sigma, example["corruption_seed"])
                    batch_rows.append((row, target_np, coarse_np, mask_np))
                max_length = max(len(row[1]) for row in batch_rows)
                target_np = np.zeros((mc.MICROBATCH, max_length, 3), dtype=np.float32)
                coarse_np = np.zeros_like(target_np)
                mask_np = np.zeros((mc.MICROBATCH, max_length), dtype=np.bool_)
                for i, (row, target_i, coarse_i, mask_i) in enumerate(batch_rows):
                    n = len(target_i)
                    target_np[i, :n] = target_i
                    coarse_np[i, :n] = coarse_i
                    mask_np[i, :n] = mask_i
                    exposures[row["sample_id"]] += 1
                target = torch.from_numpy(target_np).to("cuda")
                coarse = torch.from_numpy(coarse_np).to("cuda")
                mask = torch.from_numpy(mask_np).to("cuda")
                output = model(coarse, mask)
                valid = mask[:, :, None].to(target.dtype)
                denom = (3 * mask.sum(1)).clamp_min(1).to(target.dtype)
                coordinate = ((output["prediction"] - target).square() * valid).sum((1, 2)) / denom
                residual = (output["delta"].square() * valid).sum((1, 2)) / denom
                structure_loss = coordinate + 1e-5 * residual
                (structure_loss.mean() * (mc.MICROBATCH / mc.EFFECTIVE_BATCH)).backward()
                total_loss += float(structure_loss.sum().detach().cpu())
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_max_norm"])
            if not torch.isfinite(grad):
                raise FloatingPointError(f"non-finite gradient at optimizer update {cursor + 1}")
            optimizer.step()
            gradient_norm = float(grad.detach().cpu())
            optimizer.zero_grad(set_to_none=True)
            del grad, output, valid, denom, coordinate, residual, structure_loss, target, coarse, mask
            del target_np, coarse_np, mask_np, batch_rows, max_length
            cursor += 1
            loss_history.append(total_loss / mc.EFFECTIVE_BATCH)
            state = _fresh_state(model, optimizer, scaler, contract, cursor, exposures, loss_history)
            pending = mc.STAGING / "pending.pt"
            _torch_save(pending, state)
            checkpoint_hash = mc.file_sha(pending)
            _append_event(
                mc.STAGING / "journal.jsonl",
                {
                    "global_update": cursor,
                    "mean_structure_loss": loss_history[-1],
                    "gradient_norm": gradient_norm,
                    "checkpoint_sha256": checkpoint_hash,
                    "training_started": True,
                    "prospective_accessed": False,
                    "authorization": {"phase4b": False, "prospective": False, "downstream": False},
                },
            )
            os.replace(pending, mc.STAGING / "latest.pt")
            for boundary, example_count, per_stratum in mc.BOUNDARIES:
                if cursor == boundary:
                    if sum(exposures.values()) != example_count or any(
                        sum(exposures[r["sample_id"]] for r in seed_rows if r["stratum"] == s) != per_stratum
                        for s in mc.STRATA
                    ):
                        raise ValueError(f"boundary {cursor} does not have exactly one-fifth exposure per stratum")
                    cp = mc.STAGING / f"checkpoint_update_{cursor:04d}.pt"
                    shutil.copy2(mc.STAGING / "latest.pt", cp)
                    _publish_boundary(model, dev_rows, dev_seed_rows, baseline_rows, cursor, cp, config)
        if cursor != mc.UPDATES or len(journal_events()) != mc.UPDATES:
            raise ValueError("completed execution does not contain exactly 1,092 journaled optimizer updates")
        boundary_records = [mc.load_json(mc.STAGING / f"development_update_{n:04d}.json") for n, _, _ in mc.BOUNDARIES]
        selected = min(
            boundary_records,
            key=lambda r: (r["primary_panel"]["metrics"]["refined_mean_aligned_rmse_angstrom"], r["optimizer_updates"]),
        )
        all_pass = all(selected["primary_panel"]["metrics"]["gate_checks"].values())
        classification = "pass_all_original_gates" if all_pass else "fail_one_or_more_original_gates"
        # Decision support reports the historical 25.20% endpoint comparison. It never grants permissions.
        metrics = {
            "schema": "e010_multicorruption_scientific_metrics_v1",
            "status": "completed",
            "classification": classification,
            "selection_rule": "lowest primary fixed-panel development mean aligned RMSE; earliest boundary wins ties",
            "selected_optimizer_update": selected["optimizer_updates"],
            "selected_primary_metrics": selected["primary_panel"]["metrics"],
            "boundaries": [
                {k: v for k, v in r.items() if k != "primary_panel"}
                | {"primary_panel": {k: v for k, v in r["primary_panel"].items() if k != "per_identity"}}
                for r in boundary_records
            ],
            "phase4a_reported_comparison_reduction_fraction": 0.2520166275621228,
            "delta_from_phase4a_reported_reduction_fraction": selected["primary_panel"]["metrics"][
                "rmse_reduction_fraction"
            ]
            - 0.2520166275621228,
            "decision_support": (
                "materiality above Phase 4A is reported descriptively for human review; this "
                "lifecycle does not authorize Phase 4B, prospective access, or downstream "
                "use"
            ),
            "training_identity_count": mc.TOTAL_IDENTITIES,
            "training_example_count": mc.TOTAL_EXAMPLES,
            "examples_per_stratum": {s: mc.EXAMPLES_PER_STRATUM for s in mc.STRATA},
            "optimizer_updates": mc.UPDATES,
            "selected_checkpoint_sha256": mc.file_sha(
                mc.STAGING / f"checkpoint_update_{selected['optimizer_updates']:04d}.pt"
            ),
            "authorization": {"phase4b": False, "prospective": False, "downstream": False},
            "phase4b_prepared": False,
            "prospective_accessed": False,
            "downstream_authorized": False,
            "training_started": True,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(mc.STAGING / "scientific_review.json", metrics)
        _atomic_json(mc.STAGING / "training_metrics.json", metrics)
        # Rendering is downstream of the durable scientific JSON and cannot invalidate it.
        try:
            lines = [
                "# E010 multi-corruption generalization review",
                "",
                f"Classification: `{classification}`",
                f"Selected update: {selected['optimizer_updates']}",
                "",
                "## Primary fixed-panel gates",
                "",
            ]
            for key, value in selected["primary_panel"]["metrics"]["gate_checks"].items():
                lines.append(f"- {key}: **{str(value).lower()}**")
            lines += [
                "",
                "Secondary corruption panels are descriptive only and did not affect selection or gates.",
                "All authorization fields remain false.",
                "",
            ]
            (mc.STAGING / "scientific_review.md").write_text("\n".join(lines))
        except Exception as exc:
            _atomic_json(
                mc.STAGING / "report_generation_failure.json", {"error_type": type(exc).__name__, "error": str(exc)}
            )
        os.replace(mc.STAGING, mc.FINAL)
        _publish_review(metrics)
        return {
            "status": "completed",
            "classification": classification,
            "selected_optimizer_update": selected["optimizer_updates"],
            "final_path": str(mc.FINAL.relative_to(mc.ROOT)),
            "review_path": str(mc.REVIEW.relative_to(mc.ROOT)),
            "authorization": metrics["authorization"],
        }
    except BaseException as exc:
        _atomic_json(
            mc.STAGING / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "global_update": cursor,
                "training_started": cursor > 0,
                "prospective_accessed": False,
                "authorization": {"phase4b": False, "prospective": False, "downstream": False},
            },
        )
        raise


def _publish_review(metrics: dict[str, Any]) -> None:
    if mc.REVIEW.exists():
        raise FileExistsError(mc.REVIEW)
    temp = Path(tempfile.mkdtemp(prefix=f".{mc.REVIEW.name}.", dir=mc.OUT))
    try:
        _atomic_json(temp / "review.json", metrics)
        try:
            (temp / "review.md").write_text((mc.FINAL / "scientific_review.md").read_text())
        except Exception as exc:
            (temp / "review.md").write_text(
                f"# E010 multi-corruption review\n\nMarkdown rendering failed: {type(exc).__name__}: {exc}\n"
            )
        inventory = [
            {"path": p.name, "size_bytes": p.stat().st_size, "sha256": mc.file_sha(p)}
            for p in sorted(temp.iterdir())
            if p.is_file()
        ]
        _atomic_json(
            temp / "artifact_inventory.json", {"schema": "e010_multicorruption_review_inventory_v1", "files": inventory}
        )
        sums = {p.name: mc.file_sha(p) for p in sorted(temp.iterdir()) if p.is_file()}
        _atomic_json(temp / "SHA256SUMS.json", sums)
        os.replace(temp, mc.REVIEW)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def monitor() -> dict[str, Any]:
    events = journal_events(mc.FINAL / "journal.jsonl" if mc.FINAL.exists() else mc.STAGING / "journal.jsonl")
    location = mc.FINAL if mc.FINAL.exists() else mc.STAGING
    if mc.FINAL.exists() and mc.STAGING.exists():
        raise ValueError("both staging and final paths exist")
    if events:
        latest = location / "latest.pt"
        if not latest.is_file() or mc.file_sha(latest) != events[-1]["checkpoint_sha256"]:
            raise ValueError("latest checkpoint does not match the journal prefix")
    active = _lifecycle_process_running()
    valid_checkpoint = bool(
        events
        and (location / "latest.pt").is_file()
        and mc.file_sha(location / "latest.pt") == events[-1]["checkpoint_sha256"]
    )
    status = (
        "completed"
        if mc.FINAL.exists()
        else (
            "running"
            if active
            else "recoverable"
            if valid_checkpoint
            else "prepared"
            if mc.STAGING.exists()
            else "absent"
        )
    )
    return {
        "status": status,
        "journal_updates": len(events),
        "expected_updates": mc.UPDATES,
        "latest_checkpoint_sha256": mc.file_sha(location / "latest.pt") if (location / "latest.pt").is_file() else None,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
    }


def _lifecycle_process_running() -> bool:
    """Inspect process command lines only; safe for monitor and does not import torch."""
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    script = "run_e010_phase4a_multicorruption_v2.py"
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if script in args and ("--execute" in args or "--resume" in args):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-only", action="store_true")
    group.add_argument("--validate-contract", action="store_true")
    group.add_argument("--cuda-smoke", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--monitor", action="store_true")
    group.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        from scripts.prepare_e010_phase4a_multicorruption_v2 import plan_only

        result = plan_only()
    elif args.validate_contract:
        result = _validated_contract()
    elif args.cuda_smoke:
        contract = _validated_contract()
        if contract.get("status") != "valid_read_only_contract":
            raise RuntimeError(f"CUDA smoke blocked by read-only contract: {contract.get('status')}")
        result = _cuda_smoke()
    elif args.execute:
        result = execute(resume=False)
    elif args.resume:
        result = execute(resume=True)
    else:
        result = monitor()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
