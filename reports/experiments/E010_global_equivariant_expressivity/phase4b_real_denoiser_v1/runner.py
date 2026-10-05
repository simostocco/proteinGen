#!/usr/bin/env python3
"""Non-authorizing E010 Phase 4B real frozen-denoiser lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import prepare_cache as pc  # noqa: E402

STAGE = HERE / "phase4b_training_v1.staging"
FINAL = HERE / "phase4b_training_v1.final"
REVIEW = HERE / "phase4b_real_denoiser_v1_review_v1"


def _native(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_native(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, Path): return str(x)
    return x


def _atomic_json(path: Path, value: Any) -> None:
    pc.atomic_json(path, _native(value))


def resume_identity_valid(state: dict[str, Any], journal_bytes: bytes, cache_sha256: str) -> bool:
    required = ("model", "optimizer", "scheduler", "scaler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng")
    return (state.get("global_update") == len(journal_bytes.splitlines())
        and state.get("journal_prefix_sha256") == hashlib.sha256(journal_bytes).hexdigest()
        and state.get("cache_manifest_sha256") == cache_sha256
        and all(key in state for key in required))


def _journal_bytes(directory: Path) -> bytes:
    path = directory / "journal.jsonl"
    return path.read_bytes() if path.exists() else b""


def _sync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_journal(directory: Path, raw: bytes) -> None:
    temp = directory / "journal.jsonl.tmp"
    with temp.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, directory / "journal.jsonl")
    _sync_directory(directory)


def _journal_events(raw: bytes, cache_sha256: str) -> list[dict[str, Any]]:
    lines = raw.splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    if any(line != pc.canonical(event) + b"\n"
           or event.get("global_update") != i
           or event.get("cache_manifest_sha256") != cache_sha256
           for i, (line, event) in enumerate(zip(lines, events), 1)):
        raise ValueError("training journal canonical/update/cache identity mismatch")
    return events


def _inspect_transaction(directory: Path, cache_sha256: str, *, final: bool = False):
    """Read only. A durable pending.pt is the commit point; temp files are attempts."""
    import torch

    raw = _journal_bytes(directory)
    events = _journal_events(raw, cache_sha256)
    latest_path, pending_path = directory / "latest.pt", directory / "pending.pt"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False) if latest_path.is_file() else None
    pending = torch.load(pending_path, map_location="cpu", weights_only=False) if pending_path.is_file() else None

    def valid(state, prefix):
        return (isinstance(state, dict) and state.get("schema") == "e010_phase4b_exact_state_v1"
                and state.get("authorization") == pc.AUTH
                and resume_identity_valid(state, prefix, cache_sha256))

    if pending is not None and not final:
        event = pending.get("journal_event")
        # Older checkpoints can only be recovered if their row was already published.
        if event is None:
            expected = raw
        else:
            row = pc.canonical(event) + b"\n"
            expected = raw if events and events[-1] == event else raw + row
        _journal_events(expected, cache_sha256)
        if not valid(pending, expected) or pending.get("global_update", 0) < 1:
            raise ValueError("pending checkpoint does not match recoverable journal prefix")
        prefix = b"".join(expected.splitlines(keepends=True)[:-1])
        if latest is not None and not (valid(latest, prefix) or valid(latest, expected)):
            raise ValueError("latest checkpoint does not match pending predecessor")
        if prefix and latest is None:
            raise ValueError("pending checkpoint lacks committed predecessor")
        status = "checkpoint_committed_journal_pending" if expected != raw else "journal_committed_latest_pending"
        return pending, expected, pending_path, status
    if latest is not None:
        if not events or not valid(latest, raw):
            raise ValueError("training checkpoint does not match journal")
        return latest, raw, latest_path, "consistent"
    if events:
        raise ValueError("training journal lacks full-state checkpoint")
    return None, raw, None, "zero_committed_updates"


def _prepare_execution(directory: Path, cache_sha256: str, *, resume: bool, recover: bool = False):
    state, raw, checkpoint, status = _inspect_transaction(directory, cache_sha256)
    if not resume and state is not None:
        raise ValueError("committed checkpoint exists; use --resume")
    if resume and state is None:
        raise ValueError("resume requires full-state checkpoint; zero committed updates require --execute")
    if recover and checkpoint is not None and checkpoint.name == "pending.pt":
        if raw != _journal_bytes(directory):
            _publish_journal(directory, raw)
        os.replace(checkpoint, directory / "latest.pt")
        _sync_directory(directory)
    return state, raw, status


def _commit_update(directory: Path, state: dict[str, Any], event: dict[str, Any], cache_sha256: str) -> None:
    import torch

    previous, prefix, _, status = _inspect_transaction(directory, cache_sha256)
    if status not in ("consistent", "zero_committed_updates"):
        raise ValueError("recover pending transaction before another update")
    if event.get("global_update") != (previous["global_update"] if previous else 0) + 1:
        raise ValueError("update does not follow committed cursor")
    raw = prefix + pc.canonical(event) + b"\n"
    _journal_events(raw, cache_sha256)
    state = {**state, "journal_event": event, "journal_prefix_sha256": pc.digest(raw)}
    if (state.get("schema") != "e010_phase4b_exact_state_v1" or state.get("authorization") != pc.AUTH
        or not resume_identity_valid(state, raw, cache_sha256)):
        raise ValueError("checkpoint cursor/cache/full-state mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory / "pending.pt.tmp"
    with temp.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, directory / "pending.pt")
    _sync_directory(directory)
    _publish_journal(directory, raw)
    os.replace(directory / "pending.pt", directory / "latest.pt")
    _sync_directory(directory)


def zero_shot_prediction(model: Any, coords: Any, mask: Any) -> Any:
    """Run a refiner read-only and return a detached CPU copy."""
    import torch
    was_training = bool(model.training)
    model.eval()
    with torch.inference_mode():
        result = model(coords, mask)["prediction"].detach().cpu().clone()
    model.train(was_training)
    return result


def _load_cache() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = pc.config()
    root = HERE / cfg["cache"]["final_dir"]
    manifest = pc.load_cache_manifest(root)
    pc.verify_cache(root, manifest)
    split_records = {"train": [], "development": []}
    for shard in manifest["shards"]:
        with np.load(root / shard["shard"], allow_pickle=False) as z:
            offsets = z["offsets"].copy(); target = z["target"].copy(); pred = z["prediction"].copy()
            meta = json.loads(z["records_json"].tobytes())
        for i, row in enumerate(meta):
            a, b = int(offsets[i]), int(offsets[i + 1])
            split_records[row["split"]].append({**row, "target": target[a:b],
                                                  "prediction": pred[a:b]})
    if (len(split_records["train"]) != manifest["training_record_count"]
        or len(split_records["development"]) != manifest["development_record_count"]):
        raise ValueError("loaded cache split counts mismatch")
    return split_records["train"] + split_records["development"], manifest


def _kabsch(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    p, q = pred - pred.mean(0), target - target.mean(0)
    u, _, vh = np.linalg.svd(p.T @ q)
    sign = np.linalg.det(u @ vh)
    rotation = u @ np.diag([1.0, 1.0, sign]) @ vh
    return p @ rotation


def _metrics(records: list[dict[str, Any]], output_key: str) -> dict[str, Any]:
    rows = []
    for r in records:
        pred = r[output_key]
        target = r["target"]
        aligned = _kabsch(pred, target)
        centered_target = target - target.mean(0)
        err = np.linalg.norm(aligned - centered_target, axis=1)
        local = {}
        for separation in (1, 2, 3):
            if len(target) > separation:
                pd = np.linalg.norm(aligned[separation:] - aligned[:-separation], axis=1)
                td = np.linalg.norm(centered_target[separation:] - centered_target[:-separation], axis=1)
                local[str(separation)] = float(np.sqrt(np.mean((pd - td) ** 2)))
        chirality = []
        for i in range(max(0, len(target) - 3)):
            def volume(x):
                return float(np.dot(np.cross(x[i + 1] - x[i], x[i + 2] - x[i]), x[i + 3] - x[i]))
            pv, tv = volume(aligned), volume(centered_target)
            if abs(pv) > 1e-7 and abs(tv) > 1e-7: chirality.append(pv * tv < 0)
        rg_pred = float(np.sqrt(np.mean(np.sum((pred - pred.mean(0)) ** 2, axis=1))))
        rg_target = float(np.sqrt(np.mean(np.sum(centered_target ** 2, axis=1))))
        rows.append({"sample_id": r["sample_id"], "split": r["split"], "stratum": r["stratum"],
            "timestep": int(r["timestep"]), "condition_index": int(r["condition_index"]),
            "length": int(r["length"]), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_error": float(err.mean()), "max_error": float(err.max()),
            "local_distance_rmse": local, "chirality_inversion_rate": float(np.mean(chirality)) if chirality else 0.0,
            "chirality_eligible_tetrahedra": len(chirality),
            "chirality_assessable": bool(chirality),
            "prediction_radius_gyration": rg_pred, "target_radius_gyration": rg_target,
            "finite": bool(np.isfinite(pred).all()), "prediction": pred})
    identities = sorted({r["sample_id"] for r in rows})
    # Bootstrap paired identity means with a stable RNG seed; condition rows are
    # first averaged within identity so repeated noise conditions do not inflate n.
    grouped = {sid: [r for r in rows if r["sample_id"] == sid] for sid in identities}
    per_identity = [{"sample_id": sid, "mean_rmse": float(np.mean([r["rmse"] for r in grouped[sid]])),
                     "max_rmse": float(max(r["rmse"] for r in grouped[sid])),
                     "mean_error": float(np.mean([r["mean_error"] for r in grouped[sid]])),
                     "max_error": float(max(r["max_error"] for r in grouped[sid]))} for sid in identities]
    rmse = np.array([r["mean_rmse"] for r in per_identity], dtype=np.float64)
    by_condition = {}
    for condition in sorted({r["timestep"] for r in rows}):
        group = [r for r in rows if r["timestep"] == condition]
        by_condition[str(condition)] = {"count": len(group), "mean_rmse": float(np.mean([r["rmse"] for r in group])),
            "median_rmse": float(np.median([r["rmse"] for r in group])), "max_rmse": float(max(r["rmse"] for r in group)),
            "local_distance_rmse_i_plus_1_2_3": {str(i): float(np.mean([r["local_distance_rmse"][str(i)] for r in group if str(i) in r["local_distance_rmse"]])) for i in (1, 2, 3)},
            "chirality_inversion_rate": float(np.mean([r["chirality_inversion_rate"] for r in group])),
            "finite_output_rate": float(np.mean([r["finite"] for r in group]))}
    by_stratum = {s: {"count": sum(r["stratum"] == s for r in rows),
        "mean_rmse": float(np.mean([r["rmse"] for r in rows if r["stratum"] == s])),
        "local_distance_rmse_i_plus_1_2_3": {str(i): float(np.mean([r["local_distance_rmse"][str(i)] for r in rows if r["stratum"] == s and str(i) in r["local_distance_rmse"]])) for i in (1, 2, 3)},
        "chirality_inversion_rate": float(np.mean([r["chirality_inversion_rate"] for r in rows if r["stratum"] == s])),
        "finite_output_rate": float(np.mean([r["finite"] for r in rows if r["stratum"] == s]))} for s in pc.STRATA}
    rng = np.random.default_rng(41046)
    boot = np.mean(rmse[rng.integers(0, len(rmse), size=(10000, len(rmse)))], axis=1)
    return {"count_conditions": len(rows), "count_identities": len(identities), "mean_rmse": float(rmse.mean()),
        "median_rmse": float(np.median(rmse)), "maximum_aligned_rmse": float(max(r["max_rmse"] for r in per_identity)),
        "bootstrap_identity_mean_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "length_strata": by_stratum, "conditions": by_condition,
        "per_identity": per_identity, "per_condition": [{k: v for k, v in r.items() if k != "prediction"} for r in rows],
        "local_distance_rmse_i_plus_1_2_3": {str(i): float(np.mean([r["local_distance_rmse"][str(i)] for r in rows if str(i) in r["local_distance_rmse"]])) for i in (1, 2, 3)},
        "chirality_inversion_rate": float(np.mean([r["chirality_inversion_rate"] for r in rows])),
        "chirality_eligible_tetrahedra": sum(r["chirality_eligible_tetrahedra"] for r in rows),
        "chirality_assessable": all(r["chirality_assessable"] for r in rows),
        "error_vs_length_slope": float(np.polyfit([r["length"] for r in rows], [r["rmse"] for r in rows], 1)[0]),
        "diversity_by_stratum": {s: {"radius_gyration_sd_prediction": float(np.std([r["prediction_radius_gyration"] for r in rows if r["stratum"] == s])),
            "radius_gyration_sd_target": float(np.std([r["target_radius_gyration"] for r in rows if r["stratum"] == s])),
            "collapse": bool(np.mean([r["prediction_radius_gyration"] for r in rows if r["stratum"] == s]) < 1e-3)} for s in pc.STRATA},
        "finite_output_rate": float(np.mean([r["finite"] for r in rows]))}


def _model(device: str, *, load_selected: bool = True):
    import torch
    import yaml

    from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual
    cfg = pc.config()
    model_cfg = yaml.safe_load(pc._path(cfg["e010"]["config"]).read_text())["model"]
    spec = {k: model_cfg[k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")}
    model = GlobalEquivariantResidual(**spec).to(device)
    if load_selected:
        payload = torch.load(pc._path(cfg["e010"]["checkpoint"]), map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        model.load_state_dict(state)
    return model


def plan_only() -> dict[str, Any]:
    result = pc.validate_contract()
    result.update({"mode": "plan_only", "staging_path": str(STAGE.relative_to(ROOT)),
        "final_path": str(FINAL.relative_to(ROOT)), "review_path": str(REVIEW.relative_to(ROOT)),
        "expected_uncompressed_coordinate_tensor_gib": result["expected_uncompressed_coordinate_tensor_bytes"] / (1024 ** 3),
        "estimated_uncompressed_archive_size": "about 0.5 GiB including 800,128 bytes of shard offsets and per-record JSON metadata; tensor payload is exact",
        "e010_training_forward_examples": 98280, "e010_training_forward_microbatch_invocations": 1092 * 5,
        "e010_training_backward_calls": 1092 * 5,
        "e010_zero_shot_forward_examples": 960, "e010_zero_shot_model_invocations": 960,
        "e010_adapted_evaluation_forward_examples": 3 * 960, "e010_adapted_evaluation_model_invocations": 3 * 960,
        "e010_evaluation_backward_calls": 0,
        "estimated_runtime_basis": "E007 measured 1,800 train examples / 1,268.54 s = 1.418 examples/s; a forward-only cache pass is projected at 2x that rate (34,969 s for 99,240 records), an explicit approximate inference assumption. E010 completed the same 1,092-update/98,280-example workload in 19,835.52 s. Combined estimate is about 54,805 s (15.2 h), excluding storage/I/O and boundary evaluation variation.",
        "authorization": pc.AUTH})
    return result


def validate() -> dict[str, Any]:
    report = pc.validate_contract()
    if STAGE.exists() or FINAL.exists() or REVIEW.exists():
        report["existing_output_paths"] = True
        report["status"] = "valid_contract_outputs_already_exist"
    return report


def build_cache() -> dict[str, Any]:
    return pc.build_cache()


def zero_shot() -> dict[str, Any]:
    import torch
    manifest_path = HERE / pc.config()["cache"]["final_dir"] / "manifest.json"
    cache_file_sha256 = pc.sha256(manifest_path)
    records, cache_manifest = _load_cache()
    if REVIEW.exists(): raise FileExistsError("review path already exists")
    dev = [r for r in records if r["split"] == "development"]
    torch.manual_seed(41045)
    model = _model("cuda", load_selected=True)
    model.eval()
    for r in dev:
        x = torch.from_numpy(r["prediction"]).to("cuda")[None]
        mask = torch.ones((1, len(r["prediction"])), dtype=torch.bool, device="cuda")
        r["zero_shot"] = zero_shot_prediction(model, x, mask)[0].float().numpy()
    baseline = _metrics(dev, "prediction")
    synthetic = _metrics(dev, "zero_shot")
    result = {"schema": "e010_phase4b_zero_shot_v1", "cache_sha256": hashlib.sha256(pc.canonical(cache_manifest)).hexdigest(),
        "cache_manifest_file_sha256": cache_file_sha256,
        "frozen_denoiser": baseline, "denoiser_plus_synthetic_e010": synthetic,
        "per_identity_paired": _paired(baseline, synthetic), "per_condition_paired": _paired_conditions(baseline, synthetic), "training_started": False,
        "authorization": pc.AUTH, "prospective_accessed": False}
    if pc.sha256(manifest_path) != cache_file_sha256:
        raise ValueError("published cache manifest changed during zero-shot")
    out = STAGE
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(dev, key=lambda r: (r["sample_id"], r["condition_index"]))
    offsets = np.concatenate(([0], np.cumsum([len(r["prediction"]) for r in ordered]))).astype(np.int64)
    zero_flat = np.concatenate([r["zero_shot"] for r in ordered], axis=0).astype(np.float32)
    zero_records = [{k: r[k] for k in ("sample_id", "condition_index", "timestep", "length", "stratum")} for r in ordered]
    temp_npz = out / "zero_shot_predictions.npz.tmp"
    with temp_npz.open("wb") as handle:
        np.savez_compressed(handle, prediction=zero_flat, offsets=offsets,
            records_json=np.frombuffer(json.dumps(zero_records, sort_keys=True).encode(), dtype=np.uint8))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp_npz, out / "zero_shot_predictions.npz")
    result["prediction_archive_sha256"] = pc.sha256(out / "zero_shot_predictions.npz")
    _atomic_json(out / "zero_shot.json", result)
    _atomic_json(out / "zero_shot_complete.json", {"schema": "e010_phase4b_zero_shot_commit_v1",
        "report_sha256": pc.sha256(out / "zero_shot.json"),
        "prediction_archive_sha256": result["prediction_archive_sha256"],
        "cache_sha256": result["cache_sha256"], "development_condition_count": 960,
        "cache_manifest_file_sha256": result["cache_manifest_file_sha256"],
        "training_started": False, "authorization": pc.AUTH})
    return result


def _paired(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    b = {r["sample_id"]: r["mean_rmse"] for r in base["per_identity"]}
    c = {r["sample_id"]: r["mean_rmse"] for r in candidate["per_identity"]}
    if b.keys() != c.keys(): raise ValueError("paired identity set mismatch")
    improvement = np.array([(b[s] - c[s]) / max(b[s], 1e-12) for s in sorted(b)])
    rng = np.random.default_rng(41046)
    boot = np.mean(improvement[rng.integers(0, len(improvement), size=(10000, len(improvement)))], axis=1)
    return {"identity_count": len(improvement), "mean_paired_percentage_improvement": float(improvement.mean()),
        "bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}


def _paired_conditions(base: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    left = {(r["sample_id"], r["condition_index"]): r for r in base["per_condition"]}
    right = {(r["sample_id"], r["condition_index"]): r for r in candidate["per_condition"]}
    if left.keys() != right.keys(): raise ValueError("paired identity-condition set mismatch")
    return [{"sample_id": sid, "condition_index": ci, "timestep": left[(sid, ci)]["timestep"],
        "baseline_rmse": left[(sid, ci)]["rmse"], "candidate_rmse": right[(sid, ci)]["rmse"],
        "paired_percentage_improvement": (left[(sid, ci)]["rmse"] - right[(sid, ci)]["rmse"]) / max(left[(sid, ci)]["rmse"], 1e-12)}
        for sid, ci in sorted(left)]


def adjudicate(baseline: dict[str, Any], candidate: dict[str, Any], paired: dict[str, Any]) -> dict[str, Any]:
    reduction = float(paired["mean_paired_percentage_improvement"])
    strata = {}
    for label in pc.STRATA:
        before = float(baseline["length_strata"][label]["mean_rmse"])
        after = float(candidate["length_strata"][label]["mean_rmse"])
        strata[label] = (before - after) / max(before, 1e-12)
    local_before = baseline["local_distance_rmse_i_plus_1_2_3"]
    local_after = candidate["local_distance_rmse_i_plus_1_2_3"]
    local_improvements = {k: (float(local_before[k]) - float(local_after[k])) / max(float(local_before[k]), 1e-12) for k in ("1", "2", "3")}
    local_reduction = float(np.mean(list(local_improvements.values())))
    no_collapse = all(not candidate["diversity_by_stratum"][s]["collapse"] for s in pc.STRATA)
    diversity_ok = all(candidate["diversity_by_stratum"][s]["radius_gyration_sd_prediction"] >=
        0.05 * max(candidate["diversity_by_stratum"][s]["radius_gyration_sd_target"], 1e-12) for s in pc.STRATA)
    chirality_assessable = (candidate.get("chirality_assessable", True)
        and baseline.get("chirality_assessable", True))
    safe = (candidate["finite_output_rate"] == 1.0 and no_collapse and diversity_ok and chirality_assessable
        and candidate["chirality_inversion_rate"] <= baseline["chirality_inversion_rate"]
        and candidate["error_vs_length_slope"] <= baseline["error_vs_length_slope"])
    ci = paired["bootstrap_ci95"]
    if reduction >= .30 and min(strata.values()) >= .20 and local_reduction >= .20 and safe:
        classification = "strong_pass"
    elif .20 <= reduction < .30 and min(strata.values()) >= 0 and ci[0] > 0 and safe:
        classification = "promising_real_domain_gain"
    else:
        classification = "insufficient_real_domain_gain"
    return {"classification": classification, "overall_paired_improvement": reduction,
        "paired_improvement_ci95": ci, "stratum_improvement": strata,
        "local_distance_improvement_i_plus_1_2_3": local_improvements,
        "mean_local_distance_improvement": local_reduction,
        "safety_gates": {"finite_outputs": candidate["finite_output_rate"] == 1.0,
            "chirality_assessable": chirality_assessable,
            "coordinate_noncollapse": no_collapse, "diversity_noncollapse": diversity_ok,
            "chirality_not_increased": candidate["chirality_inversion_rate"] <= baseline["chirality_inversion_rate"],
            "length_slope_not_increased": candidate["error_vs_length_slope"] <= baseline["error_vs_length_slope"]}}


def _boundary_eval(model, records, update: int) -> dict[str, Any]:
    import torch
    for r in records:
        x = torch.from_numpy(r["prediction"])[None].cuda()
        mask = torch.ones((1, len(r["prediction"])), dtype=torch.bool, device="cuda")
        r[f"adapted_{update}"] = zero_shot_prediction(model, x, mask)[0].float().numpy()
    return _metrics(records, f"adapted_{update}")


def execute(*, resume: bool = False) -> dict[str, Any]:
    import torch
    cfg = pc.config()
    cache_dir = HERE / cfg["cache"]["final_dir"]
    cache_manifest = pc.load_cache_manifest(cache_dir)
    pc.verify_cache(cache_dir, cache_manifest)
    zero_path = STAGE / "zero_shot.json"
    if not zero_path.is_file(): raise ValueError("verified zero-shot report required before training")
    zero = json.loads(zero_path.read_text())
    completion_path = STAGE / "zero_shot_complete.json"
    if not completion_path.is_file(): raise ValueError("zero-shot completion marker required")
    completion = json.loads(completion_path.read_text())
    if (completion.get("schema") != "e010_phase4b_zero_shot_commit_v1"
        or completion.get("report_sha256") != pc.sha256(zero_path)
        or completion.get("prediction_archive_sha256") != zero.get("prediction_archive_sha256")
        or completion.get("development_condition_count") != 960
        or completion.get("training_started") is not False or completion.get("authorization") != pc.AUTH):
        raise ValueError("zero-shot completion marker/report identity mismatch")
    if zero.get("cache_sha256") != hashlib.sha256(pc.canonical(cache_manifest)).hexdigest() or completion.get("cache_sha256") != zero.get("cache_sha256"):
        raise ValueError("zero-shot report cache identity mismatch")
    _validate_manifest_byte_pin(zero, completion, pc.sha256(cache_dir / "manifest.json"))
    if REVIEW.exists() or FINAL.exists(): raise FileExistsError("final/review publication already exists")
    cache_sha = pc.digest(pc.canonical(cache_manifest))
    _prepare_execution(STAGE, cache_sha, resume=resume)
    if not torch.cuda.is_available(): raise RuntimeError("--execute requires CUDA")
    records, _ = _load_cache()
    train = [r for r in records if r["split"] == "train"]
    dev = [r for r in records if r["split"] == "development"]
    with np.load(STAGE / "zero_shot_predictions.npz", allow_pickle=False) as z:
        offsets = z["offsets"].copy(); zero_pred = z["prediction"].copy(); zero_meta = json.loads(z["records_json"].tobytes())
    if pc.sha256(STAGE / "zero_shot_predictions.npz") != zero.get("prediction_archive_sha256"):
        raise ValueError("zero-shot prediction archive hash mismatch")
    indexed = {(r["sample_id"], int(r["condition_index"])): i for i, r in enumerate(zero_meta)}
    for row in dev:
        i = indexed.get((row["sample_id"], int(row["condition_index"])))
        if i is None: raise ValueError("zero-shot archive missing development condition")
        row["zero_shot"] = zero_pred[int(offsets[i]):int(offsets[i + 1])]
    torch.manual_seed(int(cfg["seed"])); torch.cuda.manual_seed_all(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"])); random.seed(int(cfg["seed"]))
    model = _model("cuda", load_selected=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    events = []
    cursor = 0
    boundaries = {}
    state, journal_bytes, _ = _prepare_execution(STAGE, cache_sha, resume=resume, recover=True)
    if state is not None:
        events = _journal_events(journal_bytes, cache_sha)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
        cursor = state["global_update"]; boundaries = state.get("boundaries", {})
    else:
        STAGE.mkdir(parents=True, exist_ok=True)
    dev_baseline = _metrics(dev, "prediction")
    synthetic = _metrics(dev, "zero_shot")
    if cache_manifest.get("training_schedule_sha256") != pc.validate_contract()["training_schedule_sha256"]:
        raise ValueError("cache does not use the pinned multicorruption v2 optimizer-step schedule")
    # Preserve the exact historical update and microbatch order.
    queues = {update: {s: sorted([r for r in train if r["stratum"] == s and r["schedule_update"] == update],
                               key=lambda r: r["microbatch_position"]) for s in pc.STRATA}
              for update in range(1, 1093)}
    if any(len(queues[u][s]) != 18 for u in queues for s in pc.STRATA):
        raise ValueError("historical optimizer schedule does not contain five 18-example microbatches")
    for update in range(cursor, 1092):
        model.train(); optimizer.zero_grad(set_to_none=True); loss_total = 0.0
        for stratum in pc.STRATA:
            batch = queues[update + 1][stratum]
            max_len = max(int(r["length"]) for r in batch)
            x = torch.zeros((len(batch), max_len, 3), device="cuda"); y = torch.zeros_like(x); mask = torch.zeros((len(batch), max_len), dtype=torch.bool, device="cuda")
            for i, row in enumerate(batch):
                n = int(row["length"]); x[i, :n] = torch.from_numpy(row["prediction"]).to("cuda"); y[i, :n] = torch.from_numpy(row["target"]).to("cuda"); mask[i, :n] = True
            # The selected Phase 4A training run used full precision. No AMP
            # validation exists for this 12.84M refiner, so preserve that mode.
            with torch.autocast("cuda", enabled=False):
                pred = model(x, mask)["prediction"]
                sq = (pred.float() - y).square().mean(dim=-1)
                per = (sq * mask).sum(1) / mask.sum(1).clamp_min(1)
                delta = (pred.float() - x).square().mean(dim=-1)
                reg = (delta * mask).sum(1) / mask.sum(1).clamp_min(1)
                loss = (per + 1e-5 * reg).mean() / 5
            loss.backward(); loss_total += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        optimizer.step(); global_update = update + 1
        event = {"global_update": global_update, "mean_loss": loss_total, "cache_manifest_sha256": hashlib.sha256(pc.canonical(cache_manifest)).hexdigest()}
        events.append(event)
        if global_update in cfg["data"]["boundaries"]:
            model.eval()
            adapted = _boundary_eval(model, dev, global_update)
            boundaries[str(global_update)] = {"frozen_denoiser": dev_baseline, "zero_shot_synthetic_e010": synthetic,
                "phase4b_adapted_e010": adapted,
                "paired_vs_frozen_denoiser": _paired(dev_baseline, adapted),
                "paired_vs_zero_shot_e010": _paired(synthetic, adapted),
                "training_mean_objective_loss_through_boundary": float(np.mean([e["mean_loss"] for e in events])),
                "legacy_incomparable_error_difference": float(np.mean([e["mean_loss"] for e in events]) - adapted["mean_rmse"] ** 2),
                "legacy_error_difference_interpretation": "Not a generalization gap: historical unaligned component MSE minus squared mean aligned RMSD has incompatible reductions and coordinate frames."}
            boundaries[str(global_update)]["adjudication"] = adjudicate(dev_baseline, adapted, boundaries[str(global_update)]["paired_vs_frozen_denoiser"])
        state = {"schema": "e010_phase4b_exact_state_v1", "global_update": global_update,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": scaler.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "cache_manifest_sha256": hashlib.sha256(pc.canonical(cache_manifest)).hexdigest(),
            "boundaries": boundaries,
            "authorization": pc.AUTH}
        _commit_update(STAGE, state, event, cache_sha)
    if len(events) != 1092: raise ValueError("Phase 4B must stop exactly at 1,092 updates")
    result = {"schema": "e010_phase4b_real_denoiser_results_v1", "status": "completed_non_authorizing",
        "updates": 1092, "training_examples": 98280, "boundaries": boundaries,
        "authorization": pc.AUTH, "prospective_accessed": False}
    _atomic_json(STAGE / "results.json", result)
    if FINAL.exists() or REVIEW.exists(): raise FileExistsError("publication target already exists")
    pc.publish_directory_once(STAGE, FINAL)
    review_tmp = REVIEW.with_name(REVIEW.name + ".tmp")
    review_tmp.mkdir(parents=True)
    shutil.copy2(FINAL / "results.json", review_tmp / "review.json")
    _atomic_json(review_tmp / "artifact_inventory.json", {"results_sha256": pc.sha256(FINAL / "results.json"), "authorization": pc.AUTH})
    pc.publish_directory_once(review_tmp, REVIEW)
    return result


def _validate_manifest_byte_pin(report: dict[str, Any], commit: dict[str, Any], file_sha256: str) -> None:
    # Historical reports only pinned canonical content. Preserve them without
    # inventing a byte pin retroactively; new reports pin both serializations.
    key = "cache_manifest_file_sha256"
    if key in report or key in commit:
        if report.get(key) != file_sha256 or commit.get(key) != file_sha256:
            raise ValueError("zero-shot manifest byte pin mismatch")


def _validated_zero_shot(stage: Path, cache_sha256: str, development_count: int) -> bool:
    report_path = stage / "zero_shot.json"
    commit_path = stage / "zero_shot_complete.json"
    archive_path = stage / "zero_shot_predictions.npz"
    if not all(path.is_file() for path in (report_path, commit_path, archive_path)):
        return False
    report = json.loads(report_path.read_text())
    commit = json.loads(commit_path.read_text())
    _validate_manifest_byte_pin(report, commit, pc.sha256(HERE / pc.config()["cache"]["final_dir"] / "manifest.json"))
    if (report.get("schema") != "e010_phase4b_zero_shot_v1"
        or report.get("cache_sha256") != cache_sha256
        or report.get("training_started") is not False
        or report.get("authorization") != pc.AUTH
        or commit.get("schema") != "e010_phase4b_zero_shot_commit_v1"
        or commit.get("report_sha256") != pc.sha256(report_path)
        or commit.get("prediction_archive_sha256") != pc.sha256(archive_path)
        or report.get("prediction_archive_sha256") != commit.get("prediction_archive_sha256")
        or commit.get("cache_sha256") != cache_sha256
        or commit.get("development_condition_count") != development_count
        or commit.get("training_started") is not False
        or commit.get("authorization") != pc.AUTH):
        raise ValueError("zero-shot publication identity mismatch")
    with np.load(archive_path, allow_pickle=False) as data:
        if set(data.files) != {"prediction", "offsets", "records_json"}:
            raise ValueError("zero-shot archive fields mismatch")
        records = json.loads(data["records_json"].tobytes())
        offsets = data["offsets"]
        prediction = data["prediction"]
        if (len(records) != development_count or len(offsets) != development_count + 1
            or offsets[0] != 0
            or offsets[-1] != len(prediction) or np.any(np.diff(offsets) <= 0)
            or prediction.ndim != 2 or prediction.shape[1] != 3
            or not np.isfinite(prediction).all()):
            raise ValueError("zero-shot archive count/shape mismatch")
    return True


def _validated_journal(directory: Path, cache_sha256: str, *, final: bool = False) -> tuple[int, str]:
    state, raw, checkpoint, _ = _inspect_transaction(directory, cache_sha256, final=final)
    if state is None:
        raise ValueError("training journal and checkpoint are both required")
    return len(_journal_events(raw, cache_sha256)), pc.sha256(checkpoint)


def monitor() -> dict[str, Any]:
    cfg = pc.config()
    cache_dir = HERE / cfg["cache"]["final_dir"]
    base = {"authorization": pc.AUTH, "cuda_initialized": False}
    if not (cache_dir / "manifest.json").is_file():
        return {**base, "status": "absent"}
    manifest = pc.load_cache_manifest(cache_dir)
    pc.verify_cache(cache_dir, manifest)
    cache_sha256 = pc.digest(pc.canonical(manifest))
    base.update({"cache_manifest": str(cache_dir / "manifest.json"),
                 "cache_records": manifest["record_count"],
                 "cache_manifest_sha256": pc.sha256(cache_dir / "manifest.json"),
                 "cache_manifest_canonical_sha256": cache_sha256,
                 "cache_manifest_mtime_ns": (cache_dir / "manifest.json").stat().st_mtime_ns,
                 "committed_updates": 0})
    if FINAL.exists():
        result_path = FINAL / "results.json"
        if not result_path.is_file():
            raise ValueError(f"training final directory lacks results.json: {FINAL}")
        if not _validated_zero_shot(FINAL, cache_sha256, manifest["development_record_count"]):
            raise ValueError("training final lacks verified zero-shot artifacts")
        updates, checkpoint_sha256 = _validated_journal(FINAL, cache_sha256, final=True)
        result = json.loads(result_path.read_text())
        if (result.get("schema") != "e010_phase4b_real_denoiser_results_v1"
            or result.get("status") != "completed_non_authorizing"
            or result.get("updates") != 1092 or updates != 1092
            or result.get("training_examples") != manifest["training_record_count"]
            or set(result.get("boundaries", {})) != {"364", "728", "1092"}
            or result.get("authorization") != pc.AUTH
            or result.get("prospective_accessed") is not False):
            raise ValueError("training results contract mismatch")
        return {**base, "status": "completed", "updates": updates,
                "committed_updates": updates, "results_sha256": pc.sha256(result_path),
                "latest_checkpoint_sha256": checkpoint_sha256}
    if not STAGE.exists() or not _validated_zero_shot(STAGE, cache_sha256,
                                                      manifest["development_record_count"]):
        return {**base, "status": "cache_complete_awaiting_zero_shot"}
    state, raw, checkpoint, recovery_status = _inspect_transaction(STAGE, cache_sha256)
    if state is not None:
        updates = state["global_update"]
        return {**base, "status": "running_or_recoverable", "journal_updates": len(_journal_events(_journal_bytes(STAGE), cache_sha256)),
                "committed_updates": updates, "recovery_status": recovery_status, "next_action": "--resume",
                "latest_checkpoint_sha256": pc.sha256(checkpoint)}
    return {**base, "status": "zero_shot_complete_awaiting_execution", "journal_updates": 0,
            "recovery_status": recovery_status, "next_action": "--execute"}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("plan-only", "validate-contract", "build-cache", "zero-shot", "execute", "resume", "monitor"):
        group.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    action = (plan_only if args.plan_only else validate if args.validate_contract else build_cache if args.build_cache else
        zero_shot if args.zero_shot else (lambda: execute(resume=False)) if args.execute else (lambda: execute(resume=True)) if args.resume else monitor)
    print(json.dumps(_native(action()), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__": main()
