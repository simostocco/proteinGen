"""Pinned Phase 4B source validation, schedule, and resumable real-denoiser cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
SCHEMA = "e010_phase4b_real_denoiser_v1"
STRATA = ("20-64", "65-128", "129-256", "257-384", "385-500")
AUTH = {"phase4b": False, "prospective": False, "production": False, "downstream": False}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"pinned {label} SHA-256 mismatch: expected {expected}; observed {actual}")
    return actual


def publish_directory_once(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to replace published path: {target}")
    os.replace(source, target)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text())
    if value.get("schema") != SCHEMA or value.get("authorization") != AUTH:
        raise ValueError("Phase 4B config schema/authorization contradiction")
    return value


def _path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError(f"pinned input escapes repository: {relative}")
    return path


def pins() -> dict[str, Any]:
    cfg = config()
    diffusion = cfg["diffusion"]
    e010 = cfg["e010"]
    checks = {
        "diffusion_checkpoint": (diffusion["checkpoint"], diffusion["checkpoint_sha256"]),
        "diffusion_source_config": (diffusion["source_config"], diffusion["source_config_sha256"]),
        "e010_selected_checkpoint": (e010["checkpoint"], e010["checkpoint_sha256"]),
        "e010_phase4a_config": (e010["config"], e010["config_sha256"]),
        "e010_model_source": (e010["model_source"], e010["model_source_sha256"]),
        "e010_review": (e010["review"], e010["review_sha256"]),
        "e010_plan": (cfg["data"]["source_plan"], cfg["data"]["source_plan_sha256"]),
        "e010_training_manifest": (cfg["data"]["training_manifest"], cfg["data"]["training_manifest_sha256"]),
        "e010_development_reference": (cfg["data"]["development_reference"], cfg["data"]["development_reference_sha256"]),
        "e007_denoiser_source": (cfg["data"]["diffusion_model_source"], cfg["data"]["diffusion_model_source_sha256"]),
        "e007_diffusion_equations": (cfg["data"]["diffusion_equations_source"], cfg["data"]["diffusion_equations_source_sha256"]),
        "e007_diffusion_schedule": (cfg["data"]["diffusion_schedule_source"], cfg["data"]["diffusion_schedule_source_sha256"]),
    }
    observed = {}
    for label, (relative, expected) in checks.items():
        path = _path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"pinned {label} missing: {relative}")
        actual = assert_sha256(path, expected, label)
        observed[label] = {"path": relative, "sha256": actual}

    review_path = _path(e010["review"])
    review = json.loads(review_path.read_text())
    if review.get("authorization", {}).get("phase4b") is not False:
        raise ValueError("historical E010 review authorization is not false")
    final_dir = _path(e010["final_dir"])
    final_review = json.loads((final_dir / "scientific_review.json").read_text())
    if final_review.get("selected_checkpoint_sha256") != e010["checkpoint_sha256"]:
        raise ValueError("selected update-1092 checkpoint does not match completed E010 publication")
    if int(final_review.get("selected_optimizer_update", -1)) != 1092:
        raise ValueError("completed E010 publication selected update is not 1,092")
    for rel in (e010["final_dir"] + "/scientific_review.json",):
        observed["e010_final_review"] = {
            "path": rel, "sha256": sha256(_path(rel))
        }
    observed["e010_publication_directory_inventory_sha256"] = directory_hash(final_dir)
    if observed["e010_publication_directory_inventory_sha256"] != e010["final_directory_inventory_sha256"]:
        raise ValueError("completed E010 final directory inventory hash mismatch")
    observed["config_sha256"] = sha256(CONFIG)
    return observed


def directory_hash(path: Path) -> str:
    rows = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rows.append({"path": file.relative_to(path).as_posix(), "sha256": sha256(file)})
    return digest(canonical(rows))


def _stratum(length: int) -> str:
    for label in STRATA:
        lo, hi = map(int, label.split("-"))
        if lo <= length <= hi:
            return label
    raise ValueError(f"length outside authorized E010 strata: {length}")


def build_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reuse the exact 98,280 v2 identity/seed schedule; balance t over each stratum."""
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts import e010_phase4a_multicorruption_v2 as mc
    cfg = config()
    base = cfg["data"]
    train_obj = json.loads(_path(base["training_manifest"]).read_text())
    dev_obj = json.loads(_path(base["development_reference"]).read_text())
    if len(train_obj["identities"]) != 16384 or train_obj.get("total_training_examples") != 98280:
        raise ValueError("authorized multicorruption identity/seed schedule count mismatch")
    train_ids = {str(r["sample_id"]) for r in train_obj["identities"]}
    dev_rows = dev_obj["identities"]
    dev_ids = {str(r["sample_id"]) for r in dev_rows}
    if len(dev_ids) != 320 or len(train_ids) != 16384 or train_ids & dev_ids:
        raise ValueError("training/development identity separation or panel-size failure")
    timesteps = list(map(int, cfg["diffusion"]["timesteps"]))
    if len(timesteps) != 3 or len(set(timesteps)) != 3 or any(t < 0 or t >= 500 for t in timesteps):
        raise ValueError("predeclared diffusion timestep set invalid")

    all_rows = []
    for row in train_obj["identities"]:
        sid, length = str(row["sample_id"]), int(row["length"])
        if _stratum(length) != row["stratum"]:
            raise ValueError(f"training identity length/stratum contradiction: {sid}")
        for i, seed in enumerate(row["corruption_seeds"]):
            all_rows.append({"split": "train", "sample_id": sid, "length": length,
                "stratum": row["stratum"], "source_path": row["source_path"],
                "source_sha256": row["source_sha256"], "seed": int(seed), "schedule_index": i})
    # Reconstruct the historical optimizer-step schedule verbatim.
    source_plan = json.loads(_path(base["source_plan"]).read_text())
    schedule = mc.build_schedule(train_obj["identities"], schedule_seed=int(mc.GLOBAL_SEED))
    schedule_hash = mc.schedule_sha256(schedule)
    if schedule_hash != source_plan["schedule_sha256"]:
        raise ValueError("multicorruption v2 identity/seed schedule hash mismatch")
    by_key = {(r["sample_id"], r["seed"]): r for r in all_rows}
    for update in schedule:
        for label, microbatch in update["stratum_microbatches"].items():
            for position, item in enumerate(microbatch):
                target = by_key.get((str(item["sample_id"]), int(item["corruption_seed"])))
                if target is None or target["schedule_index"] != int(item["corruption_index"]):
                    raise ValueError("historical schedule item is absent from selected seed manifest")
                target["schedule_update"] = int(update["global_update"])
                target["microbatch_position"] = position
    # Assign the three conditions by stratum-wise SHA-ranked round robin. Counts
    # are exact and stable, independent of development outcomes.
    for label in STRATA:
        members = sorted((r for r in all_rows if r["stratum"] == label),
                         key=lambda r: (digest(f"phase4b|{label}|{r['sample_id']}|{r['seed']}".encode()), r["sample_id"], r["seed"]))
        if len(members) != 19656:
            raise ValueError(f"training schedule not length-balanced in {label}")
        for i, row in enumerate(members):
            row["condition_index"] = i % 3
            row["timestep"] = timesteps[i % 3]
    dev_records = []
    for row in dev_rows:
        sid, length = str(row["sample_id"]), int(row["length"])
        if _stratum(length) != row["stratum"]:
            raise ValueError(f"development identity length/stratum contradiction: {sid}")
        for ci, timestep in enumerate(timesteps):
            seed = int.from_bytes(hashlib.sha256(f"phase4b|dev|41045|{sid}|{ci}".encode()).digest()[:8], "big") & ((1 << 63) - 1)
            dev_records.append({"split": "development", "sample_id": sid, "length": length,
                "stratum": row["stratum"], "source_path": row["source_path"],
                "source_sha256": row["source_sha256"], "seed": seed,
                "condition_index": ci, "timestep": timestep})
    all_rows.sort(key=lambda r: (STRATA.index(r["stratum"]), r["sample_id"], r["schedule_index"]))
    dev_records.sort(key=lambda r: (STRATA.index(r["stratum"]), r["sample_id"], r["condition_index"]))
    if len(all_rows) != 98280 or len(dev_records) != 960:
        raise ValueError("real-denoiser cache workload count mismatch")
    return all_rows, dev_records


def validate_contract() -> dict[str, Any]:
    cfg = config()
    observed_pins = pins()
    train, dev = build_records()
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in dev}:
        raise ValueError("split overlap")
    lengths = sum(r["length"] for r in train) + sum(r["length"] for r in dev)
    # coordinates are float32, input and target; masks are packed to one byte.
    expected_uncompressed = lengths * (2 * 3 * 4)
    schedule = []
    for label in STRATA:
        counts = {t: sum(r["timestep"] == t for r in train if r["stratum"] == label) for t in cfg["diffusion"]["timesteps"]}
        if len(set(counts.values())) != 1:
            raise ValueError(f"timestep assignment is not equal within {label}: {counts}")
        schedule.append({"stratum": label, "examples_by_timestep": counts})
    return {"schema": "e010_phase4b_contract_v1", "status": "valid_read_only_contract",
        "pins": observed_pins, "training_identities": len({r["sample_id"] for r in train}),
        "training_examples": len(train), "development_identities": len({r["sample_id"] for r in dev}),
        "development_examples": len(dev), "total_denoiser_forwards": len(train) + len(dev),
        "expected_uncompressed_coordinate_tensor_bytes": expected_uncompressed,
        "training_schedule_sha256": json.loads(_path(cfg["data"]["source_plan"]).read_text())["schedule_sha256"],
        "timestep_balance": schedule, "authorization": AUTH,
        "prospective_accessed": False, "training_started": False}


def reconstruct_x0(x_t: np.ndarray, v: np.ndarray, alpha_bar: float, mask: np.ndarray) -> np.ndarray:
    """Canonical coordinate-v reconstruction; inputs are normalized coordinates."""
    if x_t.shape != v.shape or x_t.ndim != 2 or x_t.shape[-1] != 3 or mask.shape != (x_t.shape[0],):
        raise ValueError("diffusion reconstruction shape contradiction")
    if not (0.0 < alpha_bar <= 1.0) or not np.isfinite(x_t).all() or not np.isfinite(v).all():
        raise ValueError("non-finite diffusion reconstruction input")
    alpha, sigma = np.sqrt(alpha_bar), np.sqrt(1.0 - alpha_bar)
    x0 = (alpha * x_t - sigma * v) * mask[:, None]
    if not np.isfinite(x0).all() or np.any(x0[~mask] != 0):
        raise ValueError("diffusion target reconstruction non-finite or padded")
    return np.asarray(x0, dtype=np.float32)


def forward_noise(clean: np.ndarray, noise: np.ndarray, alpha_bar: float, mask: np.ndarray) -> np.ndarray:
    """Exact centered-coordinate VP forward equation used by E007 training."""
    if clean.shape != noise.shape or clean.ndim != 2 or clean.shape[-1] != 3 or mask.shape != (clean.shape[0],):
        raise ValueError("forward diffusion shape contradiction")
    if not (0.0 < alpha_bar <= 1.0) or not np.isfinite(clean).all() or not np.isfinite(noise).all():
        raise ValueError("forward diffusion input invalid")
    x = (np.sqrt(alpha_bar) * clean + np.sqrt(1.0 - alpha_bar) * noise) * mask[:, None]
    x[mask] -= x[mask].mean(axis=0, keepdims=True)
    if not np.isfinite(x).all() or np.any(x[~mask] != 0):
        raise ValueError("forward diffusion produced invalid padding/output")
    return np.asarray(x, dtype=np.float32)


def validate_diffusion_parameterization(e007_config: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    """Fail closed unless the pinned E007 artifact identifies coordinate v prediction."""
    configured = e007_config.get("prediction_parameterization")
    if configured != "centered_coordinate_v":
        raise ValueError(f"pinned E007 config prediction parameterization mismatch: {configured!r}")
    arm = checkpoint.get("arm")
    if arm != "v_only":
        raise ValueError(f"pinned E007 checkpoint arm is not v_only: {arm!r}")
    # Checkpoint files do not duplicate the configuration field. Their signed
    # arm identity plus the pinned config and source target function establish v.
    checkpoint_parameterization = checkpoint.get("prediction_parameterization")
    if checkpoint_parameterization not in (None, "centered_coordinate_v", "coordinate_v", "v"):
        raise ValueError(f"checkpoint prediction parameterization contradicts E007 config: {checkpoint_parameterization!r}")
    return "centered_coordinate_v"


def validate_canonical_roundtrip() -> dict[str, float]:
    """Exercise E007's actual training target and inverse in deterministic float32."""
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    diffusion = CoordinateVPDiffusion(500)
    clean = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, -0.5, 0.2], [-0.3, 0.8, 1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True, False, False]])
    clean = clean * mask[..., None]
    noise = torch.tensor(
        [[[0.2, -0.7, 0.1], [-0.5, 0.3, 0.8], [0.3, 0.4, -0.9], [0, 0, 0], [0, 0, 0]]],
        dtype=torch.float32,
    )
    noise = noise - (noise * mask[..., None]).sum(1, keepdim=True) / mask.sum(1)[:, None, None]
    noise = noise * mask[..., None]
    errors: dict[str, float] = {}
    for timestep in (50, 250, 450):
        times = torch.tensor([timestep], dtype=torch.long)
        alpha, sigma = diffusion.alpha_sigma(times, clean)
        # Match make_training_batch's centering and target construction exactly,
        # while fixing epsilon deterministically for this algebra audit.
        from protein_distance_diffusion.training.coordinate_diffusion import center_coordinates
        centered_x0 = center_coordinates(clean, mask)
        centered_epsilon = center_coordinates(noise, mask)
        xt = center_coordinates(alpha * centered_x0 + sigma * centered_epsilon, mask)
        target = diffusion.training_target(centered_x0, centered_epsilon, times, mask)
        recovered = diffusion.reconstruct_x0(xt, times, target, mask)
        error = (recovered[mask] - centered_x0[mask]).abs().max().item()
        if error > 2e-6 or torch.count_nonzero(recovered[~mask]).item() != 0:
            raise ValueError(f"canonical E007 target round-trip failed at t={timestep}: {error}")
        errors[str(timestep)] = error
    return errors


def validate_frozen_prediction(prediction: Any, x0_hat: Any, x0: Any, mask: Any) -> float:
    """Validate inference contracts and return masked x0 RMSE without equality gating."""
    import torch

    if prediction.shape != x0_hat.shape or x0.shape != x0_hat.shape or prediction.shape[:-1] != mask.shape or prediction.shape[-1] != 3:
        raise ValueError("frozen denoiser output/x0 shape mismatch")
    if not torch.isfinite(prediction).all() or not torch.isfinite(x0_hat).all():
        raise ValueError("frozen denoiser output/x0 contains non-finite coordinates")
    if torch.count_nonzero(prediction[~mask]).item() != 0 or torch.count_nonzero(x0_hat[~mask]).item() != 0:
        raise ValueError("frozen denoiser prediction/x0 violates exact padding behavior")
    # Inference is deterministic in eval mode for a fixed input. This helper
    # deliberately does not impose a zero-error requirement on learned output.
    return float(torch.sqrt((x0_hat[mask] - x0[mask]).square().mean()).item())


def build_cache(*, device: str = "cuda") -> dict[str, Any]:
    """Generate exact-resumable compressed shards; this is the only denoiser loader."""
    import torch

    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    cfg = config()
    contract = validate_contract()
    out = HERE / cfg["cache"]["staging_dir"]
    final = HERE / cfg["cache"]["final_dir"]
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text())
        verify_cache(final, manifest)
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    staged_manifest = out / "manifest.json"
    if staged_manifest.exists():
        manifest = json.loads(staged_manifest.read_text())
        verify_cache(out, manifest)
        publish_directory_once(out, final)
        return manifest
    if any(out.glob("*.tmp.*")):
        raise ValueError("staging contains interrupted temporary writes; preserve and inspect before resume")
    train, dev = build_records()
    records = train + dev
    shard_size = int(cfg["cache"]["shard_records"])
    e007_cfg = yaml.safe_load(_path(cfg["diffusion"]["source_config"]).read_text())
    model = EquivariantPairCoordinateUNet(**e007_cfg["model"]).to(device)
    payload = torch.load(_path(cfg["diffusion"]["checkpoint"]), map_location="cpu", weights_only=False)
    parameterization = validate_diffusion_parameterization(e007_cfg, payload)
    if parameterization != cfg["diffusion"]["parameterization"]:
        raise ValueError("Phase 4B and pinned E007 prediction parameterizations disagree")
    validate_canonical_roundtrip()
    numerics = e007_cfg["numerics"]
    torch.use_deterministic_algorithms(bool(numerics["deterministic_algorithms"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(numerics["allow_matmul_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(numerics["allow_cudnn_tf32"])
    model.load_state_dict(payload["model"])
    model.eval()
    model.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("frozen E007 model has trainable parameters")
    diffusion = CoordinateVPDiffusion(500)
    scale = float(cfg["diffusion"]["coordinate_scale_angstrom"])
    alpha_cum = diffusion.alphas_cumprod.cpu().numpy()
    entries = []
    for shard_no, start in enumerate(range(0, len(records), shard_size)):
        batch = records[start:start + shard_size]
        shard_path = out / f"shard_{shard_no:05d}.npz"
        entry_path = out / f"shard_{shard_no:05d}.json"
        if shard_path.exists() != entry_path.exists():
            raise ValueError(f"staging contains an uncommitted shard artifact: {shard_no}")
        if shard_path.exists() and entry_path.exists():
            entry = json.loads(entry_path.read_text())
            if sha256(shard_path) != entry.get("archive_sha256") or entry.get("record_count") != len(batch):
                raise ValueError(f"existing cache shard is corrupt: {shard_no}")
            entries.append(entry)
            continue
        coords_list, pred_list, lengths, meta = [], [], [], []
        for row in batch:
            source = _path(row["source_path"])
            if sha256(source) != row["source_sha256"]:
                raise ValueError(f"authorized source hash changed: {row['sample_id']}")
            with np.load(source, allow_pickle=False) as z:
                coords = np.asarray(z["ca_coordinates"], dtype=np.float32)
                mask = np.asarray(z["residue_mask"], dtype=np.bool_)
                sid = str(z["sample_id"].item())
            if sid != row["sample_id"] or coords.shape != (row["length"], 3) or mask.shape != (row["length"],) or not mask.all() or not np.isfinite(coords).all():
                raise ValueError(f"source identity/mask/shape/finiteness contradiction: {sid}")
            target = coords - coords.mean(axis=0, keepdims=True)
            normalized = target / scale
            gen = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
            noise = torch.randn(normalized.shape, generator=gen, dtype=torch.float32).numpy()
            noise -= noise.mean(axis=0, keepdims=True)
            t = int(row["timestep"])
            a, s = float(np.sqrt(alpha_cum[t])), float(np.sqrt(1.0 - alpha_cum[t]))
            xt = a * normalized + s * noise
            xt -= xt.mean(axis=0, keepdims=True)
            length = len(target)
            device_t = torch.tensor([t], dtype=torch.long, device=device)
            lengths_t = torch.tensor([length], dtype=torch.long, device=device)
            mask_t = torch.ones((1, length), dtype=torch.bool, device=device)
            continuity = torch.ones((1, max(length - 1, 0)), dtype=torch.bool, device=device)
            xt_t = torch.from_numpy(np.asarray(xt, dtype=np.float32))[None].to(device)
            with torch.inference_mode():
                prediction = model(xt_t, device_t, lengths_t, mask_t, continuity)["v_prediction"]
                canonical_x0 = diffusion.reconstruct_x0(xt_t, device_t, prediction, mask_t)
            x0_t = torch.from_numpy(np.asarray(normalized, dtype=np.float32))[None].to(device)
            normalized_rmse = validate_frozen_prediction(prediction, canonical_x0, x0_t, mask_t)
            x0n = canonical_x0[0].detach().float().cpu().numpy()
            estimate = x0n * scale
            estimate -= estimate.mean(axis=0, keepdims=True)
            if estimate.shape != target.shape or not np.isfinite(estimate).all():
                raise ValueError("denoiser output invalid")
            coords_list.append(target.astype(np.float32)); pred_list.append(estimate.astype(np.float32)); lengths.append(length)
            meta.append({**row, "target_sha256": digest(np.ascontiguousarray(target, dtype="<f4").tobytes()),
                "prediction_sha256": digest(np.ascontiguousarray(estimate, dtype="<f4").tobytes()),
                "mask_all_valid": True, "mask_sha256": digest(np.ones(length, dtype=np.uint8).tobytes()),
                "denoiser_error_rmse": float(np.sqrt(np.mean((estimate - target) ** 2))),
                "normalized_x0_hat_rmse": normalized_rmse})
        offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
        target_flat = np.concatenate(coords_list, axis=0)
        pred_flat = np.concatenate(pred_list, axis=0)
        tmp = shard_path.with_name(shard_path.name + f".tmp.{os.getpid()}")
        with tmp.open("wb") as f:
            np.savez_compressed(f, target=target_flat, prediction=pred_flat, offsets=offsets,
                                records_json=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, shard_path)
        entry = {"shard": shard_path.name, "archive_sha256": sha256(shard_path), "record_count": len(batch),
                 "first_record": start, "records": meta}
        atomic_json(entry_path, entry)
        entries.append(entry)
    manifest = {"schema": "e010_phase4b_real_denoiser_cache_v1", "contract_sha256": digest(canonical(contract)),
        "diffusion_checkpoint_sha256": cfg["diffusion"]["checkpoint_sha256"], "record_count": len(records),
        "training_schedule_sha256": contract["training_schedule_sha256"],
        "training_record_count": len(train), "development_record_count": len(dev), "shards": entries,
        "error_summary_by_stratum_timestep": _error_summary(entries),
        "authorization": AUTH, "prospective_accessed": False}
    atomic_json(out / "manifest.json", manifest)
    verify_cache(out, manifest)
    publish_directory_once(out, final)
    return manifest


def _error_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for stratum in STRATA:
        result[stratum] = {}
        for timestep in config()["diffusion"]["timesteps"]:
            values = [float(r["denoiser_error_rmse"]) for shard in entries for r in shard["records"]
                      if r["stratum"] == stratum and r["timestep"] == timestep]
            result[stratum][str(timestep)] = {"count": len(values),
                "mean_rmse": float(np.mean(values)) if values else None,
                "median_rmse": float(np.median(values)) if values else None,
                "max_rmse": float(np.max(values)) if values else None}
    return result


_MANIFEST_KEYS = {"schema", "contract_sha256", "diffusion_checkpoint_sha256",
    "record_count", "training_schedule_sha256", "training_record_count",
    "development_record_count", "shards", "error_summary_by_stratum_timestep",
    "authorization", "prospective_accessed"}
_SHARD_KEYS = {"shard", "archive_sha256", "record_count", "first_record", "records"}
_ROW_KEYS = {"split", "sample_id", "length", "stratum", "source_path",
    "source_sha256", "seed", "condition_index", "timestep", "target_sha256",
    "prediction_sha256", "mask_all_valid", "mask_sha256", "denoiser_error_rmse",
    "normalized_x0_hat_rmse"}


def _exact_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing, unexpected = expected - value.keys(), value.keys() - expected
    if missing or unexpected:
        raise ValueError(f"{label} fields: missing={sorted(missing)}, unexpected={sorted(unexpected)}")


def load_cache_manifest(path: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse the single schema emitted by build_cache; preserve its exact contents."""
    value = json.loads((path / "manifest.json").read_text()) if manifest is None else manifest
    _exact_keys(value, _MANIFEST_KEYS, "cache manifest")
    if value["schema"] != "e010_phase4b_real_denoiser_cache_v1":
        raise ValueError(f"unsupported cache manifest schema: {value['schema']!r}")
    if value["authorization"] != AUTH or value["prospective_accessed"] is not False:
        raise ValueError("cache authorization/prospective access mismatch")
    cfg = config()
    if (value["diffusion_checkpoint_sha256"] != cfg["diffusion"]["checkpoint_sha256"]
        or value["training_record_count"] != cfg["data"]["train_example_count"]
        or value["development_record_count"] != cfg["data"]["development_example_count"]
        or value["record_count"] != value["training_record_count"] + value["development_record_count"]):
        raise ValueError("cache checkpoint or split record counts mismatch")
    current_contract = validate_contract()
    if value["training_schedule_sha256"] != current_contract["training_schedule_sha256"]:
        raise ValueError("cache training schedule identity mismatch")
    if value["contract_sha256"] != digest(canonical(current_contract)):
        raise ValueError("cache contract/config/source identity mismatch")
    summary = value["error_summary_by_stratum_timestep"]
    if not isinstance(summary, dict) or set(summary) != set(STRATA):
        raise ValueError("cache error summary strata mismatch")
    for stratum, by_timestep in summary.items():
        if not isinstance(by_timestep, dict) or set(by_timestep) != {str(t) for t in cfg["diffusion"]["timesteps"]}:
            raise ValueError(f"cache error summary timesteps mismatch: {stratum}")
        for timestep, metrics in by_timestep.items():
            _exact_keys(metrics, {"count", "mean_rmse", "median_rmse", "max_rmse"},
                        f"cache error summary {stratum}/{timestep}")
    if not isinstance(value["shards"], list) or not value["shards"]:
        raise ValueError("cache shards must be a nonempty list")
    next_record = 0
    split_counts = {"train": 0, "development": 0}
    timesteps = set(map(int, cfg["diffusion"]["timesteps"]))
    summary_counts = {(stratum, timestep): 0 for stratum in STRATA for timestep in timesteps}
    identities = {"train": set(), "development": set()}
    keys = set()
    names = set()
    for index, shard in enumerate(value["shards"]):
        _exact_keys(shard, _SHARD_KEYS, f"cache shard {index}")
        name = f"shard_{index:05d}.npz"
        if shard["shard"] != name or name in names:
            raise ValueError(f"cache shard name/order mismatch: {shard['shard']!r}")
        names.add(name)
        rows = shard["records"]
        if (not isinstance(rows, list) or not rows or shard["record_count"] != len(rows)
            or shard["first_record"] != next_record):
            raise ValueError(f"cache shard record count/offset mismatch: {name}")
        next_record += len(rows)
        for row in rows:
            split = row.get("split") if isinstance(row, dict) else None
            expected = _ROW_KEYS | ({"schedule_index", "schedule_update", "microbatch_position"}
                                    if split == "train" else set())
            _exact_keys(row, expected, f"cache record in {name}")
            if split not in split_counts:
                raise ValueError(f"invalid cache split label: {split!r}")
            if (not isinstance(row["sample_id"], str) or not row["sample_id"]
                or not isinstance(row["seed"], int) or row["timestep"] not in timesteps
                or row["condition_index"] != list(map(int, cfg["diffusion"]["timesteps"])).index(row["timestep"])
                or row["stratum"] not in STRATA or _stratum(row["length"]) != row["stratum"]
                or row["mask_all_valid"] is not True
                or row["mask_sha256"] != digest(np.ones(row["length"], dtype=np.uint8).tobytes())):
                raise ValueError(f"cache record identity/condition mismatch: {name}")
            key = (row["sample_id"], row["seed"], row["timestep"])
            if key in keys:
                raise ValueError(f"duplicate cache identity/seed/timestep: {key}")
            keys.add(key)
            identities[split].add(row["sample_id"])
            split_counts[split] += 1
            summary_counts[(row["stratum"], row["timestep"])] += 1
    if next_record != value["record_count"] or split_counts != {
        "train": value["training_record_count"],
        "development": value["development_record_count"]}:
        raise ValueError("cache manifest split totals mismatch")
    if identities["train"] & identities["development"]:
        raise ValueError("cache training/development identities overlap")
    if any(summary[stratum][str(timestep)]["count"] != count
           for (stratum, timestep), count in summary_counts.items()):
        raise ValueError("cache error summary record counts mismatch")
    return value


def verify_cache(path: Path, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = load_cache_manifest(path, manifest)
    count = 0
    for shard in manifest["shards"]:
        archive = path / shard["shard"]
        sidecar = archive.with_suffix(".json")
        if not archive.is_file() or sha256(archive) != shard["archive_sha256"]:
            raise ValueError(f"cache archive hash mismatch: {shard['shard']}")
        if not sidecar.is_file() or json.loads(sidecar.read_text()) != shard:
            raise ValueError(f"cache shard sidecar mismatch: {sidecar.name}")
        with np.load(archive, allow_pickle=False) as data:
            if set(data.files) != {"target", "prediction", "offsets", "records_json"}:
                raise ValueError(f"cache archive fields mismatch: {shard['shard']}")
            records = json.loads(data["records_json"].tobytes())
            offsets, target, prediction = data["offsets"], data["target"], data["prediction"]
            if (records != shard["records"] or len(offsets) != len(records) + 1
                or offsets.dtype != np.int64 or target.dtype != np.float32
                or prediction.dtype != np.float32 or data["records_json"].dtype != np.uint8
                or offsets[0] != 0 or offsets[-1] != len(target)
                or np.any(np.diff(offsets) <= 0) or target.shape != prediction.shape
                or target.ndim != 2 or target.shape[1] != 3):
                raise ValueError("cache shard shape/count contradiction")
            if not np.isfinite(target).all() or not np.isfinite(prediction).all():
                raise ValueError("cache contains non-finite coordinates")
            for i, row in enumerate(records):
                a, b = int(offsets[i]), int(offsets[i + 1])
                if b - a != int(row["length"]):
                    raise ValueError("cache record offset/length contradiction")
                if digest(np.ascontiguousarray(target[a:b], dtype="<f4").tobytes()) != row["target_sha256"] or digest(np.ascontiguousarray(prediction[a:b], dtype="<f4").tobytes()) != row["prediction_sha256"]:
                    raise ValueError("cache per-record tensor identity mismatch")
            count += len(records)
    if count != manifest["record_count"]:
        raise ValueError("cache incomplete")
    return {"verified": True, "record_count": count, "sha256": digest(canonical(manifest))}


if __name__ == "__main__":
    print(json.dumps(validate_contract(), indent=2))
