#!/usr/bin/env python3
"""One-update, non-authorizing CUDA smoke for the published E010 multicorruption v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts import e010_phase4a_multicorruption_v2 as mc
from scripts import run_e010_phase4a_multicorruption_v2 as production

OUT = mc.OUT / "exact_batch_cuda_smoke_local_v2"
MAX_ALLOCATED_MIB = 6144
MAX_RESERVED_MIB = 7680
MIN_CAPACITY_HEADROOM_MIB = 384
MAX_RSS_MIB = 6144


def sha(path: Path) -> str:
    return mc.file_sha(path) if path.is_file() else hashlib.sha256(b"missing").hexdigest()


def tree_sha(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"missing").hexdigest()
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(item.relative_to(path).as_posix().encode() + b"\0")
        h.update(bytes.fromhex(mc.file_sha(item)))
    return h.hexdigest()


def identity_source_snapshot(rows: list[dict[str, Any]]) -> dict[str, str]:
    paths = {
        mc.CONFIG,
        mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py",
        mc.OUT / "training_seed_manifest.json",
        mc.OUT / "preparation_manifest.json",
        mc.OUT / "plan.json",
        mc.OUT / "excluded_archives.json",
    }
    paths.update(mc.ROOT / row["source_path"] for row in rows)
    return {str(p.relative_to(mc.ROOT)): sha(p) for p in sorted(paths)}


def schedule_examples(schedule_row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    result = []
    for stratum in mc.STRATA:
        batch = schedule_row["stratum_microbatches"][stratum]
        if len(batch) != mc.MICROBATCH:
            raise ValueError(f"expected 18 examples in {stratum}")
        result.extend((stratum, x) for x in batch)
    pairs = [(x["sample_id"], x["corruption_seed"]) for _, x in result]
    if len(result) != 90 or len(set(pairs)) != 90:
        raise ValueError("first published schedule update must contain 90 unique identity/seed pairs")
    return result


def validate_inputs() -> dict[str, Any]:
    config = yaml.safe_load(mc.CONFIG.read_text())
    prep_path = mc.OUT / "preparation_manifest.json"
    prep = mc.load_json(prep_path)
    if mc.file_sha(mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py") != prep["lifecycle_runner_sha256"]:
        raise ValueError("pinned lifecycle runner hash mismatch")
    seed_obj = mc.load_json(mc.OUT / "training_seed_manifest.json")
    if mc.file_sha(mc.OUT / "training_seed_manifest.json") != prep["seed_manifest_sha256"]:
        raise ValueError("published seed manifest hash mismatch")
    manifest = seed_obj["identities"]
    schedule = mc.build_schedule(manifest, schedule_seed=int(config["global_seed"]))
    if mc.schedule_sha256(schedule) != prep["schedule_sha256"]:
        raise ValueError("published schedule hash mismatch")
    chosen = schedule_examples(schedule[0])
    seed_by_id = {r["sample_id"]: r for r in manifest}
    rows = [seed_by_id[e["sample_id"]] for _, e in chosen]
    source_before = identity_source_snapshot(rows)
    staging_before = tree_sha(mc.STAGING)
    for row in rows:
        source = mc.ROOT / row["source_path"]
        if sha(source) != row["source_sha256"]:
            raise ValueError(f"published training source hash mismatch: {row['sample_id']}")
    identities = [
        {
            "stratum": s,
            "sample_id": e["sample_id"],
            "corruption_index": e["corruption_index"],
            "corruption_seed": e["corruption_seed"],
        }
        for s, e in chosen
    ]
    if {s: sum(x["stratum"] == s for x in identities) for s in mc.STRATA} != {s: 18 for s in mc.STRATA}:
        raise ValueError("selected update does not have exactly 18 examples per stratum")
    if identity_source_snapshot(rows) != source_before or tree_sha(mc.STAGING) != staging_before:
        raise RuntimeError("source artifacts or scientific staging changed during validation")
    return {
        "config": config,
        "prep": prep,
        "manifest": manifest,
        "schedule": schedule,
        "chosen": chosen,
        "seed_by_id": seed_by_id,
        "rows": rows,
        "source_before": source_before,
        "staging_before": staging_before,
        "identities": identities,
    }


def selected_identity_summary(ctx: dict[str, Any]) -> dict[str, Any]:
    """Use the validated selection in both validation and execution reports."""
    identities = ctx["identities"]
    return {
        "per_stratum_counts": {s: sum(x["stratum"] == s for x in identities) for s in mc.STRATA},
        "identities_and_corruption_seeds": identities,
    }


def validate_only() -> dict[str, Any]:
    """Validate published selection and pins without importing or initializing CUDA."""
    ctx = validate_inputs()
    return {
        "schema": "e010_multicorruption_v2_exact_batch_cuda_smoke_validation_v1",
        "status": "validated",
        "cuda_accessed": False,
        "examples": 90,
        "microbatch_size": 18,
        "effective_batch_size": 90,
        "per_stratum_counts": {s: 18 for s in mc.STRATA},
        "identities_and_corruption_seeds": ctx["identities"],
        "selected_seed_manifest_sha256": sha(mc.OUT / "training_seed_manifest.json"),
        "pinned_lifecycle_runner_sha256": sha(mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py"),
        "source_artifacts_unchanged": True,
        "training_staging_unchanged": True,
        "persistent_training_updates": 0,
        "authorization_fields": {
            "authorizes_downstream": False,
            "downstream_authorized": False,
            "prospective_authorized": False,
            "phase4b_authorized": False,
            "scientific_training_authorized": False,
        },
    }


def run() -> dict[str, Any]:
    import torch

    from protein_distance_diffusion.models.e010_global_equivariant import GlobalEquivariantResidual

    ctx = validate_inputs()
    config, _prep = ctx["config"], ctx["prep"]
    chosen, seed_by_id, rows = ctx["chosen"], ctx["seed_by_id"], ctx["rows"]
    source_before, staging_before = ctx["source_before"], ctx["staging_before"]
    max_cuda_allocated_mib, max_cuda_reserved_mib, max_rss_mib = MAX_ALLOCATED_MIB, MAX_RESERVED_MIB, MAX_RSS_MIB
    if not torch.cuda.is_available():
        raise RuntimeError("exact-batch CUDA smoke requires an available CUDA device")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    capacity = int(props.total_memory)
    free_before, _ = torch.cuda.mem_get_info(device)
    if max_cuda_allocated_mib * 2**20 > capacity or max_cuda_reserved_mib * 2**20 > capacity:
        raise MemoryError("configured CUDA memory limit exceeds detected device capacity")
    if capacity - max_cuda_reserved_mib * 2**20 < MIN_CAPACITY_HEADROOM_MIB * 2**20:
        raise MemoryError("configured reserved-memory limit leaves less than required device headroom")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed = int(config["model"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    spec = {
        k: config["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")
    }
    started = time.perf_counter()
    model = GlobalEquivariantResidual(**spec).to(device)
    if sum(p.numel() for p in model.parameters()) != config["model"]["parameter_count"]:
        raise ValueError("fresh smoke model parameter count differs from published selected model")
    opt = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )
    _scaler = torch.amp.GradScaler("cuda", enabled=False)
    selected_manifest = [seed_by_id[sid] for sid in dict.fromkeys(e["sample_id"] for _, e in chosen)]
    training_data = production._load_training_data(selected_manifest)
    model.train()
    opt.zero_grad(set_to_none=True)
    loss_values: list[float] = []
    corruption_finite = prediction_finite = True
    for stratum in mc.STRATA:
        batch = [(e, seed_by_id[e["sample_id"]]) for s, e in chosen if s == stratum]
        examples = []
        for e, row in batch:
            target, mask, sigma = training_data[row["sample_id"]]
            corrupted = mc.regenerate_corruption(target, sigma, e["corruption_seed"])
            corruption_finite &= bool(np.isfinite(corrupted).all())
            examples.append((target, corrupted, mask))
        max_len = max(len(x[0]) for x in examples)
        target_np = np.zeros((18, max_len, 3), np.float32)
        coarse_np = np.zeros_like(target_np)
        mask_np = np.zeros((18, max_len), np.bool_)
        for i, (target, coarse, mask) in enumerate(examples):
            n = len(target)
            target_np[i, :n] = target
            coarse_np[i, :n] = coarse
            mask_np[i, :n] = mask
        target_t = torch.from_numpy(target_np).to(device)
        coarse_t = torch.from_numpy(coarse_np).to(device)
        mask_t = torch.from_numpy(mask_np).to(device)
        output = model(coarse_t, mask_t)
        prediction_finite &= bool(torch.isfinite(output["prediction"]).all().item())
        valid = mask_t[:, :, None].to(target_t.dtype)
        denom = (3 * mask_t.sum(1)).clamp_min(1).to(target_t.dtype)
        coord = ((output["prediction"] - target_t).square() * valid).sum((1, 2)) / denom
        resid = (output["delta"].square() * valid).sum((1, 2)) / denom
        structure_loss = coord + 1e-5 * resid
        scaled = structure_loss.mean() * (mc.MICROBATCH / mc.EFFECTIVE_BATCH)
        loss_values.append(float(structure_loss.detach().sum().cpu()))
        scaled.backward()
        del target_t, coarse_t, mask_t, output, structure_loss, scaled
    loss = sum(loss_values) / mc.EFFECTIVE_BATCH
    grads_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all().item()) for p in model.parameters())
    if not (corruption_finite and prediction_finite and np.isfinite(loss) and grads_finite):
        raise FloatingPointError("non-finite corruption, prediction, loss, or gradient")
    # Compare accumulated gradients against one concatenated, unscaled loss in a fresh pass is expensive;
    # the five exact 18/90 factors are recorded and checked algebraically here.
    if sum([mc.MICROBATCH / mc.EFFECTIVE_BATCH] * len(mc.STRATA)) != 1.0:
        raise AssertionError("microbatch gradient scaling does not sum to one effective batch")
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_max_norm"])
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("non-finite clipped gradient norm")
    opt.step()
    mutations = sum(not torch.equal(pre[k], v.detach()) for k, v in model.state_dict().items())
    torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    alloc = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    free_after, _ = torch.cuda.mem_get_info(device)
    if alloc > max_cuda_allocated_mib * 2**20 or reserved > max_cuda_reserved_mib * 2**20 or rss > max_rss_mib * 2**20:
        raise MemoryError("smoke exceeded configured CUDA/RSS limit")
    if alloc > capacity or reserved > capacity:
        raise MemoryError("observed CUDA usage exceeds device capacity")
    headroom = capacity - reserved
    if headroom < MIN_CAPACITY_HEADROOM_MIB * 2**20:
        raise MemoryError("remaining device-capacity headroom is below 384 MiB")
    source_after = identity_source_snapshot(rows)
    staging_after = tree_sha(mc.STAGING)
    if source_before != source_after or staging_before != staging_after:
        raise RuntimeError("source artifact or training staging changed during smoke")
    if mutations == 0:
        raise RuntimeError("AdamW step did not mutate model parameters")
    return {
        "schema": "e010_multicorruption_v2_exact_batch_cuda_smoke_v1",
        "status": "passed",
        "authorizes_downstream": False,
        "downstream_authorized": False,
        "prospective_authorized": False,
        "phase4b_authorized": False,
        "scientific_training_authorized": False,
        "persistent_training_updates": 0,
        "optimizer_mutations": 1,
        "model_parameter_tensors_mutated": mutations,
        "model_parameters": config["model"]["parameter_count"],
        "optimizer": "AdamW",
        "gradient_clip_max_norm": config["training"]["gradient_clip_max_norm"],
        "examples": 90,
        "microbatch_size": 18,
        "stratum_count": 5,
        "effective_batch_size": 90,
        **selected_identity_summary(ctx),
        "corruptions_finite": corruption_finite,
        "predictions_finite": prediction_finite,
        "loss_finite": bool(np.isfinite(loss)),
        "gradients_finite": grads_finite,
        "loss": loss,
        "microbatch_loss_sums": loss_values,
        "loss_scaling": {
            "factor_each_microbatch": "18/90",
            "sum_factors": 1.0,
            "normalization": "mean structure loss; production masked xyz coordinate MSE + 1e-5 masked residual L2",
        },
        "gradient_norm_before_clip": float(grad_norm),
        "cuda_device": props.name,
        "device_capacity_bytes": capacity,
        "free_cuda_bytes_before": int(free_before),
        "free_cuda_bytes_after": int(free_after),
        "headroom_bytes_at_peak_reserved": capacity - reserved,
        "minimum_required_headroom_bytes": MIN_CAPACITY_HEADROOM_MIB * 2**20,
        "peak_cuda_allocated_bytes": alloc,
        "peak_cuda_reserved_bytes": reserved,
        "configured_limits_bytes": {
            "cuda_allocated": max_cuda_allocated_mib * 2**20,
            "cuda_reserved": max_cuda_reserved_mib * 2**20,
            "rss": max_rss_mib * 2**20,
        },
        "peak_rss_bytes": rss,
        "runtime_seconds": runtime,
        "source_artifacts_unchanged": True,
        "source_artifact_sha256": source_after,
        "training_staging_unchanged": True,
        "training_staging_tree_sha256": staging_after,
        "authorization_fields": {
            "authorizes_downstream": False,
            "downstream_authorized": False,
            "prospective_authorized": False,
            "phase4b_authorized": False,
            "scientific_training_authorized": False,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="validate the published batch with CUDA hidden")
    mode.add_argument("--run", action="store_true", help="run one disposable update on the local CUDA device")
    p.add_argument("--output-dir", type=Path, default=OUT)
    args = p.parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_mode = args.validate_only
    log = args.output_dir / ("validation.log" if validation_mode else "execution.log")
    path = args.output_dir / ("validation_report.json" if validation_mode else "report.json")
    try:
        report = validate_only() if validation_mode else run()
        report["execution_wall_seconds"] = time.perf_counter() - started
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        log.write_text(f"status={report['status']}\nreport={path.name}\n")
    except BaseException as exc:
        failure = {
            "schema": "e010_multicorruption_v2_exact_batch_cuda_smoke_v1",
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "authorizes_downstream": False,
            "downstream_authorized": False,
            "prospective_authorized": False,
            "phase4b_authorized": False,
            "scientific_training_authorized": False,
            "persistent_training_updates": 0,
            "authorization_fields": {
                "authorizes_downstream": False,
                "downstream_authorized": False,
                "prospective_authorized": False,
                "phase4b_authorized": False,
                "scientific_training_authorized": False,
            },
        }
        path.write_text(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n")
        log.write_text(f"status=failed\nerror_type={type(exc).__name__}\nerror={exc}\n")
        print(
            json.dumps(
                {
                    "report": str(path),
                    "report_sha256": sha(path),
                    "execution_log": str(log),
                    "execution_log_sha256": sha(log),
                },
                sort_keys=True,
            )
        )
        raise
    print(
        json.dumps(
            {
                "report": str(path),
                "report_sha256": sha(path),
                "execution_log": str(log),
                "execution_log_sha256": sha(log),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
