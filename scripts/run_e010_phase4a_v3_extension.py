#!/usr/bin/env python3
"""Run or monitor a prepared E010 Phase 4A v3 extension (execution is explicit)."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protein_distance_diffusion.models.e010_global_equivariant import (  # noqa: E402 - CLI path bootstrap.
    GlobalEquivariantResidual,  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
)
from scripts.prepare_e010_phase4a import (  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
    bootstrap_mean_ci,
    load_cache_entry,
)
from scripts.prepare_e010_phase4a_v3_extension import (  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
    FINAL,
    STAGING,
    V2,
    V2_FINAL,
    V3,
    loadj,
    sha,
    sha_bytes,
)
from scripts.run_e010_phase4a_training_v2 import (  # noqa: E402 - CLI repository-path bootstrap precedes project imports.
    evaluate,
    summarize,
)


def atomic_json(path, obj):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save(path, state):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(state, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def events(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("truncated v3 extension journal")
    out = []
    for line in raw.splitlines():
        row = json.loads(line)
        digest = row.pop("record_sha256", None)
        if digest != sha_bytes(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()):
            raise ValueError("v3 journal record hash mismatch")
        row["record_sha256"] = digest
        out.append(row)
    if [x["extension_update"] for x in out] != list(range(1, len(out) + 1)):
        raise ValueError("v3 journal is not a contiguous extension prefix")
    return out


def append(path, row):
    digest = sha_bytes(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    full = {**row, "record_sha256": digest}
    with path.open("ab") as f:
        f.write((json.dumps(full, sort_keys=True, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def load_cache():
    m = loadj(V2 / "corruption_cache_manifest.json")
    return {x["sample_id"]: x for x in m["entries"]}


def run():
    if not STAGING.is_dir() or FINAL.exists():
        raise ValueError("prepared v3 staging must exist and v3 final must be absent")
    prep = loadj(V3 / "preparation_manifest.json")
    valid = loadj(V3 / "exact_continuation_validation.json")
    if (
        valid.get("status") != "valid"
        or prep.get("training_started") is not False
        or prep.get("authorization") != {"downstream": False, "prospective": False, "phase4b": False}
    ):
        raise ValueError("v3 preparation is not validated and non-authorizing")
    if (
        prep.get("execution_runner_sha256") != sha(Path(__file__))
        or prep.get("config_sha256") != sha(V3 / "phase4a_v3_extension_config.json")
        or prep.get("schedule_sha256") != sha(V3 / "extension_schedule.json")
        or prep.get("exact_continuation_validation_sha256") != sha(V3 / "exact_continuation_validation.json")
    ):
        raise ValueError("prepared v3 execution pins do not match")
    source_pins = {
        "v2_config_sha256": ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml",
        "v2_preparation_manifest_sha256": V2 / "preparation_manifest.json",
        "v2_plan_sha256": V2 / "phase4a_plan.json",
        "v2_input_validation_sha256": V2 / "input_validation.json",
        "v2_corruption_cache_manifest_sha256": V2 / "corruption_cache_manifest.json",
        "v2_cache_validation_sha256": V2 / "cache_validation.json",
        "v2_development_baseline_sha256": V2 / "development_corrupted_baseline.json",
        "v2_training_protocol_sha256": V2 / "training_protocol.json",
        "v2_selected_checkpoint_sha256": V2_FINAL / "selected_checkpoint.pt",
        "v2_exposure50_checkpoint_sha256": V2_FINAL / "checkpoint_at_exposure_50.pt",
        "v2_training_metrics_sha256": V2_FINAL / "training_metrics.json",
        "v2_review_json_sha256": V2 / "phase4a_v2_scientific_review_v1/review.json",
        "v2_review_inventory_sha256": V2 / "phase4a_v2_scientific_review_v1/artifact_inventory.json",
    }
    for name, path in source_pins.items():
        if sha(path) != prep["pins"].get(name):
            raise ValueError(f"v2 continuation input pin changed: {name}")
    if sha(STAGING / "latest.pt") != prep["initial_staging_checkpoint_sha256"]:
        raise ValueError("prepared exposure-50 checkpoint changed")
    cfg = yaml.safe_load((ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml").read_text())
    extension = loadj(V3 / "extension_schedule.json")
    schedule = extension["schedule"]
    base = torch.load(STAGING / "latest.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("v3 training execution requires CUDA; preparation and validation are CPU-only")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = GlobalEquivariantResidual(
        **{k: cfg["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")}
    ).to(device)
    model.load_state_dict(base["model"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training_plan"]["learning_rate"], weight_decay=cfg["training_plan"]["weight_decay"]
    )
    opt.load_state_dict(base["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scaler.load_state_dict(base["scaler"])
    random.setstate(base["python_rng_state"])
    np.random.set_state(base["numpy_rng_state"])
    torch.set_rng_state(base["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(base["torch_cuda_rng_state"])
    # Seed evaluation summaries with the already published exposure-50 record.
    boundary_records = [loadj(STAGING / "development_exposure_50.json")]
    _selected_state = base
    plan = loadj(V2 / "phase4a_plan.json")
    cache = load_cache()
    train_entries = [cache[r["sample_id"]] for r in plan["training"]]
    dev_entries = [cache[r["sample_id"]] for r in plan["development"]]
    dev_baseline = loadj(V2 / "development_corrupted_baseline.json")["per_identity_baseline_and_paired_change_slots"]
    # Exposure-50 train RMSE is measured only in explicit execution mode; the
    # preparation path above never runs inference or touches CUDA.
    train50 = evaluate(model, train_entries, device)
    train50_mean = float(np.mean([r["aligned_rmse_angstrom"] for r in train50]))
    dev50 = boundary_records[0]["metrics"]["refined_mean_aligned_rmse_angstrom"]
    boundary_records[0]["training_metrics"] = {
        "training_mean_aligned_rmse_angstrom": train50_mean,
        "development_mean_aligned_rmse_angstrom": dev50,
        "training_development_rmse_gap_angstrom": dev50 - train50_mean,
    }
    old_losses = loadj(V2_FINAL / "training_metrics.json")["training_loss"]["per_update"]
    boundary_records[0]["training_loss_mean_this_exposure"] = float(np.mean(old_losses[-23:]))
    atomic_json(STAGING / "development_exposure_50.json", boundary_records[0])
    {r["sample_id"]: 0 for r in plan["training"]}
    exposures = dict(base["identity_exposures"])
    losses = list(base.get("loss_history", []))
    journal = STAGING / "journal.jsonl"
    rows = events(journal)
    if rows:
        raise ValueError("v3 execution starts only from the prepared exposure-50 state")
    dev_history = [(50, boundary_records[0]["metrics"]["refined_mean_aligned_rmse_angstrom"])]
    worsened = 0
    stop_reason = "maximum_exposure_reached"
    started = time.time()
    try:
        for offset, item in enumerate(schedule):
            step_loss = 0.0
            total = 0
            opt.zero_grad(set_to_none=True)
            for stratum in cfg["selection"]["strata"]:
                ids = item["stratum_microbatches"][stratum["name"]]
                if not ids:
                    continue
                micro = []
                for sid in ids:
                    target_np, coarse_np, mask_np = load_cache_entry(cache[sid])
                    target = torch.from_numpy(target_np).to(device)
                    coarse = torch.from_numpy(coarse_np).to(device)
                    mask = torch.from_numpy(mask_np).to(device)
                    out = model(coarse[None], mask[None])
                    valid_mask = mask[:, None].to(target.dtype)
                    coord = ((out["prediction"][0] - target).square() * valid_mask).sum() / (3 * mask.sum())
                    resid = (out["delta"][0].square() * valid_mask).sum() / (3 * mask.sum())
                    micro.append(coord + 1e-5 * resid)
                    exposures[sid] += 1
                sl = torch.stack(micro).sum()
                total += len(micro)
                (sl / int(item["effective_batch_sample_count"])).backward()
                step_loss += float(sl.detach())
            if total != item["effective_batch_sample_count"]:
                raise ValueError("extension accumulation count mismatch")
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training_plan"]["gradient_clip_max_norm"])
            if not torch.isfinite(grad):
                raise FloatingPointError("non-finite extension gradient")
            opt.step()
            losses.append(step_loss / total)
            global_update = 1150 + offset + 1
            exposure = item["exposure"]
            # Snapshot every update to preserve exact optimizer and RNG continuation.
            state = {
                **base,
                "global_update": global_update,
                "schedule_cursor": global_update,
                "epoch": exposure,
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer": opt.state_dict(),
                "scheduler": None,
                "scaler": scaler.state_dict(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state().cpu(),
                "torch_cuda_rng_state": [x.cpu() for x in torch.cuda.get_rng_state_all()],
                "sampler_rng_state": extension["sampler_rng_state_after_schedule"],
                "training_schedule": base["training_schedule"] + schedule,
                "identity_exposures": dict(exposures),
                "planned_final_exposures": {sid: 80 for sid in exposures},
                "loss_history": list(losses),
                "authorizes_downstream": False,
                "prospective_split_accessed": False,
            }
            pending = STAGING / "pending.pt"
            save(pending, state)
            cp_hash = sha(pending)
            append(
                journal,
                {
                    "extension_update": offset + 1,
                    "global_update": global_update,
                    "exposure": exposure,
                    "step_in_exposure": item["step_in_exposure"],
                    "mean_structure_loss": losses[-1],
                    "gradient_norm": float(grad),
                    "checkpoint_sha256": cp_hash,
                    "training_started": True,
                    "prospective_split_accessed": False,
                },
            )
            os.replace(pending, STAGING / "latest.pt")
            if item["step_in_exposure"] != 23:
                continue
            if set(exposures.values()) != {exposure}:
                raise ValueError("boundary exposure count mismatch")
            cp = STAGING / f"checkpoint_at_exposure_{exposure:02d}.pt"
            shutil.copy2(STAGING / "latest.pt", cp)
            train_rows_eval = evaluate(model, train_entries, device)
            dev_rows_eval = evaluate(model, dev_entries, device)
            dev_metrics = summarize(dev_rows_eval, dev_baseline, cfg)
            train_mean = float(np.mean([r["aligned_rmse_angstrom"] for r in train_rows_eval]))
            {r["sample_id"]: r for r in dev_rows_eval}
            train_metric = {
                "training_mean_aligned_rmse_angstrom": train_mean,
                "development_mean_aligned_rmse_angstrom": dev_metrics["refined_mean_aligned_rmse_angstrom"],
                "training_development_rmse_gap_angstrom": dev_metrics["refined_mean_aligned_rmse_angstrom"]
                - train_mean,
            }
            boundary = {
                "exposure": exposure,
                "global_update": global_update,
                "checkpoint_sha256": sha(cp),
                "evaluation_complete": True,
                "development_count": len(dev_rows_eval),
                "development_per_identity": [
                    {k: v for k, v in r.items() if k != "predicted_coordinates"} for r in dev_rows_eval
                ],
                "metrics": dev_metrics,
                "training_metrics": train_metric,
                "training_loss_mean_this_exposure": float(np.mean(losses[-23:])),
            }
            atomic_json(STAGING / f"development_exposure_{exposure:02d}.json", boundary)
            boundary_records.append(boundary)
            dev_history.append((exposure, train_metric["development_mean_aligned_rmse_angstrom"]))
            _selected_state = state
            if all(dev_metrics["gate_checks"].values()):
                stop_reason = "all_original_gates_passed"
                break
            worsened = worsened + 1 if dev_history[-1][1] > dev_history[-2][1] else 0
            if worsened >= 2:
                stop_reason = "development_worsened_at_two_consecutive_boundaries"
                break
        # Select the best recorded state among 50 and each evaluated extension boundary.
        candidates = [loadj(STAGING / f"development_exposure_{b['exposure']:02d}.json") for b in boundary_records]
        chosen = min(candidates, key=lambda b: (b["metrics"]["refined_mean_aligned_rmse_angstrom"], b["exposure"]))
        chosen_state = (
            torch.load(
                STAGING / f"checkpoint_at_exposure_{chosen['exposure']:02d}.pt", map_location="cpu", weights_only=False
            )
            if chosen["exposure"] > 50
            else base
        )
        save(STAGING / "selected_checkpoint.pt", chosen_state)
        for b in candidates:
            b["selected"] = b["exposure"] == chosen["exposure"]
            atomic_json(STAGING / f"development_exposure_{b['exposure']:02d}.json", b)
        # Paired identity transitions are deterministic bootstrap resamples.
        transition = []
        v2dev = {e: loadj(V2_FINAL / f"development_exposure_{e:02d}.json") for e in (35, 50)}
        prior = {
            e: {r["sample_id"]: r["aligned_rmse_angstrom"] for r in d["development_per_identity"]}
            for e, d in v2dev.items()
        }
        for b in candidates:
            prior[b["exposure"]] = {r["sample_id"]: r["aligned_rmse_angstrom"] for r in b["development_per_identity"]}
        for lo, hi in zip((35, 50, 60, 70), (50, 60, 70, 80), strict=True):
            if hi not in prior:
                transition.append(
                    {
                        "from_exposure": lo,
                        "to_exposure": hi,
                        "status": "not_evaluated_before_stopping",
                        "paired_percentage_improvement": None,
                        "paired_bootstrap_95_ci": None,
                    }
                )
                continue
            common = sorted(set(prior[lo]) & set(prior[hi]))
            pct = np.asarray([(prior[lo][sid] - prior[hi][sid]) / prior[lo][sid] for sid in common])
            ci = bootstrap_mean_ci(pct, int(cfg["statistics"]["bootstrap_replicates"]), int(cfg["statistics"]["seed"]))
            transition.append(
                {
                    "from_exposure": lo,
                    "to_exposure": hi,
                    "status": "evaluated",
                    "paired_percentage_improvement": float(pct.mean()),
                    "paired_bootstrap_95_ci": ci,
                    "development_mean_rmse_from": float(np.mean(list(prior[lo].values()))),
                    "development_mean_rmse_to": float(np.mean(list(prior[hi].values()))),
                }
            )
        selected_metrics = chosen["metrics"]
        extension_updates = len(events(journal))
        passed_any = any(all(b["metrics"]["gate_checks"].values()) for b in candidates)
        insufficient_80 = any(
            b["exposure"] == 80 and b["metrics"]["rmse_reduction_fraction"] < 0.30 for b in candidates
        )
        classification = (
            "passed_all_original_gates"
            if all(selected_metrics["gate_checks"].values())
            else ("useful_but_insufficient_supervised_refiner" if insufficient_80 else "failed_original_gates")
        )
        metrics = {
            "schema": "e010_phase4a_v3_extension_training_metrics_v1",
            "status": "completed",
            "training_updates": 1150 + extension_updates,
            "extension_updates": extension_updates,
            "sample_exposures": int(sum(exposures.values())),
            "per_identity_exposure_count": {
                "minimum": min(exposures.values()),
                "maximum": max(exposures.values()),
                "all_exactly_equal": len(set(exposures.values())) == 1,
            },
            "evaluated_exposures": [b["exposure"] for b in candidates],
            "stop_reason": stop_reason,
            "selected_exposure": chosen["exposure"],
            "selected_checkpoint_sha256": sha(STAGING / "selected_checkpoint.pt"),
            "selected_gate_checks": selected_metrics["gate_checks"],
            "classification": classification,
            "training_trajectory": [
                {
                    "exposure": b["exposure"],
                    **b["training_metrics"],
                    "mean_training_loss_this_exposure": b.get("training_loss_mean_this_exposure"),
                }
                for b in candidates
            ],
            "development_trajectory": [
                {
                    "exposure": b["exposure"],
                    "mean_aligned_rmse_angstrom": b["metrics"]["refined_mean_aligned_rmse_angstrom"],
                    "original_gate_checks": b["metrics"]["gate_checks"],
                }
                for b in candidates
            ],
            "marginal_improvement_35_to_50_50_to_60_60_to_70_70_to_80": transition,
            "training_development_gap_by_boundary": [
                {"exposure": b["exposure"], **b.get("training_metrics", {})} for b in candidates
            ],
            "boundaries": [{k: v for k, v in b.items() if k != "development_per_identity"} for b in candidates],
            "authorizes_downstream": False,
            "downstream_authorized": False,
            "prospective_authorized": False,
            "phase4b_authorized": False,
            "phase4b_preparation_supported_only_after_human_review": passed_any,
            "prospective_split_accessed": False,
            "priors_used": False,
            "geometry_losses_used": False,
            "sampling_used": False,
            "phase4b_prepared": False,
            "authorization": {"downstream": False, "prospective": False, "phase4b": False},
            "elapsed_seconds": time.time() - started,
        }
        from scripts.e010_phase4a_v3_reporting import render_markdown, validate_adjudication

        metrics = validate_adjudication(metrics)
        atomic_json(STAGING / "training_metrics.json", metrics)
        atomic_json(STAGING / "scientific_review.json", metrics)
        md = [
            "# Phase 4A v3 extension review",
            "",
            f"**Classification:** `{metrics['classification']}`  ",
            f"**Selected exposure:** {chosen['exposure']}  ",
            f"**Stop reason:** `{stop_reason}`  ",
            "**Authorization:** downstream, prospective, and Phase 4B authorization remain false.",
            "",
            "## Training and development trajectory",
            "",
            (
                "| Exposure | Train aligned RMSE | Development aligned RMSE | Development "
                "minus training | Training loss |"
            ),
            "|---:|---:|---:|---:|---:|",
        ]
        for row in metrics["training_trajectory"]:
            md.append(
                f"| {row['exposure']} | {row['training_mean_aligned_rmse_angstrom']:.6f} | "
                f"{row['development_mean_aligned_rmse_angstrom']:.6f} | "
                f"{row['training_development_rmse_gap_angstrom']:.6f} | "
                f"{row['mean_training_loss_this_exposure']:.6g} |"
            )
        md += [
            "",
            "## Marginal paired development improvement",
            "",
            "| Transition | Paired percentage improvement | Deterministic bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
        for row in transition:
            if row.get("status") != "evaluated":
                md.append(f"| {row['from_exposure']} → {row['to_exposure']} | not evaluated | not evaluated |")
            else:
                md.append(
                    f"| {row['from_exposure']} → {row['to_exposure']} | {row['paired_percentage_improvement']:.2%} | "
                    f"{row['paired_bootstrap_95_ci']['ci95_percentile'][0]:.2%} to "
                    f"{row['paired_bootstrap_95_ci']['ci95_percentile'][1]:.2%} |"
                )
        md += ["", "## Original gate results", ""]
        for b in candidates:
            md += (
                [
                    f"### Exposure {b['exposure']}",
                    "",
                    f"Overall development RMSE reduction: {b['metrics']['rmse_reduction_fraction']:.2%}",
                    "",
                    "| Gate | Result |",
                    "|---|---|",
                ]
                + [f"| {k} | {str(v).lower()} |" for k, v in b["metrics"]["gate_checks"].items()]
                + [""]
            )
        try:
            (STAGING / "scientific_review.md").write_text(render_markdown(metrics))
        except Exception as report_exc:
            atomic_json(
                STAGING / "report_generation_failure.json",
                {"error_type": type(report_exc).__name__, "error": str(report_exc)},
            )
        os.replace(STAGING, FINAL)
        return {
            "status": "completed",
            "classification": metrics["classification"],
            "selected_exposure": chosen["exposure"],
            "final_path": FINAL.relative_to(ROOT).as_posix(),
            "authorization": metrics["authorization"],
        }
    except BaseException as exc:
        atomic_json(
            STAGING / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "authorization": {"downstream": False, "prospective": False, "phase4b": False},
            },
        )
        raise


def monitor():
    if FINAL.exists() and STAGING.exists():
        raise ValueError("both v3 final and staging exist")
    if FINAL.exists():
        metrics = loadj(FINAL / "training_metrics.json")
        if metrics.get("status") != "completed" or metrics.get("authorization") != {
            "downstream": False,
            "prospective": False,
            "phase4b": False,
        }:
            raise ValueError("v3 final evidence invalid")
        return {"status": "completed", "metrics": metrics}
    if STAGING.exists():
        ev = events(STAGING / "journal.jsonl")
        return {
            "status": "running" if ev else "prepared",
            "extension_updates": len(ev),
            "maximum_updates": 690,
            "training_started": bool(ev),
            "authorization": {"downstream": False, "prospective": False, "phase4b": False},
        }
    return {
        "status": "prepared" if V3.exists() else "absent",
        "training_started": False,
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--execute", action="store_true")
    g.add_argument("--monitor", action="store_true")
    a = p.parse_args()
    result = run() if a.execute else monitor()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
