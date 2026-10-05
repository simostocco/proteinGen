#!/usr/bin/env python3
"""Execute and monitor the prepared, non-authorizing E010 Phase 4A lifecycle."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
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
    EXPOSURE_BOUNDARIES,
    bootstrap_mean_ci,
    canonical_sha,
    file_sha,
    geometry_metrics,
    kabsch_rmse,
    load_cache_entry,
)

CONFIG = ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml"
OUT = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v2"
FINAL = OUT / "phase4a_training_v2.final"
STAGING = OUT / "phase4a_training_v2.staging"
JOURNAL = STAGING / "journal.jsonl"
COMPAT_REVIEW = OUT / "phase4a_v2_compatibility_review.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    atomic_bytes(path, (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def load_json(path):
    return json.loads(path.read_text())


def verify_file_pin(path: Path, expected_sha256: str, label: str):
    observed = file_sha(path)
    if not isinstance(expected_sha256, str) or observed != expected_sha256:
        raise ValueError(f"sha256 pin mismatch for {label}: expected {expected_sha256}, observed {observed}")
    return observed


def validate_contract(resume_config=None):
    cfg = yaml.safe_load(CONFIG.read_text())
    if yaml.safe_load(CONFIG.read_text()).get("schema") != "e010_phase4a_supervised_generalization_v2":
        raise ValueError("runner requires the exact versioned Phase 4A v2 config")
    prep = load_json(OUT / "preparation_manifest.json")
    pins = {
        "config_sha256": file_sha(CONFIG),
        "preparation_manifest_sha256": file_sha(OUT / "preparation_manifest.json"),
        "phase4a_plan_sha256": file_sha(OUT / "phase4a_plan.json"),
        "input_validation_sha256": file_sha(OUT / "input_validation.json"),
        "cache_manifest_sha256": file_sha(OUT / "corruption_cache_manifest.json"),
        "cache_validation_sha256": file_sha(OUT / "cache_validation.json"),
        "baseline_sha256": file_sha(OUT / "development_corrupted_baseline.json"),
        "resume_manifest_sha256": file_sha(OUT / "resume_manifest.json"),
        "resume_sha256": file_sha(OUT / "resume_update_0000.pt"),
        "training_protocol_sha256": file_sha(OUT / "training_protocol.json"),
    }
    required = {
        "status": "prepared_non_authorizing",
        "training_started": False,
        "prospective_accessed": False,
        "phase4b_prepared": False,
        "resume_update": 0,
        "training_identity_count": 2048,
        "development_identity_count": 320,
        "effective_batch_size": 90,
        "largest_passing_length500_microbatch": 18,
        "planned_epochs": 50,
        "planned_sample_exposures": 102400,
        "planned_optimizer_updates": 1150,
    }
    for k, v in required.items():
        if prep.get(k) != v:
            raise ValueError(f"preparation manifest contract mismatch for {k}: {prep.get(k)!r}")
    prep_pin_map = {
        "config_sha256": "config_sha256",
        "phase4a_plan_sha256": "phase4a_plan_sha256",
        "input_validation_sha256": "input_validation_sha256",
        "cache_manifest_sha256": "cache_manifest_sha256",
        "cache_validation_sha256": "cache_validation_sha256",
        "batch_smoke_sha256": "batch_size_smoke_sha256",
        "training_protocol_sha256": "training_protocol_sha256",
        "development_baseline_sha256": "baseline_sha256",
    }
    if prep.get("runner_sha256") != file_sha(Path(__file__)):
        if resume_config is None:
            raise ValueError("v2 runner source hash differs from the prepared pin; reviewed resume config required")
        reviewed = yaml.safe_load(Path(resume_config).read_text())
        review = load_json(COMPAT_REVIEW)
        if (
            reviewed.get("schema") != "e010_phase4a_exact_continuation_resume_v1"
            or reviewed.get("original_config_sha256") != pins["config_sha256"]
            or reviewed.get("preparation_manifest_sha256") != pins["preparation_manifest_sha256"]
            or reviewed.get("staging_journal_sha256") != file_sha(STAGING / "journal.jsonl")
            or reviewed.get("staging_checkpoint_sha256") != file_sha(STAGING / "latest.pt")
            or reviewed.get("incident_record_sha256") != review.get("incident_record_sha256")
            or reviewed.get("status") != "reviewed_exact_staging_continuation_only"
            or reviewed.get("may_start_fresh_run") is not False
            or reviewed.get("scientific_changes_authorized") is not False
            or review.get("resume_config_sha256") != file_sha(Path(resume_config))
            or review.get("source_diff_sha256") != file_sha(ROOT / review.get("source_diff_path", "missing"))
            or review.get("corrected_runner_sha256") != file_sha(Path(__file__))
            or review.get("old_runner_sha256") != prep.get("runner_sha256")
            or review.get("incident_journal_sha256") != file_sha(STAGING / "journal.jsonl")
            or review.get("incident_checkpoint_sha256") != file_sha(STAGING / "latest.pt")
        ):
            raise ValueError("reviewed resume pin does not authorize this exact staging continuation")
    for manifest_key, _observed_key in prep_pin_map.items():
        path_map = {
            "config_sha256": CONFIG,
            "phase4a_plan_sha256": OUT / "phase4a_plan.json",
            "input_validation_sha256": OUT / "input_validation.json",
            "cache_manifest_sha256": OUT / "corruption_cache_manifest.json",
            "cache_validation_sha256": OUT / "cache_validation.json",
            "batch_smoke_sha256": OUT / "batch_size_smoke.json",
            "training_protocol_sha256": OUT / "training_protocol.json",
            "development_baseline_sha256": OUT / "development_corrupted_baseline.json",
        }
        verify_file_pin(path_map[manifest_key], prep.get(manifest_key), manifest_key)
    plan = load_json(OUT / "phase4a_plan.json")
    input_validation = load_json(OUT / "input_validation.json")
    manifest = load_json(OUT / "corruption_cache_manifest.json")
    cacheval = load_json(OUT / "cache_validation.json")
    baseline = load_json(OUT / "development_corrupted_baseline.json")
    resume_manifest = load_json(OUT / "resume_manifest.json")
    protocol = load_json(OUT / "training_protocol.json")
    resume = torch.load(OUT / "resume_update_0000.pt", map_location="cpu", weights_only=False)
    if (
        resume.get("validation_report_sha256") != pins["input_validation_sha256"]
        or resume.get("preparation_manifest_sha256") != pins["preparation_manifest_sha256"]
        or resume.get("config_sha256") != pins["config_sha256"]
    ):
        raise ValueError("v2 update-0 metadata does not pin the exact config, report, and preparation manifest")
    if (
        resume_manifest.get("preparation_manifest_sha256") != pins["preparation_manifest_sha256"]
        or resume_manifest.get("input_validation_sha256") != pins["input_validation_sha256"]
    ):
        raise ValueError("v2 resume manifest metadata pin mismatch")
    reconciliation_path = ROOT / prep["reconciliation_record_path"]
    verify_file_pin(reconciliation_path, prep["reconciliation_record_sha256"], "reconciliation_record")
    reconciliation = load_json(reconciliation_path)
    old_checkpoint = reconciliation["conflicting_references"]["update_0_checkpoint"]
    verify_file_pin(ROOT / old_checkpoint["path"], old_checkpoint["sha256"], "old_update_0_checkpoint")
    original = torch.load(ROOT / old_checkpoint["path"], map_location="cpu", weights_only=False)

    def scientific_hashes(state):
        return {
            "model": canonical_sha(state["model"]),
            "optimizer": canonical_sha(state["optimizer"]),
            "scheduler": canonical_sha(state.get("scheduler")),
            "scaler": canonical_sha(state.get("scaler", {})),
            "python_rng": canonical_sha(state["python_rng_state"]),
            "numpy_rng": canonical_sha(state["numpy_rng_state"]),
            "torch_cpu_rng": canonical_sha(state["torch_cpu_rng_state"]),
            "torch_cuda_rng": canonical_sha(state["torch_cuda_rng_state"]),
            "sampler_rng": canonical_sha(state["sampler_rng_state"]),
        }

    if (
        scientific_hashes(original) != reconciliation["original_scientific_state_sha256"]
        or scientific_hashes(resume) != reconciliation["original_scientific_state_sha256"]
    ):
        raise ValueError(
            "v2 scientific model, optimizer, scheduler, scaler, or RNG state differs from original update 0"
        )
    if resume.get("scientific_state_sha256") != reconciliation["original_scientific_state_sha256"]:
        raise ValueError("v2 update-0 scientific state metadata hash mismatch")
    if resume_manifest.get("scientific_state_sha256") != reconciliation["original_scientific_state_sha256"]:
        raise ValueError("v2 resume manifest scientific state metadata hash mismatch")
    if resume_manifest.get("state_hashes") != resume.get("state_hashes"):
        raise ValueError("v2 resume manifest state hashes differ from update-0 checkpoint")
    if (
        resume.get("input_validation_sha256") != pins["input_validation_sha256"]
        or input_validation.get("plan_sha256") != pins["phase4a_plan_sha256"]
        or input_validation.get("authorizes_downstream") is not False
        or input_validation.get("training_started") is not False
        or input_validation.get("prospective_split_accessed") is not False
        or input_validation.get("source_payloads_finite_and_complete") is not True
        or input_validation.get("train_count") != 2048
        or input_validation.get("development_count") != 320
        or input_validation.get("identity_overlap") != 0
        or input_validation.get("phase2_corruption_reconstruction", {}).get("exact_reconstruction_verified") is not True
    ):
        raise ValueError("read-only input validation exact pin or semantic contract failed")
    if (
        plan.get("authorizes_downstream") is not False
        or plan.get("prospective_split_accessed") is not False
        or plan.get("training_started") is not False
        or len(plan.get("training", [])) != 2048
        or len(plan.get("development", [])) != 320
    ):
        raise ValueError("prepared panel/authorization contract mismatch")
    if (
        manifest.get("authorizes_downstream") is not False
        or manifest.get("total") != 2368
        or sum(x.get("split") == "train" for x in manifest.get("entries", [])) != 2048
        or sum(x.get("split") == "development" for x in manifest.get("entries", [])) != 320
        or cacheval.get("all_archive_and_tensor_hashes_verified") is not True
        or cacheval.get("count") != 2368
    ):
        raise ValueError("corruption cache identity or hash validation contract mismatch")
    if cacheval.get("cache_manifest_sha256") != pins["cache_manifest_sha256"]:
        raise ValueError("cache validation report does not pin the exact corruption manifest")
    # Re-read each pinned archive without constructing a model or touching CUDA.
    for entry in manifest["entries"]:
        load_cache_entry(entry)
    if (
        baseline.get("split") != "development_only"
        or baseline.get("count") != 320
        or baseline.get("prospective_split_accessed") is not False
    ):
        raise ValueError("development baseline contract mismatch")
    if (
        resume_manifest.get("global_update") != 0
        or resume_manifest.get("microbatch_size") != 18
        or resume_manifest.get("effective_batch_size_nominal") != 90
        or resume_manifest.get("model_state_is_phase3_weights") is not False
        or resume_manifest.get("all_rng_states_complete") is not True
    ):
        raise ValueError("exact fresh update-0 resume contract mismatch")
    if resume_manifest.get("resume_sha256") != pins["resume_sha256"]:
        raise ValueError("resume manifest checkpoint hash mismatch")
    if (
        resume.get("schema") != "e010_phase4a_exact_resume_state_v2"
        or resume.get("global_update") != 0
        or resume.get("microbatch_size") != 18
        or len(resume.get("training_schedule", [])) != 1150
        or resume.get("state_hashes", {}).get("schedule_sha256") != resume_manifest.get("schedule_sha256")
        or resume.get("authorizes_downstream") is not False
    ):
        raise ValueError("update-0 state is inconsistent with exact resume manifest")
    if canonical_sha(resume["model"]) != resume_manifest["state_hashes"]["model_sha256"]:
        raise ValueError("pinned update-0 model state hash mismatch")
    if (
        protocol.get("exposure_boundaries") != list(EXPOSURE_BOUNDARIES)
        or protocol.get("prospective_split_accessed") is not False
    ):
        raise ValueError("training protocol boundary/access contract mismatch")
    tp = cfg["training_plan"]
    model = cfg["model"]
    if (
        model["variant"] != "Large"
        or model["parameter_count"] != 12844352
        or model["width"] != 416
        or model["layers"] != 6
        or tp["epochs"] != 50
        or tp["exposures_per_identity"] != 50
        or tp["objective"].find("masked xyz coordinate MSE") < 0
        or str(tp["scheduler"]).lower() != "none"
    ):
        raise ValueError("config training contract mismatch")
    restricted = cfg["restrictions"]
    for field in (
        "prospective_split_accessed",
        "held_out_data",
        "geometry_priors",
        "geometry_losses",
        "fixed_bonds",
        "identity_embeddings",
        "sequence_features",
        "production_sampler_integration",
        "supervised_training_executed",
        "phase4b_prepared",
    ):
        if restricted.get(field) is not False:
            raise ValueError(f"prohibited Phase 4A config field is not false: {field}")
    if FINAL.exists() and STAGING.exists():
        raise ValueError("both final and staging lifecycle paths exist")
    return {
        "status": "valid",
        "config_sha256": pins["config_sha256"],
        "pins": pins,
        "training_count": 2048,
        "development_count": 320,
        "cache_count": 2368,
        "microbatch_size": 18,
        "effective_batch_size_nominal": 90,
        "planned_optimizer_updates": 1150,
        "exposure_boundaries": list(EXPOSURE_BOUNDARIES),
        "fresh_model_from_pinned_update_0": True,
        "prospective_split_accessed": False,
        "priors_geometry_losses_sampling_phase4b": False,
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }


def load_cache_map():
    manifest = load_json(OUT / "corruption_cache_manifest.json")
    return {x["sample_id"]: x for x in manifest["entries"]}


def evaluate(model, entries, device):
    model.eval()
    records = []
    with torch.no_grad():
        for entry in entries:
            target_np, coarse_np, _ = load_cache_entry(entry)
            mask_np = np.ones(len(target_np), dtype=np.bool_)
            target = torch.from_numpy(target_np).to(device)
            coarse = torch.from_numpy(coarse_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            out = model(coarse[None], mask[None])
            pred = out["prediction"][0]
            finite = bool(torch.isfinite(pred).all())
            rmse = kabsch_rmse(pred.cpu(), target.cpu()) if finite else float("inf")
            geo = geometry_metrics(pred.cpu(), target.cpu()) if finite else {}
            records.append(
                {
                    "sample_id": entry["sample_id"],
                    "length": entry["length"],
                    "stratum": entry["stratum"],
                    "aligned_rmse_angstrom": rmse,
                    "finite": finite,
                    "geometry_telemetry": geo,
                    "predicted_coordinates": pred.detach().cpu().numpy().astype(np.float32),
                }
            )
    model.train()
    return records


def summarize(records, baseline_records, cfg):
    base = {x["sample_id"]: x for x in baseline_records}
    paired = []
    for r in records:
        b = base[r["sample_id"]]
        bg = b["geometry_telemetry"]
        rg = r["geometry_telemetry"]
        paired.append(
            {
                "sample_id": r["sample_id"],
                "length": r["length"],
                "stratum": r["stratum"],
                "baseline_rmse_angstrom": b["corrupted_input_aligned_rmse_angstrom"],
                "refined_rmse_angstrom": r["aligned_rmse_angstrom"],
                "paired_rmse_change_angstrom": r["aligned_rmse_angstrom"] - b["corrupted_input_aligned_rmse_angstrom"],
                "baseline_geometry": bg,
                "refined_geometry": rg,
                "finite": r["finite"],
            }
        )
    before = np.asarray([x["baseline_rmse_angstrom"] for x in paired])
    after = np.asarray([x["refined_rmse_angstrom"] for x in paired])
    change = after - before
    strata = {}
    for s in cfg["selection"]["strata"]:
        subset = [x for x in paired if x["stratum"] == s["name"]]
        b = np.asarray([x["baseline_rmse_angstrom"] for x in subset])
        a = np.asarray([x["refined_rmse_angstrom"] for x in subset])
        strata[s["name"]] = {
            "count": len(subset),
            "baseline_mean_rmse": float(b.mean()),
            "refined_mean_rmse": float(a.mean()),
            "reduction_fraction": float((b.mean() - a.mean()) / b.mean()),
            "paired_change_ci95": bootstrap_mean_ci(
                a - b, cfg["statistics"]["bootstrap_replicates"], cfg["statistics"]["seed"] + len(strata)
            ),
        }
    bmean, amean = float(before.mean()), float(after.mean())
    baseline_slope = float(np.polyfit([x["length"] for x in paired], before, 1)[0])
    refined_slope = float(np.polyfit([x["length"] for x in paired], after, 1)[0])
    baseline_local = np.asarray(
        [sum(x["baseline_geometry"][f"i_plus_{k}_distance_rmse_angstrom"] for k in (1, 2, 3)) / 3 for x in paired]
    )
    refined_local = np.asarray(
        [sum(x["refined_geometry"][f"i_plus_{k}_distance_rmse_angstrom"] for k in (1, 2, 3)) / 3 for x in paired]
    )
    base_inv = sum(x["baseline_geometry"]["chirality_inversions"] for x in paired)
    base_trip = sum(x["baseline_geometry"]["chirality_triplets"] for x in paired)
    ref_inv = sum(x["refined_geometry"]["chirality_inversions"] for x in paired)
    ref_trip = sum(x["refined_geometry"]["chirality_triplets"] for x in paired)
    coord_collapse = [
        x["sample_id"]
        for x in paired
        if x["refined_geometry"]["prediction_radius_gyration_angstrom"] < 1.0
        or x["refined_geometry"]["prediction_radius_gyration_angstrom"]
        / max(x["refined_geometry"]["target_radius_gyration_angstrom"], 1e-8)
        < 0.5
    ]
    diversity = {}
    for name, rows in (
        (s["name"], [x for x in paired if x["stratum"] == s["name"]]) for s in cfg["selection"]["strata"]
    ):
        pr = np.asarray([x["refined_geometry"]["prediction_radius_gyration_angstrom"] for x in rows])
        tr = np.asarray([x["refined_geometry"]["target_radius_gyration_angstrom"] for x in rows])
        ratio = float(pr.std() / tr.std()) if tr.std() > 1e-12 else None
        diversity[name] = {
            "predicted_radius_gyration_sd": float(pr.std()),
            "target_radius_gyration_sd": float(tr.std()),
            "sd_ratio_prediction_to_target": ratio,
            "collapse_indicator": bool(ratio is not None and ratio < 0.5),
        }
    rules = cfg["acceptance"]
    gates = {
        "overall_rmse_reduction": float((bmean - amean) / bmean)
        >= rules["overall_development_rmse_reduction_fraction_min"],
        "every_stratum_reduction": all(
            v["reduction_fraction"] >= rules["every_stratum_rmse_reduction_fraction_min"] for v in strata.values()
        ),
        "no_stratum_worsening": all(v["refined_mean_rmse"] <= v["baseline_mean_rmse"] for v in strata.values()),
        "error_length_slope_not_increased": refined_slope <= baseline_slope,
        "finite_outputs": all(x["finite"] for x in paired),
        "chirality_not_increased": ref_inv / max(ref_trip, 1) <= base_inv / max(base_trip, 1),
        "local_distance_reduction": float((baseline_local.mean() - refined_local.mean()) / baseline_local.mean())
        >= rules["mean_i_plus_1_i_plus_2_i_plus_3_distance_rmse_reduction_fraction_min"],
        "coordinate_collapse_absent": not coord_collapse,
        "diversity_collapse_absent": not any(x["collapse_indicator"] for x in diversity.values()),
    }
    return {
        "count": len(paired),
        "baseline_mean_aligned_rmse_angstrom": bmean,
        "refined_mean_aligned_rmse_angstrom": amean,
        "rmse_reduction_fraction": float((bmean - amean) / bmean),
        "paired_change_ci95": bootstrap_mean_ci(
            change, cfg["statistics"]["bootstrap_replicates"], cfg["statistics"]["seed"]
        ),
        "baseline_error_vs_length_slope": baseline_slope,
        "refined_error_vs_length_slope": refined_slope,
        "per_stratum": strata,
        "baseline_local_distance_rmse_mean": float(baseline_local.mean()),
        "refined_local_distance_rmse_mean": float(refined_local.mean()),
        "baseline_chirality_inversion_rate": base_inv / max(base_trip, 1),
        "refined_chirality_inversion_rate": ref_inv / max(ref_trip, 1),
        "coordinate_collapse_identities": coord_collapse,
        "diversity_by_stratum": diversity,
        "gate_checks": gates,
        "gate_adjudication": "passed" if all(gates.values()) else "failed",
        "paired_per_identity": paired,
    }


def journal_events():
    if not JOURNAL.exists():
        return []
    data = JOURNAL.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("journal has a truncated final record")
    events = []
    for line in data.splitlines():
        item = json.loads(line)
        digest = item.pop("record_sha256", None)
        actual = sha_bytes(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if digest != actual:
            raise ValueError("journal record hash mismatch")
        item["record_sha256"] = digest
        events.append(item)
    if [e["global_update"] for e in events] != list(range(1, len(events) + 1)):
        raise ValueError("journal is not a contiguous update prefix")
    return events


def append_event(event):
    core = dict(event)
    digest = sha_bytes(json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    row = {**core, "record_sha256": digest}
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("ab") as f:
        f.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def save_checkpoint(path, state):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(state, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def recover_state(events, base):
    """Recover the exact journal prefix, including a checkpoint committed before latest.pt."""
    if not events:
        return base
    expected = events[-1]["checkpoint_sha256"]
    latest = STAGING / "latest.pt"
    if not latest.is_file() or file_sha(latest) != expected:
        pending = STAGING / "pending.pt"
        if not pending.is_file() or file_sha(pending) != expected:
            raise ValueError("journal prefix has no matching committed or pending checkpoint")
        os.replace(pending, latest)
    state = torch.load(latest, map_location="cpu", weights_only=False)
    if state.get("global_update") != len(events) or state.get("schedule_cursor") != len(events):
        raise ValueError("latest checkpoint does not match validated journal prefix")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "sampler_rng_state",
        "identity_exposures",
        "training_schedule",
    }
    if not required.issubset(state):
        raise ValueError("latest checkpoint is not a full-state recovery checkpoint")
    return state


def validate_boundary_files(staging=STAGING):
    """A boundary is committed only when its complete record and full checkpoint agree."""
    for path in staging.glob("development_exposure_*.json"):
        boundary = load_json(path)
        exposure = boundary.get("exposure")
        checkpoint = staging / f"checkpoint_at_exposure_{int(exposure):02d}.pt"
        if not checkpoint.is_file() or file_sha(checkpoint) != boundary.get("checkpoint_sha256"):
            raise ValueError(f"incomplete or corrupt development boundary: {path.name}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (
            state.get("global_update") != boundary.get("global_update")
            or boundary.get("evaluation_complete") is not True
            or len(boundary.get("development_per_identity", [])) != boundary.get("development_count")
        ):
            raise ValueError(f"partial development boundary: {path.name}")


def publish_boundary_record(
    exposure, checkpoint_record, state, evaluate_fn, summarize_fn, development_entries, baseline, cfg, staging=STAGING
):
    """Publish only after a durable full checkpoint and complete evaluation are available."""
    checkpoint_path = Path(checkpoint_record["path"])
    checkpoint_hash = checkpoint_record["sha256"]
    if not checkpoint_path.is_file() or file_sha(checkpoint_path) != checkpoint_hash:
        raise ValueError("boundary checkpoint record does not match a durable checkpoint")
    durable_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if durable_state.get("global_update") != state.get("global_update"):
        raise ValueError("boundary checkpoint cursor differs from in-memory state")
    checkpoint = staging / f"checkpoint_at_exposure_{exposure:02d}.pt"
    save_checkpoint(checkpoint, durable_state)
    dev = evaluate_fn()
    if not dev or len(dev) != len(development_entries):
        raise ValueError("development evaluation is incomplete")
    summary = summarize_fn(dev, baseline, cfg)
    boundary = {
        "exposure": exposure,
        "global_update": durable_state["global_update"],
        "metrics": summary,
        "evaluation_complete": True,
        "development_count": len(dev),
        "development_per_identity": [{k: v for k, v in x.items() if k != "predicted_coordinates"} for x in dev],
        "checkpoint_sha256": file_sha(checkpoint),
        "selected": False,
    }
    atomic_json(staging / f"development_exposure_{exposure:02d}.json", boundary)
    return boundary


def snapshot(model, opt, scaler, base, cursor, exposures, losses, cfg):
    return {
        "schema": "e010_phase4a_exact_resume_state_v1",
        "global_update": cursor,
        "epoch": cursor // 23,
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": opt.state_dict(),
        "scheduler": None,
        "scaler": scaler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": [v.cpu() for v in torch.cuda.get_rng_state_all()],
        "sampler_rng_state": base["sampler_rng_state"],
        "training_schedule": base["training_schedule"],
        "schedule_cursor": cursor,
        "identity_exposures": dict(exposures),
        "planned_final_exposures": base["planned_final_exposures"],
        "loss_history": list(losses),
        "microbatch_size": 18,
        "effective_batch_size_nominal": 90,
        "plan_sha256": base["plan_sha256"],
        "input_validation_sha256": base["input_validation_sha256"],
        "cache_manifest_sha256": base["cache_manifest_sha256"],
        "cache_validation_sha256": base["cache_validation_sha256"],
        "development_baseline_sha256": base["development_baseline_sha256"],
        "authorizes_downstream": False,
        "prospective_split_accessed": False,
    }


def run(resume=False, resume_config=None):
    contract = validate_contract(resume_config)
    cfg = yaml.safe_load(CONFIG.read_text())
    if FINAL.exists():
        raise FileExistsError(f"final Phase 4A result already exists: {FINAL}")
    if resume:
        if not STAGING.is_dir():
            raise FileNotFoundError(f"no resumable staging directory: {STAGING}")
    else:
        if STAGING.exists():
            raise FileExistsError(f"staging exists; use --resume: {STAGING}")
        STAGING.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 4A execution requires CUDA; CPU mode is available only to lifecycle mocks")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    base = torch.load(OUT / "resume_update_0000.pt", map_location="cpu", weights_only=False)
    if resume:
        events = journal_events()
        state = recover_state(events, base)
        validate_boundary_files(STAGING)
        if (STAGING / "failure.json").exists():
            (STAGING / "failure.json").unlink()
    else:
        events = []
        state = base
    spec = {k: cfg["model"][k] for k in ("width", "layers", "heads", "vector_channels", "max_length", "sigma_distance")}
    model = GlobalEquivariantResidual(**spec).to(device)
    model.load_state_dict(state["model"])
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["training_plan"]["learning_rate"], weight_decay=cfg["training_plan"]["weight_decay"]
    )
    opt.load_state_dict(state["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    if "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    if resume:
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
    schedule = base["training_schedule"]
    cursor = state.get("schedule_cursor", state["global_update"])
    exposures = dict(
        state.get("identity_exposures", {x["sample_id"]: 0 for x in load_json(OUT / "phase4a_plan.json")["training"]})
    )
    losses = list(state.get("loss_history", []))
    cache = load_cache_map()
    train_ids = {x["sample_id"]: x for x in load_json(OUT / "phase4a_plan.json")["training"]}
    _entries = [cache[sid] for sid in train_ids]
    dev_entries = [cache[x["sample_id"]] for x in load_json(OUT / "phase4a_plan.json")["development"]]
    baseline = load_json(OUT / "development_corrupted_baseline.json")["per_identity_baseline_and_paired_change_slots"]
    boundaries = set(EXPOSURE_BOUNDARIES)
    _started = time.time()
    model.train()

    def publish_boundary(exposure, checkpoint_record):
        return publish_boundary_record(
            exposure,
            checkpoint_record,
            state,
            lambda: evaluate(model, dev_entries, device),
            summarize,
            dev_entries,
            baseline,
            cfg,
            STAGING,
        )

    try:
        if (
            cursor
            and min(exposures.values()) in boundaries
            and not (STAGING / f"development_exposure_{min(exposures.values()):02d}.json").exists()
        ):
            recovered_path = STAGING / "latest.pt"
            publish_boundary(min(exposures.values()), {"path": recovered_path, "sha256": file_sha(recovered_path)})
        for idx in range(cursor, len(schedule)):
            item = schedule[idx]
            opt.zero_grad(set_to_none=True)
            total_count = 0
            step_loss = 0.0
            for stratum in cfg["selection"]["strata"]:
                ids = item["stratum_microbatches"][stratum["name"]]
                if not ids:
                    continue
                losses_micro = []
                for sid in ids:
                    entry = cache[sid]
                    target_np, coarse_np, mask_np = load_cache_entry(entry)
                    target = torch.from_numpy(target_np).to(device)
                    coarse = torch.from_numpy(coarse_np).to(device)
                    mask = torch.from_numpy(mask_np).to(device)
                    out = model(coarse[None], mask[None])
                    valid = mask[:, None].to(target.dtype)
                    coord = ((out["prediction"][0] - target).square() * valid).sum() / (3 * mask.sum())
                    resid = (out["delta"][0].square() * valid).sum() / (3 * mask.sum())
                    losses_micro.append(coord + 1e-5 * resid)
                    exposures[sid] += 1
                stratum_loss = torch.stack(losses_micro).sum()
                total_count += len(losses_micro)
                (stratum_loss / int(item["effective_batch_sample_count"])).backward()
                step_loss += float(stratum_loss.detach())
            if total_count != item["effective_batch_sample_count"]:
                raise ValueError("schedule accumulation count mismatch")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training_plan"]["gradient_clip_max_norm"]
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at update {idx + 1}")
            opt.step()
            cursor = idx + 1
            losses.append(step_loss / total_count)
            state = snapshot(model, opt, scaler, base, cursor, exposures, losses, cfg)
            pending = STAGING / "pending.pt"
            save_checkpoint(pending, state)
            checkpoint_hash = file_sha(pending)
            append_event(
                {
                    "global_update": cursor,
                    "epoch": item["epoch"],
                    "step_in_epoch": item["step_in_epoch"],
                    "mean_structure_loss": losses[-1],
                    "gradient_norm": float(grad_norm),
                    "checkpoint_sha256": checkpoint_hash,
                    "training_started": True,
                    "prospective_split_accessed": False,
                }
            )
            os.replace(pending, STAGING / "latest.pt")
            checkpoint_record = {"path": STAGING / "latest.pt", "sha256": checkpoint_hash}
            exposure = min(exposures.values())
            if exposure in boundaries and all(v == exposure for v in exposures.values()):
                publish_boundary(exposure, checkpoint_record)
        if set(exposures.values()) != {50}:
            raise ValueError("completed schedule does not expose every identity exactly 50 times")
        boundary_paths = [STAGING / f"development_exposure_{e:02d}.json" for e in EXPOSURE_BOUNDARIES]
        boundaries_data = [load_json(p) for p in boundary_paths]
        selected = min(
            boundaries_data, key=lambda x: (x["metrics"]["refined_mean_aligned_rmse_angstrom"], x["exposure"])
        )
        final_ckpt = STAGING / "selected_checkpoint.pt"
        selected_state = torch.load(
            STAGING / f"checkpoint_at_exposure_{selected['exposure']:02d}.pt", map_location="cpu", weights_only=False
        )
        save_checkpoint(final_ckpt, selected_state)
        # Boundary checkpoint snapshots are preserved explicitly for correct selection.
        for b in boundaries_data:
            b["selected"] = b["exposure"] == selected["exposure"]
            atomic_json(STAGING / f"development_exposure_{b['exposure']:02d}.json", b)
        result = {
            "schema": "e010_phase4a_complete_training_metrics_v1",
            "status": "completed",
            "config_sha256": contract["config_sha256"],
            "training_updates": cursor,
            "sample_exposures": sum(exposures.values()),
            "per_identity_exposure_count": {
                "minimum": min(exposures.values()),
                "maximum": max(exposures.values()),
                "all_exactly_50": True,
            },
            "training_loss": {
                "mean": float(np.mean(losses)),
                "minimum": float(np.min(losses)),
                "maximum": float(np.max(losses)),
                "per_update": losses,
            },
            "boundaries": [{k: v for k, v in b.items() if k != "development_per_identity"} for b in boundaries_data],
            "selected_exposure": selected["exposure"],
            "selected_checkpoint_sha256": file_sha(final_ckpt),
            "selected_metrics": selected["metrics"],
            "authorizes_downstream": False,
            "downstream_authorized": False,
            "prospective_authorized": False,
            "phase4b_authorized": False,
            "authorization": {"downstream": False, "prospective": False, "phase4b": False},
            "prospective_split_accessed": False,
            "priors_used": False,
            "geometry_losses_used": False,
            "sampling_used": False,
            "phase4b_prepared": False,
            "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        atomic_json(STAGING / "training_metrics.json", result)
        os.replace(STAGING, FINAL)
        return {
            "status": "completed",
            "final_path": str(FINAL.relative_to(ROOT)),
            "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        }
    except BaseException as exc:
        atomic_json(
            STAGING / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "global_update": cursor,
                "failure_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "training_started": cursor > 0,
                "prospective_split_accessed": False,
                "authorization": {"downstream": False, "prospective": False, "phase4b": False},
            },
        )
        raise


def monitor(resume_config=None):
    contract = validate_contract(resume_config)
    if FINAL.exists():
        return {
            "status": "completed",
            "path": str(FINAL.relative_to(ROOT)),
            "metrics": load_json(FINAL / "training_metrics.json"),
            "contract": contract,
        }
    if not STAGING.exists():
        return {
            "status": "prepared",
            "staging_path": str(STAGING.relative_to(ROOT)),
            "final_path": str(FINAL.relative_to(ROOT)),
            "contract": contract,
        }
    events = journal_events()
    validate_boundary_files(STAGING)
    base = torch.load(OUT / "resume_update_0000.pt", map_location="cpu", weights_only=False)
    recovered = recover_state(events, base)
    failure = load_json(STAGING / "failure.json") if (STAGING / "failure.json").exists() else None
    latest = STAGING / "latest.pt"
    return {
        "status": "failed" if failure else "running_or_interrupted",
        "staging_path": str(STAGING.relative_to(ROOT)),
        "final_path": str(FINAL.relative_to(ROOT)),
        "validated_journal_updates": len(events),
        "recoverable_global_update": recovered.get("global_update", 0),
        "latest_checkpoint_sha256": file_sha(latest) if latest.exists() else None,
        "failure": failure,
        "completed_development_boundaries": [
            e for e in EXPOSURE_BOUNDARIES if (STAGING / f"development_exposure_{e:02d}.json").exists()
        ],
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-contract", action="store_true")
    g.add_argument("--execute", action="store_true")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--monitor", action="store_true")
    g.add_argument("--plan-only", action="store_true", help="print the pinned execution plan without creating files")
    p.add_argument("--resume-config", type=Path, default=None)
    a = p.parse_args()
    result = (
        validate_contract(a.resume_config)
        if a.validate_contract
        else monitor(a.resume_config)
        if a.monitor
        else run(resume=a.resume, resume_config=a.resume_config)
        if (a.execute or a.resume)
        else {
            "status": "plan_only",
            "contract": validate_contract(a.resume_config),
            "staging_path": str(STAGING.relative_to(ROOT)),
            "final_path": str(FINAL.relative_to(ROOT)),
            "training_started": False,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
