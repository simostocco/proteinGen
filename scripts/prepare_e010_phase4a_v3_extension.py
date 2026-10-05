#!/usr/bin/env python3
"""Prepare, but never execute, the exact Phase 4A v3 continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v2"
V2_FINAL = V2 / "phase4a_training_v2.final"
V2_REVIEW = V2 / "phase4a_v2_scientific_review_v1"
V3 = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v3"
STAGING = V3 / "phase4a_training_v3.staging"
FINAL = V3 / "phase4a_training_v3.final"
BOUNDARIES = (60, 70, 80)
STATE_FIELDS = (
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "python_rng_state",
    "numpy_rng_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_state",
    "sampler_rng_state",
)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(x):
    def norm(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().contiguous().cpu()
            return {"tensor": str(t.dtype), "shape": list(t.shape), "sha256": sha_bytes(t.numpy().tobytes())}
        if isinstance(v, dict):
            return {str(k): norm(y) for k, y in sorted(v.items(), key=lambda z: str(z[0]))}
        if isinstance(v, (tuple, list)):
            return [norm(y) for y in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return {"type": type(v).__qualname__, "repr": repr(v)}

    return sha_bytes(json.dumps(norm(x), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def loadj(path):
    return json.loads(Path(path).read_text())


def final_inventory():
    return [
        {"path": p.relative_to(V2_FINAL).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(V2_FINAL.rglob("*"))
        if p.is_file()
    ]


def build_extension_schedule(state, training_rows, strata, microbatch=18):
    by_stratum = {s: [r["sample_id"] for r in training_rows if r["stratum"] == s] for s in strata}
    ids = {sid for values in by_stratum.values() for sid in values}
    old_ids = {
        sid for item in state["training_schedule"] for batch in item["stratum_microbatches"].values() for sid in batch
    }
    if ids != old_ids or len(ids) != 2048:
        raise ValueError("training identity panel differs from exposure-50 schedule")
    rng = random.Random()
    rng.setstate(state["sampler_rng_state"])
    steps = max(math.ceil(len(v) / microbatch) for v in by_stratum.values())
    schedule = []
    for exposure in range(51, 81):
        buckets = {}
        for name in strata:
            values = list(by_stratum[name])
            rng.shuffle(values)
            buckets[name] = [values[i : i + microbatch] for i in range(0, len(values), microbatch)]
        for step in range(steps):
            batches = {name: (buckets[name][step] if step < len(buckets[name]) else []) for name in strata}
            count = sum(map(len, batches.values()))
            if not all(batches.values()) or count <= 0:
                raise ValueError("extension schedule contains an incomplete stratum batch")
            schedule.append(
                {
                    "exposure": exposure,
                    "step_in_exposure": step + 1,
                    "stratum_microbatches": batches,
                    "effective_batch_sample_count": count,
                }
            )
    counts = {sid: 0 for sid in ids}
    for row in schedule:
        for batch in row["stratum_microbatches"].values():
            for sid in batch:
                counts[sid] += 1
    if set(counts.values()) != {30}:
        raise ValueError("extension schedule does not give every training identity 30 additional exposures")
    return schedule, rng.getstate()


def prepare():
    if not V2_FINAL.is_dir() or not V2_REVIEW.is_dir():
        raise FileNotFoundError("completed v2 final and scientific review are required")
    if STAGING.exists() or FINAL.exists() or V3.exists():
        raise FileExistsError(f"refusing to overwrite v3 artifacts: {V3}")
    cfg_path = ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    prep = loadj(V2 / "preparation_manifest.json")
    review = loadj(V2_REVIEW / "review.json")
    metrics = loadj(V2_FINAL / "training_metrics.json")
    loadj(V2 / "phase4a_plan.json")
    if review.get("selected_exposure") != 50 or metrics.get("selected_exposure") != 50:
        raise ValueError("v2 selected exposure is not 50")
    if review.get("classification") != "fail_one_or_more_predeclared_gates":
        raise ValueError("v3 extension requires the recorded v2 insufficient result")
    if metrics.get("authorization") != {"downstream": False, "prospective": False, "phase4b": False}:
        raise ValueError("v2 authorization is not fully false")
    if prep.get("training_identity_count") != 2048 or prep.get("development_identity_count") != 320:
        raise ValueError("v2 panel counts do not match continuation contract")
    # Verify v2 bytes against the inventory recorded before the prior code edits.
    recorded = loadj(V2_REVIEW / "artifact_inventory.json")["source_final_recursive_inventory"]["files"]
    current = final_inventory()
    if recorded != current:
        raise ValueError("completed v2 final directory no longer matches its preserved recursive inventory")
    source_selected = V2_FINAL / "selected_checkpoint.pt"
    source_boundary = V2_FINAL / "checkpoint_at_exposure_50.pt"
    selected = torch.load(source_selected, map_location="cpu", weights_only=False)
    boundary = torch.load(source_boundary, map_location="cpu", weights_only=False)
    if canonical(selected) != canonical(boundary):
        raise ValueError("selected checkpoint is not the exact exposure-50 full state")
    if (
        selected.get("global_update") != 1150
        or selected.get("schedule_cursor") != 1150
        or selected.get("identity_exposures", {}).keys() != boundary.get("identity_exposures", {}).keys()
        or set(selected["identity_exposures"].values()) != {50}
    ):
        raise ValueError("exposure-50 continuation cursors or exposure counts are invalid")
    if sum(t.numel() for t in selected["model"].values()) != 12844352:
        raise ValueError("selected model parameter count mismatch")
    for field in STATE_FIELDS:
        if field not in selected:
            raise ValueError(f"selected continuation state missing {field}")
    if selected["scheduler"] is not None or selected["scaler"] != {}:
        raise ValueError("v2 scheduler/scaler state differs from pinned continuation state")
    plan = loadj(V2 / "phase4a_plan.json")
    strata = [s["name"] for s in cfg["selection"]["strata"]]
    schedule, rng_end = build_extension_schedule(selected, plan["training"], strata, 18)
    V3.mkdir(parents=True)
    STAGING.mkdir()
    shutil.copy2(source_selected, STAGING / "latest.pt")
    shutil.copy2(source_selected, STAGING / "checkpoint_at_exposure_50.pt")
    shutil.copy2(V2_FINAL / "development_exposure_50.json", STAGING / "development_exposure_50.json")
    schedule_path = V3 / "extension_schedule.json"
    write_json(
        schedule_path,
        {
            "schema": "e010_phase4a_v3_extension_schedule_v1",
            "start_exposure": 50,
            "maximum_exposure": 80,
            "boundaries": list(BOUNDARIES),
            "optimizer_updates": len(schedule),
            "microbatch_size": 18,
            "effective_batch_size_nominal": 90,
            "schedule": schedule,
            "sampler_rng_state_after_schedule": rng_end,
        },
    )
    v3_config = {
        "schema": "e010_phase4a_supervised_generalization_v3_extension",
        "base_config_path": "configs/e010_phase4a_supervised_generalization_v2.yaml",
        "base_config_sha256": sha(cfg_path),
        "starting_exposure": 50,
        "maximum_exposure": 80,
        "evaluate_after_exposures": list(BOUNDARIES),
        "training_plan": {
            k: cfg["training_plan"][k]
            for k in (
                "optimizer",
                "learning_rate",
                "weight_decay",
                "scheduler",
                "gradient_clip_max_norm",
                "objective",
                "equal_structure_weight",
                "length_buckets",
                "effective_batch_size_rule",
            )
        },
        "model": cfg["model"],
        "panel": {
            "training_identity_count": 2048,
            "development_identity_count": 320,
            "phase4a_plan_sha256": sha(V2 / "phase4a_plan.json"),
        },
        "statistics": cfg["statistics"],
        "acceptance": cfg["acceptance"],
        "restrictions": {k: False for k in cfg["restrictions"]},
        "stopping": {
            "stop_after_first_boundary_where_every_original_gate_passes": True,
            "stop_after_two_consecutive_development_rmse_worsening_boundaries": True,
            "maximum_exposure": 80,
            "no_further_extension_if_overall_reduction_below_30_percent_at_80": True,
        },
        "selection": (
            "lowest development mean aligned RMSE among exposures 50/60/70/80; earliest boundary wins exact ties"
        ),
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }
    write_json(V3 / "phase4a_v3_extension_config.json", v3_config)
    pins = {
        "v2_config_sha256": sha(cfg_path),
        "v2_preparation_manifest_sha256": sha(V2 / "preparation_manifest.json"),
        "v2_plan_sha256": sha(V2 / "phase4a_plan.json"),
        "v2_input_validation_sha256": sha(V2 / "input_validation.json"),
        "v2_corruption_cache_manifest_sha256": sha(V2 / "corruption_cache_manifest.json"),
        "v2_cache_validation_sha256": sha(V2 / "cache_validation.json"),
        "v2_development_baseline_sha256": sha(V2 / "development_corrupted_baseline.json"),
        "v2_training_protocol_sha256": sha(V2 / "training_protocol.json"),
        "v2_selected_checkpoint_sha256": sha(source_selected),
        "v2_exposure50_checkpoint_sha256": sha(source_boundary),
        "v2_training_metrics_sha256": sha(V2_FINAL / "training_metrics.json"),
        "v2_review_json_sha256": sha(V2_REVIEW / "review.json"),
        "v2_review_inventory_sha256": sha(V2_REVIEW / "artifact_inventory.json"),
        "v2_final_recursive_inventory": current,
        "selected_state_component_sha256": {f: canonical(selected[f]) for f in STATE_FIELDS},
    }
    validation = {
        "schema": "e010_phase4a_v3_exact_continuation_validation_v1",
        "status": "valid",
        "source_exposure": 50,
        "global_update": 1150,
        "training_identity_count": 2048,
        "all_training_identities_at_50_exposures": True,
        "model_parameter_count": 12844352,
        "selected_checkpoint_matches_exposure50_full_state": True,
        "validated_state_components": list(STATE_FIELDS),
        "optimizer_continuation_validated": True,
        "scheduler_state_preserved": True,
        "scaler_state_preserved": True,
        "python_numpy_torch_cpu_torch_cuda_and_sampler_rng_states_preserved": True,
        "extension_schedule_updates": len(schedule),
        "extension_schedule_covers_each_training_identity_once_per_exposure": True,
        "extension_schedule_boundaries": list(BOUNDARIES),
        "training_panel_sha256": pins["v2_plan_sha256"],
        "all_authorization_false": True,
        "training_started": False,
        "prospective_accessed": False,
        "phase4b_prepared": False,
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }
    write_json(V3 / "exact_continuation_validation.json", validation)
    manifest = {
        "schema": "e010_phase4a_v3_non_authorizing_preparation_manifest_v1",
        "status": "prepared_non_authorizing",
        "pins": pins,
        "preparation_runner_sha256": sha(Path(__file__)),
        "execution_runner_sha256": sha(ROOT / "scripts/run_e010_phase4a_v3_extension.py"),
        "config_sha256": sha(V3 / "phase4a_v3_extension_config.json"),
        "schedule_sha256": sha(schedule_path),
        "exact_continuation_validation_sha256": sha(V3 / "exact_continuation_validation.json"),
        "staging_path": STAGING.relative_to(ROOT).as_posix(),
        "final_path": FINAL.relative_to(ROOT).as_posix(),
        "initial_staging_checkpoint_sha256": sha(STAGING / "latest.pt"),
        "selected_exposure": 50,
        "maximum_exposure": 80,
        "training_identity_count": 2048,
        "development_identity_count": 320,
        "training_started": False,
        "prospective_accessed": False,
        "priors_used": False,
        "geometry_losses_used": False,
        "sampling_used": False,
        "phase4b_prepared": False,
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
    }
    write_json(V3 / "preparation_manifest.json", manifest)
    inventory = []
    for p in sorted(V3.rglob("*")):
        if p.is_file():
            inventory.append({"path": p.relative_to(V3).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)})
    write_json(
        V3 / "artifact_inventory.json",
        {"schema": "e010_phase4a_v3_preparation_inventory_v1", "files": inventory, "final_directory_created": False},
    )
    return {
        "status": "prepared_non_authorizing",
        "v3_path": V3.relative_to(ROOT).as_posix(),
        "validation": validation,
        "final_path": FINAL.relative_to(ROOT).as_posix(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true", required=True)
    print(json.dumps(prepare(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
