#!/usr/bin/env python3
"Prepare and validate E010 multi-corruption generalization inputs, without training or CUDA."

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts import e010_phase4a_multicorruption as mc


def _dev_reference() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = mc.load_json(mc.V2 / "phase4a_plan.json")
    inputs = mc.load_json(mc.V2 / "input_validation.json")
    baseline = mc.load_json(mc.V2 / "development_corrupted_baseline.json")
    cache = mc.load_json(mc.V2 / "corruption_cache_manifest.json")
    if len(plan.get("development", [])) != 320 or inputs.get("development_count") != 320:
        raise ValueError("preserved Phase 4A development panel is not exactly 320 identities")
    panel_ids = [str(r["sample_id"]) for r in plan["development"]]
    if len(set(panel_ids)) != 320 or set(panel_ids) != {
        r["sample_id"] for r in baseline["per_identity_baseline_and_paired_change_slots"]
    }:
        raise ValueError("Phase 4A development panel or baseline identity set differs")
    input_pins = {(r["split"], r["sample_id"]): r for r in inputs["selected_source_pins"]}
    baseline_rows = {r["sample_id"]: r for r in baseline["per_identity_baseline_and_paired_change_slots"]}
    cache_rows = {r["sample_id"]: r for r in cache["entries"] if r["split"] == "development"}
    rows = []
    for row in plan["development"]:
        sid = str(row["sample_id"])
        pin = input_pins[("development", sid)]
        entry = cache_rows.get(sid)
        if entry is None or baseline_rows[sid].get("sample_id") != sid:
            raise ValueError(f"missing fixed Phase 4A baseline/corruption reference for {sid}")
        rows.append(
            {
                "sample_id": sid,
                "stratum": row["stratum"],
                "length": int(row["length"]),
                "source_path": row["source_path"],
                "source_sha256": pin["source_sha256"],
                "selection_rank_sha256": row["selection_rank_sha256"],
                "phase4a_baseline_rmse_angstrom": baseline_rows[sid]["corrupted_input_aligned_rmse_angstrom"],
                "phase4a_baseline_geometry": baseline_rows[sid]["geometry_telemetry"],
                "primary_corruption_seed": int(entry["seed"]),
                "primary_corruption_archive": entry["archive"],
                "primary_corruption_archive_sha256": entry["archive_sha256"],
                "primary_corruption_sha256": entry["corruption_sha256"],
                "target_sha256": entry["target_sha256"],
            }
        )
    rows.sort(key=lambda r: (mc.STRATA.index(r["stratum"]), r["sample_id"]))
    return rows, {
        "source_plan_sha256": mc.file_sha(mc.V2 / "phase4a_plan.json"),
        "source_input_validation_sha256": mc.file_sha(mc.V2 / "input_validation.json"),
        "source_baseline_sha256": mc.file_sha(mc.V2 / "development_corrupted_baseline.json"),
        "source_corruption_manifest_sha256": mc.file_sha(mc.V2 / "corruption_cache_manifest.json"),
        "development_identity_sha256": mc.sha_bytes("\n".join(panel_ids).encode()),
    }


def _select_training(config: dict[str, Any], dev_ids: set[str]) -> list[dict[str, Any]]:
    v2_plan = mc.load_json(mc.V2 / "phase4a_plan.json")
    pool_path = mc.ROOT / v2_plan["candidate_pool_path"]
    pool = mc.load_json(pool_path)
    selected = []
    for stratum in mc.STRATA:
        candidates = []
        for row in pool["train"][stratum]:
            sid = str(row["sample_id"])
            if sid in dev_ids:
                continue
            candidates.append(
                {
                    **row,
                    "sample_id": sid,
                    "selection_rank_sha256": mc.hash_rank(config["plan"]["namespace"], "train", sid),
                    "stratum": stratum,
                    "split": "train",
                }
            )
        candidates.sort(key=lambda r: (r["selection_rank_sha256"], r["sample_id"]))
        required = int(config["plan"]["training_counts"][stratum])
        if len(candidates) < required:
            raise ValueError(f"authorized train candidate pool short in {stratum}: {len(candidates)} < {required}")
        # All selected source archives are validated read-only. No corrupted tensors are cached.
        selected_in_stratum = 0
        for row in candidates:
            try:
                _, _ = mc.source_arrays(row)
            except (OSError, ValueError, KeyError, EOFError):
                continue
            row["source_sha256"] = mc.file_sha(mc.ROOT / row["source_path"])
            selected.append(row)
            selected_in_stratum += 1
            if selected_in_stratum % 500 == 0:
                print(f"validated {selected_in_stratum}/{required} sources in {stratum}", file=sys.stderr, flush=True)
            if selected_in_stratum == required:
                break
        actual = sum(x["stratum"] == stratum for x in selected)
        if actual != required:
            raise ValueError(f"valid authorized training identities in {stratum}: {actual}; required {required}")
    selected.sort(key=lambda r: (mc.STRATA.index(r["stratum"]), r["sample_id"]))
    if len({r["sample_id"] for r in selected}) != mc.TOTAL_IDENTITIES or {r["sample_id"] for r in selected} & dev_ids:
        raise ValueError("training/development identity separation or total count failed")
    return selected


def plan_only() -> dict[str, Any]:
    preexisting = {p.name for p in mc.OUT.iterdir()} if mc.OUT.is_dir() else set()
    if (
        (preexisting - {"source_cohort_discrepancy.json"})
        or mc.STAGING.exists()
        or mc.FINAL.exists()
        or mc.REVIEW.exists()
    ):
        raise FileExistsError(f"refusing to overwrite existing multicorruption artifacts under {mc.OUT}")
    config = yaml.safe_load(mc.CONFIG.read_text())
    if config.get("schema") != mc.SCHEMA:
        raise ValueError("wrong versioned experiment config schema")
    dev_rows, dev_pins = _dev_reference()
    dev_ids = {r["sample_id"] for r in dev_rows}
    train_rows = _select_training(config, dev_ids)
    seed_rows, seed_summary = mc.build_seed_manifest(train_rows, global_seed=int(config["global_seed"]))
    secondary = []
    for row in dev_rows:
        seeds = [
            mc.derive_seed(int(config["global_seed"]), row["sample_id"], i, role="development_secondary")
            for i in (1, 2)
        ]
        if len(set(seeds)) != 2:
            raise ValueError(f"secondary development corruption seed collision: {row['sample_id']}")
        secondary.append(
            {
                "sample_id": row["sample_id"],
                "stratum": row["stratum"],
                "corruption_indices": [1, 2],
                "corruption_seeds": seeds,
            }
        )
    schedule = mc.build_schedule(seed_rows, schedule_seed=int(config["global_seed"]))
    schedule_digest = mc.schedule_sha256(schedule)
    if len(schedule) != 1092:
        raise ValueError("generated schedule does not contain exactly 1,092 updates")
    schedule_summary = []
    for update, examples, per_stratum in mc.BOUNDARIES:
        count = update * mc.EFFECTIVE_BATCH
        row = {
            "optimizer_updates": update,
            "training_examples": examples,
            "per_stratum_examples": {s: per_stratum for s in mc.STRATA},
            "fraction_per_stratum": {s: 0.2 for s in mc.STRATA},
        }
        if count != examples or per_stratum * len(mc.STRATA) != examples:
            raise ValueError("boundary totals fail exact stratum-balance arithmetic")
        schedule_summary.append(row)
    if len({(r["sample_id"], seed) for r in seed_rows for seed in r["corruption_seeds"]}) != mc.TOTAL_EXAMPLES:
        raise ValueError("duplicate training identity/corruption-seed pair")

    mc.OUT.mkdir(parents=True, exist_ok=True)
    seed_file = mc.OUT / "training_seed_manifest.json"
    mc.write_json(seed_file, {**seed_summary, "identities": seed_rows})
    mc.write_json(
        mc.OUT / "development_secondary_seed_manifest.json",
        {
            "schema": "e010_multicorruption_development_secondary_seed_manifest_v1",
            "global_seed": int(config["global_seed"]),
            "secondary_panels_per_identity": 2,
            "used_for_selection": False,
            "used_for_gate_adjudication": False,
            "identities": secondary,
        },
    )
    mc.write_json(
        mc.OUT / "development_baseline_reference.json",
        {
            "schema": "e010_multicorruption_fixed_development_reference_v1",
            "description": (
                "The exact existing Phase 4A 320-identity fixed panel and original "
                "fixed-corruption baseline; primary boundary evaluation uses those original "
                "baseline inputs."
            ),
            **dev_pins,
            "development_identity_count": len(dev_rows),
            "identities": dev_rows,
            "prospective_accessed": False,
        },
    )
    plan = {
        "schema": "e010_phase4a_multicorruption_plan_v1",
        "status": "prepared_non_authorizing",
        "identity_allocation_description": (
            "capped identity allocation with exactly length-balanced training-example exposure"
        ),
        "training_identity_count": len(train_rows),
        "training_identity_counts_by_stratum": dict(mc.TRAIN_COUNTS),
        "development_identity_count": len(dev_rows),
        "development_panel_unchanged": True,
        "development_identity_sha256": dev_pins["development_identity_sha256"],
        "total_unique_training_examples": mc.TOTAL_EXAMPLES,
        "training_examples_by_stratum": seed_summary["example_count_by_stratum"],
        "unique_identity_corruption_seed_pairs": seed_summary["unique_identity_seed_pair_count"],
        "schedule_sha256": schedule_digest,
        "schedule_storage": "algorithmically regenerated from compact seed manifest; no per-example tensor archives",
        "optimizer_updates": mc.UPDATES,
        "effective_batch_size": mc.EFFECTIVE_BATCH,
        "microbatch_per_stratum": mc.MICROBATCH,
        "boundary_schedule": schedule_summary,
        "fresh_initialization": True,
        "phase4a_weights_loaded": False,
        "primary_development_baseline_sha256": dev_pins["source_baseline_sha256"],
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
        "prospective_accessed": False,
        "training_started": False,
    }
    mc.write_json(mc.OUT / "plan.json", plan)
    runner_path = mc.ROOT / "scripts/run_e010_phase4a_multicorruption.py"
    prep = {
        "schema": "e010_phase4a_multicorruption_preparation_v1",
        "status": "prepared_non_authorizing",
        "config_sha256": mc.file_sha(mc.CONFIG),
        "preparation_runner_sha256": mc.file_sha(Path(__file__)),
        "lifecycle_runner_sha256": mc.file_sha(runner_path) if runner_path.exists() else None,
        "shared_primitives_sha256": mc.file_sha(Path(mc.__file__)),
        "model_source_sha256": mc.file_sha(
            mc.ROOT / "src/protein_distance_diffusion/models/e010_global_equivariant.py"
        ),
        "corruption_source_sha256": mc.file_sha(mc.ROOT / "scripts/run_e009_bayesian_refiner.py"),
        "seed_manifest_sha256": mc.file_sha(seed_file),
        "development_secondary_seed_manifest_sha256": mc.file_sha(mc.OUT / "development_secondary_seed_manifest.json"),
        "development_baseline_reference_sha256": mc.file_sha(mc.OUT / "development_baseline_reference.json"),
        "plan_sha256": mc.file_sha(mc.OUT / "plan.json"),
        "schedule_sha256": schedule_digest,
        **dev_pins,
        "training_identity_count": mc.TOTAL_IDENTITIES,
        "development_identity_count": 320,
        "training_example_count": mc.TOTAL_EXAMPLES,
        "optimizer_updates": mc.UPDATES,
        "phase4a_v2_v3_modified": False,
        "tensor_archives_written": False,
        "training_started": False,
        "cuda_initialized": False,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
        "prospective_accessed": False,
        "phase4b_prepared": False,
    }
    mc.write_json(mc.OUT / "preparation_manifest.json", prep)
    mc.STAGING.mkdir()
    mc.write_json(
        mc.STAGING / "lifecycle_contract.json",
        {
            "schema": "e010_multicorruption_staging_contract_v1",
            "preparation_manifest_sha256": mc.file_sha(mc.OUT / "preparation_manifest.json"),
            "training_started": False,
            "global_update": 0,
            "authorization": prep["authorization"],
        },
    )
    # Inventory is post-plan preparation evidence, not a scientific review.
    inventory = [
        {"path": p.relative_to(mc.OUT).as_posix(), "size_bytes": p.stat().st_size, "sha256": mc.file_sha(p)}
        for p in sorted(mc.OUT.rglob("*"))
        if p.is_file()
    ]
    mc.write_json(
        mc.OUT / "preparation_inventory.json",
        {
            "schema": "e010_multicorruption_preparation_inventory_v1",
            "files": inventory,
            "final_directory_created": False,
            "review_directory_created": False,
        },
    )
    return {
        "status": "prepared_non_authorizing",
        "training_identity_count": mc.TOTAL_IDENTITIES,
        "training_example_count": mc.TOTAL_EXAMPLES,
        "optimizer_updates": mc.UPDATES,
        "seed_manifest_sha256": prep["seed_manifest_sha256"],
        "config_sha256": prep["config_sha256"],
        "runner_sha256": prep["lifecycle_runner_sha256"],
        "staging_path": str(mc.STAGING.relative_to(mc.ROOT)),
        "final_path": str(mc.FINAL.relative_to(mc.ROOT)),
        "review_path": str(mc.REVIEW.relative_to(mc.ROOT)),
        "cuda_initialized": False,
        "training_started": False,
    }


def validate_contract() -> dict[str, Any]:
    discrepancy_path = mc.OUT / "source_cohort_discrepancy.json"
    preparation_path = mc.OUT / "preparation_manifest.json"
    if discrepancy_path.is_file() and not preparation_path.exists():
        discrepancy = mc.load_json(discrepancy_path)
        return {
            "status": "blocked_source_cohort_contract",
            "reason": discrepancy["reason"],
            "requested_training_identity_counts": discrepancy["requested_training_identity_counts"],
            "observed_candidate_pool_counts": discrepancy["observed_candidate_pool_counts"],
            "385-500_valid_source_count": discrepancy["385-500_valid_source_count"],
            "385-500_invalid_source_count": discrepancy["385-500_invalid_source_count"],
            "source_discrepancy_sha256": mc.file_sha(discrepancy_path),
            "training_started": False,
            "cuda_initialized": False,
            "authorization": {"phase4b": False, "prospective": False, "downstream": False},
        }
    config = yaml.safe_load(mc.CONFIG.read_text())
    prep = mc.load_json(mc.OUT / "preparation_manifest.json")
    plan = mc.load_json(mc.OUT / "plan.json")
    seed_obj = mc.load_json(mc.OUT / "training_seed_manifest.json")
    dev_seed = mc.load_json(mc.OUT / "development_secondary_seed_manifest.json")
    dev_reference = mc.load_json(mc.OUT / "development_baseline_reference.json")
    if prep.get("status") != "prepared_non_authorizing" or prep.get("config_sha256") != mc.file_sha(mc.CONFIG):
        raise ValueError("preparation/config pin mismatch")
    if prep.get("lifecycle_runner_sha256") != mc.file_sha(mc.ROOT / "scripts/run_e010_phase4a_multicorruption.py"):
        raise ValueError("lifecycle runner source hash differs from preparation pin")
    if prep.get("shared_primitives_sha256") != mc.file_sha(Path(mc.__file__)):
        raise ValueError("shared deterministic primitives source hash differs from preparation pin")
    if prep.get("seed_manifest_sha256") != mc.file_sha(mc.OUT / "training_seed_manifest.json"):
        raise ValueError("training seed manifest hash mismatch")
    if prep.get("development_secondary_seed_manifest_sha256") != mc.file_sha(
        mc.OUT / "development_secondary_seed_manifest.json"
    ):
        raise ValueError("development secondary seed manifest hash mismatch")
    if prep.get("development_baseline_reference_sha256") != mc.file_sha(mc.OUT / "development_baseline_reference.json"):
        raise ValueError("development baseline reference hash mismatch")
    if prep.get("plan_sha256") != mc.file_sha(mc.OUT / "plan.json"):
        raise ValueError("plan hash mismatch")
    mc.validate_seed_manifest(
        seed_obj["identities"],
        {k: v for k, v in seed_obj.items() if k != "identities"},
        global_seed=int(config["global_seed"]),
    )
    if (
        seed_obj.get("unique_identity_seed_pair_count") != mc.TOTAL_EXAMPLES
        or plan.get("total_unique_training_examples") != mc.TOTAL_EXAMPLES
    ):
        raise ValueError("unique identity/corruption-seed pair count mismatch")
    schedule = mc.build_schedule(seed_obj["identities"], schedule_seed=int(config["global_seed"]))
    if mc.schedule_sha256(schedule) != prep.get("schedule_sha256"):
        raise ValueError("deterministically regenerated training schedule hash mismatch")
    counts = {s: sum(r["multiplicity"] for r in seed_obj["identities"] if r["stratum"] == s) for s in mc.STRATA}
    if counts != {s: mc.EXAMPLES_PER_STRATUM for s in mc.STRATA}:
        raise ValueError(f"training-example exposure is not exactly balanced: {counts}")
    v2_plan = mc.load_json(mc.V2 / "phase4a_plan.json")
    expected_dev = sorted(v2_plan["development"], key=lambda r: (mc.STRATA.index(r["stratum"]), r["sample_id"]))
    if [r["sample_id"] for r in dev_reference["identities"]] != [r["sample_id"] for r in expected_dev]:
        raise ValueError("development identity panel changed")
    v2_baseline = mc.V2 / "development_corrupted_baseline.json"
    if prep.get("source_baseline_sha256") != mc.file_sha(v2_baseline):
        raise ValueError("original fixed development baseline changed")
    if len(dev_seed.get("identities", [])) != 320 or any(
        len(r["corruption_seeds"]) != 2 for r in dev_seed["identities"]
    ):
        raise ValueError("secondary development seed manifest is incomplete")
    if dev_seed.get("used_for_selection") is not False or dev_seed.get("used_for_gate_adjudication") is not False:
        raise ValueError("secondary development panels must remain descriptive only")
    if (
        prep.get("authorization") != {"phase4b": False, "prospective": False, "downstream": False}
        or prep.get("training_started") is not False
    ):
        raise ValueError("non-authorizing preparation contract failed")
    # Validate source files are still byte-pinned; this is read-only and does not build corruptions.
    for row in seed_obj["identities"]:
        observed = mc.file_sha(mc.ROOT / row["source_path"])
        if observed != row["source_sha256"]:
            raise ValueError(f"training source file hash changed: {row['sample_id']}")
    return {
        "status": "valid_read_only_contract",
        "training_identity_count": mc.TOTAL_IDENTITIES,
        "development_identity_count": 320,
        "training_examples": mc.TOTAL_EXAMPLES,
        "examples_per_stratum": counts,
        "unique_identity_corruption_seed_pairs": mc.TOTAL_EXAMPLES,
        "optimizer_updates": mc.UPDATES,
        "effective_batch_size": mc.EFFECTIVE_BATCH,
        "schedule_sha256": prep["schedule_sha256"],
        "seed_manifest_sha256": prep["seed_manifest_sha256"],
        "phase4a_weights_loaded": False,
        "training_started": False,
        "cuda_initialized": False,
        "authorization": prep["authorization"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-only", action="store_true")
    group.add_argument("--validate-contract", action="store_true")
    args = parser.parse_args()
    result = plan_only() if args.plan_only else validate_contract()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
