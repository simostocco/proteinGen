#!/usr/bin/env python3
"Prepare and validate E010 multi-corruption generalization inputs, without training or CUDA."

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts import e010_phase4a_multicorruption_v2 as mc


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


def _exclusions(pool: dict[str, Any]) -> tuple[list[dict[str, Any]], set[tuple[int, int]]]:
    if mc.file_sha(mc.V1_DISCREPANCY) != "2ca4c24c429df80d3db6ddd7aa30e5f2cbebec4e1c3dea798dd7d5caf65ade62":
        raise ValueError("blocked v1 source discrepancy changed; refusing to reinterpret its evidence")
    v1 = mc.load_json(mc.V1_DISCREPANCY)
    exclusions = []
    excluded_ids = set()
    excluded_paths = set()
    fingerprints: set[tuple[int, int]] = set()
    for row in v1["invalid_sources"]:
        coordinate_mismatch = row["observed_coordinate_shape"] != [row["expected_length"], 3]
        category = "coordinate_length_mismatch" if coordinate_mismatch else "payload_sample_id_mismatch"
        reason = "coordinate-length mismatch" if coordinate_mismatch else "payload sample-ID mismatch"
        source_path = str(row["source_path"])
        exclusions.append({**row, "exclusion_class": category, "exclusion_reason": reason})
        excluded_ids.add(str(row["sample_id"]))
        excluded_paths.add(source_path)
        stat = (mc.ROOT / source_path).stat()
        fingerprints.add((stat.st_dev, stat.st_ino))
    if sum(r["exclusion_class"] == "coordinate_length_mismatch" for r in exclusions) != 18:
        raise ValueError("v1 evidence no longer contains exactly 18 coordinate-length mismatches")
    if sum(r["exclusion_class"] == "payload_sample_id_mismatch" for r in exclusions) != 14:
        raise ValueError("v1 evidence no longer contains exactly 14 payload sample-ID mismatches")
    # Case-distinct candidate names can resolve to the same physical archive on the data volume.
    # Publish those aliases as exclusions too, without opening their contents.
    for stratum in mc.STRATA:
        for candidate in pool["train"][stratum]:
            sid = str(candidate["sample_id"])
            source_path = str(candidate["source_path"])
            path = mc.ROOT / source_path
            stat = path.stat()
            fingerprint = (stat.st_dev, stat.st_ino)
            if fingerprint not in fingerprints or sid in excluded_ids:
                continue
            exclusions.append(
                {
                    "sample_id": sid,
                    "source_path": source_path,
                    "same_physical_archive_as_excluded_path": True,
                    "exclusion_class": "excluded_archive_physical_alias",
                    "exclusion_reason": (
                        "source path resolves to the same physical archive as an explicitly excluded mismatched archive"
                    ),
                }
            )
            excluded_ids.add(sid)
            excluded_paths.add(source_path)
    return exclusions, fingerprints


def _split_membership() -> tuple[dict[str, str], str]:
    import pandas as pd

    split_path = mc.ROOT / "data/full/splits/split_assignments.parquet"
    frame = pd.read_parquet(split_path, columns=["sample_id", "split"])
    membership = {str(row.sample_id): str(row.split) for row in frame.itertuples(index=False)}
    if len(membership) != len(frame):
        raise ValueError("split membership metadata contains duplicate sample IDs")
    return membership, mc.file_sha(split_path)


def _select_training(
    config: dict[str, Any],
    dev_ids: set[str],
    pool: dict[str, Any],
    excluded_fingerprints: set[tuple[int, int]],
    membership: dict[str, str],
) -> list[dict[str, Any]]:
    selected = []
    for stratum in mc.STRATA:
        candidates = []
        for row in pool["train"][stratum]:
            sid = str(row["sample_id"])
            if sid in dev_ids or membership.get(sid) != "train":
                continue
            source = mc.ROOT / row["source_path"]
            stat = source.stat()
            if (stat.st_dev, stat.st_ino) in excluded_fingerprints:
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
    if any(membership.get(r["sample_id"]) != "train" for r in selected):
        raise ValueError("selected identity does not belong to the training split")
    return selected


def plan_only() -> dict[str, Any]:
    if mc.OUT.exists() or mc.STAGING.exists() or mc.FINAL.exists() or mc.REVIEW.exists():
        raise FileExistsError(f"refusing to overwrite existing multicorruption artifacts under {mc.OUT}")
    config = yaml.safe_load(mc.CONFIG.read_text())
    if config.get("schema") != mc.SCHEMA:
        raise ValueError("wrong versioned experiment config schema")
    dev_rows, dev_pins = _dev_reference()
    dev_ids = {r["sample_id"] for r in dev_rows}
    v2_plan = mc.load_json(mc.V2 / "phase4a_plan.json")
    pool_path = mc.ROOT / v2_plan["candidate_pool_path"]
    pool = mc.load_json(pool_path)
    exclusion_rows, excluded_fingerprints = _exclusions(pool)
    membership, split_membership_sha256 = _split_membership()
    non_train_ids = {sid for sid, split in membership.items() if split in {"validation", "test"}}
    train_rows = _select_training(config, dev_ids, pool, excluded_fingerprints, membership)
    if {r["sample_id"] for r in train_rows} & non_train_ids:
        raise ValueError("selected identity occurs in development or prospective split membership")
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
    mc.write_json(
        mc.OUT / "excluded_archives.json",
        {
            "schema": "e010_multicorruption_v2_excluded_archives_v1",
            "source_discrepancy_path": mc.V1_DISCREPANCY.relative_to(mc.ROOT).as_posix(),
            "source_discrepancy_sha256": mc.file_sha(mc.V1_DISCREPANCY),
            "excluded_count_by_reason": {
                "coordinate_length_mismatch": sum(
                    r["exclusion_class"] == "coordinate_length_mismatch" for r in exclusion_rows
                ),
                "payload_sample_id_mismatch": sum(
                    r["exclusion_class"] == "payload_sample_id_mismatch" for r in exclusion_rows
                ),
                "excluded_archive_physical_alias": sum(
                    r["exclusion_class"] == "excluded_archive_physical_alias" for r in exclusion_rows
                ),
            },
            "archives": exclusion_rows,
            "training_loader_must_skip_every_listed_physical_archive": True,
        },
    )
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
        "schema": "e010_phase4a_multicorruption_plan_v2",
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
        "excluded_archive_manifest_sha256": mc.file_sha(mc.OUT / "excluded_archives.json"),
        "selected_source_identity_length_checks": (
            "exact declared ID == payload ID; coordinate length == declared length"
        ),
        "selected_split_membership": {
            "train_count": len(train_rows),
            "development_overlap": 0,
            "prospective_overlap": 0,
            "split_membership_metadata_sha256": split_membership_sha256,
            "prospective_structure_data_accessed": False,
        },
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
    runner_path = mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py"
    prep = {
        "schema": "e010_phase4a_multicorruption_preparation_v2",
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
        "excluded_archive_manifest_sha256": mc.file_sha(mc.OUT / "excluded_archives.json"),
        "source_split_membership_sha256": split_membership_sha256,
        "development_secondary_seed_manifest_sha256": mc.file_sha(mc.OUT / "development_secondary_seed_manifest.json"),
        "development_baseline_reference_sha256": mc.file_sha(mc.OUT / "development_baseline_reference.json"),
        "plan_sha256": mc.file_sha(mc.OUT / "plan.json"),
        "schedule_sha256": schedule_digest,
        **dev_pins,
        "training_identity_count": mc.TOTAL_IDENTITIES,
        "training_identity_counts_by_stratum": dict(mc.TRAIN_COUNTS),
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
            "schema": "e010_multicorruption_v2_staging_contract_v1",
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
    config = yaml.safe_load(mc.CONFIG.read_text())
    prep = mc.load_json(mc.OUT / "preparation_manifest.json")
    plan = mc.load_json(mc.OUT / "plan.json")
    seed_obj = mc.load_json(mc.OUT / "training_seed_manifest.json")
    exclusions = mc.load_json(mc.OUT / "excluded_archives.json")
    dev_seed = mc.load_json(mc.OUT / "development_secondary_seed_manifest.json")
    dev_reference = mc.load_json(mc.OUT / "development_baseline_reference.json")
    if prep.get("status") != "prepared_non_authorizing" or prep.get("config_sha256") != mc.file_sha(mc.CONFIG):
        raise ValueError("preparation/config pin mismatch")
    runner_path = mc.ROOT / "scripts/run_e010_phase4a_multicorruption_v2.py"
    if prep.get("lifecycle_runner_sha256") != mc.file_sha(runner_path):
        raise ValueError("lifecycle runner source hash differs from preparation pin")
    if prep.get("preparation_runner_sha256") != mc.file_sha(Path(__file__)):
        raise ValueError("preparation runner source hash differs from preparation pin")
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
    exclusion_path = mc.OUT / "excluded_archives.json"
    if prep.get("excluded_archive_manifest_sha256") != mc.file_sha(exclusion_path):
        raise ValueError("excluded archive manifest hash mismatch")
    if exclusions.get("source_discrepancy_sha256") != mc.file_sha(mc.V1_DISCREPANCY):
        raise ValueError("preserved v1 discrepancy evidence hash mismatch")
    if (
        exclusions.get("source_discrepancy_sha256")
        != "2ca4c24c429df80d3db6ddd7aa30e5f2cbebec4e1c3dea798dd7d5caf65ade62"
    ):
        raise ValueError("v1 discrepancy evidence no longer matches its preserved hash")
    archive_exclusions = exclusions["archives"]
    if sum(r["exclusion_class"] == "coordinate_length_mismatch" for r in archive_exclusions) != 18:
        raise ValueError("v2 exclusion list must preserve 18 coordinate-length mismatches")
    if sum(r["exclusion_class"] == "payload_sample_id_mismatch" for r in archive_exclusions) != 14:
        raise ValueError("v2 exclusion list must preserve 14 payload sample-ID mismatches")
    excluded_fingerprints = set()
    excluded_ids = set()
    for row in archive_exclusions:
        path = mc.ROOT / row["source_path"]
        stat = path.stat()
        excluded_fingerprints.add((stat.st_dev, stat.st_ino))
        excluded_ids.add(str(row["sample_id"]))
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
    identity_counts = {s: sum(r["stratum"] == s for r in seed_obj["identities"]) for s in mc.STRATA}
    if identity_counts != mc.TRAIN_COUNTS:
        raise ValueError(f"real cohort identity counts mismatch: {identity_counts}")
    expected_extras = {"20-64": 1171, "65-128": 1171, "129-256": 1171, "257-384": 1006, "385-500": 900}
    for stratum, extra_count in expected_extras.items():
        expected_high = 13 if stratum == "385-500" else 6
        if (
            sum(r["stratum"] == stratum and r["multiplicity"] == expected_high for r in seed_obj["identities"])
            != extra_count
        ):
            raise ValueError(f"extra-corruption multiplicity count mismatch in {stratum}")
    boundary_results = []
    for update, examples, per_stratum in mc.BOUNDARIES:
        observed = {s: sum(len(schedule[i]["stratum_microbatches"][s]) for i in range(update)) for s in mc.STRATA}
        if observed != {s: per_stratum for s in mc.STRATA} or sum(observed.values()) != examples:
            raise ValueError(f"schedule boundary {update} is not exactly balanced: {observed}")
        boundary_results.append({"updates": update, "examples": examples, "examples_by_stratum": observed})
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
    membership, membership_hash = _split_membership()
    if membership_hash != prep.get("source_split_membership_sha256"):
        raise ValueError("train/development/prospective split membership evidence changed")
    selected_ids = {r["sample_id"] for r in seed_obj["identities"]}
    non_train = {sid for sid, split in membership.items() if split != "train"}
    if selected_ids & non_train or selected_ids & excluded_ids:
        raise ValueError("selected identity overlaps a non-training split or excluded identity")
    if selected_ids & {r["sample_id"] for r in dev_reference["identities"]}:
        raise ValueError("selected identity overlaps the fixed development panel")
    # Re-open only selected, non-excluded archives to verify exact payload identity and length.
    for row in seed_obj["identities"]:
        source_path = mc.ROOT / row["source_path"]
        stat = source_path.stat()
        if (stat.st_dev, stat.st_ino) in excluded_fingerprints:
            raise ValueError(f"selected source resolves to an explicitly excluded archive: {row['sample_id']}")
        observed = mc.file_sha(source_path)
        if observed != row["source_sha256"]:
            raise ValueError(f"training source file hash changed: {row['sample_id']}")
        mc.source_arrays(row)
    return {
        "status": "valid_read_only_contract",
        "training_identity_count": mc.TOTAL_IDENTITIES,
        "training_identity_counts_by_stratum": identity_counts,
        "development_identity_count": 320,
        "training_examples": mc.TOTAL_EXAMPLES,
        "examples_per_stratum": counts,
        "unique_identity_corruption_seed_pairs": mc.TOTAL_EXAMPLES,
        "optimizer_updates": mc.UPDATES,
        "effective_batch_size": mc.EFFECTIVE_BATCH,
        "boundary_checks": boundary_results,
        "excluded_archive_counts": exclusions["excluded_count_by_reason"],
        "selected_archives_exact_identity_and_length_verified": True,
        "selected_identities_in_validation_or_test_splits": 0,
        "excluded_archives_opened_during_training": 0,
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
