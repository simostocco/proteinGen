"""Read-only, torch-free Phase 3I.5 path and artifact contract checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

VERSION = "e007_phase3i5_sampler_unroll_v1"
FALSE_AUTHORIZATION = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_phase3j": False,
    "authorizes_production": False,
    "authorizes_additional_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_downstream_generation": False,
    "authorizes_sampler_correction": False,
    "authorizes_larger_sampler_aware_experiment": False,
}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            key: leaf
            for k, child in value.items()
            for key, leaf in _flatten(child, f"{prefix}.{k}" if prefix else k).items()
        }
    return {prefix: value}


def validate_contract(config_path: str | Path, *, validate_paths: bool = True) -> dict[str, Any]:
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text())
    version = cfg.get("version")
    if (
        version
        not in {
            VERSION,
            "e007_phase3i5_sampler_unroll_v2",
            "e007_phase3i5_sampler_unroll_v3_reviewed",
            "e007_phase3i5_sampler_unroll_v4_reviewed",
            "e007_phase3i5_sampler_unroll_v5_reviewed",
        }
        or cfg.get("updates") != 25
    ):
        raise ValueError("Phase 3I.5 is fixed at exactly 25 updates in this configuration version")
    if cfg.get("arms") != ["continued_v_only", "v_plus_sampler_unroll"]:
        raise ValueError("matched arm contract changed")
    if cfg.get("evaluation_boundaries") != [0, 10, 25] or cfg.get("sampling_boundaries") != [0, 25]:
        raise ValueError("evaluation boundary schedule changed")
    if cfg.get("unroll_transitions") != 2 or cfg.get("batch_size") != 1:
        raise ValueError("two-transition, batch-one contract changed")
    if version.endswith(("_v2", "_v3_reviewed", "_v4_reviewed", "_v5_reviewed")):
        if cfg.get("numerics", {}).get("activation_checkpointing") is not True:
            raise ValueError("v2 requires non-reentrant activation checkpointing")
        if cfg.get("numerics", {}).get("amp_enabled") is not True or cfg.get("numerics", {}).get("amp_preference") != [
            "bfloat16",
            "float16",
        ]:
            raise ValueError("v2 requires BF16-first mixed precision")
        if cfg.get("memory_limit_mib", {}).get("allocated") != 6144:
            raise ValueError("v2 allocated limit must remain 6144 MiB")
    source_keys = (
        "checkpoint",
        "source_config",
        "phase3i4_protocol",
        "phase3i4_v2_config",
        "development_panel",
        "phase3i4_calibration_report",
        "phase3i4_performance_report",
        "phase3i3_report",
    )
    for key in source_keys:
        source = Path(cfg[key])
        if sha256(source) != cfg[f"{key}_sha256"]:
            raise ValueError(f"pinned source hash mismatch: {key}")
    panel = json.loads(Path(cfg["development_panel"]).read_text())
    if sha256(panel["source_manifest"]) != panel["source_manifest_sha256"]:
        raise ValueError("development source manifest hash changed")
    if sha256(panel["calibration_exclusion"]["source"]) != cfg["phase3i4_calibration_panel_sha256"]:
        raise ValueError("Phase 3I.4 calibration panel hash changed")
    if panel["calibration_exclusion"].get("sha256") != cfg["phase3i4_calibration_panel_sha256"]:
        raise ValueError("development panel calibration exclusion hash mismatch")
    if sha256(panel["holdout_exclusion"]["identity_hash_record"]) != cfg["phase3i4_holdout_hash_record_sha256"]:
        raise ValueError("Phase 3I.4 holdout identity-hash record changed")
    if panel["holdout_exclusion"]["canonical_identity_sha256"] != cfg["phase3i4_holdout_identity_sha256"]:
        raise ValueError("Phase 3I.4 holdout identity fingerprint changed")
    calibration_panel = json.loads(Path(panel["calibration_exclusion"]["source"]).read_text())
    if sorted(row["identity"] for row in calibration_panel) != panel["calibration_exclusion"]["identities"]:
        raise ValueError("calibration exclusion set does not match pinned Phase 3I.4 calibration identities")
    rows = panel.get("records", [])
    if panel.get("version") != "e007_phase3i5_development_panel_v1" or len(rows) != 10:
        raise ValueError("development panel version or size mismatch")
    calibration_ids = set(panel["calibration_exclusion"]["identities"])
    panel_ids = [str(row["sample_id"]) for row in rows]
    if len(set(panel_ids)) != len(panel_ids) or set(panel_ids) & calibration_ids:
        raise ValueError("development identities overlap calibration or duplicate")
    target_lengths = {64, 128, 256, 384, 500}
    if {int(row["target_length"]) for row in rows} != target_lengths:
        raise ValueError("development panel does not cover all required lengths")
    if any(abs(int(row["observed_length"]) - int(row["target_length"])) > 4 for row in rows):
        raise ValueError("development identity is outside the declared target-length window")
    if any(sum(int(row["target_length"]) == target for row in rows) != 2 for target in target_lengths):
        raise ValueError("development panel requires two paired identities per target length")
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("development panel must be selected from validation split")
    try:
        import pyarrow.parquet as pq

        manifest = pq.read_table(
            panel["source_manifest"], columns=["sample_id", "split", "coordinate_accepted", "length"]
        )
        rows_by_id = {str(row["sample_id"]): row for row in manifest.to_pylist()}
        for row in rows:
            source_row = rows_by_id.get(str(row["sample_id"]))
            if (
                source_row is None
                or source_row["split"] != "validation"
                or source_row["coordinate_accepted"] is not True
                or int(source_row["length"]) != int(row["observed_length"])
            ):
                raise ValueError("development identity is not present in pinned clean validation manifest")
        expected = []
        for target in (64, 128, 256, 384, 500):
            candidates = [
                record
                for record in rows_by_id.values()
                if record["split"] == "validation"
                and record["coordinate_accepted"] is True
                and abs(int(record["length"]) - target) <= 4
            ]
            candidates.sort(
                key=lambda record: hashlib.sha256(f"e007-phase3i5-dev:{target}:{record['sample_id']}".encode()).digest()
            )
            expected.extend(str(record["sample_id"]) for record in candidates[:2])
        if panel_ids != expected:
            raise ValueError("development panel is not the deterministic hash-ranked selection")
    except ImportError as error:
        raise RuntimeError("pyarrow is required for read-only panel membership validation") from error
    if any(str(row["sample_id"]).startswith("holdout-n") for row in rows):
        raise ValueError("development panel entered the protected holdout namespace")
    if panel["training_exclusion"] != {
        "source_split": "train",
        "development_split": "validation",
        "disjoint_by_split": True,
    }:
        raise ValueError("training exclusion split contract changed")
    if panel["holdout_exclusion"].get("excluded_namespace") != "holdout-n":
        raise ValueError("holdout namespace exclusion missing")
    if panel["holdout_exclusion"].get("canonical_identity_sha256") != cfg["phase3i4_holdout_identity_sha256"]:
        raise ValueError("holdout identity-hash metadata changed")
    if float(cfg.get("geometry_coefficient", -1)) != 0.02:
        raise ValueError("protected local-geometry coefficient changed")
    if (
        cfg.get("decision", {}).get("meaningful_adjacent_rmse_improvement_angstrom") != 0.10
        or cfg.get("decision", {}).get("desirable_improvement_angstrom") != 0.20
    ):
        raise ValueError("predeclared decision thresholds changed")
    if cfg.get("coordinate_scale_angstrom") != 12.22820347644835:
        raise ValueError("coordinate scale changed from protected source contract")
    if cfg.get("maximum_updates") is not None:
        raise ValueError("automatic Phase 3I.5 extension is forbidden")
    authorization = cfg.get("authorization", {})
    if any(authorization.values()) or authorization != FALSE_AUTHORIZATION:
        raise ValueError("Phase 3I.5 authorization flags must all remain false")
    if version.endswith(("_v3_reviewed", "_v4_reviewed", "_v5_reviewed")):
        for key in ("source_v2_config", "reviewed_smoke_report", "reviewed_smoke_log", "reviewed_smoke_decision"):
            if sha256(cfg[key]) != cfg[f"{key}_sha256"]:
                raise ValueError(f"reviewed evidence hash mismatch: {key}")
        decision = json.loads(Path(cfg["reviewed_smoke_decision"]).read_text())
        if decision.get("status") != "reviewed_smoke_passed_bounded_experiment_authorized":
            raise ValueError("reviewed-smoke decision is not authorizing the bounded experiment")
        if (
            decision.get("authorization", {}).get("maximum_updates_per_arm") != 25
            or decision.get("authorization", {}).get("automatic_extension_to_50_updates") is not False
        ):
            raise ValueError("reviewed-smoke authorization exceeds the fixed 25-update experiment")
        for item in decision.get("protected_evidence", []):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"protected evidence hash mismatch: {item['path']}")
    if validate_paths:
        output, staging, smoke = (Path(cfg[k]) for k in ("output_dir", "staging_dir", "smoke_output_dir"))
        if len({output.resolve(), staging.resolve(), smoke.resolve()}) != 3:
            raise ValueError("scientific, staging, and smoke paths must be distinct")
        if any("phase3i4" in p.name.lower() for p in (output, staging, smoke)):
            raise ValueError("Phase 3I.5 paths cannot target protected Phase 3I.4 paths")
        if version.endswith(("_v2", "_v3_reviewed", "_v4_reviewed", "_v5_reviewed")) and any(
            "phase3i5" not in p.name.lower() or not any(tag in p.name.lower() for tag in ("_v2", "_v3", "_v4", "_v5"))
            for p in (output, staging, smoke)
        ):
            raise ValueError("v2 paths must be fresh and versioned")
    if version.endswith("_v4_reviewed"):
        source_v3 = Path(cfg["source_v3_config"])
        if sha256(source_v3) != cfg["source_v3_config_sha256"]:
            raise ValueError("pinned v3 source contract hash mismatch")
        source_cfg = yaml.safe_load(source_v3.read_text())
        allowed = {
            "version",
            "output_dir",
            "staging_dir",
            "smoke_output_dir",
            "memory_limit_mib.rss",
            "memory_policy.fail_closed_current_rss_mib",
            "memory_policy.rss_monotonic_growth_tolerance_mib",
            "memory_policy.rss_comparison_boundaries",
            "memory_policy.detect_unexpected_tensor_or_trajectory_accumulation",
            "source_v3_config",
            "source_v3_config_sha256",
            "incident_record",
            "incident_record_sha256",
        }

        old, new = _flatten(source_cfg), _flatten(cfg)
        changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
        if changed - allowed:
            raise ValueError(f"v4 changed protected scientific or execution fields: {sorted(changed - allowed)}")
        if cfg["memory_limit_mib"].get("allocated") != source_cfg["memory_limit_mib"].get("allocated") or cfg[
            "memory_limit_mib"
        ].get("reserved") != source_cfg["memory_limit_mib"].get("reserved"):
            raise ValueError("v4 CUDA memory limits must remain unchanged")
        if cfg["memory_limit_mib"].get("rss") != 6144:
            raise ValueError("v4 current RSS limit must be 6144 MiB")
        rss_policy = cfg.get("memory_policy", {})
        if (
            rss_policy.get("fail_closed_current_rss_mib") != 6144
            or rss_policy.get("rss_monotonic_growth_tolerance_mib") != 256
            or rss_policy.get("rss_comparison_boundaries") != [0, 10, 25]
            or rss_policy.get("detect_unexpected_tensor_or_trajectory_accumulation") is not True
        ):
            raise ValueError("v4 RSS growth and accumulation safeguards changed")
        incident_path = Path(cfg["incident_record"])
        if not incident_path.is_file() or sha256(incident_path) != cfg["incident_record_sha256"]:
            raise ValueError("non-authorizing v3 incident preservation record is missing")
        incident = json.loads(incident_path.read_text())
        if incident.get("status") != "preserved_non_authorizing" or any(
            sha256(incident_path.parent / item["path"]) != item["sha256"] for item in incident.get("files", [])
        ):
            raise ValueError("preserved v3 incident evidence failed hash validation")
    if version.endswith("_v5_reviewed"):
        source_v4 = Path(cfg["source_v4_config"])
        incident_path = Path(cfg["v4_incident_record"])
        if sha256(source_v4) != cfg["source_v4_config_sha256"]:
            raise ValueError("pinned v4 source contract hash mismatch")
        source_cfg = yaml.safe_load(source_v4.read_text())
        old, new = _flatten(source_cfg), _flatten(cfg)
        allowed = {
            "version",
            "output_dir",
            "staging_dir",
            "smoke_output_dir",
            "memory_policy.rss_monotonic_growth_tolerance_mib",
            "memory_policy.rss_comparison_boundaries",
            "memory_policy.rss_reference_boundary",
            "memory_policy.measurement_stage",
            "memory_policy.fail_for_resource_accumulation",
            "source_v4_config",
            "source_v4_config_sha256",
            "v4_incident_record",
            "v4_incident_record_sha256",
        }
        changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
        if changed - allowed:
            raise ValueError(f"v5 changed protected scientific or execution fields: {sorted(changed - allowed)}")
        if cfg["memory_limit_mib"] != source_cfg["memory_limit_mib"]:
            raise ValueError("v5 RSS correction must preserve all configured memory caps")
        policy = cfg["memory_policy"]
        if (
            policy.get("fail_closed_current_rss_mib") != 6144
            or policy.get("rss_monotonic_growth_tolerance_mib") != 256
            or policy.get("rss_comparison_boundaries") != [10, 25]
            or policy.get("rss_reference_boundary") != 10
            or policy.get("measurement_stage") != "post_cleanup_after_gc_and_phase_local_release"
            or policy.get("fail_for_resource_accumulation") is not True
        ):
            raise ValueError("v5 corrected post-warm-up RSS policy mismatch")
        if sha256(incident_path) != cfg["v4_incident_record_sha256"]:
            raise ValueError("preserved v4 incident record hash mismatch")
        incident = json.loads(incident_path.read_text())
        if incident.get("status") != "preserved_non_authorizing" or any(incident.get("authorization", {}).values()):
            raise ValueError("v4 incident record is not non-authorizing")
        for item in incident.get("files", []):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"preserved v4 evidence hash mismatch: {item['path']}")
    return {
        "configuration_sha256": sha256(path),
        "pinned_hashes": {k: cfg[f"{k}_sha256"] for k in (*source_keys,)},
        "development_panel_sha256": sha256(cfg["development_panel"]),
        "development_panel_size": len(rows),
        "authorization": dict(FALSE_AUTHORIZATION),
        "status": "read_only_contract_validated",
    }


def plan(config_path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    validation = validate_contract(config_path)
    training = {"continued_v_only": 25, "v_plus_sampler_unroll": 50}
    audit_by_arm = {"continued_v_only": 25, "v_plus_sampler_unroll": 50}
    evaluation = 60
    sampling = 20_000
    forward_total = sum(training.values()) + sum(audit_by_arm.values()) + evaluation + sampling
    root = Path(__file__).resolve().parents[3]
    execution_hashes = {
        "training_implementation": sha256(
            root / "src/protein_distance_diffusion/training/e007_phase3i5_sampler_unroll.py"
        ),
        "contract_implementation": sha256(root / "src/protein_distance_diffusion/training/e007_phase3i5_contract.py"),
        "execution_script": sha256(root / "scripts/run_e007_phase3i5_sampler_unroll.py"),
        "panel_builder": sha256(root / "scripts/build_e007_phase3i5_development_panel.py"),
    }
    smoke_projection = None
    smoke_report_path = Path(cfg["smoke_output_dir"]) / "report.json"
    if smoke_report_path.is_file():
        smoke_report = json.loads(smoke_report_path.read_text())
        if smoke_report.get("status") == "completed_non_authorizing":
            smoke_projection = smoke_report.get("post_smoke_empirical_runtime_projection_seconds")
    return {
        **validation,
        "status": "planned_non_authorizing",
        "output_dir": cfg["output_dir"],
        "staging_dir": cfg["staging_dir"],
        "smoke_output_dir": cfg["smoke_output_dir"],
        "updates_per_arm": 25,
        "workload_counts": {
            "training_forwards_by_arm": training,
            "audit_forwards_by_arm": audit_by_arm,
            "audit_forwards": sum(audit_by_arm.values()),
            "one_step_evaluation_forwards": evaluation,
            "production_sampling_forwards": sampling,
            "total_forwards": forward_total,
            "training_backward_calls_by_arm": {"continued_v_only": 75, "v_plus_sampler_unroll": 125},
            "component_gradient_audits_by_arm": {"continued_v_only": 50, "v_plus_sampler_unroll": 100},
            "total_objective_backward_by_arm": {"continued_v_only": 25, "v_plus_sampler_unroll": 25},
            "evaluation_backward_calls": 0,
            "sampling_backward_calls": 0,
            "total_backward_calls": 200,
            "sampler_autograd_transition_steps": 50,
        },
        "pre_smoke_conservative_runtime_bound_seconds": cfg["conservative_pre_smoke_runtime_bound_seconds"],
        "pre_smoke_runtime_bound_label": "conservative pre-smoke upper bound, not an empirical estimate",
        "post_smoke_empirical_runtime_projection_seconds": smoke_projection,
        "execution_evidence_sha256": execution_hashes,
        "smoke_command": (
            "CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python "
            f"scripts/run_e007_phase3i5_sampler_unroll.py --config {config_path} --cuda-memory-smoke"
        ),
        "cuda_initialized": False,
        "checkpoint_loaded": False,
        "output_created": False,
    }
