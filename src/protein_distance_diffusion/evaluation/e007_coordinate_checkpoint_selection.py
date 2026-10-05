"""Publication-only checkpoint selection for E007 Phase 3H.1."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
import yaml

from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file

VERSION = "e007_coordinate_checkpoint_selection_v1"
CLASSIFICATION = "selected_research_checkpoint_with_documented_tail_limitation"
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created": False,
    "cuda_used": False,
    "sampling_performed": False,
    "backward_performed": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "dataset_scanned": False,
    "dataset_modified": False,
    "checkpoint_modified": False,
    "authorizes_training": False,
    "authorizes_additional_training": False,
    "authorizes_production_training": False,
    "authorizes_real_data_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("version") != VERSION:
        raise ValueError("E007 Phase-3H.1 configuration version contradiction")
    if config.get("classification") != CLASSIFICATION:
        raise ValueError("E007 Phase-3H.1 classification changed")
    selected = config.get("selected_checkpoint", {})
    if int(selected.get("optimizer_update", -1)) != 9000:
        raise ValueError("E007 Phase-3H.1 selected checkpoint update changed")
    if selected.get("sha256") != "eb445f0b39067b8a00a47db27f966a6db97dc0e30bf81078b34ea54f111c7d82":
        raise ValueError("E007 Phase-3H.1 selected checkpoint hash changed")
    evidence = config.get("expected_evidence", {})
    if (int(evidence.get("total_samples", 0)), int(evidence.get("pass_count", 0))) != (160, 158):
        raise ValueError("E007 Phase-3H.1 sample-accounting contract changed")
    if float(evidence.get("original_adjacent_error_limit_angstrom", -1)) != 1.0:
        raise ValueError("E007 Phase-3H.1 original all-record gate changed")
    return config


def _verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"E007 Phase-3H.1 prerequisite is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3H.1 prerequisite hash contradiction: {label}")
    return observed


def _verify_inventory(source: Path, section: Mapping[str, Any]) -> dict[str, Any]:
    path = source / "artifact_inventory.json"
    _verify_file(path, section["artifact_inventory_sha256"], "phase3h_artifact_inventory")
    payload = json.loads(path.read_text())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != int(section["artifact_inventory_entry_count"]):
        raise ValueError("E007 Phase-3H.1 Phase-3H inventory count contradiction")
    if payload.get("aggregate_sha256") != section["artifact_inventory_aggregate_sha256"]:
        raise ValueError("E007 Phase-3H.1 Phase-3H inventory aggregate contradiction")
    if _canonical_sha(artifacts) != payload["aggregate_sha256"]:
        raise ValueError("E007 Phase-3H.1 Phase-3H inventory serialization contradiction")
    excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
    expected_paths = {
        item.relative_to(source).as_posix()
        for item in source.rglob("*")
        if item.is_file() and item.name not in excluded and ".tmp" not in item.name
    }
    observed_paths: set[str] = set()
    root = source.resolve()
    for artifact in artifacts:
        relative = Path(str(artifact["path"]))
        resolved = (source / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root):
            raise ValueError(f"E007 Phase-3H.1 inventory path escapes source: {relative}")
        logical = relative.as_posix()
        if logical in observed_paths:
            raise ValueError(f"E007 Phase-3H.1 inventory duplicate path: {logical}")
        observed_paths.add(logical)
        if not resolved.is_file() or resolved.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"E007 Phase-3H.1 inventory file contradiction: {logical}")
        if sha256_file(resolved) != artifact["sha256"]:
            raise ValueError(f"E007 Phase-3H.1 inventory hash contradiction: {logical}")
    if observed_paths != expected_paths:
        raise ValueError("E007 Phase-3H.1 Phase-3H inventory membership contradiction")
    return {"entry_count": len(artifacts), "aggregate_sha256": payload["aggregate_sha256"], "verified": True}


def verify_prerequisites(config: Mapping[str, Any], *, read_metrics: bool = True) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    phase3h = config["phase3h"]
    phase3h_source = Path(phase3h["source_dir"])
    phase3h_files = {
        "config": Path(phase3h["config_path"]),
        "report": phase3h_source / "report.json",
        "protocol": phase3h_source / "protocol.json",
        "sample_metrics": phase3h_source / "sample_metrics.parquet",
        "paired_comparisons": phase3h_source / "paired_comparisons.json",
        "checkpoint_pareto": phase3h_source / "checkpoint_pareto.json",
        "artifact_inventory": phase3h_source / "artifact_inventory.json",
    }
    for name, path in phase3h_files.items():
        hashes[f"phase3h_{name}"] = _verify_file(path, phase3h[f"{name}_sha256"], f"phase3h_{name}")
    inventory = _verify_inventory(phase3h_source, phase3h)

    for label in ("continuation", "phase3f", "phase3g"):
        section = config[label]
        source = Path(section["source_dir"])
        files = {
            "config": Path(section["config_path"]),
            "report": source / "report.json",
            "protocol": source / "protocol.json",
        }
        if label == "phase3g":
            files["artifact_inventory"] = source / "artifact_inventory.json"
        for name, path in files.items():
            hashes[f"{label}_{name}"] = _verify_file(path, section[f"{name}_sha256"], f"{label}_{name}")

    selected = config["selected_checkpoint"]
    hashes["selected_checkpoint"] = _verify_file(Path(selected["path"]), selected["sha256"], "selected_checkpoint")
    hashes["selected_checkpoint_metadata"] = _verify_file(
        Path(selected["metadata_path"]), selected["metadata_sha256"], "selected_checkpoint_metadata"
    )
    phase3h_report = json.loads(phase3h_files["report"].read_text())
    phase3h_protocol = json.loads(phase3h_files["protocol"].read_text())
    continuation_report = json.loads((Path(config["continuation"]["source_dir"]) / "report.json").read_text())
    pareto = json.loads(phase3h_files["checkpoint_pareto"].read_text())
    _verify_source_contracts(config, phase3h_report, phase3h_protocol, continuation_report, pareto)
    evidence = _verify_sample_evidence(config, phase3h_files["sample_metrics"]) if read_metrics else None
    return {
        "hashes": hashes,
        "phase3h_inventory": inventory,
        "evidence": evidence,
        "protected_inputs_verified": True,
    }


def _verify_source_contracts(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    continuation_report: Mapping[str, Any],
    pareto: Mapping[str, Any],
) -> None:
    if report.get("status") != "completed_non_authorizing" or protocol.get("status") != report["status"]:
        raise ValueError("E007 Phase-3H.1 Phase-3H completion status contradiction")
    authorization_fields = [key for key in NON_AUTHORIZING if key.startswith("authorizes_")]
    if any(bool(report.get(key)) or bool(protocol.get(key)) for key in authorization_fields):
        raise ValueError("E007 Phase-3H.1 source unexpectedly authorizes training")
    if protocol.get("configuration_sha256") != config["phase3h"]["config_sha256"]:
        raise ValueError("E007 Phase-3H.1 Phase-3H configuration identity contradiction")
    if protocol.get("report_sha256") != config["phase3h"]["report_sha256"]:
        raise ValueError("E007 Phase-3H.1 Phase-3H report identity contradiction")
    best = continuation_report.get("best_denoising", {})
    selected = config["selected_checkpoint"]
    if (
        int(best.get("optimizer_update", -1)) != 9000
        or best.get("checkpoint_sha256") != selected["sha256"]
        or best.get("criterion") != selected["criterion"]
        or float(best.get("validation_coordinate_v_mse", -1)) != float(selected["validation_coordinate_v_mse"])
    ):
        raise ValueError("E007 Phase-3H.1 minimum-validation checkpoint contradiction")
    if any("score" in key.lower() for key in _all_keys(pareto)):
        raise ValueError("E007 Phase-3H.1 Pareto evidence contains a scalar score")
    candidates = {int(row["checkpoint_update"]): row for row in pareto["candidates"]}
    objectives = list(pareto["objectives"])
    if not all(float(candidates[9000][field]) <= float(candidates[10000][field]) for field in objectives):
        raise ValueError("E007 Phase-3H.1 step 9000 does not dominate step 10000")
    if not any(float(candidates[9000][field]) < float(candidates[10000][field]) for field in objectives):
        raise ValueError("E007 Phase-3H.1 step 9000 strict dominance is absent")
    if pareto.get("dominated_by", {}).get("10000") != [9000]:
        raise ValueError("E007 Phase-3H.1 Pareto domination record contradiction")
    if 7500 not in pareto.get("nondominated_checkpoints", []):
        raise ValueError("E007 Phase-3H.1 step 7500 nondominance contradiction")
    if not (
        float(candidates[7500]["contact_density_reference_error_mean"])
        < float(candidates[9000]["contact_density_reference_error_mean"])
    ):
        raise ValueError("E007 Phase-3H.1 step 7500 contact-density advantage is absent")
    summaries = report["summaries"]
    if int(summaries["7500"]["adjacent_failure_count"]) != 42:
        raise ValueError("E007 Phase-3H.1 step 7500 failure count contradiction")
    if int(summaries["10000"]["adjacent_failure_count"]) != 21:
        raise ValueError("E007 Phase-3H.1 step 10000 failure count contradiction")
    if bool(report["decision"]["all_record_gate_pass"]["9000"]):
        raise ValueError("E007 Phase-3H.1 original all-record gate was unexpectedly waived")


def _all_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value] + [key for item in value.values() for key in _all_keys(item)]
    if isinstance(value, list):
        return [key for item in value for key in _all_keys(item)]
    return []


def _verify_sample_evidence(config: Mapping[str, Any], metrics_path: Path) -> dict[str, Any]:
    columns = [
        "checkpoint_update",
        "length",
        "sample_index",
        "seed",
        "adjacent_original_gate_pass",
        "adjacent_reference_error_angstrom",
        "radius_of_gyration_relative_reference_error",
        "contact_density_6a_reference_error",
        "contact_density_8a_reference_error",
        "contact_density_10a_reference_error",
    ]
    table = ds.dataset(metrics_path, format="parquet").to_table(
        columns=columns,
        filter=ds.field("checkpoint_update") == 9000,
    )
    rows = table.to_pylist()
    expected = config["expected_evidence"]
    if len(rows) != int(expected["total_samples"]):
        raise ValueError("E007 Phase-3H.1 selected-checkpoint sample count contradiction")
    failures = sorted(
        (row for row in rows if not bool(row["adjacent_original_gate_pass"])),
        key=lambda row: (int(row["length"]), int(row["sample_index"])),
    )
    if len(rows) - len(failures) != int(expected["pass_count"]) or len(failures) != int(expected["failure_count"]):
        raise ValueError("E007 Phase-3H.1 pass/failure accounting contradiction")
    pass_counts = Counter(int(row["length"]) for row in rows if bool(row["adjacent_original_gate_pass"]))
    expected_counts = {int(length): int(count) for length, count in expected["per_length_pass_counts"].items()}
    if dict(pass_counts) != expected_counts:
        raise ValueError("E007 Phase-3H.1 per-length pass-count contradiction")
    expected_failures = sorted(expected["failures"], key=lambda row: (int(row["length"]), int(row["sample_index"])))
    for observed, expected_row in zip(failures, expected_failures, strict=True):
        for field, value in expected_row.items():
            if isinstance(value, float):
                if abs(float(observed[field]) - value) > 1e-12:
                    raise ValueError(f"E007 Phase-3H.1 failure metric contradiction: {field}")
            elif int(observed[field]) != int(value):
                raise ValueError(f"E007 Phase-3H.1 failure identity contradiction: {field}")
    return {
        "sample_count": len(rows),
        "pass_count": len(rows) - len(failures),
        "failure_count": len(failures),
        "pass_fraction": (len(rows) - len(failures)) / len(rows),
        "per_length_pass_counts": {str(key): value for key, value in sorted(pass_counts.items())},
        "failures": [
            {key: row[key] for key in expected_row}
            for row, expected_row in zip(failures, expected_failures, strict=True)
        ],
        "original_all_record_gate_pass": False,
        "original_adjacent_error_limit_angstrom": 1.0,
    }


def plan_checkpoint_selection(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3H.1 output exists: {output} or {staging}")
    prerequisites = verify_prerequisites(config)
    return {
        "status": "planned_publication_only_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "classification": CLASSIFICATION,
        "selected_checkpoint": config["selected_checkpoint"],
        "evidence": prerequisites["evidence"],
        "protected_inputs_verified": True,
        "output_dir": str(output),
        "output_created": False,
        "scientific_review_required": True,
        "selected_for_research_downstream_evaluation": True,
        "every_scientific_gate_passed": False,
        **NON_AUTHORIZING,
    }


def publish_checkpoint_selection(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3H.1 output exists: {output} or {staging}")
    before = verify_prerequisites(config)
    staging.mkdir(parents=True)
    heartbeat = staging / "heartbeat.json"
    _atomic_json(heartbeat, {"status": "publishing", "updated_utc": _utc_now(), **NON_AUTHORIZING})
    try:
        selected = {
            "version": VERSION,
            "classification": CLASSIFICATION,
            "checkpoint_path": config["selected_checkpoint"]["path"],
            "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
            "optimizer_update": 9000,
            "selection_basis": ["minimum_validation_coordinate_v_mse", "strict_pareto_dominance_over_step_10000"],
            "selected_for_research_downstream_evaluation": True,
            "original_all_record_gate_pass": False,
            "scientific_review_required": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "selected_checkpoint.json", selected)
        inventory_rows = [
            {
                "path": "selected_checkpoint.json",
                "size_bytes": (staging / "selected_checkpoint.json").stat().st_size,
                "sha256": sha256_file(staging / "selected_checkpoint.json"),
            }
        ]
        inventory = {"artifacts": inventory_rows, "aggregate_sha256": _canonical_sha(inventory_rows)}
        _atomic_json(staging / "artifact_inventory.json", inventory)
        report = {
            "status": "completed_publication_only_non_authorizing",
            "version": VERSION,
            "classification": CLASSIFICATION,
            "selected_checkpoint": selected,
            "selection_evidence": before["evidence"],
            "step_7500": {"pareto_nondominated": True, "adjacent_failure_count": 42, "retained": False},
            "step_10000": {"dominated_by_step_9000": True, "adjacent_failure_count": 21, "retained": False},
            "interpretation": {
                "selected_for_research_downstream_evaluation": True,
                "claim_every_scientific_gate_passed": False,
                "original_1a_all_record_gate_waived_or_redefined": False,
                "scientific_review_required": True,
                "additional_training_authorized": False,
            },
            "scalar_score_used": False,
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "classification": CLASSIFICATION,
            "report_sha256": sha256_file(staging / "report.json"),
            "selected_checkpoint_record_sha256": sha256_file(staging / "selected_checkpoint.json"),
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            "source_hashes": before["hashes"],
            "completed_utc": _utc_now(),
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        if (staging / "report.json").read_bytes() == (staging / "protocol.json").read_bytes():
            raise ValueError("E007 Phase-3H.1 report/protocol publication separation failure")
        after = verify_prerequisites(config)
        if after != before:
            raise ValueError("E007 Phase-3H.1 protected inputs changed during publication")
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "report_sha256": protocol["report_sha256"],
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return {
            "status": report["status"],
            "output_dir": str(output),
            "classification": CLASSIFICATION,
            "selected_checkpoint_sha256": selected["checkpoint_sha256"],
            **NON_AUTHORIZING,
        }
    except BaseException as error:
        _atomic_json(
            heartbeat,
            {
                "status": "failed",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error)[:2000],
                **NON_AUTHORIZING,
            },
        )
        raise
