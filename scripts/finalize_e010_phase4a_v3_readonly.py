#!/usr/bin/env python3
"""Read-only CPU validation and publication of the E010 Phase 4A v3 review.

This module deliberately does not import the training package. Checkpoints are
loaded with map_location='cpu' solely to validate state, cursor, exposure and
hash evidence. No model is constructed and no inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v3"
STAGING = SOURCE / "phase4a_training_v3.staging"
DEST = SOURCE / "phase4a_v3_readonly_scientific_review_v1"
BOUNDARIES = (50, 51, 52, 53, 54)
AUTH_FALSE = {"downstream": False, "phase4b": False, "prospective": False}
HISTORICAL_EXECUTION_RUNNER_SHA256 = "b0a87848cea1e4128e8d2828bb3cccfc14c3d0b74dbe5b34eb31c42a35e732a2"
REPORTING_REPAIR_RUNNER_SHA256 = "e5121a6cca8fd487f00d7bb41aedb9f360cd20aa676a5938cf513cda41ff8740"


def preparation_contract_checks(config: dict[str, Any], prep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Check distinct historical execution facts and forward authorizations.

    Preparation is a non-authorizing record written before extension execution;
    the completed journal is the evidence that execution later occurred.
    """
    config_auth = config.get("authorization")
    prep_auth = prep.get("authorization")
    return {
        "config_downstream_authorization": {"observed": (config_auth or {}).get("downstream"), "expected": False},
        "config_phase4b_authorization": {"observed": (config_auth or {}).get("phase4b"), "expected": False},
        "config_prospective_authorization": {"observed": (config_auth or {}).get("prospective"), "expected": False},
        "preparation_downstream_authorization": {"observed": (prep_auth or {}).get("downstream"), "expected": False},
        "preparation_phase4b_authorization": {"observed": (prep_auth or {}).get("phase4b"), "expected": False},
        "preparation_prospective_authorization": {"observed": (prep_auth or {}).get("prospective"), "expected": False},
        "preparation_training_started_at_preparation": {"observed": prep.get("training_started"), "expected": False},
        "preparation_phase4b_prepared": {"observed": prep.get("phase4b_prepared"), "expected": False},
        "preparation_prospective_accessed": {"observed": prep.get("prospective_accessed"), "expected": False},
        "historical_bounded_extension_execution_authorization": {
            "observed": prep_auth.get("bounded_extension_execution") if isinstance(prep_auth, dict) else None,
            "expected": None,
            "interpretation": (
                "This schema does not encode historical execution authorization; execution is evidenced by the journal."
            ),
        },
    }


def preparation_pin_checks(root: Path, prep: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute every source pin recorded by the preparation manifest."""
    v2 = root.parent / "phase4a_supervised_generalization_v2"
    pins = prep.get("pins", {})
    paths = {
        "v2_config_sha256": ROOT / "configs/e010_phase4a_supervised_generalization_v2.yaml",
        "v2_preparation_manifest_sha256": v2 / "preparation_manifest.json",
        "v2_plan_sha256": v2 / "phase4a_plan.json",
        "v2_input_validation_sha256": v2 / "input_validation.json",
        "v2_corruption_cache_manifest_sha256": v2 / "corruption_cache_manifest.json",
        "v2_cache_validation_sha256": v2 / "cache_validation.json",
        "v2_development_baseline_sha256": v2 / "development_corrupted_baseline.json",
        "v2_training_protocol_sha256": v2 / "training_protocol.json",
        "v2_selected_checkpoint_sha256": v2 / "phase4a_training_v2.final/selected_checkpoint.pt",
        "v2_exposure50_checkpoint_sha256": v2 / "phase4a_training_v2.final/checkpoint_at_exposure_50.pt",
        "v2_training_metrics_sha256": v2 / "phase4a_training_v2.final/training_metrics.json",
        "v2_review_json_sha256": v2 / "phase4a_v2_scientific_review_v1/review.json",
        "v2_review_inventory_sha256": v2 / "phase4a_v2_scientific_review_v1/artifact_inventory.json",
        "exact_continuation_validation_sha256": root / "exact_continuation_validation.json",
        "config_sha256": root / "phase4a_v3_extension_config.json",
        "schedule_sha256": root / "extension_schedule.json",
    }
    checks = {}
    for name, path in paths.items():
        observed = sha(path) if path.is_file() else None
        checks[name] = {"path": str(path), "observed": observed, "expected": prep.get(name, pins.get(name))}
    selected_path = v2 / "phase4a_training_v2.final/selected_checkpoint.pt"
    selected_state = (
        load_checkpoint(selected_path)
        if selected_path.is_file() and pins.get("selected_state_component_sha256")
        else {}
    )
    for name, value in pins.get("selected_state_component_sha256", {}).items():
        checks[f"selected_state_component_sha256.{name}"] = {
            "observed": preparation_canonical(selected_state[name]) if name in selected_state else None,
            "expected": value,
            "source": "v2 selected checkpoint canonical component",
        }
    return checks


def runner_lineage_checks(root: Path, prep: dict[str, Any]) -> dict[str, Any]:
    """Validate historical and repaired runner identities as separate artifacts."""

    incident = _read_json(root / "non_authorizing_incident_audit.json")
    inv_path = root / "recursive_inventory_precode_edit.json"
    pre_inventory = _read_json(inv_path)
    snapshot = Path(incident.get("snapshot", {}).get("path", ""))
    snapshot_prep = snapshot / "preparation_manifest.json"
    snapshot_config = snapshot / "phase4a_v3_extension_config.json"
    snapshot_journal = snapshot / "phase4a_training_v3.staging/journal.jsonl"
    expected_execution = HISTORICAL_EXECUTION_RUNNER_SHA256
    expected_repair = REPORTING_REPAIR_RUNNER_SHA256
    current_runner = ROOT / "scripts/run_e010_phase4a_v3_extension.py"
    report_text = (root / "non_authorizing_incident_report.md").read_text()
    checks = {
        "execution_runner_sha256": {
            "expected": expected_execution,
            "preparation_manifest": prep.get("execution_runner_sha256"),
            "pre_edit_snapshot_preparation_manifest": (
                _read_json(snapshot_prep).get("execution_runner_sha256") if snapshot_prep.is_file() else None
            ),
            "incident_audit": incident.get("runner_hashes", {}).get("old_sha256_from_preparation_pin"),
            "current_path_observed_sha256": sha(current_runner) if current_runner.is_file() else None,
            "current_path_is_historical_source": False,
        },
        "reporting_repair_runner_sha256": {
            "expected": expected_repair,
            "current_path_observed_sha256": sha(current_runner) if current_runner.is_file() else None,
            "incident_audit": incident.get("runner_hashes", {}).get("corrected_sha256"),
            "incident_report": expected_repair in report_text,
        },
        "pre_edit_snapshot": {
            "inventory_schema": pre_inventory.get("schema"),
            "captured_before_code_edits": pre_inventory.get("captured_before_code_edits"),
            "snapshot_path": str(snapshot),
            "original_runner_copy_found": any(
                p.is_file() and p.name in {"run_e010_phase4a_v3_extension.py", "run_e010_phase4a_v3_extension.py.old"}
                for p in snapshot.rglob("*")
            )
            if snapshot.is_dir()
            else False,
            "original_runner_copy_limitation": (
                "Original runner bytes are absent from the preserved incident snapshot; "
                "historical identity is therefore cross-checked against the preparation "
                "manifest and compatibility audit."
            ),
            "snapshot_manifest_sha256": sha(snapshot_prep) if snapshot_prep.is_file() else None,
            "snapshot_config_sha256": sha(snapshot_config) if snapshot_config.is_file() else None,
            "snapshot_journal_sha256": sha(snapshot_journal) if snapshot_journal.is_file() else None,
            "pre_edit_inventory_matches_snapshot_files": True,
        },
        "execution_log_lineage": {
            "audit_execution_journal_sha256": incident.get("execution", {}).get("journal_sha256"),
            "current_journal_sha256": sha(root / "phase4a_training_v3.staging/journal.jsonl"),
            "snapshot_journal_sha256": sha(snapshot_journal) if snapshot_journal.is_file() else None,
        },
        "execution_config_lineage": {
            "current_config_sha256": sha(root / "phase4a_v3_extension_config.json"),
            "snapshot_config_sha256": sha(snapshot_config) if snapshot_config.is_file() else None,
            "config_pin_in_preparation_manifest": prep.get("config_sha256"),
            "historical_execution_runner_pin_in_preparation_manifest": prep.get("execution_runner_sha256"),
        },
        "compatibility_source_diff": {
            "exact_v3_source_diff_preserved": False,
            "incident_record_claims_scientific_loop_evaluation_gates_selection_unchanged": all(
                phrase in report_text
                for phrase in ("update loop", "evaluation call paths", "gate calculations", "selection metric")
            ),
            "independent_exact_diff_verification": False,
            "limitation": (
                "The repaired runner bytes are present and hash correctly, but the original "
                "runner and an exact v3 source diff were not preserved; unchanged-function "
                "claims remain incident-record assertions."
            ),
        },
    }
    # Recompute every file captured by the pre-edit inventory against both the
    # preserved snapshot and its recorded size/hash. Ignore later-added files.
    inventory_ok = bool(snapshot.is_dir() and pre_inventory.get("captured_before_code_edits") is True)
    for row in pre_inventory.get("files", []):
        snap_file = snapshot / row["path"]
        if (
            not snap_file.is_file()
            or snap_file.stat().st_size != row.get("size_bytes")
            or sha(snap_file) != row.get("sha256")
        ):
            inventory_ok = False
    checks["pre_edit_snapshot"]["pre_edit_inventory_matches_snapshot_files"] = inventory_ok
    # Exact runner hash attestations must agree in every preserved source that
    # carries the historical pin. The current execution config is separately
    # pinned and is not expected to equal the reporting repair runner.
    return checks


def validate_runner_lineage(checks: dict[str, Any]) -> None:
    historical = checks["execution_runner_sha256"]
    repair = checks["reporting_repair_runner_sha256"]
    exec_ok = all(
        historical[k] == HISTORICAL_EXECUTION_RUNNER_SHA256
        for k in ("expected", "preparation_manifest", "pre_edit_snapshot_preparation_manifest", "incident_audit")
    )
    repair_ok = all(
        (
            repair["expected"] == REPORTING_REPAIR_RUNNER_SHA256,
            repair["current_path_observed_sha256"] == REPORTING_REPAIR_RUNNER_SHA256,
            repair["incident_audit"] == REPORTING_REPAIR_RUNNER_SHA256,
            repair["incident_report"] is True,
        )
    )
    snapshot = checks["pre_edit_snapshot"]
    snapshot_ok = snapshot["pre_edit_inventory_matches_snapshot_files"] is True
    execution_log = checks["execution_log_lineage"]
    log_ok = (
        execution_log["audit_execution_journal_sha256"]
        == execution_log["current_journal_sha256"]
        == execution_log["snapshot_journal_sha256"]
    )
    config = checks["execution_config_lineage"]
    config_ok = (
        config["current_config_sha256"]
        == config["snapshot_config_sha256"]
        == config["config_pin_in_preparation_manifest"]
    )
    checks["execution_runner_sha256"]["passed"] = exec_ok
    repair["passed"] = repair_ok
    snapshot["passed"] = snapshot_ok
    execution_log["passed"] = log_ok
    config["passed"] = config_ok
    if not all((exec_ok, repair_ok, snapshot_ok, log_ok, config_ok)):
        raise ValueError("runner lineage evidence mismatch: " + json.dumps(checks, sort_keys=True))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    """Stable hash for nested checkpoint state, keeping tensor bytes on CPU."""
    import torch

    def norm(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            return {
                "tensor": str(t.dtype),
                "shape": list(t.shape),
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            return {"array": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return {"repr": repr(x), "type": type(x).__qualname__}

    return hashlib.sha256(
        json.dumps(norm(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def preparation_canonical(value: Any) -> str:
    """Reproduce the preparation writer's original canonical state hashing."""
    import torch

    def norm(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            t = x.detach().contiguous().cpu()
            return {
                "tensor": str(t.dtype),
                "shape": list(t.shape),
                "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest(),
            }
        if isinstance(x, dict):
            return {str(k): norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return {"type": type(x).__qualname__, "repr": repr(x)}

    payload = json.dumps(norm(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def finite_tree(x: Any) -> bool:
    import torch

    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all())
    if isinstance(x, np.ndarray):
        return bool(np.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def bootstrap(
    values: list[float], reps: int, seed: int, quantity: str = "paired_percentage_improvement"
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for start in range(0, reps, 256):
        n = min(256, reps - start)
        means[start : start + n] = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_percentile": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "replicates": reps,
        "unit": "development_identity",
        "quantity": quantity,
    }


def worsening_stop(values: list[float]) -> tuple[list[bool], str]:
    flags = [float(values[i]) > float(values[i - 1]) for i in range(1, len(values))]
    reason = (
        "development_worsened_at_two_consecutive_boundaries"
        if len(flags) >= 2 and flags[-1] and flags[-2]
        else "available_evidence_ends_without_two_consecutive_worsening_boundaries"
    )
    return flags, reason


def select_checkpoint(boundaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not boundaries:
        raise ValueError("no evaluated development boundaries")
    return min(boundaries, key=lambda x: (float(x["development_mean_rmse"]), int(x["exposure"])))


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint is not a state dictionary: {path}")
    return state


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required evidence: {path}")
    return json.loads(path.read_text())


def configured_strata_names(config: dict[str, Any]) -> list[str]:
    """Resolve strata from the v3 config when present, otherwise its pinned base config."""
    selection = config.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("strata"), list):
        return [str(s["name"]) for s in selection["strata"]]
    base_path = ROOT / str(config.get("base_config_path", ""))
    if not base_path.is_file():
        raise FileNotFoundError(f"missing pinned base config for stratum schema: {base_path}")
    import yaml

    base = yaml.safe_load(base_path.read_text())
    strata = base.get("selection", {}).get("strata", [])
    if not strata:
        raise ValueError("base config does not define selection strata")
    return [str(s["name"]) for s in strata]


def validate(root: Path = SOURCE) -> dict[str, Any]:
    """Validate source evidence and recompute trajectory/adjudication; no writes."""
    stage = root / "phase4a_training_v3.staging"
    config = _read_json(root / "phase4a_v3_extension_config.json")
    prep = _read_json(root / "preparation_manifest.json")
    auth_checks = preparation_contract_checks(config, prep)
    failed_auth = {k: v for k, v in auth_checks.items() if v["observed"] != v["expected"]}
    print(json.dumps({"authorization_checks": auth_checks}, sort_keys=True), flush=True)
    if failed_auth:
        raise ValueError("preparation authorization contract mismatch: " + json.dumps(failed_auth, sort_keys=True))
    pin_checks = preparation_pin_checks(root, prep)
    print(json.dumps({"preparation_evidence_pin_checks": pin_checks}, sort_keys=True), flush=True)
    failed_pins = {k: v for k, v in pin_checks.items() if v["observed"] != v["expected"]}
    if failed_pins:
        raise ValueError("preparation evidence pin mismatch: " + json.dumps(failed_pins, sort_keys=True))
    lineage_checks = runner_lineage_checks(root, prep)
    validate_runner_lineage(lineage_checks)
    print(json.dumps({"runner_lineage_checks": lineage_checks}, sort_keys=True), flush=True)

    journal_path = stage / "journal.jsonl"
    if not journal_path.is_file():
        raise FileNotFoundError(f"missing required evidence: {journal_path}")
    raw = journal_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("journal is truncated")
    journal = []
    for i, line in enumerate(raw.splitlines(), 1):
        row = json.loads(line)
        digest = row.pop("record_sha256", None)
        expected = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if digest != expected:
            raise ValueError(f"journal record hash mismatch at row {i}")
        row["record_sha256"] = digest
        journal.append(row)
    execution_observed = all(r.get("training_started") is True for r in journal)
    print(
        json.dumps(
            {
                "authorization_and_execution_checks": {
                    "bounded_v3_extension_execution_observed_in_journal": {
                        "observed": execution_observed,
                        "expected": True,
                    },
                    "phase4b_authorization": {
                        "observed": (prep.get("authorization") or {}).get("phase4b"),
                        "expected": False,
                    },
                    "prospective_access": {"observed": prep.get("prospective_accessed"), "expected": False},
                    "downstream_authorization": {
                        "observed": (prep.get("authorization") or {}).get("downstream"),
                        "expected": False,
                    },
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if len(journal) != 92 or [r.get("extension_update") for r in journal] != list(range(1, 93)):
        raise ValueError("journal is not the exact contiguous 92-update prefix")
    if [r.get("global_update") for r in journal] != list(range(1151, 1243)):
        raise ValueError("journal global update prefix mismatch")
    if any(r.get("training_started") is not True or r.get("prospective_split_accessed") is not False for r in journal):
        raise ValueError("journal authorization evidence mismatch")
    if any(r.get("exposure") != 51 + i // 23 or r.get("step_in_exposure") != i % 23 + 1 for i, r in enumerate(journal)):
        raise ValueError("journal does not form the exact 23-update exposure 51 through 54 suffix")

    # Verify every scheduled identity exposure from the independently pinned full-state checkpoints.
    boundary_data, boundary_states = {}, {}
    dev_ids = None
    for exposure in BOUNDARIES:
        evpath = stage / f"development_exposure_{exposure}.json"
        cppath = stage / f"checkpoint_at_exposure_{exposure}.pt"
        ev = _read_json(evpath)
        if not cppath.is_file():
            raise FileNotFoundError(f"missing required evidence: {cppath}")
        if (
            ev.get("exposure") != exposure
            or ev.get("evaluation_complete") is not True
            or ev.get("development_count") != 320
        ):
            raise ValueError(f"incomplete boundary evaluation at exposure {exposure}")
        rows = ev.get("development_per_identity", [])
        ids = [x.get("sample_id") for x in rows]
        if len(ids) != 320 or len(set(ids)) != 320 or (dev_ids is not None and set(ids) != dev_ids):
            raise ValueError(f"development identity panel mismatch at exposure {exposure}")
        dev_ids = set(ids)
        # Exposure 50 evaluation was produced against the v2 boundary file.
        # Preparation copied the selected file into v3 staging; its bytes differ
        # although the complete checkpoint state is identical.
        eval_checkpoint_path = (
            root.parent / "phase4a_supervised_generalization_v2/phase4a_training_v2.final/checkpoint_at_exposure_50.pt"
            if exposure == 50
            else cppath
        )
        eval_checkpoint_expected = sha(eval_checkpoint_path)
        checkpoint_hash_check = {
            "exposure": exposure,
            "evaluation_recorded_sha256": ev.get("checkpoint_sha256"),
            "evaluation_source_checkpoint_path": str(eval_checkpoint_path),
            "evaluation_source_checkpoint_sha256": eval_checkpoint_expected,
            "staged_checkpoint_path": str(cppath),
            "staged_checkpoint_sha256": sha(cppath),
            "separate_serialization_same_state_required": exposure == 50,
        }
        print(json.dumps({"raw_evaluation_checkpoint_hash_check": checkpoint_hash_check}, sort_keys=True), flush=True)
        if ev.get("checkpoint_sha256") != eval_checkpoint_expected:
            raise ValueError(
                f"boundary evaluation source checkpoint hash mismatch at exposure {exposure}: "
                + json.dumps(checkpoint_hash_check, sort_keys=True)
            )
        if not finite_tree(ev):
            raise ValueError(f"non-finite boundary evaluation at exposure {exposure}")
        state = load_checkpoint(cppath)
        if not finite_tree(state):
            raise ValueError(f"non-finite checkpoint state at exposure {exposure}")
        if exposure == 50:
            source_state = load_checkpoint(eval_checkpoint_path)
            state_match = canonical(state) == canonical(source_state)
            print(
                json.dumps(
                    {
                        "exposure50_staged_checkpoint_full_state_matches_evaluation_source": {
                            "observed": state_match,
                            "expected": True,
                        }
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if not state_match:
                raise ValueError("exposure-50 staged checkpoint state differs from evaluation source checkpoint")
        if state.get("schedule_cursor") != exposure * 23 or state.get("global_update") != exposure * 23:
            raise ValueError(f"checkpoint cursor mismatch at exposure {exposure}")
        exposure_counts = state.get("identity_exposures", {})
        if len(exposure_counts) != 2048 or set(exposure_counts.values()) != {exposure}:
            raise ValueError(f"not exactly {exposure} exposures for all 2048 training identities")
        if exposure == 54 and (
            not {
                "model",
                "optimizer",
                "scheduler",
                "scaler",
                "python_rng_state",
                "numpy_rng_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
                "sampler_rng_state",
            }.issubset(state)
        ):
            raise ValueError("latest full-state checkpoint lacks required optimizer/RNG components")
        boundary_data[exposure] = ev
        boundary_states[exposure] = state

    latest_path = stage / "latest.pt"
    if not latest_path.is_file():
        raise FileNotFoundError(f"missing required evidence: {latest_path}")
    latest = load_checkpoint(latest_path)
    if sha(latest_path) != sha(stage / "checkpoint_at_exposure_54.pt"):
        raise ValueError("latest checkpoint does not equal exposure-54 checkpoint")
    if canonical(latest) != canonical(boundary_states[54]):
        raise ValueError("latest full state differs from exposure-54 boundary state")
    if journal[-1].get("checkpoint_sha256") != sha(latest_path) or journal[-1].get("global_update") != 1242:
        raise ValueError("latest checkpoint does not match final journal row")

    # Each raw boundary file carries paired baseline/refined rows. Recompute all reported statistics.
    reviewed = []
    strata_names = configured_strata_names(config)
    for exposure in BOUNDARIES:
        ev = boundary_data[exposure]
        pairs = ev.get("metrics", {}).get("paired_per_identity", [])
        if len(pairs) != 320 or {p.get("sample_id") for p in pairs} != dev_ids:
            raise ValueError(f"missing raw paired evaluation evidence at exposure {exposure}")
        b = np.asarray([p["baseline_rmse_angstrom"] for p in pairs], dtype=float)
        a = np.asarray([p["refined_rmse_angstrom"] for p in pairs], dtype=float)
        if not np.isfinite(b).all() or not np.isfinite(a).all() or (b <= 0).any():
            raise ValueError("invalid raw paired RMSE values")
        imp = (b - a) / b
        strata = {}
        for name in strata_names:
            subset = [p for p in pairs if p["stratum"] == name]
            bb = np.asarray([p["baseline_rmse_angstrom"] for p in subset])
            aa = np.asarray([p["refined_rmse_angstrom"] for p in subset])
            strata[name] = {
                "count": len(subset),
                "baseline_rmse": float(bb.mean()),
                "refined_rmse": float(aa.mean()),
                "paired_percentage_improvement": float(((bb - aa) / bb).mean()),
                "paired_rmse_improvement_fraction_of_mean": float((bb.mean() - aa.mean()) / bb.mean()),
            }
        lkeys = [f"i_plus_{i}_distance_rmse_angstrom" for i in (1, 2, 3)]
        base_local = np.asarray([[p["baseline_geometry"][k] for k in lkeys] for p in pairs], dtype=float)
        ref_local = np.asarray([[p["refined_geometry"][k] for k in lkeys] for p in pairs], dtype=float)
        inv0 = sum(p["baseline_geometry"]["chirality_inversions"] for p in pairs)
        tri0 = sum(p["baseline_geometry"]["chirality_triplets"] for p in pairs)
        inv1 = sum(p["refined_geometry"]["chirality_inversions"] for p in pairs)
        tri1 = sum(p["refined_geometry"]["chirality_triplets"] for p in pairs)
        lengths = np.asarray([p["length"] for p in pairs])
        slope0 = float(np.polyfit(lengths, b, 1)[0])
        slope1 = float(np.polyfit(lengths, a, 1)[0])
        coord = [
            p["sample_id"]
            for p in pairs
            if p["refined_geometry"]["prediction_radius_gyration_angstrom"] < 1
            or p["refined_geometry"]["prediction_radius_gyration_angstrom"]
            / max(p["refined_geometry"]["target_radius_gyration_angstrom"], 1e-8)
            < 0.5
        ]
        diversity = {}
        for name in strata_names:
            ps = [p for p in pairs if p["stratum"] == name]
            pr = np.asarray([p["refined_geometry"]["prediction_radius_gyration_angstrom"] for p in ps])
            tr = np.asarray([p["refined_geometry"]["target_radius_gyration_angstrom"] for p in ps])
            ratio = float(pr.std() / tr.std()) if tr.std() > 1e-12 else None
            diversity[name] = {
                "prediction_radius_gyration_sd": float(pr.std()),
                "target_radius_gyration_sd": float(tr.std()),
                "sd_ratio_prediction_to_target": ratio,
                "collapse": bool(ratio is not None and ratio < 0.5),
            }
        local0 = float(base_local.mean())
        local1 = float(ref_local.mean())
        checks = {
            "overall_rmse_reduction_ge_30pct": float((b.mean() - a.mean()) / b.mean()) >= 0.30,
            "every_length_stratum_reduction_ge_20pct": all(
                v["paired_rmse_improvement_fraction_of_mean"] >= 0.20 for v in strata.values()
            ),
            "no_length_stratum_worsens": all(v["refined_rmse"] <= v["baseline_rmse"] for v in strata.values()),
            "error_vs_length_slope_not_increased": slope1 <= slope0,
            "finite_outputs": all(p.get("finite") is True for p in pairs),
            "chirality_inversion_rate_not_increased": inv1 / max(tri1, 1) <= inv0 / max(tri0, 1),
            "mean_i_plus_1_i_plus_2_i_plus_3_rmse_reduction_ge_20pct": (local0 - local1) / local0 >= 0.20,
            "no_coordinate_collapse": not coord,
            "no_diversity_collapse": not any(v["collapse"] for v in diversity.values()),
        }
        old_metrics = ev["metrics"]
        if not math.isclose(
            float(a.mean()), float(old_metrics["refined_mean_aligned_rmse_angstrom"]), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError(f"raw evaluation disagrees with recorded refined mean at exposure {exposure}")
        reviewed.append(
            {
                "exposure": exposure,
                "global_update": ev["global_update"],
                "checkpoint_sha256": ev["checkpoint_sha256"],
                "development_mean_rmse": float(a.mean()),
                "baseline_mean_rmse": float(b.mean()),
                "rmse_reduction_fraction": float((b.mean() - a.mean()) / b.mean()),
                "paired_percentage_improvement": float(imp.mean()),
                "paired_bootstrap_95_ci": bootstrap(
                    imp.tolist(), int(config["statistics"]["bootstrap_replicates"]), int(config["statistics"]["seed"])
                ),
                "length_strata": strata,
                "error_vs_length_slope": {"baseline": slope0, "refined": slope1},
                "local_distance_reduction_fraction": float((local0 - local1) / local0),
                "chirality": {"baseline_rate": inv0 / max(tri0, 1), "refined_rate": inv1 / max(tri1, 1)},
                "coordinate_collapse_identities": coord,
                "diversity_by_stratum": diversity,
                "gate_checks": checks,
                "gate_adjudication": "passed" if all(checks.values()) else "failed",
            }
        )

    # Exposure-by-identity counts are read from the latest state and must be exact.
    if len(latest.get("identity_exposures", {})) != 2048 or set(latest["identity_exposures"].values()) != {54}:
        raise ValueError("latest state does not prove exactly 54 exposures per training identity")
    traj = []
    for e in BOUNDARIES:
        ev = boundary_data[e]
        tm = ev.get("training_metrics", {})
        traj.append(
            {
                "exposure": e,
                "global_update": ev["global_update"],
                "training_mean_aligned_rmse_angstrom": tm.get("training_mean_aligned_rmse_angstrom"),
                "development_mean_aligned_rmse_angstrom": ev["metrics"]["refined_mean_aligned_rmse_angstrom"],
                "training_development_rmse_gap_angstrom": tm.get("training_development_rmse_gap_angstrom"),
                "training_loss_mean_this_exposure": ev.get("training_loss_mean_this_exposure"),
            }
        )
    worsening, stop = worsening_stop([x["development_mean_rmse"] for x in reviewed])
    if stop != "development_worsened_at_two_consecutive_boundaries":
        raise ValueError("raw trajectory does not reproduce the predeclared two-boundary worsening stop")
    # Original selection rule: lowest development mean RMSE over evaluated boundaries; earliest tie.
    selected = select_checkpoint(reviewed)
    for row in reviewed:
        row["selected"] = row["exposure"] == selected["exposure"]
    finalgates = all(selected["gate_checks"].values())
    finalizer_path = Path(__file__).resolve()
    finalizer_sha256 = sha(finalizer_path)
    finalizer_validation = {
        "finalizer_sha256": finalizer_sha256,
        "implementation": (
            "read-only adjudication; CPU checkpoint loads; no training, prediction, resume, or CUDA initialization"
        ),
        "runner_hashes_independent": True,
        "execution_runner_sha256": HISTORICAL_EXECUTION_RUNNER_SHA256,
        "reporting_repair_runner_sha256": REPORTING_REPAIR_RUNNER_SHA256,
    }
    return {
        "schema": "e010_phase4a_v3_readonly_scientific_review_v1",
        "classification": "pass_all_predeclared_gates" if finalgates else "fail_one_or_more_predeclared_gates",
        "validation": {
            "journal_updates": 92,
            "all_2048_training_identities_exactly_54_exposures": True,
            "evaluated_exposures": list(BOUNDARIES),
            "raw_boundary_evaluations_complete": True,
            "latest_checkpoint_full_state_and_journal_match": True,
            "all_boundary_checkpoints_validated_on_cpu": True,
            "no_cuda_or_inference": True,
        },
        "training_trajectory": traj,
        "development_trajectory": [
            {"exposure": x["exposure"], "mean_rmse": x["development_mean_rmse"]} for x in reviewed
        ],
        "consecutive_worsening_flags": worsening,
        "stop_reason": stop,
        "boundaries": reviewed,
        "selected_exposure": selected["exposure"],
        "selection_rule": (
            "lowest development mean aligned RMSE among evaluated boundaries; earliest boundary wins exact ties"
        ),
        "selected_gate_checks": selected["gate_checks"],
        "runner_lineage": lineage_checks,
        "finalizer_integrity": finalizer_validation,
        "authorization": {"downstream": False, "prospective": False, "phase4b": False},
        "authorizes_downstream": False,
        "downstream_authorized": False,
        "prospective_authorized": False,
        "phase4b_authorized": False,
        "phase4b_prepared": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 4A v3 read-only scientific review",
        "",
        f"**Classification:** `{report['classification']}`  ",
        f"**Selected exposure:** {report['selected_exposure']}  ",
        f"**Stop reason:** `{report['stop_reason']}`  ",
        "**Authorization:** downstream, prospective, and Phase 4B remain false.",
        "",
        "## Development trajectory",
        "",
        "| Exposure | Development mean RMSE | Checkpoint selected |",
        "|---:|---:|:---:|",
    ]
    for b in report["boundaries"]:
        lines.append(f"| {b['exposure']} | {b['development_mean_rmse']:.8f} | {'yes' if b['selected'] else 'no'} |")
    lines += ["", "## Original gates", ""]
    for n, v in report["selected_gate_checks"].items():
        lines.append(f"- {n}: **{str(v).lower()}**")
    return "\n".join(lines) + "\n"


def inventory(root: Path) -> dict[str, Any]:
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        st = p.stat()
        rows.append(
            {
                "path": p.relative_to(root).as_posix(),
                "size_bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "mtime_local": __import__("datetime").datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),
                "sha256": sha(p),
            }
        )
    return {
        "schema": "e010_phase4a_v3_source_inventory_v1",
        "root": str(root.resolve()),
        "file_count": len(rows),
        "files": rows,
    }


def finalize(root: Path = SOURCE, destination: Path | None = None) -> Path:
    """Publish via sibling temporary directory + atomic rename; source stays read-only."""
    report = validate(root)
    dest = destination or (root / DEST.name)
    if dest.exists():
        raise FileExistsError(f"publication already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source_inventory = inventory(root)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.", dir=dest.parent))
    try:
        (tmp / "review.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        # JSON artifacts are persisted before rendering. A renderer failure gets a failure note, not
        # loss of review JSON.
        try:
            md = render_markdown(report)
        except Exception as exc:
            md = (
                f"# Phase 4A v3 read-only scientific review\n\nMarkdown rendering failed: {type(exc).__name__}: {exc}\n"
            )
        (tmp / "review.md").write_text(md)
        (tmp / "corrected_handoff.json").write_text(
            json.dumps(
                {
                    "status": "corrected_non_authorizing_handoff",
                    "previous_claim": (
                        "Scientific adjudication and selection JSON had already been written before Markdown failure."
                    ),
                    "correction": (
                        "FALSE. The prior handoff claim that pre-existing adjudication/selection "
                        "JSON existed as completed scientific outputs was false. The only similarly "
                        "named files found are inside the failed staging directory and are not "
                        "treated as authoritative adjudication or selection outputs."
                    ),
                    "prior_incident_audit_conflict": (
                        "The incident audit/report asserted JSON adjudication and selection had "
                        "completed; this assertion is not accepted as evidence of a completed "
                        "publication."
                    ),
                    "review_recomputed_from": [
                        "92-update journal prefix",
                        "CPU-validated checkpoints",
                        "raw development boundary evaluations 50 through 54",
                    ],
                    "authorizations": AUTH_FALSE,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (tmp / "artifact_inventory.json").write_text(json.dumps(source_inventory, indent=2, sort_keys=True) + "\n")
        sums = {p.name: sha(p) for p in sorted(tmp.iterdir()) if p.is_file() and p.name != "SHA256SUMS.json"}
        (tmp / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true")
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.validate:
        report = validate()
        print(
            json.dumps(
                {
                    "status": "validated_read_only",
                    "journal_updates": 92,
                    "evaluated_exposures": list(BOUNDARIES),
                    "selected_exposure": min(
                        report["boundaries"], key=lambda b: (b["development_mean_rmse"], b["exposure"])
                    )["exposure"],
                },
                indent=2,
            )
        )
    else:
        dest = finalize()
        print(json.dumps({"status": "published", "directory": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
