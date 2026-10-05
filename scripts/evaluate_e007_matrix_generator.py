#!/usr/bin/env python3
"""Plan or execute the non-authorizing E007 matrix-generator audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PLAN_VERSION = "e007_matrix_generator_audit_plan_v1"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(root: Path, record: dict[str, Any], *, label: str) -> dict[str, str]:
    raw_path = Path(str(record["path"]))
    path = raw_path if raw_path.is_absolute() else root / raw_path
    if not path.is_file():
        raise FileNotFoundError(f"E007 {label} is missing: {path}")
    expected = str(record["sha256"]).lower()
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 {label} SHA-256 contradiction: expected {expected}, observed {observed}: {path}")
    return {"path": str(raw_path), "sha256": observed}


def build_plan(config_path: str | Path, *, repository_root: str | Path = ".") -> dict[str, Any]:
    """Validate pinned inputs and return a read-only audit plan."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if config.get("version") != PLAN_VERSION:
        raise ValueError(f"Unsupported E007 audit-plan version: {config.get('version')!r}")
    output = root / str(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E007 matrix-audit output already exists and will not be overwritten: {output}")

    verified = {
        name: _verified_file(root, record, label=name.replace("_", " "))
        for name, record in config["protected_inputs"].items()
    }
    declared_hashes = {
        "generator_checkpoint": str(config["generator"]["checkpoint_sha256"]).lower(),
        "generator_training_config": str(config["generator"]["training_config_sha256"]).lower(),
        "normalization": str(config["generator"]["normalization_sha256"]).lower(),
        "train_manifest": str(config["dataset"]["train_manifest_sha256"]).lower(),
        "validation_manifest": str(config["dataset"]["validation_manifest_sha256"]).lower(),
    }
    for name, expected_hash in declared_hashes.items():
        if verified[name]["sha256"] != expected_hash:
            raise ValueError(f"E007 declared {name} identity contradicts its protected-input hash")
    protocol = json.loads((root / verified["generator_protocol"]["path"]).read_text(encoding="utf-8"))
    generator = config["generator"]
    protocol_contract = {
        "checkpoint_path": protocol.get("checkpoint_path"),
        "checkpoint_sha256": protocol.get("checkpoint_sha256"),
        "training_config_path": protocol.get("training_config_path"),
        "training_config_sha256": protocol.get("training_config_sha256"),
        "normalization_path": protocol.get("normalization_path"),
        "normalization_sha256": protocol.get("normalization_sha256"),
        "reference_manifest_path": protocol.get("reference_manifest_path"),
        "reference_manifest_sha256": protocol.get("reference_manifest_sha256"),
        "status": protocol.get("status"),
    }
    expected_contract = {
        "checkpoint_path": generator["checkpoint_path"],
        "checkpoint_sha256": verified["generator_checkpoint"]["sha256"],
        "training_config_path": generator["training_config_path"],
        "training_config_sha256": verified["generator_training_config"]["sha256"],
        "normalization_path": generator["normalization_path"],
        "normalization_sha256": verified["normalization"]["sha256"],
        "reference_manifest_path": config["dataset"]["validation_manifest_path"],
        "reference_manifest_sha256": verified["validation_manifest"]["sha256"],
        "status": "completed",
    }
    if protocol_contract != expected_contract:
        raise ValueError("E007 generator protocol contradicts the pinned checkpoint/configuration/data identity")
    normalization = json.loads((root / verified["normalization"]["path"]).read_text(encoding="utf-8"))
    expected_normalization = config["matrix_representation"]["normalization"]
    if normalization.get("mode") != expected_normalization["mode"] or float(normalization.get("scale")) != float(
        expected_normalization["scale_angstrom"]
    ):
        raise ValueError("E007 normalization contract contradicts the pinned normalization artifact")

    lengths = [int(value) for value in config["audit"]["lengths"]]
    requested = {str(key): int(value) for key, value in config["audit"]["generated_counts_by_length"].items()}
    if sorted(map(int, requested)) != sorted(lengths) or any(value <= 0 for value in requested.values()):
        raise ValueError("generated_counts_by_length must contain one positive count for each configured length")
    real_requested = {str(key): int(value) for key, value in config["audit"]["real_counts_by_length"].items()}
    if sorted(map(int, real_requested)) != sorted(lengths) or any(value <= 0 for value in real_requested.values()):
        raise ValueError("real_counts_by_length must contain one positive count for each configured length")
    if not bool(config["bounds"]["process_candidates_individually"]) or not bool(
        config["bounds"]["prohibit_cross_candidate_averaging"]
    ):
        raise ValueError("E007 audit requires individual candidate processing and prohibits cross-candidate averaging")
    return {
        "version": PLAN_VERSION,
        "status": "planned",
        "mode": "plan_only",
        "experiment_id": config["experiment_id"],
        "authoritative_generator_identified": True,
        "generator": generator,
        "dataset": config["dataset"],
        "matrix_representation": config["matrix_representation"],
        "panels": config["audit"]["panels"],
        "lengths": lengths,
        "generated_counts_by_length": requested,
        "generated_candidate_count": sum(requested.values()),
        "real_reference_counts_by_length": real_requested,
        "control_definitions": config["audit"]["corruption_controls"],
        "missing_valid_pair_control_supported": bool(config["audit"]["missing_valid_pair_control_supported"]),
        "scientific_threshold_policy": config["audit"]["scientific_threshold_policy"],
        "fatal_contract_thresholds": config["fatal_contract_thresholds"],
        "candidate_semantics": "each generated matrix remains an independent seed-identified candidate",
        "cross_candidate_averaging": False,
        "bounds": config["bounds"],
        "output_dir": str(config["output_dir"]),
        "output_directory_absent": True,
        "protected_inputs": verified,
        "protocol_contract_verified": True,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "real_audit_executed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args.config)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    from protein_distance_diffusion.evaluation.e007_matrix_audit import run_matrix_generator_audit

    output = run_matrix_generator_audit(args.config, plan=plan)
    print(output)


if __name__ == "__main__":
    main()
