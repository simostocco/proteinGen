"""Read-only Phase-3A planning helpers for the E007 coordinate generator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import REQUIRED_COLUMNS
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("E007 configuration must be a mapping")
    return payload


def verify_metadata(dataset: dict[str, Any]) -> dict[str, Any]:
    root = Path(dataset["root"])
    identities = {
        "protocol": ("protocol.json", "protocol_sha256"),
        "schema": ("schema.json", "schema_sha256"),
        "vocabulary": ("vocabulary.json", "vocabulary_sha256"),
        "normalization": ("normalization.json", "normalization_sha256"),
        "shard_inventory": ("shard_hashes.sha256", "shard_inventory_sha256"),
    }
    observed = {}
    for name, (filename, key) in identities.items():
        value = sha256_file(root / filename)
        if value != dataset[key]:
            raise ValueError(f"E007 {name} hash contradiction")
        observed[name] = value
    protocol = json.loads((root / "protocol.json").read_text())
    schema = json.loads((root / "schema.json").read_text())
    available = sorted(schema["columns"])
    missing = sorted(set(REQUIRED_COLUMNS) - set(available))
    if missing:
        raise ValueError(f"E007 authoritative sidecar schema lacks columns: {missing}")
    if schema["schema_version"] != dataset["required_schema_version"]:
        raise ValueError("E007 sidecar schema-version contradiction")
    return {
        "hashes": observed,
        "available_schema_columns": available,
        "required_projected_columns": list(REQUIRED_COLUMNS),
        "missing_required_columns": [],
        "candidate_counts": dict(protocol["definitive_observed_counts"]),
        "eligible_split_counts": dict(protocol["eligible_split_counts"]),
    }


def generator_plan(config: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_metadata(config["dataset"])
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E007 future output already exists: {output}")
    model = EquivariantPairCoordinateUNet(**config["model"])
    return {
        "status": "planned_phase3a_non_authorizing",
        "architecture_version": config["architecture_version"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "downsample_factor": model.downsample_factor,
        "pair_grid_complexity": "O(N^2)",
        "coordinate_state_complexity": "O(N)",
        "coordinate_normalization_status": "pending_real_train_split_calibration",
        "future_output_directory": str(output),
        "dataset": metadata,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_sequence_conditioning": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "real_data_scanned": False,
        "dataset_reprocessed": False,
        "e004_weights_loaded": False,
    }


def normalization_plan(config: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_metadata(config["dataset"])
    output = Path(config["output_path"])
    if output.exists():
        raise FileExistsError(f"E007 future normalization artifact already exists: {output}")
    split_counts = metadata["eligible_split_counts"]
    return {
        "status": "planned_phase3a_non_authorizing",
        "version": config["version"],
        "formula": config["formula"],
        "split": "train",
        "selection_policy": config["selection_policy"],
        "train_candidate_count": int(split_counts["train"]),
        "accepted_sample_count": None,
        "rejected_sample_count": None,
        "acceptance_count_status": "pending_required_real_train_split_scan",
        "future_output_path": str(output),
        "dataset": metadata,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_sequence_conditioning": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "real_data_scanned": False,
        "dataset_reprocessed": False,
        "e004_weights_loaded": False,
    }
