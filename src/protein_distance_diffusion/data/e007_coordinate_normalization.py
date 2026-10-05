"""Read-only train-split coordinate normalization for E007 Phase 3E-A."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from protein_distance_diffusion.data.e007_coordinate_dataset import E007CoordinateDataset
from protein_distance_diffusion.data.rich_geometry import (
    RichDatasetAuthorization,
    authorize_rich_geometry_dataset,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file, verify_metadata

NORMALIZATION_VERSION = "e007_global_rms_coordinate_radius_v1"
ESTIMATOR_FORMULA = "sqrt(sum(||x_i - protein_centroid||^2) / (3 * total_valid_training_residues))"
NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_joint_training": False,
    "optimizer_created": False,
    "optimizer_updates": 0,
    "backward_executed": False,
    "sampling_executed": False,
    "dataset_modified": False,
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text
    temporary.replace(path)


class CompensatedSum:
    """Kahan-Neumaier scalar accumulation."""

    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        updated = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - updated) + value
        else:
            self.correction += (value - updated) + self.total
        self.total = updated

    @property
    def value(self) -> float:
        return self.total + self.correction


class DeterministicReservoir:
    """Bounded order-deterministic reservoir for descriptive quantiles."""

    def __init__(self, maximum_values: int, seed: int) -> None:
        if maximum_values <= 0:
            raise ValueError("E007 normalization reservoir size must be positive")
        self.maximum_values = maximum_values
        self.values: list[float] = []
        self.seen = 0
        self.generator = np.random.default_rng(seed)

    def add_many(self, values: np.ndarray | list[float]) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            self.seen += 1
            if len(self.values) < self.maximum_values:
                self.values.append(float(value))
                continue
            replacement = int(self.generator.integers(0, self.seen))
            if replacement < self.maximum_values:
                self.values[replacement] = float(value)

    def percentiles(self, probabilities: list[float]) -> dict[str, float]:
        if not self.values:
            return {}
        values = np.asarray(self.values, dtype=np.float64)
        return {f"p{probability * 100:g}": float(np.quantile(values, probability)) for probability in probabilities}


@dataclass
class CalibrationAccumulator:
    maximum_values: int
    percentiles: list[float]
    clash_distance_angstrom: float
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    valid_residue_count: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    candidate_lengths: Counter[str] = field(default_factory=Counter)
    accepted_lengths: Counter[str] = field(default_factory=Counter)
    rejected_lengths: Counter[str] = field(default_factory=Counter)
    missing_coordinate_patterns: Counter[str] = field(default_factory=Counter)
    sequence_length_counts: Counter[int] = field(default_factory=Counter)
    valid_residue_count_distribution: Counter[int] = field(default_factory=Counter)
    finite_valid_coordinate_sample_count: int = 0
    nonfinite_valid_coordinate_sample_count: int = 0
    mask_padding_consistent_sample_count: int = 0
    mask_padding_inconsistent_sample_count: int = 0
    total_continuity_true: int = 0
    total_continuity_false: int = 0
    total_chain_breaks: int = 0
    total_nonneighbor_pairs: int = 0
    total_clashes: int = 0
    duplicate_sample_id_count: int = 0
    duplicate_coordinate_hash_count: int = 0
    radius_values: list[float] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    seen_sample_ids: set[str] = field(default_factory=set)
    seen_coordinate_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.centered_squared_sum = CompensatedSum()
        self.axis_sums = [CompensatedSum() for _ in range(3)]
        self.axis_squared_sums = [CompensatedSum() for _ in range(3)]
        self.centroid_residual = DeterministicReservoir(self.maximum_values, 71001)
        self.coordinate_components = [DeterministicReservoir(self.maximum_values, 71011 + axis) for axis in range(3)]
        self.adjacent_distances = DeterministicReservoir(self.maximum_values, 71021)

    def reject(self, row: dict[str, Any], stratum: str, reasons: list[str]) -> dict[str, Any]:
        self.rejected_count += 1
        self.rejected_lengths[stratum] += 1
        for reason in reasons:
            self.rejection_reasons[reason] += 1
        return {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "sequence_length": int(row["sequence_length"]),
            "length_stratum": stratum,
            "accepted": False,
            "rejection_reasons": sorted(reasons),
            "valid_residue_count": int(row["residue_mask"].sum()),
        }

    def accept(self, row: dict[str, Any], stratum: str) -> dict[str, Any]:
        coordinates = row["coordinates"].detach().cpu().numpy().astype(np.float64, copy=False)
        mask = row["residue_mask"].detach().cpu().numpy().astype(bool, copy=False)
        continuity = row["chain_continuity_mask"].detach().cpu().numpy().astype(bool, copy=False)
        valid = coordinates[mask]
        centroid = valid.mean(axis=0, dtype=np.float64)
        centered = valid - centroid
        squared_norms = np.einsum("ij,ij->i", centered, centered, dtype=np.float64)
        centered_squared = float(squared_norms.sum(dtype=np.float64))
        radius = math.sqrt(centered_squared / len(valid))
        self.centered_squared_sum.add(centered_squared)
        self.valid_residue_count += len(valid)
        self.accepted_count += 1
        self.accepted_lengths[stratum] += 1
        self.radius_values.append(radius)
        residual = centered.mean(axis=0, dtype=np.float64)
        self.centroid_residual.add_many(np.abs(residual))
        for axis in range(3):
            self.axis_sums[axis].add(float(centered[:, axis].sum(dtype=np.float64)))
            self.axis_squared_sums[axis].add(float(np.square(centered[:, axis]).sum(dtype=np.float64)))
            self.coordinate_components[axis].add_many(centered[:, axis])
        if len(valid) > 1:
            adjacent = np.linalg.norm(np.diff(valid, axis=0), axis=1)
            self.adjacent_distances.add_many(adjacent[continuity])
        if len(valid) > 2:
            differences = valid[:, None, :] - valid[None, :, :]
            distances = np.linalg.norm(differences, axis=-1)
            nonneighbor = np.triu(np.ones((len(valid), len(valid)), dtype=bool), k=2)
            self.total_nonneighbor_pairs += int(nonneighbor.sum())
            self.total_clashes += int(((distances < self.clash_distance_angstrom) & nonneighbor).sum())
        coordinate_hash = hashlib.sha256()
        coordinate_hash.update(np.ascontiguousarray(coordinates, dtype="<f4").tobytes())
        coordinate_hash.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
        digest = coordinate_hash.hexdigest()
        if digest in self.seen_coordinate_hashes:
            self.duplicate_coordinate_hash_count += 1
        self.seen_coordinate_hashes.add(digest)
        self.sample_ids.append(str(row["sample_id"]))
        return {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "sequence_length": int(row["sequence_length"]),
            "length_stratum": stratum,
            "accepted": True,
            "rejection_reasons": [],
            "valid_residue_count": len(valid),
            "radius_of_gyration_angstrom": radius,
            "centroid_residual_max_abs_angstrom": float(np.abs(residual).max()),
            "coordinate_sha256": digest,
            "source_sha256": row["source_sha256"],
            "npz_sha256": row["npz_sha256"],
        }

    def observe_candidate(self, row: dict[str, Any], stratum: str) -> list[str]:
        self.candidate_count += 1
        self.candidate_lengths[stratum] += 1
        self.sequence_length_counts[int(row["sequence_length"])] += 1
        sample_id = str(row["sample_id"])
        if sample_id in self.seen_sample_ids:
            self.duplicate_sample_id_count += 1
        self.seen_sample_ids.add(sample_id)
        mask = row["residue_mask"].detach().cpu().numpy().astype(bool, copy=False)
        coordinates = row["coordinates"].detach().cpu().numpy()
        continuity = row["chain_continuity_mask"].detach().cpu().numpy().astype(bool, copy=False)
        missing = int((~mask).sum())
        self.valid_residue_count_distribution[int(mask.sum())] += 1
        self.missing_coordinate_patterns["complete" if missing == 0 else f"missing_{missing}"] += 1
        self.total_continuity_true += int(continuity.sum())
        self.total_continuity_false += int((~continuity).sum())
        self.total_chain_breaks += int((~continuity).sum())
        finite_valid = bool(np.isfinite(coordinates[mask]).all())
        masked_zero = bool(np.all(coordinates[~mask] == 0))
        self.finite_valid_coordinate_sample_count += int(finite_valid)
        self.nonfinite_valid_coordinate_sample_count += int(not finite_valid)
        self.mask_padding_consistent_sample_count += int(masked_zero)
        self.mask_padding_inconsistent_sample_count += int(not masked_zero)
        reasons = []
        if not mask.any():
            reasons.append("no_valid_calpha")
        if not mask.all():
            reasons.append("missing_calpha")
        if not continuity.all():
            reasons.append("chain_break")
        if not finite_valid:
            reasons.append("nonfinite_valid_coordinates")
        if not masked_zero:
            reasons.append("masked_coordinate_not_zero")
        return reasons

    def result(self) -> dict[str, Any]:
        if not self.accepted_count or not self.valid_residue_count:
            raise ValueError("E007 normalization has no accepted training coordinates")
        radii = np.asarray(self.radius_values, dtype=np.float64)
        scale = math.sqrt(self.centered_squared_sum.value / (3 * self.valid_residue_count))
        protein_equal_scale = math.sqrt(float(np.square(radii).mean()) / 3)
        axis_means = [value.value / self.valid_residue_count for value in self.axis_sums]
        axis_standard_deviations = [
            math.sqrt(max(0.0, value.value / self.valid_residue_count - mean**2))
            for value, mean in zip(self.axis_squared_sums, axis_means, strict=True)
        ]
        return {
            "candidate_sample_count": self.candidate_count,
            "accepted_sample_count": self.accepted_count,
            "rejected_sample_count": self.rejected_count,
            "rejection_reason_counts": dict(sorted(self.rejection_reasons.items())),
            "valid_residue_count": self.valid_residue_count,
            "coordinate_scale_angstrom": scale,
            "protein_equal_median_radius_of_gyration_angstrom": float(np.median(radii)),
            "token_weighted_rms_radius_angstrom": math.sqrt(self.centered_squared_sum.value / self.valid_residue_count),
            "alternative_protein_equal_coordinate_scale_angstrom": protein_equal_scale,
            "coordinate_component_means_after_centering_angstrom": axis_means,
            "coordinate_component_standard_deviations_angstrom": axis_standard_deviations,
            "coordinate_component_percentiles_angstrom": {
                axis: reservoir.percentiles(self.percentiles)
                for axis, reservoir in zip(("x", "y", "z"), self.coordinate_components, strict=True)
            },
            "radius_of_gyration_percentiles_angstrom": {
                f"p{probability * 100:g}": float(np.quantile(radii, probability)) for probability in self.percentiles
            },
            "adjacent_calpha_distance_percentiles_angstrom": self.adjacent_distances.percentiles(self.percentiles),
            "centroid_residual_absolute_percentiles_angstrom": self.centroid_residual.percentiles(self.percentiles),
            "nonneighbor_clash_fraction": self.total_clashes / max(self.total_nonneighbor_pairs, 1),
            "nonneighbor_pair_count": self.total_nonneighbor_pairs,
            "nonneighbor_clash_count": self.total_clashes,
            "continuity_true_count": self.total_continuity_true,
            "continuity_false_count": self.total_continuity_false,
            "chain_break_count": self.total_chain_breaks,
            "missing_coordinate_pattern_counts": dict(sorted(self.missing_coordinate_patterns.items())),
            "sequence_length_counts": {
                str(length): count for length, count in sorted(self.sequence_length_counts.items())
            },
            "valid_residue_count_distribution": {
                str(count): samples for count, samples in sorted(self.valid_residue_count_distribution.items())
            },
            "finite_coordinate_status": {
                "finite_valid_coordinate_samples": self.finite_valid_coordinate_sample_count,
                "nonfinite_valid_coordinate_samples": self.nonfinite_valid_coordinate_sample_count,
            },
            "padding_mask_consistency": {
                "consistent_samples": self.mask_padding_consistent_sample_count,
                "inconsistent_samples": self.mask_padding_inconsistent_sample_count,
            },
            "split_identity": "train",
            "duplicate_sample_id_count": self.duplicate_sample_id_count,
            "duplicate_coordinate_hash_count": self.duplicate_coordinate_hash_count,
            "fitted_sample_id_sha256": _canonical_sha256(sorted(self.sample_ids)),
            "length_strata": {
                "candidate": dict(sorted(self.candidate_lengths.items())),
                "accepted": dict(sorted(self.accepted_lengths.items())),
                "rejected": dict(sorted(self.rejected_lengths.items())),
            },
            "robust_percentile_method": {
                "algorithm": "deterministic_reservoir_v1",
                "maximum_values_per_stream": self.maximum_values,
                "coordinate_values_seen_by_axis": [item.seen for item in self.coordinate_components],
                "adjacent_values_seen": self.adjacent_distances.seen,
            },
        }


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != NORMALIZATION_VERSION:
        raise ValueError("E007 normalization configuration version contradiction")
    if payload.get("split") != "train":
        raise ValueError("E007 normalization must fit the training split only")
    if payload.get("formula") != ESTIMATOR_FORMULA:
        raise ValueError("E007 normalization estimator formula changed")
    if payload.get("coordinate_units") != "angstrom":
        raise ValueError("E007 normalization requires Angstrom coordinates")
    probabilities = [float(value) for value in payload["robust_percentiles"]]
    if probabilities != sorted(probabilities) or any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("E007 normalization percentiles are invalid")
    return payload


def _verify_phase3d(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "phase3d_config": (config["phase3d"]["config_path"], config["phase3d"]["config_sha256"]),
        "phase3d_report": (config["phase3d"]["report_path"], config["phase3d"]["report_sha256"]),
        "phase3d_protocol": (config["phase3d"]["protocol_path"], config["phase3d"]["protocol_sha256"]),
        "coordinate_model_contract": (
            config["coordinate_model_contract"]["path"],
            config["coordinate_model_contract"]["sha256"],
        ),
    }
    observed = {}
    for name, (path_text, expected) in paths.items():
        observed[name] = sha256_file(Path(path_text))
        if observed[name] != expected:
            raise ValueError(f"E007 normalization prerequisite hash contradiction: {name}")
    report = json.loads(Path(paths["phase3d_report"][0]).read_text())
    protocol = json.loads(Path(paths["phase3d_protocol"][0]).read_text())
    if report.get("status") != "completed" or protocol.get("status") != "completed":
        raise ValueError("E007 Phase-3D evidence is incomplete")
    if report.get("classification") != "small_capacity_sufficient":
        raise ValueError("E007 Phase-3D capacity classification contradiction")
    if report.get("parameter_counts", {}).get("production") != 7_586_505:
        raise ValueError("E007 production coordinate model parameter count contradiction")
    production = [row for row in report.get("capacity_seed_results", []) if row.get("capacity") == "production"]
    if len(production) != 3 or any(
        row.get("joint_polymer_quality", {}).get("passed") is not True for row in production
    ):
        raise ValueError("E007 production capacity did not pass every global polymer gate")
    if report.get("protected_inputs_unchanged") is not True:
        raise ValueError("E007 Phase-3D protected inputs changed")
    return {"hashes": observed, "report": report}


def verify_normalization_prerequisites(config: dict[str, Any]) -> dict[str, Any]:
    """Verify metadata and completed evidence without scanning scientific shards."""
    phase3d = _verify_phase3d(config)
    metadata = verify_metadata(config["dataset"])
    root = Path(config["dataset"]["root"])
    protocol = json.loads((root / "protocol.json").read_text())
    dataset_normalization = json.loads((root / "normalization.json").read_text())
    if protocol.get("status") != "completed" or protocol.get("mode") != "full":
        raise ValueError("E007 normalization requires a completed full sidecar dataset")
    if protocol.get("authorizes_training") is not True:
        raise ValueError("E007 sidecar dataset is not authorized")
    if dataset_normalization.get("coordinate_units") != "angstrom":
        raise ValueError("E007 sidecar coordinate units are not Angstrom")
    return {
        "phase3d_hashes": phase3d["hashes"],
        "dataset_hashes": metadata["hashes"],
        "dataset": metadata,
        "coordinate_units": dataset_normalization["coordinate_units"],
    }


def _authorize_dataset(config: dict[str, Any]) -> RichDatasetAuthorization:
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["root"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
    )


def _protected_identity(authorization: RichDatasetAuthorization) -> dict[str, Any]:
    return {
        "protocol_sha256": authorization.protocol_sha256,
        "schema_sha256": authorization.schema_sha256,
        "vocabulary_sha256": authorization.vocabulary_sha256,
        "normalization_sha256": authorization.normalization_sha256,
        "shard_inventory_sha256": authorization.shard_inventory_sha256,
        "scientific_shard_count": len(authorization.observed_shard_hashes),
        "scientific_shard_hashes_sha256": _canonical_sha256(authorization.observed_shard_hashes),
        "split_counts": authorization.split_counts,
    }


def _length_stratum(length: int, strata: list[dict[str, Any]]) -> str:
    matches = [str(record["name"]) for record in strata if int(record["minimum"]) <= length <= int(record["maximum"])]
    if len(matches) != 1:
        raise ValueError(f"E007 sequence length has no unique configured stratum: {length}")
    return matches[0]


def plan_coordinate_normalization(config_path: str | Path) -> dict[str, Any]:
    """Verify Phase-3E-A metadata without opening any scientific shard."""
    config_path = Path(config_path)
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 normalization output already exists: {output} or {staging}")
    prerequisites = verify_normalization_prerequisites(config)
    return {
        "status": "planned_phase3e_a_non_authorizing",
        "version": NORMALIZATION_VERSION,
        "output_dir": str(output),
        "fitting_split": "train",
        "candidate_sample_count": int(prerequisites["dataset"]["eligible_split_counts"]["train"]),
        "estimator_formula": ESTIMATOR_FORMULA,
        "coordinate_units": "angstrom",
        "centering_policy": "per_protein_mean_of_finite_valid_calpha_coordinates",
        "mask_policy": "existing_calpha_and_chain_continuity_masks",
        "selection_policy": config["selection_policy"],
        "validation_coordinates_inspected": False,
        "sequence_token_identities_used": False,
        "coordinate_payloads_scanned": False,
        "model_created": False,
        "phase3d_scientific_selection": "small_capacity_sufficient",
        "subsequent_engineering_capacity_choice": {
            "model": "production",
            "parameter_count": 7_586_505,
            "interpretation": "owner design choice; Phase 3D did not prove production capacity necessary",
        },
        "prerequisite_hashes": {
            **prerequisites["phase3d_hashes"],
            **{f"dataset_{key}": value for key, value in prerequisites["dataset_hashes"].items()},
        },
        "configuration_sha256": sha256_file(config_path),
        **NON_AUTHORIZING,
    }


def calibrate_coordinate_normalization(config_path: str | Path) -> dict[str, Any]:
    """Scan the immutable train split and atomically publish Phase-3E-A evidence."""
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    plan = plan_coordinate_normalization(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = datetime.now(UTC).isoformat()
    _atomic_json(staging / "heartbeat.json", {"status": "running", "stage": "authorization", **NON_AUTHORIZING})
    try:
        authorization_before = _authorize_dataset(config)
        protected_before = _protected_identity(authorization_before)
        dataset = E007CoordinateDataset(authorization_before, split="train")
        accumulator = CalibrationAccumulator(
            maximum_values=int(config["maximum_streaming_values"]),
            percentiles=[float(value) for value in config["robust_percentiles"]],
            clash_distance_angstrom=float(config["clash_distance_angstrom"]),
        )
        audit_path = staging / "per_sample_audit.jsonl.gz"
        with _deterministic_gzip_text(audit_path) as audit:
            for index in range(len(dataset)):
                row = dataset[index]
                if row["split"] != "train":
                    raise ValueError("E007 validation leakage reached normalization fitting")
                stratum = _length_stratum(int(row["sequence_length"]), config["length_strata"])
                reasons = accumulator.observe_candidate(row, stratum)
                record = accumulator.reject(row, stratum, reasons) if reasons else accumulator.accept(row, stratum)
                if config.get("write_per_sample_audit", True):
                    audit.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                if (index + 1) % 1000 == 0:
                    _atomic_json(
                        staging / "heartbeat.json",
                        {
                            "status": "running",
                            "stage": "train_split_scan",
                            "processed_samples": index + 1,
                            "candidate_samples": len(dataset),
                            "accepted_samples": accumulator.accepted_count,
                            "rejected_samples": accumulator.rejected_count,
                            "updated_utc": datetime.now(UTC).isoformat(),
                            **NON_AUTHORIZING,
                        },
                    )
        if accumulator.duplicate_sample_id_count:
            raise ValueError("E007 normalization encountered duplicate sample IDs")
        statistics = accumulator.result()
        _atomic_json(staging / "length_stratified_statistics.json", statistics["length_strata"])
        payload_hashes = {
            "per_sample_audit.jsonl.gz": sha256_file(audit_path),
            "length_stratified_statistics.json": sha256_file(staging / "length_stratified_statistics.json"),
        }
        normalization = {
            "version": NORMALIZATION_VERSION,
            "coordinate_scale_angstrom": statistics["coordinate_scale_angstrom"],
            "estimator_formula": ESTIMATOR_FORMULA,
            "coordinate_units": "angstrom",
            "centering_policy": "subtract each protein's finite valid-residue centroid",
            "rotation_policy": "none",
            "alignment_policy": "none",
            "per_protein_rescaling": False,
            "mask_policy": "finite valid C-alpha residues from immutable ca_mask",
            "fitting_split": "train",
            "accepted_sample_count": statistics["accepted_sample_count"],
            "valid_residue_count": statistics["valid_residue_count"],
            "fitted_sample_id_sha256": statistics["fitted_sample_id_sha256"],
            "dataset_prerequisite_hashes": plan["prerequisite_hashes"],
            "configuration_sha256": plan["configuration_sha256"],
            "output_payload_hashes": payload_hashes,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "normalization.json", normalization)
        authorization_after = _authorize_dataset(config)
        protected_after = _protected_identity(authorization_after)
        if protected_before != protected_after:
            raise ValueError("E007 normalization protected dataset identity changed")
        report = {
            **plan,
            "status": "completed_non_authorizing",
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "statistics": statistics,
            "normalization_sha256": sha256_file(staging / "normalization.json"),
            "output_payload_hashes": payload_hashes,
            "protected_identity_before": protected_before,
            "protected_identity_after": protected_after,
            "protected_inputs_unchanged": True,
            "validation_coordinates_inspected": False,
            "sequence_token_identities_used": False,
            "model_created": False,
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": NORMALIZATION_VERSION,
            "status": "completed_non_authorizing",
            "report_sha256": sha256_file(staging / "report.json"),
            "normalization_sha256": report["normalization_sha256"],
            "protected_inputs_unchanged": True,
            "fitting_split": "train",
            "configuration_sha256": plan["configuration_sha256"],
            **NON_AUTHORIZING,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "completed_utc": report["completed_utc"],
                "processed_samples": statistics["candidate_sample_count"],
                "accepted_samples": statistics["accepted_sample_count"],
                "rejected_samples": statistics["rejected_sample_count"],
                "report_sha256": protocol["report_sha256"],
                **NON_AUTHORIZING,
            },
        )
        staging.replace(output)
        return report
    except BaseException as error:
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "failed_utc": datetime.now(UTC).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                **NON_AUTHORIZING,
            },
        )
        raise
