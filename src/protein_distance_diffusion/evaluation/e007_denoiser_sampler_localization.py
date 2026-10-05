"""Bounded E007 Phase-3I.1 denoiser-versus-sampler localization."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from protein_distance_diffusion.evaluation import e007_geometry_generator_capability as phase3i
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file

VERSION = "e007_denoiser_sampler_localization_v1"
NON_AUTHORIZING = {
    "training_performed": False,
    "model_created": False,
    "optimizer_created": False,
    "backward_performed": False,
    "optimizer_updates": 0,
    "checkpoint_modified": False,
    "dataset_modified": False,
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_additional_training": False,
    "authorizes_phase3j": False,
}
DECISION_CATEGORIES = {
    "denoiser_local_geometry_failure",
    "reverse_sampler_error_accumulation",
    "combined_denoiser_and_sampler_failure",
    "chirality_symmetry_limitation",
    "long_length_scaling_failure",
    "bounded_diagnostic_inconclusive",
}
ENVELOPE_FIELDS = (
    "adjacent_distance_rmse_to_3_8_angstrom",
    "discontinuity_fraction",
    "distance_i_plus_2_mean_angstrom",
    "distance_i_plus_3_mean_angstrom",
    "bond_angle_mean_degrees",
    "clash_fraction",
    "signed_pseudo_dihedral_mean",
    "signed_pseudo_dihedral_positive_fraction",
    "signed_tetrahedral_volume_mean",
    "radius_of_gyration_angstrom",
    "contact_density_8a",
    "contact_order_8a",
    "contact_graph_component_count_8a",
    "contact_graph_largest_component_fraction_8a",
    "asphericity",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]], compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary, compression=compression)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"E007 Phase-3I.1 prerequisite is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3I.1 prerequisite hash contradiction: {label}")
    return observed


def load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase-3I.1 configuration version contradiction")
    panel = payload.get("panel", {})
    if panel.get("lengths") != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase-3I.1 length contract changed")
    if int(panel.get("samples_per_length", 0)) != 8 or int(panel.get("reverse_seeds_per_length", 0)) != 4:
        raise ValueError("E007 Phase-3I.1 panel cardinality changed")
    timesteps = [int(item["timestep"]) for item in payload["one_step"]["timesteps"]]
    if len(timesteps) != 4 or timesteps != sorted(timesteps) or timesteps[-1] >= int(payload["diffusion_steps"]):
        raise ValueError("E007 Phase-3I.1 one-step timestep contract is invalid")
    milestones = list(map(int, payload["reverse_trajectory"]["milestones"]))
    if milestones != sorted(set(milestones), reverse=True) or milestones[0] != 499 or milestones[-1] != 0:
        raise ValueError("E007 Phase-3I.1 reverse milestones are invalid")
    if payload["reverse_trajectory"].get("save_full_coordinate_tensors") is not False:
        raise ValueError("E007 Phase-3I.1 must not retain full trajectory tensors")
    if int(payload["selected_checkpoint"].get("optimizer_update", -1)) != 9000:
        raise ValueError("E007 Phase-3I.1 checkpoint contract changed")
    recovery = payload.get("recovery", {})
    expected = {
        "expected_one_step_units": 320,
        "expected_reverse_trajectory_units": 20,
        "expected_total_units": 340,
        "expected_total_forwards": 10320,
        "one_step_native_rows_per_unit": 3,
        "one_step_reflected_rows_per_unit": 2,
        "reverse_rows_per_unit": 16,
    }
    if any(int(recovery.get(key, -1)) != value for key, value in expected.items()):
        raise ValueError("E007 Phase-3I.1 recovery cardinality contract changed")
    if recovery.get("append_and_fsync_journal") is not True:
        raise ValueError("E007 Phase-3I.1 journal durability contract changed")
    return payload


def _phase3i_files(config: Mapping[str, Any]) -> dict[str, tuple[Path, str]]:
    section = config["phase3i"]
    root = Path(section["root"])
    return {
        "phase3i_configuration": (Path(section["configuration_path"]), section["configuration_sha256"]),
        "phase3i_report": (root / "report.json", section["report_sha256"]),
        "phase3i_protocol": (root / "protocol.json", section["protocol_sha256"]),
        "phase3i_per_sample_metrics": (root / "per_sample_metrics.parquet", section["per_sample_metrics_sha256"]),
        "phase3i_reference_manifest": (root / "reference_manifest.parquet", section["reference_manifest_sha256"]),
        "phase3i_reference_selection": (root / "reference_selection.json", section["reference_selection_sha256"]),
        "phase3i_local_geometry": (root / "local_geometry_summary.json", section["local_geometry_summary_sha256"]),
        "phase3i_global_geometry": (root / "global_geometry_summary.json", section["global_geometry_summary_sha256"]),
        "phase3i_chirality": (root / "chirality_summary.json", section["chirality_summary_sha256"]),
        "phase3i_inventory": (root / "artifact_inventory.json", section["artifact_inventory_sha256"]),
    }


def verify_prerequisites(config: Mapping[str, Any], *, full: bool) -> dict[str, Any]:
    hashes = {
        label: _verify_file(path, expected, label)
        for label, (path, expected) in {
            "selected_checkpoint": (
                Path(config["selected_checkpoint"]["path"]),
                config["selected_checkpoint"]["sha256"],
            ),
            "phase3f_configuration": (
                Path(config["model_source"]["phase3f_config_path"]),
                config["model_source"]["phase3f_config_sha256"],
            ),
            "continuation_configuration": (
                Path(config["model_source"]["continuation_config_path"]),
                config["model_source"]["continuation_config_sha256"],
            ),
            **_phase3i_files(config),
        }.items()
    }
    inventory = json.loads((Path(config["phase3i"]["root"]) / "artifact_inventory.json").read_text())
    if inventory.get("aggregate_sha256") != config["phase3i"]["artifact_inventory_aggregate_sha256"]:
        raise ValueError("E007 Phase-3I.1 Phase-3I inventory aggregate contradiction")
    if _canonical_sha(inventory.get("artifacts")) != inventory["aggregate_sha256"]:
        raise ValueError("E007 Phase-3I.1 Phase-3I inventory serialization contradiction")
    verified_inventory_entries = 0
    if full:
        root = Path(config["phase3i"]["root"]).resolve()
        seen: set[str] = set()
        for record in inventory["artifacts"]:
            relative = Path(str(record["path"]))
            artifact = (root / relative).resolve()
            logical = relative.as_posix()
            if relative.is_absolute() or not artifact.is_relative_to(root) or logical in seen:
                raise ValueError(f"E007 Phase-3I.1 invalid Phase-3I inventory path: {logical}")
            seen.add(logical)
            if not artifact.is_file() or artifact.stat().st_size != int(record["size_bytes"]):
                raise ValueError(f"E007 Phase-3I.1 Phase-3I inventory size contradiction: {logical}")
            if sha256_file(artifact) != record["sha256"]:
                raise ValueError(f"E007 Phase-3I.1 Phase-3I inventory hash contradiction: {logical}")
            verified_inventory_entries += 1
    report = json.loads((Path(config["phase3i"]["root"]) / "report.json").read_text())
    protocol = json.loads((Path(config["phase3i"]["root"]) / "protocol.json").read_text())
    if report.get("status") != "completed_read_only_non_authorizing" or protocol.get("status") != report["status"]:
        raise ValueError("E007 Phase-3I.1 Phase-3I completion contract failed")
    if report.get("authorizes_training") is not False or protocol.get("protected_inputs_unchanged") is not True:
        raise ValueError("E007 Phase-3I.1 Phase-3I safety contract failed")
    return {
        "hashes": hashes,
        "phase3i_artifact_count": len(inventory["artifacts"]),
        "phase3i_inventory_entries_reverified": verified_inventory_entries,
        "protected_inputs_unchanged": True,
    }


def _phase3i_panel_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = phase3i.load_config(config["phase3i"]["configuration_path"])
    source["panel"] = dict(config["panel"])
    return source


def select_validation_panel(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, diagnostics = phase3i.select_reference_panel(_phase3i_panel_config(config))
    expected = len(config["panel"]["lengths"]) * int(config["panel"]["samples_per_length"])
    if len(rows) != expected or len({row["sample_id"] for row in rows}) != expected:
        raise ValueError("E007 Phase-3I.1 panel count/uniqueness contradiction")
    diagnostics["record_sha256"] = _canonical_sha(
        [
            {
                key: row[key]
                for key in (
                    "sample_id",
                    "target_length",
                    "actual_length",
                    "dataset_shard_path",
                    "shard_row_index",
                )
            }
            for row in rows
        ]
    )
    return rows, diagnostics


def _authoritative_panel_row(
    config: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct one strict Phase-3F coordinate row from its immutable locator."""
    from protein_distance_diffusion.data.e007_coordinate_dataset import REQUIRED_COLUMNS, validate_coordinate_row

    source_config = source_config or _phase3i_panel_config(config)
    root = Path(source_config["dataset"]["root"]).resolve()
    relative = Path(str(record.get("dataset_shard_path", "")))
    shard = (root / relative).resolve()
    if relative.is_absolute() or not shard.is_relative_to(root):
        raise ValueError("E007 Phase-3I.1 authoritative shard escapes dataset root")
    if not shard.is_file():
        raise FileNotFoundError(f"E007 Phase-3I.1 authoritative shard is absent: {relative.as_posix()}")
    raw_columns = tuple(dict.fromkeys((*REQUIRED_COLUMNS, "source_path", "npz_path")))
    raw = phase3i._read_parquet_row(shard, int(record["shard_row_index"]), raw_columns)
    canonical = validate_coordinate_row(raw, split="validation")
    sample_id = str(record["sample_id"])
    if canonical["sample_id"] != sample_id or str(raw["sample_id"]) != sample_id:
        raise ValueError(f"E007 Phase-3I.1 authoritative sample-ID contradiction: {sample_id}")
    if record.get("split") != "validation" or record.get("coordinate_accepted") is not True:
        raise ValueError(f"E007 Phase-3I.1 clean-validation membership contradiction: {sample_id}")
    if canonical.get("accepted_contiguous_single_chain") is not True:
        raise ValueError(f"E007 Phase-3I.1 authoritative coordinate acceptance is false: {sample_id}")
    if canonical.get("coordinate_acceptance_reasons") != ():
        raise ValueError(f"E007 Phase-3I.1 authoritative acceptance reasons contradict eligibility: {sample_id}")
    actual_length = int(record["actual_length"])
    if int(record["length"]) != actual_length or int(canonical["sequence_length"]) != actual_length:
        raise ValueError(f"E007 Phase-3I.1 authoritative length contradiction: {sample_id}")
    if str(record.get("source_path")) != str(raw["source_path"]):
        raise ValueError(f"E007 Phase-3I.1 authoritative source-path contradiction: {sample_id}")

    coordinates, provenance = phase3i.load_reference_coordinates(source_config, record)
    raw_coordinates = np.asarray(raw["ca_coordinates"], dtype=np.float64)
    if not np.array_equal(coordinates, raw_coordinates):
        raise ValueError(f"E007 Phase-3I.1 authoritative coordinate contradiction: {sample_id}")
    canonical_coordinates = canonical["coordinates"].detach().cpu().numpy()
    if not np.array_equal(canonical_coordinates, raw_coordinates.astype(np.float32)):
        raise ValueError(f"E007 Phase-3I.1 canonical coordinate conversion contradiction: {sample_id}")
    if not np.array_equal(canonical["residue_mask"].numpy(), np.asarray(raw["ca_mask"], dtype=bool)):
        raise ValueError(f"E007 Phase-3I.1 authoritative C-alpha mask contradiction: {sample_id}")
    if not np.array_equal(
        canonical["chain_continuity_mask"].numpy(), np.asarray(raw["chain_continuity_mask"], dtype=bool)
    ):
        raise ValueError(f"E007 Phase-3I.1 authoritative continuity mask contradiction: {sample_id}")
    if canonical["source_sha256"] != str(raw["source_sha256"]) or provenance["source_sha256"] != str(
        raw["source_sha256"]
    ):
        raise ValueError(f"E007 Phase-3I.1 authoritative source hash contradiction: {sample_id}")
    if canonical["npz_sha256"] != str(raw["npz_sha256"]) or provenance["npz_sha256"] != str(raw["npz_sha256"]):
        raise ValueError(f"E007 Phase-3I.1 authoritative NPZ hash contradiction: {sample_id}")
    sequence = str(raw["sequence"])
    provenance.update(
        {
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "authoritative_acceptance": True,
            "authoritative_acceptance_policy": "contiguous_single_chain_complete_calpha_v1",
            "canonical_required_fields": list(REQUIRED_COLUMNS),
        }
    )
    return canonical, provenance


def reconstruct_authoritative_panel(
    config: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    *,
    source_config: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every compact selection record against its canonical sidecar row."""
    from protein_distance_diffusion.data.e007_coordinate_dataset import REQUIRED_COLUMNS

    expected = len(config["panel"]["lengths"]) * int(config["panel"]["samples_per_length"])
    if len(panel) != expected:
        raise ValueError("E007 Phase-3I.1 canonical panel count contradiction")
    reconstructed = []
    seen_samples: set[str] = set()
    seen_locators: set[tuple[str, int]] = set()
    resolved_casefold: dict[str, str] = {}
    for selection in panel:
        sample_id = str(selection["sample_id"])
        locator = (str(selection["dataset_shard_path"]), int(selection["shard_row_index"]))
        if sample_id in seen_samples or locator in seen_locators:
            raise ValueError(f"E007 Phase-3I.1 duplicate canonical panel identity: {sample_id}")
        canonical, provenance = _authoritative_panel_row(config, selection, source_config=source_config)
        for name in ("source_resolved_path", "npz_resolved_path"):
            resolved = str(Path(provenance[name]).resolve())
            folded = resolved.casefold()
            prior = resolved_casefold.get(folded)
            if prior is not None and prior != resolved:
                raise ValueError(f"E007 Phase-3I.1 relocation case-collision ambiguity: {prior} != {resolved}")
            resolved_casefold[folded] = resolved
        seen_samples.add(sample_id)
        seen_locators.add(locator)
        reconstructed.append(
            {
                "selection": dict(selection),
                "canonical_row": canonical,
                "provenance": provenance,
            }
        )
    identities = [
        {
            "sample_id": item["selection"]["sample_id"],
            "target_length": int(item["selection"]["target_length"]),
            "actual_length": int(item["selection"]["actual_length"]),
            "dataset_shard_path": item["provenance"]["dataset_shard_path"],
            "shard_row_index": int(item["provenance"]["shard_row_index"]),
            "sequence_sha256": item["provenance"]["sequence_sha256"],
            "coordinate_sha256": item["provenance"]["coordinate_sha256"],
            "source_sha256": item["provenance"]["source_sha256"],
            "npz_sha256": item["provenance"]["npz_sha256"],
        }
        for item in reconstructed
    ]
    return reconstructed, {
        "status": "passed",
        "validated_row_count": len(reconstructed),
        "accepted_row_count": sum(
            item["canonical_row"]["accepted_contiguous_single_chain"] is True for item in reconstructed
        ),
        "required_source_columns": list(dict.fromkeys((*REQUIRED_COLUMNS, "source_path", "npz_path"))),
        "canonical_collator_fields": [
            "sample_id",
            "sequence_length",
            "coordinates",
            "residue_mask",
            "chain_continuity_mask",
            "accepted_contiguous_single_chain",
            "coordinate_acceptance_reasons",
            "source_sha256",
            "npz_sha256",
            "sidecar_schema_version",
        ],
        "identity_sha256": _canonical_sha(identities),
        "casefold_unique_resolved_path_count": len(resolved_casefold),
        "model_created": False,
        "cuda_initialized": False,
    }


def validate_panel_schema(config_path: str | Path) -> dict[str, Any]:
    """Bounded read-only canonical-row validation with no model or CUDA activity."""
    config_path = Path(config_path)
    config = load_config(config_path)
    prerequisites = verify_prerequisites(config, full=False)
    panel, panel_diagnostics = select_validation_panel(config)
    _rows, canonical = reconstruct_authoritative_panel(config, panel)
    return {
        "status": "panel_schema_validated_read_only",
        "configuration_sha256": sha256_file(config_path),
        "panel_sha256": panel_diagnostics["record_sha256"],
        "canonical_panel": canonical,
        "protected_hashes": prerequisites["hashes"],
        "dataset_scan_scope": "40_authoritative_rows_only",
        "output_created": False,
        "model_created": False,
        "cuda_initialized": False,
        **NON_AUTHORIZING,
    }


def reverse_seed_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    namespace = str(config["reverse_trajectory"]["seed_namespace"])
    base = int(config["reverse_trajectory"]["seed_base"])
    count = int(config["panel"]["reverse_seeds_per_length"])
    for length_index, length in enumerate(map(int, config["panel"]["lengths"])):
        for sample_index in range(count):
            seed = base + length_index * 100_000 + sample_index
            records.append(
                {
                    "requested_length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "noise_identity_sha256": _canonical_sha(
                        {"namespace": namespace, "seed": seed, "shape": [1, length, 3]}
                    ),
                }
            )
    return records


def build_work_units(panel: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the immutable, strictly ordered recovery boundaries."""
    units: list[dict[str, Any]] = []
    noise_base = int(config["one_step"]["noise_seed_base"])
    for panel_index, record in enumerate(panel):
        for timestep_index, specification in enumerate(config["one_step"]["timesteps"]):
            seed = noise_base + panel_index * 100 + timestep_index
            noise_identity = _canonical_sha(
                {
                    "namespace": "e007_phase3i1_paired_one_step_noise_v1",
                    "seed": seed,
                    "sample_id": record["sample_id"],
                    "timestep": int(specification["timestep"]),
                    "shape": [1, int(record["actual_length"]), 3],
                }
            )
            for orientation in ("native", "reflected"):
                unit_id = (
                    f"one_step:{record['sample_id']}:{int(record['target_length'])}:"
                    f"{int(record['actual_length'])}:{int(specification['timestep'])}:{orientation}"
                )
                units.append(
                    {
                        "order": len(units),
                        "unit_id": unit_id,
                        "unit_kind": "one_step",
                        "sample_id": str(record["sample_id"]),
                        "panel_index": panel_index,
                        "target_length": int(record["target_length"]),
                        "actual_length": int(record["actual_length"]),
                        "timestep": int(specification["timestep"]),
                        "timestep_name": str(specification["name"]),
                        "orientation": orientation,
                        "seed": seed,
                        "paired_noise_identity_sha256": noise_identity,
                        "expected_row_count": 3 if orientation == "native" else 2,
                        "forward_count": 1,
                    }
                )
    for identity in reverse_seed_records(config):
        unit_id = (
            f"reverse_trajectory:{int(identity['requested_length'])}:"
            f"{int(identity['sample_index'])}:{int(identity['seed'])}"
        )
        units.append(
            {
                "order": len(units),
                "unit_id": unit_id,
                "unit_kind": "reverse_trajectory",
                "requested_length": int(identity["requested_length"]),
                "sample_index": int(identity["sample_index"]),
                "seed": int(identity["seed"]),
                "orientation": "trajectory",
                "paired_noise_identity_sha256": identity["noise_identity_sha256"],
                "expected_row_count": 2 * len(config["reverse_trajectory"]["milestones"]),
                "forward_count": int(config["diffusion_steps"]),
            }
        )
    recovery = config["recovery"]
    one_step_count = sum(unit["unit_kind"] == "one_step" for unit in units)
    reverse_count = sum(unit["unit_kind"] == "reverse_trajectory" for unit in units)
    if one_step_count != int(recovery["expected_one_step_units"]):
        raise ValueError("E007 Phase-3I.1 one-step work-unit count contradiction")
    if reverse_count != int(recovery["expected_reverse_trajectory_units"]):
        raise ValueError("E007 Phase-3I.1 reverse work-unit count contradiction")
    if len(units) != int(recovery["expected_total_units"]):
        raise ValueError("E007 Phase-3I.1 total work-unit count contradiction")
    if sum(int(unit["forward_count"]) for unit in units) != int(recovery["expected_total_forwards"]):
        raise ValueError("E007 Phase-3I.1 forward-count contract contradiction")
    if len({unit["unit_id"] for unit in units}) != len(units):
        raise ValueError("E007 Phase-3I.1 duplicate work-unit identity")
    return units


def _unit_artifact_relative(unit: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(str(unit["unit_id"]).encode()).hexdigest()[:20]
    return Path("units") / f"{int(unit['order']):04d}-{unit['unit_kind']}-{digest}.parquet"


def _journal_path(staging: Path) -> Path:
    return staging / "block_journal.jsonl"


def _append_journal(path: Path, record: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n") or not line.strip():
                raise ValueError(f"E007 Phase-3I.1 malformed journal line: {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"E007 Phase-3I.1 malformed journal line: {line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"E007 Phase-3I.1 non-object journal line: {line_number}")
            records.append(value)
    return records


def verify_journaled_units(
    staging: Path,
    units: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Verify a strict committed prefix and every referenced Parquet artifact."""
    records = _read_journal(_journal_path(staging))
    if len(records) > len(units):
        raise ValueError("E007 Phase-3I.1 journal contains excess units")
    root = staging.resolve()
    seen_paths: set[str] = set()
    seen_units: set[str] = set()
    for index, record in enumerate(records):
        unit = units[index]
        if int(record.get("sequence", -1)) != index or int(record.get("unit_order", -1)) != index:
            raise ValueError("E007 Phase-3I.1 journal is out of order")
        if record.get("unit_id") != unit["unit_id"] or record.get("unit_kind") != unit["unit_kind"]:
            raise ValueError("E007 Phase-3I.1 journal unit conflicts with the immutable plan")
        if record["unit_id"] in seen_units:
            raise ValueError("E007 Phase-3I.1 journal duplicates a unit")
        seen_units.add(str(record["unit_id"]))
        relative = Path(str(record.get("path", "")))
        artifact = (staging / relative).resolve()
        logical = relative.as_posix()
        if relative.is_absolute() or not artifact.is_relative_to(root):
            raise ValueError(f"E007 Phase-3I.1 journal path escapes staging: {logical}")
        if logical in seen_paths:
            raise ValueError("E007 Phase-3I.1 journal duplicates an artifact path")
        seen_paths.add(logical)
        if not artifact.is_file() or artifact.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"E007 Phase-3I.1 journaled artifact size contradiction: {logical}")
        if sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"E007 Phase-3I.1 journaled artifact hash contradiction: {logical}")
        parquet = pq.ParquetFile(artifact)
        if parquet.metadata.num_rows != int(record.get("row_count", -1)):
            raise ValueError(f"E007 Phase-3I.1 journaled artifact row-count contradiction: {logical}")
        if int(record["row_count"]) != int(unit["expected_row_count"]):
            raise ValueError("E007 Phase-3I.1 journal row count conflicts with its work unit")
        schema_names = parquet.schema_arrow.names
        if schema_names != list(record.get("schema_columns", [])):
            raise ValueError(f"E007 Phase-3I.1 journaled artifact schema contradiction: {logical}")
        required = {
            "unit_id",
            "unit_kind",
            "unit_order",
            "unit_orientation",
            "paired_noise_identity_sha256",
            "representation",
            "requested_length",
            "timestep",
        }
        if unit["unit_kind"] == "one_step":
            required.update({"sample_id", "timestep_name", "noise_seed", "coordinate_v_mse"})
        else:
            required.update({"sample_index", "seed", "noise_identity_sha256"})
        if not required <= set(schema_names):
            raise ValueError(f"E007 Phase-3I.1 unit artifact lacks recovery columns: {logical}")
        identity_rows = pq.read_table(
            artifact,
            columns=[
                "unit_id",
                "unit_kind",
                "unit_order",
                "unit_orientation",
                "paired_noise_identity_sha256",
            ],
        ).to_pylist()
        if any(
            row["unit_id"] != unit["unit_id"]
            or row["unit_kind"] != unit["unit_kind"]
            or int(row["unit_order"]) != index
            or row["unit_orientation"] != unit["orientation"]
            or row["paired_noise_identity_sha256"] != unit["paired_noise_identity_sha256"]
            for row in identity_rows
        ):
            raise ValueError(f"E007 Phase-3I.1 artifact row identity contradiction: {logical}")
        if int(record.get("forward_count", -1)) != int(unit["forward_count"]):
            raise ValueError("E007 Phase-3I.1 journal forward count conflicts with its work unit")
        if record.get("journal_version") != config["recovery"]["journal_version"]:
            raise ValueError("E007 Phase-3I.1 journal version contradiction")
        if record.get("numerics_policy_sha256") != _canonical_sha(config["numerics"]):
            raise ValueError("E007 Phase-3I.1 journal numerical-policy contradiction")
    return records


def cleanup_uncommitted_units(staging: Path, records: Sequence[Mapping[str, Any]]) -> list[str]:
    units_root = (staging / "units").resolve()
    staging_root = staging.resolve()
    if not units_root.is_relative_to(staging_root):
        raise ValueError("E007 Phase-3I.1 units root escapes staging")
    committed = {(staging / str(record["path"])).resolve() for record in records}
    removed = []
    if units_root.exists():
        for path in sorted(units_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(units_root):
                raise ValueError("E007 Phase-3I.1 uncommitted path escapes units root")
            if resolved not in committed:
                removed.append(path.relative_to(staging).as_posix())
                path.unlink()
        _fsync_directory(units_root)
    return removed


def commit_unit_artifact(
    staging: Path,
    unit: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(rows) != int(unit["expected_row_count"]):
        raise ValueError("E007 Phase-3I.1 unit result row-count contradiction")
    enriched = [
        {
            **dict(row),
            "unit_id": unit["unit_id"],
            "unit_kind": unit["unit_kind"],
            "unit_order": int(unit["order"]),
            "unit_orientation": unit["orientation"],
            "paired_noise_identity_sha256": unit["paired_noise_identity_sha256"],
        }
        for row in rows
    ]
    relative = _unit_artifact_relative(unit)
    artifact = staging / relative
    _atomic_parquet(artifact, enriched, config["publication"]["parquet_compression"])
    schema_columns = pq.ParquetFile(artifact).schema_arrow.names
    record = {
        "journal_version": config["recovery"]["journal_version"],
        "sequence": int(unit["order"]),
        "unit_order": int(unit["order"]),
        "unit_id": unit["unit_id"],
        "unit_kind": unit["unit_kind"],
        "path": relative.as_posix(),
        "row_count": len(enriched),
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
        "schema_columns": schema_columns,
        "forward_count": int(unit["forward_count"]),
        "numerics_policy_sha256": _canonical_sha(config.get("numerics", {})),
        "committed_utc": _utc_now(),
    }
    _append_journal(_journal_path(staging), record)
    return record


def progress_from_records(records: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    forwards = sum(int(record["forward_count"]) for record in records)
    total_forwards = sum(int(unit["forward_count"]) for unit in units)
    return {
        "completed_units": len(records),
        "total_units": len(units),
        "completed_forwards": forwards,
        "estimated_total_forwards": total_forwards,
        "forward_progress_percent": 100.0 * forwards / total_forwards if total_forwards else 0.0,
        "completed_one_step_blocks": sum(record["unit_kind"] == "one_step" for record in records),
        "completed_trajectories": sum(record["unit_kind"] == "reverse_trajectory" for record in records),
        "latest_verified_artifact": records[-1]["path"] if records else None,
    }


def native_reference_envelopes(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["phase3i"]["root"]) / "per_sample_metrics.parquet"
    columns = ["source", "requested_length", *ENVELOPE_FIELDS]
    rows = [row for row in pq.read_table(path, columns=columns).to_pylist() if row["source"] == "reference"]
    lower_q, upper_q = map(float, config["metrics"]["native_envelope_quantiles"])
    envelopes: dict[str, Any] = {}
    for length in map(int, config["panel"]["lengths"]):
        subset = [row for row in rows if int(row["requested_length"]) == length]
        if len(subset) != 32:
            raise ValueError(f"E007 Phase-3I.1 frozen Phase-3I reference count contradiction: {length}")
        envelopes[str(length)] = {
            field: {
                "lower": float(np.quantile([float(row[field]) for row in subset], lower_q)),
                "upper": float(np.quantile([float(row[field]) for row in subset], upper_q)),
            }
            for field in ENVELOPE_FIELDS
        }
        envelopes[str(length)]["reference_count"] = len(subset)
    return {
        "source": "immutable_phase3i_per_sample_reference_rows",
        "quantiles": [lower_q, upper_q],
        "by_requested_length": envelopes,
        "sha256": _canonical_sha(envelopes),
    }


def metric_envelope_status(row: Mapping[str, Any], envelope: Mapping[str, Any]) -> dict[str, bool]:
    return {
        field: float(bounds["lower"]) <= float(row[field]) <= float(bounds["upper"])
        for field, bounds in envelope.items()
        if field != "reference_count" and field in row
    }


def trajectory_transitions(records: Sequence[Mapping[str, Any]], envelope: Mapping[str, Any]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: int(row["timestep"]), reverse=True)
    result: dict[str, Any] = {}
    for field in ENVELOPE_FIELDS:
        states = [
            float(envelope[field]["lower"]) <= float(row[field]) <= float(envelope[field]["upper"]) for row in ordered
        ]
        entries = [
            [int(ordered[index - 1]["timestep"]), int(ordered[index]["timestep"])]
            for index in range(1, len(states))
            if not states[index - 1] and states[index]
        ]
        exits = [
            [int(ordered[index - 1]["timestep"]), int(ordered[index]["timestep"])]
            for index in range(1, len(states))
            if states[index - 1] and not states[index]
        ]
        result[field] = {
            "initial_inside": states[0],
            "final_inside": states[-1],
            "first_entry_interval": entries[0] if entries else None,
            "first_exit_interval": exits[0] if exits else None,
        }
    return result


def stable_relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(observed - expected) / denominator)


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def controlled_length_slopes(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    result = {}
    timesteps = sorted({int(row["timestep"]) for row in rows})
    for timestep in timesteps:
        subset = [row for row in rows if int(row["timestep"]) == timestep and row["representation"] == "predicted_x0"]
        lengths = np.asarray([float(row["requested_length"]) for row in subset])
        values = np.asarray([float(row[metric]) for row in subset])
        result[str(timestep)] = {
            "slope_per_residue": float(np.polyfit(lengths, values, 1)[0]) if len(np.unique(lengths)) > 1 else None,
            "pearson_r": pearson_correlation(lengths, values),
            "record_count": len(subset),
        }
    return result


def classify_localization(
    *,
    denoiser_local_failure: bool,
    sampler_accumulation: bool,
    chirality_limitation: bool,
    long_length_failure: bool,
) -> list[str]:
    categories: set[str] = set()
    if denoiser_local_failure and sampler_accumulation:
        categories.add("combined_denoiser_and_sampler_failure")
    else:
        if denoiser_local_failure:
            categories.add("denoiser_local_geometry_failure")
        if sampler_accumulation:
            categories.add("reverse_sampler_error_accumulation")
    if chirality_limitation:
        categories.add("chirality_symmetry_limitation")
    if long_length_failure:
        categories.add("long_length_scaling_failure")
    if not categories:
        categories.add("bounded_diagnostic_inconclusive")
    if not categories <= DECISION_CATEGORIES:
        raise AssertionError("unknown E007 Phase-3I.1 decision category")
    return sorted(categories)


def _memory(device: Any) -> dict[str, float | None]:
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if device.type != "cuda":
        return {"peak_rss_mib": rss, "peak_cuda_allocated_mib": None, "peak_cuda_reserved_mib": None}
    import torch

    return {
        "peak_rss_mib": rss,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }


def _enforce_memory(device: Any, config: Mapping[str, Any]) -> None:
    values = _memory(device)
    limits = config["memory"]
    if float(values["peak_rss_mib"]) > float(limits["maximum_rss_mib"]):
        raise MemoryError(f"E007 Phase-3I.1 RSS limit exceeded: {values}")
    if device.type == "cuda" and (
        float(values["peak_cuda_allocated_mib"]) > float(limits["maximum_cuda_allocated_mib"])
        or float(values["peak_cuda_reserved_mib"]) > float(limits["maximum_cuda_reserved_mib"])
    ):
        raise MemoryError(f"E007 Phase-3I.1 CUDA memory limit exceeded: {values}")


def _load_model(config: Mapping[str, Any], device: Any) -> Any:
    import torch

    from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet

    payload = torch.load(config["selected_checkpoint"]["path"], map_location="cpu", weights_only=False)
    if payload.get("version") != "e007_coordinate_real_continuation_to_10000_v1":
        raise ValueError("E007 Phase-3I.1 checkpoint version contradiction")
    if int(payload.get("optimizer_update", -1)) != 9000:
        raise ValueError("E007 Phase-3I.1 checkpoint update contradiction")
    if payload.get("continuation_configuration_sha256") != config["model_source"]["continuation_config_sha256"]:
        raise ValueError("E007 Phase-3I.1 checkpoint configuration contradiction")
    source = yaml.safe_load(Path(config["model_source"]["phase3f_config_path"]).read_text())
    model = EquivariantPairCoordinateUNet(**source["model"])
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != int(config["model_source"]["expected_parameter_count"]):
        raise ValueError("E007 Phase-3I.1 parameter-count contradiction")
    model.load_state_dict(payload["model"])
    model.requires_grad_(False).eval().to(device)
    return model


def _prepared_reference(config: Mapping[str, Any], canonical_row: Mapping[str, Any]) -> dict[str, Any]:
    from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import prepare_coordinate_batch

    required = {
        "sample_id",
        "sequence_length",
        "coordinates",
        "residue_mask",
        "chain_continuity_mask",
        "accepted_contiguous_single_chain",
    }
    missing = sorted(required - set(canonical_row))
    if missing:
        raise ValueError(f"E007 Phase-3I.1 incomplete canonical collator row: {missing}")
    if canonical_row["accepted_contiguous_single_chain"] is not True:
        raise ValueError("E007 Phase-3I.1 canonical collator row is not accepted")
    return prepare_coordinate_batch(
        [dict(canonical_row)],
        float(config["coordinate_scale_angstrom"]),
        int(config["expected_downsample_factor"]),
    )


def paired_reflection_batch(
    diffusion: Any, clean: Any, mask: Any, timestep: Any, generator: Any, reflection: Any
) -> tuple[Any, Any]:
    """Return paired native/reflected diffusion batches with reflected noise."""
    native = diffusion.make_training_batch(clean, mask, timesteps=timestep, generator=generator)
    reflected_clean = clean @ reflection.T
    reflected_noise = native.coordinate_noise @ reflection.T
    alpha, sigma = diffusion.alpha_sigma(timestep, clean)
    reflected_noisy = alpha * reflected_clean + sigma * reflected_noise
    reflected_target = alpha * reflected_noise - sigma * reflected_clean
    return native, (reflected_clean, reflected_noisy, reflected_target)


def _metric_row(
    coordinates: Any,
    *,
    scale: float,
    requested_length: int,
    sample_id: str,
    source: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    physical = coordinates.detach().cpu().double().numpy() * scale
    row, arrays = phase3i.geometry_metrics(
        physical, source=source, sample_id=sample_id, length=len(physical), config=config
    )
    for name in (
        "distance_i_plus_2",
        "distance_i_plus_3",
        "bond_angle_radians",
        "signed_pseudo_dihedral_radians",
    ):
        values = np.asarray(arrays[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            if name == "bond_angle_radians":
                values = np.degrees(values)
            row[f"{name}_p05"] = float(np.quantile(values, 0.05))
            row[f"{name}_median"] = float(np.median(values))
            row[f"{name}_p95"] = float(np.quantile(values, 0.95))
    vectors = np.diff(physical, axis=0)
    volumes = np.einsum("ij,ij->i", np.cross(vectors[:-2], vectors[1:-1]), vectors[2:])
    row["signed_tetrahedral_volume_p05"] = float(np.quantile(volumes, 0.05))
    row["signed_tetrahedral_volume_median"] = float(np.median(volumes))
    row["signed_tetrahedral_volume_p95"] = float(np.quantile(volumes, 0.95))
    row["requested_length"] = requested_length
    return row


def _one_step_unit_records(
    model: Any,
    diffusion: Any,
    unit: Mapping[str, Any],
    panel_item: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    scale = float(config["coordinate_scale_angstrom"])
    record = panel_item["selection"]
    reflection = torch.tensor(config["one_step"]["reflection_matrix"], dtype=torch.float32, device=device)
    if not torch.allclose(reflection.T @ reflection, torch.eye(3, device=device), atol=1e-7, rtol=0):
        raise ValueError("E007 Phase-3I.1 reflection matrix is not orthogonal")
    prepared = _prepared_reference(config, panel_item["canonical_row"])
    provenance = panel_item["provenance"]
    clean = prepared["coordinates"].to(device)
    mask = prepared["residue_mask"].to(device)
    lengths = prepared["lengths"].to(device)
    continuity = prepared["chain_continuity_mask"].to(device)
    source_adjacent = torch.linalg.vector_norm(clean[:, 1:] - clean[:, :-1], dim=-1)
    timestep = torch.tensor([int(unit["timestep"])], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(int(unit["seed"]))
    native, reflected = paired_reflection_batch(diffusion, clean, mask, timestep, generator, reflection)
    reflected_clean, reflected_noisy, reflected_target = reflected
    valid = mask[0]
    if unit["orientation"] == "native":
        prediction = model(native.noisy_coordinates, timestep, lengths, mask, continuity)["v_prediction"]
        predicted_x0 = diffusion.reconstruct_x0(native.noisy_coordinates, timestep, prediction, mask)
        target = native.coordinate_v_target
        orientation_source = clean
        values_by_name = (
            ("source_x0", clean),
            ("corrupted_xt", native.noisy_coordinates),
            ("predicted_x0", predicted_x0),
        )
    elif unit["orientation"] == "reflected":
        prediction = model(reflected_noisy, timestep, lengths, mask, continuity)["v_prediction"]
        predicted_x0 = diffusion.reconstruct_x0(reflected_noisy, timestep, prediction, mask)
        target = reflected_target
        orientation_source = reflected_clean
        values_by_name = (
            ("reflected_source_x0", reflected_clean),
            ("reflected_predicted_x0", predicted_x0),
        )
    else:
        raise ValueError(f"E007 Phase-3I.1 invalid one-step orientation: {unit['orientation']}")
    v_mse = (prediction[0, valid] - target[0, valid]).square().mean()
    rows = []
    for name, values in values_by_name:
        row = _metric_row(
            values[0, valid],
            scale=scale,
            requested_length=int(record["target_length"]),
            sample_id=str(record["sample_id"]),
            source=name,
            config=config,
        )
        row.update(
            {
                "representation": name,
                "timestep_name": unit["timestep_name"],
                "timestep": int(unit["timestep"]),
                "noise_seed": int(unit["seed"]),
                "coordinate_v_mse": float(v_mse.detach().cpu()),
                "coordinate_rmse_to_source_angstrom": float(
                    ((values[0, valid] - orientation_source[0, valid]).square().mean().sqrt() * scale).detach().cpu()
                ),
                "adjacent_distance_rmse_to_native_angstrom": float(
                    (
                        (torch.linalg.vector_norm(values[:, 1:] - values[:, :-1], dim=-1) - source_adjacent)
                        .square()
                        .mean()
                        .sqrt()
                        * scale
                    )
                    .detach()
                    .cpu()
                ),
            }
        )
        rows.append(row)
    _enforce_memory(device, config)
    return rows, provenance


def _reverse_unit_records(
    model: Any,
    diffusion: Any,
    unit: Mapping[str, Any],
    config: Mapping[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise

    scale = float(config["coordinate_scale_angstrom"])
    milestones = set(map(int, config["reverse_trajectory"]["milestones"]))
    rows = []
    length, seed = int(unit["requested_length"]), int(unit["seed"])
    generator = torch.Generator(device=device).manual_seed(seed)
    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    continuity = torch.ones((1, length - 1), dtype=torch.bool, device=device)
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    coordinates = centered_coordinate_noise(torch.empty((1, length, 3), device=device), mask, generator=generator)
    for step in range(diffusion.timesteps - 1, -1, -1):
        timestep = torch.tensor([step], dtype=torch.long, device=device)
        prediction = model(coordinates, timestep, lengths, mask, continuity)["v_prediction"]
        coordinates, predicted_x0, _ = diffusion.deterministic_reverse_step(coordinates, timestep, prediction, mask)
        if step in milestones:
            for representation, values in (("predicted_x0", predicted_x0), ("reverse_state", coordinates)):
                row = _metric_row(
                    values[0],
                    scale=scale,
                    requested_length=length,
                    sample_id=f"reverse_N{length}_i{unit['sample_index']}",
                    source=representation,
                    config=config,
                )
                row.update(
                    {
                        "requested_length": length,
                        "sample_index": int(unit["sample_index"]),
                        "seed": seed,
                        "noise_identity_sha256": unit["paired_noise_identity_sha256"],
                        "representation": representation,
                        "timestep": step,
                    }
                )
                rows.append(row)
    _enforce_memory(device, config)
    return rows


def _summarize(
    one_step: Sequence[Mapping[str, Any]],
    trajectory: Sequence[Mapping[str, Any]],
    envelopes: Mapping[str, Any],
) -> dict[str, Any]:
    predicted = [row for row in one_step if row["representation"] == "predicted_x0"]
    low_medium = [row for row in predicted if row["timestep_name"] in {"low", "medium"}]
    final_states = [row for row in trajectory if row["representation"] == "reverse_state" and int(row["timestep"]) == 0]
    domains = {
        "local_geometry": (
            "adjacent_distance_rmse_to_3_8_angstrom",
            "discontinuity_fraction",
            "distance_i_plus_2_mean_angstrom",
            "distance_i_plus_3_mean_angstrom",
            "bond_angle_mean_degrees",
        ),
        "global_topology": (
            "radius_of_gyration_angstrom",
            "contact_density_8a",
            "contact_order_8a",
            "contact_graph_component_count_8a",
            "contact_graph_largest_component_fraction_8a",
            "asphericity",
        ),
        "pseudo_chirality": (
            "signed_pseudo_dihedral_mean",
            "signed_pseudo_dihedral_positive_fraction",
            "signed_tetrahedral_volume_mean",
        ),
    }

    def outside_fraction(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> float | None:
        outside = [
            not all(
                metric_envelope_status(row, envelopes["by_requested_length"][str(row["requested_length"])])[field]
                for field in fields
            )
            for row in rows
        ]
        return float(np.mean(outside)) if outside else None

    denoiser_domains = {name: outside_fraction(low_medium, fields) for name, fields in domains.items()}
    sampler_domains = {name: outside_fraction(final_states, fields) for name, fields in domains.items()}
    slopes = controlled_length_slopes(one_step, "adjacent_distance_rmse_to_native_angstrom")
    correlations = {
        field: pearson_correlation(
            [float(row["coordinate_v_mse"]) for row in predicted],
            [float(row[field]) for row in predicted],
        )
        for field in (
            "coordinate_rmse_to_source_angstrom",
            "adjacent_distance_rmse_to_native_angstrom",
            "discontinuity_fraction",
            "clash_fraction",
            "radius_of_gyration_angstrom",
            "contact_order_8a",
        )
    }
    reflected = [row for row in one_step if row["representation"] == "reflected_predicted_x0"]
    reflected_by_key = {(row["sample_id"], int(row["timestep"])): row for row in reflected}
    reflection_proxy_residuals = []
    for row in predicted:
        counterpart = reflected_by_key[(row["sample_id"], int(row["timestep"]))]
        reflection_proxy_residuals.extend(
            (
                abs(float(row["signed_pseudo_dihedral_mean"]) + float(counterpart["signed_pseudo_dihedral_mean"])),
                abs(float(row["signed_tetrahedral_volume_mean"]) + float(counterpart["signed_tetrahedral_volume_mean"]))
                / max(abs(float(row["signed_tetrahedral_volume_mean"])), 1e-12),
                abs(
                    float(row["signed_pseudo_dihedral_positive_fraction"])
                    + float(counterpart["signed_pseudo_dihedral_positive_fraction"])
                    - 1.0
                ),
            )
        )
    source_by_key = {
        (row["sample_id"], int(row["timestep"])): row for row in one_step if row["representation"] == "source_x0"
    }
    source_chirality = [
        float(source_by_key[(row["sample_id"], int(row["timestep"]))]["signed_pseudo_dihedral_mean"])
        for row in predicted
    ]
    predicted_chirality = [float(row["signed_pseudo_dihedral_mean"]) for row in predicted]
    preservation_correlation = pearson_correlation(source_chirality, predicted_chirality)
    attenuation = float(
        np.mean(np.abs(predicted_chirality)) / max(np.mean(np.abs(source_chirality)), np.finfo(float).eps)
    )
    if attenuation < 0.25:
        chirality_behavior = "handedness_signal_erased_or_strongly_attenuated"
    elif preservation_correlation is not None and preservation_correlation >= 0.5:
        chirality_behavior = "input_pseudo_handedness_preserved_equivariantly"
    else:
        chirality_behavior = "reflection_symmetric_without_clear_native_preference"
    denoiser_failure = any(value is not None and value > 0.5 for value in denoiser_domains.values())
    sampler_failure = any(value is not None and value > 0.5 for value in sampler_domains.values())
    chirality = True
    positive_slopes = sum(
        item["slope_per_residue"] is not None and item["slope_per_residue"] > 0 for item in slopes.values()
    )
    long_length = positive_slopes >= 3
    return {
        "decision_categories": classify_localization(
            denoiser_local_failure=denoiser_failure,
            sampler_accumulation=sampler_failure,
            chirality_limitation=chirality,
            long_length_failure=long_length,
        ),
        "denoiser_outside_native_envelope_fraction_by_domain": denoiser_domains,
        "final_sampler_outside_native_envelope_fraction_by_domain": sampler_domains,
        "coordinate_v_mse_structural_metric_correlations": correlations,
        "length_slopes_controlled_by_timestep": slopes,
        "reflection_proxy_residual_maximum": max(reflection_proxy_residuals) if reflection_proxy_residuals else None,
        "pseudo_chirality_behavior": chirality_behavior,
        "pseudo_chirality_source_prediction_correlation": preservation_correlation,
        "pseudo_chirality_absolute_signal_ratio": attenuation,
        "chirality_interpretation": (
            "The model is O(3)-equivariant. Native handedness is therefore not identifiable from the declared "
            "coordinate objective alone; mirrored inputs are an architectural symmetry control, not native residue "
            "chirality."
        ),
        "no_scalar_score": True,
    }


def plan_denoiser_sampler_localization(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3I.1 output exists: {output} or {staging}")
    prerequisites = verify_prerequisites(config, full=False)
    return {
        "status": "planned_read_only_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "output_dir": str(output),
        "reference_panel_records": 40,
        "references_per_requested_length": 8,
        "reverse_seed_records": 20,
        "reverse_seeds_per_requested_length": 4,
        "one_step_units": config["recovery"]["expected_one_step_units"],
        "reverse_trajectory_units": config["recovery"]["expected_reverse_trajectory_units"],
        "total_units": config["recovery"]["expected_total_units"],
        "one_step_timesteps": config["one_step"]["timesteps"],
        "reverse_milestones": config["reverse_trajectory"]["milestones"],
        "estimated_model_forward_batches": config["runtime"]["estimated_model_forward_batches"],
        "maximum_wall_seconds": config["runtime"]["maximum_wall_seconds"],
        "memory_limits": config["memory"],
        "dataset_scan_performed": False,
        "model_created": False,
        "cuda_initialized": False,
        "output_created": False,
        "prerequisite_hashes": prerequisites["hashes"],
        **NON_AUTHORIZING,
    }


def _run_contract(
    config_path: Path,
    config: Mapping[str, Any],
    prerequisites: Mapping[str, Any],
    panel_diagnostics: Mapping[str, Any],
    canonical_panel_diagnostics: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
        "protected_hashes": prerequisites["hashes"],
        "panel_sha256": panel_diagnostics["record_sha256"],
        "canonical_panel_identity_sha256": canonical_panel_diagnostics["identity_sha256"],
        "work_unit_identity_sha256": _canonical_sha(
            [
                {
                    key: unit[key]
                    for key in (
                        "order",
                        "unit_id",
                        "unit_kind",
                        "orientation",
                        "paired_noise_identity_sha256",
                        "expected_row_count",
                        "forward_count",
                    )
                }
                for unit in units
            ]
        ),
        "expected_total_units": len(units),
        "expected_total_forwards": sum(int(unit["forward_count"]) for unit in units),
    }


def _load_committed_rows(
    staging: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    one_step: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for record in records:
        rows = pq.read_table(staging / str(record["path"])).to_pylist()
        target = one_step if record["unit_kind"] == "one_step" else trajectory
        target.extend(rows)
    return one_step, trajectory


def _heartbeat_payload(
    *,
    status: str,
    records: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    current_unit: Mapping[str, Any] | None,
    execution: Mapping[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    current = current_unit or {}
    return {
        "status": status,
        "updated_utc": _utc_now(),
        **progress_from_records(records, units),
        "current_phase": current.get("unit_kind", "finalization" if len(records) == len(units) else "initialization"),
        "current_unit_id": current.get("unit_id"),
        "current_sample": current.get("sample_id"),
        "current_timestep": current.get("timestep"),
        "current_trajectory": (
            {"requested_length": current["requested_length"], "seed": current["seed"]}
            if current.get("unit_kind") == "reverse_trajectory"
            else None
        ),
        **NON_AUTHORIZING,
        **execution,
        **extra,
    }


def monitor_denoiser_sampler_localization(config_path: str | Path) -> dict[str, Any]:
    """Read the latest heartbeat without validating payloads or writing state."""
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    candidates = [("final", output / "heartbeat.json"), ("staging", staging / "heartbeat.json")]
    for source, path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text())
            return {"source": source, "heartbeat_path": str(path), **payload}
    raise FileNotFoundError("E007 Phase-3I.1 has no staging or final heartbeat")


def diagnose_denoiser_sampler_localization(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    import torch

    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    config_path = Path(config_path)
    config = load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists():
        raise FileExistsError(f"E007 Phase-3I.1 completed output exists: {output}")
    if resume:
        if not staging.is_dir():
            raise FileNotFoundError(f"E007 Phase-3I.1 resume staging is absent: {staging}")
    elif staging.exists():
        raise FileExistsError(f"E007 Phase-3I.1 staging exists; use --resume: {staging}")
    prerequisites_before = verify_prerequisites(config, full=True)
    panel, panel_diagnostics = select_validation_panel(config)
    canonical_panel, canonical_panel_diagnostics = reconstruct_authoritative_panel(config, panel)
    units = build_work_units(panel, config)
    envelopes = native_reference_envelopes(config)
    contract = _run_contract(
        config_path,
        config,
        prerequisites_before,
        panel_diagnostics,
        canonical_panel_diagnostics,
        units,
    )
    if not resume:
        staging.mkdir(parents=True)
        _fsync_directory(staging.parent)
    heartbeat = staging / "heartbeat.json"
    started = time.monotonic()
    execution = {
        "model_created": False,
        "cuda_initialized": False,
        "forward_performed": False,
        "sampling_performed": False,
    }
    records: list[dict[str, Any]] = []
    try:
        contract_path = staging / "run_contract.json"
        if resume:
            if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
                raise ValueError("E007 Phase-3I.1 resume run-contract contradiction")
            records = verify_journaled_units(staging, units, config)
            cleanup_uncommitted_units(staging, records)
            execution.update(
                {
                    "model_created": bool(records),
                    "forward_performed": bool(records),
                    "sampling_performed": any(record["unit_kind"] == "reverse_trajectory" for record in records),
                }
            )
        else:
            _atomic_parquet(staging / "panel_manifest.parquet", panel, config["publication"]["parquet_compression"])
            _atomic_json(staging / "panel_selection.json", panel_diagnostics)
            _atomic_json(staging / "canonical_panel_validation.json", canonical_panel_diagnostics)
            _atomic_json(staging / "reference_envelopes.json", envelopes)
            _atomic_json(contract_path, contract)
        _atomic_json(
            heartbeat,
            _heartbeat_payload(
                status="resuming" if resume else "initializing",
                records=records,
                units=units,
                current_unit=units[len(records)] if len(records) < len(units) else None,
                execution=execution,
                resumed=resume,
            ),
        )
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3I.1 configured diagnostic requires CUDA")
        device = torch.device("cuda")
        execution["cuda_initialized"] = True
        torch.cuda.reset_peak_memory_stats(device)
        backend: dict[str, Any] = {"not_entered_all_units_already_committed": True}
        if len(records) < len(units):
            model = _load_model(config, device)
            execution["model_created"] = True
            diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
            panel_by_index = {index: row for index, row in enumerate(canonical_panel)}
            with coordinate_model_execution_context(config["numerics"], device) as backend:
                with torch.no_grad():
                    for unit in units[len(records) :]:
                        _atomic_json(
                            heartbeat,
                            _heartbeat_payload(
                                status="running",
                                records=records,
                                units=units,
                                current_unit=unit,
                                execution=execution,
                                resumed=resume,
                            ),
                        )
                        if unit["unit_kind"] == "one_step":
                            rows, _provenance = _one_step_unit_records(
                                model,
                                diffusion,
                                unit,
                                panel_by_index[int(unit["panel_index"])],
                                config,
                                device,
                            )
                        else:
                            rows = _reverse_unit_records(model, diffusion, unit, config, device)
                            execution["sampling_performed"] = True
                        execution["forward_performed"] = True
                        committed = commit_unit_artifact(staging, unit, rows, config)
                        records.append(committed)
                        _atomic_json(
                            heartbeat,
                            _heartbeat_payload(
                                status="running",
                                records=records,
                                units=units,
                                current_unit=units[len(records)] if len(records) < len(units) else None,
                                execution=execution,
                                resumed=resume,
                            ),
                        )
            if not backend["restored"]:
                raise RuntimeError("E007 Phase-3I.1 numerical backend was not restored")
        records = verify_journaled_units(staging, units, config)
        if len(records) != int(config["recovery"]["expected_total_units"]):
            raise ValueError("E007 Phase-3I.1 finalization refused an incomplete journal")
        completed_forwards = sum(int(record["forward_count"]) for record in records)
        if completed_forwards != int(config["recovery"]["expected_total_forwards"]):
            raise ValueError("E007 Phase-3I.1 final forward-count conservation failed")
        one_step, trajectory = _load_committed_rows(staging, records)
        if len(one_step) != int(config["runtime"]["estimated_one_step_records"]):
            raise ValueError("E007 Phase-3I.1 one-step row-count conservation failed")
        if len(trajectory) != int(config["runtime"]["estimated_reverse_milestone_records"]):
            raise ValueError("E007 Phase-3I.1 trajectory row-count conservation failed")
        if len({row["unit_id"] for row in one_step}) != int(config["recovery"]["expected_one_step_units"]):
            raise ValueError("E007 Phase-3I.1 one-step unit conservation failed")
        if len({row["unit_id"] for row in trajectory}) != int(config["recovery"]["expected_reverse_trajectory_units"]):
            raise ValueError("E007 Phase-3I.1 trajectory unit conservation failed")
        transitions = {}
        for identity in reverse_seed_records(config):
            key = f"{identity['requested_length']}:{identity['sample_index']}"
            subset = [
                row
                for row in trajectory
                if int(row["requested_length"]) == int(identity["requested_length"])
                and int(row["sample_index"]) == int(identity["sample_index"])
                and row["representation"] == "reverse_state"
            ]
            transitions[key] = trajectory_transitions(
                subset, envelopes["by_requested_length"][str(identity["requested_length"])]
            )
        summary = _summarize(one_step, trajectory, envelopes)
        _atomic_parquet(staging / "one_step_metrics.parquet", one_step, config["publication"]["parquet_compression"])
        _atomic_parquet(
            staging / "reverse_trajectory_metrics.parquet", trajectory, config["publication"]["parquet_compression"]
        )
        _atomic_json(staging / "trajectory_transitions.json", transitions)
        provenance = [item["provenance"] for item in canonical_panel]
        _atomic_json(staging / "protected_reference_provenance.json", provenance)
        prerequisites_after = verify_prerequisites(config, full=True)
        if prerequisites_after["hashes"] != prerequisites_before["hashes"]:
            raise ValueError("E007 Phase-3I.1 protected inputs changed")
        report = {
            "status": "completed_read_only_non_authorizing",
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "selected_checkpoint": config["selected_checkpoint"],
            "panel": panel_diagnostics,
            "canonical_panel_validation": canonical_panel_diagnostics,
            "one_step_record_count": len(one_step),
            "reverse_trajectory_record_count": len(trajectory),
            "reverse_seed_count": len(reverse_seed_records(config)),
            "completed_units": len(records),
            "completed_forwards": completed_forwards,
            "summary": summary,
            "memory": _memory(device),
            "backend": backend,
            "elapsed_seconds": time.monotonic() - started,
            "protected_inputs_unchanged": True,
            **execution,
            **{key: value for key, value in NON_AUTHORIZING.items() if key not in {"model_created"}},
        }
        report["model_created"] = execution["model_created"]
        report["sampling_performed"] = execution["sampling_performed"]
        _atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": report["configuration_sha256"],
            "report_sha256": sha256_file(staging / "report.json"),
            "checkpoint_sha256": config["selected_checkpoint"]["sha256"],
            "phase3i_report_sha256": config["phase3i"]["report_sha256"],
            "panel_sha256": panel_diagnostics["record_sha256"],
            "protected_inputs_unchanged": True,
            **execution,
            **NON_AUTHORIZING,
        }
        protocol["model_created"] = execution["model_created"]
        protocol["sampling_performed"] = execution["sampling_performed"]
        _atomic_json(staging / "protocol.json", protocol)
        artifacts = [
            {
                "path": path.relative_to(staging).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name not in {"artifact_inventory.json", "heartbeat.json"}
        ]
        _atomic_json(
            staging / "artifact_inventory.json", {"artifacts": artifacts, "aggregate_sha256": _canonical_sha(artifacts)}
        )
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "completed_utc": _utc_now(),
                "report_sha256": protocol["report_sha256"],
                **progress_from_records(records, units),
                **NON_AUTHORIZING,
                **execution,
            },
        )
        staging.replace(output)
        _fsync_directory(output.parent)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        _atomic_json(
            heartbeat,
            _heartbeat_payload(
                status=status,
                records=records,
                units=units,
                current_unit=units[len(records)] if units and len(records) < len(units) else None,
                execution=execution,
                error_type=type(error).__name__,
                error_message=str(error),
                recovery_boundary_verified=bool(records),
            ),
        )
        raise


__all__ = [
    "DECISION_CATEGORIES",
    "NON_AUTHORIZING",
    "classify_localization",
    "build_work_units",
    "cleanup_uncommitted_units",
    "commit_unit_artifact",
    "controlled_length_slopes",
    "diagnose_denoiser_sampler_localization",
    "load_config",
    "metric_envelope_status",
    "monitor_denoiser_sampler_localization",
    "native_reference_envelopes",
    "paired_reflection_batch",
    "pearson_correlation",
    "plan_denoiser_sampler_localization",
    "reverse_seed_records",
    "select_validation_panel",
    "trajectory_transitions",
    "validate_panel_schema",
    "verify_journaled_units",
]
