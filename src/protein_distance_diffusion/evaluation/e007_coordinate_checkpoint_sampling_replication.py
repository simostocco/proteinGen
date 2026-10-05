"""Independent checkpoint sampling replication for E007 Phase 3H."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.evaluation import e007_coordinate_sampling_replication as phase3g
from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_plan import sha256_file

VERSION = "e007_coordinate_checkpoint_sampling_replication_v1"
NON_AUTHORIZING = dict(phase3g.NON_AUTHORIZING)
NUMERIC_FIELDS = (*phase3g.NUMERIC_FIELDS, "rank3_residual_energy_fraction")
PARETO_FIELDS = (
    "adjacent_reference_error_mean",
    "radius_relative_reference_error_mean",
    "clash_reference_excess_mean",
    "contact_density_reference_error_mean",
    "rank3_residual_energy_fraction_mean",
    "adjacent_failure_fraction",
)
PAIRED_DECISION_FIELDS = (
    "adjacent_reference_error_angstrom",
    "radius_of_gyration_relative_reference_error",
    "non_neighbor_clash_reference_excess",
    "contact_density_6a_reference_error",
    "contact_density_8a_reference_error",
    "contact_density_10a_reference_error",
    "rank3_residual_energy_fraction",
)
PAIRWISE_COMPARISONS = ((9000, 7500), (10000, 7500), (10000, 9000))


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("E007 Phase-3H configuration version contradiction")
    if payload.get("lengths") != [64, 128, 256, 384, 500]:
        raise ValueError("E007 Phase-3H sampling lengths changed")
    if int(payload.get("samples_per_length", 0)) != 32:
        raise ValueError("E007 Phase-3H requires 32 samples per checkpoint/length block")
    if [int(item["optimizer_update"]) for item in payload.get("checkpoints", [])] != [7500, 9000, 10000]:
        raise ValueError("E007 Phase-3H checkpoint set changed")
    if float(payload.get("coordinate_scale_angstrom", 0)) != phase3g.EXPECTED_SCALE:
        raise ValueError("E007 Phase-3H coordinate scale changed")
    if int(payload.get("expected_parameter_count", 0)) != phase3g.EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3H parameter-count contract changed")
    if float(payload["metrics"].get("adjacent_error_limit_angstrom", -1)) != 1.0:
        raise ValueError("E007 Phase-3H original adjacent-error gate changed")
    records = paired_seed_records(payload)
    observed = {int(record["seed"]) for record in records}
    for prior in payload.get("known_prior_sampling_seed_ranges", []):
        previous = set(range(int(prior["minimum"]), int(prior["maximum"]) + 1))
        if observed & previous:
            raise ValueError(f"E007 Phase-3H seed namespace overlaps {prior['namespace']}")
    return payload


def paired_seed_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    base = int(config["sampling_seed_base"])
    namespace = str(config["sampling_seed_namespace"])
    for length_index, length in enumerate(map(int, config["lengths"])):
        for sample_index in range(int(config["samples_per_length"])):
            seed = base + length_index * 100_000 + sample_index
            records.append(
                {
                    "length": length,
                    "sample_index": sample_index,
                    "seed": seed,
                    "noise_identity_sha256": phase3g._canonical_sha(
                        {
                            "namespace": namespace,
                            "seed": seed,
                            "length": length,
                            "shape": [1, length, 3],
                            "noise_algorithm": "centered_coordinate_noise_torch_generator_v1",
                        }
                    ),
                    "reverse_draw_provenance_sha256": phase3g._canonical_sha(
                        {
                            "sampler": "deterministic_ddim",
                            "diffusion_steps": int(config["diffusion_steps"]),
                            "stochastic_reverse_draws": 0,
                        }
                    ),
                }
            )
    return records


def _verify_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"E007 Phase-3H prerequisite is absent: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"E007 Phase-3H prerequisite hash contradiction: {label}")
    return observed


def canonical_directory_fingerprint(directory: Path) -> str:
    """Hash a completed directory independently of its checkout location."""
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        digest.update(f"{relative}\0{path.stat().st_size}\0{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def verify_phase3g_artifact_inventory(source: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    inventory_path = source / "artifact_inventory.json"
    _verify_file(inventory_path, record["artifact_inventory_sha256"], "phase3g_artifact_inventory")
    payload = json.loads(inventory_path.read_text())
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("E007 Phase-3H Phase-3G artifact inventory is malformed")
    if len(artifacts) != int(record["artifact_inventory_entry_count"]):
        raise ValueError("E007 Phase-3H Phase-3G artifact inventory count contradiction")
    if payload.get("aggregate_sha256") != record["artifact_inventory_aggregate_sha256"]:
        raise ValueError("E007 Phase-3H Phase-3G artifact inventory aggregate contradiction")
    if phase3g._canonical_sha(artifacts) != payload["aggregate_sha256"]:
        raise ValueError("E007 Phase-3H Phase-3G artifact inventory serialization contradiction")
    excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
    expected_paths = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name
    }
    observed_paths: set[str] = set()
    root = source.resolve()
    for artifact in artifacts:
        relative = Path(str(artifact.get("path", "")))
        path = (source / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"E007 Phase-3H Phase-3G inventory path escapes source: {relative}")
        logical_path = relative.as_posix()
        if logical_path in observed_paths:
            raise ValueError(f"E007 Phase-3H Phase-3G inventory duplicates a path: {logical_path}")
        observed_paths.add(logical_path)
        if not path.is_file():
            raise FileNotFoundError(f"E007 Phase-3H Phase-3G durable artifact is absent: {logical_path}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"E007 Phase-3H Phase-3G artifact size contradiction: {logical_path}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"E007 Phase-3H Phase-3G artifact hash contradiction: {logical_path}")
    if observed_paths != expected_paths:
        raise ValueError("E007 Phase-3H Phase-3G durable artifact membership contradiction")
    return {
        "inventory_sha256": record["artifact_inventory_sha256"],
        "inventory_aggregate_sha256": payload["aggregate_sha256"],
        "entry_count": len(artifacts),
        "all_entries_verified": True,
    }


def verify_prerequisites(config: Mapping[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for label in ("continuation", "phase3f", "phase3g"):
        record = config[label]
        source = Path(record["source_dir"])
        hashes[f"{label}_report"] = _verify_file(source / "report.json", record["report_sha256"], f"{label}_report")
        hashes[f"{label}_protocol"] = _verify_file(
            source / "protocol.json", record["protocol_sha256"], f"{label}_protocol"
        )
        hashes[f"{label}_config"] = _verify_file(
            Path(record["config_path"]), record["config_sha256"], f"{label}_config"
        )
        if record.get("fingerprint_algorithm") == "relative_path_nul_size_nul_sha256_newline_v1":
            fingerprint = canonical_directory_fingerprint(source)
        else:
            fingerprint = phase3g.directory_fingerprint(source)
        if fingerprint != record["aggregate_fingerprint"]:
            raise ValueError(f"E007 Phase-3H {label} aggregate fingerprint contradiction")
        hashes[f"{label}_aggregate_fingerprint"] = fingerprint
        if "completed_file_count" in record:
            observed_count = sum(item.is_file() for item in source.rglob("*"))
            if observed_count != int(record["completed_file_count"]):
                raise ValueError(f"E007 Phase-3H {label} completed-file count contradiction")

    continuation = config["continuation"]
    report = json.loads((Path(continuation["source_dir"]) / "report.json").read_text())
    protocol = json.loads((Path(continuation["source_dir"]) / "protocol.json").read_text())
    if report.get("global_optimizer_update") != 10000 or protocol.get("global_optimizer_update") != 10000:
        raise ValueError("E007 Phase-3H continuation completion contradiction")
    if report.get("parameter_count") != phase3g.EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3H continuation parameter-count contradiction")
    if any(bool(protocol.get(field)) for field in NON_AUTHORIZING if field.startswith("authorizes_")):
        raise ValueError("E007 Phase-3H source protocol unexpectedly authorizes production")

    for item in config["checkpoints"]:
        update = int(item["optimizer_update"])
        hashes[f"checkpoint_{update}"] = _verify_file(Path(item["path"]), item["sha256"], f"checkpoint_{update}")
        metadata_path = Path(item["metadata_path"])
        hashes[f"checkpoint_metadata_{update}"] = _verify_file(
            metadata_path, item["metadata_sha256"], f"checkpoint_metadata_{update}"
        )
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("optimizer_update", -1)) != update:
            raise ValueError(f"E007 Phase-3H checkpoint metadata update contradiction: {update}")
        if metadata.get("sha256") not in (None, item["sha256"]):
            raise ValueError(f"E007 Phase-3H checkpoint metadata payload hash contradiction: {update}")
    inventory = verify_phase3g_artifact_inventory(Path(config["phase3g"]["source_dir"]), config["phase3g"])
    return {"hashes": hashes, "phase3g_artifact_inventory": inventory, "protected_inputs_verified": True}


def plan_checkpoint_sampling_replication(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    prerequisites = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 Phase-3H output exists: {output} or {staging}")
    seeds = paired_seed_records(config)
    return {
        "status": "planned_non_authorizing",
        "version": VERSION,
        "configuration_sha256": sha256_file(config_path),
        "output_dir": str(output),
        "checkpoints": [int(item["optimizer_update"]) for item in config["checkpoints"]],
        "lengths": list(map(int, config["lengths"])),
        "samples_per_checkpoint_length": int(config["samples_per_length"]),
        "block_count": 15,
        "total_samples": 480,
        "paired_noise_identity_count": 160,
        "paired_seed_manifest_sha256": phase3g._canonical_sha(seeds),
        "independent_seed_namespace": config["sampling_seed_namespace"],
        "reference_population": "identity_30_clean_validation_only",
        "checkpoint_length_block_execution": True,
        "resume_boundary": "verified completed checkpoint-length block",
        "model_created": False,
        "cuda_touched": False,
        "coordinate_payloads_scanned": False,
        "coordinate_samples_generated": False,
        "output_created": False,
        "prerequisite_hashes": prerequisites,
        **NON_AUTHORIZING,
    }


def _rank3_residual(distances: torch.Tensor) -> float:
    squared = distances.double().square()
    gram = -0.5 * (squared - squared.mean(0, keepdim=True) - squared.mean(1, keepdim=True) + squared.mean())
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    energy = eigenvalues.square()
    return float(energy[3:].sum() / energy.sum().clamp_min(1e-24))


def coordinate_metrics(
    coordinates: torch.Tensor,
    *,
    checkpoint: int,
    length: int,
    sample_index: int,
    seed: int,
    reference: Mapping[str, float],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    row, rigid = phase3g.coordinate_metrics(
        coordinates,
        checkpoint=checkpoint,
        length=length,
        sample_index=sample_index,
        seed=seed,
        reference=reference,
        config=config,
    )
    physical = coordinates[0].detach().double() * float(config["coordinate_scale_angstrom"])
    distances = coordinates_to_distance_matrix(physical, diagnostic_float64=True)
    row["rank3_residual_energy_fraction"] = _rank3_residual(distances)
    return row, rigid


def _bootstrap_ci(values: Sequence[float], *, seed: int, replicates: int) -> list[float]:
    return phase3g._bootstrap_ci(values, seed=seed, replicates=replicates)


def descriptive_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise ValueError("E007 Phase-3H cannot summarize an empty group")
    result: dict[str, Any] = {"count": len(rows), "metrics": {}}
    base = int(config["aggregation"]["bootstrap_seed"])
    replicates = int(config["aggregation"]["bootstrap_replicates"])
    for offset, field in enumerate(NUMERIC_FIELDS):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        result["metrics"][field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "standard_deviation": float(values.std()),
            "quantile_05": float(np.quantile(values, 0.05)),
            "quantile_25": float(np.quantile(values, 0.25)),
            "quantile_75": float(np.quantile(values, 0.75)),
            "quantile_95": float(np.quantile(values, 0.95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean_bootstrap_ci_95": _bootstrap_ci(values, seed=base + offset, replicates=replicates),
        }
    gate_failures = _all_record_gate_failures(rows, config)
    failures = gate_failures["adjacent_reference_error"]
    result.update(
        {
            "finite_coordinate_fraction": float(np.mean([bool(row["finite_coordinates"]) for row in rows])),
            "adjacent_all_record_gate_pass": not failures,
            "adjacent_pass_fraction": 1.0 - len(failures) / len(rows),
            "adjacent_failure_count": len(failures),
            "failure_examples": [
                {
                    "checkpoint_update": int(row["checkpoint_update"]),
                    "length": int(row["length"]),
                    "sample_index": int(row["sample_index"]),
                    "seed": int(row["seed"]),
                    "adjacent_reference_error_angstrom": float(row["adjacent_reference_error_angstrom"]),
                }
                for row in sorted(
                    failures,
                    key=lambda item: (-float(item["adjacent_reference_error_angstrom"]), int(item["seed"])),
                )[: int(config["publication"]["maximum_failure_examples"])]
            ],
            "one_record_removal": {
                "underlying_failure_count": len(failures),
                "would_pass_after_removing_one_record": len(failures) <= 1,
                "descriptive_only": True,
            },
            "all_record_hard_gates": {
                name: {
                    "pass": not records,
                    "failure_count": len(records),
                    "pass_fraction": 1.0 - len(records) / len(rows),
                    "one_record_removal_would_pass": len(records) <= 1,
                    "failing_records": [
                        {
                            "checkpoint_update": int(row["checkpoint_update"]),
                            "length": int(row["length"]),
                            "sample_index": int(row["sample_index"]),
                            "seed": int(row["seed"]),
                        }
                        for row in records[: int(config["publication"]["maximum_failure_examples"])]
                    ],
                }
                for name, records in gate_failures.items()
            },
        }
    )
    return result


def _all_record_gate_failures(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, list[Mapping[str, Any]]]:
    metrics = config["metrics"]
    predicates = {
        "finite_coordinates": lambda row: bool(row["finite_coordinates"]),
        "deterministic_replay": lambda row: bool(row["deterministic_replay"]),
        "centering": lambda row: (
            float(row["centroid_max_abs_angstrom"]) <= float(metrics["centroid_max_abs_tolerance_angstrom"])
        ),
        "distance_symmetry": lambda row: (
            float(row["distance_symmetry_error_angstrom"]) <= float(metrics["distance_symmetry_tolerance_angstrom"])
        ),
        "distance_exact_zero_diagonal": lambda row: float(row["distance_diagonal_error_angstrom"]) == 0.0,
        "triangle": lambda row: (
            float(row["sampled_maximum_triangle_violation_angstrom"]) <= float(metrics["triangle_tolerance_angstrom"])
        ),
        "gram_negative_eigenmass": lambda row: (
            float(row["centered_gram_negative_eigenmass_fraction"])
            <= float(metrics["gram_negative_eigenmass_tolerance"])
        ),
        "adjacent_reference_error": lambda row: bool(row["adjacent_original_gate_pass"]),
    }
    return {name: [row for row in rows if not predicate(row)] for name, predicate in predicates.items()}


def validate_count_conservation(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, int]:
    expected = {
        (int(checkpoint["optimizer_update"]), int(length)): int(config["samples_per_length"])
        for checkpoint in config["checkpoints"]
        for length in config["lengths"]
    }
    observed = Counter((int(row["checkpoint_update"]), int(row["length"])) for row in rows)
    if observed != expected:
        raise ValueError(f"E007 Phase-3H count conservation contradiction: {dict(observed)} != {expected}")
    return {"block_count": len(observed), "sample_count": len(rows), "samples_per_block": 32}


def validate_paired_provenance(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["length"]), int(row["sample_index"])), []).append(row)
    expected = {7500, 9000, 10000}
    for identity, records in grouped.items():
        if {int(row["checkpoint_update"]) for row in records} != expected:
            raise ValueError(f"E007 Phase-3H paired checkpoint membership contradiction: {identity}")
        for field in ("seed", "noise_identity_sha256", "initial_noise_tensor_sha256", "reverse_draw_provenance_sha256"):
            if len({row[field] for row in records}) != 1:
                raise ValueError(f"E007 Phase-3H paired provenance contradiction: {identity} {field}")
    if len(grouped) != 160:
        raise ValueError("E007 Phase-3H paired identity count contradiction")
    compact = [
        {
            "length": key[0],
            "sample_index": key[1],
            "seed": int(records[0]["seed"]),
            "initial_noise_tensor_sha256": records[0]["initial_noise_tensor_sha256"],
        }
        for key, records in sorted(grouped.items())
    ]
    return {
        "paired_identity_count": 160,
        "checkpoint_count_per_identity": 3,
        "paired_provenance_sha256": phase3g._canonical_sha(compact),
    }


def paired_comparisons(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    indexed = {(int(row["checkpoint_update"]), int(row["length"]), int(row["sample_index"])): row for row in rows}
    output: dict[str, Any] = {
        "difference_sign": (
            "candidate minus baseline; negative is better only for predeclared error/objective fields; "
            "raw topology descriptors remain descriptive"
        ),
        "bootstrap_replicates": int(config["aggregation"]["bootstrap_replicates"]),
        "comparisons": {},
    }
    base_seed = int(config["aggregation"]["bootstrap_seed"]) + 10_000
    identities = sorted({(int(row["length"]), int(row["sample_index"])) for row in rows})
    for comparison_index, (candidate, baseline) in enumerate(PAIRWISE_COMPARISONS):
        result = {"candidate": candidate, "baseline": baseline, "count": len(identities), "global": {}, "by_length": {}}
        for field_index, field in enumerate(NUMERIC_FIELDS):
            differences = [
                float(indexed[(candidate, length, sample)][field]) - float(indexed[(baseline, length, sample)][field])
                for length, sample in identities
            ]
            result["global"][field] = _difference_summary(
                differences,
                seed=base_seed + comparison_index * 1000 + field_index,
                replicates=int(config["aggregation"]["bootstrap_replicates"]),
            )
        for length in map(int, config["lengths"]):
            selected = [(item_length, sample) for item_length, sample in identities if item_length == length]
            result["by_length"][str(length)] = {}
            for field_index, field in enumerate(NUMERIC_FIELDS):
                differences = [
                    float(indexed[(candidate, item_length, sample)][field])
                    - float(indexed[(baseline, item_length, sample)][field])
                    for item_length, sample in selected
                ]
                result["by_length"][str(length)][field] = _difference_summary(
                    differences,
                    seed=base_seed + comparison_index * 1000 + 100 + length + field_index,
                    replicates=int(config["aggregation"]["bootstrap_replicates"]),
                )
        output["comparisons"][f"{candidate}_minus_{baseline}"] = result
    return output


def _difference_summary(values: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std()),
        "quantile_05": float(np.quantile(array, 0.05)),
        "quantile_95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "fraction_candidate_better": float(np.mean(array < 0)),
        "mean_bootstrap_ci_95": _bootstrap_ci(values, seed=seed, replicates=replicates),
    }


def checkpoint_pareto(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = []
    for checkpoint in (7500, 9000, 10000):
        selected = [row for row in rows if int(row["checkpoint_update"]) == checkpoint]
        candidates.append(
            {
                "checkpoint_update": checkpoint,
                "adjacent_reference_error_mean": float(
                    np.mean([row["adjacent_reference_error_angstrom"] for row in selected])
                ),
                "radius_relative_reference_error_mean": float(
                    np.mean([row["radius_of_gyration_relative_reference_error"] for row in selected])
                ),
                "clash_reference_excess_mean": float(
                    np.mean([row["non_neighbor_clash_reference_excess"] for row in selected])
                ),
                "contact_density_reference_error_mean": float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    row["contact_density_6a_reference_error"],
                                    row["contact_density_8a_reference_error"],
                                    row["contact_density_10a_reference_error"],
                                ]
                            )
                            for row in selected
                        ]
                    )
                ),
                "rank3_residual_energy_fraction_mean": float(
                    np.mean([row["rank3_residual_energy_fraction"] for row in selected])
                ),
                "adjacent_failure_fraction": float(
                    np.mean([not row["adjacent_original_gate_pass"] for row in selected])
                ),
            }
        )
    dominated_by: dict[str, list[int]] = {}
    nondominated = []
    for candidate in candidates:
        dominators = [
            int(other["checkpoint_update"])
            for other in candidates
            if other is not candidate
            and all(float(other[field]) <= float(candidate[field]) for field in PARETO_FIELDS)
            and any(float(other[field]) < float(candidate[field]) for field in PARETO_FIELDS)
        ]
        if dominators:
            dominated_by[str(candidate["checkpoint_update"])] = dominators
        else:
            nondominated.append(int(candidate["checkpoint_update"]))
    return {
        "direction": "all predeclared objectives minimized independently; no scalar score",
        "objectives": list(PARETO_FIELDS),
        "candidates": candidates,
        "nondominated_checkpoints": nondominated,
        "dominated_by": dominated_by,
    }


def decision_summary(
    summaries: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Any],
    pareto: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_rows = {int(row["checkpoint_update"]): row for row in pareto["candidates"]}

    def dominates(left: int, right: int) -> bool:
        return all(candidate_rows[left][field] <= candidate_rows[right][field] for field in PARETO_FIELDS) and any(
            candidate_rows[left][field] < candidate_rows[right][field] for field in PARETO_FIELDS
        )

    strict = {"9000_dominates_10000": dominates(9000, 10000), "10000_dominates_9000": dominates(10000, 9000)}
    paired = comparisons["comparisons"]["10000_minus_9000"]["global"]
    bootstrap_support = {
        "10000_over_9000": all(paired[field]["mean_bootstrap_ci_95"][1] < 0 for field in PAIRED_DECISION_FIELDS),
        "9000_over_10000": all(paired[field]["mean_bootstrap_ci_95"][0] > 0 for field in PAIRED_DECISION_FIELDS),
    }
    all_record = {
        str(checkpoint): all(
            bool(gate["pass"]) for gate in summaries[str(checkpoint)]["all_record_hard_gates"].values()
        )
        for checkpoint in (7500, 9000, 10000)
    }
    recommendation: int | None = None
    if strict["9000_dominates_10000"] and bootstrap_support["9000_over_10000"] and all_record["9000"]:
        recommendation = 9000
    if strict["10000_dominates_9000"] and bootstrap_support["10000_over_9000"] and all_record["10000"]:
        recommendation = 10000
    return {
        "classification": "checkpoint_recommendation_supported" if recommendation else "inconclusive",
        "recommended_checkpoint": recommendation,
        "strict_dominance": strict,
        "paired_bootstrap_support": bootstrap_support,
        "all_record_gate_pass": all_record,
        "length_or_tail_limitations": {
            str(checkpoint): {
                "failing_lengths": [
                    length
                    for length in (64, 128, 256, 384, 500)
                    if not summaries[f"{checkpoint}:{length}"]["adjacent_all_record_gate_pass"]
                ],
                "failure_count": int(summaries[str(checkpoint)]["adjacent_failure_count"]),
            }
            for checkpoint in (7500, 9000, 10000)
        },
        "scientific_review_required": True,
        "authorizing": False,
    }


def _checkpoint_model(item: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    payload = torch.load(item["path"], map_location="cpu", weights_only=False)
    if payload.get("version") != "e007_coordinate_real_continuation_to_10000_v1":
        raise ValueError("E007 Phase-3H checkpoint version contradiction")
    if int(payload.get("optimizer_update", -1)) != int(item["optimizer_update"]):
        raise ValueError("E007 Phase-3H checkpoint update contradiction")
    if payload.get("continuation_configuration_sha256") != config["continuation"]["config_sha256"]:
        raise ValueError("E007 Phase-3H checkpoint configuration contradiction")
    source_config = yaml.safe_load(Path(config["phase3f"]["config_path"]).read_text())
    model = EquivariantPairCoordinateUNet(**source_config["model"])
    if sum(parameter.numel() for parameter in model.parameters()) != phase3g.EXPECTED_PARAMETER_COUNT:
        raise ValueError("E007 Phase-3H model parameter-count contradiction")
    model.load_state_dict(payload["model"])
    model.requires_grad_(False).eval().to(device)
    del payload
    return model


def _parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _block_path(staging: Path, checkpoint: int, length: int) -> Path:
    return staging / "blocks" / f"step-{checkpoint:05d}-length-{length}.parquet"


def _prepare_uncommitted_sample_block(staging: Path, *, checkpoint: int, length: int) -> tuple[Path, Path]:
    checkpoint_dir = staging / "samples" / f"step-{checkpoint:05d}"
    sample_dir = checkpoint_dir / f"length-{length}"
    temporary_dir = checkpoint_dir / f".length-{length}.inprogress"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for uncommitted in (temporary_dir, sample_dir):
        if uncommitted.exists():
            shutil.rmtree(uncommitted)
    temporary_dir.mkdir()
    return sample_dir, temporary_dir


def _completed_blocks(staging: Path) -> dict[str, dict[str, Any]]:
    path = staging / "block_journal.json"
    return json.loads(path.read_text()) if path.exists() else {}


def verify_completed_blocks(staging: Path, journal: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    root = staging.resolve()
    permitted = {
        f"{int(checkpoint['optimizer_update'])}:{int(length)}"
        for checkpoint in config["checkpoints"]
        for length in config["lengths"]
    }
    for key, record in journal.items():
        if key not in permitted:
            raise ValueError(f"E007 Phase-3H journal contains an unknown block: {key}")
        relative = Path(str(record["path"]))
        path = (staging / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"E007 Phase-3H journal block path escapes staging: {relative}")
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"E007 Phase-3H completed block hash contradiction: {key}")
        table = pq.read_table(path)
        if table.num_rows != 32 or int(record["row_count"]) != 32:
            raise ValueError(f"E007 Phase-3H completed block row-count contradiction: {key}")
        for row in table.select(["artifact_path", "artifact_sha256"]).to_pylist():
            relative_artifact = Path(str(row["artifact_path"]))
            artifact = (staging / relative_artifact).resolve()
            if relative_artifact.is_absolute() or not artifact.is_relative_to(root):
                raise ValueError(f"E007 Phase-3H sample path escapes staging: {relative_artifact}")
            if not artifact.is_file() or sha256_file(artifact) != row["artifact_sha256"]:
                raise ValueError(f"E007 Phase-3H completed sample hash contradiction: {key}")


def _sample_block(
    *,
    checkpoint: Mapping[str, Any],
    length: int,
    model: torch.nn.Module,
    diffusion: CoordinateVPDiffusion,
    reference: Mapping[str, float],
    config: Mapping[str, Any],
    staging: Path,
    device: torch.device,
    progress: Any | None = None,
) -> list[dict[str, Any]]:
    rows = []
    sample_dir, temporary_dir = _prepare_uncommitted_sample_block(
        staging, checkpoint=int(checkpoint["optimizer_update"]), length=length
    )
    for identity in [item for item in paired_seed_records(config) if int(item["length"]) == length]:
        result = diffusion.sample(model, length=length, seed=int(identity["seed"]), device=device)
        replay = diffusion.sample(model, length=length, seed=int(identity["seed"]), device=device)
        coordinates = result["coordinates"].detach().cpu()
        deterministic = bool(torch.equal(coordinates, replay["coordinates"].detach().cpu()))
        row, _ = coordinate_metrics(
            coordinates,
            checkpoint=int(checkpoint["optimizer_update"]),
            length=length,
            sample_index=int(identity["sample_index"]),
            seed=int(identity["seed"]),
            reference=reference,
            config=config,
        )
        noise = phase3g.initial_noise_provenance(length, int(identity["seed"]), device)
        physical = coordinates[0].double() * float(config["coordinate_scale_angstrom"])
        distances = coordinates_to_distance_matrix(physical, diagnostic_float64=True).float()
        artifact = temporary_dir / f"sample-{int(identity['sample_index']):03d}.npz"
        np.savez_compressed(artifact, coordinates=coordinates.numpy(), distance_matrix=distances.numpy())
        row.update(
            {
                "deterministic_replay": deterministic,
                "artifact_path": (sample_dir / artifact.name).relative_to(staging).as_posix(),
                "artifact_sha256": sha256_file(artifact),
                "noise_identity_sha256": identity["noise_identity_sha256"],
                "initial_noise_tensor_sha256": noise["tensor_sha256"],
                "reverse_draw_provenance_sha256": identity["reverse_draw_provenance_sha256"],
            }
        )
        rows.append(row)
        phase3g._enforce_memory(device, config)
        if progress is not None:
            progress(len(rows))
    temporary_dir.replace(sample_dir)
    return rows


def _inventory(staging: Path) -> list[dict[str, Any]]:
    excluded = {"artifact_inventory.json", "report.json", "protocol.json", "heartbeat.json"}
    return [
        {"path": path.relative_to(staging).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(staging.rglob("*"))
        if path.is_file() and path.name not in excluded and ".tmp" not in path.name
    ]


def run_checkpoint_sampling_replication(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    prerequisites_before = verify_prerequisites(config)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists():
        raise FileExistsError(f"E007 Phase-3H output already exists: {output}")
    if resume:
        if not staging.is_dir():
            raise FileNotFoundError("E007 Phase-3H resume requires an existing staging directory")
    elif staging.exists():
        raise FileExistsError(f"E007 Phase-3H staging already exists: {staging}")
    else:
        staging.mkdir(parents=True)
        (staging / "blocks").mkdir()
    heartbeat_path = staging / "heartbeat.json"
    started = time.monotonic()

    def heartbeat(status: str, **values: Any) -> None:
        phase3g._atomic_json(
            heartbeat_path, {"status": status, "updated_utc": phase3g._utc_now(), **values, **NON_AUTHORIZING}
        )

    journal = _completed_blocks(staging)
    verify_completed_blocks(staging, journal, config)
    heartbeat("initializing", completed_blocks=len(journal), total_blocks=15)
    try:
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E007 Phase-3H configured audit requires CUDA")
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(device)
        references, reference_provenance = phase3g._reference_distributions(config)
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        parameter_integrity: dict[str, Any] = {}
        with coordinate_model_execution_context(config["numerics"], device) as backend:
            for checkpoint in config["checkpoints"]:
                pending = [
                    int(length)
                    for length in config["lengths"]
                    if f"{int(checkpoint['optimizer_update'])}:{int(length)}" not in journal
                ]
                if not pending:
                    continue
                model = _checkpoint_model(checkpoint, config, device)
                parameter_before = _parameter_sha256(model)
                for length in pending:
                    key = f"{int(checkpoint['optimizer_update'])}:{length}"
                    prior = sum(int(record["row_count"]) for record in journal.values())

                    def block_progress(
                        count: int,
                        *,
                        prior_count: int = prior,
                        checkpoint_update: int = int(checkpoint["optimizer_update"]),
                        current_length: int = length,
                    ) -> None:
                        completed = prior_count + count
                        elapsed = time.monotonic() - started
                        heartbeat(
                            "running",
                            checkpoint_update=checkpoint_update,
                            length=current_length,
                            completed_samples=completed,
                            total_samples=480,
                            memory=phase3g._memory(device),
                            elapsed_seconds=elapsed,
                            eta_seconds=elapsed / max(completed, 1) * (480 - completed),
                        )

                    rows = _sample_block(
                        checkpoint=checkpoint,
                        length=length,
                        model=model,
                        diffusion=diffusion,
                        reference=references[length],
                        config=config,
                        staging=staging,
                        device=device,
                        progress=block_progress,
                    )
                    block_path = _block_path(staging, int(checkpoint["optimizer_update"]), length)
                    phase3g._atomic_parquet(block_path, rows, config["publication"]["parquet_compression"])
                    journal[key] = {
                        "checkpoint_update": int(checkpoint["optimizer_update"]),
                        "length": length,
                        "row_count": len(rows),
                        "path": block_path.relative_to(staging).as_posix(),
                        "sha256": sha256_file(block_path),
                        "model_parameter_sha256": parameter_before,
                    }
                    phase3g._atomic_json(staging / "block_journal.json", journal)
                parameter_after = _parameter_sha256(model)
                if parameter_after != parameter_before:
                    raise ValueError(
                        f"E007 Phase-3H inference changed checkpoint {checkpoint['optimizer_update']} parameters"
                    )
                parameter_integrity[str(checkpoint["optimizer_update"])] = {
                    "before_sha256": parameter_before,
                    "after_sha256": parameter_after,
                    "unchanged": True,
                }
                del model
                torch.cuda.empty_cache()

            verify_completed_blocks(staging, journal, config)
            expected = {f"{step}:{length}" for step in (7500, 9000, 10000) for length in config["lengths"]}
            if set(journal) != expected:
                raise ValueError("E007 Phase-3H completed block set is incomplete")
            for update in (7500, 9000, 10000):
                if str(update) in parameter_integrity:
                    continue
                hashes = {
                    str(record.get("model_parameter_sha256", ""))
                    for key, record in journal.items()
                    if key.startswith(f"{update}:")
                }
                if len(hashes) != 1 or not next(iter(hashes)):
                    raise ValueError(f"E007 Phase-3H resume lacks model-integrity evidence for {update}")
                parameter_hash = next(iter(hashes))
                parameter_integrity[str(update)] = {
                    "before_sha256": parameter_hash,
                    "after_sha256": parameter_hash,
                    "unchanged": True,
                    "restored_from_verified_block_journal": True,
                }
            rows = []
            for _key, record in sorted(journal.items()):
                table = pq.read_table(staging / record["path"])
                rows.extend(table.to_pylist())
            count_conservation = validate_count_conservation(rows, config)
            paired_provenance = validate_paired_provenance(rows, config)
            summaries: dict[str, Any] = {}
            for checkpoint in (7500, 9000, 10000):
                selected = [row for row in rows if int(row["checkpoint_update"]) == checkpoint]
                summaries[str(checkpoint)] = descriptive_summary(selected, config)
                for length in map(int, config["lengths"]):
                    summaries[f"{checkpoint}:{length}"] = descriptive_summary(
                        [row for row in selected if int(row["length"]) == length], config
                    )
            comparisons = paired_comparisons(rows, config)
            pareto = checkpoint_pareto(rows)
            decision = decision_summary(summaries, comparisons, pareto)
            sample_metrics_path = staging / "sample_metrics.parquet"
            phase3g._atomic_parquet(sample_metrics_path, rows, config["publication"]["parquet_compression"])
            phase3g._atomic_json(staging / "paired_comparisons.json", comparisons)
            phase3g._atomic_json(staging / "checkpoint_pareto.json", pareto)
            phase3g._atomic_json(
                staging / "block_inventory.json",
                {
                    "journal": journal,
                    "count_conservation": count_conservation,
                    "paired_provenance": paired_provenance,
                    "reference_provenance": reference_provenance,
                },
            )
        prerequisites_after = verify_prerequisites(config)
        if prerequisites_after != prerequisites_before:
            raise ValueError("E007 Phase-3H protected inputs changed")
        inventory = _inventory(staging)
        phase3g._atomic_json(
            staging / "artifact_inventory.json",
            {"artifacts": inventory, "aggregate_sha256": phase3g._canonical_sha(inventory)},
        )
        continuation_report = json.loads((Path(config["continuation"]["source_dir"]) / "report.json").read_text())
        report = {
            "status": "completed_non_authorizing",
            "version": VERSION,
            "scientific_question": "retain step 9000 or 10000, with step 7500 as paired comparison",
            "sample_count": len(rows),
            "paired_noise_identity_count": 160,
            "summaries": summaries,
            "paired_comparisons": comparisons,
            "checkpoint_pareto": pareto,
            "decision": decision,
            "denoising_evidence_kept_separate": {
                "source": "immutable completed continuation report",
                "best_denoising": continuation_report.get("best_denoising"),
            },
            "original_phase3f_or_phase3g_classification_revised": False,
            "protected_inputs_unchanged": True,
            "model_parameters_unchanged": parameter_integrity,
            "numerical_backend": backend,
            "memory": phase3g._memory(device),
            "elapsed_seconds": time.monotonic() - started,
            **NON_AUTHORIZING,
        }
        phase3g._atomic_json(staging / "report.json", report)
        protocol = {
            "status": report["status"],
            "version": VERSION,
            "configuration_sha256": sha256_file(config_path),
            "source_hashes": prerequisites_before,
            "report_sha256": sha256_file(staging / "report.json"),
            "sample_metrics_sha256": sha256_file(staging / "sample_metrics.parquet"),
            "paired_comparisons_sha256": sha256_file(staging / "paired_comparisons.json"),
            "checkpoint_pareto_sha256": sha256_file(staging / "checkpoint_pareto.json"),
            "block_inventory_sha256": sha256_file(staging / "block_inventory.json"),
            "artifact_inventory_sha256": sha256_file(staging / "artifact_inventory.json"),
            "completed_utc": phase3g._utc_now(),
            "protected_inputs_unchanged": True,
            **NON_AUTHORIZING,
        }
        phase3g._atomic_json(staging / "protocol.json", protocol)
        report_bytes = (staging / "report.json").read_bytes()
        protocol_bytes = (staging / "protocol.json").read_bytes()
        if (
            report_bytes == protocol_bytes
            or len(report_bytes) == len(protocol_bytes)
            or sha256_file(staging / "report.json") == sha256_file(staging / "protocol.json")
        ):
            raise ValueError("E007 Phase-3H report/protocol publication separation failure")
        heartbeat("completed", completed_samples=480, total_samples=480, report_sha256=protocol["report_sha256"])
        staging.replace(output)
        return {"status": report["status"], "decision": decision, "output_dir": str(output), **NON_AUTHORIZING}
    except KeyboardInterrupt as error:
        heartbeat(
            "interrupted",
            error_type=type(error).__name__,
            error_message="SIGINT",
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
    except MemoryError as error:
        heartbeat(
            "memory_limit_exceeded",
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
    except BaseException as error:
        heartbeat(
            "failed",
            error_type=type(error).__name__,
            error_message=str(error)[:2000],
            completed_blocks=len(journal),
            resumable=True,
        )
        raise
