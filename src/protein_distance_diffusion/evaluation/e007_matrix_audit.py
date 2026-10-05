"""Bounded, read-only E007 audit of the frozen E004 matrix generator."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    prediction_parameterization_from_config,
)
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.evaluation.distance_matrix_quality import MatrixQualityConfig, assess_distance_matrix
from protein_distance_diffusion.evaluation.repairability import classical_mds_rank3_projection
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.trainer import build_model_from_config

AUDIT_VERSION = "e007_matrix_generator_audit_v1"
MIB = 1024**2
NUMERIC_METRICS = (
    "symmetry_error_max_angstrom",
    "diagonal_error_max_angstrom",
    "negative_distance_fraction",
    "adjacent_residue_distance_rmse_angstrom",
    "adjacent_residue_plausible_fraction",
    "triangle_violation_fraction",
    "negative_eigenmass_fraction",
    "rank3_reconstruction_rmse_angstrom",
    "rank3_residual_energy_fraction",
    "contact_density.6",
    "contact_density.8",
    "contact_density.10",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def candidate_seed_schedule(counts: dict[Any, Any], master_seed: int) -> dict[int, list[int]]:
    """Create a stable independent seed bank for every requested length."""
    schedule: dict[int, list[int]] = {}
    for raw_length, raw_count in sorted(counts.items(), key=lambda item: int(item[0])):
        length, count = int(raw_length), int(raw_count)
        if length <= 0 or count <= 0:
            raise ValueError("E007 generated lengths and counts must be positive")
        rng = np.random.default_rng(int(master_seed) + length * 1_000_003)
        schedule[length] = [int(value) for value in rng.integers(0, 2**31 - 1, size=count)]
    return schedule


def generated_candidate_id(length: int, index: int, seed: int) -> str:
    """Return the stable identity of one independent matrix candidate."""
    return f"e007_generated_N{int(length):04d}_i{int(index):05d}_seed{int(seed)}"


def panel_identity(sample_ids: list[str], *, seed: int, algorithm: str) -> str:
    """Hash ordered panel membership and its selection contract."""
    return _sha256_json({"algorithm": algorithm, "sample_ids": sample_ids, "seed": int(seed)})


def select_real_reference_panel(
    manifest_path: str | Path,
    *,
    counts_by_length: dict[Any, Any],
    seed: int,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an exact-length validation panel using scan-order-independent hashes."""
    required = {int(key): int(value) for key, value in counts_by_length.items()}
    candidates = {length: [] for length in required}
    parquet = pq.ParquetFile(manifest_path)
    columns = ["sample_id", "length", "path", "pdb_id", "chain_id", "model_number", "split"]
    if any(column not in parquet.schema_arrow.names for column in columns):
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        raise ValueError(f"E007 validation manifest lacks required columns: {missing}")
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=columns):
        for row in batch.to_pylist():
            length = int(row["length"])
            if length not in candidates:
                continue
            if str(row["split"]).strip().lower() != "validation":
                raise ValueError(f"E007 reference row is not validation-owned: {row['sample_id']}")
            sample_id = str(row["sample_id"])
            rank = hashlib.sha256(f"{int(seed)}|validation|{sample_id}".encode()).hexdigest()
            candidates[length].append((rank, sample_id, row))
    selected: list[dict[str, Any]] = []
    counts_available: dict[str, int] = {}
    for length, count in sorted(required.items()):
        ordered = sorted(candidates[length], key=lambda item: (item[0], item[1]))
        counts_available[str(length)] = len(ordered)
        if len(ordered) < count:
            raise ValueError(f"E007 length {length} has {len(ordered)} validation rows; {count} are required")
        selected.extend(dict(item[2]) | {"requested_length": length} for item in ordered[:count])
    identities = [str(row["sample_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("E007 real-reference panel contains duplicate sample IDs")
    algorithm = "sha256_rank_exact_length_validation_v1"
    return selected, {
        "algorithm": algorithm,
        "seed": int(seed),
        "counts_available_by_length": counts_available,
        "counts_selected_by_length": {str(key): int(value) for key, value in sorted(required.items())},
        "sample_ids": identities,
        "sample_id_sha256": panel_identity(identities, seed=seed, algorithm=algorithm),
        "unique": True,
    }


def residue_permuted_control(matrix: np.ndarray, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one deterministic residue permutation to both matrix axes."""
    source = np.asarray(matrix)
    permutation = np.random.default_rng(int(seed)).permutation(source.shape[0])
    value = source[np.ix_(permutation, permutation)].copy()
    return value, {
        "control_type": "residue_permuted",
        "seed": int(seed),
        "permutation": permutation.tolist(),
        "operation": "D_prime = P D P_transpose",
        "generic_non_euclidean_control": False,
        "purpose": "disrupt_chain_order_and_adjacent_residue_relationships",
    }


def explicit_corruption(
    matrix: np.ndarray,
    *,
    corruption_type: str,
    settings: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one named, deterministic corruption without combining mechanisms."""
    value = np.asarray(matrix, dtype=np.float64).copy()
    n = value.shape[0]
    if n < 4:
        raise ValueError("E007 explicit controls require at least four residues")
    provenance: dict[str, Any] = {"control_type": corruption_type, "settings": dict(settings)}
    if corruption_type == "asymmetry":
        value[0, 1] += float(settings["delta_angstrom"])
        provenance["operation"] = "add_delta_to_D_0_1_only"
    elif corruption_type == "nonzero_diagonal":
        value[0, 0] = float(settings["value_angstrom"])
        provenance["operation"] = "set_D_0_0"
    elif corruption_type == "negative_distance":
        value[0, 1] = value[1, 0] = float(settings["value_angstrom"])
        provenance["operation"] = "set_symmetric_pair_0_1"
    elif corruption_type == "local_chain_disruption":
        delta = float(settings["adjacent_delta_angstrom"])
        indices = np.arange(n - 1)
        value[indices, indices + 1] += delta
        value[indices + 1, indices] += delta
        provenance["operation"] = "add_delta_to_all_adjacent_symmetric_pairs"
    elif corruption_type == "triangle_violation":
        excess = float(settings["excess_angstrom"])
        value[0, 1] = value[1, 0] = value[0, 2] + value[2, 1] + excess
        provenance["operation"] = "set_D_0_1_to_D_0_2_plus_D_2_1_plus_excess"
    elif corruption_type == "non_euclidean_four_point":
        scale = float(settings["scale_angstrom"])
        cycle = np.array(
            [
                [0.0, scale, 2 * scale, scale],
                [scale, 0.0, scale, 2 * scale],
                [2 * scale, scale, 0.0, scale],
                [scale, 2 * scale, scale, 0.0],
            ],
            dtype=np.float64,
        )
        value[:4, :4] = cycle
        provenance["operation"] = "replace_first_four_pairs_with_cycle_shortest_path_metric"
        provenance["generic_non_euclidean_control"] = True
    else:
        raise ValueError(f"Unknown E007 corruption type: {corruption_type}")
    return value.astype(np.float32), provenance


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    value: Any = row["assessment"]
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if value is None or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _bootstrap_ci(values: np.ndarray, *, iterations: int, seed: int) -> list[float] | None:
    if values.size < 2 or iterations <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        means[index] = np.mean(rng.choice(values, size=values.size, replace=True))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_metrics(
    rows: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Compute global and length-stratified robust summaries."""
    aggregates: list[dict[str, Any]] = []
    panels = sorted({str(row["panel"]) for row in rows})
    lengths: list[int | None] = [None, *sorted({int(row["requested_length"]) for row in rows})]
    for panel in panels:
        for length in lengths:
            subset = [
                row
                for row in rows
                if row["panel"] == panel and (length is None or int(row["requested_length"]) == length)
            ]
            if not subset:
                continue
            for metric in NUMERIC_METRICS:
                values = np.asarray(
                    [value for row in subset if (value := _metric_value(row, metric)) is not None], dtype=np.float64
                )
                if not values.size:
                    continue
                aggregates.append(
                    {
                        "panel": panel,
                        "requested_length": length,
                        "metric": metric,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "standard_deviation": float(np.std(values)),
                        "quantile_05": float(np.quantile(values, 0.05)),
                        "quantile_25": float(np.quantile(values, 0.25)),
                        "quantile_75": float(np.quantile(values, 0.75)),
                        "quantile_95": float(np.quantile(values, 0.95)),
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                        "mean_bootstrap_ci_95": _bootstrap_ci(
                            values,
                            iterations=bootstrap_iterations,
                            seed=bootstrap_seed + len(aggregates),
                        ),
                    }
                )
    return aggregates


def paired_control_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-source control-minus-real differences."""
    real = {str(row["candidate_id"]): row for row in rows if row["panel"] == "real_validation"}
    output: list[dict[str, Any]] = []
    panels = sorted({row["panel"] for row in rows if row.get("source_matrix_id")})
    for panel in panels:
        controls = [row for row in rows if row["panel"] == panel]
        for metric in NUMERIC_METRICS:
            differences = []
            for control in controls:
                source = real.get(str(control["source_matrix_id"]))
                if source is None:
                    continue
                left, right = _metric_value(control, metric), _metric_value(source, metric)
                if left is not None and right is not None:
                    differences.append(left - right)
            if differences:
                values = np.asarray(differences, dtype=np.float64)
                output.append(
                    {
                        "panel": panel,
                        "metric": metric,
                        "difference_definition": "control_minus_corresponding_real",
                        "count": int(values.size),
                        "mean_difference": float(np.mean(values)),
                        "median_difference": float(np.median(values)),
                        "fraction_control_lower": float(np.mean(values < 0.0)),
                    }
                )
    return output


def aggregate_panel_comparisons(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare predeclared panel pairs using their separately reported means."""
    lookup = {(row["panel"], row["requested_length"], row["metric"]): row for row in aggregates}
    control_panels = sorted(
        {
            row["panel"]
            for row in aggregates
            if row["panel"].startswith("corruption_")
            or row["panel"] in {"residue_permuted", "real_rank3_reconstruction"}
        }
    )
    panel_pairs = [("real_validation", "generated")]
    panel_pairs.extend(("real_validation", panel) for panel in control_panels)
    panel_pairs.extend(("generated", panel) for panel in control_panels)
    comparisons = []
    for left, right in panel_pairs:
        keys = sorted(
            {
                (length, metric)
                for panel, length, metric in lookup
                if panel == left and (right, length, metric) in lookup
            },
            key=lambda item: (-1 if item[0] is None else int(item[0]), item[1]),
        )
        for length, metric in keys:
            left_row, right_row = lookup[(left, length, metric)], lookup[(right, length, metric)]
            comparisons.append(
                {
                    "left_panel": left,
                    "right_panel": right,
                    "requested_length": length,
                    "metric": metric,
                    "left_count": left_row["count"],
                    "right_count": right_row["count"],
                    "left_mean": left_row["mean"],
                    "right_mean": right_row["mean"],
                    "left_minus_right_mean": left_row["mean"] - right_row["mean"],
                }
            )
    return comparisons


def fatal_validity_reasons(assessment: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    """Apply only predeclared output-contract validity rules."""
    reasons = []
    if bool(thresholds["require_finite"]) and not bool(assessment.get("finite")):
        reasons.append("nonfinite_values")
    if float(assessment.get("symmetry_error_max_angstrom", 0.0)) > float(thresholds["maximum_symmetry_error_angstrom"]):
        reasons.append("symmetry_contract_violation")
    if float(assessment.get("diagonal_error_max_angstrom", 0.0)) > float(thresholds["maximum_diagonal_error_angstrom"]):
        reasons.append("diagonal_contract_violation")
    if int(assessment.get("negative_distance_count", 0)) > int(thresholds["maximum_negative_distance_count"]):
        reasons.append("negative_distance_contract_violation")
    return reasons


def apply_reference_warnings(
    rows: list[dict[str, Any]],
    *,
    quantiles: tuple[float, float],
) -> list[dict[str, Any]]:
    """Attach warning-only empirical reference comparisons to generated candidates."""
    low, high = quantiles
    references: dict[tuple[int, str], tuple[float, float]] = {}
    for length in sorted({int(row["requested_length"]) for row in rows}):
        real = [row for row in rows if row["panel"] == "real_validation" and int(row["requested_length"]) == length]
        for metric in NUMERIC_METRICS:
            values = np.asarray([value for row in real if (value := _metric_value(row, metric)) is not None])
            if values.size:
                references[(length, metric)] = (float(np.quantile(values, low)), float(np.quantile(values, high)))
    warnings = []
    for row in rows:
        if row["panel"] != "generated":
            continue
        for metric in NUMERIC_METRICS:
            value = _metric_value(row, metric)
            bounds = references.get((int(row["requested_length"]), metric))
            if value is not None and bounds is not None and not bounds[0] <= value <= bounds[1]:
                warning = {
                    "candidate_id": row["candidate_id"],
                    "metric": metric,
                    "value": value,
                    "reference_quantile_low": bounds[0],
                    "reference_quantile_high": bounds[1],
                    "classification": "quality_warning_not_fatal",
                }
                warnings.append(warning)
                row.setdefault("quality_warnings", []).append(warning)
    return warnings


def _memory_telemetry(device: torch.device | None = None) -> dict[str, float | None]:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    result: dict[str, float | None] = {"peak_rss_mib": rss, "cuda_allocated_mib": None, "cuda_reserved_mib": None}
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        result["cuda_allocated_mib"] = float(torch.cuda.memory_allocated(device) / MIB)
        result["cuda_reserved_mib"] = float(torch.cuda.memory_reserved(device) / MIB)
    return result


def _enforce_memory(bounds: dict[str, Any], device: torch.device | None = None) -> dict[str, float | None]:
    memory = _memory_telemetry(device)
    if float(memory["peak_rss_mib"] or 0.0) > float(bounds["maximum_rss_mib"]):
        raise MemoryError(f"E007 RSS limit exceeded: {memory}")
    if device is not None and device.type == "cuda":
        if float(memory["cuda_allocated_mib"] or 0.0) > float(bounds["maximum_cuda_allocated_mib"]):
            raise MemoryError(f"E007 CUDA allocated-memory limit exceeded: {memory}")
        if float(memory["cuda_reserved_mib"] or 0.0) > float(bounds["maximum_cuda_reserved_mib"]):
            raise MemoryError(f"E007 CUDA reserved-memory limit exceeded: {memory}")
    return memory


def _load_real_matrix(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = Path(str(record["path"]))
    path = path if path.is_absolute() else root / path
    with np.load(path, allow_pickle=False) as data:
        matrix = np.asarray(data["distance_matrix"], dtype=np.float32)
    length = int(record["length"])
    if matrix.shape != (length, length):
        raise ValueError(f"E007 real matrix shape contradiction for {record['sample_id']}: {matrix.shape}")
    return matrix


def _quality_config(config: dict[str, Any], *, seed: int) -> MatrixQualityConfig:
    bounds = config["bounds"]
    return MatrixQualityConfig(
        triangle_exact_max_length=int(bounds["triangle_exact_max_length"]),
        triangle_sample_count=int(bounds["triangle_sample_count"]),
        triangle_seed=int(seed),
        eigen_exact_max_length=int(bounds["eigen_exact_max_length"]),
        eigen_sample_size=min(256, int(bounds["eigen_exact_max_length"])),
        eigen_seed=int(seed),
    )


def _assessed_row(
    matrix: np.ndarray,
    *,
    candidate_id: str,
    panel: str,
    requested_length: int,
    metric_seed: int,
    config: dict[str, Any],
    source_matrix_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    length = int(matrix.shape[0])
    mask = np.ones((length, length), dtype=bool)
    assessment = assess_distance_matrix(
        matrix,
        pair_mask=mask,
        sequence_length=length,
        candidate_id=candidate_id,
        config=_quality_config(config, seed=metric_seed),
    )
    return {
        "candidate_id": candidate_id,
        "panel": panel,
        "requested_length": int(requested_length),
        "actual_valid_length": length,
        "source_matrix_id": source_matrix_id,
        "provenance": provenance or {},
        "assessment": assessment,
        "fatal_validity_failures": fatal_validity_reasons(assessment, config["fatal_contract_thresholds"]),
        "quality_warnings": [],
    }


def _checkpoint_model(
    root: Path, config: dict[str, Any], device: torch.device
) -> tuple[Any, GaussianDiffusion, dict[str, Any]]:
    checkpoint_path = root / config["generator"]["checkpoint_path"]
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    training_config = yaml.safe_load((root / config["generator"]["training_config_path"]).read_text(encoding="utf-8"))
    if checkpoint["config"]["model"] != training_config["model"]:
        raise ValueError("E007 checkpoint model configuration contradicts the pinned training configuration")
    if int(checkpoint["config"]["diffusion_steps"]) != int(config["generator"]["diffusion_steps"]):
        raise ValueError("E007 checkpoint diffusion schedule contradicts the audit configuration")
    parameterization = prediction_parameterization_from_config(checkpoint["config"])
    if parameterization.value != str(config["generator"]["prediction_parameterization"]):
        raise ValueError("E007 checkpoint prediction parameterization contradicts the audit configuration")
    model = build_model_from_config(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["ema"])
    model.requires_grad_(False)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 7_582_833:
        raise ValueError(f"E007 E004 parameter-count contradiction: {parameter_count}")
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["generator"]["diffusion_steps"]))).to(device)
    runtime_checkpoint = {"config": checkpoint["config"]}
    del checkpoint
    return model, diffusion, runtime_checkpoint


def _sample_candidate(
    *,
    model: Any,
    diffusion: GaussianDiffusion,
    checkpoint: dict[str, Any],
    length: int,
    seed: int,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = ((int(length) + int(model.downsample_factor) - 1) // int(model.downsample_factor)) * int(
        model.downsample_factor
    )
    lengths = torch.tensor([length], dtype=torch.long, device=device)
    pair_mask = make_pair_mask(lengths, side).to(device)
    separation = make_sequence_separation(lengths, side).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.inference_mode():
        sampled = sample_ddpm(
            model,
            diffusion,
            lengths=lengths,
            pair_mask=pair_mask,
            sequence_separation=separation,
            device=device,
            generator=generator,
            prediction_type=prediction_parameterization_from_config(checkpoint["config"]),
        )
    normalized = sampled[0, 0, :length, :length].detach().cpu().numpy().astype(np.float32)
    mask = pair_mask[0, 0, :length, :length].detach().cpu().numpy().astype(bool)
    return normalized, normalized * np.float32(scale), mask


def load_candidate_npz(path: str | Path) -> dict[str, Any]:
    """Load and validate one lossless generated-candidate artifact."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "candidate_id",
            "requested_length",
            "actual_valid_length",
            "sampling_seed",
            "normalized_matrix",
            "physical_matrix_angstrom",
            "pair_mask",
            "metadata",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"E007 candidate NPZ lacks fields: {missing}")
        result = {
            "candidate_id": str(data["candidate_id"]),
            "requested_length": int(data["requested_length"]),
            "actual_valid_length": int(data["actual_valid_length"]),
            "sampling_seed": int(data["sampling_seed"]),
            "normalized_matrix": np.asarray(data["normalized_matrix"], dtype=np.float32),
            "physical_matrix_angstrom": np.asarray(data["physical_matrix_angstrom"], dtype=np.float32),
            "pair_mask": np.asarray(data["pair_mask"], dtype=bool),
            "metadata": json.loads(str(data["metadata"])),
        }
    length = result["actual_valid_length"]
    if result["normalized_matrix"].shape != (length, length) or result["pair_mask"].shape != (length, length):
        raise ValueError("E007 candidate NPZ shape contradiction")
    return result


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_hashes(directory: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    ignored = exclude or set()
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and str(path.relative_to(directory)) not in ignored
    }


def _publish_completed(
    staging: Path,
    output: Path,
    *,
    report: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    payload_hashes = _artifact_hashes(staging, exclude={"heartbeat.json", "protocol.json", "report.json"})
    report["published_payload_hashes"] = payload_hashes
    report["hash_scope_note"] = "report/protocol self-hashes are recorded by the final heartbeat, not self-embedded"
    _atomic_json(staging / "report.json", report)
    protocol["report_sha256"] = sha256_file(staging / "report.json")
    protocol["published_payload_hashes"] = payload_hashes
    _atomic_json(staging / "protocol.json", protocol)
    heartbeat = {
        "version": AUDIT_VERSION,
        "status": "completed",
        "completed_utc": protocol["completed_utc"],
        "processed_candidates": report["counts"]["matrix_metric_rows"],
        "report_path": str(output / "report.json"),
        "report_sha256": sha256_file(staging / "report.json"),
        "protocol_sha256": sha256_file(staging / "protocol.json"),
    }
    _atomic_json(staging / "heartbeat.json", heartbeat)
    staging.replace(output)


def run_matrix_generator_audit(
    config_path: str | Path,
    *,
    plan: dict[str, Any],
    repository_root: str | Path = ".",
    sample_candidate: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> Path:
    """Execute the bounded audit after a caller has verified the Phase-1 plan."""
    root = Path(repository_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = root / str(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 audit output or in-progress directory already exists: {output}")
    if plan.get("authorizes_training") or plan.get("authorizes_joint_training"):
        raise ValueError("E007 audit plan must be non-authorizing")
    protected_before = {name: sha256_file(root / record["path"]) for name, record in plan["protected_inputs"].items()}
    started = _utc_now()
    staging.mkdir(parents=True)
    _atomic_json(
        staging / "heartbeat.json",
        {"version": AUDIT_VERSION, "status": "running", "stage": "authorization", "started_utc": started},
    )
    try:
        bounds = config["bounds"]
        device = torch.device(str(bounds["runtime_device"]))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("E007 real audit requires configured CUDA but CUDA is unavailable")
        panel, panel_record = select_real_reference_panel(
            root / config["dataset"]["validation_manifest_path"],
            counts_by_length=config["audit"]["real_counts_by_length"],
            seed=int(config["audit"]["panel_selection_seed"]),
        )
        _atomic_json(
            staging / "heartbeat.json",
            {"version": AUDIT_VERSION, "status": "running", "stage": "load_generator", "started_utc": started},
        )
        model, diffusion, checkpoint = _checkpoint_model(root, config, device)
        sampler = sample_candidate or _sample_candidate
        scale = float(config["matrix_representation"]["normalization"]["scale_angstrom"])
        rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        schedule = candidate_seed_schedule(
            config["audit"]["generated_counts_by_length"], int(config["audit"]["generated_master_seed"])
        )
        total_generated = sum(map(len, schedule.values()))
        completed_generated = 0
        for length, seeds in schedule.items():
            for index, seed in enumerate(seeds):
                candidate_id = generated_candidate_id(length, index, seed)
                normalized, physical, pair_mask = sampler(
                    model=model,
                    diffusion=diffusion,
                    checkpoint=checkpoint,
                    length=length,
                    seed=seed,
                    scale=scale,
                    device=device,
                )
                relative = Path("candidates") / f"{candidate_id}.npz"
                mask_sha = hashlib.sha256(np.ascontiguousarray(pair_mask).tobytes()).hexdigest()
                metadata = {
                    "candidate_id": candidate_id,
                    "generator_checkpoint_path": config["generator"]["checkpoint_path"],
                    "generator_checkpoint_sha256": config["generator"]["checkpoint_sha256"],
                    "generator_training_config": config["generator"]["training_config_path"],
                    "generator_training_config_sha256": config["generator"]["training_config_sha256"],
                    "diffusion_steps": int(config["generator"]["diffusion_steps"]),
                    "diffusion_schedule": "cosine_beta_schedule",
                    "sampler": "protein_distance_diffusion.diffusion.sampling.sample_ddpm",
                    "prediction_parameterization": config["generator"]["prediction_parameterization"],
                    "normalization": config["matrix_representation"]["normalization"],
                    "physical_conversion": "normalized_matrix * scale_angstrom",
                    "pair_mask_sha256": mask_sha,
                    "sampling_seed": seed,
                }
                _atomic_npz(
                    staging / relative,
                    candidate_id=np.asarray(candidate_id),
                    requested_length=np.asarray(length),
                    actual_valid_length=np.asarray(physical.shape[0]),
                    sampling_seed=np.asarray(seed),
                    normalized_matrix=normalized,
                    physical_matrix_angstrom=physical,
                    pair_mask=pair_mask,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                artifact_sha = sha256_file(staging / relative)
                manifest_rows.append(
                    metadata
                    | {
                        "panel": "generated",
                        "requested_length": length,
                        "actual_valid_length": int(physical.shape[0]),
                        "candidate_artifact_path": str(relative),
                        "candidate_artifact_sha256": artifact_sha,
                        "pair_mask_sha256": mask_sha,
                        "normalized_representation": "physical_distance_angstrom / 53.775000000000006",
                        "physical_unit": "angstrom",
                    }
                )
                rows.append(
                    _assessed_row(
                        physical,
                        candidate_id=candidate_id,
                        panel="generated",
                        requested_length=length,
                        metric_seed=seed,
                        config=config,
                        provenance=metadata | {"candidate_artifact_path": str(relative), "pair_mask_sha256": mask_sha},
                    )
                )
                completed_generated += 1
                memory = _enforce_memory(bounds, device)
                _atomic_json(
                    staging / "heartbeat.json",
                    {
                        "version": AUDIT_VERSION,
                        "status": "running",
                        "stage": "generated_candidates",
                        "started_utc": started,
                        "processed": completed_generated,
                        "total": total_generated,
                        "last_candidate_id": candidate_id,
                        "memory": memory,
                    },
                )
                del normalized, physical, pair_mask
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        del model, diffusion, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
        controls = config["audit"]["corruption_controls"]
        for panel_index, record in enumerate(panel):
            matrix = _load_real_matrix(root, record)
            source_id = f"real_validation::{record['sample_id']}"
            requested_length = int(record["requested_length"])
            source_provenance = {
                "sample_id": str(record["sample_id"]),
                "matrix_path": str(record["path"]),
                "pdb_id": str(record["pdb_id"]),
                "chain_id": str(record["chain_id"]),
                "model_number": int(record["model_number"]),
            }
            rows.append(
                _assessed_row(
                    matrix,
                    candidate_id=source_id,
                    panel="real_validation",
                    requested_length=requested_length,
                    metric_seed=int(config["audit"]["corruption_seed"]) + panel_index,
                    config=config,
                    provenance=source_provenance,
                )
            )
            reconstructed = classical_mds_rank3_projection(matrix).projected_distances.astype(np.float32)
            rows.append(
                _assessed_row(
                    reconstructed,
                    candidate_id=f"rank3_reconstruction::{record['sample_id']}",
                    panel="real_rank3_reconstruction",
                    requested_length=requested_length,
                    metric_seed=int(config["audit"]["corruption_seed"]) + panel_index,
                    config=config,
                    source_matrix_id=source_id,
                    provenance={
                        "source_matrix_id": source_id,
                        "operation": "classical_mds_rank3_mathematical_reconstruction",
                        "predicted_or_experimental_coordinates": False,
                    },
                )
            )
            permutation_seed = int(config["audit"]["corruption_seed"]) + panel_index * 101
            permuted, permutation_provenance = residue_permuted_control(matrix, seed=permutation_seed)
            rows.append(
                _assessed_row(
                    permuted,
                    candidate_id=f"residue_permuted::{record['sample_id']}",
                    panel="residue_permuted",
                    requested_length=requested_length,
                    metric_seed=int(config["audit"]["corruption_seed"]) + panel_index,
                    config=config,
                    source_matrix_id=source_id,
                    provenance=permutation_provenance | {"source_matrix_id": source_id},
                )
            )
            for corruption_type, settings in controls.items():
                corrupted, corruption_provenance = explicit_corruption(
                    matrix, corruption_type=corruption_type, settings=settings
                )
                rows.append(
                    _assessed_row(
                        corrupted,
                        candidate_id=f"{corruption_type}::{record['sample_id']}",
                        panel=f"corruption_{corruption_type}",
                        requested_length=requested_length,
                        metric_seed=int(config["audit"]["corruption_seed"]) + panel_index,
                        config=config,
                        source_matrix_id=source_id,
                        provenance=corruption_provenance | {"source_matrix_id": source_id},
                    )
                )
            _enforce_memory(bounds, device)
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "version": AUDIT_VERSION,
                    "status": "running",
                    "stage": "reference_and_controls",
                    "started_utc": started,
                    "processed": panel_index + 1,
                    "total": len(panel),
                    "last_sample_id": str(record["sample_id"]),
                },
            )
            del matrix, reconstructed, permuted

        warnings = apply_reference_warnings(
            rows, quantiles=tuple(float(value) for value in config["audit"]["reference_warning_quantiles"])
        )
        aggregates = aggregate_metrics(
            rows,
            bootstrap_iterations=int(config["audit"]["bootstrap_iterations"]),
            bootstrap_seed=int(config["audit"]["bootstrap_seed"]),
        )
        paired = paired_control_statistics(rows)
        panel_comparisons = aggregate_panel_comparisons(aggregates)
        generated_rows = [row for row in rows if row["panel"] == "generated"]
        generated_fatal = [
            {"candidate_id": row["candidate_id"], "reasons": row["fatal_validity_failures"]}
            for row in generated_rows
            if row["fatal_validity_failures"]
        ]
        classification = "completed_requires_scientific_review"
        recommendation = "generator_correction_required" if generated_fatal else "review_generator_quality"
        _atomic_jsonl(staging / "candidate_manifest.jsonl", manifest_rows)
        _atomic_jsonl(staging / "matrix_metrics.jsonl", rows)
        protected_after = {
            name: sha256_file(root / record["path"]) for name, record in plan["protected_inputs"].items()
        }
        if protected_after != protected_before:
            raise RuntimeError("E007 protected inputs changed during the audit")
        completed = _utc_now()
        counts_by_panel = {
            panel_name: sum(row["panel"] == panel_name for row in rows)
            for panel_name in sorted({row["panel"] for row in rows})
        }
        report = {
            "version": AUDIT_VERSION,
            "status": classification,
            "scientific_review_classification": classification,
            "recommendation": recommendation,
            "threshold_policy": config["audit"]["scientific_threshold_policy"],
            "no_scalar_quality_score": True,
            "generator": plan["generator"],
            "dataset": plan["dataset"],
            "configuration_path": str(config_file.relative_to(root)),
            "configuration_sha256": sha256_file(config_file),
            "matrix_representation": plan["matrix_representation"],
            "panel_identities": {"real_validation": panel_record, "generated_seed_schedule": schedule},
            "control_definitions": {
                "residue_permuted": "same deterministic permutation on both axes; Euclidean validity may be preserved",
                "explicit_corruptions": controls,
                "missing_valid_pairs": (
                    "not generated because the complete-square pair-mask contract does not support holes"
                ),
                "rank3_reconstruction": (
                    "mathematical classical-MDS audit reconstruction, not predicted or experimental coordinates"
                ),
            },
            "metric_definitions": {
                "source": "e007 distance_matrix_quality v1",
                "metrics": list(NUMERIC_METRICS),
                "triangle": "three inequalities per exact or deterministically sampled unordered triplet",
                "edm": "B=-0.5*J*(D squared)*J; exact or declared deterministic principal submatrix",
            },
            "counts": {
                "generated_candidates": len(generated_rows),
                "real_reference_samples": len(panel),
                "matrix_metric_rows": len(rows),
                "by_panel": counts_by_panel,
            },
            "fatal_generated_failures": generated_fatal,
            "quality_warnings": warnings,
            "aggregates": aggregates,
            "length_stratified_aggregates_included": True,
            "paired_control_statistics": paired,
            "aggregate_panel_comparisons": panel_comparisons,
            "candidate_diversity_is_per_matrix_uncertainty": False,
            "independent_candidates_averaged": False,
            "protected_inputs_before": protected_before,
            "protected_inputs_after": protected_after,
            "protected_inputs_unchanged": True,
            "phase1_e006_inventory_fingerprint": "26d41da3a2a34e9f76cc02d53c90bfcb0fad90d45487b79ea133e45a544953a7",
            "memory": _memory_telemetry(device),
            "training_performed": False,
            "backward_performed": False,
            "optimizer_created": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
        }
        protocol = {
            "version": AUDIT_VERSION,
            "status": "completed",
            "started_utc": started,
            "completed_utc": completed,
            "git_commit": _git_commit(root),
            "configuration_path": str(config_file.relative_to(root)),
            "configuration_sha256": sha256_file(config_file),
            "generator_checkpoint_sha256": config["generator"]["checkpoint_sha256"],
            "dataset_validation_manifest_sha256": config["dataset"]["validation_manifest_sha256"],
            "training_performed": False,
            "backward_performed": False,
            "optimizer_created": False,
            "authorizes_training": False,
            "authorizes_joint_training": False,
        }
        _publish_completed(staging, output, report=report, protocol=protocol)
        return output
    except BaseException as exc:
        if staging.exists():
            _atomic_json(
                staging / "heartbeat.json",
                {
                    "version": AUDIT_VERSION,
                    "status": "failed",
                    "failed_utc": _utc_now(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "completed_output_published": False,
                    "resumable": False,
                },
            )
        raise


def remove_synthetic_staging(path: str | Path) -> None:
    """Test helper for explicitly disposable synthetic staging directories."""
    shutil.rmtree(path)
