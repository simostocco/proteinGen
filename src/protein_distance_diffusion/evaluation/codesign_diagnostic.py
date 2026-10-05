"""Bounded, provenance-strict diagnostics for a completed E005-Large model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.sequence_geometry import (
    CANONICAL_AMINO_ACIDS,
    SequenceGeometryDataset,
    SequenceGeometryVocabulary,
    build_geometry_corruption,
    collate_sequence_geometry,
)
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.evaluation.repairability import (
    classical_mds_rank3_projection,
    trace_metrics,
)
from protein_distance_diffusion.models.codesign import GATE_ABLATION_DISABLED_PATHS, GATE_ABLATIONS
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.codesign import (
    _configuration_sha256,
    _gate_statistics,
    _sha256_file,
    masked_sequence_inputs,
)
from protein_distance_diffusion.training.codesign_large import (
    LARGE_ARCHITECTURE_VERSION,
    _dataset_identity,
    _model,
    validate_memory_telemetry,
)
from protein_distance_diffusion.training.codesign_pilot import _memory

DIAGNOSTIC_SCHEMA_VERSION = "e005_large_diagnostic_v1"
PANEL_SAMPLER_VERSION = "e005_diagnostic_stratified_backfill_v2"
EXPECTED_TRAJECTORY_STEPS = (
    0,
    2500,
    5000,
    7500,
    10000,
    12500,
    14000,
    15000,
    17500,
    20000,
    22500,
    25000,
    27500,
    28000,
    30000,
    32500,
    35000,
)
GATE_CONDITIONS = (
    "all_learned",
    "all_disabled",
    "sequence_to_geometry_disabled",
    "incoming_geometry_to_sequence_disabled",
    "returned_geometry_to_sequence_disabled",
    "both_geometry_to_sequence_disabled",
    "all_forced_one",
)


def validate_diagnostic_config(config: dict[str, Any]) -> None:
    """Validate the preregistered, bounded diagnostic design."""
    expected_masks = (0.15, 0.30, 0.50, 1.00)
    expected_noise = (0.0, 0.25, 0.50, 0.75, 1.00)
    if int(config.get("panel_size", 0)) != 1024 or int(config.get("stochastic_seeds_per_sample", 0)) != 3:
        raise ValueError("paper diagnostic requires 1,024 proteins and three seeds")
    if tuple(float(value) for value in config.get("mask_fractions", ())) != expected_masks:
        raise ValueError("paper diagnostic mask-fraction grid is incomplete")
    if tuple(float(value) for value in config.get("noise_levels", ())) != expected_noise:
        raise ValueError("paper diagnostic noise-level grid is incomplete")
    if tuple(config.get("geometry_conditions", ())) != ("clean", "configured_corrupted"):
        raise ValueError("paper diagnostic geometry conditions are incomplete")
    if tuple(config.get("gate_conditions", ())) != GATE_CONDITIONS or set(config["gate_conditions"]) != GATE_ABLATIONS:
        raise ValueError("diagnostic gate conditions are incomplete or reordered")
    if int(config.get("bootstrap_iterations", 0)) < 10_000:
        raise ValueError("paper diagnostic requires at least 10,000 bootstrap replicates")
    geometry_count = int(config.get("geometry_diagnostic_sample_count", 0))
    if not 1 <= geometry_count <= int(config["panel_size"]):
        raise ValueError("geometry diagnostic subset must be bounded by the panel")
    primary_mask = float(config.get("primary_mask_fraction", -1))
    primary_noise = float(config.get("primary_noise_level", -1))
    if primary_mask not in expected_masks or primary_noise not in expected_noise:
        raise ValueError("primary comparison condition must belong to the declared grid")
    max_rss = int(config.get("max_rss_mib", 0))
    max_cuda = int(config.get("max_cuda_memory_mib", 0))
    if not 0 < max_rss <= 6144 or not 0 < max_cuda <= 8192:
        raise ValueError("diagnostic memory limits must be positive and within production bounds")
    if config.get("device") != "cuda":
        raise ValueError("the definitive E005-Large diagnostic requires CUDA")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def streaming_amino_acid_frequencies(path: str | Path, *, batch_size: int = 4096) -> dict[str, Any]:
    """Count eligible training residues without materializing the manifest."""
    if not 1 <= batch_size <= 4096:
        raise ValueError("frequency batch_size must be in [1, 4096]")
    dataset = ds.dataset(str(path), format="parquet")
    columns = ["sequence"]
    if "practical_training_eligibility" in dataset.schema.names:
        columns.append("practical_training_eligibility")
    counts = Counter({residue: 0 for residue in CANONICAL_AMINO_ACIDS})
    rows = 0
    for batch in dataset.scanner(columns=columns, batch_size=batch_size, use_threads=False).to_batches():
        for row in batch.to_pylist():
            if str(row.get("practical_training_eligibility", "")) == "excluded_or_unresolved":
                raise ValueError("eligible_train contains an excluded row")
            sequence = str(row["sequence"])
            invalid = set(sequence) - set(CANONICAL_AMINO_ACIDS)
            if invalid:
                raise ValueError(f"noncanonical training sequence tokens: {sorted(invalid)}")
            counts.update(sequence)
            rows += 1
    total = sum(counts.values())
    if rows == 0 or total == 0:
        raise ValueError("eligible_train has no canonical sequences")
    probabilities = {residue: counts[residue] / total for residue in CANONICAL_AMINO_ACIDS}
    return {
        "sample_count": rows,
        "token_count": total,
        "counts": dict(counts),
        "probabilities": probabilities,
        "most_frequent_residue": max(CANONICAL_AMINO_ACIDS, key=lambda residue: (counts[residue], residue)),
    }


def sequence_baseline_metrics(validation_counts: dict[str, int], training_counts: dict[str, int]) -> dict[str, Any]:
    """Compute uniform and train-only unigram validation baselines."""
    total_validation = sum(validation_counts.values())
    total_training = sum(training_counts.values())
    if total_validation <= 0 or total_training <= 0:
        raise ValueError("baseline counts must contain training and validation tokens")
    probabilities = {residue: training_counts.get(residue, 0) / total_training for residue in CANONICAL_AMINO_ACIDS}
    if any(value <= 0 for value in probabilities.values()):
        raise ValueError("training unigram baseline requires positive frequency for every canonical residue")
    unigram_ce = -sum(
        validation_counts.get(residue, 0) * math.log(probabilities[residue]) for residue in CANONICAL_AMINO_ACIDS
    )
    unigram_ce /= total_validation
    most_frequent = max(CANONICAL_AMINO_ACIDS, key=lambda residue: (training_counts.get(residue, 0), residue))
    most_accuracy = validation_counts.get(most_frequent, 0) / total_validation
    return {
        "uniform_20_class": {
            "cross_entropy": math.log(20.0),
            "perplexity": 20.0,
            "expected_accuracy": 0.05,
            "expected_top_3_accuracy": 0.15,
            "expected_top_5_accuracy": 0.25,
        },
        "training_unigram": {
            "cross_entropy": unigram_ce,
            "perplexity": math.exp(unigram_ce),
            "probabilities": probabilities,
        },
        "most_frequent_residue": most_frequent,
        "most_frequent_residue_accuracy": most_accuracy,
        "training_unigram_top_k_accuracy": {
            f"top_{count}": sum(
                validation_counts.get(residue, 0)
                for residue in sorted(
                    CANONICAL_AMINO_ACIDS,
                    key=lambda residue: (-training_counts.get(residue, 0), residue),
                )[:count]
            )
            / total_validation
            for count in (1, 3, 5)
        },
        "validation_token_count": total_validation,
    }


def noise_level_to_timestep(level: float, diffusion_steps: int) -> int:
    """Map a normalized level to the configured schedule endpoints."""
    if not 0 <= level <= 1 or diffusion_steps < 2:
        raise ValueError("noise level must be in [0, 1] and diffusion_steps must exceed one")
    return int(round(level * (diffusion_steps - 1)))


def journal_fingerprint(sample_id: str, seed_index: int, seed: int, *, mask_probability: float, steps: int) -> str:
    """Fingerprint the deterministic historical validation-input protocol."""
    return _canonical_hash(
        {
            "sample_id": sample_id,
            "seed_index": seed_index,
            "stochastic_seed": seed,
            "mask_probability": mask_probability,
            "diffusion_steps": steps,
            "fingerprint_scope": "protocol_reconstructed_not_historical_tensor_hash",
        }
    )


def reconstruct_validation_trajectory(
    path: str | Path,
    *,
    expected_steps: Iterable[int],
    seed: int,
    mask_probability: float,
    diffusion_steps: int,
    plateau_relative_threshold: float,
    degradation_relative_threshold: float,
) -> dict[str, Any]:
    """Validate journal membership and classify its terminal learning trajectory."""
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    steps = [int(entry["optimizer_step"]) for entry in entries]
    expected = list(expected_steps)
    if steps != expected:
        raise ValueError(f"validation trajectory steps differ: observed={steps}, expected={expected}")
    reference_keys: set[tuple[str, str, int]] | None = None
    sample_indices: dict[str, int] = {}
    trajectory = []
    fingerprints = set()
    for entry in entries:
        records = entry.get("records", [])
        keys = {(str(row["sample_id"]), str(row["mode"]), int(row["seed_index"])) for row in records}
        if len(keys) != len(records):
            raise ValueError(f"duplicate validation records at step {entry['optimizer_step']}")
        if reference_keys is None:
            reference_keys = keys
            for row in records:
                sample_id = str(row["sample_id"])
                if sample_id not in sample_indices:
                    sample_indices[sample_id] = len(sample_indices)
        elif keys != reference_keys:
            raise ValueError(f"validation panel/mode/seed membership differs at step {entry['optimizer_step']}")
        learned = [row for row in records if row["mode"] == "learned_geometry_gating"]
        token_total = sum(int(row["token_count"]) for row in learned)
        sequence_loss = sum(float(row["sequence_loss"]) * int(row["token_count"]) for row in learned) / token_total
        trajectory.append({"optimizer_step": int(entry["optimizer_step"]), "sequence_loss": sequence_loss})
        seed_count = max((key[2] for key in keys), default=-1) + 1
        for sample_id, _, seed_index in keys:
            stochastic_seed = seed + 90_000_001 + sample_indices[sample_id] * seed_count + seed_index
            fingerprints.add(
                journal_fingerprint(
                    sample_id,
                    seed_index,
                    stochastic_seed,
                    mask_probability=mask_probability,
                    steps=diffusion_steps,
                )
            )
    losses = np.asarray([row["sequence_loss"] for row in trajectory], dtype=np.float64)
    best_index = int(np.argmin(losses))
    final = float(losses[-1])
    best = float(losses[best_index])
    recent_reference = float(losses[-2])
    recent_improvement = (recent_reference - final) / max(abs(recent_reference), 1e-12)
    if final > best * (1.0 + degradation_relative_threshold):
        classification = "degraded_after_best"
    elif recent_improvement >= plateau_relative_threshold:
        classification = "still_improving"
    else:
        classification = "plateaued"
    return {
        "steps": trajectory,
        "best_step": trajectory[best_index]["optimizer_step"],
        "best_sequence_loss": best,
        "final_sequence_loss": final,
        "recent_relative_improvement": recent_improvement,
        "classification": classification,
        "rule": {
            "degraded": f"final > best * (1 + {degradation_relative_threshold})",
            "improving": (
                f"not degraded and final improves over the prior scheduled point by >= {plateau_relative_threshold}"
            ),
            "otherwise": "plateaued",
        },
        "record_membership_comparable": True,
        "fingerprint_scope": "protocol_reconstructed_not_historical_tensor_hash",
        "fingerprint_count": len(fingerprints),
        "historical_tensor_hashes_available": False,
    }


def clustered_paired_bootstrap(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str,
    sample_key: str = "sample_id",
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap paired differences after averaging repeated seeds per protein."""
    if iterations < 10_000:
        raise ValueError("paper-quality bootstrap requires at least 10,000 replicates")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if math.isfinite(value):
            grouped[str(row[sample_key])].append(value)
    values = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64)
    if values.size < 2:
        raise ValueError("clustered bootstrap requires at least two protein samples")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    batch = 1000
    for start in range(0, iterations, batch):
        count = min(batch, iterations - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    standard_deviation = float(values.std(ddof=1))
    p_value = min(1.0, 2.0 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {
        "protein_count": int(values.size),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "standardized_effect_size": float(values.mean() / standard_deviation) if standard_deviation > 0 else 0.0,
        "two_sided_bootstrap_p_value": p_value,
        "bootstrap_iterations": iterations,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Holm family-wise adjusted p-values."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def geometry_diagnostics(
    matrix_normalized: np.ndarray, *, scale: float, triangle_samples: int, seed: int
) -> dict[str, Any]:
    """Measure numerical and rank-3 geometry in normalized and physical units."""
    normalized = np.asarray(matrix_normalized, dtype=np.float64)
    physical = normalized * float(scale)
    finite = bool(np.isfinite(normalized).all())
    symmetry_normalized = float(np.nanmax(np.abs(normalized - normalized.T)))
    diagonal_normalized = float(np.nanmax(np.abs(np.diag(normalized))))
    negative_count = int(np.sum(normalized < 0))
    if not finite:
        return {
            "finite": False,
            "symmetry_error_normalized": symmetry_normalized,
            "symmetry_error_angstrom": symmetry_normalized * scale,
            "diagonal_error_normalized": diagonal_normalized,
            "diagonal_error_angstrom": diagonal_normalized * scale,
            "negative_distance_count": negative_count,
            "triangle_violation_fraction": float("nan"),
            "negative_eigenvalue_mass_fraction": float("nan"),
            "coordinate_reconstruction_stress": float("nan"),
            "adjacent_distance_mean_normalized": float("nan"),
            "adjacent_distance_mean_angstrom": float("nan"),
            "adjacent_distance_std_normalized": float("nan"),
            "adjacent_distance_std_angstrom": float("nan"),
            "interpretation": "Non-finite prediction; no physical validity or foldability claim is possible.",
        }
    rng = np.random.default_rng(seed)
    if normalized.shape[0] >= 3:
        triples = rng.integers(0, normalized.shape[0], size=(triangle_samples, 3))
        tolerance = 1e-8 * max(float(np.max(np.abs(normalized))), 1.0)
        violations = (
            normalized[triples[:, 0], triples[:, 1]]
            > normalized[triples[:, 0], triples[:, 2]] + normalized[triples[:, 2], triples[:, 1]] + tolerance
        )
        triangle_rate = float(np.mean(violations))
    else:
        triangle_rate = 0.0
    projection = classical_mds_rank3_projection(physical)
    trace = trace_metrics(projection.coordinates)
    denominator = max(float(np.linalg.norm(physical)), 1e-12)
    stress = float(np.linalg.norm(projection.projected_distances - physical) / denominator)
    return {
        "finite": finite,
        "symmetry_error_normalized": symmetry_normalized,
        "symmetry_error_angstrom": symmetry_normalized * scale,
        "diagonal_error_normalized": diagonal_normalized,
        "diagonal_error_angstrom": diagonal_normalized * scale,
        "negative_distance_count": negative_count,
        "triangle_violation_fraction": triangle_rate,
        "negative_eigenvalue_mass_fraction": projection.negative_eigenvalue_mass_fraction,
        "coordinate_reconstruction_stress": stress,
        "adjacent_distance_mean_normalized": trace["ca_adjacent_distance_mean"] / scale,
        "adjacent_distance_mean_angstrom": trace["ca_adjacent_distance_mean"],
        "adjacent_distance_std_normalized": trace["ca_adjacent_distance_std"] / scale,
        "adjacent_distance_std_angstrom": trace["ca_adjacent_distance_std"],
        "interpretation": "Numerical/rank-3 diagnostics do not establish backbone validity or foldability.",
    }


def calibration_metrics(probabilities: np.ndarray, targets: np.ndarray, *, bins: int = 15) -> dict[str, float]:
    """Return multiclass Brier score and top-label expected calibration error."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    one_hot = np.eye(probabilities.shape[1])[targets]
    return {
        "top_label_ece": ece,
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
    }


def verify_effective_gate_condition(condition: str, gates: dict[str, dict[str, float]]) -> None:
    """Verify that evaluation-only clamps took effect on exactly their declared paths."""
    key_by_path = {
        "incoming_geometry_to_sequence": "geometry_to_sequence",
        "sequence_to_geometry": "sequence_to_geometry",
        "returned_geometry_to_sequence": "return_geometry_to_sequence",
    }
    for path in GATE_ABLATION_DISABLED_PATHS[condition]:
        values = gates[key_by_path[path]]
        if values["minimum"] != 0 or values["maximum"] != 0:
            raise RuntimeError(f"gate ablation failed to disable {path}")
    if condition == "all_forced_one":
        for path, key in key_by_path.items():
            values = gates[key]
            if values["minimum"] != 1 or values["maximum"] != 1:
                raise RuntimeError(f"gate ablation failed to force {path}")


@dataclass
class _AtomicParquetRows:
    final_path: Path
    buffer_size: int = 2048

    def __post_init__(self) -> None:
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.final_path.with_name(f".{self.final_path.name}.{os.getpid()}.tmp")
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.temporary_path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows.clear()

    def publish(self) -> None:
        self.flush()
        if self.writer is None:
            raise ValueError("cannot publish an empty sample-level table")
        self.writer.close()
        self.final_path.unlink(missing_ok=True)
        self.temporary_path.replace(self.final_path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.temporary_path.unlink(missing_ok=True)


def _topk_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    maximum = min(5, logits.shape[-1])
    indices = logits.topk(maximum, dim=-1).indices
    return {
        f"top_{value}_accuracy": float((indices[:, :value] == targets[:, None]).any(dim=1).float().mean())
        for value in (1, 3, 5)
    }


class PanelSelectionError(ValueError):
    """A structured failure to construct the requested validation panel."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__(f"insufficient_filtered_validation_population: {json.dumps(diagnostics, sort_keys=True)}")


def _panel_rank(seed: int, sample_id: str, *, purpose: str = "stratified") -> str:
    value = f"{seed}:{sample_id}" if purpose == "stratified" else f"{purpose}:{seed}:{sample_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def _panel_length_bin(length: int) -> str:
    for maximum in (64, 128, 256, 384, 500):
        if length <= maximum:
            return f"le_{maximum}"
    return "above_500"


def _dimension_counts(
    strata: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]],
) -> dict[str, dict[str, int]]:
    counts = {
        "length_bin": Counter(),
        "experimental_method": Counter(),
        "pairing_classification": Counter(),
    }
    for (length_bin, method, classification), rows in strata.items():
        counts["length_bin"][length_bin] += len(rows)
        counts["experimental_method"][method] += len(rows)
        counts["pairing_classification"][classification] += len(rows)
    return {name: dict(sorted(values.items())) for name, values in counts.items()}


def select_diagnostic_panel(
    path: str | Path,
    *,
    panel_size: int,
    seed: int,
    maximum_length: int = 500,
    batch_size: int = 4096,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a stable stratified panel and redistribute exhausted-stratum quota."""
    if panel_size < 1 or maximum_length < 1 or not 1 <= batch_size <= 4096:
        raise ValueError("panel size, maximum length, and Arrow batch size must be within bounds")
    dataset = ds.dataset(str(path), format="parquet")
    method_column = next((name for name in ("experimental_method", "method") if name in dataset.schema.names), None)
    class_column = next(
        (name for name in ("pairing_classification", "v3_pairing_classification") if name in dataset.schema.names),
        None,
    )
    required = {
        "sample_id",
        "schema_version",
        "sequence",
        "sequence_length",
        "matrix_length",
        "matrix_path",
        "practical_training_eligibility",
    }
    missing = sorted(required - set(dataset.schema.names))
    if missing:
        raise ValueError(f"validation manifest is missing panel column(s): {', '.join(missing)}")
    columns = sorted(required | {name for name in (method_column, class_column) if name is not None})
    rejected = {
        "missing_sample_id": 0,
        "invalid_sequence_length": 0,
        "length_above_maximum": 0,
        "ineligible": 0,
        "duplicate_sample_id": 0,
        "conflicting_duplicate_sample_id": 0,
    }
    candidates: dict[str, dict[str, Any]] = {}
    for batch in dataset.scanner(columns=columns, batch_size=batch_size, use_threads=False).to_batches():
        for source_row in batch.to_pylist():
            sample_id = str(source_row.get("sample_id") or "").strip()
            if not sample_id:
                rejected["missing_sample_id"] += 1
                continue
            try:
                length = int(source_row["sequence_length"])
            except (TypeError, ValueError):
                rejected["invalid_sequence_length"] += 1
                continue
            if length < 1:
                rejected["invalid_sequence_length"] += 1
                continue
            if length > maximum_length:
                rejected["length_above_maximum"] += 1
                continue
            if str(source_row.get("practical_training_eligibility") or "") == "excluded_or_unresolved":
                rejected["ineligible"] += 1
                continue
            row = dict(source_row)
            row["sample_id"] = sample_id
            row["experimental_method"] = str(row.get(method_column) or "unknown")
            row["pairing_classification"] = str(row.get(class_column) or "unknown")
            row["length_bin"] = _panel_length_bin(length)
            previous = candidates.get(sample_id)
            if previous is not None:
                rejected["duplicate_sample_id"] += 1
                if _canonical_hash(previous) != _canonical_hash(row):
                    rejected["conflicting_duplicate_sample_id"] += 1
                    raise ValueError(f"conflicting duplicate validation sample_id: {sample_id}")
                continue
            candidates[sample_id] = row

    strata: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for sample_id, row in candidates.items():
        key = (row["length_bin"], row["experimental_method"], row["pairing_classification"])
        strata[key].append((_panel_rank(seed, sample_id), row))
    for rows in strata.values():
        rows.sort(key=lambda item: (item[0], item[1]["sample_id"]))

    available_count = len(candidates)
    dimension_counts = _dimension_counts(strata)
    base_diagnostics = {
        "requested_panel_size": panel_size,
        "filtered_candidate_count": available_count,
        "filtered_candidate_row_count": available_count + rejected["duplicate_sample_id"],
        "rejected_counts": rejected,
        "available_counts_by_dimension": dimension_counts,
        "seed": seed,
        "sampler_version": PANEL_SAMPLER_VERSION,
    }
    if available_count < panel_size:
        raise PanelSelectionError(base_diagnostics)

    per_stratum_quota = max(2, panel_size // 32)
    initial_groups = {key: rows[:per_stratum_quota] for key, rows in strata.items()}
    selected: list[tuple[str, dict[str, Any]]] = []
    for position in range(max((len(rows) for rows in initial_groups.values()), default=0)):
        for key in sorted(initial_groups):
            group = initial_groups[key]
            if position < len(group) and len(selected) < panel_size:
                selected.append(group[position])
    initially_stratified_count = len(selected)
    selected_ids = {row["sample_id"] for _, row in selected}
    selected_by_stratum = Counter(
        (row["length_bin"], row["experimental_method"], row["pairing_classification"]) for _, row in selected
    )

    deficit = panel_size - len(selected)
    if deficit:
        underrepresented = sorted(
            strata,
            key=lambda key: (
                selected_by_stratum[key] / len(strata[key]),
                selected_by_stratum[key],
                key,
            ),
        )
        for key in underrepresented:
            candidate = next((item for item in strata[key] if item[1]["sample_id"] not in selected_ids), None)
            if candidate is not None:
                selected.append(candidate)
                selected_ids.add(candidate[1]["sample_id"])
                selected_by_stratum[key] += 1
                deficit -= 1
            if deficit == 0:
                break
    if deficit:
        remaining = [
            (_panel_rank(seed, row["sample_id"], purpose="global_backfill"), row)
            for rows in strata.values()
            for _, row in rows
            if row["sample_id"] not in selected_ids
        ]
        remaining.sort(key=lambda item: (item[0], item[1]["sample_id"]))
        for _, row in remaining[:deficit]:
            selected.append((_panel_rank(seed, row["sample_id"]), row))
            selected_ids.add(row["sample_id"])
            key = (row["length_bin"], row["experimental_method"], row["pairing_classification"])
            selected_by_stratum[key] += 1

    panel = [row for _, row in selected]
    if len(panel) != panel_size or len(selected_ids) != panel_size:
        raise RuntimeError("panel quota redistribution failed its exact-size invariant")
    per_stratum = [
        {
            "length_bin": key[0],
            "experimental_method": key[1],
            "pairing_classification": key[2],
            "available_count": len(strata[key]),
            "selected_count": selected_by_stratum[key],
        }
        for key in sorted(strata)
    ]
    diagnostics = {
        **base_diagnostics,
        "initially_stratified_count": initially_stratified_count,
        "backfilled_count": panel_size - initially_stratified_count,
        "final_panel_size": len(panel),
        "unique_sample_count": len(selected_ids),
        "per_stratum": per_stratum,
        "panel_sample_id_sha256": _canonical_hash([row["sample_id"] for row in panel]),
    }
    return panel, diagnostics


def _panel(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_directory = Path(config["dataset_directory"])
    rows, diagnostics = select_diagnostic_panel(
        dataset_directory / "eligible_validation.parquet",
        panel_size=int(config["panel_size"]),
        maximum_length=500,
        seed=int(config["seed"]),
    )
    panel_ids = {row["sample_id"] for row in rows}
    train = ds.dataset(str(dataset_directory / "eligible_train.parquet"), format="parquet")
    for batch in train.scanner(columns=["sample_id"], batch_size=4096, use_threads=False).to_batches():
        overlap = panel_ids & set(batch.column(0).to_pylist())
        if overlap:
            raise ValueError(f"diagnostic validation panel leaks training sample: {sorted(overlap)[0]}")
    diagnostics["training_overlap_count"] = 0
    return rows, diagnostics


def _forward_condition(
    config: dict[str, Any],
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    item: dict[str, Any],
    *,
    mask_fraction: float,
    noise_level: float,
    geometry_condition: str,
    gate_condition: str,
    seed: int,
    scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    batch = collate_sequence_geometry([item], pad_id=model.pad_token_id, pad_to_multiple=model.downsample_factor)
    tokens = batch["sequence_token_ids"].to(device)
    residue_mask = batch["sequence_mask"].to(device)
    pair_mask = (residue_mask[:, None, :, None] & residue_mask[:, None, None, :]).bool()
    clean = batch["distance_matrices"][:, None].to(device) / scale
    corruption_fingerprint = "clean"
    if geometry_condition == "configured_corrupted":
        corruption = build_geometry_corruption(config["geometry_corruption"])
        generator = torch.Generator(device="cpu").manual_seed(seed + 31)
        corrupted, corrupted_mask = corruption(batch["distance_matrices"][0], batch["pair_mask"][0], generator)
        clean = corrupted[None, None].to(device) / scale
        pair_mask = corrupted_mask[None, None].to(device)
        corruption_fingerprint = hashlib.sha256(corrupted.numpy().tobytes()).hexdigest()
    token_inputs, masked = masked_sequence_inputs(
        tokens,
        residue_mask,
        mask_token_id=model.mask_token_id,
        probability=mask_fraction,
        seed=seed,
        step=0,
    )
    timestep = noise_level_to_timestep(noise_level, diffusion.timesteps)
    timesteps = torch.tensor([timestep], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 47)
    noisy, epsilon = diffusion.q_sample(clean, timesteps, pair_mask, generator=generator)
    lengths = batch["lengths"].to(device)
    separation = make_sequence_separation(batch["lengths"], noisy.shape[-1]).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(
                sequence_token_ids=token_inputs,
                residue_mask=residue_mask,
                noisy_geometry=noisy,
                timesteps=timesteps,
                lengths=lengths,
                sequence_separation=separation,
                pair_mask=pair_mask,
                geometry_conditioning_mask=torch.ones(1, dtype=torch.bool, device=device),
                mode="learned_geometry_gating",
                gate_ablation=gate_condition,
            )
    selected_logits = output["sequence_logits"][masked].float()
    selected_targets = tokens[masked]
    cross_entropy = torch.nn.functional.cross_entropy(selected_logits, selected_targets)
    probabilities = selected_logits.softmax(dim=-1)
    topk = _topk_metrics(selected_logits, selected_targets)
    prediction_type = str(config["training_configuration"]["diffusion"]["prediction_parameterization"])
    x0, _ = diffusion.predict_x0_epsilon_from_model_output(
        x_t=noisy,
        t=timesteps,
        model_output=output["geometry_prediction"],
        prediction_type=prediction_type,
    )
    upper = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
    valid_pairs = pair_mask & upper
    geometry_error = float(((x0 - clean).square() * valid_pairs).sum() / valid_pairs.sum().clamp_min(1))
    consistency_loss = float(
        (
            (output["geometry_prediction"].float() - output["sequence_pair_prediction"].float()).square() * valid_pairs
        ).sum()
        / valid_pairs.sum().clamp_min(1)
    )
    gates = _gate_statistics(
        output["geometry_to_sequence_gate"],
        output["sequence_to_geometry_gate"],
        output["return_geometry_gate"],
        residue_mask,
    )
    verify_effective_gate_condition(gate_condition, gates)
    mask_hash = hashlib.sha256(masked.detach().cpu().numpy().tobytes()).hexdigest()
    noise_hash = hashlib.sha256(epsilon.detach().cpu().numpy().tobytes()).hexdigest()
    row = {
        "sample_id": str(item["sample_id"]),
        "length": int(batch["lengths"][0]),
        "mask_fraction": mask_fraction,
        "noise_level": noise_level,
        "timestep": timestep,
        "geometry_condition": geometry_condition,
        "gate_condition": gate_condition,
        "stochastic_seed": seed,
        "mask_sha256": mask_hash,
        "noise_sha256": noise_hash,
        "corruption_sha256": corruption_fingerprint,
        "token_count": int(masked.sum()),
        "sequence_cross_entropy": float(cross_entropy),
        "sequence_perplexity": float(torch.exp(cross_entropy).clamp_max(1e12)),
        **topk,
        "geometry_denoising_error": geometry_error,
        "geometry_x0_rmse_angstrom": math.sqrt(max(geometry_error, 0.0)) * scale,
        "cross_branch_consistency_loss": consistency_loss,
        "gate_statistics_json": json.dumps(gates, sort_keys=True),
        "intervened_gate_paths_json": json.dumps(sorted(GATE_ABLATION_DISABLED_PATHS[gate_condition])),
    }
    return (
        row,
        probabilities.detach().cpu().numpy(),
        selected_targets.detach().cpu().numpy(),
        x0[0, 0, : int(lengths[0]), : int(lengths[0])].detach().cpu().numpy(),
    )


def run_codesign_diagnostic(config_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Run the bounded E005-Large diagnostic and publish only derived artifacts."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    validate_diagnostic_config(config)
    destination = Path(output_dir or config["output_dir"])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {destination}")
    destination.mkdir(parents=True)
    protocol_path = destination / "protocol.json"
    started = time.monotonic()
    checkpoint_path = Path(config["checkpoint_path"])
    summary_path = Path(config["training_summary_path"])
    journal_path = Path(config["training_validation_journal"])
    training_config_path = Path(config["training_configuration_path"])
    training_configuration = load_yaml(training_config_path)
    config["training_configuration"] = training_configuration
    device = torch.device(config["device"])
    protected = [checkpoint_path, summary_path, journal_path, training_config_path]
    protected_before = {str(path): _sha256_file(path) for path in protected}
    dataset_config = {
        **training_configuration,
        "dataset": {**training_configuration["dataset"], "directory": config["dataset_directory"]},
    }
    dataset_before = _dataset_identity(dataset_config, synthetic=False)
    dataset_metadata_hashes = {
        name: _sha256_file(Path(config["dataset_directory"]) / name)
        for name in ("protocol.json", "schema.json", "vocabulary.json")
    }
    report_base = {
        "status": "running",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": protected_before[str(checkpoint_path)],
        "configuration_path": str(config_path),
        "configuration_sha256": _sha256_file(config_path),
        "training_configuration_path": str(training_config_path),
        "training_configuration_sha256": protected_before[str(training_config_path)],
        "dataset_identity_before": dataset_before,
        "dataset_metadata_hashes": dataset_metadata_hashes,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
    }
    writer = _AtomicParquetRows(destination / "sample_level.parquet")
    optimizer_step: int | None = None
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        training_summary = json.loads(summary_path.read_text())
        if training_summary.get("status") != "completed" or int(training_summary.get("optimizer_steps", -1)) != 35_000:
            raise ValueError("training summary is incomplete or does not attest 35,000 optimizer steps")
        if checkpoint.get("architecture_version") != LARGE_ARCHITECTURE_VERSION:
            raise ValueError("checkpoint architecture is not E005-Large")
        if checkpoint.get("config_sha256") != _configuration_sha256(training_configuration):
            raise ValueError("checkpoint/training configuration hash contradiction")
        if checkpoint.get("dataset_sha256") != dataset_before["sha256"]:
            raise ValueError("checkpoint/dataset hash contradiction")
        optimizer_step = int(checkpoint["optimizer_step"])
        model = _model(training_configuration)
        model.load_state_dict(checkpoint["model"])
        del checkpoint
        model.to(device)
        model.eval()
        panel, panel_selection = _panel(config)
        _atomic_json(
            destination / "panel.json",
            {
                "definition": {
                    "dataset": "eligible_validation",
                    "size": len(panel),
                    "strata": ["length_bin", "experimental_method", "pairing_classification"],
                    "seed": config["seed"],
                },
                "selection_diagnostics": panel_selection,
                "samples": [
                    {
                        "sample_id": row["sample_id"],
                        "length": row["sequence_length"],
                        "experimental_method": row["experimental_method"],
                        "pairing_classification": row["pairing_classification"],
                        "matrix_sha256": (matrix_sha256 := _sha256_file(Path(row["matrix_path"]))),
                        "fingerprint": _canonical_hash({"row": row, "matrix_sha256": matrix_sha256}),
                    }
                    for row in panel
                ],
            },
        )
        train_frequency = streaming_amino_acid_frequencies(Path(config["dataset_directory"]) / "eligible_train.parquet")
        trajectory = reconstruct_validation_trajectory(
            journal_path,
            expected_steps=EXPECTED_TRAJECTORY_STEPS,
            seed=int(training_configuration["seed"]),
            mask_probability=float(training_configuration["masked_token_probability"]),
            diffusion_steps=int(training_configuration["diffusion"]["steps"]),
            plateau_relative_threshold=float(config["trajectory"]["plateau_relative_threshold"]),
            degradation_relative_threshold=float(config["trajectory"]["degradation_relative_threshold"]),
        )
        scale = float(json.loads(Path(training_configuration["normalization_file"]).read_text())["scale"])
        diffusion = GaussianDiffusion(cosine_beta_schedule(int(training_configuration["diffusion"]["steps"]))).to(
            device
        )
        validation_counts = Counter({residue: 0 for residue in CANONICAL_AMINO_ACIDS})
        primary_gate: list[dict[str, Any]] = []
        primary_oracle: list[dict[str, Any]] = []
        calibration_probabilities = []
        calibration_targets = []
        model_baseline_rows = []
        geometry_rows = []
        evaluated = 0
        for panel_index, row in enumerate(panel):
            item = SequenceGeometryDataset(
                pa.Table.from_pylist([row]), mode="geometry_conditioned", seed=config["seed"]
            )[0]
            sequence = SequenceGeometryVocabulary().decode(item["sequence_token_ids"])
            validation_counts.update(sequence)
            for seed_index in range(int(config["stochastic_seeds_per_sample"])):
                stochastic_seed = int(config["seed"]) + panel_index * 10_007 + seed_index
                for geometry_condition in config["geometry_conditions"]:
                    for mask_fraction in config["mask_fractions"]:
                        for noise_level in config["noise_levels"]:
                            result, probabilities, targets, predicted = _forward_condition(
                                config,
                                model,
                                diffusion,
                                item,
                                mask_fraction=float(mask_fraction),
                                noise_level=float(noise_level),
                                geometry_condition=geometry_condition,
                                gate_condition="all_learned",
                                seed=stochastic_seed,
                                scale=scale,
                                device=device,
                            )
                            result.update(
                                experimental_method=row["experimental_method"],
                                pairing_classification=row["pairing_classification"],
                                length_bin=row["length_bin"],
                                evaluation_family="oracle_grid",
                            )
                            writer.append(result)
                            evaluated += 1
                            if float(mask_fraction) == float(config["primary_mask_fraction"]) and float(
                                noise_level
                            ) == float(config["primary_noise_level"]):
                                primary_oracle.append(dict(result))
                                if geometry_condition == "clean":
                                    calibration_probabilities.append(probabilities)
                                    calibration_targets.append(targets)
                                    model_baseline_rows.append(dict(result))
                            if (
                                panel_index < int(config["geometry_diagnostic_sample_count"])
                                and seed_index == 0
                                and float(mask_fraction) == float(config["primary_mask_fraction"])
                            ):
                                geometry_rows.append(
                                    {
                                        **{
                                            key: result[key]
                                            for key in (
                                                "sample_id",
                                                "noise_level",
                                                "geometry_condition",
                                                "stochastic_seed",
                                            )
                                        },
                                        **geometry_diagnostics(
                                            predicted,
                                            scale=scale,
                                            triangle_samples=int(config["geometry_triangle_samples"]),
                                            seed=stochastic_seed,
                                        ),
                                    }
                                )
                for gate_condition in GATE_CONDITIONS:
                    result, _, _, _ = _forward_condition(
                        config,
                        model,
                        diffusion,
                        item,
                        mask_fraction=float(config["primary_mask_fraction"]),
                        noise_level=float(config["primary_noise_level"]),
                        geometry_condition="clean",
                        gate_condition=gate_condition,
                        seed=stochastic_seed,
                        scale=scale,
                        device=device,
                    )
                    result.update(
                        experimental_method=row["experimental_method"],
                        pairing_classification=row["pairing_classification"],
                        length_bin=row["length_bin"],
                        evaluation_family="gate_ablation",
                    )
                    writer.append(result)
                    evaluated += 1
                    if gate_condition in {"all_learned", "both_geometry_to_sequence_disabled"}:
                        primary_gate.append(dict(result))
            memory = _memory(device)
            validate_memory_telemetry(
                memory,
                cuda=device.type == "cuda",
                limits={
                    "current_rss_mib": float(config["max_rss_mib"]),
                    "peak_rss_mib": float(config["max_rss_mib"]),
                    **(
                        {
                            "cuda_allocated_mib": float(config["max_cuda_memory_mib"]),
                            "cuda_reserved_mib": float(config["max_cuda_memory_mib"]),
                            "peak_cuda_allocated_mib": float(config["max_cuda_memory_mib"]),
                            "peak_cuda_reserved_mib": float(config["max_cuda_memory_mib"]),
                        }
                        if device.type == "cuda"
                        else {}
                    ),
                },
                strict=False,
            )
        writer.publish()
        expected_records = (
            len(panel)
            * int(config["stochastic_seeds_per_sample"])
            * (
                len(config["geometry_conditions"]) * len(config["mask_fractions"]) * len(config["noise_levels"])
                + len(config["gate_conditions"])
            )
        )
        expected_geometry_records = (
            int(config["geometry_diagnostic_sample_count"])
            * len(config["geometry_conditions"])
            * len(config["noise_levels"])
        )
        if evaluated != expected_records or len(geometry_rows) != expected_geometry_records:
            raise RuntimeError("diagnostic output count contract failed")
        pq.write_table(
            pa.Table.from_pylist(geometry_rows), destination / ".geometry_diagnostics.parquet.tmp", compression="zstd"
        )
        (destination / ".geometry_diagnostics.parquet.tmp").replace(destination / "geometry_diagnostics.parquet")
        baseline = sequence_baseline_metrics(dict(validation_counts), train_frequency["counts"])
        model_tokens = sum(row["token_count"] for row in model_baseline_rows)
        model_cross_entropy = (
            sum(row["sequence_cross_entropy"] * row["token_count"] for row in model_baseline_rows) / model_tokens
        )
        baseline["model_primary_clean_condition"] = {
            "mask_fraction": 0.15,
            "noise_level": 0.5,
            "token_count": model_tokens,
            "token_weighted_cross_entropy": model_cross_entropy,
            "perplexity": math.exp(min(model_cross_entropy, 27.0)),
            **{
                key: sum(row[key] * row["token_count"] for row in model_baseline_rows) / model_tokens
                for key in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy")
            },
        }
        baseline["model_calibration"] = calibration_metrics(
            np.concatenate(calibration_probabilities), np.concatenate(calibration_targets)
        )
        gate_by_key = defaultdict(dict)
        for row in primary_gate:
            key = (row["sample_id"], row["stochastic_seed"])
            gate_by_key[key][row["gate_condition"]] = row
        gate_differences = [
            {
                "sample_id": key[0],
                "difference": values["all_learned"]["sequence_cross_entropy"]
                - values["both_geometry_to_sequence_disabled"]["sequence_cross_entropy"],
            }
            for key, values in gate_by_key.items()
            if set(values) == {"all_learned", "both_geometry_to_sequence_disabled"}
            and values["all_learned"]["mask_sha256"] == values["both_geometry_to_sequence_disabled"]["mask_sha256"]
            and values["all_learned"]["noise_sha256"] == values["both_geometry_to_sequence_disabled"]["noise_sha256"]
        ]
        oracle_by_key = defaultdict(dict)
        for row in primary_oracle:
            key = (row["sample_id"], row["stochastic_seed"])
            oracle_by_key[key][row["geometry_condition"]] = row
        oracle_differences = [
            {
                "sample_id": key[0],
                "difference": values["clean"]["sequence_cross_entropy"]
                - values["configured_corrupted"]["sequence_cross_entropy"],
            }
            for key, values in oracle_by_key.items()
            if set(values) == {"clean", "configured_corrupted"}
            and values["clean"]["mask_sha256"] == values["configured_corrupted"]["mask_sha256"]
            and values["clean"]["noise_sha256"] == values["configured_corrupted"]["noise_sha256"]
        ]
        expected_primary = len(panel) * int(config["stochastic_seeds_per_sample"])
        if len(gate_differences) != expected_primary or len(oracle_differences) != expected_primary:
            raise ValueError("primary comparison fingerprints are incomplete or non-comparable")
        bootstrap_iterations = int(config["bootstrap_iterations"])
        comparisons = {
            "learned_minus_both_geometry_to_sequence_disabled": clustered_paired_bootstrap(
                gate_differences,
                value_key="difference",
                iterations=bootstrap_iterations,
                seed=int(config["bootstrap_seed"]),
            ),
            "clean_minus_configured_corrupted": clustered_paired_bootstrap(
                oracle_differences,
                value_key="difference",
                iterations=bootstrap_iterations,
                seed=int(config["bootstrap_seed"]) + 1,
            ),
        }
        adjusted = holm_adjust(
            {name: float(value["two_sided_bootstrap_p_value"]) for name, value in comparisons.items()}
        )
        for name, value in adjusted.items():
            comparisons[name]["holm_adjusted_p_value"] = value
        effect_figure, effect_axis = plt.subplots(figsize=(8, 4.5))
        comparison_names = list(comparisons)
        means = [float(comparisons[name]["mean_difference"]) for name in comparison_names]
        lower = [means[index] - float(comparisons[name]["ci95_low"]) for index, name in enumerate(comparison_names)]
        upper = [float(comparisons[name]["ci95_high"]) - means[index] for index, name in enumerate(comparison_names)]
        effect_axis.errorbar(range(len(means)), means, yerr=[lower, upper], fmt="o", capsize=5)
        effect_axis.axhline(0.0, color="black", linewidth=1)
        effect_axis.set_xticks(range(len(means)), ["Gate-path effect", "Clean-oracle effect"])
        effect_axis.set(ylabel="Paired sequence cross-entropy difference", title="E005-Large primary effects")
        effect_axis.grid(axis="y", alpha=0.25)
        _atomic_figure(effect_figure, destination / "primary_sequence_effects.png")
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(
            [row["optimizer_step"] for row in trajectory["steps"]],
            [row["sequence_loss"] for row in trajectory["steps"]],
            marker="o",
        )
        axis.set(
            xlabel="Optimizer step",
            ylabel="Token-weighted sequence cross-entropy",
            title="E005-Large held-out trajectory",
        )
        axis.grid(alpha=0.25)
        _atomic_figure(figure, destination / "validation_trajectory.png")
        dataset_after = _dataset_identity(dataset_config, synthetic=False)
        protected_after = {str(path): _sha256_file(path) for path in protected}
        if dataset_after != dataset_before or protected_after != protected_before:
            raise RuntimeError("evaluation_input_mutation_detected")
        memory = _memory(device)
        derived_artifacts = [
            destination / "panel.json",
            destination / "sample_level.parquet",
            destination / "geometry_diagnostics.parquet",
            destination / "validation_trajectory.png",
            destination / "primary_sequence_effects.png",
        ]
        report = {
            **report_base,
            "status": "completed",
            "optimizer_step": optimizer_step,
            "dataset_identity_after": dataset_after,
            "dataset_metadata_hashes": dataset_metadata_hashes,
            "dataset_inputs_unchanged": True,
            "protected_input_hashes_before": protected_before,
            "protected_input_hashes_after": protected_after,
            "panel_definition": {
                "dataset": "eligible_validation",
                "sample_count": len(panel),
                "stochastic_seeds_per_sample": config["stochastic_seeds_per_sample"],
                "stratification": ["length_bin", "experimental_method", "pairing_classification"],
                "panel_path": str(destination / "panel.json"),
            },
            "panel_selection": panel_selection,
            "training_unigram": train_frequency,
            "sequence_baselines": baseline,
            "validation_trajectory": trajectory,
            "primary_comparisons": comparisons,
            "sample_level_path": str(destination / "sample_level.parquet"),
            "sample_level_record_count": evaluated,
            "geometry_diagnostics_path": str(destination / "geometry_diagnostics.parquet"),
            "geometry_diagnostic_record_count": len(geometry_rows),
            "derived_artifact_hashes": {str(path): _sha256_file(path) for path in derived_artifacts},
            "memory": memory,
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "failure": None,
            "authorizes_decisions": True,
            "scientific_limitations": [
                "Denoising and rank-3 diagnostics do not establish backbone validity or foldability.",
                "Historical validation tensor hashes were not stored; trajectory fingerprints "
                "are protocol-reconstructed.",
            ],
        }
        _atomic_json(protocol_path, report)
        return report
    except BaseException as error:
        writer.abort()
        try:
            dataset_after = _dataset_identity(dataset_config, synthetic=False)
        except BaseException:
            dataset_after = None
        protected_after = {str(path): _sha256_file(path) if path.exists() else None for path in protected}
        failure = {
            **report_base,
            "status": "failed",
            "optimizer_step": optimizer_step,
            "dataset_identity_after": dataset_after,
            "dataset_inputs_unchanged": dataset_after == dataset_before,
            "protected_input_hashes_before": protected_before,
            "protected_input_hashes_after": protected_after,
            "elapsed_seconds": time.monotonic() - started,
            "memory": _memory(device),
            "failure": {"type": type(error).__name__, "message": str(error)},
            "completed_utc": datetime.now(UTC).isoformat(),
            "authorizes_decisions": False,
        }
        _atomic_json(protocol_path, failure)
        raise
