"""Bounded construction of immutable E006 rich-geometry residue sidecars."""

from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import os
import resource
import signal
import sqlite3
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryVocabulary
from protein_distance_diffusion.evaluation.e006_geometry_source_audit import (
    AUDIT_SCHEMA_VERSION,
    COORDINATE_ANCHOR_POLICY_VERSION,
    _alternative_identifier_inputs,
    analyze_backbone_residues,
    dihedral_angle,
    enrich_source_locators,
    frame_quality,
    load_npz_geometry_evidence,
    local_frame,
    parse_backbone_mmcif,
    pseudo_cb_coordinate,
    reconcile_npz_calpha_anchors,
    residue_geometry_diagnostics,
    resolve_rich_geometry_eligibility,
    resolve_verified_target_mapping,
    resolve_within_roots,
    sha256_file,
)

SIDECAR_SCHEMA_VERSION = "e006_rich_geometry_sidecar_v2"
TORSION_CONVENTION_VERSION = "e006_backbone_torsion_sincos_v1"
TORSION_NEUTRAL_SIN_COS = (0.0, 1.0)
TORSION_FLOAT32_ATOL = 2e-6
CUDA_FEATURE_FLOAT32_ATOL = 5e-5
JOURNAL_SCHEMA_VERSION = "e006_rich_geometry_construction_journal_v3"
EXPECTED_AUDIT_SCHEMA = "e006_rich_geometry_source_audit_v6"
ATOM_NAMES = ("N", "CA", "C", "O", "CB")
TORSION_DEFINITIONS = {
    "phi": (("C", -1), ("N", 0), ("CA", 0), ("C", 0)),
    "psi": (("N", 0), ("CA", 0), ("C", 0), ("N", 1)),
    "omega": (("CA", -1), ("C", -1), ("N", 0), ("CA", 0)),
}
INPUT_DATASET_SPLITS = {
    "eligible_train": "train",
    "eligible_validation": "validation",
}
SPLIT_LIKE_FIELDS = ("split", "dataset_split", "split_name")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "eligible_train": "train",
    "eligible_train.parquet": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "eligible_validation": "validation",
    "eligible_validation.parquet": "validation",
}
PERFORMANCE_STAGE_NAMES = (
    "arrow_row_retrieval",
    "mmcif_gzip_reading",
    "gemmi_parsing",
    "npz_loading",
    "residue_mapping",
    "ca_anchor_resolution",
    "frame_torsion_cb_feature_derivation",
    "parquet_encoding_writing",
    "hashing",
    "final_verification",
    "cuda_transfer",
    "cuda_compute",
)


@dataclass(frozen=True)
class StageObservation:
    stage: str
    seconds: float
    sample_count: int = 1
    bytes_read: int = 0
    bytes_written: int = 0


class StageProfiler:
    """Bounded deterministic latency profiler for construction and benchmark runs."""

    def __init__(self, maximum_latency_samples: int = 8192) -> None:
        self.maximum_latency_samples = int(maximum_latency_samples)
        self.totals: dict[str, dict[str, float | int]] = defaultdict(
            lambda: {
                "total_seconds": 0.0,
                "sample_count": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "observation_count": 0,
            }
        )
        self.latencies: dict[str, list[float]] = defaultdict(list)

    def add(self, observation: StageObservation) -> None:
        if observation.stage not in PERFORMANCE_STAGE_NAMES:
            raise ValueError(f"Unknown E006 performance stage: {observation.stage}")
        if observation.seconds < 0 or observation.sample_count < 0:
            raise ValueError("Stage timing values must be nonnegative")
        total = self.totals[observation.stage]
        total["total_seconds"] += float(observation.seconds)
        total["sample_count"] += int(observation.sample_count)
        total["bytes_read"] += int(observation.bytes_read)
        total["bytes_written"] += int(observation.bytes_written)
        total["observation_count"] += 1
        latencies = self.latencies[observation.stage]
        if len(latencies) < self.maximum_latency_samples:
            divisor = max(int(observation.sample_count), 1)
            latencies.append(float(observation.seconds) / divisor)

    def extend(self, observations: tuple[StageObservation, ...] | list[StageObservation]) -> None:
        for observation in observations:
            self.add(observation)

    def report(self, elapsed_seconds: float) -> dict[str, dict[str, float | int | bool]]:
        result = {}
        denominator = max(float(elapsed_seconds), 1e-12)
        for stage in PERFORMANCE_STAGE_NAMES:
            total = self.totals[stage]
            latencies = np.asarray(self.latencies[stage], dtype=np.float64)
            result[stage] = {
                **total,
                "percentage_of_runtime": 100.0 * float(total["total_seconds"]) / denominator,
                "mean_latency_seconds": (
                    float(total["total_seconds"]) / int(total["sample_count"]) if int(total["sample_count"]) else 0.0
                ),
                "median_latency_seconds": float(np.median(latencies)) if latencies.size else 0.0,
                "p95_latency_seconds": float(np.quantile(latencies, 0.95)) if latencies.size else 0.0,
                "latency_sample_count": int(latencies.size),
                "latency_samples_truncated": int(total["observation_count"]) > len(latencies),
            }
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_latency_samples": self.maximum_latency_samples,
            "totals": {stage: dict(values) for stage, values in self.totals.items()},
            "latencies": {stage: list(values) for stage, values in self.latencies.items()},
        }

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> StageProfiler:
        profiler = cls(int(value["maximum_latency_samples"]))
        for stage, totals in value.get("totals", {}).items():
            if stage not in PERFORMANCE_STAGE_NAMES:
                raise ValueError(f"Unknown persisted E006 performance stage: {stage}")
            profiler.totals[stage] = dict(totals)
        for stage, latencies in value.get("latencies", {}).items():
            if stage not in PERFORMANCE_STAGE_NAMES:
                raise ValueError(f"Unknown persisted E006 performance stage: {stage}")
            profiler.latencies[stage] = [float(item) for item in latencies]
        return profiler


@dataclass(frozen=True)
class EvaluationOutcome:
    row: dict[str, Any]
    decision: dict[str, Any] | None
    sidecar: dict[str, Any] | None
    observations: tuple[StageObservation, ...]
    error: BaseException | None


class CaseMemorySampler:
    """Measure case-local RSS without inheriting an earlier case high-water mark."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = float(interval_seconds)
        self.started_current_rss_mib = _rss_mib()
        self.case_peak_rss_mib = self.started_current_rss_mib
        self.measurement_count = 1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True, name="e006-case-rss")

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            current = _rss_mib()
            self.case_peak_rss_mib = max(self.case_peak_rss_mib, current)
            self.measurement_count += 1

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict[str, float | int]:
        self._stop.set()
        self._thread.join()
        current = _rss_mib()
        self.case_peak_rss_mib = max(self.case_peak_rss_mib, current)
        self.measurement_count += 1
        return {
            "current_rss_mib_before_case": self.started_current_rss_mib,
            "current_rss_mib_after_case": current,
            "case_peak_rss_mib": self.case_peak_rss_mib,
            "process_peak_rss_mib": _peak_rss_mib(),
            "rss_measurement_count": self.measurement_count,
        }


def _evaluate_with_timings(
    row: dict[str, Any],
    config: dict[str, Any],
    canonical_split: str,
) -> EvaluationOutcome:
    observations: list[StageObservation] = []

    def observe(stage: str, seconds: float, bytes_read: int, bytes_written: int) -> None:
        observations.append(
            StageObservation(
                stage,
                seconds,
                sample_count=1,
                bytes_read=bytes_read,
                bytes_written=bytes_written,
            )
        )

    try:
        decision, sidecar = evaluate_sample(
            row,
            config,
            canonical_split=canonical_split,
            stage_observer=observe,
        )
        return EvaluationOutcome(row, decision, sidecar, tuple(observations), None)
    except (OSError, RuntimeError, ValueError) as error:
        return EvaluationOutcome(row, None, None, tuple(observations), error)


def _ordered_evaluations(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    canonical_split: str,
) -> Iterator[EvaluationOutcome]:
    worker_count = int(config.get("worker_count", 1))
    queue_size = int(config.get("worker_queue_size", max(2, worker_count * 2)))
    if worker_count == 1:
        for row in rows:
            yield _evaluate_with_timings(row, config, canonical_split)
        return
    pending: deque[Future[EvaluationOutcome]] = deque()
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="e006-sidecar") as executor:
        for row in rows:
            pending.append(executor.submit(_evaluate_with_timings, row, config, canonical_split))
            if len(pending) >= queue_size:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _cuda_dihedral_sincos(torch: Any, a: Any, b: Any, c: Any, d: Any) -> tuple[Any, Any]:
    b0 = b - a
    b1 = c - b
    b2 = d - c
    b1_norm = torch.linalg.vector_norm(b1, dim=-1)
    safe_b1 = b1 / b1_norm.clamp_min(1e-8).unsqueeze(-1)
    v = b0 - (b0 * safe_b1).sum(dim=-1, keepdim=True) * safe_b1
    w = b2 - (b2 * safe_b1).sum(dim=-1, keepdim=True) * safe_b1
    valid = (
        (b1_norm >= 1e-8)
        & (torch.linalg.vector_norm(v, dim=-1) >= 1e-8)
        & (torch.linalg.vector_norm(w, dim=-1) >= 1e-8)
    )
    angle = torch.atan2((torch.linalg.cross(safe_b1, v, dim=-1) * w).sum(dim=-1), (v * w).sum(dim=-1))
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1), valid


def _cuda_pseudo_cb_contradiction(
    row: dict[str, Any],
    *,
    residue_index: int,
    cb_source: int,
    observed: np.ndarray,
    input_dtype: np.dtype[Any],
) -> dict[str, Any] | None:
    if cb_source != 0:
        return None
    stored = np.asarray(row["cb_coordinates"][residue_index], dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    absolute_error = np.abs(stored - observed)
    if np.allclose(stored, observed, atol=CUDA_FEATURE_FLOAT32_ATOL, rtol=0):
        return None
    return {
        "sample_id": row["sample_id"],
        "residue_index": residue_index,
        "residue_id": row["residue_ids"][residue_index],
        "cb_source": int(cb_source),
        "cb_source_meaning": "pseudo_cb",
        "native_cb_compared": False,
        "backbone_atom_masks": {atom: bool(row[f"{atom.lower()}_mask"][residue_index]) for atom in ("N", "CA", "C")},
        "stored_cpu_pseudo_cb": stored.tolist(),
        "cuda_pseudo_cb": observed.tolist(),
        "absolute_coordinate_error": absolute_error.tolist(),
        "maximum_coordinate_error": float(absolute_error.max()),
        "cpu_dtype": str(stored.dtype),
        "cuda_result_dtype": str(observed.dtype),
        "comparison_absolute_tolerance": CUDA_FEATURE_FLOAT32_ATOL,
        "input_coordinate_dtype": str(input_dtype),
        "diagnosis": (
            "stored CPU pseudo-CB was derived before float32 coordinate serialization; "
            "CUDA diagnostic recomputation uses serialized float32 backbone coordinates"
        ),
    }


def _apply_cuda_feature_batching(
    sidecars: list[dict[str, Any]],
    config: dict[str, Any],
    profiler: StageProfiler,
) -> tuple[str, str | None, float | None]:
    """Re-derive vectorizable features in residue-budgeted CUDA batches."""
    if config.get("feature_backend", "cpu") != "cuda":
        return "cpu", None, None
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu", "pytorch_unavailable", None
    if not torch.cuda.is_available():
        return "cpu", "cuda_unavailable", None
    torch.cuda.reset_peak_memory_stats()
    token_budget = int(config.get("cuda_residue_token_budget", 8192))
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for row in sidecars:
        length = len(row["sequence"])
        if current and current_tokens + length > token_budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += length
    if current:
        chunks.append(current)
    peak_allocated = 0.0
    for chunk in chunks:
        maximum_length = max(len(row["sequence"]) for row in chunk)
        batch_size = len(chunk)
        coordinates = np.zeros((batch_size, maximum_length, 3, 3), dtype=np.float32)
        masks = np.zeros((batch_size, maximum_length, 3), dtype=np.bool_)
        cb_sources = np.full((batch_size, maximum_length), -1, dtype=np.int8)
        thresholds = np.zeros(batch_size, dtype=np.float32)
        for batch_index, row in enumerate(chunk):
            length = len(row["sequence"])
            for atom_index, atom in enumerate(("N", "CA", "C")):
                coordinates[batch_index, :length, atom_index] = np.asarray(
                    row[f"{atom.lower()}_coordinates"], dtype=np.float32
                )
                masks[batch_index, :length, atom_index] = np.asarray(row[f"{atom.lower()}_mask"], dtype=np.bool_)
            cb_sources[batch_index, :length] = np.asarray(row["cb_source"], dtype=np.int8)
            thresholds[batch_index] = float(row["peptide_bond_threshold_angstrom"])
        transfer_started = time.perf_counter()
        coordinate_tensor = torch.as_tensor(coordinates, device="cuda")
        mask_tensor = torch.as_tensor(masks, device="cuda")
        threshold_tensor = torch.as_tensor(thresholds, device="cuda")
        torch.cuda.synchronize()
        profiler.add(
            StageObservation(
                "cuda_transfer",
                time.perf_counter() - transfer_started,
                sample_count=batch_size,
                bytes_read=coordinates.nbytes + masks.nbytes + thresholds.nbytes,
            )
        )
        compute_started = time.perf_counter()
        n_coord, ca_coord, c_coord = (coordinate_tensor[:, :, index] for index in range(3))
        n_mask, ca_mask, c_mask = (mask_tensor[:, :, index] for index in range(3))
        frame_cross = torch.linalg.cross(c_coord - ca_coord, n_coord - ca_coord, dim=-1)
        frame_valid = (
            n_mask
            & ca_mask
            & c_mask
            & (torch.linalg.vector_norm(c_coord - ca_coord, dim=-1) >= 1e-8)
            & (torch.linalg.vector_norm(frame_cross, dim=-1) >= 1e-8)
        )
        pseudo_cb = (
            ca_coord
            - 0.58273431 * torch.linalg.cross(ca_coord - n_coord, c_coord - ca_coord, dim=-1)
            + 0.56802827 * (ca_coord - n_coord)
            - 0.54067466 * (c_coord - ca_coord)
        )
        continuity = (
            c_mask[:, :-1]
            & n_mask[:, 1:]
            & (torch.linalg.vector_norm(c_coord[:, :-1] - n_coord[:, 1:], dim=-1) <= threshold_tensor[:, None])
        )
        neutral = torch.tensor(TORSION_NEUTRAL_SIN_COS, dtype=torch.float32, device="cuda")
        torsions = {}
        for name in TORSION_DEFINITIONS:
            values = neutral.expand(batch_size, maximum_length, 2).clone()
            valid = torch.zeros((batch_size, maximum_length), dtype=torch.bool, device="cuda")
            if maximum_length > 1:
                if name == "phi":
                    derived, nondegenerate = _cuda_dihedral_sincos(
                        torch, c_coord[:, :-1], n_coord[:, 1:], ca_coord[:, 1:], c_coord[:, 1:]
                    )
                    allowed = continuity & n_mask[:, 1:] & ca_mask[:, 1:] & c_mask[:, 1:] & nondegenerate
                    values[:, 1:] = torch.where(allowed[..., None], derived, values[:, 1:])
                    valid[:, 1:] = allowed
                elif name == "psi":
                    derived, nondegenerate = _cuda_dihedral_sincos(
                        torch, n_coord[:, :-1], ca_coord[:, :-1], c_coord[:, :-1], n_coord[:, 1:]
                    )
                    allowed = continuity & n_mask[:, :-1] & ca_mask[:, :-1] & c_mask[:, :-1] & nondegenerate
                    values[:, :-1] = torch.where(allowed[..., None], derived, values[:, :-1])
                    valid[:, :-1] = allowed
                else:
                    derived, nondegenerate = _cuda_dihedral_sincos(
                        torch, ca_coord[:, :-1], c_coord[:, :-1], n_coord[:, 1:], ca_coord[:, 1:]
                    )
                    allowed = continuity & ca_mask[:, :-1] & c_mask[:, :-1] & n_mask[:, 1:] & nondegenerate
                    values[:, 1:] = torch.where(allowed[..., None], derived, values[:, 1:])
                    valid[:, 1:] = allowed
            torsions[name] = (values, valid)
        torch.cuda.synchronize()
        compute_seconds = time.perf_counter() - compute_started
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / (1024**2))
        if peak_allocated > float(config.get("maximum_cuda_allocated_mib", 6144)):
            raise MemoryError(
                f"E006 CUDA feature allocation exceeded limit: peak={peak_allocated:.1f} MiB, "
                f"limit={float(config.get('maximum_cuda_allocated_mib', 6144)):.1f} MiB"
            )
        profiler.add(StageObservation("cuda_compute", compute_seconds, sample_count=batch_size))
        transfer_started = time.perf_counter()
        frame_cpu = frame_valid.cpu().numpy()
        pseudo_cpu = pseudo_cb.cpu().numpy()
        continuity_cpu = continuity.cpu().numpy()
        torsion_cpu = {name: (values.cpu().numpy(), valid.cpu().numpy()) for name, (values, valid) in torsions.items()}
        torch.cuda.synchronize()
        profiler.add(StageObservation("cuda_transfer", time.perf_counter() - transfer_started, sample_count=batch_size))
        for batch_index, row in enumerate(chunk):
            length = len(row["sequence"])
            if list(row["local_frame_valid"]) != frame_cpu[batch_index, :length].tolist():
                raise ValueError(f"CUDA local-frame mask contradicts CPU backend: {row['sample_id']}")
            expected_continuity = np.asarray(row["chain_continuity_mask"], dtype=np.bool_)
            if not np.array_equal(expected_continuity, continuity_cpu[batch_index, : max(length - 1, 0)]):
                raise ValueError(f"CUDA continuity mask contradicts CPU backend: {row['sample_id']}")
            for residue_index, source in enumerate(cb_sources[batch_index, :length]):
                contradiction = _cuda_pseudo_cb_contradiction(
                    row,
                    residue_index=residue_index,
                    cb_source=int(source),
                    observed=pseudo_cpu[batch_index, residue_index],
                    input_dtype=coordinates.dtype,
                )
                if contradiction is not None:
                    raise ValueError(
                        "CUDA pseudo-CB contradicts CPU backend: " + json.dumps(contradiction, sort_keys=True)
                    )
            for name, (values, valid) in torsion_cpu.items():
                expected_mask = np.asarray(row[f"{name}_mask"], dtype=np.bool_)
                observed_mask = valid[batch_index, :length]
                if not np.array_equal(expected_mask, observed_mask):
                    raise ValueError(f"CUDA {name} mask contradicts CPU backend: {row['sample_id']}")
                if expected_mask.any() and not np.allclose(
                    np.asarray(row[f"{name}_sin_cos"])[expected_mask],
                    values[batch_index, :length][expected_mask],
                    atol=CUDA_FEATURE_FLOAT32_ATOL,
                    rtol=0,
                ):
                    raise ValueError(f"CUDA {name} values contradict CPU backend: {row['sample_id']}")
    return "cuda", None, peak_allocated


def _validate_cuda_features_from_shards(
    output: Path,
    shards: list[dict[str, Any]],
    config: dict[str, Any],
    profiler: StageProfiler,
) -> tuple[str, str | None, float | None]:
    if config.get("feature_backend", "cpu") != "cuda":
        return "cpu", None, None
    budget = int(config.get("cuda_residue_token_budget", 8192))
    buffered: list[dict[str, Any]] = []
    buffered_tokens = 0
    peak = 0.0
    for shard in shards:
        if str(shard["dataset"]) not in {"train", "validation"}:
            continue
        parquet = pq.ParquetFile(output / str(shard["path"]))
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                length = len(row["sequence"])
                if buffered and buffered_tokens + length > budget:
                    try:
                        backend, reason, allocated = _apply_cuda_feature_batching(buffered, config, profiler)
                    except (MemoryError, RuntimeError) as error:
                        return "cpu", f"cuda_runtime_fallback:{type(error).__name__}:{str(error)[:160]}", peak
                    if backend != "cuda":
                        return backend, reason, allocated
                    peak = max(peak, float(allocated or 0.0))
                    buffered = []
                    buffered_tokens = 0
                buffered.append(row)
                buffered_tokens += length
    if buffered:
        try:
            backend, reason, allocated = _apply_cuda_feature_batching(buffered, config, profiler)
        except (MemoryError, RuntimeError) as error:
            return "cpu", f"cuda_runtime_fallback:{type(error).__name__}:{str(error)[:160]}", peak
        if backend != "cuda":
            return backend, reason, allocated
        peak = max(peak, float(allocated or 0.0))
    return "cuda", None, peak


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_seconds(started_utc: str, completed_utc: str) -> float:
    started = datetime.fromisoformat(started_utc)
    completed = datetime.fromisoformat(completed_utc)
    elapsed = (completed - started).total_seconds()
    if elapsed < 0:
        raise ValueError("Sidecar completion precedes its start timestamp")
    return elapsed


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _enforce_memory(limit_mib: float) -> None:
    current = _rss_mib()
    if current > limit_mib:
        raise MemoryError(
            f"E006 Phase-1 RSS limit exceeded: current={current:.1f} MiB, "
            f"peak={_peak_rss_mib():.1f} MiB, limit={limit_mib:.1f} MiB; resume with --resume"
        )


def _runtime_rss_limits(
    config: dict[str, Any],
    maximum_rss_mib: float | None,
) -> tuple[float, float, bool]:
    configured = float(config["maximum_rss_mib"])
    if maximum_rss_mib is None:
        return configured, configured, False
    effective = float(maximum_rss_mib)
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("--maximum-rss-mib must be finite and positive")
    if effective < configured:
        raise ValueError(
            "--maximum-rss-mib must be at least the configured maximum_rss_mib "
            f"({configured:g} MiB); received {effective:g} MiB"
        )
    return configured, effective, True


def _physical_split(physical_dataset_name: str) -> str:
    try:
        return INPUT_DATASET_SPLITS[physical_dataset_name]
    except KeyError as error:
        raise ValueError(f"Unknown physical pairing dataset ownership: {physical_dataset_name}") from error


def _validate_canonical_split(canonical_split: str) -> None:
    if canonical_split not in {"train", "validation"}:
        raise ValueError(f"Invalid canonical split: {canonical_split!r}")


def _normalize_split(value: Any, *, field_name: str) -> str:
    text = str(value).strip().lower()
    try:
        return SPLIT_ALIASES[text]
    except KeyError as error:
        raise ValueError(f"Unknown split value in {field_name}: {value!r}") from error


def _inject_canonical_split(row: dict[str, Any], *, canonical_split: str, physical_dataset_name: str) -> dict[str, Any]:
    _validate_canonical_split(canonical_split)
    recorded_physical_name = row.get("physical_dataset_name")
    if recorded_physical_name not in {None, "", physical_dataset_name}:
        raise ValueError(
            "Physical dataset provenance contradiction: "
            f"unit records {physical_dataset_name!r}, "
            f"row records {recorded_physical_name!r}"
        )
    for field_name in SPLIT_LIKE_FIELDS:
        value = row.get(field_name)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        observed = _normalize_split(value, field_name=field_name)
        if observed != canonical_split:
            raise ValueError(
                f"Pairing split contradiction: physical dataset {physical_dataset_name!r} owns "
                f"{canonical_split!r}, "
                f"but {field_name} records {observed!r}"
            )
    return {
        **row,
        "physical_dataset_name": physical_dataset_name,
        "split": canonical_split,
    }


@dataclass(frozen=True)
class InputUnit:
    """One bounded batch with physical ownership translated exactly once."""

    physical_dataset_name: str
    canonical_split: str
    dataset_path: Path
    fragment_identity: str
    input_unit_identity: str
    rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if self.physical_dataset_name not in INPUT_DATASET_SPLITS:
            raise ValueError(f"Unknown physical pairing dataset ownership: {self.physical_dataset_name}")
        _validate_canonical_split(self.canonical_split)
        if (self.physical_dataset_name, self.canonical_split) not in {
            ("eligible_train", "train"),
            ("eligible_validation", "validation"),
        }:
            raise ValueError(
                "Physical dataset and canonical split disagree: "
                f"{self.physical_dataset_name!r} with {self.canonical_split!r}"
            )
        if self.dataset_path.name != f"{self.physical_dataset_name}.parquet":
            raise ValueError("Input-unit dataset path contradicts physical dataset name")
        if not self.fragment_identity or not self.input_unit_identity:
            raise ValueError("Input-unit fragment and unit identities must be nonempty")
        for row in self.rows:
            if row.get("split") != self.canonical_split:
                raise ValueError("Input-unit row does not carry the canonical split")
            if row.get("physical_dataset_name") != self.physical_dataset_name:
                raise ValueError("Input-unit row does not carry physical dataset provenance")


def _assign_physical_split(row: dict[str, Any], *, input_unit: InputUnit) -> dict[str, Any]:
    """Reassert a unit's established ownership without translating it again."""
    return _inject_canonical_split(
        row,
        canonical_split=input_unit.canonical_split,
        physical_dataset_name=input_unit.physical_dataset_name,
    )


def rich_geometry_schema() -> pa.Schema:
    vector3 = pa.list_(pa.float32(), 3)
    angle2 = pa.list_(pa.float32(), 2)
    fields = [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("token_ids", pa.list_(pa.int16()), nullable=False),
        pa.field("residue_ids", pa.list_(pa.string()), nullable=False),
        pa.field("insertion_codes", pa.list_(pa.string()), nullable=False),
    ]
    fields.extend(pa.field(f"{atom.lower()}_coordinates", pa.list_(vector3), nullable=False) for atom in ATOM_NAMES)
    fields.extend(pa.field(f"{atom.lower()}_mask", pa.list_(pa.bool_()), nullable=False) for atom in ATOM_NAMES)
    fields.extend(
        [
            pa.field("cb_source", pa.list_(pa.int8()), nullable=False),
            pa.field("local_frame_valid", pa.list_(pa.bool_()), nullable=False),
            pa.field("torsion_convention_version", pa.string(), nullable=False),
            pa.field("phi_sin_cos", pa.list_(angle2), nullable=False),
            pa.field("phi_mask", pa.list_(pa.bool_()), nullable=False),
            pa.field("psi_sin_cos", pa.list_(angle2), nullable=False),
            pa.field("psi_mask", pa.list_(pa.bool_()), nullable=False),
            pa.field("omega_sin_cos", pa.list_(angle2), nullable=False),
            pa.field("omega_mask", pa.list_(pa.bool_()), nullable=False),
            pa.field("chain_continuity_mask", pa.list_(pa.bool_()), nullable=False),
            pa.field("chain_break_mask", pa.list_(pa.bool_()), nullable=False),
            pa.field("residue_classification", pa.list_(pa.string()), nullable=False),
            pa.field("authoritative_parent_comp_id", pa.list_(pa.string())),
            pa.field("authoritative_parent_source", pa.list_(pa.string())),
            pa.field("selected_calpha_conformer", pa.list_(pa.string())),
            pa.field(
                "selected_atom_conformers",
                pa.list_(pa.struct([pa.field(atom.lower(), pa.string()) for atom in ATOM_NAMES])),
            ),
            pa.field("model_number", pa.int32(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
            pa.field("source_sha256", pa.string(), nullable=False),
            pa.field("npz_path", pa.string(), nullable=False),
            pa.field("npz_sha256", pa.string(), nullable=False),
            pa.field("npz_metadata_sha256", pa.string(), nullable=False),
            pa.field("mapping_version", pa.string(), nullable=False),
            pa.field("canonicalization_version", pa.string(), nullable=False),
            pa.field("coordinate_anchor_policy_version", pa.string(), nullable=False),
            pa.field("mapping_evidence_sha256", pa.string(), nullable=False),
            pa.field("npz_internal_matrix_rmse_angstrom", pa.float64(), nullable=False),
            pa.field("npz_internal_matrix_maximum_error_angstrom", pa.float64(), nullable=False),
            pa.field("source_to_npz_calpha_coordinate_rmse_angstrom", pa.float64(), nullable=False),
            pa.field("source_to_npz_calpha_coordinate_maximum_error_angstrom", pa.float64(), nullable=False),
            pa.field("ca_candidate_counts", pa.list_(pa.int16()), nullable=False),
            pa.field("blank_altloc_fallback_count", pa.int32(), nullable=False),
            pa.field("conflicting_nonblank_conformer_count", pa.int32(), nullable=False),
            pa.field("peptide_bond_threshold_angstrom", pa.float32(), nullable=False),
            pa.field(
                "anchor_failure_examples",
                pa.list_(pa.struct([pa.field("category", pa.string()), pa.field("evidence", pa.string())])),
            ),
        ]
    )
    return pa.schema(fields, metadata={b"schema_version": SIDECAR_SCHEMA_VERSION.encode()})


def eligibility_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("eligible", pa.bool_(), nullable=False),
            pa.field("exclusion_reason", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("source_sha256", pa.string()),
            pa.field("npz_path", pa.string(), nullable=False),
            pa.field("npz_sha256", pa.string(), nullable=False),
            pa.field("sequence_length", pa.int32(), nullable=False),
            pa.field("coordinate_anchor_policy_version", pa.string(), nullable=False),
            pa.field("diagnostics", pa.string(), nullable=False),
        ],
        metadata={b"schema_version": SIDECAR_SCHEMA_VERSION.encode()},
    )


def exclusion_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("exclusion_reason", pa.string(), nullable=False),
            pa.field("source_path", pa.string()),
            pa.field("npz_path", pa.string(), nullable=False),
            pa.field("bounded_evidence", pa.string(), nullable=False),
        ],
        metadata={b"schema_version": SIDECAR_SCHEMA_VERSION.encode()},
    )


def _referenced_artifacts(protocol: dict[str, Any]) -> list[tuple[Path, str, int | None]]:
    artifacts: list[tuple[Path, str, int | None]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("path") and value.get("sha256"):
                artifacts.append(
                    (
                        Path(str(value["path"])),
                        str(value["sha256"]),
                        value.get("row_count"),
                    )
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(protocol)
    return artifacts


def attest_phase0(protocol_path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    path = Path(protocol_path)
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError("Phase-0 protocol SHA-256 contradiction")
    protocol = json.loads(path.read_text())
    failures = []
    if protocol.get("status") != "completed":
        failures.append("protocol_not_completed")
    if protocol.get("schema_version") != EXPECTED_AUDIT_SCHEMA or AUDIT_SCHEMA_VERSION != EXPECTED_AUDIT_SCHEMA:
        failures.append("incorrect_phase0_schema")
    if protocol.get("phase1_authorization", {}).get("authorized") is not True:
        failures.append("phase1_not_authorized")
    before = protocol.get("input_hashes_before")
    after = protocol.get("input_hashes_after")
    if not isinstance(before, dict) or before != after or protocol.get("dataset_inputs_unchanged") is not True:
        failures.append("phase0_input_preservation_not_attested")
    for artifact, digest, row_count in _referenced_artifacts(protocol):
        if not artifact.is_file() or sha256_file(artifact) != digest:
            failures.append(f"referenced_artifact_hash_mismatch:{artifact}")
        elif row_count is not None and artifact.suffix == ".parquet":
            if pq.ParquetFile(artifact).metadata.num_rows != int(row_count):
                failures.append(f"referenced_artifact_row_count_mismatch:{artifact}")
    if failures:
        raise ValueError("Phase-0 authorization verification failed: " + ", ".join(failures))
    return protocol


def validate_sidecar_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SIDECAR_SCHEMA_VERSION}")
    expected_protocol_hash = str(config.get("phase0_protocol_sha256") or "")
    if len(expected_protocol_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_protocol_hash
    ):
        raise ValueError("phase0_protocol_sha256 must be a lowercase SHA-256 digest")
    shard_size = int(config.get("shard_size", 0))
    row_group_size = int(config.get("row_group_size", 0))
    if not 1 <= shard_size <= 4096:
        raise ValueError("shard_size must be in [1, 4096]")
    if not 1 <= row_group_size <= shard_size:
        raise ValueError("row_group_size must be in [1, shard_size]")
    if not 1 <= int(config.get("maximum_failure_examples", 0)) <= 100:
        raise ValueError("maximum_failure_examples must be in [1, 100]")
    if not 0 < float(config.get("maximum_rss_mib", 0)) <= 4096:
        raise ValueError("maximum_rss_mib must be in (0, 4096]")
    if int(config.get("pilot_sample_count", 0)) < 1:
        raise ValueError("pilot_sample_count must be positive")
    if int(config.get("heartbeat_frequency_samples", 25)) < 1:
        raise ValueError("heartbeat_frequency_samples must be positive")
    worker_count = int(config.get("worker_count", 1))
    queue_size = int(config.get("worker_queue_size", max(2, worker_count * 2)))
    if not 1 <= worker_count <= 4:
        raise ValueError("worker_count must be in [1, 4]")
    if not worker_count <= queue_size <= 32:
        raise ValueError("worker_queue_size must be in [worker_count, 32]")
    if int(config.get("maximum_stage_latency_samples", 8192)) < 512:
        raise ValueError("maximum_stage_latency_samples must be at least 512")
    if config.get("feature_backend", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("feature_backend must be cpu or cuda")
    if config.get("mmcif_parse_mode", "single_parse") not in {"single_parse", "legacy_double_parse"}:
        raise ValueError("mmcif_parse_mode must be single_parse or legacy_double_parse")
    if int(config.get("cuda_residue_token_budget", 8192)) < 1:
        raise ValueError("cuda_residue_token_budget must be positive")
    if not 0 < float(config.get("maximum_cuda_allocated_mib", 6144)) <= 6144:
        raise ValueError("maximum_cuda_allocated_mib must be in (0, 6144]")
    if Path(config["pilot_output_dir"]).resolve() == Path(config["output_dir"]).resolve():
        raise ValueError("Pilot and definitive outputs must differ")


def verify_protected_inputs(protocol: dict[str, Any]) -> dict[str, str]:
    expected = protocol["input_hashes_before"]
    current = {}
    for path_text, digest in expected.items():
        path = Path(path_text)
        if not path.is_file():
            raise ValueError(f"Protected Phase-0 input is missing: {path}")
        current[path_text] = sha256_file(path)
        if current[path_text] != digest:
            raise ValueError(f"Protected Phase-0 input changed: {path}")
    return current


def derive_backbone_torsions(
    coordinates: dict[str, Any],
    atom_masks: dict[str, Any],
    *,
    peptide_bond_threshold_angstrom: float,
) -> dict[str, Any]:
    """Derive the canonical masked backbone torsion representation.

    Coordinates are quantized to their persisted float32 precision first. Atom order is
    phi_i=(C_(i-1), N_i, CA_i, C_i), psi_i=(N_i, CA_i, C_i, N_(i+1)), and
    omega_i=(CA_(i-1), C_(i-1), N_i, CA_i). Masked values use the neutral
    [sin(0), cos(0)] = [0, 1] representation.
    """
    required_atoms = {atom for definition in TORSION_DEFINITIONS.values() for atom, _ in definition}
    coordinate_arrays = {atom: np.asarray(coordinates[atom], dtype=np.float32) for atom in required_atoms}
    mask_arrays = {atom: np.asarray(atom_masks[atom], dtype=np.bool_) for atom in required_atoms}
    lengths = {values.shape[0] for values in coordinate_arrays.values()} | {
        values.shape[0] for values in mask_arrays.values()
    }
    if len(lengths) != 1:
        raise ValueError("Backbone coordinate and atom-mask lengths disagree")
    count = lengths.pop()
    for atom in required_atoms:
        if coordinate_arrays[atom].shape != (count, 3) or mask_arrays[atom].shape != (count,):
            raise ValueError(f"Invalid canonical torsion input shape for {atom}")

    connected = []
    for index in range(max(count - 1, 0)):
        available = bool(mask_arrays["C"][index] and mask_arrays["N"][index + 1])
        finite = bool(
            np.isfinite(coordinate_arrays["C"][index]).all() and np.isfinite(coordinate_arrays["N"][index + 1]).all()
        )
        connected.append(
            available
            and finite
            and bool(
                np.linalg.norm(coordinate_arrays["C"][index] - coordinate_arrays["N"][index + 1])
                <= peptide_bond_threshold_angstrom
            )
        )
    output: dict[str, Any] = {
        "chain_continuity_mask": connected,
        "chain_break_mask": [not value for value in connected],
    }
    for name, definition in TORSION_DEFINITIONS.items():
        values = [list(TORSION_NEUTRAL_SIN_COS) for _ in range(count)]
        mask = [False] * count
        for index in range(count):
            permitted = (
                (index > 0 and connected[index - 1])
                if name in {"phi", "omega"}
                else (index < count - 1 and connected[index])
            )
            if not permitted:
                continue
            atom_indices = [(atom, index + offset) for atom, offset in definition]
            if not all(
                mask_arrays[atom][atom_index] and np.isfinite(coordinate_arrays[atom][atom_index]).all()
                for atom, atom_index in atom_indices
            ):
                continue
            try:
                angle = dihedral_angle(*(coordinate_arrays[atom][atom_index] for atom, atom_index in atom_indices))
                vector = np.asarray([np.sin(angle), np.cos(angle)], dtype=np.float64)
                vector /= np.linalg.norm(vector)
                values[index] = vector.astype(np.float32).tolist()
                mask[index] = True
            except ValueError:
                pass
        output[f"{name}_sin_cos"] = values
        output[f"{name}_mask"] = mask
    return output


def _sidecar_row(
    row: dict[str, Any],
    residues: list[Any],
    mapping: dict[str, Any],
    anchor: dict[str, Any],
    npz: Any,
    *,
    canonical_split: str,
) -> dict[str, Any]:
    sequence = str(row["sequence"])
    token_ids = SequenceGeometryVocabulary().encode(sequence)
    coordinates: dict[str, list[list[float]]] = {atom: [] for atom in ATOM_NAMES}
    masks: dict[str, list[bool]] = {atom: [] for atom in ATOM_NAMES}
    cb_source = []
    frames = []
    for residue in residues:
        for atom in ATOM_NAMES:
            available = atom in residue.atoms and np.isfinite(residue.atoms[atom]).all()
            masks[atom].append(bool(available))
            coordinates[atom].append(residue.atoms[atom].astype(np.float32).tolist() if available else [0.0, 0.0, 0.0])
        geometry = residue_geometry_diagnostics(residue)
        frames.append(bool(geometry["frame_valid"]))
        if masks["CB"][-1]:
            cb_source.append(1)
        else:
            try:
                pseudo_cb = pseudo_cb_coordinate(residue.atoms["N"], residue.atoms["CA"], residue.atoms["C"])
                coordinates["CB"][-1] = pseudo_cb.astype(np.float32).tolist()
                masks["CB"][-1] = True
                cb_source.append(0)
            except (KeyError, ValueError):
                cb_source.append(-1)
    angle = derive_backbone_torsions(
        coordinates,
        masks,
        peptide_bond_threshold_angstrom=float(row.get("peptide_bond_threshold_angstrom", 2.0)),
    )
    failures = json.loads(anchor["coordinate_anchor_failure_examples_json"])
    failure_rows = [
        {"category": category, "evidence": json.dumps(item, sort_keys=True)}
        for category, examples in sorted(failures.items())
        for item in examples
    ]
    source_path = str(row["resolved_source_path"])
    npz_path = str(row["resolved_npz_path"])
    mapping_hash = hashlib.sha256(
        json.dumps(
            {
                "residue_ids": mapping["resolved_target_residue_ids"],
                "classifications": mapping["target_residue_classifications"],
                "mapping_version": mapping["mapping_version"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    result = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "sample_id": str(row["sample_id"]),
        "split": canonical_split,
        "sequence": sequence,
        "token_ids": token_ids,
        "residue_ids": list(mapping["resolved_target_residue_ids"]),
        "insertion_codes": [residue.insertion_code for residue in residues],
        "cb_source": cb_source,
        "local_frame_valid": frames,
        "torsion_convention_version": TORSION_CONVENTION_VERSION,
        **angle,
        "residue_classification": list(mapping["target_residue_classifications"]),
        "authoritative_parent_comp_id": [residue.authoritative_parent_comp_id for residue in residues],
        "authoritative_parent_source": [residue.authoritative_parent_source for residue in residues],
        "selected_calpha_conformer": json.loads(anchor["selected_calpha_conformers_json"]),
        "selected_atom_conformers": [
            {atom.lower(): (residue.selected_altlocs or {}).get(atom) for atom in ATOM_NAMES} for residue in residues
        ],
        "model_number": int(row.get("model_number") or 1),
        "source_path": source_path,
        "source_sha256": str(row["actual_source_sha256"]),
        "npz_path": npz_path,
        "npz_sha256": str(row["actual_npz_sha256"]),
        "npz_metadata_sha256": npz.metadata_sha256,
        "mapping_version": str(mapping["mapping_version"]),
        "canonicalization_version": str(mapping["canonicalization_version"]),
        "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
        "mapping_evidence_sha256": mapping_hash,
        "npz_internal_matrix_rmse_angstrom": float(anchor["npz_internal_matrix_rmse_angstrom"]),
        "npz_internal_matrix_maximum_error_angstrom": float(anchor["npz_internal_matrix_maximum_error_angstrom"]),
        "source_to_npz_calpha_coordinate_rmse_angstrom": float(anchor["source_to_npz_calpha_coordinate_rmse_angstrom"]),
        "source_to_npz_calpha_coordinate_maximum_error_angstrom": float(
            anchor["source_to_npz_calpha_coordinate_maximum_error_angstrom"]
        ),
        "ca_candidate_counts": json.loads(anchor["ca_candidate_counts_json"]),
        "blank_altloc_fallback_count": int(anchor["blank_altloc_fallback_count"]),
        "conflicting_nonblank_conformer_count": int(anchor["conflicting_nonblank_conformer_count"]),
        "peptide_bond_threshold_angstrom": float(row.get("peptide_bond_threshold_angstrom", 2.0)),
        "anchor_failure_examples": failure_rows,
    }
    for atom in ATOM_NAMES:
        result[f"{atom.lower()}_coordinates"] = coordinates[atom]
        result[f"{atom.lower()}_mask"] = masks[atom]
    return result


def evaluate_sample(
    row: dict[str, Any],
    config: dict[str, Any],
    *,
    canonical_split: str,
    stage_observer: Callable[[str, float, int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply the complete v6 policy to one pairing row."""
    if canonical_split not in {"train", "validation"}:
        raise ValueError(f"Unknown canonical split: {canonical_split!r}")
    if str(row.get("split") or "") != canonical_split:
        raise ValueError("Candidate row does not carry its canonical physical split")
    if (row.get("physical_dataset_name"), canonical_split) not in {
        ("eligible_train", "train"),
        ("eligible_validation", "validation"),
    }:
        raise ValueError("Candidate row does not carry valid physical dataset provenance")
    npz_path = resolve_within_roots(row["matrix_path"], config["allowed_matrix_roots"])
    source_path = resolve_within_roots(row["source_file"], config["allowed_structure_roots"])
    started = time.perf_counter()
    npz_hash = sha256_file(npz_path)
    source_hash = sha256_file(source_path)
    if stage_observer is not None:
        stage_observer(
            "hashing",
            time.perf_counter() - started,
            npz_path.stat().st_size + source_path.stat().st_size,
            0,
        )
    started = time.perf_counter()
    npz = load_npz_geometry_evidence(
        npz_path, internal_tolerance_angstrom=float(config["npz_internal_matrix_tolerance_angstrom"])
    )
    if stage_observer is not None:
        stage_observer("npz_loading", time.perf_counter() - started, npz_path.stat().st_size, 0)
    residues, model_metadata = parse_backbone_mmcif(
        source_path,
        chain_id=str(row.get("chain_id") or row.get("auth_asym_id") or row.get("label_asym_id")),
        model_number=int(row.get("model_number") or 1),
        residue_mappings=config.get("residue_mappings"),
        stage_observer=stage_observer,
        parse_mode=str(config.get("mmcif_parse_mode", "single_parse")),
    )
    started = time.perf_counter()
    convention = str(row.get("selected_residue_id_convention") or "")
    if not convention:
        convention = next(
            (
                name
                for field, name in (
                    ("auth_residue_ids_match", "auth_seq_id_insertion"),
                    ("label_residue_ids_match", "label_seq_id"),
                    ("zero_based_positions_match", "position_zero_based"),
                    ("one_based_positions_match", "position_one_based"),
                )
                if row.get(field)
            ),
            "",
        )
    alternatives, authorized = _alternative_identifier_inputs(row)
    mapping = resolve_verified_target_mapping(
        residues,
        sample_id=str(row["sample_id"]),
        expected_sequence=str(row["sequence"]),
        residue_ids=row.get("residue_ids"),
        insertion_codes=row.get("insertion_codes"),
        residue_id_convention=convention,
        pairing_canonicalization_version=(
            str(row["modified_residue_mapping_version"]) if row.get("modified_residue_mapping_version") else None
        ),
        alternative_residue_ids=alternatives,
        authorized_residue_id_conventions=authorized,
        audit_source_record=row.get("_identifier_audit_source_record"),
        audit_source_artifact=str(row.get("_identifier_audit_source_artifact") or "compact_audit"),
        direct_source_artifact=str(row.get("pairing_source_artifact") or "pairing_dataset"),
    )
    target_residues = mapping.pop("target_residues")
    model_residues = model_metadata.pop("_model_residues")
    model_mappings = [
        resolve_verified_target_mapping(
            values,
            sample_id=str(row["sample_id"]),
            expected_sequence=str(row["sequence"]),
            residue_ids=row.get("residue_ids"),
            insertion_codes=row.get("insertion_codes"),
            residue_id_convention=convention,
            pairing_canonicalization_version=(
                str(row["modified_residue_mapping_version"]) if row.get("modified_residue_mapping_version") else None
            ),
            alternative_residue_ids=alternatives,
            authorized_residue_id_conventions=authorized,
            audit_source_record=row.get("_identifier_audit_source_record"),
            audit_source_artifact=str(row.get("_identifier_audit_source_artifact") or "compact_audit"),
            direct_source_artifact=str(row.get("pairing_source_artifact") or "pairing_dataset"),
        )
        for values in model_residues.values()
    ]
    cross_model_consistent = bool(
        model_mappings
        and all(
            item["mapping_verified"] and item["observed_target_sequence"] == str(row["sequence"])
            for item in model_mappings
        )
    )
    if stage_observer is not None:
        stage_observer("residue_mapping", time.perf_counter() - started, 0, 0)
    anchor: dict[str, Any] = {"coordinate_anchor_exclusion_reason": None}
    anchored = []
    started = time.perf_counter()
    if mapping["mapping_verified"]:
        anchor = reconcile_npz_calpha_anchors(
            target_residues,
            resolved_target_residue_ids=list(mapping["resolved_target_residue_ids"]),
            npz=npz,
            anchor_tolerance_angstrom=float(config["npz_calpha_anchor_tolerance_angstrom"]),
            scientific_tolerance_angstrom=float(config["phase1_authorization"]["maximum_matrix_rmse_angstrom"]),
            maximum_examples=int(config["maximum_failure_examples"]),
            expected_sample_id=str(row["sample_id"]),
            expected_pdb_id=str(row.get("pdb_id") or ""),
            expected_chain_id=str(row.get("chain_id") or ""),
            expected_sequence=str(row["sequence"]),
            expected_model_number=int(row.get("model_number") or 1),
            expected_source_path=source_path,
            recorded_source_sha256=str(row["source_sha256"]) if row.get("source_sha256") else None,
            actual_source_sha256=source_hash,
        )
        anchored = anchor.pop("_anchored_residues")
    if stage_observer is not None:
        stage_observer("ca_anchor_resolution", time.perf_counter() - started, 0, 0)
    mapping.update(anchor)
    feature_started = time.perf_counter()
    analysis = analyze_backbone_residues(
        anchored,
        expected_sequence=str(row["sequence"]),
        stored_matrix=npz.distance_matrix,
        peptide_bond_threshold_angstrom=float(config["peptide_bond_threshold_angstrom"]),
        pseudo_cb_maximum_distance_angstrom=float(config["pseudo_cb_maximum_distance_angstrom"]),
        pseudo_cb_maximum_angle_degrees=float(config["pseudo_cb_maximum_angle_degrees"]),
    )
    eligibility = resolve_rich_geometry_eligibility(
        mapping,
        analysis,
        cross_model_sequence_consistent=cross_model_consistent,
    )
    row = {
        **row,
        "resolved_source_path": str(source_path),
        "resolved_npz_path": str(npz_path),
        "actual_source_sha256": source_hash,
        "actual_npz_sha256": npz_hash,
    }
    diagnostics = {
        "mapping_errors": mapping["mapping_errors"],
        "anchor_exclusion": anchor.get("coordinate_anchor_exclusion_reason"),
        "anchor_failures": json.loads(anchor.get("coordinate_anchor_failure_examples_json", "{}")),
    }
    decision = {
        "sample_id": str(row["sample_id"]),
        "split": canonical_split,
        "eligible": bool(eligibility["rich_geometry_eligible"]),
        "exclusion_reason": eligibility["exclusion_reason"],
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "npz_path": str(npz_path),
        "npz_sha256": npz_hash,
        "sequence_length": len(str(row["sequence"])),
        "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
        "diagnostics": json.dumps(diagnostics, sort_keys=True),
    }
    sidecar = (
        _sidecar_row(
            row,
            anchored,
            mapping,
            anchor,
            npz,
            canonical_split=canonical_split,
        )
        if decision["eligible"]
        else None
    )
    if stage_observer is not None:
        stage_observer(
            "frame_torsion_cb_feature_derivation",
            time.perf_counter() - feature_started,
            0,
            0,
        )
    return decision, sidecar


class ConstructionJournal:
    def __init__(self, path: Path, *, config_hash: str, resume: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        self.connection = sqlite3.connect(path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS completed_units(
                unit_id TEXT PRIMARY KEY,
                physical_dataset_name TEXT NOT NULL,
                canonical_split TEXT NOT NULL,
                unit_index INTEGER NOT NULL,
                input_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shards(
                path TEXT PRIMARY KEY, dataset TEXT NOT NULL, row_count INTEGER NOT NULL, sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions(
                sample_id TEXT NOT NULL, split TEXT NOT NULL, eligible INTEGER NOT NULL, exclusion_reason TEXT,
                PRIMARY KEY(split,sample_id)
            );
            CREATE TABLE IF NOT EXISTS observed_inputs(
                path TEXT PRIMARY KEY, sha256 TEXT NOT NULL
            );
            """
        )
        previous = self.get("config_hash")
        if existed and not resume:
            raise FileExistsError("Construction journal exists; use --resume")
        if resume and previous != config_hash:
            raise ValueError("Construction resume configuration hash mismatch")
        if not existed:
            self.set("schema_version", JOURNAL_SCHEMA_VERSION)
            self.set("config_hash", config_hash)
            self.set("status", "running")
            self.connection.commit()

    def get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, str(value)))

    def completed(self, unit_id: str) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM completed_units WHERE unit_id=?", (unit_id,)).fetchone() is not None
        )

    def record_observed_input(self, path: str, digest: str) -> None:
        previous = self.connection.execute("SELECT sha256 FROM observed_inputs WHERE path=?", (path,)).fetchone()
        if previous is not None and str(previous[0]) != digest:
            raise ValueError(f"Observed input hash contradiction during construction: {path}")
        self.connection.execute("INSERT OR IGNORE INTO observed_inputs VALUES (?,?)", (path, digest))


def _write_part(
    path: Path,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    row_group_size: int,
    *,
    profiler: StageProfiler | None = None,
) -> tuple[int, str]:
    started = time.perf_counter()
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, temporary, compression="zstd", row_group_size=row_group_size)
    if profiler is not None:
        profiler.add(
            StageObservation(
                "parquet_encoding_writing",
                time.perf_counter() - started,
                sample_count=len(rows),
                bytes_written=temporary.stat().st_size,
            )
        )
    started = time.perf_counter()
    digest = sha256_file(temporary)
    hashed_bytes = temporary.stat().st_size
    if path.exists():
        existing_digest = sha256_file(path)
        hashed_bytes += path.stat().st_size
        if existing_digest != digest or pq.ParquetFile(path).metadata.num_rows != len(rows):
            temporary.unlink()
            raise ValueError(f"Existing resumable shard contradicts deterministic output: {path}")
        temporary.unlink()
    else:
        temporary.replace(path)
    if profiler is not None:
        profiler.add(
            StageObservation(
                "hashing",
                time.perf_counter() - started,
                sample_count=len(rows),
                bytes_read=hashed_bytes,
            )
        )
    return len(rows), digest


def _shard_summaries(output: Path, shards: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summaries = {
        dataset: {"file_count": 0, "row_count": 0, "compressed_bytes": 0}
        for dataset in ("train", "validation", "eligibility_manifest.parquet", "exclusions.parquet")
    }
    for shard in shards:
        dataset = str(shard["dataset"])
        if dataset not in summaries:
            raise ValueError(f"Unknown sidecar dataset in shard summary: {dataset}")
        path = output / str(shard["path"])
        if not path.is_file():
            raise ValueError(f"Sidecar shard is missing while summarizing: {path}")
        summaries[dataset]["file_count"] += 1
        summaries[dataset]["row_count"] += int(shard["row_count"])
        summaries[dataset]["compressed_bytes"] += path.stat().st_size
    return summaries


def _sample_order_sha256(output: Path, shards: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    eligibility_shards = [shard for shard in shards if str(shard["dataset"]) == "eligibility_manifest.parquet"]
    for shard in sorted(eligibility_shards, key=lambda item: str(item["path"])):
        parquet = pq.ParquetFile(output / str(shard["path"]))
        for batch in parquet.iter_batches(
            columns=["sample_id", "split", "eligible", "exclusion_reason"],
            batch_size=4096,
        ):
            for row in batch.to_pylist():
                digest.update(
                    (
                        f"{row['split']}\0{row['sample_id']}\0{int(row['eligible'])}\0{row['exclusion_reason'] or ''}\n"
                    ).encode()
                )
    return digest.hexdigest()


def _input_units(
    root: Path,
    physical_dataset_name: str,
    batch_size: int,
    *,
    columns: tuple[str, ...] | None = None,
    profiler: StageProfiler | None = None,
) -> Iterator[InputUnit]:
    canonical_split = _physical_split(physical_dataset_name)
    directory = root / f"{physical_dataset_name}.parquet"
    dataset = ds.dataset(str(directory), format="parquet")
    projected_columns = None
    if columns is not None:
        projected_columns = list(
            dict.fromkeys((*columns, *(field for field in SPLIT_LIKE_FIELDS if field in dataset.schema.names)))
        )
    for fragment in sorted(dataset.get_fragments(), key=lambda item: str(item.path)):
        fragment_identity = str(fragment.path)
        scanner = fragment.scanner(
            columns=projected_columns,
            batch_size=batch_size,
            use_threads=False,
        )
        batches = iter(scanner.to_batches())
        index = 0
        while True:
            started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            if profiler is not None:
                profiler.add(
                    StageObservation(
                        "arrow_row_retrieval",
                        time.perf_counter() - started,
                        sample_count=batch.num_rows,
                        bytes_read=batch.nbytes,
                    )
                )
            unit_identity = f"{physical_dataset_name}:{fragment_identity}:{index}"
            rows = tuple(
                _inject_canonical_split(
                    row,
                    canonical_split=canonical_split,
                    physical_dataset_name=physical_dataset_name,
                )
                for row in batch.to_pylist()
            )
            yield InputUnit(
                physical_dataset_name=physical_dataset_name,
                canonical_split=canonical_split,
                dataset_path=directory,
                fragment_identity=fragment_identity,
                input_unit_identity=unit_identity,
                rows=rows,
            )
            index += 1


def _pilot_ids(
    root: Path,
    limit: int,
    seed: int,
    *,
    profiler: StageProfiler | None = None,
) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    split_targets = (
        ("eligible_train", "train", (limit + 1) // 2),
        ("eligible_validation", "validation", limit // 2),
    )
    for physical_dataset_name, canonical_split, split_limit in split_targets:
        if split_limit == 0:
            continue
        retained: list[tuple[str, str]] = []
        for input_unit in _input_units(
            root,
            physical_dataset_name,
            4096,
            columns=("sample_id",),
            profiler=profiler,
        ):
            if input_unit.canonical_split != canonical_split:
                raise AssertionError("Pilot input unit changed canonical split")
            for row in input_unit.rows:
                sample_text = str(row["sample_id"])
                rank = hashlib.sha256(f"{seed}:{canonical_split}:{sample_text}".encode()).hexdigest()
                bisect.insort(retained, (rank, sample_text))
                if len(retained) > split_limit:
                    retained.pop()
        selected.update((canonical_split, sample_id) for _, sample_id in retained)
    if len(selected) != limit:
        raise ValueError(
            f"Pilot selection could not produce {limit} unique train/validation samples; selected={len(selected)}"
        )
    return selected


def _heartbeat(path: Path, *, status: str, stage: str, processed: int, **extra: Any) -> None:
    _atomic_json(
        path,
        {
            "status": status,
            "stage": stage,
            "processed": processed,
            "current_rss_mib": _rss_mib(),
            "peak_rss_mib": _peak_rss_mib(),
            "heartbeat_utc": _utc_now(),
            **extra,
        },
    )


def _journal_position(journal: ConstructionJournal, processed: int) -> dict[str, Any]:
    index = journal.get("last_committed_unit_index")
    return {
        "processed_samples": processed,
        "unit_id": journal.get("last_committed_unit_id"),
        "unit_index": int(index) if index is not None else None,
        "physical_dataset_name": journal.get("last_committed_physical_dataset_name"),
        "split": journal.get("last_committed_split"),
    }


def _publish_failure_heartbeat(
    path: Path,
    *,
    status: str,
    stage: str,
    processed: int,
    error: BaseException,
    journal: ConstructionJournal,
    resumable: bool,
    runtime_rss_metadata: dict[str, Any] | None = None,
) -> None:
    _heartbeat(
        path,
        status=status,
        stage=stage,
        processed=processed,
        error_type=type(error).__name__,
        error_message=str(error)[:500],
        latest_committed_journal_position=_journal_position(journal, processed),
        resumable=resumable,
        resumability_status="resumable" if resumable else "not_resumable",
        resume_required=resumable,
        **(runtime_rss_metadata or {}),
    )


def _config_hash(config: dict[str, Any], *, mode: str, output: Path) -> str:
    payload = {**config, "mode": mode, "effective_output": str(output)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _prepare_metadata(output: Path, config: dict[str, Any], phase0: dict[str, Any]) -> None:
    schema = rich_geometry_schema()
    _atomic_json(
        output / "schema.json",
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "columns": {field.name: str(field.type) for field in schema},
            "dense_pair_features_stored": False,
            "lazy_pair_features": ["distances", "orientations", "relative_rotations", "local_frame_vectors"],
            "cb_source_codes": {"-1": "unavailable", "0": "pseudo_cb", "1": "native_cb"},
            "torsion_convention": {
                "version": TORSION_CONVENTION_VERSION,
                "phi_i_atom_order": ["C_(i-1)", "N_i", "CA_i", "C_i"],
                "psi_i_atom_order": ["N_i", "CA_i", "C_i", "N_(i+1)"],
                "omega_i_atom_order": ["CA_(i-1)", "C_(i-1)", "N_i", "CA_i"],
                "representation": "normalized_[sin(theta),cos(theta)]_float32",
                "masked_neutral_representation": list(TORSION_NEUTRAL_SIN_COS),
                "comparison_absolute_tolerance": TORSION_FLOAT32_ATOL,
            },
        },
    )
    _atomic_json(output / "vocabulary.json", SequenceGeometryVocabulary().as_dict())
    normalization_path = Path(config["normalization_source"])
    _atomic_json(
        output / "normalization.json",
        {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "coordinate_units": "angstrom",
            "coordinates_stored_unscaled": True,
            "torsions": "sin_cos_unit_circle",
            "torsion_convention_version": TORSION_CONVENTION_VERSION,
            "lazy_distance_normalization_source": str(normalization_path),
            "lazy_distance_normalization_sha256": sha256_file(normalization_path),
            "phase0_protocol_sha256": sha256_file(Path(config["phase0_protocol"])),
            "phase0_authorization": phase0["phase1_authorization"],
        },
    )


def _verify_publication_contract(
    protocol: dict[str, Any],
    *,
    require_training_authorization: bool,
) -> None:
    mode = protocol.get("mode")
    if mode not in {"pilot", "full"}:
        raise ValueError(f"Unknown rich-geometry publication mode: {mode!r}")
    definitive_authorized = protocol.get("authorizes_definitive_dataset")
    training_authorized = protocol.get("authorizes_training")
    if not isinstance(definitive_authorized, bool) or not isinstance(training_authorized, bool):
        raise ValueError("Sidecar protocol authorization fields are missing or malformed")
    if mode == "pilot" and (definitive_authorized or training_authorized):
        raise ValueError("Pilot sidecars cannot authorize a definitive dataset or training")
    if protocol.get("status") == "finalizing" and (definitive_authorized or training_authorized):
        raise ValueError("Finalizing sidecars cannot publish authorization before verification")
    full_safety_gate = (
        mode == "full"
        and protocol.get("status") == "completed"
        and protocol.get("protected_inputs_unchanged") is True
        and protocol.get("observed_phase1_inputs_unchanged") is True
        and int(protocol.get("unexplained_failure_count", -1)) == 0
    )
    if protocol.get("status") == "completed" and mode == "full":
        if definitive_authorized != full_safety_gate or training_authorized != full_safety_gate:
            raise ValueError("Definitive sidecar authorization contradicts the completed safety gate")
    legacy_authorization = protocol.get("authorizes_full_dataset")
    if legacy_authorization is not None and legacy_authorization != definitive_authorized:
        raise ValueError("Legacy and canonical definitive-dataset authorization fields disagree")
    if require_training_authorization and not (
        mode == "full" and definitive_authorized and training_authorized and full_safety_gate
    ):
        raise ValueError("Sidecar dataset is not an authorized definitive training dataset")


def _verify_completed_timing(protocol: dict[str, Any]) -> None:
    required = (
        "started_utc",
        "completed_utc",
        "elapsed_seconds",
        "processed_samples",
        "samples_per_second",
        "eligible_samples_per_second",
    )
    if any(protocol.get(field) is None for field in required):
        raise ValueError("Completed sidecar protocol has incomplete timing or throughput fields")
    elapsed = _elapsed_seconds(str(protocol["started_utc"]), str(protocol["completed_utc"]))
    recorded_elapsed = float(protocol["elapsed_seconds"])
    if not math.isfinite(recorded_elapsed) or abs(recorded_elapsed - elapsed) > 1e-6:
        raise ValueError("Sidecar elapsed time contradicts publication timestamps")
    processed = int(protocol["processed_samples"])
    counts = protocol["definitive_observed_counts"]
    if processed != int(counts["total"]):
        raise ValueError("Sidecar processed-sample count contradicts observed counts")
    denominator = max(recorded_elapsed, 1e-9)
    expected_rates = {
        "samples_per_second": processed / denominator,
        "eligible_samples_per_second": int(counts["eligible"]) / denominator,
    }
    for field, expected in expected_rates.items():
        observed = float(protocol[field])
        if not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Sidecar throughput field contradicts counts and elapsed time: {field}")
    stage_timing = protocol.get("stage_timing")
    if not isinstance(stage_timing, dict) or set(stage_timing) != set(PERFORMANCE_STAGE_NAMES):
        raise ValueError("Completed sidecar protocol has incomplete stage timing")
    required_stage_fields = {
        "total_seconds",
        "percentage_of_runtime",
        "sample_count",
        "mean_latency_seconds",
        "median_latency_seconds",
        "p95_latency_seconds",
        "bytes_read",
        "bytes_written",
    }
    for stage, values in stage_timing.items():
        if not isinstance(values, dict) or not required_stage_fields <= set(values):
            raise ValueError(f"Completed sidecar protocol has malformed stage timing: {stage}")
        numeric_values = [float(values[field]) for field in required_stage_fields]
        if not all(math.isfinite(value) and value >= 0 for value in numeric_values):
            raise ValueError(f"Completed sidecar protocol has invalid stage timing: {stage}")
    cpu_utilization = float(protocol.get("cpu_utilization_percent", float("nan")))
    if not math.isfinite(cpu_utilization) or cpu_utilization < 0:
        raise ValueError("Completed sidecar protocol has invalid CPU utilization")


def verify_sidecar_dataset(
    output_dir: str | Path,
    *,
    verify_inputs: bool = True,
    require_training_authorization: bool = False,
    _allow_finalizing: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    protocol = json.loads((output / "protocol.json").read_text())
    accepted_statuses = {"completed", "finalizing"} if _allow_finalizing else {"completed"}
    if protocol.get("status") not in accepted_statuses or protocol.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("Rich-geometry sidecar protocol is not completed")
    if protocol.get("torsion_convention_version") != TORSION_CONVENTION_VERSION:
        raise ValueError("Rich-geometry sidecar torsion convention is incompatible")
    _verify_publication_contract(
        protocol,
        require_training_authorization=require_training_authorization,
    )
    if protocol["status"] == "completed":
        _verify_completed_timing(protocol)
    memberships: dict[str, set[str]] = {
        "eligibility_manifest.parquet": set(),
        "exclusions.parquet": set(),
        "train": set(),
        "validation": set(),
    }
    checked = 0
    eligibility_eligible: set[str] = set()
    eligibility_excluded: set[str] = set()
    eligibility_counts_by_split = {
        "train": {"total": 0, "eligible": 0, "excluded": 0},
        "validation": {"total": 0, "eligible": 0, "excluded": 0},
    }
    seen_shard_paths: set[Path] = set()
    expected_schemas = {
        "eligibility_manifest.parquet": eligibility_schema(),
        "exclusions.parquet": exclusion_schema(),
        "train": rich_geometry_schema(),
        "validation": rich_geometry_schema(),
    }
    for shard in protocol["shards"]:
        path = (output / str(shard["path"])).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError(f"Sidecar shard path escapes output directory: {shard['path']}")
        if path in seen_shard_paths:
            raise ValueError(f"Duplicate sidecar shard path: {shard['path']}")
        seen_shard_paths.add(path)
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"Sidecar shard hash contradiction: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(shard["row_count"]):
            raise ValueError(f"Sidecar shard row-count contradiction: {path}")
        shard_collection = str(shard["dataset"])
        if shard_collection not in expected_schemas:
            raise ValueError(f"Unknown sidecar dataset in protocol: {shard_collection}")
        if not parquet.schema_arrow.remove_metadata().equals(expected_schemas[shard_collection].remove_metadata()):
            raise ValueError(f"Sidecar schema contradiction: {path}")
        target = memberships[shard_collection]
        columns = ["sample_id", "split"]
        if shard_collection == "eligibility_manifest.parquet":
            columns.append("eligible")
        for batch in parquet.iter_batches(columns=columns, batch_size=4096):
            records = batch.to_pylist()
            for record in records:
                sample_id = str(record["sample_id"])
                split = str(record["split"])
                if split not in {"train", "validation"}:
                    raise ValueError(f"Unknown sidecar split for {sample_id}: {split!r}")
                if sample_id in target:
                    raise ValueError(f"Duplicate sample ID in {shard_collection}: {sample_id}")
                if shard_collection in {"train", "validation"} and split != shard_collection:
                    raise ValueError(f"Sidecar split contradiction for {sample_id}")
                target.add(sample_id)
                if shard_collection == "eligibility_manifest.parquet":
                    (eligibility_eligible if record["eligible"] else eligibility_excluded).add(sample_id)
                    eligibility_counts_by_split[split]["total"] += 1
                    status = "eligible" if record["eligible"] else "excluded"
                    eligibility_counts_by_split[split][status] += 1
        if shard_collection in {"train", "validation"} and parquet.metadata.num_rows:
            first_batch = next(parquet.iter_batches(batch_size=1))
            _verify_reopened_sidecar(first_batch.to_pylist()[0])
            checked += 1
    train = memberships["train"]
    validation = memberships["validation"]
    exclusions = memberships["exclusions.parquet"]
    all_samples = memberships["eligibility_manifest.parquet"]
    if train & validation or train & exclusions or validation & exclusions:
        raise ValueError("Rich-geometry output memberships are not disjoint")
    if train | validation | exclusions != all_samples:
        raise ValueError("Rich-geometry output membership union contradicts eligibility manifest")
    if train | validation != eligibility_eligible or exclusions != eligibility_excluded:
        raise ValueError("Rich-geometry eligibility decisions contradict output membership")
    if eligibility_counts_by_split != protocol.get("observed_split_counts"):
        raise ValueError("Rich-geometry split counts contradict protocol")
    counts = protocol["definitive_observed_counts"]
    if (len(all_samples), len(train) + len(validation), len(exclusions)) != (
        int(counts["total"]),
        int(counts["eligible"]),
        int(counts["excluded"]),
    ):
        raise ValueError("Rich-geometry output count contract contradiction")
    expected_hash_manifest = "".join(f"{item['sha256']}  {item['path']}\n" for item in protocol["shards"])
    if (output / "shard_hashes.sha256").read_text() != expected_hash_manifest:
        raise ValueError("Sidecar shard hash manifest contradiction")
    summaries = _shard_summaries(output, protocol["shards"])
    if summaries != protocol.get("shard_summaries_by_dataset"):
        raise ValueError("Sidecar shard summaries contradict verified files")
    journal_path = output / "construction_journal.sqlite"
    if not journal_path.is_file():
        raise ValueError("Rich-geometry construction journal is missing")
    connection = sqlite3.connect(journal_path)
    try:
        journal_mode_row = connection.execute("SELECT value FROM metadata WHERE key='construction_mode'").fetchone()
        journal_mode = str(journal_mode_row[0]) if journal_mode_row else None
        if journal_mode != protocol.get("mode"):
            raise ValueError("Protocol mode contradicts immutable construction-journal provenance")
        journal_status_row = connection.execute("SELECT value FROM metadata WHERE key='status'").fetchone()
        journal_status = str(journal_status_row[0]) if journal_status_row else None
        if protocol["status"] == "completed" and journal_status != "completed":
            raise ValueError("Completed protocol contradicts construction-journal completion state")
        observed = _verify_observed_inputs(connection) if verify_inputs else None
    finally:
        connection.close()
    if verify_inputs:
        phase0 = attest_phase0(protocol["phase0_protocol"], expected_sha256=protocol["phase0_protocol_sha256"])
        verify_protected_inputs(phase0)
        if observed != protocol.get("observed_phase1_input_verification"):
            raise ValueError("Observed Phase-1 input inventory contradicts protocol")
    return {"status": "verified", "shard_count": len(protocol["shards"]), "reopened_shard_count": checked}


def _verify_observed_inputs(connection: sqlite3.Connection) -> dict[str, Any]:
    inventory_hash = hashlib.sha256()
    count = 0
    rows = connection.execute("SELECT path,sha256 FROM observed_inputs ORDER BY path")
    for path_text, expected_digest in rows:
        path = Path(str(path_text))
        if not path.is_file() or sha256_file(path) != str(expected_digest):
            raise ValueError(f"Observed Phase-1 input changed during construction: {path}")
        inventory_hash.update(f"{path}\0{expected_digest}\n".encode())
        count += 1
    return {"count": count, "inventory_sha256": inventory_hash.hexdigest()}


def _torsion_failure_context(
    row: dict[str, Any],
    atom_masks: dict[str, np.ndarray],
    recomputed: dict[str, Any],
    *,
    name: str,
    index: int,
    stored_vector: np.ndarray,
    recomputed_vector: np.ndarray,
    stored_mask: bool,
    recomputed_mask: bool,
) -> dict[str, Any]:
    required_masks = {}
    for atom, offset in TORSION_DEFINITIONS[name]:
        atom_index = index + offset
        label = f"{atom}_{'i' if offset == 0 else f'i{offset:+d}'}"
        required_masks[label] = bool(atom_masks[atom][atom_index]) if 0 <= atom_index < len(atom_masks[atom]) else False
    continuity_index = index - 1 if name in {"phi", "omega"} else index
    continuity = None
    if 0 <= continuity_index < len(recomputed["chain_continuity_mask"]):
        continuity = {
            "bond_index": continuity_index,
            "stored": bool(row["chain_continuity_mask"][continuity_index]),
            "recomputed": bool(recomputed["chain_continuity_mask"][continuity_index]),
        }
    vector_error = None
    angular_error = None
    if stored_mask and recomputed_mask and np.isfinite(stored_vector).all() and np.isfinite(recomputed_vector).all():
        stored_norm = float(np.linalg.norm(stored_vector))
        recomputed_norm = float(np.linalg.norm(recomputed_vector))
        if stored_norm > 0 and recomputed_norm > 0:
            normalized_stored = stored_vector / stored_norm
            normalized_recomputed = recomputed_vector / recomputed_norm
            vector_error = float(np.linalg.norm(normalized_stored - normalized_recomputed))
            stored_angle = float(np.arctan2(normalized_stored[0], normalized_stored[1]))
            recomputed_angle = float(np.arctan2(normalized_recomputed[0], normalized_recomputed[1]))
            angular_error = abs(
                float(np.arctan2(np.sin(stored_angle - recomputed_angle), np.cos(stored_angle - recomputed_angle)))
            )
    return {
        "sample_id": str(row["sample_id"]),
        "split": str(row["split"]),
        "torsion_name": name,
        "residue_index": index,
        "residue_id": str(row["residue_ids"][index]),
        "stored_sin_cos": stored_vector.tolist(),
        "recomputed_sin_cos": recomputed_vector.tolist(),
        "stored_mask": stored_mask,
        "recomputed_mask": recomputed_mask,
        "vector_error": vector_error,
        "wrapped_angular_error_radians": angular_error,
        "required_atom_masks": required_masks,
        "continuity_mask_values": continuity,
    }


def _verify_reopened_sidecar(row: dict[str, Any]) -> None:
    length = len(str(row["sequence"]))
    residue_fields = ["token_ids", "residue_ids", "insertion_codes", "cb_source", "local_frame_valid"]
    residue_fields.extend(f"{atom.lower()}_{suffix}" for atom in ATOM_NAMES for suffix in ("coordinates", "mask"))
    residue_fields.extend(f"{name}_{suffix}" for name in ("phi", "psi", "omega") for suffix in ("sin_cos", "mask"))
    residue_fields.extend(
        [
            "residue_classification",
            "authoritative_parent_comp_id",
            "authoritative_parent_source",
            "selected_calpha_conformer",
            "selected_atom_conformers",
            "ca_candidate_counts",
        ]
    )
    contradictions = [field for field in residue_fields if len(row[field]) != length]
    if len(row["chain_continuity_mask"]) != max(length - 1, 0) or len(row["chain_break_mask"]) != max(length - 1, 0):
        contradictions.append("chain_masks")
    if contradictions:
        raise ValueError(f"Sidecar sequence/residue length contradiction: {contradictions}")
    vocabulary_size = len(SequenceGeometryVocabulary().tokens)
    if any(not 0 <= int(value) < vocabulary_size for value in row["token_ids"]):
        raise ValueError("Sidecar token ID is outside the canonical vocabulary")
    if row.get("torsion_convention_version") != TORSION_CONVENTION_VERSION:
        raise ValueError("Stored torsion convention version is unsupported")
    coordinates = {}
    atom_masks = {}
    for atom in ATOM_NAMES:
        values = np.asarray(row[f"{atom.lower()}_coordinates"], dtype=np.float64)
        mask = np.asarray(row[f"{atom.lower()}_mask"], dtype=np.bool_)
        if values.shape != (length, 3) or not np.isfinite(values[mask]).all():
            raise ValueError(f"Invalid masked {atom} coordinates")
        coordinates[atom] = values
        atom_masks[atom] = mask
    for index, valid in enumerate(row["local_frame_valid"]):
        if valid:
            frame, _ = local_frame(coordinates["N"][index], coordinates["CA"][index], coordinates["C"][index])
            if not frame_quality(frame)["frame_valid"]:
                raise ValueError("Stored local-frame validity contradicts coordinates")
    recomputed = derive_backbone_torsions(
        coordinates,
        atom_masks,
        peptide_bond_threshold_angstrom=float(row["peptide_bond_threshold_angstrom"]),
    )
    for field in ("chain_continuity_mask", "chain_break_mask"):
        if list(row[field]) != recomputed[field]:
            raise ValueError(f"Stored {field} contradicts coordinates")
    torsion_failures = []
    failure_counts = {}
    maximum_vector_error = 0.0
    for name in TORSION_DEFINITIONS:
        mask_field = f"{name}_mask"
        value_field = f"{name}_sin_cos"
        stored_masks = np.asarray(row[mask_field], dtype=np.bool_)
        recomputed_masks = np.asarray(recomputed[mask_field], dtype=np.bool_)
        stored_values = np.asarray(row[value_field], dtype=np.float64)
        recomputed_values = np.asarray(recomputed[value_field], dtype=np.float64)
        name_failures = 0
        for index in range(length):
            stored_mask = bool(stored_masks[index])
            recomputed_mask = bool(recomputed_masks[index])
            stored_vector = stored_values[index]
            recomputed_vector = recomputed_values[index]
            contradiction = stored_mask != recomputed_mask
            if stored_mask:
                norm = float(np.linalg.norm(stored_vector))
                contradiction |= not np.isfinite(stored_vector).all() or abs(norm - 1.0) > TORSION_FLOAT32_ATOL
            else:
                contradiction |= not np.allclose(
                    stored_vector,
                    TORSION_NEUTRAL_SIN_COS,
                    atol=TORSION_FLOAT32_ATOL,
                    rtol=0,
                )
            if stored_mask and recomputed_mask and np.isfinite(stored_vector).all():
                stored_norm = float(np.linalg.norm(stored_vector))
                recomputed_norm = float(np.linalg.norm(recomputed_vector))
                if stored_norm > 0 and recomputed_norm > 0:
                    vector_error = float(
                        np.linalg.norm(stored_vector / stored_norm - recomputed_vector / recomputed_norm)
                    )
                    maximum_vector_error = max(maximum_vector_error, vector_error)
                    contradiction |= vector_error > TORSION_FLOAT32_ATOL
            if contradiction:
                name_failures += 1
                if len(torsion_failures) < 100:
                    torsion_failures.append(
                        _torsion_failure_context(
                            row,
                            atom_masks,
                            recomputed,
                            name=name,
                            index=index,
                            stored_vector=stored_vector,
                            recomputed_vector=recomputed_vector,
                            stored_mask=stored_mask,
                            recomputed_mask=recomputed_mask,
                        )
                    )
        failure_counts[name] = name_failures
    if any(failure_counts.values()):
        raise ValueError(
            "Stored torsion features contradict coordinates: "
            + json.dumps(
                {
                    "sample_id": str(row["sample_id"]),
                    "split": str(row["split"]),
                    "failure_counts_by_torsion": failure_counts,
                    "maximum_vector_error": maximum_vector_error,
                    "comparison_absolute_tolerance": TORSION_FLOAT32_ATOL,
                    "examples": torsion_failures,
                },
                sort_keys=True,
            )
        )
    npz_path = Path(row["npz_path"])
    if sha256_file(npz_path) != row["npz_sha256"]:
        raise ValueError(f"Reopened sidecar NPZ hash contradiction: {npz_path}")
    npz = load_npz_geometry_evidence(npz_path)
    if tuple(row["residue_ids"]) != npz.residue_ids:
        raise ValueError("Reopened sidecar NPZ residue order contradiction")
    ca_mask = np.asarray(row["ca_mask"], dtype=np.bool_)
    if not ca_mask.all() or not np.allclose(coordinates["CA"], npz.ca_coordinates, atol=1e-4, rtol=0):
        raise ValueError("Reopened sidecar C-alpha anchors contradict immutable NPZ")


def construct_sidecars(
    config_path: str | Path,
    *,
    mode: str,
    resume: bool = False,
    maximum_rss_mib: float | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_sidecar_config(config)
    configured_maximum_rss_mib, effective_maximum_rss_mib, runtime_rss_override_applied = _runtime_rss_limits(
        config,
        maximum_rss_mib,
    )
    runtime_rss_metadata = {
        "configured_maximum_rss_mib": configured_maximum_rss_mib,
        "effective_maximum_rss_mib": effective_maximum_rss_mib,
        "runtime_rss_override_applied": runtime_rss_override_applied,
    }
    phase0 = attest_phase0(config["phase0_protocol"], expected_sha256=config.get("phase0_protocol_sha256"))
    input_hashes_before = verify_protected_inputs(phase0)
    if mode == "plan-only":
        return {
            "status": "planned",
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "torsion_convention_version": TORSION_CONVENTION_VERSION,
            "phase0_authorized": True,
            "authorizes_definitive_dataset": False,
            "authorizes_training": False,
            "protected_input_count": len(input_hashes_before),
            "dense_pair_features_stored": False,
            **runtime_rss_metadata,
        }
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be plan-only, pilot, or full")
    output = Path(config["pilot_output_dir"] if mode == "pilot" else config["output_dir"])
    if mode == "pilot" and output == Path(config["output_dir"]):
        raise ValueError("Pilot output must be separate from definitive output")
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite Phase-1 output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("train", "validation", "eligibility_manifest.parquet", "exclusions.parquet"):
        (output / directory).mkdir(exist_ok=True)
    config_hash = _config_hash(config, mode=mode, output=output)
    journal = ConstructionJournal(output / "construction_journal.sqlite", config_hash=config_hash, resume=resume)
    recorded_mode = journal.get("construction_mode")
    if recorded_mode is not None and recorded_mode != mode:
        journal.connection.close()
        raise ValueError("Construction mode contradicts resumable journal provenance")
    if recorded_mode is None:
        journal.set("construction_mode", mode)
    started_utc = journal.get("started_utc")
    if started_utc is None:
        started_utc = _utc_now()
        journal.set("started_utc", started_utc)
    journal.connection.commit()
    if journal.get("status") == "completed":
        try:
            verify_sidecar_dataset(output, verify_inputs=True)
            protocol = json.loads((output / "protocol.json").read_text())
            _heartbeat(
                output / "heartbeat.json",
                status="completed",
                stage="finalization",
                processed=int(journal.get("processed") or 0),
                protocol_sha256=sha256_file(output / "protocol.json"),
                **runtime_rss_metadata,
            )
            return protocol
        finally:
            journal.connection.close()
    heartbeat_path = output / "heartbeat.json"
    profiler_snapshot = journal.get("stage_profiler")
    profiler = (
        StageProfiler.from_snapshot(json.loads(profiler_snapshot))
        if profiler_snapshot is not None
        else StageProfiler(int(config.get("maximum_stage_latency_samples", 8192)))
    )
    prior_cpu_seconds = float(journal.get("profiled_cpu_seconds") or 0.0)
    process_cpu_started = time.process_time()
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous = signal.signal(signal.SIGINT, request_stop)
    processed = int(journal.get("processed") or 0)
    pairing_root = Path(config["pairing_dataset"])
    current_stage = "initialization"
    try:
        _heartbeat(
            heartbeat_path,
            status="running",
            stage=current_stage,
            processed=processed,
            **runtime_rss_metadata,
        )
        current_stage = "metadata_publication"
        _prepare_metadata(output, config, phase0)
        current_stage = "pilot_selection" if mode == "pilot" else "candidate_scanning"
        selected_ids = (
            _pilot_ids(
                pairing_root,
                int(config["pilot_sample_count"]),
                int(config["seed"]),
                profiler=profiler,
            )
            if mode == "pilot"
            else None
        )
        unit_index = -1
        for physical_dataset_name in ("eligible_train", "eligible_validation"):
            current_stage = "candidate_scanning"
            for input_unit in _input_units(
                pairing_root,
                physical_dataset_name,
                int(config["shard_size"]),
                profiler=profiler,
            ):
                canonical_split = input_unit.canonical_split
                unit_id = input_unit.input_unit_identity
                raw_rows = list(input_unit.rows)
                unit_index += 1
                if selected_ids is not None:
                    raw_rows = [row for row in raw_rows if (canonical_split, str(row["sample_id"])) in selected_ids]
                    if not raw_rows:
                        continue
                if journal.completed(unit_id):
                    continue
                current_stage = "provenance_enrichment"
                enriched = enrich_source_locators(
                    raw_rows,
                    processed_manifest=config["processed_manifest"],
                    audit_provenance=config["audit_provenance"],
                )
                enriched = [_assign_physical_split(row, input_unit=input_unit) for row in enriched]
                _enforce_memory(effective_maximum_rss_mib)
                decisions = []
                sidecars = []
                exclusions = []
                current_stage = "sample_evaluation"
                for outcome in _ordered_evaluations(enriched, config, canonical_split):
                    if stop_requested:
                        raise KeyboardInterrupt
                    row = outcome.row
                    profiler.extend(outcome.observations)
                    decision = outcome.decision
                    sidecar = outcome.sidecar
                    if outcome.error is not None:
                        error = outcome.error
                        npz_path = str(row.get("matrix_path") or "")
                        decision = {
                            "sample_id": str(row["sample_id"]),
                            "split": canonical_split,
                            "eligible": False,
                            "exclusion_reason": "unexplained_source_resolution_failure",
                            "source_path": str(row.get("source_file") or ""),
                            "source_sha256": None,
                            "npz_path": npz_path,
                            "npz_sha256": sha256_file(npz_path) if Path(npz_path).is_file() else "",
                            "sequence_length": len(str(row.get("sequence") or "")),
                            "coordinate_anchor_policy_version": COORDINATE_ANCHOR_POLICY_VERSION,
                            "diagnostics": json.dumps(
                                {"exception_type": type(error).__name__, "message": str(error)[:500]}
                            ),
                        }
                    assert decision is not None
                    decisions.append(decision)
                    if decision["split"] != canonical_split:
                        raise ValueError(f"Evaluation split contradiction for {decision['sample_id']}")
                    if sidecar is not None and sidecar["split"] != canonical_split:
                        raise ValueError(f"Sidecar split contradiction for {decision['sample_id']}")
                    if sidecar is not None:
                        sidecars.append(sidecar)
                    else:
                        exclusions.append(
                            {
                                "sample_id": decision["sample_id"],
                                "split": canonical_split,
                                "exclusion_reason": decision["exclusion_reason"],
                                "source_path": decision["source_path"],
                                "npz_path": decision["npz_path"],
                                "bounded_evidence": decision["diagnostics"],
                            }
                        )
                    _enforce_memory(effective_maximum_rss_mib)
                    if len(decisions) % int(config.get("heartbeat_frequency_samples", 25)) == 0:
                        _heartbeat(
                            heartbeat_path,
                            status="running",
                            stage="sample_evaluation",
                            processed=processed + len(decisions),
                            split=canonical_split,
                            unit_id=unit_id,
                            **runtime_rss_metadata,
                        )
                outputs = [
                    (f"eligibility_manifest.parquet/part-{unit_index:06d}.parquet", decisions, eligibility_schema()),
                    (
                        f"{canonical_split}/part-{unit_index:06d}.parquet",
                        sidecars,
                        rich_geometry_schema(),
                    ),
                    (f"exclusions.parquet/part-{unit_index:06d}.parquet", exclusions, exclusion_schema()),
                ]
                written = []
                current_stage = "shard_writing"
                for relative, rows, schema in outputs:
                    if not rows:
                        continue
                    count, digest = _write_part(
                        output / relative,
                        rows,
                        schema,
                        int(config["row_group_size"]),
                        profiler=profiler,
                    )
                    written.append((relative, relative.split("/")[0], count, digest))
                current_stage = "journal_commit"
                with journal.connection:
                    for decision in decisions:
                        journal.connection.execute(
                            "INSERT INTO decisions VALUES (?,?,?,?)",
                            (
                                decision["sample_id"],
                                canonical_split,
                                int(decision["eligible"]),
                                decision["exclusion_reason"],
                            ),
                        )
                        if decision["source_path"] and decision["source_sha256"]:
                            journal.record_observed_input(str(decision["source_path"]), str(decision["source_sha256"]))
                        if decision["npz_path"] and decision["npz_sha256"]:
                            journal.record_observed_input(str(decision["npz_path"]), str(decision["npz_sha256"]))
                    for relative, shard_collection, count, digest in written:
                        journal.connection.execute(
                            "INSERT INTO shards VALUES (?,?,?,?)",
                            (relative, shard_collection, count, digest),
                        )
                    journal.connection.execute(
                        "INSERT INTO completed_units VALUES (?,?,?,?,?)",
                        (
                            unit_id,
                            input_unit.physical_dataset_name,
                            canonical_split,
                            unit_index,
                            len(raw_rows),
                        ),
                    )
                    processed += len(raw_rows)
                    journal.set("processed", processed)
                    journal.set("last_committed_unit_id", unit_id)
                    journal.set("last_committed_unit_index", unit_index)
                    journal.set(
                        "last_committed_physical_dataset_name",
                        input_unit.physical_dataset_name,
                    )
                    journal.set("last_committed_split", canonical_split)
                    journal.set("stage_profiler", json.dumps(profiler.snapshot(), sort_keys=True))
                    journal.set(
                        "profiled_cpu_seconds",
                        prior_cpu_seconds + time.process_time() - process_cpu_started,
                    )
                _heartbeat(
                    heartbeat_path,
                    status="running",
                    stage="construction",
                    processed=processed,
                    **runtime_rss_metadata,
                )
        current_stage = "count_validation"
        duplicate = journal.connection.execute(
            "SELECT sample_id FROM decisions GROUP BY sample_id HAVING COUNT(*)>1 LIMIT 1"
        ).fetchone()
        if duplicate is not None:
            raise ValueError(f"Sample ID belongs to both physical splits: {duplicate[0]}")
        counts = {
            "total": journal.connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
            "eligible": journal.connection.execute("SELECT COUNT(*) FROM decisions WHERE eligible=1").fetchone()[0],
            "excluded": journal.connection.execute("SELECT COUNT(*) FROM decisions WHERE eligible=0").fetchone()[0],
        }
        split_counts = dict(
            journal.connection.execute("SELECT split,COUNT(*) FROM decisions WHERE eligible=1 GROUP BY split")
        )
        observed_split_counts = {
            "train": {"total": 0, "eligible": 0, "excluded": 0},
            "validation": {"total": 0, "eligible": 0, "excluded": 0},
        }
        for split, total, eligible in journal.connection.execute(
            "SELECT split,COUNT(*),SUM(eligible) FROM decisions GROUP BY split ORDER BY split"
        ):
            observed_split_counts[str(split)] = {
                "total": int(total),
                "eligible": int(eligible),
                "excluded": int(total) - int(eligible),
            }
        reason_counts = dict(
            journal.connection.execute(
                "SELECT exclusion_reason,COUNT(*) FROM decisions WHERE eligible=0 GROUP BY exclusion_reason"
            )
        )
        unexplained_failure_count = int(reason_counts.get("unexplained_source_resolution_failure", 0))
        shards = [
            {"path": path, "dataset": dataset, "row_count": count, "sha256": digest}
            for path, dataset, count, digest in journal.connection.execute(
                "SELECT path,dataset,row_count,sha256 FROM shards ORDER BY path"
            )
        ]
        _atomic_text(
            output / "shard_hashes.sha256",
            "".join(f"{item['sha256']}  {item['path']}\n" for item in shards),
        )
        shard_summaries = _shard_summaries(output, shards)
        sample_order_sha256 = _sample_order_sha256(output, shards)
        current_stage = "cuda_feature_equivalence"
        effective_feature_backend, feature_backend_fallback_reason, peak_cuda_allocated_mib = (
            _validate_cuda_features_from_shards(output, shards, config, profiler)
        )
        current_stage = "input_reverification"
        hashing_started = time.perf_counter()
        input_hashes_after = verify_protected_inputs(phase0)
        if input_hashes_after != input_hashes_before:
            raise RuntimeError("Protected source inputs changed during Phase-1 construction")
        observed_input_verification = _verify_observed_inputs(journal.connection)
        protected_bytes = sum(Path(path).stat().st_size for path in input_hashes_before)
        observed_bytes = sum(
            Path(str(row[0])).stat().st_size for row in journal.connection.execute("SELECT path FROM observed_inputs")
        )
        profiler.add(
            StageObservation(
                "hashing",
                time.perf_counter() - hashing_started,
                sample_count=int(counts["total"]),
                bytes_read=protected_bytes + observed_bytes,
            )
        )
        finalizing_utc = _utc_now()
        elapsed = _elapsed_seconds(started_utc, finalizing_utc)
        throughput_denominator = max(elapsed, 1e-9)
        protocol = {
            "status": "finalizing",
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "torsion_convention_version": TORSION_CONVENTION_VERSION,
            "mode": mode,
            "authorizes_definitive_dataset": False,
            "authorizes_training": False,
            "authorizes_full_dataset": False,
            "phase0_protocol": str(Path(config["phase0_protocol"])),
            "phase0_protocol_sha256": sha256_file(Path(config["phase0_protocol"])),
            "phase0_authorization": phase0["phase1_authorization"],
            "phase0_panel_projection_used_for_eligibility": False,
            "definitive_observed_counts": counts,
            "eligible_split_counts": split_counts,
            "observed_split_counts": observed_split_counts,
            "exclusion_reason_counts": reason_counts,
            "unexplained_failure_count": unexplained_failure_count,
            "shards": shards,
            "shard_summaries_by_dataset": shard_summaries,
            "sample_order_sha256": sample_order_sha256,
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "protected_inputs_unchanged": True,
            "observed_phase1_inputs_unchanged": True,
            "observed_phase1_input_verification": observed_input_verification,
            "configuration_sha256": sha256_file(Path(config_path)),
            "configuration_hash": config_hash,
            "peak_rss_mib": _peak_rss_mib(),
            **runtime_rss_metadata,
            "peak_cuda_allocated_mib": peak_cuda_allocated_mib,
            "worker_count": int(config.get("worker_count", 1)),
            "worker_queue_size": int(config.get("worker_queue_size", 2)),
            "requested_feature_backend": str(config.get("feature_backend", "cpu")),
            "effective_feature_backend": effective_feature_backend,
            "mmcif_parse_mode": str(config.get("mmcif_parse_mode", "single_parse")),
            "feature_backend_fallback_reason": feature_backend_fallback_reason,
            "cuda_feature_equivalence_passed": (
                effective_feature_backend == "cuda" if config.get("feature_backend", "cpu") == "cuda" else None
            ),
            "cuda_feature_float32_tolerance": CUDA_FEATURE_FLOAT32_ATOL,
            "started_utc": started_utc,
            "completed_utc": None,
            "elapsed_seconds": elapsed,
            "processed_samples": int(counts["total"]),
            "samples_per_second": int(counts["total"]) / throughput_denominator,
            "eligible_samples_per_second": int(counts["eligible"]) / throughput_denominator,
        }
        current_stage = "final_verification"
        _atomic_json(output / "protocol.json", protocol)
        verification_started = time.perf_counter()
        verify_sidecar_dataset(output, verify_inputs=True, _allow_finalizing=True)
        profiler.add(
            StageObservation(
                "final_verification",
                time.perf_counter() - verification_started,
                sample_count=int(counts["total"]),
            )
        )
        journal.set("status", "verification_passed")
        journal.connection.commit()
        completed_utc = _utc_now()
        elapsed = _elapsed_seconds(started_utc, completed_utc)
        throughput_denominator = max(elapsed, 1e-9)
        full_authorized = (
            mode == "full" and input_hashes_after == input_hashes_before and unexplained_failure_count == 0
        )
        protocol["status"] = "completed"
        protocol["authorizes_definitive_dataset"] = full_authorized
        protocol["authorizes_training"] = full_authorized
        protocol["authorizes_full_dataset"] = full_authorized
        protocol["completed_utc"] = completed_utc
        protocol["elapsed_seconds"] = elapsed
        protocol["samples_per_second"] = int(counts["total"]) / throughput_denominator
        protocol["eligible_samples_per_second"] = int(counts["eligible"]) / throughput_denominator
        protocol["stage_timing"] = profiler.report(elapsed)
        protocol["cpu_utilization_percent"] = (
            100.0 * (prior_cpu_seconds + time.process_time() - process_cpu_started) / max(elapsed, 1e-9)
        )
        _atomic_json(output / "protocol.json", protocol)
        journal.set("status", "completed")
        journal.connection.commit()
        _heartbeat(
            heartbeat_path,
            status="completed",
            stage="finalization",
            processed=processed,
            protocol_sha256=sha256_file(output / "protocol.json"),
            **runtime_rss_metadata,
        )
        return protocol
    except KeyboardInterrupt as error:
        status = "interrupted"
        journal.set("status", status)
        journal.connection.commit()
        _publish_failure_heartbeat(
            heartbeat_path,
            status=status,
            stage=current_stage,
            processed=processed,
            error=error,
            journal=journal,
            resumable=True,
            runtime_rss_metadata=runtime_rss_metadata,
        )
        raise
    except MemoryError as error:
        status = "memory_limit_exceeded"
        journal.set("status", status)
        journal.connection.commit()
        _publish_failure_heartbeat(
            heartbeat_path,
            status=status,
            stage=current_stage,
            processed=processed,
            error=error,
            journal=journal,
            resumable=True,
            runtime_rss_metadata=runtime_rss_metadata,
        )
        raise
    except Exception as error:
        status = "failed"
        journal.set("status", status)
        journal.connection.commit()
        _publish_failure_heartbeat(
            heartbeat_path,
            status=status,
            stage=current_stage,
            processed=processed,
            error=error,
            journal=journal,
            resumable=False,
            runtime_rss_metadata=runtime_rss_metadata,
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous)
        journal.connection.close()


def _benchmark_case_summary(name: str, protocol: dict[str, Any], full_sample_count: int) -> dict[str, Any]:
    throughput = float(protocol["samples_per_second"])
    return {
        "name": name,
        "status": protocol["status"],
        "worker_count": protocol["worker_count"],
        "requested_feature_backend": protocol["requested_feature_backend"],
        "effective_feature_backend": protocol["effective_feature_backend"],
        "mmcif_parse_mode": protocol["mmcif_parse_mode"],
        "feature_backend_fallback_reason": protocol["feature_backend_fallback_reason"],
        "cuda_feature_equivalence_passed": protocol["cuda_feature_equivalence_passed"],
        "cuda_feature_float32_tolerance": protocol["cuda_feature_float32_tolerance"],
        "samples_per_second": throughput,
        "eligible_samples_per_second": protocol["eligible_samples_per_second"],
        "projected_full_build_hours": full_sample_count / max(throughput, 1e-12) / 3600,
        "peak_rss_mib": protocol["peak_rss_mib"],
        "peak_cuda_allocated_mib": protocol["peak_cuda_allocated_mib"],
        "cpu_utilization_percent": protocol["cpu_utilization_percent"],
        "stage_timing": protocol["stage_timing"],
        "sample_order_sha256": protocol["sample_order_sha256"],
        "definitive_observed_counts": protocol["definitive_observed_counts"],
        "observed_split_counts": protocol["observed_split_counts"],
        "exclusion_reason_counts": protocol["exclusion_reason_counts"],
        "shard_scientific_hashes": {item["path"]: item["sha256"] for item in protocol["shards"]},
        "authorizes_definitive_dataset": protocol["authorizes_definitive_dataset"],
        "authorizes_training": protocol["authorizes_training"],
    }


def _safe_fallback_configuration() -> dict[str, Any]:
    return {
        "case": "baseline_cpu_1",
        "recommendation_type": "verified_safe_fallback",
        "optimization_winner": False,
        "worker_count": 1,
        "worker_queue_size": 2,
        "feature_backend": "cpu",
        "mmcif_parse_mode": "legacy_double_parse",
    }


def _run_benchmark_case(
    name: str,
    case_config: dict[str, Any],
    case_path: Path,
    *,
    full_sample_count: int,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _atomic_json(case_path, case_config)
    sampler = CaseMemorySampler()
    sampler.start()
    try:
        protocol = construct_sidecars(
            case_path,
            mode="pilot",
            resume=resume and Path(case_config["pilot_output_dir"]).exists(),
        )
    finally:
        memory = sampler.finish()
    if float(memory["case_peak_rss_mib"]) > float(case_config["maximum_rss_mib"]):
        raise MemoryError(
            f"E006 benchmark case RSS limit exceeded: case={name}, "
            f"case_peak={float(memory['case_peak_rss_mib']):.1f} MiB, "
            f"process_peak={float(memory['process_peak_rss_mib']):.1f} MiB, "
            f"limit={float(case_config['maximum_rss_mib']):.1f} MiB"
        )
    summary = _benchmark_case_summary(name, protocol, full_sample_count)
    summary["protocol_reported_process_peak_rss_mib"] = summary.pop("peak_rss_mib")
    summary.update(memory)
    return summary, protocol


def _publish_benchmark_failure(
    output_root: Path,
    heartbeat_path: Path,
    *,
    failed_case: str,
    completed_cases: list[dict[str, Any]],
    error: BaseException,
    resumable: bool,
) -> None:
    completed_utc = _utc_now()
    partial_report = {
        "status": "failed",
        "mode": "non_authorizing_performance_benchmark_v2",
        "failed_case": failed_case,
        "completed_cases": completed_cases,
        "completed_case_count": len(completed_cases),
        "exception": {"type": type(error).__name__, "message": str(error)[:1000]},
        "resumable": resumable,
        "resumability_status": "resume_supported" if resumable else "new_output_required",
        "fallback_production_configuration": _safe_fallback_configuration(),
        "authorizes_definitive_dataset": False,
        "authorizes_training": False,
        "completed_utc": completed_utc,
    }
    partial_path = output_root / "benchmark_partial_report.json"
    _atomic_json(partial_path, partial_report)
    _atomic_json(
        heartbeat_path,
        {
            "status": "failed",
            "completed_utc": completed_utc,
            "failed_case": failed_case,
            "completed_cases": [item["name"] for item in completed_cases],
            "completed_case_count": len(completed_cases),
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:1000],
            "partial_report_path": str(partial_path),
            "partial_report_sha256": sha256_file(partial_path),
            "resumable": resumable,
            "resumability_status": "resume_supported" if resumable else "new_output_required",
            "fallback_production_configuration": _safe_fallback_configuration(),
        },
    )


def _benchmark_equivalence(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "sample_order_sha256",
        "definitive_observed_counts",
        "observed_split_counts",
        "exclusion_reason_counts",
        "shard_scientific_hashes",
    )
    mismatches = [field for field in fields if reference[field] != candidate[field]]
    authorization_safe = not candidate["authorizes_definitive_dataset"] and not candidate["authorizes_training"]
    return {
        "equivalent": not mismatches and authorization_safe,
        "mismatching_fields": mismatches,
        "identical_membership_and_order": reference["sample_order_sha256"] == candidate["sample_order_sha256"],
        "identical_eligibility_decisions": (
            reference["definitive_observed_counts"] == candidate["definitive_observed_counts"]
            and reference["exclusion_reason_counts"] == candidate["exclusion_reason_counts"]
        ),
        "identical_tensor_shards": reference["shard_scientific_hashes"] == candidate["shard_scientific_hashes"],
        "float32_tolerance": CUDA_FEATURE_FLOAT32_ATOL,
        "authorization_safe": authorization_safe,
    }


def run_performance_benchmark(config_path: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Run the non-authorizing deterministic E006 Phase-1 v2 sweep."""
    base = load_yaml(config_path)
    validate_sidecar_config(base)
    benchmark = dict(base.get("performance_benchmark") or {})
    if int(benchmark.get("version", 2)) != 2:
        raise ValueError("E006 performance benchmark configuration version must be 2")
    panel_size = int(benchmark.get("panel_sample_count", 512))
    if panel_size != 512:
        raise ValueError("E006 performance benchmark panel_sample_count must be 512")
    output_root = Path(benchmark["output_root"])
    if output_root in {Path(base["pilot_output_dir"]), Path(base["output_dir"])}:
        raise ValueError("Performance benchmark output must not overlap pilot or definitive output")
    if output_root.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite E006 performance benchmark: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_heartbeat = output_root / "heartbeat.json"
    _atomic_json(
        benchmark_heartbeat,
        {"status": "running", "stage": "configuration_sweep", "completed_cases": 0, "updated_utc": _utc_now()},
    )
    cases = [
        ("baseline_cpu_1", 1, "legacy_double_parse"),
        ("cpu_1_single_parse", 1, "single_parse"),
        ("cpu_2_workers", 2, "single_parse"),
    ]
    full_sample_count = int(benchmark.get("projected_full_sample_count", 293405))
    try:
        case_summaries: list[dict[str, Any]] = []
        baseline: dict[str, Any] | None = None
        current_case = "initialization"
        for name, workers, parse_mode in cases:
            current_case = name
            case_config = copy.deepcopy(base)
            case_config.update(
                pilot_sample_count=panel_size,
                pilot_output_dir=str(output_root / "runs" / name),
                maximum_rss_mib=float(benchmark.get("maximum_rss_mib", 4096)),
                worker_count=workers,
                worker_queue_size=max(workers, int(benchmark.get("worker_queue_multiplier", 2)) * workers),
                feature_backend="cpu",
                mmcif_parse_mode=parse_mode,
            )
            summary, _protocol = _run_benchmark_case(
                name,
                case_config,
                output_root / "configs" / f"{name}.json",
                full_sample_count=full_sample_count,
                resume=resume,
            )
            if baseline is None:
                baseline = summary
                summary["equivalence_to_baseline"] = {"equivalent": True, "mismatching_fields": []}
            else:
                summary["equivalence_to_baseline"] = _benchmark_equivalence(baseline, summary)
                summary["speedup_fraction_over_baseline"] = (
                    float(summary["samples_per_second"]) / float(baseline["samples_per_second"]) - 1.0
                )
            case_summaries.append(summary)
            _atomic_json(
                benchmark_heartbeat,
                {
                    "status": "running",
                    "stage": "configuration_sweep",
                    "completed_cases": [item["name"] for item in case_summaries],
                    "completed_case_count": len(case_summaries),
                    "latest_case": name,
                    "updated_utc": _utc_now(),
                },
            )

        assert baseline is not None
        feature_fraction = (
            float(baseline["stage_timing"]["frame_torsion_cb_feature_derivation"]["percentage_of_runtime"]) / 100.0
        )
        cuda_threshold = float(benchmark.get("minimum_cuda_feature_stage_fraction", 0.20))
        case_summaries.append(
            {
                "name": "cuda_production_features",
                "status": "not_scheduled",
                "production_backend": False,
                "reason": "insufficient_feature_stage_acceleration_headroom",
                "baseline_feature_stage_fraction": feature_fraction,
                "minimum_feature_stage_fraction": cuda_threshold,
                "post_hoc_verification_counts_as_acceleration": False,
                "v1_pseudo_cb_failure_diagnosis": (
                    "CPU pseudo-CB values were derived from source float64 coordinates before float32 storage, "
                    "while diagnostic CUDA values were derived from stored float32 backbone coordinates; "
                    "v2 compares only cb_source=0 residues and emits bounded coordinate/dtype evidence."
                ),
                "authorizes_definitive_dataset": False,
                "authorizes_training": False,
            }
        )

        minimum_speedup = float(benchmark.get("minimum_end_to_end_speedup_fraction", 0.20))
        eligible_candidates = [
            item
            for item in case_summaries[1:]
            if item["status"] == "completed"
            and item["equivalence_to_baseline"]["equivalent"]
            and float(item["samples_per_second"]) >= float(baseline["samples_per_second"]) * (1.0 + minimum_speedup)
        ]
        winner = (
            max(eligible_candidates, key=lambda item: float(item["samples_per_second"]))
            if eligible_candidates
            else None
        )
        production = (
            {
                "case": winner["name"],
                "recommendation_type": "measured_optimization_winner",
                "optimization_winner": True,
                "worker_count": winner["worker_count"],
                "worker_queue_size": winner["worker_count"] * int(benchmark.get("worker_queue_multiplier", 2)),
                "feature_backend": winner["effective_feature_backend"],
                "mmcif_parse_mode": winner["mmcif_parse_mode"],
                "measured_speedup_fraction": (
                    float(winner["samples_per_second"]) / float(baseline["samples_per_second"]) - 1.0
                ),
            }
            if winner is not None
            else _safe_fallback_configuration()
        )
        report = {
            "status": "completed",
            "mode": "non_authorizing_performance_benchmark_v2",
            "panel_sample_count": panel_size,
            "minimum_end_to_end_speedup_fraction": minimum_speedup,
            "baseline_samples_per_second": baseline["samples_per_second"],
            "cases": case_summaries,
            "optimization_winner": winner["name"] if winner is not None else None,
            "recommended_production_configuration": production,
            "fallback_production_configuration": _safe_fallback_configuration(),
            "selection_reason": (
                "equivalent_and_at_least_20_percent_faster"
                if winner is not None
                else "safe_fallback_retained_no_optimized_case_met_speedup_gate"
            ),
            "authorizes_definitive_dataset": False,
            "authorizes_training": False,
            "completed_utc": _utc_now(),
        }
        report_path = output_root / "benchmark_report.json"
        _atomic_json(report_path, report)
        _atomic_json(
            benchmark_heartbeat,
            {
                "status": "completed",
                "stage": "finalization",
                "completed_utc": report["completed_utc"],
                "completed_cases": [item["name"] for item in case_summaries],
                "completed_case_count": len(case_summaries),
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "fallback_production_configuration": _safe_fallback_configuration(),
            },
        )
        return report
    except KeyboardInterrupt as error:
        completed = locals().get("case_summaries", [])
        _publish_benchmark_failure(
            output_root,
            benchmark_heartbeat,
            failed_case=locals().get("current_case", "initialization"),
            completed_cases=completed,
            error=error,
            resumable=True,
        )
        raise
    except Exception as error:
        completed = locals().get("case_summaries", [])
        _publish_benchmark_failure(
            output_root,
            benchmark_heartbeat,
            failed_case=locals().get("current_case", "initialization"),
            completed_cases=completed,
            error=error,
            resumable=isinstance(error, MemoryError),
        )
        raise
