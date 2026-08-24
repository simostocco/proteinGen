"""Bounded-memory training normalization statistics."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.data.preprocess import load_manifest
from protein_distance_diffusion.utils.hashing import sha256_file

STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HistogramConfig:
    percentile: float = 95.0
    histogram_bin_width_angstrom: float = 0.05
    histogram_max_distance_angstrom: float = 2000.0
    overflow_fraction_tolerance: float = 0.0


@dataclass
class HistogramState:
    config: HistogramConfig
    histogram: np.ndarray
    valid_distance_count: int = 0
    overflow_distance_count: int = 0
    negative_distance_count: int = 0
    nonfinite_distance_count: int = 0
    zero_distance_count: int = 0
    sum_distance: float = 0.0
    sum_sq_distance: float = 0.0
    observed_min_distance_angstrom: float | None = None
    observed_max_distance_angstrom: float = 0.0
    processed_samples: int = 0
    rejected_sample_count: int = 0
    next_row_index: int = 0
    total_manifest_rows: int = 0
    train_manifest_path: str | None = None
    train_manifest_sha256: str | None = None
    config_hash: str | None = None
    completed: bool = False
    rejected_samples: list[dict[str, Any]] | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _config_hash(config: HistogramConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _manifest_metadata(path: str | Path, row_count: int, config: HistogramConfig) -> dict[str, Any]:
    manifest = Path(path)
    return {
        "train_manifest_path": str(manifest.resolve()),
        "train_manifest_sha256": sha256_file(manifest),
        "total_manifest_rows": int(row_count),
        "config_hash": _config_hash(config),
    }


def _empty_state(config: HistogramConfig) -> HistogramState:
    bins = int(math.ceil(config.histogram_max_distance_angstrom / config.histogram_bin_width_angstrom))
    return HistogramState(config=config, histogram=np.zeros(bins, dtype=np.int64), rejected_samples=[])


def _valid_values(row: pd.Series) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    sample_id = str(row.get("sample_id", ""))
    path = Path(str(row["path"]))
    if not path.exists():
        return None, {"sample_id": sample_id, "path": str(path), "reason": "missing_npz"}
    try:
        data = np.load(path, allow_pickle=False)
        matrix = np.asarray(data["distance_matrix"], dtype=np.float32)
    except Exception:
        return None, {"sample_id": sample_id, "path": str(path), "reason": "corrupt_npz"}
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None, {"sample_id": sample_id, "path": str(path), "reason": "non_square_distance_matrix"}
    n = int(row.get("length", matrix.shape[0]))
    residue_mask = np.ones(matrix.shape[0], dtype=bool)
    if "residue_mask" in data:
        residue_mask = np.asarray(data["residue_mask"], dtype=bool)
    residue_mask[n:] = False
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    upper = np.triu(pair_mask, k=1)
    values = matrix[upper].astype(np.float64)
    if np.any(~np.isfinite(values)):
        return None, {"sample_id": sample_id, "path": str(path), "reason": "nonfinite_distance"}
    if np.any(values < 0):
        return None, {"sample_id": sample_id, "path": str(path), "reason": "negative_distance"}
    return values, None


def _chunk_statistics(chunk: pd.DataFrame, config: HistogramConfig) -> HistogramState:
    state = _empty_state(config)
    assert state.rejected_samples is not None
    for _, row in chunk.iterrows():
        values, rejection = _valid_values(row)
        state.processed_samples += 1
        if rejection is not None:
            state.rejected_samples.append(rejection)
            state.rejected_sample_count += 1
            continue
        if values is None or values.size == 0:
            continue
        state.zero_distance_count += int((values == 0).sum())
        state.nonfinite_distance_count += int((~np.isfinite(values)).sum())
        state.negative_distance_count += int((values < 0).sum())
        overflow = values >= config.histogram_max_distance_angstrom
        state.overflow_distance_count += int(overflow.sum())
        values = values[~overflow]
        if values.size == 0:
            continue
        indices = np.floor(values / config.histogram_bin_width_angstrom).astype(np.int64)
        indices = np.clip(indices, 0, len(state.histogram) - 1)
        state.histogram += np.bincount(indices, minlength=len(state.histogram))
        state.valid_distance_count += int(values.size)
        state.sum_distance += float(values.sum())
        state.sum_sq_distance += float(np.square(values).sum())
        observed_min = float(values.min(initial=float("inf")))
        if np.isfinite(observed_min):
            state.observed_min_distance_angstrom = (
                observed_min
                if state.observed_min_distance_angstrom is None
                else min(state.observed_min_distance_angstrom, observed_min)
            )
        state.observed_max_distance_angstrom = max(state.observed_max_distance_angstrom, float(values.max(initial=0.0)))
    return state


def _merge(left: HistogramState, right: HistogramState) -> HistogramState:
    left.histogram += right.histogram
    left.valid_distance_count += right.valid_distance_count
    left.overflow_distance_count += right.overflow_distance_count
    left.negative_distance_count += right.negative_distance_count
    left.nonfinite_distance_count += right.nonfinite_distance_count
    left.zero_distance_count += right.zero_distance_count
    left.sum_distance += right.sum_distance
    left.sum_sq_distance += right.sum_sq_distance
    if right.observed_min_distance_angstrom is not None:
        left.observed_min_distance_angstrom = (
            right.observed_min_distance_angstrom
            if left.observed_min_distance_angstrom is None
            else min(left.observed_min_distance_angstrom, right.observed_min_distance_angstrom)
        )
    left.observed_max_distance_angstrom = max(left.observed_max_distance_angstrom, right.observed_max_distance_angstrom)
    left.processed_samples += right.processed_samples
    left.rejected_sample_count += right.rejected_sample_count
    assert left.rejected_samples is not None and right.rejected_samples is not None
    left.rejected_samples.extend(right.rejected_samples)
    return left


def _state_paths(output: str | Path) -> tuple[Path, Path]:
    out = Path(output)
    return out.with_suffix(".state.npz"), out.with_suffix(".state.json")


def _write_state(output: str | Path, state: HistogramState, *, manifest_path: str | Path) -> None:
    npz, js = _state_paths(output)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz.with_name(f".{npz.name}.{os.getpid()}.tmp.npz")
    tmp_json = js.with_name(f".{js.name}.{os.getpid()}.tmp")
    edges = np.arange(len(state.histogram) + 1, dtype=np.float64) * state.config.histogram_bin_width_angstrom
    np.savez_compressed(tmp_npz, histogram=state.histogram, histogram_edges=edges)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "config": asdict(state.config),
        "config_hash": state.config_hash,
        "manifest_path": str(manifest_path),
        "train_manifest_path": state.train_manifest_path,
        "train_manifest_sha256": state.train_manifest_sha256,
        "total_manifest_rows": state.total_manifest_rows,
        "next_row_index": state.next_row_index,
        "valid_distance_count": state.valid_distance_count,
        "overflow_distance_count": state.overflow_distance_count,
        "negative_distance_count": state.negative_distance_count,
        "nonfinite_distance_count": state.nonfinite_distance_count,
        "zero_distance_count": state.zero_distance_count,
        "sum_distance": state.sum_distance,
        "sum_sq_distance": state.sum_sq_distance,
        "observed_min_distance_angstrom": state.observed_min_distance_angstrom,
        "observed_max_distance_angstrom": state.observed_max_distance_angstrom,
        "processed_samples": state.processed_samples,
        "rejected_sample_count": state.rejected_sample_count,
        "rejected_samples": state.rejected_samples or [],
        "completed": state.completed,
        "updated_utc": _utc_now(),
    }
    tmp_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_npz.replace(npz)
    tmp_json.replace(js)


def _load_state(output: str | Path, config: HistogramConfig) -> HistogramState:
    npz, js = _state_paths(output)
    payload = json.loads(js.read_text())
    if int(payload.get("schema_version", 0)) != STATE_SCHEMA_VERSION:
        raise ValueError("statistics state schema version is incompatible; use restart=True to recompute")
    if payload["config"] != asdict(config):
        raise ValueError("statistics state configuration differs")
    state = _empty_state(config)
    state.histogram = np.load(npz)["histogram"].astype(np.int64)
    state.valid_distance_count = int(payload["valid_distance_count"])
    state.overflow_distance_count = int(payload["overflow_distance_count"])
    state.negative_distance_count = int(payload.get("negative_distance_count", 0))
    state.nonfinite_distance_count = int(payload.get("nonfinite_distance_count", 0))
    state.zero_distance_count = int(payload.get("zero_distance_count", 0))
    state.sum_distance = float(payload["sum_distance"])
    state.sum_sq_distance = float(payload["sum_sq_distance"])
    state.observed_min_distance_angstrom = payload.get("observed_min_distance_angstrom")
    if state.observed_min_distance_angstrom is not None:
        state.observed_min_distance_angstrom = float(state.observed_min_distance_angstrom)
    state.observed_max_distance_angstrom = float(payload["observed_max_distance_angstrom"])
    state.processed_samples = int(payload["processed_samples"])
    state.rejected_sample_count = int(payload.get("rejected_sample_count", len(payload.get("rejected_samples", []))))
    state.next_row_index = int(payload.get("next_row_index", state.processed_samples))
    state.total_manifest_rows = int(payload.get("total_manifest_rows", 0))
    state.train_manifest_path = payload.get("train_manifest_path")
    state.train_manifest_sha256 = payload.get("train_manifest_sha256")
    state.config_hash = payload.get("config_hash")
    state.completed = bool(payload.get("completed", False))
    state.rejected_samples = list(payload.get("rejected_samples", []))
    return state


def _attach_metadata(state: HistogramState, metadata: dict[str, Any]) -> HistogramState:
    state.train_manifest_path = str(metadata["train_manifest_path"])
    state.train_manifest_sha256 = str(metadata["train_manifest_sha256"])
    state.total_manifest_rows = int(metadata["total_manifest_rows"])
    state.config_hash = str(metadata["config_hash"])
    return state


def _verify_state_compatible(state: HistogramState, metadata: dict[str, Any], config: HistogramConfig) -> None:
    expected = {
        "train_manifest_sha256": metadata["train_manifest_sha256"],
        "total_manifest_rows": int(metadata["total_manifest_rows"]),
        "config_hash": metadata["config_hash"],
    }
    actual = {
        "train_manifest_sha256": state.train_manifest_sha256,
        "total_manifest_rows": state.total_manifest_rows,
        "config_hash": state.config_hash,
    }
    if actual != expected or state.config != config:
        raise ValueError(
            "statistics state is incompatible with the requested manifest/configuration; "
            "use restart=True or --restart to recompute"
        )
    if not 0 <= state.next_row_index <= state.total_manifest_rows:
        raise ValueError("statistics state has invalid next_row_index; use restart=True or --restart to recompute")


def _load_final_output(path: Path, metadata: dict[str, Any], config: HistogramConfig) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("train_manifest_sha256") != metadata["train_manifest_sha256"]:
        raise ValueError("existing normalization JSON was computed from a different train manifest")
    if payload.get("config_hash") != metadata["config_hash"] or payload.get("percentile") != config.percentile:
        raise ValueError(
            "existing normalization JSON configuration differs; use restart=True or --restart to recompute"
        )
    return payload


def _finalize(state: HistogramState) -> dict[str, Any]:
    if state.valid_distance_count <= 0:
        raise ValueError("Manifest contains no samples with valid off-diagonal distances")
    if state.next_row_index != state.total_manifest_rows:
        raise RuntimeError("Cannot finalize incomplete normalization statistics state")
    overflow_total = state.valid_distance_count + state.overflow_distance_count
    overflow_fraction = state.overflow_distance_count / max(overflow_total, 1)
    if overflow_fraction > state.config.overflow_fraction_tolerance:
        raise RuntimeError("Histogram overflow exceeded tolerance")
    cdf = np.cumsum(state.histogram)
    target = state.config.percentile / 100.0 * state.valid_distance_count
    idx = int(np.searchsorted(cdf, target, side="left"))
    scale = (idx + 0.5) * state.config.histogram_bin_width_angstrom
    mean = state.sum_distance / state.valid_distance_count
    variance = max(state.sum_sq_distance / state.valid_distance_count - mean * mean, 0.0)
    return {
        "mode": "scale",
        "scale": float(scale),
        "percentile": float(state.config.percentile),
        "histogram_bin_width_angstrom": float(state.config.histogram_bin_width_angstrom),
        "histogram_max_distance_angstrom": float(state.config.histogram_max_distance_angstrom),
        "percentile_error_bound_angstrom": float(state.config.histogram_bin_width_angstrom),
        "valid_distance_count": int(state.valid_distance_count),
        "overflow_distance_count": int(state.overflow_distance_count),
        "negative_distance_count": int(state.negative_distance_count),
        "nonfinite_distance_count": int(state.nonfinite_distance_count),
        "zero_distance_count": int(state.zero_distance_count),
        "mean_distance_angstrom": float(mean),
        "std_distance_angstrom": float(math.sqrt(variance)),
        "observed_min_distance_angstrom": state.observed_min_distance_angstrom,
        "observed_max_distance_angstrom": float(state.observed_max_distance_angstrom),
        "train_manifest_path": state.train_manifest_path,
        "train_manifest_sha256": state.train_manifest_sha256,
        "total_manifest_rows": int(state.total_manifest_rows),
        "processed_samples": int(state.processed_samples),
        "rejected_sample_count": int(state.rejected_sample_count),
        "config_hash": state.config_hash,
        "completed": True,
        "updated_utc": _utc_now(),
    }


def compute_scale_statistics(
    train_manifest: str | Path,
    *,
    percentile: float = 95.0,
    histogram_bin_width_angstrom: float = 0.05,
    histogram_max_distance_angstrom: float = 2000.0,
    overflow_fraction_tolerance: float = 0.0,
    output: str | Path | None = None,
    workers: int = 1,
    checkpoint_every: int | None = None,
    resume: bool = False,
    restart: bool = False,
    chunk_size: int = 512,
    interrupt_after_rows: int | None = None,
) -> dict[str, Any]:
    """Compute train-only normalization with fixed histogram memory."""
    if workers != 1:
        raise ValueError("statistics are currently computed serially; set workers=1")
    config = HistogramConfig(
        percentile, histogram_bin_width_angstrom, histogram_max_distance_angstrom, overflow_fraction_tolerance
    )
    frame = load_manifest(train_manifest)
    if frame.empty:
        raise ValueError("Manifest contains no samples")
    frame = frame.reset_index(drop=True)
    metadata = _manifest_metadata(train_manifest, len(frame), config)
    if output is not None and resume and not restart:
        final = _load_final_output(Path(output), metadata, config)
        if final is not None:
            return final
    state = _empty_state(config)
    _attach_metadata(state, metadata)
    if (
        output is not None
        and resume
        and not restart
        and _state_paths(output)[0].exists()
        and _state_paths(output)[1].exists()
    ):
        state = _load_state(output, config)
        _verify_state_compatible(state, metadata, config)
        if state.completed and state.next_row_index == state.total_manifest_rows:
            result = _finalize(state)
            if state.rejected_samples:
                raise RuntimeError(f"Rejected {len(state.rejected_samples)} sample(s) while computing statistics")
            return result
    try:
        step = max(1, int(chunk_size))
        while state.next_row_index < len(frame):
            start = state.next_row_index
            end = min(start + step, len(frame))
            chunk_state = _chunk_statistics(frame.iloc[start:end], config)
            _merge(state, chunk_state)
            state.next_row_index = end
            if interrupt_after_rows is not None and state.next_row_index >= int(interrupt_after_rows):
                if output is not None:
                    state.completed = False
                    _write_state(output, state, manifest_path=train_manifest)
                raise KeyboardInterrupt
            if (
                output is not None
                and checkpoint_every is not None
                and state.processed_samples % int(checkpoint_every) == 0
            ):
                state.completed = False
                _write_state(output, state, manifest_path=train_manifest)
    except KeyboardInterrupt:
        if output is not None:
            state.completed = False
            _write_state(output, state, manifest_path=train_manifest)
        raise
    if state.rejected_samples:
        if output is not None:
            state.completed = state.next_row_index == state.total_manifest_rows
            _write_state(output, state, manifest_path=train_manifest)
        raise RuntimeError(f"Rejected {len(state.rejected_samples)} sample(s) while computing statistics")
    state.completed = state.next_row_index == state.total_manifest_rows
    result = _finalize(state)
    if output is not None:
        _write_state(output, state, manifest_path=train_manifest)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        tmp.replace(out)
    return result


def write_normalization(
    train_manifest: str | Path,
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute and write the normalization JSON."""
    return compute_scale_statistics(train_manifest, output=output, **kwargs)
