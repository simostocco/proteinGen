"""Bounded-memory training normalization statistics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.data.preprocess import load_manifest


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
    sum_distance: float = 0.0
    sum_sq_distance: float = 0.0
    observed_max_distance_angstrom: float = 0.0
    processed_samples: int = 0
    rejected_samples: list[dict[str, Any]] | None = None


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
    return matrix[upper].astype(np.float64), None


def _chunk_statistics(chunk: pd.DataFrame, config: HistogramConfig) -> HistogramState:
    state = _empty_state(config)
    assert state.rejected_samples is not None
    for _, row in chunk.iterrows():
        values, rejection = _valid_values(row)
        state.processed_samples += 1
        if rejection is not None:
            state.rejected_samples.append(rejection)
            continue
        if values is None or values.size == 0:
            continue
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
        state.observed_max_distance_angstrom = max(state.observed_max_distance_angstrom, float(values.max(initial=0.0)))
    return state


def _merge(left: HistogramState, right: HistogramState) -> HistogramState:
    left.histogram += right.histogram
    left.valid_distance_count += right.valid_distance_count
    left.overflow_distance_count += right.overflow_distance_count
    left.sum_distance += right.sum_distance
    left.sum_sq_distance += right.sum_sq_distance
    left.observed_max_distance_angstrom = max(left.observed_max_distance_angstrom, right.observed_max_distance_angstrom)
    left.processed_samples += right.processed_samples
    assert left.rejected_samples is not None and right.rejected_samples is not None
    left.rejected_samples.extend(right.rejected_samples)
    return left


def _state_paths(output: str | Path) -> tuple[Path, Path]:
    out = Path(output)
    return out.with_suffix(".state.npz"), out.with_suffix(".state.json")


def _write_state(output: str | Path, state: HistogramState, *, manifest_path: str | Path) -> None:
    npz, js = _state_paths(output)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, histogram=state.histogram)
    payload = {
        "config": asdict(state.config),
        "manifest_path": str(manifest_path),
        "valid_distance_count": state.valid_distance_count,
        "overflow_distance_count": state.overflow_distance_count,
        "sum_distance": state.sum_distance,
        "sum_sq_distance": state.sum_sq_distance,
        "observed_max_distance_angstrom": state.observed_max_distance_angstrom,
        "processed_samples": state.processed_samples,
        "rejected_samples": state.rejected_samples or [],
    }
    js.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_state(output: str | Path, config: HistogramConfig) -> HistogramState:
    npz, js = _state_paths(output)
    payload = json.loads(js.read_text())
    if payload["config"] != asdict(config):
        raise ValueError("statistics state configuration differs")
    state = _empty_state(config)
    state.histogram = np.load(npz)["histogram"].astype(np.int64)
    state.valid_distance_count = int(payload["valid_distance_count"])
    state.overflow_distance_count = int(payload["overflow_distance_count"])
    state.sum_distance = float(payload["sum_distance"])
    state.sum_sq_distance = float(payload["sum_sq_distance"])
    state.observed_max_distance_angstrom = float(payload["observed_max_distance_angstrom"])
    state.processed_samples = int(payload["processed_samples"])
    state.rejected_samples = list(payload.get("rejected_samples", []))
    return state


def _finalize(state: HistogramState) -> dict[str, Any]:
    if state.valid_distance_count <= 0:
        raise ValueError("Manifest contains no samples with valid off-diagonal distances")
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
        "mean_distance_angstrom": float(mean),
        "std_distance_angstrom": float(math.sqrt(variance)),
        "observed_max_distance_angstrom": float(state.observed_max_distance_angstrom),
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
) -> dict[str, Any]:
    """Compute train-only normalization with fixed histogram memory."""
    del workers
    config = HistogramConfig(
        percentile, histogram_bin_width_angstrom, histogram_max_distance_angstrom, overflow_fraction_tolerance
    )
    if (
        output is not None
        and resume
        and not restart
        and _state_paths(output)[0].exists()
        and _state_paths(output)[1].exists()
    ):
        state = _load_state(output, config)
        result = _finalize(state)
        if state.rejected_samples:
            raise RuntimeError(f"Rejected {len(state.rejected_samples)} sample(s) while computing statistics")
        return result
    frame = load_manifest(train_manifest)
    if frame.empty:
        raise ValueError("Manifest contains no samples")
    state = _empty_state(config)
    try:
        for start in range(0, len(frame), max(1, int(chunk_size))):
            chunk_state = _chunk_statistics(frame.iloc[start : start + max(1, int(chunk_size))], config)
            _merge(state, chunk_state)
            if (
                output is not None
                and checkpoint_every is not None
                and state.processed_samples % int(checkpoint_every) == 0
            ):
                _write_state(output, state, manifest_path=train_manifest)
    except KeyboardInterrupt:
        if output is not None:
            _write_state(output, state, manifest_path=train_manifest)
        raise
    if output is not None:
        _write_state(output, state, manifest_path=train_manifest)
    if state.rejected_samples:
        raise RuntimeError(f"Rejected {len(state.rejected_samples)} sample(s) while computing statistics")
    result = _finalize(state)
    if output is not None:
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
