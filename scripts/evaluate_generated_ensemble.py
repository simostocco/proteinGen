#!/usr/bin/env python3
"""Resumable ensemble evaluation for generated distance maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteingen_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation  # noqa: E402
from protein_distance_diffusion.data.preprocess import load_manifest  # noqa: E402
from protein_distance_diffusion.diffusion.gaussian import (  # noqa: E402
    GaussianDiffusion,
    prediction_parameterization_from_config,
)
from protein_distance_diffusion.diffusion.sampling import sample_ddpm  # noqa: E402
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule  # noqa: E402
from protein_distance_diffusion.evaluation.contact_maps import binary_contact_map, offdiagonal_pair_values  # noqa: E402
from protein_distance_diffusion.evaluation.metrics import generated_matrix_report  # noqa: E402
from protein_distance_diffusion.evaluation.plots import save_heatmap  # noqa: E402
from protein_distance_diffusion.models.unet import DistanceUNet  # noqa: E402
from protein_distance_diffusion.training.checkpointing import load_checkpoint  # noqa: E402
from protein_distance_diffusion.utils.hashing import sha256_file  # noqa: E402

LONG_RANGE_MIN_SEPARATION = 12
ADJACENT_CA_MEAN_ANGSTROM = 3.8
ADJACENT_CA_STD_ANGSTROM = 0.2
DESCRIPTOR_COLUMNS = (
    "distance_mean",
    "distance_std",
    "adjacent_residue_distance_mean",
    "contact_fraction_6A",
    "contact_fraction_8A",
    "contact_fraction_10A",
    "long_range_contact_fraction",
    "radius_of_gyration",
    "triangle_violation_fraction",
    "negative_eigenvalue_mass_fraction",
    "rank3_residual_energy_fraction",
)
SCALAR_DISTRIBUTION_COLUMNS = (
    "adjacent_residue_distance_mean",
    "contact_fraction_6A",
    "contact_fraction_8A",
    "contact_fraction_10A",
    "long_range_contact_fraction",
    "distance_mean",
    "distance_std",
    "radius_of_gyration",
    "triangle_violation_fraction",
    "triangle_violation_mean",
    "negative_eigenvalue_mass_fraction",
    "rank3_residual_energy_fraction",
)
METRIC_DEFINITIONS = {
    "validity": "Per-matrix numerical, triangle, EDM, and protein-like geometry diagnostics.",
    "distribution_matching": "Descriptor comparisons between generated samples and validation/test controls.",
    "diversity": "Within-length pairwise distance-map and contact-map comparisons.",
    "novelty": "Approximate two-stage nearest-neighbour search against training descriptors.",
    "long_range_contact_fraction": f"Fraction of 8A contacts with |i-j|>{LONG_RANGE_MIN_SEPARATION}.",
    "radius_of_gyration": "Inferred from centered pairwise squared distances when EDM moments are finite.",
}


@dataclass(frozen=True)
class EvaluationConfig:
    """Canonical runtime options for generated ensemble evaluation."""

    checkpoint: Path
    weights: str
    config_path: Path | None
    normalization_file: Path
    train_manifest: Path
    reference_manifest: Path
    output_dir: Path
    lengths: tuple[int, ...]
    samples_per_length: int
    samples_by_length: dict[int, int] | None
    master_seed: int
    seeds: tuple[int, ...] | None
    contact_threshold: float
    num_triangles: int
    novelty_candidate_count: int
    workers: int
    resume: bool
    restart: bool
    plots: bool
    control_count: int
    real_length_tolerance: int
    diversity_pair_limit: int
    bootstrap_iterations: int


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(dst)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object atomically."""
    atomic_write_text(path, json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a CSV file atomically."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    tmp.replace(dst)


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a parquet file atomically."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(dst)


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    """Write compressed NumPy arrays atomically."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def to_jsonable(value: Any) -> Any:
    """Convert common scientific scalar/container types to JSON-safe objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def git_commit() -> str | None:
    """Return the current Git commit if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def path_sha256(path: str | Path | None) -> str | None:
    """Return SHA-256 for a path, or None for absent values."""
    if path is None:
        return None
    src = Path(path)
    return sha256_file(src) if src.exists() else None


def seed_schedule(
    lengths: tuple[int, ...],
    samples_per_length: int | dict[int, int],
    master_seed: int,
) -> dict[int, list[int]]:
    """Create deterministic per-length sample seeds."""
    schedule: dict[int, list[int]] = {}
    for length in lengths:
        count = samples_per_length[int(length)] if isinstance(samples_per_length, dict) else samples_per_length
        rng = np.random.default_rng(master_seed + int(length) * 1_000_003)
        schedule[int(length)] = [int(x) for x in rng.integers(0, 2**31 - 1, size=int(count))]
    return schedule


def explicit_seed_schedule(lengths: tuple[int, ...], seeds: tuple[int, ...]) -> dict[int, list[int]]:
    """Use the same explicit seed bank for every evaluated length."""
    return {int(length): [int(seed) for seed in seeds] for length in lengths}


def build_seed_schedule(cfg: EvaluationConfig) -> dict[int, list[int]]:
    """Build the seed schedule implied by explicit seeds or sample counts."""
    if cfg.seeds is not None:
        return explicit_seed_schedule(cfg.lengths, cfg.seeds)
    return seed_schedule(cfg.lengths, cfg.samples_by_length or cfg.samples_per_length, cfg.master_seed)


def protocol_core(cfg: EvaluationConfig, checkpoint: dict[str, Any], schedule: dict[int, list[int]]) -> dict[str, Any]:
    """Build the protocol content used to detect incompatible resumes."""
    runtime = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
    }
    return {
        "checkpoint_path": str(cfg.checkpoint),
        "checkpoint_sha256": path_sha256(cfg.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_next_epoch": checkpoint.get("next_epoch"),
        "checkpoint_optimizer_step": checkpoint.get("optimizer_step", checkpoint.get("global_step")),
        "training_config_path": str(cfg.config_path) if cfg.config_path is not None else None,
        "training_config_sha256": path_sha256(cfg.config_path),
        "normalization_path": str(cfg.normalization_file),
        "normalization_sha256": path_sha256(cfg.normalization_file),
        "train_manifest_path": str(cfg.train_manifest),
        "train_manifest_sha256": path_sha256(cfg.train_manifest),
        "reference_manifest_path": str(cfg.reference_manifest),
        "reference_manifest_sha256": path_sha256(cfg.reference_manifest),
        "git_commit": git_commit(),
        "runtime": runtime,
        "selected_weights": cfg.weights,
        "lengths": list(cfg.lengths),
        "seed_schedule": schedule,
        "samples_per_length": None if cfg.samples_by_length is not None else cfg.samples_per_length,
        "sample_counts_by_length": {str(length): len(seeds) for length, seeds in schedule.items()},
        "contact_threshold_angstrom": cfg.contact_threshold,
        "num_sampled_triangles": cfg.num_triangles,
        "novelty_candidate_count": cfg.novelty_candidate_count,
        "worker_count": cfg.workers,
        "control_count": cfg.control_count,
        "real_length_tolerance": cfg.real_length_tolerance,
        "diversity_pair_limit": cfg.diversity_pair_limit,
        "bootstrap_iterations": cfg.bootstrap_iterations,
        "metric_definitions": METRIC_DEFINITIONS,
    }


def protocol_hash(core: dict[str, Any]) -> str:
    """Hash protocol settings that must match on resume."""
    payload = json.dumps(to_jsonable(core), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_protocol(
    path: Path,
    core: dict[str, Any],
    *,
    status: str,
    started_utc: str,
    completed_utc: str | None,
) -> None:
    """Write protocol JSON with status timestamps."""
    payload = dict(core)
    payload["protocol_sha256"] = protocol_hash(core)
    payload["status"] = status
    payload["started_utc"] = started_utc
    payload["completed_utc"] = completed_utc
    atomic_write_json(path, payload)


def validate_resume_protocol(path: Path, core: dict[str, Any], *, resume: bool, restart: bool) -> str:
    """Validate existing protocol compatibility and return the start timestamp."""
    if not path.exists():
        return utc_now()
    existing = json.loads(path.read_text())
    if existing.get("protocol_sha256") != protocol_hash(core):
        if restart:
            return utc_now()
        raise ValueError("Existing protocol is incompatible with requested evaluation. Use --restart with care.")
    if not resume and not restart:
        raise FileExistsError("Output directory already contains a protocol. Use --resume or --restart.")
    return str(existing.get("started_utc") or utc_now())


def init_state(path: Path, *, protocol_sha256: str, restart: bool) -> sqlite3.Connection:
    """Initialize the resumable SQLite state database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS generated_samples (
            sample_key TEXT PRIMARY KEY,
            length INTEGER NOT NULL,
            sample_index INTEGER NOT NULL,
            seed INTEGER NOT NULL,
            path TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT NOT NULL DEFAULT '',
            updated_utc TEXT NOT NULL
        );
        """
    )
    if restart:
        conn.executescript("DELETE FROM generated_samples; DELETE FROM metadata;")
    saved = conn.execute("SELECT value FROM metadata WHERE key='protocol_sha256'").fetchone()
    if saved is not None and saved[0] != protocol_sha256 and not restart:
        raise ValueError("SQLite state belongs to a different evaluation protocol.")
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('protocol_sha256', ?)",
        (protocol_sha256,),
    )
    conn.commit()
    return conn


def sample_key(length: int, index: int, seed: int) -> str:
    """Stable identifier for one generated sample."""
    return f"N{int(length):04d}_i{int(index):05d}_seed{int(seed)}"


def generated_sample_path(output_dir: Path, length: int, index: int, seed: int) -> Path:
    """Return the canonical NPZ path for a generated sample."""
    return output_dir / "generated" / f"N{int(length):04d}" / f"{sample_key(length, index, seed)}.npz"


def load_normalization_scale(path: Path, checkpoint: dict[str, Any]) -> float:
    """Load distance normalization scale, preferring the explicit file."""
    if path.exists():
        normalization = json.loads(path.read_text())
    else:
        normalization = checkpoint.get("normalization", {})
    return float(normalization.get("scale", 1.0))


def build_model(checkpoint: dict[str, Any], *, weights: str, device: torch.device) -> DistanceUNet:
    """Build and load a checkpointed DistanceUNet."""
    model_cfg = dict(checkpoint["config"]["model"])
    if "channel_multipliers" in model_cfg:
        model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg).to(device)
    if weights == "ema":
        if "ema" not in checkpoint:
            raise ValueError("EMA weights requested but checkpoint does not contain an 'ema' state")
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def validate_generated_npz(path: Path, *, length: int, seed: int, protocol_sha256: str) -> bool:
    """Return True when an existing generated NPZ is complete and compatible."""
    if not path.exists():
        return False
    try:
        data = np.load(path, allow_pickle=False)
        matrix = np.asarray(data["physical_distance_matrix_angstrom"], dtype=np.float32)
        return (
            matrix.shape == (length, length)
            and int(data["requested_length"]) == int(length)
            and int(data["seed"]) == int(seed)
            and str(data["protocol_sha256"]) == protocol_sha256
            and bool(np.isfinite(matrix).all())
        )
    except Exception:
        return False


def generate_one_sample(
    *,
    model: DistanceUNet,
    diffusion: GaussianDiffusion,
    checkpoint: dict[str, Any],
    length: int,
    index: int,
    seed: int,
    scale: float,
    output_dir: Path,
    weights: str,
    protocol_sha256: str,
    device: torch.device,
) -> Path:
    """Generate and atomically save one physical distance matrix."""
    path = generated_sample_path(output_dir, length, index, seed)
    if validate_generated_npz(path, length=length, seed=seed, protocol_sha256=protocol_sha256):
        return path
    factor = int(getattr(model, "downsample_factor", 1))
    side = ((int(length) + factor - 1) // factor) * factor
    lengths = torch.tensor([int(length)], dtype=torch.long, device=device)
    pair_mask = make_pair_mask(lengths, side).to(device)
    sep = make_sequence_separation(lengths, side).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        sampled = sample_ddpm(
            model,
            diffusion,
            lengths=lengths,
            pair_mask=pair_mask,
            sequence_separation=sep,
            device=device,
            generator=generator,
            prediction_type=prediction_parameterization_from_config(checkpoint["config"]),
        )
    normalized = sampled[0, 0, :length, :length].detach().cpu().numpy().astype(np.float32)
    physical = normalized * np.float32(scale)
    projected = 0.5 * (physical + physical.T)
    np.fill_diagonal(projected, 0.0)
    atomic_write_npz(
        path,
        sample_id=np.asarray(sample_key(length, index, seed)),
        requested_length=np.asarray(int(length)),
        sample_index=np.asarray(int(index)),
        seed=np.asarray(int(seed)),
        raw_normalized_matrix=normalized,
        raw_physical_distance_matrix_angstrom=physical.astype(np.float32),
        physical_distance_matrix_angstrom=projected.astype(np.float32),
        checkpoint=np.asarray(str(checkpoint.get("path", ""))),
        weights=np.asarray(weights),
        protocol_sha256=np.asarray(protocol_sha256),
    )
    return path


def matrix_edm_diagnostics(matrix: np.ndarray) -> dict[str, float | int]:
    """Compute detailed EDM diagnostics from a distance matrix."""
    d = np.asarray(matrix, dtype=np.float64)
    n = d.shape[0]
    if n == 0 or not np.isfinite(d).all():
        return {
            "negative_eigenvalue_mass_fraction": float("nan"),
            "materially_negative_eigenvalues": 0,
            "energy_outside_top3_positive_fraction": float("nan"),
            "rank3_residual_energy_fraction": float("nan"),
            "classical_mds_stress": float("nan"),
        }
    j = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * j @ (d * d) @ j
    eigvals, eigvecs = np.linalg.eigh(gram)
    total_abs = float(np.sum(np.abs(eigvals))) + 1e-12
    negative = eigvals[eigvals < 0.0]
    positive_order = np.argsort(eigvals)[::-1]
    positive_order = [idx for idx in positive_order if eigvals[idx] > 0.0]
    top3 = positive_order[:3]
    outside = [idx for idx in positive_order[3:]]
    coords = np.zeros((n, 3), dtype=np.float64)
    for axis, idx in enumerate(top3):
        coords[:, axis] = eigvecs[:, idx] * math.sqrt(max(float(eigvals[idx]), 0.0))
    reconstructed = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    denom = float(np.linalg.norm(d)) + 1e-12
    return {
        "negative_eigenvalue_mass_fraction": float(np.sum(np.abs(negative)) / total_abs),
        "materially_negative_eigenvalues": int(np.sum(eigvals < -1e-6 * max(float(np.max(np.abs(eigvals))), 1.0))),
        "energy_outside_top3_positive_fraction": float(np.sum(eigvals[outside]) / total_abs) if outside else 0.0,
        "rank3_residual_energy_fraction": float(np.sum(eigvals[outside]) / total_abs) if outside else 0.0,
        "classical_mds_stress": float(np.linalg.norm(reconstructed - d) / denom),
    }


def radius_of_gyration_from_distances(matrix: np.ndarray) -> float:
    """Infer radius of gyration from pairwise squared distances."""
    d = np.asarray(matrix, dtype=np.float64)
    if d.size == 0 or not np.isfinite(d).all():
        return float("nan")
    mean_sq = float(np.mean(d * d))
    return math.sqrt(max(mean_sq / 2.0, 0.0))


def sampled_triangle_metrics(matrix: np.ndarray, *, num_triangles: int, seed: int) -> dict[str, float]:
    """Sample triangle inequality violations deterministically."""
    d = np.asarray(matrix, dtype=np.float64)
    n = d.shape[0]
    if n < 3 or num_triangles <= 0:
        return {
            "triangle_violation_fraction": 0.0,
            "triangle_violation_mean": 0.0,
            "triangle_violation_max": 0.0,
            "sampled_triangle_count": 0,
        }
    rng = np.random.default_rng(int(seed))
    triples = rng.integers(0, n, size=(int(num_triangles), 3))
    violations = np.maximum(
        0.0,
        d[triples[:, 0], triples[:, 1]] - d[triples[:, 0], triples[:, 2]] - d[triples[:, 2], triples[:, 1]],
    )
    return {
        "triangle_violation_fraction": float(np.mean(violations > 0.0)),
        "triangle_violation_mean": float(np.mean(violations)),
        "triangle_violation_max": float(np.max(violations)),
        "sampled_triangle_count": int(num_triangles),
    }


def descriptor_from_matrix(
    matrix: np.ndarray,
    *,
    sample_id: str,
    length: int,
    seed: int,
    contact_threshold: float,
    num_triangles: int,
    source: str,
    path: str | None = None,
    pdb_id: str | None = None,
) -> dict[str, Any]:
    """Compute one-row matrix descriptors used by validity, distribution, and novelty."""
    d = np.asarray(matrix, dtype=np.float32)
    offdiag = offdiagonal_pair_values(d)
    report = generated_matrix_report(d, scale=1.0)
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "pdb_id": pdb_id,
        "source": source,
        "path": path,
        "length": int(length),
        "seed": int(seed),
        "finite": bool(np.isfinite(d).all()),
        "nonfinite_fraction": float(np.mean(~np.isfinite(d))),
        "distance_mean": float(np.nanmean(offdiag)) if offdiag.size else float("nan"),
        "distance_std": float(np.nanstd(offdiag)) if offdiag.size else float("nan"),
        "distance_quantile_05": float(np.nanquantile(offdiag, 0.05)) if offdiag.size else float("nan"),
        "distance_quantile_50": float(np.nanquantile(offdiag, 0.50)) if offdiag.size else float("nan"),
        "distance_quantile_95": float(np.nanquantile(offdiag, 0.95)) if offdiag.size else float("nan"),
        "minimum_physical_distance": float(np.nanmin(offdiag)) if offdiag.size else float("nan"),
        "maximum_physical_distance": float(np.nanmax(offdiag)) if offdiag.size else float("nan"),
    }
    row.update(report)
    row.update(sampled_triangle_metrics(d, num_triangles=num_triangles, seed=seed))
    row.update(matrix_edm_diagnostics(d))
    adjacent = np.diag(d, k=1).astype(np.float64)
    if adjacent.size:
        deviation = adjacent - ADJACENT_CA_MEAN_ANGSTROM
        row["adjacent_residue_distance_rmse"] = float(np.sqrt(np.mean(deviation * deviation)))
        row["adjacent_residue_distance_empirical_z_mean"] = float(np.mean(deviation / ADJACENT_CA_STD_ANGSTROM))
    else:
        row["adjacent_residue_distance_rmse"] = float("nan")
        row["adjacent_residue_distance_empirical_z_mean"] = float("nan")
    sep2 = np.diag(d, k=2).astype(np.float64)
    row["sep_2_mean"] = float(np.nanmean(sep2)) if sep2.size else float("nan")
    row["sep_2_std"] = float(np.nanstd(sep2)) if sep2.size else float("nan")
    idx = np.arange(d.shape[0])
    near = np.abs(idx[:, None] - idx[None, :]) <= 2
    off_near = np.triu(~near, k=1)
    row["clash_fraction"] = float(np.mean(d[off_near] < 2.0)) if np.any(off_near) else 0.0
    for threshold in (6.0, 8.0, 10.0):
        contacts = binary_contact_map(d, threshold_angstrom=threshold, exclude_near_diagonal=0)
        row[f"contact_fraction_{int(threshold)}A"] = float(np.mean(contacts[np.triu(np.ones_like(contacts), k=1)]))
    long_contacts = binary_contact_map(
        d,
        threshold_angstrom=contact_threshold,
        exclude_near_diagonal=LONG_RANGE_MIN_SEPARATION,
    )
    long_mask = np.triu(np.ones_like(long_contacts, dtype=bool), k=LONG_RANGE_MIN_SEPARATION + 1)
    row["long_range_contact_fraction"] = float(np.mean(long_contacts[long_mask])) if np.any(long_mask) else 0.0
    row["radius_of_gyration"] = radius_of_gyration_from_distances(d)
    row["numerically_valid"] = bool(
        row["finite"]
        and row["negative_distance_fraction"] == 0.0
        and row["max_abs_diagonal"] < 1e-5
        and row["symmetry_error"] < 1e-5
    )
    row["edm_compatible"] = bool(
        row["numerically_valid"]
        and row["negative_eigenvalue_mass_fraction"] < 0.05
        and row["rank3_residual_energy_fraction"] < 0.20
    )
    row["chain_like"] = bool(row["numerically_valid"] and row["adjacent_residue_distance_rmse"] < 1.0)
    row["protein_like"] = bool(row["edm_compatible"] and row["chain_like"] and row["clash_fraction"] < 0.05)
    return row


def load_npz_matrix(path: str | Path, *, generated: bool = False) -> np.ndarray:
    """Load a generated or processed real-control distance matrix."""
    data = np.load(path, allow_pickle=False)
    if generated and "physical_distance_matrix_angstrom" in data:
        return np.asarray(data["physical_distance_matrix_angstrom"], dtype=np.float32)
    if "distance_matrix" not in data and "physical_distance_matrix_angstrom" in data:
        return np.asarray(data["physical_distance_matrix_angstrom"], dtype=np.float32)
    return np.asarray(data["distance_matrix"], dtype=np.float32)


def select_real_controls(
    manifest: pd.DataFrame,
    *,
    length: int,
    count: int,
    tolerance: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select deterministic exact-length controls, then tolerance controls if needed."""
    exact = manifest[manifest["length"].astype(int) == int(length)].copy()
    exact_count = min(int(count), len(exact))
    rng = np.random.default_rng(seed + int(length) * 97)
    chosen_indices: list[int] = []
    if exact_count:
        chosen_indices.extend(sorted(rng.choice(exact.index.to_numpy(), size=exact_count, replace=False).tolist()))
    remaining = int(count) - len(chosen_indices)
    tolerance_count = 0
    if remaining > 0 and tolerance > 0:
        lengths = manifest["length"].astype(int)
        candidates = manifest[(lengths != int(length)) & ((lengths - int(length)).abs() <= int(tolerance))].copy()
        if len(candidates):
            take = min(remaining, len(candidates))
            picked = sorted(rng.choice(candidates.index.to_numpy(), size=take, replace=False).tolist())
            chosen_indices.extend(picked)
            tolerance_count = len(picked)
    selected = manifest.loc[chosen_indices].copy().reset_index(drop=True)
    return selected, {
        "requested_count": int(count),
        "selected_count": int(len(selected)),
        "exact_length_count": int(exact_count),
        "tolerance_matched_count": int(tolerance_count),
    }


def descriptor_vector(row: dict[str, Any] | pd.Series) -> np.ndarray:
    """Return a numeric descriptor vector with NaN converted to zero."""
    values = [float(row.get(column, np.nan)) for column in DESCRIPTOR_COLUMNS]
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def descriptor_distance(left: dict[str, Any] | pd.Series, right: dict[str, Any] | pd.Series) -> float:
    """Euclidean descriptor distance."""
    return float(np.linalg.norm(descriptor_vector(left) - descriptor_vector(right)))


def map_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Normalized distance-map RMSE for same-length matrices."""
    if left.shape != right.shape:
        raise ValueError("Distance-map RMSE requires matrices with identical shapes")
    denom = float(np.nanmean(np.abs(right))) + 1e-8
    return float(np.sqrt(np.nanmean(np.square(left.astype(np.float64) - right.astype(np.float64)))) / denom)


def contact_distances(left: np.ndarray, right: np.ndarray, *, threshold: float) -> tuple[float, float]:
    """Return contact Hamming distance and Jaccard distance for same-length matrices."""
    if left.shape != right.shape:
        raise ValueError("Contact-map comparisons require matrices with identical shapes")
    a = binary_contact_map(left, threshold_angstrom=threshold, exclude_near_diagonal=2)
    b = binary_contact_map(right, threshold_angstrom=threshold, exclude_near_diagonal=2)
    mask = np.triu(np.ones(a.shape, dtype=bool), k=1)
    av = a[mask]
    bv = b[mask]
    union = np.logical_or(av, bv)
    hamming = float(np.mean(av != bv)) if av.size else 0.0
    jaccard_similarity = float(np.sum(np.logical_and(av, bv)) / max(np.sum(union), 1))
    return hamming, 1.0 - jaccard_similarity


def deterministic_pair_indices(count: int, *, limit: int, seed: int) -> list[tuple[int, int]]:
    """Return all or a deterministic subset of pair indices."""
    pairs = [(i, j) for i in range(count) for j in range(i + 1, count)]
    if limit <= 0 or len(pairs) <= limit:
        return pairs
    rng = random.Random(seed)
    return sorted(rng.sample(pairs, limit))


def diversity_pairs_for_length(
    samples: pd.DataFrame,
    *,
    contact_threshold: float,
    pair_limit: int,
    seed: int,
) -> pd.DataFrame:
    """Compute deterministic pairwise diversity rows for same-length generated samples."""
    rows = samples.sort_values("sample_id").reset_index(drop=True)
    pairs = deterministic_pair_indices(len(rows), limit=pair_limit, seed=seed)
    out = []
    matrices: dict[int, np.ndarray] = {}
    for i, j in pairs:
        left = matrices.setdefault(i, load_npz_matrix(rows.iloc[i]["path"], generated=True))
        right = matrices.setdefault(j, load_npz_matrix(rows.iloc[j]["path"], generated=True))
        hamming, jaccard = contact_distances(left, right, threshold=contact_threshold)
        desc_dist = descriptor_distance(rows.iloc[i], rows.iloc[j])
        out.append(
            {
                "length": int(rows.iloc[i]["length"]),
                "sample_id_a": rows.iloc[i]["sample_id"],
                "sample_id_b": rows.iloc[j]["sample_id"],
                "distance_map_rmse": map_rmse(left, right),
                "contact_hamming_distance": hamming,
                "contact_jaccard_distance": jaccard,
                "descriptor_distance": desc_dist,
            }
        )
    return pd.DataFrame(out)


def cluster_generated_samples(
    pairs: pd.DataFrame,
    sample_ids: list[str],
    *,
    rmse_threshold: float = 0.02,
) -> pd.DataFrame:
    """Assign simple duplicate/near-duplicate clusters from pairwise RMSE threshold."""
    parent = {sample_id: sample_id for sample_id in sample_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if not pairs.empty:
        for row in pairs.itertuples(index=False):
            if float(row.distance_map_rmse) <= rmse_threshold:
                union(str(row.sample_id_a), str(row.sample_id_b))
    roots = {sample_id: find(sample_id) for sample_id in sample_ids}
    ordered_roots = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "diversity_cluster_id": ordered_roots[root],
                "near_duplicate_cluster_root": root,
            }
            for sample_id, root in sorted(roots.items())
        ]
    )


def scalar_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    """Small dependency-free 1D Wasserstein distance."""
    a = np.sort(np.asarray(left, dtype=np.float64))
    b = np.sort(np.asarray(right, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, max(a.size, b.size))
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    """Small dependency-free two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(left, dtype=np.float64))
    b = np.sort(np.asarray(right, dtype=np.float64))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / a.size
    cdf_b = np.searchsorted(b, values, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def bootstrap_ci(values: np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    """Bootstrap confidence interval for a mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or iterations <= 0:
        mean = float(np.mean(arr)) if arr.size else float("nan")
        return mean, mean
    rng = np.random.default_rng(seed)
    means = [float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)])) for _ in range(iterations)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def distribution_matching(generated: pd.DataFrame, real: pd.DataFrame, *, bootstrap_iterations: int) -> pd.DataFrame:
    """Compare generated and real descriptor distributions by requested length."""
    rows = []
    for length in sorted(generated["length"].astype(int).unique()):
        gen_len = generated[generated["length"].astype(int) == int(length)]
        real_len = real[real["requested_length"].astype(int) == int(length)] if "requested_length" in real else real
        for column in SCALAR_DISTRIBUTION_COLUMNS:
            if column not in gen_len or column not in real_len:
                continue
            gen_values = gen_len[column].to_numpy(dtype=float)
            real_values = real_len[column].to_numpy(dtype=float)
            gen_ci = bootstrap_ci(gen_values, iterations=bootstrap_iterations, seed=length + len(column))
            real_ci = bootstrap_ci(real_values, iterations=bootstrap_iterations, seed=length + 17 + len(column))
            rows.append(
                {
                    "length": int(length),
                    "descriptor": column,
                    "generated_count": int(np.isfinite(gen_values).sum()),
                    "real_count": int(np.isfinite(real_values).sum()),
                    "generated_mean": float(np.nanmean(gen_values)) if np.isfinite(gen_values).any() else float("nan"),
                    "real_mean": float(np.nanmean(real_values)) if np.isfinite(real_values).any() else float("nan"),
                    "generated_std": float(np.nanstd(gen_values)) if np.isfinite(gen_values).any() else float("nan"),
                    "real_std": float(np.nanstd(real_values)) if np.isfinite(real_values).any() else float("nan"),
                    "generated_q05": float(np.nanquantile(gen_values, 0.05))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "generated_q50": float(np.nanquantile(gen_values, 0.50))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "generated_q95": float(np.nanquantile(gen_values, 0.95))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "real_q05": float(np.nanquantile(real_values, 0.05))
                    if np.isfinite(real_values).any()
                    else float("nan"),
                    "real_q50": float(np.nanquantile(real_values, 0.50))
                    if np.isfinite(real_values).any()
                    else float("nan"),
                    "real_q95": float(np.nanquantile(real_values, 0.95))
                    if np.isfinite(real_values).any()
                    else float("nan"),
                    "generated_mean_ci_low": gen_ci[0],
                    "generated_mean_ci_high": gen_ci[1],
                    "real_mean_ci_low": real_ci[0],
                    "real_mean_ci_high": real_ci[1],
                    "wasserstein_distance": scalar_wasserstein(gen_values, real_values),
                    "ks_statistic": ks_statistic(gen_values, real_values),
                    "warning": "insufficient_sample_size"
                    if np.isfinite(gen_values).sum() < 2 or np.isfinite(real_values).sum() < 2
                    else "",
                }
            )
    return pd.DataFrame(rows)


def descriptor_cache_for_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    contact_threshold: float,
    num_triangles: int,
    resume: bool,
) -> pd.DataFrame:
    """Build or load a descriptor cache for a real manifest one matrix at a time."""
    if resume and output_path.exists():
        return pd.read_parquet(output_path)
    manifest = load_manifest(manifest_path)
    rows = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"Descriptors {manifest_path.name}"):
        matrix = load_npz_matrix(row.path)
        rows.append(
            descriptor_from_matrix(
                matrix,
                sample_id=str(row.sample_id),
                pdb_id=str(getattr(row, "pdb_id", "")),
                length=int(getattr(row, "length", matrix.shape[0])),
                seed=0,
                contact_threshold=contact_threshold,
                num_triangles=num_triangles,
                source="real",
                path=str(row.path),
            )
        )
    frame = pd.DataFrame(rows)
    atomic_write_parquet(output_path, frame)
    return frame


def approximate_novelty(
    generated: pd.DataFrame,
    training_descriptors: pd.DataFrame,
    *,
    candidate_count: int,
    length_tolerance: int,
    contact_threshold: float,
) -> pd.DataFrame:
    """Approximate novelty by descriptor retrieval followed by refined map/contact comparison."""
    rows = []
    train_by_length = training_descriptors.copy()
    train_by_length["length"] = train_by_length["length"].astype(int)
    for gen in generated.sort_values("sample_id").itertuples(index=False):
        length = int(gen.length)
        exact = train_by_length[train_by_length["length"] == length]
        candidates = exact
        match_mode = "exact_length"
        if candidates.empty and length_tolerance > 0:
            candidates = train_by_length[(train_by_length["length"] - length).abs() <= int(length_tolerance)]
            match_mode = "tolerance"
        if candidates.empty:
            rows.append(
                {
                    "sample_id": gen.sample_id,
                    "length": length,
                    "match_mode": "none",
                    "nearest_training_sample_id": None,
                    "nearest_training_pdb_id": None,
                    "descriptor_distance": float("nan"),
                    "refined_distance_map_rmse": float("nan"),
                    "refined_contact_jaccard_distance": float("nan"),
                    "approximate": True,
                }
            )
            continue
        gen_row = gen._asdict()
        distances = candidates.apply(lambda row, gen_row=gen_row: descriptor_distance(gen_row, row), axis=1)
        nearest = (
            candidates.assign(_descriptor_distance=distances)
            .sort_values(["_descriptor_distance", "sample_id"])
            .head(max(1, int(candidate_count)))
        )
        gen_matrix = load_npz_matrix(gen.path, generated=True)
        best: dict[str, Any] | None = None
        for train in nearest.to_dict(orient="records"):
            train_matrix = load_npz_matrix(train["path"])
            refined = map_rmse(gen_matrix, train_matrix) if gen_matrix.shape == train_matrix.shape else float("nan")
            _, jaccard = (
                contact_distances(gen_matrix, train_matrix, threshold=contact_threshold)
                if gen_matrix.shape == train_matrix.shape
                else (float("nan"), float("nan"))
            )
            item = {
                "sample_id": gen.sample_id,
                "length": length,
                "match_mode": match_mode,
                "nearest_training_sample_id": train["sample_id"],
                "nearest_training_pdb_id": train.get("pdb_id"),
                "descriptor_distance": float(train["_descriptor_distance"]),
                "refined_distance_map_rmse": refined,
                "refined_contact_jaccard_distance": jaccard,
                "approximate": True,
            }
            current_refined = np.nan_to_num(refined, nan=float("inf"))
            best_refined = (
                float("inf") if best is None else np.nan_to_num(best["refined_distance_map_rmse"], nan=float("inf"))
            )
            if current_refined < best_refined:
                best = item
        rows.append(best)
    return pd.DataFrame(rows)


def novelty_calibration(
    reference: pd.DataFrame,
    training: pd.DataFrame,
    *,
    candidate_count: int,
    length_tolerance: int,
    contact_threshold: float,
) -> pd.DataFrame:
    """Run the same approximate novelty search for real controls against training."""
    generated_like = reference.copy()
    generated_like["source"] = "real_control"
    return approximate_novelty(
        generated_like,
        training,
        candidate_count=candidate_count,
        length_tolerance=length_tolerance,
        contact_threshold=contact_threshold,
    ).rename(columns={"sample_id": "real_control_sample_id"})


def write_basic_figures(
    output_dir: Path,
    *,
    generated: pd.DataFrame,
    real: pd.DataFrame,
    distribution: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    novelty: pd.DataFrame,
) -> None:
    """Write compact publication-oriented diagnostic figures."""
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if not generated.empty and "protein_like" in generated:
        fig, ax = plt.subplots(figsize=(6, 4))
        generated.groupby("length")["protein_like"].mean().plot(ax=ax, marker="o")
        counts = generated.groupby("length").size()
        for x, y in generated.groupby("length")["protein_like"].mean().items():
            ax.text(x, y, f"n={counts.loc[x]}", fontsize=8)
        ax.set_title("Generated protein-like diagnostic fraction")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel("Fraction")
        fig.tight_layout()
        fig.savefig(figures / "validity_metrics_by_length.png", dpi=160)
        plt.close(fig)
    if not distribution.empty:
        subset = distribution[distribution["descriptor"] == "distance_mean"]
        if not subset.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(subset["length"], subset["generated_mean"], marker="o", label="generated")
            ax.plot(subset["length"], subset["real_mean"], marker="s", label="real controls")
            for row in subset.itertuples(index=False):
                ax.text(row.length, row.generated_mean, f"g={row.generated_count} r={row.real_count}", fontsize=8)
            ax.set_title("Generated versus real mean distances")
            ax.set_xlabel("Requested length N")
            ax.set_ylabel("Mean distance (A)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / "generated_vs_real_distance_mean.png", dpi=160)
            plt.close(fig)
    if not diversity_summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(diversity_summary["length"], diversity_summary["distance_map_rmse_mean"], marker="o")
        ax.set_title("Generated diversity by length")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel("Mean normalized distance-map RMSE")
        fig.tight_layout()
        fig.savefig(figures / "diversity_by_length.png", dpi=160)
        plt.close(fig)
    if not novelty.empty and "descriptor_distance" in novelty:
        fig, ax = plt.subplots(figsize=(6, 4))
        novelty.groupby("length")["descriptor_distance"].mean().plot(ax=ax, marker="o")
        ax.set_title("Approximate generated novelty by length")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel("Mean nearest descriptor distance")
        fig.tight_layout()
        fig.savefig(figures / "novelty_calibration.png", dpi=160)
        plt.close(fig)
    for frame, prefix in ((generated, "generated"), (real, "real")):
        if frame.empty:
            continue
        for row in frame.head(4).itertuples(index=False):
            try:
                matrix = load_npz_matrix(row.path, generated=prefix == "generated")
                save_heatmap(matrix, figures / f"{prefix}_{row.sample_id}.png", title=f"{prefix} n={row.length}")
            except Exception:
                continue


def summarize_metrics(
    *,
    generated: pd.DataFrame,
    real: pd.DataFrame,
    diversity_summary: pd.DataFrame,
    novelty: pd.DataFrame,
    calibration: pd.DataFrame,
) -> dict[str, Any]:
    """Create a high-level non-composite summary."""
    return {
        "completed_utc": utc_now(),
        "generated_sample_count": int(len(generated)),
        "real_control_count": int(len(real)),
        "lengths": sorted(int(x) for x in generated["length"].unique()) if not generated.empty else [],
        "validity": {
            "numerically_valid_fraction": (
                float(generated["numerically_valid"].mean()) if len(generated) else float("nan")
            ),
            "edm_compatible_fraction": float(generated["edm_compatible"].mean()) if len(generated) else float("nan"),
            "chain_like_fraction": float(generated["chain_like"].mean()) if len(generated) else float("nan"),
            "protein_like_diagnostic_fraction": (
                float(generated["protein_like"].mean()) if len(generated) else float("nan")
            ),
        },
        "diversity": diversity_summary.to_dict(orient="records"),
        "novelty": {
            "approximate": True,
            "generated_nearest_descriptor_distance_mean": float(novelty["descriptor_distance"].mean())
            if len(novelty)
            else float("nan"),
            "real_calibration_nearest_descriptor_distance_mean": float(calibration["descriptor_distance"].mean())
            if len(calibration) and "descriptor_distance" in calibration
            else float("nan"),
        },
    }


def copy_training_length_distribution(output_dir: Path) -> None:
    """Copy the known training length distribution artifact into E000 if present."""
    src = Path("reports/recovered_full_b2_v/epoch8_baseline/training_length_distribution.csv")
    if src.exists():
        dst = output_dir / "metadata" / "training_length_distribution.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.read_bytes() != dst.read_bytes():
            tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(dst)


def evaluate_existing_outputs(cfg: EvaluationConfig, *, protocol_sha256_value: str) -> dict[str, pd.DataFrame]:
    """Compute metrics from generated samples and real/training descriptors."""
    metrics_dir = cfg.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    generated_rows = []
    for length, seeds in build_seed_schedule(cfg).items():
        for index, seed in enumerate(seeds):
            path = generated_sample_path(cfg.output_dir, length, index, seed)
            if validate_generated_npz(path, length=length, seed=seed, protocol_sha256=protocol_sha256_value):
                matrix = load_npz_matrix(path, generated=True)
                generated_rows.append(
                    descriptor_from_matrix(
                        matrix,
                        sample_id=sample_key(length, index, seed),
                        length=length,
                        seed=seed,
                        contact_threshold=cfg.contact_threshold,
                        num_triangles=cfg.num_triangles,
                        source="generated",
                        path=str(path),
                    )
                )
    generated = pd.DataFrame(generated_rows)
    atomic_write_parquet(metrics_dir / "validity_per_sample.parquet", generated)

    reference_manifest = load_manifest(cfg.reference_manifest)
    real_rows = []
    control_reports = []
    for length in cfg.lengths:
        controls, report = select_real_controls(
            reference_manifest,
            length=length,
            count=cfg.control_count,
            tolerance=cfg.real_length_tolerance,
            seed=cfg.master_seed,
        )
        report["requested_length"] = int(length)
        control_reports.append(report)
        for row in controls.itertuples(index=False):
            matrix = load_npz_matrix(row.path)
            real_rows.append(
                descriptor_from_matrix(
                    matrix,
                    sample_id=str(row.sample_id),
                    pdb_id=str(getattr(row, "pdb_id", "")),
                    length=int(getattr(row, "length", matrix.shape[0])),
                    seed=0,
                    contact_threshold=cfg.contact_threshold,
                    num_triangles=cfg.num_triangles,
                    source="real_control",
                    path=str(row.path),
                )
                | {"requested_length": int(length)}
            )
    real = pd.DataFrame(real_rows)
    atomic_write_parquet(metrics_dir / "real_control_metrics.parquet", real)
    atomic_write_csv(metrics_dir / "real_control_selection.csv", pd.DataFrame(control_reports))

    distribution = distribution_matching(generated, real, bootstrap_iterations=cfg.bootstrap_iterations)
    atomic_write_csv(metrics_dir / "distribution_matching.csv", distribution)

    pair_frames = []
    cluster_frames = []
    for length in cfg.lengths:
        samples = generated[generated["length"].astype(int) == int(length)]
        pairs = diversity_pairs_for_length(
            samples,
            contact_threshold=cfg.contact_threshold,
            pair_limit=cfg.diversity_pair_limit,
            seed=cfg.master_seed + int(length),
        )
        if not pairs.empty:
            pair_frames.append(pairs)
        cluster_frames.append(cluster_generated_samples(pairs, samples["sample_id"].astype(str).tolist()))
    diversity_pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    diversity_clusters = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    diversity_summary = (
        diversity_pairs.groupby("length", as_index=False)
        .agg(
            pair_count=("distance_map_rmse", "size"),
            distance_map_rmse_mean=("distance_map_rmse", "mean"),
            contact_hamming_distance_mean=("contact_hamming_distance", "mean"),
            contact_jaccard_distance_mean=("contact_jaccard_distance", "mean"),
            descriptor_distance_mean=("descriptor_distance", "mean"),
        )
        .sort_values("length")
        if not diversity_pairs.empty
        else pd.DataFrame()
    )
    atomic_write_parquet(metrics_dir / "diversity_pairs.parquet", diversity_pairs)
    atomic_write_csv(metrics_dir / "diversity_summary.csv", diversity_summary)
    atomic_write_parquet(metrics_dir / "diversity_clusters.parquet", diversity_clusters)

    train_descriptors = descriptor_cache_for_manifest(
        cfg.train_manifest,
        cfg.output_dir / "state" / "training_descriptors.parquet",
        contact_threshold=cfg.contact_threshold,
        num_triangles=cfg.num_triangles,
        resume=cfg.resume,
    )
    novelty = approximate_novelty(
        generated,
        train_descriptors,
        candidate_count=cfg.novelty_candidate_count,
        length_tolerance=cfg.real_length_tolerance,
        contact_threshold=cfg.contact_threshold,
    )
    calibration = novelty_calibration(
        real,
        train_descriptors,
        candidate_count=cfg.novelty_candidate_count,
        length_tolerance=cfg.real_length_tolerance,
        contact_threshold=cfg.contact_threshold,
    )
    atomic_write_parquet(metrics_dir / "novelty_per_sample.parquet", novelty)
    atomic_write_parquet(metrics_dir / "novelty_calibration.parquet", calibration)
    summary = summarize_metrics(
        generated=generated,
        real=real,
        diversity_summary=diversity_summary,
        novelty=novelty,
        calibration=calibration,
    )
    atomic_write_json(metrics_dir / "summary.json", summary)
    if cfg.plots:
        write_basic_figures(
            cfg.output_dir,
            generated=generated,
            real=real,
            distribution=distribution,
            diversity_summary=diversity_summary,
            novelty=novelty,
        )
    return {
        "generated": generated,
        "real": real,
        "distribution": distribution,
        "diversity_pairs": diversity_pairs,
        "diversity_summary": diversity_summary,
        "diversity_clusters": diversity_clusters,
        "novelty": novelty,
        "calibration": calibration,
    }


def run_evaluation(cfg: EvaluationConfig) -> Path:
    """Run resumable generated ensemble evaluation."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata", "state", "generated", "real_controls", "metrics", "figures"):
        (cfg.output_dir / name).mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(cfg.checkpoint, map_location="cpu")
    checkpoint["path"] = str(cfg.checkpoint)
    schedule = build_seed_schedule(cfg)
    core = protocol_core(cfg, checkpoint, schedule)
    core_hash = protocol_hash(core)
    protocol_path = cfg.output_dir / "protocol.json"
    started_utc = validate_resume_protocol(protocol_path, core, resume=cfg.resume, restart=cfg.restart)
    write_protocol(protocol_path, core, status="partial", started_utc=started_utc, completed_utc=None)
    copy_training_length_distribution(cfg.output_dir)
    conn = init_state(
        cfg.output_dir / "state" / "evaluation_state.sqlite",
        protocol_sha256=core_hash,
        restart=cfg.restart,
    )
    scale = load_normalization_scale(cfg.normalization_file, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, weights=cfg.weights, device=device)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(checkpoint["config"].get("diffusion_steps", 100)))).to(
        device
    )
    tasks = [(length, index, seed) for length, seeds in schedule.items() for index, seed in enumerate(seeds)]
    interrupted = False

    def _interrupt(signum: int, frame: Any) -> None:
        del signum, frame
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGINT, _interrupt)
    try:
        for length, index, seed in tqdm(tasks, desc="Generating ensemble", unit="sample"):
            key = sample_key(length, index, seed)
            path = generated_sample_path(cfg.output_dir, length, index, seed)
            row = conn.execute("SELECT status FROM generated_samples WHERE sample_key=?", (key,)).fetchone()
            if (
                cfg.resume
                and row is not None
                and row[0] == "completed"
                and validate_generated_npz(path, length=length, seed=seed, protocol_sha256=core_hash)
            ):
                continue
            try:
                generated_path = generate_one_sample(
                    model=model,
                    diffusion=diffusion,
                    checkpoint=checkpoint,
                    length=length,
                    index=index,
                    seed=seed,
                    scale=scale,
                    output_dir=cfg.output_dir,
                    weights=cfg.weights,
                    protocol_sha256=core_hash,
                    device=device,
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO generated_samples(
                        sample_key, length, sample_index, seed, path, status, error_message, updated_utc
                    )
                    VALUES (?, ?, ?, ?, ?, 'completed', '', ?)
                    """,
                    (key, int(length), int(index), int(seed), str(generated_path), utc_now()),
                )
                conn.commit()
            except KeyboardInterrupt:
                interrupted = True
                raise
            except Exception as exc:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO generated_samples(
                        sample_key, length, sample_index, seed, path, status, error_message, updated_utc
                    )
                    VALUES (?, ?, ?, ?, ?, 'failed', ?, ?)
                    """,
                    (key, int(length), int(index), int(seed), str(path), str(exc), utc_now()),
                )
                conn.commit()
                raise
    except KeyboardInterrupt:
        interrupted = True
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        conn.close()
    if interrupted:
        write_protocol(protocol_path, core, status="partial", started_utc=started_utc, completed_utc=None)
        print(f"Interrupted. Resume with: {resume_command(cfg)}", file=sys.stderr)
        return protocol_path
    evaluate_existing_outputs(cfg, protocol_sha256_value=core_hash)
    write_protocol(protocol_path, core, status="completed", started_utc=started_utc, completed_utc=utc_now())
    return protocol_path


def parse_lengths(value: str) -> tuple[int, ...]:
    """Parse comma-separated requested lengths."""
    lengths = tuple(int(item) for item in value.replace(" ", "").split(",") if item)
    if not lengths:
        raise ValueError("At least one length is required")
    if any(length <= 0 for length in lengths):
        raise ValueError("Lengths must be positive")
    return lengths


def parse_seeds(value: str | None) -> tuple[int, ...] | None:
    """Parse optional explicit seed list."""
    if value is None or value == "":
        return None
    seeds = tuple(int(item) for item in value.replace(" ", "").split(",") if item)
    if not seeds:
        return None
    return seeds


def parse_length_samples(value: str | None) -> dict[int, int] | None:
    """Parse per-length counts like '64:100,128:100'."""
    if value is None or value == "":
        return None
    out: dict[int, int] = {}
    for item in value.replace(" ", "").split(","):
        if not item:
            continue
        length, count = item.split(":", maxsplit=1)
        out[int(length)] = int(count)
    if any(length <= 0 or count <= 0 for length, count in out.items()):
        raise ValueError("--length-samples lengths and counts must be positive")
    return out or None


def config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Build EvaluationConfig from CLI args."""
    seeds = parse_seeds(args.seeds)
    samples_by_length = parse_length_samples(args.length_samples)
    lengths = tuple(sorted(samples_by_length)) if samples_by_length is not None else parse_lengths(args.lengths)
    samples_per_length = len(seeds) if seeds is not None else int(args.samples_per_length)
    if samples_per_length <= 0:
        raise ValueError("--samples-per-length must be positive")
    return EvaluationConfig(
        checkpoint=args.checkpoint,
        weights=args.weights,
        config_path=args.config,
        normalization_file=args.normalization_file,
        train_manifest=args.train_manifest,
        reference_manifest=args.reference_manifest,
        output_dir=args.output_dir,
        lengths=lengths,
        samples_per_length=samples_per_length,
        samples_by_length=samples_by_length,
        master_seed=int(args.master_seed),
        seeds=seeds,
        contact_threshold=float(args.contact_threshold),
        num_triangles=int(args.num_triangles),
        novelty_candidate_count=int(args.novelty_candidate_count),
        workers=int(args.workers),
        resume=bool(args.resume),
        restart=bool(args.restart),
        plots=bool(args.plots),
        control_count=int(args.control_count),
        real_length_tolerance=int(args.real_length_tolerance),
        diversity_pair_limit=int(args.diversity_pair_limit),
        bootstrap_iterations=int(args.bootstrap_iterations),
    )


def resume_command(cfg: EvaluationConfig) -> str:
    """Return a shell command that resumes this evaluation."""
    sample_count_args = (
        f"--length-samples {','.join(f'{length}:{count}' for length, count in sorted(cfg.samples_by_length.items()))} "
        if cfg.samples_by_length is not None
        else f"--samples-per-length {cfg.samples_per_length} "
    )
    return (
        "python scripts/evaluate_generated_ensemble.py "
        f"--checkpoint {cfg.checkpoint} "
        f"--weights {cfg.weights} "
        f"--normalization-file {cfg.normalization_file} "
        f"--train-manifest {cfg.train_manifest} "
        f"--reference-manifest {cfg.reference_manifest} "
        f"--output-dir {cfg.output_dir} "
        f"--lengths {','.join(str(x) for x in cfg.lengths)} "
        f"{sample_count_args}"
        f"--master-seed {cfg.master_seed} "
        f"--contact-threshold {cfg.contact_threshold} "
        f"--num-triangles {cfg.num_triangles} "
        f"--novelty-candidate-count {cfg.novelty_candidate_count} "
        "--resume"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--config", type=Path, default=None, help="Training config path recorded in protocol.")
    parser.add_argument("--normalization-file", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lengths", default="", help="Comma-separated requested lengths.")
    parser.add_argument("--samples-per-length", type=int, default=4)
    parser.add_argument("--length-samples", default=None, help="Per-length counts, e.g. 64:100,128:100.")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated explicit seed list per length.")
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    parser.add_argument("--num-triangles", type=int, default=2048)
    parser.add_argument("--novelty-candidate-count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1, help="Reserved for safe future parallel metric work.")
    parser.add_argument("--control-count", type=int, default=64)
    parser.add_argument("--real-length-tolerance", type=int, default=8)
    parser.add_argument("--diversity-pair-limit", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    start = time.time()
    cfg = config_from_args(args)
    protocol = run_evaluation(cfg)
    print(f"Wrote evaluation protocol to {protocol} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
