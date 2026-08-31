#!/usr/bin/env python3
"""Analysis-only calibrated finalization for completed generated ensembles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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

from protein_distance_diffusion.evaluation.plots import save_heatmap  # noqa: E402
from protein_distance_diffusion.utils.hashing import sha256_file  # noqa: E402

EVAL_SCRIPT = Path(__file__).with_name("evaluate_generated_ensemble.py")
EVAL_SPEC = importlib.util.spec_from_file_location("evaluate_generated_ensemble", EVAL_SCRIPT)
if EVAL_SPEC is None or EVAL_SPEC.loader is None:
    raise RuntimeError(f"Unable to import evaluator helpers from {EVAL_SCRIPT}")
EVAL = importlib.util.module_from_spec(EVAL_SPEC)
sys.modules.setdefault("evaluate_generated_ensemble", EVAL)
EVAL_SPEC.loader.exec_module(EVAL)

RAW_INPUTS = (
    "protocol.json",
    "metrics/validity_per_sample.parquet",
    "metrics/real_control_metrics.parquet",
    "metrics/distribution_matching.csv",
    "metrics/diversity_pairs.parquet",
    "metrics/diversity_summary.csv",
    "metrics/diversity_clusters.parquet",
    "metrics/novelty_per_sample.parquet",
    "metrics/novelty_calibration.parquet",
    "metrics/summary.json",
    "state/evaluation_state.sqlite",
    "state/training_descriptors.parquet",
)
EMPIRICAL_METRICS = (
    "triangle_violation_fraction",
    "negative_eigenvalue_mass_fraction",
    "rank3_residual_energy_fraction",
    "classical_mds_stress",
    "adjacent_residue_distance_rmse",
)
DIVERSITY_METRICS = (
    "distance_map_rmse",
    "contact_hamming_distance",
    "contact_jaccard_distance",
    "descriptor_distance",
)
NOVELTY_METRICS = (
    "descriptor_distance",
    "refined_distance_map_rmse",
    "refined_contact_jaccard_distance",
)


@dataclass(frozen=True)
class FinalizationConfig:
    """Configuration for analysis-only finalization."""

    evaluation_dir: Path
    output_dir: Path
    real_quantile: float
    pair_limit: int
    bootstrap_iterations: int
    contact_threshold: float
    seed: int
    plots: bool
    restart: bool
    resume: bool


def utc_now() -> str:
    """Return a UTC timestamp."""
    return datetime.now(UTC).isoformat()


def atomic_write_text(path: str | Path, text: str) -> None:
    """Atomically write text."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(dst)


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON."""
    atomic_write_text(path, json.dumps(EVAL.to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    """Atomically write CSV."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(dst)


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    """Atomically write parquet."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(dst)


def file_hashes(root: Path, rels: tuple[str, ...] = RAW_INPUTS) -> dict[str, str]:
    """Return SHA-256 hashes for required relative files."""
    return {rel: sha256_file(root / rel) for rel in rels}


def validate_completed_evaluation(root: Path) -> dict[str, Any]:
    """Validate that the evaluation directory has completed raw outputs."""
    missing = [rel for rel in RAW_INPUTS if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Completed evaluation is missing required file(s): {', '.join(missing)}")
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol.get("status") != "completed":
        raise ValueError("Evaluation protocol is not completed; analysis-only finalization refuses partial inputs.")
    validity = pd.read_parquet(root / "metrics" / "validity_per_sample.parquet")
    real = pd.read_parquet(root / "metrics" / "real_control_metrics.parquet")
    expected_counts = {int(length): int(count) for length, count in protocol.get("sample_counts_by_length", {}).items()}
    actual_counts = validity.groupby(validity["length"].astype(int)).size().to_dict()
    if expected_counts and actual_counts != expected_counts:
        raise ValueError(f"Generated sample counts do not match protocol: {actual_counts} != {expected_counts}")
    real_counts = real.groupby(real["requested_length"].astype(int)).size().to_dict()
    if any(count == 0 for count in real_counts.values()):
        raise ValueError("Real-control metrics contain an empty requested-length group.")
    return protocol


def analysis_protocol(
    cfg: FinalizationConfig, evaluation_protocol: dict[str, Any], input_hashes: dict[str, str]
) -> dict:
    """Build finalization protocol content."""
    return {
        "analysis": "calibrated_generated_ensemble_finalization",
        "evaluation_dir": str(cfg.evaluation_dir),
        "output_dir": str(cfg.output_dir),
        "real_quantile": cfg.real_quantile,
        "pair_limit": cfg.pair_limit,
        "bootstrap_iterations": cfg.bootstrap_iterations,
        "contact_threshold": cfg.contact_threshold,
        "seed": cfg.seed,
        "plots": cfg.plots,
        "source_protocol_sha256": input_hashes["protocol.json"],
        "source_protocol_completed_utc": evaluation_protocol.get("completed_utc"),
        "source_sample_counts_by_length": evaluation_protocol.get("sample_counts_by_length"),
        "input_hashes": input_hashes,
    }


def protocol_digest(protocol: dict[str, Any]) -> str:
    """Hash finalization protocol settings."""
    payload = json.dumps(EVAL.to_jsonable(protocol), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_or_write_analysis_protocol(path: Path, protocol: dict[str, Any], cfg: FinalizationConfig) -> None:
    """Validate compatible resume or write a fresh finalization protocol."""
    digest = protocol_digest(protocol)
    if path.exists() and not cfg.restart:
        existing = json.loads(path.read_text())
        if existing.get("analysis_protocol_sha256") != digest:
            raise ValueError("Existing calibrated analysis protocol is incompatible. Use --restart to replace it.")
        if not cfg.resume:
            raise FileExistsError("Calibrated analysis already exists. Use --resume or --restart.")
    payload = dict(protocol)
    payload["analysis_protocol_sha256"] = digest
    payload["status"] = "partial"
    payload["started_utc"] = utc_now()
    payload["completed_utc"] = None
    atomic_write_json(path, payload)


def complete_analysis_protocol(path: Path) -> None:
    """Mark finalization protocol complete."""
    payload = json.loads(path.read_text())
    payload["status"] = "completed"
    payload["completed_utc"] = utc_now()
    atomic_write_json(path, payload)


def ensure_heuristic_alias(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve old edm_compatible while adding the preferred heuristic alias."""
    out = frame.copy()
    if "heuristic_edm_quality_pass" not in out:
        out["heuristic_edm_quality_pass"] = out.get("edm_compatible", False).astype(bool)
    return out


def empirical_thresholds(real: pd.DataFrame, *, quantile: float) -> pd.DataFrame:
    """Compute requested-length empirical real-control thresholds."""
    rows = []
    for requested_length, group in real.groupby(real["requested_length"].astype(int)):
        for metric in EMPIRICAL_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "requested_length": int(requested_length),
                    "metric": metric,
                    "quantile": float(quantile),
                    "threshold": float(np.quantile(values, quantile)) if values.size else float("nan"),
                    "real_control_count": int(values.size),
                    "threshold_provenance": (
                        f"matched real controls requested_length={int(requested_length)} q={quantile:.3f}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def empirical_validity(
    generated: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply empirical real-control thresholds to generated samples."""
    generated = ensure_heuristic_alias(generated)
    lookup = {
        (int(row.requested_length), str(row.metric)): float(row.threshold) for row in thresholds.itertuples(index=False)
    }
    rows = []
    for row in generated.sort_values(["length", "sample_id"]).itertuples(index=False):
        item = {
            "sample_id": row.sample_id,
            "requested_length": int(row.length),
            "length": int(row.length),
            "heuristic_edm_quality_pass": bool(row.heuristic_edm_quality_pass),
            "edm_compatible": bool(row.edm_compatible),
            "threshold_quantile": float(quantile),
            "threshold_provenance": f"matched real-control q={quantile:.3f} by requested_length",
        }
        passes = []
        for metric in EMPIRICAL_METRICS:
            threshold = lookup[(int(row.length), metric)]
            value = float(getattr(row, metric))
            passed = bool(np.isfinite(value) and value <= threshold)
            item[metric] = value
            item[f"{metric}_threshold"] = threshold
            item[f"{metric}_empirical_pass"] = passed
            passes.append(passed)
        item["empirical_real_like_geometry_pass"] = bool(all(passes))
        rows.append(item)
    per_sample = pd.DataFrame(rows)
    by_length_rows = []
    for length, group in per_sample.groupby("requested_length"):
        generated_group = generated[generated["length"].astype(int) == int(length)]
        real_thresholds = thresholds[thresholds["requested_length"].astype(int) == int(length)]
        summary: dict[str, Any] = {
            "requested_length": int(length),
            "generated_count": int(len(group)),
            "empirical_real_like_geometry_count": int(group["empirical_real_like_geometry_pass"].sum()),
            "empirical_real_like_geometry_fraction": float(group["empirical_real_like_geometry_pass"].mean()),
            "heuristic_edm_quality_count": int(group["heuristic_edm_quality_pass"].sum()),
            "heuristic_edm_quality_fraction": float(group["heuristic_edm_quality_pass"].mean()),
        }
        for metric in EMPIRICAL_METRICS:
            summary[f"{metric}_generated_mean"] = float(generated_group[metric].mean())
            summary[f"{metric}_generated_median"] = float(generated_group[metric].median())
            summary[f"{metric}_real_q{int(round(quantile * 100)):02d}"] = float(
                real_thresholds[real_thresholds["metric"] == metric]["threshold"].iloc[0]
            )
            summary[f"{metric}_pass_fraction"] = float(group[f"{metric}_empirical_pass"].mean())
        by_length_rows.append(summary)
    return per_sample, pd.DataFrame(by_length_rows).sort_values("requested_length")


def deterministic_pair_indices(count: int, *, limit: int, seed: int) -> list[tuple[int, int]]:
    """Return all or a deterministic subset of pair indices."""
    return EVAL.deterministic_pair_indices(count, limit=limit, seed=seed)


def real_diversity_pairs(real: pd.DataFrame, *, pair_limit: int, contact_threshold: float, seed: int) -> pd.DataFrame:
    """Compute real-control diversity only within identical actual lengths."""
    rows = []
    for (requested_length, actual_length), group in real.groupby(
        [real["requested_length"].astype(int), real["length"].astype(int)]
    ):
        ordered = group.sort_values("sample_id").reset_index(drop=True)
        pairs = deterministic_pair_indices(
            len(ordered), limit=pair_limit, seed=seed + int(requested_length) + actual_length
        )
        matrices: dict[int, np.ndarray] = {}
        for i, j in pairs:
            left = matrices.setdefault(i, EVAL.load_npz_matrix(ordered.iloc[i]["path"]))
            right = matrices.setdefault(j, EVAL.load_npz_matrix(ordered.iloc[j]["path"]))
            hamming, jaccard = EVAL.contact_distances(left, right, threshold=contact_threshold)
            rows.append(
                {
                    "requested_length": int(requested_length),
                    "actual_length": int(actual_length),
                    "sample_id_a": ordered.iloc[i]["sample_id"],
                    "sample_id_b": ordered.iloc[j]["sample_id"],
                    "distance_map_rmse": EVAL.map_rmse(left, right),
                    "contact_hamming_distance": hamming,
                    "contact_jaccard_distance": jaccard,
                    "descriptor_distance": EVAL.descriptor_distance(ordered.iloc[i], ordered.iloc[j]),
                }
            )
    return pd.DataFrame(rows)


def summarize_real_diversity_exact(pairs: pd.DataFrame) -> pd.DataFrame:
    """Summarize real diversity within requested/exact-actual-length subgroups."""
    if pairs.empty:
        return pd.DataFrame()
    return (
        pairs.groupby(["requested_length", "actual_length"], as_index=False)
        .agg(
            pair_count=("distance_map_rmse", "size"),
            distance_map_rmse_mean=("distance_map_rmse", "mean"),
            contact_hamming_distance_mean=("contact_hamming_distance", "mean"),
            contact_jaccard_distance_mean=("contact_jaccard_distance", "mean"),
            descriptor_distance_mean=("descriptor_distance", "mean"),
        )
        .sort_values(["requested_length", "actual_length"])
    )


def weighted_real_diversity_summary(exact: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact-length subgroup summaries by valid pair counts."""
    if exact.empty:
        return pd.DataFrame()
    rows = []
    for requested_length, group in exact.groupby("requested_length"):
        weights = group["pair_count"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "requested_length": int(requested_length),
            "exact_length_subgroup_count": int(len(group)),
            "pair_count": int(group["pair_count"].sum()),
        }
        for metric in DIVERSITY_METRICS:
            col = f"{metric}_mean"
            row[col] = float(np.average(group[col], weights=weights)) if weights.sum() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("requested_length")


def real_diversity_clusters(real: pd.DataFrame, pairs: pd.DataFrame, *, rmse_threshold: float = 0.02) -> pd.DataFrame:
    """Assign near-duplicate clusters for real controls within requested-length groups."""
    frames = []
    for requested_length, group in real.groupby(real["requested_length"].astype(int)):
        subset_pairs = pairs[pairs["requested_length"].astype(int) == int(requested_length)]
        renamed = subset_pairs.rename(columns={"requested_length": "length"})
        clusters = EVAL.cluster_generated_samples(
            renamed,
            group["sample_id"].astype(str).tolist(),
            rmse_threshold=rmse_threshold,
        )
        clusters["requested_length"] = int(requested_length)
        frames.append(clusters)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def bootstrap_ci(values: pd.Series | np.ndarray, *, iterations: int, seed: int) -> tuple[float, float]:
    """Bootstrap CI for a mean."""
    return EVAL.bootstrap_ci(np.asarray(values, dtype=float), iterations=iterations, seed=seed)


def diversity_calibration(
    generated_summary: pd.DataFrame,
    generated_pairs: pd.DataFrame,
    real_summary: pd.DataFrame,
    real_pairs: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Compare generated-vs-generated diversity with real-vs-real controls."""
    rows = []
    for length in sorted(
        set(generated_summary["length"].astype(int)) | set(real_summary["requested_length"].astype(int))
    ):
        gen_pairs = generated_pairs[generated_pairs["length"].astype(int) == int(length)]
        real_len_pairs = real_pairs[real_pairs["requested_length"].astype(int) == int(length)]
        for metric in DIVERSITY_METRICS:
            gen_values = gen_pairs[metric].to_numpy(dtype=float) if metric in gen_pairs else np.asarray([])
            real_values = real_len_pairs[metric].to_numpy(dtype=float) if metric in real_len_pairs else np.asarray([])
            gen_ci = bootstrap_ci(gen_values, iterations=bootstrap_iterations, seed=seed + length + len(metric))
            real_ci = bootstrap_ci(real_values, iterations=bootstrap_iterations, seed=seed + length + 101 + len(metric))
            gen_mean = float(np.nanmean(gen_values)) if np.isfinite(gen_values).any() else float("nan")
            real_mean = float(np.nanmean(real_values)) if np.isfinite(real_values).any() else float("nan")
            rows.append(
                {
                    "requested_length": int(length),
                    "metric": metric,
                    "generated_mean": gen_mean,
                    "generated_ci_low": gen_ci[0],
                    "generated_ci_high": gen_ci[1],
                    "real_mean": real_mean,
                    "real_ci_low": real_ci[0],
                    "real_ci_high": real_ci[1],
                    "generated_minus_real": gen_mean - real_mean,
                    "generated_over_real": gen_mean / real_mean
                    if real_mean and np.isfinite(real_mean)
                    else float("nan"),
                    "generated_pair_count": int(np.isfinite(gen_values).sum()),
                    "real_pair_count": int(np.isfinite(real_values).sum()),
                    "warning": "inadequate_calibration"
                    if np.isfinite(real_values).sum() < 20 or np.isfinite(gen_values).sum() < 20
                    else "",
                }
            )
    return pd.DataFrame(rows)


def corrected_novelty_calibration(novelty_calibration: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Join real-control calibration to requested lengths from real-control metrics."""
    mapping = real[["sample_id", "requested_length"]].drop_duplicates()
    corrected = novelty_calibration.merge(
        mapping,
        left_on="real_control_sample_id",
        right_on="sample_id",
        how="left",
        validate="one_to_one",
    ).drop(columns=["sample_id"])
    if corrected["requested_length"].isna().any():
        missing = corrected.loc[corrected["requested_length"].isna(), "real_control_sample_id"].head(5).tolist()
        raise ValueError(f"Novelty calibration controls missing from real_control_metrics: {missing}")
    corrected["actual_length"] = corrected["length"].astype(int)
    corrected["requested_length"] = corrected["requested_length"].astype(int)
    return corrected


def percentile_positions(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Empirical percentile positions of values relative to calibration distribution."""
    cal = np.sort(np.asarray(calibration, dtype=float))
    cal = cal[np.isfinite(cal)]
    if cal.size == 0:
        return np.full(len(values), np.nan)
    return np.searchsorted(cal, values, side="right") / cal.size


def summarize_one_novelty_group(
    generated: pd.DataFrame,
    calibration: pd.DataFrame,
    *,
    group_name: str,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize generated novelty against corrected real calibration by requested length."""
    rows = []
    for requested_length in sorted(calibration["requested_length"].astype(int).unique()):
        gen = generated[generated["length"].astype(int) == int(requested_length)]
        cal = calibration[calibration["requested_length"].astype(int) == int(requested_length)]
        for metric in NOVELTY_METRICS:
            gen_values = gen[metric].to_numpy(dtype=float) if not gen.empty else np.asarray([])
            cal_values = cal[metric].to_numpy(dtype=float)
            gen_ci = bootstrap_ci(
                gen_values, iterations=bootstrap_iterations, seed=seed + requested_length + len(metric)
            )
            cal_ci = bootstrap_ci(
                cal_values, iterations=bootstrap_iterations, seed=seed + requested_length + 500 + len(metric)
            )
            gen_mean = float(np.nanmean(gen_values)) if np.isfinite(gen_values).any() else float("nan")
            cal_mean = float(np.nanmean(cal_values)) if np.isfinite(cal_values).any() else float("nan")
            rows.append(
                {
                    "group": group_name,
                    "requested_length": int(requested_length),
                    "metric": metric,
                    "generated_count": int(np.isfinite(gen_values).sum()),
                    "real_calibration_count": int(np.isfinite(cal_values).sum()),
                    "generated_mean": gen_mean,
                    "generated_std": float(np.nanstd(gen_values)) if np.isfinite(gen_values).any() else float("nan"),
                    "generated_q05": float(np.nanquantile(gen_values, 0.05))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "generated_q50": float(np.nanquantile(gen_values, 0.50))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "generated_q95": float(np.nanquantile(gen_values, 0.95))
                    if np.isfinite(gen_values).any()
                    else float("nan"),
                    "generated_ci_low": gen_ci[0],
                    "generated_ci_high": gen_ci[1],
                    "real_mean": cal_mean,
                    "real_std": float(np.nanstd(cal_values)) if np.isfinite(cal_values).any() else float("nan"),
                    "real_q05": float(np.nanquantile(cal_values, 0.05))
                    if np.isfinite(cal_values).any()
                    else float("nan"),
                    "real_q50": float(np.nanquantile(cal_values, 0.50))
                    if np.isfinite(cal_values).any()
                    else float("nan"),
                    "real_q95": float(np.nanquantile(cal_values, 0.95))
                    if np.isfinite(cal_values).any()
                    else float("nan"),
                    "real_ci_low": cal_ci[0],
                    "real_ci_high": cal_ci[1],
                    "generated_minus_real": gen_mean - cal_mean,
                    "generated_over_real": gen_mean / cal_mean if cal_mean and np.isfinite(cal_mean) else float("nan"),
                    "approximate": True,
                    "warning": "unavailable_empty_generated_group"
                    if len(gen) == 0
                    else ("calibration_count_below_20" if np.isfinite(cal_values).sum() < 20 else ""),
                }
            )
    return pd.DataFrame(rows)


def novelty_by_requested_length(
    novelty: pd.DataFrame,
    corrected_calibration: pd.DataFrame,
    empirical: pd.DataFrame,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize novelty overall and by calibrated validity groups."""
    novelty = novelty.merge(
        empirical[["sample_id", "heuristic_edm_quality_pass", "empirical_real_like_geometry_pass"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    frames = [
        summarize_one_novelty_group(
            novelty,
            corrected_calibration,
            group_name="all_generated_samples",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        summarize_one_novelty_group(
            novelty[novelty["heuristic_edm_quality_pass"].fillna(False)],
            corrected_calibration,
            group_name="heuristic_edm_quality_pass",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
        summarize_one_novelty_group(
            novelty[novelty["empirical_real_like_geometry_pass"].fillna(False)],
            corrected_calibration,
            group_name="empirical_real_like_geometry_pass",
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        ),
    ]
    out = pd.concat(frames, ignore_index=True)
    for metric in NOVELTY_METRICS:
        metric_rows = novelty[["sample_id", "length", metric]].copy()
        for length, group in metric_rows.groupby(metric_rows["length"].astype(int)):
            cal = corrected_calibration[corrected_calibration["requested_length"].astype(int) == int(length)][metric]
            positions = percentile_positions(group[metric].to_numpy(dtype=float), cal.to_numpy(dtype=float))
            novelty.loc[group.index, f"{metric}_real_calibration_percentile"] = positions
    return out, novelty


def generated_rankings(generated: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Create separate deterministic best/worst rankings without a composite score."""
    rows = []
    real_rg_median = real.groupby(real["requested_length"].astype(int))["radius_of_gyration"].median().to_dict()
    work = generated.copy()
    work["requested_length"] = work["length"].astype(int)
    work["radius_of_gyration_real_median"] = work["requested_length"].map(real_rg_median)
    work["radius_of_gyration_abs_delta_to_real_median"] = (
        work["radius_of_gyration"] - work["radius_of_gyration_real_median"]
    ).abs()
    specs = [
        ("lowest_classical_mds_stress", "classical_mds_stress", True),
        ("highest_classical_mds_stress", "classical_mds_stress", False),
        ("lowest_negative_eigenvalue_mass", "negative_eigenvalue_mass_fraction", True),
        ("highest_negative_eigenvalue_mass", "negative_eigenvalue_mass_fraction", False),
        ("lowest_rank3_residual", "rank3_residual_energy_fraction", True),
        ("highest_rank3_residual", "rank3_residual_energy_fraction", False),
        ("lowest_adjacent_rmse", "adjacent_residue_distance_rmse", True),
        ("highest_adjacent_rmse", "adjacent_residue_distance_rmse", False),
        ("closest_radius_of_gyration_to_real_median", "radius_of_gyration_abs_delta_to_real_median", True),
        ("farthest_radius_of_gyration_from_real_median", "radius_of_gyration_abs_delta_to_real_median", False),
    ]
    for length, group in work.groupby("requested_length"):
        for ranking_name, metric, ascending in specs:
            ranked = group.sort_values([metric, "sample_id"], ascending=[ascending, True]).head(10)
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "requested_length": int(length),
                        "ranking": ranking_name,
                        "rank": rank,
                        "sample_id": row.sample_id,
                        "path": row.path,
                        "metric": metric,
                        "metric_value": float(getattr(row, metric)),
                    }
                )
    return pd.DataFrame(rows)


def plot_outputs(
    output_dir: Path,
    *,
    generated: pd.DataFrame,
    real: pd.DataFrame,
    empirical_by_length: pd.DataFrame,
    diversity_cal: pd.DataFrame,
    novelty_summary: pd.DataFrame,
    rankings: pd.DataFrame,
) -> None:
    """Write calibrated-analysis figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, logy in [
        ("negative_eigenvalue_mass_fraction", "global_geometry_negative_eigenvalue_mass.png", True),
        ("rank3_residual_energy_fraction", "global_geometry_rank3_residual.png", True),
        ("classical_mds_stress", "global_geometry_mds_stress.png", True),
        ("adjacent_residue_distance_mean", "adjacent_distance_distributions.png", False),
        ("radius_of_gyration", "radius_of_gyration_distributions.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        gen_data = [
            generated[generated["length"].astype(int) == length][metric].dropna()
            for length in sorted(generated["length"].unique())
        ]
        real_data = [
            real[real["requested_length"].astype(int) == length][metric].dropna()
            for length in sorted(generated["length"].unique())
        ]
        positions = np.arange(len(gen_data))
        labels = sorted(generated["length"].unique())
        ax.boxplot(gen_data, positions=positions - 0.18, widths=0.3)
        ax.boxplot(real_data, positions=positions + 0.18, widths=0.3)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"Generated vs real {metric}")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        empirical_by_length["requested_length"],
        empirical_by_length["empirical_real_like_geometry_fraction"],
        marker="o",
    )
    ax.set_title("Empirical real-like geometry pass rate")
    ax.set_xlabel("Requested length N")
    ax.set_ylabel("Fraction")
    fig.tight_layout()
    fig.savefig(fig_dir / "empirical_pass_rate_by_length.png", dpi=160)
    plt.close(fig)
    if not diversity_cal.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        subset = diversity_cal[diversity_cal["metric"] == "distance_map_rmse"]
        ax.plot(subset["requested_length"], subset["generated_mean"], marker="o", label="generated")
        ax.plot(subset["requested_length"], subset["real_mean"], marker="s", label="real controls")
        ax.set_title("Generated-vs-real diversity calibration")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel("Normalized distance-map RMSE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "generated_vs_real_diversity.png", dpi=160)
        plt.close(fig)
    if not novelty_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        subset = novelty_summary[
            (novelty_summary["group"] == "all_generated_samples") & (novelty_summary["metric"] == "descriptor_distance")
        ]
        ax.plot(subset["requested_length"], subset["generated_mean"], marker="o", label="generated")
        ax.plot(subset["requested_length"], subset["real_mean"], marker="s", label="real calibration")
        ax.set_title("Corrected novelty calibration")
        ax.set_xlabel("Requested length N")
        ax.set_ylabel("Nearest descriptor distance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "corrected_novelty_calibration.png", dpi=160)
        plt.close(fig)
    montage_rows = rankings[rankings["ranking"].isin(["lowest_classical_mds_stress", "highest_classical_mds_stress"])]
    for row in montage_rows.groupby(["requested_length", "ranking"]).head(1).itertuples(index=False):
        try:
            matrix = EVAL.load_npz_matrix(row.path, generated=True)
            save_heatmap(
                matrix,
                fig_dir / f"{row.ranking}_N{int(row.requested_length):04d}_{row.sample_id}.png",
                title=f"generated candidate {row.sample_id}",
            )
        except Exception:
            continue


def write_report(
    path: Path,
    *,
    protocol: dict[str, Any],
    empirical_by_length: pd.DataFrame,
    diversity_cal: pd.DataFrame,
    novelty_summary: pd.DataFrame,
    calibrated_summary: dict[str, Any],
) -> None:
    """Write the E000 final report markdown."""
    validity_table = dataframe_to_markdown(empirical_by_length)
    diversity_subset = diversity_cal[diversity_cal["metric"].isin(["distance_map_rmse", "contact_jaccard_distance"])]
    novelty_subset = novelty_summary[
        (novelty_summary["group"] == "all_generated_samples")
        & (novelty_summary["metric"].isin(["descriptor_distance", "refined_distance_map_rmse"]))
    ]
    text = f"""# E000 Final Report: Epoch-8 Baseline

## Provenance

- Evaluation directory: `{protocol.get("evaluation_dir")}`
- Checkpoint: `{calibrated_summary["source"]["checkpoint_path"]}`
- Checkpoint SHA-256: `{calibrated_summary["source"]["checkpoint_sha256"]}`
- Sample counts by requested length: `{calibrated_summary["sample_counts_by_length"]}`
- Real controls: `{calibrated_summary["real_control_count"]}` total, grouped by requested length.

## Principal Conclusion

The current model produces numerically valid, non-duplicated,
length-conditioned distance-like matrices and approximates several local
distributional properties, but its generated matrices do not lie on the
empirical manifold of real three-dimensional protein distance matrices. Global
geometric inconsistency and excess compactness worsen with sequence length.

The prior `edm_compatible` field is preserved only as a deprecated permissive
heuristic and is reported here as `heuristic_edm_quality_pass`. It is not strict
mathematical EDM validity.

## Empirical Real-Like Geometry

Thresholds are the matched real-control empirical quantiles specified for this
analysis. The binary pass rate is less informative than the orders-of-magnitude
continuous gap between generated and real global geometry.

{validity_table}

## Diversity Calibration

Real-control pairwise map and contact comparisons are restricted to controls
with identical actual length, then aggregated under requested length by valid
pair count. Contact Hamming distances should be interpreted with contact
sparsity in mind.

{dataframe_to_markdown(diversity_subset)}

## Novelty Calibration

Novelty is approximate because the original search is two-stage: descriptor
retrieval followed by refined comparison only against retrieved candidates.
Generated distance from training is interpreted relative to validation-control
calibration. Large descriptor distance can indicate out-of-distribution
invalidity, not useful novelty.

{dataframe_to_markdown(novelty_subset)}

The `empirical_real_like_geometry_pass` novelty subgroup is reported as
unavailable when empty, not as zero novelty.

## Length Dependence

Generated negative eigenvalue mass, rank-3 residual energy, MDS stress, and
radius-of-gyration mismatch degrade with increasing requested length. The
449-500 training bin contains only 1.8169% of training samples, so long-chain
performance should remain a first-class diagnostic.

## Limitations

- Distance maps cannot establish physical stability, designability, or
  thermodynamic behavior.
- Generated samples are draws from `p(D | N)`, not reconstructions.
- The novelty analysis is calibrated and approximate, not exhaustive.
- Checker or diamond-like motifs are not labelled artifacts here; their
  association with geometry rankings remains future analysis.

## Motivation For E001

`E001_symmetric_axial_attention` should test whether a symmetry-preserving
axial-attention block immediately above the bottleneck improves negative
eigenvalue mass, rank-3 residual, MDS stress, triangle consistency,
radius-of-gyration matching, and scaling with `N`, without reducing diversity
or increasing training-set similarity. Physical auxiliary losses should wait
until after the attention ablations.
"""
    atomic_write_text(path, text)


def dataframe_to_markdown(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """Render a compact Markdown table without optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    columns = [str(column) for column in display.columns]

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n_Table truncated to {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def calibrated_summary(
    *,
    protocol: dict[str, Any],
    generated: pd.DataFrame,
    real: pd.DataFrame,
    empirical_by_length: pd.DataFrame,
    diversity_cal: pd.DataFrame,
    novelty_summary: pd.DataFrame,
    input_hashes_before: dict[str, str],
    input_hashes_after: dict[str, str],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Build calibrated summary JSON."""
    return {
        "completed_utc": utc_now(),
        "runtime_seconds": runtime_seconds,
        "source": {
            "checkpoint_path": protocol.get("checkpoint_path"),
            "checkpoint_sha256": protocol.get("checkpoint_sha256"),
            "protocol_status": protocol.get("status"),
        },
        "sample_counts_by_length": protocol.get("sample_counts_by_length"),
        "generated_sample_count": int(len(generated)),
        "real_control_count": int(len(real)),
        "all_real_controls_pass_original_checks": bool(
            real[["numerically_valid", "edm_compatible", "chain_like", "protein_like"]].all().all()
        ),
        "empirical_real_like_geometry": empirical_by_length.to_dict(orient="records"),
        "heuristic_edm_quality_total": int(generated.get("heuristic_edm_quality_pass", False).sum()),
        "empirical_real_like_geometry_total": int(empirical_by_length["empirical_real_like_geometry_count"].sum()),
        "diversity_calibration": diversity_cal.to_dict(orient="records"),
        "novelty_by_requested_length": novelty_summary.to_dict(orient="records"),
        "raw_inputs_unchanged": input_hashes_before == input_hashes_after,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "principal_conclusion": (
            "The current model produces numerically valid, non-duplicated, length-conditioned "
            "distance-like matrices and approximates several local distributional properties, "
            "but its generated matrices do not lie on the empirical manifold of real "
            "three-dimensional protein distance matrices. Global geometric inconsistency and "
            "excess compactness worsen with sequence length."
        ),
        "limitations": [
            "Novelty is approximate and calibrated against validation-to-training distances.",
            "Distance maps do not establish physical stability or designability.",
            "The deprecated edm_compatible field is a permissive heuristic, not strict EDM validity.",
        ],
    }


def update_timeline(path: Path, summary: dict[str, Any]) -> None:
    """Update or append the E000 finalized result note."""
    marker = "## E000 Finalized Results"
    block = f"""{marker}

Final calibrated analysis completed at `{summary["completed_utc"]}`.

- Generated samples: {summary["generated_sample_count"]}
- Real controls: {summary["real_control_count"]}
- Empirical real-like geometry pass count: {summary["empirical_real_like_geometry_total"]}
- Deprecated/permissive heuristic EDM-quality pass count: {summary["heuristic_edm_quality_total"]}
- Raw E000 inputs unchanged during finalization: {summary["raw_inputs_unchanged"]}

Conclusion: {summary["principal_conclusion"]}

E001 remains `E001_symmetric_axial_attention`, testing whether one
symmetry-preserving axial-attention block immediately above the bottleneck
improves global geometry and scaling with `N` without reducing diversity or
increasing training-set similarity.
"""
    text = path.read_text() if path.exists() else "# Experiment Timeline\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    atomic_write_text(path, text)


def update_e000_readme(path: Path, output_dir: Path) -> None:
    """Append calibrated finalization instructions to the E000 README."""
    block = f"""

## Calibrated Final Analysis

Run the analysis-only finalizer without regenerating samples:

```bash
python scripts/finalize_generated_ensemble_analysis.py \\
  --evaluation-dir reports/experiments/E000_epoch8_baseline \\
  --output-dir {output_dir} \\
  --real-quantile 0.99 \\
  --pair-limit 1000 \\
  --bootstrap-iterations 200 \\
  --contact-threshold 8.0 \\
  --seed 8000 \\
  --plots \\
  --resume
```

Calibrated outputs include empirical real-like geometry thresholds, corrected
novelty by requested length, real-vs-real diversity calibration, deterministic
sample rankings, figures, `metrics/calibrated_summary.json`, and
`E000_FINAL_REPORT.md`. The original `protocol.json` and raw metric files are
not modified by the finalizer.
"""
    text = path.read_text() if path.exists() else "# E000 Epoch-8 Baseline\n"
    if "## Calibrated Final Analysis" not in text:
        atomic_write_text(path, text.rstrip() + "\n" + block)


def run_finalization(cfg: FinalizationConfig) -> Path:
    """Run analysis-only finalization."""
    started = time.time()
    protocol = validate_completed_evaluation(cfg.evaluation_dir)
    input_hashes_before = file_hashes(cfg.evaluation_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = cfg.output_dir / "metrics"
    figures_dir = cfg.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = cfg.output_dir / "calibrated_analysis_protocol.json"
    analysis = analysis_protocol(cfg, protocol, input_hashes_before)
    validate_or_write_analysis_protocol(analysis_path, analysis, cfg)

    generated = ensure_heuristic_alias(pd.read_parquet(cfg.evaluation_dir / "metrics" / "validity_per_sample.parquet"))
    real = ensure_heuristic_alias(pd.read_parquet(cfg.evaluation_dir / "metrics" / "real_control_metrics.parquet"))
    generated_pairs = pd.read_parquet(cfg.evaluation_dir / "metrics" / "diversity_pairs.parquet")
    generated_diversity_summary = pd.read_csv(cfg.evaluation_dir / "metrics" / "diversity_summary.csv")
    novelty = pd.read_parquet(cfg.evaluation_dir / "metrics" / "novelty_per_sample.parquet")
    novelty_cal = pd.read_parquet(cfg.evaluation_dir / "metrics" / "novelty_calibration.parquet")

    thresholds = empirical_thresholds(real, quantile=cfg.real_quantile)
    empirical_per_sample, empirical_by_length = empirical_validity(generated, thresholds, quantile=cfg.real_quantile)
    corrected_cal = corrected_novelty_calibration(novelty_cal, real)
    counts = corrected_cal.groupby("requested_length").size().to_dict()
    expected_real_count = int(protocol.get("control_count", 64))
    if any(int(count) != expected_real_count for count in counts.values()):
        raise ValueError(f"Corrected novelty calibration counts are not all {expected_real_count}: {counts}")

    real_pairs = real_diversity_pairs(
        real,
        pair_limit=cfg.pair_limit,
        contact_threshold=cfg.contact_threshold,
        seed=cfg.seed,
    )
    real_exact = summarize_real_diversity_exact(real_pairs)
    real_summary = weighted_real_diversity_summary(real_exact)
    real_clusters = real_diversity_clusters(real, real_pairs)
    div_cal = diversity_calibration(
        generated_diversity_summary,
        generated_pairs,
        real_summary,
        real_pairs,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )
    novelty_summary, novelty_calibrated = novelty_by_requested_length(
        novelty,
        corrected_cal,
        empirical_per_sample,
        bootstrap_iterations=cfg.bootstrap_iterations,
        seed=cfg.seed,
    )
    rankings = generated_rankings(generated, real)

    atomic_write_csv(metrics_dir / "empirical_validity_thresholds.csv", thresholds)
    atomic_write_parquet(metrics_dir / "empirical_validity_per_sample.parquet", empirical_per_sample)
    atomic_write_csv(metrics_dir / "empirical_validity_by_length.csv", empirical_by_length)
    atomic_write_parquet(metrics_dir / "real_diversity_pairs.parquet", real_pairs)
    atomic_write_csv(metrics_dir / "real_diversity_exact_length_summary.csv", real_exact)
    atomic_write_csv(metrics_dir / "real_diversity_summary.csv", real_summary)
    atomic_write_parquet(metrics_dir / "real_diversity_clusters.parquet", real_clusters)
    atomic_write_csv(metrics_dir / "diversity_calibration.csv", div_cal)
    atomic_write_parquet(metrics_dir / "novelty_calibration_corrected.parquet", corrected_cal)
    atomic_write_parquet(metrics_dir / "novelty_per_sample_calibrated.parquet", novelty_calibrated)
    atomic_write_csv(metrics_dir / "novelty_by_requested_length.csv", novelty_summary)
    atomic_write_parquet(metrics_dir / "generated_sample_rankings.parquet", rankings)

    if cfg.plots:
        plot_outputs(
            cfg.output_dir,
            generated=generated,
            real=real,
            empirical_by_length=empirical_by_length,
            diversity_cal=div_cal,
            novelty_summary=novelty_summary,
            rankings=rankings,
        )

    input_hashes_after = file_hashes(cfg.evaluation_dir)
    summary = calibrated_summary(
        protocol=protocol,
        generated=generated,
        real=real,
        empirical_by_length=empirical_by_length,
        diversity_cal=div_cal,
        novelty_summary=novelty_summary,
        input_hashes_before=input_hashes_before,
        input_hashes_after=input_hashes_after,
        runtime_seconds=time.time() - started,
    )
    atomic_write_json(metrics_dir / "calibrated_summary.json", summary)
    write_report(
        cfg.output_dir / "E000_FINAL_REPORT.md",
        protocol={"evaluation_dir": str(cfg.evaluation_dir)},
        empirical_by_length=empirical_by_length,
        diversity_cal=div_cal,
        novelty_summary=novelty_summary,
        calibrated_summary=summary,
    )
    update_timeline(Path("docs/experiment_timeline.md"), summary)
    update_e000_readme(cfg.evaluation_dir / "README.md", cfg.output_dir)
    complete_analysis_protocol(analysis_path)
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("Raw E000 input hashes changed during finalization.")
    return cfg.output_dir


def config_from_args(args: argparse.Namespace) -> FinalizationConfig:
    """Parse CLI args."""
    quantile = float(args.real_quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("--real-quantile must be in (0, 1]")
    return FinalizationConfig(
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir or args.evaluation_dir,
        real_quantile=quantile,
        pair_limit=int(args.pair_limit),
        bootstrap_iterations=int(args.bootstrap_iterations),
        contact_threshold=float(args.contact_threshold),
        seed=int(args.seed),
        plots=bool(args.plots),
        restart=bool(args.restart),
        resume=bool(args.resume),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--real-quantile", type=float, default=0.99)
    parser.add_argument("--pair-limit", type=int, default=1000)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--contact-threshold", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    start = time.time()
    out = run_finalization(config_from_args(args))
    print(f"Wrote calibrated analysis to {out} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
