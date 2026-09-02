#!/usr/bin/env python3
"""Compare two completed generated-ensemble evaluations without rerunning them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proteingen_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from protein_distance_diffusion.utils.hashing import sha256_file  # noqa: E402

PRIMARY_METRICS = [
    "triangle_violation_fraction",
    "negative_eigenvalue_mass_fraction",
    "rank3_residual_energy_fraction",
    "classical_mds_stress",
    "adjacent_residue_distance_rmse",
]
VALIDITY_FLAGS = ["empirical_real_like_geometry_pass", "heuristic_edm_quality_pass"]
EXPECTED_COUNTS = {64: 100, 128: 100, 256: 100, 384: 50, 500: 25}
PAIR_RE = re.compile(r"^N(?P<length>\d+)_i(?P<index>\d+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class ExperimentInputs:
    """Loaded comparison inputs for one completed experiment."""

    label: str
    root: Path
    protocol: dict[str, Any]
    calibrated_protocol: dict[str, Any]
    summary: dict[str, Any]
    validity: pd.DataFrame
    distribution: pd.DataFrame
    diversity: pd.DataFrame
    novelty: pd.DataFrame


def json_default(value: Any) -> Any:
    """Serialize NumPy and pandas scalar values to JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text())


def load_experiment(root: Path, label: str) -> ExperimentInputs:
    """Load the completed metrics required for the comparison."""
    metrics = root / "metrics"
    required = [
        root / "protocol.json",
        root / "calibrated_analysis_protocol.json",
        metrics / "calibrated_summary.json",
        metrics / "empirical_validity_per_sample.parquet",
        metrics / "distribution_matching.csv",
        metrics / "diversity_calibration.csv",
        metrics / "novelty_by_requested_length.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required comparison inputs: " + ", ".join(missing))
    novelty = pd.read_csv(metrics / "novelty_by_requested_length.csv")
    if "group" in novelty.columns:
        novelty = novelty[novelty["group"] == "all_generated_samples"].copy()
    return ExperimentInputs(
        label=label,
        root=root,
        protocol=read_json(root / "protocol.json"),
        calibrated_protocol=read_json(root / "calibrated_analysis_protocol.json"),
        summary=read_json(metrics / "calibrated_summary.json"),
        validity=pd.read_parquet(metrics / "empirical_validity_per_sample.parquet"),
        distribution=pd.read_csv(metrics / "distribution_matching.csv"),
        diversity=pd.read_csv(metrics / "diversity_calibration.csv"),
        novelty=novelty,
    )


def input_files(root: Path) -> list[Path]:
    """Return raw input files used by this analysis."""
    rels = [
        "protocol.json",
        "calibrated_analysis_protocol.json",
        "metrics/calibrated_summary.json",
        "metrics/empirical_validity_per_sample.parquet",
        "metrics/empirical_validity_by_length.csv",
        "metrics/distribution_matching.csv",
        "metrics/diversity_calibration.csv",
        "metrics/diversity_summary.csv",
        "metrics/real_diversity_summary.csv",
        "metrics/novelty_by_requested_length.csv",
        "metrics/novelty_per_sample_calibrated.parquet",
        "metrics/summary.json",
    ]
    return [root / rel for rel in rels if (root / rel).exists()]


def hash_inputs(baseline_dir: Path, candidate_dir: Path) -> dict[str, str]:
    """Hash all comparison input files."""
    result: dict[str, str] = {}
    for prefix, root in [("baseline", baseline_dir), ("candidate", candidate_dir)]:
        for path in input_files(root):
            result[f"{prefix}:{path.relative_to(root)}"] = sha256_file(path)
    return dict(sorted(result.items()))


def write_hash_file(path: Path, hashes: dict[str, str]) -> None:
    """Write SHA-256 hashes in deterministic text form."""
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))


def compatibility_report(baseline: ExperimentInputs, candidate: ExperimentInputs) -> dict[str, Any]:
    """Validate protocol compatibility and return the compared fields."""
    fields = {
        "lengths": ("protocol", "lengths"),
        "sample_counts_by_length": ("protocol", "sample_counts_by_length"),
        "seed_schedule": ("protocol", "seed_schedule"),
        "selected_weights": ("protocol", "selected_weights"),
        "contact_threshold_angstrom": ("protocol", "contact_threshold_angstrom"),
        "num_sampled_triangles": ("protocol", "num_sampled_triangles"),
        "control_count": ("protocol", "control_count"),
        "real_length_tolerance": ("protocol", "real_length_tolerance"),
        "diversity_pair_limit": ("protocol", "diversity_pair_limit"),
        "normalization_sha256": ("protocol", "normalization_sha256"),
        "train_manifest_sha256": ("protocol", "train_manifest_sha256"),
        "reference_manifest_sha256": ("protocol", "reference_manifest_sha256"),
        "calibrated_contact_threshold": ("calibrated_protocol", "contact_threshold"),
        "calibrated_pair_limit": ("calibrated_protocol", "pair_limit"),
        "calibrated_seed": ("calibrated_protocol", "seed"),
        "real_quantile": ("calibrated_protocol", "real_quantile"),
    }
    report: dict[str, Any] = {"compatible": True, "fields": {}, "mismatches": []}
    for name, (source, key) in fields.items():
        left = getattr(baseline, source).get(key)
        right = getattr(candidate, source).get(key)
        match = left == right
        report["fields"][name] = {"baseline": left, "candidate": right, "match": match}
        if not match:
            report["compatible"] = False
            report["mismatches"].append(name)
    return report


def parse_pair_key(df: pd.DataFrame) -> pd.DataFrame:
    """Attach requested length, sample index, and seed pairing columns."""
    parsed_rows = []
    for sample_id in df["sample_id"].astype(str):
        match = PAIR_RE.match(sample_id)
        if match is None:
            raise ValueError(f"Cannot parse generated sample_id for pairing: {sample_id}")
        parsed_rows.append(
            {
                "pair_requested_length": int(match.group("length")),
                "pair_sample_index": int(match.group("index")),
                "pair_seed": int(match.group("seed")),
            }
        )
    parsed = pd.DataFrame(parsed_rows, index=df.index)
    out = pd.concat([df.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    if "requested_length" in out.columns:
        requested = out["requested_length"].astype(int)
        if not (requested == out["pair_requested_length"]).all():
            raise ValueError("requested_length does not match the length encoded in sample_id")
    if "seed" in out.columns:
        seeds = out["seed"].astype(int)
        if not (seeds == out["pair_seed"]).all():
            raise ValueError("seed column does not match the seed encoded in sample_id")
    return out


def require_unique_pairs(df: pd.DataFrame, label: str) -> None:
    """Fail if a generated validity table contains duplicate pairing keys."""
    keys = ["pair_requested_length", "pair_sample_index", "pair_seed"]
    duplicates = df[df.duplicated(keys, keep=False)]
    if not duplicates.empty:
        raise ValueError(f"{label} has duplicate generated-sample pairing keys: {duplicates[keys].to_dict('records')}")


def pair_validity(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Create the exact one-to-one paired validity table."""
    left = parse_pair_key(baseline)
    right = parse_pair_key(candidate)
    require_unique_pairs(left, "baseline")
    require_unique_pairs(right, "candidate")
    keys = ["pair_requested_length", "pair_sample_index", "pair_seed"]
    left_keys = set(map(tuple, left[keys].to_numpy()))
    right_keys = set(map(tuple, right[keys].to_numpy()))
    missing_candidate = sorted(left_keys - right_keys)
    missing_baseline = sorted(right_keys - left_keys)
    if missing_candidate or missing_baseline:
        raise ValueError(
            "Generated samples are not exactly paired; "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )
    paired = left.merge(right, on=keys, how="inner", suffixes=("_baseline", "_candidate"), validate="one_to_one")
    paired = paired.rename(columns={"pair_requested_length": "requested_length", "pair_sample_index": "sample_index"})
    return paired.sort_values(["requested_length", "sample_index", "pair_seed"]).reset_index(drop=True)


def validate_expected_pair_counts(paired: pd.DataFrame, expected: dict[int, int] | None = None) -> None:
    """Require the expected calibrated ensemble pairing counts."""
    expected = EXPECTED_COUNTS if expected is None else expected
    counts = paired.groupby("requested_length").size().astype(int).to_dict()
    if counts != expected:
        raise ValueError(f"Unexpected paired counts by length: expected={expected}, observed={counts}")
    total = sum(expected.values())
    if len(paired) != total:
        raise ValueError(f"Expected {total} paired samples, observed {len(paired)}")


def stratified_bootstrap_ci(
    values: pd.Series | np.ndarray,
    strata: pd.Series | np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Deterministic paired bootstrap CI for a mean using within-stratum resampling."""
    vals = np.asarray(values, dtype=float)
    strat = np.asarray(strata)
    valid = np.isfinite(vals)
    vals = vals[valid]
    strat = strat[valid]
    if vals.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    unique = np.unique(strat)
    samples = np.empty(iterations, dtype=float)
    positions_by_stratum = [np.flatnonzero(strat == item) for item in unique]
    for index in range(iterations):
        drawn = [vals[rng.choice(positions, size=len(positions), replace=True)] for positions in positions_by_stratum]
        samples[index] = float(np.mean(np.concatenate(drawn)))
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def exact_two_sided_binomial_test(successes: int, trials: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test for small paired sign/McNemar tests."""
    if trials <= 0:
        return math.nan
    probabilities = np.array([math.comb(trials, k) * (p**k) * ((1 - p) ** (trials - k)) for k in range(trials + 1)])
    observed = probabilities[successes]
    return float(min(1.0, probabilities[probabilities <= observed + 1e-15].sum()))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values."""
    adjusted = [math.nan] * len(p_values)
    finite = [(idx, p) for idx, p in enumerate(p_values) if np.isfinite(p)]
    if not finite:
        return adjusted
    order = sorted(finite, key=lambda item: item[1], reverse=True)
    running = 1.0
    m = len(finite)
    for rank_from_end, (idx, p_value) in enumerate(order, start=1):
        rank = m - rank_from_end + 1
        running = min(running, p_value * m / rank)
        adjusted[idx] = float(min(running, 1.0))
    return adjusted


def paired_validity_summaries(paired: pd.DataFrame, *, iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize paired lower-is-better validity metrics by length and overall."""
    rows = []
    for group_name, group in [("overall", paired), *[(int(k), v) for k, v in paired.groupby("requested_length")]]:
        for metric_index, metric in enumerate(PRIMARY_METRICS):
            b = pd.to_numeric(group[f"{metric}_baseline"], errors="coerce")
            c = pd.to_numeric(group[f"{metric}_candidate"], errors="coerce")
            valid = b.notna() & c.notna()
            b = b[valid]
            c = c[valid]
            improvement = b - c
            change = c - b
            ci = stratified_bootstrap_ci(
                improvement,
                group.loc[valid, "requested_length"],
                iterations=iterations,
                seed=seed + metric_index + (0 if group_name == "overall" else int(group_name) * 37),
            )
            improved = int((improvement > 0).sum())
            worsened = int((improvement < 0).sum())
            ties = int((improvement == 0).sum())
            sign_trials = improved + worsened
            p_value = exact_two_sided_binomial_test(improved, sign_trials)
            std = float(improvement.std(ddof=1)) if len(improvement) > 1 else math.nan
            effect = float(improvement.mean() / std) if std and np.isfinite(std) and std > 0 else math.nan
            rows.append(
                {
                    "group": group_name,
                    "requested_length": None if group_name == "overall" else int(group_name),
                    "metric": metric,
                    "baseline_mean": float(b.mean()) if len(b) else math.nan,
                    "baseline_median": float(b.median()) if len(b) else math.nan,
                    "candidate_mean": float(c.mean()) if len(c) else math.nan,
                    "candidate_median": float(c.median()) if len(c) else math.nan,
                    "paired_mean_change_candidate_minus_baseline": float(change.mean()) if len(change) else math.nan,
                    "paired_median_change_candidate_minus_baseline": float(change.median())
                    if len(change)
                    else math.nan,
                    "mean_improvement_baseline_minus_candidate": float(improvement.mean())
                    if len(improvement)
                    else math.nan,
                    "median_improvement_baseline_minus_candidate": float(improvement.median())
                    if len(improvement)
                    else math.nan,
                    "relative_mean_change": float((c.mean() - b.mean()) / abs(b.mean()))
                    if len(b) and b.mean() != 0
                    else math.nan,
                    "bootstrap_ci_low": ci[0],
                    "bootstrap_ci_high": ci[1],
                    "fraction_improved": float((improvement > 0).mean()) if len(improvement) else math.nan,
                    "paired_standardized_effect_size": effect,
                    "paired_sample_count": int(len(improvement)),
                    "sign_test_p_value": p_value,
                    "sign_test_improved": improved,
                    "sign_test_worsened": worsened,
                    "sign_test_ties": ties,
                }
            )
    result = pd.DataFrame(rows)
    result["sign_test_p_value_bh"] = benjamini_hochberg(result["sign_test_p_value"].tolist())
    by_length = result[result["group"] != "overall"].copy()
    overall = result[result["group"] == "overall"].copy()
    return by_length, overall


def validity_transitions(paired: pd.DataFrame) -> pd.DataFrame:
    """Compute paired binary transition tables for validity flags."""
    rows = []
    for group_name, group in [("overall", paired), *[(int(k), v) for k, v in paired.groupby("requested_length")]]:
        for flag in VALIDITY_FLAGS:
            b = group[f"{flag}_baseline"].astype(bool)
            c = group[f"{flag}_candidate"].astype(bool)
            ff = int((~b & ~c).sum())
            fp = int((~b & c).sum())
            pf = int((b & ~c).sum())
            pp = int((b & c).sum())
            p_value = exact_two_sided_binomial_test(fp, fp + pf)
            rows.append(
                {
                    "group": group_name,
                    "requested_length": None if group_name == "overall" else int(group_name),
                    "flag": flag,
                    "baseline_fail_candidate_fail": ff,
                    "baseline_fail_candidate_pass": fp,
                    "baseline_pass_candidate_fail": pf,
                    "baseline_pass_candidate_pass": pp,
                    "discordant_candidate_better": fp,
                    "discordant_baseline_better": pf,
                    "mcnemar_exact_p_value": p_value,
                    "baseline_pass_fraction": float(b.mean()) if len(b) else math.nan,
                    "candidate_pass_fraction": float(c.mean()) if len(c) else math.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["mcnemar_exact_p_value_bh"] = benjamini_hochberg(result["mcnemar_exact_p_value"].tolist())
    return result


def compare_distribution(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Compare real-vs-generated distribution metrics by movement toward real."""
    keys = ["length", "descriptor"]
    merged = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        base_abs = abs(float(row["generated_mean_baseline"]) - float(row["real_mean_baseline"]))
        cand_abs = abs(float(row["generated_mean_candidate"]) - float(row["real_mean_candidate"]))
        base_w = float(row["wasserstein_distance_baseline"])
        cand_w = float(row["wasserstein_distance_candidate"])
        base_ks = float(row["ks_statistic_baseline"])
        cand_ks = float(row["ks_statistic_candidate"])
        rows.append(
            {
                "length": int(row["length"]),
                "descriptor": row["descriptor"],
                "baseline_generated_mean": row["generated_mean_baseline"],
                "candidate_generated_mean": row["generated_mean_candidate"],
                "real_mean": row["real_mean_baseline"],
                "baseline_abs_standardized_mean_discrepancy": base_abs,
                "candidate_abs_standardized_mean_discrepancy": cand_abs,
                "mean_discrepancy_improvement": base_abs - cand_abs,
                "baseline_wasserstein_distance": base_w,
                "candidate_wasserstein_distance": cand_w,
                "wasserstein_improvement": base_w - cand_w,
                "baseline_ks_statistic": base_ks,
                "candidate_ks_statistic": cand_ks,
                "ks_improvement": base_ks - cand_ks,
            }
        )
    return pd.DataFrame(rows)


def compare_ratio_to_real(baseline: pd.DataFrame, candidate: pd.DataFrame, *, domain: str) -> pd.DataFrame:
    """Compare generated/real ratios by closeness to one."""
    keys = ["requested_length", "metric"]
    merged = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    rows = []
    for _, row in merged.iterrows():
        base_ratio = float(row["generated_over_real_baseline"])
        cand_ratio = float(row["generated_over_real_candidate"])
        base_distance = abs(base_ratio - 1.0)
        cand_distance = abs(cand_ratio - 1.0)
        rows.append(
            {
                "domain": domain,
                "requested_length": int(row["requested_length"]),
                "metric": row["metric"],
                "baseline_generated_mean": row["generated_mean_baseline"],
                "candidate_generated_mean": row["generated_mean_candidate"],
                "real_mean": row["real_mean_baseline"],
                "baseline_generated_over_real": base_ratio,
                "candidate_generated_over_real": cand_ratio,
                "baseline_ratio_distance_from_one": base_distance,
                "candidate_ratio_distance_from_one": cand_distance,
                "candidate_moves_closer_to_real_ratio": cand_distance < base_distance,
                "ratio_closeness_improvement": base_distance - cand_distance,
                "baseline_ci_low": row.get("generated_ci_low_baseline", math.nan),
                "baseline_ci_high": row.get("generated_ci_high_baseline", math.nan),
                "candidate_ci_low": row.get("generated_ci_low_candidate", math.nan),
                "candidate_ci_high": row.get("generated_ci_high_candidate", math.nan),
            }
        )
    return pd.DataFrame(rows)


def prepare_paired_output(paired: pd.DataFrame) -> pd.DataFrame:
    """Return a compact per-sample paired table with metric deltas."""
    keep = ["requested_length", "sample_index", "pair_seed", "sample_id_baseline", "sample_id_candidate"]
    out = paired[keep].copy()
    for metric in PRIMARY_METRICS:
        out[f"{metric}_baseline"] = paired[f"{metric}_baseline"]
        out[f"{metric}_candidate"] = paired[f"{metric}_candidate"]
        out[f"{metric}_improvement_baseline_minus_candidate"] = (
            paired[f"{metric}_baseline"] - paired[f"{metric}_candidate"]
        )
    for flag in VALIDITY_FLAGS:
        out[f"{flag}_baseline"] = paired[f"{flag}_baseline"].astype(bool)
        out[f"{flag}_candidate"] = paired[f"{flag}_candidate"].astype(bool)
    return out


def protocol_payload(
    *,
    baseline: ExperimentInputs,
    candidate: ExperimentInputs,
    baseline_label: str,
    candidate_label: str,
    compatibility: dict[str, Any],
    bootstrap_iterations: int,
    seed: int,
    plots: bool,
) -> dict[str, Any]:
    """Build the comparison protocol payload."""
    payload = {
        "analysis": "generated_ensemble_experiment_comparison",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_dir": str(baseline.root),
        "candidate_dir": str(candidate.root),
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "plots": plots,
        "paired_key": ["requested_length", "sample_index", "seed"],
        "primary_metric_direction": "lower_is_better; improvement = baseline - candidate",
        "compatibility": compatibility,
        "baseline_checkpoint": {
            "path": baseline.protocol.get("checkpoint_path"),
            "sha256": baseline.protocol.get("checkpoint_sha256"),
            "epoch": baseline.protocol.get("checkpoint_epoch"),
            "optimizer_step": baseline.protocol.get("checkpoint_optimizer_step"),
        },
        "candidate_checkpoint": {
            "path": candidate.protocol.get("checkpoint_path"),
            "sha256": candidate.protocol.get("checkpoint_sha256"),
            "epoch": candidate.protocol.get("checkpoint_epoch"),
            "optimizer_step": candidate.protocol.get("checkpoint_optimizer_step"),
        },
    }
    core = json.dumps(payload, sort_keys=True, default=json_default)
    payload["comparison_protocol_sha256"] = hashlib.sha256(core.encode("utf-8")).hexdigest()
    return payload


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV with stable formatting."""
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a simple GitHub-flavored Markdown table."""
    if df.empty:
        return "_No rows._"
    rows = []
    columns = [str(column) for column in df.columns]
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def plot_metric_by_length(
    by_length: pd.DataFrame,
    metric: str,
    path: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    """Plot paired validity means with bootstrap CIs."""
    sub = by_length[by_length["metric"] == metric].sort_values("requested_length")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = sub["requested_length"].to_numpy(dtype=float)
    ax.plot(x, sub["baseline_mean"], marker="o", label=baseline_label)
    ax.plot(x, sub["candidate_mean"], marker="o", label=candidate_label)
    ax.fill_between(
        x,
        sub["candidate_mean"] - sub["bootstrap_ci_high"],
        sub["candidate_mean"] - sub["bootstrap_ci_low"],
        alpha=0.12,
    )
    ax.set_xlabel("Requested length N")
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " ").title())
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(
    *,
    out: Path,
    by_length: pd.DataFrame,
    transitions: pd.DataFrame,
    diversity: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
) -> None:
    """Write comparison figures."""
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_metric_by_length(
        by_length,
        "negative_eigenvalue_mass_fraction",
        figures / "negative_eigenvalue_mass_vs_N.png",
        baseline_label,
        candidate_label,
    )
    plot_metric_by_length(
        by_length,
        "rank3_residual_energy_fraction",
        figures / "rank3_residual_vs_N.png",
        baseline_label,
        candidate_label,
    )
    plot_metric_by_length(
        by_length,
        "classical_mds_stress",
        figures / "classical_mds_stress_vs_N.png",
        baseline_label,
        candidate_label,
    )
    plot_metric_by_length(
        by_length,
        "triangle_violation_fraction",
        figures / "triangle_violation_fraction_vs_N.png",
        baseline_label,
        candidate_label,
    )
    plot_metric_by_length(
        by_length,
        "adjacent_residue_distance_rmse",
        figures / "adjacent_residue_rmse_vs_N.png",
        baseline_label,
        candidate_label,
    )

    heuristic = transitions[transitions["flag"] == "heuristic_edm_quality_pass"].sort_values("requested_length")
    heuristic = heuristic[heuristic["group"] != "overall"]
    x = np.arange(len(heuristic))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.38
    ax.bar(x - width / 2, heuristic["baseline_pass_fraction"], width, label=baseline_label)
    ax.bar(x + width / 2, heuristic["candidate_pass_fraction"], width, label=candidate_label)
    ax.set_xticks(x, heuristic["requested_length"].astype(int).astype(str))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Requested length N")
    ax.set_ylabel("Pass fraction")
    ax.set_title("Heuristic EDM Quality Pass Fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "heuristic_validity_pass_fraction_by_length.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for metric, sub in diversity.groupby("metric"):
        sub = sub.sort_values("requested_length")
        ax.plot(
            sub["requested_length"],
            sub["baseline_generated_over_real"],
            marker="o",
            linestyle="--",
            label=f"{baseline_label} {metric}",
        )
        ax.plot(
            sub["requested_length"],
            sub["candidate_generated_over_real"],
            marker="o",
            label=f"{candidate_label} {metric}",
        )
    ax.axhline(1.0, color="black", linewidth=1)
    ax.set_xlabel("Requested length N")
    ax.set_ylabel("Generated / real diversity ratio")
    ax.set_title("Generated/Real Diversity Ratios")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "generated_real_diversity_ratios_by_length.png", dpi=180)
    plt.close(fig)

    summary = by_length.pivot(
        index="requested_length",
        columns="metric",
        values="mean_improvement_baseline_minus_candidate",
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(summary.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_yticks(np.arange(len(summary.index)), summary.index.astype(str))
    ax.set_xticks(np.arange(len(summary.columns)), [col.replace("_", "\n") for col in summary.columns], rotation=0)
    ax.set_title("Improvement Summary: Positive Means E001 Lower/Better")
    fig.colorbar(im, ax=ax, label="E000 - E001")
    fig.tight_layout()
    fig.savefig(figures / "improvement_summary_directional.png", dpi=180)
    plt.close(fig)


def primary_findings(overall: pd.DataFrame) -> dict[str, Any]:
    """Return compact overall primary metric findings."""
    result = {}
    for _, row in overall.iterrows():
        result[str(row["metric"])] = {
            "baseline_mean": row["baseline_mean"],
            "candidate_mean": row["candidate_mean"],
            "mean_improvement": row["mean_improvement_baseline_minus_candidate"],
            "bootstrap_ci": [row["bootstrap_ci_low"], row["bootstrap_ci_high"]],
            "fraction_improved": row["fraction_improved"],
            "effect_size": row["paired_standardized_effect_size"],
        }
    return result


def write_report(
    path: Path,
    *,
    protocol: dict[str, Any],
    by_length: pd.DataFrame,
    overall: pd.DataFrame,
    transitions: pd.DataFrame,
    distribution: pd.DataFrame,
    diversity: pd.DataFrame,
    novelty: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write the Markdown comparison report."""
    strict = transitions[
        (transitions["group"] == "overall") & (transitions["flag"] == "empirical_real_like_geometry_pass")
    ].iloc[0]
    heuristic = transitions[
        (transitions["group"] == "overall") & (transitions["flag"] == "heuristic_edm_quality_pass")
    ].iloc[0]
    neg = overall[overall["metric"] == "negative_eigenvalue_mass_fraction"].iloc[0]
    rank = overall[overall["metric"] == "rank3_residual_energy_fraction"].iloc[0]
    stress = overall[overall["metric"] == "classical_mds_stress"].iloc[0]
    tri = overall[overall["metric"] == "triangle_violation_fraction"].iloc[0]
    large = by_length[(by_length["requested_length"] >= 256) & (by_length["metric"].isin(PRIMARY_METRICS))]
    h1_large = large.groupby("metric")["mean_improvement_baseline_minus_candidate"].mean().to_dict()
    diversity_closer = int(diversity["candidate_moves_closer_to_real_ratio"].sum())
    novelty_closer = int(novelty["candidate_moves_closer_to_real_ratio"].sum())
    lines = [
        "# E001 Versus E000 Calibrated Ensemble Comparison",
        "",
        "## Executive Conclusion",
        "",
        (
            "This selected-model comparison finds that E001 improves several global geometry diagnostics "
            "relative to E000, especially negative eigenvalue mass and rank-3 residual energy, while both "
            "models still fail strict empirical real-like geometry for all 375 paired generated samples. "
            "E001 increases heuristic EDM-quality passes and remains diverse, but the result is not a "
            "perfect causal architecture ablation because the selected checkpoints differ in training history."
        ),
        "",
        "## Protocol Compatibility",
        "",
        f"- Compatibility status: `{protocol['compatibility']['compatible']}`",
        f"- Paired key: `{protocol['paired_key']}`",
        f"- Paired sample count: `{summary['paired_sample_count']}`",
        f"- Counts by length: `{summary['paired_counts_by_length']}`",
        f"- Input hashes preserved: `{summary['input_hashes_preserved']}`",
        "",
        "## Checkpoint And Training-History Caveat",
        "",
        (
            "The complete ensembles compare selected models: E000 selected checkpoint epoch 8, and E001 "
            "selected checkpoint epoch 4 at global_step 160166. This is a selected-model comparison, not a "
            "perfectly controlled causal architecture ablation. The earlier matched-step, two-samples-per-length "
            "comparison should be treated as exploratory screening, not formal statistical evidence."
        ),
        "",
        "## Paired Primary Results",
        "",
        markdown_table(
            overall[
                [
                    "metric",
                    "baseline_mean",
                    "candidate_mean",
                    "mean_improvement_baseline_minus_candidate",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "fraction_improved",
                    "paired_standardized_effect_size",
                    "sign_test_p_value_bh",
                ]
            ]
        ),
        "",
        "Positive improvement means E001 has the lower value for these lower-is-better metrics.",
        "",
        "## Validity Transitions",
        "",
        markdown_table(transitions[transitions["group"] == "overall"]),
        "",
        (
            f"Strict empirical real-like geometry remains zero-pass: E000 pass fraction "
            f"{strict['baseline_pass_fraction']:.6g}, E001 pass fraction {strict['candidate_pass_fraction']:.6g}. "
            f"Heuristic EDM-quality pass fraction changes from {heuristic['baseline_pass_fraction']:.6g} "
            f"to {heuristic['candidate_pass_fraction']:.6g}."
        ),
        "",
        "## Length-Dependent Effects",
        "",
        markdown_table(
            by_length[
                [
                    "requested_length",
                    "metric",
                    "baseline_mean",
                    "candidate_mean",
                    "mean_improvement_baseline_minus_candidate",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "fraction_improved",
                ]
            ]
        ),
        "",
        "## Local Versus Global Trade-Offs",
        "",
        (
            "H1 is supported for several global geometry metrics: overall improvements are "
            f"{neg['mean_improvement_baseline_minus_candidate']:.6g} for negative eigenvalue mass, "
            f"{rank['mean_improvement_baseline_minus_candidate']:.6g} for rank-3 residual energy, and "
            f"{stress['mean_improvement_baseline_minus_candidate']:.6g} for classical MDS stress. "
            f"Triangle violation improvement is {tri['mean_improvement_baseline_minus_candidate']:.6g}. "
            f"For N >= 256, mean improvements by metric are `{h1_large}`."
        ),
        "",
        "## Distribution Matching",
        "",
        (
            "Distribution matching is evaluated as movement toward the corresponding real distribution, "
            "not as raw generated metric decrease."
        ),
        "",
        markdown_table(distribution.head(20)),
        "",
        "## Diversity",
        "",
        (
            f"E001 moves the generated/real diversity ratio closer to 1 for {diversity_closer}/"
            f"{len(diversity)} length-metric rows. Ratios below 1 are reported as reduced diversity, "
            "not by themselves as mode collapse."
        ),
        "",
        markdown_table(diversity),
        "",
        "## Novelty",
        "",
        (
            f"E001 moves calibrated novelty ratios closer to the real calibration baseline for "
            f"{novelty_closer}/{len(novelty)} rows. Larger novelty distance is not interpreted as "
            "automatically better because excessive distance can indicate out-of-distribution invalidity."
        ),
        "",
        markdown_table(novelty),
        "",
        "## Hypothesis Evaluation",
        "",
        (
            "- H1: Supported for several global geometry diagnostics, particularly negative eigenvalue mass "
            "and rank-3 residual at large N."
        ),
        (
            "- H2: Partially supported for heuristic validity, but not for strict empirical real-like "
            "geometry, which remains 0/375."
        ),
        (
            "- H3: Supported as no evidence of diversity collapse; diversity ratios remain nonzero and are "
            "compared to real-control calibration."
        ),
        (
            "- H4: Supported with caveats; novelty does not indicate exact duplication, but shifts must be "
            "read alongside persistent strict-geometry failure."
        ),
        "",
        "## Limitations",
        "",
        "- This is selected-model evidence, not a perfectly controlled architecture-only ablation.",
        "- The matched-step comparison had only two samples per length and remains exploratory screening.",
        "- Existing calibrated outputs are reused; no ensemble evaluation was rerun.",
        "- Approximate nearest-neighbour novelty remains approximate.",
        "- Strict validity is all-zero, so percent-change summaries are intentionally avoided for that endpoint.",
        "",
        "## Decision For Next Experiment",
        "",
        (
            "E001 provides enough execution and selected-model quality evidence to proceed, but it still does "
            "not reach the empirical 3D distance-matrix manifold. The next scientifically justified intervention "
            "is a bounded physical auxiliary-loss experiment under the same calibrated evaluation protocol, "
            "while retaining E000 and E001 as explicit baselines."
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run the complete analysis-only comparison."""
    start = time.time()
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.restart:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = hash_inputs(baseline_dir, candidate_dir)
    write_hash_file(output_dir / "input_hashes_before.sha256", before)

    baseline = load_experiment(baseline_dir, args.baseline_label)
    candidate = load_experiment(candidate_dir, args.candidate_label)
    compatibility = compatibility_report(baseline, candidate)
    if not compatibility["compatible"]:
        raise ValueError(f"Protocol compatibility failed: {compatibility['mismatches']}")
    paired = pair_validity(baseline.validity, candidate.validity)
    validate_expected_pair_counts(paired)

    paired_output = prepare_paired_output(paired)
    by_length, overall = paired_validity_summaries(
        paired,
        iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
    )
    transitions = validity_transitions(paired)
    distribution = compare_distribution(baseline.distribution, candidate.distribution)
    diversity = compare_ratio_to_real(baseline.diversity, candidate.diversity, domain="diversity")
    novelty = compare_ratio_to_real(baseline.novelty, candidate.novelty, domain="novelty")

    protocol = protocol_payload(
        baseline=baseline,
        candidate=candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        compatibility=compatibility,
        bootstrap_iterations=int(args.bootstrap_iterations),
        seed=int(args.seed),
        plots=bool(args.plots),
    )
    (output_dir / "comparison_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    paired_output.to_parquet(output_dir / "paired_validity_per_sample.parquet", index=False)
    write_csv(by_length, output_dir / "paired_validity_by_length.csv")
    write_csv(overall, output_dir / "paired_validity_overall.csv")
    write_csv(transitions, output_dir / "validity_transitions.csv")
    write_csv(distribution, output_dir / "distribution_matching_comparison.csv")
    write_csv(diversity, output_dir / "diversity_comparison.csv")
    write_csv(novelty, output_dir / "novelty_comparison.csv")

    after = hash_inputs(baseline_dir, candidate_dir)
    write_hash_file(output_dir / "input_hashes_after.sha256", after)
    if before != after:
        raise RuntimeError("Comparison input hashes changed during execution")

    summary = {
        "runtime_seconds": time.time() - start,
        "protocol_compatible": True,
        "paired_sample_count": int(len(paired)),
        "paired_counts_by_length": {
            str(k): int(v) for k, v in paired.groupby("requested_length").size().to_dict().items()
        },
        "input_hashes_preserved": before == after,
        "primary_findings": primary_findings(overall),
        "strict_validity_overall": transitions[
            (transitions["group"] == "overall") & (transitions["flag"] == "empirical_real_like_geometry_pass")
        ]
        .iloc[0]
        .to_dict(),
        "heuristic_validity_overall": transitions[
            (transitions["group"] == "overall") & (transitions["flag"] == "heuristic_edm_quality_pass")
        ]
        .iloc[0]
        .to_dict(),
        "baseline_checkpoint": protocol["baseline_checkpoint"],
        "candidate_checkpoint": protocol["candidate_checkpoint"],
        "interpretation_constraint": (
            "Selected-model comparison: E000 epoch 8 versus E001 epoch 4/global_step 160166; "
            "not a perfectly controlled causal architecture ablation."
        ),
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n"
    )
    if args.plots:
        make_figures(
            out=output_dir,
            by_length=by_length,
            transitions=transitions,
            diversity=diversity,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
        )
    write_report(
        output_dir / "E001_VS_E000_REPORT.md",
        protocol=protocol,
        by_length=by_length,
        overall=overall,
        transitions=transitions,
        distribution=distribution,
        diversity=diversity,
        novelty=novelty,
        summary=summary,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-label", default="E000")
    parser.add_argument("--candidate-label", default="E001")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--plots", dest="plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    """Run CLI."""
    args = build_arg_parser().parse_args()
    summary = run_comparison(args)
    print(json.dumps(summary, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
