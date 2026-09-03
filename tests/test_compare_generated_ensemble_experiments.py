"""Tests for completed-ensemble experiment comparison."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_generated_ensemble_experiments",
    ROOT / "scripts" / "compare_generated_ensemble_experiments.py",
)
assert SPEC is not None
assert SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _protocol(counts: dict[int, int], *, checkpoint_epoch: int = 0, checkpoint_optimizer_step: int = 1) -> dict:
    seeds = {str(length): [1000 + length + index for index in range(count)] for length, count in sorted(counts.items())}
    return {
        "lengths": sorted(counts),
        "sample_counts_by_length": {str(k): v for k, v in sorted(counts.items())},
        "samples_per_length": None,
        "seed_schedule": seeds,
        "selected_weights": "ema",
        "contact_threshold_angstrom": 8.0,
        "num_sampled_triangles": 2048,
        "control_count": 4,
        "real_length_tolerance": 8,
        "diversity_pair_limit": 1000,
        "normalization_sha256": "norm",
        "train_manifest_sha256": "train",
        "reference_manifest_sha256": "ref",
        "checkpoint_path": "checkpoint.pt",
        "checkpoint_sha256": "ckpt",
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_optimizer_step": checkpoint_optimizer_step,
        "status": "completed",
    }


def _calibrated_protocol(counts: dict[int, int]) -> dict:
    return {
        "contact_threshold": 8.0,
        "pair_limit": 1000,
        "seed": 8000,
        "real_quantile": 0.99,
        "source_sample_counts_by_length": {str(k): v for k, v in sorted(counts.items())},
        "status": "completed",
    }


def _validity_rows(counts: dict[int, int], *, candidate: bool = False) -> list[dict]:
    rows = []
    for length, count in sorted(counts.items()):
        for index in range(count):
            seed = 1000 + length + index
            factor = 0.7 if candidate else 1.0
            rows.append(
                {
                    "sample_id": f"N{length:04d}_i{index:05d}_seed{seed}",
                    "requested_length": length,
                    "length": length,
                    "seed": seed,
                    "heuristic_edm_quality_pass": candidate and index == 0,
                    "edm_compatible": candidate and index == 0,
                    "triangle_violation_fraction": factor * (0.10 + index * 0.01),
                    "negative_eigenvalue_mass_fraction": factor * (0.20 + index * 0.01),
                    "rank3_residual_energy_fraction": factor * (0.30 + index * 0.01),
                    "classical_mds_stress": factor * (0.40 + index * 0.01),
                    "adjacent_residue_distance_rmse": factor * (0.50 + index * 0.01),
                    "empirical_real_like_geometry_pass": False,
                }
            )
    return rows


def _write_experiment(
    root: Path,
    counts: dict[int, int],
    *,
    candidate: bool = False,
    checkpoint_epoch: int = 0,
    checkpoint_optimizer_step: int = 1,
) -> None:
    metrics = root / "metrics"
    metrics.mkdir(parents=True)
    (root / "protocol.json").write_text(
        json.dumps(
            _protocol(
                counts,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_optimizer_step=checkpoint_optimizer_step,
            )
        )
    )
    (root / "calibrated_analysis_protocol.json").write_text(json.dumps(_calibrated_protocol(counts)))
    (metrics / "calibrated_summary.json").write_text(
        json.dumps(
            {
                "generated_sample_count": sum(counts.values()),
                "real_control_count": 4,
                "sample_counts_by_length": {str(k): v for k, v in sorted(counts.items())},
                "source": {"checkpoint_sha256": "ckpt"},
                "raw_inputs_unchanged": True,
            }
        )
    )
    (metrics / "summary.json").write_text(json.dumps({"generated_sample_count": sum(counts.values())}))
    pd.DataFrame(_validity_rows(counts, candidate=candidate)).to_parquet(
        metrics / "empirical_validity_per_sample.parquet",
        index=False,
    )
    pd.DataFrame({"requested_length": list(counts), "generated_count": list(counts.values())}).to_csv(
        metrics / "empirical_validity_by_length.csv",
        index=False,
    )
    distribution = []
    for length in counts:
        distribution.append(
            {
                "length": length,
                "descriptor": "contact_fraction_8A",
                "generated_count": counts[length],
                "real_count": 4,
                "generated_mean": 0.4 if candidate else 0.2,
                "real_mean": 0.5,
                "generated_std": 0.1,
                "real_std": 0.1,
                "wasserstein_distance": 0.1 if candidate else 0.3,
                "ks_statistic": 0.2 if candidate else 0.4,
            }
        )
    pd.DataFrame(distribution).to_csv(metrics / "distribution_matching.csv", index=False)
    diversity = []
    novelty = []
    for length in counts:
        for metric in [
            "distance_map_rmse",
            "contact_hamming_distance",
            "contact_jaccard_distance",
            "descriptor_distance",
        ]:
            diversity.append(
                {
                    "requested_length": length,
                    "metric": metric,
                    "generated_mean": 0.9 if candidate else 0.5,
                    "generated_ci_low": 0.8 if candidate else 0.4,
                    "generated_ci_high": 1.0 if candidate else 0.6,
                    "real_mean": 1.0,
                    "real_ci_low": 0.9,
                    "real_ci_high": 1.1,
                    "generated_over_real": 0.9 if candidate else 0.5,
                    "generated_pair_count": 2,
                    "real_pair_count": 2,
                }
            )
        for metric in ["descriptor_distance", "refined_distance_map_rmse", "refined_contact_jaccard_distance"]:
            novelty.append(
                {
                    "group": "all_generated_samples",
                    "requested_length": length,
                    "metric": metric,
                    "generated_count": counts[length],
                    "real_calibration_count": 4,
                    "generated_mean": 1.1 if candidate else 1.4,
                    "generated_ci_low": 1.0 if candidate else 1.3,
                    "generated_ci_high": 1.2 if candidate else 1.5,
                    "real_mean": 1.0,
                    "generated_over_real": 1.1 if candidate else 1.4,
                }
            )
    pd.DataFrame(diversity).to_csv(metrics / "diversity_calibration.csv", index=False)
    pd.DataFrame(diversity).to_csv(metrics / "diversity_summary.csv", index=False)
    pd.DataFrame(diversity).to_csv(metrics / "real_diversity_summary.csv", index=False)
    pd.DataFrame(novelty).to_csv(metrics / "novelty_by_requested_length.csv", index=False)
    pd.DataFrame({"sample_id": ["dummy"]}).to_parquet(metrics / "novelty_per_sample_calibrated.parquet", index=False)


def _load_pair(tmp_path: Path) -> tuple[compare.ExperimentInputs, compare.ExperimentInputs]:
    counts = {64: 2, 128: 1}
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_experiment(left, counts)
    _write_experiment(right, counts, candidate=True)
    return compare.load_experiment(left, "E000"), compare.load_experiment(right, "E001")


def test_protocol_compatibility_validation(tmp_path: Path) -> None:
    """Compatibility validation catches mismatched protocol fields."""
    baseline, candidate = _load_pair(tmp_path)
    report = compare.compatibility_report(baseline, candidate)
    assert report["compatible"] is True
    candidate.protocol["selected_weights"] = "model"
    report = compare.compatibility_report(baseline, candidate)
    assert report["compatible"] is False
    assert "selected_weights" in report["mismatches"]


def test_exact_pair_matching_and_duplicate_missing_rejection(tmp_path: Path) -> None:
    """Pairing requires exact unique requested-length/sample-index/seed keys."""
    baseline, candidate = _load_pair(tmp_path)
    paired = compare.pair_validity(baseline.validity, candidate.validity)
    assert len(paired) == 3
    duplicated = pd.concat([candidate.validity, candidate.validity.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        compare.pair_validity(baseline.validity, duplicated)
    missing = candidate.validity.iloc[:-1]
    with pytest.raises(ValueError, match="not exactly paired"):
        compare.pair_validity(baseline.validity, missing)


def test_improvement_sign_bootstrap_and_all_zero_strict_validity(tmp_path: Path) -> None:
    """Positive improvement means baseline minus candidate for lower-is-better metrics."""
    baseline, candidate = _load_pair(tmp_path)
    paired = compare.pair_validity(baseline.validity, candidate.validity)
    by_length, overall = compare.paired_validity_summaries(paired, iterations=100, seed=8000)
    row = overall[overall["metric"] == "negative_eigenvalue_mass_fraction"].iloc[0]
    assert row["mean_improvement_baseline_minus_candidate"] > 0
    assert np.isfinite(row["bootstrap_ci_low"])
    again = compare.paired_validity_summaries(paired, iterations=100, seed=8000)[1]
    pd.testing.assert_frame_equal(overall.reset_index(drop=True), again.reset_index(drop=True))
    transitions = compare.validity_transitions(paired)
    strict = transitions[
        (transitions["group"] == "overall") & (transitions["flag"] == "empirical_real_like_geometry_pass")
    ].iloc[0]
    assert strict["baseline_pass_fraction"] == 0
    assert strict["candidate_pass_fraction"] == 0


def test_nan_and_empty_subgroup_handling() -> None:
    """Bootstrap and summaries return NaN for empty numeric groups without crashing."""
    ci = compare.stratified_bootstrap_ci([np.nan], [64], iterations=10, seed=1)
    assert np.isnan(ci[0])
    assert np.isnan(ci[1])


def test_diversity_ratio_closeness_to_one_logic(tmp_path: Path) -> None:
    """Ratio comparison is based on distance from one."""
    baseline, candidate = _load_pair(tmp_path)
    rows = compare.compare_ratio_to_real(baseline.diversity, candidate.diversity, domain="diversity")
    assert rows["candidate_moves_closer_to_real_ratio"].all()
    assert (rows["ratio_closeness_improvement"] > 0).all()


def test_input_hash_preservation_report_and_figure_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full synthetic comparison writes hashes, tables, report, and figures."""
    counts = {64: 2, 128: 1}
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    output_dir = tmp_path / "out"
    _write_experiment(baseline_dir, counts)
    _write_experiment(candidate_dir, counts, candidate=True)
    monkeypatch.setattr(compare, "EXPECTED_COUNTS", counts)
    summary = compare.run_comparison(
        argparse.Namespace(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            baseline_label="E000",
            candidate_label="E001",
            bootstrap_iterations=100,
            seed=8000,
            plots=True,
            restart=True,
            resume=True,
        )
    )
    assert summary["input_hashes_preserved"] is True
    assert summary["paired_sample_count"] == 3
    assert (output_dir / "input_hashes_before.sha256").read_text() == (
        output_dir / "input_hashes_after.sha256"
    ).read_text()
    assert (output_dir / "paired_validity_per_sample.parquet").exists()
    assert (output_dir / "E001_VS_E000_REPORT.md").exists()
    assert len(list((output_dir / "figures").glob("*.png"))) == 8


def test_arbitrary_labels_drive_selected_model_interpretation_and_report_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison caveats and report filenames use CLI labels and checkpoint metadata."""
    counts = {64: 2, 128: 1}
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    output_dir = tmp_path / "out"
    _write_experiment(baseline_dir, counts, checkpoint_epoch=3, checkpoint_optimizer_step=160166)
    _write_experiment(
        candidate_dir,
        counts,
        candidate=True,
        checkpoint_epoch=4,
        checkpoint_optimizer_step=169181,
    )
    monkeypatch.setattr(compare, "EXPECTED_COUNTS", counts)

    summary = compare.run_comparison(
        argparse.Namespace(
            baseline_dir=baseline_dir,
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            baseline_label="E001",
            candidate_label="E002",
            bootstrap_iterations=100,
            seed=8000,
            plots=False,
            restart=True,
            resume=True,
        )
    )

    expected = (
        "Selected-model comparison: E001 epoch 4/global step 160166 versus "
        "E002 epoch 5/global step 169181. Selected epochs, optimizer steps, and training histories differ, "
        "so this is not a perfectly controlled causal auxiliary-loss ablation."
    )
    assert summary["interpretation_constraint"] == expected
    report = output_dir / "E002_VS_E001_REPORT.md"
    assert report.exists()
    text = report.read_text()
    assert expected in text
    assert "E000 epoch 8" not in text
    assert "E001 epoch 4 at global_step 160166" not in text
