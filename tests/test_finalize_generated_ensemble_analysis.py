"""Synthetic tests for analysis-only E000 finalization."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_finalizer_module():
    script = Path(__file__).parents[1] / "scripts" / "finalize_generated_ensemble_analysis.py"
    spec = importlib.util.spec_from_file_location("finalize_generated_ensemble_analysis", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["finalize_generated_ensemble_analysis"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


FINAL = _load_finalizer_module()
EVAL = FINAL.EVAL


def _write_npz(path: Path, matrix: np.ndarray, *, generated: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if generated:
        np.savez_compressed(path, physical_distance_matrix_angstrom=matrix.astype(np.float32))
    else:
        np.savez_compressed(path, distance_matrix=matrix.astype(np.float32))


def _line_matrix(length: int, *, scale: float = 3.8) -> np.ndarray:
    coords = np.stack([np.arange(length) * scale, np.zeros(length), np.zeros(length)], axis=1).astype(np.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def _descriptor_row(
    tmp_path: Path,
    *,
    sample_id: str,
    requested_length: int,
    actual_length: int,
    generated: bool,
    scale: float = 3.8,
) -> dict:
    matrix = _line_matrix(actual_length, scale=scale)
    path = tmp_path / ("generated" if generated else "real") / f"{sample_id}.npz"
    _write_npz(path, matrix, generated=generated)
    row = EVAL.descriptor_from_matrix(
        matrix,
        sample_id=sample_id,
        pdb_id=sample_id[:4].upper(),
        length=actual_length,
        seed=1,
        contact_threshold=8.0,
        num_triangles=16,
        source="generated" if generated else "real_control",
        path=str(path),
    )
    if not generated:
        row["requested_length"] = requested_length
    return row


def _minimal_completed_eval(tmp_path: Path) -> Path:
    root = tmp_path / "E000"
    metrics = root / "metrics"
    state = root / "state"
    metrics.mkdir(parents=True)
    state.mkdir()
    generated = pd.DataFrame(
        [
            _descriptor_row(
                tmp_path,
                sample_id="g64_a",
                requested_length=64,
                actual_length=4,
                generated=True,
                scale=3.4,
            ),
            _descriptor_row(
                tmp_path,
                sample_id="g64_b",
                requested_length=64,
                actual_length=4,
                generated=True,
                scale=3.5,
            ),
        ]
    )
    generated["length"] = 64
    real = pd.DataFrame(
        [
            _descriptor_row(tmp_path, sample_id="r64_a", requested_length=64, actual_length=4, generated=False),
            _descriptor_row(tmp_path, sample_id="r64_b", requested_length=64, actual_length=4, generated=False),
            _descriptor_row(tmp_path, sample_id="r64_c", requested_length=64, actual_length=5, generated=False),
            _descriptor_row(tmp_path, sample_id="r64_d", requested_length=64, actual_length=5, generated=False),
        ]
    )
    diversity_pairs = pd.DataFrame(
        [
            {
                "length": 64,
                "sample_id_a": "g64_a",
                "sample_id_b": "g64_b",
                "distance_map_rmse": 0.1,
                "contact_hamming_distance": 0.2,
                "contact_jaccard_distance": 0.3,
                "descriptor_distance": 0.4,
            }
        ]
    )
    diversity_summary = pd.DataFrame(
        [
            {
                "length": 64,
                "pair_count": 1,
                "distance_map_rmse_mean": 0.1,
                "contact_hamming_distance_mean": 0.2,
                "contact_jaccard_distance_mean": 0.3,
                "descriptor_distance_mean": 0.4,
            }
        ]
    )
    novelty = pd.DataFrame(
        [
            {
                "sample_id": "g64_a",
                "length": 64,
                "match_mode": "exact_length",
                "nearest_training_sample_id": "t1",
                "nearest_training_pdb_id": "T001",
                "descriptor_distance": 2.0,
                "refined_distance_map_rmse": 0.5,
                "refined_contact_jaccard_distance": 0.6,
                "approximate": True,
            },
            {
                "sample_id": "g64_b",
                "length": 64,
                "match_mode": "exact_length",
                "nearest_training_sample_id": "t1",
                "nearest_training_pdb_id": "T001",
                "descriptor_distance": 3.0,
                "refined_distance_map_rmse": 0.6,
                "refined_contact_jaccard_distance": 0.7,
                "approximate": True,
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "real_control_sample_id": sample_id,
                "length": actual_length,
                "match_mode": "exact_length",
                "nearest_training_sample_id": "t1",
                "nearest_training_pdb_id": "T001",
                "descriptor_distance": 1.0 + idx,
                "refined_distance_map_rmse": 0.2 + idx / 10,
                "refined_contact_jaccard_distance": 0.3 + idx / 10,
                "approximate": True,
            }
            for idx, (sample_id, actual_length) in enumerate([("r64_a", 4), ("r64_b", 4), ("r64_c", 5), ("r64_d", 5)])
        ]
    )
    generated.to_parquet(metrics / "validity_per_sample.parquet", index=False)
    real.to_parquet(metrics / "real_control_metrics.parquet", index=False)
    diversity_pairs.to_parquet(metrics / "diversity_pairs.parquet", index=False)
    diversity_summary.to_csv(metrics / "diversity_summary.csv", index=False)
    pd.DataFrame({"sample_id": ["g64_a", "g64_b"]}).to_parquet(metrics / "diversity_clusters.parquet", index=False)
    novelty.to_parquet(metrics / "novelty_per_sample.parquet", index=False)
    calibration.to_parquet(metrics / "novelty_calibration.parquet", index=False)
    pd.DataFrame({"descriptor": ["distance_mean"]}).to_csv(metrics / "distribution_matching.csv", index=False)
    (metrics / "summary.json").write_text(json.dumps({"generated_sample_count": 2}) + "\n")
    pd.DataFrame([real.iloc[0].to_dict()]).to_parquet(state / "training_descriptors.parquet", index=False)
    (state / "evaluation_state.sqlite").write_bytes(b"sqlite placeholder")
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "sample_counts_by_length": {"64": 2},
                "control_count": 4,
                "checkpoint_path": "checkpoint.pt",
                "checkpoint_sha256": "abc",
            }
        )
        + "\n"
    )
    return root


def test_corrected_requested_length_novelty_join_recovers_all_controls(tmp_path: Path) -> None:
    """Corrected calibration groups by requested_length rather than actual length."""
    root = _minimal_completed_eval(tmp_path)
    real = pd.read_parquet(root / "metrics" / "real_control_metrics.parquet")
    calibration = pd.read_parquet(root / "metrics" / "novelty_calibration.parquet")

    corrected = FINAL.corrected_novelty_calibration(calibration, real)

    assert corrected.groupby("requested_length").size().to_dict() == {64: 4}
    assert sorted(corrected["actual_length"].unique()) == [4, 5]


def test_real_pair_comparisons_restricted_to_identical_actual_lengths(tmp_path: Path) -> None:
    """Real diversity compares only same-actual-length control pairs."""
    root = _minimal_completed_eval(tmp_path)
    real = pd.read_parquet(root / "metrics" / "real_control_metrics.parquet")

    pairs = FINAL.real_diversity_pairs(real, pair_limit=100, contact_threshold=8.0, seed=1)

    assert len(pairs) == 2
    assert set(pairs["actual_length"]) == {4, 5}


def test_weighted_real_diversity_aggregation_by_pair_count() -> None:
    """Requested-length real diversity means are weighted by valid pair count."""
    exact = pd.DataFrame(
        [
            {
                "requested_length": 64,
                "actual_length": 63,
                "pair_count": 1,
                "distance_map_rmse_mean": 1.0,
                "contact_hamming_distance_mean": 1.0,
                "contact_jaccard_distance_mean": 1.0,
                "descriptor_distance_mean": 1.0,
            },
            {
                "requested_length": 64,
                "actual_length": 64,
                "pair_count": 3,
                "distance_map_rmse_mean": 3.0,
                "contact_hamming_distance_mean": 3.0,
                "contact_jaccard_distance_mean": 3.0,
                "descriptor_distance_mean": 3.0,
            },
        ]
    )

    summary = FINAL.weighted_real_diversity_summary(exact)

    assert summary.iloc[0]["distance_map_rmse_mean"] == pytest.approx(2.5)
    assert summary.iloc[0]["pair_count"] == 4


def test_deterministic_pair_sampling() -> None:
    """Pair subsampling is deterministic for the same seed."""
    assert FINAL.deterministic_pair_indices(10, limit=8, seed=9) == FINAL.deterministic_pair_indices(
        10, limit=8, seed=9
    )


def test_empirical_threshold_computation_and_heuristic_alias(tmp_path: Path) -> None:
    """Empirical thresholds and deprecated heuristic alias are computed explicitly."""
    root = _minimal_completed_eval(tmp_path)
    generated = pd.read_parquet(root / "metrics" / "validity_per_sample.parquet").drop(
        columns=["heuristic_edm_quality_pass"],
        errors="ignore",
    )
    real = pd.read_parquet(root / "metrics" / "real_control_metrics.parquet")

    thresholds = FINAL.empirical_thresholds(real, quantile=0.99)
    per_sample, by_length = FINAL.empirical_validity(generated, thresholds, quantile=0.99)

    assert set(thresholds["metric"]) == set(FINAL.EMPIRICAL_METRICS)
    assert "heuristic_edm_quality_pass" in per_sample
    assert "empirical_real_like_geometry_pass" in per_sample
    assert by_length.iloc[0]["generated_count"] == 2


def test_empty_empirical_pass_novelty_group_is_unavailable(tmp_path: Path) -> None:
    """Empty empirical-pass novelty subgroup is marked unavailable, not zero novelty."""
    root = _minimal_completed_eval(tmp_path)
    novelty = pd.read_parquet(root / "metrics" / "novelty_per_sample.parquet")
    real = pd.read_parquet(root / "metrics" / "real_control_metrics.parquet")
    calibration = FINAL.corrected_novelty_calibration(
        pd.read_parquet(root / "metrics" / "novelty_calibration.parquet"),
        real,
    )
    empirical = pd.DataFrame(
        {
            "sample_id": ["g64_a", "g64_b"],
            "heuristic_edm_quality_pass": [False, False],
            "empirical_real_like_geometry_pass": [False, False],
        }
    )

    summary, calibrated = FINAL.novelty_by_requested_length(
        novelty,
        calibration,
        empirical,
        bootstrap_iterations=2,
        seed=1,
    )
    empty = summary[summary["group"] == "empirical_real_like_geometry_pass"]

    assert set(empty["warning"]) == {"unavailable_empty_generated_group"}
    assert empty["generated_count"].sum() == 0
    assert "descriptor_distance_real_calibration_percentile" in calibrated


def test_rejects_incomplete_inputs(tmp_path: Path) -> None:
    """Finalization rejects partial or missing E000 inputs."""
    root = tmp_path / "partial"
    root.mkdir()
    (root / "protocol.json").write_text(json.dumps({"status": "partial"}) + "\n")

    with pytest.raises(FileNotFoundError):
        FINAL.validate_completed_evaluation(root)


def test_run_finalization_does_not_modify_raw_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Analysis writes calibrated outputs while raw input hashes stay unchanged."""
    root = _minimal_completed_eval(tmp_path)
    before = FINAL.file_hashes(root)
    cfg = FINAL.FinalizationConfig(
        evaluation_dir=root,
        output_dir=root / "calibrated",
        real_quantile=0.99,
        pair_limit=100,
        bootstrap_iterations=2,
        contact_threshold=8.0,
        seed=7,
        plots=False,
        restart=True,
        resume=True,
    )
    monkeypatch.setattr(FINAL, "update_timeline", lambda path, summary: None)
    monkeypatch.setattr(FINAL, "update_e000_readme", lambda path, output_dir: None)

    FINAL.run_finalization(cfg)
    after = FINAL.file_hashes(root)

    assert before == after
    assert (cfg.output_dir / "metrics" / "calibrated_summary.json").exists()
    assert (cfg.output_dir / "metrics" / "novelty_by_requested_length.csv").exists()
    assert (cfg.output_dir / "metrics" / "novelty_per_sample_calibrated.parquet").exists()
    summary = json.loads((cfg.output_dir / "metrics" / "calibrated_summary.json").read_text())
    assert summary["raw_inputs_unchanged"] is True
