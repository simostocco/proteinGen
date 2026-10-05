"""Tests for Distance-AF dry-run benchmark helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from protein_distance_diffusion.evaluation.distance_af import (
    build_distance_af_command,
    eligible_pairs,
    evaluate_restraint_satisfaction,
    reject_malformed_prediction_table,
    require_sequence_for_cohort,
    restraints_to_frame,
    run_external_command,
    select_restraints,
    write_distance_af_restraints,
)
from protein_distance_diffusion.evaluation.repairability import pairwise_distances, sha256_file


def _matrix(length: int = 24) -> np.ndarray:
    coords = np.stack(
        [np.arange(length) * 3.8, np.sin(np.arange(length) / 2.0), np.cos(np.arange(length) / 3.0)],
        axis=1,
    )
    return pairwise_distances(coords)


def _load_prepare():
    path = Path("scripts/prepare_distance_af_benchmark.py")
    spec = importlib.util.spec_from_file_location("prepare_distance_af_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_primary_jobs_require_explicit_sequence_source() -> None:
    with pytest.raises(ValueError, match="sequence_provenance"):
        require_sequence_for_cohort("generated_e004", "AAAA", None)
    with pytest.raises(ValueError, match="generated or inverse-folded"):
        require_sequence_for_cohort("generated_e004", "AAAA", "native_pdb_sequence")
    require_sequence_for_cohort("generated_e004", "AAAA", "external_inverse_folding_sequence")


def test_eligible_pairs_are_upper_triangular_one_based_and_not_near_diagonal() -> None:
    pairs = eligible_pairs(_matrix(8), min_sequence_separation=3)
    assert (pairs["i_zero_based"] < pairs["j_zero_based"]).all()
    assert (pairs["j_zero_based"] - pairs["i_zero_based"] >= 3).all()
    assert (pairs["i_distance_af"] == pairs["i_zero_based"] + 1).all()
    assert (pairs["j_distance_af"] == pairs["j_zero_based"] + 1).all()


def test_restraint_selection_is_deterministic_disjoint_and_stratified() -> None:
    matrix = _matrix(32)
    left = restraints_to_frame(
        select_restraints(
            matrix,
            sample_id="s",
            matrix_source="real_native",
            strategy="contact_enriched",
            count=8,
            heldout_multiplier=1,
            min_sequence_separation=3,
            seed=11,
        )
    )
    right = restraints_to_frame(
        select_restraints(
            matrix,
            sample_id="s",
            matrix_source="real_native",
            strategy="contact_enriched",
            count=8,
            heldout_multiplier=1,
            min_sequence_separation=3,
            seed=11,
        )
    )
    pd.testing.assert_frame_equal(left, right)
    guidance_frame = left[left["split"] == "guidance"]
    heldout_frame = left[left["split"] == "heldout"]
    guidance = set(zip(guidance_frame.i_zero_based, guidance_frame.j_zero_based, strict=True))
    heldout = set(zip(heldout_frame.i_zero_based, heldout_frame.j_zero_based, strict=True))
    assert guidance.isdisjoint(heldout)
    assert left["target_distance_bin"].nunique() >= 2


def test_projection_consistent_prefers_small_projection_residuals() -> None:
    matrix = _matrix(20)
    selected = restraints_to_frame(
        select_restraints(
            matrix,
            sample_id="s",
            matrix_source="real_native",
            strategy="projection_consistent",
            count=4,
            seed=5,
        )
    )
    assert selected["projection_residual_angstrom"].max() < 1e-8


def test_distance_af_restraint_file_uses_one_based_csv(tmp_path: Path) -> None:
    restraints = restraints_to_frame(
        select_restraints(
            _matrix(16),
            sample_id="s",
            matrix_source="real_native",
            strategy="uniform_long_range",
            count=3,
            seed=1,
        )
    )
    path = tmp_path / "restraints.txt"
    write_distance_af_restraints(path, restraints)
    line = path.read_text().splitlines()[0]
    i, j, dist = line.split(",")
    assert int(i) >= 1
    assert int(j) > int(i)
    assert float(dist) > 0.0


def test_command_builder_and_dry_run_do_not_use_shell() -> None:
    command = build_distance_af_command(
        distance_af_python="python",
        distance_af_root=Path("/external/Distance-AF"),
        target_file=Path("target.fasta"),
        dist_info=Path("dist.txt"),
        fasta_file=Path("target.fasta"),
        output_dir=Path("out"),
    )
    assert command[0] == "python"
    assert "Distance_AF.py" in command[1]
    result = run_external_command(command, execute=False)
    assert result["status"] == "dry_run"
    assert result["return_code"] is None


def test_malformed_prediction_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        reject_malformed_prediction_table(pd.DataFrame([{"job_id": "a"}]))


def test_restraint_satisfaction_metrics() -> None:
    target = _matrix(16)
    predicted = target.copy()
    restraints = restraints_to_frame(
        select_restraints(
            target,
            sample_id="s",
            matrix_source="real_native",
            strategy="uniform_long_range",
            count=4,
            seed=2,
        )
    )
    metrics = evaluate_restraint_satisfaction(target, predicted, restraints, split="guidance")
    assert metrics["count"] == 4
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["fraction_within_1A"] == pytest.approx(1.0)


def test_prepare_benchmark_synthetic_smoke_blocks_generated_and_preserves_inputs(tmp_path: Path) -> None:
    module = _load_prepare()
    config = {
        "output_root": str(tmp_path / "distance_af"),
        "synthetic_smoke": True,
        "execute_distance_af": False,
        "restraint_budgets": [4],
        "restraint_strategies": ["uniform_long_range"],
        "min_sequence_separation": 4,
        "seed": 3,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    protocol_path = module.prepare_distance_af_benchmark(config_path)
    protocol = json.loads(protocol_path.read_text())
    assert protocol["status"] == "dry_run_prepared"
    manifest = pd.read_parquet(tmp_path / "distance_af" / "job_manifest.parquet")
    assert set(manifest["status"]) == {"dry_run_ready", "blocked_pending_sequence_model"}
    for path, digest in protocol["raw_input_hashes_before"].items():
        assert sha256_file(path) == digest


def test_prepare_rejects_sequence_matrix_length_mismatch(tmp_path: Path) -> None:
    module = _load_prepare()
    matrix_path = tmp_path / "matrix.npz"
    with matrix_path.open("wb") as handle:
        np.savez(handle, distance_matrix=_matrix(6))
    sample_manifest = tmp_path / "samples.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "bad",
                "cohort": "real_native",
                "matrix_path": str(matrix_path),
                "sequence": "AAA",
                "sequence_provenance": "native_pdb_sequence",
            }
        ]
    ).to_parquet(sample_manifest, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "out"),
                "sample_manifest": str(sample_manifest),
                "execute_distance_af": False,
                "restraint_budgets": [2],
            }
        )
    )
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        module.prepare_distance_af_benchmark(config_path)
