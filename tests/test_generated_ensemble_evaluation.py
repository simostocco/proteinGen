"""Synthetic tests for generated ensemble evaluation utilities."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from protein_distance_diffusion.data.preprocess import ProteinSample, compute_distance_matrix, save_processed_sample
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import save_checkpoint


def _load_eval_module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_generated_ensemble.py"
    spec = importlib.util.spec_from_file_location("evaluate_generated_ensemble", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_generated_ensemble"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EVAL = _load_eval_module()


def _distance_from_coords(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def _write_real_manifest(tmp_path: Path, name: str, lengths: list[int]) -> Path:
    rows = []
    for index, length in enumerate(lengths):
        coords = np.stack([np.arange(length), np.zeros(length) + index, np.zeros(length)], axis=1).astype(np.float32)
        sample = ProteinSample(
            sample_id=f"{name}_{index}",
            pdb_id=f"{name[:4].upper()}{index}",
            chain_id="A",
            sequence="A" * length,
            residue_ids=[str(i) for i in range(length)],
            ca_coordinates=coords,
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.0},
        )
        rows.append(save_processed_sample(sample, tmp_path / f"{name}_samples"))
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_generated_npz(tmp_path: Path, matrix: np.ndarray, *, sample_id: str, length: int, seed: int) -> Path:
    path = tmp_path / f"{sample_id}.npz"
    EVAL.atomic_write_npz(
        path,
        sample_id=np.asarray(sample_id),
        requested_length=np.asarray(length),
        sample_index=np.asarray(0),
        seed=np.asarray(seed),
        physical_distance_matrix_angstrom=matrix.astype(np.float32),
        protocol_sha256=np.asarray("protocol"),
    )
    return path


def _tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    train = _write_real_manifest(tmp_path, "train", [4, 5])
    ref = _write_real_manifest(tmp_path, "ref", [4, 5])
    norm = tmp_path / "normalization.json"
    norm.write_text(json.dumps({"mode": "scale", "scale": 1.0}) + "\n")
    config = {
        "diffusion_steps": 2,
        "prediction_parameterization": "epsilon",
        "model": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 4,
            "channel_multipliers": [1],
            "residual_blocks_per_level": 1,
            "attention_heads": 1,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 8,
        },
    }
    model = DistanceUNet(**config["model"])
    checkpoint = tmp_path / "tiny.pt"
    payload = {
        "model": model.state_dict(),
        "ema": model.state_dict(),
        "epoch": 0,
        "next_epoch": 1,
        "global_step": 0,
        "optimizer_step": 0,
        "config": config,
        "normalization": {"mode": "scale", "scale": 1.0},
    }
    save_checkpoint(checkpoint, payload)
    return checkpoint, norm, train, ref


def test_exact_3d_edm_has_no_material_negative_mass() -> None:
    """Distances from known 3D coordinates are EDM-compatible."""
    matrix = _distance_from_coords(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    diagnostics = EVAL.matrix_edm_diagnostics(matrix)

    assert diagnostics["negative_eigenvalue_mass_fraction"] < 1e-8
    assert diagnostics["materially_negative_eigenvalues"] == 0
    assert diagnostics["rank3_residual_energy_fraction"] == pytest.approx(0.0)
    assert diagnostics["classical_mds_stress"] < 1e-6


def test_corrupted_non_edm_and_triangle_violation_are_detected() -> None:
    """A hand-corrupted matrix violates triangle inequalities and EDM diagnostics."""
    matrix = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    triangle = EVAL.sampled_triangle_metrics(matrix, num_triangles=512, seed=7)
    diagnostics = EVAL.matrix_edm_diagnostics(matrix)

    assert triangle["triangle_violation_fraction"] > 0.0
    assert triangle["triangle_violation_max"] > 0.0
    assert diagnostics["negative_eigenvalue_mass_fraction"] > 0.0


def test_symmetry_and_diagonal_violations_are_reported() -> None:
    """Basic matrix validity retains symmetry and diagonal violations."""
    matrix = np.asarray([[2.0, 1.0, 4.0], [3.0, 0.0, 2.0], [4.0, 2.0, 0.0]], dtype=np.float32)
    row = EVAL.descriptor_from_matrix(
        matrix,
        sample_id="bad",
        length=3,
        seed=1,
        contact_threshold=8.0,
        num_triangles=64,
        source="generated",
    )

    assert row["max_abs_diagonal"] == pytest.approx(2.0)
    assert row["symmetry_error"] > 0.0
    assert row["numerically_valid"] is False


def test_triangle_sampling_is_deterministic() -> None:
    """Triangle subsampling is fixed by sample seed."""
    coords = np.stack([np.arange(12), np.zeros(12), np.zeros(12)], axis=1)
    matrix = _distance_from_coords(coords.astype(np.float32))

    left = EVAL.sampled_triangle_metrics(matrix, num_triangles=128, seed=99)
    right = EVAL.sampled_triangle_metrics(matrix, num_triangles=128, seed=99)

    assert left == right


def test_rank3_residual_detects_four_dimensional_energy() -> None:
    """A true 4D distance matrix has positive energy outside the top three dimensions."""
    coords = np.eye(5, 4, dtype=np.float32)
    matrix = _distance_from_coords(coords)
    diagnostics = EVAL.matrix_edm_diagnostics(matrix)

    assert diagnostics["energy_outside_top3_positive_fraction"] > 0.01
    assert diagnostics["rank3_residual_energy_fraction"] > 0.01


def test_descriptor_reproducibility() -> None:
    """Descriptor rows are deterministic for the same matrix and seed."""
    matrix = compute_distance_matrix(np.stack([np.arange(6), np.zeros(6), np.zeros(6)], axis=1).astype(np.float32))

    left = EVAL.descriptor_from_matrix(
        matrix,
        sample_id="x",
        length=6,
        seed=123,
        contact_threshold=8.0,
        num_triangles=256,
        source="generated",
    )
    right = EVAL.descriptor_from_matrix(
        matrix,
        sample_id="x",
        length=6,
        seed=123,
        contact_threshold=8.0,
        num_triangles=256,
        source="generated",
    )

    assert left == right


def test_diversity_duplicate_detection(tmp_path: Path) -> None:
    """Identical generated matrices land in the same near-duplicate cluster."""
    base = compute_distance_matrix(np.stack([np.arange(5), np.zeros(5), np.zeros(5)], axis=1).astype(np.float32))
    other = compute_distance_matrix(np.stack([np.arange(5) * 2.0, np.zeros(5), np.zeros(5)], axis=1).astype(np.float32))
    rows = []
    for sample_id, matrix in [("a", base), ("b", base), ("c", other)]:
        path = _write_generated_npz(tmp_path, matrix, sample_id=sample_id, length=5, seed=1)
        rows.append(
            EVAL.descriptor_from_matrix(
                matrix,
                sample_id=sample_id,
                length=5,
                seed=1,
                contact_threshold=8.0,
                num_triangles=32,
                source="generated",
                path=str(path),
            )
        )
    frame = pd.DataFrame(rows)
    pairs = EVAL.diversity_pairs_for_length(frame, contact_threshold=8.0, pair_limit=0, seed=1)
    clusters = EVAL.cluster_generated_samples(pairs, ["a", "b", "c"], rmse_threshold=0.001)

    cluster_map = dict(zip(clusters["sample_id"], clusters["diversity_cluster_id"], strict=True))
    assert cluster_map["a"] == cluster_map["b"]
    assert len(set(cluster_map.values())) == 2


def test_novelty_retrieval_is_deterministic(tmp_path: Path) -> None:
    """Approximate novelty retrieval is deterministic and tie-broken by sample ID."""
    matrix = compute_distance_matrix(np.stack([np.arange(4), np.zeros(4), np.zeros(4)], axis=1).astype(np.float32))
    generated_path = _write_generated_npz(tmp_path, matrix, sample_id="gen", length=4, seed=1)
    train_path = _write_generated_npz(tmp_path, matrix, sample_id="train", length=4, seed=1)
    generated = pd.DataFrame(
        [
            EVAL.descriptor_from_matrix(
                matrix,
                sample_id="gen",
                length=4,
                seed=1,
                contact_threshold=8.0,
                num_triangles=16,
                source="generated",
                path=str(generated_path),
            )
        ]
    )
    training = pd.DataFrame(
        [
            EVAL.descriptor_from_matrix(
                matrix,
                sample_id="train_b",
                pdb_id="BBBB",
                length=4,
                seed=1,
                contact_threshold=8.0,
                num_triangles=16,
                source="real",
                path=str(train_path),
            ),
            EVAL.descriptor_from_matrix(
                matrix,
                sample_id="train_a",
                pdb_id="AAAA",
                length=4,
                seed=1,
                contact_threshold=8.0,
                num_triangles=16,
                source="real",
                path=str(train_path),
            ),
        ]
    )

    left = EVAL.approximate_novelty(
        generated,
        training,
        candidate_count=2,
        length_tolerance=0,
        contact_threshold=8.0,
    )
    right = EVAL.approximate_novelty(
        generated,
        training,
        candidate_count=2,
        length_tolerance=0,
        contact_threshold=8.0,
    )

    pd.testing.assert_frame_equal(left, right)
    assert left.iloc[0]["nearest_training_sample_id"] == "train_a"


def test_real_control_selection_exact_then_tolerance(tmp_path: Path) -> None:
    """Real controls prefer exact length and then deterministic tolerance matches."""
    manifest_path = _write_real_manifest(tmp_path, "controls", [4, 4, 5, 7])
    manifest = pd.read_parquet(manifest_path)

    controls, report = EVAL.select_real_controls(manifest, length=4, count=3, tolerance=1, seed=11)

    assert len(controls) == 3
    assert report["exact_length_count"] == 2
    assert report["tolerance_matched_count"] == 1
    assert controls["length"].tolist().count(5) == 1


def test_per_length_sample_counts_build_distinct_seed_schedule() -> None:
    """The CLI-facing per-length count format supports the E000 seed bank."""
    counts = EVAL.parse_length_samples("64:2,128:1")
    assert counts == {64: 2, 128: 1}

    schedule = EVAL.seed_schedule((64, 128), counts, master_seed=8000)

    assert sorted(schedule) == [64, 128]
    assert len(schedule[64]) == 2
    assert len(schedule[128]) == 1
    assert schedule == EVAL.seed_schedule((64, 128), counts, master_seed=8000)


def test_protocol_samples_per_length_is_null_for_explicit_per_length_counts(tmp_path: Path) -> None:
    """Protocol avoids reporting an unused default when --length-samples is supplied."""
    cfg = EVAL.EvaluationConfig(
        checkpoint=tmp_path / "checkpoint.pt",
        weights="ema",
        config_path=None,
        normalization_file=tmp_path / "normalization.json",
        train_manifest=tmp_path / "train.parquet",
        reference_manifest=tmp_path / "validation.parquet",
        output_dir=tmp_path / "eval",
        lengths=(64,),
        samples_per_length=4,
        samples_by_length={64: 1},
        master_seed=8000,
        seeds=None,
        contact_threshold=8.0,
        num_triangles=16,
        novelty_candidate_count=1,
        workers=1,
        resume=True,
        restart=False,
        plots=False,
        control_count=1,
        real_length_tolerance=1,
        diversity_pair_limit=10,
        bootstrap_iterations=4,
    )
    schedule = EVAL.build_seed_schedule(cfg)
    protocol = EVAL.protocol_core(
        cfg,
        {
            "epoch": 7,
            "next_epoch": 8,
            "global_step": 10,
            "config": {"model": {}},
        },
        schedule,
    )

    assert protocol["sample_counts_by_length"] == {"64": 1}
    assert protocol["samples_per_length"] is None


def test_protocol_incompatibility_rejects_resume(tmp_path: Path) -> None:
    """Resume refuses to mix incompatible protocol settings."""
    path = tmp_path / "protocol.json"
    core = {"checkpoint_path": "a", "lengths": [4]}
    EVAL.write_protocol(path, core, status="partial", started_utc="start", completed_utc=None)

    with pytest.raises(ValueError, match="incompatible"):
        EVAL.validate_resume_protocol(path, {"checkpoint_path": "a", "lengths": [5]}, resume=True, restart=False)


def test_atomic_npz_validation_detects_corruption(tmp_path: Path) -> None:
    """Resume validation accepts complete outputs and rejects corrupt partial files."""
    matrix = np.zeros((4, 4), dtype=np.float32)
    path = EVAL.generated_sample_path(tmp_path, 4, 0, 99)
    EVAL.atomic_write_npz(
        path,
        requested_length=np.asarray(4),
        seed=np.asarray(99),
        physical_distance_matrix_angstrom=matrix,
        protocol_sha256=np.asarray("abc"),
    )
    assert EVAL.validate_generated_npz(path, length=4, seed=99, protocol_sha256="abc")
    path.write_bytes(b"not a valid npz")
    assert not EVAL.validate_generated_npz(path, length=4, seed=99, protocol_sha256="abc")


def test_tiny_smoke_evaluation_and_resume(tmp_path: Path) -> None:
    """A tiny synthetic checkpoint can generate, evaluate, and resume without rewriting completed samples."""
    checkpoint, norm, train, ref = _tiny_checkpoint(tmp_path)
    output = tmp_path / "eval"
    cfg = EVAL.EvaluationConfig(
        checkpoint=checkpoint,
        weights="ema",
        config_path=None,
        normalization_file=norm,
        train_manifest=train,
        reference_manifest=ref,
        output_dir=output,
        lengths=(4,),
        samples_per_length=1,
        samples_by_length=None,
        master_seed=3,
        seeds=None,
        contact_threshold=8.0,
        num_triangles=16,
        novelty_candidate_count=1,
        workers=1,
        resume=True,
        restart=True,
        plots=False,
        control_count=1,
        real_length_tolerance=1,
        diversity_pair_limit=10,
        bootstrap_iterations=4,
    )
    protocol = EVAL.run_evaluation(cfg)
    generated = sorted((output / "generated" / "N0004").glob("*.npz"))
    assert len(generated) == 1
    first_mtime = generated[0].stat().st_mtime_ns
    payload = json.loads(protocol.read_text())
    assert payload["status"] == "completed"
    assert (output / "metrics" / "validity_per_sample.parquet").exists()
    assert (output / "metrics" / "novelty_per_sample.parquet").exists()

    resumed_cfg = EVAL.EvaluationConfig(**{**cfg.__dict__, "restart": False})
    EVAL.run_evaluation(resumed_cfg)
    assert generated[0].stat().st_mtime_ns == first_mtime
    conn = sqlite3.connect(output / "state" / "evaluation_state.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM generated_samples WHERE status='completed'").fetchone()[0] == 1
    finally:
        conn.close()
