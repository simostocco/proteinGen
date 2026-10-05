"""Synthetic tests for E007 distance-matrix quality diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from protein_distance_diffusion.evaluation.distance_matrix_quality import (
    MatrixQualityConfig,
    assess_distance_matrices,
    assess_distance_matrix,
)


def _distances(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(delta, axis=-1)


def test_valid_rank3_euclidean_matrix() -> None:
    coordinates = np.array([[0, 0, 0], [3.8, 0, 0], [3.8, 3.8, 0], [0, 3.8, 1.0]], dtype=float)
    report = assess_distance_matrix(_distances(coordinates), candidate_id="valid")
    assert report["candidate_id"] == "valid"
    assert report["finite"]
    assert report["euclidean_gram_valid"]
    assert report["negative_distance_count"] == 0
    assert report["rank3_reconstruction_rmse_angstrom"] == pytest.approx(0.0, abs=1e-7)
    assert report["triangle_calculation"] == "exact"


def test_asymmetry_diagonal_and_negative_values_are_reported() -> None:
    matrix = np.array([[1.0, -2.0, 3.0], [2.5, 0.0, 1.0], [3.0, 1.0, 0.0]])
    report = assess_distance_matrix(matrix)
    assert report["symmetry_error_max_angstrom"] == pytest.approx(4.5)
    assert report["diagonal_error_max_angstrom"] == pytest.approx(1.0)
    assert report["negative_distance_count"] == 1


def test_non_euclidean_matrix_has_negative_gram_mass() -> None:
    matrix = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 3.0], [1.0, 3.0, 0.0]])
    report = assess_distance_matrix(matrix)
    assert not report["euclidean_gram_valid"]
    assert report["negative_eigenmass_fraction"] > 0.0
    assert report["triangle_violation_fraction"] > 0.0


def test_padding_invariance_and_pair_mask() -> None:
    core = _distances(np.array([[0, 0, 0], [3.8, 0, 0], [3.8, 3.8, 0]], dtype=float))
    padded = np.full((8, 8), np.nan)
    padded[:3, :3] = core
    mask = np.zeros((8, 8), dtype=bool)
    mask[:3, :3] = True
    core_report = assess_distance_matrix(core)
    padded_report = assess_distance_matrix(padded, pair_mask=mask, sequence_length=3)
    for key in (
        "finite",
        "symmetry_error_max_angstrom",
        "triangle_violation_fraction",
        "negative_eigenmass_fraction",
        "rank3_reconstruction_rmse_angstrom",
    ):
        assert padded_report[key] == pytest.approx(core_report[key])
    bad_mask = mask.copy()
    bad_mask[0, 1] = False
    with pytest.raises(ValueError, match="symmetric"):
        assess_distance_matrix(padded, pair_mask=bad_mask)


def test_triangle_sampling_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    matrix = _distances(rng.normal(size=(20, 3)))
    matrix[0, 1] = matrix[1, 0] = 100.0
    config = MatrixQualityConfig(triangle_exact_max_length=3, triangle_sample_count=50, triangle_seed=91)
    first = assess_distance_matrix(matrix, config=config)
    second = assess_distance_matrix(matrix, config=config)
    assert first["triangle_calculation"] == "sampled_without_replacement"
    assert first["triangle_triplet_count"] == 50
    assert first["triangle_violation_fraction"] == second["triangle_violation_fraction"]


@pytest.mark.parametrize("length", [0, 1, 2])
def test_very_short_sequences(length: int) -> None:
    report = assess_distance_matrix(np.zeros((length, length)))
    assert report["triangle_calculation"] == "not_applicable"
    assert report["triangle_violation_fraction"] == 0.0
    assert report["rank3_reconstruction_rmse_angstrom"] == 0.0


def test_input_is_not_mutated_and_candidates_are_not_averaged() -> None:
    matrices = np.stack([np.zeros((3, 3)), np.full((3, 3), 2.0)])
    np.fill_diagonal(matrices[1], 0.0)
    original = matrices.copy()
    reports = assess_distance_matrices(matrices, candidate_ids=["a", "b"])
    assert np.array_equal(matrices, original)
    assert [row["candidate_id"] for row in reports] == ["a", "b"]
    assert all(row["candidate_count"] == 1 and row["candidate_aggregation"] == "none" for row in reports)
    assert (
        reports[0]["adjacent_residue_distance_mean_angstrom"] != reports[1]["adjacent_residue_distance_mean_angstrom"]
    )


def test_malformed_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="square"):
        assess_distance_matrix(np.zeros((2, 3)))
    report = assess_distance_matrix(np.array([[0.0, np.inf], [np.inf, 0.0]]))
    assert not report["finite"]
    assert not report["metrics_available"]
    assert report["edm_calculation"] == "unavailable"


def test_bounded_eigenvalue_mode() -> None:
    matrix = _distances(np.arange(30, dtype=float)[:, None])
    config = MatrixQualityConfig(eigen_exact_max_length=10, eigen_sample_size=8, eigen_seed=3)
    report = assess_distance_matrix(matrix, config=config)
    assert report["edm_calculation"] == "deterministic_principal_submatrix"
    assert report["edm_evaluated_length"] == 8
