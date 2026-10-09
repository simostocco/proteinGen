"""Focused population diagnostics: no paired coordinate loss."""

import numpy as np
import pytest
import torch

from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    Comparison,
    ReferenceStandardizer,
    bandwidth,
    calibrated_gain,
    compare_populations,
    descriptor,
    distances,
    projections,
)
from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import d_path, noise_seed


def test_descriptor_rotation_translation_invariance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    np.testing.assert_allclose(descriptor(x), descriptor(x @ q + 7), rtol=1e-10, atol=1e-10)


def test_reference_only_standardization():
    b = np.arange(80).reshape(20, 4).astype(float)
    s = ReferenceStandardizer(b + 1, b)
    np.testing.assert_allclose(np.median(s.transform(b), axis=0), 0, atol=1e-12)
    location = s.location.copy()
    s.transform(b + 1000)
    np.testing.assert_array_equal(location, s.location)


def test_missing_separation_not_fabricated():
    short = descriptor(np.random.default_rng(2).normal(size=(20, 3)))
    assert np.isnan(short[25:27]).all()  # >32 separation channels
    a = np.stack([short, short * 2, short * 3])
    s = ReferenceStandardizer(a, a)
    assert 25 not in s.columns and 26 not in s.columns


def test_energy_and_mmd_implementations():
    a = np.array([[0.0], [2.0]])
    b = np.array([[1.0], [3.0]])
    p = projections(1)
    c = Comparison(a, b, a, b, 1.0, p, [0])
    metrics = c.metrics()
    assert metrics["energy"] == pytest.approx(1.0)

    def kernel(x, y):
        return np.exp(-((x - y) ** 2) / 2)

    expected = (
        sum(kernel(x, y) for x in a[:, 0] for y in a[:, 0]) / 4
        + sum(kernel(x, y) for x in b[:, 0] for y in b[:, 0]) / 4
        - 2 * sum(kernel(x, y) for x in a[:, 0] for y in b[:, 0]) / 4
    )
    assert metrics["mmd2"] == pytest.approx(expected)
    identical = Comparison(a, a, a, a, 1.0, p, [0]).metrics()
    assert identical["energy"] == pytest.approx(0)
    assert identical["mmd2"] == pytest.approx(0)


def test_sliced_wasserstein_deterministic_and_permutation_invariant():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
    p = projections(4)
    np.testing.assert_array_equal(p, projections(4))
    c = Comparison(a, b, a, b, bandwidth(a, b), p, [1, 2, 3])
    d = Comparison(a[::-1], b[::-1], a[::-1], b[::-1], bandwidth(a, b), p, [1, 2, 3])
    for key, value in c.metrics().items():
        assert d.metrics()[key] == pytest.approx(value, abs=1e-10)


def test_near_identical_distance_cancellation_and_permutation():
    a = np.random.default_rng(14).normal(size=(100, 34)) + 100
    b = a + 1e-9
    d = distances(a, b)
    assert np.all(np.diag(d) > 0)
    np.testing.assert_array_equal(distances(a, a).diagonal(), np.zeros(100))
    np.testing.assert_array_equal(distances(a, b[::-1]), d[:, ::-1])


def test_finite_sample_calibration_and_noop():
    assert calibrated_gain(2, 3, 1) == 0.5
    assert calibrated_gain(3, 3, 1) == 0
    assert calibrated_gain(1, 3, 1) == 1
    assert calibrated_gain(1, 1, 2) is None
    rng = np.random.default_rng(5)
    b = rng.normal(size=(20, 4))
    a = rng.normal(size=(20, 4))
    pop = dict(ref_a=a, ref_b=b, noop=a, model=a)
    result = compare_populations(pop, pop, np.arange(4), 19, [10, 10], replicates=5)
    assert result["status"] == "unresolved"
    assert all(r["gain"] is None for r in result["metrics"])
    assert all(r["model_minus_noop_ci"] == [0.0, 0.0] for r in result["metrics"])


def test_equal_protein_weighting():
    # One descriptor row per protein, independent of residue/pair counts.
    a = np.array([[0.0, 0.0], [2.0, 2.0]])
    b = np.zeros_like(a)
    m = Comparison(a, b, a, b, 1.0, projections(2), [1]).metrics()
    assert m["mean_error"] == pytest.approx(1)


def test_length_strata_use_separate_reference_normalizers():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(100, 4))
    s1 = ReferenceStandardizer(a, a)
    s2 = ReferenceStandardizer(a * 100, a * 100)
    np.testing.assert_allclose(s1.transform(a), s2.transform(a * 100), atol=1e-12)


def test_independent_seed_families():
    seeds = [noise_seed(f"e014-marginal-{f}-v1", "protein", 490) for f in ("source", "ref-a", "ref-b", "gaussian")]
    assert len(set(seeds)) == 4


def test_pure_gaussian_initialization_masks_centers_and_has_no_target():
    mask = torch.arange(50)[None] < torch.tensor([20, 50])[:, None]
    placeholder = torch.zeros(2, 50, 3)
    first = centered_coordinate_noise(placeholder, mask, generator=torch.Generator().manual_seed(42))
    second = centered_coordinate_noise(placeholder, mask, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first, second)
    assert torch.equal(first[~mask], torch.zeros_like(first[~mask]))
    assert torch.allclose(first.sum(1), torch.zeros(2, 3), atol=3e-6)


def test_d_paths():
    assert str(d_path("/mnt/d/Simone/proteinGen/e014_artifacts/e014/marginal_reverse_50pct_v1")).startswith("/mnt/d/")
    with pytest.raises(ValueError):
        d_path("/tmp/marginal")


def test_inference_contract():
    import ast
    from pathlib import Path

    source = Path("scripts/collect_e014_marginal_reverse.py").read_text()
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "model"
    ]
    assert len(calls) == 2
    assert all(
        [arg.id for arg in call.args] == ["current", "t", "lengths", "mask"] and not call.keywords for call in calls
    )
