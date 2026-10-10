"""Scientific matching invariants for the observational architecture comparison."""

# ruff: noqa: I001
import sys
from pathlib import Path
import pytest

if (
    not str(Path(__file__).resolve()).startswith("/mnt/d/")
    or not Path("/mnt/d/Simone/proteinGen/e014_artifacts/population/development.parquet").exists()
):
    pytest.skip("Requires the frozen D:-resident E014 data and workspace", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e007_e014_comparison_protocol as contract
import inspect
import tempfile
import numpy as np
import torch
import yaml
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    centered_coordinate_noise,
    center_coordinates,
)
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import ReferenceStandardizer
from collect_e014_marginal_reverse import FAMILIES
from collect_e007_e014_comparison import prefix_steps
from analyze_e007_e014_comparison import distribution_difference


def test_shared_panel_and_subset():
    panel, subset = contract.population()
    assert len(panel) == 500 and len(subset) == 100
    assert all(sum(r["length_stratum"] == s for r in panel) == 100 for s in contract.STRATA)
    assert all(sum(r["length_stratum"] == s for r in subset) == 20 for s in contract.STRATA)


@pytest.mark.parametrize("t", contract.CORE)
def test_same_paired_noise_and_schedule(t):
    clean = torch.arange(192, dtype=torch.float32).reshape(1, 64, 3) / 100
    mask = torch.ones((1, 64), dtype=torch.bool)
    seed = contract.noise_seed("e014-paired-validation-v1", "fixed-test", t)
    batches = [
        CoordinateVPDiffusion(500).make_training_batch(
            clean, mask, timesteps=torch.tensor([t]), generator=torch.Generator().manual_seed(seed)
        )
        for _ in range(2)
    ]
    for field in ("noisy_coordinates", "coordinate_noise", "coordinate_v_target"):
        assert torch.equal(getattr(batches[0], field), getattr(batches[1], field))
    d = CoordinateVPDiffusion(500)
    alpha, sigma = d.alpha_sigma(torch.tensor([t]), clean)
    # Frozen make_training_batch centers clean once and training_target centers
    # clean/noise again. Match that operation order, including FP32 roundoff.
    expected = center_coordinates(
        alpha * center_coordinates(batches[0].coordinate_noise, mask)
        - sigma * center_coordinates(center_coordinates(clean, mask), mask),
        mask,
    )
    torch.testing.assert_close(batches[0].coordinate_v_target, expected, rtol=0, atol=0)
    assert torch.equal(d.betas, CoordinateVPDiffusion(500).betas)


def test_identical_normalization_semantics():
    cfg = [
        yaml.safe_load(Path(p).read_text())
        for p in (
            "configs/e007_coordinate_real_pilot_v1.yaml",
            "configs/e014_unconditional_coordinate_v_scaleup_v1.yaml",
        )
    ]
    assert cfg[0]["coordinate_scale_angstrom"] == cfg[1]["coordinate_scale_angstrom"] == contract.SCALE

    class Cache:
        def get(self, _):
            return np.arange(60, dtype=np.float32).reshape(20, 3)

    x, _, mask = coordinate_batch([dict(length=20, sample_id="x")], Cache(), contract.SCALE, device="cpu")
    expected = torch.arange(60, dtype=torch.float32).reshape(1, 20, 3) / contract.SCALE
    torch.testing.assert_close(x, center_coordinates(expected, mask))


def test_paired_bootstrap_constant_effect_and_equal_weight():
    strata = np.repeat(contract.STRATA, 4)
    a = np.arange(20, dtype=float)
    result = contract.paired_estimate(a, a + 2, strata, replicates=200)
    assert result["difference"] == 2 and result["ci95"] == [2, 2]
    # Every protein contributes once despite differing chain lengths.
    result = contract.paired_estimate(np.zeros(20), a, strata, replicates=200)
    assert result["e014"] == np.mean(a)
    indices = contract.bootstrap_indices(strata, replicates=20)
    assert all(np.sum(strata[row] == s) == 4 for row in indices for s in contract.STRATA)
    assert np.array_equal(indices, contract.bootstrap_indices(strata, replicates=20))


def test_length_strata_boundaries():
    panel, _ = contract.population()
    for row in panel:
        low, high = map(int, row["length_stratum"].split("-"))
        assert low <= row["length"] <= high


def test_gaussian_initialization_identical_and_bounded():
    mask = torch.ones((1, 128), dtype=torch.bool)
    seed = contract.gaussian_seed(7, 128, contract.noise_seed)
    initial = [
        centered_coordinate_noise(torch.zeros((1, 128, 3)), mask, generator=torch.Generator().manual_seed(seed))
        for _ in range(2)
    ]
    assert torch.equal(*initial)
    assert initial[0].mean(1).abs().max() < 1e-6
    assert min(prefix_steps()) == 351 and max(prefix_steps()) == 499
    assert len(prefix_steps()) == 149


def test_reference_independence_and_reference_only_fit():
    seeds = [contract.noise_seed(FAMILIES[f], "fixed-test", 400) for f in ("source", "ref-a", "ref-b")]
    assert len(set(seeds)) == 3
    rng = np.random.default_rng(14007)
    a, b = rng.normal(size=(20, 34)), rng.normal(size=(20, 34))
    norm = ReferenceStandardizer(a, b)
    before = (norm.location.copy(), norm.scale.copy())
    norm.transform(np.ones((20, 34)) * 1000)
    assert np.array_equal(norm.location, before[0]) and np.array_equal(norm.scale, before[1])


def test_no_sequence_or_target_inference_inputs():
    allowed = ["self", "noisy_coordinates", "timesteps", "lengths", "residue_mask"]
    assert list(inspect.signature(E007ScaleupControl.forward).parameters) == allowed
    # E014 retains this same public interface.
    assert list(inspect.signature(E014CapacityDenoiser.forward).parameters) == allowed
    code = inspect.getsource(E007ScaleupControl.forward)
    assert "sequence" not in code and "clean_coordinates" not in code


def test_D_only_paths_and_no_escape():
    assert str(contract.OUT.resolve()).startswith("/mnt/d/")
    assert str(tempfile.gettempdir()).startswith(str(contract.OUT))
    with pytest.raises(ValueError, match="D evaluation root"):
        contract.write("../escape.json", {})
    with pytest.raises(ValueError, match="D evaluation root"):
        contract.table("../../escape.parquet", [dict(a=1)])


def test_distribution_bootstrap_uses_joint_draws():
    def group(point, draws):
        return dict(
            metrics=[
                dict(metric="energy", model_ref=point, ref_ref=0.01, noop_ref=0.4, model_minus_noop_ci=[-0.2, -0.1])
            ],
            model_bootstrap_draws={"energy": draws},
        )

    result = distribution_difference(group(0.2, [0.1, 0.2, 0.3]), group(0.25, [0.15, 0.25, 0.35]), "energy")
    assert result["difference"] == pytest.approx(0.05)
    np.testing.assert_allclose(result["ci95"], [0.05, 0.05], atol=1e-15)
