"""Regression tests for timestep-resolved diffusion diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch

from protein_distance_diffusion.data.collate import make_pair_mask
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, sample_symmetric_noise
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet


def _load_diagnostic_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_timestep_errors.py"
    spec = importlib.util.spec_from_file_location("analyze_timestep_errors", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


diag = _load_diagnostic_module()


def test_amplification_factor_calculation() -> None:
    """Amplification is sqrt((1-alpha_bar)/alpha_bar)."""
    assert diag.amplification_factor(0.25) == pytest.approx(math.sqrt(3.0))
    assert math.isinf(diag.amplification_factor(0.0))


def test_timestep_boundaries_include_terminal_step() -> None:
    """The timestep grid clips requested high timesteps to T-1."""
    assert diag.selected_timesteps(3) == [0, 1, 2]
    steps = diag.selected_timesteps(500)
    assert steps[0] == 0
    assert steps[-1] == 499
    assert steps == sorted(set(steps))


def test_deterministic_subset_selection_handles_small_manifests() -> None:
    """Subset selection is deterministic and does not oversample small manifests."""
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"]})
    left = diag.deterministic_subset(frame, num_samples=2, seed=7)
    right = diag.deterministic_subset(frame, num_samples=2, seed=7)
    assert left.equals(right)
    assert len(diag.deterministic_subset(frame, num_samples=10, seed=7)) == 3


def test_empty_manifest_is_rejected() -> None:
    """An empty manifest cannot produce timestep diagnostics."""
    with pytest.raises(ValueError, match="Manifest is empty"):
        diag.deterministic_subset(pd.DataFrame(), num_samples=1, seed=1)


def test_refuses_test_manifest_before_loading_checkpoint(tmp_path: Path) -> None:
    """The diagnostic refuses test manifests."""
    args = argparse.Namespace(
        checkpoint=tmp_path / "missing.pt",
        config=tmp_path / "missing.yaml",
        manifest=tmp_path / "test.parquet",
        output_dir=tmp_path / "out",
        num_samples=1,
        seed=1,
        weights="ema",
        workers=0,
    )
    with pytest.raises(ValueError, match="test manifest"):
        diag.analyze(args)


def test_oracle_epsilon_reconstruction_is_exact() -> None:
    """The oracle epsilon baseline reconstructs x0 within numerical tolerance."""
    diffusion = GaussianDiffusion(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    lengths = torch.tensor([4])
    mask = make_pair_mask(lengths, 4)
    valid = diag.upper_mask(4, 4, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(4)
    x0 = torch.randn(1, 1, 4, 4, generator=generator)
    x0 = 0.5 * (x0 + x0.transpose(-1, -2))
    x0 = x0.masked_fill(torch.eye(4, dtype=torch.bool)[None, None], 0.0) * mask.float()
    eps = sample_symmetric_noise(tuple(x0.shape), mask, generator=generator)
    t = torch.tensor([2])
    x_t, eps = diffusion.q_sample(x0, t, mask, noise=eps)
    x0_hat = diffusion.predict_x0_from_epsilon(x_t, t, eps)
    metrics = diag._metrics_for_prediction(
        name="oracle",
        eps_true=eps,
        eps_hat=eps,
        x0=x0,
        x0_hat=x0_hat,
        valid=valid,
        scale=10.0,
    )
    assert metrics["epsilon_mse"] == pytest.approx(0.0)
    assert metrics["x0_reconstruction_mae_normalized"] < 1e-6


def test_masked_upper_triangular_metrics_ignore_diagonal_and_lower_triangle() -> None:
    """Metric calculations use only valid upper-triangular off-diagonal entries."""
    valid = diag.upper_mask(3, 3, device=torch.device("cpu"))
    eps_true = torch.zeros(1, 1, 3, 3)
    eps_hat = torch.zeros_like(eps_true)
    eps_hat[0, 0, 0, 1] = 2.0
    eps_hat[0, 0, 1, 0] = 1000.0
    eps_hat[0, 0, 0, 0] = 1000.0
    x0 = torch.zeros_like(eps_true)
    metrics = diag._metrics_for_prediction(
        name="masked",
        eps_true=eps_true,
        eps_hat=eps_hat,
        x0=x0,
        x0_hat=x0,
        valid=valid,
        scale=1.0,
    )
    assert metrics["epsilon_mse"] == pytest.approx(4.0 / 3.0)


def test_v_diagnostic_metrics_include_target_and_reconstructed_epsilon() -> None:
    """v diagnostics report v target metrics plus reconstructed epsilon metrics."""
    valid = diag.upper_mask(3, 3, device=torch.device("cpu"))
    v_true = torch.zeros(1, 1, 3, 3)
    v_hat = torch.ones_like(v_true)
    eps_true = torch.zeros_like(v_true)
    eps_hat = torch.ones_like(v_true) * 2.0
    x0 = torch.zeros_like(v_true)
    x0_hat = torch.zeros_like(v_true)
    metrics = diag._metrics_for_prediction(
        name="model",
        target_name="v",
        target_true=v_true,
        target_hat=v_hat,
        eps_true=eps_true,
        eps_hat=eps_hat,
        x0=x0,
        x0_hat=x0_hat,
        valid=valid,
        scale=1.0,
    )
    assert metrics["v_mse"] == pytest.approx(1.0)
    assert metrics["epsilon_mse"] == pytest.approx(4.0)
    assert metrics["epsilon_reconstruction_mse"] == pytest.approx(4.0)


def _factor_eight_model() -> DistanceUNet:
    return DistanceUNet(
        input_channels=3,
        output_channels=1,
        base_channels=1,
        channel_multipliers=(1, 1, 1, 1),
        residual_blocks_per_level=1,
        dropout=0.0,
        group_norm_groups=1,
        attention_heads=1,
        use_bottleneck_attention=False,
        time_embedding_dim=8,
        length_embedding_dim=8,
        max_length=500,
    )


def _item(length: int) -> dict[str, object]:
    return {
        "sample_id": f"sample_{length}",
        "length": length,
        "distance_matrix": torch.ones(length, length, dtype=torch.float32),
    }


def test_diagnostic_collation_pads_311_to_312() -> None:
    """A length-311 diagnostic sample is padded to the model-compatible side 312."""
    batch, metadata = diag._collate_diagnostic_item(_item(311), _factor_eight_model())
    assert batch["distance_matrices"].shape == (1, 1, 312, 312)
    assert batch["pair_masks"].shape == (1, 1, 312, 312)
    assert metadata["original_length"] == 311
    assert metadata["padded_length"] == 312
    assert metadata["downsample_factor"] == 8


def test_diagnostic_collation_leaves_312_unchanged() -> None:
    """Already divisible lengths are not over-padded."""
    batch, metadata = diag._collate_diagnostic_item(_item(312), _factor_eight_model())
    assert batch["distance_matrices"].shape[-1] == 312
    assert metadata["padded_length"] == 312


def test_diagnostic_collation_pads_small_nondivisible_length() -> None:
    """Small non-divisible lengths are padded to the next model-compatible side."""
    batch, metadata = diag._collate_diagnostic_item(_item(5), _factor_eight_model())
    assert batch["distance_matrices"].shape[-1] == 8
    assert metadata["model_input_shape"] == [1, 3, 8, 8]
    assert metadata["model_output_shape"] == [1, 1, 8, 8]
    assert metadata["evaluated_crop_shape"] == [1, 1, 5, 5]


def test_padded_cells_are_excluded_from_epsilon_and_x0_metrics() -> None:
    """Huge padded-cell errors do not affect cropped biological metrics."""
    length = 5
    side = 8
    valid = diag.upper_mask(length, length, device=torch.device("cpu"))
    eps_true = torch.zeros(1, 1, side, side)
    eps_hat = torch.zeros_like(eps_true)
    x0 = torch.zeros_like(eps_true)
    x0_hat = torch.zeros_like(eps_true)
    eps_hat[..., length:, :] = 1_000_000.0
    eps_hat[..., :, length:] = 1_000_000.0
    x0_hat[..., length:, :] = 1_000_000.0
    x0_hat[..., :, length:] = 1_000_000.0
    metrics = diag._metrics_for_prediction(
        name="cropped",
        eps_true=diag._crop_to_original(eps_true, length),
        eps_hat=diag._crop_to_original(eps_hat, length),
        x0=diag._crop_to_original(x0, length),
        x0_hat=diag._crop_to_original(x0_hat, length),
        valid=valid,
        scale=1.0,
    )
    assert metrics["epsilon_mse"] == pytest.approx(0.0)
    assert metrics["x0_reconstruction_mse_normalized"] == pytest.approx(0.0)


def test_cropping_restores_exact_original_shape() -> None:
    """Cropping a padded tensor restores exactly the biological N x N region."""
    tensor = torch.randn(1, 1, 8, 8)
    assert diag._crop_to_original(tensor, 5).shape == (1, 1, 5, 5)


def test_sequence_separation_padding_matches_collate_utility() -> None:
    """Sequence separation is nonzero only inside the original biological region."""
    batch, _ = diag._collate_diagnostic_item(_item(5), _factor_eight_model())
    sep = batch["sequence_separation"]
    assert sep.shape == (1, 1, 8, 8)
    assert float(sep[0, 0, 0, 4]) == pytest.approx(1.0)
    assert float(sep[0, 0, 5:, :].abs().sum()) == 0.0
    assert float(sep[0, 0, :, 5:].abs().sum()) == 0.0


def test_padded_ema_and_raw_models_accept_diagnostic_batch() -> None:
    """Both EMA and raw model diagnostics accept the padded direct model input."""
    config = {
        "model": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 1,
            "channel_multipliers": [1, 1, 1, 1],
            "residual_blocks_per_level": 1,
            "dropout": 0.0,
            "group_norm_groups": 1,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 8,
            "length_embedding_dim": 8,
            "max_length": 500,
        }
    }
    base = DistanceUNet(**config["model"])
    raw_state = {key: value.clone() for key, value in base.state_dict().items()}
    ema_state = {
        key: value.clone() + 0.25 if torch.is_floating_point(value) else value.clone()
        for key, value in raw_state.items()
    }
    for weights in ("model", "ema"):
        model = diag._build_model(config, {"model": raw_state, "ema": ema_state}, weights=weights)
        batch, _ = diag._collate_diagnostic_item(_item(9), model)
        out = model(
            batch["distance_matrices"],
            torch.tensor([0]),
            batch["lengths"],
            batch["sequence_separation"],
            batch["pair_masks"],
        )
        assert out.shape == (1, 1, 16, 16)


def test_streaming_terminal_histograms_and_forward_gaussian_comparison() -> None:
    """Streaming histograms expose finite q(xT) versus Gaussian distances."""
    bins = np.linspace(-4.0, 4.0, 41)
    forward = diag.MomentAccumulator(bins=bins)
    gaussian = diag.MomentAccumulator(bins=bins)
    forward.update(torch.tensor([-1.0, 0.0, 1.0]))
    gaussian.update(torch.tensor([-0.5, 0.0, 0.5]))
    assert forward.summary()["count"] == 3
    assert gaussian.summary()["count"] == 3
    assert math.isfinite(diag.histogram_l1_distance(forward, gaussian))


def test_ema_and_raw_weight_selection() -> None:
    """The model loader selects EMA or raw checkpoint weights explicitly."""
    config = {
        "model": {
            "input_channels": 3,
            "output_channels": 1,
            "base_channels": 2,
            "channel_multipliers": [1],
            "residual_blocks_per_level": 1,
            "dropout": 0.0,
            "group_norm_groups": 1,
            "attention_heads": 1,
            "use_bottleneck_attention": False,
            "time_embedding_dim": 8,
            "length_embedding_dim": 8,
            "max_length": 500,
        }
    }
    base = DistanceUNet(**config["model"])
    raw_state = {key: value.clone() for key, value in base.state_dict().items()}
    ema_state = {
        key: value.clone() + 0.25 if torch.is_floating_point(value) else value.clone()
        for key, value in raw_state.items()
    }
    raw = diag._build_model(config, {"model": raw_state, "ema": ema_state}, weights="model")
    ema = diag._build_model(config, {"model": raw_state, "ema": ema_state}, weights="ema")
    key = next(key for key, value in raw.state_dict().items() if torch.is_floating_point(value))
    assert not torch.allclose(raw.state_dict()[key], ema.state_dict()[key])


def test_schedule_summary_reports_terminal_standard_normal_flag(tmp_path: Path) -> None:
    """Schedule summaries contain terminal SNR and the q(xT) standard-normal flag."""
    diffusion = GaussianDiffusion(cosine_beta_schedule(10))
    summary = diag.write_schedule_summary(
        output_dir=tmp_path,
        diffusion=diffusion,
        config={"diffusion_steps": 10, "prediction_type": "epsilon"},
        normalization={"mode": "scale", "scale": 54.625},
        timesteps=diag.selected_timesteps(10),
    )
    assert "terminal_snr" in summary
    assert "q_xT_close_to_standard_normal" in summary
    assert (tmp_path / "schedule_summary.json").exists()


def _aggregated_schema_frame() -> pd.DataFrame:
    rows = []
    for timestep, snr, amplification in ((0, 9997.34082, 0.010001), (499, 0.0, 20_291.0)):
        for predictor, offset in (("model", 0.1), ("zero", 0.2), ("noisy_input", 0.3), ("oracle", 0.001)):
            rows.append(
                {
                    "timestep": timestep,
                    "predictor": predictor,
                    "epsilon_mse_mean": offset + timestep * 0.001,
                    "epsilon_mae_mean": offset,
                    "epsilon_correlation_mean": 0.5,
                    "x0_reconstruction_mse_normalized_mean": offset,
                    "x0_reconstruction_mae_normalized_mean": offset,
                    "x0_reconstruction_mae_angstrom_mean": offset,
                    "x0_reconstruction_rmse_angstrom_mean": offset + 0.1,
                    "negative_reconstructed_physical_fraction_mean": 0.01,
                    "nonfinite_fraction_mean": 0.0,
                    "snr_first": snr,
                    "amplification_factor_first": amplification,
                    "valid_pair_count_sum": 10,
                    "sample_count_sum": 1,
                    "original_length_min": 5,
                    "original_length_max": 5,
                    "padded_length_min": 8,
                    "padded_length_max": 8,
                    "epsilon_prediction_min_min": -1.0,
                    "epsilon_prediction_max_max": 1.0,
                    "epsilon_prediction_mean_mean": 0.0,
                    "epsilon_prediction_std_mean": 1.0,
                    "true_epsilon_min_min": -1.0,
                    "true_epsilon_max_max": 1.0,
                    "x0_hat_min_min": -1.0,
                    "x0_hat_max_max": 1.0,
                    "x0_hat_mean_mean": 0.0,
                    "x0_hat_std_mean": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_plots_only_uses_canonical_aggregated_schema_without_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plots-only mode consumes the suffixed CSV schema and does not instantiate a model."""
    _aggregated_schema_frame().to_csv(tmp_path / "timestep_metrics.csv", index=False)
    (tmp_path / "terminal_distribution.json").write_text(
        json.dumps(
            {
                "forward_q_xT": {"std": 1.0},
                "histogram_l1_distance_q_xT_vs_gaussian": 0.05,
            }
        )
    )
    (tmp_path / "schedule_summary.json").write_text(json.dumps({"q_xT_close_to_standard_normal": True}))

    def fail_model_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("plots-only must not instantiate or load the neural network")

    monkeypatch.setattr(diag, "_build_model", fail_model_load)
    diag.generate_plots_and_recommendations(tmp_path, weights="ema")

    for spec in diag.PLOT_SPECS:
        assert (tmp_path / spec.filename).exists()
    recommendations = (tmp_path / "recommendations.md").read_text()
    assert "High-Timestep Baseline Comparison" in recommendations
    assert "`ema`" in recommendations


def test_aggregated_schema_validation_reports_missing_columns() -> None:
    """Missing canonical aggregated columns produce a clear error."""
    frame = _aggregated_schema_frame().drop(columns=["epsilon_mse_mean"])
    with pytest.raises(ValueError, match="epsilon_mse_mean"):
        diag.validate_aggregated_metrics_schema(frame)
