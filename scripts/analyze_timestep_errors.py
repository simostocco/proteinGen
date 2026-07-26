#!/usr/bin/env python3
"""Timestep-resolved diagnostics for trained epsilon-prediction DDPM models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import collate_distance_maps
from protein_distance_diffusion.data.preprocess import load_manifest
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    ensure_config_matches_checkpoint_parameterization,
    prediction_parameterization_from_config,
    sample_symmetric_noise,
)
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint

DEFAULT_TIMESTEPS = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 450, 475, 490, 499]
PREDICTORS = ("model", "zero", "noisy_input", "oracle")


class PlotSpec(NamedTuple):
    """One plot generated from the aggregated timestep metrics CSV."""

    label: str
    y_column: str
    filename: str
    x_column: str = "timestep"
    x_label: str = "timestep"
    log_x: bool = False
    log_y: bool = False
    compare_predictors: bool = True


PLOT_SPECS = (
    PlotSpec("epsilon MSE", "epsilon_mse_mean", "epsilon_mse_mean_by_timestep.png", log_y=True),
    PlotSpec("epsilon MAE", "epsilon_mae_mean", "epsilon_mae_mean_by_timestep.png"),
    PlotSpec(
        "x0 reconstruction RMSE (Angstrom)",
        "x0_reconstruction_rmse_angstrom_mean",
        "x0_reconstruction_rmse_angstrom_mean_by_timestep.png",
        log_y=True,
    ),
    PlotSpec(
        "negative reconstructed physical fraction",
        "negative_reconstructed_physical_fraction_mean",
        "negative_reconstructed_physical_fraction_mean_by_timestep.png",
    ),
    PlotSpec(
        "epsilon-to-x0 amplification factor",
        "amplification_factor_first",
        "amplification_factor_first_by_timestep.png",
        log_y=True,
        compare_predictors=False,
    ),
    PlotSpec(
        "epsilon MSE",
        "epsilon_mse_mean",
        "epsilon_mse_mean_by_snr.png",
        x_column="snr_first",
        x_label="SNR",
        log_x=True,
        log_y=True,
    ),
)


def selected_timesteps(total_steps: int) -> list[int]:
    """Return the diagnostic timestep grid clipped to `[0, total_steps - 1]`."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    values = sorted({min(t, total_steps - 1) for t in DEFAULT_TIMESTEPS if t < total_steps or t == 499})
    if values[-1] != total_steps - 1:
        values.append(total_steps - 1)
    return values


def amplification_factor(alpha_bar: torch.Tensor | float) -> float:
    """Return sqrt((1-alpha_bar)/alpha_bar), the epsilon-to-x0 amplification."""
    value = float(alpha_bar)
    if value <= 0:
        return float("inf")
    return math.sqrt(max(1.0 - value, 0.0) / value)


def deterministic_subset(frame: pd.DataFrame, *, num_samples: int, seed: int) -> pd.DataFrame:
    """Select a deterministic validation subset without replacement."""
    if len(frame) == 0:
        raise ValueError("Manifest is empty")
    count = min(int(num_samples), len(frame))
    if count <= 0:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(frame), size=count, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def upper_mask(length: int, side: int, *, device: torch.device) -> torch.Tensor:
    """Upper-triangular off-diagonal valid-pair mask."""
    mask = torch.zeros((side, side), dtype=torch.bool, device=device)
    mask[:length, :length] = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)
    return mask[None, None]


def _downsample_stages_from_model(model: DistanceUNet) -> int:
    """Return the collate padding stage count implied by the model architecture."""
    factor = int(getattr(model, "downsample_factor", 1))
    if factor < 1 or factor & (factor - 1):
        raise ValueError(f"Model downsample_factor must be a positive power of two, got {factor}")
    return int(math.log2(factor))


def _collate_diagnostic_item(item: dict[str, Any], model: DistanceUNet) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pad one diagnostic sample with the same collation path used by training."""
    batch = collate_distance_maps([item], downsample_stages=_downsample_stages_from_model(model))
    original_length = int(batch["lengths"][0])
    padded_length = int(batch["distance_matrices"].shape[-1])
    return batch, {
        "original_length": original_length,
        "padded_length": padded_length,
        "downsample_factor": int(getattr(model, "downsample_factor", 1)),
        "model_input_shape": [1, 3, padded_length, padded_length],
        "model_output_shape": [1, 1, padded_length, padded_length],
        "evaluated_crop_shape": [1, 1, original_length, original_length],
    }


def _crop_to_original(x: torch.Tensor, length: int) -> torch.Tensor:
    """Crop a padded spatial tensor back to the biological N x N region."""
    return x[..., :length, :length]


def tensor_summary(values: torch.Tensor) -> dict[str, float]:
    """Compact scalar summary for a 1D tensor."""
    if values.numel() == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    vals = values.float()
    return {
        "min": float(vals.min().cpu()),
        "max": float(vals.max().cpu()),
        "mean": float(vals.mean().cpu()),
        "std": float(vals.std(unbiased=False).cpu()),
    }


class MomentAccumulator:
    """Streaming moments and histogram for terminal distribution diagnostics."""

    def __init__(self, *, bins: np.ndarray) -> None:
        self.bins = bins
        self.hist = np.zeros(len(bins) - 1, dtype=np.int64)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.sum_cube = 0.0
        self.outside_three = 0

    def update(self, values: torch.Tensor) -> None:
        arr = np.asarray(values.detach().cpu().tolist(), dtype=np.float64)
        if arr.size == 0:
            return
        self.hist += np.histogram(arr, bins=self.bins)[0]
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sum_sq += float(np.square(arr).sum())
        self.sum_cube += float((arr**3).sum())
        self.outside_three += int(np.sum((arr < -3.0) | (arr > 3.0)))

    def summary(self) -> dict[str, float | list[float]]:
        if self.count == 0:
            return {"count": 0}
        mean = self.sum / self.count
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        std = math.sqrt(variance)
        skew = (self.sum_cube / self.count - 3 * mean * variance - mean**3) / max(std**3, 1e-12)
        cdf = np.cumsum(self.hist)
        quantiles = []
        for q in (0.01, 0.05, 0.5, 0.95, 0.99):
            idx = int(np.searchsorted(cdf, q * max(cdf[-1], 1), side="left"))
            idx = min(idx, len(self.bins) - 2)
            quantiles.append(float(0.5 * (self.bins[idx] + self.bins[idx + 1])))
        return {
            "count": self.count,
            "mean": mean,
            "std": std,
            "quantiles_1_5_50_95_99": quantiles,
            "skewness": float(skew),
            "fraction_outside_minus3_3": self.outside_three / self.count,
        }


def histogram_l1_distance(left: MomentAccumulator, right: MomentAccumulator) -> float:
    """Return L1 histogram distance between two normalized histograms."""
    if left.count == 0 or right.count == 0:
        return float("nan")
    left_hist = left.hist / left.hist.sum()
    right_hist = right.hist / right.hist.sum()
    return float(np.abs(left_hist - right_hist).sum())


def _load_item(row: pd.Series, normalization: dict[str, Any]) -> dict[str, Any]:
    data = np.load(row["path"], allow_pickle=False)
    distance = data["distance_matrix"].astype(np.float32)
    if normalization.get("mode") == "scale":
        distance = distance / np.float32(normalization["scale"])
    return {
        "sample_id": str(data["sample_id"]),
        "length": int(distance.shape[0]),
        "distance_matrix": torch.tensor(distance.tolist(), dtype=torch.float32),
    }


def _metrics_for_prediction(
    *,
    name: str,
    target_name: str = "epsilon",
    target_true: torch.Tensor | None = None,
    target_hat: torch.Tensor | None = None,
    eps_true: torch.Tensor,
    eps_hat: torch.Tensor,
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
) -> dict[str, Any]:
    target_true = eps_true if target_true is None else target_true
    target_hat = eps_hat if target_hat is None else target_hat
    target_values = target_true[valid]
    target_pred = target_hat[valid]
    true = eps_true[valid]
    pred = eps_hat[valid]
    clean = x0[valid]
    recon = x0_hat[valid]
    target_err = target_pred - target_values
    target_corr = float("nan")
    if target_values.numel() > 1 and float(target_values.std()) > 0 and float(target_pred.std()) > 0:
        target_corr = float(torch.corrcoef(torch.stack([target_values.float(), target_pred.float()]))[0, 1].cpu())
    err = pred - true
    x0_err = recon - clean
    corr = float("nan")
    if true.numel() > 1 and float(true.std()) > 0 and float(pred.std()) > 0:
        corr = float(torch.corrcoef(torch.stack([true.float(), pred.float()]))[0, 1].cpu())
    pred_summary = tensor_summary(pred)
    true_summary = tensor_summary(true)
    x0_summary = tensor_summary(recon)
    physical = recon * float(scale)
    metrics = {
        "predictor": name,
        "target_parameterization": target_name,
        "target_mse": float(target_err.square().mean().cpu()),
        "target_mae": float(target_err.abs().mean().cpu()),
        "target_correlation": target_corr,
        "epsilon_mse": float(err.square().mean().cpu()),
        "epsilon_mae": float(err.abs().mean().cpu()),
        "epsilon_correlation": corr,
        "epsilon_prediction_min": pred_summary["min"],
        "epsilon_prediction_max": pred_summary["max"],
        "epsilon_prediction_mean": pred_summary["mean"],
        "epsilon_prediction_std": pred_summary["std"],
        "true_epsilon_min": true_summary["min"],
        "true_epsilon_max": true_summary["max"],
        "true_epsilon_mean": true_summary["mean"],
        "true_epsilon_std": true_summary["std"],
        "x0_reconstruction_mse_normalized": float(x0_err.square().mean().cpu()),
        "x0_reconstruction_mae_normalized": float(x0_err.abs().mean().cpu()),
        "x0_reconstruction_mae_angstrom": float((x0_err.abs() * scale).mean().cpu()),
        "x0_reconstruction_rmse_angstrom": float((x0_err.square().mean().sqrt() * scale).cpu()),
        "x0_hat_min": x0_summary["min"],
        "x0_hat_max": x0_summary["max"],
        "x0_hat_mean": x0_summary["mean"],
        "x0_hat_std": x0_summary["std"],
        "negative_reconstructed_physical_fraction": float((physical < 0).float().mean().cpu()),
        "nonfinite_fraction": float((~torch.isfinite(recon)).float().mean().cpu()),
    }
    metrics[f"{target_name}_mse"] = metrics["target_mse"]
    metrics[f"{target_name}_mae"] = metrics["target_mae"]
    metrics[f"{target_name}_correlation"] = metrics["target_correlation"]
    if target_name == "v":
        metrics["epsilon_reconstruction_mse"] = metrics["epsilon_mse"]
        metrics["epsilon_reconstruction_mae"] = metrics["epsilon_mae"]
        metrics["epsilon_reconstruction_correlation"] = metrics["epsilon_correlation"]
    return metrics


def _build_model(config: dict[str, Any], checkpoint: dict[str, Any], *, weights: str) -> DistanceUNet:
    model_cfg = dict(config["model"])
    if "channel_multipliers" in model_cfg:
        model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    if weights == "ema":
        if "ema" not in checkpoint:
            raise ValueError("EMA weights requested but checkpoint has no EMA state")
        model.load_state_dict(checkpoint["ema"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def write_schedule_summary(
    *,
    output_dir: Path,
    diffusion: GaussianDiffusion,
    config: dict[str, Any],
    normalization: dict[str, Any],
    timesteps: list[int],
) -> dict[str, Any]:
    """Write schedule_summary.json."""
    alpha_bars = diffusion.alphas_cumprod
    selected = {
        str(t): {
            "alpha_bar": float(alpha_bars[t]),
            "sqrt_alpha_bar": float(alpha_bars[t].sqrt()),
            "amplification_factor": amplification_factor(alpha_bars[t]),
        }
        for t in timesteps
    }
    terminal = float(alpha_bars[-1])
    summary = {
        "diffusion_steps": diffusion.timesteps,
        "beta_schedule": {"name": config.get("beta_schedule", "cosine"), "s": 0.008},
        "alpha_bar_t0": float(alpha_bars[0]),
        "alpha_bar_t_terminal": terminal,
        "sqrt_alpha_bar_t0": float(alpha_bars[0].sqrt()),
        "sqrt_alpha_bar_t_terminal": float(alpha_bars[-1].sqrt()),
        "terminal_snr": terminal / max(1.0 - terminal, 1e-12),
        "selected_timesteps": selected,
        "prediction_parameterization": prediction_parameterization_from_config(config).value,
        "normalization": {"mode": normalization.get("mode"), "scale": normalization.get("scale")},
        "q_xT_close_to_standard_normal": bool(terminal < 1e-4),
    }
    (output_dir / "schedule_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _required_plot_columns() -> set[str]:
    columns = {"timestep", "predictor"}
    for spec in PLOT_SPECS:
        columns.add(spec.x_column)
        columns.add(spec.y_column)
    return columns


def validate_aggregated_metrics_schema(frame: pd.DataFrame) -> None:
    """Validate the canonical aggregated timestep metric columns."""
    missing = sorted(_required_plot_columns() - set(frame.columns))
    if missing:
        raise ValueError(
            "timestep_metrics.csv is missing required aggregated column(s): "
            + ", ".join(missing)
            + ". Expected canonical names such as epsilon_mse_mean and snr_first."
        )


def _plot_spec(frame: pd.DataFrame, output_dir: Path, spec: PlotSpec) -> None:
    plt.figure()
    predictors = PREDICTORS if spec.compare_predictors else ("model",)
    for predictor in predictors:
        rows = frame[frame["predictor"] == predictor].sort_values(spec.x_column)
        if rows.empty:
            continue
        plot_rows = rows
        if spec.log_x:
            plot_rows = plot_rows[plot_rows[spec.x_column] > 0]
        if spec.log_y:
            plot_rows = plot_rows[plot_rows[spec.y_column] > 0]
        if plot_rows.empty:
            continue
        plt.plot(plot_rows[spec.x_column], plot_rows[spec.y_column], marker="o", label=predictor)
    if spec.log_x:
        plt.xscale("log")
    if spec.log_y:
        plt.yscale("log")
    plt.xlabel(spec.x_label)
    plt.ylabel(spec.label)
    if spec.compare_predictors:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / spec.filename)
    plt.close()


def generate_recommendations(output_dir: Path, frame: pd.DataFrame, *, weights: str) -> None:
    """Generate recommendations.md from existing numerical outputs."""
    terminal = _load_json_if_present(output_dir / "terminal_distribution.json")
    schedule = _load_json_if_present(output_dir / "schedule_summary.json")
    model_rows = frame[frame["predictor"] == "model"].sort_values("timestep")
    high = model_rows.tail(3)
    high_epsilon_mse = float(high["epsilon_mse_mean"].mean()) if not high.empty else float("nan")
    high_rmse = float(high["x0_reconstruction_rmse_angstrom_mean"].mean()) if not high.empty else float("nan")
    terminal_stats = terminal.get("forward_q_xT", {})
    terminal_std = terminal_stats.get("std")
    terminal_close = bool(isinstance(terminal_std, int | float) and 0.8 < float(terminal_std) < 1.2)
    theoretical_terminal_close = schedule.get("q_xT_close_to_standard_normal", "unknown")
    terminal_l1 = terminal.get("histogram_l1_distance_q_xT_vs_gaussian", "unknown")
    high_t = int(high["timestep"].max()) if not high.empty else "unknown"
    high_t_rows = frame[frame["timestep"] == high_t] if isinstance(high_t, int) else pd.DataFrame()
    predictor_lines = []
    if not high_t_rows.empty:
        ranked = high_t_rows.sort_values("epsilon_mse_mean")[["predictor", "epsilon_mse_mean"]]
        predictor_lines = [
            f"- `{row.predictor}` epsilon MSE at t={high_t}: `{float(row.epsilon_mse_mean):.6g}`"
            for row in ranked.itertuples()
        ]
    rec = [
        "# Timestep Diagnostic Recommendations",
        "",
        f"Weights evaluated: `{weights}`.",
        f"Terminal q(x_T) empirically close to N(0,I): `{terminal_close}`.",
        f"Terminal q(x_T) theoretically close to N(0,I): `{theoretical_terminal_close}`.",
        f"Histogram L1 distance q(x_T) vs N(0,I): `{terminal_l1}`.",
        f"High-timestep model epsilon MSE mean: `{high_epsilon_mse:.6g}`.",
        f"High-timestep model x0 RMSE Angstrom mean: `{high_rmse:.6g}`.",
        "",
        "## High-Timestep Baseline Comparison",
        "",
        *(predictor_lines or ["- No high-timestep predictor comparison was available."]),
        "",
        "## Decision Notes",
        "",
        "- Continue the same training only if repeated diagnostics show high-timestep errors improving.",
        "- Consider zero-terminal-SNR or schedule changes if q(x_T) is not close to N(0,I).",
        "- Consider centered/standardized normalization if scale-only normalization leaves a large terminal shift.",
        "- Consider v-prediction if epsilon-to-x0 amplification dominates high-timestep failures.",
        "- Consider min-SNR loss weighting if timestep performance is strongly unbalanced.",
        "- Treat x0 clipping/dynamic thresholding only as a stabilizer, not proof of valid generation.",
    ]
    (output_dir / "recommendations.md").write_text("\n".join(rec) + "\n")


def generate_plots_and_recommendations(output_dir: Path, *, weights: str) -> None:
    """Generate plots and recommendations from canonical aggregated diagnostic outputs."""
    metrics_path = output_dir / "timestep_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing aggregated metrics CSV: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    validate_aggregated_metrics_schema(frame)
    for spec in PLOT_SPECS:
        _plot_spec(frame, output_dir, spec)
    generate_recommendations(output_dir, frame, weights=weights)


def analyze(args: argparse.Namespace) -> None:
    """Run the full diagnostic."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in Path(args.manifest).name.lower():
        raise ValueError("Refusing to run timestep diagnostics on a test manifest")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    prediction_type = ensure_config_matches_checkpoint_parameterization(
        config=load_yaml(args.config),
        checkpoint_config=config,
    )
    normalization = checkpoint.get("normalization") or json.loads(
        Path(load_yaml(args.config)["normalization_file"]).read_text()
    )
    scale = float(normalization.get("scale", 1.0))
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100))))
    timesteps = selected_timesteps(diffusion.timesteps)
    write_schedule_summary(
        output_dir=output_dir,
        diffusion=diffusion,
        config=config,
        normalization=normalization,
        timesteps=timesteps,
    )

    frame = deterministic_subset(load_manifest(args.manifest), num_samples=args.num_samples, seed=args.seed)
    model = _build_model(config, checkpoint, weights=args.weights)
    model.to(torch.device("cpu"))
    rows: list[dict[str, Any]] = []
    sample_metadata: list[dict[str, Any]] = []
    bins = np.linspace(-8.0, 8.0, 321)
    terminal_forward = MomentAccumulator(bins=bins)
    terminal_gaussian = MomentAccumulator(bins=bins)
    terminal_residual = MomentAccumulator(bins=bins)
    terminal_behavior: dict[str, MomentAccumulator] = {
        "forward_eps_prediction": MomentAccumulator(bins=bins),
        "gaussian_eps_prediction": MomentAccumulator(bins=bins),
        "forward_x0_hat": MomentAccumulator(bins=np.linspace(-1000, 1000, 401)),
        "gaussian_x0_hat": MomentAccumulator(bins=np.linspace(-1000, 1000, 401)),
    }

    progress = tqdm(total=len(frame) * len(timesteps), desc="Timestep diagnostics", unit="case", dynamic_ncols=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for _, row in frame.iterrows():
        item = _load_item(row, normalization)
        length = int(item["length"])
        batch, metadata = _collate_diagnostic_item(item, model)
        padded_length = int(metadata["padded_length"])
        sample_metadata.append({"sample_id": str(item["sample_id"]), **metadata})
        x0 = batch["distance_matrices"]
        mask = batch["pair_masks"]
        sep = batch["sequence_separation"]
        valid = upper_mask(length, length, device=torch.device("cpu"))
        for t_value in timesteps:
            t = torch.tensor([t_value], dtype=torch.long)
            eps = sample_symmetric_noise(tuple(x0.shape), mask, generator=generator)
            x_t, eps = diffusion.q_sample(x0, t, mask, noise=eps)
            target = diffusion.training_target(
                x_start=x0,
                t=t,
                epsilon=eps,
                prediction_type=prediction_type,
            )
            with torch.no_grad():
                model_output = model(x_t.float(), t, torch.tensor([length]), sep, mask).float()
            alpha_bar = diffusion.alphas_cumprod[t_value]
            predictors = {
                "model": model_output,
                "zero": torch.zeros_like(target),
                "noisy_input": x_t,
                "oracle": target,
            }
            for name, output_hat in predictors.items():
                x0_hat, eps_hat = diffusion.predict_x0_epsilon_from_model_output(
                    x_t=x_t,
                    t=t,
                    model_output=output_hat,
                    prediction_type=prediction_type,
                )
                metrics = _metrics_for_prediction(
                    name=name,
                    target_name=prediction_type.value,
                    target_true=_crop_to_original(target, length),
                    target_hat=_crop_to_original(output_hat, length),
                    eps_true=_crop_to_original(eps, length),
                    eps_hat=_crop_to_original(eps_hat, length),
                    x0=_crop_to_original(x0, length),
                    x0_hat=_crop_to_original(x0_hat, length),
                    valid=valid,
                    scale=scale,
                )
                metrics.update(
                    {
                        "timestep": t_value,
                        "snr": float(alpha_bar / max(1.0 - float(alpha_bar), 1e-12)),
                        "amplification_factor": amplification_factor(alpha_bar),
                        "sample_count": 1,
                        "valid_pair_count": int(valid.sum()),
                        "original_length": length,
                        "padded_length": padded_length,
                    }
                )
                rows.append(metrics)
            if t_value == diffusion.timesteps - 1:
                pure = sample_symmetric_noise(tuple(x0.shape), mask, generator=generator)
                residual = alpha_bar.sqrt() * x0
                terminal_forward.update(_crop_to_original(x_t, length)[valid])
                terminal_gaussian.update(_crop_to_original(pure, length)[valid])
                terminal_residual.update(_crop_to_original(residual, length)[valid])
                with torch.no_grad():
                    output_forward = model(x_t.float(), t, torch.tensor([length]), sep, mask).float()
                    output_pure = model(pure.float(), t, torch.tensor([length]), sep, mask).float()
                _, eps_forward = diffusion.predict_x0_epsilon_from_model_output(
                    x_t=x_t,
                    t=t,
                    model_output=output_forward,
                    prediction_type=prediction_type,
                )
                _, eps_pure = diffusion.predict_x0_epsilon_from_model_output(
                    x_t=pure,
                    t=t,
                    model_output=output_pure,
                    prediction_type=prediction_type,
                )
                terminal_behavior["forward_eps_prediction"].update(_crop_to_original(eps_forward, length)[valid])
                terminal_behavior["gaussian_eps_prediction"].update(_crop_to_original(eps_pure, length)[valid])
                terminal_behavior["forward_x0_hat"].update(
                    _crop_to_original(
                        diffusion.predict_x0_epsilon_from_model_output(
                            x_t=x_t,
                            t=t,
                            model_output=output_forward,
                            prediction_type=prediction_type,
                        )[0],
                        length,
                    )[valid]
                )
                terminal_behavior["gaussian_x0_hat"].update(
                    _crop_to_original(
                        diffusion.predict_x0_epsilon_from_model_output(
                            x_t=pure,
                            t=t,
                            model_output=output_pure,
                            prediction_type=prediction_type,
                        )[0],
                        length,
                    )[valid]
                )
            progress.update()
    progress.close()

    raw = pd.DataFrame(rows)
    group_cols = ["timestep", "predictor"]
    aggregations: dict[str, Any] = {
        "target_mse": "mean",
        "target_mae": "mean",
        "target_correlation": "mean",
        "epsilon_mse": "mean",
        "epsilon_mae": "mean",
        "epsilon_correlation": "mean",
        "x0_reconstruction_mse_normalized": "mean",
        "x0_reconstruction_mae_normalized": "mean",
        "x0_reconstruction_mae_angstrom": "mean",
        "x0_reconstruction_rmse_angstrom": "mean",
        "negative_reconstructed_physical_fraction": "mean",
        "nonfinite_fraction": "mean",
        "snr": "first",
        "amplification_factor": "first",
        "valid_pair_count": "sum",
        "sample_count": "sum",
        "original_length": ["min", "max"],
        "padded_length": ["min", "max"],
        "epsilon_prediction_min": "min",
        "epsilon_prediction_max": "max",
        "epsilon_prediction_mean": "mean",
        "epsilon_prediction_std": "mean",
        "true_epsilon_min": "min",
        "true_epsilon_max": "max",
        "x0_hat_min": "min",
        "x0_hat_max": "max",
        "x0_hat_mean": "mean",
        "x0_hat_std": "mean",
    }
    for optional in (
        "v_mse",
        "v_mae",
        "v_correlation",
        "epsilon_reconstruction_mse",
        "epsilon_reconstruction_mae",
        "epsilon_reconstruction_correlation",
    ):
        if optional in raw.columns:
            aggregations[optional] = "mean"
    summary = raw.groupby(group_cols, as_index=False).agg(aggregations)
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col) for col in summary.columns
    ]
    raw.to_json(output_dir / "timestep_metrics_raw.json", orient="records", indent=2)
    summary.to_json(output_dir / "timestep_metrics.json", orient="records", indent=2)
    summary.to_csv(output_dir / "timestep_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (output_dir / "diagnostic_sample_metadata.json").write_text(
        json.dumps(sample_metadata, indent=2, sort_keys=True) + "\n"
    )

    terminal = {
        "forward_q_xT": terminal_forward.summary(),
        "pure_gaussian": terminal_gaussian.summary(),
        "residual_sqrt_alpha_bar_x0": terminal_residual.summary(),
        "histogram_l1_distance_q_xT_vs_gaussian": histogram_l1_distance(terminal_forward, terminal_gaussian),
        "model_behavior": {key: value.summary() for key, value in terminal_behavior.items()},
    }
    (output_dir / "terminal_distribution.json").write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")

    if not getattr(args, "skip_plots", False):
        generate_plots_and_recommendations(output_dir, weights=args.weights)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze timestep-resolved DDPM epsilon errors.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate plots/recommendations from existing outputs.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write numerical diagnostics without plots/recommendations.",
    )
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.plots_only:
        generate_plots_and_recommendations(args.output_dir, weights=args.weights)
        return
    missing = [name for name in ("checkpoint", "config", "manifest") if getattr(args, name) is None]
    if missing:
        parser.error("normal diagnostic mode requires: " + ", ".join(f"--{name}" for name in missing))
    analyze(args)


if __name__ == "__main__":
    main()
