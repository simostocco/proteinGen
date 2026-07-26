#!/usr/bin/env python3
"""Timestep-resolved diagnostics for trained epsilon-prediction DDPM models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.data.preprocess import load_manifest
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    sample_symmetric_noise,
    validate_prediction_type,
)
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint

DEFAULT_TIMESTEPS = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 450, 475, 490, 499]


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
    eps_true: torch.Tensor,
    eps_hat: torch.Tensor,
    x0: torch.Tensor,
    x0_hat: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
) -> dict[str, Any]:
    true = eps_true[valid]
    pred = eps_hat[valid]
    clean = x0[valid]
    recon = x0_hat[valid]
    err = pred - true
    x0_err = recon - clean
    corr = float("nan")
    if true.numel() > 1 and float(true.std()) > 0 and float(pred.std()) > 0:
        corr = float(torch.corrcoef(torch.stack([true.float(), pred.float()]))[0, 1].cpu())
    pred_summary = tensor_summary(pred)
    true_summary = tensor_summary(true)
    x0_summary = tensor_summary(recon)
    physical = recon * float(scale)
    return {
        "predictor": name,
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
        "prediction_parameterization": config.get("prediction_type", "epsilon"),
        "normalization": {"mode": normalization.get("mode"), "scale": normalization.get("scale")},
        "q_xT_close_to_standard_normal": bool(terminal < 1e-4),
    }
    (output_dir / "schedule_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def analyze(args: argparse.Namespace) -> None:
    """Run the full diagnostic."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in Path(args.manifest).name.lower():
        raise ValueError("Refusing to run timestep diagnostics on a test manifest")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    validate_prediction_type(config.get("prediction_type", "epsilon"))
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
        x0 = item["distance_matrix"][None, None]
        mask = make_pair_mask(torch.tensor([length]), length)
        sep = make_sequence_separation(torch.tensor([length]), length)
        valid = upper_mask(length, length, device=torch.device("cpu"))
        for t_value in timesteps:
            t = torch.tensor([t_value], dtype=torch.long)
            eps = sample_symmetric_noise(tuple(x0.shape), mask, generator=generator)
            x_t, eps = diffusion.q_sample(x0, t, mask, noise=eps)
            with torch.no_grad():
                eps_model = model(x_t.float(), t, torch.tensor([length]), sep, mask).float()
            alpha_bar = diffusion.alphas_cumprod[t_value]
            predictors = {
                "model": eps_model,
                "zero": torch.zeros_like(eps),
                "noisy_input": x_t,
                "oracle": eps,
            }
            for name, eps_hat in predictors.items():
                x0_hat = diffusion.predict_x0_from_epsilon(x_t, t, eps_hat)
                metrics = _metrics_for_prediction(
                    name=name,
                    eps_true=eps,
                    eps_hat=eps_hat,
                    x0=x0,
                    x0_hat=x0_hat,
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
                    }
                )
                rows.append(metrics)
            if t_value == diffusion.timesteps - 1:
                pure = sample_symmetric_noise(tuple(x0.shape), mask, generator=generator)
                residual = alpha_bar.sqrt() * x0
                terminal_forward.update(x_t[valid])
                terminal_gaussian.update(pure[valid])
                terminal_residual.update(residual[valid])
                with torch.no_grad():
                    eps_forward = model(x_t.float(), t, torch.tensor([length]), sep, mask).float()
                    eps_pure = model(pure.float(), t, torch.tensor([length]), sep, mask).float()
                terminal_behavior["forward_eps_prediction"].update(eps_forward[valid])
                terminal_behavior["gaussian_eps_prediction"].update(eps_pure[valid])
                terminal_behavior["forward_x0_hat"].update(
                    diffusion.predict_x0_from_epsilon(x_t, t, eps_forward)[valid]
                )
                terminal_behavior["gaussian_x0_hat"].update(diffusion.predict_x0_from_epsilon(pure, t, eps_pure)[valid])
            progress.update()
    progress.close()

    raw = pd.DataFrame(rows)
    group_cols = ["timestep", "predictor"]
    summary = raw.groupby(group_cols, as_index=False).agg(
        {
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
    )
    raw.to_json(output_dir / "timestep_metrics_raw.json", orient="records", indent=2)
    summary.to_json(output_dir / "timestep_metrics.json", orient="records", indent=2)
    summary.to_csv(output_dir / "timestep_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    terminal = {
        "forward_q_xT": terminal_forward.summary(),
        "pure_gaussian": terminal_gaussian.summary(),
        "residual_sqrt_alpha_bar_x0": terminal_residual.summary(),
        "histogram_l1_distance_q_xT_vs_gaussian": histogram_l1_distance(terminal_forward, terminal_gaussian),
        "model_behavior": {key: value.summary() for key, value in terminal_behavior.items()},
    }
    (output_dir / "terminal_distribution.json").write_text(json.dumps(terminal, indent=2, sort_keys=True) + "\n")

    model_rows = summary[summary["predictor"] == "model"]
    for metric in ("epsilon_mse", "x0_reconstruction_mae_angstrom"):
        plt.figure()
        plt.plot(model_rows["timestep"], model_rows[metric], marker="o")
        plt.xlabel("timestep")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_timestep.png")
        plt.close()
        plt.figure()
        plt.plot(model_rows["snr"], model_rows[metric], marker="o")
        plt.xscale("log")
        plt.xlabel("SNR")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_snr.png")
        plt.close()

    high = model_rows.sort_values("timestep").tail(3)
    terminal_close = bool(terminal["forward_q_xT"].get("std", 0) > 0.8 and terminal["forward_q_xT"].get("std", 0) < 1.2)
    rec = [
        "# Timestep Diagnostic Recommendations",
        "",
        f"Weights evaluated: `{args.weights}`.",
        f"Terminal q(x_T) close to N(0,I): `{terminal_close}`.",
        f"High-timestep model epsilon MSE mean: `{float(high['epsilon_mse'].mean()):.6g}`.",
        f"High-timestep x0 MAE Angstrom mean: `{float(high['x0_reconstruction_mae_angstrom'].mean()):.6g}`.",
        "",
        "- Continue the same training only if repeated diagnostics show high-timestep errors improving.",
        "- Consider zero-terminal-SNR or schedule changes if q(x_T) is not close to N(0,I).",
        "- Consider centered/standardized normalization if terminal residual or mean shift is large.",
        "- Consider v-prediction if epsilon-to-x0 amplification dominates high-timestep failures.",
        "- Consider min-SNR loss weighting if timestep errors are strongly unbalanced.",
        "- Treat x0 clipping/dynamic thresholding only as a stabilizer, not proof of valid generation.",
    ]
    (output_dir / "recommendations.md").write_text("\n".join(rec) + "\n")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyze timestep-resolved DDPM epsilon errors.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    analyze(args)


if __name__ == "__main__":
    main()
