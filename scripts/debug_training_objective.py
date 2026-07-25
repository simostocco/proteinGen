#!/usr/bin/env python3
"""Audit the diffusion training objective without updating model weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_distance_diffusion.config import load_yaml  # noqa: E402
from protein_distance_diffusion.data.collate import (  # noqa: E402
    collate_distance_maps,
    make_pair_mask,
    make_sequence_separation,
)
from protein_distance_diffusion.data.dataset import DistanceMapDataset  # noqa: E402
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion, masked_upper_triangular_loss  # noqa: E402
from protein_distance_diffusion.diffusion.sampling import sample_ddpm  # noqa: E402
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule  # noqa: E402
from protein_distance_diffusion.training.checkpointing import load_checkpoint  # noqa: E402
from protein_distance_diffusion.training.trainer import build_model_from_config  # noqa: E402


@dataclass
class BatchDiagnostics:
    loss: float
    numerator: float
    denominator: float
    clean: dict[str, float]
    noise: dict[str, float]
    noisy: dict[str, float]
    prediction: dict[str, float]
    valid_entries: int
    t_min: int
    t_max: int
    t_mean: float
    zero_predictor_loss: float
    noisy_predictor_loss: float
    untrained_loss: float
    noisy_target_alias: bool
    clean_noisy_alias: bool
    clean_target_alias: bool


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6e}"


def _stats(name: str, tensor: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    values = tensor[valid]
    if values.numel() == 0:
        return {
            f"{name}_count": 0.0,
            f"{name}_mean": float("nan"),
            f"{name}_std": float("nan"),
            f"{name}_min": float("nan"),
            f"{name}_max": float("nan"),
            f"{name}_mse_vs_zero": float("nan"),
        }
    return {
        f"{name}_count": float(values.numel()),
        f"{name}_mean": float(values.mean().detach().cpu()),
        f"{name}_std": float(values.std(unbiased=False).detach().cpu()),
        f"{name}_min": float(values.min().detach().cpu()),
        f"{name}_max": float(values.max().detach().cpu()),
        f"{name}_mse_vs_zero": float(values.pow(2).mean().detach().cpu()),
    }


def _upper_valid(pair_mask: torch.Tensor) -> torch.Tensor:
    matrix_size = pair_mask.shape[-1]
    upper = torch.triu(torch.ones((matrix_size, matrix_size), dtype=torch.bool, device=pair_mask.device), diagonal=1)
    return pair_mask.bool() & upper[None, None]


def _loss_parts(epsilon: torch.Tensor, epsilon_hat: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, float]:
    sq = (epsilon - epsilon_hat).pow(2) * valid.float()
    per_sample_denominator = valid.flatten(1).sum(dim=1).clamp_min(1.0)
    per_sample_numerator = sq.flatten(1).sum(dim=1)
    per_sample_loss = per_sample_numerator / per_sample_denominator
    return (
        float(per_sample_numerator.sum().detach().cpu()),
        float(per_sample_denominator.sum().detach().cpu()),
        float(per_sample_loss.mean().detach().cpu()),
    )


def _loader(
    manifest_path: str | Path,
    normalization: dict[str, Any],
    *,
    batch_size: int,
    downsample_stages: int,
) -> DataLoader:
    dataset = DistanceMapDataset(manifest_path, normalization)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_distance_maps(items, downsample_stages=downsample_stages),
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("distance_matrices", "pair_masks", "lengths", "sequence_separation"):
        moved[key] = batch[key].to(device)
    return moved


def _deterministic_batch_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: dict[str, Any],
    *,
    device: torch.device,
    seed: int,
    model_config: dict[str, Any],
) -> BatchDiagnostics:
    batch = _move_batch(batch, device)
    clean = batch["distance_matrices"]
    pair_mask = batch["pair_masks"]
    lengths = batch["lengths"]
    sep = batch["sequence_separation"]
    generator = torch.Generator(device=device).manual_seed(seed)
    t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device, generator=generator)
    noisy, epsilon = diffusion.q_sample(clean, t, pair_mask, generator=generator)
    prediction = model(noisy, t, lengths, sep, pair_mask)
    valid = _upper_valid(pair_mask)
    numerator, denominator, explicit_loss = _loss_parts(epsilon, prediction, valid)
    untrained = build_model_from_config(model_config).to(device)
    untrained.eval()
    untrained_prediction = untrained(noisy, t, lengths, sep, pair_mask)
    return BatchDiagnostics(
        loss=float(masked_upper_triangular_loss(epsilon, prediction, pair_mask).detach().cpu()),
        numerator=numerator,
        denominator=denominator,
        clean=_stats("clean", clean, valid),
        noise=_stats("epsilon", epsilon, valid),
        noisy=_stats("noisy", noisy, valid),
        prediction=_stats("epsilon_hat", prediction, valid),
        valid_entries=int(valid.sum().detach().cpu()),
        t_min=int(t.min().detach().cpu()),
        t_max=int(t.max().detach().cpu()),
        t_mean=float(t.float().mean().detach().cpu()),
        zero_predictor_loss=float(masked_upper_triangular_loss(epsilon, torch.zeros_like(epsilon), pair_mask).cpu()),
        noisy_predictor_loss=float(masked_upper_triangular_loss(epsilon, noisy, pair_mask).detach().cpu()),
        untrained_loss=float(masked_upper_triangular_loss(epsilon, untrained_prediction, pair_mask).detach().cpu()),
        noisy_target_alias=noisy.data_ptr() == epsilon.data_ptr(),
        clean_noisy_alias=clean.data_ptr() == noisy.data_ptr(),
        clean_target_alias=clean.data_ptr() == epsilon.data_ptr(),
    )


@torch.no_grad()
def _exact_loss(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
    label: str,
) -> tuple[float, int, int, int]:
    generator = torch.Generator(device=device).manual_seed(seed)
    losses: list[torch.Tensor] = []
    valid_total = 0
    zero_valid_samples = 0
    sample_count = 0
    for batch in tqdm(loader, desc=f"exact {label}", dynamic_ncols=True, mininterval=0.5):
        batch = _move_batch(batch, device)
        clean = batch["distance_matrices"]
        pair_mask = batch["pair_masks"]
        lengths = batch["lengths"]
        sep = batch["sequence_separation"]
        t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device, generator=generator)
        noisy, epsilon = diffusion.q_sample(clean, t, pair_mask, generator=generator)
        prediction = model(noisy, t, lengths, sep, pair_mask)
        valid = _upper_valid(pair_mask)
        valid_per_sample = valid.flatten(1).sum(dim=1)
        valid_total += int(valid_per_sample.sum().detach().cpu())
        zero_valid_samples += int((valid_per_sample == 0).sum().detach().cpu())
        sample_count += int(clean.shape[0])
        losses.append(masked_upper_triangular_loss(epsilon, prediction, pair_mask).detach().cpu())
    return (
        float(torch.stack(losses).mean()) if losses else float("nan"),
        valid_total,
        zero_valid_samples,
        sample_count,
    )


def _summarize_clean(loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    chunks = []
    valid_entries = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        valid = _upper_valid(batch["pair_masks"])
        values = batch["distance_matrices"][valid]
        valid_entries += int(values.numel())
        if values.numel() > 0:
            chunks.append(values.detach().cpu())
    if not chunks:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    values = torch.cat(chunks)
    return {
        "count": float(valid_entries),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _scaled_stats(stats: dict[str, float], scale: float) -> dict[str, float]:
    return {
        "count": stats["count"],
        "mean": stats["mean"] * scale,
        "std": stats["std"] * scale,
        "min": stats["min"] * scale,
        "max": stats["max"] * scale,
    }


def _timestep_distribution(timesteps: int, *, seed: int, samples: int, device: torch.device) -> dict[str, float]:
    generator = torch.Generator(device=device).manual_seed(seed)
    t = torch.randint(0, timesteps, (samples,), device=device, generator=generator)
    unique = t.unique().numel()
    return {
        "samples": float(samples),
        "unique_timesteps": float(unique),
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.float().mean()),
        "std": float(t.float().std(unbiased=False)),
        "fraction_t0": float((t == 0).float().mean()),
    }


@torch.no_grad()
def _generated_distribution(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    *,
    lengths: Iterable[int],
    normalization: dict[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    length_list = list(lengths)
    if not length_list:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    side = max(length_list)
    factor = getattr(model, "downsample_factor", 1)
    side = ((side + factor - 1) // factor) * factor
    length_tensor = torch.tensor(length_list, dtype=torch.long, device=device)
    pair_mask = make_pair_mask(length_tensor, side).to(device)
    sep = make_sequence_separation(length_tensor, side).to(device)
    samples = sample_ddpm(
        model,
        diffusion,
        lengths=length_tensor,
        pair_mask=pair_mask,
        sequence_separation=sep,
        device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    scale = float(normalization.get("scale", 1.0))
    valid = _upper_valid(pair_mask)
    values = samples[valid].detach().cpu() * scale
    if values.numel() == 0:
        return {"count": 0.0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": float(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _print_mapping(title: str, mapping: dict[str, float | int | str | bool]) -> None:
    print(f"\n{title}")
    for key, value in mapping.items():
        if isinstance(value, bool | str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {_fmt(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit diffusion objective values from a checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--timestep-samples", type=int, default=4096)
    parser.add_argument("--num-generated", type=int, default=4)
    args = parser.parse_args()

    config = load_yaml(args.config)
    checkpoint_path = args.checkpoint or Path(config["output_dir"]) / "checkpoints" / "latest.pt"
    device_name = args.device or ("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    device = torch.device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    normalization = checkpoint.get("normalization") or json.loads(Path(config["normalization_file"]).read_text())
    downsample_stages = len(config["model"].get("channel_multipliers", [1, 2, 4, 8])) - 1
    batch_size = int(config.get("batch_size", 1))
    train_loader = _loader(
        config["train_manifest"],
        normalization,
        batch_size=batch_size,
        downsample_stages=downsample_stages,
    )
    validation_loader = _loader(
        config["validation_manifest"],
        normalization,
        batch_size=batch_size,
        downsample_stages=downsample_stages,
    )
    model = build_model_from_config(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100)))).to(device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint_global_step: {checkpoint.get('global_step')}")
    print("checkpoint_state_for_loss: model")
    print(f"train_manifest: {config['train_manifest']}")
    print(f"validation_manifest: {config['validation_manifest']}")
    print(f"same_manifest: {Path(config['train_manifest']).resolve() == Path(config['validation_manifest']).resolve()}")
    print("objective: mean over batch of sum_upper((epsilon - epsilon_hat)^2) / valid_upper_pair_count")
    print("conditioning_channels: noisy_distance, sequence_separation, pair_mask")

    train_loss, train_valid, train_zero_valid, train_samples = _exact_loss(
        model,
        diffusion,
        train_loader,
        device=device,
        seed=int(config.get("seed", 42)),
        label="train",
    )
    validation_loss, validation_valid, validation_zero_valid, validation_samples = _exact_loss(
        model,
        diffusion,
        validation_loader,
        device=device,
        seed=int(config.get("seed", 42)),
        label="validation",
    )
    _print_mapping(
        "Exact checkpoint losses",
        {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_valid_upper_entries": train_valid,
            "validation_valid_upper_entries": validation_valid,
            "train_zero_valid_samples": train_zero_valid,
            "validation_zero_valid_samples": validation_zero_valid,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
        },
    )

    train_batch = next(iter(train_loader))
    validation_batch = next(iter(validation_loader))
    train_diag = _deterministic_batch_loss(
        model,
        diffusion,
        train_batch,
        device=device,
        seed=int(config.get("seed", 42)) + 1,
        model_config=config["model"],
    )
    validation_diag = _deterministic_batch_loss(
        model,
        diffusion,
        validation_batch,
        device=device,
        seed=int(config.get("seed", 42)) + 2,
        model_config=config["model"],
    )
    batch_diagnostics = (
        ("Train batch objective parts", train_diag),
        ("Validation batch objective parts", validation_diag),
    )
    for label, diag in batch_diagnostics:
        _print_mapping(
            label,
            {
                "loss": diag.loss,
                "loss_numerator": diag.numerator,
                "loss_denominator": diag.denominator,
                "explicit_normalized_loss": diag.loss,
                "valid_upper_entries": diag.valid_entries,
                "t_min": diag.t_min,
                "t_max": diag.t_max,
                "t_mean": diag.t_mean,
                "zero_predictor_loss": diag.zero_predictor_loss,
                "noisy_input_as_epsilon_loss": diag.noisy_predictor_loss,
                "untrained_model_loss": diag.untrained_loss,
                "clean_noisy_alias": diag.clean_noisy_alias,
                "clean_target_alias": diag.clean_target_alias,
                "noisy_target_alias": diag.noisy_target_alias,
            },
        )
        _print_mapping(f"{label}: clean normalized distance stats", diag.clean)
        _print_mapping(f"{label}: Gaussian noise stats", diag.noise)
        _print_mapping(f"{label}: noisy input stats", diag.noisy)
        _print_mapping(f"{label}: model epsilon output stats", diag.prediction)

    _print_mapping(
        "Sampled timestep distribution",
        _timestep_distribution(
            diffusion.timesteps,
            seed=int(config.get("seed", 42)),
            samples=args.timestep_samples,
            device=device,
        ),
    )
    train_clean_stats = _summarize_clean(train_loader, device=device)
    validation_clean_stats = _summarize_clean(validation_loader, device=device)
    _print_mapping("Train clean normalized distance statistics", train_clean_stats)
    _print_mapping(
        "Validation clean normalized distance statistics",
        validation_clean_stats,
    )
    _print_mapping(
        "Validation clean physical distance distribution",
        _scaled_stats(validation_clean_stats, float(normalization.get("scale", 1.0))),
    )
    validation_lengths = [int(length) for length in validation_batch["lengths"].flatten()[: args.num_generated]]
    _print_mapping(
        "Generated physical distance distribution",
        _generated_distribution(
            model,
            diffusion,
            lengths=validation_lengths,
            normalization=normalization,
            device=device,
            seed=int(config.get("seed", 42)),
        ),
    )


if __name__ == "__main__":
    main()
