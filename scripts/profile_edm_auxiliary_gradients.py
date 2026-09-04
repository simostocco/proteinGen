#!/usr/bin/env python
"""Profile physical auxiliary-loss gradient scale without optimizer updates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protein_distance_diffusion.data.collate import collate_distance_maps
from protein_distance_diffusion.data.dataset import DistanceMapDataset
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    masked_upper_triangular_loss,
    prediction_parameterization_from_config,
)
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.physical_auxiliary import (
    adjacent_auxiliary_config_from_mapping,
    adjacent_auxiliary_weight,
    adjacent_chain_smooth_l1_loss,
    physical_auxiliary_config_from_mapping,
    physical_auxiliary_weight,
    stochastic_edm_spectral_loss,
)
from protein_distance_diffusion.training.trainer import _amp_dtype, _autocast_context, build_model_from_config

DEFAULT_FIXED_TIMESTEPS = [0, 50, 100, 200, 300, 400, 450, 475, 490, 499]
DEFAULT_LENGTH_BINS = [(20, 64), (65, 128), (129, 192), (193, 256), (257, 320), (321, 384), (385, 448), (449, 500)]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(torch.sum(grad * grad).cpu())
    return float(np.sqrt(total))


def _grad_vector(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for parameter in parameters:
        if parameter.grad is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(parameter.grad.detach().float().cpu().reshape(-1))
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm() * b.norm())
    if denom <= 0.0:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _active_grad_vector(
    parameters: list[torch.nn.Parameter],
    loss: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    for parameter in parameters:
        parameter.grad = None
    loss.backward(retain_graph=retain_graph)
    vector = _grad_vector(parameters)
    for parameter in parameters:
        parameter.grad = None
    return vector


def _bin_label(value: int, edges: list[int]) -> str:
    previous = 0
    for edge in edges:
        if value <= edge:
            return f"{previous + 1}-{edge}"
        previous = edge
    return f"{previous + 1}+"


def _length_bin_label(bounds: tuple[int, int]) -> str:
    return f"{int(bounds[0])}-{int(bounds[1])}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_length_bins(value: str) -> list[tuple[int, int]]:
    bins: list[tuple[int, int]] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Length bin must be LOW-HIGH, got {token!r}")
        low, high = token.split("-", 1)
        start = int(low)
        end = int(high)
        if start > end:
            raise ValueError(f"Length bin start must be <= end, got {token!r}")
        bins.append((start, end))
    return bins


def _seed_for_observation(seed: int, *, timestep: int, length_bin_index: int, batch_index: int) -> int:
    return int(seed) + 1_000_003 * int(timestep) + 10_007 * int(length_bin_index) + int(batch_index)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device.type if device.type == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _load_profile_state(
    *,
    config: dict[str, Any],
    checkpoint: Path | None,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    DistanceMapDataset,
    torch.nn.Module,
    list[torch.nn.Parameter],
    GaussianDiffusion,
    Any,
    dict[str, Any],
    int,
]:
    normalization = json.loads(Path(config["normalization_file"]).read_text())
    dataset = DistanceMapDataset(config["train_manifest"], normalization)
    model = build_model_from_config(config["model"]).to(device)
    checkpoint_info: dict[str, Any] = {"path": None, "loaded": False}
    active_optimizer_step = 0
    if checkpoint is not None:
        loaded = load_checkpoint(checkpoint, map_location=device)
        model.load_state_dict(loaded["model"])
        active_optimizer_step = int(loaded.get("global_step", 0))
        checkpoint_info = {
            "path": str(checkpoint),
            "loaded": True,
            "epoch": loaded.get("epoch"),
            "global_step": loaded.get("global_step"),
        }
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100)))).to(device)
    prediction_type = prediction_parameterization_from_config(config)
    return normalization, dataset, model, parameters, diffusion, prediction_type, checkpoint_info, active_optimizer_step


def _profile_one(
    *,
    config: dict[str, Any],
    checkpoint: Path | None,
    candidate_weights: list[float],
    candidate_adjacent_weights: list[float],
    max_batches: int,
    subset_size: int,
    subsets_per_sample: int,
    adjacent_beta_angstrom: float,
    min_length: int | None,
    max_length: int | None,
    device: torch.device,
) -> dict[str, Any]:
    normalization, dataset, model, parameters, diffusion, prediction_type, checkpoint_info, active_optimizer_step = (
        _load_profile_state(config=config, checkpoint=checkpoint, device=device)
    )
    del active_optimizer_step
    if min_length is not None:
        dataset.frame = dataset.frame[dataset.frame["length"].astype(int) >= int(min_length)].reset_index(drop=True)
    if max_length is not None:
        dataset.frame = dataset.frame[dataset.frame["length"].astype(int) <= int(max_length)].reset_index(drop=True)
    if len(dataset) == 0:
        raise ValueError("No training samples remain after length filtering")
    downsample_stages = len(config["model"].get("channel_multipliers", [1, 2, 4, 8])) - 1
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_distance_maps(items, downsample_stages=downsample_stages),
    )
    aux_config = physical_auxiliary_config_from_mapping(
        {
            **config,
            "physical_auxiliary_loss_enabled": True,
            "edm_subset_size": subset_size,
            "edm_subsets_per_sample": subsets_per_sample,
        }
    )
    adjacent_config = adjacent_auxiliary_config_from_mapping(
        {
            **config,
            "adjacent_auxiliary_loss_enabled": True,
            "adjacent_auxiliary_huber_beta_angstrom": adjacent_beta_angstrom,
        }
    )

    torch.manual_seed(int(config.get("seed", 42)))
    rows = []
    for batch_index, batch in enumerate(tqdm(loader, desc="profile", total=min(max_batches, len(loader))), start=1):
        if batch_index > max_batches:
            break
        clean = batch["distance_matrices"].to(device)
        pair_mask = batch["pair_masks"].to(device)
        lengths = batch["lengths"].to(device)
        sep = batch["sequence_separation"].to(device)
        t = torch.randint(0, diffusion.timesteps, (clean.shape[0],), device=device)

        model.zero_grad(set_to_none=True)
        noisy, eps = diffusion.q_sample(clean, t, pair_mask)
        target = diffusion.training_target(x_start=clean, t=t, epsilon=eps, prediction_type=prediction_type)
        prediction = model(noisy, t, lengths, sep, pair_mask)
        diffusion_loss = masked_upper_triangular_loss(target.float(), prediction.float(), pair_mask)
        x0_hat, _ = diffusion.predict_x0_epsilon_from_model_output(
            x_t=noisy,
            t=t,
            model_output=prediction,
            prediction_type=prediction_type,
        )
        auxiliary = stochastic_edm_spectral_loss(
            x0_hat_normalized=x0_hat,
            pair_mask=pair_mask,
            lengths=lengths,
            normalization_scale=float(normalization.get("scale", 1.0)),
            config=aux_config,
            sample_ids=list(batch["sample_ids"]),
            optimizer_step=0,
            microbatch=batch_index,
        )
        adjacent = adjacent_chain_smooth_l1_loss(
            x0_hat_normalized=x0_hat,
            clean_normalized=clean,
            pair_mask=pair_mask,
            lengths=lengths,
            normalization_scale=float(normalization.get("scale", 1.0)),
            config=adjacent_config,
        )

        diffusion_loss.backward(retain_graph=True)
        diffusion_norm = _grad_norm(parameters)
        diffusion_grad = _grad_vector(parameters)
        model.zero_grad(set_to_none=True)
        auxiliary.loss.backward(retain_graph=True)
        auxiliary_norm = _grad_norm(parameters)
        auxiliary_grad = _grad_vector(parameters)
        model.zero_grad(set_to_none=True)
        adjacent.loss.backward()
        adjacent_norm = _grad_norm(parameters)
        adjacent_grad = _grad_vector(parameters)
        model.zero_grad(set_to_none=True)

        base = {
            "batch_index": batch_index,
            "min_length": int(lengths.min().detach().cpu()),
            "max_length": int(lengths.max().detach().cpu()),
            "length_bin": _bin_label(int(lengths.float().mean().detach().cpu()), [64, 128, 256, 384, 500]),
            "min_timestep": int(t.min().detach().cpu()),
            "max_timestep": int(t.max().detach().cpu()),
            "timestep_bin": _bin_label(int(t.float().mean().detach().cpu()), [99, 199, 299, 399, 499]),
            "diffusion_loss": float(diffusion_loss.detach().cpu()),
            "auxiliary_loss": float(auxiliary.loss.detach().cpu()),
            "auxiliary_negative_loss": float(auxiliary.negative_loss.detach().cpu()),
            "auxiliary_rank3_loss": float(auxiliary.rank3_loss.detach().cpu()),
            "adjacent_loss": float(adjacent.loss.detach().cpu()),
            "diffusion_grad_norm": diffusion_norm,
            "edm_spectral_grad_norm": auxiliary_norm,
            "adjacent_grad_norm": adjacent_norm,
            "cosine_diffusion_edm": _cosine(diffusion_grad, auxiliary_grad),
            "cosine_diffusion_adjacent": _cosine(diffusion_grad, adjacent_grad),
            "cosine_edm_adjacent": _cosine(auxiliary_grad, adjacent_grad),
            "edm_eligible_fraction": auxiliary.eligible_fraction,
            "edm_mean_subset_size": auxiliary.mean_subset_size,
            "edm_subset_count": auxiliary.subset_count,
            "adjacent_eligible_fraction": adjacent.eligible_fraction,
            "adjacent_eligible_pair_count": adjacent.eligible_pair_count,
        }
        for edm_weight in candidate_weights:
            weighted_edm_grad = float(edm_weight) * auxiliary_grad
            weighted_edm_ratio = (
                float(weighted_edm_grad.norm()) / diffusion_norm if diffusion_norm > 0.0 else float("nan")
            )
            for adjacent_weight in candidate_adjacent_weights:
                weighted_adjacent_grad = float(adjacent_weight) * adjacent_grad
                total_auxiliary_grad = weighted_edm_grad + weighted_adjacent_grad
                row = dict(base)
                row["candidate_edm_weight"] = float(edm_weight)
                row["candidate_adjacent_weight"] = float(adjacent_weight)
                row["weighted_edm_to_diffusion_grad_ratio"] = weighted_edm_ratio
                row["weighted_adjacent_to_diffusion_grad_ratio"] = (
                    float(weighted_adjacent_grad.norm()) / diffusion_norm if diffusion_norm > 0.0 else float("nan")
                )
                row["weighted_total_auxiliary_to_diffusion_grad_ratio"] = (
                    float(total_auxiliary_grad.norm()) / diffusion_norm if diffusion_norm > 0.0 else float("nan")
                )
                rows.append(row)
    reference_edm_weight = float(
        config.get("physical_auxiliary_loss_weight", candidate_weights[0] if candidate_weights else 0.0)
    )
    ratios_by_weight: dict[str, dict[str, float]] = {}
    for weight in candidate_adjacent_weights:
        ratios = [
            row["weighted_adjacent_to_diffusion_grad_ratio"]
            for row in rows
            if row["candidate_adjacent_weight"] == weight and row["candidate_edm_weight"] == reference_edm_weight
        ]
        finite = np.asarray([ratio for ratio in ratios if np.isfinite(ratio)], dtype=np.float64)
        ratios_by_weight[str(weight)] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    total_ratios_by_adjacent_weight: dict[str, dict[str, float]] = {}
    for weight in candidate_adjacent_weights:
        ratios = [
            row["weighted_total_auxiliary_to_diffusion_grad_ratio"]
            for row in rows
            if row["candidate_adjacent_weight"] == weight and row["candidate_edm_weight"] == reference_edm_weight
        ]
        finite = np.asarray([ratio for ratio in ratios if np.isfinite(ratio)], dtype=np.float64)
        total_ratios_by_adjacent_weight[str(weight)] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    return {
        "checkpoint": checkpoint_info,
        "max_batches": max_batches,
        "min_length_filter": min_length,
        "max_length_filter": max_length,
        "subset_size": subset_size,
        "subsets_per_sample": subsets_per_sample,
        "adjacent_beta_angstrom": adjacent_beta_angstrom,
        "candidate_weights": candidate_weights,
        "candidate_edm_weights": candidate_weights,
        "candidate_adjacent_weights": candidate_adjacent_weights,
        "rows": rows,
        "adjacent_ratios_by_weight": ratios_by_weight,
        "total_auxiliary_ratios_by_adjacent_weight": total_ratios_by_adjacent_weight,
    }


def _numeric_summary(values: pd.Series, *, bootstrap_iterations: int, seed: int) -> dict[str, float | int]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    count = int(array.size)
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
        }
    result: dict[str, float | int] = {
        "count": count,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "bootstrap_ci_low": float("nan"),
        "bootstrap_ci_high": float("nan"),
    }
    if count >= 2 and bootstrap_iterations > 0:
        rng = np.random.default_rng(int(seed))
        means = [float(np.mean(rng.choice(array, size=count, replace=True))) for _ in range(int(bootstrap_iterations))]
        result["bootstrap_ci_low"] = float(np.percentile(means, 2.5))
        result["bootstrap_ci_high"] = float(np.percentile(means, 97.5))
    return result


def _summarize_observations(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    metrics = [
        "edm_to_diffusion_grad_ratio",
        "adjacent_to_diffusion_grad_ratio",
        "total_auxiliary_to_diffusion_grad_ratio",
        "cosine_diffusion_edm",
        "cosine_diffusion_adjacent",
        "cosine_edm_adjacent",
        "cosine_diffusion_total_auxiliary",
        "diffusion_loss",
        "edm_loss",
        "adjacent_loss",
    ]
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "metric", "count", "mean", "median", "p10", "p90"])
    grouped = frame.groupby(group_columns, dropna=False) if group_columns else [((), frame)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        group_values = dict(zip(group_columns, key, strict=True))
        for metric in metrics:
            summary = _numeric_summary(group[metric], bootstrap_iterations=bootstrap_iterations, seed=seed)
            rows.append({**group_values, "metric": metric, **summary})
    return pd.DataFrame(rows)


def _write_heatmap(frame: pd.DataFrame, *, value_column: str, output: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No observations", ha="center", va="center")
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return
    pivot = frame.pivot_table(index="length_bin", columns="fixed_timestep", values=value_column, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Fixed timestep")
    ax.set_ylabel("Length bin")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(x)) for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_diagnostic_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# E003 Gradient-Interaction Diagnostic

This directory contains an analysis-only post-E003 diagnostic. It computes
separate gradients for diffusion loss, weighted stochastic EDM spectral loss,
and weighted adjacent-chain loss at fixed diffusion timesteps and length bins.
It does not perform optimizer steps, EMA updates, scheduler updates, sampling,
checkpoint writes, preprocessing, splitting, or ensemble evaluation.

Outputs:

- `observations.parquet`: one row per profiled batch, length bin, and fixed timestep.
- `summary_by_timestep.csv`: long-form summaries grouped by timestep.
- `summary_by_length.csv`: long-form summaries grouped by length bin.
- `summary_by_length_timestep.csv`: long-form summaries grouped by both.
- `diagnostic_summary.json`: run metadata and high-level conflict counts.
- `heatmap_cosine_edm_adjacent.png`: mean cosine between EDM and adjacent gradients.
- `heatmap_cosine_diffusion_adjacent.png`: mean cosine between diffusion and adjacent gradients.
- `heatmap_adjacent_diffusion_norm_ratio.png`: mean adjacent/diffusion gradient-norm ratio.

Interpretation guardrails:

- Negative cosine values indicate objective-gradient conflict for that batch.
- Large auxiliary/diffusion norm ratios indicate gradient-scale dominance, not
  necessarily directional conflict.
- Small observation counts are absence-of-evidence zones, not evidence of
  compatibility.

Observation count: `{summary["observation_count"]}`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _run_gradient_interaction_diagnostic(
    *,
    config: dict[str, Any],
    checkpoint: Path | None,
    output_dir: Path,
    fixed_timesteps: list[int],
    length_bins: list[tuple[int, int]],
    max_batches_per_group: int,
    subset_size: int,
    subsets_per_sample: int,
    adjacent_beta_angstrom: float,
    bootstrap_iterations: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    normalization, dataset, model, parameters, diffusion, prediction_type, checkpoint_info, checkpoint_step = (
        _load_profile_state(config=config, checkpoint=checkpoint, device=device)
    )
    if any(timestep < 0 or timestep >= diffusion.timesteps for timestep in fixed_timesteps):
        raise ValueError(f"Fixed timesteps must be in [0, {diffusion.timesteps - 1}]")
    downsample_stages = len(config["model"].get("channel_multipliers", [1, 2, 4, 8])) - 1
    aux_config = physical_auxiliary_config_from_mapping(
        {
            **config,
            "physical_auxiliary_loss_enabled": True,
            "edm_subset_size": subset_size,
            "edm_subsets_per_sample": subsets_per_sample,
            "physical_auxiliary_seed": seed,
        }
    )
    adjacent_config = adjacent_auxiliary_config_from_mapping(
        {
            **config,
            "adjacent_auxiliary_loss_enabled": True,
            "adjacent_auxiliary_huber_beta_angstrom": adjacent_beta_angstrom,
        }
    )
    optimizer_step = int(checkpoint_step)
    edm_weight = physical_auxiliary_weight(aux_config, optimizer_step)
    adj_weight = adjacent_auxiliary_weight(adjacent_config, optimizer_step)
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype = _amp_dtype(str(config.get("amp_dtype", "float16")))

    rows: list[dict[str, Any]] = []
    for length_bin_index, bounds in enumerate(length_bins):
        low, high = bounds
        frame = dataset.frame[
            (dataset.frame["length"].astype(int) >= int(low)) & (dataset.frame["length"].astype(int) <= int(high))
        ].reset_index(drop=True)
        if frame.empty:
            continue
        bin_dataset = copy.copy(dataset)
        bin_dataset.frame = frame
        loader = DataLoader(
            bin_dataset,
            batch_size=int(config.get("batch_size", 2)),
            shuffle=False,
            num_workers=0,
            collate_fn=lambda items: collate_distance_maps(items, downsample_stages=downsample_stages),
        )
        selected_batches = []
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > max_batches_per_group:
                break
            selected_batches.append((batch_index, batch))

        for fixed_timestep in fixed_timesteps:
            for batch_index, batch in selected_batches:
                observation_seed = _seed_for_observation(
                    seed, timestep=fixed_timestep, length_bin_index=length_bin_index, batch_index=batch_index
                )
                clean = batch["distance_matrices"].to(device)
                pair_mask = batch["pair_masks"].to(device)
                lengths = batch["lengths"].to(device)
                sep = batch["sequence_separation"].to(device)
                t = torch.full((clean.shape[0],), int(fixed_timestep), dtype=torch.long, device=device)
                with _autocast_context(enabled=amp_enabled, dtype=amp_dtype):
                    noisy, eps = diffusion.q_sample(
                        clean,
                        t,
                        pair_mask,
                        generator=_make_generator(device, observation_seed),
                    )
                    target = diffusion.training_target(
                        x_start=clean,
                        t=t,
                        epsilon=eps,
                        prediction_type=prediction_type,
                    )
                    prediction = model(noisy, t, lengths, sep, pair_mask)
                    diffusion_loss = masked_upper_triangular_loss(target.float(), prediction.float(), pair_mask)
                    x0_hat, _ = diffusion.predict_x0_epsilon_from_model_output(
                        x_t=noisy,
                        t=t,
                        model_output=prediction,
                        prediction_type=prediction_type,
                    )
                    auxiliary = stochastic_edm_spectral_loss(
                        x0_hat_normalized=x0_hat,
                        pair_mask=pair_mask,
                        lengths=lengths,
                        normalization_scale=float(normalization.get("scale", 1.0)),
                        config=aux_config,
                        sample_ids=list(batch["sample_ids"]),
                        optimizer_step=int(fixed_timestep),
                        microbatch=0,
                    )
                    adjacent = adjacent_chain_smooth_l1_loss(
                        x0_hat_normalized=x0_hat,
                        clean_normalized=clean,
                        pair_mask=pair_mask,
                        lengths=lengths,
                        normalization_scale=float(normalization.get("scale", 1.0)),
                        config=adjacent_config,
                    )
                    weighted_edm_loss = auxiliary.loss * float(edm_weight)
                    weighted_adjacent_loss = adjacent.loss * float(adj_weight)

                diffusion_grad = _active_grad_vector(parameters, diffusion_loss, retain_graph=True)
                edm_grad = _active_grad_vector(parameters, weighted_edm_loss, retain_graph=True)
                adjacent_grad = _active_grad_vector(parameters, weighted_adjacent_loss, retain_graph=False)
                total_aux_grad = edm_grad + adjacent_grad
                diffusion_norm = float(diffusion_grad.norm())
                edm_norm = float(edm_grad.norm())
                adjacent_norm = float(adjacent_grad.norm())
                total_aux_norm = float(total_aux_grad.norm())
                lengths_cpu = [int(item) for item in lengths.detach().cpu().tolist()]
                row = {
                    "length_bin": _length_bin_label(bounds),
                    "length_bin_start": int(low),
                    "length_bin_end": int(high),
                    "fixed_timestep": int(fixed_timestep),
                    "batch_index": int(batch_index),
                    "sample_ids": json.dumps([str(item) for item in batch["sample_ids"]]),
                    "actual_lengths": json.dumps(lengths_cpu),
                    "min_length": int(min(lengths_cpu)),
                    "max_length": int(max(lengths_cpu)),
                    "seed": int(observation_seed),
                    "optimizer_step_for_weights": int(optimizer_step),
                    "edm_weight": float(edm_weight),
                    "adjacent_weight": float(adj_weight),
                    "diffusion_loss": float(diffusion_loss.detach().cpu()),
                    "edm_loss": float(auxiliary.loss.detach().cpu()),
                    "edm_negative_loss": float(auxiliary.negative_loss.detach().cpu()),
                    "edm_rank3_loss": float(auxiliary.rank3_loss.detach().cpu()),
                    "weighted_edm_loss": float(weighted_edm_loss.detach().cpu()),
                    "adjacent_loss": float(adjacent.loss.detach().cpu()),
                    "weighted_adjacent_loss": float(weighted_adjacent_loss.detach().cpu()),
                    "diffusion_grad_norm": diffusion_norm,
                    "edm_grad_norm": edm_norm,
                    "adjacent_grad_norm": adjacent_norm,
                    "total_auxiliary_grad_norm": total_aux_norm,
                    "edm_to_diffusion_grad_ratio": edm_norm / diffusion_norm if diffusion_norm > 0.0 else float("nan"),
                    "adjacent_to_diffusion_grad_ratio": (
                        adjacent_norm / diffusion_norm if diffusion_norm > 0.0 else float("nan")
                    ),
                    "total_auxiliary_to_diffusion_grad_ratio": (
                        total_aux_norm / diffusion_norm if diffusion_norm > 0.0 else float("nan")
                    ),
                    "cosine_diffusion_edm": _cosine(diffusion_grad, edm_grad),
                    "cosine_diffusion_adjacent": _cosine(diffusion_grad, adjacent_grad),
                    "cosine_edm_adjacent": _cosine(edm_grad, adjacent_grad),
                    "cosine_diffusion_total_auxiliary": _cosine(diffusion_grad, total_aux_grad),
                    "edm_eligible_fraction": float(auxiliary.eligible_fraction),
                    "edm_mean_subset_size": float(auxiliary.mean_subset_size),
                    "edm_subset_count": int(auxiliary.subset_count),
                    "adjacent_eligible_fraction": float(adjacent.eligible_fraction),
                    "adjacent_eligible_pair_count": int(adjacent.eligible_pair_count),
                }
                rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    observations = pd.DataFrame(rows)
    observations.to_parquet(output_dir / "observations.parquet", index=False)
    _summarize_observations(
        observations,
        group_columns=["fixed_timestep"],
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    ).to_csv(output_dir / "summary_by_timestep.csv", index=False)
    _summarize_observations(
        observations,
        group_columns=["length_bin"],
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    ).to_csv(output_dir / "summary_by_length.csv", index=False)
    _summarize_observations(
        observations,
        group_columns=["length_bin", "fixed_timestep"],
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    ).to_csv(output_dir / "summary_by_length_timestep.csv", index=False)
    _write_heatmap(
        observations,
        value_column="cosine_edm_adjacent",
        output=output_dir / "heatmap_cosine_edm_adjacent.png",
        title="Mean cosine: EDM vs adjacent gradients",
    )
    _write_heatmap(
        observations,
        value_column="cosine_diffusion_adjacent",
        output=output_dir / "heatmap_cosine_diffusion_adjacent.png",
        title="Mean cosine: diffusion vs adjacent gradients",
    )
    _write_heatmap(
        observations,
        value_column="adjacent_to_diffusion_grad_ratio",
        output=output_dir / "heatmap_adjacent_diffusion_norm_ratio.png",
        title="Mean adjacent/diffusion gradient-norm ratio",
    )

    empty_groups = [
        {"length_bin": _length_bin_label(bounds), "fixed_timestep": int(timestep)}
        for bounds in length_bins
        for timestep in fixed_timesteps
        if observations.empty
        or observations[
            (observations["length_bin"] == _length_bin_label(bounds))
            & (observations["fixed_timestep"] == int(timestep))
        ].empty
    ]
    small_groups = []
    if not observations.empty:
        counts = observations.groupby(["length_bin", "fixed_timestep"]).size()
        small_groups = [
            {"length_bin": str(index[0]), "fixed_timestep": int(index[1]), "count": int(count)}
            for index, count in counts.items()
            if int(count) < 3
        ]
    summary = {
        "mode": "gradient_interaction_diagnostic",
        "checkpoint": checkpoint_info,
        "observation_count": int(len(observations)),
        "fixed_timesteps": [int(item) for item in fixed_timesteps],
        "length_bins": [_length_bin_label(bounds) for bounds in length_bins],
        "max_batches_per_group": int(max_batches_per_group),
        "seed": int(seed),
        "edm_weight": float(edm_weight),
        "adjacent_weight": float(adj_weight),
        "subset_size": int(subset_size),
        "subsets_per_sample": int(subsets_per_sample),
        "adjacent_beta_angstrom": float(adjacent_beta_angstrom),
        "negative_cosine_counts": {
            column: int((observations[column] < 0.0).sum()) if column in observations else 0
            for column in ["cosine_diffusion_edm", "cosine_diffusion_adjacent", "cosine_edm_adjacent"]
        },
        "large_auxiliary_gradient_ratio_threshold": 0.25,
        "large_auxiliary_gradient_ratio_counts": {
            column: int((observations[column] > 0.25).sum()) if column in observations else 0
            for column in [
                "edm_to_diffusion_grad_ratio",
                "adjacent_to_diffusion_grad_ratio",
                "total_auxiliary_to_diffusion_grad_ratio",
            ]
        },
        "empty_groups": empty_groups,
        "small_sample_groups": small_groups,
        "interpretation": {
            "negative_cosine": "objective-gradient conflict for the profiled batch",
            "large_auxiliary_gradient_magnitude": "scale dominance risk, not necessarily directional conflict",
            "small_sample_count": "absence of evidence, not evidence of compatibility",
        },
        "safety": {
            "optimizer_step_performed": False,
            "ema_update_performed": False,
            "scheduler_update_performed": False,
            "checkpoint_written": False,
            "sampling_performed": False,
        },
    }
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_diagnostic_readme(output_dir, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subsets-per-sample", type=int, default=1)
    parser.add_argument("--candidate-weights", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--candidate-adjacent-weights", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--adjacent-beta-angstrom", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gradient-interaction-diagnostic", action="store_true")
    parser.add_argument(
        "--diagnostic-output-dir",
        default="reports/experiments/E003_adjacent_chain_geometry/gradient_interaction_diagnostic",
    )
    parser.add_argument("--fixed-timesteps", default=",".join(str(item) for item in DEFAULT_FIXED_TIMESTEPS))
    parser.add_argument(
        "--length-bins",
        default=",".join(_length_bin_label(bounds) for bounds in DEFAULT_LENGTH_BINS),
    )
    parser.add_argument("--max-batches-per-group", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    parser.add_argument("--diagnostic-seed", type=int, default=3003)
    args = parser.parse_args()

    config = _load_yaml(args.config)
    config = copy.deepcopy(config)
    config["device"] = args.device
    device = torch.device(args.device)
    candidate_weights = [float(item) for item in args.candidate_weights.split(",") if item.strip()]
    candidate_adjacent_weights = [float(item) for item in args.candidate_adjacent_weights.split(",") if item.strip()]
    if args.gradient_interaction_diagnostic:
        summary = _run_gradient_interaction_diagnostic(
            config=config,
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            output_dir=Path(args.diagnostic_output_dir),
            fixed_timesteps=_parse_int_list(args.fixed_timesteps),
            length_bins=_parse_length_bins(args.length_bins),
            max_batches_per_group=int(args.max_batches_per_group),
            subset_size=int(args.subset_size),
            subsets_per_sample=int(args.subsets_per_sample),
            adjacent_beta_angstrom=float(args.adjacent_beta_angstrom),
            bootstrap_iterations=int(args.bootstrap_iterations),
            seed=int(args.diagnostic_seed),
            device=device,
        )
        print(f"Wrote diagnostic outputs to {args.diagnostic_output_dir}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --gradient-interaction-diagnostic is used")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _profile_one(
        config=config,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        candidate_weights=candidate_weights,
        candidate_adjacent_weights=candidate_adjacent_weights,
        max_batches=int(args.max_batches),
        subset_size=int(args.subset_size),
        subsets_per_sample=int(args.subsets_per_sample),
        adjacent_beta_angstrom=float(args.adjacent_beta_angstrom),
        min_length=args.min_length,
        max_length=args.max_length,
        device=device,
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
