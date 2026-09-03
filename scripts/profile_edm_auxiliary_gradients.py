#!/usr/bin/env python
"""Profile physical auxiliary-loss gradient scale without optimizer updates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
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
    adjacent_chain_smooth_l1_loss,
    physical_auxiliary_config_from_mapping,
    stochastic_edm_spectral_loss,
)
from protein_distance_diffusion.training.trainer import build_model_from_config


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


def _bin_label(value: int, edges: list[int]) -> str:
    previous = 0
    for edge in edges:
        if value <= edge:
            return f"{previous + 1}-{edge}"
        previous = edge
    return f"{previous + 1}+"


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
    normalization = json.loads(Path(config["normalization_file"]).read_text())
    dataset = DistanceMapDataset(config["train_manifest"], normalization)
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
    model = build_model_from_config(config["model"]).to(device)
    checkpoint_info: dict[str, Any] = {"path": None, "loaded": False}
    if checkpoint is not None:
        loaded = load_checkpoint(checkpoint, map_location=device)
        model.load_state_dict(loaded["model"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subsets-per-sample", type=int, default=1)
    parser.add_argument("--candidate-weights", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--candidate-adjacent-weights", default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--adjacent-beta-angstrom", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    config = copy.deepcopy(config)
    config["device"] = args.device
    device = torch.device(args.device)
    candidate_weights = [float(item) for item in args.candidate_weights.split(",") if item.strip()]
    candidate_adjacent_weights = [float(item) for item in args.candidate_adjacent_weights.split(",") if item.strip()]
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
