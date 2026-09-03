#!/usr/bin/env python
"""Profile EDM auxiliary-loss gradient scale without optimizer updates."""

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
    max_batches: int,
    subset_size: int,
    subsets_per_sample: int,
    device: torch.device,
) -> dict[str, Any]:
    normalization = json.loads(Path(config["normalization_file"]).read_text())
    dataset = DistanceMapDataset(config["train_manifest"], normalization)
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

        diffusion_loss.backward(retain_graph=True)
        diffusion_norm = _grad_norm(parameters)
        model.zero_grad(set_to_none=True)
        auxiliary.loss.backward()
        auxiliary_norm = _grad_norm(parameters)
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
            "diffusion_grad_norm": diffusion_norm,
            "auxiliary_grad_norm": auxiliary_norm,
            "eligible_fraction": auxiliary.eligible_fraction,
            "mean_subset_size": auxiliary.mean_subset_size,
            "subset_count": auxiliary.subset_count,
        }
        for weight in candidate_weights:
            row = dict(base)
            row["candidate_weight"] = float(weight)
            row["weighted_aux_to_diffusion_grad_ratio"] = (
                float(weight) * auxiliary_norm / diffusion_norm if diffusion_norm > 0.0 else float("nan")
            )
            rows.append(row)
    ratios_by_weight: dict[str, dict[str, float]] = {}
    for weight in candidate_weights:
        ratios = [row["weighted_aux_to_diffusion_grad_ratio"] for row in rows if row["candidate_weight"] == weight]
        finite = np.asarray([ratio for ratio in ratios if np.isfinite(ratio)], dtype=np.float64)
        ratios_by_weight[str(weight)] = {
            "median": float(np.median(finite)) if finite.size else float("nan"),
            "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
            "max": float(np.max(finite)) if finite.size else float("nan"),
        }
    return {
        "checkpoint": checkpoint_info,
        "max_batches": max_batches,
        "subset_size": subset_size,
        "subsets_per_sample": subsets_per_sample,
        "candidate_weights": candidate_weights,
        "rows": rows,
        "ratios_by_weight": ratios_by_weight,
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = _load_yaml(args.config)
    config = copy.deepcopy(config)
    config["device"] = args.device
    device = torch.device(args.device)
    candidate_weights = [float(item) for item in args.candidate_weights.split(",") if item.strip()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _profile_one(
        config=config,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        candidate_weights=candidate_weights,
        max_batches=int(args.max_batches),
        subset_size=int(args.subset_size),
        subsets_per_sample=int(args.subsets_per_sample),
        device=device,
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
