#!/usr/bin/env python3
"""Validate a diffusion checkpoint and generated distance matrices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import collate_distance_maps
from protein_distance_diffusion.data.dataset import DistanceMapDataset
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.evaluation.metrics import (
    generated_matrix_report,
)
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.trainer import validate_loss


def main() -> None:
    """Write checkpoint and optional generated-matrix validation report."""
    parser = argparse.ArgumentParser(description="Validate denoising/generative distance-map outputs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--matrix", type=Path, default=None, help="Optional generated .npz matrix to inspect.")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/metrics.json"))
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    config = load_yaml(args.config)
    checkpoint_config = ckpt["config"]
    normalization = ckpt.get("normalization") or json.loads(Path(config["normalization_file"]).read_text())
    model_cfg = dict(checkpoint_config["model"])
    if "channel_multipliers" in model_cfg:
        model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    if args.weights == "ema":
        if "ema" not in ckpt:
            raise ValueError("EMA weights requested but checkpoint does not contain EMA state")
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    device = torch.device("cpu")
    model.to(device)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(checkpoint_config.get("diffusion_steps", 100)))).to(device)
    dataset = DistanceMapDataset(config["validation_manifest"], normalization)
    factor = int(getattr(model, "downsample_factor", 1))
    downsample_stages = int(math.log2(factor))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=lambda items: collate_distance_maps(items, downsample_stages=downsample_stages),
    )
    validation = validate_loss(model, diffusion, loader, device=device, seed=int(config.get("seed", 42)))
    valid_pair_count = int(sum(int(length) * (int(length) - 1) // 2 for length in dataset.frame["length"]))
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "weights": args.weights,
        "prediction_type": checkpoint_config.get("prediction_type", "epsilon"),
        "validation_loss": validation,
        "validation_sample_count": int(len(dataset)),
        "validation_valid_pair_count": valid_pair_count,
    }
    if args.matrix:
        data = np.load(args.matrix)
        key = "raw_normalized_matrix" if "raw_normalized_matrix" in data else "normalized_matrix"
        report["generated_matrix"] = generated_matrix_report(data[key], scale=float(normalization.get("scale", 1.0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
