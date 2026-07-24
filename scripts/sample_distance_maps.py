#!/usr/bin/env python3
"""Sample generated distance matrices from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.evaluation.plots import save_heatmap
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint


def main() -> None:
    """Run DDPM sampling."""
    parser = argparse.ArgumentParser(description="Generate unconditional distance matrices conditioned on N.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["config"]
    model_cfg = dict(config["model"])
    if "channel_multipliers" in model_cfg:
        model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    model.load_state_dict(ckpt.get("ema", ckpt["model"]))
    model.eval()
    device = torch.device("cpu")
    side = args.length
    factor = 2 ** (len(model_cfg.get("channel_multipliers", (1, 2, 4, 8))) - 1)
    side = ((side + factor - 1) // factor) * factor
    lengths = torch.full((args.num_samples,), args.length, dtype=torch.long)
    pair_mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100))))
    samples = sample_ddpm(
        model, diffusion, lengths=lengths, pair_mask=pair_mask, sequence_separation=sep, device=device, generator=gen
    )
    norm = ckpt["normalization"]
    scale = float(norm.get("scale", 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        normalized = samples[i, 0, : args.length, : args.length].cpu().numpy()
        physical = normalized * scale
        unclamped = physical.copy()
        clamped = np.clip(physical, 0.0, None)
        np.savez_compressed(
            args.output_dir / f"sample_{i:04d}.npz",
            requested_length=args.length,
            normalized_matrix=normalized,
            physical_distance_matrix_angstrom=physical,
            unclamped_matrix_angstrom=unclamped,
            clamped_matrix_angstrom=clamped,
            random_seed=args.seed,
            checkpoint=str(args.checkpoint),
            sampling_config=json.dumps({"method": "ddpm"}),
        )
        save_heatmap(clamped, args.output_dir / f"sample_{i:04d}.png", title=f"N={args.length}")
    print(f"Wrote {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
