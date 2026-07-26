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
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--trace-every", type=int, default=None)
    parser.add_argument("--trace-output", type=Path, default=None)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["config"]
    prediction_type = str(config.get("prediction_type", "epsilon"))
    if prediction_type != "epsilon":
        raise ValueError(f"Unsupported checkpoint prediction_type: {prediction_type}")
    model_cfg = dict(config["model"])
    if "channel_multipliers" in model_cfg:
        model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    if args.weights == "ema":
        if "ema" not in ckpt:
            raise ValueError("EMA weights requested but checkpoint does not contain an 'ema' state")
        model.load_state_dict(ckpt["ema"])
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cpu")
    factor = int(getattr(model, "downsample_factor", 1))
    side = ((args.length + factor - 1) // factor) * factor
    lengths = torch.full((args.num_samples,), args.length, dtype=torch.long)
    pair_mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config.get("diffusion_steps", 100))))
    sampled = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=pair_mask,
        sequence_separation=sep,
        device=device,
        generator=gen,
        prediction_type=prediction_type,
        trace_every=args.trace_every,
    )
    if args.trace_every is not None:
        samples, trace = sampled
    else:
        samples = sampled
        trace = None
    norm = ckpt["normalization"]
    scale = float(norm.get("scale", 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if trace is not None and args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    for i in range(args.num_samples):
        normalized = np.asarray(samples[i, 0, : args.length, : args.length].detach().cpu().tolist(), dtype=np.float32)
        physical = normalized * scale
        projected = 0.5 * (physical + physical.T)
        np.fill_diagonal(projected, 0.0)
        np.savez_compressed(
            args.output_dir / f"sample_{i:04d}.npz",
            requested_length=args.length,
            raw_normalized_matrix=normalized,
            raw_physical_distance_matrix_angstrom=physical,
            projected_physical_distance_matrix_angstrom=projected,
            random_seed=args.seed,
            checkpoint=str(args.checkpoint),
            weights=args.weights,
            sampling_config=json.dumps({"method": "ddpm", "prediction_type": prediction_type}),
        )
        save_heatmap(projected, args.output_dir / f"sample_{i:04d}.png", title=f"N={args.length} raw/projected")
    print(f"Wrote {args.num_samples} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
