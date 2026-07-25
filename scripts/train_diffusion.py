#!/usr/bin/env python3
"""Train the conditional distance-map diffusion model."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.trainer import train_from_config


def main() -> None:
    """Run training."""
    parser = argparse.ArgumentParser(description="Train conditional U-Net DDPM on distance matrices.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_optimizer_steps is not None:
        config["max_optimizer_steps"] = args.max_optimizer_steps
    if args.resume_from is not None:
        config["resume_from"] = str(args.resume_from)
    ckpt = train_from_config(config)
    print(f"Wrote checkpoint {ckpt}")


if __name__ == "__main__":
    main()
