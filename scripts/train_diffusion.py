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
    args = parser.parse_args()
    ckpt = train_from_config(load_yaml(args.config))
    print(f"Wrote checkpoint {ckpt}")


if __name__ == "__main__":
    main()
