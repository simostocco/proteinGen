#!/usr/bin/env python3
"""Build leakage-safe split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.splitting import build_splits_from_files


def main() -> None:
    """Run split construction."""
    parser = argparse.ArgumentParser(description="Assign sequence clusters to train/validation/test splits.")
    parser.add_argument("--config", required=True, type=Path, help="YAML split config.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    build_splits_from_files(
        cfg["manifest_path"],
        cfg["cluster_assignments_path"],
        cfg["output_dir"],
        seed=int(cfg.get("split_seed", 42)),
        fractions=(
            float(cfg.get("train_fraction", 0.8)),
            float(cfg.get("validation_fraction", 0.1)),
            float(cfg.get("test_fraction", 0.1)),
        ),
    )
    print(f"Wrote split files to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
