#!/usr/bin/env python3
"""Compute training-only normalization statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.data.statistics import write_normalization


def main() -> None:
    """Run normalization-statistics computation."""
    parser = argparse.ArgumentParser(description="Compute scale-only normalization from train split only.")
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()
    write_normalization(args.train_manifest, args.output, percentile=args.percentile)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
