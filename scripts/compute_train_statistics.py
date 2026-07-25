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
    parser.add_argument("--histogram-bin-width-angstrom", type=float, default=0.05)
    parser.add_argument("--histogram-max-distance-angstrom", type=float, default=2000.0)
    parser.add_argument("--overflow-fraction-tolerance", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    write_normalization(
        args.train_manifest,
        args.output,
        percentile=args.percentile,
        histogram_bin_width_angstrom=args.histogram_bin_width_angstrom,
        histogram_max_distance_angstrom=args.histogram_max_distance_angstrom,
        overflow_fraction_tolerance=args.overflow_fraction_tolerance,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        restart=args.restart,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
