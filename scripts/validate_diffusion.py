#!/usr/bin/env python3
"""Validate a checkpoint and write a compact metrics report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.evaluation.metrics import basic_identity_metrics


def main() -> None:
    """Write a placeholder validation report for generated sample directories."""
    parser = argparse.ArgumentParser(description="Validate denoising/generative distance-map outputs.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--matrix", type=Path, default=None, help="Optional generated .npz matrix to inspect.")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/metrics.json"))
    args = parser.parse_args()
    report = {"checkpoint": str(args.checkpoint), "config": str(args.config)}
    if args.matrix:
        import numpy as np

        data = np.load(args.matrix)
        report["basic_identity_metrics"] = basic_identity_metrics(data["physical_distance_matrix_angstrom"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
