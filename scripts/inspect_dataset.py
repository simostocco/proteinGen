#!/usr/bin/env python3
"""Inspect a processed manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.data.preprocess import load_manifest


def main() -> None:
    """Run dataset inspection."""
    parser = argparse.ArgumentParser(description="Create simple dataset summary tables.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    frame = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.describe(include="all").to_csv(args.output_dir / "manifest_summary.csv")
    print(f"Wrote {args.output_dir / 'manifest_summary.csv'}")


if __name__ == "__main__":
    main()
