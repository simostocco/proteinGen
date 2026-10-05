#!/usr/bin/env python3
"""Run the bounded paper-quality E005-Large diagnostic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.evaluation.codesign_diagnostic import run_codesign_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--inspect", type=Path, help="Print a compact summary from an existing protocol")
    args = parser.parse_args()
    if args.inspect:
        report = json.loads(args.inspect.read_text())
        compact = {
            key: report.get(key)
            for key in (
                "status",
                "checkpoint_sha256",
                "optimizer_step",
                "sample_level_record_count",
                "geometry_diagnostic_record_count",
                "elapsed_seconds",
                "memory",
                "validation_trajectory",
                "primary_comparisons",
                "failure",
            )
        }
        print(json.dumps(compact, indent=2, sort_keys=True))
        return
    if args.config is None:
        parser.error("--config is required unless --inspect is used")
    run_codesign_diagnostic(args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
