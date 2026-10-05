#!/usr/bin/env python3
"""Run the production-bounded paired-arm E005 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.codesign_pilot import finalize_completed_heartbeat, run_bounded_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--finalize-heartbeat", action="store_true")
    parser.add_argument("--stage-updates", help="Comma-separated optimizer updates overriding each stage")
    parser.add_argument("--max-rss-mib", type=int)
    parser.add_argument("--max-cuda-memory-mib", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.stage_updates:
        updates = [int(value) for value in args.stage_updates.split(",")]
        stages = config.setdefault("pilot", {}).get("curriculum")
        if stages is None or len(updates) != len(stages):
            parser.error("--stage-updates must contain one value per configured curriculum stage")
        for stage, value in zip(stages, updates, strict=True):
            stage["optimizer_updates"] = value
    if args.max_rss_mib is not None:
        config.setdefault("pilot", {})["max_rss_mib"] = args.max_rss_mib
    if args.max_cuda_memory_mib is not None:
        config.setdefault("pilot", {})["max_cuda_memory_mib"] = args.max_cuda_memory_mib
    output_dir = args.output_dir or Path(config["output_dir"])
    if args.finalize_heartbeat:
        if args.resume:
            parser.error("--finalize-heartbeat and --resume are mutually exclusive")
        finalize_completed_heartbeat(config, output_dir=output_dir, synthetic=args.synthetic)
        return
    run_bounded_pilot(config, output_dir=output_dir, synthetic=args.synthetic, resume=args.resume)


if __name__ == "__main__":
    main()
