#!/usr/bin/env python3
"""Run the strictly bounded E005 co-design integration harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.codesign import MemoryStageReporter, run_codesign_dry_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--maximum-length", type=int)
    parser.add_argument("--max-memory-mib", type=int)
    args = parser.parse_args()
    reporter = MemoryStageReporter(args.report, max_memory_mib=args.max_memory_mib or 4096)
    reporter.record("imports_startup")
    try:
        config = load_yaml(args.config)
        reporter.record("configuration_load", config_path=str(args.config))
        run_codesign_dry_run(
            config,
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            real_data=args.real_data,
            steps=args.steps,
            sample_count=args.sample_count,
            maximum_length=args.maximum_length,
            max_memory_mib=args.max_memory_mib,
            reporter=reporter,
            tiny_overfit=args.tiny_overfit,
        )
    except BaseException as error:
        if reporter.payload.get("status") != "incomplete":
            reporter.incomplete(error)
        raise


if __name__ == "__main__":
    main()
