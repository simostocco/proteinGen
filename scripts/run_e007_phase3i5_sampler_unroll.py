#!/usr/bin/env python3
"""Phase 3I.5 bounded sampler-unroll plan and lifecycle controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.training.e007_phase3i5_contract import plan, validate_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e007_phase3i5_sampler_unroll_v1.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--validate-contract", action="store_true")
    modes.add_argument("--cuda-memory-smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--monitor", action="store_true")
    modes.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan(args.config)
    elif args.validate_contract:
        result = validate_contract(args.config)
    elif args.monitor:
        import yaml

        staging = Path(yaml.safe_load(Path(args.config).read_text())["staging_dir"])
        journal = staging / "failure.json"
        result = (
            json.loads(journal.read_text()) if journal.is_file() else {"status": "no_run", "staging_dir": str(staging)}
        )
    else:
        from protein_distance_diffusion.training.e007_phase3i5_sampler_unroll import execute

        result = execute(
            args.config,
            "cuda-memory-smoke" if args.cuda_memory_smoke else "resume" if args.resume else "execute",
            args.resume,
        )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
