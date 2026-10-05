#!/usr/bin/env python3
"""Phase 3I.4 bounded calibration, holdout, and read-only lifecycle CLI."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_phase3i4_sampler_correction import (
    candidates,
    execute,
    load_config,
    monitor,
    panels,
    plan,
    resume,
    validate_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e007_phase3i4_sampler_correction_v1.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--validate-contract", action="store_true")
    mode.add_argument("--monitor", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan(args.config)
    elif args.validate_contract:
        result = validate_contract(args.config)
    elif args.monitor:
        result = monitor(args.config)
    elif args.calibrate:
        result = execute(args.config, "calibration")
    elif args.holdout:
        result = execute(args.config, "holdout")
    elif args.resume:
        result = resume(args.config)
    else:
        cfg = load_config(args.config)
        calibration, holdout = panels(cfg)
        result = {"calibration": calibration, "holdout": holdout, "candidates": candidates(cfg)}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
