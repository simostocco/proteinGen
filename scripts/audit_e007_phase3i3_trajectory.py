#!/usr/bin/env python3
"""Phase 3I.3 trajectory audit entrypoint."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import (
    audit,
    monitor,
    plan,
    validate_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e007_denoiser_sampler_trajectory_audit_v1.yaml")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--validate-contract", action="store_true")
    modes.add_argument("--monitor", action="store_true")
    modes.add_argument("--audit", action="store_true")
    modes.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan(args.config)
    elif args.validate_contract:
        result = validate_contract(args.config)
    elif args.monitor:
        result = monitor(args.config)
    else:
        result = audit(args.config, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
