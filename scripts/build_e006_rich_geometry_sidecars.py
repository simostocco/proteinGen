#!/usr/bin/env python3
"""Plan, construct, resume, verify, or inspect E006 Phase-1 sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.data.rich_geometry_sidecars import (
    construct_sidecars,
    run_performance_benchmark,
    verify_sidecar_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--verify-only", type=Path)
    modes.add_argument("--inspect", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--maximum-rss-mib",
        type=float,
        help="Runtime-only Phase-1 RSS ceiling; excluded from the construction configuration hash.",
    )
    parser.add_argument("--require-training-authorization", action="store_true")
    args = parser.parse_args()
    if args.require_training_authorization and not args.verify_only:
        parser.error("--require-training-authorization is valid only with --verify-only")
    if args.maximum_rss_mib is not None and (args.inspect or args.verify_only or args.benchmark):
        parser.error("--maximum-rss-mib is valid only with --plan-only, --pilot, or --full")
    if args.inspect:
        print(json.dumps(json.loads(args.inspect.read_text()), indent=2, sort_keys=True))
        return
    if args.verify_only:
        print(
            json.dumps(
                verify_sidecar_dataset(
                    args.verify_only,
                    require_training_authorization=args.require_training_authorization,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.config is None:
        parser.error("--config is required for planning or construction")
    if args.resume and args.plan_only:
        parser.error("--resume is not valid with --plan-only")
    if args.benchmark:
        print(json.dumps(run_performance_benchmark(args.config, resume=args.resume), indent=2, sort_keys=True))
        return
    mode = "plan-only" if args.plan_only else "pilot" if args.pilot else "full"
    result = construct_sidecars(
        args.config,
        mode=mode,
        resume=args.resume,
        maximum_rss_mib=args.maximum_rss_mib,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
