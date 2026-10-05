#!/usr/bin/env python3
"""Plan or run bounded, non-authorizing E006 Stage-A v5 evidence harnesses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.training.stage_a_context_smoke import (
    comparison_preflight,
    plan_loader_smoke,
    plan_synthetic_smoke,
    run_comparison_pilot,
    run_loader_smoke,
    run_synthetic_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-smoke", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    mode.add_argument("--plan-loader-smoke", action="store_true")
    mode.add_argument("--loader-smoke", action="store_true")
    mode.add_argument("--plan-comparison-pilot", action="store_true")
    mode.add_argument("--comparison-pilot", action="store_true")
    args = parser.parse_args()
    if args.plan_smoke:
        result = plan_synthetic_smoke(args.config)
    elif args.synthetic_smoke:
        result = run_synthetic_smoke(args.config)
    elif args.plan_loader_smoke:
        result = plan_loader_smoke(args.config)
    elif args.loader_smoke:
        result = run_loader_smoke(args.config)
    elif args.plan_comparison_pilot:
        result = comparison_preflight(args.config)
    else:
        result = run_comparison_pilot(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
