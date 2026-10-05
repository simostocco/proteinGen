#!/usr/bin/env python3
"""Plan or run non-authorizing E006 Stage-A contextual objective v6 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.training.stage_a_context_v6_smoke import (
    comparison_preflight_v6,
    plan_synthetic_smoke_v6,
    run_comparison_pilot_v6,
    run_synthetic_smoke_v6,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-smoke", action="store_true")
    modes.add_argument("--synthetic-smoke", action="store_true")
    modes.add_argument("--plan-comparison-pilot", action="store_true")
    modes.add_argument("--comparison-pilot", action="store_true")
    args = parser.parse_args()
    if args.plan_smoke:
        result = plan_synthetic_smoke_v6(args.config)
    elif args.synthetic_smoke:
        result = run_synthetic_smoke_v6(args.config)
    elif args.plan_comparison_pilot:
        result = comparison_preflight_v6(args.config)
    else:
        result = run_comparison_pilot_v6(args.config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
