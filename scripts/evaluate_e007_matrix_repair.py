#!/usr/bin/env python3
"""Plan or execute the non-authorizing E007 matrix-repair audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.evaluation.e007_matrix_repair import build_repair_plan, run_repair_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_repair_plan(args.config)
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(run_repair_audit(args.config, plan=plan))


if __name__ == "__main__":
    main()
