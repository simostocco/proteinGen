#!/usr/bin/env python3
"""Plan or run the read-only E006 Stage-A contextual-learning diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.evaluation.e006_stage_a_context import (
    plan_diagnostic,
    run_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = plan_diagnostic(args.config) if args.plan_only else run_diagnostic(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
