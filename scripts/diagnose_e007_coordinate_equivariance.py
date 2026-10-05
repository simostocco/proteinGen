#!/usr/bin/env python3
"""Run the bounded, no-update E007 coordinate-equivariance diagnostic."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_coordinate_equivariance import (
    plan_equivariance_diagnostic,
    run_equivariance_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run-diagnostic", action="store_true")
    arguments = parser.parse_args()
    result = (
        plan_equivariance_diagnostic(arguments.config)
        if arguments.plan_only
        else run_equivariance_diagnostic(arguments.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
