#!/usr/bin/env python3
"""Plan or run the bounded E007 Phase-3F real-data pilot."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_real_pilot import (
    plan_real_coordinate_pilot,
    run_real_coordinate_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run-pilot", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        if arguments.resume_from:
            parser.error("--resume-from is valid only with --run-pilot")
        result = plan_real_coordinate_pilot(arguments.config)
    else:
        result = run_real_coordinate_pilot(arguments.config, resume_from=arguments.resume_from)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
