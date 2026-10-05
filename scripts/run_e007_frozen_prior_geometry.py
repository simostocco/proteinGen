#!/usr/bin/env python3
"""Plan or execute the bounded E007 Phase 4C experiment."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_frozen_prior_geometry import plan, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--loader-forward-backward-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--resume-pilot", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan(arguments.config)
    elif arguments.loader_forward_backward_smoke:
        result = run(arguments.config, mode="smoke")
    else:
        result = run(arguments.config, mode="pilot", resume=arguments.resume_pilot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
