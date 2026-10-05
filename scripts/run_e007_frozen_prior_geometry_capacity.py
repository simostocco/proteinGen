#!/usr/bin/env python3
"""Plan, execute, resume, or monitor E007 Phase 4C.1."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_frozen_prior_geometry_capacity import (
    monitor,
    plan,
    run,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--loader-forward-backward-smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--resume-pilot", action="store_true")
    mode.add_argument("--monitor-smoke", action="store_true")
    mode.add_argument("--monitor-pilot", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan(arguments.config)
    elif arguments.loader_forward_backward_smoke:
        result = run(arguments.config, mode="smoke")
    elif arguments.pilot or arguments.resume_pilot:
        result = run(arguments.config, mode="pilot", resume=arguments.resume_pilot)
    else:
        result = monitor(arguments.config, mode="smoke" if arguments.monitor_smoke else "pilot")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
