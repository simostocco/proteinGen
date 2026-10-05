#!/usr/bin/env python3
"""Plan or inspect the Phase-3A E007 coordinate generator; training is unavailable."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_plan import generator_plan, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--inspect-architecture", action="store_true")
    arguments = parser.parse_args()
    plan = generator_plan(load_yaml(arguments.config))
    plan["mode"] = "inspect_architecture" if arguments.inspect_architecture else "plan_only"
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
