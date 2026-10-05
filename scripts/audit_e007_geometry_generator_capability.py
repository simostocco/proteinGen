#!/usr/bin/env python3
"""Plan or run the read-only E007 Phase-3I capability audit."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_geometry_generator_capability import (
    audit_geometry_generator_capability,
    plan_geometry_generator_capability,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    arguments = parser.parse_args()
    result = (
        plan_geometry_generator_capability(arguments.config)
        if arguments.plan_only
        else audit_geometry_generator_capability(arguments.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
