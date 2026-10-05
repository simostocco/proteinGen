#!/usr/bin/env python3
"""Plan or run the bounded E007 synthetic coordinate-learning smoke."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_smoke import plan_coordinate_smoke, run_coordinate_smoke


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--synthetic-smoke", action="store_true")
    arguments = parser.parse_args()
    result = (
        run_coordinate_smoke(arguments.config) if arguments.synthetic_smoke else plan_coordinate_smoke(arguments.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
