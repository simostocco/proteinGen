#!/usr/bin/env python3
"""Plan or run the bounded synthetic E007 Phase-3C paired pilot."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_objective_pilot import (
    plan_objective_correction_pilot,
    run_objective_correction_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run-pilot", action="store_true")
    arguments = parser.parse_args()
    result = (
        run_objective_correction_pilot(arguments.config)
        if arguments.run_pilot
        else plan_objective_correction_pilot(arguments.config)
    )
    summary = result
    if arguments.run_pilot:
        summary = {
            "status": result["status"],
            "classification": result["classification"],
            "output_dir": result["output_dir"],
            "protected_inputs_unchanged": result["protected_inputs_unchanged"],
            **{key: result[key] for key in result if key.startswith("authorizes_")},
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
