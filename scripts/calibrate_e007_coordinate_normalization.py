#!/usr/bin/env python3
"""Plan or execute E007 Phase-3E-A coordinate normalization."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.data.e007_coordinate_normalization import (
    calibrate_coordinate_normalization,
    plan_coordinate_normalization,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    arguments = parser.parse_args()
    result = (
        calibrate_coordinate_normalization(arguments.config)
        if arguments.calibrate
        else plan_coordinate_normalization(arguments.config)
    )
    summary = result
    if arguments.calibrate:
        statistics = result["statistics"]
        summary = {
            "status": result["status"],
            "output_dir": result["output_dir"],
            "coordinate_scale_angstrom": statistics["coordinate_scale_angstrom"],
            "candidate_sample_count": statistics["candidate_sample_count"],
            "accepted_sample_count": statistics["accepted_sample_count"],
            "rejected_sample_count": statistics["rejected_sample_count"],
            "protected_inputs_unchanged": result["protected_inputs_unchanged"],
            **{key: result[key] for key in result if key.startswith("authorizes_")},
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
