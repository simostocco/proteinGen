#!/usr/bin/env python3
"""Plan or read-only validate the E007 Phase-3I.2 v3 calibration."""

import argparse
import json

from protein_distance_diffusion.training.e007_local_backbone_calibration_v3 import (
    plan_only,
    run_calibration,
    validate_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--validate-panel", action="store_true")
    mode.add_argument("--gradient-calibration", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        result = plan_only(
            args.config,
            "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v2/report.json",
        )
    elif args.validate_panel:
        result, _ = validate_panel(args.config)
    else:
        result = run_calibration(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
