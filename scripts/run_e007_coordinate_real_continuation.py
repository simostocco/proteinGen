#!/usr/bin/env python3
"""Plan or run the exact-state E007 Phase-3F continuation."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_real_continuation import (
    plan_coordinate_continuation,
    run_allocator_diagnostic,
    run_coordinate_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--diagnostic-report")
    parser.add_argument(
        "--diagnostic-cycles",
        type=int,
        default=2,
        help="Additional repetitions of the observed five-stratum allocator lifecycle",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run-continuation", action="store_true")
    mode.add_argument("--allocator-diagnostic", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        if arguments.resume_from:
            parser.error("--resume-from is valid only with --run-continuation")
        result = plan_coordinate_continuation(arguments.config)
    elif arguments.run_continuation:
        if arguments.diagnostic_report:
            parser.error("--diagnostic-report is valid only with --allocator-diagnostic")
        result = run_coordinate_continuation(arguments.config, resume_from=arguments.resume_from)
    else:
        if arguments.resume_from:
            parser.error("--resume-from is valid only with --run-continuation")
        if not arguments.diagnostic_report:
            parser.error("--allocator-diagnostic requires --diagnostic-report")
        result = run_allocator_diagnostic(
            arguments.config,
            report_path=arguments.diagnostic_report,
            cycles=arguments.diagnostic_cycles,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
