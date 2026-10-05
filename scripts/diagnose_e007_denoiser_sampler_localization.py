#!/usr/bin/env python3
"""Plan or execute the non-authorizing E007 Phase-3I.1 diagnostic."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_denoiser_sampler_localization import (
    diagnose_denoiser_sampler_localization,
    monitor_denoiser_sampler_localization,
    plan_denoiser_sampler_localization,
    validate_panel_schema,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--diagnose", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--monitor", action="store_true")
    mode.add_argument("--validate-panel-schema", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan_denoiser_sampler_localization(arguments.config)
    elif arguments.monitor:
        result = monitor_denoiser_sampler_localization(arguments.config)
    elif arguments.validate_panel_schema:
        result = validate_panel_schema(arguments.config)
    else:
        result = diagnose_denoiser_sampler_localization(arguments.config, resume=arguments.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
