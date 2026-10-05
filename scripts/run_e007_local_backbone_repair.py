#!/usr/bin/env python3
"""Plan or execute the bounded non-authorizing E007 Phase-3I.2 experiment."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_local_backbone_repair import (
    monitor_dynamic_preflight,
    monitor_local_backbone_pilot,
    plan_local_backbone_repair,
    run_dense_timestep_preflight,
    run_dynamic_memory_smoke,
    run_dynamic_stability_preflight,
    run_gradient_calibration,
    run_local_backbone_pilot,
    run_validate_calibration_panel,
    validate_coefficient_table,
    validate_dynamic_memory_smoke_contract,
    validate_dynamic_preflight_contract,
    validate_pilot_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--validate-coefficient-table", action="store_true")
    mode.add_argument("--validate-pilot-contract", action="store_true")
    mode.add_argument("--dense-timestep-preflight", action="store_true")
    mode.add_argument("--gradient-calibration-smoke", action="store_true")
    mode.add_argument("--validate-calibration-panel", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--lifecycle-smoke", action="store_true")
    mode.add_argument("--resume-pilot", action="store_true")
    mode.add_argument("--monitor-pilot", action="store_true")
    mode.add_argument("--validate-dynamic-preflight-contract", action="store_true")
    mode.add_argument("--monitor-dynamic-preflight", action="store_true")
    mode.add_argument("--dynamic-stability-preflight", action="store_true")
    mode.add_argument("--resume-dynamic-preflight", action="store_true")
    mode.add_argument("--dynamic-memory-smoke", action="store_true")
    mode.add_argument("--dynamic-multi-cell-smoke", action="store_true")
    mode.add_argument("--validate-dynamic-memory-smoke-contract", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan_local_backbone_repair(arguments.config)
    elif arguments.validate_coefficient_table:
        result = validate_coefficient_table(arguments.config)
    elif arguments.validate_pilot_contract:
        result = validate_pilot_contract(arguments.config)
    elif arguments.validate_dynamic_preflight_contract:
        result = validate_dynamic_preflight_contract(arguments.config)
    elif arguments.validate_dynamic_memory_smoke_contract:
        result = validate_dynamic_memory_smoke_contract(arguments.config)
    elif arguments.monitor_dynamic_preflight:
        result = monitor_dynamic_preflight(arguments.config)
    elif arguments.dynamic_memory_smoke:
        result = run_dynamic_memory_smoke(arguments.config)
    elif arguments.dynamic_multi_cell_smoke:
        result = run_dynamic_memory_smoke(arguments.config, multi_cell=True)
    elif arguments.dynamic_stability_preflight or arguments.resume_dynamic_preflight:
        result = run_dynamic_stability_preflight(arguments.config, resume=arguments.resume_dynamic_preflight)
    elif arguments.dense_timestep_preflight:
        result = run_dense_timestep_preflight(arguments.config)
    elif arguments.validate_calibration_panel:
        result = run_validate_calibration_panel(arguments.config)
    elif arguments.gradient_calibration_smoke:
        result = run_gradient_calibration(arguments.config)
    elif arguments.monitor_pilot:
        result = monitor_local_backbone_pilot(arguments.config)
    elif arguments.lifecycle_smoke:
        result = run_local_backbone_pilot(arguments.config, lifecycle_smoke=True)
    else:
        result = run_local_backbone_pilot(arguments.config, resume=arguments.resume_pilot)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
