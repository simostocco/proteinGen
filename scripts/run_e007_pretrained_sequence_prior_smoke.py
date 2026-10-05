#!/usr/bin/env python3
"""Acquire, verify, or smoke E007 Phase 4B pretrained sequence priors."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_pretrained_sequence_prior_smoke import (
    acquire_artifacts,
    plan,
    resume,
    run_smoke,
    validate_loader_metadata,
    verify_artifacts_offline,
    verify_environment_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--acquire-artifacts", action="store_true")
    mode.add_argument("--verify-artifacts-offline", action="store_true")
    mode.add_argument("--verify-environment", action="store_true")
    mode.add_argument("--validate-loader-metadata", action="store_true")
    mode.add_argument("--run-smoke", action="store_true")
    mode.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan(arguments.config)
    elif arguments.acquire_artifacts:
        result = acquire_artifacts(arguments.config)
    elif arguments.verify_artifacts_offline:
        result = verify_artifacts_offline(arguments.config)
    elif arguments.verify_environment:
        result = verify_environment_readiness(arguments.config)
    elif arguments.validate_loader_metadata:
        result = validate_loader_metadata(arguments.config)
    elif arguments.run_smoke:
        result = run_smoke(arguments.config)
    else:
        result = resume(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
