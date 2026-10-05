#!/usr/bin/env python3
"""Plan or run the E007 Phase-3H checkpoint sampling replication."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_coordinate_checkpoint_sampling_replication import (
    plan_checkpoint_sampling_replication,
    run_checkpoint_sampling_replication,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        if arguments.resume:
            parser.error("--resume is valid only with --audit")
        result = plan_checkpoint_sampling_replication(arguments.config)
    else:
        result = run_checkpoint_sampling_replication(arguments.config, resume=arguments.resume)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
