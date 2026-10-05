#!/usr/bin/env python3
"""Plan or publish the E007 Phase 4A pretrained-prior audit."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_pretrained_sequence_prior import (
    plan_audit,
    plan_phase4b,
    publish_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--phase4b-plan-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan_audit(arguments.config)
    elif arguments.audit:
        result = publish_audit(arguments.config)
    else:
        result = plan_phase4b(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
