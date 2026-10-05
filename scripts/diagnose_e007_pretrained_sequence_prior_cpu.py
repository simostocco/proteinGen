#!/usr/bin/env python3
"""Plan or run the E007 Phase 4B.2 CPU-only real-artifact diagnostic."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_pretrained_cpu_diagnostic import plan, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    result = plan(arguments.config) if arguments.plan_only else run(arguments.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
