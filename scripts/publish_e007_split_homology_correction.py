#!/usr/bin/env python3
"""Plan or publish the E007 Phase-3E-C.1 integrity correction."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_split_homology_publication import (
    plan_publication_correction,
    publish_correction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--publish", action="store_true")
    arguments = parser.parse_args()
    result = (
        plan_publication_correction(arguments.config) if arguments.plan_only else publish_correction(arguments.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
