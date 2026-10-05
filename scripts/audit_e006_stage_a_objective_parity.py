#!/usr/bin/env python3
"""Run the bounded read-only E006 Stage-A v5 objective-parity audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.evaluation.e006_stage_a_objective_parity import (
    published_artifacts,
    run_objective_parity_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_objective_parity_audit(args.config)
    output = Path(load_yaml(args.config)["output_dir"])
    artifacts = published_artifacts(output)
    if artifacts:
        result = {**result, "published_artifacts": artifacts}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
