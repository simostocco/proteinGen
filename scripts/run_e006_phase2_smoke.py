#!/usr/bin/env python3
"""Plan or run bounded, non-authorizing E006 Phase-2 smoke workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_smoke import plan_e006_smoke, run_e006_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--loader-smoke", action="store_true")
    modes.add_argument("--train-smoke", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(plan_e006_smoke(load_yaml(args.config)), indent=2, sort_keys=True))
        return
    mode = "loader-smoke" if args.loader_smoke else "train-smoke"
    print(json.dumps(run_e006_smoke(args.config, mode=mode), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
