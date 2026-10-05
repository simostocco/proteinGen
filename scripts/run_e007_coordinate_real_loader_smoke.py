#!/usr/bin/env python3
"""Plan or run one bounded E007 Phase-3E-B real-data smoke."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.training.e007_coordinate_real_loader_smoke import (
    plan_real_loader_smoke,
    run_real_loader_smoke,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--loader-smoke", action="store_true")
    mode.add_argument("--forward-backward-smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.plan_only:
        result = plan_real_loader_smoke(arguments.config)
        summary = result
    else:
        selected_mode = "loader-smoke" if arguments.loader_smoke else "forward-backward-smoke"
        result = run_real_loader_smoke(arguments.config, mode=selected_mode)
        summary = {
            "status": result["status"],
            "mode": result["mode"],
            "classification": result["classification"],
            "output_dir": result["output_dir"],
            "protected_inputs_unchanged": result["protected_inputs_unchanged"],
            "optimizer_updates": result["optimizer_updates"],
            **{key: result[key] for key in result if key.startswith("authorizes_")},
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
