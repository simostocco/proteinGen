#!/usr/bin/env python3
"""Publish the read-only E007 Phase-3F decision-parity audit."""

from __future__ import annotations

import argparse
import json

from protein_distance_diffusion.evaluation.e007_phase3f_decision_parity import publish_decision_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="store_true", required=True)
    arguments = parser.parse_args()
    print(json.dumps(publish_decision_audit(arguments.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
