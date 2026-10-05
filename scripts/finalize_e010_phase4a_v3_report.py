#!/usr/bin/env python3
"""Rebuild Phase 4A v3 reports from existing JSON adjudication only."""

import argparse
import json
from pathlib import Path

from scripts.e010_phase4a_v3_reporting import finalize_from_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    report = finalize_from_json(args.staging)
    print(json.dumps({"status": "report_finalized", "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
