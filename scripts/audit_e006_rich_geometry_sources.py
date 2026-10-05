#!/usr/bin/env python3
"""Run the bounded, read-only E006 rich-geometry source audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.evaluation.e006_geometry_source_audit import run_e006_geometry_source_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--inspect", type=Path, help="Read an existing protocol instead of running the audit")
    args = parser.parse_args()
    if args.inspect is not None:
        report = json.loads(args.inspect.read_text())
        fields = {
            key: report.get(key)
            for key in (
                "status",
                "schema_version",
                "panel_selection",
                "panel_observed_eligibility_counts",
                "full_corpus_eligibility_projection",
                "eligibility_manifest",
                "phase1_authorization",
                "dataset_inputs_unchanged",
                "current_rss_mib",
                "peak_rss_mib",
                "elapsed_seconds",
                "failure",
            )
        }
        print(json.dumps(fields, indent=2, sort_keys=True))
        return
    if args.config is None:
        parser.error("--config is required unless --inspect is used")
    run_e006_geometry_source_audit(args.config)


if __name__ == "__main__":
    main()
