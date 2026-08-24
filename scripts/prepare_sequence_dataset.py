#!/usr/bin/env python3
"""Validate leakage-safe sequence split manifests."""

from __future__ import annotations

import argparse

from protein_sequence_generation.config import load_yaml
from protein_sequence_generation.prepare import prepare_sequence_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/audit structure-unconditioned sequence manifests.")
    parser.add_argument("--config", required=True, help="YAML sequence data config.")
    args = parser.parse_args()
    summary = prepare_sequence_dataset(load_yaml(args.config))
    print(
        "Accepted sequences: train={train}, validation={validation}, test={test}".format(
            train=summary["splits"]["train"]["accepted_sequences"],
            validation=summary["splits"]["validation"]["accepted_sequences"],
            test=summary["splits"]["test"]["accepted_sequences"],
        )
    )


if __name__ == "__main__":
    main()
