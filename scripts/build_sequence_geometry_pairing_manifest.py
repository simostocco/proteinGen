#!/usr/bin/env python3
"""Build a versioned sequence-geometry pairing dataset from completed audit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.data.pairing_builder import (
    build_sequence_geometry_pairing,
    validate_sequence_geometry_pairing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--normalization-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-memory-mib", type=int, default=4096)
    parser.add_argument("--checkpoint-frequency", type=int, default=1000)
    parser.add_argument("--maximum-failure-examples", type=int, default=100)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--expected-eligible", type=int)
    parser.add_argument("--expected-excluded", type=int)
    parser.add_argument(
        "--eligibility-policy",
        choices=("strict", "practical", "all_with_status"),
        default="practical",
    )
    parser.add_argument(
        "--allow-pilot-evidence",
        action="store_true",
        help="Allow diagnostic preview output from a completed bounded raw pilot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        if args.validation_report is None:
            raise SystemExit("--validate-only requires --validation-report")
        if args.output_dir is not None:
            raise SystemExit("--validate-only does not accept --output-dir")
        report = validate_sequence_geometry_pairing(
            processed_manifest=args.processed_manifest,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            audit_dir=args.audit_dir,
            normalization_file=args.normalization_file,
            validation_report=args.validation_report,
            eligibility_policy=args.eligibility_policy,
            allow_pilot_evidence=args.allow_pilot_evidence,
            expected_total=args.expected_total,
            expected_eligible=args.expected_eligible,
            expected_excluded=args.expected_excluded,
            state_dir=args.state_dir,
            resume=args.resume,
            batch_size=args.batch_size,
            max_memory_mib=args.max_memory_mib,
            checkpoint_frequency=args.checkpoint_frequency,
            maximum_failure_examples=args.maximum_failure_examples,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["status"] == "passed" else 1)
    if args.output_dir is None:
        raise SystemExit("normal build mode requires --output-dir")
    protocol = build_sequence_geometry_pairing(
        processed_manifest=args.processed_manifest,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        audit_dir=args.audit_dir,
        normalization_file=args.normalization_file,
        output_dir=args.output_dir,
        eligibility_policy=args.eligibility_policy,
        allow_pilot_evidence=args.allow_pilot_evidence,
        state_dir=args.state_dir,
        resume=args.resume,
        batch_size=args.batch_size,
        max_memory_mib=args.max_memory_mib,
        checkpoint_frequency=args.checkpoint_frequency,
        maximum_failure_examples=args.maximum_failure_examples,
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
