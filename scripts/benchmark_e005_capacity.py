#!/usr/bin/env python3
"""Run isolated E005 read-only capacity benchmark cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.capacity_benchmark import (
    DEFAULT_LENGTHS,
    DEFAULT_MODES,
    run_capacity_benchmark,
    run_capacity_case,
    write_failed_worker_report,
)


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--lengths", type=_csv_ints, default=DEFAULT_LENGTHS)
    parser.add_argument("--modes", type=_csv_strings, default=DEFAULT_MODES)
    parser.add_argument("--seed", type=int, default=5005)
    parser.add_argument("--max-rss-mib", type=int, default=4096)
    parser.add_argument("--max-cuda-memory-mib", type=int, default=8192)
    parser.add_argument("--case-timeout-seconds", type=int, default=900)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--worker-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--target-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mode", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_case:
        if args.worker_report is None or args.target_length is None or args.mode is None:
            parser.error("worker mode requires --worker-report, --target-length, and --mode")
        try:
            result = run_capacity_case(
                load_yaml(args.config),
                target_length=args.target_length,
                mode=args.mode,
                seed=args.seed,
                max_rss_mib=args.max_rss_mib,
                max_cuda_memory_mib=args.max_cuda_memory_mib,
                synthetic=args.synthetic,
            )
            from protein_distance_diffusion.training.capacity_benchmark import _atomic_json

            _atomic_json(args.worker_report, result)
        except BaseException as error:
            write_failed_worker_report(
                args.worker_report,
                target_length=args.target_length,
                mode=args.mode,
                error=error,
            )
            raise
        return
    if args.report is None:
        parser.error("--report is required")
    run_capacity_benchmark(
        config_path=args.config,
        report_path=args.report,
        lengths=args.lengths,
        modes=args.modes,
        seed=args.seed,
        max_rss_mib=args.max_rss_mib,
        max_cuda_memory_mib=args.max_cuda_memory_mib,
        timeout_seconds=args.case_timeout_seconds,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
