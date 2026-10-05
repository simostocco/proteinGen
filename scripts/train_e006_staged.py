#!/usr/bin/env python3
"""Plan, calibrate, train, resume, or inspect authorized E006 stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.rich_codesign_production import (
    inspect_artifact,
    plan_continuation,
    plan_phase3,
    run_calibration,
    run_training_stage,
    verify_checkpoint_artifact,
)
from protein_distance_diffusion.training.stage_a_context_production import format_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan-only", choices=("calibrate", "sequence-pretrain", "joint-train"))
    modes.add_argument("--calibrate", action="store_true")
    modes.add_argument("--sequence-pretrain", action="store_true")
    modes.add_argument("--joint-train", action="store_true")
    modes.add_argument("--plan-continuation", action="store_true")
    modes.add_argument("--continue-from-checkpoint", action="store_true")
    modes.add_argument("--inspect", type=Path)
    modes.add_argument("--verify-checkpoint", type=Path)
    modes.add_argument("--monitor-trajectory", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--review-decision", type=Path)
    parser.add_argument("--expected-review-decision-sha256")
    args = parser.parse_args()
    if args.resume and not (args.sequence_pretrain or args.joint_train):
        parser.error("--resume is valid only with --sequence-pretrain or --joint-train")
    if bool(args.review_decision) != bool(args.expected_review_decision_sha256):
        parser.error("--review-decision and --expected-review-decision-sha256 must be supplied together")
    if args.review_decision and not args.resume:
        parser.error("--review-decision is valid only with --resume")
    if args.inspect:
        print(json.dumps(inspect_artifact(args.inspect), indent=2, sort_keys=True))
        return
    if args.monitor_trajectory:
        print(format_trajectory(args.monitor_trajectory))
        return
    if args.config is None:
        parser.error("--config is required for this mode")
    config = load_yaml(args.config)
    if args.plan_continuation:
        print(json.dumps(plan_continuation(config), indent=2, sort_keys=True))
        return
    if args.verify_checkpoint:
        if not args.expected_checkpoint_sha256:
            parser.error("--verify-checkpoint requires --expected-checkpoint-sha256")
        payload = verify_checkpoint_artifact(
            args.verify_checkpoint,
            args.expected_checkpoint_sha256,
            config=config,
        )
        print(json.dumps({"status": "verified", "stage": payload["stage"]}, indent=2))
        return
    if args.plan_only:
        print(json.dumps(plan_phase3(config, mode=args.plan_only), indent=2, sort_keys=True))
        return
    if args.calibrate:
        result = run_calibration(args.config)
    elif args.continue_from_checkpoint:
        result = run_training_stage(args.config, mode="sequence-pretrain", continuation=True)
    else:
        mode = "sequence-pretrain" if args.sequence_pretrain else "joint-train"
        result = run_training_stage(
            args.config,
            mode=mode,
            resume=args.resume,
            review_decision_path=args.review_decision,
            expected_review_decision_sha256=args.expected_review_decision_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
