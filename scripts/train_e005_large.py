#!/usr/bin/env python3
"""Launch-gate, train, resume, or evaluate the definitive E005-Large model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.codesign import _configuration_sha256
from protein_distance_diffusion.training.codesign_large import (
    _dataset_identity,
    evaluate_large,
    run_large_launch_gate,
    run_large_training,
    validate_large_config,
    validate_resume_checkpoint,
)
from protein_distance_diffusion.training.codesign_pilot import _atomic_json, _model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--launch-gate", action="store_true")
    parser.add_argument("--launch-gate-report", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--synthetic", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_yaml(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    if args.launch_gate:
        if not args.report:
            parser.error("--launch-gate requires --report")
        run_large_launch_gate(config, report_path=args.report, synthetic=args.synthetic)
        return
    if args.evaluate_only:
        if not args.checkpoint or not args.report:
            parser.error("--evaluate-only requires --checkpoint and --report")
        validate_large_config(config, synthetic=args.synthetic)
        device = torch.device(config["device"])
        model = _model(config).to(device)
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)
        plan_path = output_dir / "training_plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"E005-Large training plan is missing: {plan_path}")
        plan = json.loads(plan_path.read_text())
        identity = _dataset_identity(config, synthetic=args.synthetic)
        validate_resume_checkpoint(
            checkpoint,
            config_hash=_configuration_sha256(config),
            dataset_hash=identity["sha256"],
            plan_hash=plan["sha256"],
        )
        model.load_state_dict(checkpoint["model"])
        result = evaluate_large(
            config,
            model,
            plan,
            device=device,
            synthetic=args.synthetic,
            optimizer_step=int(checkpoint["optimizer_step"]),
        )
        identity_after = _dataset_identity(config, synthetic=args.synthetic)
        if identity_after != identity:
            raise RuntimeError("dataset_mutation_detected")
        result["dataset_identity_before"] = identity
        result["dataset_identity_after"] = identity_after
        result["dataset_inputs_unchanged"] = True
        _atomic_json(args.report, result)
        return
    if not args.launch_gate_report:
        parser.error("training requires --launch-gate-report")
    run_large_training(
        config,
        launch_gate_report=args.launch_gate_report,
        output_dir=output_dir,
        resume=args.resume,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
