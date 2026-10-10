"""Persistent E015 training entry point. No full t=0 sampling implementation."""

import argparse
import json

import e015_bootstrap

from protein_distance_diffusion.training.e015_runtime import Runtime, e007_active
from protein_distance_diffusion.training.e015_trainer import Trainer

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runtime = Runtime(CONFIG)
    if e007_active(runtime.cfg):
        raise RuntimeError("E007 active")
    calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
    certificate = json.loads(runtime.path("preflight/full_certification.json").read_text())
    if not certificate["passed"] or certificate["calibration_contract"] != runtime.contract(calibration):
        raise RuntimeError("Mandatory CUDA certification is stale or failed")
    if not args.smoke_updates:
        for n in (100, 1000):
            if not runtime.path(f"preflight/smoke_{n}/PASSED.json").exists():
                raise RuntimeError("Mandatory TRAIN smoke incomplete")
            passed = json.loads(runtime.path(f"preflight/smoke_{n}/PASSED.json").read_text())
            if not passed["passed"] or passed["contract"] != runtime.contract(calibration):
                raise RuntimeError("Mandatory TRAIN smoke is stale or failed")
        if not runtime.path("preflight/full_certification.json").exists():
            raise RuntimeError("Resume/symmetry/full preflight incomplete")
    trainer = Trainer(runtime, "smoke" if args.smoke_updates else "production", args.smoke_updates, args.resume)
    try:
        trainer.run()
    except Exception as exc:
        # A real write/numerical failure stops only E015. No speculative cleanup.
        evidence = dict(
            phase="failed_closed",
            error=str(exc),
            precision=trainer.engine.precision,
            update=trainer.update,
            exposures=trainer.exposures,
            disk_reserve_gate=False,
        )
        for action in (
            lambda: trainer.save(f"failure_update_{trainer.update}"),
            lambda: runtime.write("FAILURE.json", evidence),
        ):
            try:
                action()
            except Exception as write_error:
                print(json.dumps(dict(failure=str(exc), evidence_write_error=str(write_error))), flush=True)
        raise


if __name__ == "__main__":
    main()
