"""Persistent E015 training entry point. No full t=0 sampling implementation."""

import argparse

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
    if not args.smoke_updates:
        for n in (100, 1000):
            if not runtime.path(f"preflight/smoke_{n}/PASSED.json").exists():
                raise RuntimeError("Mandatory TRAIN smoke incomplete")
        if not runtime.path("preflight/full_certification.json").exists():
            raise RuntimeError("Resume/symmetry/full preflight incomplete")
    Trainer(runtime, "smoke" if args.smoke_updates else "production", args.smoke_updates, args.resume).run()


if __name__ == "__main__":
    main()
