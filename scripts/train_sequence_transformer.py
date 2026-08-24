#!/usr/bin/env python3
"""Train the standalone sequence-only protein Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_sequence_generation.config import load_yaml
from protein_sequence_generation.training import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train p(S | N) with a causal protein sequence Transformer.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--unsafe-resume-override", action="store_true")
    args = parser.parse_args()
    checkpoint = train_from_config(
        load_yaml(args.config),
        resume_from=args.resume_from,
        max_optimizer_steps=args.max_optimizer_steps,
        unsafe_resume_override=args.unsafe_resume_override,
    )
    print(f"Wrote sequence checkpoint {checkpoint}")


if __name__ == "__main__":
    main()
