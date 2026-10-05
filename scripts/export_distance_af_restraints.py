#!/usr/bin/env python
"""Export Distance-AF-compatible CA distance restraints from one local matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protein_distance_diffusion.evaluation.distance_af import (
    restraints_to_frame,
    select_restraints,
    write_distance_af_restraints,
)


def _load_matrix(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    for key in ("physical_distance_matrix_angstrom", "distance_matrix", "raw_physical_distance_matrix_angstrom"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64)
    raise ValueError(f"No supported distance matrix key found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restraint-table", type=Path, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--matrix-source", default="generated_or_real_distance_matrix")
    parser.add_argument("--strategy", default="uniform_long_range")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--heldout-multiplier", type=int, default=1)
    parser.add_argument("--min-sequence-separation", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrix = _load_matrix(args.matrix_npz)
    restraints = restraints_to_frame(
        select_restraints(
            matrix,
            sample_id=args.sample_id or args.matrix_npz.stem,
            matrix_source=args.matrix_source,
            strategy=args.strategy,
            count=args.count,
            heldout_multiplier=args.heldout_multiplier,
            min_sequence_separation=args.min_sequence_separation,
            seed=args.seed,
        )
    )
    write_distance_af_restraints(args.output, restraints)
    if args.restraint_table is not None:
        args.restraint_table.parent.mkdir(parents=True, exist_ok=True)
        restraints.to_parquet(args.restraint_table, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
