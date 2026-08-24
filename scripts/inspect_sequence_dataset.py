#!/usr/bin/env python3
"""Inspect a sequence manifest without reading structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from protein_sequence_generation.dataset import load_sequence_manifest, validate_records
from protein_sequence_generation.utils import atomic_write_json
from protein_sequence_generation.vocabulary import ProteinVocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a protein sequence manifest or FASTA file.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--allow-unknown", action="store_true")
    args = parser.parse_args()
    frame = load_sequence_manifest(args.manifest)
    _, audit = validate_records(
        frame,
        source_path=args.manifest,
        vocabulary=ProteinVocabulary(),
        min_length=args.min_length,
        max_length=args.max_length,
        allow_unknown=args.allow_unknown,
    )
    output = args.output_dir / "sequence_dataset_summary.json"
    atomic_write_json(output, audit)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
