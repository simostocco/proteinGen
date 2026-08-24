#!/usr/bin/env python3
"""Evaluate a sequence Transformer checkpoint on held-out sequence manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protein_sequence_generation.checkpointing import load_checkpoint
from protein_sequence_generation.dataset import ProteinSequenceDataset, parse_fasta
from protein_sequence_generation.evaluation import (
    evaluate_generated_sequences,
    evaluate_language_model,
    write_evaluation,
)
from protein_sequence_generation.model import ProteinSequenceTransformer
from protein_sequence_generation.vocabulary import ProteinVocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out protein sequence language-model metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--generated-fasta", type=Path, default=None)
    parser.add_argument("--training-manifest", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    vocabulary = ProteinVocabulary(tokens=tuple(checkpoint["vocabulary"]["tokens"]))
    model = ProteinSequenceTransformer(checkpoint["model_config"])
    state = checkpoint["ema"] if args.weights == "ema" and "ema" in checkpoint else checkpoint["model"]
    model.load_state_dict(state)
    data_cfg = checkpoint["training_config"].get("data", {})
    dataset = ProteinSequenceDataset(
        args.manifest,
        vocabulary=vocabulary,
        min_length=int(data_cfg.get("min_length", 20)),
        max_length=int(data_cfg.get("max_length", checkpoint["model_config"].get("max_length", 500))),
    )
    metrics = evaluate_language_model(model, dataset, batch_size=args.batch_size, device=args.device)
    output = args.output_dir / "language_model_metrics.json"
    write_evaluation(output, metrics)
    print(f"Wrote {output}")
    if args.generated_fasta is not None:
        if args.training_manifest is None:
            raise ValueError("--training-manifest is required with --generated-fasta")
        generated = parse_fasta(args.generated_fasta)["sequence"].astype(str).tolist()
        generated_metrics = evaluate_generated_sequences(
            generated,
            training_manifest=args.training_manifest,
            vocabulary=vocabulary,
            min_length=1,
            max_length=int(data_cfg.get("max_length", checkpoint["model_config"].get("max_length", 500))),
        )
        generated_output = args.output_dir / "generated_sequence_metrics.json"
        write_evaluation(generated_output, generated_metrics)
        print(f"Wrote {generated_output}")


if __name__ == "__main__":
    main()
