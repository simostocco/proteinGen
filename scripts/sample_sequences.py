#!/usr/bin/env python3
"""Sample exact-length protein sequences from a sequence Transformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from protein_sequence_generation.sampling import generate_sequences, load_model_for_sampling, write_fasta_and_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample exact-length canonical protein sequences.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--num-sequences", required=True, type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model, vocabulary, _ = load_model_for_sampling(args.checkpoint, weights=args.weights)
    sequences = generate_sequences(
        model,
        vocabulary=vocabulary,
        length=args.length,
        num_sequences=args.num_sequences,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        device=args.device,
    )
    fasta, metadata = write_fasta_and_jsonl(
        sequences=sequences,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        length=args.length,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    print(f"Wrote {fasta}")
    print(f"Wrote {metadata}")


if __name__ == "__main__":
    main()
