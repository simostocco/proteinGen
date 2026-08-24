"""Autoregressive sampling for the structure-unconditioned sequence model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from protein_sequence_generation.checkpointing import load_checkpoint
from protein_sequence_generation.model import ProteinSequenceTransformer
from protein_sequence_generation.utils import atomic_write_text, seed_everything
from protein_sequence_generation.vocabulary import CANONICAL_AMINO_ACIDS, ProteinVocabulary


def validate_sampling_args(*, length: int, temperature: float, top_k: int | None, top_p: float | None) -> None:
    """Validate sampling controls."""
    if length <= 0:
        raise ValueError("length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or null")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")


def filter_logits(logits: torch.Tensor, *, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Apply top-k and nucleus filtering to one logits vector."""
    filtered = logits.clone()
    if top_k is not None and top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered = filtered.masked_fill(filtered < threshold, torch.finfo(filtered.dtype).min)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[sorted_idx[remove]] = torch.finfo(filtered.dtype).min
    return filtered


@torch.no_grad()
def generate_sequences(
    model: ProteinSequenceTransformer,
    *,
    vocabulary: ProteinVocabulary,
    length: int,
    num_sequences: int,
    seed: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    greedy: bool = False,
    device: torch.device | str = "cpu",
) -> list[str]:
    """Generate exactly N canonical amino-acid sequences."""
    validate_sampling_args(length=length, temperature=temperature, top_k=top_k, top_p=top_p)
    if num_sequences <= 0:
        raise ValueError("num_sequences must be positive")
    device = torch.device(device)
    model.eval().to(device)
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = torch.full((num_sequences, 1), vocabulary.bos_id, dtype=torch.long, device=device)
    lengths = torch.full((num_sequences,), length, dtype=torch.long, device=device)
    allowed = torch.full((len(vocabulary.tokens),), False, dtype=torch.bool, device=device)
    allowed[vocabulary.canonical_ids] = True
    for _ in range(length):
        attention_mask = torch.ones_like(generated, dtype=torch.bool)
        logits = model(input_ids=generated, lengths=lengths, attention_mask=attention_mask)[:, -1, :]
        logits = logits.masked_fill(~allowed[None, :], torch.finfo(logits.dtype).min)
        logits = logits / temperature
        next_tokens: list[torch.Tensor] = []
        for row in range(num_sequences):
            row_logits = filter_logits(logits[row], top_k=top_k, top_p=top_p)
            if greedy:
                token = torch.argmax(row_logits).view(1)
            else:
                probs = torch.softmax(row_logits, dim=-1)
                token = torch.multinomial(probs, num_samples=1, generator=generator)
            next_tokens.append(token)
        generated = torch.cat([generated, torch.stack(next_tokens).to(device)], dim=1)
    sequences = [vocabulary.decode(row[1:].detach().cpu().tolist()) for row in generated]
    canonical = set(CANONICAL_AMINO_ACIDS)
    for sequence in sequences:
        if len(sequence) != length or any(residue not in canonical for residue in sequence):
            raise RuntimeError("Generated sequence failed length/canonical validation")
    return sequences


def load_model_for_sampling(
    checkpoint_path: str | Path, *, weights: str = "model"
) -> tuple[ProteinSequenceTransformer, ProteinVocabulary, dict[str, Any]]:
    """Load model, vocabulary, and checkpoint payload."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    vocabulary = ProteinVocabulary(tokens=tuple(checkpoint["vocabulary"]["tokens"]))
    model = ProteinSequenceTransformer(checkpoint["model_config"])
    state_key = "ema" if weights == "ema" and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key])
    return model, vocabulary, checkpoint


def write_fasta_and_jsonl(
    *,
    sequences: list[str],
    output_dir: str | Path,
    checkpoint: str | Path,
    length: int,
    seed: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> tuple[Path, Path]:
    """Write generated sequences and metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(checkpoint).name
    fasta_lines: list[str] = []
    jsonl_lines: list[str] = []
    for index, sequence in enumerate(sequences):
        fasta_lines.append(
            f">sample={index} length={length} seed={seed} temperature={temperature} top_p={top_p} "
            f"top_k={top_k} checkpoint={checkpoint_name}"
        )
        fasta_lines.append(sequence)
        jsonl_lines.append(
            json.dumps(
                {
                    "sample_index": index,
                    "sequence": sequence,
                    "requested_length": length,
                    "seed": seed,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "checkpoint": str(checkpoint),
                },
                sort_keys=True,
            )
        )
    fasta_path = out / "sequences.fasta"
    jsonl_path = out / "metadata.jsonl"
    atomic_write_text(fasta_path, "\n".join(fasta_lines) + "\n")
    atomic_write_text(jsonl_path, "\n".join(jsonl_lines) + "\n")
    return fasta_path, jsonl_path
