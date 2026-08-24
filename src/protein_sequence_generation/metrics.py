"""Losses and lightweight sequence quality metrics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from protein_sequence_generation.vocabulary import CANONICAL_AMINO_ACIDS


def sequence_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    reduction: str = "sequence_mean",
) -> torch.Tensor:
    """Compute masked next-token cross entropy.

    `sequence_mean` averages tokens within each protein first, then proteins.
    `token_mean` returns the standard token-weighted language-model NLL.
    """
    if reduction not in {"sequence_mean", "token_mean"}:
        raise ValueError("loss_reduction must be 'sequence_mean' or 'token_mean'")
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction="none")
    losses = losses.view_as(target_ids)
    mask = attention_mask.to(losses.dtype)
    if reduction == "token_mean":
        return (losses * mask).sum() / mask.sum().clamp_min(1.0)
    per_sequence = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return per_sequence.mean()


@torch.no_grad()
def language_model_metrics(
    logits: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, Any]:
    """Return token NLL, sequence loss, perplexity, and accuracy."""
    token_nll = sequence_cross_entropy(logits, target_ids, attention_mask, reduction="token_mean")
    seq_loss = sequence_cross_entropy(logits, target_ids, attention_mask, reduction="sequence_mean")
    predictions = logits.argmax(dim=-1)
    mask = attention_mask.bool()
    correct = (predictions == target_ids) & mask
    return {
        "token_nll": float(token_nll.detach().cpu()),
        "sequence_loss": float(seq_loss.detach().cpu()),
        "perplexity": float(torch.exp(token_nll.detach()).cpu()),
        "token_accuracy": float(correct.float().sum().cpu() / mask.float().sum().clamp_min(1.0).cpu()),
        "valid_token_count": int(mask.sum().cpu()),
        "sequence_count": int(target_ids.shape[0]),
    }


def amino_acid_frequencies(sequences: list[str]) -> dict[str, float]:
    """Return normalized amino-acid frequencies across sequences."""
    counts: Counter[str] = Counter()
    for sequence in sequences:
        counts.update(sequence)
    total = sum(counts.values())
    if total == 0:
        return {aa: 0.0 for aa in CANONICAL_AMINO_ACIDS}
    return {aa: counts.get(aa, 0) / total for aa in CANONICAL_AMINO_ACIDS}


def jensen_shannon_divergence(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence in nats for two amino-acid distributions."""
    p = np.array([left.get(aa, 0.0) for aa in CANONICAL_AMINO_ACIDS], dtype=np.float64)
    q = np.array([right.get(aa, 0.0) for aa in CANONICAL_AMINO_ACIDS], dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        valid = a > 0
        return float(np.sum(a[valid] * np.log(a[valid] / b[valid])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def max_homopolymer_run(sequence: str) -> int:
    """Return the longest identical-residue run."""
    best = 0
    current = 0
    previous = None
    for residue in sequence:
        current = current + 1 if residue == previous else 1
        previous = residue
        best = max(best, current)
    return best


def shannon_entropy(sequence: str) -> float:
    """Return per-sequence residue entropy in bits."""
    if not sequence:
        return 0.0
    counts = Counter(sequence)
    total = len(sequence)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def repeated_substring_fraction(sequence: str, *, k: int = 3) -> float:
    """Return fraction of k-mers that are repeats beyond their first occurrence."""
    if len(sequence) < k:
        return 0.0
    kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    return 1.0 - (len(set(kmers)) / len(kmers))


def generated_sequence_metrics(generated: list[str], *, training_sequences: list[str]) -> dict[str, Any]:
    """Summarize generated sequence validity, uniqueness, novelty, and composition."""
    train_set = set(training_sequences)
    canonical = set(CANONICAL_AMINO_ACIDS)
    valid = [all(residue in canonical for residue in sequence) for sequence in generated]
    unique = len(set(generated))
    entropies = [shannon_entropy(sequence) for sequence in generated]
    homopolymers = [max_homopolymer_run(sequence) for sequence in generated]
    repeat_fracs = [repeated_substring_fraction(sequence) for sequence in generated]
    generated_freq = amino_acid_frequencies(generated)
    training_freq = amino_acid_frequencies(training_sequences)
    return {
        "sequence_count": len(generated),
        "canonical_valid_fraction": float(np.mean(valid)) if valid else 0.0,
        "exact_unique_count": unique,
        "duplicate_fraction": 1.0 - unique / max(len(generated), 1),
        "exact_novel_fraction": float(np.mean([sequence not in train_set for sequence in generated]))
        if generated
        else 0.0,
        "amino_acid_frequencies": generated_freq,
        "training_amino_acid_frequencies": training_freq,
        "amino_acid_js_divergence": jensen_shannon_divergence(generated_freq, training_freq),
        "average_shannon_entropy_bits": float(np.mean(entropies)) if entropies else 0.0,
        "low_complexity_fraction": float(np.mean([entropy < 2.0 for entropy in entropies])) if entropies else 0.0,
        "max_homopolymer_run": int(max(homopolymers)) if homopolymers else 0,
        "mean_repeated_3mer_fraction": float(np.mean(repeat_fracs)) if repeat_fracs else 0.0,
        "nearest_exact_match_count": int(sum(sequence in train_set for sequence in generated)),
    }
