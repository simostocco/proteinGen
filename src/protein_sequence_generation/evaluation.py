"""Evaluation helpers for held-out and generated protein sequences."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from protein_sequence_generation.collate import collate_sequences
from protein_sequence_generation.dataset import ProteinSequenceDataset, load_sequence_manifest, validate_records
from protein_sequence_generation.metrics import generated_sequence_metrics, language_model_metrics
from protein_sequence_generation.utils import atomic_write_json
from protein_sequence_generation.vocabulary import ProteinVocabulary


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    dataset: ProteinSequenceDataset,
    *,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
    show_progress: bool = True,
) -> dict[str, Any]:
    """Evaluate token NLL, perplexity, accuracy, and diagnostic slices."""
    device = torch.device(device)
    model.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda items: collate_sequences(items, vocabulary=dataset.vocabulary),
    )
    totals = defaultdict(float)
    residue_correct: dict[str, int] = defaultdict(int)
    residue_total: dict[str, int] = defaultdict(int)
    position_loss: dict[int, list[float]] = defaultdict(list)
    length_loss: dict[int, list[float]] = defaultdict(list)
    vocab = dataset.vocabulary
    id_to_token = vocab.id_to_token
    for batch in tqdm(loader, desc="evaluating sequences", dynamic_ncols=True, disable=not show_progress):
        input_ids = batch["input_ids"].to(device)
        targets = batch["target_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids=input_ids, lengths=lengths, attention_mask=mask)
        metrics = language_model_metrics(logits, targets, mask)
        token_count = metrics["valid_token_count"]
        totals["token_nll_sum"] += metrics["token_nll"] * token_count
        totals["sequence_loss_sum"] += metrics["sequence_loss"] * metrics["sequence_count"]
        totals["valid_token_count"] += token_count
        totals["sequence_count"] += metrics["sequence_count"]
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        preds = logits.argmax(dim=-1)
        for row in range(targets.shape[0]):
            length = int(lengths[row].cpu())
            row_loss = losses[row, :length].detach().cpu().numpy()
            length_loss[length].append(float(row_loss.mean()))
            for pos, value in enumerate(row_loss):
                position_loss[pos].append(float(value))
            for token_id, pred_id in zip(
                targets[row, :length].cpu().tolist(), preds[row, :length].cpu().tolist(), strict=True
            ):
                residue = id_to_token[int(token_id)]
                residue_total[residue] += 1
                residue_correct[residue] += int(token_id == pred_id)
    token_nll = totals["token_nll_sum"] / max(totals["valid_token_count"], 1.0)
    sequence_loss = totals["sequence_loss_sum"] / max(totals["sequence_count"], 1.0)
    return {
        "token_weighted_nll": token_nll,
        "perplexity": float(np.exp(token_nll)),
        "sequence_weighted_loss": sequence_loss,
        "token_accuracy": sum(residue_correct.values()) / max(sum(residue_total.values()), 1),
        "accuracy_by_residue": {
            residue: residue_correct[residue] / max(residue_total[residue], 1) for residue in sorted(residue_total)
        },
        "nll_by_position": {str(pos): float(np.mean(values)) for pos, values in sorted(position_loss.items())},
        "nll_by_length": {str(length): float(np.mean(values)) for length, values in sorted(length_loss.items())},
        "valid_token_count": int(totals["valid_token_count"]),
        "sequence_count": int(totals["sequence_count"]),
    }


def evaluate_generated_sequences(
    generated_sequences: list[str],
    *,
    training_manifest: str | Path,
    vocabulary: ProteinVocabulary | None = None,
    min_length: int = 1,
    max_length: int = 500,
) -> dict[str, Any]:
    """Evaluate generated sequence uniqueness, novelty, composition, and low complexity."""
    vocab = vocabulary or ProteinVocabulary()
    train_frame = load_sequence_manifest(training_manifest)
    training_records, _ = validate_records(
        train_frame,
        source_path=training_manifest,
        vocabulary=vocab,
        min_length=min_length,
        max_length=max_length,
    )
    return generated_sequence_metrics(
        generated_sequences, training_sequences=[record.sequence for record in training_records]
    )


def write_evaluation(path: str | Path, metrics: dict[str, Any]) -> None:
    """Write evaluation metrics JSON."""
    atomic_write_json(path, metrics)
