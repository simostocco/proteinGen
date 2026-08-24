"""Variable-length collation for autoregressive protein sequences."""

from __future__ import annotations

from typing import Any

import torch

from protein_sequence_generation.vocabulary import ProteinVocabulary


def collate_sequences(items: list[dict[str, Any]], *, vocabulary: ProteinVocabulary | None = None) -> dict[str, Any]:
    """Pad teacher-forced sequence examples to the longest sequence in the batch."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    vocab = vocabulary or ProteinVocabulary()
    lengths = torch.stack([item["length"] for item in items]).long()
    max_length = int(lengths.max().item())
    batch_size = len(items)
    input_ids = torch.full((batch_size, max_length), vocab.pad_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_length), vocab.pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for row, item in enumerate(items):
        length = int(item["length"].item())
        input_ids[row, :length] = item["input_ids"]
        target_ids[row, :length] = item["target_ids"]
        attention_mask[row, :length] = True
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "sample_ids": [item["sample_id"] for item in items],
        "sequences": [item["sequence"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }
