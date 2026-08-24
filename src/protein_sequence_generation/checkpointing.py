"""Atomic checkpoint helpers for sequence Transformer experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from protein_sequence_generation.utils import sha256_file
from protein_sequence_generation.vocabulary import ProteinVocabulary


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically save a checkpoint."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(dst)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint."""
    return torch.load(path, map_location=map_location, weights_only=False)


def assert_resume_compatible(
    checkpoint: dict[str, Any],
    *,
    model_config: dict[str, Any],
    vocabulary: ProteinVocabulary,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    unsafe_override: bool = False,
) -> None:
    """Refuse incompatible resume attempts unless explicitly overridden."""
    if unsafe_override:
        return
    saved_vocab = checkpoint.get("vocabulary", {}).get("tokens")
    if tuple(saved_vocab or ()) != vocabulary.tokens:
        raise ValueError("Refusing resume because the vocabulary differs")
    saved_model = dict(checkpoint.get("model_config", {}))
    current_model = dict(model_config)
    if saved_model != current_model:
        raise ValueError("Refusing resume because the architecture config differs")
    if int(saved_model.get("max_length", -1)) != int(current_model.get("max_length", -2)):
        raise ValueError("Refusing resume because max_length differs")
    hashes = checkpoint.get("manifest_hashes", {})
    if hashes.get("train") != sha256_file(train_manifest):
        raise ValueError("Refusing resume because the train manifest hash differs")
    if hashes.get("validation") != sha256_file(validation_manifest):
        raise ValueError("Refusing resume because the validation manifest hash differs")
