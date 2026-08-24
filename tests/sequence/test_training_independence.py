"""Sequence training and independence tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from protein_sequence_generation.checkpointing import load_checkpoint
from protein_sequence_generation.dataset import ProteinSequenceDataset
from protein_sequence_generation.model import ProteinSequenceTransformer, SequenceTransformerConfig
from protein_sequence_generation.training import train_from_config


def write_manifest(path: Path, sequences: list[str]) -> None:
    pd.DataFrame(
        [{"sample_id": f"s{i}", "sequence": sequence, "length": len(sequence)} for i, sequence in enumerate(sequences)]
    ).to_parquet(path)


def tiny_config(tmp_path: Path) -> dict:
    train = tmp_path / "train.parquet"
    val = tmp_path / "validation.parquet"
    write_manifest(train, ["ACDEFG", "CDEFGH", "DEFGHI", "EFGHIK"])
    write_manifest(val, ["ACDEFG", "CDEFGH"])
    return {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "data": {"train_manifest": str(train), "validation_manifest": str(val), "min_length": 1, "max_length": 16},
        "model": {
            "vocabulary_size": 24,
            "d_model": 32,
            "num_layers": 1,
            "num_attention_heads": 4,
            "feedforward_dimension": 64,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "max_length": 16,
        },
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "num_workers": 0,
        "epochs": 3,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "loss_reduction": "sequence_mean",
        "checkpoint_every_optimizer_steps": 1,
        "validation_every_epochs": 1,
        "early_stopping_patience": 3,
        "tensorboard": False,
        "progress_bar": False,
    }


def test_one_cpu_optimizer_step_checkpoint_and_resume(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    checkpoint = train_from_config(config, max_optimizer_steps=1)
    payload = load_checkpoint(checkpoint)
    assert payload["global_step"] == 1
    assert payload["vocabulary"]["tokens"][0] == "<PAD>"
    assert payload["manifest_hashes"]["train"]
    resumed = train_from_config(config, resume_from=checkpoint, max_optimizer_steps=2)
    resumed_payload = load_checkpoint(resumed)
    assert resumed_payload["global_step"] >= 2


def test_checkpoint_resume_rejects_architecture_mismatch(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    checkpoint = train_from_config(config, max_optimizer_steps=1)
    changed = tiny_config(tmp_path)
    changed["model"]["d_model"] = 64
    with pytest.raises(ValueError, match="architecture"):
        train_from_config(changed, resume_from=checkpoint, max_optimizer_steps=2)


def test_tiny_dataset_can_reduce_loss(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    config["epochs"] = 8
    checkpoint = train_from_config(config, max_optimizer_steps=6)
    payload = load_checkpoint(checkpoint)
    assert torch.isfinite(torch.tensor(payload["best_validation_metric"]))


def test_sequence_package_independent_of_distance_data() -> None:
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "protein_distance_diffusion.data" or name.startswith("protein_distance_diffusion.data."):
            sys.modules.pop(name)
    importlib.import_module("protein_sequence_generation.dataset")
    importlib.import_module("protein_sequence_generation.training")
    imported = [name for name in sys.modules if name.startswith("protein_sequence_generation")]
    assert imported
    assert "protein_distance_diffusion.data" not in sys.modules


def test_dataset_smoke_without_distance_package(tmp_path: Path) -> None:
    path = tmp_path / "toy.parquet"
    write_manifest(path, ["ACDEFG"])
    dataset = ProteinSequenceDataset(path, min_length=1, max_length=10)
    assert dataset[0]["sequence"] == "ACDEFG"


def test_rope_through_configured_max_length() -> None:
    model = ProteinSequenceTransformer(
        SequenceTransformerConfig(
            vocabulary_size=24,
            d_model=32,
            num_layers=1,
            num_attention_heads=4,
            feedforward_dimension=64,
            max_length=16,
            dropout=0.0,
            attention_dropout=0.0,
        )
    )
    ids = torch.ones((1, 16), dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    logits = model(ids, torch.tensor([16]), mask)
    assert logits.shape == torch.Size([1, 16, 24])
