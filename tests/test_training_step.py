"""CPU training and sampling smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import protein_distance_diffusion.training.trainer as trainer_module
from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.data.preprocess import ProteinSample, save_processed_sample
from protein_distance_diffusion.data.statistics import write_normalization
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.sampling import sample_ddpm
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.unet import DistanceUNet
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.trainer import train_from_config


def _write_split(tmp_path: Path, name: str, sample_ids: list[str]) -> Path:
    rows = []
    for idx, sample_id in enumerate(sample_ids):
        n = 8
        coords = np.stack([np.arange(n), np.zeros(n) + idx, np.zeros(n)], axis=1).astype(np.float32)
        sample = ProteinSample(
            sample_id=sample_id,
            pdb_id=sample_id[:4].upper(),
            chain_id="A",
            sequence="ACDEFGHI",
            residue_ids=[str(i) for i in range(n)],
            ca_coordinates=coords,
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.0},
        )
        rows.append(save_processed_sample(sample, tmp_path / "samples"))
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _tiny_training_config(tmp_path: Path, *, output_name: str = "out") -> dict:
    train_manifest = _write_split(tmp_path, "train", ["trn1", "trn2"])
    val_manifest = _write_split(tmp_path, "validation", ["val1"])
    norm = tmp_path / "normalization.json"
    write_normalization(train_manifest, norm)
    return {
        "train_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "normalization_file": str(norm),
        "output_dir": str(tmp_path / output_name),
        "seed": 123,
        "device": "cpu",
        "batch_size": 1,
        "epochs": 1,
        "diffusion_steps": 4,
        "learning_rate": 1e-3,
        "checkpoint_every_steps": 1,
        "logging": {
            "progress_bar": False,
            "tensorboard": False,
            "log_every_steps": 1,
            "validation_every_epochs": 1,
            "image_every_epochs": 99,
        },
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "residual_blocks_per_level": 1,
            "attention_heads": 1,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        },
    }


def test_cpu_training_checkpoint_and_reproducible_sampling(tmp_path: Path) -> None:
    """A tiny CPU run trains, validates, checkpoints, reloads, and preserves metadata."""
    config = _tiny_training_config(tmp_path)
    ckpt_path = train_from_config(config)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["global_step"] == 2
    assert ckpt["next_epoch"] == 1
    assert "rng_state" in ckpt
    assert ckpt["normalization"]["mode"] == "scale"
    assert Path(tmp_path / "out" / "logs" / "train.jsonl").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "last.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "latest.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "best_validation.pt").exists()
    assert Path(tmp_path / "out" / "checkpoints" / "step_00000001.pt").exists()
    assert json.loads(Path(config["normalization_file"]).read_text())["scale"] > 0
    assert isinstance(ckpt["model"], dict)
    model_cfg = dict(config["model"])
    model_cfg["channel_multipliers"] = tuple(model_cfg["channel_multipliers"])
    model = DistanceUNet(**model_cfg)
    model.load_state_dict(ckpt["ema"])
    lengths = torch.tensor([8])
    mask = make_pair_mask(lengths, 8)
    sep = make_sequence_separation(lengths, 8)
    diffusion = GaussianDiffusion(cosine_beta_schedule(4))
    sample_a = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(9),
    )
    sample_b = sample_ddpm(
        model,
        diffusion,
        lengths=lengths,
        pair_mask=mask,
        sequence_separation=sep,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(9),
    )
    assert torch.allclose(sample_a, sample_b)


def test_keyboard_interrupt_writes_interrupted_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A KeyboardInterrupt produces a resumable interrupted checkpoint."""
    config = _tiny_training_config(tmp_path, output_name="interrupted")

    def interrupt_loss(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer_module, "masked_upper_triangular_loss", interrupt_loss)
    ckpt_path = train_from_config(config)
    assert ckpt_path.name == "interrupted.pt"
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["global_step"] == 0
    assert ckpt["next_epoch"] == 0
    assert "optimizer" in ckpt
    assert "rng_state" in ckpt


def test_resume_restores_checkpoint_state(tmp_path: Path) -> None:
    """Resuming continues from the saved epoch and global step."""
    config = _tiny_training_config(tmp_path, output_name="resume")
    first_ckpt = train_from_config(config)
    resumed = dict(config)
    resumed["epochs"] = 2
    resumed["resume_from"] = str(first_ckpt)
    second_ckpt = train_from_config(resumed)
    ckpt = load_checkpoint(second_ckpt)
    assert ckpt["epoch"] == 1
    assert ckpt["next_epoch"] == 2
    assert ckpt["global_step"] == 4


def test_tensorboard_logging_when_available(tmp_path: Path) -> None:
    """TensorBoard event files are written when SummaryWriter is enabled."""
    pytest.importorskip("tensorboard")
    config = _tiny_training_config(tmp_path, output_name="tensorboard")
    config["logging"] = {
        "progress_bar": False,
        "tensorboard": True,
        "tensorboard_dir": str(tmp_path / "tensorboard-events"),
        "log_every_steps": 1,
        "validation_every_epochs": 1,
        "image_every_epochs": 1,
    }
    train_from_config(config)
    assert list((tmp_path / "tensorboard-events").glob("events.out.tfevents.*"))
