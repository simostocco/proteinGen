"""Symmetric axial-attention architecture tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.data.preprocess import ProteinSample, save_processed_sample
from protein_distance_diffusion.data.statistics import write_normalization
from protein_distance_diffusion.diffusion.gaussian import (
    GaussianDiffusion,
    PredictionType,
    masked_upper_triangular_loss,
)
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.attention import SymmetricAxialAttentionBlock
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.trainer import build_model_from_config, train_from_config


def _block(*, dropout: float = 0.0, chunk_size: int | None = 3) -> SymmetricAxialAttentionBlock:
    return SymmetricAxialAttentionBlock(
        channels=8,
        cond_dim=16,
        heads=2,
        groups=4,
        dropout=dropout,
        chunk_size=chunk_size,
    )


def _symmetric_input(batch: int, channels: int, side: int, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = make_pair_mask(lengths, side)
    x = torch.randn(batch, channels, side, side)
    x = 0.5 * (x + x.transpose(-1, -2))
    return x * mask.float(), mask


def _model_config(*, axial: bool) -> dict:
    return {
        "base_channels": 4,
        "channel_multipliers": [1, 2, 4],
        "residual_blocks_per_level": 2,
        "dropout": 0.0,
        "group_norm_groups": 4,
        "attention_heads": 2,
        "use_bottleneck_attention": True,
        "use_pre_bottleneck_axial_attention": axial,
        "axial_attention_heads": 2,
        "axial_attention_dropout": 0.0,
        "axial_attention_chunk_size": 5,
        "time_embedding_dim": 32,
        "length_embedding_dim": 32,
        "max_length": 64,
    }


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


def _tiny_axial_training_config(tmp_path: Path) -> dict:
    train_manifest = _write_split(tmp_path, "train", ["ax01", "ax02"])
    val_manifest = _write_split(tmp_path, "validation", ["ax03"])
    norm = tmp_path / "normalization.json"
    write_normalization(train_manifest, norm)
    cfg = {
        "train_manifest": str(train_manifest),
        "validation_manifest": str(val_manifest),
        "normalization_file": str(norm),
        "output_dir": str(tmp_path / "axial-train"),
        "seed": 123,
        "device": "cpu",
        "batch_size": 1,
        "epochs": 1,
        "gradient_accumulation_steps": 1,
        "mixed_precision": False,
        "diffusion_steps": 4,
        "learning_rate": 1e-3,
        "checkpoint_every_steps": 1,
        "max_optimizer_steps": 1,
        "prediction_parameterization": "v",
        "logging": {
            "progress_bar": False,
            "tensorboard": False,
            "log_every_steps": 1,
            "validation_every_epochs": 1,
            "image_every_epochs": 99,
        },
        "model": _model_config(axial=True),
    }
    cfg["model"].update(
        {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "attention_heads": 1,
            "axial_attention_heads": 1,
            "time_embedding_dim": 16,
            "length_embedding_dim": 16,
            "max_length": 16,
        }
    )
    return cfg


def test_axial_block_shape_symmetry_and_padded_zero() -> None:
    """The block preserves square shape, symmetry, and padded zeros."""
    torch.manual_seed(1)
    block = _block()
    lengths = torch.tensor([7, 5])
    x, mask = _symmetric_input(2, 8, 9, lengths)
    cond = torch.randn(2, 16)
    out = block(x, cond, mask)
    assert out.shape == x.shape
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)
    assert torch.count_nonzero(out.masked_select(~mask.expand_as(out))) == 0


def test_axial_block_transpose_equivariance_and_deterministic_eval() -> None:
    """Shared row/column projections make eval output transpose-equivariant."""
    torch.manual_seed(2)
    block = _block()
    block.eval()
    lengths = torch.tensor([7])
    x, mask = _symmetric_input(1, 8, 7, lengths)
    cond = torch.randn(1, 16)
    with torch.no_grad():
        out = block(x, cond, mask)
        repeated = block(x, cond, mask)
        transposed = block(x.transpose(-1, -2), cond, mask.transpose(-1, -2))
    assert torch.allclose(out, repeated, atol=0, rtol=0)
    assert torch.allclose(transposed, out.transpose(-1, -2), atol=1e-6)


def test_axial_block_padding_invariance() -> None:
    """Valid-region output is independent of extra square padding."""
    torch.manual_seed(3)
    block = _block()
    block.eval()
    cond = torch.randn(1, 16)
    lengths = torch.tensor([7])
    x7, mask7 = _symmetric_input(1, 8, 7, lengths)
    x9 = torch.zeros(1, 8, 9, 9)
    x9[:, :, :7, :7] = x7
    mask9 = make_pair_mask(lengths, 9)
    with torch.no_grad():
        out7 = block(x7, cond, mask7)
        out9 = block(x9, cond, mask9)
    assert torch.allclose(out9[:, :, :7, :7], out7, atol=1e-5)
    assert torch.count_nonzero(out9[:, :, 7:, :]) == 0
    assert torch.count_nonzero(out9[:, :, :, 7:]) == 0


def test_axial_block_finite_forward_backward() -> None:
    """The axial block supports finite gradients."""
    torch.manual_seed(4)
    block = _block(dropout=0.1)
    lengths = torch.tensor([9, 6])
    x, mask = _symmetric_input(2, 8, 9, lengths)
    x.requires_grad_(True)
    cond = torch.randn(2, 16)
    loss = block(x, cond, mask).square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in block.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def _assert_autocast_axial_block(
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int | None,
) -> None:
    torch.manual_seed(44)
    block = _block(chunk_size=chunk_size).to(device)
    block.eval()
    lengths = torch.tensor([9], device=device)
    x, mask = _symmetric_input(1, 8, 11, lengths.cpu())
    x = x.to(device).requires_grad_(True)
    mask = mask.to(device)
    cond = torch.randn(1, 16, device=device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = block(x, cond, mask)
        loss = out.square().mean()
    loss.backward()
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), out.float().transpose(-1, -2), atol=1e-4)
    assert torch.count_nonzero(out.masked_select(~mask.expand_as(out))) == 0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("chunk_size", [None, 128])
def test_axial_block_cpu_bfloat16_autocast_regression(chunk_size: int | None) -> None:
    """CPU bf16 autocast covers mixed source/destination dtypes portably."""
    try:
        _assert_autocast_axial_block(device=torch.device("cpu"), dtype=torch.bfloat16, chunk_size=chunk_size)
    except RuntimeError as exc:
        message = str(exc)
        if "bfloat16" in message.lower() or "BFloat16" in message:
            pytest.skip(f"CPU bfloat16 autocast is unsupported by this PyTorch build: {message}")
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("chunk_size", [None, 128])
def test_axial_block_cuda_float16_autocast_regression(chunk_size: int | None) -> None:
    """CUDA fp16 autocast catches float32 destination / fp16 attention source mismatches."""
    _assert_autocast_axial_block(device=torch.device("cuda"), dtype=torch.float16, chunk_size=chunk_size)


def test_axial_unet_variable_lengths_odd_non_power_and_padded_output() -> None:
    """Axial U-Net handles odd/non-power lengths in a masked batch."""
    torch.manual_seed(5)
    model = build_model_from_config(_model_config(axial=True))
    lengths = torch.tensor([17, 31, 32])
    side = 32
    mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    noisy = torch.randn(3, 1, side, side) * mask.float()
    out = model(noisy, torch.tensor([0, 1, 2]), lengths, sep, mask)
    assert out.shape == noisy.shape
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)
    assert torch.count_nonzero(out.masked_select(~mask.expand_as(out))) == 0


def test_axial_config_parsing_disabled_baseline_and_checkpoint_compatibility(tmp_path: Path) -> None:
    """Config keys are optional; old non-axial state dicts load into non-axial models."""
    baseline_cfg = _model_config(axial=False)
    baseline = build_model_from_config(baseline_cfg)
    assert not any(isinstance(module, SymmetricAxialAttentionBlock) for module in baseline.modules())
    reloaded = build_model_from_config(baseline_cfg)
    reloaded.load_state_dict(baseline.state_dict())

    axial = build_model_from_config(_model_config(axial=True))
    axial_blocks = [module for module in axial.modules() if isinstance(module, SymmetricAxialAttentionBlock)]
    assert len(axial_blocks) == 1
    assert axial.pre_bottleneck_axial_level == 1


def test_axial_requires_a_preserved_convolutional_block() -> None:
    """Enabling axial attention requires a level where one conv residual remains."""
    cfg = _model_config(axial=True)
    cfg["residual_blocks_per_level"] = 1
    with pytest.raises(ValueError, match="residual_blocks_per_level"):
        build_model_from_config(cfg)


def test_axial_parameter_count_and_no_full_2d_attention_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Axial attention has bounded axis-wise sequence lengths, not H*W token attention."""
    base = build_model_from_config(_model_config(axial=False))
    axial = build_model_from_config(_model_config(axial=True))
    base_count = sum(p.numel() for p in base.parameters())
    axial_count = sum(p.numel() for p in axial.parameters())
    assert axial_count != base_count
    assert 0.5 * base_count < axial_count < 2.0 * base_count

    seen_lengths_by_embed_dim: dict[int, list[int]] = {}
    original_forward = torch.nn.MultiheadAttention.forward

    def recording_forward(self, query, key, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_lengths_by_embed_dim.setdefault(int(self.embed_dim), []).append(int(query.shape[1]))
        return original_forward(self, query, key, value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.MultiheadAttention, "forward", recording_forward)
    lengths = torch.tensor([32])
    side = 32
    mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    noisy = torch.randn(1, 1, side, side) * mask.float()
    axial(noisy, torch.tensor([0]), lengths, sep, mask)
    axial_lengths = seen_lengths_by_embed_dim[8]
    assert max(axial_lengths) <= side // 2
    assert side * side not in axial_lengths


def test_axial_one_cpu_training_step_with_v_prediction(tmp_path: Path) -> None:
    """A bounded CPU training step works with v-prediction and axial attention."""
    ckpt_path = train_from_config(_tiny_axial_training_config(tmp_path))
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["global_step"] == 1
    assert ckpt["config"]["prediction_parameterization"] == PredictionType.V.value


def test_axial_manual_v_prediction_step() -> None:
    """The axial model participates in the v-prediction objective."""
    torch.manual_seed(6)
    model = build_model_from_config(_model_config(axial=True))
    diffusion = GaussianDiffusion(cosine_beta_schedule(4))
    lengths = torch.tensor([16])
    mask = make_pair_mask(lengths, 16)
    sep = make_sequence_separation(lengths, 16)
    clean = torch.randn(1, 1, 16, 16) * mask.float()
    timesteps = torch.tensor([1])
    noisy, eps = diffusion.q_sample(clean, timesteps, mask)
    target = diffusion.training_target(
        x_start=clean,
        t=timesteps,
        epsilon=eps,
        prediction_type=PredictionType.V,
    )
    prediction = model(noisy, timesteps, lengths, sep, mask)
    loss = masked_upper_triangular_loss(target, prediction, mask)
    loss.backward()
    assert torch.isfinite(loss)
