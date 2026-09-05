"""Symmetric triangle-multiplicative update tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.data.collate import make_pair_mask, make_sequence_separation
from protein_distance_diffusion.models.attention import (
    SymmetricAxialAttentionBlock,
    SymmetricTriangleMultiplicativeUpdate,
)
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.trainer import build_model_from_config


def _block(*, chunk_size: int | None = 2) -> SymmetricTriangleMultiplicativeUpdate:
    return SymmetricTriangleMultiplicativeUpdate(
        channels=8,
        hidden_channels=6,
        groups=4,
        dropout=0.0,
        chunk_size=chunk_size,
    )


def _symmetric_input(batch: int, channels: int, side: int, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = make_pair_mask(lengths, side)
    x = torch.randn(batch, channels, side, side)
    x = 0.5 * (x + x.transpose(-1, -2))
    return x * mask.float(), mask


def _model_config(*, triangle: bool) -> dict:
    return {
        "base_channels": 4,
        "channel_multipliers": [1, 2, 4],
        "residual_blocks_per_level": 2,
        "dropout": 0.0,
        "group_norm_groups": 4,
        "attention_heads": 2,
        "use_bottleneck_attention": True,
        "use_pre_bottleneck_axial_attention": True,
        "axial_attention_heads": 2,
        "axial_attention_dropout": 0.0,
        "axial_attention_chunk_size": 4,
        "use_pre_bottleneck_triangle_multiplication": triangle,
        "triangle_hidden_channels": 6,
        "triangle_dropout": 0.0,
        "triangle_chunk_size": 2,
        "time_embedding_dim": 32,
        "length_embedding_dim": 32,
        "max_length": 64,
    }


def test_triangle_block_shape_symmetry_masking_and_gradients() -> None:
    torch.manual_seed(1)
    block = _block()
    lengths = torch.tensor([7, 5])
    x, mask = _symmetric_input(2, 8, 9, lengths)
    x.requires_grad_(True)

    out = block(x, mask)
    assert out.shape == x.shape
    assert torch.allclose(out, out.transpose(-1, -2), atol=1e-6)
    assert torch.count_nonzero(out.masked_select(~mask.expand_as(out))) == 0

    loss = out.square().mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [parameter.grad for parameter in block.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_triangle_block_padding_invariance() -> None:
    torch.manual_seed(2)
    block = _block()
    block.eval()
    lengths = torch.tensor([7])
    x7, mask7 = _symmetric_input(1, 8, 7, lengths)
    x10 = torch.zeros(1, 8, 10, 10)
    x10[:, :, :7, :7] = x7
    mask10 = make_pair_mask(lengths, 10)

    with torch.no_grad():
        out7 = block(x7, mask7)
        out10 = block(x10, mask10)

    assert torch.allclose(out10[:, :, :7, :7], out7, atol=1e-5)
    assert torch.count_nonzero(out10[:, :, 7:, :]) == 0
    assert torch.count_nonzero(out10[:, :, :, 7:]) == 0


def test_triangle_block_transpose_equivariance() -> None:
    torch.manual_seed(3)
    block = _block()
    block.eval()
    lengths = torch.tensor([8])
    x, mask = _symmetric_input(1, 8, 8, lengths)

    with torch.no_grad():
        out = block(x, mask)
        transposed = block(x.transpose(-1, -2), mask.transpose(-1, -2))

    assert torch.allclose(transposed, out.transpose(-1, -2), atol=1e-6)


def test_triangle_block_chunked_and_unchunked_equivalence() -> None:
    torch.manual_seed(4)
    chunked = _block(chunk_size=2)
    unchunked = _block(chunk_size=None)
    unchunked.load_state_dict(chunked.state_dict())
    chunked.eval()
    unchunked.eval()
    x, mask = _symmetric_input(2, 8, 8, torch.tensor([8, 6]))

    with torch.no_grad():
        out_chunked = chunked(x, mask)
        out_unchunked = unchunked(x, mask)

    assert torch.allclose(out_chunked, out_unchunked, atol=1e-6)


def test_triangle_block_all_padding_and_degenerate_masks_are_safe() -> None:
    torch.manual_seed(5)
    block = _block()
    x = torch.randn(2, 8, 4, 4)
    mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    out = block(x, mask)
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0
    assert torch.isfinite(out).all()


def test_triangle_uses_bounded_bmm_not_triplet_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(6)
    block = _block(chunk_size=2)
    x, mask = _symmetric_input(1, 8, 9, torch.tensor([9]))
    source = inspect.getsource(SymmetricTriangleMultiplicativeUpdate._triangle_message)
    assert "einsum" not in source
    assert "bmm" in source

    seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_bmm = torch.bmm

    def recording_bmm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        seen_shapes.append((tuple(left.shape), tuple(right.shape)))
        assert left.ndim == 3
        assert right.ndim == 3
        assert left.numel() <= 1 * block.hidden_channels * 9 * 9
        assert right.numel() <= 1 * block.hidden_channels * 9 * 9
        return original_bmm(left, right)

    monkeypatch.setattr(torch, "bmm", recording_bmm)
    block(x, mask)
    assert seen_shapes


def _assert_autocast_triangle_block(device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(7)
    block = _block(chunk_size=2).to(device)
    lengths = torch.tensor([8], device=device)
    x, mask = _symmetric_input(1, 8, 8, lengths.cpu())
    x = x.to(device).requires_grad_(True)
    mask = mask.to(device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = block(x, mask)
        loss = out.square().mean()
    loss.backward()
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out.float(), out.float().transpose(-1, -2), atol=1e-4)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_triangle_block_cpu_bfloat16_autocast_dtype_safety() -> None:
    try:
        _assert_autocast_triangle_block(torch.device("cpu"), torch.bfloat16)
    except RuntimeError as exc:
        message = str(exc)
        if "bfloat16" in message.lower() or "BFloat16" in message:
            pytest.skip(f"CPU bfloat16 autocast is unsupported by this PyTorch build: {message}")
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triangle_block_cuda_float16_autocast_dtype_safety() -> None:
    _assert_autocast_triangle_block(torch.device("cuda"), torch.float16)


def test_unet_inserts_triangle_after_axial_at_pre_bottleneck_level() -> None:
    model = build_model_from_config(_model_config(triangle=True))
    triangle_blocks = [
        module for module in model.modules() if isinstance(module, SymmetricTriangleMultiplicativeUpdate)
    ]
    axial_blocks = [module for module in model.modules() if isinstance(module, SymmetricAxialAttentionBlock)]
    assert len(triangle_blocks) == 1
    assert len(axial_blocks) == 1
    assert model.pre_bottleneck_axial_level == 1
    assert "1" in model.pre_bottleneck_triangle

    lengths = torch.tensor([17, 31])
    side = 32
    mask = make_pair_mask(lengths, side)
    sep = make_sequence_separation(lengths, side)
    noisy = torch.randn(2, 1, side, side) * mask.float()
    out = model(noisy, torch.tensor([0, 1]), lengths, sep, mask)
    assert out.shape == noisy.shape
    assert torch.count_nonzero(out.masked_select(~mask.expand_as(out))) == 0


def test_e002_config_behavior_unchanged_when_triangle_disabled() -> None:
    config = yaml.safe_load(Path("configs/train_recovered_full_v_axial_edm_e002_full.yaml").read_text())
    model = build_model_from_config(config["model"])
    assert not any(isinstance(module, SymmetricTriangleMultiplicativeUpdate) for module in model.modules())
    assert any(isinstance(module, SymmetricAxialAttentionBlock) for module in model.modules())


def test_disabled_triangle_model_loads_existing_style_checkpoint(tmp_path: Path) -> None:
    config = _model_config(triangle=False)
    model = build_model_from_config(config)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        {"model": model.state_dict(), "config": {"model": config}, "epoch": 0, "global_step": 0},
    )
    loaded = load_checkpoint(checkpoint)
    reloaded = build_model_from_config(config)
    reloaded.load_state_dict(loaded["model"])


def test_triangle_parameter_count_reporting() -> None:
    e002 = build_model_from_config(_model_config(triangle=False))
    e004 = build_model_from_config(_model_config(triangle=True))
    e002_count = sum(parameter.numel() for parameter in e002.parameters())
    e004_count = sum(parameter.numel() for parameter in e004.parameters())
    assert e004_count > e002_count
    assert e004_count - e002_count == 372
