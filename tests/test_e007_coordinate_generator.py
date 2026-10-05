from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import (
    EquivariantPairCoordinateUNet,
)
from protein_distance_diffusion.models.unet import DistanceUNet


def _model() -> EquivariantPairCoordinateUNet:
    return EquivariantPairCoordinateUNet(
        rbf_bins=4,
        base_channels=8,
        channel_multipliers=(1, 2),
        residual_blocks_per_level=2,
        group_norm_groups=4,
        attention_heads=2,
        axial_attention_heads=2,
        axial_attention_chunk_size=8,
        use_bottleneck_attention=True,
        use_pre_bottleneck_axial_attention=True,
        use_pre_bottleneck_triangle_multiplication=True,
        triangle_hidden_channels=4,
        triangle_chunk_size=2,
        time_embedding_dim=16,
        length_embedding_dim=16,
        max_length=500,
    ).eval()


def _inputs(length: int = 7, side: int | None = None) -> tuple[torch.Tensor, ...]:
    side = side or length
    coordinates = torch.zeros((1, side, 3))
    coordinates[:, :length] = torch.randn((1, length, 3))
    mask = torch.zeros((1, side), dtype=torch.bool)
    mask[:, :length] = True
    continuity = torch.zeros((1, max(side - 1, 0)), dtype=torch.bool)
    continuity[:, : length - 1] = True
    return coordinates, torch.tensor([23]), torch.tensor([length]), mask, continuity


def _rotation(*, reflection: bool = False) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] *= -1
    if reflection:
        q[:, 0] *= -1
    return q


@pytest.mark.parametrize("reflection", [False, True])
def test_o3_equivariance_and_translation_invariance(reflection: bool) -> None:
    torch.manual_seed(8)
    model = _model()
    inputs = _inputs()
    baseline = model(*inputs)["v_prediction"]
    rotation = _rotation(reflection=reflection)
    transformed = inputs[0] @ rotation + torch.tensor([[[5.0, -2.0, 1.0]]])
    observed = model(transformed, *inputs[1:])["v_prediction"]
    assert torch.allclose(observed, baseline @ rotation, atol=2e-5, rtol=2e-5)


def test_coefficients_masks_center_and_padding_invariance() -> None:
    torch.manual_seed(4)
    model = _model()
    short = _inputs(length=5, side=5)
    padded = _inputs(length=5, side=11)
    padded = (torch.nn.functional.pad(short[0], (0, 0, 0, 6)), *padded[1:])
    left = model(*short)
    right = model(*padded)
    coefficients = right["pair_coefficients"]
    assert torch.allclose(left["v_prediction"], right["v_prediction"][:, :5], atol=1e-6)
    assert torch.allclose(coefficients, coefficients.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(coefficients, dim1=-2, dim2=-1)) == 0
    assert torch.count_nonzero(coefficients[:, :, 5:]) == 0
    assert torch.count_nonzero(right["v_prediction"][:, 5:]) == 0
    assert torch.allclose(right["v_prediction"][:, :5].sum(dim=1), torch.zeros((1, 3)), atol=1e-6)


def test_gradients_flow_and_padding_receives_no_gradient() -> None:
    model = _model().train()
    inputs = _inputs(length=5, side=8)
    coordinates = inputs[0].requires_grad_()
    output = model(coordinates, *inputs[1:])["v_prediction"]
    output.square().sum().backward()
    assert coordinates.grad is not None
    assert torch.count_nonzero(coordinates.grad[:, 5:]) == 0
    assert any(parameter.grad is not None for parameter in model.pair_trunk.parameters())
    assert model.pair_trunk.coefficient_head.weight.grad is not None


def test_coincident_coordinates_remain_finite() -> None:
    model = _model()
    inputs = list(_inputs())
    inputs[0].zero_()
    result = model(*inputs)
    assert torch.isfinite(result["v_prediction"]).all()
    assert torch.count_nonzero(result["v_prediction"]) == 0


def test_forward_length_500_has_bounded_shapes() -> None:
    model = EquivariantPairCoordinateUNet(
        rbf_bins=2,
        base_channels=2,
        channel_multipliers=(1,),
        residual_blocks_per_level=1,
        group_norm_groups=1,
        attention_heads=1,
        use_bottleneck_attention=False,
        use_pre_bottleneck_axial_attention=False,
        use_pre_bottleneck_triangle_multiplication=False,
        time_embedding_dim=8,
        length_embedding_dim=8,
        max_length=500,
    ).eval()
    inputs = _inputs(500)
    with torch.no_grad():
        result = model(*inputs)
    assert result["v_prediction"].shape == (1, 500, 3)
    assert result["pair_coefficients"].shape == (1, 1, 500, 500)


def test_production_parameter_count() -> None:
    config = yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())
    model = EquivariantPairCoordinateUNet(**config["model"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_586_505


def test_pinned_e004_checkpoint_still_loads_without_mutation() -> None:
    checkpoint_path = Path(
        "outputs/recovered_full_b2_v_axial_edm_triangle_e004/checkpoints/final_validation_selected.pt"
    )
    before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DistanceUNet(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["ema"], strict=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_582_833
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before


def test_model_forward_has_no_token_argument() -> None:
    import inspect

    names = tuple(inspect.signature(EquivariantPairCoordinateUNet.forward).parameters)
    assert "sequence" not in names
    assert "token_ids" not in names
