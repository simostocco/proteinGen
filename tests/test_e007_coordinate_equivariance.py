from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.models.coordinate_equivariance import (
    coordinate_backend_policy,
    coordinate_backend_state,
    coordinate_model_execution_context,
    equivariance_criterion,
    equivariance_metrics,
    strict_equivariance_numerics,
)
from protein_distance_diffusion.models.embeddings import LengthEmbedding, SinusoidalTimeEmbedding
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import (
    EquivariantPairCoordinateUNet,
)


def _model(dtype: torch.dtype = torch.float32) -> EquivariantPairCoordinateUNet:
    return (
        EquivariantPairCoordinateUNet(
            rbf_bins=4,
            base_channels=4,
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
        )
        .to(dtype=dtype)
        .eval()
    )


def _inputs(length: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(length)
    coordinates = torch.randn((1, length, 3), generator=generator, dtype=dtype)
    mask = torch.ones((1, length), dtype=torch.bool)
    continuity = torch.ones((1, length - 1), dtype=torch.bool)
    return coordinates, torch.tensor([11]), torch.tensor([length]), mask, continuity


def _strict_policy() -> dict[str, object]:
    return {
        "allow_matmul_tf32": False,
        "allow_cudnn_tf32": False,
        "deterministic_algorithms": True,
        "autocast_enabled": False,
        "equivariance_policy": "strict_o3_v1",
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_embeddings_follow_model_dtype(dtype: torch.dtype) -> None:
    time = SinusoidalTimeEmbedding(8).to(dtype=dtype)
    length = LengthEmbedding(8, max_length=500).to(dtype=dtype)
    assert time(torch.tensor([3])).dtype == dtype
    assert length(torch.tensor([31])).dtype == dtype


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize("reflection", [False, True])
def test_shared_equivariance_metrics_cover_rotations_reflections_and_lengths(
    dtype: torch.dtype, tolerance: float, reflection: bool
) -> None:
    torch.manual_seed(7)
    model = _model(dtype)
    for length in (7, 31, 127):
        coordinates, timestep, lengths, mask, continuity = _inputs(length, dtype)
        transform = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0 if reflection else 1.0]],
            dtype=dtype,
        )
        with strict_equivariance_numerics(deterministic=True), torch.no_grad():
            reference = model(coordinates, timestep, lengths, mask, continuity)
            transformed_coordinates = coordinates @ transform
            transformed = model(transformed_coordinates, timestep, lengths, mask, continuity)
        metrics = equivariance_metrics(
            reference=reference,
            transformed=transformed,
            transformation=transform,
            reference_coordinates=coordinates,
            transformed_coordinates=transformed_coordinates,
            residue_mask=mask,
        )
        criterion = equivariance_criterion(
            metrics,
            absolute_tolerance=tolerance,
            relative_l2_tolerance=tolerance,
            coefficient_tolerance=tolerance,
        )
        assert criterion["passed"]


def test_strict_numerics_restores_backend_state() -> None:
    before = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    with strict_equivariance_numerics(deterministic=True):
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.are_deterministic_algorithms_enabled()
    assert before == (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )


def test_coordinate_execution_policy_reports_and_restores_success() -> None:
    device = torch.device("cpu")
    before = coordinate_backend_state(device)
    with coordinate_model_execution_context(_strict_policy(), device) as telemetry:
        assert telemetry["before"] == before
        assert telemetry["during"]["allow_matmul_tf32"] is False
        assert telemetry["during"]["allow_cudnn_tf32"] is False
        assert telemetry["during"]["deterministic_algorithms"] is True
        assert telemetry["during"]["autocast_enabled"] is False
    assert telemetry["after"] == before
    assert telemetry["restored"] is True


def test_coordinate_execution_policy_restores_after_exception() -> None:
    device = torch.device("cpu")
    before = coordinate_backend_state(device)
    with pytest.raises(RuntimeError, match="deliberate"):
        with coordinate_model_execution_context(_strict_policy(), device) as telemetry:
            raise RuntimeError("deliberate failure")
    assert telemetry["after"] == before
    assert telemetry["restored"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("allow_matmul_tf32", True),
        ("allow_cudnn_tf32", True),
        ("deterministic_algorithms", False),
        ("autocast_enabled", True),
        ("equivariance_policy", "ambient_defaults"),
    ],
)
def test_coordinate_execution_policy_rejects_mismatch(field: str, value: object) -> None:
    policy = _strict_policy()
    policy[field] = value
    with pytest.raises(ValueError, match="policy mismatch"):
        coordinate_backend_policy(policy)


def test_equivariance_criterion_rejects_real_scientific_difference() -> None:
    metrics = {
        "maximum_absolute_output_error": 1e-3,
        "relative_l2_output_error": 1e-3,
        "pair_coefficient_invariance_error": 1e-3,
    }
    result = equivariance_criterion(
        metrics,
        absolute_tolerance=5e-5,
        relative_l2_tolerance=5e-5,
        coefficient_tolerance=5e-5,
    )
    assert not result["passed"]
    assert not any(result["checks"].values())


def test_padding_does_not_change_strict_equivariance_result() -> None:
    model = _model()
    coordinates, timestep, lengths, mask, continuity = _inputs(31, torch.float32)
    with strict_equivariance_numerics(deterministic=True), torch.no_grad():
        reference = model(coordinates, timestep, lengths, mask, continuity)["v_prediction"]
        padded = model(
            torch.nn.functional.pad(coordinates, (0, 0, 0, 8)),
            timestep,
            lengths,
            torch.nn.functional.pad(mask, (0, 8), value=False),
            torch.nn.functional.pad(continuity, (0, 8), value=False),
        )["v_prediction"]
    assert torch.allclose(padded[:, :31], reference, atol=1e-7, rtol=0)
    assert torch.count_nonzero(padded[:, 31:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_production_capacity_cuda_strict_equivariance() -> None:
    config = yaml.safe_load(Path("configs/e007_coordinate_real_loader_smoke_v1.yaml").read_text())
    torch.manual_seed(config["seed"])
    model = EquivariantPairCoordinateUNet(**config["model"]).cuda().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) == 7_586_505
    coordinates, timestep, lengths, mask, continuity = _inputs(57, torch.float32)
    coordinates = coordinates.cuda()
    timestep = timestep.cuda()
    lengths = lengths.cuda()
    mask = mask.cuda()
    continuity = continuity.cuda()
    transform = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], device="cuda")
    with strict_equivariance_numerics(deterministic=True), torch.no_grad():
        reference = model(coordinates, timestep, lengths, mask, continuity)
        transformed_coordinates = coordinates @ transform
        transformed = model(transformed_coordinates, timestep, lengths, mask, continuity)
    metrics = equivariance_metrics(
        reference=reference,
        transformed=transformed,
        transformation=transform,
        reference_coordinates=coordinates,
        transformed_coordinates=transformed_coordinates,
        residue_mask=mask,
    )
    assert equivariance_criterion(
        metrics,
        absolute_tolerance=5e-5,
        relative_l2_tolerance=5e-5,
        coefficient_tolerance=5e-5,
    )["passed"]
