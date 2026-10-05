"""Numerically strict O(3)-equivariance diagnostics for coordinate models."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch

from protein_distance_diffusion.training.coordinate_diffusion import coordinates_to_distance_matrix

STRICT_O3_POLICY = "strict_o3_v1"


@dataclass(frozen=True)
class CoordinateBackendPolicy:
    """Explicit numerical contract for coordinate-model execution."""

    allow_matmul_tf32: bool
    allow_cudnn_tf32: bool
    deterministic_algorithms: bool
    autocast_enabled: bool
    equivariance_policy: str


def coordinate_backend_policy(value: dict[str, Any]) -> CoordinateBackendPolicy:
    """Parse and require the validated strict O(3) numerical policy."""
    required = {
        "allow_matmul_tf32",
        "allow_cudnn_tf32",
        "deterministic_algorithms",
        "autocast_enabled",
        "equivariance_policy",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"coordinate numerics must declare exactly {sorted(required)}")
    policy = CoordinateBackendPolicy(**value)
    expected = CoordinateBackendPolicy(False, False, True, False, STRICT_O3_POLICY)
    if policy != expected:
        raise ValueError(f"coordinate numerical backend policy mismatch: {asdict(policy)} != {asdict(expected)}")
    return policy


def _autocast_enabled(device: torch.device) -> bool:
    try:
        return bool(torch.is_autocast_enabled(device.type))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def coordinate_backend_state(device: torch.device) -> dict[str, Any]:
    """Return the ambient backend state relevant to coordinate execution."""
    return {
        "device_type": device.type,
        "allow_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "autocast_enabled": _autocast_enabled(device),
    }


def require_coordinate_backend_policy(policy: CoordinateBackendPolicy, device: torch.device) -> dict[str, Any]:
    """Fail if active PyTorch state differs from the declared policy."""
    state = coordinate_backend_state(device)
    mismatches = {
        key: {"expected": expected, "observed": state[key]}
        for key, expected in (
            ("allow_matmul_tf32", policy.allow_matmul_tf32),
            ("allow_cudnn_tf32", policy.allow_cudnn_tf32),
            ("deterministic_algorithms", policy.deterministic_algorithms),
            ("autocast_enabled", policy.autocast_enabled),
        )
        if state[key] != expected
    }
    if mismatches:
        raise RuntimeError(f"active coordinate numerical backend contradicts policy: {mismatches}")
    return state


@contextmanager
def coordinate_model_execution_context(value: dict[str, Any], device: torch.device) -> Iterator[dict[str, Any]]:
    """Apply the coordinate backend contract and restore ambient state."""
    policy = coordinate_backend_policy(value)
    before = coordinate_backend_state(device)
    telemetry: dict[str, Any] = {
        "policy": asdict(policy),
        "before": before,
        "during": None,
        "after": None,
        "restored": False,
    }
    try:
        torch.backends.cuda.matmul.allow_tf32 = policy.allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = policy.allow_cudnn_tf32
        torch.use_deterministic_algorithms(policy.deterministic_algorithms)
        with torch.autocast(device_type=device.type, enabled=policy.autocast_enabled):
            telemetry["during"] = require_coordinate_backend_policy(policy, device)
            yield telemetry
    finally:
        torch.use_deterministic_algorithms(bool(before["deterministic_algorithms"]))
        torch.backends.cuda.matmul.allow_tf32 = bool(before["allow_matmul_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(before["allow_cudnn_tf32"])
        telemetry["after"] = coordinate_backend_state(device)
        telemetry["restored"] = telemetry["after"] == before
        if not telemetry["restored"]:
            raise RuntimeError("coordinate numerical backend state restoration failed")


@contextmanager
def strict_equivariance_numerics(*, deterministic: bool = False) -> Iterator[None]:
    """Disable reduced-precision CUDA kernels while preserving global settings."""
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(deterministic)
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def equivariance_metrics(
    *,
    reference: dict[str, torch.Tensor],
    transformed: dict[str, torch.Tensor],
    transformation: torch.Tensor,
    reference_coordinates: torch.Tensor,
    transformed_coordinates: torch.Tensor,
    residue_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure output equivariance and invariant pair-path parity."""
    expected = reference["v_prediction"] @ transformation
    output_delta = transformed["v_prediction"] - expected
    coefficient_delta = transformed["pair_coefficients"] - reference["pair_coefficients"]
    reference_distances = coordinates_to_distance_matrix(reference_coordinates, residue_mask)
    transformed_distances = coordinates_to_distance_matrix(transformed_coordinates, residue_mask)
    valid = residue_mask[..., None].expand_as(output_delta)
    centered = (transformed["v_prediction"] * residue_mask[..., None]).sum(dim=1) / residue_mask.sum(dim=1).clamp_min(
        1
    )[:, None]
    denominator = expected[valid].double().norm().clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "maximum_absolute_output_error": float(output_delta[valid].abs().max().detach().cpu()),
        "rms_output_error": float(output_delta[valid].double().square().mean().sqrt().detach().cpu()),
        "relative_l2_output_error": float((output_delta[valid].double().norm() / denominator).detach().cpu()),
        "maximum_output_magnitude": float(expected[valid].abs().max().detach().cpu()),
        "pair_coefficient_invariance_error": float(coefficient_delta.abs().max().detach().cpu()),
        "pair_distance_invariance_error": float(
            (transformed_distances - reference_distances).abs().max().detach().cpu()
        ),
        "centered_output_residual": float(centered.abs().max().detach().cpu()),
    }


def equivariance_criterion(
    metrics: dict[str, float],
    *,
    absolute_tolerance: float,
    relative_l2_tolerance: float,
    coefficient_tolerance: float,
) -> dict[str, Any]:
    """Apply the shared strict absolute, relative, and scalar-invariant gates."""
    checks = {
        "absolute_output": metrics["maximum_absolute_output_error"] <= absolute_tolerance,
        "relative_l2_output": metrics["relative_l2_output_error"] <= relative_l2_tolerance,
        "pair_coefficients": metrics["pair_coefficient_invariance_error"] <= coefficient_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "absolute_tolerance": absolute_tolerance,
        "relative_l2_tolerance": relative_l2_tolerance,
        "coefficient_tolerance": coefficient_tolerance,
        "formula": (
            "max_abs<=absolute_tolerance AND relative_l2<=relative_l2_tolerance "
            "AND pair_coefficient_error<=coefficient_tolerance"
        ),
    }
