"""Performance-corrected Phase 3I.4 kernels and read-only reuse audit.

This module contains CPU-safe kernels only. It does not load a model, touch
CUDA, inspect trajectory outcomes, or authorize scientific execution.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

NON_AUTHORIZING = {
    "authorizes_training": False,
    "authorizes_real_data_training": False,
    "authorizes_production_training": False,
    "authorizes_additional_training": False,
    "authorizes_joint_training": False,
    "authorizes_sequence_conditioning": False,
    "authorizes_downstream_generation": False,
    "authorizes_sampler_correction": False,
    "authorizes_pilot_execution": False,
    "authorizes_phase3j": False,
    "authorizes_larger_validation": False,
    "optimizer_created": False,
    "backward_performed": False,
    "parameter_mutation": False,
}


@lru_cache(maxsize=64)
def _topology(length: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    """Cache immutable length/device topology tensors (never mutated in kernels)."""
    idx = torch.arange(length, device=device)
    pair_ids: dict[str, torch.Tensor] = {}
    for offset in (1, 2, 3):
        if length > offset:
            pair_ids[f"i{offset}"] = torch.arange(length - offset, device=device)
            pair_ids[f"j{offset}"] = torch.arange(offset, length, device=device)
            pair_ids[f"structural_mask{offset}"] = torch.ones((1, length - offset, 1), dtype=torch.bool, device=device)
    if length >= 3:
        pair_ids["angle_i"] = idx[:-2]
        pair_ids["angle_j"] = idx[1:-1]
        pair_ids["angle_k"] = idx[2:]
    if length >= 5:
        i, j = torch.triu_indices(length, length, offset=4, device=device)
        pair_ids["clash_i"], pair_ids["clash_j"] = i, j
    return pair_ids


@lru_cache(maxsize=256)
def _target(length: int, dtype: torch.dtype, device: torch.device, value: float) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


@lru_cache(maxsize=128)
def _cached_schedule(length: int, dtype: torch.dtype, device: torch.device, start: int) -> tuple[int, ...]:
    # Length/dtype/device keying keeps the schedule cache lifecycle aligned
    # with the sampler's other immutable per-shape constants.
    del length, dtype, device
    return tuple(sparse_timesteps(start))


def prepare_cache(
    length: int, dtype: torch.dtype, device: torch.device | str, starts: tuple[int, ...] = (425, 250, 150)
) -> None:
    """Warm all topology, target, and schedule constants before reverse steps."""
    dev = torch.device(device)
    _topology(length, dtype, dev)
    for value in (3.8, 6.2, 8.0, 3.0, 110.0 * torch.pi / 180.0):
        _target(length, dtype, dev, float(value))
    for start in starts:
        _cached_schedule(length, dtype, dev, start)


def clear_kernel_cache() -> None:
    _topology.cache_clear()


@torch.no_grad()
def _center(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    avg = (delta * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    return torch.where(m, delta - avg, torch.zeros_like(delta))


@torch.no_grad()
def pair_delta(
    x: torch.Tensor,
    mask: torch.Tensor,
    offset: int,
    target: float,
    strength: float,
    topology: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Symmetric endpoint pair projection using indexed accumulation."""
    b, n, _ = x.shape
    if n <= offset:
        return torch.zeros_like(x)
    ids = topology or _topology(n, x.dtype, x.device)
    i, j = ids[f"i{offset}"], ids[f"j{offset}"]
    d = x.index_select(1, j) - x.index_select(1, i)
    r = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
    active = (mask.index_select(1, i) & mask.index_select(1, j)).unsqueeze(-1)
    active = active & ids[f"structural_mask{offset}"]
    target_value = _target(n, x.dtype, x.device, float(target))
    step = torch.where(active, (target_value - r) * d / r * (0.5 * strength), 0.0)
    out = torch.zeros_like(x)
    out.index_add_(1, i, -step)
    out.index_add_(1, j, step)
    return _center(out, mask)


@torch.no_grad()
def angle_delta(
    x: torch.Tensor,
    mask: torch.Tensor,
    strength: float,
    target_radians: float = 1.9198621771937625,
    topology: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Analytic gradient of unsigned angle error; equivariant under O(3)."""
    if x.shape[1] < 3:
        return torch.zeros_like(x)
    ids = topology or _topology(x.shape[1], x.dtype, x.device)
    i, j, k = ids["angle_i"], ids["angle_j"], ids["angle_k"]
    left = x.index_select(1, i) - x.index_select(1, j)
    right = x.index_select(1, k) - x.index_select(1, j)
    ln = torch.linalg.vector_norm(left, dim=-1, keepdim=True).clamp_min(1e-8)
    rn = torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-8)
    u, v = left / ln, right / rn
    c = (u * v).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7)
    s = torch.sqrt((1 - c.square()).clamp_min(1e-12))
    theta = torch.acos(c)
    gi = -(v - c * u) / (ln * s)
    gk = -(u - c * v) / (rn * s)
    error = theta - _target(x.shape[1], x.dtype, x.device, float(target_radians))
    di = -strength * error * gi
    dk = -strength * error * gk
    active = (mask.index_select(1, i) & mask.index_select(1, j) & mask.index_select(1, k)).unsqueeze(-1)
    di, dk = torch.where(active, di, 0.0), torch.where(active, dk, 0.0)
    out = torch.zeros_like(x)
    out.index_add_(1, i, di)
    out.index_add_(1, k, dk)
    out.index_add_(1, j, -(di + dk))
    return _center(out, mask)


@torch.no_grad()
def clash_delta(
    x: torch.Tensor,
    mask: torch.Tensor,
    strength: float,
    cutoff: float = 3.0,
    topology: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Vectorized nonlocal clash repulsion; local |i-j|<4 pairs are excluded."""
    ids = topology or _topology(x.shape[1], x.dtype, x.device)
    if "clash_i" not in ids:
        return torch.zeros_like(x)
    i, j = ids["clash_i"], ids["clash_j"]
    d = x.index_select(1, j) - x.index_select(1, i)
    r = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(1e-8)
    active = (mask.index_select(1, i) & mask.index_select(1, j)).unsqueeze(-1)
    cutoff_value = _target(x.shape[1], x.dtype, x.device, float(cutoff))
    push = torch.where(active & (r < cutoff_value), (cutoff_value - r) * d / r * strength, 0.0)
    out = torch.zeros_like(x)
    out.index_add_(1, i, -push)
    out.index_add_(1, j, push)
    return _center(out, mask)


@torch.no_grad()
def project_x0(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    family: str,
    iterations: int,
    displacement_cap: float,
    strength: float = 0.35,
    scheduled_clash: bool = True,
) -> torch.Tensor:
    """Cached vector kernels with bounded, centroid-preserving total motion."""
    if family not in {"bond", "composite"} or iterations < 1 or displacement_cap <= 0:
        raise ValueError("invalid projection policy")
    if x.ndim != 3 or x.shape[-1] != 3 or mask.shape != x.shape[:2]:
        raise ValueError("coordinates/mask shape mismatch")
    ids = _topology(x.shape[1], x.dtype, x.device)
    valid = mask.unsqueeze(-1)
    origin = torch.where(valid, x, 0.0)
    y = origin
    for _ in range(iterations):
        delta = pair_delta(y, mask, 1, 3.8, strength, ids)
        if family == "composite":
            delta = delta + pair_delta(y, mask, 2, 6.2, strength * 0.20, ids)
            delta = delta + pair_delta(y, mask, 3, 8.0, strength * 0.12, ids)
            delta = delta + angle_delta(y, mask, strength * 0.10, topology=ids)
            if scheduled_clash:
                delta = delta + clash_delta(y, mask, strength * 0.04, topology=ids)
        delta = _center(delta, mask)
        lengths = torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(displacement_cap / lengths, max=1.0).amin(1, keepdim=True)
        delta = _center(delta * scale, mask)
        y = torch.where(valid, y + delta, 0.0)
    disp = _center(y - origin, mask)
    max_disp = torch.linalg.vector_norm(disp, dim=-1, keepdim=True).amax(1, keepdim=True).clamp_min(1e-12)
    return torch.where(valid, origin + disp * torch.clamp(displacement_cap / max_disp, max=1.0), 0.0)


def sparse_timesteps(start_timestep: int) -> list[int]:
    """Descending indices; intervals [start,151], [150,51], [50,1], then 0.

    Each interval is anchored at its upper endpoint and includes it. The
    candidate start is included. Timestep zero is always included exactly once.
    """
    if not 1 <= start_timestep <= 499:
        raise ValueError("candidate start must be in 1..499")
    bands = []
    high_band = [start_timestep - 16 * k for k in range((start_timestep - 151) // 16 + 1)]
    bands.extend(t for t in high_band if t >= 151)
    if start_timestep <= 150:
        bands = []
    bands.extend(range(150, 50, -8))
    bands.extend(range(50, 0, -2))
    bands.append(0)
    return bands


def scheduled_correction(
    timestep: int,
    start_timestep: int,
    *,
    length: int = 256,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> bool:
    """Host-indexed scheduling; never reads a device scalar or synchronizes."""
    return int(timestep) in _cached_schedule(length, dtype, torch.device(device), start_timestep)


@torch.no_grad()
def guided_reverse_step(
    diffusion: Any,
    x_t: torch.Tensor,
    timestep_tensor: torch.Tensor,
    timestep_index: int,
    v_hat: torch.Tensor,
    mask: torch.Tensor,
    *,
    start_timestep: int,
    signal_fraction: float,
    family: str,
    iterations: int,
    displacement_cap: float,
    coordinate_scale_angstrom: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float, bool]:
    """Sparse composite correction using a host loop index (no device readback)."""
    x0 = diffusion.reconstruct_x0(x_t, timestep_tensor, v_hat, mask)
    t = int(timestep_index)
    fraction = float(signal_fraction) * (start_timestep - t) / start_timestep
    fraction = min(max(fraction, 0.0), float(signal_fraction))
    active = family == "bond" or scheduled_correction(
        t, start_timestep, length=x_t.shape[1], dtype=x_t.dtype, device=x_t.device
    )
    applied = active and fraction > 0.0 and family in {"bond", "composite"}
    if not applied:
        next_state, _, _ = diffusion.deterministic_reverse_step(x_t, timestep_tensor, v_hat, mask)
        return next_state, x0, fraction, False
    projected_angstrom = project_x0(
        x0 * coordinate_scale_angstrom,
        mask,
        family=family,
        iterations=iterations,
        displacement_cap=displacement_cap,
        scheduled_clash=(family == "composite"),
    )
    projected = projected_angstrom / coordinate_scale_angstrom
    guided = (1.0 - fraction) * x0 + fraction * projected
    alpha, sigma = diffusion.alpha_sigma(timestep_tensor, x_t)
    v_guided = (alpha * x_t - guided) / sigma.clamp_min(1e-8)
    next_state, _, _ = diffusion.deterministic_reverse_step(x_t, timestep_tensor, v_guided, mask)
    return next_state, guided, fraction, True


def schedule_hashes() -> dict[str, str]:
    return {
        str(start): hashlib.sha256(json.dumps(sparse_timesteps(start), separators=(",", ":")).encode()).hexdigest()
        for start in (425, 250, 150)
    }


def verify_reuse(v1_root: str | Path, v1_metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate candidate and identity prefix without reading trajectory payloads.

    Reuse stays rejected unless a byte hash for the v1 sampler implementation
    was recorded. This avoids treating a matching config as proof of matching
    numerical execution.
    """
    root = Path(v1_root)
    journal_path = root / "journal.jsonl"
    if journal_path.is_symlink() or not journal_path.is_file():
        return {"accepted": False, "reason": "v1 journal missing or symlinked", "units": []}
    records = [json.loads(line) for line in journal_path.read_text().splitlines()]
    expected_candidates = v1_metadata["expected_reusable_candidates"]
    expected_identities = v1_metadata["calibration_panel"]
    units: list[dict[str, Any]] = []
    for candidate in expected_candidates:
        for identity in expected_identities:
            index = len(units)
            if index >= 40 or index >= len(records):
                return {"accepted": False, "reason": "v1 completed reuse prefix incomplete", "units": []}
            row = records[index]
            unit = row.get("unit", {})
            expected_unit = {"phase": "calibration", **candidate, **identity}
            if any(unit.get(k) != v for k, v in expected_unit.items()):
                return {"accepted": False, "reason": f"reuse identity mismatch at index {index}", "units": []}
            rel = Path(row.get("path", ""))
            artifact = root / rel
            if rel.is_absolute() or ".." in rel.parts or artifact.is_symlink() or not artifact.is_file():
                return {"accepted": False, "reason": f"reuse artifact missing or unsafe at index {index}", "units": []}
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != row.get("sha256"):
                return {"accepted": False, "reason": f"reuse artifact hash mismatch at index {index}", "units": []}
            units.append({"index": index, "unit": unit, "path": artifact.as_posix(), "sha256": digest})
    if len(units) != 40:
        return {"accepted": False, "reason": "reuse prefix cardinality mismatch", "units": []}
    if not v1_metadata.get("v1_sampler_sha256"):
        return {"accepted": False, "reason": "v1 sampler implementation hash was not recorded", "units": units}
    return {"accepted": True, "reason": "all pinned identity and implementation contracts match", "units": units}


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


__all__ = [
    "NON_AUTHORIZING",
    "angle_delta",
    "clash_delta",
    "clear_kernel_cache",
    "guided_reverse_step",
    "pair_delta",
    "prepare_cache",
    "project_x0",
    "schedule_hashes",
    "scheduled_correction",
    "sparse_timesteps",
    "verify_reuse",
]
