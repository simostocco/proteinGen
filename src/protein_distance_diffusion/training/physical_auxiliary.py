"""Physical auxiliary losses for distance-map diffusion training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from protein_distance_diffusion.diffusion.gaussian import project_symmetric_zero_diagonal


@dataclass(frozen=True)
class PhysicalAuxiliaryLossConfig:
    """Configuration for the stochastic EDM spectral auxiliary loss."""

    enabled: bool = False
    weight: float = 0.0
    warmup_steps: int = 0
    subset_size: int = 64
    subsets_per_sample: int = 1
    negative_weight: float = 1.0
    rank3_weight: float = 1.0
    seed: int = 0
    eps: float = 1e-8


@dataclass(frozen=True)
class PhysicalAuxiliaryLossResult:
    """Scalar loss and diagnostic components."""

    loss: torch.Tensor
    negative_loss: torch.Tensor
    rank3_loss: torch.Tensor
    eligible_fraction: float
    mean_subset_size: float
    subset_count: int


def physical_auxiliary_config_from_mapping(config: dict[str, Any]) -> PhysicalAuxiliaryLossConfig:
    """Build and validate physical auxiliary-loss config from a training config."""
    cfg = PhysicalAuxiliaryLossConfig(
        enabled=bool(config.get("physical_auxiliary_loss_enabled", False)),
        weight=float(config.get("physical_auxiliary_loss_weight", 0.0)),
        warmup_steps=int(config.get("physical_auxiliary_loss_warmup_steps", 0)),
        subset_size=int(config.get("edm_subset_size", 64)),
        subsets_per_sample=int(config.get("edm_subsets_per_sample", 1)),
        negative_weight=float(config.get("edm_negative_weight", 1.0)),
        rank3_weight=float(config.get("edm_rank3_weight", 1.0)),
        seed=int(config.get("physical_auxiliary_seed", 0)),
        eps=float(config.get("edm_loss_eps", 1e-8)),
    )
    if cfg.weight < 0.0:
        raise ValueError("physical_auxiliary_loss_weight must be non-negative")
    if cfg.warmup_steps < 0:
        raise ValueError("physical_auxiliary_loss_warmup_steps must be non-negative")
    if cfg.subset_size < 4:
        raise ValueError("edm_subset_size must be at least 4")
    if cfg.subsets_per_sample < 1:
        raise ValueError("edm_subsets_per_sample must be positive")
    if cfg.negative_weight < 0.0:
        raise ValueError("edm_negative_weight must be non-negative")
    if cfg.rank3_weight < 0.0:
        raise ValueError("edm_rank3_weight must be non-negative")
    if cfg.eps <= 0.0:
        raise ValueError("edm_loss_eps must be positive")
    return cfg


def physical_auxiliary_weight(config: PhysicalAuxiliaryLossConfig, optimizer_step: int) -> float:
    """Return the active auxiliary weight after linear warmup."""
    if not config.enabled or config.weight <= 0.0:
        return 0.0
    if config.warmup_steps <= 0:
        return config.weight
    progress = min(1.0, max(0.0, float(optimizer_step + 1) / float(config.warmup_steps)))
    return config.weight * progress


def _stable_subset_seed(
    *,
    base_seed: int,
    sample_id: str,
    optimizer_step: int,
    microbatch: int,
    subset_index: int,
) -> int:
    payload = f"{base_seed}|{sample_id}|{optimizer_step}|{microbatch}|{subset_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


def deterministic_subset_indices(
    *,
    length: int,
    subset_size: int,
    base_seed: int,
    sample_id: str,
    optimizer_step: int,
    microbatch: int,
    subset_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample deterministic sorted principal-submatrix indices without touching global RNG."""
    n = int(length)
    size = min(int(subset_size), n)
    seed = _stable_subset_seed(
        base_seed=int(base_seed),
        sample_id=str(sample_id),
        optimizer_step=int(optimizer_step),
        microbatch=int(microbatch),
        subset_index=int(subset_index),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(n, generator=generator)[:size].sort().values
    return indices.to(device=device, dtype=torch.long)


def _zero_like_loss(x: torch.Tensor) -> PhysicalAuxiliaryLossResult:
    zero = x.float().sum() * 0.0
    return PhysicalAuxiliaryLossResult(
        loss=zero,
        negative_loss=zero,
        rank3_loss=zero,
        eligible_fraction=0.0,
        mean_subset_size=0.0,
        subset_count=0,
    )


def _spectral_components(distance_matrix: torch.Tensor, *, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    m = int(distance_matrix.shape[-1])
    d = 0.5 * (distance_matrix + distance_matrix.transpose(-1, -2))
    diagonal = torch.eye(m, dtype=torch.bool, device=d.device)
    d = d.masked_fill(diagonal, 0.0)
    identity = torch.eye(m, dtype=torch.float32, device=d.device)
    centering = identity - torch.ones((m, m), dtype=torch.float32, device=d.device) / float(m)
    gram = -0.5 * centering @ d.float().square() @ centering
    eigenvalues = torch.linalg.eigvalsh(0.5 * (gram + gram.transpose(-1, -2)))
    denom_all = eigenvalues.square().sum().clamp_min(float(eps))
    positive = torch.relu(eigenvalues)
    denom_positive = positive.square().sum().clamp_min(float(eps))
    negative_loss = torch.relu(-eigenvalues).square().sum() / denom_all
    rank3_loss = positive[:-3].square().sum() / denom_positive if m > 3 else eigenvalues.sum() * 0.0
    return negative_loss, rank3_loss


def stochastic_edm_spectral_loss(
    *,
    x0_hat_normalized: torch.Tensor,
    pair_mask: torch.Tensor,
    lengths: torch.Tensor,
    normalization_scale: float,
    config: PhysicalAuxiliaryLossConfig,
    sample_ids: list[str] | tuple[str, ...] | None = None,
    optimizer_step: int = 0,
    microbatch: int = 0,
) -> PhysicalAuxiliaryLossResult:
    """Compute stochastic spectral EDM loss on reconstructed physical distances."""
    if x0_hat_normalized.ndim != 4 or x0_hat_normalized.shape[1] != 1:
        raise ValueError("x0_hat_normalized must have shape [B, 1, L, L]")
    if pair_mask.shape != x0_hat_normalized.shape:
        raise ValueError("pair_mask must match x0_hat_normalized shape")
    if lengths.numel() != x0_hat_normalized.shape[0]:
        raise ValueError("lengths must contain one length per sample")
    if not config.enabled:
        return _zero_like_loss(x0_hat_normalized)

    scale = float(normalization_scale)
    if scale <= 0.0:
        raise ValueError("normalization_scale must be positive")
    sample_ids = list(sample_ids) if sample_ids is not None else [str(i) for i in range(x0_hat_normalized.shape[0])]
    if len(sample_ids) != x0_hat_normalized.shape[0]:
        raise ValueError("sample_ids must contain one id per sample")

    with torch.amp.autocast(device_type=x0_hat_normalized.device.type, enabled=False):
        physical = project_symmetric_zero_diagonal(x0_hat_normalized.float() * scale, pair_mask.bool())
        negative_terms: list[torch.Tensor] = []
        rank3_terms: list[torch.Tensor] = []
        subset_sizes: list[int] = []
        eligible = 0
        for batch_index in range(physical.shape[0]):
            n = int(lengths[batch_index].detach().cpu())
            if n < 4:
                continue
            eligible += 1
            for subset_index in range(config.subsets_per_sample):
                indices = deterministic_subset_indices(
                    length=n,
                    subset_size=config.subset_size,
                    base_seed=config.seed,
                    sample_id=sample_ids[batch_index],
                    optimizer_step=optimizer_step,
                    microbatch=microbatch,
                    subset_index=subset_index,
                    device=physical.device,
                )
                sub = physical[batch_index, 0].index_select(0, indices).index_select(1, indices)
                negative, rank3 = _spectral_components(sub, eps=config.eps)
                negative_terms.append(negative)
                rank3_terms.append(rank3)
                subset_sizes.append(int(indices.numel()))
        if not negative_terms:
            return _zero_like_loss(x0_hat_normalized)
        negative_loss = torch.stack(negative_terms).mean()
        rank3_loss = torch.stack(rank3_terms).mean()
        loss = config.negative_weight * negative_loss + config.rank3_weight * rank3_loss
        return PhysicalAuxiliaryLossResult(
            loss=loss,
            negative_loss=negative_loss,
            rank3_loss=rank3_loss,
            eligible_fraction=float(eligible) / float(max(1, physical.shape[0])),
            mean_subset_size=float(sum(subset_sizes)) / float(len(subset_sizes)),
            subset_count=len(subset_sizes),
        )
