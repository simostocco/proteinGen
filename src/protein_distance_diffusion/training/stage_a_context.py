"""Versioned E006 Stage-A contextual objective and corruption contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from protein_distance_diffusion.training.codesign import masked_sequence_inputs

STAGE_A_CONTEXT_OBJECTIVE_VERSION = "e006_stage_a_context_objective_v5"
STAGE_A_CONTEXT_OBJECTIVE_V6 = "e006_stage_a_context_objective_v6"
CANONICAL_TOKEN_START = 2
CANONICAL_TOKEN_COUNT = 20


@dataclass(frozen=True)
class ContextCorruption:
    inputs: torch.Tensor
    shuffled_inputs: torch.Tensor
    corrupted_mask: torch.Tensor


def visible_shuffle_inputs(
    inputs: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    seed: int,
    step: int,
) -> torch.Tensor:
    """Shuffle only visible valid tokens, independently within each protein."""
    shuffled = inputs.clone()
    for row in range(inputs.shape[0]):
        positions = torch.nonzero(residue_mask[row].bool() & ~corrupted_mask[row], as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(step) * 3_000_017 + row * 1_000_003)
        permutation = torch.randperm(positions.numel(), generator=generator)
        if torch.equal(permutation, torch.arange(positions.numel())):
            permutation = torch.roll(permutation, 1)
        shuffled[row, positions] = inputs[row, positions[permutation].to(positions.device)]
    return shuffled


def context_corruption(
    targets: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    mask_token_id: int,
    probability: float,
    seed: int,
    step: int,
    allow_zero_probability: bool = False,
) -> ContextCorruption:
    """Create paired ordered and shuffled inputs from one deterministic mask draw."""
    if probability == 0 and allow_zero_probability:
        corrupted = torch.zeros_like(residue_mask, dtype=torch.bool)
        inputs = targets.clone()
    else:
        inputs, corrupted = masked_sequence_inputs(
            targets,
            residue_mask,
            mask_token_id=mask_token_id,
            probability=probability,
            seed=seed,
            step=step,
        )
    if (corrupted & ~residue_mask.bool()).any():
        raise ValueError("E006 Stage-A corruption includes padding")
    if corrupted.any() and not torch.equal(inputs[corrupted], torch.full_like(inputs[corrupted], int(mask_token_id))):
        raise ValueError("E006 Stage-A corrupted positions do not contain the mask token")
    return ContextCorruption(
        inputs=inputs,
        shuffled_inputs=visible_shuffle_inputs(
            inputs,
            corrupted,
            residue_mask,
            seed=seed + 97_409,
            step=step,
        ),
        corrupted_mask=corrupted,
    )


def canonical_corrupted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute canonical 20-way CE exclusively over corrupted valid positions."""
    selected = corrupted_mask.bool() & residue_mask.bool()
    if not selected.any():
        raise ValueError("E006 Stage-A contextual objective requires corrupted valid positions")
    canonical_targets = targets[selected].long() - CANONICAL_TOKEN_START
    if (canonical_targets < 0).any() or (canonical_targets >= CANONICAL_TOKEN_COUNT).any():
        raise ValueError("E006 Stage-A contextual objective received a noncanonical target")
    canonical_logits = logits[selected, CANONICAL_TOKEN_START : CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT]
    return F.cross_entropy(canonical_logits.float(), canonical_targets)


def contextual_stage_a_loss(
    normal_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    targets: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    contrast_weight: float,
    contrast_margin_nats: float,
) -> dict[str, torch.Tensor]:
    """Favor ordered context without optimizing the counterfactual branch upward."""
    if contrast_weight < 0 or contrast_margin_nats < 0:
        raise ValueError("E006 Stage-A context contrast settings must be nonnegative")
    normal = canonical_corrupted_cross_entropy(normal_logits, targets, corrupted_mask, residue_mask)
    shuffled = canonical_corrupted_cross_entropy(shuffled_logits, targets, corrupted_mask, residue_mask)
    penalty = F.relu(normal - shuffled.detach() + float(contrast_margin_nats))
    weighted = penalty * float(contrast_weight)
    return {
        "sequence": normal,
        "shuffled_sequence": shuffled,
        "context_gap": shuffled - normal,
        "context_contrast": penalty,
        "context_contrast_weighted": weighted,
        "geometry": normal.new_zeros(()),
        "consistency": normal.new_zeros(()),
        "total": normal + weighted,
    }


@dataclass(frozen=True)
class PairedDropoutEvidence:
    cpu_rng_paired: bool
    cuda_rng_paired: bool
    global_rng_advanced_once: bool


def _rng_states() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    return torch.get_rng_state().clone(), (
        [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    )


def _set_rng_states(states: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    torch.set_rng_state(states[0])
    if states[1] is not None:
        torch.cuda.set_rng_state_all(states[1])


def _states_equal(
    first: tuple[torch.Tensor, list[torch.Tensor] | None],
    second: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> tuple[bool, bool]:
    cpu_equal = torch.equal(first[0], second[0])
    if first[1] is None or second[1] is None:
        return cpu_equal, first[1] is None and second[1] is None
    return cpu_equal, len(first[1]) == len(second[1]) and all(
        torch.equal(left, right) for left, right in zip(first[1], second[1], strict=True)
    )


def paired_dropout_forwards(
    model: torch.nn.Module,
    normal_inputs: torch.Tensor,
    shuffled_inputs: torch.Tensor,
    residue_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, PairedDropoutEvidence]:
    """Run paired branches with identical stochastic state and one net RNG advance."""
    before = _rng_states()
    normal_logits = model.forward_sequence_pretraining(normal_inputs, residue_mask)
    after_normal = _rng_states()
    _set_rng_states(before)
    restored_before = _rng_states()
    shuffled_logits = model.forward_sequence_pretraining(shuffled_inputs, residue_mask)
    after_shuffled = _rng_states()
    cpu_paired, cuda_paired = _states_equal(after_normal, after_shuffled)
    restored_cpu, restored_cuda = _states_equal(before, restored_before)
    _set_rng_states(after_normal)
    final_state = _rng_states()
    final_cpu, final_cuda = _states_equal(after_normal, final_state)
    return (
        normal_logits,
        shuffled_logits,
        PairedDropoutEvidence(
            cpu_rng_paired=restored_cpu and cpu_paired,
            cuda_rng_paired=restored_cuda and cuda_paired,
            global_rng_advanced_once=final_cpu and final_cuda,
        ),
    )


def _canonical_token_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    canonical_targets = targets[selected].long() - CANONICAL_TOKEN_START
    if (canonical_targets < 0).any() or (canonical_targets >= CANONICAL_TOKEN_COUNT).any():
        raise ValueError("E006 Stage-A v6 received a noncanonical target")
    canonical_logits = logits[selected, CANONICAL_TOKEN_START : CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT]
    return F.cross_entropy(canonical_logits.float(), canonical_targets, reduction="none")


def contextual_stage_a_loss_v6(
    normal_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    targets: torch.Tensor,
    corrupted_mask: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    contrast_weight: float,
    contrast_margin_nats: float,
    paired_dropout_evidence: PairedDropoutEvidence | None = None,
) -> dict[str, torch.Tensor]:
    """Canonical CE plus an equal-protein, per-sample contextual hinge."""
    if contrast_weight < 0 or contrast_margin_nats < 0:
        raise ValueError("E006 Stage-A v6 context contrast settings must be nonnegative")
    selected = corrupted_mask.bool() & residue_mask.bool()
    if not selected.any():
        raise ValueError("E006 Stage-A v6 requires corrupted valid positions")
    normal_tokens = _canonical_token_losses(normal_logits, targets, selected)
    shuffled_tokens = _canonical_token_losses(shuffled_logits, targets, selected)
    normal = normal_tokens.mean()
    shuffled = shuffled_tokens.mean()
    coordinates = torch.nonzero(selected, as_tuple=False)
    sample_indices = coordinates[:, 0]
    sample_margins = []
    sample_hinges = []
    for sample_index in range(targets.shape[0]):
        member = sample_indices == sample_index
        if not member.any():
            continue
        margin = normal_tokens[member].mean() - shuffled_tokens[member].mean()
        sample_margins.append(margin)
        sample_hinges.append(F.relu(margin + float(contrast_margin_nats)))
    if not sample_hinges:
        raise ValueError("E006 Stage-A v6 has no eligible per-sample contrast terms")
    margins = torch.stack(sample_margins)
    hinges = torch.stack(sample_hinges)
    penalty = hinges.mean()
    weighted = penalty * float(contrast_weight)
    quantiles = torch.quantile(
        margins.detach().float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=margins.device)
    )
    paired = paired_dropout_evidence or PairedDropoutEvidence(False, False, False)
    paired_verified = paired.cpu_rng_paired and paired.cuda_rng_paired and paired.global_rng_advanced_once
    return {
        "sequence": normal,
        "shuffled_sequence": shuffled,
        "context_gap": shuffled - normal,
        "normal_minus_shuffled_batch_margin": normal - shuffled,
        "context_contrast": penalty,
        "context_contrast_weighted": weighted,
        "context_active_hinge_sample_fraction": (hinges > 0).float().mean(),
        "context_sample_margin_mean": margins.detach().mean(),
        "context_sample_margin_q10": quantiles[0],
        "context_sample_margin_q25": quantiles[1],
        "context_sample_margin_q50": quantiles[2],
        "context_sample_margin_q75": quantiles[3],
        "context_sample_margin_q90": quantiles[4],
        "paired_dropout_verified": normal.new_tensor(float(paired_verified)),
        "geometry": normal.new_zeros(()),
        "consistency": normal.new_zeros(()),
        "total": normal + weighted,
    }
