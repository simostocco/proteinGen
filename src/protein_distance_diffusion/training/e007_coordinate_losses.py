"""Individually reported E007 coordinate-diffusion objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from protein_distance_diffusion.training.coordinate_diffusion import coordinates_to_distance_matrix


@dataclass(frozen=True)
class CoordinateLossWeights:
    adjacent: float = 0.02
    stratified_pair: float = 0.02
    soft_contact: float = 0.01
    steric_clash: float = 0.005


@dataclass(frozen=True)
class ObjectiveCorrectionWeights:
    """Dimensionless Phase-3C clean-geometry auxiliary coefficients."""

    pair_distance: float = 0.02
    adjacent_distance: float = 0.02
    radius_of_gyration: float = 0.01
    steric_clash: float = 0.005


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def _per_sample_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    dimensions = tuple(range(1, values.ndim))
    return (values * weights).sum(dim=dimensions) / weights.sum(dim=dimensions).clamp_min(1)


def coordinate_objective_correction_losses(
    *,
    v_prediction: torch.Tensor,
    v_target: torch.Tensor,
    predicted_clean_coordinates: torch.Tensor,
    clean_coordinates: torch.Tensor,
    timesteps: torch.Tensor,
    timestep_weights: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_continuity_mask: torch.Tensor,
    auxiliary_weights: ObjectiveCorrectionWeights | None = None,
    clash_distance_normalized: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Return equally protein-weighted Phase-3C objective components.

    Coordinates and distance thresholds use the model's normalized coordinate
    units. Every residue/pair reduction is performed per protein before the
    batch mean, preventing longer proteins from dominating the objective.
    """
    if v_prediction.shape != v_target.shape or v_prediction.shape != clean_coordinates.shape:
        raise ValueError("E007 coordinate objective tensor shapes disagree")
    if predicted_clean_coordinates.shape != clean_coordinates.shape:
        raise ValueError("E007 predicted-clean coordinate shape disagrees")
    if residue_mask.shape != clean_coordinates.shape[:2]:
        raise ValueError("E007 residue mask shape disagrees")
    if chain_continuity_mask.shape != (clean_coordinates.shape[0], clean_coordinates.shape[1] - 1):
        raise ValueError("E007 continuity mask shape disagrees")
    if timesteps.shape != (clean_coordinates.shape[0],):
        raise ValueError("E007 timestep shape disagrees")
    if timestep_weights.ndim != 1 or int(timesteps.max()) >= len(timestep_weights):
        raise ValueError("E007 timestep-weight table does not cover the batch")
    auxiliary_weights = auxiliary_weights or ObjectiveCorrectionWeights()
    if any(value < 0 for value in vars(auxiliary_weights).values()):
        raise ValueError("E007 auxiliary coefficients must be nonnegative")

    coordinate_mask = residue_mask[..., None].expand_as(v_target)
    per_sample_v = _per_sample_masked_mean((v_prediction - v_target).square(), coordinate_mask)
    selected_timestep_weights = timestep_weights.to(v_prediction.device, v_prediction.dtype)[timesteps]
    per_sample_weighted_v = per_sample_v * selected_timestep_weights

    predicted_distances = coordinates_to_distance_matrix(predicted_clean_coordinates, residue_mask)
    target_distances = coordinates_to_distance_matrix(clean_coordinates, residue_mask)
    batch, side = residue_mask.shape
    indices = torch.arange(side, device=residue_mask.device)
    separation = (indices[:, None] - indices[None, :]).abs()
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
    upper_pair_mask = pair_mask & (indices[:, None] < indices[None, :])[None]
    pair_distance = _per_sample_masked_mean((predicted_distances - target_distances).square(), upper_pair_mask)

    adjacent_mask = torch.zeros((batch, side, side), dtype=torch.bool, device=residue_mask.device)
    if side > 1:
        adjacent_valid = chain_continuity_mask.bool() & residue_mask[:, :-1] & residue_mask[:, 1:]
        adjacent_mask[:, :-1, 1:] = torch.diag_embed(adjacent_valid)
    adjacent_distance = _per_sample_masked_mean((predicted_distances - target_distances).square(), adjacent_mask)

    residue_weights = residue_mask.to(clean_coordinates.dtype)[..., None]
    valid_counts = residue_weights.sum(dim=1).clamp_min(1)
    predicted_centroid = (predicted_clean_coordinates * residue_weights).sum(dim=1) / valid_counts
    target_centroid = (clean_coordinates * residue_weights).sum(dim=1) / valid_counts
    predicted_centered = (predicted_clean_coordinates - predicted_centroid[:, None]) * residue_weights
    target_centered = (clean_coordinates - target_centroid[:, None]) * residue_weights
    predicted_radius = torch.linalg.vector_norm(predicted_centered.flatten(1), dim=1) / valid_counts.squeeze(-1).sqrt()
    target_radius = torch.linalg.vector_norm(target_centered.flatten(1), dim=1) / valid_counts.squeeze(-1).sqrt()
    radius_of_gyration = (predicted_radius - target_radius).square()

    nonneighbor_mask = upper_pair_mask & (separation[None] > 1)
    steric_clash = _per_sample_masked_mean(
        F.relu(float(clash_distance_normalized) - predicted_distances).square(), nonneighbor_mask
    )
    raw = {
        "pair_distance": pair_distance,
        "adjacent_distance": adjacent_distance,
        "radius_of_gyration": radius_of_gyration,
        "steric_clash": steric_clash,
    }
    weighted = {
        "pair_distance": raw["pair_distance"] * auxiliary_weights.pair_distance,
        "adjacent_distance": raw["adjacent_distance"] * auxiliary_weights.adjacent_distance,
        "radius_of_gyration": raw["radius_of_gyration"] * auxiliary_weights.radius_of_gyration,
        "steric_clash": raw["steric_clash"] * auxiliary_weights.steric_clash,
    }
    per_sample_auxiliary = sum(weighted.values(), torch.zeros_like(per_sample_v))
    per_sample_total = per_sample_weighted_v + per_sample_auxiliary
    return {
        "coordinate_v": per_sample_v.mean(),
        "weighted_coordinate_v": per_sample_weighted_v.mean(),
        **{f"raw_{name}": value.mean() for name, value in raw.items()},
        **{f"weighted_{name}": value.mean() for name, value in weighted.items()},
        "weighted_auxiliary_total": per_sample_auxiliary.mean(),
        "total": per_sample_total.mean(),
        "per_sample_coordinate_v": per_sample_v,
        "per_sample_weighted_coordinate_v": per_sample_weighted_v,
        "per_sample_auxiliary_total": per_sample_auxiliary,
        "per_sample_total": per_sample_total,
    }


def coordinate_diffusion_losses(
    *,
    v_prediction: torch.Tensor,
    v_target: torch.Tensor,
    clean_coordinates: torch.Tensor,
    predicted_clean_coordinates: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_continuity_mask: torch.Tensor,
    weights: CoordinateLossWeights | None = None,
    coordinate_scale_angstrom: float = 1.0,
    adjacent_huber_beta: float = 0.25,
    clash_distance: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Compute primary coordinate-v loss and conservative geometric auxiliaries."""
    weights = weights or CoordinateLossWeights()
    valid_coordinates = residue_mask[..., None].expand_as(v_target)
    coordinate_v = _masked_mean((v_prediction - v_target).square(), valid_coordinates)
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
    side = residue_mask.shape[1]
    indices = torch.arange(side, device=residue_mask.device)
    separation = (indices[:, None] - indices[None, :]).abs()
    upper = indices[:, None] < indices[None, :]
    valid_pairs = pair_mask & upper[None]
    target_distances = coordinates_to_distance_matrix(clean_coordinates, residue_mask) * coordinate_scale_angstrom
    predicted_distances = (
        coordinates_to_distance_matrix(predicted_clean_coordinates, residue_mask) * coordinate_scale_angstrom
    )

    adjacent_mask = torch.zeros_like(pair_mask)
    if side > 1:
        adjacent_valid = chain_continuity_mask.bool() & residue_mask[:, :-1] & residue_mask[:, 1:]
        adjacent_mask[:, :-1, 1:] |= torch.diag_embed(adjacent_valid)
    adjacent = _masked_mean(
        F.smooth_l1_loss(predicted_distances, target_distances, beta=adjacent_huber_beta, reduction="none"),
        adjacent_mask,
    )

    pair_error = F.smooth_l1_loss(predicted_distances, target_distances, beta=1.0, reduction="none")
    strata = (
        valid_pairs & (separation[None] >= 2) & (separation[None] <= 4),
        valid_pairs & (separation[None] > 4) & (target_distances <= 10.0),
        valid_pairs & (separation[None] > 4) & (target_distances > 10.0),
    )
    stratum_losses = [_masked_mean(pair_error, mask) for mask in strata if bool(mask.any())]
    stratified = torch.stack(stratum_losses).mean() if stratum_losses else coordinate_v.new_zeros(())

    contact_losses = []
    for threshold in (6.0, 8.0, 10.0):
        target_contact = torch.sigmoid((threshold - target_distances) / 0.5)
        predicted_contact = torch.sigmoid((threshold - predicted_distances) / 0.5)
        contact_losses.append(_masked_mean((predicted_contact - target_contact).square(), valid_pairs))
    soft_contact = torch.stack(contact_losses).mean()

    nonneighbor = valid_pairs & (separation[None] > 1)
    steric_clash = _masked_mean(F.relu(clash_distance - predicted_distances).square(), nonneighbor)
    total = (
        coordinate_v
        + weights.adjacent * adjacent
        + weights.stratified_pair * stratified
        + weights.soft_contact * soft_contact
        + weights.steric_clash * steric_clash
    )
    return {
        "total": total,
        "coordinate_v": coordinate_v,
        "adjacent_distance_huber": adjacent,
        "stratified_pair_distance": stratified,
        "soft_contact": soft_contact,
        "steric_clash": steric_clash,
    }
