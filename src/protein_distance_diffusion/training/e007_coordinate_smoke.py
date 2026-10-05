"""Bounded non-authorizing synthetic-polymer learning smoke for E007."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import resource
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.coordinate_equivariance import (
    equivariance_criterion,
    equivariance_metrics,
    strict_equivariance_numerics,
)
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.coordinate_diffusion import (
    CoordinateVPDiffusion,
    center_coordinates,
    centered_coordinate_noise,
    coordinates_to_distance_matrix,
)
from protein_distance_diffusion.training.e007_coordinate_losses import (
    CoordinateLossWeights,
    coordinate_diffusion_losses,
)

SMOKE_VERSION = "e007_coordinate_synthetic_polymer_smoke_v1"
SMOKE_VERSION_V2 = "e007_coordinate_synthetic_polymer_smoke_v2"
FAMILIES = ("helix", "hairpin", "open_ring", "compact_smooth")
CLASSIFICATIONS = (
    "synthetic_learning_verified",
    "memorization_without_heldout_learning",
    "denoising_learned_but_sampling_not_learned",
    "equivariant_lifting_underexpressive",
    "sampler_contract_failed",
    "numerically_unstable",
    "inconclusive_requires_review",
)


@dataclass(frozen=True)
class PolymerSample:
    sample_id: str
    split: str
    family: str
    length: int
    parameter_seed: int
    coordinates: torch.Tensor
    coordinate_sha256: str
    shape_sha256: str
    target_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("version") not in {SMOKE_VERSION, SMOKE_VERSION_V2}:
        raise ValueError("E007 coordinate smoke configuration version contradiction")
    seeds = list(map(int, payload.get("seeds", [])))
    lengths = list(map(int, payload.get("lengths", [])))
    updates = int(payload.get("optimizer_updates_per_seed", 0))
    maximum = int((payload.get("decision_thresholds") or {}).get("maximum_optimizer_updates_per_seed", 0))
    evaluation_updates = set(map(int, payload.get("evaluation_updates", [])))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("E007 coordinate smoke requires at least three unique seeds")
    if not lengths or len(set(lengths)) != len(lengths) or any(length < 4 for length in lengths):
        raise ValueError("E007 coordinate smoke lengths must be unique and at least four")
    if not any(length % 8 for length in lengths):
        raise ValueError("E007 coordinate smoke must include a nonmultiple-of-eight length")
    hard_maximum = 1000 if payload.get("version") == SMOKE_VERSION_V2 else 300
    if not 0 < updates <= maximum <= hard_maximum:
        raise ValueError("E007 coordinate smoke update bound is invalid")
    if 0 not in evaluation_updates or updates not in evaluation_updates:
        raise ValueError("E007 coordinate smoke must evaluate initialization and completion")
    return payload


def _constant_bond_trace(points: torch.Tensor, bond_length: float) -> torch.Tensor:
    deltas = points[1:] - points[:-1]
    norms = torch.linalg.vector_norm(deltas, dim=-1, keepdim=True)
    if bool((norms <= 1e-8).any()):
        raise ValueError("degenerate synthetic polymer step")
    steps = bond_length * deltas / norms
    trace = torch.cat((torch.zeros((1, 3), dtype=torch.float64), torch.cumsum(steps, dim=0)))
    return trace - trace.mean(dim=0, keepdim=True)


def generate_polymer(family: str, length: int, parameter_seed: int, *, bond_length: float = 3.8) -> torch.Tensor:
    """Generate one deterministic centered C-alpha-like trace."""
    if family not in FAMILIES or length < 4:
        raise ValueError("invalid synthetic polymer family or length")
    generator = torch.Generator().manual_seed(parameter_seed)
    index = torch.arange(length, dtype=torch.float64)
    jitter = float(torch.rand((), generator=generator))
    if family == "helix":
        angle = (0.72 + 0.08 * jitter) * index
        radius = 2.0 + 0.8 * float(torch.rand((), generator=generator))
        rise = 0.55 + 0.3 * float(torch.rand((), generator=generator))
        raw = torch.stack((radius * angle.cos(), radius * angle.sin(), rise * index), dim=-1)
    elif family == "hairpin":
        turn = (length - 1) / 2
        x = torch.where(index <= turn, index, 2 * turn - index)
        y = 2.5 * torch.tanh((index - turn) / 1.5)
        z = (0.15 + 0.2 * jitter) * torch.sin(index * 0.55)
        raw = torch.stack((x, y, z), dim=-1)
    elif family == "open_ring":
        angle = torch.linspace(0, (1.65 + 0.15 * jitter) * math.pi, length, dtype=torch.float64)
        radius = 5.0 + 2.0 * float(torch.rand((), generator=generator))
        raw = torch.stack((radius * angle.cos(), radius * angle.sin(), 0.4 * torch.sin(2 * angle)), dim=-1)
    else:
        directions = torch.randn((length - 1, 3), generator=generator, dtype=torch.float64)
        for step in range(1, length - 1):
            directions[step] = 0.72 * directions[step - 1] + 0.28 * directions[step]
        raw = torch.cat((torch.zeros((1, 3), dtype=torch.float64), torch.cumsum(directions, dim=0)))
    return _constant_bond_trace(raw, bond_length).float()


def build_polymer_panels(config: dict[str, Any]) -> tuple[list[PolymerSample], list[PolymerSample], dict[str, Any]]:
    """Build balanced train and held-out panels with disjoint parameter seeds."""
    lengths = tuple(int(value) for value in config["lengths"])
    families = tuple(str(value) for value in config["families"])
    if families != FAMILIES or len(set(lengths)) != len(lengths):
        raise ValueError("E007 polymer families/lengths must be unique and canonical")
    panels: dict[str, list[PolymerSample]] = {"train": [], "heldout": []}
    split_settings = {
        "train": (int(config["train_replicates_per_family_length"]), int(config["train_parameter_seed_offset"])),
        "heldout": (
            int(config["heldout_replicates_per_family_length"]),
            int(config["heldout_parameter_seed_offset"]),
        ),
    }
    for split, (replicates, offset) in split_settings.items():
        for family_index, family in enumerate(families):
            for length_index, length in enumerate(lengths):
                for replicate in range(replicates):
                    parameter_seed = offset + family_index * 10000 + length_index * 100 + replicate
                    coordinates = generate_polymer(
                        family, length, parameter_seed, bond_length=float(config["bond_length_angstrom"])
                    )
                    coordinate_hash = _tensor_sha256(coordinates)
                    shape_hash = _tensor_sha256(torch.cdist(coordinates, coordinates))
                    sample_id = f"{split}:{family}:N{length}:r{replicate}:s{parameter_seed}"
                    target_hash = _canonical_hash(
                        {"sample_id": sample_id, "coordinate_sha256": coordinate_hash, "target": "coordinate_v"}
                    )
                    panels[split].append(
                        PolymerSample(
                            sample_id,
                            split,
                            family,
                            length,
                            parameter_seed,
                            coordinates,
                            coordinate_hash,
                            shape_hash,
                            target_hash,
                        )
                    )
    train_parameters = {sample.parameter_seed for sample in panels["train"]}
    heldout_parameters = {sample.parameter_seed for sample in panels["heldout"]}
    train_hashes = {sample.coordinate_sha256 for sample in panels["train"]}
    heldout_hashes = {sample.coordinate_sha256 for sample in panels["heldout"]}
    train_shapes = {sample.shape_sha256 for sample in panels["train"]}
    heldout_shapes = {sample.shape_sha256 for sample in panels["heldout"]}
    if train_parameters & heldout_parameters or train_hashes & heldout_hashes or train_shapes & heldout_shapes:
        raise ValueError("E007 synthetic train/held-out panels are not disjoint")
    metadata = {
        "family_counts": {
            split: dict(sorted(Counter(sample.family for sample in values).items())) for split, values in panels.items()
        },
        "length_counts": {
            split: dict(sorted(Counter(sample.length for sample in values).items())) for split, values in panels.items()
        },
        "sample_id_sha256": {
            split: _canonical_hash([sample.sample_id for sample in values]) for split, values in panels.items()
        },
        "coordinate_sha256": {
            split: _canonical_hash([sample.coordinate_sha256 for sample in values]) for split, values in panels.items()
        },
        "target_sha256": {
            split: _canonical_hash([sample.target_sha256 for sample in values]) for split, values in panels.items()
        },
        "shape_sha256": {
            split: _canonical_hash([sample.shape_sha256 for sample in values]) for split, values in panels.items()
        },
        "parameter_seed_disjoint": True,
        "coordinate_hash_disjoint": True,
        "rigid_shape_hash_disjoint": True,
    }
    return panels["train"], panels["heldout"], metadata


def proper_rotation(seed: int, *, device: torch.device | str = "cpu") -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((3, 3), generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(matrix)
    q = q * torch.sign(torch.diag(r)).masked_fill(torch.diag(r) == 0, 1)
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q.float().to(device)


def _sample_batch(sample: PolymerSample, scale: float, device: torch.device) -> dict[str, torch.Tensor]:
    coordinates = (sample.coordinates / scale).to(device)[None]
    mask = torch.ones((1, sample.length), dtype=torch.bool, device=device)
    return {
        "coordinates": coordinates,
        "residue_mask": mask,
        "continuity": torch.ones((1, sample.length - 1), dtype=torch.bool, device=device),
        "lengths": torch.tensor([sample.length], dtype=torch.long, device=device),
    }


def verify_oracle_sampler_contract(
    diffusion: CoordinateVPDiffusion,
    *,
    length: int = 17,
    seed: int = 707,
    device: torch.device | str = "cpu",
    tolerance: float = 2e-5,
) -> dict[str, Any]:
    """Verify v algebra, one-step reversal, centering, masks, and full sampling."""
    device = torch.device(device)
    mask = torch.ones((1, length), dtype=torch.bool, device=device)
    x0 = center_coordinates(generate_polymer("helix", length, seed).to(device)[None] / 10.0, mask)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    noise = centered_coordinate_noise(x0, mask, generator=generator)
    timestep = torch.tensor([diffusion.timesteps - 1], dtype=torch.long, device=device)
    alpha, sigma = diffusion.alpha_sigma(timestep, x0)
    xt = center_coordinates(alpha * x0 + sigma * noise, mask)
    exact_v = center_coordinates(alpha * noise - sigma * x0, mask)
    previous, reconstructed, reconstructed_noise = diffusion.deterministic_reverse_step(xt, timestep, exact_v, mask)
    previous_timestep = timestep - 1
    previous_alpha, previous_sigma = diffusion.alpha_sigma(previous_timestep, x0)
    expected_previous = center_coordinates(previous_alpha * x0 + previous_sigma * noise, mask)

    class OracleDenoiser:
        def __call__(self, coordinates, timesteps, requested_lengths, residue_mask, continuity_mask):
            del requested_lengths, continuity_mask
            local_alpha, local_sigma = diffusion.alpha_sigma(timesteps, coordinates)
            local_noise = (coordinates - local_alpha * x0) / local_sigma.clamp_min(1e-12)
            return {"v_prediction": center_coordinates(local_alpha * local_noise - local_sigma * x0, residue_mask)}

    oracle = OracleDenoiser()
    sampled = diffusion.sample(
        oracle,
        length=length,
        seed=seed + 2,
        device=device,
        return_trajectory=True,
    )
    repeated = diffusion.sample(oracle, length=length, seed=seed + 2, device=device)
    trajectory = sampled["trajectory"]
    trajectory_centered = all(
        bool(torch.allclose(value.sum(dim=1), torch.zeros((1, 3), device=device), atol=tolerance))
        for value in trajectory
    )
    trajectory_masked = all(bool(torch.count_nonzero(value * ~mask[..., None]) == 0) for value in trajectory)
    result = {
        "x0_reconstruction_max_error": float((reconstructed - x0).abs().max()),
        "epsilon_reconstruction_max_error": float((reconstructed_noise - noise).abs().max()),
        "one_step_reverse_max_error": float((previous - expected_previous).abs().max()),
        "full_reverse_x0_max_error": float((sampled["coordinates"] - x0).abs().max()),
        "trajectory_state_count": len(trajectory),
        "trajectory_centered": trajectory_centered,
        "trajectory_masked": trajectory_masked,
        "fixed_seed_deterministic": bool(torch.equal(sampled["coordinates"], repeated["coordinates"])),
    }
    result["passed"] = bool(
        max(
            result["x0_reconstruction_max_error"],
            result["epsilon_reconstruction_max_error"],
            result["one_step_reverse_max_error"],
            result["full_reverse_x0_max_error"],
        )
        <= tolerance
        and trajectory_centered
        and trajectory_masked
        and result["fixed_seed_deterministic"]
    )
    return result


def _paired_diffusion_batch(
    sample: PolymerSample,
    diffusion: CoordinateVPDiffusion,
    *,
    scale: float,
    draw_seed: int,
    rotate: bool,
    device: torch.device,
    forced_timestep: int | None = None,
) -> tuple[dict[str, torch.Tensor], Any]:
    batch = _sample_batch(sample, scale, device)
    if rotate:
        batch["coordinates"] = batch["coordinates"] @ proper_rotation(draw_seed + 17, device=device)
    generator = torch.Generator(device=device).manual_seed(draw_seed)
    timestep = (
        torch.tensor([forced_timestep], dtype=torch.long, device=device)
        if forced_timestep is not None
        else torch.randint(diffusion.timesteps, (1,), generator=generator, device=device)
    )
    diffused = diffusion.make_training_batch(
        batch["coordinates"], batch["residue_mask"], timesteps=timestep, generator=generator
    )
    return batch, diffused


def _kabsch_mse(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.double() - predicted.double().mean(dim=0, keepdim=True)
    target = target.double() - target.double().mean(dim=0, keepdim=True)
    u, _, vh = torch.linalg.svd(predicted.T @ target)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return float(((predicted @ rotation - target) ** 2).mean())


def _evaluation_metrics(
    model: EquivariantPairCoordinateUNet,
    panel: Iterable[PolymerSample],
    diffusion: CoordinateVPDiffusion,
    *,
    scale: float,
    corruption_seed: int,
    device: torch.device,
    zero_message: bool = False,
    forced_timestep: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, float | str | int]] = []
    noisy_hashes: list[str] = []
    target_hashes: list[str] = []
    model.eval()
    with torch.no_grad():
        for panel_index, sample in enumerate(panel):
            batch, diffused = _paired_diffusion_batch(
                sample,
                diffusion,
                scale=scale,
                draw_seed=corruption_seed + panel_index * 997,
                rotate=False,
                device=device,
                forced_timestep=forced_timestep,
            )
            noisy_hashes.append(_tensor_sha256(diffused.noisy_coordinates))
            target_hashes.append(_tensor_sha256(diffused.coordinate_v_target))
            if zero_message:
                prediction = torch.zeros_like(diffused.coordinate_v_target)
            else:
                prediction = model(
                    diffused.noisy_coordinates,
                    diffused.timesteps,
                    batch["lengths"],
                    batch["residue_mask"],
                    batch["continuity"],
                )["v_prediction"]
            reconstructed = diffusion.reconstruct_x0(
                diffused.noisy_coordinates, diffused.timesteps, prediction, batch["residue_mask"]
            )
            predicted_angstrom = reconstructed[0] * scale
            target_angstrom = batch["coordinates"][0] * scale
            predicted_distances = coordinates_to_distance_matrix(predicted_angstrom, diagnostic_float64=True)
            target_distances = coordinates_to_distance_matrix(target_angstrom, diagnostic_float64=True)
            upper = torch.triu(torch.ones_like(predicted_distances, dtype=torch.bool), diagonal=1)
            nonneighbor = torch.triu(torch.ones_like(upper), diagonal=2)
            adjacent_prediction = torch.diagonal(predicted_distances, offset=1)
            adjacent_target = torch.diagonal(target_distances, offset=1)
            contact_losses: dict[str, float] = {}
            for threshold in (6.0, 8.0, 10.0):
                left = torch.sigmoid((threshold - predicted_distances) / 0.5)
                right = torch.sigmoid((threshold - target_distances) / 0.5)
                contact_losses[f"soft_contact_loss_{int(threshold)}a"] = float(
                    ((left[upper] - right[upper]) ** 2).mean()
                )
            predicted_rg = torch.sqrt((predicted_angstrom.square().sum(-1)).mean())
            target_rg = torch.sqrt((target_angstrom.square().sum(-1)).mean())
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "length": sample.length,
                    "coordinate_v_mse": float((prediction - diffused.coordinate_v_target).square().mean()),
                    "aligned_x0_coordinate_mse_angstrom2": _kabsch_mse(predicted_angstrom, target_angstrom),
                    "pair_distance_rmse_angstrom": float(
                        torch.sqrt(((predicted_distances[upper] - target_distances[upper]) ** 2).mean())
                    ),
                    "adjacent_distance_rmse_angstrom": float(
                        torch.sqrt(((adjacent_prediction - adjacent_target) ** 2).mean())
                    ),
                    **contact_losses,
                    "soft_contact_loss": float(np.mean(list(contact_losses.values()))),
                    "nonneighbor_clash_fraction": float((predicted_distances[nonneighbor] < 3.0).float().mean()),
                    "radius_of_gyration_error_angstrom": float(torch.abs(predicted_rg - target_rg)),
                }
            )
    metric_names = [name for name in rows[0] if name not in {"sample_id", "length"}]
    return {
        "count": len(rows),
        "corruption_identity_sha256": _canonical_hash(
            [(row["sample_id"], corruption_seed + index * 997) for index, row in enumerate(rows)]
        ),
        "noisy_coordinate_sha256": _canonical_hash(noisy_hashes),
        "coordinate_v_target_sha256": _canonical_hash(target_hashes),
        "means": {name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names},
        "per_sample": rows,
        "forced_timestep": forced_timestep,
    }


def _timestep_stratified_metrics(
    model: EquivariantPairCoordinateUNet,
    panel: list[PolymerSample],
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    *,
    scale: float,
    corruption_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    configured = config.get("timestep_evaluation_bins") or [
        {"name": "very_low_noise", "fraction": 0.0},
        {"name": "low_noise", "fraction": 0.25},
        {"name": "intermediate_noise", "fraction": 0.5},
        {"name": "high_noise", "fraction": 0.75},
        {"name": "very_high_noise", "fraction": 1.0},
    ]
    result = {}
    for record in configured:
        fraction = float(record["fraction"])
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("E007 timestep-bin fractions must be in [0,1]")
        timestep = int(round(fraction * (diffusion.timesteps - 1)))
        metrics = _evaluation_metrics(
            model,
            panel,
            diffusion,
            scale=scale,
            corruption_seed=corruption_seed,
            device=device,
            forced_timestep=timestep,
        )
        result[str(record["name"])] = {
            "fraction": fraction,
            "timestep": timestep,
            "coordinate_v_mse": metrics["means"]["coordinate_v_mse"],
            "pair_distance_rmse_angstrom": metrics["means"]["pair_distance_rmse_angstrom"],
            "adjacent_distance_rmse_angstrom": metrics["means"]["adjacent_distance_rmse_angstrom"],
            "radius_of_gyration_error_angstrom": metrics["means"]["radius_of_gyration_error_angstrom"],
            "coordinate_v_target_sha256": metrics["coordinate_v_target_sha256"],
            "noisy_coordinate_sha256": metrics["noisy_coordinate_sha256"],
        }
    return result


def _matrix_geometry(coordinates: torch.Tensor) -> dict[str, float | bool]:
    coordinates = coordinates.double()
    distances = coordinates_to_distance_matrix(coordinates, diagnostic_float64=True)
    length = len(coordinates)
    upper = torch.triu(torch.ones((length, length), dtype=torch.bool, device=coordinates.device), diagonal=1)
    nonneighbor = torch.triu(torch.ones_like(upper), diagonal=2)
    squared = distances.square()
    gram = -0.5 * (squared - squared.mean(0)[None] - squared.mean(1)[:, None] + squared.mean())
    eigenvalues = torch.linalg.eigvalsh(gram.double())
    scale = max(float(eigenvalues.abs().max()), 1.0)
    negative = eigenvalues[eigenvalues < -1e-6 * scale].abs().sum()
    positive = eigenvalues.clamp_min(0)
    rank_tail = positive[:-3].sum() if length > 3 else positive.new_zeros(())
    triangle = (distances[:, None, :] - distances[:, :, None] - distances[None, :, :]).clamp_min(0)
    adjacent = torch.diagonal(distances, offset=1)
    return {
        "finite_coordinates": bool(torch.isfinite(coordinates).all()),
        "centre_of_mass_magnitude": float(torch.linalg.vector_norm(coordinates.mean(0))),
        "adjacent_distance_mean": float(adjacent.mean()),
        "adjacent_distance_std": float(adjacent.std(unbiased=False)),
        "clash_fraction": float((distances[nonneighbor] < 3.0).float().mean()),
        "radius_of_gyration": float(torch.sqrt(coordinates.square().sum(-1).mean())),
        "symmetry_error": float((distances - distances.T).abs().max()),
        "diagonal_error": float(torch.diagonal(distances).abs().max()),
        "minimum_distance": float(distances.min()),
        "maximum_triangle_violation": float(triangle.max()),
        "centred_gram_negative_eigenmass_fraction": float(negative / positive.sum().clamp_min(1e-12)),
        "rank3_residual_fraction": float(rank_tail / positive.sum().clamp_min(1e-12)),
        "rank3_reconstruction_error": float(torch.sqrt((rank_tail / max(length, 1)).clamp_min(0))),
    }


def _sampling_panel(
    model: EquivariantPairCoordinateUNet,
    diffusion: CoordinateVPDiffusion,
    config: dict[str, Any],
    *,
    seed_offset: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    model.eval()
    for index, length in enumerate(map(int, config["sampling_lengths"])):
        seed = int(config["sampling_seed"]) + seed_offset + index
        sampled = diffusion.sample(model, length=length, seed=seed, device=device)
        coordinates = sampled["coordinates"][0] * float(config["coordinate_scale_angstrom"])
        rows.append({"length": length, "seed": seed, **_matrix_geometry(coordinates)})
    return rows


def _gaussian_sampling_baseline(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, length in enumerate(map(int, config["sampling_lengths"])):
        seed = int(config["sampling_seed"]) + index
        generator = torch.Generator().manual_seed(seed)
        coordinates = torch.randn((length, 3), generator=generator) * float(config["coordinate_scale_angstrom"])
        coordinates -= coordinates.mean(0, keepdim=True)
        rows.append({"length": length, "seed": seed, **_matrix_geometry(coordinates)})
    return rows


def _target_distribution(samples: Iterable[PolymerSample], lengths: set[int]) -> dict[str, float]:
    rows = [_matrix_geometry(sample.coordinates) for sample in samples if sample.length in lengths]
    return {
        "adjacent_distance_mean": float(np.mean([row["adjacent_distance_mean"] for row in rows])),
        "radius_of_gyration_mean": float(np.mean([row["radius_of_gyration"] for row in rows])),
        "clash_fraction_mean": float(np.mean([row["clash_fraction"] for row in rows])),
    }


def _sampling_distribution_error(rows: list[dict[str, Any]], target: dict[str, float]) -> dict[str, float]:
    adjacent = float(np.mean([row["adjacent_distance_mean"] for row in rows]))
    radius = float(np.mean([row["radius_of_gyration"] for row in rows]))
    return {
        "adjacent_absolute_error": abs(adjacent - target["adjacent_distance_mean"]),
        "radius_absolute_error": abs(radius - target["radius_of_gyration_mean"]),
    }


def _sampling_geometry_valid(rows: list[dict[str, Any]], tolerance: float) -> bool:
    for row in rows:
        scale = max(float(row["adjacent_distance_mean"]), 1.0)
        if not row["finite_coordinates"]:
            return False
        if float(row["symmetry_error"]) > tolerance * scale:
            return False
        if float(row["diagonal_error"]) > tolerance * scale:
            return False
        if float(row["minimum_distance"]) < -tolerance * scale:
            return False
        if float(row["maximum_triangle_violation"]) > tolerance * scale:
            return False
        if float(row["centred_gram_negative_eigenmass_fraction"]) > tolerance:
            return False
        if float(row["rank3_residual_fraction"]) > tolerance:
            return False
    return True


def _joint_polymer_quality(
    rows: list[dict[str, Any]], target: dict[str, float], thresholds: dict[str, Any]
) -> dict[str, float | bool]:
    adjacent = float(np.mean([row["adjacent_distance_mean"] for row in rows]))
    radius = float(np.mean([row["radius_of_gyration"] for row in rows]))
    clash = float(np.mean([row["clash_fraction"] for row in rows]))
    adjacent_error = abs(adjacent - target["adjacent_distance_mean"]) / max(target["adjacent_distance_mean"], 1e-12)
    radius_error = abs(radius - target["radius_of_gyration_mean"]) / max(target["radius_of_gyration_mean"], 1e-12)
    clash_excess = max(0.0, clash - target["clash_fraction_mean"])
    joint = (adjacent_error + radius_error + clash_excess) / 3.0
    passed = (
        adjacent_error <= float(thresholds["maximum_target_normalized_adjacent_error"])
        and radius_error <= float(thresholds["maximum_target_normalized_radius_error"])
        and clash_excess <= float(thresholds["maximum_clash_fraction_excess"])
        and joint <= float(thresholds["maximum_joint_polymer_quality_error"])
    )
    return {
        "target_normalized_adjacent_error": adjacent_error,
        "target_normalized_radius_error": radius_error,
        "clash_fraction": clash,
        "target_clash_fraction": target["clash_fraction_mean"],
        "clash_fraction_excess": clash_excess,
        "joint_error": joint,
        "passed": passed,
    }


def _trained_contract_checks(
    model: EquivariantPairCoordinateUNet,
    *,
    tolerance: float,
    device: torch.device,
) -> dict[str, bool]:
    model.eval()
    torch.manual_seed(711)
    coordinates = torch.randn((1, 17, 3), device=device)
    mask = torch.ones((1, 17), dtype=torch.bool, device=device)
    continuity = torch.ones((1, 16), dtype=torch.bool, device=device)
    timestep = torch.tensor([7], device=device)
    length = torch.tensor([17], device=device)
    rotation = proper_rotation(82, device=device)
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device))
    padded_coordinates = torch.nn.functional.pad(coordinates, (0, 0, 0, 6))
    padded_mask = torch.nn.functional.pad(mask, (0, 6), value=False)
    padded_continuity = torch.nn.functional.pad(continuity, (0, 6), value=False)
    with strict_equivariance_numerics(deterministic=True):
        rotated_coordinates = coordinates @ rotation
        reflected_coordinates = coordinates @ reflection
        base_result = model(coordinates, timestep, length, mask, continuity)
        rotated_result = model(rotated_coordinates, timestep, length, mask, continuity)
        reflected_result = model(reflected_coordinates, timestep, length, mask, continuity)
        translated = model(coordinates + 4.0, timestep, length, mask, continuity)["v_prediction"]
        padded = model(padded_coordinates, timestep, length, padded_mask, padded_continuity)["v_prediction"]
        repeated = model(coordinates, timestep, length, mask, continuity)["v_prediction"]
    rotation_metrics = equivariance_metrics(
        reference=base_result,
        transformed=rotated_result,
        transformation=rotation,
        reference_coordinates=coordinates,
        transformed_coordinates=rotated_coordinates,
        residue_mask=mask,
    )
    reflection_metrics = equivariance_metrics(
        reference=base_result,
        transformed=reflected_result,
        transformation=reflection,
        reference_coordinates=coordinates,
        transformed_coordinates=reflected_coordinates,
        residue_mask=mask,
    )
    rotation_passed = equivariance_criterion(
        rotation_metrics,
        absolute_tolerance=tolerance,
        relative_l2_tolerance=tolerance,
        coefficient_tolerance=tolerance,
    )["passed"]
    reflection_passed = equivariance_criterion(
        reflection_metrics,
        absolute_tolerance=tolerance,
        relative_l2_tolerance=tolerance,
        coefficient_tolerance=tolerance,
    )["passed"]
    base = base_result["v_prediction"]
    return {
        "proper_rotation_equivariance": bool(rotation_passed),
        "translation_invariance": bool(torch.allclose(translated, base, atol=tolerance, rtol=tolerance)),
        "reflection_equivariance": bool(reflection_passed),
        "padding_invariance": bool(torch.allclose(padded[:, :17], base, atol=tolerance, rtol=tolerance)),
        "padding_exact_zero": bool(torch.count_nonzero(padded[:, 17:]) == 0),
        "fixed_input_determinism": bool(torch.equal(repeated, base)),
    }


def _parameter_change(model: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    squared = 0.0
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            squared += float((value.detach().cpu() - initial[name]).double().square().sum())
    return math.sqrt(squared)


def require_finite_training_state(
    loss: torch.Tensor,
    model: torch.nn.Module,
    *,
    seed: int,
    update: int,
) -> dict[str, torch.Tensor]:
    """Reject non-finite losses and missing/non-finite trainable gradients."""
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"E007 non-finite synthetic loss at seed={seed}, update={update}")
    named_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    missing_gradients = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    if (
        missing_gradients
        or not named_gradients
        or any(not bool(torch.isfinite(gradient).all()) for gradient in named_gradients.values())
    ):
        raise FloatingPointError(
            f"E007 missing/non-finite synthetic gradient at seed={seed}, update={update}; "
            f"missing={missing_gradients[:20]}"
        )
    return named_gradients


def classify_smoke(
    seed_reports: list[dict[str, Any]],
    thresholds: dict[str, Any],
    *,
    sampler_contract_passed: bool = True,
) -> str:
    """Apply the predeclared Phase-3B decision table."""
    if any(not report["finite_losses_and_gradients"] or report["successful_updates"] == 0 for report in seed_reports):
        return "numerically_unstable"
    if not sampler_contract_passed:
        return "sampler_contract_failed"
    train_improved = all(
        report["relative_improvements"]["train_coordinate_v_mse"]
        >= float(thresholds["minimum_train_v_mse_relative_improvement"])
        for report in seed_reports
    )
    heldout_v = all(
        report["relative_improvements"]["heldout_coordinate_v_mse"]
        >= float(thresholds["minimum_heldout_v_mse_relative_improvement_per_seed"])
        for report in seed_reports
    )
    heldout_pair = all(
        report["relative_improvements"]["heldout_pair_distance_rmse"]
        >= float(thresholds["minimum_heldout_pair_rmse_relative_improvement_per_seed"])
        for report in seed_reports
    )
    gradients = all(
        report["coefficient_head_gradient_observed"]
        and report["unet_trunk_gradient_observed"]
        and report["parameter_l2_change"] > 0
        for report in seed_reports
    )
    contracts = all(all(report["trained_contract_checks"].values()) for report in seed_reports)
    geometry = all(report["sampling_geometry_valid"] for report in seed_reports)
    if "maximum_joint_polymer_quality_error" in thresholds:
        sampling = all(report["joint_polymer_quality"]["passed"] for report in seed_reports)
    else:
        sampling = all(
            min(
                report["sampling_improvements"]["adjacent_vs_untrained"],
                report["sampling_improvements"]["adjacent_vs_gaussian"],
            )
            >= float(thresholds["minimum_sampling_adjacent_distribution_improvement"])
            and min(
                report["sampling_improvements"]["radius_vs_untrained"],
                report["sampling_improvements"]["radius_vs_gaussian"],
            )
            >= float(thresholds["minimum_sampling_radius_distribution_improvement"])
            for report in seed_reports
        )
    if train_improved and heldout_v and heldout_pair and gradients and contracts and geometry and sampling:
        return "synthetic_learning_verified"
    if train_improved and not (heldout_v and heldout_pair):
        return "memorization_without_heldout_learning"
    if train_improved and heldout_v and heldout_pair and gradients and contracts and geometry and not sampling:
        return "denoising_learned_but_sampling_not_learned"
    if not gradients:
        return "equivariant_lifting_underexpressive"
    return "inconclusive_requires_review"


def _verify_sources(config: dict[str, Any]) -> dict[str, str]:
    phase3a = config["phase3a"]
    paths = {
        "generator_config": Path(phase3a["generator_config_path"]),
        "architecture_contract": Path(phase3a["architecture_contract_path"]),
        "e004_checkpoint": Path(phase3a["e004_checkpoint_path"]),
    }
    expected = {
        "generator_config": phase3a["generator_config_sha256"],
        "architecture_contract": phase3a["architecture_contract_sha256"],
        "e004_checkpoint": phase3a["e004_checkpoint_sha256"],
    }
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    for name, value in expected.items():
        if observed[name] != value:
            raise ValueError(f"E007 Phase-3A source hash contradiction: {name}")
    contract = json.loads(paths["architecture_contract"].read_text())
    if contract["status"] != "phase3a_architecture_contract_verified_non_authorizing":
        raise ValueError("E007 Phase-3A architecture contract status contradiction")
    if contract["permissions"]["authorizes_training"] is not False:
        raise ValueError("E007 Phase-3A architecture contract unexpectedly authorizes training")
    return observed


def _verify_v1_evidence(config: dict[str, Any]) -> dict[str, Any] | None:
    record = config.get("phase3b_v1")
    if not record:
        return None
    report_path = Path(record["report_path"])
    protocol_path = Path(record["protocol_path"])
    if _sha256_file(report_path) != record["report_sha256"]:
        raise ValueError("E007 Phase-3B-v1 report hash contradiction")
    if _sha256_file(protocol_path) != record["protocol_sha256"]:
        raise ValueError("E007 Phase-3B-v1 protocol hash contradiction")
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if report.get("status") != "completed" or protocol.get("status") != "completed":
        raise ValueError("E007 Phase-3B-v1 evidence is not completed")
    if protocol.get("authorizes_training") is not False:
        raise ValueError("E007 Phase-3B-v1 evidence unexpectedly authorizes training")
    expected = record["expected_panel_hashes"]
    observed = report["polymer_panels"]
    for field, values in expected.items():
        if observed.get(field) != values:
            raise ValueError(f"E007 Phase-3B-v1 frozen panel hash contradiction: {field}")
    return {
        "report_path": str(report_path),
        "report_sha256": record["report_sha256"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": record["protocol_sha256"],
        "panel_hashes": expected,
        "classification": report["classification"],
        "successful_optimizer_updates": report["successful_optimizer_updates"],
    }


def _model_counts(config: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
    phase3a = yaml.safe_load(Path(config["phase3a"]["generator_config_path"]).read_text())
    smoke = EquivariantPairCoordinateUNet(**config["smoke_model"])
    production = EquivariantPairCoordinateUNet(**phase3a["model"])
    smoke_count = sum(parameter.numel() for parameter in smoke.parameters())
    production_count = sum(parameter.numel() for parameter in production.parameters())
    return smoke_count, production_count, phase3a["model"]


def plan_coordinate_smoke(config_path: str | Path) -> dict[str, Any]:
    """Build a read-only plan without forward, backward, optimizer, or output writes."""
    config = _load_config(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    if output.exists() or staging.exists():
        raise FileExistsError(f"E007 coordinate smoke output already exists: {output} or {staging}")
    if int(config["optimizer_updates_per_seed"]) > int(
        config["decision_thresholds"]["maximum_optimizer_updates_per_seed"]
    ):
        raise ValueError("E007 smoke update count exceeds its predeclared bound")
    sources = _verify_sources(config)
    v1_evidence = _verify_v1_evidence(config)
    smoke_count, production_count, _ = _model_counts(config)
    train_count = len(config["families"]) * len(config["lengths"]) * int(config["train_replicates_per_family_length"])
    heldout_count = (
        len(config["families"]) * len(config["lengths"]) * int(config["heldout_replicates_per_family_length"])
    )
    return {
        "status": "planned_non_authorizing",
        "version": config["version"],
        "output_dir": str(output),
        "lengths": list(map(int, config["lengths"])),
        "families": list(config["families"]),
        "seeds": list(map(int, config["seeds"])),
        "train_sample_count": train_count,
        "heldout_sample_count": heldout_count,
        "optimizer_updates_per_seed": int(config["optimizer_updates_per_seed"]),
        "maximum_optimizer_updates_per_seed": int(config["decision_thresholds"]["maximum_optimizer_updates_per_seed"]),
        "evaluation_updates": list(map(int, config["evaluation_updates"])),
        "decision_thresholds": dict(config["decision_thresholds"]),
        "smoke_model_parameter_count": smoke_count,
        "production_model_parameter_count": production_count,
        "source_hashes": sources,
        "phase3b_v1_evidence": v1_evidence,
        "smoke_config_sha256": _sha256_file(Path(config_path)),
        "model_input_fields": [
            "noisy_coordinates",
            "timesteps",
            "lengths",
            "residue_mask",
            "chain_continuity_mask",
        ],
        "clean_coordinate_feature_leakage": False,
        "sequence_token_inputs": False,
        "synthetic_dataset_written": False,
        "exact_model_implementation_reused": True,
        "production_forward_backward_check_planned": True,
        "oracle_sampler_contract_check_planned": True,
        "timestep_evaluation_bins": config.get("timestep_evaluation_bins", []),
        "optimizer_created": False,
        "forward_executed": False,
        "backward_executed": False,
        "authorizes_training": False,
        "authorizes_joint_training": False,
        "authorizes_sequence_conditioning": False,
        "real_data_scanned": False,
        "dataset_reprocessed": False,
        "pretrained_weights_loaded": False,
        "e004_weights_loaded": False,
    }


def _production_compatibility_check(
    model_config: dict[str, Any], diffusion: CoordinateVPDiffusion, *, scale: float, device: torch.device
) -> dict[str, Any]:
    model = EquivariantPairCoordinateUNet(**model_config).to(device).train()
    sample = PolymerSample("compatibility", "synthetic", "helix", 16, 1, generate_polymer("helix", 16, 1), "", "", "")
    batch, diffused = _paired_diffusion_batch(sample, diffusion, scale=scale, draw_seed=991, rotate=True, device=device)
    output = model(
        diffused.noisy_coordinates,
        diffused.timesteps,
        batch["lengths"],
        batch["residue_mask"],
        batch["continuity"],
    )
    loss = (output["v_prediction"] - diffused.coordinate_v_target).square().mean()
    loss.backward()
    finite = bool(torch.isfinite(loss)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
    )
    result = {
        "length": 16,
        "loss": float(loss.detach()),
        "finite": finite,
        "forward_completed": True,
        "backward_completed": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
    }
    del model, output, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_seed(
    seed: int,
    config: dict[str, Any],
    train_panel: list[PolymerSample],
    heldout_panel: list[PolymerSample],
    diffusion: CoordinateVPDiffusion,
    *,
    device: torch.device,
    metrics_handle: Any,
    staging: Path,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = EquivariantPairCoordinateUNet(**config["smoke_model"]).to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    scale = float(config["coordinate_scale_angstrom"])
    loss_weights = CoordinateLossWeights(
        adjacent=float(config["loss"]["adjacent_weight"]),
        stratified_pair=float(config["loss"]["stratified_pair_weight"]),
        soft_contact=float(config["loss"]["soft_contact_weight"]),
        steric_clash=float(config["loss"]["steric_clash_weight"]),
    )
    evaluations: dict[str, Any] = {}
    evaluation_updates = set(map(int, config["evaluation_updates"]))
    zero_message = _evaluation_metrics(
        model,
        heldout_panel,
        diffusion,
        scale=scale,
        corruption_seed=seed + 50000,
        device=device,
        zero_message=True,
    )
    initial_sampling = _sampling_panel(model, diffusion, config, seed_offset=seed * 10, device=device)
    target_distribution = _target_distribution(heldout_panel, set(map(int, config["sampling_lengths"])))
    successful = 0
    attempted = 0
    finite = True
    coefficient_gradient = False
    trunk_gradient = False
    latest_gradient_norm = 0.0
    latest_head_gradient_norm = 0.0
    latest_trunk_gradient_norm = 0.0

    def evaluate(update: int) -> None:
        evaluations[str(update)] = {
            "train": _evaluation_metrics(
                model, train_panel, diffusion, scale=scale, corruption_seed=seed + 40000, device=device
            ),
            "heldout": _evaluation_metrics(
                model, heldout_panel, diffusion, scale=scale, corruption_seed=seed + 50000, device=device
            ),
        }
        if config.get("timestep_evaluation_bins"):
            evaluations[str(update)]["heldout_timestep_bins"] = _timestep_stratified_metrics(
                model,
                heldout_panel,
                diffusion,
                config,
                scale=scale,
                corruption_seed=seed + 60000,
                device=device,
            )

    evaluate(0)
    model.train()
    for update in range(1, int(config["optimizer_updates_per_seed"]) + 1):
        attempted += 1
        sample = train_panel[(update - 1) % len(train_panel)]
        batch, diffused = _paired_diffusion_batch(
            sample,
            diffusion,
            scale=scale,
            draw_seed=seed * 1_000_000 + update,
            rotate=True,
            device=device,
        )
        output = model(
            diffused.noisy_coordinates,
            diffused.timesteps,
            batch["lengths"],
            batch["residue_mask"],
            batch["continuity"],
        )
        reconstructed = diffusion.reconstruct_x0(
            diffused.noisy_coordinates,
            diffused.timesteps,
            output["v_prediction"],
            batch["residue_mask"],
        )
        losses = coordinate_diffusion_losses(
            v_prediction=output["v_prediction"],
            v_target=diffused.coordinate_v_target,
            clean_coordinates=batch["coordinates"],
            predicted_clean_coordinates=reconstructed,
            residue_mask=batch["residue_mask"],
            chain_continuity_mask=batch["continuity"],
            weights=loss_weights,
            coordinate_scale_angstrom=scale,
            adjacent_huber_beta=float(config["loss"]["adjacent_huber_beta_angstrom"]),
            clash_distance=float(config["loss"]["steric_clash_distance_angstrom"]),
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        try:
            named_gradients = require_finite_training_state(losses["total"], model, seed=seed, update=update)
        except FloatingPointError as error:
            finite = False
            error.attempted_updates = attempted  # type: ignore[attr-defined]
            error.successful_updates = successful  # type: ignore[attr-defined]
            raise
        latest_gradient_norm = math.sqrt(
            sum(float(gradient.double().square().sum()) for gradient in named_gradients.values())
        )
        head = [gradient for name, gradient in named_gradients.items() if "coefficient_head" in name]
        trunk = [
            gradient
            for name, gradient in named_gradients.items()
            if "pair_trunk" in name and "coefficient_head" not in name
        ]
        latest_head_gradient_norm = math.sqrt(sum(float(value.double().square().sum()) for value in head))
        latest_trunk_gradient_norm = math.sqrt(sum(float(value.double().square().sum()) for value in trunk))
        coefficient_gradient |= latest_head_gradient_norm > 0
        trunk_gradient |= latest_trunk_gradient_norm > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        successful += 1
        if update % int(config["metrics_frequency"]) == 0 or update in evaluation_updates:
            record = {
                "seed": seed,
                "attempted_update": attempted,
                "successful_update": successful,
                "sample_id": sample.sample_id,
                "timestep": int(diffused.timesteps.item()),
                "losses": {name: float(value.detach()) for name, value in losses.items()},
                "gradient_norm": latest_gradient_norm,
                "coefficient_head_gradient_norm": latest_head_gradient_norm,
                "unet_trunk_gradient_norm": latest_trunk_gradient_norm,
            }
            metrics_handle.write(json.dumps(record, sort_keys=True) + "\n")
            metrics_handle.flush()
        if update in evaluation_updates:
            evaluate(update)
            model.train()
    final_sampling = _sampling_panel(model, diffusion, config, seed_offset=seed * 10, device=device)
    initial_train = evaluations["0"]["train"]["means"]
    final_train = evaluations[str(successful)]["train"]["means"]
    initial_heldout = evaluations["0"]["heldout"]["means"]
    final_heldout = evaluations[str(successful)]["heldout"]["means"]

    def relative(before: float, after: float) -> float:
        return (before - after) / max(abs(before), 1e-12)

    initial_sampling_error = _sampling_distribution_error(initial_sampling, target_distribution)
    final_sampling_error = _sampling_distribution_error(final_sampling, target_distribution)
    report = {
        "seed": seed,
        "attempted_updates": attempted,
        "successful_updates": successful,
        "finite_losses_and_gradients": finite,
        "coefficient_head_gradient_observed": coefficient_gradient,
        "unet_trunk_gradient_observed": trunk_gradient,
        "final_gradient_norm": latest_gradient_norm,
        "final_coefficient_head_gradient_norm": latest_head_gradient_norm,
        "final_unet_trunk_gradient_norm": latest_trunk_gradient_norm,
        "parameter_l2_change": _parameter_change(model, initial_state),
        "evaluations": evaluations,
        "zero_message_heldout_baseline": zero_message,
        "relative_improvements": {
            "train_coordinate_v_mse": relative(initial_train["coordinate_v_mse"], final_train["coordinate_v_mse"]),
            "heldout_coordinate_v_mse": relative(
                initial_heldout["coordinate_v_mse"], final_heldout["coordinate_v_mse"]
            ),
            "heldout_pair_distance_rmse": relative(
                initial_heldout["pair_distance_rmse_angstrom"],
                final_heldout["pair_distance_rmse_angstrom"],
            ),
        },
        "sampling": {
            "initial": initial_sampling,
            "final": final_sampling,
            "target_distribution": target_distribution,
            "initial_distribution_error": initial_sampling_error,
            "final_distribution_error": final_sampling_error,
        },
        "sampling_improvements": {
            "adjacent_vs_untrained": initial_sampling_error["adjacent_absolute_error"]
            - final_sampling_error["adjacent_absolute_error"],
            "radius_vs_untrained": initial_sampling_error["radius_absolute_error"]
            - final_sampling_error["radius_absolute_error"],
        },
        "sampling_geometry_valid": _sampling_geometry_valid(
            initial_sampling + final_sampling,
            float(config["decision_thresholds"]["euclidean_scaled_tolerance"]),
        ),
        "trained_contract_checks": _trained_contract_checks(
            model, tolerance=float(config["decision_thresholds"]["equivariance_atol"]), device=device
        ),
    }
    if "maximum_joint_polymer_quality_error" in config["decision_thresholds"]:
        report["joint_polymer_quality"] = _joint_polymer_quality(
            final_sampling,
            target_distribution,
            config["decision_thresholds"],
        )
    checkpoint = staging / f"synthetic_seed_{seed}.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "version": config["version"],
            "seed": seed,
            "model": model.state_dict(),
            "model_config": config["smoke_model"],
            "diffusion_steps": int(config["diffusion_steps"]),
            "synthetic_only": True,
            "non_production": True,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
            "successful_optimizer_updates": successful,
        },
    )
    report["checkpoint"] = {"path": checkpoint.name, "sha256": _sha256_file(checkpoint)}
    return report


def run_coordinate_smoke(config_path: str | Path) -> dict[str, Any]:
    """Execute and atomically publish the bounded synthetic smoke."""
    config = _load_config(config_path)
    plan = plan_coordinate_smoke(config_path)
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = datetime.now(UTC).isoformat()
    _atomic_json(staging / "heartbeat.json", {"status": "running", "stage": "synthetic_panel", "started_utc": started})
    source_before = {**_verify_sources(config), "smoke_config": _sha256_file(Path(config_path))}
    try:
        train_panel, heldout_panel, panel_metadata = build_polymer_panels(config)
        v1_evidence = _verify_v1_evidence(config)
        if v1_evidence is not None:
            for field, values in v1_evidence["panel_hashes"].items():
                if panel_metadata[field] != values:
                    raise ValueError(f"E007 Phase-3B-v2 panel identity changed: {field}")
        device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        diffusion = CoordinateVPDiffusion(int(config["diffusion_steps"]))
        oracle_contract = verify_oracle_sampler_contract(
            diffusion,
            device=device,
            tolerance=float(config["decision_thresholds"].get("oracle_sampler_tolerance", 2e-5)),
        )
        _, _, production_config = _model_counts(config)
        compatibility = _production_compatibility_check(
            production_config,
            diffusion,
            scale=float(config["coordinate_scale_angstrom"]),
            device=device,
        )
        _atomic_json(
            staging / "heartbeat.json",
            {"status": "running", "stage": "training_seeds", "started_utc": started},
        )
        seed_reports = []
        with (staging / "metrics.jsonl").open("w") as metrics_handle:
            for seed in map(int, config["seeds"]):
                try:
                    seed_reports.append(
                        _run_seed(
                            seed,
                            config,
                            train_panel,
                            heldout_panel,
                            diffusion,
                            device=device,
                            metrics_handle=metrics_handle,
                            staging=staging,
                        )
                    )
                except FloatingPointError as error:
                    seed_reports.append(
                        {
                            "seed": seed,
                            "attempted_updates": int(getattr(error, "attempted_updates", 0)),
                            "successful_updates": int(getattr(error, "successful_updates", 0)),
                            "finite_losses_and_gradients": False,
                            "failure": {"type": type(error).__name__, "message": str(error)},
                        }
                    )
        gaussian = _gaussian_sampling_baseline(config)
        for seed_report in seed_reports:
            if not seed_report["finite_losses_and_gradients"]:
                continue
            target_distribution = seed_report["sampling"]["target_distribution"]
            gaussian_error = _sampling_distribution_error(gaussian, target_distribution)
            final_error = seed_report["sampling"]["final_distribution_error"]
            seed_report["sampling"]["gaussian_distribution_error"] = gaussian_error
            seed_report["sampling_improvements"].update(
                {
                    "adjacent_vs_gaussian": gaussian_error["adjacent_absolute_error"]
                    - final_error["adjacent_absolute_error"],
                    "radius_vs_gaussian": gaussian_error["radius_absolute_error"]
                    - final_error["radius_absolute_error"],
                }
            )
        classification = classify_smoke(
            seed_reports,
            config["decision_thresholds"],
            sampler_contract_passed=bool(oracle_contract["passed"]),
        )
        source_after = {**_verify_sources(config), "smoke_config": _sha256_file(Path(config_path))}
        protected = source_before == source_after
        peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        peak_cuda_allocated = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        peak_cuda_reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
        report = {
            **plan,
            "status": "completed",
            "classification": classification,
            "oracle_sampler_contract": oracle_contract,
            "device": str(device),
            "polymer_panels": panel_metadata,
            "production_forward_backward_compatibility": compatibility,
            "seed_reports": seed_reports,
            "gaussian_coordinate_baseline": gaussian,
            "successful_optimizer_updates": sum(report["successful_updates"] for report in seed_reports),
            "attempted_optimizer_updates": sum(report["attempted_updates"] for report in seed_reports),
            "optimizer_created": True,
            "forward_executed": True,
            "backward_executed": True,
            "execution_field_scope": "synthetic_smoke_training_only",
            "peak_rss_mib": peak_rss_mib,
            "peak_cuda_allocated_mib": peak_cuda_allocated,
            "peak_cuda_reserved_mib": peak_cuda_reserved,
            "source_hashes_before": source_before,
            "source_hashes_after": source_after,
            "protected_inputs_unchanged": protected,
            "target_leakage_absent": model_has_no_sequence_inputs(),
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        if not protected:
            raise ValueError("E007 protected Phase-3A sources changed during synthetic smoke")
        _atomic_json(staging / "report.json", report)
        protocol = {
            "version": config["version"],
            "status": "completed",
            "classification": classification,
            "report_sha256": _sha256_file(staging / "report.json"),
            "metrics_sha256": _sha256_file(staging / "metrics.jsonl"),
            "synthetic_only": True,
            "non_production": True,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_sequence_conditioning": False,
            "real_data_scanned": False,
            "dataset_reprocessed": False,
            "pretrained_weights_loaded": False,
            "e004_weights_loaded": False,
            "protected_inputs_unchanged": True,
            "optimizer_created": True,
            "forward_executed": True,
            "backward_executed": True,
        }
        _atomic_json(staging / "protocol.json", protocol)
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "completed",
                "classification": classification,
                "completed_utc": report["completed_utc"],
                "report_sha256": protocol["report_sha256"],
            },
        )
        staging.replace(output)
        return report
    except BaseException as error:
        _atomic_json(
            staging / "heartbeat.json",
            {
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "failed_utc": datetime.now(UTC).isoformat(),
                "authorizes_training": False,
            },
        )
        raise


def model_has_no_sequence_inputs() -> bool:
    names = set(inspect.signature(EquivariantPairCoordinateUNet.forward).parameters)
    return names.isdisjoint({"sequence", "token_ids", "clean_coordinates"})


def remove_failed_test_staging(path: Path) -> None:
    """Test-only cleanup helper; production execution never removes staging."""
    shutil.rmtree(path)
