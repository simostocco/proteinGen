"""Bounded, non-authorizing E006 Phase-2 smoke workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.rich_geometry import (
    RICH_FEATURE_VERSION,
    RICH_PAIR_FEATURE_DIM,
    RichGeometryDataset,
    authorize_rich_geometry_dataset,
    collate_rich_geometry,
    deterministic_length_bucket_sample,
    invariant_rich_features,
)
from protein_distance_diffusion.data.sequence_geometry import SequenceGeometryVocabulary
from protein_distance_diffusion.models.rich_codesign import E006LossWeights, E006RichGeometryCoDesign, e006_losses
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.codesign import masked_sequence_inputs

SMOKE_PROTOCOL_VERSION = "e006_phase2_smoke_v1"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _memory(device: torch.device | None = None) -> dict[str, float | None]:
    current = 0.0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) / 1024
                break
    except OSError:
        pass
    result: dict[str, float | None] = {
        "current_rss_mib": current,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024,
        "cuda_allocated_mib": None,
        "cuda_reserved_mib": None,
        "peak_cuda_allocated_mib": None,
        "peak_cuda_reserved_mib": None,
    }
    if device is not None and device.type == "cuda":
        result.update(
            cuda_allocated_mib=torch.cuda.memory_allocated(device) / 1024**2,
            cuda_reserved_mib=torch.cuda.memory_reserved(device) / 1024**2,
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
            peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        )
    return result


def _enforce_memory(config: dict[str, Any], device: torch.device | None = None) -> None:
    telemetry = _memory(device)
    rss_limit = float(config["smoke"]["maximum_rss_mib"])
    if float(telemetry["current_rss_mib"] or 0) > rss_limit:
        raise MemoryError(
            f"E006 smoke RSS limit exceeded: current={telemetry['current_rss_mib']:.1f} MiB, "
            f"peak={telemetry['peak_rss_mib']:.1f} MiB, limit={rss_limit:.1f} MiB"
        )
    if device is not None and device.type == "cuda":
        cuda_limit = float(config["smoke"]["maximum_cuda_allocated_mib"])
        if float(telemetry["cuda_allocated_mib"] or 0) > cuda_limit:
            raise MemoryError(
                f"E006 smoke CUDA limit exceeded: allocated={telemetry['cuda_allocated_mib']:.1f} MiB, "
                f"peak={telemetry['peak_cuda_allocated_mib']:.1f} MiB, limit={cuda_limit:.1f} MiB"
            )


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def _authorization(config: dict[str, Any]):
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["directory"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
    )


def _model(config: dict[str, Any]) -> E006RichGeometryCoDesign:
    values = dict(config["model"])
    geometry = values.pop("geometry_model")
    return E006RichGeometryCoDesign(geometry_model=geometry, **values)


def _parameter_groups(model: E006RichGeometryCoDesign) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "sequence": [
            *model.token_embedding.parameters(),
            *model.position_embedding.parameters(),
            *model.sequence_layers.parameters(),
            *model.sequence_norm.parameters(),
            *model.sequence_output.parameters(),
        ],
        "geometry": [*model.geometry_model.parameters(), *model.sequence_to_geometry.parameters()],
        "fusion": [
            *model.rich_encoder.parameters(),
            *model.fusions.parameters(),
            *model.sequence_to_geometry_gate.parameters(),
        ],
    }


def _synthetic_row(sample_id: str, length: int, split: str = "train") -> dict[str, Any]:
    vocabulary = SequenceGeometryVocabulary()
    sequence = "".join(vocabulary.tokens[2 + index % 20] for index in range(length))
    angle = np.arange(length, dtype=np.float32) * 1.1
    ca = np.stack((3.8 * np.arange(length), 0.35 * np.sin(angle), 0.35 * np.cos(angle)), axis=-1).astype(np.float32)
    n = ca + np.asarray([-0.55, 1.2, 0.15], dtype=np.float32)
    c = ca + np.asarray([1.45, 0.1, -0.1], dtype=np.float32)
    o = c + np.asarray([0.55, -0.9, 0.1], dtype=np.float32)
    cb = ca + np.asarray([-0.4, -0.7, 1.2], dtype=np.float32)
    torsion = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (length, 1))
    torsion_mask = np.ones(length, dtype=bool)
    torsion_mask[0] = False
    return {
        "schema_version": "e006_rich_geometry_sidecar_v2",
        "torsion_convention_version": "e006_backbone_torsion_sincos_v1",
        "sample_id": sample_id,
        "split": split,
        "sequence": sequence,
        "token_ids": vocabulary.encode(sequence),
        "residue_ids": [str(index + 1) for index in range(length)],
        "insertion_codes": [""] * length,
        "n_coordinates": n.tolist(),
        "ca_coordinates": ca.tolist(),
        "c_coordinates": c.tolist(),
        "o_coordinates": o.tolist(),
        "cb_coordinates": cb.tolist(),
        "n_mask": [True] * length,
        "ca_mask": [True] * length,
        "c_mask": [True] * length,
        "o_mask": [True] * length,
        "cb_mask": [True] * length,
        "cb_source": [1 if index % 2 else 0 for index in range(length)],
        "local_frame_valid": [True] * length,
        "phi_sin_cos": torsion.tolist(),
        "psi_sin_cos": torsion.tolist(),
        "omega_sin_cos": torsion.tolist(),
        "phi_mask": torsion_mask.tolist(),
        "psi_mask": np.roll(torsion_mask, -1).tolist(),
        "omega_mask": torsion_mask.tolist(),
        "chain_continuity_mask": [True] * max(length - 1, 0),
        "chain_break_mask": [False] * max(length - 1, 0),
        "experimental_method": "synthetic",
    }


def _load_rows(config: dict[str, Any], authorization: Any | None, *, split: str, count: int) -> list[dict[str, Any]]:
    smoke = config["smoke"]
    if smoke.get("synthetic", False):
        lengths = list(smoke.get("synthetic_lengths", [8, 12]))
        return [
            _synthetic_row(f"synthetic-{split}-{index}", int(lengths[index % len(lengths)]), split)
            for index in range(count)
        ]
    if authorization is None:
        raise ValueError("E006 real smoke requires an authorized rich-geometry dataset")
    dataset = RichGeometryDataset(authorization, split=split)
    indices = deterministic_length_bucket_sample(
        dataset,
        count=count,
        seed=int(config["seed"]),
        maximum_length=int(smoke["maximum_length"]),
    )
    return [dataset[index] for index in indices]


def plan_e006_smoke(config: dict[str, Any]) -> dict[str, Any]:
    authorization = None if config["smoke"].get("synthetic", False) else _authorization(config)
    model = _model(config)
    maximum_length = int(config["smoke"]["maximum_length"])
    if maximum_length**2 > int(config["smoke"]["maximum_pair_elements"]):
        raise ValueError("E006 maximum length exceeds the configured pair-feature element budget")
    pair_bytes = maximum_length**2 * RICH_PAIR_FEATURE_DIM * 4
    batch_size = int(config["smoke"]["batch_size"])
    return {
        "status": "planned",
        "protocol_version": SMOKE_PROTOCOL_VERSION,
        "architecture_version": model.architecture_version,
        "feature_version": RICH_FEATURE_VERSION,
        "parameter_counts": model.parameter_counts(),
        "expected_pair_feature_memory_mib": pair_bytes / 1024**2,
        "expected_batch_pair_feature_memory_mib": pair_bytes * batch_size / 1024**2,
        "dataset_protocol_sha256": authorization.protocol_sha256 if authorization else None,
        "dataset_metadata_sha256": {
            name: config["dataset"][name]
            for name in (
                "schema_sha256",
                "vocabulary_sha256",
                "normalization_sha256",
                "shard_inventory_sha256",
            )
        },
        "authorizes_definitive_training": False,
        "authorizes_training": False,
        "training_performed": False,
    }


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _rigid_invariance_check(row: dict[str, Any], maximum_pair_elements: int) -> bool:
    original = invariant_rich_features(row, maximum_pair_elements=maximum_pair_elements)
    transformed = deepcopy(row)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    translation = np.asarray([7.0, -11.0, 3.5], dtype=np.float32)
    for atom in ("n", "ca", "c", "o", "cb"):
        coordinates = np.asarray(row[f"{atom}_coordinates"], dtype=np.float32)
        transformed[f"{atom}_coordinates"] = (coordinates @ rotation.T + translation).tolist()
    changed = invariant_rich_features(transformed, maximum_pair_elements=maximum_pair_elements)
    return bool(
        torch.allclose(original["residue_features"], changed["residue_features"], atol=2e-5, rtol=0)
        and torch.allclose(original["pair_features"], changed["pair_features"], atol=2e-5, rtol=0)
        and torch.equal(original["pair_feature_mask"], changed["pair_feature_mask"])
    )


def _forward(
    model: E006RichGeometryCoDesign,
    batch: dict[str, Any],
    *,
    config: dict[str, Any],
    step: int,
    mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    targets = batch["sequence_token_ids"]
    mask_fractions = list(config["objective"].get("mask_fractions", [config["objective"]["mask_fraction"]]))
    mask_fraction = float(mask_fractions[step % len(mask_fractions)])
    inputs, masked = masked_sequence_inputs(
        targets,
        batch["residue_mask"],
        mask_token_id=1,
        probability=mask_fraction,
        seed=int(config["seed"]),
        step=step,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + step * 17)
    timestep = int(torch.randint(0, int(config["diffusion"]["steps"]), (1,), generator=generator).item())
    noise_levels = list(
        config["objective"].get("geometry_noise_levels", [0.0, config["objective"]["geometry_noise_std"]])
    )
    noise_level = float(noise_levels[step % len(noise_levels)])
    noise = torch.randn(batch["distance_matrices"].shape, generator=generator).to(batch["distance_matrices"].device)
    noisy = (batch["distance_matrices"] + noise * noise_level) * batch["pair_mask"].to(torch.float32)
    keep = torch.ones(len(batch["lengths"]), dtype=torch.bool, device=targets.device)
    if mode != "sequence_only" and float(config["objective"]["conditioning_dropout_probability"]) > 0:
        draws = torch.rand(len(keep), generator=generator)
        keep = (draws >= float(config["objective"]["conditioning_dropout_probability"])).to(targets.device)
    rich_residue = batch["rich_residue_features"]
    rich_pair = batch["rich_pair_features"]
    if noise_level:
        rich_residue = (
            rich_residue + torch.randn(rich_residue.shape, generator=generator).to(rich_residue.device) * noise_level
        ) * batch["residue_mask"][..., None]
        rich_pair = (
            rich_pair + torch.randn(rich_pair.shape, generator=generator).to(rich_pair.device) * noise_level
        ) * batch["pair_feature_mask"][:, 0, ..., None]
    outputs = model(
        sequence_token_ids=inputs,
        residue_mask=batch["residue_mask"],
        rich_residue_features=rich_residue,
        rich_pair_features=rich_pair,
        pair_feature_mask=batch["pair_feature_mask"],
        noisy_geometry=noisy,
        timesteps=torch.full((len(keep),), timestep, dtype=torch.long, device=targets.device),
        lengths=batch["lengths"],
        sequence_separation=make_sequence_separation(batch["lengths"], noisy.shape[-1]),
        pair_mask=batch["pair_mask"],
        geometry_conditioning_mask=keep,
        mode=mode,
    )
    outputs["masked_token_mask"] = masked
    configured = config["objective"]
    weights = E006LossWeights(
        float(configured["sequence_weight"]),
        float(configured["geometry_weight"]),
        float(configured["consistency_weight"]),
    ).at_step(step + 1, int(configured["auxiliary_warmup_steps"]))
    losses = e006_losses(
        outputs,
        sequence_targets=targets,
        masked_token_mask=masked,
        geometry_target=batch["distance_matrices"],
        weights=weights,
    )
    metadata = {"mask_fraction": mask_fraction, "noise_level": noise_level, "timestep": timestep}
    return outputs, losses, metadata


def stratified_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dimensions = (
        "mask_fraction",
        "geometry_corruption_level",
        "diffusion_timestep",
        "protein_length",
        "experimental_method",
        "pseudo_cb_coverage",
        "frame_coverage",
        "torsion_coverage",
        "conditioning_mode",
    )
    output = []
    metric_names = (
        "sequence_cross_entropy",
        "perplexity",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
        "geometry_loss",
        "consistency_loss",
        "fusion_gate_mean",
        "fusion_gate_std",
        "fusion_gate_saturated_fraction",
        "sequence_to_geometry_gate_mean",
    )
    for dimension in dimensions:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record[dimension])].append(record)
        for value, rows in sorted(groups.items()):
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "count": len(rows),
                    **{
                        name: float(np.mean([row[name] for row in rows]))
                        for name in metric_names
                        if all(name in row for row in rows)
                    },
                }
            )
    return output


def _metrics(
    outputs: dict[str, torch.Tensor],
    losses: dict[str, torch.Tensor],
    batch: dict[str, Any],
    metadata: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    residue_mask = batch["residue_mask"]
    sequence_mask = outputs["masked_token_mask"] & residue_mask
    logits = outputs["sequence_logits"][sequence_mask]
    targets = batch["sequence_token_ids"][sequence_mask]
    top = logits.topk(5, dim=-1).indices
    frame_coverage = float(batch["frame_mask"].sum() / residue_mask.sum())
    torsion_count = sum(value.sum() for value in batch["torsion_masks"].values())
    fusion_mask = residue_mask[:, None, :, None].expand_as(outputs["fusion_gates"])
    fusion_gates = outputs["fusion_gates"][fusion_mask]
    sequence_gates = outputs["sequence_to_geometry_gate"][residue_mask]
    return {
        "mask_fraction": round(metadata["mask_fraction"], 3),
        "geometry_corruption_level": metadata["noise_level"],
        "diffusion_timestep": metadata["timestep"],
        "protein_length": int(batch["lengths"].max()),
        "experimental_method": batch["experimental_methods"][0],
        "pseudo_cb_coverage": round(float(batch["pseudo_cb_mask"].sum() / residue_mask.sum()), 2),
        "frame_coverage": round(frame_coverage, 2),
        "torsion_coverage": round(float(torsion_count / (3 * residue_mask.sum())), 2),
        "conditioning_mode": mode,
        "sequence_cross_entropy": float(losses["sequence"].detach()),
        "perplexity": float(torch.exp(losses["sequence"].detach())),
        "top1_accuracy": float((top[:, :1] == targets[:, None]).any(dim=-1).float().mean()),
        "top3_accuracy": float((top[:, :3] == targets[:, None]).any(dim=-1).float().mean()),
        "top5_accuracy": float((top == targets[:, None]).any(dim=-1).float().mean()),
        "geometry_loss": float(losses["geometry"].detach()),
        "consistency_loss": float(losses["consistency"].detach()),
        "fusion_gate_mean": float(fusion_gates.mean()),
        "fusion_gate_std": float(fusion_gates.std(unbiased=False)),
        "fusion_gate_saturated_fraction": float(((fusion_gates < 0.01) | (fusion_gates > 0.99)).float().mean()),
        "sequence_to_geometry_gate_mean": float(sequence_gates.mean()),
    }


def run_e006_smoke(config_path: str | Path, *, mode: str) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if mode not in {"loader-smoke", "train-smoke"}:
        raise ValueError("E006 executable smoke mode must be loader-smoke or train-smoke")
    output = Path(config["smoke"]["output_dir"]) / mode
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E006 smoke output: {output}")
    report_path = output / "protocol.json"
    heartbeat_path = output / "heartbeat.json"
    output.mkdir(parents=True)
    started = _utc_now()
    _atomic_json(heartbeat_path, {"status": "running", "stage": "authorization", "started_utc": started})
    protected: list[Path] = []
    before: dict[str, str] = {}
    try:
        protected = [
            Path(config["dataset"]["directory"]) / name
            for name in (
                "protocol.json",
                "schema.json",
                "vocabulary.json",
                "normalization.json",
                "shard_hashes.sha256",
            )
        ]
        if not config["smoke"].get("synthetic", False):
            before = {str(path): _sha256(path) for path in protected}
        authorization = None if config["smoke"].get("synthetic", False) else _authorization(config)
        if mode == "train-smoke" and not 1 <= int(config["smoke"]["steps"]) <= 16:
            raise ValueError("E006 train-smoke steps must be in [1, 16]")
        count = int(config["smoke"]["sample_count"])
        batch_size = int(config["smoke"]["batch_size"])
        if not 1 <= count <= batch_size:
            raise ValueError("E006 smoke sample_count must be positive and no greater than batch_size")
        if mode == "loader-smoke" and count < 2:
            raise ValueError("E006 loader-smoke needs at least two samples to exercise both splits")
        if mode == "loader-smoke":
            train_count = (count + 1) // 2
            validation_count = count // 2
            rows = _load_rows(config, authorization, split="train", count=train_count)
            rows.extend(_load_rows(config, authorization, split="validation", count=validation_count))
        else:
            rows = _load_rows(config, authorization, split="train", count=count)
        batch = collate_rich_geometry(rows, maximum_pair_elements=int(config["smoke"]["maximum_pair_elements"]))
        _enforce_memory(config)
        report: dict[str, Any] = {
            "status": "running",
            "protocol_version": SMOKE_PROTOCOL_VERSION,
            "mode": mode,
            "configuration_sha256": _sha256(config_path),
            "configuration_hash": _config_hash(config),
            "dataset_protocol_sha256": config["dataset"]["protocol_sha256"],
            "dataset_metadata_sha256": {
                name: config["dataset"][name]
                for name in (
                    "schema_sha256",
                    "vocabulary_sha256",
                    "normalization_sha256",
                    "shard_inventory_sha256",
                )
            },
            "sample_ids": batch["sample_ids"],
            "tensor_shapes": {
                key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)
            },
            "authorizes_definitive_training": False,
            "authorizes_training": False,
            "started_utc": started,
        }
        if mode == "loader-smoke":
            report["loader_checks"] = {
                "split_ownership": all(value in {"train", "validation"} for value in batch["splits"]),
                "both_splits_present": set(batch["splits"]) == {"train", "validation"},
                "padding_masked": bool(
                    (batch["rich_residue_features"] * ~batch["residue_mask"][..., None]).eq(0).all()
                ),
                "pair_padding_masked": bool(
                    (batch["rich_pair_features"] * ~batch["pair_mask"][:, 0, ..., None]).eq(0).all()
                ),
                "finite_features": bool(
                    torch.isfinite(batch["rich_residue_features"]).all()
                    and torch.isfinite(batch["rich_pair_features"]).all()
                ),
                "rigid_transform_invariant": _rigid_invariance_check(
                    rows[0],
                    int(config["smoke"]["maximum_pair_elements"]),
                ),
            }
        else:
            device = torch.device(config["smoke"].get("device", "cpu"))
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("E006 train-smoke requested CUDA but CUDA is unavailable")
            batch = _move(batch, device)
            validation_rows = _load_rows(
                config,
                authorization,
                split="validation",
                count=int(config["smoke"].get("validation_sample_count", count)),
            )
            if not 1 <= len(validation_rows) <= batch_size:
                raise ValueError("E006 validation_sample_count must be positive and no greater than batch_size")
            validation_batch = _move(
                collate_rich_geometry(
                    validation_rows,
                    maximum_pair_elements=int(config["smoke"]["maximum_pair_elements"]),
                ),
                device,
            )
            torch.manual_seed(int(config["seed"]))
            model = _model(config).to(device)
            _enforce_memory(config, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["smoke"]["learning_rate"]))
            records = []
            initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            steps = int(config["smoke"]["steps"])
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                outputs, losses, metadata = _forward(
                    model,
                    batch,
                    config=config,
                    step=step,
                    mode="learned_geometry_gating",
                )
                if not all(torch.isfinite(value) for value in losses.values()):
                    raise FloatingPointError("E006 smoke produced a non-finite loss")
                losses["total"].backward()
                gradient_norms = {}
                for name, parameters in _parameter_groups(model).items():
                    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                    if not gradients:
                        raise RuntimeError(f"E006 active parameter group has no gradients: {name}")
                    gradient_norms[name] = math.sqrt(
                        sum(float(gradient.float().square().sum()) for gradient in gradients)
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["smoke"]["gradient_clip_norm"]))
                optimizer.step()
                _enforce_memory(config, device)
                record = {
                    "step": step + 1,
                    **{name: float(value.detach()) for name, value in losses.items()},
                    "gradient_norms": gradient_norms,
                    "fusion_gate_mean": float(outputs["fusion_gates"].mean().detach()),
                }
                records.append(record)
            parameter_changes = {
                name: float((model.state_dict()[name] - value).float().norm()) for name, value in initial_state.items()
            }
            if not any(value > 0 for value in parameter_changes.values()):
                raise RuntimeError("E006 train-smoke parameters did not change")
            checkpoint = output / "smoke_checkpoint.pt"
            payload = {
                "version": SMOKE_PROTOCOL_VERSION,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configuration_hash": _config_hash(config),
                "dataset_protocol_sha256": config["dataset"]["protocol_sha256"],
                "authorizes_definitive_training": False,
            }
            save_checkpoint(checkpoint, payload)
            reloaded = _model(config).to(device)
            saved = load_checkpoint(checkpoint, map_location=device)
            reloaded.load_state_dict(saved["model"])
            model.eval()
            reloaded.eval()
            with torch.no_grad():
                original, losses, metadata = _forward(
                    model, batch, config=config, step=steps, mode="learned_geometry_gating"
                )
                repeated, _, _ = _forward(reloaded, batch, config=config, step=steps, mode="learned_geometry_gating")
            if not torch.equal(original["sequence_logits"], repeated["sequence_logits"]) or not torch.equal(
                original["geometry_prediction"], repeated["geometry_prediction"]
            ):
                raise RuntimeError("E006 checkpoint reload is not deterministic")
            validation_records = []
            with torch.no_grad():
                for validation_mode in (
                    "sequence_only",
                    "learned_geometry_gating",
                    "forced_geometry_conditioning",
                ):
                    outputs, losses, metadata = _forward(
                        reloaded,
                        validation_batch,
                        config=config,
                        step=steps,
                        mode=validation_mode,
                    )
                    validation_records.append(_metrics(outputs, losses, validation_batch, metadata, validation_mode))
            overfit_steps = int(config["smoke"].get("overfit_steps", 4))
            if not 1 <= overfit_steps <= 8:
                raise ValueError("E006 train-smoke overfit_steps must be in [1, 8]")
            torch.manual_seed(int(config["seed"]))
            overfit_model = _model(config).to(device)
            overfit_optimizer = torch.optim.AdamW(
                overfit_model.parameters(),
                lr=float(config["smoke"]["learning_rate"]),
            )
            overfit_losses = []
            for _ in range(overfit_steps):
                overfit_optimizer.zero_grad(set_to_none=True)
                _, fixed_losses, _ = _forward(
                    overfit_model,
                    batch,
                    config=config,
                    step=0,
                    mode="learned_geometry_gating",
                )
                fixed_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    overfit_model.parameters(),
                    float(config["smoke"]["gradient_clip_norm"]),
                )
                overfit_optimizer.step()
                overfit_losses.append(float(fixed_losses["total"].detach()))
                _enforce_memory(config, device)
            report.update(
                parameter_counts=model.parameter_counts(),
                losses_by_step=records,
                parameter_change_norms=parameter_changes,
                checkpoint_path=str(checkpoint),
                checkpoint_sha256=_sha256(checkpoint),
                checkpoint_reload_deterministic=True,
                stratified_validation=stratified_metrics(validation_records),
                overfit_diagnostic={
                    "fixed_sample_ids": batch["sample_ids"],
                    "fixed_mask_noise_and_timestep": True,
                    "steps": overfit_steps,
                    "losses": overfit_losses,
                    "initial_total": overfit_losses[0],
                    "final_total": overfit_losses[-1],
                    "total_reduced": overfit_losses[-1] < overfit_losses[0],
                },
            )
            if overfit_steps >= 2 and not report["overfit_diagnostic"]["total_reduced"]:
                raise RuntimeError("E006 tiny fixed-subset overfit loss did not decrease")
        after = {str(path): _sha256(path) for path in protected} if before else {}
        if after != before:
            raise RuntimeError("E006 protected rich-geometry inputs changed during smoke workflow")
        shard_hashes_after = (
            {relative: _sha256(authorization.root / relative) for relative in authorization.observed_shard_hashes}
            if authorization is not None
            else {}
        )
        if authorization is not None and shard_hashes_after != authorization.observed_shard_hashes:
            raise RuntimeError("E006 rich-geometry shard changed during smoke workflow")
        report.update(
            status="completed",
            completed_utc=_utc_now(),
            protected_input_hashes_before=before,
            protected_input_hashes_after=after,
            protected_inputs_unchanged=True,
            observed_shard_hashes=shard_hashes_after,
            memory=_memory(torch.device(config["smoke"].get("device", "cpu")) if mode == "train-smoke" else None),
        )
        _atomic_json(report_path, report)
        _atomic_json(
            heartbeat_path,
            {
                "status": "completed",
                "stage": "finalization",
                "completed_utc": report["completed_utc"],
                "report_path": str(report_path),
                "report_sha256": _sha256(report_path),
            },
        )
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "protocol_version": SMOKE_PROTOCOL_VERSION,
            "mode": mode,
            "error_type": type(error).__name__,
            "error_message": str(error)[:1000],
            "authorizes_definitive_training": False,
            "authorizes_training": False,
            "completed_utc": _utc_now(),
            "memory": _memory(),
        }
        _atomic_json(report_path, failure)
        _atomic_json(
            heartbeat_path,
            {
                "status": "failed",
                "stage": "failure",
                "completed_utc": failure["completed_utc"],
                "error_type": failure["error_type"],
                "error_message": failure["error_message"],
                "report_path": str(report_path),
                "report_sha256": _sha256(report_path),
            },
        )
        raise
