"""Bounded, resumable E005 paired-arm pilot training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.checkpoint import checkpoint

from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryDataset,
    SequenceGeometryVocabulary,
    collate_sequence_geometry,
)
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.codesign import (
    E005_ARCHITECTURE_VERSION,
    CoDesignLossWeights,
    E005SequenceGeometryCoDesign,
    codesign_losses,
)
from protein_distance_diffusion.training.checkpointing import load_checkpoint, save_checkpoint
from protein_distance_diffusion.training.codesign import (
    _configuration_sha256,
    _gate_statistics,
    _parameter_change_norms,
    _parameter_groups,
    _parameter_snapshots,
    _peak_rss_mib,
    _rss_mib,
    _sha256_file,
    _synthetic_items,
    _tensor_sha256,
    conditioning_mask,
    masked_sequence_inputs,
)
from protein_distance_diffusion.training.trainer import _restore_rng_state, _rng_state

TRAINING_ARMS = ("sequence_only", "learned_geometry_gating")
VALIDATION_MODES = (*TRAINING_ARMS, "forced_geometry_conditioning")
DEFAULT_STAGES = ((128, 200), (256, 200), (500, 100))
DEFAULT_PAIR_BUDGET = ((64, 16), (128, 4), (256, 1), (384, 1), (500, 1))
MAX_RSS_MIB = 4096
MAX_DIAGNOSTIC_EXAMPLES = 100


@dataclass(frozen=True)
class CurriculumStage:
    maximum_length: int
    optimizer_updates: int


class PilotInterrupted(RuntimeError):
    """Raised after a clean resumable interruption checkpoint is written."""


def accumulation_for_length(length: int, budget: Iterable[tuple[int, int]] = DEFAULT_PAIR_BUDGET) -> int:
    """Return the configured microbatch count for one padded square side."""
    for maximum, microbatches in budget:
        padded_threshold = ((maximum + 7) // 8) * 8
        if length <= padded_threshold:
            return microbatches
    raise ValueError(f"No pair-budget accumulation rule covers padded length {length}")


def curriculum_stages(config: dict[str, Any]) -> tuple[CurriculumStage, ...]:
    values = config.get("pilot", {}).get("curriculum")
    if values is None:
        return tuple(CurriculumStage(*item) for item in DEFAULT_STAGES)
    stages = tuple(CurriculumStage(int(item["maximum_length"]), int(item["optimizer_updates"])) for item in values)
    if not stages or any(item.maximum_length < 1 or item.optimizer_updates < 1 for item in stages):
        raise ValueError("Pilot curriculum stages require positive lengths and optimizer-update counts")
    if tuple(item.maximum_length for item in stages) != tuple(sorted(item.maximum_length for item in stages)):
        raise ValueError("Pilot curriculum maximum lengths must be non-decreasing")
    maximum_updates = int(config.get("pilot", {}).get("maximum_optimizer_updates", 500))
    if sum(item.optimizer_updates for item in stages) > maximum_updates or maximum_updates > 500:
        raise ValueError("The bounded pilot permits at most 500 configured optimizer updates per arm")
    return stages


def pair_budget(config: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    values = config.get("pilot", {}).get("pair_budget_accumulation")
    budget = (
        tuple((int(item["maximum_padded_length"]), int(item["microbatches"])) for item in values)
        if values is not None
        else DEFAULT_PAIR_BUDGET
    )
    if not budget or any(maximum < 1 or count < 1 for maximum, count in budget):
        raise ValueError("Pair-budget accumulation values must be positive")
    if tuple(maximum for maximum, _ in budget) != tuple(sorted(maximum for maximum, _ in budget)):
        raise ValueError("Pair-budget thresholds must be sorted")
    return budget


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _memory(device: torch.device) -> dict[str, float | None]:
    return {
        "current_rss_mib": _rss_mib(),
        "peak_rss_mib": _peak_rss_mib(),
        "cuda_allocated_mib": torch.cuda.memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "cuda_reserved_mib": torch.cuda.memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "peak_cuda_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None),
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
    }


def _guard_memory(device: torch.device, rss_limit: int, cuda_limit: int) -> None:
    memory = _memory(device)
    if float(memory["current_rss_mib"] or 0) > rss_limit:
        raise MemoryError(
            f"E005 pilot RSS limit exceeded: limit={rss_limit} MiB, "
            f"current={memory['current_rss_mib']:.1f} MiB, peak={memory['peak_rss_mib']:.1f} MiB"
        )
    cuda_usage = max(float(memory["cuda_allocated_mib"] or 0), float(memory["cuda_reserved_mib"] or 0))
    if device.type == "cuda" and cuda_usage > cuda_limit:
        raise MemoryError(
            f"E005 pilot CUDA limit exceeded: limit={cuda_limit} MiB, "
            f"allocated={memory['cuda_allocated_mib']:.1f} MiB, reserved={memory['cuda_reserved_mib']:.1f} MiB"
        )


def _length_bin(length: int) -> str:
    for maximum in (64, 128, 256, 384, 500):
        if length <= maximum:
            return f"le_{maximum}"
    return "above_500"


def _metadata_column(schema: pa.Schema, candidates: tuple[str, ...], default: str) -> str | None:
    del default
    return next((name for name in candidates if name in schema.names), None)


def _scan_bounded_rows(path: Path, *, maximum_length: int, seed: int, capacity: int) -> list[dict[str, Any]]:
    """Keep a deterministic bounded reservoir per scientific stratum."""
    dataset = ds.dataset(str(path), format="parquet")
    method_column = _metadata_column(dataset.schema, ("experimental_method", "method"), "unknown")
    class_column = _metadata_column(dataset.schema, ("pairing_classification", "v3_pairing_classification"), "unknown")
    columns = [
        "sample_id",
        "schema_version",
        "sequence",
        "sequence_length",
        "matrix_length",
        "matrix_path",
        "practical_training_eligibility",
        "__filename",
    ]
    columns.extend(name for name in (method_column, class_column) if name is not None)
    reservoirs: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    per_stratum = max(2, capacity // 32)
    for batch in dataset.scanner(columns=columns, batch_size=4096, use_threads=False).to_batches():
        for row in batch.to_pylist():
            length = int(row["sequence_length"])
            if length > maximum_length:
                continue
            row["experimental_method"] = str(row.get(method_column) or "unknown")
            row["pairing_classification"] = str(row.get(class_column) or "unknown")
            row["length_bin"] = _length_bin(length)
            key = (row["length_bin"], row["experimental_method"], row["pairing_classification"])
            rank = hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()
            reservoir = reservoirs[key]
            reservoir.append((rank, row))
            reservoir.sort(key=lambda item: item[0])
            del reservoir[per_stratum:]
    ordered_groups = [sorted(values, key=lambda item: item[0]) for _, values in sorted(reservoirs.items())]
    rows = []
    for position in range(max(map(len, ordered_groups), default=0)):
        for group in ordered_groups:
            if position < len(group) and len(rows) < capacity:
                rows.append(group[position][1])
    if not rows:
        raise ValueError(f"No eligible rows with length <= {maximum_length} in {path}")
    return rows


def _synthetic_rows(config: dict[str, Any], *, validation: bool) -> list[dict[str, Any]]:
    lengths = [int(value) for value in config.get("pilot", {}).get("synthetic_lengths", [8, 12])]
    items = _synthetic_items(lengths, int(config["seed"]) + (1 if validation else 0))
    rows = []
    for index, item in enumerate(items):
        rows.append(
            {
                "sample_id": item["sample_id"] + ("_validation" if validation else "_train"),
                "sequence_length": item["length"],
                "matrix_length": item["length"],
                "experimental_method": "synthetic",
                "pairing_classification": "synthetic_verified",
                "length_bin": _length_bin(item["length"]),
                "synthetic_index": index,
            }
        )
    return rows


def _verify_dataset_partitions(
    directory: Path,
    protocol: dict[str, Any],
) -> dict[str, str]:
    """Verify protocol partitions by their canonical root-relative paths."""
    base = directory.resolve(strict=True)
    records = protocol.get("partitions")
    if not isinstance(records, list) or not records:
        raise ValueError("Pairing dataset protocol has no partition records")
    seen_paths: set[Path] = set()
    seen_indices: set[tuple[str, int]] = set()
    verified: dict[str, str] = {}
    expected_paths = {path.resolve(strict=True) for path in directory.rglob("*.parquet") if path.is_file()}
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Malformed partition record at index {record_index}")
        required = {"dataset", "partition_index", "path", "row_count", "sha256"}
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"Malformed partition record at index {record_index}: missing {', '.join(missing)}")
        dataset_name = record["dataset"]
        relative_value = record["path"]
        digest = record["sha256"]
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError(f"Malformed partition dataset at index {record_index}")
        if not isinstance(relative_value, str) or not relative_value or Path(relative_value).is_absolute():
            raise ValueError(f"Malformed partition path at index {record_index}")
        relative_path = Path(relative_value)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Malformed partition SHA-256 at index {record_index}")
        try:
            partition_index = int(record["partition_index"])
            row_count = int(record["row_count"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Malformed numeric partition metadata at index {record_index}") from error
        if partition_index < 0 or row_count < 0:
            raise ValueError(f"Negative partition metadata at index {record_index}")
        unresolved = base / relative_path
        candidate = unresolved.resolve(strict=False)
        if not candidate.is_relative_to(base):
            raise ValueError(f"Pairing dataset partition escapes dataset directory: {relative_value}")
        try:
            resolved = unresolved.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"Missing pairing dataset partition: {relative_value}") from error
        if not resolved.is_relative_to(base):
            raise ValueError(f"Pairing dataset partition escapes dataset directory: {relative_value}")
        if not relative_path.parts or relative_path.parts[0] != f"{dataset_name}.parquet":
            raise ValueError(f"Partition path contradicts dataset ownership: {relative_value}")
        if not resolved.is_file() or resolved.suffix != ".parquet":
            raise ValueError(f"Invalid pairing dataset partition file: {relative_value}")
        logical_key = (dataset_name, partition_index)
        if resolved in seen_paths:
            raise ValueError(f"Duplicate pairing dataset partition path: {relative_value}")
        if logical_key in seen_indices:
            raise ValueError(f"Duplicate pairing dataset partition index: {dataset_name}/{partition_index}")
        seen_paths.add(resolved)
        seen_indices.add(logical_key)
        actual_digest = _sha256_file(resolved)
        if actual_digest != digest:
            raise ValueError(f"Pairing dataset partition SHA-256 contradiction: {relative_value}")
        actual_rows = pq.ParquetFile(resolved).metadata.num_rows
        if actual_rows != row_count:
            raise ValueError(f"Pairing dataset partition row-count contradiction: {relative_value}")
        verified[str(resolved)] = actual_digest
    verified_paths = {Path(path) for path in verified}
    missing_records = sorted(expected_paths - verified_paths, key=str)
    extra_records = sorted(verified_paths - expected_paths, key=str)
    if missing_records:
        raise ValueError(f"Parquet partition is missing from protocol: {missing_records[0]}")
    if extra_records:
        raise ValueError(f"Protocol records an unexpected Parquet partition: {extra_records[0]}")
    return verified


def build_paired_plan(config: dict[str, Any], *, synthetic: bool = False) -> dict[str, Any]:
    """Create the arm-independent deterministic sample and stochastic plan."""
    stages = curriculum_stages(config)
    budget = pair_budget(config)
    seed = int(config["seed"])
    if synthetic:
        train_rows = _synthetic_rows(config, validation=False)
        validation_rows = _synthetic_rows(config, validation=True)
    else:
        directory = Path(config["dataset"]["directory"])
        capacity = int(config.get("pilot", {}).get("maximum_selected_rows", 2048))
        train_rows = _scan_bounded_rows(
            directory / str(config["dataset"].get("train_dataset", "eligible_train.parquet")),
            maximum_length=max(stage.maximum_length for stage in stages),
            seed=seed,
            capacity=capacity,
        )
        validation_rows = _scan_bounded_rows(
            directory / str(config["dataset"].get("validation_dataset", "eligible_validation.parquet")),
            maximum_length=max(stage.maximum_length for stage in stages),
            seed=seed + 1,
            capacity=int(config.get("pilot", {}).get("validation_panel_size", 48)),
        )
        overlap = {row["sample_id"] for row in train_rows} & {row["sample_id"] for row in validation_rows}
        if overlap:
            raise ValueError(f"Train/validation sample leakage in pilot panel: {sorted(overlap)[0]}")
    updates = []
    for stage_index, stage in enumerate(stages):
        eligible = [row for row in train_rows if int(row["sequence_length"]) <= stage.maximum_length]
        if not eligible:
            raise ValueError(f"No samples are eligible for curriculum stage {stage_index + 1}")
        strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            strata[(row["length_bin"], row["experimental_method"], row["pairing_classification"])].append(row)
        ordered_strata = []
        for key, rows in sorted(strata.items()):
            rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{stage_index}:{row['sample_id']}".encode()).hexdigest())
            ordered_strata.append((key, rows))
        for update_index in range(stage.optimizer_updates):
            _, selected_stratum = ordered_strata[update_index % len(ordered_strata)]
            anchor = selected_stratum[(update_index // len(ordered_strata)) % len(selected_stratum)]
            accumulation = accumulation_for_length(int(anchor["sequence_length"]), budget)
            same_bin = [row for row in eligible if row["length_bin"] == anchor["length_bin"]]
            microbatches = [
                same_bin[(update_index * accumulation + offset) % len(same_bin)] for offset in range(accumulation)
            ]
            updates.append(
                {
                    "stage_index": stage_index,
                    "stage_maximum_length": stage.maximum_length,
                    "stage_update_index": update_index,
                    "microbatches": [
                        {
                            "row_index": train_rows.index(row),
                            "sample_id": row["sample_id"],
                            "seed": seed + len(updates) * 1_000_003 + offset,
                        }
                        for offset, row in enumerate(microbatches)
                    ],
                }
            )
    core = {
        "version": "e005_bounded_pilot_plan_v1",
        "seed": seed,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "updates": updates,
    }
    core["plan_sha256"] = _canonical_hash(core)
    return core


def _dataset_identity(config: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {"kind": "synthetic", "sha256": _canonical_hash(config.get("pilot", {}).get("synthetic_lengths"))}
    directory = Path(config["dataset"]["directory"])
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "completed" or protocol.get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise ValueError("Pairing dataset protocol is incomplete or incompatible")
    if config["dataset"].get("immutable") is not True:
        raise ValueError("E005 pilot requires dataset.immutable=true")
    schema_path = directory / "schema.json"
    vocabulary_path = directory / "vocabulary.json"
    if json.loads(schema_path.read_text()).get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise ValueError("Pairing dataset schema metadata is incompatible")
    if json.loads(vocabulary_path.read_text()) != SequenceGeometryVocabulary().as_dict():
        raise ValueError("Pairing dataset vocabulary metadata is incompatible")
    if int(protocol.get("failure_count", 0)) != 0:
        raise ValueError("Pairing dataset protocol records validation failures")
    if protocol.get("input_hashes_preserved") is not True:
        raise ValueError("Pairing dataset protocol does not attest preserved source hashes")
    counts = protocol.get("validated_membership_counts", protocol)
    train_count = int(counts.get("eligible_train_count", -1))
    validation_count = int(counts.get("eligible_validation_count", -1))
    if train_count < 1 or validation_count < 1:
        raise ValueError("Pairing dataset protocol has invalid eligible train/validation counts")
    paths = [protocol_path]
    paths.extend(
        path
        for path in (directory / "schema.json", directory / "vocabulary.json", directory / "input_hashes.sha256")
        if path.exists()
    )
    datasets = (
        ("train_dataset", "eligible_train.parquet", train_count),
        ("validation_dataset", "eligible_validation.parquet", validation_count),
    )
    for key, fallback, expected_count in datasets:
        root = directory / str(config["dataset"].get(key, fallback))
        if int(ds.dataset(str(root), format="parquet").count_rows()) != expected_count:
            raise ValueError(f"Pairing dataset count contradiction for {key}")
    hashes = {str(path): _sha256_file(path) for path in paths}
    normalization_path = Path(str(config["normalization_file"]))
    hashes[str(normalization_path)] = _sha256_file(normalization_path)
    hashes.update(_verify_dataset_partitions(directory, protocol))
    return {"kind": "sequence_geometry_pairing_v1", "files": hashes, "sha256": _canonical_hash(hashes)}


def _model(config: dict[str, Any]) -> E005SequenceGeometryCoDesign:
    model_config = dict(config["model"])
    geometry_config = dict(model_config.pop("geometry_model"))
    return E005SequenceGeometryCoDesign(geometry_model=geometry_config, **model_config)


def _item_for_row(config: dict[str, Any], row: dict[str, Any], *, synthetic: bool, validation: bool) -> dict[str, Any]:
    if synthetic:
        items = _synthetic_items(
            [int(value) for value in config.get("pilot", {}).get("synthetic_lengths", [8, 12])],
            int(config["seed"]) + (1 if validation else 0),
        )
        item = items[int(row["synthetic_index"])]
        item["sample_id"] = row["sample_id"]
        return item
    table = pa.Table.from_pylist([row])
    return SequenceGeometryDataset(table, mode="geometry_conditioned", seed=int(config["seed"]))[0]


def _prepare_batch(
    config: dict[str, Any], model: E005SequenceGeometryCoDesign, item: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor | list[str]]:
    batch = collate_sequence_geometry([item], pad_id=model.pad_token_id, pad_to_multiple=model.downsample_factor)
    residue_mask = batch["sequence_mask"].to(device)
    pair_mask = (residue_mask[:, None, :, None] & residue_mask[:, None, None, :]).bool()
    if not torch.equal(pair_mask[:, 0].cpu(), batch["pair_mask"]):
        raise ValueError("invalid_masks: collated pair mask does not match residue mask")
    scale = float(config.get("normalization_scale_angstrom", 50.0))
    if config.get("normalization_file"):
        normalization = json.loads(Path(config["normalization_file"]).read_text())
        scale = float(normalization["scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization scale must be positive and finite")
    return {
        "sample_ids": batch["sample_ids"],
        "tokens": batch["sequence_token_ids"].to(device),
        "residue_mask": residue_mask,
        "pair_mask": pair_mask,
        "clean": batch["distance_matrices"][:, None].to(device) / scale,
        "lengths": batch["lengths"].to(device),
        "separation": make_sequence_separation(batch["lengths"], int(pair_mask.shape[-1])).to(device),
    }


def _forward_loss(
    config: dict[str, Any],
    model: E005SequenceGeometryCoDesign,
    diffusion: GaussianDiffusion,
    batch: dict[str, Any],
    *,
    mode: str,
    stochastic_seed: int,
    dropout_probability: float,
    activation_checkpointing: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    device = batch["clean"].device
    generator = torch.Generator(device=device).manual_seed(stochastic_seed)
    timesteps = torch.randint(0, diffusion.timesteps, (1,), generator=generator, device=device)
    noisy, epsilon = diffusion.q_sample(batch["clean"], timesteps, batch["pair_mask"], generator=generator)
    token_inputs, masked = masked_sequence_inputs(
        batch["tokens"],
        batch["residue_mask"],
        mask_token_id=model.mask_token_id,
        probability=float(config["masked_token_probability"]),
        seed=stochastic_seed,
        step=0,
    )
    availability = conditioning_mask(
        1,
        probability=dropout_probability,
        seed=stochastic_seed,
        step=0,
        device=device,
    )

    def tensors(noisy_input: torch.Tensor):
        output = model(
            sequence_token_ids=token_inputs,
            residue_mask=batch["residue_mask"],
            noisy_geometry=noisy_input,
            timesteps=timesteps,
            lengths=batch["lengths"],
            sequence_separation=batch["separation"],
            pair_mask=batch["pair_mask"],
            geometry_conditioning_mask=availability,
            mode=mode,
        )
        return (
            output["sequence_logits"],
            output["geometry_prediction"],
            output["sequence_pair_prediction"],
            output["geometry_to_sequence_gate"],
            output["sequence_to_geometry_gate"],
            output["return_geometry_gate"],
        )

    context = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
    with context:
        values = checkpoint(tensors, noisy, use_reentrant=False) if activation_checkpointing else tensors(noisy)
        logits, geometry, sequence_pair, geometry_gate, pair_gate, return_gate = values
        outputs = {
            "sequence_logits": logits,
            "geometry_prediction": geometry,
            "sequence_pair_prediction": sequence_pair,
            "residue_mask": batch["residue_mask"],
            "pair_mask": batch["pair_mask"],
        }
        target = diffusion.training_target(
            x_start=batch["clean"],
            t=timesteps,
            epsilon=epsilon,
            prediction_type=str(config["diffusion"]["prediction_parameterization"]),
        )
        loss_config = config["loss"]
        losses = codesign_losses(
            outputs,
            sequence_targets=batch["tokens"],
            masked_token_mask=masked,
            geometry_target=target,
            weights=CoDesignLossWeights(
                sequence=float(loss_config["sequence_weight"]),
                geometry=float(loss_config["geometry_weight"]),
                consistency=float(loss_config["consistency_weight"]),
            ),
        )
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions[masked] == batch["tokens"][masked]).float().mean())
    diagnostics = {
        "sequence_accuracy": accuracy,
        "sequence_perplexity": float(torch.exp(losses["sequence"].detach().float()).clamp_max(1e12)),
        "geometry_denoising_error": float(losses["geometry"].detach()),
        "conditioning_dropped": not bool(availability.item()),
        "timestep": int(timesteps.item()),
        "stochastic_seed": stochastic_seed,
        "masked_sequence_sha256": _tensor_sha256(token_inputs),
        "masked_token_mask_sha256": _tensor_sha256(masked),
        "geometry_noise_sha256": _tensor_sha256(epsilon),
        "masked_token_count": int(masked.sum()),
        "valid_pair_count": int(torch.triu(batch["pair_mask"][:, 0], diagonal=1).sum()),
    }
    gates = _gate_statistics(geometry_gate, pair_gate, return_gate, batch["residue_mask"])
    return losses, diagnostics, gates


def _active_parameter_groups(mode: str) -> set[str]:
    if mode == "sequence_only":
        return {"sequence_branch", "geometry_branch"}
    return {
        "sequence_branch",
        "geometry_branch",
        "sequence_to_geometry_feedback",
        "geometry_to_sequence_feedback",
        "gates",
    }


def _group_gradient_metrics(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]], mode: str
) -> dict[str, float | None]:
    active = _active_parameter_groups(mode)
    result: dict[str, float | None] = {}
    for name, parameters in groups.items():
        with_grad = [parameter for _, parameter in parameters if parameter.grad is not None]
        if name in active and len(with_grad) != len(parameters):
            missing = [parameter_name for parameter_name, parameter in parameters if parameter.grad is None]
            raise RuntimeError(f"missing_gradients:{name}:{','.join(missing[:10])}")
        if not with_grad:
            result[name] = None
            continue
        norm = sum(float(parameter.grad.detach().float().square().sum()) for parameter in with_grad) ** 0.5
        if not np.isfinite(norm):
            raise FloatingPointError(f"nonfinite_gradient:{name}")
        result[name] = norm
    return result


def _gradient_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _restore_gradients(model: torch.nn.Module, gradients: dict[str, torch.Tensor], device: torch.device) -> None:
    parameters = dict(model.named_parameters())
    for name, value in gradients.items():
        parameters[name].grad = value.to(device=device, dtype=parameters[name].dtype)


def _mean_metrics(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(np.mean([float(record[key]) for record in records])) for key in keys}


def _stratified_training_metrics(path: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["record_type"] == "microbatch":
                key = (
                    record["arm"],
                    record["stage"],
                    record["length_bin"],
                    record["experimental_method"],
                    record["pairing_classification"],
                )
                groups[key].append(record)
    result = []
    for key, records in sorted(groups.items()):
        loss_names = ("total", "sequence", "geometry", "consistency")
        result.append(
            {
                "arm": key[0],
                "stage": key[1],
                "length_bin": key[2],
                "experimental_method": key[3],
                "pairing_classification": key[4],
                "record_count": len(records),
                "mean_losses": {
                    name: float(np.mean([float(record["losses"][name]) for record in records])) for name in loss_names
                },
                **_mean_metrics(records, ("sequence_accuracy", "sequence_perplexity", "geometry_denoising_error")),
                "conditioning_dropout_frequency": float(
                    np.mean([bool(record["conditioning_dropped"]) for record in records])
                ),
                "mean_gate_statistics": {
                    gate_name: {
                        statistic: float(
                            np.mean([record["gate_statistics"][gate_name][statistic] for record in records])
                        )
                        for statistic in ("minimum", "maximum", "mean", "standard_deviation", "saturated_fraction")
                    }
                    for gate_name in (
                        "geometry_to_sequence",
                        "sequence_to_geometry",
                        "return_geometry_to_sequence",
                    )
                },
            }
        )
    return result


def _stratified_validation_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (
                record["checkpoint_arm"],
                record["validation_mode"],
                record["length_bin"],
                record["experimental_method"],
                record["pairing_classification"],
            )
        ].append(record)
    result = []
    for key, values in sorted(groups.items()):
        result.append(
            {
                "checkpoint_arm": key[0],
                "validation_mode": key[1],
                "length_bin": key[2],
                "experimental_method": key[3],
                "pairing_classification": key[4],
                "record_count": len(values),
                "mean_losses": {
                    name: float(np.mean([float(record["losses"][name]) for record in values]))
                    for name in ("total", "sequence", "geometry", "consistency")
                },
                **_mean_metrics(values, ("sequence_accuracy", "sequence_perplexity", "geometry_denoising_error")),
            }
        )
    return result


class _AtomicJsonl:
    def __init__(self, final_path: Path, *, resume: bool) -> None:
        self.final_path = final_path
        self.working_path = final_path.with_suffix(final_path.suffix + ".inprogress")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.working_path.unlink(missing_ok=True)
        elif self.final_path.exists() and not self.working_path.exists():
            self.final_path.replace(self.working_path)

    def append(self, payload: dict[str, Any]) -> None:
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
        with self.working_path.open("ab", buffering=0) as handle:
            handle.write(encoded)
            os.fsync(handle.fileno())

    def offset(self) -> int:
        return self.working_path.stat().st_size if self.working_path.exists() else 0

    def truncate(self, offset: int) -> None:
        if not self.working_path.exists() and offset:
            raise ValueError("Metrics journal is missing for a nonzero checkpoint offset")
        with self.working_path.open("ab") as handle:
            handle.truncate(offset)

    def publish(self) -> None:
        self.working_path.replace(self.final_path)


def _checkpoint_payload(
    *,
    arm: str,
    config_hash: str,
    dataset_hash: str,
    plan_hash: str,
    initialization_hash: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    update_cursor: int,
    microstep: int,
    optimizer_step: int,
    accumulated_pair_tokens: int,
    curriculum_stage: int | None,
    metrics_offset: int,
    counters: dict[str, int],
) -> dict[str, Any]:
    return {
        "version": "e005_bounded_pilot_checkpoint_v1",
        "architecture_version": E005_ARCHITECTURE_VERSION,
        "arm": arm,
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
        "plan_sha256": plan_hash,
        "initialization_sha256": initialization_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "grad_scaler": scaler.state_dict(),
        "rng_state": _rng_state(),
        "sampler_cursor": update_cursor,
        "curriculum_stage": curriculum_stage,
        "microstep": microstep,
        "optimizer_step": optimizer_step,
        "accumulated_pair_token_count": accumulated_pair_tokens,
        "gradients": _gradient_state(model),
        "counters": counters,
        "metrics_offset": metrics_offset,
    }


def _validate_checkpoint(
    checkpoint_payload: dict[str, Any], *, arm: str, config_hash: str, dataset_hash: str, plan_hash: str
) -> None:
    expected = {
        "arm": arm,
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
        "plan_sha256": plan_hash,
        "architecture_version": E005_ARCHITECTURE_VERSION,
    }
    mismatches = [name for name, value in expected.items() if checkpoint_payload.get(name) != value]
    if mismatches:
        raise ValueError(f"Incompatible E005 pilot checkpoint fields: {', '.join(mismatches)}")


def _validate_pilot_config(config: dict[str, Any], *, synthetic: bool) -> None:
    pilot = config.get("pilot", {})
    if int(pilot.get("physical_batch_size", 1)) != 1:
        raise ValueError("E005 bounded pilot requires physical_batch_size=1")
    if str(config.get("amp_dtype", "float16")) != "float16":
        raise ValueError("E005 bounded pilot requires amp_dtype=float16")
    if pilot.get("activation_checkpointing", True) is not True:
        raise ValueError("E005 bounded pilot requires non-reentrant activation checkpointing")
    if int(pilot.get("max_rss_mib", MAX_RSS_MIB)) > MAX_RSS_MIB:
        raise ValueError("E005 bounded pilot RSS limit must not exceed 4096 MiB")
    if int(pilot.get("max_cuda_memory_mib", 8192)) > 8192:
        raise ValueError("E005 bounded pilot CUDA-memory limit must not exceed 8192 MiB")
    if not 1 <= int(pilot.get("checkpoint_frequency", 10)) <= 100:
        raise ValueError("checkpoint_frequency must be in [1, 100]")
    if not 1 <= int(pilot.get("validation_panel_size", 48)) <= MAX_DIAGNOSTIC_EXAMPLES:
        raise ValueError(f"validation_panel_size must be in [1, {MAX_DIAGNOSTIC_EXAMPLES}]")
    if int(pilot.get("maximum_selected_rows", 2048)) > 4096:
        raise ValueError("maximum_selected_rows must not exceed 4096")
    stages = curriculum_stages(config)
    budget = pair_budget(config)
    if not synthetic:
        if tuple(stage.maximum_length for stage in stages) != (128, 256, 500):
            raise ValueError("Real E005 pilot curriculum lengths must be 128, 256, 500")
        if budget != DEFAULT_PAIR_BUDGET:
            raise ValueError("Real E005 pilot pair-budget accumulation must retain the validated schedule")
        if config.get("mixed_precision") is not True:
            raise ValueError("Real E005 pilot requires CUDA AMP")
        if str(config.get("device")) != "cuda":
            raise ValueError("Real E005 pilot requires device=cuda")
        dataset = config.get("dataset", {})
        if Path(str(dataset.get("train_dataset", ""))).stem != "eligible_train":
            raise ValueError("E005 pilot training input must be eligible_train")
        if Path(str(dataset.get("validation_dataset", ""))).stem != "eligible_validation":
            raise ValueError("E005 pilot validation input must be eligible_validation")


def _run_arm(
    config: dict[str, Any],
    *,
    arm: str,
    plan: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    initialization_hash: str,
    dataset_hash: str,
    output_dir: Path,
    synthetic: bool,
    resume: bool,
    interrupted: list[bool],
) -> dict[str, Any]:
    device = torch.device(str(config.get("device", "cpu")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E005 pilot requests CUDA but CUDA is unavailable")
    pilot = config["pilot"]
    rss_limit = int(pilot.get("max_rss_mib", MAX_RSS_MIB))
    cuda_limit = int(pilot.get("max_cuda_memory_mib", 8192))
    if rss_limit > MAX_RSS_MIB:
        raise ValueError(f"max_rss_mib must not exceed {MAX_RSS_MIB}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    random.seed(int(config["seed"]))
    model = _model(config).to(device)
    model.load_state_dict(initial_state)
    model.train()
    groups = _parameter_groups(model)
    initial_parameters = _parameter_snapshots(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    total_updates = len(plan["updates"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_updates, 1))
    amp_enabled = bool(config.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = output_dir / "checkpoints" / f"{arm}_latest.pt"
    metrics = _AtomicJsonl(output_dir / "metrics" / f"{arm}.jsonl", resume=resume)
    config_hash = _configuration_sha256(config)
    update_cursor = microstep = optimizer_step = accumulated_pair_tokens = 0
    counters = {
        "samples": 0,
        "pair_tokens": 0,
        "skipped_updates": 0,
        "consecutive_skipped_updates": 0,
        "nonfinite": 0,
        "dropout": 0,
    }
    if resume and checkpoint_path.exists():
        saved = load_checkpoint(checkpoint_path, map_location=device)
        _validate_checkpoint(
            saved,
            arm=arm,
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            plan_hash=plan["plan_sha256"],
        )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["grad_scaler"])
        _restore_rng_state(saved.get("rng_state"))
        update_cursor = int(saved["sampler_cursor"])
        microstep = int(saved["microstep"])
        optimizer_step = int(saved["optimizer_step"])
        accumulated_pair_tokens = int(saved["accumulated_pair_token_count"])
        counters = {name: int(value) for name, value in saved["counters"].items()}
        _restore_gradients(model, saved.get("gradients", {}), device)
        metrics.truncate(int(saved.get("metrics_offset", 0)))
    elif checkpoint_path.exists():
        raise FileExistsError(f"Pilot checkpoint already exists; use --resume or a new output: {checkpoint_path}")

    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["diffusion"]["steps"]))).to(device)
    amp_dtype = torch.float16
    activation_checkpointing = bool(pilot.get("activation_checkpointing", True))
    dropout_probability = 0.0 if arm == "sequence_only" else float(config["conditioning"]["dropout_probability"])
    checkpoint_frequency = int(pilot.get("checkpoint_frequency", 10))
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True) if microstep == 0 else None
    last_stage = None
    backward_in_progress = False
    try:
        while update_cursor < total_updates:
            update = plan["updates"][update_cursor]
            last_stage = int(update["stage_index"])
            microbatches = update["microbatches"]
            for micro_index in range(microstep, len(microbatches)):
                _guard_memory(device, rss_limit, cuda_limit)
                entry = microbatches[micro_index]
                row = plan["train_rows"][int(entry["row_index"])]
                item = _item_for_row(config, row, synthetic=synthetic, validation=False)
                batch = _prepare_batch(config, model, item, device)
                padded = int(batch["pair_mask"].shape[-1])
                if len(microbatches) != accumulation_for_length(padded, pair_budget(config)):
                    raise RuntimeError("pair_budget_plan_contradiction")
                losses, diagnostics, gates = _forward_loss(
                    config,
                    model,
                    diffusion,
                    batch,
                    mode=arm,
                    stochastic_seed=int(entry["seed"]),
                    dropout_probability=dropout_probability,
                    activation_checkpointing=activation_checkpointing,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
                values = {name: float(value.detach()) for name, value in losses.items()}
                if not all(np.isfinite(value) for value in values.values()):
                    counters["nonfinite"] += 1
                    raise FloatingPointError(f"nonfinite_loss:{values}")
                backward_in_progress = True
                scaler.scale(losses["total"] / len(microbatches)).backward()
                backward_in_progress = False
                microstep = micro_index + 1
                pair_tokens = int(batch["pair_mask"].sum())
                accumulated_pair_tokens += pair_tokens
                counters["samples"] += 1
                counters["pair_tokens"] += pair_tokens
                counters["dropout"] += int(diagnostics["conditioning_dropped"])
                elapsed = max(time.monotonic() - started, 1e-12)
                metrics.append(
                    {
                        "record_type": "microbatch",
                        "arm": arm,
                        "stage": last_stage + 1,
                        "optimizer_step": optimizer_step,
                        "microstep": microstep,
                        "sample_id": row["sample_id"],
                        "actual_length": int(row["sequence_length"]),
                        "padded_length": padded,
                        "length_bin": row["length_bin"],
                        "experimental_method": row["experimental_method"],
                        "pairing_classification": row["pairing_classification"],
                        "losses": values,
                        **diagnostics,
                        "gate_statistics": gates,
                        "samples": counters["samples"],
                        "pair_tokens": counters["pair_tokens"],
                        "examples_per_second": counters["samples"] / elapsed,
                        "pair_tokens_per_second": counters["pair_tokens"] / elapsed,
                        "memory": _memory(device),
                    }
                )
                del item, batch, losses
                _guard_memory(device, rss_limit, cuda_limit)
                if interrupted[0]:
                    raise PilotInterrupted("SIGINT received; resumable state was requested")
            scaler.unscale_(optimizer)
            gradient_norms = _group_gradient_metrics(groups, arm)
            old_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            skipped = float(scaler.get_scale()) < old_scale
            if skipped:
                counters["skipped_updates"] += 1
                counters["consecutive_skipped_updates"] += 1
                if counters["consecutive_skipped_updates"] >= int(pilot.get("maximum_consecutive_skipped_updates", 8)):
                    raise FloatingPointError("maximum_consecutive_amp_overflows_reached")
            else:
                optimizer_step += 1
                scheduler.step()
                counters["consecutive_skipped_updates"] = 0
            optimizer.zero_grad(set_to_none=True)
            if not skipped:
                update_cursor += 1
            microstep = 0
            accumulated_pair_tokens = 0
            metrics.append(
                {
                    "record_type": "optimizer_step",
                    "arm": arm,
                    "stage": last_stage + 1,
                    "optimizer_step": optimizer_step,
                    "gradient_norms": gradient_norms,
                    "parameter_change_norms": _parameter_change_norms(groups, initial_parameters),
                    "length_bin": row["length_bin"],
                    "experimental_method": row["experimental_method"],
                    "pairing_classification": row["pairing_classification"],
                    "amp_scale": float(scaler.get_scale()),
                    "update_skipped": skipped,
                    "memory": _memory(device),
                }
            )
            if update_cursor % checkpoint_frequency == 0 or update_cursor == total_updates:
                save_checkpoint(
                    checkpoint_path,
                    _checkpoint_payload(
                        arm=arm,
                        config_hash=config_hash,
                        dataset_hash=dataset_hash,
                        plan_hash=plan["plan_sha256"],
                        initialization_hash=initialization_hash,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        update_cursor=update_cursor,
                        microstep=microstep,
                        optimizer_step=optimizer_step,
                        accumulated_pair_tokens=accumulated_pair_tokens,
                        curriculum_stage=last_stage,
                        metrics_offset=metrics.offset(),
                        counters=counters,
                    ),
                )
            print(
                f"arm={arm} stage={last_stage + 1} update={update_cursor}/{total_updates} "
                f"optimizer_step={optimizer_step} rss={_rss_mib():.1f}MiB",
                flush=True,
            )
            _atomic_json(
                output_dir / "heartbeat.json",
                {
                    "status": "running",
                    "arm": arm,
                    "curriculum_stage": last_stage + 1,
                    "sampler_cursor": update_cursor,
                    "optimizer_step": optimizer_step,
                    "memory": _memory(device),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                },
            )
    except BaseException:
        if backward_in_progress:
            optimizer.zero_grad(set_to_none=True)
            microstep = 0
            accumulated_pair_tokens = 0
        save_checkpoint(
            checkpoint_path,
            _checkpoint_payload(
                arm=arm,
                config_hash=config_hash,
                dataset_hash=dataset_hash,
                plan_hash=plan["plan_sha256"],
                initialization_hash=initialization_hash,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                update_cursor=update_cursor,
                microstep=microstep,
                optimizer_step=optimizer_step,
                accumulated_pair_tokens=accumulated_pair_tokens,
                curriculum_stage=last_stage,
                metrics_offset=metrics.offset(),
                counters=counters,
            ),
        )
        raise
    metrics.publish()
    metrics_path = output_dir / "metrics" / f"{arm}.jsonl"
    return {
        "status": "completed",
        "arm": arm,
        "initialization_sha256": initialization_hash,
        "optimizer_steps": optimizer_step,
        "microsteps": counters["samples"],
        "samples": counters["samples"],
        "pair_tokens": counters["pair_tokens"],
        "conditioning_dropout_frequency": counters["dropout"] / max(counters["samples"], 1),
        "skipped_updates": counters["skipped_updates"],
        "nonfinite_count": counters["nonfinite"],
        "parameter_change_norms": _parameter_change_norms(groups, initial_parameters),
        "memory": _memory(device),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "stratified_metrics": _stratified_training_metrics(metrics_path),
    }


@torch.no_grad()
def _validate_checkpoints(
    config: dict[str, Any], plan: dict[str, Any], output_dir: Path, *, synthetic: bool
) -> list[dict[str, Any]]:
    device = torch.device(str(config.get("device", "cpu")))
    diffusion = GaussianDiffusion(cosine_beta_schedule(int(config["diffusion"]["steps"]))).to(device)
    results = []
    for checkpoint_arm, modes in (
        ("sequence_only", ("sequence_only",)),
        ("learned_geometry_gating", VALIDATION_MODES),
    ):
        checkpoint_payload = load_checkpoint(
            output_dir / "checkpoints" / f"{checkpoint_arm}_latest.pt", map_location=device
        )
        model = _model(config).to(device)
        model.load_state_dict(checkpoint_payload["model"])
        model.eval()
        for row_index, row in enumerate(plan["validation_rows"]):
            item = _item_for_row(config, row, synthetic=synthetic, validation=True)
            batch = _prepare_batch(config, model, item, device)
            for mode in modes:
                losses, diagnostics, gates = _forward_loss(
                    config,
                    model,
                    diffusion,
                    batch,
                    mode=mode,
                    stochastic_seed=int(config["seed"]) + row_index * 31,
                    dropout_probability=0.0,
                    activation_checkpointing=False,
                    amp_enabled=bool(config.get("mixed_precision", True)) and device.type == "cuda",
                    amp_dtype=torch.float16,
                )
                results.append(
                    {
                        "checkpoint_arm": checkpoint_arm,
                        "sample_id": row["sample_id"],
                        "validation_mode": mode,
                        "length_bin": row["length_bin"],
                        "experimental_method": row["experimental_method"],
                        "pairing_classification": row["pairing_classification"],
                        "losses": {name: float(value) for name, value in losses.items()},
                        **diagnostics,
                        "gate_statistics": gates,
                    }
                )
        del model
    return results


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        (path for path in (output_dir / "checkpoints").glob("*_latest.pt") if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    return checkpoints[-1] if checkpoints else None


def _verified_final_position(
    config: dict[str, Any], summary: dict[str, Any], plan: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    stages = curriculum_stages(config)
    updates = plan.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ValueError("Completed pilot plan has no optimizer updates")
    expected_updates = sum(stage.optimizer_updates for stage in stages)
    if len(updates) != expected_updates:
        raise ValueError("Completed pilot plan/configuration update-count contradiction")
    stage_counts = [0] * len(stages)
    for update in updates:
        stage_index = int(update.get("stage_index", -1))
        if not 0 <= stage_index < len(stages):
            raise ValueError("Completed pilot plan contains an invalid curriculum stage")
        if int(update.get("stage_maximum_length", -1)) != stages[stage_index].maximum_length:
            raise ValueError("Completed pilot plan/configuration stage-length contradiction")
        stage_counts[stage_index] += 1
    if stage_counts != [stage.optimizer_updates for stage in stages]:
        raise ValueError("Completed pilot plan/configuration stage-count contradiction")
    arm_summaries = summary.get("training_arms")
    if not isinstance(arm_summaries, list) or [item.get("arm") for item in arm_summaries] != list(TRAINING_ARMS):
        raise ValueError("Completed pilot training-arm order is malformed")
    final_arm_summary = arm_summaries[-1]
    final_arm = str(final_arm_summary["arm"])
    standalone_path = output_dir / f"{final_arm}_summary.json"
    standalone = json.loads(standalone_path.read_text())
    if standalone != final_arm_summary or standalone.get("status") != "completed":
        raise ValueError("Completed pilot final-arm summary contradiction")
    checkpoint_path = output_dir / "checkpoints" / f"{final_arm}_latest.pt"
    checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
    final_stage_index = len(stages) - 1
    optimizer_step = int(final_arm_summary.get("optimizer_steps", -1))
    if (
        checkpoint_payload.get("arm") != final_arm
        or int(checkpoint_payload.get("curriculum_stage", -1)) != final_stage_index
        or int(checkpoint_payload.get("sampler_cursor", -1)) != len(updates)
        or int(checkpoint_payload.get("optimizer_step", -1)) != optimizer_step
        or optimizer_step != expected_updates
        or int(updates[-1].get("stage_index", -1)) != final_stage_index
    ):
        raise ValueError("Completed pilot final execution position is contradictory")
    return {
        "arm": final_arm,
        "curriculum_stage": final_stage_index + 1,
        "optimizer_step": optimizer_step,
    }


def _completed_heartbeat(
    summary_path: Path,
    summary: dict[str, Any],
    position: dict[str, Any],
    *,
    checkpoint_hashes: dict[str, str],
    arm_summary_hashes: dict[str, str],
) -> dict[str, Any]:
    heartbeat = {
        "status": "completed",
        "completed_utc": summary["completed_utc"],
        **position,
        "final_optimizer_step": position["optimizer_step"],
        "final_arm": position["arm"],
        "final_stage": position["curriculum_stage"],
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "checkpoint_sha256": checkpoint_hashes,
        "arm_summary_sha256": arm_summary_hashes,
    }
    return heartbeat


def _heartbeat_is_fully_valid(existing: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(existing, dict) or existing.get("status") != "completed":
        return False
    required = {
        "status",
        "completed_utc",
        "arm",
        "curriculum_stage",
        "optimizer_step",
        "summary_path",
        "summary_sha256",
        "checkpoint_sha256",
        "arm_summary_sha256",
    }
    return required <= set(existing) and all(existing.get(key) == expected.get(key) for key in required)


def _terminal_status(error: BaseException) -> str:
    if isinstance(error, PilotInterrupted):
        return "interrupted"
    if isinstance(error, MemoryError):
        return "memory_limit_exceeded"
    return "failed"


def _publish_terminal_heartbeat(output_dir: Path, error: BaseException) -> None:
    checkpoint = _latest_checkpoint(output_dir)
    _atomic_json(
        output_dir / "heartbeat.json",
        {
            "status": _terminal_status(error),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "latest_resumable_checkpoint": str(checkpoint) if checkpoint else None,
            "latest_resumable_checkpoint_sha256": _sha256_file(checkpoint) if checkpoint else None,
        },
    )


def _run_bounded_pilot(
    config: dict[str, Any], *, output_dir: str | Path, synthetic: bool = False, resume: bool = False
) -> dict[str, Any]:
    """Train paired E005 pilot arms and publish a bounded report."""
    output_dir = Path(output_dir)
    _validate_pilot_config(config, synthetic=synthetic)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
        raise FileExistsError(f"Refusing to overwrite completed E005 pilot: {output_dir}")
    if output_dir.exists() and not resume and any(output_dir.iterdir()):
        raise FileExistsError(f"E005 pilot output is non-empty; use --resume or a new directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _dataset_identity(config, synthetic=synthetic)
    plan_path = output_dir / "paired_plan.json"
    if resume:
        plan = json.loads(plan_path.read_text())
    else:
        plan = build_paired_plan(config, synthetic=synthetic)
        _atomic_json(plan_path, plan)
    if plan.get("plan_sha256") != _canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"}):
        raise ValueError("Paired pilot plan hash mismatch")
    torch.manual_seed(int(config["seed"]))
    initial_model = _model(config)
    initial_state = {name: value.detach().cpu().clone() for name, value in initial_model.state_dict().items()}
    initialization_hash = _state_dict_hash(initial_state)
    del initial_model
    interrupted = [False]
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_interrupt(signum: int, frame: Any) -> None:
        del signum, frame
        interrupted[0] = True

    signal.signal(signal.SIGINT, request_interrupt)
    started = time.monotonic()
    arms = []
    try:
        for arm in TRAINING_ARMS:
            arm_summary = output_dir / f"{arm}_summary.json"
            if resume and arm_summary.exists() and json.loads(arm_summary.read_text()).get("status") == "completed":
                arms.append(json.loads(arm_summary.read_text()))
                continue
            result = _run_arm(
                config,
                arm=arm,
                plan=plan,
                initial_state=initial_state,
                initialization_hash=initialization_hash,
                dataset_hash=identity["sha256"],
                output_dir=output_dir,
                synthetic=synthetic,
                resume=resume,
                interrupted=interrupted,
            )
            _atomic_json(arm_summary, result)
            arms.append(result)
        if interrupted[0]:
            raise PilotInterrupted("SIGINT received before validation")
        validation = _validate_checkpoints(config, plan, output_dir, synthetic=synthetic)
        if interrupted[0]:
            raise PilotInterrupted("SIGINT received during validation")
        validation_metrics = _AtomicJsonl(output_dir / "metrics" / "learned_validation.jsonl", resume=False)
        for record in validation:
            validation_metrics.append(record)
        validation_metrics.publish()
        identity_after = _dataset_identity(config, synthetic=synthetic)
        if identity_after != identity:
            raise RuntimeError("dataset_mutation_detected")
        summary = {
            "status": "completed",
            "experiment": "E005_bounded_real_data_pilot",
            "architecture_version": E005_ARCHITECTURE_VERSION,
            "config_sha256": _configuration_sha256(config),
            "dataset_identity_before": identity,
            "dataset_identity_after": identity_after,
            "dataset_inputs_unchanged": True,
            "paired_plan_sha256": plan["plan_sha256"],
            "shared_initialization_sha256": initialization_hash,
            "training_arms": arms,
            "validation_modes": list(VALIDATION_MODES),
            "validation_records": validation,
            "stratified_validation_metrics": _stratified_validation_metrics(validation),
            "elapsed_seconds": time.monotonic() - started,
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(summary_path, summary)
        return summary
    except BaseException as error:
        _atomic_json(
            output_dir / "incomplete.json",
            {
                "status": _terminal_status(error),
                "error_type": type(error).__name__,
                "error": str(error),
                "resume_command_required": True,
                "dataset_identity": identity,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
        )
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def run_bounded_pilot(
    config: dict[str, Any], *, output_dir: str | Path, synthetic: bool = False, resume: bool = False
) -> dict[str, Any]:
    """Run the pilot and atomically publish its terminal heartbeat state."""
    destination = Path(output_dir)
    summary_path = destination / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed":
        raise FileExistsError(f"Refusing to overwrite completed E005 pilot: {destination}")
    try:
        summary = _run_bounded_pilot(config, output_dir=destination, synthetic=synthetic, resume=resume)
    except BaseException as error:
        if not (summary_path.exists() and json.loads(summary_path.read_text()).get("status") == "completed"):
            started = (destination / "paired_plan.json").exists() or (destination / "checkpoints").exists()
            if started:
                _publish_terminal_heartbeat(destination, error)
        raise
    plan = json.loads((destination / "paired_plan.json").read_text())
    position = _verified_final_position(config, summary, plan, destination)
    checkpoint_hashes = {
        str(item["arm"]): _sha256_file(destination / "checkpoints" / f"{item['arm']}_latest.pt")
        for item in summary["training_arms"]
    }
    arm_summary_hashes = {
        str(item["arm"]): _sha256_file(destination / f"{item['arm']}_summary.json") for item in summary["training_arms"]
    }
    heartbeat = _completed_heartbeat(
        summary_path,
        summary,
        position,
        checkpoint_hashes=checkpoint_hashes,
        arm_summary_hashes=arm_summary_hashes,
    )
    _atomic_json(destination / "heartbeat.json", heartbeat)
    return summary


def finalize_completed_heartbeat(
    config: dict[str, Any], *, output_dir: str | Path, synthetic: bool = False
) -> dict[str, Any]:
    """Read-only attest a completed run, then replace only its stale heartbeat."""
    destination = Path(output_dir)
    summary_path = destination / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed pilot summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "completed":
        raise ValueError("Heartbeat repair requires a completed summary")
    if summary.get("config_sha256") != _configuration_sha256(config):
        raise ValueError("Heartbeat repair configuration hash mismatch")
    plan_path = destination / "paired_plan.json"
    plan = json.loads(plan_path.read_text())
    plan_hash = _canonical_hash({key: value for key, value in plan.items() if key != "plan_sha256"})
    if plan.get("plan_sha256") != plan_hash or summary.get("paired_plan_sha256") != plan_hash:
        raise ValueError("Heartbeat repair paired-plan hash mismatch")
    current_identity = _dataset_identity(config, synthetic=synthetic)
    if (
        summary.get("dataset_inputs_unchanged") is not True
        or summary.get("dataset_identity_before") != current_identity
        or summary.get("dataset_identity_after") != current_identity
    ):
        raise ValueError("Heartbeat repair dataset-preservation verification failed")
    checkpoint_hashes = {}
    arm_summary_hashes = {}
    arm_summaries = summary.get("training_arms")
    if not isinstance(arm_summaries, list) or [item.get("arm") for item in arm_summaries] != list(TRAINING_ARMS):
        raise ValueError("Heartbeat repair training-arm summaries are malformed")
    for arm_summary in arm_summaries:
        arm = str(arm_summary["arm"])
        arm_summary_path = destination / f"{arm}_summary.json"
        if json.loads(arm_summary_path.read_text()) != arm_summary:
            raise ValueError(f"Heartbeat repair arm-summary contradiction for {arm}")
        checkpoint_path = destination / "checkpoints" / f"{arm}_latest.pt"
        checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
        _validate_checkpoint(
            checkpoint_payload,
            arm=arm,
            config_hash=str(summary["config_sha256"]),
            dataset_hash=str(current_identity["sha256"]),
            plan_hash=plan_hash,
        )
        if checkpoint_payload.get("initialization_sha256") != summary.get("shared_initialization_sha256") or int(
            checkpoint_payload.get("optimizer_step", -1)
        ) != int(arm_summary.get("optimizer_steps", -2)):
            raise ValueError(f"Heartbeat repair checkpoint provenance mismatch for {arm}")
        checkpoint_digest = _sha256_file(checkpoint_path)
        expected_digest = arm_summary.get("checkpoint_sha256")
        if expected_digest is not None and checkpoint_digest != expected_digest:
            raise ValueError(f"Heartbeat repair checkpoint SHA-256 mismatch for {arm}")
        checkpoint_hashes[arm] = checkpoint_digest
        arm_summary_hashes[arm] = _sha256_file(arm_summary_path)
    position = _verified_final_position(config, summary, plan, destination)
    heartbeat = _completed_heartbeat(
        summary_path,
        summary,
        position,
        checkpoint_hashes=checkpoint_hashes,
        arm_summary_hashes=arm_summary_hashes,
    )
    heartbeat_path = destination / "heartbeat.json"
    existing = json.loads(heartbeat_path.read_text()) if heartbeat_path.is_file() else None
    if _heartbeat_is_fully_valid(existing, heartbeat):
        return existing
    heartbeat["repair_finalization"] = True
    _atomic_json(heartbeat_path, heartbeat)
    return heartbeat
