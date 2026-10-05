"""Bounded E005 co-design dry-run utilities."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.checkpoint import checkpoint

from protein_distance_diffusion.data.collate import make_sequence_separation
from protein_distance_diffusion.data.sequence_geometry import (
    PAIRING_SCHEMA_VERSION,
    SequenceGeometryDataset,
    SequenceGeometryVocabulary,
    build_geometry_corruption,
    collate_sequence_geometry,
)
from protein_distance_diffusion.diffusion.gaussian import GaussianDiffusion
from protein_distance_diffusion.diffusion.schedules import cosine_beta_schedule
from protein_distance_diffusion.models.codesign import (
    CONDITIONING_MODES,
    E005_ARCHITECTURE_VERSION,
    CoDesignLossWeights,
    E005SequenceGeometryCoDesign,
    codesign_losses,
)
from protein_distance_diffusion.training.checkpointing import save_checkpoint

MAX_DRY_RUN_STEPS = 10
MAX_DRY_RUN_SAMPLES = 8
MAX_DRY_RUN_LENGTH = 500
MAX_DRY_RUN_MEMORY_MIB = 4096
MAX_TINY_OVERFIT_STEPS = 50
MAX_TINY_OVERFIT_LENGTH = 128


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _guard_memory(limit_mib: int, device: torch.device | None = None) -> None:
    current = _rss_mib()
    peak = _peak_rss_mib()
    if current > limit_mib:
        raise MemoryError(
            f"E005 dry-run RSS exceeded {limit_mib} MiB (current_rss_mib={current:.1f}, peak_rss_mib={peak:.1f})"
        )
    if device is not None and device.type == "cuda":
        allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)
        if allocated_mib > limit_mib:
            raise MemoryError(
                f"E005 dry-run CUDA allocation exceeded {limit_mib} MiB "
                f"(cuda_allocated_mib={allocated_mib:.1f}, current_rss_mib={current:.1f}, "
                f"peak_rss_mib={peak:.1f})"
            )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class MemoryStageReporter:
    """Atomically checkpoint dry-run stage and absolute-process memory."""

    def __init__(self, path: str | Path, *, max_memory_mib: int) -> None:
        self.path = Path(path)
        self.max_memory_mib = int(max_memory_mib)
        self.started = time.monotonic()
        self.stages: list[dict[str, Any]] = []
        self.payload: dict[str, Any] = {"status": "running", "stages": self.stages}

    def set_limit(self, max_memory_mib: int) -> None:
        self.max_memory_mib = int(max_memory_mib)

    def record(self, stage: str, **details: Any) -> None:
        current = _rss_mib()
        entry = {
            "stage": stage,
            "current_rss_mib": current,
            "peak_rss_mib": _peak_rss_mib(),
            "elapsed_seconds": time.monotonic() - self.started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **details,
        }
        self.stages.append(entry)
        self.payload.update(
            {
                "status": "running",
                "current_stage": stage,
                "current_rss_mib": current,
                "peak_rss_mib": entry["peak_rss_mib"],
                "stages": self.stages,
            }
        )
        _atomic_json(self.path, self.payload)
        _guard_memory(self.max_memory_mib)

    def incomplete(self, error: BaseException) -> None:
        self.payload.update(
            {
                "status": "incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "current_rss_mib": _rss_mib(),
                "peak_rss_mib": _peak_rss_mib(),
                "stages": self.stages,
            }
        )
        _atomic_json(self.path, self.payload)

    def completed(self, result: dict[str, Any]) -> None:
        result["stages"] = self.stages
        self.payload = result
        _atomic_json(self.path, self.payload)


def _configuration_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _parameter_groups(model: E005SequenceGeometryCoDesign) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    modules = {
        "sequence_branch": (
            model.token_embedding,
            model.position_embedding,
            model.sequence_encoder,
            model.sequence_norm,
            model.sequence_logits,
        ),
        "geometry_branch": (model.geometry_model,),
        "sequence_to_geometry_feedback": (model.sequence_to_pair,),
        "geometry_to_sequence_feedback": (
            model.geometry_to_sequence,
            model.return_geometry_to_sequence,
        ),
        "gates": (
            model.geometry_to_sequence_gate,
            model.sequence_to_pair_gate,
            model.return_geometry_gate,
        ),
    }
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    assigned: set[int] = set()
    for group_name, group_modules in modules.items():
        entries = []
        for module in group_modules:
            for parameter in module.parameters():
                if id(parameter) not in assigned:
                    entries.append((names_by_id[id(parameter)], parameter))
                    assigned.add(id(parameter))
        groups[group_name] = entries
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if assigned != trainable:
        raise RuntimeError("E005 diagnostic parameter groups do not cover every trainable parameter exactly once")
    return groups


def _gradient_norms(groups: dict[str, list[tuple[str, torch.nn.Parameter]]]) -> dict[str, float]:
    result = {}
    for group_name, parameters in groups.items():
        missing = [name for name, parameter in parameters if parameter.grad is None]
        if missing:
            raise RuntimeError(f"missing_gradients:{group_name}:{','.join(missing[:10])}")
        squared = sum(float(parameter.grad.detach().float().square().sum()) for _, parameter in parameters)
        norm = squared**0.5
        if not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"invalid_gradient_norm:{group_name}:{norm}")
        result[group_name] = norm
    return result


def _parameter_snapshots(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        group_name: {name: parameter.detach().cpu().clone() for name, parameter in parameters}
        for group_name, parameters in groups.items()
    }


def _parameter_change_norms(
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
    snapshots: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    changes = {}
    for group_name, parameters in groups.items():
        squared = 0.0
        for name, parameter in parameters:
            difference = parameter.detach().cpu().float() - snapshots[group_name][name].float()
            squared += float(difference.square().sum())
        changes[group_name] = squared**0.5
    return changes


def _gate_statistics(
    geometry_gate: torch.Tensor,
    pair_gate: torch.Tensor,
    return_gate: torch.Tensor,
    residue_mask: torch.Tensor,
) -> dict[str, dict[str, float]]:
    statistics_by_gate = {}
    tensors = {
        "geometry_to_sequence": geometry_gate,
        "sequence_to_geometry": pair_gate,
        "return_geometry_to_sequence": return_gate,
    }
    for name, values in tensors.items():
        valid = residue_mask if values.ndim == 2 else residue_mask[..., None].expand_as(values)
        selected = values[valid].detach().float()
        if selected.numel() == 0 or not torch.isfinite(selected).all():
            raise RuntimeError(f"invalid_gate_values:{name}")
        statistics_by_gate[name] = {
            "minimum": float(selected.min()),
            "maximum": float(selected.max()),
            "mean": float(selected.mean()),
            "standard_deviation": float(selected.std(unbiased=False)),
            "saturated_fraction": float(((selected <= 0.01) | (selected >= 0.99)).float().mean()),
        }
    return statistics_by_gate


def _loss_reduction_summary(
    losses_by_step: list[dict[str, float]],
    *,
    required_relative_reduction: float,
) -> dict[str, dict[str, float | bool]]:
    window = max(1, len(losses_by_step) // 4)
    result: dict[str, dict[str, float | bool]] = {}
    for name in ("total", "sequence", "geometry"):
        initial = statistics.median(item[name] for item in losses_by_step[:window])
        final = statistics.median(item[name] for item in losses_by_step[-window:])
        reduction = (initial - final) / max(abs(initial), 1e-12)
        result[name] = {
            "initial_window_median": initial,
            "final_window_median": final,
            "relative_reduction": reduction,
            "passed": reduction >= required_relative_reduction,
        }
    return result


def conditioning_mask(
    batch_size: int,
    *,
    probability: float,
    seed: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw deterministic per-sample conditioning availability flags."""
    if not 0 <= probability <= 1:
        raise ValueError("conditioning_dropout_probability must be in [0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(step) * 1_000_003)
    keep = torch.rand(batch_size, generator=generator) >= probability
    return keep.to(device=device)


def masked_sequence_inputs(
    targets: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    mask_token_id: int,
    probability: float,
    seed: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask canonical tokens deterministically and retain at least one per sample."""
    if not 0 < probability <= 1:
        raise ValueError("masked_token_probability must be in (0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + int(step) * 2_000_003)
    draws = torch.rand(targets.shape, generator=generator, device="cpu").to(targets.device)
    masked = (draws < probability) & residue_mask.bool()
    for batch_index in range(targets.shape[0]):
        if not masked[batch_index].any():
            first_valid = torch.nonzero(residue_mask[batch_index], as_tuple=False)[0, 0]
            masked[batch_index, first_valid] = True
    inputs = targets.clone()
    inputs[masked] = int(mask_token_id)
    return inputs, masked


def _synthetic_items(lengths: list[int], seed: int) -> list[dict[str, Any]]:
    vocabulary = SequenceGeometryVocabulary()
    amino_acids = vocabulary.tokens[2:]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    items = []
    for index, length in enumerate(lengths):
        sequence = "".join(amino_acids[(index + position) % len(amino_acids)] for position in range(length))
        steps = torch.randn((length, 3), generator=generator)
        coordinates = torch.cumsum(steps / steps.norm(dim=1, keepdim=True).clamp_min(1e-6) * 3.8, dim=0)
        matrix = torch.cdist(coordinates, coordinates)
        items.append(
            {
                "sample_id": f"synthetic_{index:03d}",
                "sequence_token_ids": torch.tensor(vocabulary.encode(sequence), dtype=torch.long),
                "sequence_mask": torch.ones(length, dtype=torch.bool),
                "distance_matrix": matrix,
                "pair_mask": torch.ones((length, length), dtype=torch.bool),
                "length": length,
                "geometry_availability_flag": True,
                "geometry_conditioning_flag": True,
            }
        )
    return items


def _select_real_rows(
    dataset: ds.Dataset,
    *,
    sample_count: int,
    maximum_length: int,
    seed: int,
    batch_size: int = 4096,
) -> pa.Table:
    """Select a deterministic bounded sample without materializing the dataset."""
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
    selected: list[tuple[str, dict[str, Any]]] = []
    scanner = dataset.scanner(
        columns=columns,
        batch_size=min(int(batch_size), 4096),
        use_threads=False,
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            length = int(row["sequence_length"])
            if length > maximum_length:
                continue
            rank = hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest()
            selected.append((rank, row))
            selected.sort(key=lambda item: item[0])
            del selected[sample_count:]
    if len(selected) < sample_count:
        raise ValueError(f"Only {len(selected)} eligible rows have length <= {maximum_length}")
    return pa.Table.from_pylist([row for _, row in selected])


def _real_items(
    config: dict[str, Any],
    sample_count: int,
    maximum_length: int,
    reporter: MemoryStageReporter,
) -> tuple[list[dict[str, Any]], list[Path], dict[str, str]]:
    dataset_dir = Path(config["dataset"]["directory"])
    protocol = json.loads((dataset_dir / "protocol.json").read_text())
    if protocol.get("status") != "completed" or protocol.get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise ValueError("The immutable pairing dataset protocol is incomplete or incompatible")
    manifest = dataset_dir / str(config["dataset"].get("train_dataset", "eligible_train.parquet"))
    dataset = ds.dataset(str(manifest), format="parquet")
    reporter.record(
        "dataset_construction",
        manifest_path=str(manifest),
        manifest_columns=len(dataset.schema),
    )
    selected = _select_real_rows(
        dataset,
        sample_count=sample_count,
        maximum_length=maximum_length,
        seed=int(config["seed"]),
    )
    del dataset
    reporter.record("sample_selection", selected_rows=selected.num_rows)
    rows = selected.to_pylist()
    protected_paths = [dataset_dir / "protocol.json"]
    protected_paths.extend(Path(str(row["matrix_path"])) for row in rows)
    protected_paths.extend(Path(str(row["__filename"])) for row in rows)
    protected_paths = sorted(set(protected_paths), key=str)
    input_hashes_before = {str(path): _sha256_file(path) for path in protected_paths}
    selected_dataset = SequenceGeometryDataset(selected, mode="geometry_conditioned")
    items = [selected_dataset[index] for index in range(len(selected_dataset))]
    if {str(path): _sha256_file(path) for path in protected_paths} != input_hashes_before:
        raise RuntimeError("dataset_mutation_detected_during_npz_loading")
    reporter.record("npz_loading", loaded_npz_count=len(items))
    return items, protected_paths, input_hashes_before


def _validate_limits(
    *,
    steps: int,
    sample_count: int,
    maximum_length: int,
    max_memory_mib: int,
    tiny_overfit: bool,
) -> None:
    maximum_steps = MAX_TINY_OVERFIT_STEPS if tiny_overfit else MAX_DRY_RUN_STEPS
    if not 1 <= steps <= maximum_steps:
        raise ValueError(f"steps must be in [1, {maximum_steps}]")
    if tiny_overfit and steps < 4:
        raise ValueError("tiny-overfit requires at least 4 steps")
    if not 1 <= sample_count <= MAX_DRY_RUN_SAMPLES:
        raise ValueError(f"sample_count must be in [1, {MAX_DRY_RUN_SAMPLES}]")
    if not 2 <= maximum_length <= MAX_DRY_RUN_LENGTH:
        raise ValueError(f"maximum_length must be in [2, {MAX_DRY_RUN_LENGTH}]")
    if tiny_overfit and maximum_length > MAX_TINY_OVERFIT_LENGTH:
        raise ValueError(f"tiny-overfit maximum_length must be <= {MAX_TINY_OVERFIT_LENGTH}")
    if tiny_overfit and sample_count != 1:
        raise ValueError("tiny-overfit requires exactly one sample")
    if not 128 <= max_memory_mib <= MAX_DRY_RUN_MEMORY_MIB:
        raise ValueError(f"max_memory_mib must be in [128, {MAX_DRY_RUN_MEMORY_MIB}]")


def run_codesign_dry_run(
    config: dict[str, Any],
    *,
    report_path: str | Path,
    real_data: bool = False,
    steps: int | None = None,
    sample_count: int | None = None,
    maximum_length: int | None = None,
    max_memory_mib: int | None = None,
    checkpoint_path: str | Path | None = None,
    reporter: MemoryStageReporter | None = None,
    tiny_overfit: bool = False,
) -> dict[str, Any]:
    """Run a strictly bounded synthetic or one-batch real-data integration."""
    limits = config.get("tiny_overfit" if tiny_overfit else "dry_run", {})
    steps = int(steps if steps is not None else limits.get("steps", 4))
    sample_count = int(sample_count if sample_count is not None else limits.get("sample_count", 2))
    maximum_length = int(maximum_length if maximum_length is not None else limits.get("maximum_length", 64))
    max_memory_mib = int(max_memory_mib if max_memory_mib is not None else limits.get("max_memory_mib", 2048))
    _validate_limits(
        steps=steps,
        sample_count=sample_count,
        maximum_length=maximum_length,
        max_memory_mib=max_memory_mib,
        tiny_overfit=tiny_overfit,
    )
    if reporter is None:
        reporter = MemoryStageReporter(report_path, max_memory_mib=max_memory_mib)
        reporter.record("imports_startup")
        reporter.record("configuration_load")
    else:
        reporter.set_limit(max_memory_mib)
    checkpoint_destination = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint_destination is not None and checkpoint_destination.exists():
        error = FileExistsError(f"Refusing to overwrite dry-run checkpoint: {checkpoint_destination}")
        reporter.incomplete(error)
        raise error
    try:
        return _run_codesign_dry_run(
            config,
            report_path=report_path,
            real_data=real_data,
            steps=steps,
            sample_count=sample_count,
            maximum_length=maximum_length,
            max_memory_mib=max_memory_mib,
            checkpoint_path=checkpoint_path,
            reporter=reporter,
            tiny_overfit=tiny_overfit,
        )
    except BaseException as error:
        if checkpoint_path is not None:
            pending_checkpoint = Path(checkpoint_path).with_name(f".{Path(checkpoint_path).name}.pending")
            pending_checkpoint.unlink(missing_ok=True)
            Path(checkpoint_path).unlink(missing_ok=True)
        reporter.incomplete(error)
        raise


def _run_codesign_dry_run(
    config: dict[str, Any],
    *,
    report_path: str | Path,
    real_data: bool,
    steps: int,
    sample_count: int,
    maximum_length: int,
    max_memory_mib: int,
    checkpoint_path: str | Path | None,
    reporter: MemoryStageReporter,
    tiny_overfit: bool,
) -> dict[str, Any]:
    """Implementation body with failure publication managed by the wrapper."""
    del report_path
    limits = config.get("tiny_overfit" if tiny_overfit else "dry_run", {})
    mode = (
        "learned_geometry_gating"
        if tiny_overfit
        else str(config.get("conditioning", {}).get("mode", "learned_geometry_gating"))
    )
    if mode not in CONDITIONING_MODES:
        raise ValueError(f"Unsupported E005 conditioning mode: {mode}")
    seed = int(config.get("seed", 5005))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device_name = str(config.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E005 config requests CUDA but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if real_data:
        items, protected_input_paths, input_hashes_before = _real_items(
            config,
            sample_count,
            maximum_length,
            reporter,
        )
        input_kind = "immutable_sequence_geometry_pairing_v1"
    else:
        reporter.record("dataset_construction", input_kind="synthetic")
        configured_lengths = [int(value) for value in limits.get("synthetic_lengths", [24, 37])]
        lengths = configured_lengths[:sample_count]
        while len(lengths) < sample_count:
            lengths.append(configured_lengths[len(lengths) % len(configured_lengths)])
        if max(lengths) > maximum_length:
            raise ValueError("A configured synthetic length exceeds maximum_length")
        items = _synthetic_items(lengths, seed)
        reporter.record("sample_selection", selected_rows=len(items))
        reporter.record("npz_loading", loaded_npz_count=0)
        input_kind = "synthetic"
        protected_input_paths = []
        input_hashes_before = {}
    model_config = dict(config["model"])
    geometry_config = dict(model_config.pop("geometry_model"))
    model = E005SequenceGeometryCoDesign(geometry_model=geometry_config, **model_config).to(device)
    reporter.record(
        "model_construction",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    _guard_memory(max_memory_mib, device)
    batch = collate_sequence_geometry(
        items,
        pad_id=model.pad_token_id,
        pad_to_multiple=model.downsample_factor,
    )
    _guard_memory(max_memory_mib, device)
    biological_pair_mask = batch["sequence_mask"][:, None, :, None] & batch["sequence_mask"][:, None, None, :]
    physical_geometry = batch["distance_matrices"][:, None].clone()
    corruption_config = limits.get("geometry_corruption") if tiny_overfit else None
    if tiny_overfit:
        corruption = build_geometry_corruption(corruption_config)
        if corruption is not None:
            for index in range(sample_count):
                generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 5005)) + 4_000_037)
                length = int(batch["lengths"][index])
                matrix, corrupted_mask = corruption(
                    physical_geometry[index, 0, :length, :length].clone(),
                    biological_pair_mask[index, 0, :length, :length].clone(),
                    generator,
                )
                if not corrupted_mask.all():
                    raise ValueError("Tiny-overfit corruption must preserve every biological pair")
                physical_geometry[index, 0, :length, :length] = matrix
    clean = physical_geometry.to(device)
    pair_mask = biological_pair_mask.to(device)
    normalization_path_value = config.get("normalization_file")
    normalization_sha256 = None
    if real_data:
        if not normalization_path_value:
            raise ValueError("Real-data integration requires normalization_file")
        normalization_path = Path(str(normalization_path_value))
        normalization = json.loads(normalization_path.read_text())
        if normalization.get("mode") != "scale" or float(normalization.get("scale", 0)) <= 0:
            raise ValueError("E005 requires positive scale normalization metadata")
        normalization_scale = float(normalization["scale"])
        normalization_sha256 = _sha256_file(normalization_path)
        protected_input_paths.append(normalization_path)
        input_hashes_before[str(normalization_path)] = normalization_sha256
    else:
        normalization_scale = float(config.get("normalization_scale_angstrom", 50.0))
    clean = clean / normalization_scale
    lengths = batch["lengths"].to(device)
    residue_mask = batch["sequence_mask"].to(device)
    token_targets = batch["sequence_token_ids"].to(device)
    separation = make_sequence_separation(lengths, clean.shape[-1]).to(device)
    diffusion_steps = int(config.get("diffusion", {}).get("steps", 500))
    prediction_type = str(config.get("diffusion", {}).get("prediction_parameterization", "v"))
    diffusion = GaussianDiffusion(cosine_beta_schedule(diffusion_steps)).to(device)
    loss_config = config.get("loss", {})
    weights = CoDesignLossWeights(
        sequence=float(loss_config.get("sequence_weight", 1.0)),
        geometry=float(loss_config.get("geometry_weight", 1.0)),
        consistency=float(loss_config.get("consistency_weight", 0.0)),
    )
    if tiny_overfit and not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Tiny-overfit requires every E005 parameter to be trainable")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
    repeat_fixed_batch = True if tiny_overfit else bool(limits.get("repeat_fixed_batch", False))
    activation_checkpointing = bool(limits.get("activation_checkpointing", False))
    mixed_precision = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    amp_dtype_name = str(config.get("amp_dtype", "float16"))
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp_dtype_name)
    if amp_dtype is None:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    parameter_groups = _parameter_groups(model) if tiny_overfit else {}
    initial_parameters = _parameter_snapshots(parameter_groups) if tiny_overfit else {}
    gradient_norms_by_step: list[dict[str, float]] = []
    gate_statistics_by_step: list[dict[str, dict[str, float]]] = []
    condition_fingerprints_by_step: list[dict[str, Any]] = []
    start = time.monotonic()
    losses_by_step = []
    output_shapes: dict[str, list[int]] = {}
    for step in range(steps):
        _guard_memory(max_memory_mib, device)
        stochastic_step = 0 if repeat_fixed_batch else step
        generator = torch.Generator(device=device).manual_seed(seed + stochastic_step * 3_000_017)
        timesteps = torch.randint(0, diffusion_steps, (sample_count,), generator=generator, device=device)
        noisy, epsilon = diffusion.q_sample(clean, timesteps, pair_mask, generator=generator)
        token_inputs, masked_tokens = masked_sequence_inputs(
            token_targets,
            residue_mask,
            mask_token_id=model.mask_token_id,
            probability=float(config.get("masked_token_probability", 0.15)),
            seed=seed,
            step=stochastic_step,
        )
        availability = conditioning_mask(
            sample_count,
            probability=(
                0.0 if tiny_overfit else float(config.get("conditioning", {}).get("dropout_probability", 0.0))
            ),
            seed=seed,
            step=stochastic_step,
            device=device,
        )
        if tiny_overfit:
            condition_fingerprints_by_step.append(
                {
                    "masked_sequence_sha256": _tensor_sha256(token_inputs),
                    "masked_token_mask_sha256": _tensor_sha256(masked_tokens),
                    "diffusion_timesteps": timesteps.detach().cpu().tolist(),
                    "geometry_noise_sha256": _tensor_sha256(epsilon),
                    "noisy_geometry_sha256": _tensor_sha256(noisy),
                    "corrupted_geometry_sha256": _tensor_sha256(clean),
                }
            )
            torch.manual_seed(seed + 5_000_041)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 5_000_041)

        def model_tensors(
            noisy_input: torch.Tensor,
            sequence_input: torch.Tensor,
            timestep_input: torch.Tensor,
            availability_input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            result = model(
                sequence_token_ids=sequence_input,
                residue_mask=residue_mask,
                noisy_geometry=noisy_input,
                timesteps=timestep_input,
                lengths=lengths,
                sequence_separation=separation,
                pair_mask=pair_mask,
                geometry_conditioning_mask=availability_input,
                mode=mode,
            )
            return (
                result["sequence_logits"],
                result["geometry_prediction"],
                result["sequence_pair_prediction"],
                result["geometry_to_sequence_gate"],
                result["sequence_to_geometry_gate"],
                result["return_geometry_gate"],
            )

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=mixed_precision):
            if activation_checkpointing:
                (
                    sequence_logits,
                    geometry_prediction,
                    sequence_pair_prediction,
                    geometry_gate,
                    pair_gate,
                    return_gate,
                ) = checkpoint(
                    model_tensors,
                    noisy,
                    token_inputs,
                    timesteps,
                    availability,
                    use_reentrant=False,
                )
            else:
                (
                    sequence_logits,
                    geometry_prediction,
                    sequence_pair_prediction,
                    geometry_gate,
                    pair_gate,
                    return_gate,
                ) = model_tensors(noisy, token_inputs, timesteps, availability)
            outputs = {
                "sequence_logits": sequence_logits,
                "geometry_prediction": geometry_prediction,
                "sequence_pair_prediction": sequence_pair_prediction,
                "residue_mask": residue_mask,
                "pair_mask": pair_mask,
            }
            target = diffusion.training_target(
                x_start=clean,
                t=timesteps,
                epsilon=epsilon,
                prediction_type=prediction_type,
            )
            losses = codesign_losses(
                outputs,
                sequence_targets=token_targets,
                masked_token_mask=masked_tokens,
                geometry_target=target,
                weights=weights,
            )
        loss_values = {name: float(value.detach()) for name, value in losses.items()}
        if not all(np.isfinite(value) for value in loss_values.values()):
            raise RuntimeError(f"nonfinite_loss:{loss_values}")
        if tiny_overfit:
            gate_stats = _gate_statistics(geometry_gate, pair_gate, return_gate, residue_mask)
            saturation_limit = float(limits.get("maximum_gate_saturated_fraction", 0.95))
            saturated = [
                name
                for name, values in gate_stats.items()
                if values["saturated_fraction"] >= saturation_limit or values["mean"] <= 0.01 or values["mean"] >= 0.99
            ]
            if saturated:
                raise RuntimeError(f"saturated_gates:{','.join(saturated)}")
            gate_statistics_by_step.append(gate_stats)
        reporter.record("forward", step=step + 1)
        _guard_memory(max_memory_mib, device)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        if tiny_overfit:
            scaler.unscale_(optimizer)
            gradient_norms_by_step.append(_gradient_norms(parameter_groups))
        reporter.record("backward", step=step + 1)
        _guard_memory(max_memory_mib, device)
        scaler.step(optimizer)
        scaler.update()
        reporter.record("optimizer_step", step=step + 1)
        _guard_memory(max_memory_mib, device)
        losses_by_step.append(loss_values)
        output_shapes = {
            "sequence_logits": list(outputs["sequence_logits"].shape),
            "geometry_prediction": list(outputs["geometry_prediction"].shape),
            "residue_mask": list(residue_mask.shape),
            "pair_mask": list(pair_mask.shape),
        }
    elapsed = time.monotonic() - start
    _guard_memory(max_memory_mib, device)
    input_hashes_after = {path: _sha256_file(Path(path)) for path in input_hashes_before}
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("dataset_mutation_detected")
    reporter.record(
        "dataset_integrity_verification",
        protected_input_count=len(input_hashes_before),
        unchanged=True,
    )
    parameter_change_norms = _parameter_change_norms(parameter_groups, initial_parameters) if tiny_overfit else {}
    unchanged_groups = [name for name, value in parameter_change_norms.items() if not np.isfinite(value) or value <= 0]
    if unchanged_groups:
        raise RuntimeError(f"unchanged_trainable_branches:{','.join(unchanged_groups)}")
    required_reduction = float(limits.get("required_relative_loss_reduction", 0.05))
    if not 0 < required_reduction < 1:
        raise ValueError("required_relative_loss_reduction must be in (0, 1)")
    reduction_summary = (
        _loss_reduction_summary(losses_by_step, required_relative_reduction=required_reduction) if tiny_overfit else {}
    )
    failed_reductions = [name for name, values in reduction_summary.items() if not values["passed"]]
    if failed_reductions:
        raise RuntimeError(f"insufficient_tiny_overfit_loss_reduction:{','.join(failed_reductions)}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "status": "completed",
        "architecture_version": E005_ARCHITECTURE_VERSION,
        "pairing_schema_version": PAIRING_SCHEMA_VERSION,
        "input_kind": input_kind,
        "configuration_sha256": _configuration_sha256(config),
        "configuration": config,
        "dataset_directory": str(config.get("dataset", {}).get("directory", "")) or None,
        "dataset_protocol_sha256": (
            _sha256_file(Path(config["dataset"]["directory"]) / "protocol.json") if real_data else None
        ),
        "normalization_file": str(normalization_path_value) if normalization_path_value else None,
        "normalization_sha256": normalization_sha256,
        "normalization_scale_angstrom": normalization_scale,
        "seed": seed,
        "device": str(device),
        "conditioning_mode": mode,
        "conditioning_dropout_probability": (
            0.0 if tiny_overfit else float(config.get("conditioning", {}).get("dropout_probability", 0.0))
        ),
        "tiny_overfit": tiny_overfit,
        "steps": steps,
        "repeat_fixed_batch": repeat_fixed_batch,
        "activation_checkpointing": activation_checkpointing,
        "mixed_precision": mixed_precision,
        "sample_count": sample_count,
        "sample_ids": batch["sample_ids"],
        "lengths": batch["lengths"].tolist(),
        "padded_side": int(clean.shape[-1]),
        "tensor_shapes": output_shapes,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "geometry_parameter_count": sum(parameter.numel() for parameter in model.geometry_model.parameters()),
        "losses_by_step": losses_by_step,
        "gradient_norms_by_step": gradient_norms_by_step,
        "gate_statistics_by_step": gate_statistics_by_step,
        "condition_fingerprints_by_step": condition_fingerprints_by_step,
        "parameter_change_norms": parameter_change_norms,
        "loss_reduction_summary": reduction_summary,
        "fixed_conditions": (
            {
                "sample_reused_every_step": True,
                "masked_token_positions": torch.nonzero(masked_tokens, as_tuple=False).cpu().tolist(),
                "diffusion_timesteps": timesteps.detach().cpu().tolist(),
                "geometry_noise_sha256": _tensor_sha256(epsilon),
                "corrupted_geometry_sha256": _tensor_sha256(clean),
                "geometry_corruption": corruption_config,
                "model_dropout_seed": seed + 5_000_041,
            }
            if tiny_overfit
            else None
        ),
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "dataset_inputs_unchanged": input_hashes_before == input_hashes_after,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_count * steps / max(elapsed, 1e-12),
        "peak_rss_mib": _peak_rss_mib(),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        ),
        "limits": {
            "maximum_steps": MAX_TINY_OVERFIT_STEPS if tiny_overfit else MAX_DRY_RUN_STEPS,
            "maximum_samples": MAX_DRY_RUN_SAMPLES,
            "maximum_length": MAX_DRY_RUN_LENGTH,
            "maximum_memory_mib": max_memory_mib,
        },
    }
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        pending_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.pending")
        save_checkpoint(
            pending_checkpoint,
            {
                "architecture_version": E005_ARCHITECTURE_VERSION,
                "config": config,
                "configuration_sha256": report["configuration_sha256"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "steps": steps,
                "seed": seed,
                "dry_run": True,
                "tiny_overfit": tiny_overfit,
                "diagnostics": {
                    "conditioning_mode": mode,
                    "conditioning_dropout_probability": (
                        0.0 if tiny_overfit else float(config.get("conditioning", {}).get("dropout_probability", 0.0))
                    ),
                    "loss_reduction_summary": reduction_summary,
                    "gradient_norms_by_step": gradient_norms_by_step,
                    "gate_statistics_by_step": gate_statistics_by_step,
                    "condition_fingerprints_by_step": condition_fingerprints_by_step,
                    "parameter_change_norms": parameter_change_norms,
                    "dataset_inputs_unchanged": input_hashes_before == input_hashes_after,
                    "input_hashes_before": input_hashes_before,
                    "input_hashes_after": input_hashes_after,
                },
                "scaler_state_dict": scaler.state_dict(),
            },
        )
        reporter.record("checkpoint_writing", pending_path=str(pending_checkpoint))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        pending_checkpoint.replace(checkpoint_path)
        report["checkpoint_path"] = str(checkpoint_path)
    else:
        reporter.record("checkpoint_writing", skipped=True)
    reporter.completed(report)
    return report
