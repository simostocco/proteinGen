"""Read-only E006 Stage-A contextual-learning diagnostic."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.data.rich_geometry import (
    RichDatasetAuthorization,
    authorize_rich_geometry_dataset,
)
from protein_distance_diffusion.data.rich_geometry_sidecars import rich_geometry_schema
from protein_distance_diffusion.models.rich_codesign import E006_ARCHITECTURE_VERSION
from protein_distance_diffusion.training.checkpointing import load_checkpoint
from protein_distance_diffusion.training.codesign import masked_sequence_inputs
from protein_distance_diffusion.training.rich_codesign_production import (
    _dataset_identity,
    _model,
    _protected_hashes,
    collate_sequence_pretraining,
    verify_stage_a_checkpoint,
)
from protein_distance_diffusion.training.rich_codesign_smoke import _memory, _sha256

DIAGNOSTIC_VERSION = "e006_stage_a_context_diagnostic_v1"
SAMPLER_VERSION = "e006_context_disjoint_length_stratified_v2"
CANONICAL_TOKEN_START = 2
CANONICAL_TOKEN_COUNT = 20
CONDITIONS = ("normal", "visible_shuffle", "permuted_conditioning", "null_conditioning")
SUBSETS = ("all_valid", "corrupted", "visible")
LENGTH_BOUNDARIES = (64, 128, 256, 384, 500)
LENGTH_BUCKETS = ((1, 64), (65, 128), (129, 256), (257, 384), (385, 500))
SEQUENCE_ONLY_CAPABILITIES = {
    "identity_and_split": ("sample_id", "split"),
    "sequence_and_mask_derivation": ("sequence", "token_ids"),
    "immutable_provenance": (
        "schema_version",
        "source_path",
        "source_sha256",
        "npz_path",
        "npz_sha256",
        "mapping_evidence_sha256",
    ),
}
REQUIRED_SEQUENCE_COLUMNS = tuple(column for columns in SEQUENCE_ONLY_CAPABILITIES.values() for column in columns)


@dataclass(frozen=True)
class _Locator:
    path: Path
    row_group: int
    row_count: int


@dataclass(frozen=True)
class PanelMember:
    index: int
    sample_id: str
    length: int
    length_bucket: int


class SequenceOnlyRichDataset(Sequence[dict[str, Any]]):
    """Projected split reader that cannot materialize rich geometry columns."""

    columns = REQUIRED_SEQUENCE_COLUMNS

    def __init__(self, authorization: RichDatasetAuthorization, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown sequence-only split: {split}")
        self.authorization = authorization
        self.split = split
        protocol = json.loads((authorization.root / "protocol.json").read_text())
        published_schema = json.loads((authorization.root / "schema.json").read_text())
        published_column_types = published_schema.get("columns", {})
        published_columns = tuple(published_column_types)
        validate_sequence_schema_columns(published_columns, source="published schema.json")
        self.available_schema_columns = tuple(sorted(published_columns))
        self.required_projected_columns = self.columns
        canonical_schema = rich_geometry_schema()
        self._locators: list[_Locator] = []
        self._ends: list[int] = []
        total = 0
        for record in protocol["shards"]:
            if record["dataset"] != split:
                continue
            path = authorization.root / record["path"]
            parquet = pq.ParquetFile(path)
            names = tuple(parquet.schema_arrow.names)
            validate_sequence_schema_columns(names, source=f"Parquet shard {record['path']}")
            if set(names) != set(published_columns):
                raise ValueError(f"E006 Parquet shard schema contradicts published schema: {record['path']}")
            type_contradictions = [
                name
                for name in self.columns
                if not parquet.schema_arrow.field(name).type.equals(canonical_schema.field(name).type)
            ]
            if type_contradictions:
                raise ValueError(
                    f"E006 Parquet shard required-column types contradict published schema: "
                    f"{record['path']}: {type_contradictions}"
                )
            for row_group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(row_group).num_rows
                self._locators.append(_Locator(path, row_group, count))
                total += count
                self._ends.append(total)
        if total != authorization.split_counts[split]:
            raise ValueError(f"E006 sequence-only {split} count contradicts authorization")
        self._cached_locator: int | None = None
        self._cached_rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("E006 sequence-only indices must be scalar integers")
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        locator_index = bisect.bisect_right(self._ends, index)
        start = 0 if locator_index == 0 else self._ends[locator_index - 1]
        if self._cached_locator != locator_index:
            locator = self._locators[locator_index]
            table = pq.ParquetFile(locator.path).read_row_group(locator.row_group, columns=list(self.columns))
            self._cached_rows = table.to_pylist()
            self._cached_locator = locator_index
        row = dict(self._cached_rows[index - start])
        self._validate_row(row)
        return row

    def _validate_row(self, row: dict[str, Any]) -> None:
        if str(row.get("split")) != self.split:
            raise ValueError("E006 sequence-only row contradicts physical split ownership")
        sequence = str(row.get("sequence") or "")
        tokens = [int(value) for value in row.get("token_ids") or []]
        if not sequence or len(sequence) != len(tokens):
            raise ValueError(f"E006 sequence/token contradiction: {row.get('sample_id')}")
        if any(not CANONICAL_TOKEN_START <= value < CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT for value in tokens):
            raise ValueError(f"E006 noncanonical prediction target: {row.get('sample_id')}")
        for name in SEQUENCE_ONLY_CAPABILITIES["immutable_provenance"]:
            if row.get(name) in (None, ""):
                raise ValueError(f"E006 sequence-only row lacks required provenance {name}: {row.get('sample_id')}")

    def iter_metadata(self) -> Iterator[tuple[int, str, int]]:
        offset = 0
        for locator in self._locators:
            values = (
                pq.ParquetFile(locator.path)
                .read_row_group(locator.row_group, columns=["sample_id", "sequence"])
                .to_pydict()
            )
            for local, (sample_id, sequence) in enumerate(zip(values["sample_id"], values["sequence"], strict=True)):
                yield offset + local, str(sample_id), len(str(sequence))
            offset += locator.row_count

    def iter_token_ids(self) -> Iterator[tuple[int, str, list[int]]]:
        offset = 0
        for locator in self._locators:
            values = (
                pq.ParquetFile(locator.path)
                .read_row_group(locator.row_group, columns=["sample_id", "token_ids"])
                .to_pydict()
            )
            for local, (sample_id, token_ids) in enumerate(zip(values["sample_id"], values["token_ids"], strict=True)):
                yield offset + local, str(sample_id), [int(value) for value in token_ids]
            offset += locator.row_count


def validate_sequence_schema_columns(columns: Iterable[str], *, source: str) -> dict[str, Any]:
    """Validate the minimal published capabilities needed by sequence-only evaluation."""
    available = tuple(sorted(str(value) for value in columns))
    missing = sorted(set(REQUIRED_SEQUENCE_COLUMNS) - set(available))
    if missing:
        raise ValueError(f"E006 sequence-only {source} lacks required columns: {missing}")
    return {
        "available_schema_columns": list(available),
        "required_projected_columns": list(REQUIRED_SEQUENCE_COLUMNS),
        "missing_required_columns": [],
        "capabilities": {name: list(values) for name, values in SEQUENCE_ONLY_CAPABILITIES.items()},
        "geometry_columns_projected": [],
        "constructs_rich_pair_features": False,
        "feature_complexity": "O(N)",
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _authorization(config: dict[str, Any]) -> RichDatasetAuthorization:
    dataset = config["dataset"]
    return authorize_rich_geometry_dataset(
        dataset["directory"],
        expected_protocol_sha256=dataset["protocol_sha256"],
        expected_schema_sha256=dataset["schema_sha256"],
        expected_vocabulary_sha256=dataset["vocabulary_sha256"],
        expected_normalization_sha256=dataset["normalization_sha256"],
        expected_shard_inventory_sha256=dataset["shard_inventory_sha256"],
    )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("version") != DIAGNOSTIC_VERSION:
        raise ValueError("E006 contextual diagnostic version contradiction")
    if config.get("architecture_version") != E006_ARCHITECTURE_VERSION:
        raise ValueError("E006 contextual diagnostic architecture contradiction")
    panels = config.get("panels", {})
    if int(panels.get("original_size", 0)) != 256:
        raise ValueError("E006 original validation panel must contain 256 samples")
    if int(panels.get("independent_size", 0)) < 2048:
        raise ValueError("E006 independent contextual panel must contain at least 2048 samples")
    if int(panels.get("maximum_length", 0)) != LENGTH_BOUNDARIES[-1]:
        raise ValueError("E006 contextual panel maximum_length must be 500")
    minimum = int(panels.get("minimum_per_nonempty_bucket", 0))
    if minimum < 64:
        raise ValueError("E006 contextual panel requires at least 64 samples per nonempty length bucket")
    expected_original = str(panels.get("expected_original_sample_id_sha256", ""))
    if len(expected_original) != 64:
        raise ValueError("E006 original-panel expected SHA-256 is required")
    fractions = [float(value) for value in config.get("corruption", {}).get("mask_fractions", [])]
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("E006 contextual mask fractions must be in (0, 1]")
    if float(config.get("baselines", {}).get("smoothing", 0)) <= 0:
        raise ValueError("E006 unigram smoothing must be positive")
    if int(config.get("bootstrap", {}).get("iterations", 0)) < 100:
        raise ValueError("E006 contextual bootstrap requires at least 100 iterations")
    gate = config.get("interpretation_gate", {})
    for name in ("unigram_improvement_nats", "context_degradation_nats"):
        if float(gate.get(name, -1)) < 0:
            raise ValueError(f"E006 invalid interpretation threshold: {name}")
    if float(config.get("memory", {}).get("maximum_rss_mib", 0)) <= 0:
        raise ValueError("E006 contextual diagnostic RSS limit must be positive")


def _bucket(length: int, *, maximum_length: int = LENGTH_BOUNDARIES[-1]) -> int:
    length = int(length)
    if length <= 0 or length > maximum_length:
        raise ValueError(f"E006 invalid contextual-panel sequence length: {length}")
    for lower, upper in LENGTH_BUCKETS:
        if lower <= length <= upper:
            return upper
    raise ValueError(f"E006 sequence length is outside the declared bucket convention: {length}")


def _bucket_label(boundary: int) -> str:
    for lower, upper in LENGTH_BUCKETS:
        if boundary == upper:
            return f"{lower}-{upper}"
    raise ValueError(f"E006 unknown contextual-panel length boundary: {boundary}")


def _allocate_independent_quotas(
    available: dict[int, int],
    *,
    count: int,
    minimum_per_nonempty_bucket: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    total = sum(available.values())
    if total < count:
        raise ValueError(
            "E006 independent panel filtered population is insufficient: "
            f"requested={count}, available={total}, counts={available}"
        )
    mandatory = {
        boundary: min(available[boundary], minimum_per_nonempty_bucket) if available[boundary] else 0
        for boundary in LENGTH_BOUNDARIES
    }
    if sum(mandatory.values()) > count:
        raise ValueError(
            "E006 independent panel minimum quotas exceed the requested panel size: "
            f"requested={count}, mandatory={mandatory}"
        )
    quotas = dict(mandatory)
    remaining = count - sum(quotas.values())
    capacities = {boundary: available[boundary] - quotas[boundary] for boundary in LENGTH_BOUNDARIES}
    capacity_total = sum(capacities.values())
    exact_additions = {
        boundary: (remaining * capacities[boundary] / capacity_total if capacity_total else 0.0)
        for boundary in LENGTH_BOUNDARIES
    }
    floor_additions = {
        boundary: min(capacities[boundary], math.floor(exact_additions[boundary])) for boundary in LENGTH_BOUNDARIES
    }
    for boundary, addition in floor_additions.items():
        quotas[boundary] += addition
    seats = count - sum(quotas.values())
    remainder_order = sorted(
        LENGTH_BOUNDARIES,
        key=lambda boundary: (
            -(exact_additions[boundary] - floor_additions[boundary]),
            boundary,
        ),
    )
    for boundary in remainder_order:
        if not seats:
            break
        if quotas[boundary] < available[boundary]:
            quotas[boundary] += 1
            seats -= 1
    if seats:
        raise ValueError(f"E006 independent panel allocation left {seats} unassigned samples")
    events = [
        {
            "event": "minimum_quota_shortage",
            "length_bucket": _bucket_label(boundary),
            "available": available[boundary],
            "configured_minimum": minimum_per_nonempty_bucket,
            "included": quotas[boundary],
            "shortage": minimum_per_nonempty_bucket - available[boundary],
        }
        for boundary in LENGTH_BOUNDARIES
        if 0 < available[boundary] < minimum_per_nonempty_bucket
    ]
    return quotas, {
        "allocation_method": "minimum_then_largest_remainder_over_residual_capacity",
        "minimum_per_nonempty_bucket": minimum_per_nonempty_bucket,
        "proportional_ideal_allocation": {
            _bucket_label(boundary): count * available[boundary] / total for boundary in LENGTH_BOUNDARIES
        },
        "redistribution_events": events,
    }


def _ordered_original_panel(dataset: SequenceOnlyRichDataset, *, count: int, seed: int) -> list[PanelMember]:
    metadata = list(dataset.iter_metadata())
    per_bucket = math.ceil(count / len(LENGTH_BOUNDARIES))
    selected: list[tuple[str, int, str, int]] = []
    global_candidates: list[tuple[str, int, str, int]] = []
    for index, sample_id, length in metadata:
        rank = hashlib.sha256(f"{seed + 1}:validation:{sample_id}".encode()).hexdigest()
        candidate = (rank, index, sample_id, length)
        global_candidates.append(candidate)
    for boundary in LENGTH_BOUNDARIES:
        candidates = sorted(item for item in global_candidates if _bucket(item[3]) == boundary)
        selected.extend(candidates[:per_bucket])
    selected_ids = {item[1] for item in selected}
    selected.extend(item for item in sorted(global_candidates) if item[1] not in selected_ids)
    selected = sorted(selected)[:count]
    if len(selected) != count:
        raise ValueError(f"E006 original panel has {len(selected)} rows, expected {count}")
    ordered = sorted(
        (
            (
                _bucket(length),
                hashlib.sha256(f"{seed}:validation:{sample_id}".encode()).hexdigest(),
                index,
                sample_id,
                length,
            )
            for _, index, sample_id, length in selected
        )
    )
    return [PanelMember(index, sample_id, length, boundary) for boundary, _, index, sample_id, length in ordered]


def _independent_panel(
    dataset: SequenceOnlyRichDataset,
    *,
    count: int,
    seed: int,
    excluded_sample_ids: set[str],
    minimum_per_nonempty_bucket: int,
    maximum_length: int,
) -> tuple[list[PanelMember], dict[str, Any]]:
    candidates: list[tuple[int, str, int, str, int]] = []
    before_counts = {boundary: 0 for boundary in LENGTH_BOUNDARIES}
    after_counts = {boundary: 0 for boundary in LENGTH_BOUNDARIES}
    seen_sample_ids: set[str] = set()
    for index, sample_id, length in dataset.iter_metadata():
        boundary = _bucket(length, maximum_length=maximum_length)
        if sample_id in seen_sample_ids:
            raise ValueError(f"E006 validation candidate sample_id is duplicated: {sample_id}")
        seen_sample_ids.add(sample_id)
        before_counts[boundary] += 1
        if sample_id in excluded_sample_ids:
            continue
        after_counts[boundary] += 1
        rank = hashlib.sha256(f"{seed}:validation:{sample_id}".encode()).hexdigest()
        candidates.append((boundary, rank, index, sample_id, length))
    quotas, allocation = _allocate_independent_quotas(
        after_counts,
        count=count,
        minimum_per_nonempty_bucket=minimum_per_nonempty_bucket,
    )
    selected: list[tuple[int, str, int, str, int]] = []
    for boundary in LENGTH_BOUNDARIES:
        selected.extend(sorted(item for item in candidates if item[0] == boundary)[: quotas[boundary]])
    selected = sorted(selected, key=lambda value: (value[0], value[1], value[3]))
    if len(selected) != count or len({item[3] for item in selected}) != count:
        raise ValueError(f"E006 independent panel has {len(selected)} unique rows, expected {count}")
    if excluded_sample_ids.intersection(item[3] for item in selected):
        raise ValueError("E006 contextual panels overlap")
    members = [PanelMember(index, sample_id, length, boundary) for boundary, _, index, sample_id, length in selected]
    realized = {boundary: sum(item.length_bucket == boundary for item in members) for boundary in LENGTH_BOUNDARIES}
    after_total = sum(after_counts.values())
    diagnostics = {
        "bucket_boundary_convention": [
            {"label": f"{lower}-{upper}", "minimum_inclusive": lower, "maximum_inclusive": upper}
            for lower, upper in LENGTH_BUCKETS
        ],
        "maximum_length": maximum_length,
        "candidate_population_before_original_panel_exclusion": {
            "total": sum(before_counts.values()),
            "counts_by_length_bucket": {
                _bucket_label(boundary): before_counts[boundary] for boundary in LENGTH_BOUNDARIES
            },
        },
        "candidate_population_after_original_panel_exclusion": {
            "total": after_total,
            "counts_by_length_bucket": {
                _bucket_label(boundary): after_counts[boundary] for boundary in LENGTH_BOUNDARIES
            },
        },
        "requested_allocation_by_length_bucket": {
            _bucket_label(boundary): quotas[boundary] for boundary in LENGTH_BOUNDARIES
        },
        "realized_allocation_by_length_bucket": {
            _bucket_label(boundary): realized[boundary] for boundary in LENGTH_BOUNDARIES
        },
        "population_fraction_by_length_bucket": {
            _bucket_label(boundary): after_counts[boundary] / after_total for boundary in LENGTH_BOUNDARIES
        },
        "panel_fraction_by_length_bucket": {
            _bucket_label(boundary): realized[boundary] / count for boundary in LENGTH_BOUNDARIES
        },
        "absolute_fraction_deviation_by_length_bucket": {
            _bucket_label(boundary): abs(realized[boundary] / count - after_counts[boundary] / after_total)
            for boundary in LENGTH_BOUNDARIES
        },
        **allocation,
    }
    return members, diagnostics


def select_panels(
    dataset: SequenceOnlyRichDataset,
    *,
    original_size: int,
    independent_size: int,
    training_seed: int,
    diagnostic_seed: int,
    minimum_per_nonempty_bucket: int = 64,
    maximum_length: int = LENGTH_BOUNDARIES[-1],
    expected_original_sample_id_sha256: str | None = None,
) -> tuple[dict[str, list[PanelMember]], dict[str, Any]]:
    original = _ordered_original_panel(dataset, count=original_size, seed=training_seed)
    original_ids = {item.sample_id for item in original}
    original_hash = _canonical_hash([item.sample_id for item in original])
    if expected_original_sample_id_sha256 is not None and original_hash != expected_original_sample_id_sha256:
        raise ValueError("E006 original validation panel hash contradiction")
    independent, allocation_diagnostics = _independent_panel(
        dataset,
        count=independent_size,
        seed=diagnostic_seed,
        excluded_sample_ids=original_ids,
        minimum_per_nonempty_bucket=minimum_per_nonempty_bucket,
        maximum_length=maximum_length,
    )
    repeated, _ = _independent_panel(
        dataset,
        count=independent_size,
        seed=diagnostic_seed,
        excluded_sample_ids=original_ids,
        minimum_per_nonempty_bucket=minimum_per_nonempty_bucket,
        maximum_length=maximum_length,
    )
    deterministic_repeatability = [item.sample_id for item in independent] == [item.sample_id for item in repeated]
    panels = {"original_validation": original, "independent_diagnostic": independent}
    realized = {boundary: sum(item.length_bucket == boundary for item in independent) for boundary in LENGTH_BOUNDARIES}
    available = {
        boundary: allocation_diagnostics["candidate_population_after_original_panel_exclusion"][
            "counts_by_length_bucket"
        ][_bucket_label(boundary)]
        for boundary in LENGTH_BOUNDARIES
    }
    gate = {
        "exact_requested_sample_count": len(independent) == independent_size,
        "exact_unique_sample_count": len({item.sample_id for item in independent}) == independent_size,
        "disjoint_from_original_panel": not original_ids.intersection(item.sample_id for item in independent),
        "every_nonempty_bucket_represented": all(
            not available[value] or realized[value] for value in LENGTH_BOUNDARIES
        ),
        "minimum_quota_policy_satisfied": all(
            not available[value]
            or realized[value] >= minimum_per_nonempty_bucket
            or realized[value] == available[value]
            for value in LENGTH_BOUNDARIES
        ),
        "exact_allocation_sum": sum(realized.values()) == independent_size,
        "deterministic_repeatability": deterministic_repeatability,
    }
    gate["passed"] = all(gate.values())
    if not gate["passed"]:
        raise ValueError(f"E006 independent panel selection gate failed: {gate}")
    diagnostics = {
        "sampler_version": SAMPLER_VERSION,
        "training_seed": training_seed,
        "diagnostic_seed": diagnostic_seed,
        "panels_disjoint": True,
        "validation_split_only": True,
        "independent_panel_allocation": allocation_diagnostics,
        "plan_gate": gate,
        "panels": {
            name: {
                "sample_count": len(values),
                "unique_sample_count": len({item.sample_id for item in values}),
                "sample_id_sha256": _canonical_hash([item.sample_id for item in values]),
                "counts_by_length_bucket": {
                    str(boundary): sum(item.length_bucket == boundary for item in values)
                    for boundary in LENGTH_BOUNDARIES
                },
            }
            for name, values in panels.items()
        },
    }
    return panels, diagnostics


def training_unigram(
    dataset: SequenceOnlyRichDataset,
    *,
    smoothing: float,
) -> dict[str, Any]:
    counts = np.zeros(CANONICAL_TOKEN_COUNT, dtype=np.int64)
    bucket_counts = {boundary: np.zeros(CANONICAL_TOKEN_COUNT, dtype=np.int64) for boundary in LENGTH_BOUNDARIES}
    sample_count = 0
    for _, _, token_ids in dataset.iter_token_ids():
        values = np.asarray(token_ids, dtype=np.int64) - CANONICAL_TOKEN_START
        if (values < 0).any() or (values >= CANONICAL_TOKEN_COUNT).any():
            raise ValueError("E006 training unigram encountered a noncanonical token")
        counts += np.bincount(values, minlength=CANONICAL_TOKEN_COUNT)
        bucket_counts[_bucket(len(values))] += np.bincount(values, minlength=CANONICAL_TOKEN_COUNT)
        sample_count += 1

    def probabilities(values: np.ndarray) -> list[float]:
        smoothed = values.astype(np.float64) + smoothing
        return (smoothed / smoothed.sum()).tolist()

    return {
        "support_token_ids": list(range(CANONICAL_TOKEN_START, CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT)),
        "smoothing": smoothing,
        "training_sample_count": sample_count,
        "token_count": int(counts.sum()),
        "token_counts": counts.tolist(),
        "token_frequencies": probabilities(counts),
        "length_bucketed": {
            str(boundary): {
                "token_count": int(values.sum()),
                "token_counts": values.tolist(),
                "token_frequencies": probabilities(values),
            }
            for boundary, values in bucket_counts.items()
        },
    }


def _donor_map(panel: list[PanelMember]) -> dict[str, PanelMember]:
    groups: dict[int, list[PanelMember]] = defaultdict(list)
    for member in panel:
        groups[member.length_bucket].append(member)
    donors: dict[str, PanelMember] = {}
    for length_bucket, members in groups.items():
        ordered = sorted(members, key=lambda item: item.sample_id)
        if len(ordered) < 2:
            raise ValueError(f"E006 contextual panel lacks two compatible donors in length bucket {length_bucket}")
        for index, member in enumerate(ordered):
            donors[member.sample_id] = ordered[(index + 1) % len(ordered)]
    return donors


def _resize_donor(tokens: torch.Tensor, length: int) -> torch.Tensor:
    if tokens.numel() >= length:
        return tokens[:length]
    repeats = math.ceil(length / max(tokens.numel(), 1))
    return tokens.repeat(repeats)[:length]


def _visible_shuffle(
    inputs: torch.Tensor,
    masked: torch.Tensor,
    residue_mask: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    result = inputs.clone()
    for index in range(inputs.shape[0]):
        positions = torch.nonzero(residue_mask[index] & ~masked[index], as_tuple=False).flatten()
        generator = torch.Generator(device="cpu").manual_seed(seed + index * 1_000_003)
        permutation = positions[torch.randperm(len(positions), generator=generator)] if len(positions) else positions
        result[index, positions] = inputs[index, permutation]
    return result


def _canonical_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    support = logits[..., CANONICAL_TOKEN_START : CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT].float()
    if not torch.isfinite(support).all():
        raise FloatingPointError("E006 contextual diagnostic produced non-finite canonical logits")
    return F.log_softmax(support, dim=-1)


def _sample_metric(log_probs: torch.Tensor, targets: torch.Tensor, selected: torch.Tensor) -> dict[str, float | int]:
    positions = torch.nonzero(selected, as_tuple=False).flatten()
    if not len(positions):
        return {"token_count": 0, "ce": math.nan, "top1": math.nan, "top3": math.nan, "top5": math.nan}
    values = log_probs[positions]
    canonical_targets = targets[positions] - CANONICAL_TOKEN_START
    ce = -values.gather(1, canonical_targets[:, None]).squeeze(1)
    top = values.topk(5, dim=-1).indices
    return {
        "token_count": len(positions),
        "ce": float(ce.mean()),
        "top1": float((top[:, :1] == canonical_targets[:, None]).any(-1).float().mean()),
        "top3": float((top[:, :3] == canonical_targets[:, None]).any(-1).float().mean()),
        "top5": float((top == canonical_targets[:, None]).any(-1).float().mean()),
    }


def _baseline_metric(
    probabilities: Sequence[float], targets: torch.Tensor, selected: torch.Tensor
) -> dict[str, float | int]:
    log_probs = torch.log(torch.as_tensor(probabilities, dtype=torch.float64)).expand(len(targets), -1)
    return _sample_metric(log_probs, targets, selected)


def _bootstrap_ci(values: Sequence[float], *, iterations: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [math.nan, math.nan]
    generator = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        means[index] = generator.choice(array, size=len(array), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _aggregate(
    records: list[dict[str, Any]], *, bootstrap_iterations: int, bootstrap_seed: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dimensions = (("overall", "all"), ("length_bucket", str(record["length_bucket"])))
        for dimension, value in dimensions:
            grouped[
                (
                    record["model_variant"],
                    record["panel"],
                    record["mask_fraction"],
                    record["subset"],
                    dimension,
                    value,
                )
            ].append(record)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        variant, panel, fraction, subset, dimension, value = key
        condition_metrics = {}
        for condition in (*CONDITIONS, "uniform", "training_unigram", "length_bucketed_unigram"):
            token_count = sum(int(item["metrics"][condition]["token_count"]) for item in values)
            condition_metrics[condition] = {
                "sample_count": sum(int(item["metrics"][condition]["token_count"]) > 0 for item in values),
                "token_count": token_count,
                **{
                    name: (
                        sum(
                            float(item["metrics"][condition][name]) * int(item["metrics"][condition]["token_count"])
                            for item in values
                            if int(item["metrics"][condition]["token_count"]) > 0
                        )
                        / token_count
                        if token_count
                        else math.nan
                    )
                    for name in ("ce", "top1", "top3", "top5")
                },
            }
            condition_metrics[condition]["perplexity"] = math.exp(min(condition_metrics[condition]["ce"], 50.0))
        differences = {}
        comparisons = {
            "normal_minus_training_unigram": ("normal", "training_unigram"),
            "normal_minus_visible_shuffle": ("normal", "visible_shuffle"),
            "normal_minus_permuted_conditioning": ("normal", "permuted_conditioning"),
            "normal_minus_null_conditioning": ("normal", "null_conditioning"),
        }
        for name, (first, second) in comparisons.items():
            paired = [
                float(item["metrics"][first]["ce"]) - float(item["metrics"][second]["ce"])
                for item in values
                if int(item["metrics"][first]["token_count"]) > 0
            ]
            differences[name] = {
                "sample_count": len(paired),
                "mean_nats_per_token": float(np.mean(paired)) if paired else math.nan,
                "bootstrap_95_ci": _bootstrap_ci(
                    paired,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + int(hashlib.sha256(repr((key, name)).encode()).hexdigest()[:8], 16),
                ),
            }
        output.append(
            {
                "model_variant": variant,
                "panel": panel,
                "mask_fraction": fraction,
                "subset": subset,
                "stratum_dimension": dimension,
                "stratum_value": value,
                "conditions": condition_metrics,
                "paired_ce_differences": differences,
            }
        )
    return output


def _parameter_hashes(model: torch.nn.Module, *, sequence_only: bool = False) -> dict[str, str]:
    prefixes = ("token_embedding.", "position_embedding.", "sequence_layers.", "sequence_norm.", "sequence_output.")
    result = {}
    for name, value in model.state_dict().items():
        if sequence_only and not name.startswith(prefixes):
            continue
        array = value.detach().cpu().contiguous().numpy()
        result[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return result


def _parse_history(config: dict[str, Any]) -> dict[str, Any]:
    source = config["training_history"]
    for record in source["files"]:
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"E006 training-history hash contradiction: {path}")
    rolling = []
    chunk: list[float] = []
    chunk_start = None
    chunk_last = None
    maximum_overflows = 0
    final_scale = None
    row_count = 0
    chunk_size = int(source.get("rolling_window_records", 1000))
    for record in source["files"]:
        if record["kind"] != "metrics":
            continue
        with Path(record["path"]).open() as handle:
            for line in handle:
                item = json.loads(line)
                step = int(item.get("optimizer_step", -1))
                if step < int(record.get("minimum_optimizer_step", 0)) or step > int(
                    record.get("maximum_optimizer_step", 2**63 - 1)
                ):
                    continue
                loss = item.get("losses", {}).get("sequence", item.get("sequence_loss"))
                if loss is None:
                    continue
                row_count += 1
                chunk_start = step if chunk_start is None else chunk_start
                chunk_last = step
                chunk.append(float(loss))
                maximum_overflows = max(maximum_overflows, int(item.get("amp_overflows_total", 0)))
                final_scale = item.get("amp_scale_after", item.get("amp_scale", final_scale))
                if len(chunk) >= chunk_size:
                    rolling.append(
                        {
                            "first_optimizer_step": chunk_start,
                            "last_optimizer_step": int(item.get("optimizer_step", 0)),
                            "record_count": len(chunk),
                            "mean_sequence_loss": float(np.mean(chunk)),
                            "median_sequence_loss": float(np.median(chunk)),
                        }
                    )
                    chunk = []
                    chunk_start = None
                    chunk_last = None
    if chunk:
        rolling.append(
            {
                "first_optimizer_step": chunk_start,
                "last_optimizer_step": chunk_last,
                "record_count": len(chunk),
                "mean_sequence_loss": float(np.mean(chunk)),
                "median_sequence_loss": float(np.median(chunk)),
            }
        )
    validation_by_step = {}
    for record in source["files"]:
        if record["kind"] != "validation":
            continue
        with Path(record["path"]).open() as handle:
            for line in handle:
                item = json.loads(line)
                if "sequence_cross_entropy" in item and "optimizer_step" in item:
                    step = int(item["optimizer_step"])
                    if step < int(record.get("minimum_optimizer_step", 0)) or step > int(
                        record.get("maximum_optimizer_step", 2**63 - 1)
                    ):
                        continue
                    validation_by_step[step] = {
                        "optimizer_step": step,
                        "sequence_cross_entropy": float(item["sequence_cross_entropy"]),
                    }
    trajectory = [validation_by_step[key] for key in sorted(validation_by_step)]
    best = min(trajectory, key=lambda item: (item["sequence_cross_entropy"], item["optimizer_step"]))
    return {
        "metrics_record_count": row_count,
        "rolling_sequence_loss": rolling,
        "validation_trajectory": trajectory,
        "best_selection": best,
        "amp_overflow_count": maximum_overflows,
        "final_amp_scale": final_scale,
    }


def _verify_checkpoint(config: dict[str, Any], authorization: RichDatasetAuthorization) -> dict[str, Any]:
    checkpoint = config["checkpoint"]
    payload = verify_stage_a_checkpoint(
        checkpoint["path"],
        checkpoint["sha256"],
        dataset_identity=_dataset_identity(authorization),
        calibration_sha256=checkpoint["calibration_sha256"],
        production_selection_sha256=checkpoint["production_selection_sha256"],
    )
    if int(payload.get("optimizer_step", -1)) != int(checkpoint["optimizer_step"]):
        raise ValueError("E006 contextual checkpoint optimizer-step contradiction")
    if not math.isclose(
        float(payload.get("selected_validation_sequence_cross_entropy", math.inf)),
        float(checkpoint["validation_sequence_cross_entropy"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("E006 contextual checkpoint validation-CE contradiction")
    return payload


def _verify_protected_stage_a_inputs(config: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for record in config["protected_stage_a_inputs"]:
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"E006 protected Stage-A input hash contradiction: {path}")
        observed[str(path)] = str(record["sha256"])
    return observed


def _load_model(config: dict[str, Any], payload: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = _model(config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _step0_payload(
    config: dict[str, Any], best_payload: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    specification = config.get("step0_checkpoint") or {}
    if not specification.get("path") or not specification.get("sha256"):
        return None, {"status": "unavailable", "reason": "no_verified_step0_checkpoint_pinned"}
    path = Path(specification["path"])
    if not path.is_file() or _sha256(path) != specification["sha256"]:
        raise ValueError("E006 step-0 checkpoint hash contradiction")
    payload = load_checkpoint(path, map_location="cpu")
    expected = {
        "optimizer_step": 0,
        "architecture_version": best_payload.get("architecture_version"),
        "dataset_identity": best_payload.get("dataset_identity"),
        "calibration_sha256": best_payload.get("calibration_sha256"),
        "production_selection_sha256": best_payload.get("production_selection_sha256"),
        "stage": "sequence-pretrain",
        "status": "recovery_only",
        "authorizes_training": False,
    }
    contradictions = [name for name, value in expected.items() if payload.get(name) != value]
    if contradictions:
        raise ValueError("E006 step-0 checkpoint compatibility contradiction")
    if set(payload.get("model", {})) != set(best_payload.get("model", {})):
        raise ValueError("E006 step-0 model-state schema contradiction")
    return payload, {"status": "verified", "path": str(path), "sha256": specification["sha256"]}


def _evaluate_model(
    model: torch.nn.Module,
    *,
    model_variant: str,
    dataset: SequenceOnlyRichDataset,
    panels: dict[str, list[PanelMember]],
    unigram: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    heartbeat_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    change_accumulators: dict[str, dict[str, float]] = defaultdict(
        lambda: {"absolute_sum": 0.0, "value_count": 0.0, "maximum": 0.0}
    )
    batch_size = int(config["evaluation"]["batch_size"])
    mask_token = int(config["model"]["pad_token_id"]) + 1
    fractions = [float(value) for value in config["corruption"]["mask_fractions"]]
    completed_batches = 0
    for panel_name, members in panels.items():
        donors = _donor_map(members)
        for fraction_index, fraction in enumerate(fractions):
            for start in range(0, len(members), batch_size):
                selected_members = members[start : start + batch_size]
                rows = [dataset[item.index] for item in selected_members]
                donor_rows = [dataset[donors[item.sample_id].index] for item in selected_members]
                batch = collate_sequence_pretraining(rows)
                targets = batch["sequence_token_ids"]
                residue_mask = batch["residue_mask"]
                normal, masked = masked_sequence_inputs(
                    targets,
                    residue_mask,
                    mask_token_id=mask_token,
                    probability=fraction,
                    seed=int(config["seed"]),
                    step=10_000 * fraction_index + start,
                )
                shuffled = _visible_shuffle(
                    normal,
                    masked,
                    residue_mask,
                    seed=int(config["seed"]) + 100_000 * fraction_index + start,
                )
                permuted = torch.full_like(targets, int(config["model"]["pad_token_id"]))
                for row_index, (member, donor) in enumerate(zip(selected_members, donor_rows, strict=True)):
                    length = member.length
                    donor_tokens = _resize_donor(torch.as_tensor(donor["token_ids"], dtype=torch.long), length)
                    permuted[row_index, :length] = donor_tokens
                permuted[masked] = mask_token
                null = torch.where(residue_mask, torch.full_like(targets, mask_token), targets)
                inputs = {
                    "normal": normal,
                    "visible_shuffle": shuffled,
                    "permuted_conditioning": permuted,
                    "null_conditioning": null,
                }
                targets_hidden = all(
                    torch.equal(values[masked], torch.full_like(values[masked], mask_token))
                    for values in inputs.values()
                )
                if not targets_hidden:
                    raise RuntimeError("E006 contextual diagnostic exposed targets at corrupted positions")
                logits_by_condition = {}
                canonical_logits_by_condition = {}
                with torch.inference_mode():
                    for condition, values in inputs.items():
                        logits = model.forward_sequence_pretraining(values.to(device), residue_mask.to(device))
                        canonical_logits_by_condition[condition] = (
                            logits[..., CANONICAL_TOKEN_START : CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT]
                            .float()
                            .cpu()
                        )
                        logits_by_condition[condition] = _canonical_log_probabilities(logits).cpu()
                normal_logits = canonical_logits_by_condition["normal"]
                for condition in CONDITIONS[1:]:
                    difference = (normal_logits - canonical_logits_by_condition[condition]).abs()[residue_mask]
                    key = f"{model_variant}:{panel_name}:{fraction}:{condition}"
                    accumulator = change_accumulators[key]
                    accumulator["absolute_sum"] += float(difference.sum())
                    accumulator["value_count"] += int(difference.numel())
                    accumulator["maximum"] = max(accumulator["maximum"], float(difference.max()))
                for row_index, member in enumerate(selected_members):
                    subsets = {
                        "all_valid": residue_mask[row_index],
                        "corrupted": masked[row_index],
                        "visible": residue_mask[row_index] & ~masked[row_index],
                    }
                    global_probabilities = unigram["token_frequencies"]
                    bucket_probabilities = unigram["length_bucketed"][str(member.length_bucket)]["token_frequencies"]
                    uniform = [1.0 / CANONICAL_TOKEN_COUNT] * CANONICAL_TOKEN_COUNT
                    for subset_name, subset in subsets.items():
                        metrics = {
                            condition: _sample_metric(
                                logits_by_condition[condition][row_index], targets[row_index], subset
                            )
                            for condition in CONDITIONS
                        }
                        metrics.update(
                            uniform=_baseline_metric(uniform, targets[row_index], subset),
                            training_unigram=_baseline_metric(global_probabilities, targets[row_index], subset),
                            length_bucketed_unigram=_baseline_metric(bucket_probabilities, targets[row_index], subset),
                        )
                        records.append(
                            {
                                "model_variant": model_variant,
                                "panel": panel_name,
                                "sample_id": member.sample_id,
                                "length": member.length,
                                "length_bucket": member.length_bucket,
                                "mask_fraction": fraction,
                                "subset": subset_name,
                                "metrics": metrics,
                            }
                        )
                completed_batches += 1
                if heartbeat_path is not None:
                    _atomic_json(
                        heartbeat_path,
                        {
                            "status": "running",
                            "stage": "sequence_only_evaluation",
                            "model_variant": model_variant,
                            "panel": panel_name,
                            "mask_fraction": fraction,
                            "processed_batches": completed_batches,
                            "memory": _memory(device),
                            "timestamp_utc": _utc_now(),
                            "authorizes_training": False,
                        },
                    )
                if float(_memory(device)["current_rss_mib"] or 0) > float(config["memory"]["maximum_rss_mib"]):
                    raise MemoryError("E006 contextual diagnostic RSS limit exceeded")
    changes = {
        key: {
            "value_count": int(values["value_count"]),
            "mean_absolute_logit_change": values["absolute_sum"] / max(values["value_count"], 1),
            "maximum_absolute_logit_change": values["maximum"],
        }
        for key, values in sorted(change_accumulators.items())
    }
    return records, changes


def _gate(aggregates: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["interpretation_gate"]
    required = float(thresholds["unigram_improvement_nats"])
    context_required = float(thresholds["context_degradation_nats"])
    minimum_fraction = float(thresholds["minimum_passing_strata_fraction"])

    def decision(threshold: float, context_threshold: float) -> dict[str, Any]:
        panel_results = {}
        for panel in ("original_validation", "independent_diagnostic"):
            overall = [
                item
                for item in aggregates
                if item["model_variant"] == "best"
                and item["panel"] == panel
                and item["subset"] == "corrupted"
                and item["stratum_dimension"] == "overall"
            ]
            checks = []
            for item in overall:
                differences = item["paired_ce_differences"]
                unigram = differences["normal_minus_training_unigram"]
                shuffled = differences["normal_minus_visible_shuffle"]
                null = differences["normal_minus_null_conditioning"]
                checks.append(
                    {
                        "mask_fraction": item["mask_fraction"],
                        "unigram_pass": unigram["mean_nats_per_token"] <= -threshold
                        and unigram["bootstrap_95_ci"][1] < 0,
                        "shuffle_pass": shuffled["mean_nats_per_token"] <= -context_threshold
                        and shuffled["bootstrap_95_ci"][1] < 0,
                        "null_pass": null["mean_nats_per_token"] <= -context_threshold
                        and null["bootstrap_95_ci"][1] < 0,
                    }
                )
            strata = [
                item
                for item in aggregates
                if item["model_variant"] == "best"
                and item["panel"] == panel
                and item["subset"] == "corrupted"
                and item["stratum_dimension"] == "length_bucket"
            ]
            passing_strata = 0
            for item in strata:
                values = item["paired_ce_differences"]
                if (
                    values["normal_minus_training_unigram"]["mean_nats_per_token"] <= -threshold
                    and values["normal_minus_null_conditioning"]["mean_nats_per_token"] <= -context_threshold
                ):
                    passing_strata += 1
            panel_results[panel] = {
                "overall_checks": checks,
                "overall_pass": bool(checks)
                and all(all(value for key, value in item.items() if key != "mask_fraction") for item in checks),
                "passing_length_corruption_strata": passing_strata,
                "total_length_corruption_strata": len(strata),
                "strata_pass_fraction": passing_strata / max(len(strata), 1),
                "strata_consistency_pass": passing_strata / max(len(strata), 1) >= minimum_fraction,
            }
        verified = all(item["overall_pass"] and item["strata_consistency_pass"] for item in panel_results.values())
        return {
            "threshold": threshold,
            "context_threshold": context_threshold,
            "panels": panel_results,
            "passed": verified,
        }

    primary = decision(required, context_required)
    overall_checks = [check for panel in primary["panels"].values() for check in panel["overall_checks"]]
    if primary["passed"]:
        classification = "contextual_learning_verified"
    elif overall_checks and all(not item["unigram_pass"] for item in overall_checks):
        classification = "marginal_frequency_only"
    elif (
        overall_checks
        and all(item["unigram_pass"] for item in overall_checks)
        and all(not item["shuffle_pass"] and not item["null_pass"] for item in overall_checks)
    ):
        classification = "conditioning_path_ineffective"
    else:
        classification = "inconclusive"
    sensitivity = [decision(float(value), float(value)) for value in thresholds.get("sensitivity_thresholds_nats", [])]
    return {
        "classification": classification,
        "primary": primary,
        "sensitivity": sensitivity,
        "acceptable_for_stage_b": classification == "contextual_learning_verified",
    }


def prepare_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    authorization = _authorization(config)
    checkpoint = _verify_checkpoint(config, authorization)
    protected_stage_a_inputs = _verify_protected_stage_a_inputs(config)
    train = SequenceOnlyRichDataset(authorization, split="train")
    validation = SequenceOnlyRichDataset(authorization, split="validation")
    if train.available_schema_columns != validation.available_schema_columns:
        raise ValueError("E006 train/validation sequence-only schema contradiction")
    schema_projection = validate_sequence_schema_columns(
        validation.available_schema_columns,
        source="validated Phase-1 dataset",
    )
    schema_projection["validated_splits"] = ["train", "validation"]
    panels, panel_diagnostics = select_panels(
        validation,
        original_size=int(config["panels"]["original_size"]),
        independent_size=int(config["panels"]["independent_size"]),
        training_seed=int(config["training_seed"]),
        diagnostic_seed=int(config["seed"]),
        minimum_per_nonempty_bucket=int(config["panels"]["minimum_per_nonempty_bucket"]),
        maximum_length=int(config["panels"]["maximum_length"]),
        expected_original_sample_id_sha256=str(config["panels"]["expected_original_sample_id_sha256"]),
    )
    return {
        "authorization": authorization,
        "checkpoint": checkpoint,
        "train": train,
        "validation": validation,
        "panels": panels,
        "panel_diagnostics": panel_diagnostics,
        "schema_projection": schema_projection,
        "protected_stage_a_inputs": protected_stage_a_inputs,
    }


def plan_diagnostic(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    prepared = prepare_diagnostic(config)
    history = _parse_history(config)
    _, step0_status = _step0_payload(config, prepared["checkpoint"])
    output = Path(config["output_dir"])
    return {
        "status": "planned" if not output.exists() else "blocked_existing_output",
        "version": DIAGNOSTIC_VERSION,
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "checkpoint_optimizer_step": int(prepared["checkpoint"]["optimizer_step"]),
        "dataset_identity": _dataset_identity(prepared["authorization"]),
        "panel_selection": prepared["panel_diagnostics"],
        "schema_projection": prepared["schema_projection"],
        "step0_checkpoint": step0_status,
        "training_history": {
            "metrics_record_count": history["metrics_record_count"],
            "validation_count": len(history["validation_trajectory"]),
            "best_selection": history["best_selection"],
            "amp_overflow_count": history["amp_overflow_count"],
        },
        "protected_stage_a_inputs": prepared["protected_stage_a_inputs"],
        "sequence_only": True,
        "feature_complexity": "O(N)",
        "constructs_rich_residue_features": False,
        "constructs_rich_pair_features": False,
        "optimizer_created": False,
        "backward_performed": False,
        "authorizes_training": False,
        "authorizes_evaluation": False,
        "output_dir": str(output),
    }


def run_diagnostic(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    validate_config(config)
    output = Path(config["output_dir"])
    if output.exists():
        raise FileExistsError(f"E006 contextual diagnostic output already exists: {output}")
    prepared = prepare_diagnostic(config)
    output.mkdir(parents=True)
    heartbeat = output / "heartbeat.json"
    report_path = output / "report.json"
    protocol_path = output / "protocol.json"
    started = time.monotonic()
    started_utc = _utc_now()
    authorization: RichDatasetAuthorization = prepared["authorization"]
    protected_before = _protected_hashes(authorization)
    source_hashes_before = prepared["protected_stage_a_inputs"]
    try:
        train = prepared["train"]
        unigram = training_unigram(train, smoothing=float(config["baselines"]["smoothing"]))
        history = _parse_history(config)
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("E006 contextual diagnostic requested unavailable CUDA")
        best_model = _load_model(config, prepared["checkpoint"], device)
        best_before = _parameter_hashes(best_model)
        records, logit_changes = _evaluate_model(
            best_model,
            model_variant="best",
            dataset=prepared["validation"],
            panels=prepared["panels"],
            unigram=unigram,
            config=config,
            device=device,
            heartbeat_path=heartbeat,
        )
        best_after = _parameter_hashes(best_model)
        if best_before != best_after:
            raise RuntimeError("E006 contextual diagnostic mutated checkpoint parameters")
        del best_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        step0_payload, step0_status = _step0_payload(config, prepared["checkpoint"])
        parameter_change = {"status": "unavailable"}
        if step0_payload is not None:
            step0_model = _load_model(config, step0_payload, device)
            step0_records, step0_changes = _evaluate_model(
                step0_model,
                model_variant="step0",
                dataset=prepared["validation"],
                panels=prepared["panels"],
                unigram=unigram,
                config=config,
                device=device,
                heartbeat_path=heartbeat,
            )
            records.extend(step0_records)
            logit_changes.update(step0_changes)
            del step0_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            changed_norm = 0.0
            reference_norm = 0.0
            for name, value in prepared["checkpoint"]["model"].items():
                if name.startswith(
                    (
                        "token_embedding.",
                        "position_embedding.",
                        "sequence_layers.",
                        "sequence_norm.",
                        "sequence_output.",
                    )
                ):
                    difference = value.float() - step0_payload["model"][name].float()
                    changed_norm += float(difference.square().sum())
                    reference_norm += float(step0_payload["model"][name].float().square().sum())
            parameter_change = {
                "status": "computed",
                "sequence_trunk_l2_change": math.sqrt(changed_norm),
                "step0_sequence_trunk_l2_norm": math.sqrt(reference_norm),
                "relative_l2_change": math.sqrt(changed_norm) / max(math.sqrt(reference_norm), 1e-12),
            }
        aggregates = _aggregate(
            records,
            bootstrap_iterations=int(config["bootstrap"]["iterations"]),
            bootstrap_seed=int(config["bootstrap"]["seed"]),
        )
        gate = _gate(aggregates, config)
        authorization_after = _authorization(config)
        protected_after = _protected_hashes(authorization_after)
        source_hashes_after = {record["path"]: _sha256(record["path"]) for record in config["protected_stage_a_inputs"]}
        if protected_before != protected_after or source_hashes_before != source_hashes_after:
            raise RuntimeError("E006 contextual diagnostic protected inputs changed")
        report = {
            "status": "completed",
            "version": DIAGNOSTIC_VERSION,
            "classification": gate["classification"],
            "scientific_gate": gate,
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "checkpoint": {
                "path": config["checkpoint"]["path"],
                "sha256": config["checkpoint"]["sha256"],
                "optimizer_step": int(prepared["checkpoint"]["optimizer_step"]),
                "validation_sequence_cross_entropy": float(
                    prepared["checkpoint"]["selected_validation_sequence_cross_entropy"]
                ),
                "independently_verified": True,
                "existing_authorizes_joint_training": True,
            },
            "dataset_identity": _dataset_identity(authorization),
            "dataset_protocol_sha256": authorization.protocol_sha256,
            "panel_selection": prepared["panel_diagnostics"],
            "schema_projection": prepared["schema_projection"],
            "unigram_baseline": unigram,
            "corruption_contract": {
                "mask_fractions": [float(value) for value in config["corruption"]["mask_fractions"]],
                "diffusion_timestep_distribution": "disabled",
                "paired_draws_across_conditions": True,
                "permuted_conditioning_policy": "deterministic_derangement_within_length_bucket",
            },
            "aggregates": aggregates,
            "logit_sensitivity": logit_changes,
            "step0_checkpoint": step0_status,
            "sequence_trunk_parameter_change": parameter_change,
            "training_history": history,
            "leakage_checks": {
                "corrupted_targets_exposed": False,
                "mask_token_id": int(config["model"]["pad_token_id"]) + 1,
                "canonical_prediction_support": list(
                    range(CANONICAL_TOKEN_START, CANONICAL_TOKEN_START + CANONICAL_TOKEN_COUNT)
                ),
            },
            "execution_contract": {
                "sequence_only": True,
                "complexity": "O(N)",
                "rich_residue_features_constructed": False,
                "rich_pair_features_constructed": False,
                "geometry_and_fusion_paths_used": False,
                "optimizer_created": False,
                "backward_performed": False,
                "parameters_unchanged": True,
            },
            "protected_inputs_before": protected_before,
            "protected_inputs_after": protected_after,
            "protected_inputs_unchanged": True,
            "stage_a_inputs_before": source_hashes_before,
            "stage_a_inputs_after": source_hashes_after,
            "stage_a_inputs_unchanged": True,
            "memory": _memory(device),
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_evaluation": False,
        }
        _atomic_json(report_path, report)
        protocol = {
            "status": "completed",
            "version": DIAGNOSTIC_VERSION,
            "classification": report["classification"],
            "report_path": str(report_path),
            "report_sha256": _sha256(report_path),
            "checkpoint_sha256": config["checkpoint"]["sha256"],
            "dataset_protocol_sha256": authorization.protocol_sha256,
            "protected_inputs_unchanged": True,
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_evaluation": False,
            "completed_utc": report["completed_utc"],
        }
        _atomic_json(protocol_path, protocol)
        _atomic_json(
            heartbeat,
            {
                "status": "completed",
                "completed_utc": report["completed_utc"],
                "report_sha256": protocol["report_sha256"],
                "protocol_sha256": _sha256(protocol_path),
                "authorizes_training": False,
            },
        )
        return report
    except BaseException as error:
        failure = {
            "status": "failed",
            "version": DIAGNOSTIC_VERSION,
            "error_type": type(error).__name__,
            "error": str(error)[:1000],
            "completed_utc": _utc_now(),
            "authorizes_training": False,
            "authorizes_joint_training": False,
            "authorizes_evaluation": False,
        }
        _atomic_json(protocol_path, failure)
        _atomic_json(heartbeat, failure)
        raise
