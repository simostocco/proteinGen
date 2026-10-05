"""Authorized, bounded E006 rich-geometry loading and invariant featurization."""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import math
import operator
import random
import sqlite3
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from protein_distance_diffusion.data.rich_geometry_sidecars import (
    SIDECAR_SCHEMA_VERSION,
    TORSION_CONVENTION_VERSION,
    TORSION_FLOAT32_ATOL,
    TORSION_NEUTRAL_SIN_COS,
    sha256_file,
)

RICH_FEATURE_VERSION = "e006_se3_invariant_rich_features_v1"
RICH_RESIDUE_FEATURE_DIM = 32
DEFAULT_RBF_BINS = 16
RICH_PAIR_FEATURE_DIM = DEFAULT_RBF_BINS + 15
SPLIT_DATASETS = {"train": "train", "validation": "validation"}


@dataclass(frozen=True)
class RichDatasetAuthorization:
    root: Path
    protocol_sha256: str
    schema_sha256: str
    vocabulary_sha256: str
    normalization_sha256: str
    shard_inventory_sha256: str
    split_counts: dict[str, int]
    observed_shard_hashes: dict[str, str]
    protected_input_resolution_counts: dict[str, int] = field(default_factory=dict)
    relocated_protected_inputs: tuple[str, ...] = ()
    protected_input_total: int = 0
    relocated_protected_input_identities: tuple[dict[str, str], ...] = ()


@dataclass(frozen=True)
class _RowGroupLocator:
    path: Path
    row_group: int
    row_count: int


def _inside(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_protected_input_relocations(relocations: list[dict[str, str]] | None) -> list[tuple[Path, Path]]:
    if not isinstance(relocations, list) or not relocations:
        raise ValueError("E006 Phase-1 protected input relocation list is absent or empty")
    validated: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for record in relocations:
        if not isinstance(record, dict) or set(record) != {"recorded_root", "verification_root"}:
            raise ValueError("E006 Phase-1 protected input relocation schema is invalid")
        recorded_text, verification_text = record["recorded_root"], record["verification_root"]
        if not isinstance(recorded_text, str) or not isinstance(verification_text, str):
            raise ValueError("E006 Phase-1 protected input relocation roots must be paths")
        recorded_root, verification_root = Path(recorded_text), Path(verification_text)
        if any(".." in item.parts or "\\" in str(item) for item in (recorded_root, verification_root)):
            raise ValueError("E006 Phase-1 protected input relocation roots are malformed")
        if not recorded_root.is_absolute() or not verification_root.is_absolute():
            raise ValueError("E006 Phase-1 protected input relocation roots must be absolute")
        if recorded_root == verification_root:
            raise ValueError("E006 Phase-1 protected input relocation roots must be lexically distinct")
        key = recorded_root.as_posix()
        if key in seen:
            raise ValueError(f"E006 Phase-1 duplicate protected input relocation root: {key}")
        seen.add(key)
        validated.append((recorded_root, verification_root))
    return validated


def _resolve_protected_input(
    path_text: str,
    expected_sha256: str,
    relocations: list[dict[str, str]] | None,
) -> tuple[Path, str]:
    """Resolve one immutable input without conflating path and content identity."""
    validated_relocations = validate_protected_input_relocations(relocations) if relocations else []

    recorded_path = Path(path_text)
    if not recorded_path.is_absolute() or ".." in recorded_path.parts or "\\" in path_text:
        raise ValueError(f"E006 Phase-1 protected input path is not absolute: {path_text}")
    if recorded_path.is_file() and sha256_file(recorded_path) == expected_sha256:
        return recorded_path, "recorded_path"

    for recorded_root, resolved_root in validated_relocations:
        try:
            relative = recorded_path.relative_to(recorded_root)
        except ValueError:
            continue
        if ".." in relative.parts or relative.is_absolute():
            raise ValueError(f"E006 Phase-1 protected input relocation path is malformed: {path_text}")
        candidate = (resolved_root / relative).resolve()
        resolved_verification_root = resolved_root.resolve()
        if not _inside(resolved_verification_root, candidate):
            raise ValueError(f"E006 Phase-1 protected input relocation escapes its root: {path_text}")
        if candidate.is_file() and sha256_file(candidate) == expected_sha256:
            return candidate, "relocated_verification_path"
    raise ValueError(f"E006 Phase-1 protected input hash contradiction: {recorded_path}")


def authorize_rich_geometry_dataset(
    root: str | Path,
    *,
    expected_protocol_sha256: str,
    expected_schema_sha256: str,
    expected_vocabulary_sha256: str,
    expected_normalization_sha256: str,
    expected_shard_inventory_sha256: str,
    protected_input_relocations: list[dict[str, str]] | None = None,
) -> RichDatasetAuthorization:
    """Attest a full Phase-1 dataset before it can be used for training."""
    directory = Path(root).resolve()
    if protected_input_relocations:
        validate_protected_input_relocations(protected_input_relocations)
    protocol_path = directory / "protocol.json"
    if sha256_file(protocol_path) != expected_protocol_sha256:
        raise ValueError("E006 Phase-1 protocol SHA-256 contradiction")
    protocol = json.loads(protocol_path.read_text())
    requirements = {
        "status": "completed",
        "mode": "full",
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "torsion_convention_version": TORSION_CONVENTION_VERSION,
        "authorizes_training": True,
        "authorizes_definitive_dataset": True,
    }
    contradictions = [key for key, value in requirements.items() if protocol.get(key) != value]
    if contradictions:
        raise ValueError(f"E006 Phase-1 authorization contradiction: {', '.join(contradictions)}")
    if protocol.get("unexplained_failure_count") != 0:
        raise ValueError("E006 Phase-1 dataset has unexplained failures")
    if not protocol.get("protected_inputs_unchanged") or not protocol.get("observed_phase1_inputs_unchanged"):
        raise ValueError("E006 Phase-1 protected inputs were not preserved")
    protected_before = protocol.get("input_hashes_before")
    if not isinstance(protected_before, dict) or protected_before != protocol.get("input_hashes_after"):
        raise ValueError("E006 Phase-1 protected input inventories contradict")
    resolution_counts = {"recorded_path": 0, "relocated_verification_path": 0}
    relocated_paths = []
    relocated_identities = []
    for path_text, expected in protected_before.items():
        _resolved, resolution = _resolve_protected_input(
            str(path_text),
            str(expected),
            protected_input_relocations,
        )
        resolution_counts[resolution] += 1
        if resolution == "relocated_verification_path":
            relocated_paths.append(str(path_text))
            relocated_identities.append({"identity": str(path_text), "sha256": str(expected)})

    metadata = {
        "schema.json": expected_schema_sha256,
        "vocabulary.json": expected_vocabulary_sha256,
        "normalization.json": expected_normalization_sha256,
        "shard_hashes.sha256": expected_shard_inventory_sha256,
    }
    for name, expected in metadata.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"E006 Phase-1 metadata SHA-256 contradiction: {name}")
    schema = json.loads((directory / "schema.json").read_text())
    if schema.get("schema_version") != SIDECAR_SCHEMA_VERSION or schema.get("dense_pair_features_stored") is not False:
        raise ValueError("E006 Phase-1 schema metadata is incompatible")
    vocabulary = json.loads((directory / "vocabulary.json").read_text())
    if vocabulary.get("version") != "canonical_20_pad_mask_v1" or len(vocabulary.get("tokens", [])) != 22:
        raise ValueError("E006 Phase-1 vocabulary metadata is incompatible")
    inventory_lines = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (directory / "shard_hashes.sha256").read_text().splitlines()
        if "  " in line
    }

    seen_paths: set[str] = set()
    shard_hashes: dict[str, str] = {}
    split_counts = {"train": 0, "validation": 0}
    with tempfile.TemporaryDirectory(prefix="e006-membership-") as temporary:
        connection = sqlite3.connect(Path(temporary) / "membership.sqlite")
        connection.execute("CREATE TABLE membership(split TEXT,sample_id TEXT,PRIMARY KEY(sample_id))")
        for record in protocol.get("shards", []):
            relative = str(record.get("path") or "")
            dataset = str(record.get("dataset") or "")
            if not relative or relative in seen_paths:
                raise ValueError(f"E006 Phase-1 missing or duplicate shard path: {relative!r}")
            seen_paths.add(relative)
            path = (directory / relative).resolve()
            if not _inside(directory, path) or not path.is_file():
                raise ValueError(f"E006 Phase-1 shard is missing or escapes its root: {relative}")
            digest = sha256_file(path)
            if digest != str(record.get("sha256") or ""):
                raise ValueError(f"E006 Phase-1 shard SHA-256 contradiction: {relative}")
            expected_rows = int(record.get("row_count", -1))
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != expected_rows:
                raise ValueError(f"E006 Phase-1 shard row-count contradiction: {relative}")
            if inventory_lines.get(relative) != digest:
                raise ValueError(f"E006 Phase-1 shard inventory contradiction: {relative}")
            shard_hashes[relative] = digest
            if dataset not in SPLIT_DATASETS:
                continue
            split_counts[dataset] += expected_rows
            for batch in parquet.iter_batches(columns=["sample_id", "split"], batch_size=4096):
                rows = batch.to_pydict()
                for sample_id, row_split in zip(rows["sample_id"], rows["split"], strict=True):
                    if str(row_split) != dataset:
                        raise ValueError(f"E006 physical/row split contradiction: {sample_id}")
                    try:
                        connection.execute("INSERT INTO membership VALUES (?,?)", (dataset, str(sample_id)))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"E006 duplicate or cross-split sample ID: {sample_id}") from error
        connection.close()
    if set(inventory_lines) != seen_paths:
        raise ValueError("E006 Phase-1 shard inventory membership contradiction")
    recorded = {key: int(value) for key, value in protocol.get("eligible_split_counts", {}).items()}
    if split_counts != recorded:
        raise ValueError(f"E006 Phase-1 split count contradiction: observed={split_counts}, recorded={recorded}")
    if sum(split_counts.values()) + int(protocol["definitive_observed_counts"]["excluded"]) != int(
        protocol["processed_samples"]
    ):
        raise ValueError("E006 Phase-1 processed/eligible/excluded count contradiction")
    return RichDatasetAuthorization(
        root=directory,
        protocol_sha256=expected_protocol_sha256,
        schema_sha256=expected_schema_sha256,
        vocabulary_sha256=expected_vocabulary_sha256,
        normalization_sha256=expected_normalization_sha256,
        shard_inventory_sha256=expected_shard_inventory_sha256,
        split_counts=split_counts,
        observed_shard_hashes=shard_hashes,
        protected_input_resolution_counts=resolution_counts,
        relocated_protected_inputs=tuple(sorted(relocated_paths)),
        protected_input_total=len(protected_before),
        relocated_protected_input_identities=tuple(sorted(relocated_identities, key=lambda row: row["identity"])),
    )


def _validate_vector(name: str, value: Any, length: int, width: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    expected = (length,) if width is None else (length, width)
    if array.shape != expected:
        raise ValueError(f"E006 {name} shape contradiction: {array.shape} != {expected}")
    return array


def validate_rich_row(row: dict[str, Any], *, split: str) -> None:
    if split not in SPLIT_DATASETS or str(row.get("split")) != split:
        raise ValueError(f"E006 rich row split ownership contradiction: {row.get('sample_id')}")
    if row.get("schema_version") != SIDECAR_SCHEMA_VERSION:
        raise ValueError("E006 rich row schema contradiction")
    if row.get("torsion_convention_version") != TORSION_CONVENTION_VERSION:
        raise ValueError("E006 rich row torsion convention contradiction")
    sequence = str(row.get("sequence") or "")
    length = len(sequence)
    if not length or len(row.get("token_ids") or []) != length:
        raise ValueError("E006 sequence/token length contradiction")
    if any(int(token) < 2 or int(token) >= 22 for token in row["token_ids"]):
        raise ValueError("E006 canonical token range contradiction")
    for atom in ("n", "ca", "c", "o", "cb"):
        coordinates = _validate_vector(f"{atom}_coordinates", row[f"{atom}_coordinates"], length, 3).astype(np.float32)
        mask = _validate_vector(f"{atom}_mask", row[f"{atom}_mask"], length).astype(bool)
        if not np.isfinite(coordinates[mask]).all() or np.any(coordinates[~mask] != 0):
            raise ValueError(f"E006 {atom} coordinate/mask contradiction")
    for name in ("phi", "psi", "omega"):
        values = _validate_vector(f"{name}_sin_cos", row[f"{name}_sin_cos"], length, 2).astype(np.float32)
        mask = _validate_vector(f"{name}_mask", row[f"{name}_mask"], length).astype(bool)
        if not np.isfinite(values).all():
            raise ValueError(f"E006 {name} contains non-finite values")
        if mask.any() and not np.allclose(np.linalg.norm(values[mask], axis=-1), 1.0, atol=TORSION_FLOAT32_ATOL):
            raise ValueError(f"E006 {name} unmasked values are not normalized")
        if (~mask).any() and not np.allclose(
            values[~mask], np.asarray(TORSION_NEUTRAL_SIN_COS), atol=TORSION_FLOAT32_ATOL, rtol=0
        ):
            raise ValueError(f"E006 {name} masked values are not neutral")
    for name in ("local_frame_valid", "cb_source"):
        _validate_vector(name, row[name], length)
    for name in ("chain_continuity_mask", "chain_break_mask"):
        _validate_vector(name, row[name], max(length - 1, 0))
    cb_source = np.asarray(row["cb_source"], dtype=np.int8)
    cb_mask = np.asarray(row["cb_mask"], dtype=bool)
    if not np.isin(cb_source, (-1, 0, 1)).all() or not np.array_equal(cb_mask, cb_source >= 0):
        raise ValueError("E006 native/pseudo C-beta provenance contradiction")


class RichGeometryDataset(Dataset[dict[str, Any]]):
    """Random row-group access without materializing the Phase-1 corpus."""

    def __init__(self, authorization: RichDatasetAuthorization, *, split: str) -> None:
        if split not in SPLIT_DATASETS:
            raise ValueError(f"Unknown E006 split: {split}")
        self.authorization = authorization
        self.split = split
        protocol = json.loads((authorization.root / "protocol.json").read_text())
        locators: list[_RowGroupLocator] = []
        for record in protocol["shards"]:
            if record["dataset"] != split:
                continue
            path = authorization.root / record["path"]
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                count = parquet.metadata.row_group(row_group).num_rows
                locators.append(_RowGroupLocator(path, row_group, count))
        self._locators = tuple(locators)
        self._ends: list[int] = []
        total = 0
        for locator in self._locators:
            total += locator.row_count
            self._ends.append(total)
        if total != authorization.split_counts[split]:
            raise ValueError(f"E006 {split} locator count contradicts authorization")
        self._cached_locator: int | None = None
        self._cached_rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    @property
    def cached_row_count(self) -> int:
        return len(self._cached_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("RichGeometryDataset indices must be scalar integers, not booleans")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise TypeError(
                f"RichGeometryDataset indices must be scalar integers; received {type(index).__name__}"
            ) from error
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        locator_index = bisect.bisect_right(self._ends, index)
        start = 0 if locator_index == 0 else self._ends[locator_index - 1]
        if self._cached_locator != locator_index:
            locator = self._locators[locator_index]
            self._cached_rows = pq.ParquetFile(locator.path).read_row_group(locator.row_group).to_pylist()
            self._cached_locator = locator_index
        row = self._cached_rows[index - start]
        validate_rich_row(row, split=self.split)
        return row

    def iter_metadata(self) -> Iterator[tuple[int, str, int]]:
        offset = 0
        for locator in self._locators:
            table = pq.ParquetFile(locator.path).read_row_group(
                locator.row_group,
                columns=["sample_id", "sequence"],
            )
            values = table.to_pydict()
            for local_index, (sample_id, sequence) in enumerate(
                zip(values["sample_id"], values["sequence"], strict=True)
            ):
                yield offset + local_index, str(sample_id), len(str(sequence))
            offset += locator.row_count


def deterministic_length_bucket_sample(
    dataset: RichGeometryDataset,
    *,
    count: int,
    seed: int,
    boundaries: tuple[int, ...] = (64, 128, 256, 384, 500),
    maximum_length: int | None = None,
) -> list[int]:
    if count < 1:
        raise ValueError("E006 sample count must be positive")
    retained: dict[int, list[tuple[int, int]]] = {boundary: [] for boundary in boundaries}
    global_retained: list[tuple[int, int]] = []
    per_bucket = math.ceil(count / len(boundaries))

    def keep_smallest(heap: list[tuple[int, int]], rank: int, index: int, capacity: int) -> None:
        candidate = (-rank, -index)
        if len(heap) < capacity:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)

    for index, sample_id, length in dataset.iter_metadata():
        if maximum_length is not None and length > maximum_length:
            continue
        boundary = next((value for value in boundaries if length <= value), None)
        if boundary is None:
            continue
        rank = int.from_bytes(hashlib.sha256(f"{seed}:{dataset.split}:{sample_id}".encode()).digest(), "big")
        keep_smallest(retained[boundary], rank, index, per_bucket)
        keep_smallest(global_retained, rank, index, count)
    selected = [(-rank, -index) for bucket in retained.values() for rank, index in bucket]
    selected_indices = {index for _, index in selected}
    selected.extend((-rank, -index) for rank, index in global_retained if -index not in selected_indices)
    selected.sort()
    selected = selected[:count]
    if len(selected) < count:
        raise ValueError(f"E006 {dataset.split} has only {len(selected)} selectable rows, requested {count}")
    return [index for _, index in selected]


def _frames(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(row["n_coordinates"], dtype=np.float32)
    ca = np.asarray(row["ca_coordinates"], dtype=np.float32)
    c = np.asarray(row["c_coordinates"], dtype=np.float32)
    x = c - ca
    helper = n - ca
    z = np.cross(x, helper)
    x_norm = np.linalg.norm(x, axis=-1)
    z_norm = np.linalg.norm(z, axis=-1)
    valid = (
        np.asarray(row["local_frame_valid"], dtype=bool)
        & np.asarray(row["n_mask"], dtype=bool)
        & np.asarray(row["ca_mask"], dtype=bool)
        & np.asarray(row["c_mask"], dtype=bool)
        & (x_norm > 1e-8)
        & (z_norm > 1e-8)
    )
    frames = np.zeros((len(ca), 3, 3), dtype=np.float32)
    if valid.any():
        x[valid] /= x_norm[valid, None]
        z[valid] /= z_norm[valid, None]
        y = np.cross(z[valid], x[valid])
        frames[valid] = np.stack((x[valid], y, z[valid]), axis=-1)
    return frames, valid


def invariant_rich_features(
    row: dict[str, Any],
    *,
    rbf_bins: int = DEFAULT_RBF_BINS,
    maximum_distance: float = 32.0,
    maximum_pair_elements: int = 500 * 500,
) -> dict[str, torch.Tensor]:
    """Build SE(3)-invariant O(N) residue and bounded O(N^2) pair inputs."""
    length = len(row["sequence"])
    if length * length > maximum_pair_elements:
        raise MemoryError(f"E006 pair feature budget exceeded for length {length}")
    frames, frame_mask = _frames(row)
    ca = np.asarray(row["ca_coordinates"], dtype=np.float32)
    ca_mask = np.asarray(row["ca_mask"], dtype=bool)
    atom_masks = np.stack(
        [np.asarray(row[f"{atom}_mask"], dtype=bool) for atom in ("n", "ca", "c", "o", "cb")], axis=-1
    )
    local_positions = []
    for atom in ("n", "c", "o", "cb"):
        coordinates = np.asarray(row[f"{atom}_coordinates"], dtype=np.float32)
        valid = frame_mask & np.asarray(row[f"{atom}_mask"], dtype=bool)
        local = np.zeros((length, 3), dtype=np.float32)
        local[valid] = np.einsum("ni,nij->nj", coordinates[valid] - ca[valid], frames[valid])
        local_positions.append(local)
    torsions = np.concatenate(
        [np.asarray(row[f"{name}_sin_cos"], dtype=np.float32) for name in ("phi", "psi", "omega")], axis=-1
    )
    torsion_masks = np.stack(
        [np.asarray(row[f"{name}_mask"], dtype=np.float32) for name in ("phi", "psi", "omega")], axis=-1
    )
    continuity = np.asarray(row["chain_continuity_mask"], dtype=np.float32)
    previous_continuity = np.pad(continuity, (1, 0))[:length]
    next_continuity = np.pad(continuity, (0, 1))[:length]
    cb_source = np.asarray(row["cb_source"], dtype=np.int8)
    cb_provenance = np.stack((cb_source == 1, cb_source == 0, cb_source == -1), axis=-1).astype(np.float32)
    residue = np.concatenate(
        (
            torsions,
            torsion_masks,
            atom_masks.astype(np.float32),
            frame_mask[:, None].astype(np.float32),
            previous_continuity[:, None],
            next_continuity[:, None],
            cb_provenance,
            *local_positions,
        ),
        axis=-1,
    )
    if residue.shape != (length, RICH_RESIDUE_FEATURE_DIM):
        raise AssertionError(f"Internal E006 residue feature width error: {residue.shape}")

    displacement = ca[None, :, :] - ca[:, None, :]
    distances = np.linalg.norm(displacement, axis=-1).astype(np.float32)
    pair_mask = frame_mask[:, None] & frame_mask[None, :] & ca_mask[:, None] & ca_mask[None, :]
    local_displacement = np.zeros((length, length, 3), dtype=np.float32)
    local_displacement[pair_mask] = np.einsum(
        "nij,nijk->nik",
        displacement,
        np.broadcast_to(frames[:, None], (length, length, 3, 3)),
    )[pair_mask]
    relative_orientation = np.einsum("nki,mkj->nmij", frames, frames).reshape(length, length, 9)
    relative_orientation[~pair_mask] = 0
    centers = np.linspace(0.0, maximum_distance, rbf_bins, dtype=np.float32)
    width = maximum_distance / max(rbf_bins - 1, 1)
    rbf = np.exp(-(((distances[..., None] - centers) / max(width, 1e-6)) ** 2)).astype(np.float32)
    rbf[~(ca_mask[:, None] & ca_mask[None, :])] = 0
    direct_continuity = np.zeros((length, length), dtype=np.float32)
    direct_break = np.zeros((length, length), dtype=np.float32)
    if length > 1:
        indices = np.arange(length - 1)
        direct_continuity[indices, indices + 1] = continuity
        direct_continuity[indices + 1, indices] = continuity
        breaks = np.asarray(row["chain_break_mask"], dtype=np.float32)
        direct_break[indices, indices + 1] = breaks
        direct_break[indices + 1, indices] = breaks
    pair = np.concatenate(
        (
            rbf,
            local_displacement,
            relative_orientation,
            pair_mask[..., None].astype(np.float32),
            direct_continuity[..., None],
            direct_break[..., None],
        ),
        axis=-1,
    )
    return {
        "residue_features": torch.from_numpy(residue),
        "pair_features": torch.from_numpy(pair),
        "pair_feature_mask": torch.from_numpy(pair_mask),
        "distance_matrix": torch.from_numpy(distances),
        "residue_mask": torch.ones(length, dtype=torch.bool),
    }


def collate_rich_geometry(
    rows: list[dict[str, Any]],
    *,
    pad_to_multiple: int = 8,
    maximum_pair_elements: int = 512 * 512,
) -> dict[str, Any]:
    if not rows or pad_to_multiple < 1:
        raise ValueError("E006 collation requires rows and a positive padding multiple")
    features = [invariant_rich_features(row, maximum_pair_elements=maximum_pair_elements) for row in rows]
    lengths = [len(row["sequence"]) for row in rows]
    side = math.ceil(max(lengths) / pad_to_multiple) * pad_to_multiple
    if side * side > maximum_pair_elements:
        raise MemoryError(f"E006 padded pair feature budget exceeded for side {side}")
    batch = len(rows)
    tokens = torch.zeros((batch, side), dtype=torch.long)
    residue_mask = torch.zeros((batch, side), dtype=torch.bool)
    residue_features = torch.zeros((batch, side, RICH_RESIDUE_FEATURE_DIM), dtype=torch.float32)
    pair_features = torch.zeros((batch, side, side, RICH_PAIR_FEATURE_DIM), dtype=torch.float32)
    pair_feature_mask = torch.zeros((batch, side, side), dtype=torch.bool)
    distances = torch.zeros((batch, 1, side, side), dtype=torch.float32)
    atom_masks = {atom: torch.zeros((batch, side), dtype=torch.bool) for atom in ("n", "ca", "c", "o", "cb")}
    torsion_masks = {name: torch.zeros((batch, side), dtype=torch.bool) for name in ("phi", "psi", "omega")}
    frame_mask = torch.zeros((batch, side), dtype=torch.bool)
    native_cb_mask = torch.zeros((batch, side), dtype=torch.bool)
    pseudo_cb_mask = torch.zeros((batch, side), dtype=torch.bool)
    continuity_mask = torch.zeros((batch, max(side - 1, 0)), dtype=torch.bool)
    for index, (row, item, length) in enumerate(zip(rows, features, lengths, strict=True)):
        tokens[index, :length] = torch.as_tensor(row["token_ids"], dtype=torch.long)
        residue_mask[index, :length] = True
        residue_features[index, :length] = item["residue_features"]
        pair_features[index, :length, :length] = item["pair_features"]
        pair_feature_mask[index, :length, :length] = item["pair_feature_mask"]
        distances[index, 0, :length, :length] = item["distance_matrix"]
        for atom in atom_masks:
            atom_masks[atom][index, :length] = torch.as_tensor(row[f"{atom}_mask"], dtype=torch.bool)
        for name in torsion_masks:
            torsion_masks[name][index, :length] = torch.as_tensor(row[f"{name}_mask"], dtype=torch.bool)
        frame_mask[index, :length] = torch.as_tensor(row["local_frame_valid"], dtype=torch.bool)
        cb_source = torch.as_tensor(row["cb_source"], dtype=torch.int8)
        native_cb_mask[index, :length] = cb_source == 1
        pseudo_cb_mask[index, :length] = cb_source == 0
        if length > 1:
            continuity_mask[index, : length - 1] = torch.as_tensor(row["chain_continuity_mask"], dtype=torch.bool)
    biological_pair_mask = residue_mask[:, None, :, None] & residue_mask[:, None, None, :]
    return {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "splits": [str(row["split"]) for row in rows],
        "sequence_token_ids": tokens,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "residue_mask": residue_mask,
        "pair_mask": biological_pair_mask,
        "rich_residue_features": residue_features,
        "rich_pair_features": pair_features,
        "pair_feature_mask": pair_feature_mask[:, None],
        "distance_matrices": distances,
        "atom_masks": atom_masks,
        "torsion_masks": torsion_masks,
        "frame_mask": frame_mask,
        "native_cb_mask": native_cb_mask,
        "pseudo_cb_mask": pseudo_cb_mask,
        "chain_continuity_mask": continuity_mask,
        "experimental_methods": [str(row.get("experimental_method") or "unknown") for row in rows],
    }


def seed_rich_worker(worker_id: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_rich_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    shuffle: bool = False,
) -> DataLoader[dict[str, Any]]:
    if not 0 <= num_workers <= 4 or not 1 <= prefetch_factor <= 2:
        raise ValueError("E006 loader requires 0-4 workers and prefetch_factor in [1, 2]")
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {}
    if num_workers:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        worker_init_fn=seed_rich_worker,
        collate_fn=collate_rich_geometry,
        persistent_workers=False,
        **kwargs,
    )
