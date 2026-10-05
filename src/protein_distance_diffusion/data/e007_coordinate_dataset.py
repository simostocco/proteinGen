"""Bounded coordinate-only view of immutable E006 rich-geometry sidecars."""

from __future__ import annotations

import bisect
import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from protein_distance_diffusion.data.rich_geometry import RichDatasetAuthorization

COORDINATE_VIEW_VERSION = "e007_coordinate_sidecar_view_v1"
REQUIRED_COLUMNS = (
    "sample_id",
    "split",
    "sequence",
    "ca_coordinates",
    "ca_mask",
    "chain_continuity_mask",
    "chain_break_mask",
    "source_sha256",
    "npz_sha256",
    "schema_version",
)


def coordinate_acceptance_reasons(
    *,
    sequence_length: int,
    ca_mask: Any,
    chain_continuity_mask: Any,
    chain_break_mask: Any,
) -> tuple[str, ...]:
    """Apply the canonical contiguous, complete-C-alpha coordinate policy."""
    residue_mask = np.asarray(ca_mask, dtype=bool)
    continuity = np.asarray(chain_continuity_mask, dtype=bool)
    breaks = np.asarray(chain_break_mask, dtype=bool)
    if residue_mask.shape != (sequence_length,):
        raise ValueError("E007 C-alpha mask length contradiction")
    expected_links = max(sequence_length - 1, 0)
    if continuity.shape != (expected_links,) or breaks.shape != continuity.shape:
        raise ValueError("E007 chain-continuity shape contradiction")
    if not np.array_equal(breaks, ~continuity):
        raise ValueError("E007 continuity/break masks contradict")
    reasons = []
    if not residue_mask.any():
        reasons.append("no_valid_calpha")
    if not residue_mask.all():
        reasons.append("missing_calpha")
    if not continuity.all():
        reasons.append("chain_break")
    return tuple(reasons)


@dataclass(frozen=True)
class _Locator:
    path: Path
    row_group: int
    count: int


def validate_coordinate_row(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Validate and project one sidecar row without exposing amino-acid tokens."""
    missing = [name for name in REQUIRED_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"E007 coordinate row lacks required columns: {missing}")
    if row["split"] != split:
        raise ValueError("E007 coordinate row split contradiction")
    sequence = str(row["sequence"])
    coordinates = np.asarray(row["ca_coordinates"], dtype=np.float32)
    residue_mask = np.asarray(row["ca_mask"], dtype=bool)
    continuity = np.asarray(row["chain_continuity_mask"], dtype=bool)
    breaks = np.asarray(row["chain_break_mask"], dtype=bool)
    length = len(sequence)
    if coordinates.shape != (length, 3):
        raise ValueError("E007 sequence/coordinate/mask length contradiction")
    rejection_reasons = coordinate_acceptance_reasons(
        sequence_length=length,
        ca_mask=residue_mask,
        chain_continuity_mask=continuity,
        chain_break_mask=breaks,
    )
    if not np.isfinite(coordinates[residue_mask]).all():
        raise ValueError("E007 valid C-alpha coordinates are non-finite")
    accepted = not rejection_reasons
    return {
        "sample_id": str(row["sample_id"]),
        "split": split,
        "sequence_length": length,
        "coordinates": torch.from_numpy(coordinates.copy()),
        "residue_mask": torch.from_numpy(residue_mask.copy()),
        "chain_continuity_mask": torch.from_numpy(continuity.copy()),
        "accepted_contiguous_single_chain": accepted,
        "coordinate_acceptance_reasons": rejection_reasons,
        "source_sha256": str(row["source_sha256"]),
        "npz_sha256": str(row["npz_sha256"]),
        "sidecar_schema_version": str(row["schema_version"]),
    }


class E007CoordinateDataset(Dataset[dict[str, Any]]):
    """Lazy projected sidecar loader; only one row group is cached."""

    def __init__(self, authorization: RichDatasetAuthorization, *, split: str) -> None:
        if split not in {"train", "validation"}:
            raise ValueError(f"unknown E007 split: {split}")
        self.authorization = authorization
        self.split = split
        import json

        protocol = json.loads((authorization.root / "protocol.json").read_text())
        locators: list[_Locator] = []
        for shard in protocol["shards"]:
            if shard["dataset"] != split:
                continue
            path = authorization.root / shard["path"]
            parquet = pq.ParquetFile(path)
            names = set(parquet.schema_arrow.names)
            missing = sorted(set(REQUIRED_COLUMNS) - names)
            if missing:
                raise ValueError(f"E007 {path.name} lacks coordinate columns: {missing}")
            for row_group in range(parquet.num_row_groups):
                locators.append(_Locator(path, row_group, parquet.metadata.row_group(row_group).num_rows))
        self._locators = tuple(locators)
        self._ends: list[int] = []
        total = 0
        for locator in locators:
            total += locator.count
            self._ends.append(total)
        if total != authorization.split_counts[split]:
            raise ValueError("E007 sidecar row count contradicts authorization")
        self._cached_index: int | None = None
        self._cached_rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("E007 coordinate indices must be scalar integers")
        try:
            index = operator.index(index)
        except TypeError as error:
            raise TypeError("E007 coordinate indices must be scalar integers") from error
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        locator_index = bisect.bisect_right(self._ends, index)
        start = 0 if locator_index == 0 else self._ends[locator_index - 1]
        if locator_index != self._cached_index:
            locator = self._locators[locator_index]
            self._cached_rows = (
                pq.ParquetFile(locator.path)
                .read_row_group(locator.row_group, columns=list(REQUIRED_COLUMNS))
                .to_pylist()
            )
            self._cached_index = locator_index
        return validate_coordinate_row(self._cached_rows[index - start], split=self.split)


def _proper_rotation(generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    matrix = torch.randn((3, 3), generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diag(r)).masked_fill(torch.diag(r) == 0, 1)
    q = q * signs
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def stable_center_valid_coordinates(coordinates: torch.Tensor, residue_mask: torch.Tensor) -> torch.Tensor:
    """Center valid residues in float64 before padding, returning the source dtype."""
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError("E007 coordinates must have shape [N,3]")
    if residue_mask.shape != coordinates.shape[:1] or not bool(residue_mask.any()):
        raise ValueError("E007 centering requires a nonempty aligned residue mask")
    valid = coordinates[residue_mask].double()
    if not bool(torch.isfinite(valid).all()):
        raise ValueError("E007 centering received non-finite valid coordinates")
    centroid = valid.mean(dim=0, keepdim=True)
    centered = torch.zeros_like(coordinates)
    centered[residue_mask] = (valid - centroid).to(coordinates.dtype)
    return centered


def collate_e007_coordinates(
    rows: list[dict[str, Any]],
    *,
    augment_rotation: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Dynamically pad and center coordinate rows without returning tokens."""
    if not rows or any(not row["accepted_contiguous_single_chain"] for row in rows):
        raise ValueError("E007 batches require accepted contiguous single-chain rows")
    maximum = max(int(row["sequence_length"]) for row in rows)
    coordinates = torch.zeros((len(rows), maximum, 3), dtype=torch.float32)
    residue_mask = torch.zeros((len(rows), maximum), dtype=torch.bool)
    continuity = torch.zeros((len(rows), max(maximum - 1, 0)), dtype=torch.bool)
    lengths = torch.tensor([int(row["sequence_length"]) for row in rows], dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for index, row in enumerate(rows):
        length = int(row["sequence_length"])
        value = row["coordinates"].float()
        if augment_rotation:
            value = value @ _proper_rotation(generator, value.dtype)
        value = stable_center_valid_coordinates(value, row["residue_mask"].bool())
        coordinates[index, :length] = value
        residue_mask[index, :length] = row["residue_mask"]
        continuity[index, : max(length - 1, 0)] = row["chain_continuity_mask"]
    pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]
    indices = torch.arange(maximum)
    relative_separation = (indices[:, None] - indices[None, :]).abs()[None].float()
    relative_separation = relative_separation / lengths.sub(1).clamp_min(1)[:, None, None]
    return {
        "coordinates": coordinates,
        "residue_mask": residue_mask,
        "pair_mask": pair_mask,
        "chain_continuity_mask": continuity,
        "relative_separation": relative_separation,
        "lengths": lengths,
        "sample_ids": [str(row["sample_id"]) for row in rows],
    }


def global_rms_coordinate_radius(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Calculate the train-only pooled centered coordinate-component scale."""
    squared_sum = 0.0
    count = 0
    for row in rows:
        coordinates = row["coordinates"].double()
        mask = row["residue_mask"].bool()
        valid = coordinates[mask]
        if not len(valid):
            continue
        centered = valid - valid.mean(dim=0, keepdim=True)
        squared_sum += float(centered.square().sum())
        count += len(valid)
    if count == 0:
        raise ValueError("coordinate normalization requires valid training coordinates")
    return {
        "coordinate_scale_angstrom": math.sqrt(squared_sum / (3 * count)),
        "train_sample_count": len(rows),
        "valid_coordinate_count": count,
        "valid_residue_count": count,
    }
