"""Versioned sequence-geometry pairing datasets and corruption interfaces."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from torch.utils.data import Dataset

PAIRING_SCHEMA_VERSION = "sequence_geometry_pairing_v1"
VOCABULARY_VERSION = "canonical_20_pad_mask_v1"
MODIFIED_RESIDUE_MAPPING_VERSION = "mse_to_met_v1"
CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
VOCABULARY_TOKENS = ("<PAD>", "<MASK>", *CANONICAL_AMINO_ACIDS)


@dataclass(frozen=True)
class SequenceGeometryVocabulary:
    """Strict 20-residue vocabulary with padding and conditioning-mask tokens."""

    tokens: tuple[str, ...] = VOCABULARY_TOKENS
    version: str = VOCABULARY_VERSION

    def __post_init__(self) -> None:
        if self.tokens != VOCABULARY_TOKENS or self.version != VOCABULARY_VERSION:
            raise ValueError(f"Unsupported vocabulary version: {self.version}")

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<MASK>"]

    def encode(self, sequence: str) -> list[int]:
        mapping = self.token_to_id
        invalid = sorted(set(sequence) - set(CANONICAL_AMINO_ACIDS))
        if invalid:
            raise ValueError(f"Unexpected noncanonical residue token(s): {invalid}")
        if not sequence:
            raise ValueError("Sequence must not be empty")
        return [mapping[residue] for residue in sequence]

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        reverse = dict(enumerate(self.tokens))
        residues = []
        for value in token_ids:
            token = reverse[int(value)]
            if token not in CANONICAL_AMINO_ACIDS:
                raise ValueError(f"Cannot decode special token {token} as a residue")
            residues.append(token)
        return "".join(residues)

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "tokens": list(self.tokens), "unknown_token": None}


class GeometryCorruption(Protocol):
    """A deterministic-generator-aware geometry corruption."""

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class CompositeGeometryCorruption:
    """Apply an explicitly ordered collection of configured corruptions."""

    transforms: tuple[GeometryCorruption, ...]

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError("A corruption mixture must contain at least one transform")

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            matrix, pair_mask = transform(matrix, pair_mask, generator)
        return matrix, pair_mask


def _validate_geometry(matrix: torch.Tensor, pair_mask: torch.Tensor) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Geometry matrix must be square")
    if pair_mask.shape != matrix.shape:
        raise ValueError("Pair mask must have the same shape as the geometry matrix")
    if not torch.isfinite(matrix).all():
        raise ValueError("Geometry matrix contains non-finite values")


def _finalize_corruption(matrix: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    matrix = matrix.clamp_min(0)
    matrix.fill_diagonal_(0)
    pair_mask = pair_mask & pair_mask.transpose(0, 1)
    return matrix * pair_mask.to(matrix.dtype), pair_mask


@dataclass(frozen=True)
class AdditiveSymmetricNoise:
    standard_deviation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.standard_deviation) or self.standard_deviation < 0:
            raise ValueError("standard_deviation must be finite and non-negative")

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_geometry(matrix, pair_mask)
        noise = torch.randn(matrix.shape, dtype=matrix.dtype, device=matrix.device, generator=generator)
        noise = torch.triu(noise, diagonal=1)
        noise = noise + noise.transpose(0, 1)
        return _finalize_corruption(matrix + noise * self.standard_deviation, pair_mask.clone())


@dataclass(frozen=True)
class LongRangePairMask:
    minimum_separation: int
    mask_probability: float

    def __post_init__(self) -> None:
        if self.minimum_separation < 1 or not 0 <= self.mask_probability <= 1:
            raise ValueError("Long-range masking requires minimum_separation >= 1 and probability in [0, 1]")

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_geometry(matrix, pair_mask)
        indices = torch.arange(matrix.shape[0], device=matrix.device)
        eligible = (indices[:, None] - indices[None, :]).abs() >= self.minimum_separation
        draws = torch.rand(matrix.shape, device=matrix.device, generator=generator)
        remove = torch.triu(eligible & (draws < self.mask_probability), diagonal=1)
        remove = remove | remove.transpose(0, 1)
        return _finalize_corruption(matrix.clone(), pair_mask & ~remove)


@dataclass(frozen=True)
class ContactDeletion:
    contact_threshold: float
    deletion_probability: float

    def __post_init__(self) -> None:
        if self.contact_threshold <= 0 or not 0 <= self.deletion_probability <= 1:
            raise ValueError("Contact deletion requires a positive threshold and probability in [0, 1]")

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_geometry(matrix, pair_mask)
        contacts = (matrix > 0) & (matrix <= self.contact_threshold)
        draws = torch.rand(matrix.shape, device=matrix.device, generator=generator)
        remove = torch.triu(contacts & (draws < self.deletion_probability), diagonal=1)
        remove = remove | remove.transpose(0, 1)
        return _finalize_corruption(matrix.clone(), pair_mask & ~remove)


@dataclass(frozen=True)
class LowRankDistanceDistortion:
    rank: int
    strength: float

    def __post_init__(self) -> None:
        if self.rank < 1 or not 0 <= self.strength <= 1:
            raise ValueError("Low-rank distortion requires rank >= 1 and strength in [0, 1]")

    def __call__(
        self, matrix: torch.Tensor, pair_mask: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del generator
        _validate_geometry(matrix, pair_mask)
        u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = min(self.rank, singular_values.numel())
        approximation = (u[:, :rank] * singular_values[:rank]) @ vh[:rank]
        distorted = torch.lerp(matrix, approximation, self.strength)
        return _finalize_corruption(distorted, pair_mask.clone())


def build_geometry_corruption(config: dict[str, Any] | None) -> GeometryCorruption | None:
    """Build one explicitly configured corruption; distributions remain opt-in."""
    if not config or config.get("type", "none") == "none":
        return None
    kind = str(config["type"])
    if kind == "additive_symmetric_noise":
        return AdditiveSymmetricNoise(float(config["standard_deviation"]))
    if kind == "long_range_pair_mask":
        return LongRangePairMask(int(config["minimum_separation"]), float(config["mask_probability"]))
    if kind == "contact_deletion":
        return ContactDeletion(float(config["contact_threshold"]), float(config["deletion_probability"]))
    if kind == "low_rank_distance_distortion":
        return LowRankDistanceDistortion(int(config["rank"]), float(config["strength"]))
    if kind == "mixture":
        transforms = config.get("transforms")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError("A corruption mixture requires a non-empty transforms list")
        built = tuple(build_geometry_corruption(item) for item in transforms)
        if any(item is None for item in built):
            raise ValueError("A corruption mixture cannot contain a disabled transform")
        return CompositeGeometryCorruption(built)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported geometry corruption type: {kind}")


DATASET_MODES = {
    "sequence_only",
    "geometry_conditioned",
    "geometry_conditioned_with_dropout",
    "corrupted_real_geometry",
}

_LOADER_COLUMNS = (
    "sample_id",
    "schema_version",
    "sequence",
    "sequence_length",
    "matrix_length",
    "matrix_path",
    "practical_training_eligibility",
)
_OPTIONAL_LOADER_COLUMNS = (
    "experimental_method",
    "method",
    "pairing_classification",
    "v3_pairing_classification",
)
_LOADER_BATCH_SIZE = 4096


class _ArrowManifestRows:
    """Random-access projected manifest rows with one-row-group caching."""

    def __init__(self, source: str | Path | pa.Table) -> None:
        self._table: pa.Table | None = None
        self._fragments: list[Any] = []
        self._ends: list[int] = []
        self._cached_fragment_index: int | None = None
        self._cached_local_start = 0
        self._cached_table: pa.Table | None = None
        if isinstance(source, pa.Table):
            missing = sorted(set(_LOADER_COLUMNS) - set(source.column_names))
            if missing:
                raise ValueError(f"Pairing rows are missing loader column(s): {', '.join(missing)}")
            self._table = source.select(_LOADER_COLUMNS)
            self._length = self._table.num_rows
        else:
            dataset = ds.dataset(str(source), format="parquet")
            missing = sorted(set(_LOADER_COLUMNS) - set(dataset.schema.names))
            if missing:
                raise ValueError(f"Pairing manifest is missing loader column(s): {', '.join(missing)}")
            self._columns = (
                *_LOADER_COLUMNS,
                *(name for name in _OPTIONAL_LOADER_COLUMNS if name in dataset.schema.names),
            )
            total = 0
            for fragment in sorted(dataset.get_fragments(), key=lambda item: str(item.path)):
                row_group_fragments = fragment.split_by_row_group()
                for row_group in row_group_fragments or [fragment]:
                    count = int(row_group.count_rows())
                    if count:
                        total += count
                        self._fragments.append(row_group)
                        self._ends.append(total)
            self._length = total
        if self._length == 0:
            raise ValueError("Pairing manifest contains no samples")

    def __len__(self) -> int:
        return self._length

    @property
    def cached_row_count(self) -> int:
        return 0 if self._cached_table is None else self._cached_table.num_rows

    def row(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        if self._table is not None:
            return self._table.slice(index, 1).to_pylist()[0]
        fragment_index = bisect.bisect_right(self._ends, index)
        start = 0 if fragment_index == 0 else self._ends[fragment_index - 1]
        local_index = index - start
        cache_hit = (
            fragment_index == self._cached_fragment_index
            and self._cached_table is not None
            and self._cached_local_start <= local_index < self._cached_local_start + self._cached_table.num_rows
        )
        if not cache_hit:
            batch_start = 0
            scanner = self._fragments[fragment_index].scanner(
                columns=list(self._columns),
                batch_size=_LOADER_BATCH_SIZE,
                use_threads=False,
            )
            for batch in scanner.to_batches():
                if batch_start <= local_index < batch_start + batch.num_rows:
                    self._cached_table = pa.Table.from_batches([batch])
                    self._cached_fragment_index = fragment_index
                    self._cached_local_start = batch_start
                    break
                batch_start += batch.num_rows
        assert self._cached_table is not None
        return self._cached_table.slice(local_index - self._cached_local_start, 1).to_pylist()[0]


class SequenceGeometryDataset(Dataset):
    """Load immutable sequence/matrix pairs without changing stored NPZ samples."""

    def __init__(
        self,
        manifest_path: str | Path | pa.Table,
        *,
        mode: str,
        conditioning_dropout_probability: float = 0.0,
        corruption: GeometryCorruption | None = None,
        seed: int = 0,
        include_metadata: bool = False,
        vocabulary: SequenceGeometryVocabulary | None = None,
    ) -> None:
        if mode not in DATASET_MODES:
            raise ValueError(f"mode must be one of {sorted(DATASET_MODES)}")
        if not 0 <= conditioning_dropout_probability <= 1:
            raise ValueError("conditioning_dropout_probability must be in [0, 1]")
        if mode != "geometry_conditioned_with_dropout" and conditioning_dropout_probability != 0:
            raise ValueError("Conditioning dropout is only valid in geometry_conditioned_with_dropout mode")
        if mode == "corrupted_real_geometry" and corruption is None:
            raise ValueError("corrupted_real_geometry mode requires an explicit corruption")
        if mode != "corrupted_real_geometry" and corruption is not None:
            raise ValueError("Geometry corruption is only valid in corrupted_real_geometry mode")
        self.path = Path(manifest_path) if not isinstance(manifest_path, pa.Table) else None
        self._rows = _ArrowManifestRows(manifest_path)
        self.mode = mode
        self.dropout_probability = float(conditioning_dropout_probability)
        self.corruption = corruption
        self.seed = int(seed)
        self.include_metadata = include_metadata
        self.vocabulary = vocabulary or SequenceGeometryVocabulary()

    def __len__(self) -> int:
        return len(self._rows)

    def row_metadata(self, index: int) -> dict[str, Any]:
        """Return one projected manifest row without opening its NPZ matrix."""
        return dict(self._rows.row(int(index)))

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + int(index)) % (2**63 - 1))
        return generator

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows.row(int(index))
        if str(row["schema_version"]) != PAIRING_SCHEMA_VERSION:
            raise ValueError("Pairing manifest schema version is unsupported or inconsistent")
        if str(row["practical_training_eligibility"]) == "excluded_or_unresolved":
            raise ValueError("Training manifests must not include unresolved pairings")
        sequence = str(row["sequence"])
        token_ids = torch.tensor(self.vocabulary.encode(sequence), dtype=torch.long)
        length = len(sequence)
        if int(row["sequence_length"]) != length or int(row["matrix_length"]) != length:
            raise ValueError(f"Stored pairing length mismatch for {row['sample_id']}")
        pair_mask = torch.ones((length, length), dtype=torch.bool)
        matrix = torch.zeros((length, length), dtype=torch.float32)
        geometry_available = bool(row.get("matrix_path"))
        if self.mode != "sequence_only":
            with np.load(str(row["matrix_path"]), allow_pickle=False) as data:
                if str(data["sequence"]) != sequence:
                    raise ValueError(f"Manifest/NPZ sequence mismatch for {row['sample_id']}")
                matrix = torch.as_tensor(np.asarray(data["distance_matrix"], dtype=np.float32)).clone()
            _validate_geometry(matrix, pair_mask)
            if matrix.shape != (length, length):
                raise ValueError(f"Matrix shape mismatch for {row['sample_id']}: {tuple(matrix.shape)}")
        generator = self._generator(int(index))
        conditioning = geometry_available and self.mode != "sequence_only"
        if self.mode == "geometry_conditioned_with_dropout":
            conditioning = bool(torch.rand((), generator=generator).item() >= self.dropout_probability)
        if self.mode == "corrupted_real_geometry":
            assert self.corruption is not None
            matrix, pair_mask = self.corruption(matrix, pair_mask, generator)
        if not conditioning:
            matrix.zero_()
            pair_mask.zero_()
        item = {
            "sample_id": str(row["sample_id"]),
            "sequence_token_ids": token_ids,
            "sequence_mask": torch.ones(length, dtype=torch.bool),
            "distance_matrix": matrix,
            "pair_mask": pair_mask,
            "length": length,
            "geometry_availability_flag": bool(geometry_available),
            "geometry_conditioning_flag": bool(conditioning),
        }
        if self.include_metadata:
            item["metadata"] = dict(row)
        return item


def collate_sequence_geometry(
    items: list[dict[str, Any]],
    *,
    pad_id: int = 0,
    pad_to_multiple: int = 1,
) -> dict[str, Any]:
    """Pad mixed-length sequence/geometry items while preserving all masks."""
    if not items:
        raise ValueError("Cannot collate an empty sequence-geometry batch")
    if pad_to_multiple < 1:
        raise ValueError("pad_to_multiple must be positive")
    lengths = torch.tensor([int(item["length"]) for item in items], dtype=torch.long)
    maximum_length = int(lengths.max())
    side = ((maximum_length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    tokens = torch.full((len(items), side), int(pad_id), dtype=torch.long)
    sequence_mask = torch.zeros((len(items), side), dtype=torch.bool)
    matrices = torch.zeros((len(items), side, side), dtype=torch.float32)
    pair_mask = torch.zeros((len(items), side, side), dtype=torch.bool)
    for batch_index, item in enumerate(items):
        length = int(item["length"])
        tokens[batch_index, :length] = item["sequence_token_ids"]
        sequence_mask[batch_index, :length] = item["sequence_mask"]
        matrices[batch_index, :length, :length] = item["distance_matrix"]
        pair_mask[batch_index, :length, :length] = item["pair_mask"]
    matrices *= pair_mask.to(matrices.dtype)
    return {
        "sample_ids": [str(item["sample_id"]) for item in items],
        "sequence_token_ids": tokens,
        "sequence_mask": sequence_mask,
        "distance_matrices": matrices,
        "pair_mask": pair_mask,
        "lengths": lengths,
        "geometry_availability_flags": torch.tensor(
            [bool(item["geometry_availability_flag"]) for item in items], dtype=torch.bool
        ),
        "geometry_conditioning_flags": torch.tensor(
            [bool(item["geometry_conditioning_flag"]) for item in items], dtype=torch.bool
        ),
        "metadata": [item.get("metadata") for item in items] if any("metadata" in item for item in items) else None,
    }
