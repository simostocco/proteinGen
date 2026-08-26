"""Processed distance-map sample I/O."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from protein_distance_diffusion.constants import AA_TO_TOKEN


@dataclass(frozen=True)
class ProteinSample:
    """One accepted chain with ordered C-alpha coordinates."""

    sample_id: str
    pdb_id: str
    chain_id: str
    sequence: str
    residue_ids: list[str]
    ca_coordinates: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StructureRejection:
    """A deterministic reason a source file or chain was not accepted."""

    source_file: str
    reason: str
    message: str
    pdb_id: str | None = None
    chain_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "pdb_id": self.pdb_id,
            "chain_id": self.chain_id,
            "reason": self.reason,
            "message": self.message,
        }


def compute_distance_matrix(ca_coordinates: np.ndarray) -> np.ndarray:
    """Return a symmetric C-alpha distance matrix in Angstrom."""
    coords = np.asarray(ca_coordinates, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("ca_coordinates must have shape (N, 3)")
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float32), dtype=np.float32)
    distances = 0.5 * (distances + distances.T)
    np.fill_diagonal(distances, 0.0)
    return distances.astype(np.float32, copy=False)


def _sequence_tokens(sequence: str) -> np.ndarray:
    return np.asarray([AA_TO_TOKEN[aa] for aa in sequence], dtype=np.int64)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_processed_sample(sample: ProteinSample, samples_dir: str | Path) -> dict[str, Any]:
    """Write one sample `.npz` atomically and return its manifest row."""
    samples_path = Path(samples_dir)
    sample_path = samples_path / f"{sample.sample_id}.npz"
    coords = np.asarray(sample.ca_coordinates, dtype=np.float32)
    if coords.shape != (len(sample.sequence), 3):
        raise ValueError("sample coordinates must have shape (len(sequence), 3)")
    distance = compute_distance_matrix(coords)
    residue_mask = np.ones(len(sample.sequence), dtype=np.bool_)
    metadata = dict(sample.metadata)
    metadata.setdefault("original_chain_length", len(sample.sequence))
    metadata.setdefault("residue_ids", list(sample.residue_ids))
    _atomic_npz(
        sample_path,
        sample_id=np.asarray(sample.sample_id),
        pdb_id=np.asarray(sample.pdb_id),
        chain_id=np.asarray(sample.chain_id),
        sequence=np.asarray(sample.sequence),
        sequence_tokens=_sequence_tokens(sample.sequence),
        residue_ids=np.asarray(sample.residue_ids),
        residue_mask=residue_mask,
        ca_coordinates=coords,
        distance_matrix=distance,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "pdb_id": sample.pdb_id,
        "chain_id": sample.chain_id,
        "sequence": sample.sequence,
        "length": len(sample.sequence),
        "path": str(sample_path),
        "experimental_method": metadata.get("experimental_method"),
        "resolution_angstrom": metadata.get("resolution_angstrom"),
        "source_file": metadata.get("source_file"),
    }
    for key in (
        "structure_format",
        "model_number",
        "original_chain_length",
        "retained_start_label_seq_id",
        "retained_end_label_seq_id",
        "trimmed_n_terminal_residues",
        "trimmed_c_terminal_residues",
        "terminal_trimming_applied",
        "missing_calpha_policy",
    ):
        if key in metadata:
            row[key] = metadata[key]
    return row


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a manifest parquet file."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(dst, index=False)
    return dst


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a parquet or CSV manifest."""
    src = Path(path)
    if src.suffix.lower() == ".csv":
        return pd.read_csv(src)
    return pd.read_parquet(src)
