"""Manifest and FASTA datasets for structure-unconditioned sequence modeling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from protein_sequence_generation.utils import sha256_file, sha256_text
from protein_sequence_generation.vocabulary import CANONICAL_AMINO_ACIDS, ProteinVocabulary, normalize_sequence


@dataclass(frozen=True)
class SequenceRecord:
    """One accepted biological sequence and optional provenance metadata."""

    sample_id: str
    sequence: str
    length: int
    metadata: dict[str, Any]


def sequence_hash(sequence: str) -> str:
    """Hash a normalized biological sequence."""
    return sha256_text(normalize_sequence(sequence))


def parse_fasta(path: str | Path) -> pd.DataFrame:
    """Load a FASTA file into a manifest-like DataFrame."""
    records: list[dict[str, Any]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                sequence = "".join(chunks)
                records.append({"sample_id": header.split()[0], "sequence": sequence, "length": len(sequence)})
            header = line[1:].strip() or f"record_{len(records)}"
            chunks = []
        else:
            chunks.append(line)
    if header is not None:
        sequence = "".join(chunks)
        records.append({"sample_id": header.split()[0], "sequence": sequence, "length": len(sequence)})
    return pd.DataFrame(records)


def load_sequence_manifest(path: str | Path) -> pd.DataFrame:
    """Load Parquet, CSV, TSV, or FASTA sequence manifests."""
    src = Path(path)
    suffixes = "".join(src.suffixes).lower()
    if src.suffix.lower() == ".parquet":
        return pd.read_parquet(src)
    if src.suffix.lower() == ".csv":
        return pd.read_csv(src)
    if src.suffix.lower() in {".tsv", ".txt"} and not suffixes.endswith((".fasta", ".fa", ".faa")):
        return pd.read_csv(src, sep="\t")
    if src.suffix.lower() in {".fasta", ".fa", ".faa"}:
        return parse_fasta(src)
    raise ValueError(f"Unsupported sequence input format: {src}")


def validate_records(
    frame: pd.DataFrame,
    *,
    source_path: str | Path,
    vocabulary: ProteinVocabulary,
    min_length: int = 20,
    max_length: int = 500,
    allow_unknown: bool = False,
) -> tuple[list[SequenceRecord], dict[str, Any]]:
    """Validate rows and return accepted records plus an audit summary."""
    if min_length <= 0 or max_length <= 0 or min_length > max_length:
        raise ValueError("min_length and max_length must be positive with min_length <= max_length")
    if "sequence" not in frame.columns:
        raise ValueError("sequence input must contain a sequence column")
    rejection_counts: Counter[str] = Counter()
    records: list[SequenceRecord] = []
    seen_ids: set[str] = set()
    aa_counts: Counter[str] = Counter()
    duplicate_sequences: Counter[str] = Counter()
    for index, row in frame.reset_index(drop=True).iterrows():
        sample_id = str(row.get("sample_id", f"row_{index}"))
        try:
            if sample_id in seen_ids:
                raise ValueError("duplicate_sample_id")
            seen_ids.add(sample_id)
            sequence = normalize_sequence(str(row["sequence"]))
            stored_length = row.get("length")
            if stored_length is not None and not pd.isna(stored_length) and int(stored_length) != len(sequence):
                raise ValueError("length_mismatch")
            if len(sequence) < min_length:
                raise ValueError("length_below_minimum")
            if len(sequence) > max_length:
                raise ValueError("length_above_maximum")
            vocabulary.encode(sequence, allow_unknown=allow_unknown)
        except ValueError as exc:
            reason = str(exc)
            if reason.startswith("Invalid noncanonical residue"):
                reason = "noncanonical_residue"
            rejection_counts[reason] += 1
            continue
        metadata = {column: row[column] for column in frame.columns if column not in {"sequence", "length"}}
        seq_hash = str(row.get("sequence_hash", sequence_hash(sequence)))
        metadata["sequence_hash"] = seq_hash
        records.append(SequenceRecord(sample_id=sample_id, sequence=sequence, length=len(sequence), metadata=metadata))
        aa_counts.update(sequence)
        duplicate_sequences[seq_hash] += 1
    lengths = np.array([record.length for record in records], dtype=np.float64)
    input_path = Path(source_path)
    audit = {
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() and input_path.is_file() else None,
        "accepted_sequences": len(records),
        "rejected_sequences": int(sum(rejection_counts.values())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "length": {
            "min": int(lengths.min()) if lengths.size else None,
            "mean": float(lengths.mean()) if lengths.size else None,
            "median": float(np.median(lengths)) if lengths.size else None,
            "max": int(lengths.max()) if lengths.size else None,
        },
        "amino_acid_frequencies": {aa: int(aa_counts.get(aa, 0)) for aa in CANONICAL_AMINO_ACIDS},
        "unique_sequence_count": int(sum(1 for count in duplicate_sequences.values() if count > 0)),
        "exact_duplicate_count": int(sum(max(count - 1, 0) for count in duplicate_sequences.values())),
        "pdb_entry_count": int(frame["pdb_id"].nunique()) if "pdb_id" in frame.columns else None,
        "cluster_count": int(frame["cluster_id"].nunique()) if "cluster_id" in frame.columns else None,
    }
    return records, audit


def assert_no_split_overlap(records_by_split: dict[str, list[SequenceRecord]]) -> None:
    """Reject exact sample-id or sequence-hash overlap across splits."""
    seen_sample_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for split, records in records_by_split.items():
        for record in records:
            previous = seen_sample_ids.setdefault(record.sample_id, split)
            if previous != split:
                raise ValueError(f"sample_id {record.sample_id} appears in both {previous} and {split}")
            seq_hash = str(record.metadata.get("sequence_hash", sequence_hash(record.sequence)))
            previous_hash = seen_hashes.setdefault(seq_hash, split)
            if previous_hash != split:
                raise ValueError(f"sequence hash {seq_hash} appears in both {previous_hash} and {split}")


class ProteinSequenceDataset(Dataset):
    """Teacher-forced autoregressive protein sequence dataset."""

    def __init__(
        self,
        path: str | Path,
        *,
        vocabulary: ProteinVocabulary | None = None,
        min_length: int = 20,
        max_length: int = 500,
        allow_unknown: bool = False,
    ) -> None:
        self.path = Path(path)
        self.vocabulary = vocabulary or ProteinVocabulary()
        frame = load_sequence_manifest(self.path)
        self.records, self.audit = validate_records(
            frame,
            source_path=self.path,
            vocabulary=self.vocabulary,
            min_length=min_length,
            max_length=max_length,
            allow_unknown=allow_unknown,
        )
        if not self.records:
            raise ValueError(f"No valid sequences found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = self.vocabulary.encode(record.sequence)
        input_ids = [self.vocabulary.bos_id] + encoded[:-1]
        target_ids = encoded
        return {
            "sample_id": record.sample_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "length": torch.tensor(record.length, dtype=torch.long),
            "sequence": record.sequence,
            "metadata": record.metadata,
        }
