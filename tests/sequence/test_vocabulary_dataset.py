"""Sequence vocabulary and dataset tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from protein_sequence_generation.collate import collate_sequences
from protein_sequence_generation.dataset import (
    ProteinSequenceDataset,
    assert_no_split_overlap,
    load_sequence_manifest,
    parse_fasta,
    validate_records,
)
from protein_sequence_generation.vocabulary import CANONICAL_AMINO_ACIDS, ProteinVocabulary


def test_vocabulary_deterministic_round_trip_and_invalid_residue(tmp_path: Path) -> None:
    vocab = ProteinVocabulary()
    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.unk_id == 3
    assert len(vocab.tokens) == 24
    assert vocab.decode(vocab.encode("ACDE")) == "ACDE"
    with pytest.raises(ValueError, match="Invalid noncanonical"):
        vocab.encode("ABZ")
    path = tmp_path / "vocab.json"
    vocab.to_json(path)
    assert ProteinVocabulary.from_json(path).tokens == vocab.tokens


def test_dataset_shift_padding_and_masks(tmp_path: Path) -> None:
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {"sample_id": "a", "sequence": "ACDE", "length": 4},
            {"sample_id": "b", "sequence": "ACDEF", "length": 5},
        ]
    ).to_parquet(manifest)
    vocab = ProteinVocabulary()
    dataset = ProteinSequenceDataset(manifest, vocabulary=vocab, min_length=1, max_length=10)
    item = dataset[0]
    assert item["input_ids"].tolist() == [vocab.bos_id] + vocab.encode("ACD")
    assert item["target_ids"].tolist() == vocab.encode("ACDE")
    batch = collate_sequences([dataset[0], dataset[1]], vocabulary=vocab)
    assert batch["input_ids"].shape == torch.Size([2, 5])
    assert batch["attention_mask"].tolist() == [[True, True, True, True, False], [True] * 5]
    assert batch["target_ids"][0, 4].item() == vocab.pad_id


def test_dataset_validation_rejections_and_duplicates(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "ok", "sequence": "ACDE", "length": 4},
            {"sample_id": "bad_len", "sequence": "ACDE", "length": 5},
            {"sample_id": "bad_res", "sequence": "ACDX", "length": 4},
            {"sample_id": "short", "sequence": "AC", "length": 2},
            {"sample_id": "dup", "sequence": "ACDE", "length": 4},
        ]
    )
    path = tmp_path / "manifest.parquet"
    frame.to_parquet(path)
    records, audit = validate_records(
        frame,
        source_path=path,
        vocabulary=ProteinVocabulary(),
        min_length=3,
        max_length=10,
    )
    assert [record.sample_id for record in records] == ["ok", "dup"]
    assert audit["exact_duplicate_count"] == 1
    assert audit["rejection_counts"] == {
        "length_below_minimum": 1,
        "length_mismatch": 1,
        "noncanonical_residue": 1,
    }


def test_split_overlap_detection(tmp_path: Path) -> None:
    vocab = ProteinVocabulary()
    left_frame = pd.DataFrame([{"sample_id": "a", "sequence": "ACDE", "length": 4}])
    right_frame = pd.DataFrame([{"sample_id": "b", "sequence": "ACDE", "length": 4}])
    left, _ = validate_records(left_frame, source_path=tmp_path / "l.parquet", vocabulary=vocab, min_length=1)
    right, _ = validate_records(right_frame, source_path=tmp_path / "r.parquet", vocabulary=vocab, min_length=1)
    with pytest.raises(ValueError, match="sequence hash"):
        assert_no_split_overlap({"train": left, "validation": right})


def test_fasta_loading(tmp_path: Path) -> None:
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">seq1\nACDE\n>seq2 description\nFGHI\n")
    frame = parse_fasta(fasta)
    assert frame["sample_id"].tolist() == ["seq1", "seq2"]
    assert load_sequence_manifest(fasta)["sequence"].tolist() == ["ACDE", "FGHI"]
    assert set("".join(frame["sequence"])) <= set(CANONICAL_AMINO_ACIDS)
