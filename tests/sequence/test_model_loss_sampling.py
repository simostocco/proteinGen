"""Sequence Transformer model, loss, and sampling tests."""

from __future__ import annotations

import math

import pytest
import torch

from protein_sequence_generation.metrics import (
    generated_sequence_metrics,
    jensen_shannon_divergence,
    max_homopolymer_run,
    sequence_cross_entropy,
)
from protein_sequence_generation.model import ProteinSequenceTransformer, SequenceTransformerConfig
from protein_sequence_generation.sampling import generate_sequences, validate_sampling_args
from protein_sequence_generation.vocabulary import CANONICAL_AMINO_ACIDS, ProteinVocabulary


def tiny_model() -> ProteinSequenceTransformer:
    return ProteinSequenceTransformer(
        SequenceTransformerConfig(
            vocabulary_size=24,
            d_model=32,
            num_layers=2,
            num_attention_heads=4,
            feedforward_dimension=64,
            dropout=0.0,
            attention_dropout=0.0,
            max_length=16,
        )
    ).eval()


def test_model_shapes_causal_padding_and_length_conditioning() -> None:
    model = tiny_model()
    input_ids = torch.tensor([[1, 4, 5, 6, 0], [1, 4, 5, 0, 0]])
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])
    lengths = torch.tensor([5, 3])
    logits = model(input_ids=input_ids, lengths=lengths, attention_mask=mask)
    assert logits.shape == torch.Size([2, 5, 24])

    changed_future = input_ids.clone()
    changed_future[0, 4] = 12
    logits_changed = model(input_ids=changed_future, lengths=lengths, attention_mask=mask)
    torch.testing.assert_close(logits[0, :4], logits_changed[0, :4], atol=1e-5, rtol=1e-5)

    changed_padding = input_ids.clone()
    changed_padding[1, 3:] = torch.tensor([12, 13])
    logits_padding = model(input_ids=changed_padding, lengths=lengths, attention_mask=mask)
    torch.testing.assert_close(logits[1, :3], logits_padding[1, :3], atol=1e-5, rtol=1e-5)

    logits_length = model(input_ids=input_ids, lengths=torch.tensor([6, 6]), attention_mask=mask)
    assert not torch.allclose(logits[:, :3], logits_length[:, :3])


def test_loss_reductions_and_perplexity_definition() -> None:
    logits = torch.log(torch.tensor([[[0.1, 0.9], [0.8, 0.2]], [[0.5, 0.5], [0.5, 0.5]]]))
    targets = torch.tensor([[1, 0], [0, 0]])
    mask = torch.tensor([[True, True], [True, False]])
    token_mean = sequence_cross_entropy(logits, targets, mask, reduction="token_mean")
    seq_mean = sequence_cross_entropy(logits, targets, mask, reduction="sequence_mean")
    expected_token = (-math.log(0.9) - math.log(0.8) - math.log(0.5)) / 3
    expected_seq = ((-math.log(0.9) - math.log(0.8)) / 2 + -math.log(0.5)) / 2
    assert token_mean.item() == pytest.approx(expected_token)
    assert seq_mean.item() == pytest.approx(expected_seq)
    assert math.exp(token_mean.item()) == pytest.approx(math.exp(expected_token))


def test_sampling_exact_length_canonical_and_deterministic() -> None:
    vocab = ProteinVocabulary()
    model = tiny_model()
    seqs1 = generate_sequences(
        model,
        vocabulary=vocab,
        length=7,
        num_sequences=3,
        seed=123,
        temperature=1.0,
        top_p=0.95,
    )
    seqs2 = generate_sequences(
        model,
        vocabulary=vocab,
        length=7,
        num_sequences=3,
        seed=123,
        temperature=1.0,
        top_p=0.95,
    )
    assert seqs1 == seqs2
    assert all(len(sequence) == 7 for sequence in seqs1)
    assert all(set(sequence) <= set(CANONICAL_AMINO_ACIDS) for sequence in seqs1)
    assert all("<" not in sequence for sequence in seqs1)


def test_sampling_validation() -> None:
    with pytest.raises(ValueError, match="temperature"):
        validate_sampling_args(length=3, temperature=0.0, top_k=None, top_p=0.9)
    with pytest.raises(ValueError, match="top_p"):
        validate_sampling_args(length=3, temperature=1.0, top_k=None, top_p=1.5)
    with pytest.raises(ValueError, match="top_k"):
        validate_sampling_args(length=3, temperature=1.0, top_k=0, top_p=0.9)


def test_generated_sequence_metrics() -> None:
    metrics = generated_sequence_metrics(["AAAA", "ACDE", "ACDE"], training_sequences=["ACDE", "FGHI"])
    assert metrics["exact_unique_count"] == 2
    assert metrics["duplicate_fraction"] == pytest.approx(1 / 3)
    assert metrics["nearest_exact_match_count"] == 2
    assert max_homopolymer_run("AAACCC") == 3
    assert jensen_shannon_divergence(metrics["amino_acid_frequencies"], metrics["training_amino_acid_frequencies"]) >= 0
