"""Independent causality, data, n-gram and diagnostic invariants for E012."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from protein_sequence_generation.e012 import (
    VOCAB,
    baseline_ce,
    batch,
    build_baselines,
    positions,
    prefix_batch,
    shuffle_prefix,
)
from protein_sequence_generation.embeddings import ContinuousLengthEmbedding, RotaryEmbedding
from protein_sequence_generation.metrics import sequence_cross_entropy
from protein_sequence_generation.model import ProteinSequenceTransformer


def tiny():
    torch.manual_seed(12012)
    return ProteinSequenceTransformer(
        {
            "d_model": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "feedforward_dimension": 64,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "max_length": 500,
        }
    ).eval()


@pytest.mark.parametrize("length", [1, 2, 5, 16, 31, 64])
def test_every_future_perturbation_is_hidden(length):
    net = tiny()
    tokens = torch.randint(4, 24, (1, length))
    tokens[0, 0] = 1
    mask = torch.ones_like(tokens, dtype=torch.bool)
    reference = net(tokens, torch.tensor([length]), mask)
    for index in range(length):
        altered = tokens.clone()
        altered[:, index + 1 :] = (altered[:, index + 1 :] - 4 + 7) % 20 + 4
        result = net(altered, torch.tensor([length]), mask)
        torch.testing.assert_close(result[:, : index + 1], reference[:, : index + 1], atol=2e-6, rtol=2e-6)


def test_padding_and_bos_earlier_residue_effects():
    net = tiny()
    tokens = torch.tensor([[1, 4, 5, 6]])
    lengths = torch.tensor([20])
    reference = net(tokens, lengths, torch.ones_like(tokens, dtype=torch.bool))
    padded = torch.cat([tokens, torch.tensor([[12, 13, 14]])], 1)
    mask = torch.tensor([[True] * 4 + [False] * 3])
    torch.testing.assert_close(net(padded, lengths, mask)[:, :4], reference, atol=2e-6, rtol=2e-6)
    for index in [0, 1]:
        changed = tokens.clone()
        changed[:, index] = 15
        assert not torch.allclose(
            net(changed, lengths, torch.ones_like(tokens, dtype=torch.bool))[:, -1], reference[:, -1]
        )


@pytest.mark.parametrize("length", [1, 7, 31, 500])
def test_actual_causal_mask_every_length(monkeypatch, length):
    from torch.nn import functional as functional

    original = functional.scaled_dot_product_attention
    captured = []

    def spy(q, k, v, **kwargs):
        captured.append(kwargs["attn_mask"])
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(functional, "scaled_dot_product_attention", spy)
    tokens = torch.ones((2, length), dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    if length > 1:
        mask[1, -1] = False
    tiny()(tokens, torch.tensor([length, max(1, length - 1)]), mask)
    expected = torch.ones(length, length, dtype=torch.bool).tril()[None, None] & mask[:, None, None]
    assert all(torch.equal(c, expected) for c in captured)


def test_rope_math_offsets_and_norm():
    rope = RotaryEmbedding(8, max_length=500)
    q = torch.randn(2, 3, 500, 8)
    k = torch.randn_like(q)
    rotated_q, rotated_k = rope(q, k)
    torch.testing.assert_close(rotated_q[:, :, 0], q[:, :, 0])
    torch.testing.assert_close(rotated_k.square().sum(-1), k.square().sum(-1), atol=3e-6, rtol=2e-6)
    i = 127
    even, odd = q[..., i, 0::2], q[..., i, 1::2]
    expected = torch.stack(
        [even * rope.cos[i] - odd * rope.sin[i], even * rope.sin[i] + odd * rope.cos[i]], -1
    ).flatten(-2)
    torch.testing.assert_close(rotated_q[..., i, :], expected)
    with pytest.raises(ValueError):
        RotaryEmbedding(7, max_length=500)


def test_exact_teacher_forcing_and_loss_padding():
    rows = [{"sample_id": "a", "sequence": "ACD"}, {"sample_id": "b", "sequence": "EF"}]
    data = batch(rows, "cpu")
    assert data["input_ids"].tolist() == [[1, 4, 5], [1, 7, 0]]
    assert data["target_ids"].tolist() == [[4, 5, 6], [7, 8, 0]]
    logits = torch.randn(2, 3, 24, requires_grad=True)
    loss = sequence_cross_entropy(logits, data["target_ids"], data["attention_mask"])
    loss.backward()
    assert not logits.grad[1, 2].any()
    individual = [
        torch.nn.functional.cross_entropy(logits[i, :n], data["target_ids"][i, :n]) for i, n in enumerate([3, 2])
    ]
    torch.testing.assert_close(loss, torch.stack(individual).mean())


def test_length_embedding_determinism_and_neutralization():
    emb = ContinuousLengthEmbedding(32, max_length=500)
    lengths = torch.tensor([20, 64, 500])
    assert torch.equal(emb(lengths), emb(lengths))
    assert not torch.equal(emb(lengths)[0], emb(lengths)[2])
    net = tiny()
    data = batch([{"sample_id": "a", "sequence": "ACDE"}], "cpu")
    normal = net(data["input_ids"], data["lengths"], data["attention_mask"])
    hook = net.length_embedding.register_forward_hook(lambda _m, _i, v: torch.zeros_like(v))
    neutral = net(data["input_ids"], data["lengths"], data["attention_mask"])
    hook.remove()
    assert not torch.allclose(normal, neutral)


@pytest.mark.parametrize("length", [20, 64, 128, 256, 384, 500])
def test_positions_and_shuffle_are_deterministic_and_prefix_only(length):
    chosen = positions(length)
    assert chosen == positions(length) and len(chosen) <= 16
    assert chosen == sorted(set(chosen)) and all(2 <= p <= length - 2 for p in chosen)
    tokens = list(range(length))
    for p in chosen:
        altered = shuffle_prefix(tokens, p, "identity")
        assert Counter(altered) == Counter(tokens[:p])
        assert max(altered) < p
        assert altered == shuffle_prefix(tokens, p, "identity")


@pytest.mark.parametrize("window", [1, 4, 8, 16, 32, 64])
def test_windows_preserve_absolute_positions_and_remove_earlier_dependencies(window):
    net = tiny()
    row = {"sample_id": "a", "sequence": "ACDEFGHIKLMNPQRSTVWY" * 4}
    inputs, lengths, mask, indices, target = prefix_batch([(row, 70)], "cpu", window=window)
    assert indices.item() == 70 and lengths.item() == 80
    assert target.item() == VOCAB.encode(row["sequence"])[70]
    assert mask[0, 71 - window : 71].all() and not mask[0, : 71 - window].any()
    assert inputs.shape[1] == 71
    reference = net(inputs, lengths, mask)
    changed = inputs.clone()
    changed[~mask] = 19
    torch.testing.assert_close(reference[:, 70], net(changed, lengths, mask)[:, 70], atol=2e-6, rtol=2e-6)


def test_full_prefix_matches_teacher_forced_current_logits():
    row = {"sample_id": "a", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    net = tiny()
    data = batch([row], "cpu")
    full = net(data["input_ids"], data["lengths"], data["attention_mask"])
    for index in [2, 8, 17]:
        x, n, m, _, _ = prefix_batch([(row, index)], "cpu")
        torch.testing.assert_close(net(x, n, m)[:, index], full[:, index], atol=2e-6, rtol=2e-6)


def test_ngram_bos_counts_train_only_and_smoothing():
    rows = [{"split": "train", "sequence": "AC" * 10}, {"split": "train", "sequence": "AA" * 10}]
    baseline = build_baselines(rows)
    bigram, trigram = np.array(baseline["counts"]["bigram"]), np.array(baseline["counts"]["trigram"])
    assert bigram[20, 0] == trigram[20, 20, 0] == 2
    assert trigram[20, 0, 1] == 1 and trigram[20, 0, 0] == 1
    assert bigram.sum() == trigram.sum() == 40
    assert all(np.isfinite(v) for v in baseline_ce("WY" * 10, baseline).values())
    with pytest.raises(ValueError, match="TRAIN"):
        build_baselines([{"split": "validation", "sequence": "AC" * 10}])


def test_model_instantiates_only_causal_sequence_modules():
    names = {type(m).__name__ for m in tiny().modules()}
    assert not names & {"SequenceContextTransformer", "E006RichGeometryCoDesign", "GeometryEncoder", "DistanceUNet"}
    assert {"CausalSelfAttention", "RotaryEmbedding", "ContinuousLengthEmbedding"} <= names


def test_sequence_reader_rejects_geometry_schema(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import scripts.run_e012_causal_rope as runner

    monkeypatch.setattr(runner, "OUT", tmp_path)
    pq.write_table(
        pa.Table.from_pylist(
            [{"sample_id": "a", "split": "train", "sequence": "AC", "length": 2, "coordinates": [1.0]}]
        ),
        tmp_path / "train.parquet",
    )
    with pytest.raises(AssertionError):
        runner.load_rows("train")


def test_identity_overlap_rejected():
    from protein_sequence_generation.dataset import SequenceRecord, assert_no_split_overlap

    r = SequenceRecord("a", "AC", 2, {})
    with pytest.raises(ValueError):
        assert_no_split_overlap({"train": [r], "validation": [r]})


def test_actual_bounded_evaluator_matches_manual_prefix_ce(monkeypatch):
    from contextlib import nullcontext

    import scripts.run_e012_causal_rope as runner

    original = prefix_batch
    monkeypatch.setattr(runner, "prefix_batch", lambda cases, _device, **kw: original(cases, "cpu", **kw))
    monkeypatch.setattr(runner, "autocast", nullcontext)
    row = {"sample_id": "a", "sequence": "ACDEFGHIKLMNPQRSTVWY"}
    net = tiny()
    selected = [2, 8, 17]
    result = runner.diagnostics(net, [row], {"a": selected})["a"]
    values = []
    for index in selected:
        inputs, lengths, mask, _, target = original([(row, index)], "cpu")
        values.append(float(torch.nn.functional.cross_entropy(net(inputs, lengths, mask)[:, index], target).detach()))
    assert result["prefix_normal"] == pytest.approx(float(np.mean(values)), abs=1e-6)
    assert 0 <= result["top1_change"] <= 1 and 0 <= result["top3_change"] <= 1
    assert result["kl"] >= -1e-6 and result["js"] >= -1e-6
    assert all(np.isfinite(v) for v in result.values())


@pytest.mark.parametrize("active", [True, False])
def test_research_owner_guard_blocks_training_but_allows_diagnostics(monkeypatch, active):
    import scripts.run_e012_causal_rope as runner

    command = (
        "77 python -u scripts/run_e010_phase4d_recurrent_capacity_v3.py train"
        if active
        else "77 python scripts/diagnose_e010_phase4d.py"
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_a, **_k: command)
    if active:
        with pytest.raises(RuntimeError, match="CUDA execution deferred"):
            runner.check_known_research_owners()
    else:
        runner.check_known_research_owners()
