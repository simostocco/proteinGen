"""Matched-panel evaluation and non-training guarantees for E012 V3."""

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from protein_sequence_generation.e012 import batch, positions, prefix_batch
from protein_sequence_generation.e012_generalization import (
    bulk_prefix_batch,
    evaluation_only_guard,
    metrics,
    relative_bins,
    select_train_panel,
)
from scripts import audit_e012_generalization as audit
from scripts import run_e012_causal_rope as old


def row(sid, length, split="train"):
    return {"sample_id": sid, "length": length, "sequence": ("ACDEFGHIKLMNPQRSTVWY" * 25)[:length], "split": split}


def test_panel_is_deterministic_train_only_and_length_matched():
    train = [row(f"t{s}_{i}", length) for s, length in enumerate([20, 65, 129, 257, 385]) for i in range(5)]
    held = [row(f"v{s}", n, "validation") for s, n in enumerate([20, 65, 129, 257, 385])]
    first = select_train_panel(train, held, {r["sample_id"] for r in held})
    assert first == select_train_panel(list(reversed(train)), held, {r["sample_id"] for r in held})
    assert first["stratum_counts"] == [1] * 5
    assert len(first["sample_ids"]) == len(set(first["sample_ids"])) == 5
    assert set(first["sample_ids"]) <= {r["sample_id"] for r in train}
    assert first["selection_uses_performance"] is False
    with pytest.raises(ValueError, match="overlap"):
        select_train_panel(train, held, {train[0]["sample_id"]})
    with pytest.raises(ValueError, match="TRAIN-only"):
        select_train_panel([row("bad", 20, "validation")], held, set())


@pytest.mark.parametrize("length", [20, 64, 65, 128, 129, 256, 257, 384, 385, 500])
def test_relative_position_bins_exact_historical_parity(length):
    assert np.array_equal(relative_bins(length), np.minimum(np.arange(length) * 10 // length, 9))


@pytest.mark.parametrize("window", [None, 1, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("shuffle", [False, True])
def test_bulk_prefix_constructor_exact_historical_parity(window, shuffle):
    cases = [(row("a", 20), i) for i in positions(20)] + [(row("b", 500), i) for i in positions(500)]
    expected = prefix_batch(cases, "cpu", shuffled=shuffle, window=window)
    actual = bulk_prefix_batch(cases, "cpu", shuffled=shuffle, window=window)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True))


def test_guard_blocks_optimizer_backward_and_checkpoint_writes(tmp_path):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    with evaluation_only_guard():
        assert not torch.is_grad_enabled()
        with pytest.raises(RuntimeError, match="evaluation-only"):
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
        with pytest.raises(RuntimeError, match="evaluation-only"):
            optimizer.step()
        with pytest.raises(RuntimeError, match="evaluation-only"):
            torch.ones(1, requires_grad=True).backward()
        with pytest.raises(RuntimeError, match="evaluation-only"):
            torch.save({}, tmp_path / "forbidden.pt")
    assert not (tmp_path / "forbidden.pt").exists()
    assert torch.is_grad_enabled()
    optimizer.step()


def test_primary_and_token_reductions_do_not_confuse_lengths():
    records = []
    for n, ce in [(20, 1.0), (100, 3.0)]:
        records.append(
            {
                "length": n,
                "normal": ce,
                "loss_sum": n * ce,
                "top1": 0.1,
                "top3": 0.3,
                "relative_position_ce": [ce] * 10,
                "predictive_entropy": 2.0,
                "max_probability": 0.3,
                "correct_token_probability": 0.2,
                "aa_counts": [n] + [0] * 19,
                "target_aa_frequencies": [1.0] + [0.0] * 19,
                "mean_predicted_aa_probabilities": [0.049] * 20,
                "aa_nll_contributions": [ce] + [0.0] * 19,
            }
        )
    result = metrics(records)
    assert result["equal_protein_ce"] == 2.0
    assert result["token_weighted_ce"] == 320 / 120
    assert sum(result["equal_protein_aa_nll_contributions"]) == result["equal_protein_ce"]
    assert sum(result["token_weighted_aa_nll_contributions"]) == result["token_weighted_ce"]
    assert result["conditional_token_nll_by_aa"][0] == 320 / 120
    assert result["equal_protein_special_probability_mass"] == pytest.approx(0.02)


def test_audit_calls_same_historical_likelihood_for_every_panel_and_never_training_runner():
    source = inspect.getsource(audit.run)
    assert 'for panel in ["train", "primary", "independent"]' in source
    assert "old.likelihood(net, rows, baselines)" in source
    assert "net.requires_grad_(False)" in source
    assert "evaluation_only_guard()" in source
    assert "opt.step" not in source and "old.train(" not in source and "old.restore(" not in source
    assert "old.weights_hash(net) == fingerprint" in source
    assert "net.eval()" in inspect.getsource(old.likelihood)
    assert "@torch.no_grad()" in inspect.getsource(old.likelihood)


def test_frozen_real_panel_membership_hashes_and_historical_protection():
    if not (audit.REPORT / "contract.json").exists():
        pytest.skip("selection contract not yet prepared")
    # Isolate the large real-data hash/membership check so its Python/Arrow arenas
    # cannot invalidate unrelated tests' 2GiB per-process RSS guards.
    code = """
from scripts import audit_e012_generalization as audit
from scripts import run_e012_causal_rope as old
from protein_sequence_generation.e012_generalization import select_train_panel
contract = audit.read(audit.REPORT / 'contract.json')
for name, digest in contract['source_hashes'].items():
    assert old.sha(audit.ROOT / name) == digest
training, panels = audit.populations()
selected = select_train_panel(training, panels['primary'], {r['sample_id'] for r in old.load_rows('validation')})
panel = audit.read(audit.REPORT / 'train_panel.json')
assert selected['sample_ids'] == panel['sample_ids']
assert panel['stratum_counts'] == [410,410,410,409,409]
assert all(len(v) == 2048 for v in panels.values())
assert audit.verify_historical()
"""
    env = {**os.environ, "PYTHONPATH": str(audit.ROOT / "src") + os.pathsep + str(audit.ROOT)}
    result = subprocess.run([sys.executable, "-c", code], cwd=audit.ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_log_semantics_are_explicit_in_original_source():
    from scripts import run_e012_continuation as continuation

    source = inspect.getsource(continuation.train)
    assert "net.train()" in source and 'ce += stats["ce"] * len(chosen) / 64' in source
    assert "sequence_mean_ce" in source
    assert 'reduction="sequence_mean"' in inspect.getsource(continuation.backward)
    assert "CE={ce:.5f}" in source
    assert "rolling" not in source


def test_teacher_forcing_padding_is_shared():
    inputs = batch([row("a", 20), row("b", 50)], "cpu")
    assert set(inputs) == {"input_ids", "target_ids", "lengths", "attention_mask"}
    assert not inputs["attention_mask"][0, 20:].any()
    assert torch.equal(inputs["input_ids"][1, 1:50], inputs["target_ids"][1, :49])
