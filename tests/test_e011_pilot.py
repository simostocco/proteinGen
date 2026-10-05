from __future__ import annotations

import torch

from protein_sequence_generation.context import SequenceContextTransformer, objective, paired_forwards
from scripts.run_e011_sequence_context_pilot import REGIMES, batches, training_conditions


def rows():
    return [
        {"sample_id": f"sample-{i}", "sequence": "AC" * n, "split": "train", "token_ids": [2, 3] * n}
        for i, n in enumerate([10, 32, 64, 100, 128, 160, 192, 200, 250] * 8)
    ]


def test_schedule_exact_coverage_determinism_and_regime():
    records = rows()
    schedule = list(batches(records, 0))
    flattened = [i for batch in schedule for i in batch]
    assert sorted(flattened) == list(range(len(records)))
    assert schedule == list(batches(records, 0))
    assert schedule != list(batches(records, 1))
    for batch in schedule:
        regime = next(
            (maximum, count)
            for maximum, count in REGIMES
            if max(len(records[i]["token_ids"]) for i in batch) <= maximum
        )
        assert len(batch) <= regime[1]


def test_training_context_reproduces_no_target_leak_and_optimizer_mutates():
    model = SequenceContextTransformer(d_model=16, layers=2, heads=4, ffn=32, dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = rows()[:2]
    first = training_conditions(batch, 0.3, 0)
    second = training_conditions(batch, 0.3, 0)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    targets, valid, selected, normal, shuffled = first
    assert (normal[selected] == 1).all() and (shuffled[selected] == 1).all()
    assert (normal[~valid] == 0).all()
    before = model.sequence_output.weight.detach().clone()
    n, s = paired_forwards(model, normal, shuffled, valid)
    loss = objective(n, s, targets, selected, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    assert not torch.equal(before, model.sequence_output.weight)
    assert not s.requires_grad


def test_batched_diagnostics_match_sealed_evaluator():
    from protein_sequence_generation.context import evaluate_panel, train_unigrams
    from scripts.run_e011_sequence_context_pilot import evaluate

    batch = rows()[:2]
    donors = {batch[0]["sample_id"]: batch[1]["sample_id"], batch[1]["sample_id"]: batch[0]["sample_id"]}
    # Donor lengths must share a stratum; 20 and 64 do.
    baseline = train_unigrams(batch)
    model = SequenceContextTransformer(d_model=16, layers=2, heads=4, ffn=32, dropout=0.1).eval()
    actual = evaluate(model, batch, donors, baseline)
    for fraction in (0.15, 0.3, 0.5):
        expected = evaluate_panel(model, batch, donors, baseline, fraction, seed=6111)
        observed = {r["sample_id"]: r for r in actual[str(fraction)]}
        for record in expected:
            for name, value in record["ce"].items():
                assert abs(observed[record["sample_id"]]["ce"][name] - value) < 1e-6
