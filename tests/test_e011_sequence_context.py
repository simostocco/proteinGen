import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from protein_sequence_generation.context import (
    SequenceContextTransformer,
    collate,
    conditions,
    deterministic_mask,
    objective,
    paired_forwards,
    paired_gate,
    protein_ce,
    sequence_rows,
    train_unigrams,
)


def tiny():
    return SequenceContextTransformer(d_model=16, layers=2, heads=4, ffn=32, dropout=0.1)


def inputs():
    t = torch.tensor([[2, 3, 4, 5, 6] * 4, [7, 8, 9, 10, 11] * 4])
    v = torch.ones_like(t, dtype=torch.bool)
    m = deterministic_mask(t, v, ["a", "b"], 0.3)
    donors = [{"sample_id": "b", "token_ids": t[1].tolist()}, {"sample_id": "a", "token_ids": t[0].tolist()}]
    return t, v, m, conditions(t, v, m, ["a", "b"], donors)


def test_historical_parity_full_capacity():
    from protein_distance_diffusion.models.rich_codesign import E006RichGeometryCoDesign

    historical = E006RichGeometryCoDesign(
        sequence_hidden_dim=256,
        sequence_layers=8,
        sequence_heads=8,
        sequence_feedforward_dim=1024,
        sequence_dropout=0.1,
        geometry_model={
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "group_norm_groups": 1,
            "attention_heads": 1,
        },
    ).eval()
    model = SequenceContextTransformer().eval()
    model.copy_historical_weights(historical)
    t, v, m, c = inputs()
    v[1, 15:] = False
    with torch.no_grad():
        torch.testing.assert_close(
            model(c["normal"], v), historical.forward_sequence_pretraining(c["normal"], v)[..., 2:22], rtol=0, atol=0
        )
    assert not any("geometry" in n or "fusion" in n or "length_embedding" in n for n, _ in model.named_modules())


def test_padding_invariance():
    model = tiny().eval()
    t, v, _, _ = inputs()
    v[:, 15:] = False
    changed = t.clone()
    changed[:, 15:] = 21
    with torch.no_grad():
        torch.testing.assert_close(model(t, v)[v], model(changed, v)[v])
        torch.testing.assert_close(model(t[:, :15], v[:, :15]), model(t, v)[:, :15], atol=1e-6, rtol=1e-5)


def test_conditions_hidden_shuffle_null_and_donor():
    t, v, m, c = inputs()
    for condition in c.values():
        assert (condition[m] == 1).all()
    assert (c["null_context"][v] == 1).all()
    for i in range(2):
        assert sorted(c["normal"][i, ~m[i]].tolist()) == sorted(c["visible_shuffle"][i, ~m[i]].tolist())
    donors = [{"sample_id": "a", "token_ids": t[0].tolist()}] * 2
    with pytest.raises(ValueError):
        conditions(t, v, m, ["a", "b"], donors)
    changed = t.clone()
    changed[m] = 21
    # Hidden target mutation cannot change any condition.
    donors = [{"sample_id": "b", "token_ids": t[1].tolist()}, {"sample_id": "a", "token_ids": t[0].tolist()}]
    other = conditions(changed, v, m, ["a", "b"], donors)
    for name in c:
        assert torch.equal(c[name], other[name])


def test_mask_reproducible_and_batch_independent():
    t, v, m, _ = inputs()
    assert torch.equal(m, deterministic_mask(t, v, ["a", "b"], 0.3))
    assert torch.equal(m[1:], deterministic_mask(t[1:], v[1:], ["b"], 0.3))


def test_equal_protein_loss_and_detached_shuffle():
    t, v, m, _ = inputs()
    v[1, 10:] = False
    m[1] = True
    t[1, 0] = 1  # Noncanonical never enters the loss.
    n = torch.randn(2, 20, 20, requires_grad=True)
    s = torch.zeros_like(n, requires_grad=True)
    loss = objective(n, s, t, m, v)
    loss.backward()
    chosen = m & v & (t >= 2)
    assert (n.grad[~chosen] == 0).all()
    assert s.grad is None
    values = protein_ce(n.detach(), t, m, v)
    assert len(values) == 2
    reference = torch.stack([torch.nn.functional.cross_entropy(n[i, chosen[i]], t[i, chosen[i]] - 2) for i in range(2)])
    torch.testing.assert_close(values, reference)


def test_paired_dropout_and_single_rng_advance():
    model = tiny().train()
    t, v, _, _ = inputs()
    state = torch.get_rng_state()
    direct = model(t, v)
    after = torch.get_rng_state()
    torch.set_rng_state(state)
    n, s = paired_forwards(model, t, t, v)
    torch.testing.assert_close(n, s, atol=0, rtol=0)
    torch.testing.assert_close(n, direct, atol=0, rtol=0)
    assert torch.equal(after, torch.get_rng_state())
    assert not s.requires_grad


def test_train_only_baselines_and_gates():
    row = {"sample_id": "a", "split": "train", "token_ids": [2] * 20}
    b = train_unigrams([row])
    assert b["global"][0] > b["global"][1]
    with pytest.raises(ValueError):
        train_unigrams([{**row, "split": "validation"}])
    assert paired_gate([2.0, 2.1], [2.2, 2.3])["passes"]
    assert not paired_gate([2.0, 2.1], [2.0, 2.1])["passes"]


def test_reader_projects_sequence_only(tmp_path, monkeypatch):
    (tmp_path / "train").mkdir()
    row = {
        "sample_id": "a",
        "split": "train",
        "sequence": "A" * 20,
        "token_ids": [2] * 20,
        "ca_coordinates": [[999.0] * 3] * 20,
    }
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "train" / "part.parquet")
    original = pq.ParquetFile.iter_batches
    calls = []

    def spy(self, *args, **kwargs):
        calls.append(kwargs["columns"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    loaded = list(sequence_rows(tmp_path, "train"))
    assert calls == [["sample_id", "split", "sequence", "token_ids"]]
    assert "ca_coordinates" not in loaded[0]
    t, v = collate(loaded)
    assert t.shape == (1, 20) and v.all()


def test_diagnostic_evaluator_identical_targets_and_baselines():
    from protein_sequence_generation.context import evaluate_panel

    rows = [
        {"sample_id": "a", "split": "validation", "sequence": "AC" * 10, "token_ids": [2, 3] * 10},
        {"sample_id": "b", "split": "validation", "sequence": "CA" * 10, "token_ids": [3, 2] * 10},
    ]
    baselines = train_unigrams([{**r, "split": "train"} for r in rows])
    model = tiny().train()
    result = evaluate_panel(model, rows, {"a": "b", "b": "a"}, baselines, 0.3)
    assert model.training
    assert len(result) == 2 and all(r["length_stratum"] == 0 for r in result)
    assert all(
        set(r["ce"])
        == {
            "normal",
            "visible_shuffle",
            "null_context",
            "permuted_context",
            "uniform",
            "global_unigram",
            "bucket_unigram",
        }
        for r in result
    )
    assert result == evaluate_panel(model, rows, {"a": "b", "b": "a"}, baselines, 0.3)


def test_classification_labels(monkeypatch):
    import protein_sequence_generation.context as context

    assert context.classify_s1({}) == "S1-E"
    panels = {f: [{"length_stratum": b} for b in range(5)] for f in ["0.15", "0.3", "0.5"]}

    def fake(pass_all=False, a=False, uniform=False):
        return {
            "eligible": True,
            "all_pass": pass_all,
            "gates": {
                k: {"passes": pass_all or (a and k.startswith("A_")) or (uniform and k == "uniform")}
                for k in ["A_global", "A_bucket", "B", "C", "D", "uniform"]
            },
        }

    monkeypatch.setattr(context, "gate_panel", lambda records: fake(pass_all=True))
    assert context.classify_s1(panels) == "S1-A"
    monkeypatch.setattr(context, "gate_panel", lambda records: fake(a=True))
    assert context.classify_s1(panels) == "S1-C"
    monkeypatch.setattr(context, "gate_panel", lambda records: fake(uniform=True))
    assert context.classify_s1(panels) == "S1-B"
    monkeypatch.setattr(context, "gate_panel", lambda records: fake(pass_all=len(records) == 1))
    assert context.classify_s1(panels) == "S1-D"
    monkeypatch.setattr(context, "gate_panel", lambda records: {"eligible": False, "all_pass": False})
    assert context.classify_s1(panels) == "S1-E"


def test_contract_seal_detects_tampering(tmp_path, monkeypatch):
    import hashlib
    import json

    from scripts import prepare_e011_sequence_context as preparation

    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "REPORT", tmp_path)
    assert not preparation.verify_contract()
    artifact = tmp_path / "contract.yaml"
    artifact.write_text("stage: preparation_only\n")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "contract_manifest.json").write_text(json.dumps({"sha256": {"contract.yaml": digest}}))
    assert preparation.verify_contract()
    artifact.write_text("stage: changed\n")
    with pytest.raises(ValueError, match="sealed S1 contract changed"):
        preparation.verify_contract()
