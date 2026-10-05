from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts import e010_phase4a_multicorruption_v2 as mc
from scripts import smoke_e010_multicorruption_v2_exact_batch as exact_smoke
from scripts.prepare_e010_phase4a_multicorruption_v2 import _split_membership
from scripts.run_e010_phase4a_multicorruption_v2 import (
    _append_event,
    _atomic_json,
    _evaluate_panel,
    _fresh_state,
    _load_execution_contract,
    _load_training_data,
    _recover_missing_boundary,
    _schedule_from_cursor,
    _validate_checkpoint_journal,
    journal_events,
)


def _synthetic_cohort():
    rows = []
    for stratum, count in mc.TRAIN_COUNTS.items():
        lo, hi = (int(x) for x in stratum.split("-"))
        length = (lo + hi) // 2
        for i in range(count):
            sid = f"{stratum}:{i:05d}"
            rows.append(
                {
                    "sample_id": sid,
                    "stratum": stratum,
                    "length": length,
                    "source_path": f"synthetic/{sid}.npz",
                    "source_sha256": "0" * 64,
                    "selection_rank_sha256": mc.hash_rank(mc.SCHEMA, "train", sid),
                }
            )
    return rows


def test_v2_revised_synthetic_counts_multiplicities_and_schedule():
    manifest, summary = mc.build_seed_manifest(_synthetic_cohort())
    assert len(manifest) == 16384
    assert {s: sum(r["stratum"] == s for r in manifest) for s in mc.STRATA} == mc.TRAIN_COUNTS
    assert summary["example_count_by_stratum"] == {s: 19656 for s in mc.STRATA}
    assert len({(r["sample_id"], seed) for r in manifest for seed in r["corruption_seeds"]}) == 98280
    assert {
        s: sum(r["stratum"] == s and r["multiplicity"] == (13 if s == "385-500" else 6) for r in manifest)
        for s in mc.STRATA
    } == {"20-64": 1171, "65-128": 1171, "129-256": 1171, "257-384": 1006, "385-500": 900}
    schedule = mc.build_schedule(manifest)
    assert len(schedule) == 1092
    for item in schedule:
        assert item["effective_batch_sample_count"] == 90
        assert {s: len(item["stratum_microbatches"][s]) for s in mc.STRATA} == {s: 18 for s in mc.STRATA}
    for update, examples, per_stratum in mc.BOUNDARIES:
        assert update * 90 == examples and update * 18 == per_stratum
        assert 5 * per_stratum == examples


def test_exact_batch_smoke_selects_published_first_update_without_duplicates():
    manifest, _ = mc.build_seed_manifest(_synthetic_cohort())
    row = mc.build_schedule(manifest)[0]
    examples = exact_smoke.schedule_examples(row)
    assert len(examples) == 90
    assert {s: sum(stratum == s for stratum, _ in examples) for s in mc.STRATA} == {s: 18 for s in mc.STRATA}
    assert len({(x["sample_id"], x["corruption_seed"]) for _, x in examples}) == 90
    assert sum([mc.MICROBATCH / mc.EFFECTIVE_BATCH] * 5) == 1.0


def test_real_v2_plan_has_revised_cohort_and_exact_source_samples():
    plan = json.loads((mc.OUT / "plan.json").read_text())
    prep = json.loads((mc.OUT / "preparation_manifest.json").read_text())
    seed = json.loads((mc.OUT / "training_seed_manifest.json").read_text())
    excluded = json.loads((mc.OUT / "excluded_archives.json").read_text())
    identities = seed["identities"]
    assert len(identities) == mc.TOTAL_IDENTITIES == 16384
    assert {s: sum(r["stratum"] == s for r in identities) for s in mc.STRATA} == mc.TRAIN_COUNTS
    assert plan["training_identity_count"] == 16384
    assert plan["total_unique_training_examples"] == 98280
    assert prep["optimizer_updates"] == 1092
    assert prep["seed_manifest_sha256"] == mc.file_sha(mc.OUT / "training_seed_manifest.json")
    assert excluded["excluded_count_by_reason"]["coordinate_length_mismatch"] == 18
    assert excluded["excluded_count_by_reason"]["payload_sample_id_mismatch"] == 14
    assert mc.file_sha(mc.V1_DISCREPANCY) == excluded["source_discrepancy_sha256"]

    membership, _ = _split_membership()
    selected = {r["sample_id"] for r in identities}
    assert all(membership[sid] == "train" for sid in selected)
    assert not selected.intersection({sid for sid, split in membership.items() if split in {"validation", "test"}})
    excluded_fingerprints = set()
    for row in excluded["archives"]:
        st = (mc.ROOT / row["source_path"]).stat()
        excluded_fingerprints.add((st.st_dev, st.st_ino))
    assert all(
        (st.st_dev, st.st_ino) not in excluded_fingerprints
        for st in ((mc.ROOT / row["source_path"]).stat() for row in identities)
    )

    # Touch a real source in every stratum; full archive validation is performed by --validate-contract.
    for stratum in mc.STRATA:
        row = next(r for r in identities if r["stratum"] == stratum)
        coords, mask = mc.source_arrays(row)
        assert coords.shape == (row["length"], 3)
        assert mask.shape == (row["length"],) and bool(mask.all())


def test_training_loader_never_opens_excluded_archive(monkeypatch):
    seed = json.loads((mc.OUT / "training_seed_manifest.json").read_text())
    exclusions = json.loads((mc.OUT / "excluded_archives.json").read_text())["archives"]
    opened = []
    original_load = np.load

    def observed_load(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", observed_load)
    _load_training_data(seed["identities"][:1])
    excluded_paths = [mc.ROOT / row["source_path"] for row in exclusions]
    assert opened
    assert all(not any(os.path.samefile(path, excluded) for excluded in excluded_paths) for path in opened)


def test_bounded_v2_corruption_regeneration_hash_is_repeatable():
    target = np.arange(48, dtype=np.float32).reshape(16, 3) / 13
    first = mc.regenerate_corruption(target, 0.35, 90123)
    second = mc.regenerate_corruption(target, 0.35, 90123)
    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(second.tobytes()).digest()


def test_fresh_state_saves_pinned_metadata_from_validated_real_contract(tmp_path, monkeypatch):
    import torch

    contract = _load_execution_contract()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    class CPUScaler:
        def state_dict(self):
            return {}

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    state = _fresh_state(model, optimizer, CPUScaler(), contract)
    checkpoint = tmp_path / "fresh_state.pt"
    torch.save(state, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config_sha256"] == contract["config_sha256"] == mc.file_sha(mc.CONFIG)
    assert saved["seed_manifest_sha256"] == contract["seed_manifest_sha256"]
    assert saved["schedule_sha256"] == contract["schedule_sha256"]
    assert saved["global_update"] == saved["schedule_cursor"] == 0


def test_atomic_json_converts_numpy_scalars_arrays_and_paths(tmp_path):
    path = tmp_path / "nested.json"
    _atomic_json(
        path,
        {
            "nested": {
                "finite": np.bool_(True),
                "count": np.int64(364),
                "metric": np.float32(0.25),
                "values": np.array([1, 2]),
                "path": tmp_path / "artifact",
            }
        },
    )
    record = json.loads(path.read_text())
    assert record["nested"] == {
        "finite": True,
        "count": 364,
        "metric": 0.25,
        "values": [1, 2],
        "path": str(tmp_path / "artifact"),
    }
    assert type(record["nested"]["finite"]) is bool
    assert type(record["nested"]["count"]) is int
    assert type(record["nested"]["metric"]) is float
    with np.testing.assert_raises_regex(TypeError, "not JSON serializable"):
        _atomic_json(tmp_path / "unsupported.json", {"object": object()})


def test_evaluation_microbatch_preserves_order_identity_and_metrics(monkeypatch):
    import torch

    from scripts import run_e010_phase4a_multicorruption_v2 as runner

    class DeterministicModel:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, coarse, mask):
            assert torch.is_inference_mode_enabled()
            return {"prediction": coarse, "delta": torch.zeros_like(coarse)}

    rows = [{"sample_id": f"dev-{i}", "length": n, "stratum": "20-64"} for i, n in enumerate((4, 6, 5, 3, 7))]
    values = {}
    for row in rows:
        n = row["length"]
        target = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / (n + 1)
        coarse = target + np.float32(0.1)
        values[row["sample_id"]] = (target, coarse, np.ones(n, dtype=np.bool_))
    monkeypatch.setattr(runner, "_load_primary_dev", lambda row: values[row["sample_id"]])
    one = _evaluate_panel(DeterministicModel(), rows, "cpu", microbatch_size=1)
    four = _evaluate_panel(DeterministicModel(), rows, "cpu", microbatch_size=4)
    assert [r["sample_id"] for r in one] == [r["sample_id"] for r in four] == [r["sample_id"] for r in rows]
    assert [r["length"] for r in one] == [r["length"] for r in four]
    assert [r["finite"] for r in one] == [r["finite"] for r in four]
    np.testing.assert_allclose(
        [r["aligned_rmse_angstrom"] for r in one], [r["aligned_rmse_angstrom"] for r in four], rtol=1e-6, atol=1e-7
    )


def test_resume_recovers_committed_update_364_boundary_and_next_update(tmp_path, monkeypatch):
    import torch

    from scripts import run_e010_phase4a_multicorruption_v2 as runner

    monkeypatch.setattr(mc, "STAGING", tmp_path)
    state = {"global_update": 364, "schedule_cursor": 364, "payload": "exact committed state"}
    torch.save(state, tmp_path / "latest.pt")
    checkpoint_hash = mc.file_sha(tmp_path / "latest.pt")
    _append_event(
        tmp_path / "journal.jsonl",
        {
            "global_update": 1,
            "training_started": True,
            "prospective_accessed": False,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    # Build a valid 364-row prefix while pinning the checkpoint hash only at its tip.
    rows = (tmp_path / "journal.jsonl").read_text().splitlines()
    first = json.loads(rows[0])
    events = []
    for update in range(1, 365):
        row = dict(first, global_update=update, checkpoint_sha256=checkpoint_hash)
        payload = {k: v for k, v in row.items() if k != "record_sha256"}
        row["record_sha256"] = mc.sha_bytes(mc.canonical_json(payload))
        events.append(json.dumps(row, separators=(",", ":")))
    (tmp_path / "journal.jsonl").write_text("\n".join(events) + "\n")
    assert len(journal_events(tmp_path / "journal.jsonl")) == 364
    restored = _validate_checkpoint_journal(tmp_path, journal_events(tmp_path / "journal.jsonl"))
    assert restored == state
    published = []

    def publish(_model, _dev, _seed, _baseline, update, checkpoint, _config):
        published.append((update, mc.file_sha(checkpoint)))
        _atomic_json(
            tmp_path / f"development_update_{update:04d}.json",
            {"optimizer_updates": update, "checkpoint_sha256": mc.file_sha(checkpoint)},
        )

    monkeypatch.setattr(runner, "_publish_boundary", publish)
    _recover_missing_boundary(restored["global_update"], object(), [], [], [], {})
    assert published == [(364, checkpoint_hash)]
    assert json.loads((tmp_path / "development_update_0364.json").read_text())["optimizer_updates"] == 364
    optimizer_updates = _schedule_from_cursor([{"global_update": n} for n in range(1, 1093)], restored["global_update"])
    assert optimizer_updates[0]["global_update"] == 365
