from __future__ import annotations

import ast
import hashlib

import numpy as np

from scripts import e010_phase4a_multicorruption as mc
from scripts import run_e010_phase4a_multicorruption as lifecycle


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


def test_capped_identity_allocation_and_exact_exposure_schedule():
    manifest, summary = mc.build_seed_manifest(_synthetic_cohort())
    assert len(manifest) == mc.TOTAL_IDENTITIES == 16384
    assert summary["example_count_by_stratum"] == {s: 19656 for s in mc.STRATA}
    assert summary["unique_identity_seed_pair_count"] == 98280
    assert {s: sum(r["multiplicity"] == 6 for r in manifest if r["stratum"] == s) for s in mc.STRATA} == {
        "20-64": 1171,
        "65-128": 1171,
        "129-256": 1171,
        "257-384": 1176,
        "385-500": 0,
    }
    assert sum(r["multiplicity"] == 13 for r in manifest if r["stratum"] == "385-500") == 492
    pairs = [(r["sample_id"], seed) for r in manifest for seed in r["corruption_seeds"]]
    assert len(pairs) == len(set(pairs)) == mc.TOTAL_EXAMPLES
    schedule = mc.build_schedule(manifest)
    assert len(schedule) == 1092
    for update in schedule:
        assert update["effective_batch_sample_count"] == 90
        assert {s: len(update["stratum_microbatches"][s]) for s in mc.STRATA} == {s: 18 for s in mc.STRATA}
    for updates, examples, per_stratum in mc.BOUNDARIES:
        assert updates * 90 == examples
        assert updates * 18 == per_stratum
        assert 5 * per_stratum == examples


def test_bounded_deterministic_corruption_tensor_hashes():
    # A small CPU tensor proves on-demand regeneration is byte reproducible.
    target = np.arange(36, dtype=np.float32).reshape(12, 3) / 10
    a = mc.regenerate_corruption(target, 0.4, 123456)
    b = mc.regenerate_corruption(target, 0.4, 123456)
    c = mc.regenerate_corruption(target, 0.4, 123457)
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()
    assert hashlib.sha256(a.tobytes()).hexdigest() != hashlib.sha256(c.tobytes()).hexdigest()


def test_journal_prefix_hash_validation_is_cpu_only(tmp_path):
    event = {
        "global_update": 1,
        "training_started": True,
        "prospective_accessed": False,
        "checkpoint_sha256": "a" * 64,
        "authorization": {"phase4b": False, "prospective": False, "downstream": False},
    }
    event["record_sha256"] = lifecycle._record_hash(event)
    path = tmp_path / "journal.jsonl"
    path.write_bytes(mc.canonical_json(event) + b"\n")
    assert lifecycle.journal_events(path) == [event]
    event["checkpoint_sha256"] = "b" * 64
    path.write_bytes(mc.canonical_json(event) + b"\n")
    try:
        lifecycle.journal_events(path)
    except ValueError as exc:
        assert "journal hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered journal record was accepted")


def test_preparation_and_read_only_validation_do_not_reference_cuda():
    for name in ("prepare_e010_phase4a_multicorruption.py",):
        tree = ast.parse((mc.ROOT / "scripts" / name).read_text())
        assert not any(
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "torch"
            and n.attr == "cuda"
            for n in ast.walk(tree)
        )
