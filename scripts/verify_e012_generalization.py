"""Independent CPU verification of V3 scalar statistics, intervals and immutable sources."""

from __future__ import annotations

import json

import numpy as np

from scripts import audit_e012_generalization as audit


def reproduce(values):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(12112)
    samples = np.empty(10000)
    for start in range(0, 10000, 100):
        draw = generator.integers(0, values.size, (100, values.size))
        samples[start : start + 100] = np.add.reduce(values[draw], axis=1) / values.size
    return float(values.mean()), np.percentile(samples, [2.5, 97.5])


def reproduce_gap(train, held):
    train, held = np.asarray(train), np.asarray(held)
    generator = np.random.default_rng(12112)
    samples = np.empty(10000)
    for start in range(0, 10000, 100):
        a = generator.integers(0, train.size, (100, train.size))
        b = generator.integers(0, held.size, (100, held.size))
        samples[start : start + 100] = (
            np.add.reduce(held[b], axis=1) / held.size - np.add.reduce(train[a], axis=1) / train.size
        )
    return float(held.mean() - train.mean()), np.percentile(samples, [2.5, 97.5])


def check_interval(values, result):
    mean, ci = reproduce(values)
    assert abs(mean - result["delta"]) < 1e-12
    assert np.allclose(ci, result["ci95"], atol=1e-12, rtol=0)


def verify():
    result = audit.read(audit.REPORT / "results.json")
    verified = audit.verify_historical()
    c = audit.read(audit.REPORT / "contract.json")
    for name, digest in c["source_hashes"].items():
        assert audit.old.sha(audit.ROOT / name) == digest
    for name, digest in audit.read(audit.REPORT / "execution_source_hashes.json").items():
        assert audit.old.sha(audit.ROOT / name) == digest
    panel = audit.read(audit.REPORT / "train_panel.json")
    assert len(set(panel["sample_ids"])) == 2048
    raw = {}
    interval_count = 0
    for step in audit.STEPS:
        raw[step] = {}
        for pop in ["train", "primary", "independent"]:
            records = audit.read(audit.OUT / f"{pop}_likelihood_{step:05d}.json")
            raw[step][pop] = {r["sample_id"]: r for r in records}
            for record in records:
                for key in ["normal", "loss_sum", "predictive_entropy", "max_probability", "correct_token_probability"]:
                    assert np.isfinite(record[key])
                assert 0 <= record["max_probability"] <= 1 and 0 <= record["correct_token_probability"] <= 1
                assert sum(record["aa_counts"]) == record["length"]
            context_values = audit.read(audit.OUT / f"{pop}_context_{step:05d}.json")
            assert all(np.isfinite(v) for sample in context_values.values() for v in sample.values())
            if pop == "train":
                assert set(raw[step][pop]) == set(panel["sample_ids"])
            for s in [None, 0, 1, 2, 3, 4]:
                rows = [r for r in records if s is None or r["stratum"] == s]
                section = result["matched_metrics"][str(step)][pop]
                aggregate = section["aggregate"] if s is None else section["strata"][str(s)]
                ce = sum(r["normal"] for r in rows) / len(rows)
                token = sum(r["loss_sum"] for r in rows) / sum(r["length"] for r in rows)
                assert abs(ce - aggregate["equal_protein_ce"]) < 1e-12
                assert abs(token - aggregate["token_weighted_ce"]) < 1e-12
                assert sum(aggregate["target_aa_counts"]) == aggregate["valid_tokens"]
                assert abs(sum(aggregate["equal_protein_aa_nll_contributions"]) - ce) < 1e-6
                assert all(np.isfinite(r["normal"]) for r in rows)
                context = audit.read(audit.OUT / f"{pop}_context_{step:05d}.json")
                summary = result["context_diagnostics"][str(step)][pop]
                section = summary["aggregate"] if s is None else summary["strata"][str(s)]
                for key, condition in [
                    ("shuffle_delta", "shuffle"),
                    ("full_minus_last8", "last8"),
                    ("full_minus_last1", "last1"),
                ]:
                    ordered = sorted(rows, key=lambda r: r["sample_id"]) if s is None else rows
                    check_interval(
                        [
                            context[r["sample_id"]]["prefix_normal"] - context[r["sample_id"]][condition]
                            for r in ordered
                        ],
                        section[key],
                    )
                    interval_count += 1
            for base in ["global_unigram", "bucket_unigram", "bigram", "trigram"]:
                check_interval(
                    [r["normal"] - r[base] for r in records], result["baseline_comparisons"][str(step)][pop][base]
                )
                interval_count += 1
        for pop in ["primary", "independent"]:
            for s in [None, 0, 1, 2, 3, 4]:
                train = [r["normal"] for r in raw[step]["train"].values() if s is None or r["stratum"] == s]
                held = [r["normal"] for r in raw[step][pop].values() if s is None or r["stratum"] == s]
                section = result["generalization_gaps"][str(step)][pop]
                stored = section["equal_protein"] if s is None else section["strata"][str(s)]
                mean, ci = reproduce_gap(train, held)
                assert abs(mean - stored["delta"]) < 1e-12 and np.allclose(ci, stored["ci95"], atol=1e-12, rtol=0)
                interval_count += 1
    for pop in ["train", "primary", "independent"]:
        a, b = raw[5000][pop], raw[10000][pop]
        for name, key in [
            ("ce_10000_minus_5000", "normal"),
            ("entropy_10000_minus_5000", "predictive_entropy"),
            ("confidence_10000_minus_5000", "max_probability"),
        ]:
            check_interval([b[s][key] - a[s][key] for s in sorted(a)], result["paired_5k_to_10k_changes"][pop][name])
            interval_count += 1
        for stratum in range(5):
            check_interval(
                [b[s]["normal"] - a[s]["normal"] for s in sorted(a) if a[s]["stratum"] == stratum],
                result["paired_5k_to_10k_changes"][pop]["strata_ce_changes"][str(stratum)],
            )
            interval_count += 1
    output = {
        "passed": True,
        "intervals_independently_reproduced": interval_count,
        "resamples_each": 10000,
        "seed": 12112,
        "protected_historical_files_verified": len(verified),
        "checkpoint_hashes_verified": {str(s): h for s, (_, h) in audit.CHECKPOINTS.items()},
        "panel_hash_verified": audit.old.sha(audit.REPORT / "train_panel.json"),
        "scalar_and_aa_contribution_checks": "passed",
        "training_launched": False,
    }
    audit.old.save(audit.REPORT / "independent_verification.json", output, exclusive=True)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    verify()
