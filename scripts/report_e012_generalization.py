"""Reproducible matched-panel E012 V3 statistics, without model execution or training."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from scripts import audit_e012_generalization as audit
from scripts import run_e012_causal_rope as old

PANELS = ["train", "primary", "independent"]
STRATA = ["20–64", "65–128", "129–256", "257–384", "385–500"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def interval(values):
    """Independent implementation of historical 10000-resample identity bootstrap."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(12112)
    means = []
    for _ in range(100):
        indices = rng.integers(0, len(values), size=(100, len(values)))
        means.extend(values[indices].mean(1).tolist())
    return {
        "delta": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "resampling": "paired identity",
    }


def gap_interval(train, heldout):
    """Independent identity resampling for disjoint panels; no fabricated pairing."""
    train, heldout = np.asarray(train), np.asarray(heldout)
    rng = np.random.default_rng(12112)
    values = []
    for _ in range(100):
        a = rng.integers(0, len(train), size=(100, len(train)))
        b = rng.integers(0, len(heldout), size=(100, len(heldout)))
        values.extend((heldout[b].mean(1) - train[a].mean(1)).tolist())
    return {
        "delta": float(heldout.mean() - train.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "resampling": "independent identities in disjoint panels",
    }


def contextual(records):
    values = list(records.values())
    return {
        "means": {key: float(np.mean([r[key] for r in values])) for key in values[0]},
        "shuffle_delta": interval([r["prefix_normal"] - r["shuffle"] for r in values]),
        "full_minus_last8": interval([r["prefix_normal"] - r["last8"] for r in values]),
        "full_minus_last1": interval([r["prefix_normal"] - r["last1"] for r in values]),
    }


def build():
    completed = audit.read(audit.OUT / "completed.json")
    assert (
        not completed["training_launched"] and completed["optimizer_steps"] == 0 and completed["checkpoint_writes"] == 0
    )
    data, summaries, contexts = {}, {}, {}
    for step in audit.STEPS:
        data[step], summaries[step], contexts[step] = {}, {}, {}
        for panel in PANELS:
            data[step][panel] = audit.read(audit.OUT / f"{panel}_likelihood_{step:05d}.json")
            summaries[step][panel] = audit.read(audit.OUT / f"{panel}_summary_{step:05d}.json")
            raw = audit.read(audit.OUT / f"{panel}_context_{step:05d}.json")
            contexts[step][panel] = {
                "aggregate": contextual(raw),
                "strata": {
                    str(s): contextual(
                        {r["sample_id"]: raw[r["sample_id"]] for r in data[step][panel] if r["stratum"] == s}
                    )
                    for s in range(5)
                },
            }
    gaps = {}
    trajectory = []
    for step in audit.STEPS:
        gaps[step] = {}
        item = {"checkpoint": step}
        train = data[step]["train"]
        for panel in PANELS:
            agg = summaries[step][panel]["aggregate"]
            con = contexts[step][panel]["aggregate"]
            item[panel + "_ce"] = agg["equal_protein_ce"]
            item[panel + "_shuffle_delta"] = con["shuffle_delta"]["delta"]
            item[panel + "_full_minus_last8"] = con["full_minus_last8"]["delta"]
            if panel == "train":
                continue
            held = data[step][panel]
            gap = gap_interval([r["normal"] for r in train], [r["normal"] for r in held])
            item[panel + "_train_gap"] = gap["delta"]
            token_gap = agg["token_weighted_ce"] - summaries[step]["train"]["aggregate"]["token_weighted_ce"]
            gaps[step][panel] = {
                "equal_protein": gap,
                "token_weighted_gap": token_gap,
                "strata": {
                    str(s): gap_interval(
                        [r["normal"] for r in train if r["stratum"] == s],
                        [r["normal"] for r in held if r["stratum"] == s],
                    )
                    for s in range(5)
                },
            }
        trajectory.append(item)
    changes = {}
    for panel in PANELS:
        a = {r["sample_id"]: r for r in data[5000][panel]}
        b = {r["sample_id"]: r for r in data[10000][panel]}
        assert set(a) == set(b)
        changes[panel] = {
            "ce_10000_minus_5000": interval([b[s]["normal"] - a[s]["normal"] for s in sorted(a)]),
            "entropy_10000_minus_5000": interval(
                [b[s]["predictive_entropy"] - a[s]["predictive_entropy"] for s in sorted(a)]
            ),
            "confidence_10000_minus_5000": interval(
                [b[s]["max_probability"] - a[s]["max_probability"] for s in sorted(a)]
            ),
            "strata_ce_changes": {
                str(s): interval([b[k]["normal"] - a[k]["normal"] for k in sorted(a) if a[k]["stratum"] == s])
                for s in range(5)
            },
        }
    panel_ids = set(audit.read(audit.REPORT / "train_panel.json")["sample_ids"])
    exposure_counts, exposure_history = Counter(), {}
    natural_strata = Counter()
    for path in [old.OUT / "telemetry.jsonl", audit.CONT_OUT / "telemetry.jsonl"]:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            exposure_counts.update(sid for sid in row["sample_ids"] if sid in panel_ids)
            natural_strata.update({s: row["length_stratum_exposures"][s] for s in range(5)})
            if row["update"] in audit.STEPS:
                values = [exposure_counts[s] for s in panel_ids]
                exposure_history[str(row["update"])] = {
                    "panel_exposures": sum(values),
                    "seen_identities": sum(v > 0 for v in values),
                    "unseen_identities": sum(v == 0 for v in values),
                    "histogram": dict(Counter(values)),
                    "mean_exposures": float(np.mean(values)),
                }
    training, _ = audit.populations()
    sequence_counts = Counter(r["sequence"] for r in training)
    diversity = {
        "training_identities": len(training),
        "training_unique_exact_sequences": len(sequence_counts),
        "train_panel_unique_exact_sequences": len({r["sequence"] for r in training if r["sample_id"] in panel_ids}),
        "natural_training_stratum_exposures_through_10k": {STRATA[s]: natural_strata[s] for s in range(5)},
    }
    comparisons = [audit.read(audit.OUT / f"same_batch_{step:05d}.json") for step in audit.STEPS]
    baseline = {
        step: {
            p: {
                k: interval([r["normal"] - r[k] for r in data[step][p]])
                for k in ["global_unigram", "bucket_unigram", "bigram", "trigram"]
            }
            for p in PANELS
        }
        for step in audit.STEPS
    }
    result = {
        "trajectory": trajectory,
        "matched_metrics": summaries,
        "generalization_gaps": gaps,
        "context_diagnostics": contexts,
        "paired_5k_to_10k_changes": changes,
        "baseline_comparisons": baseline,
        "train_panel_exposure_history": exposure_history,
        "training_diversity_telemetry": diversity,
        "same_batch_reconstruction": comparisons,
        "aa_order": AA,
        "relative_position_deciles": [f"{i * 10}–{(i + 1) * 10}%" for i in range(10)],
        "training_log_semantics": {
            "mode": "train(), dropout active",
            "scope": (
                "instantaneous 64-protein optimizer batch before the update; console every50, raw telemetry everyupdate"
            ),
            "reduction": (
                "per-protein mean over valid canonical residue targets, then 64-protein mean; each "
                "physical microbatch weighted n/64"
            ),
            "physical_batching": "pilot8x8; continuation frozen length-routed unequal microbatches",
            "padding": "excluded using attention mask",
            "masking": "teacher-forced causal target at every valid residue; no MLM masking or extra filtering",
            "length_sampling": "natural TRAIN shuffled-identity distribution, not equal stratum quotas",
            "rolling_average": False,
            "directly_comparable_to_fixed_panel_eval": False,
            "mathematical_reduction_matches": True,
            "precision_nuance": (
                "training F.cross_entropy under AMP versus canonical float32 log-softmax can differ "
                "numerically; preflight difference0.0000775nats; same-batch measures actual difference"
            ),
        },
        "execution": completed,
        "training_launched": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "historical_CSEQ_classification_unchanged": "CSEQ-C / CONT-B",
    }
    old.save(audit.REPORT / "results.json", result, exclusive=True)
    print(
        json.dumps(
            {"trajectory": trajectory, "paired_changes": changes, "same_batch": comparisons, "diversity": diversity},
            indent=2,
        )
    )


if __name__ == "__main__":
    build()
