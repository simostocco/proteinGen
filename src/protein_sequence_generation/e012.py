"""Preregistered causal sequence diagnostics; no structural dependencies."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from protein_sequence_generation.collate import collate_sequences
from protein_sequence_generation.vocabulary import ProteinVocabulary

VOCAB = ProteinVocabulary()
STRATA = [(20, 64), (65, 128), (129, 256), (257, 384), (385, 500)]
WINDOWS = [1, 4, 8, 16, 32, 64]


def stratum(length):
    for i, (lo, hi) in enumerate(STRATA):
        if lo <= length <= hi:
            return i
    raise ValueError("length outside population")


def positions(length):
    """Zero-based target indices: 16 evenly spaced interior positions, inclusive."""
    lo, hi = max(2, int(np.floor(0.1 * length))), min(length - 2, int(np.floor(0.9 * length)))
    if hi < lo:
        return []
    return np.unique(np.linspace(lo, hi, min(16, hi - lo + 1)).astype(int)).tolist()


def shuffle_prefix(tokens, target_index, sample_id):
    prefix = np.asarray(tokens[:target_index], dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(f"E012:12012:{sample_id}:{target_index}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed).permutation(prefix).tolist()


def example(row):
    tokens = VOCAB.encode(row["sequence"])
    return {
        "sample_id": row["sample_id"],
        "sequence": row["sequence"],
        "metadata": {},
        "input_ids": torch.tensor([VOCAB.bos_id] + tokens[:-1]),
        "target_ids": torch.tensor(tokens),
        "length": torch.tensor(len(tokens)),
    }


def batch(rows, device):
    result = collate_sequences([example(r) for r in rows])
    return {k: result[k].to(device) for k in ["input_ids", "target_ids", "lengths", "attention_mask"]}


def prefix_batch(cases, device, *, shuffled=False, window=None):
    """Retain absolute input positions; block removed keys in every layer."""
    maximum = max(i + 1 for _, i in cases)
    inputs = torch.zeros(len(cases), maximum, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    indices, targets, lengths = [], [], []
    for j, (row, index) in enumerate(cases):
        tokens = VOCAB.encode(row["sequence"])
        prefix = shuffle_prefix(tokens, index, row["sample_id"]) if shuffled else tokens[:index]
        inputs[j, : index + 1] = torch.tensor([VOCAB.bos_id] + prefix, device=device)
        start = 0 if window is None else max(0, index - window + 1)
        mask[j, start : index + 1] = True
        inputs[j, :start] = VOCAB.pad_id
        indices.append(index)
        targets.append(tokens[index])
        lengths.append(len(tokens))
    return (
        inputs,
        torch.tensor(lengths, device=device),
        mask,
        torch.tensor(indices, device=device),
        torch.tensor(targets, device=device),
    )


def build_baselines(rows, alpha=1.0):
    """TRAIN-only canonical 20-way counts; BOS is history index 20, twice for trigram start."""
    if any(r["split"] != "train" for r in rows):
        raise ValueError("baselines require TRAIN only")
    global_counts = np.zeros(20, dtype=np.int64)
    bucket_counts = np.zeros((5, 20), dtype=np.int64)
    bigram = np.zeros((21, 20), dtype=np.int64)
    trigram = np.zeros((21, 21, 20), dtype=np.int64)
    for r in rows:
        tokens = np.asarray(VOCAB.encode(r["sequence"]), dtype=np.int64) - 4
        previous = np.r_[20, tokens[:-1]]
        previous2 = np.r_[20, 20, tokens][:-2]
        np.add.at(global_counts, tokens, 1)
        np.add.at(bucket_counts[stratum(len(tokens))], tokens, 1)
        np.add.at(bigram, (previous, tokens), 1)
        np.add.at(trigram, (previous2, previous, tokens), 1)
    counts = {"global_unigram": global_counts, "bucket_unigram": bucket_counts, "bigram": bigram, "trigram": trigram}
    return {"alpha": alpha, "training_proteins": len(rows), "counts": {k: v.tolist() for k, v in counts.items()}}


def baseline_ce(sequence, baseline):
    tokens = np.asarray(VOCAB.encode(sequence)) - 4
    previous, previous2 = np.r_[20, tokens[:-1]], np.r_[20, 20, tokens][:-2]
    result = {}
    for name, values in baseline["counts"].items():
        counts = np.asarray(values, dtype=np.float64) + baseline["alpha"]
        logp = np.log(counts / counts.sum(-1, keepdims=True))
        if name == "global_unigram":
            chosen = logp[tokens]
        elif name == "bucket_unigram":
            chosen = logp[stratum(len(tokens)), tokens]
        elif name == "bigram":
            chosen = logp[previous, tokens]
        else:
            chosen = logp[previous2, previous, tokens]
        result[name] = float(-chosen.mean())
    return result


def paired_interval(differences, seed=12112):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(100):
        draws.extend(values[rng.integers(len(values), size=(100, len(values)))].mean(1))
    return {
        "delta": float(values.mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "identities": len(values),
        "resamples": 10000,
    }


def summarize(records):
    """Paired identity intervals and fixed gates. Independent confirmation has explicit rules."""

    def section(rows):
        names = ["global_unigram", "bucket_unigram", "bigram", "trigram", "shuffle", "last8", "last1"]
        deltas = {}
        for name in names:
            left = "normal" if name in names[:4] else "prefix_normal"
            deltas[name] = paired_interval([r[left] - r[name] for r in rows])
        means = {
            k: float(np.mean([r[k] for r in rows]))
            for k in [
                "normal",
                *names,
                "prefix_normal",
                "last4",
                "last16",
                "last32",
                "last64",
                "neutral_length",
                "kl",
                "js",
                "top1_change",
                "top3_change",
            ]
        }
        return {"means": means, "deltas": deltas}

    aggregate = section(records)
    strata = {str(i): section([r for r in records if r["stratum"] == i]) for i in range(5)}
    d = aggregate["deltas"]

    def favorable(k):
        return d[k]["delta"] < 0 and d[k]["ci95"][1] < 0

    major_failure = any(
        s["deltas"]["bucket_unigram"]["delta"] > 0 or s["deltas"]["shuffle"]["delta"] >= 0.02 for s in strata.values()
    )
    four_strata = sum(s["deltas"]["shuffle"]["delta"] < 0 for s in strata.values()) >= 4
    gates = {
        "A": all(favorable(k) and d[k]["delta"] <= -0.05 for k in ["global_unigram", "bucket_unigram"]),
        "B": favorable("bigram") and favorable("trigram") and d["trigram"]["delta"] <= -0.01,
        "C": favorable("shuffle") and d["shuffle"]["delta"] <= -0.02,
        "D": favorable("last8") and d["last8"]["delta"] <= -0.01 and favorable("last1"),
        "E": not major_failure and four_strata,
    }
    confirmation = {
        "A": all(d[k]["delta"] < 0 for k in ["global_unigram", "bucket_unigram"]),
        "B": all(d[k]["delta"] < 0 for k in ["bigram", "trigram"]),
        "C": favorable("shuffle"),
        "D": all(d[k]["delta"] < 0 for k in ["last8", "last1"]),
        "E": gates["E"],
    }
    return {"aggregate": aggregate, "strata": strata, "gates": gates, "confirmation": confirmation}


def classification(primary, independent):
    gates = primary["gates"]
    if all(gates.values()) and all(independent["confirmation"].values()):
        return "CSEQ-A"
    if all(gates[k] for k in "ABCD") and (not gates["E"] or not independent["confirmation"]["E"]):
        return "CSEQ-D"
    if gates["C"] and gates["D"] and independent["confirmation"]["C"] and independent["confirmation"]["D"]:
        return "CSEQ-C"
    return "CSEQ-B"
