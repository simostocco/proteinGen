"""Evaluation-only E012 matched-panel generalization audit utilities."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import torch

from protein_sequence_generation.e012 import prefix_batch, stratum

PANEL_SALT = "E012-V3:13012:"


def select_train_panel(training, heldout, heldout_ids):
    """Length quotas only; rank TRAIN identities by SHA256 of fixed salt plus identity."""
    ids = [r["sample_id"] for r in training]
    if len(ids) != len(set(ids)) or any(r["split"] != "train" for r in training):
        raise ValueError("unique TRAIN-only population required")
    if set(ids) & set(heldout_ids):
        raise ValueError("TRAIN/heldout identity overlap")
    quotas = [sum(stratum(r["length"]) == s for r in heldout) for s in range(5)]
    selected = []
    for s, quota in enumerate(quotas):
        candidates = [r for r in training if stratum(r["length"]) == s]
        candidates.sort(key=lambda r: (hashlib.sha256((PANEL_SALT + r["sample_id"]).encode()).digest(), r["sample_id"]))
        if len(candidates) < quota:
            raise ValueError("insufficient TRAIN identities for stratum quota")
        selected.extend(r["sample_id"] for r in candidates[:quota])
    return {
        "sample_ids": selected,
        "stratum_counts": quotas,
        "size": len(selected),
        "selection_algorithm": (
            "Within each historical length stratum: ascending "
            "SHA256(UTF8('E012-V3:13012:'+sample_id)), sample_id tie-break; take heldout stratum "
            "quota; concatenate strata 0..4"
        ),
        "selection_uses_performance": False,
        "salt": PANEL_SALT,
    }


def relative_bins(length):
    """Exact historical ten equal relative-position deciles, zero-based target positions."""
    return np.minimum(np.arange(length) * 10 // length, 9)


def bulk_prefix_batch(cases, device, *, shuffled=False, window=None):
    """Identical historical construction on CPU, with five bulk transfers instead of per-residue CUDA writes."""
    return tuple(v.to(device) for v in prefix_batch(cases, "cpu", shuffled=shuffled, window=window))


@contextmanager
def evaluation_only_guard():
    """Forbid optimizer construction/steps, backward and torch checkpoint writes in the audit process."""

    def forbidden(*args, **kwargs):
        raise RuntimeError("E012 V3 is evaluation-only: training/checkpoint writes forbidden")

    objects = [(torch.optim.Optimizer, "__init__"), (torch.Tensor, "backward"), (torch, "save")]
    objects.extend(
        (value, "step")
        for value in vars(torch.optim).values()
        if isinstance(value, type) and issubclass(value, torch.optim.Optimizer) and "step" in value.__dict__
    )
    saved = [(obj, name, getattr(obj, name)) for obj, name in objects]
    try:
        for obj, name, _ in saved:
            setattr(obj, name, forbidden)
        with torch.no_grad():
            yield
    finally:
        for obj, name, previous in saved:
            setattr(obj, name, previous)


def metrics(records):
    """One common aggregation for all three panels; equal-protein primary and token-weighted secondary."""
    if not records:
        raise ValueError("empty panel")
    proteins, tokens = len(records), sum(r["length"] for r in records)
    equal = float(np.mean([r["normal"] for r in records]))
    token = sum(r["loss_sum"] for r in records) / tokens
    scalar = ["predictive_entropy", "max_probability", "correct_token_probability"]
    result = {
        "identities": proteins,
        "valid_tokens": tokens,
        "equal_protein_ce": equal,
        "token_weighted_ce": token,
        "equal_protein_perplexity": float(np.exp(equal)),
        "token_weighted_perplexity": float(np.exp(token)),
        "equal_protein_top1": float(np.mean([r["top1"] for r in records])),
        "equal_protein_top3": float(np.mean([r["top3"] for r in records])),
        "token_weighted_top1": sum(r["top1"] * r["length"] for r in records) / tokens,
        "token_weighted_top3": sum(r["top3"] * r["length"] for r in records) / tokens,
        "relative_position_ce": np.mean([r["relative_position_ce"] for r in records], axis=0).tolist(),
    }
    for key in scalar:
        result["equal_protein_" + key] = float(np.mean([r[key] for r in records]))
        result["token_weighted_" + key] = sum(r[key] * r["length"] for r in records) / tokens
    result["equal_protein_top1_confidence"] = result["equal_protein_max_probability"]
    result["token_weighted_top1_confidence"] = result["token_weighted_max_probability"]
    # Canonical probabilities retain their unnormalized mass under the unchanged 24-way output.
    for key in ["target_aa_frequencies", "mean_predicted_aa_probabilities", "aa_nll_contributions"]:
        values = np.asarray([r[key] for r in records])
        result["equal_protein_" + key] = values.mean(0).tolist()
        result["token_weighted_" + key] = (
            (values * np.asarray([r["length"] for r in records])[:, None]).sum(0).__truediv__(tokens).tolist()
        )
    counts = np.sum([r["aa_counts"] for r in records], axis=0)
    sums = np.sum([np.asarray(r["aa_nll_contributions"]) * r["length"] for r in records], axis=0)
    result["target_aa_counts"] = counts.tolist()
    result["conditional_token_nll_by_aa"] = (sums / np.maximum(counts, 1)).tolist()
    result["equal_protein_special_probability_mass"] = 1 - sum(result["equal_protein_mean_predicted_aa_probabilities"])
    return result


def panel_summary(records):
    return {
        "aggregate": metrics(records),
        "strata": {str(s): metrics([r for r in records if r["stratum"] == s]) for s in range(5)},
    }
