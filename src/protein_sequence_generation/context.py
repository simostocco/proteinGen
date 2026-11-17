"""E011 sequence-only masked context model and controlled counterfactuals.

Vocabulary: PAD=0, MASK=1, canonical ACDEFGHIKLMNPQRSTVWY=2..21.
No geometry or length embedding is constructed or accepted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

STRATA = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
COLUMNS = ("sample_id", "split", "sequence", "token_ids")


class SequenceContextTransformer(nn.Module):
    """Exact E006 Stage-A trunk with only the canonical output rows."""

    def __init__(self, d_model=256, layers=8, heads=8, ffn=1024, dropout=0.1, max_length=500):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(22, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.sequence_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model, heads, ffn, dropout, activation="gelu", batch_first=True, norm_first=True
                )
                for _ in range(layers)
            ]
        )
        self.sequence_norm = nn.LayerNorm(d_model)
        self.sequence_output = nn.Linear(d_model, 20)

    def copy_historical_weights(self, historical):
        state = historical.state_dict()
        own = self.state_dict()
        for name in own:
            own[name] = state[name][2:22] if name.startswith("sequence_output.") else state[name]
        self.load_state_dict(own, strict=True)

    def forward(self, tokens, valid):
        if tokens.shape != valid.shape or tokens.ndim != 2:
            raise ValueError("tokens and valid must have identical [B,L] shapes")
        if tokens.shape[1] > self.max_length or not valid.bool().any(dim=1).all():
            raise ValueError("overlength or empty protein")
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for layer in self.sequence_layers:
            hidden = layer(hidden, src_key_padding_mask=~valid.bool())
        hidden = self.sequence_norm(hidden) * valid[..., None]
        return self.sequence_output(hidden) * valid[..., None]


def sequence_rows(directory, split):
    """Stream only sequence columns; never deserialize geometry columns."""
    if split not in {"train", "validation"}:
        raise ValueError("unsupported split")
    for path in sorted((Path(directory) / split).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=list(COLUMNS)):
            for row in batch.to_pylist():
                tokens = row["token_ids"]
                if row["split"] != split or not 20 <= len(tokens) <= 500:
                    raise ValueError("split or length contradiction")
                encoded = ["ACDEFGHIKLMNPQRSTVWY".index(aa) + 2 for aa in row["sequence"]]
                if tokens != encoded:
                    raise ValueError("token/sequence contradiction")
                yield row


def collate(rows):
    tokens = torch.zeros(len(rows), max(len(r["token_ids"]) for r in rows), dtype=torch.long)
    valid = torch.zeros_like(tokens, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["token_ids"])
        tokens[i, :n] = torch.tensor(row["token_ids"])
        valid[i, :n] = True
    return tokens, valid


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") % (2**63 - 1)


def deterministic_mask(targets, valid, sample_ids, fraction, seed=6011, epoch=0):
    """Exact rounded mask count (at least one), keyed independently of batching."""
    if not 0 < fraction < 1 or len(sample_ids) != len(targets):
        raise ValueError("invalid fraction or sample IDs")
    selected = torch.zeros_like(valid, dtype=torch.bool)
    for i, sample_id in enumerate(sample_ids):
        positions = torch.where(valid[i].cpu().bool())[0]
        if not len(positions):
            raise ValueError("empty protein")
        g = torch.Generator().manual_seed(stable_seed(seed, epoch, sample_id, fraction, "mask"))
        count = max(1, int(len(positions) * fraction + 0.5))
        selected[i, positions[torch.randperm(len(positions), generator=g)[:count]].to(valid.device)] = True
    return selected


def conditions(targets, valid, selected, sample_ids, donors, seed=6011):
    """Same hidden targets/masks in four conditions; donors supply visible sites only.

    Donors must have a distinct sample and sequence and the same length stratum.
    Relative-position mapping accommodates different lengths. Donor selection
    is performed on the frozen panel.
    """
    normal = targets.clone().masked_fill(~valid.bool(), 0).masked_fill(selected, 1)
    shuffled, null, permuted = normal.clone(), normal.clone(), normal.clone()
    for i, sample_id in enumerate(sample_ids):
        positions = torch.where(valid[i].bool() & ~selected[i])[0]
        g = torch.Generator().manual_seed(stable_seed(seed, sample_id, "shuffle"))
        order = torch.randperm(len(positions), generator=g).to(positions.device)
        shuffled[i, positions] = normal[i, positions[order]]
        null[i, positions] = 1
        donor = donors[i]
        n = int(valid[i].sum())
        if (
            donor["sample_id"] == sample_id
            or bucket(len(donor["token_ids"])) != bucket(n)
            or donor["token_ids"] == targets[i, :n].tolist()
        ):
            raise ValueError("invalid permuted donor")
        donor_tokens = torch.tensor(donor["token_ids"], device=targets.device)
        donor_positions = positions * len(donor_tokens) // n
        permuted[i, positions] = donor_tokens[donor_positions]
    return {"normal": normal, "visible_shuffle": shuffled, "null_context": null, "permuted_context": permuted}


def paired_forwards(model, normal, shuffled, valid):
    """Identical CPU/device dropout RNG; global state advances once, even on error."""
    devices = [normal.device.index or 0] if normal.is_cuda else []
    cpu = torch.get_rng_state()
    cuda = [torch.cuda.get_rng_state(d) for d in devices]
    normal_logits = model(normal, valid)
    # fork restores the state after the normal branch, including on exceptions.
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu)
        for d, state in zip(devices, cuda, strict=True):
            torch.cuda.set_rng_state(state, d)
        with torch.no_grad():
            shuffled_logits = model(shuffled, valid)
    return normal_logits, shuffled_logits


def protein_ce(logits, targets, selected, valid):
    """Mean CE within each protein; only masked valid canonical targets count."""
    chosen = selected.bool() & valid.bool() & (targets >= 2) & (targets < 22)
    values = []
    for i in range(len(targets)):
        if not chosen[i].any():
            raise ValueError("protein has no masked valid canonical target")
        values.append(F.cross_entropy(logits[i, chosen[i]].float(), targets[i, chosen[i]] - 2))
    return torch.stack(values)


def objective(normal, shuffled, targets, selected, valid, weight=0.25, margin=0.05):
    n = protein_ce(normal, targets, selected, valid)
    s = protein_ce(shuffled, targets, selected, valid)
    return n.mean() + weight * F.relu(n - s.detach() + margin).mean()


def bucket(length):
    for i, (low, high) in enumerate(STRATA):
        if low <= length <= high:
            return i
    raise ValueError("length outside S1 strata")


def train_unigrams(rows, smoothing=1.0):
    counts = np.zeros((5, 20), dtype=np.int64)
    count = 0
    for row in rows:
        if row["split"] != "train":
            raise ValueError("baseline accepts TRAIN only")
        tokens = np.asarray(row["token_ids"]) - 2
        if np.any((tokens < 0) | (tokens >= 20)):
            raise ValueError("noncanonical baseline token")
        counts[bucket(len(tokens))] += np.bincount(tokens, minlength=20)
        count += 1
    if count == 0 or smoothing <= 0:
        raise ValueError("empty TRAIN or invalid smoothing")
    global_counts = counts.sum(axis=0)
    return {
        "train_count": count,
        "counts": counts.tolist(),
        "uniform": [0.05] * 20,
        "global": ((global_counts + smoothing) / (global_counts.sum() + 20 * smoothing)).tolist(),
        "length_bucketed": ((counts + smoothing) / (counts.sum(axis=1, keepdims=True) + 20 * smoothing)).tolist(),
    }


def paired_gate(normal, comparator, seed=6211, iterations=2000):
    """Equal-protein paired bootstrap; caller also applies per-stratum."""
    delta = np.asarray(normal, dtype=float) - np.asarray(comparator, dtype=float)
    if delta.ndim != 1 or len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("insufficient or nonfinite paired observations")
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(iterations)]
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "delta": float(delta.mean()),
        "ci95": [float(low), float(high)],
        "passes": bool(delta.mean() <= -0.05 and high < 0),
        "proteins": len(delta),
    }


def evaluate_panel(model, rows, donor_ids, baselines, fraction, seed=6111):
    """Emit paired per-protein CE for a frozen panel; no batch mixing of donors."""
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate panel IDs")
    device = next(model.parameters()).device
    previous_mode = model.training
    records = []
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                sid = row["sample_id"]
                targets, valid = collate([row])
                targets, valid = targets.to(device), valid.to(device)
                selected = deterministic_mask(targets, valid, [sid], fraction, seed)
                panel = conditions(targets, valid, selected, [sid], [by_id[donor_ids[sid]]], seed)
                metrics = {
                    name: float(protein_ce(model(tokens, valid), targets, selected, valid)[0])
                    for name, tokens in panel.items()
                }
                labels = targets[selected].cpu().numpy() - 2
                b = bucket(len(row["token_ids"]))
                for name, probabilities in (
                    ("uniform", baselines["uniform"]),
                    ("global_unigram", baselines["global"]),
                    ("bucket_unigram", baselines["length_bucketed"][b]),
                ):
                    metrics[name] = float(-np.log(np.asarray(probabilities)[labels]).mean())
                records.append({"sample_id": sid, "length_stratum": b, "ce": metrics})
    finally:
        model.train(previous_mode)
    return records


def gate_panel(records):
    """A requires both TRAIN unigrams; B-D are matched context interventions."""
    if len(records) < 64:
        return {"eligible": False, "all_pass": False}
    normal = [r["ce"]["normal"] for r in records]
    comparators = {
        "A_global": "global_unigram",
        "A_bucket": "bucket_unigram",
        "B": "visible_shuffle",
        "C": "null_context",
        "D": "permuted_context",
        "uniform": "uniform",
    }
    gates = {name: paired_gate(normal, [r["ce"][key] for r in records]) for name, key in comparators.items()}
    return {
        "eligible": True,
        "gates": gates,
        "all_pass": all(gates[k]["passes"] for k in ("A_global", "A_bucket", "B", "C", "D")),
    }


def classify_s1(panels):
    """Exactly one preregistered S1 label, with incomplete evidence handled first."""
    if set(panels) != {"0.15", "0.3", "0.5"}:
        return "S1-E"
    overall = [gate_panel(panels[key]) for key in sorted(panels)]
    strata = [
        [gate_panel([r for r in panels[key] if r["length_stratum"] == b]) for b in range(5)] for key in sorted(panels)
    ]
    if not all(x["eligible"] for x in overall) or not all(x["eligible"] for group in strata for x in group):
        return "S1-E"
    if all(x["all_pass"] for x in overall) and all(x["all_pass"] for group in strata for x in group):
        return "S1-A"
    if any(x["all_pass"] for x in overall) or any(x["all_pass"] for group in strata for x in group):
        return "S1-D"
    a_pass = [x["gates"]["A_global"]["passes"] and x["gates"]["A_bucket"]["passes"] for x in overall]
    context_fails = all(not x["gates"][k]["passes"] for x in overall for k in ("B", "C", "D"))
    if all(a_pass) and context_fails:
        return "S1-C"
    if not any(a_pass) and context_fails and all(x["gates"]["uniform"]["passes"] for x in overall):
        return "S1-B"
    return "S1-E"
