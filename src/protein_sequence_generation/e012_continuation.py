"""E012 continuation execution semantics; historical architecture and gates are reused."""

from __future__ import annotations

import math

import numpy as np
import torch

SOURCE_SHA256 = "ff666eccdb6d28d51f2d3d561242152512e2dddf74fc0f251429d55b886d0188"
EFFECTIVE_BATCH = 64


def continuation_lr(global_update, lr_checkpoint, maximum=0.0003):
    """LR used by update u: checkpoint LR at 2000, linear restart then terminal cosine."""
    if not 2000 <= global_update <= 10000:
        raise ValueError("continuation update outside [2000,10000]")
    if global_update <= 2100:
        return lr_checkpoint + (maximum - lr_checkpoint) * (global_update - 2000) / 100
    return maximum * 0.5 * (1 + math.cos(math.pi * (global_update - 2100) / 7900))


class ContinuationScheduler:
    """Declared new scheduler; never loads or reinterprets the historical LambdaLR."""

    def __init__(self, optimizer, lr_checkpoint):
        self.optimizer = optimizer
        self.lr_checkpoint = float(lr_checkpoint)
        self.global_update = 2000
        self.set_update(2000)

    def set_update(self, update):
        lr = continuation_lr(update, self.lr_checkpoint)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.global_update = update
        return lr

    def state_dict(self):
        return {
            "kind": "explicit_2k_to_10k_warm_restart_v1",
            "global_update": self.global_update,
            "lr_checkpoint": self.lr_checkpoint,
            "maximum": 0.0003,
            "restart_end": 2100,
            "end": 10000,
        }

    def load_state_dict(self, state):
        if state["kind"] != "explicit_2k_to_10k_warm_restart_v1":
            raise ValueError("historical scheduler is not a continuation scheduler")
        assert state["maximum"] == 0.0003 and state["restart_end"] == 2100 and state["end"] == 10000
        self.lr_checkpoint = state["lr_checkpoint"]
        self.set_update(state["global_update"])


def partition64(capacity):
    if not isinstance(capacity, int) or not 1 <= capacity <= 64:
        raise ValueError("invalid physical capacity")
    full, remainder = divmod(64, capacity)
    return [capacity] * full + ([remainder] if remainder else [])


def routed_microbatches(rows, plan):
    """Keep optimizer-batch membership; sort inside its 64 identities only."""
    from protein_sequence_generation.e012 import stratum

    if len(rows) != 64:
        raise ValueError("optimizer batch must contain exactly 64 proteins")
    ordered = sorted(rows, key=lambda r: (r["length"], r["sample_id"]))
    parts = plan[str(stratum(max(r["length"] for r in rows)))]["partition"]
    if sum(parts) != 64 or any(n < 1 for n in parts):
        raise ValueError("frozen partitions must sum to 64")
    cursor, result = 0, []
    for n in parts:
        result.append(ordered[cursor : cursor + n])
        cursor += n
    return result


def weighted_loss(mean_protein_loss, protein_count):
    if not 1 <= protein_count <= 64:
        raise ValueError("invalid microbatch protein count")
    return mean_protein_loss * (protein_count / 64)


class IdentitySampler:
    """Resume exact historical permutation/cursor; deterministically reshuffle only at epoch boundary."""

    def __init__(self, order, cursor, epoch=0):
        self.order = np.asarray(order, dtype=np.int64).copy()
        self.cursor = int(cursor)
        self.epoch = int(epoch)
        assert 0 <= self.cursor <= len(self.order)

    def take64(self):
        chosen = []
        while len(chosen) < 64:
            if self.cursor == len(self.order):
                self.epoch += 1
                self.order = np.random.default_rng(12012 + self.epoch).permutation(len(self.order))
                self.cursor = 0
            n = min(64 - len(chosen), len(self.order) - self.cursor)
            chosen.extend(self.order[self.cursor : self.cursor + n].tolist())
            self.cursor += n
        return chosen

    def state_dict(self):
        return {
            "order": self.order.copy(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "new_epoch_seed": "12012 + epoch",
        }

    @classmethod
    def from_state(cls, state):
        assert state["new_epoch_seed"] == "12012 + epoch"
        return cls(state["order"], state["cursor"], state["epoch"])


def finite_state(net, optimizer):
    return all(torch.isfinite(p).all() for p in net.parameters()) and all(
        torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)
    )


def continuation_classification(primary, independent, cseq):
    """Prioritize principal likelihood-versus-stratum blocker when context survives."""
    if cseq == "CSEQ-A":
        return "CONT-A"
    strong = all(primary["gates"][k] and independent["confirmation"][k] for k in "CD")
    likelihood = all(primary["gates"][k] and independent["confirmation"][k] for k in "AB")
    if strong and likelihood and (not primary["gates"]["E"] or not independent["confirmation"]["E"]):
        return "CONT-D"
    if strong:
        return "CONT-B"
    return "CONT-C"
