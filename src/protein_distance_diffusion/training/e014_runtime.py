"""D-only atomic persistence, deterministic complete epochs and exact resume."""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

STRATA = [(20, 64), (65, 128), (129, 256), (257, 384), (385, 500)]


def d_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to("/mnt/d"):
        raise ValueError(f"generated artifact escaped D: {path}")
    return path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, payload):
    path = d_path(path)
    temp = path.with_name(path.name + ".partial")
    with open(temp, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    json.loads(temp.read_text())
    os.replace(temp, path)


def epoch_schedule(lengths, batch_sizes, seed, epoch):
    """Shuffle within fixed length buckets, then shuffle batches; no resampling."""
    rng = np.random.Generator(np.random.PCG64(seed + epoch * 1000003))
    batches = []
    for (low, high), size in zip(STRATA, batch_sizes, strict=True):
        ids = np.flatnonzero((np.asarray(lengths) >= low) & (np.asarray(lengths) <= high))
        rng.shuffle(ids)
        batches.extend([ids[z : z + size].tolist() for z in range(0, len(ids), size)])
    rng.shuffle(batches)
    flattened = [i for batch in batches for i in batch]
    if sorted(flattened) != list(range(len(lengths))):
        raise ValueError("epoch must cover every identity exactly once")
    return batches


def select_panels(rows, development_per_stratum=100, prospective_per_stratum=400, exclude_ids=()):
    """Metric-blind SHA256 rank; disjoint dev/prospective and historical panel."""
    development, prospective = [], []
    excluded = set(exclude_ids)
    for low, high in STRATA:
        candidates = [r for r in rows if low <= r["length"] <= high and r["sample_id"] not in excluded]
        candidates.sort(key=lambda r: hashlib.sha256(f"e014-panel-v1|{r['sample_id']}".encode()).hexdigest())
        if len(candidates) < development_per_stratum + prospective_per_stratum:
            raise ValueError("insufficient clean validation candidates for frozen panels")
        development.extend(candidates[:development_per_stratum])
        prospective.extend(candidates[development_per_stratum : development_per_stratum + prospective_per_stratum])
    return development, prospective


def noise_seed(namespace, sample_id, timestep):
    return int.from_bytes(hashlib.sha256(f"{namespace}|{sample_id}|{timestep}".encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(directory, name, payload, rolling=False):
    directory = d_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pt"
    if rolling and name not in ("recovery_0", "recovery_1"):
        raise ValueError("rolling replacement restricted to two E014 recovery slots")
    if rolling and (directory / "LATEST.json").exists():
        previous = json.loads((directory / "LATEST.json").read_text())
        if previous["path"] == str(path):
            path = directory / ("recovery_1.pt" if name == "recovery_0" else "recovery_0.pt")
    if path.exists() and not rolling:
        raise FileExistsError(f"checkpoint boundary already committed: {path}")
    temp = path.with_suffix(".pt.partial")
    with open(temp, "wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    restored = torch.load(temp, map_location="cpu", weights_only=False)

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
        if isinstance(left, np.ndarray):
            return isinstance(right, np.ndarray) and np.array_equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
        return left == right

    if not equal(payload, restored):
        raise ValueError("full checkpoint/optimizer/RNG verification failed")
    if restored["contract"] != payload["contract"] or restored["position"] != payload["position"]:
        raise ValueError("checkpoint verification failed")
    for key, value in payload["model"].items():
        if not torch.equal(value.detach().cpu(), restored["model"][key]):
            raise ValueError("checkpoint tensor verification failed")
    if not np.array_equal(np.asarray(payload["exposures"]), np.asarray(restored["exposures"])):
        raise ValueError("checkpoint exposure verification failed")
    digest = sha(temp)
    os.replace(temp, path)
    atomic_json(directory / "LATEST.json", dict(path=str(path), sha256=digest, position=payload["position"]))
    return path


def load_checkpoint(directory, contract):
    directory = d_path(directory)
    latest = json.loads((directory / "LATEST.json").read_text())
    path = d_path(latest["path"])
    if not path.is_relative_to(directory) or sha(path) != latest["sha256"]:
        raise ValueError("checkpoint integrity failure")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["contract"] != contract:
        raise ValueError("config/data/normalization/source hash mismatch; refusing resume")
    return state
