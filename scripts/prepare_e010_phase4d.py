#!/usr/bin/env python3
"""CPU preparation only: freeze training panel, provenance and gradient telemetry.

No optimizer, no update, no CUDA. Historical inputs are read only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import load_frozen_hybrid
from protein_distance_diffusion.training.e010_phase4d import (
    aggregate_metrics,
    distance_diversity,
    example_metrics,
    hybrid_losses,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1"
HISTORY = Path("/home/simostocco/proteinGen/reports/experiments/E010_global_equivariant_expressivity")
MANIFEST = HISTORY / "phase4a_multicorruption_v2/training_seed_manifest.json"
CACHE = HISTORY / "phase4b_real_denoiser_v1/phase4b_real_denoiser_v1.final"
MANIFEST_SHA = "69696551bbfbe00c0b7140a6e1b6e3c79de0f050dba25269ccf3068dd9f56cd0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def select_panel(identities, namespace):
    selected = []
    for stratum in ("20-64", "65-128", "129-256", "257-384", "385-500"):
        candidates = [r for r in identities if r["stratum"] == stratum]

        def rank(r):
            return (hashlib.sha256(f"{namespace}|train|{r['sample_id']}".encode()).hexdigest(), r["sample_id"])

        candidates.sort(key=rank)
        if len(candidates) < 4:
            raise ValueError("insufficient training identities")
        selected.extend(
            {"sample_id": r["sample_id"], "length": r["length"], "stratum": stratum, "rank_sha256": rank(r)[0]}
            for r in candidates[:4]
        )
    if len({r["sample_id"] for r in selected}) != 20:
        raise ValueError("duplicate identities")
    return selected


def prepare():
    torch.set_num_threads(2)
    torch.manual_seed(41047)
    cfg = yaml.safe_load((ROOT / "configs/e010_phase4d_hybrid_local_global_v1.yaml").read_text())
    if sha(MANIFEST) != MANIFEST_SHA:
        raise ValueError("training manifest hash")
    cm = json.loads((CACHE / "manifest.json").read_text())
    coverage = {}
    for shard in cm["shards"]:
        for r in shard["records"]:
            if r["split"] == "train":
                coverage.setdefault(r["sample_id"], set()).add(r["timestep"])
    identities = [
        r
        for r in json.loads(MANIFEST.read_text())["identities"]
        if coverage.get(r["sample_id"], set()) == {50, 250, 450}
    ]
    panel = select_panel(identities, cfg["panel"]["namespace"])
    ids = {r["sample_id"] for r in panel}
    chosen = {}
    for shard in cm["shards"]:
        for index, r in enumerate(shard["records"]):
            if r["split"] != "train" or r["sample_id"] not in ids:
                continue
            key = (r["sample_id"], r["timestep"])
            if key not in chosen or (r["seed"], r["schedule_index"]) < (
                chosen[key][2]["seed"],
                chosen[key][2]["schedule_index"],
            ):
                chosen[key] = (shard, index, r)
    if len(chosen) != 60:
        raise ValueError("incomplete 60-example panel")
    model = load_frozen_hybrid(cfg["global"]["checkpoint"])
    grads = {
        k: [torch.zeros_like(p) for p in model.local.parameters()]
        for k in ("local_mean", "cartesian", "guard", "displacement")
    }
    global_state_hash = hashlib.sha256()
    for name, tensor in sorted(model.global_model.state_dict().items()):
        global_state_hash.update(name.encode())
        global_state_hash.update(str(tensor.dtype).encode())
        global_state_hash.update(str(tuple(tensor.shape)).encode())
        global_state_hash.update(tensor.contiguous().numpy().tobytes())
    diversity_inputs = {}
    metrics = []
    examples = []
    parity = True
    eligible = degenerate = 0
    for key in sorted(chosen):
        shard, index, r = chosen[key]
        path = CACHE / shard["shard"]
        if sha(path) != shard["archive_sha256"]:
            raise ValueError("cache archive hash")
        with np.load(path, allow_pickle=False) as z:
            lo, hi = z["offsets"][index : index + 2]
            xx = z["prediction"][lo:hi].copy()
            yy = z["target"][lo:hi].copy()
        for arr, label in ((xx, "prediction"), (yy, "target")):
            if hashlib.sha256(arr.tobytes()).hexdigest() != r[f"{label}_sha256"]:
                raise ValueError("tensor hash")
        x, y = torch.from_numpy(xx)[None], torch.from_numpy(yy)[None]
        m = torch.ones(x.shape[:2], dtype=torch.bool)
        out = model(x, m)
        diversity_inputs.setdefault(r["sample_id"], []).append(out["global_prediction"][0].detach())
        parity &= torch.equal(out["prediction"], out["global_prediction"])
        losses = hybrid_losses(out["prediction"], out["global_prediction"], x, y, m)
        for name in grads:
            gg = torch.autograd.grad(
                losses[name], tuple(model.local.parameters()), retain_graph=True, allow_unused=True
            )
            for acc, g in zip(grads[name], gg, strict=True):
                if g is not None:
                    acc.add_(g / 60)
        metric = example_metrics(out["prediction"], out["global_prediction"], x, y, m)[0]
        metric.update(sample_id=r["sample_id"], condition=r["timestep"], stratum=r["stratum"])
        metrics.append(metric)
        eligible += int(out["eligible"].sum())
        degenerate += int(out["degenerate"].sum())
        examples.append(
            {**r, "shard": shard["shard"], "archive_sha256": shard["archive_sha256"], "index_in_shard": index}
        )
    norms = {k: float(sum(g.double().square().sum() for g in vv).sqrt()) for k, vv in grads.items()}
    dot = sum((a.double() * b.double()).sum() for a, b in zip(grads["local_mean"], grads["cartesian"], strict=True))
    cosine = (
        float(dot / (norms["local_mean"] * norms["cartesian"])) if norms["cartesian"] * norms["local_mean"] else None
    )
    assert parity and norms["local_mean"] > 0
    assert all(p.grad is None and not p.requires_grad for p in model.global_model.parameters())
    assert all(torch.isfinite(g).all() for vv in grads.values() for g in vv)
    write(
        "tiny_panel.json",
        {
            "namespace": cfg["panel"]["namespace"],
            "eligibility": "training identities with cached records at all three conditions; no metric filtering",
            "training_manifest_sha256": MANIFEST_SHA,
            "cache_manifest_sha256": sha(CACHE / "manifest.json"),
            "identities": panel,
            "examples": examples,
        },
    )
    write(
        "preparation.json",
        {
            "training_launched": False,
            "device": "cpu",
            "checkpoint_sha256": sha(cfg["global"]["checkpoint"]),
            "checkpoint_model_key": "model",
            "model_state_sha256": global_state_hash.hexdigest(),
            "model_state_hash_convention": (
                "sorted keys; key, dtype, tuple shape UTF-8 then contiguous CPU tensor bytes"
            ),
            "source_checkpoint_metadata": model.source_metadata,
            "condition_distance_diversity_by_identity": {
                sid: distance_diversity(xs, [torch.ones(x.shape[0], dtype=torch.bool) for x in xs])
                for sid, xs in diversity_inputs.items()
            },
            "global_parameter_count": sum(p.numel() for p in model.global_model.parameters()),
            "local_parameter_count": sum(p.numel() for p in model.local.parameters()),
            "zero_init_exact_parity_60_examples": parity,
            "global_gradient_isolation": True,
            "local_gradient_norms_equal_example_mean": norms,
            "local_cartesian_gradient_cosine": cosine,
            "frame_eligible": eligible,
            "frame_degenerate": degenerate,
            "metrics": aggregate_metrics(metrics),
            "gradient_audit_scope": "initialization only; guard/displacement inactive; no step or optimizer update",
            "coefficients_proposed": cfg["objective_candidate"],
        },
    )
    print(
        json.dumps(
            {"examples": 60, "norms": norms, "eligible": eligible, "degenerate": degenerate, "zero_parity": parity}
        )
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
