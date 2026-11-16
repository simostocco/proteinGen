"""D:-only E012 certified-corpus warm start, bounded preflight and resumable training."""

# Environment setup must precede numerical imports; long prose embeds the D: report.
# ruff: noqa: E402, E501
from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(
    "/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1"
)
RUN = Path("/mnt/d/Simone/proteinGen_data/sequence_foundation/e012_large_corpus_warmstart_20m_20261013_v1")
PARENT = ROOT / "outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1/checkpoint-10000.pt"
PARENT_SHA = "23d4aad8327c491d31d22192e2b4f51bda8e68f13e1c213ed7925cfcafb7bf89"
HIST = ROOT / "reports/experiments/E012_causal_rope_sequence/pilot_v1"
SEED = 22012
# Set every artifact/cache path before importing numerical libraries.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for key in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "TORCH_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "HF_HOME",
    "MPLCONFIGDIR",
):
    os.environ[key] = str(RUN / "scratch" / key.lower())
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.nn import functional as F

from protein_sequence_generation.e012 import VOCAB, batch, paired_interval, positions, stratum
from protein_sequence_generation.e012_generalization import bulk_prefix_batch
from protein_sequence_generation.metrics import sequence_cross_entropy
from protein_sequence_generation.model import ProteinSequenceTransformer, parameter_count
from protein_sequence_generation.sampling import generate_sequences

CONFIG = {
    "experiment": "large-corpus warm-start continuation",
    "seed": SEED,
    "physical_batch": 16,
    "effective_batch": 64,
    "accumulation": 4,
    "optimizer": "AdamW",
    "betas": [0.9, 0.95],
    "weight_decay": 0.01,
    "maximum_lr": 0.0001,
    "minimum_lr": 0.00001,
    "warmup_updates": 500,
    "schedule_horizon_updates": 312500,
    "gradient_clip": 1.0,
    "precision": "BF16 forward, FP32 weights/moments/loss",
    "loss": "sequence_mean",
    "milestone_unique_exposures": 1000000,
    "full_epoch_exposures": 20000000,
    "rolling_checkpoint_updates": 250,
    "light_validation_updates": 1000,
    "named_exposures": [250000, 500000, 750000, 1000000],
    "smoke_updates": [100, 1000],
    "smoke_counts_in_production": False,
    "minimum_d_free_bytes": 8 * 1024**3,
    "maximum_cuda_reserved_bytes": 5 * 1024**3,
    "minimum_wsl_available_bytes": 3 * 1024**3,
    "maximum_swap_used_bytes": 512 * 1024**2,
    "minimum_host_free_bytes": 4 * 1024**3,
}
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(name, value):
    p = RUN / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".pending")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def append(name, value):
    with (RUN / name).open("a") as f:
        f.write(json.dumps(value, sort_keys=True) + "\n")
        f.flush()


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1) if not getattr(seed, "done", False) else None
    seed.done = True
    torch.use_deterministic_algorithms(True)


def rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(r):
    random.setstate(r["python"])
    np.random.set_state(r["numpy"])
    torch.set_rng_state(r["cpu"])
    torch.cuda.set_rng_state_all(r["cuda"])


def architecture():
    return yaml.safe_load((ROOT / "configs/e012_causal_rope_sequence_v1.yaml").read_text())["model"]


def inventory():
    return {
        line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in (DATA / "corpus/SHA256SUMS").read_text().splitlines()
    }


class Cursor:
    """One shuffled shard at a time; a bijection over every certified row per epoch."""

    def __init__(self, state=None):
        self.epoch, self.shard_pos, self.row_pos, self.exposures = (0, 0, 0, 0)
        if state:
            self.epoch, self.shard_pos, self.row_pos, self.exposures = (
                state[k] for k in ("epoch", "shard_pos", "row_pos", "exposures")
            )
        self.loaded = None
        self.rows = None
        self.expected = inventory()

    def state(self):
        return {k: getattr(self, k) for k in ("epoch", "shard_pos", "row_pos", "exposures")} | {
            "seed": SEED,
            "algorithm": "PCG64 shuffled shards then PCG64 shuffled rows v1",
        }

    def take(self, n):
        out = []
        while len(out) < n:
            if self.shard_pos == 512:
                self.epoch += 1
                self.shard_pos = 0
                self.row_pos = 0
            sid = int(np.random.default_rng(SEED + self.epoch).permutation(512)[self.shard_pos])
            key = (self.epoch, sid)
            if key != self.loaded:
                rel = f"corpus/shards/part-{sid:03d}.parquet"
                assert sha(DATA / rel) == self.expected[rel], rel
                self.rows = pq.read_table(DATA / rel, use_threads=False).to_pylist()
                self.order = np.random.default_rng(np.random.SeedSequence([SEED, self.epoch, sid, 1])).permutation(
                    len(self.rows)
                )
                self.loaded = key
            count = min(n - len(out), len(self.rows) - self.row_pos)
            out.extend(self.rows[int(i)] for i in self.order[self.row_pos : self.row_pos + count])
            self.row_pos += count
            self.exposures += count
            if self.row_pos == len(self.rows):
                self.shard_pos += 1
                self.row_pos = 0
        return out


def resources(host=False):
    mem = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in ["MemAvailable", "MemTotal", "SwapFree", "SwapTotal"]
    }
    disk = os.statvfs("/mnt/d")
    result = {
        "time": time.time(),
        "wsl": mem,
        "d_free_bytes": disk.f_bavail * disk.f_frsize,
        "load_average": os.getloadavg(),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip(),
    }
    if host:
        command = '$o=Get-CimInstance Win32_OperatingSystem; @{free_bytes=([long]$o.FreePhysicalMemory*1024);total_bytes=([long]$o.TotalVisibleMemorySize*1024);processes=@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match "python|mmseqs|colab|fold"} | Select-Object ProcessId,Name,CommandLine)} | ConvertTo-Json -Depth 5 -Compress'
        result["host"] = json.loads(
            subprocess.check_output(["powershell.exe", "-NoProfile", "-Command", command], text=True)
        )
    return result


def guard(ownership=False, host=False):
    r = resources(host=host)
    assert r["d_free_bytes"] > CONFIG["minimum_d_free_bytes"], "D: free-space gate"
    assert r["wsl"]["MemAvailable"] > CONFIG["minimum_wsl_available_bytes"], "WSL RAM gate"
    assert r["wsl"]["SwapTotal"] - r["wsl"]["SwapFree"] < CONFIG["maximum_swap_used_bytes"], "swap gate"
    if host:
        assert r["host"]["free_bytes"] > CONFIG["minimum_host_free_bytes"], "host RAM gate"
        assert not r["host"]["processes"], "native Windows research workload active"
    if ownership:
        ps = subprocess.run(
            ["ps", "-C", "python", "-C", "python3", "-o", "pid=,args="], text=True, capture_output=True
        ).stdout
        conflicts = [line for line in ps.splitlines() if int(line.split()[0]) != os.getpid()]
        assert not conflicts, f"Other Python workloads: {conflicts}"
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True
        ).strip()
        assert not gpu or all(str(os.getpid()) in line for line in gpu.splitlines()), gpu
        r["wsl_python_processes"] = ps
        r["compute_owners"] = gpu
    if torch.cuda.is_initialized():
        assert torch.cuda.max_memory_reserved() < CONFIG["maximum_cuda_reserved_bytes"], "CUDA memory ceiling"
    return r


def model_optimizer(checkpoint=None, *, allow_runner_change=False):
    seed()
    net = ProteinSequenceTransformer(architecture()).cuda()
    c = torch.load(checkpoint or PARENT, map_location="cpu", weights_only=False)
    net.load_state_dict(c["model"], strict=True)
    assert parameter_count(net) == 15533952
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    opt = torch.optim.AdamW(
        net.parameters(), lr=CONFIG["maximum_lr"], betas=tuple(CONFIG["betas"]), weight_decay=CONFIG["weight_decay"]
    )
    state = {"updates": 0, "exposures": 0, "tokens": 0}
    cursor = Cursor()
    if checkpoint:
        assert c["config"] == CONFIG and c["parent_sha256"] == PARENT_SHA
        assert c["protocol_sha256"] == sha(RUN / "protocol.json")
        current_sources = source_hashes()
        previous_sources = c["source_hashes"].copy()
        if allow_runner_change:
            # Disposable prelaunch replay permits runner-only cleanup; model/data remain pinned.
            previous_sources.pop("scripts/run_e012_large_corpus.py")
            current_sources.pop("scripts/run_e012_large_corpus.py")
        assert previous_sources == current_sources
        assert c["dataset_hashes"] == json.loads((RUN / "audit.json").read_text())["dataset_hashes"]
        assert c["vocabulary"]["tokens"] == list(VOCAB.tokens)
        opt.load_state_dict(c["optimizer"])
        cursor = Cursor(c["identity_cursor"])
        state = c["counters"]
        assert cursor.exposures == state["exposures"]
        assert c["scheduler"]["completed_updates"] == state["updates"]
        restore_rng(c["rng"])
    else:
        assert not opt.state, "warm start must have fresh moments"
    return net, opt, cursor, state


def lr(update):
    if update <= CONFIG["warmup_updates"]:
        return CONFIG["maximum_lr"] * update / CONFIG["warmup_updates"]
    t = (update - CONFIG["warmup_updates"]) / (CONFIG["schedule_horizon_updates"] - CONFIG["warmup_updates"])
    return (
        CONFIG["minimum_lr"] + (CONFIG["maximum_lr"] - CONFIG["minimum_lr"]) * (1 + math.cos(math.pi * min(t, 1))) / 2
    )


def source_hashes():
    files = ["scripts/run_e012_large_corpus.py", "configs/e012_causal_rope_sequence_v1.yaml"]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "src/protein_sequence_generation").glob("*.py"))]
    return {p: sha(ROOT / p) for p in files}


def checkpoint(name, net, opt, cursor, counters):
    payload = {
        "model": net.state_dict(),
        "optimizer": opt.state_dict(),
        "rng": rng(),
        "scheduler": {
            "kind": "linear_warmup_cosine_full_20m_v1",
            "completed_updates": counters["updates"],
            "next_lr": lr(counters["updates"] + 1),
        },
        "identity_cursor": cursor.state(),
        "counters": counters.copy(),
        "config": CONFIG,
        "model_config": architecture(),
        "vocabulary": {"tokens": list(VOCAB.tokens)},
        "tokenizer_sha256": digest(list(VOCAB.tokens)),
        "parent_checkpoint": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "protocol_sha256": sha(RUN / "protocol.json"),
        "source_hashes": source_hashes(),
        "dataset_hashes": json.loads((RUN / "audit.json").read_text())["dataset_hashes"],
    }
    p = RUN / name
    temp = p.with_suffix(".pending")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, p)
    save(name + ".json", {"sha256": sha(p), "counters": counters, "path": str(p)})


def update(net, opt, cursor, counters, count=64):
    rows = cursor.take(count)
    assert all(20 <= r["length"] <= 500 and r["length"] == len(r["sequence"]) for r in rows)
    net.train()
    opt.zero_grad(set_to_none=True)
    loss_mean, loss_sum, correct = 0.0, 0.0, 0
    # Sorting inside the already-selected optimizer batch preserves sampling probabilities.
    ordered = sorted(rows, key=lambda r: (r["length"], r["sample_id"]))
    for start in range(0, count, CONFIG["physical_batch"]):
        chosen = ordered[start : start + CONFIG["physical_batch"]]
        b = batch(chosen, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(b["input_ids"], b["lengths"], b["attention_mask"])
            loss = sequence_cross_entropy(logits, b["target_ids"], b["attention_mask"], reduction="sequence_mean")
        assert torch.isfinite(loss)
        (loss * len(chosen) / count).backward()
        loss_mean += float(loss.detach()) * len(chosen) / count
        with torch.no_grad():
            ce = F.cross_entropy(logits.float().flatten(0, 1), b["target_ids"].flatten(), reduction="none").view_as(
                b["target_ids"]
            )
            loss_sum += float((ce * b["attention_mask"]).sum())
            correct += int(((logits.argmax(-1) == b["target_ids"]) & b["attention_mask"]).sum())
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True))
    used_lr = lr(counters["updates"] + 1)
    for group in opt.param_groups:
        group["lr"] = used_lr
    opt.step()
    tokens = sum(r["length"] for r in rows)
    counters["updates"] += 1
    counters["exposures"] += count
    counters["tokens"] += tokens
    assert counters["exposures"] == cursor.exposures
    assert torch.cuda.max_memory_reserved() < CONFIG["maximum_cuda_reserved_bytes"]
    return {
        "updates": counters["updates"],
        "unique_sequence_exposures": min(counters["exposures"], 20000000),
        "sequence_exposures": counters["exposures"],
        "token_exposures": counters["tokens"],
        "train_sequence_ce": loss_mean,
        "train_token_ce": loss_sum / tokens,
        "train_token_accuracy": correct / tokens,
        "gradient_norm": norm,
        "learning_rate": used_lr,
        "batch_tokens": tokens,
        "length_min": min(r["length"] for r in rows),
        "length_max": max(r["length"] for r in rows),
        "length_mean": tokens / count,
        "identities_sha256": digest([r["sample_id"] for r in rows]),
    }


def aggregate(records):
    def metrics(rr):
        ce = float(np.mean([r["normal"] for r in rr]))
        tokens = sum(r["length"] for r in rr)
        tc = sum(r["loss_sum"] for r in rr) / tokens
        return {
            "proteins": len(rr),
            "tokens": tokens,
            "sequence_ce": ce,
            "sequence_perplexity": math.exp(ce),
            "token_ce": tc,
            "token_perplexity": math.exp(tc),
            "token_accuracy": sum(r["top1"] * r["length"] for r in rr) / tokens,
            "corpus_composition_ce": float(np.mean([r["corpus_composition_ce"] for r in rr])),
            "model_minus_corpus_composition": paired_interval([r["normal"] - r["corpus_composition_ce"] for r in rr]),
        }

    return {
        "aggregate": metrics(records),
        "strata": {str(s): metrics([r for r in records if stratum(r["length"]) == s]) for s in range(5)},
    }


@torch.no_grad()
def likelihood(net, rows):
    net.eval()
    result = []
    frequencies = json.loads((RUN / "audit.json").read_text())["corpus"]["aa_frequencies"]
    for start in range(0, len(rows), 16):
        rr = rows[start : start + 16]
        b = batch(rr, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(b["input_ids"], b["lengths"], b["attention_mask"])
        loss = -logits.float().log_softmax(-1).gather(-1, b["target_ids"][..., None]).squeeze(-1)
        for j, r in enumerate(rr):
            v = loss[j, : r["length"]].cpu().numpy()
            result.append(
                {
                    "sample_id": r["sample_id"],
                    "length": r["length"],
                    "normal": float(v.mean()),
                    "loss_sum": float(v.sum()),
                    "corpus_composition_ce": -sum(math.log(frequencies[a]) for a in r["sequence"]) / r["length"],
                    "top1": float(
                        (logits[j, : r["length"]].argmax(-1) == b["target_ids"][j, : r["length"]]).float().mean()
                    ),
                }
            )
    return result


@torch.no_grad()
def context(net, rows):
    net.eval()
    cases = [(r, i) for r in rows for i in positions(r["length"])]
    cases.sort(key=lambda c: c[1])
    values = {r["sample_id"]: collections.defaultdict(list) for r in rows}
    for start in range(0, len(cases), 16):
        chosen = cases[start : start + 16]
        for condition in ["prefix_normal", "shuffle", "last8", "last1"]:
            b = bulk_prefix_batch(
                chosen,
                "cuda",
                shuffled=condition == "shuffle",
                window=int(condition[4:]) if condition.startswith("last") else None,
            )
            inputs, lengths, mask, indices, targets = b
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(inputs, lengths, mask)
            ce = -(
                out[torch.arange(len(chosen), device="cuda"), indices]
                .float()
                .log_softmax(-1)
                .gather(-1, targets[:, None])
                .squeeze(-1)
            )
            ce = ce.cpu().tolist()
            for (r, _), v in zip(chosen, ce, strict=False):
                values[r["sample_id"]][condition].append(v)
    records = [
        {
            "sample_id": r["sample_id"],
            "length": r["length"],
            **{k: float(np.mean(v)) for k, v in values[r["sample_id"]].items()},
        }
        for r in rows
    ]
    summary = {
        "means": {k: float(np.mean([r[k] for r in records])) for k in ["prefix_normal", "shuffle", "last8", "last1"]},
        "normal_minus_control": {
            k: paired_interval([r["prefix_normal"] - r[k] for r in records]) for k in ["shuffle", "last8", "last1"]
        },
        "strata": {
            str(s): {
                k: paired_interval([r["prefix_normal"] - r[k] for r in records if stratum(r["length"]) == s])
                for k in ["shuffle", "last8", "last1"]
            }
            for s in range(5)
        },
    }
    return records, summary


def eval_panels():
    panels = json.loads((RUN / "panels.json").read_text())
    for name in ["primary", "independent"]:
        panels[name] = sorted(panels[name], key=lambda r: (r["length"], r["sample_id"]))
    return panels


def evaluate(net, label, full=False, full_likelihood=False):
    saved = rng()
    panels = eval_panels()
    results = {}
    for name, all_rows in panels.items():
        if name == "train_reference":
            continue
        rows = all_rows
        if not full and not full_likelihood:
            rows = [r for s in range(5) for r in [x for x in all_rows if stratum(x["length"]) == s][:52]]
        records = likelihood(net, rows)
        results[name] = aggregate(records)
        if full or full_likelihood:
            save(f"evaluations/{label}_{name}_likelihood.json", records)
        if full:
            crecords, summary = context(net, rows)
            save(f"evaluations/{label}_{name}_context.json", crecords)
            results[name]["context"] = summary
        if label == "parent":
            previous = (
                ROOT / f"outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1/{name}_likelihood_10000.json"
            )
            prior = json.loads(previous.read_text())
            assert [r["sample_id"] for r in records] == [r["sample_id"] for r in prior]
            difference = max(abs(a["normal"] - b["normal"]) for a, b in zip(records, prior, strict=False))
            assert difference < 0.0001, "historical likelihood parity failure"
            results[name]["historical_likelihood_parity"] = {
                "max_ce_difference": difference,
                "historical_records_sha256": sha(previous),
            }
        print("evaluation", label, name, results[name]["aggregate"], flush=True)
    save(f"evaluations/{label}.json", results)
    restore_rng(saved)
    return results


def generation(net, label):
    saved = rng()
    seqs = []
    for length in [64, 128]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            seqs += generate_sequences(
                net,
                vocabulary=VOCAB,
                length=length,
                num_sequences=16,
                seed=32012 + length,
                device="cuda",
                temperature=1.0,
                top_p=0.95,
            )
    panels = eval_panels()
    refs = {"sampled_train": panels["train_reference"], "protected_panels": panels["primary"] + panels["independent"]}
    nearest = []

    def kmers(s):
        return {s[i : i + 3] for i in range(len(s) - 2)}

    for group, rows in refs.items():
        indexed = [(r, kmers(r["sequence"])) for r in rows]
        for s in seqs:
            ks = kmers(s)
            selected = sorted(indexed, key=lambda x: len(ks & x[1]) / max(1, len(ks | x[1])), reverse=True)[:10]
            rr = max(
                selected, key=lambda x: difflib.SequenceMatcher(None, s, x[0]["sequence"], autojunk=False).ratio()
            )[0]
            nearest.append(
                {
                    "group": group,
                    "generated_sha256": hashlib.sha256(s.encode()).hexdigest(),
                    "reference_id": rr["sample_id"],
                    "sequence_matcher_ratio": difflib.SequenceMatcher(None, s, rr["sequence"], autojunk=False).ratio(),
                    "exact": any(s == r["sequence"] for r in rows),
                }
            )
    counts = collections.Counter("".join(seqs))
    total = sum(counts.values())
    homopolymers = [max(len(list(g)) for _, g in __import__("itertools").groupby(s)) for s in seqs]
    result = {
        "sequences": seqs,
        "canonical_valid": all(set(s) <= set("ACDEFGHIKLMNPQRSTVWY") for s in seqs),
        "length_adherence": all(len(s) == ([64] * 16 + [128] * 16)[i] for i, s in enumerate(seqs)),
        "unique": len(set(seqs)),
        "n": len(seqs),
        "aa_frequencies": {a: counts[a] / total for a in "ACDEFGHIKLMNPQRSTVWY"},
        "composition_entropy_bits": -sum((c / total) * math.log2(c / total) for c in counts.values()),
        "maximum_homopolymer": max(homopolymers),
        "mean_unique_trigram_fraction": float(np.mean([len(kmers(s)) / (len(s) - 2) for s in seqs])),
        "max_generated_pair_sequence_matcher_ratio": max(
            difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
            for i, a in enumerate(seqs)
            for b in seqs[i + 1 :]
        ),
        "nearest": nearest,
        "search_scope": {k: len(v) for k, v in refs.items()},
        "limitations": "3mer Jaccard shortlist then SequenceMatcher ratio is a bounded heuristic, not full-corpus or sensitive homology alignment. Canonical validity and length are enforced by sampling. No folding/stability inference.",
    }
    save(f"generation/{label}.json", result)
    restore_rng(saved)


def prepare():
    assert not (RUN / "protocol.json").exists(), "isolated experiment already prepared"
    RUN.mkdir(parents=True, exist_ok=True)
    for v in ["scratch"] + [
        "scratch/" + k.lower()
        for k in (
            "TMPDIR",
            "TMP",
            "TEMP",
            "XDG_CACHE_HOME",
            "TORCH_HOME",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "HF_HOME",
            "MPLCONFIGDIR",
        )
    ]:
        (RUN / v).mkdir(parents=True, exist_ok=True)
    save("resources_initial.json", guard(ownership=True, host=True))
    assert sha(PARENT) == PARENT_SHA
    handoff = json.loads((DATA / "reports/chatgpt_handoff.json").read_text())
    cert = json.loads((DATA / "corpus/certification.json").read_text())
    assert (
        handoff["final_corpus_n"] == 20000000
        and handoff["total_residues"] == 4042471015
        and handoff["protected_overlap_zero"]
    )
    assert (
        cert["certified"]
        and cert["final_exact_unique"]
        and cert["protected_exact_overlap"] == cert["protected_detected_homology_overlap"] == 0
    )
    frozen = json.loads((HIST / "contract.json").read_text())
    for name, h in frozen["protected_hashes"].items():
        assert sha(ROOT / name) == h, name
    assert (
        sha(ROOT / "outputs/e012_causal_rope_sequence/pilot_v1/validation.parquet")
        == "cd6097d3d29f0c4dd4f913a2ac07af45cfcdd029f0b6c5900771a2b698edb925"
    )
    validation = pq.read_table(
        ROOT / "outputs/e012_causal_rope_sequence/pilot_v1/validation.parquet", use_threads=False
    ).to_pylist()
    ids = json.loads((HIST / "panels.json").read_text())
    lookup = {r["sample_id"]: r for r in validation}
    assert not set(ids["primary"]) & set(ids["independent"])
    panels = {name: [lookup[i] for i in vals] for name, vals in ids.items()}
    cursor = Cursor()
    panels["train_reference"] = cursor.take(4096)
    protected = {hashlib.sha256(r["sequence"].encode()).digest() for r in validation}
    # Consumer leakage proof over certified manifest; no rescreening or recertification.
    counts = []
    overlaps = 0
    hashes = inventory()
    for sid in range(512):
        rel = f"corpus/manifest/part-{sid:03d}.parquet"
        p = DATA / rel
        assert sha(p) == hashes[rel]
        pf = pq.ParquetFile(p)
        counts.append(pf.metadata.num_rows)
        for b in pf.iter_batches(columns=["sequence_sha256"], use_threads=False):
            overlaps += sum(h in protected for h in b.column(0).to_pylist())
        if sid % 64 == 0:
            print("manifest leakage audit", sid, "overlap", overlaps, flush=True)
    assert sum(counts) == 20000000 and overlaps == 0
    save("panels.json", panels)
    seed()
    net = ProteinSequenceTransformer(architecture())
    c = torch.load(PARENT, map_location="cpu", weights_only=False)
    net.load_state_dict(c["model"], strict=True)
    assert parameter_count(net) == 15533952
    audit = {
        "parent_path": str(PARENT),
        "parent_sha256": PARENT_SHA,
        "parent_updates": c["successful_updates"],
        "parameters": parameter_count(net),
        "architecture": dataclasses.asdict(net.config),
        "vocabulary": list(VOCAB.tokens),
        "tokenizer_sha256": digest(list(VOCAB.tokens)),
        "strict_weights_load": True,
        "fresh_optimizer": True,
        "parent_cursor_optimizer_scheduler_discarded": True,
        "bos": 1,
        "eos": 2,
        "pad": 0,
        "eos_training_target": False,
        "teacher_forcing": "BOS + residues[:-1] -> residues; no EOS target; requested-length conditioned",
        "causal_mask": "triangular plus key-padding and hidden-state padding masks",
        "position_encoding": "q/k RoPE base10000, dimension64, nonpersistent regenerated sin/cos",
        "corpus": handoff,
        "manifest_rows": counts,
        "historical_validation_exact_overlap": overlaps,
        "leakage_proof": "All 20M certified manifest hashes compared against all 23307 historical validation sequences; zero. Certified exact uniqueness plus bijective cursor ensures no TRAIN/eval identity overlap. Protected 30%/80% homology exclusions retained from certification.",
        "dataset_hashes": {
            "handoff": sha(DATA / "reports/chatgpt_handoff.json"),
            "certification": sha(DATA / "corpus/certification.json"),
            "inventory": sha(DATA / "corpus/SHA256SUMS"),
            "panels": sha(RUN / "panels.json"),
            "final_fasta_certified": hashes["corpus/e012_uniref50_2026_03_20000000.fasta.gz"],
        },
        "historical_overfitting_report": str(
            ROOT / "reports/experiments/E012_causal_rope_sequence/generalization_audit_v1/FINAL_REPORT.md"
        ),
    }
    save("audit.json", audit)
    protocol = {
        "configuration": CONFIG,
        "parent_sha256": PARENT_SHA,
        "model": architecture(),
        "tokenizer": list(VOCAB.tokens),
        "dataset_hashes": audit["dataset_hashes"],
        "sampling": "PCG64(seed+epoch) permutation of 512 shards; PCG64 SeedSequence(seed,epoch,shard,1) permutation of every row. Natural length distribution; no replacement per epoch; one shard in memory; no dataset copy.",
        "evaluation": "Original protected primary and independent 2048 panels. Full 24-way BF16 forward / FP32 log-softmax batch16. Equal-protein CE primary, token CE/perplexity/accuracy and five length strata secondary. Same historical 16 target positions, E012:12012 prefix shuffle and last8/last1 masks; original minus control CE and 10000 paired identity bootstrap seed12112. Full at parent and 1M; likelihood-only at named checkpoints. Lightweight first52 per stratum per panel every1000 updates.",
        "generation": "32 fixed canonical samples:16 length64 +16 length128; seeds32076/32140, temperature1, top_p.95; parent and 1M only. Composition entropy, unique3mers, homopolymers, validity, length, exact/near duplicates. Bounded comparison to first4096 deterministic TRAIN and4096 protected panel references, 3mer shortlist10 and SequenceMatcher heuristic.",
        "interpretation": "Lower TRAIN loss alone is insufficient. Compare matched held-out CE/perplexity, composition baselines and normal-shuffle/last8/last1 sensitivity including strata and confidence intervals. Generation validity/length are constrained. No stability/folding inference. Smoke outputs are operational only.",
        "scratch_control": {
            "train_now": False,
            "initialization_seed": SEED,
            "weights": "fresh original architecture initialization",
            "all_other_settings": "identical frozen corpus/cursor/optimizer/schedule/evaluation; matched exposures and report measured compute",
        },
        "milestone_policy": "Stop automatically at exactly1000000 unique sequences (15625 updates), save immutable checkpoint, full frozen evaluation and generation, mark paused for review. Full20M horizon declared from outset; extension requires review. Rolling alternating2 checkpoints every250 plus fixed named checkpoints; fsync and atomic replacement.",
        "smoke_policy": "Disposable warm-start100 and1000 cumulative updates with exact next-update save/restore replay at100; production restarts parent with fresh optimizer/RNG/cursor. Longest-length16x4 CUDA preflight before100. Resource/numerical failure stops.",
    }
    save("protocol.json", protocol)
    save("source_hashes.json", source_hashes())
    (RUN / "HISTORICAL_REPORT.md").write_text(
        (ROOT / "reports/experiments/E012_causal_rope_sequence/generalization_audit_v1/FINAL_REPORT.md").read_text()
    )
    print("prepared", RUN, flush=True)


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def smoke():
    assert not (RUN / "launch.json").exists(), "preflight cannot mutate a launched experiment"
    # Seal operational implementation before any smoke optimizer update.
    save("source_hashes.json", source_hashes())
    save("resources_preflight.json", guard(ownership=True, host=True))
    net, opt, cursor, counters = model_optimizer()
    torch.cuda.reset_peak_memory_stats()
    net.eval()
    rr = [
        {"sample_id": "long", "sequence": "ACDEFGHIKLMNPQRSTVWY" * 2, "length": 40},
        {"sample_id": "short", "sequence": "ACDEFGHIKLMNPQRSTVWY", "length": 20},
    ]
    b = batch(rr, "cuda")
    assert b["input_ids"][0, 0].item() == VOCAB.bos_id
    assert torch.equal(b["input_ids"][0, 1:], b["target_ids"][0, :-1])
    assert b["attention_mask"].sum().item() == 60 and not b["attention_mask"][1, 20:].any()
    with torch.no_grad():
        baseline = net(b["input_ids"], b["lengths"], b["attention_mask"])
        changed = b["input_ids"].clone()
        changed[0, 25:] = VOCAB.canonical_ids[-1]
        future = net(changed, b["lengths"], b["attention_mask"])
        assert torch.equal(baseline[0, :25], future[0, :25]), "future leakage"
        changed = b["input_ids"].clone()
        changed[1, 20:] = VOCAB.canonical_ids[-1]
        padded = net(changed, b["lengths"], b["attention_mask"])
        assert torch.equal(baseline[1, :20], padded[1, :20]), "padding leakage"
        ce = F.cross_entropy(baseline.flatten(0, 1), b["target_ids"].flatten(), reduction="none").view_as(
            b["target_ids"]
        )
        sequence = ((ce * b["attention_mask"]).sum(1) / b["lengths"]).mean()
        token = (ce * b["attention_mask"]).sum() / b["lengths"].sum()
        assert torch.allclose(
            sequence, sequence_cross_entropy(baseline, b["target_ids"], b["attention_mask"], reduction="sequence_mean")
        )
    sample = Cursor()
    first = sample.take(4096)
    assert len({r["sample_id"] for r in first}) == 4096
    assert len({r["sequence"] for r in first}) == 4096
    next_a = sample.take(64)
    next_b = Cursor({"epoch": 0, "shard_pos": 0, "row_pos": 4096, "exposures": 4096}).take(64)
    assert next_a == next_b and not {r["sample_id"] for r in first} & {r["sample_id"] for r in next_a}
    save(
        "loader_model_tests.json",
        {
            "passed": True,
            "bos_causal_shift": True,
            "padding_invariance": True,
            "future_invariance": True,
            "sequence_loss_aggregation": float(sequence),
            "token_loss_aggregation": float(token),
            "unique_first4096": True,
            "deterministic_cursor_replay": True,
        },
    )
    # Disposable worst-length physical16 x accumulation4; original model unchanged.
    worst = [{"sample_id": f"memory-{i}", "sequence": ("ACDEFGHIKLMNPQRSTVWY" * 25), "length": 500} for i in range(64)]

    class Synthetic:
        exposures = 0

        def take(self, n):
            self.exposures += n
            return worst[:n]

    t = time.monotonic()
    update(net, opt, Synthetic(), counters)
    memory = {
        "worst_length": 500,
        "physical_batch": 16,
        "accumulation": 4,
        "effective_batch": 64,
        "allocated_peak": torch.cuda.max_memory_allocated(),
        "reserved_peak": torch.cuda.max_memory_reserved(),
        "seconds": time.monotonic() - t,
    }
    save("memory_preflight.json", memory)
    guard(host=True)
    del net, opt
    torch.cuda.empty_cache()
    net, opt, cursor, counters = model_optimizer()
    t = time.monotonic()
    for _ in range(100):
        last = update(net, opt, cursor, counters)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t
    checkpoint("smoke_boundary.pt", net, opt, cursor, counters)
    first = update(net, opt, cursor, counters)
    checkpoint("smoke_expected.pt", net, opt, cursor, counters)
    del net, opt
    torch.cuda.empty_cache()
    net, opt, cursor, counters = model_optimizer(RUN / "smoke_boundary.pt")
    replay = update(net, opt, cursor, counters)
    checkpoint("smoke_replay.pt", net, opt, cursor, counters)
    expected = torch.load(RUN / "smoke_expected.pt", map_location="cpu", weights_only=False)
    actual = torch.load(RUN / "smoke_replay.pt", map_location="cpu", weights_only=False)
    assert first == replay and state_equal(expected, actual), "exact checkpoint replay failed"
    del expected, actual
    save(
        "smoke_100.json",
        {
            "passed": True,
            "updates": 100,
            "counters_at_boundary": {"exposures": 6400},
            "seconds": elapsed,
            "sequences_per_second": 6400 / elapsed,
            "exact_replay_all_checkpoint_fields": True,
            "next_update": replay,
            "memory": memory,
        },
    )
    guard(host=True)
    for _ in range(101, 1000):
        last = update(net, opt, cursor, counters)
        if counters["updates"] % 100 == 0:
            telemetry = guard(host=True)
            append("smoke_telemetry.jsonl", last | telemetry)
            print("smoke", last["updates"], last["train_sequence_ce"], flush=True)
    torch.cuda.synchronize()
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if isinstance(v, torch.Tensor))
    checkpoint("smoke_1000.pt", net, opt, cursor, counters)
    elapsed = time.monotonic() - t
    save(
        "smoke_1000.json",
        {
            "passed": True,
            "counters": counters,
            "seconds": elapsed,
            "sequences_per_second": 64000 / elapsed,
            "tokens_per_second": counters["tokens"] / elapsed,
            "last": last,
            "maximum_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scientific_evidence": False,
        },
    )
    save("resources_post_smoke.json", guard(host=True))
    assert sha(PARENT) == PARENT_SHA


def request_stop(signum, frame):
    global STOP
    STOP = True


def report():
    def read(name):
        p = RUN / name
        return json.loads(p.read_text()) if p.exists() else None

    status = read("status.json")
    launch = read("launch.json")
    smoke = read("smoke_1000.json")
    recoveries = [
        p
        for p in RUN.glob("*.pt")
        if p.name.startswith(("production_initial", "recovery_", "milestone_", "shutdown_pause", "resource_pause"))
    ]
    newest = max(recoveries, key=lambda p: read(p.name + ".json")["counters"]["updates"]) if recoveries else None
    metadata = {
        "experiment": str(RUN),
        "classification": "large-corpus warm-start continuation",
        "audit": read("audit.json"),
        "configuration": CONFIG,
        "frozen_protocol_sha256": sha(RUN / "protocol.json"),
        "memory_preflight": read("memory_preflight.json"),
        "loader_tests": read("loader_model_tests.json"),
        "smoke100": read("smoke_100.json"),
        "smoke1000": smoke,
        "launch": launch,
        "status": status,
        "parent_evaluation": read("evaluations/parent.json"),
        "milestone_evaluation": read("evaluations/milestone_1m.json"),
        "milestone_comparison": read("milestone_comparison.json"),
        "parent_generation": read("generation/parent.json"),
        "milestone_generation": read("generation/milestone_1m.json"),
        "storage": "All newly generated experiment artifacts, caches and scratch reside under this D: experiment. Source code remains in WSL. Historical checkpoint/reservoir unchanged.",
        "other_workloads": "E007, E015 production and external designability remain stopped; ownership/resources checked at launch and runtime. No structure or external evaluation launched.",
        "resume_checkpoint": str(newest) if newest else None,
        "resume_command": f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH={ROOT}/src /home/simostocco/miniforge3/envs/proteingen/bin/python -B {ROOT}/scripts/run_e012_large_corpus.py train --resume {newest if newest else '<D:-checkpoint.pt>'}",
        "checkpoint_files": [str(p) for p in sorted(RUN.glob("*.pt"))],
    }
    save("REPORT.json", metadata)
    measured = (status or {}).get("telemetry", {})
    rate = measured.get("sequences_per_second") or (smoke or {}).get("sequences_per_second")
    text = f"""# E012 certified 20M warm-start continuation

Experiment: `{RUN}`. Parent: `{PARENT}`; SHA256 `{PARENT_SHA}`.
Corpus: 20,000,000 certified exact-unique proteins, 4,042,471,015 residues, 512 shards.
Strict load: 15,533,952 parameters; width384, eight blocks, six heads, FF1536, q/k RoPE, unchanged continuous requested-length conditioning and24-token vocabulary. BOS1, PAD0, EOS2 exists but is never a training target. Causal shift predicts canonical residues only.
All20M manifest hashes were compared with all23,307 historical validation sequences: zero exact overlap. The certified protected30% identity/80% coverage exclusions are retained. No corpus stages were rerun and no reservoir was altered.

Fresh AdamW, betas(.9,.95), decay.01, clip1; BF16 forward with FP32 weights/moments. Physical16 ×4 accumulation =64 proteins/update. LR warms linearly over500 updates to.0001, cosine to.00001 at312,500 updates (20M exposures). Smoke100 and1000 are disposable and excluded from production counters. The source checkpoint optimizer, schedule, RNG and cursor are discarded.
Deterministic without-replacement exposure: shuffled512 certified shards, independently shuffled rows per shard/epoch, seed22012. One shard in memory. Unique exposures, optimizer updates and residue tokens are distinct counters. Exactly1M unique exposures =15,625 updates; stop for scientific review after durable checkpoint and full frozen evaluation. Named checkpoints near250k/500k/750k and exactly1M; alternating two rolling recovery checkpoints every250 updates.

Operational status: `{(status or {}).get("state", "prepared/preflight")}`. PID: `{(launch or {}).get("pid", "not launched")}`.
Counters: `{(status or {}).get("counters", {})}`.
Measured sequence rate: `{rate}` proteins/sec. Rate scope: `{"production including periodic work" if measured else "disposable smoke; provisional"}`.
Projected1M duration: `{1000000 / rate if rate else None}` seconds; full20M epoch: `{20000000 / rate if rate else None}` seconds. These are measured-rate projections, not a commitment to run20M.

Historical reference at10k: TRAIN equal-protein perplexity3.902; primary18.067 and independent18.258. Broad expanding overfitting gap. Historical normal-minus-shuffle CE: primary−.06483, independent−.05154 nats; moderate held-out order dependence. Fresh frozen parent metrics and sample details are in evaluations/parent.json and generation/parent.json once complete.
Protected primary and independent panels remain2048 each. Full24-way likelihood, equal-protein and token-weighted CE/perplexity, token accuracy, five length strata; original16 positions, composition-preserving prefix shuffle, last8/last1 controls and paired identity bootstrap10,000/seed12112. Same evaluations for parent,1M and future scratch control. Fixed260-identity lightweight panels each1000 updates, full likelihood at named intermediate boundaries.
Generation at parent and1M:32 seeded samples, lengths64/128, temperature1, top_p.95. Canonical validity/length are constrained. Composition, entropy, repetitions and diversity recorded. Bounded memorization search uses4096 scheduled TRAIN and4096 protected references,3mer shortlist10 then SequenceMatcher ratio; this is heuristic and cannot exclude unsampled or remote homologous memorization.

Scientific conclusion: {"1M evaluation is available in milestone_comparison.json; assess paired likelihood and order controls before extension." if read("milestone_comparison.json") else "Pending the1M milestone. Smoke loss and early TRAIN improvement do not establish the data-scaling hypothesis."} No claim of protein stability or correct folding follows from perplexity.
Raw weights, optimizer, schedule, RNG, identity cursor, exposure/token counters, configuration, dataset/tokenizer hashes, source hashes and parent SHA256 are stored in every checkpoint. Resume from the newest hash-verified recovery or named checkpoint using the command in REPORT.json; interrupted updates after the last durable save replay from that cursor.
E007/E015 production and external designability remain stopped. No structural evaluation launched. All new artifacts are on D:; historical sources remain unchanged.
"""
    (RUN / "REPORT.md").write_text(text)


def verify_replay():
    """Recheck final operational source at the 1000-smoke boundary before launch."""
    assert not (RUN / "launch.json").exists()
    assert json.loads((RUN / "smoke_1000.json").read_text())["passed"]
    save("source_hashes.json", source_hashes())
    guard(ownership=True, host=True)
    net, opt, cursor, counters = model_optimizer(RUN / "smoke_1000.pt", allow_runner_change=True)
    first = update(net, opt, cursor, counters)
    checkpoint("final_replay_expected.pt", net, opt, cursor, counters)
    del net, opt
    torch.cuda.empty_cache()
    net, opt, cursor, counters = model_optimizer(RUN / "smoke_1000.pt", allow_runner_change=True)
    second = update(net, opt, cursor, counters)
    checkpoint("final_replay_actual.pt", net, opt, cursor, counters)
    a = torch.load(RUN / "final_replay_expected.pt", map_location="cpu", weights_only=False)
    b = torch.load(RUN / "final_replay_actual.pt", map_location="cpu", weights_only=False)
    assert first == second and state_equal(a, b)
    save(
        "final_replay.json",
        {"passed": True, "all_checkpoint_fields_equal": True, "source_hashes": source_hashes(), "next_update": second},
    )
    test_rows = [next(r for r in eval_panels()["primary"] if stratum(r["length"]) == s) for s in range(5)]
    tested, summary = context(net, test_rows)
    assert all(
        math.isfinite(r[k]) and r[k] >= 0 for r in tested for k in ["prefix_normal", "shuffle", "last8", "last1"]
    )
    save("evaluation_preflight.json", {"passed": True, "proteins": len(tested), "summary": summary})
    report()


def launch():
    assert not (RUN / "launch.json").exists(), "launch already recorded; use explicit checkpoint resume"
    assert json.loads((RUN / "final_replay.json").read_text())["passed"]
    assert json.loads((RUN / "evaluation_preflight.json").read_text())["passed"]
    guard(ownership=True, host=True)
    command = [__import__("sys").executable, "-B", str(Path(__file__).resolve()), "train"]
    with (RUN / "training.log").open("a") as log:
        child = subprocess.Popen(
            command, cwd=RUN, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    save(
        "launch_command.json", {"pid": child.pid, "command": command, "cwd": str(RUN), "log": str(RUN / "training.log")}
    )
    print("launched production PID", child.pid, flush=True)


def recover_evaluation_launch():
    guard(ownership=True, host=True)
    initial = RUN / "production_initial.pt"
    assert sha(initial) == json.loads((RUN / "production_initial.pt.json").read_text())["sha256"]
    failed = torch.load(initial, map_location="cpu", weights_only=False)
    assert failed["counters"] == {"updates": 0, "exposures": 0, "tokens": 0}
    assert failed["parent_sha256"] == PARENT_SHA
    for old, new in [
        ("production_initial.pt", "archived_failed_evaluation_initial.pt"),
        ("production_initial.pt.json", "archived_failed_evaluation_initial.pt.json"),
        ("launch.json", "archived_failed_evaluation_launch.json"),
        ("launch_command.json", "archived_failed_evaluation_launch_command.json"),
    ]:
        assert not (RUN / new).exists()
        os.rename(RUN / old, RUN / new)
    save(
        "evaluation_failure_recovery.json",
        {
            "before_production_update": True,
            "counters": failed["counters"],
            "reason": "Unary negation incorrectly applied after converting order-control tensor to list; evaluator fixed. Initial state and logs preserved; relaunch uses unchanged parent/fresh optimizer.",
            "archived_checkpoint_sha256": sha(RUN / "archived_failed_evaluation_initial.pt"),
        },
    )
    del failed
    verify_replay()
    launch()


def train(resume=None):
    if not resume:
        assert not (RUN / "production_initial.pt").exists(), (
            "Existing production run requires explicit checkpoint resume"
        )
    for name in ["smoke_100.json", "smoke_1000.json", "final_replay.json"]:
        assert json.loads((RUN / name).read_text())["passed"]
    assert source_hashes() == json.loads((RUN / "source_hashes.json").read_text()), "source drift"
    save("resources_launch.json", guard(ownership=True, host=True))
    assert sha(PARENT) == PARENT_SHA
    if resume:
        assert sha(resume) == json.loads(Path(str(resume) + ".json").read_text())["sha256"]
    net, opt, cursor, counters = model_optimizer(resume)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    save(
        "launch.json",
        {
            "pid": os.getpid(),
            "time": time.time(),
            "command": __import__("sys").argv,
            "resume": str(resume) if resume else None,
            "config": CONFIG,
        },
    )
    checkpoint("production_initial.pt", net, opt, cursor, counters) if not resume else None
    save("status.json", {"state": "frozen_parent_evaluation", "pid": os.getpid(), "counters": counters})
    report()
    if not (RUN / "evaluations/parent.json").exists():
        evaluate(net, "parent", full=True)
    if not (RUN / "generation/parent.json").exists():
        assert counters["updates"] == 0, "parent generation cannot use continued weights"
        generation(net, "parent")
    begin = time.monotonic()
    initial = counters.copy()
    while counters["exposures"] < CONFIG["milestone_unique_exposures"]:
        count = min(64, CONFIG["milestone_unique_exposures"] - counters["exposures"])
        last = update(net, opt, cursor, counters, count)
        elapsed = time.monotonic() - begin
        if counters["updates"] % 25 == 0:
            rate = (counters["exposures"] - initial["exposures"]) / elapsed
            last |= {
                "time": time.time(),
                "sequences_per_second": rate,
                "tokens_per_second": (counters["tokens"] - initial["tokens"]) / elapsed,
                "eta_1m_seconds": (1000000 - counters["exposures"]) / rate,
                "full_20m_epoch_estimate_seconds": 20000000 / rate,
                "cuda_allocated": torch.cuda.memory_allocated(),
                "cuda_reserved": torch.cuda.memory_reserved(),
            }
            try:
                last["resources"] = guard(host=counters["updates"] % 250 == 0)
            except Exception as exc:
                checkpoint("resource_pause.pt", net, opt, cursor, counters)
                save("status.json", {"state": "paused_resource_gate", "reason": str(exc), "counters": counters})
                return
            append("telemetry.jsonl", last)
            save("status.json", {"state": "training", "pid": os.getpid(), "counters": counters, "telemetry": last})
            print("train", counters, "ce", last["train_sequence_ce"], "seq/s", rate, flush=True)
        if counters["updates"] % 250 == 0:
            checkpoint(f"recovery_{(counters['updates'] // 250) % 2}.pt", net, opt, cursor, counters)
            report()
        for exposure in CONFIG["named_exposures"]:
            if counters["exposures"] >= exposure and counters["exposures"] - count < exposure:
                checkpoint(f"milestone_{exposure:07d}.pt", net, opt, cursor, counters)
                if exposure < 1000000:
                    evaluate(net, f"exposures_{counters['exposures']:07d}", full_likelihood=True)
        if STOP:
            checkpoint("shutdown_pause.pt", net, opt, cursor, counters)
            save("status.json", {"state": "paused_signal", "counters": counters})
            return
        if counters["updates"] % 1000 == 0 and counters["exposures"] < 1000000:
            evaluate(net, f"light_{counters['updates']:06d}")
    save("status.json", {"state": "milestone_checkpoint_saved_evaluation_pending", "counters": counters})
    final = evaluate(net, "milestone_1m", full=True)
    generation(net, "milestone_1m")
    parent = json.loads((RUN / "evaluations/parent.json").read_text())
    comparison = {}
    for name in ["primary", "independent"]:
        old = json.loads((RUN / f"evaluations/parent_{name}_likelihood.json").read_text())
        new = json.loads((RUN / f"evaluations/milestone_1m_{name}_likelihood.json").read_text())
        assert [r["sample_id"] for r in old] == [r["sample_id"] for r in new]
        comparison[name] = {
            "paired_ce_change": paired_interval([b["normal"] - a["normal"] for a, b in zip(old, new, strict=True)]),
            "parent": parent[name],
            "milestone": final[name],
        }
        old_context = json.loads((RUN / f"evaluations/parent_{name}_context.json").read_text())
        new_context = json.loads((RUN / f"evaluations/milestone_1m_{name}_context.json").read_text())
        assert [r["sample_id"] for r in old_context] == [r["sample_id"] for r in new_context]
        comparison[name]["paired_order_sensitivity_change"] = {
            control: paired_interval(
                [
                    (b["prefix_normal"] - b[control]) - (a["prefix_normal"] - a[control])
                    for a, b in zip(old_context, new_context, strict=True)
                ]
            )
            for control in ["shuffle", "last8", "last1"]
        }
    save("milestone_comparison.json", comparison)
    save(
        "status.json",
        {
            "state": "paused_1m_pending_scientific_review",
            "counters": counters,
            "checkpoint": str(RUN / "milestone_1000000.pt"),
            "comparison": comparison,
        },
    )
    assert sha(PARENT) == PARENT_SHA
    report()


def main():
    os.nice(10)
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:2]))
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage", choices=["prepare", "smoke", "train", "report", "verify-replay", "launch", "recover-evaluation-launch"]
    )
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "smoke":
        smoke()
    elif args.stage == "train":
        train(args.resume)
    elif args.stage == "verify-replay":
        verify_replay()
    elif args.stage == "launch":
        launch()
    elif args.stage == "recover-evaluation-launch":
        recover_evaluation_launch()
    else:
        report()


if __name__ == "__main__":
    main()
