"""Fixed-budget E012 execution, input audit and preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from torch.nn import functional as F

from protein_sequence_generation.e012 import (
    VOCAB,
    WINDOWS,
    baseline_ce,
    batch,
    build_baselines,
    classification,
    positions,
    prefix_batch,
    stratum,
    summarize,
)
from protein_sequence_generation.metrics import sequence_cross_entropy
from protein_sequence_generation.model import ProteinSequenceTransformer, parameter_count
from protein_sequence_generation.sampling import generate_sequences
from protein_sequence_generation.training import _lr_lambda

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/pilot_v1"
OUT = ROOT / "outputs/e012_causal_rope_sequence/pilot_v1"
E011 = ROOT.parent / "proteinGen-sequence-context"
BASE = "30b657be81cd7bd943918980812d68ecaa0778de"
CONFIG = ROOT / "configs/e012_causal_rope_sequence_v1.yaml"
BOUNDARIES = [0, 250, 500, 1000, 1500, 2000]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as h:
        for chunk in iter(lambda: h.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, data, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as h:
        json.dump(data, h, indent=2, sort_keys=True)
        h.write("\n")


def seed():
    torch.set_num_threads(2)
    random.seed(12012)
    np.random.seed(12012)
    torch.manual_seed(12012)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12012)
    torch.use_deterministic_algorithms(True)


def model():
    return ProteinSequenceTransformer(yaml.safe_load(CONFIG.read_text())["model"])


def weights_hash(net):
    h = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    assert not (REPORT / "contract.json").exists()
    audit_path = ROOT / "reports/experiments/E011_sequence_context_only/data_audit.json"
    audit = json.loads(audit_path.read_text())
    hist_pilot = ROOT / "reports/experiments/E011_sequence_context_only/pilot_v1"
    integrity = json.loads((hist_pilot / "dataset_integrity.json").read_text())
    for name, digest in json.loads((hist_pilot / "result_manifest.json").read_text())[
        "published_artifact_sha256"
    ].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in audit["input_hashes"].items():
        assert sha(name) == digest
    data_root = Path(audit["directory"])
    protocol = json.loads((data_root / "protocol.json").read_text())
    inventory = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (data_root / "shard_hashes.sha256").read_text().splitlines()
    }
    for j, row in enumerate(protocol["shards"]):
        path = data_root / row["path"]
        assert sha(path) == row["sha256"] == inventory[row["path"]]
        assert pq.ParquetFile(path).metadata.num_rows == row["row_count"]
        if (j + 1) % 64 == 0:
            print(f"E012 raw hash audit {j + 1}/{len(protocol['shards'])}", flush=True)
    all_rows = {}
    for split in ["train", "validation"]:
        prior = integrity["sequence_caches"][split]
        path = E011 / prior["path"]
        assert sha(path) == prior["sha256"]
        projected = pq.read_table(path, columns=["sample_id", "split", "sequence", "token_ids"]).to_pylist()
        assert len(projected) == {"train": 231743, "validation": 23307}[split]
        rows = []
        for row in projected:
            assert row["split"] == split
            assert VOCAB.encode(row["sequence"]) == [int(v) + 2 for v in row["token_ids"]]
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "sequence": row["sequence"],
                    "length": len(row["sequence"]),
                }
            )
        derived = OUT / f"{split}.parquet"
        assert not derived.exists()
        pq.write_table(pa.Table.from_pylist(rows), derived)
        all_rows[split] = rows
    ids = {s: {r["sample_id"] for r in rows} for s, rows in all_rows.items()}
    seqs = {s: {r["sequence"] for r in rows} for s, rows in all_rows.items()}
    assert not ids["train"] & ids["validation"] and not seqs["train"] & seqs["validation"]
    memberships = {}
    for split in ids:
        source = next(p for p in audit["input_hashes"] if p.endswith(f"/{split}.parquet"))
        memberships[split] = [
            r
            for r in pq.read_table(source, columns=["sample_id", "cluster_id", "pdb_id", "split_group_id"]).to_pylist()
            if r["sample_id"] in ids[split]
        ]
        assert len(memberships[split]) == len(all_rows[split])
    for key in ["cluster_id", "pdb_id", "split_group_id"]:
        assert not {r[key] for r in memberships["train"]} & {r[key] for r in memberships["validation"]}
    primary = json.loads((hist_pilot / "primary_panel.json").read_text())
    independent_path = ROOT / "reports/experiments/E011_sequence_context_only/diagnostic_panel.json"
    independent = json.loads(independent_path.read_text())
    panels = {"primary": primary["sample_ids"], "independent": independent["sample_ids"]}
    assert all(len(v) == len(set(v)) == 2048 and set(v) <= ids["validation"] for v in panels.values())
    assert not set(panels["primary"]) & set(panels["independent"])
    save(REPORT / "panels.json", panels, exclusive=True)
    lookup = {r["sample_id"]: r for r in all_rows["validation"]}
    selected = {
        name: {sid: positions(len(lookup[sid]["sequence"])) for sid in values} for name, values in panels.items()
    }
    save(REPORT / "positions.json", selected, exclusive=True)
    print("Building fixed TRAIN-only uni/bi/trigram baselines", flush=True)
    save(REPORT / "baselines.json", build_baselines(all_rows["train"]), exclusive=True)
    seed()
    net = model()
    initial = weights_hash(net)
    seed()
    assert initial == weights_hash(model())
    save(
        REPORT / "data_integrity.json",
        {
            "status": "passed",
            "raw_shards_verified": len(protocol["shards"]),
            "train_count": len(all_rows["train"]),
            "validation_count": len(all_rows["validation"]),
            "cross_split_exact_sequence_overlap": 0,
            "cross_split_sample_overlap": 0,
            "protected_cluster_pdb_split_group_overlap": 0,
            "homology_policy": {"identity": 0.30, "coverage": 0.80},
            "derived_columns": ["sample_id", "split", "sequence", "length"],
            "geometry_materialized": False,
            "primary_panel_sha256": sha(hist_pilot / "primary_panel.json"),
            "independent_panel_sha256": sha(independent_path),
            "derived_manifest_sha256": {s: sha(OUT / f"{s}.parquet") for s in all_rows},
            "source_cache_sha256": {s: integrity["sequence_caches"][s]["sha256"] for s in all_rows},
            "parameter_count": parameter_count(net),
            "initialization_seed": 12012,
            "initial_weights_sha256": initial,
        },
        exclusive=True,
    )
    print("E012 data integrity and fresh initialization passed", flush=True)


def autocast():
    precision = json.loads((REPORT / "cuda_smoke.json").read_text())["selected_precision"]
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def optimizer(net):
    opt = torch.optim.AdamW(net.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup_steps=100, total_steps=2000))
    return opt, sched


def check_known_research_owners():
    """Additional host guard; operator must also inspect driver GPU utilization."""
    processes = subprocess.check_output(["ps", "-C", "python", "-C", "python3", "-o", "pid=,args="], text=True)
    for line in processes.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        command = fields[1]
        known_runner = any(
            name in command
            for name in ["run_e010_", "run_e011_", "run_e012_", "train_sequence_transformer.py", "train_diffusion.py"]
        )
        if known_runner and (
            " train" in command or "train_sequence_transformer.py" in command or "train_diffusion.py" in command
        ):
            raise RuntimeError(f"E012 CUDA execution deferred: another research trainer is active (PID {fields[0]})")


def smoke():
    check_known_research_owners()
    assert torch.cuda.is_available()
    seed()
    net = model().cuda().train()
    torch.cuda.reset_peak_memory_stats()
    opt, sched = optimizer(net)
    rows = [{"sample_id": str(i), "sequence": "ACDEFGHIKLMNPQRSTVWY" * 25} for i in range(8)]
    data = batch(rows, "cuda")
    bf16 = torch.cuda.is_bf16_supported()

    def context():
        return torch.amp.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()

    def update():
        opt.zero_grad(set_to_none=True)
        with context():
            logits = net(**{k: data[k] for k in ["input_ids", "lengths", "attention_mask"]})
            loss = sequence_cross_entropy(logits, data["target_ids"], data["attention_mask"])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        sched.step()
        return float(loss.detach())

    before = net.output.weight.detach().clone()
    loss = update()
    assert not torch.equal(before, net.output.weight)
    save_path = OUT / "smoke_resume.pt"
    torch.save(
        {"model": net.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "rng": rng_state()},
        save_path,
    )
    next_loss = update()
    expected = {k: v.detach().clone() for k, v in net.state_dict().items()}
    restored = torch.load(save_path, map_location="cuda", weights_only=False)
    net.load_state_dict(restored["model"])
    opt.load_state_dict(restored["optimizer"])
    sched.load_state_dict(restored["scheduler"])
    # CPU RNG tensors must remain on CPU when restoring a GPU checkpoint.
    restored["rng"]["cpu"] = restored["rng"]["cpu"].cpu()
    restored["rng"]["cuda"] = [v.cpu() for v in restored["rng"]["cuda"]]
    restore_rng(restored["rng"])
    assert update() == next_loss
    assert all(torch.equal(v, expected[k]) for k, v in net.state_dict().items())
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    generated = generate_sequences(
        net, vocabulary=VOCAB, length=20, num_sequences=2, seed=12013, device="cuda", top_p=0.95
    )
    assert all(len(s) == 20 and set(s) <= set("ACDEFGHIKLMNPQRSTVWY") for s in generated)
    save(
        REPORT / "cuda_smoke.json",
        {
            "status": "passed",
            "gpu": torch.cuda.get_device_name(),
            "bf16_supported": bf16,
            "selected_precision": "bfloat16" if bf16 else "float32",
            "scaler": None,
            "longest_length": 500,
            "physical_batch": 8,
            "gradient_accumulation": 8,
            "effective_proteins": 64,
            "loss": loss,
            "finite_gradients": True,
            "optimizer_mutation": True,
            "checkpoint_exact_resume": True,
            "checkpoint_sha256": sha(save_path),
            "peak_allocated_mib": allocated / 1024**2,
            "peak_reserved_mib": reserved / 1024**2,
            "exact_length_canonical_sampling": {"passed": True, "requested_length": 20, "sequences": 2},
            "torch_version": torch.__version__,
        },
        exclusive=True,
    )
    print("CUDA longest-length, mixed precision and exact resume smoke passed", flush=True)


def load_rows(split):
    path = OUT / f"{split}.parquet"
    columns = ["sample_id", "split", "sequence", "length"]
    assert pq.ParquetFile(path).schema_arrow.names == columns
    return pq.read_table(path, columns=columns).to_pylist()


@torch.no_grad()
def likelihood(net, rows, baselines, *, neutral=False):
    records = []
    net.eval()
    handle = (
        net.length_embedding.register_forward_hook(lambda _m, _i, value: torch.zeros_like(value)) if neutral else None
    )
    try:
        for start in range(0, len(rows), 16):
            chosen = rows[start : start + 16]
            data = batch(chosen, "cuda")
            with autocast():
                logits = net(**{k: data[k] for k in ["input_ids", "lengths", "attention_mask"]})
            logp = logits.float().log_softmax(-1)
            losses = -logp.gather(-1, data["target_ids"][..., None]).squeeze(-1)
            top = logits.topk(3, dim=-1).indices
            for j, row in enumerate(chosen):
                length = row["length"]
                target = data["target_ids"][j, :length]
                vals = losses[j, :length].cpu().numpy()
                record = {
                    "sample_id": row["sample_id"],
                    "length": length,
                    "stratum": stratum(length),
                    "normal": float(vals.mean()),
                    "loss_sum": float(vals.sum()),
                    "top1": float((top[j, :length, 0] == target).float().mean()),
                    "top3": float((top[j, :length] == target[:, None]).any(-1).float().mean()),
                }
                if not neutral:
                    record.update(baseline_ce(row["sequence"], baselines))
                    cuts = np.minimum((np.arange(length) * 10 // length), 9)
                    record["relative_position_ce"] = [float(vals[cuts == k].mean()) for k in range(10)]
                records.append(record)
    finally:
        if handle is not None:
            handle.remove()
    return records


@torch.no_grad()
def diagnostics(net, rows, frozen_positions):
    """Seven window conditions plus a composition-preserving shuffle at final checkpoint."""
    net.eval()
    cases = [(r, i) for r in rows for i in frozen_positions[r["sample_id"]]]
    cases.sort(key=lambda c: c[1])
    per_sample = {r["sample_id"]: [] for r in rows}
    for start in range(0, len(cases), 16):
        chosen = cases[start : start + 16]
        logps = {}
        targets = None
        for condition in ["prefix_normal", "shuffle", *[f"last{w}" for w in WINDOWS]]:
            window = int(condition[4:]) if condition.startswith("last") else None
            inputs, lengths, mask, indices, targets = prefix_batch(
                chosen, "cuda", shuffled=condition == "shuffle", window=window
            )
            with autocast():
                out = net(inputs, lengths, mask)
            logps[condition] = out[torch.arange(len(chosen), device=out.device), indices].float().log_softmax(-1)
        normal, shuffled = logps["prefix_normal"], logps["shuffle"]
        p, q = normal.exp(), shuffled.exp()
        logm = ((p + q) * 0.5).log()
        kl = (p * (normal - shuffled)).sum(-1)
        js = 0.5 * ((p * (normal - logm)).sum(-1) + (q * (shuffled - logm)).sum(-1))
        change1 = normal.argmax(-1) != shuffled.argmax(-1)
        change3 = (normal.topk(3, -1).indices.sort(-1).values != shuffled.topk(3, -1).indices.sort(-1).values).any(-1)
        ces = {k: (-v.gather(-1, targets[:, None]).squeeze(-1)).cpu().tolist() for k, v in logps.items()}
        for j, (row, _) in enumerate(chosen):
            per_sample[row["sample_id"]].append(
                {
                    **{k: v[j] for k, v in ces.items()},
                    "kl": float(kl[j]),
                    "js": float(js[j]),
                    "top1_change": float(change1[j]),
                    "top3_change": float(change3[j]),
                }
            )
        if start % 8192 == 0:
            print(f"Prefix diagnostic {start}/{len(cases)}", flush=True)
    return {sid: {k: float(np.mean([x[k] for x in values])) for k in values[0]} for sid, values in per_sample.items()}


def checkpoint(net, opt, sched, step, order, cursor, proteins, residues):
    path = OUT / f"checkpoint-{step:05d}.pt"
    assert not path.exists()
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": None,
            "rng": rng_state(),
            "data_order": order,
            "data_cursor": cursor,
            "successful_updates": step,
            "processed_proteins": proteins,
            "processed_residues": residues,
            "contract_sha256": sha(REPORT / "contract.json"),
        },
        path,
    )
    check = torch.load(path, map_location="cpu", weights_only=False)
    assert check["successful_updates"] == step
    assert all(torch.equal(v.cpu(), check["model"][k]) for k, v in net.state_dict().items())
    return {"filename": path.name, "sha256": sha(path), "updates": step}


def train():
    check_known_research_owners()
    contract_path = REPORT / "contract.json"
    contract = json.loads(contract_path.read_text())
    assert contract["updates"] == 2000 and contract["boundaries"] == BOUNDARIES
    assert contract["quality_passed"] and contract["cuda_smoke_passed"]
    for name, digest in contract["protected_hashes"].items():
        assert sha(ROOT / name) == digest, name
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip() == ""
    assert (
        subprocess.check_output(["git", "rev-parse", "origin/e012-causal-rope-sequence"], cwd=ROOT, text=True).strip()
        == head
    )
    assert not (OUT / "telemetry.jsonl").exists()
    seed()
    net = model()
    assert weights_hash(net) == contract["initial_weights_sha256"]
    net.cuda()
    opt, sched = optimizer(net)
    training, validation = load_rows("train"), load_rows("validation")
    lookup = {r["sample_id"]: r for r in validation}
    panels = json.loads((REPORT / "panels.json").read_text())
    panels = {
        k: sorted([lookup[s] for s in ids], key=lambda r: (r["length"], r["sample_id"])) for k, ids in panels.items()
    }
    frozen = json.loads((REPORT / "positions.json").read_text())
    baseline = json.loads((REPORT / "baselines.json").read_text())
    order = np.random.default_rng(12012).permutation(len(training))
    cursor, proteins, residues = 0, 0, 0
    checkpoints, trajectory = [], {}
    start_time = time.monotonic()

    def boundary(step):
        state = rng_state()
        checkpoints.append(checkpoint(net, opt, sched, step, order, cursor, proteins, residues))
        entry = {}
        for pop, rows in panels.items():
            records = likelihood(net, rows, baseline)
            save(OUT / f"{pop}_likelihood_{step:05d}.json", records, exclusive=True)
            entry[pop] = {
                "normal_ce": float(np.mean([r["normal"] for r in records])),
                "token_weighted_ce": sum(r["loss_sum"] for r in records) / sum(r["length"] for r in records),
                "top1": float(np.mean([r["top1"] for r in records])),
                "top3": float(np.mean([r["top3"] for r in records])),
            }
        trajectory[str(step)] = entry
        save(REPORT / "trajectory.json", trajectory)
        restore_rng(state)
        net.train()
        print(f"Boundary {step}: {entry}", flush=True)

    boundary(0)
    with (OUT / "telemetry.jsonl").open("x") as log:
        for step in range(1, 2001):
            t0 = time.monotonic()
            net.train()
            opt.zero_grad(set_to_none=True)
            counters = {"sequence_ce": 0.0, "loss_sum": 0.0, "correct": 0, "residues": 0}
            lengths, sample_ids = [], []
            for _ in range(8):
                indices = order[cursor : cursor + 8]
                assert len(indices) == 8
                cursor += 8
                rows = [training[int(i)] for i in indices]
                data = batch(rows, "cuda")
                with autocast():
                    logits = net(**{k: data[k] for k in ["input_ids", "lengths", "attention_mask"]})
                    loss = sequence_cross_entropy(
                        logits, data["target_ids"], data["attention_mask"], reduction="sequence_mean"
                    )
                assert torch.isfinite(loss), "non-finite loss"
                (loss / 8).backward()
                losses = F.cross_entropy(
                    logits.float().reshape(-1, 24), data["target_ids"].reshape(-1), reduction="none"
                ).reshape_as(data["target_ids"])
                counters["sequence_ce"] += float(loss.detach()) / 8
                counters["loss_sum"] += float((losses * data["attention_mask"]).sum().detach())
                counters["correct"] += int(((logits.argmax(-1) == data["target_ids"]) & data["attention_mask"]).sum())
                counters["residues"] += int(data["attention_mask"].sum())
                lengths.extend(data["lengths"].tolist())
                sample_ids.extend(r["sample_id"] for r in rows)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            lr = opt.param_groups[0]["lr"]
            opt.step()
            assert all(torch.isfinite(p).all() for p in net.parameters())
            assert all(torch.isfinite(v).all() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
            sched.step()
            proteins += 64
            residues += counters["residues"]
            token_ce = counters["loss_sum"] / counters["residues"]
            record = {
                "update": step,
                "sequence_mean_ce": counters["sequence_ce"],
                "token_weighted_ce": token_ce,
                "perplexity": math.exp(token_ce),
                "top1_accuracy": counters["correct"] / counters["residues"],
                "gradient_norm": float(norm),
                "clip_coefficient": min(1.0, 1.0 / (float(norm) + 1e-6)),
                "clipped": bool(norm > 1.0),
                "learning_rate": lr,
                "processed_proteins": proteins,
                "processed_residues": residues,
                "length_stratum_exposures": [sum(stratum(n) == i for n in lengths) for i in range(5)],
                "protein_lengths": lengths,
                "sample_ids": sample_ids,
                "amp_overflows": 0,
                "scaler": None,
                "residues_per_second": counters["residues"] / (time.monotonic() - t0),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 50 == 0:
                print(
                    f"E012 update {step}/2000 CE={counters['sequence_ce']:.5f} "
                    f"elapsed={time.monotonic() - start_time:.1f}s",
                    flush=True,
                )
            if step in BOUNDARIES:
                boundary(step)
    final = {}
    for pop, rows in panels.items():
        print(f"Final {pop} order/windows/length diagnostic", flush=True)
        prefix = diagnostics(net, rows, frozen[pop])
        neutral = {r["sample_id"]: r["normal"] for r in likelihood(net, rows, baseline, neutral=True)}
        records = json.loads((OUT / f"{pop}_likelihood_02000.json").read_text())
        for row in records:
            row.update(prefix[row["sample_id"]])
            row["neutral_length"] = neutral[row["sample_id"]]
        save(OUT / f"{pop}_final_records.json", records, exclusive=True)
        final[pop] = summarize(records)
    outcome = classification(final["primary"], final["independent"])
    save(
        REPORT / "results.json",
        {
            "classification": outcome,
            "updates_completed": 2000,
            "preparation_commit": head,
            "contract_sha256": sha(contract_path),
            "trajectory": trajectory,
            "final": final,
            "checkpoints": checkpoints,
            "processed_proteins": proteins,
            "processed_residues": residues,
            "elapsed_seconds": time.monotonic() - start_time,
            "biological_generation_launched": False,
        },
        exclusive=True,
    )
    print(f"Completed E012: {outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "smoke", "train"])
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "train": train}[args.mode]()


if __name__ == "__main__":
    main()
