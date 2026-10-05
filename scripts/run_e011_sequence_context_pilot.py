"""Fixed 2,000-update E011 pilot; immutable science imported from sealed source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pyarrow.parquet as pq
import torch

from protein_sequence_generation.context import (
    COLUMNS,
    SequenceContextTransformer,
    bucket,
    classify_s1,
    collate,
    conditions,
    deterministic_mask,
    gate_panel,
    objective,
    paired_forwards,
    protein_ce,
    stable_seed,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/e011_sequence_context_only/pilot_v1"
REPORT = ROOT / "reports/experiments/E011_sequence_context_only/pilot_v1"
PREP = REPORT.parent
BOUNDARIES = (0, 250, 500, 1000, 1500, 2000)
FRACTIONS = (0.15, 0.30, 0.50)
REGIMES = ((128, 32), (256, 16), (384, 8), (500, 8))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    assert not path.exists(), f"immutable artifact exists: {path}"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def inputs():
    for name, digest in json.loads((PREP / "contract_manifest.json").read_text())["sha256"].items():
        assert sha(ROOT / name) == digest, name
    integrity = json.loads((REPORT / "dataset_integrity.json").read_text())
    assert integrity["status"] == "passed" and integrity["raw_shard_count"] == 744
    loaded = {}
    for split, entry in integrity["sequence_caches"].items():
        path = ROOT / entry["path"]
        assert sha(path) == entry["sha256"]
        assert pq.ParquetFile(path).schema_arrow.names == list(COLUMNS)
        loaded[split] = pq.read_table(path, columns=list(COLUMNS)).to_pylist()
    model = SequenceContextTransformer()
    assert sum(p.numel() for p in model.parameters()) == 6457364
    assert not any("geometry" in n or "fusion" in n or "length_embedding" in n for n, _ in model.named_modules())
    return loaded


def training_conditions(rows, fraction, epoch):
    targets, valid = collate(rows)
    ids = [r["sample_id"] for r in rows]
    selected = deterministic_mask(targets, valid, ids, fraction, 6011, epoch)
    # Call the sealed construction with synthetic distinct same-stratum donors;
    # only normal/shuffle are used for training, and donors are not model input.
    donors = []
    for row in rows:
        donor = ((np.asarray(row["token_ids"]) - 2 + 1) % 20 + 2).tolist()
        donors.append({"sample_id": "unused-training-donor:" + row["sample_id"], "token_ids": donor})
    panel = conditions(targets, valid, selected, ids, donors, 6011)
    assert torch.equal(selected, deterministic_mask(targets, valid, ids, fraction, 6011, epoch))
    assert all((t[selected] == 1).all() for t in panel.values())
    return targets, valid, selected, panel["normal"], panel["visible_shuffle"]


def batches(rows, epoch):
    order = np.random.default_rng(6011 + epoch).permutation(len(rows))
    queues = [[] for _ in REGIMES]
    for index in order:
        row = rows[int(index)]
        regime = next(i for i, (maximum, _) in enumerate(REGIMES) if len(row["token_ids"]) <= maximum)
        queues[regime].append(int(index))
        if len(queues[regime]) == REGIMES[regime][1]:
            yield queues[regime]
            queues[regime] = []
    for q in queues:
        if q:
            yield q


def cuda_smoke(data):
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.manual_seed(6011)
    torch.cuda.manual_seed_all(6011)
    model = SequenceContextTransformer().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    evidence = []
    for maximum, count in REGIMES:
        source = max((r for r in data["train"] if len(r["token_ids"]) <= maximum), key=lambda r: len(r["token_ids"]))
        rows = [{**source, "sample_id": f"smoke-{maximum}-{i}"} for i in range(count)]
        t, v, m, n, s = (x.cuda() for x in training_conditions(rows, 0.30, 0))
        torch.cuda.reset_peak_memory_stats()
        before = model.sequence_output.weight.detach().clone()
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        with torch.no_grad():
            logits = model(n, v)
        after_cpu, after_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda)
        left, right = paired_forwards(model, n, n, v)
        assert torch.equal(left, right) and torch.equal(left, logits)
        assert torch.equal(after_cpu, torch.get_rng_state()) and torch.equal(after_cuda, torch.cuda.get_rng_state())
        left, right = paired_forwards(model, n, s, v)
        loss = objective(left, right, t, m, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        assert not torch.equal(before, model.sequence_output.weight)
        assert all(
            torch.isfinite(value).all()
            for state in opt.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        torch.cuda.synchronize()
        evidence.append(
            {
                "maximum_length": maximum,
                "physical_batch": count,
                "accumulation": 1,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "optimizer_mutated": True,
                "paired_rng_verified": True,
            }
        )
        del logits, left, right, loss
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    save_json(
        REPORT / "cuda_smoke_deterministic.json",
        {
            "status": "passed",
            "precision": "float32",
            "mixed_precision_enabled": False,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "regimes": evidence,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
    )
    print("CUDA smoke passed", flush=True)


def freeze_panels(data, primary_count):
    frozen = json.loads((PREP / "diagnostic_panel.json").read_text())
    excluded = set(frozen["sample_ids"])
    groups = [
        sorted(
            (r for r in data["validation"] if r["sample_id"] not in excluded and bucket(len(r["token_ids"])) == b),
            key=lambda r: stable_seed(6121, r["sample_id"]),
        )
        for b in range(5)
    ]
    selected, pos = [], [0] * 5
    while len(selected) < primary_count:
        for b in range(5):
            if pos[b] < len(groups[b]) and len(selected) < primary_count:
                selected.append(groups[b][pos[b]])
                pos[b] += 1
    donor_ids = {}
    for row in selected:
        choices = [
            r
            for r in selected
            if bucket(len(r["token_ids"])) == bucket(len(row["token_ids"]))
            and r["sample_id"] != row["sample_id"]
            and r["sequence"] != row["sequence"]
        ]
        donor_ids[row["sample_id"]] = min(choices, key=lambda r: stable_seed(6121, row["sample_id"], r["sample_id"]))[
            "sample_id"
        ]
    save_json(
        REPORT / "primary_panel.json",
        {
            "sample_ids": [r["sample_id"] for r in selected],
            "donor_ids": donor_ids,
            "seed": 6121,
            "strata_counts": pos,
            "disjoint_diagnostic": True,
        },
    )


def evaluate(model, rows, donor_ids, baseline):
    by_id = {r["sample_id"]: r for r in rows}
    panels = {}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for fraction in FRACTIONS:
            records = []
            # Sorting changes only physical batches; masks/donors stay sample-keyed.
            ordered = sorted(rows, key=lambda r: len(r["token_ids"]))
            for start in range(0, len(rows), 8):
                batch = ordered[start : start + 8]
                targets, valid = collate(batch)
                ids = [r["sample_id"] for r in batch]
                selected = deterministic_mask(targets, valid, ids, fraction, 6111)
                panel = conditions(targets, valid, selected, ids, [by_id[donor_ids[sid]] for sid in ids], 6111)
                assert all((x[selected] == 1).all() for x in panel.values())
                scores = {
                    name: protein_ce(
                        model(tokens.to(device), valid.to(device)),
                        targets.to(device),
                        selected.to(device),
                        valid.to(device),
                    )
                    .cpu()
                    .tolist()
                    for name, tokens in panel.items()
                }
                for i, row in enumerate(batch):
                    b = bucket(len(row["token_ids"]))
                    labels = targets[i, selected[i]].numpy() - 2
                    metrics = {name: values[i] for name, values in scores.items()}
                    for name, probs in [
                        ("uniform", baseline["uniform"]),
                        ("global_unigram", baseline["global"]),
                        ("bucket_unigram", baseline["length_bucketed"][b]),
                    ]:
                        metrics[name] = float(-np.log(np.asarray(probs)[labels]).mean())
                    records.append({"sample_id": row["sample_id"], "length_stratum": b, "ce": metrics})
            panels[str(fraction)] = records
    model.train()
    return panels


def summarize(panels):
    result = {}
    for fraction, records in panels.items():
        result[fraction] = {
            "mean_ce": {name: float(np.mean([r["ce"][name] for r in records])) for name in records[0]["ce"]},
            "overall": gate_panel(records),
            "strata": {str(b): gate_panel([r for r in records if r["length_stratum"] == b]) for b in range(5)},
        }
    per_fraction = [{r["sample_id"]: r for r in records} for records in panels.values()]
    aggregate = []
    for sid, first in per_fraction[0].items():
        aggregate.append(
            {
                "sample_id": sid,
                "length_stratum": first["length_stratum"],
                "ce": {
                    name: float(np.mean([records[sid]["ce"][name] for records in per_fraction])) for name in first["ce"]
                },
            }
        )
    return {
        "mask_fractions": result,
        "classification": classify_s1(panels),
        "aggregate": {
            "mean_ce": {name: float(np.mean([r["ce"][name] for r in aggregate])) for name in aggregate[0]["ce"]},
            "overall": gate_panel(aggregate),
            "strata": {str(b): gate_panel([r for r in aggregate if r["length_stratum"] == b]) for b in range(5)},
            "unit": "protein_mean_over_three_mask_fractions",
        },
    }


def checkpoint(model, opt, step, epoch, cursor, proteins, tokens):
    path = OUT / f"checkpoint-{step:05d}.pt"
    assert not path.exists()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "successful_updates": step,
            "epoch": epoch,
            "batch_cursor": cursor,
            "processed_proteins": proteins,
            "processed_valid_tokens": tokens,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "preparation_commit": "ef30a6f759a4032e89c6be881c4c137f5de6dac7",
            "execution_contract_sha256": sha(REPORT / "execution_contract.json"),
        },
        path,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["successful_updates"] == step
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), state["model"][name]), f"checkpoint tensor mismatch: {name}"
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "successful_updates": step}


def train(data):
    smoke = json.loads((REPORT / "cuda_smoke_deterministic.json").read_text())
    assert smoke["status"] == "passed" and smoke["deterministic_algorithms"] is True
    contract = json.loads((REPORT / "execution_contract.json").read_text())
    assert contract["updates"] == 2000 and contract["boundaries"] == list(BOUNDARIES)
    assert contract["full_tests_passed"] is True
    torch.manual_seed(6011)
    np.random.seed(6011)
    random.seed(6011)
    torch.cuda.manual_seed_all(6011)
    model = SequenceContextTransformer().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    baseline = json.loads((PREP / "train_baselines.json").read_text())
    by_id = {r["sample_id"]: r for r in data["validation"]}
    definitions = {
        "primary": json.loads((REPORT / "primary_panel.json").read_text()),
        "independent": json.loads((PREP / "diagnostic_panel.json").read_text()),
    }
    assert not set(definitions["primary"]["sample_ids"]) & set(definitions["independent"]["sample_ids"])
    boundaries, checks, epoch, cursor, processed_proteins, processed_tokens = {}, [], 0, 0, 0, 0
    schedule = list(batches(data["train"], epoch))

    def boundary(step):
        checks.append(checkpoint(model, opt, step, epoch, cursor, processed_proteins, processed_tokens))
        entry = {}
        for name, definition in definitions.items():
            rows = [by_id[sid] for sid in definition["sample_ids"]]
            panels = evaluate(model, rows, definition["donor_ids"], baseline)
            save_json(REPORT / f"{name}_paired_records_{step:05d}.json", panels)
            entry[name] = summarize(panels)
        save_json(REPORT / f"evaluation_{step:05d}.json", entry)
        boundaries[str(step)] = entry
        print(
            f"Evaluation boundary {step}: {json.dumps({n: e['classification'] for n, e in entry.items()})}", flush=True
        )

    boundary(0)
    telemetry = (OUT / "telemetry.jsonl").open("x")
    start = time.monotonic()
    for step in range(1, 2001):
        if cursor >= len(schedule):
            epoch += 1
            cursor = 0
            schedule = list(batches(data["train"], epoch))
        rows = [data["train"][i] for i in schedule[cursor]]
        cursor += 1
        fraction = FRACTIONS[(step - 1) % 3]
        targets, valid, selected, normal, shuffled = (x.cuda() for x in training_conditions(rows, fraction, epoch))
        lr = 1e-4 * min(step / 1000, 1.0)
        for group in opt.param_groups:
            group["lr"] = lr
        n, s = paired_forwards(model, normal, shuffled, valid)
        ce_n, ce_s = protein_ce(n, targets, selected, valid), protein_ce(s, targets, selected, valid)
        hinge = torch.relu(ce_n - ce_s.detach() + 0.05)
        loss = objective(n, s, targets, selected, valid)
        assert torch.isfinite(n).all() and torch.isfinite(s).all() and torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        assert all(
            torch.isfinite(v).all() for state in opt.state.values() for v in state.values() if torch.is_tensor(v)
        )
        processed_proteins += len(rows)
        processed_tokens += int(valid.sum())
        record = {
            "update": step,
            "normal_ce": float(ce_n.mean().detach()),
            "shuffled_ce": float(ce_s.mean()),
            "context_gap": float((ce_s - ce_n.detach()).mean()),
            "context_hinge": float(hinge.mean().detach()),
            "active_hinge_fraction": float((hinge > 0).float().mean()),
            "gradient_norm": float(norm),
            "clip_coefficient": min(1.0, 1.0 / (float(norm) + 1e-6)),
            "learning_rate": lr,
            "mask_fraction": fraction,
            "protein_lengths": valid.sum(1).tolist(),
            "sample_ids": [r["sample_id"] for r in rows],
            "epoch": epoch,
            "cursor": cursor,
            "processed_proteins": processed_proteins,
            "processed_valid_tokens": processed_tokens,
        }
        telemetry.write(json.dumps(record) + "\n")
        telemetry.flush()
        if step % 50 == 0:
            print(
                f"Update {step}/2000 CE={record['normal_ce']:.5f} elapsed={time.monotonic() - start:.1f}s", flush=True
            )
        if step in BOUNDARIES:
            boundary(step)
    telemetry.close()
    final = boundaries["2000"]
    primary, independent = final["primary"]["classification"], final["independent"]["classification"]
    classification = primary if primary == independent else ("S1-D" if "S1-D" in (primary, independent) else "S1-E")
    save_json(
        REPORT / "results.json",
        {
            "classification": classification,
            "updates_completed": 2000,
            "checkpoints": checks,
            "boundaries": boundaries,
            "processed_proteins": processed_proteins,
            "processed_valid_tokens": processed_tokens,
            "elapsed_training_and_evaluation_seconds": time.monotonic() - start,
            "S2_started": False,
            "hyperparameter_selection": False,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "freeze-panels", "train"])
    parser.add_argument("--primary-count", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = inputs()
    if args.mode == "smoke":
        cuda_smoke(data)
    elif args.mode == "freeze-panels":
        freeze_panels(data, args.primary_count)
    else:
        train(data)
