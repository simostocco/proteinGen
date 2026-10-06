"""Preregistered E012 high-throughput continuation, calibration and fixed-boundary evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from protein_sequence_generation.e012 import STRATA, batch, classification, stratum, summarize
from protein_sequence_generation.e012_continuation import (
    SOURCE_SHA256,
    ContinuationScheduler,
    IdentitySampler,
    continuation_classification,
    finite_state,
    partition64,
    routed_microbatches,
    weighted_loss,
)
from protein_sequence_generation.metrics import sequence_cross_entropy
from scripts import run_e012_causal_rope as old

ROOT = old.ROOT
HIST = old.REPORT
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/continuation_2k_to_10k_v1"
OUT = ROOT / "outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1"
SOURCE = old.OUT / "checkpoint-02000.pt"
BOUNDARIES = [3000, 5000, 7500, 10000]
FULL = [5000, 10000]


def verify_inputs():
    assert old.sha(SOURCE) == SOURCE_SHA256
    c = json.loads((HIST / "contract.json").read_text())
    m = json.loads((HIST / "result_manifest.json").read_text())
    assert old.sha(HIST / "contract.json") == m["contract_sha256"]
    for name, digest in c["protected_hashes"].items():
        assert old.sha(ROOT / name) == digest, name
    for name, digest in m["published_artifact_sha256"].items():
        assert old.sha(HIST / name) == digest, name
    return c


def gpu_guard():
    commands = subprocess.run(
        ["ps", "-C", "python", "-C", "python3", "-o", "pid=,args="], capture_output=True, text=True, check=False
    ).stdout
    for line in commands.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) < 2 or int(fields[0]) == os.getpid():
            continue
        cmd = fields[1]
        if "run_e010_phase4d_cartesian_oracle_v5.py run" in cmd:
            # Audited CPU-only oracle, no package/environment changes.
            assert (
                "device: cpu"
                in (
                    ROOT.parent / "proteinGen-hybrid-local-global/configs/e010_phase4d_cartesian_oracle_v5.yaml"
                ).read_text()
            )
            continue
        if any(x in cmd for x in ["run_e010_", "run_e011_", "run_e012_", "train_sequence", "train_diffusion"]):
            raise RuntimeError(f"CUDA ownership conflict: {line}")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return commands


def restore(path=SOURCE):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old.seed()
    net = old.model().cuda()
    net.load_state_dict(ck["model"], strict=True)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0003, betas=(0.9, 0.95), weight_decay=0.01)
    opt.load_state_dict(ck["optimizer"])
    source_ids = ck["optimizer"]["param_groups"][0]["params"]
    for parameter, identifier in zip(opt.param_groups[0]["params"], source_ids, strict=True):
        for key in ["exp_avg", "exp_avg_sq", "step"]:
            assert torch.equal(opt.state[parameter][key].cpu(), ck["optimizer"]["state"][identifier][key])
    if path == SOURCE:
        assert ck["successful_updates"] == 2000 and ck["processed_proteins"] == 128000
        lr = ck["optimizer"]["param_groups"][0]["lr"]
        sched = ContinuationScheduler(opt, lr)
        sampler = IdentitySampler(ck["data_order"], ck["data_cursor"])
        assert all(float(s["step"]) == 2000 for s in opt.state.values())
    else:
        sched = ContinuationScheduler(opt, ck["scheduler"]["lr_checkpoint"])
        sched.load_state_dict(ck["scheduler"])
        sampler = IdentitySampler.from_state(ck["sampler"])
    assert finite_state(net, opt)
    assert all(torch.equal(v.cpu(), ck["model"][k]) for k, v in net.state_dict().items())
    old.restore_rng(ck["rng"])
    return net, opt, sched, sampler, ck


def backward(net, rows, *, weighted=True):
    data = batch(rows, "cuda")
    with old.autocast():
        logits = net(data["input_ids"], data["lengths"], data["attention_mask"])
        loss = sequence_cross_entropy(logits, data["target_ids"], data["attention_mask"], reduction="sequence_mean")
    assert torch.isfinite(loss)
    (weighted_loss(loss, len(rows)) if weighted else loss).backward()
    detached = logits.detach().float()
    losses = F.cross_entropy(detached.reshape(-1, 24), data["target_ids"].reshape(-1), reduction="none").reshape_as(
        data["target_ids"]
    )
    return {
        "ce": float(loss.detach()),
        "loss_sum": float((losses * data["attention_mask"]).sum()),
        "correct": int(((detached.argmax(-1) == data["target_ids"]) & data["attention_mask"]).sum()),
        "residues": sum(r["length"] for r in rows),
    }


def calibration():
    verify_inputs()
    owners = gpu_guard()
    assert not (REPORT / "calibration.json").exists()
    net, opt, _, _, ck = restore()
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    # Free includes resident model/moments. Budget preserves both device and desktop margin.
    resident_reserved = torch.cuda.memory_reserved()
    threshold = min(int(total * 0.90), free + resident_reserved - int(0.35 * 1024**3))
    records, plan = [], {}
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for index, (_, length) in enumerate(STRATA):
        rows = [
            {"sample_id": f"calibration-{i}", "length": length, "sequence": (alphabet * 25)[:length]} for i in range(64)
        ]
        previous_peak = None
        for capacity in [8, 16, 24, 32, 40, 48, 56, 64]:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            parts = partition64(capacity)
            # Avoid known unsafe allocations; longest successful case estimates incremental activations.
            if previous_peak is not None:
                estimate = resident_reserved + (previous_peak[1] - resident_reserved) * capacity / previous_peak[0]
                if estimate > threshold * 1.05:
                    records.append(
                        {
                            "stratum": index,
                            "length": length,
                            "physical_proteins": capacity,
                            "partition": parts,
                            "status": "skipped_predicted_unsafe",
                            "estimated_bytes": estimate,
                        }
                    )
                    continue
            try:
                timings = []
                net.train()
                for _repeat in range(3):
                    old.restore_rng(ck["rng"])
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    began = time.monotonic()
                    offset = 0
                    for n in parts:
                        backward(net, rows[offset : offset + n])
                        offset += n
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), float("inf"), error_if_nonfinite=True)
                    assert torch.isfinite(norm)
                    torch.cuda.synchronize()
                    timings.append(time.monotonic() - began)
                allocated = torch.cuda.max_memory_allocated()
                reserved = torch.cuda.max_memory_reserved()
                previous_peak = (capacity, reserved)
                elapsed = float(np.median(timings[1:]))
                result = {
                    "stratum": index,
                    "length": length,
                    "physical_proteins": capacity,
                    "partition": parts,
                    "status": "passed",
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "forward_backward_seconds": elapsed,
                    "proteins_per_second": 64 / elapsed,
                    "residues_per_second": 64 * length / elapsed,
                    "finite_loss": True,
                    "finite_gradients": True,
                    "safe": reserved <= threshold,
                }
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                result = {
                    "stratum": index,
                    "length": length,
                    "physical_proteins": capacity,
                    "partition": parts,
                    "status": "calibration_oom",
                    "safe": False,
                }
                previous_peak = (capacity, threshold * 1.2)
            records.append(result)
            print(f"Calibration {index} length={length} capacity={capacity}: {result}", flush=True)
        safe = [r for r in records if r["stratum"] == index and r.get("safe")]
        assert safe, "no safe execution plan"
        best = min(safe, key=lambda r: r["forward_backward_seconds"])
        plan[str(index)] = {
            k: best[k]
            for k in [
                "physical_proteins",
                "partition",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "forward_backward_seconds",
            ]
        }
    opt.zero_grad(set_to_none=True)
    # Controlled eval-mode full-model gradient and Adam update equivalence, mixed precision disabled.
    fixed = [
        {"sample_id": f"equivalence-{i}", "length": 20 + i % 17, "sequence": (alphabet * 3)[: 20 + i % 17]}
        for i in range(64)
    ]
    eq = []
    for parts in [partition64(8), plan["0"]["partition"]]:
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        net.eval()
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        offset = 0
        for n in parts:
            data = batch(fixed[offset : offset + n], "cuda")
            offset += n
            logits = net(data["input_ids"], data["lengths"], data["attention_mask"])
            loss = sequence_cross_entropy(logits, data["target_ids"], data["attention_mask"])
            weighted_loss(loss, n).backward()
            loss_total += float(loss.detach()) * n / 64
        grad = torch.cat([p.grad.flatten() for p in net.parameters()]).detach().cpu()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        for group in opt.param_groups:
            group["lr"] = 0.0003
        opt.step()
        weights = torch.cat([p.flatten() for p in net.parameters()]).detach().cpu()
        eq.append((loss_total, grad, weights))
    cosine = float(F.cosine_similarity(eq[0][1], eq[1][1], dim=0))
    grad_relative = float(torch.linalg.vector_norm(eq[0][1] - eq[1][1]) / torch.linalg.vector_norm(eq[0][1]))
    update_max = float((eq[0][2] - eq[1][2]).abs().max())
    assert abs(eq[0][0] - eq[1][0]) < 1e-5 and cosine > 0.99999 and grad_relative < 1e-4 and update_max < 1e-5
    result = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(),
        "total_device_bytes": total,
        "initial_free_bytes": free,
        "reserved_safety_threshold_bytes": threshold,
        "target_fraction": 0.90,
        "desktop_margin_bytes": int(0.35 * 1024**3),
        "candidates": records,
        "selected_plan": plan,
        "effective_proteins": 64,
        "precision": "bfloat16",
        "calibration_optimizer_updates": 0,
        "equivalence_smoke_optimizer_steps_on_disposable_restorations": 2,
        "equivalence": {
            "dropout_control": "eval only, architecture dropout unchanged for training",
            "precision": "float32",
            "loss_difference": eq[0][0] - eq[1][0],
            "gradient_cosine": cosine,
            "gradient_relative_error": grad_relative,
            "optimizer_weight_max_difference": update_max,
        },
        "process_snapshot": owners,
    }
    old.save(REPORT / "calibration.json", result, exclusive=True)
    print("CUDA calibration and numerical batch semantics passed", flush=True)


def save_checkpoint(net, opt, sched, sampler, step, proteins, residues, contract):
    path = OUT / f"checkpoint-{step:05d}.pt"
    assert not path.exists()
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "successful_updates": step,
            "processed_proteins": proteins,
            "processed_residues": residues,
            "rng": old.rng_state(),
            "sampler": sampler.state_dict(),
            "scaler": None,
            "batch_plan": contract["selected_plan"],
            "contract_sha256": old.sha(REPORT / "contract.json"),
        },
        path,
    )
    return {"filename": path.name, "sha256": old.sha(path), "updates": step}


def resume_smoke():
    verify_inputs()
    gpu_guard()
    calibration = json.loads((REPORT / "calibration.json").read_text())
    net, opt, sched, sampler, ck = restore()
    training = old.load_rows("train")

    def update():
        net.train()
        opt.zero_grad(set_to_none=True)
        rows = [training[i] for i in sampler.take64()]
        loss = 0.0
        for chosen in routed_microbatches(rows, calibration["selected_plan"]):
            stats = backward(net, chosen)
            loss += stats["ce"] * len(chosen) / 64
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        sched.set_update(sched.global_update + 1)
        opt.step()
        assert finite_state(net, opt)
        return loss

    # Two disposable smoke updates, not continuation optimizer budget.
    update()
    tmp = OUT / "resume_smoke.pt"
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng": old.rng_state(),
            "sampler": sampler.state_dict(),
        },
        tmp,
    )
    expected_loss = update()
    expected_hash = old.weights_hash(net)
    del net, opt
    torch.cuda.empty_cache()
    net, opt, sched, sampler, _ = restore(tmp)
    actual = update()
    assert actual == expected_loss and old.weights_hash(net) == expected_hash
    old.save(
        REPORT / "resume_smoke.json",
        {
            "status": "passed",
            "exact_loss": actual,
            "exact_weights": True,
            "Adam_moments_restored": True,
            "source_sha256": old.sha(SOURCE),
            "smoke_checkpoint_sha256": old.sha(tmp),
            "budget_optimizer_updates": 0,
            "disposable_smoke_updates": 3,
        },
        exclusive=True,
    )


def positional_summary(records):
    # Preserve historical ten equal relative-position deciles, now reported within all length strata.
    return {
        str(s): {
            "identities": len(rows),
            "relative_position_ce_deciles": np.mean([r["relative_position_ce"] for r in rows], axis=0).tolist(),
        }
        for s in range(5)
        if (rows := [r for r in records if r["stratum"] == s])
    }


def train():
    verify_inputs()
    owners = gpu_guard()
    contract = json.loads((REPORT / "contract.json").read_text())
    for name, digest in contract["protected_hashes"].items():
        assert old.sha(ROOT / name) == digest, name
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    assert (
        head
        == subprocess.check_output(
            ["git", "rev-parse", "origin/e012-causal-rope-sequence"], cwd=ROOT, text=True
        ).strip()
    )
    assert not (OUT / "telemetry.jsonl").exists()
    net, opt, sched, sampler, ck = restore()
    proteins, residues = ck["processed_proteins"], ck["processed_residues"]
    training, validation = old.load_rows("train"), old.load_rows("validation")
    lookup = {r["sample_id"]: r for r in validation}
    panels = {
        pop: sorted([lookup[sid] for sid in ids], key=lambda r: (r["length"], r["sample_id"]))
        for pop, ids in json.loads((HIST / "panels.json").read_text()).items()
    }
    baseline = json.loads((HIST / "baselines.json").read_text())
    frozen = json.loads((HIST / "positions.json").read_text())
    old.save(
        REPORT / "execution_start.json",
        {
            "preparation_commit": head,
            "contract_sha256": old.sha(REPORT / "contract.json"),
            "source_checkpoint_sha256": old.sha(SOURCE),
            "source_lr": sched.lr_checkpoint,
            "source_scheduler_record": ck["scheduler"],
            "process_snapshot": owners,
        },
        exclusive=True,
    )
    historical = json.loads((HIST / "trajectory.json").read_text())["2000"]
    trajectory = {"2000": historical}
    checkpoints = []
    science = {}
    start = time.monotonic()
    training_seconds = 0.0
    with (OUT / "telemetry.jsonl").open("x") as log:
        for step in range(2001, 10001):
            began = time.monotonic()
            net.train()
            opt.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            indices = sampler.take64()
            rows = [training[i] for i in indices]
            micro = routed_microbatches(rows, contract["selected_plan"])
            ce, loss_sum, correct, n_res = 0.0, 0.0, 0, 0
            for chosen in micro:
                stats = backward(net, chosen)
                ce += stats["ce"] * len(chosen) / 64
                loss_sum += stats["loss_sum"]
                correct += stats["correct"]
                n_res += stats["residues"]
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            lr = sched.set_update(step)
            opt.step()
            assert finite_state(net, opt), "non-finite model or Adam state"
            torch.cuda.synchronize()
            elapsed = time.monotonic() - began
            training_seconds += elapsed
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            assert reserved <= contract["reserved_safety_threshold_bytes"], "frozen memory safety ceiling exceeded"
            proteins += 64
            residues += n_res
            record = {
                "update": step,
                "sequence_mean_ce": ce,
                "token_weighted_ce": loss_sum / n_res,
                "perplexity": math.exp(loss_sum / n_res),
                "top1_accuracy": correct / n_res,
                "learning_rate": lr,
                "gradient_norm": float(norm),
                "clipped": bool(norm > 1),
                "clip_coefficient": min(1.0, 1.0 / (float(norm) + 1e-6)),
                "microbatches": [len(x) for x in micro],
                "protein_lengths": [r["length"] for r in rows],
                "sample_ids": [r["sample_id"] for r in rows],
                "length_stratum_exposures": [sum(stratum(r["length"]) == s for r in rows) for s in range(5)],
                "processed_proteins": proteins,
                "processed_residues": residues,
                "sampler_epoch": sampler.epoch,
                "sampler_cursor": sampler.cursor,
                "time_per_update_seconds": elapsed,
                "proteins_per_second": 64 / elapsed,
                "residues_per_second": n_res / elapsed,
                "cuda_peak_allocated_bytes": allocated,
                "cuda_peak_reserved_bytes": reserved,
                "cumulative_training_seconds": training_seconds,
                "cumulative_wall_seconds": time.monotonic() - start,
                "amp_overflows": 0,
                "scaler": None,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 50 == 0:
                print(
                    f"Continuation {step}/10000 CE={ce:.5f} updates/sec={1 / elapsed:.3f} "
                    f"elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
            if step in BOUNDARIES:
                state = old.rng_state()
                checkpoints.append(save_checkpoint(net, opt, sched, sampler, step, proteins, residues, contract))
                entry = {}
                for pop, chosen in panels.items():
                    records = old.likelihood(net, chosen, baseline)
                    old.save(OUT / f"{pop}_likelihood_{step:05d}.json", records, exclusive=True)
                    entry[pop] = {
                        "normal_ce": float(np.mean([r["normal"] for r in records])),
                        "token_weighted_ce": sum(r["loss_sum"] for r in records) / sum(r["length"] for r in records),
                        "top1": float(np.mean([r["top1"] for r in records])),
                        "top3": float(np.mean([r["top3"] for r in records])),
                        "relative_position_by_stratum": positional_summary(records),
                    }
                    if step in FULL:
                        prefix = old.diagnostics(net, chosen, frozen[pop])
                        neutral = {
                            r["sample_id"]: r["normal"] for r in old.likelihood(net, chosen, baseline, neutral=True)
                        }
                        for row in records:
                            row.update(prefix[row["sample_id"]])
                            row["neutral_length"] = neutral[row["sample_id"]]
                        old.save(OUT / f"{pop}_final_records_{step:05d}.json", records, exclusive=True)
                        science.setdefault(str(step), {})[pop] = summarize(records)
                trajectory[str(step)] = entry
                old.save(REPORT / "trajectory.json", trajectory)
                old.save(REPORT / "scientific_boundaries.json", science)
                old.restore_rng(state)
                net.train()
                print(f"Boundary {step}: {entry}", flush=True)
    final = science["10000"]
    cseq = classification(final["primary"], final["independent"])
    cont = continuation_classification(final["primary"], final["independent"], cseq)
    old.save(
        REPORT / "results.json",
        {
            "CSEQ_classification": cseq,
            "continuation_classification": cont,
            "preparation_commit": head,
            "additional_updates": 8000,
            "global_update": 10000,
            "processed_proteins": proteins,
            "processed_residues": residues,
            "training_seconds": training_seconds,
            "wall_seconds": time.monotonic() - start,
            "trajectory": trajectory,
            "scientific_boundaries": science,
            "checkpoints": checkpoints,
            "contract_sha256": old.sha(REPORT / "contract.json"),
            "biological_generation_launched": False,
        },
        exclusive=True,
    )
    print(f"Completed E012 continuation: {cseq}, {cont}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["calibrate", "resume-smoke", "train"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    {"calibrate": calibration, "resume-smoke": resume_smoke, "train": train}[args.mode]()


if __name__ == "__main__":
    main()
