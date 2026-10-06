"""E012 V3 fixed-checkpoint evaluation only. Never construct an optimizer or save a checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from protein_sequence_generation.e012 import batch, positions
from protein_sequence_generation.e012_continuation import IdentitySampler, routed_microbatches
from protein_sequence_generation.e012_generalization import (
    bulk_prefix_batch,
    evaluation_only_guard,
    panel_summary,
    relative_bins,
    select_train_panel,
)
from protein_sequence_generation.metrics import sequence_cross_entropy
from scripts import run_e012_causal_rope as old

ROOT = old.ROOT
HIST = ROOT / "reports/experiments/E012_causal_rope_sequence/continuation_2k_to_10k_v1"
CONT_OUT = ROOT / "outputs/e012_causal_rope_sequence/continuation_2k_to_10k_v1"
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/generalization_audit_v1"
OUT = ROOT / "outputs/e012_causal_rope_sequence/generalization_audit_v1"
STEPS = [2000, 5000, 7500, 10000]
CHECKPOINTS = {
    2000: (old.OUT / "checkpoint-02000.pt", "ff666eccdb6d28d51f2d3d561242152512e2dddf74fc0f251429d55b886d0188"),
    5000: (CONT_OUT / "checkpoint-05000.pt", "06ef3eddc9560ee4d3160a8aeb280d14bd1112c334d19f023556dbf3e2b32df4"),
    7500: (CONT_OUT / "checkpoint-07500.pt", "533340bf81b01111401c2765cb0e0ed16c57429a36bf3dc8608b82d66057bd11"),
    10000: (CONT_OUT / "checkpoint-10000.pt", "23d4aad8327c491d31d22192e2b4f51bda8e68f13e1c213ed7925cfcafb7bf89"),
}


def read(path):
    return json.loads(path.read_text())


def verify_historical():
    verified = {}
    for directory in [old.REPORT, HIST]:
        manifest = read(directory / "result_manifest.json")
        for name, digest in manifest["published_artifact_sha256"].items():
            path = directory / name
            assert old.sha(path) == digest, str(path)
            verified[str(path.relative_to(ROOT))] = digest
        for name, digest in read(directory / "contract.json")["protected_hashes"].items():
            path = ROOT / name
            assert old.sha(path) == digest, name
            verified[name] = digest
    for step, (path, digest) in CHECKPOINTS.items():
        assert old.sha(path) == digest, step
        verified[str(path.relative_to(ROOT))] = digest
    for directory, manifest_key in [(HIST, "raw_telemetry_sha256")]:
        digest = read(directory / "result_manifest.json")[manifest_key]
        assert old.sha(CONT_OUT / "telemetry.jsonl") == digest
        verified[str((CONT_OUT / "telemetry.jsonl").relative_to(ROOT))] = digest
    verified[str((old.OUT / "telemetry.jsonl").relative_to(ROOT))] = old.sha(old.OUT / "telemetry.jsonl")
    return verified


def protected_worktrees():
    result = {}
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    for path in [
        ROOT.parent / "proteinGen-sequence-context",
        Path("/mnt/d/Simone/proteinGen"),
        ROOT.parent / "proteinGen-hybrid-local-global",
    ]:
        result[str(path)] = {
            k: subprocess.check_output(["git", *cmd], cwd=path, env=env, text=True).strip()
            for k, cmd in [
                ("head", ["rev-parse", "HEAD"]),
                ("branch", ["branch", "--show-current"]),
                ("status", ["status", "--porcelain"]),
            ]
        }
    return result


def populations():
    training, validation = old.load_rows("train"), old.load_rows("validation")
    assert len(training) == 231743 and len(validation) == 23307
    assert not {r["sample_id"] for r in training} & {r["sample_id"] for r in validation}
    assert not {r["sequence"] for r in training} & {r["sequence"] for r in validation}
    lookup = {r["sample_id"]: r for r in training + validation}
    panels = read(old.REPORT / "panels.json")
    panels["train"] = read(REPORT / "train_panel.json")["sample_ids"]
    rows = {
        name: sorted([lookup[sid] for sid in ids], key=lambda r: (r["length"], r["sample_id"]))
        for name, ids in panels.items()
    }
    assert all(len(p) == len({r["sample_id"] for r in p}) == 2048 for p in rows.values())
    assert all(r["split"] == "train" for r in rows["train"])
    assert all(r["split"] == "validation" for n in ["primary", "independent"] for r in rows[n])
    return training, rows


def prepare():
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        == "8d1a265f19c02bf2a0ec6734d30aa02e90e6cc6b"
    )
    assert not (REPORT / "contract.json").exists()
    hashes = verify_historical()
    training, validation = old.load_rows("train"), old.load_rows("validation")
    panels = read(old.REPORT / "panels.json")
    lookup = {r["sample_id"]: r for r in validation}
    primary = [lookup[sid] for sid in panels["primary"]]
    panel = select_train_panel(training, primary, {r["sample_id"] for r in validation})
    assert panel["stratum_counts"] == [410, 410, 410, 409, 409] and panel["size"] == 2048
    panel["source_train_sha256"] = old.sha(old.OUT / "train.parquet")
    old.save(REPORT / "train_panel.json", panel, exclusive=True)
    lookup.update({r["sample_id"]: r for r in training})
    old.save(
        REPORT / "train_positions.json",
        {sid: positions(lookup[sid]["length"]) for sid in panel["sample_ids"]},
        exclusive=True,
    )
    hashes[str((REPORT / "train_panel.json").relative_to(ROOT))] = old.sha(REPORT / "train_panel.json")
    hashes[str((REPORT / "train_positions.json").relative_to(ROOT))] = old.sha(REPORT / "train_positions.json")
    old.save(
        REPORT / "contract.json",
        {
            "historical_result_commit": "8d1a265f19c02bf2a0ec6734d30aa02e90e6cc6b",
            "checkpoints": STEPS,
            "checkpoint_selection": "fixed retrospectively, no model selection",
            "panels": ["train", "primary", "independent"],
            "panel_size": 2048,
            "precision": "historical CUDA bfloat16 forward, float32 log-softmax; identical across panels",
            "canonical_evaluator": "scripts.run_e012_causal_rope.likelihood, unmodified, eval/no_grad, batch16",
            "context_evaluator": (
                "scripts.run_e012_causal_rope.diagnostics with tensor-identical CPU construction/bulk CUDA transfers"
            ),
            "context_steps": STEPS,
            "historical_heldout_context_reused": [2000, 5000, 10000],
            "relative_position_bins": "historical 10 deciles: min(floor(10*i/N),9), zero-based i",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 12112,
            "interpretation": (
                "GEN-A: improving matched TRAIN CE and expanding gap with plateau/worse heldout; GEN-B: "
                "joint plateau; GEN-C: early/short failure on both without broad gap; GEN-D: dominant "
                "predeclared stratum shift; GEN-E: unreconciled or invalid. Report magnitudes; no "
                "invented numerical gate or causal proof of corpus size."
            ),
            "training_authorized": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "source_hashes": hashes,
            "protected_worktrees_before": protected_worktrees(),
            "same_batch_protocol": (
                "2000/5000/7500: exact next historical optimizer batch from saved sampler plus log, saved "
                "incoming RNG; 10000: reconstruct last recorded batch from sampler7500, final LR zero "
                "means weights exact but incoming dropout RNG unavailable; use saved outgoing RNG "
                "explicitly. No optimizer/backward. Compare historical train-mode computation to "
                "canonical eval on same identities."
            ),
            "aa_statistics": (
                "20 canonical class probabilities retain full24-head mass; predictive entropy full24; "
                "equal-protein and token-weighted frequencies/NLL contributions; conditional per-class "
                "NLL token-weighted"
            ),
        },
        exclusive=True,
    )
    print("TRAIN panel frozen", old.sha(REPORT / "train_panel.json"), flush=True)


def gpu_check():
    commands = subprocess.run(
        ["ps", "-C", "python", "-C", "python3", "-o", "pid=,args="], capture_output=True, text=True
    ).stdout
    for line in commands.splitlines():
        if any(f"run_e010_local_feasibility_v{v}.py {mode}" in line for v in [7, 8] for mode in ["run", "reproduce"]):
            version = 8 if "local_feasibility_v8.py" in line else 7
            config = (
                ROOT.parent / f"proteinGen-hybrid-local-global/configs/e010_phase4d_local_feasibility_v{version}.yaml"
            )
            assert "device: cpu" in config.read_text() and "no_cuda: true" in config.read_text()
        elif any(name in line for name in ["run_e010_", "run_e011_", "run_e012_", "train_sequence", "train_diffusion"]):
            raise RuntimeError(f"Other research execution requires GPU ownership review: {line}")
    driver = subprocess.run(["cmd.exe", "/c", "nvidia-smi"], capture_output=True, text=True, errors="replace").stdout
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {
        "processes": commands,
        "driver": driver,
        "CUDA_used": True,
        "precision": "bfloat16",
        "audited_CPU_only_v7_v8_owners_allowed": True,
    }


def enrich(net, rows, records):
    net.eval()
    lookup = {r["sample_id"]: r for r in records}
    for start in range(0, len(rows), 16):
        chosen = rows[start : start + 16]
        data = batch(chosen, "cuda")
        with old.autocast():
            logits = net(data["input_ids"], data["lengths"], data["attention_mask"])
        logp = logits.float().log_softmax(-1)
        probs = logp.exp()
        entropy = -(probs * logp).sum(-1)
        maxp = probs.max(-1).values
        correctp = probs.gather(-1, data["target_ids"][..., None]).squeeze(-1)
        lp, pp, ent, mx, cp, targets = [
            v.cpu().numpy() for v in (logp, probs, entropy, maxp, correctp, data["target_ids"])
        ]
        for j, row in enumerate(chosen):
            n = row["length"]
            aa = targets[j, :n] - 4
            vals = -lp[j, np.arange(n), targets[j, :n]]
            r = lookup[row["sample_id"]]
            assert float(vals.mean()) == r["normal"], "canonical evaluator parity failed"
            assert np.allclose(
                [vals[relative_bins(n) == k].mean() for k in range(10)], r["relative_position_ce"], atol=0, rtol=0
            )
            r.update(
                {
                    "predictive_entropy": float(ent[j, :n].mean()),
                    "max_probability": float(mx[j, :n].mean()),
                    "correct_token_probability": float(cp[j, :n].mean()),
                    "aa_counts": np.bincount(aa, minlength=20).tolist(),
                    "target_aa_frequencies": (np.bincount(aa, minlength=20) / n).tolist(),
                    "mean_predicted_aa_probabilities": pp[j, :n, 4:].mean(0).tolist(),
                    "aa_nll_contributions": (np.bincount(aa, weights=vals, minlength=20) / n).tolist(),
                }
            )
    return records


def historical_context(step, panel):
    directory = old.REPORT if step == 2000 else HIST
    name = (
        f"{panel}_paired_identity_metrics.json" if step == 2000 else f"{panel}_paired_identity_metrics_{step:05d}.json"
    )
    records = read(directory / name)
    if isinstance(records, dict):
        raise ValueError("unexpected historical record container")
    return {
        r["sample_id"]: {
            k: r[k]
            for k in [
                "prefix_normal",
                "shuffle",
                "last1",
                "last4",
                "last8",
                "last16",
                "last32",
                "last64",
                "kl",
                "js",
                "top1_change",
                "top3_change",
            ]
        }
        for r in records
    }


def telemetry():
    return {
        r["update"]: r for line in (CONT_OUT / "telemetry.jsonl").read_text().splitlines() if (r := json.loads(line))
    }


def same_batch(net, ck, step, training, logs):
    if step == 2000:
        sampler = IdentitySampler(ck["data_order"], ck["data_cursor"])
    elif step < 10000:
        sampler = IdentitySampler.from_state(ck["sampler"])
    else:
        prior = torch.load(CHECKPOINTS[7500][0], map_location="cpu", weights_only=False)
        sampler = IdentitySampler.from_state(prior["sampler"])
        for _ in range(2499):
            sampler.take64()
    update = step + 1 if step < 10000 else 10000
    selected = [training[i] for i in sampler.take64()]
    assert [r["sample_id"] for r in selected] == logs[update]["sample_ids"], "exact sampler reconstruction failed"
    plan = read(HIST / "contract.json")["selected_plan"]
    chunks = routed_microbatches(selected, plan)
    assert [len(x) for x in chunks] == logs[update]["microbatches"]
    if step == 10000:
        assert ck["optimizer"]["param_groups"][0]["lr"] == 0 and logs[10000]["learning_rate"] == 0
    rng = old.rng_state()
    old.restore_rng(ck["rng"])
    net.train()
    ce = 0.0
    for chosen in chunks:
        data = batch(chosen, "cuda")
        with old.autocast():
            logits = net(data["input_ids"], data["lengths"], data["attention_mask"])
            loss = sequence_cross_entropy(logits, data["target_ids"], data["attention_mask"], reduction="sequence_mean")
        ce += float(loss) * len(chosen) / 64
    net.eval()
    canonical = old.likelihood(
        net, sorted(selected, key=lambda r: (r["length"], r["sample_id"])), read(old.REPORT / "baselines.json")
    )
    eval_ce = float(np.mean([r["normal"] for r in canonical]))
    old.restore_rng(rng)
    return {
        "checkpoint": step,
        "historical_batch_update": update,
        "sample_ids": [r["sample_id"] for r in selected],
        "microbatches": [len(x) for x in chunks],
        "sampler_membership_exact": True,
        "incoming_RNG_exact": step < 10000,
        "original_logged_ce": logs[update]["sequence_mean_ce"],
        "training_computation_no_grad_ce": ce,
        "canonical_eval_same_batch_ce": eval_ce,
        "train_mode_minus_eval_mode": ce - eval_ce,
        "reconstructed_minus_original_logged_ce": ce - logs[update]["sequence_mean_ce"],
        "limitation": (
            "No-grad forward can use different kernels than original grad-enabled forward; measure "
            "reproduction difference. At10000 incoming dropout RNG unavailable; saved outgoing RNG "
            "used, final LR zero preserves batch10000 pre-update weights."
        ),
    }


def run():
    contract = read(REPORT / "contract.json")
    for name, digest in contract["source_hashes"].items():
        assert old.sha(ROOT / name) == digest, name
    verify_historical()
    assert not (OUT / "completed.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    old.seed()
    owners = gpu_check()
    old.save(
        OUT / "execution_start.json",
        {
            **owners,
            "git_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "contract_sha256": old.sha(REPORT / "contract.json"),
        },
        exclusive=True,
    )
    training, panels = populations()
    frozen = read(old.REPORT / "positions.json")
    frozen["train"] = read(REPORT / "train_positions.json")
    baselines, logs = read(old.REPORT / "baselines.json"), telemetry()
    began = time.monotonic()
    fingerprints = {}
    original_builder = old.prefix_batch
    old.prefix_batch = bulk_prefix_batch
    try:
        with evaluation_only_guard():
            for step in STEPS:
                path, _ = CHECKPOINTS[step]
                ck = torch.load(path, map_location="cpu", weights_only=False)
                assert ck["successful_updates"] == step
                net = old.model().cuda()
                net.load_state_dict(ck["model"], strict=True)
                net.requires_grad_(False)
                net.eval()
                fingerprint = old.weights_hash(net)
                print(f"Checkpoint {step}: matched likelihood/confidence, all panels", flush=True)
                for panel in ["train", "primary", "independent"]:
                    rows = panels[panel]
                    records = old.likelihood(net, rows, baselines)
                    enrich(net, rows, records)
                    old.save(OUT / f"{panel}_likelihood_{step:05d}.json", records, exclusive=True)
                    old.save(OUT / f"{panel}_summary_{step:05d}.json", panel_summary(records), exclusive=True)
                    print(f"{step} {panel} CE={np.mean([r['normal'] for r in records]):.7f}", flush=True)
                    if panel != "train":
                        historic_out = old.OUT if step == 2000 else CONT_OUT
                        previous = read(historic_out / f"{panel}_likelihood_{step:05d}.json")
                        prev = {r["sample_id"]: r for r in previous}
                        assert all(abs(r["normal"] - prev[r["sample_id"]]["normal"]) <= 1e-6 for r in records), (
                            "historical evaluator reproducibility failure"
                        )
                old.save(OUT / f"same_batch_{step:05d}.json", same_batch(net, ck, step, training, logs), exclusive=True)
                for panel in ["train", "primary", "independent"]:
                    if panel != "train" and step != 7500:
                        context = historical_context(step, panel)
                    else:
                        print(f"{step} {panel}: exact frozen prefix/window diagnostic", flush=True)
                        context = old.diagnostics(net, panels[panel], frozen[panel])
                    assert set(context) == {r["sample_id"] for r in panels[panel]}
                    old.save(OUT / f"{panel}_context_{step:05d}.json", context, exclusive=True)
                assert old.weights_hash(net) == fingerprint
                assert all(p.grad is None for p in net.parameters())
                fingerprints[str(step)] = fingerprint
                del net, ck
                torch.cuda.empty_cache()
                print(f"Checkpoint {step} complete, elapsed={time.monotonic() - began:.1f}s", flush=True)
    finally:
        old.prefix_batch = original_builder
    hashes = verify_historical()
    old.save(
        OUT / "completed.json",
        {
            "seconds": time.monotonic() - began,
            "source_hashes_after": hashes,
            "model_fingerprints_unchanged": fingerprints,
            "training_launched": False,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "CUDA_used": True,
            "precision": "bfloat16",
            "protected_worktrees_after": protected_worktrees(),
        },
        exclusive=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "run":
        run()
    else:
        print(json.dumps(verify_historical(), indent=2))
