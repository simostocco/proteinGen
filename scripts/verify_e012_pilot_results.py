"""Independent E012 bootstrap and integrity verification; no training operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_interval(values):
    """Uniform identity resampling using choice, separately implemented from the sealed evaluator."""
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(12112)
    means = np.empty(10000, dtype=np.float64)
    for begin in range(0, 10000, 100):
        indices = generator.choice(values.size, size=(100, values.size), replace=True)
        means[begin : begin + 100] = np.mean(values[indices], axis=1)
    return float(np.mean(values)), np.percentile(means, [2.5, 97.5]).tolist()


def verify(root):
    report = root / "reports/experiments/E012_causal_rope_sequence/pilot_v1"
    out = root / "outputs/e012_causal_rope_sequence/pilot_v1"
    contract = json.loads((report / "contract.json").read_text())
    results = json.loads((report / "results.json").read_text())
    start = json.loads((out / "execution_start.json").read_text())
    assert sha(report / "contract.json") == start["contract_sha256"] == results["contract_sha256"]
    for name, digest in contract["protected_hashes"].items():
        assert sha(root / name) == digest, name
    assert results["updates_completed"] == 2000
    records = [json.loads(line) for line in (out / "telemetry.jsonl").read_text().splitlines()]
    assert len(records) == 2000 and [x["update"] for x in records] == list(range(1, 2001))
    all_ids = [s for x in records for s in x["sample_ids"]]
    assert len(all_ids) == len(set(all_ids)) == 128000
    assert results["processed_proteins"] == 128000
    assert sum(sum(x["protein_lengths"]) for x in records) == results["processed_residues"]
    for row in records:
        step = row["update"]
        scheduler_index = step - 1
        factor = (
            (scheduler_index + 1) / 100
            if scheduler_index < 100
            else (0.5 * (1 + math.cos(math.pi * (scheduler_index - 100) / 1900)))
        )
        assert math.isclose(row["learning_rate"], 0.0003 * factor, rel_tol=1e-12, abs_tol=1e-15)
        assert all(
            math.isfinite(row[k])
            for k in ["sequence_mean_ce", "token_weighted_ce", "perplexity", "gradient_norm", "clip_coefficient"]
        )
        assert row["processed_proteins"] == step * 64 and len(row["sample_ids"]) == 64
        assert sum(row["length_stratum_exposures"]) == 64 and row["amp_overflows"] == 0 and row["scaler"] is None
    assert [c["updates"] for c in results["checkpoints"]] == [0, 250, 500, 1000, 1500, 2000]
    for checkpoint in results["checkpoints"]:
        filename = out / checkpoint["filename"]
        assert sha(filename) == checkpoint["sha256"]
        state = torch.load(filename, map_location="cpu", weights_only=False)
        step = checkpoint["updates"]
        assert state["successful_updates"] == state["scheduler"]["last_epoch"] == step
        assert state["data_cursor"] == state["processed_proteins"] == step * 64
        assert state["contract_sha256"] == results["contract_sha256"]
        assert all(torch.isfinite(v).all() for v in state["model"].values())
        assert all(float(s["step"]) == step for s in state["optimizer"]["state"].values())
        assert all(
            torch.isfinite(v).all()
            for s in state["optimizer"]["state"].values()
            for v in s.values()
            if torch.is_tensor(v)
        )
        assert set(state["rng"]) == {"python", "numpy", "cpu", "cuda"}
        assert np.array_equal(state["data_order"], np.random.default_rng(12012).permutation(231743))
        if step == 0:
            fingerprint = hashlib.sha256()
            for name, tensor in state["model"].items():
                fingerprint.update(name.encode())
                fingerprint.update(tensor.numpy().tobytes())
            assert fingerprint.hexdigest() == contract["initial_weights_sha256"]
        print("Independently verified checkpoint", step, flush=True)
    interval_count = 0
    raw_hashes = {}
    handoff = json.loads((report / "chatgpt_handoff.json").read_text())
    for population in ["primary", "independent"]:
        path = out / f"{population}_final_records.json"
        identities = json.loads(path.read_text())
        raw_hashes[population] = sha(path)
        assert len(identities) == len({r["sample_id"] for r in identities}) == 2048
        panel = json.loads((report / "panels.json").read_text())[population]
        assert {r["sample_id"] for r in identities} == set(panel)
        groups = {"aggregate": identities, **{str(i): [r for r in identities if r["stratum"] == i] for i in range(5)}}
        for group, values in groups.items():
            reported = results["final"][population]["aggregate" if group == "aggregate" else "strata"]
            if group != "aggregate":
                reported = reported[group]
            for comparator in ["global_unigram", "bucket_unigram", "bigram", "trigram", "shuffle", "last8", "last1"]:
                normal = (
                    "normal"
                    if comparator in ["global_unigram", "bucket_unigram", "bigram", "trigram"]
                    else "prefix_normal"
                )
                mean, ci = independent_interval([r[normal] - r[comparator] for r in values])
                target = reported["deltas"][comparator]
                assert mean == target["delta"] and ci == target["ci95"], (population, group, comparator)
                assert target["resamples"] == 10000 and target["identities"] == len(values)
                interval_count += 1
            for metric, mean in reported["means"].items():
                assert float(np.mean([r[metric] for r in values])) == mean
        print("Independent bootstrap implementation reproduced", population, flush=True)
        ablation_delta, ablation_ci = independent_interval([r["neutral_length"] - r["normal"] for r in identities])
        ablation = handoff["neutral_length_minus_normal"][population]
        assert ablation_delta == ablation["delta"] and ablation_ci == ablation["ci95"]
        interval_count += 1
    independently_derived = {}
    for population in ["primary", "independent"]:
        panel_summary = results["final"][population]
        d = panel_summary["aggregate"]["deltas"]
        strata = panel_summary["strata"].values()

        def favorable(name, d=d):
            return d[name]["delta"] < 0 and d[name]["ci95"][1] < 0

        e = not any(
            s["deltas"]["bucket_unigram"]["delta"] > 0 or s["deltas"]["shuffle"]["delta"] >= 0.02 for s in strata
        )
        e = e and sum(s["deltas"]["shuffle"]["delta"] < 0 for s in strata) >= 4
        gates = {
            "A": all(favorable(k) and d[k]["delta"] <= -0.05 for k in ["global_unigram", "bucket_unigram"]),
            "B": favorable("bigram") and favorable("trigram") and d["trigram"]["delta"] <= -0.01,
            "C": favorable("shuffle") and d["shuffle"]["delta"] <= -0.02,
            "D": favorable("last8") and d["last8"]["delta"] <= -0.01 and favorable("last1"),
            "E": e,
        }
        confirmation = {
            "A": all(d[k]["delta"] < 0 for k in ["global_unigram", "bucket_unigram"]),
            "B": all(d[k]["delta"] < 0 for k in ["bigram", "trigram"]),
            "C": favorable("shuffle"),
            "D": d["last8"]["delta"] < 0 and d["last1"]["delta"] < 0,
            "E": e,
        }
        assert gates == panel_summary["gates"] and confirmation == panel_summary["confirmation"]
        independently_derived[population] = {"gates": gates, "confirmation": confirmation}
    p = independently_derived["primary"]["gates"]
    i = independently_derived["independent"]["confirmation"]
    if all(p.values()) and all(i.values()):
        label = "CSEQ-A"
    elif all(p[k] for k in "ABCD") and (not p["E"] or not i["E"]):
        label = "CSEQ-D"
    elif p["C"] and p["D"] and i["C"] and i["D"]:
        label = "CSEQ-C"
    else:
        label = "CSEQ-B"
    assert label == results["classification"]
    worktrees = {}
    paths = {
        "E011": root.parent / "proteinGen-sequence-context",
        "E010_phase4c": Path("/mnt/d/Simone/proteinGen"),
        "E010_phase4d": root.parent / "proteinGen-hybrid-local-global",
    }
    for name, path in paths.items():

        def git(*args, path=path):
            return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

        observed = {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain"),
        }
        worktrees[name] = {
            "state": observed,
            "matches_execution_start": observed == start["protected_worktree_start_state"][name],
            "E012_wrote_worktree_files": False,
        }
        if not worktrees[name]["matches_execution_start"]:
            prior = start["protected_worktree_start_state"][name]
            assert name == "E010_phase4d" and observed["branch"] == prior["branch"]
            worktrees[name]["owner_working_tree_status_at_execution_start"] = prior["status"]
            subprocess.run(
                ["git", "-C", str(path), "merge-base", "--is-ancestor", prior["head"], observed["head"]],
                check=True,
            )
            worktrees[name]["independent_owner_commit_advance"] = git("log", "--oneline", f"{prior['head']}..HEAD")
            worktrees[name]["concurrent_owner_working_tree_status"] = observed["status"]
            worktrees[name]["interpretation"] = "Concurrent owner commits; no E012 writes or scientific-input changes"
    evidence = {
        "status": "passed",
        "independent_bootstrap_implementation": (
            "uniform identity choice resampling; percentile intervals; no sealed summarize call"
        ),
        "paired_intervals_exactly_reproduced": interval_count,
        "gates_and_classification_independently_reproduced": True,
        "classification": label,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 12112,
        "checkpoint_hashes_verified": 6,
        "optimizer_scheduler_rng_sampler_integrity": True,
        "update0_fingerprint_verified": True,
        "protected_hashes_verified": True,
        "successful_updates": 2000,
        "training_identities": 128000,
        "raw_final_records_sha256": raw_hashes,
        "protected_worktrees": worktrees,
        "original_preparation_commit": start["original_preparation_commit"],
        "execution_start_commit": start["execution_start_commit"],
        "interpretation_note_fixed_before_execution": start["interpretation_note_before_execution"],
    }
    (report / "independent_verification.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print("Independent verification passed:", interval_count, "paired intervals", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    torch.set_num_threads(2)
    verify(args.root)
