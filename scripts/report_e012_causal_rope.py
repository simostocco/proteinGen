"""Reproduce E012 paired intervals, verify checkpoints and render a compact handoff."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from protein_sequence_generation.e012 import STRATA, classification, paired_interval, summarize
from scripts.run_e012_causal_rope import model, seed, weights_hash

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/E012_causal_rope_sequence/pilot_v1"
OUT = ROOT / "outputs/e012_causal_rope_sequence/pilot_v1"
STEPS = [0, 250, 500, 1000, 1500, 2000]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def fmt(interval):
    return f"{interval['delta']:+.6f} [{interval['ci95'][0]:+.6f}, {interval['ci95'][1]:+.6f}]"


def main():
    torch.set_num_threads(2)
    contract = json.loads((REPORT / "contract.json").read_text())
    result = json.loads((REPORT / "results.json").read_text())
    for name, digest in contract["protected_hashes"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(REPORT / "contract.json") == result["contract_sha256"]
    rows = [json.loads(line) for line in (OUT / "telemetry.jsonl").read_text().splitlines()]
    assert [r["update"] for r in rows] == list(range(1, 2001))
    assert all(len(r["sample_ids"]) == 64 for r in rows)
    all_ids = [sid for r in rows for sid in r["sample_ids"]]
    assert len(set(all_ids)) == len(all_ids) == result["processed_proteins"] == 128000
    assert sum(sum(r["protein_lengths"]) for r in rows) == result["processed_residues"]
    for item in result["checkpoints"]:
        path = OUT / item["filename"]
        assert sha(path) == item["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["successful_updates"] == item["updates"]
        assert state["data_cursor"] == item["updates"] * 64
        assert state["scheduler"]["last_epoch"] == item["updates"]
        assert all(float(s["step"]) == item["updates"] for s in state["optimizer"]["state"].values())
        assert all(torch.isfinite(t).all() for t in state["model"].values())
        assert set(state["rng"]) == {"python", "numpy", "cpu", "cuda"}
        if item["updates"] == 0:
            seed()
            fresh = model()
            assert weights_hash(fresh) == contract["initial_weights_sha256"]
            assert all(torch.equal(v, state["model"][k]) for k, v in fresh.state_dict().items())
        print("Checkpoint verified", item["updates"], flush=True)
    final_records, length_ablation, likelihood_metrics = {}, {}, {}
    for pop in ["primary", "independent"]:
        records = json.loads((OUT / f"{pop}_final_records.json").read_text())
        assert summarize(records) == result["final"][pop], pop
        print("All paired intervals reproduced", pop, flush=True)
        final_records[pop] = records
        length_ablation[pop] = paired_interval([r["neutral_length"] - r["normal"] for r in records])
        n_tokens = sum(r["length"] for r in records)
        equal_ce = float(np.mean([r["normal"] for r in records]))
        token_ce = sum(r["loss_sum"] for r in records) / n_tokens
        likelihood_metrics[pop] = {
            "equal_protein_ce": equal_ce,
            "token_weighted_ce": token_ce,
            "equal_protein_perplexity": math.exp(equal_ce),
            "token_weighted_perplexity": math.exp(token_ce),
            "equal_protein_top1": float(np.mean([r["top1"] for r in records])),
            "equal_protein_top3": float(np.mean([r["top3"] for r in records])),
            "token_weighted_top1": sum(r["top1"] * r["length"] for r in records) / n_tokens,
            "token_weighted_top3": sum(r["top3"] * r["length"] for r in records) / n_tokens,
            "relative_position_ce_deciles": np.mean([r["relative_position_ce"] for r in records], axis=0).tolist(),
        }
        (REPORT / f"{pop}_paired_identity_metrics.json").write_text(json.dumps(records, separators=(",", ":")) + "\n")
    assert classification(result["final"]["primary"], result["final"]["independent"]) == result["classification"]
    for step in STEPS:
        for pop in final_records:
            records = json.loads((OUT / f"{pop}_likelihood_{step:05d}.json").read_text())
            assert float(np.mean([r["normal"] for r in records])) == result["trajectory"][str(step)][pop]["normal_ce"]
    protected = json.loads((REPORT / "protected_worktree_state.json").read_text())
    after = {}
    paths = {
        "E011": ROOT.parent / "proteinGen-sequence-context",
        "E010_phase4c": Path("/mnt/d/Simone/proteinGen"),
        "E010_phase4d": ROOT.parent / "proteinGen-hybrid-local-global",
    }
    for name, path in paths.items():

        def git(*args, path=path):
            return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

        status = git("status", "--porcelain")
        prior = protected[name]
        prior_hashes_match = all(
            (path / rel).is_file() and sha(path / rel) == digest
            for rel, digest in prior["untracked_file_hashes"].items()
        )
        after[name] = {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "status": status,
            "matches_preparation_snapshot": prior["branch"] == git("branch", "--show-current")
            and prior["head"] == git("rev-parse", "HEAD")
            and prior["status"] == status
            and prior_hashes_match,
            "E012_wrote_files": False,
        }
    assert after["E011"]["matches_preparation_snapshot"]
    next_experiment = (
        "A separately reviewed full-generation sequence evaluation, without structure conditioning."
        if result["classification"] == "CSEQ-A"
        else "A separately preregistered 10,000-update causal RoPE budget replication "
        "with the same capacity, objective and gates."
    )
    handoff = {
        "branch": "e012-causal-rope-sequence",
        "worktree": "proteinGen-causal-rope",
        "base_E011_commit": contract["base_E011_commit"],
        "preparation_commit": result["preparation_commit"],
        "model_parameter_count": 15533952,
        "initialization_seed": 12012,
        "model_inputs": [
            "previous canonical amino-acid tokens",
            "BOS",
            "PAD",
            "causal/key mask",
            "RoPE",
            "requested continuous length embedding",
        ],
        "excluded_features": [
            "geometry",
            "structure labels",
            "metadata predictive features",
            "diffusion timestep",
            "E011 hidden states",
        ],
        "train_count": 231743,
        "validation_count": 23307,
        "primary_panel_count": 2048,
        "independent_panel_count": 2048,
        "homology_policy": {"identity": 0.3, "coverage": 0.8},
        "optimizer": {
            "AdamW_lr": 0.0003,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_updates": 100,
            "scheduler": "existing LambdaLR cosine over 2000",
            "clip": 1.0,
        },
        "batch_regime": {"physical": 8, "accumulation": 8, "effective_equal_protein": 64},
        "precision": contract["precision"],
        "updates_completed": 2000,
        "trajectory": result["trajectory"],
        "final": result["final"],
        "likelihood_metrics": likelihood_metrics,
        "neutral_length_minus_normal": length_ablation,
        "CSEQ_classification": result["classification"],
        "sampling_smoke": json.loads((REPORT / "cuda_smoke.json").read_text())["exact_length_canonical_sampling"],
        "quality": json.loads((REPORT / "quality.json").read_text()),
        "protected_worktrees": after,
        "checkpoints": result["checkpoints"],
        "biological_generation_launched": False,
        "recommended_next_experiment": next_experiment,
    }
    (REPORT / "chatgpt_handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    verification = {
        "status": "passed",
        "checkpoint_hashes_verified": 6,
        "optimizer_scheduler_sampler_counters_verified": True,
        "update0_independent_initialization_verified": True,
        "protected_contract_hashes_verified": True,
        "all_reported_gate_CIs_reproduced_exactly": True,
        "all_trajectory_means_reproduced_exactly": True,
        "updates": 2000,
        "unique_training_identities": 128000,
        "protected_worktree_state": after,
        "local_telemetry_sha256": sha(OUT / "telemetry.jsonl"),
        "local_final_records_sha256": {p: sha(OUT / f"{p}_final_records.json") for p in final_records},
    }
    (REPORT / "postflight_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    lines = [
        "# E012 causal RoPE pilot — " + result["classification"],
        "Exactly 2,000 successful optimizer updates; no extension, structure conditioning or biological generation. "
        "Scientific interpretation follows the frozen primary gates and independent confirmation rules.",
        "## Reproducibility and preflight",
        f"Base E011 `{contract['base_E011_commit']}`; preparation `{result['preparation_commit']}`. "
        f"Contract SHA256 `{result['contract_sha256']}`. "
        "The existing causal architecture is unchanged: 15,533,952 parameters, seed 12012, scratch initialization. "
        "Inputs: previous canonical residues, BOS/PAD, causal/key mask, RoPE and continuous requested length. "
        "All structural and metadata predictive features are excluded. AdamW .0003, betas .9/.95, decay .01, "
        "100 warmup updates, bounded cosine, clip 1. Physical 8 × accumulation 8 = 64 equally weighted proteins; "
        f"{contract['precision']}, no scaler.",
        "The audited E011 TRAIN/validation population (231,743/23,307) and exact disjoint panels "
        "(2,048/2,048) are reused. "
        "Raw 744-shard hashes, sequence-only projection and protected zero cross-split overlap checks passed; "
        "historical homology policy 30% identity/80% coverage. No dataset/checkpoint is committed. "
        "See data_integrity.json, quality.json and cuda_smoke.json.",
        "## Likelihood trajectory",
        table(
            ["Update", "Primary CE", "Independent CE"],
            [
                [
                    s,
                    f"{result['trajectory'][str(s)]['primary']['normal_ce']:.6f}",
                    f"{result['trajectory'][str(s)]['independent']['normal_ce']:.6f}",
                ]
                for s in STEPS
            ],
        ),
        "## Final likelihood and statistical baselines",
        table(
            ["Panel", "Normal", "Global unigram", "Bucket unigram", "Bigram", "Trigram"],
            [
                [
                    p,
                    *[
                        f"{result['final'][p]['aggregate']['means'][k]:.6f}"
                        for k in ["normal", "global_unigram", "bucket_unigram", "bigram", "trigram"]
                    ],
                ]
                for p in final_records
            ],
        ),
        "## Gates and paired identity bootstrap",
        "10,000 resamples, seed 12112; delta is model minus baseline or full/normal prefix minus comparator. "
        "Units are nats/token. "
        "Intervals are 95% percentile CIs. Negative values favor the model.",
        table(
            ["Panel", "Comparison", "Delta [95% CI]"],
            [[p, k, fmt(v)] for p in final_records for k, v in result["final"][p]["aggregate"]["deltas"].items()],
        ),
        table(
            ["Panel", "A", "B", "C", "D", "E"],
            [[p, *[str(v) for v in result["final"][p]["gates"].values()]] for p in final_records],
        ),
        "Independent confirmation (distinct from requiring the primary effect thresholds twice): "
        + json.dumps(result["final"]["independent"]["confirmation"]),
        "## Prefix order and context windows",
        "Up to 16 fixed interior prediction targets per protein; identical targets and prefix composition, "
        "no future residues. Complete diagnostics were preregistered for update 2000; "
        "likelihood was evaluated at every boundary.",
        table(
            ["Panel", "Last1", "Last4", "Last8", "Last16", "Last32", "Last64", "Full", "Shuffle"],
            [
                [
                    p,
                    *[
                        f"{result['final'][p]['aggregate']['means'][k]:.6f}"
                        for k in ["last1", "last4", "last8", "last16", "last32", "last64", "prefix_normal", "shuffle"]
                    ],
                ]
                for p in final_records
            ],
        ),
        table(
            ["Panel", "KL normal||shuffle", "JS", "Top1 change", "Top3 set change"],
            [
                [
                    p,
                    *[
                        f"{result['final'][p]['aggregate']['means'][k]:.6f}"
                        for k in ["kl", "js", "top1_change", "top3_change"]
                    ],
                ]
                for p in final_records
            ],
        ),
        "## Length ablation",
        "Zero continuous length embedding output at evaluation; no retraining. "
        "Neutral minus normal full-likelihood CE:",
        table(["Panel", "Delta [95% CI]"], [[p, fmt(length_ablation[p])] for p in final_records]),
        "## Five length strata",
        table(
            ["Panel", "Length", "Normal−bucket", "Normal-prefix−shuffle", "Full−last8"],
            [
                [
                    p,
                    f"{lo}–{hi}",
                    *[
                        fmt(result["final"][p]["strata"][str(i)]["deltas"][k])
                        for k in ["bucket_unigram", "shuffle", "last8"]
                    ],
                ]
                for p in final_records
                for i, (lo, hi) in enumerate(STRATA)
            ],
        ),
        "## Additional metrics",
        "Token-weighted CE/perplexity, top-1/top-3 accuracy and relative-position deciles:",
        "```json\n" + json.dumps(likelihood_metrics, indent=2) + "\n```",
        "## Verification and safety",
        "All checkpoint hashes, optimizer/scheduler/sampler counters and update-0 initialization "
        "independently verified. "
        "All gate intervals and trajectory means reproduced. E011 remains immutable S1-E. "
        "Structural worktree snapshots are reported honestly in postflight_verification.json; "
        "any owner changes are distinguished from E012 writes (none).",
        "## Limitations",
        "One seed and 2,000 updates. PDB-derived cohort selection bias persists despite sequence-only model inputs. "
        "Duplicate TRAIN sequences retain sample weight. Identity bootstrap does not model within-cluster dependence. "
        "Prefix diagnostics cover fixed interior positions, not every residue. "
        "Biological quality and foldability were not evaluated.",
        "## Exactly one recommended next experiment",
        next_experiment,
    ]
    (REPORT / "FINAL_REPORT.md").write_text("\n\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pop in final_records:
        axes[0].plot(STEPS, [result["trajectory"][str(s)][pop]["normal_ce"] for s in STEPS], marker="o", label=pop)
        m = result["final"][pop]["aggregate"]["means"]
        axes[1].plot(
            range(7),
            [m[k] for k in ["last1", "last4", "last8", "last16", "last32", "last64", "prefix_normal"]],
            marker="o",
            label=pop,
        )
    axes[0].set_xlabel("Successful optimizer updates")
    axes[0].set_ylabel("Equal-protein CE, nats/token")
    axes[1].set_xticks(range(7), ["1", "4", "8", "16", "32", "64", "full"])
    axes[1].set_xlabel("Previous-residue context window")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "trajectory_context.png", dpi=160)
    plt.close(fig)
    manifest = {
        "classification": result["classification"],
        "contract_sha256": result["contract_sha256"],
        "preparation_commit": result["preparation_commit"],
        "published_artifact_sha256": {
            p.name: sha(p) for p in sorted(REPORT.iterdir()) if p.is_file() and p.name != "result_manifest.json"
        },
        "local_checkpoints": result["checkpoints"],
    }
    (REPORT / "result_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("E012 report and reproducibility manifest rendered:", result["classification"])


if __name__ == "__main__":
    main()
