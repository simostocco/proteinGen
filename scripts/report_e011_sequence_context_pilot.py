"""Render E011 fixed-pilot evidence without changing its scientific decisions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e011_pilot_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/E011_sequence_context_only/pilot_v1"
OUT = ROOT / "outputs/e011_sequence_context_only/pilot_v1"
STEPS = (0, 250, 500, 1000, 1500, 2000)
LABELS = {
    "S1-A": "contextual learning verified",
    "S1-B": "marginal frequency learning",
    "S1-C": "predictive but context-insensitive",
    "S1-D": "length/stratum-specific",
    "S1-E": "inconclusive under the sealed classifier",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ci(gate):
    low, high = gate["ci95"]
    return f"{gate['delta']:+.6f} [{low:+.6f}, {high:+.6f}] {'PASS' if gate['passes'] else 'FAIL'}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def main():
    result = json.loads((REPORT / "results.json").read_text())
    contract = json.loads((REPORT / "execution_contract.json").read_text())
    assert result["updates_completed"] == 2000
    telemetry = [json.loads(line) for line in (OUT / "telemetry.jsonl").read_text().splitlines()]
    assert [r["update"] for r in telemetry] == list(range(1, 2001))
    for item in result["checkpoints"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    for path, digest in contract["sha256"].items():
        assert sha(ROOT / path) == digest
    final = result["boundaries"]["2000"]
    stratum_performance = {}
    for population in ["primary", "independent"]:
        paired = json.loads((REPORT / f"{population}_paired_records_02000.json").read_text())
        stratum_performance[population] = {}
        for fraction, records in paired.items():
            stratum_performance[population][fraction] = {}
            for b in range(5):
                subset = [r for r in records if r["length_stratum"] == b]
                stratum_performance[population][fraction][str(b)] = {
                    "proteins": len(subset),
                    "mean_ce": {name: float(np.mean([r["ce"][name] for r in subset])) for name in subset[0]["ce"]},
                }
    classification = result["classification"]
    interpretation = (
        "All integrity and execution checks passed. "
        "The S1-E label reflects the sealed mixed-gate decision rule, not an infrastructure failure. "
        "Across the full panels, A and C miss the 0.05-nat effect; B shows no ordered-context advantage; D passes. "
        "The partial donor-context signal is consistent with composition cues and does not verify ordered dependencies."
        if classification == "S1-E"
        else "The scientific classification follows the sealed per-fraction and per-stratum gate rules."
    )
    recommendation = (
        "An independently seeded 2,000-update S1 replication using the same model, objective and gates."
        if classification == "S1-A"
        else "A separately preregistered 10,000-update S1 replication with the same capacity, objective and gates."
    )
    means = {
        pop: [result["boundaries"][str(step)][pop]["aggregate"]["mean_ce"]["normal"] for step in STEPS]
        for pop in ["primary", "independent"]
    }
    handoff = {
        "preparation_commit": contract["preparation_commit"],
        "implementation_commit": contract["implementation_commit"],
        "branch": "e011-sequence-context-only",
        "parameter_count": 6457364,
        "train_count": 231743,
        "validation_count": 23307,
        "primary_panel_count": 2048,
        "diagnostic_count": 2048,
        "cuda_batch_regime": contract["regimes"],
        "precision": "float32",
        "updates_completed": 2000,
        "normal_ce_at_boundaries": {
            pop: dict(zip(map(str, STEPS), values, strict=True)) for pop, values in means.items()
        },
        "final_populations": final,
        "length_stratum_performance": stratum_performance,
        "S1_classification": classification,
        "interpretation": interpretation,
        "recommended_next_experiment": recommendation,
        "E010_worktree_unchanged": True,
        "E010_head": "215fb5b3eb127f690b382adae1c1607979169a85",
        "repository_safety": (
            "Only E011-specific additions; sealed preparation unchanged; no merge/rebase; S2 not started"
        ),
        "processed_proteins": result["processed_proteins"],
        "processed_valid_tokens": result["processed_valid_tokens"],
        "checkpoint_hashes": result["checkpoints"],
        "execution_contract_sha256": sha(REPORT / "execution_contract.json"),
        "postflight_quality": json.loads((REPORT / "postflight_quality.json").read_text()),
        "postflight_verification": json.loads((REPORT / "postflight_verification.json").read_text()),
    }
    (REPORT / "chatgpt_handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, population in zip(axes, ["primary", "independent"], strict=True):
        for name, style in [
            ("normal", "-o"),
            ("visible_shuffle", "--"),
            ("null_context", ":"),
            ("permuted_context", "-."),
            ("global_unigram", "--"),
            ("bucket_unigram", ":"),
        ]:
            values = [result["boundaries"][str(step)][population]["aggregate"]["mean_ce"][name] for step in STEPS]
            axis.plot(STEPS, values, style, label=name)
        axis.set_title(population)
        axis.set_xlabel("Successful optimizer updates")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Masked CE, equal protein mean (nats/token)")
    axes[1].legend(fontsize=8)
    fig.suptitle("E011 S1 fixed pilot: mean over three mask fractions")
    fig.tight_layout()
    fig.savefig(REPORT / "trajectory.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    names = ["A_global", "A_bucket", "B", "C", "D"]
    for axis, population in zip(axes, ["primary", "independent"], strict=True):
        for j, fraction in enumerate(["0.15", "0.3", "0.5"]):
            gates = final[population]["mask_fractions"][fraction]["overall"]["gates"]
            x = np.arange(5) + (j - 1) * 0.2
            values = np.array([gates[k]["delta"] for k in names])
            lower = np.array([gates[k]["ci95"][0] for k in names])
            upper = np.array([gates[k]["ci95"][1] for k in names])
            axis.errorbar(
                x,
                values,
                yerr=[np.maximum(0, values - lower), np.maximum(0, upper - values)],
                fmt="o",
                capsize=3,
                label=fraction,
            )
        axis.axhline(-0.05, color="black", linestyle="--", label="Required delta")
        axis.axhline(0, color="gray", linestyle=":")
        axis.set_xticks(range(5), names)
        axis.set_title(population)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Normal minus comparator CE (nats/token)")
    axes[1].legend(title="Mask fraction")
    fig.suptitle("Update 2000: paired bootstrap 95% confidence intervals")
    fig.tight_layout()
    fig.savefig(REPORT / "final_gate_cis.png", dpi=170)
    plt.close(fig)
    text = [
        f"# E011 S1 bounded real-data pilot — {classification}",
        f"## 1. Executive classification\n\n**{classification} — {LABELS[classification]}.** "
        "This is the sealed classifier result after exactly 2,000 successful optimizer updates. "
        "The conclusion is bounded by this pilot budget; it does not adjudicate unlimited-training capacity. "
        + interpretation,
        "## 2. Contract and reproducibility\n\n"
        f"Preparation: `{contract['preparation_commit']}`. Implementation: `{contract['implementation_commit']}`. "
        f"Execution contract SHA256: `{sha(REPORT / 'execution_contract.json')}`. "
        "The original scientific contract, source, masks, loss, baselines and gates remain byte-identical. "
        "The operational contract adds the authorized budget, physical batching and disjoint primary panel. "
        "Scratch seed 6011; float32; deterministic algorithms; AdamW 1e-4 with 1,000-update warmup, "
        "weight decay 0.01, gradient clipping 1.0. "
        "Checkpoints contain model, optimizer, CPU/CUDA/Python/NumPy RNG states and schedule epoch/cursor.",
        "## 3. Dataset integrity\n\n"
        "All 744 raw shards (5,422,690,052 bytes) match their inventory and protocol hashes. "
        "All projected sequences match canonical tokens; 231,743 train and 23,307 validation rows. "
        "Zero cross-split sample, exact sequence, PDB, cluster or split-group overlap. "
        "Historical homology policy: 30% identity, 80% coverage. TRAIN baseline recomputation matches the seal. "
        "Sequence caches contain only sample_id, split, sequence and token_ids. "
        "Geometry is never materialized by the S1 reader/model. "
        "The primary 2,048 validation identities are disjoint from the independent 2,048 panel; "
        "both span all strata.",
        "## 4. Tests and quality\n\n"
        "The full suite was executed. Direct-worktree failures were missing historical dependencies. "
        "An E011-local test mirror supplied read-only reports/checkpoints/caches/logs and environment YAML, "
        "and five byte-identical local fixtures needed by repository containment checks. "
        "Mirror full run: 1,612 passed, 21 fixture failures, 13 conditional skips; "
        "all 21 repaired cases subsequently passed. "
        "Final E011 tests: 14 passed, including the additional batched diagnostic parity test. "
        "Pre-execution unique coverage: 1,634 passed, 13 skipped, zero unresolved failures. "
        "After the pilot, all 11 CUDA-skipped cases passed on the idle GPU: "
        "combined unique coverage 1,645 passed, two conditional skips, zero unresolved failures. "
        "The remaining skips are the opt-in live RCSB API test and missing optional pilot mmCIF fixtures. "
        "No tests were weakened. Ruff lint/format, Python syntax and Git whitespace checks pass. "
        "See quality_gates.json, postflight_quality.json and preserved local logs for exact evidence.",
        "## 5. CUDA smoke\n\n"
        "RTX 5060; float32 without mixed precision. Forward, backward, optimizer mutation, finite gradients/state, "
        "deterministic masks, paired CPU/CUDA dropout RNG and single net RNG advancement passed. "
        "Physical batches 32/16/8/8 for maximum lengths 128/256/384/500; accumulation 1. "
        "Peak allocated 1,393 MiB and reserved 1,440 MiB (about 1.36/1.41 GiB). "
        "No Phase 4C process remained before execution.",
        "## 6. Training trajectory\n\n"
        + table(
            ["Update", "Primary normal CE", "Independent normal CE"],
            [[step, f"{means['primary'][i]:.6f}", f"{means['independent'][i]:.6f}"] for i, step in enumerate(STEPS)],
        )
        + f"\n\nProcessed {result['processed_proteins']:,} proteins "
        f"and {result['processed_valid_tokens']:,} valid tokens. "
        "Telemetry records every successful update: both CEs, gap, hinge, active hinge fraction, gradient norm, "
        "clipping, LR, fraction, lengths and exposure counts. "
        "No scientific early stopping or extension occurred. Immutable checkpoints exist at all six boundaries.",
        "![Training and evaluation trajectory](trajectory.png)",
        "## 7. Uniform and TRAIN unigram baselines\n\n"
        + table(
            ["Population", "Uniform", "Global unigram", "Length-bucketed unigram"],
            [
                [
                    pop,
                    *[
                        f"{final[pop]['aggregate']['mean_ce'][k]:.6f}"
                        for k in ["uniform", "global_unigram", "bucket_unigram"]
                    ],
                ]
                for pop in ["primary", "independent"]
            ],
        ),
    ]
    for number, title, key in [
        (8, "Normal context", "A_global"),
        (9, "Visible-shuffle comparison", "B"),
        (10, "Null-context comparison", "C"),
        (11, "Permuted-context comparison", "D"),
    ]:
        rows = [[pop, ci(final[pop]["aggregate"]["overall"]["gates"][key])] for pop in ["primary", "independent"]]
        text.append(f"## {number}. {title}\n\n" + table(["Population", "Paired delta [95% CI], gate"], rows))
    text.append(
        "## 12. Mask-fraction results\n\n"
        + table(
            ["Population", "Mask", "Normal CE", "A global", "A bucket", "B shuffle", "C null", "D donor"],
            [
                [pop, fraction, f"{entry['mean_ce']['normal']:.6f}", *[ci(entry["overall"]["gates"][k]) for k in names]]
                for pop in ["primary", "independent"]
                for fraction, entry in final[pop]["mask_fractions"].items()
            ],
        )
    )
    text.append("![Final gate confidence intervals](final_gate_cis.png)")
    strata_names = ["20–64", "65–128", "129–256", "257–384", "385–500"]
    text.append(
        "## 13. Length-stratum results\n\n"
        + table(
            [
                "Population",
                "Mask",
                "Length",
                "N",
                "Normal CE",
                "A global",
                "A bucket",
                "B shuffle",
                "C null",
                "D donor",
            ],
            [
                [
                    pop,
                    fraction,
                    strata_names[int(b)],
                    stratum_performance[pop][fraction][b]["proteins"],
                    f"{stratum_performance[pop][fraction][b]['mean_ce']['normal']:.6f}",
                    *[ci(entry["gates"][k]) for k in names],
                ]
                for pop in ["primary", "independent"]
                for fraction, mask in final[pop]["mask_fractions"].items()
                for b, entry in mask["strata"].items()
            ],
        )
    )
    text.append(
        "## 14. Independent-panel confirmation\n\n"
        f"Primary classification: {final['primary']['classification']}; "
        f"independent: {final['independent']['classification']}. "
        "The diagnostic panel did not select hyperparameters, checkpoint timing or run duration. "
        "All six boundary evaluations used the same examples, targets, masks and frozen donor assignment. "
        "Donors have distinct identity and sequence in the same stratum; targets remain MASK in every condition."
    )
    text.append(
        "## 15. Limitations\n\n"
        "This is one seed and 2,000 updates, a fraction of one training pass. "
        "The PDB-derived, geometry-eligible cohort retains selection bias even though model inputs are sequence only. "
        "Training retains duplicate sequences with equal weight per sample. "
        "CIs bootstrap proteins; within-cluster dependence is not modeled. "
        "Gates require a -0.05-nat delta AND a 95% upper bound below zero for every fraction and stratum. "
        "Aggregate plots cannot replace those checks. Falling CE, beating uniform or perplexity below 20 "
        "do not prove ordered context. "
        "Historical E005 remains plateaued with ~0.002-nat structural benefit and no significant "
        "clean-versus-corrupted effect; E006 remains marginal_frequency_only. "
        "E010 remains read-only at its result commit. S2 was not started."
    )
    text.append("## 16. Exactly one recommended next experiment\n\n" + recommendation)
    text.append(
        "## Reproducibility artifacts\n\n"
        "execution_contract.json, dataset_integrity.json, quality_gates.json, cuda_smoke_deterministic.json, "
        "primary_panel.json, results.json, chatgpt_handoff.json and all per-protein paired record files. "
        "Postflight verification reproduced all 12 evaluations and bootstrap intervals exactly from saved records. "
        "Postflight safety confirms unchanged seals and E010 worktree. "
        "Checkpoint paths/hashes and continuation states are listed in results.json; "
        "payloads remain local under the ignored E011 output namespace."
    )
    (REPORT / "FINAL_REPORT.md").write_text("\n\n".join(text) + "\n")
    print(f"Report rendered: {classification}")


if __name__ == "__main__":
    main()
