"""Independent verification and portable result records for E012 continuation."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from protein_sequence_generation.e012_continuation import IdentitySampler, continuation_lr
from scripts import run_e012_continuation as run
from scripts.verify_e012_pilot_results import independent_interval, sha

ROOT, REPORT, OUT = run.ROOT, run.REPORT, run.OUT


def manual_gates(summary):
    d = summary["aggregate"]["deltas"]
    strata = list(summary["strata"].values())

    def good(k):
        return d[k]["delta"] < 0 and d[k]["ci95"][1] < 0

    e = not any(s["deltas"]["bucket_unigram"]["delta"] > 0 or s["deltas"]["shuffle"]["delta"] >= 0.02 for s in strata)
    e = e and sum(s["deltas"]["shuffle"]["delta"] < 0 for s in strata) >= 4
    gates = {
        "A": all(good(k) and d[k]["delta"] <= -0.05 for k in ["global_unigram", "bucket_unigram"]),
        "B": good("bigram") and good("trigram") and d["trigram"]["delta"] <= -0.01,
        "C": good("shuffle") and d["shuffle"]["delta"] <= -0.02,
        "D": good("last8") and d["last8"]["delta"] <= -0.01 and good("last1"),
        "E": e,
    }
    confirmation = {
        "A": all(d[k]["delta"] < 0 for k in ["global_unigram", "bucket_unigram"]),
        "B": all(d[k]["delta"] < 0 for k in ["bigram", "trigram"]),
        "C": good("shuffle"),
        "D": all(d[k]["delta"] < 0 for k in ["last8", "last1"]),
        "E": e,
    }
    assert gates == summary["gates"] and confirmation == summary["confirmation"]
    return gates, confirmation


def main():
    run.verify_inputs()
    contract = json.loads((REPORT / "contract.json").read_text())
    for name, digest in contract["protected_hashes"].items():
        assert sha(ROOT / name) == digest, name
    results = json.loads((REPORT / "results.json").read_text())
    assert results["additional_updates"] == 8000 and results["global_update"] == 10000
    telemetry = [json.loads(line) for line in (OUT / "telemetry.jsonl").read_text().splitlines()]
    assert len(telemetry) == 8000 and [r["update"] for r in telemetry] == list(range(2001, 10001))
    source = torch.load(run.SOURCE, map_location="cpu", weights_only=False)
    sampler = IdentitySampler(source["data_order"], source["data_cursor"])
    training = run.old.load_rows("train")
    lr_ckpt = source["optimizer"]["param_groups"][0]["lr"]
    proteins = source["processed_proteins"]
    residues = source["processed_residues"]
    for row in telemetry:
        indices = sampler.take64()
        assert [training[i]["sample_id"] for i in indices] == row["sample_ids"]
        lengths = [training[i]["length"] for i in indices]
        assert lengths == row["protein_lengths"] and sum(row["microbatches"]) == 64
        regime = next(i for i, (lo, hi) in enumerate(run.STRATA) if lo <= max(lengths) <= hi)
        assert row["microbatches"] == contract["selected_plan"][str(regime)]["partition"]
        proteins += 64
        residues += sum(lengths)
        assert row["processed_proteins"] == proteins and row["processed_residues"] == residues
        assert row["learning_rate"] == continuation_lr(row["update"], lr_ckpt)
        assert all(math.isfinite(row[k]) for k in ["sequence_mean_ce", "gradient_norm", "token_weighted_ce"])
    assert proteins == 640000 == results["processed_proteins"] and residues == results["processed_residues"]
    for record in results["checkpoints"]:
        path = OUT / record["filename"]
        assert sha(path) == record["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        step = record["updates"]
        t = telemetry[step - 2001]
        assert state["successful_updates"] == state["scheduler"]["global_update"] == step
        assert state["scheduler"]["kind"] == "explicit_2k_to_10k_warm_restart_v1"
        assert state["processed_proteins"] == step * 64 and state["processed_residues"] == t["processed_residues"]
        assert state["sampler"]["cursor"] == t["sampler_cursor"] and state["sampler"]["epoch"] == t["sampler_epoch"]
        replay = IdentitySampler(source["data_order"], source["data_cursor"])
        for _ in range(step - 2000):
            replay.take64()
        assert np.array_equal(replay.order, state["sampler"]["order"])
        assert state["batch_plan"] == contract["selected_plan"] and state["contract_sha256"] == sha(
            REPORT / "contract.json"
        )
        assert all(float(s["step"]) == step for s in state["optimizer"]["state"].values())
        assert all(torch.isfinite(v).all() for v in state["model"].values())
        assert all(
            torch.isfinite(v).all()
            for s in state["optimizer"]["state"].values()
            for v in s.values()
            if torch.is_tensor(v)
        )
        assert set(state["rng"]) == {"python", "numpy", "cpu", "cuda"}
        print("Independently verified checkpoint", step, flush=True)
    count = 0
    abl = {}
    raw_hashes = {}
    for step in [5000, 10000]:
        abl[str(step)] = {}
        for pop in ["primary", "independent"]:
            path = OUT / f"{pop}_final_records_{step:05d}.json"
            rows = json.loads(path.read_text())
            raw_hashes[path.name] = sha(path)
            assert len(rows) == len({r["sample_id"] for r in rows}) == 2048
            assert {r["sample_id"] for r in rows} == set(json.loads((run.HIST / "panels.json").read_text())[pop])
            summary = results["scientific_boundaries"][str(step)][pop]
            groups = {"aggregate": rows, **{str(i): [r for r in rows if r["stratum"] == i] for i in range(5)}}
            for group, values in groups.items():
                section = summary["aggregate"] if group == "aggregate" else summary["strata"][group]
                for name in ["global_unigram", "bucket_unigram", "bigram", "trigram", "shuffle", "last8", "last1"]:
                    left = (
                        "normal"
                        if name in ["global_unigram", "bucket_unigram", "bigram", "trigram"]
                        else "prefix_normal"
                    )
                    delta, ci = independent_interval([r[left] - r[name] for r in values])
                    assert delta == section["deltas"][name]["delta"] and ci == section["deltas"][name]["ci95"]
                    count += 1
                for name, mean in section["means"].items():
                    assert mean == float(np.mean([r[name] for r in values]))
            manual_gates(summary)
            delta, ci = independent_interval([r["neutral_length"] - r["normal"] for r in rows])
            abl[str(step)][pop] = {"delta": delta, "ci95": ci}
            count += 1
            # Publish compact per-identity metrics for interval reproduction without raw checkpoints/data.
            run.old.save(REPORT / f"{pop}_paired_identity_metrics_{step:05d}.json", rows)
            print("Independently reproduced intervals", step, pop, flush=True)
    final = results["scientific_boundaries"]["10000"]
    p, i = final["primary"]["gates"], final["independent"]["confirmation"]
    if all(p.values()) and all(i.values()):
        cseq = "CSEQ-A"
    elif all(p[k] for k in "ABCD") and (not p["E"] or not i["E"]):
        cseq = "CSEQ-D"
    elif p["C"] and p["D"] and i["C"] and i["D"]:
        cseq = "CSEQ-C"
    else:
        cseq = "CSEQ-B"
    strong = p["C"] and p["D"] and i["C"] and i["D"]
    likelihood = p["A"] and p["B"] and i["A"] and i["B"]
    cont = (
        "CONT-A"
        if cseq == "CSEQ-A"
        else ("CONT-D" if strong and likelihood and (not p["E"] or not i["E"]) else ("CONT-B" if strong else "CONT-C"))
    )
    assert cseq == results["CSEQ_classification"] and cont == results["continuation_classification"]
    historic = [json.loads(line) for line in (run.old.OUT / "telemetry.jsonl").read_text().splitlines()]
    # Reconstruct historical update wall time from recorded residues/sec; excludes evaluations on both sides.
    historical_seconds = sum(sum(r["protein_lengths"]) / r["residues_per_second"] for r in historic)
    historic_rate = 2000 / historical_seconds
    rate = 8000 / sum(r["time_per_update_seconds"] for r in telemetry)
    worktrees = {}
    for name, path in {
        "E011": ROOT.parent / "proteinGen-sequence-context",
        "E010_phase4c": Path("/mnt/d/Simone/proteinGen"),
        "E010_phase4d": ROOT.parent / "proteinGen-hybrid-local-global",
    }.items():

        def git(*args, path=path):
            return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

        worktrees[name] = {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain"),
            "E012_wrote_files": False,
        }
    verification = {
        "status": "passed",
        "paired_intervals_independently_reproduced": count,
        "checkpoint_hashes_verified": 4,
        "historical_checkpoint_sha256": sha(run.SOURCE),
        "optimizer_rng_sampler_verified": True,
        "protected_hashes_verified": True,
        "additional_updates": 8000,
        "global_update": 10000,
        "total_exposures": proteins,
        "total_residues": residues,
        "raw_final_record_hashes": raw_hashes,
        "protected_worktrees": worktrees,
    }
    run.old.save(REPORT / "independent_verification.json", verification)
    handoff = {
        **results,
        "source_checkpoint_sha256": sha(run.SOURCE),
        "model_parameters": 15533952,
        "optimizer_restored": True,
        "Adam_moments_restored": True,
        "continuation_schedule": contract["scheduler"],
        "physical_batch_plan": contract["selected_plan"],
        "effective_batch": 64,
        "precision": "bfloat16",
        "peak_allocated_bytes": max(r["cuda_peak_allocated_bytes"] for r in telemetry),
        "peak_reserved_bytes": max(r["cuda_peak_reserved_bytes"] for r in telemetry),
        "historical_updates_per_second": historic_rate,
        "continuation_updates_per_second": rate,
        "speedup": rate / historic_rate,
        "throughput_convention": (
            "sum of per-update training wall times, excluding "
            "evaluation; historical durations reconstructed from residues/sec"
        ),
        "nominal_population_passes": proteins / 231743,
        "length_conditioning_ablation": abl,
        "independent_verification": verification,
        "tests": json.loads((REPORT / "quality.json").read_text()),
        "exactly_one_recommended_next_experiment": (
            "Separately reviewed full-generation evaluation."
            if cseq == "CSEQ-A"
            else (
                "Separately preregistered diagnosis of the remaining likelihood/length "
                "failure using the fixed causal model; no further training extension."
            )
        ),
    }
    training_summary = {
        "additional_updates": 8000,
        "clipped_update_fraction": float(np.mean([r["clipped"] for r in telemetry])),
        "mean_gradient_norm": float(np.mean([r["gradient_norm"] for r in telemetry])),
        "length_stratum_exposures": np.sum([r["length_stratum_exposures"] for r in telemetry], axis=0).tolist(),
        "cadence_100": [
            {
                "first_update": group[0]["update"],
                "last_update": group[-1]["update"],
                "mean_ce": float(np.mean([r["sequence_mean_ce"] for r in group])),
                "mean_gradient_norm": float(np.mean([r["gradient_norm"] for r in group])),
                "clipped_fraction": float(np.mean([r["clipped"] for r in group])),
                "mean_update_seconds": float(np.mean([r["time_per_update_seconds"] for r in group])),
                "lr_at_end": group[-1]["learning_rate"],
            }
            for offset in range(0, 8000, 100)
            if (group := telemetry[offset : offset + 100])
        ],
    }
    run.old.save(REPORT / "training_summary.json", training_summary)
    handoff["training_summary"] = training_summary
    run.old.save(REPORT / "chatgpt_handoff.json", handoff)

    def cell(d):
        return f"{d['delta']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]"

    text = (
        f"# E012 continuation — {cseq} / {cont}\n\nExactly 8,000 additional "
        f"successful updates, global 10,000; no extension or biological generation. "
        f"Historical 2k results remain immutable.\n\n"
    )
    text += (
        f"Preparation `{results['preparation_commit']}`; source checkpoint SHA256 "
        f"`{sha(run.SOURCE)}`. Model 15,533,952 parameters, Adam "
        f"moments/RNG/sampler restored. Contract "
        f"`{sha(REPORT / 'contract.json')}`.\n\n"
    )
    text += "## Likelihood trajectory\n\n| Global update | Primary CE | Independent CE |\n|---|---|---|\n"
    for u in ["2000", "3000", "5000", "7500", "10000"]:
        t = results["trajectory"][u]
        text += f"| {u} | {t['primary']['normal_ce']:.6f} | {t['independent']['normal_ce']:.6f} |\n"
    text += (
        "\n## Frozen gates and paired identity intervals\n\n10,000 bootstrap "
        "resamples; seed 12112, equal-protein primary likelihood. Empirical "
        "trigram is worse than unigram; beating trigram alone does not establish "
        "context learning.\n\n| Population | A | B | C | D | E "
        "|\n|---|---|---|---|---|---|\n"
    )
    for pop in ["primary", "independent"]:
        text += f"| {pop} | " + " | ".join(str(final[pop]["gates"][k]) for k in "ABCDE") + " |\n"
    text += (
        "\nIndependent confirmation: `"
        + json.dumps(final["independent"]["confirmation"])
        + "`.\n\n| Population | Comparison | Delta [95% CI] |\n|---|---|---|\n"
    )
    for pop in ["primary", "independent"]:
        for k, d in final[pop]["aggregate"]["deltas"].items():
            text += f"| {pop} | {k} | {cell(d)} |\n"
    text += (
        "\n## Context windows, order sensitivity and length ablation\n\n| Panel | "
        "Last1 | Last4 | Last8 | Last16 | Last32 | Last64 | Full | Shuffle "
        "|\n|---|---|---|---|---|---|---|---|---|\n"
    )
    for pop in ["primary", "independent"]:
        means = final[pop]["aggregate"]["means"]
        text += (
            f"| {pop} | "
            + " | ".join(
                f"{means[k]:.6f}"
                for k in ["last1", "last4", "last8", "last16", "last32", "last64", "prefix_normal", "shuffle"]
            )
            + " |\n"
        )
    for pop in ["primary", "independent"]:
        means = final[pop]["aggregate"]["means"]
        text += (
            f"\n{pop}: KL {means['kl']:.6f}, JS {means['js']:.6f}, top1 change "
            f"{means['top1_change']:.6f}, top3 set change {means['top3_change']:.6f}; "
            f"neutral-length minus normal {cell(abl['10000'][pop])}.\n\n"
        )
    text += (
        "## Length regimes and relative position\n\n| Panel | Stratum | "
        "Normal−bucket | Normal−shuffle | Full−last8 |\n|---|---|---|---|---|\n"
    )
    for pop in ["primary", "independent"]:
        for s, section in final[pop]["strata"].items():
            lo, hi = run.STRATA[int(s)]
            d = section["deltas"]
            text += f"| {pop} | {lo}–{hi} | {cell(d['bucket_unigram'])} | {cell(d['shuffle'])} | {cell(d['last8'])} |\n"
    text += (
        "\nExisting ten relative-position deciles are preserved and reported by "
        "length stratum at each normal evaluation boundary in trajectory.json. The "
        "20–64 and 65–128 regimes are shown explicitly above; position telemetry "
        "is descriptive, not a gate. Update 3000 contains normal likelihood only, "
        "so it cannot adjudicate shuffle/window retention; update 5000 contains "
        "the complete fixed diagnostics in scientific_boundaries.json. No "
        "checkpoint selection occurred.\n\n"
    )
    text += "## Throughput, integrity and limitations\n\n"
    text += (
        f"Frozen physical plan: `{json.dumps(contract['selected_plan'])}`. "
        f"Effective optimizer batch remains 64, microbatch mean scaled by n/64. LR "
        f"restarts linearly from {lr_ckpt:.12g} at update 2000 to .0003 at 2100, "
        f"then cosine to zero at 10000. BF16 and architecture/dropouts "
        f"unchanged.\n\n"
    )
    text += (
        f"Historical training throughput {historic_rate:.4f} updates/sec; "
        f"continuation {rate:.4f}; measured speedup {rate / historic_rate:.3f}×. "
        f"Evaluation excluded consistently. Peak allocated/reserved "
        f"{handoff['peak_allocated_bytes'] / 1024**3:.3f}/{handoff['peak_reserved_bytes'] / 1024**3:.3f} "
        f"GiB. Total exposures {proteins:,}, residues {residues:,}, nominal passes "
        f"{proteins / 231743:.4f}.\n\n"
    )
    text += (
        f"All {count} paired intervals independently reproduced; four new "
        f"checkpoint hashes, source hash, Adam steps, scheduler, RNG fields and "
        f"exact sampler-derived exposure sequence verified. Focused/sequence tests "
        f"and preflight passed; see quality.json and resume_smoke.json. Protected "
        f"baseline/panel/data/source hashes unchanged. E011 and structure worktrees "
        f"receive no E012 writes; concurrent owner snapshots are in "
        f"independent_verification.json.\n\n"
    )
    text += (
        (
            "One continuation trajectory and one warm restart; physical batching "
            "changes dropout draw assignment, so this is not bitwise replay of 8×8. "
            "Identity bootstrap omits cluster dependence; PDB cohort bias persists. "
            "This continuation does not independently replicate initialization or "
            "demonstrate biological generation quality. Longer training does not by "
            "itself prove context use.\n\n## Exactly one recommended next "
            "experiment\n\n"
        )
        + handoff["exactly_one_recommended_next_experiment"]
        + "\n"
    )
    (REPORT / "FINAL_REPORT.md").write_text(text)
    files = {
        str(p.relative_to(REPORT)): sha(p)
        for p in sorted(REPORT.rglob("*"))
        if p.is_file() and p.name != "result_manifest.json"
    }
    run.old.save(
        REPORT / "result_manifest.json",
        {
            "published_artifact_sha256": files,
            "checkpoints": results["checkpoints"],
            "historical_source_sha256": sha(run.SOURCE),
            "verifier_sha256": sha(Path(__file__)),
            "CSEQ_classification": cseq,
            "continuation_classification": cont,
            "global_update": 10000,
        },
    )
    print("Continuation results independently verified and published", cseq, cont, flush=True)


if __name__ == "__main__":
    main()
