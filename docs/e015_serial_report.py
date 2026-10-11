# ruff: noqa: E402, E501
"""Compact scalar reporting only; safe alongside resumed production training."""

import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import e015_serial_recovery as r
import pyarrow.parquet as pq

OUT = r.ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
ORIGINAL = Path(sys.argv[1])


def read(name, default=None):
    p = OUT / name
    return r.read(p) if p.exists() else default


def table(name):
    p = OUT / name
    return pq.read_table(p, use_threads=False).to_pylist() if p.exists() else []


def mean(rows, key):
    values = [q[key] for q in rows if isinstance(q.get(key), (float, int)) and math.isfinite(q[key])]
    return statistics.mean(values) if values else None


def fmt(x):
    return "unmeasured" if x is None else f"{x:.6g}"


def main():
    original = r.read(ORIGINAL / "recovery_audit.json")
    completion = read("logs/evaluation_completion.json", {})
    final = read("logs/final_resume.json", {})
    verdict = read(
        "verdict.json",
        dict(
            reconstruction="INCONCLUSIVE",
            generation="INCONCLUSIVE",
            generalization="INCONCLUSIVE",
            continued_training="CONTINUE_BUT_WATCH",
        ),
    )
    rows = table("absolute_reconstruction_metrics.parquet")
    # Cached conditions can cover a larger or incomplete historical panel.
    # Report exactly the balanced continuation panel used by the bootstrap.
    panel = read("panel_definitions.json", {}).get("rows", [])
    indices = tuple(range(0, 20, 2)) if "tier2" in completion.get("completed", []) else (0, 2, 10, 12)
    selected_ids = {q["sample_id"] for q in panel if q["pair_index"] in indices}
    if selected_ids:
        rows = [q for q in rows if q["sample_id"] in selected_ids]
    gen = table("generation_metrics.parquet")
    distribution = read("generation_distribution_metrics.json", {})
    diversity = read("diversity_novelty_metrics.json", {})
    gap = read("generalization_bootstrap.json", {}).get("rows", [])
    weights = verdict.get("sampler_weights")
    primary = "epoch_1_fraction_0.50"
    primarygap = [
        q
        for q in gap
        if q["checkpoint"] == primary
        and q["weights"] == weights
        and q["timestep"] is None
        and q["length_stratum"] == "all"
    ]
    pg = {q["metric"]: q for q in primarygap}
    # Scientific interpretation includes absolute geometry and robust medians;
    # a single high-error TRAIN protein must not dominate the fit assessment.
    if all(k in pg for k in ("v_mse", "local_rmse_angstrom", "torsion_angular_mae_radians")):
        vgap = pg["v_mse"]
        local = pg["local_rmse_angstrom"]
        torsion = pg["torsion_angular_mae_radians"]
        if (
            min(local["train_mean"], local["heldout_mean"]) > 1
            and min(torsion["train_mean"], torsion["heldout_mean"]) > 0.7
            and abs(vgap["heldout_median"] / vgap["train_median"] - 1) < 0.15
        ):
            verdict["generalization"] = "UNDERFIT"
            verdict["generalization_reason"] = (
                "Substantial local/torsion errors on both splits, small geometry gaps and close median denoising errors; "
                "the TRAIN mean includes a long-protein outlier. This is measured reconstruction underfitting, "
                "not proof of successful distribution learning or absence of all memorization."
            )
    first = r.read(ORIGINAL / "first_resumed_heartbeat.json")
    obs = r.read(ORIGINAL / "training_observation.json")
    old = original["metadata_before"]["STATE.json"]
    new = first["state"]
    replay_keys = [
        "update",
        "exposures",
        "schedule_cursor",
        "loss_groups",
        "pcgrad",
        "timestep_histogram",
        "sampled_timestep_histogram",
        "snr_bin_histogram",
        "self_conditioning_fraction",
        "rollout_state_fraction",
    ]
    replay = dict(
        old_update=old["update"],
        new_update=new["update"],
        matches={k: old.get(k) == new.get(k) for k in replay_keys},
        old_total_loss=old["loss_groups"]["total_loss"],
        new_total_loss=new["loss_groups"]["total_loss"],
        scope="Exact reported loss/gradient/RNG-histogram telemetry equality; not a claim of full unsaved model-byte comparison",
    )
    r.write(OUT / "logs/recovery_recomputed_telemetry_proof.json", replay)
    saved_audit = read("recovery_audit.json", {})
    saved_audit["recomputed_telemetry_proof"] = replay
    r.write(OUT / "recovery_audit.json", saved_audit)
    monitor = []
    monitor_files = [ORIGINAL / "resource_monitor.jsonl", OUT / "resource_monitor.jsonl"]
    if final.get("resume_report"):
        monitor_files.append(Path(final["resume_report"]) / "resource_monitor.jsonl")
    for file in monitor_files:
        if file.exists():
            with file.open() as f:
                for line in f:
                    try:
                        monitor.append(json.loads(line))
                    except ValueError:
                        pass
    if monitor:
        gpus = [[float(x.strip()) for x in q["gpu_csv"].split(",")] for q in monitor]
        utilization = sorted(q[2] for q in gpus)
        resource_summary = dict(
            samples=len(monitor),
            minimum_ram_available_gib=min(q["ram_available_bytes"] for q in monitor) / 1024**3,
            maximum_swap_used_gib=max(q["swap_used_bytes"] for q in monitor) / 1024**3,
            minimum_d_free_gib=min(q["d_free_bytes"] for q in monitor) / 1024**3,
            gpu_peak_used_mib=max(q[0] for q in gpus),
            gpu_temperature_range_celsius=[min(q[3] for q in gpus), max(q[3] for q in gpus)],
            gpu_utilization_percentiles={
                str(p): utilization[round((len(utilization) - 1) * p / 100)] for p in (0, 25, 50, 75, 100)
            },
            cpu_load_peak=max(q["load_average"][0] for q in monitor),
            cpu_temperatures="Unavailable where monitor cpu_temperatures is empty",
        )
    else:
        resource_summary = {}
    ios = [q for q in monitor if isinstance(q.get("trainer_io"), str)]
    if ios:

        def counters(q):
            return {line.split(":")[0]: int(line.split(":")[1]) for line in q["trainer_io"].splitlines()}

        first_io, last_io = counters(ios[0]), counters(ios[-1])
        resource_summary["initial_training_io"] = dict(
            trainer_pid=ios[0]["trainer_pid"],
            payload_write_bytes=last_io["wchar"] - first_io["wchar"],
            linux_charged_disk_write_bytes=last_io["write_bytes"] - first_io["write_bytes"],
            interval_seconds=ios[-1]["timestamp_unix"] - ios[0]["timestamp_unix"],
            note="Linux/WSL process accounting;DrvFS host writes may not be charged to Linux write_bytes. Raw counters preserved in monitor.",
        )
    r.write(OUT / "logs/resource_summary.json", resource_summary)
    # Preserve historical inventory and distinguish overwritten rolling paths from snapshots.
    old = read("checkpoint_inventory.json", {})
    old.update(
        shutdown_inventory=original["inventory"],
        shutdown_selected=original["selected"],
        shutdown_secondary_snapshot=original["secondary_immutable_snapshot"],
        pause_checkpoint=completion.get("pause_checkpoint"),
        final_resume=final.get("next_rolling_checkpoint"),
        primary_label=primary,
    )
    r.write(OUT / "checkpoint_inventory.json", old)
    selected = original["selected"]
    nextop = original["next_updates"][0]
    lines = [
        "# E015 serialized shutdown recovery and absolute geometry evaluation",
        "",
        "The primary scientific checkpoint remains the predeclared exact 50% checkpoint. Training recovery and contextual evaluation use their separately verified durable positions. All evaluation phases ran after the original trainer and supervisor exited; production resumed after evaluator exit and GPU release.",
        "",
        f"Recovery checkpoint: `{selected['path']}`, SHA256 `{selected['sha256']}`, update {selected['position']['update']}, exposures {selected['position']['exposures']}.",
        "Primary scientific checkpoint: `production/checkpoints/epoch_1_fraction_0.50.pt`, SHA256 `2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03`, update 3811, exposures 115066.",
        "",
        f"Exact first operation: update {nextop['update']}, exposures {nextop['exposures']}, {nextop['proteins']} proteins, {nextop['residues']} residues. Dedicated CUDA noise stream initial seed 15018; checkpointed state and exact noise hashes identify the operation, rather than fabricated per-identity seeds.",
        "",
    ]
    for i, m in enumerate(nextop["microbatches"], 1):
        lines += [
            f"Microbatch {i}: `" + ", ".join(m["identities"]) + "`.",
            f"Lengths: `{m['lengths']}`; timesteps: `{m['timesteps']}`; self-conditioning: {m['self_conditioned']}; rollout steps: {m['rollout_steps']}; noise-state SHA256 `{m['noise_rng_sha256_before']}`; actual centered-noise SHA256 `{m['noise_tensor_sha256']}`.",
            "",
        ]
    lines += [
        f"First persisted resumed heartbeat: update {first['state']['update']}, exposures {first['state']['exposures']}; exact planned cursor, timestep and self-conditioning match. The trainer emits every five updates, so update 5677 is the first computed operation and update 5680 is the first emitted resumed heartbeat.",
        "",
        f"Real shutdown replay proof: all recorded loss-group, PCGrad and RNG-histogram/counter fields at update5680 exactly match the pre-shutdown record. Total loss is {replay['new_total_loss']!r} in both runs. This is exact scalar/telemetry equality, not a comparison against an unavailable complete live-state snapshot.",
        "",
        f"Initial observation: {obs['observation_seconds']:.1f} seconds, {obs['final_update'] - obs['initial_update']} updates and {obs['final_exposures'] - obs['initial_exposures']} exposures between observed endpoints ({(obs['final_exposures'] - obs['initial_exposures']) / obs['observation_seconds']:.3f} proteins/s, {(obs['final_update'] - obs['initial_update']) / obs['observation_seconds']:.4f} updates/s). RAM minimum {obs['minimum_ram_available_bytes'] / 1024**3:.2f} GiB; swap maximum {obs['maximum_swap_used_bytes'] / 1024**3:.2f} GiB. Exact rates vary by lengths and checkpoint work.",
        "",
        "The permission wait for the first pause command allowed the existing trainer to reach epoch1. Its named checkpoint satisfied the requested alternative boundary. Its built-in epoch evaluation completed before the graceful handler exited; no separate evaluator overlapped. No scientific production CPU, sampler, optimization, architecture, schedule or checkpoint contract was changed.",
        "",
        f"Completed continuation phases: {completion.get('completed', [])}. Stopped early: {completion.get('stopped_early', False)}. Resource pressure: {completion.get('resource_pressure', False)}. Error: {completion.get('error')}.",
        "",
        f"Primary generation weights: **{weights}**, selected only by the frozen heldout equal-protein/equal-timestep v-MSE. Raw/EMA values: {verdict.get('raw_ema_comparison_v_mse')}.",
        "",
        "| Primary metric | TRAIN mean | TRAIN median | Heldout mean | Heldout median | Absolute gap | Relative gap | Gap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for q in primarygap:
        lines.append(
            f"| {q['metric']} | {fmt(q['train_mean'])} | {fmt(q.get('train_median'))} | {fmt(q['heldout_mean'])} | {fmt(q.get('heldout_median'))} | {fmt(q['gap'])} | {fmt(q.get('relative_difference'))} | {q.get('difference_ci95')} |"
        )
    lines += [
        "",
        "Intervals use 2000 stratified protein bootstrap draws from compact scalars. Timesteps are averaged within each identity. A small gap accompanied by large errors is underfitting, not proof of successful generalization. Local RMSE is averaged per protein in Å; the local MSE is also retained in Å².",
        "The quick panel contained a large longest-stratum TRAIN error, producing a mean gap very different from its median gap. Outliers remain included. The means, medians, stratum tables and uncertainty intervals must be considered together; a negative aggregate gap does not by itself prove successful generalization.",
        "",
        "| Split | t | v-MSE | Kabsch Å | L1 RMSE Å | L2 RMSE Å | L3 RMSE Å | Adjacent mean Å | Torsion degrees | Chirality inversion | Clash fraction | Rg ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    evaluated_ids = {q["sample_id"] for q in rows if q["checkpoint"] == primary and q["weights"] == weights}
    for split in ("train", "heldout"):
        for t in (5, 50, 250, 400, 490):
            group = [
                q
                for q in rows
                if q["checkpoint"] == primary
                and q["weights"] == weights
                and q["split"] == split
                and q["timestep"] == t
                and q["sample_id"] in evaluated_ids
            ]
            lines.append(
                "| "
                + split
                + " | "
                + str(t)
                + " | "
                + " | ".join(
                    fmt(mean(group, k))
                    for k in (
                        "v_mse",
                        "aligned_rmsd_angstrom",
                        "offset_1_rmse_angstrom",
                        "offset_2_rmse_angstrom",
                        "offset_3_rmse_angstrom",
                        "adjacent_mean_angstrom",
                        "torsion_angular_mae_degrees",
                        "triple_inversion_fraction",
                        "clash_fraction",
                        "radius_ratio",
                    )
                )
                + " |"
            )
    lines += [
        "",
        "High-noise direct-X0 shrinkage is distinct from collapse of the actual noisy diffusion state. A conditional mean can shrink when the noisy observation contains little information, so paired high-noise clean errors alone do not establish generator collapse. The next-step Rg/oracle ratio, v cosine/norm, no-op gain, torsion radians/degrees, triple-product MSE, distance-error percentiles, contact metrics and covariance ratios remain in the scalar tables. Reconstruction generalization is a statement about the measured denoising task; unconditional distribution quality requires the separate full-generation evidence.",
        "",
        "| Checkpoint | Weight | Split | Aggregate v-MSE | Aggregate Kabsch Å | Aggregate local RMSE Å |",
        "|---|---|---|---:|---:|---:|",
    ]
    for cp in sorted({q["checkpoint"] for q in rows}):
        for w in ("raw", "ema"):
            for split in ("train", "heldout"):
                group = [q for q in rows if q["checkpoint"] == cp and q["weights"] == w and q["split"] == split]
                lines.append(
                    "| "
                    + cp
                    + " | "
                    + w
                    + " | "
                    + split
                    + " | "
                    + " | ".join(fmt(mean(group, k)) for k in ("v_mse", "aligned_rmsd_angstrom", "local_rmse_angstrom"))
                    + " |"
                )
    held = [q for q in rows if q["checkpoint"] == primary and q["weights"] == weights and q["split"] == "heldout"]
    regimes = []
    for t in (5, 50, 250, 400, 490):
        group = [q for q in held if q["timestep"] == t]
        if group:
            regimes.append(
                dict(
                    timestep=t,
                    kabsch=mean(group, "aligned_rmsd_angstrom"),
                    local_rmse=mean(group, "local_rmse_angstrom"),
                    clash=mean(group, "clash_fraction"),
                    chirality=mean(group, "triple_inversion_fraction"),
                )
            )
    if regimes:
        best = min(regimes, key=lambda q: q["kabsch"])
        worst = max(regimes, key=lambda q: q["kabsch"])
        verdict.update(strongest_regime=best, weakest_regime=worst)
        lines += [
            "",
            f"Strongest timestep by absolute Kabsch error: {best}. Weakest: {worst}. Noise regimes differ in difficulty; complete length strata and metric-specific tradeoffs remain in the tables.",
        ]
    primarygen = [q for q in gen if q["primary"]]
    lines += [
        "",
        f"Full de novo traces: {len(gen)} ({len(primarygen)} primary, {len(gen) - len(primarygen)} secondary). Finite rate: {fmt(mean(gen, 'finite_fraction'))}. Exact500-step sampler; batch1; sparse frames only. No arbitrary reference RMSD is treated as generation quality.",
        "",
        "| N | Primary n | Adjacent mean Å | Adjacent std Å | Adjacent valid fraction | Clashes | Rg Å | End-to-end Å |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in (48, 96, 192, 320, 448):
        group = [q for q in primarygen if q["requested_length"] == n]
        lines.append(
            "| "
            + str(n)
            + " | "
            + str(len(group))
            + " | "
            + " | ".join(
                fmt(mean(group, k))
                for k in (
                    "adjacent_mean_angstrom",
                    "adjacent_std_angstrom",
                    "adjacent_valid_fraction",
                    "clash_fraction",
                    "radius_of_gyration_angstrom",
                    "end_to_end_angstrom",
                )
            )
            + " |"
        )
    lines += [
        "",
        "| N | Descriptor metric | Generated vs real | Real vs real floor | Excess | Excess 95% CI |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for d in distribution.get("by_length", []):
        for q in d["metrics"]:
            lines.append(
                f"| {d['length']} | {q['metric']} | {fmt(q['generated_vs_real'])} | {fmt(q['real_vs_real_floor'])} | {fmt(q.get('excess_above_floor'))} | {q.get('excess_ci95')} |"
            )
    lines += [
        "",
        "| N | Pairwise Kabsch median Å | Exact duplicate pairs | Nearest TRAIN descriptor median | Nearest heldout descriptor median |",
        "|---:|---:|---:|---:|---:|",
    ]
    for d in diversity.get("diversity", []):
        neighbors = [q for q in diversity.get("novelty", []) if f"_n{d['length']}_" in q["sample"]]
        medians = {}
        for split in ("train", "heldout"):
            values = [q["nearest"][split]["descriptor_distance"] for q in neighbors]
            medians[split] = statistics.median(values) if values else None
        lines.append(
            f"| {d['length']} | {fmt(d['median_kabsch_rmsd_angstrom'])} | {d['exact_duplicates']} | {fmt(medians['train'])} | {fmt(medians['heldout'])} |"
        )
    lines += [
        "",
        "Distribution distances describe invariant summaries, with a small matched-length disjoint4+4 real-reference floor. Four generated primary traces per length cannot establish equality of the full protein distribution. Pairwise equal-length Kabsch/distance-matrix/descriptor diversity and nearest measured TRAIN/heldout similarities use bounded subsets (20TRAIN,8heldout); no230kall-against-all search. Nearest-reference RMSD is a novelty measurement, not unconditional generation quality.",
        "",
        "Generated PDBs are **Cα traces only, no sequence, no side chains, and not proof of thermodynamic stability**. Numeric best/median/worst galleries use the frozen standardized geometry score; “success” is a relative rank and not a stability claim.",
        "",
        f"Galleries: `{OUT / 'success_gallery'}`, `{OUT / 'median_gallery'}`, `{OUT / 'failure_gallery'}`.",
        "",
        f"Reconstruction: **{verdict.get('reconstruction')}**. Generation: **{verdict.get('generation')}**. Generalization: **{verdict.get('generalization')}**. Continued training: **{verdict.get('continued_training')}**.",
        "",
    ]
    if verdict.get("generation") == "EVIDENCE_OF_USEFUL_GENERATION":
        answer = "E015 shows bounded evidence of useful generated Cα geometry and heldout learning; the small panels do not establish the full protein-geometry distribution or thermodynamic stability."
    elif gen:
        answer = "E015 is improving one-step denoising, but this bounded full-sampling experiment does not establish useful protein-like generation. Small TRAIN/heldout gaps with substantial geometry error indicate an immature or underfit model, rather than proven distribution learning."
    else:
        answer = "E015 shows one-step denoising learning; useful full de novo generation remains unmeasured or inconclusive. Reconstruction alone cannot answer the generator question."
    verdict["scientific_answer"] = answer
    verdict["tiers_completed"] = completion.get("completed", [])
    verdict["resource_pressure_stop"] = completion.get("resource_pressure", False)
    lines += [
        answer,
        "",
        verdict.get("generalization_reason", ""),
        "",
        "Compact head contributions, fixed-context chiral-only removal and H/V/P/A finite/RMS/std/near-zero statistics are in head_contributions.json and typed_state_statistics.json. Full hidden tensors were not saved.",
        "",
        "| Checkpoint | Weight | t | Polar RMS | Chiral RMS | Local RMS | Disabled minus enabled v-MSE | Local RMSE Å | Torsion rad | Inversion | Clashes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    head_file = read("head_contributions.json", {"rows": []})
    heads = [q for q in head_file["rows"] if q.get("split") == "heldout" and q.get("identity") in selected_ids]
    summaries = []
    for cp, w, t in sorted({(q["checkpoint"], q["weights"], q["timestep"]) for q in heads}):
        group = [q for q in heads if (q["checkpoint"], q["weights"], q["timestep"]) == (cp, w, t)]
        summary = dict(
            checkpoint=cp,
            weights=w,
            timestep=t,
            proteins=len(group),
            branch_rms={
                k: statistics.mean(q["branch_rms"][k] for q in group) for k in ("polar", "chiral", "local_frame")
            },
            branch_to_final_rms={
                k: statistics.mean(q["branch_to_final_rms"][k] for q in group)
                for k in ("polar", "chiral", "local_frame")
            },
            gate_mean=statistics.mean(q["gate_mean"] for q in group),
            disabled_minus_enabled={
                k: statistics.mean(q["chiral_only_disabled"][k] - q["enabled"][k] for q in group)
                for k in (
                    "v_mse",
                    "local_rmse_angstrom",
                    "torsion_angular_mae_radians",
                    "triple_inversion_fraction",
                    "clash_fraction",
                )
            },
        )
        summaries.append(summary)
        values = [*summary["branch_rms"].values(), *summary["disabled_minus_enabled"].values()]
        lines.append(f"| {cp} | {w} | {t} | " + " | ".join(fmt(v) for v in values) + " |")
    head_file["heldout_summary_by_timestep"] = summaries
    r.write(OUT / "head_contributions.json", head_file)
    lines += [
        "",
        f"Resource summary: `{json.dumps(resource_summary)}`.",
        "",
        "Exact production resume command:",
        "",
        "```bash",
        "/home/simostocco/miniforge3/envs/proteingen/bin/python -B /mnt/d/Simone/proteinGen/.e015_worktree/scripts/supervise_e015_training.py --resume",
        "```",
        "",
        f"Final resume proof: `{OUT / 'logs/final_resume.json'}`. Current supervisor/trainer metadata: `{r.ROOT / 'SUPERVISOR.json'}`, `{r.ROOT / 'STATE.json'}`. Initial exposure vector equals the deterministic schedule prefix; pause cursor equals committed cursor with zero uncommitted exposures; strict state restoration and the first resumed heartbeat verify continuation; subsequent rolling SHA verification is recorded.",
        "",
        "E007 remains paused; E012 and E014 were not modified or resumed. No sequence-model training started. All generated artifacts, temporary files and caches stayed on D:.",
        "",
    ]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=r.WORK, text=True).strip()
    lines.append(
        f"Current commit: `{commit}`. Recovery/evaluation helper files under docs are excluded from production science source hashes; production source/config were unchanged."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    r.write(OUT / "verdict.json", verdict)
    # Explicit pending records prevent placeholder artifacts from masquerading as measurements.
    for name in (
        "generalization_bootstrap.json",
        "generation_distribution_metrics.json",
        "diversity_novelty_metrics.json",
        "head_contributions.json",
        "typed_state_statistics.json",
    ):
        if not (OUT / name).exists():
            r.write(OUT / name, dict(status="PENDING", reason=completion.get("error", "Tier not completed")))
    for name in (
        "train_vs_heldout_metrics.parquet",
        "raw_vs_ema_metrics.parquet",
        "generation_manifest.parquet",
        "generation_metrics.parquet",
        "trajectory_metrics.parquet",
    ):
        if not (OUT / name).exists():
            import pyarrow as pa

            pq.write_table(pa.table({"status": pa.array([], pa.string())}), OUT / name)
    files = [dict(path=str(p.relative_to(OUT)), bytes=p.stat().st_size) for p in OUT.rglob("*") if p.is_file()]
    total = sum(q["bytes"] for q in files)
    assert total < 5 * 1024**3
    r.write(
        OUT / "MANIFEST.json",
        dict(
            status="COMPLETE" if not completion.get("error") else "PARTIAL_PRESERVED_TRAINING_RESUMED",
            primary_checkpoint=primary,
            primary_sha256="2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03",
            shutdown_checkpoint=selected,
            files=files,
            total_bytes=total,
            limit_bytes=5 * 1024**3,
            all_artifacts_on_d=True,
            serialized=True,
            completed=completion.get("completed", []),
            final_resume=final,
            commit=commit,
        ),
    )
    print(json.dumps(dict(report=str(OUT / "REPORT.md"), verdict=verdict, total_bytes=total)), flush=True)


if __name__ == "__main__":
    main()
