# ruff: noqa: E402, E501
"""Serialized bounded CUDA evaluation, reusing frozen panels and compact metrics."""

import contextlib
import fcntl
import json
import math
import os
import shutil
import sys
import time
import types
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
WORK = Path("/mnt/d/Simone/proteinGen/.e015_worktree")
# Reuse existing diagnostic implementation, with single-thread evaluation settings.
# Its source is preserved; no production contract file is changed.
source = (WORK / "docs/e015_serial_helpers.py").read_text()
source = source.replace("os.environ[key]='2'", "os.environ[key]='1'").replace(
    "torch.set_num_threads(2)", "torch.set_num_threads(1)"
)
old_argv = sys.argv[:]
sys.argv = [str(WORK / "docs/e015_serial_helpers.py"), "generate"]
b = types.ModuleType("e015_existing_absolute")
exec(compile(source, str(WORK / "docs/e015_serial_helpers.py"), "exec"), b.__dict__)
sys.argv = old_argv
ROOT, OUT, torch, np = b.ROOT, b.OUT, b.torch, b.np
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
b.PRIMARY = [*b.PRIMARY, "local_rmse_angstrom"]
import e015_serial_recovery as recovery

from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import ks, w1
from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import State, TypedBlock
from protein_distance_diffusion.training.e015_self_conditioning import make_self_condition


def pulse(phase, sample=None):
    now = time.monotonic()
    if now - pulse.last >= 30:
        r = recovery.resources()
        r.update(evaluation_phase=phase, current_sample=sample, evaluator_pid=os.getpid(), trainer_pid=None)
        with (OUT / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        pulse.last = now
        if (
            r["ram_available_bytes"] < 2 * 1024**3
            or r["d_free_bytes"] < 30 * 1024**3
            or r["swap_used_bytes"] - pulse.swap > 256 * 1024**2
        ):
            b.write("logs/resource_pressure.json", r)
            raise RuntimeError("Resource pressure: clean evaluation stop")
        pulse.swap = r["swap_used_bytes"]
    if (OUT / "STOP_EVALUATION.json").exists():
        raise RuntimeError("Clean evaluation stop requested")


pulse.last = 0
pulse.swap = 0


def acquire_lease():
    lease = (ROOT / "GPU_WORKER.lock").open("rb")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError("Existing E015 GPU lease is occupied") from None
    return lease


def inputs():
    # Existing frozen panel is authoritative; do not select a new panel.
    panel = b.read("panel_definitions.json")["rows"]
    cache = b.CoordinateCache(Path(b.cfg["population_root"]), verify_payload=False)
    return panel, cache


def records(report):
    a = recovery.read(Path(report) / "recovery_audit.json")
    inventory = b.read("checkpoint_inventory.json")["checkpoints"]
    rows = [r for r in inventory if r["label"] in b.CP]
    selected = a["selected"]
    rows.append(
        dict(
            label="shutdown_context_u" + str(selected["position"]["update"]),
            source_path=a["secondary_immutable_snapshot"],
            sha256=selected["sha256"],
            position=selected["position"],
        )
    )
    return rows


@contextlib.contextmanager
def capture(model, sink, row, label, t, weights):
    handles = []

    def save(stage, state):
        if not isinstance(state, State):
            return
        out = {}
        for k, v in zip(("H", "V", "P", "A"), (state.h, state.v, state.p, state.a), strict=True):
            v = v.detach().float()
            out[k] = dict(
                rms=float(v.square().mean().sqrt()),
                std=float(v.std()),
                finite_fraction=float(torch.isfinite(v).float().mean()),
                near_zero_fraction=float((v.abs() < 1e-8).float().mean()),
            )
        sink.append(
            dict(checkpoint=label, weights=weights, identity=row["sample_id"], timestep=t, stage=stage, types=out)
        )

    for name, m in model.named_modules():
        if isinstance(m, TypedBlock) or name in ("up_medium", "up_fine"):
            handles.append(m.register_forward_hook(lambda m, args, out, name=name: save(name, out)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def measure(e, model, row, cache, label, weights, heads, typed):
    # CPU-generated noise is identical to the saved deterministic noise family.
    x, n, m = b.coordinate_batch([row], cache, b.SCALE, device="cpu")
    for t0 in b.GRID:
        pulse("reconstruction", row["sample_id"])
        t = torch.tensor([t0])
        seed = b.noise_seed("e015-absolute-matched-noise-v1", row["pair_id"], t0)
        batch = e.diffusion.make_training_batch(x, m, timesteps=t, generator=torch.Generator().manual_seed(seed))
        xt = batch.noisy_coordinates.cuda()
        ng = n.cuda()
        mg = m.cuda()
        tg = t.cuda()
        prev, available = make_self_condition(model, e.diffusion, xt, tg, ng, mg, model.self_conditioning_enabled)
        analyze_head = label == b.CP[-1] or label.startswith("shutdown_context")
        instrument = analyze_head and row["pair_index"] == 0 and row["eval_split"] == "heldout"
        with capture(model, typed, row, label, t0, weights) if instrument else contextlib.nullcontext():
            vg, comp = model.with_components(xt, tg, ng, mg, prev, available)
        v = vg.cpu()
        components = {k: z.cpu() if isinstance(z, torch.Tensor) else z for k, z in comp[0].items()}
        nxt, xhat, _ = e.diffusion.deterministic_reverse_step(batch.noisy_coordinates, t, v, m)
        oracle = b.paired_state(e.diffusion, x, batch.coordinate_noise, t - 1, m)
        out = b.extended_metrics(xhat[0], x[0], v[0], batch.coordinate_v_target[0], b.SCALE, components)
        out.update(b.vector_metrics(v, batch.coordinate_v_target, m, "v"))
        out["local_rmse_angstrom"] = math.sqrt(out["local_mse_angstrom2"])
        assert out["finite"] and torch.isfinite(nxt).all()
        noop = float((batch.noisy_coordinates - oracle).square()[m].mean())
        step = float((nxt - oracle).square()[m].mean())
        a = xhat[0] * b.SCALE
        tar = x[0] * b.SCALE
        ia = b.intrinsic(a.numpy())
        out.update({k: z for k, z in ia.items() if k not in ("descriptor", "length", "torsion_assessable_count")})
        out.update(
            centered_component_mse_angstrom2=float((a - tar).square().mean()),
            centered_coordinate_rmse_angstrom=float((a - tar).square().mean().sqrt()),
            centered_euclidean_rmsd_angstrom=float((a - tar).square().sum(-1).mean().sqrt()),
            v_predicted_rms=out["v_predicted_norm"] / math.sqrt(3 * row["length"]),
            v_target_rms=out["v_target_norm"] / math.sqrt(3 * row["length"]),
            next_step_coordinate_mse_normalized=step,
            next_step_coordinate_mse_angstrom2=step * b.SCALE**2,
            noop_next_step_mse_normalized=noop,
            step_gain=1 - step / noop if noop > 0 else None,
            covariance_trace_ratio=out["radius_ratio"] ** 2,
            torsion_angular_mae_degrees=out["torsion_angular_mae_radians"] * 180 / math.pi
            if out["torsion_angular_mae_radians"] is not None
            else None,
            direct_x0_collapse_count=int(out["collapsed"]),
            noisy_next_rg_ratio_to_oracle=float(
                nxt.square().sum(-1).mean().sqrt() / oracle.square().sum(-1).mean().sqrt()
            ),
            sample_id=row["sample_id"],
            split=row["eval_split"],
            pair_id=row["pair_id"],
            length=row["length"],
            length_stratum=row["length_stratum"],
            timestep=t0,
            noise_seed=seed,
            checkpoint=label,
            weights=weights,
            inference_backend="CUDA FP32; identical CPU noise; existing CPU records reused",
        )
        for k in (1, 2, 3):
            dp = (a[k:] - a[:-k]).norm(dim=-1)
            dt = (tar[k:] - tar[:-k]).norm(dim=-1)
            out[f"offset_{k}_rmse_angstrom"] = out[f"offset_{k}_mse_angstrom2"] ** 0.5
            out[f"offset_{k}_absolute_error_quantiles_angstrom"] = torch.quantile(
                (dp - dt).abs(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
            ).tolist()
        fp, _, _ = b.local_geometry(xhat[0])
        ft, _, _ = b.local_geometry(x[0])
        ass = (fp[:, 29] > 0) & (ft[:, 29] > 0)
        out["torsion_sin_mse"] = float((fp[:, 25] - ft[:, 25]).square()[ass].mean()) if ass.any() else None
        out["torsion_cos_mse"] = float((fp[:, 26] - ft[:, 26]).square()[ass].mean()) if ass.any() else None
        if analyze_head:
            rms = float(v.square().mean().sqrt())
            branches = {
                k: float(components[z].square().mean().sqrt())
                for k, z in [("polar", "global_v"), ("chiral", "chiral_effective"), ("local_frame", "local_effective")]
            }
            ab = v - components["chiral_effective"][None]
            disabled = b.extended_metrics(
                e.diffusion.reconstruct_x0(batch.noisy_coordinates, t, ab, m)[0],
                x[0],
                ab[0],
                batch.coordinate_v_target[0],
                b.SCALE,
                components,
            )
            disabled["local_rmse_angstrom"] = math.sqrt(disabled["local_mse_angstrom2"])
            heads.append(
                dict(
                    checkpoint=label,
                    weights=weights,
                    identity=row["sample_id"],
                    split=row["eval_split"],
                    length_stratum=row["length_stratum"],
                    timestep=t0,
                    branch_rms=branches,
                    branch_to_final_rms={k: z / max(rms, 1e-12) for k, z in branches.items()},
                    gate_mean=float(components["gate"].mean()),
                    gate_quantiles=torch.quantile(
                        components["gate"], torch.tensor([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
                    ).tolist(),
                    enabled={k: out[k] for k in b.PRIMARY},
                    chiral_only_disabled={k: disabled[k] for k in b.PRIMARY},
                    ablation="Fixed self-context/trunk additive chiral-only removal; local frame contribution retained; existing combined switch would also disable local head",
                )
            )
        yield out


def reconstruction(report, tier):
    recovery.gate()
    _lease = acquire_lease()
    panel, cache = inputs()
    e = b.model_engine()
    path = OUT / "absolute_reconstruction_metrics.parquet"
    rows = b.pq.read_table(path, use_threads=False).to_pylist() if path.exists() else []
    identities = {r["sample_id"]: r for r in panel}
    for row in rows:
        frozen = identities[row["sample_id"]]
        assert row["noise_seed"] == b.noise_seed("e015-absolute-matched-noise-v1", frozen["pair_id"], row["timestep"])
        assert row["split"] == frozen["eval_split"] and row["length"] == frozen["length"]
        row["local_rmse_angstrom"] = math.sqrt(row["local_mse_angstrom2"])
    done = {(r["checkpoint"], r["weights"], r["sample_id"], r["timestep"]) for r in rows}
    heads = b.read("head_contributions.json").get("rows", []) if (OUT / "head_contributions.json").exists() else []
    typed = (
        b.read("typed_state_statistics.json").get("rows", []) if (OUT / "typed_state_statistics.json").exists() else []
    )
    # Four exact-length pairs per stratum for Tier1; ten per stratum for Tier2.
    subset = [r for r in panel if r["pair_index"] in ((0, 2, 10, 12) if tier == 1 else tuple(range(0, 20, 2)))]
    started = time.monotonic()
    count = 0
    for rec in records(report):
        saved, h = b.pinned_load(rec["source_path"])
        assert h == rec["sha256"]
        b.verify(saved, e)
        for weights, model in [("raw", e.model), ("ema", e.ema)]:
            model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
            model.cuda().eval()
            for row in subset:
                if all((rec["label"], weights, row["sample_id"], t) in done for t in b.GRID):
                    continue
                start = time.monotonic()
                for out in measure(e, model, row, cache, rec["label"], weights, heads, typed):
                    ident = (rec["label"], weights, row["sample_id"], out["timestep"])
                    if ident not in done:
                        rows.append(out)
                        done.add(ident)
                count += 1
                b.table("absolute_reconstruction_metrics.parquet", rows)
                b.write("head_contributions.json", dict(rows=heads))
                b.write("typed_state_statistics.json", dict(rows=typed, full_hidden_tensors_saved=False))
                b.progress(
                    "tier" + str(tier),
                    checkpoint=rec["label"],
                    weights=weights,
                    identity=row["sample_id"],
                    seconds=time.monotonic() - start,
                    conditions=len(rows),
                )
            model.cpu()
            torch.cuda.empty_cache()
        del saved
    b.write(
        "logs/tier" + str(tier) + "_complete.json",
        dict(
            stable=True,
            seconds=time.monotonic() - started,
            new_protein_weight_conditions=count,
            train_count=len(subset) // 2,
            heldout_count=len(subset) // 2,
            workers=0,
            batch_size=1,
            threads=1,
        ),
    )
    summaries(rows, report, tier)


def summaries(rows, report, tier):
    primary = b.CP[-1]
    panel, _ = inputs()
    valid = {
        r["sample_id"] for r in panel if r["pair_index"] in ((0, 2, 10, 12) if tier == 1 else tuple(range(0, 20, 2)))
    }
    rows = [r for r in rows if r["sample_id"] in valid]
    gaps = []
    rawema = []
    means = []
    for cp in [r["label"] for r in records(report)]:
        for weights in ("raw", "ema"):
            for t in [None, *b.GRID]:
                for st in ["all", *b.STRATA]:
                    selected = [
                        r
                        for r in rows
                        if r["checkpoint"] == cp
                        and r["weights"] == weights
                        and (t is None or r["timestep"] == t)
                        and (st == "all" or r["length_stratum"] == st)
                    ]
                    pairs = {}
                    for r in selected:
                        pairs.setdefault(r["pair_id"], {}).setdefault(r["split"], []).append(r)
                    for metric in b.PRIMARY:
                        a = []
                        bb = []
                        strata = []
                        for pair in pairs.values():
                            if all(pair.get(s) for s in ("train", "heldout")):
                                a.append(
                                    float(np.mean([r[metric] for r in pair["train"] if r.get(metric) is not None]))
                                )
                                bb.append(
                                    float(np.mean([r[metric] for r in pair["heldout"] if r.get(metric) is not None]))
                                )
                                strata.append(pair["train"][0]["length_stratum"])
                        if not a:
                            continue
                        est = b.estimate(a, bb, strata)
                        gaps.append(
                            dict(
                                checkpoint=cp,
                                weights=weights,
                                timestep=t,
                                length_stratum=st,
                                metric=metric,
                                train_median=float(np.median(a)),
                                heldout_median=float(np.median(bb)),
                                train_mean=est["old_mean"],
                                heldout_mean=est["new_mean"],
                                gap=est["absolute_difference"],
                                **est,
                            )
                        )
                    for split in ("train", "heldout"):
                        for metric, stat in b.groupmean([r for r in selected if r["split"] == split]).items():
                            means.append(
                                dict(
                                    checkpoint=cp,
                                    weights=weights,
                                    timestep=t,
                                    length_stratum=st,
                                    split=split,
                                    metric=metric,
                                    **stat,
                                )
                            )
        for t in [None, *b.GRID]:
            for split in ("train", "heldout"):
                for metric in b.PRIMARY:
                    groups = {}
                    for r in rows:
                        if r["checkpoint"] == cp and r["split"] == split and (t is None or r["timestep"] == t):
                            groups.setdefault(r["sample_id"], {}).setdefault(r["weights"], []).append(r)
                    a = []
                    bb = []
                    strata = []
                    for g in groups.values():
                        if all(g.get(w) for w in ("raw", "ema")):
                            a.append(np.mean([r[metric] for r in g["ema"] if r.get(metric) is not None]))
                            bb.append(np.mean([r[metric] for r in g["raw"] if r.get(metric) is not None]))
                            strata.append(g["raw"][0]["length_stratum"])
                    if a:
                        est = b.estimate(a, bb, strata)
                        rawema.append(
                            dict(
                                checkpoint=cp,
                                timestep=t,
                                split=split,
                                metric=metric,
                                ema_mean=est["old_mean"],
                                raw_mean=est["new_mean"],
                                **est,
                            )
                        )
    b.table("train_vs_heldout_metrics.parquet", means)
    b.table("raw_vs_ema_metrics.parquet", rawema)
    b.write(
        "generalization_bootstrap.json",
        dict(
            rows=gaps,
            replicates=2000,
            tier=tier,
            aggregation="Equal proteins, timesteps averaged within identity, paired exact-length TRAIN/heldout; stratified protein bootstrap",
        ),
    )
    rv = next(
        r
        for r in rawema
        if r["checkpoint"] == primary and r["timestep"] is None and r["split"] == "heldout" and r["metric"] == "v_mse"
    )
    choice = "raw" if rv["raw_mean"] < rv["ema_mean"] else "ema"
    b.write(
        "sampler_weight_selection.json",
        dict(
            primary_weights=choice,
            raw_v_mse=rv["raw_mean"],
            ema_v_mse=rv["ema_mean"],
            selection="Frozen heldout equal-protein equal-timestep aggregate v-MSE; tie EMA; primary checkpoint fixed 50%",
        ),
    )
    selected = [r for r in rows if r["checkpoint"] == primary and r["weights"] == choice and r["split"] == "heldout"]
    vg = next(
        r
        for r in gaps
        if r["checkpoint"] == primary
        and r["weights"] == choice
        and r["timestep"] is None
        and r["length_stratum"] == "all"
        and r["metric"] == "v_mse"
    )
    bond = np.mean([r["adjacent_valid_fraction"] for r in selected])
    generation = (
        b.read("verdict.json").get("generation", "INCONCLUSIVE") if (OUT / "verdict.json").exists() else "INCONCLUSIVE"
    )
    general = (
        "UNDERFIT"
        if bond < 0.9 and abs(vg["relative_difference"]) < 0.15
        else "MILD_OVERFIT"
        if vg["difference_ci95"][0] > 0 and vg["relative_difference"] > 0.15
        else "GENERALIZING"
        if bond >= 0.9 and abs(vg["relative_difference"]) < 0.15
        else "INCONCLUSIVE"
    )
    b.write(
        "verdict.json",
        dict(
            reconstruction="PROMISING_BUT_IMMATURE",
            generation=generation,
            generalization=general,
            continued_training="CONTINUE_BUT_WATCH",
            sampler_weights=choice,
            primary_sha256=b.SHA,
            primary_update=3811,
            primary_exposures=115066,
            tier_reconstruction=tier,
            raw_ema_comparison_v_mse=dict(raw=rv["raw_mean"], ema=rv["ema_mean"]),
            all_outputs_on_d=True,
        ),
    )
    b.progress(
        "reconstruction_summary_complete",
        tier=tier,
        weights=choice,
        train_v_mse=vg["train_mean"],
        heldout_v_mse=vg["heldout_mean"],
        gap=vg["gap"],
    )


@torch.no_grad()
def generation(report, pilot):
    recovery.gate()
    _lease = acquire_lease()
    panel, cache = inputs()
    e = b.model_engine()
    saved, h = b.pinned_load(ROOT / "production/checkpoints/epoch_1_fraction_0.50.pt")
    assert h == b.SHA
    b.verify(saved, e)
    chosen = b.read("sampler_weight_selection.json")["primary_weights"]
    rows = (
        b.pq.read_table(OUT / "generation_metrics.parquet", use_threads=False).to_pylist()
        if (OUT / "generation_metrics.parquet").exists()
        else []
    )
    samples = (
        b.pq.read_table(OUT / "generation_manifest.parquet", use_threads=False).to_pylist()
        if (OUT / "generation_manifest.parquet").exists()
        else []
    )
    traj = (
        b.pq.read_table(OUT / "trajectory_metrics.parquet", use_threads=False).to_pylist()
        if (OUT / "trajectory_metrics.parquet").exists()
        else []
    )
    done = {r["sample_key"] for r in samples}
    frames = {499, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0}
    sets = (
        [(chosen, 2, [48, 192, 448])]
        if pilot
        else [(chosen, 4, b.LENGTHS), ("ema" if chosen == "raw" else "raw", 1, b.LENGTHS)]
    )
    for weights, count, lengths in sets:
        model = e.model if weights == "raw" else e.ema
        model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
        model.cuda().eval()
        for n in lengths:
            for i in range(count):
                seed = 15015000 + i
                name = f"{weights}_n{n}_s{seed}"
                if name in done:
                    continue
                pulse("pilot" if pilot else "generation_panel", name)
                mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
                length = torch.tensor([n], device="cuda")
                g = torch.Generator(device="cuda").manual_seed(seed)
                current = b.centered_coordinate_noise(torch.empty(1, n, 3, device="cuda"), mask, generator=g)
                context = None
                trajectory = {}
                local = []
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                for t0 in range(499, -1, -1):
                    pulse("pilot" if pilot else "generation_panel", name + ":t" + str(t0))
                    t = torch.tensor([t0], device="cuda")
                    available = torch.tensor([context is not None], device="cuda")
                    v, comp = model.with_components(current, t, length, mask, context, available)
                    if t0 in frames and t0 != 0:
                        xyz = (current[0] * b.SCALE).cpu().numpy()
                        trajectory[t0] = xyz
                        local.append(dict(b.intrinsic(xyz), sample_key=name, timestep=t0))
                    current, context, _ = e.diffusion.deterministic_reverse_step(current, t, v, mask)
                    context = context.detach()
                    if not torch.isfinite(current).all():
                        b.write("logs/nonfinite_generation.json", dict(sample=name, timestep=t0))
                        raise FloatingPointError(name)
                xyz = (current[0] * b.SCALE).cpu().numpy()
                trajectory[0] = xyz
                stat = b.intrinsic(xyz)
                local.append(dict(stat, sample_key=name, timestep=0))
                folder = OUT / "samples" / name
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder / "final_coordinates.npz", coordinates_angstrom=xyz)
                np.savez_compressed(folder / "sparse_trajectory.npz", **{f"t{t}": a for t, a in trajectory.items()})
                pdb = [
                    "REMARK   1 CA TRACE ONLY; NO SEQUENCE; NO SIDE CHAINS",
                    "REMARK   2 NOT PROOF OF THERMODYNAMIC STABILITY",
                    f"REMARK   3 E015 {weights} SHA256 {b.SHA}",
                ]
                for j, (a, c, d) in enumerate(xyz, 1):
                    pdb.append(f"ATOM  {j:5d}  CA  UNK A{j:4d}    {a:8.3f}{c:8.3f}{d:8.3f}  1.00  0.00           C")
                (folder / "trace_ca_only.pdb").write_text("\n".join(pdb + ["TER", "END"]) + "\n")
                meta = dict(
                    sample_key=name,
                    requested_length=n,
                    seed=seed,
                    weights=weights,
                    primary=weights == chosen,
                    checkpoint_sha256=b.SHA,
                    path=str(folder),
                    runtime_seconds=time.monotonic() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    sampler="Exact production CoordinateVPDiffusion deterministic_reverse_step;500steps499..0, generated self-context only,FP32 as production evaluator",
                    terminal_frame="0 is final estimate after terminal update; intermediate frames are pre-update states",
                )
                b.write(str(folder.relative_to(OUT) / "metrics.json"), dict(meta=meta, metrics=stat))
                b.sample_images(meta, stat, trajectory)
                samples.append(meta)
                rows.append(
                    dict(stat, **{k: meta[k] for k in ("sample_key", "requested_length", "seed", "weights", "primary")})
                )
                traj += local
                done.add(name)
                b.table("generation_manifest.parquet", samples)
                b.table("generation_metrics.parquet", rows)
                b.table("trajectory_metrics.parquet", traj)
                b.progress(
                    "generated",
                    sample=name,
                    seconds=meta["runtime_seconds"],
                    adjacent_angstrom=stat["adjacent_mean_angstrom"],
                    rg_angstrom=stat["radius_of_gyration_angstrom"],
                    clash=stat["clash_fraction"],
                )
        model.cpu()
        torch.cuda.empty_cache()
    del saved, e
    torch.cuda.empty_cache()
    rate = float(np.mean([r["finite_fraction"] == 1 for r in rows]))
    b.write(
        "logs/tier" + ("3" if pilot else "4") + "_complete.json",
        dict(stable=True, finite_rate=rate, samples=len(rows), primary_samples=sum(r["primary"] for r in rows)),
    )
    if pilot:
        pilot_analysis(rows, cache)
    else:
        distribution_analysis(samples, rows, cache)


def refs(cache, n, count=16):
    dev = b.pq.read_table(
        Path(b.cfg["population_root"]) / "population/development.parquet", use_threads=False
    ).to_pylist()
    rows = sorted(dev, key=lambda r: (abs(r["length"] - n), b.key("e015-generation-reference-v1", r)))[:count]
    return [
        dict(b.intrinsic(cache.get(r["sample_id"])), sample_id=r["sample_id"], length=r["length"]) for r in rows
    ], rows


def pilot_analysis(rows, cache):
    reference = []
    for n in (48, 192, 448):
        rr, _ = refs(cache, n)
        reference.append(
            dict(
                length=n,
                reference_ids=[r["sample_id"] for r in rr],
                metrics={
                    k: dict(
                        reference_percentiles=np.quantile([r[k] for r in rr], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                        generated=[r[k] for r in rows if r["requested_length"] == n],
                    )
                    for k in (
                        "adjacent_mean_angstrom",
                        "radius_of_gyration_angstrom",
                        "clash_fraction",
                        "end_to_end_angstrom",
                    )
                    if k in rr[0]
                },
            )
        )
    b.write(
        "generation_pilot_reference.json",
        dict(
            rows=reference,
            finite=all(r["finite_fraction"] == 1 for r in rows),
            expansion_gate="All6finite and hoststable; physical deficits remain scientific evidence rather than expansion-selection criterion",
        ),
    )


def population_metrics(c, ia, ib):
    # Unconditional samples are independent of the real reference identities.
    count = len(ia)
    wa = np.bincount(ia, minlength=count) / count
    wb = np.bincount(ib, minlength=count) / count
    aa, bb = c.a[ia], c.b[ib]
    ca = np.cov(aa, rowvar=False, ddof=1)
    cb = np.cov(bb, rowvar=False, ddof=1)
    return dict(
        mean_error=float(np.linalg.norm(aa.mean(0) - bb.mean(0)) / np.sqrt(aa.shape[1])),
        covariance_error=float(np.linalg.norm(ca - cb) / max(np.linalg.norm(cb), 1e-12)),
        energy=float(max(2 * wa @ c.ab @ wb - wa @ c.aa @ wa - wb @ c.bb @ wb, 0)),
        mmd2=float(max(wa @ c.kaa @ wa + wb @ c.kbb @ wb - 2 * wa @ c.kab @ wb, 0)),
        sliced_wasserstein=w1(c.pa[ia], c.pb[ib]),
        rg_wasserstein=w1(c.raw_a[ia, 0], c.raw_b[ib, 0]),
        rg_ks=ks(c.raw_a[ia, 0], c.raw_b[ib, 0]),
        covariance_spectrum_w1=w1(aa[:, c.spectrum_columns], bb[:, c.spectrum_columns]),
    )


def distribution_analysis(samples, generated, cache):
    # Use four generated samples and independent real subsets of four per length.
    # The small population floor is explicit; no precision beyond this panel is implied.
    train = b.pq.read_table(Path(b.cfg["population_root"]) / "population/train.parquet", use_threads=False).to_pylist()
    distribution = []
    diversity = []
    novelty = []
    scored = []
    gallery = []
    collapse = []
    reference_rows = []
    for n in b.LENGTHS:
        pulse("distribution_analysis", str(n))
        rr, held = refs(cache, n, 8)
        raw_a = np.array([r["descriptor"] for r in rr[::2]])
        raw_b = np.array([r["descriptor"] for r in rr[1::2]])
        standard = b.ReferenceStandardizer(raw_a, raw_b)
        a = standard.transform(raw_a)
        bb = standard.transform(raw_b)
        h = b.bandwidth(a, bb)
        p = b.projections(len(standard.columns))
        spectrum = [
            i
            for i, c in enumerate(standard.columns)
            if b.NAMES[c] in ("eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3")
        ]
        prim = [r for r in generated if r["requested_length"] == n and r["primary"]]
        assert len(prim) == 4
        reference_rg = float(np.median([r["radius_of_gyration_angstrom"] for r in rr]))
        collapse += [r["radius_of_gyration_angstrom"] < 0.25 * reference_rg for r in prim]
        raw = np.array([r["descriptor"] for r in prim])
        gen = b.Comparison(standard.transform(raw), bb, raw, raw_b, h, p, spectrum)
        floor = b.Comparison(a, bb, raw_a, raw_b, h, p, spectrum)
        gm, fm = gen.metrics(), floor.metrics()
        gd = {k: [] for k in gm}
        fd = {k: [] for k in fm}
        rng = np.random.default_rng(15015)
        for _ in range(2000):
            ig = rng.integers(0, 4, 4)
            ia = rng.integers(0, 4, 4)
            ib = rng.integers(0, 4, 4)
            g, f = population_metrics(gen, ig, ib), population_metrics(floor, ia, ib)
            for k in gm:
                gd[k].append(g[k])
                fd[k].append(f[k])
        metrics = [
            dict(
                metric=k,
                generated_vs_real=gm[k],
                real_vs_real_floor=fm[k],
                excess_above_floor=gm[k] - fm[k],
                excess_ci95=np.quantile(np.array(gd[k]) - fd[k], [0.025, 0.975]).tolist(),
            )
            for k in gm
        ]
        distribution.append(
            dict(
                length=n,
                n=4,
                metrics=metrics,
                reference_ids=[r["sample_id"] for r in rr],
                reference_lengths=[r["length"] for r in rr],
                reference_floor="Closest-length disjoint4+4heldout;2000independent-protein bootstrap draws;shared realB anchor;standardizer frozen",
                mmd_estimator="Finite-sample biased V-statistic,calibrated against equal-size real-real floor",
                descriptor_columns=[b.NAMES[c] for c in standard.columns],
            )
        )
        all_length = [r for r in generated if r["requested_length"] == n]
        coord = {
            r["sample_key"]: np.load(OUT / "samples" / r["sample_key"] / "final_coordinates.npz")[
                "coordinates_angstrom"
            ].astype(float)
            for r in all_length
        }
        pairs = []
        for i, r in enumerate(prim):
            for o in prim[i + 1 :]:
                aa, cc = coord[r["sample_key"]], coord[o["sample_key"]]
                dm1 = np.linalg.norm(aa[:, None] - aa[None, :], axis=-1)
                dm2 = np.linalg.norm(cc[:, None] - cc[None, :], axis=-1)
                pairs.append(
                    dict(
                        a=r["sample_key"],
                        b=o["sample_key"],
                        kabsch_rmsd_angstrom=b.aligned_rmsd(aa, cc),
                        distance_matrix_rmse_angstrom=float(np.sqrt(np.mean((dm1 - dm2) ** 2))),
                        descriptor_distance=float(
                            b.distances(
                                standard.transform(np.array([r["descriptor"]])),
                                standard.transform(np.array([o["descriptor"]])),
                            )[0, 0]
                        ),
                    )
                )
        diversity.append(
            dict(
                length=n,
                pairs=pairs,
                exact_duplicates=sum(q["kabsch_rmsd_angstrom"] < 1e-6 for q in pairs),
                median_kabsch_rmsd_angstrom=float(np.median([q["kabsch_rmsd_angstrom"] for q in pairs])),
            )
        )
        trains = sorted(train, key=lambda r: (abs(r["length"] - n), b.key("e015-generation-novelty-v1", r)))[:20]
        tr = [
            dict(b.intrinsic(cache.get(r["sample_id"])), sample_id=r["sample_id"], length=r["length"]) for r in trains
        ]
        reference_rows += [
            dict(r, requested_length=n, split=split)
            for split, values in [("train", tr), ("heldout", rr)]
            for r in values
        ]
        center = np.median(np.array([b.score_vector(r) for r in rr]), axis=0)
        scale = np.maximum(
            (
                np.quantile(np.array([b.score_vector(r) for r in rr]), 0.75, axis=0)
                - np.quantile(np.array([b.score_vector(r) for r in rr]), 0.25, axis=0)
            )
            / 1.349,
            1e-8,
        )
        ranked = []
        for r in all_length:
            match = {}
            for key, value in r.items():
                if not isinstance(value, (float, int)) or isinstance(value, bool) or key not in rr[0]:
                    continue
                values = [q[key] for q in rr if isinstance(q.get(key), (float, int)) and math.isfinite(q[key])]
                if values:
                    match[key] = dict(
                        reference_quantiles=np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                        generated_percentile=float(np.mean(np.asarray(values) <= value)),
                        generated=value,
                    )
            sample_path = f"samples/{r['sample_key']}/metrics.json"
            sample = b.read(sample_path)
            sample["matched_length_reference"] = dict(
                reference_ids=[q["sample_id"] for q in rr],
                reference_lengths=[q["length"] for q in rr],
                scalar_metrics=match,
            )
            b.write(sample_path, sample)
            vector = b.score_vector(r)
            assessable = bool(np.isfinite(vector).all())
            score = float(np.sqrt(np.mean(((vector - center) / scale) ** 2))) if assessable else float("inf")
            if r["primary"]:
                ranked.append(
                    dict(
                        sample_key=r["sample_key"],
                        score=score,
                        score_assessable=assessable,
                        adjacent_mean_angstrom=r["adjacent_mean_angstrom"],
                        clash_fraction=r["clash_fraction"],
                        rg_angstrom=r["radius_of_gyration_angstrom"],
                    )
                )
            nearest = {}
            for split, ref, metadata in [("train", tr, trains), ("heldout", rr, held)]:
                ds = b.distances(
                    standard.transform(np.array([r["descriptor"]])),
                    standard.transform(np.array([q["descriptor"] for q in ref])),
                )[0]
                j = int(np.argmin(ds))
                exact = []
                for q in metadata:
                    if q["length"] == n:
                        exact.append(
                            dict(
                                identity=q["sample_id"],
                                kabsch_rmsd_angstrom=b.aligned_rmsd(coord[r["sample_key"]], cache.get(q["sample_id"])),
                            )
                        )
                nearest[split] = dict(
                    descriptor_nearest_identity=ref[j]["sample_id"],
                    descriptor_distance=float(ds[j]),
                    descriptor_reference_length=ref[j]["length"],
                    nearest_equal_length_kabsch=min(exact, key=lambda q: q["kabsch_rmsd_angstrom"]) if exact else None,
                    measured_reference_count=len(ref),
                )
            novelty.append(
                dict(
                    sample=r["sample_key"],
                    nearest=nearest,
                    interpretation="Bounded measured subset; extreme distance may indicate artifact; no fullTRAINmemorization claim",
                )
            )
        ranked.sort(key=lambda r: (r["score"], r["sample_key"]))
        scored += ranked
        for category, j in [("success_gallery", 0), ("median_gallery", len(ranked) // 2), ("failure_gallery", -1)]:
            r = ranked[j]
            folder = OUT / category / r["sample_key"]
            folder.mkdir(exist_ok=True)
            for name in ("trace_two_views.png", "distance_map_profiles.png", "trajectory_geometry.png"):
                shutil.copyfile(OUT / "samples" / r["sample_key"] / name, folder / name)
            b.write(
                str(folder.relative_to(OUT) / "selection.json"),
                dict(
                    r,
                    category=category,
                    criterion="Frozen standardized adjacentmean/std,clash,Rg,signedchirality,covarianceeigenvalues score; no visual selection",
                ),
            )
            gallery.append(dict(r, category=category, length=n))
    b.write(
        "generation_distribution_metrics.json",
        dict(
            status="COMPLETE",
            by_length=distribution,
            scope="34 invariant descriptor summaries; full geometry distribution equality and thermodynamic stability not established",
        ),
    )
    b.write(
        "diversity_novelty_metrics.json",
        dict(status="COMPLETE", diversity=diversity, novelty=novelty, galleries=gallery, all_against_all_train=False),
    )
    b.table("generation_scores.parquet", scored)
    b.table("generation_reference_metrics.parquet", reference_rows)
    b.generation_plots(generated, distribution, diversity)
    primary = [r for r in generated if r["primary"]]
    finite = float(np.mean([r["finite_fraction"] == 1 for r in generated]))
    bond = float(np.mean([r["adjacent_valid_fraction"] for r in primary]))
    clash = float(np.mean([r["clash_fraction"] for r in primary]))
    floor_compatible = all(
        next(q for q in d["metrics"] if q["metric"] == "energy")["excess_ci95"][0] <= 0 for d in distribution
    )
    collapsed_fraction = float(np.mean(collapse))
    verdict = b.read("verdict.json")
    verdict.update(
        generation="COLLAPSED_OR_UNSTABLE"
        if finite < 1 or collapsed_fraction > 0.5
        else "EVIDENCE_OF_USEFUL_GENERATION"
        if bond >= 0.9 and clash <= 0.01 and floor_compatible
        else "BOUNDED_BUT_NOT_PROTEIN_LIKE",
        full_de_novo_samples=len(generated),
        primary_de_novo_samples=len(primary),
        secondary_de_novo_samples=len(generated) - len(primary),
        finite_generation_fraction=finite,
        primary_adjacent_valid_fraction=bond,
        primary_clash_fraction=clash,
        descriptor_floor_compatible=floor_compatible,
        generation_rg_collapse_fraction=collapsed_fraction,
        generation_rg_collapse_definition="Generated Rg<0.25closest-length real-reference median; descriptive,not a thermodynamic criterion",
    )
    b.write("verdict.json", verdict)


if __name__ == "__main__":
    b.write("logs/evaluator_owner.json", dict(pid=os.getpid(), phase=sys.argv[1], timestamp_unix=time.time()))
    try:
        if sys.argv[1] == "tier1":
            reconstruction(sys.argv[2], 1)
        elif sys.argv[1] == "tier2":
            reconstruction(sys.argv[2], 2)
        elif sys.argv[1] == "pilot":
            generation(sys.argv[2], True)
        elif sys.argv[1] == "panel":
            generation(sys.argv[2], False)
        else:
            raise ValueError(sys.argv[1])
    except Exception as exc:
        b.write(
            "logs/clean_evaluation_stop.json",
            dict(time_unix=time.time(), error=str(exc), phase=sys.argv[1], partial_preserved=True),
        )
        raise
