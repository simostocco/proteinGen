# ruff: noqa: E402, E501
"""Latest-checkpoint absolute science, using frozen historical implementations."""

import ast
import contextlib
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
OUT = Path(os.environ["E015_LATEST_OUT"])
ROOT = OUT.parent.parent
OLD = OUT.parent / "absolute_generation_u3811_2986d44e_v1"
source = (WORK / "docs/e015_serial_evaluation.py").read_text()
source = source.replace(
    'source = (WORK / "docs/e015_serial_helpers.py").read_text()',
    'source = (WORK / "docs/e015_serial_helpers.py").read_text().replace(\'OUT = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"\', \'OUT = Path(os.environ["E015_LATEST_OUT"])\')',
)
source = source.replace(
    'out.update(b.vector_metrics(v, batch.coordinate_v_target, m, "v"))',
    'out.update(b.vector_metrics(v, batch.coordinate_v_target, m, "v"))\n        out.update(b.vector_metrics(nxt-batch.noisy_coordinates, oracle-batch.noisy_coordinates, m, "step"))',
)
source = source.replace("branch_rms=branches,", "branch_rms=branches, head_details=head_details(v[0], components),")
source = source.replace("range(2000)", "range(5000)").replace("2000independent", "5000independent")
source = source.replace(
    "            1e-8,\n        )\n        ranked = []", "            b.quality_floor,\n        )\n        ranked = []"
)
source = source.replace(
    'OUT / "samples" / r["sample_key"]', 'OUT / "samples" / r["sample_key"].split("_", 1)[0] / r["sample_key"]'
)
source = source.replace(
    "f\"samples/{r['sample_key']}/metrics.json\"",
    "f\"samples/{r['sample_key'].split('_', 1)[0]}/{r['sample_key']}/metrics.json\"",
)
base = types.ModuleType("latest_existing_science")
exec(compile(source, str(WORK / "docs/e015_serial_evaluation.py"), "exec"), base.__dict__)
b = base.b
helper_source = (WORK / "docs/e015_serial_helpers.py").read_text()
image_node = next(
    n for n in ast.parse(helper_source).body if isinstance(n, ast.FunctionDef) and n.name == "sample_images"
)
image_source = "\n".join(helper_source.splitlines()[image_node.lineno - 1 : image_node.end_lineno])
image_source = image_source.replace("add_subplot(3, 4, i", "add_subplot(4, 4, i").replace(
    "Predeclared trajectory montage:", "Individually rescaled context montage:"
)
exec(compile(image_source, "latest_sample_plot_capacity_fix", "exec"), b.__dict__)
np, torch = b.np, b.torch
selection = b.read("checkpoint_selection.json")
SHA = selection["source"]["sha256"]
LATEST = "latest_u" + str(selection["source"]["position"]["update"])
b.SHA = SHA
b.CP = ["initialization", "epoch_1_fraction_0.25", "epoch_1_fraction_0.50", "epoch_1_fraction_1.00", LATEST]
GRID = b.GRID
FRAMES = [499, 475, 450, 425, 400, 350, 300, 250, 200, 150, 100, 50, 0]
PRIMARY = [
    "v_mse",
    "aligned_rmsd_angstrom",
    "local_rmse_angstrom",
    "torsion_angular_mae_radians",
    "triple_inversion_fraction",
    "clash_fraction",
    "rg_discrepancy",
]
VERSION = "latest_absolute_v2_extended_fields_and_tuple_state_hooks"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def table(name):
    return b.pq.read_table(OUT / name, use_threads=False).to_pylist() if (OUT / name).exists() else []


def pulse(phase, sample=None):
    if (OUT / "STOP_EVALUATION.json").exists():
        raise RuntimeError("Clean resource-pressure stop requested")
    if time.monotonic() - pulse.last >= 30:
        r = base.recovery.resources()
        r.update(phase=phase, current_sample=sample, evaluator_pid=os.getpid(), trainer_pid=None)
        with (OUT / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        pulse.last = time.monotonic()
        b.write("logs/progress.json", dict(phase=phase, sample=sample, timestamp_unix=time.time()))
        if (
            r["ram_available_bytes"] < 3 * 1024**3
            or r["d_free_bytes"] < 30 * 1024**3
            or r["swap_used_bytes"] > 256 * 1024**2
        ):
            b.write("STOP_EVALUATION.json", r)
            raise RuntimeError("Resource pressure; preserve partials and exit")


pulse.last = 0
base.pulse = pulse


def cosine(x, y):
    d = float(x.norm() * y.norm())
    return float((x * y).sum()) / d if d > 1e-12 else None


def head_details(v, c):
    heads = {"polar": c["global_v"], "chiral": c["chiral_effective"], "local": c["local_effective"]}
    return dict(
        final_rms=float(v.square().mean().sqrt()),
        cosine_final={k: cosine(x, v) for k, x in heads.items()},
        cosine_pairs={
            f"{k}_{j}": cosine(heads[k], heads[j])
            for k, j in [("polar", "chiral"), ("polar", "local"), ("chiral", "local")]
        },
        rms={k: float(x.square().mean().sqrt()) for k, x in heads.items()},
        gate_p10_p50_p90=torch.quantile(
            c["gate"].float(), torch.tensor([0.1, 0.5, 0.9], device=c["gate"].device)
        ).tolist(),
        gate_mean=float(c["gate"].mean()),
        reliability_p10_p50_p90=torch.quantile(
            c["reliability"].float(), torch.tensor([0.1, 0.5, 0.9], device=c["gate"].device)
        ).tolist(),
    )


base.head_details = head_details
original_extended = b.extended_metrics


def extended(xhat, x, v, target, scale, comp):
    out = original_extended(xhat, x, v, target, scale, comp)
    xyz, ref = xhat.float() * scale, x.float() * scale
    err = ((xyz[1:] - xyz[:-1]).norm(dim=-1) - (ref[1:] - ref[:-1]).norm(dim=-1)).abs()
    out.update(
        adjacent_distance_mae_angstrom=float(err.mean()),
        adjacent_distance_error_p50_p90_p95_p99_angstrom=torch.quantile(
            err, torch.tensor([0.5, 0.9, 0.95, 0.99], device=err.device)
        ).tolist(),
        rg_discrepancy=abs(out["radius_ratio"] - 1),
        metric_version=VERSION,
    )

    def angles(z):
        q = z[1:] - z[:-1]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.acos(-(q[:-1] * q[1:]).sum(-1).clamp(-1, 1))

    out["bond_angle_mae_radians"] = float((angles(xyz) - angles(ref)).abs().mean())
    fp, _, _ = b.local_geometry(xhat)
    ft, _, _ = b.local_geometry(x)
    mask = (fp[:, 29] > 0) & (ft[:, 29] > 0)
    sign = mask & (ft[:, 27].abs() > 1e-3)
    out.update(
        torsion_pair_assessable_count=int(mask.sum()),
        torsion_pair_assessable_fraction=float(mask.sum()) / max(len(x) - 3, 1),
        chirality_assessable_count=int(sign.sum()),
        chirality_assessable_fraction=float(sign.sum()) / max(len(x) - 3, 1),
    )
    return out


b.extended_metrics = extended


@contextlib.contextmanager
def capture(model, sink, row, label, t, weights):
    def record(stage, state):
        if isinstance(state, tuple) and len(state) == 4 and all(isinstance(x, torch.Tensor) for x in state):
            state = base.State(*state)
        if not isinstance(state, base.State):
            return
        for k, v in zip(("H", "V", "P", "A"), (state.h, state.v, state.p, state.a), strict=True):
            v = v.detach().float()
            stat = dict(
                checkpoint=label,
                weights=weights,
                identity=row["sample_id"],
                timestep=t,
                stage=stage,
                type=k,
                setting=row.get("setting", "reconstruction"),
                rms=float(v.square().mean().sqrt()),
                mean=float(v.mean()),
                std=float(v.std()),
                finite_fraction=float(torch.isfinite(v).float().mean()),
                near_zero_fraction=float((v.abs() < 1e-8).float().mean()),
                nodes=v.shape[0],
                vector_norm_mean=None,
                vector_norm_rms=None,
            )
            if k in ("V", "A"):
                norms = v.norm(dim=-1)
                stat.update(vector_norm_mean=float(norms.mean()), vector_norm_rms=float(norms.square().mean().sqrt()))
            sink.append(stat)

    handles = []
    original = model.prepare_single

    def prepare(*args, **kwargs):
        out = original(*args, **kwargs)
        record("embedding", out[0])
        return out

    model.prepare_single = prepare
    for name, module in model.named_modules():
        if isinstance(module, base.TypedBlock) or name in ("up_medium", "up_fine"):
            handles.append(module.register_forward_hook(lambda m, args, out, name=name: record(name, out)))
    try:
        yield
    finally:
        model.prepare_single = original
        for handle in handles:
            handle.remove()


base.capture = capture


def records():
    audit = base.recovery.read(Path(selection["audit"]) / "recovery_audit.json")
    inventory = audit["inventory"]
    out = []
    for label in b.CP:
        if label == LATEST:
            out.append(
                dict(label=label, path=selection["snapshot"], sha256=SHA, position=selection["source"]["position"])
            )
        else:
            q = next(r for r in inventory if Path(r["path"]).stem == label)
            out.append(dict(label=label, path=q["path"], sha256=q["sha256"], position=q["position"]))
    return out


def reconstruction():
    _lease = base.acquire_lease()
    frozen = json.loads((OLD / "panel_definitions.json").read_text())
    panel = frozen["rows"]
    train = {q["sample_id"] for q in panel if q["eval_split"] == "train"}
    held = {q["sample_id"] for q in panel if q["eval_split"] == "heldout"}
    assert len(train) == len(held) == 100 and not train & held
    for st in b.STRATA:
        assert sum(q["length_stratum"] == st and q["eval_split"] == "train" for q in panel) == 20
    b.write(
        "panel_definitions.json",
        dict(
            frozen,
            source=str(OLD / "panel_definitions.json"),
            source_sha256=sha(OLD / "panel_definitions.json"),
            no_split_overlap=True,
            selection="Existing frozen full100+100; no reselection; identical pair/noise family",
        ),
    )
    b.write(
        "metric_contract.json",
        dict(
            version=VERSION,
            scale_angstrom=b.SCALE,
            inference="FP32 as historical and production evaluator; trainer BF16 unchanged",
            bootstrap_draws=5000,
            workers=0,
            batch_size=1,
            threads=1,
            same_noise_family="e015-absolute-matched-noise-v1",
            historical_extensions="Additional quantiles, step field, full tuple/input typed hooks require inference; valid prior scalar columns retained and tracked",
        ),
    )
    prior = b.pq.read_table(OLD / "absolute_reconstruction_metrics.parquet", use_threads=False).to_pylist()
    old = {(q["checkpoint"], q["weights"], q["sample_id"], q["timestep"]): q for q in prior if q["checkpoint"] in b.CP}
    rows, heads, typed = (
        table("absolute_reconstruction_metrics.parquet"),
        table("head_contributions.parquet"),
        table("typed_state_statistics.parquet"),
    )
    done = {(q["checkpoint"], q["weights"], q["sample_id"], q["timestep"]) for q in rows}
    cache = b.CoordinateCache(Path(b.cfg["population_root"]), verify_payload=False)
    e = b.model_engine()
    for rec in records():
        assert sha(rec["path"]) == rec["sha256"]
        saved, _ = b.pinned_load(rec["path"])
        b.verify(saved, e)
        if rec["label"] == LATEST:
            total, norm = 0.0, 0.0
            for k in dict(e.model.named_parameters()):
                d = saved["model"][k].float() - saved["ema"][k].float()
                total += float(d.double().square().sum())
                norm += float(saved["model"][k].double().square().sum())
            b.write(
                "numerical/ema_parameter_distance.json",
                dict(
                    l2=math.sqrt(total),
                    relative_to_raw_l2=math.sqrt(total / norm),
                    rms=math.sqrt(total / 33128139),
                    residual_initialization_coefficient=b.cfg["ema_decay"] ** rec["position"]["update"],
                    decay=b.cfg["ema_decay"],
                    update=rec["position"]["update"],
                    buffer_note="EMA parameters mix initialization recursively; buffers are copied each step",
                ),
            )
        for weights in ("raw", "ema"):
            model = e.model if weights == "raw" else e.ema
            model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
            model.cuda().eval()
            for row in panel:
                label = rec["label"]
                if all((label, weights, row["sample_id"], t) in done for t in GRID):
                    continue
                original_primary = b.CP[-1]
                # Instrument both checkpoints with corrected tuple/input hooks.
                if label in (LATEST, "epoch_1_fraction_0.50"):
                    b.CP[-1] = label
                for out in base.measure(e, model, row, cache, label, weights, heads, typed):
                    key = (label, weights, row["sample_id"], out["timestep"])
                    if key in done:
                        continue
                    if key in old:
                        cached = old[key]
                        assert cached["noise_seed"] == out["noise_seed"] and cached["length"] == out["length"]
                        reused = [k for k in cached if k not in ("inference_backend", "metric_version")]
                        out["extension_inference_v_mse_minus_cached"] = out["v_mse"] - cached["v_mse"]
                        out.update({k: cached[k] for k in reused})
                        out["historical_reused"] = True
                    else:
                        out["historical_reused"] = False
                    out["metric_version"] = VERSION
                    out["rg_discrepancy"] = abs(out["radius_ratio"] - 1)
                    rows.append(out)
                    done.add(key)
                b.CP[-1] = original_primary
                b.table("absolute_reconstruction_metrics.parquet", rows)
                b.table("head_contributions.parquet", heads)
                if typed:
                    b.table("typed_state_statistics.parquet", typed)
                print(
                    json.dumps(
                        dict(checkpoint=label, weights=weights, identity=row["sample_id"], conditions=len(rows))
                    ),
                    flush=True,
                )
            model.cpu()
            torch.cuda.empty_cache()
        del saved
    assert len(rows) == 10000
    b.write(
        "logs/reconstruction_complete.json",
        dict(
            rows=len(rows),
            historical_scalar_conditions_reused=sum(q["historical_reused"] for q in rows),
            full_panel=True,
        ),
    )
    # Select sampling weights from the frozen complete panel; no checkpoint selection.
    means = {
        w: float(
            np.mean(
                [
                    q["v_mse"]
                    for q in rows
                    if q["checkpoint"] == LATEST and q["weights"] == w and q["split"] == "heldout"
                ]
            )
        )
        for w in ("raw", "ema")
    }
    choice = min(means, key=means.get)
    b.write(
        "sampler_weight_selection.json",
        dict(
            primary_weights=choice,
            heldout_v_mse=means,
            raw_minus_ema=means["raw"] - means["ema"],
            selection="Frozen100heldout equal-protein equal-timestep v-MSE; no generation-based selection",
        ),
    )
    b.write(
        "verdict.json",
        dict(
            reconstruction="PROMISING_BUT_IMMATURE",
            generation="INCONCLUSIVE",
            generalization="INCONCLUSIVE",
            continued_training="CONTINUE_BUT_WATCH",
            sampler_weights=choice,
        ),
    )
    # Extra500 panel is frozen development data, streamed without coordinate preload.
    dev = b.pq.read_table(
        Path(b.cfg["population_root"]) / "population/development.parquet", use_threads=False
    ).to_pylist()
    assert len(dev) == 500 and not train & {q["sample_id"] for q in dev}
    extra = table("numerical/heldout500_metrics.parquet")
    finished = {(q["weights"], q["sample_id"], q["timestep"]) for q in extra}
    saved, _ = b.pinned_load(selection["snapshot"])
    for weights in ("raw", "ema"):
        model = e.model if weights == "raw" else e.ema
        model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
        model.cuda().eval()
        for row0 in dev:
            if all((weights, row0["sample_id"], t) in finished for t in GRID):
                continue
            row = dict(row0, eval_split="heldout", pair_index=1, pair_id="heldout500:" + row0["sample_id"])
            for out in base.measure(e, model, row, cache, "heldout500_latest", weights, [], []):
                extra.append(out)
                finished.add((weights, row0["sample_id"], out["timestep"]))
            b.table("numerical/heldout500_metrics.parquet", extra)
        model.cpu()
        torch.cuda.empty_cache()
    b.write(
        "logs/heldout500_complete.json",
        dict(
            rows=len(extra),
            frozen_population=str(Path(b.cfg["population_root"]) / "population/development.parquet"),
            separate_noise_family="same fixed matched-noise implementation with heldout500:identity pair keys",
        ),
    )


def augment_intrinsic(x):
    out = b.intrinsic(x)
    adj = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    out.update(
        output_length=len(x),
        centering_error_angstrom=float(np.linalg.norm(x.mean(0))),
        disconnected_adjacent_count=int((adj > 2 * 4.2).sum()),
        fragmentation_definition="Adjacent>8.4Å descriptive indicator; not a frozen biological connectivity classifier",
        collapsed_coordinate_count=int(np.sum(np.linalg.norm(x - x.mean(0), axis=1) < 1e-6)),
    )
    return out


def score_vector(r):
    keys = [
        "adjacent_mean_angstrom",
        "adjacent_std_angstrom",
        "adjacent_valid_fraction",
        "clash_fraction",
        "radius_of_gyration_angstrom",
        "signed_triple_mean",
        "negative_signed_triple_fraction",
        "bond_angle_mean_radians",
        "long_range_contact_density",
        "long_range_contact_order",
    ]
    values = [r.get(k) if r.get(k) is not None else 0 for k in keys]
    return np.array(
        [
            *values,
            *r["covariance_eigenvalues_angstrom2"],
            r["offset_2_distance_quantiles_angstrom"][2],
            r["offset_3_distance_quantiles_angstrom"][2],
            *(r["pseudo_torsion_quantiles_radians"] or [float("nan")] * 5),
        ],
        float,
    )


b.score_vector = score_vector
b.quality_floor = np.array(
    [0.1, 0.1, 0.02, 0.001, 1, 0.03, 0.03, 0.03, 0.005, 0.02, 1, 1, 1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
)


@torch.no_grad()
def generation():
    _lease = base.acquire_lease()
    e = b.model_engine()
    saved, digest = b.pinned_load(selection["snapshot"])
    assert digest == SHA
    b.verify(saved, e)
    chosen = b.read("sampler_weight_selection.json")["primary_weights"]
    rows, manifest, trajectory = (
        table("generation_metrics.parquet"),
        table("generation_manifest.parquet"),
        table("trajectory_metrics.parquet"),
    )
    heads = table("head_contributions.parquet")
    typed = table("typed_state_statistics.parquet")
    for q in typed:
        q.setdefault("vector_norm_mean", None)
        q.setdefault("vector_norm_rms", None)
        q.setdefault("setting", "reconstruction")
    done = {r["sample_key"] for r in manifest}
    # Recover any completed sample whose plotting failed before table commit.
    for metrics_path in (OUT / "samples").glob("*/*/metrics.json"):
        recovered = b.read(str(metrics_path.relative_to(OUT)))
        meta, stat = recovered["meta"], recovered["metrics"]
        if meta["sample_key"] in done:
            continue
        assert meta["checkpoint_sha256"] == SHA and meta["same_saved_x499"]
        folder = metrics_path.parent
        xyz = np.load(folder / "final_coordinates.npz")["coordinates_angstrom"]
        frames = np.load(folder / "sparse_trajectory.npz")
        assert xyz.shape == (meta["requested_length"], 3) and np.isfinite(xyz).all()
        assert set(frames.files) == {f"t{t}" for t in FRAMES}
        original = np.load(meta["initial_gaussian_source"])["t499"]
        assert np.array_equal(frames["t499"], original)
        b.sample_images(meta, stat, {int(k[1:]): frames[k] for k in frames.files})
        manifest.append(meta)
        rows.append(
            dict(stat, **{k: meta[k] for k in ("sample_key", "requested_length", "seed", "weights", "primary")})
        )
        trajectory += recovered["trajectory"]
        for q in recovered["trajectory"]:
            heads.append(
                dict(
                    checkpoint=LATEST,
                    weights=meta["weights"],
                    identity=meta["sample_key"],
                    split="generation",
                    timestep=q["timestep"],
                    head_details=q["head_details"],
                )
            )
        vectors = np.load(folder / "vector_fields.npz")
        _, _, reliability = b.local_geometry(torch.from_numpy(vectors["xt_490"] / b.SCALE))
        gate = e.model.gate(torch.tensor(490), reliability)
        components = dict(
            global_v=torch.from_numpy(vectors["polar_490"]),
            chiral_effective=torch.from_numpy(vectors["chiral_490"]),
            local_effective=torch.from_numpy(vectors["local_490"]),
            gate=gate,
            reliability=reliability,
        )
        heads.append(
            dict(
                checkpoint=LATEST,
                weights=meta["weights"],
                identity=meta["sample_key"],
                split="generation",
                timestep=490,
                head_details=head_details(torch.from_numpy(vectors["total_490"]), components),
                recovered_from_saved_vectors=True,
            )
        )
        done.add(meta["sample_key"])
        b.write(
            "logs/recovered_completed_sample.json", dict(sample=meta["sample_key"], forward_sampling_not_repeated=True)
        )
    if manifest:
        b.table("generation_manifest.parquet", manifest)
        b.table("generation_metrics.parquet", rows)
        b.table("trajectory_metrics.parquet", trajectory)
        b.table("head_contributions.parquet", heads)
    b.write(
        "numerical/quality_score_contract.json",
        dict(
            definition="RMS deviation from matched-real median, divided by max(IQR/1.349, predeclared physical resolution floor); equal scalar terms",
            resolution_floors=b.quality_floor.tolist(),
            fields="adjacentmean/std/validfraction,clash,Rg,signedtriplemean/negativefraction,bondangle,contactdensity/order,covariance3eigenvalues,i+2/i+3median,torsion5quantiles",
            missing_contact_order="Zero contacts represented by order0; absence is separately represented in contact density",
            predeclared_before_generation=True,
            no_visual_selection=True,
        ),
    )
    old_manifest = b.pq.read_table(OLD / "generation_manifest.parquet", use_threads=False).to_pylist()
    for weights, count in [(chosen, 4), ("ema" if chosen == "raw" else "raw", 1)]:
        model = e.model if weights == "raw" else e.ema
        model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
        model.cuda().eval()
        for n in b.LENGTHS:
            for i in range(count):
                seed = 15015000 + i
                name = f"{weights}_n{n}_s{seed}"
                if name in done:
                    continue
                pulse("generation", name)
                historical = next(
                    q
                    for q in old_manifest
                    if q["requested_length"] == n and q["seed"] == seed and q["weights"] == "raw"
                )
                assert (
                    historical["checkpoint_sha256"]
                    == "2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03"
                )
                oldpath = Path(historical["path"]) / "sparse_trajectory.npz"
                initial_angstrom = np.load(oldpath)["t499"].astype(np.float32)
                # Use the saved tensor directly; no new Gaussian draw is substituted.
                current = torch.from_numpy(initial_angstrom.copy()).cuda()[None] / b.SCALE
                initial_hash = hashlib.sha256(initial_angstrom.tobytes()).hexdigest()
                mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
                length = torch.tensor([n], device="cuda")
                # Recover the original normalized FP32 representation and prove
                # that its serialized physical tensor equals the saved start.
                recovered = b.centered_coordinate_noise(
                    torch.empty(1, n, 3, device="cuda"),
                    mask,
                    generator=torch.Generator(device="cuda").manual_seed(seed),
                )
                assert np.array_equal((recovered[0] * b.SCALE).cpu().numpy(), initial_angstrom), (
                    "Saved Gaussian does not match exact historical seed/sampler contract"
                )
                roundtrip_max = float((current - recovered).abs().max())
                current = recovered
                normalized_initial_hash = hashlib.sha256(current[0].cpu().numpy().tobytes()).hexdigest()
                context = None
                frames, story, vector, stats = {}, {}, {}, []
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                for t0 in range(499, -1, -1):
                    pulse("generation", name + ":t" + str(t0))
                    t = torch.tensor([t0], device="cuda")
                    representative_seed = 1 if n == 48 and "raw_n48_s15015000" in done else 0
                    instrument = weights == chosen and i == representative_seed and t0 in (499, 490, 400, 250, 50, 0)
                    with (
                        capture(model, typed, dict(sample_id=name, setting="generation"), LATEST, t0, weights)
                        if instrument
                        else contextlib.nullcontext()
                    ):
                        v, comp = model.with_components(
                            current, t, length, mask, context, torch.tensor([context is not None], device="cuda")
                        )
                    nxt, xhat, _ = e.diffusion.deterministic_reverse_step(current, t, v, mask)
                    if t0 in FRAMES or t0 == 490:
                        xt = (current[0] * b.SCALE).cpu().numpy()
                        clean = (xhat[0] * b.SCALE).cpu().numpy()
                        xp = (nxt[0] * b.SCALE).cpu().numpy()
                        c = {k: z.detach().cpu() if isinstance(z, torch.Tensor) else z for k, z in comp[0].items()}
                        hd = head_details(v[0].cpu(), c)
                        alpha, sigma = e.diffusion.alpha_sigma(t, current)
                        record = dict(
                            augment_intrinsic(xp if t0 == 0 else xt),
                            sample_key=name,
                            timestep=t0,
                            alpha=float(alpha.flatten()[0]),
                            sigma=float(sigma.flatten()[0]),
                            logsnr=float(torch.log(alpha.square() / sigma.square()).flatten()[0]),
                            head_details=hd,
                            clean_estimate_rg_angstrom=augment_intrinsic(clean)["radius_of_gyration_angstrom"],
                        )
                        heads.append(
                            dict(
                                checkpoint=LATEST,
                                weights=weights,
                                identity=name,
                                split="generation",
                                timestep=t0,
                                head_details=hd,
                            )
                        )
                        if t0 in FRAMES:
                            frames[f"t{t0}"] = xp if t0 == 0 else (initial_angstrom if t0 == 499 else xt)
                            story.update({f"xt_{t0}": xt, f"x0hat_{t0}": clean, f"next_{t0}": xp})
                            stats.append(record)
                        if t0 in (490, 400, 250, 50):
                            vector.update(
                                {
                                    f"xt_{t0}": xt,
                                    f"total_{t0}": v[0].cpu().numpy(),
                                    f"polar_{t0}": c["global_v"].numpy(),
                                    f"chiral_{t0}": c["chiral_effective"].numpy(),
                                    f"local_{t0}": c["local_effective"].numpy(),
                                }
                            )
                    if not torch.isfinite(nxt).all():
                        b.write("logs/nonfinite_generation.json", dict(sample=name, timestep=t0))
                        raise FloatingPointError(name)
                    current, context = nxt, xhat.detach()
                xyz = (current[0] * b.SCALE).cpu().numpy()
                stat = augment_intrinsic(xyz)
                folder = OUT / "samples" / weights / name
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "final_coordinates.npz", coordinates_angstrom=xyz)
                np.savez_compressed(folder / "sparse_trajectory.npz", **frames)
                np.savez_compressed(folder / "inference_states.npz", **story)
                np.savez_compressed(folder / "vector_fields.npz", **vector)
                pdb = [
                    "REMARK CA TRACE ONLY; NO SEQUENCE; NO SIDE CHAINS",
                    "REMARK NOT ALL ATOM; NOT PROOF OF THERMODYNAMIC STABILITY",
                    f"REMARK E015 {weights} SHA256 {SHA}",
                ]
                pdb += [
                    f"ATOM  {j:5d}  CA  UNK A{j:4d}    {a:8.3f}{c:8.3f}{d:8.3f}  1.00  0.00           C"
                    for j, (a, c, d) in enumerate(xyz, 1)
                ]
                (folder / "trace_ca_only.pdb").write_text("\n".join(pdb + ["TER", "END"]) + "\n")
                meta = dict(
                    sample_key=name,
                    requested_length=n,
                    seed=seed,
                    weights=weights,
                    primary=weights == chosen,
                    checkpoint_sha256=SHA,
                    path=str(folder),
                    runtime_seconds=time.monotonic() - started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    initial_gaussian_source=str(oldpath),
                    initial_gaussian_sha256=initial_hash,
                    same_saved_x499=True,
                    normalized_gaussian_sha256=normalized_initial_hash,
                    initial_roundtrip_max_normalized=roundtrip_max,
                    historical_raw_path=historical["path"],
                    sampler="Production CoordinateVPDiffusion.deterministic_reverse_step;500steps499..0;FP32 as historical evaluator;generated self-conditioning",
                    initial_precision_note="Saved X499 in Å is authoritative. Exact historical normalized FP32 Gaussian recovered using the original seed/code and accepted only after bitwise equality of its serialized X499 to saved tensor. No different initial state permitted.",
                    terminal_frame="t0 final after update; other frames are pre-update",
                )
                b.write(str(folder.relative_to(OUT) / "sampler_configuration.json"), meta)
                b.write(str(folder.relative_to(OUT) / "metrics.json"), dict(meta=meta, metrics=stat, trajectory=stats))
                b.sample_images(meta, stat, {int(k[1:]): z for k, z in frames.items()})
                manifest.append(meta)
                rows.append(
                    dict(stat, **{k: meta[k] for k in ("sample_key", "requested_length", "seed", "weights", "primary")})
                )
                trajectory += stats
                done.add(name)
                b.table("generation_manifest.parquet", manifest)
                b.table("generation_metrics.parquet", rows)
                b.table("trajectory_metrics.parquet", trajectory)
                b.table("head_contributions.parquet", heads)
                b.table("typed_state_statistics.parquet", typed)
                print(
                    json.dumps(
                        dict(
                            sample=name,
                            seconds=meta["runtime_seconds"],
                            adjacent=stat["adjacent_mean_angstrom"],
                            valid=stat["adjacent_valid_fraction"],
                            rg=stat["radius_of_gyration_angstrom"],
                        )
                    ),
                    flush=True,
                )
        model.cpu()
        torch.cuda.empty_cache()
    b.write(
        "logs/generation_complete.json",
        dict(samples=len(manifest), finite_fraction=float(np.mean([q["finite_fraction"] for q in rows]))),
    )


def bootstrap(a, c, strata, draws=5000):
    a, c = np.array(a, float), np.array(c, float)
    rng = np.random.default_rng(15015)
    groups = [np.flatnonzero(np.array(strata) == s) for s in sorted(set(strata))]
    idx = np.concatenate([rng.choice(g, (draws, len(g)), replace=True) for g in groups], axis=1)
    ad, cd = a[idx].mean(1), c[idx].mean(1)
    am, cm = float(a.mean()), float(c.mean())
    ratio = np.divide(cd, ad, out=np.full_like(cd, np.nan), where=np.abs(ad) > 1e-12)
    return dict(
        train_mean=am,
        heldout_mean=cm,
        train_median=float(np.median(a)),
        heldout_median=float(np.median(c)),
        gap=cm - am,
        relative_gap=cm / am - 1 if abs(am) > 1e-12 else None,
        ratio=cm / am if abs(am) > 1e-12 else None,
        train_ci95=np.quantile(ad, [0.025, 0.975]).tolist(),
        heldout_ci95=np.quantile(cd, [0.025, 0.975]).tolist(),
        gap_ci95=np.quantile(cd - ad, [0.025, 0.975]).tolist(),
        ratio_ci95=np.nanquantile(ratio, [0.025, 0.975]).tolist() if np.isfinite(ratio).any() else None,
        draws=draws,
        proteins=len(a),
    )


def analysis():
    repair_typed_statistics()
    train_ids = set(
        b.pq.read_table(
            Path(b.cfg["population_root"]) / "population/train.parquet", columns=["sample_id"], use_threads=False
        )
        .column(0)
        .to_pylist()
    )
    held_ids = set(
        b.pq.read_table(
            Path(b.cfg["population_root"]) / "population/development.parquet", columns=["sample_id"], use_threads=False
        )
        .column(0)
        .to_pylist()
    )
    assert not train_ids & held_ids
    b.write(
        "numerical/full_population_split_proof.json",
        dict(
            train_count=len(train_ids),
            heldout_count=len(held_ids),
            overlap=0,
            frozen_population_hash=b.cfg["population_hash"],
            performance_selection=False,
        ),
    )
    rows = table("absolute_reconstruction_metrics.parquet")
    gaps, rawema, learning = [], [], []
    panel = b.read("panel_definitions.json")["rows"]
    grouped = {}
    for q in rows:
        grouped.setdefault((q["checkpoint"], q["weights"], q["timestep"], q["sample_id"]), q)
    for cp in b.CP:
        for w in ("raw", "ema"):
            for t in [None, *GRID]:
                for st in ["all", *b.STRATA]:
                    for metric in PRIMARY:
                        pairs = {}
                        for q in panel:
                            if st != "all" and q["length_stratum"] != st:
                                continue
                            conditions = [grouped[(cp, w, s, q["sample_id"])] for s in (GRID if t is None else [t])]
                            val = float(np.mean([r[metric] for r in conditions]))
                            pairs.setdefault(q["pair_id"], {})[q["eval_split"]] = (val, q["length_stratum"])
                        a = [p["train"][0] for p in pairs.values()]
                        c = [p["heldout"][0] for p in pairs.values()]
                        strata = [p["train"][1] for p in pairs.values()]
                        gaps.append(
                            dict(
                                checkpoint=cp,
                                weights=w,
                                timestep=t,
                                length_stratum=st,
                                metric=metric,
                                **bootstrap(a, c, strata),
                            )
                        )
            for split in ("train", "heldout"):
                for t in [None, *GRID]:
                    for metric in PRIMARY:
                        a, c, strata = [], [], []
                        for q in panel:
                            if q["eval_split"] != split:
                                continue
                            a.append(
                                float(
                                    np.mean(
                                        [
                                            grouped[(cp, "raw", s, q["sample_id"])][metric]
                                            for s in (GRID if t is None else [t])
                                        ]
                                    )
                                )
                            )
                            c.append(
                                float(
                                    np.mean(
                                        [
                                            grouped[(cp, "ema", s, q["sample_id"])][metric]
                                            for s in (GRID if t is None else [t])
                                        ]
                                    )
                                )
                            )
                            strata.append(q["length_stratum"])
                        rawema.append(
                            dict(
                                checkpoint=cp,
                                split=split,
                                timestep=t,
                                metric=metric,
                                raw_mean=float(np.mean(a)),
                                ema_mean=float(np.mean(c)),
                                ema_minus_raw=bootstrap(a, c, strata),
                            )
                        )
    positions = {q["label"]: q["position"] for q in records()}
    pairs = list(zip(b.CP[:-1], b.CP[1:], strict=True)) + [(b.CP[0], LATEST), ("epoch_1_fraction_0.50", LATEST)]
    for old, new in pairs:
        for w in ("raw", "ema"):
            for split in ("train", "heldout"):
                for metric in PRIMARY:
                    a, c, strata = [], [], []
                    for q in panel:
                        if q["eval_split"] != split:
                            continue
                        a.append(np.mean([grouped[(old, w, t, q["sample_id"])][metric] for t in GRID]))
                        c.append(np.mean([grouped[(new, w, t, q["sample_id"])][metric] for t in GRID]))
                        strata.append(q["length_stratum"])
                    learning.append(
                        dict(
                            old=old,
                            new=new,
                            weights=w,
                            split=split,
                            metric=metric,
                            old_position=positions[old],
                            new_position=positions[new],
                            **bootstrap(a, c, strata),
                        )
                    )
    b.table("train_vs_heldout_metrics.parquet", gaps)
    b.write(
        "generalization_bootstrap.json",
        dict(rows=gaps, draws=5000, unit="Stratified matched protein pairs; timesteps averaged within protein"),
    )
    b.table("raw_vs_ema_metrics.parquet", rawema)
    b.table("checkpoint_trajectory_metrics.parquet", learning)
    _, cache = base.inputs()
    samples = table("generation_manifest.parquet")
    generated = table("generation_metrics.parquet")
    base.distribution_analysis(samples, generated, cache)
    dist = b.read("generation_distribution_metrics.json")
    for d in dist["by_length"]:
        for q in d["metrics"]:
            q["generated_to_floor_ratio"] = (
                q["generated_vs_real"] / q["real_vs_real_floor"] if q["real_vs_real_floor"] > 1e-12 else None
            )
            q["excess_ratio"] = (
                q["excess_above_floor"] / q["real_vs_real_floor"] if q["real_vs_real_floor"] > 1e-12 else None
            )
    b.write("generation_distribution_metrics.json", dist)
    # Reuse old final coordinates and sampler trajectories, never resample old generation.
    compare = []
    for q in samples:
        if not q["primary"]:
            continue
        oldcoord = np.load(Path(q["historical_raw_path"]) / "final_coordinates.npz")["coordinates_angstrom"]
        oldstat = augment_intrinsic(oldcoord)
        newstat = next(r for r in generated if r["sample_key"] == q["sample_key"])
        compare.append(
            dict(
                sample=q["sample_key"],
                length=q["requested_length"],
                seed=q["seed"],
                old_weights="raw",
                latest_weights=q["weights"],
                initial_gaussian_sha256=q["initial_gaussian_sha256"],
                same_saved_start=True,
                old=oldstat,
                new=newstat,
                differences={
                    k: newstat[k] - oldstat[k]
                    for k in (
                        "adjacent_valid_fraction",
                        "clash_fraction",
                        "radius_of_gyration_angstrom",
                        "signed_triple_mean",
                        "negative_signed_triple_fraction",
                        "long_range_contact_density",
                    )
                },
            )
        )
    b.write(
        "same_seed_comparison/metrics.json",
        dict(
            rows=compare,
            old_generation_reused=True,
            scope="Same saved physical X499; historical FP32->Å serialization round-trip noted; weight choices explicitly labelled",
        ),
    )
    supplemental_analysis(rows, cache, generated, samples, compare)
    prim = [
        q
        for q in gaps
        if q["checkpoint"] == LATEST
        and q["weights"] == b.read("sampler_weight_selection.json")["primary_weights"]
        and q["timestep"] is None
        and q["length_stratum"] == "all"
    ]
    v = {q["metric"]: q for q in prim}
    verdict = b.read("verdict.json")
    under = (
        min(v["local_rmse_angstrom"]["train_mean"], v["local_rmse_angstrom"]["heldout_mean"]) > 1
        and min(v["torsion_angular_mae_radians"]["train_mean"], v["torsion_angular_mae_radians"]["heldout_mean"]) > 0.7
    )
    verdict["generalization"] = (
        "UNDERFIT"
        if under
        else "MILD_OVERFIT"
        if v["v_mse"]["gap_ci95"][0] > 0 and v["v_mse"]["relative_gap"] > 0.15
        else "GENERALIZING"
        if v["local_rmse_angstrom"]["heldout_mean"] < 1
        else "INCONCLUSIVE"
    )
    verdict.update(
        primary_sha256=SHA,
        primary_position=selection["source"]["position"],
        raw_ema=b.read("sampler_weight_selection.json"),
        ema_parameters=b.read("numerical/ema_parameter_distance.json"),
    )
    b.write("verdict.json", verdict)
    b.write(
        "logs/analysis_complete.json",
        dict(gap_rows=len(gaps), learning_rows=len(learning), same_seed_pairs=len(compare)),
    )


def supplemental_analysis(rows, cache, generated, samples, compare):
    panel = b.read("panel_definitions.json")
    panel.update(
        reason="Existing frozen identities reused; complete100TRAIN+100heldout after clean production pause",
        executed_panel_count=dict(train=100, heldout=100),
        historical_execution_notes=panel.get("reason"),
    )
    b.write("panel_definitions.json", panel)
    heads = table("head_contributions.parquet")
    summaries = []
    for cp, w, split, t in sorted({(q["checkpoint"], q["weights"], q["split"], q["timestep"]) for q in heads}):
        group = [q for q in heads if (q["checkpoint"], q["weights"], q["split"], q["timestep"]) == (cp, w, split, t)]
        result = dict(checkpoint=cp, weights=w, split=split, timestep=t, proteins=len(group), heads={})
        for name in ("polar", "chiral", "local"):
            values = [q["head_details"]["rms"][name] for q in group]
            ratios = [
                q["head_details"]["rms"][name] / q["head_details"]["final_rms"]
                for q in group
                if q["head_details"]["final_rms"] > 1e-12
            ]
            cosines = [
                q["head_details"]["cosine_final"][name]
                for q in group
                if q["head_details"]["cosine_final"][name] is not None
            ]
            result["heads"][name] = dict(
                mean_rms=float(np.mean(values)),
                median_rms=float(np.median(values)),
                mean_ratio_to_final=float(np.mean(ratios)) if ratios else None,
                mean_cosine_final=float(np.mean(cosines)) if cosines else None,
                cosine_assessed=len(cosines),
            )
        result["gate_equal_protein_mean_p10_p50_p90"] = np.mean(
            [q["head_details"]["gate_p10_p50_p90"] for q in group], axis=0
        ).tolist()
        result["gate_mean"] = float(np.mean([q["head_details"]["gate_mean"] for q in group]))
        result["cosine_pairs"] = {
            k: float(
                np.mean(
                    [
                        q["head_details"]["cosine_pairs"][k]
                        for q in group
                        if q["head_details"]["cosine_pairs"][k] is not None
                    ]
                )
            )
            if any(q["head_details"]["cosine_pairs"][k] is not None for q in group)
            else None
            for k in ("polar_chiral", "polar_local", "chiral_local")
        }
        summaries.append(result)
    b.write(
        "numerical/head_summary_by_timestep.json",
        dict(rows=summaries, weighting="Equal protein; gate summaries average per-protein node quantiles"),
    )
    novelty = b.read("diversity_novelty_metrics.json")
    for q in novelty["novelty"]:
        for ref in q["nearest"].values():
            identity = ref["descriptor_nearest_identity"]
            index = cache.positions[identity]
            ref["reference_path"] = str(cache.root / "coordinates.f32")
            ref["coordinate_row_range"] = [int(cache.offsets[index]), int(cache.offsets[index + 1])]
    for q in novelty["diversity"]:
        q["near_duplicates"] = sum(
            p["kabsch_rmsd_angstrom"] < 0.5 and p["distance_matrix_rmse_angstrom"] < 0.5 for p in q["pairs"]
        )
        q["near_duplicate_rate"] = q["near_duplicates"] / len(q["pairs"])
        q["near_duplicate_definition"] = "Descriptive pairwise proper-rotation Kabsch<0.5Å and distance-matrixRMSE<0.5Å"
    b.write("diversity_novelty_metrics.json", novelty)
    trajectories = table("trajectory_metrics.parquet")
    events = []
    paired_distribution = []
    descriptor_parts = []
    scores = table("generation_scores.parquet")
    for n in b.LENGTHS:
        pulse("supplemental_analysis", str(n))
        rr, _ = base.refs(cache, n, 8)
        median_rg = float(np.median([q["radius_of_gyration_angstrom"] for q in rr]))
        center = np.median(np.array([score_vector(q) for q in rr]), axis=0)
        scale = np.maximum(
            (
                np.quantile(np.array([score_vector(q) for q in rr]), 0.75, axis=0)
                - np.quantile(np.array([score_vector(q) for q in rr]), 0.25, axis=0)
            )
            / 1.349,
            b.quality_floor,
        )
        for q in generated:
            if q["requested_length"] == n and not q["primary"]:
                vector = score_vector(q)
                scores.append(
                    dict(
                        sample_key=q["sample_key"],
                        score=float(np.sqrt(np.mean(((vector - center) / scale) ** 2)))
                        if np.isfinite(vector).all()
                        else None,
                        score_assessable=bool(np.isfinite(vector).all()),
                        adjacent_mean_angstrom=q["adjacent_mean_angstrom"],
                        clash_fraction=q["clash_fraction"],
                        rg_angstrom=q["radius_of_gyration_angstrom"],
                    )
                )
        for meta in [q for q in samples if q["requested_length"] == n]:
            trace = sorted(
                [q for q in trajectories if q["sample_key"] == meta["sample_key"]], key=lambda q: -q["timestep"]
            )
            for q in trace:
                q["actual_rg_to_real_median"] = q["radius_of_gyration_angstrom"] / median_rg
                q["gross_collapse_flag"] = q["actual_rg_to_real_median"] < 0.25
                q["gross_expansion_flag"] = q["actual_rg_to_real_median"] > 4
            events.append(
                dict(
                    sample=meta["sample_key"],
                    collapse_onset=next((q["timestep"] for q in trace if q["gross_collapse_flag"]), None),
                    expansion_onset=next((q["timestep"] for q in trace if q["gross_expansion_flag"]), None),
                    threshold_note="Descriptive actual-stateRg<0.25 or>4realmedian; high-noise states need not match clean distribution",
                )
            )
        pairs = [q for q in compare if q["length"] == n]
        raw_a = np.array([q["descriptor"] for q in rr[::2]])
        raw_b = np.array([q["descriptor"] for q in rr[1::2]])
        standard = b.ReferenceStandardizer(raw_a, raw_b)
        real = standard.transform(raw_b)
        h = b.bandwidth(standard.transform(raw_a), real)
        projections = b.projections(len(standard.columns))
        spectrum = [
            i
            for i, column in enumerate(standard.columns)
            if b.NAMES[column] in ("eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3")
        ]
        oldraw = np.array([q["old"]["descriptor"] for q in pairs])
        newraw = np.array([q["new"]["descriptor"] for q in pairs])
        oldcmp = b.Comparison(standard.transform(oldraw), real, oldraw, raw_b, h, projections, spectrum)
        newcmp = b.Comparison(standard.transform(newraw), real, newraw, raw_b, h, projections, spectrum)
        oldmetrics, newmetrics = oldcmp.metrics(), newcmp.metrics()
        for part in ("local_polymer", "global_shape"):
            cols = [
                j
                for j, col in enumerate(standard.columns)
                if (b.NAMES[col].startswith(("sep_1_1_", "sep_2_2_", "sep_3_3_"))) == (part == "local_polymer")
            ]
            a0 = standard.transform(raw_a)[:, cols]
            rb = real[:, cols]
            oldpart = standard.transform(oldraw)[:, cols]
            newpart = standard.transform(newraw)[:, cols]
            bandwidth = b.bandwidth(a0, rb)
            proj = b.projections(len(cols))
            comparisons = {
                name: b.Comparison(z, rb, raw, raw_b, bandwidth, proj, list(range(len(cols))))
                for name, z, raw in [("old", oldpart, oldraw), ("latest", newpart, newraw), ("floor", a0, raw_a)]
            }
            metrics = {name: q.metrics() for name, q in comparisons.items()}
            keys = ("energy", "mmd2", "sliced_wasserstein")
            delta = {k: [] for k in keys}
            rng_part = np.random.default_rng(15015)
            for _ in range(5000):
                ig, ir = rng_part.integers(0, 4, 4), rng_part.integers(0, 4, 4)
                old_draw, new_draw = (
                    base.population_metrics(comparisons["old"], ig, ir),
                    base.population_metrics(comparisons["latest"], ig, ir),
                )
                for k in keys:
                    delta[k].append(new_draw[k] - old_draw[k])
            descriptor_parts.append(
                dict(
                    length=n,
                    part=part,
                    columns=[b.NAMES[standard.columns[j]] for j in cols],
                    metrics=[
                        dict(
                            metric=k,
                            old=metrics["old"][k],
                            latest=metrics["latest"][k],
                            real_real_floor=metrics["floor"][k],
                            difference=metrics["latest"][k] - metrics["old"][k],
                            difference_ci95=np.quantile(delta[k], [0.025, 0.975]).tolist(),
                        )
                        for k in keys
                    ],
                )
            )
        draws = {k: [] for k in oldmetrics}
        rng = np.random.default_rng(15015)
        for _ in range(5000):
            ix, ir = rng.integers(0, 4, 4), rng.integers(0, 4, 4)
            a, z = base.population_metrics(oldcmp, ix, ir), base.population_metrics(newcmp, ix, ir)
            for k in draws:
                draws[k].append(z[k] - a[k])
        for j, q in enumerate(pairs):

            def contribution(raw, standard=standard, j=j, real=real):
                transformed = standard.transform(raw)
                return float(
                    2 * b.distances(transformed[j : j + 1], real).mean()
                    - b.distances(transformed[j : j + 1], transformed).mean()
                    - b.distances(real, real).mean()
                )

            q["old_energy_contribution"] = contribution(oldraw)
            q["latest_energy_contribution"] = contribution(newraw)
        paired_distribution.append(
            dict(
                length=n,
                metrics=[
                    dict(
                        metric=k,
                        old=oldmetrics[k],
                        latest=newmetrics[k],
                        difference=newmetrics[k] - oldmetrics[k],
                        difference_ci95=np.quantile(draws[k], [0.025, 0.975]).tolist(),
                    )
                    for k in draws
                ],
                bootstrap="5000 paired generated-seed resamples and shared real-reference resamples; fixed reference standardizer",
            )
        )
    b.table("trajectory_metrics.parquet", trajectories)
    b.table("generation_scores.parquet", scores)
    b.write("numerical/trajectory_events.json", dict(rows=events))
    b.write("same_seed_comparison/distribution_differences.json", dict(by_length=paired_distribution))
    b.write(
        "same_seed_comparison/local_vs_global_descriptor_differences.json",
        dict(
            rows=descriptor_parts,
            partition="Local=chain separations1,2,3 mean/std; global=remaining invariant scale/shape/pair/nonlocal chain-separation descriptors; fixed real standardization and5000pairedseedbootstrap",
        ),
    )
    b.write(
        "same_seed_comparison/metrics.json",
        dict(rows=compare, old_generation_reused=True, exact_saved_start_verified=True),
    )
    strength = []
    choice = b.read("sampler_weight_selection.json")["primary_weights"]
    for t in GRID:
        for st in b.STRATA:
            for split in ("train", "heldout"):
                group = [
                    q
                    for q in rows
                    if q["checkpoint"] == LATEST
                    and q["weights"] == choice
                    and q["timestep"] == t
                    and q["length_stratum"] == st
                    and q["split"] == split
                ]
                local = float(np.mean([q["local_rmse_angstrom"] for q in group]))
                rmsd = float(np.mean([q["aligned_rmsd_angstrom"] for q in group]))
                category = (
                    "STRONG"
                    if local < 0.5 and rmsd < 1
                    else "MODERATE"
                    if local < 1.5 and rmsd < 3
                    else "FAILED"
                    if np.mean([q["direct_x0_collapse_count"] for q in group]) > 0.5
                    else "WEAK"
                )
                strength.append(
                    dict(
                        setting="paired reconstruction",
                        timestep=t,
                        stratum=st,
                        split=split,
                        classification=category,
                        local_rmse_angstrom=local,
                        kabsch_angstrom=rmsd,
                        scope="Descriptive absolute paired-clean error; high-noise conditional uncertainty is not sampler collapse",
                    )
                )
    b.write(
        "numerical/strength_weakness_map.json",
        dict(
            rows=strength,
            thresholds="STRONGlocal<0.5Å/Kabsch<1Å;MODERATElocal<1.5Å/Kabsch<3Å;FAILEDpairedcollapse>50%;otherwiseWEAK;descriptive not thermodynamic",
        ),
    )


@torch.no_grad()
def repair_typed_statistics():
    """Bounded hook replay ensures vector-norm columns survive Arrow inference."""
    existing = table("typed_state_statistics.parquet")
    if existing and all(q.get("vector_norm_mean") is not None for q in existing if q["type"] in ("V", "A")):
        return
    _lease = base.acquire_lease()
    panel = [
        q for q in b.read("panel_definitions.json")["rows"] if q["eval_split"] == "heldout" and q["pair_index"] == 0
    ]
    cache = b.CoordinateCache(Path(b.cfg["population_root"]), verify_payload=False)
    engine = b.model_engine()
    typed = []
    for rec in records():
        if rec["label"] not in (LATEST, "epoch_1_fraction_0.50"):
            continue
        saved, _ = b.pinned_load(rec["path"])
        for weights in ("raw", "ema"):
            model = engine.model if weights == "raw" else engine.ema
            model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
            model.cuda().eval()
            previous = b.CP[-1]
            b.CP[-1] = rec["label"]
            for row in panel:
                list(base.measure(engine, model, row, cache, rec["label"], weights, [], typed))
            b.CP[-1] = previous
            model.cpu()
            torch.cuda.empty_cache()
        del saved
    typed += [q for q in existing if q.get("setting") == "generation"]
    b.table("typed_state_statistics.parquet", typed)
    b.write(
        "logs/typed_hook_schema_verified.json",
        dict(
            rows=len(typed),
            bounded_hook_replay=True,
            identities_per_checkpoint_weights=5,
            full_hidden_tensors_saved=False,
        ),
    )


def main(phase):
    pulse(phase)
    if phase == "reconstruction":
        reconstruction()
    elif phase == "generation":
        generation()
    else:
        import e015_latest_visuals as visual

        if phase == "analysis":
            analysis()
        elif phase == "visuals":
            visual.visuals(sys.modules[__name__])
        elif phase == "report":
            visual.report(sys.modules[__name__])
