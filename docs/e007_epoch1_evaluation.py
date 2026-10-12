"""Serialized E007 epoch-one evaluation using existing E015 frozen metric functions."""

import os, sys, json, time, pathlib, types, hashlib, math

W = pathlib.Path("/mnt/d/Simone/proteinGen/.e014_worktree")
C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
E = pathlib.Path(
    "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/evaluations/latest_complete_u10382_4951d838_v1"
)
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = "1"
for k in ["TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_HOME"]:
    p = C / "cache" / k.lower()
    p.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(p)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import e007_epoch1_mission as mission

mission.helper().host(True)
import fcntl

leases = []
for lockpath in [W / "e007_matched_fullscale_v1/OWNER.lock", E.parent.parent / "GPU_WORKER.lock"]:
    lease = lockpath.open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    leases.append(lease)

import e007_comparison_metrics as b

np, torch = b.np, b.torch
sys.path[:0] = [str(W / "src"), str(W / "scripts")]
os.chdir(W)
import e007_fullscale_runtime as rt
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, centered_coordinate_noise

cfg, pop, train, dev = rt.verified_inputs()
D = CoordinateVPDiffusion(500)
cache = b.CoordinateCache(rt.DATA, verify_payload=False)
checkpoint = json.loads((C / "E007_EPOCH1_CHECKPOINT.json").read_text())
cp = pathlib.Path(checkpoint["checkpoint"])
assert b.sha(cp) == checkpoint["sha256"]
saved = torch.load(cp, map_location="cpu", weights_only=False)
assert int(saved["exposures"].sum()) == 230132 and bool((saved["exposures"] == 1).all())
model = E007ScaleupControl(**rt.yaml.safe_load(rt.MODEL_CONFIG.read_text())["model"])
model.load_state_dict(saved["model"], strict=True)
del saved
model = model.cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
S = cfg["coordinate_scale_angstrom"]
assert S == b.SCALE
FR = [499, 475, 450, 425, 400, 350, 300, 250, 200, 150, 100, 50, 0]


def put(name, x):
    b.write(name, x)


def guard(unit):
    if time.monotonic() - getattr(guard, "last", 0) < 30:
        return
    guard.last = time.monotonic()
    r = mission.helper().host(False)
    assert r["swap_used_bytes"] < 256 * 1024**2
    owners = [int(x.split(",")[0]) for x in r["cuda_compute_apps"].splitlines() if x.strip()]
    assert not any(p != os.getpid() for p in owners)
    with (C / "resource_monitor.jsonl").open("a") as f:
        f.write(json.dumps(dict(unit=unit, **r)) + "\n")


def pdb(path, x):
    text = "REMARK C-alpha trace only; no sequence or side chains; not proof of stability\n"
    for i, (a, c, d) in enumerate(x, 1):
        text += f"ATOM  {i:5d}  CA  GLY A{i:4d}    {a:8.3f}{c:8.3f}{d:8.3f}  1.00  0.00           C\n"
    path.write_text(text + "TER\nEND\n")


@torch.no_grad()
def reconstruct():
    panel = json.loads((E / "panel_definitions.json").read_text())["rows"]
    assert len(panel) == 200
    train_ids = {r["sample_id"] for r in train}
    held_ids = {r["sample_id"] for r in dev}
    assert (
        sum(r["eval_split"] == "train" for r in panel) == 100
        and sum(r["eval_split"] == "heldout" for r in panel) == 100
    )
    assert {r["sample_id"] for r in panel if r["eval_split"] == "train"} <= train_ids
    assert {r["sample_id"] for r in panel if r["eval_split"] == "heldout"} <= held_ids
    assert not train_ids & held_ids

    path = C / "reconstruction_units.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done = {(x["sample_id"], x["timestep"]) for x in rows}
    for row in panel:
        guard(row["sample_id"])
        x, n, m = b.coordinate_batch([row], cache, S, device="cpu")
        for t0 in b.GRID:
            if (row["sample_id"], t0) in done:
                continue
            t = torch.tensor([t0])
            seed = b.noise_seed("e015-absolute-matched-noise-v1", row["pair_id"], t0)
            batch = D.make_training_batch(x, m, timesteps=t, generator=torch.Generator().manual_seed(seed))
            v, comp = model.telemetry(batch.noisy_coordinates.cuda(), t.cuda(), n.cuda(), m.cuda())
            v = v.cpu()
            comp = {k: z.cpu() if isinstance(z, torch.Tensor) else z for k, z in comp[0].items()}
            nxt, x0, _ = D.deterministic_reverse_step(batch.noisy_coordinates, t, v, m)
            out = b.extended_metrics(x0[0], x[0], v[0], batch.coordinate_v_target[0], S, comp)
            out.update(b.vector_metrics(v, batch.coordinate_v_target, m, "v"))
            oracle = b.paired_state(D, x, batch.coordinate_noise, t - 1, m)
            noop = float((batch.noisy_coordinates - oracle).square()[m].mean())
            step = float((nxt - oracle).square()[m].mean())
            out.update(b.vector_metrics(nxt - batch.noisy_coordinates, oracle - batch.noisy_coordinates, m, "step"))
            out.update(
                step_gain=1 - step / noop if noop else None,
                next_step_coordinate_mse_normalized=step,
                next_step_coordinate_mse_angstrom2=step * S * S,
                noop_next_step_mse_normalized=noop,
                centered_euclidean_rmsd_angstrom=float(((x0 - x) * S).square().sum(-1).mean().sqrt()),
                v_predicted_rms=out["v_predicted_norm"] / math.sqrt(3 * row["length"]),
                v_target_rms=out["v_target_norm"] / math.sqrt(3 * row["length"]),
            )
            for key in [
                "global_output_norm",
                "local_effective_output_norm",
                "local_global_norm_ratio",
                "global_local_cosine",
                "local_gate_mean",
                "local_gate_median",
            ]:
                out[key] = None
            out["head_components"] = (
                "NOT_APPLICABLE: adapter geometry telemetry is not a learned polar/chiral/local head decomposition"
            )
            for k in [1, 2, 3]:
                out[f"offset_{k}_rmse_angstrom"] = out[f"offset_{k}_mse_angstrom2"] ** 0.5

            out.update(b.intrinsic((x0[0] * S).numpy()))
            out["local_rmse_angstrom"] = out["local_mse_angstrom2"] ** 0.5
            out["rg_discrepancy"] = abs(out["radius_ratio"] - 1)
            out.update(
                sample_id=row["sample_id"],
                split=row["eval_split"],
                pair_id=row["pair_id"],
                length_stratum=row["length_stratum"],
                timestep=t0,
                noise_seed=seed,
                checkpoint="E007_epoch1",
                weights="raw",
                checkpoint_sha256=checkpoint["sha256"],
                centered_coordinate_rmse_angstrom=float(((x0 - x) * S).square().mean().sqrt()),
                covariance_trace_ratio=out["radius_ratio"] ** 2,
                torsion_angular_mae_degrees=out["torsion_angular_mae_radians"] * 180 / math.pi
                if out["torsion_angular_mae_radians"] is not None
                else None,
            )
            assert out["finite"]
            rows.append(out)
            with path.open("a") as f:
                f.write(json.dumps(b.clean(out)) + "\n")
                f.flush()
                os.fsync(f.fileno())
        put("progress.json", dict(stage="E007_RECONSTRUCTION", completed=len(rows), total=1000))
    b.table("e007_reconstruction.parquet", rows)
    put(
        "E007_RECONSTRUCTION.json",
        dict(
            status="COMPLETE",
            conditions=len(rows),
            panel_sha256=b.sha(E / "panel_definitions.json"),
            same_noise_family=True,
            schedule="Shared CoordinateVPDiffusion cosine500; identical normalization",
            EMA="NOT_APPLICABLE",
            parity_heads="NOT_APPLICABLE",
        ),
    )


@torch.no_grad()
def generate():
    old = E.parent / "absolute_generation_u3811_2986d44e_v1"
    historical = b.pq.read_table(old / "generation_manifest.parquet", use_threads=False).to_pylist()
    manifest = []
    stats = []
    for n in b.LENGTHS:
        for seed in range(15015000, 15015004):
            key = f"raw_n{n}_s{seed}"
            folder = C / "e007_samples" / key
            folder.mkdir(exist_ok=True)
            done = folder / "COMPLETE.json"
            if done.exists():
                q = json.loads(done.read_text())
                assert q["checkpoint_sha256"] == checkpoint["sha256"] and all(
                    b.sha(p) == h for p, h in q["files"].items()
                )
                manifest.append(q)
                stats.append(json.loads((folder / "metrics.json").read_text()))
                continue
            guard(key)
            ref = next(
                q for q in historical if q["requested_length"] == n and q["seed"] == seed and q["weights"] == "raw"
            )
            start = np.load(pathlib.Path(ref["path"]) / "sparse_trajectory.npz")["t499"].astype(np.float32)
            current = torch.tensor(start, device="cuda")[None] / S
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            length = torch.tensor([n], device="cuda")
            recovered = centered_coordinate_noise(
                torch.empty(1, n, 3, device="cuda"), mask, generator=torch.Generator(device="cuda").manual_seed(seed)
            )
            assert np.array_equal((recovered[0] * S).cpu().numpy(), start), (
                "Historical X499 identity failed; no replacement noise permitted"
            )
            current = recovered

            frames = {}
            clean = {}
            metrics = []
            beg = time.monotonic()
            for t0 in range(499, -1, -1):
                t = torch.tensor([t0], device="cuda")
                v = model(current, t, length, mask)
                nxt, xhat, _ = D.deterministic_reverse_step(current, t, v, mask)
                assert bool(torch.isfinite(nxt).all())
                if t0 in FR:
                    frames[f"t{t0}"] = (current[0] * S).cpu().numpy()
                    clean[f"t{t0}"] = (xhat[0] * S).cpu().numpy()
                    metrics.append(dict(timestep=t0, **b.intrinsic(frames[f"t{t0}"])))
                current = nxt
                if t0 % 50 == 0:
                    guard(key + ":" + str(t0))
            final = (current[0] * S).cpu().numpy()
            frames["t0"] = final
            metrics = [q for q in metrics if q["timestep"] != 0] + [dict(timestep=0, **b.intrinsic(final))]
            stat = b.intrinsic(final)
            stat.update(sample_key=key, requested_length=n, seed=seed, weights="raw", primary=True)
            np.savez_compressed(folder / "final_coordinates.npz", coordinates=final)
            np.savez_compressed(folder / "sparse_trajectory.npz", **frames)
            np.savez_compressed(folder / "direct_x0.npz", **clean)
            pdb(folder / "final_ca.pdb", final)
            for name, obj in [("metrics.json", stat), ("trajectory_metrics.json", metrics)]:
                mission.write(folder / name, b.clean(obj))
            q = dict(
                sample_key=key,
                requested_length=n,
                seed=seed,
                weights="raw",
                primary=True,
                path=str(folder),
                checkpoint_sha256=checkpoint["sha256"],
                initial_gaussian_sha256=hashlib.sha256(start.tobytes()).hexdigest(),
                sampler="Native CoordinateVPDiffusion.deterministic_reverse_step;500steps;no self-conditioning/teacher/target",
                seconds=time.monotonic() - beg,
            )
            q["files"] = {str(p): b.sha(p) for p in folder.iterdir() if p.is_file()}
            for name in q["files"]:
                with pathlib.Path(name).open("rb") as f:
                    os.fsync(f.fileno())
            mission.write(done, q)
            manifest.append(q)
            stats.append(stat)
            b.table("generation_manifest.parquet", manifest)
            b.table("generation_metrics.parquet", stats)
            put("progress.json", dict(stage="E007_GENERATION", completed=len(stats), total=20))
    put(
        "E007_GENERATION.json",
        dict(
            status="COMPLETE",
            samples=len(stats),
            finite_fraction=float(np.mean([q["finite_fraction"] for q in stats])),
            normalization=S,
            checkpoint_sha256=checkpoint["sha256"],
            head_analysis="NOT_APPLICABLE",
        ),
    )


if __name__ == "__main__":
    {"reconstruction": reconstruct, "generation": generate}[sys.argv[1]]()
