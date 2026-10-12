"""Bounded exposure-matched E015 epoch-one context; existing sampler and starts."""

import os, sys, pathlib, types, json, time, fcntl, hashlib

C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
OUT = C / "e015_epoch1_context"
OUT.mkdir(exist_ok=True)
W = pathlib.Path("/mnt/d/Simone/proteinGen/.e015_worktree")
ROOT = pathlib.Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")
LATEST = ROOT / "evaluations/latest_complete_u10382_4951d838_v1"
OLD = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
import e007_epoch1_mission as mission

mission.helper().host(True)
leases = []
for p in [ROOT / "GPU_WORKER.lock", mission.R / "OWNER.lock"]:
    l = p.open("a+")
    fcntl.flock(l, fcntl.LOCK_EX | fcntl.LOCK_NB)
    leases.append(l)
source = (
    (W / "docs/e015_serial_helpers.py")
    .read_text()
    .replace('OUT = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"', "OUT = Path(" + repr(str(OUT)) + ")")
)
argv = sys.argv[:]
sys.argv = ["metric_helper", "generate"]
b = types.ModuleType("E015_metrics")
exec(compile(source, "existing_E015_science", "exec"), b.__dict__)
sys.argv = argv
np, torch = b.np, b.torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cp = ROOT / "production/checkpoints/epoch_1_fraction_1.00.pt"
digest = b.sha(cp)
record = next(r for r in json.loads((LATEST / "checkpoint_inventory.json").read_text()) if r["path"] == str(cp))
assert digest == record["sha256"] and record["position"]["exposures"] == 230132

saved = torch.load(cp, map_location="cpu", weights_only=False)
assert saved["position"]["exposures"] == 230132
engine = b.model_engine()
b.verify(saved, engine)
oldrec = b.pq.read_table(LATEST / "absolute_reconstruction_metrics.parquet", use_threads=False).to_pylist()
means = {
    w: np.mean(
        [
            r["v_mse"]
            for r in oldrec
            if r["checkpoint"] == "epoch_1_fraction_1.00" and r["weights"] == w and r["split"] == "heldout"
        ]
    )
    for w in ["raw", "ema"]
}
weights = "raw" if means["raw"] < means["ema"] else "ema"
model = engine.model if weights == "raw" else engine.ema
model.load_state_dict(saved["model" if weights == "raw" else "ema"], strict=True)
del saved
model = model.cuda().eval()
historical = b.pq.read_table(OLD / "generation_manifest.parquet", use_threads=False).to_pylist()
samples = []
stats = []
FR = [499, 475, 450, 425, 400, 350, 300, 250, 200, 150, 100, 50, 0]


@torch.no_grad()
def run():
    for n in b.LENGTHS:
        for seed in range(15015000, 15015004):
            key = f"raw_n{n}_s{seed}"
            folder = OUT / "samples" / key
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / "COMPLETE.json"
            if done.exists():
                meta = json.loads(done.read_text())
                assert meta["checkpoint_sha256"] == digest and all(b.sha(p) == h for p, h in meta["files"].items())
                samples.append(meta)
                stats.append(json.loads((folder / "metrics.json").read_text()))
                continue
            before = mission.helper().host(False)
            assert not any(
                int(r.split(",")[0]) != os.getpid() for r in before["cuda_compute_apps"].splitlines() if r.strip()
            )
            old = next(
                r for r in historical if r["requested_length"] == n and r["seed"] == seed and r["weights"] == "raw"
            )
            initial = np.load(pathlib.Path(old["path"]) / "sparse_trajectory.npz")["t499"].astype(np.float32)
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            length = torch.tensor([n], device="cuda")
            current = b.centered_coordinate_noise(
                torch.empty(1, n, 3, device="cuda"), mask, generator=torch.Generator(device="cuda").manual_seed(seed)
            )
            assert np.array_equal((current[0] * b.SCALE).cpu().numpy(), initial)
            context = None
            frames = {}
            clean = {}
            beg = time.monotonic()
            for t0 in range(499, -1, -1):
                t = torch.tensor([t0], device="cuda")
                v, _ = model.with_components(
                    current, t, length, mask, context, torch.tensor([context is not None], device="cuda")
                )
                nxt, xhat, _ = engine.diffusion.deterministic_reverse_step(current, t, v, mask)
                assert bool(torch.isfinite(nxt).all())
                if t0 in FR:
                    frames[f"t{t0}"] = (nxt[0] if t0 == 0 else current[0]).cpu().numpy() * b.SCALE
                    clean[f"t{t0}"] = xhat[0].cpu().numpy() * b.SCALE
                current, context = nxt, xhat.detach()
                if t0 % 50 == 0 and time.monotonic() - getattr(run, "last_monitor", 0) >= 30:
                    run.last_monitor = time.monotonic()
                    resource = mission.helper().host(False)
                    assert not any(
                        int(r.split(",")[0]) != os.getpid()
                        for r in resource["cuda_compute_apps"].splitlines()
                        if r.strip()
                    )
            final = current[0].cpu().numpy() * b.SCALE
            stat = b.intrinsic(final)
            stat.update(sample_key=key, requested_length=n, seed=seed, weights=weights, primary=True)
            np.savez_compressed(folder / "final_coordinates.npz", coordinates_angstrom=final)
            pdb = ["REMARK C-alpha trace only; no sequence or side chains; not proof of thermodynamic stability"]
            for i, (a, c, d) in enumerate(final, 1):
                pdb.append(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {a:8.3f}{c:8.3f}{d:8.3f}  1.00  0.00           C")
            (folder / "final_ca.pdb").write_text("\n".join(pdb + ["TER", "END"]) + "\n")

            np.savez_compressed(folder / "sparse_trajectory.npz", **frames)
            np.savez_compressed(folder / "direct_x0.npz", **clean)
            mission.write(folder / "metrics.json", b.clean(stat))
            meta = dict(
                sample_key=key,
                requested_length=n,
                seed=seed,
                weights=weights,
                path=str(folder),
                checkpoint_sha256=digest,
                exposures=230132,
                initial_gaussian_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                runtime_seconds=time.monotonic() - beg,
                sampler="Existing nativeCoordinateVPDiffusion500step DDIM; generated self-conditioning; FP32 as historical evaluator",
            )
            meta["files"] = {str(p): b.sha(p) for p in folder.iterdir() if p.is_file()}
            for name in meta["files"]:
                with pathlib.Path(name).open("rb") as f:
                    os.fsync(f.fileno())
            mission.write(done, meta)
            samples.append(meta)
            stats.append(stat)
            b.table("generation_manifest.parquet", samples)
            b.table("generation_metrics.parquet", stats)
            mission.write(
                C / "progress.json",
                dict(
                    stage="MATCHED_E015_EPOCH1_GENERATION",
                    completed=len(stats),
                    total=20,
                    automatic_training_resume=False,
                ),
            )
    b.write(
        "MANIFEST.json",
        dict(
            status="COMPLETE",
            checkpoint=str(cp),
            position=record["position"],
            parameters=33128139,
            sha256=digest,
            exposures=230132,
            samples=20,
            weights=weights,
            weight_selection=means,
            matching="Identical original normalized Gaussian recovered and bitwise checked against serialized50%starts; common500step schedule and coordinate scale",
        ),
    )


if __name__ == "__main__":
    run()
