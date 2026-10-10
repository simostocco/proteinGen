"""Full CUDA symmetry, serialized-resume and storage certification."""

import copy
import json

import e015_bootstrap
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.e014_runtime import atomic_checkpoint, load_checkpoint, restore_rng, rng_state
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import GPULease, Runtime, e007_active

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def certify(runtime):
    if e007_active(runtime.cfg):
        raise RuntimeError("E007 active")
    lease = GPULease(runtime)
    calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
    torch.manual_seed(runtime.cfg["seed"])
    engine = Engine(runtime.cfg, calibration)
    row = next(r for r in runtime.train if r["length"] == 500)
    x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
    t = torch.tensor([20], device="cuda")
    context = x * 0.8
    available = torch.tensor([True], device="cuda")
    r, _ = torch.linalg.qr(torch.randn(3, 3, device="cuda"))
    r[:, 0] *= torch.det(r)
    tests = []
    for name, model in [("raw", engine.model), ("ema", engine.ema)]:
        model.eval()
        for precision in ("fp32", engine.precision):
            with (
                torch.no_grad(),
                torch.autocast(
                    "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16, enabled=precision != "fp32"
                ),
            ):
                a = model(x, t, n, m, context, available)
                b = model(x @ r.T + 3, t, n, m, context @ r.T - 4, available)
                error = float((b - a @ r.T).norm() / a.norm().clamp_min(1e-8))
            tolerance = 5e-5 if precision == "fp32" else 0.03
            if error > tolerance:
                raise RuntimeError(f"{name} {precision} rotation relative error {error}>{tolerance}")
            tests.append(
                dict(
                    model=name,
                    precision=precision,
                    relative_error=error,
                    tolerance=tolerance,
                    justification="5e-5 FP32; 3% bound for BF16 accumulated sparse arithmetic"
                    if precision != "fp32"
                    else "FP32 E014 tolerance",
                )
            )
    engine.model.train()
    row = next(r for r in runtime.train if r["length"] == 64)
    x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])

    def step():
        engine.begin()
        observation = engine.microbatch(x, n, m, 115066, 1.0, force_sc=True, force_rollout=True, force_rollout_steps=3)
        gradient = engine.finish()
        return observation, gradient

    step()
    contract = runtime.contract(calibration)
    payload = dict(
        model=engine.model.state_dict(),
        ema=engine.ema.state_dict(),
        optimizer=engine.optimizer.state_dict(),
        scaler=engine.scaler.state_dict(),
        engine_rng=engine.rng_state(),
        rng=rng_state(),
        contract=contract,
        position=dict(update=1, epoch=0, exposures=1, cursor=1),
        exposures=[1],
    )
    directory = runtime.path("preflight/exact_resume/checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    cp = atomic_checkpoint(directory, "resume_state", payload)
    expected = step()
    weights = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
    ema_weights = {k: v.detach().cpu().clone() for k, v in engine.ema.state_dict().items()}
    optimizer = copy.deepcopy(engine.optimizer.state_dict())
    saved = load_checkpoint(directory, contract)
    engine.model.load_state_dict(saved["model"])
    engine.ema.load_state_dict(saved["ema"])
    engine.optimizer.load_state_dict(saved["optimizer"])
    engine.scaler.load_state_dict(saved["scaler"])
    engine.restore_rng(saved["engine_rng"])
    restore_rng(saved["rng"])
    actual = step()
    if expected != actual:
        raise RuntimeError("Serialized CUDA resume metrics not exact")
    if any(not torch.equal(weights[k], v.cpu()) for k, v in engine.model.state_dict().items()):
        raise RuntimeError("CUDA raw model resume not exact")
    if any(not torch.equal(ema_weights[k], v.cpu()) for k, v in engine.ema.state_dict().items()):
        raise RuntimeError("CUDA EMA resume not exact")
    for i, state in optimizer["state"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and not torch.equal(
                value, engine.optimizer.state_dict()["state"][i][key]
            ):
                raise RuntimeError("CUDA optimizer resume not exact")
    from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor

    with torch.no_grad():
        initial = torch.randn_like(x)
        initial = initial - initial.mean(1, keepdim=True)
        descriptors = []
        for _repeat in range(3):
            current = initial.clone()
            context = None
            for s in range(499, 489, -1):
                times = torch.tensor([s], device="cuda")
                v = engine.ema(current, times, n, m, context, torch.tensor([context is not None], device="cuda"))
                current, context, _ = engine.diffusion.deterministic_reverse_step(current, times, v, m)
            descriptors.append(descriptor(current[0].cpu().numpy()))
        import numpy as np

        envelope = float(np.max(np.abs(np.array(descriptors) - descriptors[0])))
        if envelope > 5e-6:
            raise RuntimeError("E015 descriptor exact-replay tolerance exceeded")
    runtime.write(
        "preflight/replay_numerics.json",
        dict(
            e015_envelope=envelope,
            tolerance=5e-6,
            passed=True,
            e014_historical_envelope=8.962951e-6,
            e014_certified=False,
            tolerance_relaxed=False,
        ),
    )
    report = dict(
        passed=True,
        symmetry=tests,
        serialized_resume_exact=True,
        checkpoint=str(cp),
        prospective_opened=False,
        full_t0_sampling=False,
        calibration_contract=contract,
        storage_budget=runtime.disk_check(),
    )
    runtime.write("preflight/full_certification.json", report)
    print(json.dumps(report), flush=True)
    lease.stream.close()


if __name__ == "__main__":
    certify(Runtime(CONFIG))
