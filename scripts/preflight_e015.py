"""Full CUDA precision, typed symmetry, strict serialized replay certification."""

import gc
import json

import e015_bootstrap
import numpy as np
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor
from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import stable_knn
from protein_distance_diffusion.training.e014_runtime import atomic_checkpoint, load_checkpoint, restore_rng, rng_state
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_replay import replay_snapshot
from protein_distance_diffusion.training.e015_runtime import GPULease, Runtime
from protein_distance_diffusion.training.e015_symmetry import autocast, error, require_symmetry, symmetry

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def candidate(runtime, calibration, precision):
    torch.manual_seed(runtime.cfg["seed"])
    engine = Engine(runtime.cfg, calibration, precision=precision)
    row = next(r for r in runtime.train if r["length"] == 500)
    x, n, m = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
    t = torch.tensor([20], device="cuda")
    # Keep the original geometric fixture, but transform it in FP32.
    previous, available = x * 0.8, torch.tensor([True], device="cuda")
    results = []
    for name, model in [("raw", engine.model), ("ema", engine.ema)]:
        for dtype in ("fp32", precision):
            for seed in (7715, 7716, 7717):
                result = symmetry(model, x, t, n, m, previous, available, dtype, seed)
                result.update(model=name, tolerance=require_symmetry(result, dtype))
                results.append(result)
    model = engine.model.eval()
    generator = torch.Generator().manual_seed(7715)
    r, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator))
    r[:, 0] *= torch.det(r)
    r = r.cuda()
    reflection = torch.diag(x.new_tensor([-1.0, 1.0, 1.0]))
    knn_cases = []
    for case in ("random", "symmetric", "near_tied", "repeated"):
        if case == "random":
            cloud = torch.randn(32, 3, generator=torch.Generator().manual_seed(5015)).cuda()
        else:
            cloud = x.new_tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
            )
            if case == "near_tied":
                cloud[1, 0] += 1e-7
            if case == "repeated":
                cloud[1] = cloud[0]
        for k in (1, 2, 3):
            indices = stable_knn(cloud, k)
            same = torch.equal(indices, stable_knn(cloud @ r.T + 0.75, k))
            if not same:
                raise RuntimeError(f"CUDA numerical-tie kNN identity failed: {case}/{k}")
            knn_cases.append(dict(case=case, k=k, rotated_translated_identity=True))
    trunk = []
    with torch.no_grad():
        state = model.trunk_single(x[0], t[0], previous[0])[0]
        for transform in (r, reflection):
            determinant = float(torch.det(transform).round())
            transformed = model.trunk_single(x[0] @ transform.T + 3, t[0], previous[0] @ transform.T - 4)[0]
            for key in ("h", "v", "p", "a"):
                expected = getattr(state, key)
                if key in ("v", "a"):
                    expected = expected @ transform.T
                if key in ("p", "a"):
                    expected = expected * determinant
                metric = error(getattr(transformed, key), expected)
                if metric["relative_error"] > 5e-5:
                    raise RuntimeError(f"FP32 typed trunk {key} parity error {metric}")
                trunk.append(dict(state=key, determinant=determinant, **metric))
    # A helical synthetic geometry guarantees an assessable chiral test fixture.
    theta = torch.arange(64, device="cuda") * 0.6
    helix = torch.stack([theta.cos(), theta.sin(), theta * 0.2], -1)[None]
    hn, hm, ht = torch.tensor([64], device="cuda"), torch.ones(1, 64, device="cuda", dtype=torch.bool), t
    original_heads = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith(("chiral_a.", "chiral_p.", "local.2."))
    }
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("weight") and name in original_heads:
                p.fill_(0.01)
    reflected_helix = helix @ reflection.T
    reflection_results = []
    for dtype in ("fp32", precision):
        with torch.no_grad(), autocast(dtype, "cuda"):
            a = model(helix, ht, hn, hm)
            b = model(reflected_helix, ht, hn, hm)
        gap = error(b, a.float() @ reflection.T)
        if gap["absolute_error"] <= 1e-4:
            raise RuntimeError("Enabled chiral synthetic reflection gap absent")
        model.chiral_enabled = False
        with torch.no_grad(), autocast(dtype, "cuda"):
            a = model(helix, ht, hn, hm)
            b = model(reflected_helix, ht, hn, hm)
        control = error(b, a.float() @ reflection.T)
        if control["relative_error"] > (5e-5 if dtype == "fp32" else 0.03):
            raise RuntimeError("Disabled chiral head failed reflection control")
        model.chiral_enabled = True
        reflection_results.append(dict(precision=dtype, enabled_gap=gap, disabled_control=control))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in original_heads:
                p.copy_(original_heads[name])
    engine.model.train()
    engine.capture_replay = True

    def step():
        engine.begin()
        observations = engine.microbatch(x, n, m, 115066, 1.0, force_sc=True, force_rollout=True, force_rollout_steps=3)
        gradients = engine.finish()
        return replay_snapshot(engine, [observations], gradients)

    step()  # populate actual AdamW and EMA state at N500 with all paths active
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
    directory = runtime.path(f"continuation_v2/cuda_replay_{precision}/checkpoints")
    directory.mkdir(parents=True, exist_ok=True)
    cp = atomic_checkpoint(directory, "resume_state", payload)
    expected = step()
    saved = load_checkpoint(directory, contract)
    engine.model.load_state_dict(saved["model"])
    engine.ema.load_state_dict(saved["ema"])
    engine.optimizer.load_state_dict(saved["optimizer"])
    engine.scaler.load_state_dict(saved["scaler"])
    engine.restore_rng(saved["engine_rng"])
    restore_rng(saved["rng"])
    actual = step()
    if expected != actual:
        runtime.write(f"continuation_v2/replay_failure_{precision}.json", dict(expected=expected, actual=actual))
        raise RuntimeError("Strict CUDA scientific replay is not bitwise exact")
    # Independent descriptor replay retains the historical unchanged5e-6 rule.
    row = next(r for r in runtime.train if r["length"] == 64)
    sx, sn, sm = coordinate_batch([row], runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
    generator = torch.Generator(device="cuda").manual_seed(9015)
    initial = torch.randn(sx.shape, device="cuda", generator=generator)
    initial -= initial.mean(1, keepdim=True)
    descriptors = []
    engine.ema.eval()
    with torch.no_grad():
        for _ in range(3):
            current, context = initial.clone(), None
            for s in range(499, 490, -1):
                times = torch.tensor([s], device="cuda")
                with autocast(precision, "cuda"):
                    v = engine.ema(current, times, sn, sm, context, torch.tensor([context is not None], device="cuda"))
                current, context, _ = engine.diffusion.deterministic_reverse_step(current, times, v.float(), sm)
            descriptors.append(descriptor(current[0].cpu().numpy()))
    envelope = float(np.max(np.abs(np.array(descriptors) - descriptors[0])))
    if envelope > 5e-6:
        raise RuntimeError("Descriptor replay exceeds unchanged5e-6 tolerance")
    report = dict(
        passed=True,
        precision=precision,
        symmetry=results,
        typed_trunk_o3=trunk,
        cuda_knn_cases=knn_cases,
        reflection=reflection_results,
        serialized_resume_exact=True,
        strict_replay=actual,
        checkpoint=str(cp),
        descriptor_envelope=envelope,
        descriptor_tolerance=5e-6,
        endpoint=490,
        prospective_opened=False,
        full_t0_sampling=False,
        calibration_contract=contract,
        storage_telemetry=runtime.disk_check(),
        fp32_geometry_islands=True,
    )
    runtime.write(
        "preflight/replay_numerics.json",
        dict(
            e015_envelope=envelope,
            endpoint=490,
            tolerance=5e-6,
            passed=True,
            e014_historical_envelope=8.962951e-6,
            e014_certified=False,
            tolerance_relaxed=False,
        ),
    )
    return report


def main():
    runtime = Runtime(CONFIG)
    lease = GPULease(runtime)
    calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
    requested = runtime.cfg["preflight"]["precision"]
    candidates = ["bf16", "fp16"] if requested == "auto" else [requested]
    failures = []
    for precision in candidates:
        try:
            result = candidate(runtime, calibration, precision)
            result["previous_candidate_failures"] = failures
            runtime.write("preflight/full_certification.json", result)
            runtime.write(
                "precision_selection.json",
                dict(
                    passed=True,
                    precision=precision,
                    fp32_geometry_islands=True,
                    proper_rotation_maximum=0.03,
                    contract=result["calibration_contract"],
                ),
            )
            print(json.dumps({k: v for k, v in result.items() if k != "calibration_contract"}), flush=True)
            lease.stream.close()
            return
        except Exception as exc:
            failures.append(dict(precision=precision, error=str(exc)))
            runtime.write(f"continuation_v2/certification_failure_{precision}.json", failures[-1])
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError(f"All selected precision candidates failed closed: {failures}")


if __name__ == "__main__":
    main()
