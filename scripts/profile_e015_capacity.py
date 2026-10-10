"""Isolated worst-path probes: real optimizer/EMA/SC/PCGrad/induced states."""

import argparse
import gc
import json
import subprocess
import time

import e015_bootstrap
import torch

from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.models.e015_multiscale_parity_denoiser import E015MultiscaleParityDenoiser
from protein_distance_diffusion.training.e014_runtime import canonical_hash
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import GPULease, Runtime, e007_active, status

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def fallback_calibration(runtime):
    reliability = json.loads(runtime.path("calibration/reliability.json").read_text())
    return dict(
        midpoints=reliability["midpoints"],
        gate_width_logsnr=1.0,
        v_weights=[1.0] * 8,
        coefficients=dict(distance=0.15, torsion=0.1, chirality=0.1, clash=0.05),
    )


def profile(runtime, n, batch, repeats, name, precision=None):
    if e007_active(runtime.cfg):
        raise RuntimeError("E007 active; CUDA probe deferred")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not visible")
    lease = GPULease(runtime)
    torch.manual_seed(runtime.cfg["seed"])
    path = runtime.path("calibration/frozen.json")
    calibration = json.loads(path.read_text()) if path.exists() else fallback_calibration(runtime)
    selection = json.loads(runtime.path("precision_selection.json").read_text())
    precision = precision or selection["precision"]
    engine = Engine(runtime.cfg, calibration, precision=precision)
    probe_contract_hash = canonical_hash(dict(contract=runtime.contract(calibration), precision=precision))
    # Real TRAIN shape for each endpoint, padded to the declared worst length.
    rows = sorted((r for r in runtime.train if r["length"] <= n), key=lambda r: (-r["length"], r["sample_id"]))[:batch]
    if any(row["length"] != n for row in rows):
        raise RuntimeError(f"Not enough exact TRAIN length-{n} examples for batch {batch}")
    x, lengths, mask = coordinate_batch(rows, runtime.cache, runtime.cfg["coordinate_scale_angstrom"])
    with torch.no_grad(), engine.autocast():
        independent = torch.randn(x.shape, device=x.device, generator=engine.noise_rng)
        independent = independent - independent.mean(1, keepdim=True)
        stress_t = torch.full((batch,), 50, device=x.device)
        stress_v = engine.model(independent, stress_t, lengths, mask)
        stress_context = engine.diffusion.reconstruct_x0(independent, stress_t, stress_v, mask)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.reset_peak_memory_stats()
    forward, backward, optimizer = [], [], []
    conflicts = 0
    training_forward, pure_backward, pcgrad, pure_optimizer, utilization = [], [], [], [], []
    for i in range(repeats + 1):
        if e007_active(runtime.cfg):
            raise RuntimeError("E007 became active; stop E015 probe")
        engine.begin()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad(), engine.autocast():
            engine.model(x, torch.full((batch,), 50, device="cuda"), lengths, mask)
        torch.cuda.synchronize()
        f = time.perf_counter() - start
        start = time.perf_counter()
        result = engine.microbatch(
            x,
            lengths,
            mask,
            115066,
            0.5,
            force_sc=True,
            force_rollout=True,
            force_rollout_steps=3,
            force_sc_context=stress_context,
        )
        first_timing = engine.last_microbatch_timing.copy()
        # A subsequent physical batch runs with existing FP32 objective buffers.
        # This bounds mixed-bucket accumulation paths, not just an empty update.
        result = engine.microbatch(
            x,
            lengths,
            mask,
            115066,
            0.5,
            force_sc=True,
            force_rollout=True,
            force_rollout_steps=3,
            force_sc_context=stress_context,
        )
        second_timing = engine.last_microbatch_timing.copy()
        torch.cuda.synchronize()
        b = time.perf_counter() - start
        start = time.perf_counter()
        gradient = engine.finish()
        torch.cuda.synchronize()
        o = time.perf_counter() - start
        conflicts += int(gradient["conflict"])
        if i:
            forward.append(f)
            backward.append(b)
            optimizer.append(o)
            training_forward.append(first_timing["forward_seconds"] + second_timing["forward_seconds"])
            pure_backward.append(first_timing["backward_seconds"] + second_timing["backward_seconds"])
            pcgrad.append(engine.last_finish_timing["pcgrad_seconds"])
            pure_optimizer.append(engine.last_finish_timing["optimizer_seconds"])
            sample = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if sample.returncode == 0:
                utilization.append(float(sample.stdout.strip()))
        runtime.write(
            f"preflight/{name}.heartbeat.json",
            status("probe", n=n, batch=batch, repetition=i, v_loss=result["v_loss"], struct_loss=result["struct_loss"]),
        )
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    free, _ = torch.cuda.mem_get_info()
    peak_free_estimate = free - max(0, reserved - torch.cuda.memory_reserved())
    limits = runtime.cfg["preflight"]
    passed = reserved <= limits["max_reserved_fraction"] * total and peak_free_estimate >= limits["free_margin_bytes"]
    elapsed = sum(backward) + sum(optimizer)
    output = dict(
        profile_version=4,
        fp32_geometry_islands=True,
        probe_contract_hash=probe_contract_hash,
        independent_model_generated_graph_stress=True,
        n=n,
        physical_batch=batch,
        parameter_count=sum(p.numel() for p in engine.model.parameters()),
        precision=engine.precision,
        peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved,
        reserved_fraction=reserved / total,
        total_vram_bytes=total,
        free_after_probe_bytes=free,
        estimated_free_at_peak_bytes=peak_free_estimate,
        passed=passed,
        repetitions=repeats,
        forward_seconds=sum(forward) / len(forward),
        backward_pcgrad_seconds=sum(backward) / len(backward),
        optimizer_seconds=sum(pure_optimizer) / len(pure_optimizer),
        forward_with_sc_rollout_seconds=sum(training_forward) / len(training_forward),
        backward_seconds=sum(pure_backward) / len(pure_backward),
        pcgrad_seconds=sum(pcgrad) / len(pcgrad),
        probe_accumulation_steps=2,
        total_device_used_bytes=total - free,
        estimated_peak_total_device_used_bytes=total - peak_free_estimate,
        free_device_bytes=free,
        gpu_utilization_percent=sum(utilization) / max(len(utilization), 1),
        proteins_per_second=2 * batch * repeats / elapsed,
        residues_per_second=2 * batch * n * repeats / elapsed,
        pcgrad_conflict_fraction=conflicts / (repeats + 1),
        paths_tested=[
            "AdamW",
            "FP32 EMA",
            "self-conditioning",
            "three-or-fewer EMA reverse steps",
            "dual PCGrad",
            "geometry",
            "activation checkpointing",
        ],
        calibration_frozen=path.exists(),
    )
    runtime.write(f"preflight/{name}.json", output)
    print(json.dumps(output), flush=True)
    del engine, x
    gc.collect()
    torch.cuda.empty_cache()
    lease.stream.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--name", default="capacity_n500_b1")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()
    runtime = Runtime(CONFIG)
    if (
        not 25_000_000
        <= sum(p.numel() for p in E015MultiscaleParityDenoiser(**runtime.cfg["model"]).parameters())
        <= 35_000_000
    ):
        raise RuntimeError("Capacity outside frozen target")
    profile(runtime, args.n, args.batch, args.repeats, args.name, args.precision)


if __name__ == "__main__":
    main()
