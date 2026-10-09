"""Bounded clone profiling from a frozen training checkpoint; live job untouched."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
import protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser as core
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha, epoch_schedule

OUT = ROOT / "throughput_optimization"


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    latest = json.loads((ROOT / "e014/checkpoints/LATEST.json").read_text())
    frozen = OUT / "frozen_profile_checkpoint.pt"
    if not frozen.exists():
        shutil.copyfile(latest["path"], frozen)
        if sha(frozen) != latest["sha256"]:
            raise ValueError("frozen profiling checkpoint copy failed")
    state = torch.load(frozen, map_location="cpu", weights_only=False)
    model = E014CapacityDenoiser(**state["contract"]["capacity_config"]).cuda().train()
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    optimizer.load_state_dict(state["optimizer"])
    cache = CoordinateCache(ROOT)
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], state["epoch"])
    diffusion = CoordinateVPDiffusion(500)
    cpu_times, events = defaultdict(list), defaultdict(list)

    def measured(name, fn):
        def wrapped(*args, **kwargs):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start_event.record()
            with torch.profiler.record_function(name):
                output = fn(*args, **kwargs)
            end_event.record()
            cpu_times[name].append(time.perf_counter() - begin)
            events[name].append((start_event, end_event))
            return output

        return wrapped

    for name in ("edges", "edge_features", "local_geometry"):
        setattr(core, name, measured(name, getattr(core, name)))
    for name, modules in (("sparse_block", model.blocks), ("coarse_global_block", model.global_blocks)):
        for block in modules:
            block.forward = measured(name, block.forward)
    for name in ("broadcast_h", "broadcast_v"):
        getattr(model, name).forward = measured(name, getattr(model, name).forward)

    batch_times = []
    # Warm one complete original update, then profile three fixed real TRAIN batches.
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
    for z, ids in enumerate(schedule[state["position"] : state["position"] + 4]):
        rows = [train[i] for i in ids]
        wall = time.perf_counter()
        x, lengths, mask = coordinate_batch(rows, cache, cfg["coordinate_scale_angstrom"])
        cpu_times["batch_prepare_h2d_center"].append(time.perf_counter() - wall)
        noisy, targets, ts = [], [], []
        wall = time.perf_counter()
        for i, row in enumerate(rows):
            n = row["length"]
            g = torch.Generator(device="cuda").manual_seed(
                noise_seed(f"e014-train-epoch-{state['epoch']}", row["sample_id"], 0)
            )
            batch = diffusion.make_training_batch(x[i : i + 1, :n], mask[i : i + 1, :n], generator=g)
            noisy.append(torch.nn.functional.pad(batch.noisy_coordinates, (0, 0, 0, x.shape[1] - n)))
            targets.append(torch.nn.functional.pad(batch.coordinate_v_target, (0, 0, 0, x.shape[1] - n)))
            ts.append(batch.timesteps)
        xt, target, t = torch.cat(noisy), torch.cat(targets), torch.cat(ts)
        cpu_times["corruption"].append(time.perf_counter() - wall)
        if z == 1:
            profiler.start()
        optimizer.zero_grad(set_to_none=True)
        begin = time.perf_counter()
        prediction = measured("forward", model)(xt, t, lengths, mask)
        loss = coordinate_v_loss(prediction, target, mask)
        measured("backward", loss.backward)()
        measured("gradient_clipping", torch.nn.utils.clip_grad_norm_)(model.parameters(), 1.0, error_if_nonfinite=True)
        measured("optimizer_step", optimizer.step)()
        torch.cuda.synchronize()
        batch_times.append(
            dict(lengths=[r["length"] for r in rows], proteins=len(rows), elapsed=time.perf_counter() - begin)
        )
        if z >= 1:
            profiler.step()
    profiler.stop()
    for name, pairs in events.items():
        events[name] = [a.elapsed_time(b) / 1000 for a, b in pairs]
    operators = []
    for item in profiler.key_averages():
        operators.append(
            dict(
                name=item.key,
                count=item.count,
                self_cpu_ms=item.self_cpu_time_total / 1000,
                cpu_ms=item.cpu_time_total / 1000,
                self_cuda_ms=item.self_device_time_total / 1000,
                cuda_ms=item.device_time_total / 1000,
            )
        )
    operators.sort(key=lambda r: r["self_cpu_ms"], reverse=True)
    report = dict(
        checkpoint=str(frozen),
        checkpoint_sha256=sha(frozen),
        epoch=state["epoch"],
        frozen_update=state["updates"],
        frozen_exposures=int(state["exposures"].sum()),
        stage_host_seconds={k: dict(total=sum(v), calls=len(v), mean=sum(v) / len(v)) for k, v in cpu_times.items()},
        stage_cuda_event_seconds={k: dict(total=sum(v), calls=len(v), mean=sum(v) / len(v)) for k, v in events.items()},
        operators=operators,
        real_train_batches=batch_times,
        trace_exported=False,
        live_run_modified=False,
        loader_wait_seconds=0,
        note=(
            "CUDA event intervals include dispatch/queue delays; operator self CUDA time isolates kernels. "
            "Stage scopes overlap."
        ),
    )
    atomic_json(OUT / "bounded_profile.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "operators"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
