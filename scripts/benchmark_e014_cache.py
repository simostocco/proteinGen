"""Frozen-checkpoint equivalence and bounded throughput audit; no live mutations."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import argparse
import pyarrow.parquet as pq
import yaml
import time
from pathlib import Path
import torch
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.models.coordinate_execution_cache import CachedCoordinateDenoiser
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import epoch_schedule, noise_seed
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha

OUT = ROOT / "throughput_optimization"
TOL = dict(output=5e-6, loss=1e-6, gradient=2e-5, parameter=5e-7, symmetry=5e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-control", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    atomic_json(OUT / "cache_tolerances.json", TOL)
    path = OUT / "frozen_profile_checkpoint.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    models = [
        cls(**state["contract"]["capacity_config"]).cuda()
        for cls in (E014CapacityDenoiser, E014CapacityDenoiser if args.original_control else CachedCoordinateDenoiser)
    ]
    optimizers = []
    for model in models:
        model.load_state_dict(state["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
        opt.load_state_dict(state["optimizer"])
        optimizers.append(opt)
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], state["epoch"])
    cache = CoordinateCache(ROOT)
    diffusion = CoordinateVPDiffusion(500)
    results = []
    for step, ids in enumerate(schedule[state["position"] : state["position"] + 5]):
        rows = [train[i] for i in ids]
        x, lengths, mask = coordinate_batch(rows, cache, cfg["coordinate_scale_angstrom"])
        noisy, targets, ts = [], [], []
        n = x.shape[1]
        for i, row in enumerate(rows):
            size = row["length"]
            gen = torch.Generator(device="cuda").manual_seed(
                noise_seed(f"e014-train-epoch-{state['epoch']}", row["sample_id"], 0)
            )
            batch = diffusion.make_training_batch(x[i : i + 1, :size], mask[i : i + 1, :size], generator=gen)
            noisy.append(torch.nn.functional.pad(batch.noisy_coordinates, (0, 0, 0, n - size)))
            targets.append(torch.nn.functional.pad(batch.coordinate_v_target, (0, 0, 0, n - size)))
            ts.append(batch.timesteps)
        x, target, t = torch.cat(noisy), torch.cat(targets), torch.cat(ts)
        outputs, losses, grads = [], [], []
        for model, opt in zip(models, optimizers, strict=True):
            opt.zero_grad(set_to_none=True)
            out = model(x, t, lengths, mask)
            loss = coordinate_v_loss(out, target, mask)
            loss.backward()
            outputs.append(out.detach())
            losses.append(loss.detach())
            grads.append([p.grad.clone() for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        errors = dict(
            output=float((outputs[0] - outputs[1]).abs().max()),
            loss=float((losses[0] - losses[1]).abs()),
            gradient=max(float((a - b).abs().max()) for a, b in zip(*grads, strict=True)),
            parameter=max(
                float((a.detach() - b.detach()).abs().max())
                for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True)
            ),
        )
        results.append(dict(step=step, length=n, errors=errors))
    timings = []
    for n in (64, 128, 256, 384, 500):
        x = torch.randn(1, n, 3, device="cuda")
        t = torch.tensor([250], device="cuda")
        lengths = torch.tensor([n], device="cuda")
        mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
        row = dict(length=n)
        for label, model, opt in zip(("original", "cached"), models, optimizers, strict=True):
            seconds = []
            for _repeat in range(6):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(x, t, lengths, mask)
                out.square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
            row[label] = sum(seconds[1:]) / 5
        timings.append(row)
    atomic_json(
        OUT / ("original_repeat_equivalence.json" if args.original_control else "cache_real_equivalence.json"),
        dict(
            passed=all(all(v <= TOL[k] for k, v in r["errors"].items()) for r in results),
            results=results,
            tolerances=TOL,
            checkpoint_sha256=sha(path),
            synthetic_inputs=False,
            live_run_modified=False,
            timings=timings,
        ),
    )
    print(json.dumps(dict(results=results, timings=timings)), flush=True)


if __name__ == "__main__":
    main()
