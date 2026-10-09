"""Metric-blind capacity selection and full-data provenance preflight."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import argparse
import json
import subprocess
import time
from pathlib import Path
import torch
import yaml
from protein_distance_diffusion.data.e014_population import prepare_manifests, build_coordinate_cache
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser, LADDER
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["population", "capacity", "cache"], required=True)
    args = parser.parse_args()
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "population":
        print(json.dumps(prepare_manifests(cfg), indent=2), flush=True)
        return
    if args.stage == "cache":
        if not json.loads((ROOT / "capacity_selection.json").read_text())["passed"]:
            raise ValueError("capacity prerequisite failed")
        report = build_coordinate_cache(cfg)
        print(json.dumps({k: v for k, v in report.items() if k != "source_shard_hashes"}, indent=2), flush=True)
        return
    population = prepare_manifests(cfg)
    results = []
    selected = None
    for name, kwargs in list(LADDER.items()) + [("e007_control", None)]:
        if name == "xl" and selected is not None:
            continue
        torch.manual_seed(cfg["seed"])
        model = (
            E014CapacityDenoiser(**kwargs)
            if kwargs
            else EquivariantPairCoordinateUNet(
                **yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
            )
        )
        count = sum(p.numel() for p in model.parameters())
        by_module = {name: sum(p.numel() for p in module.parameters()) for name, module in model.named_children()}
        model = model.cuda().eval()
        measurements = []
        for n in (64, 128, 256, 384, 500):
            x = torch.randn(1, n, 3, device="cuda")
            t = torch.tensor([50], device="cuda")
            lengths = torch.tensor([n], device="cuda")
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            continuity = torch.ones(1, n - 1, dtype=torch.bool, device="cuda")

            def run(
                current_model=model,
                current_x=x,
                current_t=t,
                current_lengths=lengths,
                current_mask=mask,
                current_continuity=continuity,
                control=kwargs is None,
            ):
                if control:
                    return current_model(current_x, current_t, current_lengths, current_mask, current_continuity)[
                        "v_prediction"
                    ]
                return current_model(current_x, current_t, current_lengths, current_mask)

            with torch.no_grad():
                y = run()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    y = run()
                torch.cuda.synchronize()
                forward = (time.perf_counter() - start) / 5
                rotation, _ = torch.linalg.qr(torch.randn(3, 3, device="cuda"))
                if torch.linalg.det(rotation) < 0:
                    rotation[:, 0] *= -1
                if kwargs:
                    rotated = model(x @ rotation + 2, t, lengths, mask)
                    reflected = model(x * torch.tensor([-1.0, 1, 1], device="cuda"), t, lengths, mask)
                    reflection_gap = float((reflected - y * torch.tensor([-1.0, 1, 1], device="cuda")).norm())
                else:
                    rotated = model(x @ rotation + 2, t, lengths, mask, continuity)["v_prediction"]
                    reflection_gap = None
                delta = rotated - y @ rotation
                absolute = float(delta.abs().max())
                relative = float(delta.norm() / (y.norm() + 1e-12))
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            run().square().mean().backward()
            torch.cuda.synchronize()
            finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()) and bool(
                torch.isfinite(y).all()
            )
            row = dict(
                length=n,
                forward_seconds=forward,
                forward_backward_seconds=time.perf_counter() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                equivariance_abs=absolute,
                equivariance_relative_l2=relative,
                reflection_gap=reflection_gap,
                finite=finite,
            )
            measurements.append(row)
            del run
        passed = all(
            r["finite"]
            and r["equivariance_abs"] <= 5e-5
            and r["equivariance_relative_l2"] <= 5e-5
            and (r["reflection_gap"] is None or r["reflection_gap"] > 1e-6)
            for r in measurements
        )
        fits = max(r["peak_reserved_bytes"] for r in measurements) <= cfg["preflight"]["max_peak_reserved_bytes"]
        entry = dict(
            name=name,
            config=kwargs,
            parameters=count,
            parameters_by_module=by_module,
            measurements=measurements,
            passed=passed,
            memory_comfortable=fits,
        )
        results.append(entry)
        atomic_json(ROOT / f"capacity_{name}.json", entry)
        print(json.dumps(entry), flush=True)
        if kwargs and selected is None and 5_000_000 <= count <= 10_000_000 and passed and fits:
            selected = entry
        del model
        torch.cuda.empty_cache()
    if selected is None:
        raise ValueError("no predeclared capacity meets target and resource/symmetry gates")
    estimates = {}
    for result in results:
        one_epoch = 0.0
        for (low, high), _size in zip(
            [(20, 64), (65, 128), (129, 256), (257, 384), (385, 500)], cfg["batch_sizes"], strict=True
        ):
            count = population["train_length_strata"][f"{low}-{high}"]
            seconds = next(r["forward_backward_seconds"] for r in result["measurements"] if r["length"] == high)
            # Current inherited model loops separately over proteins; batching
            # reduces optimizer overhead but does not vectorize residue blocks.
            one_epoch += count * seconds
        estimates[result["name"]] = dict(
            one_epoch_gpu_hours=one_epoch / 3600,
            two_epochs_gpu_hours=one_epoch / 1800,
            four_epochs_gpu_hours=one_epoch / 900,
            excludes_evaluation_and_loader_overhead=True,
        )
    report = dict(
        passed=True,
        selected=selected,
        selection_policy="smallest predeclared 5M-10M passing member; no validation metrics",
        config_sha256=sha(cfgpath),
        population_hash=population["train_sample_id_hash"],
        estimates=estimates,
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        cpu_threads=cfg["cpu_threads"],
    )
    atomic_json(ROOT / "capacity_selection.json", report)


if __name__ == "__main__":
    main()
