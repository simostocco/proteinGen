"""Actual FP32 AdamW lifecycle, disposable historical E007, bounded lengths."""

# ruff: noqa: I001
from e007_fullscale_runtime import (
    RUN,
    CONFIG,
    MODEL_CONFIG,
    verified_inputs,
    baseline,
    atomic_json,
    sha,
    torch,
    yaml,
    np,
)
import time
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, center_coordinates


def main():
    cfg, population, _, _ = verified_inputs()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    torch.manual_seed(cfg["seed"])
    model = E007ScaleupControl(**yaml.safe_load(MODEL_CONFIG.read_text())["model"]).cuda()
    assert sum(p.numel() for p in model.parameters()) == 7586505
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    diffusion = CoordinateVPDiffusion(500)
    results = []
    for n, size in ((64, 4), (128, 2), (256, 1), (500, 1)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(size, n, 3, device="cuda")
        mask = torch.ones(size, n, device="cuda", dtype=torch.bool)
        lengths = torch.full((size,), n, device="cuda", dtype=torch.long)
        batch = diffusion.make_training_batch(
            center_coordinates(x, mask), mask, timesteps=torch.full((size,), 250, device="cuda", dtype=torch.long)
        )
        fw, whole = [], []
        try:
            for _repeat in range(5):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    model(batch.noisy_coordinates, batch.timesteps, lengths, mask)
                torch.cuda.synchronize()
                fw.append(time.perf_counter() - start)
                start = time.perf_counter()
                prediction = model(batch.noisy_coordinates, batch.timesteps, lengths, mask)
                loss = coordinate_v_loss(prediction, batch.coordinate_v_target, mask)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                whole.append(time.perf_counter() - start)
                assert bool(torch.isfinite(loss)) and bool(torch.isfinite(norm))
                del prediction, loss
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            external = max(0, total - free - torch.cuda.memory_reserved())
            safe = total - reserved - external >= 512 * 2**20
            r = dict(
                N=n,
                physical_batch=size,
                effective_batch=size,
                accumulation=1,
                peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved,
                external_bytes=external,
                total_bytes=total,
                safety_headroom_bytes=total - reserved - external,
                safe=safe,
                forward_median_seconds=float(np.median(fw[2:])),
                forward_backward_optimizer_median_seconds=float(np.median(whole[2:])),
                proteins_per_second=size / float(np.median(whole[2:])),
                optimizer_states_initialized=len(optimizer.state),
                actual_optimizer_steps=5,
            )
        except torch.cuda.OutOfMemoryError as exc:
            r = dict(N=n, physical_batch=size, safe=False, OOM=True, error=str(exc))
        results.append(r)
        print(r, flush=True)
        atomic_json(
            RUN / "compute_profile.json",
            dict(
                results=results,
                passed=all(x["safe"] for x in results) and len(results) == 4,
                device=torch.cuda.get_device_name(),
                optimizer_included=True,
                original_comfort_limit_bytes=5 * 2**30,
                active_safety_rule=(
                    "At least 512 MiB residual VRAM after external use and reserved peak; "
                    "direct user authorization supersedes old comfortable-memory launch gate"
                ),
                config_hash=sha(CONFIG),
                train_population_hash=population["train_sample_id_hash"],
            ),
        )
        if not r["safe"]:
            raise RuntimeError("No safe frozen-batch execution; stop before production, report resource feasibility")
        del x, mask, lengths, batch
    atomic_json(RUN / "E012_BASELINE.json", baseline())


if __name__ == "__main__":
    main()
