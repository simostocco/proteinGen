"""Bounded physical-batch memory study; never changes effective training batches."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import gc
import json
import time
import torch
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.e014_runtime import atomic_json

OUT = ROOT / "throughput_optimization"


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(OUT / "frozen_profile_checkpoint.pt", map_location="cpu", weights_only=False)
    model = E014CapacityDenoiser(**state["contract"]["capacity_config"]).cuda()
    model.load_state_dict(state["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    opt.load_state_dict(state["optimizer"])
    results = []
    ceiling = 5 * 1024**3  # existing E014 stricter envelope; below 80% of physical GPU
    for n in (64, 128, 256, 384, 500):
        previous = None
        for b in (1, 2, 4, 8, 16, 32):
            if previous is not None:
                base = 250 * 1024**2
                projected = base + (previous["allocated"] - base) * (b / previous["batch"]) * 1.2
                if projected > ceiling:
                    results.append(
                        dict(length=n, batch=b, status="not_run_projected_memory_guard", projected_bytes=projected)
                    )
                    break
            opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            if free < 2 * 1024**3:
                raise RuntimeError("coexistence free-memory guard")
            x = torch.randn(b, n, 3, device="cuda")
            t = torch.full((b,), 250, device="cuda", dtype=torch.long)
            lengths = torch.full((b,), n, device="cuda", dtype=torch.long)
            mask = torch.ones(b, n, device="cuda", dtype=torch.bool)
            elapsed = []
            torch.cuda.reset_peak_memory_stats()
            for _repeat in range(3):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model(x, t, lengths, mask)
                out.square().mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                elapsed.append(time.perf_counter() - start)
            row = dict(
                length=n,
                batch=b,
                status="passed",
                allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(),
                seconds=sum(elapsed[1:]) / 2,
            )
            row["proteins_per_second"] = b / row["seconds"]
            row["seconds_per_protein"] = row["seconds"] / b
            results.append(row)
            atomic_json(
                OUT / "headroom.json", dict(results=results, ceiling_bytes=ceiling, scientific_batches_unchanged=True)
            )
            if row["reserved"] > ceiling:
                raise RuntimeError("memory envelope exceeded; no further tests")
            previous = row
            del x, t, lengths, mask, out
    atomic_json(OUT / "headroom.json", dict(results=results, ceiling_bytes=ceiling, scientific_batches_unchanged=True))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
