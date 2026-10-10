"""Isolated latency/backward on disposable copies and E007 bounded replay."""

# ruff: noqa: I001, E501, F403, F405
from e007_e014_comparison_protocol import *
import time
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import edges
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, centered_coordinate_noise
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor


def load_e014():
    cp = next(
        r
        for r in json.loads((E014 / "checkpoint_inventory.json").read_text())["selected"]
        if r["label"] == E014_LABELS[0]
    )
    model = E014CapacityDenoiser(**cp["contract"]["capacity_config"]).eval()
    model.load_state_dict(torch.load(cp["path"], map_location="cpu", weights_only=False)["model"], strict=True)
    return model


def benchmark():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for label in (E007_LABELS[0], E014_LABELS[0]):
        model = (load_e007(label)[0] if label in E007_LABELS else load_e014()).cuda()
        frozen = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        for n in (64, 128, 256, 500):
            generator = torch.Generator(device="cuda").manual_seed(14007000 + n)
            x = torch.randn((1, n, 3), device="cuda", generator=generator)
            x -= x.mean(1, keepdim=True)
            lengths = torch.tensor([n], device="cuda")
            mask = torch.ones((1, n), device="cuda", dtype=torch.bool)
            t = torch.tensor([250], device="cuda")
            input_digest = canonical_hash(x.cpu().tolist())
            counts = dict(
                dense_padded_pairs=((n + 7) // 8 * 8) ** 2,
                e007_bottleneck_pair_tokens=((n + 7) // 8) ** 2,
                e007_bottleneck_token_pairs=((n + 7) // 8) ** 4,
                sparse_directed_edges=len(edges(x[0])[0]),
                coarse_directed_edges=((n + 7) // 8) * ((n + 7) // 8 - 1),
            )
            for backward in (False, True):
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times = []
                try:
                    for repetition in range(8):
                        model.zero_grad(set_to_none=True)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if backward:
                            value = model(x, t, lengths, mask)
                            value.square().mean().backward()
                        else:
                            with torch.no_grad():
                                value = model(x, t, lengths, mask)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if repetition >= 3:
                            times.append(elapsed)
                        del value
                    result = dict(
                        checkpoint=label,
                        N=n,
                        phase="forward_backward" if backward else "forward",
                        median_seconds=float(np.median(times)),
                        p95_seconds=float(np.quantile(times, 0.95)),
                        repetitions=len(times),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        input_sha256=input_digest,
                        **counts,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    result = dict(
                        checkpoint=label,
                        N=n,
                        phase="forward_backward" if backward else "forward",
                        OOM=True,
                        error=str(exc),
                        **counts,
                    )
                results.append(result)
                print(json.dumps(result), flush=True)
                write(
                    "compute_comparison.json",
                    dict(
                        results=results,
                        device=torch.cuda.get_device_name(),
                        torch_version=torch.__version__,
                        precision="FP32; TF32 off; no AMP/compile",
                        cpu_threads=1,
                        batch_size=1,
                        timestep=250,
                        optimizer_steps=0,
                        backward_on_disposable_copies=True,
                        adam_state_included=False,
                        complexity="E007 dense N² pair states; axial attention/triangle cubic compute; full bottleneck attention over (ceil(N/8))² pair-grid tokens has O((ceil(N/8))⁴) score storage/compute in the historical need_weights=True implementation. E014 O(N²) graph-construction storage / O(N² log N) sorting, O((k+6)N) sparse messages, O((ceil(N/8))²) coarse residue global graph. Neither total architecture is linear in N.",
                    ),
                )
        assert all(torch.equal(v.cpu(), frozen[k]) for k, v in model.state_dict().items())
        del model, frozen
        torch.cuda.empty_cache()


@torch.no_grad()
def replay():
    _, subset = population()
    index, row = next((i, r) for i, r in enumerate(subset) if r["sample_id"] == "6nof_A")
    n = row["length"]
    mask = torch.ones((1, n), device="cuda", dtype=torch.bool)
    lengths = torch.tensor([n], device="cuda")
    d = CoordinateVPDiffusion(500)
    result = dict(
        e014=json.loads((E014 / "replay_numerics.json").read_text()),
        e014_record_sha256=sha(E014 / "replay_numerics.json"),
        tolerance=5e-6,
        tolerance_unchanged=True,
        e007=[],
    )
    for label in E007_LABELS:
        model, _, path = load_e007(label)
        digest = sha(path)
        model = model.cuda().requires_grad_(False)
        coordinates, descriptors, initial = [], [], []
        for repeat in range(10):
            g = torch.Generator(device="cuda").manual_seed(gaussian_seed(index, n, noise_seed))
            x = centered_coordinate_noise(torch.zeros((1, n, 3), device="cuda"), mask, generator=g)
            initial.append(x.cpu().numpy())
            for t in range(499, 440, -1):
                ts = torch.tensor([t], device="cuda")
                x, _, _ = d.deterministic_reverse_step(x, ts, model(x, ts, lengths, mask), mask)
            coordinates.append(x.cpu().numpy())
            descriptors.append(descriptor(coordinates[-1][0]))
            print(f"E007 replay {label} {repeat + 1}/10", flush=True)
        coord = np.array(coordinates)
        desc = np.array(descriptors)
        envelope = float(np.nanmax(np.nanmax(desc, axis=0) - np.nanmin(desc, axis=0)))
        initial_identical = all(np.array_equal(initial[0], z) for z in initial)
        assert initial_identical and sha(path) == digest
        result["e007"].append(
            dict(
                checkpoint=label,
                sample_id=row["sample_id"],
                N=n,
                endpoint=440,
                repetitions=10,
                initial_identical=initial_identical,
                initial_sha256=canonical_hash(initial[0].tolist()),
                coordinate_envelope=float(np.max(np.max(coord, axis=0) - np.min(coord, axis=0))),
                descriptor_envelope=envelope,
                repeatability_certified=envelope <= 5e-6,
                deterministic_within_test=bool(np.array_equal(coord, np.broadcast_to(coord[0], coord.shape))),
                historical_exact_replay_certified=False,
                reason="Repeated current execution; no identical historical E007 Gaussian-prefix record exists.",
            )
        )
        write("replay_numerics.json", result)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if not (OUT / "checkpoint_identity.json").exists():
        print("Waiting for frozen inference to finish before exclusive GPU cost/replay probes.", flush=True)
    while not (OUT / "checkpoint_identity.json").exists():
        time.sleep(10)
    benchmark()
    replay()
