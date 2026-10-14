"""Synthetic benchmark, dry-run default. GPU execution requires explicit approved card."""

import argparse, json, os, time, subprocess, resource
from pathlib import Path


def estimate(N, width, heads, layers, strategy):
    L = N + 1
    # Conservative budgeting envelope, not an allocator/VRAM prediction.
    encoder = 40 * L * L * 8 + 32 * L * L * 4 + heads * layers * L * L * 3 * 4 + layers * L * width * 32
    copies = 5 if strategy == "streamed" else 41
    return {
        "N": N,
        "encoder_activation_budget_bytes": encoder,
        "SAC_live_graph_factor": copies,
        "SAC_peak_budget_bytes": int(2 * copies * encoder + 0.75 * 1024**3),
        "status": "INFERRED_CONSERVATIVE_ENVELOPE_NOT_MEASURED",
        "includes": "geometry,pair-MLP,attention,2x overhead and0.75GiB context margin",
        "excludes": "desktop/other processes,allocator fragmentation uncertainty; must measure",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--approval-card")
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 500])
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--backend", choices=["manual", "sdpa"], default="manual")
    p.add_argument("--sac-strategy", choices=["streamed", "eager"], default="streamed")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    root = Path(os.environ["E016_ARTIFACT_ROOT"]).resolve()
    out = Path(args.output).resolve()
    if not str(root).startswith("/mnt/d/") or not str(out).startswith(str(root) + "/"):
        raise SystemExit("D-only output under versioned root required")
    if out.exists():
        raise SystemExit("immutable output collision")
    if (
        args.width % args.heads
        or (args.width // args.heads) % 2
        or not 1 <= args.layers <= 6
        or any(N not in [64, 128, 256, 500] for N in args.lengths)
    ):
        raise SystemExit("invalid frozen benchmark configuration")
    estimates = [estimate(N, args.width, args.heads, args.layers, args.sac_strategy) for N in args.lengths]
    if not args.execute:
        out.write_text(
            json.dumps(
                {
                    "executed": False,
                    "estimates": estimates,
                    "GPU_calls": 0,
                    "requires": "explicit approval card before execution",
                },
                indent=2,
            )
            + "\n"
        )
        print("DRY_RUN_ONLY")
        return
    if not args.approval_card:
        raise SystemExit("BLOCKED_EXPLICIT_APPROVAL")
    card_path = Path(args.approval_card).resolve()
    if not str(card_path).startswith(str(root) + "/"):
        raise SystemExit("D-only approval under versioned root")
    card = json.loads(card_path.read_text())
    required = {"approval": "EXPLICIT_USER_APPROVAL", "scope": "E016_SYNTHETIC_BENCHMARK_ONLY"}
    if any(card.get(k) != v for k, v in required.items()) or card.get("device") != args.device:
        raise SystemExit("BLOCKED_APPROVAL_SCOPE")
    if card.get("source_commit") != subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip():
        raise SystemExit("BLOCKED_SOURCE_CHANGED")
    if (
        any(N not in card["lengths"] for N in args.lengths)
        or max(x["SAC_peak_budget_bytes"] for x in estimates) > card["max_peak_budget_bytes"]
    ):
        raise SystemExit("BLOCKED_ESTIMATE_EXCEEDS_APPROVAL")
    if (args.width, args.heads, args.layers, args.backend, args.sac_strategy) != (
        card["width"],
        card["heads"],
        card["layers"],
        card["backend"],
        card["sac_strategy"],
    ):
        raise SystemExit("BLOCKED_CONFIGURATION_CHANGED")
    if (
        card["max_wall_seconds"] > 600
        or card["max_peak_allocated_bytes"] > 5 * 1024**3
        or card["max_artifact_bytes"] > 256 * 1024**2
    ):
        raise SystemExit("BLOCKED_UNBOUNDED_CARD")
    # All guards before torch/CUDA initialization. A scoped approved card is required.
    import shutil

    if shutil.disk_usage(root).free < 10 * 1024**3 + card["max_artifact_bytes"]:
        raise SystemExit("BLOCKED_DISK")
    start = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if args.device == "cuda" else ""
    scientific = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text().strip()
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if (
            int(proc.name) != os.getpid()
            and comm.startswith(("python", "mmseqs"))
            and any(s in cmd.lower() for s in ["run_e015", "run_e007", "run_e012", "colabfold_batch", "train_sac"])
        ):
            scientific.append(proc.name)
    if scientific:
        raise SystemExit("BLOCKED_WORKER_CONFLICT")
    # Native host scientific-process query. No broad filesystem scan; fail closed if interop unavailable.
    native = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe' OR Name='mmseqs.exe'\" | Select-Object ProcessId,Name | ConvertTo-Json -Compress",
        ],
        capture_output=True,
        text=True,
    )
    if native.returncode or native.stdout.strip():
        raise SystemExit("BLOCKED_NATIVE_WORKER_OR_UNKNOWN")
    lock = root / "scratch/exclusive_gpu_benchmark.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise SystemExit("BLOCKED_EXCLUSIVE_LOCK")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        import torch
        from protein_joint_sac.contracts import BuildAction, GeneratedProtein, BuildEpisode
        from protein_joint_sac.geometry.ca_builder import replay, mask_for_step
        from protein_joint_sac.models.dense_geometry_attention import ModelConfig
        from protein_joint_sac.models.joint_actor import JointActor
        from protein_joint_sac.models.build_critic import TwinCritics
        from protein_joint_sac.rl.hybrid_sac_build import HybridSAC
        from protein_joint_sac.rl.delayed_episode_replay import EpisodeReplay

        torch.set_num_threads(1)
        device = torch.device(args.device)
        torch.manual_seed(1616)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise SystemExit("BLOCKED_CUDA_UNAVAILABLE")
            free, total = torch.cuda.mem_get_info()
            if free < max(x["SAC_peak_budget_bytes"] for x in estimates) + 1024**3:
                raise SystemExit("BLOCKED_VRAM_HEADROOM")
            torch.cuda.set_per_process_memory_fraction(card["max_peak_allocated_bytes"] / total)
        results = []
        for N in args.lengths:
            if time.monotonic() - start > card["max_wall_seconds"]:
                raise RuntimeError("BLOCKED_WALL_BUDGET")
            # Fixed synthetic geometry is a benchmark fixture, never a fitted protein prior.
            actions = tuple(
                BuildAction(
                    "A",
                    torch.tensor(
                        [3.8 if i else 0.0, 1.9 if i >= 2 else 0.0, 0.4 if i >= 3 else 0.0],
                        dtype=torch.float64,
                        device=device,
                    ),
                    mask_for_step(i),
                )
                for i in range(N)
            )
            state = replay(N, actions[:-1], device=device)
            c = ModelConfig(width=args.width, heads=args.heads, layers=args.layers, backend=args.backend)
            actor = JointActor(c).to(device)
            q = TwinCritics(ModelConfig(width=96, heads=4, layers=3, backend=args.backend)).to(device)
            alg = HybridSAC(actor, q)

            def timed(name, fn):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t = time.monotonic()
                value = fn()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                row = {
                    "N": N,
                    "operation": name,
                    "wall_seconds": time.monotonic() - t,
                    "device": args.device,
                    "dtype": "FP32_models_FP64_geometry_no_autocast",
                    "RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                }
                if args.device == "cuda":
                    row.update(
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                # Preserve measurements even when a guard rejects the operation.
                with out.with_suffix('.guard_telemetry.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
                if row.get("peak_allocated_bytes", 0) > card["max_peak_allocated_bytes"]:
                    raise RuntimeError("BLOCKED_PEAK_VRAM")
                if row["RSS_bytes"] > card["max_RSS_bytes"] or time.monotonic() - start > card["max_wall_seconds"]:
                    raise RuntimeError("BLOCKED_RAM_OR_WALL")
                row['operations_per_second'] = 1 / row['wall_seconds']
                row['residues_per_second'] = N / row['wall_seconds']
                with out.with_suffix('.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n'); f.flush(); os.fsync(f.fileno())
                print(json.dumps(row), flush=True)
                results.append(row)
                return value

            from benchmark_n64_checks import prechecks, gradient_check, postchecks
            timed('feature_device_mask_and_checkpoint_checks', lambda: prechecks(state, actor, q, root))
            actor.zero_grad(set_to_none=True)
            timed("actor_backward", lambda: actor.logits(state).square().mean().backward())
            q.zero_grad(set_to_none=True)
            timed(
                "independent_Q_twins_backward", lambda: (q.q1(state, actions[-1]) + q.q2(state, actions[-1])).backward()
            )
            actor.zero_grad(set_to_none=True)
            if args.sac_strategy == "streamed":
                timed("exact20_SAC_actor_backward_no_step", lambda: alg.backward_actor_exact_streamed(state))
            else:
                for parameter in list(q.q1.parameters()) + list(q.q2.parameters()):
                    parameter.requires_grad_(False)
                timed("eager20_SAC_actor_backward_no_step", lambda: alg.actor_loss(state).backward())
                for parameter in list(q.q1.parameters()) + list(q.q2.parameters()):
                    parameter.requires_grad_(True)
            timed('eager_streamed_exact_gradient_agreement', lambda: gradient_check(state, actor, q, alg, root))
            q.zero_grad(set_to_none=True)
            with torch.no_grad():
                target = alg.soft_value(state, targets=True, estimator="sampled")
            timed(
                "SAC_critic_backward_no_step",
                lambda: (
                    (q.q1(state, actions[-1]) - target).square() + (q.q2(state, actions[-1]) - target).square()
                ).backward(),
            )
            timed("action_replay_reconstruction", lambda: replay(N, actions, device=device))
            replaydb = EpisodeReplay(root / "scratch" / f"benchmark_replay_{N}.sqlite")
            replaydb.store(
                BuildEpisode(f"benchmark-{N}", N, actions), GeneratedProtein(replay(N, actions, device=device))
            )
            replaydb.close()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in list(actor.parameters()) + list(q.parameters())
            ):
                raise RuntimeError("NONFINITE_GRADIENT")
            timed('synthetic_optimizer_target_and_portable_replay_checks', lambda: postchecks(state, actor, q, alg, actions, root))
            actor = q = alg = state = None
            if args.device == "cuda":
                torch.cuda.empty_cache()
        out.write_text(
            json.dumps(
                {"executed": True, "synthetic_only": True, "optimizer_steps": {"actor":1,"critic":1,"temperature":0}, "AF_calls": 0, "results": results},
                indent=2,
            )
            + "\n"
        )
    except BaseException as exc:
        out.with_suffix('.failure.json').write_text(json.dumps({'status':'STOP_FAILED','error':repr(exc),'partial_results':locals().get('results',[]),'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-start,'CUDA_allocated_peak_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,'CUDA_reserved_peak_bytes':torch.cuda.max_memory_reserved() if args.device=='cuda' else None},indent=2))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
