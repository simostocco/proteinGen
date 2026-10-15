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
    raise SystemExit('LEGACY_EXECUTOR_RETIRED_USE_run_cuda_ladder.py_WITH_LIVE_RESOURCE_MONITORING')


if __name__ == '__main__':
    main()
