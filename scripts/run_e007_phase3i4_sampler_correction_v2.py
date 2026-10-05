#!/usr/bin/env python3
"""Phase 3I.4 v2 read-only planning and bounded mock performance smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.evaluation import e007_phase3i4_performance_v2 as kernels
from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction_v2 as lifecycle


def _validate_v2_metadata(cfg: dict) -> dict:
    protocol_path = Path(cfg["protocol_file"])
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != cfg["protocol_sha256"]:
        raise ValueError("v2 prospective protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text())
    baseline_protocol = json.loads(Path("configs/e007_phase3i4_protocol_v1.json").read_text())
    if (
        protocol.get("version") != "e007_phase3i4_sampler_correction_v2"
        or protocol.get("selection") != baseline_protocol.get("selection")
        or protocol.get("authorization") != baseline_protocol.get("authorization")
        or protocol.get("primary_checkpoint") != baseline_protocol.get("primary_checkpoint")
    ):
        raise ValueError("v2 protocol changed protected gates, authorization, or checkpoint")
    manifest_path = Path(cfg["panel_candidate_manifest"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != cfg["panel_candidate_sha256"]:
        raise ValueError("v2 panel/candidate manifest hash mismatch")
    panel_manifest = json.loads(manifest_path.read_text())
    if panel_manifest["sparse_timestep_sha256"] != kernels.schedule_hashes():
        raise ValueError("v2 sparse timestep list/hash mismatch")
    reuse_path = Path(cfg["reuse_manifest"])
    if hashlib.sha256(reuse_path.read_bytes()).hexdigest() != cfg["reuse_manifest_sha256"]:
        raise ValueError("reuse manifest hash mismatch")
    reuse = json.loads(reuse_path.read_text())
    if reuse.get("accepted") is not False or reuse.get("reused_unit_count") != 0:
        raise ValueError("reuse contract must remain fail-closed")
    preservation_path = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/phase3i4_performance_correction_v2/preservation_hashes.json"
    )
    preservation = json.loads(preservation_path.read_text())
    root = Path(preservation["v1_staging_tree_root"])
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    if (
        len(files) != preservation["v1_staging_file_count"]
        or digest.hexdigest() != preservation["v1_staging_tree_sha256"]
    ):
        raise ValueError("protected v1 staging tree changed")
    if hashlib.sha256((root / "journal.jsonl").read_bytes()).hexdigest() != preservation["v1_journal_sha256"]:
        raise ValueError("protected v1 journal changed")
    for name, expected in preservation["v1_logs"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"protected v1 log changed: {name}")
    requested_log = Path("logs/e007_phase3i4_sampler_correction_calibration_v2.log")
    requested_log_hash = "1424e3ebbe679122020f4fb8e2de890d3cb0a1015f2100754e2ccdfa465a0fa4"
    if requested_log.is_symlink() or hashlib.sha256(requested_log.read_bytes()).hexdigest() != requested_log_hash:
        raise ValueError("protected v2 calibration failure log changed")
    cuda_report = Path(
        "reports/experiments/E007_matrix_sequence_cogeneration/phase3i4_cuda_performance_smoke_v2/report.json"
    )
    cuda_hash = "1ae78012819dfa9a54c005933908dfdca64b2d6f84c09480e584cd5ae8a7f7a5"
    if hashlib.sha256(cuda_report.read_bytes()).hexdigest() != cuda_hash:
        raise ValueError("pinned CUDA performance-smoke report changed")
    for path_key in ("calibration_output_dir", "calibration_staging_dir", "holdout_output_dir", "holdout_staging_dir"):
        if "_v1" in Path(cfg[path_key]).name:
            raise ValueError("v2 workflow refuses v1 output/staging paths")
    return {
        "v2_calibration_failure_log_sha256": requested_log_hash,
        "cuda_performance_smoke_report_sha256": cuda_hash,
        "runtime_projection_70_trajectories_seconds": 1753.4892744252284,
        "v1_tree_sha256": digest.hexdigest(),
        "v1_journal_sha256": preservation["v1_journal_sha256"],
        "v1_logs": preservation["v1_logs"],
    }


def _smoke(cfg: dict) -> dict:
    """One length-256 synthetic trajectory fragment; no model/checkpoint/data."""
    torch.manual_seed(20260927)
    n, device = 256, torch.device("cpu")
    x = torch.randn((1, n, 3), device=device)
    x = x - x.mean(1, keepdim=True)
    mask = torch.ones((1, n), dtype=torch.bool, device=device)
    mock_weight = torch.randn((3, 3), device=device)
    kernels.prepare_cache(n, x.dtype, device)
    times_forward: list[float] = []
    times_correction: list[float] = []
    apps = 0
    scheduled = kernels.sparse_timesteps(425)
    # Thirty two fragment steps, representative early/mid/late scheduled calls.
    fragment = [t for t in scheduled if t in {425, 409, 250, 150, 142, 50, 48, 2, 0}]
    for _t in fragment:
        begin = time.perf_counter()
        _v_hat = x @ mock_weight
        times_forward.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        x = kernels.project_x0(
            x,
            mask,
            family="composite",
            iterations=2,
            displacement_cap=cfg["grid"]["displacement_cap_angstrom"],
            scheduled_clash=True,
        )
        times_correction.append(time.perf_counter() - begin)
        apps += 1
    return {
        "status": "bounded_cpu_mock_smoke_no_scientific_selection",
        "trajectory_fragment_length": n,
        "fragment_timesteps": fragment,
        "model_forward_mock_seconds_total": sum(times_forward),
        "model_forward_mock_seconds_per_step": sum(times_forward) / max(len(times_forward), 1),
        "correction_kernel_seconds_total": sum(times_correction),
        "correction_kernel_seconds_per_application": sum(times_correction) / max(apps, 1),
        "correction_applications_in_fragment": apps,
        "planned_correction_applications_per_candidate": {
            str(start): {
                "scheduled_reverse_steps_including_zero_effect_start": len(kernels.sparse_timesteps(start)),
                "effective_projection_calls_per_trajectory": len(kernels.sparse_timesteps(start)) - 1,
                "kernel_iteration_calls_per_trajectory": (len(kernels.sparse_timesteps(start)) - 1)
                * cfg["grid"]["iterations"],
            }
            for start in (425, 250, 150)
        },
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_cuda_memory_bytes": None,
        "cuda_initialized": False,
        "checkpoint_loaded": False,
        "scientific_selection_performed": False,
        **kernels.NON_AUTHORIZING,
    }


def _cuda_smoke(cfg: dict) -> dict:
    """Four non-authorizing end-to-end trajectories; never reads/reuses published rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("--cuda-performance-smoke requires CUDA")
    from protein_distance_diffusion.evaluation import e007_phase3i4_sampler_correction as prod
    from protein_distance_diffusion.evaluation.e007_phase3i3_trajectory import (
        _load_model,
        _parameter_fingerprints,
        _representation_record,
    )
    from protein_distance_diffusion.models.coordinate_equivariance import coordinate_model_execution_context
    from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion

    smoke_started = time.perf_counter()
    device = torch.device("cuda")
    model_cfg = dict(cfg)
    model_cfg["checkpoints"] = {"v_only": cfg["checkpoint"]}
    t0 = time.perf_counter()
    model = _load_model(model_cfg, "v_only", device)
    diffusion = CoordinateVPDiffusion(cfg["timesteps"])
    startup_seconds = time.perf_counter() - t0
    parameter_hashes = _parameter_fingerprints(model)
    schedules = {name: kernels.sparse_timesteps(425) for name in ("control", "composite_t425")}
    units = [
        {**identity, "candidate": name}
        for length in (256, 500)
        for identity in [
            {"identity": f"smoke-cal-n{length}-00", "length": length, "seed": cfg["panel"]["seed_base"] + length * 100}
        ]
        for name in ("control", "composite_t425")
    ]
    totals = {k: 0.0 for k in ("forward", "correction", "transition", "metric_serialization")}
    application_count = 0
    trajectories = []
    torch.cuda.reset_peak_memory_stats(device)
    with coordinate_model_execution_context(cfg["numerics"], device):
        for unit in units:
            n = unit["length"]
            mask = torch.ones((1, n), dtype=torch.bool, device=device)
            lengths = torch.tensor([n], dtype=torch.long, device=device)
            x = prod.paired_initial_noise(unit["seed"], n, device=device)
            initial_hash = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
            kernels.prepare_cache(n, x.dtype, device, starts=(425,))
            model_seconds = correction_seconds = transition_seconds = 0.0
            with torch.inference_mode():
                for step in range(499, -1, -1):
                    t = torch.tensor([step], dtype=torch.long, device=device)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    v = model(x, t, lengths, mask, mask[:, :-1])["v_prediction"]
                    torch.cuda.synchronize()
                    model_seconds += time.perf_counter() - begin
                    if unit["candidate"] == "control" or step not in schedules["composite_t425"] or step == 425:
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        next_state, _, _ = diffusion.deterministic_reverse_step(x, t, v, mask)
                        torch.cuda.synchronize()
                        transition_seconds += time.perf_counter() - begin
                    else:
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        x0 = diffusion.reconstruct_x0(x, t, v, mask)
                        projected = (
                            kernels.project_x0(
                                x0 * cfg["coordinate_scale_angstrom"],
                                mask,
                                family="composite",
                                iterations=cfg["grid"]["iterations"],
                                displacement_cap=cfg["grid"]["displacement_cap_angstrom"],
                                scheduled_clash=step in schedules["composite_t425"],
                            )
                            / cfg["coordinate_scale_angstrom"]
                        )
                        frac = prod.schedule_fraction(step, 425, cfg["grid"]["signal_fraction"])
                        guided = (1 - frac) * x0 + frac * projected
                        alpha, sigma = diffusion.alpha_sigma(t, x)
                        vg = (alpha * x - guided) / sigma.clamp_min(1e-8)
                        torch.cuda.synchronize()
                        correction_seconds += time.perf_counter() - begin
                        application_count += 1
                        torch.cuda.synchronize()
                        begin = time.perf_counter()
                        next_state, _, _ = diffusion.deterministic_reverse_step(x, t, vg, mask)
                        torch.cuda.synchronize()
                        transition_seconds += time.perf_counter() - begin
                    x = next_state
            totals["forward"] += model_seconds
            totals["correction"] += correction_seconds
            totals["transition"] += transition_seconds
            torch.cuda.synchronize()
            begin = time.perf_counter()
            values = x[0].detach().cpu().numpy()
            metric = _representation_record(values, cfg)
            try:
                json.dumps(metric, sort_keys=True, allow_nan=False)
                finite = bool(np.isfinite(values).all())
            except (TypeError, ValueError):
                finite = False
            trajectories.append(
                {
                    "identity": unit["identity"],
                    "length": n,
                    "candidate": unit["candidate"],
                    "seed": unit["seed"],
                    "initial_noise_sha256": initial_hash,
                    "finite": finite,
                    "metrics": metric,
                }
            )
            json.dumps(trajectories[-1], sort_keys=True, allow_nan=False)
            totals["metric_serialization"] += time.perf_counter() - begin
    if parameter_hashes != _parameter_fingerprints(model):
        raise RuntimeError("frozen model parameter mutation detected")
    if not all(row["finite"] for row in trajectories):
        raise FloatingPointError("non-finite CUDA smoke trajectory/metric")
    # Pairing is explicitly verified by identical seed and generated noise digest.
    for length in (256, 500):
        pair = [r for r in trajectories if r["length"] == length]
        if len(pair) != 2 or pair[0]["initial_noise_sha256"] != pair[1]["initial_noise_sha256"]:
            raise RuntimeError("paired initial noise mismatch")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    allocated, reserved = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)
    elapsed = time.perf_counter() - smoke_started
    per_trajectory = elapsed / len(units)
    report = {
        "status": "bounded_cuda_end_to_end_smoke_no_scientific_selection",
        "startup_seconds": startup_seconds,
        "model_forward_seconds": totals["forward"],
        "correction_kernel_seconds": totals["correction"],
        "sampler_transition_seconds": totals["transition"],
        "metric_serialization_seconds": totals["metric_serialization"],
        "total_seconds": elapsed,
        "total_seconds_per_trajectory": per_trajectory,
        "projected_70_trajectory_calibration_seconds": per_trajectory * 70,
        "model_forward_count": 2000,
        "trajectory_count": 4,
        "correction_applications": application_count,
        "correction_applications_by_candidate": {"control": 0, "composite_t425": len(schedules["composite_t425"]) - 1},
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "rss_peak_mib": rss,
        "correction_device": "cuda",
        "reverse_loop_host_device_transfers": False,
        "autograd_graphs": False,
        "backward_pass": False,
        "parameter_mutation": False,
        "paired_noise_verified": True,
        "finite_completion": True,
        "trajectories": trajectories,
        "checkpoint_sha256": hashlib.sha256(Path(cfg["checkpoint"]).read_bytes()).hexdigest(),
        "scientific_selection_performed": False,
        "calibration_or_holdout_reuse": False,
        "output_path": "reports/experiments/E007_matrix_sequence_cogeneration/phase3i4_cuda_performance_smoke_v2",
        **kernels.NON_AUTHORIZING,
    }
    out = Path(report["output_path"])
    if out.exists():
        raise FileExistsError(f"smoke output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        (tmp / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp, out)
    except Exception:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e007_phase3i4_sampler_correction_v2.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--validate-contract", action="store_true")
    mode.add_argument("--performance-smoke", action="store_true")
    mode.add_argument(
        "--cuda-performance-smoke",
        action="store_true",
        help="Run four bounded CUDA production-sampler trajectories; no selection or artifact reuse",
    )
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    mode.add_argument("--monitor", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("version") != "e007_phase3i4_sampler_correction_v2":
        raise ValueError("v2 config required")
    protected = _validate_v2_metadata(cfg)
    if args.performance_smoke:
        result = _smoke(cfg)
    elif args.cuda_performance_smoke:
        result = _cuda_smoke(cfg)
    elif args.validate_contract:
        result = lifecycle.validate_contract(args.config)
        result["v2_preservation_hashes"] = protected
    elif args.monitor:
        result = lifecycle.monitor(args.config)
    elif args.calibrate:
        result = lifecycle.execute(args.config, "calibration")
    elif args.holdout:
        result = lifecycle.execute(args.config, "holdout")
    elif args.resume:
        result = lifecycle.resume(args.config)
    else:
        result = lifecycle.plan(args.config)
        result["runtime_projection_70_trajectories_seconds"] = 1753.4892744252284
        result["cuda_performance_smoke_report_sha256"] = protected["cuda_performance_smoke_report_sha256"]
        result["v2_preservation_hashes"] = protected
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
