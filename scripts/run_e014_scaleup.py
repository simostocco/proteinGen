"""Complete-epoch E014 trainer, exact resume and sequence contention guard."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import argparse
import json
import os
import shutil
import signal
import fcntl
import subprocess
import sys
import time
import random
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.data.e014_population import CoordinateCache, verify
from protein_distance_diffusion.evaluation.e014_evaluation import (
    coordinate_batch,
    evaluate,
    primary_v,
    adjudicate_final,
)
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.training.e014_runtime import (
    atomic_checkpoint,
    atomic_json,
    canonical_hash,
    epoch_schedule,
    load_checkpoint,
    noise_seed,
    restore_rng,
    rng_state,
    sha,
)
from protein_distance_diffusion.training.e014_contention import baseline, compare

STOP = False


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def optimizer_statistics(model, before=None):
    groups = {"trunk": [], "local_branch": [], "output_heads": []}
    for name, p in model.named_parameters():
        group = (
            "output_heads"
            if name.startswith(("head.", "local_head."))
            else "local_branch"
            if name.startswith(("local_", "gate."))
            else "trunk"
        )
        groups[group].append((name, p))
    output = {}
    for group, params in groups.items():
        if not params:
            output[group] = dict(parameter_norm=0.0, gradient_norm=None, optimizer_step_norm=None)
            continue
        output[group] = dict(
            parameter_norm=float(torch.stack([p.detach().square().sum() for _, p in params]).sum().sqrt()),
            gradient_norm=float(
                torch.stack([p.grad.detach().square().sum() for _, p in params if p.grad is not None]).sum().sqrt()
            )
            if any(p.grad is not None for _, p in params)
            else None,
            optimizer_step_norm=float(
                torch.stack([(p.detach() - before[n]).square().sum() for n, p in params]).sum().sqrt()
            )
            if before
            else None,
        )
    return output


def symmetry_check(model):
    if not isinstance(model, E014CapacityDenoiser):
        return dict(applicable=False, passed=True)
    with torch.no_grad():
        x = torch.randn(1, 500, 3, device="cuda", generator=torch.Generator(device="cuda").manual_seed(140015))
        t = torch.tensor([50], device="cuda")
        lengths = torch.tensor([500], device="cuda")
        mask = torch.ones(1, 500, dtype=torch.bool, device="cuda")
        r = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]], device="cuda")
        y = model(x, t, lengths, mask)
        delta = model(x @ r + 2, t, lengths, mask) - y @ r
        result = dict(max_abs=float(delta.abs().max()), relative_l2=float(delta.norm() / (y.norm() + 1e-12)))
        result["passed"] = result["max_abs"] <= 5e-5 and result["relative_l2"] <= 5e-5
        if not result["passed"]:
            raise ValueError("trained symmetry regression; hard stop")
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", choices=["e014", "e007_control"], default="e014")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_requested)
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
    torch.set_num_threads(cfg["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    capacity = json.loads((ROOT / "capacity_selection.json").read_text())
    population = json.loads((ROOT / "population/population.json").read_text())
    batch_preflight = json.loads((ROOT / "EXECUTION_BOUNDARY_PREFLIGHT.json").read_text())
    if (
        not capacity["passed"]
        or capacity["config_sha256"] != sha(cfgpath)
        or capacity["population_hash"] != population["train_sample_id_hash"]
        or not batch_preflight["passed"]
        or batch_preflight["config_hash"] != sha(cfgpath)
        or batch_preflight["train_population_hash"] != population["train_sample_id_hash"]
        or batch_preflight["selected_parameters"] != capacity["selected"]["parameters"]
    ):
        raise ValueError("capacity/config/population prerequisite mismatch")
    for path, digest in population["manifest_hashes"].items():
        verify(path, digest)
    for path, digest in population["source_hashes"].items():
        verify(path, digest)
    runtime_sources = [Path("scripts/run_e014_scaleup.py"), Path("scripts/e014_bootstrap.py")]
    runtime_sources.extend(Path("src/protein_distance_diffusion").glob("**/e014_*.py"))
    runtime_sources.extend(
        [
            Path("src/protein_distance_diffusion/models/e013_unconditional_equivariant_denoiser.py"),
            Path("src/protein_distance_diffusion/training/coordinate_diffusion.py"),
            Path("src/protein_distance_diffusion/training/e013_losses.py"),
            Path("src/protein_distance_diffusion/evaluation/e013_one_step_benchmark.py"),
        ]
    )
    runtime_sources.sort()
    contract = dict(
        config_hash=sha(cfgpath),
        data_hash=population["train_sample_id_hash"],
        normalization_hash=cfg["normalization_sha256"],
        capacity_config=capacity["selected"]["config"],
        population_manifest_hash=sha(ROOT / "population/population.json"),
        cache_ready_hash=sha(ROOT / "coordinate_cache/READY.json"),
        model=args.model,
        source_hashes={str(p): sha(p) for p in runtime_sources},
    )
    run = ROOT / args.model
    if run.exists() and not args.resume:
        raise FileExistsError("existing primary run; exact --resume required")
    run.mkdir(exist_ok=args.resume)
    lock = open(run / "OWNER.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another E014 trainer owns this run; refusing concurrent resume") from exc
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()) + "\n")
    lock.flush()
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    cache = CoordinateCache(ROOT)
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    dev = pq.read_table(ROOT / "population/development.parquet", use_threads=False).to_pylist()
    prospective = pq.read_table(ROOT / "population/prospective.parquet", use_threads=False).to_pylist()
    historical = pq.read_table(ROOT / "population/historical.parquet", use_threads=False).to_pylist()
    lengths = np.asarray([r["length"] for r in train])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    if args.model == "e014":
        model = E014CapacityDenoiser(**capacity["selected"]["config"]).cuda()
    else:
        if not (ROOT / "e014/FINAL_REPORT.json").exists():
            raise ValueError("matched control cannot precede primary completion")
        if not (ROOT / "MATCHED_CONTROL_RESOURCE_APPROVED.json").exists():
            raise ValueError("post-primary compute/resource assessment must precede matched-control execution")
        model = E007ScaleupControl(
            **yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
        ).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    diffusion = CoordinateVPDiffusion(cfg["diffusion_steps"])
    exposures = np.zeros(len(train), dtype=np.int32)
    epoch, position, updates = 1, 0, 0
    curves = []
    train_loss_sum = 0.0
    train_loss_proteins = 0
    epoch_loss_sum = 0.0
    epoch_loss_proteins = 0
    clipping_count = 0
    elapsed_previous = 0.0
    gpu_seconds_previous = 0.0
    probe_passed = False
    probe_start = None
    before = None
    pending_boundary = None
    optimizer_update_in_progress = False
    start = time.monotonic()
    gpu_seconds = 0.0
    recovery_slot = 0
    last_checkpoint = time.monotonic()
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    attempt = f"{time.time_ns()}"
    if args.resume:
        state = load_checkpoint(checkpoints, contract)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, position, updates = state["epoch"], state["position"], state["updates"]
        exposures = state["exposures"].copy()
        curves = state["curves"]
        train_loss_sum, train_loss_proteins = state["train_loss_sum"], state["train_loss_proteins"]
        epoch_loss_sum, epoch_loss_proteins = state["epoch_loss_sum"], state["epoch_loss_proteins"]
        clipping_count = state["clipping_count"]
        elapsed_previous, gpu_seconds_previous = state["elapsed_seconds"], state["gpu_seconds"]
        probe_passed = state["probe_passed"]
        pending_boundary = state.get("pending_boundary")
        before = state.get("contention_baseline")
        restore_rng(state["rng"])
        atomic_json(
            run / f"RESUME_{attempt}.json",
            dict(
                resumed_committed_update=updates,
                epoch=epoch,
                position=position,
                explicit_rollback_policy=(
                    "records after LATEST committed checkpoint are provisional; "
                    "they are never counted as committed exposure"
                ),
                attempt=attempt,
                exact_next_batch_verified=True,
            ),
        )
    else:
        atomic_json(
            run / "RUN_CONTRACT.json",
            dict(
                contract=contract,
                git_sha=git_sha,
                capacity=capacity["selected"],
                torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda,
                python_version=sys.version,
                numpy_version=np.__version__,
                rng_seed=cfg["seed"],
                physical_batch_preflight_hash=sha(ROOT / "EXECUTION_BOUNDARY_PREFLIGHT.json"),
                optimizer=cfg["optimizer"],
                train_population=population,
                minimum_epochs=2,
                maximum_epochs=4,
                interpretation_cases={
                    "A": "v and geometry improve substantially",
                    "B": "v improves while geometry divergence persists",
                    "C": "v fails to improve",
                    "D": "high-noise improves, low-noise refinement remains poor",
                    "E": "low-noise improves, high-noise prior remains poor",
                },
                all_generated_paths_on_d=True,
            ),
        )
    schedule = []

    def save(name, rolling=False):
        nonlocal last_checkpoint
        payload = dict(
            contract=contract,
            git_sha=git_sha,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            position=position,
            updates=updates,
            exposures=exposures.copy(),
            rng=rng_state(),
            schedule_hash=canonical_hash(schedule),
            next_batch=schedule[position] if position < len(schedule) else None,
            curves=curves,
            train_loss_sum=train_loss_sum,
            train_loss_proteins=train_loss_proteins,
            epoch_loss_sum=epoch_loss_sum,
            epoch_loss_proteins=epoch_loss_proteins,
            clipping_count=clipping_count,
            elapsed_seconds=elapsed_previous + time.monotonic() - start,
            gpu_seconds=gpu_seconds_previous + gpu_seconds,
            probe_passed=probe_passed,
            attempt=attempt,
        )
        payload["pending_boundary"] = pending_boundary
        payload["contention_baseline"] = before
        path = atomic_checkpoint(checkpoints, name, payload, rolling=rolling)
        last_checkpoint = time.monotonic()
        atomic_json(
            run / "COMMITTED.json",
            dict(
                checkpoint=str(path),
                epoch=epoch,
                position=position,
                updates=updates,
                total_protein_exposures=int(exposures.sum()),
                minimum_exposure=int(exposures.min()),
                maximum_exposure=int(exposures.max()),
            ),
        )
        return path

    def heartbeat(phase, loss=None):
        atomic_json(
            run / "HEARTBEAT.json",
            dict(
                phase=phase,
                epoch=epoch,
                update=updates,
                position=position,
                identities_seen_this_epoch=int((exposures == epoch).sum()),
                total_identities=len(train),
                total_protein_exposures=int(exposures.sum()),
                exposure_min=int(exposures.min()),
                exposure_max=int(exposures.max()),
                loss=loss,
                gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(),
                proteins_per_second=int(exposures.sum()) / max(1.0, elapsed_previous + time.monotonic() - start),
                timestamp_unix=time.time(),
                probe_passed=probe_passed,
                git_sha=git_sha,
                attempt=attempt,
            ),
        )

    def check_free_disk():
        if shutil.disk_usage(ROOT).free < cfg["minimum_free_disk_gib"] * 2**30:
            raise RuntimeError("D: capacity danger; pausing structural run")

    # Initial diagnostics never select capacity; panels and model are frozen.
    schedule = epoch_schedule(lengths, cfg["batch_sizes"], cfg["seed"], epoch)
    if args.resume:
        if state["schedule_hash"] != canonical_hash(schedule) or state["next_batch"] != (
            schedule[position] if position < len(schedule) else None
        ):
            raise ValueError("exact next batch/schedule resume verification failed")
        expected = np.full(len(train), epoch - 1, dtype=np.int32)
        for batch in schedule[:position]:
            expected[batch] += 1
        if not np.array_equal(expected, exposures):
            raise ValueError("resume exposure accounting contradiction")
    elif updates == 0:
        save("initialization")
    metrics_dir = run / "evaluation"
    heartbeat("initialization_evaluation" if updates == 0 else "resuming")
    initial = (
        evaluate(model, diffusion, dev, cache, cfg, metrics_dir, "initialization_development")
        if updates == 0
        else json.loads((metrics_dir / "initialization_development.summary.json").read_text())
    )
    if updates == 0:
        evaluate(model, diffusion, historical, cache, cfg, metrics_dir, "initialization_historical", historical=True)
        if not curves:
            curves.append(
                dict(
                    epoch=0.0, updates=0, protein_exposures=0, development_v_mse=primary_v(initial), training_v_mse=None
                )
            )
        save("recovery_0", rolling=True)
    if pending_boundary:
        summary = evaluate(model, diffusion, dev, cache, cfg, metrics_dir, pending_boundary + "_development")
        seen = int((exposures == epoch).sum())
        curves.append(
            dict(
                epoch=epoch - 1 + seen / len(train),
                updates=updates,
                protein_exposures=int(exposures.sum()),
                development_v_mse=primary_v(summary),
                training_v_mse=epoch_loss_sum / max(1, epoch_loss_proteins),
                resumed_pending_boundary=pending_boundary,
                symmetry=symmetry_check(model),
            )
        )
        pending_boundary = None
        atomic_json(run / "CURVES.json", curves)
        save("recovery_0", rolling=True)
    log = open(run / "TRAINING.jsonl", "a", buffering=1)
    last_guard = 0.0
    try:
        while epoch <= cfg["maximum_epochs"]:
            schedule = epoch_schedule(lengths, cfg["batch_sizes"], cfg["seed"], epoch)
            schedule_path = run / f"EPOCH_{epoch}_SCHEDULE.json"
            if schedule_path.exists():
                if canonical_hash(json.loads(schedule_path.read_text())) != canonical_hash(schedule):
                    raise ValueError("persisted deterministic epoch schedule mismatch")
            else:
                atomic_json(schedule_path, schedule)
            if not probe_passed and before is None:
                before = baseline(cfg["contention"])
                probe_start = time.monotonic()
                atomic_json(run / f"CONTENTION_BASELINE_{attempt}.json", before)
            if not probe_passed and probe_start is None:
                probe_start = time.monotonic()
            thresholds = [int(np.ceil(f * len(train))) for f in cfg["checkpoint_fractions"]]
            seen = int((exposures == epoch).sum())
            crossed = {i for i, n in enumerate(thresholds) if seen >= n}
            while position < len(schedule):
                if STOP:
                    save(f"pause_{attempt}_{updates}")
                    heartbeat("paused_signal")
                    return
                check_free_disk()
                batch_ids = schedule[position]
                rows = [train[i] for i in batch_ids]
                if any(exposures[i] != epoch - 1 for i in batch_ids):
                    raise ValueError("duplicate/missing TRAIN exposure in epoch")
                x, length_tensor, mask = coordinate_batch(rows, cache, cfg["coordinate_scale_angstrom"])
                # Per-identity/epoch RNG streams guarantee comparator corruptions,
                # independent of batching and checkpoint boundaries.
                noisy = []
                targets = []
                ts = []
                for i, row in enumerate(rows):
                    n = row["length"]
                    g = torch.Generator(device="cuda").manual_seed(
                        noise_seed(f"e014-train-epoch-{epoch}", row["sample_id"], 0)
                    )
                    corrupt = diffusion.make_training_batch(x[i : i + 1, :n], mask[i : i + 1, :n], generator=g)
                    noisy.append(torch.nn.functional.pad(corrupt.noisy_coordinates, (0, 0, 0, x.shape[1] - n)))
                    targets.append(torch.nn.functional.pad(corrupt.coordinate_v_target, (0, 0, 0, x.shape[1] - n)))
                    ts.append(corrupt.timesteps)
                xt, target, t = torch.cat(noisy), torch.cat(targets), torch.cat(ts)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                telemetry_step = updates % 500 == 0
                before_params = {n: p.detach().clone() for n, p in model.named_parameters()} if telemetry_step else None
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
                prediction = model(xt, t, length_tensor, mask)
                loss = coordinate_v_loss(prediction, target, mask)
                if not bool(torch.isfinite(loss)) or float(loss) > cfg["hard_loss_maximum"]:
                    raise ValueError("nonfinite/exploding primary loss; no optimizer retuning")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["optimizer"]["gradient_clip"], error_if_nonfinite=True
                )
                raw_grad_norm = float(grad_norm)
                clipping_count += raw_grad_norm > cfg["optimizer"]["gradient_clip"]
                optimizer_update_in_progress = True
                optimizer.step()
                event_end.record()
                torch.cuda.synchronize()
                gpu_seconds += event_start.elapsed_time(event_end) / 1000
                updates += 1
                position += 1
                exposures[batch_ids] += 1
                optimizer_update_in_progress = False
                seen += len(batch_ids)
                train_loss_sum += float(loss) * len(batch_ids)
                train_loss_proteins += len(batch_ids)
                epoch_loss_sum += float(loss) * len(batch_ids)
                epoch_loss_proteins += len(batch_ids)
                record = dict(
                    epoch=epoch,
                    update=updates,
                    position=position,
                    identity_count=len(batch_ids),
                    sample_ids=[r["sample_id"] for r in rows],
                    loss=float(loss),
                    gradient_norm=raw_grad_norm,
                    clipped=raw_grad_norm > 1.0,
                    padding_fraction=1 - float(mask.sum()) / mask.numel(),
                    attempt=attempt,
                    commit_status="provisional_until_checkpoint",
                )
                if telemetry_step:
                    record["module_statistics"] = optimizer_statistics(model, before_params)
                log.write(json.dumps(record, allow_nan=False) + "\n")
                if updates % 20 == 0:
                    heartbeat("training", float(loss))
                if updates % 200 == 0:
                    print(json.dumps(record), flush=True)
                now = time.monotonic()
                if now - last_checkpoint >= cfg["extra_recovery_checkpoint_seconds"]:
                    recovery_slot = 1 - recovery_slot
                    save(f"recovery_{recovery_slot}", rolling=True)
                if now - last_guard >= 30:
                    last_guard = now
                    status = compare(before, cfg["contention"]) if before else None
                    if status:
                        atomic_json(run / "CONTENTION_CURRENT.json", status)
                        screen_finished = status["phase"] in ("completed", "complete", "screen_complete", "finished")
                        stale = (not screen_finished) and time.time() - status["timestamp_unix"] > cfg["contention"][
                            "stale_heartbeat_seconds"
                        ]
                        if status["persistent_material_degradation"] or stale:
                            save(f"contention_pause_{attempt}_{updates}")
                            heartbeat(
                                "paused_sequence_contention" if not stale else "paused_sequence_heartbeat_uncertain"
                            )
                            return
                    if not probe_passed and now - probe_start >= cfg["contention"]["probe_seconds"]:
                        if not status["enough_completed_batches"]:
                            save(f"probe_inconclusive_{attempt}_{updates}")
                            heartbeat("paused_contention_probe_inconclusive")
                            return
                        probe_passed = True
                        atomic_json(
                            run / "CONTENTION_PROBE.json",
                            dict(passed=True, seconds=now - probe_start, before=before, comparison=status),
                        )
                        save(f"probe_passed_{attempt}_{updates}")
                        if args.probe_only:
                            heartbeat("probe_completed_resumable")
                            return
                for i, threshold in enumerate(thresholds):
                    if i not in crossed and seen >= threshold:
                        crossed.add(i)
                        label = f"epoch_{epoch}_fraction_{cfg['checkpoint_fractions'][i]:.2f}"
                        pending_boundary = label
                        checkpoint = save(label)
                        symmetry = symmetry_check(model)
                        summary = evaluate(model, diffusion, dev, cache, cfg, metrics_dir, label + "_development")
                        boundary = dict(
                            epoch=epoch - 1 + seen / len(train),
                            updates=updates,
                            protein_exposures=int(exposures.sum()),
                            development_v_mse=primary_v(summary),
                            training_v_mse=epoch_loss_sum / epoch_loss_proteins,
                            checkpoint=str(checkpoint),
                            symmetry=symmetry,
                            module_statistics=optimizer_statistics(model),
                            gradient_clipping_fraction=clipping_count / updates,
                        )
                        curves.append(boundary)
                        pending_boundary = None
                        atomic_json(run / "CURVES.json", curves)
                        save(f"recovery_{recovery_slot}", rolling=True)
            if not np.all(exposures == epoch):
                raise ValueError("full epoch failed exact equal-exposure invariant")
            atomic_json(
                run / f"EXPOSURE_EPOCH_{epoch}.json",
                dict(
                    epoch=epoch,
                    identity_count=len(train),
                    minimum=int(exposures.min()),
                    maximum=int(exposures.max()),
                    total_exposures=int(exposures.sum()),
                    duplicates=0,
                    missing=0,
                    updates=updates,
                    schedule_sha256=canonical_hash(schedule),
                    exposure_sha256=canonical_hash(exposures.tolist()),
                ),
            )
            end_summary = evaluate(
                model, diffusion, dev, cache, cfg, metrics_dir, f"epoch_{epoch}_fraction_1.00_development"
            )
            if not any(abs(c["epoch"] - epoch) < 1e-12 for c in curves):
                curves.append(
                    dict(
                        epoch=epoch,
                        updates=updates,
                        protein_exposures=int(exposures.sum()),
                        development_v_mse=primary_v(end_summary),
                        training_v_mse=epoch_loss_sum / max(1, epoch_loss_proteins),
                    )
                )
            if epoch == 1:
                evaluate(model, diffusion, historical, cache, cfg, metrics_dir, "epoch_1_historical", historical=True)
            complete = [c for c in curves if abs(c["epoch"] - round(c["epoch"])) < 1e-12]
            plateau = False
            if epoch >= cfg["minimum_epochs"] and len(complete) >= 3:
                improvements = [
                    1 - complete[-k]["development_v_mse"] / complete[-k - 1]["development_v_mse"] for k in (1, 2)
                ]
                plateau = all(v < cfg["plateau"]["relative_improvement"] for v in improvements)
            save(f"recovery_{recovery_slot}", rolling=True)
            if plateau or epoch == cfg["maximum_epochs"]:
                final = evaluate(model, diffusion, prospective, cache, cfg, metrics_dir, "final_prospective")
                prospective_initial_path = metrics_dir / "initialization_prospective.summary.json"
                if not prospective_initial_path.exists():
                    # No initial prospective metrics were inspected for optimization.
                    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    initial_state = torch.load(
                        checkpoints / "initialization.pt", map_location="cpu", weights_only=False
                    )
                    try:
                        model.load_state_dict(initial_state["model"])
                        prospective_initial = evaluate(
                            model, diffusion, prospective, cache, cfg, metrics_dir, "initialization_prospective"
                        )
                    finally:
                        # A diagnostic failure must never checkpoint the initialization
                        # weights alongside the fully trained optimizer/exposure state.
                        model.load_state_dict(final_state)
                    del final_state, initial_state
                else:
                    prospective_initial = json.loads(prospective_initial_path.read_text())
                evaluate(model, diffusion, historical, cache, cfg, metrics_dir, "final_historical", historical=True)
                final_decision = adjudicate_final(prospective_initial, final, epoch, cfg)
                atomic_json(
                    run / "FINAL_REPORT.json",
                    dict(
                        completed_epochs=epoch,
                        optimizer_updates=updates,
                        total_protein_exposures=int(exposures.sum()),
                        exposure_min=int(exposures.min()),
                        exposure_max=int(exposures.max()),
                        wall_seconds=elapsed_previous + time.monotonic() - start,
                        gpu_training_seconds=gpu_seconds_previous + gpu_seconds,
                        plateau_triggered=plateau,
                        curves=curves,
                        final_prospective=final,
                        decision=final_decision,
                        matched_e007_control_run=False,
                        stage_d_run=False,
                        git_sha=git_sha,
                        contract=contract,
                    ),
                )
                atomic_json(
                    ROOT / "MATCHED_CONTROL_RESOURCE_ASSESSMENT.json",
                    dict(
                        primary_completed=True,
                        run_control=False,
                        reason=(
                            "E007 N=500 reserved memory exceeded the predeclared comfortable 5 GiB envelope; "
                            "prepared control requires a separate resource assessment"
                        ),
                        preflight_gpu_hour_estimates=capacity["estimates"].get("e007_control"),
                        historical_comparisons="non-matched context",
                    ),
                )
                heartbeat("completed")
                try:
                    subprocess.run([sys.executable, "-B", "scripts/finalize_e014_status.py", "--commit"], check=True)
                except subprocess.CalledProcessError as exc:
                    atomic_json(run / "DOCUMENTATION_FAILURE.json", dict(error=str(exc), scientific_run_completed=True))
                    print(f"Scientific run completed; documentation publication failed: {exc}", flush=True)
                return
            epoch += 1
            position = 0
            epoch_loss_sum = 0.0
            epoch_loss_proteins = 0
    except Exception as exc:
        # Last verified committed checkpoint remains authoritative on hard failures.
        try:
            if not optimizer_update_in_progress:
                save(f"hard_stop_{attempt}_{updates}")
        except Exception as save_exc:
            atomic_json(run / f"CHECKPOINT_FAILURE_{attempt}.json", dict(error=str(save_exc)))
        atomic_json(
            run / f"HARD_STOP_{attempt}.json",
            dict(
                error=str(exc),
                epoch=epoch,
                position=position,
                updates=updates,
                architecture_conclusion=None,
                minimum_exposure_completed=epoch >= 3,
            ),
        )
        heartbeat("hard_stop")
        raise
    finally:
        log.close()


if __name__ == "__main__":
    main()
