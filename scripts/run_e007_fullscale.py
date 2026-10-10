"""Unchanged E007, frozen full-data epochs, no prospective access or epoch 3."""

# ruff: noqa: I001
from e007_fullscale_runtime import baseline
from e007_fullscale_runtime import (
    RUN,
    DATA,
    CONFIG,
    MODEL_CONFIG,
    verified_inputs,
    expected_exposures,
    sequence_guard,
    atomic_json,
    canonical_hash,
    epoch_schedule,
    sha,
    torch,
    yaml,
    np,
)
import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, evaluate, primary_v
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import atomic_checkpoint, load_checkpoint, rng_state, restore_rng

STOP = False


def request_stop(_signal, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, request_stop)
    cfg, population, train, dev = verified_inputs()
    profile = json.loads((RUN / "compute_profile.json").read_text())
    assert (
        profile["passed"]
        and profile["config_hash"] == sha(CONFIG)
        and profile["train_population_hash"] == population["train_sample_id_hash"]
    )
    assert [r["physical_batch"] for r in profile["results"]] == [4, 2, 1, 1]
    lock = open(RUN / "OWNER.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    model = E007ScaleupControl(**yaml.safe_load(MODEL_CONFIG.read_text())["model"]).cuda()
    assert sum(p.numel() for p in model.parameters()) == 7586505
    optimizer = torch.optim.AdamW(
        model.parameters(), **dict(lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"])
    )
    diffusion = CoordinateVPDiffusion(500)
    cache = CoordinateCache(DATA)
    checkpoints = RUN / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    sources = sorted(Path("src/protein_distance_diffusion").rglob("*.py")) + [
        Path("scripts") / name
        for name in (
            "run_e007_fullscale.py",
            "e007_fullscale_runtime.py",
            "preflight_e007_fullscale.py",
            "evaluate_e007_fullscale.py",
            "supervise_e007_fullscale.py",
            "e007_e014_comparison_protocol.py",
            "collect_e007_e014_comparison.py",
            "analyze_e007_e014_comparison.py",
            "analyze_e014_marginal_reverse.py",
            "analyze_e014_high_noise.py",
            "collect_e014_marginal_reverse.py",
            "diagnose_e014_reverse_field.py",
        )
    ]
    contract = dict(
        config_hash=sha(CONFIG),
        model_config_hash=sha(MODEL_CONFIG),
        data_hash=population["train_sample_id_hash"],
        normalization_hash=cfg["normalization_sha256"],
        cache_ready_hash=sha(DATA / "coordinate_cache/READY.json"),
        source_hashes={str(p): sha(p) for p in sources},
        maximum_epochs=2,
        physical_batch_sizes=cfg["batch_sizes"],
        accumulation=1,
        owner_authorization=(
            "2026-10-10 explicit matched full-data E007 control; supersedes old "
            "primary-final-stop/resource-approval-file gate, with actual safe CUDA preflight"
        ),
    )
    epoch, position, updates = 1, 0, 0
    exposures = np.zeros(len(train), dtype=np.int32)
    curves, epoch_sum, epoch_count = [], 0.0, 0
    clipping, elapsed_previous, gpu_previous = 0, 0.0, 0.0
    pending = None
    probe_passed = False
    before = json.loads((RUN / "E012_BASELINE.json").read_text())
    if args.resume:
        state = load_checkpoint(checkpoints, contract)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch, position, updates = state["epoch"], state["position"], state["updates"]
        assert all(int(s["step"]) == updates for s in optimizer.state.values())
        exposures = state["exposures"]
        schedule, expected = expected_exposures(
            [r["length"] for r in train], epoch, position, cfg["batch_sizes"], cfg["seed"]
        )
        assert np.array_equal(expected, exposures)
        assert state["schedule_hash"] == canonical_hash(schedule)
        assert state["next_batch"] == (schedule[position] if position < len(schedule) else None)
        restore_rng(state["rng"])
        curves, epoch_sum, epoch_count = state["curves"], state["epoch_loss_sum"], state["epoch_loss_proteins"]
        clipping, elapsed_previous, gpu_previous = (
            state["clipping_count"],
            state["elapsed_seconds"],
            state["gpu_seconds"],
        )
        pending, probe_passed = state["pending_boundary"], state["probe_passed"]
        before = state["contention_baseline"]
        atomic_json(
            RUN / f"RESUME_{time.time_ns()}.json",
            dict(
                epoch=epoch,
                position=position,
                updates=updates,
                total_exposures=int(exposures.sum()),
                no_skipped_or_duplicated_identities=True,
                schedule_and_next_batch_verified=True,
            ),
        )
    else:
        if (checkpoints / "LATEST.json").exists():
            raise FileExistsError("Existing training run requires exact --resume")
        before = baseline()
        atomic_json(RUN / "E012_BASELINE.json", before)
    start, gpu_seconds, last_save, last_guard = time.monotonic(), 0.0, time.monotonic(), 0.0
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], epoch)

    def status(phase, loss=None):
        record = dict(
            pid=os.getpid(),
            phase=phase,
            epoch=epoch,
            schedule_cursor=position,
            update=updates,
            total_exposures=int(exposures.sum()),
            identities_seen_this_epoch=int((exposures == epoch).sum()),
            population_hash=contract["data_hash"],
            config_hash=contract["config_hash"],
            minimum_exposure=int(exposures.min()),
            maximum_exposure=int(exposures.max()),
            loss=loss,
            elapsed_seconds=elapsed_previous + time.monotonic() - start,
            gpu_seconds=gpu_previous + gpu_seconds,
            proteins_per_second=int(exposures.sum()) / max(1.0, elapsed_previous + time.monotonic() - start),
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            timestamp_unix=time.time(),
            probe_passed=probe_passed,
            git_sha=git_sha,
        )
        atomic_json(RUN / "HEARTBEAT.json", record)
        atomic_json(RUN / "STATE.json", record)

    def save(name, rolling=False):
        nonlocal last_save
        digest = hashlib.sha256()
        for key, tensor in sorted(model.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        payload = dict(
            contract=contract,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(),
            epoch=epoch,
            position=position,
            updates=updates,
            exposures=exposures.copy(),
            schedule_hash=canonical_hash(schedule),
            next_batch=schedule[position] if position < len(schedule) else None,
            curves=curves,
            epoch_loss_sum=epoch_sum,
            epoch_loss_proteins=epoch_count,
            clipping_count=clipping,
            elapsed_seconds=elapsed_previous + time.monotonic() - start,
            gpu_seconds=gpu_previous + gpu_seconds,
            pending_boundary=pending,
            probe_passed=probe_passed,
            contention_baseline=before,
            model_hash=digest.hexdigest(),
            population_hash=contract["data_hash"],
            config_hash=contract["config_hash"],
            total_exposures=int(exposures.sum()),
            identities_seen_this_epoch=int((exposures == epoch).sum()),
            git_sha=git_sha,
        )
        path = atomic_checkpoint(checkpoints, name, payload, rolling=rolling)
        latest = json.loads((checkpoints / "LATEST.json").read_text())
        record = dict(
            label=path.stem,
            path=str(path),
            sha256=latest["sha256"],
            epoch=epoch,
            schedule_cursor=position,
            updates=updates,
            exposures=int(exposures.sum()),
            model_hash=payload["model_hash"],
            model="E007",
            parameters=7586505,
            population_hash=contract["data_hash"],
            config_hash=contract["config_hash"],
            hash_valid_loadable=True,
        )
        inventory_path = RUN / "checkpoint_inventory.json"
        inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else dict(checkpoints=[])
        inventory["checkpoints"] = [r for r in inventory["checkpoints"] if r["path"] != str(path)] + [record]
        atomic_json(inventory_path, inventory)
        atomic_json(RUN / "COMMITTED.json", record)
        last_save = time.monotonic()
        return path

    def boundary(name):
        nonlocal pending
        optimizer.zero_grad(set_to_none=True)
        status("boundary_evaluation")
        summary = evaluate(model, diffusion, dev, cache, cfg, RUN / "evaluation", name + "_development")
        if not any(r["label"] == name for r in curves):
            curves.append(
                dict(
                    label=name,
                    epoch=epoch,
                    exposures=int(exposures.sum()),
                    updates=updates,
                    v_mse=primary_v(summary),
                    training_v_loss=epoch_sum / max(1, epoch_count),
                )
            )
        atomic_json(RUN / "CURVES.json", curves)
        pending = None
        save("recovery_0", rolling=True)

    if not args.resume:
        save("initialization")
        boundary("initialization")
    elif pending:
        boundary(pending)
    if position == len(schedule):
        if epoch >= 2:
            status("epoch2_training_complete_awaiting_comparison")
            return
        epoch += 1
        position, epoch_sum, epoch_count = 0, 0.0, 0
    logdir = RUN / "logs"
    logdir.mkdir(exist_ok=True)
    with open(logdir / "TRAINING.jsonl", "a", buffering=1) as log:
        while epoch <= 2:
            schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], epoch)
            schedule_path = RUN / f"EPOCH_{epoch}_SCHEDULE.json"
            if schedule_path.exists():
                assert canonical_hash(json.loads(schedule_path.read_text())) == canonical_hash(schedule)
            else:
                atomic_json(schedule_path, schedule)
            thresholds = [int(np.ceil(f * len(train))) for f in cfg["checkpoint_fractions"]]
            seen = int((exposures == epoch).sum())
            crossed = {i for i, v in enumerate(thresholds) if seen >= v}
            while position < len(schedule):
                if STOP:
                    save(f"pause_{time.time_ns()}_{updates}")
                    status("paused_signal")
                    return
                if shutil.disk_usage(RUN).free < 20 * 2**30:
                    save(f"disk_pause_{updates}")
                    status("paused_disk")
                    return
                if time.monotonic() - last_guard >= 30:
                    guard = sequence_guard(before)
                    atomic_json(RUN / "E012_CONTENTION_CURRENT.json", guard)
                    last_guard = time.monotonic()
                    if guard["pause_reason"]:
                        save(f"sequence_pause_{updates}")
                        status("paused_" + guard["pause_reason"])
                        return
                    if not probe_passed and time.monotonic() - start >= 600:
                        if not guard["enough_batches"]:
                            save(f"inconclusive_probe_{updates}")
                            status("paused_inconclusive_E012_probe")
                            return
                        probe_passed = True
                        atomic_json(
                            RUN / "E012_CONTENTION_PROBE.json", dict(passed=True, before=before, comparison=guard)
                        )
                        save("recovery_1", rolling=True)
                ids = schedule[position]
                assert all(exposures[i] == epoch - 1 for i in ids)
                rows = [train[i] for i in ids]
                x, lengths, mask = coordinate_batch(rows, cache, cfg["coordinate_scale_angstrom"])
                noisy, targets, ts = [], [], []
                from protein_distance_diffusion.training.e014_runtime import noise_seed

                for i, row in enumerate(rows):
                    n = row["length"]
                    generator = torch.Generator(device="cuda").manual_seed(
                        noise_seed(f"e014-train-epoch-{epoch}", row["sample_id"], 0)
                    )
                    b = diffusion.make_training_batch(x[i : i + 1, :n], mask[i : i + 1, :n], generator=generator)
                    noisy.append(torch.nn.functional.pad(b.noisy_coordinates, (0, 0, 0, x.shape[1] - n)))
                    targets.append(torch.nn.functional.pad(b.coordinate_v_target, (0, 0, 0, x.shape[1] - n)))
                    ts.append(b.timesteps)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                first.record()
                prediction = model(torch.cat(noisy), torch.cat(ts), lengths, mask)
                loss = coordinate_v_loss(prediction, torch.cat(targets), mask)
                assert bool(torch.isfinite(loss)) and float(loss) <= cfg["hard_loss_maximum"]
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                last.record()
                torch.cuda.synchronize()
                gpu_seconds += first.elapsed_time(last) / 1000
                updates += 1
                position += 1
                exposures[ids] += 1
                seen += len(ids)
                value = float(loss)
                clipping += float(norm) > 1
                epoch_sum += value * len(ids)
                epoch_count += len(ids)
                record = dict(
                    epoch=epoch,
                    update=updates,
                    position=position,
                    identity_count=len(ids),
                    sample_ids=[r["sample_id"] for r in rows],
                    loss=value,
                    gradient_norm=float(norm),
                    clipped=float(norm) > 1,
                    commit_status="provisional_until_checkpoint",
                )
                log.write(json.dumps(record) + "\n")
                if updates % 20 == 0:
                    status("training", value)
                if updates % 200 == 0:
                    print(json.dumps(record), flush=True)
                del prediction, loss, x, lengths, mask, noisy, targets, ts, b
                if updates == 20 or time.monotonic() - last_save >= cfg["extra_recovery_checkpoint_seconds"]:
                    save("recovery_0", rolling=True)
                for index, threshold in enumerate(thresholds):
                    if index not in crossed and seen >= threshold:
                        crossed.add(index)
                        pending = f"epoch_{epoch}_fraction_{cfg['checkpoint_fractions'][index]:.2f}"
                        save(pending)
                        boundary(pending)
            assert np.all(exposures == epoch)
            atomic_json(
                RUN / f"EXPOSURE_EPOCH_{epoch}.json",
                dict(
                    epoch=epoch,
                    identities=len(train),
                    total_exposures=int(exposures.sum()),
                    duplicates=0,
                    missing=0,
                    updates=updates,
                    exposure_hash=canonical_hash(exposures.tolist()),
                ),
            )
            status(
                "epoch1_complete_awaiting_diagnostics" if epoch == 1 else "epoch2_training_complete_awaiting_comparison"
            )
            return


if __name__ == "__main__":
    main()
