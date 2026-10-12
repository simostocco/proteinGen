"""Serialized E015 hold / exact frozen E007 recovery and observation."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/mnt/d/Simone/proteinGen/.e015_worktree")
E015 = Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")
E007WORK = Path("/mnt/d/Simone/proteinGen/.e014_worktree")
E007 = E007WORK / "e007_matched_fullscale_v1"
EXT = E015 / "external_designability_loop_v1"
PY = "/home/simostocco/miniforge3/envs/proteingen/bin/python"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
for k in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "HF_HOME",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
):
    os.environ[k] = str(E007 / "runtime" / k.lower())
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

os.nice(10)


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    assert p.is_relative_to("/mnt/d")
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".partial")
    with q.open("w") as f:
        json.dump(x, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    q.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(1024**2):
            h.update(b)
    return h.hexdigest()


def processes():
    result = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if any(
                k in cmd
                for k in (
                    "run_e007",
                    "supervise_e007",
                    "run_e015_integrated.py",
                    "supervise_e015_training.py",
                    "colabfold_batch",
                    "protein_mpnn_run.py",
                    "e015_external_designability.py screen",
                    "sequence_foundation/scripts",
                )
            ):
                result.append(dict(pid=int(p.name), command=cmd))
        except OSError:
            pass
    return result


def host(require_idle=False):
    mem = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in ("MemAvailable", "SwapTotal", "SwapFree")
    }
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    x = dict(
        timestamp_unix=time.time(),
        ram_available_bytes=mem["MemAvailable"],
        swap_used_bytes=mem["SwapTotal"] - mem["SwapFree"],
        load_average=list(os.getloadavg()),
        d_free_bytes=shutil.disk_usage(E007).free,
        gpu_csv=gpu,
        cuda_compute_apps=apps,
        targeted_processes=processes(),
    )
    if shutil.which("powershell.exe"):
        x["windows_ram_available_bytes"] = (
            int(
                subprocess.check_output(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-Command",
                        "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory",
                    ],
                    text=True,
                    timeout=20,
                ).strip()
            )
            * 1024
        )
        if require_idle:
            command = (
                'Get-CimInstance Win32_Process | Where-Object {$_.Name -match "python" '
                '-and $_.CommandLine -match "run_e007|supervise_e007|e015|colabfold|protein_mpnn"} '
                "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            )
            x["native_windows_targeted_processes"] = subprocess.check_output(
                ["powershell.exe", "-NoProfile", "-Command", command], text=True, timeout=20
            ).strip()
            assert not x["native_windows_targeted_processes"], "Native competing process"
    assert x["ram_available_bytes"] >= 3 * 1024**3 and x.get("windows_ram_available_bytes", 3 * 1024**3) >= 3 * 1024**3
    assert x["d_free_bytes"] >= 30 * 1024**3
    if require_idle:
        assert not apps, "GPU owner remains"
        assert not x["targeted_processes"], "Heavy process remains"
        assert int(gpu.split(",")[0]) < 2200, "GPU did not return to display/background baseline"
        assert not list((E015 / "production/checkpoints").glob("*.partial"))
        assert not list((E007 / "checkpoints").glob("*.partial"))
    return x


def lock_available(p):
    with open(p, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    return True


def e015_queue():
    before = host(True)
    cmd = [PY, "-B", str(WORK / "docs/e015_serial_recovery.py"), "audit-latest"]
    with (EXT / "logs/switchover_e015_strict_audit.log").open("w") as f:
        subprocess.run(cmd, cwd=WORK, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=600)
    rows = [
        json.loads(line)
        for line in (EXT / "logs/switchover_e015_strict_audit.log").read_text().splitlines()
        if line.startswith("{")
    ]
    report = Path(next(r["report"] for r in reversed(rows) if "report" in r))
    audit = read(report / "recovery_audit.json")
    cp = audit["selected"]
    s = read(E015 / "STATE.json")
    assert cp == read(E015 / "COMMITTED.json") == read(E015 / "production/checkpoints/LATEST.json")
    assert cp["position"]["exposures"] == s["exposures"] and cp["position"]["update"] == s["update"]
    continuation = PY + " -B " + str(WORK / "docs/e015_external_designability.py") + " screen --release-hold"
    record = dict(
        timestamp_unix=time.time(),
        reason="GPU reassigned to matched E007 control",
        checkpoint=cp,
        checkpoint_bytes=Path(cp["path"]).stat().st_size,
        epoch=cp["position"]["epoch"],
        update=cp["position"]["update"],
        exposures=cp["position"]["exposures"],
        cursor=cp["position"]["cursor"],
        completed_epochs=s["completed_epochs"],
        uncommitted_exposures=0,
        exact_next_operation=audit["next_updates"][0],
        exact_resume_command=audit["command"],
        evaluation_status="SUSPENDED",
        evaluation_continuation_command=continuation,
        automatic_resume=False,
        intentionally_paused=True,
        strict_audit=str(report),
        strict_raw_ema_optimizer_rng_accounting_verified=True,
        resources=before,
    )
    write(E015 / "QUEUED_AFTER_E007.json", record)
    status = read(E015 / "EVALUATION_SWITCHOVER_STATUS.json")
    status.update(
        continuation_command=continuation,
        elapsed_seconds_at_initial_process_inspection=702,
        active_pids_at_initial_inspection=[1189240, 1204775],
        e015_auto_resume=False,
    )
    write(E015 / "EVALUATION_SWITCHOVER_STATUS.json", status)
    manifest = read(EXT / "MANIFEST.json")
    manifest["continuation"] = continuation
    write(EXT / "MANIFEST.json", manifest)
    host(True)
    print(
        json.dumps(
            dict(
                e015_checkpoint=cp,
                zero_uncommitted=True,
                next_update=audit["next_updates"][0]["update"],
                next_exposures=audit["next_updates"][0]["exposures"],
            )
        )
    )


def e007_audit(path=None, initial=True):
    # CPU only: no CUDA initialization/model, including while checking a later
    # rolling checkpoint. CUDA RNG blobs remain CPU uint8 tensors.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(E007WORK / "scripts"), str(E007WORK / "src")]
    os.chdir(E007WORK)
    import random

    import e007_fullscale_runtime as rt
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl
    from protein_distance_diffusion.training.e014_runtime import canonical_hash, noise_seed

    before = host(initial)
    cfg, pop, train, dev = rt.verified_inputs()
    if path is None:
        path = Path(read(E007 / "checkpoints/LATEST.json")["path"])
    else:
        path = Path(path)
    digest = sha(path)
    if initial:
        assert (
            path.name == "sequence_pause_3580.pt"
            and digest == "47d58d914400c08f2d90af8d19a7f2751f7575efd4e63a8f137818fb9357693c"
        )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    contract = saved["contract"]
    assert contract["config_hash"] == sha(E007WORK / rt.CONFIG)
    assert contract["model_config_hash"] == sha(E007WORK / rt.MODEL_CONFIG)
    assert contract["data_hash"] == pop["train_sample_id_hash"]
    assert contract["normalization_hash"] == cfg["normalization_sha256"]
    assert (
        contract["maximum_epochs"] == 2
        and contract["accumulation"] == 1
        and contract["physical_batch_sizes"] == cfg["batch_sizes"]
    )
    for f, h in contract["source_hashes"].items():
        assert sha(E007WORK / f) == h, f
    model = E007ScaleupControl(**rt.yaml.safe_load(rt.MODEL_CONFIG.read_text())["model"])
    assert sum(p.numel() for p in model.parameters()) == 7586505
    model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
    )
    optimizer.load_state_dict(saved["optimizer"])
    assert all(int(s["step"]) == saved["updates"] for s in optimizer.state.values())
    assert all(
        g["lr"] == cfg["optimizer"]["learning_rate"]
        and g["weight_decay"] == cfg["optimizer"]["weight_decay"]
        and g["betas"] == (0.9, 0.999)
        and g["eps"] == 1e-8
        for g in optimizer.param_groups
    )
    torch.set_rng_state(saved["rng"]["torch"])
    np.random.set_state(saved["rng"]["numpy"])
    random.setstate(saved["rng"]["python"])
    assert all(t.dtype == torch.uint8 and t.numel() > 0 and t.device.type == "cpu" for t in saved["rng"]["cuda"])
    schedule, expected = rt.expected_exposures(
        [r["length"] for r in train], saved["epoch"], saved["position"], cfg["batch_sizes"], cfg["seed"]
    )
    assert np.array_equal(expected, saved["exposures"])
    assert canonical_hash(schedule) == saved["schedule_hash"]
    assert schedule[saved["position"]] == saved["next_batch"]
    ids = saved["next_batch"]
    batch = [
        dict(
            identity=train[i]["sample_id"],
            length=train[i]["length"],
            noise_seed=noise_seed(f"e014-train-epoch-{saved['epoch']}", train[i]["sample_id"], 0),
            noise_seed_exact_decimal=str(noise_seed(f"e014-train-epoch-{saved['epoch']}", train[i]["sample_id"], 0)),
        )
        for i in ids
    ]
    exposures = int(expected.sum())
    nextop = dict(
        update=saved["updates"] + 1, exposures=exposures + len(ids), cursor=saved["position"] + 1, batch=batch
    )
    if initial:
        assert (saved["epoch"], saved["updates"], saved["position"], exposures) == (1, 3580, 3580, 4549)
        assert batch == [
            dict(
                identity="9g30_o",
                length=230,
                noise_seed=8522206775665383470,
                noise_seed_exact_decimal="8522206775665383470",
            )
        ]
        assert read(E007 / "STATE.json")["total_exposures"] == exposures
        inventory = read(E007 / "checkpoint_inventory.json")["checkpoints"]
        assert max(r["updates"] for r in inventory) == 3580
        for p in (E007 / "checkpoints").glob("*.pt"):
            if p == path:
                continue
            other = torch.load(p, map_location="cpu", weights_only=False)
            assert other["updates"] <= 3580
            del other
    handoff = read(rt.CORRECTED / "reports/chatgpt_handoff.json")
    state12 = read(rt.CORRECTED / "production/STATE.json")
    assert (
        state12["phase"] == "frozen_corpus_ready"
        and not state12["training_launched"]
        and handoff["final_corpus_n"] == 20000000
    )
    stage = read(rt.CORRECTED / "production/reports/stage_handoff.json")
    assert not stage["training_launched"] and stage["wall_seconds"] > 0
    assert (
        not handoff["training_launched"]
        and Path(handoff["final_manifest"]).exists()
        and Path(handoff["final_fasta"]).exists()
    )
    guard = rt.sequence_guard(saved["contention_baseline"])
    assert guard["pause_reason"] is None, guard
    assert sha(path) == digest
    result = dict(
        timestamp_unix=time.time(),
        checkpoint=str(path),
        sha256=digest,
        bytes=path.stat().st_size,
        epoch=saved["epoch"],
        update=saved["updates"],
        cursor=saved["position"],
        exposures=exposures,
        completed_epochs=saved["epoch"] - 1,
        zero_uncommitted=initial,
        model_strict_cpu_load=True,
        optimizer_strict_cpu_load=True,
        optimizer_steps=saved["updates"],
        cpu_rng_restored=True,
        cuda_rng_blobs_valid_preserved_on_cpu=True,
        exposure_vector_exact=True,
        no_skipped_or_duplicate_durable_exposure=True,
        population_hash=contract["data_hash"],
        config_hash=contract["config_hash"],
        frozen_source_hashes_unchanged=True,
        maximum_epochs=2,
        target_exposures=460264,
        exact_next_operation=nextop,
        sequence_guard=guard,
        E012_handoff_complete=True,
        E012_frozen_sequences=20000000,
        resource=before,
        exact_launch_command=[PY, "-B", str(E007WORK / "scripts/supervise_e007_fullscale.py"), "--launch", "--resume"],
    )
    write(
        E007 / ("E015_TO_E007_SWITCHOVER.json" if initial else "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json"), result
    )
    print(json.dumps(result))


def launch():
    before = host(True)
    audit = read(E007 / "E015_TO_E007_SWITCHOVER.json")
    queued = read(E015 / "QUEUED_AFTER_E007.json")
    assert queued["intentionally_paused"] and not queued["automatic_resume"] and (EXT / "HOLD_E015.json").exists()
    assert sha(audit["checkpoint"]) == audit["sha256"]
    assert read(E007 / "checkpoints/LATEST.json")["path"] == audit["checkpoint"]
    locks = {
        str(p): lock_available(p)
        for p in [
            E015 / "SUPERVISOR.lock",
            E015 / "GPU_WORKER.lock",
            E007 / "OWNER.lock",
            E007 / "SUPERVISOR_OWNER.lock",
        ]
    }
    command = audit["exact_launch_command"]
    assert command == [PY, "-B", str(E007WORK / "scripts/supervise_e007_fullscale.py"), "--launch", "--resume"]
    source = (E007WORK / "scripts/supervise_e007_fullscale.py").read_text()
    assert "workflow(args.resume)" in source and 'command.append("--resume")' in source
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONPATH"] = str(E007WORK / "src")
    started = time.time()
    out = subprocess.check_output(command, cwd=E007WORK, env=env, text=True, timeout=120)
    announced = json.loads(out.strip().splitlines()[-1])
    sup = announced["supervisor_pid"]
    audit.update(
        launch_timestamp_unix=started,
        launch_command=command,
        supervisor_pid=sup,
        resources_before_launch=before,
        locks_available_before_launch=locks,
        E015_intentionally_paused=True,
        E015_automatic_resume=False,
        frozen_scientific_protocol_unchanged=True,
        E012_guard_changed=False,
    )
    write(E007 / "E015_TO_E007_SWITCHOVER.json", audit)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        owner = read(E007 / "SUPERVISOR.json")
        s = read(E007 / "STATE.json")
        if owner.get("child_pid") and s.get("pid") == owner["child_pid"] and s.get("phase") == "training":
            audit.update(trainer_pid=owner["child_pid"], initial_live_state=s)
            write(E007 / "E015_TO_E007_SWITCHOVER.json", audit)
            print(
                json.dumps(
                    dict(
                        supervisor_pid=sup,
                        trainer_pid=owner["child_pid"],
                        phase=s["phase"],
                        update=s["update"],
                        exposures=s["total_exposures"],
                    )
                )
            )
            return
        if s.get("timestamp_unix", 0) > started and s.get("phase", "").startswith("paused_"):
            raise RuntimeError("E007 guard pause after launch: " + json.dumps(s))
        time.sleep(2)
    raise RuntimeError("E007 launch not yet verified; inspect preserved supervisor log")


def observe(seconds=930):
    audit = read(E007 / "E015_TO_E007_SWITCHOVER.json")
    started = audit["launch_timestamp_unix"]
    samples = []
    first = None
    roll = None
    deadline = time.monotonic() + seconds
    path = E007 / "POST_RESUME_EXCLUSIVE_OBSERVATION.json"
    while time.monotonic() < deadline:
        x = host()
        s = read(E007 / "STATE.json")
        owner = read(E007 / "SUPERVISOR.json")
        x.update(
            update=s["update"],
            exposures=s["total_exposures"],
            state_age_seconds=time.time() - s["timestamp_unix"],
            loss=s["loss"],
            phase=s["phase"],
            trainer_pid=s.get("pid"),
            supervisor_pid=owner.get("pid"),
            gpu_allocated_bytes=s.get("gpu_allocated_bytes"),
            gpu_reserved_bytes=s.get("gpu_reserved_bytes"),
        )
        assert not [
            r for r in x["targeted_processes"] if r["pid"] not in (audit["supervisor_pid"], audit["trainer_pid"])
        ], "Competing heavy process"
        cuda_pids = [int(row.split(",")[0]) for row in x["cuda_compute_apps"].splitlines() if row.strip()]
        assert cuda_pids == [audit["trainer_pid"]], "Ambiguous or competing CUDA ownership"
        assert s["phase"] == "training" and x["state_age_seconds"] < 120, "Stale/stopped E007"
        assert s["loss"] is None or math.isfinite(s["loss"])
        guard = read(E007 / "E012_CONTENTION_CURRENT.json")
        assert guard["pause_reason"] is None
        if first is None:
            log = E007 / "logs/TRAINING.jsonl"
            with log.open("rb") as f:
                f.seek(max(0, f.seek(0, 2) - 5 * 1024**2))
                lines = f.read().splitlines()
            rows = []
            for line in lines:
                try:
                    r = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if r["update"] > 3580:
                    rows.append(r)
            if rows:
                first = min(rows, key=lambda r: r["update"])
                assert (
                    first["update"] == 3581
                    and first["position"] == 3581
                    and first["sample_ids"] == ["9g30_o"]
                    and first["identity_count"] == 1
                    and math.isfinite(first["loss"])
                )
                first = dict(
                    log_record=first,
                    exposures=4550,
                    length=230,
                    noise_seed=8522206775665383470,
                    noise_seed_exact_decimal="8522206775665383470",
                    seed_verification=(
                        "First identity directly logged; length and exact noise seed independently "
                        "reconstructed from frozen TRAIN/source function; "
                        "no seed is logged by the frozen trainer"
                    ),
                    match=True,
                )
                audit["first_resumed_operation"] = first
                write(E007 / "E015_TO_E007_SWITCHOVER.json", audit)
        committed = read(E007 / "COMMITTED.json")
        if committed["updates"] > 3580 and roll is None:
            roll = committed
        samples.append(x)
        result = dict(
            status="OBSERVING",
            started_unix=started,
            observation_seconds=time.time() - started,
            samples=samples,
            first_resumed_operation=first,
            first_new_durable_checkpoint=roll,
            E015_remains_paused=True,
            exclusive_heavy_gpu_owner=True,
            scientific_protocol_unchanged=True,
        )
        write(path, result)
        print(
            json.dumps(
                {
                    k: x[k]
                    for k in (
                        "timestamp_unix",
                        "update",
                        "exposures",
                        "loss",
                        "state_age_seconds",
                        "gpu_csv",
                        "ram_available_bytes",
                        "swap_used_bytes",
                        "d_free_bytes",
                    )
                }
            ),
            flush=True,
        )
        time.sleep(30)
    assert first is not None and roll is not None, "First operation or new rolling checkpoint missing"
    fresh = [s for s in samples if s["state_age_seconds"] < 120]
    dt = fresh[-1]["timestamp_unix"] - fresh[0]["timestamp_unix"]
    pps = (fresh[-1]["exposures"] - fresh[0]["exposures"]) / dt
    ups = (fresh[-1]["update"] - fresh[0]["update"]) / dt
    remaining = 460264 - fresh[-1]["exposures"]
    eta = remaining / pps
    # Reconcile logged increments with live exposure counts without model work.
    with (E007 / "logs/TRAINING.jsonl").open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - 5 * 1024**2))
        lines = f.read().splitlines()
    rows = []
    for line in lines:
        try:
            r = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if 3580 < r["update"] <= fresh[-1]["update"]:
            rows.append(r)
    assert [r["update"] for r in rows] == list(range(3581, fresh[-1]["update"] + 1))
    assert 4549 + sum(r["identity_count"] for r in rows) == fresh[-1]["exposures"]
    result.update(
        status="COMPLETE",
        observation_seconds=time.time() - started,
        proteins_per_second=pps,
        updates_per_second=ups,
        current_exposures=fresh[-1]["exposures"],
        remaining_exposures=remaining,
        active_training_eta_seconds=eta,
        practical_wall_time_range_seconds=[eta, eta * 1.25],
        practical_range_basis=(
            "Planning range with0–25% allowance for declared serial evaluation/checkpoint phases; "
            "overhead is not yet separately measured in this post-resume window"
        ),
        no_skipped_or_duplicate_logged_update=True,
        logged_exposure_increments_match_live_state=True,
    )
    write(path, result)
    audit.update(
        observation_file=str(path),
        recent_proteins_per_second=pps,
        recent_updates_per_second=ups,
        first_new_rolling=roll,
    )
    write(E007 / "E015_TO_E007_SWITCHOVER.json", audit)
    print(
        json.dumps(
            dict(
                observed_seconds=result["observation_seconds"],
                proteins_per_second=pps,
                updates_per_second=ups,
                current_exposures=result["current_exposures"],
                eta_hours=eta / 3600,
                first_rolling=roll,
            )
        )
    )


def stop_on_failed_verification(error):
    import ctypes
    import signal

    evidence = dict(
        timestamp_unix=time.time(),
        error=repr(error),
        action="Request existing E007 SIGINT graceful-stop handler; preserve checkpoint and logs",
    )
    write(E007 / "SWITCHOVER_VERIFICATION_FAILURE.json", evidence)
    pid = read(E007 / "E015_TO_E007_SWITCHOVER.json").get("trainer_pid")
    if pid and Path(f"/proc/{pid}/cmdline").exists():
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        assert b"run_e007_fullscale.py" in cmd
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, pid, 0)
        assert fd >= 0
        try:
            assert Path(f"/proc/{pid}/cmdline").read_bytes() == cmd
            assert libc.syscall(424, fd, int(signal.SIGINT), 0, 0) == 0
        finally:
            os.close(fd)


def finalize():
    import statistics

    audit = read(E007 / "E015_TO_E007_SWITCHOVER.json")
    obs = read(E007 / "POST_RESUME_EXCLUSIVE_OBSERVATION.json")
    rolling = read(E007 / "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json")
    queued = read(E015 / "QUEUED_AFTER_E007.json")
    status = read(E015 / "EVALUATION_SWITCHOVER_STATUS.json")
    assert (
        obs["status"] == "COMPLETE"
        and obs["samples"][-1]["timestamp_unix"] - obs["samples"][0]["timestamp_unix"] >= 900
    )
    assert rolling["sha256"] == obs["first_new_durable_checkpoint"]["sha256"]
    resources = host()
    state = read(E007 / "STATE.json")
    owner = read(E007 / "SUPERVISOR.json")
    assert (
        owner["pid"] == audit["supervisor_pid"]
        and state["pid"] == audit["trainer_pid"]
        and state["phase"] == "training"
    )
    assert [int(row.split(",")[0]) for row in resources["cuda_compute_apps"].splitlines() if row.strip()] == [
        state["pid"]
    ]
    assert not [p for p in resources["targeted_processes"] if p["pid"] not in (owner["pid"], state["pid"])]
    assert (
        read(E015 / "COMMITTED.json") == queued["checkpoint"]
        and read(E015 / "STATE.json")["update"] == queued["update"]
    )
    assert (EXT / "HOLD_E015.json").exists() and not queued["automatic_resume"]
    assert lock_available(E015 / "GPU_WORKER.lock") and lock_available(E015 / "SUPERVISOR.lock")
    held = {}
    for p in (E007 / "OWNER.lock", E007 / "SUPERVISOR_OWNER.lock"):
        try:
            lock_available(p)
            held[str(p)] = False
        except BlockingIOError:
            held[str(p)] = True
    assert all(held.values())
    jobs = read(EXT / "colabfold/jobs.json")
    completed = [j for j in jobs.values() if j["status"] == "COMPLETE"]
    assert len(completed) == 25
    for job in completed:
        for p, h in job["files"].items():
            assert Path(p).exists() and sha(p) == h, p
    rolling.pop("zero_uncommitted", None)
    rolling.update(
        checkpoint_internal_uncommitted_exposures=0,
        live_uncommitted_exposures_at_final_inspection=state["total_exposures"] - rolling["exposures"],
        note=(
            "Verified durable checkpoint is internally exact; active trainer has advanced past it. "
            "Those live exposures would be recomputed after an unexpected shutdown."
        ),
    )
    write(E007 / "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json", rolling)
    samples = obs["samples"]
    gpu_rows = [[int(a.strip()) for a in r["gpu_csv"].split(",")] for r in samples]
    performance = dict(
        proteins_per_second=obs["proteins_per_second"],
        updates_per_second=obs["updates_per_second"],
        current_update=state["update"],
        current_exposures=state["total_exposures"],
        remaining_exposures=460264 - state["total_exposures"],
        active_training_eta_seconds=(460264 - state["total_exposures"]) / obs["proteins_per_second"],
        practical_range_basis=obs["practical_range_basis"],
        gpu_utilization_median=statistics.median(r[2] for r in gpu_rows),
        gpu_utilization_min=min(r[2] for r in gpu_rows),
        gpu_utilization_max=max(r[2] for r in gpu_rows),
        gpu_used_mib_min=min(r[0] for r in gpu_rows),
        gpu_used_mib_max=max(r[0] for r in gpu_rows),
        gpu_temperature_max_c=max(r[3] for r in gpu_rows),
        minimum_ram_available_bytes=min(r["ram_available_bytes"] for r in samples),
        maximum_swap_used_bytes=max(r["swap_used_bytes"] for r in samples),
        maximum_state_age_seconds=max(r["state_age_seconds"] for r in samples),
        minimum_d_free_bytes=min(r["d_free_bytes"] for r in samples),
    )
    eta = performance["active_training_eta_seconds"]
    performance["practical_wall_time_range_seconds"] = [eta, eta * 1.25]
    audit.update(
        status="VERIFIED_EXCLUSIVE_TRAINING",
        final_timestamp_unix=time.time(),
        final_live_state=state,
        final_resources=resources,
        performance=performance,
        first_new_rolling_strict_verification=rolling,
        owner_locks_held=held,
        E015_no_automatic_resume=True,
        no_sequence_training_launched=True,
        E012_E014_scientific_evidence_unchanged=True,
        all_generated_artifacts_on_D=True,
        guard_source_changed=False,
    )
    write(E007 / "E015_TO_E007_SWITCHOVER.json", audit)
    obs.update(
        final_live_state=state,
        final_resources=resources,
        performance=performance,
        rolling_strict_verification_file=str(E007 / "FIRST_NEW_ROLLING_EXCLUSIVE_VERIFICATION.json"),
    )
    write(E007 / "POST_RESUME_EXCLUSIVE_OBSERVATION.json", obs)
    status.update(
        completed_artifacts_hash_verified=True,
        E007_supervisor_pid=owner["pid"],
        E007_trainer_pid=state["pid"],
        E015_no_automatic_resume=True,
    )
    write(E015 / "EVALUATION_SWITCHOVER_STATUS.json", status)
    print(
        json.dumps(
            dict(status=audit["status"], performance=performance, resources=resources, first_new_rolling=rolling),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["e015-queue", "e007-audit", "launch", "observe", "finalize"])
    parser.add_argument("--checkpoint")
    a = parser.parse_args()
    if a.stage == "e015-queue":
        e015_queue()
    elif a.stage == "e007-audit":
        e007_audit(a.checkpoint, initial=a.checkpoint is None)
    elif a.stage == "launch":
        launch()
    elif a.stage == "finalize":
        finalize()
    else:
        try:
            observe()
        except Exception as exc:
            stop_on_failed_verification(exc)
            raise
