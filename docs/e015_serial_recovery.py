# ruff: noqa: E402, E501
"""Bounded recovery audit; no production-science source or configuration edits."""

import os
from pathlib import Path

WORK = Path("/mnt/d/Simone/proteinGen/.e015_worktree")
ROOT = Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")
for key in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "HF_HOME",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
):
    os.environ[key] = str(ROOT / "runtime" / key.lower())
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import datetime
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time

tempfile.tempdir = os.environ["TMPDIR"]
sys.path.insert(0, str(WORK / "src"))
os.nice(10)
REPORT = ROOT / "reports" / ("shutdown_recovery_" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ"))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    assert path.is_relative_to(Path("/mnt/d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    partial.replace(path)


def processes():
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            parts = (p / "cmdline").read_bytes().split(b"\0")
            if parts and Path(parts[0].decode(errors="replace")).name.startswith("python"):
                rows.append(dict(pid=int(p.name), command=[x.decode(errors="replace") for x in parts if x]))
        except OSError:
            pass
    assert len(list(Path("/proc").glob("[0-9]*"))) > 20, "Host PID namespace required"
    return rows


def resources():
    mem = {
        a.split(":")[0]: int(a.split()[1]) * 1024
        for a in Path("/proc/meminfo").read_text().splitlines()
        if a.split(":")[0] in ("MemAvailable", "SwapTotal", "SwapFree")
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
    temps = {}
    for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps[str(p)] = int(p.read_text()) / 1000
        except (OSError, ValueError):
            pass
    return dict(
        timestamp_unix=time.time(),
        ram_available_bytes=mem["MemAvailable"],
        swap_used_bytes=mem["SwapTotal"] - mem["SwapFree"],
        d_free_bytes=shutil.disk_usage(ROOT).free,
        gpu_csv=gpu,
        gpu_compute_apps=apps,
        load_average=list(os.getloadavg()),
        cpu_temperatures=temps,
        python_processes=processes(),
    )


def gate():
    r = resources()
    # WSL availability and native host availability are separate constraints.
    if shutil.which("powershell.exe"):
        result = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            text=True,
            timeout=20,
        )
        r["windows_ram_available_bytes"] = int(result.strip()) * 1024
        assert r["windows_ram_available_bytes"] >= 2 * 1024**3, "Native Windows RAM below2GiB"
    assert r["ram_available_bytes"] >= 2 * 1024**3
    assert r["d_free_bytes"] >= 30 * 1024**3
    assert not r["gpu_compute_apps"], r["gpu_compute_apps"]
    assert not [
        p
        for p in r["python_processes"]
        if p["pid"] != os.getpid()
        and any(
            x in " ".join(p["command"])
            for x in (
                "supervise_e015_training.py",
                "run_e015_integrated.py",
                "e015_absolute_evaluation.py",
                "run_e007",
                "e015_serial_evaluation.py",
            )
        )
    ]
    return r


def audit(only_latest=False):
    before = gate()
    REPORT.mkdir(parents=True)
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import yaml

    from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise
    from protein_distance_diffusion.training.e014_runtime import canonical_hash, sha
    from protein_distance_diffusion.training.e015_engine import Engine
    from protein_distance_diffusion.training.e015_runtime import Runtime
    from protein_distance_diffusion.training.e015_snr import SNRBinSampler
    from protein_distance_diffusion.training.e015_trainer import identity_schedule, plan_update

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cp = WORK / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"
    cfg = yaml.safe_load(cp.read_text())
    cal = read(ROOT / "calibration/frozen.json")
    batching = read(ROOT / "batch_autotuning.json")
    rt = object.__new__(Runtime)
    rt.config_path = cp
    rt.cfg = cfg
    contract = rt.contract(cal)
    contract["batching_hash"] = canonical_hash(batching)
    engine = Engine(cfg, cal, device="cpu", precision="fp32")
    assert sum(p.numel() for p in engine.model.parameters() if p.requires_grad) == 33128139
    train = pq.read_table(Path(cfg["population_root"]) / "population/train.parquet", use_threads=False).to_pylist()
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == cfg["population_hash"]
    snapshots = {}
    for name in (
        "STATE.json",
        "HEARTBEAT.json",
        "COMMITTED.json",
        "SUPERVISOR.json",
        "production/STATE.json",
        "production/COMMITTED.json",
        "production/checkpoints/LATEST.json",
    ):
        snapshots[name] = read(ROOT / name)
    latest = snapshots["production/checkpoints/LATEST.json"]
    assert latest == snapshots["COMMITTED.json"] == snapshots["production/COMMITTED.json"]
    inventory = []
    selected = None
    paths = [Path(latest["path"])] if only_latest else sorted((ROOT / "production/checkpoints").glob("*.pt"))
    for path in paths:
        digest = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["contract"] == contract
        assert saved["population_hash"] == cfg["population_hash"]
        assert saved["architecture_hash"] == canonical_hash(cfg["model"])
        engine.model.load_state_dict(saved["model"], strict=True)
        engine.ema.load_state_dict(saved["ema"], strict=True)
        engine.optimizer.load_state_dict(saved["optimizer"])
        engine.scaler.load_state_dict(saved["scaler"])
        ids = [i for g in saved["optimizer"]["param_groups"] for i in g["params"]]
        params = list(engine.model.parameters())
        assert len(ids) == len(params)
        assert not saved["position"]["update"] or set(saved["optimizer"]["state"]) == set(ids), (
            "Missing optimizer parameter state"
        )
        steps = []
        for i, p in zip(ids, params, strict=True):
            state = saved["optimizer"]["state"].get(i, {})
            for k, v in state.items():
                assert torch.isfinite(v).all()
                if k in ("exp_avg", "exp_avg_sq"):
                    assert v.shape == p.shape
                if k == "step":
                    steps.append(int(v))
        pos = saved["position"]
        assert not steps or set(steps) == {pos["update"]}
        for state in (saved["model"], saved["ema"]):
            assert all(torch.isfinite(v).all() for v in state.values())
        flat, ends = identity_schedule(train, batching["batch_sizes"], cfg["seed"], pos["epoch"])
        assert flat == saved["epoch_schedule"] and ends == saved["microbatch_ends"]
        expected = np.full(len(train), pos["epoch"] - 1, dtype=np.uint8)
        expected[flat[: pos["cursor"]]] += 1
        assert np.array_equal(expected, saved["exposures"]) and int(expected.sum()) == pos["exposures"]
        # Reconstruct update-count contract, including truncated fraction boundaries.
        cursor = 0
        update = 0
        for previous_epoch in range(1, pos["epoch"]):
            previous_flat, previous_ends = identity_schedule(
                train, batching["batch_sizes"], cfg["seed"], previous_epoch
            )
            previous_cursor = 0
            while previous_cursor < len(train):
                previous_stop = next(
                    (
                        math.ceil(len(train) * f)
                        for f in cfg["checkpoint_fractions"]
                        if math.ceil(len(train) * f) > previous_cursor
                    ),
                    len(train),
                )
                _, previous_cursor, _ = plan_update(
                    train, previous_flat, previous_ends, previous_cursor, previous_stop, cfg["residue_budget"]
                )
                update += 1
        while cursor < pos["cursor"]:
            stop = next(
                (math.ceil(len(train) * f) for f in cfg["checkpoint_fractions"] if math.ceil(len(train) * f) > cursor),
                len(train),
            )
            _, cursor, _ = plan_update(train, flat, ends, cursor, stop, cfg["residue_budget"])
            update += 1
        assert cursor == pos["cursor"] and update == pos["update"], (cursor, update, pos)
        for key in ("torch",):
            g = torch.Generator()
            g.set_state(saved["rng"][key])
            assert torch.equal(g.get_state(), saved["rng"][key])
        for key in ("sc", "rollout"):
            g = torch.Generator()
            g.set_state(saved["engine_rng"][key])
            assert torch.equal(g.get_state(), saved["engine_rng"][key])
        sampler = SNRBinSampler(cfg["seed"])
        sampler.load_state_dict(saved["engine_rng"]["sampler"])
        import random

        random.Random().setstate(saved["rng"]["python"])
        np.random.RandomState().set_state(saved["rng"]["numpy"])
        entry = dict(
            path=str(path),
            mtime_ns=path.stat().st_mtime_ns,
            sha256=digest,
            position=pos,
            strict_model_load=True,
            strict_ema_load=True,
            optimizer_load_shapes_finite=True,
            optimizer_steps=sorted(set(steps)),
            optimizer_state_count=len(saved["optimizer"]["state"]),
            optimizer_parameter_count=len(ids),
            scaler_state=saved["scaler"],
            rng_keys=list(saved["rng"]),
            engine_rng_keys=list(saved["engine_rng"]),
            exposure_vector_accounting=True,
            schedule_reconstructed=True,
            update_cursor_accounting=True,
            train_population_hash=saved["population_hash"],
            config_hash=saved["contract"]["config_hash"],
            precision=saved["precision"],
            fp32_geometry_islands=saved["fp32_geometry_islands"],
        )
        entry["rng_tensor_sha256"] = dict(
            torch=sha_tensor(saved["rng"]["torch"]),
            cuda=[sha_tensor(v) for v in saved["rng"]["cuda"]],
            sampler=sha_tensor(saved["engine_rng"]["sampler"]["rng"]),
            sc=sha_tensor(saved["engine_rng"]["sc"]),
            rollout=sha_tensor(saved["engine_rng"]["rollout"]),
            noise=sha_tensor(saved["engine_rng"]["noise"]),
        )
        inventory.append(entry)
        if str(path) == latest["path"]:
            assert digest == latest["sha256"] and pos == latest["position"]
            selected = saved
        if path.name == "epoch_1_fraction_0.50.pt":
            assert (
                digest == "2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03"
                and pos["exposures"] == 115066
            )
        engine.optimizer.state.clear()
        print(json.dumps(dict(verified=path.name, position=pos, sha256=digest)), flush=True)
        del saved
    assert selected is not None
    s = selected
    pos = s["position"]
    flat = s["epoch_schedule"]
    ends = s["microbatch_ends"]
    cursor = pos["cursor"]
    sampler = SNRBinSampler(cfg["seed"])
    sampler.load_state_dict(s["engine_rng"]["sampler"])
    sc = torch.Generator()
    sc.set_state(s["engine_rng"]["sc"])
    rollout = torch.Generator()
    rollout.set_state(s["engine_rng"]["rollout"])
    noise = torch.Generator(device="cuda")
    noise.set_state(s["engine_rng"]["noise"])
    assert torch.equal(noise.get_state(), s["engine_rng"]["noise"])
    next_updates = []
    for offset in range(1, 6):
        stop = next(
            (math.ceil(len(train) * f) for f in cfg["checkpoint_fractions"] if math.ceil(len(train) * f) > cursor),
            len(train),
        )
        groups, new_cursor, residues = plan_update(train, flat, ends, cursor, stop, cfg["residue_budget"])
        micro = []
        for group in groups:
            times, bins = sampler.sample(len(group), "cpu", cfg["features"].get("snr_balancing", True), shared_bin=True)
            state_before = sha_tensor(noise.get_state())
            lengths = [train[i]["length"] for i in group]
            mask = torch.arange(max(lengths), device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
            eps = centered_coordinate_noise(
                torch.empty(len(group), max(lengths), 3, device="cuda"), mask, generator=noise
            )
            do_rollout = (
                bool(torch.rand((), generator=rollout) < cfg["rollout_probability"])
                and cfg["features"]["rollout_states"]
                and pos["exposures"] + cursor - pos["cursor"] >= cfg["rollout_warmup_exposures"]
            )
            count = int(torch.randint(1, 4, (), generator=rollout)) if do_rollout else 0
            if do_rollout:
                times = times.clamp_min(count + 1)
                eps = centered_coordinate_noise(
                    torch.empty(len(group), max(lengths), 3, device="cuda"), mask, generator=noise
                )
                times -= count
            do_sc = (
                bool(torch.rand((), generator=sc) < cfg["self_conditioning_probability"])
                and cfg["model"]["self_conditioning"]
            )
            micro.append(
                dict(
                    indices=group,
                    identities=[train[i]["sample_id"] for i in group],
                    lengths=lengths,
                    timesteps=times.tolist(),
                    self_conditioned=do_sc,
                    rollout=do_rollout,
                    rollout_steps=count,
                    noise_generator_initial_seed=cfg["seed"] + 3,
                    noise_rng_sha256_before=state_before,
                    noise_tensor_sha256=sha_tensor(eps.cpu()),
                    noise_rng_sha256_after=sha_tensor(noise.get_state()),
                )
            )
        next_updates.append(
            dict(
                update=pos["update"] + offset,
                cursor=new_cursor,
                exposures=pos["exposures"] + new_cursor - pos["cursor"],
                proteins=new_cursor - cursor,
                residues=residues,
                microbatches=micro,
            )
        )
        cursor = new_cursor
    old = snapshots["STATE.json"]
    result = dict(
        status="VERIFIED",
        selected=latest,
        inventory=inventory,
        metadata_before=snapshots,
        resources_before=before,
        next_updates=next_updates,
        uncommitted_exposures_recomputed=max(0, old["exposures"] - pos["exposures"]),
        no_skipped_or_duplicate_durable_exposure=True,
        noise_seed_semantics="Production uses a checkpointed CUDA Philox stream seeded 15018, not independently reseeded per identity; exact state and generated-noise hashes recorded.",
        command=[
            "/home/simostocco/miniforge3/envs/proteingen/bin/python",
            "-B",
            str(WORK / "scripts/supervise_e015_training.py"),
            "--resume",
        ],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORK, text=True).strip(),
        other_experiments_modified=False,
        all_artifacts_on_d=True,
    )
    write(REPORT / "recovery_audit.json", result)
    write(REPORT / "checkpoint_inventory.json", inventory)
    write(REPORT.parent / (REPORT.name + ".json"), result)
    print(json.dumps(dict(report=str(REPORT), selected=latest, next_update=next_updates[0])), flush=True)
    return REPORT


def sha_tensor(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def launch(report):
    report = Path(report)
    audit = read(report / "recovery_audit.json")
    before = gate()
    assert read(ROOT / "production/checkpoints/LATEST.json") == audit["selected"]
    selected = audit["selected"]
    target = (
        ROOT
        / "evaluations/absolute_generation_u3811_2986d44e_v1/snapshots"
        / f"secondary_u{selected['position']['update']}_{selected['sha256'][:8]}.pt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(".pt.partial")
        digest = hashlib.sha256()
        with open(selected["path"], "rb") as source, part.open("wb") as dest:
            while block := source.read(8 * 1024**2):
                digest.update(block)
                dest.write(block)
            dest.flush()
            os.fsync(dest.fileno())
        assert digest.hexdigest() == selected["sha256"]
        part.replace(target)
    else:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                digest.update(block)
        assert digest.hexdigest() == selected["sha256"], "Existing immutable snapshot hash mismatch"
    audit["secondary_immutable_snapshot"] = str(target)
    write(report / "recovery_audit.json", audit)
    command = audit["command"]
    with (report / "supervisor.log").open("a") as f:
        p = subprocess.Popen(
            command,
            cwd=WORK,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    write(
        report / "launch.json",
        dict(supervisor_pid=p.pid, command=command, timestamp_unix=time.time(), resources_before=before),
    )
    print(json.dumps(dict(supervisor_pid=p.pid, report=str(report))), flush=True)


def observe(report, seconds):
    report = Path(report)
    start = time.monotonic()
    samples = []
    audit = read(report / "recovery_audit.json")
    first = None
    while time.monotonic() - start < seconds:
        r = resources()
        state = read(ROOT / "STATE.json")
        owner = read(ROOT / "SUPERVISOR.json")
        r.update(
            phase=state.get("phase"),
            update=state.get("update"),
            exposures=state.get("exposures"),
            trainer_pid=state.get("pid"),
            supervisor_pid=owner.get("pid"),
            state_age_seconds=time.time() - state.get("timestamp_unix", 0),
        )
        try:
            r["trainer_io"] = Path("/proc").joinpath(str(state["pid"]), "io").read_text()
        except OSError:
            r["trainer_io"] = None
        if (
            state.get("timestamp_unix", 0) > read(report / "launch.json")["timestamp_unix"]
            and state.get("update", 0) > audit["selected"]["position"]["update"]
            and first is None
        ):
            # Tool-approval latency can allow several heartbeats before observer starts.
            # Inspect a bounded tail and verify the first persisted resumed heartbeat.
            with (ROOT / "production/logs/training.jsonl").open("rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4 * 1024**2))
                tail = f.read().splitlines()
            candidates = []
            for line in tail:
                try:
                    saved = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                if (
                    saved.get("timestamp_unix", 0) > read(report / "launch.json")["timestamp_unix"]
                    and saved.get("update", 0) > audit["selected"]["position"]["update"]
                ):
                    candidates.append(saved)
            state = min(candidates, key=lambda x: x["timestamp_unix"]) if candidates else state
            first = state
            expected = next(x for x in audit["next_updates"] if x["update"] == state["update"])
            assert state["exposures"] == expected["exposures"] and state["schedule_cursor"] == expected["cursor"]
            assert state["precision"] == "bf16" and state["fp32_geometry_islands"] and state["raw_and_ema"]
            assert math.isfinite(state["loss_groups"]["total_loss"])
            assert state["loss_groups"]["timesteps"] == expected["microbatches"][-1]["timesteps"]
            assert state["loss_groups"]["self_conditioned"] == expected["microbatches"][-1]["self_conditioned"]
            write(
                report / "first_resumed_heartbeat.json",
                dict(
                    state=state,
                    expected=expected,
                    match=True,
                    first_computed_update=audit["next_updates"][0]["update"],
                    first_computed_exposures=audit["next_updates"][0]["exposures"],
                    note="Trainer heartbeats every five updates; first operation is derived from strict restore and deterministic plan. First emitted resumed heartbeat independently checks cursor, timesteps and self-conditioning.",
                ),
            )
        with (report / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        samples.append(r)
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "timestamp_unix",
                        "phase",
                        "update",
                        "exposures",
                        "state_age_seconds",
                        "ram_available_bytes",
                        "swap_used_bytes",
                        "d_free_bytes",
                        "gpu_csv",
                        "load_average",
                    )
                }
            ),
            flush=True,
        )
        if r["ram_available_bytes"] < 2 * 1024**3 or r["d_free_bytes"] < 30 * 1024**3:
            write(report / "resource_pressure.json", r)
        time.sleep(30)
    fresh = [r for r in samples if r["state_age_seconds"] < 120]
    summary = dict(
        observation_seconds=time.monotonic() - start,
        samples=len(samples),
        first_heartbeat_verified=first is not None,
        minimum_ram_available_bytes=min(r["ram_available_bytes"] for r in samples),
        maximum_swap_used_bytes=max(r["swap_used_bytes"] for r in samples),
        minimum_d_free_bytes=min(r["d_free_bytes"] for r in samples),
        initial_update=fresh[0]["update"] if fresh else None,
        final_update=fresh[-1]["update"] if fresh else None,
        initial_exposures=fresh[0]["exposures"] if fresh else None,
        final_exposures=fresh[-1]["exposures"] if fresh else None,
    )
    write(report / "training_observation.json", summary)


def pause(report):
    import signal

    report = Path(report)
    launch = read(report / "launch.json")
    state = read(ROOT / "STATE.json")
    latest = read(ROOT / "production/checkpoints/LATEST.json")
    assert time.time() - launch["timestamp_unix"] >= 1800, "Thirty minutes stable training required"
    assert latest["position"]["update"] > read(report / "recovery_audit.json")["selected"]["position"]["update"]
    named = Path(latest["path"]).name.startswith("epoch_")
    assert named or Path(latest["path"]).name.startswith("recovery_"), (
        "Named fraction/epoch or normal rolling boundary required"
    )
    assert named or Path(latest["path"]).stat().st_mtime >= launch["timestamp_unix"] + 1800, (
        "Rolling checkpoint must follow thirty minutes of training"
    )
    pid = state["pid"]
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    assert b"run_e015_integrated.py" in cmd and b"--resume" in cmd
    # Existing Trainer.request_stop handles SIGINT by setting stopping=True.
    # No forceful termination: finish current update, atomic verified save, exit.
    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        fd = os.pidfd_open(pid)
        signal.pidfd_send_signal(fd, signal.SIGINT)
        os.close(fd)
    else:
        # Conda Python was built without pidfd wrappers; use the same Linux ABI.
        import ctypes

        assert os.uname().machine == "x86_64"
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(434, pid, 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "pidfd_open")
        try:
            assert Path(f"/proc/{pid}/cmdline").read_bytes() == cmd
            if libc.syscall(424, fd, int(signal.SIGINT), 0, 0) < 0:
                raise OSError(ctypes.get_errno(), "pidfd_send_signal")
        finally:
            os.close(fd)
    write(
        report / "graceful_pause_request.json",
        dict(
            timestamp_unix=time.time(),
            trainer_pid=pid,
            mechanism="Existing Trainer.request_stop SIGINT handler via PID-safe fd; completes current update and calls atomic_checkpoint; no forceful termination",
            durable_before=latest,
        ),
    )
    print(json.dumps(dict(graceful_pause_requested=pid)), flush=True)


def wait_boundary(report):
    report = Path(report)
    launch = read(report / "launch.json")
    start = time.monotonic()
    while time.monotonic() - start < 3600:
        r = resources()
        s = read(ROOT / "STATE.json")
        latest = read(ROOT / "production/checkpoints/LATEST.json")
        r.update(
            update=s.get("update"),
            exposures=s.get("exposures"),
            trainer_pid=s.get("pid"),
            supervisor_pid=launch["supervisor_pid"],
            phase=s.get("phase"),
            state_age_seconds=time.time() - s.get("timestamp_unix", 0),
        )
        with (report / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        assert r["ram_available_bytes"] >= 2 * 1024**3 and r["d_free_bytes"] >= 30 * 1024**3, (
            "Pressure: evaluation forbidden"
        )
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in (
                        "timestamp_unix",
                        "phase",
                        "update",
                        "exposures",
                        "ram_available_bytes",
                        "swap_used_bytes",
                        "gpu_csv",
                        "state_age_seconds",
                    )
                }
            ),
            flush=True,
        )
        if (
            Path(latest["path"]).name.startswith("recovery_")
            and Path(latest["path"]).stat().st_mtime >= launch["timestamp_unix"] + 1800
        ):
            write(
                report / "evaluation_boundary.json",
                dict(latest=latest, resources=r, elapsed_since_launch=time.time() - launch["timestamp_unix"]),
            )
            return
        time.sleep(30)
    raise RuntimeError("No eligible durable rolling boundary within one hour; evaluation deferred")


def watch_eval(pid):
    out = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
    previous_swap = None
    while Path(f"/proc/{pid}/cmdline").exists():
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"e015_serial_evaluation.py" not in cmd:
            break
        r = resources()
        progress = read(out / "logs/progress.json") if (out / "logs/progress.json").exists() else {}
        r.update(
            evaluator_pid=pid,
            trainer_pid=None,
            evaluation_phase=progress.get("phase"),
            current_sample=progress.get("sample", progress.get("identity")),
        )
        with (out / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        if (
            r["ram_available_bytes"] < 2 * 1024**3
            or r["d_free_bytes"] < 30 * 1024**3
            or (previous_swap is not None and r["swap_used_bytes"] - previous_swap > 256 * 1024**2)
        ):
            write(out / "STOP_EVALUATION.json", dict(reason="Resource pressure", resources=r))
        previous_swap = r["swap_used_bytes"]
        time.sleep(30)


if __name__ == "__main__":
    if sys.argv[1] == "audit":
        audit()
    elif sys.argv[1] == "audit-latest":
        audit(True)
    elif sys.argv[1] == "launch":
        launch(sys.argv[2])
    elif sys.argv[1] == "resources":
        print(json.dumps(resources()))
    elif sys.argv[1] == "observe":
        observe(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == "pause":
        pause(sys.argv[2])
    elif sys.argv[1] == "watch-eval":
        watch_eval(int(sys.argv[2]))
    elif sys.argv[1] == "wait-boundary":
        wait_boundary(sys.argv[2])
