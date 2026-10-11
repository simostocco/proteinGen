# ruff: noqa: E402, E501
"""One serial owner: wait for pause, bounded tiers, guaranteed resume after clean exit."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import e015_serial_recovery as r

ROOT = r.ROOT
WORK = r.WORK
OUT = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
PY = "/home/simostocco/miniforge3/envs/proteingen/bin/python"
ORIGINAL = Path(sys.argv[1])


def alive(pid, name):
    try:
        return name.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def log(phase, **kwargs):
    obj = dict(phase=phase, timestamp_unix=time.time(), controller_pid=os.getpid(), **kwargs)
    r.write(OUT / "logs/serial_cycle.json", obj)
    print(json.dumps(obj), flush=True)


def monitor(event, phase, pid):
    previous_swap = None
    while not event.is_set():
        try:
            resource = r.resources()
            resource.update(evaluation_phase=phase, evaluator_pid=pid, trainer_pid=None)
            try:
                resource["current_sample"] = r.read(OUT / "logs/progress.json")
            except (OSError, ValueError):
                resource["current_sample"] = None
            with (OUT / "resource_monitor.jsonl").open("a") as f:
                f.write(json.dumps(resource) + "\n")
            if (
                resource["ram_available_bytes"] < 2 * 1024**3
                or resource["d_free_bytes"] < 30 * 1024**3
                or (previous_swap is not None and resource["swap_used_bytes"] - previous_swap > 256 * 1024**2)
            ):
                r.write(OUT / "STOP_EVALUATION.json", dict(reason="Monitor resource pressure", resources=resource))
            previous_swap = resource["swap_used_bytes"]
        except Exception as error:
            log("monitor_error", error=str(error))
        event.wait(30)


def tier(name):
    before = r.gate()
    log("starting_" + name, resources=before)
    with (OUT / "logs" / f"{name}.log").open("a") as f:
        child = subprocess.Popen(
            [PY, "-B", str(WORK / "docs/e015_serial_evaluation.py"), name, str(ORIGINAL)],
            cwd=WORK,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        event = threading.Event()
        thread = threading.Thread(target=monitor, args=(event, name, child.pid), daemon=True)
        thread.start()
        result = child.wait()
        event.set()
        thread.join(timeout=5)
    # Child has fully exited; wait for driver bookkeeping to release GPU ownership.
    for _ in range(12):
        try:
            after = r.gate()
            break
        except AssertionError:
            time.sleep(5)
    else:
        raise RuntimeError("GPU ownership not released after evaluation")
    log("finished_" + name, exit_code=result, resources=after)
    if result:
        raise RuntimeError(f"{name} exited {result}; compact partials preserved")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def audit_latest(label):
    path = OUT / "logs" / f"{label}_checkpoint_audit.log"
    with path.open("w") as f:
        subprocess.run(
            [PY, "-B", str(WORK / "docs/e015_serial_recovery.py"), "audit-latest"],
            cwd=WORK,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    result = json.loads(path.read_text().splitlines()[-1])
    return Path(result["report"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    launch = r.read(ORIGINAL / "launch.json")
    owner = launch["supervisor_pid"]
    trainer = r.read(ORIGINAL / "graceful_pause_request.json")["trainer_pid"]
    log("waiting_for_original_clean_exit", trainer_pid=trainer, supervisor_pid=owner)
    started = time.monotonic()
    while alive(trainer, "run_e015_integrated.py") or alive(owner, "supervise_e015_training.py"):
        resource = r.resources()
        heartbeat = r.read(ROOT / "HEARTBEAT.json")
        resource.update(
            phase="waiting_for_existing_graceful_pause", trainer_pid=trainer, supervisor_pid=owner, heartbeat=heartbeat
        )
        with (OUT / "resource_monitor.jsonl").open("a") as f:
            f.write(json.dumps(resource) + "\n")
        if time.monotonic() - started > 6 * 3600:
            raise RuntimeError(
                "Existing evaluator has not returned; no forceful termination or second workload permitted"
            )
        time.sleep(30)
    r.gate()
    state = r.read(ROOT / "STATE.json")
    assert state["phase"] == "paused_guard", state["phase"]
    pause = r.read(ROOT / "production/checkpoints/LATEST.json")
    assert state["update"] == pause["position"]["update"] and state["exposures"] == pause["position"]["exposures"]
    assert pause == r.read(ROOT / "COMMITTED.json") == r.read(ROOT / "production/COMMITTED.json")
    log(
        "pause_complete",
        state=dict(update=state["update"], exposures=state["exposures"]),
        checkpoint=pause,
        zero_uncommitted_exposures=True,
    )
    pause_report = audit_latest("pause")
    r.write(
        OUT / "recovery_audit.json",
        dict(
            initial_recovery=r.read(ORIGINAL / "recovery_audit.json"),
            pause_checkpoint=pause,
            pause_verified_report=str(pause_report),
            zero_uncommitted_exposures=True,
        ),
    )
    completed = []
    error = None
    try:
        for phase in ("tier1", "tier2", "pilot", "panel"):
            tier(phase)
            completed.append(phase)
    except Exception as exc:
        error = str(exc)
        log("evaluation_stopped", error=error, completed=completed, traceback=traceback.format_exc())
    finally:
        # Never restart while an evaluator remains alive or owns CUDA.
        r.gate()
        assert r.read(ROOT / "production/checkpoints/LATEST.json") == pause
        assert sha(pause["path"]) == pause["sha256"]
        resume_report = audit_latest("resume")
        r.write(
            OUT / "logs/evaluation_completion.json",
            dict(
                completed=completed,
                error=error,
                stopped_early=error is not None,
                resource_pressure=(OUT / "STOP_EVALUATION.json").exists(),
                pause_checkpoint=pause,
            ),
        )
        r.launch(resume_report)
        log("training_resumed", resume_report=str(resume_report), completed=completed, error=error)
        r.observe(resume_report, 1200)
        latest = r.read(ROOT / "production/checkpoints/LATEST.json")
        assert latest["position"]["update"] > pause["position"]["update"], "Next rolling checkpoint not yet committed"
        assert sha(latest["path"]) == latest["sha256"]
        r.write(
            OUT / "logs/final_resume.json",
            dict(
                resume_report=str(resume_report),
                launch=r.read(resume_report / "launch.json"),
                first_resumed_heartbeat=r.read(resume_report / "first_resumed_heartbeat.json"),
                next_rolling_checkpoint=latest,
                rolling_sha_verified=True,
                completed=completed,
                error=error,
            ),
        )
        log("cycle_complete", completed=completed, error=error, next_rolling_checkpoint=latest)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "report":
        subprocess.run([PY, "-B", str(WORK / "docs/e015_serial_report.py"), str(ORIGINAL)], cwd=WORK, check=True)
    else:
        main()
