# ruff: noqa: E501
"""Continue valid latest evaluation artifacts after a recoverable stage error."""

import json
import subprocess
import time
from pathlib import Path

import e015_latest_control as control

r = control.r


def main(out):
    out = Path(out)
    requested = r.read(r.ROOT / "production/checkpoints/LATEST.json")
    # The original owner must finish its safe resume observation and exit.
    while control.alive(518279):
        time.sleep(10)
    previous = r.read(out / "recovery_resume_proof.json")
    assert previous["completed"] == ["reconstruction"] and previous["resumed"]
    r.write(out / "logs/resume_after_plot_error.json", previous)
    state = r.read(r.ROOT / "STATE.json")
    owner = r.read(r.ROOT / "SUPERVISOR.json")
    assert state["phase"].startswith("training") and owner["child_pid"] == state["pid"]
    while (
        r.read(r.ROOT / "production/checkpoints/LATEST.json")["position"]["update"] <= requested["position"]["update"]
    ):
        time.sleep(30)
    request = out / "logs/continuation_pause"
    request.mkdir(exist_ok=True)
    while time.time() - owner["timestamp_unix"] < 1800:
        time.sleep(10)
    r.write(request / "launch.json", dict(timestamp_unix=owner["timestamp_unix"], supervisor_pid=owner["pid"]))
    r.write(request / "recovery_audit.json", dict(selected=requested))
    r.pause(request)
    while control.alive(state["pid"]) or control.alive(owner["pid"]):
        time.sleep(5)
    control.gate()
    pause_audit = control.run_audit(out / "logs", "continuation_pause_audit")
    pause = r.read(pause_audit / "recovery_audit.json")["selected"]
    state = r.read(r.ROOT / "STATE.json")
    assert (
        state["phase"] == "paused_guard"
        and state["update"] == pause["position"]["update"]
        and state["exposures"] == pause["position"]["exposures"]
    )
    r.write(
        out / "logs/continuation_pause.json",
        dict(
            checkpoint=pause,
            audit=str(pause_audit),
            primary_evaluation_unchanged=r.read(out / "checkpoint_selection.json"),
            zero_uncommitted_exposures=True,
        ),
    )
    completed = ["reconstruction"]
    error = None
    try:
        for phase in ("generation", "analysis", "visuals", "report"):
            resource = control.gate()
            r.write(out / "logs/stage.json", dict(phase=phase, resources=resource, continuation=True))
            with (out / "logs" / (phase + "_continuation.log")).open("a") as stream:
                child = subprocess.Popen(
                    [control.PY, "-B", str(r.WORK / "docs/e015_latest_evaluation.py"), phase, str(out)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                while child.poll() is None:
                    data = r.resources()
                    data.update(phase=phase, evaluator_pid=child.pid, trainer_pid=None, continuation=True)
                    with (out / "resource_monitor.jsonl").open("a") as f:
                        f.write(json.dumps(data) + "\n")
                    if (
                        data["ram_available_bytes"] < 3 * 1024**3
                        or data["d_free_bytes"] < 30 * 1024**3
                        or data["swap_used_bytes"] > 256 * 1024**2
                    ):
                        r.write(out / "STOP_EVALUATION.json", data)
                    time.sleep(30)
                assert child.returncode == 0, f"{phase} exited {child.returncode}"
            control.gate()
            completed.append(phase)
            print(json.dumps(dict(stage=phase, complete=True)), flush=True)
    except Exception as exc:
        error = str(exc)
        print(json.dumps(dict(error=error, completed=completed)), flush=True)
    finally:
        control.gate()
        audit = control.run_audit(out / "logs", "continuation_resume_audit")
        assert r.read(r.ROOT / "production/checkpoints/LATEST.json") == pause
        command = r.read(audit / "recovery_audit.json")["command"]
        with (audit / "supervisor.log").open("a") as stream:
            supervisor = subprocess.Popen(
                command,
                cwd=r.WORK,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        r.write(audit / "launch.json", dict(supervisor_pid=supervisor.pid, command=command, timestamp_unix=time.time()))
        r.observe(audit, 1200)
        latest = r.read(r.ROOT / "production/checkpoints/LATEST.json")
        assert latest["position"]["update"] > pause["position"]["update"]
        import hashlib

        h = hashlib.sha256()
        with Path(latest["path"]).open("rb") as f:
            while block := f.read(8 * 1024**2):
                h.update(block)
        assert h.hexdigest() == latest["sha256"]
        r.write(
            out / "recovery_resume_proof.json",
            dict(
                selected=pause,
                primary_evaluation=r.read(out / "checkpoint_selection.json"),
                resumed=True,
                training_contract_complete=False,
                audit=str(audit),
                first_operation=r.read(audit / "first_resumed_heartbeat.json"),
                rolling=latest,
                rolling_sha_verified=True,
                supervisor_pid=supervisor.pid,
                trainer_pid=r.read(r.ROOT / "STATE.json")["pid"],
                completed=completed,
                error=error,
            ),
        )
        print(json.dumps(dict(phase="continuation_complete", out=str(out), error=error)), flush=True)
