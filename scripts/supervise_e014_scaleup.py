"""Resume E014 after sequence CPU availability; never restart technical failures."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
import yaml
from protein_distance_diffusion.training.e014_runtime import atomic_json

STOP = False
CHILD = None


def stop_requested(_sig, _frame):
    global STOP
    STOP = True
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def main():
    global CHILD
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    run = ROOT / "e014"
    while not STOP:
        CHILD = subprocess.Popen([sys.executable, "-B", "scripts/run_e014_scaleup.py", "--resume"])
        while CHILD.poll() is None:
            atomic_json(
                run / "SUPERVISOR.json",
                dict(
                    phase="training_process_active",
                    child_pid=CHILD.pid,
                    timestamp_unix=time.time(),
                    sequence_job_modified=False,
                ),
            )
            time.sleep(30)
        if STOP:
            return
        heartbeat = json.loads((run / "HEARTBEAT.json").read_text())
        phase = heartbeat["phase"]
        if phase == "completed":
            atomic_json(run / "SUPERVISOR.json", dict(phase="completed", timestamp_unix=time.time()))
            return
        allowed_pauses = (
            "paused_sequence_phase_changed",
            "paused_sequence_contention",
            "paused_sequence_heartbeat_uncertain",
        )
        if phase not in allowed_pauses:
            atomic_json(
                run / "SUPERVISOR.json",
                dict(
                    phase="stopped_fail_closed",
                    trainer_phase=phase,
                    child_exit_code=CHILD.returncode,
                    timestamp_unix=time.time(),
                    automatic_restart=False,
                ),
            )
            return
        # Native production source confirms this explicit terminal phase. Other
        # phases keep priority; no production process/files are modified.
        while not STOP:
            sequence = json.loads(Path(cfg["contention"]["state_path"]).read_text())
            atomic_json(
                run / "SUPERVISOR.json",
                dict(
                    phase="waiting_sequence_cpu_availability",
                    trainer_phase=phase,
                    sequence_phase=sequence["phase"],
                    committed_updates=heartbeat["update"],
                    timestamp_unix=time.time(),
                ),
            )
            if sequence["phase"] == "frozen_corpus_ready":
                break
            if sequence["phase"] == "failed":
                atomic_json(
                    run / "SUPERVISOR.json", dict(phase="stopped_sequence_failure_uncertain", automatic_restart=False)
                )
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
