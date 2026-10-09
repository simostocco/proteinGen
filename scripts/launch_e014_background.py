"""Continue a passing E014 probe in a persistent, D-logged process."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import fcntl
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-concurrent-sequence", action="store_true")
    args = parser.parse_args()
    run = ROOT / "e014"
    heartbeat = json.loads((run / "HEARTBEAT.json").read_text())
    probe = json.loads((run / "CONTENTION_PROBE.json").read_text())
    if not args.allow_concurrent_sequence and (
        heartbeat["phase"] != "probe_completed_resumable" or not probe["passed"]
    ):
        raise ValueError("a verified passing contention probe is required; no automatic failed-gate restart")
    # Check trainer ownership without changing the protected sequence job.
    with open(run / "OWNER.lock", "a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(owner, fcntl.LOCK_UN)
    stamp = time.time_ns()
    logfile = run / f"background_{stamp}.log"
    env = dict(os.environ, PYTHONPATH="src", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", "scripts/supervise_e014_scaleup.py"]
    if args.allow_concurrent_sequence:
        command = [sys.executable, "-B", "scripts/run_e014_scaleup.py", "--resume", "--allow-concurrent-sequence"]
    with logfile.open("xb") as stream:
        child = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record = dict(
        pid=child.pid,
        command=command,
        logfile=str(logfile),
        timestamp_unix=time.time(),
        config_sha256=sha("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml"),
        launch_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        persistent_process=True,
        all_generated_paths_on_d=True,
        sequence_job_modified=False,
    )
    atomic_json(run / "BACKGROUND_LAUNCH.json", record)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
