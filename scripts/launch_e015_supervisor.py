"""Launch one detached owner and save an honest D-resident launch record."""

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import e015_bootstrap

from protein_distance_diffusion.training.e014_runtime import sha
from protein_distance_diffusion.training.e015_runtime import Runtime

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def main():
    runtime = Runtime(CONFIG)
    lock = open(runtime.path("LAUNCH.lock"), "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    existing = runtime.path("SUPERVISOR.json")
    if existing.exists():
        owner = json.loads(existing.read_text()).get("pid")
        cmd = Path(f"/proc/{owner}/cmdline")
        if cmd.exists() and b"supervise_e015_training" in cmd.read_bytes():
            raise RuntimeError("E015 supervisor already active")
    if runtime.path("production/checkpoints/LATEST.json").exists():
        raise RuntimeError("Use explicit supervisor --resume for existing production state")
    command = [sys.executable, "-B", str(e015_bootstrap.WORKTREE / "scripts/supervise_e015_training.py")]
    with open(runtime.path("logs/supervisor.log"), "a") as stream:
        process = subprocess.Popen(
            command,
            cwd=e015_bootstrap.WORKTREE,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    record = dict(
        supervisor_pid=process.pid,
        command=command,
        worktree=str(e015_bootstrap.WORKTREE),
        artifact_root=str(runtime.root),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=e015_bootstrap.WORKTREE, text=True
        ).strip(),
        config_sha256=sha(CONFIG),
        launch_timestamp_unix=time.time(),
        phase="supervisor_started_gates_pending",
        production_training_confirmed=False,
        e007_modified=False,
        e012_modified=False,
        full_t0_sampling=False,
    )
    runtime.write("LAUNCH.json", record)
    print(json.dumps(record), flush=True)
    fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
