"""Single persistent owner, CUDA queue, mandatory gates and automatic launch."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import e015_bootstrap
import yaml

from protein_distance_diffusion.training.e015_runtime import Runtime, e007_active, status

CONFIG = e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    root = Path(cfg["output_dir"])
    lock = open(root / "SUPERVISOR.lock", "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("E015 supervisor already owns the run") from None
    os.nice(10)
    runtime = Runtime(CONFIG)

    def report(phase, **kwargs):
        record = status(phase, **kwargs)
        runtime.write("SUPERVISOR.json", record)
        if phase.startswith("waiting"):
            runtime.write("HEARTBEAT.json", record)

    def wait_gpu():
        while True:
            runtime.disk_check()
            other_e015 = False
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) == os.getpid():
                    continue
                try:
                    parts = (entry / "cmdline").read_bytes().split(b"\0")
                except OSError:
                    continue
                if parts and Path(parts[0].decode(errors="ignore")).name.startswith("python"):
                    if any(
                        any(
                            name in part
                            for name in (
                                b"autotune_e015_batches.py",
                                b"profile_e015_capacity.py",
                                b"run_e015_integrated.py",
                                b"preflight_e015.py",
                            )
                        )
                        for part in parts[1:]
                    ):
                        other_e015 = True
                        break
            if not e007_active(cfg) and not other_e015:
                apps = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                if apps.returncode == 0 and not apps.stdout.strip():
                    return
                # Windows/WSL may report nonprocess rows; conservatively require
                # <1.5 GiB external use in addition to no active E007 trainer.
                memory = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                )
                if memory.returncode == 0 and int(memory.stdout.strip()) < 1536:
                    return
            report("waiting_for_gpu", e007_active=e007_active(cfg), e007_modified=False)
            time.sleep(30)

    def child(script, arguments, phase):
        wait_gpu()
        report(phase)
        command = [sys.executable, "-B", str(e015_bootstrap.WORKTREE / "scripts" / script), *arguments]
        with open(runtime.path("logs/" + phase + ".log"), "a") as stream:
            process = subprocess.Popen(command, cwd=e015_bootstrap.WORKTREE, stdout=stream, stderr=subprocess.STDOUT)
            report(phase, child_pid=process.pid, command=command)
            result = process.wait()
        if result:
            raise RuntimeError(f"{phase} failed, exit={result}; see D: log")

    try:
        cpu = json.loads(runtime.path("preflight/cpu_tests.json").read_text())
        calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
        contract = runtime.contract(calibration)
        if (
            not cpu["passed"]
            or cpu["source_hashes"] != contract["source_hashes"]
            or cpu["config_hash"] != contract["config_hash"]
        ):
            raise RuntimeError("CPU/lint certification missing or stale")
        if not runtime.path("calibration/reliability.json").exists():
            child("calibrate_e015.py", ["reliability"], "reliability_calibration")
        if not runtime.path("calibration/frozen.json").exists():
            child("calibrate_e015.py", ["gradients"], "gradient_calibration")
        if not runtime.path("batch_autotuning.json").exists() or any(
            r.get("profile_version") != 3
            for r in json.loads(runtime.path("batch_autotuning.json").read_text())["selected"].values()
        ):
            child("autotune_e015_batches.py", [], "batch_autotuning")
        if not runtime.path("preflight/full_certification.json").exists():
            child("preflight_e015.py", [], "full_cuda_certification")
        for updates in (100, 1000):
            folder = f"preflight/smoke_{updates}"
            if runtime.path(folder + "/PASSED.json").exists():
                continue
            command = ["--smoke-updates", str(updates)]
            if runtime.path(folder + "/checkpoints/LATEST.json").exists():
                command += ["--resume"]
            child("run_e015_integrated.py", command, f"train_smoke_{updates}")
            if not runtime.path(folder + "/PASSED.json").exists():
                raise RuntimeError("Smoke exited without passing; guard/pause requires review")
        command = []
        if runtime.path("production/checkpoints/LATEST.json").exists():
            if not args.resume:
                raise RuntimeError("Existing production state requires --resume; no duplicate launch")
            command = ["--resume"]
        child("run_e015_integrated.py", command, "production_training")
        state = json.loads(runtime.path("STATE.json").read_text())
        report("complete" if state["phase"] == "complete" else "paused", training_active=False, final_state=state)
    except Exception as error:
        report("failed_closed", training_active=False, error=str(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
