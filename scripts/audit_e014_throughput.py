"""Read-only sustained telemetry and actual durable checkpoint inventory."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import argparse
import json
import subprocess
import time
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha
from protein_distance_diffusion.training.e014_contention import baseline

OUT = ROOT / "throughput_optimization"
OUT.mkdir(exist_ok=True)


def inventory():
    run = ROOT / "e014"
    latest = json.loads((run / "checkpoints/LATEST.json").read_text())
    records = []
    for path in sorted((run / "checkpoints").glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        record = dict(
            path=str(path),
            epoch=state["epoch"],
            position=state["position"],
            update=state["updates"],
            exposures=int(state["exposures"].sum()),
            timestamp_unix=path.stat().st_mtime,
            age_seconds=time.time() - path.stat().st_mtime,
            bytes=path.stat().st_size,
            optimizer_state_entries=len(state["optimizer"]["state"]),
            rng_keys=list(state["rng"]),
            scheduler_state="none; frozen optimizer uses no scheduler",
            next_batch=state["next_batch"],
            schedule_hash=state["schedule_hash"],
            exposure_min=int(state["exposures"].min()),
            exposure_max=int(state["exposures"].max()),
        )
        records.append(record)
    if sha(latest["path"]) != latest["sha256"]:
        raise ValueError("latest durable checkpoint hash failed")
    return dict(latest=latest, checkpoints=records, all_generated_paths_on_d=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    run = ROOT / "e014"
    pid = json.loads((run / "SUPERVISOR.json").read_text())["child_pid"]

    def cpu_ticks():
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(values[11]) + int(values[12])

    previous_cpu, previous_wall = cpu_ticks(), time.monotonic()
    checkpoints = inventory()
    sequence_before = baseline(cfg["contention"])
    samples = []
    start = time.monotonic()
    with (OUT / f"{args.name}.telemetry.jsonl").open("x", buffering=1) as stream:
        while time.monotonic() - start < args.seconds:
            query = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            gpu = [float(v.strip()) if v.strip() != "[N/A]" else None for v in query.split(",")]
            heartbeat = json.loads((run / "HEARTBEAT.json").read_text())
            io = {
                line.split(":")[0]: int(line.split(":")[1]) for line in Path(f"/proc/{pid}/io").read_text().splitlines()
            }
            current_cpu, current_wall = cpu_ticks(), time.monotonic()
            cpu_percent = 100 * (current_cpu - previous_cpu) / os.sysconf("SC_CLK_TCK") / (current_wall - previous_wall)
            previous_cpu, previous_wall = current_cpu, current_wall
            row = dict(
                timestamp_unix=time.time(),
                gpu_utilization_percent=gpu[0],
                memory_utilization_percent=gpu[1],
                gpu_memory_mib=gpu[2],
                gpu_power_watts=gpu[3],
                sm_clock_mhz=gpu[4],
                trainer_cpu_percent=cpu_percent,
                trainer_read_bytes=io["read_bytes"],
                trainer_read_chars=io["rchar"],
                trainer_write_bytes=io["write_bytes"],
                trainer_children=Path(f"/proc/{pid}/task/{pid}/children").read_text().split(),
                heartbeat=heartbeat,
            )
            samples.append(row)
            stream.write(json.dumps(row) + "\n")
            time.sleep(2)
    first, last = samples[0], samples[-1]
    elapsed = last["timestamp_unix"] - first["timestamp_unix"]
    h0, h1 = first["heartbeat"], last["heartbeat"]
    active = h1["timestamp_unix"] - h0["timestamp_unix"]
    summary = dict(
        seconds=elapsed,
        sample_count=len(samples),
        pid=pid,
        proteins_per_second=(h1["total_protein_exposures"] - h0["total_protein_exposures"]) / active,
        updates_per_second=(h1["update"] - h0["update"]) / active,
        trainer_read_bytes_per_second=(last["trainer_read_bytes"] - first["trainer_read_bytes"]) / elapsed,
        trainer_read_chars_per_second=(last["trainer_read_chars"] - first["trainer_read_chars"]) / elapsed,
        checkpoints=checkpoints,
        sequence_before=sequence_before,
        sequence_after=baseline(cfg["contention"]),
        sm_active_utilization="not exposed by nvidia-smi query; SM clock and engine utilization recorded",
        physical_batch_sizes=cfg["batch_sizes"],
        gradient_accumulation=1,
        loader_workers=0,
        first_heartbeat=h0,
        last_heartbeat=h1,
        all_generated_paths_on_d=True,
    )
    for key in (
        "gpu_utilization_percent",
        "memory_utilization_percent",
        "gpu_memory_mib",
        "gpu_power_watts",
        "trainer_cpu_percent",
    ):
        values = [r[key] for r in samples if r[key] is not None]
        summary[key] = dict(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p05=float(np.quantile(values, 0.05)),
            p95=float(np.quantile(values, 0.95)),
            minimum=min(values),
            maximum=max(values),
        )
    atomic_json(OUT / f"{args.name}.summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
