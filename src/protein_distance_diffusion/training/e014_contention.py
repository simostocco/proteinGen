"""Read-only UniRef50 heartbeat comparison; pause E014, never mutate UniRef."""

import json
import re
from pathlib import Path

import numpy as np


def completed_records(log_path, phase="screen", state_path=None):
    path = Path(log_path)
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 2 * 1024 * 1024))
        lines = stream.read().splitlines()
    records = {}
    for line in lines:
        try:
            r = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if phase == "verify-protected" and r.get("phase") == phase:
            audit = r.get("audit", {})
            if audit.get("batch", 0) > 0:
                records.setdefault(audit["batch"], dict(audit, timestamp_unix=r["timestamp_unix"]))
        elif (
            phase == "screen"
            and r.get("phase", "screen") == phase
            and r.get("recent_candidates_s", 0) > 0
            and r.get("committed_batch", 0) > 0
        ):
            # First heartbeat observed at each commit carries the native batch rate.
            records.setdefault(r["committed_batch"], r)
    if phase != "verify-protected":
        return [records[k] for k in sorted(records)]
    # The native screening-rate field stays frozen during verification. Instead
    # use completed audit population deltas and archived /usr/bin/time output.
    archive = Path(state_path).parent.parent / "scientific/reports/command_logs/mmseqs/verify_protected"
    result = []
    for batch in sorted(records):
        if batch - 1 not in records:
            continue
        path = archive / f"batch-{batch:04d}/command.rss.txt"
        if not path.exists():
            continue
        native = path.read_text()
        elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", native)
        if not elapsed or not re.search(r"Exit status:\s*0\s*$", native):
            continue
        seconds = 0.0
        for field in elapsed.group(1).split(":"):
            seconds = 60 * seconds + float(field)
        count = records[batch]["screened"] - records[batch - 1]["screened"]
        if seconds <= 0 or count <= 0:
            continue
        result.append(
            dict(
                committed_batch=batch,
                recent_candidates_s=count / seconds,
                population_count=count,
                command_wall_seconds=seconds,
                phase=phase,
                timestamp_unix=records[batch]["timestamp_unix"],
                native_timing_path=str(path),
            )
        )
    return result


def baseline(cfg):
    phase = json.loads(Path(cfg["state_path"]).read_text()).get("phase")
    if phase not in ("screen", "verify-protected"):
        raise ValueError(f"unsupported active sequence phase for contention measurement: {phase}")
    rows = completed_records(cfg["heartbeat_log"], phase, cfg["state_path"])[-cfg["rolling_completed_batches"] :]
    if len(rows) < cfg["minimum_completed_batches"]:
        raise ValueError("insufficient before-probe sequence throughput evidence")
    rates = np.asarray([r["recent_candidates_s"] for r in rows])
    median = float(np.median(rates))
    return dict(
        phase=phase,
        records=rows,
        median_candidates_s=median,
        mad_candidates_s=float(np.median(abs(rates - median))),
        committed_batch=rows[-1]["committed_batch"],
    )


def compare(before, cfg):
    state = json.loads(Path(cfg["state_path"]).read_text())
    phase = state.get("phase")
    changed = phase != before.get("phase", "screen")
    rows = (
        []
        if changed
        else [
            r
            for r in completed_records(cfg["heartbeat_log"], phase, cfg["state_path"])
            if r["committed_batch"] > before["committed_batch"]
        ]
    )
    recent = rows[-cfg["minimum_completed_batches"] :]
    enough = len(recent) >= cfg["minimum_completed_batches"]
    median = float(np.median([r["recent_candidates_s"] for r in recent])) if recent else None
    threshold = min(
        (1 - cfg["degradation_fraction"]) * before["median_candidates_s"],
        before["median_candidates_s"] - 2 * before["mad_candidates_s"],
    )
    persistent = enough and all(r["recent_candidates_s"] < threshold for r in recent)
    return dict(
        phase=state.get("phase"),
        phase_changed=changed,
        screened=state.get("screened"),
        timestamp_unix=state.get("timestamp_unix"),
        during_records=recent,
        enough_completed_batches=enough,
        before_median_candidates_s=before["median_candidates_s"],
        during_median_candidates_s=median,
        throughput_ratio=median / before["median_candidates_s"] if median else None,
        degradation_threshold_candidates_s=threshold,
        persistent_material_degradation=persistent,
    )
