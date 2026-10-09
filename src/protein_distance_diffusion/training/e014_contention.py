"""Read-only UniRef50 heartbeat comparison; pause E014, never mutate UniRef."""

import json
from pathlib import Path

import numpy as np


def completed_records(log_path):
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
        if r.get("recent_candidates_s", 0) > 0 and r.get("committed_batch", 0) > 0:
            # First heartbeat observed at each commit carries the native batch rate.
            records.setdefault(r["committed_batch"], r)
    return [records[k] for k in sorted(records)]


def baseline(cfg):
    rows = completed_records(cfg["heartbeat_log"])[-cfg["rolling_completed_batches"] :]
    if len(rows) < cfg["minimum_completed_batches"]:
        raise ValueError("insufficient before-probe sequence throughput evidence")
    rates = np.asarray([r["recent_candidates_s"] for r in rows])
    median = float(np.median(rates))
    return dict(
        records=rows,
        median_candidates_s=median,
        mad_candidates_s=float(np.median(abs(rates - median))),
        committed_batch=rows[-1]["committed_batch"],
    )


def compare(before, cfg):
    state = json.loads(Path(cfg["state_path"]).read_text())
    rows = [r for r in completed_records(cfg["heartbeat_log"]) if r["committed_batch"] > before["committed_batch"]]
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
