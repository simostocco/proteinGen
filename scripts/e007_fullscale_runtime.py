"""D-resident matched-control contract; historical model and E014 are read-only."""

# ruff: noqa: E402
import os
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "e007_matched_fullscale_v1"
DATA = Path("/mnt/d/Simone/proteinGen/e014_artifacts")
CORRECTED = Path(
    "/mnt/d/Simone/proteinGen_data/sequence_foundation/uniref50_2026_03_e012_v2/v2c_optimization/scientific_corrected_protein_v1"
)


def configure():
    assert RUN.resolve().is_relative_to("/mnt/d")
    RUN.mkdir(exist_ok=True)
    for key, sub in {
        "TMPDIR": "tmp",
        "TMP": "tmp",
        "TEMP": "tmp",
        "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "cache/torch",
        "CUDA_CACHE_PATH": "cache/cuda",
        "TRITON_CACHE_DIR": "cache/triton",
        "MPLCONFIGDIR": "cache/matplotlib",
    }.items():
        path = RUN / "runtime" / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    tempfile.tempdir = os.environ["TMPDIR"]


configure()
import json
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch as torch
import yaml

torch.set_num_threads(1)

from protein_distance_diffusion.training.e014_runtime import atomic_json as atomic_json
from protein_distance_diffusion.training.e014_runtime import canonical_hash, epoch_schedule, sha

CONFIG = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
MODEL_CONFIG = Path("configs/e007_coordinate_generator_v1.yaml")
POPULATION_SHA = "d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9"
PANEL_SHA = "3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf"


def verified_inputs():
    control = yaml.safe_load(Path("configs/e014_matched_e007_control_v1.yaml").read_text())
    for key in ("shared_training_config", "model_config"):
        assert sha(control[key]) == control[key + "_sha256"]
    cfg = yaml.safe_load(CONFIG.read_text())
    population = json.loads((DATA / "population/population.json").read_text())
    for path, digest in {**population["source_hashes"], **population["manifest_hashes"]}.items():
        assert sha(path) == digest, path
    train = pq.read_table(DATA / "population/train.parquet", use_threads=False).to_pylist()
    dev = pq.read_table(DATA / "population/development.parquet", use_threads=False).to_pylist()
    assert len(train) == len({r["sample_id"] for r in train}) == 230132
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == population["train_sample_id_hash"] == POPULATION_SHA
    assert canonical_hash([r["sample_id"] for r in dev]) == PANEL_SHA and len(dev) == 500
    normalization = json.loads(Path(cfg["normalization_artifact"]).read_text())
    assert cfg["coordinate_scale_angstrom"] == 12.22820347644835
    assert "12.22820347644835" in json.dumps(normalization)
    assert cfg["diffusion_steps"] == 500
    return cfg, population, train, dev


def expected_exposures(lengths, epoch, position, batch_sizes, seed):
    schedule = epoch_schedule(lengths, batch_sizes, seed, epoch)
    assert 0 <= position <= len(schedule)
    expected = np.full(len(lengths), epoch - 1, dtype=np.int32)
    for ids in schedule[:position]:
        expected[ids] += 1
    return schedule, expected


def sequence_snapshot():
    state = json.loads((CORRECTED / "production/STATE.json").read_text())
    heartbeat = json.loads((CORRECTED / "production/reports/HEARTBEAT.json").read_text())
    records = []
    # Native command time is the relevant throughput during protected verification.
    for path in sorted((CORRECTED / "reports/command_logs/mmseqs/verify_protected").glob("batch-*/command.rss.txt")):
        if int(path.parent.name.split("-")[-1]) >= state.get("audit_active_batch", 0):
            continue
        text = path.read_text()
        match = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)", text)
        if match and re.search(r"Exit status:\s*0\s*$", text):
            seconds = 0.0
            for value in match[1].split(":"):
                seconds = 60 * seconds + float(value)
            records.append(
                dict(batch=int(path.parent.name.split("-")[-1]), seconds=seconds, candidates_s=100000 / seconds)
            )
    return dict(
        phase=state["phase"],
        timestamp=heartbeat["timestamp_unix"],
        training_launched=state.get("training_launched", False),
        records=records[-8:],
        driver_sha256=state["driver_sha256"],
    )


def sequence_guard(before):
    now = sequence_snapshot()
    rows = [r for r in now["records"] if r["batch"] > before["last_batch"]][-3:]
    persistent = len(rows) >= 3 and all(r["candidates_s"] < before["threshold"] for r in rows)
    terminal = now["phase"] in ("completed", "complete", "frozen_corpus_ready", "failed")
    reason = (
        "persistent_E012_throughput_degradation"
        if persistent
        else "E012_training_started"
        if now["training_launched"]
        else "stale_E012_heartbeat"
        if not terminal and time.time() - now["timestamp"] > 180
        else "E012_phase_changed"
        if not terminal and now["phase"] != before["phase"]
        else None
    )
    return dict(snapshot=now, after_records=rows, pause_reason=reason, enough_batches=len(rows) >= 3)


def baseline():
    now = sequence_snapshot()
    assert now["phase"] == "verify-protected" and len(now["records"]) >= 3
    rates = np.array([r["candidates_s"] for r in now["records"][-5:]])
    median = float(np.median(rates))
    mad = float(np.median(abs(rates - median)))
    return dict(
        phase=now["phase"],
        last_batch=now["records"][-1]["batch"],
        records=now["records"][-5:],
        median=median,
        mad=mad,
        threshold=min(0.9 * median, median - 2 * mad),
        created=time.time(),
    )
