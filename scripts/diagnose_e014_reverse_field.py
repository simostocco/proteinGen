"""Bounded, observational paired-path diagnostic; never launches a trainer."""

# ruff: noqa: E501, I001
from e014_bootstrap import ROOT
import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from coordinate_execution_policy import canonical_runner
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import (
    correlation,
    geometry_metrics,
    paired_state,
    reverse_identity,
    rollout_timesteps,
    select_rollout_subset,
    summarize,
    vector_metrics,
)
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import (
    atomic_json,
    canonical_hash,
    d_path,
    epoch_schedule,
    noise_seed,
    sha,
)

RUN = ROOT / "e014"
OUT = RUN / "reverse_field_50pct_v1"
CP = RUN / "checkpoints/epoch_1_fraction_0.50.pt"
CP_SHA = "2e3eb451b4581931c3eefdf79d3c974a289ed5e61489a72310faec673e51cd29"
PANEL_SHA = "3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf"
STRATA = ("20-64", "65-128", "129-256", "257-384", "385-500")
TIMES = (5, 50, 250, 400, 490)
IDENTITY_TOLERANCE = 5e-6
REPLAY_MSE_TOLERANCE = 5e-6


def heartbeat():
    return json.loads((RUN / "HEARTBEAT.json").read_text())


def write_csv(name, rows):
    path = d_path(OUT / name)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key in ("sample_id", "length_stratum"):
                continue
            row[key] = None if value == "" else (value == "True") if value in ("True", "False") else float(value)
        row["length"], row["timestep"] = int(row["length"]), int(row["timestep"])
    return rows


def write_once_json(path, payload):
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(payload)), path
    else:
        atomic_json(path, payload)


def gpu_snapshot():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader"], text=True
    ).strip()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue-diagnostic", action="store_true", help="Reuse complete saved one-step CSV; never resume training"
    )
    continuation = parser.parse_args().continue_diagnostic
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d_path(OUT).mkdir(exist_ok=continuation)
    start = time.time()
    if continuation:
        previous_progress = json.loads((OUT / "HEARTBEAT.json").read_text())
        assert previous_progress["phase"] == "one_step" and previous_progress["identities"] == 500
        start = (OUT / "HEARTBEAT.json").stat().st_mtime - previous_progress["elapsed_seconds"]
        write_once_json(
            OUT / "CONTINUATION_RECORD.json",
            dict(
                reason="summary writer boolean quantile TypeError; measurements were complete",
                one_step_csv_sha256=sha(OUT / "one_step_per_example.csv"),
                oracle_identity_sha256=sha(OUT / "oracle_identity.json"),
                continuation_timestamp_utc=datetime.now(UTC).isoformat(),
                training_untouched=True,
            ),
        )
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
    population = json.loads((ROOT / "population/population.json").read_text())
    panelpath = ROOT / "population/development.parquet"
    assert sha(panelpath) == population["manifest_hashes"][str(panelpath)]
    panel = pq.read_table(panelpath, use_threads=False).to_pylist()
    assert canonical_hash([r["sample_id"] for r in panel]) == PANEL_SHA
    assert len(panel) == len({r["sample_id"] for r in panel}) == 500
    assert all(r["split"] == "validation" for r in panel)
    assert all(sum(r["length_stratum"] == s for r in panel) == 100 for s in STRATA)
    subset = select_rollout_subset(panel, STRATA)
    write_once_json(
        OUT / "FROZEN_PLAN.json",
        dict(
            checkpoint_sha256=CP_SHA,
            panel_sha256=PANEL_SHA,
            rollout_ids=[r["sample_id"] for r in subset],
            rollout_subset_hash=canonical_hash([r["sample_id"] for r in subset]),
            rollout_subset_size=100,
            per_stratum=20,
            timesteps=TIMES,
            rollout_input_timesteps=rollout_timesteps(),
            reverse_identity_max_abs_tolerance=IDENTITY_TOLERANCE,
            frozen_v_mse_replay_max_abs_tolerance=REPLAY_MSE_TOLERANCE,
            noise_namespace="e014-paired-validation-v1",
            noise_generator="CUDA FP32",
            rollout_noise="fixed initial t=490 noise, reused at every paired reference timestep",
            metric_units="coordinate component MSE normalized; distance MSE Angstrom^2",
            collapse_definition="direct X0 Rg ratio < 0.25; not applied to noisy next states",
            scientific_gates_changed=False,
            full_de_novo=False,
            model_dtype="float32",
            tf32=False,
        ),
    )
    assert sha(CP) == CP_SHA
    state = torch.load(CP, map_location="cpu", weights_only=False)
    contract = state["contract"]
    assert contract["config_hash"] == sha(cfgpath)
    assert contract["data_hash"] == population["train_sample_id_hash"]
    assert contract["population_manifest_hash"] == sha(ROOT / "population/population.json")
    assert contract["normalization_hash"] == sha(cfg["normalization_artifact"]) == cfg["normalization_sha256"]
    assert contract["cache_ready_hash"] == sha(ROOT / "coordinate_cache/READY.json")
    for file, digest in contract["source_hashes"].items():
        actual = (
            hashlib.sha256(canonical_runner(Path(file).read_text()).encode()).hexdigest()
            if file == "scripts/run_e014_scaleup.py"
            else sha(file)
        )
        assert actual == digest, file
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == contract["data_hash"]
    assert set(r["sample_id"] for r in panel).isdisjoint(r["sample_id"] for r in train)
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], 1)
    exposures = np.zeros(len(train), dtype=np.int32)
    for batchids in schedule[: state["position"]]:
        exposures[batchids] += 1
    assert state["schedule_hash"] == canonical_hash(schedule)
    assert np.array_equal(exposures, state["exposures"])
    assert int(exposures.sum()) == 115067
    assert state["updates"] == state["position"] == 90413 and state["epoch"] == 1
    assert state["next_batch"] == schedule[state["position"]]
    model = E014CapacityDenoiser(**contract["capacity_config"]).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 5119069
    assert all(torch.isfinite(p).all() for p in model.parameters())
    before_model = {k: v.clone() for k, v in model.state_dict().items()}
    identity = dict(
        path=str(CP),
        sha256=CP_SHA,
        epoch=1,
        update=90413,
        schedule_cursor=90413,
        exposures=115067,
        timestamp_utc=datetime.fromtimestamp(CP.stat().st_mtime, UTC).isoformat(),
        contract=contract,
        development_panel_hash=PANEL_SHA,
        training_before=heartbeat(),
        gpu_before=gpu_snapshot(),
    )
    if continuation:
        original_identity = json.loads((OUT / "checkpoint_identity.json").read_text())
        assert original_identity["sha256"] == identity["sha256"] and original_identity["contract"] == contract
        identity = original_identity
    else:
        atomic_json(OUT / "checkpoint_identity.json", identity)
    del state, train, schedule, exposures
    diffusion = CoordinateVPDiffusion(cfg["diffusion_steps"])
    cache = CoordinateCache(ROOT)
    scale = cfg["coordinate_scale_angstrom"]
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    old = json.loads((RUN / "evaluation/epoch_1_fraction_0.50_development.metrics.json").read_text())
    old = {(r["sample_id"], r["timestep"]): r for r in old}
    one_step, rollout = [], []
    max_identity, max_replay = 0.0, 0.0
    if continuation:
        one_step = read_csv(OUT / "one_step_per_example.csv")
        assert len(one_step) == 2500
        assert {(r["sample_id"], r["timestep"]) for r in one_step} == {
            (r["sample_id"], t) for r in panel for t in TIMES
        }
        max_replay = max(abs(r["v_mse"] - old[(r["sample_id"], r["timestep"])]["v_mse"]) for r in one_step)
        assert max_replay <= REPLAY_MSE_TOLERANCE
        saved_identity = json.loads((OUT / "oracle_identity.json").read_text())
        assert saved_identity["passed"] and saved_identity["conditions"] == 2500
        max_identity = saved_identity["max_abs_error"]
        assert max_identity <= IDENTITY_TOLERANCE

    def prepare(row, timestep):
        clean, lengths, mask = coordinate_batch([row], cache, scale)
        t = torch.tensor([timestep], device="cuda")
        generator = torch.Generator(device="cuda").manual_seed(
            noise_seed("e014-paired-validation-v1", row["sample_id"], timestep)
        )
        batch = diffusion.make_training_batch(clean, mask, timesteps=t, generator=generator)
        return clean, lengths, mask, t, batch

    # Oracle identity passes on every real panel condition BEFORE any model evaluation.
    for row in [] if continuation else panel:
        for timestep in TIMES:
            clean, _, mask, t, batch = prepare(row, timestep)
            max_identity = max(
                max_identity, reverse_identity(diffusion, clean, batch.coordinate_noise, t, mask, IDENTITY_TOLERANCE)
            )
    write_once_json(
        OUT / "oracle_identity.json",
        dict(max_abs_error=max_identity, tolerance=IDENTITY_TOLERANCE, conditions=2500, passed=True),
    )
    print(f"oracle identity passed: maximum absolute error {max_identity:.9g}", flush=True)

    def progress(phase, identities):
        status = dict(
            phase=phase,
            identities=identities,
            elapsed_seconds=time.time() - start,
            training=heartbeat(),
            gpu=gpu_snapshot(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        atomic_json(OUT / "HEARTBEAT.json", status)
        print(json.dumps(status), flush=True)
        if status["peak_reserved_bytes"] > 2 * 1024**3:
            raise RuntimeError("bounded diagnostic exceeded its 2 GiB reserved envelope")

    for position, row in enumerate([] if continuation else panel):
        for timestep in TIMES:
            clean, lengths, mask, t, batch = prepare(row, timestep)
            current, target = batch.noisy_coordinates, batch.coordinate_v_target
            predicted = model(current, t, lengths, mask)
            previous, xhat, _ = diffusion.deterministic_reverse_step(current, t, predicted, mask)
            oracle = paired_state(diffusion, clean, batch.coordinate_noise, t - 1, mask)
            model_mse = float((previous[mask].double() - oracle[mask].double()).square().mean())
            noop_mse = float((current[mask].double() - oracle[mask].double()).square().mean())
            v_mse = float((predicted - target).square()[mask].mean())
            replay_error = abs(v_mse - old[(row["sample_id"], timestep)]["v_mse"])
            max_replay = max(max_replay, replay_error)
            if replay_error > REPLAY_MSE_TOLERANCE:
                raise ValueError(f"frozen paired-noise v-MSE replay mismatch {replay_error}")
            direct_rg = float(
                xhat[mask].double().square().sum(-1).mean().sqrt() / clean[mask].double().square().sum(-1).mean().sqrt()
            )
            result = dict(
                sample_id=row["sample_id"],
                length=row["length"],
                length_stratum=row["length_stratum"],
                timestep=timestep,
                model_mse=model_mse,
                noop_mse=noop_mse,
                relative_gain=1 - model_mse / noop_mse,
                v_mse=v_mse,
                direct_rg_ratio=direct_rg,
                direct_collapsed=direct_rg < 0.25,
                **vector_metrics(previous - current, oracle - current, mask, "step"),
                **vector_metrics(predicted, target, mask, "v"),
                **geometry_metrics(previous, oracle, mask, scale),
            )
            if not result["finite"]:
                raise ValueError("nonfinite diagnostic prediction")
            one_step.append(result)
        if (position + 1) % 25 == 0:
            progress("one_step", position + 1)
    if not continuation:
        write_csv("one_step_per_example.csv", one_step)
    high = [r for r in one_step if r["timestep"] == 490]
    summary = dict(
        overall=summarize(one_step),
        by_timestep={str(t): summarize([r for r in one_step if r["timestep"] == t]) for t in TIMES},
        by_length={s: summarize([r for r in one_step if r["length_stratum"] == s]) for s in STRATA},
        by_length_timestep={
            f"{s}|{t}": summarize([r for r in one_step if r["length_stratum"] == s and r["timestep"] == t])
            for s in STRATA
            for t in TIMES
        },
        t490_correlations={
            k: correlation([r["direct_rg_ratio"] for r in high], [r[k] for r in high])
            for k in ("v_norm_ratio", "v_cosine", "step_norm_ratio", "step_cosine")
        },
        true_v_identity_max_abs_error=max_identity,
        paired_noise_replay_max_v_mse_error=max_replay,
    )
    atomic_json(OUT / "metrics.json", summary)

    for position, row in enumerate(subset):
        clean, lengths, mask, t, batch = prepare(row, 490)
        epsilon, current = batch.coordinate_noise, batch.noisy_coordinates
        metadata = dict(sample_id=row["sample_id"], length=row["length"], length_stratum=row["length_stratum"])
        rollout.append(
            dict(
                **metadata,
                timestep=490,
                input_timestep=491,
                **geometry_metrics(current, current, mask, scale),
                forward_mse=0.0,
                rollout_mse=0.0,
                exposure_gap=0.0,
                exposure_ratio=None,
            )
        )
        for s in rollout_timesteps():
            t = torch.tensor([s], device="cuda")
            reference = paired_state(diffusion, clean, epsilon, t, mask)
            reference_next = paired_state(diffusion, clean, epsilon, t - 1, mask)
            forward_v = model(reference, t, lengths, mask)
            forward_next, _, _ = diffusion.deterministic_reverse_step(reference, t, forward_v, mask)
            rollout_v = model(current, t, lengths, mask)
            current, _, _ = diffusion.deterministic_reverse_step(current, t, rollout_v, mask)
            fe = float((forward_next[mask].double() - reference_next[mask].double()).square().mean())
            re = float((current[mask].double() - reference_next[mask].double()).square().mean())
            result = dict(
                **metadata,
                timestep=s - 1,
                input_timestep=s,
                **geometry_metrics(current, reference_next, mask, scale),
                forward_mse=fe,
                rollout_mse=re,
                exposure_gap=re - fe,
                exposure_ratio=re / (fe + 1e-12),
            )
            if not result["finite"]:
                raise ValueError("nonfinite short rollout")
            rollout.append(result)
        if (position + 1) % 5 == 0:
            progress("short_rollout", position + 1)
    write_csv("rollout_per_step.csv", rollout)
    rollout_summary = dict(
        subset_size=100,
        subset_hash=canonical_hash([r["sample_id"] for r in subset]),
        by_output_timestep={str(s): summarize([r for r in rollout if r["timestep"] == s]) for s in range(490, 439, -1)},
        reference="paired original-epsilon path; not the unique valid generative trajectory",
    )
    atomic_json(OUT / "rollout_summary.json", rollout_summary)
    assert sha(CP) == CP_SHA
    assert all(torch.equal(value.cpu(), before_model[key]) for key, value in model.state_dict().items())
    identity.update(
        training_after=heartbeat(),
        gpu_after=gpu_snapshot(),
        elapsed_seconds=time.time() - start,
        checkpoint_hash_unchanged=True,
        model_weights_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    atomic_json(OUT / "checkpoint_identity.json", identity)
    progress("complete", 100)


if __name__ == "__main__":
    main()
