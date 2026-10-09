"""Frozen checkpoint preflight and bounded dense-grid inference; never train."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import argparse
import hashlib
import json
import time
import subprocess
from datetime import UTC, datetime
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from coordinate_execution_policy import canonical_runner
from diagnose_e014_reverse_field import PANEL_SHA, heartbeat, gpu_snapshot
from collect_e014_marginal_reverse import FAMILIES, SUBSET_SHA
from upgrade_e014_terminal_guard import FROZEN_PROBE_SOURCE
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    GRID,
    HIGH,
    LOW_MID,
    MARGINAL_GRID,
    ENDPOINTS,
    LABELS,
    STRATA,
    gaussian_seed,
    prefix_steps,
    schedule_values,
)
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import descriptor
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import (
    paired_state,
    reverse_identity,
    select_rollout_subset,
    vector_metrics,
)
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import (
    atomic_json,
    canonical_hash,
    d_path,
    epoch_schedule,
    noise_seed,
    sha,
)

OUT = ROOT / "e014/high_noise_75pct_v1"
CONFIG = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
BASE_MARGINAL = ROOT / "e014/marginal_reverse_50pct_v1"


def save(name, rows):
    target = d_path(OUT / name)
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_suffix(".partial.parquet")
    pq.write_table(pa.Table.from_pylist(rows), partial, compression="zstd")
    assert pq.read_metadata(partial).num_rows == len(rows)
    partial.rename(target)


def population():
    metadata = json.loads((ROOT / "population/population.json").read_text())
    panelpath = ROOT / "population/development.parquet"
    assert sha(panelpath) == metadata["manifest_hashes"][str(panelpath)]
    panel = pq.read_table(panelpath, use_threads=False).to_pylist()
    assert len(panel) == len({r["sample_id"] for r in panel}) == 500
    assert canonical_hash([r["sample_id"] for r in panel]) == PANEL_SHA
    assert all(r["split"] == "validation" for r in panel)
    assert all(sum(r["length_stratum"] == s for r in panel) == 100 for s in STRATA)
    subset = select_rollout_subset(panel, STRATA)
    assert canonical_hash([r["sample_id"] for r in subset]) == SUBSET_SHA
    return metadata, panel, subset


def preflight():
    d_path(OUT).mkdir(exist_ok=False)
    cfg = yaml.safe_load(CONFIG.read_text())
    metadata, panel, subset = population()
    protocol = dict(
        git_base_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        grid=list(GRID),
        high_noise_grid=list(HIGH),
        low_mid_grid=list(LOW_MID),
        marginal_grid=list(MARGINAL_GRID),
        checkpoints=list(LABELS),
        gaussian_endpoints=list(ENDPOINTS),
        paired_namespace="e014-paired-validation-v1",
        marginal_families=FAMILIES,
        panel_hash=PANEL_SHA,
        subset_hash=SUBSET_SHA,
        gaussian_seeds=[gaussian_seed(i, r["length"], noise_seed) for i, r in enumerate(subset)],
        descriptor_definition_sha256=sha(BASE_MARGINAL / "descriptor_definition.json"),
        descriptor_code_sha256=sha("src/protein_distance_diffusion/evaluation/marginal_distribution_diagnostic.py"),
        diffusion_source_sha256=sha("src/protein_distance_diffusion/training/coordinate_diffusion.py"),
        front_rule="energy model-minus-noop 95% percentile bootstrap CI upper<0 AND (MMD2 OR slicedW1 upper<0), mean Rg and trace ratios each in[0.8,1.2]; highest sampled timestep, also contiguous sampled-prefix front",
        negative_rule="energy CI lower>0 AND (MMD2 OR slicedW1 lower>0); else unresolved",
        bootstraps=500,
        paired_bootstrap_seed_namespace="e014-high-noise-paired-bootstrap-v1",
        geometry_front_rule="75 vs50 and50 vs25: equal-protein mean local MSE difference 95% paired protein bootstrap CI upper<0; highest sampled LOW_MID timestep plus contiguous prefix; torsion/chirality separately descriptive",
        direct_onset="first sampled t with median Rg ratio<0.5; separately historical collapse fraction>0.5 (per-protein Rg ratio<0.25)",
        noisy_collapse="descriptive: population mean Rg or covariance trace below0.8 times independent reference; report per-protein scale distributions without matched collapse assignment",
        preprocessing="reference-only fit at each t/stratum; exact same reference populations and normalizer shared across all checkpoints; previous fixed34-descriptor definitions retained",
        true_v_tolerance=5e-6,
        permutation_tolerance=1e-10,
        checkpoint_mutation=False,
        training_mutation=False,
        no_final_adjudication=True,
        statistical_caveats="conditional fixed-reference fits, biased V-statistic bootstrap, exploratory unadjusted multiple comparisons; isolated fronts are not monotone laws or scientific gates",
        created_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(OUT / "PROTOCOL.json", protocol)
    atomic_json(
        OUT / "descriptor_definition.json", json.loads((BASE_MARGINAL / "descriptor_definition.json").read_text())
    )
    atomic_json(
        OUT / "subset_identity.json",
        dict(
            panel_hash=PANEL_SHA,
            subset_hash=SUBSET_SHA,
            ids=[r["sample_id"] for r in subset],
            lengths=[r["length"] for r in subset],
            per_stratum=20,
        ),
    )
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    assert len(train) == 230132 and set(r["sample_id"] for r in train).isdisjoint(r["sample_id"] for r in panel)
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == metadata["train_sample_id_hash"]
    schedule = epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], 1)
    prefix = np.concatenate([[0], np.cumsum([len(batch) for batch in schedule])])
    runner = canonical_runner(Path("scripts/run_e014_scaleup.py").read_text()).encode()
    old_runner = subprocess.check_output(["git", "show", f"{FROZEN_PROBE_SOURCE}:scripts/run_e014_scaleup.py"])
    checkpoints = []
    science = None
    for label in LABELS:
        path = ROOT / "e014/checkpoints" / f"{label}.pt"
        digest = sha(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        contract = state["contract"]
        assert contract["config_hash"] == sha(CONFIG)
        assert contract["data_hash"] == metadata["train_sample_id_hash"]
        assert contract["normalization_hash"] == sha(cfg["normalization_artifact"]) == cfg["normalization_sha256"]
        assert contract["population_manifest_hash"] == sha(ROOT / "population/population.json")
        assert contract["cache_ready_hash"] == sha(ROOT / "coordinate_cache/READY.json")
        for file, value in contract["source_hashes"].items():
            actual = (
                hashlib.sha256(old_runner if label == "initialization" else runner).hexdigest()
                if file == "scripts/run_e014_scaleup.py"
                else sha(file)
            )
            assert actual == value, file
        current_science = {k: v for k, v in contract.items() if k != "source_hashes"}
        if science is None:
            science = current_science
        assert science == current_science
        assert state["epoch"] == 1 and state["updates"] == state["position"]
        expected = np.zeros(len(train), np.int32)
        for batch in schedule[: state["position"]]:
            expected[batch] += 1
        assert np.array_equal(expected, state["exposures"])
        if state["updates"]:
            assert state["schedule_hash"] == canonical_hash(schedule)
            assert state["next_batch"] == schedule[state["position"]]
            assert len(state["optimizer"]["state"]) == 150
        assert set(state["rng"]) == {"python", "numpy", "torch", "cuda"}
        model = E014CapacityDenoiser(**contract["capacity_config"]).eval()
        model.load_state_dict(state["model"], strict=True)
        assert sum(p.numel() for p in model.parameters()) == 5119069
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert sha(path) == digest
        boundary = 0 if label == "initialization" else int(round(float(label.rsplit("_", 1)[1]) * 230132))
        checkpoints.append(
            dict(
                label=label,
                path=str(path),
                sha256=digest,
                epoch=1,
                update=state["updates"],
                schedule_cursor=state["position"],
                exposures=int(expected.sum()),
                nominal_boundary=boundary,
                boundary_offset=int(expected.sum()) - boundary,
                preceding_exposures=int(prefix[max(0, state["position"] - 1)]),
                schedule_hash=canonical_hash(schedule),
                timestamp_utc=datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                contract=contract,
                loadable=True,
                consistent=True,
                model_complete=True,
            )
        )
        del model, state
    atomic_json(
        OUT / "checkpoint_identity.json",
        dict(checkpoints=checkpoints, training_before=heartbeat(), protocol_sha256=sha(OUT / "PROTOCOL.json")),
    )
    print(json.dumps(checkpoints[-1]), flush=True)


@torch.no_grad()
def collect():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(CONFIG.read_text())
    _, panel, subset = population()
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    assert identity["protocol_sha256"] == sha(OUT / "PROTOCOL.json")
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    assert protocol["descriptor_code_sha256"] == sha(
        "src/protein_distance_diffusion/evaluation/marginal_distribution_diagnostic.py"
    )
    cache = CoordinateCache(ROOT)
    diffusion = CoordinateVPDiffusion(500)
    scale = cfg["coordinate_scale_angstrom"]
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    def progress(phase, n, label=None):
        status = dict(
            phase=phase,
            label=label,
            identities=n,
            elapsed_seconds=time.time() - start,
            training=heartbeat(),
            gpu=gpu_snapshot(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        atomic_json(OUT / "HEARTBEAT.json", status)
        print(json.dumps({k: v for k, v in status.items() if k not in ("training",)}), flush=True)
        if status["peak_reserved_bytes"] > 2 * 1024**3:
            raise RuntimeError("diagnostic2GiB envelope exceeded")

    def batch(clean, mask, s, namespace, sample_id):
        t = torch.tensor([s], device="cuda")
        g = torch.Generator(device="cuda").manual_seed(noise_seed(namespace, sample_id, s))
        return diffusion.make_training_batch(clean, mask, timesteps=t, generator=g)

    def desc(row, s, label, x):
        return dict(
            sample_id=row["sample_id"],
            length=row["length"],
            length_stratum=row["length_stratum"],
            timestep=s,
            population=label,
            descriptor=descriptor(x[0].detach().cpu().numpy()).tolist(),
        )

    maximum = 0.0
    for row in subset[::4]:
        clean, _, mask = coordinate_batch([row], cache, scale)
        for s in MARGINAL_GRID:
            b = batch(clean, mask, s, FAMILIES["source"], row["sample_id"])
            maximum = max(
                maximum, reverse_identity(diffusion, clean, b.coordinate_noise, torch.tensor([s], device="cuda"), mask)
            )
    atomic_json(
        OUT / "oracle_identity.json",
        dict(max_abs_error=maximum, tolerance=5e-6, conditions=25 * len(MARGINAL_GRID), passed=True),
    )
    # Independent reference descriptors are generated once and shared unchanged.
    refs = []
    for i, row in enumerate(panel):
        clean, _, mask = coordinate_batch([row], cache, scale)
        for s in MARGINAL_GRID:
            for label, family, target_t in [
                ("ref_a", "ref-a", s - 1),
                ("ref_b", "ref-b", s - 1),
                ("noop", "source", s),
            ]:
                refs.append(
                    desc(
                        row,
                        s,
                        label,
                        batch(clean, mask, target_t, FAMILIES[family], row["sample_id"]).noisy_coordinates,
                    )
                )
        if (i + 1) % 25 == 0:
            progress("references", i + 1)
    save("shared_reference_descriptors.parquet", refs)
    for checkpoint in identity["checkpoints"]:
        label = checkpoint["label"]
        path = Path(checkpoint["path"])
        assert sha(path) == checkpoint["sha256"]
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = E014CapacityDenoiser(**checkpoint["contract"]["capacity_config"]).eval().requires_grad_(False)
        model.load_state_dict(state["model"], strict=True)
        frozen = {k: v.clone() for k, v in model.state_dict().items()}
        del state
        model.cuda()
        metrics = []
        population_rows = list(refs)
        for i, row in enumerate(panel):
            clean, lengths, mask = coordinate_batch([row], cache, scale)
            for s in GRID:
                t = torch.tensor([s], device="cuda")
                b = batch(clean, mask, s, "e014-paired-validation-v1", row["sample_id"])
                prediction, components = model.telemetry(b.noisy_coordinates, t, lengths, mask)
                previous, xhat, _ = diffusion.deterministic_reverse_step(b.noisy_coordinates, t, prediction, mask)
                oracle = paired_state(diffusion, clean, b.coordinate_noise, t - 1, mask)
                result = extended_metrics(
                    xhat[0], clean[0], prediction[0], b.coordinate_v_target[0], scale, components[0]
                )
                result.update(vector_metrics(prediction, b.coordinate_v_target, mask, "v"))
                result.update(
                    vector_metrics(previous - b.noisy_coordinates, oracle - b.noisy_coordinates, mask, "step")
                )
                model_mse = float((previous - oracle).square()[mask].mean())
                noop_mse = float((b.noisy_coordinates - oracle).square()[mask].mean())
                result.update(
                    checkpoint=label,
                    sample_id=row["sample_id"],
                    length=row["length"],
                    length_stratum=row["length_stratum"],
                    timestep=s,
                    model_mse=model_mse,
                    noop_mse=noop_mse,
                    step_gain=1 - model_mse / noop_mse,
                    coordinate_norm_ratio=float(xhat.norm() / clean.norm()),
                    covariance_trace_ratio=float(xhat.square().sum() / clean.square().sum()),
                    **schedule_values(diffusion, s),
                )
                if not result["finite"]:
                    raise ValueError("nonfinite diagnostic output")
                metrics.append(result)
                if s in MARGINAL_GRID:
                    mb = batch(clean, mask, s, FAMILIES["source"], row["sample_id"])
                    mv = model(mb.noisy_coordinates, t, lengths, mask)
                    mp, _, _ = diffusion.deterministic_reverse_step(mb.noisy_coordinates, t, mv, mask)
                    population_rows.append(desc(row, s, "model", mp))
            if (i + 1) % 25 == 0:
                progress("paired_and_marginal", i + 1, label)
        save(f"{label}.paired.parquet", metrics)
        save(f"{label}.descriptors.parquet", population_rows)
        del metrics, population_rows
        if label in LABELS[-2:]:
            rollout = []
            for i, row in enumerate(subset):
                clean, lengths, mask = coordinate_batch([row], cache, scale)
                g = torch.Generator(device="cuda").manual_seed(gaussian_seed(i, row["length"], noise_seed))
                current = centered_coordinate_noise(torch.zeros_like(clean), mask, generator=g)
                initial = current.clone()
                for s in prefix_steps():
                    if s in ENDPOINTS:
                        for pop, x in [
                            ("model", current),
                            ("noop", initial),
                            ("ref_a", batch(clean, mask, s, FAMILIES["ref-a"], row["sample_id"]).noisy_coordinates),
                            ("ref_b", batch(clean, mask, s, FAMILIES["ref-b"], row["sample_id"]).noisy_coordinates),
                        ]:
                            rollout.append(desc(row, s, pop, x))
                    t = torch.tensor([s], device="cuda")
                    v = model(current, t, lengths, mask)
                    current, _, _ = diffusion.deterministic_reverse_step(current, t, v, mask)
                    if not torch.isfinite(current).all():
                        raise ValueError("nonfinite Gaussian prefix")
                s = 400
                for pop, x in [
                    ("model", current),
                    ("noop", initial),
                    ("ref_a", batch(clean, mask, s, FAMILIES["ref-a"], row["sample_id"]).noisy_coordinates),
                    ("ref_b", batch(clean, mask, s, FAMILIES["ref-b"], row["sample_id"]).noisy_coordinates),
                ]:
                    rollout.append(desc(row, s, pop, x))
                if (i + 1) % 5 == 0:
                    progress("gaussian_prefix", i + 1, label)
            save(f"{label}.gaussian_descriptors.parquet", rollout)
        assert all(torch.equal(v.cpu(), frozen[k]) for k, v in model.state_dict().items())
        assert sha(path) == checkpoint["sha256"]
        atomic_json(
            OUT / f"{label}.INFERENCE_COMPLETE.json",
            dict(
                checkpoint_sha256=checkpoint["sha256"],
                checkpoint_unchanged=True,
                model_unchanged=True,
                paired_conditions=500 * len(GRID),
                marginal_conditions=500 * len(MARGINAL_GRID),
            ),
        )
        del model, frozen
        progress("checkpoint_complete", 500, label)
    identity.update(
        training_after=heartbeat(),
        elapsed_seconds=time.time() - start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_and_model_unchanged=True,
    )
    atomic_json(OUT / "checkpoint_identity.json", identity)
    progress("inference_complete", 500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["preflight", "collect"])
    args = parser.parse_args()
    preflight() if args.phase == "preflight" else collect()
