"""Frozen checkpoint preflight and bounded dense-grid inference; never train."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import json
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from diagnose_e014_reverse_field import PANEL_SHA, heartbeat, gpu_snapshot
from collect_e014_marginal_reverse import FAMILIES, SUBSET_SHA
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.high_noise_resolution import (
    GRID,
    MARGINAL_GRID,
    STRATA,
    gaussian_seed,
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
    noise_seed,
    sha,
)

from evaluate_e014_post_reboot import OUT, OLD, SELECTED

ENDPOINTS = (490, 480, 470, 460, 450, 440, 420, 400, 350)


def prefix_steps():
    return tuple(range(499, 350, -1))


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


@torch.no_grad()
def collect():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(CONFIG.read_text())
    _, panel, subset = population()
    identity = json.loads((OUT / "checkpoint_identity.json").read_text())
    protocol = json.loads((OUT / "PROTOCOL.json").read_text())
    assert protocol["reference_sha256"] == sha(OLD / "shared_reference_descriptors.parquet")
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
    refs = pq.read_table(OLD / "shared_reference_descriptors.parquet", use_threads=False).to_pylist()
    for checkpoint in identity["checkpoints"]:
        label = checkpoint["label"]
        if (OUT / f"{label}.INFERENCE_COMPLETE.json").exists():
            continue
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
        if label in SELECTED:
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
                s = 350
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
    collect()
