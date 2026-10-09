"""Frozen-checkpoint inference only; independent reference marginal populations."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT
import json
import hashlib
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from coordinate_execution_policy import canonical_runner
from diagnose_e014_reverse_field import CP, CP_SHA, PANEL_SHA, STRATA, heartbeat, gpu_snapshot
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import NAMES, descriptor
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import reverse_identity, select_rollout_subset
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, d_path, noise_seed, sha

OUT = ROOT / "e014/marginal_reverse_50pct_v1"
TIMES = (5, 50, 250, 400, 490)
FAMILIES = {label: f"e014-marginal-{label}-v1" for label in ("source", "ref-a", "ref-b", "gaussian")}
SUBSET_SHA = "c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0"


def save_rows(name, rows):
    path = d_path(OUT / name)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(".partial.parquet")
    pq.write_table(pa.Table.from_pylist(rows), partial, compression="zstd")
    assert pq.read_metadata(partial).num_rows == len(rows)
    partial.rename(path)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d_path(OUT).mkdir(exist_ok=False)
    start = time.time()
    panelpath = ROOT / "population/development.parquet"
    population = json.loads((ROOT / "population/population.json").read_text())
    assert sha(panelpath) == population["manifest_hashes"][str(panelpath)]
    panel = pq.read_table(panelpath, use_threads=False).to_pylist()
    assert len(panel) == len({r["sample_id"] for r in panel}) == 500
    assert all(r["split"] == "validation" for r in panel)
    assert canonical_hash([r["sample_id"] for r in panel]) == PANEL_SHA
    assert all(sum(r["length_stratum"] == s for r in panel) == 100 for s in STRATA)
    subset = select_rollout_subset(panel, STRATA)
    assert canonical_hash([r["sample_id"] for r in subset]) == SUBSET_SHA
    atomic_json(
        OUT / "subset_identity.json",
        dict(
            panel_sha256=PANEL_SHA,
            subset_sha256=SUBSET_SHA,
            subset_ids=[r["sample_id"] for r in subset],
            lengths=[r["length"] for r in subset],
            gaussian_seeds=[
                noise_seed(FAMILIES["gaussian"], f"sample-{i}-N{r['length']}", 499) for i, r in enumerate(subset)
            ],
            per_stratum=20,
            panel_per_stratum=100,
            seed_families=FAMILIES,
        ),
    )
    atomic_json(
        OUT / "descriptor_definition.json",
        dict(
            names=NAMES,
            count=34,
            coordinates="centered normalized C-alpha",
            covariance="X.T X / N; sorted descending eigenvalues; trace; eig1/trace and eig3/trace",
            component_rms="scalar sqrt(mean squared xyz) = Rg/sqrt(3), rotation invariant",
            pairs="unique unordered i<j; Euclidean distance mean/population std/p05/p25/p50/p75/p95",
            separations=[[1, 1], [2, 2], [3, 3], [4, 8], [9, 32], [33, None]],
            missing="unavailable separation >32 yields NaN; reference availability selects columns, never model output",
            norms="per-residue norm mean/population std and same five quantiles",
            standardization="reference B median and max(IQR/1.349,1e-8); ref A/B jointly define available nonconstant columns",
            pooling="standardize separately within each length stratum on common reference-available columns; equal proteins",
            energy="biased empirical V-statistic, Euclidean/sqrt(D)",
            mmd="biased MMD^2, Gaussian RBF; bandwidth median positive normalized Euclidean distance in pooled references A/B",
            sliced_wasserstein="64 fixed unit Gaussian projections, PCG64 seed14014014; mean empirical one-dimensional W1",
            bootstrap="500 deterministic protein-unit resamples; same identity indices across independent populations; pooled resampling stratified; reference normalization/bandwidth held fixed",
            uncertainty="conditional on frozen panel/checkpoint/reference preprocessing; no multiplicity correction; exploratory",
            classification="improves/worsens if >=3 of 5 primary model-minus-noop 95% CIs exclude zero in same direction, none in opposite direction; else unresolved",
            gain="1-(Dmodel-Dfloor)/(Dnoop-Dfloor); undefined if denominator<=1e-12; gain bootstrap valid count reported",
            permutation_tolerance=1e-10,
            true_v_tolerance=5e-6,
            clean_protein_thresholds_used=False,
            limitations="low-dimensional invariant descriptor match does not prove full joint distribution, chirality, topology or foldability",
        ),
    )
    assert sha(CP) == CP_SHA
    state = torch.load(CP, map_location="cpu", weights_only=False)
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
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
    certification = json.loads((ROOT / "e014/interim_50pct/checkpoint_identity.json").read_text())["checkpoints"][-1]
    assert certification["sha256"] == CP_SHA and certification["contract"] == contract
    assert state["updates"] == state["position"] == 90413 and state["epoch"] == 1
    assert len(state["exposures"]) == 230132 and int(state["exposures"].sum()) == 115067
    assert state["schedule_hash"] == certification["schedule_hash"]
    model = E014CapacityDenoiser(**contract["capacity_config"]).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 5119069
    before_model = {k: v.clone() for k, v in model.state_dict().items()}
    identity = dict(
        path=str(CP),
        sha256=CP_SHA,
        epoch=1,
        update=90413,
        position=90413,
        exposures=115067,
        contract=contract,
        training_before=heartbeat(),
        gpu_before=gpu_snapshot(),
    )
    atomic_json(OUT / "checkpoint_identity.json", identity)
    del state
    cache = CoordinateCache(ROOT)
    diffusion = CoordinateVPDiffusion(cfg["diffusion_steps"])
    scale = cfg["coordinate_scale_angstrom"]
    model.cuda()
    torch.cuda.reset_peak_memory_stats()

    def forward(clean, mask, timestep, family, sample_id):
        t = torch.tensor([timestep], device="cuda")
        g = torch.Generator(device="cuda").manual_seed(noise_seed(FAMILIES[family], sample_id, timestep))
        return diffusion.make_training_batch(clean, mask, timesteps=t, generator=g)

    maximum = 0.0
    for row in subset[::4]:
        clean, _, mask = coordinate_batch([row], cache, scale)
        for s in TIMES:
            batch = forward(clean, mask, s, "source", row["sample_id"])
            maximum = max(
                maximum,
                reverse_identity(diffusion, clean, batch.coordinate_noise, torch.tensor([s], device="cuda"), mask),
            )
    atomic_json(OUT / "oracle_identity.json", dict(conditions=125, max_abs_error=maximum, tolerance=5e-6, passed=True))
    print(f"true-v identity passed {maximum:.9g}", flush=True)

    def append(rows, row, phase, s, population_label, x):
        z = descriptor(x[0].detach().cpu().numpy())
        rows.append(
            dict(
                sample_id=row["sample_id"],
                length=row["length"],
                length_stratum=row["length_stratum"],
                phase=phase,
                timestep=s,
                population=population_label,
                descriptor=z.tolist(),
            )
        )

    def progress(phase, n):
        status = dict(
            phase=phase,
            identities=n,
            elapsed_seconds=time.time() - start,
            training=heartbeat(),
            gpu=gpu_snapshot(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        atomic_json(OUT / "HEARTBEAT.json", status)
        print(json.dumps(status), flush=True)
        if status["peak_reserved_bytes"] > 2 * 1024**3:
            raise RuntimeError("diagnostic exceeded bounded 2 GiB reserved envelope")

    rows = []
    for i, row in enumerate(panel):
        clean, lengths, mask = coordinate_batch([row], cache, scale)
        for s in TIMES:
            current = forward(clean, mask, s, "source", row["sample_id"]).noisy_coordinates
            t = torch.tensor([s], device="cuda")
            v = model(current, t, lengths, mask)
            previous, _, _ = diffusion.deterministic_reverse_step(current, t, v, mask)
            for label, x in (
                ("model", previous),
                ("noop", current),
                ("ref_a", forward(clean, mask, s - 1, "ref-a", row["sample_id"]).noisy_coordinates),
                ("ref_b", forward(clean, mask, s - 1, "ref-b", row["sample_id"]).noisy_coordinates),
            ):
                append(rows, row, "one_step", s, label, x)
        if (i + 1) % 25 == 0:
            progress("one_step", i + 1)
    save_rows("one_step_descriptors.parquet", rows)
    del rows
    for phase, start_t, endpoints in (
        ("teacher_rollout", 490, (490, 480, 465, 440)),
        ("gaussian_prefix", 499, (499, 490, 480, 465, 440)),
    ):
        rows = []
        for i, row in enumerate(subset):
            clean, lengths, mask = coordinate_batch([row], cache, scale)
            if phase == "teacher_rollout":
                current = forward(clean, mask, 490, "source", row["sample_id"]).noisy_coordinates
            else:
                g = torch.Generator(device="cuda").manual_seed(
                    noise_seed(FAMILIES["gaussian"], f"sample-{i}-N{row['length']}", 499)
                )
                current = centered_coordinate_noise(torch.zeros_like(clean), mask, generator=g)
            initial = current.clone()
            for s in range(start_t, 439, -1):
                if s in endpoints:
                    for label, x in (
                        ("model", current),
                        ("noop", initial),
                        ("ref_a", forward(clean, mask, s, "ref-a", row["sample_id"]).noisy_coordinates),
                        ("ref_b", forward(clean, mask, s, "ref-b", row["sample_id"]).noisy_coordinates),
                    ):
                        append(rows, row, phase, s, label, x)
                if s > 440:
                    t = torch.tensor([s], device="cuda")
                    v = model(current, t, lengths, mask)
                    current, _, _ = diffusion.deterministic_reverse_step(current, t, v, mask)
            if (i + 1) % 5 == 0:
                progress(phase, i + 1)
        save_rows(f"{phase}_descriptors.parquet", rows)
    assert sha(CP) == CP_SHA
    assert all(torch.equal(v.cpu(), before_model[k]) for k, v in model.state_dict().items())
    identity.update(
        training_after=heartbeat(),
        gpu_after=gpu_snapshot(),
        elapsed_seconds=time.time() - start,
        checkpoint_unchanged=True,
        model_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    atomic_json(OUT / "checkpoint_identity.json", identity)
    progress("inference_complete", 100)


if __name__ == "__main__":
    main()
