"""Repeat the historical Gaussian-prefix outlier without changing tolerance."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT

import json
import numpy as np
import pyarrow.parquet as pq
import torch

from diagnose_e014_high_noise import population
from evaluate_e014_post_reboot import OLD, OUT
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.evaluation.high_noise_resolution import gaussian_seed, STRATA
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    descriptor,
    NAMES,
    ReferenceStandardizer,
    Comparison,
    bandwidth,
    projections,
)
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion, centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    atomic_json(
        OUT / "REPLAY_POLICY.json",
        dict(
            sample_id="6nof_A",
            timestep=440,
            checkpoint="epoch_1_fraction_0.50",
            repetitions=10,
            tolerance=5e-6,
            deterministic_enforcement_probe=True,
            historical_outlier_preserved=True,
        ),
    )
    cp = next(
        r
        for r in json.loads((OUT / "checkpoint_inventory.json").read_text())["checkpoints"]
        if r["label"] == "epoch_1_fraction_0.50"
    )
    assert sha(cp["path"]) == cp["sha256"]
    state = torch.load(cp["path"], map_location="cpu", weights_only=False)
    m = E014CapacityDenoiser(**cp["contract"]["capacity_config"]).eval().requires_grad_(False)
    m.load_state_dict(state["model"], strict=True)
    frozen = {k: v.clone() for k, v in m.state_dict().items()}
    del state
    m.cuda()
    d = CoordinateVPDiffusion(500)
    _, _, subset = population()
    i, row = next((i, r) for i, r in enumerate(subset) if r["sample_id"] == "6nof_A")
    clean, lengths, mask = coordinate_batch([row], CoordinateCache(ROOT), 12.22820347644835)
    seed = gaussian_seed(i, row["length"], noise_seed)
    coords, descs, initial = [], [], []
    for repeat in range(10):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = centered_coordinate_noise(torch.zeros_like(clean), mask, generator=g)
        initial.append(x.cpu().numpy())
        for t in range(499, 440, -1):
            ts = torch.tensor([t], device="cuda")
            x, _, _ = d.deterministic_reverse_step(x, ts, m(x, ts, lengths, mask), mask)
        coords.append(x.cpu().numpy())
        descs.append(descriptor(coords[-1][0]))
        print(f"repeat {repeat + 1}/10 complete", flush=True)
    try:
        torch.use_deterministic_algorithms(True)
        m(clean, torch.tensor([440], device="cuda"), lengths, mask)
        deterministic = dict(supported=True, inference="one forward only, not a trajectory certificate")
    except RuntimeError as exc:
        deterministic = dict(supported=False, reason=str(exc))
    finally:
        torch.use_deterministic_algorithms(False)
    assert all(torch.equal(v.cpu(), frozen[k]) for k, v in m.state_dict().items())
    assert sha(cp["path"]) == cp["sha256"]
    historical = pq.read_table(
        ROOT / "e014/marginal_reverse_50pct_v1/gaussian_prefix_descriptors.parquet", use_threads=False
    ).to_pylist()
    collected = pq.read_table(OLD / "epoch_1_fraction_0.50.gaussian_descriptors.parquet", use_threads=False).to_pylist()

    def match(rows):
        return np.array(
            next(
                r["descriptor"]
                for r in rows
                if r["sample_id"] == "6nof_A" and r["population"] == "model" and r["timestep"] == 440
            )
        )

    old, outlier = match(historical), match(collected)
    # Quantify population metric sensitivity to replacing only the preserved
    # outlier descriptor by each replay. This does not modify historical rows.
    raw_by_stratum, norms = {}, {}
    for stratum in STRATA:
        raw = {
            pop: np.array(
                [
                    r["descriptor"]
                    for r in sorted(collected, key=lambda r: r["sample_id"])
                    if r["timestep"] == 440 and r["population"] == pop and r["length_stratum"] == stratum
                ]
            )
            for pop in ("ref_a", "ref_b", "noop", "model")
        }
        raw_by_stratum[stratum] = raw
        norms[stratum] = ReferenceStandardizer(raw["ref_a"], raw["ref_b"])
    columns = sorted(set.intersection(*(set(n.columns) for n in norms.values())))
    normalized = {
        pop: np.concatenate(
            [
                ReferenceStandardizer(raw_by_stratum[s]["ref_a"], raw_by_stratum[s]["ref_b"], columns).transform(
                    raw_by_stratum[s][pop]
                )
                for s in STRATA
            ]
        )
        for pop in ("ref_a", "ref_b", "noop", "model")
    }
    raw = {pop: np.concatenate([raw_by_stratum[s][pop] for s in STRATA]) for pop in normalized}
    h = bandwidth(normalized["ref_a"], normalized["ref_b"])
    proj = projections(len(columns))
    spectrum = [
        i for i, c in enumerate(columns) if NAMES[c] in ("eig1", "eig2", "eig3", "trace", "anisotropy1", "anisotropy3")
    ]

    def metrics(model_raw):
        pieces, offset = [], 0
        for s in STRATA:
            pieces.append(
                ReferenceStandardizer(raw_by_stratum[s]["ref_a"], raw_by_stratum[s]["ref_b"], columns).transform(
                    model_raw[offset : offset + 20]
                )
            )
            offset += 20
        return Comparison(
            np.concatenate(pieces), normalized["ref_b"], model_raw, raw["ref_b"], h, proj, spectrum
        ).metrics()

    ordered = [
        r
        for s in STRATA
        for r in sorted(collected, key=lambda r: r["sample_id"])
        if r["timestep"] == 440 and r["population"] == "model" and r["length_stratum"] == s
    ]
    ix = next(i for i, r in enumerate(ordered) if r["sample_id"] == "6nof_A")
    base = metrics(raw["model"])
    sensitivity = []
    for z in descs:
        modified = raw["model"].copy()
        modified[ix] = z
        sensitivity.append({k: v - base[k] for k, v in metrics(modified).items()})
    record = dict(
        sample_id=row["sample_id"],
        checkpoint_sha256=cp["sha256"],
        repetitions=10,
        seed=seed,
        tolerance_unchanged=5e-6,
        initial_coordinates_bitwise_identical=all(np.array_equal(initial[0], x) for x in initial),
        pairwise_coordinate_max_absolute=max(float(np.abs(a - b).max()) for a in coords for b in coords),
        pairwise_descriptor_max_absolute=max(float(np.nanmax(np.abs(a - b))) for a in descs for b in descs),
        old_replay_errors=[float(np.nanmax(np.abs(z - old))) for z in descs],
        outlier_replay_errors=[float(np.nanmax(np.abs(z - outlier))) for z in descs],
        historical_old_vs_outlier=float(np.nanmax(np.abs(old - outlier))),
        largest_difference_descriptor=NAMES[int(np.nanargmax(np.abs(old - outlier)))],
        observed_bitwise_repeatable=all(np.array_equal(coords[0], z) for z in coords),
        deterministic_algorithms_probe=deterministic,
        population_metric_absolute_sensitivity_envelope={k: max(abs(r[k]) for r in sensitivity) for k in base},
        exact_historical_replay_certified=False,
        model_checkpoint_unchanged=True,
        cause="Within-run variation is measured separately. The historical outlier's exact cause cannot be proved from saved descriptors alone.",
    )
    np.savez_compressed(
        OUT / "replay_coordinates.npz",
        coordinates=np.array(coords),
        descriptors=np.array(descs),
        initial=np.array(initial),
    )
    atomic_json(OUT / "replay_numerics.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
