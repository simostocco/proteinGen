"""Frozen read-only comparison contract; configure writable D caches first."""

# ruff: noqa: I001, E501, E402
import os
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "e014_artifacts/e007_vs_e014_matched_v1"
assert str(OUT.resolve()).startswith("/mnt/d/"), "Require a D:-resident workspace before creating outputs"
import e014_bootstrap

ROOT = e014_bootstrap.ROOT
E014 = OUT.parent / "e014/post_reboot_full_evaluation_v1"
HIST = Path("/mnt/d/Simone/proteinGen/outputs/e007_coordinate_real_continuation_to_10000_v1")
OLD = ROOT / "e014/high_noise_75pct_v1"
OUT.mkdir(parents=True, exist_ok=True)
for key, suffix in {
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache",
    "TORCH_HOME": "cache/torch",
    "CUDA_CACHE_PATH": "cache/cuda",
    "TRITON_CACHE_DIR": "cache/triton",
    "MPLCONFIGDIR": "cache/matplotlib",
}.items():
    path = OUT / "runtime" / suffix
    path.mkdir(parents=True, exist_ok=True)
    assert str(path.resolve()).startswith("/mnt/d/") and os.access(path, os.W_OK)
    os.environ[key] = str(path)
tempfile.tempdir = os.environ["TMPDIR"]
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, noise_seed, sha
from protein_distance_diffusion.evaluation.high_noise_resolution import GRID, HIGH, MARGINAL_GRID, STRATA, gaussian_seed
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import select_rollout_subset
from protein_distance_diffusion.models.e014_e007_control import E007ScaleupControl

CORE = (5, 50, 250, 400, 490)
ENDPOINTS = (490, 480, 470, 460, 450, 440, 420, 400, 350)
E007_LABELS = ("e007_step_10000", "e007_step_09000")
E014_LABELS = ("epoch_2_fraction_1.00", "recovery_0")
PANEL_SHA = "3a36bccfee801b618674594762ed2e6e394220ca21db94260ccb5becf7328dcf"
SUBSET_SHA = "c3e94b415bd7a13269bbb7f3a26372764cbba33b91b16589c7f6906a563020a0"
NEWEST_SHA = "935337d559f87b079a0bc9bb8bce6ecf90e971840d59c4b6163f036697c0f4e7"
SCALE = 12.22820347644835
PRIMARY = (
    "aggregate_v_mse",
    "aggregate_local_mse",
    "gaussian_energy_400",
    "gaussian_energy_350",
    "torsion_angular_400",
    "inversion_rate_400",
)


def write(name, value):
    path = (OUT / name).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError("comparison outputs must stay within the D evaluation root")
    atomic_json(path, value)


def table(name, rows):
    path = (OUT / name).resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError("comparison outputs must stay within the D evaluation root")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    partial = path.with_name(path.name + ".partial")
    pq.write_table(pa.Table.from_pylist([{k: r.get(k) for k in keys} for r in rows]), partial, compression="zstd")
    assert pq.read_metadata(partial).num_rows == len(rows)
    partial.replace(path)


def population():
    rows = pq.read_table(ROOT / "population/development.parquet", use_threads=False).to_pylist()
    assert len(rows) == len({r["sample_id"] for r in rows}) == 500
    assert canonical_hash([r["sample_id"] for r in rows]) == PANEL_SHA
    assert all(sum(r["length_stratum"] == s for r in rows) == 100 for s in STRATA)
    subset = select_rollout_subset(rows, STRATA)
    assert canonical_hash([r["sample_id"] for r in subset]) == SUBSET_SHA
    return rows, subset


def load_e007(label):
    step = int(label.rsplit("_", 1)[1])
    path = HIST / f"checkpoints/step-{step:05d}.pt"
    config = yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = E007ScaleupControl(**config).eval()
    model.core.load_state_dict(state["model"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 7586505
    return model, state, path


def bootstrap_indices(strata, replicates=5000, seed=14014007):
    strata = np.asarray(strata)
    rng = np.random.Generator(np.random.PCG64(seed))
    groups = [np.flatnonzero(strata == s) for s in STRATA if np.any(strata == s)]
    return np.concatenate([rng.choice(g, size=(replicates, len(g)), replace=True) for g in groups], axis=1)


def paired_estimate(a, b, strata, replicates=5000, seed=14014007):
    """Equal protein means; delta is E014 minus E007. Bootstrap jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return dict(assessed=0, e007=None, e014=None, difference=None, ci95=None)
    indices = bootstrap_indices(np.asarray(strata)[valid], replicates, seed)
    aa, bb = a[valid][indices].mean(1), b[valid][indices].mean(1)
    delta = bb - aa
    relative = bb[aa != 0] / aa[aa != 0] - 1
    return dict(
        assessed=int(valid.sum()),
        population_size=len(a),
        e007=float(a[valid].mean()),
        e014=float(b[valid].mean()),
        difference=float((b[valid] - a[valid]).mean()),
        relative_difference=float(b[valid].mean() / a[valid].mean() - 1) if a[valid].mean() else None,
        ci95=np.quantile(delta, [0.025, 0.975]).tolist(),
        relative_ci95=np.quantile(relative, [0.025, 0.975]).tolist() if len(relative) else None,
        relative_effect_unstable=bool(np.any(aa <= 0)),
        e007_ci95=np.quantile(aa, [0.025, 0.975]).tolist(),
        e014_ci95=np.quantile(bb, [0.025, 0.975]).tolist(),
        ci_familywise=np.quantile(delta, [0.05 / (2 * len(PRIMARY)), 1 - 0.05 / (2 * len(PRIMARY))]).tolist(),
        draws=delta.tolist(),
        bootstrap_replicates=replicates,
    )


__all__ = [
    "ROOT",
    "OUT",
    "OLD",
    "E014",
    "HIST",
    "Path",
    "json",
    "np",
    "pq",
    "torch",
    "yaml",
    "E007ScaleupControl",
    "GRID",
    "HIGH",
    "MARGINAL_GRID",
    "STRATA",
    "CORE",
    "ENDPOINTS",
    "E007_LABELS",
    "E014_LABELS",
    "PANEL_SHA",
    "SUBSET_SHA",
    "NEWEST_SHA",
    "SCALE",
    "PRIMARY",
    "noise_seed",
    "sha",
    "canonical_hash",
    "gaussian_seed",
    "write",
    "table",
    "population",
    "load_e007",
    "bootstrap_indices",
    "paired_estimate",
]
