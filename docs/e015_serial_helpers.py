# ruff: noqa: E402, E501, F401
"""Read-only E015 absolute audit. CPU now; guarded CUDA generation later.
All outputs/caches remain in the dedicated D: evaluation folder.
No Runtime initialization, training, signaling, or production metadata writes.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

WORK = Path("/mnt/d/Simone/proteinGen/.e015_worktree")
ROOT = Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")
SHA = "2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03"
OUT = ROOT / "evaluations/absolute_generation_u3811_2986d44e_v1"
MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
OUT.mkdir(parents=True, exist_ok=True)
for key in (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
):
    p = OUT / "runtime" / key.lower()
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
tempfile.tempdir = os.environ["TMPDIR"]
if MODE != "generate":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.nice(19)
sys.path.insert(0, str(WORK / "src"))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch, extended_metrics
from protein_distance_diffusion.evaluation.e015_evaluation import prediction
from protein_distance_diffusion.evaluation.marginal_distribution_diagnostic import (
    NAMES,
    Comparison,
    ReferenceStandardizer,
    bandwidth,
    descriptor,
    distances,
    projections,
)
from protein_distance_diffusion.evaluation.reverse_field_diagnostic import paired_state, vector_metrics
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry
from protein_distance_diffusion.training.coordinate_diffusion import centered_coordinate_noise
from protein_distance_diffusion.training.e014_runtime import canonical_hash, noise_seed, sha
from protein_distance_diffusion.training.e015_engine import Engine
from protein_distance_diffusion.training.e015_runtime import Runtime

GRID = [5, 50, 250, 400, 490]
STRATA = ["20-64", "65-128", "129-256", "257-384", "385-500"]
LENGTHS = [48, 96, 192, 320, 448]
CP = ["initialization", "epoch_1_fraction_0.25", "epoch_1_fraction_0.50"]
PRIMARY = [
    "v_mse",
    "aligned_rmsd_angstrom",
    "local_mse_angstrom2",
    "torsion_angular_mae_radians",
    "triple_inversion_fraction",
    "clash_fraction",
]
SCALE = 12.22820347644835
CFG_PATH = WORK / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml"
cfg = yaml.safe_load(CFG_PATH.read_text())
cal = json.loads((ROOT / "calibration/frozen.json").read_text())
rt = object.__new__(Runtime)
rt.config_path = CFG_PATH
rt.cfg = cfg
CONTRACT = rt.contract(cal)
assert CONTRACT == json.loads((ROOT / "preflight/full_certification.json").read_text())["calibration_contract"]
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
for name in ("samples", "plots", "success_gallery", "median_gallery", "failure_gallery", "logs"):
    (OUT / name).mkdir(exist_ok=True)


def clean(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): clean(a) for k, a in v.items()}
    if isinstance(v, (tuple, list, np.ndarray)):
        return [clean(a) for a in v]
    return v


def write(name, v):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + ".partial")
    q.write_text(json.dumps(clean(v), separators=(",", ":"), allow_nan=False) + "\n")
    q.replace(p)


def read(name):
    return json.loads((OUT / name).read_text())


def table(name, rows):
    p = OUT / name
    q = p.with_name(p.name + ".partial")
    t = pa.Table.from_pylist([clean(r) for r in rows]) if rows else pa.table({"status": pa.array([], pa.string())})
    pq.write_table(t, q, compression="zstd")
    assert pq.read_metadata(q).num_rows == len(rows)
    q.replace(p)


def progress(phase, **kw):
    r = dict(phase=phase, time_unix=time.time(), mode=MODE, **kw)
    write("logs/progress.json", r)
    print(json.dumps(r), flush=True)


def model_engine():
    e = Engine(cfg, cal, device="cpu", precision="fp32")
    assert sum(p.numel() for p in e.model.parameters() if p.requires_grad) == 33128139
    return e


def finite(obj):
    if isinstance(obj, torch.Tensor):
        return not obj.is_floating_point() or bool(torch.isfinite(obj).all())
    if isinstance(obj, dict):
        return all(finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(finite(v) for v in obj)
    return not isinstance(obj, float) or math.isfinite(obj)


def pinned_load(path):
    with Path(path).open("rb") as f:
        b = os.fstat(f.fileno())
        h = hashlib.sha256()
        while True:
            block = f.read(8 * 1024**2)
            if not block:
                break
            h.update(block)
        f.seek(0)
        s = torch.load(f, map_location="cpu", weights_only=False)
        a = os.fstat(f.fileno())
    assert (a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_ino, b.st_size, b.st_mtime_ns)
    return s, h.hexdigest()


def verify(s, e):
    assert {k: v for k, v in s["contract"].items() if k != "batching_hash"} == CONTRACT
    assert s["contract"]["batching_hash"] == canonical_hash(json.loads((ROOT / "batch_autotuning.json").read_text()))
    assert s["population_hash"] == cfg["population_hash"] and s["architecture_hash"] == canonical_hash(cfg["model"])
    e.model.load_state_dict(s["model"], strict=True)
    e.ema.load_state_dict(s["ema"], strict=True)
    e.optimizer.load_state_dict(s["optimizer"])
    e.scaler.load_state_dict(s["scaler"])
    ps = list(e.model.parameters())
    ids = [i for g in s["optimizer"]["param_groups"] for i in g["params"]]
    assert len(ids) == len(ps)
    for pid, p in zip(ids, ps, strict=True):
        for k, v in s["optimizer"]["state"].get(pid, {}).items():
            if k in ("exp_avg", "exp_avg_sq"):
                assert tuple(v.shape) == tuple(p.shape)
    assert finite({k: s[k] for k in ("model", "ema", "optimizer", "rng", "engine_rng")})
    pos = s["position"]
    seen = np.asarray(s["exposures"])
    expected = np.full(len(seen), pos["epoch"] - 1, dtype=np.uint8)
    expected[np.asarray(s["epoch_schedule"][: pos["cursor"]], dtype=np.int64)] += 1
    assert np.array_equal(seen, expected) and int(seen.sum()) == pos["exposures"]
    assert len(set(s["epoch_schedule"])) == 230132 and len(seen) == 230132
    assert len(s["engine_rng"]["actual_t_hist"]) == 500
    e.optimizer.state.clear()


def key(namespace, row):
    return hashlib.sha256(f"{namespace}|{row['sample_id']}".encode()).hexdigest()


def intrinsic(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    z = x - x.mean(0)
    eig = np.linalg.eigvalsh(z.T @ z / n)[::-1]
    trace = eig.sum()
    ds = np.linalg.norm(z[:, None] - z[None, :], axis=-1)
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    non = sep > 3
    lr = sep >= 12
    adj = np.linalg.norm(np.diff(z, axis=0), axis=-1)
    b = np.diff(z, axis=0)
    a = -b[:-1]
    c = b[1:]
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(c, axis=-1)
    valid = den > 1e-12
    angles = np.arccos(np.clip((a * c).sum(-1)[valid] / den[valid], -1, 1))
    features, _, _ = local_geometry(torch.tensor(z, dtype=torch.float32))
    f = features.numpy()
    assess = f[:, 29] > 0
    phi = np.arctan2(f[assess, 25], f[assess, 26])
    q = f[assess, 27]
    contact = (ds < 8) & lr
    ci, cj = np.where(np.triu(contact, 1))
    result = dict(
        length=n,
        finite_fraction=float(np.isfinite(x).mean()),
        centroid_norm_angstrom=float(np.linalg.norm(x.mean(0))),
        radius_of_gyration_angstrom=float(np.sqrt(max(trace, 0))),
        end_to_end_angstrom=float(np.linalg.norm(z[-1] - z[0])),
        covariance_eigenvalues_angstrom2=eig.tolist(),
        covariance_trace_angstrom2=float(trace),
        anisotropy_largest_fraction=float(eig[0] / max(trace, 1e-12)),
        adjacent_mean_angstrom=float(adj.mean()),
        adjacent_std_angstrom=float(adj.std()),
        adjacent_min_angstrom=float(adj.min()),
        adjacent_max_angstrom=float(adj.max()),
        adjacent_valid_fraction=float(((adj >= 3.4) & (adj <= 4.2)).mean()),
        clash_count=int(((ds < 3) & non).sum()),
        clash_pair_count=int(non.sum()),
        clash_fraction=float(((ds < 3) & non).sum() / max(non.sum(), 1)),
        long_range_contact_density=float(contact.sum() / max(lr.sum(), 1)),
        long_range_contact_order=float((cj - ci).mean() / n) if len(ci) else None,
        exact_duplicate_coordinate_pairs=int(((ds == 0) & (sep > 0)).sum() // 2),
        adjacent_out_of_frozen_range_count=int(((adj < 3.4) | (adj > 4.2)).sum()),
        torsion_assessable_fraction=float(assess.mean()),
        torsion_assessable_count=int(assess.sum()),
        signed_triple_mean=float(q.mean()) if len(q) else None,
        negative_signed_triple_fraction=float((q < 0).mean()) if len(q) else None,
        bond_angle_mean_radians=float(angles.mean()) if len(angles) else None,
        bond_angle_std_radians=float(angles.std()) if len(angles) else None,
        pseudo_torsion_quantiles_radians=np.quantile(phi, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() if len(phi) else None,
        signed_triple_quantiles=np.quantile(q, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() if len(q) else None,
        pair_distance_quantiles_angstrom=np.quantile(ds[np.triu_indices(n, 1)], [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
    )
    for k in (1, 2, 3):
        v = np.linalg.norm(z[k:] - z[:-k], axis=-1)
        result[f"offset_{k}_distance_quantiles_angstrom"] = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist()
    result["descriptor"] = descriptor(z).tolist()
    return result


def estimate(a, b, strata, paired=True):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    strata = np.asarray(strata)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    strata = strata[valid]
    if not len(a):
        return dict(n=0)
    rng = np.random.default_rng(15015)
    ad = np.zeros(2000)
    bd = np.zeros(2000)
    for st in sorted(set(strata)):
        ix = np.flatnonzero(strata == st)
        draw = rng.choice(ix, (2000, len(ix)), replace=True)
        ad += a[draw].sum(1) / len(a)
        drawb = draw if paired else rng.choice(ix, (2000, len(ix)), replace=True)
        bd += b[drawb].sum(1) / len(b)
    ratio = bd / ad
    return dict(
        n=len(a),
        old_mean=float(a.mean()),
        new_mean=float(b.mean()),
        absolute_difference=float((b - a).mean()),
        relative_difference=float((b.mean() - a.mean()) / a.mean()) if abs(a.mean()) > 1e-12 else None,
        ratio=float(b.mean() / a.mean()) if abs(a.mean()) > 1e-12 else None,
        old_ci95=np.quantile(ad, [0.025, 0.975]).tolist(),
        new_ci95=np.quantile(bd, [0.025, 0.975]).tolist(),
        difference_ci95=np.quantile(bd - ad, [0.025, 0.975]).tolist(),
        ratio_ci95=np.quantile(ratio, [0.025, 0.975]).tolist() if np.isfinite(ratio).all() else None,
        bootstrap_replicates=2000,
        paired=paired,
    )


def groupmean(rows):
    result = {}
    keys = sorted(
        {
            k
            for r in rows
            for k, v in r.items()
            if isinstance(v, (int, float, bool)) and k not in ("length", "noise_seed", "timestep")
        }
    )
    for k in keys:
        v = [
            float(r[k])
            for r in rows
            if r.get(k) is not None and isinstance(r[k], (int, float, bool)) and math.isfinite(float(r[k]))
        ]
        if v:
            result[k] = dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v), std=float(np.std(v)))
    return result


def sample_images(meta, stat, trajectory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    x = trajectory[0]
    n = len(x)
    directory = Path(meta["path"])
    segments = np.stack([x[:-1], x[1:]], 1)
    fig = plt.figure(figsize=(10, 4.5), constrained_layout=True)
    caption = f"{meta['weights']} N{n}, seed{meta['seed']} | CA only | adjacent {stat['adjacent_mean_angstrom']:.2f}±{stat['adjacent_std_angstrom']:.2f}Å, clash {stat['clash_fraction']:.3f}, Rg{stat['radius_of_gyration_angstrom']:.2f}Å"
    fig.suptitle(caption, fontsize=9)
    for i, (az, el) in enumerate([(35, 25), (125, 20)], 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        line = Line3DCollection(segments, cmap="viridis", norm=colors.Normalize(0, n - 1), linewidth=1.4)
        line.set_array(np.arange(n - 1))
        ax.add_collection3d(line)
        ax.scatter(*x[0], color="purple", s=20, label="N")
        ax.scatter(*x[-1], color="gold", s=20, label="C")
        limits = (x.min(0), x.max(0))
        center = (limits[0] + limits[1]) / 2
        span = max((limits[1] - limits[0]).max(), 1) / 2
        ax.set(
            xlim=(center[0] - span, center[0] + span),
            ylim=(center[1] - span, center[1] + span),
            zlim=(center[2] - span, center[2] + span),
            xlabel="x Å",
            ylabel="y Å",
            zlabel="z Å",
        )
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(el, az)
    fig.savefig(directory / "trace_two_views.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    img = axes[0].imshow(d, cmap="magma", origin="lower")
    fig.colorbar(img, ax=axes[0], label="CA distance Å")
    axes[0].set(title="Distance map", xlabel="Residue", ylabel="Residue")
    for k in (1, 2, 3):
        axes[1].plot(np.arange(n - k), np.linalg.norm(x[k:] - x[:-k], axis=-1), label=f"i,i+{k}", linewidth=0.8)
    axes[1].axhspan(3.4, 4.2, alpha=0.15, color="gray", label="Adjacent valid range")
    axes[1].set(title="Local distance profiles", xlabel="Residue", ylabel="Å")
    axes[1].legend(fontsize=7)
    fig.suptitle(caption, fontsize=9)
    fig.savefig(directory / "distance_map_profiles.png", dpi=160)
    plt.close(fig)
    ts = sorted(trajectory, reverse=True)
    stats = [intrinsic(trajectory[t]) for t in ts]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ["radius_of_gyration_angstrom", "covariance_trace_angstrom2", "adjacent_mean_angstrom"],
        ["Rg Å", "Covariance trace Å²", "Adjacent distance Å"],
        strict=True,
    ):
        ax.plot(ts, [s[key] for s in stats], marker="o", linewidth=1)
        ax.invert_xaxis()
        ax.set(xlabel="Reverse timestep", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle(caption, fontsize=9)
    fig.savefig(directory / "trajectory_geometry.png", dpi=160)
    plt.close(fig)
    # Predeclared first seed per length/weight: sparse montage, not selected visually.
    if meta["seed"] == 15015000:
        fig = plt.figure(figsize=(12, 7), constrained_layout=True)
        for i, t in enumerate(ts, 1):
            a = trajectory[t]
            ax = fig.add_subplot(3, 4, i, projection="3d")
            ax.plot(*a.T, linewidth=0.6)
            ax.set_title(f"t={t}", fontsize=9)
            ax.set_box_aspect([1, 1, 1])
            ax.set_axis_off()
            ax.view_init(25, 35)
        fig.suptitle(f"Predeclared trajectory montage: {meta['sample_key']}, C-alpha only", fontsize=10)
        fig.savefig(directory / "trajectory_montage.png", dpi=160)
        plt.close(fig)


def score_vector(r):
    return np.array(
        [
            r["adjacent_mean_angstrom"],
            r["adjacent_std_angstrom"],
            r["clash_fraction"],
            r["radius_of_gyration_angstrom"],
            r["signed_triple_mean"],
            *r["covariance_eigenvalues_angstrom2"],
        ],
        dtype=float,
    )


def aligned_rmsd(a, b):
    assert len(a) == len(b)
    a = a - a.mean(0)
    b = b - b.mean(0)
    u, _, vh = np.linalg.svd(a.T @ b)
    fix = np.diag([1, 1, np.linalg.det(u @ vh)])
    return float(np.sqrt(np.mean(np.sum((a @ (u @ fix @ vh) - b) ** 2, axis=-1))))


def generation_plots(rows, distribution, diversity):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    primary = [r for r in rows if r["primary"]]
    for ax, key, label in [
        (axes[0, 0], "adjacent_valid_fraction", "Adjacent valid fraction"),
        (axes[0, 1], "clash_fraction", "Clash fraction"),
        (axes[1, 0], "radius_of_gyration_angstrom", "Rg Å"),
    ]:
        ax.boxplot([[r[key] for r in primary if r["requested_length"] == n] for n in LENGTHS], tick_labels=LENGTHS)
        ax.set(xlabel="Length", ylabel=label)
        ax.grid(alpha=0.2)
    axes[1, 1].plot(LENGTHS, [r["median_kabsch_rmsd_angstrom"] for r in diversity], marker="o")
    axes[1, 1].set(xlabel="Length", ylabel="Median generated-pair Kabsch RMSD Å")
    axes[1, 1].grid(alpha=0.2)
    fig.savefig(OUT / "plots/generation_quality_diversity.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for key, label in [("generated_vs_real", "Generated vs real"), ("real_vs_real_floor", "Real vs real")]:
        ax.plot(
            LENGTHS,
            [next(q[key] for q in d["metrics"] if q["metric"] == "energy") for d in distribution],
            marker="o",
            label=label,
        )
    ax.set(xlabel="Length", ylabel="Standardized descriptor energy distance", title="Descriptor population comparison")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "plots/generation_descriptor_floor.png", dpi=160)
    plt.close(fig)
