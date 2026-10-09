"""Physical-batch CUDA smoke and frozen-source/storage audit before training."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import os
import subprocess
import time
from pathlib import Path
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha, d_path
from protein_distance_diffusion.training.e013_losses import coordinate_v_loss


def main():
    cfgpath = Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
    cfg = yaml.safe_load(cfgpath.read_text())
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen = json.loads(Path("/mnt/d/Simone/proteinGen/e013_artifacts/completion_manifest.json").read_text())
    for path, record in frozen["artifact_hashes"].items():
        if sha(path) != record["sha256"]:
            raise ValueError(f"historical E013 artifact changed: {path}")
    for path, digest in frozen["final_source_hashes"].items():
        if path not in ("README.md", "docs/experiment_timeline.md") and sha(path) != digest:
            raise ValueError(f"historical E013 source changed: {path}")
    for path in (
        "src/protein_distance_diffusion/models/equivariant_pair_coordinate_unet.py",
        "src/protein_distance_diffusion/training/coordinate_diffusion.py",
        "src/protein_distance_diffusion/training/e007_coordinate_losses.py",
        "configs/e007_coordinate_generator_v1.yaml",
    ):
        base = subprocess.check_output(["git", "show", f"dd421bde2a6b3a9b5eb549f72b874ae2ca875bac:{path}"])
        if Path(path).read_bytes() != base:
            raise ValueError(f"historical E007 source changed: {path}")
    population = json.loads((ROOT / "population/population.json").read_text())
    train = pq.read_table(
        ROOT / "population/train.parquet", columns=["sample_id", "cluster_id"], use_threads=False
    ).to_pylist()
    clean = pq.read_table(cfg["clean_validation"], columns=["sample_id", "cluster_id"], use_threads=False).to_pylist()
    for extract in (
        lambda r: r["sample_id"],
        lambda r: r["cluster_id"],
        lambda r: r["sample_id"].split("_")[0].lower(),
    ):
        if {extract(r) for r in train} & {extract(r) for r in clean}:
            raise ValueError("TRAIN/clean-validation identity, homology or PDB-entry leakage")
    for key in (
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "MPLCONFIGDIR",
    ):
        d_path(os.environ[key])
    selected = json.loads((ROOT / "capacity_selection.json").read_text())["selected"]
    torch.manual_seed(cfg["seed"])
    model = E014CapacityDenoiser(**selected["config"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.01)
    results = []
    for n, size in zip((64, 128, 256, 384, 500), cfg["batch_sizes"], strict=True):
        x = torch.randn(size, n, 3, device="cuda")
        x -= x.mean(1, keepdim=True)
        mask = torch.ones(size, n, device="cuda", dtype=torch.bool)
        lengths = torch.full((size,), n, device="cuda", dtype=torch.long)
        t = torch.full((size,), 250, device="cuda", dtype=torch.long)
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            y = model(x, t, lengths, mask)
            loss = coordinate_v_loss(y, x, mask)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)
            if not torch.isfinite(y).all() or not torch.isfinite(grad_norm):
                raise ValueError("nonfinite physical-batch CUDA preflight")
        reserved = torch.cuda.max_memory_reserved()
        if reserved > cfg["preflight"]["max_peak_reserved_bytes"]:
            raise ValueError("physical-batch memory comfort failed")
        results.append(
            dict(
                length=n,
                physical_batch_size=size,
                warmed_update_seconds=sum(durations[1:]) / 2,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=reserved,
                finite=True,
            )
        )
    report = dict(
        passed=True,
        selected_parameters=selected["parameters"],
        physical_batches=results,
        e013_artifact_hashes_verified=len(frozen["artifact_hashes"]),
        historical_source_preserved=True,
        train_clean_validation_identity_cluster_pdb_overlap=0,
        train_population_hash=population["train_sample_id_hash"],
        config_hash=sha(cfgpath),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        all_generated_paths_on_d=True,
        cpu_threads=1,
        loader_workers=0,
        runtime_directories={key: os.environ[key] for key in ("TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH")},
    )
    atomic_json(ROOT / "EXECUTION_BOUNDARY_PREFLIGHT.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
