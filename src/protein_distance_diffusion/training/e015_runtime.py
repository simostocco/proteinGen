"""E015-only verified inputs, strict storage, RNG and durable contract."""

import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, sha
from protein_distance_diffusion.training.e015_storage import disk_budget, generated_path


class Runtime:
    def __init__(self, config_path, verify_cache=True):
        self.config_path = Path(config_path).resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        if self.cfg["coordinate_scale_angstrom"] != 12.22820347644835 or self.cfg["diffusion_steps"] != 500:
            raise ValueError("Frozen diffusion/normalization changed")
        self.root = Path(self.cfg["output_dir"]).resolve()
        generated_path(self.root, self.root)
        torch.set_num_threads(self.cfg["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = Path(self.cfg["population_root"])
        metadata = json.loads((data / "population/population.json").read_text())
        for path, digest in metadata["manifest_hashes"].items():
            # Prospective file hash only: no holdout rows/coordinates are read.
            if sha(path) != digest:
                raise ValueError("Frozen manifest checksum mismatch")
        if sha(self.cfg["normalization_artifact"]) != self.cfg["normalization_sha256"]:
            raise ValueError("Normalization checksum mismatch")
        self.train = pq.read_table(data / "population/train.parquet", use_threads=False).to_pylist()
        self.dev = pq.read_table(data / "population/development.parquet", use_threads=False).to_pylist()
        if (
            len(self.train) != 230132
            or canonical_hash(sorted(r["sample_id"] for r in self.train)) != self.cfg["population_hash"]
        ):
            raise ValueError("TRAIN population mismatch")
        if len(self.dev) != 500 or canonical_hash([r["sample_id"] for r in self.dev]) != self.cfg["development_hash"]:
            raise ValueError("Development panel mismatch")
        self.cache = CoordinateCache(data, verify_payload=verify_cache)
        self.disk_check()
        self.write(
            "INPUT_VERIFICATION.json",
            dict(
                population_hash=self.cfg["population_hash"],
                train_count=len(self.train),
                development_hash=self.cfg["development_hash"],
                normalization_sha256=self.cfg["normalization_sha256"],
                prospective_opened=False,
            ),
        )

    def path(self, name):
        p = generated_path(self.root, self.root / name)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self, name, value):
        atomic_json(self.path(name), value)

    def disk_check(self, additional_bytes=None):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        resources = self.cfg["resources"]
        # Owner amendment: space/budget/reserve are informational only.
        protected_reserve = resources["e012_original_reserve_bytes"]
        return disk_budget(
            self.root,
            protected_reserve,
            max(0, resources["peak_artifact_budget_bytes"] - used) if additional_bytes is None else additional_bytes,
        )

    def contract(self, calibration):
        worktree = self.config_path.parent.parent
        files = sorted(
            list((worktree / "src/protein_distance_diffusion").rglob("*.py"))
            + list((worktree / "scripts").glob("*e015*.py"))
        )
        return dict(
            config_hash=sha(self.config_path),
            torch_version=str(torch.__version__),
            normalization_hash=self.cfg["normalization_sha256"],
            population_hash=self.cfg["population_hash"],
            source_hashes={str(p.relative_to(worktree)): sha(p) for p in files},
            calibration_hash=canonical_hash(calibration),
            diffusion_steps=500,
            scale=self.cfg["coordinate_scale_angstrom"],
        )


def e007_active(cfg):
    root = Path(cfg["resources"]["e007_root"])
    # Recorded state is secondary to live process ownership. Never signal E007.
    for file in ("STATE.json", "SUPERVISOR.json"):
        path = root / file
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        pid = state.get("pid")
        if not pid:
            continue
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and b"e007" in command.read_bytes().lower():
            if state.get("training_active", False) or str(state.get("phase", "")).startswith("train"):
                return True
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().lower()
        except (OSError, PermissionError):
            continue
        if b"python" in command and b"run_e007" in command and b"matched" in command:
            return True
    return False


def dedicated_rng(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def status(phase, **kwargs):
    return dict(phase=phase, pid=os.getpid(), timestamp_unix=time.time(), **kwargs)


class GPULease:
    """Prevent independent E015 GPU entry points from overlapping."""

    def __init__(self, runtime):
        import fcntl

        self.stream = open(runtime.path("GPU_WORKER.lock"), "a")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise RuntimeError("Another E015 GPU worker owns the device") from None
        if e007_active(runtime.cfg):
            self.stream.close()
            raise RuntimeError("Active E007 trainer owns the GPU")
