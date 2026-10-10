"""D-only storage paths; free-space/reserve measurements are telemetry only."""

import os
import shutil
from pathlib import Path

CACHE_KEYS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "HF_HOME",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
)
DEFAULT_ROOT = Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")


def generated_path(root, path):
    root = Path(root).resolve()
    path = Path(path).resolve()
    if not root.is_relative_to(Path("/mnt/d")) or not path.is_relative_to(root):
        raise ValueError(f"Generated path is outside the declared D: root: {path}")
    return path


def configure_storage(root):
    root = generated_path(root, root)
    root.mkdir(parents=True, exist_ok=True)
    for key in CACHE_KEYS:
        p = generated_path(root, root / "runtime" / key.lower())
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    # Set before importing tempfile, torch or plotting libraries.
    import tempfile

    tempfile.tempdir = os.environ["TMPDIR"]
    return root


def disk_budget(root, e012_reserve_bytes, planned_peak_bytes, safety_reserve_bytes=20 * 1024**3):
    root = generated_path(root, root)
    free = shutil.disk_usage(root).free
    reserve = max(e012_reserve_bytes, safety_reserve_bytes)
    budget = dict(
        free_bytes=free,
        e012_reserve_bytes=e012_reserve_bytes,
        reserve_bytes=reserve,
        planned_peak_bytes=planned_peak_bytes,
        remaining_after_peak_bytes=free - planned_peak_bytes,
        passed=True,
        disk_policy="telemetry_only_owner_amendment",
        reserve_enforced=False,
        would_cross_historical_reserve=free - planned_peak_bytes < reserve,
    )
    return budget
