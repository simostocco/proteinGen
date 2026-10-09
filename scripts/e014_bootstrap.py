"""Route every library runtime cache/temp directory to D before imports."""

import os
from pathlib import Path

ROOT = Path("/mnt/d/Simone/proteinGen/e014_artifacts")
ROOT.mkdir(parents=True, exist_ok=True)
for key, subdir in [
    ("TMPDIR", "tmp"),
    ("TMP", "tmp"),
    ("TEMP", "tmp"),
    ("XDG_CACHE_HOME", "cache"),
    ("TORCH_HOME", "cache/torch"),
    ("CUDA_CACHE_PATH", "cache/cuda"),
    ("TRITON_CACHE_DIR", "cache/triton"),
    ("MPLCONFIGDIR", "cache/matplotlib"),
]:
    path = ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
