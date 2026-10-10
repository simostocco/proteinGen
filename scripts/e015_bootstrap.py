"""All E015 entry points import this before any scientific framework."""
# ruff: noqa: E402 -- D cache bootstrap must precede scientific imports.

import os
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
from protein_distance_diffusion.training.e015_storage import DEFAULT_ROOT, configure_storage

ROOT = configure_storage(DEFAULT_ROOT)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["TRITON_CACHE_DIR"] = str(ROOT / "runtime" / "triton")
(ROOT / "runtime" / "triton").mkdir(parents=True, exist_ok=True)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
