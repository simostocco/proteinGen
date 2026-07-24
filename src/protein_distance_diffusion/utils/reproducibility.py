"""Reproducibility helpers for seeded research runs."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Integer seed.
        deterministic_torch: Whether to request deterministic PyTorch algorithms.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed.

    Args:
        worker_id: Worker index supplied by PyTorch.

    Returns:
        None.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class RuntimeVersions:
    """Serializable runtime version metadata."""

    python: str
    torch: str
    cuda: str | None
    git_commit: str | None


def collect_runtime_versions(repo: str | Path = ".") -> RuntimeVersions:
    """Collect lightweight runtime metadata for checkpoints.

    Args:
        repo: Repository path used to query the git commit.

    Returns:
        RuntimeVersions with Python, PyTorch, CUDA, and git commit values.
    """
    commit: str | None = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo),
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = None
    return RuntimeVersions(
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        git_commit=commit,
    )
