"""Configuration loading and validation for sequence baseline scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def require_keys(config: dict[str, Any], keys: list[str], *, context: str) -> None:
    """Validate required keys."""
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required {context} config keys: {missing}")
