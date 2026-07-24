"""YAML configuration loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed mapping, or an empty mapping for an empty YAML document.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the top-level YAML value is not a mapping.
    """
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {cfg_path}")
    return data


def require_keys(config: dict[str, Any], keys: list[str], *, context: str) -> None:
    """Validate that required keys are present in a configuration mapping.

    Args:
        config: Configuration mapping.
        keys: Required top-level keys.
        context: Human-readable name included in errors.

    Returns:
        None.
    """
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"{context} config is missing required key(s): {', '.join(missing)}")
