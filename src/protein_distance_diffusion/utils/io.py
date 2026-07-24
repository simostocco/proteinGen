"""I/O helpers with explicit atomic writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write text atomically by first writing a sibling temporary file.

    Args:
        path: Destination path.
        text: Text content.

    Returns:
        None.
    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(dst)


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable indentation.

    Args:
        path: Destination path.
        data: JSON-serializable object.

    Returns:
        None.
    """
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")
