"""Local JSONL logging for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Append metrics to a JSONL file.

    Args:
        path: Log file path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict[str, Any]) -> None:
        """Append one JSON-serializable metric row."""
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
