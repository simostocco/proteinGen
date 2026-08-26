"""PyTorch dataset for processed C-alpha distance maps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from protein_distance_diffusion.data.preprocess import load_manifest


class DistanceMapDataset(Dataset):
    """Load variable-length processed distance matrices from a manifest."""

    def __init__(self, manifest_path: str | Path, normalization: dict[str, Any] | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.frame = load_manifest(self.manifest_path).reset_index(drop=True)
        self.normalization = normalization or {"mode": "none"}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[int(index)]
        data = np.load(row["path"], allow_pickle=False)
        matrix = np.asarray(data["distance_matrix"], dtype=np.float32)
        length = int(row.get("length", matrix.shape[0]))
        if self.normalization.get("mode") == "scale":
            matrix = matrix / np.float32(self.normalization["scale"])
        metadata = {}
        if "metadata" in data:
            metadata = json.loads(str(data["metadata"]))
        return {
            "sample_id": str(row["sample_id"]),
            "pdb_id": str(row.get("pdb_id", data["pdb_id"] if "pdb_id" in data else "")),
            "chain_id": str(row.get("chain_id", data["chain_id"] if "chain_id" in data else "")),
            "sequence": str(row.get("sequence", data["sequence"] if "sequence" in data else "")),
            "length": length,
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "distance_matrix": torch.as_tensor(matrix, dtype=torch.float32),
            "metadata": metadata,
        }
