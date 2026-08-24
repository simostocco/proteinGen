"""Test fixtures for the distance diffusion package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_manifest(tmp_path: Path) -> Path:
    """Create a tiny processed manifest with three synthetic proteins."""
    from protein_distance_diffusion.data.preprocess import ProteinSample, save_processed_sample

    rows = []
    samples = [
        ProteinSample(
            sample_id="s1",
            pdb_id="AAAA",
            chain_id="A",
            sequence="ACDEFG",
            residue_ids=[str(i) for i in range(1, 7)],
            ca_coordinates=np.stack([np.arange(6), np.zeros(6), np.zeros(6)], axis=1).astype(np.float32),
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.0},
        ),
        ProteinSample(
            sample_id="s2",
            pdb_id="BBBB",
            chain_id="A",
            sequence="ACDEFG",
            residue_ids=[str(i) for i in range(1, 7)],
            ca_coordinates=np.stack([np.arange(6), np.ones(6), np.zeros(6)], axis=1).astype(np.float32),
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 2.0},
        ),
        ProteinSample(
            sample_id="s3",
            pdb_id="CCCC",
            chain_id="A",
            sequence="HIKLMNPQ",
            residue_ids=[str(i) for i in range(1, 9)],
            ca_coordinates=np.stack([np.arange(8), np.zeros(8), np.ones(8)], axis=1).astype(np.float32),
            metadata={"experimental_method": "synthetic", "resolution_angstrom": 1.5},
        ),
    ]
    for sample in samples:
        rows.append(save_processed_sample(sample, tmp_path / "samples"))
    path = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
