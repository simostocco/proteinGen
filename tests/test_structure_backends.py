"""Structure backend selection and MDAnalysis parser tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from protein_distance_diffusion.data.mdanalysis_parser import StructureParseError, parse_pdb_file
from protein_distance_diffusion.data.preprocess import compute_distance_matrix
from protein_distance_diffusion.data.structure_parser import detect_structure_format, select_backend


def test_backend_auto_selection_and_unsupported_format() -> None:
    """Auto mode sends mmCIF to Gemmi and PDB to MDAnalysis, with clear unsupported errors."""
    assert detect_structure_format("1abc.cif") == "mmcif"
    assert detect_structure_format("1abc.mmcif.gz") == "mmcif"
    assert detect_structure_format("1abc.pdb") == "pdb"
    assert select_backend("1abc.cif", "auto") == "gemmi"
    assert select_backend("1abc.pdb", "auto") == "mdanalysis"
    with pytest.raises(ValueError, match="Unsupported structure format"):
        detect_structure_format("structure.xyz")


@pytest.mark.skipif(importlib.util.find_spec("MDAnalysis") is None, reason="MDAnalysis is not installed")
def test_mdanalysis_ca_selection_ordering_and_distance() -> None:
    """A two-residue PDB fixture preserves order and gives a manually checkable 5 A distance."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue.pdb"
    samples = parse_pdb_file(fixture, min_length=1, max_length=10, chain_id="A")
    sample = samples[0]
    assert sample.sequence == "GA"
    assert sample.residue_ids == ["1", "2"]
    assert sample.ca_coordinates.shape == (2, 3)
    distance = compute_distance_matrix(sample.ca_coordinates)
    assert distance.shape == (2, 2)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert distance.min() >= 0.0
    assert np.isclose(distance[0, 1], 5.0)


@pytest.mark.skipif(importlib.util.find_spec("MDAnalysis") is None, reason="MDAnalysis is not installed")
def test_mdanalysis_missing_ca_failure_is_clear() -> None:
    """Missing C-alpha atoms produce a clear parse failure."""
    fixture = Path(__file__).parent / "fixtures" / "missing_ca.pdb"
    with pytest.raises(StructureParseError, match="missing C-alpha|No accepted"):
        parse_pdb_file(fixture, min_length=1, max_length=10, chain_id="A")
