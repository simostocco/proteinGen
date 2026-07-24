"""Gemmi mmCIF parser regression tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from protein_distance_diffusion.data.mmcif_parser import parse_mmcif_file


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_missing_optional_metadata_does_not_crash() -> None:
    """Missing `_exptl.method` metadata is optional and must not crash Gemmi parsing."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_missing_metadata.cif"
    samples = parse_mmcif_file(fixture, min_length=None, max_length=128)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sequence == "GA"
    assert sample.metadata["experimental_method"] is None
    assert sample.ca_coordinates.shape == (2, 3)
    assert np.isclose(np.linalg.norm(sample.ca_coordinates[0] - sample.ca_coordinates[1]), 5.0)


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_min_length_none_disables_lower_bound() -> None:
    """`min_length=None` accepts short fixtures while a positive integer lower bound filters them."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_missing_metadata.cif"
    assert len(parse_mmcif_file(fixture, min_length=None, max_length=128)) == 1
    assert parse_mmcif_file(fixture, min_length=3, max_length=128) == []
    with pytest.raises(ValueError, match="min_length"):
        parse_mmcif_file(fixture, min_length=0, max_length=128)


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_xray_metadata_and_resolution_filters() -> None:
    """X-ray method and resolution are extracted and method-specific thresholds apply."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_xray.cif"
    samples = parse_mmcif_file(
        fixture,
        min_length=None,
        allowed_methods=["X-RAY DIFFRACTION"],
        max_xray_resolution_angstrom=2.0,
    )
    assert len(samples) == 1
    assert samples[0].metadata["experimental_method"] == "X-RAY DIFFRACTION"
    assert samples[0].metadata["resolution_angstrom"] == 1.7
    assert (
        parse_mmcif_file(
            fixture,
            min_length=None,
            allowed_methods=["X-RAY DIFFRACTION"],
            max_xray_resolution_angstrom=1.0,
        )
        == []
    )


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_mmcif_nmr_rejection_and_optional_method_filter() -> None:
    """NMR is rejected by an exact allowed-method list and accepted when filtering is disabled."""
    fixture = Path(__file__).parent / "fixtures" / "two_residue_nmr.cif"
    assert (
        parse_mmcif_file(
            fixture,
            min_length=None,
            allowed_methods=["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"],
        )
        == []
    )
    samples = parse_mmcif_file(fixture, min_length=None, allowed_methods=None)
    assert len(samples) == 1
    assert samples[0].metadata["experimental_method"] == "SOLUTION NMR"
    assert samples[0].metadata["resolution_angstrom"] is None
