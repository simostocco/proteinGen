from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from protein_distance_diffusion.data.e007_coordinate_dataset import (
    collate_e007_coordinates,
    global_rms_coordinate_radius,
    stable_center_valid_coordinates,
    validate_coordinate_row,
)


def _row(length: int, sample_id: str = "sample") -> dict:
    coordinates = np.stack((np.arange(length) * 3.8, np.zeros(length), np.zeros(length)), axis=1)
    return {
        "sample_id": sample_id,
        "split": "train",
        "sequence": "A" * length,
        "ca_coordinates": coordinates.tolist(),
        "ca_mask": [True] * length,
        "chain_continuity_mask": [True] * (length - 1),
        "chain_break_mask": [False] * (length - 1),
        "source_sha256": "a" * 64,
        "npz_sha256": "b" * 64,
        "schema_version": "e006_rich_geometry_sidecar_v2",
    }


def test_coordinate_projection_alignment_and_no_tokens() -> None:
    projected = validate_coordinate_row(_row(5), split="train")
    assert projected["coordinates"].shape == (5, 3)
    assert projected["accepted_contiguous_single_chain"] is True
    assert "sequence" not in projected
    assert "token_ids" not in projected


@pytest.mark.parametrize("mutation", ["length", "nonfinite", "continuity", "split"])
def test_malformed_coordinate_rows_are_rejected(mutation: str) -> None:
    row = _row(5)
    if mutation == "length":
        row["ca_coordinates"].pop()
    elif mutation == "nonfinite":
        row["ca_coordinates"][2][0] = float("nan")
    elif mutation == "continuity":
        row["chain_break_mask"][0] = True
    else:
        row["split"] = "validation"
    with pytest.raises(ValueError):
        validate_coordinate_row(row, split="train")


def test_collation_centers_pads_masks_and_is_rotation_deterministic() -> None:
    rows = [validate_coordinate_row(_row(3, "a"), split="train"), validate_coordinate_row(_row(5, "b"), split="train")]
    first = collate_e007_coordinates(rows, augment_rotation=True, seed=11)
    second = collate_e007_coordinates(rows, augment_rotation=True, seed=11)
    assert first["coordinates"].shape == (2, 5, 3)
    assert torch.equal(first["coordinates"], second["coordinates"])
    assert torch.allclose(first["coordinates"][0, :3].mean(dim=0), torch.zeros(3), atol=1e-6)
    assert torch.count_nonzero(first["coordinates"][0, 3:]) == 0
    assert torch.equal(first["pair_mask"], first["residue_mask"][:, :, None] & first["residue_mask"][:, None, :])
    assert set(first) == {
        "coordinates",
        "residue_mask",
        "pair_mask",
        "chain_continuity_mask",
        "relative_separation",
        "lengths",
        "sample_ids",
    }


def test_contiguous_policy_rejects_missing_or_broken_rows_at_collation() -> None:
    projected = validate_coordinate_row(_row(4), split="train")
    broken = copy.deepcopy(projected)
    broken["accepted_contiguous_single_chain"] = False
    with pytest.raises(ValueError, match="contiguous"):
        collate_e007_coordinates([broken])


def test_global_rms_coordinate_radius_formula() -> None:
    rows = [validate_coordinate_row(_row(3, "a"), split="train")]
    result = global_rms_coordinate_radius(rows)
    expected = np.sqrt((3.8**2 + 0.0 + 3.8**2) / (3 * 3))
    assert result["coordinate_scale_angstrom"] == pytest.approx(expected)
    assert result["valid_coordinate_count"] == 3


def test_stable_centering_uses_only_valid_residues_and_preserves_dtype() -> None:
    coordinates = torch.tensor(
        [[1000.0, -500.0, 20.0], [1001.0, -499.0, 22.0], [9000.0, 9000.0, 9000.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor([True, True, False])
    centered = stable_center_valid_coordinates(coordinates, mask)
    assert centered.dtype == torch.float32
    assert torch.allclose(centered[:2].double().mean(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.count_nonzero(centered[~mask]) == 0


@pytest.mark.parametrize("length", [55, 500])
def test_large_offset_float32_centering_matches_float64_reference(length: int) -> None:
    generator = torch.Generator().manual_seed(2)
    coordinates = torch.randn((length, 3), generator=generator) * 15
    coordinates += torch.tensor([333.5, -200.2, 100.7])
    mask = torch.ones(length, dtype=torch.bool)
    centered = stable_center_valid_coordinates(coordinates, mask)
    reference = coordinates.double() - coordinates.double().mean(dim=0, keepdim=True)
    assert torch.allclose(centered.double(), reference, atol=2e-5, rtol=0)
    assert float(centered.mean(dim=0).abs().max()) < 2e-6 * 12.22820347644835
