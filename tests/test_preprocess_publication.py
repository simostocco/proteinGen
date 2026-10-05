"""Publication must preserve distinct identities and reject malformed payloads."""

from dataclasses import replace

import numpy as np
import pytest

from protein_distance_diffusion.data.preprocess import ProteinSample, SamplePublicationError, save_processed_sample


def _sample():
    return ProteinSample("7ar7_M", "7ar7", "M", "AAA", ["1", "2", "3"], np.arange(9).reshape(3, 3), {})


def test_publication_identity_collision_preserves_existing_bytes(tmp_path):
    # Model an already-resolved case-insensitive destination on any filesystem.
    path = tmp_path / "7ar7_M.npz"
    np.savez(path, sample_id=np.asarray("7ar7_m"), ca_coordinates=np.zeros((67, 3)))
    before = path.read_bytes()
    with pytest.raises(SamplePublicationError, match="cannot publish") as error:
        save_processed_sample(_sample(), tmp_path)
    assert error.value.reason == "publication_identity_collision"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"ca_coordinates": np.zeros((2, 3))}, "coordinate_length_mismatch"),
        ({"residue_ids": ["1"]}, "residue_count_mismatch"),
        ({"ca_coordinates": np.full((3, 3), np.nan)}, "nonfinite_coordinates"),
        ({"ca_coordinates": np.full((3, 3), np.inf)}, "nonfinite_coordinates"),
    ],
)
def test_malformed_publication_fails_before_creating_archive(tmp_path, mutation, reason):
    with pytest.raises(SamplePublicationError) as error:
        save_processed_sample(replace(_sample(), **mutation), tmp_path)
    assert error.value.reason == reason
    assert not list(tmp_path.iterdir())


def test_same_identity_can_be_republished(tmp_path):
    sample = _sample()
    save_processed_sample(sample, tmp_path)
    save_processed_sample(sample, tmp_path)
    with np.load(tmp_path / "7ar7_M.npz", allow_pickle=False) as archive:
        assert archive["sample_id"].item() == sample.sample_id


def test_worker_records_collision_exclusion_without_overwriting(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import preprocess_pdb as worker

    path = tmp_path / "7ar7_M.npz"
    np.savez(path, sample_id=np.asarray("7ar7_m"))
    before = path.read_bytes()
    monkeypatch.setattr(worker, "parse_structure_file_with_rejections", lambda *args, **kwargs: ([_sample()], []))
    config = SimpleNamespace(
        backend="gemmi",
        min_length=1,
        max_length=500,
        chain_id=None,
        residue_mappings={},
        allowed_methods=None,
        max_xray_resolution_angstrom=None,
        max_cryoem_resolution_angstrom=None,
        missing_calpha_policy="reject",
        max_terminal_trim_fraction=0,
        baseline_rows_by_sample_id=None,
        samples_dir=tmp_path,
    )
    result = worker.worker_process_source(SimpleNamespace(path="source.cif", size=1, mtime_ns=1), config)
    assert result["status"] == worker.DETERMINISTIC_REJECTION_STATUS
    assert result["manifest_rows"] == []
    assert result["rejections"][0]["reason"] == "publication_identity_collision"
    assert result["rejections"][0]["sample_id"] == "7ar7_M"
    assert path.read_bytes() == before
