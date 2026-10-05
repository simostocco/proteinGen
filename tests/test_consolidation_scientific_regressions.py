"""Small regression cases for archive validation and exact smoke provenance."""

import numpy as np
import pytest

from scripts import prepare_e010_phase4a as prep
from scripts import smoke_e010_multicorruption_v2_exact_batch as smoke


@pytest.mark.parametrize("violation", ["identity", "length", "mask_shape", "mask_value", "nan", "inf"])
def test_authorized_source_arrays_reject_invalid_payloads(tmp_path, monkeypatch, violation):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    coords = np.arange(12, dtype=np.float32).reshape(4, 3)
    mask = np.ones(4, dtype=bool)
    sid = "good_A"
    if violation == "identity":
        sid = "good_a"
    elif violation == "length":
        coords = coords[:3]
    elif violation == "mask_shape":
        mask = mask[:3]
    elif violation == "mask_value":
        mask[1] = False
    elif violation == "nan":
        coords[0, 0] = np.nan
    elif violation == "inf":
        coords[0, 0] = np.inf
    path = tmp_path / "good_A.npz"
    np.savez(path, sample_id=np.asarray(sid), ca_coordinates=coords, residue_mask=mask)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="invalid structure arrays"):
        prep.load_source_arrays({"sample_id": "good_A", "source_path": path.name, "length": 4})
    assert path.read_bytes() == before


def test_exact_smoke_report_retains_validated_identity_and_seed_order():
    identities = [
        {"stratum": s, "sample_id": f"{s}:{i}", "corruption_seed": 100 + i, "corruption_index": i}
        for s in smoke.mc.STRATA
        for i in range(18)
    ]
    report = smoke.selected_identity_summary({"identities": identities})
    assert report["identities_and_corruption_seeds"] == identities
    assert report["per_stratum_counts"] == dict.fromkeys(smoke.mc.STRATA, 18)
