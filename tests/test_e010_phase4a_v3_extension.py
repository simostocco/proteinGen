import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v3"


def test_v3_preparation_is_exact_and_non_authorizing():
    manifest = json.loads((V3 / "preparation_manifest.json").read_text())
    validation = json.loads((V3 / "exact_continuation_validation.json").read_text())
    assert validation["status"] == "valid"
    assert validation["selected_checkpoint_matches_exposure50_full_state"] is True
    assert validation["optimizer_continuation_validated"] is True
    assert validation["scheduler_state_preserved"] is True
    assert validation["scaler_state_preserved"] is True
    assert validation["python_numpy_torch_cpu_torch_cuda_and_sampler_rng_states_preserved"] is True
    assert manifest["training_started"] is False
    assert manifest["authorization"] == {"downstream": False, "prospective": False, "phase4b": False}
    assert not (V3 / "phase4a_training_v3.final").exists()


def test_extension_schedule_gives_each_training_identity_one_exposure_per_epoch():
    data = json.loads((V3 / "extension_schedule.json").read_text())
    assert data["optimizer_updates"] == 690
    assert data["boundaries"] == [60, 70, 80]
    counts = {}
    for row in data["schedule"]:
        for ids in row["stratum_microbatches"].values():
            for sid in ids:
                counts[sid] = counts.get(sid, 0) + 1
    assert len(counts) == 2048
    assert set(counts.values()) == {30}


def test_v2_final_and_scientific_review_match_the_preserved_pins():
    v2 = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v2"
    final = v2 / "phase4a_training_v2.final"
    expected = json.loads((V3 / "preparation_manifest.json").read_text())["pins"]["v2_final_recursive_inventory"]
    observed = []
    for path in sorted(final.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            observed.append(
                {
                    "path": path.relative_to(final).as_posix(),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    assert observed == expected
    review_dir = (
        ROOT
        / "reports/experiments/E010_global_equivariant_expressivity"
        / "phase4a_supervised_generalization_v2/phase4a_v2_scientific_review_v1"
    )
    sums = json.loads((review_dir / "SHA256SUMS.json").read_text())
    for name, digest in sums.items():
        observed = hashlib.sha256((review_dir / name).read_bytes()).hexdigest()
        assert observed == digest
