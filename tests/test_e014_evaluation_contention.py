import json

import torch

from protein_distance_diffusion.evaluation.e014_evaluation import adjudicate_final, aggregate
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_contention import baseline, compare
from protein_distance_diffusion.training.e014_runtime import noise_seed


def test_paired_noise_and_no_target_inference():
    import inspect

    assert list(inspect.signature(E014CapacityDenoiser.forward).parameters) == [
        "self",
        "noisy_coordinates",
        "timesteps",
        "lengths",
        "residue_mask",
    ]
    diffusion = CoordinateVPDiffusion(500)
    x = torch.randn(1, 20, 3)
    mask = torch.ones(1, 20, dtype=torch.bool)
    batches = [
        diffusion.make_training_batch(
            x,
            mask,
            timesteps=torch.tensor([490]),
            generator=torch.Generator().manual_seed(noise_seed("dev", "sample", 490)),
        )
        for _ in range(2)
    ]
    torch.testing.assert_close(batches[0].noisy_coordinates, batches[1].noisy_coordinates, atol=0, rtol=0)
    torch.testing.assert_close(batches[0].coordinate_v_target, batches[1].coordinate_v_target, atol=0, rtol=0)


def test_denominators_and_minimum_exposure_gate():
    rows = [
        dict(
            sample_id=f"id_{i}",
            length=20,
            length_stratum="20-64",
            timestep=490,
            v_mse=1.0,
            radius_ratio=0.1 if i == 0 else 0.5,
            collapsed=i == 0,
            finite=True,
            clash_count=2,
            clash_pair_count=10,
            clash_fraction=0.2,
            torsion_assessed=4,
            triple_sign_assessed=3,
            triple_inversion_count=1,
            triple_inversion_fraction=1 / 3,
            local_mse_angstrom2=1.0,
            adjacent_valid_fraction=1.0,
        )
        for i in range(2)
    ]
    groups = aggregate(rows)
    total = next(g for g in groups if g["dimension"] == "all")
    assert total["collapse_count"] == 1 and total["protein_condition_denominator"] == 2
    assert total["triple_inversion_count_sum"] == 2 and total["triple_sign_assessed_sum"] == 6
    cfg = {"minimum_epochs": 2}
    assert adjudicate_final({}, {}, 1, cfg)["architectural_conclusion"] is None


def test_contention_persistent_not_single_noisy_batch(tmp_path):
    log = tmp_path / "heartbeat.jsonl"
    state = tmp_path / "state.json"
    cfg = dict(
        heartbeat_log=str(log),
        state_path=str(state),
        rolling_completed_batches=5,
        minimum_completed_batches=3,
        degradation_fraction=0.10,
    )
    records = [dict(committed_batch=i, recent_candidates_s=1000.0 + i, timestamp_unix=i) for i in range(1, 6)]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    state.write_text(json.dumps(dict(phase="screen", screened=500000, timestamp_unix=5)))
    before = baseline(cfg)
    with log.open("a") as stream:
        stream.write(json.dumps(dict(committed_batch=6, recent_candidates_s=700, timestamp_unix=6)) + "\n")
    assert not compare(before, cfg)["persistent_material_degradation"]
    with log.open("a") as stream:
        for i in (7, 8):
            stream.write(json.dumps(dict(committed_batch=i, recent_candidates_s=700, timestamp_unix=i)) + "\n")
    assert compare(before, cfg)["persistent_material_degradation"]


def test_verification_phase_uses_native_audit_timing(tmp_path):
    production = tmp_path / "screen_production"
    production.mkdir()
    state = production / "STATE.json"
    log = production / "HEARTBEAT.jsonl"
    archive = tmp_path / "scientific/reports/command_logs/mmseqs/verify_protected"
    records = []
    for batch in range(1, 7):
        records.append(
            dict(
                phase="verify-protected",
                timestamp_unix=batch,
                recent_candidates_s=99999,
                committed_batch=241,
                audit=dict(batch=batch, screened=batch * 100000),
            )
        )
        directory = archive / f"batch-{batch:04d}"
        directory.mkdir(parents=True)
        (directory / "command.rss.txt").write_text(
            "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:40.00\nExit status: 0\n"
        )
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    state.write_text(json.dumps(dict(phase="verify-protected", timestamp_unix=6)))
    cfg = dict(
        heartbeat_log=str(log),
        state_path=str(state),
        rolling_completed_batches=5,
        minimum_completed_batches=3,
        degradation_fraction=0.10,
    )
    before = baseline(cfg)
    assert before["phase"] == "verify-protected"
    assert before["median_candidates_s"] == 1000
    assert before["committed_batch"] == 6
    state.write_text(json.dumps(dict(phase="unknown-new-phase", timestamp_unix=7)))
    assert compare(before, cfg)["phase_changed"]
