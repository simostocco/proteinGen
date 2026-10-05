from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

from protein_distance_diffusion.training.e007_local_backbone_repair import (
    LOCAL_TERMS,
    PREFLIGHT_V1_HASHES,
    PREFLIGHT_V2_TIMESTEPS,
    _canonical_sha,
    _finite_counts,
    _lower_activity_timesteps,
    _upper_safety_gate_failures,
    _validate_recovery_state,
    coefficient_tensor,
    load_coefficient_table,
    local_backbone_losses,
    monitor_dynamic_preflight,
    plan_local_backbone_repair,
    validate_dynamic_preflight_contract,
    validate_pilot_contract,
    weighted_local_objective,
    zero_update_drift_audit,
)

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "configs/e007_local_backbone_repair_budget_medium_coefficients_v1.json"
TABLE_V2 = ROOT / "configs/e007_local_backbone_repair_budget_medium_coefficients_v2.json"
PREFLIGHT_V2 = ROOT / "configs/e007_local_backbone_repair_dense_preflight_v2.yaml"
TABLE_V3 = ROOT / "configs/e007_local_backbone_repair_budget_medium_coefficients_v3.json"
DYNAMIC_V1 = ROOT / "configs/e007_local_backbone_repair_dynamic_stability_preflight_v1.yaml"
DYNAMIC_V2 = ROOT / "configs/e007_local_backbone_repair_dynamic_stability_preflight_v2.yaml"


def test_v2_safety_factors_are_exactly_derived_from_protected_v1_report() -> None:
    table = load_coefficient_table(TABLE_V2)
    derivation = table["safety_derivation"]
    assert derivation["preflight_v1_hashes"] == PREFLIGHT_V1_HASHES
    for anchor in derivation["anchors"]:
        assert anchor["safety_factor"] == pytest.approx(min(1.0, 0.12 / anchor["observed_max_combined_ratio"]))
        assert anchor["target_combined_ratio"] == pytest.approx(
            anchor["observed_max_combined_ratio"] * anchor["safety_factor"]
        )
        assert anchor["target_combined_ratio"] <= 0.12 + 1e-14


def test_v2_schedule_is_positive_complete_relative_and_v_mse_unchanged() -> None:
    table = load_coefficient_table(TABLE_V2)
    original = load_coefficient_table(TABLE)
    assert len(table["timesteps"]) == 500
    assert all(float(row["v_mse"]) == 1.0 for row in table["timesteps"].values())
    for t in range(500):
        row, base = table["timesteps"][str(t)]["local_terms"], original["timesteps"][str(t)]["local_terms"]
        ratios = [row[name] / base[name] for name in LOCAL_TERMS]
        assert min(ratios) > 0
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-12)
    anchor_timesteps = sorted(table["anchors"])
    for left, right in zip(anchor_timesteps, anchor_timesteps[1:], strict=False):
        midpoint = (left + right) // 2
        for t in (left, midpoint, right):
            factor = (
                table["timesteps"][str(t)]["local_terms"][LOCAL_TERMS[0]]
                / original["timesteps"][str(t)]["local_terms"][LOCAL_TERMS[0]]
            )
            assert factor > 0
        left_factor = (
            table["timesteps"][str(left)]["local_terms"][LOCAL_TERMS[0]]
            / original["timesteps"][str(left)]["local_terms"][LOCAL_TERMS[0]]
        )
        right_factor = (
            table["timesteps"][str(right)]["local_terms"][LOCAL_TERMS[0]]
            / original["timesteps"][str(right)]["local_terms"][LOCAL_TERMS[0]]
        )
        expected_midpoint = math.exp(
            math.log(left_factor) * (1 - (midpoint - left) / (right - left))
            + math.log(right_factor) * ((midpoint - left) / (right - left))
        )
        observed_midpoint = (
            table["timesteps"][str(midpoint)]["local_terms"][LOCAL_TERMS[0]]
            / original["timesteps"][str(midpoint)]["local_terms"][LOCAL_TERMS[0]]
        )
        assert observed_midpoint == pytest.approx(expected_midpoint, rel=1e-12)


def test_v2_plan_and_pilot_contract_are_non_authorizing_and_250_cells() -> None:
    import yaml

    config = yaml.safe_load(PREFLIGHT_V2.read_text())
    assert tuple(config["dense_preflight"]["timesteps"]) == PREFLIGHT_V2_TIMESTEPS
    assert len(PREFLIGHT_V2_TIMESTEPS) * len(config["lengths"]) * 2 == 250
    assert config["dense_preflight"]["optimizer_updates"] == 0
    assert config["dense_preflight"]["activity_gates"]["exempt_timesteps"] == [0]
    assert config["pilot_reviewed"] is False and config["pilot_authorized"] is False
    assert config["v3_pins"]["dense_preflight_report_sha256"] is None
    output_path = ROOT / config["dense_preflight"]["final_output_dir"]
    staging_path = ROOT / config["dense_preflight"]["staging_output_dir"]
    # The reviewed v2 evidence occupies the historical final_output_dir. A plan
    # must preserve it byte-for-byte and must not create a staging directory.
    report_hash_before = __import__("hashlib").sha256((output_path / "report.json").read_bytes()).hexdigest()
    assert output_path.is_dir() and not staging_path.exists()
    plan = plan_local_backbone_repair(PREFLIGHT_V2)
    contract = validate_pilot_contract(PREFLIGHT_V2)
    assert __import__("hashlib").sha256((output_path / "report.json").read_bytes()).hexdigest() == report_hash_before
    assert not staging_path.exists()
    assert plan["planned_dense_preflight_cells"] == 250
    assert contract["status"] == "blocked_pending_reviewed_dense_preflight"
    assert contract["pilot_reviewed"] is False and contract["pilot_authorized"] is False
    assert contract["dense_preflight_report_sha256"] is None
    assert contract["output_created"] is False and contract["model_created"] is False


def test_timestep_zero_skips_only_lower_activity_and_keeps_upper_safety() -> None:
    assert _lower_activity_timesteps((0, 6, 12), (0,)) == (6, 12)
    cell = {
        "identity": {"timestep": 0},
        "individual_term_v_ratios": {name: 0.01 for name in LOCAL_TERMS},
        "combined_auxiliary_v_ratio": 0.21,
        "total_v_ratio": 1.0,
    }
    assert _upper_safety_gate_failures(cell) == ["combined_ratio"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"individual_term_v_ratios": {name: 0.21 for name in LOCAL_TERMS}}, ["individual_ratio"]),
        ({"combined_auxiliary_v_ratio": 0.201}, ["combined_ratio"]),
        ({"total_v_ratio": 0.79}, ["total_ratio"]),
        ({"total_v_ratio": 1.31}, ["total_ratio"]),
    ],
)
def test_each_dynamic_upper_safety_gate_fails_closed(changes, expected) -> None:
    cell = {
        "individual_term_v_ratios": {name: 0.1 for name in LOCAL_TERMS},
        "combined_auxiliary_v_ratio": 0.1,
        "total_v_ratio": 1.0,
        **changes,
    }
    assert _upper_safety_gate_failures(cell) == expected


def test_non_finite_gradient_count_fails_finiteness_gate() -> None:
    counts = _finite_counts(torch.tensor([1.0, float("inf"), float("nan")]))
    assert counts == {"finite_count": 1, "total_count": 3, "non_finite_count": 2, "all_finite": False}


def test_protected_v1_files_remain_immutable() -> None:
    import hashlib

    v1_dir = ROOT / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dense_preflight_v1"
    paths = {
        "configuration_sha256": ROOT / "configs/e007_local_backbone_repair_dense_preflight_v1.yaml",
        "report_sha256": v1_dir / "report.json",
        "protocol_sha256": v1_dir / "protocol.json",
        "cells_sha256": v1_dir / "cells.jsonl",
    }
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PREFLIGHT_V1_HASHES[name]


def test_table_validates_full_schedule_anchors_interpolation_and_clamp() -> None:
    table = load_coefficient_table(TABLE)
    assert len(table["timesteps"]) == 500
    for term in LOCAL_TERMS:
        assert table["timesteps"]["0"]["local_terms"][term] == table["timesteps"]["25"]["local_terms"][term]
        assert table["timesteps"]["499"]["local_terms"][term] == table["anchors"][499][term]
        assert table["timesteps"]["250"]["local_terms"][term] == table["anchors"][250][term]
        start = table["anchors"][25][term]
        end = table["anchors"][250][term]
        expected = math.exp(math.log(start) * (1 - 50 / 225) + math.log(end) * (50 / 225))
        assert table["timesteps"]["75"]["local_terms"][term] == pytest.approx(expected, rel=1e-12)


def test_table_hash_and_missing_duplicate_or_bad_timestep_rows_fail(tmp_path: Path) -> None:
    original = json.loads(TABLE.read_text())
    bad_hash = tmp_path / "bad_hash.json"
    bad = copy.deepcopy(original)
    bad["timesteps"]["0"]["local_terms"]["adjacent"] *= 2
    bad_hash.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_coefficient_table(bad_hash)

    for mutation in ("missing", "out_of_order"):
        rows = copy.deepcopy(original)
        if mutation == "missing":
            rows["timesteps"].pop("10")
        else:
            rows["timesteps"] = dict(reversed(list(rows["timesteps"].items())))
        content = {key: rows[key] for key in ("policy", "selected_tier", "timesteps")}
        digest = _canonical_sha(content)
        rows["coefficient_table_sha256"] = digest
        path = tmp_path / f"{mutation}.json"
        path.write_text(json.dumps(rows))
        with pytest.raises(ValueError, match="hash mismatch|ordered unique"):
            load_coefficient_table(path, expected_sha256=digest)

    text = TABLE.read_text()
    duplicate_row = json.loads(TABLE.read_text())["timesteps"]["10"]
    duplicate_fragment = f'\n    "10": {json.dumps(duplicate_row)},'
    text = text.replace('  "timesteps": {', '  "timesteps": {', 1)
    insert_at = text.find('\n    "11":')
    text = text[:insert_at] + duplicate_fragment + text[insert_at:]
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(text)
    with pytest.raises(ValueError, match="duplicate coefficient-table key"):
        load_coefficient_table(duplicate_path)


def test_mixed_timestep_rows_are_per_sample_and_permutation_invariant() -> None:
    table = load_coefficient_table(TABLE)
    timesteps = torch.tensor([25, 250, 425])
    coefficients = coefficient_tensor(table, timesteps, torch.device("cpu"))
    assert not torch.equal(coefficients[0], coefficients[1])
    losses = {name: torch.tensor([1.0, 2.0, 3.0]) * (index + 1) for index, name in enumerate(LOCAL_TERMS)}
    weights = {name: coefficients[:, index] for index, name in enumerate(LOCAL_TERMS)}
    reduced, _ = weighted_local_objective(losses, weights)
    permutation = torch.tensor([2, 0, 1])
    permuted = {name: losses[name][permutation] for name in LOCAL_TERMS}
    permuted_weights = {name: weights[name][permutation] for name in LOCAL_TERMS}
    permuted_reduced, _ = weighted_local_objective(permuted, permuted_weights)
    torch.testing.assert_close(reduced, permuted_reduced)
    equal_time = coefficient_tensor(table, torch.tensor([250, 250]), torch.device("cpu"))
    assert torch.equal(equal_time[0], equal_time[1])
    single, _ = weighted_local_objective(
        {name: losses[name][1] for name in LOCAL_TERMS},
        {name: coefficients[1, index] for index, name in enumerate(LOCAL_TERMS)},
    )
    repeated, _ = weighted_local_objective(
        {name: losses[name][1].repeat(2) for name in LOCAL_TERMS},
        {name: coefficients[1, index].repeat(2) for index, name in enumerate(LOCAL_TERMS)},
    )
    torch.testing.assert_close(single, repeated)


def test_v3_high_noise_multiplier_anchors_interpolation_and_invariants() -> None:
    v2 = load_coefficient_table(TABLE_V2)
    v3 = load_coefficient_table(TABLE_V3)
    assert v3["anchors"] == {0: 1.0, 450: 1.0, 475: 0.8, 487: 0.6, 499: 0.375}
    assert len(v3["timesteps"]) == 500
    assert list(v3["timesteps"]) == [str(i) for i in range(500)]
    for timestep in range(500):
        row, parent = v3["timesteps"][str(timestep)], v2["timesteps"][str(timestep)]
        assert row["v_mse"] == 1.0
        assert math.isfinite(row["multiplier"]) and row["multiplier"] > 0
        assert all(math.isfinite(value) and value > 0 for value in row["local_terms"].values())
        if timestep <= 450:
            assert row["local_terms"] == parent["local_terms"]
        factors = [row["local_terms"][term] / parent["local_terms"][term] for term in LOCAL_TERMS]
        assert max(factors) == pytest.approx(min(factors), rel=1e-14)
    for left, right in ((450, 475), (475, 487), (487, 499)):
        mid = (left + right) // 2
        q = (mid - left) / (right - left)
        expected = math.exp(math.log(v3["anchors"][left]) * (1 - q) + math.log(v3["anchors"][right]) * q)
        assert v3["timesteps"][str(mid)]["multiplier"] == pytest.approx(expected, rel=1e-14)


def test_dynamic_preflight_contract_is_read_only_and_has_no_sampling_path() -> None:
    config = __import__("yaml").safe_load(DYNAMIC_V1.read_text())
    contract = config["dynamic_preflight"]
    assert config["arms"] == ["v_only"]
    assert contract["updates"] == 10 and contract["audit_updates"] == list(range(11))
    assert contract["required_cell"] == {"sample_id": "3qoc_C", "length": 128, "timestep": 499}
    assert len(contract["first_10_update_sample_ids"]) == 10
    assert contract["audit_sample_ids_by_length"]["128"] == "3qoc_C"
    assert all(
        contract[name] is False
        for name in (
            "sampling_performed",
            "reverse_sampling",
            "guided_sampling",
            "trajectory_generation",
            "expensive_evaluation_sampling",
        )
    )
    final = ROOT / contract["final_output_dir"]
    staging = ROOT / contract["staging_output_dir"]
    # Preserve the failed v1 staging evidence for forensic review.
    assert not final.exists() and staging.is_dir()
    result = validate_dynamic_preflight_contract(DYNAMIC_V1)
    assert result["mode"] == "validate_dynamic_preflight_contract"
    assert result["rows_0_450_unchanged"] and result["all_coefficients_finite_positive"]
    assert result["audit_updates"] == list(range(11))
    assert result["training_schedule_updates"] == 500
    assert result["training_corruption_seeds"] == [3914002 + i for i in range(10)]
    assert result["audit_sample_ids_by_length"]["128"] == "3qoc_C"
    assert result["staging_created"] is result["model_created"] is False
    assert result["checkpoint_loaded"] is result["cuda_initialized"] is False
    assert result["forward_pass"] is result["backward_pass"] is result["sampling_performed"] is False
    assert result["optimizer_updates"] == 0
    monitor = monitor_dynamic_preflight(DYNAMIC_V1)
    assert monitor["staging_exists"] is True
    assert monitor["sampling_performed"] is False
    assert not final.exists() and staging.is_dir()


def test_dynamic_v2_uses_isolated_paths_and_contract() -> None:
    import yaml

    config = yaml.safe_load(DYNAMIC_V2.read_text())
    contract = config["dynamic_preflight"]
    assert contract["version"] == "e007_phase3i2_dynamic_stability_preflight_v2"
    assert contract["final_output_dir"].endswith("dynamic_stability_preflight_v2")
    assert contract["staging_output_dir"].endswith(".local_backbone_repair_dynamic_stability_preflight_v2.inprogress")
    assert "dynamic_stability_preflight_v1" not in contract["final_output_dir"]
    assert not (ROOT / contract["final_output_dir"]).exists()
    assert not (ROOT / contract["staging_output_dir"]).exists()
    assert validate_dynamic_preflight_contract(DYNAMIC_V2)["status"] == "validated_read_only"


def test_dynamic_recovery_boundary_contract_and_resume_fail_closed() -> None:
    protected = {"checkpoint": "a" * 64}
    state = {
        "version": "e007_local_backbone_repair_v1",
        "configuration_sha256": __import__("hashlib").sha256(DYNAMIC_V1.read_bytes()).hexdigest(),
        "protected_hashes_sha256": _canonical_sha(protected),
        "protected_hashes": protected,
        "arm": "v_only",
        "paired_schedule_sha256": "b" * 64,
        "coefficient_table_sha256": load_coefficient_table(TABLE_V3)["sha256"],
        "successful_optimizer_boundary": True,
        "optimizer_update": 3,
        "completed_audit_updates": [0, 1, 2, 3],
        "next_required_audit_boundary": 4,
    }
    _validate_recovery_state(
        state,
        DYNAMIC_V1,
        protected,
        "v_only",
        "b" * 64,
        load_coefficient_table(TABLE_V3)["sha256"],
        audit_boundaries=range(11),
    )
    state["next_required_audit_boundary"] = 5
    with pytest.raises(ValueError, match="next-audit boundary contradiction"):
        _validate_recovery_state(
            state,
            DYNAMIC_V1,
            protected,
            "v_only",
            "b" * 64,
            load_coefficient_table(TABLE_V3)["sha256"],
            audit_boundaries=range(11),
        )


def test_dynamic_counterfactual_audit_restores_every_training_state(monkeypatch) -> None:
    import types

    import protein_distance_diffusion.training.e007_coordinate_real_loader_smoke as loader
    import protein_distance_diffusion.training.e007_coordinate_real_pilot as pilot
    import protein_distance_diffusion.training.e007_local_backbone_repair as repair

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(repair, "verify_protected_evidence", lambda _config: {"protected": "pinned"})

    def prepare(_rows, _scale, _factor):
        return {
            "coordinates": torch.zeros((1, 4, 3)),
            "residue_mask": torch.ones((1, 4), dtype=torch.bool),
            "chain_continuity_mask": torch.ones((1, 3), dtype=torch.bool),
            "lengths": torch.tensor([4]),
        }

    monkeypatch.setattr(loader, "prepare_coordinate_batch", prepare)
    monkeypatch.setattr(
        pilot, "uniform_coordinate_v_mse", lambda prediction, target, _mask: ((prediction - target) ** 2).mean()
    )
    monkeypatch.setattr(repair, "coefficient_tensor", lambda *_args: torch.full((1, 6), 1e-8))

    def local_losses(prediction, _clean, _mask, _continuity, _settings, *, per_structure):
        assert per_structure
        values = {name: ((prediction - 0.1) ** 2).mean().reshape(1) for name in LOCAL_TERMS}
        return values

    monkeypatch.setattr(repair, "local_backbone_losses", local_losses)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, noisy, _timestep, _lengths, _mask, _continuity):
            return {"v_prediction": noisy * self.weight}

    class Diffusion:
        def make_training_batch(self, clean, _mask, *, timesteps, generator):
            noise = torch.randn(clean.shape, generator=generator) * 0.01
            return types.SimpleNamespace(noisy_coordinates=clean + noise, coordinate_v_target=torch.ones_like(clean))

        def reconstruct_x0(self, noisy, _timestep, prediction, _mask):
            return noisy - prediction

    model = Model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    before = {
        "model": repair._fingerprint(model.state_dict()),
        "optimizer": repair._fingerprint(optimizer.state_dict()),
        "scheduler": repair._fingerprint(scheduler.state_dict()),
        "rng": repair._fingerprint(repair._rng_state()),
    }
    config = {
        "dynamic_preflight": {"maximum_combined_auxiliary_v_ratio": 0.2},
        "dataset_source_config": "configs/e007_denoiser_sampler_localization_v1.yaml",
        "local_objective": {},
        "coordinate_scale_angstrom": 1.0,
    }
    table = {"sha256": "table", "timesteps": {"25": {"local_terms": dict.fromkeys(LOCAL_TERMS, 1e-8)}}}
    audit = zero_update_drift_audit(
        model,
        optimizer,
        scheduler,
        Diffusion(),
        [(64, {"sample_id": "audit-64"})],
        config,
        table,
        torch.device("cpu"),
        update=0,
        data_cursor=0,
        audit_timesteps=[25],
        update_identity={"boundary": 0},
    )
    after = {
        "model": repair._fingerprint(model.state_dict()),
        "optimizer": repair._fingerprint(optimizer.state_dict()),
        "scheduler": repair._fingerprint(scheduler.state_dict()),
        "rng": repair._fingerprint(repair._rng_state()),
    }
    assert audit["pass"]
    assert audit["state_hashes_before"]["model"] == audit["state_hashes_after"]["model"]
    assert audit["state_hashes_before"]["optimizer"] == audit["state_hashes_after"]["optimizer"]
    assert audit["state_hashes_before"]["scheduler"] == audit["state_hashes_after"]["scheduler"]
    assert audit["state_hashes_before"]["rng"] == audit["state_hashes_after"]["rng"]
    assert audit["state_hashes_before"]["cpu_rng"] == audit["state_hashes_after"]["cpu_rng"]
    assert audit["state_hashes_before"]["cuda_rng"] == audit["state_hashes_after"]["cuda_rng"]
    assert audit["state_hashes_before"]["data_cursor"] == audit["state_hashes_after"]["data_cursor"]
    assert before == after
    assert model.training is True


def test_per_structure_masks_exclude_padding_and_chain_breaks() -> None:
    scale = 12.22820347644835
    native = torch.zeros((2, 8, 3), dtype=torch.float64)
    native[..., 0] = torch.arange(8, dtype=torch.float64) * 3.8 / scale
    prediction = native.clone().requires_grad_(True)
    mask = torch.ones((2, 8), dtype=torch.bool)
    mask[1, 6:] = False
    continuity = torch.ones((2, 7), dtype=torch.bool)
    continuity[1, 3] = False
    losses = local_backbone_losses(
        prediction,
        native,
        mask,
        continuity,
        {
            "coordinate_scale_angstrom": scale,
            "distance_normalizer_angstrom": 3.8,
            "discontinuity_threshold_angstrom": 4.5,
            "clash_threshold_angstrom": 3.0,
            "clash_minimum_sequence_separation": 3,
            "maximum_clash_pairs_per_structure": 32,
            "smooth_tail_beta": 4.0,
        },
        per_structure=True,
    )
    assert all(losses[name].shape == (2,) for name in LOCAL_TERMS)
    assert all(torch.isfinite(losses[name]).all() for name in LOCAL_TERMS)
    assert losses["denominators"]["adjacent"][1] == 4
