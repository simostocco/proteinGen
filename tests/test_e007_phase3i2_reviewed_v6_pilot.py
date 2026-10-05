"""Regression checks for the exact reviewed-v6 production pilot configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from protein_distance_diffusion.training import e007_local_backbone_repair as repair

CONFIG = Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v2.yaml")
DECISION = Path(
    "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_phase3i2_reviewed_v6_decision_v1.json"
)


def test_exact_production_configuration_and_reviewed_evidence_are_pinned() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    decision = json.loads(DECISION.read_text())
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == decision["pilot_configuration_sha256"]
    assert config["arms"] == ["v_only", "v_plus_local"]
    assert config["selected_checkpoint"] == decision["selected_checkpoint"]
    assert config["selected_checkpoint"]["optimizer_update"] == 9000
    assert config["pilot"]["initialization"] == "step_9000_model_weights_only"
    assert config["pilot"]["optimizer_state"] == config["pilot"]["scheduler_state"] == "fresh_identical_per_arm"
    assert config["pilot"]["successful_optimizer_updates_per_arm"] == 500
    assert config["pilot"]["evaluation_updates"] == [0, 50, 100, 250, 500]
    assert config["gradient_audit"]["updates"] == list(repair.GRADIENT_AUDIT_UPDATES)
    assert config["pilot_audit_semantics"] == "reviewed_v6"
    assert config["online_drift_monitor"]["individual_ratio_max"] == 0.2
    assert config["online_drift_monitor"]["combined_auxiliary_warning_threshold"] == 0.2
    assert "combined_auxiliary_ratio_max" not in config["online_drift_monitor"]
    assert config["online_drift_monitor"]["total_ratio_min"] == 0.8
    assert config["online_drift_monitor"]["total_ratio_max"] == 1.3
    assert all(value is False for value in decision["downstream_authorizations"].values())
    assert all(value is False for key, value in config["authorization"].items() if key != "pilot_authorized")
    for item in decision["v6_artifacts"].values():
        assert repair._sha256_file(Path(item["path"])) == item["sha256"]
    assert (
        repair._sha256_file(Path(decision["coefficient_schedule"]["path"]))
        == decision["coefficient_schedule"]["sha256"]
    )


def test_control_arm_never_computes_auxiliary(monkeypatch: pytest.MonkeyPatch) -> None:
    prediction = torch.tensor([[[1.0]]])
    batch = SimpleNamespace(coordinate_v_target=torch.zeros_like(prediction))
    corruption = {
        "batch": batch,
        "mask": torch.ones(prediction.shape[:2], dtype=torch.bool),
        "timesteps": torch.tensor([499]),
    }
    prepared = {}
    monkeypatch.setattr(repair, "local_backbone_losses", lambda *_args, **_kwargs: pytest.fail("auxiliary computed"))
    monkeypatch.setattr(repair, "coefficient_tensor", lambda *_args, **_kwargs: pytest.fail("coefficient gathered"))
    total, v_loss, raw, weighted, rows, auxiliary = repair._pilot_training_objective(
        "v_only", prediction, corruption, prepared, object(), {}, {}, torch.device("cpu")
    )
    assert total is v_loss and float(total) == 1.0
    assert (raw, weighted, rows, auxiliary) == ({}, {}, None, None)


def test_both_arms_get_identical_weights_and_independent_fresh_training_state() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["selected_checkpoint"]["sha256"] == json.loads(DECISION.read_text())["selected_checkpoint"]["sha256"]
    checkpoint_weights = torch.nn.Linear(2, 1).state_dict()
    optimizer_spec = {"learning_rate": 1e-4, "weight_decay": 0.0, "betas": [0.9, 0.999]}
    first = repair._initialize_pilot_arm(
        torch.nn.Linear, {"in_features": 2, "out_features": 1}, checkpoint_weights, optimizer_spec, torch.device("cpu")
    )
    second = repair._initialize_pilot_arm(
        torch.nn.Linear, {"in_features": 2, "out_features": 1}, checkpoint_weights, optimizer_spec, torch.device("cpu")
    )
    for name, value in checkpoint_weights.items():
        assert torch.equal(first[0].state_dict()[name], value)
        assert torch.equal(second[0].state_dict()[name], value)
        assert first[0].state_dict()[name].data_ptr() != second[0].state_dict()[name].data_ptr()
    assert first[1].state_dict() == second[1].state_dict()
    assert first[1].state_dict()["state"] == second[1].state_dict()["state"] == {}
    assert first[2].state_dict() == second[2].state_dict()


def test_v3_coefficient_lookup_is_timestep_indexed() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    table = repair.load_coefficient_table(config["local_objective"]["coefficient_table_path"])
    times = torch.tensor([0, 450, 475, 487, 499])
    rows = repair.coefficient_tensor(table, times, torch.device("cpu"))
    for index, timestep in enumerate(times.tolist()):
        assert rows[index].tolist() == pytest.approx(
            [table["timesteps"][str(timestep)]["local_terms"][term] for term in repair.LOCAL_TERMS]
        )
    assert rows[0, 0] > rows[-1, 0]


def test_production_safeguards_are_reported_separately() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    records = [
        {
            "sampler": "native_reverse",
            "requested_length": length,
            "sample_index": index,
            "seed": 3917000 + length * 100 + index,
            "finite_coordinates": True,
            "signed_pseudo_dihedral_positive_fraction": 0.55,
        }
        for length in config["lengths"]
        for index in range(config["pilot"]["samples_per_length_per_sampler"])
    ]
    expected = (
        len(config["lengths"])
        * config["pilot"]["samples_per_length_per_sampler"]
        * (1 + len(config["guidance"]["strengths"]))
    )
    control = {
        "sampling_record_count": expected,
        "sampling_records": records,
        "denoising": {"global": {"coordinate_v_mse": 0.20}},
        "descriptor_space_dispersion": {"mean_pairwise_distance": 2.0},
        "duplicate_record_fraction": 0.0,
    }
    candidate = {
        **control,
        "denoising": {"global": {"coordinate_v_mse": 0.21}},
    }
    safeguards = repair._pilot_safeguards({"v_only": control, "v_plus_local": candidate}, config)
    assert safeguards["components"]["denoising_objective"]["passed"] is False
    assert safeguards["components"]["diversity"]["passed"] is True
    assert safeguards["components"]["chirality"]["passed"] is True
    assert safeguards["components"]["finite_coordinate_rate"]["passed"] is True
    assert safeguards["components"]["sampling_completion"]["passed"] is True
    assert safeguards["all_passed_excluding_global_topology"] is False


def test_read_only_plan_and_contract_do_not_create_output() -> None:
    prepared_config = Path("configs/e007_local_backbone_repair_pilot_phase3i2_reviewed_v6_v3.yaml")
    config = yaml.safe_load(prepared_config.read_text())
    output = Path(config["output_dir"])
    staging = output.with_name(f".{output.name}.inprogress")
    failed_heartbeats = [
        Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            ".local_backbone_repair_pilot_phase3i2_final_v1.inprogress/heartbeat.json"
        ),
        Path(
            "reports/experiments/E007_matrix_sequence_cogeneration/"
            ".local_backbone_repair_dynamic_stability_preflight_v5.inprogress/heartbeat.json"
        ),
    ]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in failed_heartbeats}
    assert not output.exists() and not staging.exists()
    plan = repair.plan_local_backbone_repair(prepared_config)
    assert plan["mode"] == "plan_only" and plan["output_created"] is False
    contract = repair.validate_pilot_contract(prepared_config)
    assert contract["status"] == "pilot_contract_prepared"
    assert contract["training_panel_updates"] == 500
    assert len(contract["evaluation_panel_identities"]) == 5
    assert contract["coefficient_lookup_timesteps_verified"] == list(range(500))
    assert contract["protected_dataset_inventory"]["total"] == 3736
    assert contract["protected_dataset_inventory"]["relocated_resolutions"] == 11
    assert all(
        contract[key] is False
        for key in (
            "model_created",
            "checkpoint_loaded",
            "cuda_initialized",
            "optimizer_created",
            "output_created",
            "staging_created",
            "forward_pass",
            "backward_pass",
            "sampling_performed",
        )
    )
    assert contract["optimizer_updates"] == 0
    assert not output.exists() and not staging.exists()
    assert before == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in failed_heartbeats}
