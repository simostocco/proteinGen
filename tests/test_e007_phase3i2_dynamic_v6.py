"""V6 descriptive warning, retained hard gates, and immutable v5 pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_e007_phase3i2_audit_lifecycle import TinyDiffusion, _setup

from protein_distance_diffusion.training import e007_local_backbone_repair as repair

CONFIG = Path("configs/e007_local_backbone_repair_dynamic_stability_preflight_v6.yaml")
INCIDENT = Path(
    "reports/experiments/E007_matrix_sequence_cogeneration/"
    "local_backbone_repair_dynamic_stability_v5_incident_review_v1.json"
)


def _audit(monkeypatch, change=None):
    model, optimizer, scheduler, config, panel, _ = _setup(monkeypatch)
    config["dynamic_preflight"] = {
        "version": "e007_phase3i2_dynamic_stability_preflight_v6",
        "combined_auxiliary_warning_threshold": 0.20,
    }

    def cell(*_args, **_kwargs):
        record = {
            "update": 7,
            "sample_id": "3qoc_C",
            "length": 128,
            "timestep": 425,
            "finite_counts": {"v": {"all_finite": True}},
            "individual_ratios": {name: 0.1 for name in repair.LOCAL_TERMS},
            "combined_auxiliary_ratio": 0.3132148858397767,
            "total_v_ratio": 0.9970738914566103,
            "total_v_cosine": 0.95,
            "loss_values": {"v": 0.1, "raw": {"x": 0.1}, "weighted": {"x": 0.1}},
        }
        if change:
            change(record, model)
        return record

    monkeypatch.setattr(repair, "_zero_update_drift_cell", cell)
    return repair.zero_update_drift_audit(
        model,
        optimizer,
        scheduler,
        TinyDiffusion(),
        panel[1:2],
        config,
        {"sha256": "table"},
        torch.device("cpu"),
        update=7,
        data_cursor=7,
        audit_timesteps=[425],
    )


def test_exact_v5_failure_is_v6_warning_with_complete_serializable_telemetry(monkeypatch):
    result = _audit(monkeypatch)
    assert result["pass"] is True
    assert result["violations"] == []
    assert result["warning_count"] == 1
    warning = result["warnings"][0]
    assert warning["classification"] == "descriptive_warning"
    assert (warning["update"], warning["sample_id"], warning["length"], warning["timestep"]) == (7, "3qoc_C", 128, 425)
    assert warning["combined_auxiliary_v_ratio"] == 0.3132148858397767
    assert warning["total_v_ratio"] == 0.9970738914566103
    summary = repair._dynamic_warning_telemetry([result])
    assert summary["global"]["warning_count"] == 1
    assert summary["by_boundary"]["7"]["total_v_cosines"] == [0.95]
    assert summary["global"]["maximum_combined_auxiliary_identity"]["sample_id"] == "3qoc_C"
    json.dumps({"audit": result, "summary": summary}, allow_nan=False)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda record, _model: record["individual_ratios"].update(adjacent=0.200001), "individual_ratio"),
        (lambda record, _model: record.update(total_v_ratio=1.300001), "total_v_ratio"),
        (lambda record, _model: record["finite_counts"]["v"].update(all_finite=False), "non_finite_gradient"),
        (lambda record, model: model.weight.data.add_(1), "audit_mutated_state_or_protected_hash"),
    ],
)
def test_v6_hard_gates_remain_closed(monkeypatch, change, reason):
    result = _audit(monkeypatch, change)
    assert result["pass"] is False
    assert reason in [violation["reason"] for violation in result["violations"]]


def test_v6_memory_limits_remain_hard():
    limits = {"maximum_rss_mib": 4096, "maximum_cuda_allocated_mib": 6144, "maximum_cuda_reserved_mib": 7680}
    observed = {"peak_rss_mib": 4000, "run_peak_cuda_allocated_mib": 6000, "run_peak_cuda_reserved_mib": 7600}
    assert not repair._dynamic_memory_violation(observed, limits)
    for key in observed:
        excessive = dict(
            observed,
            **{
                key: limits[
                    {
                        "peak_rss_mib": "maximum_rss_mib",
                        "run_peak_cuda_allocated_mib": "maximum_cuda_allocated_mib",
                        "run_peak_cuda_reserved_mib": "maximum_cuda_reserved_mib",
                    }[key]
                ]
                + 1
            },
        )
        assert repair._dynamic_memory_violation(excessive, limits)


def test_v5_evidence_pinned_and_read_only_contract_plan():
    incident = json.loads(INCIDENT.read_text())
    before = {
        name: hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        for name, item in incident["v5_evidence"].items()
    }
    assert before == {name: item["sha256"] for name, item in incident["v5_evidence"].items()}
    contract = repair.validate_dynamic_preflight_contract(CONFIG)
    plan = repair.plan_local_backbone_repair(CONFIG)
    assert contract["status"] == "validated_read_only"
    assert contract["staging_output_exists"] is False
    assert contract["final_output_exists"] is True
    assert contract["model_created"] is False and contract["cuda_initialized"] is False
    assert plan["mode"] == "plan_only"
    assert plan["planned_sampling_records_including_each_guidance_strength"] == 0
    assert not Path(repair.load_config(CONFIG)["dynamic_preflight"]["staging_output_dir"]).exists()
    assert before == {
        name: hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        for name, item in incident["v5_evidence"].items()
    }
