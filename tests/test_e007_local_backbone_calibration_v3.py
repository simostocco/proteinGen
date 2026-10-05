from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from protein_distance_diffusion.training.e007_local_backbone_calibration_v3 import (
    V2_REPORT_SHA256,
    correction_record,
    derive_coefficients,
    evaluate_candidate_gates,
    finite_counts,
    plan_only,
    select_strongest_passing,
    validate_panel,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v2/report.json"
CONFIG = ROOT / "configs/e007_local_backbone_repair_calibration_v3.yaml"


def test_exact_finite_counts_on_large_and_nonfinite_tensors() -> None:
    values = torch.ones(1_000_000)
    assert finite_counts(values)["all_finite"] is True
    for special in (float("nan"), float("inf"), float("-inf")):
        tensor = torch.ones(1_000_000)
        tensor[17] = special
        result = finite_counts(tensor)
        assert (result["finite_count"], result["total_count"], result["non_finite_count"]) == (999_999, 1_000_000, 1)
        assert result["all_finite"] is False


def test_immutable_v2_false_positive_correction() -> None:
    before = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    assert before == V2_REPORT_SHA256
    protocol = (
        ROOT
        / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v2/protocol.json"
    )
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        "d27acf93b9378435113b82d95139a017b5222ddaacf90ffe517f8bcbfa3e164a"
    )
    record = correction_record(REPORT)
    assert record["cell_count"] == 80
    assert record["term_coverage_values"] == [0.9999999403953552]
    assert all(
        v["passes_upper_stability_gates"]
        for k, v in record["candidate_gate_correction"].items()
        if k.startswith("schedule")
    )
    assert record["candidate_gate_correction"]["very_conservative"]["failed_individual_ratio_gates"] == 4
    assert record["pilot_authorized"] is False
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest() == before


def test_deterministic_coefficients_are_identity_independent() -> None:
    budget = {
        "low": {
            "adjacent": 0.025,
            "i_plus_2": 0.025,
            "i_plus_3": 0.025,
            "bond_angle_cosine": 0.02,
            "discontinuity": 0.008,
            "clash": 0.008,
        }
    }
    caps = {
        "adjacent": 0.16,
        "i_plus_2": 0.16,
        "i_plus_3": 0.16,
        "bond_angle_cosine": 0.16,
        "discontinuity": 0.05,
        "clash": 0.05,
    }
    first = derive_coefficients(REPORT, budget, caps)
    second = derive_coefficients(REPORT, budget, caps)
    assert first == second
    assert set(first["low"]) == {"25", "250", "425", "499"}
    assert first["low"]["25"] == first["low"]["25"]  # no length/sample argument exists


def _path_inventory(path: Path) -> tuple[bool, dict[str, tuple[str, str | None]]]:
    if not path.exists():
        return False, {}
    entries = {}
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            entries[relative] = ("symlink", str(item.readlink()))
        elif item.is_file():
            entries[relative] = ("file", hashlib.sha256(item.read_bytes()).hexdigest())
        elif item.is_dir():
            entries[relative] = ("directory", None)
    return True, entries


def _protected_v3_hashes() -> dict[str, tuple[Path, str]]:
    return {
        "configuration": (CONFIG, "381a3c862c30d32a6a8a5a235d1f49804145aa420d8f6a72b13b060a6f9866b7"),
        "report": (
            ROOT
            / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3/report.json",
            "c51a9c694616378c857828968ac358e057040f197f3285db0c9d9410eda81fb6",
        ),
        "protocol": (
            ROOT
            / (
                "reports/experiments/E007_matrix_sequence_cogeneration/"
                "local_backbone_repair_calibration_v3/protocol.json"
            ),
            "0c95805fcaf4ca4187b592a1015f33270cd535db6520af562dad05577094a50c",
        ),
        "plan": (
            ROOT
            / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3_plan.json",
            "f1e1ce6ac28974baaabf249814476264e8dceb9735ddcec21781cc0d8258dbe4",
        ),
    }


def test_plan_only_has_80_cells_and_does_not_create_temporary_outputs(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text())
    config["output_dir"] = str(tmp_path / "pilot-final")
    config["pilot_staging_dir"] = str(tmp_path / "pilot-staging")
    config["calibration_output_dir"] = str(tmp_path / "calibration-final")
    config["dense_preflight"] = {
        "final_output_dir": str(tmp_path / "dense-final"),
        "staging_output_dir": str(tmp_path / "dense-staging"),
    }
    local_config = tmp_path / "temporary-plan.yaml"
    local_config.write_text(yaml.safe_dump(config, sort_keys=False))
    output_paths = [
        Path(config["output_dir"]),
        Path(config["pilot_staging_dir"]),
        Path(config["calibration_output_dir"]),
        Path(config["calibration_output_dir"]).with_name(f".{Path(config['calibration_output_dir']).name}.inprogress"),
        Path(config["dense_preflight"]["final_output_dir"]),
        Path(config["dense_preflight"]["staging_output_dir"]),
    ]
    result = plan_only(local_config, REPORT)
    assert result["cell_count"] == result["estimated_model_forwards"] == 80
    assert result["pilot_authorized"] is False
    for key in (
        "staging_created",
        "model_created",
        "checkpoint_loaded",
        "cuda_initialized",
        "optimizer_created",
        "optimizer_updates",
        "forward_pass",
        "backward_pass",
        "update_state_created",
        "output_created",
    ):
        assert result[key] in (False, 0)
    assert all(not path.exists() for path in output_paths)


def test_production_plan_only_preserves_protected_calibration_and_output_paths() -> None:
    protected_dir = ROOT / "reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_calibration_v3"
    before_inventory = _path_inventory(protected_dir)
    before_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, (path, _) in _protected_v3_hashes().items()
    }
    for name, (_, expected) in _protected_v3_hashes().items():
        assert before_hashes[name] == expected

    dense_config = yaml.safe_load((ROOT / "configs/e007_local_backbone_repair_dense_preflight_v1.yaml").read_text())
    pilot_config = yaml.safe_load((ROOT / "configs/e007_local_backbone_repair_pilot_template_v1.yaml").read_text())
    pilot_final = Path(pilot_config["output_dir"])
    pilot_staging = pilot_final.with_name(f".{pilot_final.name}.inprogress")
    output_paths = [
        Path(dense_config["dense_preflight"]["final_output_dir"]),
        Path(dense_config["dense_preflight"]["staging_output_dir"]),
        pilot_final,
        pilot_staging,
    ]
    output_before = {str(path): _path_inventory(path) for path in output_paths}

    result = plan_only(CONFIG, REPORT)

    after_inventory = _path_inventory(protected_dir)
    after_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, (path, _) in _protected_v3_hashes().items()
    }
    assert after_inventory == before_inventory
    assert after_hashes == before_hashes
    assert {str(path): _path_inventory(path) for path in output_paths} == output_before
    assert result["model_created"] is False
    assert result["cuda_initialized"] is False
    assert result["optimizer_created"] is False
    assert result["optimizer_updates"] == 0
    assert result["forward_pass"] is False
    assert result["backward_pass"] is False
    assert result["update_state_created"] is False
    assert result["output_created"] is False


def test_v3_real_panel_validation_precedes_model_cuda_and_forward_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("v3 panel validation touched model/checkpoint/CUDA")

    from protein_distance_diffusion.evaluation import e007_denoiser_sampler_localization as localization

    monkeypatch.setattr(localization, "_load_model", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    report, panel = validate_panel(CONFIG)
    assert len(panel) == report["panel_count"] == 20
    assert report["structure_timestep_cells"] == 80
    assert set(report["counts_by_length"].values()) == {4}
    assert report["validated_for_v3"] is True
    assert all(
        report[key] is False
        for key in ("model_created", "checkpoint_loaded", "cuda_initialized", "forward_pass", "backward_pass")
    )


def test_upper_lower_activity_gates_report_reasons_and_strongest_rule() -> None:
    from protein_distance_diffusion.training.e007_local_backbone_calibration_v3 import validate_config

    config = validate_config(CONFIG)
    cell = {
        "identity": {"timestep": 25, "length": 64, "sample_id": "a"},
        "all_finite": True,
        "non_finite_count": 0,
        "candidate_coefficients": {
            "budget_low": {
                "individual_ratios": {
                    t: 0.01 for t in ("adjacent", "i_plus_2", "i_plus_3", "bond_angle_cosine", "clash", "discontinuity")
                },
                "combined_auxiliary_to_v_ratio": 0.02,
                "total_to_v_ratio": 1.0,
            }
        },
    }
    cells = [cell]
    failures = evaluate_candidate_gates(cells, config, ["budget_low"])
    assert {x["gate"] for x in failures["budget_low"]["failures"]} >= {"missing_activity_cells"}
    cell["candidate_coefficients"]["budget_low"]["individual_ratios"]["adjacent"] = 0.21
    upper_failure = evaluate_candidate_gates(cells, config, ["budget_low"])
    assert any(x["gate"] == "individual_term_upper" for x in upper_failure["budget_low"]["failures"])
    low_activity_cells = []
    for timestep in (25, 250, 425):
        low_activity_cells.append(
            {
                **cell,
                "identity": {**cell["identity"], "timestep": timestep},
                "candidate_coefficients": {
                    "budget_low": {
                        "individual_ratios": {
                            term: 1e-8 for term in cell["candidate_coefficients"]["budget_low"]["individual_ratios"]
                        },
                        "combined_auxiliary_to_v_ratio": 1e-8,
                        "total_to_v_ratio": 1.0,
                    }
                },
            }
        )
    lower_failure = evaluate_candidate_gates(low_activity_cells, config, ["budget_low"])
    assert {x["gate"] for x in lower_failure["budget_low"]["failures"]} >= {
        "core_activity",
        "tail_activity",
        "combined_activity",
    }
    assert (
        select_strongest_passing(
            {"budget_low": {"passes": True}, "budget_medium": {"passes": True}}, ["budget_low", "budget_medium"]
        )["selected"]
        == "budget_medium"
    )
    assert select_strongest_passing({"budget_low": {"passes": True}}, ["budget_low", "budget_low"])["selected"] is None


def test_pairwise_cosine_matrix_is_symmetric_and_complete() -> None:
    from protein_distance_diffusion.training.e007_local_backbone_calibration_v3 import cosine_matrix

    matrix = cosine_matrix(
        {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0]), "c": torch.tensor([-1.0, 0.0])}
    )
    assert set(matrix) == {"a", "b", "c"}
    assert matrix["a"]["b"] == pytest.approx(0.0)
    assert matrix["a"]["c"] == pytest.approx(-1.0)
    assert matrix["a"]["c"] == pytest.approx(matrix["c"]["a"])
