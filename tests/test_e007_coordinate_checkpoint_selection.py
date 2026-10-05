from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

import protein_distance_diffusion.evaluation.e007_coordinate_checkpoint_selection as selection

CONFIG_PATH = Path("configs/e007_coordinate_checkpoint_selection_v1.yaml")


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _temporary_config(tmp_path: Path) -> Path:
    config = _config()
    config["output_dir"] = str(tmp_path / "selection")
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_exact_selected_checkpoint_and_hash() -> None:
    config = selection._load_config(CONFIG_PATH)
    assert config["selected_checkpoint"] == {
        "optimizer_update": 9000,
        "path": "outputs/e007_coordinate_real_continuation_to_10000_v1/checkpoints/step-09000.pt",
        "sha256": "eb445f0b39067b8a00a47db27f966a6db97dc0e30bf81078b34ea54f111c7d82",
        "metadata_path": "outputs/e007_coordinate_real_continuation_to_10000_v1/checkpoints/step-09000.json",
        "metadata_sha256": "bdc0c9f74766f3e7103847950e39c2e708c76863cdbc950e93ccc2cee16d5234",
        "criterion": "minimum_validation_coordinate_v_mse",
        "validation_coordinate_v_mse": 0.4709909982886165,
    }
    verified = selection.verify_prerequisites(config)
    assert verified["hashes"]["selected_checkpoint"] == config["selected_checkpoint"]["sha256"]


def test_exact_pass_accounting_failures_and_per_length_counts() -> None:
    evidence = selection.verify_prerequisites(selection._load_config(CONFIG_PATH))["evidence"]
    assert evidence["sample_count"] == 160
    assert evidence["pass_count"] == 158
    assert evidence["pass_fraction"] == pytest.approx(0.9875)
    assert evidence["failure_count"] == 2
    assert evidence["per_length_pass_counts"] == {"64": 31, "128": 32, "256": 32, "384": 31, "500": 32}
    assert evidence["failures"] == [
        {
            "length": 64,
            "sample_index": 18,
            "seed": 8300019,
            "adjacent_reference_error_angstrom": 1.0460878764214092,
            "radius_of_gyration_relative_reference_error": 0.327660697796258,
            "contact_density_6a_reference_error": 0.027630942050711538,
            "contact_density_8a_reference_error": 0.05260726294915584,
            "contact_density_10a_reference_error": 0.10212983814785712,
        },
        {
            "length": 384,
            "sample_index": 7,
            "seed": 8600008,
            "adjacent_reference_error_angstrom": 1.0706711252252576,
        },
    ]
    assert evidence["original_all_record_gate_pass"] is False
    assert evidence["original_adjacent_error_limit_angstrom"] == 1.0


def test_pareto_contract_has_no_scalar_score_and_preserves_failed_gate() -> None:
    config = selection._load_config(CONFIG_PATH)
    source = Path(config["phase3h"]["source_dir"])
    pareto = json.loads((source / "checkpoint_pareto.json").read_text())
    assert not any("score" in key.lower() for key in selection._all_keys(pareto))
    assert pareto["dominated_by"]["10000"] == [9000]
    assert pareto["nondominated_checkpoints"] == [7500, 9000]
    report = json.loads((source / "report.json").read_text())
    assert report["decision"]["all_record_gate_pass"]["9000"] is False


def test_publication_is_atomic_separated_and_non_authorizing(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    output = tmp_path / "selection"
    staging = tmp_path / ".selection.inprogress"
    result = selection.publish_checkpoint_selection(config_path)
    assert result["classification"] == selection.CLASSIFICATION
    assert output.is_dir() and not staging.exists()
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "protocol.json",
        "selected_checkpoint.json",
        "artifact_inventory.json",
        "heartbeat.json",
    }
    assert (output / "report.json").read_bytes() != (output / "protocol.json").read_bytes()
    report = json.loads((output / "report.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    selected = json.loads((output / "selected_checkpoint.json").read_text())
    assert report["classification"] == selection.CLASSIFICATION
    assert report["scalar_score_used"] is False
    assert report["interpretation"]["claim_every_scientific_gate_passed"] is False
    assert selected["checkpoint_sha256"] == _config()["selected_checkpoint"]["sha256"]
    assert {field: protocol[field] for field in selection.NON_AUTHORIZING} == selection.NON_AUTHORIZING
    with pytest.raises(FileExistsError):
        selection.publish_checkpoint_selection(config_path)


def test_protected_inputs_remain_unchanged(tmp_path: Path) -> None:
    config = selection._load_config(CONFIG_PATH)
    before = selection.verify_prerequisites(config)
    selection.publish_checkpoint_selection(_temporary_config(tmp_path))
    assert selection.verify_prerequisites(config) == before


def test_plan_only_has_no_side_effect_or_model_workload(tmp_path: Path) -> None:
    config_path = _temporary_config(tmp_path)
    output = tmp_path / "selection"
    result = selection.plan_checkpoint_selection(config_path)
    assert result["output_created"] is False
    assert result["model_created"] is False
    assert result["cuda_used"] is False
    assert result["sampling_performed"] is False
    assert result["optimizer_created"] is False
    assert result["backward_performed"] is False
    assert result["dataset_scanned"] is False
    assert not output.exists()
    assert not (tmp_path / ".selection.inprogress").exists()
    source = inspect.getsource(selection)
    assert "import torch" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source


def test_hash_mismatch_refuses_before_output_creation(tmp_path: Path) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "selection")
    config["phase3h"]["report_sha256"] = "0" * 64
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ValueError, match="phase3h_report"):
        selection.publish_checkpoint_selection(path)
    assert not Path(config["output_dir"]).exists()
    assert not (tmp_path / ".selection.inprogress").exists()
