import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import finalize_e010_phase4a_v3_readonly as f


def test_validate_is_read_only_inventory_hashes_and_stats(tmp_path):
    (tmp_path / "evidence.txt").write_text("pinned\n")
    before = (tmp_path / "evidence.txt").read_bytes()
    inv = f.inventory(tmp_path)
    assert inv["file_count"] == 1
    assert inv["files"][0]["sha256"] == hashlib.sha256(before).hexdigest()
    assert inv["files"][0]["size_bytes"] == len(before)
    assert "mtime_ns" in inv["files"][0]
    assert (tmp_path / "evidence.txt").read_bytes() == before


def test_cuda_and_training_are_never_invoked():
    tree = ast.parse(Path(f.__file__).read_text())
    calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = [
        x.id
        if isinstance(x, ast.Name)
        else f"{x.value.id}.{x.attr}"
        if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
        else ""
        for x in calls
    ]
    assert not any("cuda" in n.lower() and n.lower() not in {"torch.load"} for n in names)
    assert not any("training" in n.lower() or "inference" in n.lower() for n in names)
    imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any("training" in x for x in imports)


def test_missing_evidence_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required evidence"):
        f.validate(tmp_path)


def test_real_preparation_record_separates_forward_permissions_from_execution():
    prep = json.loads((f.SOURCE / "preparation_manifest.json").read_text())
    config = json.loads((f.SOURCE / "phase4a_v3_extension_config.json").read_text())
    checks = f.preparation_contract_checks(config, prep)
    assert checks["preparation_training_started_at_preparation"] == {"observed": False, "expected": False}
    assert checks["preparation_phase4b_authorization"] == {"observed": False, "expected": False}
    assert checks["preparation_prospective_authorization"] == {"observed": False, "expected": False}
    assert checks["preparation_downstream_authorization"] == {"observed": False, "expected": False}
    # Historical bounded execution is not a forward authorization field in this schema.
    assert checks["historical_bounded_extension_execution_authorization"]["observed"] is None
    assert checks["historical_bounded_extension_execution_authorization"]["expected"] is None


def test_historical_execution_and_repaired_reporting_hashes_share_runner_path(tmp_path, monkeypatch):
    # The working runner can evolve after the reviewed historical repair. Its
    # old SHA must remain exact evidence, not be changed to match current code.
    preserved = Path(__file__).parent / "fixtures/e010_phase4a_v3_reporting_repair.py.txt"
    assert f.sha(preserved) == f.REPORTING_REPAIR_RUNNER_SHA256
    runner = tmp_path / "scripts/run_e010_phase4a_v3_extension.py"
    runner.parent.mkdir()
    runner.write_bytes(preserved.read_bytes())
    monkeypatch.setattr(f, "ROOT", tmp_path)
    prep = json.loads((f.SOURCE / "preparation_manifest.json").read_text())
    checks = f.runner_lineage_checks(f.SOURCE, prep)
    historical = checks["execution_runner_sha256"]
    repaired = checks["reporting_repair_runner_sha256"]
    assert historical["preparation_manifest"] == f.HISTORICAL_EXECUTION_RUNNER_SHA256
    assert historical["current_path_observed_sha256"] == f.REPORTING_REPAIR_RUNNER_SHA256
    assert historical["current_path_is_historical_source"] is False
    assert repaired["current_path_observed_sha256"] == f.REPORTING_REPAIR_RUNNER_SHA256
    f.validate_runner_lineage(checks)


def test_unreviewed_runner_bytes_fail_historical_lineage_validation(tmp_path, monkeypatch):
    runner = tmp_path / "scripts/run_e010_phase4a_v3_extension.py"
    runner.parent.mkdir()
    runner.write_bytes(b"unreviewed runner bytes\n")
    monkeypatch.setattr(f, "ROOT", tmp_path)
    prep = json.loads((f.SOURCE / "preparation_manifest.json").read_text())
    checks = f.runner_lineage_checks(f.SOURCE, prep)
    with pytest.raises(ValueError, match="runner lineage evidence mismatch"):
        f.validate_runner_lineage(checks)


def test_real_v3_config_resolves_strata_from_its_pinned_base_config():
    config = json.loads((f.SOURCE / "phase4a_v3_extension_config.json").read_text())
    assert isinstance(config["selection"], str)
    assert f.configured_strata_names(config) == ["20-64", "65-128", "129-256", "257-384", "385-500"]


def test_report_render_failure_does_not_prevent_json_publication(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "published"
    monkeypatch.setattr(
        f, "validate", lambda root: {"classification": "failed", "selected_exposure": 54, "stop_reason": "recorded"}
    )

    def broken(_):
        raise RuntimeError("renderer fault")

    monkeypatch.setattr(f, "render_markdown", broken)
    result = f.finalize(source, destination)
    assert result == destination
    assert json.loads((result / "review.json").read_text())["selected_exposure"] == 54
    assert "renderer fault" in (result / "review.md").read_text()
    assert (result / "corrected_handoff.json").is_file()
    assert (result / "SHA256SUMS.json").is_file()


def test_worsening_stop_and_earliest_best_checkpoint_selection():
    flags, reason = f.worsening_stop([1.0, 1.1, 1.2])
    assert flags == [True, True]
    assert reason == "development_worsened_at_two_consecutive_boundaries"
    assert f.worsening_stop([1.0, 1.1, 1.0])[1] != "development_worsened_at_two_consecutive_boundaries"
    rows = [
        {"exposure": 50, "development_mean_rmse": 0.8},
        {"exposure": 51, "development_mean_rmse": 0.7},
        {"exposure": 52, "development_mean_rmse": 0.7},
    ]
    assert f.select_checkpoint(rows)["exposure"] == 51
