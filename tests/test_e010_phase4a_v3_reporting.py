import json

import pytest

from scripts.e010_phase4a_v3_reporting import (
    finalize_from_json,
    render_markdown,
    validate_adjudication,
    validate_resume_prefix,
)

KEY = "marginal_improvement_35_to_50_50_to_60_60_to_70_70_to_80"


def adjudication():
    return {
        "schema": "e010_phase4a_v3_extension_training_metrics_v1",
        "classification": "pending_review",
        "selected_exposure": 50,
        "stop_reason": "development_worsened_at_two_consecutive_boundaries",
        "training_trajectory": [],
        "boundaries": [],
        KEY: [
            {
                "from_exposure": 35,
                "to_exposure": 50,
                "status": "evaluated",
                "paired_percentage_improvement": 0.1,
                "paired_bootstrap_95_ci": {"ci95_percentile": [0.05, 0.15]},
                "development_mean_rmse_from": 1.0,
                "development_mean_rmse_to": 0.9,
            },
            {
                "from_exposure": 50,
                "to_exposure": 60,
                "status": "not_evaluated_before_stopping",
                "paired_percentage_improvement": None,
                "paired_bootstrap_95_ci": None,
            },
            {
                "from_exposure": 60,
                "to_exposure": 70,
                "status": "not_evaluated_before_stopping",
                "paired_percentage_improvement": None,
                "paired_bootstrap_95_ci": None,
            },
            {
                "from_exposure": 70,
                "to_exposure": 80,
                "status": "not_evaluated_before_stopping",
                "paired_percentage_improvement": None,
                "paired_bootstrap_95_ci": None,
            },
        ],
    }


def test_evaluated_marginal_row_renders():
    md = render_markdown(adjudication())
    assert "35 → 50 | 10.00% | 5.00% to 15.00%" in md


def test_unevaluated_intervals_are_explicit_and_render_without_optional_keys():
    data = validate_adjudication(adjudication())
    md = render_markdown(data)
    assert md.count("not evaluated | not evaluated") == 3


def test_schema_rejects_missing_status_instead_of_assuming_optional_key():
    data = adjudication()
    del data[KEY][0]["status"]
    with pytest.raises(ValueError, match="schema"):
        validate_adjudication(data)


def test_complete_execution_survives_report_render_failure(monkeypatch, tmp_path):
    import scripts.e010_phase4a_v3_reporting as reporting

    payload = adjudication()
    (tmp_path / "training_metrics.json").write_text(json.dumps(payload))
    (tmp_path / "scientific_review.json").write_text("old")
    (tmp_path / "scientific_review.md").write_text("old")

    def fail(_):
        raise RuntimeError("mock Markdown failure")

    monkeypatch.setattr(reporting, "render_markdown", fail)
    with pytest.raises(RuntimeError):
        reporting.finalize_from_json(tmp_path)
    assert json.loads((tmp_path / "scientific_review.json").read_text())[KEY][0]["status"] == "evaluated"


def test_read_only_finalization_uses_json_only(tmp_path):
    (tmp_path / "training_metrics.json").write_text(json.dumps(adjudication()))
    report = finalize_from_json(tmp_path)
    assert report.read_text().startswith("# Phase 4A v3 extension review")
    assert not list(tmp_path.glob("*.pt"))


def test_finalizer_repairs_legacy_rows_before_validation(tmp_path):
    data = adjudication()
    del data[KEY][0]["status"]
    (tmp_path / "training_metrics.json").write_text(json.dumps(data))
    finalize_from_json(tmp_path)
    repaired = json.loads((tmp_path / "scientific_review.json").read_text())
    assert repaired[KEY][0]["status"] == "evaluated"


def test_partial_execution_resume_gate_requires_journal_matched_full_state():
    import hashlib

    checkpoint = b"mock full state checkpoint"
    digest = hashlib.sha256(checkpoint).hexdigest()
    row = {"extension_update": 12, "global_update": 1262, "checkpoint_sha256": digest}
    state = {
        key: {}
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "python_rng_state",
            "numpy_rng_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_state",
            "sampler_rng_state",
            "identity_exposures",
        )
    }
    state["schedule_cursor"] = 1262
    assert validate_resume_prefix([row], digest, state) == 1
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_prefix([row], "0" * 64, state)
