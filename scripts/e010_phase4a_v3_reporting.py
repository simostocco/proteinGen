"""Schema validation and pure Markdown rendering for Phase 4A v3 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MARGINAL_REQUIRED = {
    "from_exposure",
    "to_exposure",
    "status",
    "paired_percentage_improvement",
    "paired_bootstrap_95_ci",
}
MARGINAL_EVALUATED = {"development_mean_rmse_from", "development_mean_rmse_to"}


def validate_adjudication(data: dict[str, Any]) -> dict[str, Any]:
    """Return a validated JSON-safe adjudication; reject incomplete rows."""
    if data.get("schema") != "e010_phase4a_v3_extension_training_metrics_v1":
        raise ValueError("unexpected Phase 4A v3 adjudication schema")
    key = "marginal_improvement_35_to_50_50_to_60_60_to_70_70_to_80"
    rows = data.get(key)
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("marginal improvement table must contain four transitions")
    normalized = []
    for raw in rows:
        if not isinstance(raw, dict) or not MARGINAL_REQUIRED.issubset(raw):
            raise ValueError("marginal row does not match schema")
        row = dict(raw)
        if row["status"] == "evaluated":
            if (
                not MARGINAL_EVALUATED.issubset(row)
                or row["paired_percentage_improvement"] is None
                or not isinstance(row["paired_bootstrap_95_ci"], dict)
            ):
                raise ValueError("evaluated marginal row is incomplete")
        elif row["status"] == "not_evaluated_before_stopping":
            if row["paired_percentage_improvement"] is not None or row["paired_bootstrap_95_ci"] is not None:
                raise ValueError("unevaluated marginal row must have null statistics")
        else:
            raise ValueError("unknown marginal row status")
        normalized.append(row)
    result = dict(data)
    result[key] = normalized
    # Ensure the object can be persisted without implicit NaN values.
    json.dumps(result, allow_nan=False)
    return result


def render_markdown(data: dict[str, Any]) -> str:
    """Pure rendering from a validated JSON adjudication."""
    d = validate_adjudication(data)
    md = [
        "# Phase 4A v3 extension review",
        "",
        f"**Classification:** `{d['classification']}`  ",
        f"**Selected exposure:** {d['selected_exposure']}  ",
        f"**Stop reason:** `{d['stop_reason']}`  ",
        "**Authorization:** downstream, prospective, and Phase 4B authorization remain false.",
        "",
        "## Training and development trajectory",
        "",
        ("| Exposure | Train aligned RMSE | Development aligned RMSE | Development minus training | Training loss |"),
        "|---:|---:|---:|---:|---:|",
    ]
    for row in d["training_trajectory"]:
        md.append(
            f"| {row['exposure']} | {row['training_mean_aligned_rmse_angstrom']:.6f} | "
            f"{row['development_mean_aligned_rmse_angstrom']:.6f} | "
            f"{row['training_development_rmse_gap_angstrom']:.6f} | "
            f"{row['mean_training_loss_this_exposure']:.6g} |"
        )
    key = "marginal_improvement_35_to_50_50_to_60_60_to_70_70_to_80"
    md += [
        "",
        "## Marginal paired development improvement",
        "",
        "| Transition | Paired percentage improvement | Deterministic bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for row in d[key]:
        label = f"{row['from_exposure']} → {row['to_exposure']}"
        if row["status"] == "evaluated":
            ci = row["paired_bootstrap_95_ci"]["ci95_percentile"]
            md.append(f"| {label} | {row['paired_percentage_improvement']:.2%} | {ci[0]:.2%} to {ci[1]:.2%} |")
        else:
            md.append(f"| {label} | not evaluated | not evaluated |")
    md += ["", "## Original gate results", ""]
    for boundary in d["boundaries"]:
        m = boundary["metrics"]
        md += [
            f"### Exposure {boundary['exposure']}",
            "",
            f"Overall development RMSE reduction: {m['rmse_reduction_fraction']:.2%}",
            "",
            "| Gate | Result |",
            "|---|---|",
        ]
        md += [f"| {name} | {str(value).lower()} |" for name, value in m["gate_checks"].items()] + [""]
    return "\n".join(md) + "\n"


def finalize_from_json(staging: Path) -> Path:
    """Repair reports from existing adjudication JSON without model imports/inference."""
    source = staging / "training_metrics.json"
    raw = json.loads(source.read_text())
    key = "marginal_improvement_35_to_50_50_to_60_60_to_70_70_to_80"
    # Upgrade the failed runner's persisted v1 rows from their complete fields.
    # This is schema construction from existing statistics, not recomputation.
    if isinstance(raw.get(key), list):
        fixed = []
        for item in raw[key]:
            row = dict(item)
            if "status" not in row:
                evaluated = (
                    all(
                        name in row
                        for name in (
                            "paired_percentage_improvement",
                            "paired_bootstrap_95_ci",
                            "development_mean_rmse_from",
                            "development_mean_rmse_to",
                        )
                    )
                    and row.get("paired_percentage_improvement") is not None
                )
                row["status"] = "evaluated" if evaluated else "not_evaluated_before_stopping"
            fixed.append(row)
        raw[key] = fixed
    data = validate_adjudication(raw)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Persist the normalized adjudication first. Markdown is a pure projection.
    (staging / "training_metrics.json").write_text(payload)
    (staging / "scientific_review.json").write_text(payload)
    report = staging / "scientific_review.md"
    report.write_text(render_markdown(data))
    return report


def validate_resume_prefix(journal_rows: list[dict[str, Any]], checkpoint_sha256: str, state: dict[str, Any]) -> int:
    """Return next cursor only when the last journal row pins a complete state."""
    required = {
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
    }
    if not journal_rows or journal_rows[-1].get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("resume checkpoint does not match the journal prefix")
    if not required.issubset(state):
        raise ValueError("resume checkpoint lacks full optimizer or RNG state")
    cursor = state.get("schedule_cursor")
    if not isinstance(cursor, int) or cursor != journal_rows[-1].get("global_update"):
        raise ValueError("resume checkpoint cursor does not match journal")
    return len(journal_rows)
