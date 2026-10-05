from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from protein_distance_diffusion.config import load_yaml
from protein_distance_diffusion.training.capacity_benchmark import (
    DEFAULT_MODES,
    run_capacity_benchmark,
    run_capacity_case,
)


def test_capacity_case_records_required_synthetic_metrics() -> None:
    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    result = run_capacity_case(
        config,
        target_length=8,
        mode="learned_geometry_gating",
        seed=91,
        max_rss_mib=4096,
        max_cuda_memory_mib=8192,
        synthetic=True,
    )
    assert result["status"] == "passed"
    assert result["target_length"] == result["actual_length"] == result["padded_length"] == 8
    assert result["batch_size"] == 1
    assert set(result["losses"]) == {"sequence", "geometry", "consistency", "total"}
    assert set(result["gate_statistics"]) == {
        "geometry_to_sequence",
        "sequence_to_geometry",
        "return_geometry_to_sequence",
    }
    assert result["memory"]["peak_rss_mib"] < 4096
    assert result["dataset_inputs_unchanged"] is True
    assert [item["stage"] for item in result["memory_stages"]] == [
        "startup",
        "sample_loading",
        "model_construction",
        "forward",
        "backward",
    ]


def test_capacity_coordinator_isolates_all_modes_and_continues_after_failures(tmp_path: Path) -> None:
    config = load_yaml("configs/e005_sequence_geometry_codesign_synthetic_dry_run.yaml")
    config["model"]["max_length"] = 8
    config["model"]["geometry_model"]["max_length"] = 8
    config_path = tmp_path / "capacity.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report_path = tmp_path / "capacity.json"
    report = run_capacity_benchmark(
        config_path=config_path,
        report_path=report_path,
        lengths=(8, 16),
        modes=DEFAULT_MODES,
        seed=92,
        max_rss_mib=4096,
        max_cuda_memory_mib=8192,
        timeout_seconds=60,
        synthetic=True,
    )

    assert json.loads(report_path.read_text()) == report
    assert report["case_count"] == 6
    assert report["passed_case_count"] == 3
    assert report["failed_case_count"] == 3
    assert report["persistent_checkpoint_written"] is False
    worker_pids = [case["worker_pid"] for case in report["cases"]]
    assert len(set(worker_pids)) == 6
    assert os.getpid() not in worker_pids
    assert all(case["status"] == "passed" for case in report["cases"] if case["target_length"] == 8)
    assert all(case["status"] == "failed" for case in report["cases"] if case["target_length"] == 16)
    assert report["recommended_pilot_schedule"] == [
        {
            "target_length": 8,
            "recommended_batch_size": 1,
            "recommendation": "bounded_pilot_batch_1",
            "failed_modes": [],
            "interpretation": "All three isolated batch-1 cases passed; no evidence supports a larger batch.",
        },
        {
            "target_length": 16,
            "recommended_batch_size": 0,
            "recommendation": "defer",
            "failed_modes": list(DEFAULT_MODES),
            "interpretation": "At least one conditioning mode failed; exclude this length from the first pilot.",
        },
    ]
    assert not list(tmp_path.rglob("*.pt"))
