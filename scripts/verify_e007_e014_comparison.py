"""Close the read-only comparison with coverage and preservation evidence."""

# ruff: noqa: I001, E501, F403, F405
from e007_e014_comparison_protocol import *
import os
import subprocess


def verify():
    protected = json.loads((OUT / "PROTECTED_INPUT_HASHES.json").read_text())
    assert all(sha(name) == digest for name, digest in protected.items())
    inventory = json.loads((OUT / "checkpoint_inventory.json").read_text())
    assert all(sha(r["path"]) == r["sha256"] for r in inventory["checkpoints"])
    assert json.loads((OUT / "matching_assessment.json").read_text())["classification"] == "M2"
    result = json.loads((OUT / "COMPARISON_COMPLETE.json").read_text())
    assert result["primary_classification"] == "C5" and not result["architecture_attribution_allowed"]
    assert not result["full_sampling"] and not result["training"]
    intervals = json.loads((OUT / "bootstrap_intervals.json").read_text())
    assert {r["metric"] for r in intervals["primary"]} == set(PRIMARY)
    assert all(r["bootstrap_replicates"] == 5000 for r in intervals["primary"])
    expected = dict(
        core_metrics=10000,
        geometry_metrics=10000,
        chirality_metrics=10000,
        high_noise_grid=30000,
        marginal_metrics=4 * 18 * 6 * 8,
        gaussian_prefix_metrics=4 * 9 * 6 * 8,
    )
    for name, count in expected.items():
        assert pq.read_metadata(OUT / f"{name}.parquet").num_rows == count, (name, count)
    for name in ("length_strata.parquet", "compute_comparison.json", "replay_numerics.json", "REPORT.md"):
        assert (OUT / name).exists()
    for label in E007_LABELS:
        complete = json.loads((OUT / f"{label}.INFERENCE_COMPLETE.json").read_text())
        assert complete["checkpoint_unchanged"] and complete["model_unchanged"]
        assert complete["paired_conditions"] == 12000 and complete["marginal_conditions"] == 9000
    input_checks = json.loads((OUT / "MATCHED_EVALUATION_INPUT_CHECKS.json").read_text())
    assert all(
        r["paired_noop_max_difference"] == r["gaussian_init_and_independent_references_max_descriptor_difference"] == 0
        for r in input_checks["checks"]
    )
    paths = [
        "v_mse_by_t",
        "local_mse_by_t",
        "torsion_by_t",
        "inversion_rate_by_t",
        "high_noise_v_cosine",
        "high_noise_step_gain",
        "gaussian_prefix_energy",
        "gaussian_prefix_rg",
        "performance_vs_length",
    ]
    assert all((OUT / f"plots/{name}.png").exists() for name in paths)
    assert all(
        str(Path(os.environ[k]).resolve()).startswith("/mnt/d/")
        for k in (
            "TMPDIR",
            "TMP",
            "TEMP",
            "XDG_CACHE_HOME",
            "TORCH_HOME",
            "CUDA_CACHE_PATH",
            "TRITON_CACHE_DIR",
            "MPLCONFIGDIR",
        )
    )
    main = subprocess.check_output(["git", "rev-parse", "main"], text=True).strip()
    e014 = subprocess.check_output(
        ["git", "rev-parse", "e014_unconditional_coordinate_v_scaleup_v1"], text=True
    ).strip()
    assert main == "3b200ffb0389be87cc928c391117ce96bd8b5eb2"
    assert e014 == "053ec32dd256349a6ff8504a0f796167f5c1c727"
    record = dict(
        protected_files_unchanged=len(protected),
        checkpoint_files_unchanged=len(inventory["checkpoints"]),
        required_table_counts=expected,
        matched_evaluation_inputs=True,
        sequence_branch_unchanged=True,
        E014_branch_unchanged=True,
        main_commit=main,
        e014_commit=e014,
        all_task_artifacts_on_D=True,
        cache_and_tmp_paths={
            k: os.environ[k]
            for k in (
                "TMPDIR",
                "TMP",
                "TEMP",
                "XDG_CACHE_HOME",
                "TORCH_HOME",
                "CUDA_CACHE_PATH",
                "TRITON_CACHE_DIR",
                "MPLCONFIGDIR",
            )
        },
        training_launched=False,
        full_sampling_launched=False,
        minimum_sampling_endpoint=350,
        comparison_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    write("FINAL_PRESERVATION_AUDIT.json", record)
    files = {
        str(p.relative_to(OUT)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in OUT.rglob("*")
        if p.is_file() and not p.is_relative_to(OUT / "runtime") and p.name != "FINAL_EVIDENCE_HASHES.json"
    }
    write("FINAL_EVIDENCE_HASHES.json", dict(files=files, root=str(OUT)))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    verify()
