"""Scalable preprocessing state and resume tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_preprocess_module():
    script = Path(__file__).parents[1] / "scripts" / "preprocess_pdb.py"
    spec = importlib.util.spec_from_file_location("preprocess_pdb", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["preprocess_pdb"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREPROCESS = _load_preprocess_module()
PreprocessOptions = PREPROCESS.PreprocessOptions
run_preprocessing = PREPROCESS.run_preprocessing


def _write_poly_gly_mmcif(path: Path, *, length: int, pdb_id: str) -> None:
    rows = []
    for atom_id, resid in enumerate(range(1, length + 1), start=1):
        rows.append(
            f"ATOM {atom_id} C CA . GLY A 1 {resid} ? {float(resid):.3f} 0.000 0.000 1.00 20.00 ? {resid} GLY A CA 1"
        )
    path.write_text(
        "\n".join(
            [
                f"data_{pdb_id}",
                "#",
                "_exptl.method 'X-RAY DIFFRACTION'",
                "_refine.ls_d_res_high 1.50",
                "#",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.id",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_alt_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_entity_id",
                "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "_atom_site.occupancy",
                "_atom_site.B_iso_or_equiv",
                "_atom_site.pdbx_formal_charge",
                "_atom_site.auth_seq_id",
                "_atom_site.auth_comp_id",
                "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id",
                "_atom_site.pdbx_PDB_model_num",
                *rows,
                "#",
            ]
        )
        + "\n"
    )


def _write_config(
    tmp_path: Path,
    raw_dir: Path,
    *,
    name: str,
    max_length: int = 500,
    missing_calpha_policy: str = "reject",
    max_terminal_trim_fraction: str | float | None = None,
) -> Path:
    processed = tmp_path / name / "processed"
    config = tmp_path / name / "preprocess.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    trim_fraction_text = "null" if max_terminal_trim_fraction is None else str(max_terminal_trim_fraction)
    config.write_text(
        "\n".join(
            [
                f"source_dir: {raw_dir}",
                f"samples_dir: {processed / 'samples'}",
                f"manifest_path: {processed / 'manifest.parquet'}",
                f"summary_path: {processed / 'preprocess_summary.json'}",
                f"state_db_path: {processed / 'preprocess_state.sqlite'}",
                "backend: auto",
                "chain_id:",
                "min_length:",
                f"max_length: {max_length}",
                f"missing_calpha_policy: {missing_calpha_policy}",
                f"max_terminal_trim_fraction: {trim_fraction_text}",
                "allowed_methods:",
                "  - X-RAY DIFFRACTION",
                "max_xray_resolution_angstrom: 3.0",
                "max_cryoem_resolution_angstrom: 4.0",
            ]
        )
        + "\n"
    )
    return config


def test_max_terminal_trim_fraction_config_validation(tmp_path: Path) -> None:
    """The optional terminal-trim fraction config accepts null/[0,1] and rejects out-of-range values."""
    raw = tmp_path / "raw"
    raw.mkdir()
    null_config = _write_config(tmp_path, raw, name="null_fraction", max_terminal_trim_fraction=None)
    assert PREPROCESS.normalized_config(PREPROCESS.load_yaml(null_config))["max_terminal_trim_fraction"] is None

    zero_config = _write_config(tmp_path, raw, name="zero_fraction", max_terminal_trim_fraction=0.0)
    assert PREPROCESS.normalized_config(PREPROCESS.load_yaml(zero_config))["max_terminal_trim_fraction"] == 0.0

    one_config = _write_config(tmp_path, raw, name="one_fraction", max_terminal_trim_fraction=1.0)
    assert PREPROCESS.normalized_config(PREPROCESS.load_yaml(one_config))["max_terminal_trim_fraction"] == 1.0

    bad_config = _write_config(tmp_path, raw, name="bad_fraction", max_terminal_trim_fraction=1.01)
    with pytest.raises(ValueError, match="max_terminal_trim_fraction"):
        PREPROCESS.normalized_config(PREPROCESS.load_yaml(bad_config))


def _manifest_without_paths(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "sample_id" in frame.columns:
        frame = frame.sort_values("sample_id").reset_index(drop=True)
    return frame.drop(columns=["path"], errors="ignore")


def _run(config: Path, *, workers: int, **kwargs) -> Path:
    return run_preprocessing(
        config,
        PreprocessOptions(
            workers=workers,
            resume=kwargs.get("resume", True),
            restart=kwargs.get("restart", False),
            retry_failures=kwargs.get("retry_failures", False),
            checkpoint_every=kwargs.get("checkpoint_every", 1),
            show_progress=False,
            interrupt_after_completed=kwargs.get("interrupt_after_completed"),
        ),
    )


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_sequential_and_parallel_preprocessing_are_identical(tmp_path: Path) -> None:
    """Sequential and parallel stateful runs produce identical manifests and summaries."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_poly_gly_mmcif(raw / "aaaa.cif", length=4, pdb_id="AAAA")
    _write_poly_gly_mmcif(raw / "bbbb.cif", length=5, pdb_id="BBBB")
    _write_poly_gly_mmcif(raw / "long.cif", length=6, pdb_id="LONG")
    sequential = _write_config(tmp_path, raw, name="sequential", max_length=5)
    parallel = _write_config(tmp_path, raw, name="parallel", max_length=5)

    seq_manifest = _run(sequential, workers=1, restart=True)
    par_manifest = _run(parallel, workers=2, restart=True)

    pd.testing.assert_frame_equal(_manifest_without_paths(seq_manifest), _manifest_without_paths(par_manifest))
    seq_summary = json.loads((seq_manifest.parent / "preprocess_summary.json").read_text())
    par_summary = json.loads((par_manifest.parent / "preprocess_summary.json").read_text())
    assert seq_summary["rejection_counts"] == par_summary["rejection_counts"]


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_interruption_then_resume_equals_uninterrupted_run(tmp_path: Path) -> None:
    """A partial run followed by resume matches a clean uninterrupted run."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for pdb_id in ["AAAA", "BBBB", "CCCC"]:
        _write_poly_gly_mmcif(raw / f"{pdb_id.lower()}.cif", length=4, pdb_id=pdb_id)
    clean = _write_config(tmp_path, raw, name="clean")
    resumed = _write_config(tmp_path, raw, name="resumed")

    clean_manifest = _run(clean, workers=1, restart=True)
    _run(resumed, workers=1, restart=True, interrupt_after_completed=1)
    resumed_manifest = _run(resumed, workers=1, resume=True)

    pd.testing.assert_frame_equal(_manifest_without_paths(clean_manifest), _manifest_without_paths(resumed_manifest))


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_relaxed_preprocessing_resume_equals_uninterrupted_run(tmp_path: Path) -> None:
    """Relaxed missing-CA policy participates in resumable preprocessing equivalence."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for pdb_id in ["AAAA", "BBBB", "CCCC"]:
        _write_poly_gly_mmcif(raw / f"{pdb_id.lower()}.cif", length=20, pdb_id=pdb_id)
    clean = _write_config(tmp_path, raw, name="relaxed_clean", missing_calpha_policy="trim_terminal")
    resumed = _write_config(tmp_path, raw, name="relaxed_resumed", missing_calpha_policy="trim_terminal")

    clean_manifest = _run(clean, workers=1, restart=True)
    _run(resumed, workers=1, restart=True, interrupt_after_completed=1)
    resumed_manifest = _run(resumed, workers=1, resume=True)

    pd.testing.assert_frame_equal(_manifest_without_paths(clean_manifest), _manifest_without_paths(resumed_manifest))


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_resume_creates_no_duplicate_manifest_rows(tmp_path: Path) -> None:
    """Repeated resume runs do not duplicate samples in SQLite or the manifest."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_poly_gly_mmcif(raw / "aaaa.cif", length=4, pdb_id="AAAA")
    config = _write_config(tmp_path, raw, name="resume")

    manifest = _run(config, workers=1, restart=True)
    manifest = _run(config, workers=1, resume=True)

    frame = pd.read_parquet(manifest)
    assert frame["sample_id"].is_unique
    conn = sqlite3.connect(manifest.parent / "preprocess_state.sqlite")
    try:
        assert conn.execute("SELECT COUNT(*) FROM manifest_rows").fetchone()[0] == len(frame)
    finally:
        conn.close()


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_resume_rejects_config_mismatch(tmp_path: Path) -> None:
    """Changing preprocessing behavior refuses resume unless restarted."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_poly_gly_mmcif(raw / "aaaa.cif", length=4, pdb_id="AAAA")
    config = _write_config(tmp_path, raw, name="mismatch", max_length=500)
    _run(config, workers=1, restart=True)
    config.write_text(config.read_text().replace("max_length: 500", "max_length: 499"))

    with pytest.raises(ValueError, match="config hash differs"):
        _run(config, workers=1, resume=True)


def test_retry_failures_only_retries_failed_sources(tmp_path: Path) -> None:
    """Technical failures are skipped by default and retried only with retry_failures."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "bad.cif").write_text("not a valid mmCIF\n")
    config = _write_config(tmp_path, raw, name="failures")

    manifest = _run(config, workers=1, restart=True)
    conn = sqlite3.connect(manifest.parent / "preprocess_state.sqlite")
    try:
        assert conn.execute("SELECT status, attempt_count FROM source_files").fetchone() == ("failed", 1)
    finally:
        conn.close()

    _run(config, workers=1, resume=True, retry_failures=False)
    conn = sqlite3.connect(manifest.parent / "preprocess_state.sqlite")
    try:
        assert conn.execute("SELECT attempt_count FROM source_files").fetchone()[0] == 1
    finally:
        conn.close()

    _run(config, workers=1, resume=True, retry_failures=True)
    conn = sqlite3.connect(manifest.parent / "preprocess_state.sqlite")
    try:
        assert conn.execute("SELECT attempt_count FROM source_files").fetchone()[0] == 2
    finally:
        conn.close()


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_sample_writes_are_atomic(tmp_path: Path) -> None:
    """Successful sample writes leave completed NPZs and no temporary sample files."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_poly_gly_mmcif(raw / "aaaa.cif", length=4, pdb_id="AAAA")
    config = _write_config(tmp_path, raw, name="atomic")
    manifest = _run(config, workers=1, restart=True)
    frame = pd.read_parquet(manifest)

    assert Path(frame.iloc[0]["path"]).exists()
    assert not list((manifest.parent / "samples").glob("*.tmp"))


@pytest.mark.skipif(importlib.util.find_spec("gemmi") is None, reason="Gemmi is not installed")
def test_real_pilot_resume_smoke_when_local_files_exist(tmp_path: Path) -> None:
    """A small copied pilot subset can be interrupted, resumed, and matched to a clean run."""
    pilot_files = sorted((Path(__file__).parents[1] / "data" / "pilot" / "raw" / "mmcif").glob("*.cif.gz"))[:2]
    if len(pilot_files) < 2:
        pytest.skip("Local pilot mmCIF files are not available")
    raw = tmp_path / "raw"
    raw.mkdir()
    for source in pilot_files:
        shutil.copy2(source, raw / source.name)
    clean = _write_config(tmp_path, raw, name="pilot_clean")
    resumed = _write_config(tmp_path, raw, name="pilot_resumed")

    clean_manifest = _run(clean, workers=1, restart=True)
    _run(resumed, workers=1, restart=True, interrupt_after_completed=1)
    resumed_manifest = _run(resumed, workers=1, resume=True)

    pd.testing.assert_frame_equal(_manifest_without_paths(clean_manifest), _manifest_without_paths(resumed_manifest))
