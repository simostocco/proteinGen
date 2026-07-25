"""Streaming train-normalization statistics tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import protein_distance_diffusion.data.statistics as statistics_module
from protein_distance_diffusion.data.dataset import DistanceMapDataset
from protein_distance_diffusion.data.statistics import (
    HistogramConfig,
    _empty_state,
    compute_scale_statistics,
    write_normalization,
)


def _write_npz(path: Path, matrix: np.ndarray, *, residue_mask: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = matrix.shape[0]
    np.savez_compressed(
        path,
        sample_id=np.asarray(path.stem),
        sequence_tokens=np.arange(size, dtype=np.int64),
        residue_mask=np.ones(size, dtype=np.bool_) if residue_mask is None else residue_mask,
        distance_matrix=matrix.astype(np.float32),
        metadata=np.asarray("{}"),
    )


def _manifest(tmp_path: Path, rows: list[tuple[str, np.ndarray, int | None, np.ndarray | None]]) -> Path:
    manifest_rows = []
    for sample_id, matrix, length, residue_mask in rows:
        path = tmp_path / "samples" / f"{sample_id}.npz"
        _write_npz(path, matrix, residue_mask=residue_mask)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "pdb_id": sample_id[:4].upper(),
                "chain_id": "A",
                "sequence": "A" * (length if length is not None else matrix.shape[0]),
                "length": length if length is not None else matrix.shape[0],
                "path": str(path),
            }
        )
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(manifest_rows).to_parquet(manifest, index=False)
    return manifest


def _upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu(np.ones(matrix.shape, dtype=bool), k=1)]


def test_streaming_percentile_matches_numpy_within_bin_width(tmp_path: Path) -> None:
    """Histogram percentile is deterministic and within one configured bin width."""
    matrix = np.zeros((10, 10), dtype=np.float32)
    matrix[np.triu_indices(10, k=1)] = np.arange(1, 46, dtype=np.float32)
    matrix = matrix + matrix.T
    manifest = _manifest(tmp_path, [("s1", matrix, None, None)])
    stats = compute_scale_statistics(
        manifest,
        percentile=95.0,
        histogram_bin_width_angstrom=1.0,
        histogram_max_distance_angstrom=100.0,
        output=tmp_path / "normalization.json",
        workers=1,
        chunk_size=1,
    )
    exact = float(np.percentile(_upper_values(matrix), 95.0))
    assert abs(stats["scale"] - exact) <= stats["percentile_error_bound_angstrom"]
    assert stats["valid_distance_count"] == 45


def test_diagonal_and_lower_triangle_are_excluded(tmp_path: Path) -> None:
    """Large diagonal/lower-triangle values do not affect the train scale."""
    matrix = np.array(
        [
            [999, 1, 2],
            [777, 999, 3],
            [777, 777, 999],
        ],
        dtype=np.float32,
    )
    manifest = _manifest(tmp_path, [("s1", matrix, None, None)])
    stats = compute_scale_statistics(
        manifest,
        percentile=100.0,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=10.0,
        output=tmp_path / "normalization.json",
        workers=1,
    )
    assert stats["valid_distance_count"] == 3
    assert stats["scale"] <= 3.1


def test_padding_is_excluded_by_residue_mask(tmp_path: Path) -> None:
    """Residue masks exclude padded rows and columns from valid-pair statistics."""
    matrix = np.array(
        [
            [0, 1, 2, 999],
            [1, 0, 3, 999],
            [2, 3, 0, 999],
            [999, 999, 999, 0],
        ],
        dtype=np.float32,
    )
    manifest = _manifest(tmp_path, [("s1", matrix, 3, np.array([True, True, True, False]))])
    stats = compute_scale_statistics(
        manifest,
        percentile=100.0,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=10.0,
        output=tmp_path / "normalization.json",
        workers=1,
    )
    assert stats["valid_distance_count"] == 3
    assert stats["overflow_distance_count"] == 0


def test_histogram_memory_is_fixed_by_bin_count() -> None:
    """Histogram accumulator memory depends on bins, not streamed pair count."""
    small = _empty_state(HistogramConfig(histogram_bin_width_angstrom=1.0, histogram_max_distance_angstrom=10.0))
    large = _empty_state(HistogramConfig(histogram_bin_width_angstrom=1.0, histogram_max_distance_angstrom=10.0))
    small.valid_distance_count = 10
    large.valid_distance_count = 10_000_000
    assert small.histogram.nbytes == large.histogram.nbytes == 10 * np.dtype(np.int64).itemsize


def test_sequential_and_parallel_outputs_match(tmp_path: Path) -> None:
    """Chunked multiprocessing returns the same histogram-derived result as sequential execution."""
    rows = []
    for idx in range(6):
        base = float(idx + 1)
        matrix = np.array([[0, base, base + 1], [base, 0, base + 2], [base + 1, base + 2, 0]], dtype=np.float32)
        rows.append((f"s{idx}", matrix, None, None))
    manifest = _manifest(tmp_path, rows)
    seq = compute_scale_statistics(
        manifest,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=20.0,
        output=tmp_path / "seq.json",
        workers=1,
        chunk_size=2,
    )
    par = compute_scale_statistics(
        manifest,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=20.0,
        output=tmp_path / "par.json",
        workers=2,
        chunk_size=2,
    )
    comparable = [
        "scale",
        "valid_distance_count",
        "mean_distance_angstrom",
        "std_distance_angstrom",
        "observed_max_distance_angstrom",
    ]
    assert {key: seq[key] for key in comparable} == {key: par[key] for key in comparable}


def test_resume_matches_uninterrupted_and_config_mismatch_rejects(tmp_path: Path) -> None:
    """Completed checkpoint state can be resumed, but incompatible config is refused."""
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    manifest = _manifest(
        tmp_path,
        [("s1", matrix, None, None), ("s2", matrix + np.eye(3, dtype=np.float32), None, None)],
    )
    output = tmp_path / "normalization.json"
    first = compute_scale_statistics(
        manifest,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=10.0,
        output=output,
        workers=1,
        checkpoint_every=1,
        chunk_size=1,
    )
    resumed = compute_scale_statistics(
        manifest,
        histogram_bin_width_angstrom=0.1,
        histogram_max_distance_angstrom=10.0,
        output=output,
        workers=1,
        resume=True,
        chunk_size=1,
    )
    assert first["scale"] == resumed["scale"]
    with pytest.raises(ValueError, match="configuration differs"):
        compute_scale_statistics(
            manifest,
            histogram_bin_width_angstrom=0.2,
            histogram_max_distance_angstrom=10.0,
            output=output,
            workers=1,
            resume=True,
            chunk_size=1,
        )


def test_interruption_writes_checkpoint_but_not_final_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C leaves resumable state but does not create final normalization JSON."""
    matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)
    manifest = _manifest(tmp_path, [("s1", matrix, None, None), ("s2", matrix, None, None)])
    output = tmp_path / "normalization.json"
    original = statistics_module._chunk_statistics
    calls = {"count": 0}

    def interrupt_second_chunk(chunk, config):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return original(chunk, config)

    monkeypatch.setattr(statistics_module, "_chunk_statistics", interrupt_second_chunk)
    with pytest.raises(KeyboardInterrupt):
        write_normalization(output=output, train_manifest=manifest, workers=1, checkpoint_every=1, chunk_size=1)
    assert not output.exists()
    assert output.with_suffix(".state.npz").exists()
    assert output.with_suffix(".state.json").exists()


def test_corrupt_missing_and_non_square_samples_fail_with_diagnostics(tmp_path: Path) -> None:
    """Invalid training samples are recorded in checkpoint JSON and fail by default."""
    missing = tmp_path / "missing.npz"
    non_square = tmp_path / "non_square.npz"
    np.savez_compressed(non_square, distance_matrix=np.ones((2, 3), dtype=np.float32))
    manifest = tmp_path / "train.parquet"
    pd.DataFrame(
        [
            {"sample_id": "missing", "length": 2, "path": str(missing)},
            {"sample_id": "bad", "length": 2, "path": str(non_square)},
        ]
    ).to_parquet(manifest, index=False)
    output = tmp_path / "normalization.json"
    with pytest.raises(RuntimeError, match="Rejected 2"):
        write_normalization(manifest, output, workers=1, chunk_size=1)
    diagnostics = json.loads(output.with_suffix(".state.json").read_text())
    reasons = {item["reason"] for item in diagnostics["rejected_samples"]}
    assert reasons == {"missing_npz", "non_square_distance_matrix"}
    assert not output.exists()


def test_overflow_is_detected_not_clipped(tmp_path: Path) -> None:
    """Distances outside the configured histogram range fail by default."""
    matrix = np.array([[0, 1, 20], [1, 0, 2], [20, 2, 0]], dtype=np.float32)
    manifest = _manifest(tmp_path, [("s1", matrix, None, None)])
    with pytest.raises(RuntimeError, match="Histogram overflow"):
        compute_scale_statistics(
            manifest,
            histogram_bin_width_angstrom=0.1,
            histogram_max_distance_angstrom=10.0,
            output=tmp_path / "normalization.json",
            workers=1,
        )


def test_empty_one_sample_and_training_loader_compatibility(tmp_path: Path) -> None:
    """Empty manifests fail clearly; one-sample output remains loader-compatible."""
    empty = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["sample_id", "length", "path"]).to_parquet(empty, index=False)
    with pytest.raises(ValueError, match="no samples"):
        compute_scale_statistics(empty, output=tmp_path / "empty.json", workers=1)

    matrix = np.array([[0, 1], [1, 0]], dtype=np.float32)
    manifest = _manifest(tmp_path, [("s1", matrix, None, None)])
    output = tmp_path / "normalization.json"
    write_normalization(manifest, output, workers=1)
    normalization = json.loads(output.read_text())
    assert normalization["mode"] == "scale"
    assert normalization["scale"] > 0
    item = DistanceMapDataset(manifest, normalization)[0]
    assert item["distance_matrix"].shape == (2, 2)
