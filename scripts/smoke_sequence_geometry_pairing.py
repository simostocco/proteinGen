#!/usr/bin/env python3
"""Run a bounded, read-only smoke test of a completed pairing dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch

from protein_distance_diffusion.data.sequence_geometry import (
    DATASET_MODES,
    PAIRING_SCHEMA_VERSION,
    VOCABULARY_VERSION,
    SequenceGeometryDataset,
    SequenceGeometryVocabulary,
    build_geometry_corruption,
    collate_sequence_geometry,
)
from protein_distance_diffusion.evaluation.sequence_readiness import sha256_file

DATASETS = (
    "all_pairs",
    "eligible_train",
    "eligible_validation",
    "excluded_pairs",
    "pairing_ineligible",
    "pairing_eligible_original_split_excluded",
)
MEMBERSHIP_DATASETS = ("all_pairs", "eligible_train", "eligible_validation", "excluded_pairs")
LENGTH_BINS = ((20, 64), (65, 128), (129, 256), (257, 384), (385, 500))
CORRUPTION_CONFIGS = {
    "additive_symmetric_noise": {
        "type": "additive_symmetric_noise",
        "standard_deviation": 0.2,
    },
    "long_range_pair_mask": {
        "type": "long_range_pair_mask",
        "minimum_separation": 2,
        "mask_probability": 0.5,
    },
    "contact_deletion": {
        "type": "contact_deletion",
        "contact_threshold": 8.0,
        "deletion_probability": 0.5,
    },
    "low_rank_distance_distortion": {
        "type": "low_rank_distance_distortion",
        "rank": 3,
        "strength": 0.25,
    },
    "mixture": {
        "type": "mixture",
        "transforms": [
            {"type": "additive_symmetric_noise", "standard_deviation": 0.1},
            {"type": "long_range_pair_mask", "minimum_separation": 2, "mask_probability": 0.25},
        ],
    },
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _peak_rss_mib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024


def _guard_memory(max_memory_mib: int) -> None:
    if _rss_mib() > max_memory_mib:
        raise MemoryError(f"Smoke RSS exceeded {max_memory_mib} MiB")


def _length_bin(length: int) -> str:
    for lower, upper in LENGTH_BINS:
        if lower <= length <= upper:
            return f"{lower}-{upper}"
    return "outside-20-500"


def _stable_rank(seed: int, split: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{sample_id}".encode()).hexdigest()


def _scan_batches(path: Path, columns: list[str], batch_size: int):
    scanner = ds.dataset(str(path), format="parquet").scanner(
        columns=columns,
        batch_size=batch_size,
        use_threads=False,
    )
    yield from scanner.to_batches()


def _verify_metadata(dataset_dir: Path, batch_size: int, max_memory_mib: int) -> dict[str, Any]:
    protocol_path = dataset_dir / "protocol.json"
    schema_path = dataset_dir / "schema.json"
    vocabulary_path = dataset_dir / "vocabulary.json"
    ledger_path = dataset_dir / "input_hashes.sha256"
    protocol = json.loads(protocol_path.read_text())
    schema = json.loads(schema_path.read_text())
    vocabulary = json.loads(vocabulary_path.read_text())
    expected_vocabulary = SequenceGeometryVocabulary().as_dict()
    if protocol.get("status") != "completed":
        raise ValueError("Pairing protocol is not completed")
    if protocol.get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise ValueError("Pairing protocol schema version is unsupported")
    if protocol.get("input_hashes_preserved") is not True:
        raise ValueError("Pairing protocol does not attest preserved inputs")
    if schema.get("schema_version") != PAIRING_SCHEMA_VERSION or not schema.get("partitioned_parquet"):
        raise ValueError("Pairing schema metadata is inconsistent")
    if vocabulary != expected_vocabulary or vocabulary.get("version") != VOCABULARY_VERSION:
        raise ValueError("Pairing vocabulary is inconsistent")

    ledger = {}
    for line in ledger_path.read_text().splitlines():
        digest, path = line.split("  ", 1)
        ledger[path] = digest
    if ledger != protocol.get("input_hashes"):
        raise ValueError("Input hash ledger does not match the protocol")
    for path_text, expected_hash in sorted(ledger.items()):
        path = Path(path_text)
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Pairing input hash mismatch: {path}")
        _guard_memory(max_memory_mib)

    partition_metadata = protocol.get("partitions", [])
    expected_paths = set()
    rows_by_dataset: defaultdict[str, int] = defaultdict(int)
    schema_columns = schema.get("columns", {})
    for item in partition_metadata:
        path = dataset_dir / item["path"]
        expected_paths.add(path.resolve())
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"Pairing partition hash mismatch: {path}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != int(item["row_count"]):
            raise ValueError(f"Pairing partition row-count mismatch: {path}")
        actual_schema = {field.name: str(field.type) for field in parquet.schema_arrow}
        if actual_schema != schema_columns:
            raise ValueError(f"Pairing partition schema mismatch: {path}")
        rows_by_dataset[str(item["dataset"])] += int(item["row_count"])
        _guard_memory(max_memory_mib)
    actual_paths = {
        path.resolve() for name in DATASETS for path in (dataset_dir / f"{name}.parquet").glob("part-*.parquet")
    }
    if expected_paths != actual_paths:
        raise ValueError("Published partition set differs from protocol metadata")
    return {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "schema_sha256": sha256_file(schema_path),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "input_hash_count": len(ledger),
        "partition_count": len(partition_metadata),
        "rows_by_dataset": dict(sorted(rows_by_dataset.items())),
        "batch_size": batch_size,
    }


def _index_membership(
    dataset_dir: Path,
    connection: sqlite3.Connection,
    *,
    batch_size: int,
    max_memory_mib: int,
) -> dict[str, int]:
    connection.execute(
        "CREATE TABLE membership (sample_id TEXT PRIMARY KEY, in_all INTEGER NOT NULL DEFAULT 0, "
        "in_train INTEGER NOT NULL DEFAULT 0, in_validation INTEGER NOT NULL DEFAULT 0, "
        "in_excluded INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID"
    )
    column_by_dataset = {
        "all_pairs": "in_all",
        "eligible_train": "in_train",
        "eligible_validation": "in_validation",
        "excluded_pairs": "in_excluded",
    }
    counts = {}
    for name in MEMBERSHIP_DATASETS:
        column = column_by_dataset[name]
        count = 0
        columns = ["sample_id", "schema_version"] if name == "all_pairs" else ["sample_id"]
        for batch in _scan_batches(dataset_dir / f"{name}.parquet", columns, batch_size):
            rows = [(str(value),) for value in batch.column(0).to_pylist()]
            if name == "all_pairs":
                versions = {str(value) for value in batch.column(1).to_pylist()}
                if versions != {PAIRING_SCHEMA_VERSION}:
                    raise ValueError("all_pairs contains an unsupported row schema version")
                try:
                    connection.executemany("INSERT INTO membership(sample_id,in_all) VALUES (?,1)", rows)
                except sqlite3.IntegrityError as error:
                    raise ValueError("Duplicate sample_id in all_pairs") from error
            else:
                for (sample_id,) in rows:
                    updated = connection.execute(
                        f"UPDATE membership SET {column}=1 WHERE sample_id=? AND {column}=0",
                        (sample_id,),
                    ).rowcount
                    if updated != 1:
                        raise ValueError(f"Duplicate or unknown sample_id in {name}: {sample_id}")
            count += len(rows)
            connection.commit()
            _guard_memory(max_memory_mib)
        counts[name] = count
    overlap = connection.execute(
        "SELECT sample_id FROM membership WHERE in_train+in_validation+in_excluded != 1 LIMIT 1"
    ).fetchone()
    if overlap:
        raise ValueError(f"Pairing membership is overlapping or incomplete: {overlap[0]}")
    if connection.execute("SELECT COUNT(*) FROM membership WHERE in_all != 1").fetchone()[0]:
        raise ValueError("Pairing membership contains rows outside all_pairs")
    return counts


def _select_rows(
    dataset_dir: Path,
    split: str,
    *,
    maximum: int,
    seed: int,
    batch_size: int,
    max_memory_mib: int,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "sequence",
        "sequence_length",
        "matrix_length",
        "matrix_path",
        "matrix_sha256",
        "schema_version",
        "practical_training_eligibility",
        "actual_length",
        "experimental_method",
        "pairing_classification",
        "normalization_file",
        "normalization_sha256",
    ]
    best: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    path = dataset_dir / f"eligible_{split}.parquet"
    for batch in _scan_batches(path, columns, batch_size):
        for row in batch.to_pylist():
            sample_id = str(row["sample_id"])
            rank = _stable_rank(seed, split, sample_id)
            facets = (
                ("length", _length_bin(int(row["actual_length"]))),
                ("method", str(row["experimental_method"])),
                ("classification", str(row["pairing_classification"])),
            )
            for facet in facets:
                current = best.get(facet)
                if current is None or rank < current[0]:
                    best[facet] = (rank, row)
        _guard_memory(max_memory_mib)
    candidates = {str(row["sample_id"]): row for _, row in best.values()}
    selected: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    while candidates and len(selected) < maximum:
        ranked = []
        for sample_id, row in candidates.items():
            facets = {
                ("length", _length_bin(int(row["actual_length"]))),
                ("method", str(row["experimental_method"])),
                ("classification", str(row["pairing_classification"])),
            }
            ranked.append((-len(facets - covered), _stable_rank(seed, split, sample_id), sample_id, facets))
        _, _, sample_id, facets = min(ranked)
        selected.append(candidates.pop(sample_id))
        covered.update(facets)
    available = set(best)
    if not available <= covered:
        raise ValueError(
            f"--samples-per-split={maximum} cannot cover all observed {split} strata; "
            f"missing {sorted(available - covered)}"
        )
    return pd.DataFrame(selected).sort_values("sample_id", kind="stable").reset_index(drop=True)


def _assert_item(item: dict[str, Any], vocabulary: SequenceGeometryVocabulary) -> None:
    length = int(item["length"])
    tokens = item["sequence_token_ids"]
    matrix = item["distance_matrix"]
    pair_mask = item["pair_mask"]
    if tokens.shape != (length,) or matrix.shape != (length, length) or pair_mask.shape != matrix.shape:
        raise ValueError(f"Loader shape mismatch for {item['sample_id']}")
    if not torch.isfinite(matrix).all() or not torch.equal(pair_mask, pair_mask.T):
        raise ValueError(f"Non-finite matrix or asymmetric mask for {item['sample_id']}")
    if not torch.allclose(matrix, matrix.T) or not torch.equal(torch.diag(matrix), torch.zeros(length)):
        raise ValueError(f"Asymmetric matrix or nonzero diagonal for {item['sample_id']}")
    if (matrix[pair_mask] < 0).any():
        raise ValueError(f"Negative distance in valid geometry for {item['sample_id']}")
    if not torch.equal(matrix[~pair_mask], torch.zeros_like(matrix[~pair_mask])):
        raise ValueError(f"Masked matrix entries are nonzero for {item['sample_id']}")
    if int(tokens.min()) < 2 or int(tokens.max()) >= len(vocabulary.tokens):
        raise ValueError(f"Residue token is outside the canonical vocabulary for {item['sample_id']}")


def _exercise_loaders(
    selected: pd.DataFrame,
    *,
    temporary_dir: Path,
    seed: int,
    max_memory_mib: int,
) -> dict[str, Any]:
    vocabulary = SequenceGeometryVocabulary()
    manifest = temporary_dir / "bounded_samples.parquet"
    selected.to_parquet(manifest, index=False)
    loaded_items = 0
    started = time.monotonic()
    mode_results = {}
    for mode in sorted(DATASET_MODES - {"corrupted_real_geometry"}):
        probability = 0.5 if mode == "geometry_conditioned_with_dropout" else 0.0
        dataset = SequenceGeometryDataset(
            manifest,
            mode=mode,
            conditioning_dropout_probability=probability,
            seed=seed,
        )
        first = dataset[0]
        repeated = dataset[0]
        _assert_item(first, vocabulary)
        if not torch.equal(first["distance_matrix"], repeated["distance_matrix"]):
            raise ValueError(f"Loader mode is nondeterministic for identical seed: {mode}")
        loaded_items += 2
        mode_results[mode] = {
            "geometry_conditioning_flag": first["geometry_conditioning_flag"],
            "pair_mask_count": int(first["pair_mask"].sum()),
        }

    dropout_off = SequenceGeometryDataset(
        manifest,
        mode="geometry_conditioned_with_dropout",
        conditioning_dropout_probability=0.0,
        seed=seed,
    )[0]
    dropout_on = SequenceGeometryDataset(
        manifest,
        mode="geometry_conditioned_with_dropout",
        conditioning_dropout_probability=1.0,
        seed=seed,
    )[0]
    if not dropout_off["geometry_conditioning_flag"] or dropout_on["geometry_conditioning_flag"]:
        raise ValueError("Conditioning dropout endpoints are incorrect")
    if dropout_on["pair_mask"].any() or dropout_on["distance_matrix"].any():
        raise ValueError("Dropped geometry was not fully masked")
    loaded_items += 2

    corruption_results = {}
    for name, config in CORRUPTION_CONFIGS.items():
        corruption = build_geometry_corruption(config)
        first = SequenceGeometryDataset(manifest, mode="corrupted_real_geometry", corruption=corruption, seed=seed)[0]
        repeated = SequenceGeometryDataset(manifest, mode="corrupted_real_geometry", corruption=corruption, seed=seed)[
            0
        ]
        changed_seed = SequenceGeometryDataset(
            manifest, mode="corrupted_real_geometry", corruption=corruption, seed=seed + 1
        )[0]
        _assert_item(first, vocabulary)
        if not torch.equal(first["distance_matrix"], repeated["distance_matrix"]):
            raise ValueError(f"Corruption is nondeterministic for identical seed: {name}")
        seed_changed_output = not (
            torch.equal(first["distance_matrix"], changed_seed["distance_matrix"])
            and torch.equal(first["pair_mask"], changed_seed["pair_mask"])
        )
        if name != "low_rank_distance_distortion" and not seed_changed_output:
            raise ValueError(f"Stochastic corruption did not change with seed: {name}")
        corruption_results[name] = {
            "seed_changed_output": seed_changed_output,
            "deterministic_by_design": name == "low_rank_distance_distortion",
        }
        loaded_items += 3

    conditioned = SequenceGeometryDataset(manifest, mode="geometry_conditioned", seed=seed)
    mixed = [conditioned[index] for index in range(len(conditioned))]
    for item in mixed:
        _assert_item(item, vocabulary)
    batch = collate_sequence_geometry(mixed, pad_id=vocabulary.pad_id)
    maximum_length = int(batch["lengths"].max())
    if batch["sequence_token_ids"].shape != (len(mixed), maximum_length):
        raise ValueError("Mixed-length token collation shape is incorrect")
    for index, length in enumerate(batch["lengths"].tolist()):
        if not torch.all(batch["sequence_token_ids"][index, length:] == vocabulary.pad_id):
            raise ValueError("PAD token was not used for sequence padding")
        if batch["sequence_mask"][index, length:].any() or batch["pair_mask"][index, length:].any():
            raise ValueError("Padded positions remain unmasked")
    masked_tokens = batch["sequence_token_ids"].clone()
    masked_tokens[:, 0] = vocabulary.mask_id
    if not torch.all(masked_tokens[:, 0] == vocabulary.mask_id) or vocabulary.mask_id == vocabulary.pad_id:
        raise ValueError("MASK and PAD token semantics are inconsistent")
    loaded_items += len(mixed)

    normalization_paths = set(selected["normalization_file"].astype(str))
    normalization_hashes = set(selected["normalization_sha256"].astype(str))
    if len(normalization_paths) != 1 or len(normalization_hashes) != 1:
        raise ValueError("Selected rows disagree on normalization provenance")
    normalization_path = Path(next(iter(normalization_paths)))
    if sha256_file(normalization_path) != next(iter(normalization_hashes)):
        raise ValueError("Normalization file hash does not match selected rows")
    normalization = json.loads(normalization_path.read_text())
    scale = float(normalization.get("scale", 0))
    if normalization.get("mode") != "scale" or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Unsupported or invalid distance normalization")
    normalized = mixed[0]["distance_matrix"] / scale
    if not torch.isfinite(normalized).all():
        raise ValueError("Normalized sampled geometry is non-finite")
    _guard_memory(max_memory_mib)
    elapsed = time.monotonic() - started
    return {
        "supported_modes": sorted(DATASET_MODES),
        "mode_results": mode_results,
        "corruption_results": corruption_results,
        "loaded_item_count": loaded_items,
        "loader_elapsed_seconds": elapsed,
        "loader_items_per_second": loaded_items / elapsed,
        "normalization": {
            "path": str(normalization_path),
            "sha256": next(iter(normalization_hashes)),
            "mode": "scale",
            "scale": scale,
        },
        "pad_id": vocabulary.pad_id,
        "mask_id": vocabulary.mask_id,
    }


def run_smoke(
    *,
    dataset_dir: Path,
    report_path: Path,
    samples_per_split: int = 12,
    seed: int = 2026,
    batch_size: int = 4_096,
    max_memory_mib: int = 2_048,
) -> dict[str, Any]:
    if samples_per_split < 1 or samples_per_split > 32:
        raise ValueError("samples_per_split must be between 1 and 32")
    if batch_size < 1 or batch_size > 4_096:
        raise ValueError("batch_size must be between 1 and 4096")
    if max_memory_mib < 512 or max_memory_mib > 2_048:
        raise ValueError("max_memory_mib must be between 512 and 2048")
    started = time.monotonic()
    metadata_hashes_before = {
        name: sha256_file(dataset_dir / name)
        for name in ("protocol.json", "schema.json", "vocabulary.json", "input_hashes.sha256")
    }
    metadata = _verify_metadata(dataset_dir, batch_size, max_memory_mib)
    with tempfile.TemporaryDirectory(prefix="sequence-geometry-smoke-") as temporary:
        temporary_dir = Path(temporary)
        with sqlite3.connect(temporary_dir / "membership.sqlite") as connection:
            membership_counts = _index_membership(
                dataset_dir,
                connection,
                batch_size=batch_size,
                max_memory_mib=max_memory_mib,
            )
        protocol_counts = metadata["protocol"]["validated_membership_counts"]
        expected_counts = {
            "all_pairs": protocol_counts["all_pair_count"],
            "eligible_train": protocol_counts["eligible_train_count"],
            "eligible_validation": protocol_counts["eligible_validation_count"],
            "excluded_pairs": protocol_counts["derived_dataset_excluded_count"],
        }
        if membership_counts != expected_counts:
            raise ValueError(
                f"Dataset membership counts do not match protocol: {membership_counts} versus {expected_counts}"
            )
        if sum(membership_counts[name] for name in MEMBERSHIP_DATASETS[1:]) != membership_counts["all_pairs"]:
            raise ValueError("Eligible and excluded membership counts do not partition all_pairs")

        selected_frames = []
        selection_report = {}
        for split in ("train", "validation"):
            frame = _select_rows(
                dataset_dir,
                split,
                maximum=samples_per_split,
                seed=seed,
                batch_size=batch_size,
                max_memory_mib=max_memory_mib,
            )
            selected_frames.append(frame)
            selection_report[split] = {
                "sample_ids": frame["sample_id"].astype(str).tolist(),
                "length_bins": sorted({_length_bin(int(value)) for value in frame["actual_length"]}),
                "experimental_methods": sorted(set(frame["experimental_method"].astype(str))),
                "pairing_classifications": sorted(set(frame["pairing_classification"].astype(str))),
            }
        selected = pd.concat(selected_frames, ignore_index=True)
        for row in selected.to_dict("records"):
            matrix_path = Path(str(row["matrix_path"]))
            if not matrix_path.is_file() or sha256_file(matrix_path) != str(row["matrix_sha256"]):
                raise ValueError(f"Selected matrix hash mismatch: {matrix_path}")
        loader = _exercise_loaders(
            selected,
            temporary_dir=temporary_dir,
            seed=seed,
            max_memory_mib=max_memory_mib,
        )

    metadata_hashes_after = {name: sha256_file(dataset_dir / name) for name in metadata_hashes_before}
    if metadata_hashes_before != metadata_hashes_after:
        raise RuntimeError("Pairing metadata changed during the read-only smoke")
    elapsed = time.monotonic() - started
    peak_rss_mib = _peak_rss_mib()
    if peak_rss_mib > max_memory_mib:
        raise MemoryError(f"Smoke peak RSS {peak_rss_mib:.1f} MiB exceeded {max_memory_mib} MiB")
    report = {
        "status": "passed",
        "dataset_dir": str(dataset_dir.resolve()),
        "schema_version": PAIRING_SCHEMA_VERSION,
        "vocabulary_version": VOCABULARY_VERSION,
        "seed": seed,
        "samples_per_split_limit": samples_per_split,
        "selected_sample_count": len(selected),
        "selection": selection_report,
        "metadata_integrity": {key: value for key, value in metadata.items() if key != "protocol"},
        "membership_counts": membership_counts,
        "loader": loader,
        "elapsed_seconds": elapsed,
        "loader_throughput_items_per_second": loader["loader_items_per_second"],
        "peak_rss_mib": peak_rss_mib,
        "max_memory_mib": max_memory_mib,
        "bounded_npz_sample_count": len(selected),
        "dataset_metadata_hashes_preserved": True,
        "dataset_writes_performed": 0,
    }
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples-per-split", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-memory-mib", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(
        dataset_dir=args.dataset_dir,
        report_path=args.report,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
        batch_size=args.batch_size,
        max_memory_mib=args.max_memory_mib,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
