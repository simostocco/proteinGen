"""Read-only raw integrity audit and projected sequence cache for E011 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from protein_sequence_generation.context import COLUMNS, bucket, sequence_rows, train_unigrams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/e011_sequence_context_only/pilot_v1"
REPORT = ROOT / "reports/experiments/E011_sequence_context_only/pilot_v1"
PREP = REPORT.parent
DATA = Path("/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run():
    for name, expected in json.loads((PREP / "contract_manifest.json").read_text())["sha256"].items():
        assert sha(ROOT / name) == expected, name
    audit = json.loads((PREP / "data_audit.json").read_text())
    for name, expected in audit["input_hashes"].items():
        assert sha(name) == expected, name
    historical = Path(
        "/mnt/d/Simone/proteinGen/reports/experiments/E006_rich_geometry_codesign/phase3_stage_a_context_diagnostic_v1/report.json"
    )
    assert sha(historical) == "8798c5ce0c4b35e60f45f056225e497404a836a42b4baabcc7cb34a1b5d8cd8a"
    protocol = json.loads((DATA / "protocol.json").read_text())
    inventory = dict(
        (line.split("  ", 1)[1], line.split("  ", 1)[0])
        for line in (DATA / "shard_hashes.sha256").read_text().splitlines()
    )
    verified = []
    for i, record in enumerate(protocol["shards"]):
        path = DATA / record["path"]
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        observed = sha(path)
        assert observed == record["sha256"] == inventory[record["path"]], str(path)
        assert before == (path.stat().st_size, path.stat().st_mtime_ns), str(path)
        assert pq.ParquetFile(path).metadata.num_rows == record["row_count"], str(path)
        verified.append({"path": record["path"], "sha256": observed, "bytes": before[0]})
        if (i + 1) % 32 == 0:
            print(f"Raw shards verified: {i + 1}/{len(protocol['shards'])}", flush=True)
    assert set(inventory) == {r["path"] for r in verified}
    print("Raw shard verification passed", flush=True)
    ids, hashes, cached, counts, strata = {}, {}, {}, {}, {}
    for split in ["train", "validation"]:
        rows = list(sequence_rows(DATA, split))
        ids[split] = {r["sample_id"] for r in rows}
        hashes[split] = {hashlib.sha256(r["sequence"].encode()).hexdigest() for r in rows}
        counts[split] = len(rows)
        assert counts[split] == len(ids[split]) == audit["counts"][split]
        lengths = [0] * 5
        for r in rows:
            lengths[bucket(len(r["token_ids"]))] += 1
        strata[split] = lengths
        assert lengths == audit["strata_counts"][split]
        path = OUT / f"{split}_sequences.parquet"
        assert not path.exists(), "immutable cache already exists"
        pq.write_table(pa.Table.from_pylist(rows).select(list(COLUMNS)), path)
        cached[split] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "columns": list(COLUMNS)}
        if split == "train":
            assert train_unigrams(rows) == json.loads((PREP / "train_baselines.json").read_text())
        del rows
    assert not ids["train"] & ids["validation"] and not hashes["train"] & hashes["validation"]
    structural = Path("/mnt/d/Simone/proteinGen/data/full/splits_recovered_all_structures")
    membership = {}
    for split in ["train", "validation"]:
        projected = pq.read_table(
            structural / f"{split}.parquet", columns=["sample_id", "cluster_id", "pdb_id", "split_group_id"]
        ).to_pylist()
        membership[split] = [r for r in projected if r["sample_id"] in ids[split]]
        assert len(membership[split]) == counts[split]
    for key in ["cluster_id", "pdb_id", "split_group_id"]:
        assert not ({r[key] for r in membership["train"]} & {r[key] for r in membership["validation"]})
    for name, expected in audit["input_hashes"].items():
        assert sha(name) == expected, name
    panel = json.loads((PREP / "diagnostic_panel.json").read_text())
    assert len(panel["sample_ids"]) == 2048 and set(panel["sample_ids"]) <= ids["validation"]
    result = {
        "status": "passed",
        "raw_shards": verified,
        "raw_shard_count": len(verified),
        "counts": counts,
        "strata": strata,
        "cross_split_overlaps": 0,
        "train_only_baselines_exact_match": True,
        "geometry_columns_materialized": 0,
        "sequence_caches": cached,
        "seal_sha256": sha(PREP / "contract_manifest.json"),
    }
    (REPORT / "dataset_integrity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Complete data integrity, split, train baseline and sequence-only cache verification passed", flush=True)


if __name__ == "__main__":
    run()
