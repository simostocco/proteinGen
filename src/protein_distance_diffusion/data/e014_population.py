"""Hash-verified full accepted TRAIN and fixed disjoint VALIDATION panels."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, d_path, select_panels, sha

pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def verify(path, expected):
    observed = sha(path)
    if observed != expected:
        raise ValueError(f"provenance mismatch: {path}")
    return observed


def prepare_manifests(cfg):
    out = d_path(cfg["output_dir"]) / "population"
    if out.exists():
        report = json.loads((out / "population.json").read_text())
        for path, digest in report["source_hashes"].items():
            verify(path, digest)
        if (
            report["development_count"] != 5 * cfg["development_per_stratum"]
            or report["prospective_count"] != 5 * cfg["prospective_per_stratum"]
        ):
            raise ValueError("frozen panel configuration mismatch")
        for path, digest in report["manifest_hashes"].items():
            verify(path, digest)
        return report
    source_hashes = {}
    for key in ("assignments", "clean_validation", "historical_panel", "normalization_artifact"):
        digest_key = "normalization_sha256" if key == "normalization_artifact" else f"{key}_sha256"
        source_hashes[cfg[key]] = verify(cfg[key], cfg[digest_key])
    data = Path(cfg["dataset_root"])
    source_hashes[str(data / "protocol.json")] = verify(data / "protocol.json", cfg["dataset_protocol_sha256"])
    source_hashes[str(data / "shard_hashes.sha256")] = verify(
        data / "shard_hashes.sha256", cfg["dataset_inventory_sha256"]
    )
    columns = ["sample_id", "split", "cluster_id", "length", "length_stratum", "dataset_shard_path", "shard_row_index"]
    assignments = pq.read_table(cfg["assignments"], columns=columns, use_threads=False).to_pylist()
    train = sorted([r for r in assignments if r["split"] == "train"], key=lambda r: r["sample_id"])
    clean = pq.read_table(cfg["clean_validation"], columns=columns, use_threads=False).to_pylist()
    if len({r["sample_id"] for r in train}) != len(train) or any(r["split"] != "validation" for r in clean):
        raise ValueError("invalid population identities/splits")
    train_ids = {r["sample_id"] for r in train}
    if train_ids & {r["sample_id"] for r in clean}:
        raise ValueError("train/validation identity leakage")
    if {name.split("_")[0].lower() for name in train_ids} & {r["sample_id"].split("_")[0].lower() for r in clean}:
        raise ValueError("train/validation PDB-entry leakage")
    if {r["cluster_id"] for r in train} & {r["cluster_id"] for r in clean}:
        raise ValueError("train/validation homology leakage")
    norm = json.loads(Path(cfg["normalization_artifact"]).read_text())
    if (
        norm["accepted_sample_count"] != len(train)
        or norm["fitting_split"] != "train"
        or norm["coordinate_scale_angstrom"] != cfg["coordinate_scale_angstrom"]
    ):
        raise ValueError("accepted population disagrees with frozen normalization provenance")
    historical = json.loads(Path(cfg["historical_panel"]).read_text())["selected_identities"]
    historical_ids = {r["sample_id"] for r in historical}
    dev, prospective = select_panels(
        clean, cfg["development_per_stratum"], cfg["prospective_per_stratum"], historical_ids
    )
    lookup = {r["sample_id"]: r for r in assignments}
    history = [
        dict(lookup[r["sample_id"]], historical_split=r["split"], historical_index=z)
        for split in ("train", "validation")
        for z, r in enumerate([r for r in historical if r["split"] == split])
    ]
    out.mkdir(exist_ok=False)
    hashes = {}
    for name, rows in [("train", train), ("development", dev), ("prospective", prospective), ("historical", history)]:
        path = out / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        hashes[str(path)] = sha(path)
    report = dict(
        train_count=len(train),
        train_sample_id_hash=canonical_hash([r["sample_id"] for r in train]),
        sample_id_hash_definition="SHA256 canonical JSON of sorted TRAIN sample-ID strings",
        train_length_strata=dict(Counter(r["length_stratum"] for r in train)),
        accepted_clean_validation_count=len(clean),
        development_count=len(dev),
        prospective_count=len(prospective),
        panel_selection="SHA256 e014-panel-v1|sample_id within stratum; disjoint panels, historical excluded",
        development_sample_id_hash=canonical_hash([r["sample_id"] for r in dev]),
        prospective_sample_id_hash=canonical_hash([r["sample_id"] for r in prospective]),
        source_hashes=source_hashes,
        manifest_hashes=hashes,
        normalization_sha256=cfg["normalization_sha256"],
        split_policy="frozen accepted coordinate assignments; 30%-identity/80%-coverage clean validation",
        sequence_identity_read=False,
    )
    atomic_json(out / "population.json", report)
    return report


def build_coordinate_cache(cfg):
    """One pass over each accessed source shard; validate every accepted row."""
    root = d_path(cfg["output_dir"])
    population = json.loads((root / "population/population.json").read_text())
    for path, digest in population["manifest_hashes"].items():
        verify(path, digest)
    out = root / "coordinate_cache"
    if (out / "READY.json").exists():
        return json.loads((out / "READY.json").read_text())
    if out.exists():
        raise ValueError("incomplete coordinate cache; preserve it and repair explicitly")
    rows = {}
    for name in ("train", "development", "prospective", "historical"):
        for row in pq.read_table(root / f"population/{name}.parquet", use_threads=False).to_pylist():
            rows[row["sample_id"]] = row
    rows = sorted(rows.values(), key=lambda r: r["sample_id"])
    positions = {r["sample_id"]: i for i, r in enumerate(rows)}
    lengths = np.asarray([r["length"] for r in rows], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    out.mkdir()
    np.save(out / "lengths.npy", lengths)
    np.save(out / "offsets.npy", offsets)
    atomic_json(out / "identities.json", [r["sample_id"] for r in rows])
    coordinates = np.memmap(out / "coordinates.f32", dtype="<f4", mode="w+", shape=(int(offsets[-1]), 3))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["dataset_shard_path"]].append(r)
    data = Path(cfg["dataset_root"])
    protocol = json.loads((data / "protocol.json").read_text())
    authoritative = {s["path"]: s["sha256"] for s in protocol["shards"]}
    hashes = {}
    for number, (shard, selected) in enumerate(sorted(grouped.items())):
        path = data / shard
        hashes[str(path)] = verify(path, authoritative[shard])
        columns = ["sample_id", "ca_coordinates", "ca_mask", "chain_continuity_mask", "source_sha256", "npz_sha256"]
        table = pq.read_table(path, columns=columns, use_threads=False)
        for meta in selected:
            row = table.slice(meta["shard_row_index"], 1).to_pylist()[0]
            x = np.asarray(row["ca_coordinates"], dtype=np.float32)
            if row["sample_id"] != meta["sample_id"] or x.shape != (meta["length"], 3):
                raise ValueError("source row contradiction")
            if not all(row["ca_mask"]) or not all(row["chain_continuity_mask"]) or not np.isfinite(x).all():
                raise ValueError("corrupted/non-contiguous accepted source")
            i = positions[meta["sample_id"]]
            coordinates[offsets[i] : offsets[i + 1]] = x
        if (number + 1) % 25 == 0:
            print(f"cache source shards verified {number + 1}/{len(grouped)}", flush=True)
    coordinates.flush()
    del coordinates
    digests = {str(p): sha(p) for p in out.iterdir() if p.is_file()}
    report = dict(
        identity_count=len(rows),
        total_residues=int(offsets[-1]),
        source_shard_hashes=hashes,
        cache_hashes=digests,
        population_hash=population["train_sample_id_hash"],
        sequence_input=False,
    )
    atomic_json(out / "READY.json", report)
    return report


class CoordinateCache:
    def __init__(self, root, verify_payload=True):
        self.root = d_path(root) / "coordinate_cache"
        self.report = json.loads((self.root / "READY.json").read_text())
        if verify_payload:
            for path, digest in self.report["cache_hashes"].items():
                verify(path, digest)
        self.identities = json.loads((self.root / "identities.json").read_text())
        self.positions = {name: z for z, name in enumerate(self.identities)}
        self.offsets = np.load(self.root / "offsets.npy", allow_pickle=False)
        self.values = np.memmap(self.root / "coordinates.f32", dtype="<f4", mode="r", shape=(int(self.offsets[-1]), 3))

    def get(self, sample_id):
        z = self.positions[sample_id]
        return np.array(self.values[self.offsets[z] : self.offsets[z + 1]], copy=True)
