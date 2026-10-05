"Deterministic, archive-free E010 multi-corruption experiment primitives."

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e010_phase4a_multicorruption_v2.yaml"
V2 = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_supervised_generalization_v2"
OUT = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4a_multicorruption_v2"
STAGING = OUT / "phase4a_multicorruption_v2.staging"
FINAL = OUT / "phase4a_multicorruption_v2.final"
REVIEW = OUT / "phase4a_multicorruption_v2_scientific_review_v1"
V1_DISCREPANCY = (
    ROOT
    / "reports/experiments/E010_global_equivariant_expressivity"
    / "phase4a_multicorruption_v1/source_cohort_discrepancy.json"
)
SCHEMA = "e010_phase4a_multicorruption_v2"
GLOBAL_SEED = 41043
STRATA = ("20-64", "65-128", "129-256", "257-384", "385-500")
TRAIN_COUNTS = {"20-64": 3697, "65-128": 3697, "129-256": 3697, "257-384": 3730, "385-500": 1563}
EXAMPLES_PER_STRATUM = 19656
TOTAL_IDENTITIES = 16384
TOTAL_EXAMPLES = 98280
UPDATES = 1092
MICROBATCH = 18
EFFECTIVE_BATCH = 90
BOUNDARIES = ((364, 32760, 6552), (728, 65520, 13104), (1092, 98280, 19656))


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def hash_rank(namespace: str, role: str, sample_id: str) -> str:
    return sha_bytes(f"{namespace}|{role}|{sample_id}".encode())


def choose_extra_ids(rows: list[dict[str, Any]], count: int, *, global_seed: int, stratum: str) -> set[str]:
    ranked = sorted(
        rows,
        key=lambda r: (sha_bytes(f"{SCHEMA}|{global_seed}|extra|{stratum}|{r['sample_id']}".encode()), r["sample_id"]),
    )
    if not 0 <= count <= len(ranked):
        raise ValueError(f"invalid extra-corruption count for {stratum}: {count}/{len(ranked)}")
    return {str(r["sample_id"]) for r in ranked[:count]}


def derive_seed(global_seed: int, sample_id: str, corruption_index: int, role: str = "training") -> int:
    raw = f"{SCHEMA}|{global_seed}|{role}|{sample_id}|{corruption_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def source_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = (ROOT / row["source_path"]).resolve()
    if ROOT.resolve() not in source.parents or not source.is_file():
        raise FileNotFoundError(f"authorized source file missing or outside repository: {row['sample_id']}")
    with np.load(source, allow_pickle=False) as z:
        coords = np.asarray(z["ca_coordinates"], dtype=np.float32)
        mask = np.asarray(z["residue_mask"], dtype=np.bool_)
        sid = str(z["sample_id"].item())
    n = int(row["length"])
    if (
        sid != str(row["sample_id"])
        or coords.shape != (n, 3)
        or mask.shape != (n,)
        or not mask.all()
        or not np.isfinite(coords).all()
    ):
        raise ValueError(f"source identity, shape, mask, or finiteness mismatch: {row['sample_id']}")
    return np.ascontiguousarray(coords - coords.mean(axis=0, keepdims=True), dtype=np.float32), np.ascontiguousarray(
        mask
    )


def tensor_sha(array: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes())


def build_seed_manifest(
    train_rows: list[dict[str, Any]], *, global_seed: int = GLOBAL_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(train_rows) != TOTAL_IDENTITIES or len({r["sample_id"] for r in train_rows}) != TOTAL_IDENTITIES:
        raise ValueError("training cohort must contain exactly 16,384 unique identities")
    by_stratum = {s: [r for r in train_rows if r["stratum"] == s] for s in STRATA}
    if set(by_stratum) != set(STRATA) or any(r["stratum"] not in STRATA for r in train_rows):
        raise ValueError("training cohort contains an unknown or missing length stratum")
    manifest_rows: list[dict[str, Any]] = []
    seed_pairs: set[tuple[str, int]] = set()
    expected = {
        "20-64": (3697, 1171, 5, 6),
        "65-128": (3697, 1171, 5, 6),
        "129-256": (3697, 1171, 5, 6),
        "257-384": (3730, 1006, 5, 6),
        "385-500": (1563, 900, 12, 13),
    }
    for stratum in STRATA:
        rows = by_stratum[stratum]
        n, n_extra, ordinary, extra = expected[stratum]
        if len(rows) != n:
            raise ValueError(f"cohort stratum count mismatch for {stratum}: {len(rows)} != {n}")
        if any(
            not (int(r["length"]) >= int(stratum.split("-")[0]) and int(r["length"]) <= int(stratum.split("-")[1]))
            for r in rows
        ):
            raise ValueError(f"identity length falls outside declared stratum {stratum}")
        extras = choose_extra_ids(rows, n_extra, global_seed=global_seed, stratum=stratum)
        for row in sorted(rows, key=lambda r: r["sample_id"]):
            multiplicity = extra if row["sample_id"] in extras else ordinary
            seeds = [derive_seed(global_seed, row["sample_id"], i) for i in range(multiplicity)]
            if len(set(seeds)) != multiplicity:
                raise ValueError(f"corruption-seed collision within identity {row['sample_id']}")
            for seed in seeds:
                pair = (str(row["sample_id"]), seed)
                if pair in seed_pairs:
                    raise ValueError(f"duplicate (sample_id, corruption_seed) pair: {pair}")
                seed_pairs.add(pair)
            manifest_rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "stratum": stratum,
                    "length": int(row["length"]),
                    "source_path": str(row["source_path"]),
                    "source_sha256": str(row["source_sha256"]),
                    "selection_rank_sha256": str(row["selection_rank_sha256"]),
                    "multiplicity": multiplicity,
                    "corruption_seeds": seeds,
                }
            )
    manifest_rows.sort(key=lambda r: (STRATA.index(r["stratum"]), r["sample_id"]))
    counts = {s: sum(r["multiplicity"] for r in manifest_rows if r["stratum"] == s) for s in STRATA}
    if counts != {s: EXAMPLES_PER_STRATUM for s in STRATA} or len(seed_pairs) != TOTAL_EXAMPLES:
        raise ValueError(f"seed manifest example totals incorrect: {counts}, unique pairs={len(seed_pairs)}")
    summary = {
        "schema": "e010_multicorruption_seed_manifest_v1",
        "immutable": True,
        "seed_derivation": "first_63_bits_of_sha256(schema|global_seed|role|sample_id|corruption_index)",
        "extra_selection": "sha256(schema|global_seed|extra|stratum|sample_id), ascending; sample_id tie break",
        "global_seed": global_seed,
        "identity_allocation_description": (
            "capped identity allocation with exactly length-balanced training-example exposure"
        ),
        "training_identity_count": len(manifest_rows),
        "unique_identity_seed_pair_count": len(seed_pairs),
        "example_count_by_stratum": counts,
        "total_training_examples": len(seed_pairs),
        "tensor_archives_written": False,
        "on_demand_regeneration": True,
    }
    return manifest_rows, summary


def build_schedule(seed_manifest: list[dict[str, Any]], *, schedule_seed: int = GLOBAL_SEED) -> list[dict[str, Any]]:
    ranked = {}
    for stratum in STRATA:
        examples = []
        for row in seed_manifest:
            if row["stratum"] != stratum:
                continue
            for idx, seed in enumerate(row["corruption_seeds"]):
                key = sha_bytes(f"{schedule_seed}|{stratum}|{row['sample_id']}|{idx}|{seed}".encode())
                examples.append((key, row["sample_id"], idx, seed))
        if len(examples) != EXAMPLES_PER_STRATUM:
            raise ValueError(f"wrong schedule source example count for {stratum}: {len(examples)}")
        examples.sort(key=lambda x: (x[0], x[1], x[2]))
        ranked[stratum] = examples
    schedule = []
    for update in range(UPDATES):
        batches = {}
        for stratum in STRATA:
            batch = ranked[stratum][update * MICROBATCH : (update + 1) * MICROBATCH]
            batches[stratum] = [
                {"sample_id": sid, "corruption_index": idx, "corruption_seed": seed} for _, sid, idx, seed in batch
            ]
        schedule.append(
            {
                "global_update": update + 1,
                "stratum_microbatches": batches,
                "effective_batch_sample_count": EFFECTIVE_BATCH,
            }
        )
    return schedule


def schedule_sha256(schedule: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in schedule:
        h.update(canonical_json(row) + b"\n")
    return h.hexdigest()


def regenerate_corruption(target: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    "Exact CPU E009 corruption; a local generator leaves global RNG untouched."
    import torch

    t = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(t.shape, generator=gen) * float(sigma)
    noise -= noise.mean(0, keepdim=True)
    walk = torch.cumsum(noise, 0)
    walk -= walk.mean(0, keepdim=True)
    walk *= float(sigma) * 0.12 / walk.square().mean().sqrt().clamp_min(1e-6)
    return np.ascontiguousarray((t + noise + walk).numpy().astype(np.float32, copy=False))


def validate_seed_manifest(
    rows: list[dict[str, Any]], summary: dict[str, Any], *, global_seed: int = GLOBAL_SEED
) -> None:
    regenerated, expected_summary = build_seed_manifest(
        [
            {
                "sample_id": r["sample_id"],
                "stratum": r["stratum"],
                "length": r["length"],
                "source_path": r["source_path"],
                "source_sha256": r["source_sha256"],
                "selection_rank_sha256": r["selection_rank_sha256"],
            }
            for r in rows
        ],
        global_seed=global_seed,
    )
    if regenerated != rows or expected_summary != summary:
        raise ValueError("seed manifest does not reproduce exactly from its identity rows")
