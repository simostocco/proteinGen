"""CPU-only E011 audit/preparation. No definitive training entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Set before importing torch: this script must never contend for the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pyarrow.parquet as pq
import torch

from protein_sequence_generation.context import (
    STRATA,
    SequenceContextTransformer,
    bucket,
    collate,
    conditions,
    deterministic_mask,
    objective,
    paired_forwards,
    protein_ce,
    sequence_rows,
    stable_seed,
    train_unigrams,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/E011_sequence_context_only"
DATA = Path("/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2")
HIST = Path("/mnt/d/Simone/proteinGen")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_contract():
    """A sealed v1 preparation cannot be overwritten; changes require a new version."""
    seal = REPORT / "contract_manifest.json"
    if not seal.exists():
        return False
    for name, digest in json.loads(seal.read_text())["sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"sealed S1 contract changed: {name}")
    print("Sealed S1 v1 contract verified; no files overwritten", flush=True)
    return True


def historical():
    inventory = []
    seen = set()
    for root in [
        HIST,
        Path("/home/simostocco/proteinGen"),
        Path("/home/simostocco/proteinGen.pre-relocation-20260922T064202Z"),
    ]:
        for sub in ["reports/experiments", "outputs"]:
            for track in sorted((root / sub).glob("*")):
                if not track.name.lower().startswith(("e005", "e006")):
                    continue
                for p in sorted(track.rglob("*")):
                    if p.is_file() and p.suffix in {".json", ".jsonl", ".md", ".yaml", ".sha256"}:
                        resolved = str(p.resolve())
                        if resolved in seen:
                            continue
                        seen.add(resolved)
                        inventory.append({"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size})
    p = HIST / "reports/experiments/E006_rich_geometry_codesign/phase3_stage_a_context_diagnostic_v1/report.json"
    protocol = json.loads(p.with_name("protocol.json").read_text())
    assert sha(p) == protocol["report_sha256"] and protocol["status"] == "completed"
    e005 = HIST / "reports/experiments/E005_sequence_geometry_codesign/large_diagnostic_v2/protocol.json"
    d = json.loads(e005.read_text())
    completed = []
    for entry in inventory:
        if entry["path"].endswith("report.json"):
            result = json.loads(Path(entry["path"]).read_text())

            # Recursive search records emitted labels; configs are never evidence.
            def walk(x, entry=entry):
                if isinstance(x, dict):
                    for k, v in x.items():
                        if (
                            k == "classification"
                            and isinstance(v, str)
                            and v
                            in {
                                "contextual_learning_verified",
                                "marginal_frequency_only",
                                "conditioning_path_ineffective",
                                "inconclusive",
                            }
                        ):
                            completed.append({**entry, "classification": v})
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)

            walk(result)
    write(
        "historical_audit.json",
        {
            "artifact_inventory": inventory,
            "emitted_contextual_classifications": completed,
            "e006_final_completed_context_classification": protocol["classification"],
            "e006_report_path": str(p),
            "e006_report_sha256": sha(p),
            "e006_postmortem": json.loads((p.parents[1] / "STAGE_A_SEQUENCE_ONLY_POSTMORTEM_V1.json").read_text()),
            "e005_result_path": str(e005),
            "e005_result_sha256": sha(e005),
            "e005_primary_comparisons": d["primary_comparisons"],
            "e005_scientific_limitations": d["scientific_limitations"],
            "e005_validation_trajectory": d["validation_trajectory"],
            "objective_defect": (
                "v6 sample hinge uses attached shuffled CE; paired_forwards and production caller retain its graph"
            ),
            "e011_correction": "detach shuffled CE; equal-protein CE plus equal-protein hinge",
        },
    )


def audit_data():
    identities, counts, strata, hashes, samples = {}, {}, {}, {}, {}
    panel_candidates = [[] for _ in STRATA]
    smoke = []
    for split in ["train", "validation"]:
        ids, seq_hashes = set(), set()
        lengths = [0] * 5
        for row in sequence_rows(DATA, split):
            sid = row["sample_id"]
            if sid in ids:
                raise ValueError("duplicate sample ID")
            ids.add(sid)
            h = hashlib.sha256(row["sequence"].encode()).hexdigest()
            seq_hashes.add(h)
            b = bucket(len(row["token_ids"]))
            lengths[b] += 1
            if split == "validation":
                panel_candidates[b].append((stable_seed(6111, sid), row))
            elif len(smoke) < 2 and len(row["token_ids"]) <= 64:
                smoke.append(row)
        samples[split] = ids
        hashes[split] = seq_hashes
        counts[split] = len(ids)
        strata[split] = lengths
    overlap = {
        "sample_id": len(samples["train"] & samples["validation"]),
        "canonical_sequence_hash": len(hashes["train"] & hashes["validation"]),
    }
    homology = {}
    membership = {}
    for split in ["train", "validation"]:
        path = HIST / f"data/full/splits_recovered_all_structures/{split}.parquet"
        columns = ["sample_id", "cluster_id", "pdb_id", "split_group_id"]
        rows = pq.read_table(path, columns=columns).to_pylist()
        subset = [r for r in rows if r["sample_id"] in samples[split]]
        membership[split] = subset
        identities[str(path)] = sha(path)
        homology[f"{split}_missing_membership"] = len(samples[split] - {r["sample_id"] for r in subset})
    for key in ["cluster_id", "pdb_id", "split_group_id"]:
        homology[key + "_overlap"] = len(
            {r[key] for r in membership["train"]} & {r[key] for r in membership["validation"]}
        )
    if any(overlap.values()) or any(homology.values()):
        raise ValueError("S1 split isolation failed")
    protocol = json.loads((DATA / "protocol.json").read_text())
    if counts != protocol["eligible_split_counts"] or protocol["status"] != "completed":
        raise ValueError("S1 dataset protocol/count contradiction")
    # Fixed 2048 panel, minimum 64 per stratum; round-robin fills remaining.
    candidates = [sorted(c, key=lambda x: x[0]) for c in panel_candidates]
    panel = [r for c in candidates for _, r in c[:64]]
    positions = [min(64, len(c)) for c in candidates]
    while len(panel) < 2048:
        progress = False
        for b, c in enumerate(candidates):
            if positions[b] < len(c) and len(panel) < 2048:
                panel.append(c[positions[b]][1])
                positions[b] += 1
                progress = True
        if not progress:
            raise ValueError("insufficient panel")
    # Donor assignment is deterministic, different ID and canonical sequence.
    donor_ids = {}
    for row in panel:
        choices = [
            r
            for r in panel
            if bucket(len(r["token_ids"])) == bucket(len(row["token_ids"]))
            and r["sample_id"] != row["sample_id"]
            and r["sequence"] != row["sequence"]
        ]
        if not choices:
            raise ValueError("no comparable donor")
        donor_ids[row["sample_id"]] = min(choices, key=lambda r: stable_seed(6111, row["sample_id"], r["sample_id"]))[
            "sample_id"
        ]
    for name in ["protocol.json", "schema.json", "vocabulary.json", "normalization.json", "shard_hashes.sha256"]:
        identities[str(DATA / name)] = sha(DATA / name)
    write(
        "diagnostic_panel.json",
        {
            "sample_ids": [r["sample_id"] for r in panel],
            "donor_ids": donor_ids,
            "strata_counts": positions,
            "seed": 6111,
        },
    )
    write("train_baselines.json", train_unigrams(sequence_rows(DATA, "train")))
    write(
        "data_audit.json",
        {
            "directory": str(DATA),
            "input_hashes": identities,
            "counts": counts,
            "strata_counts": strata,
            "unique_sequence_counts": {s: len(h) for s, h in hashes.items()},
            "cross_split_overlap": overlap,
            "homology": homology,
            "cluster_metadata": json.loads(
                (HIST / "data/full/processed/mmseqs_clusters_cluster.metadata.json").read_text()
            ),
            "geometry_column_reads": 0,
            "limitation": (
                "E006 geometry-eligible cohort is reused; selection bias persists "
                "though no structural features are loaded."
            ),
        },
    )
    return smoke


def real_smoke(rows):
    torch.manual_seed(6011)
    model = SequenceContextTransformer().train()
    targets, valid = collate(rows)
    ids = [r["sample_id"] for r in rows]
    mask = deterministic_mask(targets, valid, ids, 0.3)
    normal = targets.masked_fill(mask, 1)
    shuffled = normal.clone()
    for i in range(len(rows)):
        positions = torch.where(valid[i] & ~mask[i])[0]
        shuffled[i, positions] = normal[i, positions.flip(0)]
    n, s = paired_forwards(model, normal, shuffled, valid)
    assert torch.isfinite(n).all() and torch.isfinite(s).all()
    loss = objective(n, s, targets, mask, valid)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    write(
        "real_cpu_smoke.json",
        {
            "device": "cpu",
            "samples": ids,
            "lengths": valid.sum(1).tolist(),
            "loss": float(loss.detach()),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "forward_backward_finite": True,
            "cuda": "deferred_due_to_active_phase4c_gpu_workload",
        },
    )


def synthetic_smoke():
    """Random-phase alternating motif: same marginal at every position.

    Separate train/evaluation seeds; ordered visible residues identify phase.
    Small capacity is an infrastructure proof, never an S1 scientific result.
    """
    torch.manual_seed(7011)
    model = SequenceContextTransformer(d_model=32, layers=2, heads=4, ffn=64, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    def batch(seed, count):
        g = torch.Generator().manual_seed(seed)
        phase = torch.randint(0, 2, (count, 1), generator=g)
        t = ((torch.arange(24)[None] + phase) % 2) + 2
        return t, torch.ones_like(t, dtype=torch.bool)

    for step in range(150):
        targets, valid = batch(8000 + step, 16)
        ids = [f"train-{step}-{i}" for i in range(16)]
        fraction = [0.15, 0.3, 0.5][step % 3]
        mask = deterministic_mask(targets, valid, ids, fraction)
        normal = targets.masked_fill(mask, 1)
        shuffled = normal.clone()
        for i in range(16):
            visible = torch.where(~mask[i])[0]
            g = torch.Generator().manual_seed(stable_seed(step, i, "synthetic-shuffle"))
            shuffled[i, visible] = normal[i, visible[torch.randperm(len(visible), generator=g)]]
        n, s = paired_forwards(model, normal, shuffled, valid)
        loss = objective(n, s, targets, mask, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    results = {}
    targets, valid = batch(17011, 64)
    ids = [f"eval-{i}" for i in range(64)]
    donors = [{"sample_id": f"donor-{i}", "token_ids": (5 - targets[i]).tolist()} for i in range(64)]
    with torch.no_grad():
        for fraction in [0.15, 0.3, 0.5]:
            mask = deterministic_mask(targets, valid, ids, fraction)
            c = conditions(targets, valid, mask, ids, donors)
            scores = {name: float(protein_ce(model(t, valid), targets, mask, valid).mean()) for name, t in c.items()}
            scores["train_unigram"] = float(np.log(2))
            scores["passes"] = all(
                scores["normal"] < scores[k] - 0.05
                for k in ["visible_shuffle", "null_context", "permuted_context", "train_unigram"]
            )
            results[str(fraction)] = scores
    write(
        "synthetic_cpu_proof.json",
        {
            "optimizer_updates": 150,
            "seed": 7011,
            "device": "cpu",
            "architecture": "32/2/4/64; reduced synthetic infrastructure fixture",
            "results": results,
            "passed": all(v["passes"] for v in results.values()),
            "scientific_classification": "not_a_real_data_S1_result",
        },
    )
    assert all(v["passes"] for v in results.values()), results


if __name__ == "__main__":
    REPORT.mkdir(parents=True, exist_ok=True)
    if verify_contract():
        raise SystemExit(0)
    torch.set_num_threads(2)
    historical()
    print("Historical audit complete", flush=True)
    rows = audit_data()
    print("Data/split audit complete", flush=True)
    real_smoke(rows)
    print("Real CPU forward/backward complete", flush=True)
    synthetic_smoke()
    print("Synthetic contextual proof complete", flush=True)
