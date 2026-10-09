"""Verify the executed control-only handoff and exact first resumed batch."""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import numpy as np
import pyarrow.parquet as pq
import torch
from protein_distance_diffusion.training.e014_runtime import atomic_json, sha


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def main():
    torch.set_num_threads(1)
    run = ROOT / "e014"
    handoff = json.loads((run / "TERMINAL_GUARD_HANDOFF.json").read_text())
    original = handoff["original_checkpoint"]
    if sha(original["path"]) != original["sha256"]:
        raise ValueError("original passing-probe checkpoint integrity failed")
    before = torch.load(original["path"], map_location="cpu", weights_only=False)
    after = torch.load(handoff["checkpoint"], map_location="cpu", weights_only=False)
    fields = set(before) - {"contract", "git_sha"}
    if not all(equal(before[key], after[key]) for key in fields):
        raise ValueError("scientific state changed during terminal guard handoff")
    old_contract, new_contract = before["contract"].copy(), after["contract"].copy()
    old_sources, new_sources = old_contract.pop("source_hashes"), new_contract.pop("source_hashes")
    if old_contract != new_contract or [k for k in old_sources if old_sources[k] != new_sources[k]] != [
        "scripts/run_e014_scaleup.py"
    ]:
        raise ValueError("unexpected contract handoff")
    ids = (
        pq.read_table(ROOT / "population/train.parquet", columns=["sample_id"], use_threads=False).column(0).to_pylist()
    )
    expected_ids = [ids[i] for i in before["next_batch"]]
    resumed = None
    with open(run / "TRAINING.jsonl") as stream:
        for line in stream:
            record = json.loads(line)
            if record["update"] == before["updates"] + 1:
                resumed = record
                break
    if resumed is None or resumed["sample_ids"] != expected_ids or resumed["position"] != before["position"] + 1:
        raise ValueError("exact first resumed batch verification failed")
    atomic_json(
        run / "LIVE_RESUME_VERIFICATION.json",
        dict(
            passed=True,
            committed_probe_updates=before["updates"],
            committed_probe_protein_exposures=int(before["exposures"].sum()),
            first_resumed_update=resumed["update"],
            first_resumed_sample_ids=expected_ids,
            exact_scientific_payload_fields_verified=sorted(fields),
            model_optimizer_rng_schedule_exposures_unchanged=True,
            committed_updates_replayed=0,
            original_checkpoint_sha256=original["sha256"],
            replacement_checkpoint_sha256=sha(handoff["checkpoint"]),
        ),
    )
    print("Live resume and full scientific-state handoff verified", flush=True)


if __name__ == "__main__":
    main()
