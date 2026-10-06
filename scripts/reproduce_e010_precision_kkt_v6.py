#!/usr/bin/env python3
"""Repeat the fixed-state audit only; never run an optimizer or change inputs."""

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts import audit_e010_phase4d_precision_kkt_v6 as run
from scripts.run_e010_phase4d_recurrent_capacity_v3 import load_cache, write


def repeat(i):
    expected = json.loads((run.OUT / "examples" / f"example_{i:02d}.json").read_text())
    captured = []

    def compare(path, value):
        assert value == expected, f"fixed-state audit reproduction mismatch: {i}"
        captured.append(value)

    run.write = compare
    run.inspect(i)
    assert len(captured) == 1
    return dict(
        index=i,
        exact_match=True,
        record_sha256=hashlib.sha256(json.dumps(expected, sort_keys=True, allow_nan=False).encode()).hexdigest(),
    )


if __name__ == "__main__":
    run.CFG = run.contract()
    contract = json.loads((run.OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    run.CACHE, manifest = load_cache()
    ids = run.CFG["stalled_indices"] + run.CFG["control_indices"]
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(repeat, i) for i in ids]):
            row = future.result()
            results.append(row)
            print("reproduced", row["index"], flush=True)
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        run.OUT / "fixed_state_reproduction.json",
        dict(
            examples=22,
            all_exact_matches=True,
            records=sorted(results, key=lambda r: r["index"]),
            protected_hashes_checked=True,
            optimizer_steps=0,
            cuda_used=False,
        ),
    )
