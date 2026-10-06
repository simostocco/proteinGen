"""Validate already-finalized states; no optimization and no scientific selection."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins
from scripts import run_e010_strict_scientific_v10 as runner

if __name__ == "__main__":
    runner.setup()
    runner.CONTRACT = json.loads((runner.OUT / "execution_contract.json").read_text())
    assert_file_pins(runner.CONTRACT["protected_sha256"])
    runner.CACHE, _ = runner.load_cache()
    completed = sorted(int(p.stem.split("_")[-1]) for p in (runner.OUT / "B").glob("example_*.json"))
    pending = [i for i in completed if not (runner.OUT / "reproduction" / f"example_{i:02d}.json").exists()]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(runner.example, i, "reproduce") for i in pending]):
            print("independent_fixed_state_validation", *future.result(), flush=True)
    assert_file_pins(runner.CONTRACT["protected_sha256"])
    assert_file_pins(runner.CONTRACT["protected_input_sha256"])
