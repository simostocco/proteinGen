"""Rebuild recovery aggregate records in a temporary directory, without optimization."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import report_e010_strict_scientific_v10 as report
from scripts import summarize_e010_strict_scientific_v10 as summary
from scripts.recover_e010_strict_scientific_v10 import INDICES, OUT, original, setup


def main():
    setup()
    contract = json.loads((OUT / "contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    reproduced = json.loads((OUT / "reproduction_complete.json").read_text())
    assert reproduced["examples"] == 60 and all(r["reproduced"] for r in reproduced["results"])
    for i in range(60):
        record = OUT / "B" / f"example_{i:02d}.json"
        if i not in INDICES:
            assert record.read_bytes() == (original.OUT / "B" / record.name).read_bytes()
        else:
            data = json.loads(record.read_text())
            assert data["solver_invocations"] == 1
            assert data["state_provenance"] == "technical recovery replay state"
    names = ["result.json", "RESULTS.md", "handoff_statistics.json"]
    with tempfile.TemporaryDirectory(prefix="e010_v10_publication_") as folder:
        destination = Path(folder)
        for name in ["B", "reproduction", "reproduction_complete.json"]:
            (destination / name).symlink_to((OUT / name).resolve())
        report.OUT = destination
        summary.OUT = destination
        with contextlib.redirect_stdout(io.StringIO()):
            report.main()
            summary.main()
        for name in names:
            assert (destination / name).read_bytes() == (OUT / name).read_bytes(), name
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    record = dict(
        aggregate_reproduction_byte_exact=True,
        reproduced_examples=60,
        original_saved_states_unchanged=43,
        technical_recovery_states=17,
        historical_solver_reruns=0,
        verification_optimizer_invocations=0,
        protected_hashes_verified=True,
        outputs_sha256={name: file_hash(OUT / name) for name in names},
        verifier_sha256=file_hash(Path(__file__)),
        cuda_used=False,
        neural_training_launched=False,
    )
    original.write(OUT / "publication_verification.json", record)
    print("All three publication outputs reproduced byte-for-byte; protected hashes verified.")


if __name__ == "__main__":
    main()
