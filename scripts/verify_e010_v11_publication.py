"""Byte-exact aggregate reproduction and convergence-label checks; no optimization."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
from scripts import report_e010_direct_correction_v11 as report
from scripts.run_e010_direct_correction_v11 import OUT, frozen
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def main():
    frozen()
    records = [json.loads((OUT / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    assert len(list((OUT / "attempts").glob("*.json"))) == 60
    for row in records:
        assert row["solver_invocations"] == 1
        cert = row["optimizer"]["physical_certificate"]
        assert row["optimizer"]["converged"] == cert["converged"] == all(cert["gates"].values())
        assert row["optimizer"]["iterations"] <= 1000
        assert json.loads((OUT / "reproduction" / f"example_{row['index']:02d}.json").read_text())[
            "physical_certificate_exact"
        ]
    names = ["result.json", "RESULTS.md"]
    with tempfile.TemporaryDirectory(prefix="e010_v11_publication_") as folder:
        target = Path(folder)
        for name in ["execution_contract.json", "reproduction_complete.json", "attempts", "B", "untracked_states"]:
            (target / name).symlink_to((OUT / name).resolve())
        report.OUT = target
        with contextlib.redirect_stdout(io.StringIO()):
            report.main()
        for name in names:
            assert (target / name).read_bytes() == (OUT / name).read_bytes(), name
    frozen()
    write(
        OUT / "publication_verification.json",
        dict(
            examples=60,
            fixed_state_metrics_and_certificates_exact=True,
            convergence_labels_independently_verified=True,
            aggregate_outputs_byte_exact=True,
            protected_hashes_verified=True,
            panel_solver_invocations=60,
            outcome_based_reruns=0,
            verification_optimizer_invocations=0,
            verifier_sha256=file_hash(Path(__file__)),
            publication_sha256={name: file_hash(OUT / name) for name in names},
            cuda_used=False,
            neural_training_launched=False,
        ),
    )
    print("All60 labels verified; aggregate result and report reproduced byte-for-byte.")


if __name__ == "__main__":
    main()
