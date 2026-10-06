"""Fresh aggregate reproduction and protected-hash / once-only publication checks."""

import json
import tempfile
from pathlib import Path

from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
from scripts import report_e010_matched_direct_space_v13 as report
from scripts import run_e010_matched_direct_space_v13 as runner
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def main():
    runner.frozen()
    out = runner.OUT
    rows = [json.loads((out / "B" / f"example_{i:02d}.json").read_text()) for i in range(60)]
    assert len(list((out / "attempts").glob("*.json"))) == 60
    for r in rows:
        cert = r["optimizer"]["physical_certificate"]
        assert r["solver_invocations"] == 1 and r["optimizer"]["iterations"] <= 2000
        assert r["optimizer"]["converged"] == cert["converged"] == all(cert["gates"].values())
        assert json.loads((out / "reproduction" / f"example_{r['index']:02d}.json").read_text())[
            "physical_certificate_exact"
        ]
        runner.prefix_match(r["index"])
    names = ["result.json", "RESULTS.md"]
    with tempfile.TemporaryDirectory(prefix="e010_v13_publication_") as folder:
        target = Path(folder)
        for name in ["execution_contract.json", "reproduction_complete.json", "attempts", "B", "untracked_states"]:
            (target / name).symlink_to((out / name).resolve())
        report.OUT = target
        report.main()
        for name in names:
            assert (target / name).read_bytes() == (out / name).read_bytes(), name
    runner.frozen()
    write(
        out / "publication_verification.json",
        dict(
            examples=60,
            all_metrics_and_certificates_exact=True,
            aggregate_reports_byte_exact=True,
            historical_solver_prefixes_exact=True,
            protected_hashes_verified=True,
            panel_solver_invocations=60,
            outcome_based_reruns=0,
            verification_optimizer_invocations=0,
            verifier_sha256=file_hash(Path(__file__)),
            publication_sha256={name: file_hash(out / name) for name in names},
            cuda_used=False,
            neural_training_launched=False,
        ),
    )
    print("All60 V13 states/certificates, paired prefixes and aggregate publication verified.", flush=True)


if __name__ == "__main__":
    main()
