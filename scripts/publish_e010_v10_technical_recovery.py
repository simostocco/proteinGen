"""Publish unchanged V10 adjudication over 43 original plus 17 technical states."""

import json

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import report_e010_strict_scientific_v10 as report
from scripts import summarize_e010_strict_scientific_v10 as summary
from scripts.recover_e010_strict_scientific_v10 import INDICES, OUT, original, setup


def main():
    setup()
    contract = json.loads((OUT / "contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    run = json.loads((OUT / "run_complete.json").read_text())
    assert run["examples"] == 17 and all(r["saved"] for r in run["results"])
    reproduction = json.loads((OUT / "reproduction_complete.json").read_text())
    assert reproduction["examples"] == 60 and all(r["reproduced"] for r in reproduction["results"])
    assert len(list((OUT / "attempts").glob("*.json"))) == 17
    states = []
    for i in range(60):
        path = OUT / "B" / f"example_{i:02d}.json"
        record = json.loads(path.read_text())
        if i in INDICES:
            assert record["state_provenance"] == "technical recovery replay state"
            assert record["solver_invocations"] == 1
        else:
            assert path.read_bytes() == (original.OUT / "B" / path.name).read_bytes()
        states.append(
            dict(
                index=i,
                record=record["record"],
                provenance="technical recovery replay state" if i in INDICES else "historical saved original state",
                record_sha256=file_hash(path),
                final_coordinate_sha256=record["state_coordinate_sha256"][-1],
                solver_invocations_this_recovery=1 if i in INDICES else 0,
            )
        )
    original.write(
        OUT / "provenance.json",
        dict(
            states=states,
            historical_saved_states=43,
            technical_recovery_replay_states=17,
            historical_solver_reruns=0,
            replay_solver_invocations=17,
            recovered_states_are_serialized_originals=False,
        ),
    )
    report.OUT = OUT
    report.main()
    summary.OUT = OUT
    summary.main()
    result = json.loads((OUT / "result.json").read_text())
    text = (
        "# V10 authorized technical recovery — complete-panel adjudication\n\n"
        "Provenance: **43 immutable historical saved original states + 17 technical recovery replay states**. "
        "The 17 are explicitly not claimed to be serialized originals. "
        "No historical saved example was optimized again. "
        "Each missing example was invoked exactly once from zero.\n\n"
        "Only the previously validated one-line empty-active-set telemetry guard changed. "
        "The objective, K8/.04 Å, data, constraints, precision, solver, iterations, physical contract and shadows "
        "are unchanged. Saved controls preserved coordinate hashes, objective, metrics and certificates exactly; "
        "iteration/status metadata are unchanged and no control optimizer was run.\n\n"
        f"Classification: **{result['classification']}**. Physical convergence: "
        f"**{result['successful_physical_convergence']}/60**.\n\n"
        "All 60 final trajectories, metrics and physical certificates were independently reconstructed from "
        "hashed saved variables. The original failed V10 record remains immutable. "
        "See RESULTS.md, result.json, handoff_statistics.json and provenance.json for full-panel results.\n\n"
        "CPU float64. CUDA: NO. Neural training: NO. No follow-on intervention launched.\n"
    )
    (OUT / "RECOVERY.md").write_text(text)
    assert_file_pins(contract["protected_sha256"])
    print("Published complete60 adjudication with explicit43/17 provenance", flush=True)


if __name__ == "__main__":
    main()
