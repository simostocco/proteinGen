"""Explicit, narrowly verified checkpoint handoff for one control-only fix.

Never weakens the trainer's source-hash mismatch refusal. Preserve the original
checkpoint and create a new verified checkpoint only if the complete runner
diff is the single recognized terminal-phase guard line. No scientific state,
model, optimizer, corruption, schedule or exposure change is permitted.
"""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import hashlib
import fcntl
import json
import subprocess
import time
from pathlib import Path
from protein_distance_diffusion.training.e014_runtime import atomic_checkpoint, atomic_json, load_checkpoint, sha

FROZEN_PROBE_SOURCE = "e7fb20e6a65f504e16206bd80c245d57fd27457a"
RUNNER = "scripts/run_e014_scaleup.py"
OLD_LINE = (
    '                        screen_finished = status["phase"] in '
    '("completed", "complete", "screen_complete", "finished")\n'
)
ADDED_LINE = '                        screen_finished = screen_finished or status["phase"] == "frozen_corpus_ready"\n'


def verify_guard_only_change(old, new):
    if old.count(OLD_LINE.encode()) != 1 or new != old.replace(OLD_LINE.encode(), (OLD_LINE + ADDED_LINE).encode()):
        raise ValueError("only the audited terminal-phase guard handoff is allowed")


def main():
    run = ROOT / "e014"
    owner = open(run / "OWNER.lock", "a+")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    heartbeat = json.loads((run / "HEARTBEAT.json").read_text())
    if heartbeat["phase"] != "probe_completed_resumable":
        raise ValueError("guard handoff requires a committed passing probe boundary")
    old = subprocess.check_output(["git", "show", f"{FROZEN_PROBE_SOURCE}:{RUNNER}"])
    new = Path(RUNNER).read_bytes()
    verify_guard_only_change(old, new)
    contract = json.loads((run / "RUN_CONTRACT.json").read_text())["contract"]
    if contract["source_hashes"][RUNNER] != hashlib.sha256(old).hexdigest():
        raise ValueError("unexpected probe source provenance")
    for path, digest in contract["source_hashes"].items():
        if path != RUNNER and sha(path) != digest:
            raise ValueError("additional runtime source change refused")
    checkpoints = run / "checkpoints"
    previous = json.loads((checkpoints / "LATEST.json").read_text())
    state = load_checkpoint(checkpoints, contract)
    if not state["probe_passed"]:
        raise ValueError("probe checkpoint must have passed")
    state["contract"]["source_hashes"][RUNNER] = sha(RUNNER)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    state["git_sha"] = revision
    stamp = time.time_ns()
    state["source_compatibility_handoff"] = dict(
        reason="recognize native frozen_corpus_ready terminal phase",
        original_checkpoint=previous,
        original_source_commit=FROZEN_PROBE_SOURCE,
        replacement_source_commit=revision,
        model_optimizer_rng_schedule_exposures_unchanged=True,
    )
    path = atomic_checkpoint(checkpoints, f"terminal_guard_handoff_{stamp}", state)
    atomic_json(
        run / "TERMINAL_GUARD_HANDOFF.json",
        dict(
            checkpoint=str(path),
            original_checkpoint=previous,
            updates=state["updates"],
            epoch=state["epoch"],
            position=state["position"],
            model_optimizer_rng_schedule_exposures_unchanged=True,
            original_runner_sha256=hashlib.sha256(old).hexdigest(),
            replacement_runner_sha256=sha(RUNNER),
            new_source_commit=revision,
        ),
    )
    atomic_json(
        run / "COMMITTED.json",
        dict(
            checkpoint=str(path),
            epoch=state["epoch"],
            position=state["position"],
            updates=state["updates"],
            total_protein_exposures=int(state["exposures"].sum()),
            minimum_exposure=int(state["exposures"].min()),
            maximum_exposure=int(state["exposures"].max()),
        ),
    )
    print(json.dumps(state["source_compatibility_handoff"], indent=2), flush=True)


if __name__ == "__main__":
    main()
