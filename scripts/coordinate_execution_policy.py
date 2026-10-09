"""Explicit user-authorized sequence concurrency; frozen scientific source audit."""

import hashlib
from pathlib import Path

from e014_bootstrap import ROOT

from protein_distance_diffusion.training.e014_runtime import atomic_json, sha

FROZEN_RUNNER_SHA = "3063e842f6cd93f0fdcc8a86d1bcd70b0598b51386e01c8dad0da06dd4608601"
# Each change is control/provenance only. Reversal must recover the exact frozen
# runner, so model/optimization/data/checkpoint science cannot drift unnoticed.
PATCHES = (
    ("from coordinate_execution_policy import execution_policy, sequence_status\n", ""),
    ('    parser.add_argument("--allow-concurrent-sequence", action="store_true")\n', ""),
    ("    policy = execution_policy(contract, args.allow_concurrent_sequence)\n", ""),
    ('        payload["execution_policy"] = policy\n', ""),
    (
        '                    status = sequence_status(compare, before, cfg["contention"], '
        "args.allow_concurrent_sequence)\n",
        '                    status = compare(before, cfg["contention"]) if before else None\n',
    ),
    (
        "                        if (\n"
        '                            status["persistent_material_degradation"] or stale or phase_changed\n'
        "                        ) and not args.allow_concurrent_sequence:\n",
        '                        if status["persistent_material_degradation"] or stale or phase_changed:\n',
    ),
)


def canonical_runner(source):
    for new, old in PATCHES:
        if source.count(new) != 1:
            raise ValueError("unexpected concurrency patch; refusing source compatibility")
        source = source.replace(new, old, 1)
    if hashlib.sha256(source.encode()).hexdigest() != FROZEN_RUNNER_SHA:
        raise ValueError("scientific runner differs from frozen source; refusing resume")
    return source


def execution_policy(contract, allow_concurrent):
    path = Path("scripts/run_e014_scaleup.py")
    canonical_runner(path.read_text())
    contract["source_hashes"][str(path)] = FROZEN_RUNNER_SHA
    policy = dict(
        allow_concurrent_sequence=allow_concurrent,
        sequence_telemetry_only=allow_concurrent,
        explicitly_authorized_by_user=True if allow_concurrent else False,
        frozen_scientific_runner_sha256=FROZEN_RUNNER_SHA,
        actual_execution_runner_sha256=sha(path),
        policy_helper_sha256=sha(__file__),
        scientific_protocol_unchanged=True,
    )
    atomic_json(ROOT / "e014/EXECUTION_POLICY.json", policy)
    return policy


def sequence_status(compare, before, config, allow_concurrent):
    try:
        return compare(before, config) if before else None
    except Exception as exc:
        if not allow_concurrent:
            raise
        atomic_json(ROOT / "e014/SEQUENCE_TELEMETRY_ERROR.json", dict(error=str(exc), blocking=False))
        return None
