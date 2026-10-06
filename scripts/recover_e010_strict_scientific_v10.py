"""Authorized, once-only V10 technical replay and complete-panel reproduction."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_strict_scientific_v10 as original

BASE = "f90ea80e57bd1239d5d21049bbd11df8b512be60"
OUT = original.OUT / "technical_recovery"
MODULE = "src/protein_distance_diffusion/training/e010_strict_scientific_v10.py"
INDICES = [0, 5, 15, 16, 17, 18, 24, 29, 30, 32, 37, 47, 51, 53, 55, 56, 59]
CONTROLS = [8, 9, 13, 42]
OLD_LINE = "maximum_linearized_cone_violation=float(np.max(a @ direction)),"
NEW_LINE = "maximum_linearized_cone_violation=float(np.max(a @ direction)) if len(a) else 0.0,"


def canonical(value):
    return json.loads(json.dumps(value))


def setup():
    cfg, physical, limits = original.setup()
    old_source = subprocess.check_output(["git", "show", f"{BASE}:{MODULE}"], text=True)
    assert old_source.count(OLD_LINE) == 1
    assert (original.ROOT / MODULE).read_text() == old_source.replace(OLD_LINE, NEW_LINE)
    original.CONTRACT = json.loads((original.OUT / "execution_contract.json").read_text())
    pins = original.CONTRACT["protected_sha256"].copy()
    old_module = str(original.ROOT / MODULE)
    assert hashlib.sha256(old_source.encode()).hexdigest() == pins.pop(old_module)
    assert_file_pins(pins)
    assert_file_pins(original.CONTRACT["protected_input_sha256"])
    original.CACHE, manifest = original.load_cache()
    assert manifest["cache_sha256"] == original.CONTRACT["cache_sha256"]
    return cfg, physical, limits, old_source


def rebuild(i, record, statepath, *, compare_original=False):
    _, physical, limits, _ = setup()
    assert file_hash(statepath) == record["state_file_sha256"]
    saved = np.load(statepath)
    b = original.data(i)
    tr = original.v10.v9.v8.physical_trajectory(
        b["pg"], b["mask"], 0.04 * torch.from_numpy(saved["z"].copy()).reshape(8, *b["pg"].shape)
    )
    assert np.array_equal(torch.stack([p.detach() for p in tr["states"]]).numpy(), saved["states"])
    certificate = original.v10.certificate(
        b, torch.from_numpy(saved["delta"].copy()), saved["multipliers"], physical, limits
    )
    assert canonical(certificate) == record["optimizer"]["physical_certificate"]
    assert certificate["converged"] == record["optimizer"]["converged"]
    rebuilt = original.build(i, tr, record["optimizer"], saved["z"])
    for key, value in rebuilt.items():
        assert canonical(value) == record[key], f"reproduction mismatch {i}/{key}"
    if compare_original:
        namespace = vars(original.v10).copy()
        old_source = subprocess.check_output(["git", "show", f"{BASE}:{MODULE}"], text=True)
        exec(compile(old_source, "<original-v10-telemetry>", "exec"), namespace)
        old_certificate = namespace["certificate"](
            b, torch.from_numpy(saved["delta"].copy()), saved["multipliers"], physical, limits
        )
        assert canonical(old_certificate) == canonical(certificate)
    return b, saved, tr, certificate


def register():
    cfg, physical, limits, old_source = setup()
    failure = json.loads((original.OUT / "original_execution_failure.json").read_text())
    assert failure["missing_indices"] == INDICES
    historical = [i for i in range(60) if i not in INDICES]
    assert failure["saved_indices"] == historical and len(historical) == 43
    control_results = []
    for i in CONTROLS:
        path = original.OUT / "B" / f"example_{i:02d}.json"
        record = json.loads(path.read_text())
        before = file_hash(path)
        b, saved, tr, cert = rebuild(
            i, record, original.OUT / "untracked_states" / f"example_{i:02d}.npz", compare_original=True
        )
        point = original.v10.v9.Oracle(b).point(saved["z"])
        log = record["optimizer"]
        assert log["history"][-1]["iteration"] == log["iterations"]
        assert float(point.objective.detach()) == log["history"][-1]["normalized_objective"]
        assert log["scipy_status"] == 3 and cert["converged"]
        assert file_hash(path) == before
        control_results.append(
            dict(
                index=i,
                record=record["record"],
                final_coordinate_hash=record["state_coordinate_sha256"][-1],
                iterations=log["iterations"],
                scipy_status=log["scipy_status"],
                normalized_objective=float(point.objective.detach()),
                metrics_exact=True,
                certificate_exact=True,
                original_vs_patched_certificate_exact=True,
                metadata_file_unchanged=True,
                optimizer_executed=False,
            )
        )
    # Persisted control metadata and its trajectory are verified without forbidden solver reruns.
    # Every original certificate call that returned had a nonempty active set; that branch is identical.
    pins = {}
    for name in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
        pins[str(original.ROOT / name)] = file_hash(original.ROOT / name)
    for name in [
        "scripts/recover_e010_strict_scientific_v10.py",
        "tests/test_e010_v10_technical_recovery.py",
        "scripts/publish_e010_v10_technical_recovery.py",
        "docs/e010_v10_technical_recovery.md",
    ]:
        pins[str(original.ROOT / name)] = file_hash(original.ROOT / name)
    for i in historical:
        p = original.OUT / "untracked_states" / f"example_{i:02d}.npz"
        record = json.loads((original.OUT / "B" / f"example_{i:02d}.json").read_text())
        assert file_hash(p) == record["state_file_sha256"]
        pins[str(p)] = file_hash(p)
        target = OUT / "B" / f"example_{i:02d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        target.write_bytes((original.OUT / "B" / target.name).read_bytes())
    original.write(
        OUT / "contract.json",
        dict(
            original_result_commit=BASE,
            replay_indices=INDICES,
            historical_indices=historical,
            controls=control_results,
            solver_config=cfg,
            physical_config=physical,
            physical_limits=limits,
            original_module_sha256=hashlib.sha256(old_source.encode()).hexdigest(),
            patched_module_sha256=file_hash(original.ROOT / MODULE),
            only_scientific_source_change=NEW_LINE,
            protected_sha256=pins,
            protected_input_sha256=original.CONTRACT["protected_input_sha256"],
            cache_sha256=original.CONTRACT["cache_sha256"],
            workers=4,
            initialization="exact_zero",
            maximum_solver_calls_per_replay_index=1,
            state_label="technical recovery replay state",
            historical_solver_reruns=0,
            iteration_status_parity_method=(
                "saved metadata/hash preservation and identical nonempty certificate branch; no control optimization"
            ),
        ),
    )
    print(
        "Control hashes/iterations/status/objective/metrics/certificates verified; exactly17 replays frozen", flush=True
    )


def replay(i):
    cfg, physical, limits, _ = setup()
    assert i in INDICES
    original.write(
        OUT / "attempts" / f"example_{i:02d}.json",
        dict(
            index=i,
            state_provenance="technical recovery replay state",
            solver_invocation=1,
            zero_initialization=True,
            original_state_available=False,
        ),
    )
    try:
        tr, log, z = original.v10.solve(original.data(i), cfg, physical, limits)
        statepath = OUT / "untracked_states" / f"example_{i:02d}.npz"
        statepath.parent.mkdir(parents=True, exist_ok=True)
        assert not statepath.exists()
        np.savez(
            statepath,
            z=z,
            multipliers=np.asarray(log["multipliers"]),
            delta=torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy(),
            states=torch.stack([p.detach() for p in tr["states"]]).numpy(),
        )
        result = original.build(i, tr, log, z)
        result.update(
            state_file_sha256=file_hash(statepath),
            state_provenance="technical recovery replay state",
            solver_invocations=1,
        )
        original.write(OUT / "B" / f"example_{i:02d}.json", result)
        return dict(index=i, saved=True, converged=log["converged"], iterations=log["iterations"])
    except Exception:
        error = dict(index=i, saved=False, solver_invocations=1, traceback=traceback.format_exc())
        original.write(OUT / "errors" / f"example_{i:02d}.json", error)
        return error


def reproduce(i):
    record = json.loads((OUT / "B" / f"example_{i:02d}.json").read_text())
    root = OUT if i in INDICES else original.OUT
    rebuild(i, record, root / "untracked_states" / f"example_{i:02d}.npz")
    original.write(
        OUT / "reproduction" / f"example_{i:02d}.json",
        dict(
            index=i,
            metrics_exact=True,
            physical_certificate_exact=True,
            hashes_verified=True,
            state_provenance="technical recovery replay state" if i in INDICES else "historical saved original state",
        ),
    )
    return dict(index=i, reproduced=True)


def execute(mode):
    setup()
    contract = json.loads((OUT / "contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    if mode == "run":
        assert not list((OUT / "attempts").glob("*.json")), "once-only replay already attempted"
        indices, function = INDICES, replay
    else:
        assert all((OUT / "B" / f"example_{i:02d}.json").exists() for i in range(60))
        indices, function = list(range(60)), reproduce
    results = []
    with ProcessPoolExecutor(max_workers=contract["workers"], mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(function, i) for i in indices]):
            result = future.result()
            results.append(result)
            print(mode, result, flush=True)
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(contract["protected_input_sha256"])
    original.write(
        OUT / ("run_complete.json" if mode == "run" else "reproduction_complete.json"),
        dict(
            examples=len(indices),
            results=sorted(results, key=lambda r: r["index"]),
            protected_hashes_verified=True,
            historical_solver_reruns=0,
            cuda_used=False,
            neural_training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "run", "reproduce"])
    mode = parser.parse_args().mode
    register() if mode == "register" else execute(mode)
