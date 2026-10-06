#!/usr/bin/env python3
"""Frozen CPU float64 local-feasibility panel and exact result reproduction."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import scipy
import torch
import yaml

from protein_distance_diffusion.training import e010_local_feasibility_v7 as v7
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_precision_kkt_v6 import directional_checks
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = (
    ROOT
    / "reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1/local_feasibility_v7"
)
CONFIG = ROOT / "configs/e010_phase4d_local_feasibility_v7.yaml"
CACHE = None
CFG = None
MODE = None


def setup():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["solver"]["scipy_version"] == scipy.__version__
    assert cfg["recurrence"]["K"] == 4 and cfg["recurrence"]["s_max_angstrom"] == 0.04
    assert cfg["objective"] == "local_mean_only"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return cfg


def load_example(i):
    return {k: v.double() if v.is_floating_point() else v for k, v in batch(CACHE, [i], "cpu").items()}


def register():
    global CACHE, CFG
    CFG = setup()
    CACHE, manifest = load_cache()
    paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in paths}
    for p in (
        CONFIG,
        ROOT / "docs/e010_phase4d_local_feasibility_v7.md",
        ROOT / "src/protein_distance_diffusion/training/e010_local_feasibility_v7.py",
        ROOT / "scripts/run_e010_local_feasibility_v7.py",
        ROOT / "tests/test_e010_local_feasibility_v7.py",
    ):
        pins[str(p)] = file_hash(p)
    write(
        OUT / "execution_contract.json",
        dict(
            config=CFG,
            protected_sha256=pins,
            cache_sha256=manifest["cache_sha256"],
            order=manifest["order"],
            source_result_commit="36113e708194d280ead04af4359e6a8d599a6955",
        ),
    )
    checks = []
    for i in range(60):
        b = load_example(i)
        p = b["pg"].clone().requires_grad_()

        def fun(x, data=b):
            return v7.aligned_rmsd(x, data["target"], data["mask"])

        g = torch.autograd.grad(fun(p), p)[0]
        fd = directional_checks(fun, p.detach(), g, CFG["gradient_validation"]["epsilons_angstrom"])
        failed = [j for j in range(3) if not any(r["passed"] for r in fd if r["direction"] == j)]
        checks.append(dict(index=i, record=CACHE["records"][i], checks=fd, failed_directions=failed))
        if failed:
            write(OUT / "aligned_gradient_blocker.json", dict(index=i, checks=fd, failed_directions=failed))
            raise RuntimeError("Aligned RMSD float64 gradient unreliable; STOP before any optimization")
    assert_file_pins(pins)
    write(OUT / "aligned_gradient_preflight.json", dict(examples=60, directions=180, all_passed=True, records=checks))
    print("Registered and aligned gradients passed for all60 examples", flush=True)


def digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def example(i, arm):
    b = load_example(i)
    record = CACHE["records"][i]
    tr, log, z = v7.solve(b, arm, CFG)
    states = [v7.metric_row(p, b, record, tr, t) for t, p in enumerate(tr["states"])]
    assert all(
        s["finite"]
        and s["step_correction_max"] <= 0.04
        and s["displacement_max"] <= 0.160001
        and s["path_length_max"] <= 0.160001
        for s in states
    )
    result = dict(
        index=i,
        arm=arm,
        record=record,
        baseline=states[0],
        oracle=states[-1],
        states=states,
        optimizer=log,
        variables_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
        state_coordinate_sha256=[digest(p) for p in tr["states"]],
        temporary_assessability_losses=[s["chirality_assessability_lost"] for s in states],
        frame_eligibility=[s["frame_eligible"] for s in states],
    )
    path = OUT / arm / f"example_{i:02d}.json"
    if MODE == "reproduce":
        expected = json.loads(path.read_text())
        x, y = json.loads(json.dumps(result)), json.loads(json.dumps(expected))
        for r in (x, y):
            r["optimizer"].pop("runtime_seconds")
        assert x == y, f"non-timing result reproduction mismatch {arm}/{i}"
        write(
            OUT / "reproduction" / arm / f"example_{i:02d}.json",
            dict(
                index=i,
                arm=arm,
                exact_non_timing_match=True,
                scientific_record_sha256=hashlib.sha256(
                    json.dumps(x, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
            ),
        )
    else:
        directory = OUT / "untracked_states" / arm
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(z=torch.from_numpy(z.copy()), states=[p.detach() for p in tr["states"]]),
            directory / f"example_{i:02d}.pt",
        )
        write(path, result)
    return (
        arm,
        i,
        log["converged"],
        log["iterations"],
        log["constraints"],
        log["stationarity"]["normalized_ball_kkt_max"],
    )


def execute(mode):
    global CACHE, CFG, MODE
    CFG = setup()
    MODE = mode
    contract = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(contract["protected_sha256"])
    assert json.loads((OUT / "aligned_gradient_preflight.json").read_text())["all_passed"]
    CACHE, manifest = load_cache()
    pending = [
        (i, arm)
        for i in range(60)
        for arm in ("A", "B")
        if not (
            OUT / "reproduction" / arm / f"example_{i:02d}.json"
            if mode == "reproduce"
            else OUT / arm / f"example_{i:02d}.json"
        ).exists()
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i, arm) for i, arm in pending]):
            print(mode, *future.result(), flush=True)
    assert_file_pins(contract["protected_sha256"])
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        OUT / ("reproduction_complete.json" if mode == "reproduce" else "execution_complete.json"),
        dict(
            examples=60,
            arms=2,
            results=120,
            protected_hashes_verified=True,
            exact_non_timing_reproduction=mode == "reproduce",
            cuda_used=False,
            neural_training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "run", "reproduce"])
    args = parser.parse_args()
    register() if args.mode == "register" else execute(args.mode)
