"""Frozen V10 CPU scientific execution and fixed-state independent reproduction."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import yaml

from protein_distance_diffusion.training import e010_strict_scientific_v10 as v10
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from protein_distance_diffusion.training.e010_recurrent_capacity import batch
from scripts.run_e010_no_new_inversion_v9 import OUT as V9OUT
from scripts.run_e010_no_new_inversion_v9 import setup as historical_setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = V9OUT.parent / "strict_scientific_v10"
V9B = V9OUT.parent / "strict_constraint_conditioning_v9b"
CACHE = None
CONTRACT = None


def setup():
    cfg = historical_setup()
    physical = yaml.safe_load((ROOT / "configs/e010_phase4d_conditioning_v9b.yaml").read_text())
    conclusion = json.loads((V9B / "conclusion.json").read_text())
    limits = conclusion["future_contract"]
    assert conclusion["classification"] == "COND-A"
    assert (
        limits["stationarity"]
        == "physical normalized L2 residual <= 1e-6/(sqrt(M)*1e-4/0.04), with derivative uncertainty tenfold smaller"
    )
    assert limits["normalized_complementarity_max"] == 1e-6 and limits["dual_negativity_max"] == 1e-8
    return cfg, physical, limits


def data(i):
    return {k: v.double() if v.is_floating_point() else v for k, v in batch(CACHE, [i], "cpu").items()}


def register():
    global CACHE
    cfg, physical, limits = setup()
    CACHE, manifest = load_cache()
    paths = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    paths += [
        "src/protein_distance_diffusion/training/e010_strict_scientific_v10.py",
        "scripts/run_e010_strict_scientific_v10.py",
        "docs/e010_phase4d_strict_scientific_v10.md",
        "tests/test_e010_strict_scientific_v10.py",
        "scripts/report_e010_strict_scientific_v10.py",
    ]
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in paths}
    for module, commit in [
        (
            "src/protein_distance_diffusion/training/e010_no_new_inversion_v9.py",
            "244ac4d1ba4e8375cca30e4172fd44e51c9d0a7f",
        ),
        (
            "src/protein_distance_diffusion/training/e010_conditioning_v9b.py",
            "c0156099973f69461f5dc0d892e94c10441da172",
        ),
    ]:
        old = subprocess.check_output(["git", "show", f"{commit}:{module}"], cwd=ROOT)
        assert hashlib.sha256(old).hexdigest() == file_hash(ROOT / module)
    checks = json.loads((V9OUT / "baseline_constraint_preflight.json").read_text())
    assert (
        checks["all_passed"]
        and checks["examples"] == 60
        and checks["constraint_directions"] == 180
        and checks["aligned_directions"] == 180
    )
    baseline = []
    for i in range(60):
        b = data(i)
        o = v10.v9.Oracle(b)
        z = np.zeros(np.prod(o.shape))
        c = o.cfun(z)
        assert (c <= 0).all()
        q = o.quartets
        from protein_distance_diffusion.models.e010_hybrid_local import PSEUDOSCALAR_INDEX, local_representation

        assert torch.equal(q.q0, local_representation(b["pg"], b["mask"])["features"][:, 1:-2, PSEUDOSCALAR_INDEX])
        baseline.append(dict(index=i, record=CACHE["records"][i], counts=q.counts, exact_feasible=True))
    assert sum(r["counts"]["assessable"] for r in baseline) == 13029
    assert_file_pins(pins)
    write(
        OUT / "execution_contract.json",
        dict(
            solver_config=cfg,
            physical_config=physical,
            physical_limits=limits,
            protected_sha256=pins,
            cache_sha256=manifest["cache_sha256"],
            protected_input_sha256=manifest["protected_input_sha256"],
            order=manifest["order"],
            workers=4,
            callback_check_interval=cfg["history_interval"],
            initialization="exact_zero",
            derivative_uncertainty=(
                "S times best fixed-epsilon objective/Lagrangian directional absolute derivative error; "
                "require threshold/10; validate fixed, projected and Lagrangian directions"
            ),
            historical_derivative_validation_unchanged=True,
            baseline_checks=baseline,
        ),
    )
    print(
        "All60 baselines exactly feasible; all13029 signed quantities bitwise; "
        "historical180+180 derivative validations pinned",
        flush=True,
    )


def build(i, tr, log, z):
    b = data(i)
    record = CACHE["records"][i]
    states = [v10.v9.v8.metric_row(p, b, record, tr, t) for t, p in enumerate(tr["states"])]
    q = v10.v9.QuartetConstraints(b)
    return dict(
        index=i,
        record=record,
        baseline=states[0],
        oracle=states[-1],
        states=states,
        optimizer=log,
        quartet_states=[q.telemetry(p, CONTRACT["physical_config"]["active_normalized_slack"]) for p in tr["states"]],
        correction_direction_telemetry=v10.v9.v8.correction_telemetry(tr),
        variables_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
        state_coordinate_sha256=[hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() for p in tr["states"]],
    )


def example(i, mode):
    b = data(i)
    cfg, physical, limits = setup()
    path = OUT / "B" / f"example_{i:02d}.json"
    statepath = OUT / "untracked_states" / f"example_{i:02d}.npz"
    if mode == "run":
        tr, log, z = v10.solve(b, cfg, physical, limits)
        statepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            statepath,
            z=z,
            multipliers=np.asarray(log["multipliers"]),
            delta=torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy(),
            states=torch.stack([p.detach() for p in tr["states"]]).numpy(),
        )
        result = build(i, tr, log, z)
        result["state_file_sha256"] = file_hash(statepath)
        write(path, result)
    else:
        expected = json.loads(path.read_text())
        assert file_hash(statepath) == expected["state_file_sha256"]
        saved = np.load(statepath)
        z = saved["z"]
        tr = v10.v9.v8.physical_trajectory(
            b["pg"], b["mask"], 0.04 * torch.from_numpy(z.copy()).reshape(8, *b["pg"].shape)
        )
        assert np.array_equal(torch.stack([p.detach() for p in tr["states"]]).numpy(), saved["states"])
        cert = v10.certificate(b, torch.from_numpy(saved["delta"].copy()), saved["multipliers"], physical, limits)
        assert json.loads(json.dumps(cert)) == expected["optimizer"]["physical_certificate"], (
            f"certificate reproduction mismatch {i}"
        )
        rebuilt = build(i, tr, expected["optimizer"], z)
        for key in rebuilt:
            assert json.loads(json.dumps(rebuilt[key])) == expected[key], f"metric reproduction mismatch {i}/{key}"
        write(
            OUT / "reproduction" / f"example_{i:02d}.json",
            dict(index=i, metrics_exact=True, physical_certificate_exact=True, hashes_verified=True),
        )
        result = expected
    return (
        i,
        result["optimizer"]["converged"],
        result["optimizer"]["iterations"],
        result["record"],
        result["optimizer"]["physical_certificate"]["gates"],
    )


def execute(mode):
    global CACHE, CONTRACT
    setup()
    CONTRACT = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    CACHE, manifest = load_cache()
    assert manifest["cache_sha256"] == CONTRACT["cache_sha256"]
    pending = [
        i
        for i in range(60)
        if not (
            OUT / "B" / f"example_{i:02d}.json" if mode == "run" else OUT / "reproduction" / f"example_{i:02d}.json"
        ).exists()
    ]
    with ProcessPoolExecutor(max_workers=CONTRACT["workers"], mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i, mode) for i in pending]):
            print(mode, *future.result(), flush=True)
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    write(
        OUT / ("execution_complete.json" if mode == "run" else "reproduction_complete.json"),
        dict(examples=60, protected_hashes_verified=True, mode=mode, cuda_used=False, neural_training_launched=False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "run", "reproduce"])
    args = parser.parse_args()
    register() if args.mode == "register" else execute(args.mode)
