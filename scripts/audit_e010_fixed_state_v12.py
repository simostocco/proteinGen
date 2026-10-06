"""All60 V11 fixed-state audit with immutable provenance and forbidden optimizer guard."""

import argparse
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import yaml

from protein_distance_diffusion.training import e010_fixed_state_audit_v12 as audit
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_direct_correction_v11 as historical
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, write

OUT = historical.OUT.parent / "v11_fixed_state_numerical_audit_v12"
CONFIG = ROOT / "configs/e010_phase4d_fixed_state_audit_v12.yaml"
BASE = "4a82eff5169298022683f47cd345af5ceae02a2c"
CONTRACT = None


def setup():
    assert (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        == "e010-phase4d-hybrid-local-global"
    )
    historical.frozen()
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["nonlinear_optimizer_invocations"] == 0 and cfg["K"] == 8 and cfg["s_max_angstrom"] == 0.04
    return cfg


def load_fixed(i):
    path = historical.OUT / "B" / f"example_{i:02d}.json"
    content = path.read_bytes()
    committed = subprocess.check_output(["git", "show", f"{BASE}:{path.relative_to(ROOT)}"], cwd=ROOT)
    assert content == committed, f"Historical record mutation {i}"
    row = json.loads(content)
    statepath = historical.OUT / "untracked_states" / f"example_{i:02d}.npz"
    assert file_hash(statepath) == row["state_file_sha256"], f"Fixed state hash mismatch {i}"
    b = historical.data(i)
    with np.load(statepath) as saved:
        z = saved["z"].copy()
        delta = saved["delta"].copy()
        mu = saved["multipliers"].copy()
        tr = historical.direct.trajectory(b["pg"], b["mask"], torch.from_numpy(z).reshape(8, *b["pg"].shape))
        assert np.array_equal(torch.stack(tr["states"]).detach().numpy(), saved["states"])
        assert np.array_equal(torch.stack([s["delta"] for s in tr["steps"]]).numpy(), delta)
    assert hashlib.sha256(z.tobytes()).hexdigest() == row["variables_sha256"]
    rebuilt = historical.build(i, tr, row["optimizer"], z)
    for key in rebuilt:
        assert json.loads(json.dumps(rebuilt[key])) == row[key], f"Metric/coordinate reproduction {i}/{key}"
    oracle = historical.direct.Oracle(b)
    point = oracle.point(z)
    assert float(point.objective.detach()) == row["optimizer"]["history"][-1]["normalized_objective"]
    assert row["optimizer"]["iterations"] == row["optimizer"]["history"][-1]["iteration"]
    np.testing.assert_array_equal(oracle.cfun(z), row["optimizer"]["constraints"])
    np.testing.assert_array_equal(oracle.ball_values(z), row["optimizer"]["ball_constraint_values"])
    np.testing.assert_array_equal(mu, row["optimizer"]["multipliers"])
    return b, row, torch.from_numpy(delta), mu, statepath


def group(row):
    if row["optimizer"]["converged"]:
        return "A"
    return "B" if any(s["material"] for s in row["optimizer"]["physical_certificate"]["shadow_steps"]) else "C"


def register():
    cfg = setup()
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    names += [
        str(CONFIG.relative_to(ROOT)),
        "docs/e010_phase4d_fixed_state_audit_v12.md",
        "src/protein_distance_diffusion/training/e010_fixed_state_audit_v12.py",
        "scripts/audit_e010_fixed_state_v12.py",
        "scripts/report_e010_fixed_state_v12.py",
        "tests/test_e010_fixed_state_v12.py",
    ]
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in set(names)}
    cases = []
    with audit.forbid_nonlinear_optimization():
        for i in range(60):
            _, row, _, _, path = load_fixed(i)
            pins[str(path)] = file_hash(path)
            cases.append(
                dict(
                    index=i,
                    record=row["record"],
                    group=group(row),
                    record_sha256=file_hash(historical.OUT / "B" / f"example_{i:02d}.json"),
                    fixed_state_sha256=file_hash(path),
                )
            )
            print("fixed state verified", i, flush=True)
    assert sum(r["group"] == "A" for r in cases) == 4
    assert sum(r["group"] == "B" for r in cases) == 55
    assert sum(r["group"] == "C" for r in cases) == 1
    from scipy.optimize._trustregion_constr import minimize_trustregion_constr, tr_interior_point

    installed = {str(m.__file__): file_hash(m.__file__) for m in [minimize_trustregion_constr, tr_interior_point]}
    pins.update(installed)
    assert_file_pins(pins)
    write(
        OUT / "preregistration.json",
        dict(
            config=cfg,
            cases=cases,
            protected_sha256=pins,
            protected_input_sha256=historical.CONTRACT["protected_input_sha256"],
            physical_config=historical.CONTRACT["physical_config"],
            physical_limits=historical.CONTRACT["physical_limits"],
            historical_solver_config=historical.CONTRACT["solver_config"],
            installed_solver_sources=installed,
            optimizer_invocations=0,
            cuda_used=False,
            neural_training_launched=False,
        ),
    )


def evidence(record, config):
    windows = record["history_windows"]
    w = windows["100"]
    objective = w["normalized_objective"]
    stat = w["physical_stationarity"]
    direction = record["direction_distribution"]
    power = direction["tangential_l2"] ** 2 / max(direction["total_l2"] ** 2, 1e-300)
    return dict(
        meaningful_objective_progress_last100=bool(
            objective["available"] and -objective["change"] >= config["material_normalized_local_decrease"]
        ),
        stationarity_drop_at_least_one_percent_last100=bool(
            stat["available"] and stat["relative_change"] is not None and stat["relative_change"] <= -0.01
        ),
        tangential_direction_power_fraction=power,
        tangential_direction_majority=power >= 0.5,
        trust_radius_at_or_below_historical_xtol=record["telemetry"]["trust_radius"]["value"]
        <= CONTRACT["historical_solver_config"]["solver"]["xtol"],
        barrier_reduced_in_last250=windows["250"]["barrier_parameter"]["change"] < 0,
    )


def example(i):
    before = file_hash(historical.OUT / "untracked_states" / f"example_{i:02d}.npz")
    with audit.forbid_nonlinear_optimization():
        b, row, delta, mu, path = load_fixed(i)
        result = audit.local_analysis(
            b, delta, mu, row["optimizer"], CONTRACT["physical_config"], CONTRACT["physical_limits"]
        )
    assert json.loads(json.dumps(result["physical_certificate"])) == row["optimizer"]["physical_certificate"], (
        f"STOP certificate mismatch {i}"
    )
    result.update(
        index=i,
        record=row["record"],
        group=group(row),
        telemetry=audit.telemetry(row["optimizer"]),
        history_windows=audit.history_windows(row["optimizer"], CONTRACT["config"]["history_windows"]),
        fixed_state_sha256=before,
        coordinate_hashes_exact=True,
        metrics_exact=True,
        objective_and_constraints_exact=True,
        physical_certificate_exact=True,
        iteration_and_status_committed_metadata_verified=True,
        optimizer_invocations=0,
    )
    result["evidence_flags"] = evidence(result, CONTRACT["physical_config"])
    assert file_hash(path) == before
    write(OUT / "cases" / f"example_{i:02d}.json", result)
    return dict(index=i, group=group(row), reproduced=True, optimizer_invocations=0)


def execute():
    global CONTRACT
    setup()
    CONTRACT = json.loads((OUT / "preregistration.json").read_text())
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i) for i in range(60)]):
            result = future.result()
            results.append(result)
            print("audited", result, flush=True)
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    write(
        OUT / "audit_complete.json",
        dict(
            examples=60,
            results=sorted(results, key=lambda r: r["index"]),
            protected_hashes_verified=True,
            optimizer_invocations=0,
            cuda_used=False,
            neural_training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "audit"])
    register() if parser.parse_args().mode == "register" else execute()
