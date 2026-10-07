"""Preregistered sparse preflights, gated once-only panel, and fixed-state reproduction."""

import argparse
import hashlib
import json
import platform
import subprocess

import numpy as np
import scipy
import torch
import yaml

from protein_distance_diffusion.training import e010_sparse_active_v15 as solver
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_direct_correction_v11 as historical
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = historical.OUT.parent / "sparse_feasible_active_v15"
ADDED = [
    "src/protein_distance_diffusion/training/e010_sparse_active_v15.py",
    "scripts/run_e010_sparse_active_v15.py",
    "tests/test_e010_sparse_active_v15.py",
    "configs/e010_phase4d_sparse_active_v15.yaml",
    "docs/e010_phase4d_sparse_active_v15.md",
]


def register():
    assert (
        subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
        == "e010-phase4d-hybrid-local-global"
    )
    config = yaml.safe_load((ROOT / ADDED[3]).read_text())
    assert config["solver"] == solver.SETTINGS
    _, physical, limits, _ = historical.setup()
    cache, manifest = load_cache()
    historical.CACHE = cache
    previous = json.loads((historical.OUT / "execution_contract.json").read_text())
    validation = []
    for i in range(60):
        record = dict(index=i, record=cache["records"][i], **historical.validate(historical.data(i)))
        assert record == previous["validation"][i]
        validation.append(record)
        print("validated", i, flush=True)
    files = set(subprocess.check_output(["git", "ls-files"], text=True).splitlines() + ADDED)
    pins = {str(ROOT / f): file_hash(ROOT / f) for f in files}
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        OUT / "execution_contract.json",
        dict(
            config=config,
            physical_config=physical,
            physical_limits=limits,
            protected_sha256=pins,
            protected_input_sha256=manifest["protected_input_sha256"],
            cache_sha256=manifest["cache_sha256"],
            validation=validation,
            runtime=dict(
                python=platform.python_version(),
                scipy=scipy.__version__,
                CPU_threads=1,
                external_sparse_QP_installed=False,
            ),
            order=manifest["order"],
        ),
    )


def frozen():
    c = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(c["protected_sha256"])
    assert_file_pins(c["protected_input_sha256"])
    cache, manifest = load_cache()
    assert manifest["cache_sha256"] == c["cache_sha256"]
    historical.CACHE, historical.CONTRACT = cache, c
    return c


def raw_save(name, x, meta):
    path = OUT / "untracked_states" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez(path, z=x)
    write(
        path.with_suffix(".json"),
        dict(**meta, file_sha256=file_hash(path), variable_sha256=hashlib.sha256(x.tobytes()).hexdigest()),
    )


def run_case(name, b, c, smoke=False):
    tr, log, x = solver.solve(
        b,
        c["physical_config"],
        c["physical_limits"],
        lambda x, meta: raw_save(name, x, meta),
        smoke_steps=25 if smoke else None,
    )
    delta = torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy()
    states = torch.stack([s.detach() for s in tr["states"]]).numpy()
    path = OUT / "untracked_states" / f"{name}_certified.npz"
    np.savez(path, z=x, delta=delta, states=states, multipliers=np.asarray(log["multipliers"]))
    o = solver.direct.Oracle(b)
    baseline = float(o.fun(np.zeros_like(x))[0])
    final = float(o.fun(x)[0])
    report = dict(
        name=name,
        optimizer=log,
        initial_normalized_objective=baseline,
        final_normalized_objective=final,
        exact_primal_feasible=solver.feasible(o, x),
        objective_decreased=final < baseline,
        state_file_sha256=file_hash(path),
        quartet_telemetry=o.quartets.telemetry(tr["prediction"], 1e-5),
        derivative_validation=historical.validate(b),
    )
    write(OUT / f"{name}.json", report)
    return report


def reproduce_case(name, b, c):
    record = json.loads((OUT / f"{name}.json").read_text())
    path = OUT / "untracked_states" / f"{name}_certified.npz"
    assert file_hash(path) == record["state_file_sha256"]
    with np.load(path) as saved:
        x, delta, states = saved["z"], saved["delta"], saved["states"]
    o = solver.direct.Oracle(b)
    tr = o.point(x).tr
    assert np.array_equal(states, torch.stack([p.detach() for p in tr["states"]]).numpy())
    assert np.array_equal(delta, torch.stack([p["delta"].detach() for p in tr["steps"]]).numpy())
    cert, mu, reconstruction = solver.certify(b, o, x, c["physical_config"], c["physical_limits"])
    assert json.loads(json.dumps(cert)) == record["optimizer"]["physical_certificate"]
    assert json.loads(json.dumps(reconstruction)) == record["optimizer"]["multiplier_reconstruction"]
    assert float(o.fun(x)[0]) == record["final_normalized_objective"]
    assert json.loads(json.dumps(o.quartets.telemetry(tr["prediction"], 1e-5))) == record["quartet_telemetry"]
    assert solver.feasible(o, x) == record["exact_primal_feasible"]
    return dict(
        name=name,
        coordinates_exact=True,
        corrections_exact=True,
        objective_exact=True,
        certificate_exact=True,
        constraints_exact=True,
        hashes_exact=True,
        optimizer_invocations=0,
    )


def preflight():
    c = frozen()
    for n in [12, 32, 500]:
        run_case(f"preflight_{n}", synthetic(n), c, smoke=n == 500)


def adjudicate():
    c = frozen()
    reproduction = [reproduce_case(f"preflight_{n}", synthetic(n), c) for n in [12, 32, 500]]
    records = [json.loads((OUT / f"preflight_{n}.json").read_text()) for n in [12, 32, 500]]
    gates = dict(
        small_cases_physical_convergence=all(r["optimizer"]["converged"] for r in records[:2]),
        exact_primal_feasibility=all(r["exact_primal_feasible"] for r in records),
        objective_decrease=all(r["objective_decreased"] for r in records),
        memory=all(r["optimizer"]["process_peak_rss_bytes"] < c["config"]["peak_rss_limit_bytes"] for r in records),
        implementation_valid=all(not r["optimizer"]["termination"].startswith("solver_exception") for r in records),
        independent_reproduction=True,
    )
    write(
        OUT / "preflight_adjudication.json",
        dict(gates=gates, passed=all(gates.values()), independent_reproduction=reproduction),
    )
    if not all(gates.values()):
        write(
            OUT / "conclusion.json",
            dict(
                classification="ACTIVE-D",
                numerical_classification="SPARSE-D",
                reason="Frozen numerical preflight gate failed; no scientific panel launched",
                full_panel_launched=False,
                physical_convergence=None,
                material_shadow=None,
                certified_sufficient=False,
                oracle_feasibility_program_complete=False,
                CUDA_used=False,
                neural_training_launched=False,
                new_dependency_installed=False,
                next_experiment=("Benchmark a sparse local-QP method with reliable primal-dual convergence "
                    "on the same frozen synthetic strict cases before any scientific panel."),
            ),
        )
        return
    for i in range(60):
        tr_record = run_case(f"panel_{i:02d}", historical.data(i), c)
        print(i, tr_record["optimizer"]["converged"], flush=True)
    from scripts.report_e010_local_feasibility_v7 import percent, summarize

    rows = []
    for i in range(60):
        name = f"panel_{i:02d}"
        reproduce_case(name, historical.data(i), c)
        record = json.loads((OUT / f"{name}.json").read_text())
        with np.load(OUT / "untracked_states" / f"{name}_certified.npz") as state:
            x = state["z"].copy()
        tr = solver.direct.Oracle(historical.data(i)).point(x).tr
        row = historical.build(i, tr, record["optimizer"], x)
        write(OUT / "B" / f"example_{i:02d}.json", row)
        rows.append(row)
    baseline, final = (summarize([r[k] for r in rows]) for k in ["baseline", "oracle"])
    conditions = {
        condition: dict(
            local_gain_pct=-percent(
                final["by_condition"][condition], baseline["by_condition"][condition], "mean_local_rmse"
            )
        )
        for condition in ["50", "250", "450"]
    }
    count = sum(r["optimizer"]["converged"] for r in rows)
    material = sum(any(s["material"] for s in r["optimizer"]["physical_certificate"]["shadow_steps"]) for r in rows)
    safety = all(
        r["quartet_states"][-1]["new_inversions"] == 0
        and r["quartet_states"][-1]["assessability_lost"] == 0
        and r["optimizer"]["physical_certificate"]["gates"]["primal_feasibility"]
        for r in rows
    )
    complete = count == 60 and safety and conditions["450"]["local_gain_pct"] >= 5
    major = count >= 30 and material <= 25
    write(
        OUT / "conclusion.json",
        dict(
            classification="ACTIVE-A" if complete else ("ACTIVE-B" if major else "ACTIVE-C"),
            numerical_classification="SPARSE-A" if count == 60 else ("SPARSE-B" if major else "SPARSE-C"),
            full_panel_launched=True,
            physical_convergence=count,
            material_shadow=material,
            conditions=conditions,
            baseline=baseline,
            final=final,
            certified_sufficient=complete,
            oracle_feasibility_program_complete=complete,
            CUDA_used=False,
            neural_training_launched=False,
            new_dependency_installed=False,
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["register", "preflight", "adjudicate"])
    globals()[parser.parse_args().action]()
