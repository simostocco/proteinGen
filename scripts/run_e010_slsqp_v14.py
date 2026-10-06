"""Frozen SLSQP registration, preflight/resource gate, once-only panel and reproduction."""

import argparse
import hashlib
import inspect
import json
import math
import platform
import resource
import signal
import subprocess
import time

import numpy as np
import scipy
import scipy.optimize._slsqp_py as slsqp_api
import torch
import yaml

from protein_distance_diffusion.training import e010_slsqp_v14 as sqp
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_direct_correction_v11 as historical
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = historical.OUT.parent / "direct_space_slsqp_v14"
CONFIG = ROOT / "configs/e010_phase4d_slsqp_v14.yaml"
CONTRACT = CACHE = None
ADDED = [
    "configs/e010_phase4d_slsqp_v14.yaml",
    "docs/e010_phase4d_slsqp_v14.md",
    "src/protein_distance_diffusion/training/e010_slsqp_v14.py",
    "scripts/run_e010_slsqp_v14.py",
    "scripts/report_e010_slsqp_v14.py",
    "tests/test_e010_slsqp_v14.py",
]


def setup():
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == (
        "e010-phase4d-hybrid-local-global"
    )
    own = yaml.safe_load(CONFIG.read_text())
    _, physical, limits, _ = historical.setup()
    previous = json.loads((historical.OUT.parent / "matched_direct_space_v13" / "execution_contract.json").read_text())
    assert physical == previous["physical_config"] and limits == previous["physical_limits"]
    assert own["solver"] == sqp.SETTINGS and own["K"] == 8 and own["s_max_angstrom"] == 0.04
    assert own["constraint_scaling"] == "none_beyond_unchanged_historical_normalization"
    assert scipy.__version__ == "1.18.1", "Workspace estimate must match the installed/pinned implementation"
    return own, physical, limits


def data(i):
    historical.CACHE = CACHE
    return historical.data(i)


def register():
    global CACHE
    own, physical, limits = setup()
    CACHE, manifest = load_cache()
    files = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines() + ADDED)
    pins = {str(ROOT / f): file_hash(ROOT / f) for f in files}
    sources = [inspect.getsourcefile(slsqp_api), scipy.optimize._slsqplib.__file__]
    pins.update({p: file_hash(p) for p in sources})
    prior = json.loads((historical.OUT / "execution_contract.json").read_text())
    records = []
    for i in range(60):
        record = dict(index=i, record=CACHE["records"][i], **historical.validate(data(i)))
        assert record == prior["validation"][i]
        records.append(record)
        print("validated", i, flush=True)
    assert sum(r["quartets"]["assessable"] for r in records) == 13029
    assert_file_pins(pins)
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        OUT / "execution_contract.json",
        dict(
            config=own,
            physical_config=physical,
            physical_limits=limits,
            protected_sha256=pins,
            protected_input_sha256=manifest["protected_input_sha256"],
            cache_sha256=manifest["cache_sha256"],
            order=manifest["order"],
            validation=records,
            runtime=dict(
                python=platform.python_version(),
                python_detail=platform.python_build(),
                scipy=scipy.__version__,
                slsqp_signature=str(inspect.signature(slsqp_api._minimize_slsqp)),
                analytic_objective_jacobian=True,
                vector_analytic_constraint_jacobians=True,
                solver_multipliers_available=True,
                historical_hvp_used=False,
                CPU_threads=1,
            ),
            scientific_outcome_tuning=False,
            panel_optimizer_invocations_per_example=1,
        ),
    )


def frozen():
    global CONTRACT, CACHE
    own, physical, limits = setup()
    CONTRACT = json.loads((OUT / "execution_contract.json").read_text())
    assert (
        own == CONTRACT["config"] and physical == CONTRACT["physical_config"] and limits == CONTRACT["physical_limits"]
    )
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    CACHE, manifest = load_cache()
    assert manifest["cache_sha256"] == CONTRACT["cache_sha256"]
    historical.CACHE = CACHE
    historical.CONTRACT = CONTRACT


def save_raw(path, result, elapsed):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez(
        path,
        z=np.asarray(result.x, dtype=np.float64),
        solver_multipliers=np.asarray(getattr(result, "multipliers", []), dtype=np.float64),
    )
    write(
        path.with_suffix(".json"),
        dict(
            state_file_sha256=file_hash(path),
            variables_sha256=hashlib.sha256(np.asarray(result.x).tobytes()).hexdigest(),
            iterations=int(result.nit),
            status=int(result.status),
            success=bool(result.success),
            message=str(result.message),
            objective=float(result.fun),
            runtime_seconds=elapsed,
        ),
    )


def solve_saved(b, name):
    raw = OUT / "untracked_states" / f"{name}.npz"
    tr, log, z = sqp.solve(
        b,
        CONTRACT["physical_config"],
        CONTRACT["physical_limits"],
        lambda result, elapsed: save_raw(raw, result, elapsed),
    )
    with np.load(raw) as r:
        returned = r["solver_multipliers"].copy()
    final = raw.with_name(f"{name}_certified.npz")
    np.savez(
        final,
        z=z,
        multipliers=np.asarray(log["multipliers"]),
        solver_multipliers=returned,
        delta=torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy(),
        states=torch.stack([p.detach() for p in tr["states"]]).numpy(),
    )
    return tr, log, z, final


def available_ram():
    values = {line.split(":")[0]: line.split(":")[1].strip() for line in open("/proc/meminfo")}
    available = int(values["MemAvailable"].split()[0]) * 1024
    # Respect a discoverable tighter cgroup limit, without relying on swap.
    from pathlib import Path

    for maximum, current in [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]:
        if Path(maximum).exists() and Path(current).exists() and Path(maximum).read_text().strip() != "max":
            available = min(available, max(0, int(Path(maximum).read_text()) - int(Path(current).read_text())))
    return available


def resource_smoke():
    started = time.perf_counter()
    b = synthetic(500)
    oracle = sqp.direct.Oracle(b)
    x = np.zeros(math.prod(oracle.shape))
    m = len(oracle.cfun(x)) + len(x) // 3
    required = sqp.required_arrays(len(x), m)
    gate = sqp.ram_gate(required, available_ram(), CONTRACT["config"]["resource_policy"])
    validation = historical.validate(b)
    wrapper = sqp.Constraints(oracle)
    values, jac = wrapper.science(x), wrapper.science_jac(x)
    assert np.isfinite(jac).all() and (values >= 0).all()
    record = dict(
        length=500,
        source="unchanged historical synthetic(500)",
        zero_initialization=True,
        required_allocations=required,
        **gate,
        validation=validation,
        science_constraint_magnitude=dict(min=float(values.min()), max=float(values.max())),
        science_jacobian_row_norm=dict(
            min=float(np.linalg.norm(jac, axis=1).min()), max=float(np.linalg.norm(jac, axis=1).max())
        ),
        constraint_scaling_changed=False,
        runtime_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        dense_solver_invocations=0,
        workspace_allocation_attempted=False,
    )
    if gate["resource_feasible"]:
        tr, log, _, path = solve_saved(b, "resource_smoke_500")
        record.update(
            optimizer=log,
            state_file_sha256=file_hash(path),
            dense_solver_invocations=1,
            workspace_allocation_attempted=True,
            finite=bool(torch.isfinite(tr["prediction"]).all()),
        )
    else:
        record["reason"] = (
            "Exact mandatory dense arrays exceed preregistered RAM budget before other solver/framework allocations"
        )
    write(OUT / "preflight" / "resource_smoke_500.json", record)
    return record


def preflight():
    frozen()
    assert not (OUT / "preflight_attempt.json").exists()
    write(OUT / "preflight_attempt.json", dict(lengths=[12, 32, 500], no_retries=True))
    records = []
    for n in [12, 32]:
        b = synthetic(n)
        validation = historical.validate(b)
        tr, log, z, path = solve_saved(b, f"preflight_{n}")
        row = dict(
            length=n,
            validation=validation,
            optimizer=log,
            state_file_sha256=file_hash(path),
            coordinate_sha256=hashlib.sha256(tr["prediction"].detach().numpy().tobytes()).hexdigest(),
            variables_sha256=hashlib.sha256(z.tobytes()).hexdigest(),
        )
        write(OUT / "preflight" / f"length_{n}.json", row)
        records.append(row)
        print("preflight", n, log["converged"], log["iterations"], log["scipy_status"], flush=True)

    def timeout(*_):
        raise TimeoutError("Frozen longest-length resource wall budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CONTRACT["config"]["resource_policy"]["longest_smoke_wall_cap_seconds"])
    try:
        smoke = resource_smoke()
    finally:
        signal.alarm(0)
    assert_file_pins(CONTRACT["protected_sha256"])
    write(
        OUT / "preflight_complete.json",
        dict(
            records=records,
            resource_smoke=smoke,
            implementation_valid=True,
            panel_permitted=smoke["resource_feasible"],
            scientific_panel_launched=False,
        ),
    )


def example(i, mode):
    b = data(i)
    path = OUT / "B" / f"example_{i:02d}.json"
    state = OUT / "untracked_states" / f"example_{i:02d}_certified.npz"
    if mode == "run":
        write(OUT / "attempts" / f"example_{i:02d}.json", dict(index=i, solver_invocations=1, initialization="zero"))
        tr, log, z, state = solve_saved(b, f"example_{i:02d}")
        row = historical.build(i, tr, log, z)
        row.update(state_file_sha256=file_hash(state), solver_invocations=1)
        write(path, row)
        return dict(index=i, saved=True, converged=log["converged"])
    row = json.loads(path.read_text())
    assert file_hash(state) == row["state_file_sha256"]
    with np.load(state) as saved:
        z = saved["z"].copy()
        tr = sqp.direct.trajectory(b["pg"], b["mask"], torch.from_numpy(z).reshape(8, *b["pg"].shape))
        delta = torch.stack([s["delta"].detach() for s in tr["steps"]])
        assert np.array_equal(torch.stack(tr["states"]).detach().numpy(), saved["states"])
        assert np.array_equal(delta.numpy(), saved["delta"])
        mu, _ = sqp.reconstruct(b, delta, CONTRACT["physical_config"])
        assert np.array_equal(mu, saved["multipliers"])
        cert = sqp.direct.certificate(b, delta, mu, CONTRACT["physical_config"], CONTRACT["physical_limits"])
    assert json.loads(json.dumps(cert)) == row["optimizer"]["physical_certificate"]
    for k, v in historical.build(i, tr, row["optimizer"], z).items():
        assert json.loads(json.dumps(v)) == row[k]
    write(OUT / "reproduction" / f"example_{i:02d}.json", dict(index=i, exact=True, optimizer_invocations=0))
    return dict(index=i, reproduced=True)


def execute(mode):
    frozen()
    pre = json.loads((OUT / "preflight_complete.json").read_text())
    assert pre["panel_permitted"], "SQP-E resource gate blocks the scientific panel"
    if mode == "run":
        assert not list((OUT / "attempts").glob("*.json")), "Once-only panel cannot be restarted"
    records = []
    for i in range(60):
        records.append(example(i, mode))
        print(mode, records[-1], flush=True)
    assert_file_pins(CONTRACT["protected_sha256"])
    write(OUT / f"{mode}_complete.json", dict(results=records, panel_examples=60, outcome_based_reruns=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "preflight", "run", "reproduce"])
    mode = parser.parse_args().mode
    register() if mode == "register" else preflight() if mode == "preflight" else execute(mode)
