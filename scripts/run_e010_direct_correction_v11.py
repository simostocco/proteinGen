"""Once-only V11 panel, frozen registration and independent fixed-state reproduction."""

import argparse
import copy
import hashlib
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import yaml

from protein_distance_diffusion.training import e010_direct_correction_v11 as direct
from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_strict_scientific_v10 as old
from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

OUT = old.OUT.parent / "direct_correction_solver_v11"
CONFIG = ROOT / "configs/e010_phase4d_direct_correction_v11.yaml"
CACHE = CONTRACT = None


def setup():
    cfg, physical, limits = old.setup()
    own = yaml.safe_load(CONFIG.read_text())
    cfg = copy.deepcopy(cfg)
    assert own["solver_override"] == {"maxiter": 1000}
    cfg["solver"].update(own["solver_override"])
    assert own["K"] == 8 and own["s_max_angstrom"] == 0.04
    return cfg, physical, limits, own


def data(i):
    old.CACHE = CACHE
    return old.data(i)


def validate(b):
    o = direct.Oracle(b)
    z = np.zeros(np.prod(o.shape))
    assert (o.cfun(z) <= 0).all()
    assert (o.ball_values(z) == 1).all()
    assert all(torch.equal(p, b["pg"]) for p in o.point(z).tr["states"])
    assert torch.equal(o.quartets.q0, old.v10.v9.QuartetConstraints(b).q0)
    # Interior equivalence for a fixed physical displacement, not optimization.
    zi = np.sin(np.arange(len(z)) + 1) * 0.05
    p = o.point(zi)
    delta = 0.04 * torch.from_numpy(zi).reshape(o.shape)
    v = delta / (1 - delta.square().sum(-1, keepdim=True) / 0.04**2).sqrt()
    historical = old.v10.v9.Oracle(b).point((v / 0.04).numpy().reshape(-1))
    for a, c in zip(p.tr["states"], historical.tr["states"], strict=True):
        torch.testing.assert_close(a, c, atol=1e-12, rtol=1e-12)
    for key in p.values:
        torch.testing.assert_close(p.values[key], historical.values[key], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(p.constraints, historical.constraints, atol=1e-12, rtol=1e-12)
    a = o.quartets.telemetry(p.tr["prediction"], 1e-5)
    c = o.quartets.telemetry(historical.tr["prediction"], 1e-5)
    for key in ["new_inversions", "repaired_inversions", "assessability_lost"]:
        assert a[key] == c[key]
    j = o.cjac(zi)
    gradient = o.fun(zi)[1]
    trials = []
    for phase in [0.0, 0.7, 1.3]:
        d = np.sin(np.arange(len(z)) + 1 + phase)
        d /= np.linalg.norm(d)
        expected = np.r_[gradient @ d, j @ d, o.ball_jacobian(zi) @ d]
        errors = []
        for eps_angstrom in [1e-6, 1e-5, 1e-4]:
            eps = eps_angstrom / 0.04
            plus, minus = zi + eps * d, zi - eps * d
            fd = np.r_[
                (o.fun(plus)[0] - o.fun(minus)[0]) / (2 * eps),
                (o.cfun(plus) - o.cfun(minus)) / (2 * eps),
                (o.ball_values(plus) - o.ball_values(minus)) / (2 * eps),
            ]
            errors.append(float((np.abs(fd - expected) / (1e-8 + 1e-5 * np.abs(expected))).max()))
        assert min(errors) <= 1, "Implementation derivative validation failed before optimization"
        trials.append(dict(phase=phase, maximum_scaled_errors=errors, passed=True))
    assert np.isfinite(o.hess(zi) @ d).all()
    assert np.isfinite(o.chess(zi, np.full(len(o.cfun(zi)), 0.1)) @ d).all()
    return dict(
        baseline_exact_feasible=True,
        zero_exact=True,
        interior_parity=True,
        quartets=o.quartets.counts,
        directions=trials,
        hvps_finite=True,
    )


def register():
    global CACHE
    cfg, physical, limits, own = setup()
    CACHE, manifest = load_cache()
    historical = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    added = [
        str(CONFIG.relative_to(ROOT)),
        "src/protein_distance_diffusion/training/e010_direct_correction_v11.py",
        "scripts/run_e010_direct_correction_v11.py",
        "scripts/report_e010_direct_correction_v11.py",
        "tests/test_e010_direct_correction_v11.py",
        "docs/e010_phase4d_direct_correction_v11.md",
    ]
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in set(historical + added)}
    for module, commit in [
        (
            "src/protein_distance_diffusion/training/e010_no_new_inversion_v9.py",
            "244ac4d1ba4e8375cca30e4172fd44e51c9d0a7f",
        ),
        (
            "src/protein_distance_diffusion/training/e010_conditioning_v9b.py",
            "c0156099973f69461f5dc0d892e94c10441da172",
        ),
        (
            "src/protein_distance_diffusion/training/e010_strict_scientific_v10.py",
            "c1a76f3d38af1bf02f47f8680529128942c3b28b",
        ),
    ]:
        assert (
            hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{module}"], cwd=ROOT)).hexdigest()
            == pins[str(ROOT / module)]
        )
    records = []
    for i in range(60):
        records.append(dict(index=i, record=CACHE["records"][i], **validate(data(i))))
        print("validated", i, flush=True)
    assert sum(r["quartets"]["assessable"] for r in records) == 13029
    assert_file_pins(pins)
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        OUT / "execution_contract.json",
        dict(
            config=own,
            solver_config=cfg,
            physical_config=physical,
            physical_limits=limits,
            protected_sha256=pins,
            protected_input_sha256=manifest["protected_input_sha256"],
            cache_sha256=manifest["cache_sha256"],
            order=manifest["order"],
            workers=4,
            validation=records,
            initialization="exact_zero",
            solver_invocations_per_panel_example=1,
            historical_cap=2000,
            explicitly_requested_v11_cap=1000,
            closed_ball_note=(
                "Closed intended 0.04 Angstrom radius; no enlargement. Interior physical certificates unchanged."
            ),
        ),
    )


def frozen():
    global CONTRACT, CACHE
    setup()
    CONTRACT = json.loads((OUT / "execution_contract.json").read_text())
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    CACHE, manifest = load_cache()
    assert manifest["cache_sha256"] == CONTRACT["cache_sha256"]
    old.CACHE = CACHE
    old.CONTRACT = CONTRACT


def build(i, tr, log, z):
    old.CACHE, old.CONTRACT = CACHE, CONTRACT
    return old.build(i, tr, log, z)


def save_state(path, tr, log, z):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez(
        path,
        z=z,
        multipliers=np.asarray(log["multipliers"]),
        delta=torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy(),
        states=torch.stack([p.detach() for p in tr["states"]]).numpy(),
    )


def preflight():
    frozen()
    records = []
    for n in [12, 32]:
        tr, log, z = direct.solve(
            synthetic(n), CONTRACT["solver_config"], CONTRACT["physical_config"], CONTRACT["physical_limits"]
        )
        save_state(OUT / "untracked_states" / f"preflight_{n}.npz", tr, log, z)
        if n == 12:
            tr2, log2, z2 = direct.solve(
                synthetic(n), CONTRACT["solver_config"], CONTRACT["physical_config"], CONTRACT["physical_limits"]
            )
            a, b = copy.deepcopy(log), copy.deepcopy(log2)
            a.pop("runtime_seconds")
            b.pop("runtime_seconds")
            assert json.loads(json.dumps(a)) == json.loads(json.dumps(b))
            assert np.array_equal(z, z2) and torch.equal(tr["prediction"], tr2["prediction"])
        record = dict(
            length=n,
            optimizer=log,
            state_coordinate_sha256=hashlib.sha256(tr["prediction"].detach().numpy().tobytes()).hexdigest(),
            deterministic_repeat=n == 12,
            implementation_valid=True,
        )
        records.append(record)
        write(OUT / "preflight" / f"length_{n}.json", record)
        print("preflight", n, log["converged"], log["iterations"], log["physical_certificate"]["gates"], flush=True)
    assert_file_pins(CONTRACT["protected_sha256"])
    write(
        OUT / "preflight_complete.json",
        dict(
            records=records,
            implementation_valid=True,
            scientific_outcome_used_for_tuning=False,
            proceed_per_user_even_if_physically_nonconverged=True,
        ),
    )


def example(i, mode):
    b = data(i)
    cfg, physical, limits = (CONTRACT[k] for k in ["solver_config", "physical_config", "physical_limits"])
    record_path = OUT / "B" / f"example_{i:02d}.json"
    path = OUT / "untracked_states" / f"example_{i:02d}.npz"
    if mode == "run":
        write(OUT / "attempts" / f"example_{i:02d}.json", dict(index=i, solver_invocations=1, initialization="zero"))
        try:
            tr, log, z = direct.solve(b, cfg, physical, limits)
            save_state(path, tr, log, z)  # Serialize before any optional telemetry.
            record = build(i, tr, log, z)
            record.update(
                state_file_sha256=file_hash(path),
                solver_invocations=1,
                parameterization="direct_linear_cartesian_closed_balls",
            )
            write(record_path, record)
        except Exception as exc:
            write(
                OUT / "errors" / f"example_{i:02d}.json", dict(index=i, exception=type(exc).__name__, message=str(exc))
            )
            return dict(index=i, error=True)
        return dict(index=i, saved=True, converged=log["converged"], iterations=log["iterations"])
    expected = json.loads(record_path.read_text())
    assert file_hash(path) == expected["state_file_sha256"]
    with np.load(path) as saved:
        z = saved["z"].copy()
        tr = direct.trajectory(b["pg"], b["mask"], torch.from_numpy(z).reshape(8, *b["pg"].shape))
        assert np.array_equal(torch.stack(tr["states"]).detach().numpy(), saved["states"])
        assert np.array_equal(torch.stack([s["delta"] for s in tr["steps"]]).numpy(), saved["delta"])
        cert = direct.certificate(b, torch.from_numpy(saved["delta"].copy()), saved["multipliers"], physical, limits)
    assert json.loads(json.dumps(cert)) == expected["optimizer"]["physical_certificate"]
    rebuilt = build(i, tr, expected["optimizer"], z)
    for key in rebuilt:
        assert json.loads(json.dumps(rebuilt[key])) == expected[key], f"Independent reproduction failed {i}/{key}"
    write(
        OUT / "reproduction" / f"example_{i:02d}.json",
        dict(
            index=i, metrics_exact=True, physical_certificate_exact=True, hashes_verified=True, optimizer_invocations=0
        ),
    )
    return dict(index=i, reproduced=True)


def execute(mode):
    frozen()
    assert json.loads((OUT / "preflight_complete.json").read_text())["implementation_valid"]
    if mode == "run":
        assert not list((OUT / "attempts").glob("*.json")), "Once-only panel cannot be restarted"
    else:
        complete = json.loads((OUT / "execution_complete.json").read_text())
        assert all(r.get("saved") for r in complete["results"])
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i, mode) for i in range(60)]):
            record = future.result()
            results.append(record)
            print(mode, record, flush=True)
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    write(
        OUT / ("execution_complete.json" if mode == "run" else "reproduction_complete.json"),
        dict(
            examples=60,
            results=sorted(results, key=lambda r: r["index"]),
            protected_hashes_verified=True,
            cuda_used=False,
            neural_training_launched=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["register", "preflight", "run", "reproduce"])
    mode = parser.parse_args().mode
    register() if mode == "register" else preflight() if mode == "preflight" else execute(mode)
