"""Exact V11 execution with only maxiter=2000; once-only panel and fixed-state reproduction."""

import argparse
import copy
import json
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from protein_distance_diffusion.training.e010_phase4d_diagnostic import assert_file_pins, file_hash
from scripts import run_e010_direct_correction_v11 as historical
from scripts.run_e010_phase4d_recurrent_capacity_v3 import ROOT, load_cache, write

HISTORICAL = historical.OUT
OUT = HISTORICAL.parent / "matched_direct_space_v13"
CONFIG = ROOT / "configs/e010_phase4d_matched_direct_space_v13.yaml"
CONTRACT = None


def setup():
    assert (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
        == "e010-phase4d-hybrid-local-global"
    )
    previous = json.loads((HISTORICAL / "execution_contract.json").read_text())
    cfg, physical, limits, _ = historical.setup()
    assert cfg == previous["solver_config"]
    assert physical == previous["physical_config"] and limits == previous["physical_limits"]
    own = yaml.safe_load(CONFIG.read_text())
    assert own["solver_override"] == {"maxiter": 2000}
    assert own["K"] == 8 and own["s_max_angstrom"] == 0.04
    cfg = copy.deepcopy(cfg)
    cfg["solver"]["maxiter"] = 2000
    restored = copy.deepcopy(cfg)
    restored["solver"]["maxiter"] = 1000
    assert restored == previous["solver_config"]
    return cfg, physical, limits, own


def register():
    cfg, physical, limits, own = setup()
    previous = json.loads((HISTORICAL / "execution_contract.json").read_text())
    assert_file_pins(previous["protected_sha256"])
    v12 = json.loads((HISTORICAL.parent / "v11_fixed_state_numerical_audit_v12" / "preregistration.json").read_text())
    assert_file_pins(v12["protected_sha256"])
    cache, manifest = load_cache()
    assert manifest["cache_sha256"] == previous["cache_sha256"]
    historical.CACHE = cache
    names = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    names += [
        str(CONFIG.relative_to(ROOT)),
        "docs/e010_phase4d_matched_direct_space_v13.md",
        "scripts/run_e010_matched_direct_space_v13.py",
        "scripts/report_e010_matched_direct_space_v13.py",
        "scripts/verify_e010_matched_direct_space_v13.py",
        "tests/test_e010_matched_direct_space_v13.py",
    ]
    pins = {str(ROOT / p): file_hash(ROOT / p) for p in set(names)}
    pins.update(v12["installed_solver_sources"])
    for i in range(60):
        old = json.loads((HISTORICAL / "B" / f"example_{i:02d}.json").read_text())
        path = HISTORICAL / "untracked_states" / f"example_{i:02d}.npz"
        assert file_hash(path) == old["state_file_sha256"]
        pins[str(path)] = file_hash(path)
    checks = []
    for i in range(60):
        checks.append(dict(index=i, record=cache["records"][i], **historical.validate(historical.data(i))))
        print("validated", i, flush=True)
    assert sum(c["quartets"]["assessable"] for c in checks) == 13029
    assert json.loads(json.dumps(checks)) == previous["validation"]
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
            validation=checks,
            initialization="exact_zero",
            solver_invocations_per_panel_example=1,
            workers=4,
            only_change=dict(field="solver.maxiter", before=1000, after=2000),
            solver_implementation_unchanged=True,
            original_solver_sha256=file_hash(
                ROOT / "src/protein_distance_diffusion/training/e010_direct_correction_v11.py"
            ),
        ),
    )


def frozen():
    global CONTRACT
    cfg, physical, limits, own = setup()
    CONTRACT = json.loads((OUT / "execution_contract.json").read_text())
    assert cfg == CONTRACT["solver_config"] and physical == CONTRACT["physical_config"]
    assert limits == CONTRACT["physical_limits"] and own == CONTRACT["config"]
    assert_file_pins(CONTRACT["protected_sha256"])
    assert_file_pins(CONTRACT["protected_input_sha256"])
    cache, manifest = load_cache()
    assert manifest["cache_sha256"] == CONTRACT["cache_sha256"]
    historical.CACHE, historical.CONTRACT, historical.OUT = cache, CONTRACT, OUT


def prefix_match(i):
    row = json.loads((OUT / "B" / f"example_{i:02d}.json").read_text())
    prior = json.loads((HISTORICAL / "B" / f"example_{i:02d}.json").read_text())
    assert row["record"] == prior["record"] and row["baseline"] == prior["baseline"]
    stop = prior["optimizer"]["iterations"]
    for name in ["history", "physical_screens"]:
        current = [v for v in row["optimizer"][name] if v["iteration"] <= stop]
        assert current == prior["optimizer"][name], f"Historical prefix mismatch {i}/{name}"
    if prior["optimizer"]["converged"]:
        for name in ["oracle", "quartet_states", "states", "variables_sha256"]:
            assert row[name] == prior[name], f"Early-converged mismatch {i}/{name}"
        left, right = copy.deepcopy(row["optimizer"]), copy.deepcopy(prior["optimizer"])
        left.pop("runtime_seconds")
        right.pop("runtime_seconds")
        assert left == right, f"Early-converged solver state mismatch {i}"
    return dict(
        index=i,
        history_prefix_exact=True,
        physical_screens_prefix_exact=True,
        historical_stopping_iteration=stop,
        early_converged_exact=prior["optimizer"]["converged"],
    )


def example(i, mode):
    result = historical.example(i, mode)
    if result.get("error"):
        return result
    match = prefix_match(i)
    if mode == "run":
        write(OUT / "prefix_validation" / f"example_{i:02d}.json", match)
    return dict(**result, prefix_exact=True)


def execute(mode):
    frozen()
    if mode == "run":
        assert not list((OUT / "attempts").glob("*.json")), "Once-only V13 panel cannot be restarted"
    else:
        complete = json.loads((OUT / "execution_complete.json").read_text())
        assert len(complete["results"]) == 60 and all(r.get("saved") for r in complete["results"])
    results = []
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        for future in as_completed([pool.submit(example, i, mode) for i in range(60)]):
            result = future.result()
            results.append(result)
            print(mode, result, flush=True)
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
    parser.add_argument("mode", choices=["register", "run", "reproduce"])
    mode = parser.parse_args().mode
    register() if mode == "register" else execute(mode)
