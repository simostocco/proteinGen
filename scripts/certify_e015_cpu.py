"""Source-bound CPU/lint certificate with only D-resident framework output."""

import subprocess

import e015_bootstrap
import pytest

from protein_distance_diffusion.training.e015_runtime import Runtime


def main():
    paths = sorted(
        str(p.relative_to(e015_bootstrap.WORKTREE)) for p in (e015_bootstrap.WORKTREE / "tests").glob("test_e015_*.py")
    )
    code = sorted(
        str(p.relative_to(e015_bootstrap.WORKTREE))
        for p in (e015_bootstrap.WORKTREE / "src/protein_distance_diffusion").rglob("e015_*.py")
    )
    scripts = sorted(
        str(p.relative_to(e015_bootstrap.WORKTREE)) for p in (e015_bootstrap.WORKTREE / "scripts").glob("*e015*.py")
    )
    subprocess.run(
        ["/home/simostocco/miniforge3/envs/proteingen/bin/ruff", "check", *code, *scripts, *paths], check=True
    )
    result = pytest.main(["-q", "--capture=sys", "-p", "no:cacheprovider", *paths])
    if result:
        raise RuntimeError("E015 focused tests failed")
    import json

    runtime = Runtime(e015_bootstrap.WORKTREE / "configs/e015_integrated_multiscale_parity_selfconditioned_v1.yaml")
    calibration = json.loads(runtime.path("calibration/frozen.json").read_text())
    contract = runtime.contract(calibration)
    runtime.write(
        "preflight/cpu_tests.json",
        dict(
            passed=True,
            lint_passed=True,
            source_hashes=contract["source_hashes"],
            config_hash=contract["config_hash"],
            all_generated_paths_D=True,
            disk_reserve_gate=False,
        ),
    )


if __name__ == "__main__":
    main()
