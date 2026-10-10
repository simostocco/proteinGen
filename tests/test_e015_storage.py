from pathlib import Path

import pytest

from protein_distance_diffusion.training.e015_storage import generated_path

ROOT = Path("/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1")


@pytest.mark.parametrize("path", ["/tmp/x", "/mnt/c/x", "/home/x", str(ROOT / ".." / "escaped")])
def test_rejects_outside_declared_root(path):
    with pytest.raises(ValueError):
        generated_path(ROOT, path)


def test_accepts_nested_output():
    assert generated_path(ROOT, ROOT / "logs" / "x.json") == ROOT / "logs" / "x.json"


def test_entrypoints_bootstrap_before_scientific_imports():
    import ast

    worktree = Path(__file__).resolve().parents[1]
    for name in (
        "calibrate_e015.py",
        "profile_e015_capacity.py",
        "autotune_e015_batches.py",
        "preflight_e015.py",
        "run_e015_integrated.py",
        "supervise_e015_training.py",
        "launch_e015_supervisor.py",
    ):
        tree = ast.parse((worktree / "scripts" / name).read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [node.names[0].name if isinstance(node, ast.Import) else node.module for node in imports]
        bootstrap = names.index("e015_bootstrap")
        for i, name in enumerate(names):
            if name in ("torch", "numpy", "pyarrow", "matplotlib") or name.startswith("protein_distance_diffusion"):
                assert i > bootstrap


def test_disk_reserve_is_telemetry_only(monkeypatch):
    from collections import namedtuple

    from protein_distance_diffusion.training.e015_storage import disk_budget

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("shutil.disk_usage", lambda _: Usage(100, 99, 1))
    report = disk_budget(ROOT, 100000000000, 200000000000)
    assert report["passed"]
    assert not report["reserve_enforced"]
    assert report["would_cross_historical_reserve"]
