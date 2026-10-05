"""Reversible E011-local mirror for unchanged tests requiring historical artifacts.

Run mirrored tests in the filesystem sandbox so historical links are read-only.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIST = Path("/mnt/d/Simone/proteinGen")
MIRROR = ROOT / "outputs/e011_sequence_context_only/pilot_v1/test_mirror"


def materialize_contained_sources():
    """Preserve path-containment checks using five identical local NPZ fixtures."""
    assert (MIRROR / "data").is_symlink()
    (MIRROR / "data").unlink()
    for relative, excluded in [("data", "full"), ("data/full", "processed"), ("data/full/processed", "samples")]:
        local = MIRROR / relative
        local.mkdir(exist_ok=True)
        for child in (HIST / relative).iterdir():
            if child.name != excluded:
                (local / child.name).symlink_to(child, target_is_directory=child.is_dir())
    sample_root = MIRROR / "data/full/processed/samples"
    sample_root.mkdir()
    plan = HIST / "reports/experiments/E010_global_equivariant_expressivity/phase4a_multicorruption_v2"
    rows = json.loads((plan / "training_seed_manifest.json").read_text())["identities"]
    exclusions = json.loads((plan / "excluded_archives.json").read_text())["archives"]
    copied = {next(r["source_path"] for r in rows if r["stratum"] == s) for s in {r["stratum"] for r in rows}}
    for relative in sorted({r["source_path"] for r in rows + exclusions}):
        source, target = HIST / relative, MIRROR / relative
        assert source.is_file() and target.parent == sample_root
        if relative in copied:
            shutil.copy2(source, target)
            assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
        else:
            target.symlink_to(source)


if __name__ == "__main__":
    assert not MIRROR.exists()
    MIRROR.mkdir()
    for name in ["src", "scripts", "tests", "configs", "environment"]:
        shutil.copytree(ROOT / name, MIRROR / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["pyproject.toml", "README.md", "LICENSE", ".git"]:
        shutil.copy2(ROOT / name, MIRROR / name)
    for name in ["reports", "outputs"]:
        target = MIRROR / name
        target.mkdir()
        if name == "reports":
            target = target / "experiments"
            target.mkdir()
            source = HIST / name / "experiments"
        else:
            source = HIST / name
        for item in source.iterdir():
            (target / item.name).symlink_to(item, target_is_directory=item.is_dir())
    shutil.copytree(
        ROOT / "reports/experiments/E011_sequence_context_only",
        MIRROR / "reports/experiments/E011_sequence_context_only",
    )
    for name in ["data", "external_models", "logs"]:
        (MIRROR / name).symlink_to(HIST / name, target_is_directory=True)
    materialize_contained_sources()
    print(MIRROR)
