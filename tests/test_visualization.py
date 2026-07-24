"""Dataset visualization tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_visualization_smoke(monkeypatch, synthetic_manifest: Path, tmp_path: Path) -> None:
    """Visualization reads processed samples and writes at least one PNG."""
    script = Path(__file__).parents[1] / "scripts" / "visualize_samples.py"
    spec = importlib.util.spec_from_file_location("visualize_samples", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    out = tmp_path / "figures"
    monkeypatch.setattr(
        "sys.argv",
        [
            "visualize_samples.py",
            "--manifest",
            str(synthetic_manifest),
            "--output-dir",
            str(out),
            "--num-samples",
            "1",
            "--seed",
            "1",
            "--contact-threshold",
            "8.0",
        ],
    )
    module.main()
    assert list(out.glob("*.png"))
    assert (out / "index.html").exists()
