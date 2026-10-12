"""Check report control flow with saved scalar tables and stubbed rendering.

All test writes use a temporary directory explicitly placed on D:. No framework,
checkpoint, coordinate preload, bootstrap, image creation or production writes.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.nice(19)
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

work = Path(__file__).resolve().parent.parent
source = work / "docs/e015_latest_visuals.py"
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "report")
actual = Path(sys.argv[1])
calls = dict(cards=0, stories=0, comparisons=0)

with tempfile.TemporaryDirectory(prefix="e015_report_check_", dir=work / "docs") as directory:
    target = Path(directory)
    (target / "logs").mkdir()
    shutil.copyfile(actual / "logs/final_artifact_validation.json", target / "logs/final_artifact_validation.json")
    for category in ("success_gallery", "median_gallery", "failure_gallery", "inference_story"):
        (target / category).mkdir()
    for category in ("success_gallery", "median_gallery", "failure_gallery"):
        for folder in (actual / category).iterdir():
            if folder.is_dir():
                (target / category / folder.name).mkdir()
    for story in (actual / "inference_story").glob("*_storyboard.png"):
        (target / "inference_story" / story.name).write_text("test placeholder")

    def table(name):
        rows = pq.read_table(actual / name, use_threads=False).to_pylist()
        if name == "generation_manifest.parquet":
            for row in rows:
                row["path"] = str(target / "samples" / row["weights"] / row["sample_key"])
        return rows

    def write(name, value):
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def card(c, meta, *_args):
        calls["cards"] += 1
        path = Path(meta["path"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "quality_card.png").write_text("test placeholder")

    def storyboard(c, meta):
        calls["stories"] += 1

    def comparison(c, meta, trajectory=False):
        calls["comparisons"] += 1

    selection = json.loads((actual / "checkpoint_selection.json").read_text())
    c = SimpleNamespace(
        OUT=target,
        WORK=work,
        GRID=(5, 50, 250, 400, 490),
        LATEST="latest_u10382",
        SHA=selection["source"]["sha256"],
        selection=selection,
        VERSION="test",
        table=table,
        pulse=lambda *_args: None,
        sha=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        b=SimpleNamespace(
            LENGTHS=(48, 96, 192, 320, 448), read=lambda name: json.loads((actual / name).read_text()), write=write
        ),
    )
    context = dict(
        np=np, Path=Path, shutil=shutil, subprocess=subprocess, card=card, storyboard=storyboard, comparison=comparison
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), context)
    context["report"](c)
    assert calls == dict(cards=25, stories=15, comparisons=35), calls
    text = (target / "REPORT.md").read_text()
    assert "3.6108" in text and "3.8099" in text and "Chiral-only disabled" in text
    assert json.loads((target / "MANIFEST.json").read_text())["serialized"]
    print(json.dumps(dict(status="PASSED", render_calls=calls, production_writes=False)))
