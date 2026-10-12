"""Apply the pinned E015 external toolchain and gates to frozen E007 candidates."""

import os, pathlib, types, sys, json, shutil

C = pathlib.Path("/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1")
OUT = pathlib.Path(os.environ.get("COMPARISON_EXTERNAL_OUT", str(C / "e007_inverse_folding")))
SOURCE_MODEL = os.environ.get("COMPARISON_SOURCE_MODEL", "E007_epoch1")
EXT = pathlib.Path(
    "/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1"
)
import e007_epoch1_mission as mission

mission.helper().host(True)
import fcntl

leases = []
for lockpath in [
    Path
    for Path in [
        pathlib.Path("/mnt/d/Simone/proteinGen/.e014_worktree/e007_matched_fullscale_v1/OWNER.lock"),
        EXT.parent / "GPU_WORKER.lock",
    ]
]:
    lease = lockpath.open("a+")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    leases.append(lease)

source = pathlib.Path(__file__).with_name("e015_external_designability.py").read_text()
# Existing hash-matched real controls are reused; no new control folds/designs.
source = source.replace(
    "for cat in ('real_heldout','latest_best','historical_same_seed','latest_median','latest_failure'):",
    "for cat in ('latest_best','historical_same_seed','latest_median','latest_failure'):",
)
m = types.ModuleType("matched_external")
exec(compile(source, "existing_pinned_external_protocol", "exec"), m.__dict__)
m.OUT = OUT
for k in [
    "TMPDIR",
    "TMP",
    "TEMP",
    "XDG_CACHE_HOME",
    "MPLCONFIGDIR",
    "TORCH_HOME",
    "HF_HOME",
    "CUDA_CACHE_PATH",
    "TRITON_CACHE_DIR",
    "PIP_CACHE_DIR",
    "JAX_COMPILATION_CACHE_DIR",
    "COLABFOLD_CACHE_DIR",
]:
    p = OUT / "cache" / k.lower()
    p.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(p)
shared = EXT / "cache/msa"
shared.mkdir(parents=True, exist_ok=True)
if not (OUT / "cache/msa").exists():
    (OUT / "cache/msa").symlink_to(shared, target_is_directory=True)
for f in ["logs/colabfold_help.log", "foldseek/remote_database_inventory.json"]:
    if (EXT / f).exists() and not (OUT / f).exists():
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EXT / f, OUT / f)


def mpnn():
    panel = m.read(OUT / "backbone_panel.json")
    real = [b for b in panel["backbones"] if b["source_category"] == "real_heldout"]
    # Functions read the panel; preserve original immutable panel while using an in-memory view.
    original = m.read
    m.read = lambda p: (
        dict(panel, backbones=[b for b in panel["backbones"] if b not in real])
        if pathlib.Path(p) == OUT / "backbone_panel.json"
        else original(p)
    )
    m.mpnn()


def sequence():
    m.sequences()
    import pyarrow as pa, pyarrow.parquet as pq

    for name in ["proteinmpnn_all_sequences.parquet", "proteinmpnn_selected_sequences.parquet"]:
        generated = pq.read_table(OUT / name, use_threads=False).to_pylist()
        controls = [
            r
            for r in pq.read_table(EXT / name, use_threads=False).to_pylist()
            if r["source_category"] == "real_heldout"
        ]
        pq.write_table(pa.Table.from_pylist(generated + controls), OUT / name)
    if (EXT / "controls_summary.json").exists():
        shutil.copyfile(EXT / "controls_summary.json", OUT / "REUSED_CONTROL_SOURCE.json")


def cards():
    import ast, numpy as np, pyarrow.parquet as pq

    source = pathlib.Path(__file__).with_name("e015_external_reporting.py").read_text()
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name in ["table", "ca", "cards", "neighbor_cards"]
    ]
    scope = dict(
        OUT=OUT, Path=pathlib.Path, np=np, pq=pq, read=m.read, resource=m.resource, progress=m.progress, run=m.run
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "same_external_card_protocol", "exec"), scope)
    for name in [
        "positive_control_gallery",
        "known_fold_like_gallery",
        "potentially_novel_gallery",
        "likely_artifact_gallery",
        "candidates",
    ]:
        (OUT / name).mkdir(exist_ok=True)
    scope["run"] = m.run
    scope["cards"]()
    scope["neighbor_cards"]()


def summary():
    import pyarrow.parquet as pq

    results = pq.read_table(OUT / "designability_summary.parquet", use_threads=False).to_pylist()
    control = m.read(OUT / "controls_summary.json")
    latest = [r for r in results if r["source_category"] == "latest_best"]
    tested = [r for r in results if r["source_category"].startswith("latest")]
    verdict = "INCONCLUSIVE"
    if control["status"] == "VALIDATED" and all(r["classification"] != "INCONCLUSIVE" for r in latest):
        strong = sum(r["classification"] == "STRONGLY_DESIGNABLE" for r in latest)
        promising = sum(r["classification"] in ["STRONGLY_DESIGNABLE", "PROMISINGLY_DESIGNABLE"] for r in latest)
        verdict = (
            "STRONG_EVIDENCE"
            if strong >= 3
            else "PROMISING_BEST_TAIL"
            if promising
            else "WEAK_EVIDENCE"
            if any(r["strong_count"] for r in latest)
            else "NO_EVIDENCE"
        )
    result = dict(
        model=SOURCE_MODEL,
        controls=control,
        rows=results,
        verdict=verdict,
        tested_E007=len(tested),
        selected_best_denominator=5,
        full_generation_panel=20,
        full_panel_external_yield="Not directly measured: selected best/median/failure backbones only; untested samples excluded from designability claims",
        checkpoint=m.read(OUT / "backbone_panel.json").get(
            "model_checkpoint", mission.read(C / "E007_EPOCH1_CHECKPOINT.json")
        ),
        computational_only=True,
    )
    mission.write(
        C
        / (
            "E007_EXTERNAL_DESIGNABILITY.json"
            if SOURCE_MODEL == "E007_epoch1"
            else "E015_MATCHED_EXTERNAL_DESIGNABILITY.json"
        ),
        result,
    )
    mission.write(OUT / "verdict.json", dict(designability=verdict, positive_control_status=control["status"]))


if __name__ == "__main__":
    {
        "mpnn": mpnn,
        "sequences": sequence,
        "screen": lambda: m.folding(screen=True),
        "align": m.align,
        "confirm": m.confirm,
        "confirm-align": m.confirm_align,
        "novelty": m.novelty,
        "summary": summary,
        "cards": cards,
    }[sys.argv[1]]()
