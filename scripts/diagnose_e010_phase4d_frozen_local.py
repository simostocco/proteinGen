#!/usr/bin/env python3
"""CPU-only, no-training Phase 4D finite-displacement calibration."""

import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import SOURCE_SHA256, load_frozen_hybrid
from protein_distance_diffusion.training.e010_phase4d_diagnostic import (
    assert_file_pins,
    file_hash,
    run_diagnostic,
    state_hash,
)

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "reports/experiments/E010_global_equivariant_expressivity/phase4d_hybrid_local_global_v1"
OUT = PREP / "calibration_diagnostic_v1"
CACHE = Path(
    "/home/simostocco/proteinGen/reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/phase4b_real_denoiser_v1.final"
)


def load_panel(model, pinned):
    if file_hash(CACHE / "manifest.json") != pinned["cache_manifest_sha256"]:
        raise ValueError("cache manifest pin mismatch")
    if len(pinned["examples"]) != 60 or len(pinned["identities"]) != 20:
        raise ValueError("requires unchanged 20/60 panel")
    expected = {(r["sample_id"], c) for r in pinned["identities"] for c in (50, 250, 450)}
    observed = {(r["sample_id"], r["timestep"]) for r in pinned["examples"]}
    if expected != observed:
        raise ValueError("panel identity-condition coverage mismatch")
    items = []
    archive_pins = {}
    for index, r in enumerate(pinned["examples"]):
        if r["split"] != "train":
            raise ValueError("nontraining example forbidden")
        path = CACHE / r["shard"]
        if path not in archive_pins:
            if file_hash(path) != r["archive_sha256"]:
                raise ValueError("archive pin mismatch")
            archive_pins[path] = r["archive_sha256"]
        elif archive_pins[path] != r["archive_sha256"]:
            raise ValueError("inconsistent archive pins")
        with np.load(path, allow_pickle=False) as z:
            lo, hi = z["offsets"][r["index_in_shard"] : r["index_in_shard"] + 2]
            source = z["prediction"][lo:hi].copy()
            target = z["target"][lo:hi].copy()
        for tensor, key in ((source, "prediction"), (target, "target")):
            import hashlib

            if hashlib.sha256(tensor.tobytes()).hexdigest() != r[f"{key}_sha256"]:
                raise ValueError("tensor hash mismatch")
        if source.shape != (r["length"], 3) or target.shape != source.shape or not r["mask_all_valid"]:
            raise ValueError("panel length/mask mismatch")
        x, y = torch.from_numpy(source)[None], torch.from_numpy(target)[None]
        m = torch.ones(x.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            out = model(x, m)
        if not torch.equal(out["prediction"], out["global_prediction"]):
            raise AssertionError("zero initialization parity failed")
        items.append(
            dict(
                pg=out["global_prediction"].detach(),
                source=x,
                target=y,
                mask=m,
                sample_id=r["sample_id"],
                condition=r["timestep"],
                stratum=r["stratum"],
            )
        )
        if (index + 1) % 20 == 0:
            print(f"Verified frozen Pg {index + 1}/60", flush=True)
    return items, archive_pins


def main():
    if (OUT / "results.json").exists():
        raise FileExistsError("refusing to overwrite versioned diagnostic results")
    protocol = json.loads((OUT / "protocol.json").read_text())
    if protocol["alpha"] != [0.0, 0.0001, 0.001, 0.01] or protocol["delta_cart"] != [0.0, 0.005, 0.01]:
        raise ValueError("unauthorized diagnostic grid")
    if protocol["variants"] != {
        "A": {"beta": 0.0, "rho": 0.0},
        "B": {"beta": 1.0, "rho": 0.0},
        "C": {"beta": 1.0, "rho": 0.01},
    }:
        raise ValueError("unauthorized coefficient change")
    torch.set_num_threads(protocol["threads"])
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    cfg = yaml.safe_load((ROOT / "configs/e010_phase4d_hybrid_local_global_v1.yaml").read_text())
    checkpoint = Path(cfg["global"]["checkpoint"])
    protected = [
        PREP / "PROTOCOL.md",
        PREP / "preparation.json",
        PREP / "tiny_panel.json",
        PREP / "validation.json",
        ROOT / "configs/e010_phase4d_hybrid_local_global_v1.yaml",
        OUT / "protocol.json",
    ]
    for folder in ("phase4b_real_denoiser_v1", "phase4c_local_auxiliary_v1"):
        protected.extend(p for p in (PREP.parent / folder).rglob("*") if p.is_file())
    pins = {p: file_hash(p) for p in protected}
    pins[checkpoint] = SOURCE_SHA256
    assert_file_pins(pins)
    model = load_frozen_hybrid(checkpoint)
    global_before = state_hash(model.global_model)
    prep = json.loads((PREP / "preparation.json").read_text())
    if global_before != prep["model_state_sha256"]:
        raise ValueError("prepared global state pin mismatch")
    items, archive_pins = load_panel(model, json.loads((PREP / "tiny_panel.json").read_text()))
    pins.update(archive_pins)
    result = run_diagnostic(model.local, items, protocol, progress=lambda s: print(s, flush=True))
    isolated = all(p.grad is None and not p.requires_grad for p in model.global_model.parameters())
    unchanged = state_hash(model.global_model) == global_before
    if not isolated or not unchanged:
        raise AssertionError("E010 isolation/state contract failed")
    assert_file_pins(pins)
    result.update(
        schema=protocol["schema"],
        protocol_sha256=file_hash(OUT / "protocol.json"),
        preparation_base=protocol["base_commit"],
        execution_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        device="cpu",
        cuda_used=False,
        training_launched=False,
        global_gradient_isolation=isolated,
        global_model_unchanged=unchanged,
        input_files_unchanged=True,
        checkpoint_sha256=SOURCE_SHA256,
        global_state_sha256=global_before,
        panel_sha256=file_hash(PREP / "tiny_panel.json"),
        protected_input_sha256={str(p): h for p, h in pins.items()},
        torch_version=torch.__version__,
        threads=torch.get_num_threads(),
    )
    with (OUT / "results.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print("Diagnostic complete; optimizer steps=0; CUDA used=False", flush=True)


if __name__ == "__main__":
    main()
