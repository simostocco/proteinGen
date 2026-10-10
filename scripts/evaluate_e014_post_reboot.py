"""Read-only post-shutdown checkpoint audit; outputs and runtime caches stay on D:."""

# ruff: noqa: I001, E501
from e014_bootstrap import ROOT

import argparse
import hashlib
import json
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from coordinate_execution_policy import canonical_runner
from diagnose_e014_high_noise import population
from protein_distance_diffusion.models.e014_capacity import E014CapacityDenoiser
from protein_distance_diffusion.training.e014_runtime import atomic_json, canonical_hash, epoch_schedule, sha
from upgrade_e014_terminal_guard import FROZEN_PROBE_SOURCE

OUT = Path(__file__).resolve().parents[1] / "e014_artifacts/e014/post_reboot_full_evaluation_v1"
OLD = ROOT / "e014/high_noise_75pct_v1"
SELECTED = ("epoch_1_fraction_1.00", "epoch_2_fraction_1.00", "recovery_0")


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml").read_text())
    metadata, panel, subset = population()
    train = pq.read_table(ROOT / "population/train.parquet", use_threads=False).to_pylist()
    assert len(train) == 230132
    assert canonical_hash(sorted(r["sample_id"] for r in train)) == metadata["train_sample_id_hash"]
    schedules = {e: epoch_schedule([r["length"] for r in train], cfg["batch_sizes"], cfg["seed"], e) for e in (1, 2, 3)}
    hashes = {e: canonical_hash(v) for e, v in schedules.items()}
    for e in schedules:
        stored = json.loads((ROOT / f"e014/EPOCH_{e}_SCHEDULE.json").read_text())
        assert stored == schedules[e]
    latest = json.loads((ROOT / "e014/checkpoints/LATEST.json").read_text())
    trusted = {latest["path"]: latest["sha256"]}
    old_ids = json.loads((OLD / "checkpoint_identity.json").read_text())["checkpoints"]
    trusted.update({r["path"]: r["sha256"] for r in old_ids})
    runner = canonical_runner(Path("scripts/run_e014_scaleup.py").read_text()).encode()
    old_runner = subprocess.check_output(["git", "show", f"{FROZEN_PROBE_SOURCE}:scripts/run_e014_scaleup.py"])
    inventory = []
    for path in sorted((ROOT / "e014/checkpoints").iterdir()):
        if path.suffix != ".pt":
            if "partial" in path.name:
                inventory.append(dict(path=str(path), eligible=False, reason="partial file excluded"))
            continue
        record = dict(label=path.stem, path=str(path), bytes=path.stat().st_size, mtime=path.stat().st_mtime)
        try:
            digest = sha(path)
            s = torch.load(path, map_location="cpu", weights_only=False)
            c = s["contract"]
            assert c["config_hash"] == sha("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml")
            assert c["data_hash"] == metadata["train_sample_id_hash"]
            assert c["normalization_hash"] == sha(cfg["normalization_artifact"]) == cfg["normalization_sha256"]
            assert c["population_manifest_hash"] == sha(ROOT / "population/population.json")
            assert c["cache_ready_hash"] == sha(ROOT / "coordinate_cache/READY.json")
            for file, expected in c["source_hashes"].items():
                actual = (
                    hashlib.sha256(
                        old_runner
                        if path.stem in ("initialization",) or path.stem.startswith("probe_passed")
                        else runner
                    ).hexdigest()
                    if file == "scripts/run_e014_scaleup.py"
                    else sha(file)
                )
                assert actual == expected, file
            model = E014CapacityDenoiser(**c["capacity_config"]).eval()
            model.load_state_dict(s["model"], strict=True)
            assert sum(p.numel() for p in model.parameters()) == 5119069
            assert all(torch.isfinite(v).all() for v in s["model"].values())
            e, pos, updates = s["epoch"], s["position"], s["updates"]
            assert 0 <= pos <= len(schedules[e])
            assert updates == (e - 1) * len(schedules[e]) + pos
            expected = np.full(len(train), e - 1, dtype=np.int32)
            for b in schedules[e][:pos]:
                expected[b] += 1
            assert np.array_equal(expected, s["exposures"])
            if updates:
                assert s["schedule_hash"] == hashes[e]
                assert s["next_batch"] == (schedules[e][pos] if pos < len(schedules[e]) else None)
                assert s["train_loss_proteins"] == int(expected.sum())
                assert s["epoch_loss_proteins"] == int(expected.sum()) - (e - 1) * len(train)
            opt = s["optimizer"]
            params = list(model.parameters())
            assert opt["param_groups"][0]["params"] == list(range(len(params)))
            g = opt["param_groups"][0]
            assert (
                g["lr"] == cfg["optimizer"]["learning_rate"] and g["weight_decay"] == cfg["optimizer"]["weight_decay"]
            )
            assert g["betas"] == (0.9, 0.999) and g["eps"] == 1e-8
            assert len(opt["state"]) == (len(params) if updates else 0)
            for i, v in opt["state"].items():
                assert int(v["step"]) == updates
                for name in ("exp_avg", "exp_avg_sq"):
                    assert v[name].shape == params[i].shape and torch.isfinite(v[name]).all()
                assert (v["exp_avg_sq"] >= 0).all()
            rng = s["rng"]
            assert set(rng) == {"python", "numpy", "torch", "cuda"}
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator().set_state(rng["torch"])
            assert all(x.dtype == torch.uint8 and x.ndim == 1 for x in rng["cuda"])
            if torch.cuda.is_available():
                assert len(rng["cuda"]) == torch.cuda.device_count()
                for i, state in enumerate(rng["cuda"]):
                    torch.Generator(device=f"cuda:{i}").set_state(state)
            assert sha(path) == digest
            if str(path) in trusted:
                assert digest == trusted[str(path)]
            record.update(
                sha256=digest,
                epoch=e,
                position=pos,
                updates=updates,
                exposures=int(expected.sum()),
                completed_epochs=int(expected.min()),
                maximum_exposure=int(expected.max()),
                contract=c,
                eligible=True,
                optimizer_rng_consistent=True,
                model_complete=True,
                hash_evidence="matches persisted SHA256"
                if str(path) in trusted
                else "SHA256 computed twice; ZIP CRC/load valid; no independent historical digest found",
                training_v_loss=s["train_loss_sum"] / max(1, s["train_loss_proteins"]),
                epoch_training_v_loss=s["epoch_loss_sum"] / max(1, s["epoch_loss_proteins"]),
                clipping_fraction=s["clipping_count"] / max(1, updates),
            )
            del model, s
        except Exception as exc:
            record.update(eligible=False, reason=f"{type(exc).__name__}: {exc}")
        inventory.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "contract"}), flush=True)
    eligible = [r for r in inventory if r.get("eligible")]
    newest = max(eligible, key=lambda r: (r["updates"], r["mtime"]))
    assert newest["path"] == latest["path"] and newest["label"] == SELECTED[-1]
    selected = [next(r for r in eligible if r["label"] == label) for label in SELECTED]
    atomic_json(
        OUT / "checkpoint_inventory.json",
        dict(
            checkpoints=inventory,
            newest=newest,
            selected=selected,
            latest_pointer=latest,
            partials_never_evaluated=True,
        ),
    )

    atomic_json(OUT / "checkpoint_identity.json", dict(checkpoints=selected))
    atomic_json(
        OUT / "PROTOCOL.json",
        dict(
            selected_checkpoints=[dict(label=r["label"], sha256=r["sha256"], updates=r["updates"]) for r in selected],
            panel_hash=canonical_hash([r["sample_id"] for r in panel]),
            subset_hash=canonical_hash([r["sample_id"] for r in subset]),
            prospective_policy="Frozen runner permits final prospective only at plateau or maximum complete epoch; neither reached. No prospective metrics inspected.",
            core_timesteps=cfg["noise_timesteps"],
            dense_grid=list(
                __import__("protein_distance_diffusion.evaluation.high_noise_resolution", fromlist=["GRID"]).GRID
            ),
            gaussian_endpoints=[490, 480, 470, 460, 450, 440, 420, 400, 350],
            gaussian_stop=350,
            bootstraps=500,
            reference_source=str(OLD / "shared_reference_descriptors.parquet"),
            reference_sha256=sha(OLD / "shared_reference_descriptors.parquet"),
            previous_protocol_sha256=sha(OLD / "PROTOCOL.json"),
            paired_namespace="e014-paired-validation-v1",
            replay_tolerance=5e-6,
            training_resumed=False,
            full_sampling=False,
            frozen_model_source_unchanged=True,
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        ),
    )


def crc_audit():
    path = OUT / "checkpoint_inventory.json"
    inventory = json.loads(path.read_text())
    by_path = {}
    for record in inventory["checkpoints"]:
        if not record.get("eligible"):
            continue
        with zipfile.ZipFile(record["path"]) as archive:
            bad = archive.testzip()
        assert bad is None, (record["path"], bad)
        assert sha(record["path"]) == record["sha256"]
        record["all_archive_payload_crc_valid"] = True
        by_path[record["path"]] = record
    inventory["newest"] = by_path[inventory["newest"]["path"]]
    inventory["selected"] = [by_path[r["path"]] for r in inventory["selected"]]
    atomic_json(path, inventory)
    print("All complete checkpoint archives passed every payload CRC and SHA256 recheck", flush=True)


def archive_audit():
    path = OUT / "checkpoint_inventory.json"
    inventory = json.loads(path.read_text())
    known = {r["sha256"]: r for r in inventory["checkpoints"] if r.get("eligible")}
    discovered, excluded = [], []
    primary = ROOT / "e014/checkpoints"
    for candidate in sorted(ROOT.rglob("*.pt")):
        discovered.append(str(candidate))
        if candidate.parent == primary:
            continue
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=False)
        except Exception as exc:
            excluded.append(
                dict(path=str(candidate), reason=f"unloadable non-primary payload: {type(exc).__name__}: {exc}")
            )
            continue
        contract = state.get("contract", {}) if isinstance(state, dict) else {}
        if contract.get("config_hash") != inventory["newest"]["contract"]["config_hash"]:
            excluded.append(
                dict(
                    path=str(candidate),
                    reason="test fixture or non-E014 payload; lacks frozen E014 checkpoint contract",
                )
            )
            continue
        digest = sha(candidate)
        assert digest in known, f"Unrecognized additional E014 checkpoint needs full audit: {candidate}"
        with zipfile.ZipFile(candidate) as archive:
            assert archive.testzip() is None
        record = dict(
            known[digest],
            path=str(candidate),
            label=candidate.stem,
            duplicate_of=known[digest]["path"],
            scope="archival byte-identical copy",
            all_archive_payload_crc_valid=True,
        )
        if not any(r["path"] == str(candidate) for r in inventory["checkpoints"]):
            inventory["checkpoints"].append(record)
        assert state["updates"] == record["updates"]
        assert sha(candidate) == digest
    inventory.update(all_artifact_root_pt_files=discovered, excluded_non_experiment_payloads=excluded)
    for record in inventory["selected"][:2]:
        e = record["completed_epochs"]
        exposure = json.loads((ROOT / f"e014/EXPOSURE_EPOCH_{e}.json").read_text())
        assert exposure["updates"] == record["updates"] and exposure["total_exposures"] == record["exposures"]
        assert exposure["minimum"] == exposure["maximum"] == e
        assert exposure["exposure_sha256"] == canonical_hash([e] * 230132)
        assert exposure["missing"] == exposure["duplicates"] == 0
    inventory["completed_epoch_exposure_reports_verified"] = True
    atomic_json(path, inventory)
    print(
        f"Searched all {len(discovered)} .pt files; {len(excluded)} fixtures excluded by payload contract", flush=True
    )


def snapshot():
    """Preserve the verified rolling boundary against future trainer overwrites."""
    import os
    import shutil

    path = OUT / "checkpoint_inventory.json"
    inventory = json.loads(path.read_text())
    newest = inventory["newest"]
    source = Path(newest["path"])
    target = OUT / "newest_durable_checkpoint.pt"
    assert sha(source) == newest["sha256"]
    if not target.exists():
        partial = target.with_name(target.name + ".partial")
        with source.open("rb") as src, partial.open("wb") as dst:
            shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        assert sha(partial) == newest["sha256"]
        partial.replace(target)
    assert sha(target) == newest["sha256"]
    for record in [inventory["newest"], *inventory["selected"], *inventory["checkpoints"]]:
        if record["sha256"] == newest["sha256"]:
            record["immutable_copy"] = str(target)
            record["immutable_copy_sha256"] = newest["sha256"]
    atomic_json(path, inventory)
    print(f"Preserved verified rolling checkpoint: {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["audit", "crc", "archives", "snapshot"])
    args = parser.parse_args()
    if args.phase == "audit":
        audit()
    elif args.phase == "crc":
        crc_audit()
    elif args.phase == "archives":
        archive_audit()
    else:
        snapshot()
