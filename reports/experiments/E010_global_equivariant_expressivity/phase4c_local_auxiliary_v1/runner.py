#!/usr/bin/env python3
"""Pre-registered matched 364-update Phase 4C continuation; no historical writes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "phase4b_real_denoiser_v1"
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(BASE)]
from protein_distance_diffusion.training.local_geometry import phase4c_losses  # noqa: E402

SPEC = importlib.util.spec_from_file_location("e010_phase4c_historical", BASE / "runner.py")
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)
pc = history.pc
START = BASE / "phase4b_training_v1.final/latest.pt"
START_HASH = "f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef"
PUBLIC_BASELINE = "3b200ffb0389be87cc928c391117ce96bd8b5eb2"
LAMBDA = 0.06719407652184162
BUDGET = 364
BOUNDARIES = (0, 91, 182, 273, 364)
ARMS = {"cartesian_control": 0.0, "local_auxiliary": LAMBDA}
OUT = HERE / "execution"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def digest(value):
    return hashlib.sha256(pc.canonical(value)).hexdigest()


def tensors_hash(value):
    h = hashlib.sha256()

    def visit(item, path=""):
        if torch.is_tensor(item):
            h.update(path.encode())
            h.update(str(item.dtype).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], path + "/" + str(key))
        elif isinstance(item, (list, tuple)):
            for index, entry in enumerate(item):
                visit(entry, path + "/" + str(index))

    visit(value)
    return h.hexdigest()


def atomic_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.pt")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def batch_tensors(rows, device):
    n = max(int(r["length"]) for r in rows)
    x = torch.zeros((len(rows), n, 3), device=device)
    y = torch.zeros_like(x)
    mask = torch.zeros((len(rows), n), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        length = int(row["length"])
        x[i, :length] = torch.from_numpy(row["prediction"]).to(device)
        y[i, :length] = torch.from_numpy(row["target"]).to(device)
        mask[i, :length] = True
    return x, y, mask


def schedule_records(records):
    """Replay the first 364 cached historical optimizer batches, without new noise."""
    queues = {u: {s: [] for s in pc.STRATA} for u in range(1, BUDGET + 1)}
    for row in records:
        if row["split"] == "train" and 1 <= int(row["schedule_update"]) <= BUDGET:
            queues[int(row["schedule_update"])][row["stratum"]].append(row)
    metadata = []
    for u, strata in queues.items():
        for s, rows in strata.items():
            rows.sort(key=lambda r: r["microbatch_position"])
            if len(rows) != 18 or [r["microbatch_position"] for r in rows] != list(range(18)):
                raise ValueError(f"historical 18-example stratum batch mismatch: {u}/{s}")
            for row in rows:
                metadata.append({k: v for k, v in row.items() if k not in ("prediction", "target")})
    return queues, metadata


def read_verified_cache(root=None, expected_manifest_sha256=None):
    """Validate historical bytes independently of pre-formatting source-byte pins.

    The audited cache manifest is immutable. Its archive and every individual
    input/target tensor hash are checked; current source is pinned separately.
    Historical configurations/checkpoints/cache files are never modified.
    """
    root = BASE / "phase4b_real_denoiser_v1.final" if root is None else Path(root)
    if expected_manifest_sha256 is None:
        expected_manifest_sha256 = json.loads((HERE / "diagnostic_panel.json").read_text())["cache_manifest_sha256"]
    if pc.sha256(root / "manifest.json") != expected_manifest_sha256:
        raise ValueError("historical cache manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    records = []
    counts = Counter()
    for shard in manifest["shards"]:
        path = root / shard["shard"]
        if pc.sha256(path) != shard["archive_sha256"]:
            raise ValueError("cache archive hash mismatch")
        if json.loads(path.with_suffix(".json").read_text()) != shard:
            raise ValueError("cache sidecar mismatch")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"prediction", "target", "offsets", "records_json"}:
                raise ValueError("cache fields mismatch")
            meta = json.loads(z["records_json"].tobytes())
            offsets, pred, target = z["offsets"], z["prediction"], z["target"]
            if (
                meta != shard["records"]
                or len(offsets) != len(meta) + 1
                or offsets.dtype != np.int64
                or offsets[0] != 0
                or offsets[-1] != len(pred)
                or pred.dtype != np.float32
                or target.dtype != np.float32
                or pred.shape != target.shape
                or pred.ndim != 2
                or pred.shape[1] != 3
                or not np.isfinite(pred).all()
                or not np.isfinite(target).all()
            ):
                raise ValueError("cache metadata/shape/finiteness mismatch")
            for i, row in enumerate(meta):
                a, b = int(offsets[i]), int(offsets[i + 1])
                if b - a != row["length"] or not row["mask_all_valid"]:
                    raise ValueError("cache length/mask mismatch")
                x, y = pred[a:b].copy(), target[a:b].copy()
                if (
                    hashlib.sha256(x.tobytes()).hexdigest() != row["prediction_sha256"]
                    or hashlib.sha256(y.tobytes()).hexdigest() != row["target_sha256"]
                ):
                    raise ValueError("cache tensor hash mismatch")
                records.append({**row, "prediction": x, "target": y})
                counts[row["split"]] += 1
    if (
        len(records) != manifest["record_count"]
        or counts["train"] != manifest["training_record_count"]
        or counts["development"] != manifest["development_record_count"]
    ):
        raise ValueError("cache split counts mismatch")
    return records, manifest


def verify_records(records):
    for row in records:
        x, y, n = row["prediction"], row["target"], int(row["length"])
        if x.shape != (n, 3) or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"cache integrity violation: {row['sample_id']}")
    dev = [r for r in records if r["split"] == "development"]
    if len(dev) != 960 or len({r["sample_id"] for r in dev}) != 320:
        raise ValueError("full fixed evaluation population missing")
    if len({(r["sample_id"], r["condition_index"]) for r in dev}) != 960:
        raise ValueError("duplicate evaluation identity-condition")
    return dev


def input_hashes(rows):
    return [
        {
            "sample_id": r["sample_id"],
            "condition_index": r["condition_index"],
            "seed": r["seed"],
            "source_sha256": r["source_sha256"],
            "prediction_sha256": hashlib.sha256(r["prediction"].tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(r["target"].tobytes()).hexdigest(),
        }
        for r in rows
    ]


def prepare():
    if git("status", "--porcelain=v1"):
        raise ValueError("contract requires committed, clean working tree")
    if git("branch", "--show-current") != "e010-phase4c-local-auxiliary":
        raise ValueError("dedicated experiment branch required")
    if pc.sha256(START) != START_HASH:
        raise ValueError("starting checkpoint hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("historical CUDA execution required")
    records, manifest = read_verified_cache()
    dev = verify_records(records)
    queues, schedule = schedule_records(records)
    selected = [r for strata in queues.values() for rows in strata.values() for r in rows]
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "contract.json").exists():
        raise FileExistsError("immutable contract already exists; use --execute or --resume")
    frozen_manifest = json.loads((HERE / "diagnostic_panel.json").read_text())
    panel_keys = sorted(
        [[r["sample_id"], r["seed"], r["timestep"]] for r in frozen_manifest["records"] if r["split"] == "train"]
    )
    receipt = json.loads((HERE / "source_equivalence.json").read_text())
    model_path = ROOT / "src/protein_distance_diffusion/models/e010_global_equivariant.py"
    if (
        pc.sha256(model_path) != receipt["execution_source_sha256"]
        or hashlib.sha256(ast.dump(ast.parse(model_path.read_text()), include_attributes=False).encode()).hexdigest()
        != receipt["canonical_ast_sha256"]
    ):
        raise ValueError("verified architecture/source equivalence changed")
    pins = {
        "historical_model_source_sha256": receipt["historical_source_sha256"],
        "source_equivalence": receipt,
        "historical_config_sha256": pc.sha256(BASE / "config.yaml"),
        "model_config_sha256": pc.sha256(ROOT / pc.config()["e010"]["config"]),
    }
    relevant = [
        str(Path(__file__).relative_to(ROOT)),
        "src/protein_distance_diffusion/training/local_geometry.py",
        str((BASE / "runner.py").relative_to(ROOT)),
        str((BASE / "prepare_cache.py").relative_to(ROOT)),
        "tests/test_e010_phase4c_local_auxiliary.py",
    ]
    relevant.extend(
        str((HERE / name).relative_to(ROOT)) for name in ("diagnostic_panel.json", "source_equivalence.json")
    )
    sources = {p: pc.sha256(ROOT / p) for p in relevant}
    source_data = sorted({(r["source_path"], r["source_sha256"]) for r in selected + dev})
    for path, expected in source_data:
        if pc.sha256(ROOT / path) != expected:
            raise ValueError(f"source data changed: {path}")
    state = torch.load(START, map_location="cpu", weights_only=False)
    if state["global_update"] != 1092 or state["scheduler"] is not None:
        raise ValueError("starting optimizer cursor/scheduler mismatch")
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1:
        raise ValueError("unexpected parameter grouping")
    group = {k: v for k, v in groups[0].items() if k != "params"}
    if (
        group["lr"] != 0.0003
        or group["weight_decay"] != 0
        or tuple(group["betas"]) != (0.9, 0.999)
        or group["eps"] != 1e-8
        or group["amsgrad"]
    ):
        raise ValueError("historical Adam configuration changed")
    if len(state["optimizer"]["state"]) != 129 or any(
        float(v["step"]) != 1092 for v in state["optimizer"]["state"].values()
    ):
        raise ValueError("starting Adam steps mismatch")
    contract = {
        "schema": "e010_phase4c_local_auxiliary_v1",
        "public_baseline_sha": PUBLIC_BASELINE,
        "execution_git_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "source_hashes": sources,
        "historical_pins": pins,
        "checkpoint": str(START.relative_to(ROOT)),
        "checkpoint_sha256": START_HASH,
        "initial_model_tensor_hash": tensors_hash(state["model"]),
        "initial_optimizer_tensor_hash": tensors_hash(state["optimizer"]),
        "arms": ARMS,
        "condition_weights": ["1/3", "1/3", "1/3"],
        "objective": {
            "cartesian": (
                "equal-protein masked xyz-component MSE +1e-5 masked residual-component MSE; no "
                "alignment; Angstrom squared"
            ),
            "local": (
                "mean of endpoint-masked distance-error MSE offsets1/2/3, each normalized within "
                "protein, then equal proteins"
            ),
            "microbatch_reduction": "18 proteins per stratum, 5 strata, each microbatch objective divided by 5",
        },
        "cache_manifest_sha256": pc.sha256(BASE / "phase4b_real_denoiser_v1.final/manifest.json"),
        "historical_cache_digest": digest(manifest),
        "schedule_sha256": digest(schedule),
        "realized_condition_counts": dict(Counter(int(r["timestep"]) for r in selected)),
        "condition_sampling_semantics": (
            "historical equal-probability pool, unchanged unweighted microbatch reduction; "
            "finite schedule realized counts recorded, not forced balanced"
        ),
        "training_schedule": (
            "replay historical schedule_update1..364, fixed cached noise; no augmentation or reshuffle"
        ),
        "training_example_count": len(selected),
        "training_identity_count": len({r["sample_id"] for r in selected}),
        "training_array_hashes_sha256": digest(input_hashes(selected)),
        "evaluation_array_hashes_sha256": digest(input_hashes(dev)),
        "evaluation_population": {
            "identities": 320,
            "conditions": 960,
            "identity_condition_order_sha256": digest([[r["sample_id"], r["condition_index"]] for r in dev]),
        },
        "optimizer": group,
        "scheduler": None,
        "gradient_clip_norm": 5.0,
        "precision": "historical float32, no AMP, no changed TF32 flags",
        "rng": (
            "restore identical checkpoint Python/NumPy/Torch/CUDA states separately for both "
            "arms; no stochastic model layers"
        ),
        "seed_reference": 41045,
        "bootstrap_seed": 41046,
        "bootstrap_replicates": 10000,
        "updates": BUDGET,
        "boundaries": list(BOUNDARIES),
        "global_final_update": 1456,
        "primary_endpoint": (
            "auxiliary vs matched control at continuation update364 on full fixed evaluation population"
        ),
        "gates": {
            "1": (
                "paired identity mean of mean local RMSE relative improvement >0 and bootstrap95% "
                "lower bound>0; offsets disclosed"
            ),
            "2": "existing paired aligned RMSD improvement >=-0.02 (2% non-inferiority)",
            "3": (
                "all condition mean local RMSE improvements >=-0.02; preferred each>0; major regression pre-defined>2%"
            ),
            "4": (
                "chirality assessable in both; auxiliary inversion <=control (existing Phase4B "
                "semantics, no relaxed margin)"
            ),
            "5": "finite rate1; no meanRg<1e-3; each stratum Rg SD >=.05*target SD",
            "6": (
                "no stratum mean local RMSE or aligned RMSD relative regression>2%; length slope "
                "reported, Phase4B slope gate separately reported"
            ),
        },
        "stop_rules": [
            "nonfinite loss/gradient/parameter/output",
            "cache/source/checkpoint integrity violation",
            "catastrophic coordinate collapse meanRg<1e-3",
            "unrecoverable runtime failure",
        ],
        "classification_precedence": (
            "safety->C5; local benefit with Cartesian margin fail->C2; condition/stratum->C4; "
            "local gate fail->C3; otherwiseC1"
        ),
        "gradient_panel_keys": panel_keys,
        "gradient_panel_manifest_sha256": digest(frozen_manifest),
        "gradient_panel": "fixed five established training diagnostic identities, all3conditions; explanatory only",
        "gradient_boundaries": list(BOUNDARIES),
        "module_boundaries": [0, 364],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "authorization": {"short_matched_364": True, "full_adaptation": False, "production": False},
    }
    pc.atomic_json(OUT / "training_schedule.json", schedule)
    pc.atomic_json(
        OUT / "data_manifest.json",
        {"sources": source_data, "training": input_hashes(selected), "evaluation": input_hashes(dev)},
    )
    pc.atomic_json(OUT / "contract.json", contract)
    pc.atomic_json(OUT / "contract_receipt.json", {"contract_sha256": pc.sha256(OUT / "contract.json")})
    validate_contract()
    print(
        json.dumps(
            {
                "contract_sha256": pc.sha256(OUT / "contract.json"),
                "training_examples": len(selected),
                "evaluation_examples": len(dev),
            },
            indent=2,
        )
    )


def validate_contract():
    contract = json.loads((OUT / "contract.json").read_text())
    receipt = json.loads((OUT / "contract_receipt.json").read_text())
    if pc.sha256(OUT / "contract.json") != receipt["contract_sha256"]:
        raise ValueError("immutable contract was modified")
    if git("status", "--porcelain=v1") or git("rev-parse", "HEAD") != contract["execution_git_sha"]:
        raise ValueError("execution must use exact committed clean source")
    for path, expected in contract["source_hashes"].items():
        if pc.sha256(ROOT / path) != expected:
            raise ValueError(f"source hash changed: {path}")
    if pc.sha256(START) != START_HASH:
        raise ValueError("source checkpoint modified")
    if contract["arms"] != ARMS or contract["updates"] != BUDGET or contract["boundaries"] != list(BOUNDARIES):
        raise ValueError("budget/arms/boundaries contradiction")
    if digest(json.loads((OUT / "training_schedule.json").read_text())) != contract["schedule_sha256"]:
        raise ValueError("schedule hash mismatch")
    data = json.loads((OUT / "data_manifest.json").read_text())
    if (
        digest(data["training"]) != contract["training_array_hashes_sha256"]
        or digest(data["evaluation"]) != contract["evaluation_array_hashes_sha256"]
    ):
        raise ValueError("data hashes changed")
    return contract


def restore_state(model, optimizer, state):
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
    if state["scheduler"] is not None:
        raise ValueError("historical constant scheduler required")


def full_state(model, optimizer, update, arm, contract_hash, telemetry):
    return {
        "schema": "e010_phase4c_full_state_v1",
        "continuation_update": update,
        "global_update": 1092 + update,
        "arm": arm,
        "contract_sha256": contract_hash,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "scaler": {},
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "telemetry": telemetry,
    }


def evaluate(model, dev):
    model.eval()
    evaluated = []
    cart, coord, residual = [], [], []
    with torch.inference_mode():
        for row in dev:
            x, y, mask = batch_tensors([row], next(model.parameters()).device)
            pred = model(x, mask)["prediction"]
            if not torch.isfinite(pred).all():
                raise FloatingPointError("nonfinite evaluation output")
            losses = phase4c_losses(pred, x, y, mask)
            cart.append(float(losses["cartesian"]))
            coord.append(float(losses["coordinate_mse"]))
            residual.append(float(losses["residual_mse"]))
            evaluated.append({**row, "adapted": pred[0].cpu().numpy()})
    metrics = history._metrics(evaluated, "adapted")
    metrics.update(
        unaligned_cartesian_objective=float(np.mean(cart)),
        unaligned_coordinate_mse=float(np.mean(coord)),
        residual_component_mse=float(np.mean(residual)),
    )
    metrics["mean_local_rmse"] = float(np.mean(list(metrics["local_distance_rmse_i_plus_1_2_3"].values())))
    for level in ("conditions", "length_strata"):
        for group in metrics[level].values():
            group["mean_local_rmse"] = float(np.mean(list(group["local_distance_rmse_i_plus_1_2_3"].values())))
    if any(v["collapse"] for v in metrics["diversity_by_stratum"].values()):
        raise FloatingPointError("catastrophic coordinate collapse")
    return metrics


def gradient_audit(model, optimizer, panel, weight, modules=False):
    """Bounded panel audit: autograd.grad only, virtual Adam, no persistent changes."""
    model.eval()
    named = list(model.named_parameters())
    params = [p for _, p in named]
    cart = [torch.zeros_like(p) for p in params]
    local = [torch.zeros_like(p) for p in params]
    for row in panel:
        x, y, mask = batch_tensors([row], next(model.parameters()).device)
        losses = phase4c_losses(model(x, mask)["prediction"], x, y, mask)
        for dest, key, keep in ((cart, "cartesian", True), (local, "local_mean", False)):
            grads = torch.autograd.grad(losses[key], params, retain_graph=keep, allow_unused=True)
            for acc, g in zip(dest, grads, strict=True):
                if g is not None:
                    acc.add_(g.detach() / len(panel))

    def norm(gs):
        return math.sqrt(sum(float(g.double().square().sum()) for g in gs))

    def dot(left, right):
        return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right, strict=True))

    total = [a + weight * b for a, b in zip(cart, local, strict=True)]
    cn, ln, tn = norm(cart), norm(local), norm(total)
    q = min(1.0, 5.0 / (tn + 1e-6))
    result = {
        "cart_gradient_norm": cn,
        "local_gradient_norm": ln,
        "total_gradient_norm": tn,
        "raw_cosine": dot(cart, local) / max(cn * ln, 1e-30),
        "clip_coefficient": q,
        "module_interactions": {},
    }
    if modules:
        group = optimizer.param_groups[0]
        b1, b2 = group["betas"]
        by_module = {}
        for (name, p), cg, lg, tg in zip(named, cart, local, total, strict=True):
            state = optimizer.state[p]
            t = float(state["step"]) + 1
            g = q * tg
            m = b1 * state["exp_avg"] + (1 - b1) * g
            v = b2 * state["exp_avg_sq"] + (1 - b2) * g.square()
            delta = -group["lr"] * (m / (1 - b1**t)) / (torch.sqrt(v / (1 - b2**t)) + group["eps"])
            block = ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
            r = by_module.setdefault(
                block,
                {
                    "parameters": 0,
                    "cart_norm_sq": 0.0,
                    "local_norm_sq": 0.0,
                    "raw_dot": 0.0,
                    "adam_local_derivative": 0.0,
                },
            )
            r["parameters"] += p.numel()
            r["cart_norm_sq"] += float(cg.double().square().sum())
            r["local_norm_sq"] += float(lg.double().square().sum())
            r["raw_dot"] += float((cg.double() * lg.double()).sum())
            r["adam_local_derivative"] += float((lg.double() * delta.double()).sum())
        for r in by_module.values():
            r["raw_cosine"] = r["raw_dot"] / max(math.sqrt(r["cart_norm_sq"] * r["local_norm_sq"]), 1e-30)
        result["module_interactions"] = by_module
    return result


def paired_local(left, right):
    def identity_means(metrics):
        grouped = {}
        for row in metrics["per_condition"]:
            grouped.setdefault(row["sample_id"], []).append(np.mean(list(row["local_distance_rmse"].values())))
        return {sid: float(np.mean(v)) for sid, v in grouped.items()}

    b, c = identity_means(left), identity_means(right)
    if b.keys() != c.keys():
        raise ValueError("paired local identity mismatch")
    gains = np.array([(b[s] - c[s]) / max(b[s], 1e-12) for s in sorted(b)])
    rng = np.random.default_rng(41046)
    boot = np.mean(gains[rng.integers(0, len(gains), size=(10000, len(gains)))], axis=1)
    return {
        "identity_count": len(gains),
        "mean_paired_percentage_improvement": float(gains.mean()),
        "bootstrap_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def compare(left, right):
    cart = history._paired(left, right)
    local = paired_local(left, right)

    def improvements(level, key):
        return {s: (left[level][s][key] - right[level][s][key]) / max(left[level][s][key], 1e-12) for s in left[level]}

    condition = improvements("conditions", "mean_local_rmse")
    strata = improvements("length_strata", "mean_local_rmse")
    stratum_cart = improvements("length_strata", "mean_rmse")
    gates = {
        "1": local["mean_paired_percentage_improvement"] > 0 and local["bootstrap_ci95"][0] > 0,
        "2": cart["mean_paired_percentage_improvement"] >= -0.02,
        "3": min(condition.values()) >= -0.02,
        "4": left["chirality_assessable"]
        and right["chirality_assessable"]
        and right["chirality_inversion_rate"] <= left["chirality_inversion_rate"],
        "5": right["finite_output_rate"] == 1
        and all(
            not r["collapse"]
            and r["radius_gyration_sd_prediction"] >= 0.05 * max(r["radius_gyration_sd_target"], 1e-12)
            for r in right["diversity_by_stratum"].values()
        ),
        "6": min(strata.values()) >= -0.02 and min(stratum_cart.values()) >= -0.02,
    }
    if not gates["4"] or not gates["5"]:
        classification = "C5"
    elif gates["1"] and not gates["2"]:
        classification = "C2"
    elif not gates["3"] or not gates["6"]:
        classification = "C4"
    elif not gates["1"]:
        classification = "C3"
    else:
        classification = "C1"
    return {
        "classification": classification,
        "gates": gates,
        "paired_cartesian": cart,
        "paired_mean_local": local,
        "condition_local_improvement": condition,
        "stratum_local_improvement": strata,
        "stratum_cartesian_improvement": stratum_cart,
        "legacy_length_slope_not_increased": right["error_vs_length_slope"] <= left["error_vs_length_slope"],
    }


def execute(resume=False):
    contract = validate_contract()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    records, manifest = read_verified_cache()
    queues, schedule = schedule_records(records)
    dev = verify_records(records)
    if digest(schedule) != contract["schedule_sha256"] or digest(manifest) != contract["historical_cache_digest"]:
        raise ValueError("schedule/cache identity changed")
    selected = [r for strata in queues.values() for rows in strata.values() for r in rows]
    if (
        digest(input_hashes(selected)) != contract["training_array_hashes_sha256"]
        or digest(input_hashes(dev)) != contract["evaluation_array_hashes_sha256"]
    ):
        raise ValueError("training/evaluation tensors changed")
    keys = {tuple(r) for r in contract["gradient_panel_keys"]}
    panel = [r for r in records if r["split"] == "train" and (r["sample_id"], r["seed"], r["timestep"]) in keys]
    if len(panel) != 15:
        raise ValueError("fixed gradient panel mismatch")
    ch = pc.sha256(OUT / "contract.json")
    execution = {
        "contract_sha256": ch,
        "git_sha": git("rev-parse", "HEAD"),
        "starting_checkpoint_sha256": START_HASH,
        "status": "running",
        "arms": {},
        "source_unchanged": None,
    }
    pc.atomic_json(OUT / "execution_manifest.json", execution)
    start_time = time.monotonic()
    for arm, weight in ARMS.items():
        directory = OUT / arm
        directory.mkdir(exist_ok=True)
        latest = directory / "latest.pt"
        if latest.exists():
            if not resume:
                raise FileExistsError("existing execution state; use --resume")
            continue
        source = torch.load(START, map_location="cpu", weights_only=False)
        model = history._model("cuda", load_selected=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0)
        restore_state(model, optimizer, source)
        telemetry = []
        if (
            tensors_hash(model.state_dict()) != contract["initial_model_tensor_hash"]
            or tensors_hash(optimizer.state_dict()) != contract["initial_optimizer_tensor_hash"]
        ):
            raise ValueError("arm initialization differs from checkpoint")
        metrics = evaluate(model, dev)
        audit = gradient_audit(model, optimizer, panel, weight, modules=True)
        pc.atomic_json(directory / "evaluation_000.json", metrics)
        pc.atomic_json(directory / "gradient_000.json", audit)
        state = full_state(model, optimizer, 0, arm, ch, telemetry)
        atomic_checkpoint(directory / "checkpoint_000.pt", state)
        atomic_checkpoint(latest, state)
        execution["arms"][arm] = {
            "initial_model_tensor_hash": tensors_hash(model.state_dict()),
            "initial_optimizer_tensor_hash": tensors_hash(optimizer.state_dict()),
        }
        if arm == "local_auxiliary":
            control = json.loads((OUT / "cartesian_control/evaluation_000.json").read_text())
            if metrics != control:
                raise ValueError("matched update0 evaluation not identical")
            pc.atomic_json(
                OUT / "update0_matching.json",
                {
                    "metrics_exactly_equal": True,
                    "model_optimizer_exactly_equal": True,
                    "schedule_sha256": contract["schedule_sha256"],
                },
            )
        del model, optimizer, source, state
        torch.cuda.empty_cache()
    if not (OUT / "update0_matching.json").exists():
        raise ValueError("update-zero match must complete before either arm trains")
    for arm, weight in ARMS.items():
        directory = OUT / arm
        source = torch.load(directory / "latest.pt", map_location="cpu", weights_only=False)
        if source["contract_sha256"] != ch or source["arm"] != arm:
            raise ValueError("resume contract/arm mismatch")
        model = history._model("cuda", load_selected=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0)
        restore_state(model, optimizer, source)
        cursor = source["continuation_update"]
        telemetry = source["telemetry"]
        latest = directory / "latest.pt"
        for update in range(cursor + 1, BUDGET + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            values = {
                key: 0.0
                for key in (
                    "total",
                    "cartesian",
                    "coordinate_mse",
                    "residual_mse",
                    "local_1",
                    "local_2",
                    "local_3",
                    "local_mean",
                )
            }
            for stratum in pc.STRATA:
                x, y, mask = batch_tensors(queues[update][stratum], "cuda")
                with torch.autocast("cuda", enabled=False):
                    losses = phase4c_losses(model(x, mask)["prediction"], x, y, mask, local_weight=weight)
                    loss = losses["total"] / 5
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                for key in values:
                    values[key] += float(losses[key].detach()) / 5
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameters")
            event = {
                "continuation_update": update,
                "global_update": 1092 + update,
                **values,
                "gradient_norm_before_clipping": float(norm),
                "clip_coefficient": min(1.0, 5.0 / (float(norm) + 1e-6)),
                "lr": optimizer.param_groups[0]["lr"],
                "schedule_update": update,
            }
            telemetry.append(event)
            if update in BOUNDARIES:
                pc.atomic_json(directory / f"evaluation_{update:03d}.json", evaluate(model, dev))
                pc.atomic_json(
                    directory / f"gradient_{update:03d}.json",
                    gradient_audit(model, optimizer, panel, weight, modules=update == 364),
                )
            state = full_state(model, optimizer, update, arm, ch, telemetry)
            atomic_checkpoint(latest, state)
            if update in BOUNDARIES:
                atomic_checkpoint(directory / f"checkpoint_{update:03d}.pt", state)
            pc.atomic_json(directory / "telemetry.json", telemetry)
            pc.atomic_json(
                OUT / "progress.json",
                {
                    "arm": arm,
                    "update": update,
                    "budget": BUDGET,
                    "elapsed_seconds": time.monotonic() - start_time,
                    **event,
                },
            )
            print(json.dumps({"arm": arm, **event}), flush=True)
        if any(float(v["step"]) != 1456 for v in optimizer.state.values()):
            raise ValueError("final optimizer steps mismatch")
        del model, optimizer, source, state
        torch.cuda.empty_cache()
    comparisons = {}
    hashes = {}
    for update in BOUNDARIES:
        left = json.loads((OUT / f"cartesian_control/evaluation_{update:03d}.json").read_text())
        right = json.loads((OUT / f"local_auxiliary/evaluation_{update:03d}.json").read_text())
        comparisons[str(update)] = compare(left, right)
        for arm in ARMS:
            path = OUT / arm / f"checkpoint_{update:03d}.pt"
            hashes[str(path.relative_to(HERE))] = pc.sha256(path)
    result = {
        "schema": "e010_phase4c_results_v1",
        "status": "completed_non_authorizing",
        "updates_per_arm": BUDGET,
        "contract_sha256": ch,
        "comparisons": comparisons,
        "primary": comparisons["364"],
        "checkpoint_hashes": hashes,
        "elapsed_seconds": time.monotonic() - start_time,
        "source_checkpoint_unchanged": pc.sha256(START) == START_HASH,
    }
    if not result["source_checkpoint_unchanged"]:
        raise ValueError("historical checkpoint changed")
    pc.atomic_json(OUT / "results.json", result)
    pc.atomic_json(OUT / "adjudication.json", result["primary"])
    pc.atomic_json(OUT / "checkpoint_hashes.json", hashes)
    execution.update(
        status="completed_non_authorizing", source_unchanged=True, elapsed_seconds=result["elapsed_seconds"]
    )
    pc.atomic_json(OUT / "execution_manifest.json", execution)
    print(json.dumps(result["primary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "execute", "resume"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "validate":
        print(json.dumps(validate_contract(), indent=2))
    else:
        execute(resume=args.mode == "resume")


if __name__ == "__main__":
    main()
