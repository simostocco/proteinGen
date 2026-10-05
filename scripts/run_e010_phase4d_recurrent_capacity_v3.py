#!/usr/bin/env python3
"""Immutable-cache, CUDA preflight and matched bounded recurrent capacity run."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.models.e010_hybrid_local import load_frozen_hybrid
from protein_distance_diffusion.models.e010_recurrent_local import COUNTS, RecurrentLocalRefiner
from protein_distance_diffusion.training.e010_phase4d_diagnostic import (
    assert_file_pins,
    file_hash,
    module_norms,
    state_hash,
)
from protein_distance_diffusion.training.e010_phase4d_objective_v2 import freeze_chirality, gradient_audit
from protein_distance_diffusion.training.e010_recurrent_capacity import (
    batch,
    gates,
    matched_order,
    objective_components,
    restore_checkpoint,
    save_checkpoint,
    select_capacity,
)
from scripts.diagnose_e010_phase4d_frozen_local import PREP, ROOT, load_panel

OUT = PREP / "recurrent_capacity_v3"
CONFIG = ROOT / "configs/e010_phase4d_recurrent_capacity_v3.yaml"
CACHE = OUT / "frozen_panel.npz"


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def config():
    cfg = yaml.safe_load(CONFIG.read_text())
    if (
        cfg["training"]["updates"] != 500
        or cfg["objective"]["beta"] != 16.8
        or cfg["objective"]["gamma"] != 2
        or cfg["recurrence"]["steps"] != 4
        or cfg["bound"]["s_max_angstrom"] != 0.04
    ):
        raise ValueError("contract mutation")
    return cfg


def prepare():
    if CACHE.exists():
        raise FileExistsError("cache immutable; use existing manifest")
    torch.set_num_threads(2)
    cfg = config()
    source = yaml.safe_load((ROOT / "configs/e010_phase4d_hybrid_local_global_v1.yaml").read_text())["global"][
        "checkpoint"
    ]
    pins = {p: file_hash(p) for p in PREP.rglob("*") if p.is_file() and OUT not in p.parents}
    for name in ("phase4b_real_denoiser_v1", "phase4c_local_auxiliary_v1"):
        pins.update({p: file_hash(p) for p in (PREP.parent / name).rglob("*") if p.is_file()})
    pins[Path(source)] = source and file_hash(source)
    model = load_frozen_hybrid(source)
    global_pin = state_hash(model.global_model)
    items, archives = load_panel(model, json.loads((PREP / "tiny_panel.json").read_text()))
    pins.update(archives)
    items.sort(key=lambda r: (r["sample_id"], r["condition"]))
    n = max(r["pg"].shape[1] for r in items)
    data = {k: torch.zeros(60, n, 3) for k in ("pg", "source", "target")}
    data["mask"] = torch.zeros(60, n, dtype=torch.bool)
    records = []
    lengths = []
    quartets = 0
    for i, r in enumerate(items):
        length = r["pg"].shape[1]
        lengths.append(length)
        for k in data:
            data[k][i, :length] = r[k][0]
        records.append({k: r[k] for k in ("sample_id", "condition", "stratum")})
        quartets += int(freeze_chirality(r["pg"], r["target"], r["mask"])["eligible"].sum())
    arrays = {k: t.numpy() for k, t in data.items()}
    arrays["lengths"] = np.array(lengths)
    arrays["records_json"] = np.frombuffer(json.dumps(records, sort_keys=True).encode(), dtype=np.uint8)
    np.savez(CACHE, **arrays)
    # Roundtrip exact parity for all 60 cached predictions, masks, targets/sources.
    with np.load(CACHE, allow_pickle=False) as loaded:
        assert all(np.array_equal(arrays[k], loaded[k]) for k in arrays)
    assert_file_pins(pins)
    write(
        OUT / "cache_manifest.json",
        {
            "schema": "e010_frozen_panel_v3",
            "cache_sha256": file_hash(CACHE),
            "examples": 60,
            "identities": 20,
            "quartets": quartets,
            "global_state_sha256": global_pin,
            "source_checkpoint_sha256": file_hash(source),
            "exact_live_parity_60": True,
            "panel_sha256": file_hash(PREP / "tiny_panel.json"),
            "protected_input_sha256": {str(p): h for p, h in pins.items()},
            "config_sha256": file_hash(CONFIG),
            "variants": cfg["variants"],
            "order": records,
        },
    )
    print("Frozen cache prepared, exact live parity; quartets=", quartets, flush=True)


def load_cache():
    manifest = json.loads((OUT / "cache_manifest.json").read_text())
    if file_hash(CACHE) != manifest["cache_sha256"]:
        raise ValueError("immutable Pg cache mismatch")
    with np.load(CACHE, allow_pickle=False) as z:
        cache = {k: torch.from_numpy(z[k].copy()) for k in ("pg", "source", "target", "mask", "lengths")}
        cache["records"] = json.loads(z["records_json"].tobytes())
    if matched_order(cache["records"]) != list(range(60)):
        raise ValueError("cache schedule mismatch")
    assert_file_pins(manifest["protected_input_sha256"])
    return cache, manifest


def optimizer(model, cfg):
    t = cfg["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        betas=tuple(t["betas"]),
        eps=t["epsilon"],
        weight_decay=t["weight_decay"],
    )


def setup_cuda():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU training fallback")


def preflight():
    cfg = config()
    setup_cuda()
    outcomes = {}
    for variant in ("S", "M", "L"):
        torch.manual_seed(cfg["seed"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = RecurrentLocalRefiner(variant, activation_checkpoint=True).cuda()
        op = optimizer(model, cfg)
        model_pin = state_hash(model)
        torch.manual_seed(27)
        pg = torch.randn(60, 500, 3, device="cuda").cumsum(1)
        target = pg + torch.randn_like(pg) * 0.2
        b = {"pg": pg, "source": pg, "target": target, "mask": torch.ones(60, 500, dtype=torch.bool, device="cuda")}
        eligible = int(freeze_chirality(pg, target, b["mask"])["eligible"].sum())
        start = time.perf_counter()
        try:
            result = model(pg, b["mask"])
            objective_components(result["prediction"], b, examples_total=60, quartets_total=eligible)[
                "total"
            ].backward()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(result["prediction"]).all()) and all(
                p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
            )
            reserved = torch.cuda.max_memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            outcomes[variant] = {
                "feasible": finite and reserved < cfg["preflight"]["max_reserved_fraction_total"] * total,
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": reserved,
                "latency_seconds": time.perf_counter() - start,
                "finite_outputs_gradients": finite,
                "optimizer_state_mutated": bool(op.state),
                "model_mutated": state_hash(model) != model_pin,
                "length": 500,
                "batch": 60,
                "K": 4,
                "device": torch.cuda.get_device_name(0),
            }
        except torch.cuda.OutOfMemoryError as error:
            outcomes[variant] = {
                "feasible": False,
                "reason": "CUDA OOM",
                "error": str(error),
                "length": 500,
                "batch": 60,
                "K": 4,
            }
        del model, op, pg, target, b
        if "result" in locals():
            del result
        torch.cuda.empty_cache()
        print("Preflight", variant, outcomes[variant], flush=True)
    write(
        OUT / "cuda_preflight.json",
        {"config_sha256": file_hash(CONFIG), "variants": outcomes, "compute_ownership_checked": True},
    )


def evaluator_module():
    spec = importlib.util.spec_from_file_location(
        "e010_v2_frozen_evaluator", ROOT / "scripts/diagnose_e010_phase4d_objective_v2.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(model, cache, manifest, boundary, mechanistic):
    # Historical evaluator reused only for per-protein metric reductions. Recurrence
    # final prediction/state slices are exposed by a temporary read-only adapter.
    mod = evaluator_module()
    panel = []
    device = next(model.parameters()).device
    for i, r in enumerate(cache["records"]):
        b = batch(cache, [i], device)
        panel.append({**r, **b, "frozen": freeze_chirality(b["pg"], b["target"], b["mask"])})
    per_state = []
    frame_ok = True
    step_max = 0.0
    original = model.training
    model.eval()
    with torch.no_grad():
        trajectories = [model(item["pg"], item["mask"]) for item in panel]
    for t in range(5):
        rows = []
        from protein_distance_diffusion.models.e010_hybrid_local import local_representation
        from protein_distance_diffusion.training.e010_phase4d import example_metrics, hybrid_losses
        from protein_distance_diffusion.training.e010_phase4d_diagnostic import aggregate_rows, signed_status
        from protein_distance_diffusion.training.e010_phase4d_objective_v2 import chiral_sum

        for item, tr in zip(panel, trajectories, strict=True):
            p = tr["states"][t]
            pg = item["pg"]
            y = item["target"]
            m = item["mask"]
            row = example_metrics(p, pg, item["source"], y, m)[0]
            raw = hybrid_losses(p, pg, item["source"], y, m)
            row.update({k: item[k] for k in ("sample_id", "condition", "stratum")})
            a, inv = signed_status(p, y, m)
            ba, bi = signed_status(pg, y, m)
            common = a & ba
            rep = local_representation(p, m)
            br = local_representation(pg, m)
            frame_ok &= bool(torch.equal(rep["eligible"], br["eligible"]))
            count = int(item["frozen"]["eligible"].sum())
            ch = float(chiral_sum(p, m, item["frozen"]))
            delta = tr["steps"][t - 1]["delta"] if t else torch.zeros_like(p)
            step = float(delta.norm(dim=-1)[m].max())
            step_max = max(step_max, step)
            row.update(
                local_mse={str(k): float(raw[f"local_{k}"]) for k in (1, 2, 3)},
                displacement_mean_square=float(raw["displacement"]),
                chirality_common_assessable=int(common.sum()),
                chirality_common_inversions=int((inv & common).sum()),
                chirality_common_baseline_inversions=int((bi & common).sum()),
                chirality_assessability_lost=int((ba & ~a).sum()),
                chirality_assessability_gained=int((a & ~ba).sum()),
                input_frame_eligible=int(br["eligible"].sum()),
                input_frame_degenerate=int(br["degenerate"].sum()),
                chiral_error_sum=ch,
                chiral_loss_eligible=count,
                step_correction_rms=float(delta[m].square().sum(-1).mean().sqrt()),
                step_correction_max=step,
            )
            rows.append(row)
        summary = aggregate_rows(rows)
        groups = [(summary["overall"], rows)]
        groups.extend(
            (summary["by_condition"][c], [r for r in rows if str(r["condition"]) == c]) for c in summary["by_condition"]
        )
        groups.extend((summary["by_stratum"][s], [r for r in rows if r["stratum"] == s]) for s in summary["by_stratum"])
        for s, rr in groups:
            count = sum(r["chiral_loss_eligible"] for r in rr)
            s["continuous_chiral_loss"] = sum(r["chiral_error_sum"] for r in rr) / count
            s["step_correction_rms"] = sum(r["step_correction_rms"] for r in rr) / len(rr)
            s["step_correction_max"] = max(r["step_correction_max"] for r in rr)
        per_state.append({"state": t, "metrics": summary, "per_example": rows})
    interaction = None
    if mechanistic:
        # v2 streaming evaluator calls recurrent final-state model directly.
        _, gs = mod.evaluate(model, panel)
        interaction = {**gradient_audit(gs), "per_module": {k: module_norms(model, g) for k, g in gs.items()}}
    model.train(original)
    return {
        "update": boundary,
        "states": per_state,
        "metrics": per_state[-1]["metrics"],
        "step_max": step_max,
        "frame_assessability_preserved": frame_ok,
        "gradient_interactions": interaction,
    }


def train_all(resume=False):
    cfg = config()
    setup_cuda()
    cache, manifest = load_cache()
    pre = json.loads((OUT / "cuda_preflight.json").read_text())
    if pre["config_sha256"] != file_hash(CONFIG):
        raise ValueError("preflight contract mismatch")
    arms = {}
    for variant in ("S", "M", "L"):
        arm_dir = OUT / variant
        arm_dir.mkdir(exist_ok=True)
        if not pre["variants"][variant]["feasible"]:
            arms[variant] = {"status": "resource_infeasible", "updates": 0}
            continue
        if (arm_dir / "result.json").exists():
            arms[variant] = json.loads((arm_dir / "result.json").read_text())
            continue
        torch.manual_seed(cfg["seed"])
        model = RecurrentLocalRefiner(variant, activation_checkpoint=True).cuda()
        op = optimizer(model, cfg)
        history = []
        cursor = 0
        gradient_norms = []
        clipped = 0
        start = time.perf_counter()
        prior_runtime = 0.0
        torch.cuda.reset_peak_memory_stats()
        if resume and (arm_dir / "latest.pt").exists():
            state = restore_checkpoint(arm_dir / "latest.pt", model, op, file_hash(CONFIG), manifest["cache_sha256"])
            cursor = state["successful_updates"]
            history = state["history"]
            stats = state.get("statistics", {})
            gradient_norms = stats.get("gradient_norms", [])
            clipped = stats.get("clipped", 0)
            prior_runtime = stats.get("elapsed_seconds", 0.0)
        if not history:
            initial = evaluate(model, cache, manifest, 0, True)
            history.append(initial)
            write(arm_dir / "boundary_000.json", initial)
        model.train()
        b = batch(cache, list(range(60)), "cuda")
        for update in range(cursor + 1, 501):
            op.zero_grad(set_to_none=True)
            out = model(b["pg"], b["mask"])
            loss = objective_components(out["prediction"], b, examples_total=60, quartets_total=manifest["quartets"])[
                "total"
            ]
            if not torch.isfinite(loss):
                raise RuntimeError("operational safety: nonfinite loss")
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["training"]["gradient_clip_norm"], error_if_nonfinite=True
            )
            gradient_norms.append(float(gn))
            clipped += int(gn > cfg["training"]["gradient_clip_norm"])
            if (
                not torch.isfinite(out["prediction"]).all()
                or float(out["delta"].norm(dim=-1)[b["mask"]].max()) > 0.160001
            ):
                raise RuntimeError("operational safety: output/bound failure")
            last_loss = float(loss.detach())
            op.step()
            del out, loss
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("operational safety: parameter failure")
            if update in cfg["training"]["boundaries"]:
                ev = evaluate(model, cache, manifest, update, update == 500)
                history.append(ev)
                write(arm_dir / f"boundary_{update:03d}.json", ev)
                save_checkpoint(
                    arm_dir / "latest.pt",
                    model,
                    op,
                    update,
                    file_hash(CONFIG),
                    manifest["cache_sha256"],
                    history,
                    statistics={
                        "gradient_norms": gradient_norms,
                        "clipped": clipped,
                        "elapsed_seconds": prior_runtime + time.perf_counter() - start,
                    },
                )
                print(
                    variant,
                    "boundary",
                    update,
                    "local",
                    ev["metrics"]["overall"]["mean_local_rmse"],
                    "chirality",
                    ev["metrics"]["overall"]["chirality_inversions"],
                    flush=True,
                )
            elif update % 10 == 0:
                print(variant, "update", update, "loss", last_loss, flush=True)
        final = history[-1]
        gate = gates(final, history[0])
        improvement = gate["local_improvement_fraction"]
        result = {
            "variant": variant,
            "parameters": COUNTS[variant],
            "updates": 500,
            "status": "completed",
            "gates": gate,
            "final_metrics": final["metrics"],
            "baseline_metrics": history[0]["metrics"],
            "correction_step_max": final["step_max"],
            "gradient_interactions_0": history[0]["gradient_interactions"],
            "gradient_interactions_500": final["gradient_interactions"],
            "gradient_norms": gradient_norms,
            "clipping_fraction": clipped / 500,
            "runtime_seconds": prior_runtime + time.perf_counter() - start,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "local_improvement_pct_per_million_parameters": 100 * improvement / (COUNTS[variant] / 1e6),
        }
        write(arm_dir / "result.json", result)
        arms[variant] = result
        del model, op, b
        torch.cuda.empty_cache()
    assert_file_pins(manifest["protected_input_sha256"])
    write(
        OUT / "capacity_result.json",
        {
            "arms": arms,
            **select_capacity(arms),
            "global_frozen": True,
            "held_out_launched": False,
            "config_sha256": file_hash(CONFIG),
            "cache_sha256": manifest["cache_sha256"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "train"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "preflight": preflight, "train": lambda: train_all(args.resume)}[args.mode]()
