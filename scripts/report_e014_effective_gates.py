"""Read-only checkpoint telemetry for the complete reliability/noise gate.

The main evaluator records the learned sigmoid gate and actual effective vector
norms. This inexpensive post-processing adds the exact mean/median of their
combined scalar multiplier, without rerunning the sparse/global trunk.
"""

# ruff: noqa: I001
from e014_bootstrap import ROOT
import json
import math
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import torch
import yaml
from protein_distance_diffusion.data.e014_population import CoordinateCache
from protein_distance_diffusion.evaluation.e014_evaluation import coordinate_batch
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import local_geometry, mlp
from protein_distance_diffusion.training.coordinate_diffusion import CoordinateVPDiffusion
from protein_distance_diffusion.training.e014_runtime import atomic_json, noise_seed, sha


@torch.no_grad()
def report(checkpoint, panel, label, cache, cfg, out, device="cuda"):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{label}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["checkpoint_sha256"] != sha(checkpoint):
            raise ValueError("effective-gate checkpoint provenance mismatch")
        return result
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    time_module, gate_module = mlp(64, 64, 64).to(device), mlp(65, 64, 1).to(device)
    for prefix, module in (("time", time_module), ("gate", gate_module)):
        module.load_state_dict(
            {k[len(prefix) + 1 :]: v for k, v in state["model"].items() if k.startswith(prefix + ".")}
        )
    del state
    diffusion = CoordinateVPDiffusion(cfg["diffusion_steps"])
    groups = defaultdict(list)
    for row in panel:
        x, _lengths, mask = coordinate_batch([row], cache, cfg["coordinate_scale_angstrom"], device=device)
        for timestep in cfg["noise_timesteps"]:
            t = torch.tensor([timestep], device=x.device)
            generator = torch.Generator(device=x.device).manual_seed(
                noise_seed("e014-paired-validation-v1", row["sample_id"], timestep)
            )
            batch = diffusion.make_training_batch(x, mask, timesteps=t, generator=generator)
            current = batch.noisy_coordinates[0]
            current = current - current.mean(0)
            _features, _frame, reliability = local_geometry(current)
            # Exact frozen E013 _single time/gate mathematics; no target feature.
            freq = torch.exp(-math.log(10000) * torch.arange(32, device=current.device, dtype=current.dtype) / 32)
            phase = t[0].to(current.dtype) * freq
            context = time_module(torch.cat([phase.sin(), phase.cos()]))
            learned = gate_module(torch.cat([context.expand(len(current), -1), reliability[:, None]], -1)).sigmoid()[
                :, 0
            ]
            envelope = (1 - t[0].to(current.dtype) / (cfg["diffusion_steps"] - 1)).clamp(0, 1).square()
            effective = learned * reliability * envelope
            values = dict(
                learned_gate_mean=float(learned.mean()),
                learned_gate_median=float(learned.median()),
                effective_gate_mean=float(effective.mean()),
                effective_gate_median=float(effective.median()),
                reliability_mean=float(reliability.mean()),
                noise_envelope=float(envelope),
            )
            for dimension, value in (
                ("timestep", timestep),
                ("length", row["length_stratum"]),
                ("length_timestep", f"{row['length_stratum']}|{timestep}"),
            ):
                groups[(dimension, value)].append(values)
    summary = []
    for (dimension, value), rows in groups.items():
        summary.append(
            dict(
                dimension=dimension,
                value=value,
                protein_condition_denominator=len(rows),
                **{key: sum(r[key] for r in rows) / len(rows) for key in rows[0]},
            )
        )
    result = dict(
        label=label,
        checkpoint_sha256=sha(checkpoint),
        groups=summary,
        training_modified=False,
        gate_definition="sigmoid(time,reliability) * reliability * (1-t/499)^2",
    )
    atomic_json(path, result)
    return result


def main():
    cfg = yaml.safe_load(open("configs/e014_unconditional_coordinate_v_scaleup_v1.yaml"))
    torch.set_num_threads(1)
    run = ROOT / "e014"
    final = json.loads((run / "FINAL_REPORT.json").read_text())
    if final["completed_epochs"] < 2:
        raise ValueError("post-training gate supplement requires minimum exposure completion")
    cache = CoordinateCache(ROOT)
    dev = pq.read_table(ROOT / "population/development.parquet", use_threads=False).to_pylist()
    prospective = pq.read_table(ROOT / "population/prospective.parquet", use_threads=False).to_pylist()
    outputs = []
    checkpoints = run / "checkpoints"
    outputs.append(
        report(
            checkpoints / "initialization.pt", dev, "initialization_development", cache, cfg, run / "effective_gates"
        )
    )
    for curve in final["curves"]:
        if curve.get("checkpoint"):
            checkpoint = curve["checkpoint"]
        elif curve.get("resumed_pending_boundary"):
            checkpoint = checkpoints / (curve["resumed_pending_boundary"] + ".pt")
        else:
            continue
        label = Path(checkpoint).stem
        outputs.append(report(checkpoint, dev, label + "_development", cache, cfg, run / "effective_gates"))
    final_checkpoint = checkpoints / f"epoch_{final['completed_epochs']}_fraction_1.00.pt"
    outputs.append(report(final_checkpoint, prospective, "final_prospective", cache, cfg, run / "effective_gates"))
    atomic_json(run / "EFFECTIVE_GATE_TELEMETRY.json", dict(boundaries=outputs, training_modified=False))


if __name__ == "__main__":
    main()
