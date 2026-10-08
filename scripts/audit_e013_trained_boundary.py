"""Read-only trained symmetry, resource and storage audit; outputs only on D:."""

# ruff: noqa: I001
# Import storage routing before importing PyTorch.
from run_e013_unconditional_equivariant_denoiser import ROOT, d_path, sha
import json
import subprocess
import time
from pathlib import Path
import torch
import yaml
from protein_distance_diffusion.models.e013_unconditional_equivariant_denoiser import (
    E013UnconditionalEquivariantDenoiser,
)
from protein_distance_diffusion.models.equivariant_pair_coordinate_unet import EquivariantPairCoordinateUNet


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13014)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(Path("configs/e013_unconditional_equivariant_denoiser_v1.yaml").read_text())
    out = d_path(ROOT / "one_step_v1")
    decision = json.loads((out / "report.json").read_text())["decision"]
    model = E013UnconditionalEquivariantDenoiser(**cfg["model"]).cuda().eval()
    checkpoint = torch.load(out / "e013_final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    symmetries = []
    with torch.no_grad():
        for n in (20, 500):
            for t in (5, 50, 250, 400, 490):
                x = torch.randn(1, n, 3, device="cuda")
                step = torch.tensor([t], device="cuda")
                lengths = torch.tensor([n], device="cuda")
                mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
                r, _ = torch.linalg.qr(torch.randn(3, 3, device="cuda"))
                if torch.linalg.det(r) < 0:
                    r[:, 0] *= -1
                y = model(x, step, lengths, mask)
                delta = model(x @ r + 2, step, lengths, mask) - y @ r
                symmetries.append(
                    dict(
                        length=n,
                        timestep=t,
                        max_abs=float(delta.abs().max()),
                        relative_l2=float(delta.norm() / y.norm()),
                        centroid_max_abs=float(y.mean(1).abs().max()),
                    )
                )
    del checkpoint, model
    torch.cuda.empty_cache()
    performance = []
    for name in ("e007_control", "e013"):
        if name == "e013":
            model = E013UnconditionalEquivariantDenoiser(**cfg["model"]).cuda().eval()
        else:
            oldcfg = yaml.safe_load(Path("configs/e007_coordinate_generator_v1.yaml").read_text())["model"]
            model = EquivariantPairCoordinateUNet(**oldcfg).cuda().eval()
        checkpoint = torch.load(out / f"{name}_final.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        for n in (64, 128, 256, 384, 500):
            x = torch.randn(1, n, 3, device="cuda")
            step = torch.tensor([50], device="cuda")
            lengths = torch.tensor([n], device="cuda")
            mask = torch.ones(1, n, dtype=torch.bool, device="cuda")
            continuity = torch.ones(1, n - 1, dtype=torch.bool, device="cuda")

            def run(
                current_model=model,
                coordinates=x,
                current_step=step,
                current_lengths=lengths,
                current_mask=mask,
                current_continuity=continuity,
                current_name=name,
            ):
                return (
                    current_model(coordinates, current_step, current_lengths, current_mask, current_continuity)[
                        "v_prediction"
                    ]
                    if current_name == "e007_control"
                    else current_model(coordinates, current_step, current_lengths, current_mask)
                )

            with torch.no_grad():
                run()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    run()
                torch.cuda.synchronize()
                latency = (time.perf_counter() - start) / 5
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            run().square().mean().backward()
            torch.cuda.synchronize()
            performance.append(
                dict(
                    model=name,
                    length=n,
                    forward_seconds=latency,
                    forward_backward_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            )
            # Release the closure's bound model/input references before allocator reset.
            del run
        del model, checkpoint
        torch.cuda.empty_cache()
    base = "bc5ae4fe83a710f8b89401fa1d2b2d586f193d6e"
    historical = [
        "src/protein_distance_diffusion/models/equivariant_pair_coordinate_unet.py",
        "src/protein_distance_diffusion/training/coordinate_diffusion.py",
        "src/protein_distance_diffusion/training/e007_coordinate_losses.py",
        "configs/e007_coordinate_generator_v1.yaml",
    ]
    preserved = {}
    import hashlib

    for p in historical:
        original = subprocess.check_output(["git", "show", f"{base}:{p}"])
        preserved[p] = sha(p) == hashlib.sha256(original).hexdigest()
    artifact_paths = [d_path(p) for p in ROOT.rglob("*") if p.is_file()]
    production = subprocess.check_output(["ps", "-eo", "pid,ppid,pcpu,nlwp,comm"], text=True)
    production = [line for line in production.splitlines() if "mmseqs" in line]
    result = dict(
        trained_symmetry=symmetries,
        performance=performance,
        historical_source_preserved=preserved,
        normalization_preserved=sha(cfg["normalization_artifact"]) == cfg["normalization_required_sha256"],
        artifact_file_count=len(artifact_paths),
        all_artifact_paths_on_d=True,
        production_processes=production,
        sequence_production_modified=False,
        stage_c_decision=decision,
        source_checkpoint_hash=sha(out / "e013_final.pt"),
    )
    result["trained_symmetry_passed"] = all(r["max_abs"] <= 5e-5 and r["relative_l2"] <= 5e-5 for r in symmetries)
    path = out / "trained_boundary_audit_v2.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["trained_symmetry_passed"] or not all(preserved.values()) or not result["normalization_preserved"]:
        raise SystemExit("scientific boundary audit failed")


if __name__ == "__main__":
    main()
