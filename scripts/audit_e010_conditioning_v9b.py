"""Reproducible fixed-state V9B audit. No scientific data or coordinate optimizer."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from protein_distance_diffusion.training import e010_conditioning_v9b as audit
from scripts.recover_e010_conditioning_v9b import OUT, digest, synthetic
from scripts.run_e010_no_new_inversion_v9 import setup

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e010_phase4d_conditioning_v9b.yaml"


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def freeze():
    paths = subprocess.check_output(["git", "ls-files"], text=True, cwd=ROOT).splitlines()
    pins = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    write(
        OUT / "preregistration.json",
        dict(
            config=yaml.safe_load(CONFIG.read_text()),
            config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            historical_tracked_sha256=pins,
            additional_intermediate_size_case=(
                "None previously available under V9 strict feasible set; no new selection"
            ),
            recovery_scope=(
                "Unchanged synthetic telemetry replay; all historical non-timing evidence must match; "
                "historical coordinate hashes were not stored"
            ),
        ),
    )


def check_pins():
    for source in json.loads((OUT / "installed_solver_sources.json").read_text()):
        assert hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() == source["sha256"]
    p = json.loads((OUT / "preregistration.json").read_text())
    assert p["config_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    for file, h in p["historical_tracked_sha256"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == h, file


def summaries(arr):
    arr = np.asarray(arr)
    return dict(
        minimum=float(arr.min()),
        median=float(np.median(arr)),
        maximum=float(arr.max()),
        p95=float(np.quantile(arr, 0.95)),
    )


def run():
    setup()
    check_pins()
    cfg = yaml.safe_load(CONFIG.read_text())
    recovery = json.loads((OUT / "recovery.json").read_text())
    assert [r["length"] for r in recovery] == cfg["cases"], "Recovery incomplete; STOP"
    cases = []
    for record in recovery:
        n = record["length"]
        b = synthetic(n)
        path = OUT / f"recovered_length_{n}.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["state_file_sha256"]
        data = np.load(path)
        for k, h in record["array_sha256"].items():
            assert digest(data[k]) == h
        for k, h in record["input_sha256"].items():
            assert digest(b[k].numpy()) == h
        delta = torch.from_numpy(data["delta"].copy())
        d, f, c, tr, o, g, j = audit.physical_jacobians(b, delta)
        assert digest(tr["prediction"].detach().numpy()) == record["coordinate_sha256"]
        cv = c.detach().numpy()
        mu = data["multipliers"]
        eligible = tr["eligible"].numpy()
        a, ids, balls = audit.active_system(
            data["delta"], eligible, cv, j, cfg["active_normalized_slack"], cfg["boundary_relative_tolerance"]
        )
        direction, multipliers = audit.cone_projection(g * 0.04, a)
        # Active-only constrained multiplier reconstruction, independent of SciPy's barrier multipliers.
        reconstructed = np.zeros_like(mu)
        reconstructed[ids] = multipliers[: len(ids)]
        original = audit.components(data["delta"], eligible, g, j, cv, mu)
        fitted = audit.components(data["delta"], eligible, g, j, cv, reconstructed)
        base = float(o.baseline["local"])
        for result in (original, fitted):
            result["raw_stationarity_max_angstrom"] = float(
                np.linalg.norm(result.pop("residual") * base, axis=-1).max()
            )
            result["ball_multipliers"] = result["ball_multipliers"].tolist()
            result["raw_complementarity_angstrom_squared"] = result["normalized_complementarity"] * base
        # Verify exact radial-chain reconstruction of solver-space Lagrangian gradient.
        z = torch.from_numpy(data["z"].copy()).requires_grad_()
        point = o.point(z.detach().numpy())
        gz = (
            torch.autograd.grad(point.objective + (point.constraints * torch.from_numpy(mu)).sum(), point.z)[0]
            .detach()
            .numpy()
        )
        assert np.allclose(gz, data["lagrangian_grad"], rtol=1e-5, atol=1e-14)
        rad = audit.radial((0.04 * data["z"]).reshape(delta.shape))
        rad_rows = []
        for t in range(8):
            valid = eligible[t]
            rad_rows.append(dict(step=t, **{k: summaries(v[t][valid]) for k, v in rad.items()}))
        # Include radial-map effects by multiplying physical active-system columns by exact D_delta/D_z.
        v = (0.04 * data["z"]).reshape(-1, 3)
        av = rad["tangential"].reshape(-1)
        blocks = 0.04 * (
            av[:, None, None] * np.eye(3)[None] - av[:, None, None] ** 3 * v[:, :, None] * v[:, None, :] / 0.04**2
        )
        mapped = np.einsum("rni,nij->rnj", a.reshape(len(a), -1, 3) / 0.04, blocks).reshape(a.shape)
        normalized_rows = a / np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-300)
        normdir = direction.reshape(delta.shape).copy()
        maxnorm = np.linalg.norm(normdir, axis=-1).max()
        shadows = []
        if maxnorm:
            normdir /= maxnorm
        for step in cfg["shadow_steps_angstrom"]:
            candidate = audit.project_balls(data["delta"] + step * normdir)
            ff, cc, tt, oo = audit.physical(b, torch.from_numpy(candidate))
            tele = oo.quartets.telemetry(tt["prediction"], 1e-5)
            constraints = cc.detach().numpy()
            feasible = bool(
                max(constraints[:2]) <= 1e-8
                and tele["exact_extra_feasible"]
                and not tele["new_inversions"]
                and not tele["assessability_lost"]
            )
            improvement = float(f.detach() - ff.detach())
            shadows.append(
                dict(
                    step_angstrom=step,
                    normalized_loss_decrease=improvement,
                    raw_loss_decrease_angstrom_squared=improvement * base,
                    feasible=feasible,
                    material=feasible and improvement >= cfg["material_normalized_local_decrease"],
                    maximum_constraint=float(constraints.max()),
                    quartets=tele,
                )
            )
        checks = audit.finite_checks(
            b, delta, g, j, mu, ids, cfg["finite_difference_eps_angstrom"], cfg["finite_difference_directions"]
        )
        assert all(
            any(r["passed"] for r in checks if r["phase"] == phase) for phase in cfg["finite_difference_directions"]
        ), "Derivative defect: STOP"
        # Row rescaling transforms multipliers oppositely; unchanged physical stationarity.
        rownorm = np.linalg.norm(j * 0.04, axis=1)
        rescaled_j = j / np.maximum(rownorm[:, None], 1e-300)
        rescaled_mu = mu * rownorm
        scaling_error = float(np.max(np.abs(j.T @ mu - rescaled_j.T @ rescaled_mu)))
        groups = {name: list(range(lo + 2, hi + 2)) for name, (lo, hi) in o.quartets.slices.items()}
        groups.update(aligned_rmsd=[0], continuous_chirality=[1])
        raw_scales = np.r_[
            float(1.01 * o.baseline["aligned"]),
            float(o.baseline["chiral"]),
            o.quartets.q0[o.quartets.correct].abs().numpy(),
            o.quartets.q0[o.quartets.inverted].square().numpy(),
            o.quartets.turn0[o.quartets.assess].numpy(),
            o.quartets.bonds0.reshape(-1).numpy(),
        ]
        groupstats = {
            name: dict(
                active=sum(i in ids for i in indices),
                multiplier=summaries(mu[indices]),
                raw_multiplier=summaries(base * mu[indices] / raw_scales[indices]),
                raw_primal_violation=float(np.maximum(cv[indices] * raw_scales[indices], 0).max()),
                gradient_row_norm_per_angstrom=summaries(np.linalg.norm(j[indices], axis=1)),
                lagrangian_contribution_norm_per_angstrom=float(np.linalg.norm(j[indices].T @ mu[indices])),
            )
            for name, indices in groups.items()
            if indices
        }
        output = dict(
            length=n,
            baseline_local_MSE_angstrom_squared=base,
            normalized_final_local=float(f.detach()),
            solver=record["solver"],
            physical_solver_multipliers=original,
            physical_reconstructed_multipliers=fitted,
            active_constraint_indices=ids.tolist(),
            active_correction_balls=len(balls),
            constraint_groups=groupstats,
            normalized_constraint_values=cv.tolist(),
            normalized_solver_multipliers=mu.tolist(),
            reconstructed_active_multipliers=multipliers.tolist(),
            physical_active_spectrum=audit.spectrum(a),
            row_normalized_active_spectrum=audit.spectrum(normalized_rows),
            optimizer_active_spectrum=audit.spectrum(mapped),
            non_ball_active_spectrum=audit.spectrum(j[ids] * 0.04),
            radial_by_step=rad_rows,
            radial_variables={k: v.tolist() for k, v in rad.items()},
            solver_gradient_reconstruction_max_error=float(np.max(np.abs(gz - data["lagrangian_grad"]))),
            optimizer_gradient_norm=float(np.linalg.norm(gz)),
            physical_local_gradient_norm_per_angstrom=float(np.linalg.norm(g)),
            raw_local_gradient_norm_angstrom=float(base * np.linalg.norm(g)),
            projected_gradient_norm_normalized=float(np.linalg.norm(direction)),
            maximum_linearized_cone_violation=float(np.max(a @ direction)),
            scaling_stationarity_max_error=scaling_error,
            finite_difference_checks=checks,
            shadow_steps=shadows,
            coordinate_sha256=record["coordinate_sha256"],
            quartets=o.quartets.telemetry(tr["prediction"], 1e-5),
        )
        cases.append(output)
        print(
            n,
            "stationarity",
            original["physical_normalized_stationarity_max"],
            "reconstructed",
            fitted["physical_normalized_stationarity_max"],
            "cone",
            output["projected_gradient_norm_normalized"],
            "shadows",
            [(s["feasible"], s["normalized_loss_decrease"]) for s in shadows],
            flush=True,
        )
    check_pins()
    result = dict(cases=cases, scientific_panel_launched=False, cuda_used=False, neural_training_launched=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "audit", "reproduce"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "audit":
        write(OUT / "audit.json", run())
    else:
        result = json.loads(json.dumps(run()))
        assert result == json.loads((OUT / "audit.json").read_text())
        write(OUT / "reproduction.json", dict(exact_audit_reproduction=True))
        print("Exact audit reproduction passed")
