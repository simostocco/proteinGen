"""Recover missing synthetic states through unchanged V9 telemetry replay only."""

import hashlib
import json

import numpy as np
import torch

from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from scripts.run_e010_no_new_inversion_v9 import OUT as OLD
from scripts.run_e010_no_new_inversion_v9 import setup

OUT = OLD.parent / "strict_constraint_conditioning_v9b"


def synthetic(n):
    t = torch.arange(n, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
    return dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


if __name__ == "__main__":
    cfg = setup()
    original = v9.v7.constrained_minimize
    captured = {}

    def instrument(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["result"] = result
        return result

    v9.v7.constrained_minimize = instrument
    historical = json.loads((OLD / "synthetic_preflight.json").read_text())
    manifest = []
    for row in historical["records"]:
        n = row["length"]
        b = synthetic(n)
        tr, log, z = v9.solve(b, cfg)
        expected = dict(row["optimizer"])
        expected.pop("runtime_seconds")
        actual = json.loads(json.dumps(log))
        actual.pop("runtime_seconds")
        (OUT / f"recovery_evidence_length_{n}.json").write_text(json.dumps(actual, indent=2) + "\n")
        assert actual == expected, f"STOP: historical length-{n} evidence mismatch"
        result = captured["result"]
        arrays = dict(
            z=z,
            delta=torch.stack([s["delta"].detach() for s in tr["steps"]]).numpy(),
            prediction=tr["prediction"].detach().numpy(),
            multipliers=np.asarray(log["multipliers"]),
            lagrangian_grad=result.lagrangian_grad,
        )
        path = OUT / f"recovered_length_{n}.npz"
        np.savez(path, **arrays)
        telemetry = {
            k: float(result[k])
            for k in [
                "optimality",
                "constr_violation",
                "barrier_parameter",
                "barrier_tolerance",
                "tr_radius",
                "cg_stop_cond",
            ]
        }
        record = dict(
            length=n,
            exact_historical_non_timing_log=True,
            coordinate_sha256=digest(arrays["prediction"]),
            array_sha256={k: digest(a) for k, a in arrays.items()},
            input_sha256={k: digest(a.numpy()) for k, a in b.items()},
            solver=telemetry,
            historical_coordinate_hash_available=False,
            state_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        manifest.append(record)
        (OUT / "recovery.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print("Recovered exact historical evidence", n, telemetry, flush=True)
