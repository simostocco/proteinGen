"""Synthetic-only K=8 numerical-capacity preflight, before panel optimization."""

import json

import torch

from protein_distance_diffusion.training import e010_local_feasibility_v8 as v8
from scripts.run_e010_local_feasibility_v8 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write

cfg = setup()
records = []
for n in (12, 128):
    t = torch.arange(n, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
    b = dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))
    for arm in ("A", "B"):
        tr, log, z = v8.solve(b, arm, cfg)
        assert all(torch.isfinite(s).all() for s in tr["states"])
        assert all(s["delta"].norm(dim=-1).max() <= 0.04 for s in tr["steps"])
        records.append(dict(length=n, arm=arm, optimizer=log))
        print(n, arm, log["scipy_success"], log["converged"], log["iterations"], flush=True)
passed = all(
    r["optimizer"]["scipy_success"] and r["optimizer"]["iterations"] < cfg["solver"]["maxiter"] for r in records
)
write(
    OUT / "synthetic_preflight.json", dict(passed=passed, records=records, config=cfg, scientific_panel_optimized=False)
)
print(json.dumps(dict(passed=passed)))
assert passed, "Synthetic preflight blocks panel; do not change settings using panel outcomes"
