"""Frozen synthetic expanded-inequality preflight before any V9 panel result."""

import torch

from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from scripts.run_e010_no_new_inversion_v9 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write

cfg = setup()
records = []
for n in (12, 32):
    t = torch.arange(n, dtype=torch.float64)
    p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
    y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
    b = dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))
    tr, log, z = v9.solve(b, cfg)
    records.append(dict(length=n, optimizer=log))
    print(n, log["scipy_success"], log["converged"], log["iterations"], log["quartets"], flush=True)
passed = all(r["optimizer"]["converged"] for r in records)
write(
    OUT / "synthetic_preflight.json", dict(passed=passed, records=records, config=cfg, scientific_panel_optimized=False)
)
assert passed, "Numerical preflight blocks panel; no scientific optimization permitted"
