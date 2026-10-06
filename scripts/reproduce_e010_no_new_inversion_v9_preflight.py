"""Reproduce the failed synthetic preflight only; never optimize a panel input."""

import json

import torch

from protein_distance_diffusion.training import e010_no_new_inversion_v9 as v9
from scripts.run_e010_no_new_inversion_v9 import OUT, setup
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write

cfg = setup()
original = json.loads((OUT / "synthetic_preflight.json").read_text())
old = json.loads((OUT / "preflight_cap_1000/synthetic_preflight.json").read_text())


def strip(log):
    x = json.loads(json.dumps(log))
    x.pop("runtime_seconds")
    return x


assert strip(original["records"][0]["optimizer"]) == strip(old["records"][0]["optimizer"])
assert (
    original["records"][1]["optimizer"]["history"][: len(old["records"][1]["optimizer"]["history"])]
    == old["records"][1]["optimizer"]["history"]
)
n = 32
t = torch.arange(n, dtype=torch.float64)
p = torch.stack([2 * t, t.sin(), t.cos()], -1)[None]
y = p + torch.stack([0.7 * torch.sin(1.7 * t), 0.8 * torch.cos(1.2 * t), 0.6 * torch.sin(2.3 * t)], -1)[None]
b = dict(pg=p, target=y, source=p + 0.1, mask=torch.ones(1, n, dtype=torch.bool))
tr, log, z = v9.solve(b, cfg)
assert strip(log) == strip(original["records"][1]["optimizer"]), "Synthetic failure reproduction differs"
write(
    OUT / "preflight_reproduction.json",
    dict(
        exact_non_timing_32_match=True,
        exact_12_match_across_caps=True,
        exact_1000_history_prefix=True,
        scientific_panel_launched=False,
        cuda_used=False,
        neural_training_launched=False,
    ),
)
print("Exact failed-preflight reproduction passed", flush=True)
