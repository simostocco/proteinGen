import json
import numpy as np
import torch

from protein_distance_diffusion.training.e010_phase4d_diagnostic import file_hash
from scripts import run_e010_slsqp_v14 as r

r.frozen()
records = []
for n in [12, 32]:
    b = r.synthetic(n)
    expected = json.loads((r.OUT / "preflight" / f"length_{n}.json").read_text())
    path = r.OUT / "untracked_states" / f"preflight_{n}_certified.npz"
    assert file_hash(path) == expected["state_file_sha256"]
    with np.load(path) as saved:
        tr = r.sqp.direct.trajectory(b["pg"], b["mask"], torch.from_numpy(saved["z"].copy()).reshape(8, *b["pg"].shape))
        assert np.array_equal(torch.stack(tr["states"]).detach().numpy(), saved["states"])
    states = [r.sqp.direct.v9.v8.metric_row(p, b, dict(synthetic_length=n, panel_example=False, state=t), tr, t)
              for t, p in enumerate(tr["states"])]
    repeated = [r.sqp.direct.v9.v8.metric_row(p, b, dict(synthetic_length=n, panel_example=False, state=t), tr, t)
                for t, p in enumerate(tr["states"])]
    assert states == repeated
    row = dict(length=n, synthetic_only=True, baseline=states[0], final=states[-1], states=states,
               correction_direction_telemetry=r.sqp.direct.v9.v8.correction_telemetry(tr),
               metrics_reproduced_exactly=True, optimizer_invocations=0)
    records.append(row)
result = dict(records=records, scientific_panel_launched=False, optimizer_invocations=0,
              metric_units=dict(coordinates="Angstrom", RMSE="Angstrom", distance_MSE="Angstrom squared",
                                continuous_chirality="dimensionless"))
(r.OUT / "preflight_fixed_state_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
print([(x["length"], x["baseline"]["chirality_inversions"], x["final"]["chirality_inversions"],
        x["final"]["step_correction_max"], x["final"]["chirality_assessable"]) for x in records])
