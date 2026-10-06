"""Fixed-state preflight comparison; no historical or new optimization."""

import json

import numpy as np
import torch

from scripts.recover_e010_conditioning_v9b import synthetic
from scripts.run_e010_direct_correction_v11 import OUT, frozen, old
from scripts.run_e010_phase4d_recurrent_capacity_v3 import write


def main():
    frozen()
    _, physical, limits = old.setup()
    historical = json.loads((old.V9OUT / "synthetic_preflight.json").read_text())
    comparison = []
    for n in [12, 32]:
        prior = next(r["optimizer"] for r in historical["records"] if r["length"] == n)
        with np.load(old.V9B / f"recovered_length_{n}.npz") as saved:
            cert = old.v10.certificate(
                synthetic(n), torch.from_numpy(saved["delta"].copy()), saved["multipliers"], physical, limits
            )
        direct = json.loads((OUT / "preflight" / f"length_{n}.json").read_text())["optimizer"]

        def record(log, certificate):
            return dict(
                iterations=log["iterations"],
                raw_optimality=log["optimality"],
                physical_converged=certificate["converged"],
                gates=certificate["gates"],
                normalized_stationarity=certificate["physical"]["physical_normalized_stationarity_l2"],
                primal=certificate["physical"]["normalized_primal"],
                complementarity=certificate["physical"]["normalized_complementarity"],
                dual_negativity=certificate["physical"]["normalized_dual"],
                projected_gradient=certificate["projected_feasible_gradient_norm"],
                shadow_steps=certificate["shadow_steps"],
            )

        comparison.append(
            dict(
                length=n,
                historical_radial=record(prior, cert),
                direct_v11=record(direct, direct["physical_certificate"]),
            )
        )
    write(
        OUT / "preflight_comparison.json",
        dict(
            cases=comparison,
            historical_optimization_rerun=False,
            direct_maximum_iterations=1000,
            historical_maximum_iterations=2000,
            settings_tuned=False,
            valid_implementation=True,
        ),
    )


if __name__ == "__main__":
    main()
