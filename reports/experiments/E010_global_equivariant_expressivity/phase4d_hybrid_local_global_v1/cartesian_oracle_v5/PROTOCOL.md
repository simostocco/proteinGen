# Cartesian constrained-oracle v5

Historical v4 result `f9b42c36459f17c9f7fb42c20496b7c7b58a4ea6` stays immutable.
The fixed protocol is in
[the v5 document](../../../../../docs/e010_phase4d_cartesian_oracle_v5.md) and
[config](../../../../../configs/e010_phase4d_cartesian_oracle_v5.yaml).

Only the free correction basis changes. K=4, s_max=.04 Å, beta=16.8, gamma=2,
original panel/targets/Pg/masks/eligibility/chirality and objective reductions
remain fixed. The historical v4 solver code object is reused unchanged in an
isolated globals copy with only the trajectory binding replaced. No global
monkeypatch, neural parameters, CUDA, optimizer fallback or coefficient sweep.

Focused CPU tests: **92 passed**, including 12 v5 cases. V4 optimizer/config,
solver/convergence code and metric functions have exact parity. Bound, geometric
eligibility, zero initialization, no global mutation, deterministic reproduction,
paired counts/classification and protected hash checks passed. Ruff checks pass.
The previous project-wide 134 inherited failures are documented separately;
the complete suite is not rerun for this isolated diagnostic.

`execution_contract.json` freezes implementation, config, protocol document and
historical records before panel execution. Strong convergence is predeclared as
>=48/60 overall and >=18/20 at condition 450. Otherwise CART-O5.
Run the fixed 60 examples with four one-thread CPU workers; then repeat all 60
from zero and compare every non-timing record and coordinate hash exactly.
No commit/push until tests and full reproduction succeed.

Every example records baseline/final metrics, P0–P4 eligibility and temporary
assessability, initial/final objective, history, iterations and projected residual.
Paired comparisons cover both-converged and newly-converged examples. Datasets
and coordinate checkpoints are never committed. No follow-on experiment runs.
