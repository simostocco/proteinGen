# V14 SLSQP resource-stop handoff

**SQP-E — solver resource/scaling failure. No scientific interpretation.**
The 60-example scientific panel was not launched. K=8/s_max=0.04 is not certified,
and trust-constr's barrier is not established as the primary blocker.

Python 3.12.14; installed SciPy 1.18.1; deterministic CPU float64, one arithmetic
thread. SLSQP supports analytic vector-valued constraints and returned inequality
multipliers. The frozen settings are maxiter=2000, ftol=1e-12, zero initialization,
with no additional constraint scaling. The solver uses internal dense BFGS/QP
machinery rather than historical exact Hessian-vector products. All scientific
constraints, geometry, cached panel inputs and V9B physical gates remain unchanged.

## Preflights

| Quantity | Length 12 | Length 32 |
|---|---:|---:|
| SLSQP iterations | 144 | 313 |
| SLSQP success/status | true / 0 | true / 0 |
| Wall seconds | 2.4265 | 14.2471 |
| Process peak RSS GiB | 0.6819 | 0.7099 |
| Function / Jacobian evaluations | 146 / 144 | 314 / 313 |
| Physical certificate | FAIL | FAIL |
| Normalized physical stationarity | 1.75756e-8 | 4.19589e-8 |
| Normalized complementarity | 1.08597e-16 | 3.35177e-17 |
| Dual negativity | 0 | 0 |
| Projected feasible-gradient norm | 1.75756e-8 | 4.19589e-8 |
| Material feasible shadow descent | none | none |
| Active balls / signed constraints | 80 / 0 | 224 / 1 |
| Baseline → final inversions | 4 → 4 | 11 → 11 |
| New / repaired inversions | 0 / 0 | 0 / 0 |
| Assessability | 9 → 9 | 29 → 29 |

Both fail only the frozen physical primal gate: maximum corrections are
0.04000000000000047 and 0.04000000000000027 Å. Ball inequality minima are
-2.37588e-14 and -1.37668e-14. These tiny numerical overshoots were neither clipped
nor accepted under a relaxed bound. Scientific signed/assessability and continuous
constraints remain satisfied. Solver success is not substituted for certification.

Scientific multiplier L2 differences between solver reports and independently
reconstructed physical multipliers are 0 and 7.16981e-10 respectively. Physical
reconstruction uses the unchanged historical active-system and NNLS machinery;
certification does not depend on SLSQP's multipliers alone.

For context, saved V11 direct trust-constr controls took 20 and 1000 iterations,
2.3776 and 70.7751 seconds, with physical certificates true and false. Their
normalized physical stationarity was 6.15151e-8 and 0.00452158. This is a descriptive
comparison with historical controls, not a full-panel solver superiority claim.

## Longest-length resource smoke

The unchanged deterministic historical synthetic(500) has 12,000 variables and
5,495 inequalities. Exact installed SLSQP mandatory allocation:

- Float64 workspace: 11,378,363,864 bytes (10.596927 GiB).
- Dense constraint normals: 527,520,000 bytes (0.491291 GiB).
- Combined mandatory arrays: 11,905,883,864 bytes (11.088218 GiB).
- Available RAM at the resource gate: 11.069851 GiB.
- Frozen safety reserve: 25% of available RAM, at least 2 GiB; no swap allowance.
- Remaining allocation budget: 8.302388 GiB.

The mandatory arrays alone exceed all available RAM before additional dense
Jacobians, solver arrays, PyTorch and OS overhead. The preregistered gate refused
the dense allocation. There was no length-500 optimizer invocation or allocation
attempt. Geometry, vector constraints and historical float64 directional
derivatives were validated at full length and K. This plumbing smoke took 1.4606
seconds and process peak RSS was 0.9543 GiB; that is not a measured full-SLSQP peak.
No sequence length or K was reduced.

## Full-panel fields

V14 physical convergence, material-shadow counts, iteration distribution,
per-example runtime/memory, condition 50/250/450 gains, i+1/i+2/i+3 gains,
Cartesian/chirality changes, inversion counts, assessability, active constraints,
physical residual distributions and P0–P8 metrics are **not evaluated** for the
scientific panel. There were zero scientific-panel solver invocations.
Historical V13 reference remains 6/60 physical passes, 51/60 material-shadow cases,
and gains 19.1694%, 8.3859%, 8.5007%; these are not V14 results.

Complete synthetic P0–P8 metrics, correction paths, saturation and consecutive
cosines are recorded in `preflight_fixed_state_metrics.json` and explicitly
labelled as synthetic rather than scientific-panel examples.

## Verification

31 focused CPU tests passed; Ruff passed. All 60 historical baseline/derivative
records matched exactly, including 13,029 assessable quartets. Private saved
preflight states and physical certificates independently reproduced exactly.
Fresh-process aggregation reproduced publication files byte-for-byte while
optimizer functions were forbidden. Protected input/source/history hashes remain
intact. Completed variables are saved before certificate telemetry; regression
tests cover failed solves, missing optional fields and empty active sets.

No environment changes, CUDA, neural training, scientific outcome-based settings
changes, warm starts, real-example optimization, or follow-on solver occurred.
V11/V12/V13 and older historical artifacts are unchanged; private arrays are
excluded from Git.

Exactly one recommended next experiment: preregister a sparse active-set/SQP
benchmark with feasibility-preserving closed-ball updates, retaining the same
strict feasible set and frozen physical convergence contract.
