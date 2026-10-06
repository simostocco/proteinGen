# V13 matched-budget adjudication

**MATCH-C — little numerical convergence improvement despite doubling the budget.**
The complete historical feasibility target remains uncertified. These are safe
recorded endpoints, not 60 certified constrained optima.

Only `maxiter` changed from 1000 to 2000. The original V11 implementation,
CPU float64 arithmetic, exact Hessian-vector products, zero initialization,
K=8, s_max=0.04 Å, direct Cartesian variables, objective, strict constraints,
solver settings, V9B physical contract and shadow protocol were reused unchanged.
All 60 historical examples ran exactly once. No outcome-based reruns occurred.

## Matched numerical comparison

| Quantity | V11 / 1000 cap | V13 / 2000 cap |
|---|---:|---:|
| Physical convergence | 4/60 | 6/60 |
| Material feasible shadow descent | 55/60 | 51/60 |
| Iteration-cap endpoints | 56/60 | 54/60 |
| Median normalized physical stationarity | 0.00485193 | 0.00437807 |
| Maximum normalized physical stationarity | 0.01381235 | 0.01327486 |
| Median normalized complementarity | 6.93859e-7 | 2.96948e-7 |
| Maximum normalized complementarity | 2.30465e-5 | 2.59548e-5 |
| Median barrier parameter | 1e-7 | 2e-8 |
| Median trust radius (optimizer units) | 0.0601655 | 0.0228262 |

The two new physical passes are examples 12 and 14 (2mdw_A, conditions 50 and
450), at iterations 1320 and 1170. All four original converged controls reproduced
their original endpoints and stopping iterations exactly. All 60 trajectories
matched their recorded V11 history and physical-screen prefixes exactly.

V13 physical stationarity spans 4.32014e-9–0.01327486; complementarity spans
9.53251e-10–2.59548e-5; raw trust-constr optimality spans
6.53987e-10–6.46982e-6. Trust radii span 0.00116845–8117.58042 and barrier
parameters span 1.6e-10–1e-7. Raw optimality is telemetry, not certification.
Stationarity fails on 54 examples, complementarity on 19, dual feasibility on
45, and the material-shadow gate on 51. These overlapping failures are retained.

## Full-panel endpoint metrics

| Condition | V11 local gain % | V13 local gain % | i+1 / i+2 / i+3 gains % | Raw Cartesian change % | Aligned RMSD change % | Continuous chirality change % | Physical passes |
|---|---:|---:|---|---:|---:|---:|---:|
| 50 | 19.155202 | 19.169373 | 17.351915 / 20.902717 / 19.028673 | -8.012775 | -4.328701 | -3.475220 | 3/20 |
| 250 | 8.358575 | 8.385916 | 10.822962 / 7.887046 / 7.372853 | -0.250128 | -0.149087 | -0.004633 | 1/20 |
| 450 | 8.486524 | 8.500692 | 11.957331 / 9.065883 / 6.640850 | +0.258816 | +0.137313 | -0.003925 | 2/20 |

Condition-450 gain changes by only +0.014168 percentage points. Inversions are
5493 → 5492: one repaired, zero new, zero per-example failures. Assessability is
13029 → 13029, with no temporary loss. All endpoint constraints pass, outputs
are finite, no collapse occurs, and frame eligibility is preserved.

Maximum per-step correction is 0.0399999918 Å. Final net displacement RMS is
0.3150394051 Å; net/path maximum is approximately 0.3199999342 Å. The original
0.04 Å step and 0.32 Å cumulative limits remain respected. Complete P0–P8,
condition, length, per-example and paired numerical telemetry is in `result.json`
and the `B/` records.

## Verification and publication

The 21 focused CPU tests passed. Every panel implementation validation matched
V11 exactly. Independent fixed-state reconstruction reproduced all coordinates,
corrections, metrics and convergence certificates without optimization. Fresh
aggregate reconstruction reproduced `result.json` and `RESULTS.md` byte-for-byte.
All protected historical/source/input/state hashes passed verification. Private
state arrays are retained locally and excluded from Git.

No CUDA, neural training, modified objective, enlarged correction budget,
historical artifact changes or automatic follow-on experiment occurred.

Recommended next experiment: benchmark a direct correction-space SQP/active-set
solver on the same strict feasible set and frozen physical convergence contract.
