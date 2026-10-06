# V9B strict-chirality conditioning audit — COND-A

V9 synthetic preflight is practically converged under the frozen physical
sensitivity checks. Historical INV-D remains unchanged. No scientific panel,
CUDA, or neural training was launched. K=8, s_max=0.04 Å and all V9 constraints,
objective, arithmetic and solver tolerances remain unchanged.

| Quantity | Length 12 control | Length 32 failed preflight |
|---|---:|---:|
| Exact historical non-timing reproduction | yes | yes |
| Iterations / termination | 39 / gtol | 1807 / xtol |
| Solver optimality, infinity norm in z | 4.44841e-13 | 3.87247e-10 |
| Final trust radius | 4.73225e8 | 1.00000e-13 |
| Barrier parameter / subproblem tolerance | 2.56e-13 / 2.56e-13 | 2.56e-13 / 2.56e-13 |
| Last CG stop | tolerance (4) | trust boundary (2) |
| Raw physical stationarity max, Å | 1.17757e-8 | 3.11318e-7 |
| Normalized physical stationarity max | 1.03037e-9 | 2.24950e-8 |
| Normalized physical stationarity L2 | 6.16945e-9 | 1.36461e-7 |
| Active-only reconstructed stationarity max | 1.03035e-9 | 2.49101e-8 |
| Primal violation (normalized and all raw groups) | 0 | 0 |
| Normalized complementarity incl. near-bound balls | 6.70806e-10 | 1.29485e-9 |
| Raw complementarity, Å² | 3.06655e-10 | 7.16798e-10 |
| Normalized dual negativity | 0 | 1.12220e-10 |
| Active non-ball inequalities / balls | 0 / 80 | 1 signed / 224 |
| Physical active Jacobian rows / rank | 80 / 80 | 225 / 225 |
| Physical singular-value range | 1–1 | 0.532331–1.037824 |
| Physical condition number | 1 | 1.94958 |
| Row-normalized condition number | 1 | 1.51214 |
| Active system after radial chain: condition | 7.95399 | 1.60887e6 |
| Duplicate/nearly parallel active row pairs | 0 | 0 |
| Feasible projected gradient L2 (dimensionless) | 6.16938e-9 | 1.34234e-7 |
| Raw local correction-gradient L2, Å | 0.668985 | 0.486061 |
| Maximum directional FD absolute error | 1.15830e-9 | 9.87298e-9 |
| Median / max directional relative error | 1.31592e-8 / 1.19033e-7 | 2.56670e-8 / 1.92072e-7 |

Recovered-state/input hashes are verified and all historical log fields match exactly.
V9 did not store coordinate hashes or final arrays; the new hashes certify the
recovered states, not a nonexistent historical coordinate hash. No existing
intermediate-size strict synthetic case was available. Only the mandatory two
cases were used. Synthetic replay recovered missing state/terminal telemetry;
fixed-state audit reproduction then requires exact canonical JSON equality.

## Radial conditioning and multipliers

Eligible length-12 variables are **more** saturated than length 32:

| Eligible-variable quantity (min / median / max) | Length 12 | Length 32 |
|---|---|---|
| Correction/radius | .999999818 / .999999916 / .999999954 | .392030 / .999999512 / .999999734 |
| Radial eigenvalue | 2.77238e-11 / 6.97018e-11 / 2.20515e-10 | 3.87853e-10 / 9.64022e-10 / .778567 |
| Tangential eigenvalue | .000302657 / .000410562 / .000604151 | .000729271 / .000987852 / .919952 |
| Radial condition number | 2.73973e6 / 5.97533e6 / 1.09169e7 | 1.18160 / 1.02477e6 / 1.88028e6 |

Per-variable values and step summaries are in `audit.json`; fixed endpoint rows
are excluded from these summaries. Radial suppression is substantial, but neither
case retains material feasible descent. The physical active system has full row
rank and mild conditioning. There is no evidence of a blocking signed/assessability
constraint rank defect. The optimizer-mapped active matrix is an analysis of the
conceptual ball KKT system; ball constraints are implicit in the historical radial
map, not additional constraints passed to SciPy.

Length-32's active signed multiplier is 5.02364e-5 in normalized objective/constraint
units (2.13299e-4 against the raw signed quantity). No extremely large multipliers
occur. Its correction-space Lagrangian contribution norm is 7.54023e-4 /Å versus
local objective gradient norm .878039 /Å. Assessability contributions are smaller
(about 1.70e-8 /Å for unsigned q and 6.94e-10 /Å for frame constraints); they do not
dominate stationarity. Tiny negative barrier multipliers remain below the historical
1e-8 dual tolerance. Independent nonnegative active-set reconstruction agrees on
small physical stationarity. Nondimensionalized row scaling with inverse multiplier
transformation leaves the Lagrangian unchanged to float64 roundoff; exact errors
and group statistics are in `audit.json`.

## Noniterated feasible shadows

Positive values denote normalized local-loss decrease; every tested shadow is
strictly feasible, creates no new inversion, and preserves assessability.

| Maximum vector step, Å | Length 12 decrease | Length 32 decrease |
|---|---:|---:|
| 1e-6 | -1.77529e-11 | +1.48721e-11 |
| 1e-5 | -1.85794e-9 | -1.39863e-10 |
| 1e-4 | -8.34891e-7 | -1.82789e-7 |

Length-32's best remaining improvement is ~1.49e-11 of baseline loss, versus the
**preregistered 1e-6 material threshold**. These are local checks, not a proof of
absence of distant/global improvements. All 18 float64 directional-scale checks
pass, including objective, active constraints where present, and Lagrangian.

## Prospective termination contract

Propose a local physical certificate, without modifying or rerunning V9:

- Preserve exact strict signed/assessability feasibility and discrete gates;
  global normalized constraint residual <=1e-8; exact correction bounds.
- Use fixed-scale dimensionless physical stationarity rather than SciPy z-space
  optimality. For M eligible step/residue vectors, require L2 residual
  <=1e-6/(sqrt(M)*1e-4/0.04). This is 4.47214e-5 for length 12 and 2.58199e-5
  for length 32. These follow the frozen one-ppm sensitivity and 1e-4 Å probe
  displacement by Cauchy–Schwarz, not a threshold chosen to admit either case.
- Require derivative-validation uncertainty at least tenfold smaller than the
  sensitivity threshold; numerical certification must cover the projected/
  Lagrangian directions used in a future certificate.
- Retain dual negativity <=1e-8 and normalized complementarity <=1e-6.
- Require no material feasible descent in independent frozen shadow tests.

This is a proposal for future preregistration, not authorization to run real
examples. A physical local certificate does not prove the maximum safe local repair.

140 CPU regression tests passed (two existing SciPy singular-Jacobian warnings
from synthetic tests); focused V9B/V9 tests passed 17/17. Historical tracked-file
hash integrity is checked before and after the audit. E010 is never loaded or
mutated. Tests, installed SciPy source hashes, preregistration, exact recovery,
audit and reproduction records accompany this report.

**Exactly one recommended next action:** prepare V10 with a preregistered
physically normalized KKT termination contract.

Execution note: the first short control recovery was discarded because its comparison
used Python tuples against JSON lists. The comparator was corrected to the historical
JSON canonicalization and unchanged length-12 recovery repeated. The fixed-state
reproduction comparator received the same serialization correction. No optimizer
setting, scientific value, or historical file was changed.
