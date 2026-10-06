# V12 fixed-state numerical audit

**NUM-A — iteration-budget limited (evidence-supported priority, not a guarantee of convergence at 2000).**
All 60 original V11 endpoint files, coordinates, corrections, objectives, metrics,
constraints and certificates reproduced exactly. A fresh process independently
reproduced all 60 audit records and both aggregates byte for byte. No trust-constr
call, optimizer continuation, endpoint mutation, CUDA or neural training occurred.
Only the authorized fixed-state NNLS tangent-cone diagnostic was solved.

## Frozen groups

| Group | Overall | 50 | 250 | 450 |
|---|---:|---:|---:|---:|
| A: physically converged | 4 | 2 | 1 | 1 |
| B: nonconverged, material feasible shadow | 55 | 18 | 19 | 18 |
| C: nonconverged, no material shadow | 1 | 0 | 0 | 1 |

The four A indices are 9,10,11,45. C is index14, 2mdw_A/450. Stratum20–64
contains4/7/1; each other historical stratum contains0/12/0. Nothing was filtered.
V11 remains DIRECT-D, with its endpoint gains19.1552%,8.3586%,8.4865% unchanged.
Those gains remain uncertified as a complete-panel feasibility result.

## Physical geometry and remaining descent

The frozen active set contains1336 correction balls,13 continuous chirality
constraints and16 unsigned assessability constraints. Signed no-new-inversion,
frame/bond and aligned-RMSD active counts are zero. Forty nonconverged endpoints
have an empty frozen active set, despite often being near the radius boundary.
All20 nonempty active Jacobians have full row rank; none has nearly parallel rows.
Ranks range1–336. Frozen normalized condition numbers range1–8892.98; row-unit
companion conditions range1–2.634. The worst raw scaling occurs in a converged
control, not a failure. Nonconverged nonempty conditions have median2.467 and
maximum1217.89 (row-unit maximum2.634). This does not support a harder physical
active-set geometry as the dominant failure mechanism. Row-unit calculations are
analysis only; the optimization problem and certificate were not rescaled.

Across all states, normalized physical stationarity ranges9.20e-9–1.38123e-2
(median4.85193e-3); projected feasible-gradient norm ranges1.28217e-8–1.38153e-2
(median4.85952e-3). Primal violation is exactly zero. Normalized complementarity
ranges1.65216e-9–2.30465e-5 (median6.93859e-7); dual negativity maximum is2.43824e-4.
The same4/60 certificates pass; no tolerance was changed.

Physical local-loss gradient is for normalized L_local/L_local(Pg). Multiplication
by S=.04 gives the frozen dimensionless gradient convention; raw MSE gradients
and physical singular values are also present in the per-case reports. For active
balls, outward radial gradient norm ranges0–.0252140, inward norm is zero, and
tangential norm ranges0–.00123671. All-eligible outward median is.00575272;
tangential median.000265885. An outward gradient on a true active ball is not
available descent beyond its bound.

The **projected directions for all56 nonconverged endpoints are predominantly
radial**, with tangential squared-norm fraction at most2.2111% (B median.3152%).
Summed predicted decrease is approximately99.75% radial and over99.999% on variables
outside the frozen active band. The eight recurrent steps contribute almost
equally; full residue and normalized-position contributions are recorded. The
dominant residual is not tangential movement along many already-active balls.
The median nonconverged99%-saturation fraction is about90.3%, while median frozen
active fraction is zero. These two definitions must not be conflated.

The unchanged NNLS projection is an approximate block-radius linear-loss direction,
not a claim to solve the block-radius linear program globally. Maximum linearized
violation is4.13e-18. Exact nonlinear checks reproduce historical shadows:

| Physical scale (Å) | Feasible trials | Material trials |
|---|---:|---:|
| 1e-6 | 59 | 54 |
| 1e-5 | 46 | 44 |
| 1e-4 | 13 | 12 |

There are55 material-shadow examples and110 material trials. Largest feasible
normalized loss decrease is3.49468e-4. Infeasible trials are not counted as useful
descent; every shadow starts independently from the fixed endpoint.

## Trust-constr and budget evidence

All56 failures reached1000 iterations. Their final trust radii range.00161690–2.56901
(B median.0493176), far above frozen xtol1e-12. These are augmented optimizer/slack
radii, not observed Angstrom steps. No endpoint has a collapsed final radius.
The failure barrier parameters are1e-7 in40 cases and2e-8 in16 cases, still20,000–100,000
times configured barrier_tol. That fact alone is not a failure proof: converged
controls also terminate physically at barriers above barrier_tol.

Saved histories are ten-iteration snapshots. Of56 failures,55 have a normalized
objective decrease at least the frozen materiality1e-6 during the available
iteration910–1000 window. B median decrease is3.52228e-5, with median slope
-3.73827e-7 per iteration. Over iteration760–1000, B median decrease is1.30143e-4.
However physical stationarity is mostly flat: only4/56 show a1% drop in the last
window, and B median relative change is approximately-1.315e-5. B median
complementarity improves4.14% in the last window and18.95% over the250-window.
Raw optimality fluctuates rather than decreasing monotonically. Seven B cases
reduce their barrier in the last250-window. All available window slopes/spans
are published; no convergence iteration is extrapolated.

Installed SciPy reports optimality as the infinity norm of the z-space Lagrangian
gradient, including scientific and quadratic-ball multipliers. Reconstructed
values agree within1e-12 absolute/1e-8 relative, and range1.33605e-9–4.30036e-6.
The barrier subproblem also has slack-space optimality and equality-feasibility
conditions. Internal slacks, constraint penalty, actual accepted step norms and
multiplier histories were not serialized. They remain unavailable. True-slack
centering proxies are explicitly labeled inferred, not actual internal telemetry.

Frozen inferred active-ball multipliers range3.89304e-4–3.92655e-3
(median9.54263e-4). Active continuous-chirality multipliers range.00673471–.0245857;
unsigned assessability multipliers range.0107882–10.8170. These numbers depend on
the historical constraint scaling; they are not directly comparable across
families. No saved multiplier trajectory supports a claim about ongoing balance.

## Historical length32 comparison and interpretation limits

The immutable radial length32 preflight finished at1807 iterations, with physical
stationarity1.36461e-7 and projected gradient1.34234e-7, no material shadows. Its
last250-window objective decrease was only1.05152e-7, below the frozen materiality;
its barrier fell from6.4e-12 to2.56e-13. Final-window trust radii cycled widely.
This provides descriptive evidence that1000 need not be a sufficient budget.
Its physical KKT and multiplier iteration histories were not saved, so the
earlier trajectory cannot be quantitatively matched to V11 physical histories.

The iteration-progress signature covers55/56 failures (**98.21%**, or91.67% of all60).
The specified strong tangential-barrier signature covers0/56 (**0%**). Index8
is unresolved by that progress signature. These are evidence signatures, not
causal case allocations. Barrier inefficiency is not excluded: nearly flat
physical stationarity and persistent early barrier stages warrant scrutiny.
Nonetheless continued material objective progress, noncollapsed radii, benign
row-unit active geometry and predominantly interior radial descent support
testing the matched budget before replacing the solver. No defensible guarantee
of2000-iteration convergence follows from these records.

**One next experiment:** repeat V11 from zero with exactly2000 iterations and
otherwise identical settings and the unchanged physical convergence contract.
This experiment is justified but was not launched. A different solver is not
established as the primary next intervention by this fixed-state evidence.

## Validation and provenance

24 focused CPU tests passed, covering V12 diagnostics and inherited V9B/V11
contracts. Ruff passed. All60 protected state hashes, all historical tracked-file
hashes, frozen inputs and installed source hashes passed before/after both
independent audits. Every final coordinate/metric/constraint/certificate agrees
with V11. No checkpoint, dataset or temporary solver array is published.
`preregistration.json`, per-case reports, aggregates, `interpretation.json` and
`reproduction.json` provide the complete evidence. Historical artifacts remain
unchanged. CPU float64 only; optimization launched:NO; neural training:NO; CUDA:NO.
