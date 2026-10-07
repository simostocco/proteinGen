# V16B length-500 derivative resolution audit

**FD-D — mixed: small-step finite-difference under-resolution plus a separate analytic chain-scaling precision defect.**

The exact V16A check state, deterministic directions and every available historical derivative comparison were reproduced bitwise. V16A has no saved derivative-state coordinate hash; new state/input/eligibility hashes document the recovered state without claiming nonexistent historical hashes. Historical V16A and all scientific settings remain unchanged.

## Checks and recovery

The nine checks cross phase 0/.7/1.3 with dimensionless h=1e-6/1e-5/1e-4. Each includes a scalar normalized local objective and a 1,495-component vector of safety constraints (500 case): aligned RMSD, continuous chirality, signed no-new-inversion, unsigned assessability, frame assessability and bond assessability. Ball constraints are not part of the historical nine-check vector; its construction is preserved exactly. Every vector component must pass 1e-8 + 1e-5*abs(analytic derivative). Full family-wise records and all vector components are retained in curve JSON and reproducible ignored numerical arrays with versioned hashes.

| Phase | h | Worst vector family | Analytic Jd | Centered FD | Absolute error | Relative error | Frozen allowed error | Failed vector rows | Objective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 1e-06 | signed | -0.000167327962257 | -0.000167254210481 | 7.37518e-08 | 0.000440762 | 1.16733e-08 | 32 | pass |
| 0.0 | 1e-05 | signed | -0.000148064048146 | -0.000148057172078 | 6.87607e-09 | 4.64398e-05 | 1.14806e-08 | 0 | pass |
| 0.0 | 1e-04 | signed | -0.000148064048146 | -0.000148064959737 | 9.11592e-10 | 6.15674e-06 | 1.14806e-08 | 0 | pass |
| 0.7 | 1e-06 | unsigned_assessability | -0.00135405634968 | -0.00135388555922 | 1.7079e-07 | 0.000126132 | 2.35406e-08 | 50 | pass |
| 0.7 | 1e-05 | unsigned_assessability | -8.86679478394e-05 | -8.86604623052e-05 | 7.48553e-09 | 8.44221e-05 | 1.08867e-08 | 0 | pass |
| 0.7 | 1e-04 | unsigned_assessability | -8.86679478394e-05 | -8.86672785194e-05 | 6.6932e-10 | 7.54861e-06 | 1.08867e-08 | 0 | pass |
| 1.3 | 1e-06 | unsigned_assessability | -1.63024110652e-05 | -1.64134261738e-05 | 1.11015e-07 | 0.00680974 | 1.0163e-08 | 36 | pass |
| 1.3 | 1e-05 | unsigned_assessability | -1.63024110652e-05 | -1.63083102578e-05 | 5.89919e-09 | 0.00036186 | 1.0163e-08 | 0 | pass |
| 1.3 | 1e-04 | unsigned_assessability | 0.000188479767589 | 0.00018848026806 | 5.0047e-10 | 2.6553e-06 | 1.18848e-08 | 0 | pass |

Historical failures are exactly (phase0,h1e-6):32 rows, (phase.7,h1e-6):50 rows, (phase1.3,h1e-6):36 rows. All six larger-h checks and all nine scalar-objective checks pass unchanged tolerances.

## Physical perturbations

| Phase | h | Intended max Å | Intended RMS Å | Stored max Å | Stored RMS Å |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 1e-06 | 4.59382e-09 | 4.46318e-09 | 4.59382e-09 | 4.46318e-09 |
| 0.0 | 1e-05 | 4.59382e-08 | 4.46318e-08 | 4.59382e-08 | 4.46318e-08 |
| 0.0 | 1e-04 | 4.59382e-07 | 4.46318e-07 | 4.59382e-07 | 4.46318e-07 |
| 0.7 | 1e-06 | 4.59595e-09 | 4.46319e-09 | 4.59599e-09 | 4.46319e-09 |
| 0.7 | 1e-05 | 4.59595e-08 | 4.46319e-08 | 4.59596e-08 | 4.46318e-08 |
| 0.7 | 1e-04 | 4.59595e-07 | 4.46319e-07 | 4.59595e-07 | 4.46319e-07 |
| 1.3 | 1e-06 | 4.59724e-09 | 4.46319e-09 | 4.59724e-09 | 4.46319e-09 |
| 1.3 | 1e-05 | 4.59724e-08 | 4.46319e-08 | 4.59724e-08 | 4.46319e-08 |
| 1.3 | 1e-04 | 4.59724e-07 | 4.46319e-07 | 4.59724e-07 | 4.46319e-07 |

z is dimensionless; physical delta=.10*z at eligible residues. RMS is across all masked residues (endpoints zero). The full 13-value grid retains intended/stored physical perturbations for every h, including unchanged-coordinate fractions.

## Resolution curve: length500, maximum across all three directions

| h | Max vector absolute error | Max error / frozen allowed error | Failed rows, summed | All checks pass |
| --- | --- | --- | --- | --- |
| 1e-10 | 0.00183972 | 66085.6 | 4478 | False |
| 3e-10 | 0.000661128 | 19844.6 | 4457 | False |
| 1e-09 | 0.000227115 | 12259.1 | 4389 | False |
| 3e-09 | 7.44273e-05 | 2020.96 | 4221 | False |
| 1e-08 | 2.26314e-05 | 1164.09 | 3774 | False |
| 3e-08 | 6.62991e-06 | 276.581 | 3054 | False |
| 1e-07 | 1.8645e-06 | 90.2 | 1757 | False |
| 3e-07 | 7.33086e-07 | 22.4288 | 716 | False |
| 1e-06 | 2.31639e-07 | 10.9234 | 118 | False |
| 3e-06 | 6.77119e-08 | 1.99618 | 10 | False |
| 1e-05 | 2.02051e-08 | 0.687587 | 0 | True |
| 3e-05 | 7.51773e-09 | 0.289321 | 0 | True |
| 1e-04 | 3.07994e-09 | 0.0794025 | 0 | True |

## Cancellation at the three failed checks

| Phase | Worst family | f(x) | f+ | f− | Absolute numerator | eps64 * local function scale | Numerator / roundoff scale | Reliable relative digits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | signed | 0.98015415136961037 | 0.9801541512023566 | 0.98015415153686503 | 3.34508e-10 | 2.17638e-16 | 1.53699e+06 | 3.356 |
| 0.7 | unsigned_assessability | 1.0238917996767818 | 1.0238917983227509 | 1.023891801030522 | 2.70777e-09 | 2.2735e-16 | 1.19102e+07 | 3.899 |
| 1.3 | unsigned_assessability | 1.041069653243087 | 1.0410696532265413 | 1.0410696532593682 | 3.28269e-11 | 2.31164e-16 | 142007 | 2.167 |

Small h produces physical vector changes only ~4.6e-9 Å at coordinates near 998 Å. Subtraction of nearby translated coordinates to form bonds introduces finite precision before the centered numerator is formed. Function numerators are tiny relative to O(1) normalized constraint values; the local eps64 comparison is a lower bound, not a complete forward-error bound for this geometry pipeline. Error increases strongly as h shrinks, then decreases in the larger-h window. This supports cancellation/under-resolution rather than an h-independent derivative error as the cause of the three historical failures.

## Analytic precision defect

The independent analytic paths are (A) the historical full coordinate Jacobian followed by the stored mask chain; (B) direct autograd JVP through the entire current-geometry function; plus scalar reverse-autograd family contractions. Objective and global constraints agree. Quartet rows do not pass the unchanged preregistered analytic-consistency tolerance 1e-12+1e-10*abs(Jd).

V16A `.1 * Boolean eligibility` promotes to torch.float32, yielding **0.10000000149011612** rather than float64 0.1. The historical quartet derivatives therefore have multiplicative factor **1.0000000149011612**. Maximum analytic discrepancy across lengths/directions is **7.69101e-09**; the factor model explains it to **1.66533e-16**. Maximum scalar-contraction discrepancy is **6.25684e-09**. No historical source was repaired in V16B. This small defect is separate from the much larger small-h finite-difference errors and prevents an FD-A claim that all analytic paths agree.

## Length controls, stable windows and Richardson

| Length | Phase | Passing grid h | Proposed future h |
| --- | --- | --- | --- |
| 12 | 0.0 | 3e-08, 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 12 | 0.7 | 1e-08, 3e-08, 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 12 | 1.3 | 1e-08, 3e-08, 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 32 | 0.0 | 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 32 | 0.7 | 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 32 | 1.3 | 1e-07, 3e-07, 1e-06, 3e-06, 1e-05, 3e-05, 1e-04 | 1e-4 |
| 500 | 0.0 | 1e-05, 3e-05, 1e-04 | 1e-4 |
| 500 | 0.7 | 1e-05, 3e-05, 1e-04 | 1e-4 |
| 500 | 1.3 | 1e-05, 3e-05, 1e-04 | 1e-4 |

Common window across every objective/vector component, direction and length: **1e-5,3e-5,1e-4**, with original FD tolerances. The lower edge shifts upward with length: all-direction length12 window begins at3e-8, length32 at1e-7, length500 at1e-5.

| Length | Phase | h | Max abs(Dh−Dhalf) | Max Richardson extrapolate error | Half-step all-pass |
| --- | --- | --- | --- | --- | --- |
| 12 | 0.0 | 1e-05 | 3.33067e-10 | 7.87379e-09 | True |
| 12 | 0.0 | 3e-05 | 1.5173e-10 | 7.5216e-09 | True |
| 12 | 0.0 | 1e-04 | 9.60343e-11 | 7.58162e-09 | True |
| 12 | 0.7 | 1e-05 | 5.66214e-10 | 6.44117e-09 | True |
| 12 | 0.7 | 3e-05 | 1.29526e-10 | 5.70719e-09 | True |
| 12 | 0.7 | 1e-04 | 4.44089e-11 | 5.68585e-09 | True |
| 12 | 1.3 | 1e-05 | 4.82947e-10 | 7.00708e-09 | True |
| 12 | 1.3 | 3e-05 | 1.29526e-10 | 7.32411e-09 | True |
| 12 | 1.3 | 1e-04 | 3.77476e-11 | 7.17879e-09 | True |
| 32 | 0.0 | 1e-05 | 1.84297e-09 | 6.38931e-09 | True |
| 32 | 0.0 | 3e-05 | 5.9952e-10 | 5.05278e-09 | True |
| 32 | 0.0 | 1e-04 | 1.40998e-10 | 4.84418e-09 | True |
| 32 | 0.7 | 1e-05 | 1.90958e-09 | 6.62639e-09 | True |
| 32 | 0.7 | 3e-05 | 5.18104e-10 | 5.3114e-09 | True |
| 32 | 0.7 | 1e-04 | 2.15383e-10 | 5.00732e-09 | True |
| 32 | 1.3 | 1e-05 | 1.02141e-09 | 3.92337e-09 | True |
| 32 | 1.3 | 3e-05 | 5.58812e-10 | 4.45381e-09 | True |
| 32 | 1.3 | 1e-04 | 1.76525e-10 | 4.34637e-09 | True |
| 500 | 0.0 | 1e-05 | 3.80251e-08 | 4.88677e-08 | False |
| 500 | 0.0 | 3e-05 | 1.76359e-08 | 1.79176e-08 | True |
| 500 | 0.0 | 1e-04 | 4.09672e-09 | 4.49768e-09 | True |
| 500 | 0.7 | 1e-05 | 4.0834e-08 | 5.12004e-08 | False |
| 500 | 0.7 | 3e-05 | 1.36816e-08 | 1.63198e-08 | True |
| 500 | 0.7 | 1e-04 | 3.91909e-09 | 4.49546e-09 | True |
| 500 | 1.3 | 1e-05 | 4.20886e-08 | 5.45147e-08 | False |
| 500 | 1.3 | 3e-05 | 1.41738e-08 | 1.78726e-08 | True |
| 500 | 1.3 | 1e-04 | 3.9535e-09 | 5.33005e-09 | True |

Full Dh and Dhalf vectors are retained with hashes. The half-step comparison supports resolution in the stable window; Richardson does not consistently improve a roundoff-dominated estimate and is not used to replace the analytic derivative. No clear truncation-dominated O(h²) regime emerges within the fixed grid ending at1e-4; no larger h was added after inspection.

## Proposal and compliance

Proposed future validation rule: **fixed dimensionless h=1e-4**, contingent on separately repairing/validating the float32 chain-scaling defect in a new version. This was proposed from numerical scale before running the grid; its length500 maximum/RMS physical perturbations are about4.6e-7/4.46e-7 Å. The proposal is not applied to V16A. Historical tolerances, bounds, constraints and solver settings remain immutable.

The audit/reproducer itself invokes no optimizer and uses no teacher-quality outcomes. A validation-command mistake included one inherited test that executed **two zero-start length12 synthetic SLSQP solves**; no real-panel examples or saved endpoints were touched. Those test outputs are excluded from all audit evidence and conclusions. This task-level deviation is explicitly recorded in compliance_deviation.json; subsequent validation uses only optimizer-guarded audit tests. The historical scientific panel was never launched, CUDA was never used, and no neural training occurred.

Recommended next action: **prepare a versioned correction of the Boolean-mask chain to explicit float64, then validate it independently at the unchanged tolerances before considering any teacher replay**.
