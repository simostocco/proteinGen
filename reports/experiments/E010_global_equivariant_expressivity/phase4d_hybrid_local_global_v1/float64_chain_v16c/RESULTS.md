# V16C float64 Jacobian-chain correction

**PRECISION-A — float64 chain repaired.** No optimization, teacher panel, CUDA or neural training.

Historical V16A/V16B files, reports and failure classifications are unchanged. Only the new versioned implementation is corrected. All solver settings, scientific equations, tau, bounds and eligibility semantics are preserved.

## Precision changes

- OneStep constructs scale using current.new_tensor(0.10), casts Boolean eligibility to current.dtype only for arithmetic, and uses both explicit float64 objects in the coordinate/Jacobian chains.
- NumPy ball inputs are explicitly float64. Objective/constraint definitions, K_max=16 and s_max=0.10 Å are unchanged.
- The comprehensive trace also found PyTorch determinant-backward 0/1 temporary tensors in default float32. These are exactly representable and caused no value defect, but violate an all-float64 intermediate contract. A scoped float64 default during jac/cjac differentiation removes them and restores the prior default in finally. It is designed for the preregistered isolated single-threaded teacher workers. Kabsch equations are unchanged.
- Float64Trace is an opt-in operation-level assertion covering nested geometry, objective, constraints and autograd. Boolean/integer objects stay Boolean/integer. No trace overhead when disabled.

## Value parity before derivatives

At every exact recovered V16B state, coordinates, local/cartesian/aligned/chiral terms, signed q, turn/bond assessability quantities and every safety constraint are bitwise identical old/new: **all reported differences are zero**. Eligibility, inversion/assessability and safety classifications are exactly unchanged. Archived state/direction/input/eligibility hashes reproduce where available; no optimization was used for recovery.

## Primary finite differences: h=1e-4

Unchanged FD tolerance: 1e-8 +1e-5*abs(analytic directional derivative). Unchanged analytic tolerance:1e-12 +1e-10*abs(analytic directional derivative). Each row includes scalar objective and all vector safety components. Worst vector component is selected by error/tolerance ratio for reporting, never for acceptance.

| Length | Phase | Analytic max discrepancy | Worst family | Analytic | FD | Abs error | Rel error | Allowed error | Physical max/RMS Å | Pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 0.0 | 1.11022e-16 | aligned | -0.000491007234236 | -0.000491007240977 | 6.74057e-12 | 1.37281e-08 | 1.49101e-08 | 2.9234e-06/2.64262e-06 | True |
| 12 | 0.7 | 5.55112e-17 | aligned | 0.00117048651296 | 0.00117048650883 | 4.12847e-12 | 3.52714e-09 | 2.17049e-08 | 2.96704e-06/2.63295e-06 | True |
| 12 | 1.3 | 5.55112e-17 | aligned | 0.00221606661885 | 0.00221606662354 | 4.68731e-12 | 2.11515e-09 | 3.21607e-08 | 2.91883e-06/2.62654e-06 | True |
| 32 | 0.0 | 5.55112e-17 | bond_assessability | -0.00232250914999 | -0.00232250915833 | 8.33242e-12 | 3.58768e-09 | 3.32251e-08 | 1.80848e-06/1.7114e-06 | True |
| 32 | 0.7 | 5.55112e-17 | bond_assessability | 0.000221810384749 | 0.000221810373957 | 1.07919e-11 | 4.86539e-08 | 1.22181e-08 | 1.82261e-06/1.71181e-06 | True |
| 32 | 1.3 | 1.11022e-16 | signed | -0.00358064727485 | -0.0035806473031 | 2.82517e-11 | 7.89011e-09 | 4.58065e-08 | 1.8273e-06/1.71195e-06 | True |
| 500 | 0.0 | 4.16334e-17 | signed | -0.000148064045939 | -0.000148064959737 | 9.13798e-10 | 6.17164e-06 | 1.14806e-08 | 4.59382e-07/4.46318e-07 | True |
| 500 | 0.7 | 2.77556e-17 | unsigned_assessability | -8.86679465182e-05 | -8.86672785194e-05 | 6.67999e-10 | 7.53371e-06 | 1.08867e-08 | 4.59595e-07/4.46319e-07 | True |
| 500 | 1.3 | 2.77556e-17 | unsigned_assessability | 0.000188479764781 | 0.00018848026806 | 5.03279e-10 | 2.6702e-06 | 1.18848e-08 | 4.59724e-07/4.46319e-07 | True |

Full-Jacobian vs independent whole-function autograd JVP maximum discrepancy: **1.11022e-16**. Direct scalar family contractions maximum discrepancy: **6.93889e-17**. All pass unchanged analytic tolerances.

Nine distinct length×phase primary validations pass. At each length, the historical nine phase×old-epsilon slots map to the fixed primary h=1e-4 (all9 slots pass); three reused slots per direction are explicitly labeled, not claimed as additional independent validations. Full per-family scalar/vector values, errors and physical perturbations are in cases JSON.

## Resolution sanity (not used for acceptance)

| Length | Phase | h | All components pass | Failed rows |
| --- | --- | --- | --- | --- |
| 12 | 0.0 | 1e-06 | True | 0 |
| 12 | 0.0 | 3e-05 | True | 0 |
| 12 | 0.0 | 1e-04 | True | 0 |
| 12 | 0.7 | 1e-06 | True | 0 |
| 12 | 0.7 | 3e-05 | True | 0 |
| 12 | 0.7 | 1e-04 | True | 0 |
| 12 | 1.3 | 1e-06 | True | 0 |
| 12 | 1.3 | 3e-05 | True | 0 |
| 12 | 1.3 | 1e-04 | True | 0 |
| 32 | 0.0 | 1e-06 | True | 0 |
| 32 | 0.0 | 3e-05 | True | 0 |
| 32 | 0.0 | 1e-04 | True | 0 |
| 32 | 0.7 | 1e-06 | True | 0 |
| 32 | 0.7 | 3e-05 | True | 0 |
| 32 | 0.7 | 1e-04 | True | 0 |
| 32 | 1.3 | 1e-06 | True | 0 |
| 32 | 1.3 | 3e-05 | True | 0 |
| 32 | 1.3 | 1e-04 | True | 0 |
| 500 | 0.0 | 1e-06 | False | 32 |
| 500 | 0.0 | 3e-05 | True | 0 |
| 500 | 0.0 | 1e-04 | True | 0 |
| 500 | 0.7 | 1e-06 | False | 50 |
| 500 | 0.7 | 3e-05 | True | 0 |
| 500 | 0.7 | 1e-04 | True | 0 |
| 500 | 1.3 | 1e-06 | False | 36 |
| 500 | 1.3 | 3e-05 | True | 0 |
| 500 | 1.3 | 1e-04 | True | 0 |

## Dtype and integrity evidence

cases/length_12.json, length_32.json and length_500.json report Python/NumPy/Torch types, dtypes/devices for state, scale, masks, every objective term, q/turn/bonds, constants and constraints, plus operator-level old/new dtype traces. Every new floating operation in geometry and differentiation was CPU float64 under strict mode. Old traces preserve both contamination paths. The process default dtype was restored after differentiation.

Optimizer calls are fail-fast blocked in audit and focused tests. Tests requiring real historical optimization are excluded rather than run. No synthetic or real optimization is authorized or invoked. New-lineage tests validate scale storage, Boolean semantics, value parity, old-defect detection, strict new dtype assertions, analytic/FD agreement and unchanged scientific settings. Independent reproduction must match all numerical records and original protected hashes.

Exactly one recommended next action: **prepare a new versioned sequential-teacher replay using V16C float64 implementation and fixed h=1e-4 validation; retain all other V16A scientific and solver settings.** No replay or teacher generation is launched by V16C.
