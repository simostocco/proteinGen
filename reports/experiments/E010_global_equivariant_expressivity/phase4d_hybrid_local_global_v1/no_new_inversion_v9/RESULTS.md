# E010 Phase4D V9 numerical preflight result

**INV-D — numerically inconclusive. Scientific panel NOT launched.**

The exact signed-quartet formulation is baseline-feasible on all60 fixed examples; all13,029 raw q quantities match the historical evaluator bitwise. Baseline correct7,536; inverted5,493. Correct quartets receive target-signed q>=nextafter(1e-6,+inf). Inverted quartets receive unsigned q²>=tau², with no target-sign restriction. Historical frame/bond assessability floors are supported explicitly. New constraints admit no relaxed final feasibility tolerance.

K=8 and s_max=.04 Å, targets, Pg, objective and continuous safety constraints remain unchanged. Sparse exact Jacobian storage is required for the expanded inequalities. Exact Hessian-vector products remain in use.

The1000-cap synthetic result is archived. The same length32 V8 control converges50 iterations. One globally applicable preflight-only cap revision to2000 was permitted and frozen before scientific outcomes. No further cap/tolerance/solver change occurred.

| Synthetic length | Converged | Iterations | Termination | Optimality | Ball KKT | Exact feasible | New inversions | Assessment loss | Active signed | Minimum signed margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | True | 39 | `gtol` termination condition is satisfied. | 4.448412835335301e-13 | 7.035081776590858e-08 | True | 0 | 0 | 0 | 0.0590740199448086 |
| 32 | False | 1807 | `xtol` termination condition is satisfied. | 3.872466951170198e-10 | 4.61970916772828e-06 | True | 0 | 0 | 1 | 1.4509512428429283e-08 |

The length32 result passes physical KKT/projection thresholds and every constraint/discrete gate, but fails the retained1e-12 parameter-space convergence requirement. Its xtol termination at1807 iterations means another cap extension alone is not justified. Preserve this failure; do not reinterpret it as a converged oracle.

Initial finite-difference checks exposed a validation-units error: Angstrom epsilons were applied to dimensionless z, shrinking physical perturbations25-fold. The corrected epsilon_z=epsilon_angstrom/.04 preserves the fixed epsilon set and mixed tolerance. All180 constraint directional validations and all180 aligned-gradient validations now pass on all60 baselines, together with zero-state Hessian checks. Prior failed checks and direct-autograd diagnoses remain archived. No optimizer or constraint arithmetic changed. The only remaining blocker is strict synthetic convergence.

Condition50/250/450 local gains, offsets, aligned/chiral changes, final inversions, active constraints, minimum margins and P0-P8/path/saturation metrics are **not evaluated for V9**. V8 historical safe gains19.2207/8.4448/8.5719% are comparisons only and cannot substitute for a V9 result. The complete historical target is neither passed nor disproved.

Validation:130 focused CPU tests pass, including sign-crossing rejection, inverted-to-correct permission, assessability support, exact sparse/autograd Jacobian agreement and float64 Hessian finite differences. All60 baselines satisfy the exact formulation; historical/source/input hashes remain intact. Synthetic reproduction evidence is separate from scientific convergence. No CUDA, neural training, E010 mutation, environment change, bound sweep or development evaluation.

Recommended next experiment: a preregistered numerical conditioning audit of the V9 endpoint constraints and radial-variable trust-constr solve, retaining K=8/s_max=.04 and the exact scientific feasible set, before any panel optimization.
