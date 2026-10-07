# V16A sequential privileged teacher: preflight stopped

**TEACH-E — numerical preflight failure. No teacher corpus is authorized.**

The scientific panel and worker calibration were not launched. No real trajectory or transition was generated (0/60, 0/960). No neural training, CUDA, package changes, solver fallback or tolerance changes occurred. All historical tracked files and protected inputs were verified against preregistered hashes.

The prepared sequential policy uses K_max=16, s_max=0.10 Å, zero-initialized one-step Cartesian variables, CPU float64 SciPy 1.18.1 SLSQP (maxiter=1000, ftol=1e-12), historical local-only objective, original-Pg strict safety constraints, and deterministic best-feasible-iterate selection. Material improvement is >1e-6 of original Pg local MSE, unchanged from the V9B loss-sensitivity budget. This is a privileged safe teacher proposal, not a globally optimal oracle.

## Preflight evidence

| Length | SLSQP iterations | Solver success | Selected action safety | Normalized local MSE | Solve wall time | Peak worker RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 72 | yes | passes | 0.6776586963 | 0.968 s | 0.672 GiB |
| 32 | 143 | yes | passes | 0.7218532509 | 2.350 s | 0.672 GiB |
| 500 | not invoked | unavailable | not evaluated | unavailable | unavailable | resource solve not run |

All 60 original Pg baselines passed exact safety checks; their 13,029 assessable quartet ownership remains unchanged. Focused CPU tests: **42 passed**. Covered direct one-step geometry, zero parity, historical constraint parity, signed chirality, assessability, derivatives on small geometry, exact NumPy ball feasibility, minimum inward guard, deterministic best-feasible selection, invalid solver output/zero action, original-baseline ownership, target privilege and local-frame reconstruction. Process-parallel execution remains unvalidated because calibration was not launched. Tests do not supersede the failed length-500 numerical gate.

Length-500 variable count is 1,500; 1,995 total inequalities. Installed SLSQP's mandatory buffer and dense-normal arrays estimate is 249,329,864 bytes (0.232 GiB), excluding framework and other allocations. This estimate is not a completed memory/runtime smoke.

The frozen analytic-constraint directional comparison failed in 3/9 direction/epsilon checks, all at dimensionless epsilon=1e-6. The fixed epsilon=1e-5 and 1e-4 comparisons passed for every direction, and all objective-gradient comparisons passed. Failed row counts were 32, 50 and 36. Constraint families affected were signed chirality, unsigned assessability, bond assessability and, for two directions, frame assessability. Maximum absolute constraint derivative discrepancy was 2.3164e-7; maximum error/allowed-tolerance ratio was 10.9234. Tolerance remained **1e-8 + 1e-5*abs(analytic directional derivative)**. The pattern is consistent with finite-difference cancellation at the smallest perturbation on large translated coordinates; it does not establish a gradient defect or authorize relaxing the frozen check.

Maximum inward numerical guard observed in the two small solves was **3.1141e-17 Å**. It only corrected the ball; every nonlinear constraint was re-evaluated afterwards. No scientific correction distribution, condition/horizon gain, plateau, path, inversion or V13-relative quality conclusion is available. Small synthetic actions are evidence of plumbing and safety, not panel feasibility.

## Reproduction and stopping

The analysis-only `scripts/audit_e010_sequential_teacher_v16a_preflight.py` recomputes all nine fixed length-500 derivative comparisons, verifies original protected pins and exactly reproduces the recorded failure/result/validation JSON without optimizer invocation. Small preflight telemetry and action digests are retained; no claim is made that full small solver states were serialized or independently replayed.

Recommended next experiment: **a fixed-state float64 finite-difference resolution audit of the length-500 one-step constraint Jacobian, preserving recorded tolerances and solver settings**. Do not generate teacher trajectories or launch distillation before resolving that preflight blocker.
