# E010 constraint-projected privileged teacher v1

The separate `fast_privileged_constraint_projected_teacher_v1` experiment is
complete. It improves repair over the frozen gradient pilot but misses the
approximately 80% gold rollout-retention target. Work stops at this benchmark;
no additional labels, nonlinear gold solves, models, corpus, or student training
were launched.

## Recovery and storage

Recovery found HEAD `c97ac6a`, no subsequent commits, and a clean worktree. No
interrupted projected experiment was found in the D:-resident data tree. The
FAST-PT-v1 negative pilot, completed gradient experiment, frozen gold archives,
and their source/result hashes were preserved and verified. The previous
conversation was not used as a source of execution state.

All new artifacts, including temporary files, caches, logs, Jacobians, solver
outputs, and reports, are under:

`/mnt/d/Simone/proteinGen/outputs/fast_privileged_constraint_projected_teacher_v1`

The full report is `REPORT.md`; detailed measurements and groups are in
`results.json`; trajectory archives contain all actions and certificates.
`COMPLETED.json` pins the artifacts. Large benchmark payloads remain outside git
under the repository's existing `outputs/` exclusion.

## Mechanistic result

Exactly the existing 278 transitions from 20 trajectories and seven protein
identities were analyzed. The scalar eligible negative-gradient component
explains **79.953%** of gold-action squared energy. Active starting-state safety
directions add **0.603 percentage points**, explaining only **3.007%** of the
remaining residual. Near-active directions add **0.760 percentage points**.
Mean gold cosine changes from **0.891930** to **0.898528** with active-span
correction (215 nonzero gold actions). The inequality-compatible active fit
explains **80.541%** in total.

These are uncentered action-energy fractions, not centered statistical variance.
Active/near-active thresholds are 1e-5/0.01 in frozen normalized constraint
units. Alpha is fitted nonnegative; residual NNLS uses inward Jacobian rows
`-dc/dP` with nonnegative multipliers for `c <= 0`. This is a finite-step
mechanistic diagnostic, not proof of exact nonlinear KKT optimality.

Endpoint gold action-ball normals add 17.579 percentage points in unrestricted
fits, bringing total explanation to **98.135%**. The combined compatible fit
explains only **84.545%**. Starting action-ball Jacobians are zero. Endpoint
normals use gold exclusively for diagnosis and never enter inference.

Aligned RMSD, continuous chirality, signed/inversion and assessability families
are reported separately, including solo and conditional span contributions.
Eligibility equalities are eliminated exactly; their gold residual is zero.
Radius margins/Jacobians and eligibility identity-row indices are archived in
`phase_a_auxiliary_constraints.npz`. The minimum-radius gate is inactive and
analytically redundant inside each 0.10 Å action ball. Results by identity,
corruption, exact length, length bin, and step are in `span_by`, `by`, and
`rollout_by` in `results.json`.

## Projected teacher result

Dykstra projects the desired normalized 0.10 Å negative-gradient action onto
linearized safety halfspaces and per-residue balls, with eligibility coordinates
eliminated. Each convex solve is limited to 128 sweeps. The second method adds
one endpoint relinearization while retaining the same desired total action and
balls about the original input. Proposals and scales `[1, .5, .25, .125]` receive
the historical exact nonlinear verification. The best safe local-nonworsening
candidate is retained, with exact-zero fallback.

| Method | One-step retention | Rollout retention | Median one-step latency | Archived SLSQP speedup | Useful rescues / 252 |
|---|---:|---:|---:|---:|---:|
| One projection | 15.983% | 19.066% | 92.170 ms | 2035.2× | 26 |
| Projection + one relinearization | 39.215% | 55.508% | 164.203 ms | 1068.5× | 103 |

Retention is the ratio of summed absolute mean-local-RMSE repairs. All 20
rollouts run 16 steps and compare against their available frozen gold prefix.
Speedup uses 230 existing matched solver timings; no SLSQP was rerun. Useful
rescue requires safe repair greater than 1e-8 Å. Latency includes gradient,
Jacobians, convex solves, verification and fallback, excluding Phase A/reporting.

Both methods passed all frozen gates on all 556 stored one-step actions and 640
rollout actions, independently rechecked after archival. New inversions,
assessability loss, eligibility violations, and action-cap violations are zero.
Global gates retain their historical 1e-8 normalized numerical tolerance;
quartet gates retain literal `<= 0`, and stored action norms remain `<= 0.10 Å`.

The second method has mean gold-action cosine **0.972803** on 127 nonzero-vector
pairs and mean RMS difference **0.046048 Å** across all 278 states. Its action
RMS averages **0.028775 Å**; 149 actions are exact zero. The full report includes
norm and latency distributions and aligned-RMSD ratios against the frozen
corruption baseline. Cosines on surviving nonzero actions are selective.

## Remaining gap and validation

Only 31/278 first unshrunk projected proposals passed nonlinear safety. The
first convex solve converged within its budget on 247/278 states. Across the
556 unshrunk proposals of the two-solve method, continuous chirality failed 295
times and strict quartet gates failed 384 times; 52 convex solves did not meet
the convergence rule. Failure categories overlap and rejected proposals are
never accepted merely because the convex approximation passes.

There were 414 exact failures with maximum linear violation <=1e-8. This is
approximate first-order feasibility: the convex tolerance and literal quartet
gates differ. The remaining gap therefore includes strict numerical boundary
certification, nonlinear chirality/quartet curvature, degenerating squared
assessability gradients near zero, per-residue cap geometry, and bounded convex
solver error. It cannot be attributed entirely to one curvature mechanism.
No further solver iterations or method tuning were added after measurement.

Preflight verified loss/constraint derivatives against directional finite
differences and repeated deterministic inference. Three additional analytic
tests cover a coupled halfspace/ball optimum, eligibility with relinearized
offsets, and explicitly budget-limited infeasible linearizations. All passed;
Ruff's undefined/unused-name checks passed. Completion validates input/source
hashes, archive hashes, rollout arithmetic, and every accepted action's gates.

Entry points: `scripts/run_e010_constraint_projected_v1.py` and
`scripts/report_e010_constraint_projected_v1.py`. Existing completed trajectory
files are validated and retained on restart rather than overwritten.

Frozen completion-manifest SHA256:
`6561fcb6efddaae12ae573ea9f48d66a0490bdf2d7a20d50974706038bde2689`.
Frozen full-report SHA256:
`f1c4857d7c916e4b1be3817679609056dcc9d7eb85bb070234d251c68e8c3aa9`.
Final verification checked 47 artifact hashes and 18 source hashes.
