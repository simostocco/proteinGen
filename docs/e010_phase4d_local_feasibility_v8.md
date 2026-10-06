# E010 Phase 4D V8: K=8 local-feasibility oracle

Historical V7 commit 99c4b2393524059f3dbb757a8f1deee2f7517f57 remains immutable.
The only scientific intervention is K=4 -> K=8. All free Cartesian variables
start independently at zero; no warm start. Radius remains .04 Å per step.
Loose cumulative net/path bound is .32 Å per residue; worst-case pair-distance
change is .64 Å. These are architectural bounds, not predicted movements.

Reuse V7's exact radial map, dynamic eligibility, endpoint policy, local MSE,
proper-Kabsch float64 alignment, continuous signed chirality, and 60-example
immutable panel. Import the historical implementations rather than rewriting
losses or geometry. Only the Oracle shape changes to eight steps. Arm A has
only normalized local loss. Arm B has exactly the historical aligned RMSD
<=1.01*baseline and continuous chirality<=baseline inequalities. No beta/gamma.
Binary inversions remain telemetry and final per-example gate; no gradients.

## Numerical contract before scientific execution

All V7 solver/config values are retained, including SciPy 1.18.1 trust-constr,
CPU float64, exact objective/constraint HVPs, maxiter1000, gtol/xtol/barrier_tol
1e-12, initial barrier1e-7, one thread per worker and four deterministic workers.
No restarts, per-example or condition tuning. No changed iteration cap planned.
First validate K=8 zero-state Hessians and gradients and run two fixed synthetic
chains (12 and128 residues), both arms, with the same1000 cap. Geometry is
x=(2*i,sin(i),cos(i)), target=x+(.7*sin(1.7*i),.8*cos(1.2*i),.6*sin(2.3*i)).
These are preflight correctness/convergence cases only, never panel identities.
Record their stopping behavior; a cap failure blocks panel execution pending
an explicitly frozen synthetic-only numerical revision. Also revalidate
alignment gradients and zero-state Hessians on all60 fixed inputs before any
scientific optimization, without inspecting optimized panel metrics.

All V7 convergence thresholds, strong coverage >=48/60 and primary>=18/20,
normalized feasibility1e-8, active1e-5 and ball KKT/projection .001 are unchanged.
Finite exact post-state accounting is mandatory. Histories and stopping failures
are retained without scientific early stopping. Repeat all120 independent runs
from zero and require exact non-timing record/hash reproduction before commit.
Protect all tracked historical files and checkpoint/cache input hashes.

## Telemetry and decisions

Record P0-P8 and eight corrections for every example: local offset metrics,
Cartesian/aligned metrics, continuous chirality, inversions/assessability,
net/path RMS/max, frame assessability. Consecutive cosines are defined only
for both-eligible residues with both magnitudes>1e-12 Å; retain raw values
and report count/min/quantiles/mean/max by condition and adjacent step.
Saturation denominators are eligible corrections only, at >=90/95/99% radius.
Compare with the actual V7 cached corrections at the same thresholds.

Primary gain remains5%. Material Delta_K is preregistered as >=.5 percentage
points (descriptive threshold only; no solver tuning). Strong convergence is
required in both arms before scientific classification. Preserve V7's strict
per-example inversion gate unless the user explicitly changes the interpretation
before classification; any clarification will be recorded separately.
With strong coverage: safe Arm B gain>=5%, continuous inequalities feasible,
final assessability preserved and every per-example inversion gate passing gives
K8-A. Arm A>=5% but Arm B fails that safety/repair contract gives K8-D; distinguish
continuous constraint limitation from binary-gate failure. If both arms remain
<5% with material primary gain, K8-B; marginal gain yields K8-C. Missing strong
convergence yields K8-E regardless of observed gains. Report aggregate inversion
changes even when individual gates fail. Nonconvex stationarity is not a global
optimality certificate; a failed binary gate does not prove geometric impossibility.

Historical artifacts, global E010, E011/E012 and environment stay unchanged.
No neural training, CUDA, held-out evaluation, K16/K32, radius sweep or downstream
experiment is authorized in V8. Commit only new versioned V8 records after tests
and reproduction pass, then push the existing Phase4D branch normally.

Synthetic preflight: all four cases terminate below the cap (29/38/58/47
iterations). Three pass the full historical convergence gate. Length128 Arm A
terminates on xtol with v-space optimality6.09e-11, exceeding the retained
1e-12 criterion, despite physical normalized KKT1.20e-6 and projected residual
1.57e-6. It remains explicitly nonconverged by the historical gate; no criterion
or cap changes. This does not show an iteration-cap problem. Scientific panel
convergence will be adjudicated under the unchanged strong-coverage rules.
