# E010 student feasibility margin v1

Completed offline negative feasibility pilot, separate from frozen student v1
commit fc76211. Teacher/corpus 6d1ed30 and projected method 73f3330 remain unchanged.
All generated artifacts reside at:
`/mnt/d/Simone/proteinGen/outputs/e010_student_feasibility_margin_v1`.

The original 300 validation and 300 test identities (all project TRAIN) retain
their frozen identity/group-isolated splits. Three conditions and sixteen logical
states/steps give 14,400 one-step and recursive states per split per alpha.
The fixed global alpha ladder is 1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2. No X is
used in network inputs, global scaling, or an inference shield. X scores results
offline. No retraining, new teacher data, capacity increase, or SLSQP solve.

| Test alpha | One-step repair retention | K16 repair retention | Unsafe K16 states |
|---:|---:|---:|---:|
| 1.0 | 59.08% | 59.61% | 11930/14400 |
| 0.9 | 53.20% | 53.80% | 11748/14400 |
| 0.8 | 47.31% | 47.92% | 11563/14400 |
| 0.7 | 41.41% | 41.99% | 11315/14400 |
| 0.6 | 35.51% | 36.03% | 11013/14400 |
| 0.5 | 29.60% | 30.03% | 10674/14400 |
| 0.4 | 23.69% | 24.02% | 10229/14400 |
| 0.3 | 17.78% | 18.00% | 9625/14400 |
| 0.2 | 11.85% | 11.98% | 8727/14400 |

Retention compares summed absolute mean-local-RMSE repair with the archived
original privileged teacher. Persistent inversions are counted per state, and
constraint-family failure counts overlap. Exact slacks, action errors, and
error-direction finite differences are archived per transition.

Teacher actions within 1e-5 of any native normalized hard boundary:
19.90% of nonzero test teacher
states. Error-magnitude failure AUC: 0.337;
directional error-to-margin failure AUC: 0.893.
These are mechanistic associations, not causal proof or exact boundary distances.

The Phase C gate was frozen before new test scoring: validation alpha 0.5 must
halve the failed-state rate and retain at least 25% original-teacher repair.
Observed validation failure reduction: 10.56%;
retention: 30.19%. The gate failed, so no
marginized targets or retraining were performed. The preregistered unused positive
buffer ladder is 0.001/0.01/0.05 in native normalized inequality units.

Global shrinking does not produce a deployable repair/safety point on the tested
ladder. This does not establish that marginized training cannot help; alphas below
0.2 are untested. The X-based shield from student v1 is explicitly undeployable.

Validation: all 933 frozen student artifacts,
10022 teacher artifacts, and 18 geometry sources
remain unchanged; alpha-1 archived baseline parity; independent reconstruction
of 4320 states across both splits,
all length bins and all alphas; source lint passed.

`REPORT.md` contains the full Pareto table, family margins, failure discrimination,
selected strata, verification and scope. `results_AB.json` contains all alpha,
length, condition and step aggregates. `records/` contains identity results;
`geometry/` contains exact inequality snapshots and directional derivatives;
`actions/` contains reconstructible recursive actions. `protocol.json` and
`parent_freeze.json` pin the protocol, model and split hashes. `COMPLETED.json`
hashes the generated scientific artifacts and frozen source snapshots.
