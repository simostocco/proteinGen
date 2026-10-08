# E010 structural-teacher diversity scale-out pilot v1

This experiment evaluates the provisional one-relinearization privileged teacher
from commit `73f3330` on a diverse structural TRAIN cohort. The completed
278-transition constraint-projected experiment remains frozen: 39.21% one-step
and 55.51% rollout retention, 103/252 unsafe-gradient rescues, all frozen gates
passed, approximately 1068× archived-SLSQP speedup. Its starting active safety
span explains only 3.01% of post-scalar-descent residual. This pilot does not
optimize that gold benchmark or add projected-solver complexity.

All generated artifacts live under:

`/mnt/d/Simone/proteinGen/outputs/e010_structural_teacher_diversity_scaleout_v1`

The selection contains exactly 2,000 eligible TRAIN identities, 400 per existing
length bin, and conditions 50, 250, and 450 for every identity: 6,000 trajectories.
It spans 1,517 sequence clusters and 954 split groups. Deterministic selection
round-robins existing experimental-method/resolution cells and sequence-cluster
representatives, with SHA256 rankings. The earlier 20-identity Phase4D panel is
excluded. All selected identities belong to TRAIN; no validation/test identity
or split group is selected. Five candidates failed the frozen all-quartets-
assessable input requirement and were replaced without repair-outcome filtering.

The identity/split-group reporting portions are reference (825 identities),
holdout_A (279), and holdout_B (896). All states and conditions of one identity
and split group remain together. Large split groups make identity counts differ
from the 60/20/20 hash-bucket allocation. These are reporting portions of TRAIN,
not the structural validation/test splits, and none is used to tune the teacher.

Existing TRAIN denoiser-cache bytes and the frozen Phase4B global checkpoint are
hash-pinned. P0 uses the historical CPU float32, two-thread, single-example
global inference path. A preflight corrected input-inference thread count from
one to two after finding that only two threads reproduced archived P0 bitwise.
Three archived examples passed exact P0 and frozen-teacher action parity. The
teacher itself remains CPU float64 on one thread, with its unchanged source
SHA256 `0fb50ebe07d261d00845d46a43b0847aa3d855d4eec4250094cce80d06d12d33`.

The rollout uses at most 16 structural steps. The only early termination is an
exact-zero action: the state and time-independent deterministic teacher then
repeat. No near-zero termination, projection threshold change, new SLSQP solve,
new model, or student training is introduced. The teacher may see target X;
future student features must remain non-privileged.

Compact identity inputs store three P0/source arrays, one target and one mask.
Compact action archives store accepted deltas, eligibility and condition
offsets, without redundant P_t arrays. Reconstruct states by casting P0 to
float64 and adding deltas sequentially. Step records include objectives, aligned
RMSD, chirality/inversions/assessability, norms, solver/shield outcomes and timing.
All accepted states receive online exact checks and independent reconstruction
checks. Global gates retain their frozen 1e-8 normalized tolerance; quartet
gates retain literal nonpositive inequalities and action norms remain <=0.10 Å.

During generation, an archive validator incorrectly compared tuple-valued slice
metadata against JSON lists. The validator now compares canonical JSON values.
The 161 saved identities from that run were preserved and independently
validated, with the prior runner and amendment recorded on D:. The correction
changes archive validation and restart bookkeeping, not teacher mathematics or
accepted actions. Validation certificates are stored separately for every
identity.

The decision gate was recorded before generation: zero hard-safety violations;
at least 70% useful identities and 5% mean relative local-MSE repair in both
held-out portions; at least 50% useful identities and 1% mean repair in every
held-out length/condition cell; projected cache-covered teacher wall time at
observed eight-worker throughput <=24 hours. Useful means a nonzero correction
and normalized local-MSE rollout repair >1e-6, a reporting threshold only.

The report covers horizons 1/2/4/8/12/16, late benefit after 4/8/12, action and
zero-tail distributions, safety, length/condition cells, CPU/GPU timing and
length-reweighted corpus cost. Wider TRAIN cost is conditional on input
eligibility and excludes producing missing denoiser-cache entries. If the gate
passes, only this pilot corpus is frozen for a separately authorized student
pilot. If it fails, the fixed method stops with a diagnosis of failed strata.
Neither branch automatically generates the full corpus or trains a student.

Entry points are `scripts/e010_diversity_scaleout_v1.py` (`test`, `prepare`,
`run`) and `scripts/report_e010_diversity_scaleout_v1.py`. Completed input and
trajectory checkpoints are validated and retained on restart. The exact
selection, protocol, amendments, source snapshots, results, and completion
manifest are all D:-resident.

The pilot is completed and the preregistered gate passed. All 2,000 identities
received useful repair in at least one condition, and 96.30% of the 6,000
identity/condition trajectories received useful repair. Mean relative local-MSE
reduction was 35.146%; the ratio-of-sums reduction was 24.599%. All accepted
steps passed online and independent reconstruction checks: zero new inversions,
assessability losses, cap violations, eligibility violations, exact safety
failures, or local-objective increases. Mean aligned-RMSD change was -0.050955 Å;
the action cap remained exactly <=0.10 Å. The existing aligned-RMSD allowance
and normalized numerical tolerance were retained, without tightening or
relaxing the frozen safety gate.

| Steps | Mean relative local-MSE repair |
|---|---:|
| 1 | 4.992% |
| 2 | 9.246% |
| 4 | 16.394% |
| 8 | 26.525% |
| 12 | 32.269% |
| 16 | 35.146% |

The five length bins, from shortest to longest, achieved mean repair of
48.211%, 39.536%, 34.195%, 28.258%, and 25.530%. Conditions 50/250/450 achieved
59.791%/23.048%/22.599%. Both held-out portions had 100% useful identities,
with mean repair 35.486% and 35.720%; every held-out length/condition cell
passed. No collapsed stratum was observed under the recorded criteria.

Steps after 4/8/12 supplied 55.742%/26.142%/8.729% of total aggregate MSE
repair, for 67.944%/37.979%/13.160% of measured inference cost. K=16 therefore
retains measurable benefit; no horizon change was made. There were 2,130
trajectories with no exact zero through step 16, and 21,258 implicit zero-tail
steps. First-step exact zeros occurred in 222 trajectories.

Observed generation wall time was 2.599 hours, including the interrupted
validator run and independent archive verification; completed worker CPU plus
checkpoint revalidation totaled 18.857 hours. Unrecorded interrupted in-flight
CPU work is excluded from that CPU sum. P0/eligibility preparation took 0.416
hours; GPU time was zero. Observed throughput was 769.38 identities/hour or
2,308.14 condition trajectories/hour. Length-reweighted teacher-only cost is
12.900 hours for 11,207 all-condition cached TRAIN candidates and 26.499 hours
for the 30,104-identity TRAIN catalog at the observed eight-worker throughput.
Additional P0 preparation is projected at 2.008/3.893 hours respectively.
Catalog estimates remain conditional on compatible eligibility and exclude
producing absent denoiser-cache entries; the full eligible count outside the
pilot has not been established.

`CORPUS_FROZEN.json` freezes only this 2,000-identity pilot for a separately
authorized non-privileged student pilot. No student was trained and no full
corpus was generated. `REPORT.md`, `results.json`, and `COMPLETED.json` under
the D:-resident experiment root contain the complete measurements, grouped
diagnostics, source lineage, exact selection, and artifact hashes.

Final SHA256 pins:

- `selection.json`: `82c8f1f2bfc3bb9edfe0652db987570a2f33d15da86cc920737736c76705ae1b`
- `REPORT.md`: `5f678e9680d60ce881bd55bccc0c921bebc8a3e960a042561c08c462be0a9282`
- `COMPLETED.json`: `f81c030b4b99f2c259b4d57ca28f9f7665e94d14ad631ff8e2d96bd42f519454`
