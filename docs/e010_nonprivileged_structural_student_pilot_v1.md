# E010 non-privileged structural student pilot v1

The teacher and diversity pilot from `6d1ed30` remain frozen unchanged. This
separate experiment uses only their 2,000 project-level TRAIN identities and
6,000 trajectories. No SLSQP, teacher recomputation, full-corpus generation,
sequence input, or automatic larger model is permitted.

All generated artifacts reside at
`/mnt/d/Simone/proteinGen/outputs/e010_nonprivileged_structural_student_pilot_v1`.
Temporary directories, CUDA and Torch caches, checkpoints, logs, evaluation
archives and reports are explicitly D:-resident.

The deterministic whole-split-group split has 1,400 training, 300 validation
and 300 test identities, with 280/60/60 per length bin. All identity conditions,
steps and states remain together, and no split group crosses portions. These
are portions of the frozen TRAIN pilot, never project-level validation/test
identities. Exact lists and their SHA256 hashes are in `splits.json`.
Sequence-cluster overlap between every pair of splits was checked and is zero.

The network receives only current P_t and mask, using the existing 41-channel
Phase4D local invariant/parity-sensitive representation, local frames and
LocalGeometryBranch blocks. Width 64 and two blocks provide the small capacity.
No target X, target-derived feature, teacher gradient, privileged constraint,
sequence, step index or corruption condition enters forward. P_t is float64;
the network is float32 and its equivariant action uses the float64 frame. A
differentiable per-residue cap and the frozen closed-ball ULP guard enforce
exact <=0.10 Å inference actions and eligibility zeros.

Sampling is uniform identity first, then condition, then logical step 0..15.
Archived exact zero tails are expanded logically without relabeling. Loss is
normalized over eligible residues within each sampled protein before batch
averaging. The two controlled arms have identical seed, initialization,
optimizer, sampling and capacity: action imitation only, or action imitation
plus exactly 0.1 times same-state-normalized resulting local-distance MSE.
Target X occurs only inside the auxiliary loss and evaluation. No aligned or
chirality training loss is mixed into this comparison.

Training uses AdamW lr 3e-4, weight decay 1e-4, batch eight, gradient clip one,
and at most 6,000 updates. Every 1,000 updates, a preregistered 60-identity
validation monitor evaluates shielded recursive K=16 teacher-repair retention.
Improvement must exceed 0.002; two stale checks stop training after a minimum
3,000 updates. Action loss does not select checkpoints. Full validation
selects between the early-stopped arms; only that model is tested. A conditional
same-capacity 32-state TRAIN-only memorization diagnostic runs for 1,000 updates
if selected-model training-probe error is at least half the zero-action baseline;
it is never selected or counted as a held-out result. No larger model runs.

The exact shield retains the frozen projection_2 scales 1/.5/.25/.125 and
acceptance block: choose the lowest strict-improving local objective among
exact-safe proposals, otherwise exact zero. P0-anchored aligned-RMSD,
continuous chirality, quartet, no-new-inversion, assessability and cap gates
are unchanged. No first-order projection or teacher solve runs for the student.
This shield requires X and is privileged evaluation; the network remains
target-free. Raw results are reported independently, so shield-dependent
safety cannot establish a deployable target-free safety guarantee.

The recorded gate requires >=75% shielded rollout mean-local-RMSE repair
retention, >=50% raw retention, zero shielded hard violations, <=50% shield
intervention, <=10% rejection of nonzero actions to zero, >=50% retention in
every length/condition cell, median CPU network-plus-geometry <=5 ms, and p95
shielded inference <=100 ms. Near-zero actions use a 1e-8 Å threshold.
One-step comparisons cover all 16 logical teacher states, while recursive
student rollouts use their own states without new teacher labels.

Entry points are `scripts/e010_nonprivileged_student_v1.py` and
`scripts/report_e010_nonprivileged_student_v1.py`. Preflight tests cover target
isolation, rotation/translation equivariance, cap/eligibility and deterministic
exact shielding. Final shielded action archives are independently reconstructed
and checked. The original parent completion and every pinned corpus file are
reverified at closeout. A success would justify a proposal for separately
authorized full TRAIN generation; failure stops with diagnosis. Neither branch
automatically generates that corpus or trains a larger model.

The experiment completed as a negative student pilot. Both arms stopped at
3,000 updates and selected the 1,000-update checkpoint by held-out retention.
Each has 118,723 parameters. Full validation selected imitation-only: raw /
shielded rollout retention was 59.93% / 12.71%, versus 60.76% / 12.68% for the
0.1-local-loss arm. The auxiliary term did not improve feasible repair.

The untouched 300-identity, 900-trajectory test gave:

| Metric | Raw | Shielded |
|---|---:|---:|
| One-step mean-local-RMSE repair retention | 59.083% | 13.693% |
| Recursive K=16 RMSE repair retention | 59.608% | 12.547% |
| Recursive local-MSE repair retention | 65.210% | 9.669% |
| Mean relative local-MSE repair | 22.483% | 6.526% |
| Teacher-action cosine | 0.780833 | 0.774320 |
| Teacher-action RMS error | 0.032000 Å | 0.032846 Å |
| Mean aligned-RMSD change | -0.018781 Å | -0.014302 Å |
| Exact-zero / near-zero action rate | 0% / 0% | 78.33% / 78.33% |
| Median CPU geometry + network | 3.942 ms | 3.932 ms |
| Median / p95 full inference | 3.969 / 6.221 ms | 36.298 / 51.578 ms |

Shield intervention was 83.44%, including 78.33% rejection of nonzero actions
to zero. All shielded one-step and rollout hard gates passed, with zero cap or
eligibility violations. Raw rollouts had 11,930 unsafe states out of 14,400:
10,639 failed strict quartets, 5,480 continuous chirality, 57 aligned RMSD,
10,637 states had new inversions and 27 assessability loss (families overlap).
The summed inversion count across steps was 50,346; this counts persistent
inversions again at later states. Raw caps and eligibility remained valid.

The gate failed retention, intervention, rejection and stratum stability;
latency, raw repair and shielded hard safety passed. Shielded retention across
length bins was 6.58–15.97%; conditions 50/250/450 retained 22.64%/4.65%/6.98%.
Every length/condition cell fell below the recorded 50% retention criterion.
The report contains all cell and identity results and all 16 step diagnostics.

The model learned useful directions but did not reproduce constraint-aware
feasible corrections. Raw one-step and recursive retention were similar, as
were shielded one-step and recursive retention, so recursive drift alone does
not explain the gap. Training/validation action residuals at the selected
checkpoint were 63.69%/65.00% of the zero-action baseline; there is little
ordinary split overfitting. A same-capacity 32-state memorization diagnostic
reduced residual from 74.78% to 43.56% after 1,000 updates but did not fully fit
the states. Capacity and optimization therefore remain plausible contributors.
Missing X, P0-anchored safety context and global information in the local
representation are also plausible; this pilot does not prove target-free
observability impossible or show that more labels/larger models alone suffice.

The pilot is frozen completed, with the full original parent reverified
unchanged. Full TRAIN teacher-data generation is not justified by this result.
No larger model, teacher rerun or new SLSQP solve followed the failed gate.

Final identity-list hashes use sorted UTF-8 identities, one per line with a
trailing newline:

- TRAIN (1,400): `499ca60a68c5938589b440cbc1269f7a2e4668050f5f8eacc0b5ad10a97ccb07`
- Validation (300): `b19690285c89b23d09530eecd3c6a8990a85985665b10b0f05675c6c97ed9ba0`
- Test (300): `b1ec1b3dd86060044ae922f3e617df8285cd6fa38e870b563fc27444a908bfd2`

Final artifact SHA256 pins:

- `splits.json`: `9d1a5ed5a30452729dee1bfb60b5520c432ca8e7a218de4792b798441f421d4c`
- `REPORT.md`: `ed70f76317506c08c311663b3f1ae9a93a26bc7f40e7c08a8f2b5a5b204c722d`
- `COMPLETED.json`: `8113a463cd66b2baa63433a3f473ffc11bda96b977ee2336c74185257a53d601`

All 933 completion-pinned student artifacts verified. The original 10,022
parent-pinned artifacts and frozen geometry source hashes also verified
unchanged. Preflight checks and source lint passed; final shielded trajectories
were independently reconstructed and checked.
