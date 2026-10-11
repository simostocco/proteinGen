# E015 forced-shutdown recovery and serialized absolute evaluation

The recovery restored durable update 5676 / 172609 exposures from `recovery_0.pt`
(SHA256 `0af2e6cb43db303e6d0f22debb8ded49d962ff7150405a729d490cffd3a19a71`).
The pre-shutdown live state contained 113 additional uncommitted exposures.
These were recomputed. The first operation was update 5677 / 172627 exposures.
The first emitted resumed heartbeat, update 5680, exactly reproduced the saved
loss groups, PCGrad telemetry, timestep histograms and conditioning counters.
This comparison establishes telemetry equality, not an unavailable live-model
byte comparison.

All six initial checkpoint candidates passed schedule/accounting, strict
raw/EMA/optimizer, scaler, RNG, population and configuration checks. Checkpoint
selection used matching LATEST and COMMITTED metadata. An immutable copy of the
selected recovery checkpoint preserved the secondary scientific context.

The initial trainer was observed for 663 seconds before any continuation
evaluation. It subsequently reached epoch 1 while the pause command awaited
execution. The existing built-in epoch diagnostics finished under the same
trainer before its existing graceful stop handler returned. No second model
or external evaluation overlapped that process.

The clean pause checkpoint was `guard_stop_update_7599.pt`, epoch 2 cursor 0,
230132 exposures, SHA256
`2422308e66ff7167168923ee1d10ec0f253226a6153b876119bd35a9cb656adc`.
Both original processes exited; CUDA ownership disappeared; state, LATEST and
both COMMITTED files agreed, with zero uncommitted exposures.

## Scientific continuation

Primary checkpoint: `epoch_1_fraction_0.50.pt`, update 3811 / 115066 exposures,
SHA256 `2986d44ed22aa1c2fd0dea2fe57fa1b50c0808893fd3608142240841e08cbc03`.

The continuation reused frozen identities, matched lengths, deterministic
noise families and valid scalar results. Tier 1 used 20 TRAIN and 20 held-out
proteins; Tier 2 used 50 and 50, balanced across five length strata. Both raw
and EMA were evaluated at timesteps 5, 50, 250, 400 and 490, with initialization,
25%, 50% and the immutable recovery snapshot. Protein-level stratified
bootstrap used 2000 draws from scalar tables. No whole-panel coordinate or
activation preload was introduced.

Raw held-out aggregate v-MSE was 0.447502 versus EMA 0.748252. Raw weights were
selected using this frozen reconstruction criterion. TRAIN/held-out local
RMSE was 2.69145/2.67021 Å and torsion error was 1.13239/1.13805 radians.
The TRAIN mean v-MSE includes a large long-protein error; median errors are
0.458376/0.444409. All outliers remain included. Both splits retain substantial
geometry errors, supporting reconstruction underfitting.

The six-sample full-generation pilot passed finiteness and host stability.
It was extended to 20 primary raw samples (four seeds at each of lengths
48, 96, 192, 320 and 448), plus five secondary EMA samples. All 25 completed
the production 500-step deterministic sampler with finite coordinates. Only
the eleven requested sparse trajectory frames were saved.

Primary samples had 63.26% of adjacent distances in the declared 3.4–4.2 Å
range and mean clash fraction 0.001089. No sample met the descriptive gross
Rg-collapse threshold. At lengths 192, 320 and 448, invariant descriptor
energy/MMD/sliced-Wasserstein discrepancies exceeded the small disjoint real
reference floor. Four primary samples per length are insufficient to establish
the complete distribution. Diversity showed no exact duplicates; bounded
novelty searches cannot rule out memorization against the complete corpus.

Verdicts: reconstruction `PROMISING_BUT_IMMATURE`, generation
`BOUNDED_BUT_NOT_PROTEIN_LIKE`, generalization `UNDERFIT`, continued training
`CONTINUE_BUT_WATCH`. This checkpoint improves one-step denoising and produces
finite chains with some useful local geometry, but does not yet demonstrate a
generalizable protein-geometry distribution or useful protein-like generation.
No scientific protocol was automatically changed.

## Operations and artifacts

The five `e015_serial_*.py` helpers live under `docs/`, outside production
science source hashes. They implement read-only auditing, resource admission,
one-owner serialization, bounded evaluation, scalar reporting and recovery.
The pre-existing untracked `e015_absolute_evaluation.py` was preserved.

Evaluation ran at batch size 1, zero DataLoader workers, one BLAS/OpenMP thread
and CPU nice level at least 10. Compact resource samples were recorded every
30 seconds. Minimum observed RAM exceeded 9 GiB, swap remained unused, and
no evaluation resource-pressure stop occurred. CPU temperature sensors were
unavailable; available GPU temperatures were recorded. WSL disk-write counters
may omit physical DrvFS host writes, so raw counters and that limitation are
retained.

After evaluator exit and GPU release, the pause checkpoint was verified again.
The exact supervisor resume command restored update 7600 / 230153 exposures,
matching the next deterministic epoch-2 operation, with finite loss and BF16
plus FP32 geometry. A dedicated observer verifies ongoing heartbeat freshness
and the next rolling checkpoint before recording final completion.

```bash
/home/simostocco/miniforge3/envs/proteingen/bin/python -B /mnt/d/Simone/proteinGen/.e015_worktree/scripts/supervise_e015_training.py --resume
```

The authoritative final report, manifests, scalar tables, compact head/state
statistics and numerically ranked galleries are under:

`/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/evaluations/absolute_generation_u3811_2986d44e_v1/`

The initial recovery evidence is under `reports/shutdown_recovery_20261011T160610Z/`
in the same production root. Final resumed-process evidence and rolling-checkpoint
verification are in the evaluation folder's `logs/final_resume.json`.

All generated files, caches and temporary artifacts stayed on D:. E007 remained
paused; E012 and E014 were untouched. No sequence-model training was started.
