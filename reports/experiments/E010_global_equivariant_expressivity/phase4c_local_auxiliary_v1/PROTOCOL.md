# E010 Phase 4C: short matched local auxiliary continuation

Pre-registered 2026-10-05, after the reviewed L1 frozen-panel diagnostic.
Public baseline: `3b200ffb0389be87cc928c391117ce96bd8b5eb2`.
The immutable generated contract pins the execution commit and source hashes.

Two arms load the identical Phase 4B update-1092 model, Adam moments and RNG state.
Arm A uses historical Cartesian component MSE plus 1e-5 residual-component MSE.
Arm B adds `0.06719407652184162 * mean(local_i+1, local_i+2, local_i+3)`.
The endpoint-masked local terms are individually normalized per protein, then
averaged equally. No alignment, chirality term, architecture modification or
all-pairs objective is introduced. Cartesian control's lambda=0 computation and
gradients exactly reproduce the historical formula.

Each arm replays exactly the first 364 *historical* cached optimizer batches,
including their five 18-protein stratum microbatches and original ordering.
This is a continuation through a second exposure to a pinned part of the existing
cache, not newly generated corruptions. Historical equal condition sampling is
retained. Actual per-batch and short-budget condition counts need not be exactly
one third; they are recorded without reweighting or forced rebalance. Both arms
receive byte-identical records, masks and targets. No augmentation or stochastic
layers are added. Full precision, AdamW LR=0.0003, betas=(0.9,0.999), epsilon=1e-8,
weight decay=0, global clip=5, constant scheduler remain unchanged.

Budget: exactly 364 continuation updates per arm, ending at Adam step 1456.
Evaluation/checkpoints: continuation updates 0,91,182,273,364. Both update-zero
models, optimizers and full evaluation metrics must match before either arm trains.
Full evaluation uses the existing 320 identities x three conditions, not the five
frozen diagnostic identities. Sequential arm execution preserves restored RNG
states, inputs and optimizer settings. Each update publishes a resumable latest
checkpoint, telemetry and progress; requested boundary checkpoints are retained.

Primary endpoint: Arm B vs Arm A at continuation update364. Paired relative
improvements are averaged within identity before 10,000 identity bootstrap
resamples, seed41046. Mean-local evaluation is the arithmetic mean of offset
RMSEs (Å), distinct from training MSE (Å²). All offsets are reported.

Pre-registered gates:

1. Positive paired mean-local improvement with 95% bootstrap CI above zero.
2. Paired Cartesian aligned-RMSD relative improvement >=-0.02 (non-inferiority).
3. No condition mean-local RMSE regression >2%; improvement in each preferred.
4. Chirality assessable and inversion no higher than matched control, preserving
   the strict historical gate. This loss does not resolve reflection symmetry.
5. All finite; no coordinate collapse (mean radius of gyration <1e-3); each
   stratum predicted radius-of-gyration SD >=5% of target SD.
6. No stratum mean-local RMSE or aligned-RMSD regression >2%. Length slopes are
   also reported, with the historical slope gate separately disclosed.

The 2% operational definition of a major condition/stratum regression is frozen
before execution. Classification precedence is C5 safety, then C2 local benefit
with Cartesian cost, C4 condition or stratum failure, C3 local gate failure,
otherwise C1. Intermediate scientific underperformance never stops an arm.
Operational stops: nonfinite loss/gradients/parameters/outputs, integrity mismatch,
catastrophic collapse or unrecoverable runtime failure. No budget extension.

Gradient telemetry uses only the pinned historical five-protein training panel,
all conditions, at evaluation boundaries; it is explanatory. Module interaction
summaries are retained at 0/364. Historical buffers and source checkpoint remain
untouched. Raw checkpoints/cache arrays and large execution artifacts are ignored.

Commands (proteingen environment, `PYTHONPATH=src:.`):

```sh
python reports/experiments/E010_global_equivariant_expressivity/phase4c_local_auxiliary_v1/runner.py prepare
python reports/experiments/E010_global_equivariant_expressivity/phase4c_local_auxiliary_v1/runner.py validate
python -u reports/experiments/E010_global_equivariant_expressivity/phase4c_local_auxiliary_v1/runner.py execute
# Only after an operational interruption, preserving the same contract:
python -u reports/experiments/E010_global_equivariant_expressivity/phase4c_local_auxiliary_v1/runner.py resume
```

The contract, schedule, data manifest, arm states and results live under the ignored
`execution/` child directory. Concise results may be published after completion;
no conclusions are written to main during execution. Full adaptation and production
remain unauthorized.
