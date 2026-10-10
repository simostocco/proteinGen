# Matched full-data E007 architecture control

The owner explicitly authorized this control on 2026-10-10. This supersedes the
old requirement to await E014's final-stop report and resource-approval file;
neither historical gate record is fabricated or edited. E014 and corrected E012
remain read-only to this workflow.

The model is unchanged historical `EquivariantPairCoordinateUNet`, with 7,586,505
parameters, 16 RBF channels, unsigned chain separation, adjacency, chain continuity
and biological pair masking. Its symmetric scalar coefficients lift along unit
pair vectors with inverse square-root degree normalization and centered output.
It has no persistent vector state, local parity branch, or sequence conditioning.

The control uses exact E014 TRAIN (230,132 identities), population hash
`d3f2ee345d0e4e9a27d0136aad1eca9dfe87821a3a82d90aa4c3194bd928a3f9`,
the same deterministic complete-epoch schedules, per-identity/epoch corruptions,
500-step coordinate-v diffusion and verified scale 12.22820347644835 Å.

The frozen matched-control specification predeclares E014's optimizer and loss
settings: AdamW LR 0.0002, weight decay 0.01, betas 0.9/0.999, epsilon 1e-8,
equal-protein v-loss, clipping at 1, no scheduler/EMA/AMP, TF32 disabled. These
explicitly differ from historical E007 LR 0.0003 and coordinate-pooled batch loss;
there is no validation-based retuning. Physical/effective batches are 4/2/1/1/1
by stratum, accumulation 1. Initialization is from scratch with shared seed 14014.

CUDA preflight includes actual AdamW state and five disposable optimizer steps
at each N=64/128/256/500. The default allocator failed a 512 MiB residual VRAM
reserve. Expandable CUDA segments passed without changing scientific settings:
N=500 allocated 5,520,100,864 bytes, reserved 6,046,089,216 bytes, measured residual
996,290,560 bytes. The old 5 GiB comfort criterion remains recorded as unmet;
the authorized production uses the explicitly recorded actual-device safety
test. Both profiles are preserved. No model redesign or long-protein dropping
is used to obtain a fit.

All outputs/caches/temp files are under the D-resident worktree directory
`e007_matched_fullscale_v1/`. Production uses one CPU thread and nice priority 10.
The corrected E012 protected-verification native command timings provide the
read-only contention baseline. Persistent material degradation, stale telemetry,
unexpected active phase changes, or E012 training initiation pause E007 itself.

Quarter-epoch and full-epoch checkpoints preserve model, AdamW, all RNG states,
epoch schedule/cursor, per-identity exposure counts, next batch and source/config
hashes. Alternating rolling checkpoints are fully reloaded, compared and hashed
before committing their pointer. Exact resume rejects skipped/duplicated exposure
counts, mismatched next batch, altered sources/config, or invalid checkpoint hash.
Provisional training-log rows beyond the committed checkpoint are not durable
exposure and may be repeated after a reboot.

The detached supervisor runs epoch 1, frozen diagnostics, exact resume to epoch 2,
then epoch-2 diagnostics and the paired comparison. It never starts epoch 3.
All fraction boundaries use the frozen 500-protein panel and deterministic noise.
Epoch diagnostics reuse the validated dense-grid/marginal/Gaussian protocols;
prefixes stop at t=350. The prospective population is not inspected. The primary
comparison uses E014 epoch 2 SHA
`61a3acfe1024c9a6adcb21bfd864b3d0980e978d2fb90cbb02d9f1d095b9d902`
and the six specified primary statistics with 5,000 paired bootstrap draws.
Conservative directional classification is recorded at epoch 2; mixed/unresolved
results require scientific review, and nonsignificance is not equivalence.

Launch: `python -B scripts/supervise_e007_fullscale.py --launch`.
After a reboot or an explicitly addressed pause, use the same command with
`--resume`. A failed child is not silently restarted. Owner locks prevent two
trainers/supervisors from running concurrently. The control's `STATE.json`,
`COMMITTED.json`, `SUPERVISOR.json` and logs distinguish live, durable and
evaluation progress. Scientific results remain pending until actual boundaries
are completed; no full t=0 sampling is authorized or launched.
