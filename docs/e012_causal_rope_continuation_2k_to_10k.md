# E012 V2: preregistered 2k-to-10k continuation

Historical pilot f6091b6895c98d083c9e292346e0f704ed4dd523 remains immutable CSEQ-C.
This is a continuation from update 2000, not a fresh replication. Exactly 8,000
additional successful updates lead to global update 10000. The endpoint is fixed;
no checkpoint selection, scientific early stopping or extension is authorized.

## Model, data and optimizer

Restore all 15,533,952 parameters, AdamW moments, RNG and sampler from historical
checkpoint SHA256 ff666eccdb6d28d51f2d3d561242152512e2dddf74fc0f251429d55b886d0188.
Historical architecture/config, vocabulary, q/k RoPE, causal masks, requested-length
conditioning, dropout and sequence_mean loss are reused unchanged. No geometry,
structural metadata or E010/E011 hidden states. TRAIN 231743, validation 23307;
exact historical primary and independent panels of 2048 identities each. Historical
30% identity/80% coverage policy and protected hashes remain authoritative. TRAIN-only
uni/bi/trigram baselines are reused byte-identically, never rebuilt or tuned.

AdamW remains max LR .0003, betas .9/.95, decay .01, clip 1.0, BF16 without scaler.
Effective optimizer batch is exactly 64 protein exposures, never increased.

## Explicit warm restart

Let c be the actual optimizer LR in the source checkpoint (recorded in contract).
The new scheduler sets the LR used for successful global update u BEFORE opt.step:

- 2000 <= u <= 2100: c + (.0003-c)*(u-2000)/100.
- 2100 < u <= 10000: .0003*(1+cos(pi*(u-2100)/7900))/2.

The old LambdaLR is retained only as historical metadata, never resumed or stretched.
At u=2000 LR=c, at 2100 LR=.0003, at 10000 LR=0. Adam moments remain intact.

## Throughput and sampler semantics

Calibration measured every physical candidate 8/16/24/32/40/48/56/64 in all five
regimes at their upper lengths with three deterministic forward/backward repetitions,
selecting median time of repetitions 2–3. It restored the source model/optimizer;
calibration optimizer updates do not count. Two disposable FP32 eval-mode updates
verify gradient/Adam equivalence; dropout architecture is unchanged for training.
The separate disposable BF16 smoke verifies exact next-loss/weights on save/resume.

Frozen selected optimizer-batch partitions by maximum length of its 64 proteins:

| Length | Physical partition |
|---|---|
|20–64|64|
|65–128|64|
|129–256|32+32|
|257–384|40+24|
|385–500|40+24|

The next 64 sampler exposures are selected before sorting within that optimizer batch
by length and sample ID. No cross-update length bucketing changes sampling probabilities.
Each microbatch of n proteins contributes (n/64)*mean(per-protein token CE).
Remaining historical permutation/cursor are restored. At exhaustion, epoch e uses
numpy.default_rng(12012+e).permutation(231743), starting with e=1. A batch may span
an epoch boundary; all exposures are recorded. Sampling continues for exactly 512000
additional exposures; total 640000 (about 2.76 nominal population passes).
Different physical shapes change stochastic dropout assignment; this continuation
preserves scientific batch semantics but is not bitwise equivalent to legacy 8x8.

Detected VRAM and safety ceiling are recorded in calibration.json. Reserved memory
must stay under min(90% detected device memory, initial free+resident reserved−0.35GiB).
Target 85–90% is subordinate to measured throughput and exactly 64 proteins: no dummy
allocation or scientific batch increase is used to consume unused memory. Fastest
safe partitions reserve less memory than that target. No post-freeze OOM retries or
scientific retuning are allowed; operational integrity failure stops the run.

## Evaluation and interpretation

Normal likelihood/checkpoints at global 3000,5000,7500,10000; complete historical
prefix/windows/length ablation diagnostics at 5000 and 10000. Reference historical
2000 results without recomputation. Same 16 fixed positions, shuffle seeds, exact
prefix multiset/no future residues, RoPE key masking and windows 1/4/8/16/32/64/full.
Historical ten relative-position deciles are preserved, reported separately in each
length stratum at every normal boundary. This is descriptive, never a gate.
Update 3000 has no new shuffle/window endpoint; retention is adjudicated at 5000.

All Gates A–E, 10000 identity bootstrap resamples/seed12112, confirmation rules and
CSEQ classification are reused directly from protected e012.py. The empirical trigram
baseline is worse than unigram; beating it alone is not contextual evidence.

CONT-A: CSEQ-A. CONT-D: C/D confirmed, A/B confirmed, E principal blocker.
CONT-B: C/D confirmed but likelihood/stratum gates remain unmet.
CONT-C: pre-registered order/context gates no longer confirmed (report actual effect
and CI; this label alone does not establish reversal). CONT-E: operational integrity
prevents interpretation. CONT-D takes precedence over CONT-B for a principal stratum
failure. No continuation beyond 10000 is automatic.

Save immutable continuation checkpoints 3000/5000/7500/10000 with model, optimizer,
new scheduler, global update, RNG, sampler and batching-plan metadata. Verify hashes,
source/protected inputs, sampler exposures and independently reproduce intervals.
Preparation must be committed/pushed before update2001. Only small reports/text and
one verifier are tracked; datasets/checkpoints remain local ignored artifacts.
No large sampling, fold prediction, biological generation or structure conditioning.
