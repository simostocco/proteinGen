# Experiment Timeline

## E000 Epoch-8 Baseline Generative Evaluation

### Motivation

E000 establishes a reproducible baseline evaluation for the selected epoch-8
unconditional distance-map diffusion checkpoint before architecture or objective
changes. The model is conditioned only on requested sequence length `N`, so
generated samples are draws from `p(D | N)` rather than reconstructions of any
matched reference structure.

### Hypothesis

The epoch-8 model should produce finite, symmetric, length-conditioned
distance maps with some protein-like geometric statistics, but performance may
vary strongly with `N` because long chains are underrepresented in training.

### Dataset And Split Provenance

- Training manifest: `data/full/splits_recovered_all_structures/train.parquet`
- Validation manifest: `data/full/splits_recovered_all_structures/validation.parquet`
- Test manifest: `data/full/splits_recovered_all_structures/test.parquet`
- Normalization: `data/full/processed_recovery/normalization_train.json`
- Training configuration: `configs/train_recovered_full_v.yaml`
- Split family: recovered all-structures split with leakage-safe sequence/PDB
  grouping.

### Selected Checkpoint

`outputs/recovered_full_b2_v/checkpoints/final_validation_selected.pt`

### Current Training-Length Distribution

| Length bin | Samples | Effective training weight |
| --- | ---: | ---: |
| 20-64 | 24,111 | 7,495 |
| 65-128 | 79,991 | 24,993 |
| 129-192 | 58,365 | 19,471 |
| 193-256 | 49,446 | 19,763 |
| 257-320 | 26,469 | 11,172 |
| 321-384 | 17,007 | 6,653 |
| 385-448 | 10,476 | 4,670 |
| 449-500 | 4,920 | 2,332 |

The 449-500 bin is 1.8169% of training samples, so E000 explicitly reports
metrics by length instead of collapsing to one score.

### Evaluation Protocol

E000 evaluates four separate properties:

- Validity: numerical, triangle-inequality, EDM, chain-like, and protein-like
  diagnostics.
- Distribution matching: generated descriptor distributions compared with
  validation or test controls of matching or nearby length.
- Diversity: within-length generated sample dissimilarity and near-duplicate
  clustering.
- Novelty: approximate two-stage nearest-neighbour comparison against training
  descriptors, calibrated with validation-to-training nearest neighbours.

The protocol records checkpoint/config/manifest hashes, runtime versions, seed
schedules, sample counts, thresholds, metric definitions, and completion state.

### Results

Completed calibrated analysis of 375 generated samples and 320 matched real
controls. All generated samples are numerically valid and non-duplicated under
the current diversity thresholds, but 0/375 pass empirical real-like geometry
thresholds derived from the 99th percentile of matched real controls. The
deprecated `edm_compatible` field admits 12/375 generated samples, all at
N=64, but those samples still have nonzero triangle violations, negative
eigenvalue mass around 0.03-0.05, and rank-3 residual around 0.04-0.10.

The principal result is that the epoch-8 model produces length-conditioned
distance-like matrices and captures some local statistics, but the generated
matrices do not lie on the empirical manifold of real three-dimensional protein
distance matrices. Global geometric inconsistency and excess compactness worsen
with sequence length.

### Limitations

- Novelty is approximate because it uses descriptor retrieval followed by
  refined comparisons against retrieved candidates.
- Classical MDS diagnostics evaluate realizability; generated matrices are not
  projected or repaired.
- Distance-map diagnostics cannot establish thermodynamic stability.
- Real controls are distributional references, not reconstruction targets.
- Checker or diamond-like motifs are not classified as artifacts in E000; their
  association with geometry rankings remains future analysis.

### Decision Criteria For E001

E001 should target the metric families that fail most clearly in E000:
negative eigenvalue mass, rank-3 residual, MDS stress, triangle consistency,
radius-of-gyration matching, and scaling with `N`. The planned E001 experiment
is `E001_symmetric_axial_attention`: add one symmetry-preserving axial-attention
block at the resolution immediately above the existing bottleneck attention,
while preserving convolutional residual blocks. E002 may add a corresponding
decoder block. Physical auxiliary losses come after the attention ablations.

## E000 Finalized Results

Final calibrated analysis completed at `2026-09-04T21:52:02.550306+00:00`.

- Generated samples: 375
- Real controls: 320
- Empirical real-like geometry pass count: 0
- Deprecated/permissive heuristic EDM-quality pass count: 49
- Raw E000 inputs unchanged during finalization: True

Conclusion: The current model produces numerically valid, non-duplicated, length-conditioned distance-like matrices and approximates several local distributional properties, but its generated matrices do not lie on the empirical manifold of real three-dimensional protein distance matrices. Global geometric inconsistency and excess compactness worsen with sequence length.

E001 remains `E001_symmetric_axial_attention`, testing whether one
symmetry-preserving axial-attention block immediately above the bottleneck
improves global geometry and scaling with `N` without reducing diversity or
increasing training-set similarity.

## E003 Adjacent Chain Geometry Finalized

Status: E003 full training, calibrated evaluation, and paired E002-versus-E003
analysis are complete. This reporting pass was documentation-only: no training,
preprocessing, splitting, sampling, descriptor rebuilding, ensemble evaluation,
checkpoint modification, commit, or push was performed.

Selected checkpoint provenance:

| Model | User-facing epoch | Stored epoch index | Optimizer step | SHA-256 |
| --- | ---: | ---: | ---: | --- |
| E002 | 5 | 4 | 169181 | `2b6b9967a5de6035accad7cc1c24e379743459531c641727221541fde9aa669f` |
| E003 | 5 | 4 | 169182 | `bdbaf25ee3b9d7d7a7cfe2ebe03a12b09b16261b1a8a3e95d28b2c48058ca101` |

E003 training evidence: best validation epoch 5 with validation loss
`0.0183565374`, compared with E002 epoch-5 validation loss `0.0185558852`.
E003 reached optimizer step `169182`, recorded `63` isolated AMP overflow
events across `252` log rows in affected four-microbatch windows, had maximum
consecutive overflows `1`, no non-finite losses, and accumulation state `0` at
the selected checkpoint. Diffusion, stochastic EDM spectral, and adjacent median
losses decreased across epochs.

In the 375-sample paired E002-to-E003 comparison, positive improvement means
E003 is lower/better. Adjacent-residue distance RMSE improved modestly by
`0.0083769942` with bootstrap CI `[0.0007474087, 0.0161266259]`. Negative
eigenvalue mass degraded by `-0.0042051035` with CI
`[-0.0050886455, -0.0033259991]`, and triangle violation fraction degraded by
`-0.0004375` with CI `[-0.0006771159, -0.0001874674]`. Rank-3 residual
(`-0.0007241938`, CI `[-0.0025555061, 0.0011522294]`) and classical MDS stress
(`0.0012353564`, CI `[-0.0004186714, 0.0028830080]`) are inconclusive.

Strict empirical real-like geometry remained `0/375` for both E002 and E003.
Heuristic validity changed from `0.128` to `0.1306667`, with E002-fail/E003-pass
transitions `9`, E002-pass/E003-fail transitions `8`, and exact McNemar
`p=1.0`.

`generated_count=0` warnings for `empirical_real_like_geometry_pass` novelty
subgroups are expected because no generated sample passed strict empirical
validity. They are not missing-data warnings.

Distribution matching is mixed: E003 moves closer to real controls in `25/60`
descriptor-length rows by standardized mean discrepancy, `24/60` by
Wasserstein distance, and `23/60` by KS statistic. Diversity does not collapse:
generated/real diversity ratios move closer to one in `13/20` rows. Novelty is
mixed and approximate, moving closer to the real calibration baseline in `6/15`
rows.

Scientific decision: E003 is not promoted to the new baseline. E002 remains the current
baseline. E003 is retained as evidence that a local adjacent-chain
objective can improve adjacent-residue geometry while degrading global EDM and
triangle consistency. Bootstrap confidence intervals quantify generated-pair
variability, not training-seed uncertainty.

Before E004, run an analysis-only diagnostic that stratifies diffusion,
stochastic EDM, and adjacent-chain gradient norms and pairwise cosine
similarities by diffusion timestep bins, requested-length bins, and optionally
their interaction. If conflict concentrates at high noise, consider
timestep/SNR-gated adjacent loss; if it concentrates at long lengths, consider
length-dependent weighting; if it is widespread, abandon the adjacent loss or
investigate a multi-objective gradient method; if no systematic conflict
appears, evaluate training-seed variability before changing the objective.

### Post-E003 Gradient Diagnostic Prepared

An analysis-only gradient-interaction mode was added to
`scripts/profile_edm_auxiliary_gradients.py`. It supports fixed timesteps
`0, 50, 100, 200, 300, 400, 450, 475, 490, 499` and the established recovered
training length bins 20-64, 65-128, 129-192, 193-256, 257-320, 321-384,
385-448, and 449-500. The mode writes observations, grouped summaries, heatmaps,
and a README under
`reports/experiments/E003_adjacent_chain_geometry/gradient_interaction_diagnostic/`
when run by the user.

For each available length-bin/timestep combination, the diagnostic computes
separate gradients for diffusion loss, weighted stochastic EDM spectral loss,
and weighted adjacent-chain loss over the same trainable parameter set. It
reports gradient-norm ratios, pairwise cosines, raw losses, eligibility counts,
actual lengths, sample ids, fixed timestep, and seed. Parameters with missing
gradients are zero-filled consistently in every objective vector.

This is not E004 and does not select a new objective. It is intended to decide
among conditional directions: timestep/SNR gating if conflict concentrates at
high noise, length-aware weighting if conflict concentrates in long proteins,
joint timestep-length scheduling if both concentrate, abandoning adjacent loss
or using multi-objective gradient handling if conflict is widespread, or
training-seed variability checks if no systematic conflict appears.

## E004 Symmetric Triangle Multiplication Preflight Finalized

Status: implementation, CUDA stress, 2,000-step preflight, paired step-2,000
screen, timestep diagnostic, and definitive five-epoch configuration are
complete. This update was documentation/configuration-only: no preprocessing,
splitting, training, sampling, ensemble evaluation, checkpoint modification,
generated-sample modification, descriptor-cache rebuild, commit, or push was
performed.

E004 uses E002 as the parent baseline and does not enable the E003 adjacent
auxiliary loss. It preserves the E001/E002 U-Net, v-prediction, bottleneck full
attention, pre-bottleneck symmetric axial attention, stochastic EDM spectral
loss at weight `0.01`, 500-step warmup, subset size `64`, one subset per sample,
optimizer, scheduler, EMA, diffusion schedule, data, batching, masking,
normalization, and AMP settings.

The controlled architectural intervention is exactly one symmetric
triangle-multiplicative update after the existing pre-bottleneck axial block and
before downsampling into the bottleneck. In the production config this is
encoder level `2`, block index `1`, with `96` feature channels. For requested
N=500 the padded side is `504`, so the triangle block runs at side `126`.

E004 parameter counts are E002 `7,557,681` versus E004 `7,582,833`, a delta of
`25,152`. Implementation commit: `cd31047`. Triangle multiplication is not claimed as novel: AlphaFold uses
triangle multiplicative updates and triangle attention, and Proteus already uses
graph-based triangle operations in protein backbone diffusion. E004 is an
architectural baseline/ablation in this project.

Preflight config:
`configs/train_recovered_full_v_axial_edm_triangle_e004.yaml`, with output
directory `outputs/recovered_full_b2_v_axial_edm_triangle_e004_preflight`.

Definitive full config:
`configs/train_recovered_full_v_axial_edm_triangle_e004_full.yaml`, with output
directory `outputs/recovered_full_b2_v_axial_edm_triangle_e004`.

Worst-case CUDA stress covered N=495-500 with batch size 2 and gradient
accumulation 4 for 10 optimizer steps plus resume to step 11. Peak allocated
memory was `4.22 GiB`; peak reserved memory was `7.64 GiB` initially and
`6.54 GiB` on resume. There were zero AMP overflows, zero skipped updates, clean
accumulation state, successful resume, and approximate resume throughput
`9.22 samples/s`.

The full-data preflight completed `2000` optimizer steps and `8008` JSONL
records. Step-2000 checkpoint SHA-256:
`ecd1b34780e74bf2f367bf2ced884e57e8a6ec3233f6b75f29c798cdbca80d01`. It had
`2` AMP overflow events, maximum consecutive overflows `1`, `8` skipped-window
records, no non-finite losses, no unexpected non-finite gradients, final
consecutive overflows `0`, clean microbatch accumulation state, and final
throughput `23.8145 samples/s`.

First-versus-last quartile medians improved for diffusion loss `0.053101562 ->
0.022458778`, optimization loss `0.054552520 -> 0.023218437`, EDM total
`0.30521363 -> 0.058729568`, EDM negative `0.12913704 -> 0.018167607`, and EDM
rank3 `0.17434631 -> 0.037178006`.

The paired step-2000 screen used two samples per length at N=64, 128, 256, 384,
and 500, with matching seeds and sample indices. Positive means E004 is
lower/better than E002, and projected matrices were not used for the comparison.
Overall, E004 improved triangle-violation fraction by `0.0009765625` in `7/10`
samples, negative-eigenvalue mass by `0.026845372` in `7/10`, rank3 residual by
`0.066621558` in `10/10`, and adjacent RMSE to 3.8 A by `0.66738602` in `8/10`.
Negative-distance fraction changed by `-0.0001035543` while improving in `7/10`.

Per-length negative-eigenvalue improvements were N64 `-0.019554`, N128
`0.003792`, N256 `0.038827`, N384 `0.053112`, and N500 `0.058049`.
Per-length rank3 improvements were N64 `0.052774`, N128 `0.076017`, N256
`0.055579`, N384 `0.075206`, and N500 `0.073532`. Per-length triangle
improvements were N64 `-0.007812`, N128 `-0.000488`, N256 `0.002441`, N384
`0.005859`, and N500 `0.004883`. Per-length adjacent-RMSE improvements were N64
`-0.597252`, N128 `1.205684`, N256 `0.858173`, N384 `0.894127`, and N500
`0.976197`.

The timestep diagnostic showed single-step `x0` reconstruction RMSE improvement
at 8/9 reported timesteps, with mean improvement `0.19944867 A`. Improvements
were t=0 `0.009868`, t=100 `0.330156`, t=200 `0.282385`, t=300 `0.243457`,
t=400 `0.391255`, t=450 `0.029041`, t=475 `0.261914`, t=490 `0.352500`, and
t=499 `-0.105538`. Target-MSE was slightly worse at t=450 by `0.000511` and
t=499 by `0.001043`. Negative reconstructed physical fraction at t=499 changed
from E002 `0.000924` to E004 `0.004554`, a `-0.003630` change.

E004 passes the five-epoch training gate because it produces a coherent, large
improvement in global geometric metrics, especially for N>=256. Caveats: the
screen has only two samples per length, N64 regresses in several metrics, no
strict-validity conclusion can be made from ten samples, and the isolated t=499
negative-distance caveat should be retained. No special t=499 correction should
be added before the controlled full experiment. E002 remains the baseline until
the full E004 ensemble is evaluated.
