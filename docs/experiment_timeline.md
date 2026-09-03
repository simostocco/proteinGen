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

Final calibrated analysis completed at `2026-09-03T11:42:59.388298+00:00`.

- Generated samples: 375
- Real controls: 320
- Empirical real-like geometry pass count: 0
- Deprecated/permissive heuristic EDM-quality pass count: 48
- Raw E000 inputs unchanged during finalization: True

Conclusion: The current model produces numerically valid, non-duplicated, length-conditioned distance-like matrices and approximates several local distributional properties, but its generated matrices do not lie on the empirical manifold of real three-dimensional protein distance matrices. Global geometric inconsistency and excess compactness worsen with sequence length.

E001 remains `E001_symmetric_axial_attention`, testing whether one
symmetry-preserving axial-attention block immediately above the bottleneck
improves global geometry and scaling with `N` without reducing diversity or
increasing training-set similarity.

## E002 Stochastic EDM Spectral Auxiliary Loss

Status: full training, calibrated evaluation, and 375-sample paired
E001-versus-E002 comparison completed.

E002 preserves the E001 architecture, recovered full-data splits,
normalization, batch size 2, gradient accumulation 4, optimizer, diffusion
schedule, mixed precision `float16`, and v-parameterization. The experimental
intervention is only the stochastic EDM spectral auxiliary loss: weight `0.01`,
500-step warmup, global subset size 64, one subset per sample, negative and
rank3 component weights 1.0, and physical auxiliary seed 2002.

CUDA gradient calibration at auxiliary weight `0.01` gave a median
auxiliary/diffusion gradient ratio of `0.0553471` on general full-data batches
(`p90=0.162501`, `max=0.183165`). On worst-case N=495-500 batches, the median
ratio was lower at `0.0166878` (`p90=0.0431464`, `max=0.0627834`). That lower
long-chain signal is a documented limitation of using one fixed-size 64-residue
subset, not something corrected inside E002.

Worst-case CUDA stress covered lengths 495-500 with batch size 2, gradient
accumulation 4, 10 optimizer steps plus resume to step 11. Peak allocated memory
was approximately 4.13 GiB, with zero AMP overflows, zero skipped updates, and
checkpoint resume passed.

The full-data preflight ran 2000 optimizer steps plus resume to step 2001. Peak
allocated memory was approximately 4.10 GiB. There were 2 early AMP overflows,
final AMP scale was 16384, final consecutive overflows were 0, no non-finite
losses occurred, and checkpoint serialization/resume passed.

The completed full run selected E002 epoch 5/global step 169181, SHA-256
`2b6b9967a5de6035accad7cc1c24e379743459531c641727221541fde9aa669f`, with best
validation loss `0.0185558852`. E001 best validation loss was `0.0184613932`.
Training recorded 64 isolated AMP overflows, maximum consecutive overflows 1,
and no non-finite losses. Median total auxiliary loss decreased `0.033532 ->
0.014498`; negative component `0.007644 -> 0.002655`; rank3 component
`0.023429 -> 0.010450`.

The corrected selected-model comparison is E001 epoch 4/global step 160166
versus E002 epoch 5/global step 169181. Selected epochs, optimizer steps, and
training histories differ, so this is not a perfectly controlled causal
auxiliary-loss ablation. On 375 exactly paired generated samples, E002 improved
negative eigenvalue mass by `0.00473265` (95% bootstrap CI `[0.00359787,
0.00582859]`, 68.8% improved), rank3 residual by `0.00676679` (`[0.00427944,
0.00915147]`, 65.07% improved), and classical MDS stress by `0.00381575`
(`[0.00172839, 0.00587892]`, 60.53% improved). Adjacent-residue RMSE degraded by
`-0.02684371` (`[-0.03430045, -0.01979658]`, 35.47% improved). Triangle
violation fraction changed by `-0.00026823` (`[-0.00056901, 0.00001563]`,
40.53% improved); because the interval includes zero, this is not described as
a conclusive regression.

Strict empirical real-like geometry remained 0/375 for both E001 and E002.
Heuristic validity changed from 14.4% to 12.8%, with McNemar `p=0.30746`, not
statistically significant. E001 training descriptors were reused because they
are model-independent and the manifest, normalization, descriptor implementation,
and evaluation protocol were identical; descriptor SHA-256:
`ad738db4085c42ff44f91a62d023d26e3416f548faa2d2d17090ef6434f8f865`.

Scientific conclusion: E002 confirms that stochastic spectral EDM
regularization improves global low-dimensional embeddability, negative-spectrum
consistency, and MDS stress. It does not achieve strict empirical protein
geometry and introduces a statistically supported degradation in local
adjacent-residue geometry. Retain E002 as a positive partial experiment, not as
the final model. Do not claim that E002 produces physically valid proteins.

## E003 Adjacent Chain Geometry

Status: CUDA calibration and step-2000 adjacent-chain preflight passed;
definitive full-training config prepared.

E003 is a controlled extension of E002 that preserves the E001/E002
architecture, v-prediction diffusion objective, stochastic EDM spectral loss,
optimizer, schedule, manifests, normalization, batch size 2, gradient
accumulation 4, and mixed precision settings. The only new intervention is a
disabled-by-default adjacent-chain SmoothL1 auxiliary loss on reconstructed
physical `x0_hat` distances for upper adjacent-diagonal pairs `D[i, i+1]`.

The target is the clean training distance matrix, not a hardcoded 3.8 Angstrom
constant. The configured SmoothL1 beta is `0.25` Angstrom, a robust physical
scale that treats sub-quarter-Angstrom deviations quadratically and larger
errors linearly. The selected adjacent weight is `0.0001`, with 500-step warmup.

CUDA calibration at adjacent weight `0.0001` gave a general-batch
adjacent/diffusion gradient ratio median `0.0385003`, p90 `0.0593762`, and max
`0.0722183`; combined auxiliary/diffusion median `0.0916737`, p90 `0.128118`,
and max `0.150298`. On N=495-500 batches, adjacent/diffusion median was
`0.0514392`, p90 `0.114064`, and max `0.176750`; combined auxiliary/diffusion
median was `0.0877296`, p90 `0.162382`, and max `0.235053`. Median gradient
cosines were positive in both profiles: general diffusion/adjacent `0.334203`,
EDM/adjacent `0.331063`, diffusion/EDM `0.201883`; N=495-500
diffusion/adjacent `0.312720`, EDM/adjacent `0.211347`, diffusion/EDM
`0.211596`. Negative tails were occasional and modest, with no systematic
gradient antagonism.

Worst-case CUDA stress covered N=495-500 with batch size 2, gradient
accumulation 4, 10 steps plus resume to 11, peak allocated memory approximately
4.13 GiB, zero AMP overflows, zero skipped updates, and successful checkpoint
resume. The full-data step-2000 preflight produced 8008 JSONL records, two
isolated AMP overflows, final AMP scale 16384, maximum consecutive overflows 1,
no non-finite losses, successful resume to step 2001, and frozen step-2000
SHA-256 `c1e63f84522f22aaeab1ee6627fa60906c46f55bc4ed811d872768e535635764`.

First-versus-last quartile medians improved for adjacent loss `7.39698 ->
1.88258`, diffusion loss `0.0568248 -> 0.0244930`, EDM loss `0.330965 ->
0.0825215`, EDM negative component `0.141549 -> 0.0287379`, EDM rank3 component
`0.188910 -> 0.0514521`, weighted adjacent contribution `0.000296195 ->
0.000188258`, and weighted EDM contribution `0.00128259 -> 0.000825215`.

A paired step-2000 generated screening bank used 10 paired samples, two each at
N=64, 128, 256, 384, and 500, with the same sampling seeds for E002 and E003.
Adjacent RMSE-to-3.8A improved in 9/10 samples, mean improvement `0.452242` A;
negative eigenvalue mass improved in 9/10, mean improvement `0.00192891`; rank3
residual improved in 9/10, mean improvement `0.00630272`; triangle behavior was
mixed; and negative-distance fraction was slightly improved overall. Mean
adjacent RMSE-to-3.8A changes by length were N=64 `3.57996 -> 3.56029`, N=128
`2.71956 -> 2.46473`, N=256 `2.19848 -> 1.81046`, N=384 `2.24399 -> 1.35034`,
and N=500 `2.26398 -> 1.55894`. This is a small screening bank, not final
statistical evidence; definitive evaluation must use calibrated real controls,
not only a fixed 3.8A reference.

This adjacent-chain term is established physical regularization, not a novelty
claim. The experiment tests whether combining global spectral consistency with
local backbone geometry resolves the specific E002 trade-off before any triangle,
contact, non-negativity, radius-of-gyration, projection, or architecture changes
are considered.
