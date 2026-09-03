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

Final calibrated analysis completed at `2026-08-31T06:55:24.198454+00:00`.

- Generated samples: 375
- Real controls: 320
- Empirical real-like geometry pass count: 0
- Deprecated/permissive heuristic EDM-quality pass count: 12
- Raw E000 inputs unchanged during finalization: True

Conclusion: The current model produces numerically valid, non-duplicated, length-conditioned distance-like matrices and approximates several local distributional properties, but its generated matrices do not lie on the empirical manifold of real three-dimensional protein distance matrices. Global geometric inconsistency and excess compactness worsen with sequence length.

E001 remains `E001_symmetric_axial_attention`, testing whether one
symmetry-preserving axial-attention block immediately above the bottleneck
improves global geometry and scaling with `N` without reducing diversity or
increasing training-set similarity.

## E001 Calibrated Selected-Model Comparison To E000

Completed on 2026-09-02 under
`reports/experiments/E001_symmetric_axial_attention/comparison_to_E000`.

### Selected Model Provenance

- E000 selected checkpoint: epoch 8,
  `outputs/recovered_full_b2_v/checkpoints/final_validation_selected.pt`
- E000 checkpoint SHA-256:
  `8a9da579153612556f568716b4ce5eaaa5fc3f036f82555497901f2ab7599cf8`
- E001 selected checkpoint: epoch 4, global_step 160166,
  `outputs/recovered_full_b2_v_axial_e001_epoch1/checkpoints/final_validation_selected.pt`
- E001 checkpoint SHA-256:
  `e6ddc698738718a0a73c5c4a3cbded23601990c107118ec3e1c99a185b58700a`
- E001 epoch-4 validation loss: 0.0184613932
- E001 epoch-5 validation loss: 0.0187884234
- E001 final checkpoint global_step: 194002

The E001 selected checkpoint reflects the previously documented mid-epoch
resume/replay caveat. The selected-model comparison is therefore not a perfectly
controlled causal architecture ablation. The earlier step-2000 comparison used
matched optimization steps but only two samples per length and remains
exploratory screening rather than formal statistical evidence.

### Ensemble And Comparison

- E001 complete ensemble: 375 generated samples and 320 real controls
- E001 requested-length counts: N=64: 100, N=128: 100, N=256: 100,
  N=384: 50, N=500: 25
- E001 generation/evaluation protocol runtime: 9,754.049385 seconds
- E001 calibrated-finalization runtime: 17.806966543197632 seconds
- Comparison paired samples: 375 exact pairs by requested length, sample index,
  and seed
- Comparison bootstrap: deterministic length-stratified paired bootstrap,
  2,000 iterations, seed 8000
- Input hashes preserved: true

Primary lower-is-better paired effects, defined as E000 minus E001:

| Metric | Mean improvement | 95% CI |
| --- | ---: | ---: |
| triangle_violation_fraction | 0.0037239583333333335 | [0.0030468424479166666, 0.004445377604166666] |
| negative_eigenvalue_mass_fraction | 0.03555181422234015 | [0.03379576492521358, 0.03726582691123128] |
| rank3_residual_energy_fraction | 0.03479467005340526 | [0.03128290943186638, 0.03845387632871159] |
| classical_mds_stress | 0.019002312775985533 | [0.016046398411326216, 0.021792401833207808] |
| adjacent_residue_distance_rmse | 0.07468501927806964 | [0.06101909166309191, 0.08915228026826212] |

Strict empirical real-like geometry remained 0/375 for both E000 and E001.
Heuristic EDM-quality transitions were 319 fail/fail, 44 E000-fail/E001-pass,
2 E000-pass/E001-fail, and 10 pass/pass, increasing the pass fraction from
0.032 to 0.144.

Calibrated comparison conclusion: E001 improves several selected-model global
geometry diagnostics and increases heuristic EDM-quality passes without evidence
of diversity collapse or exact-duplicate novelty failure. However, E001 still
does not reach the empirical 3D distance-matrix manifold under strict calibrated
criteria. The next justified intervention is a bounded physical auxiliary-loss
experiment evaluated against E000 and E001 under the same protocol.

## E002 Stochastic EDM Spectral Auxiliary Loss

Status: step-2000 CUDA preflight completed successfully; full epoch training
not yet started.

E002 preserves the E001 architecture, recovered full-data splits,
normalization, batch size, gradient accumulation, optimizer, diffusion schedule,
and v-parameterization. The only planned experimental intervention is a
stochastic spectral auxiliary loss on deterministic principal submatrices of
the reconstructed physical `x0_hat` distance map.

The design and prior-art boundary are documented in
`reports/experiments/E002_stochastic_edm_spectral_loss/DESIGN.md`. The
validated preflight config is `configs/train_recovered_full_v_axial_edm_e002.yaml`.
The clean full-run config is
`configs/train_recovered_full_v_axial_edm_e002_full.yaml`, with definitive output
directory `outputs/recovered_full_b2_v_axial_edm_e002`.

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

The full-data preflight ran 2000 optimizer steps plus resume to step 2001 with
auxiliary weight `0.01`, 500-step warmup, subset size 64, and one subset per
sample. Peak allocated memory was approximately 4.10 GiB. There were 2 early AMP
overflows, final AMP scale was 16384, final consecutive overflows were 0, no
non-finite losses occurred, and checkpoint serialization/resume passed.

Median first-to-last-quartile losses improved: diffusion loss `0.0578014 ->
0.0248815`, total EDM auxiliary loss `0.329871 -> 0.0849254`,
negative-spectrum loss `0.140084 -> 0.0298418`, rank-3 loss `0.188657 ->
0.0518208`, and weighted auxiliary loss `0.00129356 -> 0.000849254`.

A paired step-2000 generative screen used 10 exactly paired samples at lengths
64, 128, 256, 384, and 500, with two samples per length and identical sampling
seeds between E001 and E002. Rank-3 residual improved in 10/10 samples with mean
improvement `0.0370193`; negative eigenvalue mass improved in 7/10 with mean
improvement `0.00078317`. Triangle violation fraction worsened in 10/10, and
negative-distance fraction worsened or remained tied. Reconstruction diagnostics
generally improved, especially at `t=499`, where x0 RMSE improved from
`12.184747` to `11.810533` Angstrom and negative reconstructed fraction improved
from `0.002038` to `0.000924`.

Scientific decision: E002 shows a successful targeted rank-3 effect, but strict
EDM validity has not yet been achieved. Keep E002 spectral-only for causal
attribution, and run one complete epoch from scratch before considering triangle
or non-negativity terms.
