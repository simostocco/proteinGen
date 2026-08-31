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
