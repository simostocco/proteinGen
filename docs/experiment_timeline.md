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

## E004 Repairability And Sequence-Data Readiness

The definitive rank-3 classical-MDS repairability audit completed with 375 E002
samples, 375 E004 samples, 320 real controls, and 25 corruption-calibration
aggregate cells. It used the final-validation checkpoint SHA-256
`59db27a3dbecbc199cb20065e1263ad0cec12b3553cca1a263429f890b38ea86` and
recorded `raw_inputs_unchanged=true`.

E004 materially improves E002 projection repairability. Mean projection-RMSE
improvements increase from 0.0571 Angstrom at N=64 to 1.3476 Angstrom at N=500.
The permissive 4 Angstrom corrupted-real envelope admits 81% of N=128 samples,
but only 1% at N=256 and none at N=384 or N=500. Projected traces still have
compressed adjacent distances near 3.21-3.39 Angstrom and median adjacent RMSE
near 0.79-0.91 Angstrom. E004 is an imperfect-geometry proposal model, not a
directly valid backbone generator. Envelope passage does not prove physical
validity or foldability, and low positive rank-3 residual does not eliminate
negative Gram-matrix eigenvalue mass.

Prepared a read-only, staged sequence-data readiness audit for the prospective
sequence-geometry codesign experiment. At the observed scale of 223,709 raw
mmCIF files and 506,919 merged processed rows, the audit now separates a
complete `manifest-only` pass, a deterministic unique-source `raw-pilot`, and a
checkpointed `raw-full` pass. Manifest-only makes zero raw-parser calls. Raw
modes preserve separate declared-polymer, coordinate-record, and matrix-index
sequences and process each selected source once, with SQLite path/SHA-256 state
and atomic per-source partitions.

The full raw audit is explicitly deferred. The complete manifest report must be
reviewed first, followed by a 250-source stratified pilot covering split,
length, method, model, chain multiplicity, trimming, sequence alphabet,
repeated-source, and missing-C-alpha strata where available. Sparse strata,
parser failures, input hashes, parser calls, and completed/failed/pending counts
are protocol outputs rather than silently discarded conditions. No dataset
migration or codesign implementation is authorized until these staged gates
pass.

Implemented a dry-run Distance-AF benchmark interface that exports FASTA files,
one-based comma-separated CA-distance restraints, held-out restraint tables,
subprocess-safe command lists, and metric definitions. Primary generated E002
and E004 Distance-AF jobs remain blocked until an explicit sequence source is
provided. Supplied-restraint satisfaction is partly circular because Distance-AF
optimizes toward those restraints; independent evidence must come from held-out
restraints, positive controls, corrupted-restraint recovery, sequence-only
baselines, and confidence diagnostics.

This finalization pass did not rerun repairability analysis or start a
Distance-AF, AlphaFold/OpenFold, diffusion sampling, preprocessing, splitting,
or training workflow.
## E005 Sequence-Geometry Co-Design Scaffold

E005 adds a canonical-token sequence branch around the unchanged E004 geometry model, with gated
geometry-to-sequence, sequence-to-geometry, and return geometry-to-sequence feedback. It supports
sequence-only, learned-gating, and forced-conditioning modes plus deterministic conditioning
dropout. Explicit residue and pair masks preserve mixed-length behavior, and sequence-only mode
zeros every geometry route to prevent ground-truth geometry leakage.

The full scaffold has 8,512,649 parameters: 7,582,833 in the unchanged E004 branch and 929,816 in
the new sequence/feedback layers. A synthetic-only bounded dry-run harness and immutable-dataset
one-batch integration path are implemented. No real-data training, sampling, AlphaFold, or
Distance-AF execution was started during this preparation.

## E010 Phase 4B forensic closeout — 2026-10-02

Phase 4B completed 98,280 examples and 1,092 updates; all 129 refiner tensors changed. All three boundaries were `insufficient_real_domain_gain`. Final paired aligned-coordinate RMSD gain was 1.2499% while mean local-distance improvement was −14.0092%. Historical malformed archives were excluded from actual training/evaluation populations. Zero-update gradients and finite Adam displacements confirmed Cartesian/local-geometry conflict on fixed training and development panels; condition 450 dominated the historical equal-mixture displacement. These panels do not establish a length law. E010 remains reflection-inclusive E(3)-equivariant without a handedness preference. See [the forensic closeout](e010_phase4b_forensic_closeout.md) for metrics, limitations and checkpoint hashes.

## E010 frozen condition-weight closeout — 2026-10-05

CF1 `(5/12,5/12,1/6)` versus equal weights retained 89.00% Cartesian descent,
reduced predicted local harm 37.62% and was classified W2 (partial improvement).
CF2 `(11/24,11/24,1/12)` reversed all three aggregate training local derivatives,
but retained only 66.26% Cartesian descent versus the pre-registered 70% gate.
Development mean-local MSE remained slightly harmful. CF2 was classified X4
under the retention gate; alpha=0.01 independent copies confirmed the responses.
Condition 250, rather than 450, dominated CF2's raw gradient projection.

**Weight-only tuning is closed; CF2 should not be trained.** The next step is a
pre-registered frozen-state coefficient diagnostic for an explicit local
i+1/i+2/i+3 auxiliary objective, restoring historical equal condition weights.
Chirality-aware representation remains separate. The
[completed counterfactual note](e010_condition_weight_counterfactual.md) records
both results and the unchanged checkpoint/panels; no retraining was performed.
