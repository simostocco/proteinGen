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

## E001 Worst-Case CUDA Stress Test

Completed on 2026-08-31 for `E001_symmetric_axial_attention`.

### Setup

- GPU: NVIDIA GeForce RTX 5060 with 8,151 MiB VRAM
- Mixed precision: CUDA float16 enabled
- Batch size: 2
- Gradient accumulation steps: 4
- Effective batch size: 8
- Stress manifest: `data/benchmarks/train_length500_256.parquet`
- Sample population: 256 longest training samples
- Length range: N=495-500

### Initial Failed Attempt

The first real-CUDA attempt failed during the first forward pass and completed
zero optimizer steps. The exception was:

```text
RuntimeError: index_copy_(): self and source expected to have the same dtype,
but got (self) Float and (source) Half
```

Root cause: under CUDA autocast, `MultiheadAttention` returned `attended` as
`float16`, while `SymmetricAxialAttentionBlock._attend_axis()` had created the
destination `out` tensor as `float32`. The in-place `index_copy_()` operation
requires matching dtypes.

The corrective change converted `attended` to the destination tensor's device
and dtype before `index_copy_()`:

```python
attended = attended.to(device=out.device, dtype=out.dtype)
out.index_copy_(0, indices, attended)
```

AMP was not disabled. Architecture, masking, chunking, symmetry logic, and
parameter count were unchanged.

### Verification After Fix

- Focused axial-attention tests: 12 passed, 2 skipped
- Full test suite: 232 passed, 6 skipped, 6 warnings
- CPU bfloat16 autocast tests covered chunked and unchunked paths
- The guarded CUDA unit test was skipped in the Codex environment because CUDA
  was unavailable
- Real CUDA verification was performed on the user's RTX 5060

### Successful Retry

The successful real-CUDA stress run completed optimizer step 10 with 40
training-log records.

| Metric | Value |
| --- | ---: |
| Peak allocated GPU memory | 4.12507963180542 GiB |
| Peak reserved GPU memory | 6.380859375 GiB |
| Total AMP overflows | 0 |
| Maximum consecutive AMP overflows | 0 |
| Skipped updates | 0 |
| Final recorded training loss | 0.2867460250854492 |
| Final pre-clipping gradient norm | 3.8645412921905518 |

All critical loss and gradient values were finite. The gradient clipping
threshold was 1.0, so the reported norm of 3.8645 is a pre-clipping measurement
and is not itself an instability.

Checkpoint-resume verification resumed from
`outputs/recovered_stress_b2_v_axial_e001/checkpoints/last.pt` and successfully
advanced from optimizer step 10 to optimizer step 11 with exit code 0. The final
checkpoint recorded `optimizer_step: 11`, `global_step: 11`,
`microbatch_in_accumulation: 0`, and `amp_overflows_consecutive: 0`.

### E000 Comparison

| Metric | E000 | E001 | Difference |
| --- | ---: | ---: | ---: |
| Peak allocated GPU memory | 3.897273540496826 GiB | 4.12507963180542 GiB | +0.227806091308594 GiB, approximately +5.85% |
| Peak reserved GPU memory | 6.140625 GiB | 6.380859375 GiB | +0.240234375 GiB, approximately +3.91% |
| AMP overflows | 0 | 0 | no change |
| Skipped updates | 0 | 0 | no change |

Interpretation: the mixed-precision implementation defect is resolved on real
CUDA hardware. E001 supports worst-case N=495-500 samples at batch size 2 within
the available 8 GB GPU, with modest measured memory overhead relative to E000.
Forward pass, backward pass, optimizer update, checkpoint saving, and checkpoint
resume all succeeded. This short stress test validates execution safety and
memory feasibility, not convergence or generative quality. E001 is ready for a
bounded 2000-optimizer-step full-data preflight. It does not establish improved
physical validity or global geometry; those hypotheses remain to be tested
against E000 under an identical evaluation protocol.
