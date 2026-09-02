# E001 Symmetric Axial Attention

## Scope

E001 keeps the recovered full-data v-prediction training recipe and changes only the
distance U-Net architecture. It adds one symmetry-preserving axial-attention block
at the encoder resolution immediately above the bottleneck. The diffusion objective,
dataset, splits, normalization, batch size, gradient accumulation, decoder, and
E000 configuration remain unchanged.

## Placement

For `configs/train_recovered_full_v.yaml`, the baseline model uses:

- `base_channels: 24`
- `channel_multipliers: [1, 2, 4, 8]`
- encoder channels: `[24, 48, 96, 192]`
- `residual_blocks_per_level: 2`
- spatial downsampling factor: `8`

E001 replaces the final residual block at encoder level `2` (0-based), block index
`1` (0-based), where the feature map has `96` channels. The first convolutional
residual block at that level is preserved. The existing bottleneck self-attention
at level `3` remains enabled.

Feature-map sides for representative valid lengths are:

| Valid length | Padded side | Level 0 | Level 1 | E001 axial level | Bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 64 | 32 | 16 | 8 |
| 128 | 128 | 128 | 64 | 32 | 16 |
| 256 | 256 | 256 | 128 | 64 | 32 |
| 384 | 384 | 384 | 192 | 96 | 48 |
| 500 | 504 | 504 | 252 | 126 | 63 |

## Symmetry

The axial block uses the same `nn.MultiheadAttention` module for row and column
passes. After row attention, column attention, and a pointwise feed-forward block,
the output is projected with `0.5 * (x + x.transpose(-1, -2))` for square feature
maps and padded positions are zeroed. This makes the block transpose-equivariant
and preserves symmetric feature maps.

The block uses masked group normalization so the valid-region output does not
change when the same protein is collated with extra square padding.

## Memory Bound

At length 500, the model pads to side 504. The E001 block runs at side 126.

- Full 2D attention at that resolution would require `126^4 = 252,047,376`
  attention scores per head.
- Axial attention requires row plus column attention,
  `2 * 126^3 = 4,000,752` scores per head.
- That is a `63x` reduction at the E001 insertion point.

The existing bottleneck full attention remains at side 63, or `63^4 = 15,752,961`
attention scores per head.

## Parameter Counts

Measured with the full recovered v-prediction configs:

- E000 baseline (`configs/train_recovered_full_v.yaml`): `7,612,113` parameters
- E001 axial (`configs/train_recovered_full_v_axial_e001.yaml`): `7,557,681` parameters
- Delta: `-54,432` parameters

The parameter count decreases because E001 replaces one 96-channel convolutional
residual block with the axial block instead of appending attention.

## Configuration

The new U-Net fields are disabled by default:

- `use_pre_bottleneck_axial_attention: false`
- `axial_attention_heads: null`
- `axial_attention_dropout: null`
- `axial_attention_chunk_size: null`

The E001 config enables:

- `use_pre_bottleneck_axial_attention: true`
- `axial_attention_heads: 4`
- `axial_attention_dropout: 0.0`
- `axial_attention_chunk_size: 128`

`use_pre_bottleneck_axial_attention` requires at least two U-Net levels and at
least two residual blocks per level, so the replacement always leaves one
convolutional residual block at the insertion level.

## Benchmark Status

`data/benchmarks/train_length500_256.parquet` exists locally. CUDA was not
available in the Codex environment used for implementation
(`torch.cuda.is_available() == False`), so the real GPU stress measurement was
performed on the user's NVIDIA GeForce RTX 5060 with 8,151 MiB VRAM.

## Incident: Mixed-Precision Dtype Mismatch

A subsequent CUDA length-500 stress test failed during the first forward pass,
before any optimizer step completed. The exception was:

```text
RuntimeError: index_copy_(): self and source expected to have the same dtype,
but got (self) Float and (source) Half
```

The failure occurred in `SymmetricAxialAttentionBlock._attend_axis()` while
copying chunked `MultiheadAttention` output back into the preallocated axial
output tensor. Under CUDA autocast, the destination tensor remained `float32`
while `MultiheadAttention` returned `float16`.

The corrective change preserves the architecture and casts the attended chunk to
the destination tensor's device and dtype immediately before `index_copy_()`:

```python
attended = attended.to(device=out.device, dtype=out.dtype)
out.index_copy_(0, indices, attended)
```

Verification after the fix:

- `ruff check . --fix`
- `ruff format .`
- `ruff check .`
- `ruff format --check .`
- `PYTHONPATH=src pytest -q tests/test_axial_attention.py`
- `PYTHONPATH=src pytest -q`

The local environment still has no CUDA device, so the guarded CUDA fp16
regression test was collected but skipped locally. CPU bfloat16 autocast
regressions cover both chunked and unchunked axial paths.

## Real-CUDA Worst-Case Stress Test

Completed on 2026-08-31 using:

- GPU: NVIDIA GeForce RTX 5060 with 8,151 MiB VRAM
- Mixed precision: CUDA float16 enabled
- Batch size: 2
- Gradient accumulation steps: 4
- Effective batch size: 8
- Stress manifest: `data/benchmarks/train_length500_256.parquet`
- Sample population: 256 longest training samples
- Length range: N=495-500

The first real-CUDA attempt failed during the first forward pass and completed
zero optimizer steps. The exception was:

```text
RuntimeError: index_copy_(): self and source expected to have the same dtype,
but got (self) Float and (source) Half
```

The root cause was that CUDA autocast made `MultiheadAttention` return
`attended` as `float16`, while `_attend_axis()` had created the destination
`out` tensor as `float32`. The in-place `index_copy_()` operation requires
matching dtypes.

The corrective change was to convert `attended` to the destination tensor's
device and dtype before `index_copy_()`:

```python
attended = attended.to(device=out.device, dtype=out.dtype)
out.index_copy_(0, indices, attended)
```

AMP was not disabled. The architecture, masking, chunking, symmetry logic, and
parameter count were unchanged.

Verification after the fix:

- Focused axial-attention tests: 12 passed, 2 skipped
- Full test suite: 232 passed, 6 skipped, 6 warnings
- CPU bfloat16 autocast tests covered chunked and unchunked paths
- The guarded CUDA unit test was skipped in the Codex environment because CUDA
  was unavailable
- Real CUDA verification was performed on the user's RTX 5060

The successful retry completed optimizer step 10 with 40 training-log records.
Peak allocated GPU memory was 4.12507963180542 GiB and peak reserved GPU memory
was 6.380859375 GiB. Total AMP overflows, maximum consecutive AMP overflows,
and skipped updates were all 0. The final recorded training loss was
0.2867460250854492 and the final pre-clipping gradient norm was
3.8645412921905518. All critical loss and gradient values were finite. The
gradient clipping threshold was 1.0, so the reported 3.8645 norm is a
pre-clipping measurement and is not itself an instability.

Checkpoint-resume verification resumed from
`outputs/recovered_stress_b2_v_axial_e001/checkpoints/last.pt` and advanced
successfully from optimizer step 10 to optimizer step 11 with exit code 0. The
final checkpoint recorded `optimizer_step: 11`, `global_step: 11`,
`microbatch_in_accumulation: 0`, and `amp_overflows_consecutive: 0`. This
verifies checkpoint serialization and restoration for E001.

Comparison with the E000 worst-case batch-2 stress test:

| Metric | E000 | E001 | Difference |
| --- | ---: | ---: | ---: |
| Peak allocated GPU memory | 3.897273540496826 GiB | 4.12507963180542 GiB | +0.227806091308594 GiB, approximately +5.85% |
| Peak reserved GPU memory | 6.140625 GiB | 6.380859375 GiB | +0.240234375 GiB, approximately +3.91% |
| AMP overflows | 0 | 0 | no change |
| Skipped updates | 0 | 0 | no change |

Interpretation: the mixed-precision implementation defect is resolved on real
CUDA hardware. E001 supports worst-case N=495-500 samples at batch size 2 within
the available 8 GB GPU, and its measured memory overhead relative to E000 is
modest. Forward pass, backward pass, optimizer update, checkpoint saving, and
checkpoint resume all succeeded. This short stress test validates execution
safety and memory feasibility, not convergence or generative quality. E001 is
ready for a bounded 2000-optimizer-step full-data preflight. It does not yet
establish any improvement in physical validity or global geometry; those
hypotheses remain to be tested against E000 under an identical evaluation
protocol.

## Calibrated Comparison To E000

Completed on 2026-09-02 under
`reports/experiments/E001_symmetric_axial_attention/comparison_to_E000`.

Selected-model provenance:

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
resume/replay caveat. The complete-ensemble comparison is therefore a
selected-model comparison, not a perfectly controlled causal architecture
ablation. The earlier step-2000 comparison matched optimization steps but used
only two samples per length and remains exploratory screening.

The complete E001 ensemble has 375 generated samples and 320 real controls. The
generation/evaluation protocol runtime was 9,754.049385 seconds and calibrated
finalization runtime was 17.806966543197632 seconds. The comparison paired all
375 samples exactly by requested length, sample index, and seed, using a
deterministic length-stratified paired bootstrap with 2,000 iterations and seed
8000. Input hashes were preserved before and after the analysis.

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

Comparison conclusion: E001 improves several selected-model global geometry
diagnostics and increases heuristic EDM-quality passes without evidence of
diversity collapse or exact-duplicate novelty failure. E001 still does not
reach the empirical 3D distance-matrix manifold under strict calibrated
criteria. The next justified intervention is a bounded physical auxiliary-loss
experiment evaluated against E000 and E001 under the same protocol.
