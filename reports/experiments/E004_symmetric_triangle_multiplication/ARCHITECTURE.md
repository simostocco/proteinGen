# E004 Symmetric Triangle Multiplication

## Status

E004 is prepared as a controlled architectural extension of E002. No training,
preprocessing, splitting, sampling, ensemble evaluation, checkpoint modification,
commit, or push was performed during implementation.

## Parent Baseline

E004 uses E002 as the exact parent:
`configs/train_recovered_full_v_axial_edm_e002_full.yaml`.

It preserves the E001 U-Net architecture, symmetric axial attention above the
bottleneck, full bottleneck attention, v-prediction, E002 stochastic EDM
spectral loss, EDM weight `0.01`, EDM warmup `500`, subset size `64`, one subset
per sample, optimizer, scheduler, EMA, diffusion schedule, manifests, masking,
normalization, batching, and mixed precision settings.

E003 is not promoted. E003 showed that adjacent-chain supervision can improve
local adjacent-residue RMSE while degrading global negative eigenvalue mass and
triangle violations. E004 therefore tests a structural inductive bias rather
than another auxiliary loss. The E003 adjacent-chain auxiliary loss is not
enabled in E004.

## Insertion Point

The single triangle-multiplicative block is inserted at the same pre-bottleneck
encoder level as the existing axial block:

```text
residual/local processing
  -> symmetric axial attention
  -> symmetric triangle multiplicative update
  -> downsample
  -> bottleneck residual + full attention + residual
```

For the E002/E004 production model, `base_channels=24` and
`channel_multipliers=[1, 2, 4, 8]`. The inserted block is at encoder level `2`,
block index `1`, with `96` feature channels. For requested `N=500`, dynamic
padding gives side `504`; the triangle block runs at the N/4 side `M=126`,
before downsampling to bottleneck side `63`.

Expected triangle-block side lengths:

| Requested N | Padded side | Triangle side M |
| ---: | ---: | ---: |
| 64 | 64 | 16 |
| 128 | 128 | 32 |
| 256 | 256 | 64 |
| 384 | 384 | 96 |
| 500 | 504 | 126 |

## Operation

Let `z in R^(B x C x M x M)` be the pair representation. Internally the block
uses pair-last layout `z_ij in R^C`.

After masked group normalization, it computes:

```text
a_ik = sigmoid(W_ag z_ik) * W_a z_ik
b_kj = sigmoid(W_bg z_kj) * W_b z_kj
```

with hidden width `R=32` in the E004 config.

For each hidden channel, it computes path composition by batched matrix
multiplication:

```text
m_raw_ij =
  sum_k mask_ik mask_kj (a_ik * b_kj)
  / max(sum_k mask_ik mask_kj, 1)
```

The implementation chunks hidden channels and computes `A_r @ B_r`; it never
materializes a `B x M x M x M x R` triplet tensor.

The message is explicitly symmetrized:

```text
m_ij = 0.5 * (m_raw_ij + m_raw_ji)
```

The residual update is:

```text
delta_ij = sigmoid(W_g z_ij) * W_o(LayerNorm(m_ij))
z'_ij = z_ij + dropout(delta_ij)
```

The output is symmetrized again and multiplied by the valid pair mask. Invalid
or padded rows and columns are zeroed at the block exit. The diagonal follows
the repository's existing pair-mask semantics.

## Equivariance And Masking

The same projections are used for all pair positions, the path message is
symmetrized, and the final output is symmetrized. Therefore transposing the pair
map and mask transposes the output up to floating-point tolerance.

Only valid `i,k` and `k,j` paths contribute to `m_raw_ij`. The denominator is
clamped to one, so all-padding or degenerate masks are safe and return zeroed
invalid outputs.

## Parameters

Verified parameter counts:

| Model | Parameters |
| --- | ---: |
| E002 | 7,557,681 |
| E004 | 7,582,833 |
| Added triangle block | 25,152 |

The added block at `C=96`, `R=32` contains four gated factor projections, one
output projection, one output gate, masked GroupNorm affine parameters, and
LayerNorm affine parameters.

## Complexity

At triangle side `M`, channel width `C`, hidden width `R`, and chunk width `r`,
the block stores projected hidden pair features of order `O(B M^2 R)` and
computes channel-wise path products of order `O(B M^3 R)`. Chunking bounds each
contraction chunk to `O(B M^2 r)` activation inputs plus the output message.
The implementation intentionally avoids a `B x M x M x M x R` tensor, whose
memory would scale as `O(B M^3 R)`.

For N=500, `M=126`; with `B=2`, `R=32`, and chunk size `16`, the largest
contraction inputs are approximately `B * 16 * 126 * 126` values per factor, and
the full message output is `B * 126 * 126 * 32` values.

## Configuration

Preflight config:
`configs/train_recovered_full_v_axial_edm_triangle_e004.yaml`.

New model fields:

| Field | E004 value | Default | Meaning |
| --- | ---: | ---: | --- |
| `use_pre_bottleneck_triangle_multiplication` | `true` | `false` | Enables the single pre-bottleneck triangle update. |
| `triangle_hidden_channels` | `32` | `32` | Hidden width `R`. |
| `triangle_dropout` | `0.0` | `0.0` | Dropout on the projected update. |
| `triangle_chunk_size` | `16` | `16` | Hidden channels per contraction chunk. |

The block is disabled by default, so E000-E003 configs and existing checkpoints
remain compatible.

## Literature And Novelty Boundary

Triangle multiplication is not claimed as novel. AlphaFold uses triangle
multiplicative updates and triangle attention for pair representations. Proteus
already uses graph-based triangle operations in protein backbone diffusion.

E004 is an architectural baseline/ablation inside this project. Any later
novelty claim is outside E004 and is not implemented here.

## Preflight Acceptance Criteria

E004 proceeds to five-epoch training only if CUDA profiling and the 2,000-step
screening show:

- finite losses and gradients;
- no persistent AMP overflow sequence;
- safe checkpoint resume;
- acceptable peak memory;
- no material regression in diffusion reconstruction;
- improved negative eigenvalue mass and/or rank-3 residual;
- improved or neutral triangle violations;
- no major adjacent-distance regression;
- improvements visible across multiple lengths, not only 2/10 samples.

## Commands To Run Later

Parameter-count comparison:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python - <<'PY'
from pathlib import Path
import yaml
from protein_distance_diffusion.training.trainer import build_model_from_config

for path in [
    "configs/train_recovered_full_v_axial_edm_e002_full.yaml",
    "configs/train_recovered_full_v_axial_edm_triangle_e004.yaml",
]:
    cfg = yaml.safe_load(Path(path).read_text())
    model = build_model_from_config(cfg["model"])
    print(path, sum(p.numel() for p in model.parameters()))
PY
```

N=495-500 batch-size-2 CUDA forward/backward stress test:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/profile_edm_auxiliary_gradients.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004.yaml \
  --output reports/experiments/E004_symmetric_triangle_multiplication/gradient_profile_N495_500.json \
  --max-batches 4 \
  --candidate-weights 0.01 \
  --candidate-adjacent-weights 0.0 \
  --subset-size 64 \
  --subsets-per-sample 1 \
  --min-length 495 \
  --max-length 500 \
  --device cuda
```

Ten optimizer steps:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/train_diffusion.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004.yaml \
  --max-optimizer-steps 10
```

Resume from step 10 to step 11:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/train_diffusion.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004.yaml \
  --resume-from outputs/recovered_full_b2_v_axial_edm_triangle_e004_preflight/checkpoints/latest.pt \
  --max-optimizer-steps 11
```

Two-thousand-step full-data preflight:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/train_diffusion.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004.yaml \
  --max-optimizer-steps 2000
```

Paired step-2000 sampling at N=64,128,256,384,500 using the E002 seed schedule:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/evaluate_generated_ensemble.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004.yaml \
  --checkpoint outputs/recovered_full_b2_v_axial_edm_triangle_e004_preflight/checkpoints/step_00002000.pt \
  --output-dir reports/experiments/E004_symmetric_triangle_multiplication/step2000_screening \
  --normalization-file data/full/processed_recovery/normalization_train.json \
  --train-manifest data/full/splits_recovered_all_structures/train.parquet \
  --reference-manifest data/full/splits_recovered_all_structures/validation.parquet \
  --length-samples 64:2,128:2,256:2,384:2,500:2 \
  --master-seed 8000
```

`--master-seed 8000` reproduces the deterministic E002 per-length seed schedule
used by the calibrated comparison protocol.

Physical comparison against E002:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/compare_generated_ensemble_experiments.py \
  --baseline-dir reports/experiments/E002_stochastic_edm_spectral_loss \
  --candidate-dir reports/experiments/E004_symmetric_triangle_multiplication/step2000_screening \
  --output-dir reports/experiments/E004_symmetric_triangle_multiplication/comparison_to_E002_step2000
```
