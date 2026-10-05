# E004 Symmetric Triangle Multiplication

## Status

E004 is prepared as a controlled architectural extension of E002. The CUDA
worst-case stress test, 2,000-step preflight, paired step-2,000 screen, and
timestep diagnostic have completed successfully. E004 passes the five-epoch
training gate, but E002 remains the baseline until the full E004 ensemble is
trained and evaluated.

This documentation/configuration update did not start training, preprocessing,
splitting, sampling, ensemble evaluation, checkpoint modification, commit, or
push.

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

Implementation commit: `cd31047`.

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

Definitive five-epoch config:
`configs/train_recovered_full_v_axial_edm_triangle_e004_full.yaml`, with output
directory `outputs/recovered_full_b2_v_axial_edm_triangle_e004`.

New model fields:

| Field | E004 value | Default | Meaning |
| --- | ---: | ---: | --- |
| `use_pre_bottleneck_triangle_multiplication` | `true` | `false` | Enables the single pre-bottleneck triangle update. |
| `triangle_hidden_channels` | `32` | `32` | Hidden width `R`. |
| `triangle_dropout` | `0.0` | `0.0` | Dropout on the projected update. |
| `triangle_chunk_size` | `16` | `16` | Hidden channels per contraction chunk. |

The block is disabled by default, so E000-E003 configs and existing checkpoints
remain compatible.

## Completed Preflight Evidence

The worst-case CUDA stress test used lengths N=495-500, batch size 2, and
gradient accumulation 4. It completed 10 optimizer steps plus resume to step 11.
Peak allocated memory was `4.22 GiB`; peak reserved memory was `7.64 GiB` during
the initial run and `6.54 GiB` after resume. There were zero AMP overflows, zero
skipped updates, clean accumulation state, and the resume passed. Approximate
resume throughput was `9.22 samples/s`.

The four-batch N=495-500 gradient profile is preserved as
`reports/experiments/E004_symmetric_triangle_multiplication/gradient_profile_N495_500.json`.
It is retained as a bounded stress/profile artifact and should not be
overinterpreted as a full-data gradient distribution.

The full-data preflight completed `2000` optimizer steps and `8008` JSONL
records. The step-2000 checkpoint SHA-256 was
`ecd1b34780e74bf2f367bf2ced884e57e8a6ec3233f6b75f29c798cdbca80d01`. It had
`2` AMP overflow events, maximum consecutive overflows `1`, `8` skipped-window
records, no non-finite losses, no unexpected non-finite gradients, final
consecutive overflows `0`, clean microbatch accumulation state, and final
recorded throughput `23.8145 samples/s`.

First-versus-last quartile median losses:

| Quantity | First quartile | Last quartile |
| --- | ---: | ---: |
| Diffusion loss | 0.053101562 | 0.022458778 |
| Optimization loss | 0.054552520 | 0.023218437 |
| EDM total | 0.30521363 | 0.058729568 |
| EDM negative | 0.12913704 | 0.018167607 |
| EDM rank3 | 0.17434631 | 0.037178006 |

## Step-2000 Screen Against E002

The screen used two paired samples per requested length at N=64, 128, 256, 384,
and 500, with matching seeds and sample indices. Positive improvement means
E004 is lower/better than E002. Matrices were not projected before comparison.

Overall results:

| Metric | Mean improvement | Improved samples |
| --- | ---: | ---: |
| Negative-distance fraction | -0.0001035543 | 7/10 |
| Triangle-violation fraction | 0.0009765625 | 7/10 |
| Negative-eigenvalue mass fraction | 0.026845372 | 7/10 |
| Rank3 residual | 0.066621558 | 10/10 |
| Adjacent RMSE to 3.8 A | 0.66738602 | 8/10 |

Per-length effects:

| N | Negative eigenvalue mass | Rank3 residual | Triangle violation | Adjacent RMSE to 3.8 A |
| ---: | ---: | ---: | ---: | ---: |
| 64 | -0.019554 | 0.052774 | -0.007812 | -0.597252 |
| 128 | 0.003792 | 0.076017 | -0.000488 | 1.205684 |
| 256 | 0.038827 | 0.055579 | 0.002441 | 0.858173 |
| 384 | 0.053112 | 0.075206 | 0.005859 | 0.894127 |
| 500 | 0.058049 | 0.073532 | 0.004883 | 0.976197 |

E004 passes the preflight gate because it produces a coherent, large improvement
in global geometric metrics, especially for N>=256. The screen remains small:
it has only two samples per length, N=64 regresses in several metrics, and no
strict-validity conclusion can be made from ten samples. The result justifies
full training but is not final evidence. E002 remains the baseline until the
full E004 ensemble is evaluated.

## Timestep Diagnostic

The completed E002-to-E004 step-2000 timestep diagnostic shows single-step `x0`
reconstruction RMSE improvement in Angstrom at 8/9 reported timesteps, with mean
improvement `0.19944867 A`. Positive means E004 is lower/better.

| Timestep | x0 RMSE improvement (A) |
| ---: | ---: |
| 0 | 0.009868 |
| 100 | 0.330156 |
| 200 | 0.282385 |
| 300 | 0.243457 |
| 400 | 0.391255 |
| 450 | 0.029041 |
| 475 | 0.261914 |
| 490 | 0.352500 |
| 499 | -0.105538 |

Target-MSE improved at t=0, 100, 200, 300, 400, 475, and 490. It was slightly
worse at t=450 by `0.000511` and at t=499 by `0.001043`.

Negative reconstructed physical fraction showed small mixed changes through
t=490. At t=499, E002 was `0.000924`, E004 was `0.004554`, for a change of
`-0.003630`.

Interpretation: E004 improves single-step `x0` reconstruction across almost the
entire diffusion trajectory. The isolated t=499 regression is small relative to
the reconstruction magnitude, but the increased negative-distance fraction at
t=499 is retained as a caveat. This does not block full training, and no special
t=499 correction should be added before the controlled full experiment.

## Literature And Novelty Boundary

Triangle multiplication is not claimed as novel. AlphaFold uses triangle
multiplicative updates and triangle attention for pair representations. Proteus
already uses graph-based triangle operations in protein backbone diffusion.

E004 is an architectural baseline/ablation inside this project. Any later
novelty claim is outside E004 and is not implemented here.

## Preflight Acceptance Criteria

E004 proceeds to five-epoch training because CUDA profiling and the 2,000-step
screening showed:

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

Definitive five-epoch training:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/train_diffusion.py \
  --config configs/train_recovered_full_v_axial_edm_triangle_e004_full.yaml
```

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

## Repairability Audit

The definitive analysis-only repairability audit completed under
`reports/experiments/E004_symmetric_triangle_multiplication/repairability_analysis`
using checkpoint SHA-256
`59db27a3dbecbc199cb20065e1263ad0cec12b3553cca1a263429f890b38ea86`.
It contains 375 E002 samples, 375 E004 samples, 320 real controls, and 25
length-by-noise corruption aggregate cells. The protocol records
`raw_inputs_unchanged=true`.

The projection method is rank-3 classical MDS:

```text
G = -0.5 J D^2 J
```

The three largest positive eigenvalues are retained to recover C-alpha
coordinates and a projected distance matrix. This is a deterministic projection
diagnostic, not a mathematically nearest Euclidean-distance-matrix repair. It
does not infer side chains, chirality, Ramachandran feasibility, energetic
stability, or full-atom validity. Reflected traces have identical distance
matrices, so coordinate RMSD diagnostics allow reflection where appropriate.

Completed outputs include per-sample repairability, summaries by length,
projected C-alpha trace metrics, real-corruption calibration, a ranked candidate
table retained under the compatibility filename `pareto_candidates.csv`,
representative C-alpha-only PDB files, figures, and a protocol JSON.
Short-range distance errors use `3 <= |i-j| < 24`; long-range errors use
`|i-j| >= 24`.

Real controls must be selected independently within the ensemble evaluator's
`requested_length` cohorts; their actual structure length is retained
separately. Representative `best`, `median`, and `worst` labels are positions in
a deterministic lexicographic repairability ranking: off-diagonal projection
RMSE first and negative-eigenvalue mass second. They are not minimum projection
stress labels or a formal Pareto frontier. With two samples, the middle-index
and worst positions legitimately identify the same sample.

Mean E002-to-E004 off-diagonal projection-RMSE improvements are 0.0571,
0.4206, 0.7665, 1.2687, and 1.3476 Angstrom at N=64, 128, 256, 384, and 500.
At N=64, contact F1 improves by 0.00984 while negative Gram-matrix eigenvalue
mass worsens. Under the permissive 4 Angstrom corrupted-real joint envelope,
81% of N=128, 1% of N=256, and 0% of N=384 and N=500 E004 samples pass. This
envelope is a calibration device, not proof of validity or foldability.

Projected E004 traces retain compressed local geometry: mean adjacent C-alpha
distances are approximately 3.21-3.39 Angstrom and median adjacent RMSE is
approximately 0.79-0.91 Angstrom. Low positive rank-3 residual is distinct from
remaining negative Gram-matrix eigenvalue mass. E004 is therefore retained as
an imperfect-geometry proposal model, not a directly valid backbone generator.
The definitive interpretation is recorded in `E004_FINAL_REPORT.md`.

Historical bounded pilot command:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/analyze_distance_map_repairability.py \
  --candidate-dir reports/experiments/E004_symmetric_triangle_multiplication \
  --baseline-dir reports/experiments/E002_stochastic_edm_spectral_loss \
  --real-manifest reports/experiments/E004_symmetric_triangle_multiplication/metrics/real_control_metrics.parquet \
  --output-dir reports/experiments/E004_symmetric_triangle_multiplication/repairability_analysis_pilot \
  --expected-candidate-checkpoint-sha256 59db27a3dbecbc199cb20065e1263ad0cec12b3553cca1a263429f890b38ea86 \
  --limit-per-length 2 \
  --restart
```

Historical definitive analysis command:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/analyze_distance_map_repairability.py \
  --candidate-dir reports/experiments/E004_symmetric_triangle_multiplication \
  --baseline-dir reports/experiments/E002_stochastic_edm_spectral_loss \
  --real-manifest reports/experiments/E004_symmetric_triangle_multiplication/metrics/real_control_metrics.parquet \
  --output-dir reports/experiments/E004_symmetric_triangle_multiplication/repairability_analysis \
  --expected-candidate-checkpoint-sha256 59db27a3dbecbc199cb20065e1263ad0cec12b3553cca1a263429f890b38ea86 \
  --restart
```

## Distance-AF Dry-Run Benchmark

A dry-run Distance-AF benchmark interface is implemented for later restrained
structure experiments. It prepares FASTA files, one-based comma-separated
CA-distance restraint files, held-out restraint tables, job manifests, command
JSON files, metric definitions, and positive-control documentation without
launching external inference.

Primary generated E002/E004 jobs are intentionally blocked until an explicit
sequence source is supplied. A distance map alone is not a Distance-AF target.
Allowed sequence provenance labels are native PDB sequence, model-generated
sequence, external inverse-folded sequence, sequence-only baseline, shuffled
negative control, and mismatched negative control. Generated primary cohorts
must use a model-generated or external inverse-folded sequence.

Distance-AF supplied-restraint satisfaction is not independent validation
because the external method optimizes toward the supplied CA-distance
constraints. The benchmark therefore separates guidance restraints from held-out
restraints and includes positive controls, corrupted-restraint controls,
sequence-only baselines, and optional negative controls. The dry-run command
builder emits a list of subprocess arguments and never uses `shell=True`.

Dry-run preparation command:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/prepare_distance_af_benchmark.py \
  --config configs/evaluate_distance_af_repairability.yaml
```

Synthetic dry-run smoke command:

```bash
PYTHONPATH=src /home/simostocco/miniforge3/envs/proteingen/bin/python scripts/prepare_distance_af_benchmark.py \
  --config /tmp/proteingen_distance_af_smoke.yaml
```

External execution remains a separate manual step after Distance-AF licensing,
environment, sequence-source, and resource questions are resolved.
