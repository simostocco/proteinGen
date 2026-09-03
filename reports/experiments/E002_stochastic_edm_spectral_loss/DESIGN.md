# E002 Stochastic EDM Spectral Auxiliary Loss

## Status

Implemented as a disabled-by-default training component. CUDA gradient
calibration, worst-case stress, and the step-2000 full-data preflight have
completed successfully. Full epoch training has not yet started.

## Baseline To Preserve

E002 keeps the E001 architecture exactly: the recovered full-data v-prediction
U-Net with full bottleneck attention and symmetric axial attention immediately
above the bottleneck. The intervention is only an auxiliary loss computed from
the model-implied clean distance map `x0_hat`.

The repository already provides the required baseline mechanics:

- cosine diffusion schedule in `diffusion/schedules.py`;
- exact epsilon/v conversion in
  `GaussianDiffusion.predict_x0_epsilon_from_model_output`;
- scalar distance normalization in `DistanceMapDataset`;
- dynamic square padding and biological pair masks in `collate.py`;
- gradient accumulation, AMP and checkpoint/resume in `training/trainer.py`.

## Loss Definition

For each valid protein, E002 samples one or more deterministic global principal
submatrices of the reconstructed physical distance matrix. For a submatrix
`D` of size `m x m`:

```text
J = I - (1/m) 11^T
B = -0.5 J (D * D) J
```

For an exact 3D Euclidean distance matrix, `B` is positive semidefinite with
rank at most 3. E002 penalizes two dimensionless spectral failures:

```text
L_negative = sum relu(-lambda_i)^2 / (sum lambda_i^2 + eps)
L_rank3    = sum_{i not in largest 3} relu(lambda_i)^2
             / (sum relu(lambda_i)^2 + eps)
L_EDM      = negative_weight * L_negative + rank3_weight * L_rank3
```

The implementation uses `torch.linalg.eigvalsh` in float32 with autocast
disabled, never includes padded residues, re-symmetrizes and zeroes the diagonal,
and leaves gradients connected to the model. No eigenvectors, projections,
triangle losses, radius-of-gyration losses, contact losses, or clipping-based
geometry fixes are used.

Subset sampling is stateless: each submatrix seed is derived from
`physical_auxiliary_seed`, `sample_id`, optimizer step, microbatch index, and
subset index. Checkpoint/resume therefore needs no extra RNG payload.

## Configuration

New top-level fields default to disabled or neutral values:

| Field | Default | Meaning |
| --- | ---: | --- |
| `physical_auxiliary_loss_enabled` | `false` | Turns the component on. |
| `physical_auxiliary_loss_weight` | `0.0` | Weight multiplying raw `L_EDM`. |
| `physical_auxiliary_loss_warmup_steps` | `0` | Linear optimizer-step warmup. |
| `edm_subset_size` | `64` | Maximum principal-submatrix size. |
| `edm_subsets_per_sample` | `1` | Number of global subsets per protein. |
| `edm_negative_weight` | `1.0` | Weight for negative eigenvalues. |
| `edm_rank3_weight` | `1.0` | Weight for rank >3 positive spectrum. |
| `physical_auxiliary_seed` | `0` | Stateless subset-sampling base seed. |
| `edm_loss_eps` | `1e-8` | Numerical denominator floor. |

When enabled, JSONL and TensorBoard add diffusion-only loss, raw spectral
components, raw total EDM loss, active auxiliary weight, weighted contribution,
total optimization loss, eligible fraction, subset count, and mean subset size.
The existing `training_loss` tag remains diffusion-only.

## Prior Art Boundary

| Work | Representation | Constraint or geometry method | Cost emphasis | Difference from E002 |
| --- | --- | --- | --- | --- |
| Anand & Huang, NeurIPS 2018, [Generative modeling for protein structures](https://proceedings.neurips.cc/paper/7978-generative-modeling-for-protein-structures) | C-alpha pairwise distance maps | GAN generation plus post-hoc ADMM/convex 3D recovery | Reconstruction is a downstream optimization | E002 regularizes DDPM training-time `x0_hat` spectra and does not fold maps during training. |
| Hoffmann & Noe, [Generating valid Euclidean distance matrices](https://arxiv.org/abs/1910.03131) | EDMs for point clouds/molecules | Architecture constructs valid 3D-embeddable EDMs by design | Avoids invalid outputs structurally | E002 is a soft auxiliary penalty on an existing U-Net, not a new EDM-generating architecture. |
| Lobashev et al., [Generative inpainting of incomplete EDMs](https://arxiv.org/abs/2404.07029) | Incomplete EDM images for fBm/FISH trajectories | Conditional diffusion/inpainting over missing entries | EDM completion/inpainting task | E002 is unconditional protein distance-map training regularization, not missing-entry imputation. |
| Zhou et al., [Diffusion in SE(3)-invariant space](https://arxiv.org/abs/2403.01430) | Inter-point distance manifold | Projection-free SDE/ODE on invariant geometry | Theoretical diffusion process | E002 keeps the repo's DDPM schedule and adds only a soft spectral auxiliary. |
| Lee et al., [ProteinSGM](https://www.nature.com/articles/s43588-023-00440-3) | Image-based protein structure channels including distances/orientations | Score model plus downstream design/evaluation | Rich 6D structural image representation | E002 remains C-alpha distance-only and adds no orientation channels. |
| Anand & Achim, [Equivariant protein structure/sequence DDPM](https://arxiv.org/abs/2205.15019) | Backbone frames/full-atom coordinates plus sequence | Equivariant denoising diffusion | Coordinate/frame generation | E002 does not change to an equivariant coordinate generator. |
| Wu et al., [FoldingDiff](https://arxiv.org/abs/2209.15611) | Backbone bond/dihedral angles | Periodic angle diffusion | Intrinsic invariance via internal coordinates | E002 keeps pairwise distances and penalizes EDM spectra. |
| Jing et al., [EigenFold](https://arxiv.org/abs/2304.02198) | Sequence-conditioned structure modes | Diffusion over harmonic/eigenmode coordinates | Cascading-resolution conformational sampling | E002 is sequence-unconditioned distance-map training regularization. |
| Jumper et al., [AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | MSA, pair, and atom-frame representations | Triangle pair updates, IPA, FAPE, relaxation | End-to-end supervised structure prediction | E002 adds no triangle architecture/loss and no coordinate-frame objective. |

## What Is Established

The EDM/Gram criterion is classical: centered squared distances yield a Gram
matrix whose PSD and rank properties determine Euclidean embeddability in a
target dimension. Protein-generative prior art already uses C-alpha distance
maps, invariant coordinate/angle/frame representations, diffusion models, and
downstream physical or designability evaluation.

## Engineering Adaptation

E002 applies that EDM spectral criterion stochastically to principal submatrices
of the predicted clean physical distance map during DDPM training. This is a
bounded-memory adaptation for the current variable-length U-Net and recovered
full-data training setup. The scientific hypothesis is that modest gradients
toward PSD rank-3 local/global geometry may reduce the strict geometry failures
seen in E000/E001 without destroying the diffusion objective.

## Novelty Caution

No unsupported novelty claim is made. The most specific claim that can be tested
here is operational: in this repository, E002 isolates a stochastic spectral EDM
auxiliary loss while preserving E001 architecture, data, optimizer, schedule,
normalization, prediction parameterization, and evaluation protocol.

## Numerical Risks

- `eigvalsh` gradients can become noisy near repeated eigenvalues.
- Predicted `x0_hat` may contain negative distances early in training; the loss
  intentionally does not clamp them.
- The spectral loss operates on squared distances, so it does not distinguish
  positive and negative predicted distances.
- The rank-tail denominator can be tiny for degenerate submatrices; `eps` keeps
  the value finite.
- Auxiliary gradients may dominate at late timesteps unless the weight is small
  and warmed up.
- Principal-submatrix sampling is an estimator, not a proof that the full matrix
  is a valid 3D EDM. Soft PSD/rank penalties on sampled submatrices do not
  guarantee global triangle inequalities.

## Proposed Preflight

One-batch CPU no-update gradient profiling was run with the intended subset
size 64 against both fresh E001 initialization and the selected E001 checkpoint.
For the selected checkpoint, candidate weighted auxiliary-to-diffusion gradient
ratios were approximately `0.0586` at weight `0.01`, `0.1759` at weight `0.03`,
`0.2931` at weight `0.05`, and `0.5863` at weight `0.1`. The fresh
initialization ratios were much smaller on the same batch, topping out near
`0.0149` at weight `0.1`. Because `0.01` is the only profiled selected-checkpoint
weight in the requested 5-15% median band and the evidence is only one CPU
batch, it is selected as a proposed preflight starting point, not a settled
scientific optimum.

The initial config uses:

- `physical_auxiliary_loss_weight: 0.01`
- `physical_auxiliary_loss_warmup_steps: 500`
- `edm_subset_size: 64`
- `edm_subsets_per_sample: 1`

The generated profiling JSON files are in this directory.

## CUDA Calibration And Preflight Evidence

CUDA gradient calibration at auxiliary weight `0.01` produced:

| Batch set | Median aux/diff grad ratio | p90 | Maximum |
| --- | ---: | ---: | ---: |
| General full-data batches | 0.0553471 | 0.162501 | 0.183165 |
| Worst-case N=495-500 batches | 0.0166878 | 0.0431464 | 0.0627834 |

The lower N=500 signal is a limitation of using one fixed-size 64-residue
principal submatrix. E002 does not silently correct this with multiscale,
contiguous-window, or multiple-subset weighting because the first experiment is
intended to isolate one clean intervention.

Worst-case CUDA stress:

- lengths 495-500;
- batch size 2;
- gradient accumulation 4;
- 10 optimizer steps plus resume to step 11;
- peak allocated memory approximately 4.13 GiB;
- zero AMP overflows;
- zero skipped updates;
- checkpoint resume passed.

Full-data step-2000 preflight:

- 2000 optimizer steps plus resume to step 2001;
- physical auxiliary weight `0.01`;
- 500-step warmup;
- subset size 64;
- one subset per sample;
- peak allocated memory approximately 4.10 GiB;
- 2 early AMP overflows;
- final AMP scale 16384;
- final consecutive overflows 0;
- no non-finite losses;
- checkpoint serialization/resume passed.

Median first-versus-last-quartile losses:

| Quantity | First quartile median | Last quartile median |
| --- | ---: | ---: |
| Diffusion loss | 0.0578014 | 0.0248815 |
| Total EDM auxiliary loss | 0.329871 | 0.0849254 |
| Negative-spectrum loss | 0.140084 | 0.0298418 |
| Rank-3 loss | 0.188657 | 0.0518208 |
| Weighted auxiliary loss | 0.00129356 | 0.000849254 |

## Step-2000 Generative Screen

The paired step-2000 screen used 10 exactly paired samples: lengths 64, 128,
256, 384, and 500, two samples per length, with identical sampling seeds between
E001 and E002.

Observed effects:

- rank-3 residual improved in 10/10 samples;
- mean rank-3 improvement was `0.0370193`;
- negative eigenvalue mass improved in 7/10 samples;
- mean negative-eigenvalue-mass improvement was `0.00078317`;
- triangle violation fraction worsened in 10/10 samples;
- negative-distance fraction worsened or remained tied;
- reconstruction diagnostics generally improved, especially at `t=499`, where
  x0 RMSE improved from `12.184747` to `11.810533` Angstrom and negative
  reconstructed fraction improved from `0.002038` to `0.000924`.

Interpretation: E002 produced the intended targeted rank-3 spectral effect, but
strict EDM validity has not yet been achieved. The worsened triangle and
negative-distance diagnostics are consistent with the scope of the current loss:
it regularizes squared-distance Gram spectra on sampled submatrices, not
distance sign or all global metric inequalities.

Scientific decision:

- keep E002 spectral-only for causal attribution;
- run one complete epoch from scratch before considering triangle or
  non-negativity terms;
- do not fold triangle, non-negativity, projection, contact, or radius losses
  into E002.
