# E010 Phase 4B forensic closeout

Scientific audit: 2026-10-02. Development consolidation: 2026-10-04–05.
Audited HEAD: `b9d08d1b6cc24bf203c9ac629c7fe29d8198ec32`, with the then-existing
staged research implementation and unstaged E004 architecture addendum.
Claims below are verified unless explicitly labeled as hypotheses or unknowns.
This record preserves historical measurements; it does not authorize training.

## Implemented system

The repository retains length-conditioned distance-matrix diffusion (E000–E004),
coordinate VP diffusion (E007), geometry-native decoder prototypes (E008), a
Bayesian geometry-prior refiner (E009), and global equivariant residual refinement
(E010). Sequence generation and sequence/geometry co-design are separate research
implementations; no validated E010 structure-to-sequence production pipeline is
established.

E010 Large is a 12,844,352-parameter refiner: width 416, six global attention
blocks, eight heads, and 64 polar-vector channels. It consumes C-alpha coordinates
in Å and residue masks, combines invariant distance/position features with
relative-vector updates, and returns coordinates plus residuals. Distances are
derived from coordinates rather than an independently generated distance head.
The refiner receives no explicit corruption timestep. Its symmetry includes
reflections: O(3), and E(3) including translations. It lacks a handedness preference.

## Training integrity and objective

Phase 4B actually optimized: all 129 refiner parameter tensors changed from
initialization, all expected parameters were saved, and inspected Adam states
reached 1,092 updates. This refutes the whole-refiner-frozen/no-update explanation.
Successful numerical optimization is distinct from successful scientific learning.

For valid mask m, target X, frozen-denoiser input C, prediction P and residual
P−C, each protein contributes

```text
L_cart = sum_i m_i ||P_i − X_i||² / (3 sum_i m_i)
       + 1e-5 sum_i m_i ||P_i − C_i||² / (3 sum_i m_i).
```

Proteins are averaged equally, with equal length-stratum allocation. Coordinates
are in Å and the loss in Å². There is no alignment in the loss, pairwise-distance
loss, or local i+1/i+2/i+3 loss. Consequently O(N²) pair-loss dominance does not
explain Phase 4B. AdamW uses LR 0.0003, zero weight decay, betas (0.9, 0.999),
epsilon 1e-8 and global gradient clipping at 5. The full schedule contains 98,280
examples and 1,092 updates; conditions 50/250/450 have one-third exposure each.
These are cosine-VP corruption timesteps, not chain lengths.

Implementation: `reports/experiments/E010_global_equivariant_expressivity/phase4b_real_denoiser_v1/runner.py`,
`prepare_cache.py`, and `src/protein_distance_diffusion/models/e010_global_equivariant.py`.

## Historical corruption versus actual population

Historical `7ar7_M.npz` contains internal ID `7ar7_m` and 67 residues instead of
487. Raw mmCIF contains distinct M and m chains; the active case-insensitive
filesystem aliases their archive paths. This is a confirmed data/provenance
failure, not evidence that corrupted data entered Phase 4B.

Multicorruption v1 detected mismatches before training. Corrected v2 records
43 exclusions: 18 coordinate-length mismatches, 14 payload-ID mismatches and
11 physical aliases. The selected v2 training pool has 16,384 identities,
98,280 corruptions and 320 development identities. Length strata are 20–64,
65–128, 129–256, 257–384 and 385–500. Exclusions change the longest-stratum
identity count to 1,563 and the preceding stratum to 3,730; example exposure
remains balanced. Actual Phase 4B sources and resolved training/evaluation
populations had zero detected integrity violations. Historical malformed
archives and aliases were excluded; they are confirmed non-causal for this run.

Evidence: multicorruption v1 `source_cohort_discrepancy.json`; v2
`excluded_archives.json`, `training_seed_manifest.json`, `preparation_manifest.json`;
Phase 4B cache `manifest.json`; preserved population-integrity audit.

The consolidation adds publication-time identity collision, residue-count and
finite-coordinate checks with explicit rejection records. It preserves original
archives and does not rewrite exclusions, allocations or historical manifests.
The guard inspects an existing resolved destination; it is not a cross-process
filesystem transaction or a dataset migration.

## Results and metric meaning

The paired gain is the mean identity-level fractional improvement in proper-rotation
Kabsch-aligned coordinate RMSD. Conditions are averaged within identity before
paired bootstrap resampling. Local metrics are C-alpha distance-error RMSE at
sequence offsets 1/2/3, averaged by example; these are different quantities.
All three boundaries were classified `insufficient_real_domain_gain`.

| State | Mean aligned RMSD Å | Local i+1 Å | Local i+2 Å | Local i+3 Å | Chirality inversion |
|---|---:|---:|---:|---:|---:|
| Frozen denoiser | 7.897440 | 1.847725 | 2.351456 | 2.964447 | 0.422091 |
| Zero-shot synthetic E010 | 7.922613 | 1.275901 | 1.515996 | 2.433958 | 0.421415 |
| Adapted 364 | 7.799341 | 2.090159 | 3.022185 | 3.605858 | 0.427597 |
| Adapted 728 | 7.779544 | 2.003023 | 2.961367 | 3.569433 | 0.422605 |
| Adapted 1092 | 7.768927 | 1.904637 | 2.860713 | 3.477009 | 0.424098 |

Final paired gain: 0.012499211, bootstrap 95% CI [0.009366693, 0.015672626].
Mean local-distance improvement: −0.140091736 (−14.01%). Finite outputs,
coordinate/diversity non-collapse and length-slope gates passed; chirality
not-increased failed. Modest aggregate gain and local regression coexist.
Local geometry recovers partly from 364 to 1092 but remains worse than the frozen
baseline and markedly worse than synthetic zero-shot. Intermediate 364/728
weights are unavailable; saved boundary metrics are not recoverable checkpoints.

## Gradient and finite-displacement evidence

The fixed training panel is `1bf9_A` (41), `1chp_D` (103), `1czz_A` (187),
`1fhf_A` (304), `1k6m_A` (432), at all three conditions. Independent development:
`2elx_A`, `2gmg_A`, `3pdd_A`, `1og6_C`, `1w26_A`, also at all three conditions.
The diagnostic local objective averages the three endpoint-masked distance-error
MSEs; evaluator RMSE agrees numerically within 5.2e-7 Å. These five-identity panels
are mechanistic probes, not population-level estimates.

Initialization mean-local raw gradient cosine: −0.910426; final: −0.698754.
Adam local directional derivatives: +1.375651 and +0.136074 respectively.
Positive derivatives along the Adam displacement predict harm. Initialization
uses fresh Phase 4B moments; final uses saved 1,092-update moments.

Independent parameter copies at α = 0, 0.01, 0.1 and 1 evaluated θ₀+αΔθ, holding
noise, masks and proteins fixed. No optimizer step, moment advance or retraining
was performed. At final α=0.01, the training panel changes were:

| Quantity | Observed change |
|---|---:|
| Cartesian objective Å² | −0.002595917 |
| Mean local MSE Å² | +0.001361563 |
| Mean evaluator local RMSE Å | +0.000190822 |

The independent development panel, along the same training-derived displacement,
changes by −0.001751137 Å² Cartesian, +0.001369381 Å² local MSE and +0.000216651 Å
mean-local RMSE. Thus finite objective conflict is confirmed (F1); curvature is
also relevant (F3), especially at initialization. At final α=0.01 condition 450
accounts for approximately 90.5% of Cartesian improvement and 78.2% of local-MSE
harm. The common displacement harms condition 50 despite its isolated gradient
being locally compatible. Multi-condition competition is a strong candidate
mechanism, not yet a validated intervention.

Final short/middle examples worsen locally while the two longest improve. With
one identity per stratum this does not establish a length law. One initialization
`1k6m_A` condition-averaged RMSE sign reversal at α=0.01 remains documented:
predicted −0.000086063 Å versus observed +0.000085346 Å. Individual conditions
have correct signs; cancellation makes the aggregate sensitive to curvature.
This localized exception does not erase the mixture-level confirmation.

Checkpoint SHA-256:

- Initialization: `e2a369d9c43617918056ea0ac5bb5f161ccb7adc0a829d694fecf9be4ed81c16`
  (`phase4a_multicorruption_v2/phase4a_multicorruption_v2.final/checkpoint_update_1092.pt`).
- Final: `f5211cbc1be5092175ce15b9761a4efba761d242310287cd6b1e04df6a6744ef`
  (`phase4b_real_denoiser_v1/phase4b_training_v1.final/latest.pt`).

## Chirality and reporting corrections

The evaluator uses signed tetrahedral volumes, detects reflection, and is
invariant to proper rotation/translation. Kabsch enforces determinant +1.
Its historical zero-eligible-volume behavior reported zero inversion: a confirmed
blind spot. Future reports now expose eligible counts and assessability, and
unassessable chirality blocks authorization. The legacy zero numeric sentinel
remains for compatibility and must not be interpreted without assessability.
Historical reports have not been recomputed.

The architecture's reflection-inclusive symmetry is a separate limitation:
coordinates carry orientation, but the feature/update design lacks a preference
for handedness; distances alone cannot distinguish mirrors. Degenerate metric
coverage is not an explanation for the architecture's symmetry.

The historical `training_development_squared_error_gap` combined cumulative
unaligned component MSE with squared mean aligned RMSD: different reductions,
factor-of-three conventions, frames and observation times. It was not a valid
generalization gap. Future reports relabel it `legacy_incomparable_error_difference`
and explicitly explain the incompatibility. Losses and historical artifacts remain
unchanged.

## Limits and next controlled experiment

Train/development identity separation does not establish homology independence.
The forensic audit found shared sequence clusters and at least one development
identity exposed to frozen E007 training. Independent development here means
independent of the five-protein diagnostic training panel, not a leakage-free
prospective biological test. No production checkpoint is authorized.

The original next step was a frozen condition-weight counterfactual. Both
pre-registered diagnostics are now complete: CF1 `(5/12,5/12,1/6)` was W2,
partial improvement; CF2 `(11/24,11/24,1/12)` reversed aggregate training local
harm but retained only 66.26% of Cartesian descent, below the 70% gate, and left
development local harm. CF2 was X4 under that retention gate. Independent
alpha=0.01 copies confirmed the responses; no training or optimizer steps ran.
See [the completed condition-weight record](e010_condition_weight_counterfactual.md)
for both diagnostics, gate results, projection contributions and finite changes.

Weight-only tuning is closed; CF2 should not be trained. The next question is
`L_cart + lambda * mean(L_local_i+1,L_local_i+2,L_local_i+3)`, restoring historical
equal condition weights `(1/3,1/3,1/3)`. Pre-register the coefficient choices for
a frozen-state diagnostic before any retraining. No auxiliary objective or
architecture modification was implemented by the completed diagnostics.

Separate later tracks: objective/local-geometry balancing; orientation-sensitive
chirality features; EDM/coordinate consistency; sequence conditioning only after
a stable structural interface. Change one main variable per controlled experiment.

## Evidence preservation

Raw forensic reports, catalogs, population audit, diagnostic scripts/results,
sample/noise manifests and hashes remain outside Git in
`/home/simostocco/proteingen_audit_artifacts/20261002/`, including
`gradient_diagnostic/` and `finite_displacement_diagnostic/`. These locations are
provenance, not package runtime dependencies. Checkpoints, NPZ datasets and large
raw diagnostics are intentionally excluded from this baseline. Historical source
byte pins must be re-reviewed for future execution after formatting/repair; no
old manifest or pin was rewritten to accommodate current code.
