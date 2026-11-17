# E010 student direction observability v1

Completed identity-held-out observability diagnostic. Feasibility-margin commit 1de4619, student fc76211 and teacher/corpus 6d1ed30 remain frozen. No marginized-target training, increased capacity, new teacher data or SLSQP.

All artifacts: `/mnt/d/Simone/proteinGen/outputs/e010_student_direction_observability_v1`. Full report: `REPORT.md`; complete context/length/condition/step aggregates: `results_AB.json`; independent checks: `tests.json`.

The exact frozen input, local frame, penultimate latent and teacher action are archived for all 2,000 identities. Neighbor matching uses eight deterministic query residues per each of 96,000 logical K16 states and four reference residues per TRAIN state. The reference bank contains only the original 1,400 student TRAIN identities. Every query excludes its identity, split group and sequence cluster. Frozen validation/test each contain 300 identities.

| Representation | Test median teacher-direction cosine | Opposed teacher directions |
|---|---:|---:|
| geometry | 0.7488 | 12.91% |
| latent | 0.8924 | 7.68% |

| Added context | Validation median cosine gain | Directional variance explained | Gate |
|---|---:|---:|---|
| condition | 0.0024 | 0.40% | False |
| residue_identity | 0.0026 | -0.39% | False |
| sequence_local | 0.0025 | 0.24% | False |
| sequence_condition | 0.0039 | 0.33% | False |
| step | 0.0079 | 1.53% | False |
| coarse_scalar_latent | -0.0032 | -1.36% | False |
| initial_coarse_correction | 0.0113 | 2.02% | False |
| denoiser_geometry | 0.0855 | 17.36% | True |
| rollout_history | 0.2511 | 72.60% | True |

Context reranking uses inference-available features only and is compared with geometrically distance-matched controls. X and safety margins are offline outcomes, never matching features. Cached denoiser target-error metrics are excluded. Zero actions have undefined direction and are counted separately.

Denoiser geometry passed the validation-only observability gate, triggering the five authorized same-capacity controls. Their raw deployment evaluation uses no X-based shield. See the controlled comparison below. Results provide finite-resolution evidence about privileged-policy ambiguity in the tested representations, not proof of information-theoretic unobservability. Corpus density, neighbor metric, local receptive field and sampled spatial coverage remain limitations. Teacher-path history is assessed separately because student rollout history differs.

No inference safety improvement is claimed from a neighbor diagnostic. The undeployable X-based shield was not used as a deployment solution. Frozen source hashes, identity-list hashes, exact model parity, all reference exclusions and ANN recall checks are documented in the D:-resident report and completion manifest.

| Student control | Test raw K16 repair retention | Raw unsafe-state rate | Mean teacher-action cosine |
|---|---:|---:|---:|
| A | 59.61% | 82.85% | 0.7808 |
| B | 58.00% | 83.43% | 0.7532 |
| C | 62.05% | 85.84% | 0.7937 |
| D | 57.56% | 85.19% | 0.7667 |
| E | 58.34% | 82.62% | 0.7825 |

Validation-selected minimal useful context: None. Full raw identity/length/condition/step, directional-error AUC, latency and exact-gate metrics are in `phase_C/results.json`. No further model, teacher-data, full-corpus or final-training action followed.
