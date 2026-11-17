# Experiment map and evidence boundaries

This map supersedes the old README's open-ended E010 roadmap. The detailed [historical timeline](experiment_timeline.md) remains unchanged. An experiment number names a research question, not one accepted model. The [branch inventory](publication/branches.md) pins all inspected tips; [file provenance](publication/integration_sources.json) identifies every selected import.

| Experiment | Question / architecture | Executed evidence and interpretation | Authoritative location |
|---|---|---|---|
| E000 | Length-conditioned distance diffusion | Baseline trained/evaluated; finite matrices, 0/375 strict calibrated geometry acceptance. | Historical main; distance U-Net and diffusion package. |
| E001 | Symmetric axial attention | Trained/evaluated; selected global metrics improve, strict validity still zero. Histories differ, so not a controlled causal estimate. | Historical timeline and axial configs on main. |
| E002 | Spectral Euclidean distance-matrix regularization | Trained/evaluated; improved global realizability trades off with adjacent geometry. | Historical matrix baseline/configs on main. |
| E003 | Adjacent-distance auxiliary loss | Trained/evaluated; modest local gains with spectral/triangle regression; not promoted. | Historical timeline on main. |
| E004 | Triangle multiplication | Trained/evaluated; relational consistency improves without establishing protein-like traces. | Triangle model/tests on main. |
| E005 | Sequence/geometry feedback | Implemented and bounded training; structural utility remains ambiguous. | Co-design modules and historical timeline on main. |
| E006 | Rich backbone features and scratch sequence prior | Data/provenance infrastructure implemented; contextual usefulness and definitive joint Stage B not established. | Rich co-design modules on main. |
| E007 | Equivariant coordinate diffusion | Denoising learned; sampling/local geometry limitations. Later matched full-scale run is a separate lineage. | Main coordinate diffusion; `e007_matched_fullscale_v1` for later control. |
| E008 | Fixed-bond internal-coordinate decoder | Corrected tiny-overfit RMSE 3.591 Å fails 0.5 Å gate. | Kinematic decoder and tests on main. |
| E009 | Bayesian geometry-prior refinement | Bounded diagnostics, not a validated generalizing refiner. | Bayesian refiner/tests on main. |
| E010 | Global equivariant refinement and teacher/student diagnostics | Phase 4B trained/evaluated; Cartesian/local conflict. Later feasibility and observability studies do not establish a deployable student. | [Frozen auxiliary](e010_local_geometry_auxiliary.md), [observability](e010_student_direction_observability_v1.md); `e010-phase4c-local-auxiliary`. Divergent hybrid/KKT studies remain on `e010-phase4d-hybrid-local-global`. |
| E011 | Geometry-free masked context controls | Implemented; bounded S1 context pilot recorded. | [Report](e011_sequence_context_only.md), selected context module/tests; `e011-sequence-context-only`. |
| E012 | Causal RoPE sequence model | Implemented and trained through bounded continuation; likelihood/context protocols and retrospective generalization audit. Large-corpus acquisition is a separate engineering track. | [Protocol](e012_causal_rope_sequence.md), [audit](e012_generalization_audit.md); `e012-causal-rope-sequence`. |
| E013 | Unconditional equivariant coordinate denoiser | Implemented/trained; bounded v-only result closes failed structural gates. | [Report](e013_unconditional_equivariant_denoiser.md); `e013_unconditional_se3_coordinate_generator_v1`. |
| E014 | Coordinate capacity scale-up | Implemented and evaluated under recorded exposure; exposure and hardware contention complicate interpretation. | `e014_unconditional_coordinate_v_scaleup_v1`, later matched E007 branch. |
| E015 | Typed multiscale parity and self-conditioning | Implemented/trained/evaluated; local polymer improvement, immature global folds. | [Durable evaluation](e015_latest_evaluation_20261012.md); `e015_integrated_multiscale_parity_selfconditioned_v1`. Selected model/loss/EMA/PCGrad/self-conditioning primitives on integration branch. |
| E016 | Joint autoregressive amino acids and signed internal geometry; BC/SAC foundations and reward audits | Real-data BC and continuations ran. Better native-prefix likelihood does not imply improved free generation; +30k continuation rejected. Biological reward reliability is not certified. | [Original J1 continuation report](results/e016_global_pretraining.md); E016 branch family. Selected contracts, builder, action density and SAC targets on integration branch. |
| E017 | Frozen-reference constrained geometry post-training | R3 ran 5,000 updates/80,000 attempts; final-evaluation failure and original promotion NO-GO remain. Expanded V9.5 validation supports some yield gains, not independent replication or established meaningful goal control. V9.6 stopped asymmetrically. | R3 `91576ff071bd42a07f4306bc315b496d22d5a3bf`; [V9.5](results/e017_v95_validation.md), [V9.6](results/e017_v96_recommendation.md). |

## Latest E017 state

The expanded V9.5 validation used 768 identities and 1,536 primary goals. Filtered K1 improved by 1.758 percentage points (95% interval 0.260–3.190), but the original promotion expression remained false. Previously selected validation identities are development evidence, not independent replication. Attempts, completed chains, filtered outcomes and raw policies must not be conflated.

V9.6 arm A stopped at 500 updates/8,000 attempts and B at 250/4,000. One seed per arm and unequal budgets do not establish token-versus-extra-training causality. Later records mark historical TEST exposure **unknown**, superseding any assumption that an intact final holdout has been certified.

V9.7's [engineering report](results/e017_v97_engineering.md) precedes its [initialization admission](results/e017_v97_admission.md). The latter passes the frozen 4,096-candidate admission for both arms but reports zero production updates and no production authorization. These are qualification results, not demonstrated RL efficacy. Prepared production scripts in the snapshot are not evidence that production ran. No later completed production report was identified in the inspected evidence roots.

## Reading the evidence

“Implemented” means source exists. “Trained” requires a recorded optimizer run. “Evaluated” requires a specified population/protocol and saved result. “Proposed” means no execution evidence is claimed. Synthetic test passes establish software properties, not biological quality. Textual reports imported here are original records; their raw data/checkpoints were not recomputed in this integration. Their SHA256 and original evidence names are recorded in the file provenance manifest.
