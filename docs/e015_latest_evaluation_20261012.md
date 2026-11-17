# E015 latest durable evaluation, 2026-10-12

The interrupted evaluation continued in
`e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/evaluations/latest_complete_u10382_4951d838_v1/`.
Its scientific primary is the immutable snapshot of `guard_stop_update_10382.pt`:
SHA256 `4951d8387ebd2bce5273be9580dd26989f067e770fd124bc6bb1c12f5da8d32f`,
314,256 exposures, epoch 2, cursor 84,124, one completed epoch.
Selection used a supported graceful pause and durable accounting, not measured performance.
Later operational pause checkpoints are recorded separately in `recovery_resume_proof.json`.

The existing frozen 100 TRAIN/100 held-out identities were preserved. Their noise
families and the historical generation initial tensors were reused. All twenty
primary samples reproduce the saved physical X499 bit for bit under the original
normalized FP32 sampler contract. Historical full sampling was not rerun.

Raw held-out v-MSE is 0.439831, versus EMA 0.627664. The same full 100-identity
held-out panel gives 0.464648 for the 50% checkpoint, rather than its earlier
50-identity summary of 0.4475. Latest TRAIN/held-out aggregate local RMSE is
2.58656/2.59236 Å; its gap is 0.00579 Å (95% interval −0.05828 to 0.07035).
At t5 held-out Kabsch/local RMSE is 0.42213/0.23317 Å. At t490 it is
16.827/5.3079 Å; direct clean-estimate shrinkage must be distinguished from the
actual reverse trajectory.

All twenty raw and five bounded secondary EMA samples finish with finite
coordinates. Same-start raw adjacent-valid fraction improves from 63.260% to
83.783%, with mean clash fraction falling to 0.000218. Length-192 full descriptor
energy distance falls from 9.47 to 4.459, but its real-real floor is 0.662.
Local descriptor energy improves at all five lengths. Global descriptor energy
has a paired interval wholly below zero at only two lengths; length448 has a
worse point estimate and an interval spanning zero. This supports material local
polymer improvement, while evidence for consistent global fold formation remains
weak. Verdicts are PROMISING_BUT_IMMATURE, BOUNDED_BUT_NOT_PROTEIN_LIKE,
UNDERFIT, and CONTINUE_BUT_WATCH.

The implementation streams batch1/worker0 inference and compact scalar tables,
uses 5,000 stratified protein bootstrap draws, and records sparse trajectories,
head contributions and nineteen typed-state stages. Fixed-scale storyboards,
same-start comparisons, distance maps, head arrows, cards and three bounded GIFs
explain inference. Numerical ranking determines galleries. No full activations
or every-step trajectory dumps are saved.

A legacy twelve-frame montage failed after one completed thirteen-frame sample.
The supported owner resumed production, later paused at a new durable boundary,
and continued from the saved sample and reconstruction tables. A final serialized
report-only phase corrects card/storyboard spacing without repeating scientific
inference, bootstrap or sampling. A renderer name-shadowing error safely returned
ownership to production before the correction was retried. A model-free report
control-flow check now exercises all25 cards,15 stories and35 comparison calls
with rendering stubbed in a temporary directory explicitly onD. Operational logs
preserve these events.

Production configuration and architecture were unchanged. Heavy checkpoint
loading, evaluation, bootstrap and image creation occurred only with the trainer
stopped and GPU ownership verified. All new files/caches/temp are on D:; the
evaluation is approximately 0.66 GiB before final metadata. E007/E012/E014 were
not modified or resumed. The pre-existing untracked absolute-evaluation helper
was preserved. Final resume identities, exposures, rolling hash, PIDs, resource
summary and commit are recorded in the evaluation report and manifest.

Validation: Ruff checks on the eight new helper modules; bounded scalar/footer
validation of 10,000 reconstruction rows, 5,000 extra held-out rows, 25 samples,
325 frames, finite typed statistics and populated V/A vector norms; GIF sizes
below 0.31 MiB each. Strict production checkpoint and deterministic resume
validation are handled by the existing recovery implementation before each GPU
ownership transition.
