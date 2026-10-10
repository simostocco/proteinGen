# E007 versus E014 coordinate comparison

The primary outcome is **C5: architectural attribution remains confounded**.
Existing E007 checkpoints are **M2** for a matched architectural comparison.
This read-only historical comparison uses fixed endpoints, frozen validation
identities, identical noise tensors and Gaussian initial states. It does not
change E007, E014, historical reports, or the sequence branch.

## Comparator provenance

| Comparator | Parameters | Exposures | SHA256 |
| --- | --- | --- | --- |
| E007 fixed 10,000-update review endpoint, primary | 7,586,505 | 18,000 | acbb8250a09a45bf79a48e017d5b278e748cb630c275a6860f2b1ccf635eb65f |
| E007 previously published 9,000-update selection, sensitivity | 7,586,505 | 16,200 | eb445f0b39067b8a00a47db27f966a6db97dc0e30bf81078b34ea54f111c7d82 |
| E014 epoch 2, primary minimum-exposure boundary | 5,119,069 | 460,264 | 61a3acfe1024c9a6adcb21bfd864b3d0980e978d2fb90cbb02d9f1d095b9d902 |
| E014 newest, partial epoch 3, secondary | 5,119,069 | 607,018 | 935337d559f87b079a0bc9bb8bce6ecf90e971840d59c4b6163f036697c0f4e7 |

Both use the same 230,132-identity eligible TRAIN universe, coordinate scale
12.22820347644835 Å, 500-step cosine diffusion and centered coordinate-v target.
Historical E007 actually saw 18,000 distinct identities, whereas E014 epoch 2
completed two passes over all TRAIN identities. Training streams, learning rates,
validation/stopping rules and batch loss weighting also differ. E007's historical
training code commit is unrecorded. The checkpoint inventory preserves this
uncertainty rather than inferring a commit.

## Primary scientific results

Statistics use equal protein weighting, the frozen 500-protein panel, and paired
length-stratified 5,000-draw bootstrap. Differences are E014 minus E007. Aggregate
errors average the five frozen core timesteps within protein.

| Primary metric | E007 10k | E014 epoch 2 | Difference, 95% paired CI |
| --- | --- | --- | --- |
| Aggregate v-MSE | 0.446275 | 0.402642 | −0.043634 [−0.045578, −0.041882] |
| Aggregate local distance MSE, Å² | 8.29475 | 11.21243 | +2.91768 [+2.86666, +2.97069] |
| Gaussian-prefix energy at t=400 | 0.063567 | 0.036733 | −0.026834 [−0.121177, +0.063075] |
| Gaussian-prefix energy at t=350 | 0.095111 | 0.129043 | +0.033932 [−0.108647, +0.165641] |
| Torsion angular error at t=400, radians | 1.570296 | 1.586380 | +0.016084 [+0.003750, +0.028308] |
| Protein-weighted inversion rate at t=400 | 0.500796 | 0.503377 | +0.002582 [−0.003938, +0.009245] |

The six-statistic family also has Bonferroni 99.1667% intervals in the report.
The torsion difference crosses zero in that familywise interval. These results
establish lower v-MSE and worse aggregate local geometry for the historical
E014 comparator, without an E014 chirality or Gaussian-transport advantage.
They do not identify the cause of the difference under unmatched training.

E014 improves low-noise local geometry and torsion. E007 has better L1/L2/L3
geometry at mid/high noise; both approach chance-level inversion there. E014
epoch 2 improves some extreme-noise paired step gains principally through
smaller predicted amplitude, without a consistently extended directional front.
One-step high-noise marginal effects remain weakly resolved. Both models move
Gaussian noise toward the noisy-protein marginal within the bounded prefixes;
no population/length-stratum scale collapse was observed through t=350. Direct
X0 estimator collapse is recorded separately and is not rollout-state collapse.

## Limits and next experiment

A matched E007 control remains necessary for architecture attribution. Its
frozen configuration is prepared, but the required E014 final-stop report and
resource-approval record are absent. A completed minimum-exposure adjudication
does not satisfy the separate final-stop trigger. No training is launched and
no gate is fabricated. Synthetic forward/backward benchmarks use disposable
copies without optimizer updates; full training memory is not certified by them.

E014 remains Case E. The original Case A-only Stage D gate remains closed; no
full t=0 sampling is performed. Geometry/chirality is an observed bottleneck,
while architecture versus diffusion-objective causality remains unresolved.

The archived E014 descriptor replay envelope (~8.96e−6) exceeds its unchanged
5e−6 tolerance. That test concerns the epoch-1 50% checkpoint, not a new epoch-2
certification. E007 replay envelopes are reported separately. Scientific
uncertainty and exact replay certification remain distinct.

## Durable evidence

All tables, raw descriptors, statistical intervals, latency/memory benchmarks,
replay records, nine plots and preservation hashes are on D: under
`e014_artifacts/e007_vs_e014_matched_v1/` in this worktree. See
[the comprehensive report](../e014_artifacts/e007_vs_e014_matched_v1/REPORT.md),
[matching assessment](../e014_artifacts/e007_vs_e014_matched_v1/matching_assessment.json),
[compute comparison](../e014_artifacts/e007_vs_e014_matched_v1/compute_comparison.json),
and [preservation audit](../e014_artifacts/e007_vs_e014_matched_v1/FINAL_PRESERVATION_AUDIT.json).

New comparison code is covered by focused protocol/bootstrap tests. Historical
scientific records remain intact. E014 and sequence branch references remain
unchanged; comparison code and this document are committed on a separate branch.
