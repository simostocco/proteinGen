# E010 Phase 2 interference diagnostic v1

This is a non-authorizing, training-panel diagnostic. Phase 2 v1 and v2 remain unchanged; no joint updates, pilot preparation, held-out evaluation, geometry loss, or prior was introduced. The machine-readable result is in [`diagnostic_result.json`](../reports/experiments/E010_global_equivariant_expressivity/phase2_interference_diagnostic_v1/diagnostic_result.json), with verified input pins in [`input_audit.json`](../reports/experiments/E010_global_equivariant_expressivity/phase2_interference_diagnostic_v1/input_audit.json).

## Selection and checks

The representative in each stratum was selected by the smallest SHA-256 rank of `e010_phase2_interference_diagnostic_v1|sample_id` among the fixed 32 training identities. Selection used only identity and length metadata. All 32 corruption archives and target/corruption/mask tensor hashes were checked. The update-6400 checkpoint file and its embedded model, optimizer, Python/NumPy/Torch RNG, identity schedule, exposure and loss-history hashes matched the checkpoint manifest before use.

| Stratum | Representative | Length | Fresh isolated first passage | Update-6400 specialization first passage |
|---|---:|---:|---:|---:|
| 20–64 | 9azi_A | 27 | 137 | 1 |
| 65–128 | 1ff2_A | 106 | 118 | 3 |
| 129–256 | 3b7m_A | 216 | 217 | 17 |
| 257–384 | 4y2h_B | 352 | 278 | 26 |
| 385–500 | 1r89_A | 437 | 379 | 25 |

All first passages were to aligned RMSE ≤0.10 Å; none of the five fresh runs reached the 1,000-update cap, and all five specialization runs passed within the predeclared 50-update quick window. Full per-update RMSE trajectories, sampled local-distance/chirality telemetry, per-step gradient and optimizer-step norms, clipping indicators, runtimes and CUDA memory are recorded in the JSON result.

At first passage, aligned RMSE ranged from 0.0927 to 0.0997 Å. Fresh isolated clipping fraction was 0 for every run. Specialization clipping fraction was also 0 for every run. Per-run peak allocated CUDA memory ranged from 61,191,168 bytes at length 27 to 297,244,160 bytes at length 437; peak reserved memory was 599,785,472 bytes. Chirality inversions at the first-passage telemetry point ranged from 0 to 5; the local i+1/i+2/i+3 distance RMSE values are included per run in the result.

## Read-only gradient audit

The 32 coordinate-MSE gradients were computed independently at the exact update-6400 weights. `autograd.grad` was used; there were zero optimizer steps, and the model and optimizer state hashes were unchanged. The complete cosine matrix and per-identity norm/contribution records are in the result.

- Across 496 unordered gradient pairs, mean cosine was 0.4015, median 0.3998, minimum −0.0560, and 3/496 (0.60%) were negative.
- Within every length stratum, no pair was negative. Across strata, the only negatives were 3/42 (7.14%) pairs between the 20–64 and 385–500 strata. All other between-stratum groups had zero negatives.
- Mean gradient norm was 0.3981. Individual gradient norm increased with length in this panel (linear slope 0.001553 norm units/residue; correlation 0.7232). Signed projection onto the mean gradient had slope 0.003570/residue. Per-identity cosine with the mean gradient, signed projection and norm share are reported in the audit.
- Audit peak CUDA memory was 344,014,848 allocated / 585,105,408 reserved bytes; runtime was 1.66 seconds.

## Interpretation

The v2 joint checkpoint still missed its update-6400 gates (mean 0.2853 Å, median 0.2555 Å, maximum 0.5129 Å, length slope 0.000739 Å/residue), while all five metadata-selected identities passed both isolated fitting and rapid checkpoint specialization. This argues against a long-length expressivity failure for these five examples and supports a shared-model limitation. The gradient audit shows largely compatible directions, with rare negative pairs concentrated only at the shortest-versus-longest strata; it does not show broad cross-length gradient conflict. Under the predeclared interpretation, increasing shared capacity is the better-supported next model hypothesis, while the limited negative cross-stratum signal can be monitored if a future optimization schedule is considered. This diagnostic alone does not authorize a model change or downstream pilot.

Fresh first-passage updates rose with length (118–137 in the two shortest strata, then 217, 278 and 379). All five passed within the 1,000-update cap, but this length dependence is descriptive support for accounting for optimization difficulty in any later exposure schedule.

## Integrity pins

- Update-6400 checkpoint SHA-256: `26fd7125ea05729656c11734827bf1ed4d7ad67a238429724217da7a02b7787b`
- Checkpoint manifest SHA-256: `bf903674484857cd6fbdbb72b2ecdb89605b13e7dedab7fafbcb41f2d50c933f`
- Phase 2 v1 protocol SHA-256: `5fe84a33884f38334f5b64828cc518c2d5ea6e32f5fe6c532df8fd5b463321a2`
- Phase 2 v2 result SHA-256: `d0a73ec846930f145697f0afc568a3738a07cc51d77448215825e7905b4391d7`
- Diagnostic runner SHA-256: `51e1609641ae21bbb42823a7677f51e5b38500f2a4aececb5353a9f16d0bb3a5`
- Diagnostic config SHA-256: `dc2095fc52185e42a854287df83fd83a4738c48ac47398fc137402e1522549bd`
- Diagnostic result SHA-256: `d7caa3bdaa7ddbbcaa82989882be8c7f19357105aa305fc6cf38c75f27b68fa6`
