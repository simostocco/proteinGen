# E007 Phase 3I.2 final pilot relocation correction

## Cause and correction

The repository's logical path is `/home/simostocco/proteinGen`; its D-backed physical path is `/mnt/d/Simone/proteinGen`. On this case-insensitive active tree those names reach the same files. The old `PHASE4C_PROTECTED_INPUT_RELOCATION` therefore mapped the logical root back to the active tree and could not recover the authoritative case-collided bytes.

The authoritative preserved verification tree is the independent tree `/home/simostocco/proteinGen.pre-relocation-20260922T064202Z`. The final pilot now carries this policy:

```yaml
dataset:
  protected_input_relocations:
    - recorded_root: /home/simostocco/proteinGen/data
      verification_root: /home/simostocco/proteinGen.pre-relocation-20260922T064202Z/data
```

The recorded root identifies inventory paths. The verification root identifies the preserved bytes. Neither is an alias for the active physical repository path.

## Protected byte evidence

| File | SHA-256 | Size |
| --- | --- | ---: |
| Active case-collided `data/full/processed/samples/7ssn_D.npz` | `01658ec1722370b1394ffe7a56de8cd5e66c2bb295ddf9c8054f640d81b56d07` | 142,072 bytes |
| Preserved authoritative `data/full/processed/samples/7ssn_D.npz` | `99b26c65e29601430dea6810feff24f7b4a6a8c8ed255de6621448e8e2fdafd2` | 9,211 bytes |

Both files are preserved in place. No dataset file was mutated.

## Validation evidence

The production configuration passed read-only contract validation with 3,736 protected entries: 3,725 direct resolutions, 11 relocated resolutions, zero missing, and zero unresolved contradictions. The relocation list reports each relocated inventory identity with its authoritative hash, including `7ssn_D.npz` at the preserved hash above. Training schedule reconstruction produced 500 updates; evaluation panel reconstruction produced five strata with 16 identities per stratum. All 500 coefficient rows were validated.

Contract hashes: final configuration `f94fae55db2d5ed6c883e5127b5fe1feb0c2fea4ac4b8565e2eec35b5f275be9`; reviewed deviation `02fb8a72acd3693f1d72b5a95b9c8dfafd52447e2d30e05b0054059eab1d4090`; coefficient schedule canonical `d6bd01692d114c385143635a21a8db0475128d41c91f329f641c7611d3f83bd0`, file `e8e95e885984e9bb41d9040db69c0ef246cdd77a54b01b69db1133e543e18983`; training panel identity `0f7abfbd4f2656144f0a673eaaec06d39f6b028c0d393df76a2c1e8ceb6b2038`; evaluation panel identity `5c4f83f84d9bdf60d86d26a7def65757df7a7c460dd1ed1a8c0c2d24ad832250`; matched-arm identity `991df750afbd1b0edb8a7679d4ccda99da65224f6264ac973b856c33f1cb6b77`. The matched components are sample schedule `3f4a1ab2e015a8b2c209bb4b89d82c93db29d950c2fda7d75b2abc076ae3a2e3`, timestep/noise/corruption seed schedule `8369bc0401197b794cca5c9de48b2fcae33f9f553f6a1bed7c5b1a64286aa99e`, and evaluation identity as recorded above. Training panel counts by stratum were 808, 408, 208, 208, and 208; evaluation counts were 16 per stratum. The report records all 500 coefficient lookup timesteps and gradient-audit boundaries `0–10, 25, 50, 100, 250, 500`.

Relocated inventory entries (recorded identity → expected SHA-256):

| Identity | SHA-256 |
| --- | --- |
| `/home/simostocco/proteinGen/data/full/processed/samples/7ssn_D.npz` | `99b26c65e29601430dea6810feff24f7b4a6a8c8ed255de6621448e8e2fdafd2` |
| `/home/simostocco/proteinGen/data/full/processed/samples/8imk_G.npz` | `9a1c55c13de0d92c00e67cf1475dc0e51074adb4b72eea0649ce8be31e1df747` |
| `/home/simostocco/proteinGen/data/full/processed/samples/9h4n_E2.npz` | `30cda6a6321490549d7f1ce46bfef1d32859cd80394b64e1d910d311fff75afe` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/4u3m_S1.npz` | `995e7b2e2d8cea8ad58214b298de5ce37d87f3cc4b898a07aeb3a57a7014a239` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/7l20_q.npz` | `81121b9b15b50811ede590a7902b670ef57176f6c4f56d35c857dea483b52f3a` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/8agz_d.npz` | `85c2a83b46f6aef7a9f0e7e4e8d24524cd5f7ae9be8ec630cf57f1c116faeb5b` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/8fvy_a.npz` | `7c51c6378412f2b7cc36076d2a740a8dc1fcfc8cd0a4af961173cef5dee72c45` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/8r55_c.npz` | `680dd9a44be5e5c15775092243f68d24da324897866e0bc67786e60268139751` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/8wql_X1.npz` | `c39491cf3d3dfdec55e743fdcae5e796912e8013b7d53b24b663bc3f0f063e32` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/9cem_A.npz` | `33b93cfc7b959d556527f46482951f2ed11763baee39b5dbfab219be57d7c1c6` |
| `/home/simostocco/proteinGen/data/full/processed_recovery/samples/9jq2_c.npz` | `06ff4e2ad6243e1a5289e63ef193a5dc6de18da5cece5bccfe73e57fb2ccd1a3` |

Verification: E007 suite `548 passed, 3 skipped`; repository suite `1,351 passed, 13 skipped` (6 multiprocessing deprecation warnings); Ruff check passed; Ruff format check passed (289 files); `git -c core.fileMode=false diff --check` passed. CLI help, plan-only, coefficient-table validation, and the real production `--validate-pilot-contract` invocation passed. The final output directory and staging directory do not exist.

Read-only execution flags: `staging_created=false`, `checkpoint_loaded=false`, `model_created=false`, `optimizer_created=false`, `cuda_initialized=false`, `forward_pass=false`, `backward_pass=false`, `sampling_performed=false`, and `optimizer_updates=0`. All downstream authorization fields remain false. The accepted minor preflight deviation authorizes preparation of this bounded exploratory pilot only; it does not authorize Phase 3J, production or joint training, sequence conditioning, downstream generation, or checkpoint selection.

The validator hashes and authorizes the full protected inventory before reconstructing train/evaluation panels. The same configuration-to-Phase-3F helper supplies the validated mapping to validation and future execution. Path validation rejects empty policies, malformed or relative roots, parent traversal, duplicate recorded roots, root collisions, relocated escapes, and hash mismatches. Resolution uses exact recorded paths and the original protected SHA-256; it has no case-folded or search fallback.

## Preserved evidence

The coefficient schedule v2, dense-preflight v2 report/protocol/cells, reviewed deviation record, step-9000 checkpoint, all earlier protected evidence, and completed outputs are protected by their existing pins. No pilot, training, sampling, optimization, or CUDA workload was run for this correction.

## Future command (not run)

```bash
python scripts/run_e007_local_backbone_repair.py --config configs/e007_local_backbone_repair_pilot_phase3i2_final_v1.yaml --pilot
```

## E007 Phase 3I.2 dynamic stability follow-up

### Failure diagnosis and preserved evidence

The failed arm was `v_only`. Its update-0 drift audit passed. At update 1, the counterfactual local-objective audit failed for `3qoc_C`, length 128, timestep 499: combined auxiliary/v ratio `0.3212773740376199`, total/v ratio `0.7803045758626649`, `i_plus_2`/v `0.16581694607051847`, and discontinuity/v `0.051242970883402135`. Every gradient element was finite. The actual v-only update was finite with v and total loss `0.26944366097450256`; local auxiliary losses were not used by that update. Before/after state hashes verified restoration of model, optimizer, scheduler, RNG, CUDA RNG, CPU RNG, and data cursor. This identifies high-noise coefficient schedule instability under counterfactual local losses, not model divergence or audit-state corruption.

The immutable incident record is [local_backbone_repair_dynamic_stability_incident_v1.json](reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dynamic_stability_incident_v1.json), SHA-256 `95fe2724b73f011bf40dfa0ff214dd8c62573b9052f690299098570d09d5ec48`. Failed staging remains in place and was not resumed, rerun, renamed, or modified. Preserved evidence hashes: log `3d4743d4322890e5de762b09fde42d8f983066460a4d0c3e0aa3552dec07dc97`; heartbeat `916c9d5e81470c42efec8efd45cc7aefbd2d93484ece2dc7ff64d5e6017cf7d5`; drift audits `3147a6ad6cea71ef247a041795bf215062699baaf7a59ade54c0ac359dcb5248`; evaluations `9f23328f1c22ed46ff95fa976928e2d37037a223845c90c87e8575024f6e7579`; metrics `62fe959492d0f0e7842be4ae44e698eb198e84a997e08ebb3ae30ff088c7c1f8`; update-1 recovery checkpoint `14912aba478de707cc7688d461ca7a8ffbf8ce46bfcc1cc97f2e7bde43d3d9cd`; panel manifest `42d9d1c5dd787a555c63475772936aa2e6f8646f546dd80181e4b19900caf656`.

### Coefficient schedule v3

`configs/e007_local_backbone_repair_budget_medium_coefficients_v3.json` is mechanically derived from v2. Parent canonical/file hashes: `d6bd01692d114c385143635a21a8db0475128d41c91f329f641c7611d3f83bd0` / `e8e95e885984e9bb41d9040db69c0ef246cdd77a54b01b69db1133e543e18983`. The multiplier policy canonical hash is `2a8810c622ca01c1ed14c744851ae93680d72fa4e3a59bc91c49a192972b14c6`. Anchors are 0→1.0, 450→1.0, 475→0.8, 487→0.6, and 499→0.375, with positive log-linear interpolation. Every row 0–499 is explicit and ordered; the same multiplier is applied to all six local terms, preserving their within-timestep relative weights. Values through 450 are byte-equivalent as parsed coefficient values to v2; all coefficients are finite and positive. Resulting canonical/file hashes: `1eb9e634809d5bb8c13e5912e6495db138a0c0629d484f3bdbefb6e08d75e6bb` / `1bcbd0fe6c4253799daf4c0a542c81aa640eac8ff88810bebbd957a826d606d8`.

### Dynamic-stability preflight contract

Configuration: [e007_local_backbone_repair_dynamic_stability_preflight_v1.yaml](configs/e007_local_backbone_repair_dynamic_stability_preflight_v1.yaml). It starts from immutable step 9000, uses `v_only` for exactly 10 updates from the first 10 final-pilot identities/noise/timesteps, and audits boundaries 0–10. Each boundary audits lengths 64, 128, 256, 384, and 500 at timesteps 25, 250, 425, 450, 475, 487, and 499, explicitly including `3qoc_C/128/499`. Audits use deterministic identities and corruption noise and publish per-cell ratios, individual terms, finite counts, losses, coefficients, identities, and before/after state hashes. Each must leave all parameter, optimizer, scheduler, RNG, CUDA RNG, CPU RNG, and cursor hashes unchanged.

All finite-element, individual-term, combined auxiliary/v (`<=0.20`), total/v (`[0.80,1.30]`), protected-hash, restoration, deterministic-replay, RSS/CUDA-memory, exactly-10-update, and no-sampling gates fail closed. Existing hard caps are unchanged. Publication uses versioned final/staging paths, atomic artifacts, exact-state recovery at every update boundary, and false downstream authorization fields. A pass can authorize preparation of a new pilot configuration only; it does not authorize pilot execution. Read-only CLI modes cover plan-only, schedule validation, contract validation, monitoring, and resume inspection. Configuration, protected hashes, relocation mappings, panels, schedule, audit identities, and run contract must validate before staging, model construction, checkpoint CUDA loading, or CUDA initialization.

The planned workload is 395 model forwards (10 training and 385 audit forwards) and 2,705 backward/autograd operations (10 training backward calls and seven gradient-vector calculations for each of 385 audit cells). The runtime estimate is 7,200 seconds, bounded by configured RSS/CUDA memory limits.

No dynamic preflight, pilot, sampling, or CUDA workload was run while preparing this follow-up.

### Future execution command (not run)

```bash
python scripts/run_e007_local_backbone_repair.py --config configs/e007_local_backbone_repair_dynamic_stability_preflight_v1.yaml --dynamic-stability-preflight
```

Resume after interruption with the same configuration and `--resume-dynamic-preflight`.

## E007 Phase 3I.2 audit lifecycle repair (2026-09-26)

The failed dynamic preflight v3 is preserved. Its log is `logs/e007_local_backbone_repair_dynamic_stability_preflight_v3.log` (SHA-256 `65ae85b84cb18904fe954c238568add1c185a5e2b0e0eb678ba176efec85d572`). Its sole staging file is `.local_backbone_repair_dynamic_stability_preflight_v3.inprogress/heartbeat.json` (SHA-256 `4c5f0fd04d64bfead102c55729e2ef477bb0ded50d28f0d293bbd0643c72b7e0`). Neither was deleted, modified, or resumed.

The zero-update drift audit now constructs every cell's tensors inside `_zero_update_drift_cell`, returns scalar records, releases tensors in `finally`, and runs garbage collection and CUDA cleanup after each cell, including exceptions. The final local-term gradient releases the graph. Production-loop regression tests cover 12 consecutive cells, lengths 64 → 128 → 500, different timesteps, a repeated sample, complete serializable records, unchanged state hashes, and cleanup after an injected exception.

A bounded real CUDA lifecycle smoke passed on lengths 64, 128, and 500 at timesteps 25, 250, and 499. It used the immutable step-9000 checkpoint, made zero optimizer updates, performed no sampling, and stayed under the original 4096 MiB RSS, 6144 MiB CUDA allocated, and 7680 MiB CUDA reserved caps. Post-cell CUDA allocation was 92.986 MiB and reservation was 142 MiB for each cell. Peak allocated/reserved memory was 5813.744/6284 MiB. Its distinct non-authorizing output is `local_backbone_repair_dynamic_memory_smoke_v3` (report SHA-256 `ad931091ced5a63b6576c8a62fbca14cf61e57ec1035bde7b20c2d6d0fd8c4a2`).

Only after that smoke passed, a fresh `configs/e007_local_backbone_repair_dynamic_stability_preflight_v4.yaml` was created with new v4 final and staging paths. It pins the failed v3 evidence and the passed lifecycle smoke. Plan-only, coefficient-table validation, and the read-only dynamic contract validation passed. The ten-update preflight and pilot have not been run.

## E007 Phase 3I.2 dynamic-stability preflight v5 preparation (2026-09-26)

The v4 run failed closed at audit boundary 5. Its staging report, heartbeat, and execution log remain unchanged. The reviewed, non-authorizing [warning record](reports/experiments/E007_matrix_sequence_cogeneration/local_backbone_repair_dynamic_stability_v4_warning_review_v1.json) pins all three files. The v4 evidence contains one warning identity: update 5, `2e19_A`, length 64, timestep 425. Its combined auxiliary/v ratio is `0.23559981839253288` and total/v ratio is `0.9288470670534813`; all individual-term gates passed, every gradient was finite, and the audit restored state. The classification is `accepted_dynamic_warning_below_hard_ceiling`. The reviewed warning authorizes neither pilot execution nor downstream work.

The fresh [v5 configuration](configs/e007_local_backbone_repair_dynamic_stability_preflight_v5.yaml) pins that record, uses new v5 final and staging paths, and keeps coefficient schedule v3 byte-identical. V5 records a warning when combined auxiliary/v exceeds `0.20` and fails closed only above `0.25`. Individual-term caps remain `0.20`, total/v remains within `[0.80, 1.30]`, and all finiteness, state-restoration, memory, and protected-hash gates remain in force. Each boundary publishes warning counts and identities, and the final report aggregates them. A passing v5 can authorize preparation of a pilot only; pilot execution remains unauthorized.

The planned run starts afresh from immutable step 9000 with a new optimizer and scheduler, executes exactly ten v-only updates, and audits every boundary 0–10. It must not resume v4. This preparation does not execute v5 or the pilot.
