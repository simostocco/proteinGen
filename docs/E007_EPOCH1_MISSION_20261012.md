# E007 exact epoch-one hold and serialized comparison

Recovery selected COMMITTED/LATEST recovery_0 at update90,413 /115,067 exposures, SHA77512116a499935c8f424ad85de5ea9381874ecee4ac3d490ded02c2c20a1ce3. The CUDA unknown-error exit left a stale120,387-exposure live state;5,320 exposures must be recomputed. CPU strict model/optimizer/RNG, deterministic schedule/exposure vector and all frozen source hashes passed. First recomputed update90,414 was7n2o_F, length241, seed7492264717301116066, finite loss0.5143104791641235.

The original E007 supervisor is untouched. `supervise_e007_epoch1_hold.py` copies its frozen workflow with an operational hold after normal epoch-one diagnostics, requiring230,132 committed exposures. Its restart guard runs under the supervisor owner lock and prevents epoch2. Six focused tests exercise exact boundary, restart, crossed boundary, incomplete training, missing diagnostics and wrong final accounting. The original two-epoch460,264-exposure contract remains unchanged.

Training retains certified nice+10, all scientific thread/allocator/precision/batch/optimizer/RNG settings, with ionice best-effort priority7. No CPU affinity or numerical-environment change applies to production. The detached monitor checks Windows/WSL RAM, swap rate, CPU load, GPU owner/memory/temperature, D:reserve and trainer I/O. Sustained pressure requests the existing graceful handler.

After the hold, the detached controller verifies the named epoch-one scientific checkpoint and the latest committed resume checkpoint, including strict CPU restore and the exact epoch-two operation. It waits for complete GPU release, then resumes the existing E01582-job screen by fingerprints, completes alignment and conditional confirmation/novelty, and evaluates E007 using the frozen100+100 panel and20 exact historical Gaussian starts. E015's exact epoch-one reconstruction is reused; a bounded matched20-sample generation and, when controls/resources permit, matched external assessment avoid conflating230,132 and314,256 exposures.

Every CUDA stage is serialized. Existing real controls and tool/model snapshots are reused by hash. No E015 production or E007 epoch2 is automatically resumed. Each stage stops on errors with on-disk recovery evidence. Public-service failures remain pending; failed controls make designability inconclusive. Generated coordinates alone never prove experimental stability, solubility or function.

Authoritative output root: `/mnt/d/Simone/proteinGen/e007_e015_epoch1_comparison_v1/`. See `progress.json`, `METHODS.json`, `RECOVERY_AUDIT.json`, `FIRST_RESUMED_OPERATION.json`, `POST_RECOVERY_ROLLING_CHECKPOINT.json`, `E007_RESOURCE_OBSERVATION.json`, `LAUNCH.json` and `AFTER_EPOCH_CONTROLLER.json`.

Recovery commands, executed from the E015 worktree:

```bash
/home/simostocco/miniforge3/envs/proteingen/bin/python -B docs/e007_epoch1_mission.py resume
/home/simostocco/miniforge3/envs/proteingen/bin/python -B docs/e007_epoch1_mission.py queue-after-epoch
```

Never invoke resume while a GPU stage owns the machine. The command audits current durable metadata, not a historical checkpoint. If the exact epoch-one hold already exists, it does not train further. A boundary crossed before recovery is reported explicitly and fails closed. Completed inference results are reused only when settings/input/output hashes match.

The initial15-minute recovery and hold tests passed. Scientific comparison scripts are staged for serialized execution; their numerical results and interpretations remain pending until actual execution. Existing untracked `e015_absolute_evaluation.py` is preserved.
