# E015 host shutdown recovery, 2026-10-11

The recovery audit found a clean worktree on branch
`e015_integrated_multiscale_parity_selfconditioned_v1` at
`413aae7aaa64d1991bd3c3505e121a102030215b`. No reset, source change, or
uncommitted-work removal was performed. Later committed implementation and
durable artifacts supersede the historical `3d143e76` launch blockers.

Production had launched. The selected durable boundary is
`production/checkpoints/recovery_1.pt`, SHA256
`33fc6b0a02a96e09589ccb46e70d1fcea32942ca9755d8af0ef3afe4c2ae4ee4`:
epoch 1, update 2,691, cursor/exposures 80,801. `LATEST.json` and both committed
records agree. The last pre-shutdown heartbeat recorded update 2,725 and 81,736
exposures, but those later updates have no durable checkpoint and are recomputed.
No newer partial checkpoint was found. Initialization, quarter-epoch checkpoint,
both recovery slots, replay checkpoints, and both completed smoke checkpoints
were preserved.

Verification passed for checkpoint SHA256, unchanged config/source/calibration/
batching/population/architecture contracts, strict model and EMA loads,
model/EMA/AdamW/scaler fingerprints, and exact restoration of global and dedicated
CUDA RNG state. The regenerated epoch schedule and microbatch boundaries match;
all per-identity exposure counts agree with the saved cursor and total. Parameter
count remains 33,128,139.

BF16 remains selected with FP32 geometry islands. Source-bound raw/EMA N500
certification passes: maximum proper-rotation relative error 0.0023555527
(0.23556%), translation 0.0015735578 (0.15736%), unchanged rotation bound 3%.
Enabled reflection gap is nonzero; disabled control error is zero. Typed O(3)
trunk and stable kNN cases passed. Serialized scientific CUDA replay is bitwise
exact, with descriptor envelope 0 under the unchanged 5e-6 tolerance. The original
5.835% failure was localized to fixture rotations constructed inside autocast.

The current-source 100-update smoke passed at 2,938 exposures and its new-process
update-101 replay was bitwise exact. The continuous 1,000-update smoke passed at
30,390 exposures. Both durable smoke checkpoint positions/contracts were loaded
and verified. Renewed version-4 full-path autotuning already completed after the
geometry changes; its source-bound result is valid for exact recovery and was
preserved rather than changing checkpoint batching.

| Length | Physical batch | Allocated GiB | Reserved GiB | Total used GiB | Free MiB |
|---|---:|---:|---:|---:|---:|
| 20–64 | 68 | 5.370 | 5.615 | 7.040 | 941.5 |
| 65–128 | 30 | 5.422 | 5.592 | 7.009 | 973.4 |
| 129–256 | 14 | 5.392 | 5.580 | 6.948 | 1035.3 |
| 257–384 | 9 | 5.299 | 5.531 | 6.909 | 1075.9 |
| 385–500 | 7 | 5.419 | 5.598 | 6.959 | 1024.9 |

Each selected batch passed three repeated complete-path probes. Target residue
budget remains 4,096. Free-space/reserve accounting remains telemetry only;
actual write failures still stop E015 safely.

Host WSL and native Windows process inspection found no duplicate E015 trainer.
Ubuntu was the only running WSL distribution; GPU compute process list was empty
before recovery. Windows' sole Python process was an E012 read-only shutdown
audit. E007 was explicitly paused with automatic resume disabled, at update
3,580 / 4,549 exposures. Its `sequence_pause_3580.pt` SHA256 was verified against
its committed record. E015 was selected for the GPU. No E007/E012/E014 file or
process was modified or signalled by this recovery.

Exact detached supervisor command, run from this worktree:

```bash
/home/simostocco/miniforge3/envs/proteingen/bin/python -B /mnt/d/Simone/proteinGen/.e015_worktree/scripts/supervise_e015_training.py --resume
```

Recovery supervisor PID: 28874. Its production child uses
`scripts/run_e015_integrated.py --resume`. Recovery audit and original status
snapshots are under the artifact root's `shutdown_recovery_20261011/`; initial
supervisor output is in `logs/shutdown_recovery_20261011.log`.

Historical blocker and report records are preserved in that recovery directory.
The current production heartbeat and committed checkpoint remain the authority
for subsequent progress. This is checkpoint recovery, not a new update-zero run.
