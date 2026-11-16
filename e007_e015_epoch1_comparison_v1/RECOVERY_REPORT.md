# E007 recovery and exact epoch-one hold

Recovered from update90,413 /115,067 committed exposures after a CUDA unknown-error exit. Stale live work had reached120,387 exposures;5,320 exposures require recomputation. CPU strict model/optimizer/RNG and exact schedule/exposure vector passed; all frozen source hashes match.

The certified supervisor remains byte-for-byte unchanged. A separate wrapper stops after the normal epoch-one diagnostics, requiring230,132 committed exposures and the diagnostics completion marker. Focused execution tests cover exact boundary, restart, and an incorrect exposure count. The original two-epoch contract is preserved; epoch2 and E015 production will not auto-resume.

Scheduling only: certified nice+10 remains; ionice best-effort class2 priority7 lowers I/O contention. No thread, affinity, batch, precision, allocator, objective or deterministic noise changes. The prior CUDA error alone does not establish whether host freezes arose from the driver, CPU contention, host memory, swap, or NTFS I/O. Windows/WSL memory, swap, GPU and D:reserve are monitored; pressure invokes the existing graceful checkpoint/stop handler.

Progress and remaining-time telemetry:progress.json and resource_monitor.jsonl. Durable recovery:RECOVERY_AUDIT.json. First operation:FIRST_RESUMED_OPERATION.json. Exact commands:LAUNCH.json and AFTER_EPOCH_CONTROLLER.json. Every subsequent heavy stage waits for ownership release. Previously completed E015 predictions are hash-checked and reused. Scientific conclusions remain pending.

Epoch1 named checkpoint and exact epoch2 next operation verified in E007_EPOCH1_CHECKPOINT.json. Neither training job was restarted after evaluation.
