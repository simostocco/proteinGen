# proteinGen storage audit — 2026-10-11

D: is 99% full. E015 is actively training and can reach its next durable checkpoint with current space. E012 is **already failed**, not actively growing: its native diversity owner exited on DATA2-E (disk reserve) before this audit. Do not restart the failed pilot namespace unchanged. No data was deleted, moved, compacted or truncated.

## Filesystem and totals

- Capacity: 1,000,202,039,296 bytes (931.51 GiB).
- Initial used/free: 989,465,092,096 / 10,736,947,200 bytes.
- Final used/free: 989,473,746,944 / 10,728,292,352 bytes (9.99 GiB free). Audit output and normal writers explain the small decrease; reclaimed bytes: **0**.
- proteinGen apparent path total: 484,698,075,318 bytes (451.41 GiB); GNU du unique apparent total: 484,699,603,203 bytes.
- proteinGen_data apparent path total: 188,838,656,382 bytes (175.87 GiB); GNU du unique apparent total: 167,743,468,013 bytes. Known hardlinks explain 21,095,193,808 bytes of double-counted path size.
- The two project roots do not account for all D: usage. Approximately 337 GB of drive usage is outside their unique apparent totals or filesystem allocation overhead; this audit does not classify unrelated contents.
- DrvFS df -i is synthetic and internally inconsistent; it does not indicate real NTFS inode exhaustion.

## Largest consumers and classification

| Consumer | Apparent size | Classification |
|---|---:|---|
| outputs/ | 245.33 GiB | KEEP FOR NOW / UNKNOWN: legacy scientific checkpoint schedules and experiment outputs |
| data/ | 154.27 GiB | PROTECTED shared source/processed datasets |
| E012 sequence_foundation | 175.61 GiB path-counted | PROTECTED except separately reviewed corrupt pilot exports |
| E012 failed identity_50 pilot_tmp | 48.39 GiB | KEEP FOR NOW resumability/forensics; two malformed exports UNKNOWN |
| reports/ | 38.71 GiB | PROTECTED conservative scientific evidence |
| E015 entire run | 5.48 GiB | PROTECTED |

Legacy outputs contain 44.75 GiB in E001 checkpoints and about 38.64–38.78 GiB each in E002/E003/E004 checkpoints. These dominate long-term project storage but are not an automatic reclaim authorization. Shared data/full contains 223,709 raw mmCIF files and 333,973 processed-recovery samples; the census includes hidden worktrees. largest_directories.tsv holds the top 50; largest_files.tsv holds the top 100. storage_categories.json supplies >=1 GiB, >=250 MiB, small-file counts, checkpoint, cache, SQLite/WAL/SHM, log, archive and evaluation/trajectory inventories.

Conservatively identified protected unique apparent storage lower bound: **332,749,832,614 bytes (309.90 GiB)**. Additional unique evaluations/latest checkpoints in KEEP/UNKNOWN roots remain protected; this is not an exhaustive inode-level accounting claim. Unequivocal SAFE RECLAIM: **0 bytes**. Actually reclaimed: **0 bytes**.

## Active writers and measured growth

Linux supervisor PID 28874 owns E015 trainer PID 30192. The trainer has D: cache/temp environment settings, and both map the shared D: coordinates.f32. No Linux MMseqs/E012/E007 worker was found. Windows CIM found no Python/MMseqs/linclust worker on repeated queries. E007 state is paused, PID null. Linux lsof +L1 returned no deleted-open entries; native Windows-wide deleted-open handles were not exhaustively enumerable, so their absence is not proven.

Measured window: 350.4 seconds (5.84 minutes), using targeted regular-file sizes while E015 continued normally.

| Directory | Bytes gained | MiB/min | GiB/hour | GiB/day |
|---|---:|---:|---:|---:|
| e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/production | 146,293 | 0.02389 | 0.001400 | 0.033595 |
| e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/production/checkpoints | 0 | 0.00000 | 0.000000 | 0.000000 |
| e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/logs | 0 | 0.00000 | 0.000000 | 0.000000 |
| e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/production/logs | 146,571 | 0.02394 | 0.001402 | 0.033659 |
| e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/production/evaluation | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/clusters | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/mmseqs | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/tmp | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/production | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/production/reports | 0 | 0.00000 | 0.000000 | 0.000000 |
| E012 corrected/production/logs | 0 | 0.00000 | 0.000000 | 0.000000 |

Nested rates overlap: never sum production + its checkpoint/log/evaluation children. E015 production is the fastest observed project writer, at 0.02389 MiB/min net retained growth; E012 rate is **0**. No checkpoint burst occurred inside this measurement window. A later recovery save was observed successfully at update 3103, exposures 93047. Gross recovery serialization is about 1.98 GiB/hour when every 15-minute save occurs, mostly overwriting two retained slots rather than adding space. /proc write_bytes is zero on DrvFS despite real writes, so it is not a reliable byte-rate meter here.

Free / observed net E015 rate gives approximately 297.4 days of **log-only observational runway**. This is not a usable full-training ETA: retained quarter-epoch checkpoints produce ~0.5 GiB jumps, atomic saves transiently require another ~0.5 GiB, and unrelated D: writers are not attributed. E012's prior exports are bursty runaway writes, not current growth. No meaningful active E012 storage ETA exists while it is stopped.

## Checkpoint recoverability and continuation

Every checkpoint-like artifact found in the E015 artifact root/worktree and E007 control root is listed in checkpoint_inventory.json, with position metadata decoded without loading tensor payloads or running checkpoint globals. Recorded SHA256 values are included where available. Current COMMITTED/LATEST references take precedence over snapshot ages; no checkpoint was removed. E015 current heartbeat: update 3195, exposures 96014; durable COMMITTED position: {'epoch': 1, 'cursor': 93047, 'update': 3103, 'exposures': 93047}. E015 checkpoint storage: 5,850,715,858 bytes (5.45 GiB), including smoke and CUDA replay evidence; production inventory currently totals 1,863,227,861 bytes.

Actual source policy: rolling recovery every 900 seconds, alternates recovery_0.pt/recovery_1.pt, atomic .pt.partial on D:, full reread/model/optimizer/RNG verification, SHA256 then os.replace and LATEST publication. Immutable saves occur at 25/50/75/100% of each epoch, initialization, final, and exceptional stops. Slots are overwritten by policy; scientific boundaries are not pruned. Full payload contains model, FP32 EMA, optimizer, scaler, global/engine RNG, schedule, exposure bitmap, exact position, evaluation set, summaries and contracts. No evidence of recovery pruning failure, unintended trajectory/tensor dumping, or growing atomic .partial copies was established. Per-protein telemetry is aggregated in bounded summaries; production logs are currently small.

E015 current full checkpoint is 532,177,642 bytes. Remaining checkpoints through two epochs: approximately 4,257,421,136 bytes net (3.97 GiB); through maximum four epochs: approximately 8,514,842,272 bytes (7.93 GiB). Add one 0.50 GiB transient atomic save and evaluation/log/cache growth, which is not a hard upper bound. Therefore the next save is comfortably feasible; the full maximum schedule fits on checkpoint sizes alone but leaves only about 1.57 GiB before other writes. Long-term simultaneous continuation is not certified safe.

E007 sequence_pause_3580.pt is 92,321,475 bytes. Its SHA256 was freshly computed and matches **47d58d914400c08f2d90af8d19a7f2751f7575efd4e63a8f137818fb9357693c**. LATEST/COMMITTED and shutdown/queue recovery metadata are preserved. Source has the same 15-minute two-slot recovery and quarter-epoch immutable boundary policy, up to four epochs. With all 16 boundaries remaining, checkpoint-only future net storage is at most approximately 1,477,143,600 bytes (1.38 GiB), plus one ~92 MB transient save. Evaluation/control outputs and runtime duration cannot be bounded from current measurements; reassess D: before later resume.

## E012/MMseqs cause and required recovery

Current production STATE says phase=failed, stage_at_failure=diversity, error=runtime disk reserve breached. Its named native PID 30368 is absent. Certification/final markers remain on D:. The corrected corpus contains 512 shards and 512 shard certificates, plus canonical final corpus, per-row manifests and hashes. Historical reservoir scientific/filtered/reservoir.sqlite and all other reservoir copies/WAL/SHM remain untouched, including 4,649 historical evidence files.

The failed 10,000-sequence input is only 2,853,460 bytes, but rep_seq.fasta is 38,785,217,720 bytes and all_seqs.fasta is 13,167,845,376 bytes: **51,953,063,096 bytes (48.39 GiB), approximately 18207x input-byte amplification**. rep_seq export took ~18m35s, about 1.94 GiB/min averaged across that substep; that is historical log-based telemetry, not current write rate. Its FASTA contains NULs and out-of-record content. MMseqs ran createdb --createdb-mode 1, zero-copy symlinks to a Windows CRLF subset. Bounded 20-sequence tests reproduce corruption for CRLF/mode1, while LF/mode1 and copied mode0 outputs are clean and agree byte-for-byte. This proves the serialization bug without hashing/rereading the 20M corpus.

The source fix forces LF for the existing fixed-rank diagnostic subset writer. It keeps identity/coverage, ordering/rank, thread count, sample-size rule, scientific selection, retention, historical reservoir and all protocol files unchanged. Installed only while E012/MMseqs was inactive. **It does not repair the existing failed pilot**: easycluster.sh uses file-existence tests for rep_seq/all_seqs, so a naive rerun could publish those malformed/partial files and reuse an invalid pilot resource measurement. Preserve the remaining DB/index/cluster TSV, failed log, input and state until a reviewed replay plan establishes which substeps must be invalidated.

No identity_30 work exists yet. Corrected mmseqs/ and tmp/ roots are empty; historical MMseqs diagnostics remain F/protected. No SAFE A/B/F deletion exists. e012_mmseqs_inventory.json classifies these directories.

E012's existing reserve floor is **56,883,428,966 bytes (52.98 GiB)**, from 30% of starting_free_bytes. It is E012 runtime behavior, not an E015 scientific launch gate. Minimum reclaim merely to clear that floor now: **46,155,136,614 bytes (42.99 GiB)**, plus pilot scratch and concurrent E015 headroom. A conservative full 20 GB allowed diversity scratch peak plus a next E015 checkpoint requires about **66,687,314,256 bytes** reclaim. The existing predeclared sample-scaling rule may select a smaller diagnostic after a clean LF pilot; actual future scratch/RAM/time cannot be known until that pilot is measured. Two identities run sequentially; scratch is cleaned only after statistics/certification. Final diversity/handoff report sizes are small JSON/Markdown by source structure but not guaranteed exact byte counts.

Reclaiming the two corrupt exports, if approved, would give about 58.38 GiB free, enough to clear the reserve with a constrained pilot/diagnostic budget; it does not guarantee the full 20 GB maximum scratch plus the whole E015 schedule. E012 currently cannot complete diversity/handoff under its unchanged guard and invalid existing pilot state. Do not lower its guard, delete a reservoir, or silently select a different scientific policy.

## Duplicates and cleanup decision

E015 smoke and production initialization.pt are byte-identical (SHA256 66085f396f6b90b5cb994bf09a324d53f3e25979196f90a8de2546472fa803fc), but serve separate preflight/production evidence roles: KEEP FOR NOW. Actual/expected shutdown fingerprints are also identical but are replay evidence: keep. Audit/preservation reservoir.sqlite and WAL pairs are the same physical files through hardlinks; deleting one path reclaims **zero**. Original reservoir versus copied audited reservoir only matches sampled blocks; full equality is unproven and all are protected. Remaining size/sampled-block checkpoint similarities are documented without treating them as proof of byte equality. No corpus-wide hash scan was performed.

No deletion qualified as unequivocally recomputable, unreferenced, unopened and nonhistorical evidence. proposed_deletions.json records only the two UNKNOWN/high-value malformed exports requiring explicit review; its automatic subset is empty. executed_deletions.json records zero removals. No broad cleanup command was run.

## Source changes and next action

Fix commit: **8bcc9dff0d547535b72b3cf9c3cd13326ed87ad8**, stored in the D: sparse source checkout `e012_fix/` based on recovery commit af9ada224543c6240c7919b88a4fe93a04a53e12. It changes scripts/prepare_e012_sequence_corpus_v2b.py (LF output only) and adds tests/test_e012_diversity_newlines.py. Two tests pass: simulated Windows serialization preserves exactly 10,000 predeclared ranks; real MMseqs LF/copied-CRLF outputs match. No E015/E007/E014 source or scientific protocol/retention policy changed. The existing E012 source file at /home/simostocco/proteinGen-causal-rope/scripts/prepare_e012_sequence_corpus_v2b.py received exactly the tested change, under filesystem approval; its source repository was not committed or cleaned. All new commit objects, tests, data, reports and scratch are on D:. Before/after source hashes and deployment checks are in source_fix.json. The sparse D: checkout borrows existing base git objects read-only; its new commit lives on D:.

Recommended immediate action: let E015 continue and monitor actual free bytes at recovery/boundary saves. Approve a specific recovery/reclaim plan for the two malformed E012 exports after retaining bounded forensic evidence and establishing fresh pilot invalidation/replay. Then measure a clean LF pilot under the unchanged resource rule before resuming diversity/handoff. For durable reserve beyond those stages, review legacy E001–E004 checkpoint retention separately; do not assume their 160+ GiB is disposable.

**Explicit preservation confirmation:** no protected scientific artifact was removed or modified. No dataset, checkpoint, log, report, cache, temporary file, SQLite/WAL/SHM, MMseqs state, trajectory, evaluation or model artifact was redirected off D:. E015 trainer/supervisor were neither killed, paused nor modified; E007 checkpoint and all E012 reservoirs/evidence/corpus remain intact.
