# E016 V3 CPU foundations

New, isolated scientific experiment. This package does not launch protein training or AlphaFold. The authoritative implementation plan was supplied by the user; reports/configuration and exact source provenance are in the parent D: artifact root.

- G1: typed build/edit contracts, differentiable float64 C-alpha geometry, hash-bound existing14-feature chemistry adapter.
- G2: dense bidirectional-prefix geometric RoPE actor, sequence-only baseline, independent candidate-Q twins/targets, hybrid continuous/discrete SAC math, synthetic delayed mock queue/replay, edit interfaces and toy search accounting.
- All data/cache/temp/test output paths must stay on D:. Use parent run_cpu.py; no dependency installation.
- Native future labels never enter build state. Next action is separate supervision. Edit state only after matching complete evaluation. NO_PLAN is the only active planner.
- Dataset certification, production training, real-oracle execution, CUDA implementation/profiling and quantitative chemical priors are blocked pending separate gates.

See parent FINAL_REPORT.md and GO_NO_GO.md for measured scope, limitations and next approvals.50 new CPU tests pass; no thermodynamic/design-success claim.

## Readiness continuation from 7acd364

The new readiness worktree adds explicit CPU/CUDA devices, FP32/FP64 coordinate guards, a checkpoint-compatible streamed exact SAC backward schedule and authoritative generated-proposal snapshots for portable replay. The inherited CPU foundations remain the reference; 60 CPU tests now pass and 8 CUDA cases are deliberately skipped. No GPU certification, training or AlphaFold inference has executed.

The prepared benchmark is dry-run-first, binds a future approval card to the source commit/configuration and measures actor, twin-Q, SAC gradients and replay without optimizer steps. See the parent BLOCKER_RESOLUTION_REPORT.md and READINESS_MATRIX.md. New PDB TRAIN membership remains uncertified; the original14 chemistry allowlist and geometry version are unchanged.
