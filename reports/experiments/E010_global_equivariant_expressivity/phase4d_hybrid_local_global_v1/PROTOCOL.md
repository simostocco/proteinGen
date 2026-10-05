# Phase 4D v1 preparation record

This versioned record describes a new frozen-global/oriented-local architecture.
Training authorization: **false**. Training launched: **NO**. CUDA used: **NO**.

The authoritative scientific specification is
[the Phase 4D document](../../../../docs/e010_phase4d_hybrid_local_global.md), with
[the candidate config](../../../../configs/e010_phase4d_hybrid_local_global_v1.yaml).

- `tiny_panel.json`: deterministic 20 training identities, exact 60 cached examples,
  condition/seed/shard/tensor hashes and training-manifest provenance; no coordinates.
- `preparation.json`: verified Phase 4B source and tensor-state provenance, counts,
  exact zero-init parity, CPU initialization gradient audit and baseline telemetry.
- `validation.json`: CPU test outcomes and untouched-base failure comparison.

Candidate coefficients, displacement thresholds and tiny-overfit gates require
review and freezing before the separate 500-update experiment. Historical Phase
4B/4C records remain unchanged. No checkpoints or datasets are part of this record.
