# Bounded recurrent capacity v3

The scientific contract is [the v3 document](../../../../../docs/e010_phase4d_recurrent_capacity_v3.md)
and [the fixed config](../../../../../configs/e010_phase4d_recurrent_capacity_v3.yaml).

Only width/depth varies: S=128/4 (927875), M=192/6 (3115587), L=256/8 (7369987).
K=4 shared steps, s_max=0.04 Å, beta=16.8, gamma=2; final-state supervision only.
500 successful updates per feasible arm; boundaries 0/25/50/100/250/500.
Select smallest passing S/M/L after all feasible arms finish. No development
selection, no old one-shot contract, no held-out pilot, no scaling beyond L.

`cache_manifest.json` pins the immutable, untracked `frozen_panel.npz` and exact
live E010 parity for all 60 examples. `cuda_preflight.json` records every arm's
length-500 batch-60 K=4 smoke with zero optimizer updates. `cpu_validation.json`
records focused CPU tests and separately lists inherited full-suite failures.
Arm boundary/result records and the final capacity report are added after
execution. Checkpoints and coordinate datasets are never committed.
