# Execution-only throughput continuation

Scientific source b629783; preceding execution9c24446; recovery boundary372. Actor17,438,114 parameters only; full capacity63,389,056 retained.

The independently versioned execution config preserves effective AA64, separate geometry32, goals8, all weights/LR/moments/panels/gates. AA microbatch8, goal microbatch4; complete geometry remains scalar because more aggressive grouping failed existing numerical tolerances. Qualified TRAIN Parquet strings are held in a byte-bounded RAM LRU; no full-dataset feature cache or causal KV cache. The single CPU producer prefetches at most two AA microbatches within the current optimizer update; all work is joined before a durable boundary. Native prefix geometric invariants can be shared only for the exact same literal masked prefix; J1 outputs and actor inputs are bitwise verified. Geometry prefetch and compiled RoPE are disabled by selected configuration after isolated performance tests.

Atomic latest/best recovery is inherited unchanged. Source checkpoint/dataset evidence is never retired by this execution. The operational PAUSE_REQUEST protocol is inherited from9c24446. Global ceiling10000; future milestones1000/2500/5000/10000 and all scientific/resource gates remain sealed. No SAC.

Evidence, operational live status, exact commands and provenance: /mnt/d/Simone/proteinGen_data/e017_crisp_v3_throughput_20261021_v1/.
