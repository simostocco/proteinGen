# E012 causal RoPE pilot — preparation complete, CUDA execution deferred

**CSEQ-E — inconclusive because execution is blocked by GPU ownership.** No scientific adjudication occurred. No E012 pilot training was launched; successful pilot updates: **0**. The observed Phase 4D CUDA trainer (PID 319802, `run_e010_phase4d_recurrent_capacity_v3.py train`) owns the GPU. It was left running. E012 CUDA smokes completed before that conflicting workload was observed.

## Sealed preparation

Branch `e012-causal-rope-sequence`, worktree `proteinGen-causal-rope`. Base E011 `30b657be81cd7bd943918980812d68ecaa0778de`. Preparation commit `76814fd071a20f1bcc1e381533bfe195381aa1ca`. Contract SHA256 `1a56b8958bb9a81c7dc269bce7c3efa79f19fc6dc85498beb09cfb10f346e612`. No pilot result commit or scientific results exist.

The original causal ProteinSequenceTransformer components remain byte-identical. Model: 15,533,952 parameters, d384, eight layers, six heads, FFN1536, GELU, dropout/attention dropout .10, maximum500, q/k RoPE and continuous log-length conditioning in every block. Fresh deterministic seed12012 initialization independently reproduced; no warm start.

Inputs: previous canonical amino acids, BOS/PAD, causal/key mask, RoPE and requested length. Geometry, structure labels, predictive PDB metadata, diffusion, historical structural outputs and E011 hidden states are excluded. Teacher forcing is BOS+a[0:N-1] => a[0:N]. Existing sequence_mean averages valid tokens within each protein, then averages proteins.

## Data and fixed diagnostics

All 744 raw shards and protected hashes verified. Derived manifests preserve 231,743 TRAIN and 23,307 validation identities. Exact E011 primary and independent panels (2,048 each) are disjoint and unchanged. Exact sequence, sample, protected cluster, PDB and split-group cross-split overlaps are zero under the historical 30% identity/80% coverage policy. Only sequence columns are loaded.

All four baselines use TRAIN counts and fixed additive alpha1. BOS is a separate history index20; trigram starts with BOS,BOS. Counts total42,578,064 residues. The following are static baseline likelihoods over all panel residues, using equal-protein weighting; they are not E012 model results.

| Panel | Global unigram | Bucketed unigram | Bigram | Trigram |
| --- | --- | --- | --- | --- |
| independent | 2.901697 | 2.901314 | 2.896606 | 2.909563 |
| primary | 2.900035 | 2.899043 | 2.894704 | 2.906180 |

Up to16 targets per protein are deterministically spread across the interior. Fixed positions and their algorithm are hashed. Shuffles contain exactly the original prefix multiset and no future residues. Windows1/4/8/16/32/64/full preserve original positions and mask earlier states/keys in every layer. Length ablation zeros the embedding at evaluation only. Gates A–E, 10,000 identity-bootstrap resamples, independent confirmation rules and classification precedence are immutable. Likelihood cadence:0/250/500/1000/1500/2000. Complete order/window/length diagnostics:2000.

## Tests and CUDA preflight

Full repository suite with unchanged read-only historical fixtures:1,665 passed,13 conditional skips. Eleven CUDA-skipped cases passed separately while the GPU was free. Three later evaluator/safety cases passed in the50-case focused rerun. Combined unique coverage:**1,679 passed, two optional skips**. Remaining skips are opt-in live RCSB API integration and absent optional pilot mmCIF fixtures. No tests were weakened. Ruff lint/format, syntax and whitespace checks pass.

Production CUDA smoke:RTX5060, longest length500, physical batch8, bf16 supported. Finite loss/gradients, optimizer mutation, bitwise checkpoint/resume with dropout RNG and full diagnostic evaluator passed. Peak allocated1022.64MiB, reserved1120MiB. Two functional length20 samples had exact length and canonical-only outputs. No biological quality evaluation was performed.

Frozen training regime:AdamW LR.0003, betas.9/.95, weight decay.01,100-update warmup, existing cosine scheduler over exactly2,000 updates, clip1. Physical8 × accumulation8 =64 equally weighted proteins/update. No pilot checkpoints were created. The smoke checkpoint remains local and ignored.

## Unavailable scientific endpoints

Model CE trajectory, model-minus-baseline deltas, normal-minus-shuffled prefix and bootstrap CI, KL/JS, prediction changes, window CEs, full-minus8/1 effects, length ablation, stratum performance and Gates A–E are **not evaluated**. Independent confirmation is not evaluated. Training decline or ordered-context learning cannot be inferred from preparation or synthetic smoke.

## Repository safety

E011 remains immutable S1-E at its result commit. Phase4C remains at215fb5b3eb127f690b382adae1c1607979169a85. Phase4D independently advanced from5eaa96c… to34ed55ee… and started CUDA training; E012 did not modify its source, checkpoints or outputs. The shared Python environment was not changed. No merge/rebase, larger run, structure conditioning or biological generation occurred.

## Exactly one recommended next experiment

Execute the sealed 2,000-update E012 pilot after Phase 4D releases the GPU and a fresh ownership check passes.
