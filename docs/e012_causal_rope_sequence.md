# E012 causal RoPE sequence pilot

E012 starts fresh from E011 result commit `30b657be81cd7bd943918980812d68ecaa0778de`.
E011 remains immutable S1-E; its proposed extension is not run.

The existing ProteinSequenceTransformer is reused without architecture changes:
d_model 384, eight blocks, six heads, FFN 1536, GELU, dropout/attention dropout
0.10, maximum 500, RoPE q/k, continuous log-length embedding injected in each
block, 24-way output. Exact parameter count: 15,533,952. Seed 12012; no warm start.
The accepted inputs are previous canonical residues, BOS, PAD, causal/key masks,
RoPE and requested length. No structural features, metadata, diffusion or E011
hidden states enter the model.

For N residues, input is BOS followed by the first N-1 residues; all N canonical
residues are targets. Existing sequence_mean computes each protein's mean valid
token cross entropy, then averages proteins. Physical batch eight accumulated
eight times gives exactly 64 equally weighted proteins per update. Training
shuffles all TRAIN sample identities once with NumPy seed 12012; 128,000 proteins
are processed in 2,000 updates without replacement. Sequence duplicates remain,
matching E011 ownership. AdamW LR .0003, betas .9/.95, decay .01, 100 warmup
updates and existing LambdaLR cosine over a fixed 2,000 updates; clip norm 1.
The configuration and contract record the exact scheduler interpretation.

The exact E011 populations and primary/independent panel identities are reused.
30% identity/80% coverage is the historical protected clustering policy; no new
split or panel selection occurs. Derived sequence-only Parquet views have only
sample_id, split, sequence and length. The reader rejects any extra columns
before loading. Identifiers are audit metadata and never model features.

All statistical baselines use TRAIN token counts with additive alpha=1 for each
of 20 canonical outcomes: global unigram, five length-bucketed unigrams, causal
bigram, causal trigram. BOS is a separate history index 20. Initial trigram
history is BOS,BOS; then BOS,a1. Counts retain all TRAIN sample multiplicities.
No EOS is a target. Model likelihood includes its full 24-way normalization;
special outcomes are not removed to improve reported CE.

Likelihood evaluations occur at 0/250/500/1000/1500/2000 on both fixed panels.
Complete order/window/length diagnostics occur only at the final checkpoint.
This evaluation cadence is fixed before training. No diagnostic selects model
hyperparameters. Checkpoints store model, optimizer, scheduler, scaler (null
for bf16/float32), all RNG states and the sampler permutation/cursor.

For each length N, zero-based target indices span max(2,floor(.1*N)) through
min(N-2,floor(.9*N)), inclusive. Take up to 16 unique floor-rounded linspace
indices. The selected identity/index list is frozen and hashed. For target i,
shuffle exactly residues [0:i], using the first eight little-endian bytes of
SHA256("E012:12012:{sample_id}:{i}") as NumPy seed. BOS stays first; target and
all future residues are absent. Prefix likelihoods use the entire 24-way output.
KL is KL(normal||shuffle); JS is symmetric, both in nats. Top-3 set comparisons
ignore ranking within the set. Average targets within each identity first.

Context windows keep only the last 1/4/8/16/32/64 previous residues (and BOS if
the entire available history fits), preserving original absolute input indices.
Earlier keys and query states are masked in every block, preventing indirect
multi-layer leakage. Full prefix includes BOS and all prior residues. Requested
protein length stays fixed. Length ablation zeros the continuous length embedding
output at evaluation; block projection biases remain. No retraining occurs.

Gates use paired identity bootstrap, 10,000 draws, seed 12112, 95% percentile CI.
A: normal minus global AND bucket unigram <= -.05, upper CI <0.
B: minus bigram and trigram upper CI <0; trigram delta <= -.01.
C: normal-prefix minus shuffled-prefix <= -.02 with upper CI <0.
D: full minus last8 <= -.01 with upper CI <0, and full minus last1 upper CI <0.
Independent A/B/D require negative point deltas; C also requires upper CI <0.
E is checked on both panels: no stratum normal CE exceeds its bucketed baseline;
no normal-prefix minus shuffle delta >=+.02; at least four strata have negative
shuffle deltas. The .02 material failure threshold follows the Gate C effect.
All five strata are reported without aggregate suppression.

Classification precedence: infrastructure prevents interpretation => CSEQ-E;
all primary A-E plus independent confirmations => CSEQ-A;
primary A-D favorable but E fails in either panel => CSEQ-D;
primary C/D plus independent C/D confirmations but likelihood gates fail =>
CSEQ-C; otherwise CSEQ-B (marginal or insufficient ordered/long-context evidence).
This total decision rule is frozen before training, including no-unigram-benefit
cases that the short category descriptions do not explicitly enumerate.

CUDA smoke verifies longest-length forward/backward, finite optimizer state,
mutation and bitwise resume including dropout RNG. bf16 is used only if supported
and the full smoke passes; otherwise a verified float32 fallback is recorded.
Only the functional two-sequence, length-20 canonical sampling smoke is allowed.
No biological generation, folding, structure conditioning or larger run follows.

Repository audits and raw input hashes are prerequisites. Full tests use the
unchanged historical fixture mirror if the worktree lacks ignored historical
dependencies; baseline fixture failures are reported and resolved without test
weakening or historical writes. All new results stay in the E012 namespace.
