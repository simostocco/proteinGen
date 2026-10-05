# E011 S1 sequence-context-only preparation

Base `origin/main`: `3b200ffb0389be87cc928c391117ce96bd8b5eb2`.
Branch/worktree: `e011-sequence-context-only`, `/home/simostocco/proteinGen-sequence-context`.
The existing branch was reused without reset, rebase, merge, or another worktree.
Phase 4C remains read-only at `/mnt/d/Simone/proteinGen`, branch
`e010-phase4c-local-auxiliary`, HEAD `a91d3432ec461f334ca5d9dd27fec9efd19b61d4`.
Both worktrees were initially clean. The fetched main matched the base.

Question: can a sequence-only Transformer learn residue-to-residue dependencies
beyond amino-acid marginal frequencies? This preparation does not answer that
question on real proteins. Current status is **S1-E — inconclusive**, because no
real-data E011 training or diagnostic evaluation of a trained model has occurred.
No definitive training is authorized by this contract.

## Historical evidence

The standalone `ProteinSequenceTransformer` is a causal decoder with RoPE,
a 24-token vocabulary, d_model 384, eight layers, and continuous learned length
conditioning injected per block. Its objective is autoregressive, not the S1
masked-context objective. It is unsuitable for an exact E006 Stage-A extraction.
Existing readers can load full manifests; S1 instead projects sequence columns.

E005 is a joint sequence/geometry model with four 128-wide sequence layers,
gated incoming/returned geometry conditioning and a sequence-to-geometry branch.
The completed `large_diagnostic_v2/protocol.json` (35,000-step selected checkpoint)
reports learned-minus-disabled geometry CE of -0.00196453 nats,
95% CI [-0.00262705, -0.00130166], and clean-minus-corrupted geometry CE
of -0.00000514903, CI [-0.0000127364, 0.00000235365]. Its trajectory is plateaued.
This is a small geometry-path effect, not proof of residue contextual learning.
Checkpoint hash: `7d74e15173a14334b0db66c7f14ce614e09041449d46de8796a9fc78340392cb`.

The exact recovered E006 final completed contextual classification is
**`marginal_frequency_only`**, emitted by
`/mnt/d/Simone/proteinGen/reports/experiments/E006_rich_geometry_codesign/phase3_stage_a_context_diagnostic_v1/report.json`.
SHA256: `8798c5ce0c4b35e60f45f056225e497404a836a42b4baabcc7cb34a1b5d8cd8a`.
The completed protocol independently records this hash and classification.
It evaluated the v4 35,000-step best checkpoint, hash
`eb20183cc73f14a3099136b9f85a0789c26f0093f92ff772bd911c4dbb1ee653`.
The postmortem corroborates that result. Later v5/v6 synthetic smokes and bounded
pilots do not supply a definitive final v6 contextual classification. The v6
250-update pilot's normal-minus-shuffle CE was only -0.00218925 nats.
The preparation report inventories and hashes available local E005/E006
JSON, JSONL, Markdown, YAML and hash artifacts across the active, legacy and
pre-relocation locations. Checkpoints were not loaded or altered.

Source audit found a v6 objective defect: `contextual_stage_a_loss_v6` computes
its sample hinge with attached shuffled CE; `paired_dropout_forwards` and its
production caller preserve that graph. Minimizing the hinge can therefore
increase shuffled CE. S1 deliberately corrects this by a no-grad shuffled
forward and an explicit detached shuffled CE. It preserves the reviewed 0.25
weight, 0.05-nat margin and paired dropout. The historical v6 CE component uses
token weighting, whereas its hinge uses equal-protein weighting; S1 uses
equal-protein weighting for both. No E006 source was changed.

## Model and objective

`src/protein_sequence_generation/context.py::SequenceContextTransformer`
extracts E006's 256-wide, eight-layer, eight-head, 1024-FFN, GELU pre-norm
Transformer with dropout 0.1, learned position embeddings up to 500, and final
LayerNorm. PAD=0, MASK=1 and the canonical 20 residues occupy IDs 2–21.
The output head contains only the historical canonical rows, producing 20-way
logits. Parameter count: **6,457,364**. There are no geometry modules and no
learned length embedding. The full-capacity parity test copies identical weights
and obtains exactly equal CPU logits to E006's canonical output slice.
Scratch initialization is frozen for the future S1 experiment; no E006 checkpoint
or E004/E007/E010 output initializes this model.

Allowed inputs are visible canonical identity, MASK, PAD, position and validity
mask. Coordinates, distances, pairs, contacts, secondary structure, geometry
labels, diffusion timesteps, PDB metadata and other experiment outputs are
excluded. Sample IDs are used only for deterministic corruption and panel
membership. Length is used for batching, diagnostics and baseline stratification.

Mask fractions are 0.15, 0.30 and 0.50. Training fractions cycle per optimizer
update. Masks draw without replacement, with rounded fraction times valid length
and a minimum of one target. The seed includes seed, epoch, sample ID and
fraction, so masks are independent of batch order. Targets are replaced by MASK.

For protein i, let n_i and s_i be mean canonical CE over its masked valid
canonical residues in normal and visible-shuffled context. Then
`loss = mean_i(n_i) + 0.25 * mean_i(relu(n_i - detach(s_i) + 0.05))`.
A protein with no eligible targets raises an error. Padding and noncanonical
positions contribute nothing. Paired forwards restore CPU and active CUDA-device
RNG for the shuffled branch; net RNG advancement equals one normal forward.
CPU validation hides CUDA entirely.

## Data and diagnostics

The audited E006 sidecars are reused only as a sequence source, preserving the
historical splits: **231,743 train**, **23,307 validation**; the train set contains
87,930 unique canonical sequences, and validation has 23,307. Duplicate training
samples remain equally weighted per sample. This cohort has structural
eligibility selection bias; absence of structural input does not remove that bias.

The reader requests only `sample_id`, `split`, `sequence`, `token_ids` from
Parquet. It never reads geometry columns or follows geometry/source paths.
A complete projected scan verified token-to-sequence agreement and lengths 20–500.
Actual retained membership joins have no missing IDs and zero cross-split
sample, canonical sequence, cluster, PDB and split-group overlap. Historical
MMseqs policy is 30% identity, 80% coverage, cov-mode 0. Cluster provenance is
inherited and checked through membership; clustering was not rerun.

| Stratum | Train | Validation | Diagnostic panel |
|---|---:|---:|---:|
| 20–64 | 21,466 | 2,016 | 410 |
| 65–128 | 66,946 | 5,005 | 410 |
| 129–256 | 90,706 | 8,459 | 410 |
| 257–384 | 38,882 | 5,915 | 409 |
| 385–500 | 13,743 | 1,912 | 409 |

The frozen 2,048 validation panel is never a training input. Selection uses seed
6111, 64 minimum per stratum, then round-robin fills to 2,048. Each target protein
has a frozen donor with different sample ID and sequence in the same stratum.
Relative-position mapping accommodates donor length differences. Donor context
is inserted only at visible positions; masked targets stay hidden.

All four diagnostic conditions share identical targets and masks: ordered
normal, within-protein visible shuffle, null (MASK at every valid site), and
another protein's context. Shuffle preserves the visible residue multiset.
Null retains positional/length information, but no visible residue identity.

Uniform, global unigram and length-bucketed unigram baselines are frozen using
TRAIN only; Laplace smoothing adds one count per canonical class. All diagnostic
CEs use the same target sites and equal-protein reduction. Gate A conservatively
requires improvement over **both** global and length-bucketed train unigrams,
resolving the ambiguity of a single training-unigram comparator.

For every fraction and all five strata, A–D require mean paired delta ≤ -0.05
nats/token and paired protein bootstrap 95% upper bound < 0 (2,000 resamples,
seed 6211). B compares shuffle, C null, and D donor context. CI resampling is by
protein, not independent tokens. Related proteins within a validation cluster
may still correlate; these CIs do not claim cluster independence.

Classification precedence is fixed: incomplete panels/strata → S1-E; all A–D
pass every fraction and stratum → S1-A; any fraction or stratum fully passes but
not all → S1-D; A passes every fraction but B–D fail → S1-C; uniform improvement
at every fraction, A fails every fraction and B–D fail → S1-B; other mixed
patterns → S1-E. Minimum eligible stratum size is 64.

## Verification and frozen artifacts

The focused tests cover exact full-capacity historical parity, absence of
geometry/length modules, padding invariance, hidden targets, loss eligibility,
equal-protein reduction, detached shuffle gradients, visible multiset preservation,
paired dropout/RNG advancement, null identity removal, donor exclusions,
train-only baselines, deterministic batch-independent masks, sequence-only
Parquet projection, diagnostic reproducibility and all classification labels.
The broader CPU-safe historical context and standalone sequence tests also run.

The real CPU smoke used two unmodified length-59 train sequences, the full S1
model, and normal/shuffled masked forward/backward. All activations/loss/gradients
were finite; loss was 3.29512191. No training optimizer update was taken.

A reduced 32-wide/two-layer synthetic model trained for exactly 150 CPU updates
on random-phase alternating sequences. Each position has the same train marginal,
so position alone cannot predict phase. Evaluation used a separate seed.
Normal CE at fractions 0.15/0.30/0.50 was 0.00545/0.00542/0.01759, compared with
0.69315 for the train unigram; shuffle, null and donor controls were all worse
by >0.05. This proves the infrastructure can learn synthetic contextual
relationships, not that real proteins pass S1 gates.

CUDA work and the unrestricted full repository suite are deferred:
`deferred_due_to_active_phase4c_gpu_workload`. The observed GPU was at 100%
utilization with 7,645 MiB used. Full-suite geometry/training/resource tests are
not appropriate during this workload. The sidecar metadata and shard inventory
hashes match historical configuration. Complete raw shard-byte rehashing was
not repeated to avoid large shared-disk reads; the audit validates all projected
sequence rows and records this remaining provenance limitation.

The versioned YAML, source, panel, baselines, audits and evidence are sealed by
`reports/experiments/E011_sequence_context_only/contract_manifest.json`.
`PYTHONPATH=src python scripts/prepare_e011_sequence_context.py` verifies an
existing seal and exits without overwriting it. Any contract change requires
an explicitly reviewed new version and seal. No definitive runner or training
budget is established by this preparation.

Recommended next action: review the sealed S1 contract and set a bounded
real-data pilot budget after Phase 4C releases the GPU.
