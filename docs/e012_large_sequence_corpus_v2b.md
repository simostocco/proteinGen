# E012 DATA V2B: low-disk UniRef50 corpus

Historical DATA-E commit e3cd8ad337407a6de968fde9a24fb027f1336fda and
all E012 pilot, continuation and generalization artifacts are immutable.
No training, optimizer updates, model scoring or generation.

Source pin: UniRef50 2026_03, official RELEASE.metalink size 8,780,552,383
and MD5 0492e3cf4093276ae4319ba24d52514f. Current-release HTTPS is only a
transport URL; frozen release/checksum metadata establish identity. Check official
metadata before acquisition and recheck after transfer. The initial sequential curl
used --continue-at -. Four resumable byte ranges from the same official mirror then
replace only this task's sequential transfer and preserve its verified byte prefix.
Validate exact HTTP Content-Range responses and complete segment byte counts/hashes,
then assemble and verify the full official MD5. Keep chunk inputs until the verified
raw-source completion marker, then record their reproducible cleanup. Range assembly
peak allowance is 18,561,104,766 bytes; the overall planned peak stays 79.78 GB.
Keep compressed raw source. Verify bytes, MD5, SHA256 before any preprocessing;
full gzip CRC/end-of-stream verified by mandatory complete streaming filter pass.

Resource plan: resource_plan.json, peak 79,780,552,383 bytes, below 70% of
189,611,429,888 initially free bytes (132,728,000,921). No complete decompression,
full filtered FASTA, or full corpus clustering. Screen batches of 100,000 against
the small protected set, sequentially. Reuse a hash-certified 1.01 GB protected-target
index; its 2 GB allowance is included in the 8 GB search-stage budget. A measured
10k historical search pilot projects 3.32 GB for 100k batches with 4x disk slack. Runtime floor is 30% of starting free
capacity. Free-space monitoring every 10s, with checks at transaction boundaries
and during subprocesses, stops before that reserve is breached. Report sampled
filesystem consumption separately from planned peak; unrelated D: writes may
affect free-space measurements. Use at most eight CPU threads and no GPU.

Protected scope: DATA-E explicitly left membership unbuilt. V2B resolves that
same scope: E012 validation, both E012 panels, E011 validation and panels, plus
all historical validation/test manifests under the canonical read-only
/mnt/d/Simone/proteinGen/data. protected_definition.json freezes source inventories,
hashes, membership and 53,615 unique sequences before acquisition. Older held-out
ownership is retained even where it intersects historical TRAIN (3,161 exact
sequences). Do not protect TRAIN-only sequences. No ownership changes for yield.

FASTA filtering: streaming gzip, ASCII formatting whitespace removal only, uppercase,
reject malformed/noncanonical records, 20<=length<=500, no cropping/splitting.
Exact index stores SHA256 and uncompressed header offset, not every sequence.
Every repeated hash is checked by full sequence string, using reservoir lookup or
gzip seek into immutable raw source. True collisions retain distinct offset keys.
Representative offset is first encounter in pinned FASTA; chosen UniRef ID is
lexicographic minimum among encountered exact duplicates.

Reservoir: retain at most 24M sequences plus at most one 100k checkpoint batch.
Frozen priority = SHA256(b'E012-V2B:12014\0' + canonical sequence ASCII bytes).
Order ascending (priority, sequence SHA256, source offset). No model-dependent
selection or repeated sampling. Retain all external exact unique records if below
24M. Historical strings are matched in the same source pass using a bounded 87,930-sequence
lookup, avoiding repeated gzip seeks for ordinary historical overlaps. Union historical unique TRAIN (verified 87,930), compare strings on hash
matches, retain dual provenance once. Historical candidates can be removed by
protected screening or deterministic final selection; no repeated training weight.

Homology convention: --min-seq-id 0.30 -c 0.80 --cov-mode 0, both query and
target coverage, -s 7.5, alignment-length identity mode 0. Record installed MMseqs
version and all commands per batch. Explicit alignment-mode 3 computes identity;
max-seqs 1000 improves detection capacity over the default 300. Remaining search
defaults, including E-value 1e-3 and low-complexity masking, remain defaults.
Exact protected strings are always excluded independently of heuristic search.
Screen every candidate, then independently re-search survivors before batch tmp
cleanup, then independently re-search final selected inputs in reversed batches.
No all-vs-all candidate search. Zero detected MMseqs matches is not a mathematical
proof of absence of every possible homolog. Result TSVs are retained compressed.

Final selection: first 20M in frozen reservoir priority order after exclusion,
or all clean sequences if between 10M and 20M. Below 10M, stop DATA2-C; no alternate
source. Preserve coarse UniRef50 representative provenance, length/AA telemetry.
No full local clustering is required or claimed.

Storage: one canonical gzip FASTA (concatenated deterministic gzip members),
gzip integer/ID index, compact Parquet metadata directory and 512 zstd Parquet
sequence shards. Assign each selected sequence in priority order to the shard
with fewest residues (ties: shard index). Shards use the existing loader schema
sample_id/sequence/length; metadata stores provenance and index separately.
Final output order is shard then shard_row, with selection_rank preserved.
No sequence duplication across shards. Loader/tokenizer/causal BOS target-shift
audit uses deterministic 10,000 final-output ranks floor(j*N/10000), CPU only.

Diagnostics: create 1M sample at fixed final-output ranks floor(j*N/1000000).
For 50% and 30% linclust, pilot 10k at fixed ranks. Project runtime, retained
disk and child max-RSS linearly with 4x slack. Pick largest of 1M/250k/100k/10k
meeting <=2h, <=8GiB estimated RAM, <=20GB disk and runtime free-space reserve.
This rule is frozen before diagnostic execution. Full final corpus remains
unchanged. Report sample size prominently; never extrapolate exact cluster counts
to the full corpus. Resource projections include fixed costs and may be conservative.
If even 10k is not feasible, report unmeasured diagnostics rather than changing data.

For cluster sizes n_i, N=sum n_i, p_i=n_i/N: inverse Simpson = 1/sum(p_i^2),
entropy effective = exp(-sum(p_i log p_i)); Gini = sum_ij |n_i-n_j|/(2*K*N).
Singleton fractions use explicit cluster and sequence denominators.

Resume: transactional attached SQLite journals checkpoint dedup index and reservoir
together. Reservoir stage inputs are pinned. Search batches have checksummed result
and zero-overlap certificates before cleanup. Independent verification progress is
separate from immutable exclusion DB. Assignment commits each 100k rows; final shards
and concatenated gzip offsets checkpoint individually. On resume truncate only the
uncertified append tail. Hash-check all final shard/manifest/FASTA/index bytes and
read them against immutable candidates before final certification. Cleanup is limited
to derivable tmp/DB artifacts and requires a checksummed completion certificate.
Never delete raw, frozen canonical corpus, final manifest or checksum files.

Commands (proteingen Python environment, PYTHONDONTWRITEBYTECODE=1):
python scripts/preflight_e012_sequence_corpus_v2b.py
curl -fsSL --continue-at - --retry 3 --connect-timeout 30 --speed-time 60 --speed-limit 1000
  -o <D-root>/raw/uniref50.fasta.gz.partial <official pinned-content transport URL>
python scripts/prepare_e012_sequence_corpus_v2b.py verify-raw
python scripts/prepare_e012_sequence_corpus_v2b.py reservoir
python scripts/prepare_e012_sequence_corpus_v2b.py screen
python scripts/prepare_e012_sequence_corpus_v2b.py verify-protected
python scripts/prepare_e012_sequence_corpus_v2b.py final
python scripts/prepare_e012_sequence_corpus_v2b.py certify
python scripts/prepare_e012_sequence_corpus_v2b.py diversity
python scripts/prepare_e012_sequence_corpus_v2b.py handoff

Only code, configs, official small metadata, protocol, hashes and summaries enter Git.
All large artifacts and MMseqs temp data stay under the D-disk versioned root.
