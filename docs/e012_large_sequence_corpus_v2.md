# DATA-E — stopped before bulk download

The conservative retained-artifact planning budget is 269.81 GB,
with 189.61 GB free on D:. The preferred 2.5× margin is
674.52 GB. This is an assumption-based budget,
not measured peak usage. Actual filtering may substantially reduce disk requirements.
Largest allowances: MMseqs temporary data 81.75 GB,
final formats 35.60 GB, exact-union database 35.04 GB,
decompressed source 30.00 GB. See resource_preflight.json for every term.

Official metadata confirms release 2026_03 dated 02-Sep-2026, 38,840,027 clusters,
uniref50.fasta.gz size 8,780,552,383 bytes, MD5 0492e3cf4093276ae4319ba24d52514f.
The release note MD5 passes. Metadata SHA256 values are recorded in checksums.json.
The raw archive was NOT downloaded; raw SHA256 and checksum verification are unavailable.
The current_release URL was inspected only for discovery, not used as sole provenance.
Both release-specific directory checks return HTTP 404.

Machine: 16 CPU threads, MemTotal 16,699,813,888 bytes (15.55 GiB), about 10.95 GiB
available at inspection; 4 GiB swap. MMseqs 15-6f452+ds-2 is installed.
Existing D-disk audits total approximately 12 GB; historical data was inspected read-only.
No duplicate UniRef50 acquisition was identified in inspected canonical directories;
this was not an exhaustive scan of every unrelated D-disk directory.

No final corpus, protected set, clustering, shards, audit sample or scale-up statistic
is available. Published UniRef cluster count is not a parsed sequence count. Historical
231,743 nominal / 87,930 exact unique references remain comparison denominators only.
No DATA-A/B/C claim or protected-integrity certification is made.

Resume requires sufficient disk for a reviewed conservative budget and source pinning.
An optimized streaming lifecycle could lower the budget, but is not yet implemented or
validated here. No files are deleted to create space. Historical reports are unchanged.
Only source metadata and preflight tests ran; the requested scientific corpus tests did not.
Training launched: NO. The complete machine-readable handoff is chatgpt_handoff.json.

Recommended next experiment: after a certified frozen corpus exists, run one fixed-budget
E012 causal-RoPE data-scaling comparison against historical unique TRAIN, retaining the
architecture and protected evaluation panels.
