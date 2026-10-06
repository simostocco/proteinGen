# E012 data v2 preflight protocol

No training, optimizer steps, historical artifact changes, or alternative databases.
This commit records a failed pre-download gate, not a working corpus pipeline.

Source: official UniRef50 2026_03 representative FASTA. Preserve release metadata,
size and published MD5; a moving URL alone cannot establish scientific provenance.
Release-specific directories checked on UniProt and EBI return 404. Resolve an
immutable release URL before bulk acquisition, or explicitly review a transport
protocol that enforces the recorded release and checksum across a resumable transfer.
Never infer completion from a partial file. Verify official MD5, SHA256 and gzip
integrity before preprocessing. Do not modify raw input.

Storage: existing D-disk corpora use proteinGen_audits for audits and proteinGen/data
for historical structure data. No established external sequence foundation root was
found. Reserve the requested fallback root, with raw/checksums/filtered/protected/
mmseqs/clusters/manifests/corpus/stats/tmp subdirectories, only after the gate passes.
No directories or large files have been created there.

Historical policy evidence: configs/e007_split_homology_audit_v1.yaml and
src/protein_distance_diffusion/evaluation/e007_split_homology.py specify
--min-seq-id 0.30 -c 0.80 --cov-mode 0, sensitivity 7.5. Mode 0 requires both
query and target coverage. Historical clustering uses --cluster-mode 0 and
--remove-tmp-files 0. Do not replace bidirectional coverage with query-only coverage.
Candidate-vs-protected search and independent final verification remain unimplemented
and unexecuted; all MMseqs version-specific flags must be recorded at execution.
Installed MMseqs version: 15-6f452+ds-2.

Protected-set discovery: E012 validation.parquet and panels.json; E011 primary_panel.json
and diagnostic_panel.json (both E012 panels were historically asserted to be subsets
of validation); inspect all other historical held-out sequence manifests, including
repository test split inventories. Resolve and hash actual sequences before screening.
This discovery is incomplete and no protected set is certified by this preflight.

Resume order: disk gate; immutable source verification; resumable bulk acquisition;
streaming uppercase FASTA parse (format whitespace only), reject malformed/noncanonical
or lengths outside 20–500; disk-backed exact sequence dedup with deterministic source-ID
representatives; unique historical TRAIN union with both provenance flags; protected
screen; 50% linclust; fixed-subset 30% runtime/RAM/disk preflight; full 30% if feasible,
otherwise documented sample measurement; deterministic cluster-aware selection; independent
protected search; frozen FASTA/manifest/index; residue-balanced shards; 10k loader audit.
Every expensive stage requires atomically published completion markers with input,
output, script and command hashes. No downstream output exists in this result.

Proposed selection: one representative per local cluster first; hash cluster IDs
with seed 12013 for a deterministic order. Fill in rounds with one unused sequence
per cluster, avoiding size-proportional weights. If representatives exceed 20M,
preserve natural length-stratum proportions via deterministic largest-remainder
quotas. Freeze complete ordering/tie rules and test them before selection. Preserve
the full eligible pool. Below 20M use all; below 10M stop without supplementation.

Cluster metrics planned: for sizes n_i, N=sum n_i, p_i=n_i/N,
inverse Simpson = 1/sum(p_i**2); entropy effective = exp(-sum(p_i*log(p_i))).
Gini = sum_i sum_j |n_i-n_j|/(2*K*N), K number of clusters.
No effective-diversity results are inferred from UniRef counts alone.
All requested corpus integrity tests and causal/tokenizer loader audits remain pending.
