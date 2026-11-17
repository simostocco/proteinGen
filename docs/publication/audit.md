# Publication audit — 2026-11-17

## Preservation and selection

All 60 original local branch tips match refreshed GitHub tips; the snapshot is an additional published branch. All 56 registered original worktrees remain in place. Useful pending files still match the 282-file preservation commit `2030c692e6ebb90ddff05a91746f29978217ca65`; no newer useful source was found. The original worktrees, indexes, artifacts and branch tips were not modified. The integration is prepared in an isolated temporary clone on `research-presentation-20261117`, based on main `3b200ffb0389be87cc928c391117ce96bd8b5eb2`.

[Branch inventory](branches.md) lists every original branch and disposition; [file provenance](integration_sources.json) identifies selected source and textual evidence. No experiment or snapshot was merged wholesale. Later training/orchestration code remains branch-local. New source includes E011/E012 context diagnostics, E013/E015 denoiser and objective primitives, and E016/E017 contracts/internal-coordinate/action-density/SAC-target primitives. Existing scientific results are unchanged. Import ordering and equivalent exact-type comparisons were cleaned up in selected modules; the E011 personal data path was replaced with a relative path. These are the only intended behavior-neutral source adaptations besides license comments.

## Credentials, privacy and assets

The [history scan](history_scan.json) inventoried 4,024 unique reachable blobs across local and refreshed remote refs, including the snapshot. It scanned 4,022 blobs below 5 MB for common AWS, GitHub, Hugging Face and OpenAI token formats and private-key headers: no matches. The two larger historical text artifacts (6.46 MB per-example records and a 39.08 MB preprocessing log) were size-inventoried, not exhaustively inspected. The scan is not a guarantee that no custom-format secret or private information exists. No newly selected file is a dataset, checkpoint or model-weight payload. Raw runtime artifact trees, environments, process dumps and datasets remain excluded. Public Git author metadata and historical local machine paths already exist in history; this PR neither rewrites history nor certifies those as anonymized.

## Source licensing

Project license: MIT. An existing ProteinMPNN-derived CA inference core named its exact source revision but omitted the upstream copyright/license notice. This PR retains the MIT notice and adds attribution. A separate conservative BSD-3-Clause notice credits the existing ProGen2 compatibility architecture. See [third-party notices](../../THIRD_PARTY_NOTICES.md) and their upstream links. No proprietary vendor payload was identified among the selected source. A full legal assessment of historical assets and checkpoint licensing is outside this audit; no external weights or datasets are redistributed.

## Scientific publication limits

Imported results are original textual reports, with original byte hashes and evidence-root names recorded in the manifest. The integration did not recompute model performance. Later E017 TEST-history uncertainty is explicitly retained, alongside validation reuse, failed promotion decisions and asymmetric stopping budgets. V9.7 engineering/admission qualification is distinct from production training efficacy. The evidence map avoids claiming independent replication, biological performance or a deployable generator.

This PR is suitable for review as a research portfolio integration. It should not be described as already on main before merge. Remaining data availability and historical privacy/licensing uncertainty must not be represented as fully resolved.
