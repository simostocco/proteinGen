"""R1d compact scientific publication; distinguishes untested ranking from failure."""
from r1d_common import *
import collections,numpy as np,subprocess

def main():
 inv=json.load(open(R/'analysis/ELIGIBLE_PARENT_INVENTORY.json'));pre=json.load(open(R/'analysis/PARENT_PRELIMINARY.json'));graph=json.load(open(R/'analysis/FAMILY_GRAPH.json'));bench=json.load(open(R/'STABILITY_VARIANT_BENCHMARK.json'));cfg=json.load(open(R/'configs/VARIANT_INFERENCE_FROZEN.json'));fold=json.load(open(R/'ALPHAFOLD_VARIANT_INFERENCE_MANIFEST.json'));ranking=json.load(open(R/'STABILITY_RANKING_RESULTS.json'));download=json.load(open(R/'TSUBOYAMA_DATASET_PROVENANCE.json'));member=json.load(open(R/'data/CSV_MEMBER_PROVENANCE.json'));native=json.load(open(R/'configs/NATIVE_REFERENCES.json'));labels=json.load(open(R/'analysis/DEVELOPMENT_LABELS.json'))+json.load(open(R/'analysis/AUDIT_LABELS_SEALED.json'));by={r['id']:r for r in fold['rows']};p=inv['parents'];qualified=[n for n,m in p.items() if m['length_40_72'] and m['qualified_exact_single_substitution_rows']>0];qfamilies=len({graph['family_by_parent'][n] for n in qualified});origin=collections.Counter(m['origin'] for m in p.values());qorigin=collections.Counter(p[n]['origin'] for n in qualified);source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R/'source',text=True).strip()
 archive=R/'data/Processed_K50_dG_datasets.zip';assert sha(archive)==download['sha256'];parent=D/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';expected='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';assert sha(parent)==expected
 old=json.load(open(C/'PROVENANCE_SHA256.json'));mismatches=[]
 for f,h in old['evidence_files'].items():
  if sha(C/f)!=h:mismatches.append(f)
 assert not mismatches
 seconds=sum(v['worker_wall_seconds'] for v in fold['rows']);peak=max(float(v.split(',')[1]) for r in fold['rows'] for v in r['GPU_samples']);assert seconds<=1500 and len(fold['rows'])<=12
 # Keep the original archive receipt separately before augmenting its provenance.
 receipt=R/'data/ARCHIVE_DOWNLOAD_RECEIPT.json'
 if not receipt.exists():write('data/ARCHIVE_DOWNLOAD_RECEIPT.json',download)
 metadata=json.load(open(R/'network/PIPELINE_COMMIT.json'));provenance=dict(download,processed_member=member,source_pipeline_commit=metadata['sha'],source_pipeline_URL='https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline',quality_notebook={'path':'network/Data_quality_filtering_script.ipynb','SHA256':sha(R/'network/Data_quality_filtering_script.ipynb')},STEP5_density_inference_code={'path':'network/STEP5_module.ipynb','SHA256':sha(R/'network/STEP5_module.ipynb')},attribution='Tsuboyama et al., Nature620,434-444(2023), doi:10.1038/s41586-023-06328-6; Zenodo7992926 v2_230420, CC BY4.0',GitHub_license_note='No separate LICENSE file exposed in the pinned repository tree; source notebooks retained for audit, no assertion of a new code license',ProteinGym='not downloaded or used; repackaged Tsuboyama assays would not be independent evidence')
 write('TSUBOYAMA_DATASET_PROVENANCE.json',provenance)
 flanks=[]
 for l in labels:
  for r in l['source_rows']:
   full=r.get('aa_seq_full','');seq=l['sequence'];start=full.find(seq);assert start>=0 and full.find(seq,start+1)<0;flanks.append((full[:start],full[start+len(seq):]))
 assert len(set(flanks))==1
 write('analysis/CONSTRUCT_AND_FLANK_CHECK.json',{'all_selected_rows_core_occurs_once':True,'all_selected_assay_flanks_identical':True,'common_flanks':list(flanks[0]),'AF_input':'exact aa_seq experimental core; aa_seq_full flanking assay sequence retained in source; no unsupported fragment reinterpretation','WT_reference':'author per-parent median, not forced zero for individual WT replicates'})
 wtqueries=[]
 for j in cfg['jobs']:
  if j['WT']:wtqueries.append({'parent':j['parent'],'role':j['role'],'sequence':j['sequence'],'sequence_sha256':j['sequence_sha256'],'N':j['N'],'public_PDB_exact_core_mapping':{'pdb_id':next(v['PDB_accession'] for v in bench['selected_families'] if v['parent_key']==j['parent_key']),'native_full_chain_start0':native[j['parent_key']]['start0'],'native_core_length':j['N']}})
 write('PROPOSED_MSA_QUERY_REQUEST.json',{'status':'NOT_AUTHORIZED_NOT_EXECUTED','preferred_followup':'4 public WT-core MSA queries; then12 paired fixed-family-background local predictions, same homolog rows and mutated query row; distinct controlled diagnostic, not genuine query-specific novel inference','endpoint':'https://api.colabfold.com','official_workflow':'ColabFold/MMseqs2, serial single client, follow rate limits; no local database installation','WT_queries':wtqueries,'preferred_request_cap':4,'preferred_local_AF_cap':12,'preferred_aggregate_worker_seconds':1500,'download_cap_bytes':100*2**20,'alternative':'12 genuine query-specific searches for the published WT/mutant sequences below, with depth/coverage differences reported; requires separate12-query approval, not implied by4-WT approval','alternative_query_sequences':[{k:j[k] for k in ['id','parent','role','sequence','sequence_sha256','N']} for j in cfg['jobs']],'no_request_sent':True,'no_generated_J1_sequence':True})
 table='| Parent | Role | Length | Author WT ΔG | Moderate ΔG | Strong ΔG |\n|---|---|---:|---:|---:|---:|\n'
 for f in bench['selected_families']:
  ls=[l for l in labels if l['parent_key']==f['parent_key']];vals={l['class']:l['experimental_dG'] for l in ls};table+=f"| {f['parent']} | {f['role']} | {f['N']} | {vals['WT']:.3f} | {vals['moderate']:.3f} | {vals['strong']:.3f} |\n"
 ft='| Predicted sequence | pLDDT | pTM | PAE-good fraction | WT-native mapped recovery |\n|---|---:|---:|---:|---:|\n'
 for r in fold['rows']:
  f=r.get('features',{});rec=r.get('WT_native_fold_compatibility',{});ft+=f"| {r['parent']} WT | {f.get('mean_pLDDT',float('nan')):.3f} | {f.get('ptm',float('nan')):.2f} | {f.get('long_range_PAE_good_fraction',float('nan')):.3f} | {rec.get('robust_fixed_map_TM_similarity_fullN',float('nan')):.3f} |\n"
 write('EXPERIMENTAL_STABILITY_DATA_AUDIT.md',f'''# Experimental stability source audit

Obtained the original Zenodo7992926 processed archive, version v2_230420, {download['bytes']:,} bytes. Official MD5 verified; SHA256 `{download['sha256']}`. License CC BY4.0, attribution in the provenance JSON. The exact Dataset2/3 member is697,658,024 uncompressed bytes; streamed completely, ZIP CRC and decompressed SHA verified. Only one compact selected-column source table ({(R/'data/STABILITY_SOURCE_ROWS.parquet').stat().st_size/2**20:.2f}MiB) was written, preserving string markers and zero-based source-row IDs. The large CSV and Dataset1 were not extracted. MacOS resource-fork ZIP entries were identified and excluded by exact member path after an initial preflight assertion; no source data changed.

Reproduced776,298 rows and479 parent records: {origin['natural']} natural backgrounds (including13 author-modified natural backgrounds) and {origin['designed']} designs. These are parent records, not479 independent families. Numeric dG_ML rows675,467; censored96,294; masked/missing '-'4,537. Numeric ddG_ML607,839; missing '-'168,459. Missing/censored values are retained as such, never called unstable or converted into exact endpoints. ProteinGym was not used because the original source was accessible; its repackaging would not add independent experimental evidence.

Our additional ranking inventory requires numeric author dG_ML/ddG_ML, finite posterior bounds, combined interval width≤0.5kcal/mol, dG strictly between−0.95 and4.95, exact same-length canonical single substitutions at the declared site, original WT reference consistency, length40–72, and no introduced Cys. It contains {inv['qualified_single_rows']:,} rows / {inv['qualified_unique_sequences']:,} distinct variant sequences across {len(qualified)} parents ({qorigin['natural']} natural/{qorigin['designed']} designed) and {qfamilies} operational family groups. All exclusions are extra study restrictions, not rewritten source quality flags. Original raw values and measurements remain available in the compressed archive and source parquet.

47 parents in38 operational families provide the prespecified strong/moderate contrasts and exact TRAIN construct links before final qualified-coordinate search. Four were reserved without AF scores. Author quality metadata, protease-specific estimates, intervals and filtering code are retained. The pinned pipeline source is `{metadata['sha']}`; notebooks were inspected, not executed to refit experiments. No neural reward model was fitted.
''')
 write('STABILITY_LABEL_SEMANTICS.md',f'''# Stability labels and uncertainty

These are thermodynamic free-energy estimates inferred from measured cDNA-display proteolysis, not direct calorimetry and not a generic cellular fitness score. K50 fits and an unfolded-state cleavage model underpin the inference. `_ML` marks the authors' quality-filtered machine-learning-ready columns; it does not mean labels predicted by a neural model.

In this release, larger ΔG (kcal/mol) means greater resistance to equilibrium unfolding. Algebraically verified across {pre['assay_sign_verification']['num_uncensored_single_rows_checked']:,} uncensored single substitutions: `ddG_ML = dG_ML − author WT median`, maximum absolute discrepancy {pre['assay_sign_verification']['max_abs_dG_minus_ddG_minus_authorWT']:.3g}kcal/mol. Positive ΔΔG is stabilizing here; negative is destabilizing. This sign must not be imported from resources using the opposite folding-energy convention. Author WT median is `Single_DMS_list.wt_dg_med`. Individual WT replicate rows can have nonzero ddG; all are preserved. The displayed WT label is the author reference, with a conservative envelope of qualified replicate95% intervals, explicitly not a newly inferred95% interval for the median.

Dataset1 retains the broader processed assay pool. The Dataset2/3 member contains the curated absolute-stability population; the ΔΔG-ready tier has usable WT reference behavior. We do not manufacture ΔΔG values for Dataset2-only rows. Saturated high-stability parents cannot provide unbiased stabilizing contrasts. Symbolic '<'/'>' bounds and '-' masks remain nonnumeric. The selected pairs avoid both dynamic-range boundaries and have at least1kcal/mol separation with nonoverlapping source intervals. They are relatively destabilized, not experimentally proven nonfolding sequences.

STEP5 code reports posterior medians and2.5/97.5 percentiles; `deltaG_95CI` is interval width. Its thermodynamic conversion uses an RT factor0.58kcal/mol. The study reports PBS at approximately22°C. Posterior uncertainty does not include every systematic error or prove cooperative equilibrium folding. Trypsin/chymotrypsin are complementary assay checks, not independent confirmations of every molecule's folding. Their raw estimates/intervals remain separate, and the author filtering notebook examines inter-protease patterns, potential folded-state cleavage and Cys artefacts. We did not reinterpret the author's `Stabilizing_mut` flag as a substitute for quantitative ΔG.

The measured constructs include common assay flanks; AF receives the exact released `aa_seq` core. Selected core/flank substitutions are verified; `aa_seq_full` is preserved. This is a controlled in-vitro assay-specific comparison, not a statement about solubility, function, oligomeric behavior or in-vivo stability.

Sources: [primary paper](https://www.nature.com/articles/s41586-023-06328-6), [official data](https://zenodo.org/records/7992926), [pinned author pipeline](https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline/tree/{metadata['sha']}).
''')
 write('VARIANT_SELECTION_PROTOCOL.md',f'''# Prospective parent and variant selection

Policies are preserved in `configs/SELECTION_AND_ANALYSIS_PREDECLARED.json` and its pre-inference v2 revision. Quality/effect-size/ranking thresholds did not change after prediction. Parent selection requires a reliable author WT reference2.5–4.5kcal/mol, at least50 qualified substitutions and exact experimental construct/native-coordinate interpretation. Each triple is WT, moderate loss1.0–1.5 and strong loss≥2.5kcal/mol; all eligible pair gaps≥1 with nonoverlapping intervals. Variants are deterministic SHA choices inside these bins, not chosen for AF performance. This fits three sequences per parent; WT is the relatively stable member rather than adding an unsupported fourth query.

The first family proxy combined30%-identity short-sequence links transitively and collapsed473/479 parents. It was rejected before any AF prediction, with code/results/config preserved in `analysis/rejected_family_proxy_v1`. The replacement uses author WT_cluster, source-PDB and accepted protected-group merges plus strong50%-identity/80%-shorter-coverage global alignment links. This matches the existing split's primary near-duplicate scale and avoids pretending weak helical-sequence matches certify family homology. It yields115 components overall and {qfamilies} with qualified variants. Family separation remains operational; undiscovered remote homology and shared short folds are explicit limitations, not erased by distinct names.

Exclude operational families with exact E016 TEST construct overlap; inspect identity/split ownership only, no TEST labels/coordinates/scoring. Selected parents map to qualified TRAIN native observations. Equal-family sampling gives two development and two audit groups; roles are fixed by SHA of family ID. Prefer two64–72-residue parents and two shorter ones where eligible. Exact native substrings and source hashes are checked. Official PDBe records identify THO1, villin headpiece, RXLR effector and Sda monomers; no family is chosen using recovery scores.

{table}

Primary ranking signal pTM is fixed before scoring. Secondary pLDDT, confidence fraction, PAE-good and the fixed equal local/global composite are descriptive; no regression or audit-based metric selection. Family-level pair accuracy, Spearman/Kendall and effect-size strata are specified, with score tie tolerances in the policy. No per-variant bootstrap is allowed. OutcomeA would be promising feasibility only; two audit families cannot certify a universal reward. The frozen baseline-adequacy gate stopped inference after the first two development WT calls, so no ranking or independent audit was completed.
''')
 write('MSA_COMPARABILITY_AUDIT.md',f'''# Matched input availability and limitation

Inspected the10 persisted R1b public-control MSAs. None matches these four selected parent core sequences exactly. No unrelated MSA was borrowed, no mutant was paired against a differently prepared WT background, and no new external MSA query was made. AFDB entries and prior native predictions are not fresh predictions of these mutations and were not pooled.

A fair query-only input was technically available for every sequence: one exact query row, no experimental coordinates/templates, same model1pTM weights, three recycles, seed151502, no relaxation, maxMSA1:1, existing BF16/FP32 implementation, CPU preprocessing and GPU folding. Model weights SHA remains `5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84`. The same query-only setting applies to WT and mutants; it is not MSA-assisted inference.

{ft}

Both development WT recovery scores are below0.50 and both pTM values below0.50, triggering the explicitly frozen adequacy stop. This shows that the selected query-only prerequisite did not meet this study's adequacy domain, not that matched inputs are impossible in general or that confidence cannot ever rank stability. No mutants or audit proteins were folded; no replacement families were introduced to chase successful examples.

A useful MSA-assisted continuation needs separately approved network access. `PROPOSED_MSA_QUERY_REQUEST.json` lists all four exact public WT cores and an alternative12-query WT/mutant panel. Four WT searches support a deliberately fixed homolog-background comparison with only the query row changed; that assumption must remain explicit. Twelve genuine query-specific searches are a separate protocol with possible depth differences. Neither is authorized or executed now. No evidence for novel J1 sequences follows from either natural-family protocol alone.
''')
 write('CONFIDENCE_VS_STABILITY_ANALYSIS.md',f'''# Stability ranking is not evaluated

OutcomeD: the required mutant comparisons were not executed after the prospective WT adequacy stop. Within-parent Spearman/Kendall, pairwise stability ranking, large-effect discrimination, confidence calibration and independent audit are therefore undefined, not zero and not failed biological measurements. Two WT predictions cannot determine whether pLDDT, pTM or PAE ranks experimental stability.

{ft}

The primary pTM signal and three secondary confidence measures plus one fixed composite remain unchanged. `STABILITY_RANKING_RESULTS.json`, `REWARD_CANDIDATE_COMPARISON.json` and `INDEPENDENT_AUDIT_RESULTS.json` explicitly retain unscored cases and no fitted coefficients. Experimental values and all12 sequence definitions remain recoverable. No hidden formula search, assay-label deletion or reclassification of missing oracle results occurred.

The dataset has strong within-parent experimental contrasts, but this is not evidence that AF predicts their ordering. The historical R1b confidence-to-recovery MAE0.358 remains failed. Structural recovery, experimental stability ranking and novel-sequence applicability are separate gates. No reward certification follows from accurate MSA-assisted recovery of other PDB proteins.
''')
 write('AF2_MUTATION_SENSITIVITY_REPORT.md','''# Mutation sensitivity remains unmeasured

No mutant was folded: the two development WT calls triggered the sealed adequacy stop. Consequently no change in pLDDT, pTM, PAE, predicted geometry or WT-versus-mutant fold similarity is available. It would be incorrect to claim that AF confidently preserves a fold despite experimental destabilization, or that it detects destabilization, from these calls.

The prepared analysis compares exact same-length mutant predictions against their paired AF WT and the experimental WT conformation separately. The latter is WT-fold compatibility, not an experimentally observed mutant structure. Large relative destabilization does not imply a proven unfolded/nonfunctional mutant. The prospective analysis will retain confident destabilized sequences if they occur rather than fitting them away.

Published native-recovery evidence establishes useful structural capability in an MSA-supported scope. AlphaFold structural confidence is not itself a free-energy measurement; this study has not supplied the missing experimental correlation. No intrinsic AF mutation-insensitivity verdict is justified by an unrun comparison.
''')
 write('RL_REWARD_GO_NO_GO.md',f'''# Reward decision: OutcomeD, insufficient matched informative predictions

Data source/integrity GO; prospective experimental benchmark GO within its disclosed family/assay constraints. Query-only execution operational: {len(fold['rows'])} successful real calls. Selected query-only native adequacy NO-GO under the frozen gate. Experimental mutation ranking NOT EVALUATED. Independent audit NOT EVALUATED. Novel-sequence applicability UNVERIFIED. Primary sequence-stability reward NO-GO; production RL NO-GO.

Keep AlphaFold as a conditional structural evaluator or auxiliary structural signal supported by the earlier MSA-native evidence. It is not a certified primary stability objective. This result neither proves nor disproves mutation sensitivity; the missing mutant/audit predictions prevent that conclusion. No success threshold from R1/R1b/R1c was changed. The amyloid/context counterexample and failed MAE remain historical stress evidence.

A separately approved follow-up could acquire4 public WT-core MSAs and perform up to12 fixed-family-background local folds within25 worker minutes, under a new locked protocol. Alternative genuine query-specific inference requires12 approved MSA submissions. Exact sequences/hashes and both request scopes are in the proposed query manifest. No submission occurred. Positive feasibility would still require a larger family-separated experimental audit and evaluation of shallow-MSA/64–128-residue transfer before production rewards.
''')
 write('NEXT_E016_RL_DECISION.md','''# Next E016 decision

Do not update J1 or start SAC/policy gradient. Preserve the +7,500 initializer, frozen reference KL, PDB40/30/30 AA behavioral cloning, qualified geometry BC, sequence-first objective and immutable delayed-reward replay. R1c's Transformer-critic synthetic success remains unchanged; no real protein-return learning is demonstrated here.

The smallest concrete corrective request is four serial official public WT-core MSA queries, at most100MiB new downloads and12 serial local AF2 calls /25 aggregate worker minutes. Test WT adequacy first, then the unchanged development/audit stability ranking using the declared fixed-family-background assumption. Exact query sequences are in `PROPOSED_MSA_QUERY_REQUEST.json`. Genuine per-variant MSA inference instead needs12 queries; approve one explicitly versioned protocol, not an unspecified mixture. Neither request is executed in this task.

Before first real RL updates, require actual within-parent experimental ranking on independent families under the intended oracle protocol, uncertainty/abstention rules available at inference, stable-vs-destabilized controls, and no large high-confidence ranking failures. A first expanded proposal is16 families (8 development/8 audit),3 variants and2 seeds:96 local calls, capped100 worker minutes, plus16 WT-only MSA requests for fixed backgrounds or48 per-sequence searches for a query-specific protocol. This is a budget proposal, not a guarantee of statistical sufficiency or authorization. Use observed family-level variance to assess power, retain original errors, and stop if confidence lacks useful stability sensitivity.

Even a positive natural-family benchmark40–72 does not certify generated sequences64–128 with shallow/no MSAs. That transfer needs its own prospective controls and a budget-matched generated-sequence/reference evaluation. If experimental ranking fails, use AF only as structural support and validate a separate scientifically justified stability objective before any policy optimization. No neural reward fitting, teacher/distillation or RL execution is performed now.
''')
 # Compact resource/network ledger; browser lookups listed separately from measured HTTP payload.
 http=json.load(open(R/'network/METADATA_HTTP.json'));contexts=json.load(open(R/'analysis/PARENT_BIOLOGICAL_CONTEXT.json'))['rows'];network_bytes=sum(e['bytes'] for e in http)+download['bytes']+sum(row[k+'_request']['bytes'] for row in contexts for k in ['summary','molecules']);free=guard();resource={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_commit':source,'start':json.load(open(R/'RESOURCE_START.json')),'final_free_bytes':free,'minimum_reserve_bytes':10*2**30,'AF_calls':len(fold['rows']),'AF_successful_calls':sum(r['status']=='SUCCEEDED' for r in fold['rows']),'AF_technical_failures':sum(r['status']!='SUCCEEDED' for r in fold['rows']),'AF_worker_seconds':seconds,'AF_sampled_peak_GPU_MiB':peak,'AF_budget_ledger':fold['budget'],'stopped_by':'prospectively frozen query-only WT native-adequacy gate','unused_AF_allowance':12-len(fold['rows']),'remote_MSA_queries':0,'scripted_HTTP_GET_requests':len(http)+1+2*len(contexts),'scripted_response_bytes':network_bytes,'browser_documentation_lookups':'official paper/Zenodo/author code/ProteinGym; payload not instrumented and not counted in scripted transfer total','new_model_database_downloads':0,'J1_updates':0,'J1_forwards':0,'production_RL_updates':0,'TEST_evaluation':False,'disk_deletions':0,'J1_checkpoint_SHA256':expected,'R1c_evidence_verified_unchanged':len(old['evidence_files']),'R1c_mismatches':mismatches,'R1c_manifest_SHA256':sha(C/'PROVENANCE_SHA256.json'),'raw_data_archive_bytes':archive.stat().st_size,'compact_source_parquet_bytes':(R/'data/STABILITY_SOURCE_ROWS.parquet').stat().st_size,'compact_qualified_parquet_bytes':(R/'data/QUALIFIED_SINGLE_VARIANTS.parquet').stat().st_size,'parents_source':len(p),'parents_qualified':len(qualified),'qualified_operational_family_groups':qfamilies};write('RESOURCE_AND_NETWORK_USAGE.json',resource)
 write('EXPERIMENT_DECISIONS.md',f'''# R1d decisions and measured outcomes

Source `{source}`; starting historical R1c `{old['source_commit']}` unchanged. No J1 forward/update, production RL, new external MSA, TEST scoring, or prior-artifact deletion.

1. **Primary data:** use official release7992926, verify exact1,013,678,473-byte archive against MD5 and SHA, stream only Dataset2/3 into a compact source table. Keep original values/markers and code provenance. Original source479 parents/776,298 rows reproduced. No ProteinGym duplication.
2. **Label semantics:** verify dG/ddG/reference algebra on uncensored substitutions; positive ddG stabilizes in this release. Use author median WT reference and disclose replicate-envelope uncertainty. Do not convert censor/missing symbols into unstable labels or manufacture ddG for saturated parents.
3. **Prospective family/variant protocol:** quality thresholds and effect bins sealed before scoring. Initial30% short-sequence transitive graph produced an uninformative giant component; retain/reject that proxy, predeclare stronger50%/80% links with author/provenance clusters before any fold. This is a data-method correction, not post-audit reward tuning. Four distinct operational families reserved, two development/two audit; exact qualified TRAIN native maps and public monomer metadata confirmed. Family/AF-training overlap limitations disclosed.
4. **MSA:** no exact saved MSA matches. Do not borrow unrelated alignments or silently edit family backgrounds. Use identical query-only protocol as a limited-information feasibility gate; no external query was authorized. All12 inputs/labels were frozen but only two WT calls executed.
5. **Stop:** THO1 mapped recovery0.407/pTM0.36 and villin0.475/pTM0.38 trigger the previously frozen both-WT-below0.5 rule. Stop at two, no replacements and no mutant/audit calls. OutcomeD is insufficient inference evidence, not experimental proof that AF stability ranking fails. Preserve unscored results and unchanged ranking choices.
6. **Next request:** document exact4 public WT-core MSA queries (or separately12 query-specific sequences),12 local folds/25 minutes and explicit background semantics. No requests or further workers run. Production reward remains NO-GO; future work needs experimental ranking and transfer evidence, not additional J1 training.
''')
 write('CHANGELOG.md',f'''# R1d changes

- Source `{source}`: official source retrieval/checksums, stream audit, exact mutation/reference filters, operational family graph and contamination audit, frozen4-parent/12-sequence benchmark, query-only oracle and prospective stop.
- Author notebooks pinned to `{metadata['sha']}`; no experimental refitting or pretrained-model changes.
- Exact ZIP member paths exclude macOS resource forks. Rejected weak family proxy preserved; stronger-link correction completed before inference.
- Two real WT predictions saved; adequacy stop preserved. Ranking/audit status explicitly not evaluated. All source labels and requested follow-up sequences recoverable.
''')
 write('FINAL_E016_R1D_EXPERIMENTAL_STABILITY_REPORT.md',f'''# R1d: experimental benchmark prepared; stability reward remains untested and NO-GO

Obtained and verified the original Tsuboyama release:776,298 measurements across331 natural and148 designed parent records. Official archive MD5/SHA and full processed-member CRC/SHA checked. Preserved original raw values, masks, censor markers, WT references and posterior intervals. No full CSV extraction, ProteinGym substitution, neural reward fitting or experimental reprocessing.

The extra study filters retain {inv['qualified_single_rows']:,} single-substitution rows / {inv['qualified_unique_sequences']:,} unique variant sequences across {len(qualified)} parents and {qfamilies} operational family groups. Verified ΔΔG sign from the actual values: `ddG_ML=dG_ML−author WT median`, maximum algebraic error1.78e-15kcal/mol. Larger positive values are more stabilizing in this dataset. Four family-separated operational controls and12 sequences are frozen, with two independent audit groups reserved; remote homology remains a disclosed limitation.

{table}

Labels are in kcal/mol. WT uses the authors' reference median, with qualified-replicate interval envelopes disclosed. Exact core sequences, original assay flanks, substitutions, row IDs and native-coordinate mappings are preserved. All selected parent constructs occur in qualified E016 TRAIN; they are not J1-unseen proteins. Their experimental audit role is separate from E016's original split. No TEST labels or predictions were used.

**Only two real AlphaFold calls executed.** There is no compatible saved MSA for these parents. Matched query-only inputs were available, but both development WT calls failed the prospectively frozen native-adequacy check:

{ft}

Inference stopped immediately after that gate, leaving ten calls unused. No mutants and no audit sequences were folded. Consequently pLDDT/pTM/PAE stability ranking, mutation sensitivity, confident-destabilized error rates and independent reward-audit success are **not evaluated**. This result does not prove intrinsic AF mutation insensitivity. It establishes the current evidence/protocol blocker and preserves a ready experimental benchmark rather than manufacturing a comparison.

**Decision: OutcomeD; production sequence-stability reward and RL remain NO-GO.** AlphaFold retains its earlier MSA-supported role as a conditional structural evaluator/auxiliary signal. Native recovery, stability ranking and novel-sequence transfer remain separate. R1b's MAE0.358 and R1c's amyloid/depth results are unchanged.

Next bounded request:4 serial public WT-core MSA queries and12 paired fixed-background local predictions /25 worker minutes, explicitly distinguishing the family-background assumption from genuine query-specific novel inference. Alternatively,12 genuine query-specific requests are fully listed for separate approval. None was sent. A positive12-prediction study would still require a larger prospective family audit and64–128/shallow-MSA transfer evidence before policy updates.

Resources: {len(fold['rows'])} successful folds, {seconds:.2f}s charged worker time /1500s cap; sampled AF peak {peak:.0f}MiB. Scripted download/metadata transfer {network_bytes/2**20:.2f}MiB, zero MSA submissions. D: free {free/2**30:.2f}GiB;10GiB reserve retained. Frozen J1 SHA unchanged: `{expected}`. R1c's {len(old['evidence_files'])} sealed evidence files verify unchanged. Source `{source}`. All workers stopped; no J1 updates, RL or further oracle calls.

Details: [data audit](EXPERIMENTAL_STABILITY_DATA_AUDIT.md), [semantics](STABILITY_LABEL_SEMANTICS.md), [selection](VARIANT_SELECTION_PROTOCOL.md), [benchmark](STABILITY_VARIANT_BENCHMARK.json), [family audit](FAMILY_SPLIT_AND_CONTAMINATION_AUDIT.json), [MSA comparability](MSA_COMPARABILITY_AUDIT.md), [rank status](STABILITY_RANKING_RESULTS.json), [mutation sensitivity](AF2_MUTATION_SENSITIVITY_REPORT.md), [audit](INDEPENDENT_AUDIT_RESULTS.json), [next decision](NEXT_E016_RL_DECISION.md), [proposed queries](PROPOSED_MSA_QUERY_REQUEST.json), [resources](RESOURCE_AND_NETWORK_USAGE.json).

Primary sources: [Tsuboyama et al.2023](https://www.nature.com/articles/s41586-023-06328-6), [Zenodo7992926](https://zenodo.org/records/7992926), [author processing code](https://github.com/Rocklin-Lab/cdna-display-proteolysis-pipeline).
''')
 print(json.dumps({'report':str(R/'FINAL_E016_R1D_EXPERIMENTAL_STABILITY_REPORT.md'),'outcome':ranking['outcome'],'calls':len(fold['rows']),'worker_seconds':seconds,'free_GiB':free/2**30,'source':source}),flush=True)
if __name__=='__main__':main()
