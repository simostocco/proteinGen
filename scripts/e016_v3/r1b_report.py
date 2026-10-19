"""Synthesize measured R1b results and seal evidence. No network, folding or training."""
import json,hashlib,subprocess,os,time,sqlite3
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from r1b_common import R,PREV,write,sha,guard,events

def load(n):return json.load(open(R/n))
def git(*args):return subprocess.check_output(['git',*args],cwd=R/'source',text=True).strip()
def boot_interval(values,fn=np.mean,seed=1919):
 a=np.array(values);rng=np.random.default_rng(seed);out=fn(a[rng.integers(0,len(a),(10000,len(a)))],axis=1);return np.quantile(out,[.025,.975]).tolist()
def main():
 guard();cfg=load('configs/MSA_FOLD_FROZEN.json');panel=load('configs/NATIVE_PANEL_32.json');fold=load('MSA_ASSISTED_NATIVE_RECOVERY.json');new=fold['rows'];assert len(new)==10 and all(r['status']=='SUCCEEDED' for r in new);assert fold['budget']['attempts_reserved']==10 and fold['budget']['charged_seconds']<=2400;msa=load('COLABFOLD_MSA_QUERY_MANIFEST.json');assert msa['submissions_including_retries']<=12;match=load('ALPHAFOLD_DB_MATCH_MANIFEST.json');afdb=load('ALPHAFOLD_DB_NATIVE_RECOVERY.json');cal=load('folds/REWARD_REASSESSMENT.json');before=load('folds/TRAIN_CALIBRATION_BEFORE_VALIDATION.json');assert cal['beta_TRAIN_only']==before['beta_TRAIN_only'];critic=load('SAC_CRITIC_COMPARISON.json');old=json.load(open(PREV/'ALPHAFOLD_NATIVE_CONTROL_RESULTS.json'))['rows'];principalids=panel['MSA_first_six'];principal=[r for r in new if r['id'] in principalids];paired=[]
 for r in principal:
  o=next(v for v in old if v['id']==r['id'] and v['status']=='SUCCEEDED');paired.append({'id':r['id'],'N':r['N'],'single_mapped_TM':o['recovery']['robust_fixed_map_TM_similarity_fullN'],'MSA_mapped_TM':r['recovery']['robust_fixed_map_TM_similarity_fullN'],'delta_TM':r['recovery']['robust_fixed_map_TM_similarity_fullN']-o['recovery']['robust_fixed_map_TM_similarity_fullN'],'single_mean_pLDDT':o['features']['mean_pLDDT'],'MSA_mean_pLDDT':r['features']['mean_pLDDT'],'single_pTM':o['features']['ptm'],'MSA_pTM':r['features']['ptm'],'raw_MSA_depth':r['MSA_depth']['raw_depth'],'sampled_Neff80':r['MSA_depth']['effective_depth80_on_max512_coverage50_rows'],'seconds':r['worker_wall_seconds']})
 delta=[r['delta_TM'] for r in paired];train=[r for r in new if r['split']=='TRAIN'];val=[r for r in new if r['split']=='VALIDATION'];nativeGO=cal['TRAIN_median_recovery']>=.6 and cal['TRAIN_fraction_atleast_.50']>=2/3 and np.median([r['recovery']['robust_fixed_map_TM_similarity_fullN'] for r in val])>=.5
 residuals=[]
 for r in new:
  predicted=float(np.clip(cal['beta_TRAIN_only']*r['features']['sequence_confidence_product'],0,1));actual=r['recovery']['robust_fixed_map_TM_similarity_fullN'];role='VALIDATION_AUDIT' if r['split']=='VALIDATION' else 'TRAIN_FIT' if r['sequence_sha256'] in cal['TRAIN_fit_ids'] else 'TRAIN_HELDOUT';residuals.append({'id':r['id'],'role':role,'predicted_recovery':predicted,'measured_recovery':actual,'absolute_error':abs(predicted-actual),'pTM':r['features']['ptm'],'mean_pLDDT':r['features']['mean_pLDDT'],'long_PAE_good_fraction':r['features']['long_range_PAE_good_fraction']})
 write('REWARD_CALIBRATION_RESIDUALS.json',{'beta':cal['beta_TRAIN_only'],'rows':residuals,'fit_did_not_use_VALIDATION':True,'retuning_after_failed_audit':False,'production_weights_sealed':False})
 write('SINGLE_VS_MSA_PAIRED_STATISTICS.json',{'rows':paired,'principal_identity_count':6,'median_single_recovery':float(np.median([r['single_mapped_TM'] for r in paired])),'median_MSA_recovery':float(np.median([r['MSA_mapped_TM'] for r in paired])),'median_paired_improvement':float(np.median(delta)),'paired_identity_bootstrap_median_delta95':boot_interval(delta,np.median),'mean_paired_improvement':float(np.mean(delta)),'paired_identity_bootstrap_mean_delta95':boot_interval(delta),'bootstrap_replicates':10000,'scope':'six curated TRAIN identities; identity resampling does not establish family/population independence'})
 # Offline uncertainty/calibration summaries of fixed held-out synthetic actions.
 for arm in critic['rows']:
  p=np.array(arm['heldout_predictions']);y=np.array(arm['heldout_targets']);arm['heldout_action_bootstrap_RMSE95']=[boot_interval((v-y)**2,np.mean) for v in p];arm['heldout_action_bootstrap_RMSE95']=[[float(np.sqrt(x)) for x in interval] for interval in arm['heldout_action_bootstrap_RMSE95']];arm['calibration_linear_predicted_vs_true']=[{'slope':float(np.polyfit(y,v,1)[0]),'intercept':float(np.polyfit(y,v,1)[1])} for v in p];arm['variance_scope']='two independent Q initializations, one fixed prefix,32 held-out actions; not protein-level uncertainty'
 data=load('critic/FROZEN_DATA.json');u=np.array(data['latent']);recovered=np.column_stack((np.log(np.exp(u[:,0])),np.log((np.pi/(1+np.exp(-u[:,1])))/(np.pi-np.pi/(1+np.exp(-u[:,1])))),np.arctanh(np.tanh(u[:,2]))));critic['direct_features_inverse_action_max_error']=float(np.abs(u-recovered).max());assert critic['direct_features_inverse_action_max_error']<1e-12;write('SAC_CRITIC_COMPARISON.json',critic)
 ev=events();gets=[e for e in ev if e.get('method')=='GET' and e.get('event') is None];mhttp=[e for e in ev if e.get('event')=='MSA_HTTP_COMPLETED'];post=[e for e in ev if e.get('event')=='MSA_HTTP_RESERVED' and e['method']=='POST'];payload=sum(v.get('response_bytes',0) for v in gets+mhttp);assert payload<=750*2**20;assert len(post)==10;peak=max(int(line.split(',')[1]) for r in new for line in r['GPU_samples'] if ',' in line)
 usage={'initial_D_free_bytes':load('configs/RESOURCE_FROZEN.json')['starting_free_bytes'],'final_D_free_bytes':guard(0),'minimum_reserve_bytes':10*2**30,'HTTP_logged_completed_requests':len(gets)+len(mhttp),'HTTP_GET_non_MSA':len(gets),'HTTP_MSA_including_ticket_polls_downloads':len(mhttp),'MSA_sequence_submissions_including_retries':len(post),'MSA_unique_public_sequences':len(msa['rows']),'MSA_max_submissions':12,'AFDB_prediction_API_GETs':sum('/api/prediction/' in e['url'] for e in gets),'AFDB_schema_example_metadata_GETs':1,'SIFTS_mapping_GETs':sum('/api/mappings/' in e['url'] for e in gets),'PDBe_entry_metadata_GETs':sum('/api/pdb/entry/' in e['url'] for e in gets),'AFDB_unique_coordinate_models':match['unique_models_downloaded'],'AFDB_max_unique_models':40,'AFDB_coordinate_file_GETs':sum('/files/' in e['url'] and e['url'].endswith('.pdb') for e in gets),'AFDB_confidence_PAE_GETs':sum('/files/' in e['url'] and e['url'].endswith('.json') for e in gets),'downloaded_decoded_HTTP_payload_bytes':payload,'download_cap_bytes':750*2**20,'upload_sequence_characters':sum(r['N'] for r in new),'upload_exact_wire_bytes':'not captured; recorded public sequences/hashes and requests are authoritative','network_byte_scope':'decoded HTTP response payload, not TLS/wire bytes; requests/redirect receipts saved','browser_documentation_access':'official URLs additionally consulted via web tools; not model or MSA calls, separate from persistent HTTP ledger','local_AF2_attempts':10,'local_AF2_successes':10,'max_AF2_attempts':10,'AF2_charged_worker_seconds':fold['budget']['charged_seconds'],'max_AF2_worker_seconds':2400,'AF2_GPU_sampled_peak_MiB':peak,'AF2_memory_scope':'1Hz nvidia-smi sampled device memory, not guaranteed instantaneous allocator peak','AF2_per_control_seconds':{r['id']:r['worker_wall_seconds'] for r in new},'MSA_search_wall_seconds_total':sum(r.get('wall_seconds',0) for r in msa['rows']),'critic_optimizer_steps':1200,'critic_steps_per_arm':400,'critic_wall_seconds':sum(r['seconds'] for r in critic['rows']),'critic_peak_allocated_bytes':max(r['GPU_peak_allocated_bytes'] for r in critic['rows']),'critic_peak_reserved_bytes':max(r['GPU_peak_reserved_bytes'] for r in critic['rows']),'host_RAM_initial_total_bytes':16699813888,'host_RAM_initial_available_bytes':13574078464,'host_peak_RSS':'not captured for ended AF subprocesses; do not infer from reporting process','J1_optimizer_updates':0,'production_RL_updates':0,'generated_or_private_sequences_submitted':0,'large_database_or_weight_downloads':0,'files_deleted':0,'GPU_workloads_serialized':True,'sufficient_10GiB_reserve_maintained':True};write('RESOURCE_AND_NETWORK_USAGE.json',usage)
 cmp=load('STRUCTURE_ALIGNMENT_METHOD_COMPARISON.json');cmp['additional_protocol_results']={'AFDB':'ALPHAFOLD_DB_NATIVE_RECOVERY.json','MSA':'MSA_ASSISTED_NATIVE_RECOVERY.json'};cmp['missing_AFDB_export_repair']='AFDB_ALIGNMENT_PLACEHOLDER_REPAIR.json';write('STRUCTURE_ALIGNMENT_METHOD_COMPARISON.json',cmp)
 decision={'overall_production_RL_GO':False,'AlphaFold_execution':'GO_FOR_BOUNDED_LOCAL_NATIVE_INFERENCE','native_recovery':'PROMISING_PASSES_ORIGINAL_AGGREGATE_NATIVE_GATES' if nativeGO else 'NO_GO','sequence_reward_calibration':'NO_GO_VALIDATION_MAE_EXCEEDS_.20','novel_sequence_applicability':'UNVERIFIED_NO_SHALLOW_MSA_CERTIFICATION','intended_Transformer_critic':'NO_GO_Q2_RMSE_.113634_EXCEEDS_.100','MLP_critic':'PASS_SYNTHETIC_FIXED_PREFIX_ONLY','real_protein_critic_returns':'NOT_CERTIFIED','evidence_class':'Native recovery promising, reward calibration insufficient','production_J1_updates':0,'recommended_first_policy_method_after_reward_gates':'KL-regularized Monte Carlo sequence policy gradient baseline, separately authorized; do not infer real SAC critic readiness from MLP toy','frozen_native_protocol':'configs/MSA_FOLD_FROZEN.json','frozen_native_protocol_sha256':sha(R/'configs/MSA_FOLD_FROZEN.json'),'reward_weights_sealed':False,'next_bounded_corrections':['Predeclare TRAIN-only confidence calibration treating pTM, regional pLDDT and PAE as separate signals rather than assuming their product is calibrated; fresh held-out audit, no threshold relaxation','Establish low/no-homology and shallow-MSA reward reliability plus anti-exploitation controls before optimizing novel sequences','If SAC remains preferred, certify a critic across held-out prefix states and actual complete-episode returns; otherwise approve a simpler Monte Carlo policy-gradient comparison'],'no_training_automatically_authorized':True};write('RL_READINESS_DECISION.json',decision)
 pairtable='\n'.join(f"| {r['id']} | {r['N']} | {r['single_mapped_TM']:.3f} | {r['MSA_mapped_TM']:.3f} | {r['MSA_mapped_TM']-r['single_mapped_TM']:+.3f} | {r['raw_MSA_depth']:,} | {r['single_pTM']:.2f} → {r['MSA_pTM']:.2f} |" for r in paired)
 fulltable='\n'.join(f"| {r['id']} | {r['split']} | {r['N']} | {r['recovery']['robust_fixed_map_TM_similarity_fullN']:.3f} | {r['features']['mean_pLDDT']:.1f} | {r['features']['ptm']:.2f} | {r['MSA_depth']['raw_depth']:,} | {r['worker_wall_seconds']:.1f} |" for r in new)
 counts=Counter(r['status'] for r in match['rows']);afrows={r['id']:r for r in afdb['rows']};afTable='\n'.join(f"| {r['id']} | {r['entry_id']} | {r['N']} | {r['status']} | {(r.get('selected') or {}).get('modelEntityId','—')} | {afrows[r['id']]['native_recovery']['robust_fixed_map_TM_similarity_fullN']:.3f} |" if r['id'] in afrows and afrows[r['id']]['status']=='SCORED' else f"| {r['id']} | {r['entry_id']} | {r['N']} | {r['status']} | — | — |" for r in match['rows'])
 critTable='\n'.join(f"| {r['arm']} | {r['parameters_per_Q']:,} | {r['RMSE_heldout'][0]:.6f} | {r['RMSE_heldout'][1]:.6f} | {r['seconds']:.2f} | {r['status']} |" for r in critic['rows']);source=git('rev-parse','HEAD');assert not git('status','--porcelain'),'commit source before sealing'
 write('OFFICIAL_ALPHAFOLD_DOCUMENTATION_AUDIT.md',f'''# Official documentation and live interfaces

Accessed 2026-10-19 UTC; original pages and response dates/hashes are in `docs/ACCESS_MANIFEST.json`, `docs/*.http.json`, and the network ledger. Installed local ColabFold remains1.6.3, alphafold-colabfold2.3.20, JAX0.6.2, source revision `efbf31c37cedb38cd09c69c1b991910a9866480e`; no package or parameter replacement.

Sources reviewed: [AFDB home](https://alphafold.ebi.ac.uk/), [requested API documentation](https://alphafold.ebi.ac.uk/api-doc), [current API page](https://alphafold.com/api-docs), [API breaking changes](https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api), [release notes](https://www.ebi.ac.uk/pdbe/news/alphafold-database-release-notes), [FAQ](https://alphafold.ebi.ac.uk/faq), [PDBe](https://www.ebi.ac.uk/pdbe/), [SIFTS](https://www.ebi.ac.uk/pdbe/docs/sifts/quick.html), [ColabFold source](https://github.com/sokrypton/ColabFold), [wiki](https://github.com/sokrypton/ColabFold/wiki), [MSA database history](https://github.com/sokrypton/ColabFold/wiki/MSA-Server-Database-History), and [official AF2 confidence guidance](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/).

The API migration notice specifies25 June2026 as the old-field sunset. Live responses were inspected before retrieval implementation and expose `modelEntityId`, `sequenceStart`, `sequenceEnd`, and `sequence`. Some still include old aliases and `paeImageUrl`, despite the advertised retirement. This client requires the new fields and never depends on legacy aliases or a PAE image. A documentation URL did not expose a usable static schema through the browser; persisted actual responses are authoritative for the parser, not guessed old field names.

Selected records include GDM, NVDA and VADB providers, with different stated pipelines and model intervals, including sequenceStart70. Actual model sequence, coordinate chain and residue numbering are verified. pTM is not advertised in the retrieved DB records and is never invented from average confidence. DB PAE availability is checked per record; the2441-residue parent PAE was intentionally not downloaded under the bounded policy.

The ColabFold documentation permits small serial public-server queries from one client/IP and discourages large-scale use. We made10public native submissions, no generated/private sequences, no parallel MSA clients, no service-limit bypass and no local database installation. The published history names UniRef/ColabFoldDB releases; the actual ticket responses did not independently seal a database build revision. Saved A3Ms and archive hashes pin the inputs regardless of future server changes.

The local pair changes MSA availability/caps(1:1→128:256), keeping model1pTM weights, seed, three recycles, no templates/relaxation, BF16/outputFP32 and GPU/CPU preprocessing behavior. AFDB is a separate, mixed-provider precomputed protocol. It is not a controlled MSA-only comparison; weights, sequence interval/construct, templates, releases and other settings may differ or be unreported. Public DB data uses CC-BY-4.0 with attribution. The existing local parameter archive retains its separately recorded NC-SA notice; these are different artifacts/licenses.
''')
 mapping=load('native/MAPPING_REAUDIT.json');orig=[r for r in mapping['rows'] if r['id'] in [c['id'] for c in panel['controls'][:8]]];ctable='\n'.join(f"| {r['id']} | {r['entry_id']} | {r['auth_chain']}/{r['label_chain']} | {r['observed_CA_count']}/{r['full_deposited_polymer_length']} | {r['PDBe_sequence_agrees']} |" for r in orig)
 write('NATIVE_CONTROL_MAPPING_REAUDIT.md',f'''# Native mapping and structural comparison re-audit

All32canonical sequences/hashes, TRAIN/VALIDATION ownership and current PDBe chain sequences agree. The previous eight controls remain intact. Complete cached source coordinates superpose with the old stored gauge to numerical precision; label positions are1..N and author numbering is retained separately. The two historical CIF sources were selectively rechecked, including actual residue identities, model and alternate-conformer choices. No entire archive was reparsed or rewritten.

| Control | PDB | auth/label chain | Observed/full CA | Current deposited sequence agrees |
|---|---|---|---:|---|
{ctable}

No evidence of a sequence/coordinate mapping error explains the earlier poor recovery. In6gk3, author22–85corresponds to deposited construct positions1–64, which is correct mapping, not a numbering error. 3fm1has319/331observed positions; its missing12tailpositions remain unknown, with separate full-N and observed denominators. Every original mapped score reproduces within1e-10.

Independent proper-rotation Kabsch RMSDs and per-residue deviations, both mapped TM-style denominators, long-range contact Jaccard/distance errors, and installed US-align alignments are saved. US-align Version20261012 binary/source hashes are sealed. The eight original optimized scores remain below.45; optimizing correspondence improves some scores but does not rescue the original single-sequence result. US-align can reassign residue correspondence and reports its own observed-chain denominators; it is never substituted for fixed-map/full-N scores.

The original panel contains heterogeneous contexts: 6gk3is a beta-2-microglobulin amyloid fibril construct; 29txis a mutant viral capsid entry with240-mer assembly; 5it7and7aoiare ribosomal assemblies; 2qmiis octameric, while3fm1is a peroxidase entry containing cofactors. These presences are demonstrated by entry/source metadata. Cofactor or partner dependence of isolated-chain folding is not independently established. Monomer prediction versus an amyloid assembly is a clear comparison-context limitation, not proof that the AF monomer is physically stable or that the native is poor.

The new32panel was chosen by stable SHA ranking across length strata, with24TRAIN and8VALIDATION identities, before new AFDB scores. No easy-protein/recovery-score selection. Exact source metadata, resolution where present, numbering, insertions and selected altlocs are in `native/MAPPING_REAUDIT.json` and `SOURCE_METADATA.json`. Legacy ligand counts can include nucleic-acid residues; component names and assembly metadata are safer than treating those counters as small-molecule counts.

A new audit-only AFDB export bug was corrected: unknown mapped prediction positions were masked in fixed scores but reached temporary US-align files as zero placeholders. Failed exports/results are retained under `alignment/failed_placeholder_export` and `afdb/INITIAL_ALIGNMENT_RESULTS.json`; corrected PDB exports omit missing coordinates. Fixed scores, original R0 results and actual predictions did not change. Tests verify omission and score invariance.

Regional64-residue fixed-window recovery is available. It is a local/domain proxy, not an annotated domain or DSSP assignment. Native all-atom data are not uniformly available in the cache, so no unsupported secondary-structure labels or universal cofactor conclusions are supplied.
''')
 write('ALPHAFOLD_DB_COVERAGE_AND_PROVENANCE.md',f'''# Targeted AlphaFold DB coverage

The predetermined32controls reconcile as:

| Match class | Identities |
|---|---:|
| Exact full queried sequence | {counts['EXACT_FULL_SEQUENCE_MATCH']} |
| Verified mapped fragment/construct | {counts['VALID_MAPPED_FRAGMENT_OR_CONSTRUCT']} |
| Homologous or mutated reference only | {counts['HOMOLOGOUS_REFERENCE_ONLY']} |
| Missing/ambiguous chain mapping | {counts['AMBIGUOUS_MAPPING']} |
| Associated accession endpoint returned no prediction | {counts['NO_PREDICTION_FOUND']} |
| Total |32|

Downloaded21unique coordinate models, below40; no proteome archive. Eight missing chain/SIFTS links are not classified as proof of no AFDB structure. The three explicit accession-not-found results are scoped to the bounded resolver, not an exhaustive homolog search. Associated accessions and API-returned isoforms were inspected; at most two chain-linked accessions per identity were queried. Ties use mapping/sequence coverage and model ID, never observed recovery. A fragment can be compatible with several isoforms without identifying the biological full-length isoform.

All21coordinate files pass actual-chain sequence/interval checks. Fifteen exact-or-construct comparisons are scored;12/15reach mapped similarity>=.50. All3exact full-sequence predictions exceed.95. Qualified TRAIN median={afdb['TRAIN_median_recovery']:.6f}(n12), VALIDATION median={afdb['VALIDATION_median_recovery']:.6f}(n3). TRAIN mapped-region pLDDT/recovery Spearman={afdb['TRAIN_mapped_region_pLDDT_recovery_Spearman']:.6f}; n3VALIDATION correlation is descriptive only. The six inexact/homolog references are retained but excluded from exact/construct primary counts and are not predictions of the queried mutated sequences.

AFDB recovery is generally much stronger than the old single-sequence predictions where comparisons exist. Coverage and context selection differ. The old paired DB subset is small: only the447-residue old control is exact, and the64-residue old control is a verified fragment. The331/256controls have inexact sequence references. Do not attribute DB/local differences exclusively to MSA depth. Providers/pipelines include GDM AlphaFoldMonomer v2.0, NVDA ColabFold1.6.2/AF-PTM and VADB AlphaFold2Monomer v2.3. GDM records are largelyv6 while other providers may be v1. Release labels need not imply newly recomputed coordinates; see the official release notes.

No pTM is fabricated. Full-parent average pLDDT, mapped-region pLDDT, available mapped PAE, coordinate URLs/release IDs, source/provider, licenses and SHA256 are stored per entry. The bound excludes a large parent PAE; missing confidence remains missing. Only3exact examples and no exact VALIDATION predictions are insufficient to certify a sequence reward. Fragment comparison of AFDB's full natural parent does not evaluate a fresh prediction of the isolated deposited construct.

| Control | PDB | N | Match | Model | Mapped score(full query N) |
|---|---|---:|---|---|---:|
{afTable}
''')
 write('SINGLE_SEQUENCE_VS_MSA_COMPARISON.md',f'''# Controlled local single-sequence versus MSA comparison

Same six original TRAIN sequences, model1pTM parameters, seed151502, three recycles, no templates/relaxation, corrected cuda+cpu preprocessing, BF16/outputFP32. The changed input is the saved native-query A3M, with cluster/extra caps128:256 instead of1:1. Raw alignment depth does not mean all rows are retained as model input. Each local call has an immutable protocol/sequence/A3M binding and runtime config.

| Control | N | Single mapped score | MSA mapped score | Gain | Raw MSA depth | pTM(single→MSA) |
|---|---:|---:|---:|---:|---:|---|
{pairtable}

Median recovery increases .223639→.950479. Median paired gain={np.median(delta):.6f}, descriptive paired identity-bootstrap95% interval={boot_interval(delta,np.median)}. Five of six now exceed.90. Both historical hard cases recover strongly. The amyloid fragment remains around.14 despite95.31mean pLDDT and.77pTM. Keep this counterexample; do not classify the native protein as poor or drop it after inspection.

The sealed principal gate permitted exactly four extra public-native controls. The query budget totals10submissions, and local inference totals10calls. Two additional TRAIN controls precede the two original VALIDATION audits. The TRAIN-only fit and hold-out IDs/beta were saved before any VALIDATION fold; the final beta is identical. All calls succeeded; no failed calls replaced.

| Control | Split | N | Mapped score | Mean pLDDT | pTM | Raw MSA depth | Worker s |
|---|---|---:|---:|---:|---:|---:|---:|
{fulltable}

Raw MSA coverage/gaps and deterministic max512-row80%-identity effective-depth estimates are in `MSA_DEPTH_AND_COVERAGE.json`. That bounded Neff estimate is not full-MSA Neff and is not extrapolated. Improvements coincide with informative homology inputs, but n6does not isolate a depth-response law. Context and length confound comparisons, and the amyloid example disproves a universal depth-guarantees-native-recovery rule. The depth1paired controls remain direct evidence that the chosen oracle can behave poorly without homolog information.
''')
 write('ALPHAFOLD_REWARD_REASSESSMENT.md',f'''# Reward reassessment: native recovery promising, calibration insufficient

The earlier single-sequence NO-GO remains unchanged. Under the new MSA protocol, original aggregate native-recovery requirements pass: TRAIN median={cal['TRAIN_median_recovery']:.6f},7/8>=.50, pTM/recovery Spearman={cal['TRAIN_pTM_recovery_Spearman']:.6f}; VALIDATION scores .895361/.536176 give median .715768. This is operational/native-recovery progress, not production reward certification.

Prespecified candidate: clip(beta * mean_pLDDT/100 * pTM * fraction(long-range PAE<10A),0,1), with a TRAIN-only nonnegative scalar. Of eight TRAIN identities sorted by canonical SHA, six fit and two hold out. Beta={cal['beta_TRAIN_only']:.9f} and exact split were frozen before VALIDATION. TRAIN held-out MAE={cal['TRAIN_holdout_MAE']:.6f}passes.15. Independent VALIDATION MAE={cal['VALIDATION_audit_MAE']:.6f}fails.20. No threshold or coefficient was altered after this result. Residuals and all constituent scores remain separate in `REWARD_CALIBRATION_RESIDUALS.json`.

The confidence product is not automatically a calibrated probability of native recovery. PAE and pTM already reflect related global uncertainty; multiplying more factors can underpredict recovery for partially organized/multidomain proteins, while high local confidence can coexist with a different assembly conformation. These are hypotheses supported by the residual pattern, not an experimentally isolated causal proof. The amyloid control remains highly confident but unlike the deposited fibril, and is retained in fitting/analysis.

A bounded future correction could predeclare TRAIN-only calibration with pTM, regional pLDDT and PAE as separate features and then audit different held-out controls. That is a proposal, not a replacement selected against the now-inspected two VALIDATION examples. No new fitting or scoring policy was introduced here. AFDB metrics are not pooled with local predictions, and three exact DB controls without exact validation/available pTM cannot certify this reward either.

The primary reward remains about the completed sequence and AF prediction. J1–AF backbone agreement is supplementary; no J1 coordinates enter the native folding inputs or candidate sequence score. No geometry-agreement multiplier, AlphaFold distillation or J1 training occurred. Confidence is not experimental folding stability, solubility or function.
''')
 write('NOVEL_SEQUENCE_REWARD_LIMITATIONS.md','''# Applicability to novel sequences remains unverified

This task queried only existing public PDB sequences. No J1-generated or private sequence was sent externally. Natural native proteins often have thousands of alignment rows; novel candidates may have shallow or no homologs. The six paired depth1results demonstrate that improved natural-protein recovery cannot be assumed when the alignment information disappears.

Effective depth, coverage, composition/complexity and confidence concentration must be measured per completed sequence. A homology-dependent oracle may reward familiar families or near-duplicates rather than general protein quality. Novelty alone must never be an unlimited reward. Native assemblies, fibrils, isolated domains and full natural sequences can have different conformations; a low native agreement is not a measured biological negative.

AFDB retrieval cannot replace fresh inference for novel sequences and cannot certify models it does not cover. Mapped natural-parent structures are useful controls, not predictions of altered constructs. A public MSA server's small-scale use policy also prevents assuming an unlimited RL oracle; a larger workload requires an approved sustainable service/database strategy, not automatic local database downloads.

Before external-reward policy learning, predeclare a bounded shallow/no-MSA and candidate-distribution study with confidence calibration, duplicate/composition checks, reference-generation/search baselines, independent held-out scoring and fixed query budgets. Any external submission of generated sequences requires separate authorization. Favor sequence reward with independent modest geometry constraints; do not erase a promising sequence because J1's proposed backbone differs from AF.

No numerical certification of novel-sequence rewards, folding uniqueness, stability or biological function follows from these native controls.
''')
 write('SAC_CRITIC_CORRECTION_REPORT.md',f'''# Bounded critic-only correction

The historical twinTransformer result(.120089/.111310RMSE) and analytic hybrid SAC toy remain unchanged. The frozen synthetic benchmark is reconstructed from the same generated prefix(t8,N64), Gaussian seed19019,128candidate actions and known dimensionless return function. TRAIN indices0–95 and held-out96–127 remain fixed; all20AAclasses are represented. This is an immediate known-return regression fixture, not real protein reward or broad Bellman calibration.

The old critic receives candidate geometry indirectly through the appended-coordinate encoder. The one tested correction adds an explicit25-channel action/progress path to the existing critic readout:20-way AAone-hot, three invertibly recovered latent geometry actions and N/progress. Scales use only96TRAIN actions, not targets or VALIDATION. Inverse-action error<{critic['direct_features_inverse_action_max_error']+1e-14:.2e}; no hidden target or native future information is supplied. Zero extra input weights preserve initial predictions within FP32 rounding. J1 architecture and input normalization are unchanged.

One fixed experiment compares original Transformer, direct-action Transformer, and a candidate-action MLP. Each has two independent Qinitializations,400AdamW updates, batch32, LR.002, weight decay.01, clip5, identical minibatches/data and the unchanged RMSE<.100criterion. No architecture/LR sweep. The MLP has less compute, not extra optimizer steps. Both Transformer arms share initial encoder/readout weights where shapes match. The new baseline uses a recorded new initialization; it is not an exact reconstruction of the old RNG stream.

| Arm | Parameters/Q | Q1held-out RMSE | Q2held-out RMSE | Seconds | Result |
|---|---:|---:|---:|---:|---|
{critTable}

The direct-action correction improves one twin, but Q2still misses.100: **intended Transformer SAC critic NO-GO**. Both MLP twins pass this fixture. Training/held-out errors, per-AA errors, prediction correlations, linear calibration, clipping, twin disagreement, bootstrap action-level intervals and memory are in `SAC_CRITIC_COMPARISON.json`. The AA-only TRAIN class-mean baseline has held-out RMSE={critic['target_diagnostics']['AA_only_TRAIN_classmean_heldout_RMSE']:.6f}. Indirect action conditioning is a plausible weakness, but the correction does not establish the unique cause of the old failure.

Only one prefix is held fixed. Synthetic MLP success does not validate unseen state/sequence/length generalization, terminal AF returns, bootstrapping or critic overestimation during real SAC. The critic target is in return units, not physical density/nats; gamma1and terminal/n-step tests pass. Original target-network machinery was not needed to fit a fixed known label and is not claimed independently certified by this comparison.

There were1200critic optimizer steps total across3arms, zero J1/actor/temperature updates. Critic weights were not retained as production artifacts; fixed data, source, seeds/configuration and predictions support reproduction. After reward gates, prefer a simple KL-regularized Monte Carlo sequence-policy-gradient baseline first, with matched oracle budgets. Do not start it here or infer production SAC readiness from the MLP fixture.
''')
 # Final answers remain concise while detailed artifacts retain actual mappings/values.
 write('FINAL_E016_R1B_ALPHAFOLD_RECOVERY_REPORT.md',f'''# E016 R1b: native recovery repaired; production RL remains NO-GO

Executed the32-control PDB/SIFTS/AFDB census, independent alignment re-audit,10serial public-native MSA searches,10real local AF2calls, TRAIN-only calibration with disjoint VALIDATION audit, and one bounded critic comparison. No J1 updates, production RL, TEST evaluation, data deletion, new model-weight/database installation or external generated-sequence submission.

## Native recovery

| Original TRAIN control | N | Single score | MSA score | Gain | MSA rows | pTM(single→MSA) |
|---|---:|---:|---:|---:|---:|---|
{pairtable}

Original TRAIN median improves .224→.950. Five of six MSA predictions exceed.90; both historical hard cases reach .956/.983. Original mapped scores reproduce to1e-10and all independent optimized US-align scores on the old predictions remain below.45. No original mapping error was found. The64-residue amyloid fragment still scores.142 despite95.3pLDDT/.77pTM; assembly/context remains consequential.

The conditional audit reaches8TRAIN and2VALIDATION controls. TRAIN median=.965459; VALIDATION scores=.895361/.536176. Original aggregate native-recovery requirements pass. The TRAIN-fitted confidence product still **fails VALIDATION calibration**: MAE .358252>.20, despite TRAIN hold-out MAE .075058. No post-audit retuning.

## AFDB coverage and critic outcome

| Frozen32-control match class | Count |
|---|---:|
| Exact queried full sequence |3|
| Verified mapped construct/fragment |12|
| Inexact/homologous reference only |6|
| Missing/ambiguous mapping |8|
| No associated-accession prediction found |3|
| Total |32|

Downloaded21unique models. Of15exact/construct comparisons,12reach.50; all3exact predictions exceed.95. Qualified TRAIN median=.953676(n12), VALIDATION=.544018(n3). TRAIN regional-pLDDT/recovery Spearman=.594406. Coverage, constructs and providers differ; AFDB/local differences are not pure MSA effects. Missing SIFTS links are not proof of model absence. No AFDB pTM was invented.

| Critic arm | Parameters/Q | Q1RMSE | Q2RMSE | Seconds | Frozen gate |
|---|---:|---:|---:|---:|---|
{critTable}

The corrected Transformer still fails one twin; the MLP passes only the fixed-prefix synthetic test. No critic is calibrated against real complete-protein returns.

## Direct answers

1. **Earlier mapping/alignment defect?** None found in the original eight. Canonical sequences, label/auth positions, missingness and source gauges are consistent. Biological constructs/assemblies differ from isolated monomer settings.
2. **Optimized alignment changes interpretation?** It improves some values but none of the old eight reaches.45. The original single-sequence failure remains. A new AFDB audit-only placeholder-export bug was corrected, with failed exports retained; fixed scores and original controls were unaffected.
3. **Exact AFDB matches?**3/32. Another12are verified mapped constructs,6only inexact/homolog references,8lack adequate mappings,3associated queries return no prediction.
4. **DB recovery?**12/15comparable exact/construct entries exceed.50,3/3exact exceed.95. Some tiny/assembly-dependent fragments remain poor. This is not32successful exact predictions.
5. **Confidence correlation?** TRAIN regionalpLDDT/recovery rho=.594on12comparable entries; VALIDATIONn3is too small for certification. pTM is unavailable in retrieved records; PAE availability is explicit.
6. **MSA improves local AF2?** Strongly on5/6principal paired identities under otherwise closely matched weights/seed/recycles/templates/precision.
7. **Magnitude/depth?** Median .224→.950; median paired gain={np.median(delta):.6f}, identity-bootstrap95%={boot_interval(delta,np.median)}. MSA rows1440–11588on the six, with128:256retained-input caps. Depth/context/length are confounded; no universal depth-response law is established.
8. **Historical hard cases?**3fm1_A .218→.956 and2qmi_A .262→.983. Their source mappings remain unchanged.
9. **Reward calibration?** TRAIN hold-out passes; locked VALIDATION audit fails(.358>.20). Beta and fit/holdout IDs are identical before/after the audit. Reward remains NO-GO.
10. **Novel sequences?** Unverified. No generated sequence was submitted. Earlier depth1failures and dependence on natural homologs prevent assuming transfer to novel/no-MSA candidates.
11. **Protocol to freeze?** Keep `configs/MSA_FOLD_FROZEN.json` as the native-inference reference, not a production reward certificate. Same model1pTM/3recycles/seed151502/no templates, with persisted A3Ms and128:256caps. All wrappers, parameter and input hashes are sealed.
12. **Requests/calls?**{usage['AFDB_prediction_API_GETs']}AFDB prediction-metadata GETs(including1schema example),{usage['SIFTS_mapping_GETs']}SIFTS and{usage['PDBe_entry_metadata_GETs']}PDBe entry-metadata GETs,21coordinate models,10MSA submissions and10local AF2calls. Total logged HTTP client requests={usage['HTTP_logged_completed_requests']}, including polling/downloads/docs; receipts distinguish them.
13. **Cost?**AF2worker time={usage['AF2_charged_worker_seconds']:.2f}s/{2400}s cap; sampled GPU peak={peak}MiB. Critic wall={usage['critic_wall_seconds']:.2f}s. Downloaded decoded HTTP payload={payload/2**20:.2f}MiB/750MiB; final freeD={usage['final_D_free_bytes']/2**30:.2f}GiB. Full lengths/times/depth/memory and limitations are in the resource report.
14. **Corrected critics pass?**No, intended Transformer twin2=.113634>.100. MLP twins=.057768/.065181pass only the synthetic fixed-prefix scope.
15. **First RL method?**After reward certification and separate approval, prefer a KL-regularized Monte Carlo sequence-policy-gradient baseline. SAC requires real state/return critic certification; do not launch either now.
16. **Exact remaining blocker?**The candidate sequence reward fails independent calibration and is unverified on novel/shallow-MSA sequences; intended Transformer critics also fail their frozen accuracy gate. Native prediction execution itself is now operational and substantially improved.

## Readiness decision

| Gate | Decision |
|---|---|
| AF2 execution |GO for bounded native inference|
| Native structural recovery |Promising; original aggregate panel gates pass|
| Sequence-reward calibration |NO-GO: independent VALIDATION failure|
| Novel-sequence applicability |Unverified; no production certification|
| Transformer SAC critics |NO-GO; MLP synthetic-only pass|
| Production RL |**NO-GO**|

Keep sequence assessment primary and J1 geometry auxiliary. Do not optimize J1–AF agreement as a prerequisite for a promising sequence. No confidence/native-recovery result establishes folding stability or function. Source commit `{source}`. All workers stop after this task.

Detailed reports: [documentation](OFFICIAL_ALPHAFOLD_DOCUMENTATION_AUDIT.md), [mapping/alignment](NATIVE_CONTROL_MAPPING_REAUDIT.md), [AFDB coverage](ALPHAFOLD_DB_COVERAGE_AND_PROVENANCE.md), [paired MSA study](SINGLE_SEQUENCE_VS_MSA_COMPARISON.md), [calibration](ALPHAFOLD_REWARD_REASSESSMENT.md), [novel-sequence limits](NOVEL_SEQUENCE_REWARD_LIMITATIONS.md), [critic correction](SAC_CRITIC_CORRECTION_REPORT.md), [resource ledger](RESOURCE_AND_NETWORK_USAGE.json), [provenance](PROVENANCE_SHA256.json).
''')
 write('EXPERIMENT_DECISIONS.md',f'''# R1b experiment decisions

- Preserve +7500J1(43,000historicalupdates), immutable reference and actor initialization, original R0/R1 NO-GO and old critic RMSEs. No J1 optimizer is constructed/stepped here.
- Consult and persist current official docs/live API responses. Use new fields only; record legacy aliases still returned. Different providers/pipelines and absent pTM are not silently homogenized.
- Freeze24TRAIN/8VALIDATION canonical identities by stable hash/length strata before new AFDB recovery. Include old eight, verify mapping/source coordinates and keep assembly-dependent failures. No TEST or eligibility change.
- Retrieve only linked public entries;21models, bounded to40/750MiB. Refine missing-chain/SIFTS cases as ambiguity, not invented model absence. Initial progress classification is preserved. Exact/construct/homolog classes remain separate.
- Export correct observed residues for independent alignment. Repair new temporary AFDB placeholder-export issue, retain failed exports, verify unchanged fixed scores and omit unknown coordinates in corrected outputs. Original R0 metrics untouched.
- Query only frozen public natives through one serial official client. Preserve A3Ms, server transactions, coverage/Neff estimates. Raw depth is not retained model depth. No large local MSA installation or generated-sequence submission.
- Freeze model1pTM/seed151502/3recycles/no templates, BF16/outputFP32,128:256MSA caps and cuda+cpu preprocessing. Principal paired recovery passes the predeclared conditional gate; use exactly4additional controls.10/10local calls succeed within40min.
- Fit one confidence-product scalar using6TRAIN identities;2TRAIN holdouts; save beta/split before2VALIDATION folds. TRAIN passes, VALIDATION MAE .358fails frozen .20. Do not change formula or thresholds in response. Reward remains NO-GO despite strong native recovery.
- One critic-only explicit-action correction, one MLP baseline, original architecture comparator; matched128actions/400steps/arm, no search. One Transformer twin still fails, both MLP twins pass. No claim of real-return or variable-prefix critic certification.
- Recommend future prospective confidence recalibration with a fresh audit and shallow/no-MSA transfer study. After qualified reward, a separately approved sequence-policy-gradient baseline is simpler than unvalidated SAC. No production updates authorized/executed. Source `{source}`; configs/artifacts hashed in provenance.
''')
 write('CHANGELOG.md',f'''# R1b changes

Source `{source}` adds bounded public HTTP/provenance caching, frozen32-control selection, native-mapping and US-align re-audit, chain-specific SIFTS/AFDB sequence matching, serial official MSA search/auditing, separately versioned local MSA folding, one controlled critic experiment, correctness tests and measured reports.

No production J1 model, architecture, normalization, accepted dataset split, original checkpoints or historical reports changed. Only the new critic diagnostic includes TRAIN-fitted direct-action scaling. Preserve the original negative results and all new failed export/debug evidence. No deleted artifacts or recreated prefilters.
''')
 # Strengthen evidence with exact protected-input and historical-manifest verification.
 source_checkpoint=R.parent/'e016_j1_bidirectional_multitask_20261019_v1/runs/J1/checkpoint_07500.pt';expected='881457ff0396975c1b20dd0a8e2caac8521e65b7e1cec76a7bb531dbc437b412';assert sha(source_checkpoint)==expected
 old_manifest=json.load(open(PREV/'PROVENANCE_SHA256.json'));verified=0
 for name,h in old_manifest['evidence_files_sha256'].items():assert sha(PREV/name)==h,name;verified+=1
 for name,h in old_manifest['source_files_sha256'].items():assert sha(PREV/'source'/name)==h,name
 split=R.parent/'pdb_supervised_split_50id80cov_20261016_v1/PROPOSAL_MANIFEST.json';assert sha(split)==panel['split_manifest_sha256'];original_params=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools/alphafold_weights/params/params_model_1_ptm.npz');assert sha(original_params)==cfg['weights_sha256']
 for r in msa['rows']:assert sha(r['a3m_path'])==r['a3m_sha256']
 write('PROTECTED_INPUT_VERIFICATION.json',{'status':'PASS','J1_initializer_path':str(source_checkpoint),'J1_initializer_sha256':expected,'historical_R0_evidence_files_verified':verified,'historical_R0_source_files_verified':len(old_manifest['source_files_sha256']),'split_manifest_sha256':sha(split),'AF2_weights_sha256':sha(original_params),'MSA_artifacts_verified':len(msa['rows']),'J1_optimizer_updates':0,'production_RL_updates':0,'no_deleted_files':True,'source_commit':source})
 gpu=subprocess.run(['nvidia-smi'],capture_output=True,text=True);write('FINAL_WORKER_STATUS.json',{'GPU_snapshot':gpu.stdout,'returncode':gpu.returncode,'all_owned_fold_MSA_critic_workers_finished':True,'no_new_J1_updates':True,'source_worktree_clean':True});assert gpu.returncode==0
 tool=Path('/mnt/d/Simone/proteinGen/e015_artifacts/e015_integrated_multiscale_parity_selfconditioned_v1/external_designability_loop_v1/tools/USalign');write('ALIGNER_PROVENANCE.json',{'binary':str(tool/'USalign'),'binary_sha256':sha(tool/'USalign'),'source_cpp_sha256':sha(tool/'source/USalign.cpp'),'version':'20261012, verified from actual stdout','upstream_commit':'not asserted; installed binary/source hashes retained'})
 files={}
 for p in sorted(R.rglob('*')):
  rel=p.relative_to(R)
  if not p.is_file() or any(v in ['source','runtime'] for v in rel.parts) or p.name in ['PROVENANCE_SHA256.json','PROVENANCE_VERIFICATION.json']:continue
  files[str(rel)]=sha(p)
 sourcefiles={str(p.relative_to(R/'source')):sha(p) for p in (R/'source/scripts/e016_v3').glob('r1b_*.py')};write('PROVENANCE_SHA256.json',{'source_commit':source,'source_branch':git('branch','--show-current'),'source_files':sourcefiles,'evidence_files':files,'excluded':'runtime caches,source Git control,self-reference verification; inherited sources pinned by parent def9a0c and final source commit','historical_R0_source_commit':old_manifest['source_commit']});write('PROVENANCE_VERIFICATION.json',{'status':'PASS','files_hashed':len(files),'source_files':len(sourcefiles),'source_commit':source,'R0_original_evidence_unchanged':True,'J1_SHA_verified':expected,'calls':10,'MSA_submissions':10,'AFDB_models':21,'production_RL_GO':False,'D_free_bytes':guard(0)})
 print(json.dumps({'report':'PUBLISHED','source':source,'local_AF2_calls':10,'MSA_submissions':10,'AF2_charged_seconds':usage['AF2_charged_worker_seconds'],'downloaded_MiB':payload/2**20,'D_free_GiB':guard(0)/2**30,'production_RL_GO':False}),flush=True)
if __name__=='__main__':main()
