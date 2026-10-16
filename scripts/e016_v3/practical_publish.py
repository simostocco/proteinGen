"""Publish only a reviewed-scope diagnostic proposal, never a certified trainer input."""
import datetime,platform,resource,subprocess
from practical_split import *

PARENT_MANIFEST=OLD/'release/unified_pdb_v3_v1/RELEASE_MANIFEST.json'

def manifest(root):
 for n in ['SOURCE_INTEGRITY_CHECK.json','CROSS_SPLIT_AUDIT.json','INDEPENDENT_SPOTCHECK.json']:
  assert json.load(open(root/n))['status']=='PASS',n
 inputs=json.load(open(root/'POLICY_FROZEN.json'))['input_hashes']
 for p,h in inputs.items():assert sha(p)==h,'immutable input changed: '+p
 names=['SEQUENCE_ASSIGNMENTS.parquet','MATCHED_JOINT_SEQUENCE_VIEW.parquet','JOINT_OBSERVATION_ASSIGNMENTS.parquet','HELDOUT_NEAREST_DETECTED_TRAIN.parquet','SHORT_MATCH_AMBIGUITIES.parquet','POLICY_FROZEN.json','SOURCE_INTEGRITY_CHECK.json','CROSS_SPLIT_AUDIT.json','INDEPENDENT_SPOTCHECK.json']
 names += [n for n in ['SPLIT_COUNTS.json','LOADER_INTEGRATION_TEST.json','UNIT_TEST_RESULTS.json','DETERMINISM_CHECK.json','SHORT_MATCH_AMBIGUITY_COUNTS.json','EQUAL_LENGTH_DIRECTION_CHECK.json','CONTEXT_AND_EXCLUSIONS.json','SUPERVISED_VIEWS.json','SPLIT_POLICY.md'] if (root/n).exists()]
 save(root/'PROPOSAL_MANIFEST.json',{'dataset_version':root.name,'status':'PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING','training_authorized':False,'artifact_root':str(root),'artifacts':{n:sha(root/n) for n in names},'coordinate_provenance_manifest':{'path':str(PARENT_MANIFEST),'sha256':sha(PARENT_MANIFEST)},'input_table_manifest':{'path':str(INTAKE/'INTAKE_MANIFEST.json'),'sha256':sha(INTAKE/'INTAKE_MANIFEST.json')},'claim':'No detected cross-split protected matches under the frozen MMseqs retrieval + primary50/both80 + domain70/shorter80/minspan50 screening; not universal homology independence','source_context_embargo':False,'AF_calibration_reservation':False,'geometry_version':'ca_internal_signed_v1','immutable_inputs_rechecked':True})


def extra_stats(root):
 table=pq.read_table(root/'SEQUENCE_ASSIGNMENTS.parquet',columns=['sequence_id','split','canonical_full_sequence']);rows=table.to_pylist();assignment={r['sequence_id']:r['split'] for r in rows};context=collections.defaultdict(set);entity_entries=set();nonprimary=collections.Counter();entityeligible=0
 for batch in pq.ParquetFile(INTAKE/'polymer_entities.parquet').iter_batches(batch_size=2048,columns=['sequence_hash','entry_id','lm_eligible'],use_threads=False):
  for e in batch.to_pylist():
   if e['lm_eligible']:context[e['entry_id']].add(assignment[e['sequence_hash']]);entity_entries.add(e['entry_id']);entityeligible+=1
 for batch in pq.ParquetFile(INTAKE/'chain_observations.parquet').iter_batches(batch_size=2048,columns=['BC_eligible','BC_reason','mapping_status'],use_threads=False):
  for o in batch.to_pylist():
   if not o['BC_eligible']:nonprimary[str(o['BC_reason'])]+=1
 shared=collections.Counter();examples=[]
 for entry,sp in context.items():
  if len(sp)>1:
   if len(examples)<30:examples.append({'PDB_entry':entry,'partitions':sorted(sp)})
   for a,b in [('TRAIN','VALIDATION'),('TRAIN','TEST'),('VALIDATION','TEST')]:
    if {a,b}<=sp:shared[a+'--'+b]+=1
 tag='SSHHHHHHSSGLVPRGSHMASM';tagcounts=collections.Counter(r['split'] for r in rows if tag in r['canonical_full_sequence'])
 out={'eligible_entity_occurrences':entityeligible,'LM_contributing_PDB_entries':len(entity_entries),'LM_shared_source_PDB_entries':sum(len(x)>1 for x in context.values()),'shared_source_entry_counts_by_partition_pair':dict(shared),'shared_entry_examples':examples,'nonprimary_observation_rows':sum(nonprimary.values()),'nonprimary_observation_reasons':dict(nonprimary),'known_historical_construct_tag':tag,'eligible_full_sequences_containing_this_exact_tag_by_split':dict(tagcounts),'interpretation':'Source/context and short-construct overlap are reported rather than automatically closed transitively; the tag count is not an exhaustive construct annotation.'};save(root/'CONTEXT_AND_EXCLUSIONS.json',out);return out


def reports(root):
 c=json.load(open(root/'SPLIT_COUNTS.json'));a=json.load(open(root/'CROSS_SPLIT_AUDIT.json'));l=json.load(open(root/'LOADER_INTEGRATION_TEST.json'));i=json.load(open(root/'INDEPENDENT_SPOTCHECK.json'));src=json.load(open(root/'SOURCE_INTEGRITY_CHECK.json'));ctx=extra_stats(root)
 assert l['status']=='PASS' and i['status']=='PASS'
 def write(n,s):(root/n).write_text(s.strip()+'\n')
 table='| Partition | Unique full sequences | Joint identities | Joint observations | Joint protected groups |\n|---|---:|---:|---:|---:|\n'
 for s in SPLITS:
  x=c['partitions'][s];table+=f"| {s} | {x['sequence_identities']:,} | {x['joint_identities']:,} | {x['joint_observations']:,} | {x['joint_protected_groups']:,} |\n"
 length='| Split | Length | LM identities | Joint identities | Joint observations | Joint groups |\n|---|---|---:|---:|---:|---:|\n'
 for s in SPLITS:
  for b,x in c['partitions'][s]['length_strata'].items():length+=f"| {s} | {b} | {x['sequences']} | {x['joint_identities']} | {x['joint_observations']} | {x['joint_protected_groups']} |\n"
 write('JOINT_BC_ELIGIBILITY.md',f'''# Joint BC proposal eligibility

VERIFIED: {c['qualified_full_chain_observations']:,} primary complete full-chain observations / {c['joint_sequence_identities']:,} exact sequence identities, inherited from the hash-verified intake. No new structural filters or coordinates fabricated.

{table}
{length}

Complete deposited constructs only; every full-polymer Cα must be mapped and present, with verified peptide continuity under the existing source policy. X-ray <=3 Å, EM <=4 Å, NMR identified separately. Missing/incomplete/ambiguous observations remain outside this primary view. Existing resolution/method/model/altloc/missing/sidecar correspondence metadata remain linked. Natural gene termini and independent stability are UNKNOWN. BOS denotes deposited construct start, including any construct tags; observed fragments are not turned into complete sequences.

SEQUENCE_ASSIGNMENTS is uniform by exact full string (one vote). JOINT_OBSERVATION_ASSIGNMENTS has observation_multiplicity, uniform_identity_observation_weight=1/multiplicity, and uniform_observation_weight=1.0. No historical E006 weights are reused. Sample TRAIN identity uniformly, then one of its eligible observations uniformly for the primary matched comparison. Observation-uniform weighting is a distinct declared option.

MATCHED_JOINT_SEQUENCE_VIEW is the exact common identity subset for sequence-only versus joint comparison, across all three identical underlying splits. It contains {c['joint_sequence_identities']:,} identities. The broader LM population is a separate data-budget experiment.

VERIFIED loader checks: {l['observations']} real observations, {l['prefix_checks']} literal-prefix checks across every split/length band; max canonical replay error {l['roundtrip_max_absolute_error_A']:.3g} Å; {l['coordinate_sources_fresh_hash_checked']} linked source files freshly hash-checked. Existing canonical gauge and all 14 approved features reused. Source-ID independence, native suffix swaps to invalid coordinates, bootstrap masks and observation ownership tested. Test coverage is bounded, not a second entire-archive atomic re-audit.

BLOCKED: production optimizer admission remains unchanged and requires scientific review/promotion plus a separately approved experiment. The read-only PDBSplitProposal adapter intentionally rejects training_population(). No prior fitting or model training executed.''')
 write('SIMILARITY_AUDIT.md',f'''# Cross-split similarity audit

Status: PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING.

VERIFIED: {a['all_queries']:,} eligible full sequences were queried directly against all eligible sequences in the other partitions; all TRAIN/VALIDATION/TEST pairs covered. Exact complete-string overlap=0; protected group overlap=0; observation ownership violations={a['observation_ownership_violations']}; detected protected-match violations={len(a['protected_violation_examples'])}. No threshold changes to improve allocation. No calibration component reservations.

Grouping starts from fresh sequence-only disjoint sets. Only primary50/both80 or domain70/shorter80/minspan50 accepted matches join identities. Same-entry, assemblies, short tags, unrelated shadows, observed fragment co-occurrence, and prior parent graph membership do not join them. Already joined query/target pairs may be skipped during grouping: additional edges cannot alter that component. This is not used to skip cross-split verification.

Primary identity=identical aligned residue pairs/alignment columns, including gaps (MMseqs seq-id-mode0). Coverage=inclusive aligned position span/full deposited length, separately for each sequence; it is not identity or observed-coordinate coverage. Strong domain rule requires >=70% identity, >=80% of the shorter full sequence, and >=50 residue span on both aligned sequences. Below50-residue spans use only the full rule, with short ambiguity inventory.

Retrieval uses the existing full masked k6 sensitivity7.5 MMseqs index, candidate cap equal to full old pool (no recorded truncation). Grouping target/query eligibility excludes all14,292 shadows; the old heterogeneous graph parent array is never used. Native alignment >=50% for new grouping batches (initial coverage pilot >=30% is filtered with the same final predicates), E<=1e9, min length1. The union of >=75% target or >=75% query coverage is an engineering superset of the final80% rules. Both directed alignment orientations are considered where retrieved. Index use remains heuristic.

Cross-split screen uses the independently created reverse-order full query DB and prefilter, newly filters every eligible query/target by their different assigned splits, and performs fresh >=30% identity alignments with both coverage passes. It does not reuse the old forest or representative sequences. The MMseqs alignment and kmer implementation is shared, so this is additional direct verification, not implementation-independent exhaustive search.

{i['traceback_accounting_pairs_checked']} native aligned-string/accounting checks passed. Additional independent scalar local SW (match+2/mismatch-1/gap-2) checked {i['real_cross_split_pairs_checked']} real cross-split pairs, with {len(i['additional_conflicts'])} additional protected conflicts. Different scoring and a small spotcheck do not establish universal homology independence. No alternative alignment package was installed or downloaded.

Secondary directed alignment-row counts (two coverage passes may repeat a pair), with explicit coverage, are in CROSS_SPLIT_AUDIT.json. Held-out-identity incidence denominators are each partition's sequence count. Summary:\n\n```json\n{json.dumps(a['heldout_identities_with_detected_train_similarity'],indent=2)}\n```

HELDOUT_NEAREST_DETECTED_TRAIN.parquet contains all {a['heldout_identities']} held-out identities: {a['with_detected_train_match']} have a detected >=30% match under either75 retrieval. Maximum detected identity is reported with both coverage values, coordinates, E-value and score. NULL means no match detected under this search, never zero similarity or proof of no homology. SHORT_MATCH_AMBIGUITIES.parquet stores {a['short_ambiguity_rows_two_passes_and_directions']} directed/two-pass risk rows; they are not proven unrelated or definitive construct matches.

E-values retain the frozen native target background of162,061 sequences /42,749,953 residues (including ineligible references); only the147,769 eligible identities can participate in protected groups or cross-split pairs. These are protocol-specific significance values.

Residual lower-identity similarity, domains occupying <80% of the shorter protein, short constructs, low-complexity/masked retrieval failures, alternative alignments, remote evolutionary relationships and experimental context remain limitations. 30% identity is diagnostic only, never an implicit new exclusion. No claim of all-remote-homology removal.''')
 write('DATASET_CARD.md',f'''# PDB supervised candidate: {root.name}

PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING. Training is not authorized by this artifact.

Scope: verified existing unified intake from {src['distinct_PDB_entries']:,} distinct historical+approved-priority PDB entries ({src['source_rows']:,} source rows including mapping overlays), not all current worldwide PDB. Priority intake previously2418/2440 DONE with22 resource exclusions; no downloads in this task. {src['eligible_polymer_entity_rows']:,} verified eligible entity occurrences support {c['eligible_unique_sequences']:,} exact complete canonical 20–500 sequences ({src['eligible_sequence_residues']:,} residues). Canonicalization/mapping/revision/source hashes are inherited from the immutable intake, whose exact bytes were checked. No UniRef data or pretrained weights are imported.

{table}

Sources: {INTAKE}; linked full coordinate provenance: {PARENT_MANIFEST}. Existing raw mmCIF archives and atomic coordinates are not copied or reparsed. Original deposited sequence/monomer/mapping evidence stays at its source references. Full canonical construct sequence is distinct from observed coordinates; full natural gene sequence/true biological termini can be UNKNOWN. Standard20/MSE normalization is inherited, not silently revised.

No additional identities or observations were excluded by the new split. {src['noneligible_canonical_unique_sequences']:,} canonical unique strings in intake are outside its LM-eligible population; {src['noneligible_entity_rows']:,} entity occurrences are noneligible. Prior entity quarantines={json.load(open(INTAKE/'INTAKE_MANIFEST.json'))['counts_and_quality']['entity_status']['QUARANTINE']:,}. These overlapping denominators must not be summed. {ctx['nonprimary_observation_rows']:,} intake observation rows are outside the qualified primary joint view, including incomplete, uncertain, noneligible and superseded mapping rows; this does not make them biological negatives. CONTEXT_AND_EXCLUSIONS.json and SOURCE_INTEGRITY_CHECK.json preserve exact reason/event counts and references. No new quarantine labels invented.

Context is recorded, not embargoed: {ctx['LM_shared_source_PDB_entries']:,} source entries with eligible entities occur across partitions. Method incidence counts are multilabel for sequences; observation methods use observation denominators. Historical AF controls are retained in metadata with their new exact split shown in SPLIT_COUNTS.json; none are automatically calibration-reserved. Future independent calibration must disclose training/homology overlap and must not use final TEST for reward tuning.

Biases: experimental PDB coverage, deposited constructs/tags/mutations, repeated experiments, strict full-CA/continuity and resolution selection for joint view, length20–500, frozen source scope and resource exclusions. Repeated observations are not independent protein identities. New partitions do not certify blind evaluation of historical E007/E012/E015 models, which may have trained on these sources. Missing coordinates are not proven IDRs; complete coordinates do not prove an independently stable unique native state. Predictive AF confidence and thermodynamic folding labels are absent.

Intended first experiment: matched identity-uniform sequence-only versus joint BC, then chemistry-on/off, fixed updates/prefix exposures and resources. No sequence-generation, neural training, prior fitting, AF or RL executed here. Scientific review of these pragmatic sequence-separation claims and an explicit supervised compute protocol remain required.''')
 save(root/'SUPERVISED_VIEWS.json',{'version':'practical-PDB-supervised-views-v1','training_authorized':False,'sequence_table':'SEQUENCE_ASSIGNMENTS.parquet','joint_table':'JOINT_OBSERVATION_ASSIGNMENTS.parquet','matched_common_table':'MATCHED_JOINT_SEQUENCE_VIEW.parquet','partitions':list(SPLITS),'default_weighting':'uniform sequence identity; uniform eligible observation conditional on identity','alternative':'uniform qualified observation, explicitly different objective','native_coordinates':'referenced original files; canonical gauge reconstructed only for literal prefixes','calibration_roles':'historical metadata only; no reservation','primary_status':'PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING'})
 write('NEXT_EXECUTION_COMMANDS.md',f'''# Reproduce / inspect (no training)

Source branch pdb-supervised-split-20261016-v1, baseline9ff2fc573715bebf8598ff9b8e11ccbf194d61bc. All shell environments are in run.sh, D-only.

```bash
bash {root}/run.sh freeze
bash {root}/run.sh align
bash {root}/run.sh allocate
bash {root}/run.sh cross
bash {root}/run.sh audit
bash {root}/run.sh independent
bash {root}/run.sh loader
```

Do not rerun the mutating allocation/publication steps inside the immutable published proposal. For independent reproduction, use a new D: root and update the root argument/environment wrapper; preserve the frozen policy and referenced input hashes. Completed alignment/cross batches are journals; incomplete attempts remain unconsumed and a suffixed attempt is created on retry. The read-only loader is protein_joint_sac.data.pdb_split_proposal.PDBSplitProposal; iter_sequences(split,matched_joint=True), iter_joint(split), teacher_prefix(row,t). Training admission is intentionally blocked.

Next approval: accept limited50/both80 and70/shorter80/span50 generalization guarantees, residual30%/context overlap and weighting/common-subset design; then approve a versioned training-admission promotion plus a bounded controlled BC compute protocol. Do not fit priors or run optimizers merely because this proposal loads.''')
 c['status']='PASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING';save(root/'SPLIT_COUNTS.json',c)
 manifest(root)


def resources(root):
 receipts=[]
 for f in root.glob('**/*.DONE.json'):
  if '/source/' not in str(f):
   try:
    d=json.load(open(f))
    if 'command' in d:receipts.append(d)
   except Exception:pass
 stages=[json.load(open(f)) for f in root.glob('**/*_RUN_RESULT.json') if '/source/' not in str(f)];tele=[]
 if (root/'RESOURCE_TELEMETRY.jsonl').exists():
  for line in (root/'RESOURCE_TELEMETRY.jsonl').open():
   try:tele.append(json.loads(line))
   except ValueError:pass
 files=[f for f in root.rglob('*') if f.is_file() and '.git' not in f.parts];logical=sum(f.stat().st_size for f in files)
 report={'D_free_bytes_before_first_work':23301529600,'D_free_bytes_at_report':shutil.disk_usage(root).free,'minimum_sampled_D_free_bytes':min([shutil.disk_usage(root).free]+[d.get('minimum_free_disk',d.get('minimum_free_bytes',2**64)) for d in receipts+stages]+[d['D_free_bytes'] for d in tele]),'minimum_reserve_bytes':10*GIB,'new_root_logical_file_bytes':logical,'CPU_threads_initial':4,'CPU_threads_optimized':12,'native_sampled_process_tree_peak_RSS_bytes':max([d.get('child_peak_RSS_bytes',0) for d in receipts if 'RSS_scope' in d]+[0]),'earlier_wrapper_only_RSS_receipts_not_true_native_peak':True,'observed_pipeline_RSS_peak_bytes':max([d.get('observed_parent_and_descendants_RSS_peak_bytes',0) for d in stages]+[d['parent_and_current_child_RSS_bytes'] for d in tele]),'minimum_sampled_MemAvailable_bytes':min([d.get('minimum_MemAvailable_bytes',2**64) for d in receipts+stages]+[d['MemAvailable'] for d in tele]),'native_command_wall_seconds_sum':sum(d.get('seconds',0) for d in receipts),'wall_time_scope':'sum of successful native command receipts, excludes interrupted unreceipted work; not CPU-seconds or end-to-end wall time','stage_resources':stages,'projected_peak_budget':'SCREENING_BUDGET.json and CROSS_SCREEN_BUDGET.json','first_permissive_pilot':'preserved; rejected due predicted storage, no biological threshold changes','precision':'native MMseqs alignment + integer final predicates; float64 canonical prefix QA','GPU_jobs':0,'training_updates':0,'AF_calls':0,'downloads':0,'deletions':0,'RAM_limit':'actual MemAvailable/OS headroom; no fixed 2GiB guard','disk_caveat':'global free-space changes may include unrelated Windows activity; root file bytes are logical sizes, not claimed exact physical recovery; RSS is sampled process-tree memory, not an unsampled absolute peak','python':platform.python_version(),'numpy':np.__version__,'pyarrow':pa.__version__};save(root/'RESOURCE_REPORT.json',report);return report


def final(root):
 reports(root);r=resources(root);c=json.load(open(root/'SPLIT_COUNTS.json'));l=json.load(open(root/'LOADER_INTEGRATION_TEST.json'));a=json.load(open(root/'CROSS_SPLIT_AUDIT.json'))
 code=root/'source';commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=code,text=True).strip()
 (root/'FINAL_REPORT.md').write_text(f'''# Real PDB supervised split proposal\n\nPASS_PRIMARY_SCREEN / SCIENTIFIC_REVIEW_PENDING. No training authorization or claim of universal homology independence.\n\nSource {branch}, commit {commit}; baseline9ff2fc573715bebf8598ff9b8e11ccbf194d61bc preserved.\n\n'''+ '\n'.join(f"- {s}: {c['partitions'][s]['sequence_identities']:,} unique full sequences; {c['partitions'][s]['joint_identities']:,} joint identities / {c['partitions'][s]['joint_observations']:,} observations." for s in SPLITS)+f'''\n\nAll five length strata and counts: JOINT_BC_ELIGIBILITY.md / SPLIT_COUNTS.json. Primary50% identity with80% both full sequences; contained safeguard70% identity,80% shorter,>=50-residue spans. Identity includes gap columns; inclusive spans/full deposited length define coverage. Thresholds frozen before allocation. Protected violations detected after direct all-identity cross-screen: {len(a['protected_violation_examples'])}; exact duplicates/group/ownership violations:0. No grouping by PDB co-occurrence or shadow/context closure; no AF reservation. Old allocation TRAIN16839 / calibrationVAL130929 / TEST1 and zero ordinary joint held-out observations remains untouched.\n\nNo new split-specific exclusions. Input noneligibility/quarantines and short/domain/context ambiguities are explicitly inventoried, not silently removed. Nearest detected TRAIN similarities and secondary90/70/50/30 diagnostics retain coverage and missing-result semantics. Independent spotchecks are bounded and do not establish all remote-homology separation.\n\nRead-only E016 loader: PASS ({l['observations']} real observations / {l['prefix_checks']} prefixes; canonical roundtrip maxerror {l['roundtrip_max_absolute_error_A']:.3g} Å, all14 original features). Both matched-view models use identical underlying sequence splits. Production BC admission remains review-blocked.\n\nResource report: minimum sampled D free {r['minimum_sampled_D_free_bytes']/GIB:.3f} GiB; new root logical bytes {r['new_root_logical_file_bytes']/GIB:.3f} GiB; observed pipeline RSS {r['observed_pipeline_RSS_peak_bytes']/GIB:.3f} GiB; native commands cumulative elapsed {r['native_command_wall_seconds_sum']:.1f}s (not CPU-seconds). Disk reserve10GiB preserved. First oversized-output pilot and intentional CPU-throughput restart are retained. No GPU/AF/training/prior fitting/download/deletion.\n\nSuitable data proposal for a first controlled supervised comparison under the declared limited generalization claims, subject to scientific review of residual similarity/context, matched-identity weighting and a separate compute/admission approval. It is not yet a CERTIFIED production training release. Historical source/archive/checkpoints and previous candidate splits remain unchanged.\n\nReproduction: run.sh plus NEXT_EXECUTION_COMMANDS.md; input/output hashes PROVENANCE_SHA256.json.\n''')
 paths=[f for f in root.rglob('*') if f.is_file() and '.git' not in f.parts and '/runtime/' not in str(f) and f.name!='PROVENANCE_SHA256.json']
 hashes={str(f.relative_to(root)):sha(f) for f in sorted(paths)}
 save(root/'PROVENANCE_SHA256.json',{'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_branch':branch,'source_commit':commit,'baseline_commit':'9ff2fc573715bebf8598ff9b8e11ccbf194d61bc','files':hashes,'input_hashes':json.load(open(root/'POLICY_FROZEN.json'))['input_hashes'],'external_coordinate_inventory':{'path':str(PARENT_MANIFEST),'sha256':sha(PARENT_MANIFEST)},'self_hash_excluded':True,'runtime_caches_excluded':True,'source_snapshot_for_initial_screen':'SCREEN_SOURCE_SNAPSHOT.py','resource_thread_override':'CPU_THREAD_OVERRIDE.json'})
 print('FINAL_PROPOSAL',commit,flush=True)

if __name__=='__main__':
 import sys
 {'manifest':manifest,'reports':reports,'resources':resources,'final':final}[sys.argv[1]](Path(sys.argv[2]))
