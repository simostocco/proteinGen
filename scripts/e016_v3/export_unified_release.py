"""Immutable linked export; certification is contingent on independent measured QA."""
import os,json,sqlite3,hashlib,collections,shutil,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
R=Path(os.environ['E016_ARTIFACT_ROOT']);H=R/'homology';db=sqlite3.connect('file:'+str(R/'tables/unified_v2.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
proposal=Path(json.load(open(H/'LATEST_PROPOSAL.json'))['path']);audit=json.load(open(proposal/'independent/REPORT.json'));qa=json.load(open(R/'mapping/ATOMIC_AND_ACTION_QA.json'))
assert audit['status']=='PASS' and audit['independent_screen_complete'];assert qa['status']=='PASS_COMPLETE_DECLARED_QA'
assert json.load(open(R/'mapping/INTAKE_INTEGRITY_QA.json'))['status']=='PASS'
connectivity=json.load(open(R/'mapping/SEALED_CONNECTIVITY_DEPENDENCIES.json'));assert connectivity['status']=='PASS'
assert all(db.execute('SELECT 1 FROM stages WHERE name=?',(s,)).fetchone() for s in ['historical_complete','intake_complete','atomic_and_action_QA'])
version=1
while (R/f'release/unified_pdb_v3_v{version}').exists():version+=1
D=R/f'release/unified_pdb_v3_v{version}';D.mkdir(exist_ok=False)
intake=R/'release/intake_v1'
assert (intake/'INTAKE_MANIFEST.json').exists(),'sealed provisional intake prerequisite'
def data_rows(name):
 for batch in pq.ParquetFile(intake/(name+'.parquet')).iter_batches(batch_size=1024,use_threads=False):yield from batch.to_pylist()
seqmap={r['sequence_id']:r['canonical_full_sequence'] for r in data_rows('unique_sequences')}
assignments=pq.read_table(proposal/'split_assignments.parquet').to_pylist();byhash={r['sequence_id']:r for r in assignments};counts=collections.Counter()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(name,fields,rows):
 schema=pa.schema([(k,{'str':pa.string(),'int':pa.int64(),'bool':pa.bool_()}[v]) for k,v in fields]);batch=[];n=0
 with pq.ParquetWriter(D/name,schema,compression='zstd') as w:
  for row in rows:
   batch.append(row);n+=1
   if len(batch)>=1024:w.write_table(pa.Table.from_pylist(batch,schema=schema));batch=[]
  if batch:w.write_table(pa.Table.from_pylist(batch,schema=schema))
 counts[name]=n
 if shutil.disk_usage(R).free<12*2**30:raise RuntimeError('BLOCKED_EXPORT_DISK')
def join(seqhash):
 a=byhash.get(seqhash,{});return {k:a.get(k,'METADATA_ONLY') for k in ['homology_group','split','role']}
def srcs():
 for r in data_rows('raw_sources_manifest'):
  x=dict(r);x['metadata_json']=x.pop('metadata');yield x
write('raw_sources_manifest.parquet',[(k,'str') for k in ['source_id','source_sha','entry_id','path','status','metadata_json']],srcs())
def entities():
 for r in data_rows('polymer_entities'):
  x=dict(r);m=json.loads(x.pop('metadata'))
  if 'deposited_lexeme_ref' in m:m['canonical_CIF_lexeme_ref']=m.pop('deposited_lexeme_ref');m['original_deposited_oneletter_string']='UNKNOWN_in_legacy_cache; native_monomers_ref retained'
  x.update(join(x['sequence_hash']));x['lm_eligible']=bool(x['lm_eligible']);x['metadata_json']=json.dumps(m,separators=(',',':'));yield x
write('polymer_entities.parquet',[(k,'str') for k in ['entity_key','source_sha','entry_id','entity_id','sequence_hash','sequence','status','reason']]+[('lm_eligible','bool')]+[(k,'str') for k in ['homology_group','split','role','metadata_json']],entities())
def uniques():
 for r in data_rows('unique_sequences'):
  x={'sequence_hash':r['sequence_id'],'sequence':r['canonical_full_sequence'],'multiplicity':r['entity_multiplicity'],'eligible_occurrences':r['verified_eligible_entity_occurrences']};a=byhash.get(x['sequence_hash']);x['length']=len(x['sequence']);x['lm_eligible']=bool(a and a['lm_eligible']);x['sequence_id']=x['sequence_hash'];x.update(join(x['sequence_hash']));x['uniform_unique_weight']='1 per eligible canonical full identity';x['historical_record_weight']='occurrence links retained; not reused E006 weights';yield x
write('unique_sequences.parquet',[(k,'str') for k in ['sequence_id','sequence_hash','sequence']]+[(k,'int') for k in ['length','multiplicity','eligible_occurrences']]+[('lm_eligible','bool')]+[(k,'str') for k in ['homology_group','split','role','uniform_unique_weight','historical_record_weight']],uniques())
def obs():
 for r in data_rows('chain_observations'):
  x=dict(r);x['metadata_json']=x.pop('metadata');x['complete_CA']=bool(x['complete_CA']);x['BC_eligible']=bool(x['BC_eligible']);x.update(join(x['sequence_hash']));yield x
obsfields=[(k,'str') for k in ['observation_id','entity_key','sequence_hash','source_sha','entry_id','label_chain','auth_chain','model','mapping_status']]+[('complete_CA','bool'),('BC_eligible','bool')]+[(k,'str') for k in ['BC_reason','coordinate_kind','coordinate_path']]+[('coordinate_row','int')]+[(k,'str') for k in ['metadata_json','homology_group','split','role']]
write('chain_observations.parquet',obsfields,obs())
def bc():
 for r in data_rows('chain_observations'):
  x=dict(r);x['sequence']=seqmap.get(x['sequence_hash']);a=join(x['sequence_hash']);yield {k:x[k] for k in ['observation_id','sequence_hash','sequence','source_sha','entry_id','coordinate_kind','coordinate_path','coordinate_row']}|a|{'eligible':bool(x['BC_eligible']),'reason':x['BC_reason'],'full_polymer':bool(x['complete_CA']),'start_semantics':'deposited_construct_start; natural_gene_start_UNKNOWN','fragment_training':'NOT_ENABLED; fragments retained as metadata and leakage safeguards'}
write('joint_bc_eligibility.parquet',[(k,'str') for k in ['observation_id','sequence_hash','sequence','source_sha','entry_id','coordinate_kind','coordinate_path']]+[('coordinate_row','int')]+[(k,'str') for k in ['homology_group','split','role']]+[('eligible','bool'),('reason','str'),('full_polymer','bool'),('start_semantics','str'),('fragment_training','str')],bc())
def events():
 for r in data_rows('exclusion_events'):
  x=dict(r);x['metadata_json']=x.pop('metadata');yield x
write('exclusion_events.parquet',[(k,'str') for k in ['exclusion_id','entry_id','role','reason','metadata_json']],events())
def quarantine():
 for r in data_rows('polymer_entities'):
  if r['status']!='VERIFIED':yield {'identity':r['entity_key'],'entry_id':r['entry_id'],'status':r['status'],'reason':r['reason'],'kind':'ENTITY'}
 for r in data_rows('chain_observations'):
  if r['mapping_status'] not in ['VERIFIED_LABEL_CORRESPONDENCE','SUPERSEDED_MAPPING_VIEW']:yield {'identity':r['observation_id'],'entry_id':r['entry_id'],'status':r['mapping_status'],'reason':r['BC_reason'],'kind':'OBSERVATION'}
write('quarantine.parquet',[(k,'str') for k in ['identity','entry_id','status','reason','kind']],quarantine())
def structure_views():
 for r in data_rows('chain_observations'):
  if r['mapping_status']!='VERIFIED_LABEL_CORRESPONDENCE':continue
  m=json.loads(r['metadata']);yield {'observation_id':r['observation_id'],'sequence_id':r['sequence_hash'],'source_sha256':r['source_sha'],'coordinate_kind':r['coordinate_kind'],'coordinate_path':r['coordinate_path'],'coordinate_row':r['coordinate_row'],'joint_BC_eligible':bool(r['BC_eligible']),'atom_and_residue_masks_ref':m.get('correspondence_ref') or m.get('residue_mapping_ref'),'full_chain_CA':bool(r['complete_CA']),'experimental_disorder_evidence':'UNKNOWN','stability_label':'UNKNOWN','context_label':'conditional_context_or_unknown; experimental source metadata only',**join(r['sequence_hash'])}
write('structure_views.parquet',[(k,'str') for k in ['observation_id','sequence_id','source_sha256','coordinate_kind','coordinate_path']]+[('coordinate_row','int'),('joint_BC_eligible','bool'),('atom_and_residue_masks_ref','str'),('full_chain_CA','bool')]+[(k,'str') for k in ['experimental_disorder_evidence','stability_label','context_label','homology_group','split','role']],structure_views())
write('rejections.parquet',[(k,'str') for k in ['exclusion_id','entry_id','role','reason','metadata_json']],(r for r in events() if r['role'] in ['intake','source','BC_action_support']))
import pickle
pool=pickle.load(open(H/'initial_graph.pkl','rb'))
def edges():
 for n,line in enumerate((H/'exact_source_union_forest.jsonl').open()):
  a,b,kind,detail=json.loads(line);yield {'sequence_id_a':pool['hashes'][a],'sequence_id_b':pool['hashes'][b],'edge_kind':kind,'evidence_ref':str(H/'exact_source_union_forest.jsonl')+'#line/'+str(n),'detail':json.dumps(detail)}
 for n,line in enumerate((H/'full_edges.tsv').open()):
  x=line.split('\t');a,b=int(x[0]),int(x[1]);qc=(int(x[8])-int(x[7])+1)/int(x[11]);tc=(int(x[10])-int(x[9])+1)/int(x[12]);kind='MMSEQS_BIDIR80' if min(qc,tc)>=.8 else 'MMSEQS_SHORTER80' if max(qc,tc)>=.8 else 'MMSEQS_SIGNIFICANT_LOCAL_DOMAIN';yield {'sequence_id_a':pool['hashes'][a],'sequence_id_b':pool['hashes'][b],'edge_kind':kind,'evidence_ref':str(H/'full_edges.tsv')+'#line/'+str(n),'detail':json.dumps({'identity':float(x[2]),'query_coverage':qc,'target_coverage':tc,'Evalue':float(x[6]),'alignment_length':int(x[3])})}
 if (H/'independent_repair_edges.tsv').exists():
  for n,line in enumerate((H/'independent_repair_edges.tsv').open()):
   a,b,kind=line.split()[:3];yield {'sequence_id_a':pool['hashes'][int(a)],'sequence_id_b':pool['hashes'][int(b)],'edge_kind':kind,'evidence_ref':str(H/'independent_repair_edges.tsv')+'#line/'+str(n),'detail':'independent-screen repair witness'}
write('homology_edges.parquet',[(k,'str') for k in ['sequence_id_a','sequence_id_b','edge_kind','evidence_ref','detail']],edges())
for name in ['split_assignments.parquet','homology_groups.parquet']:shutil.copyfile(proposal/name,D/name)
(D/'INDEPENDENT_LEAKAGE_AUDIT.json').write_text(json.dumps(audit,indent=2));(D/'ATOMIC_AND_ACTION_QA.json').write_text(json.dumps(qa,indent=2))
views={};bc_counts=collections.Counter();seq_counts=collections.Counter();full_obs=collections.Counter()
for b in pq.ParquetFile(D/'joint_bc_eligibility.parquet').iter_batches(batch_size=1024):
 for r in b.to_pylist():
  if r['eligible']:
   full_obs[r['role']]+=1
   if r['role']!='NATIVE_AF_CALIBRATION':bc_counts[r['split']]+=1
for r in assignments:
 if r['lm_eligible']:seq_counts[r['split']]+=1
for split in ['TRAIN','VALIDATION','TEST']:
 views[split]={'sequences':{'table':'unique_sequences.parquet','predicate':{'lm_eligible':True,'split':split}},'ca':{'table':'joint_bc_eligibility.parquet','predicate':{'eligible':True,'split':split},'exclude_role':'NATIVE_AF_CALIBRATION'},'bc_actions':{'on_the_fly':'canonical_teacher_prefix','bootstrap':'AA; AA+d; AA+d+theta; AA+d+theta+signed_tau','coordinate_duplication':False}}
(D/'VIEWS.json').write_text(json.dumps(views,indent=2))
(D/'SCHEMA_AND_POLICY.json').write_text(json.dumps({'version':D.name,'scope_policy':json.load(open(R/'RELEASE_SCOPE_AND_POLICY.json')),'coordinate_storage':'references only; original raw atom archives never duplicated','polymer_entities':'one row per source entity; complete canonical identity independently validated from full residue scheme','unique_sequences':'one row per exact canonical full identity, all lengths metadata; LM20–500 approved methods','observations':'chain/model selected CA with source/mapping/masks/altloc refs; superseded views are aliases','unknowns':['natural_gene_termini','thermodynamic_stability','unique_native_state','experimental_disorder_without_external_evidence','additional_methods_and_original_oneletter_native_text_not_retained_in_legacy_cache'],'joint_BC':'full complete mapped chains + C-N continuity + method/resolution + finite invertible actions; fragments not enabled','homology_edges_reference':str(H/'full_edges.tsv'),'initial_exact_source_forest_reference':str(H/'exact_source_union_forest.jsonl'),'priors':'separate TRAIN-only version, never validation/test'},indent=2))
external={r['path']:r['sha256'] for r in connectivity['files']}
cache=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/sequence_data_readiness_raw_full_compact_v1')
for cp in sorted((cache/'partition_commits').glob('part-*.json')):
 c=json.load(open(cp))
 for table in ['source_identity','protein_chain_models']:
  item=c['tables'][table];external[str(cache/item['path'])]=item['sha256']
for r in data_rows('raw_sources_manifest'):
 m=json.loads(r['metadata'])
 if m.get('parsed_path'):external[m['parsed_path']]=m['parsed_sha256']
# Cache tables were freshly hash-verified during intake, and each parsed payload
# during acquisition/linking. The reader rechecks these exact source hashes once
# at certified dataset opening, not at every training transition.
status='CERTIFIED' if all(seq_counts[k]>0 and bc_counts[k]>0 for k in ['TRAIN','VALIDATION','TEST']) else 'BLOCKED_EMPTY_REQUIRED_POPULATION'
manifest={'version':D.name,'status':status,'source_entries':db.execute('SELECT count(DISTINCT entry_id) FROM sources').fetchone()[0],'source_table_rows':counts['raw_sources_manifest.parquet'],'table_rows':dict(counts),'unique_LM_sequences_by_split':dict(seq_counts),'eligible_joint_observations_by_split':dict(bc_counts),'eligible_joint_observations_by_role':dict(full_obs),'source_hash_scope':'new source bytes SHA256 verified; historical sealed raw SHA256 reused after unchanged size/mtime and refreshed complete cache-table SHA validation','independent_screen':'PASS declared local homology/exact/containment/source policy, not all remote homology','external_artifact_sha256':external,'artifact_sha256':{p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file()},'production_training_authorized':False}
(D/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,indent=2));ref={'root':str(D),'version':manifest['version'],'manifest_sha256':sha(D/'RELEASE_MANIFEST.json'),'split_sha256':sha(D/'split_assignments.parquet'),'status':'CERTIFIED' if status=='CERTIFIED' else 'PROVISIONAL'};(R/'DATASET_REF.json').write_text(json.dumps(ref,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k not in ['external_artifact_sha256','artifact_sha256']}),flush=True)
if status!='CERTIFIED':raise RuntimeError(status)
