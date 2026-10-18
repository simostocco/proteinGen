"""Additional read-only provenance checks, bounded raw-source confirmations, report assembly."""
import json,hashlib,collections,concurrent.futures,subprocess,shutil,time,gzip
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import gemmi
D=Path('/mnt/d/Simone/proteinGen_data');R=D/'e016_unused_train_census_20261018_v1';U=D/'e016_gpu_unified_pdb_20261015_v1';I=U/'release/intake_v1';P=D/'pdb_supervised_split_50id80cov_20261016_v1';Q=D/'e016_partial_supervision_20261016_v1';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(R/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def md(n,x):(R/n).write_text(x.rstrip()+'\n')
def load(n):return json.load(open(R/n))
obs=pq.read_table(R/'UNUSED_OBSERVATION_EVIDENCE.parquet').to_pylist();ids=pq.read_table(R/'UNUSED_IDENTITY_CLASSIFICATION.parquet').to_pylist();byh=collections.defaultdict(list)
for r in obs:byh[r['sequence_hash']].append(r)
# v3 resolutions are a LIST named resolutions_angstrom, not the legacy scalar field.
v3groups=collections.defaultdict(list);v3segments=0;v3longest=0
for r in obs:
 if r['coordinate_kind']!='legacy_cache':v3groups[r['coordinate_path']].append(r)
for path,rr in v3groups.items():
 d=json.load(gzip.open(path,'rt'))
 for r in rr:
  o=d['observations'][r['coordinate_row']];res=o.get('resolutions_angstrom',d.get('resolutions_angstrom',[]));r['resolution']=max(res) if res else None
  if r['trustworthy'] and r['valid_d']:
   seen=np.asarray([not z['ca_missing'] for z in o['residue_mapping']]);b=np.asarray(o['continuity_mask'])&seen[:-1]&seen[1:];run=0;rruns=[]
   for z in list(b)+[False]:
    if z:run+=1
    elif run:rruns.append(run+1);run=0
   v3segments+=len(rruns);v3longest=max(v3longest,max(rruns,default=0))
import pyarrow as pa
pq.write_table(pa.Table.from_pylist(obs),R/'UNUSED_OBSERVATION_EVIDENCE.parquet',compression='zstd')
entities={r['entity_key']:r for r in pq.read_table(I/'polymer_entities.parquet',columns=['entity_key','sequence_hash','source_sha','entry_id','status']).to_pylist()};sources={}
for r in pq.read_table(I/'raw_sources_manifest.parquet').to_pylist():sources[(r['source_sha'],r['entry_id'])]=r
for r in obs:
 e=entities[r['entity_key']];assert (e['sequence_hash'],e['source_sha'],e['entry_id'])==(r['sequence_hash'],r['source_sha'],r['entry_id']);assert (r['source_sha'],r['entry_id']) in sources
assert all(a['verified_entity_occurrences']>0 for a in ids)
release=U/'release/unified_pdb_v3_v1/RELEASE_MANIFEST.json';external=json.load(open(release))['external_artifact_sha256'];paths=sorted({r['coordinate_path'] for r in obs});print('hashing existing coordinate caches',len(paths),flush=True)
def check(p):
 h=sha(p);assert h==external[p],('coordinate source hash mismatch',p);return p,h,Path(p).stat().st_size
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:hashes=list(pool.map(check,paths))
# Check the continuity-journal references against preserved sidecar hashes as well.
journals=[];side={};valid_oids={r['observation_id'] for r in obs if r['valid_d']>0}
for p in sorted((Q/'view/continuity_recovery_journal').glob('*.json')):
 v=json.load(open(p));journals.append((str(p),sha(p)))
 for oid,pos,ref in v.get('links',[]):
  if oid in valid_oids:side[ref['path']]=v['source_sha256']
# Sidecars relevant to verified recovered targets only; no search intermediates involved.
for path,expected in side.items():assert sha(path)==expected
# Verify frozen complete-view observation ownership/counts and no arbitrary observation multiplicity.
allseq={r['sequence_id']:r for r in pq.read_table(P/'SEQUENCE_ASSIGNMENTS.parquet').to_pylist()};full=pq.read_table(P/'JOINT_OBSERVATION_ASSIGNMENTS.parquet').to_pylist();trainfull=[r for r in full if r['split']=='TRAIN'];assert len(trainfull)==63315
for r in full:assert allseq[r['sequence_hash']]['split']==r['split'] and r['complete_CA'] and r['eligible'] and not r['fragment']
# Representative raw confirmations are bounded and explicitly not a full archive audit.
select=[]
for predicate in [lambda r:r['trustworthy'] and not r['complete_CA'] and r['resolution'] is None and r['method']=='X-RAY DIFFRACTION',lambda r:r['trustworthy'] and r['complete_CA'] and r['resolution'] is None and r['method']=='X-RAY DIFFRACTION',lambda r:r['trustworthy'] and r['complete_CA'] and r['quality'],lambda r:r['trustworthy'] and not r['complete_CA'] and r['quality'],lambda r:not r['trustworthy']]:
 candidates=[r for r in obs if predicate(r)];chosen=[]
 for r in candidates:
  if r['entry_id'] not in {z['entry_id'] for z in select+chosen}:chosen.append(r)
  if len(chosen)==3:break
 select.extend(chosen)
confirm=[]
for r in select:
 src=sources[(r['source_sha'],r['entry_id'])];path=Path(src['path']);v=dict(observation_id=r['observation_id'],sequence_id=r['sequence_hash'],entry_id=r['entry_id'],source_path=str(path),source_sha256=r['source_sha'],cached_resolution=r['resolution'],cached_observed_CA=r['observed_CA'])
 try:
  assert sha(path)==r['source_sha'];b=gemmi.cif.read_file(str(path)).sole_block();res={tag:list(b.find_values(tag)) for tag in ['_refine.ls_d_res_high','_em_3d_reconstruction.resolution','_reflns.d_resolution_high']};v['raw_resolution_fields']=res
  tags=['label_atom_id','label_asym_id','pdbx_PDB_model_num','label_seq_id'];cols=[list(b.find_values('_atom_site.'+z)) for z in tags];seen=set()
  for atom,chain,model,pos in zip(*cols):
   if atom=='CA' and chain==r['label_chain'] and model==r['model'] and pos not in ['.','?']:seen.add(int(pos))
  v['raw_unique_CA_polymer_positions']=len(seen);v['raw_CA_count_equals_cache']=len(seen)==r['observed_CA'];v['status']='SOURCE_HASH_AND_BOUNDED_RAW_FIELDS_VERIFIED'
 except Exception as ex:v['status']='RAW_CONFIRMATION_UNRESOLVED';v['error']=repr(ex)
 confirm.append(v)
save('BOUNDED_RAW_SOURCE_CONFIRMATION.json',dict(records=confirm,scope='At most 15 selected entries, not an archive-wide resolution/continuity repair; no files modified.'))
# Exact first-failure mechanism for the strongest trustworthy observation of each identity.
def primary_mechanism(r):
 if not r['trustworthy']:return 'MAPPING_OR_CONFORMER_RESTRICTION'
 if r['observed_CA']==0:return 'NO_OBSERVED_CA'
 if not r['quality']:
  if r['method'] not in ['X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR']:return 'EXPERIMENTAL_METHOD_NOT_ADMITTED'
  return 'METHOD_RESOLUTION_METADATA_UNKNOWN' if r['resolution'] is None else 'METHOD_RESOLUTION_OUTSIDE_POLICY'
 if r['complete_CA']:
  return 'COMPLETE_CA_SKIPPED_PARTIAL_VIEW_WITH_VALID_LOCAL_TARGETS' if r['valid_d'] else 'COMPLETE_CA_REQUIRES_FULL_C_N_SIDECAR_PARTIAL_VIEW_SKIPS'
 if not r['valid_d']:return 'PARTIAL_NO_VERIFIED_C_N_CONTINUITY'
 if 'OBSERVED_GEOMETRY_OUTSIDE_UNCHANGED_ACTION_SUPPORT' in r['exclusion_reasons']:return 'UNSUPPORTED_EXISTING_ACTION_DENSITY'
 return 'POLICY_READY_BUT_OMITTED'
mech=collections.Counter();mech_examples={};detail=[]
for a in ids:
 rr=byh[a['sequence_id']]
 if not rr:k='NO_LINK_IN_FROZEN_INVENTORY';best=None
 else:
  # A source with actual safe local targets outranks sources without them, then quality/mapping/coverage.
  best=max(rr,key=lambda r:(r['policy_ready'],r['trustworthy'],r['quality'],r['valid_d']>0,r['complete_CA'],r['observed_CA']));k=primary_mechanism(best)
 mech[k]+=1
 if best and len(mech_examples.setdefault(k,[]))<3:mech_examples[k].append(best)
 detail.append(dict(sequence_id=a['sequence_id'],mechanism=k,observation_id=best['observation_id'] if best else None,BC_reason=best['BC_reason'] if best else None,exclusion_reasons=best['exclusion_reasons'] if best else []))
import pyarrow as pa
pq.write_table(pa.Table.from_pylist(detail),R/'UNUSED_IDENTITY_ADMISSION_DECISIONS.parquet',compression='zstd');save('IDENTITY_EXCLUSION_MECHANISMS.json',dict(counts=dict(mech),examples=mech_examples,rule='strongest policy-ready/trustworthy/quality/local-evidence/complete observation; deterministic inventory order tie-break; no overlap',total=sum(mech.values())))
# Cross-tab first-failure vs complete/partial, and unknown metadata vs actually bad resolution.
save('OBSERVATION_POLICY_CROSSTAB.json',dict(mechanisms=dict(collections.Counter(primary_mechanism(r) for r in obs)),complete_CA_observations=len([r for r in obs if r['complete_CA']]),with_verified_local_targets_by_quality=dict(collections.Counter(str(r['quality']) for r in obs if r['trustworthy'] and r['valid_d'])),method_by_resolution_availability=dict(collections.Counter(r['method']+':'+('UNKNOWN' if r['resolution'] is None else 'RECORDED') for r in obs))))
# Verified segment counts: reconstruct from frozen continuity positions only where sources are trusted.
linksets={}
for p in (Q/'view/continuity_recovery_journal').glob('*.json'):
 for oid,pos,ref in json.load(open(p)).get('links',[]):linksets.setdefault(oid,set()).update(pos)
segments=longest=0;ident_targets={};eligible=set()
for r in obs:
 if r['trustworthy'] and r['valid_d']:
  pos=sorted(linksets.get(r['observation_id'],[]));runs=[];run=0;last=None
  for x in pos:
   if last is None or x==last+1:run+=1
   else:runs.append(run+1);run=1
   last=x
  if run:runs.append(run+1)
  segments+=len(runs);longest=max(longest,max(runs,default=0));ident_targets[r['sequence_hash']]=ident_targets.get(r['sequence_hash'],0)+r['valid_d']
cov=load('STRUCTURAL_OBSERVATION_COVERAGE.json');cov['verified_C_N_segments_with_at_least_one_link_from_recovery_journals']=segments;cov['longest_verified_journal_segment']=longest;cov['verified_C_N_segments_total_with_at_least_one_link']=segments+v3segments;cov['longest_verified_C_N_segment']=max(longest,v3longest);cov['verified_v3_targets_not_in_journals']=sum(r['valid_d'] for r in obs if r['trustworthy'] and r['valid_d'] and r['coordinate_kind']!='legacy_cache');save('STRUCTURAL_OBSERVATION_COVERAGE.json',cov)
# More population-level distributions without materializing any coordinates.
sub={}
for name,rr in [('unused',obs)]:
 sub[name]=dict(distinct_PDB_entries=len({r['entry_id'] for r in rr}),distinct_sources=len({r['source_sha'] for r in rr}),distinct_entity_keys=len({r['entity_key'] for r in rr}),models_multilabel=dict(collections.Counter(r['model'] for r in rr)),observations_by_length_band={f'{lo}-{hi}':sum(lo<=allseq[r['sequence_hash']]['length']<=hi for r in rr) for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]})
comp=load('UNUSED_VS_JOINT_POPULATION_COMPARISON.json');comp['observation_context']=sub;save('UNUSED_VS_JOINT_POPULATION_COMPARISON.json',comp)
source_refs={str(R/'source/scripts/e016_v3'/n):sha(R/'source/scripts/e016_v3'/n) for n in ['build_unified_catalogue.py','finish_unified_intake.py','audit_atomic_identity_and_actions.py','prepare_partial_view.py','j1_sequence_first.py']}
save('CENSUS_SOURCE_INTEGRITY.json',dict(status='PASS',observation_entity_source_ownership_checked=len(obs),complete_TRAIN_observations=len(trainfull),all_unused_have_verified_LM_entity=True,coordinate_artifacts={p:h for p,h,n in hashes},coordinate_bytes_hashed=sum(n for p,h,n in hashes),continuity_journal_hashes=dict(journals),relevant_sidecar_hashes=side,release_manifest_sha256=sha(release),frozen_policy_source=source_refs,retired_prefilters_not_read_or_recreated=True,additional_elapsed_seconds=time.monotonic()-start))
print('post checks done',dict(mech),flush=True)
