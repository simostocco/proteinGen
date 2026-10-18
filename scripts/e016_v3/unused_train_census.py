"""Read-only TRAIN census. No optimizer, split edits, or archive reparsing."""
import os,json,hashlib,time,collections,gzip,math,resource,shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
D=Path('/mnt/d/Simone/proteinGen_data');R=D/'e016_unused_train_census_20261018_v1';I=D/'e016_gpu_unified_pdb_20261015_v1/release/intake_v1';P=D/'pdb_supervised_split_50id80cov_20261016_v1';Q=D/'e016_partial_supervision_20261016_v1'
start=time.monotonic();free0=shutil.disk_usage(R).free
assert free0>11*2**30
inputs={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,x):(R/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def loadrows(p,cols=None):return pq.read_table(p,columns=cols).to_pylist()
def verified(p,expected=None):
 z=sha(p);inputs[str(p)]=z
 if expected:assert z==expected,(p,z,expected)
 return p
im=json.load(open(verified(I/'INTAKE_MANIFEST.json')))
for n in ['chain_observations.parquet','polymer_entities.parquet','unique_sequences.parquet','raw_sources_manifest.parquet']:
 verified(I/n,im['hashes'][n])
vm=json.load(open(verified(Q/'view/VIEW_FROZEN.json')))
for n in ['view/PARTIAL_OBSERVATIONS.parquet','view/PARTIAL_VIEW_QA.json','view/SOURCE_HASHES.json']:
 expected=vm['artifacts'][n];expected=expected.get('sha256') if isinstance(expected,dict) else expected;verified(Q/n,expected)
verified(P/'SEQUENCE_ASSIGNMENTS.parquet','858fd7b2fa75f2699caadf70966cd49dcb5f06906c187803565ba529148525ad')
ss=loadrows(P/'SEQUENCE_ASSIGNMENTS.parquet');s={r['sequence_id']:r for r in ss};assert len(s)==len(ss)
train={h for h,r in s.items() if r['split']=='TRAIN'};complete={h for h in train if s[h]['joint_eligible']};partial={r['sequence_hash'] for r in loadrows(Q/'view/PARTIAL_OBSERVATIONS.parquet',['sequence_hash','split']) if r['split']=='TRAIN'};union=complete|partial;unused=train-union
assert (len(train),len(complete),len(partial),len(union),len(unused))==(118219,21475,32144,50604,67615)
assert partial<=train
strings=[s[h]['canonical_full_sequence'] for h in train];assert len(set(strings))==len(strings)
assert all(20<=len(s[h]['canonical_full_sequence'])<=500 and s[h]['length']==len(s[h]['canonical_full_sequence']) and set(s[h]['canonical_full_sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') and hashlib.sha256(s[h]['canonical_full_sequence'].encode()).hexdigest()==h for h in train)
save('TRAIN_IDENTITY_SET_RECONCILIATION.json',dict(TRAIN=len(train),complete=len(complete),partial=len(partial),overlap=len(complete&partial),union=len(union),unused=len(unused),exact_string_uniqueness=True,canonical_and_hash_checked=True,split_unchanged=True))
# Reuse the frozen source-matched C-N recovery ledger; never recreate retired search artifacts.
links={};journal_files=sorted((Q/'view/continuity_recovery_journal').glob('*.json'))
for p in journal_files:
 v=json.load(open(p))
 for oid,positions,ref in v.get('links',[]):
  links.setdefault(oid,set()).update(positions)
print('sets reconciled; continuity journal observations',len(links),flush=True)
obs=[];byfile=collections.defaultdict(dict);agg={h:dict(sequence_id=h,length=s[h]['length'],observations=0,mapped_observations=0,complete_CA_observations=0,partial_CA_observations=0,zero_CA_observations=0,max_observed_CA=0,max_coverage=0.,verified_distance_targets=0,verified_angle_targets=0,verified_torsion_targets=0,candidate_distance_targets=0,candidate_angle_targets=0,candidate_torsion_targets=0,AA_missing_CA_targets=0,observed_CA_sum=0,segments_CA_runs=0,longest_CA_run=0,trustworthy_observations=0,policy_ready_observations=0,entity_occurrences=0,verified_entity_occurrences=0) for h in unused}
reasons=collections.Counter();allbc=collections.Counter();methods=collections.Counter();entity_refs=collections.defaultdict(list)
for batch in pq.ParquetFile(I/'polymer_entities.parquet').iter_batches(batch_size=1024,use_threads=False):
 for e in batch.to_pylist():
  h=e['sequence_hash']
  if h not in unused:continue
  a=agg[h];a['entity_occurrences']+=1
  if e['status']=='VERIFIED' and e['lm_eligible']:
   assert e['sequence']==s[h]['canonical_full_sequence'];a['verified_entity_occurrences']+=1
  if len(entity_refs[h])<3:entity_refs[h].append(dict(entity_key=e['entity_key'],entry_id=e['entry_id'],source_sha=e['source_sha'],entity_id=e['entity_id'],status=e['status'],reason=e['reason']))
for batch in pq.ParquetFile(I/'chain_observations.parquet').iter_batches(batch_size=256,use_threads=False):
 for v in batch.to_pylist():
  h=v['sequence_hash']
  if h not in unused:continue
  m=json.loads(v.pop('metadata'));a=agg[h];a['observations']+=1
  assert not v['BC_eligible'],v['observation_id'];allbc[v['BC_reason']]+=1
  method=m.get('experimental_method') or ';'.join(m.get('methods',[]));res=m.get('resolution_angstrom')
  if v['coordinate_kind']=='legacy_cache':quality=method=='SOLUTION NMR' or res is not None and res<=(3 if method=='X-RAY DIFFRACTION' else 4)
  else:quality=m.get('geometry_method_resolution_eligible',False)
  rr=[]
  if v['mapping_status']!='VERIFIED_LABEL_CORRESPONDENCE':rr.append(v['mapping_status'])
  if not quality:rr.append('METHOD_RESOLUTION_INELIGIBLE')
  if v['coordinate_kind']=='legacy_cache' and m.get('chain_mapping_status')!='unambiguous':rr.append('AUTHOR_MAPPING_UNKNOWN')
  if m.get('alternate_selection_ambiguous',False):rr.append('ALTERNATE_CONFORMER_AMBIGUITY')
  r={**v,'method':method,'resolution':res,'quality':bool(quality),'reasons':rr,'seen':None,'continuity':m.get('continuity_mask'),'connectivity_ref':m.get('connectivity_ref'),'missing_CA_count_recorded':m.get('missing_CA_count'),'observed_CA':None,'candidate_d':0,'candidate_a':0,'candidate_t':0,'valid_d':0,'valid_a':0,'valid_t':0,'segments':0,'longest':0,'trustworthy':False,'policy_ready':False}
  k=len(obs);obs.append(r);byfile[v['coordinate_path']][v['coordinate_row']]=k
print('unused observations',len(obs),'coordinate files',len(byfile),flush=True)
# All coordinate access is projection of existing indexed caches, not original PDB parsing.
def arr(x):return json.loads(x) if isinstance(x,str) else x
def evaluate(r,raw):
 h=r['sequence_hash'];N=s[h]['length'];issues=r['reasons'];kind=r['coordinate_kind']
 if kind=='legacy_cache':
  seen=~np.asarray(arr(raw['missing_calpha_mask']),bool);xx=arr(raw['selected_calpha_coordinates'])
  if raw['ambiguous_altloc_position_count'] or raw['deterministic_altloc_tie_count']:issues.append('ALTERNATE_CONFORMER_AMBIGUITY')
  alt={x for x in arr(raw['selected_altlocs']) if x not in [None,'','.','?']}
  if len(alt)>1:issues.append('MULTIPLE_NONBLANK_CONFORMER_LABELS_UNRESOLVED')
  if str(raw['model_id'])!=r['model']:issues.append('SOURCE_MODEL_CONTRADICTION')
  if [int(x) for x in arr(raw['label_sequence_ids'])]!=list(range(1,N+1)) or ''.join(arr(raw['residue_tokens']))!=s[h]['canonical_full_sequence']:issues.append('AUTHORITATIVE_POLYMER_CONTRADICTION')
  keys=list(zip(map(str,arr(raw['auth_sequence_ids'])),arr(raw['insertion_codes'])))
  if len(keys)==N and len({keys[i] for i in np.flatnonzero(seen)})!=int(seen.sum()):issues.append('AMBIGUOUS_AUTHOR_NUMBERING')
 else:
  mapping=raw['residue_mapping'];seen=np.asarray([not z['ca_missing'] for z in mapping],bool);xx=[z['ca'] for z in mapping]
  if [z['label_seq_id'] for z in mapping]!=list(range(1,N+1)) or raw['canonical_full_sequence']!=s[h]['canonical_full_sequence']:issues.append('AUTHORITATIVE_POLYMER_CONTRADICTION')
 if len(seen)!=N or len(xx)!=N or [x is not None for x in xx]!=seen.tolist():issues.append('MASK_COORDINATE_CONTRADICTION');return
 x=np.asarray([z if z is not None else [0,0,0] for z in xx],float)
 if not np.isfinite(x[seen]).all():issues.append('NONFINITE_OBSERVED_CA');return
 r['observed_CA']=int(seen.sum());r['seen']=seen.tolist()
 if bool(r['complete_CA'])!=bool(seen.all()):issues.append('COMPLETE_CA_FLAG_CONTRADICTION')
 runs=[];run=0
 for v in list(seen)+[False]:
  if v:run+=1
  elif run:runs.append(run);run=0
 r['segments']=len(runs);r['longest']=max(runs,default=0)
 local=seen[:-1]&seen[1:];r['candidate_d']=int(local.sum());r['candidate_a']=int((local[:-1]&local[1:]).sum());r['candidate_t']=int((local[:-2]&local[1:-1]&local[2:]).sum())
 mapping_issues=[z for z in issues if z!='METHOD_RESOLUTION_INELIGIBLE']
 r['trustworthy']=not mapping_issues
 cont=r['continuity']
 if cont is None:
  cont=np.zeros(N-1,bool)
  if r['connectivity_ref']:cont[:]=True
  for z in links.get(r['observation_id'],[]):
   if 0<=z<N-1:cont[z]=True
 else:cont=np.asarray(cont,bool)
 if len(cont)!=N-1:issues.append('CONTINUITY_LENGTH_CONTRADICTION');return
 good=local&cont
 if not good.any():issues.append('NO_VERIFIED_CONTINUITY_EVIDENCE')
 else:
  # Same existing strict transformed-action support; finite distances and nondegenerate frames.
  bonds=np.diff(x,axis=0);d=np.linalg.norm(bonds,axis=1);good&=d>1e-10
  aa=good[:-1]&good[1:];tt=good[:-2]&good[1:-1]&good[2:]
  idx=np.flatnonzero(aa)
  if len(idx):
   angle=np.arccos(np.clip(-np.sum(bonds[idx]*bonds[idx+1],axis=1)/(d[idx]*d[idx+1]),-1,1));aa[idx]&=(angle>0)&(angle<np.pi)
  idx=np.flatnonzero(tt)
  if len(idx):
   u=bonds[idx+1]/d[idx+1,None];v=-bonds[idx];v=v-(v*u).sum(1)[:,None]*u;w=bonds[idx+2];w=w-(w*u).sum(1)[:,None]*u;vn=np.linalg.norm(v,axis=1);wn=np.linalg.norm(w,axis=1);ok=(vn>1e-10)&(wn>1e-10)
   vv=v/np.maximum(vn[:,None],1e-100);ww=w/np.maximum(wn[:,None],1e-100);tau=np.arctan2((np.cross(u,vv)*ww).sum(1),(vv*ww).sum(1));tt[idx]&=ok&(np.abs(tau)<np.pi)
  if r['trustworthy']:r['valid_d']=int(good.sum());r['valid_a']=int(aa.sum());r['valid_t']=int(tt.sum())
  if int(good.sum())<int((local&cont).sum()) or int(aa.sum())<int(((local&cont)[:-1]&(local&cont)[1:]).sum()) or int(tt.sum())<int(((local&cont)[:-2]&(local&cont)[1:-1]&(local&cont)[2:]).sum()):issues.append('OBSERVED_GEOMETRY_OUTSIDE_UNCHANGED_ACTION_SUPPORT')
 r['policy_ready']=r['trustworthy'] and r['quality'] and r['valid_d']>0 and 'OBSERVED_GEOMETRY_OUTSIDE_UNCHANGED_ACTION_SUPPORT' not in issues
cols=['selected_calpha_coordinates','missing_calpha_mask','label_sequence_ids','residue_tokens','auth_sequence_ids','insertion_codes','ambiguous_altloc_position_count','deterministic_altloc_tie_count','selected_altlocs','chain_mapping_status','model_id']
for j,(path,desired) in enumerate(sorted(byfile.items())):
 if path.endswith('.parquet'):
  off=0
  for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=cols,use_threads=False):
   for k,raw in enumerate(batch.to_pylist()):
    if off+k in desired:evaluate(obs[desired[off+k]],raw)
   off+=len(batch)
 else:
  v=json.load(gzip.open(path,'rt'))
  for row,k in desired.items():evaluate(obs[k],v['observations'][row])
 if j%10==0:print('projected coordinate files',j+1,'/',len(byfile),'elapsed',round(time.monotonic()-start),flush=True)
# One observation per coordinate row is expected; assert no indexed rows lost.
assert all(r['observed_CA'] is not None for r in obs),'coordinate inspection incomplete'
for r in obs:
 a=agg[r['sequence_hash']];N=a['length'];ca=r['observed_CA'];a['mapped_observations']+=r['mapping_status']=='VERIFIED_LABEL_CORRESPONDENCE';a['complete_CA_observations']+=ca==N;a['partial_CA_observations']+=0<ca<N;a['zero_CA_observations']+=ca==0;a['max_observed_CA']=max(a['max_observed_CA'],ca);a['max_coverage']=max(a['max_coverage'],ca/N)
 if r['trustworthy']:
  a['trustworthy_observations']+=1;a['observed_CA_sum']+=ca;a['AA_missing_CA_targets']+=N-ca;a['segments_CA_runs']+=r['segments'];a['longest_CA_run']=max(a['longest_CA_run'],r['longest'])
  for akey,rkey in [('verified_distance_targets','valid_d'),('verified_angle_targets','valid_a'),('verified_torsion_targets','valid_t'),('candidate_distance_targets','candidate_d'),('candidate_angle_targets','candidate_a'),('candidate_torsion_targets','candidate_t')]:a[akey]+=r[rkey]
 a['policy_ready_observations']+=r['policy_ready']
 for z in set(r['reasons']):reasons[z]+=1
# Exclusive hierarchy: policy-ready omitted > trustworthy complete > trustworthy partial > zero CA > mapping > no link.
for h,a in agg.items():
 if a['observations']==0:cat='A_NO_LINKED_OBSERVATION'
 elif a['policy_ready_observations']:cat='F_POLICY_READY_GEOMETRY_OMITTED'
 elif a['complete_CA_observations'] and a['trustworthy_observations']:cat='E_COMPLETE_CA_OTHER_QUALIFICATION_FAILURE'
 elif a['trustworthy_observations'] and a['max_observed_CA']>0:cat='C_PARTIAL_CA_QUALITY_OR_CONTINUITY_FAILURE'
 elif a['max_observed_CA']==0:cat='B_NO_OBSERVED_CA'
 else:cat='D_MAPPING_OR_CONFORMER_PROBLEM'
 a['category']=cat;a['example_observation_id']=None;a['example_entry_id']=None;a['example_coordinate_ref']=None;a['entity_examples']=json.dumps(entity_refs[h],separators=(',',':'))
# Fix complete-category precedence using SAME observation evidence, not independent maxima.
complete_trust={r['sequence_hash'] for r in obs if r['trustworthy'] and r['observed_CA']==s[r['sequence_hash']]['length']}
for h,a in agg.items():
 if a['category']=='E_COMPLETE_CA_OTHER_QUALIFICATION_FAILURE' and h not in complete_trust:a['category']='C_PARTIAL_CA_QUALITY_OR_CONTINUITY_FAILURE'
byh=collections.defaultdict(list)
for r in obs:byh[r['sequence_hash']].append(r)
for h,a in agg.items():
 if not byh[h]:continue
 r=max(byh[h],key=lambda r:(r['policy_ready'],r['trustworthy'],r['observed_CA']==a['length'],r['observed_CA'],r['valid_d']))
 a['example_observation_id']=r['observation_id'];a['example_entry_id']=r['entry_id'];a['example_coordinate_ref']=r['coordinate_path']+'#/row/'+str(r['coordinate_row']);a['example_exclusion_reasons']=';'.join(sorted(set(r['reasons'])));a['example_BC_reason']=r['BC_reason']
categories=collections.Counter(a['category'] for a in agg.values());assert sum(categories.values())==67615
pq.write_table(pa.Table.from_pylist([agg[h] for h in sorted(agg)]),R/'UNUSED_IDENTITY_CLASSIFICATION.parquet',compression='zstd')
compact=[]
for r in obs:
 compact.append({k:r[k] for k in ['observation_id','sequence_hash','entity_key','source_sha','entry_id','label_chain','auth_chain','model','mapping_status','complete_CA','BC_reason','coordinate_kind','coordinate_path','coordinate_row','method','resolution','quality','observed_CA','candidate_d','candidate_a','candidate_t','valid_d','valid_a','valid_t','segments','longest','trustworthy','policy_ready']}|{'exclusion_reasons':sorted(set(r['reasons'])),'CA_mask_sha256':hashlib.sha256(bytes(r['seen'])).hexdigest()})
pq.write_table(pa.Table.from_pylist(compact),R/'UNUSED_OBSERVATION_EVIDENCE.parquet',compression='zstd')
examples={k:[a for a in agg.values() if a['category']==k][:4] for k in categories}
save('UNUSED_SEQUENCE_CLASSIFICATION.json',dict(total=67615,primary_categories=dict(categories),decision_priority=['F policy-ready omission','E trustworthy complete CA failure','C trustworthy nonzero partial failure','B zero CA','D mapping/conformer','A no link'],examples=examples))
save('EXCLUSION_REASONS_BY_OBSERVATION.json',dict(observations=len(obs),BC_reason=dict(allbc),multilabel_reasons=dict(reasons),denominator='observations linked to unused TRAIN identities; reasons overlap',reason_identity_counts={k:len({r['sequence_hash'] for r in obs if k in r['reasons']}) for k in reasons}))
fields=['observations','mapped_observations','complete_CA_observations','partial_CA_observations','zero_CA_observations','trustworthy_observations','observed_CA_sum','AA_missing_CA_targets','segments_CA_runs','verified_distance_targets','verified_angle_targets','verified_torsion_targets','candidate_distance_targets','candidate_angle_targets','candidate_torsion_targets','policy_ready_observations']
summary={k:sum(a[k] for a in agg.values()) for k in fields};summary['CA_runs_are_not_verified_backbone_segments']=True
summary['coverage_max_per_identity_histogram']={str(lo)+'-'+str(hi):sum(lo<=a['max_coverage']<hi for a in agg.values()) for lo,hi in [(0,.25),(.25,.5),(.5,.8),(.8,1),(1,1.01)]}
summary['longest_CA_run_quantiles']=np.quantile([a['longest_CA_run'] for a in agg.values()],[0,.25,.5,.75,.9,.99,1]).tolist()
save('STRUCTURAL_OBSERVATION_COVERAGE.json',summary)
ready=[r for r in obs if r['policy_ready']]
save('POTENTIALLY_RECOVERABLE_GEOMETRY.json',dict(existing_policy_ready_identities=len({r['sequence_hash'] for r in ready}),existing_policy_ready_observations=len(ready),existing_policy_ready_targets={k:sum(r[k] for r in ready) for k in ['valid_d','valid_a','valid_t']},trustworthy_with_verified_local_targets_identities=sum(a['verified_distance_targets']>0 for a in agg.values()),trustworthy_CA_adjacency_candidates_requiring_C_N_evidence_or_quality_review=summary,warning='CA adjacency alone is NOT accepted C-N continuity. No observations admitted or exclusions changed.'))
AA='ACDEFGHIKLMNPQRSTVWY'
def population(ids):
 lengths=np.array([s[h]['length'] for h in ids]);count=collections.Counter(''.join(s[h]['canonical_full_sequence'] for h in ids));total=int(lengths.sum());ip={a:sum(s[h]['canonical_full_sequence'].count(a)/s[h]['length'] for h in ids)/len(ids) for a in AA}
 return dict(identities=len(ids),AA_targets=total,length_quantiles=np.quantile(lengths,[0,.1,.25,.5,.75,.9,1]).tolist(),length_mean=float(lengths.mean()),length_bands={f'{lo}-{hi}':int(((lengths>=lo)&(lengths<=hi)).sum()) for lo,hi in [(20,64),(65,128),(129,256),(257,384),(385,500)]},residue_weighted_composition={a:count[a]/total for a in AA},identity_first_composition=ip,entity_occurrences=sum(s[h]['entity_multiplicity'] for h in ids),experimental_method_identity_multilabel=dict(collections.Counter(m for h in ids for m in s[h]['experimental_methods'])))
save('UNUSED_VS_JOINT_POPULATION_COMPARISON.json',dict(unused=population(unused),joint_union=population(union),all_TRAIN=population(train),note='Exact unique sequences are not independent evolutionary families.'))
save('SEQUENCE_ONLY_TRAINING_ELIGIBILITY.json',dict(unused_identities=67615,AA_ready_identities=67615,AA_targets=sum(s[h]['length'] for h in unused),known_full_canonical_sequence=True,twenty_AA_only=True,length_20_500=True,exact_TRAIN_ownership=True,authoritative_verified_entity_support_identities=sum(a['verified_entity_occurrences']>0 for a in agg.values()),unsupported_fragment_reinterpretation=False,production_sampler_unchanged=True,standalone_AA_loader='Data.target(joint=False) supports all-missing coordinates; PartialData.target currently routes unknown observations to joint=True and needs an explicit new AA-only route before sampling these identities'))
save('CENSUS_EXECUTION.json',dict(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,free_before=free0,free_after=shutil.disk_usage(R).free,optimizer_updates=0,source_files_inspected=len(byfile),continuity_journals=len(journal_files),inputs=inputs))
print(json.dumps(dict(categories=categories,coverage=summary,ready=len(ready),elapsed=time.monotonic()-start)),flush=True)
