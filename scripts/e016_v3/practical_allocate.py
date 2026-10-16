"""Deterministic allocation and direct-screen accounting for practical_split."""
import collections,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from practical_split import *

CONTROL=Path('/mnt/d/Simone/proteinGen_data/e016_v3_blocker_resolution_20261014_v1/NATIVE_CONTROL_PANEL.json')

def experimental_methods(metadata):
 """Preserve authoritative scalar/list schema variants; never infer from filenames."""
 scalar=metadata.get('experimental_method',metadata.get('method'))
 array=metadata.get('methods')
 if array is not None:
  if not isinstance(array,list) or not all(isinstance(x,str) and x for x in array):raise ValueError('invalid experimental methods array')
  values=set(array)
  if scalar and scalar not in values:raise ValueError('contradictory scalar/array methods')
 else:
  if scalar is not None and not isinstance(scalar,str):raise ValueError('invalid scalar method')
  values={scalar} if scalar else set()
 return tuple(sorted(values)) or ('UNKNOWN',)

def observations():
 for batch in pq.ParquetFile(INTAKE/'chain_observations.parquet').iter_batches(batch_size=1024,use_threads=False):
  for row in batch.to_pylist():
   if row['BC_eligible']:
    assert row['complete_CA']==1 and row['mapping_status']=='VERIFIED_LABEL_CORRESPONDENCE'
    yield row

def groups_from_evidence(root):
 assert (root/'alignment_coverage_v1/COMPLETE.json').exists()
 p,by,active=load();uf=Components(range(len(active)));counts=collections.Counter()
 for a,b,ni,al,qs,qe,ts,te,ql,tl,ev,bits in edge_rows(root):
  assert active[a] and active[b] and ql==len(p['sequences'][a]) and tl==len(p['sequences'][b])
  full,domain=match(ni,al,qs,qe,ts,te,ql,tl)
  counts['accepted_alignment_rows']+=1;counts['full_rule_rows']+=int(full);counts['domain_rule_rows']+=int(domain)
  if full or domain:counts['unions']+=int(uf.link(a,b))
 groups=collections.defaultdict(list)
 for i in np.flatnonzero(active):groups[uf.find(int(i))].append(int(i))
 return p,by,active,groups,counts

def allocate_vectors(items):
 """Pure allocator: tuples (stable_group_id, feature_vector); no row splitting."""
 items=list(items)
 if not items:raise ValueError('nonempty grouped population required')
 total=sum((v for _,v in items),np.zeros_like(items[0][1],dtype=float));scale=np.maximum(total,1);target=np.array([.8,.1,.1]);current=np.zeros((3,len(total)));out={}
 ordered=sorted(items,key=lambda x:(-float((x[1]/scale).max()),-float(x[1][0]),hashlib.sha256(('42:'+x[0]).encode()).hexdigest()))
 for gid,v in ordered:
  before=(current/scale-target[:,None])**2;after=((current+v)/scale-target[:,None])**2
  choice=int(np.argmin((after-before).sum(axis=1)));current[choice]+=v;out[gid]=SPLITS[choice]
 return out,current

def allocate(root):
 start=time.monotonic();p,by,active,groups,ec=groups_from_evidence(root)
 hs=p['hashes'];obs_counts=collections.Counter();obs_method={};methods=collections.defaultdict(set)
 entity_owner={}
 for batch in pq.ParquetFile(INTAKE/'polymer_entities.parquet').iter_batches(batch_size=2048,use_threads=False):
  for row in batch.to_pylist():
   if row['lm_eligible']:
    assert row['status']=='VERIFIED' and row['sequence_hash'] in by and row['sequence']==by[row['sequence_hash']]['canonical_full_sequence']
    m=json.loads(row['metadata']);methods[row['sequence_hash']].update(experimental_methods(m))
    entity_owner[row['entity_key']]=(row['sequence_hash'],row['source_sha'],row['entry_id'])
 seen=set()
 for o in observations():
  assert o['observation_id'] not in seen;seen.add(o['observation_id'])
  h=o['sequence_hash'];assert h in by and entity_owner[o['entity_key']]==(h,o['source_sha'],o['entry_id'])
  obs_counts[h]+=1;meta=json.loads(o['metadata']);obs_method.setdefault(h,collections.Counter()).update(experimental_methods(meta))
 assert len(seen)==71884 and len(obs_counts)==27371
 # 25 dimensions: seq total+lengths, joint identities total+lengths,
 # joint observations total+lengths, sequence method incidences, observation methods.
 cats=('X-RAY DIFFRACTION','ELECTRON MICROSCOPY','SOLUTION NMR')
 def vector(ids):
  v=np.zeros(24,dtype=np.float64)
  for i in ids:
   h=hs[i];b=band(by[h]['length']);n=obs_counts[h]
   v[0]+=1;v[1+b]+=1
   if n:v[6]+=1;v[7+b]+=1;v[12]+=n;v[13+b]+=n
   for j,m in enumerate(cats):v[18+j]+=int(m in methods[h]);v[21+j]+=obs_method.get(h,{}).get(m,0)
  return v
 ordered=[]
 for ids in groups.values():
  ids.sort(key=lambda i:hs[i]);gid=hs[ids[0]];v=vector(ids);tie=hashlib.sha256(('42:'+gid).encode()).hexdigest();ordered.append((gid,ids,v,tie))
 grouped_assignment,current=allocate_vectors((gid,v) for gid,ids,v,tie in ordered);assignment={};gid_for={}
 for gid,ids,v,tie in ordered:
  for i in ids:assignment[hs[i]]=grouped_assignment[gid];gid_for[hs[i]]=gid
 controls=json.load(open(CONTROL))['controls'];control_hashes=collections.defaultdict(list)
 for c in controls:control_hashes[hashlib.sha256(c['full_deposited_canonical_sequence'].encode()).hexdigest()].append(c['id'])
 output=[]
 for h,r in sorted(by.items()):
  d={**r,'split':assignment[h],'protected_group_id':gid_for[h],'length_band':BANDS[band(r['length'])],'joint_observation_count':obs_counts[h],'joint_eligible':obs_counts[h]>0,'experimental_methods':sorted(methods[h]),'historical_AF_control_ids':control_hashes[h],'sequence_weight':1.0};output.append(d)
 pq.write_table(pa.Table.from_pylist(output),root/'SEQUENCE_ASSIGNMENTS.parquet',compression='zstd')
 pq.write_table(pa.Table.from_pylist([r for r in output if r['joint_eligible']]),root/'MATCHED_JOINT_SEQUENCE_VIEW.parquet',compression='zstd')
 counts={s:{'sequence_identities':0,'joint_identities':0,'joint_observations':0,'protected_groups':0,'length_strata':{b:{'sequences':0,'joint_identities':0,'joint_observations':0} for b in BANDS},'sequence_method_incidences':{},'observation_methods':{}} for s in SPLITS}
 for s in SPLITS:
  subset=[r for r in output if r['split']==s];c=counts[s];c['sequence_identities']=len(subset);c['joint_identities']=sum(r['joint_eligible'] for r in subset);c['joint_observations']=sum(r['joint_observation_count'] for r in subset);c['protected_groups']=len({r['protected_group_id'] for r in subset});c['joint_protected_groups']=len({r['protected_group_id'] for r in subset if r['joint_eligible']});c['achieved_sequence_fraction']=len(subset)/len(output)
  c['sequence_method_incidences']=dict(collections.Counter(m for r in subset for m in r['experimental_methods']))
  for b in BANDS:
   v=[r for r in subset if r['length_band']==b];c['length_strata'][b]={'sequences':len(v),'joint_identities':sum(r['joint_eligible'] for r in v),'joint_observations':sum(r['joint_observation_count'] for r in v),'joint_protected_groups':len({r['protected_group_id'] for r in v if r['joint_eligible']}),'sequence_protected_groups':len({r['protected_group_id'] for r in v})}
 writer=None;buf=[];context=collections.defaultdict(set)
 for o in observations():
  h=o['sequence_hash'];s=assignment[h];m=json.loads(o['metadata']);method_values=experimental_methods(m);c=counts[s]['observation_methods']
  for method in method_values:c[method]=c.get(method,0)+1
  d={**o,'split':s,'protected_group_id':gid_for[h],'eligible':True,'fragment':False,'role':{'TRAIN':'TRAINING','VALIDATION':'DEVELOPMENT','TEST':'FINAL_TEST'}[s],'length':by[h]['length'],'length_band':BANDS[band(by[h]['length'])],'observation_multiplicity':obs_counts[h],'uniform_identity_observation_weight':1.0/obs_counts[h],'uniform_observation_weight':1.0,'geometry_version':'ca_internal_signed_v1','full_label_positions':'1..N; complete mapping referenced by original metadata','experimental_methods':list(method_values),'training_authorized':False};buf.append(d);context[o['entry_id']].add(s)
  if len(buf)>=1024:
   tab=pa.Table.from_pylist(buf)
   if writer is None:writer=pq.ParquetWriter(root/'JOINT_OBSERVATION_ASSIGNMENTS.parquet',tab.schema,compression='zstd')
   writer.write_table(tab);buf=[]
 if buf:writer.write_table(pa.Table.from_pylist(buf,schema=writer.schema))
 writer.close()
 adequate={s:counts[s]['joint_identities']>=100 and counts[s]['joint_protected_groups']>=20 and all(x['joint_identities']>=20 for x in counts[s]['length_strata'].values()) for s in ('VALIDATION','TEST')}
 result={'status':'ALLOCATED_NOT_YET_SCREENED','eligible_unique_sequences':len(output),'qualified_full_chain_observations':len(seen),'joint_sequence_identities':len(obs_counts),'partitions':counts,'protected_group_count':len(groups),'largest_group_sizes':sorted([len(v) for v in groups.values()],reverse=True)[:30],'edge_counts_counting_both_coverage_passes_and_directions':dict(ec),'calibration_reserved':0,'method_count_semantics':'authoritative scalar/list normalization; method incidence per identity/observation, possibly multilabel; no inferred methods','historical_control_ids':{c['id']:{'sequence_hash':hashlib.sha256(c['full_deposited_canonical_sequence'].encode()).hexdigest(),'split':assignment.get(hashlib.sha256(c['full_deposited_canonical_sequence'].encode()).hexdigest(),'NOT_ELIGIBLE')} for c in controls},'ordinary_joint_evaluation_engineering_floor':{'per_heldout_joint_identities':100,'per_heldout_length_band_joint_identities':20,'per_heldout_protected_groups':20,'scientific_statistical_power':'REQUIRES_REVIEW'},'engineering_floor_pass':adequate,'additional_exclusions_by_new_split':0,'old_intake_exclusions':json.load(open(INTAKE/'INTAKE_MANIFEST.json'))['counts_and_quality']['entity_status'],'joint_source_context_shared_entries':sum(len(v)>1 for v in context.values()),'allocation_seconds':time.monotonic()-start,'training_authorized':False}
 save(root/'SPLIT_COUNTS.json',result)
 assert all(adequate.values()),'held-out populations below predeclared engineering sanity floor'
 print(json.dumps({s:counts[s]['sequence_identities'] for s in SPLITS}),flush=True)

def audit(root):
 assert (root/'cross_screen_coverage_v1/COMPLETE.json').exists()
 p,by,active=load();rows=pq.read_table(root/'SEQUENCE_ASSIGNMENTS.parquet').to_pylist();split={r['sequence_id']:r['split'] for r in rows};groups={r['sequence_id']:r['protected_group_id'] for r in rows}
 nearest={h:None for h,s in split.items() if s!='TRAIN'};bins=collections.Counter();violations=[];sample=[];nrows=0;heldout_flags=collections.defaultdict(int);ambiguities=[];ambiguity_writer=None;ambiguity_rows=0
 for a,b,ni,al,qs,qe,ts,te,ql,tl,ev,bits in edge_rows(root,'cross_screen_coverage_v1'):
  ha,hb=p['hashes'][a],p['hashes'][b];sa,sb=split[ha],split[hb];assert sa!=sb;full,domain=match(ni,al,qs,qe,ts,te,ql,tl);nrows+=1
  pair='--'.join(sorted((sa,sb)));qc=(qe-qs+1)/ql;tc=(te-ts+1)/tl;ident=ni/al
  if 10*ni>=7*al and (qc if ql<=tl else tc)>=.8 and min(qe-qs+1,te-ts+1)<50:
   ambiguities.append({'query_sequence_id':ha,'target_sequence_id':hb,'query_split':sa,'target_split':sb,'identity':ident,'query_coverage':qc,'target_coverage':tc,'query_length':ql,'target_length':tl,'query_span':qe-qs+1,'target_span':te-ts+1,'alignment_columns':al,'evalue':ev,'status':'SHORT_CONTAINMENT_AMBIGUITY_NOT_PROTECTED_BY_FROZEN_POLICY'});ambiguity_rows+=1
   if len(ambiguities)>=1024:
    tab=pa.Table.from_pylist(ambiguities)
    if ambiguity_writer is None:ambiguity_writer=pq.ParquetWriter(root/'SHORT_MATCH_AMBIGUITIES.parquet',tab.schema,compression='zstd')
    ambiguity_writer.write_table(tab);ambiguities=[]
  if full or domain:
   if len(violations)<100:violations.append({'query':ha,'target':hb,'identity':ident,'qcov':qc,'tcov':tc,'full':full,'domain':domain})
  for cutoff in (90,70,50,30):
   if 100*ni>=cutoff*al:
    bins[f'{pair}:identity>={cutoff}:either_coverage>=75_retrieval']+=1
    if qc>=.8 and tc>=.8:bins[f'{pair}:identity>={cutoff}:both80']+=1
    if (qc if ql<=tl else tc)>=.8:bins[f'{pair}:identity>={cutoff}:shorter80']+=1
  if sa=='TRAIN' and sb!='TRAIN':
   ha,hb=hb,ha;sa,sb=sb,sa;qs,qe,ts,te,ql,tl=ts,te,qs,qe,tl,ql;qc,tc=tc,qc
  if sb=='TRAIN' and sa!='TRAIN':
   for idx,cut in enumerate((90,70,50,30)):
    if 100*ni>=cut*al:
     if qc>=.8 and tc>=.8:heldout_flags[ha]|=1<<(idx*3)
     if (qc if ql<=tl else tc)>=.8:heldout_flags[ha]|=1<<(idx*3+1)
     if ev<=.001:heldout_flags[ha]|=1<<(idx*3+2)
   d={'sequence_id':ha,'split':sa,'nearest_detected_train_sequence_id':hb,'identity':ident,'query_coverage':qc,'target_coverage':tc,'alignment_columns':al,'query_start':qs,'query_end':qe,'target_start':ts,'target_end':te,'evalue':ev,'bits':bits,'alignment_search_query_sequence_id':p['hashes'][a],'retrieval':'MMseqs masked sensitivity7.5;identity>=30;either75;E<=1e9'}
   old=nearest[ha]
   if old is None or (ident,min(qc,tc),bits)>(old['identity'],min(old['query_coverage'],old['target_coverage']),old['bits']):nearest[ha]=d
  if len(sample)<30 and sa!='TRAIN' and sb=='TRAIN' and ql<=256 and tl<=256:sample.append((ha,hb))
 if ambiguities:
  tab=pa.Table.from_pylist(ambiguities)
  if ambiguity_writer is None:ambiguity_writer=pq.ParquetWriter(root/'SHORT_MATCH_AMBIGUITIES.parquet',tab.schema,compression='zstd')
  ambiguity_writer.write_table(tab)
 if ambiguity_writer is not None:ambiguity_writer.close()
 else:pq.write_table(pa.table({'query_sequence_id':pa.array([],type=pa.string()),'target_sequence_id':pa.array([],type=pa.string())}),root/'SHORT_MATCH_AMBIGUITIES.parquet')
 incidence={}
 for cutoff_idx,cut in enumerate((90,70,50,30)):
  for bit,scope in enumerate(('both80','shorter80','Evalue_le_1e-3_with_either75')):
   incidence[f'identity>={cut}:{scope}']={s:sum(split[h]==s and bool(v&(1<<(cutoff_idx*3+bit))) for h,v in heldout_flags.items()) for s in ('VALIDATION','TEST')}
 output=[]
 for h,d in sorted(nearest.items()):output.append(d or {'sequence_id':h,'split':split[h],'nearest_detected_train_sequence_id':None,'identity':None,'query_coverage':None,'target_coverage':None,'alignment_columns':None,'query_start':None,'query_end':None,'target_start':None,'target_end':None,'evalue':None,'bits':None,'alignment_search_query_sequence_id':None,'retrieval':'NO_MATCH_DETECTED_NOT_PROOF_OF_NO_SIMILARITY'})
 pq.write_table(pa.Table.from_pylist(output),root/'HELDOUT_NEAREST_DETECTED_TRAIN.parquet',compression='zstd')
 # Ownership / duplicate / group checks are independent table traversals.
 seen=set();ownerbad=0;obsseen=set();observation_counts=collections.Counter()
 for r in rows:
  assert r['canonical_full_sequence'] not in seen;seen.add(r['canonical_full_sequence'])
 group_splits=collections.defaultdict(set)
 for h,g in groups.items():group_splits[g].add(split[h])
 assert all(len(v)==1 for v in group_splits.values())
 for batch in pq.ParquetFile(root/'JOINT_OBSERVATION_ASSIGNMENTS.parquet').iter_batches(batch_size=1024,use_threads=False):
  for o in batch.to_pylist():
   assert o['observation_id'] not in obsseen;obsseen.add(o['observation_id']);observation_counts[o['sequence_hash']]+=1;ownerbad+=int(o['split']!=split[o['sequence_hash']])
 for row in rows:assert observation_counts[row['sequence_id']]==row['joint_observation_count'],'observation multiplicity mismatch'
 save(root/'CROSS_SPLIT_AUDIT.json',{'status':'PASS' if not violations and not ownerbad else 'FAIL','all_queries':int(active.sum()),'aligned_cross_split_rows_two_coverage_passes':nrows,'protected_violation_examples':violations,'observation_ownership_violations':ownerbad,'exact_string_overlap':0,'group_overlap':0,'diagnostic_counts_directed_rows_not_unique_pairs':dict(bins),'heldout_identities':len(nearest),'with_detected_train_match':sum(v is not None for v in nearest.values()),'heldout_identities_with_detected_train_similarity':incidence,'short_ambiguity_rows_two_passes_and_directions':ambiguity_rows,'nearest_definition':'maximum detected identity, then minimum bidirectional coverage, then bits; not guaranteed global nearest','independent_spotcheck_pairs':sample,'limitations':['shared MMseqs masked kmer heuristic; no proof of exhaustive all-pair recall','best local alignment per directed candidate; alternative alignments and remote homology not certified','diagnostics cover either75% alignments only, not all short local similarities','same source/context overlap is reported, not prohibited','exact scientific statistical-power approval remains pending']})
 assert not violations and not ownerbad,'violations require grouping repair before release'
 print('CROSS SCREEN PASS',nrows,'rows',flush=True)
