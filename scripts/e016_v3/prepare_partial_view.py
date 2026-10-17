"""Recover only existing, hash-verified polymer/atom evidence; never admit TEST observations; shared batches may decode unselected rows."""
import collections,functools,gzip,hashlib,json,math,resource,sqlite3,time,shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
from protein_joint_sac.data.certified_unified import file_sha
from protein_joint_sac.data.partial_geometry import derive
R=Path('/mnt/d/Simone/proteinGen_data/e016_partial_supervision_20261016_v1')
U=Path('/mnt/d/Simone/proteinGen_data/e016_gpu_unified_pdb_20261015_v1')
P=Path('/mnt/d/Simone/proteinGen_data/pdb_supervised_split_50id80cov_20261016_v1')
SIDE=Path('/mnt/d/Users/Simone Stocco/proteinGen_audits/e006_rich_geometry_sidecars_v2')

def write(p,x):
 p=Path(p);q=p.with_name(p.name+'.part');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def guard():
 if shutil.disk_usage(R).free<10*2**30+500*2**20:raise RuntimeError('disk reserve plus inventory peak')
 m={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
 if m['MemAvailable']<.1*m['MemTotal']:raise RuntimeError('insufficient actual OS RAM headroom')
def arr(x):return json.loads(x) if isinstance(x,str) else x
@functools.lru_cache(maxsize=4)
def group(path,k):
 cols=['selected_calpha_coordinates','missing_calpha_mask','label_sequence_ids','residue_tokens','auth_sequence_ids','insertion_codes','ambiguous_altloc_position_count','deterministic_altloc_tie_count','selected_altlocs','chain_mapping_status','model_id','entity_id']
 return pq.ParquetFile(path).read_row_group(k,columns=cols,use_threads=False).to_pylist()
def raw(row):
 f=pq.ParquetFile(row['coordinate_path']);off=0
 for k in range(f.num_row_groups):
  n=f.metadata.row_group(k).num_rows
  if row['coordinate_row']<off+n:return group(row['coordinate_path'],k)[row['coordinate_row']-off]
  off+=n
 raise ValueError('coordinate row missing')
def legacy(row,parent):
 v=raw(row);s=parent['canonical_full_sequence'];N=len(s)
 if v['chain_mapping_status']!='unambiguous' or str(v['model_id'])!=row['model']:raise ValueError('AUTHOR_MODEL_MAPPING_UNKNOWN')
 if v['ambiguous_altloc_position_count'] or v['deterministic_altloc_tie_count']:raise ValueError('ALTERNATE_CONFORMER_AMBIGUITY')
 if [int(k) for k in arr(v['label_sequence_ids'])]!=list(range(1,N+1)) or ''.join(arr(v['residue_tokens']))!=s:raise ValueError('AUTHORITATIVE_POLYMER_CONTRADICTION')
 alt=arr(v['selected_altlocs']);nonblank={a for a in alt if a not in [None,'','.', '?']}
 if len(nonblank)>1:raise ValueError('MULTIPLE_NONBLANK_CONFORMER_LABELS_UNRESOLVED')
 seen=~np.asarray(arr(v['missing_calpha_mask']),bool);xy=arr(v['selected_calpha_coordinates'])
 if len(xy)!=N or seen.shape!=(N,) or [a is not None for a in xy]!=seen.tolist():raise ValueError('MASK_COORDINATE_CONTRADICTION')
 x=np.array([a if a is not None else [0,0,0] for a in xy],dtype=np.float64)
 if not np.isfinite(x[seen]).all():raise ValueError('NONFINITE_OBSERVED_CA')
 keys=list(zip(map(str,arr(v['auth_sequence_ids'])),arr(v['insertion_codes'])))
 observed_keys=[keys[i] for i in np.flatnonzero(seen)]
 if len(set(observed_keys))!=len(observed_keys):raise ValueError('AMBIGUOUS_AUTHOR_NUMBERING')
 author={keys[i]:int(i) for i in np.flatnonzero(seen)}
 for (number,ins),i in list(author.items()):
  if ins and (number+ins,ins) not in author:author[(number+ins,ins)]=i
 return x,seen,author,alt

def main():
 start=time.monotonic();proposal=PDBSplitProposal(P);seqs=proposal.sequences;im=json.load(open(U/'release/intake_v1/INTAKE_MANIFEST.json'));ip=U/'release/intake_v1/chain_observations.parquet'
 if file_sha(ip)!=im['hashes'][ip.name]:raise ValueError('intake checksum mismatch')
 candidates={};bykey=collections.defaultdict(list);reasons=collections.Counter();counts=collections.Counter();sources={};primary=[];ledger=[];owned={s:set() for s in ['TRAIN','VALIDATION']}
 for b in pq.ParquetFile(ip).iter_batches(batch_size=256,use_threads=False):
  guard()
  for row in b.to_pylist():
   parent=seqs.get(row['sequence_hash'])
   if parent is None:continue
   if parent['split']=='TEST':counts['TEST_METADATA_ONLY_NOT_ADMITTED']+=1;continue
   if row['complete_CA']:continue
   split=parent['split'];counts[split+'_partial_candidate_rows']+=1
   if row['mapping_status']!='VERIFIED_LABEL_CORRESPONDENCE':reasons[split+':'+row['mapping_status']]+=1;continue
   m=json.loads(row.pop('metadata'));method=m.get('experimental_method',None)
   if row['coordinate_kind'] in ['priority_v3','regression_v3']:
    if not m.get('geometry_method_resolution_eligible',False):reasons[split+':METHOD_RESOLUTION_INELIGIBLE']+=1;continue
    if m.get('alternate_selection_ambiguous',True):reasons[split+':ALTERNATE_CONFORMER_AMBIGUITY']+=1;continue
    if m.get('full_sha256')!=row['sequence_hash']:raise ValueError('parent source SHA contradiction')
    row['continuity']=m['continuity_mask'];primary.append(row)
   else:
    res=m.get('resolution_angstrom')
    if m.get('chain_mapping_status')!='unambiguous':reasons[split+':AUTHOR_MAPPING_UNKNOWN']+=1;continue
    if not(method=='SOLUTION NMR' or res is not None and res<=(3 if method=='X-RAY DIFFRACTION' else 4)):
     reasons[split+':METHOD_RESOLUTION_INELIGIBLE']+=1;continue
    row['continuity']=[False]*(parent['length']-1);bykey[(row['source_sha'],row['model'],row['auth_chain'])].append(row['observation_id'])
   row['split']=split;row['method']=method or ';'.join(m.get('methods',[]));row['continuity_refs']=[];candidates[row['observation_id']]=row
 print(json.dumps({'candidate_observations':len(candidates),'priority':len(primary),'legacy_keys':len(bykey)}),flush=True)
 # Source hashes are checked against the sealed parent release, not filenames.
 external=proposal.external
 for path in sorted({r['coordinate_path'] for r in candidates.values()}):
  if external.get(path)!=file_sha(path):raise ValueError('untracked or changed coordinate source: '+path)
  sources[path]=external[path]
 invalid=set()
 @functools.lru_cache(maxsize=128)
 def load_candidate(oid):
  if oid in invalid:return None
  row=candidates[oid]
  try:return legacy(row,seqs[row['sequence_hash']])
  except ValueError as ex:
   invalid.add(oid);reasons[row['split']+':'+str(ex)]+=1;return None
 db=sqlite3.connect('file:'+str(U/'tables/unified_v2.sqlite')+'?mode=ro',uri=True)
 samples={k:(entry,chain) for k,entry,chain in db.execute('SELECT sample_id,entry_id,auth_chain FROM historical_samples')};db.close()
 inspected=matches=0;cols=['sample_id','sequence','residue_ids','insertion_codes','model_number','source_sha256','ca_coordinates','ca_mask','chain_continuity_mask','selected_atom_conformers','conflicting_nonblank_conformer_count']
 # Existing atom sidecars supply per-link evidence, even when the old complete-only index omitted it.
 journal=R/'view/continuity_recovery_journal';journal.mkdir(exist_ok=True)
 for path in sorted([*SIDE.glob('train/*.parquet'),*SIDE.glob('validation/*.parquet')]):
  guard()
  if str(path) not in external or file_sha(path)!=external[str(path)]:raise ValueError('sidecar provenance missing/changed '+str(path))
  sources[str(path)]=external[str(path)];offset=0
  jp=journal/(hashlib.sha256(str(path).encode()).hexdigest()+'.json')
  if jp.exists():
   saved=json.loads(jp.read_text())
   if saved['source_sha256']!=external[str(path)] or saved['version']!='continuity-recovery-v1':raise ValueError('journal source/version contradiction')
   for oid,links,reference in saved['links']:
    for k in links:candidates[oid]['continuity'][k]=True
    candidates[oid]['continuity_refs'].append(reference)
   inspected+=saved['rows'];matches+=len(saved['links'])
   print(json.dumps({'recovered_sidecar_journal':str(path),'rows_inspected':inspected,'matched_evidence':matches}),flush=True);continue
  durable_links=[];before_inspected=inspected
  for b in pq.ParquetFile(path).iter_batches(batch_size=32,columns=cols,use_threads=False):
   for j,v in enumerate(b.to_pylist()):
    inspected+=1;entry,chain=samples[v['sample_id']];key=(v['source_sha256'],str(v['model_number']),chain)
    for oid in bykey.get(key,[]):
     row=candidates[oid];loaded=load_candidate(oid)
     if loaded is None:continue
     x,seen,author,alt=loaded
     if row['entry_id']!=entry or v['conflicting_nonblank_conformer_count']:continue
     pos=[author.get((str(a),i)) for a,i in zip(v['residue_ids'],v['insertion_codes'])]
     if None in pos or len(pos)!=len(v['sequence']) or len(set(pos))!=len(pos):continue
     if ''.join(seqs[row['sequence_hash']]['canonical_full_sequence'][k] for k in pos)!=v['sequence']:continue
     ca=np.asarray(v['ca_coordinates'],dtype=np.float32);ok=np.asarray(v['ca_mask'],bool)
     if not all(seen[k] and np.array_equal(x[k].astype(np.float32),ca[i]) for i,k in enumerate(pos) if ok[i]):continue
     # Require sidecar CA conformer to agree with the authoritative selected CA.
     if any(ok[i] and (v['selected_atom_conformers'][i]['ca'] or '')!=(alt[k] or '') for i,k in enumerate(pos)):continue
     recovered=0
     for i,connected in enumerate(v['chain_continuity_mask']):
      if connected and ok[i] and ok[i+1] and pos[i+1]==pos[i]+1:
       row['continuity'][pos[i]]=True;recovered+=1
     if recovered:
      reference={'path':str(path),'row':offset+j};row['continuity_refs'].append(reference);matches+=1
      durable_links.append([oid,[pos[i] for i,connected in enumerate(v['chain_continuity_mask']) if connected and ok[i] and ok[i+1] and pos[i+1]==pos[i]+1],reference])
   offset+=len(b)
  write(jp,{'version':'continuity-recovery-v1','source_sha256':external[str(path)],'rows':inspected-before_inspected,'links':durable_links})
  group.cache_clear()
  print(json.dumps({'sidecar':path.name,'rows_inspected':inspected,'matched_evidence':matches,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}),flush=True)
 # A compact reference view stores links and masks; no coordinate archive is copied.
 admitted=[];tally={s:collections.Counter() for s in ['TRAIN','VALIDATION']};ids={s:set() for s in tally};fullids={s:{h for h,p in seqs.items() if p['split']==s and p['joint_eligible']} for s in tally}
 for oid,row in sorted(candidates.items(),key=lambda p:(p[1]['coordinate_path'],p[1]['coordinate_row'])):
  parent=seqs[row['sequence_hash']];s=parent['canonical_full_sequence'];split=row['split'];N=len(s)
  if row['coordinate_kind']=='legacy_cache':
   if not any(row['continuity']):reasons[split+':NO_VERIFIED_CONTINUITY_EVIDENCE']+=1;continue
   loaded=load_candidate(oid)
   if loaded is None:continue
   x,seen,_,_=loaded
  else:
   d=json.load(gzip.open(row['coordinate_path'],'rt'));o=d['observations'][row['coordinate_row']]
   if d['source_sha256']!=row['source_sha'] or o['canonical_full_sequence']!=s or str(o['model_num'])!=row['model']:raise ValueError('priority source identity contradiction')
   mapping=o['residue_mapping']
   if [v['label_seq_id'] for v in mapping]!=list(range(1,N+1)):raise ValueError('priority label order contradiction')
   seen=np.array([not v['ca_missing'] for v in mapping]);x=np.array([v['ca'] if not v['ca_missing'] else [0,0,0] for v in mapping],dtype=np.float64)
  try:g,mask,segments=derive(x,seen,row['continuity'])
  except ValueError as ex:
   ledger.append({'observation_id':oid,'reason':str(ex),'source_sha':row['source_sha'],'split':split});reasons[split+':'+str(ex)]+=1;continue
  if not mask.any():reasons[split+':NO_VERIFIED_GEOMETRIC_TARGETS']+=1;continue
  row.update(ca_observed=seen.tolist(),polymer_length=N,aa_known=[True]*N,coordinate_segment_id=segments.tolist(),distance_targets=int(mask[:,0].sum()),angle_targets=int(mask[:,1].sum()),torsion_targets=int(mask[:,2].sum()),AA_only_targets=int((~mask.any(1)).sum()),AA_targets_at_missing_CA=int((~seen).sum()))
  admitted.append(row);ids[split].add(row['sequence_hash']);z=tally[split];z['observations']+=1;z['AA_targets']+=N;z['observed_CA']+=int(seen.sum());z['segments']+=len(set(segments[segments>=0]));z['AA_at_missing_CA']+=int((~seen).sum());z['AA_only_targets']+=row['AA_only_targets']
  for key in ['distance_targets','angle_targets','torsion_targets']:z[key]+=row[key]
 pq.write_table(pa.Table.from_pylist(admitted),R/'view/PARTIAL_OBSERVATIONS.parquet',compression='zstd')
 write(R/'view/SOURCE_HASHES.json',sources);write(R/'view/QUARANTINE.json',ledger)
 result={'status':'MAPPING_AND_EXISTING_CONTINUITY_EVIDENCE_ONLY','population':'TRAIN and VALIDATION admitted only; shared source batches may decode unselected TEST rows, never used for features, fitting or evaluation','candidate_counts':dict(counts),'exclusion_reasons':dict(reasons),'splits':{s:{**dict(tally[s]),'identities':len(ids[s]),'additional_to_complete_identities':len(ids[s]-fullids[s]),'already_complete_identities':len(ids[s]&fullids[s])} for s in tally},'sidecar_rows_inspected':inspected,'matched_sidecar_evidence':matches,'seconds':time.monotonic()-start,'RSS_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'intake_table_sha256':file_sha(ip),'split_manifest_sha256':file_sha(P/'PROPOSAL_MANIFEST.json')}
 write(R/'PARTIAL_PDB_ELIGIBILITY_COUNTS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
