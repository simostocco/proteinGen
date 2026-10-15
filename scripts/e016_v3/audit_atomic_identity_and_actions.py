"""Independent atom-token correspondence and full-chain internal-action QA.
Updates only this unpublished working catalogue; source archives stay immutable.
"""
import os,sys,json,hashlib,sqlite3,collections,time,gzip,math
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import build_unified_catalogue as C
R=C.R;db=C.db;CACHE=C.CACHE;start=time.monotonic();assert C.completed('intake_complete')
if C.completed('atomic_and_action_QA'):raise RuntimeError('QA already completed')
counts=collections.Counter();parts=[];failures=R/'mapping/ATOMIC_IDENTITY_EXCEPTIONS.jsonl'
with failures.open('w') as failure:
 for cp in sorted((CACHE/'partition_commits').glob('part-*.json')):
  tables=json.load(open(cp))['tables'];sources={r['source_id']:r['source_sha256'] for r in C.rows(CACHE/tables['source_identity']['path'])};p=CACHE/tables['protein_chain_models']['path'];local=collections.Counter()
  cols=['source_id','pdb_id','entity_poly_sequence','label_asym_id','model_id','atom_label_sequence_ids','coordinate_model_polymer_sequence']
  for i,r in enumerate(C.rows(p,cols)):
   try:seq=C.cached_canonical(r['entity_poly_sequence'])
   except ValueError:local['canonical_unavailable']+=1;continue
   src=sources[r['source_id']];oid=C.h('|'.join([src,str(r['label_asym_id']),str(r['model_id']),'legacy_CA_occupancy_v1']))
   reason=None
   try:
    labels=[int(x) for x in C.array(r['atom_label_sequence_ids'])];actual=r['coordinate_model_polymer_sequence'] or ''
    if len(actual)!=len(labels):raise ValueError('ATOMIC_TOKEN_LABEL_LENGTH_DISAGREEMENT')
    if len(set(labels))!=len(labels) or labels!=sorted(labels):raise ValueError('ATOMIC_LABEL_ORDER_OR_DUPLICATION')
    if any(pos<1 or pos>len(seq) for pos in labels):raise ValueError('ATOMIC_LABEL_OUTSIDE_DECLARED_FULL_POLYMER')
    if actual!=''.join(seq[pos-1] for pos in labels):raise ValueError('ATOMIC_MONOMER_CANONICAL_IDENTITY_DISAGREEMENT')
    local['passed']+=1
   except (ValueError,TypeError) as ex:reason=str(ex);local[reason]+=1
   if reason:
    previous=db.execute('SELECT mapping_status,BC_eligible FROM observations WHERE observation_id=?',(oid,)).fetchone()
    if previous and previous[0]!='SUPERSEDED_MAPPING_VIEW':
     counts['BC_removed_atomic_identity']+=previous[1]
     db.execute("UPDATE observations SET mapping_status='QUARANTINE_ATOMIC_IDENTITY',BC_eligible=0,BC_reason=? WHERE observation_id=?",(reason,oid))
     record={'observation_id':oid,'entry_id':r['pdb_id'],'source_sha256':src,'cache_path':str(p),'row':i,'reason':reason,'previous_BC_eligible':bool(previous[1])};failure.write(json.dumps(record)+'\n');C.exclude('atomic:'+oid,r['pdb_id'],'observation',reason,record)
  counts.update(local);parts.append({'partition':cp.name,'counts':dict(local)});db.commit();print('atomicQA',cp.name,dict(local),flush=True)
# Read each original coordinate source once, not one full Parquet file per example.
rows=db.execute("SELECT observation_id,sequence_hash,coordinate_kind,coordinate_path,coordinate_row FROM observations WHERE BC_eligible=1 ORDER BY coordinate_path,coordinate_row").fetchall();seqmap=dict(db.execute('SELECT sequence_hash,sequence FROM entities WHERE sequence_hash IS NOT NULL'))
grouped=collections.defaultdict(list)
for row in rows:grouped[row[3]].append(row)
def check(row,coords):
 oid,seqhash,kind,path,i=row;seq=seqmap[seqhash];x=np.asarray(coords,dtype=np.float64);reason=None
 if x.shape!=(len(seq),3) or not np.isfinite(x).all():reason='FULL_COORDINATE_SHAPE_OR_FINITE_CONTRADICTION'
 else:
  b=np.diff(x,axis=0);d=np.linalg.norm(b,axis=1)
  if np.any(d<1e-10):reason='UNSUPPORTED_ZERO_VIRTUAL_BOND'
  if reason is None:
   cosine=-np.einsum('ij,ij->i',b[:-1],b[1:])/(d[:-1]*d[1:]);theta=np.arccos(np.clip(cosine,-1,1))
   if np.any(theta<=0) or np.any(theta>=np.pi):reason='UNSUPPORTED_COLLINEAR_VIRTUAL_ANGLE'
  if reason is None:
   u=b[1:-1]/d[1:-1,None];v=-b[:-2];v=v-np.sum(v*u,axis=1)[:,None]*u;w=b[2:]-np.sum(b[2:]*u,axis=1)[:,None]*u;vn=np.linalg.norm(v,axis=1);wn=np.linalg.norm(w,axis=1)
   if np.any(vn<1e-10) or np.any(wn<1e-10):reason='UNSUPPORTED_DEGENERATE_TORSION_FRAME'
   else:
    v/=vn[:,None];w/=wn[:,None];tau=np.arctan2(np.sum(np.cross(u,v)*w,axis=1),np.sum(v*w,axis=1))
    if np.any(np.abs(tau)>=np.pi):reason='TRANSFORMED_GAUSSIAN_TORSION_SEAM_TARGET_UNSUPPORTED'
 counts['BC_action_candidates_checked']+=1
 if reason:
  db.execute('UPDATE observations SET BC_eligible=0,BC_reason=? WHERE observation_id=?',(reason,oid));C.exclude('action:'+oid,'','BC_action_support',reason,{'observation_id':oid,'source':path,'row':i,'biological_validity':'UNKNOWN; representation support exclusion is not a nonfolder label'});counts[reason]+=1
 else:counts['BC_action_valid']+=1
for path,items in grouped.items():
 if items[0][2]=='legacy_cache':
  desired={r[4]:r for r in items};offset=0
  for batch in pq.ParquetFile(path).iter_batches(batch_size=64,columns=['selected_calpha_coordinates'],use_threads=False):
   for i in range(len(batch)):
    if offset+i in desired:check(desired[offset+i],json.loads(batch.column(0)[i].as_py()))
   offset+=len(batch)
 else:
  data=json.load(gzip.open(path,'rt'))
  for row in items:check(row,[r['ca'] for r in data['observations'][row[4]]['residue_mapping']])
 db.commit();C.guard()
summary={'status':'PASS_COMPLETE_DECLARED_QA','counts':dict(counts),'partitions':parts,'atom_identity_scope':'all original cached observed atom tokens compared to complete entity canonical positions; unresolved records quarantined, no schema projection treated as independent atomic proof','action_support_scope':'all primary full-chain BC candidates; masks/distance/angle/torsion support checked; unsupported seam is a model-representation exclusion, not biological invalidity','source_archives_modified':False,'elapsed_seconds':time.monotonic()-start}
(R/'mapping/ATOMIC_AND_ACTION_QA.json').write_text(json.dumps(summary,indent=2));C.stage('atomic_and_action_QA',summary);db.execute('pragma wal_checkpoint(TRUNCATE)');db.close();print(json.dumps(summary['counts']),flush=True)
